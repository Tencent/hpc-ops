// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include <cub/cub.cuh>

#include "cute/tensor.hpp"
#include "cutlass/arch/reg_reconfig.h"
#include "src/fuse_moe/fuse_moe.h"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace fuse_moe {

namespace kernels {

__global__ void count_kernel(const int *topk_ids_ptr, int *seqlens_ptr, int total_num_seq,
                             int start_expert, int end_expert) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  int iexpert = topk_ids_ptr[idx];
  if ((idx < total_num_seq) && (iexpert >= start_expert) && (iexpert < end_expert)) {
    atomicAdd(&seqlens_ptr[iexpert - start_expert], 1);
  }
}

template <int kThreadPerBlock, int kGroupPerThread, int kTileM>
__global__ void count_kernel(const int *topk_ids_ptr, int *topk_pos_ptr, int *seqlens_ptr,
                             int *cu_seqlens_ptr, int *tiles_ptr, int *cu_tiles_ptr, int num_seq,
                             int num_topk, int total_num_seq, int num_expert, int start_expert,
                             int end_expert) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  extern __shared__ int seqlens_shm[];

  for (int i = idx; i < num_expert; i += blockDim.x) {
    seqlens_shm[i] = 0;
  }
  __syncthreads();

  for (int i = idx; i < total_num_seq; i += blockDim.x) {
    int iexpert = topk_ids_ptr[i];
    if ((iexpert >= start_expert) && (iexpert < end_expert)) {
      atomicAdd(&seqlens_shm[iexpert - start_expert], 1);
    }
    topk_pos_ptr[i] = -1;
  }

  __syncthreads();

  // cusum
  int thread_seqs[kGroupPerThread];
  int thread_tiles[kGroupPerThread];
#pragma unroll
  for (int i = 0; i < kGroupPerThread; i++) {
    int igroup = idx * kGroupPerThread + i;
    if (igroup < num_expert) {
      int iseq = seqlens_shm[igroup];
      int itile_num = (iseq + kTileM - 1) / kTileM;
      thread_seqs[i] = iseq;
      thread_tiles[i] = itile_num;
      tiles_ptr[igroup] = itile_num;
    } else {
      thread_seqs[i] = 0;
      thread_tiles[i] = 0;
    }
  }
  using BlockScan = cub::BlockScan<int, kThreadPerBlock>;
  __shared__ typename BlockScan::TempStorage temp_storage1;
  __shared__ typename BlockScan::TempStorage temp_storage2;
  int seqs_aggregate, tiles_aggregate;
  BlockScan(temp_storage1).ExclusiveSum(thread_seqs, thread_seqs, seqs_aggregate);
  BlockScan(temp_storage2).ExclusiveSum(thread_tiles, thread_tiles, tiles_aggregate);

  // store
  // fill seqlens with zero
  for (int i = idx; i < num_expert; i += blockDim.x) {
    seqlens_ptr[i] = 0;  // seqlens_shm[i];
  }

#pragma unroll
  for (int i = 0; i < kGroupPerThread; i++) {
    int igroup = idx * kGroupPerThread + i;
    if (igroup < num_expert) {
      cu_seqlens_ptr[igroup] = thread_seqs[i];
      cu_tiles_ptr[igroup] = thread_tiles[i];
    }
  }
  if (idx == 0) {
    cu_seqlens_ptr[num_expert] = seqs_aggregate;
    cu_tiles_ptr[num_expert] = tiles_aggregate;
  }
}

template <typename T1, typename T2, typename TmaX, typename TmaY, int kWarpPerBlock>
__global__ void gather_kernel(const vec_t<cute::TmaDescriptor, 2> td_xy,
                              cute::TmaDescriptor *tma_xy, T1 *y_ptr, T2 *yg_ptr, const T1 *x_ptr,
                              const int *topk_ids_ptr, int *topk_pos_ptr, int *seqlens_ptr,
                              int *cu_seqlens_ptr, int total_num_seq, int hidden_size,
                              int intermediate_size, int start_expert, int end_expert,
                              int num_block_for_copy, cutlass::FastDivmod topk_divider) {
  constexpr int kThreadPerWarp = 32;
  int idx = threadIdx.x;
  int iblock = blockIdx.x;
  int iwarp = idx / kThreadPerWarp;
  int ilane = idx % kThreadPerWarp;
  int ielem = iblock * kWarpPerBlock + iwarp;

  int iexpert = topk_ids_ptr[ielem];

  if (iblock < num_block_for_copy) {
    if ((ielem < total_num_seq) && (iexpert >= start_expert) && (iexpert < end_expert)) {
      int pos_in_expert;
      if (ilane == 0) {
        pos_in_expert = atomicAdd(&seqlens_ptr[iexpert - start_expert], 1);
      }

      pos_in_expert = __shfl_sync(0xFFFFFFFF, pos_in_expert, 0);

      int irow = cu_seqlens_ptr[iexpert] + pos_in_expert;

      int iseq, itopk;
      topk_divider(iseq, itopk, ielem);

      auto y_irow_ptr = y_ptr + irow * hidden_size;
      auto x_irow_ptr = x_ptr + iseq * hidden_size;

      constexpr int kNumItemPer16B = 16 / sizeof(T1);
      int total_items = hidden_size / kNumItemPer16B;

      for (int i = ilane; i < total_items; i += kThreadPerWarp) {
        store<T1, kNumItemPer16B>(y_irow_ptr + i * kNumItemPer16B,
                                  load<T1, kNumItemPer16B>(x_irow_ptr + i * kNumItemPer16B));
      }
      topk_pos_ptr[ielem] = irow;
    }
  } else {
    if (idx < 32) {
      using namespace cute;  // NOLINT

      int igroup = iblock - num_block_for_copy;

      __shared__ cute::TmaDescriptor smem_tma_desc[2];

      int num_seq = cu_seqlens_ptr[igroup + 1] - cu_seqlens_ptr[igroup];
      int cu_seqlen = cu_seqlens_ptr[igroup];
      auto *x_ibatch_ptr = y_ptr + cu_seqlen * hidden_size;
      auto *y_ibatch_ptr = yg_ptr + cu_seqlen * intermediate_size;

      int k = hidden_size;
      int n = intermediate_size;

      if (idx < 2) {
        smem_tma_desc[idx] = td_xy[idx];
      }
      __syncwarp();

      // X
      if (idx == 0) {
        auto gX = make_tensor(make_gmem_ptr(x_ibatch_ptr), make_shape(num_seq, k),
                              make_stride(k, Int<1>{}));
        update_tma_gtensor<TmaX>(smem_tma_desc[idx], gX);
      }

      // K
      if (idx == 1) {
        auto gY = make_tensor(make_gmem_ptr(y_ibatch_ptr), make_shape(n, num_seq),
                              make_stride(Int<1>{}, n));
        update_tma_gtensor<TmaY>(smem_tma_desc[idx], gY);
      }

#pragma unroll
      for (int i = 0; i < 2; i++) {
        __syncwarp();
        if (cute::elect_one_sync()) {
          cute::tma_desc_commit_group();
          cute::tma_desc_wait_group();
        }
        tma_descriptor_cp_fence_release(tma_xy + igroup * 2 + i, smem_tma_desc[i]);
      }
    }
  }
}

}  // namespace kernels

void count_and_gather_async(void *y_ptr, void *yg_ptr, const void *x_ptr, const void *topk_ids_ptr,
                            void *topk_pos_ptr, void *seqlens_ptr, void *cu_seqlens_ptr,
                            void *tmas_ptr, void *tiles_ptr, void *cu_tiles_ptr, int num_seq,
                            int hidden_size, int intermediate_size, int num_topk, int num_expert,
                            int eprank, cudaStream_t stream) {
  using namespace cute;  // NOLINT

  using Tin = cute::float_e4m3_t;
  using Tout = cute::bfloat16_t;

  constexpr int kTileM = 16;
  constexpr int kTileN = 128;
  constexpr int kTileK = 128;
  constexpr int kStage = 8;

  int m = num_seq;
  int n = intermediate_size;
  int k = hidden_size;
  auto X = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(y_ptr)), make_shape(m, k),
                       make_stride(k, Int<1>{}));
  auto Y = make_tensor(make_gmem_ptr(reinterpret_cast<Tout *>(yg_ptr)), make_shape(n, m),
                       make_stride(Int<1>{}, n));

  auto slayout_x = tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{},
                                 make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{}));

  auto cpbox_yt = tile_to_shape(GMMA::Layout_MN_SW64_Atom<Tout>{},
                                make_shape(Int<kTileN / 2>{}, Int<kTileM>{}));

  auto tma_x = make_tma_copy(SM90_TMA_LOAD{}, X, slayout_x(_, _, 0));
  auto tma_y = make_tma_copy(SM90_TMA_STORE{}, Y, cpbox_yt);

  auto *tma_xy = static_cast<cute::TmaDescriptor *>(tmas_ptr);

  int total_num_seq = num_seq * num_topk;
  int start_expert = eprank * num_expert;
  int end_expert = (eprank + 1) * num_expert;

  // 0. count tokens
  {
    constexpr int kThreadPerBlock = 256;
    constexpr int kGroupPerThread = 2;

    dim3 block(kThreadPerBlock);
    dim3 grid(1);
    kernels::count_kernel<kThreadPerBlock, kGroupPerThread, kTileM>
        <<<grid, block, num_expert * 4, stream>>>(
            (const int *)topk_ids_ptr, (int *)topk_pos_ptr, (int *)seqlens_ptr,
            (int *)cu_seqlens_ptr, (int *)tiles_ptr, (int *)cu_tiles_ptr, num_seq, num_topk,
            total_num_seq, num_expert, start_expert, end_expert);
  }

  // 1. gather token and update tmas
  {
    vec_t<cute::TmaDescriptor, 2> td_xy{
        *tma_x.get_tma_descriptor(),
        *tma_y.get_tma_descriptor(),
    };

    constexpr int kWarpPerBlock = 4;
    int num_block_for_copy = (total_num_seq + kWarpPerBlock - 1) / kWarpPerBlock;

    cutlass::FastDivmod topk_divider(num_topk);

    dim3 block(kWarpPerBlock * 32);
    dim3 grid(num_block_for_copy + num_expert);

    kernels::gather_kernel<Tin, Tout, decltype(tma_x), decltype(tma_y), kWarpPerBlock>
        <<<grid, block, 0, stream>>>(td_xy, tma_xy, (Tin *)y_ptr, (Tout *)yg_ptr,
                                     (const Tin *)x_ptr, (const int *)topk_ids_ptr,
                                     (int *)topk_pos_ptr, (int *)seqlens_ptr, (int *)cu_seqlens_ptr,
                                     total_num_seq, hidden_size, intermediate_size, start_expert,
                                     end_expert, num_block_for_copy, topk_divider);
  }
}
}  // namespace fuse_moe
}  // namespace hpc
