// Copyright (C) 2026 Tencent.

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

__global__ void blockwise_count_tokens_kernel(const int *topk_ids_ptr, int *topk_pos_ptr,
                                              int *num_tokens_per_group_ptr, int total_num_topk,
                                              int start_expert, int end_expert) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if ((idx < total_num_topk)) {
    int iexpert = topk_ids_ptr[idx];
    if ((iexpert >= start_expert) && (iexpert < end_expert)) {
      atomicAdd(&num_tokens_per_group_ptr[iexpert - start_expert], 1);
    }
    topk_pos_ptr[idx] = -1;
  }
}

template <int kThreadPerBlock, int kGroupPerThread, int kTileM>
__global__ void blockwise_count_cutokens_kernel(const int *topk_ids_ptr, int *topk_pos_ptr,
                                                int *num_tokens_per_group_ptr,
                                                int *cu_num_tokens_per_group_ptr, int *tiles_ptr,
                                                int *cu_tiles_ptr, int num_seq, int num_topk,
                                                int total_num_topk, int num_experts,
                                                int start_expert, int end_expert) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  // cusum
  int thread_num_tokens[kGroupPerThread];
  int thread_tiles[kGroupPerThread];
#pragma unroll
  for (int i = 0; i < kGroupPerThread; i++) {
    int igroup = idx * kGroupPerThread + i;
    if (igroup < num_experts) {
      int inum_tokens = num_tokens_per_group_ptr[igroup];
      int itile_num = (inum_tokens + kTileM - 1) / kTileM;
      thread_num_tokens[i] = inum_tokens;
      thread_tiles[i] = itile_num;
      tiles_ptr[igroup] = itile_num;
    } else {
      thread_num_tokens[i] = 0;
      thread_tiles[i] = 0;
    }
  }
  using BlockScan = cub::BlockScan<int, kThreadPerBlock>;
  __shared__ typename BlockScan::TempStorage temp_storage1;
  __shared__ typename BlockScan::TempStorage temp_storage2;
  int num_tokens_aggregate, tiles_aggregate;
  BlockScan(temp_storage1).ExclusiveSum(thread_num_tokens, thread_num_tokens, num_tokens_aggregate);
  BlockScan(temp_storage2).ExclusiveSum(thread_tiles, thread_tiles, tiles_aggregate);

  // store
  // fill seqlens with zero
  for (int i = idx; i < num_experts; i += blockDim.x) {
    num_tokens_per_group_ptr[i] = 0;
  }

#pragma unroll
  for (int i = 0; i < kGroupPerThread; i++) {
    int igroup = idx * kGroupPerThread + i;
    if (igroup < num_experts) {
      cu_num_tokens_per_group_ptr[igroup] = thread_num_tokens[i];
      cu_tiles_ptr[igroup] = thread_tiles[i];
    }
  }
  if (idx == 0) {
    cu_num_tokens_per_group_ptr[num_experts] = num_tokens_aggregate;
    cu_tiles_ptr[num_experts] = tiles_aggregate;
  }
}

template <int kThreadPerBlock, int kGroupPerThread, int kTileM>
__global__ void blockwise_count_kernel(const int *topk_ids_ptr, int *topk_pos_ptr,
                                       int *num_tokens_per_group_ptr,
                                       int *cu_num_tokens_per_group_ptr, int *tiles_ptr,
                                       int *cu_tiles_ptr, int num_tokens, int num_topk,
                                       int total_num_topk, int num_experts, int start_expert,
                                       int end_expert) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  extern __shared__ int s_num_tokens_per_group[];

  for (int i = idx; i < num_experts; i += blockDim.x) {
    s_num_tokens_per_group[i] = 0;
  }
  __syncthreads();

  for (int i = idx; i < total_num_topk; i += blockDim.x) {
    int iexpert = topk_ids_ptr[i];
    if ((iexpert >= start_expert) && (iexpert < end_expert)) {
      atomicAdd(&s_num_tokens_per_group[iexpert - start_expert], 1);
    }
    topk_pos_ptr[i] = -1;
  }

  __syncthreads();

  // cusum
  int thread_num_tokens[kGroupPerThread];
  int thread_tiles[kGroupPerThread];
#pragma unroll
  for (int i = 0; i < kGroupPerThread; i++) {
    int igroup = idx * kGroupPerThread + i;
    if (igroup < num_experts) {
      int inum_tokens = s_num_tokens_per_group[igroup];
      int itile_num = (inum_tokens + kTileM - 1) / kTileM;
      thread_num_tokens[i] = inum_tokens;
      thread_tiles[i] = itile_num;
      tiles_ptr[igroup] = itile_num;
    } else {
      thread_num_tokens[i] = 0;
      thread_tiles[i] = 0;
    }
  }
  using BlockScan = cub::BlockScan<int, kThreadPerBlock>;
  __shared__ typename BlockScan::TempStorage temp_storage1;
  __shared__ typename BlockScan::TempStorage temp_storage2;
  int num_tokens_aggregate, tiles_aggregate;
  BlockScan(temp_storage1).ExclusiveSum(thread_num_tokens, thread_num_tokens, num_tokens_aggregate);
  BlockScan(temp_storage2).ExclusiveSum(thread_tiles, thread_tiles, tiles_aggregate);

  // store
  // fill seqlens with zero
  for (int i = idx; i < num_experts; i += blockDim.x) {
    num_tokens_per_group_ptr[i] = 0;
  }

#pragma unroll
  for (int i = 0; i < kGroupPerThread; i++) {
    int igroup = idx * kGroupPerThread + i;
    if (igroup < num_experts) {
      cu_num_tokens_per_group_ptr[igroup] = thread_num_tokens[i];
      cu_tiles_ptr[igroup] = thread_tiles[i];
    }
  }
  if (idx == 0) {
    cu_num_tokens_per_group_ptr[num_experts] = num_tokens_aggregate;
    cu_tiles_ptr[num_experts] = tiles_aggregate;
  }
}

template <typename T1, typename T2, typename GateUpTmaX, typename GateUpTmaY, typename DownTmaX,
          typename DownTmaY, int kWarpPerBlock, int kTileM>
__global__ void blockwise_gather_kernel(
    const vec_t<cute::TmaDescriptor, 4> td_xy, cute::TmaDescriptor *gate_up_tma_xy,
    cute::TmaDescriptor *down_tma_xy, const T1 *input_ptr, const float *input_scale_ptr,
    T1 *gate_up_input_ptr, T2 *gate_up_output_ptr, float *gate_up_input_scale_ptr,
    T1 *down_input_ptr, T2 *down_output_ptr, const int *topk_ids_ptr, int *topk_pos_ptr,
    int *num_tokens_per_group_ptr, int *cu_num_tokens_per_group_ptr, int *cu_tiles_ptr,
    int total_num_topk, int hidden_size, int intermediate_size, int input_scale_size,
    int num_padded_tokens, int start_expert, int end_expert, int num_block_for_copy,
    cutlass::FastDivmod topk_divider) {
  constexpr int kThreadPerWarp = 32;
  int idx = threadIdx.x;
  int iblock = blockIdx.x;
  int iwarp = idx / kThreadPerWarp;
  int ilane = idx % kThreadPerWarp;
  int itopk = iblock * kWarpPerBlock + iwarp;

  if (iblock < num_block_for_copy) {
    if (itopk < total_num_topk) {
      int iexpert = topk_ids_ptr[itopk];
      if ((iexpert >= start_expert) && (iexpert < end_expert)) {
        int pos_in_expert;
        if (ilane == 0) {
          pos_in_expert = atomicAdd(&num_tokens_per_group_ptr[iexpert - start_expert], 1);
        }

        pos_in_expert = __shfl_sync(0xFFFFFFFF, pos_in_expert, 0);

        int irow = cu_num_tokens_per_group_ptr[iexpert - start_expert] + pos_in_expert;

        int itoken, res;
        topk_divider(itoken, res, itopk);

        auto gate_up_input_irow_ptr = gate_up_input_ptr + irow * hidden_size;
        auto input_ptr_irow_ptr = input_ptr + itoken * hidden_size;

        constexpr int kNumItemPer16B = 16 / sizeof(T1);
        int total_items = hidden_size / kNumItemPer16B;

        for (int i = ilane; i < total_items; i += kThreadPerWarp) {
          store<T1, kNumItemPer16B>(
              gate_up_input_irow_ptr + i * kNumItemPer16B,
              load<T1, kNumItemPer16B>(input_ptr_irow_ptr + i * kNumItemPer16B));
        }
        topk_pos_ptr[itopk] = irow;

        irow = cu_tiles_ptr[iexpert - start_expert] * kTileM + pos_in_expert;
        auto gate_up_input_scale_irow_ptr = gate_up_input_scale_ptr + irow;
        auto input_scale_irow_ptr = input_scale_ptr + itoken * input_scale_size;

        for (int i = ilane; i < input_scale_size; i += kThreadPerWarp) {
          auto src_addr = input_scale_irow_ptr + i;
          auto dst_addr = gate_up_input_scale_irow_ptr + i * num_padded_tokens;
          *dst_addr = *src_addr;
        }
      }
    }
  } else {
    if (idx < 32) {
      using namespace cute;  // NOLINT

      int igroup = iblock - num_block_for_copy;

      __shared__ cute::TmaDescriptor smem_tma_desc[4];

      int num_tokens =
          cu_num_tokens_per_group_ptr[igroup + 1] - cu_num_tokens_per_group_ptr[igroup];
      int cu_num_tokens = cu_num_tokens_per_group_ptr[igroup];
      auto *gate_up_input_ibatch_ptr = gate_up_input_ptr + cu_num_tokens * hidden_size;
      auto *gate_up_output_ibatch_ptr = gate_up_output_ptr + cu_num_tokens * intermediate_size;
      auto *down_input_ibatch_ptr = down_input_ptr + cu_num_tokens * intermediate_size / 2;
      auto *down_output_ibatch_ptr = down_output_ptr + cu_num_tokens * hidden_size;

      int k = hidden_size;
      int n = intermediate_size;

      if (idx < 4) {
        smem_tma_desc[idx] = td_xy[idx];
      }
      __syncwarp();

      // gate_up_input
      if (idx == 0) {
        auto gX = make_tensor(make_gmem_ptr(gate_up_input_ibatch_ptr), make_shape(num_tokens, k),
                              make_stride(k, Int<1>{}));
        update_tma_gtensor<GateUpTmaX>(smem_tma_desc[idx], gX);
      }

      // gate_up_output
      if (idx == 1) {
        auto gY = make_tensor(make_gmem_ptr(gate_up_output_ibatch_ptr), make_shape(n, num_tokens),
                              make_stride(Int<1>{}, n));
        update_tma_gtensor<GateUpTmaY>(smem_tma_desc[idx], gY);
      }

      // down_input
      if (idx == 2) {
        auto gX = make_tensor(make_gmem_ptr(down_input_ibatch_ptr), make_shape(num_tokens, n / 2),
                              make_stride(n / 2, Int<1>{}));
        update_tma_gtensor<DownTmaX>(smem_tma_desc[idx], gX);
      }

      // down output
      if (idx == 3) {
        auto gY = make_tensor(make_gmem_ptr(down_output_ibatch_ptr), make_shape(k, num_tokens),
                              make_stride(Int<1>{}, k));
        update_tma_gtensor<DownTmaY>(smem_tma_desc[idx], gY);
      }

#pragma unroll
      for (int i = 0; i < 2; i++) {
        __syncwarp();
        if (cute::elect_one_sync()) {
          cute::tma_desc_commit_group();
          cute::tma_desc_wait_group();
        }
        tma_descriptor_cp_fence_release(gate_up_tma_xy + igroup * 2 + i, smem_tma_desc[i]);
        tma_descriptor_cp_fence_release(down_tma_xy + igroup * 2 + i, smem_tma_desc[i + 2]);
      }
    }
  }
}

}  // namespace kernels

template <int kTileM, int kTileN, int kTileK, int kStage>
void launch_blockwise_count_and_gather(
    const void *input_ptr, const void *input_scale_ptr, void *gate_up_input_ptr,
    void *gate_up_output_ptr, void *gate_up_input_scale_ptr, void *down_input_ptr,
    void *down_output_ptr, const void *topk_ids_ptr, void *topk_pos_ptr,
    void *num_tokens_per_group_ptr, void *cu_num_tokens_per_group_ptr, void *gate_up_tmas_ptr,
    void *down_tmas_ptr, void *tiles_ptr, void *cu_tiles_ptr, int num_tokens, int num_padded_tokens,
    int hidden_size, int intermediate_size, int num_topk, int num_experts, int eprank,
    int num_tokens_per_group_avg, cudaStream_t stream) {
  using namespace cute;  // NOLINT

  using Tin = cute::float_e4m3_t;
  using Tout = cute::bfloat16_t;

  int total_num_topk = num_tokens * num_topk;
  int start_expert = eprank * num_experts;
  int end_expert = (eprank + 1) * num_experts;

  int m = num_tokens;
  int n = intermediate_size;
  int k = hidden_size;

  auto X_gate_up = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(gate_up_input_ptr)),
                               make_shape(m * num_topk, k), make_stride(k, Int<1>{}));
  auto Y_gate_up = make_tensor(make_gmem_ptr(reinterpret_cast<Tout *>(gate_up_output_ptr)),
                               make_shape(n, m * num_topk), make_stride(Int<1>{}, n));
  auto X_down = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(down_input_ptr)),
                            make_shape(m * num_topk, n / 2), make_stride(n / 2, Int<1>{}));
  auto Y_down = make_tensor(make_gmem_ptr(reinterpret_cast<Tout *>(down_output_ptr)),
                            make_shape(k, m * num_topk), make_stride(Int<1>{}, k));

  auto slayout_x = tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{},
                                 make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{}));

  auto cpbox_yt = tile_to_shape(GMMA::Layout_MN_SW64_Atom<Tout>{},
                                make_shape(Int<kTileN / 2>{}, Int<kTileM>{}));

  auto gata_up_tma_x = make_tma_copy(SM90_TMA_LOAD{}, X_gate_up, slayout_x(_, _, 0));
  auto gata_up_tma_y = make_tma_copy(SM90_TMA_STORE{}, Y_gate_up, cpbox_yt);
  auto down_tma_x = make_tma_copy(SM90_TMA_LOAD{}, X_down, slayout_x(_, _, 0));
  auto down_tma_y = make_tma_copy(SM90_TMA_STORE{}, Y_down, cpbox_yt);

  auto *gate_up_tma_xy = static_cast<cute::TmaDescriptor *>(gate_up_tmas_ptr);
  auto *down_tma_xy = static_cast<cute::TmaDescriptor *>(down_tmas_ptr);

  // 0. count tokens
  {
    constexpr int kThreadPerBlock = 256;
    constexpr int kGroupPerThread = 2;

    if (num_tokens <= 128) {
      dim3 block(kThreadPerBlock);
      dim3 grid(1);
      kernels::blockwise_count_kernel<kThreadPerBlock, kGroupPerThread, kTileM>
          <<<grid, block, num_experts * 4, stream>>>(
              (const int *)topk_ids_ptr, (int *)topk_pos_ptr, (int *)num_tokens_per_group_ptr,
              (int *)cu_num_tokens_per_group_ptr, (int *)tiles_ptr, (int *)cu_tiles_ptr, num_tokens,
              num_topk, total_num_topk, num_experts, start_expert, end_expert);
    } else {
      // 0.1 count seqlens
      {
        dim3 block(kThreadPerBlock);
        dim3 grid((total_num_topk + kThreadPerBlock - 1) / kThreadPerBlock);
        kernels::blockwise_count_tokens_kernel<<<grid, block, 0, stream>>>(
            (const int *)topk_ids_ptr, (int *)topk_pos_ptr, (int *)num_tokens_per_group_ptr,
            total_num_topk, start_expert, end_expert);
      }
      // 0.2 count cu_seqlens
      {
        dim3 block(kThreadPerBlock);
        dim3 grid(1);
        kernels::blockwise_count_cutokens_kernel<kThreadPerBlock, kGroupPerThread, kTileM>
            <<<grid, block, num_experts * 4, stream>>>(
                (const int *)topk_ids_ptr, (int *)topk_pos_ptr, (int *)num_tokens_per_group_ptr,
                (int *)cu_num_tokens_per_group_ptr, (int *)tiles_ptr, (int *)cu_tiles_ptr,
                num_tokens, num_topk, total_num_topk, num_experts, start_expert, end_expert);
      }
    }
  }

  // 1. gather token and update tmas
  {
    vec_t<cute::TmaDescriptor, 4> td_xy{
        *gata_up_tma_x.get_tma_descriptor(),
        *gata_up_tma_y.get_tma_descriptor(),
        *down_tma_x.get_tma_descriptor(),
        *down_tma_y.get_tma_descriptor(),
    };

    constexpr int kWarpPerBlock = 4;
    int num_block_for_copy = (total_num_topk + kWarpPerBlock - 1) / kWarpPerBlock;

    cutlass::FastDivmod topk_divider(num_topk);

    dim3 block(kWarpPerBlock * 32);
    dim3 grid(num_block_for_copy + num_experts);

    int input_scale_size = hidden_size / 128;
    kernels::blockwise_gather_kernel<Tin, Tout, decltype(gata_up_tma_x), decltype(gata_up_tma_y),
                                     decltype(down_tma_x), decltype(down_tma_y), kWarpPerBlock,
                                     kTileM><<<grid, block, 0, stream>>>(
        td_xy, gate_up_tma_xy, down_tma_xy, (const Tin *)input_ptr, (float *)input_scale_ptr,
        (Tin *)gate_up_input_ptr, (Tout *)gate_up_output_ptr, (float *)gate_up_input_scale_ptr,
        (Tin *)down_input_ptr, (Tout *)down_output_ptr, (const int *)topk_ids_ptr,
        (int *)topk_pos_ptr, (int *)num_tokens_per_group_ptr, (int *)cu_num_tokens_per_group_ptr,
        (int *)cu_tiles_ptr, total_num_topk, hidden_size, intermediate_size, input_scale_size,
        num_padded_tokens, start_expert, end_expert, num_block_for_copy, topk_divider);
  }
}

void blockwise_count_and_gather_async(
    const void *input_ptr, const void *input_scale_ptr, void *gate_up_input_ptr,
    void *gate_up_output_ptr, void *gate_up_input_scale_ptr, void *down_input_ptr,
    void *down_output_ptr, const void *topk_ids_ptr, void *topk_pos_ptr,
    void *num_tokens_per_group_ptr, void *cu_num_tokens_per_group_ptr, void *gate_up_tmas_ptr,
    void *down_tmas_ptr, void *tiles_ptr, void *cu_tiles_ptr, int num_tokens, int num_padded_tokens,
    int hidden_size, int intermediate_size, int num_topk, int num_experts_local, int eprank,
    int num_tokens_per_group_avg, cudaStream_t stream) {
  constexpr int kTileN = 128;
  constexpr int kTileK = 128;

  if (num_tokens_per_group_avg <= 16) {
    constexpr int kTileM = 16;
    constexpr int kStage = 8;
    launch_blockwise_count_and_gather<kTileM, kTileN, kTileK, kStage>(
        input_ptr, input_scale_ptr, gate_up_input_ptr, gate_up_output_ptr, gate_up_input_scale_ptr,
        down_input_ptr, down_output_ptr, topk_ids_ptr, topk_pos_ptr, num_tokens_per_group_ptr,
        cu_num_tokens_per_group_ptr, gate_up_tmas_ptr, down_tmas_ptr, tiles_ptr, cu_tiles_ptr,
        num_tokens, num_padded_tokens, hidden_size, intermediate_size, num_topk, num_experts_local,
        eprank, num_tokens_per_group_avg, stream);
  } else if (num_tokens_per_group_avg <= 32) {
    constexpr int kTileM = 32;
    constexpr int kStage = 8;
    launch_blockwise_count_and_gather<kTileM, kTileN, kTileK, kStage>(
        input_ptr, input_scale_ptr, gate_up_input_ptr, gate_up_output_ptr, gate_up_input_scale_ptr,
        down_input_ptr, down_output_ptr, topk_ids_ptr, topk_pos_ptr, num_tokens_per_group_ptr,
        cu_num_tokens_per_group_ptr, gate_up_tmas_ptr, down_tmas_ptr, tiles_ptr, cu_tiles_ptr,
        num_tokens, num_padded_tokens, hidden_size, intermediate_size, num_topk, num_experts_local,
        eprank, num_tokens_per_group_avg, stream);
  } else if (num_tokens_per_group_avg <= 48) {
    constexpr int kTileM = 48;
    constexpr int kStage = 8;
    launch_blockwise_count_and_gather<kTileM, kTileN, kTileK, kStage>(
        input_ptr, input_scale_ptr, gate_up_input_ptr, gate_up_output_ptr, gate_up_input_scale_ptr,
        down_input_ptr, down_output_ptr, topk_ids_ptr, topk_pos_ptr, num_tokens_per_group_ptr,
        cu_num_tokens_per_group_ptr, gate_up_tmas_ptr, down_tmas_ptr, tiles_ptr, cu_tiles_ptr,
        num_tokens, num_padded_tokens, hidden_size, intermediate_size, num_topk, num_experts_local,
        eprank, num_tokens_per_group_avg, stream);
  } else {
    constexpr int kTileM = 64;
    constexpr int kStage = 8;
    launch_blockwise_count_and_gather<kTileM, kTileN, kTileK, kStage>(
        input_ptr, input_scale_ptr, gate_up_input_ptr, gate_up_output_ptr, gate_up_input_scale_ptr,
        down_input_ptr, down_output_ptr, topk_ids_ptr, topk_pos_ptr, num_tokens_per_group_ptr,
        cu_num_tokens_per_group_ptr, gate_up_tmas_ptr, down_tmas_ptr, tiles_ptr, cu_tiles_ptr,
        num_tokens, num_padded_tokens, hidden_size, intermediate_size, num_topk, num_experts_local,
        eprank, num_tokens_per_group_avg, stream);
  }
}

}  // namespace fuse_moe
}  // namespace hpc
