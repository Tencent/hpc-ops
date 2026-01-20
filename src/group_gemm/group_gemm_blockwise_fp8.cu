// Copyright (C) 2026 Tencent.

#include <cuda.h>
#include <stdio.h>

#include <cub/cub.cuh>

#include "cute/tensor.hpp"
#include "src/group_gemm/config.h"
#include "src/group_gemm/group_gemm.h"
#include "src/group_gemm/kernels.cuh"

namespace hpc {
namespace group_gemm {

namespace kernels {

template <int kNumWarpsPerCTA, int kElementsPerThread, int TM, int TN>
__global__ void reformat_x_scale_kernel(float *output_ptr, const float *xscale_ptr,
                                        const int *seqlens_ptr, const int *cu_seqlens_ptr,
                                        int num_group, int m, int n, int tilem) {
  __shared__ float shm_tile[TM][33];  // manual swizze for avoid bank conflict

  int iblock = blockIdx.x;
  int idx = threadIdx.x;

  // src location
  int src_global_row = cu_seqlens_ptr[iblock];
  int src_global_col = 0;
  constexpr int kVecsPerRow = TN / kElementsPerThread;
  int src_local_row = idx / kVecsPerRow;
  int src_local_col = idx % kVecsPerRow * kElementsPerThread;
  int src_row = src_global_row + src_local_row;
  int src_col = src_global_col + src_local_col;

  // dst location
  int dst_global_row = 0;
  int dst_global_col = 0;
#pragma unroll
  for (int i = idx; i < iblock; i += blockDim.x) {
    dst_global_col += (seqlens_ptr[i] + tilem - 1) / tilem * tilem;
  }
  using BlockReduce = cub::BlockReduce<int, kNumWarpsPerCTA * 32>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  dst_global_col = BlockReduce(temp_storage).Sum(dst_global_col);
  if (idx == 0) {
    shm_tile[0][32] = dst_global_col;
  }
  __syncthreads();
  dst_global_col = shm_tile[0][32];
  constexpr int kVecsPerCol = TM / kElementsPerThread;
  int dst_local_row = idx / kVecsPerCol;
  int dst_local_col = idx % kVecsPerCol * kElementsPerThread;
  int dst_row = dst_global_row + dst_local_row;
  int dst_col = dst_global_col + dst_local_col;

  // transpose
  int valid_seq_num = seqlens_ptr[iblock];
  int src_row_bound = src_global_row + valid_seq_num;
  int dst_col_bound = dst_global_col + valid_seq_num;

#pragma unroll 1
  for (int i = 0; i < valid_seq_num; i += TM) {
    // global memory(read row) -> shared memory(store row)
    if (src_row < src_row_bound) {
      // Because it reads 4 floats at a time, when the number of valid elements is
      // not divisible by 4, it will read extra invalid elements.
      auto r = load<float, kElementsPerThread>(xscale_ptr + src_row * n + src_col);
#pragma unroll
      for (int j = 0; j < kElementsPerThread; ++j) {
        shm_tile[src_local_row][src_local_col + j] = r[j];
      }
    }
    src_row += TM;

    __syncthreads();

    // shared memory(read column) -> global memory(store row)
    if (dst_col < dst_col_bound) {
      vec_t<float, kElementsPerThread> r;
#pragma unroll
      for (int j = 0; j < kElementsPerThread; ++j) {
        r[j] = shm_tile[dst_local_col + j][dst_local_row];
      }
      // Because it write 4 floats at a time, when the number of valid elements is
      // not divisible by 4, it will write extra invalid elements.
      // However, this doesn't matter, because the extra invalid elements will not
      // be used by the downstream group GEMM operation.
      store(output_ptr + dst_row * m + dst_col, r);
    }
    dst_col += TM;
  }
}

}  // namespace kernels

template <int kTileM, int kTileN, int kTileK, int kTileS, int kStage, int kWarpgroupM,
          int kWarpgroupN, int kSwizzleX, int kSwizzleW, int kSwizzleY>
void launch_group_gemm_blockwise_fp8(void *y_ptr, const void *x_ptr, const void *w_ptr,
                                     const void *seqlens_ptr, const void *cu_seqlens_ptr,
                                     const void *xscale_ptr, const void *wscale_ptr, void *tmas_ptr,
                                     void *tiles_ptr, void *cu_tiles_ptr, int num_group, int m,
                                     int n, int k, int m_pad, int num_block_k_pad4, bool update_tma,
                                     cudaStream_t stream) {
  using namespace cute;  // NOLINT

  using Tin = cute::float_e4m3_t;
  using Tout = cute::bfloat16_t;
  using TS = float;

  int num_block_k = k / kTileK;
  int num_block_n = n / kTileN;

  auto X = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(x_ptr)), make_shape(m, k),
                       make_stride(k, Int<1>{}));
  auto W = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(w_ptr)),
                       make_shape(n, k, num_group), make_stride(k, Int<1>{}, n * k));
  auto Y = make_tensor(make_gmem_ptr(reinterpret_cast<Tout *>(y_ptr)), make_shape(n, m),
                       make_stride(Int<1>{}, n));
  auto XS = make_tensor(make_gmem_ptr(reinterpret_cast<const TS *>(xscale_ptr)),
                        make_shape(num_block_k, m_pad), make_stride(m_pad, Int<1>{}));
  auto WS = make_tensor(make_gmem_ptr(reinterpret_cast<const TS *>(wscale_ptr)),
                        make_shape(num_block_n, num_block_k_pad4, num_group),
                        make_stride(num_block_k_pad4, Int<1>{}, num_block_n * num_block_k_pad4));

  using Config =
      GroupGEMMBlockWiseFp8Config<Tin, Tout, TS, kTileM, kTileN, kTileK, kTileS, kStage,
                                  kWarpgroupM, kWarpgroupN, kSwizzleX, kSwizzleW, kSwizzleY>;
  Config config;
  auto [tma_x, tma_w, tma_y, tma_xs, tma_ws] = config.get_tma(X, W, Y, XS, WS);

  auto *tma_xy = static_cast<cute::TmaDescriptor *>(tmas_ptr);

  // 0. update tma
  if (update_tma) {
    vec_t<cute::TmaDescriptor, 2> td_xy{
        *tma_x.get_tma_descriptor(),
        *tma_y.get_tma_descriptor(),
    };

    constexpr int kGroupPerThread = 8;
    constexpr int kThreadPerBlock = 32;
    kernels::update_grouped_tma<Tin, Tout, decltype(tma_x), decltype(tma_y), kTileM,
                                kGroupPerThread, kThreadPerBlock>
        <<<num_group + 1, kThreadPerBlock, 0, stream>>>(
            td_xy, tma_xy, (const Tin *)x_ptr, (const Tout *)y_ptr, (const int *)seqlens_ptr,
            (const int *)cu_seqlens_ptr, (int *)tiles_ptr, (int *)cu_tiles_ptr, num_group, m, n, k);
  }

  // 1. group gemm
  {
    int num_tile_n = (n + kTileN - 1) / kTileN;
    cutlass::FastDivmod flat_divider(num_tile_n);

    // dim3 block(size(Config::TiledMma{}) + 128);
    dim3 block(384);
    dim3 grid(get_sm_count());

    int shm_seq = sizeof(int) * (num_group + 1);
    int shm_size = config.get_shm_size() + shm_seq;

    if (k <= 1024 || n <= 1024) {
      constexpr bool IsLoopH = true;
      auto kernel =
          kernels::group_gemm_blockwise_fp8_kernel<decltype(config), decltype(tma_x),
                                                   decltype(tma_w), decltype(tma_y),
                                                   decltype(tma_xs), decltype(tma_ws), IsLoopH>;
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

      kernel<<<grid, block, shm_size, stream>>>(
          tma_w, tma_xs, tma_ws, tma_xy, (int *)seqlens_ptr, (float *)xscale_ptr,
          (float *)wscale_ptr, (int *)tiles_ptr, (int *)cu_tiles_ptr, num_group, m, n, k, m_pad,
          num_block_n, num_block_k, num_block_k_pad4, flat_divider);
    } else {
      constexpr bool IsLoopH = false;
      auto kernel =
          kernels::group_gemm_blockwise_fp8_kernel<decltype(config), decltype(tma_x),
                                                   decltype(tma_w), decltype(tma_y),
                                                   decltype(tma_xs), decltype(tma_ws), IsLoopH>;
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

      kernel<<<grid, block, shm_size, stream>>>(
          tma_w, tma_xs, tma_ws, tma_xy, (int *)seqlens_ptr, (float *)xscale_ptr,
          (float *)wscale_ptr, (int *)tiles_ptr, (int *)cu_tiles_ptr, num_group, m, n, k, m_pad,
          num_block_n, num_block_k, num_block_k_pad4, flat_divider);
    }
  }
}

void group_gemm_blockwise_fp8_async(void *y_ptr, const void *x_ptr, const void *w_ptr,
                                    const void *seqlens_ptr, const void *cu_seqlens_ptr,
                                    const void *xscale_ptr, const void *wscale_ptr, void *tmas_ptr,
                                    void *tiles_ptr, void *cu_tiles_ptr, int num_group, int m,
                                    int n, int k, int m_pad, int num_block_k_pad4,
                                    int num_seq_per_group_avg, bool update_tma,
                                    cudaStream_t stream) {
  constexpr int kTileN = 128;
  constexpr int kTileK = 128;
  constexpr int kTileS = 64;
  constexpr int kWarpgroupM = 2;
  constexpr int kWarpgroupN = 1;
  constexpr int kSwizzleX = 128;
  constexpr int kSwizzleW = 128;
  constexpr int kSwizzleY = 64;

  if (num_seq_per_group_avg <= 16) {
    constexpr int kTileM = 16;
    constexpr int kStage = 8;
    launch_group_gemm_blockwise_fp8<kTileM, kTileN, kTileK, kTileS, kStage, kWarpgroupM,
                                    kWarpgroupN, kSwizzleX, kSwizzleW, kSwizzleY>(
        y_ptr, x_ptr, w_ptr, seqlens_ptr, cu_seqlens_ptr, xscale_ptr, wscale_ptr, tmas_ptr,
        tiles_ptr, cu_tiles_ptr, num_group, m, n, k, m_pad, num_block_k_pad4, update_tma, stream);
  } else if (num_seq_per_group_avg <= 32) {
    constexpr int kTileM = 32;
    constexpr int kStage = 8;
    launch_group_gemm_blockwise_fp8<kTileM, kTileN, kTileK, kTileS, kStage, kWarpgroupM,
                                    kWarpgroupN, kSwizzleX, kSwizzleW, kSwizzleY>(
        y_ptr, x_ptr, w_ptr, seqlens_ptr, cu_seqlens_ptr, xscale_ptr, wscale_ptr, tmas_ptr,
        tiles_ptr, cu_tiles_ptr, num_group, m, n, k, m_pad, num_block_k_pad4, update_tma, stream);
  } else if (num_seq_per_group_avg <= 48) {
    constexpr int kTileM = 48;
    constexpr int kStage = 8;
    launch_group_gemm_blockwise_fp8<kTileM, kTileN, kTileK, kTileS, kStage, kWarpgroupM,
                                    kWarpgroupN, kSwizzleX, kSwizzleW, kSwizzleY>(
        y_ptr, x_ptr, w_ptr, seqlens_ptr, cu_seqlens_ptr, xscale_ptr, wscale_ptr, tmas_ptr,
        tiles_ptr, cu_tiles_ptr, num_group, m, n, k, m_pad, num_block_k_pad4, update_tma, stream);
  } else {
    constexpr int kTileM = 64;
    constexpr int kStage = 8;
    launch_group_gemm_blockwise_fp8<kTileM, kTileN, kTileK, kTileS, kStage, kWarpgroupM,
                                    kWarpgroupN, kSwizzleX, kSwizzleW, kSwizzleY>(
        y_ptr, x_ptr, w_ptr, seqlens_ptr, cu_seqlens_ptr, xscale_ptr, wscale_ptr, tmas_ptr,
        tiles_ptr, cu_tiles_ptr, num_group, m, n, k, m_pad, num_block_k_pad4, update_tma, stream);
  }
}

void reformat_x_scale_async(void *output_ptr, const void *xscale_ptr, const void *seqlens_ptr,
                            const void *cu_seqlens_ptr, int num_group, int m, int n, int tilem,
                            cudaStream_t stream) {
  if (n == 16) {
    dim3 block(128);
    // This is a temporary implementation
    // When seqlens[i] is large, maybe be unblance
    dim3 grid(num_group);

    constexpr int kNumWarpsPerCTA = 4;
    constexpr int kElementsPerThread = 4;
    constexpr int TM = 32;
    constexpr int TN = 16;

    kernels::reformat_x_scale_kernel<kNumWarpsPerCTA, kElementsPerThread, TM, TN>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<float *>(output_ptr), reinterpret_cast<const float *>(xscale_ptr),
            reinterpret_cast<const int *>(seqlens_ptr),
            reinterpret_cast<const int *>(cu_seqlens_ptr), num_group, m, n, tilem);
  } else {  // n == 32
    dim3 block(256);
    // This is a temporary implementation
    // When seqlens[i] is large, maybe be unblance
    dim3 grid(num_group);

    constexpr int kNumWarpsPerCTA = 8;
    constexpr int kElementsPerThread = 4;
    constexpr int TM = 32;
    constexpr int TN = 32;

    kernels::reformat_x_scale_kernel<kNumWarpsPerCTA, kElementsPerThread, TM, TN>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<float *>(output_ptr), reinterpret_cast<const float *>(xscale_ptr),
            reinterpret_cast<const int *>(seqlens_ptr),
            reinterpret_cast<const int *>(cu_seqlens_ptr), num_group, m, n, tilem);
  }
}

}  // namespace group_gemm
}  // namespace hpc
