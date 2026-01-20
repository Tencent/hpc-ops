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

template <int kTileM, int kTileN, int kTileK, int kStage, int kWarpgroupM, int kWarpgroupN,
          int kSwizzleX, int kSwizzleW, int kSwizzleY>
void launch_group_gemm_fp8(void *y_ptr, const void *x_ptr, const void *w_ptr,
                           const void *seqlens_ptr, const void *cu_seqlens_ptr, const void *y_scale,
                           void *tmas_ptr, void *tiles_ptr, void *cu_tiles_ptr, int num_group,
                           int m, int n, int k, bool update_tma, cudaStream_t stream) {
  using namespace cute;  // NOLINT

  using Tin = cute::float_e4m3_t;
  using Tout = cute::bfloat16_t;

  auto X = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(x_ptr)), make_shape(m, k),
                       make_stride(k, Int<1>{}));
  auto W = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(w_ptr)),
                       make_shape(n, k, num_group), make_stride(k, Int<1>{}, n * k));
  auto Y = make_tensor(make_gmem_ptr(reinterpret_cast<Tout *>(y_ptr)), make_shape(n, m),
                       make_stride(Int<1>{}, n));

  using Config = GroupGEMMFp8Config<Tin, Tout, kTileM, kTileN, kTileK, kStage, kWarpgroupM,
                                    kWarpgroupN, kSwizzleX, kSwizzleW, kSwizzleY>;
  Config config;
  auto [tma_x, tma_w, tma_y] = config.get_tma(X, W, Y);

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
          kernels::group_gemm_pertensor_fp8_kernel<decltype(config), decltype(tma_x),
                                                   decltype(tma_w), decltype(tma_y), IsLoopH>;
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

      kernel<<<grid, block, shm_size, stream>>>(tma_w, tma_xy, (int *)seqlens_ptr, (float *)y_scale,
                                                (int *)tiles_ptr, (int *)cu_tiles_ptr, num_group, m,
                                                n, k, flat_divider);
    } else {
      constexpr bool IsLoopH = false;
      auto kernel =
          kernels::group_gemm_pertensor_fp8_kernel<decltype(config), decltype(tma_x),
                                                   decltype(tma_w), decltype(tma_y), IsLoopH>;
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

      kernel<<<grid, block, shm_size, stream>>>(tma_w, tma_xy, (int *)seqlens_ptr, (float *)y_scale,
                                                (int *)tiles_ptr, (int *)cu_tiles_ptr, num_group, m,
                                                n, k, flat_divider);
    }
  }
}

void group_gemm_pertensor_fp8_async(void *y_ptr, const void *x_ptr, const void *w_ptr,
                                    const void *seqlens_ptr, const void *cu_seqlens_ptr,
                                    const void *y_scale, void *tmas_ptr, void *tiles_ptr,
                                    void *cu_tiles_ptr, int num_group, int m, int n, int k,
                                    int num_seq_per_group_avg, bool update_tma,
                                    cudaStream_t stream) {
  constexpr int kTileN = 128;
  constexpr int kTileK = 128;
  constexpr int kWarpgroupM = 2;
  constexpr int kWarpgroupN = 1;
  constexpr int kSwizzleX = 128;
  constexpr int kSwizzleW = 128;
  constexpr int kSwizzleY = 64;

  if (num_seq_per_group_avg <= 16) {
    constexpr int kTileM = 16;
    constexpr int kStage = 8;
    launch_group_gemm_fp8<kTileM, kTileN, kTileK, kStage, kWarpgroupM, kWarpgroupN, kSwizzleX,
                          kSwizzleW, kSwizzleY>(y_ptr, x_ptr, w_ptr, seqlens_ptr, cu_seqlens_ptr,
                                                y_scale, tmas_ptr, tiles_ptr, cu_tiles_ptr,
                                                num_group, m, n, k, update_tma, stream);
  } else if (num_seq_per_group_avg <= 32) {
    constexpr int kTileM = 32;
    constexpr int kStage = 8;
    launch_group_gemm_fp8<kTileM, kTileN, kTileK, kStage, kWarpgroupM, kWarpgroupN, kSwizzleX,
                          kSwizzleW, kSwizzleY>(y_ptr, x_ptr, w_ptr, seqlens_ptr, cu_seqlens_ptr,
                                                y_scale, tmas_ptr, tiles_ptr, cu_tiles_ptr,
                                                num_group, m, n, k, update_tma, stream);
  } else if (num_seq_per_group_avg <= 48) {
    constexpr int kTileM = 48;
    constexpr int kStage = 8;
    launch_group_gemm_fp8<kTileM, kTileN, kTileK, kStage, kWarpgroupM, kWarpgroupN, kSwizzleX,
                          kSwizzleW, kSwizzleY>(y_ptr, x_ptr, w_ptr, seqlens_ptr, cu_seqlens_ptr,
                                                y_scale, tmas_ptr, tiles_ptr, cu_tiles_ptr,
                                                num_group, m, n, k, update_tma, stream);
  } else {
    constexpr int kTileM = 64;
    constexpr int kStage = 8;
    launch_group_gemm_fp8<kTileM, kTileN, kTileK, kStage, kWarpgroupM, kWarpgroupN, kSwizzleX,
                          kSwizzleW, kSwizzleY>(y_ptr, x_ptr, w_ptr, seqlens_ptr, cu_seqlens_ptr,
                                                y_scale, tmas_ptr, tiles_ptr, cu_tiles_ptr,
                                                num_group, m, n, k, update_tma, stream);
  }
}

}  // namespace group_gemm
}  // namespace hpc
