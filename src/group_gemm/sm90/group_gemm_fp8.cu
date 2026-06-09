// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include <cub/cub.cuh>

#include "cute/tensor.hpp"
#include "src/group_gemm/sm90/config.h"
#include "src/group_gemm/sm90/group_gemm.h"
#include "src/group_gemm/sm90/kernels.cuh"

namespace hpc {
namespace group_gemm {

template <int kTileM, int kTileN, int kTileK, int kStage, int kWarpgroupM, int kWarpgroupN,
          int kSwizzleX, int kSwizzleW, int kSwizzleY>
void launch_group_gemm_fp8(void *y_ptr, const void *x_ptr, const void *w_ptr,
                           const void *seqlens_ptr, const void *cu_seqlens_ptr, const void *y_scale,
                           void *tmas_ptr, void *tiles_ptr, void *cu_tiles_ptr, void *task_map_ptr,
                           int num_waves, int num_group, int m, int n, int k, bool update_tma,
                           bool use_pdl, cudaStream_t stream) {
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
  int num_sm = get_sm_count();
  // 0. update tma
  if (update_tma) {
    vec_t<cute::TmaDescriptor, 2> td_xy{
        *tma_x.get_tma_descriptor(),
        *tma_y.get_tma_descriptor(),
    };

    constexpr int kGroupPerThread = 8;
    constexpr int kThreadPerBlock = 32;

    if (use_pdl) {
      constexpr bool kUsePDL = true;
      cudaLaunchAttribute attribute[1];
      attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attribute[0].val.programmaticStreamSerializationAllowed = 1;

      // Set the attribute in a kernel launch configuration
      cudaLaunchConfig_t launch_config{};

      // Add special attribute for PDL
      launch_config.attrs = attribute;
      launch_config.numAttrs = 1;

      if (task_map_ptr != nullptr) {
        constexpr bool kAssignTask = true;
        // Base launch configuration
        launch_config.gridDim = 2 * num_group + 1;
        launch_config.blockDim = kThreadPerBlock;
        launch_config.dynamicSmemBytes = 0;
        launch_config.stream = stream;
        auto kernel =
            kernels::update_grouped_tma<Tin, Tout, decltype(tma_x), decltype(tma_y), kTileM,
                                        kGroupPerThread, kThreadPerBlock, kAssignTask, kUsePDL>;
        cudaLaunchKernelEx(&launch_config, kernel, td_xy, tma_xy, (const Tin *)x_ptr,
                           (const Tout *)y_ptr, (const int *)seqlens_ptr,
                           (const int *)cu_seqlens_ptr, (int *)tiles_ptr, (int *)cu_tiles_ptr,
                           (int4 *)task_map_ptr, num_group, m, n, k, num_sm);
      } else {
        constexpr bool kAssignTask = false;
        // Base launch configuration
        launch_config.gridDim = num_group + 1;
        launch_config.blockDim = kThreadPerBlock;
        launch_config.dynamicSmemBytes = 0;
        launch_config.stream = stream;
        auto kernel =
            kernels::update_grouped_tma<Tin, Tout, decltype(tma_x), decltype(tma_y), kTileM,
                                        kGroupPerThread, kThreadPerBlock, kAssignTask, kUsePDL>;
        cudaLaunchKernelEx(&launch_config, kernel, td_xy, tma_xy, (const Tin *)x_ptr,
                           (const Tout *)y_ptr, (const int *)seqlens_ptr,
                           (const int *)cu_seqlens_ptr, (int *)tiles_ptr, (int *)cu_tiles_ptr,
                           (int4 *)task_map_ptr, num_group, m, n, k, num_sm);
      }
    } else {
      constexpr bool kUsePDL = false;
      if (task_map_ptr != nullptr) {
        constexpr bool kAssignTask = true;
        kernels::update_grouped_tma<Tin, Tout, decltype(tma_x), decltype(tma_y), kTileM,
                                    kGroupPerThread, kThreadPerBlock, kAssignTask, kUsePDL>
            <<<2 * num_group + 1, kThreadPerBlock, 0, stream>>>(
                td_xy, tma_xy, (const Tin *)x_ptr, (const Tout *)y_ptr, (const int *)seqlens_ptr,
                (const int *)cu_seqlens_ptr, (int *)tiles_ptr, (int *)cu_tiles_ptr,
                (int4 *)task_map_ptr, num_group, m, n, k, num_sm);
      } else {
        constexpr bool kAssignTask = false;
        kernels::update_grouped_tma<Tin, Tout, decltype(tma_x), decltype(tma_y), kTileM,
                                    kGroupPerThread, kThreadPerBlock, kAssignTask, kUsePDL>
            <<<num_group + 1, kThreadPerBlock, 0, stream>>>(
                td_xy, tma_xy, (const Tin *)x_ptr, (const Tout *)y_ptr, (const int *)seqlens_ptr,
                (const int *)cu_seqlens_ptr, (int *)tiles_ptr, (int *)cu_tiles_ptr,
                (int4 *)task_map_ptr, num_group, m, n, k, num_sm);
      }
    }
  }

  // 1. group gemm
  {
    int num_tile_n = (n + kTileN - 1) / kTileN;
    cutlass::FastDivmod flat_divider(num_tile_n);

    // dim3 block(size(Config::TiledMma{}) + 128);
    dim3 block(384);
    dim3 grid(num_sm);

    if (use_pdl) {
      constexpr bool kUsePDL = true;
      cudaLaunchAttribute attribute[1];
      attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attribute[0].val.programmaticStreamSerializationAllowed = 1;

      // Set the attribute in a kernel launch configuration
      cudaLaunchConfig_t launch_config{};

      // Add special attribute for PDL
      launch_config.attrs = attribute;
      launch_config.numAttrs = 1;

      // Base launch configuration
      launch_config.gridDim = grid;
      launch_config.blockDim = block;
      launch_config.stream = stream;

      if (task_map_ptr != nullptr) {
        constexpr bool kTaskLoopPolicy = 0;

        int shm_seq = sizeof(int4) * num_waves;
        int shm_size = config.get_shm_size() + shm_seq;
        launch_config.dynamicSmemBytes = shm_size;

        auto kernel =
            kernels::group_gemm_fp8_kernel<decltype(config), decltype(tma_x), decltype(tma_w),
                                           decltype(tma_y), kTaskLoopPolicy, kUsePDL>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

        cudaLaunchKernelEx(&launch_config, kernel, tma_w, tma_xy, (int *)seqlens_ptr,
                           (float *)y_scale, (int *)tiles_ptr, (int *)cu_tiles_ptr,
                           (int4 *)task_map_ptr, num_group, m, n, k, flat_divider);
      } else if (k <= 1024 || n <= 1024) {
        constexpr bool kTaskLoopPolicy = 1;
        int shm_seq = sizeof(int) * (num_group + 1);
        int shm_size = config.get_shm_size() + shm_seq;
        // Base launch configuration
        launch_config.dynamicSmemBytes = shm_size;
        auto kernel =
            kernels::group_gemm_fp8_kernel<decltype(config), decltype(tma_x), decltype(tma_w),
                                           decltype(tma_y), kTaskLoopPolicy, kUsePDL>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

        cudaLaunchKernelEx(&launch_config, kernel, tma_w, tma_xy, (int *)seqlens_ptr,
                           (float *)y_scale, (int *)tiles_ptr, (int *)cu_tiles_ptr,
                           (int4 *)task_map_ptr, num_group, m, n, k, flat_divider);
      } else {
        constexpr bool kTaskLoopPolicy = 2;
        int shm_seq = sizeof(int) * (num_group + 1);
        int shm_size = config.get_shm_size() + shm_seq;
        launch_config.dynamicSmemBytes = shm_size;
        auto kernel =
            kernels::group_gemm_fp8_kernel<decltype(config), decltype(tma_x), decltype(tma_w),
                                           decltype(tma_y), kTaskLoopPolicy, kUsePDL>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

        cudaLaunchKernelEx(&launch_config, kernel, tma_w, tma_xy, (int *)seqlens_ptr,
                           (float *)y_scale, (int *)tiles_ptr, (int *)cu_tiles_ptr,
                           (int4 *)task_map_ptr, num_group, m, n, k, flat_divider);
      }

    } else {
      constexpr bool kUsePDL = false;
      if (task_map_ptr != nullptr) {
        constexpr bool kTaskLoopPolicy = 0;

        int shm_seq = sizeof(int4) * num_waves;
        int shm_size = config.get_shm_size() + shm_seq;

        auto kernel =
            kernels::group_gemm_fp8_kernel<decltype(config), decltype(tma_x), decltype(tma_w),
                                           decltype(tma_y), kTaskLoopPolicy, kUsePDL>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

        kernel<<<grid, block, shm_size, stream>>>(
            tma_w, tma_xy, (int *)seqlens_ptr, (float *)y_scale, (int *)tiles_ptr,
            (int *)cu_tiles_ptr, (int4 *)task_map_ptr, num_group, m, n, k, flat_divider);
      } else if (k <= 1024 || n <= 1024) {
        constexpr bool kTaskLoopPolicy = 1;
        int shm_seq = sizeof(int) * (num_group + 1);
        int shm_size = config.get_shm_size() + shm_seq;
        auto kernel =
            kernels::group_gemm_fp8_kernel<decltype(config), decltype(tma_x), decltype(tma_w),
                                           decltype(tma_y), kTaskLoopPolicy, kUsePDL>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

        kernel<<<grid, block, shm_size, stream>>>(
            tma_w, tma_xy, (int *)seqlens_ptr, (float *)y_scale, (int *)tiles_ptr,
            (int *)cu_tiles_ptr, (int4 *)task_map_ptr, num_group, m, n, k, flat_divider);
      } else {
        constexpr bool kTaskLoopPolicy = 2;
        int shm_seq = sizeof(int) * (num_group + 1);
        int shm_size = config.get_shm_size() + shm_seq;
        auto kernel =
            kernels::group_gemm_fp8_kernel<decltype(config), decltype(tma_x), decltype(tma_w),
                                           decltype(tma_y), kTaskLoopPolicy, kUsePDL>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

        kernel<<<grid, block, shm_size, stream>>>(
            tma_w, tma_xy, (int *)seqlens_ptr, (float *)y_scale, (int *)tiles_ptr,
            (int *)cu_tiles_ptr, (int4 *)task_map_ptr, num_group, m, n, k, flat_divider);
      }
    }
  }
}

void group_gemm_fp8_async(void *y_ptr, const void *x_ptr, const void *w_ptr,
                          const void *seqlens_ptr, const void *cu_seqlens_ptr, const void *y_scale,
                          void *tmas_ptr, void *tiles_ptr, void *cu_tiles_ptr, void *task_map_ptr,
                          int num_waves, int num_group, int m, int n, int k,
                          int num_seq_per_group_avg, bool update_tma, bool use_pdl,
                          cudaStream_t stream) {
  constexpr int kTileN = 128;
  constexpr int kTileK = 128;
  constexpr int kWarpgroupM = 2;
  constexpr int kWarpgroupN = 1;
  constexpr int kSwizzleX = 128;
  constexpr int kSwizzleW = 128;
  constexpr int kSwizzleY = 64;

  use_pdl = true;

  if (num_seq_per_group_avg <= 8) {
    constexpr int kTileM = 8;
    constexpr int kStage = 8;
    launch_group_gemm_fp8<kTileM, kTileN, kTileK, kStage, kWarpgroupM, kWarpgroupN, kSwizzleX,
                          kSwizzleW, kSwizzleY>(
        y_ptr, x_ptr, w_ptr, seqlens_ptr, cu_seqlens_ptr, y_scale, tmas_ptr, tiles_ptr,
        cu_tiles_ptr, task_map_ptr, num_waves, num_group, m, n, k, update_tma, use_pdl, stream);
  } else if (num_seq_per_group_avg <= 16) {
    constexpr int kTileM = 16;
    constexpr int kStage = 8;
    launch_group_gemm_fp8<kTileM, kTileN, kTileK, kStage, kWarpgroupM, kWarpgroupN, kSwizzleX,
                          kSwizzleW, kSwizzleY>(
        y_ptr, x_ptr, w_ptr, seqlens_ptr, cu_seqlens_ptr, y_scale, tmas_ptr, tiles_ptr,
        cu_tiles_ptr, task_map_ptr, num_waves, num_group, m, n, k, update_tma, use_pdl, stream);
  } else if (num_seq_per_group_avg <= 32) {
    constexpr int kTileM = 32;
    constexpr int kStage = 8;
    launch_group_gemm_fp8<kTileM, kTileN, kTileK, kStage, kWarpgroupM, kWarpgroupN, kSwizzleX,
                          kSwizzleW, kSwizzleY>(
        y_ptr, x_ptr, w_ptr, seqlens_ptr, cu_seqlens_ptr, y_scale, tmas_ptr, tiles_ptr,
        cu_tiles_ptr, task_map_ptr, num_waves, num_group, m, n, k, update_tma, use_pdl, stream);
  } else if (num_seq_per_group_avg <= 48) {
    constexpr int kTileM = 48;
    constexpr int kStage = 8;
    launch_group_gemm_fp8<kTileM, kTileN, kTileK, kStage, kWarpgroupM, kWarpgroupN, kSwizzleX,
                          kSwizzleW, kSwizzleY>(
        y_ptr, x_ptr, w_ptr, seqlens_ptr, cu_seqlens_ptr, y_scale, tmas_ptr, tiles_ptr,
        cu_tiles_ptr, task_map_ptr, num_waves, num_group, m, n, k, update_tma, use_pdl, stream);
  } else if (num_seq_per_group_avg <= 64) {
    constexpr int kTileM = 64;
    // use kTileK=64 + SW64 to reduce padding waste when k <= 192
    if (k <= 192) {
      constexpr int kTileK = 64;
      constexpr int kStage = 8;
      constexpr int kSwizzleX = 64;
      constexpr int kSwizzleW = 64;
      launch_group_gemm_fp8<kTileM, kTileN, kTileK, kStage, kWarpgroupM, kWarpgroupN, kSwizzleX,
                            kSwizzleW, kSwizzleY>(
          y_ptr, x_ptr, w_ptr, seqlens_ptr, cu_seqlens_ptr, y_scale, tmas_ptr, tiles_ptr,
          cu_tiles_ptr, task_map_ptr, num_waves, num_group, m, n, k, update_tma, use_pdl, stream);
    } else {
      constexpr int kStage = 8;
      launch_group_gemm_fp8<kTileM, kTileN, kTileK, kStage, kWarpgroupM, kWarpgroupN, kSwizzleX,
                            kSwizzleW, kSwizzleY>(
          y_ptr, x_ptr, w_ptr, seqlens_ptr, cu_seqlens_ptr, y_scale, tmas_ptr, tiles_ptr,
          cu_tiles_ptr, task_map_ptr, num_waves, num_group, m, n, k, update_tma, use_pdl, stream);
    }
  } else if (num_seq_per_group_avg <= 96) {
    constexpr int kTileM = 48;
    constexpr int kStage = 8;
    launch_group_gemm_fp8<kTileM, kTileN, kTileK, kStage, kWarpgroupM, kWarpgroupN, kSwizzleX,
                          kSwizzleW, kSwizzleY>(
        y_ptr, x_ptr, w_ptr, seqlens_ptr, cu_seqlens_ptr, y_scale, tmas_ptr, tiles_ptr,
        cu_tiles_ptr, task_map_ptr, num_waves, num_group, m, n, k, update_tma, use_pdl, stream);
  } else if (num_seq_per_group_avg <= 128) {
    constexpr int kTileM = 32;
    constexpr int kStage = 8;
    launch_group_gemm_fp8<kTileM, kTileN, kTileK, kStage, kWarpgroupM, kWarpgroupN, kSwizzleX,
                          kSwizzleW, kSwizzleY>(
        y_ptr, x_ptr, w_ptr, seqlens_ptr, cu_seqlens_ptr, y_scale, tmas_ptr, tiles_ptr,
        cu_tiles_ptr, task_map_ptr, num_waves, num_group, m, n, k, update_tma, use_pdl, stream);
  } else if (num_seq_per_group_avg <= 144) {
    constexpr int kTileM = 48;
    constexpr int kStage = 8;
    launch_group_gemm_fp8<kTileM, kTileN, kTileK, kStage, kWarpgroupM, kWarpgroupN, kSwizzleX,
                          kSwizzleW, kSwizzleY>(
        y_ptr, x_ptr, w_ptr, seqlens_ptr, cu_seqlens_ptr, y_scale, tmas_ptr, tiles_ptr,
        cu_tiles_ptr, task_map_ptr, num_waves, num_group, m, n, k, update_tma, use_pdl, stream);
  } else {
    constexpr int kTileM = 64;
    constexpr int kStage = 8;
    launch_group_gemm_fp8<kTileM, kTileN, kTileK, kStage, kWarpgroupM, kWarpgroupN, kSwizzleX,
                          kSwizzleW, kSwizzleY>(
        y_ptr, x_ptr, w_ptr, seqlens_ptr, cu_seqlens_ptr, y_scale, tmas_ptr, tiles_ptr,
        cu_tiles_ptr, task_map_ptr, num_waves, num_group, m, n, k, update_tma, use_pdl, stream);
  }
}

}  // namespace group_gemm
}  // namespace hpc
