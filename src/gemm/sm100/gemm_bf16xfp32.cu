// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "cutlass/arch/reg_reconfig.h"
#include "src/gemm/gemm.h"
#include "src/gemm/sm100/gemm_bf16xfp32_kernels.cuh"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace gemm {

template <typename Tin, typename Tout, int kTileM, int kTileN, int kTileK, int kStageK, int kSplitK>
void launch_gemm_bf16xfp32_2sm_kernel(void *y_ptr, void *splitk_y_ptr, void *split_flag_ptr,
                                      const void *x_ptr, const void *w_high_ptr,
                                      const void *w_low_ptr, int m, int n, int k, float scale,
                                      cudaStream_t stream) {
  using namespace cute;  // NOLINT

  constexpr int kBlockSwizzle = 2;

  constexpr int kMmaSM = 2;

  constexpr int kEpiTileN = kTileN;

  constexpr int kClusterM = 2;
  constexpr int kClusterN = 1;
  constexpr int kClusterK = 1;
  constexpr int kClusters = kClusterM * kClusterN * kClusterK;

  constexpr int kCtaTileM = kTileM / kMmaSM;
  constexpr int kCtaTileN = kTileN / kMmaSM;

  constexpr int kStageTile = 2;

  static_assert(kSplitK == 1, "kSplitK must be 1 for gemm_bf16xfp32 2sm kernel.");

  using TY = std::conditional_t<(kSplitK > 1), float, Tout>;

  auto X = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(x_ptr)), make_shape(m, k),
                       make_stride(k, Int<1>{}));

  auto W_HIGH = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(w_high_ptr)),
                            make_shape(n, k), make_stride(k, Int<1>{}));

  auto W_LOW = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(w_low_ptr)),
                           make_shape(n, k), make_stride(k, Int<1>{}));

  auto Y = make_tensor(make_gmem_ptr(reinterpret_cast<TY *>(kSplitK > 1 ? splitk_y_ptr : y_ptr)),
                       make_shape(m, n, kSplitK), make_stride(n, Int<1>{}, n * m));

  auto tiled_mma = make_tiled_mma(SM100_MMA_F16BF16_2x1SM_SS<Tin, Tin, float, kTileM, kTileN,
                                                             UMMA::Major::K, UMMA::Major::K>{});

  auto slayout_x = tile_to_shape(UMMA::Layout_K_SW128_Atom<Tin>{},
                                 make_shape(Int<kCtaTileM>{}, Int<kTileK>{}, Int<kStageK>{}));
  auto slayout_w =
      tile_to_shape(UMMA::Layout_K_SW128_Atom<Tin>{},
                    make_shape(Int<kCtaTileN>{}, Int<kTileK>{}, Int<2>{}, Int<kStageK>{}));
  auto slayout_y =
      tile_to_shape(UMMA::Layout_K_SW64_Atom<TY>{}, make_shape(Int<kCtaTileM>{}, Int<kEpiTileN>{}));

  auto copybox_x =
      tile_to_shape(UMMA::Layout_K_SW128_Atom<Tin>{}, make_shape(Int<kTileM>{}, Int<kTileK>{}));
  auto copybox_w =
      tile_to_shape(UMMA::Layout_K_SW128_Atom<Tin>{}, make_shape(Int<kTileN>{}, Int<kTileK>{}));

  auto tma_x = make_tma_copy(SM100_TMA_2SM_LOAD{}, X, copybox_x, Int<kClusterN * kMmaSM>{});
  auto tma_wh = make_tma_copy(SM100_TMA_2SM_LOAD{}, W_HIGH, copybox_w, Int<kClusterM>{});
  auto tma_wl = make_tma_copy(SM100_TMA_2SM_LOAD{}, W_LOW, copybox_w, Int<kClusterM>{});
  auto tma_y = make_tma_copy(SM90_TMA_STORE{}, Y, slayout_y);

  static constexpr int shm_xw = (cosize(slayout_x) + cosize(slayout_w)) * sizeof(Tin);
  static constexpr int shm_y = cosize(slayout_y) * sizeof(Tout);
  static constexpr int shm_size = shm_xw + shm_y;

  int num_tile_m = (m + kTileM - 1) / kTileM;
  int num_tile_n = (n + kTileN - 1) / kTileN;

  auto kernel = kernels::gemm_bf16xfp32_2sm_kernel<
      Tin, TY, Tout, decltype(tiled_mma), decltype(tma_x), decltype(tma_wh), decltype(tma_wl),
      decltype(tma_y), decltype(slayout_x), decltype(slayout_w), decltype(slayout_y), kTileM,
      kTileN, kTileK, kStageK, kClusterM, kClusterN, kClusterK, kMmaSM, kEpiTileN, kStageTile,
      kBlockSwizzle, kSplitK>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

  dim3 block(256);
  dim3 grid(num_tile_m * kMmaSM * num_tile_n * kSplitK);

  cutlass::FastDivmod swizzle_divider(kBlockSwizzle * num_tile_n * kSplitK);
  cutlass::FastDivmod flat_divider(num_tile_m * kMmaSM);
  cutlass::FastDivmod reduce_flat_divider(num_tile_n);

  cudaLaunchConfig_t config;
  memset(&config, 0, sizeof(config));

  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = shm_size;

  cudaLaunchAttribute attribute[1];
  attribute[0].id = cudaLaunchAttributeClusterDimension;
  attribute[0].val.clusterDim.x = kClusters;
  attribute[0].val.clusterDim.y = 1;
  attribute[0].val.clusterDim.z = 1;

  config.numAttrs = 1;
  config.attrs = attribute;

  config.stream = stream;

  cudaLaunchKernelEx(&config, kernel, tma_x, tma_wh, tma_wl, tma_y, reinterpret_cast<Tout *>(y_ptr),
                     reinterpret_cast<float *>(splitk_y_ptr),
                     reinterpret_cast<int *>(split_flag_ptr), m, n, k, scale, num_tile_m,
                     num_tile_n, swizzle_divider, flat_divider, reduce_flat_divider);
}

template <typename Tin, typename Tout, int kTileM, int kTileN, int kTileK, int kStageK, int kSplitK>
void launch_gemm_bf16xfp32_1sm_cluster_splitk_kernel(void *y_ptr, void *splitk_y_ptr,
                                                     void *split_flag_ptr, const void *x_ptr,
                                                     const void *w_high_ptr, const void *w_low_ptr,
                                                     int m, int n, int k, float scale,
                                                     cudaStream_t stream) {
  using namespace cute;  // NOLINT

  constexpr int kBlockSwizzle = 2;

  constexpr int kEpiTileN = kTileN;

  constexpr int kClusterM = 1;
  constexpr int kClusterN = 1;
  constexpr int kClusterK = kSplitK;
  constexpr int kClusters = kClusterM * kClusterN * kClusterK;

  constexpr int kStageTile = 2;

  auto X = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(x_ptr)), make_shape(m, k),
                       make_stride(k, Int<1>{}));

  auto W_HIGH = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(w_high_ptr)),
                            make_shape(n, k), make_stride(k, Int<1>{}));

  auto W_LOW = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(w_low_ptr)),
                           make_shape(n, k), make_stride(k, Int<1>{}));

  auto Y = make_tensor(make_gmem_ptr(reinterpret_cast<Tout *>(y_ptr)), make_shape(m, n),
                       make_stride(n, Int<1>{}));

  auto tiled_mma = make_tiled_mma(
      SM100_MMA_F16BF16_SS<Tin, Tin, float, kTileM, kTileN, UMMA::Major::K, UMMA::Major::K>{});

  auto slayout_x = tile_to_shape(UMMA::Layout_K_SW128_Atom<Tin>{},
                                 make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStageK>{}));
  auto slayout_w =
      tile_to_shape(UMMA::Layout_K_SW128_Atom<Tin>{},
                    make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<2>{}, Int<kStageK>{}));
  auto slayout_splity = tile_to_shape(UMMA::Layout_K_INTER_Atom<float>{},
                                      make_shape(Int<kTileM>{}, Int<kEpiTileN>{}));
  auto slayout_y = tile_to_shape(UMMA::Layout_K_SW64_Atom<Tout>{},
                                 make_shape(Int<kTileM / kSplitK>{}, Int<kEpiTileN>{}));

  auto copybox_x =
      tile_to_shape(UMMA::Layout_K_SW128_Atom<Tin>{}, make_shape(Int<kTileM>{}, Int<kTileK>{}));
  auto copybox_w =
      tile_to_shape(UMMA::Layout_K_SW128_Atom<Tin>{}, make_shape(Int<kTileN>{}, Int<kTileK>{}));

  auto tma_x = make_tma_copy(SM90_TMA_LOAD{}, X, copybox_x);
  auto tma_wh = make_tma_copy(SM90_TMA_LOAD{}, W_HIGH, copybox_w);
  auto tma_wl = make_tma_copy(SM90_TMA_LOAD{}, W_LOW, copybox_w);
  auto tma_y = make_tma_copy(SM90_TMA_STORE{}, Y, slayout_y);

  static constexpr int shm_xw = (cosize(slayout_x) + cosize(slayout_w)) * sizeof(Tin);
  static constexpr int shm_y = cosize(slayout_y) * sizeof(Tout);
  static constexpr int shm_splity = cosize(slayout_splity) * sizeof(float);
  int shm_size = shm_xw + shm_y;

  if constexpr (kSplitK > 1) {
    shm_size += shm_splity;
  }

  using SLayoutSplitY =
      std::conditional_t<(kSplitK > 1), decltype(slayout_splity), decltype(slayout_y)>;

  int num_tile_m = (m + kTileM - 1) / kTileM;
  int num_tile_n = (n + kTileN - 1) / kTileN;

  auto kernel = kernels::gemm_bf16xfp32_1sm_cluster_splitk_kernel<
      Tin, Tout, decltype(tiled_mma), decltype(tma_x), decltype(tma_wh), decltype(tma_wl),
      decltype(tma_y), decltype(slayout_x), decltype(slayout_w), SLayoutSplitY, decltype(slayout_y),
      kTileM, kTileN, kTileK, kStageK, kClusterM, kClusterN, kClusterK, kEpiTileN, kStageTile,
      kBlockSwizzle, kSplitK>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

  dim3 block(256);
  dim3 grid(num_tile_m * num_tile_n * kSplitK);

  cutlass::FastDivmod swizzle_divider(kBlockSwizzle * num_tile_n * kSplitK);
  cutlass::FastDivmod flat_divider(num_tile_n * kSplitK);

  cudaLaunchConfig_t config;
  memset(&config, 0, sizeof(config));

  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = shm_size;

  cudaLaunchAttribute attribute[1];
  attribute[0].id = cudaLaunchAttributeClusterDimension;
  attribute[0].val.clusterDim.x = kClusters;
  attribute[0].val.clusterDim.y = 1;
  attribute[0].val.clusterDim.z = 1;

  config.numAttrs = 1;
  config.attrs = attribute;

  config.stream = stream;

  cudaLaunchKernelEx(&config, kernel, tma_x, tma_wh, tma_wl, tma_y, m, n, k, scale, num_tile_m,
                     num_tile_n, swizzle_divider, flat_divider);
}

bool gemm_bf16xfp32_async(void *y_ptr, void *splitk_y_ptr, void *split_flag_ptr, const void *x_ptr,
                          const void *w_high_ptr, const void *w_low_ptr, int m, int n, int k,
                          float scale, bool use_fp32_output, int splitk, cudaStream_t stream) {
  if (splitk == 0) {
    constexpr int kTileM = 256;
    constexpr int kTileN = 64;
    constexpr int kSplitK = 1;
    if (use_fp32_output) {
      launch_gemm_bf16xfp32_2sm_kernel<cute::bfloat16_t, float, kTileM, kTileN, 64, 4, kSplitK>(
          y_ptr, splitk_y_ptr, split_flag_ptr, x_ptr, w_high_ptr, w_low_ptr, m, n, k, scale,
          stream);
    } else {
      launch_gemm_bf16xfp32_2sm_kernel<cute::bfloat16_t, cute::bfloat16_t, kTileM, kTileN, 64, 4,
                                       kSplitK>(y_ptr, splitk_y_ptr, split_flag_ptr, x_ptr,
                                                w_high_ptr, w_low_ptr, m, n, k, scale, stream);
    }
  } else {
    if (use_fp32_output) {
      if (m >= 2048) {
        constexpr int kTileM = 256;
        constexpr int kSplitK = 1;

        if (m < 3072) {
          constexpr int kTileN = 32;
          launch_gemm_bf16xfp32_2sm_kernel<cute::bfloat16_t, float, kTileM, kTileN, 64, 4, kSplitK>(
              y_ptr, splitk_y_ptr, split_flag_ptr, x_ptr, w_high_ptr, w_low_ptr, m, n, k, scale,
              stream);
        } else {
          constexpr int kTileN = 64;
          launch_gemm_bf16xfp32_2sm_kernel<cute::bfloat16_t, float, kTileM, kTileN, 64, 4, kSplitK>(
              y_ptr, splitk_y_ptr, split_flag_ptr, x_ptr, w_high_ptr, w_low_ptr, m, n, k, scale,
              stream);
        }
      } else {
        constexpr int kTileM = 128;
        constexpr int kTileN = 32;
        if (splitk == 8) {
          constexpr int kSplitK = 8;
          launch_gemm_bf16xfp32_1sm_cluster_splitk_kernel<cute::bfloat16_t, float, kTileM, kTileN,
                                                          64, 4, kSplitK>(
              y_ptr, splitk_y_ptr, split_flag_ptr, x_ptr, w_high_ptr, w_low_ptr, m, n, k, scale,
              stream);
        } else if (splitk == 4) {
          constexpr int kSplitK = 4;
          launch_gemm_bf16xfp32_1sm_cluster_splitk_kernel<cute::bfloat16_t, float, kTileM, kTileN,
                                                          64, 8, kSplitK>(
              y_ptr, splitk_y_ptr, split_flag_ptr, x_ptr, w_high_ptr, w_low_ptr, m, n, k, scale,
              stream);
        } else {
          constexpr int kSplitK = 1;
          launch_gemm_bf16xfp32_1sm_cluster_splitk_kernel<cute::bfloat16_t, float, kTileM, kTileN,
                                                          64, 8, kSplitK>(
              y_ptr, splitk_y_ptr, split_flag_ptr, x_ptr, w_high_ptr, w_low_ptr, m, n, k, scale,
              stream);
        }
      }
    } else {
      if (m >= 2048) {
        constexpr int kTileM = 256;
        constexpr int kSplitK = 1;

        if (m < 3072) {
          constexpr int kTileN = 32;
          launch_gemm_bf16xfp32_2sm_kernel<cute::bfloat16_t, cute::bfloat16_t, kTileM, kTileN, 64,
                                           4, kSplitK>(y_ptr, splitk_y_ptr, split_flag_ptr, x_ptr,
                                                       w_high_ptr, w_low_ptr, m, n, k, scale,
                                                       stream);
        } else {
          constexpr int kTileN = 96;
          launch_gemm_bf16xfp32_2sm_kernel<cute::bfloat16_t, cute::bfloat16_t, kTileM, kTileN, 64,
                                           4, kSplitK>(y_ptr, splitk_y_ptr, split_flag_ptr, x_ptr,
                                                       w_high_ptr, w_low_ptr, m, n, k, scale,
                                                       stream);
        }
      } else {
        constexpr int kTileM = 128;
        constexpr int kTileN = 32;
        if (splitk == 8) {
          constexpr int kSplitK = 8;
          launch_gemm_bf16xfp32_1sm_cluster_splitk_kernel<cute::bfloat16_t, cute::bfloat16_t,
                                                          kTileM, kTileN, 64, 4, kSplitK>(
              y_ptr, splitk_y_ptr, split_flag_ptr, x_ptr, w_high_ptr, w_low_ptr, m, n, k, scale,
              stream);
        } else if (splitk == 4) {
          constexpr int kSplitK = 4;
          launch_gemm_bf16xfp32_1sm_cluster_splitk_kernel<cute::bfloat16_t, cute::bfloat16_t,
                                                          kTileM, kTileN, 64, 8, kSplitK>(
              y_ptr, splitk_y_ptr, split_flag_ptr, x_ptr, w_high_ptr, w_low_ptr, m, n, k, scale,
              stream);
        } else {
          constexpr int kSplitK = 1;
          launch_gemm_bf16xfp32_1sm_cluster_splitk_kernel<cute::bfloat16_t, cute::bfloat16_t,
                                                          kTileM, kTileN, 64, 8, kSplitK>(
              y_ptr, splitk_y_ptr, split_flag_ptr, x_ptr, w_high_ptr, w_low_ptr, m, n, k, scale,
              stream);
        }
      }
    }
  }
  return true;
}

}  // namespace gemm
}  // namespace hpc
