// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include <cub/cub.cuh>

#include "cute/tensor.hpp"
#include "src/group_gemm/sm100/gemm_config.h"
#include "src/group_gemm/sm100/group_gemm.h"
#include "src/group_gemm/sm100/group_gemm_fp8_with_reduce.cuh"

namespace hpc {
namespace group_gemm {
template <typename GemmConfig>

void launch_group_gemm_2sm_fp8_with_reduce(void *y_ptr, const void *x_ptr, const void *w_ptr,
                                           const void *seqlens_ptr, const void *cu_seqlens_ptr,
                                           const void *y_scale, void *tmas_ptr, void *tiles_ptr,
                                           void *cu_tiles_ptr, void *x_row_map_ptr,
                                           void *topk_scale_row_map_ptr, int num_group, int m,
                                           int n, int k, bool update_tma, bool use_pdl,
                                           cudaStream_t stream) {
  using namespace cute;  // NOLINT

  using Tin = typename GemmConfig::Tin;
  using Tout = typename GemmConfig::Tout;
  using Tacc = typename GemmConfig::Tacc;

  auto W = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(w_ptr)),
                       make_shape(n, k, num_group), make_stride(k, Int<1>{}, n * k));

  auto X = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(x_ptr)), make_shape(m, k),
                       make_stride(k, Int<1>{}));

  auto Y = make_tensor(make_gmem_ptr(reinterpret_cast<Tout *>(y_ptr)), make_shape(n, m),
                       make_stride(Int<1>{}, n));

  GemmConfig config;
  auto [tma_x, tma_w, tma_y] = config.get_tma(X, W, Y);
  constexpr int kClusters = GemmConfig::kClusters;

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

      // Base launch configuration
      launch_config.gridDim = num_group + 1;
      launch_config.blockDim = kThreadPerBlock;
      launch_config.dynamicSmemBytes = 0;
      launch_config.stream = stream;
      auto kernel = kernels::update_grouped_tma<Tin, Tout, decltype(tma_x), decltype(tma_y),
                                                GemmConfig::kTileM, kGroupPerThread,
                                                kThreadPerBlock, kUsePDL>;
      cudaLaunchKernelEx(&launch_config, kernel, td_xy, tma_xy, (const Tin *)x_ptr,
                         (const Tout *)y_ptr, (const int *)seqlens_ptr, (const int *)cu_seqlens_ptr,
                         (int *)tiles_ptr, (int *)cu_tiles_ptr, num_group, m, n, k);
    } else {
      constexpr bool kUsePDL = false;
      kernels::update_grouped_tma<Tin, Tout, decltype(tma_x), decltype(tma_y), GemmConfig::kTileM,
                                  kGroupPerThread, kThreadPerBlock, kUsePDL>
          <<<num_group + 1, kThreadPerBlock, 0, stream>>>(
              td_xy, tma_xy, (const Tin *)x_ptr, (const Tout *)y_ptr, (const int *)seqlens_ptr,
              (const int *)cu_seqlens_ptr, (int *)tiles_ptr, (int *)cu_tiles_ptr, num_group, m, n,
              k);
    }
  }

  // 1. group gemm
  {
    int num_tile_n_per_group = (n + GemmConfig::kTileN - 1) / GemmConfig::kTileN;
    cutlass::FastDivmod flat_divider(num_tile_n_per_group * GemmConfig::kMmaSM);

    int shm_seq = sizeof(int) * (num_group + 1);
    int shm_size = config.get_shm_size() + shm_seq;

    dim3 block(256);
    dim3 grid(num_sm);

    cudaLaunchConfig_t launch_config;
    memset(&launch_config, 0, sizeof(launch_config));

    launch_config.gridDim = grid;
    launch_config.blockDim = block;
    launch_config.dynamicSmemBytes = shm_size;
    launch_config.stream = stream;

    if (use_pdl) {
      constexpr bool kUsePDL = true;
      cudaLaunchAttribute attribute[2];
      attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attribute[0].val.programmaticStreamSerializationAllowed = 1;

      attribute[1].id = cudaLaunchAttributeClusterDimension;
      attribute[1].val.clusterDim.x = kClusters;
      attribute[1].val.clusterDim.y = 1;
      attribute[1].val.clusterDim.z = 1;

      launch_config.attrs = attribute;
      launch_config.numAttrs = 2;

      if (true) {
        constexpr bool kTaskLoopPolicy = 1;
        auto kernel =
            kernels::group_gemm_2sm_fp8_with_reduce_kernel<decltype(config), decltype(tma_x),
                                                           decltype(tma_w), decltype(tma_y),
                                                           kTaskLoopPolicy, kUsePDL>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

        cudaLaunchKernelEx(&launch_config, kernel, tma_w, tma_xy, (Tout *)y_ptr, (int *)seqlens_ptr,
                           (int *)cu_seqlens_ptr, (float *)y_scale, (int *)tiles_ptr,
                           (int *)cu_tiles_ptr, (int *)x_row_map_ptr,
                           (float *)topk_scale_row_map_ptr, num_group, m, n, k, flat_divider);
      }

    } else {
      constexpr bool kUsePDL = false;
      cudaLaunchAttribute attribute[1];
      attribute[0].id = cudaLaunchAttributeClusterDimension;
      attribute[0].val.clusterDim.x = kClusters;
      attribute[0].val.clusterDim.y = 1;
      attribute[0].val.clusterDim.z = 1;

      launch_config.attrs = attribute;
      launch_config.numAttrs = 1;

      if (true) {
        constexpr bool kTaskLoopPolicy = 1;
        auto kernel =
            kernels::group_gemm_2sm_fp8_with_reduce_kernel<decltype(config), decltype(tma_x),
                                                           decltype(tma_w), decltype(tma_y),
                                                           kTaskLoopPolicy, kUsePDL>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

        cudaLaunchKernelEx(&launch_config, kernel, tma_w, tma_xy, (Tout *)y_ptr, (int *)seqlens_ptr,
                           (int *)cu_seqlens_ptr, (float *)y_scale, (int *)tiles_ptr,
                           (int *)cu_tiles_ptr, (int *)x_row_map_ptr,
                           (float *)topk_scale_row_map_ptr, num_group, m, n, k, flat_divider);
      }
    }
  }
}

template <typename GemmConfig, int kCtaPerSm>
void launch_group_gemm_1sm_fp8_with_reduce(void *y_ptr, const void *x_ptr, const void *w_ptr,
                                           const void *seqlens_ptr, const void *cu_seqlens_ptr,
                                           const void *y_scale, void *tmas_ptr, void *tiles_ptr,
                                           void *cu_tiles_ptr, void *x_row_map_ptr,
                                           void *topk_scale_row_map_ptr, int num_group, int m,
                                           int n, int k, bool update_tma, bool use_pdl,
                                           cudaStream_t stream) {
  using namespace cute;  // NOLINT

  using Tin = typename GemmConfig::Tin;
  using Tout = typename GemmConfig::Tout;
  using Tacc = typename GemmConfig::Tacc;

  auto W = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(w_ptr)),
                       make_shape(n, k, num_group), make_stride(k, Int<1>{}, n * k));

  auto X = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(x_ptr)), make_shape(m, k),
                       make_stride(k, Int<1>{}));

  auto Y = make_tensor(make_gmem_ptr(reinterpret_cast<Tout *>(y_ptr)), make_shape(n, m),
                       make_stride(Int<1>{}, n));

  GemmConfig config;
  auto [tma_x, tma_w, tma_y] = config.get_tma(X, W, Y);
  constexpr int kClusters = GemmConfig::kClusters;

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

      // Base launch configuration
      launch_config.gridDim = num_group + 1;
      launch_config.blockDim = kThreadPerBlock;
      launch_config.dynamicSmemBytes = 0;
      launch_config.stream = stream;
      auto kernel = kernels::update_grouped_tma<Tin, Tout, decltype(tma_x), decltype(tma_y),
                                                GemmConfig::kTileM, kGroupPerThread,
                                                kThreadPerBlock, kUsePDL>;
      cudaLaunchKernelEx(&launch_config, kernel, td_xy, tma_xy, (const Tin *)x_ptr,
                         (const Tout *)y_ptr, (const int *)seqlens_ptr, (const int *)cu_seqlens_ptr,
                         (int *)tiles_ptr, (int *)cu_tiles_ptr, num_group, m, n, k);
    } else {
      constexpr bool kUsePDL = false;
      kernels::update_grouped_tma<Tin, Tout, decltype(tma_x), decltype(tma_y), GemmConfig::kTileM,
                                  kGroupPerThread, kThreadPerBlock, kUsePDL>
          <<<num_group + 1, kThreadPerBlock, 0, stream>>>(
              td_xy, tma_xy, (const Tin *)x_ptr, (const Tout *)y_ptr, (const int *)seqlens_ptr,
              (const int *)cu_seqlens_ptr, (int *)tiles_ptr, (int *)cu_tiles_ptr, num_group, m, n,
              k);
    }
  }

  // 1. group gemm
  {
    int num_tile_n_per_group = (n + GemmConfig::kTileN - 1) / GemmConfig::kTileN;
    cutlass::FastDivmod flat_divider(num_tile_n_per_group * GemmConfig::kMmaSM);

    int shm_seq = sizeof(int) * (num_group + 1);
    int shm_size = config.get_shm_size() + shm_seq;

    dim3 block(256);
    dim3 grid(num_sm * kCtaPerSm);

    cudaLaunchConfig_t launch_config;
    memset(&launch_config, 0, sizeof(launch_config));

    launch_config.gridDim = grid;
    launch_config.blockDim = block;
    launch_config.dynamicSmemBytes = shm_size;
    launch_config.stream = stream;

    if (use_pdl) {
      constexpr bool kUsePDL = true;
      cudaLaunchAttribute attribute[2];
      attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attribute[0].val.programmaticStreamSerializationAllowed = 1;

      attribute[1].id = cudaLaunchAttributeClusterDimension;
      attribute[1].val.clusterDim.x = kClusters;
      attribute[1].val.clusterDim.y = 1;
      attribute[1].val.clusterDim.z = 1;

      launch_config.attrs = attribute;
      launch_config.numAttrs = 2;

      if (true) {
        constexpr bool kTaskLoopPolicy = 1;

        auto kernel =
            kernels::group_gemm_1sm_fp8_with_reduce_kernel<decltype(config), decltype(tma_x),
                                                           decltype(tma_w), decltype(tma_y),
                                                           kTaskLoopPolicy, kUsePDL>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

        cudaLaunchKernelEx(&launch_config, kernel, tma_w, tma_xy, (Tout *)y_ptr, (int *)seqlens_ptr,
                           (int *)cu_seqlens_ptr, (float *)y_scale, (int *)tiles_ptr,
                           (int *)cu_tiles_ptr, (int *)x_row_map_ptr,
                           (float *)topk_scale_row_map_ptr, num_group, m, n, k, flat_divider);
      }

    } else {
      constexpr bool kUsePDL = false;
      cudaLaunchAttribute attribute[1];
      attribute[0].id = cudaLaunchAttributeClusterDimension;
      attribute[0].val.clusterDim.x = kClusters;
      attribute[0].val.clusterDim.y = 1;
      attribute[0].val.clusterDim.z = 1;

      launch_config.attrs = attribute;
      launch_config.numAttrs = 1;

      if (true) {
        constexpr bool kTaskLoopPolicy = 1;

        auto kernel =
            kernels::group_gemm_1sm_fp8_with_reduce_kernel<decltype(config), decltype(tma_x),
                                                           decltype(tma_w), decltype(tma_y),
                                                           kTaskLoopPolicy, kUsePDL>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

        cudaLaunchKernelEx(&launch_config, kernel, tma_w, tma_xy, (Tout *)y_ptr, (int *)seqlens_ptr,
                           (int *)cu_seqlens_ptr, (float *)y_scale, (int *)tiles_ptr,
                           (int *)cu_tiles_ptr, (int *)x_row_map_ptr,
                           (float *)topk_scale_row_map_ptr, num_group, m, n, k, flat_divider);
      }
    }
  }
}

void group_gemm_2sm_fp8_with_reduce_async(void *y_ptr, const void *x_ptr, const void *w_ptr,
                                          const void *seqlens_ptr, const void *cu_seqlens_ptr,
                                          const void *y_scale, void *tmas_ptr, void *tiles_ptr,
                                          void *cu_tiles_ptr, void *task_map_ptr,
                                          void *x_row_map_ptr, void *topk_scale_row_map_ptr,
                                          int num_waves, int num_group, int m, int n, int k,
                                          int num_seq_per_group_avg, bool update_tma, bool use_pdl,
                                          cudaStream_t stream) {
  using namespace cute;  // NOLINT

  constexpr GroupGemmFunc kGemmFunc = GroupGemmFunc::GROUP_GEMM_2SM_FP8;

  auto launch = [&](auto num_seq_per_group_avg_tag) {
    constexpr int kNumSeqPerGroupAvg = decltype(num_seq_per_group_avg_tag)::value;
    using GemmConfig = decltype(get_group_gemm_config<kGemmFunc, kNumSeqPerGroupAvg>());
    launch_group_gemm_2sm_fp8_with_reduce<GemmConfig>(
        y_ptr, x_ptr, w_ptr, seqlens_ptr, cu_seqlens_ptr, y_scale, tmas_ptr, tiles_ptr,
        cu_tiles_ptr, x_row_map_ptr, topk_scale_row_map_ptr, num_group, m, n, k, update_tma,
        use_pdl, stream);
  };

  if (num_seq_per_group_avg <= 32) {
    launch(std::integral_constant<int, 32>{});
  } else if (num_seq_per_group_avg <= 64) {
    launch(std::integral_constant<int, 64>{});
  } else if (num_seq_per_group_avg <= 96) {
    launch(std::integral_constant<int, 96>{});
  } else if (num_seq_per_group_avg <= 128) {
    launch(std::integral_constant<int, 128>{});
  } else if (num_seq_per_group_avg <= 160) {
    launch(std::integral_constant<int, 160>{});
  } else if (num_seq_per_group_avg <= 192) {
    launch(std::integral_constant<int, 192>{});
  } else {
    launch(std::integral_constant<int, 256>{});
  }
}

void group_gemm_1sm_fp8_with_reduce_async(void *y_ptr, const void *x_ptr, const void *w_ptr,
                                          const void *seqlens_ptr, const void *cu_seqlens_ptr,
                                          const void *y_scale, void *tmas_ptr, void *tiles_ptr,
                                          void *cu_tiles_ptr, void *task_map_ptr,
                                          void *x_row_map_ptr, void *topk_scale_row_map_ptr,
                                          int num_waves, int num_group, int m, int n, int k,
                                          int num_seq_per_group_avg, bool update_tma, bool use_pdl,
                                          cudaStream_t stream) {
  using namespace cute;  // NOLINT

  constexpr GroupGemmFunc kGemmFunc = GroupGemmFunc::GROUP_GEMM_1SM_FP8;

  auto launch = [&](auto num_seq_per_group_avg_tag, auto cta_per_sm) {
    constexpr int kNumSeqPerGroupAvg = decltype(num_seq_per_group_avg_tag)::value;
    constexpr int kCtaPerSm = decltype(cta_per_sm)::value;
    using GemmConfig = decltype(get_group_gemm_config<kGemmFunc, kNumSeqPerGroupAvg>());
    launch_group_gemm_1sm_fp8_with_reduce<GemmConfig, kCtaPerSm>(
        y_ptr, x_ptr, w_ptr, seqlens_ptr, cu_seqlens_ptr, y_scale, tmas_ptr, tiles_ptr,
        cu_tiles_ptr, x_row_map_ptr, topk_scale_row_map_ptr, num_group, m, n, k, update_tma,
        use_pdl, stream);
  };

  if (num_seq_per_group_avg <= 16) {
    launch(std::integral_constant<int, 16>{}, std::integral_constant<int, 3>{});
  } else if (num_seq_per_group_avg <= 32) {
    launch(std::integral_constant<int, 32>{}, std::integral_constant<int, 3>{});
  } else if (num_seq_per_group_avg <= 48) {
    launch(std::integral_constant<int, 48>{}, std::integral_constant<int, 1>{});
  } else if (num_seq_per_group_avg <= 64) {
    launch(std::integral_constant<int, 64>{}, std::integral_constant<int, 1>{});
  } else {
    launch(std::integral_constant<int, 128>{}, std::integral_constant<int, 1>{});
  }
}

void group_gemm_fp8_with_reduce_async(void *y_ptr, const void *x_ptr, const void *w_ptr,
                                      const void *seqlens_ptr, const void *cu_seqlens_ptr,
                                      const void *y_scale, void *tmas_ptr, void *tiles_ptr,
                                      void *cu_tiles_ptr, void *task_map_ptr, void *x_row_map_ptr,
                                      void *topk_scale_row_map_ptr, int num_waves, int num_group,
                                      int m, int n, int k, int num_seq_per_group_avg,
                                      bool update_tma, bool use_pdl, cudaStream_t stream) {
  if (n % 256 == 0 && num_seq_per_group_avg > 32) {
    group_gemm_2sm_fp8_with_reduce_async(
        y_ptr, x_ptr, w_ptr, seqlens_ptr, cu_seqlens_ptr, y_scale, tmas_ptr, tiles_ptr,
        cu_tiles_ptr, task_map_ptr, x_row_map_ptr, topk_scale_row_map_ptr, num_waves, num_group, m,
        n, k, num_seq_per_group_avg, update_tma, use_pdl, stream);
  } else {
    group_gemm_1sm_fp8_with_reduce_async(
        y_ptr, x_ptr, w_ptr, seqlens_ptr, cu_seqlens_ptr, y_scale, tmas_ptr, tiles_ptr,
        cu_tiles_ptr, task_map_ptr, x_row_map_ptr, topk_scale_row_map_ptr, num_waves, num_group, m,
        n, k, num_seq_per_group_avg, update_tma, use_pdl, stream);
  }
}

}  // namespace group_gemm
}  // namespace hpc
