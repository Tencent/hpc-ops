// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include <cub/cub.cuh>

#include "cute/tensor.hpp"
#include "src/group_gemm/sm100/config.h"
#include "src/group_gemm/sm100/group_gemm.h"
#include "src/group_gemm/sm100/group_gemm_cp_async_fp8.cuh"

namespace hpc {
namespace group_gemm {

template <int kTileM, int kTileN, int kTileK, int kStageK, int kClusterM, int kClusterN,
          int kClusterK, int kMmaSM, int kEpiTileN, int kStageTile, int kStageTMA>
void launch_group_gemm_1sm_cp_async_fp8(void *y_ptr, const void *x_ptr, const void *w_ptr,
                                        const void *seqlens_ptr, const void *cu_seqlens_ptr,
                                        const void *y_scale, void *tmas_ptr, void *tiles_ptr,
                                        void *cu_tiles_ptr, int num_group, int m_true, int n_true,
                                        int k, bool update_tma, bool use_pdl, cudaStream_t stream,
                                        const void *x_row_map_ptr = nullptr, int x_num_rows = 0) {
  using namespace cute;  // NOLINT

  using Tin = cute::float_e4m3_t;
  using Tout = cute::bfloat16_t;
  using Tacc = float;

  int m = n_true;
  int n = m_true;

  auto A = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(w_ptr)),
                       make_shape(m, k, num_group), make_stride(k, Int<1>{}, m * k));

  auto B = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(x_ptr)), make_shape(n, k),
                       make_stride(k, Int<1>{}));

  auto CT = make_tensor(make_gmem_ptr(reinterpret_cast<Tout *>(y_ptr)), make_shape(m, n),
                        make_stride(Int<1>{}, m));

  constexpr int kSwizzleX = kTileK;
  constexpr int kSwizzleW = kTileK;
  constexpr int kSwizzleY = 128;

  using GroupGEMMConfig =
      GroupGEMMFp8Config<Tin, Tout, Tacc, kTileM, kTileN, kTileK, kStageK, kClusterM, kClusterN,
                         kClusterK, kMmaSM, kEpiTileN, kStageTile, kStageTMA, kSwizzleX, kSwizzleW,
                         kSwizzleY>;

  auto tiled_mma = make_tiled_mma(
      MMA_Traits<SM100_MMA_F8F6F4_SS, Tin, Tin, Tacc, cute::C<kTileM>, cute::C<kTileN>,
                 cute::integral_constant<UMMA::Major, UMMA::Major::K>,
                 cute::integral_constant<UMMA::Major, UMMA::Major::K>,
                 cute::integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>,
                 cute::integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>>{});

  GroupGEMMConfig config;
  auto [tma_a, tma_b, tma_dt] = config.get_tma(A, B, CT);
  constexpr int kClusters = GroupGEMMConfig::kClusters;

  auto *tma_xy = static_cast<cute::TmaDescriptor *>(tmas_ptr);
  int num_sm = get_sm_count();
  // 0. update tma
  if (update_tma) {
    vec_t<cute::TmaDescriptor, 2> td_xy{
        *tma_b.get_tma_descriptor(),
        *tma_dt.get_tma_descriptor(),
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
      auto kernel = kernels::update_grouped_tma<Tin, Tout, decltype(tma_b), decltype(tma_dt),
                                                kTileN, kGroupPerThread, kThreadPerBlock, kUsePDL>;
      cudaLaunchKernelEx(&launch_config, kernel, td_xy, tma_xy, (const Tin *)x_ptr,
                         (const Tout *)y_ptr, (const int *)seqlens_ptr, (const int *)cu_seqlens_ptr,
                         (int *)tiles_ptr, (int *)cu_tiles_ptr, num_group, m_true, n_true, k);
    } else {
      constexpr bool kUsePDL = false;
      kernels::update_grouped_tma<Tin, Tout, decltype(tma_b), decltype(tma_dt), kTileN,
                                  kGroupPerThread, kThreadPerBlock, kUsePDL>
          <<<num_group + 1, kThreadPerBlock, 0, stream>>>(
              td_xy, tma_xy, (const Tin *)x_ptr, (const Tout *)y_ptr, (const int *)seqlens_ptr,
              (const int *)cu_seqlens_ptr, (int *)tiles_ptr, (int *)cu_tiles_ptr, num_group, m_true,
              n_true, k);
    }
  }

  // 1. group gemm
  {
    int num_tile_m_per_group = (m + kTileM - 1) / kTileM;
    cutlass::FastDivmod flat_divider(num_tile_m_per_group * kMmaSM);

    int shm_seq = sizeof(int) * (num_group + 1);
    int shm_size = config.get_shm_size() + shm_seq;

    dim3 block(384);
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
            kernels::group_gemm_1sm_cp_async_fp8_kernel<decltype(config), decltype(tiled_mma),
                                                        decltype(tma_a), decltype(tma_b),
                                                        decltype(tma_dt), kTaskLoopPolicy, kUsePDL>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

        cudaLaunchKernelEx(&launch_config, kernel, tma_a, tma_xy, (Tin *)w_ptr, (Tin *)x_ptr,
                           (int *)cu_seqlens_ptr, (int *)seqlens_ptr, (float *)y_scale,
                           (int *)tiles_ptr, (int *)cu_tiles_ptr, num_group, m, n, k, flat_divider,
                           (const int *)x_row_map_ptr, x_num_rows);
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
            kernels::group_gemm_1sm_cp_async_fp8_kernel<decltype(config), decltype(tiled_mma),
                                                        decltype(tma_a), decltype(tma_b),
                                                        decltype(tma_dt), kTaskLoopPolicy, kUsePDL>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

        cudaLaunchKernelEx(&launch_config, kernel, tma_a, tma_xy, (Tin *)w_ptr, (Tin *)x_ptr,
                           (int *)cu_seqlens_ptr, (int *)seqlens_ptr, (float *)y_scale,
                           (int *)tiles_ptr, (int *)cu_tiles_ptr, num_group, m, n, k, flat_divider,
                           (const int *)x_row_map_ptr, x_num_rows);
      }
    }
  }
}

void group_gemm_1sm_cp_async_fp8_async(void *y_ptr, const void *x_ptr, const void *w_ptr,
                                       const void *seqlens_ptr, const void *cu_seqlens_ptr,
                                       const void *y_scale, void *tmas_ptr, void *tiles_ptr,
                                       void *cu_tiles_ptr, void *task_map_ptr, int num_waves,
                                       int num_group, int m, int n, int k,
                                       int num_seq_per_group_avg, bool update_tma, bool use_pdl,
                                       cudaStream_t stream, const void *x_row_map_ptr = nullptr,
                                       int x_num_rows = 0) {
  using namespace cute;  // NOLINT

  constexpr int kTileM = 128;
  constexpr int kTileK = 128;

  constexpr int kClusterM = 1;
  constexpr int kClusterN = 1;
  constexpr int kClusterK = 1;

  constexpr int kMmaSM = 1;

  // use_pdl = false;

  if (num_seq_per_group_avg <= 16) {
    constexpr int kTileN = 16;
    constexpr int kEpiTileN = 16;
    constexpr int kStageTile = 4;
    constexpr int kStageTMA = 4;
    constexpr int kStage = 8;
    launch_group_gemm_1sm_cp_async_fp8<kTileM, kTileN, kTileK, kStage, kClusterM, kClusterN,
                                       kClusterK, kMmaSM, kEpiTileN, kStageTile, kStageTMA>(
        y_ptr, x_ptr, w_ptr, seqlens_ptr, cu_seqlens_ptr, y_scale, tmas_ptr, tiles_ptr,
        cu_tiles_ptr, num_group, m, n, k, update_tma, use_pdl, stream, x_row_map_ptr, x_num_rows);
  } else if (num_seq_per_group_avg <= 32) {
    constexpr int kTileN = 32;
    constexpr int kEpiTileN = 32;
    constexpr int kStageTile = 4;
    constexpr int kStageTMA = 4;
    constexpr int kStage = 6;
    launch_group_gemm_1sm_cp_async_fp8<kTileM, kTileN, kTileK, kStage, kClusterM, kClusterN,
                                       kClusterK, kMmaSM, kEpiTileN, kStageTile, kStageTMA>(
        y_ptr, x_ptr, w_ptr, seqlens_ptr, cu_seqlens_ptr, y_scale, tmas_ptr, tiles_ptr,
        cu_tiles_ptr, num_group, m, n, k, update_tma, use_pdl, stream, x_row_map_ptr, x_num_rows);
  } else if (num_seq_per_group_avg <= 48) {
    constexpr int kTileN = 48;
    constexpr int kEpiTileN = 48;
    constexpr int kStageTile = 4;
    constexpr int kStageTMA = 4;
    constexpr int kStage = 6;
    launch_group_gemm_1sm_cp_async_fp8<kTileM, kTileN, kTileK, kStage, kClusterM, kClusterN,
                                       kClusterK, kMmaSM, kEpiTileN, kStageTile, kStageTMA>(
        y_ptr, x_ptr, w_ptr, seqlens_ptr, cu_seqlens_ptr, y_scale, tmas_ptr, tiles_ptr,
        cu_tiles_ptr, num_group, m, n, k, update_tma, use_pdl, stream, x_row_map_ptr, x_num_rows);
  } else if (num_seq_per_group_avg <= 64) {
    constexpr int kTileN = 64;
    constexpr int kEpiTileN = 32;
    constexpr int kStageTile = 4;
    constexpr int kStageTMA = 4;
    constexpr int kStage = 6;
    launch_group_gemm_1sm_cp_async_fp8<kTileM, kTileN, kTileK, kStage, kClusterM, kClusterN,
                                       kClusterK, kMmaSM, kEpiTileN, kStageTile, kStageTMA>(
        y_ptr, x_ptr, w_ptr, seqlens_ptr, cu_seqlens_ptr, y_scale, tmas_ptr, tiles_ptr,
        cu_tiles_ptr, num_group, m, n, k, update_tma, use_pdl, stream, x_row_map_ptr, x_num_rows);
  } else {
    constexpr int kTileN = 128;
    constexpr int kEpiTileN = 64;
    constexpr int kStageTile = 4;
    constexpr int kStageTMA = 2;
    constexpr int kStage = 6;
    launch_group_gemm_1sm_cp_async_fp8<kTileM, kTileN, kTileK, kStage, kClusterM, kClusterN,
                                       kClusterK, kMmaSM, kEpiTileN, kStageTile, kStageTMA>(
        y_ptr, x_ptr, w_ptr, seqlens_ptr, cu_seqlens_ptr, y_scale, tmas_ptr, tiles_ptr,
        cu_tiles_ptr, num_group, m, n, k, update_tma, use_pdl, stream, x_row_map_ptr, x_num_rows);
  }
}

void group_gemm_cp_async_fp8_async(void *y_ptr, const void *x_ptr, const void *w_ptr,
                                   const void *seqlens_ptr, const void *cu_seqlens_ptr,
                                   const void *y_scale, void *tmas_ptr, void *tiles_ptr,
                                   void *cu_tiles_ptr, void *task_map_ptr, int num_waves,
                                   int num_group, int m, int n, int k, int num_seq_per_group_avg,
                                   bool update_tma, bool use_pdl, cudaStream_t stream,
                                   const void *x_row_map_ptr, int x_num_rows) {
  group_gemm_1sm_cp_async_fp8_async(y_ptr, x_ptr, w_ptr, seqlens_ptr, cu_seqlens_ptr, y_scale,
                                    tmas_ptr, tiles_ptr, cu_tiles_ptr, task_map_ptr, num_waves,
                                    num_group, m, n, k, num_seq_per_group_avg, update_tma, use_pdl,
                                    stream, x_row_map_ptr, x_num_rows);
}

}  // namespace group_gemm
}  // namespace hpc
