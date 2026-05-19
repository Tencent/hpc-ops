// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include <cub/cub.cuh>

#include "cute/tensor.hpp"
#include "src/group_gemm/sm100/gemm_config.h"
#include "src/group_gemm/sm100/group_gemm.h"
#include "src/group_gemm/sm100/group_gemm_cp_async_fp8.cuh"
#include "src/group_gemm/sm100/group_gemm_cp_async_fp8_act_mul.cuh"

namespace hpc {
namespace group_gemm {

template <typename GemmConfig>
void launch_group_gemm_1sm_cp_async_fp8_act_mul(
    void *y_ptr, const void *x_ptr, const void *w_ptr, const void *seqlens_ptr,
    const void *cu_seqlens_ptr, const void *gate_up_scale_ptr, const void *act_mul_scale_ptr,
    void *tmas_ptr, void *tiles_ptr, void *cu_tiles_ptr, int num_group, int m, int n, int k,
    bool update_tma, bool use_pdl, cudaStream_t stream, const void *x_row_map_ptr = nullptr,
    int x_num_rows = 0) {
  using namespace cute;  // NOLINT

  GemmConfig gemm_config;
  using Tin = typename GemmConfig::Tin;
  using Tout = typename GemmConfig::Tout;

  auto W_gate = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(w_ptr)),
                            make_shape(n / 2, k, num_group), make_stride(k, Int<1>{}, n * k));
  auto W_up = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(w_ptr) + n / 2 * k),
                          make_shape(n / 2, k, num_group), make_stride(k, Int<1>{}, n * k));

  auto X = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(x_ptr)), make_shape(m, k),
                       make_stride(k, Int<1>{}));

  auto Y = make_tensor(make_gmem_ptr(reinterpret_cast<Tout *>(y_ptr)), make_shape(n / 2, m),
                       make_stride(Int<1>{}, n / 2));

  auto gY = make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                        make_shape(Int<GemmConfig::kTileN>{}, Int<GemmConfig::kTileM>{}),
                        make_stride(Int<GemmConfig::kTileM>{}, Int<1>{}));
  auto tiled_mma_for_store =
      make_tiled_mma(MMA_Traits<SM100_MMA_F8F6F4_SS, Tin, Tin, typename GemmConfig::Tacc,
                                cute::C<GemmConfig::kTileN>, cute::C<GemmConfig::kTileM>,
                                cute::integral_constant<UMMA::Major, UMMA::Major::K>,
                                cute::integral_constant<UMMA::Major, UMMA::Major::K>,
                                cute::integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>,
                                cute::integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>>{});
  auto cta_mma_for_store = tiled_mma_for_store.get_slice(0);
  auto tOt_for_store = cta_mma_for_store.make_fragment_C(cta_mma_for_store.partition_C(gY));

  auto [tma_x, tma_gate, tma_y] = gemm_config.get_tma(X, W_gate, Y);
  auto tma_up = make_tma_copy(SM90_TMA_LOAD{}, W_up, typename GemmConfig::CopyBoxW{});
  auto *tma_xy = static_cast<cute::TmaDescriptor *>(tmas_ptr);
  int num_sm = get_sm_count();

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

      cudaLaunchConfig_t launch_config{};
      launch_config.attrs = attribute;
      launch_config.numAttrs = 1;
      launch_config.gridDim = num_group + 1;
      launch_config.blockDim = kThreadPerBlock;
      launch_config.dynamicSmemBytes = 0;
      launch_config.stream = stream;
      auto kernel = kernels::update_grouped_tma<Tin, Tout, decltype(tma_x), decltype(tma_y),
                                                GemmConfig::kTileM, kGroupPerThread,
                                                kThreadPerBlock, kUsePDL>;
      cudaLaunchKernelEx(&launch_config, kernel, td_xy, tma_xy, (const Tin *)x_ptr,
                         (const Tout *)y_ptr, (const int *)seqlens_ptr, (const int *)cu_seqlens_ptr,
                         (int *)tiles_ptr, (int *)cu_tiles_ptr, num_group, m, n / 2, k);
    } else {
      constexpr bool kUsePDL = false;
      kernels::update_grouped_tma<Tin, Tout, decltype(tma_x), decltype(tma_y), GemmConfig::kTileM,
                                  kGroupPerThread, kThreadPerBlock, kUsePDL>
          <<<num_group + 1, kThreadPerBlock, 0, stream>>>(
              td_xy, tma_xy, (const Tin *)x_ptr, (const Tout *)y_ptr, (const int *)seqlens_ptr,
              (const int *)cu_seqlens_ptr, (int *)tiles_ptr, (int *)cu_tiles_ptr, num_group, m,
              n / 2, k);
    }
  }

  // Launch fused kernel.
  {
    int num_tile_n_per_group = (n / 2 + GemmConfig::kTileN - 1) / GemmConfig::kTileN;
    cutlass::FastDivmod flat_divider(num_tile_n_per_group * GemmConfig::kMmaSM);

    int shm_seq = sizeof(int) * (num_group + 1);
    int shm_gemm = gemm_config.get_shm_size();
    int shm_size = shm_gemm + shm_seq;

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
      attribute[1].val.clusterDim.x = GemmConfig::kClusters;
      attribute[1].val.clusterDim.y = 1;
      attribute[1].val.clusterDim.z = 1;

      launch_config.attrs = attribute;
      launch_config.numAttrs = 2;

      constexpr int kTaskLoopPolicy = 1;
      auto kernel = kernels::group_gemm_1sm_cp_async_fp8_act_mul_kernel<
          GemmConfig, decltype(tOt_for_store), decltype(tma_gate), decltype(tma_up),
          decltype(tma_x), decltype(tma_y), kTaskLoopPolicy, kUsePDL>;
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
      cudaLaunchKernelEx(&launch_config, kernel, tma_gate, tma_up, tma_xy, (Tin *)x_ptr,
                         (int *)cu_seqlens_ptr, (int *)seqlens_ptr,
                         (const float *)gate_up_scale_ptr, (const float *)act_mul_scale_ptr,
                         (int *)tiles_ptr, (int *)cu_tiles_ptr, num_group, m, n, k, flat_divider,
                         (const int *)x_row_map_ptr, x_num_rows);
    } else {
      constexpr bool kUsePDL = false;
      cudaLaunchAttribute attribute[1];
      attribute[0].id = cudaLaunchAttributeClusterDimension;
      attribute[0].val.clusterDim.x = GemmConfig::kClusters;
      attribute[0].val.clusterDim.y = 1;
      attribute[0].val.clusterDim.z = 1;

      launch_config.attrs = attribute;
      launch_config.numAttrs = 1;

      constexpr int kTaskLoopPolicy = 1;
      auto kernel = kernels::group_gemm_1sm_cp_async_fp8_act_mul_kernel<
          GemmConfig, decltype(tOt_for_store), decltype(tma_gate), decltype(tma_up),
          decltype(tma_x), decltype(tma_y), kTaskLoopPolicy, kUsePDL>;
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
      cudaLaunchKernelEx(&launch_config, kernel, tma_gate, tma_up, tma_xy, (Tin *)x_ptr,
                         (int *)cu_seqlens_ptr, (int *)seqlens_ptr,
                         (const float *)gate_up_scale_ptr, (const float *)act_mul_scale_ptr,
                         (int *)tiles_ptr, (int *)cu_tiles_ptr, num_group, m, n, k, flat_divider,
                         (const int *)x_row_map_ptr, x_num_rows);
    }
  }
}

void group_gemm_cp_async_fp8_act_mul_async(void *y_ptr, const void *x_ptr, const void *w_ptr,
                                           const void *seqlens_ptr, const void *cu_seqlens_ptr,
                                           const void *gate_up_scale_ptr,
                                           const void *act_mul_scale_ptr, void *tmas_ptr,
                                           void *tiles_ptr, void *cu_tiles_ptr, int num_group,
                                           int m, int n, int k, int num_seq_per_group_avg,
                                           bool update_tma, bool use_pdl, cudaStream_t stream,
                                           const void *x_row_map_ptr, int x_num_rows) {
  using namespace cute;  // NOLINT

  constexpr GroupGemmFunc kGemmFunc = GroupGemmFunc::GROUP_GEMM_CP_ASYNC_FP8_ACT_MUL;

  auto launch = [&](auto num_seq_per_group_avg_tag) {
    constexpr int kNumSeqPerGroupAvg = decltype(num_seq_per_group_avg_tag)::value;
    using GemmConfig = decltype(get_group_gemm_config<kGemmFunc, kNumSeqPerGroupAvg>());
    launch_group_gemm_1sm_cp_async_fp8_act_mul<GemmConfig>(
        y_ptr, x_ptr, w_ptr, seqlens_ptr, cu_seqlens_ptr, gate_up_scale_ptr, act_mul_scale_ptr,
        tmas_ptr, tiles_ptr, cu_tiles_ptr, num_group, m, n, k, update_tma, use_pdl, stream,
        x_row_map_ptr, x_num_rows);
  };

  if (num_seq_per_group_avg <= 16) {
    launch(std::integral_constant<int, 16>{});
  } else if (num_seq_per_group_avg <= 32) {
    launch(std::integral_constant<int, 32>{});
  } else if (num_seq_per_group_avg <= 48) {
    launch(std::integral_constant<int, 48>{});
  } else if (num_seq_per_group_avg <= 64) {
    launch(std::integral_constant<int, 64>{});
  } else {
    launch(std::integral_constant<int, 128>{});
  }
}

}  // namespace group_gemm
}  // namespace hpc
