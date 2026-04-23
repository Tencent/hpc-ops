// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include <cub/cub.cuh>

#include "cute/tensor.hpp"
#include "src/group_gemm/sm100/config.h"
#include "src/group_gemm/sm100/group_gemm.h"
#include "src/group_gemm/sm100/group_gemm_cp_async_fp8.cuh"
#include "src/group_gemm/sm100/group_gemm_cp_async_fp8_act_mul.cuh"

namespace hpc {
namespace group_gemm {

// ============================================================================
// Fused gate_up GEMM + act_mul_and_quant
// ============================================================================
//
// gate_up_weight has shape [num_expert, 2*intermediate_size, hidden_size].
// We treat it as two logical slices (gate, up) where:
//     gate_base = (const Tin *)w_ptr + 0
//     up_base   = (const Tin *)w_ptr + intermediate_size * k
// Both slices share the SAME per-expert stride = 2 * intermediate_size * k.
//
// Output is fp8, shape [total_num_seq, intermediate_size] (same as down_input
// previously produced by act_mul_and_quant_async).
template <int kTileM, int kTileN, int kTileK, int kStageK, int kClusterM, int kClusterN,
          int kClusterK, int kMmaSM, int kEpiTileN, int kStageTile, int kStageTMA>
void launch_group_gemm_1sm_cp_async_fp8_act_mul(
    void *y_ptr, const void *x_ptr, const void *w_ptr, const void *seqlens_ptr,
    const void *cu_seqlens_ptr, const void *gate_up_scale_ptr, const void *act_mul_scale_ptr,
    void *tmas_ptr, void *tiles_ptr, void *cu_tiles_ptr, int num_group, int m_true, int n_true,
    int k, bool update_tma, bool use_pdl, cudaStream_t stream, const void *x_row_map_ptr = nullptr,
    int x_num_rows = 0) {
  using namespace cute;  // NOLINT

  using Tin = cute::float_e4m3_t;
  using Tout = cute::float_e4m3_t;
  using Tacc = float;

  // The caller passes `n_true` as 2*intermediate_size (the same naming the
  // non-fused gate_up gemm uses, where n_true is gate_up_weight.size(1)).
  // Treat the weight as two I-row slices where I = n_true / 2. The fused
  // output has shape (I, total_num_seq) in column-major.
  int m = n_true / 2;  // intermediate_size (per logical gate/up)
  int n = m_true;      // total_num_seq
  int stride_per_expert = 2 * m * k;

  // Two TMA descriptors: one for gate (base = w_ptr), one for up (base = w_ptr + m*k).
  // Both descriptors see a per-expert tile of shape (m, k) strided by (stride_per_expert).
  auto A_gate =
      make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(w_ptr)), make_shape(m, k, num_group),
                  make_stride(k, Int<1>{}, stride_per_expert));
  auto A_up = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(w_ptr) + m * k),
                          make_shape(m, k, num_group), make_stride(k, Int<1>{}, stride_per_expert));

  auto B = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(x_ptr)), make_shape(n, k),
                       make_stride(k, Int<1>{}));

  // Output CT: fp8, shape (m, n), col-major like the bf16 path.
  auto CT = make_tensor(make_gmem_ptr(reinterpret_cast<Tout *>(y_ptr)), make_shape(m, n),
                        make_stride(Int<1>{}, m));

  constexpr int kSwizzleX = kTileK;
  constexpr int kSwizzleW = kTileK;
  // For fp8 output the col-major stride (= m = intermediate_size/2 bytes)
  // must be a multiple of the output-swizzle size; any non-zero kSwizzleY
  // requires e.g. m % 128 == 0 for SW128 which is too restrictive for the
  // public intermediate_size enum (must only be 16-multiple). INTER atom
  // requires just 16-alignment which matches the public contract.
  constexpr int kSwizzleY = 0;
  constexpr int kSwizzleYInner = 0;

  using GroupGEMMConfig =
      GroupGEMMFp8Config<Tin, Tout, Tacc, kTileM, kTileN, kTileK, kStageK, kClusterM, kClusterN,
                         kClusterK, kMmaSM, kEpiTileN, kStageTile, kStageTMA, kSwizzleX, kSwizzleW,
                         kSwizzleY, kSwizzleYInner>;

  auto tiled_mma = make_tiled_mma(
      MMA_Traits<SM100_MMA_F8F6F4_SS, Tin, Tin, Tacc, cute::C<kTileM>, cute::C<kTileN>,
                 cute::integral_constant<UMMA::Major, UMMA::Major::K>,
                 cute::integral_constant<UMMA::Major, UMMA::Major::K>,
                 cute::integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>,
                 cute::integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>>{});

  GroupGEMMConfig config;
  // Reuse config.get_tma. The returned tma_a is the TMA descriptor for A_gate.
  // For A_up we rebuild a matching TMA descriptor with the same CopyBoxX.
  auto [tma_a_gate, tma_b, tma_dt] = config.get_tma(A_gate, B, CT);
  auto tma_a_up = make_tma_copy(SM90_TMA_LOAD{}, A_up, typename GroupGEMMConfig::CopyBoxX{});

  constexpr int kClusters = GroupGEMMConfig::kClusters;

  auto *tma_xy = static_cast<cute::TmaDescriptor *>(tmas_ptr);
  int num_sm = get_sm_count();

  if (update_tma) {
    // Rebuild per-group TMA descriptors for B (x input) and the fused fp8
    // output. The fused output has width `m = n_true / 2` columns, not
    // `n_true`; passing n_out = m to update_grouped_tma keeps the TMA Y
    // descriptor consistent with the kernel's output shape.
    vec_t<cute::TmaDescriptor, 2> td_xy{
        *tma_b.get_tma_descriptor(),
        *tma_dt.get_tma_descriptor(),
    };

    constexpr int kGroupPerThread = 8;
    constexpr int kThreadPerBlock = 32;

    int n_out = m;  // fused output width

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
      auto kernel = kernels::update_grouped_tma<Tin, Tout, decltype(tma_b), decltype(tma_dt),
                                                kTileN, kGroupPerThread, kThreadPerBlock, kUsePDL>;
      cudaLaunchKernelEx(&launch_config, kernel, td_xy, tma_xy, (const Tin *)x_ptr,
                         (const Tout *)y_ptr, (const int *)seqlens_ptr, (const int *)cu_seqlens_ptr,
                         (int *)tiles_ptr, (int *)cu_tiles_ptr, num_group, m_true, n_out, k);
    } else {
      constexpr bool kUsePDL = false;
      kernels::update_grouped_tma<Tin, Tout, decltype(tma_b), decltype(tma_dt), kTileN,
                                  kGroupPerThread, kThreadPerBlock, kUsePDL>
          <<<num_group + 1, kThreadPerBlock, 0, stream>>>(
              td_xy, tma_xy, (const Tin *)x_ptr, (const Tout *)y_ptr, (const int *)seqlens_ptr,
              (const int *)cu_seqlens_ptr, (int *)tiles_ptr, (int *)cu_tiles_ptr, num_group, m_true,
              n_out, k);
    }
  }

  // Launch fused kernel.
  {
    int num_tile_m_per_group = (m + kTileM - 1) / kTileM;
    cutlass::FastDivmod flat_divider(num_tile_m_per_group * kMmaSM);

    int shm_seq = sizeof(int) * (num_group + 1);
    // SMEM: 2 A buffers (gate+up) + 1 B buffer + 1 output C buffer.
    //   original shm_xw = (SLayoutA + SLayoutW) * sizeof(Tin)
    //   fused shm_xw    = (2*SLayoutA + SLayoutW) * sizeof(Tin)
    //   shm_y unchanged but with Tout = fp8 (1 byte instead of 2).
    using Cfg = GroupGEMMConfig;
    int shm_a_gate = cosize(typename Cfg::SLayoutX{}) * sizeof(Tin);
    int shm_a_up = cosize(typename Cfg::SLayoutX{}) * sizeof(Tin);
    int shm_b = cosize(typename Cfg::SLayoutW{}) * sizeof(Tin);
    int shm_c = cosize(typename Cfg::SLayoutY{}) * sizeof(Tout);
    int shm_size = shm_a_gate + shm_a_up + shm_b + shm_c + shm_seq;

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

      constexpr int kTaskLoopPolicy = 1;
      auto kernel = kernels::group_gemm_1sm_cp_async_fp8_act_mul_kernel<
          decltype(config), decltype(tiled_mma), decltype(tma_a_gate), decltype(tma_b),
          decltype(tma_dt), kTaskLoopPolicy, kUsePDL>;
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
      cudaLaunchKernelEx(&launch_config, kernel, tma_a_gate, tma_a_up, tma_xy, (Tin *)w_ptr,
                         (Tin *)w_ptr + m * k, (Tin *)x_ptr, (int *)cu_seqlens_ptr,
                         (int *)seqlens_ptr, (const float *)gate_up_scale_ptr,
                         (const float *)act_mul_scale_ptr, (int *)tiles_ptr, (int *)cu_tiles_ptr,
                         num_group, m, n, k, flat_divider, (const int *)x_row_map_ptr, x_num_rows);
    } else {
      constexpr bool kUsePDL = false;
      cudaLaunchAttribute attribute[1];
      attribute[0].id = cudaLaunchAttributeClusterDimension;
      attribute[0].val.clusterDim.x = kClusters;
      attribute[0].val.clusterDim.y = 1;
      attribute[0].val.clusterDim.z = 1;

      launch_config.attrs = attribute;
      launch_config.numAttrs = 1;

      constexpr int kTaskLoopPolicy = 1;
      auto kernel = kernels::group_gemm_1sm_cp_async_fp8_act_mul_kernel<
          decltype(config), decltype(tiled_mma), decltype(tma_a_gate), decltype(tma_b),
          decltype(tma_dt), kTaskLoopPolicy, kUsePDL>;
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
      cudaLaunchKernelEx(&launch_config, kernel, tma_a_gate, tma_a_up, tma_xy, (Tin *)w_ptr,
                         (Tin *)w_ptr + m * k, (Tin *)x_ptr, (int *)cu_seqlens_ptr,
                         (int *)seqlens_ptr, (const float *)gate_up_scale_ptr,
                         (const float *)act_mul_scale_ptr, (int *)tiles_ptr, (int *)cu_tiles_ptr,
                         num_group, m, n, k, flat_divider, (const int *)x_row_map_ptr, x_num_rows);
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
  // Tiling mirrors group_gemm_1sm_cp_async_fp8_async. TMEM budget with 2 accumulators
  // per stage is 2 * kStageTile * kTileN <= 512 cols.
  using namespace cute;  // NOLINT
  constexpr int kTileM = 128;
  constexpr int kTileK = 128;
  constexpr int kClusterM = 1;
  constexpr int kClusterN = 1;
  constexpr int kClusterK = 1;
  constexpr int kMmaSM = 1;

  // Fused kernel holds TWO A buffers (gate+up) per stage so kStage is halved
  // relative to the non-fused dispatcher to fit within the 228 KB dynamic
  // SMEM limit.
  if (num_seq_per_group_avg <= 16) {
    constexpr int kTileN = 16;
    constexpr int kEpiTileN = 16;
    constexpr int kStageTile = 4;
    constexpr int kStageTMA = 4;
    constexpr int kStage = 4;
    launch_group_gemm_1sm_cp_async_fp8_act_mul<kTileM, kTileN, kTileK, kStage, kClusterM, kClusterN,
                                               kClusterK, kMmaSM, kEpiTileN, kStageTile, kStageTMA>(
        y_ptr, x_ptr, w_ptr, seqlens_ptr, cu_seqlens_ptr, gate_up_scale_ptr, act_mul_scale_ptr,
        tmas_ptr, tiles_ptr, cu_tiles_ptr, num_group, m, n, k, update_tma, use_pdl, stream,
        x_row_map_ptr, x_num_rows);
  } else if (num_seq_per_group_avg <= 32) {
    constexpr int kTileN = 32;
    constexpr int kEpiTileN = 32;
    constexpr int kStageTile = 4;
    constexpr int kStageTMA = 4;
    constexpr int kStage = 3;
    launch_group_gemm_1sm_cp_async_fp8_act_mul<kTileM, kTileN, kTileK, kStage, kClusterM, kClusterN,
                                               kClusterK, kMmaSM, kEpiTileN, kStageTile, kStageTMA>(
        y_ptr, x_ptr, w_ptr, seqlens_ptr, cu_seqlens_ptr, gate_up_scale_ptr, act_mul_scale_ptr,
        tmas_ptr, tiles_ptr, cu_tiles_ptr, num_group, m, n, k, update_tma, use_pdl, stream,
        x_row_map_ptr, x_num_rows);
  } else if (num_seq_per_group_avg <= 48) {
    constexpr int kTileN = 48;
    constexpr int kEpiTileN = 48;
    constexpr int kStageTile = 4;
    constexpr int kStageTMA = 4;
    constexpr int kStage = 3;
    launch_group_gemm_1sm_cp_async_fp8_act_mul<kTileM, kTileN, kTileK, kStage, kClusterM, kClusterN,
                                               kClusterK, kMmaSM, kEpiTileN, kStageTile, kStageTMA>(
        y_ptr, x_ptr, w_ptr, seqlens_ptr, cu_seqlens_ptr, gate_up_scale_ptr, act_mul_scale_ptr,
        tmas_ptr, tiles_ptr, cu_tiles_ptr, num_group, m, n, k, update_tma, use_pdl, stream,
        x_row_map_ptr, x_num_rows);
  } else if (num_seq_per_group_avg <= 64) {
    constexpr int kTileN = 64;
    constexpr int kEpiTileN = 32;
    constexpr int kStageTile = 4;
    constexpr int kStageTMA = 4;
    constexpr int kStage = 3;
    launch_group_gemm_1sm_cp_async_fp8_act_mul<kTileM, kTileN, kTileK, kStage, kClusterM, kClusterN,
                                               kClusterK, kMmaSM, kEpiTileN, kStageTile, kStageTMA>(
        y_ptr, x_ptr, w_ptr, seqlens_ptr, cu_seqlens_ptr, gate_up_scale_ptr, act_mul_scale_ptr,
        tmas_ptr, tiles_ptr, cu_tiles_ptr, num_group, m, n, k, update_tma, use_pdl, stream,
        x_row_map_ptr, x_num_rows);
  } else {
    constexpr int kTileN = 128;
    constexpr int kEpiTileN = 64;
    constexpr int kStageTile = 2;
    constexpr int kStageTMA = 2;
    constexpr int kStage = 4;
    launch_group_gemm_1sm_cp_async_fp8_act_mul<kTileM, kTileN, kTileK, kStage, kClusterM, kClusterN,
                                               kClusterK, kMmaSM, kEpiTileN, kStageTile, kStageTMA>(
        y_ptr, x_ptr, w_ptr, seqlens_ptr, cu_seqlens_ptr, gate_up_scale_ptr, act_mul_scale_ptr,
        tmas_ptr, tiles_ptr, cu_tiles_ptr, num_group, m, n, k, update_tma, use_pdl, stream,
        x_row_map_ptr, x_num_rows);
  }
}

}  // namespace group_gemm
}  // namespace hpc
