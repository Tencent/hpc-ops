// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include <algorithm>
#include <iostream>

#include "cute/tensor.hpp"
#include "src/attention/decode/sm90/dynamic/smallm_fp8_kv_fp16_pv_compute_dim128_dynamic_splitk_kernels.cuh"
#include "src/attention/decode/smallm_dim128.h"
#include "src/attention/decode/splitk_combine_kernels.cuh"
#include "src/utils/utils.h"

namespace hpc {
namespace attention {
namespace decode {

template <int kTileM>
static constexpr auto mma_selector_qk_fp8_dyn() {
  using namespace cute;  // NOLINT
  if constexpr (kTileM == 8) {
    return SM90_64x8x32_F32E4M3E4M3_SS_TN<>{};
  } else if constexpr (kTileM == 16) {
    return SM90_64x16x32_F32E4M3E4M3_SS_TN<>{};
  } else if constexpr (kTileM == 24) {
    return SM90_64x24x32_F32E4M3E4M3_SS_TN<>{};
  } else if constexpr (kTileM == 32) {
    return SM90_64x32x32_F32E4M3E4M3_SS_TN<>{};
  }
}

template <int kTileM, cute::GMMA::Major kAMajor, cute::GMMA::Major kBMajor>
static constexpr auto mma_selector_sv_f16_dyn() {
  using namespace cute;  // NOLINT
  if constexpr (kTileM == 8) {
    return SM90_64x8x16_F32F16F16_SS<kAMajor, kBMajor>{};
  } else if constexpr (kTileM == 16) {
    return SM90_64x16x16_F32F16F16_SS<kAMajor, kBMajor>{};
  } else if constexpr (kTileM == 24) {
    return SM90_64x24x16_F32F16F16_SS<kAMajor, kBMajor>{};
  } else if constexpr (kTileM == 32) {
    return SM90_64x32x16_F32F16F16_SS<kAMajor, kBMajor>{};
  }
}

// =============================================================================
// Mode 21: Q fp8 per-(token,head) + K/V per-tensor fp8 dynamic-splitk launcher.
// =============================================================================
template <int kTileM, int kTileN, int kTileK, int kTileV, int kBlockSize, int kMaxSplitK>
static void launch_attention_decode_qfp8_kpertensor_vpertensor_fp16_pv_compute_dynamic_inner(
    void *y_ptr, void *splitk_out_ptr, void *lse_ptr, const int *task_map_ptr, int num_total_ctas,
    const void *q_ptr, void *kcache_ptr, void *vcache_ptr, const float *qscale_ptr,
    const float *kscale_ptr, const float *vscale_ptr, const int *block_ids_ptr,
    bool new_kv_included, int num_batch, int num_seq_q, int num_head_q, int num_head_k,
    int num_head_v, int heads_per_group, int num_dim_qk, int num_dim_v, int num_kvcache_blocks,
    int num_seq_max_blocks, int ldQ, int64_t kcache_block_stride, int64_t kcache_token_stride,
    int64_t kcache_head_stride, int64_t vcache_block_stride, int64_t vcache_token_stride,
    int64_t vcache_head_stride, cudaStream_t stream) {
  using namespace cute;  // NOLINT
  constexpr int kStage = 1;
  constexpr int kHeadsPerGroup = 8;
  (void)new_kv_included;

  using Tin = cute::float_e4m3_t;
  using TinPV = cute::half_t;
  using TinKV = cute::float_e4m3_t;
  using Tout = cute::bfloat16_t;

  auto Q = make_tensor(
      make_gmem_ptr(reinterpret_cast<const Tin *>(q_ptr)),
      make_shape(heads_per_group, num_dim_qk, num_head_k, num_seq_q, num_batch),
      make_stride(num_dim_qk, Int<1>{}, heads_per_group * num_dim_qk, ldQ, ldQ * num_seq_q));
  auto K = make_tensor(
      make_gmem_ptr(reinterpret_cast<const TinKV *>(kcache_ptr)),
      make_shape(kBlockSize, num_dim_qk, num_head_k, num_kvcache_blocks),
      make_stride(kcache_token_stride, Int<1>{}, kcache_head_stride, kcache_block_stride));
  auto V = make_tensor(
      make_gmem_ptr(reinterpret_cast<const TinKV *>(vcache_ptr)),
      make_shape(num_dim_v, kBlockSize, num_head_v, num_kvcache_blocks),
      make_stride(Int<1>{}, vcache_token_stride, vcache_head_stride, vcache_block_stride));
  auto splitY = make_tensor(
      make_gmem_ptr(reinterpret_cast<float *>(splitk_out_ptr)),
      make_shape(num_dim_v, heads_per_group, num_head_k, num_seq_q, kMaxSplitK, num_batch),
      make_stride(Int<1>{}, num_dim_v, heads_per_group * num_dim_v, num_dim_v * num_head_q,
                  num_dim_v * num_head_q * num_seq_q,
                  num_dim_v * num_head_q * num_seq_q * kMaxSplitK));

  auto slayout_q =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{}, make_shape(Int<kTileM>{}, Int<kTileK>{}));
  auto slayout_k = tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{},
                                 make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{}));
  auto slayout_p =
      tile_to_shape(GMMA::Layout_MN_SW128_Atom<TinPV>{}, make_shape(Int<kTileN>{}, Int<kTileM>{}));
  auto slayout_s =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<TinPV>{}, make_shape(Int<kTileM>{}, Int<kTileN>{}));
  auto slayout_v = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TinPV>{},
                                 make_shape(Int<kTileV>{}, Int<kTileN>{}, Int<kStage>{}));
  auto slayout_splity = tile_to_shape(GMMA::Layout_MN_SW128_Atom<float>{},
                                      make_shape(Int<kTileV>{}, Int<kTileM>{}, Int<1>{}));
  auto slayout_v_fp8 = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TinKV>{},
                                     make_shape(Int<kTileV>{}, Int<kTileN>{}, Int<kStage>{}));

  auto tma_copy_layout_q = tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{},
                                         make_shape(Int<kHeadsPerGroup>{}, Int<kTileK>{}));
  auto tma_copy_layout_k = tile_to_shape(GMMA::Layout_K_SW128_Atom<TinKV>{},
                                         make_shape(Int<kBlockSize>{}, Int<kTileK>{}));
  auto tma_copy_layout_v = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TinKV>{},
                                         make_shape(Int<kTileV>{}, Int<kBlockSize>{}));
  auto tma_copy_layout_splity = tile_to_shape(GMMA::Layout_MN_SW128_Atom<float>{},
                                              make_shape(Int<kTileV>{}, Int<kHeadsPerGroup>{}));

  auto tma_q = make_tma_copy(SM90_TMA_LOAD{}, Q, tma_copy_layout_q);
  auto tma_k = make_tma_copy(SM90_TMA_LOAD{}, K, tma_copy_layout_k);
  auto tma_v = make_tma_copy(SM90_TMA_LOAD{}, V, tma_copy_layout_v);
  auto tma_splity = make_tma_copy(SM90_TMA_STORE{}, splitY, tma_copy_layout_splity);

  auto qk_mma_atom = mma_selector_qk_fp8_dyn<kTileM>();
  auto sv_mma_atom = mma_selector_sv_f16_dyn<kTileM, GMMA::Major::MN, GMMA::Major::K>();

  using TiledMmaQK = decltype(make_tiled_mma(qk_mma_atom));
  using TiledMmaSV = decltype(make_tiled_mma(sv_mma_atom));

  constexpr int kWarpsPerWrapGroup = 4;
  int shm_qk = (cosize(slayout_q) + cosize(slayout_k)) * sizeof(Tin);
  int shm_pv = (cosize(slayout_p) + cosize(slayout_v)) * sizeof(TinPV) +
               sizeof(float) * kTileM * kWarpsPerWrapGroup;
  int shm_v_fp8 = static_cast<int>(cosize(slayout_v_fp8) * sizeof(TinKV));
  int shm_blk_ids = sizeof(int) * num_seq_max_blocks;
  int shm_splity = cosize(slayout_splity) * sizeof(float);
  int shm_size = std::max(shm_qk + shm_pv + shm_v_fp8 + shm_blk_ids, shm_splity);

  constexpr float kLog2e = 1.4426950408889634f;
  float one_over_dk_log2e = 1.f / sqrtf(float(num_dim_qk)) * kLog2e;

  dim3 grid(num_total_ctas);
  dim3 block(size(TiledMmaQK{}) + 32);

  auto kernel =
      kernels::attention_decode_qfp8_kpertensor_vpertensor_fp16_pv_compute_dynamic_splitk_kernel<
          Tout, Tin, kTileM, kTileN, kTileK, kTileV, kHeadsPerGroup, TiledMmaQK, TiledMmaSV,
          decltype(tma_q), decltype(tma_k), decltype(tma_v), decltype(tma_splity),
          decltype(slayout_q), decltype(slayout_k), decltype(slayout_p), decltype(slayout_s),
          decltype(slayout_v), decltype(slayout_v_fp8), decltype(slayout_splity), kBlockSize,
          kStage, kMaxSplitK>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

  int pad_heads_per_group = ((heads_per_group + 7) / 8) * 8;

  cudaLaunchConfig_t attn_config;
  memset(&attn_config, 0, sizeof(attn_config));
  attn_config.gridDim = grid;
  attn_config.blockDim = block;
  attn_config.dynamicSmemBytes = shm_size;
  cudaLaunchAttribute attn_attrs[1];
  attn_attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attn_attrs[0].val.programmaticStreamSerializationAllowed = 1;
  attn_config.numAttrs = 1;
  attn_config.attrs = attn_attrs;
  attn_config.stream = stream;

  cudaLaunchKernelEx(&attn_config, kernel, tma_q, tma_k, tma_v, tma_splity,
                     reinterpret_cast<float *>(splitk_out_ptr), reinterpret_cast<float *>(lse_ptr),
                     task_map_ptr, block_ids_ptr, qscale_ptr, kscale_ptr, vscale_ptr, num_batch,
                     num_seq_q, num_dim_qk, num_dim_v, num_head_q, num_head_k, num_head_v,
                     heads_per_group, pad_heads_per_group, num_kvcache_blocks, num_seq_max_blocks,
                     one_over_dk_log2e);

  constexpr int kCombineWarps = 4;
  dim3 combine_grid(num_head_q / kCombineWarps, num_seq_q, num_batch);
  dim3 combine_block(kCombineWarps * 32);
  cutlass::FastDivmod hpg_divider(heads_per_group);

  auto combine_kernel =
      kernels::attention_decode_dynamic_splitk_combine_kernel<__nv_bfloat16, kCombineWarps>;

  cudaLaunchConfig_t combine_config;
  memset(&combine_config, 0, sizeof(combine_config));
  combine_config.gridDim = combine_grid;
  combine_config.blockDim = combine_block;
  combine_config.dynamicSmemBytes = sizeof(float) * kCombineWarps * kMaxSplitK;
  cudaLaunchAttribute combine_attrs[1];
  combine_attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  combine_attrs[0].val.programmaticStreamSerializationAllowed = 1;
  combine_config.numAttrs = 1;
  combine_config.attrs = combine_attrs;
  combine_config.stream = stream;

  cudaLaunchKernelEx(&combine_config, combine_kernel, reinterpret_cast<__nv_bfloat16 *>(y_ptr),
                     reinterpret_cast<const float *>(splitk_out_ptr),
                     reinterpret_cast<const float *>(lse_ptr), task_map_ptr, num_total_ctas,
                     num_batch, num_seq_q, num_head_q, num_head_k, pad_heads_per_group, num_dim_v,
                     kMaxSplitK, hpg_divider);
}

// =============================================================================
// Mode 20: Q fp8 per-(token,head) + K per-(K-tok, head) + V per-head launcher.
// =============================================================================
template <int kTileM, int kTileN, int kTileK, int kTileV, int kBlockSize, int kMaxSplitK>
static void launch_attention_decode_qfp8_kpertoken_perhead_vperhead_fp16_pv_compute_dynamic_inner(
    void *y_ptr, void *splitk_out_ptr, void *lse_ptr, const int *task_map_ptr, int num_total_ctas,
    const void *q_ptr, void *kcache_ptr, void *vcache_ptr, const float *qscale_ptr,
    const float *kscale_ptr, const float *vscale_ptr, const int *block_ids_ptr,
    bool new_kv_included, int num_batch, int num_seq_q, int num_head_q, int num_head_k,
    int num_head_v, int heads_per_group, int num_dim_qk, int num_dim_v, int num_kvcache_blocks,
    int num_seq_max_blocks, int ldQ, int64_t kcache_block_stride, int64_t kcache_token_stride,
    int64_t kcache_head_stride, int64_t vcache_block_stride, int64_t vcache_token_stride,
    int64_t vcache_head_stride, cudaStream_t stream) {
  using namespace cute;  // NOLINT
  constexpr int kStage = 1;
  constexpr int kHeadsPerGroup = 8;
  constexpr int kScaleByteSize = sizeof(float);
  constexpr int kTileScale = kTileK / kScaleByteSize;
  (void)new_kv_included;

  using Tin = cute::float_e4m3_t;
  using TinPV = cute::half_t;
  using TinKV = cute::float_e4m3_t;
  using Tout = cute::bfloat16_t;

  auto Q = make_tensor(
      make_gmem_ptr(reinterpret_cast<const Tin *>(q_ptr)),
      make_shape(heads_per_group, num_dim_qk, num_head_k, num_seq_q, num_batch),
      make_stride(num_dim_qk, Int<1>{}, heads_per_group * num_dim_qk, ldQ, ldQ * num_seq_q));
  auto K = make_tensor(
      make_gmem_ptr(reinterpret_cast<const TinKV *>(kcache_ptr)),
      make_shape(kBlockSize, num_dim_qk, num_head_k, num_kvcache_blocks),
      make_stride(kcache_token_stride, Int<1>{}, kcache_head_stride, kcache_block_stride));
  auto V = make_tensor(
      make_gmem_ptr(reinterpret_cast<const TinKV *>(vcache_ptr)),
      make_shape(num_dim_v, kBlockSize, num_head_v, num_kvcache_blocks),
      make_stride(Int<1>{}, vcache_token_stride, vcache_head_stride, vcache_block_stride));
  auto splitY = make_tensor(
      make_gmem_ptr(reinterpret_cast<float *>(splitk_out_ptr)),
      make_shape(num_dim_v, heads_per_group, num_head_k, num_seq_q, kMaxSplitK, num_batch),
      make_stride(Int<1>{}, num_dim_v, heads_per_group * num_dim_v, num_dim_v * num_head_q,
                  num_dim_v * num_head_q * num_seq_q,
                  num_dim_v * num_head_q * num_seq_q * kMaxSplitK));

  int num_dim_scale = num_dim_qk / kScaleByteSize;
  auto KS = make_tensor(
      make_gmem_ptr(reinterpret_cast<const float *>(kscale_ptr)),
      make_shape(kBlockSize / num_dim_scale, num_dim_scale, num_head_k, num_kvcache_blocks),
      make_stride(kcache_token_stride / kScaleByteSize, Int<1>{},
                  kcache_head_stride / kScaleByteSize, kcache_block_stride / kScaleByteSize));

  auto slayout_q =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{}, make_shape(Int<kTileM>{}, Int<kTileK>{}));
  auto slayout_k = tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{},
                                 make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{}));
  auto slayout_p =
      tile_to_shape(GMMA::Layout_MN_SW128_Atom<TinPV>{}, make_shape(Int<kTileN>{}, Int<kTileM>{}));
  auto slayout_s =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<TinPV>{}, make_shape(Int<kTileM>{}, Int<kTileN>{}));
  auto slayout_v = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TinPV>{},
                                 make_shape(Int<kTileV>{}, Int<kTileN>{}, Int<kStage>{}));
  auto slayout_splity = tile_to_shape(GMMA::Layout_MN_SW128_Atom<float>{},
                                      make_shape(Int<kTileV>{}, Int<kTileM>{}, Int<1>{}));
  auto slayout_v_fp8 = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TinKV>{},
                                     make_shape(Int<kTileV>{}, Int<kTileN>{}, Int<kStage>{}));
  auto slayout_ks_c =
      make_layout(make_shape(Int<kTileN / kTileScale>{}, Int<kTileScale>{}, Int<kStage>{}),
                  make_stride(Int<kTileScale>{}, Int<1>{}, Int<kTileN>{}));

  auto tma_copy_layout_q = tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{},
                                         make_shape(Int<kHeadsPerGroup>{}, Int<kTileK>{}));
  auto tma_copy_layout_k = tile_to_shape(GMMA::Layout_K_SW128_Atom<TinKV>{},
                                         make_shape(Int<kBlockSize>{}, Int<kTileK>{}));
  auto tma_copy_layout_v = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TinKV>{},
                                         make_shape(Int<kTileV>{}, Int<kBlockSize>{}));
  auto tma_copy_layout_splity = tile_to_shape(GMMA::Layout_MN_SW128_Atom<float>{},
                                              make_shape(Int<kTileV>{}, Int<kHeadsPerGroup>{}));
  auto tma_copy_layout_ks =
      make_layout(make_shape(Int<kBlockSize / kTileScale>{}, Int<kTileScale>{}),
                  make_stride(Int<kTileScale>{}, Int<1>{}));

  auto tma_q = make_tma_copy(SM90_TMA_LOAD{}, Q, tma_copy_layout_q);
  auto tma_k = make_tma_copy(SM90_TMA_LOAD{}, K, tma_copy_layout_k);
  auto tma_v = make_tma_copy(SM90_TMA_LOAD{}, V, tma_copy_layout_v);
  auto tma_splity = make_tma_copy(SM90_TMA_STORE{}, splitY, tma_copy_layout_splity);
  auto tma_ks = make_tma_copy(SM90_TMA_LOAD{}, KS, tma_copy_layout_ks);

  auto qk_mma_atom = mma_selector_qk_fp8_dyn<kTileM>();
  auto sv_mma_atom = mma_selector_sv_f16_dyn<kTileM, GMMA::Major::MN, GMMA::Major::K>();

  using TiledMmaQK = decltype(make_tiled_mma(qk_mma_atom));
  using TiledMmaSV = decltype(make_tiled_mma(sv_mma_atom));

  constexpr int kWarpsPerWrapGroup = 4;
  int shm_qk = (cosize(slayout_q) + cosize(slayout_k)) * sizeof(Tin);
  int shm_pv = (cosize(slayout_p) + cosize(slayout_v)) * sizeof(TinPV) +
               sizeof(float) * kTileM * kWarpsPerWrapGroup;
  int shm_v_fp8 = static_cast<int>(cosize(slayout_v_fp8) * sizeof(TinKV));
  int shm_ks = static_cast<int>(cosize(slayout_ks_c) * sizeof(float));
  int shm_blk_ids = sizeof(int) * num_seq_max_blocks;
  int shm_splity = cosize(slayout_splity) * sizeof(float);
  int shm_size = std::max(shm_qk + shm_pv + shm_v_fp8 + shm_ks + shm_blk_ids, shm_splity);

  constexpr float kLog2e = 1.4426950408889634f;
  float one_over_dk_log2e = 1.f / sqrtf(float(num_dim_qk)) * kLog2e;

  dim3 grid(num_total_ctas);
  dim3 block(size(TiledMmaQK{}) + 32);

  auto kernel = kernels::
      attention_decode_qfp8_kpertoken_perhead_vperhead_fp16_pv_compute_dynamic_splitk_kernel<
          Tout, Tin, kTileM, kTileN, kTileK, kTileV, kHeadsPerGroup, TiledMmaQK, TiledMmaSV,
          decltype(tma_q), decltype(tma_k), decltype(tma_v), decltype(tma_splity), decltype(tma_ks),
          decltype(slayout_q), decltype(slayout_k), decltype(slayout_p), decltype(slayout_s),
          decltype(slayout_v), decltype(slayout_v_fp8), decltype(slayout_ks_c),
          decltype(slayout_splity), kBlockSize, kStage, kMaxSplitK>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

  int pad_heads_per_group = ((heads_per_group + 7) / 8) * 8;

  cudaLaunchConfig_t attn_config;
  memset(&attn_config, 0, sizeof(attn_config));
  attn_config.gridDim = grid;
  attn_config.blockDim = block;
  attn_config.dynamicSmemBytes = shm_size;
  cudaLaunchAttribute attn_attrs[1];
  attn_attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attn_attrs[0].val.programmaticStreamSerializationAllowed = 1;
  attn_config.numAttrs = 1;
  attn_config.attrs = attn_attrs;
  attn_config.stream = stream;

  cudaLaunchKernelEx(&attn_config, kernel, tma_q, tma_k, tma_v, tma_splity, tma_ks,
                     reinterpret_cast<float *>(splitk_out_ptr), reinterpret_cast<float *>(lse_ptr),
                     task_map_ptr, block_ids_ptr, qscale_ptr, kscale_ptr, vscale_ptr, num_batch,
                     num_seq_q, num_dim_qk, num_dim_v, num_head_q, num_head_k, num_head_v,
                     heads_per_group, pad_heads_per_group, num_kvcache_blocks, num_seq_max_blocks,
                     one_over_dk_log2e);

  constexpr int kCombineWarps = 4;
  dim3 combine_grid(num_head_q / kCombineWarps, num_seq_q, num_batch);
  dim3 combine_block(kCombineWarps * 32);
  cutlass::FastDivmod hpg_divider(heads_per_group);

  auto combine_kernel =
      kernels::attention_decode_dynamic_splitk_combine_kernel<__nv_bfloat16, kCombineWarps>;

  cudaLaunchConfig_t combine_config;
  memset(&combine_config, 0, sizeof(combine_config));
  combine_config.gridDim = combine_grid;
  combine_config.blockDim = combine_block;
  combine_config.dynamicSmemBytes = sizeof(float) * kCombineWarps * kMaxSplitK;
  cudaLaunchAttribute combine_attrs[1];
  combine_attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  combine_attrs[0].val.programmaticStreamSerializationAllowed = 1;
  combine_config.numAttrs = 1;
  combine_config.attrs = combine_attrs;
  combine_config.stream = stream;

  cudaLaunchKernelEx(&combine_config, combine_kernel, reinterpret_cast<__nv_bfloat16 *>(y_ptr),
                     reinterpret_cast<const float *>(splitk_out_ptr),
                     reinterpret_cast<const float *>(lse_ptr), task_map_ptr, num_total_ctas,
                     num_batch, num_seq_q, num_head_q, num_head_k, pad_heads_per_group, num_dim_v,
                     kMaxSplitK, hpg_divider);
}

#define DECODE_DYN_FP16PV_DISPATCH(launch_call)                              \
  auto dispatch_block_size = [&](auto splitk_tag, auto tilem_tag) {          \
    if (block_size == 32) {                                                  \
      launch_call(tilem_tag, std::integral_constant<int, 32>{}, splitk_tag); \
    } else if (block_size == 64) {                                           \
      launch_call(tilem_tag, std::integral_constant<int, 64>{}, splitk_tag); \
    }                                                                        \
  };                                                                         \
  auto dispatch_mtp = [&](auto splitk_tag) {                                 \
    if (num_seq_q == 1) {                                                    \
      dispatch_block_size(splitk_tag, std::integral_constant<int, 8>{});     \
    } else if (num_seq_q == 2) {                                             \
      dispatch_block_size(splitk_tag, std::integral_constant<int, 16>{});    \
    } else if (num_seq_q == 3) {                                             \
      dispatch_block_size(splitk_tag, std::integral_constant<int, 24>{});    \
    } else if (num_seq_q == 4) {                                             \
      dispatch_block_size(splitk_tag, std::integral_constant<int, 32>{});    \
    }                                                                        \
  };                                                                         \
  if (splitk == 78 * 4) {                                                    \
    dispatch_mtp(std::integral_constant<int, 78 * 4>{});                     \
  } else if (splitk == 78 * 3) {                                             \
    dispatch_mtp(std::integral_constant<int, 78 * 3>{});                     \
  } else if (splitk == 78 * 2) {                                             \
    dispatch_mtp(std::integral_constant<int, 78 * 2>{});                     \
  } else {                                                                   \
    dispatch_mtp(std::integral_constant<int, 64>{});                         \
  }

bool attention_decode_qfp8_kpertensor_vpertensor_fp16_pv_compute_dynamic_async(
    void *y_ptr, void *lse_ptr, void *splitk_out_ptr, const int *task_map_ptr, int num_total_ctas,
    const void *q_ptr, void *kcache_ptr, void *vcache_ptr, const float *qscale_ptr,
    const float *kscale_ptr, const float *vscale_ptr, const int *block_ids_ptr,
    bool new_kv_included, int splitk, int num_batch, int num_seq_q, int num_head_q, int num_head_k,
    int num_head_v, int num_dim_qk, int num_dim_v, int num_kvcache_blocks, int block_size,
    int num_seq_max_blocks, int ldQ, int64_t kcache_block_stride, int64_t kcache_token_stride,
    int64_t kcache_head_stride, int64_t vcache_block_stride, int64_t vcache_token_stride,
    int64_t vcache_head_stride, cudaStream_t stream) {
  constexpr int kTileN = 64;
  constexpr int kTileK = 128;
  constexpr int kTileV = 128;
  if (num_dim_qk != kTileK || num_dim_v != kTileV || (block_size != 32 && block_size != 64)) {
    return false;
  }
  int heads_per_group = num_head_q / num_head_k;
  if (heads_per_group != 8 && heads_per_group != 4) {
    return false;
  }

  auto launch = [&](auto tilem_tag, auto block_size_tag, auto splitk_tag) {
    constexpr int kTileM = decltype(tilem_tag)::value;
    constexpr int kBlockSize = decltype(block_size_tag)::value;
    constexpr int kMaxSplitK = decltype(splitk_tag)::value;
    launch_attention_decode_qfp8_kpertensor_vpertensor_fp16_pv_compute_dynamic_inner<
        kTileM, kTileN, kTileK, kTileV, kBlockSize, kMaxSplitK>(
        y_ptr, splitk_out_ptr, lse_ptr, task_map_ptr, num_total_ctas, q_ptr, kcache_ptr, vcache_ptr,
        qscale_ptr, kscale_ptr, vscale_ptr, block_ids_ptr, new_kv_included, num_batch, num_seq_q,
        num_head_q, num_head_k, num_head_v, heads_per_group, num_dim_qk, num_dim_v,
        num_kvcache_blocks, num_seq_max_blocks, ldQ, kcache_block_stride, kcache_token_stride,
        kcache_head_stride, vcache_block_stride, vcache_token_stride, vcache_head_stride, stream);
  };
  DECODE_DYN_FP16PV_DISPATCH(launch)
  return true;
}

bool attention_decode_qfp8_kpertoken_perhead_vperhead_fp16_pv_compute_dynamic_async(
    void *y_ptr, void *lse_ptr, void *splitk_out_ptr, const int *task_map_ptr, int num_total_ctas,
    const void *q_ptr, void *kcache_ptr, void *vcache_ptr, const float *qscale_ptr,
    const float *kscale_ptr, const float *vscale_ptr, const int *block_ids_ptr,
    bool new_kv_included, int splitk, int num_batch, int num_seq_q, int num_head_q, int num_head_k,
    int num_head_v, int num_dim_qk, int num_dim_v, int num_kvcache_blocks, int block_size,
    int num_seq_max_blocks, int ldQ, int64_t kcache_block_stride, int64_t kcache_token_stride,
    int64_t kcache_head_stride, int64_t vcache_block_stride, int64_t vcache_token_stride,
    int64_t vcache_head_stride, cudaStream_t stream) {
  constexpr int kTileN = 64;
  constexpr int kTileK = 128;
  constexpr int kTileV = 128;
  if (num_dim_qk != kTileK || num_dim_v != kTileV || (block_size != 32 && block_size != 64)) {
    return false;
  }
  int heads_per_group = num_head_q / num_head_k;
  if (heads_per_group != 8 && heads_per_group != 4) {
    return false;
  }

  auto launch = [&](auto tilem_tag, auto block_size_tag, auto splitk_tag) {
    constexpr int kTileM = decltype(tilem_tag)::value;
    constexpr int kBlockSize = decltype(block_size_tag)::value;
    constexpr int kMaxSplitK = decltype(splitk_tag)::value;
    launch_attention_decode_qfp8_kpertoken_perhead_vperhead_fp16_pv_compute_dynamic_inner<
        kTileM, kTileN, kTileK, kTileV, kBlockSize, kMaxSplitK>(
        y_ptr, splitk_out_ptr, lse_ptr, task_map_ptr, num_total_ctas, q_ptr, kcache_ptr, vcache_ptr,
        qscale_ptr, kscale_ptr, vscale_ptr, block_ids_ptr, new_kv_included, num_batch, num_seq_q,
        num_head_q, num_head_k, num_head_v, heads_per_group, num_dim_qk, num_dim_v,
        num_kvcache_blocks, num_seq_max_blocks, ldQ, kcache_block_stride, kcache_token_stride,
        kcache_head_stride, vcache_block_stride, vcache_token_stride, vcache_head_stride, stream);
  };
  DECODE_DYN_FP16PV_DISPATCH(launch)
  return true;
}

#undef DECODE_DYN_FP16PV_DISPATCH

}  // namespace decode
}  // namespace attention
}  // namespace hpc
