// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include <algorithm>
#include <iostream>

#include "cute/tensor.hpp"
#include "src/attention/decode/sm90/static/smallm_fp8_kv_fp16_pv_compute_dim128_static_splitk_kernels.cuh"
#include "src/attention/decode/smallm_dim128.h"
#include "src/attention/decode/splitk_combine_kernels.cuh"

namespace hpc {
namespace attention {
namespace decode {

template <int kTileM>
static constexpr auto mma_selector_qk_fp8_modes() {
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
static constexpr auto mma_selector_sv_f16_modes() {
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
// Mode 20: Q fp8 per-(token,head) + K per-(K-tok, head) + V per-head launcher.
// =============================================================================
template <int kTileM, int kTileN, int kTileK, int kTileV, int kBlockSize, int kSplitK,
          int kSplitMinLen>
void launch_attention_decode_qfp8_kpertoken_perhead_vperhead_fp16_pv_compute_smallm_splitk_inner(  // NOLINT(whitespace/line_length)
    void *y_ptr, void *lse_ptr, void *splitk_out_ptr, const void *q_ptr, void *kcache_ptr,
    void *vcache_ptr, const float *qscale_ptr, const float *kscale_ptr, const float *vscale_ptr,
    const int *block_ids_ptr, const int *num_seq_kvcache_ptr, int *split_flag_ptr,
    bool new_kv_included, int num_batch, int num_seq_q, int num_head_q, int num_head_k,
    int num_head_v, int heads_per_group, int num_dim_qk, int num_dim_v, int num_kvcache_blocks,
    int block_size, int num_seq_max_blocks, int qscale_pad_stride, int ldY, int ldQ,
    int64_t kcache_block_stride, int64_t kcache_token_stride, int64_t kcache_head_stride,
    int64_t vcache_block_stride, int64_t vcache_token_stride, int64_t vcache_head_stride,
    cudaStream_t stream) {
  using namespace cute;  // NOLINT
  constexpr int kStage = 1;
  constexpr int kHeadsPerGroup = 8;
  (void)qscale_pad_stride;

  constexpr int kScaleByteSize = sizeof(float);
  constexpr int kTileScale = kTileK / kScaleByteSize;

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

  auto Y = make_tensor(
      make_gmem_ptr(reinterpret_cast<const Tout *>(y_ptr)),
      make_shape(num_dim_v, heads_per_group, num_head_k, num_seq_q, num_batch),
      make_stride(Int<1>{}, num_dim_v, heads_per_group * num_dim_v, ldY, ldY * num_seq_q));

  auto splitY =
      make_tensor(make_gmem_ptr(reinterpret_cast<float *>(splitk_out_ptr)),
                  make_shape(num_dim_v, heads_per_group, num_head_k, num_seq_q, kSplitK, num_batch),
                  make_stride(Int<1>{}, num_dim_v, heads_per_group * num_dim_v,
                              num_dim_v * num_head_q, num_dim_v * num_head_q * num_seq_q,
                              num_dim_v * num_head_q * num_seq_q * kSplitK));

  int num_dim_scale = num_dim_qk / kScaleByteSize;
  auto KS = make_tensor(
      make_gmem_ptr(reinterpret_cast<const float *>(kscale_ptr)),
      make_shape(kBlockSize / num_dim_scale, num_dim_scale, num_head_k, num_kvcache_blocks),
      make_stride(kcache_token_stride / kScaleByteSize, Int<1>{},
                  kcache_head_stride / kScaleByteSize, kcache_block_stride / kScaleByteSize));

  auto slayout_q =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{}, make_shape(Int<kTileM>{}, Int<kTileK>{}));
  // fp8 K stage (WGMMA operand directly).
  auto slayout_k = tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{},
                                 make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{}));
  auto slayout_p =
      tile_to_shape(GMMA::Layout_MN_SW128_Atom<TinPV>{}, make_shape(Int<kTileN>{}, Int<kTileM>{}));
  auto slayout_s =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<TinPV>{}, make_shape(Int<kTileM>{}, Int<kTileN>{}));
  auto slayout_v = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TinPV>{},
                                 make_shape(Int<kTileV>{}, Int<kTileN>{}, Int<kStage>{}));
  auto slayout_y = tile_to_shape(GMMA::Layout_MN_SW128_Atom<Tout>{},
                                 make_shape(Int<kTileV>{}, Int<kTileM>{}, Int<1>{}));
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
  auto tma_copy_layout_y = tile_to_shape(GMMA::Layout_MN_SW128_Atom<Tout>{},
                                         make_shape(Int<kTileV>{}, Int<kHeadsPerGroup>{}));
  auto tma_copy_layout_splity = tile_to_shape(GMMA::Layout_MN_SW128_Atom<float>{},
                                              make_shape(Int<kTileV>{}, Int<kHeadsPerGroup>{}));
  auto tma_copy_layout_ks =
      make_layout(make_shape(Int<kBlockSize / kTileScale>{}, Int<kTileScale>{}),
                  make_stride(Int<kTileScale>{}, Int<1>{}));

  auto tma_q = make_tma_copy(SM90_TMA_LOAD{}, Q, tma_copy_layout_q);
  auto tma_k = make_tma_copy(SM90_TMA_LOAD{}, K, tma_copy_layout_k);
  auto tma_v = make_tma_copy(SM90_TMA_LOAD{}, V, tma_copy_layout_v);
  auto tma_y = make_tma_copy(SM90_TMA_STORE{}, Y, tma_copy_layout_y);
  auto tma_splity = make_tma_copy(SM90_TMA_STORE{}, splitY, tma_copy_layout_splity);
  auto tma_ks = make_tma_copy(SM90_TMA_LOAD{}, KS, tma_copy_layout_ks);

  auto qk_mma_atom = mma_selector_qk_fp8_modes<kTileM>();
  auto sv_mma_atom = mma_selector_sv_f16_modes<kTileM, GMMA::Major::MN, GMMA::Major::K>();

  using TiledMmaQK = decltype(make_tiled_mma(qk_mma_atom));
  using TiledMmaSV = decltype(make_tiled_mma(sv_mma_atom));

  dim3 block(size(TiledMmaQK{}) + 32);
  dim3 grid(num_head_k, num_batch, kSplitK);

  constexpr int kWarpsPerWrapGroup = 4;
  // Layout: fp8 Q | fp8 K stage | fp16 V | fp16 P | shm_max | fp8 V stage |
  //         fp32 KS | shm_kvblk_ids.
  int shm_qk = (cosize(slayout_q) + cosize(slayout_k)) * sizeof(Tin);
  int shm_pv = (cosize(slayout_p) + cosize(slayout_v)) * sizeof(TinPV) +
               sizeof(float) * kTileM * kWarpsPerWrapGroup;
  int shm_v_fp8 = static_cast<int>(cosize(slayout_v_fp8) * sizeof(TinKV));
  int shm_ks = static_cast<int>(cosize(slayout_ks_c) * sizeof(float));
  int shm_blk_ids = sizeof(int) * num_seq_max_blocks;
  int shm_y = std::max(cosize(slayout_y) * sizeof(Tout), cosize(slayout_splity) * sizeof(float));
  int shm_size = std::max(shm_qk + shm_pv + shm_v_fp8 + shm_ks + shm_blk_ids, shm_y);

  constexpr float kLog2e = 1.4426950408889634f;
  float one_over_dk_log2e = 1.f / sqrtf(float(num_dim_qk)) * kLog2e;

  auto kernel = kernels::
      attention_decode_qfp8_kpertoken_perhead_vperhead_fp16_pv_compute_multistage_ws_smallm_splitk_kernel<  // NOLINT(whitespace/line_length)
          Tout, Tin, kTileM, kTileN, kTileK, kTileV, kHeadsPerGroup, TiledMmaQK, TiledMmaSV,
          decltype(tma_q), decltype(tma_k), decltype(tma_v), decltype(tma_y), decltype(tma_splity),
          decltype(tma_ks), decltype(slayout_q), decltype(slayout_k), decltype(slayout_p),
          decltype(slayout_s), decltype(slayout_v), decltype(slayout_v_fp8), decltype(slayout_ks_c),
          decltype(slayout_y), decltype(slayout_splity), kBlockSize, kStage, kSplitK, kSplitMinLen>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

  int pad_heads_per_group = ((heads_per_group + 7) / 8) * 8;

  kernel<<<grid, block, shm_size, stream>>>(
      tma_q, tma_k, tma_v, tma_y, tma_splity, tma_ks, reinterpret_cast<Tout *>(y_ptr),
      reinterpret_cast<float *>(splitk_out_ptr), reinterpret_cast<float *>(lse_ptr), block_ids_ptr,
      num_seq_kvcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr, split_flag_ptr, new_kv_included,
      num_batch, num_seq_q, num_dim_qk, num_dim_v, num_head_q, num_head_k, num_head_v,
      heads_per_group, pad_heads_per_group, num_kvcache_blocks, num_seq_max_blocks,
      one_over_dk_log2e);
}

// =============================================================================
// Mode 21: Q fp8 per-(token,head) + K/V per-tensor fp8 launcher.
// =============================================================================
template <int kTileM, int kTileN, int kTileK, int kTileV, int kBlockSize, int kSplitK,
          int kSplitMinLen>
void launch_attention_decode_qfp8_kpertensor_vpertensor_fp16_pv_compute_dim128_smallm_splitk_inner(
    void *y_ptr, void *lse_ptr, void *splitk_out_ptr, const void *q_ptr, void *kcache_ptr,
    void *vcache_ptr, const float *qscale_ptr, const float *kscale_ptr, const float *vscale_ptr,
    const int *block_ids_ptr, const int *num_seq_kvcache_ptr, int *split_flag_ptr,
    bool new_kv_included, int num_batch, int num_seq_q, int num_head_q, int num_head_k,
    int num_head_v, int heads_per_group, int num_dim_qk, int num_dim_v, int num_kvcache_blocks,
    int block_size, int num_seq_max_blocks, int qscale_pad_stride, int ldY, int ldQ,
    int64_t kcache_block_stride, int64_t kcache_token_stride, int64_t kcache_head_stride,
    int64_t vcache_block_stride, int64_t vcache_token_stride, int64_t vcache_head_stride,
    cudaStream_t stream) {
  using namespace cute;  // NOLINT
  constexpr int kStage = 1;
  constexpr int kHeadsPerGroup = 8;
  (void)qscale_pad_stride;

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

  auto Y = make_tensor(
      make_gmem_ptr(reinterpret_cast<const Tout *>(y_ptr)),
      make_shape(num_dim_v, heads_per_group, num_head_k, num_seq_q, num_batch),
      make_stride(Int<1>{}, num_dim_v, heads_per_group * num_dim_v, ldY, ldY * num_seq_q));

  auto splitY =
      make_tensor(make_gmem_ptr(reinterpret_cast<float *>(splitk_out_ptr)),
                  make_shape(num_dim_v, heads_per_group, num_head_k, num_seq_q, kSplitK, num_batch),
                  make_stride(Int<1>{}, num_dim_v, heads_per_group * num_dim_v,
                              num_dim_v * num_head_q, num_dim_v * num_head_q * num_seq_q,
                              num_dim_v * num_head_q * num_seq_q * kSplitK));

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
  auto slayout_y = tile_to_shape(GMMA::Layout_MN_SW128_Atom<Tout>{},
                                 make_shape(Int<kTileV>{}, Int<kTileM>{}, Int<1>{}));
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
  auto tma_copy_layout_y = tile_to_shape(GMMA::Layout_MN_SW128_Atom<Tout>{},
                                         make_shape(Int<kTileV>{}, Int<kHeadsPerGroup>{}));
  auto tma_copy_layout_splity = tile_to_shape(GMMA::Layout_MN_SW128_Atom<float>{},
                                              make_shape(Int<kTileV>{}, Int<kHeadsPerGroup>{}));

  auto tma_q = make_tma_copy(SM90_TMA_LOAD{}, Q, tma_copy_layout_q);
  auto tma_k = make_tma_copy(SM90_TMA_LOAD{}, K, tma_copy_layout_k);
  auto tma_v = make_tma_copy(SM90_TMA_LOAD{}, V, tma_copy_layout_v);
  auto tma_y = make_tma_copy(SM90_TMA_STORE{}, Y, tma_copy_layout_y);
  auto tma_splity = make_tma_copy(SM90_TMA_STORE{}, splitY, tma_copy_layout_splity);

  auto qk_mma_atom = mma_selector_qk_fp8_modes<kTileM>();
  auto sv_mma_atom = mma_selector_sv_f16_modes<kTileM, GMMA::Major::MN, GMMA::Major::K>();

  using TiledMmaQK = decltype(make_tiled_mma(qk_mma_atom));
  using TiledMmaSV = decltype(make_tiled_mma(sv_mma_atom));

  dim3 block(size(TiledMmaQK{}) + 32);
  dim3 grid(num_head_k, num_batch, kSplitK);

  constexpr int kWarpsPerWrapGroup = 4;
  int shm_qk = (cosize(slayout_q) + cosize(slayout_k)) * sizeof(Tin);
  int shm_pv = (cosize(slayout_p) + cosize(slayout_v)) * sizeof(TinPV) +
               sizeof(float) * kTileM * kWarpsPerWrapGroup;
  int shm_v_fp8 = static_cast<int>(cosize(slayout_v_fp8) * sizeof(TinKV));
  int shm_blk_ids = sizeof(int) * num_seq_max_blocks;
  int shm_y = std::max(cosize(slayout_y) * sizeof(Tout), cosize(slayout_splity) * sizeof(float));
  int shm_size = std::max(shm_qk + shm_pv + shm_v_fp8 + shm_blk_ids, shm_y);

  constexpr float kLog2e = 1.4426950408889634f;
  float one_over_dk_log2e = 1.f / sqrtf(float(num_dim_qk)) * kLog2e;

  auto kernel = kernels::
      attention_decode_qfp8_kpertensor_vpertensor_fp16_pv_compute_multistage_ws_smallm_splitk_kernel<  // NOLINT(whitespace/line_length)
          Tout, Tin, kTileM, kTileN, kTileK, kTileV, kHeadsPerGroup, TiledMmaQK, TiledMmaSV,
          decltype(tma_q), decltype(tma_k), decltype(tma_v), decltype(tma_y), decltype(tma_splity),
          decltype(slayout_q), decltype(slayout_k), decltype(slayout_p), decltype(slayout_s),
          decltype(slayout_v), decltype(slayout_v_fp8), decltype(slayout_y),
          decltype(slayout_splity), kBlockSize, kStage, kSplitK, kSplitMinLen>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

  int pad_heads_per_group = ((heads_per_group + 7) / 8) * 8;

  kernel<<<grid, block, shm_size, stream>>>(
      tma_q, tma_k, tma_v, tma_y, tma_splity, reinterpret_cast<Tout *>(y_ptr),
      reinterpret_cast<float *>(splitk_out_ptr), reinterpret_cast<float *>(lse_ptr), block_ids_ptr,
      num_seq_kvcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr, split_flag_ptr, new_kv_included,
      num_batch, num_seq_q, num_dim_qk, num_dim_v, num_head_q, num_head_k, num_head_v,
      heads_per_group, pad_heads_per_group, num_kvcache_blocks, num_seq_max_blocks,
      one_over_dk_log2e);
}

// =============================================================================
// Outer launcher dispatch over kTileM x kBlockSize x kSplitK for each mode.
// =============================================================================
#define DECODE_FP16PV_DISPATCH(LAUNCH_FN)    \
  if (splitk == 1) {                         \
    constexpr int kSplitMinLen = 4096;       \
    if (num_seq_q == 1) {                    \
      if (block_size == 32) {                \
        LAUNCH_FN(8, 32, 1, kSplitMinLen);   \
      } else if (block_size == 64) {         \
        LAUNCH_FN(8, 64, 1, kSplitMinLen);   \
      }                                      \
    } else if (num_seq_q == 2) {             \
      if (block_size == 32) {                \
        LAUNCH_FN(16, 32, 1, kSplitMinLen);  \
      } else if (block_size == 64) {         \
        LAUNCH_FN(16, 64, 1, kSplitMinLen);  \
      }                                      \
    } else if (num_seq_q == 3) {             \
      if (block_size == 32) {                \
        LAUNCH_FN(24, 32, 1, kSplitMinLen);  \
      } else if (block_size == 64) {         \
        LAUNCH_FN(24, 64, 1, kSplitMinLen);  \
      }                                      \
    } else if (num_seq_q == 4) {             \
      if (block_size == 32) {                \
        LAUNCH_FN(32, 32, 1, kSplitMinLen);  \
      } else if (block_size == 64) {         \
        LAUNCH_FN(32, 64, 1, kSplitMinLen);  \
      }                                      \
    }                                        \
  } else if (splitk == 4) {                  \
    constexpr int kSplitMinLen = 4096;       \
    if (num_seq_q == 1) {                    \
      if (block_size == 32) {                \
        LAUNCH_FN(8, 32, 4, kSplitMinLen);   \
      } else if (block_size == 64) {         \
        LAUNCH_FN(8, 64, 4, kSplitMinLen);   \
      }                                      \
    } else if (num_seq_q == 2) {             \
      if (block_size == 32) {                \
        LAUNCH_FN(16, 32, 4, kSplitMinLen);  \
      } else if (block_size == 64) {         \
        LAUNCH_FN(16, 64, 4, kSplitMinLen);  \
      }                                      \
    } else if (num_seq_q == 3) {             \
      if (block_size == 32) {                \
        LAUNCH_FN(24, 32, 4, kSplitMinLen);  \
      } else if (block_size == 64) {         \
        LAUNCH_FN(24, 64, 4, kSplitMinLen);  \
      }                                      \
    } else if (num_seq_q == 4) {             \
      if (block_size == 32) {                \
        LAUNCH_FN(32, 32, 4, kSplitMinLen);  \
      } else if (block_size == 64) {         \
        LAUNCH_FN(32, 64, 4, kSplitMinLen);  \
      }                                      \
    }                                        \
  } else if (splitk == 16) {                 \
    constexpr int kSplitMinLen = 512;        \
    if (num_seq_q == 1) {                    \
      if (block_size == 32) {                \
        LAUNCH_FN(8, 32, 16, kSplitMinLen);  \
      } else if (block_size == 64) {         \
        LAUNCH_FN(8, 64, 16, kSplitMinLen);  \
      }                                      \
    } else if (num_seq_q == 2) {             \
      if (block_size == 32) {                \
        LAUNCH_FN(16, 32, 16, kSplitMinLen); \
      } else if (block_size == 64) {         \
        LAUNCH_FN(16, 64, 16, kSplitMinLen); \
      }                                      \
    } else if (num_seq_q == 3) {             \
      if (block_size == 32) {                \
        LAUNCH_FN(24, 32, 16, kSplitMinLen); \
      } else if (block_size == 64) {         \
        LAUNCH_FN(24, 64, 16, kSplitMinLen); \
      }                                      \
    } else if (num_seq_q == 4) {             \
      if (block_size == 32) {                \
        LAUNCH_FN(32, 32, 16, kSplitMinLen); \
      } else if (block_size == 64) {         \
        LAUNCH_FN(32, 64, 16, kSplitMinLen); \
      }                                      \
    }                                        \
  }

void launch_attention_decode_qfp8_kpertoken_perhead_vperhead_fp16_pv_compute_dim128_smallm_splitk(
    void *y_ptr, void *lse_ptr, void *split_out_ptr, const void *q_ptr, void *kcache_ptr,
    void *vcache_ptr, const void *qscale_ptr, const void *kscale_ptr, const void *vscale_ptr,
    const int *block_ids_ptr, const int *num_seq_kvcache_ptr, int *split_flag_ptr,
    bool new_kv_included, int splitk, int num_batch, int num_seq_q, int num_head_q, int num_head_k,
    int num_head_v, int num_dim_qk, int num_dim_v, int num_kvcache_blocks, int block_size,
    int num_seq_max_blocks, int qscale_pad_stride, int ldY, int ldQ, int64_t kcache_block_stride,
    int64_t kcache_token_stride, int64_t kcache_head_stride, int64_t vcache_block_stride,
    int64_t vcache_token_stride, int64_t vcache_head_stride, cudaStream_t stream) {
  using namespace cute;  // NOLINT
  constexpr int kTileN = 64;
  constexpr int kTileK = 128;
  constexpr int kTileV = 128;
  if (num_dim_qk != kTileK || num_dim_v != kTileV || (block_size != 32 && block_size != 64) ||
      (splitk != 1 && splitk != 4 && splitk != 16)) {
    return;
  }
  int heads_per_group = num_head_q / num_head_k;
  const auto *qs_f32 = reinterpret_cast<const float *>(qscale_ptr);
  const auto *ks_f32 = reinterpret_cast<const float *>(kscale_ptr);
  const auto *vs_f32 = reinterpret_cast<const float *>(vscale_ptr);

  // clang-format off
#define DECODE_M20_LAUNCH(_kTileM_, _kBlockSize_, _kSplitK_, _kSplitMinLen_)                      \
  launch_attention_decode_qfp8_kpertoken_perhead_vperhead_fp16_pv_compute_smallm_splitk_inner \
      <(_kTileM_), kTileN, kTileK, kTileV, (_kBlockSize_), (_kSplitK_), (_kSplitMinLen_)>(        \
          y_ptr, lse_ptr, split_out_ptr, q_ptr, kcache_ptr, vcache_ptr, qs_f32, ks_f32, vs_f32,  \
          block_ids_ptr, num_seq_kvcache_ptr, split_flag_ptr, new_kv_included, num_batch,        \
          num_seq_q, num_head_q, num_head_k, num_head_v, heads_per_group, num_dim_qk, num_dim_v, \
          num_kvcache_blocks, block_size, num_seq_max_blocks, qscale_pad_stride, ldY, ldQ,       \
          kcache_block_stride, kcache_token_stride, kcache_head_stride, vcache_block_stride,      \
          vcache_token_stride, vcache_head_stride, stream)
  // clang-format on

  DECODE_FP16PV_DISPATCH(DECODE_M20_LAUNCH)
#undef DECODE_M20_LAUNCH
}

void launch_attention_decode_qfp8_kpertensor_vpertensor_fp16_pv_compute_dim128_smallm_splitk(
    void *y_ptr, void *lse_ptr, void *split_out_ptr, const void *q_ptr, void *kcache_ptr,
    void *vcache_ptr, const void *qscale_ptr, const void *kscale_ptr, const void *vscale_ptr,
    const int *block_ids_ptr, const int *num_seq_kvcache_ptr, int *split_flag_ptr,
    bool new_kv_included, int splitk, int num_batch, int num_seq_q, int num_head_q, int num_head_k,
    int num_head_v, int num_dim_qk, int num_dim_v, int num_kvcache_blocks, int block_size,
    int num_seq_max_blocks, int qscale_pad_stride, int ldY, int ldQ, int64_t kcache_block_stride,
    int64_t kcache_token_stride, int64_t kcache_head_stride, int64_t vcache_block_stride,
    int64_t vcache_token_stride, int64_t vcache_head_stride, cudaStream_t stream) {
  using namespace cute;  // NOLINT
  constexpr int kTileN = 64;
  constexpr int kTileK = 128;
  constexpr int kTileV = 128;
  if (num_dim_qk != kTileK || num_dim_v != kTileV || (block_size != 32 && block_size != 64) ||
      (splitk != 1 && splitk != 4 && splitk != 16)) {
    return;
  }
  int heads_per_group = num_head_q / num_head_k;
  const auto *qs_f32 = reinterpret_cast<const float *>(qscale_ptr);
  const auto *ks_f32 = reinterpret_cast<const float *>(kscale_ptr);
  const auto *vs_f32 = reinterpret_cast<const float *>(vscale_ptr);

#define DECODE_M21_LAUNCH(_kTileM_, _kBlockSize_, _kSplitK_, _kSplitMinLen_)                     \
  launch_attention_decode_qfp8_kpertensor_vpertensor_fp16_pv_compute_dim128_smallm_splitk_inner< \
      (_kTileM_), kTileN, kTileK, kTileV, (_kBlockSize_), (_kSplitK_), (_kSplitMinLen_)>(        \
      y_ptr, lse_ptr, split_out_ptr, q_ptr, kcache_ptr, vcache_ptr, qs_f32, ks_f32, vs_f32,      \
      block_ids_ptr, num_seq_kvcache_ptr, split_flag_ptr, new_kv_included, num_batch, num_seq_q, \
      num_head_q, num_head_k, num_head_v, heads_per_group, num_dim_qk, num_dim_v,                \
      num_kvcache_blocks, block_size, num_seq_max_blocks, qscale_pad_stride, ldY, ldQ,           \
      kcache_block_stride, kcache_token_stride, kcache_head_stride, vcache_block_stride,         \
      vcache_token_stride, vcache_head_stride, stream)

  DECODE_FP16PV_DISPATCH(DECODE_M21_LAUNCH)
#undef DECODE_M21_LAUNCH
}

#undef DECODE_FP16PV_DISPATCH

}  // namespace decode
}  // namespace attention
}  // namespace hpc
