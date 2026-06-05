// Copyright (C) 2026 Tencent.

#ifndef SRC_STEM_STEM_KERNELS_CUH_
#define SRC_STEM_STEM_KERNELS_CUH_

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/fast_math.h"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace stem {
namespace kernels {

//  stem_prep_paged_kv_qpertoken_perhead_kvpertensor_kernel — K flat + V norm down (paged KV cache)
//
// Legacy quant_type=1 path: K-scale and V-scale are per-tensor scalars.
//
// K processing:
//   Group g sums kSamplePerBlock rows at positions {g, g+stride, g+2*stride, ...}
//   FP8 → FP32 accumulate × kscale → BF16.  Stored to Kflat segment (kStride-1-g) for GEMM
//
// V processing:
//   Warp g computes max L2 norm over kStride consecutive rows
//   [g*kStride .. (g+1)*kStride-1].  Writes FP32 to v_norm_down buffer.
template <int kBlockSize, int kStemBlockSize, int kStride, int kDimQK, int kDimV>
__global__ void __launch_bounds__(kStemBlockSize / kStride * 32)
    stem_prep_paged_kv_qpertoken_perhead_kvpertensor_kernel(
        const __nv_fp8_e4m3* __restrict__ kcache, const __nv_fp8_e4m3* __restrict__ vcache,
        const float* __restrict__ kscale_ptr, const float* __restrict__ vscale_ptr,
        const int* __restrict__ block_ids, const int* __restrict__ kv_seq_lens,
        __nv_bfloat16* __restrict__ kflat, float* __restrict__ v_norm_down, int num_head_kv,
        int max_blocks_per_req, int ldK, int ldV, int max_num_stem_blocks, int max_k_down_len) {
  constexpr int kSamplePerBlock = kStemBlockSize / kStride;
  constexpr int kElemsPerThread_K = kDimQK / 32;
  constexpr int kElemsPerThread_V = kDimV / 32;

  static_assert(kStemBlockSize % kStride == 0, "kStemBlockSize must be divisible by kStride");
  static_assert(kStemBlockSize % kBlockSize == 0, "kStemBlockSize must be divisible by kBlockSize");
  static_assert(kDimQK % 32 == 0, "kDimQK must be divisible by warp size");
  static_assert(kDimV % 32 == 0, "kDimV must be divisible by warp size");
  static_assert(kSamplePerBlock >= 1 && kSamplePerBlock <= 32,
                "kSamplePerBlock warps must be in [1, 32] for a valid CTA");

  const int stem_block_idx = blockIdx.x;
  const int ibatch = blockIdx.y / num_head_kv;
  const int ihead_kv = blockIdx.y % num_head_kv;
  const int iwarp = threadIdx.x / 32;
  const int ilane = threadIdx.x % 32;

  const int kv_len = kv_seq_lens[ibatch];
  const int k_padded_len = ((kv_len + kStemBlockSize - 1) / kStemBlockSize) * kStemBlockSize;
  const int num_stem_blocks_req = k_padded_len / kStemBlockSize;
  if (stem_block_idx >= num_stem_blocks_req) {
    return;
  }

  const int stem_row_base = stem_block_idx * kStemBlockSize;

  // K processing
  const float kscale_val = kscale_ptr[0];
#pragma unroll
  for (int group_id = iwarp; group_id < kStride; group_id += kSamplePerBlock) {
    vec_t<float, kElemsPerThread_K> k_acc;
#pragma unroll
    for (int j = 0; j < kElemsPerThread_K; j++) {
      k_acc[j] = 0.0f;
    }

#pragma unroll
    for (int isample = 0; isample < kSamplePerBlock; isample++) {
      const int global_row = stem_row_base + group_id + isample * kStride;
      if (global_row >= kv_len) {
        continue;
      }

      const int page_idx = global_row / kBlockSize;
      const int row_in_page = global_row % kBlockSize;
      const int physical_block_number = block_ids[ibatch * max_blocks_per_req + page_idx];

      const __nv_fp8_e4m3* k_ptr = kcache + static_cast<int64_t>(physical_block_number) * ldK +
                                   static_cast<int64_t>(row_in_page) * num_head_kv * kDimQK +
                                   ihead_kv * kDimQK + ilane * kElemsPerThread_K;

      auto packed_fp8 = load<__nv_fp8x4_e4m3, 1>(k_ptr);
      auto fp32_vals = to<float>(packed_fp8);

#pragma unroll
      for (int j = 0; j < kElemsPerThread_K; j++) {
        k_acc[j] += fp32_vals[j];
      }
    }

#pragma unroll
    for (int j = 0; j < kElemsPerThread_K; j++) {
      k_acc[j] *= kscale_val;
    }

    // FP32 → BF16, store with reversed group order.
    auto k_bf16 = to<__nv_bfloat16>(k_acc);

    __nv_bfloat16* kflat_out_ptr = kflat +
                                   (static_cast<int64_t>(ibatch) * num_head_kv + ihead_kv) *
                                       static_cast<int64_t>(max_num_stem_blocks) * kStride *
                                       kDimQK +
                                   static_cast<int64_t>(stem_block_idx) * kStride * kDimQK +
                                   (kStride - 1 - group_id) * kDimQK + ilane * kElemsPerThread_K;

    store(kflat_out_ptr, k_bf16);
  }

  // V processing
  const float vscale_val = vscale_ptr[0];
  float max_norm = 0.0f;

#pragma unroll
  for (int istep = 0; istep < kStride; istep++) {
    const int global_row = stem_row_base + iwarp * kStride + istep;
    if (global_row >= kv_len) {
      continue;
    }

    const int page_idx = global_row / kBlockSize;
    const int row_in_page = global_row % kBlockSize;
    const int physical_block_number = block_ids[ibatch * max_blocks_per_req + page_idx];

    const __nv_fp8_e4m3* v_ptr = vcache + static_cast<int64_t>(physical_block_number) * ldV +
                                 static_cast<int64_t>(row_in_page) * num_head_kv * kDimV +
                                 ihead_kv * kDimV + ilane * kElemsPerThread_V;

    auto packed_fp8 = load<__nv_fp8x4_e4m3, 1>(v_ptr);
    auto fp32_vals = to<float>(packed_fp8);

    float partial_square = 0.0f;
#pragma unroll
    for (int j = 0; j < kElemsPerThread_V; j++) {
      float val = fp32_vals[j] * vscale_val;
      partial_square += val * val;
    }

    float norm_square = warp_reduce_sum_xor(partial_square);
    float norm = sqrtf(norm_square);
    max_norm = fmaxf(max_norm, norm);
  }

  // warp_reduce_sum_xor broadcasts to all lanes; only lane 0 writes.
  if (ilane == 0) {
    const int k_down_len = k_padded_len / kStride;
    const int down_idx = stem_block_idx * kSamplePerBlock + iwarp;
    if (down_idx < k_down_len) {
      v_norm_down[(static_cast<int64_t>(ibatch) * num_head_kv + ihead_kv) * max_k_down_len +
                  down_idx] = max_norm;
    }
  }
}

//  stem_prep_paged_kv_qkpertoken_perhead_vperhead_kernel — K flat + V norm down (paged KV cache)
//
// New quant_type=0 path: per-KV-token K-scale + per-head V-scale, mirroring the attention
// qkpertoken_perhead_vperhead scale layout.  K-scale GMEM shape is
// [num_blocks, kScaleBlockSize, num_head_kv, num_dim_scale] with strides ldKS / ldKS1 / ldKS2
// (in fp32 elements); V-scale is vscale_ptr[ihead_kv].
template <int kBlockSize, int kStemBlockSize, int kStride, int kDimQK, int kDimV>
__global__ void __launch_bounds__(kStemBlockSize / kStride * 32)
    stem_prep_paged_kv_qkpertoken_perhead_vperhead_kernel(
        const __nv_fp8_e4m3* __restrict__ kcache, const __nv_fp8_e4m3* __restrict__ vcache,
        const float* __restrict__ kscale_ptr, const float* __restrict__ vscale_ptr,
        const int* __restrict__ block_ids, const int* __restrict__ kv_seq_lens,
        __nv_bfloat16* __restrict__ kflat, float* __restrict__ v_norm_down, int num_head_kv,
        int max_blocks_per_req, int ldK, int ldV, int ldKS, int ldKS1, int ldKS2,
        int max_num_stem_blocks, int max_k_down_len) {
  constexpr int kSamplePerBlock = kStemBlockSize / kStride;
  constexpr int kElemsPerThread_K = kDimQK / 32;
  constexpr int kElemsPerThread_V = kDimV / 32;
  // kRowsPerScaleChunk mirrors attention's num_dim_scale (the row-group fan-out of
  // the per-token K-scale layout); the historical name num_dim_scale is kept on the
  // attention side for interface continuity.
  constexpr int kRowsPerScaleChunk = kDimV / static_cast<int>(sizeof(float));

  static_assert(kStemBlockSize % kStride == 0, "kStemBlockSize must be divisible by kStride");
  static_assert(kStemBlockSize % kBlockSize == 0, "kStemBlockSize must be divisible by kBlockSize");
  static_assert(kDimQK % 32 == 0, "kDimQK must be divisible by warp size");
  static_assert(kDimV % 32 == 0, "kDimV must be divisible by warp size");
  static_assert(kSamplePerBlock >= 1 && kSamplePerBlock <= 32,
                "kSamplePerBlock warps must be in [1, 32] for a valid CTA");
  static_assert(kBlockSize % kRowsPerScaleChunk == 0,
                "kBlockSize must be a multiple of kRowsPerScaleChunk (= kDimV / sizeof(float))");

  const int stem_block_idx = blockIdx.x;
  const int ibatch = blockIdx.y / num_head_kv;
  const int ihead_kv = blockIdx.y % num_head_kv;
  const int iwarp = threadIdx.x / 32;
  const int ilane = threadIdx.x % 32;

  const int kv_len = kv_seq_lens[ibatch];
  const int k_padded_len = ((kv_len + kStemBlockSize - 1) / kStemBlockSize) * kStemBlockSize;
  const int num_stem_blocks_req = k_padded_len / kStemBlockSize;
  if (stem_block_idx >= num_stem_blocks_req) {
    return;
  }

  const int stem_row_base = stem_block_idx * kStemBlockSize;

  // Per-token K-scales for this stem block preloaded to SMEM; padding rows get 0.0f
  // (harmless because the K accumulator skips them below).
  __shared__ float s_kscale[kStemBlockSize];
#pragma unroll
  for (int i = threadIdx.x; i < kStemBlockSize; i += blockDim.x) {
    const int global_row = stem_row_base + i;
    if (global_row < kv_len) {
      const int page_idx = global_row / kBlockSize;
      const int row_in_page = global_row % kBlockSize;
      const int row_chunk = row_in_page / kRowsPerScaleChunk;
      const int row_offset = row_in_page % kRowsPerScaleChunk;
      const int phys_block = block_ids[ibatch * max_blocks_per_req + page_idx];
      const int64_t kscale_offset =
          static_cast<int64_t>(phys_block) * ldKS + static_cast<int64_t>(row_chunk) * ldKS1 +
          static_cast<int64_t>(ihead_kv) * ldKS2 + static_cast<int64_t>(row_offset);
      s_kscale[i] = kscale_ptr[kscale_offset];
    } else {
      s_kscale[i] = 0.0f;
    }
  }
  __syncthreads();

  // K processing
#pragma unroll
  for (int group_id = iwarp; group_id < kStride; group_id += kSamplePerBlock) {
    vec_t<float, kElemsPerThread_K> k_acc;
#pragma unroll
    for (int j = 0; j < kElemsPerThread_K; j++) {
      k_acc[j] = 0.0f;
    }

#pragma unroll
    for (int isample = 0; isample < kSamplePerBlock; isample++) {
      const int row_in_stem = group_id + isample * kStride;
      const int global_row = stem_row_base + row_in_stem;
      if (global_row >= kv_len) {
        continue;
      }

      const int page_idx = global_row / kBlockSize;
      const int row_in_page = global_row % kBlockSize;
      const int physical_block_number = block_ids[ibatch * max_blocks_per_req + page_idx];

      const __nv_fp8_e4m3* k_ptr = kcache + static_cast<int64_t>(physical_block_number) * ldK +
                                   static_cast<int64_t>(row_in_page) * num_head_kv * kDimQK +
                                   ihead_kv * kDimQK + ilane * kElemsPerThread_K;

      auto packed_fp8 = load<__nv_fp8x4_e4m3, 1>(k_ptr);
      auto fp32_vals = to<float>(packed_fp8);

      const float kscale_row = s_kscale[row_in_stem];
#pragma unroll
      for (int j = 0; j < kElemsPerThread_K; j++) {
        k_acc[j] += kscale_row * fp32_vals[j];
      }
    }

    // FP32 → BF16, store with reversed group order.
    auto k_bf16 = to<__nv_bfloat16>(k_acc);

    __nv_bfloat16* kflat_out_ptr = kflat +
                                   (static_cast<int64_t>(ibatch) * num_head_kv + ihead_kv) *
                                       static_cast<int64_t>(max_num_stem_blocks) * kStride *
                                       kDimQK +
                                   static_cast<int64_t>(stem_block_idx) * kStride * kDimQK +
                                   (kStride - 1 - group_id) * kDimQK + ilane * kElemsPerThread_K;

    store(kflat_out_ptr, k_bf16);
  }

  // V processing
  const float vscale_val = vscale_ptr[ihead_kv];
  float max_norm = 0.0f;

#pragma unroll
  for (int istep = 0; istep < kStride; istep++) {
    const int global_row = stem_row_base + iwarp * kStride + istep;
    if (global_row >= kv_len) {
      continue;
    }

    const int page_idx = global_row / kBlockSize;
    const int row_in_page = global_row % kBlockSize;
    const int physical_block_number = block_ids[ibatch * max_blocks_per_req + page_idx];

    const __nv_fp8_e4m3* v_ptr = vcache + static_cast<int64_t>(physical_block_number) * ldV +
                                 static_cast<int64_t>(row_in_page) * num_head_kv * kDimV +
                                 ihead_kv * kDimV + ilane * kElemsPerThread_V;

    auto packed_fp8 = load<__nv_fp8x4_e4m3, 1>(v_ptr);
    auto fp32_vals = to<float>(packed_fp8);

    float partial_square = 0.0f;
#pragma unroll
    for (int j = 0; j < kElemsPerThread_V; j++) {
      float val = fp32_vals[j] * vscale_val;
      partial_square += val * val;
    }

    float norm_square = warp_reduce_sum_xor(partial_square);
    float norm = sqrtf(norm_square);
    max_norm = fmaxf(max_norm, norm);
  }

  // warp_reduce_sum_xor broadcasts to all lanes; only lane 0 writes.
  if (ilane == 0) {
    const int k_down_len = k_padded_len / kStride;
    const int down_idx = stem_block_idx * kSamplePerBlock + iwarp;
    if (down_idx < k_down_len) {
      v_norm_down[(static_cast<int64_t>(ibatch) * num_head_kv + ihead_kv) * max_k_down_len +
                  down_idx] = max_norm;
    }
  }
}

//  stem_prep_varlen_kv_kernel — K flat + V norm down (ragged varlen)
//
// K: FP8 group-sum × kscale, reversed group order → kflat BF16.
// V: FP8 × vscale → L2 norm → v_norm_down FP32.
template <int kStemBlockSize, int kStride, int kDimQK, int kDimV>
__global__ void __launch_bounds__(kStemBlockSize / kStride * 32) stem_prep_varlen_kv_kernel(
    const __nv_fp8_e4m3* __restrict__ k_fp8, const __nv_fp8_e4m3* __restrict__ v_fp8,
    const float* __restrict__ kscale_ptr, const float* __restrict__ vscale_ptr,
    const int* __restrict__ kv_seq_lens, const int* __restrict__ cu_seqlens_kv,
    __nv_bfloat16* __restrict__ kflat, float* __restrict__ v_norm_down, int num_head_kv, int ldK,
    int ldV, int max_num_stem_blocks, int max_k_down_len) {
  constexpr int kSamplePerBlock = kStemBlockSize / kStride;
  constexpr int kElemsPerThread_K = kDimQK / 32;
  constexpr int kElemsPerThread_V = kDimV / 32;

  static_assert(kStemBlockSize % kStride == 0, "kStemBlockSize must be divisible by kStride");
  static_assert(kDimQK % 32 == 0, "kDimQK must be divisible by warp size");
  static_assert(kDimV % 32 == 0, "kDimV must be divisible by warp size");
  static_assert(kSamplePerBlock >= 1 && kSamplePerBlock <= 32,
                "kSamplePerBlock warps must be in [1, 32] for a valid CTA");

  const int stem_block_idx = blockIdx.x;
  const int ibatch = blockIdx.y / num_head_kv;
  const int ihead_kv = blockIdx.y % num_head_kv;
  const int iwarp = threadIdx.x / 32;
  const int ilane = threadIdx.x % 32;

  const int kv_len = kv_seq_lens[ibatch];
  const int k_padded_len = ((kv_len + kStemBlockSize - 1) / kStemBlockSize) * kStemBlockSize;
  const int num_stem_blocks_req = k_padded_len / kStemBlockSize;
  if (stem_block_idx >= num_stem_blocks_req) {
    return;
  }

  const int kv_start_offset = cu_seqlens_kv[ibatch];
  const int stem_row_base = stem_block_idx * kStemBlockSize;

  // K processing
  const float kscale_val = kscale_ptr[0];
#pragma unroll
  for (int group_id = iwarp; group_id < kStride; group_id += kSamplePerBlock) {
    vec_t<float, kElemsPerThread_K> k_acc;
#pragma unroll
    for (int j = 0; j < kElemsPerThread_K; j++) {
      k_acc[j] = 0.0f;
    }

#pragma unroll
    for (int isample = 0; isample < kSamplePerBlock; isample++) {
      const int global_row = stem_row_base + group_id + isample * kStride;
      if (global_row >= kv_len) {
        continue;
      }

      const __nv_fp8_e4m3* k_ptr = k_fp8 +
                                   static_cast<int64_t>(kv_start_offset + global_row) * ldK +
                                   ihead_kv * kDimQK + ilane * kElemsPerThread_K;

      if constexpr (kElemsPerThread_K == 4) {
        auto packed_fp8 = load<__nv_fp8x4_e4m3, 1>(k_ptr);
        auto fp32_vals = to<float>(packed_fp8);
#pragma unroll
        for (int j = 0; j < kElemsPerThread_K; j++) {
          k_acc[j] += fp32_vals[j];
        }
      } else if constexpr (kElemsPerThread_K == 6) {
        // ilane*6 not 4B-aligned for odd lanes; use 3 × 2B loads.
        auto fp32_seg0 = to<float>(load<__nv_fp8_e4m3, 2>(k_ptr));
        auto fp32_seg1 = to<float>(load<__nv_fp8_e4m3, 2>(k_ptr + 2));
        auto fp32_seg2 = to<float>(load<__nv_fp8_e4m3, 2>(k_ptr + 4));
#pragma unroll
        for (int j = 0; j < 2; j++) {
          k_acc[j] += fp32_seg0[j];
          k_acc[j + 2] += fp32_seg1[j];
          k_acc[j + 4] += fp32_seg2[j];
        }
      }
    }

#pragma unroll
    for (int j = 0; j < kElemsPerThread_K; j++) {
      k_acc[j] *= kscale_val;
    }

    __nv_bfloat16* kflat_out_ptr = kflat +
                                   (static_cast<int64_t>(ibatch) * num_head_kv + ihead_kv) *
                                       static_cast<int64_t>(max_num_stem_blocks) * kStride *
                                       kDimQK +
                                   static_cast<int64_t>(stem_block_idx) * kStride * kDimQK +
                                   (kStride - 1 - group_id) * kDimQK + ilane * kElemsPerThread_K;

    auto k_bf16 = to<__nv_bfloat16>(k_acc);
    if constexpr (kElemsPerThread_K == 4) {
      store(kflat_out_ptr, k_bf16);
    } else if constexpr (kElemsPerThread_K == 6) {
      // ilane*12 not 8B-aligned for odd lanes; use 3 × 4B stores.
      store(kflat_out_ptr, k_bf16[0], k_bf16[1]);
      store(kflat_out_ptr + 2, k_bf16[2], k_bf16[3]);
      store(kflat_out_ptr + 4, k_bf16[4], k_bf16[5]);
    }
  }

  // V processing
  const float vscale_val = vscale_ptr[0];
  float max_norm = 0.0f;

#pragma unroll
  for (int istep = 0; istep < kStride; istep++) {
    const int global_row = stem_row_base + iwarp * kStride + istep;
    if (global_row >= kv_len) {
      continue;
    }

    const __nv_fp8_e4m3* v_ptr = v_fp8 + static_cast<int64_t>(kv_start_offset + global_row) * ldV +
                                 ihead_kv * kDimV + ilane * kElemsPerThread_V;

    auto packed_fp8 = load<__nv_fp8x4_e4m3, 1>(v_ptr);
    auto fp32_vals = to<float>(packed_fp8);

    float partial_square = 0.0f;
#pragma unroll
    for (int j = 0; j < kElemsPerThread_V; j++) {
      float val = fp32_vals[j] * vscale_val;
      partial_square += val * val;
    }

    float norm_square = warp_reduce_sum_xor(partial_square);
    float norm = sqrtf(norm_square);
    max_norm = fmaxf(max_norm, norm);
  }

  if (ilane == 0) {
    const int k_down_len = k_padded_len / kStride;
    const int down_idx = stem_block_idx * kSamplePerBlock + iwarp;
    if (down_idx < k_down_len) {
      v_norm_down[(static_cast<int64_t>(ibatch) * num_head_kv + ihead_kv) * max_k_down_len +
                  down_idx] = max_norm;
    }
  }
}

// Block-level sum reduction (256 threads) for vbias_reduce_kernel
__device__ __forceinline__ float block_reduce_sum_256(float val, float* smem) {
  __syncthreads();  // isolate from any prior use of this smem buffer by the caller
  constexpr int kNumWarps = 8;
  val = warp_reduce_sum_xor(val);
  const int iwarp = threadIdx.x / 32;
  const int ilane = threadIdx.x % 32;
  if (ilane == 0) {
    smem[iwarp] = val;
  }
  __syncthreads();
  if (iwarp == 0) {
    float tmp = (ilane < kNumWarps) ? smem[ilane] : 0.0f;
    tmp = warp_reduce_sum_xor(tmp);
    if (ilane == 0) {
      smem[0] = tmp;
    }
  }
  __syncthreads();
  return smem[0];
}

// vbias_reduce_kernel — global normalize + block average → V_bias
//
// Part of stem_oam_prep_paged_kv / prep_varlen_kv (sub-kernel 2 of 2).
template <int kStemBlockSize, int kStride>
__global__ void __launch_bounds__(256)
    vbias_reduce_kernel(const float* __restrict__ v_norm_down, const int* __restrict__ kv_seq_lens,
                        float* __restrict__ vbias, int num_head_kv, int max_k_down_len,
                        int max_num_stem_blocks, float lambda_mag) {
  constexpr int kSamplePerBlock = kStemBlockSize / kStride;
  constexpr int kBlockDim = 256;

  const int ibatch = blockIdx.x / num_head_kv;
  const int ihead_kv = blockIdx.x % num_head_kv;

  const int kv_len = kv_seq_lens[ibatch];
  const int k_padded_len = ((kv_len + kStemBlockSize - 1) / kStemBlockSize) * kStemBlockSize;
  const int k_down_len = k_padded_len / kStride;
  if (k_down_len == 0) {
    return;
  }

  const float* v_norm_down_base =
      v_norm_down + (static_cast<int64_t>(ibatch) * num_head_kv + ihead_kv) * max_k_down_len;

  __shared__ float smem[8];

  // 1: mean of log values
  float local_sum = 0.0f;
  for (int i = threadIdx.x; i < k_down_len; i += kBlockDim) {
    local_sum += logf(v_norm_down_base[i] + 1e-6f);
  }
  const float mean = block_reduce_sum_256(local_sum, smem) / static_cast<float>(k_down_len);

  // 2: std of log values
  float local_square = 0.0f;
  for (int i = threadIdx.x; i < k_down_len; i += kBlockDim) {
    float diff = logf(v_norm_down_base[i] + 1e-6f) - mean;
    local_square += diff * diff;
  }
  const float variance = block_reduce_sum_256(local_square, smem);
  const float std_val =
      (k_down_len > 1) ? sqrtf(variance / static_cast<float>(k_down_len - 1)) : 0.0f;
  const float inv_std = 1.0f / (std_val + 1e-6f);

  // 3: normalize + ReLU + scale + block average
  const int num_stem_blocks = k_padded_len / kStemBlockSize;

  float* vbias_base =
      vbias + (static_cast<int64_t>(ibatch) * num_head_kv + ihead_kv) * max_num_stem_blocks;

  for (int iblock = threadIdx.x; iblock < num_stem_blocks; iblock += kBlockDim) {
    float block_sum = 0.0f;
#pragma unroll
    for (int isample = 0; isample < kSamplePerBlock; isample++) {
      const int idx = iblock * kSamplePerBlock + isample;
      if (idx < k_down_len) {
        float log_val = logf(v_norm_down_base[idx] + 1e-6f);
        float normalized = (log_val - mean) * inv_std;
        block_sum += lambda_mag * fmaxf(0.0f, normalized);
      }
    }
    vbias_base[iblock] = block_sum / static_cast<float>(kSamplePerBlock);
  }
}

// stem_prep_varlen_q_kernel — Q flat precompute
//
// Weighted group-sum of Q tokens with qscale (FP8 dequant scale).
//
// Scale loading strategy (compile-time, via kIsPerTensorQscale):
//   false: per-token qscale[B, Hq, seq] → shared memory (kStemBlockSize floats)
//   true:  scalar qscale[0] → register (L1 broadcast)
template <int kStemBlockSize, int kStride, int kDimQK, bool kIsPerTensorQscale>
__global__ void __launch_bounds__(kStemBlockSize / kStride * 32)
    stem_prep_varlen_q_kernel(const __nv_fp8_e4m3* __restrict__ q_fp8,
                              const float* __restrict__ qscale_ptr,
                              const int* __restrict__ q_seq_lens,
                              const int* __restrict__ cu_seqlens_q,
                              __nv_bfloat16* __restrict__ qflat, int num_head_q, int ldQ,
                              int max_num_q_blocks) {
  constexpr int kSamplePerBlock = kStemBlockSize / kStride;
  constexpr int kElemsPerThread = kDimQK / 32;
  constexpr int kFlatDim = kStride * kDimQK;

  static_assert(kStemBlockSize % kStride == 0, "kStemBlockSize must be divisible by kStride");
  static_assert(kDimQK % 32 == 0, "kDimQK must be divisible by warp size");
  static_assert(kSamplePerBlock >= 1 && kSamplePerBlock <= 32,
                "kSamplePerBlock warps must be in [1, 32] for a valid CTA");
  static_assert(kElemsPerThread == 4, "only dim128 (kElemsPerThread=4) is supported");

  const int q_block_idx = blockIdx.x;
  const int ibatch = blockIdx.y / num_head_q;
  const int ihead_q = blockIdx.y % num_head_q;
  const int iwarp = threadIdx.x / 32;
  const int ilane = threadIdx.x % 32;

  const int q_len = q_seq_lens[ibatch];
  const int q_padded_len = ((q_len + kStemBlockSize - 1) / kStemBlockSize) * kStemBlockSize;
  const int num_q_blocks_req = q_padded_len / kStemBlockSize;
  if (q_block_idx >= num_q_blocks_req) {
    return;
  }

  const int q_start_offset = cu_seqlens_q[ibatch];

  // Preload qscale: per-token in smem, or scalar in register.
  __shared__ float qscale_per_token[kIsPerTensorQscale ? 1 : kStemBlockSize];
  float qscale_per_tensor = 0.0f;
  if constexpr (kIsPerTensorQscale) {
    qscale_per_tensor = qscale_ptr[0];
  } else {
    const int block_start_idx = q_block_idx * kStemBlockSize;
    const int qs_seq_len = max_num_q_blocks * kStemBlockSize;
    const float* scale_base =
        qscale_ptr + ibatch * (num_head_q * qs_seq_len) + ihead_q * qs_seq_len;
#pragma unroll
    for (int i = threadIdx.x; i < kStemBlockSize; i += blockDim.x) {
      const int token_idx = block_start_idx + i;
      qscale_per_token[i] = (token_idx < q_len) ? scale_base[token_idx] : 0.0f;
    }
    __syncthreads();
  }

#pragma unroll
  for (int group_id = iwarp; group_id < kStride; group_id += kSamplePerBlock) {
    vec_t<float, kElemsPerThread> q_acc;
#pragma unroll
    for (int j = 0; j < kElemsPerThread; j++) {
      q_acc[j] = 0.0f;
    }

#pragma unroll
    for (int isample = 0; isample < kSamplePerBlock; isample++) {
      const int token_idx = q_block_idx * kStemBlockSize + group_id + isample * kStride;
      if (token_idx >= q_len) {
        continue;
      }

      const int global_token_idx = q_start_offset + token_idx;
      const __nv_fp8_e4m3* q_ptr = q_fp8 + static_cast<int64_t>(global_token_idx) * ldQ +
                                   ihead_q * kDimQK + ilane * kElemsPerThread;

      const float scale =
          kIsPerTensorQscale ? qscale_per_tensor : qscale_per_token[group_id + isample * kStride];

      if constexpr (kElemsPerThread == 4) {
        auto packed_fp8 = load<__nv_fp8x4_e4m3, 1>(q_ptr);
        auto fp32_vals = to<float>(packed_fp8);
#pragma unroll
        for (int j = 0; j < kElemsPerThread; j++) {
          q_acc[j] += scale * fp32_vals[j];
        }
      } else if constexpr (kElemsPerThread == 6) {
        // ilane*6 not 4B-aligned for odd lanes; use 3 × 2B loads.
        auto fp32_seg0 = to<float>(load<__nv_fp8_e4m3, 2>(q_ptr));
        auto fp32_seg1 = to<float>(load<__nv_fp8_e4m3, 2>(q_ptr + 2));
        auto fp32_seg2 = to<float>(load<__nv_fp8_e4m3, 2>(q_ptr + 4));
#pragma unroll
        for (int j = 0; j < 2; j++) {
          q_acc[j] += scale * fp32_seg0[j];
          q_acc[j + 2] += scale * fp32_seg1[j];
          q_acc[j + 4] += scale * fp32_seg2[j];
        }
      }
    }

    // FP32 → BF16, store Q flat with natural group order.
    __nv_bfloat16* qflat_out_ptr = qflat +
                                   (static_cast<int64_t>(ibatch) * num_head_q + ihead_q) *
                                       static_cast<int64_t>(max_num_q_blocks) * kFlatDim +
                                   static_cast<int64_t>(q_block_idx) * kFlatDim +
                                   group_id * kDimQK + ilane * kElemsPerThread;

    auto q_bf16 = to<__nv_bfloat16>(q_acc);
    if constexpr (kElemsPerThread == 4) {
      store(qflat_out_ptr, q_bf16);
    } else if constexpr (kElemsPerThread == 6) {
      // ilane*12 not 8B-aligned for odd lanes; use 3 × 4B stores.
      store(qflat_out_ptr, q_bf16[0], q_bf16[1]);
      store(qflat_out_ptr + 2, q_bf16[2], q_bf16[3]);
      store(qflat_out_ptr + 4, q_bf16[4], q_bf16[5]);
    }
  }
}

// get_next_tile — persistent GEMM tile dispatcher
template <int kTileM, int kTileN, int kStemBlockSize>
__device__ __forceinline__ auto get_next_tile(int iblock, cutlass::FastDivmod const& qk_tile_divmod,
                                              cutlass::FastDivmod const& k_tile_divmod,
                                              cutlass::FastDivmod const& head_q_divmod) {
  int ibatch_head_q, qk_rem;
  qk_tile_divmod(ibatch_head_q, qk_rem, iblock);

  int itile_q, itile_k;
  k_tile_divmod(itile_q, itile_k, qk_rem);

  int ibatch, ihead_q;
  head_q_divmod(ibatch, ihead_q, ibatch_head_q);

  return cute::make_tuple(ibatch, ihead_q, itile_q, itile_k);
}

// stem_oam_gemm_kernel — Frobenius GEMM (warp-spec persistent)
//
// block_logits = FrobScale × (Qflat @ Kflat^T) + V_bias,  optional causal mask.
// FrobScale = 1 / (kSamplePerBlock²) = 1/64 when stem_block=128, stride=16.
//
// Producer: TMA-only. Loads Qflat + Kflat chunks to SMEM.
// Consumer: WGMMA BF16 SS + custom epilogue.
template <bool kCausal, typename TiledMma, typename TmaQ, typename TmaK, typename TmaLogits,
          int kTileM, int kTileN, int kStage, int kStemBlockSize, int kStride, int kDimQK,
          typename SLayoutQ, typename SLayoutK, typename SLayoutLogits>
__global__ void __launch_bounds__(384, 1)
    stem_oam_gemm_kernel(const __grid_constant__ TmaQ tma_q, const __grid_constant__ TmaK tma_k,
                         const __grid_constant__ TmaLogits tma_logits,
                         const int* __restrict__ q_seq_lens, const int* __restrict__ kv_seq_lens,
                         const float* __restrict__ vbias_ptr, int num_batch, int num_head_q,
                         int num_heads_per_kv, int max_num_qb, int max_num_kb, int max_total_tiles,
                         cutlass::FastDivmod qk_tile_divmod, cutlass::FastDivmod k_tile_divmod,
                         cutlass::FastDivmod head_q_divmod) {
  using namespace cute;  // NOLINT

  using Tbf16 = cute::bfloat16_t;

  constexpr int kFlatDim = kStride * kDimQK;
  constexpr int kSamplePerBlock = kStemBlockSize / kStride;
  constexpr float kFrobScale = 1.0f / static_cast<float>(kSamplePerBlock * kSamplePerBlock);

  int idx = threadIdx.x;

  int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
  int elected = cute::elect_one_sync();
  bool is_leader_in_block = (iwarp == 0) && elected;
  bool is_leader_in_warpgroup = ((iwarp % 4) == 0) && elected;

  __shared__ uint64_t writable[kStage];
  __shared__ uint64_t readable[kStage];

  extern __shared__ uint8_t shm_data[] alignas(128);
  auto* shm_q = reinterpret_cast<Tbf16*>(shm_data);
  auto* shm_k = reinterpret_cast<Tbf16*>(shm_q + cosize(SLayoutQ{}));
  auto* shm_logits = reinterpret_cast<Tbf16*>(shm_k + cosize(SLayoutK{}));

  auto sQ = make_tensor(make_smem_ptr(shm_q), SLayoutQ{});
  auto sK = make_tensor(make_smem_ptr(shm_k), SLayoutK{});

  // get_tma_tensor coordinate space: >= kTileM/kTileN to cover at least 1 tile.
  auto gQ =
      tma_q.get_tma_tensor(make_shape(max(max_num_qb, kTileM), kFlatDim, num_head_q, num_batch));
  auto gK = tma_k.get_tma_tensor(
      make_shape(max(max_num_kb, kTileN), kFlatDim, num_head_q / num_heads_per_kv, num_batch));

  // Identity tensor for epilogue coordinate mapping
  auto gC =
      make_tensor(make_gmem_ptr(static_cast<Tbf16*>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileN>{}), make_stride(Int<kTileN>{}, Int<1>{}));

  auto tQg = tma_q.get_slice(0).partition_S(gQ);  // (TMA, TMA_Qb, TMA_K, Hq, B)
  auto tQs = tma_q.get_slice(0).partition_D(sQ);  // (TMA, _1, _1, kStage)

  auto tKg = tma_k.get_slice(0).partition_S(gK);  // (TMA, TMA_Kb, TMA_K, Hkv, B)
  auto tKs = tma_k.get_slice(0).partition_D(sK);  // (TMA, _1, _1, kStage)

  if (is_leader_in_block) {
#pragma unroll
    for (int i = 0; i < kStage; ++i) {
      initialize_barrier(readable[i], 1);
      initialize_barrier(writable[i], size(TiledMma{}) / 128);
    }
  }

  __syncthreads();

  // Producer WG (threads 256-383)
  if (idx >= 256) {
    cutlass::arch::warpgroup_reg_dealloc<24>();
    idx -= 256;
    constexpr int kTransactionBytes =
        sizeof(Tbf16) * cosize(SLayoutQ{}(_, _, 0)) + sizeof(Tbf16) * cosize(SLayoutK{}(_, _, 0));

    int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
    int is_leader_in_load = ((iwarp == 0) && elected);

    if (is_leader_in_load) {
      int phase = 1;
      int ismem_write = __shfl_sync(0xFFFFFFFF, 0, 0);
      int iblock = blockIdx.x;
      int ntile_k = size<2>(tQg);

      while (iblock < max_total_tiles) {
        auto [ibatch, ihead_q, itile_q, itile_k] = get_next_tile<kTileM, kTileN, kStemBlockSize>(
            iblock, qk_tile_divmod, k_tile_divmod, head_q_divmod);

        int ihead_kv = ihead_q / num_heads_per_kv;

        int q_len = q_seq_lens[ibatch];
        int kv_len = kv_seq_lens[ibatch];
        int q_padded = ((q_len + kStemBlockSize - 1) / kStemBlockSize) * kStemBlockSize;
        int kv_padded = ((kv_len + kStemBlockSize - 1) / kStemBlockSize) * kStemBlockSize;
        int num_qb = q_padded / kStemBlockSize;
        int num_kb = kv_padded / kStemBlockSize;

        // Skip tiles outside this request
        if (itile_q * kTileM >= num_qb || itile_k * kTileN >= num_kb) {
          iblock += gridDim.x;
          continue;
        }

        // Causal tile-level skip
        if constexpr (kCausal) {
          int block_offset = num_kb - num_qb;
          int qb_max = itile_q * kTileM + kTileM - 1;
          int kb_min = itile_k * kTileN;
          if (qb_max + block_offset < kb_min) {
            iblock += gridDim.x;
            continue;
          }
        }

        iblock += gridDim.x;

#pragma unroll 1
        for (int ik = 0; ik < ntile_k; ++ik) {
          wait_barrier(writable[ismem_write], phase);

          cute::copy(tma_q.with(readable[ismem_write]), tQg(_, itile_q, ik, ihead_q, ibatch),
                     tQs(_, 0, 0, ismem_write));

          cute::copy(tma_k.with(readable[ismem_write]), tKg(_, itile_k, ik, ihead_kv, ibatch),
                     tKs(_, 0, 0, ismem_write));

          set_barrier_transaction_bytes(readable[ismem_write], kTransactionBytes);

          ++ismem_write;
          if (ismem_write == kStage) {
            ismem_write = 0;
            phase ^= 1;
          }
        }
      }
    }

  } else {
    // Consumer WG ×2 (threads 0-255)
    cutlass::arch::warpgroup_reg_alloc<168>();

    int idx_in_warpgroup = idx % 128;
    int iwarpgroup = idx / 128;
    int iwarp_in_warpgroup = idx_in_warpgroup / 32;
    int elected_idx_in_warpgroup = ((iwarp_in_warpgroup == 0) && elected);

    TiledMma tiled_mma;

    auto thr_mma = tiled_mma.get_slice(idx);
    auto tQs4r = thr_mma.partition_A(sQ);
    auto tKs4r = thr_mma.partition_B(sK);

    auto tQr = thr_mma.make_fragment_A(tQs4r);  // (MMA, MMA_M, MMA_K, kStage)
    auto tKr = thr_mma.make_fragment_B(tKs4r);  // (MMA, MMA_N, MMA_K, kStage)

    auto tCr = thr_mma.partition_fragment_C(gC);

    // Identity tensor for epilogue coordinate mapping
    auto gI = make_identity_tensor(gC.shape());
    auto tI = thr_mma.partition_C(gI);
    auto tCr_mn = retile_fragment(tCr);  // (M, N) view of accumulator
    auto tI_mn = retile_fragment(tI);    // (M, N) view of identity
    constexpr int kM = size<0>(tCr_mn);
    constexpr int kN = size<1>(tCr_mn);

    int ismem_read = 0;
    int phase = 0;

    int iblock = blockIdx.x;
    while (iblock < max_total_tiles) {
      auto [ibatch, ihead_q, itile_q, itile_k] = get_next_tile<kTileM, kTileN, kStemBlockSize>(
          iblock, qk_tile_divmod, k_tile_divmod, head_q_divmod);

      int ihead_kv = ihead_q / num_heads_per_kv;

      int q_len = q_seq_lens[ibatch];
      int kv_len = kv_seq_lens[ibatch];
      int q_padded = ((q_len + kStemBlockSize - 1) / kStemBlockSize) * kStemBlockSize;
      int kv_padded = ((kv_len + kStemBlockSize - 1) / kStemBlockSize) * kStemBlockSize;
      int num_qb = q_padded / kStemBlockSize;
      int num_kb = kv_padded / kStemBlockSize;

      if (itile_q * kTileM >= num_qb || itile_k * kTileN >= num_kb) {
        iblock += gridDim.x;
        continue;
      }

      if constexpr (kCausal) {
        int block_offset = num_kb - num_qb;
        int qb_max = itile_q * kTileM + kTileM - 1;
        int kb_min = itile_k * kTileN;
        if (qb_max + block_offset < kb_min) {
          iblock += gridDim.x;
          continue;
        }
      }

      iblock += gridDim.x;

      tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;

      int ntile_k = size<2>(tQg);
      int ntile_todo = ntile_k;
#pragma unroll 1
      for (; ntile_todo > 0; --ntile_todo) {
        if (elected_idx_in_warpgroup) {
          wait_barrier(readable[ismem_read], phase);
        }

        warpgroup_fence_operand(tCr);
        warpgroup_arrive();
#pragma unroll
        for (int ik = 0; ik < size<2>(tQr); ++ik) {
          cute::gemm(tiled_mma, tQr(_, _, ik, ismem_read), tKr(_, _, ik, ismem_read), tCr(_, _, _));
          tiled_mma.accumulate_ = GMMA::ScaleOut::One;
        }

        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(tCr);

        if (elected_idx_in_warpgroup) {
          arrive_barrier(writable[ismem_read]);
        }

        ++ismem_read;
        if (ismem_read == kStage) {
          phase ^= 1;
          ismem_read = 0;
        }
      }

      // Epilogue: FrobScale + V_bias + causal mask
      constexpr float kNegInf = -std::numeric_limits<float>::infinity();

      // Precompute causal boundary info for this tile
      [[maybe_unused]] int block_offset = 0;
      [[maybe_unused]] bool need_mask = false;
      if constexpr (kCausal) {
        block_offset = num_kb - num_qb;
        int qb_start = itile_q * kTileM;
        int kb_start = itile_k * kTileN;
        need_mask = (qb_start + block_offset < kb_start + kTileN) &&
                    (qb_start + kTileM - 1 + block_offset >= kb_start);
      }

      // V_bias base pointer for this (batch, head_kv)
      const float* vbias_base =
          vbias_ptr + static_cast<int64_t>(ibatch) * (num_head_q / num_heads_per_kv) * max_num_kb +
          static_cast<int64_t>(ihead_kv) * max_num_kb;

#pragma unroll
      for (int im = 0; im < kM; ++im) {
        int m_local = get<0>(tI_mn(im, 0));
        int qb_idx = itile_q * kTileM + m_local;

#pragma unroll
        for (int in = 0; in < kN; ++in) {
          int n_local = get<1>(tI_mn(im, in));
          int kb_idx = itile_k * kTileN + n_local;

          float val = tCr_mn(im, in);

          if (qb_idx >= max_num_qb || kb_idx >= max_num_kb) {
            val = kNegInf;
          } else if (qb_idx >= num_qb || kb_idx >= num_kb) {
            val = kNegInf;
          } else if constexpr (kCausal) {
            if (need_mask && (qb_idx + block_offset < kb_idx)) {
              val = kNegInf;
            } else {
              val = val * kFrobScale + vbias_base[kb_idx];
            }
          } else {
            val = val * kFrobScale + vbias_base[kb_idx];
          }

          tCr_mn(im, in) = val;
        }
      }

      // FP32 → BF16 → STSM → TMA store
      auto tCrh = make_tensor_like<Tbf16>(tCr);

#pragma unroll
      for (int i = 0; i < size(tCr); ++i) {
        tCrh(i) = static_cast<Tbf16>(tCr(i));
      }

      // STSM: registers → shared memory
      auto sLogits =
          make_tensor(make_smem_ptr(reinterpret_cast<Tbf16*>(shm_logits)), SLayoutLogits{});
      using R2SCopyAtomC = Copy_Atom<cute::SM90_U32x4_STSM_N, Tbf16>;
      auto tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
      auto thr_copy_c = tiled_copy_c.get_slice(idx);

      auto tCr4s = thr_copy_c.retile_S(tCrh);
      auto tLogitsS4r = thr_copy_c.partition_D(sLogits);

      cute::tma_store_wait<0>();
      syncwarpgroup(iwarpgroup);

      cute::copy(tiled_copy_c, tCr4s, tLogitsS4r);
      syncwarpgroup(iwarpgroup);
      cute::tma_store_fence();

      // TMA store: shared memory → global memory (1 WG stores TileM/2 rows)
      if (is_leader_in_warpgroup) {
        auto gLogits = tma_logits.get_tma_tensor(
            make_shape(max(max_num_qb, kTileM), max(max_num_kb, kTileN), num_head_q, num_batch));
        auto tLogitsS = tma_logits.get_slice(0).partition_S(sLogits);  // (TMA, _2, _1)
        auto tLogitsG =
            tma_logits.get_slice(0).partition_D(gLogits);  // (TMA, TMA_Qb, TMA_Kb, Hq, B)

        cute::copy(tma_logits, tLogitsS(_, iwarpgroup, Int<0>{}),
                   tLogitsG(_, itile_q * 2 + iwarpgroup, itile_k, ihead_q, ibatch));
        tma_store_arrive();
      }
    }
  }
}

// Warp-level int32 sum reduction; result broadcast to all 32 lanes.
__device__ __forceinline__ int warp_reduce_sum_xor_i32(int x) {
#pragma unroll
  for (int offset = 16; offset >= 1; offset /= 2) {
    x += __shfl_xor_sync(0xFFFFFFFF, x, offset);
  }
  return x;
}

// Map bf16 bits to a monotonic uint16: ordered-int comparison matches float comparison.
// Positive: flip sign bit;  Negative: invert all bits.
__device__ __forceinline__ uint16_t bf16_to_ordered(uint16_t bits) {
  return (bits & 0x8000) ? ~bits : (bits ^ 0x8000);
}

// bf16 isfinite: exponent (bits 14-7) not all ones.
__device__ __forceinline__ bool bf16_isfinite(uint16_t bits) { return (bits & 0x7F80) != 0x7F80; }

// Per-row top-k budget: 3-regime k_schedule + linspace decay, both keyed on
// the FULL prompt KV length (`prompt_kv_blocks`) so the result is chunked-
// prefill invariant. `ki_blocks` only feeds `kb_offset` for absolute q_pos.
__device__ __forceinline__ int compute_budget(int q_row, int qi_blocks, int ki_blocks,
                                              int prompt_kv_blocks, float alpha,
                                              float k_block_num_rate_medium,
                                              int k_block_num_bias_medium,
                                              float k_block_num_rate_large,
                                              int k_block_num_bias_large) {
  constexpr int kSmallSeqMax = 56;
  constexpr int kMediumSeqMax = 160;

  int k_val;
  if (prompt_kv_blocks < kSmallSeqMax) {
    k_val = prompt_kv_blocks;
  } else if (prompt_kv_blocks < kMediumSeqMax) {
    k_val = static_cast<int>(prompt_kv_blocks * k_block_num_rate_medium) + k_block_num_bias_medium;
  } else {
    k_val = static_cast<int>(prompt_kv_blocks * k_block_num_rate_large) + k_block_num_bias_large;
  }

  const int kb_offset = ki_blocks - qi_blocks;
  const int q_pos = q_row + kb_offset;

  int decay_len = prompt_kv_blocks - k_val;
  if (q_pos < k_val || decay_len <= 1) {
    return k_val;
  }

  float k_end = k_val * alpha;
  float t = static_cast<float>(q_pos - k_val) / (decay_len - 1);
  int budget = static_cast<int>(floorf(k_val + t * (k_end - k_val)));
  return budget < 1 ? 1 : (budget > k_val ? k_val : budget);
}

// Find the largest threshold T such that count(ordered >= T) >= budget.
//   For each bit MSB→LSB: probe candidate = threshold | (1<<bit); keep the bit
//   if count(ordered >= candidate) >= budget.  After 16 rounds T is exact.
// kWPR=1: pure warp shuffle (zero smem, zero sync).
// kWPR>1: warp shuffle + smem aggregation across kWPR warps.
template <int kEPT, int kWPR>
__device__ __forceinline__ uint16_t radix_find_threshold(const uint16_t ordered[kEPT], int budget,
                                                         int* smem_warp_counts, int warp_in_group) {
  const int ilane = threadIdx.x % 32;

  uint16_t threshold = 0;

#pragma unroll 1
  for (int bit = 15; bit >= 0; --bit) {
    uint16_t candidate = threshold | (1u << bit);

    int local_count = 0;
#pragma unroll
    for (int j = 0; j < kEPT; ++j) {
      local_count += (ordered[j] >= candidate);
    }

    int total;
    if constexpr (kWPR == 1) {
      total = warp_reduce_sum_xor_i32(local_count);
    } else {
      int warp_count = warp_reduce_sum_xor_i32(local_count);
      if (ilane == 0) {
        smem_warp_counts[warp_in_group] = warp_count;
      }
      __syncthreads();

      total = 0;
#pragma unroll
      for (int iw = 0; iw < kWPR; ++iw) {
        total += smem_warp_counts[iw];
      }
      __syncthreads();
    }

    // keep this bit
    if (total >= budget) {
      threshold = candidate;
    }
  }
  return threshold;
}

// stem_tpd_kernel — fused budget + radix-select + mask generation
//
// Per-row: compute budget → radix top-k threshold → apply fixed patterns → write u8 mask.
//
// Scale to wider rows by increasing kWPR, not kEPT.
// Only the extreme >1M-token variant uses kEPT=128.
//
//   kEPT   elements per thread   (controls register usage)
//   kWPR   warps per row         (1 = independent, >1 = cooperative)
//   kWPC   warps per CTA         (= kWPR when cooperative, = 8 when independent)
//
//   kWPR = 1  →  kWPC = 8,    8 rows/CTA, warp-shuffle only, no __syncthreads
//   kWPR > 1  →  kWPC = kWPR, 1 row/CTA,  cross-warp radix via smem + __syncthreads
//
// launch_bounds:
//   kWPR=1: minBlocks=0 (unspecified) → compiler targets max occupancy.
//   kWPR>1: minBlocks=1 → relaxes register pressure (max 255 regs),
//           prevents spilling; runtime still places 2+ blk/SM when regs permit.
template <int kEPT, int kWPR, int kWPC>
__global__ void __launch_bounds__(kWPC * 32, (kWPR == 1) ? 0 : 1)
    stem_tpd_kernel(const __nv_bfloat16* __restrict__ block_logits, uint8_t* __restrict__ mask,
                    const int* __restrict__ q_seq_lens, const int* __restrict__ kv_seq_lens,
                    const int* __restrict__ num_prompt_tokens, float alpha, int block_size,
                    int initial_blocks, int window_size, float k_block_num_rate_medium,
                    int k_block_num_bias_medium, float k_block_num_rate_large,
                    int k_block_num_bias_large, int num_heads, int max_Qb, int max_Kb) {
  static_assert(kWPR == 1 || kWPC == kWPR, "cooperative mode requires kWPC == kWPR");
  constexpr int kRowsPerCta = kWPC / kWPR;
  constexpr int kThreadsPerRow = kWPR * 32;
  constexpr uint16_t kNegInfBits = 0xFF80;
  constexpr uint16_t kNonFiniteOrdered = 0x007Fu;

  const int iwarp = threadIdx.x / 32;
  const int ilane = threadIdx.x % 32;

  const int row_group = (kWPR == 1) ? iwarp : (iwarp / kWPR);
  const int warp_in_group = (kWPR == 1) ? 0 : (iwarp % kWPR);

  const int q_row = blockIdx.x * kRowsPerCta + row_group;
  const int ihead = blockIdx.y;
  const int ireq = blockIdx.z;

  // Number of stem blocks for this request (matches host-side max_Qb / max_Kb derivation).
  const int qi_blocks = (q_seq_lens[ireq] + block_size - 1) / block_size;
  const int ki_blocks = (kv_seq_lens[ireq] + block_size - 1) / block_size;
  const int prompt_kv_blocks = (num_prompt_tokens[ireq] + block_size - 1) / block_size;
  if (qi_blocks == 0 || ki_blocks == 0 || q_row >= qi_blocks) {
    return;
  }

  const int64_t row_offset = static_cast<int64_t>(ireq) * num_heads * max_Qb * max_Kb +
                             static_cast<int64_t>(ihead) * max_Qb * max_Kb +
                             static_cast<int64_t>(q_row) * max_Kb;
  const __nv_bfloat16* row_ptr = block_logits + row_offset;
  uint8_t* mask_ptr = mask + row_offset;

  const int linear_id = warp_in_group * 32 + ilane;

  // Strided load: thread t reads cols t, t+kThreadsPerRow, ... → 32 lanes hit one cache line.
  // Non-finite (-inf / NaN) maps to kNonFiniteOrdered (smaller than any finite ordered value).
  uint16_t ordered[kEPT];
  int local_finite = 0;
#pragma unroll
  for (int j = 0; j < kEPT; ++j) {
    int col = linear_id + j * kThreadsPerRow;
    uint16_t bits = (col < ki_blocks) ? __bfloat16_as_ushort(row_ptr[col]) : kNegInfBits;
    bool is_finite = bf16_isfinite(bits);
    ordered[j] = is_finite ? bf16_to_ordered(bits) : kNonFiniteOrdered;
    local_finite += is_finite;
  }

  const int budget =
      compute_budget(q_row, qi_blocks, ki_blocks, prompt_kv_blocks, alpha, k_block_num_rate_medium,
                     k_block_num_bias_medium, k_block_num_rate_large, k_block_num_bias_large);

  // Sum local_finite across all threads in this row.  Reuses smem_warp_counts in radix below.
  // kWPR==1 path elides the smem allocation.
  __shared__ int smem_warp_counts[kWPR];
  int total_finite;
  if constexpr (kWPR == 1) {
    total_finite = warp_reduce_sum_xor_i32(local_finite);
  } else {
    int warp_finite = warp_reduce_sum_xor_i32(local_finite);
    if (ilane == 0) {
      smem_warp_counts[warp_in_group] = warp_finite;
    }
    __syncthreads();

    total_finite = 0;
#pragma unroll
    for (int iw = 0; iw < kWPR; ++iw) {
      total_finite += smem_warp_counts[iw];
    }
    __syncthreads();
  }

  // Skip radix when budget covers all finite values (threshold = 0x0080 selects every finite).
  uint16_t threshold = kNonFiniteOrdered + 1;
  if (budget < total_finite) {
    threshold = radix_find_threshold<kEPT, kWPR>(ordered, budget, smem_warp_counts, warp_in_group);
  }

  // mask = top-k & initial sink & recent window & diagonal.
  //
  // Chunked-prefill alignment: Q is the tail of a longer KV, so a Q-block at
  // local index q_row corresponds to KV-block index (q_row + kb_offset).
  const int kb_offset = ki_blocks - qi_blocks;
  const int diag_col = q_row + kb_offset;
#pragma unroll
  for (int j = 0; j < kEPT; ++j) {
    int col = linear_id + j * kThreadsPerRow;
    if (col >= ki_blocks) {
      continue;
    }

    bool selected = (ordered[j] >= threshold);
    selected |= (col < initial_blocks);                               // initial sink (KV head)
    selected |= (col <= diag_col) && (col > diag_col - window_size);  // recent window (KV-aligned)
    selected |= (col == diag_col);                                    // diagonal (KV-aligned)

    mask_ptr[col] = static_cast<uint8_t>(selected);
  }
}

}  // namespace kernels
}  // namespace stem
}  // namespace hpc

#endif  // SRC_STEM_STEM_KERNELS_CUH_
