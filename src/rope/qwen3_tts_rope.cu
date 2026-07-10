// Copyright (C) 2026 Tencent.

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <stdint.h>

#include "src/rope/rope.h"

namespace hpc {
namespace rope {
namespace qwen3_tts_rope_kernels {

constexpr int kHeadDim = 128;
constexpr int kHalfDim = kHeadDim / 2;
constexpr int kRowsPerBlockSmall = 4;
constexpr int kRowsPerBlockLarge = 8;
constexpr int kThreadsPerRowSmall = 64;
constexpr int kThreadsPerRowLarge = 32;
constexpr unsigned kWarpMask = 0xffffffffU;

__device__ __forceinline__ int64_t shfl_sync_i64(int64_t value, int src_lane) {
  const uint64_t bits = static_cast<uint64_t>(value);
  const uint32_t lo = static_cast<uint32_t>(bits);
  const uint32_t hi = static_cast<uint32_t>(bits >> 32);
  const uint32_t shuffled_lo = __shfl_sync(kWarpMask, lo, src_lane);
  const uint32_t shuffled_hi = __shfl_sync(kWarpMask, hi, src_lane);
  return static_cast<int64_t>((static_cast<uint64_t>(shuffled_hi) << 32) | shuffled_lo);
}

template <int kThreadsPerRow>
__device__ __forceinline__ void apply_rope_row_scalar(__nv_bfloat16 *out_row,
                                                      const __nv_bfloat16 *in_row,
                                                      const __nv_bfloat16 *cos_row,
                                                      const __nv_bfloat16 *sin_row, int lane) {
  static_assert(kHalfDim % kThreadsPerRow == 0, "threads per row must divide half dim");
  constexpr int kPairsPerThread = kHalfDim / kThreadsPerRow;
#pragma unroll
  for (int r = 0; r < kPairsPerThread; ++r) {
    const int i = lane + r * kThreadsPerRow;
    const float x1 = __bfloat162float(in_row[i]);
    const float x2 = __bfloat162float(in_row[i + kHalfDim]);
    const float c1 = __bfloat162float(cos_row[i]);
    const float s1v = __bfloat162float(sin_row[i]);
    const float c2 = __bfloat162float(cos_row[i + kHalfDim]);
    const float s2v = __bfloat162float(sin_row[i + kHalfDim]);
    out_row[i] = __float2bfloat16(x1 * c1 - x2 * s1v);
    out_row[i + kHalfDim] = __float2bfloat16(x2 * c2 + x1 * s2v);
  }
}

__device__ __forceinline__ void apply_rope_row_bf16x2(__nv_bfloat16 *out_row,
                                                      const __nv_bfloat16 *in_row,
                                                      const __nv_bfloat16 *cos_row,
                                                      const __nv_bfloat16 *sin_row, int lane) {
  const int i = lane * 2;
  const float2 x1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(in_row + i));
  const float2 x2 =
      __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(in_row + i + kHalfDim));
  const float2 c1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(cos_row + i));
  const float2 s1v = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(sin_row + i));
  const float2 c2 =
      __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(cos_row + i + kHalfDim));
  const float2 s2v =
      __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(sin_row + i + kHalfDim));
  *reinterpret_cast<__nv_bfloat162 *>(out_row + i) =
      __floats2bfloat162_rn(x1.x * c1.x - x2.x * s1v.x, x1.y * c1.y - x2.y * s1v.y);
  *reinterpret_cast<__nv_bfloat162 *>(out_row + i + kHalfDim) =
      __floats2bfloat162_rn(x2.x * c2.x + x1.x * s2v.x, x2.y * c2.y + x1.y * s2v.y);
}

template <int kRowsPerBlock, int kThreadsPerRow>
__global__ void qwen3_tts_rope_kernel(__nv_bfloat16 *q_out, __nv_bfloat16 *k_out,
                                      const __nv_bfloat16 *q, const __nv_bfloat16 *k,
                                      const __nv_bfloat16 *cos, const __nv_bfloat16 *sin,
                                      Qwen3TtsRopeParams params) {
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * kRowsPerBlock + threadIdx.x / kThreadsPerRow;
  const int lane = threadIdx.x % kThreadsPerRow;
  if (row >= params.total_rows) {
    return;
  }

  if (row < params.total_q_rows) {
    const int64_t s = row % params.seq_len;
    const int64_t tmp = row / params.seq_len;
    const int64_t h = tmp % params.num_q_heads;
    const int64_t b = tmp / params.num_q_heads;
    const __nv_bfloat16 *q_row = q + b * params.q_s0 + h * params.q_s1 + s * params.q_s2;
    __nv_bfloat16 *out_row = q_out + row * kHeadDim;
    const __nv_bfloat16 *cos_row = cos + b * params.cos_s0 + s * params.cos_s1;
    const __nv_bfloat16 *sin_row = sin + b * params.sin_s0 + s * params.sin_s1;
    apply_rope_row_scalar<kThreadsPerRow>(out_row, q_row, cos_row, sin_row, lane);
  } else {
    const int64_t krow = row - params.total_q_rows;
    const int64_t s = krow % params.seq_len;
    const int64_t tmp = krow / params.seq_len;
    const int64_t h = tmp % params.num_kv_heads;
    const int64_t b = tmp / params.num_kv_heads;
    const __nv_bfloat16 *k_row = k + b * params.k_s0 + h * params.k_s1 + s * params.k_s2;
    __nv_bfloat16 *out_row = k_out + krow * kHeadDim;
    const __nv_bfloat16 *cos_row = cos + b * params.cos_s0 + s * params.cos_s1;
    const __nv_bfloat16 *sin_row = sin + b * params.sin_s0 + s * params.sin_s1;
    apply_rope_row_scalar<kThreadsPerRow>(out_row, k_row, cos_row, sin_row, lane);
  }
}

template <int kRowsPerBlock>
__global__ void qwen3_tts_rope_bf16x2_kernel(__nv_bfloat16 *q_out, __nv_bfloat16 *k_out,
                                             const __nv_bfloat16 *q, const __nv_bfloat16 *k,
                                             const __nv_bfloat16 *cos, const __nv_bfloat16 *sin,
                                             Qwen3TtsRopeParams params) {
  constexpr int kThreadsPerRow = kThreadsPerRowLarge;
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * kRowsPerBlock + threadIdx.x / kThreadsPerRow;
  const int lane = threadIdx.x % kThreadsPerRow;
  if (row >= params.total_rows) {
    return;
  }

  if (row < params.total_q_rows) {
    int64_t s = 0;
    int64_t h = 0;
    int64_t b = 0;
    if (lane == 0) {
      s = row % params.seq_len;
      const int64_t tmp = row / params.seq_len;
      h = tmp % params.num_q_heads;
      b = tmp / params.num_q_heads;
    }
    s = shfl_sync_i64(s, 0);
    h = shfl_sync_i64(h, 0);
    b = shfl_sync_i64(b, 0);
    const __nv_bfloat16 *q_row = q + b * params.q_s0 + h * params.q_s1 + s * params.q_s2;
    __nv_bfloat16 *out_row = q_out + row * kHeadDim;
    const __nv_bfloat16 *cos_row = cos + b * params.cos_s0 + s * params.cos_s1;
    const __nv_bfloat16 *sin_row = sin + b * params.sin_s0 + s * params.sin_s1;
    apply_rope_row_bf16x2(out_row, q_row, cos_row, sin_row, lane);
  } else {
    const int64_t krow = row - params.total_q_rows;
    int64_t s = 0;
    int64_t h = 0;
    int64_t b = 0;
    if (lane == 0) {
      s = krow % params.seq_len;
      const int64_t tmp = krow / params.seq_len;
      h = tmp % params.num_kv_heads;
      b = tmp / params.num_kv_heads;
    }
    s = shfl_sync_i64(s, 0);
    h = shfl_sync_i64(h, 0);
    b = shfl_sync_i64(b, 0);
    const __nv_bfloat16 *k_row = k + b * params.k_s0 + h * params.k_s1 + s * params.k_s2;
    __nv_bfloat16 *out_row = k_out + krow * kHeadDim;
    const __nv_bfloat16 *cos_row = cos + b * params.cos_s0 + s * params.cos_s1;
    const __nv_bfloat16 *sin_row = sin + b * params.sin_s0 + s * params.sin_s1;
    apply_rope_row_bf16x2(out_row, k_row, cos_row, sin_row, lane);
  }
}

}  // namespace qwen3_tts_rope_kernels

namespace {

uint32_t grid_x_for_rows(int64_t total_rows, int rows_per_block) {
  return static_cast<uint32_t>((total_rows + rows_per_block - 1) / rows_per_block);
}

bool is_aligned_4(const void *ptr) { return (reinterpret_cast<uintptr_t>(ptr) & 0x3U) == 0; }

bool is_even(int64_t value) { return (value & 1) == 0; }

bool can_use_bf16x2(const __nv_bfloat16 *q_out, const __nv_bfloat16 *k_out, const __nv_bfloat16 *q,
                    const __nv_bfloat16 *k, const __nv_bfloat16 *cos, const __nv_bfloat16 *sin,
                    const Qwen3TtsRopeParams &params) {
  return is_aligned_4(q_out) && is_aligned_4(k_out) && is_aligned_4(q) && is_aligned_4(k) &&
         is_aligned_4(cos) && is_aligned_4(sin) && is_even(params.q_s0) && is_even(params.q_s1) &&
         is_even(params.q_s2) && is_even(params.k_s0) && is_even(params.k_s1) &&
         is_even(params.k_s2) && is_even(params.cos_s0) && is_even(params.cos_s1) &&
         is_even(params.sin_s0) && is_even(params.sin_s1);
}

}  // namespace

void qwen3_tts_rope_async(__nv_bfloat16 *q_out, __nv_bfloat16 *k_out, const __nv_bfloat16 *q,
                          const __nv_bfloat16 *k, const __nv_bfloat16 *cos,
                          const __nv_bfloat16 *sin, const Qwen3TtsRopeParams &params,
                          cudaStream_t stream) {
  if (params.batch_size <= 1) {
    dim3 grid(grid_x_for_rows(params.total_rows, qwen3_tts_rope_kernels::kRowsPerBlockSmall));
    dim3 block(qwen3_tts_rope_kernels::kThreadsPerRowSmall *
               qwen3_tts_rope_kernels::kRowsPerBlockSmall);
    qwen3_tts_rope_kernels::qwen3_tts_rope_kernel<qwen3_tts_rope_kernels::kRowsPerBlockSmall,
                                                  qwen3_tts_rope_kernels::kThreadsPerRowSmall>
        <<<grid, block, 0, stream>>>(q_out, k_out, q, k, cos, sin, params);
  } else {
    dim3 grid(grid_x_for_rows(params.total_rows, qwen3_tts_rope_kernels::kRowsPerBlockLarge));
    dim3 block(qwen3_tts_rope_kernels::kThreadsPerRowLarge *
               qwen3_tts_rope_kernels::kRowsPerBlockLarge);
    if (can_use_bf16x2(q_out, k_out, q, k, cos, sin, params)) {
      qwen3_tts_rope_kernels::qwen3_tts_rope_bf16x2_kernel<
          qwen3_tts_rope_kernels::kRowsPerBlockLarge>
          <<<grid, block, 0, stream>>>(q_out, k_out, q, k, cos, sin, params);
    } else {
      qwen3_tts_rope_kernels::qwen3_tts_rope_kernel<qwen3_tts_rope_kernels::kRowsPerBlockLarge,
                                                    qwen3_tts_rope_kernels::kThreadsPerRowLarge>
          <<<grid, block, 0, stream>>>(q_out, k_out, q, k, cos, sin, params);
    }
  }
}

}  // namespace rope
}  // namespace hpc
