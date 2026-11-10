// Copyright (C) 2026 Tencent.
// Adapted from Religious-J/warp_decode, commit 1b3fff273d00d4cad4c9f366b0a74dbc80ee2e86.
// warp_decode: gate_up kernel for blockwise FP8 MoE decode.
//
// Layout (blockwise FP8 Qwen MoE):
//   X       : [B, hidden]                  fp8 e4m3
//   W       : [E, 2 * inter, hidden]       fp8 e4m3 (gate rows, then up rows)
//   x_scale : [B, hidden / 128]            fp32
//   w_scale : [E, 2 * inter / 128, ceil(hidden / 128 / 4) * 4] fp32
//   ids     : [B, topk]                    int32
//   out     : [B * topk, inter]            bf16 (silu(gate) * up, fused)
//
// Each CTA computes eight output neurons: two warps with four outputs each
// at B=7/8, otherwise four warps with two outputs each. Every warp reuses one
// decoded input vector across its gate/up weight rows.
// FP8 pairs are widened before FP32 accumulation; each vector stays within
// one 128-element scale group. Gate/up rounds to BF16 before activation.
// Global expert IDs outside this EP rank are ignored and produce zero rows.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

#include "src/fuse_moe/fuse_moe_warp_decode.h"

namespace hpc {
namespace fuse_moe {
namespace warp_decode {
namespace {

constexpr int SCALE_GROUP = 128;
constexpr int WARPS_PER_CTA = 4;
constexpr int THREADS_PER_CTA = WARPS_PER_CTA * 32;
constexpr int VEC = 16;  // 16 FP8 values == 16 B == uint4
constexpr int OUTS_PER_WARP = 2;

static_assert(SCALE_GROUP % OUTS_PER_WARP == 0);

using fp8 = __nv_fp8_e4m3;

// Prioritize activations and scales over streaming weight rows in L1.
__device__ __forceinline__ uint4 load_weight_vector(const fp8* ptr) {
  uint4 value;
  asm("ld.global.L1::evict_first.v4.u32 {%0, %1, %2, %3}, [%4];"
      : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
      : "l"(ptr));
  return value;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
  // full warp reduction via xor butterfly
  v += __shfl_xor_sync(0xffffffff, v, 16);
  v += __shfl_xor_sync(0xffffffff, v, 8);
  v += __shfl_xor_sync(0xffffffff, v, 4);
  v += __shfl_xor_sync(0xffffffff, v, 2);
  v += __shfl_xor_sync(0xffffffff, v, 1);
  return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
  // Local fmax reduction supplies nonnegative values; unsigned FP32 bits
  // preserve their ordering, including +infinity and the zero/tiny cases.
  return __uint_as_float(__reduce_max_sync(0xffffffff, __float_as_uint(v)));
}

__device__ __forceinline__ float2 unpack_fp8_pair(uint16_t packed) {
  return __half22float2(static_cast<__half2>(__nv_cvt_fp8x2_to_halfraw2(packed, __NV_E4M3)));
}

__device__ __forceinline__ float silu_f32(float x) { return __fdividef(x, 1.0f + __expf(-x)); }

__global__ void quantize_fp8_blockwise_kernel(const __nv_bfloat16* X,  // [rows, cols]
                                              fp8* out,                // [rows, cols]
                                              float* scale,            // [rows, cols / SCALE_GROUP]
                                              int rows, int cols) {
  const int lane_id = threadIdx.x & 31;
  const size_t group = (size_t)blockIdx.x * WARPS_PER_CTA + threadIdx.x / 32;
  if (group >= (size_t)rows * (cols / SCALE_GROUP)) return;

  // Each warp quantizes one group; four adjacent BF16 elements per lane.
  const size_t offset = group * SCALE_GROUP + lane_id * 4;
  uint2 packed = *reinterpret_cast<const uint2*>(X + offset);
  const auto* values = reinterpret_cast<const __nv_bfloat16*>(&packed);
  float xf[4];
  float max_abs = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    xf[i] = __bfloat162float(values[i]);
    max_abs = fmaxf(max_abs, fabsf(xf[i]));
  }

  const float group_scale = __fdiv_rn(fmaxf(warp_reduce_max(max_abs), 1.e-10f), 448.f);
  if (lane_id == 0) scale[group] = group_scale;
  const float inv_scale = __fdiv_rn(1.f, group_scale);
  __nv_fp8x2_e4m3 lo(make_float2(xf[0] * inv_scale, xf[1] * inv_scale));
  __nv_fp8x2_e4m3 hi(make_float2(xf[2] * inv_scale, xf[3] * inv_scale));
  *reinterpret_cast<uint32_t*>(out + offset) = uint32_t(lo.__x) | (uint32_t(hi.__x) << 16);
}

template <bool PREFETCH_NEXT, int OUTS = OUTS_PER_WARP, int GATE_WARPS = WARPS_PER_CTA,
          int STATIC_HIDDEN = 0, int STATIC_INTER = 0, int STATIC_EXPERTS = 0, int STATIC_TOPK = 0,
          bool FLAT_ROUTES = (STATIC_TOPK != 0), bool CACHE_FACTORS = false>
__global__ void moe_gate_up_3d_batched_kernel(
    const fp8* X,          // [B, hidden]
    const fp8* W,          // [E, 2*inter, hidden]
    const float* x_scale,  // [B, hidden / SCALE_GROUP]
    const float* w_scale,  // [E, 2*inter / SCALE_GROUP, padded K-scale columns]
    const int* topk_ids,   // [B, topk]
    __nv_bfloat16* out,    // [B*topk, inter]
    int hidden, int inter, int num_experts, int topk, int expert_start) {
  static_assert(SCALE_GROUP % OUTS == 0);
  static_assert(!CACHE_FACTORS || (STATIC_HIDDEN > 0 && STATIC_HIDDEN <= 4096));
  const int HIDDEN_SIZE = STATIC_HIDDEN ? STATIC_HIDDEN : hidden;
  const int INTER_SIZE = STATIC_INTER ? STATIC_INTER : inter;
  const int NUM_EXPERTS = STATIC_EXPERTS ? STATIC_EXPERTS : num_experts;
  const int TOPK = STATIC_TOPK ? STATIC_TOPK : topk;
  const int token_id = FLAT_ROUTES ? blockIdx.x / TOPK : blockIdx.x;
  const int flat_tok = FLAT_ROUTES ? blockIdx.x : token_id * TOPK + blockIdx.z;
  const int warp_id = threadIdx.x >> 5;
  const int lane_id = threadIdx.x & 31;
  const int neuron_n = (blockIdx.y * GATE_WARPS + warp_id) * OUTS;
  const int global_expert_id = topk_ids[flat_tok];

  if (global_expert_id < expert_start || global_expert_id >= expert_start + NUM_EXPERTS) {
    if (lane_id == 0) {
#pragma unroll
      for (int j = 0; j < OUTS; ++j) {
        out[(size_t)flat_tok * INTER_SIZE + neuron_n + j] = __float2bfloat16_rn(0.f);
      }
    }
    return;
  }

  const int expert_id = global_expert_id - expert_start;
  const int weight_scale_cols = (HIDDEN_SIZE / SCALE_GROUP + 3) / 4 * 4;
  const fp8* x_ptr = X + (size_t)token_id * HIDDEN_SIZE;
  const size_t expert_stride = (size_t)(2 * INTER_SIZE) * HIDDEN_SIZE;
  const fp8* gate_ptr = W + (size_t)expert_id * expert_stride + (size_t)neuron_n * HIDDEN_SIZE;
  const fp8* up_ptr = gate_ptr + (size_t)INTER_SIZE * HIDDEN_SIZE;
  const float* gate_scale =
      w_scale + ((size_t)expert_id * (2 * INTER_SIZE / SCALE_GROUP) + neuron_n / SCALE_GROUP) *
                    weight_scale_cols;
  const float* up_scale = gate_scale + (INTER_SIZE / SCALE_GROUP) * weight_scale_cols;

  float acc_gate[OUTS]{};
  float acc_up[OUTS]{};

  uint4 pref_g, pref_u;
  if constexpr (PREFETCH_NEXT) {
    if (lane_id * VEC < HIDDEN_SIZE) {
      pref_g = load_weight_vector(&gate_ptr[lane_id * VEC]);
      pref_u = load_weight_vector(&up_ptr[lane_id * VEC]);
    }
  }

  // Each lane holds one K-scale group's factors. The selected shape has
  // 32 groups, so K iterations can reuse them through warp shuffles.
  float cached_gate_factor = 0.f, cached_up_factor = 0.f;
  if constexpr (CACHE_FACTORS) {
    if (lane_id < HIDDEN_SIZE / SCALE_GROUP) {
      const float scale = x_scale[(size_t)token_id * (HIDDEN_SIZE / SCALE_GROUP) + lane_id];
      cached_gate_factor = scale * gate_scale[lane_id];
      cached_up_factor = scale * up_scale[lane_id];
    }
  }

#pragma unroll
  for (int k = lane_id * VEC; k < HIDDEN_SIZE; k += 32 * VEC) {
    uint4 current_g, current_u;
    if constexpr (PREFETCH_NEXT) {
      current_g = pref_g;
      current_u = pref_u;
    }
    uint4 xv = *reinterpret_cast<const uint4*>(&x_ptr[k]);
    if constexpr (PREFETCH_NEXT) {
      // Overlap the next first-output weight loads with this K block.
      if (k + 32 * VEC < HIDDEN_SIZE) {
        pref_g = load_weight_vector(&gate_ptr[k + 32 * VEC]);
        pref_u = load_weight_vector(&up_ptr[k + 32 * VEC]);
      }
    }
    const auto* x_pairs = reinterpret_cast<const uint16_t*>(&xv);
    float2 xf[VEC / 2];
#pragma unroll
    for (int i = 0; i < VEC / 2; ++i) xf[i] = unpack_fp8_pair(x_pairs[i]);
    float gate_factor, up_factor;
    if constexpr (CACHE_FACTORS) {
      gate_factor = __shfl_sync(0xffffffff, cached_gate_factor, k / SCALE_GROUP);
      up_factor = __shfl_sync(0xffffffff, cached_up_factor, k / SCALE_GROUP);
    } else {
      const float scale = x_scale[(size_t)token_id * (HIDDEN_SIZE / SCALE_GROUP) + k / SCALE_GROUP];
      gate_factor = scale * gate_scale[k / SCALE_GROUP];
      up_factor = scale * up_scale[k / SCALE_GROUP];
    }

#pragma unroll
    for (int j = 0; j < OUTS; ++j) {
      uint4 gv, uv;
      if (PREFETCH_NEXT && j == 0) {
        gv = current_g;
        uv = current_u;
      } else {
        gv = load_weight_vector(&gate_ptr[j * HIDDEN_SIZE + k]);
        uv = load_weight_vector(&up_ptr[j * HIDDEN_SIZE + k]);
      }
      const auto* g_pairs = reinterpret_cast<const uint16_t*>(&gv);
      const auto* u_pairs = reinterpret_cast<const uint16_t*>(&uv);
      float local_gate = 0.f;
      float local_up = 0.f;

#pragma unroll
      for (int i = 0; i < VEC / 2; ++i) {
        const float2 gf = unpack_fp8_pair(g_pairs[i]);
        const float2 uf = unpack_fp8_pair(u_pairs[i]);
        local_gate = fmaf(xf[i].x, gf.x, local_gate);
        local_gate = fmaf(xf[i].y, gf.y, local_gate);
        local_up = fmaf(xf[i].x, uf.x, local_up);
        local_up = fmaf(xf[i].y, uf.y, local_up);
      }
      acc_gate[j] = fmaf(local_gate, gate_factor, acc_gate[j]);
      acc_up[j] = fmaf(local_up, up_factor, acc_up[j]);
    }
  }

#pragma unroll
  for (int j = 0; j < OUTS; ++j) {
    acc_gate[j] = warp_reduce_sum(acc_gate[j]);
    acc_up[j] = warp_reduce_sum(acc_up[j]);
  }

  if (lane_id == 0) {
#pragma unroll
    for (int j = 0; j < OUTS; ++j) {
      // Match the BF16 GEMM boundary before SGLang's SiLU-and-multiply.
      const float gate = __bfloat162float(__float2bfloat16_rn(acc_gate[j]));
      const float up = __bfloat162float(__float2bfloat16_rn(acc_up[j]));
      const float y = silu_f32(gate) * up;
      out[(size_t)flat_tok * INTER_SIZE + neuron_n + j] = __float2bfloat16_rn(y);
    }
  }
}

}  // anonymous namespace

void quantize_async(const void* X, void* out, float* scale, int rows, int cols,
                    cudaStream_t stream) {
  if (!rows || !cols) return;
  const size_t total_groups = (size_t)rows * (cols / SCALE_GROUP);
  dim3 grid((total_groups + WARPS_PER_CTA - 1) / WARPS_PER_CTA);
  dim3 block(THREADS_PER_CTA);
  quantize_fp8_blockwise_kernel<<<grid, block, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(X), static_cast<fp8*>(out), scale, rows, cols);
}

namespace {

template <int STATIC_HIDDEN, int STATIC_INTER = 0, int STATIC_EXPERTS = 0, int STATIC_TOPK = 0,
          bool FLAT_ROUTES = (STATIC_TOPK != 0), bool CACHE_FACTORS = false>
void launch_gate_for_shape(const void* X, const void* W, const float* x_scale, const float* w_scale,
                           const int* topk_ids, void* out, int batch_size, int hidden, int inter,
                           int num_experts, int topk, int expert_start, cudaStream_t stream) {
  const auto* input = static_cast<const fp8*>(X);
  const auto* weights = static_cast<const fp8*>(W);
  auto* output = static_cast<__nv_bfloat16*>(out);

  // Both warp/output tilings produce eight neurons per CTA.
  // Only the original 35B path flattens routes; other paths use grid.z for
  // the expert slot, independently of whether top-k is a compile-time value.
  const dim3 grid(FLAT_ROUTES ? batch_size * STATIC_TOPK : batch_size,
                  inter / (WARPS_PER_CTA * OUTS_PER_WARP), FLAT_ROUTES ? 1 : topk);
  if (batch_size == 7 || batch_size == 8) {
    moe_gate_up_3d_batched_kernel<false, 4, 2, STATIC_HIDDEN, STATIC_INTER, STATIC_EXPERTS,
                                  STATIC_TOPK, FLAT_ROUTES, CACHE_FACTORS>
        <<<grid, 2 * 32, 0, stream>>>(input, weights, x_scale, w_scale, topk_ids, output, hidden,
                                      inter, num_experts, topk, expert_start);
  } else if (batch_size <= 3) {
    moe_gate_up_3d_batched_kernel<true, 2, 4, STATIC_HIDDEN, STATIC_INTER, STATIC_EXPERTS,
                                  STATIC_TOPK, FLAT_ROUTES, CACHE_FACTORS>
        <<<grid, THREADS_PER_CTA, 0, stream>>>(input, weights, x_scale, w_scale, topk_ids, output,
                                               hidden, inter, num_experts, topk, expert_start);
  } else {
    moe_gate_up_3d_batched_kernel<false, 2, 4, STATIC_HIDDEN, STATIC_INTER, STATIC_EXPERTS,
                                  STATIC_TOPK, FLAT_ROUTES, CACHE_FACTORS>
        <<<grid, THREADS_PER_CTA, 0, stream>>>(input, weights, x_scale, w_scale, topk_ids, output,
                                               hidden, inter, num_experts, topk, expert_start);
  }
}

}  // anonymous namespace

void gate_up_async(const void* X, const void* W, const float* x_scale, const float* w_scale,
                   const int* topk_ids, void* out, int batch_size, int hidden, int inter,
                   int num_experts, int topk, int expert_start, cudaStream_t stream) {
  if (!batch_size) return;

  // Full shape constants remove runtime stride/index arithmetic. Preserve
  // the measured batch ranges and the original grid mapping for each path.
  if (hidden == 2048 && inter == 512 && num_experts == 256 && topk == 8) {
    launch_gate_for_shape<2048, 512, 256, 8>(X, W, x_scale, w_scale, topk_ids, out, batch_size,
                                             hidden, inter, num_experts, topk, expert_start,
                                             stream);
  } else if (batch_size <= 8 && hidden == 3072 && num_experts == 256 && topk == 8 &&
             (inter == 512 || (inter == 256 && batch_size <= 6))) {
    if (inter == 512) {
      launch_gate_for_shape<3072, 512, 256, 8, false, false>(X, W, x_scale, w_scale, topk_ids, out,
                                                             batch_size, hidden, inter, num_experts,
                                                             topk, expert_start, stream);
    } else {
      launch_gate_for_shape<3072, 256, 256, 8, false, false>(X, W, x_scale, w_scale, topk_ids, out,
                                                             batch_size, hidden, inter, num_experts,
                                                             topk, expert_start, stream);
    }
  } else if (batch_size >= 7 && batch_size <= 8 && hidden == 2048 && inter == 256 &&
             num_experts == 256 && topk == 8) {
    launch_gate_for_shape<2048, 256, 256, 8, false, false>(X, W, x_scale, w_scale, topk_ids, out,
                                                           batch_size, hidden, inter, num_experts,
                                                           topk, expert_start, stream);
  } else if (batch_size <= 8 && (batch_size <= 4 || batch_size >= 6) && hidden == 2048 &&
             num_experts == 512 && topk == 10 && (inter == 256 || inter == 512)) {
    if (inter == 512) {
      launch_gate_for_shape<2048, 512, 512, 10, false, false>(
          X, W, x_scale, w_scale, topk_ids, out, batch_size, hidden, inter, num_experts, topk,
          expert_start, stream);
    } else {
      launch_gate_for_shape<2048, 256, 512, 10, false, false>(
          X, W, x_scale, w_scale, topk_ids, out, batch_size, hidden, inter, num_experts, topk,
          expert_start, stream);
    }
  } else if (batch_size <= 6 && hidden == 4096 && inter == 128 && num_experts == 512 &&
             topk == 10) {
    launch_gate_for_shape<4096, 128, 512, 10, false, true>(X, W, x_scale, w_scale, topk_ids, out,
                                                           batch_size, hidden, inter, num_experts,
                                                           topk, expert_start, stream);
  } else {
    // Common Qwen hidden sizes retain a compile-time K loop. The fallback
    // handles any positive 128-aligned hidden/intermediate dimensions.
    switch (hidden) {
      case 2048:
        launch_gate_for_shape<2048>(X, W, x_scale, w_scale, topk_ids, out, batch_size, hidden,
                                    inter, num_experts, topk, expert_start, stream);
        break;
      case 3072:
        launch_gate_for_shape<3072>(X, W, x_scale, w_scale, topk_ids, out, batch_size, hidden,
                                    inter, num_experts, topk, expert_start, stream);
        break;
      case 4096:
        launch_gate_for_shape<4096>(X, W, x_scale, w_scale, topk_ids, out, batch_size, hidden,
                                    inter, num_experts, topk, expert_start, stream);
        break;
      default:
        launch_gate_for_shape<0>(X, W, x_scale, w_scale, topk_ids, out, batch_size, hidden, inter,
                                 num_experts, topk, expert_start, stream);
    }
  }
}

}  // namespace warp_decode
}  // namespace fuse_moe
}  // namespace hpc
