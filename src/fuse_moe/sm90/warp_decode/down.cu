// Copyright (C) 2026 Tencent.
// Adapted from Religious-J/warp_decode, commit 1b3fff273d00d4cad4c9f366b0a74dbc80ee2e86.
// warp_decode: down kernel for blockwise FP8 MoE decode.
//
// Layout (blockwise FP8 Qwen MoE):
//   buffer       : [B*topk, inter]             fp8 e4m3, or bf16 with inline quantization
//   W_down       : [E, hidden, inter]          fp8 e4m3
//   buffer_scale : [B*topk, inter / 128]       fp32; unused with inline quantization
//   w_scale      : [E, hidden / 128, ceil(inter / 128 / 4) * 4] fp32
//   topk_ids     : [B, topk]                   int32
//   topk_scale   : [B, topk]                   fp32
//   out          : [B, hidden]                bf16
//
// Subgroups reuse each decoded input vector across four or eight output channels.
// Narrow intermediate sizes use 8-lane groups with parallel experts;
// the original 2048/512/256/topk8 shape retains its tuned full-warp path.
// All dot products, scaling, shared output addition and expert reductions use FP32.
// Global expert IDs outside this EP rank are ignored.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

#include "src/fuse_moe/fuse_moe_warp_decode.h"

namespace hpc {
namespace fuse_moe {
namespace warp_decode {
namespace {

constexpr int HIDDEN_SIZE = 2048;
constexpr int INTER_SIZE = 512;
constexpr int NUM_EXPERTS = 256;
constexpr int TOPK = 8;
constexpr int SCALE_GROUP = 128;
constexpr int WARP_SIZE_DOWN = 32;
constexpr int WARPS_PER_CTA_DOWN = 4;
// Four outputs per full warp in the original kernel; also the default
// output count per subgroup in the general kernel.
constexpr int OUTS_PER_WARP = 4;
constexpr int VEC_DOWN = 16;

using fp8 = __nv_fp8_e4m3;

__device__ __forceinline__ float warp_reduce_sum_d(float v) {
  v += __shfl_xor_sync(0xffffffff, v, 16);
  v += __shfl_xor_sync(0xffffffff, v, 8);
  v += __shfl_xor_sync(0xffffffff, v, 4);
  v += __shfl_xor_sync(0xffffffff, v, 2);
  v += __shfl_xor_sync(0xffffffff, v, 1);
  return v;
}

__device__ __forceinline__ float2 unpack_fp8_pair_d(uint16_t packed) {
  return __half22float2(static_cast<__half2>(__nv_cvt_fp8x2_to_halfraw2(packed, __NV_E4M3)));
}

__device__ __forceinline__ void store_output(__nv_bfloat16* out, const __nv_bfloat16* shared_output,
                                             size_t index, float value) {
  if (shared_output != nullptr) value += __bfloat162float(shared_output[index]);
  out[index] = __float2bfloat16_rn(value);
}

// Original 2048/512/256/topk8 kernel: each full warp computes four outputs.
template <bool SPLIT_TOPK, bool EVICT_FIRST = false>
__global__ void moe_down_3d_batched_kernel(const fp8* buffer, const fp8* W,
                                           const float* buffer_scale, const float* w_scale,
                                           const int* topk_ids, const float* topk_scale,
                                           __nv_bfloat16* out, const __nv_bfloat16* shared_output,
                                           int expert_start) {
  const int warp_id = threadIdx.x / WARP_SIZE_DOWN;
  const int lane_id = threadIdx.x & (WARP_SIZE_DOWN - 1);
  const int token_id = blockIdx.x;
  constexpr int GROUPS_PER_CTA = SPLIT_TOPK ? 1 : WARPS_PER_CTA_DOWN;
  const int group_idx = blockIdx.y * GROUPS_PER_CTA + (SPLIT_TOPK ? 0 : warp_id);
  const int h_base = group_idx * OUTS_PER_WARP;
  const int k = lane_id * VEC_DOWN;
  const int scale_group = k / SCALE_GROUP;
  float acc[OUTS_PER_WARP]{};

#pragma unroll 1
  for (int i = 0; i < (SPLIT_TOPK ? 1 : TOPK); ++i) {
    const int s = SPLIT_TOPK ? warp_id : i;
    const int flat_tok = token_id * TOPK + s;
    const int global_expert_id = topk_ids[flat_tok];
    if (global_expert_id < expert_start || global_expert_id >= expert_start + NUM_EXPERTS) {
      continue;
    }
    const int expert_id = global_expert_id - expert_start;

    const fp8* x_ptr = buffer + (size_t)flat_tok * INTER_SIZE + k;
    const fp8* w_ptr = W + ((size_t)expert_id * HIDDEN_SIZE + h_base) * INTER_SIZE + k;
    const uint4 xv = *reinterpret_cast<const uint4*>(x_ptr);
    const uint32_t xpacked[4] = {xv.x, xv.y, xv.z, xv.w};
    float x[VEC_DOWN];
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      const float2 lo = unpack_fp8_pair_d(uint16_t(xpacked[p]));
      const float2 hi = unpack_fp8_pair_d(uint16_t(xpacked[p] >> 16));
      x[4 * p] = lo.x;
      x[4 * p + 1] = lo.y;
      x[4 * p + 2] = hi.x;
      x[4 * p + 3] = hi.y;
    }
    const size_t weight_scale_idx =
        ((size_t)expert_id * (HIDDEN_SIZE / SCALE_GROUP) + h_base / SCALE_GROUP) *
            (INTER_SIZE / SCALE_GROUP) +
        scale_group;
    const float scale = buffer_scale[(size_t)flat_tok * (INTER_SIZE / SCALE_GROUP) + scale_group] *
                        w_scale[weight_scale_idx];

#pragma unroll
    for (int j = 0; j < OUTS_PER_WARP; ++j) {
      uint4 wv;
      if constexpr (EVICT_FIRST) {
        asm("ld.global.L1::evict_first.v4.u32 {%0, %1, %2, %3}, [%4];"
            : "=r"(wv.x), "=r"(wv.y), "=r"(wv.z), "=r"(wv.w)
            : "l"(w_ptr + j * INTER_SIZE));
      } else {
        wv = *reinterpret_cast<const uint4*>(w_ptr + j * INTER_SIZE);
      }
      const uint32_t wpacked[4] = {wv.x, wv.y, wv.z, wv.w};
      float dot = 0.f;
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        const float2 lo = unpack_fp8_pair_d(uint16_t(wpacked[p]));
        const float2 hi = unpack_fp8_pair_d(uint16_t(wpacked[p] >> 16));
        dot = fmaf(x[4 * p], lo.x, dot);
        dot = fmaf(x[4 * p + 1], lo.y, dot);
        dot = fmaf(x[4 * p + 2], hi.x, dot);
        dot = fmaf(x[4 * p + 3], hi.y, dot);
      }
      const float scaled = dot * scale;
      if constexpr (SPLIT_TOPK) {
        acc[j] = scaled;
      } else {
        acc[j] = fmaf(scaled, topk_scale[flat_tok], acc[j]);
      }
    }
  }

#pragma unroll
  for (int j = 0; j < OUTS_PER_WARP; ++j) acc[j] = warp_reduce_sum_d(acc[j]);

  if constexpr (SPLIT_TOPK) {
    __shared__ float partial[TOPK][OUTS_PER_WARP];
    if (lane_id == 0) {
#pragma unroll
      for (int j = 0; j < OUTS_PER_WARP; ++j) partial[warp_id][j] = acc[j];
    }
    // Nonlocal experts also reach the barrier with zero partials.
    __syncthreads();
    if (threadIdx.x < OUTS_PER_WARP) {
      float value = 0.f;
#pragma unroll
      for (int s = 0; s < TOPK; ++s) {
        const int global_expert_id = topk_ids[token_id * TOPK + s];
        if (global_expert_id < expert_start || global_expert_id >= expert_start + NUM_EXPERTS)
          continue;
        value = fmaf(partial[s][threadIdx.x], topk_scale[token_id * TOPK + s], value);
      }
      store_output(out, shared_output, (size_t)token_id * HIDDEN_SIZE + h_base + threadIdx.x,
                   value);
    }
  } else if (lane_id == 0) {
#pragma unroll
    for (int j = 0; j < OUTS_PER_WARP; ++j) {
      store_output(out, shared_output, (size_t)token_id * HIDDEN_SIZE + h_base + j, acc[j]);
    }
  }
}

// Inline quantization matches the standalone BF16-to-FP8 conversion exactly.
// With VEC=16, eight lanes together cover one 128-element scale group.
template <int VEC>
__device__ __forceinline__ void quantize_inline(const __nv_bfloat16* x, uint32_t (&packed)[VEC / 4],
                                                float& scale) {
  float values[VEC];
  float maximum = 0.f;
#pragma unroll
  for (int p = 0; p < VEC / 4; ++p) {
    const uint2 v = *reinterpret_cast<const uint2*>(x + p * 4);
    const __nv_bfloat16* bf = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      values[p * 4 + j] = __bfloat162float(bf[j]);
      maximum = fmaxf(maximum, fabsf(values[p * 4 + j]));
    }
  }
  constexpr int GROUP_LANES = SCALE_GROUP / VEC;
  const unsigned mask = __activemask();
#pragma unroll
  for (int delta = GROUP_LANES / 2; delta; delta /= 2)
    maximum = fmaxf(maximum, __shfl_xor_sync(mask, maximum, delta, GROUP_LANES));
  scale = __fdiv_rn(fmaxf(maximum, 1.e-10f), 448.f);
  const float inv = __fdiv_rn(1.f, scale);
#pragma unroll
  for (int p = 0; p < VEC / 4; ++p) {
    const __nv_fp8x2_e4m3 lo(make_float2(values[4 * p] * inv, values[4 * p + 1] * inv));
    const __nv_fp8x2_e4m3 hi(make_float2(values[4 * p + 2] * inv, values[4 * p + 3] * inv));
    packed[p] = uint32_t(lo.__x) | (uint32_t(hi.__x) << 16);
  }
}

// Runtime shapes retain contiguous expert rows and 128-element scale groups.
// Template order: intermediate size, split experts, lanes per subgroup,
// inline BF16 quantization, top-k, hidden size, expert count, outputs per subgroup.
// A zero STATIC_* parameter uses the corresponding runtime dimension.
// OUTPUTS belongs to one SUBGROUP, not necessarily a full warp: with eight
// lanes and OUTPUTS=4, four subgroups compute 16 output channels per warp.
// SPLIT_TOPK assigns one expert to each warp; otherwise each warp visits all
// selected experts before the final reduction.
template <int STATIC_INTER, bool SPLIT_TOPK, int SUBGROUP = 32, bool FUSED_QUANT = false,
          int STATIC_TOPK = 0, int STATIC_HIDDEN = 0, int STATIC_EXPERTS = 0,
          int OUTPUTS = OUTS_PER_WARP>
__global__ void moe_down_3d_batched_general_kernel(
    const fp8* buffer, const fp8* W, const float* buffer_scale, const float* w_scale,
    const int* topk_ids, const float* topk_scale, __nv_bfloat16* out,
    const __nv_bfloat16* shared_output, int hidden_arg, int inter_arg, int num_experts_arg,
    int topk_arg, int expert_start) {
  const int hidden = STATIC_HIDDEN ? STATIC_HIDDEN : hidden_arg;
  const int num_experts = STATIC_EXPERTS ? STATIC_EXPERTS : num_experts_arg;
  const int topk = STATIC_TOPK ? STATIC_TOPK : topk_arg;
  constexpr int VEC = SUBGROUP < 32         ? 16
                      : STATIC_INTER == 128 ? 4
                      : STATIC_INTER == 256 ? 8
                                            : VEC_DOWN;
  constexpr int K_UNROLL =
      SUBGROUP == 32 && STATIC_INTER ? (STATIC_INTER + SUBGROUP * VEC - 1) / (SUBGROUP * VEC) : 1;
  const int inter = STATIC_INTER ? STATIC_INTER : inter_arg;
  const int scale_cols = inter / SCALE_GROUP;
  const int weight_scale_cols = (scale_cols + 3) / 4 * 4;
  const int warp_id = threadIdx.x / WARP_SIZE_DOWN;
  const int lane_id = threadIdx.x & (SUBGROUP - 1);
  const int subgroup_id = (threadIdx.x & (WARP_SIZE_DOWN - 1)) / SUBGROUP;
  constexpr int SUBGROUPS = WARP_SIZE_DOWN / SUBGROUP;
  const int token_id = blockIdx.x;
  constexpr int GROUPS_PER_CTA = (SPLIT_TOPK ? 1 : WARPS_PER_CTA_DOWN) * SUBGROUPS;
  const int h_base =
      (blockIdx.y * GROUPS_PER_CTA + (SPLIT_TOPK ? 0 : warp_id * SUBGROUPS) + subgroup_id) *
      OUTPUTS;
  float acc[OUTPUTS]{};

#pragma unroll 1
  for (int s = SPLIT_TOPK ? warp_id : 0; s < (SPLIT_TOPK ? warp_id + 1 : topk); ++s) {
    const int flat_tok = token_id * topk + s;
    const int global_expert_id = topk_ids[flat_tok];
    if (global_expert_id < expert_start || global_expert_id >= expert_start + num_experts) {
      continue;
    }
    const int expert_id = global_expert_id - expert_start;
    const fp8* x_ptr = buffer + (size_t)flat_tok * inter;
    const fp8* w_ptr = W + ((size_t)expert_id * hidden + h_base) * inter;
    const float* xs_ptr = FUSED_QUANT ? nullptr : buffer_scale + (size_t)flat_tok * scale_cols;
    const float* ws_ptr =
        w_scale +
        ((size_t)expert_id * (hidden / SCALE_GROUP) + h_base / SCALE_GROUP) * weight_scale_cols;
    float expert_acc[OUTPUTS]{};
#pragma unroll K_UNROLL
    for (int k = lane_id * VEC; k < inter; k += SUBGROUP * VEC) {
      uint32_t xp[VEC / 4];
      float inline_scale = 0.f;
      if constexpr (FUSED_QUANT) {
        static_assert(VEC == 16);
        quantize_inline<VEC>(
            reinterpret_cast<const __nv_bfloat16*>(buffer) + (size_t)flat_tok * inter + k, xp,
            inline_scale);
      } else if constexpr (VEC == 16) {
        const uint4 v = *reinterpret_cast<const uint4*>(x_ptr + k);
        xp[0] = v.x;
        xp[1] = v.y;
        xp[2] = v.z;
        xp[3] = v.w;
      } else if constexpr (VEC == 8) {
        const uint2 v = *reinterpret_cast<const uint2*>(x_ptr + k);
        xp[0] = v.x;
        xp[1] = v.y;
      } else {
        xp[0] = *reinterpret_cast<const uint32_t*>(x_ptr + k);
      }
      float x[VEC];
#pragma unroll
      for (int p = 0; p < VEC / 4; ++p) {
        const float2 lo = unpack_fp8_pair_d(uint16_t(xp[p]));
        const float2 hi = unpack_fp8_pair_d(uint16_t(xp[p] >> 16));
        x[4 * p] = lo.x;
        x[4 * p + 1] = lo.y;
        x[4 * p + 2] = hi.x;
        x[4 * p + 3] = hi.y;
      }
      const float scale =
          (FUSED_QUANT ? inline_scale : xs_ptr[k / SCALE_GROUP]) * ws_ptr[k / SCALE_GROUP];
#pragma unroll
      for (int j = 0; j < OUTPUTS; ++j) {
        uint32_t wp[VEC / 4];
        if constexpr (VEC == 16) {
          const uint4 v = *reinterpret_cast<const uint4*>(w_ptr + j * inter + k);
          wp[0] = v.x;
          wp[1] = v.y;
          wp[2] = v.z;
          wp[3] = v.w;
        } else if constexpr (VEC == 8) {
          const uint2 v = *reinterpret_cast<const uint2*>(w_ptr + j * inter + k);
          wp[0] = v.x;
          wp[1] = v.y;
        } else {
          wp[0] = *reinterpret_cast<const uint32_t*>(w_ptr + j * inter + k);
        }
        float dot = 0.f;
#pragma unroll
        for (int p = 0; p < VEC / 4; ++p) {
          const float2 lo = unpack_fp8_pair_d(uint16_t(wp[p]));
          const float2 hi = unpack_fp8_pair_d(uint16_t(wp[p] >> 16));
          dot = fmaf(x[4 * p], lo.x, dot);
          dot = fmaf(x[4 * p + 1], lo.y, dot);
          dot = fmaf(x[4 * p + 2], hi.x, dot);
          dot = fmaf(x[4 * p + 3], hi.y, dot);
        }
        expert_acc[j] = fmaf(dot, scale, expert_acc[j]);
      }
    }
#pragma unroll
    for (int j = 0; j < OUTPUTS; ++j) {
      if constexpr (SPLIT_TOPK) {
        acc[j] = expert_acc[j];
      } else {
        acc[j] = fmaf(expert_acc[j], topk_scale[flat_tok], acc[j]);
      }
    }
  }
#pragma unroll
  for (int j = 0; j < OUTPUTS; ++j) {
#pragma unroll
    for (int delta = SUBGROUP / 2; delta; delta /= 2)
      acc[j] += __shfl_xor_sync(0xffffffff, acc[j], delta, SUBGROUP);
  }
  if constexpr (SPLIT_TOPK) {
    __shared__ float partial[STATIC_TOPK ? STATIC_TOPK : 32][SUBGROUPS * OUTPUTS];
    if (lane_id == 0) {
#pragma unroll
      for (int j = 0; j < OUTPUTS; ++j) partial[warp_id][subgroup_id * OUTPUTS + j] = acc[j];
    }
    __syncthreads();
    if (threadIdx.x < SUBGROUPS * OUTPUTS) {
      float value = 0.f;
      for (int s = 0; s < topk; ++s) {
        const int global_expert_id = topk_ids[token_id * topk + s];
        if (global_expert_id < expert_start || global_expert_id >= expert_start + num_experts)
          continue;
        value = fmaf(partial[s][threadIdx.x], topk_scale[token_id * topk + s], value);
      }
      store_output(out, shared_output,
                   (size_t)token_id * hidden + blockIdx.y * (SUBGROUPS * OUTPUTS) + threadIdx.x,
                   value);
    }
  } else if (lane_id == 0) {
#pragma unroll
    for (int j = 0; j < OUTPUTS; ++j)
      store_output(out, shared_output, (size_t)token_id * hidden + h_base + j, acc[j]);
  }
}

// Full-warp fallback: split experts for B<=2, otherwise visit them serially.
template <int STATIC_INTER>
void launch_moe_down_general(const void* buffer, const void* W, const float* buffer_scale,
                             const float* w_scale, const int* topk_ids, const float* topk_scale,
                             const void* shared_output, void* out, int B, int hidden, int inter,
                             int num_experts, int topk, int expert_start, cudaStream_t stream) {
  if (B <= 2 && topk <= 32) {
    const dim3 grid(B, hidden / OUTS_PER_WARP);
    moe_down_3d_batched_general_kernel<STATIC_INTER, true>
        <<<grid, topk * WARP_SIZE_DOWN, 0, stream>>>(
            static_cast<const fp8*>(buffer), static_cast<const fp8*>(W), buffer_scale, w_scale,
            topk_ids, topk_scale, static_cast<__nv_bfloat16*>(out),
            static_cast<const __nv_bfloat16*>(shared_output), hidden, inter, num_experts, topk,
            expert_start);
  } else {
    const dim3 grid(B, hidden / OUTS_PER_WARP / WARPS_PER_CTA_DOWN);
    moe_down_3d_batched_general_kernel<STATIC_INTER, false>
        <<<grid, WARPS_PER_CTA_DOWN * WARP_SIZE_DOWN, 0, stream>>>(
            static_cast<const fp8*>(buffer), static_cast<const fp8*>(W), buffer_scale, w_scale,
            topk_ids, topk_scale, static_cast<__nv_bfloat16*>(out),
            static_cast<const __nv_bfloat16*>(shared_output), hidden, inter, num_experts, topk,
            expert_start);
  }
}

// Template order: intermediate size, subgroup lanes, inline quantization,
// top-k, outputs per subgroup. Every warp handles one selected expert.
template <int STATIC_INTER, int SUBGROUP, bool FUSED_QUANT = false, int STATIC_TOPK = 0,
          int OUTPUTS = OUTS_PER_WARP>
void launch_moe_down_subgroup(const void* buffer, const void* W, const float* buffer_scale,
                              const float* w_scale, const int* topk_ids, const float* topk_scale,
                              const void* shared_output, void* out, int B, int hidden, int inter,
                              int num_experts, int topk, int expert_start, cudaStream_t stream) {
  constexpr int OUTS_PER_CTA = (32 / SUBGROUP) * OUTPUTS;
  const dim3 grid(B, hidden / OUTS_PER_CTA);
  moe_down_3d_batched_general_kernel<STATIC_INTER, true, SUBGROUP, FUSED_QUANT, STATIC_TOPK, 0, 0,
                                     OUTPUTS><<<grid, topk * WARP_SIZE_DOWN, 0, stream>>>(
      static_cast<const fp8*>(buffer), static_cast<const fp8*>(W), buffer_scale, w_scale, topk_ids,
      topk_scale, static_cast<__nv_bfloat16*>(out),
      static_cast<const __nv_bfloat16*>(shared_output), hidden, inter, num_experts, topk,
      expert_start);
}

// Select the output tile after top-k has been resolved. Inline quantization
// and runtime top-k always retain four outputs per eight-lane subgroup.
template <int STATIC_INTER, bool FUSED_QUANT, int STATIC_TOPK>
void launch_moe_down_narrow_tile(const void* buffer, const void* W, const float* buffer_scale,
                                 const float* w_scale, const int* topk_ids, const float* topk_scale,
                                 const void* shared_output, void* out, int B, int hidden, int inter,
                                 int num_experts, int topk, int expert_start, cudaStream_t stream) {
  if constexpr (!FUSED_QUANT && STATIC_TOPK != 0) {
    if (B >= 4) {
      launch_moe_down_subgroup<STATIC_INTER, 8, false, STATIC_TOPK, 8>(
          buffer, W, buffer_scale, w_scale, topk_ids, topk_scale, shared_output, out, B, hidden,
          inter, num_experts, topk, expert_start, stream);
      return;
    }
  }
  launch_moe_down_subgroup<STATIC_INTER, 8, FUSED_QUANT, STATIC_TOPK>(
      buffer, W, buffer_scale, w_scale, topk_ids, topk_scale, shared_output, out, B, hidden, inter,
      num_experts, topk, expert_start, stream);
}

template <int STATIC_INTER, bool FUSED_QUANT>
void dispatch_moe_down_narrow_topk(const void* buffer, const void* W, const float* buffer_scale,
                                   const float* w_scale, const int* topk_ids,
                                   const float* topk_scale, const void* shared_output, void* out,
                                   int B, int hidden, int inter, int num_experts, int topk,
                                   int expert_start, cudaStream_t stream) {
  if (topk == 8) {
    launch_moe_down_narrow_tile<STATIC_INTER, FUSED_QUANT, 8>(
        buffer, W, buffer_scale, w_scale, topk_ids, topk_scale, shared_output, out, B, hidden,
        inter, num_experts, topk, expert_start, stream);
  } else if (topk == 10) {
    launch_moe_down_narrow_tile<STATIC_INTER, FUSED_QUANT, 10>(
        buffer, W, buffer_scale, w_scale, topk_ids, topk_scale, shared_output, out, B, hidden,
        inter, num_experts, topk, expert_start, stream);
  } else {
    launch_moe_down_narrow_tile<STATIC_INTER, FUSED_QUANT, 0>(
        buffer, W, buffer_scale, w_scale, topk_ids, topk_scale, shared_output, out, B, hidden,
        inter, num_experts, topk, expert_start, stream);
  }
}

// The small-batch split uses eight outputs to reuse activation decoding;
// serial experts retain four outputs to limit register use.
template <int H, int E, int K>
void launch_moe_down_fixed_512(const void* buffer, const void* W, const float* buffer_scale,
                               const float* w_scale, const int* topk_ids, const float* topk_scale,
                               const void* shared_output, void* out, int B, int expert_start,
                               cudaStream_t stream) {
  if (B <= 2) {
    constexpr int OUTPUTS = 8;
    const dim3 grid(B, H / OUTPUTS);
    moe_down_3d_batched_general_kernel<512, true, 32, false, K, H, E, OUTPUTS>
        <<<grid, K * WARP_SIZE_DOWN, 0, stream>>>(
            static_cast<const fp8*>(buffer), static_cast<const fp8*>(W), buffer_scale, w_scale,
            topk_ids, topk_scale, static_cast<__nv_bfloat16*>(out),
            static_cast<const __nv_bfloat16*>(shared_output), H, 512, E, K, expert_start);
  } else {
    const dim3 grid(B, H / OUTS_PER_WARP / WARPS_PER_CTA_DOWN);
    moe_down_3d_batched_general_kernel<512, false, 32, false, K, H, E>
        <<<grid, WARPS_PER_CTA_DOWN * WARP_SIZE_DOWN, 0, stream>>>(
            static_cast<const fp8*>(buffer), static_cast<const fp8*>(W), buffer_scale, w_scale,
            topk_ids, topk_scale, static_cast<__nv_bfloat16*>(out),
            static_cast<const __nv_bfloat16*>(shared_output), H, 512, E, K, expert_start);
  }
}

// Preserve the original shape's batch-dependent split and cache policy.
void launch_moe_down_original_shape(const void* buffer, const void* W, const float* buffer_scale,
                                    const float* w_scale, const int* topk_ids,
                                    const float* topk_scale, const void* shared_output, void* out,
                                    int B, int expert_start, cudaStream_t stream) {
  if (B <= 2) {
    const dim3 grid(B, HIDDEN_SIZE / OUTS_PER_WARP);
    moe_down_3d_batched_kernel<true><<<grid, TOPK * WARP_SIZE_DOWN, 0, stream>>>(
        static_cast<const fp8*>(buffer), static_cast<const fp8*>(W), buffer_scale, w_scale,
        topk_ids, topk_scale, static_cast<__nv_bfloat16*>(out),
        static_cast<const __nv_bfloat16*>(shared_output), expert_start);
  } else {
    const dim3 grid(B, HIDDEN_SIZE / OUTS_PER_WARP / WARPS_PER_CTA_DOWN);
    if (B >= 6 && B <= 8) {
      moe_down_3d_batched_kernel<false, true>
          <<<grid, WARPS_PER_CTA_DOWN * WARP_SIZE_DOWN, 0, stream>>>(
              static_cast<const fp8*>(buffer), static_cast<const fp8*>(W), buffer_scale, w_scale,
              topk_ids, topk_scale, static_cast<__nv_bfloat16*>(out),
              static_cast<const __nv_bfloat16*>(shared_output), expert_start);
    } else {
      moe_down_3d_batched_kernel<false><<<grid, WARPS_PER_CTA_DOWN * WARP_SIZE_DOWN, 0, stream>>>(
          static_cast<const fp8*>(buffer), static_cast<const fp8*>(W), buffer_scale, w_scale,
          topk_ids, topk_scale, static_cast<__nv_bfloat16*>(out),
          static_cast<const __nv_bfloat16*>(shared_output), expert_start);
    }
  }
}

}  // anonymous namespace

void down_async(const void* buffer, const void* W, const float* buffer_scale, const float* w_scale,
                const int* topk_ids, const float* topk_scale, const void* shared_output, void* out,
                int B, int hidden, int inter, int num_experts, int topk, int expert_start,
                cudaStream_t stream) {
  if (!B) return;

  if (B <= 8 && inter == 512) {
    if (hidden == 3072 && num_experts == 256 && topk == 8) {
      launch_moe_down_fixed_512<3072, 256, 8>(buffer, W, buffer_scale, w_scale, topk_ids,
                                              topk_scale, shared_output, out, B, expert_start,
                                              stream);
      return;
    }
    if (hidden == 2048 && num_experts == 512 && topk == 10) {
      launch_moe_down_fixed_512<2048, 512, 10>(buffer, W, buffer_scale, w_scale, topk_ids,
                                               topk_scale, shared_output, out, B, expert_start,
                                               stream);
      return;
    }
  }

  // Narrow intermediates use one expert per warp, split into eight-lane groups.
  if (B <= 8 && topk <= 32 && (inter == 128 || inter == 256)) {
    if (inter == 128) {
      dispatch_moe_down_narrow_topk<128, false>(buffer, W, buffer_scale, w_scale, topk_ids,
                                                topk_scale, shared_output, out, B, hidden, inter,
                                                num_experts, topk, expert_start, stream);
    } else {
      dispatch_moe_down_narrow_topk<256, false>(buffer, W, buffer_scale, w_scale, topk_ids,
                                                topk_scale, shared_output, out, B, hidden, inter,
                                                num_experts, topk, expert_start, stream);
    }
    return;
  }

  if (hidden == HIDDEN_SIZE && inter == INTER_SIZE && num_experts == NUM_EXPERTS && topk == TOPK) {
    launch_moe_down_original_shape(buffer, W, buffer_scale, w_scale, topk_ids, topk_scale,
                                   shared_output, out, B, expert_start, stream);
    return;
  }

  // Keep common intermediate sizes static and all other aligned sizes dynamic.
  switch (inter) {
    case 128:
      launch_moe_down_general<128>(buffer, W, buffer_scale, w_scale, topk_ids, topk_scale,
                                   shared_output, out, B, hidden, inter, num_experts, topk,
                                   expert_start, stream);
      break;
    case 256:
      launch_moe_down_general<256>(buffer, W, buffer_scale, w_scale, topk_ids, topk_scale,
                                   shared_output, out, B, hidden, inter, num_experts, topk,
                                   expert_start, stream);
      break;
    case 512:
      launch_moe_down_general<512>(buffer, W, buffer_scale, w_scale, topk_ids, topk_scale,
                                   shared_output, out, B, hidden, inter, num_experts, topk,
                                   expert_start, stream);
      break;
    default:
      launch_moe_down_general<0>(buffer, W, buffer_scale, w_scale, topk_ids, topk_scale,
                                 shared_output, out, B, hidden, inter, num_experts, topk,
                                 expert_start, stream);
  }
}

void down_bf16_async(const void* buffer, const void* W, const float* w_scale, const int* topk_ids,
                     const float* topk_scale, const void* shared_output, void* out, int B,
                     int hidden, int inter, int num_experts, int topk, int expert_start,
                     cudaStream_t stream) {
  if (!B) return;
  // The binding selects this entry only for inter=128 or 256. Quantization
  // remains inline, with four outputs per subgroup for every top-k and batch.
  if (inter == 128) {
    dispatch_moe_down_narrow_topk<128, true>(buffer, W, nullptr, w_scale, topk_ids, topk_scale,
                                             shared_output, out, B, hidden, inter, num_experts,
                                             topk, expert_start, stream);
  } else {
    dispatch_moe_down_narrow_topk<256, true>(buffer, W, nullptr, w_scale, topk_ids, topk_scale,
                                             shared_output, out, B, hidden, inter, num_experts,
                                             topk, expert_start, stream);
  }
}

}  // namespace warp_decode
}  // namespace fuse_moe
}  // namespace hpc
