// Copyright (C) 2026 Tencent.
// Output-oriented FP8 decode, adapted from Religious-J/warp_decode commit 1b3fff2.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <type_traits>

#include "src/fuse_moe/fuse_moe_warp_decode.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace fuse_moe {
namespace warp_decode {
namespace {

// Hunyuan-v3 TP8: per-tensor scales allow I=192 without padding to 128 groups.
constexpr int kHidden = 4096;
constexpr int kInter = 192;
constexpr int kTopk = 8;
constexpr int kWarp = 32;

constexpr int kGateWarps = 4;
constexpr int kGateOutputs = 2;  // Outputs per warp.
constexpr int kGateVector = 16;
constexpr int kDownSubgroup = 8;
constexpr int kDownGroups = kWarp / kDownSubgroup;

using Fp8 = __nv_fp8_e4m3;
using Bf16 = __nv_bfloat16;

__device__ __forceinline__ float2 unpack_pair(uint16_t packed) {
  return __half22float2(static_cast<__half2>(__nv_cvt_fp8x2_to_halfraw2(packed, __NV_E4M3)));
}

__device__ __forceinline__ uint4 load_weight(const Fp8* pointer) {
  uint4 value;
  asm("ld.global.L1::evict_first.v4.u32 {%0, %1, %2, %3}, [%4];"
      : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
      : "l"(pointer));
  return value;
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int delta = kWarp / 2; delta; delta /= 2) {
    value += __shfl_xor_sync(0xffffffff, value, delta);
  }
  return value;
}

__device__ __forceinline__ bool is_local_expert(int expert, int expert_start,
                                                int num_local_experts) {
  return expert >= expert_start && expert < expert_start + num_local_experts;
}

// Both modes round gate/up to BF16 before activation. kUseBf16Multiply also
// rounds SiLU and its product with up to BF16. act_scale is a multiplier.
template <bool kUseBf16Multiply>
__device__ __forceinline__ Fp8 activate_and_quantize(float gate, float up, float act_scale) {
  gate = __bfloat162float(__float2bfloat16_rn(gate));
  up = __bfloat162float(__float2bfloat16_rn(up));
  float value;
  if constexpr (kUseBf16Multiply) {
    const Bf16 activated = __float2bfloat16_rn(hpc::silu(gate));
    value = __bfloat162float(__hmul(activated, __float2bfloat16_rn(up)));
  } else {
    value = hpc::silu(gate) * up;
  }
  return Fp8(value * act_scale);
}

// Four warps produce two neurons each. Reuse decoded activations across the
// two outputs; prefetching, when enabled, applies only to the first output.
template <bool kPrefetch, bool kUseBf16Multiply>
__global__ void gate_up_pertensor_kernel(const Fp8* input, const Fp8* weight,
                                         const float* gate_up_scale, const float* act_scale,
                                         const int* ids, Fp8* buffer, int num_local_experts,
                                         int expert_start) {
  const int token = blockIdx.x;
  const int route = token * kTopk + blockIdx.z;
  const int lane = threadIdx.x & (kWarp - 1);
  const int warp = threadIdx.x / kWarp;
  const int neuron = (blockIdx.y * kGateWarps + warp) * kGateOutputs;
  const int global_expert = ids[route];
  if (!is_local_expert(global_expert, expert_start, num_local_experts)) {
    if (lane == 0) {
#pragma unroll
      for (int j = 0; j < kGateOutputs; ++j) buffer[route * kInter + neuron + j] = Fp8(0.f);
    }
    return;
  }
  const int expert = global_expert - expert_start;
  const Fp8* x = input + token * kHidden;
  const Fp8* gate = weight + (static_cast<size_t>(expert) * (2 * kInter) + neuron) * kHidden;
  const Fp8* up = gate + kInter * kHidden;
  float gate_acc[kGateOutputs]{};
  float up_acc[kGateOutputs]{};

  uint4 next_gate, next_up;
  if constexpr (kPrefetch) {
    next_gate = load_weight(gate + lane * kGateVector);
    next_up = load_weight(up + lane * kGateVector);
  }
#pragma unroll
  for (int k = lane * kGateVector; k < kHidden; k += kWarp * kGateVector) {
    uint4 current_gate, current_up;
    if constexpr (kPrefetch) {
      current_gate = next_gate;
      current_up = next_up;
    }
    const uint4 packed_x = *reinterpret_cast<const uint4*>(x + k);
    if constexpr (kPrefetch) {
      if (k + kWarp * kGateVector < kHidden) {
        next_gate = load_weight(gate + k + kWarp * kGateVector);
        next_up = load_weight(up + k + kWarp * kGateVector);
      }
    }
    const auto* x_pairs = reinterpret_cast<const uint16_t*>(&packed_x);
    float2 decoded[kGateVector / 2];
#pragma unroll
    for (int i = 0; i < kGateVector / 2; ++i) decoded[i] = unpack_pair(x_pairs[i]);
#pragma unroll
    for (int j = 0; j < kGateOutputs; ++j) {
      uint4 packed_gate, packed_up;
      if (kPrefetch && j == 0) {
        packed_gate = current_gate;
        packed_up = current_up;
      } else {
        packed_gate = load_weight(gate + j * kHidden + k);
        packed_up = load_weight(up + j * kHidden + k);
      }
      const auto* gate_pairs = reinterpret_cast<const uint16_t*>(&packed_gate);
      const auto* up_pairs = reinterpret_cast<const uint16_t*>(&packed_up);
      float gate_partial = 0.f, up_partial = 0.f;
#pragma unroll
      for (int i = 0; i < kGateVector / 2; ++i) {
        const float2 g = unpack_pair(gate_pairs[i]);
        const float2 u = unpack_pair(up_pairs[i]);
        gate_partial = fmaf(decoded[i].x, g.x, gate_partial);
        gate_partial = fmaf(decoded[i].y, g.y, gate_partial);
        up_partial = fmaf(decoded[i].x, u.x, up_partial);
        up_partial = fmaf(decoded[i].y, u.y, up_partial);
      }
      gate_acc[j] += gate_partial;
      up_acc[j] += up_partial;
    }
  }
#pragma unroll
  for (int j = 0; j < kGateOutputs; ++j) {
    gate_acc[j] = warp_sum(gate_acc[j]);
    up_acc[j] = warp_sum(up_acc[j]);
  }
  if (lane == 0) {
    const float multiplier = gate_up_scale[expert];
    const float quant_multiplier = act_scale[0];
#pragma unroll
    for (int j = 0; j < kGateOutputs; ++j) {
      buffer[route * kInter + neuron + j] = activate_and_quantize<kUseBf16Multiply>(
          gate_acc[j] * multiplier, up_acc[j] * multiplier, quant_multiplier);
    }
  }
}

// Each expert warp has four eight-lane groups, each computing kOutputs rows.
// VEC=8 covers I=192 with three full iterations; VEC=16 uses 128 + 64 elements.
template <int kOutputs, int kVector>
__global__ void down_pertensor_kernel(const Fp8* buffer, const Fp8* weight, const float* down_scale,
                                      const int* ids, const float* route_scale,
                                      const Bf16* shared_output, Bf16* out, int num_local_experts,
                                      int expert_start) {
  static_assert(kVector == 8 || kVector == 16);
  using Packed = std::conditional_t<kVector == 8, uint2, uint4>;
  constexpr int kOutputTile = kDownGroups * kOutputs;
  const int token = blockIdx.x;
  const int warp = threadIdx.x / kWarp;
  const int lane = threadIdx.x & (kDownSubgroup - 1);
  const int group = (threadIdx.x & (kWarp - 1)) / kDownSubgroup;
  const int output_base = (blockIdx.y * kDownGroups + group) * kOutputs;
  const int route = token * kTopk + warp;
  const int global_expert = ids[route];
  const bool local = is_local_expert(global_expert, expert_start, num_local_experts);
  float acc[kOutputs]{};
  if (local) {
    const int expert = global_expert - expert_start;
    const Fp8* x = buffer + route * kInter;
    const Fp8* w = weight + (static_cast<size_t>(expert) * kHidden + output_base) * kInter;
#pragma unroll 1
    for (int k = lane * kVector; k < kInter; k += kDownSubgroup * kVector) {
      const Packed packed_x = *reinterpret_cast<const Packed*>(x + k);
      const auto* x_pairs = reinterpret_cast<const uint16_t*>(&packed_x);
      float2 decoded[kVector / 2];
#pragma unroll
      for (int i = 0; i < kVector / 2; ++i) decoded[i] = unpack_pair(x_pairs[i]);
#pragma unroll
      for (int j = 0; j < kOutputs; ++j) {
        const Packed packed_w = *reinterpret_cast<const Packed*>(w + j * kInter + k);
        const auto* w_pairs = reinterpret_cast<const uint16_t*>(&packed_w);
        float partial = 0.f;
#pragma unroll
        for (int i = 0; i < kVector / 2; ++i) {
          const float2 values = unpack_pair(w_pairs[i]);
          partial = fmaf(decoded[i].x, values.x, partial);
          partial = fmaf(decoded[i].y, values.y, partial);
        }
        acc[j] += partial;
      }
    }
  }
#pragma unroll
  for (int j = 0; j < kOutputs; ++j) {
#pragma unroll
    for (int delta = kDownSubgroup / 2; delta; delta /= 2) {
      acc[j] += __shfl_xor_sync(0xffffffff, acc[j], delta, kDownSubgroup);
    }
  }
  __shared__ float expert_outputs[kTopk][kOutputTile];
  if (lane == 0) {
    const float multiplier = local ? down_scale[global_expert - expert_start] : 0.f;
#pragma unroll
    for (int j = 0; j < kOutputs; ++j) {
      // Match the per-expert BF16 GEMM output before FP32 route reduction.
      expert_outputs[warp][group * kOutputs + j] =
          __bfloat162float(__float2bfloat16_rn(acc[j] * multiplier));
    }
  }
  __syncthreads();
  if (threadIdx.x < kOutputTile) {
    float value = 0.f;
#pragma unroll
    for (int slot = 0; slot < kTopk; ++slot) {
      const int offset = token * kTopk + slot;
      if (is_local_expert(ids[offset], expert_start, num_local_experts)) {
        value = fmaf(expert_outputs[slot][threadIdx.x], route_scale[offset], value);
      }
    }
    const size_t index =
        static_cast<size_t>(token) * kHidden + blockIdx.y * kOutputTile + threadIdx.x;
    if (shared_output != nullptr) value += __bfloat162float(shared_output[index]);
    out[index] = __float2bfloat16_rn(value);
  }
}

template <bool kUseBf16Multiply>
void launch_gate_up_pertensor(const Fp8* input, const Fp8* weight, const float* gate_up_scale,
                              const float* act_scale, const int* ids, Fp8* buffer, int batch_size,
                              int num_local_experts, int expert_start, cudaStream_t stream) {
  constexpr int kOutputTile = kGateWarps * kGateOutputs;
  static_assert(kInter % kOutputTile == 0);
  const dim3 grid(batch_size, kInter / kOutputTile, kTopk);
  if (batch_size <= 3) {
    gate_up_pertensor_kernel<true, kUseBf16Multiply><<<grid, kGateWarps * kWarp, 0, stream>>>(
        input, weight, gate_up_scale, act_scale, ids, buffer, num_local_experts, expert_start);
  } else {
    gate_up_pertensor_kernel<false, kUseBf16Multiply><<<grid, kGateWarps * kWarp, 0, stream>>>(
        input, weight, gate_up_scale, act_scale, ids, buffer, num_local_experts, expert_start);
  }
}

template <int kOutputs, int kVector>
void launch_down_pertensor(const Fp8* buffer, const Fp8* weight, const float* down_scale,
                           const int* ids, const float* topk_scale, const Bf16* shared_output,
                           Bf16* out, int batch_size, int num_local_experts, int expert_start,
                           cudaStream_t stream) {
  constexpr int kOutputTile = kDownGroups * kOutputs;
  static_assert(kHidden % kOutputTile == 0);
  const dim3 grid(batch_size, kHidden / kOutputTile);
  down_pertensor_kernel<kOutputs, kVector>
      <<<grid, kTopk * kWarp, 0, stream>>>(buffer, weight, down_scale, ids, topk_scale,
                                           shared_output, out, num_local_experts, expert_start);
}

}  // namespace

void gate_up_pertensor_async(const void* input, const void* weight, const float* gate_up_scale,
                             const float* act_and_mul_scale, const int* ids, void* buffer,
                             int batch_size, int num_local_experts, int expert_start,
                             bool use_bf16_mul, cudaStream_t stream) {
  if (batch_size == 0) return;
  const auto* fp8_input = static_cast<const Fp8*>(input);
  const auto* fp8_weight = static_cast<const Fp8*>(weight);
  auto* fp8_buffer = static_cast<Fp8*>(buffer);
  if (use_bf16_mul) {
    launch_gate_up_pertensor<true>(fp8_input, fp8_weight, gate_up_scale, act_and_mul_scale, ids,
                                   fp8_buffer, batch_size, num_local_experts, expert_start, stream);
  } else {
    launch_gate_up_pertensor<false>(fp8_input, fp8_weight, gate_up_scale, act_and_mul_scale, ids,
                                    fp8_buffer, batch_size, num_local_experts, expert_start,
                                    stream);
  }
}

void down_pertensor_async(const void* buffer, const void* weight, const float* down_scale,
                          const int* ids, const float* topk_scale, const void* shared_output,
                          void* out, int batch_size, int num_local_experts, int expert_start,
                          cudaStream_t stream) {
  if (batch_size == 0) return;
  const auto* fp8_buffer = static_cast<const Fp8*>(buffer);
  const auto* fp8_weight = static_cast<const Fp8*>(weight);
  const auto* shared = static_cast<const Bf16*>(shared_output);
  auto* output = static_cast<Bf16*>(out);
  if (batch_size == 7 || batch_size == 8) {
    launch_down_pertensor<8, 8>(fp8_buffer, fp8_weight, down_scale, ids, topk_scale, shared, output,
                                batch_size, num_local_experts, expert_start, stream);
  } else if (batch_size <= 3) {
    launch_down_pertensor<4, 16>(fp8_buffer, fp8_weight, down_scale, ids, topk_scale, shared,
                                 output, batch_size, num_local_experts, expert_start, stream);
  } else {
    launch_down_pertensor<8, 16>(fp8_buffer, fp8_weight, down_scale, ids, topk_scale, shared,
                                 output, batch_size, num_local_experts, expert_start, stream);
  }
}

}  // namespace warp_decode
}  // namespace fuse_moe
}  // namespace hpc
