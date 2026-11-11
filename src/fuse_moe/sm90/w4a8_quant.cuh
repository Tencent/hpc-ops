// Copyright (C) 2026 Tencent.

#ifndef SRC_FUSE_MOE_SM90_W4A8_QUANT_CUH_
#define SRC_FUSE_MOE_SM90_W4A8_QUANT_CUH_

#include <cuda_runtime.h>
#include <stdint.h>

#include "src/utils/utils.cuh"

namespace hpc {
namespace fuse_moe {
namespace w4a8_kernels {

// All lanes in a warp supply four contiguous values from one group of 128.
__device__ __forceinline__ void quantize_w4a8_group(vec_t<float, 4> values, __nv_fp8_e4m3 *output,
                                                    float *scale_output) {
  float maximum = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) maximum = fmaxf(maximum, fabsf(values[i]));
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, offset));
  }
  // The frozen DynamicGroup FP32 scale uses a floor, not an additive epsilon.
  const float scale = fmaxf(maximum / 448.f, 1e-30f);
  const float inverse = 1.f / scale;
#pragma unroll
  for (int i = 0; i < 4; ++i) values[i] *= inverse;
  store(output, to<__nv_fp8x4_e4m3>(values));
  if (threadIdx.x % 32 == 0) *scale_output = scale;
}

// Paired C stores BF16 gate then up, each TileN/2 columns wide.
// The caller completes R2S and a CTA barrier first, and retains a CTA barrier
// after this helper before persistent-task reuse. There are exactly 8 warps.
template <int kTaskM, int kTileN, class SharedC>
__device__ __forceinline__ void quantize_paired_gateup(const SharedC &c, __nv_fp8_e4m3 *middle,
                                                       float *middle_scale, int row_base,
                                                       int valid_rows, int tile_n) {
  static_assert((kTaskM == 8 || kTaskM == 16 || kTaskM == 40) && (kTileN == 256 || kTileN == 512));
  constexpr int kHalfN = kTileN / 2;
  constexpr int kGroupsPerTile = kTileN / 256;
  const int lane = threadIdx.x % 32;
#pragma unroll
  for (int group = threadIdx.x / 32; group < kTaskM * kGroupsPerTile; group += 8) {
    const int local_row = group / kGroupsPerTile;
    const int group128 = group % kGroupsPerTile;
    const int column = group128 * 128 + lane * 4;
    // A smaller adaptive math tile writes only the valid row prefix.
    // This predicate is warp-uniform, preserving the full-mask reduction.
    if (local_row < valid_rows) {
      auto values = to<float>(load<__nv_bfloat162, 2>(&c(column, local_row)));
      const auto up = to<float>(load<__nv_bfloat162, 2>(&c(column + kHalfN, local_row)));
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        values[i] = __bfloat162float(__float2bfloat16_rn(silu(values[i]) * up[i]));
      }
      const int64_t row = static_cast<int64_t>(row_base) + local_row;
      quantize_w4a8_group(values, middle + row * 512 + tile_n * kHalfN + column,
                          middle_scale + row * 4 + tile_n * kGroupsPerTile + group128);
    }
  }
}

// One warp quantizes one contiguous group of 128 values. The scale layout is
// row-major [rows,kWidth/128], matching the native W4A8 GEMM's input contract.
template <int kWidth, bool kActivate, bool kReset = false>
__global__ void w4a8_quant_kernel(const __nv_bfloat16 *input, __nv_fp8_e4m3 *output, float *scales,
                                  int rows, int *reset_seqlens = nullptr) {
  if constexpr (kReset) {
    // Complete native MoE counter reset even for warps past the quantization tail.
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < 256; i += gridDim.x * blockDim.x)
      reset_seqlens[i] = 0;
  }
  const int lane = threadIdx.x % 32;
  constexpr int groups_per_row = kWidth / 128;
  const int64_t group = static_cast<int64_t>(blockIdx.x) * 4 + threadIdx.x / 32;
  if (group >= static_cast<int64_t>(rows) * groups_per_row) return;

  const int64_t row = group / groups_per_row;
  const int col = static_cast<int>(group % groups_per_row) * 128 + lane * 4;
  const auto *source = input + row * kWidth * (kActivate ? 2 : 1) + col;
  auto values = to<float>(load<__nv_bfloat162, 2>(source));
  if constexpr (kActivate) {
    const auto up = to<float>(load<__nv_bfloat162, 2>(source + kWidth));
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      // Preserve Humming's BF16 activation boundary before the second dynamic
      // quantization, even though activation and quantization share one kernel.
      values[i] = __bfloat162float(__float2bfloat16_rn(silu(values[i]) * up[i]));
    }
  }

  quantize_w4a8_group(values, output + row * kWidth + col, scales + group);
}

}  // namespace w4a8_kernels

// Buffers are contiguous and naturally aligned. For activation, input has
// shape [rows,2*kWidth]; the two private MoE widths are known at compile time.
template <int kWidth, bool kActivate>
inline void w4a8_quantize_async(const void *input, void *output, float *scales, int rows,
                                cudaStream_t stream, int *reset_seqlens = nullptr) {
  static_assert(kWidth > 0 && kWidth % 128 == 0);
  if (rows == 0) return;
  const int64_t groups = static_cast<int64_t>(rows) * (kWidth / 128);
  const dim3 grid(static_cast<unsigned int>((groups + 3) / 4));
  const auto *source = static_cast<const __nv_bfloat16 *>(input);
  auto *destination = static_cast<__nv_fp8_e4m3 *>(output);
  if constexpr (kActivate) {
    w4a8_kernels::w4a8_quant_kernel<kWidth, true>
        <<<grid, 128, 0, stream>>>(source, destination, scales, rows);
  } else if (reset_seqlens != nullptr) {
    w4a8_kernels::w4a8_quant_kernel<kWidth, false, true>
        <<<grid, 128, 0, stream>>>(source, destination, scales, rows, reset_seqlens);
  } else {
    w4a8_kernels::w4a8_quant_kernel<kWidth, false>
        <<<grid, 128, 0, stream>>>(source, destination, scales, rows);
  }
}

}  // namespace fuse_moe
}  // namespace hpc

#endif  // SRC_FUSE_MOE_SM90_W4A8_QUANT_CUH_
