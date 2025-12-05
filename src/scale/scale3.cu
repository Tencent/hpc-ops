// Copyright 2025 hpc-ops authors
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "src/scale/scale3.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace scale {
namespace kernels {

template <int kHiddenSize, bool kIsMoe>
__global__ void scale3_kernel(const __nv_bfloat16 *input_ptr, const float *scale,
                              const float *scale2, __nv_fp8_e4m3 *output_ptr_fp8,
                              __nv_fp8_e4m3 *output_ptr_fp8_scale2, float *output_ptr_fp32,
                              int num_tokens) {
  constexpr int kVecSize = 8;

  using T = __nv_bfloat162;
  constexpr int kN = kVecSize / 2;

  const int token_start = blockIdx.x;
  const int icol = threadIdx.x * kVecSize;

  float inv_scale = 1;
  float inv_scale2 = 1;
  if constexpr (kIsMoe) {
    inv_scale = scale[0];
    inv_scale2 = scale2[0];
  } else {
    inv_scale = scale[0];
  }

  for (int itoken = token_start; itoken < num_tokens; itoken += gridDim.x) {
    const int offset = itoken * kHiddenSize + icol;
    const auto *input = input_ptr + offset;
    auto *output_fp8 = output_ptr_fp8 + offset;
    auto *output_fp8_scale2 = output_ptr_fp8_scale2 + offset;
    auto *output_fp32 = output_ptr_fp32 + offset;

    auto in = to<float>(load<T, kN>(input));
    if constexpr (kIsMoe) {
      auto &in_view = reshape<2, 4>(in);
      store(output_fp32, in_view[0]);
      store(output_fp32 + kVecSize / 2, in_view[1]);

      vec_t<float, kVecSize> o, o_scale2;
#pragma unroll
      for (int i = 0; i < size(in); i++) {
        o[i] = in[i] * inv_scale;
        o_scale2[i] = in[i] * inv_scale2;
      }
      store(output_fp8, to<__nv_fp8x4_e4m3>(o));
      store(output_fp8_scale2, to<__nv_fp8x4_e4m3>(o_scale2));

    } else {
      vec_t<float, kVecSize> o;
#pragma unroll
      for (int i = 0; i < size(in); i++) {
        o[i] = in[i] * inv_scale;
      }
      store(output_fp8, to<__nv_fp8x4_e4m3>(o));
    }
  }
}

}  // namespace kernels

void scale3_async(void *input_ptr, void *scale, void *scale2, void *output_ptr,
                  void *output_fp8_scale2_ptr, void *output_fp32_ptr, int num_tokens,
                  int hidden_states, bool is_moe, cudaStream_t stream) {
  using Tin = __nv_bfloat16;
  using Tout = __nv_fp8_e4m3;

  dim3 grid(num_tokens);
  dim3 block(512);

  constexpr int kHiddenSize = 4096;

  if (is_moe) {
    constexpr bool kIsMoe = true;
    kernels::scale3_kernel<kHiddenSize, kIsMoe><<<grid, block, 0, stream>>>(
        reinterpret_cast<const Tin *>(input_ptr), reinterpret_cast<const float *>(scale),
        reinterpret_cast<const float *>(scale2), reinterpret_cast<Tout *>(output_ptr),
        reinterpret_cast<Tout *>(output_fp8_scale2_ptr), reinterpret_cast<float *>(output_fp32_ptr),
        num_tokens);
  } else {
    constexpr bool kIsMoe = false;
    kernels::scale3_kernel<kHiddenSize, kIsMoe><<<grid, block, 0, stream>>>(
        reinterpret_cast<const Tin *>(input_ptr), reinterpret_cast<const float *>(scale),
        reinterpret_cast<const float *>(scale2), reinterpret_cast<Tout *>(output_ptr),
        reinterpret_cast<Tout *>(output_fp8_scale2_ptr), reinterpret_cast<float *>(output_fp32_ptr),
        num_tokens);
  }
}

}  // namespace scale
}  // namespace hpc
