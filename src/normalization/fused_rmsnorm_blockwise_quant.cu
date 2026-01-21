// Copyright 2025 hpc-ops authors
#include <cuda.h>

#include <iostream>

#include "src/normalization/fused_rmsnorm_blockwise_quant.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace normalization {

namespace kernels {

template <typename Tout, int kWarpsPerRow, int kHiddenSize>
__global__ void fused_rmsnorm_blockwise_quant_kernel(Tout* y_ptr, float* y_scale_ptr,
                                                     const __nv_bfloat16* input_ptr,
                                                     const __nv_bfloat16* weight_ptr, const int m,
                                                     const float eps,
                                                     const bool with_blockwise_quant) {
  constexpr int kN = 16 / sizeof(__nv_bfloat16);
  constexpr int kElementsPerWarp = 32 * kN;
  constexpr int kRowsPerBlock = 4 / kWarpsPerRow;
  constexpr float kInvHiddenSize = 1.0f / kHiddenSize;
  constexpr int kIerPerRow =
      (kHiddenSize + kWarpsPerRow * kElementsPerWarp - 1) / (kWarpsPerRow * kElementsPerWarp - 1);
  constexpr int kWarpsPerBlock = 4;

  __shared__ float smem_sum[kWarpsPerBlock];
  vec_t<float, kN> input[kIerPerRow];
  vec_t<float, kN> weight[kIerPerRow];

  int idx = threadIdx.x;
  int iwarp = idx / 32;
  int ilane = idx % 32;

  int irow = blockIdx.x * kRowsPerBlock + iwarp / kWarpsPerRow;
  if (irow >= m) {
    return;
  }
  auto* input_row_ptr = input_ptr + irow * kHiddenSize;
  auto* y_row_ptr = y_ptr + irow * kHiddenSize;
  int icol = ((iwarp % kWarpsPerRow) * 32 + ilane) * kN;

  float rms = 0.0f;
#pragma unroll
  for (int iter = 0; iter < kIerPerRow; iter++) {
    int col = icol + iter * kWarpsPerRow * kElementsPerWarp;
    if (col < kHiddenSize) {
      input[iter] = to<float>(load<__nv_bfloat162, kN / 2>(input_row_ptr + col));
      weight[iter] = to<float>(load<__nv_bfloat162, kN / 2>(weight_ptr + col));
      for (int i = 0; i < kN; i++) {
        rms += input[iter][i] * input[iter][i];
      }
    }
  }
  // warp reduce
  rms = warp_reduce_sum_xor(rms);
  if (ilane == 0) {
    smem_sum[iwarp] = rms;
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < kWarpsPerRow; i++) {
    if (iwarp % kWarpsPerRow != i) {
      rms += smem_sum[iwarp / kWarpsPerRow * kWarpsPerRow + i];
    }
  }
  rms = rsqrtf_ftz(rms * kInvHiddenSize + eps);
#pragma unroll
  for (int iter = 0; iter < kIerPerRow; iter++) {
    int col = icol + iter * kWarpsPerRow * kElementsPerWarp;
    if (col < kHiddenSize) {
#pragma unroll
      for (int i = 0; i < kN; i++) {
        input[iter][i] *= (rms * weight[iter][i]);
      }
      // do not need blockwise quant, store output
      if constexpr (std::is_same_v<Tout, __nv_bfloat16>) {
        store(y_row_ptr + col, to<__nv_bfloat16>(input[iter]));
      }
    }
  }

  // need blockwise quant
  if constexpr (std::is_same_v<Tout, __nv_fp8_e4m3>) {
    auto* y_scale_row_ptr = y_scale_ptr + irow * kHiddenSize / 128;
#pragma unroll
    for (int iter = 0; iter < kIerPerRow; iter++) {
      int col = icol + iter * kWarpsPerRow * kElementsPerWarp;
      if (col < kHiddenSize) {
        float max = 0.0f;
#pragma unroll
        for (int i = 0; i < kN; i++) {
          max = fmaxf(max, fabsf(input[iter][i]));
        }
        max = half_warp_reduce_max_down(max);
        float scale = max / 448.0f;
        float inv_scale = rcpf_ftz(scale + eps);
#pragma unroll
        for (int i = 0; i < kN; i++) {
          input[iter][i] *= inv_scale;
        }
        int pos = iwarp % kWarpsPerRow * 2 + iter * kWarpsPerRow * 2;
        if (ilane == 0) {
          store(y_scale_row_ptr + pos, scale);
        } else if (ilane == 16) {
          store(y_scale_row_ptr + pos + 1, scale);
        }
        store(y_row_ptr + col, to<__nv_fp8x4_e4m3>(input[iter]));
      }
    }
  }
}
}  // namespace kernels

void fused_rmsnorm_blockwise_quant_async(void* y_ptr, void* y_scale_ptr, const void* input_ptr,
                                         const void* weight_ptr, const int m, const int hidden_size,
                                         const float eps, bool with_blockwise_quant,
                                         cudaStream_t stream) {
  constexpr int kWarpCount = 4;
  constexpr int kWarpSize = 32;
  dim3 block(kWarpSize * kWarpCount);
  if (hidden_size == 128) {
    constexpr int kHiddenSize = 128;
    constexpr int kWarpsPerRow = 1;
    constexpr int kRowsPerBlock = kWarpCount / kWarpsPerRow;
    dim3 grid((m + kRowsPerBlock - 1) / kRowsPerBlock);
    if (with_blockwise_quant) {
      kernels::fused_rmsnorm_blockwise_quant_kernel<__nv_fp8_e4m3, kWarpsPerRow, kHiddenSize>
          <<<grid, block, 0, stream>>>((__nv_fp8_e4m3*)y_ptr, (float*)y_scale_ptr,
                                       (__nv_bfloat16*)input_ptr, (__nv_bfloat16*)weight_ptr, m,
                                       eps, with_blockwise_quant);
    } else {
      kernels::fused_rmsnorm_blockwise_quant_kernel<__nv_bfloat16, kWarpsPerRow, kHiddenSize>
          <<<grid, block, 0, stream>>>((__nv_bfloat16*)y_ptr, nullptr, (__nv_bfloat16*)input_ptr,
                                       (__nv_bfloat16*)weight_ptr, m, eps, with_blockwise_quant);
    }
  }
  if (hidden_size == 512) {
    constexpr int kHiddenSize = 512;
    constexpr int kWarpsPerRow = 2;
    constexpr int kRowsPerBlock = kWarpCount / kWarpsPerRow;
    dim3 grid((m + kRowsPerBlock - 1) / kRowsPerBlock);
    if (with_blockwise_quant) {
      kernels::fused_rmsnorm_blockwise_quant_kernel<__nv_fp8_e4m3, kWarpsPerRow, kHiddenSize>
          <<<grid, block, 0, stream>>>((__nv_fp8_e4m3*)y_ptr, (float*)y_scale_ptr,
                                       (__nv_bfloat16*)input_ptr, (__nv_bfloat16*)weight_ptr, m,
                                       eps, with_blockwise_quant);
    } else {
      kernels::fused_rmsnorm_blockwise_quant_kernel<__nv_bfloat16, kWarpsPerRow, kHiddenSize>
          <<<grid, block, 0, stream>>>((__nv_bfloat16*)y_ptr, nullptr, (__nv_bfloat16*)input_ptr,
                                       (__nv_bfloat16*)weight_ptr, m, eps, with_blockwise_quant);
    }
  }
  if (hidden_size == 1024) {
    constexpr int kHiddenSize = 1024;
    constexpr int kWarpsPerRow = 4;
    constexpr int kRowsPerBlock = kWarpCount / kWarpsPerRow;
    dim3 grid((m + kRowsPerBlock - 1) / kRowsPerBlock);
    if (with_blockwise_quant) {
      kernels::fused_rmsnorm_blockwise_quant_kernel<__nv_fp8_e4m3, kWarpsPerRow, kHiddenSize>
          <<<grid, block, 0, stream>>>((__nv_fp8_e4m3*)y_ptr, (float*)y_scale_ptr,
                                       (__nv_bfloat16*)input_ptr, (__nv_bfloat16*)weight_ptr, m,
                                       eps, with_blockwise_quant);
    } else {
      kernels::fused_rmsnorm_blockwise_quant_kernel<__nv_bfloat16, kWarpsPerRow, kHiddenSize>
          <<<grid, block, 0, stream>>>((__nv_bfloat16*)y_ptr, nullptr, (__nv_bfloat16*)input_ptr,
                                       (__nv_bfloat16*)weight_ptr, m, eps, with_blockwise_quant);
    }
  }
  if (hidden_size == 4096) {
    constexpr int kHiddenSize = 4096;
    constexpr int kWarpsPerRow = 4;
    constexpr int kRowsPerBlock = kWarpCount / kWarpsPerRow;
    dim3 grid((m + kRowsPerBlock - 1) / kRowsPerBlock);
    if (with_blockwise_quant) {
      kernels::fused_rmsnorm_blockwise_quant_kernel<__nv_fp8_e4m3, kWarpsPerRow, kHiddenSize>
          <<<grid, block, 0, stream>>>((__nv_fp8_e4m3*)y_ptr, (float*)y_scale_ptr,
                                       (__nv_bfloat16*)input_ptr, (__nv_bfloat16*)weight_ptr, m,
                                       eps, with_blockwise_quant);
    } else {
      kernels::fused_rmsnorm_blockwise_quant_kernel<__nv_bfloat16, kWarpsPerRow, kHiddenSize>
          <<<grid, block, 0, stream>>>((__nv_bfloat16*)y_ptr, nullptr, (__nv_bfloat16*)input_ptr,
                                       (__nv_bfloat16*)weight_ptr, m, eps, with_blockwise_quant);
    }
  }
}

}  // namespace normalization
}  // namespace hpc
