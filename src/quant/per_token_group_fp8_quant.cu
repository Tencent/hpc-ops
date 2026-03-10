// Copyright 2025 hpc-ops authors
#include <cuda.h>

#include <iostream>

#include "src/quant/per_token_group_fp8_quant.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace quant {
namespace kernels {
template <int kHiddenSize, int kWarpsPerRow, int kRowsPerBlock>
__global__ void per_token_group_fp8_quant(const __nv_bfloat16 *input_ptr, __nv_fp8_e4m3 *y_fp8_ptr,
                                          float *y_scale_ptr, float quant_eps, int batch_size) {
  constexpr int kN = 16 / sizeof(__nv_bfloat16);
  constexpr int kElementsPerWarp = 32 * kN;
  constexpr int kIerPerRow =
      (kHiddenSize + kWarpsPerRow * kElementsPerWarp - 1) / (kWarpsPerRow * kElementsPerWarp);
  constexpr float kInvFp8Max = 1.f / 448.f;

  vec_t<float, kN> input[kIerPerRow];
  int idx = threadIdx.x;
  int iwarp = idx / 32;
  int ilane = idx % 32;

  int irow = blockIdx.x * kRowsPerBlock + iwarp / kWarpsPerRow;
  if (irow >= batch_size) {
    return;
  }

  auto *input_row_ptr = input_ptr + irow * kHiddenSize;
  auto *y_scale_row_ptr = y_scale_ptr + irow * kHiddenSize / 128;
  auto *y_fp8_row_ptr = y_fp8_ptr + irow * kHiddenSize;

  int icol = ((iwarp % kWarpsPerRow) * 32 + ilane) * kN;
#pragma unroll
  for (int iter = 0; iter < kIerPerRow; iter++) {
    int col = icol + iter * kWarpsPerRow * kElementsPerWarp;
    if (col < kHiddenSize) {
      // load input
      input[iter] = to<float>(load<__nv_bfloat162, kN / 2>(input_row_ptr + col));

      // cal max
      float max = 0.0f;
#pragma unroll
      for (int i = 0; i < kN; i++) {
        max = fmaxf(max, fabsf(input[iter][i]));
      }
      max = half_warp_reduce_max_down(max);
      float scale = max * kInvFp8Max;
      float inv_scale = rcpf_ftz(scale + quant_eps);

      // quant
#pragma unroll
      for (int i = 0; i < kN; i++) {
        input[iter][i] *= inv_scale;
      }

      // store scale
      int pos = iwarp % kWarpsPerRow * 2 + iter * kWarpsPerRow * 2;
      if (ilane == 0) {
        store(y_scale_row_ptr + pos, scale);
      } else if (ilane == 16) {
        store(y_scale_row_ptr + pos + 1, scale);
      }

      // store output
      store(y_fp8_row_ptr + col, to<__nv_fp8x4_e4m3>(input[iter]));
    }
  }
}

}  // namespace kernels

template <int kHiddenSize, int kWarpsPerRow>
void launch_per_token_group_fp8_quant(const void *input_ptr, void *output_ptr, void *quant_scale,
                                      int group_size, float quant_eps, int hidden_size,
                                      int batch_size, cudaStream_t stream) {
  constexpr int kWarpCount = 8;
  constexpr int kWarpSize = 32;
  constexpr int kRowsPerBlock = kWarpCount / kWarpsPerRow;
  dim3 block(kWarpSize * kWarpCount);
  dim3 grid((batch_size + kRowsPerBlock - 1) / kRowsPerBlock);
  kernels::per_token_group_fp8_quant<kHiddenSize, kWarpsPerRow, kRowsPerBlock>
      <<<grid, block, 0, stream>>>((__nv_bfloat16 *)(input_ptr), (__nv_fp8_e4m3 *)(output_ptr),
                                   (float *)(quant_scale), quant_eps, batch_size);
}

bool per_token_group_fp8_quant_async(const void *input_ptr, void *output_ptr, void *quant_scale,
                                     int group_size, float quant_eps, int hidden_size,
                                     int batch_size, cudaStream_t stream) {
  if (hidden_size == 128) {
    constexpr int kHiddenSize = 128;
    constexpr int kWarpsPerRow = 1;
    launch_per_token_group_fp8_quant<kHiddenSize, kWarpsPerRow>(
        input_ptr, output_ptr, quant_scale, group_size, quant_eps, hidden_size, batch_size, stream);
  } else if (hidden_size == 256) {
    constexpr int kHiddenSize = 256;
    constexpr int kWarpsPerRow = 1;
    launch_per_token_group_fp8_quant<kHiddenSize, kWarpsPerRow>(
        input_ptr, output_ptr, quant_scale, group_size, quant_eps, hidden_size, batch_size, stream);
  } else if (hidden_size == 384) {
    constexpr int kHiddenSize = 384;
    constexpr int kWarpsPerRow = 2;
    launch_per_token_group_fp8_quant<kHiddenSize, kWarpsPerRow>(
        input_ptr, output_ptr, quant_scale, group_size, quant_eps, hidden_size, batch_size, stream);
  } else if (hidden_size == 512) {
    constexpr int kHiddenSize = 512;
    constexpr int kWarpsPerRow = 2;
    launch_per_token_group_fp8_quant<kHiddenSize, kWarpsPerRow>(
        input_ptr, output_ptr, quant_scale, group_size, quant_eps, hidden_size, batch_size, stream);
  } else if (hidden_size == 768) {
    constexpr int kHiddenSize = 768;
    constexpr int kWarpsPerRow = 4;
    launch_per_token_group_fp8_quant<kHiddenSize, kWarpsPerRow>(
        input_ptr, output_ptr, quant_scale, group_size, quant_eps, hidden_size, batch_size, stream);
  } else if (hidden_size == 1024) {
    constexpr int kHiddenSize = 1024;
    constexpr int kWarpsPerRow = 4;
    launch_per_token_group_fp8_quant<kHiddenSize, kWarpsPerRow>(
        input_ptr, output_ptr, quant_scale, group_size, quant_eps, hidden_size, batch_size, stream);
  } else if (hidden_size == 1536) {
    constexpr int kHiddenSize = 1536;
    constexpr int kWarpsPerRow = 8;
    launch_per_token_group_fp8_quant<kHiddenSize, kWarpsPerRow>(
        input_ptr, output_ptr, quant_scale, group_size, quant_eps, hidden_size, batch_size, stream);
  } else if (hidden_size == 2048) {
    constexpr int kHiddenSize = 2048;
    constexpr int kWarpsPerRow = 8;
    launch_per_token_group_fp8_quant<kHiddenSize, kWarpsPerRow>(
        input_ptr, output_ptr, quant_scale, group_size, quant_eps, hidden_size, batch_size, stream);
  } else if (hidden_size == 3072) {
    constexpr int kHiddenSize = 3072;
    constexpr int kWarpsPerRow = 8;
    launch_per_token_group_fp8_quant<kHiddenSize, kWarpsPerRow>(
        input_ptr, output_ptr, quant_scale, group_size, quant_eps, hidden_size, batch_size, stream);
  } else if (hidden_size == 4096) {
    constexpr int kHiddenSize = 4096;
    constexpr int kWarpsPerRow = 8;
    launch_per_token_group_fp8_quant<kHiddenSize, kWarpsPerRow>(
        input_ptr, output_ptr, quant_scale, group_size, quant_eps, hidden_size, batch_size, stream);
  } else if (hidden_size == 5120) {
    constexpr int kHiddenSize = 5120;
    constexpr int kWarpsPerRow = 8;
    launch_per_token_group_fp8_quant<kHiddenSize, kWarpsPerRow>(
        input_ptr, output_ptr, quant_scale, group_size, quant_eps, hidden_size, batch_size, stream);
  } else if (hidden_size == 6144) {
    constexpr int kHiddenSize = 6144;
    constexpr int kWarpsPerRow = 8;
    launch_per_token_group_fp8_quant<kHiddenSize, kWarpsPerRow>(
        input_ptr, output_ptr, quant_scale, group_size, quant_eps, hidden_size, batch_size, stream);
  } else if (hidden_size == 7168) {
    constexpr int kHiddenSize = 7168;
    constexpr int kWarpsPerRow = 8;
    launch_per_token_group_fp8_quant<kHiddenSize, kWarpsPerRow>(
        input_ptr, output_ptr, quant_scale, group_size, quant_eps, hidden_size, batch_size, stream);
  } else if (hidden_size == 13824) {
    constexpr int kHiddenSize = 13824;
    constexpr int kWarpsPerRow = 8;
    launch_per_token_group_fp8_quant<kHiddenSize, kWarpsPerRow>(
        input_ptr, output_ptr, quant_scale, group_size, quant_eps, hidden_size, batch_size, stream);
  } else {
    std::cout << "not supported hidden_size for per_token_group_fp8_quant_async:" << hidden_size
              << std::endl;
    return false;
  }
  return true;
}
}  // namespace quant
}  // namespace hpc
