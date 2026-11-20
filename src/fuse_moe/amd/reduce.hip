// Copyright (C) 2026 Tencent.
#include <cstdint>

#include "src/fuse_moe/amd/reduce.h"

namespace hpc::fuse_moe::amd {
namespace kernels {

__global__ void reduce_kernel(hip_bfloat16* output, const hip_bfloat16* input, const int* positions,
                              const float* scales, const hip_bfloat16* shared_output,
                              int input_rows, int hidden, int topk) {
  const int row = blockIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= hidden) return;
  float value = 0.0f;
  for (int i = 0; i < topk; ++i) {
    const int64_t route = static_cast<int64_t>(row) * topk + i;
    const int pos = positions[route];
    if (pos >= 0 && pos < input_rows) {
      value += static_cast<float>(input[static_cast<int64_t>(pos) * hidden + col]) * scales[route];
    }
  }
  const int64_t offset = static_cast<int64_t>(row) * hidden + col;
  if (shared_output) value += static_cast<float>(shared_output[offset]);
  output[offset] = hip_bfloat16(value);
}

}  // namespace kernels

void reduce_async(hip_bfloat16* output, const hip_bfloat16* input, const int* positions,
                  const float* scales, const hip_bfloat16* shared_output, int rows, int input_rows,
                  int hidden, int topk, hipStream_t stream) {
  if (rows == 0 || hidden == 0) return;
  hipLaunchKernelGGL(kernels::reduce_kernel, dim3((hidden + 255) / 256, rows), dim3(256), 0, stream,
                     output, input, positions, scales, shared_output, input_rows, hidden, topk);
}

}  // namespace hpc::fuse_moe::amd
