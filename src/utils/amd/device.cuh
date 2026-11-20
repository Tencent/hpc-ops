// Copyright (C) 2026 Tencent.
#ifndef SRC_UTILS_AMD_DEVICE_CUH_
#define SRC_UTILS_AMD_DEVICE_CUH_

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

namespace hpc::amd {

// Use the compiled target's wave size: RDNA wave32 and CDNA wave64 share this code.
template <int kThreads>
__device__ __forceinline__ float block_sum(float value, float* scratch) {
  const int lane = threadIdx.x % warpSize;
  const int wave = threadIdx.x / warpSize;
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down(value, offset);
  }
  if (lane == 0) scratch[wave] = value;
  __syncthreads();
  if (wave == 0) {
    value = lane < kThreads / warpSize ? scratch[lane] : 0.0f;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      value += __shfl_down(value, offset);
    }
    if (lane == 0) scratch[0] = value;
  }
  __syncthreads();
  const float result = scratch[0];
  __syncthreads();  // Every wave must read the result before scratch is reused.
  return result;
}

__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

}  // namespace hpc::amd
#endif  // SRC_UTILS_AMD_DEVICE_CUH_
