// Copyright 2025 hpc-ops authors

#ifndef SRC_HADAMARD_HADAMARD_DEVICE_CUH_
#define SRC_HADAMARD_HADAMARD_DEVICE_CUH_

#include <cuda.h>

#include "src/utils/utils.cuh"

namespace hpc {
namespace hadamard {
namespace device {

// a unit hadamard that do a H2 kron product
template <int kStride, int kBaseSize>
__device__ __forceinline__ void unit_hadamard(vec_t<float, kBaseSize>& data, int ilane) {
  const float sign = (ilane & kStride) ? -1.f : 1.f;
#pragma unroll
  for (int i = 0; i < kBaseSize; i++) {
    float other = __shfl_xor_sync(0xFFFFFFFF, data[i], kStride);
    data[i] = sign * data[i] + other;
  }
}

template <int kBaseSize>
__device__ __forceinline__ void base_hadamard(vec_t<float, kBaseSize>& x);

template <>
__device__ __forceinline__ void base_hadamard<4>(vec_t<float, 4>& x) {
  float y0 = x[0] + x[1] + x[2] + x[3];
  float y1 = x[0] - x[1] + x[2] - x[3];
  float y2 = x[0] + x[1] - x[2] - x[3];
  float y3 = x[0] - x[1] - x[2] + x[3];
  x[0] = y0;
  x[1] = y1;
  x[2] = y2;
  x[3] = y3;
}

// n=64 device Hadamard Compute
// 1 warp processes 2 rows of n=64 hadamard (lanes 0-15 row A, lanes 16-31 row B)
__device__ __forceinline__ void hadamard_n64_warp(vec_t<float, 4>& data, int ilane) {
  unit_hadamard<1, 4>(data, ilane);
  unit_hadamard<2, 4>(data, ilane);
  unit_hadamard<4, 4>(data, ilane);
  unit_hadamard<8, 4>(data, ilane);
  base_hadamard<4>(data);
}

// n=128 device Hadamard Compute
// 1 warp processes one row of n=128 hadamard.
__device__ __forceinline__ void hadamard_n128_warp(vec_t<float, 4>& data, int ilane) {
  unit_hadamard<1, 4>(data, ilane);
  unit_hadamard<2, 4>(data, ilane);
  unit_hadamard<4, 4>(data, ilane);
  unit_hadamard<8, 4>(data, ilane);
  unit_hadamard<16, 4>(data, ilane);
  base_hadamard<4>(data);
}

}  // namespace device
}  // namespace hadamard
}  // namespace hpc

#endif  // SRC_HADAMARD_HADAMARD_DEVICE_CUH_
