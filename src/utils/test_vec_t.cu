// Copyright (C) 2026 Tencent.

#include <cuda.h>
#include <cuda_bf16.h>
#include <stdio.h>

#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {

__global__ void bfloat16_to_float(float *output, const __nv_bfloat16 *input) {
  auto fx8 = to<float>(load<__half2, 4>(input));

#pragma unroll
  for (int i = 0; i < fx8.num; ++i) {
    output[threadIdx.x + i] = fx8[i];
  }
}

__global__ void floatx2(float *output, const float *input) {
  auto x2 = load<float, 2>(input);
  store(output, x2[1], x2[0]);
}

__global__ void floatx4(float *output, const float *input) {
  auto x4 = load<float, 4>(input);

  store(output, x4[3], x4[2], x4[1], x4[0]);
}

__global__ void bfloat16x4(__nv_bfloat16 *output, const __nv_bfloat16 *input) {
  auto x4 = load<__nv_bfloat16, 4>(input);

  store(output, x4);
}

__global__ void bfloat16x8(__nv_bfloat16 *output, const __nv_bfloat16 *input) {
  auto x8 = load<__nv_bfloat16, 8>(input);

  store(output, x8);
}

__global__ void reshape(float *output, const float *input) {
  vec_t<float, 8> v;
  auto v2 = reshape<2, 4>(v);

  v2[0] = load<float, 4>(input);
  v2[1] = load<float, 4>(input + 4);

  auto bv2 = to<__nv_bfloat162>(v2[0]);
  auto hv2 = to<__half2>(v2[1]);

  store(output, bv2);
  store(output + 2, hv2);
}

__global__ void reshape2(float *output, const float *input) {
  vec_t<int, 30> v;

  auto &vs = reshape<2, 3, 5>(v);
  for (int i = 0; i < size(vs); ++i) {
    for (int j = 0; j < size(vs[i]); ++j) {
      for (int k = 0; k < size(vs[i][j]); ++k) {
        vs[i][j][k] = (i + 1) * 100 + (j + 1) * 10 + (k + 1);
      }
    }
  }

  for (int i = 0; i < size(v); ++i) {
    printf("v[%d] = %d\n", i, v[i]);
  }
}

}  // namespace hpc

int main() {
  float *ptr = nullptr;
  hpc::reshape2<<<1, 1>>>(ptr, ptr);

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  printf("err = %d, str = %s\n", err, cudaGetErrorString(err));
}
