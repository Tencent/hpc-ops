#include <cuda.h>
#include <cuda_bf16.h>
#include <stdio.h>

#include "utils.cuh"

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

}  // namespace hpc

int main() {}
