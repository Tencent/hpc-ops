// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include <algorithm>
#include <string>

#include "src/math/math.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace math {

constexpr int kPrintLimit = 17;

namespace kernels {

template <typename T>
__global__ void hasnan_kernel(const void *ptr, int64_t num, vec_t<char, kPrintLimit> tag) {
  int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  int idx_in_block = threadIdx.x;

  bool yes = false;
  for (int64_t i = idx; i < num; i += blockDim.x * gridDim.x) {
    T v = reinterpret_cast<const T *>(ptr)[idx];

    if constexpr (std::is_same_v<T, float>) {
      yes = std::isnan(v);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      yes = __hisnan(v);
    } else if constexpr (std::is_same_v<T, __half>) {
      yes = __hisnan(v);
    } else if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
      bool yes_1 = *(reinterpret_cast<uint8_t *>(&v)) == uint8_t(0x7F);
      bool yes_2 = *(reinterpret_cast<uint8_t *>(&v)) == uint8_t(0xFF);
      yes = yes_1 | yes_2;
    } else if constexpr (std::is_same_v<T, __nv_fp8_e5m2>) {
      bool yes_1 = *(reinterpret_cast<uint8_t *>(&v)) == uint8_t(0x7F);
      bool yes_2 = *(reinterpret_cast<uint8_t *>(&v)) == uint8_t(0xFF);
      yes = yes_1 | yes_2;
    } else if constexpr (std::is_same_v<T, uint8_t>) {
      yes = *(reinterpret_cast<uint8_t *>(&v)) == uint8_t(0x80);
    } else if constexpr (std::is_same_v<T, __nv_fp8_e8m0>) {
      yes = *(reinterpret_cast<uint8_t *>(&v)) == uint8_t(0xFF);
    }
    if (yes) {
      break;
    }
  }
  yes = __syncthreads_or(yes);

  if ((idx_in_block == 0) && yes) {
    printf("nan exists for %s\n", tag.data);
  }
}

}  // namespace kernels

static auto to_vec(const std::string &input) {
  vec_t<char, kPrintLimit> ret;

  int len = input.size();

  if (len < kPrintLimit) {
    int i = 0;
    for (; i < len; ++i) {
      ret.data[i] = input[i];
    }
    ret.data[i] = 0;
  } else {
    int i = 0;
    for (; i < kPrintLimit - 3; ++i) {
      ret.data[i] = input[i];
    }
    for (; i < kPrintLimit - 1; ++i) {
      ret.data[i] = '.';
    }
    ret.data[i] = 0;
  }

  return ret;
}

void hasnan_async(const void *ptr, int64_t num, torch::ScalarType dtype, const std::string &tag,
                  int num_warning_blocks, cudaStream_t stream) {
  // for one tensor we print `num_warning_blocks` warning info at most.
  // here we use `num_warning_blocks` block to reduce the noisy information.
  dim3 block(1024);
  dim3 grid(std::min(get_sm_count(), num_warning_blocks));

  auto ctag = to_vec(tag);

  switch (dtype) {
    case torch::kFloat32: {
      kernels::hasnan_kernel<float><<<grid, block, 0, stream>>>(ptr, num, ctag);
      break;
    }
    case torch::kFloat16: {
      kernels::hasnan_kernel<half><<<grid, block, 0, stream>>>(ptr, num, ctag);
      break;
    }
    case torch::kBFloat16: {
      kernels::hasnan_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(ptr, num, ctag);
      break;
    }
    case torch::kFloat8_e4m3fn: {
      kernels::hasnan_kernel<__nv_fp8_e4m3><<<grid, block, 0, stream>>>(ptr, num, ctag);
      break;
    }
    case torch::kFloat8_e5m2: {
      kernels::hasnan_kernel<__nv_fp8_e5m2><<<grid, block, 0, stream>>>(ptr, num, ctag);
      break;
    }
    case torch::kFloat8_e4m3fnuz:
    case torch::kFloat8_e5m2fnuz: {
      kernels::hasnan_kernel<uint8_t><<<grid, block, 0, stream>>>(ptr, num, ctag);
      break;
    }
    case torch::kFloat8_e8m0fnu: {
      kernels::hasnan_kernel<__nv_fp8_e8m0><<<grid, block, 0, stream>>>(ptr, num, ctag);
      break;
    }
    default: {
      throw std::invalid_argument("type not support yet!");
      break;
    }
  }
}

}  // namespace math
}  // namespace hpc
