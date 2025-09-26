// Copyright 2025 hpc-ops authors

#ifndef SRC_MATH_MATH_H_
#define SRC_MATH_MATH_H_

#include <cuda_runtime_api.h>
#include <stdint.h>
#include <torch/all.h>

#include <string>
#include <vector>

namespace hpc {
namespace math {

void hasnan_async(const void *ptr, int64_t num, torch::ScalarType dtype, const std::string &tag,
                  int num_warning_blocks, cudaStream_t stream);

}  // namespace math
}  // namespace hpc

#endif  // SRC_MATH_MATH_H_
