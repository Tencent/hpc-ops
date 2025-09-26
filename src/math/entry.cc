// Copyright 2025 hpc-ops authors

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include <string>

#include "src/math/math.h"

namespace hpc {
namespace math {

void hasnan_entry(const torch::Tensor &a, const std::string &tag, int64_t num_warning_blocks) {
  auto stream = at::cuda::getCurrentCUDAStream(a.get_device());
  TORCH_CHECK(a.is_contiguous(), "input tensor must be contiguous");

  const void *ptr = a.const_data_ptr();

  auto num = a.numel();
  torch::ScalarType dtype = a.scalar_type();

  hasnan_async(ptr, num, dtype, tag, num_warning_blocks, stream);
}

}  // namespace math
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) { m.def("has_nan", &hpc::math::hasnan_entry); }
