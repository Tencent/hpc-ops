#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include "src/cast/cast.h"

namespace hpc {
namespace cast {

torch::Tensor entry(const torch::Tensor &a, torch::Dtype dtype) {
  auto stream = at::cuda::getCurrentCUDAStream(a.get_device());
  TORCH_CHECK(a.is_contiguous(), "input tensor must be contigous");

  torch::Tensor c = torch::empty_like(a, dtype);
  TORCH_CHECK(c.is_contiguous(), "output tensor must be contigous");

  const auto *aptr = a.const_data_ptr();
  auto *cptr = c.mutable_data_ptr();

  auto num = a.numel();

  auto tin = a.scalar_type();
  auto tout = c.scalar_type();

  cast_async(cptr, aptr, num, tout, tin, stream);

  return c;
}

TORCH_LIBRARY_FRAGMENT(hpc, m) { m.def("cast", &entry); }

}  // namespace cast
}  // namespace hpc
