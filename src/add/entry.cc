#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include "src/add/add.h"

namespace hpc {
namespace add {

torch::Tensor entry(const torch::Tensor &a, const torch::Tensor &b) {
  auto stream = at::cuda::getCurrentCUDAStream(a.get_device());
  TORCH_CHECK(a.is_contiguous(), "input tensor must be contiguous");
  TORCH_CHECK(b.is_contiguous(), "input tensor must be contiguous");

  torch::Tensor c = torch::empty_like(a);

  const auto *aptr = a.const_data_ptr<float>();
  const auto *bptr = b.const_data_ptr<float>();
  auto *cptr = c.mutable_data_ptr<float>();

  auto num = a.numel();

  add_async(cptr, aptr, bptr, num, stream);

  return c;
}

}  // namespace add
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) { m.def("add", &hpc::add::entry); }
