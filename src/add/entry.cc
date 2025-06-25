#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "src/add/add.h"

namespace hpc {
namespace add {

torch::Tensor entry(torch::Tensor& a, torch::Tensor& b) {
  auto stream = at::cuda::getCurrentCUDAStream(a.get_device());

  torch::Tensor c = torch::empty_like(a);

  float* aptr = reinterpret_cast<float*>(a.data_ptr());
  float* bptr = reinterpret_cast<float*>(b.data_ptr());
  float* cptr = reinterpret_cast<float*>(c.data_ptr());

  auto num = a.numel();

  add_async(cptr, aptr, bptr, num, stream);

  return c;
}

TORCH_LIBRARY_FRAGMENT(hpc, m) { m.def("add", &entry); }

}  // namespace add
}  // namespace hpc
