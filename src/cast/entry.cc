#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include "src/cast/cast.h"

namespace hpc {
namespace cast {

torch::Tensor entry(torch::Tensor& a, torch::Dtype dtype) {
  auto stream = at::cuda::getCurrentCUDAStream(a.get_device());

  torch::Tensor c = torch::empty_like(a, dtype);

  float* aptr = reinterpret_cast<float*>(a.data_ptr());
  float* cptr = reinterpret_cast<float*>(c.data_ptr());

  auto num = a.numel();

  auto tin = a.scalar_type();
  auto tout = c.scalar_type();

  cast_async(cptr, aptr, num, tout, tin, stream);

  return c;
}

TORCH_LIBRARY_FRAGMENT(hpc, m) { m.def("cast", &entry); }

}  // namespace cast
}  // namespace hpc
