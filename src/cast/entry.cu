#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include "src/cast/cast.cuh"

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

  if (tin == torch::kFloat32) {
    switch (tout) {
      case torch::kFloat32: {
        cast_async<float, float>(cptr, aptr, num, stream);
        break;
      }
      case torch::kFloat64: {
        cast_async<double, float>(cptr, aptr, num, stream);
        break;
      }
      case torch::kFloat16: {
        cast_async<half, float>(cptr, aptr, num, stream);
        break;
      }
      default: {
        printf("not support yet!");
        break;
      }
    }  // switch
  }    // if

  return c;
}

TORCH_LIBRARY_FRAGMENT(hpc, m) { m.def("cast", &entry); }

}  // namespace cast
}  // namespace hpc
