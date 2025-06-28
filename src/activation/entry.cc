#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include "src/activation/activation.h"

namespace hpc {
namespace activation {

torch::Tensor entry(torch::Tensor &input, torch::Tensor &scale) {
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

  std::vector<int64_t> output_shape(input.sizes().begin(), input.sizes().end());
  output_shape[output_shape.size() - 1] /= 2;

  auto options = input.options().dtype(torch::kFloat8_e4m3fn);

  torch::Tensor output = torch::empty(output_shape, options);

  using Tin = __nv_bfloat16;
  using Tout = __nv_fp8_e4m3;


  Tin *input_ptr = reinterpret_cast<Tin*>(input.data_ptr());
  Tout *output_ptr = reinterpret_cast<Tout*>(output.data_ptr());
  float *scale_ptr = scale.data_ptr<float>();

  auto input_shape = input.sizes();
  int num_col = input_shape[input_shape.size() - 1];
  int num_row = 1;
  for (int i = 0; i < input_shape.size() - 1; ++i) {
    num_row *= input_shape[i];
  }

  act_mul_and_quant_async(output_ptr, input_ptr, scale_ptr, num_row, num_col,
                          stream);

  return output;
}

}  // namespace activation
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def("act_mul_and_quant", &hpc::activation::entry);
}
