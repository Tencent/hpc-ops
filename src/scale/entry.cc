// Copyright 2025 hpc-ops authors

#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <torch/library.h>

#include <tuple>

#include "src/scale/scale3.h"

namespace hpc {
namespace scale {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> scale3_entry(torch::Tensor &input,
                                                                     torch::Tensor &scale,
                                                                     torch::Tensor &scale2,
                                                                     bool is_moe) {
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

  TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "input must be bfloat16.");

  torch::Tensor output = torch::empty_like(input, torch::kFloat8_e4m3fn);
  torch::Tensor output_scale2 = torch::empty_like(input, torch::kFloat8_e4m3fn);
  torch::Tensor output_fp32 = torch::empty_like(input, torch::kFloat32);

  void *scale2_ptr = nullptr;
  void *output_scale2_ptr = nullptr;
  void *output_fp32_ptr = nullptr;
  if (is_moe) {
    output_scale2_ptr = output_scale2.mutable_data_ptr();
    output_fp32_ptr = output_fp32.mutable_data_ptr();
    scale2_ptr = scale2.data_ptr();
  }

  int hidden_state = input.size(input.dim() - 1);
  int num_tokens = input.numel() / hidden_state;
  TORCH_CHECK(hidden_state == 4096, "Only support hidden_state == 4096 now!");

  scale3_async(input.data_ptr(), scale.data_ptr(), scale2_ptr, output.mutable_data_ptr(),
               output_scale2_ptr, output_fp32_ptr, num_tokens, hidden_state, is_moe, stream);

  return std::make_tuple(output, output_scale2, output_fp32);
}

}  // namespace scale
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "scale3(Tensor input, Tensor scale, Tensor scale2, bool is_moe) -> (Tensor, Tensor, Tensor)");
  m.impl("scale3", torch::kCUDA, &hpc::scale::scale3_entry);
}
