// Copyright 2025 hpc-ops authors

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include <tuple>

#include "src/normalization/fused_rms_norm_with_scale.h"

namespace hpc {
namespace normalization {
namespace fused_rms_norm_with_scale {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> entry(torch::Tensor &input,
                                                              torch::Tensor &weight,
                                                              torch::Tensor &scale, double eps,
                                                              bool is_moe) {
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

  TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && weight.scalar_type() == torch::kBFloat16,
              "input and weight must be bfloat16.");

  torch::Tensor output = torch::empty_like(input, torch::kFloat8_e4m3fn);
  torch::Tensor output_fp32 = torch::empty_like(input, torch::kFloat32);
  torch::Tensor output_scale2 = torch::empty_like(input, torch::kFloat8_e4m3fn);

  void *output_fp32_ptr = nullptr;
  void *output_scale2_ptr = nullptr;
  if (is_moe) {
    output_fp32_ptr = output_fp32.mutable_data_ptr();
    output_scale2_ptr = output_scale2.mutable_data_ptr();
  }

  int hidden_state = input.size(input.dim() - 1);
  int batch_size = input.numel() / hidden_state;

  auto running = fused_rms_norm_with_scale_async(
      input.data_ptr(), weight.data_ptr(), output.mutable_data_ptr(), output_fp32_ptr,
      output_scale2_ptr, scale.data_ptr(), eps, batch_size, hidden_state, is_moe, stream);

  TORCH_CHECK(running, "fused_rms_norm_with_scale_async launch failed!");

  return std::make_tuple(output, output_fp32, output_scale2);
}

}  // namespace fused_rms_norm_with_scale
}  // namespace normalization
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "fused_rms_norm_with_scale(Tensor! input, Tensor! weight, Tensor! scale, float eps, bool "
      "is_moe) -> (Tensor, Tensor, Tensor)");
  m.impl("fused_rms_norm_with_scale", torch::kCUDA,
         &hpc::normalization::fused_rms_norm_with_scale::entry);
}
