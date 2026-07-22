// Copyright (C) 2026 Tencent.

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include <limits>
#include <tuple>
#include <vector>

#include "src/activation/activation.h"

namespace hpc {
namespace activation {

torch::Tensor act_mul_and_quant_entry(const torch::Tensor &input, const torch::Tensor &scale,
                                      bool use_bf16_mul, std::optional<torch::Tensor> output) {
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

  std::vector<int64_t> output_shape(input.sizes().begin(), input.sizes().end());
  output_shape[output_shape.size() - 1] /= 2;

  auto options = input.options().dtype(torch::kFloat8_e4m3fn);

  torch::Tensor output_tensor;
  if (output.has_value()) {
    output_tensor = output.value();
  } else {
    output_tensor = torch::empty(output_shape, options);
  }

  using Tin = __nv_bfloat16;
  using Tout = __nv_fp8_e4m3;

  const auto *input_ptr = reinterpret_cast<const Tin *>(input.const_data_ptr());
  auto *output_ptr = reinterpret_cast<Tout *>(output_tensor.mutable_data_ptr());
  const float *scale_ptr = scale.const_data_ptr<float>();

  auto input_shape = input.sizes();
  int num_col = input_shape[input_shape.size() - 1];
  int num_row = 1;
  for (uint32_t i = 0; i < input_shape.size() - 1; ++i) {
    num_row *= input_shape[i];
  }

  act_mul_and_quant_async(output_ptr, input_ptr, scale_ptr, num_row, num_col, use_bf16_mul, stream);

  return output_tensor;
}

torch::Tensor masked_act_mul_and_quant_entry(const torch::Tensor &input, torch::Tensor &scale,
                                             const torch::Tensor &num_per_expert,
                                             std::optional<torch::Tensor> output) {
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

  TORCH_CHECK(input.is_contiguous(), "input tensor must be contiguous");
  TORCH_CHECK(scale.is_contiguous(), "scale tensor must be contiguous");
  TORCH_CHECK(num_per_expert.is_contiguous(), "num_per_expert tensor must be contiguous");

  TORCH_CHECK(input.device().is_cuda(), "input tensor's device must be cuda");
  TORCH_CHECK(scale.device().is_cuda(), "scale tensor's device must be cuda");
  TORCH_CHECK(num_per_expert.device().is_cuda(), "num_per_expert tensor's device must be cuda");

  TORCH_CHECK(input.size(-1) / 2 % 8 == 0, "hidden dim must be divided by 8")
  TORCH_CHECK(scale.numel() == 1, "only support per tensor qunat")

  std::vector<int64_t> output_shape(input.sizes().begin(), input.sizes().end());
  output_shape[output_shape.size() - 1] /= 2;

  auto options = input.options().dtype(torch::kFloat8_e4m3fn);

  torch::Tensor output_tensor;
  if (output.has_value()) {
    output_tensor = output.value();
  } else {
    output_tensor = torch::empty(output_shape, options);
  }

  using Tin = __nv_bfloat16;
  using Tout = __nv_fp8_e4m3;

  const auto *input_ptr = reinterpret_cast<const Tin *>(input.const_data_ptr());
  const auto *scale_ptr = reinterpret_cast<const float *>(scale.const_data_ptr());
  auto *output_ptr = reinterpret_cast<Tout *>(output_tensor.mutable_data_ptr());

  const auto *num_per_expert_ptr = num_per_expert.const_data_ptr<int>();

  int num_experts = num_per_expert.size(0);
  int num_total_tokens = input.size(0);
  int num_tokens_per_expert = num_total_tokens / num_experts;

  int num_intermediate_size = input.size(1) / 2;

  masked_act_mul_and_quant_async(output_ptr, input_ptr, scale_ptr, num_per_expert_ptr,
                                 num_total_tokens, num_intermediate_size, num_tokens_per_expert,
                                 stream);

  return output_tensor;
}

std::tuple<torch::Tensor, torch::Tensor> masked_act_mul_and_blockwise_quant_entry(
    const torch::Tensor &input, const torch::Tensor &num_per_expert,
    std::optional<torch::Tensor> output, std::optional<torch::Tensor> output_scale) {
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

  TORCH_CHECK(input.is_contiguous(), "input tensor must be contiguous");
  TORCH_CHECK(num_per_expert.is_contiguous(), "num_per_expert tensor must be contiguous");

  TORCH_CHECK(input.device().is_cuda(), "input tensor's device must be cuda");
  TORCH_CHECK(num_per_expert.device().is_cuda(), "num_per_expert tensor's device must be cuda");

  TORCH_CHECK(input.size(-1) / 2 % 128 == 0, "hidden dim must be divided by 128")

  std::vector<int64_t> output_shape(input.sizes().begin(), input.sizes().end());
  output_shape[output_shape.size() - 1] /= 2;

  auto output_scale_shape = output_shape;
  output_scale_shape[output_scale_shape.size() - 1] /= 128;

  auto options = input.options();

  torch::Tensor output_tensor;
  if (output.has_value()) {
    output_tensor = output.value();
  } else {
    output_tensor = torch::empty(output_shape, options.dtype(torch::kFloat8_e4m3fn));
  }
  torch::Tensor output_scale_tensor;
  if (output_scale.has_value()) {
    output_scale_tensor = output_scale.value();
  } else {
    output_scale_tensor = torch::empty(output_scale_shape, options.dtype(torch::kFloat32));
  }

  using Tin = __nv_bfloat16;
  using Tout = __nv_fp8_e4m3;

  const auto *input_ptr = reinterpret_cast<const Tin *>(input.const_data_ptr());
  auto *output_ptr = reinterpret_cast<Tout *>(output_tensor.mutable_data_ptr());
  auto *output_scale_ptr = reinterpret_cast<float *>(output_scale_tensor.mutable_data_ptr());

  const auto *num_per_expert_ptr = num_per_expert.const_data_ptr<int>();

  int num_experts = num_per_expert.size(0);
  int num_total_tokens = input.size(0);
  int num_tokens_per_expert = num_total_tokens / num_experts;

  int num_intermediate_size = input.size(1) / 2;

  masked_act_mul_and_blockwise_quant_async(output_ptr, output_scale_ptr, input_ptr,
                                           num_per_expert_ptr, num_total_tokens,
                                           num_intermediate_size, num_tokens_per_expert, stream);

  return std::make_tuple(output_tensor, output_scale_tensor);
}

std::tuple<torch::Tensor, torch::Tensor> scaled_fp8_quant_entry(
    const torch::Tensor &input, std::optional<torch::Tensor> scale,
    std::optional<torch::Tensor> output) {
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

  TORCH_CHECK(input.device().is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.numel() > 0, "input must be non-empty");
  TORCH_CHECK(input.scalar_type() == torch::kFloat32 || input.scalar_type() == torch::kFloat16 ||
                  input.scalar_type() == torch::kBFloat16,
              "input dtype must be float32, float16, or bfloat16");

  auto output_tensor = output.has_value()
                           ? output.value()
                           : torch::empty_like(input, input.options().dtype(torch::kFloat8_e4m3fn));
  TORCH_CHECK(output_tensor.device().is_cuda(), "output must be a CUDA tensor");
  TORCH_CHECK(output_tensor.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(output_tensor.sizes() == input.sizes(), "output shape must match input shape");
  TORCH_CHECK(output_tensor.scalar_type() == torch::kFloat8_e4m3fn,
              "output dtype must be float8_e4m3fn");

  TORCH_CHECK(scale.has_value(), "scale is required for scaled_fp8_quant");
  torch::Tensor scale_tensor = scale.value();
  TORCH_CHECK(scale_tensor.device().is_cuda(), "scale must be a CUDA tensor");
  TORCH_CHECK(scale_tensor.scalar_type() == torch::kFloat32, "scale dtype must be float32");
  TORCH_CHECK(scale_tensor.numel() == 1, "scale must contain one element");

  using Tout = __nv_fp8_e4m3;
  auto *output_ptr = reinterpret_cast<Tout *>(output_tensor.mutable_data_ptr());
  auto *scale_ptr = scale_tensor.mutable_data_ptr<float>();
  const int64_t numel = input.numel();

  if (input.scalar_type() == torch::kBFloat16) {
    const auto *input_ptr = reinterpret_cast<const __nv_bfloat16 *>(input.const_data_ptr());
    scaled_fp8_quant_async(output_ptr, input_ptr, scale_ptr, numel, stream);
  } else if (input.scalar_type() == torch::kFloat16) {
    const auto *input_ptr = reinterpret_cast<const __half *>(input.const_data_ptr());
    scaled_fp8_quant_async(output_ptr, input_ptr, scale_ptr, numel, stream);
  } else {
    const auto *input_ptr = input.const_data_ptr<float>();
    scaled_fp8_quant_async(output_ptr, input_ptr, scale_ptr, numel, stream);
  }

  return std::make_tuple(output_tensor, scale_tensor);
}

}  // namespace activation
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "act_mul_and_quant(Tensor input, Tensor scale, bool use_bf16_mul, Tensor? output) -> "
      "(Tensor)");
  m.impl("act_mul_and_quant", torch::kCUDA, &hpc::activation::act_mul_and_quant_entry);
  m.def(
      "masked_act_mul_and_quant(Tensor input, Tensor scale, Tensor num_per_expert, Tensor? output) "
      "-> (Tensor)");
  m.impl("masked_act_mul_and_quant", torch::kCUDA,
         &hpc::activation::masked_act_mul_and_quant_entry);
  m.def(
      "masked_act_mul_and_blockwise_quant(Tensor input, Tensor num_per_expert, Tensor? output, "
      "Tensor? output_scale) -> (Tensor output, "
      "Tensor output_scale)");
  m.impl("masked_act_mul_and_blockwise_quant", torch::kCUDA,
         &hpc::activation::masked_act_mul_and_blockwise_quant_entry);
  m.def("scaled_fp8_quant(Tensor input, Tensor? scale, Tensor? output) -> (Tensor, Tensor)");
  m.impl("scaled_fp8_quant", torch::kCUDA, &hpc::activation::scaled_fp8_quant_entry);
}
