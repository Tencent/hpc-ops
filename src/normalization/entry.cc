// Copyright 2025 hpc-ops authors

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include <limits>
#include <tuple>

#include "cutlass/float8.h"
#include "src/normalization/fused_layer_norm_with_scale_quant.h"
#include "src/normalization/fused_rms_norm_with_scale.h"

namespace hpc {
namespace normalization {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fused_rms_norm_with_scale_entry(
    const torch::Tensor &input, const torch::Tensor &weight, const torch::Tensor &scale, double eps,
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

  const auto *input_ptr = input.const_data_ptr();
  const auto *weight_ptr = weight.const_data_ptr();
  const auto *scale_ptr = scale.const_data_ptr();
  auto *output_ptr = output.mutable_data_ptr();

  auto running = fused_rms_norm_with_scale_async(input_ptr, weight_ptr, output_ptr, output_fp32_ptr,
                                                 output_scale2_ptr, scale_ptr, eps, batch_size,
                                                 hidden_state, is_moe, stream);

  TORCH_CHECK(running, "fused_rms_norm_with_scale_async launch failed!");

  return std::make_tuple(output, output_fp32, output_scale2);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fused_layer_norm_with_scale_quant_entry(
    const torch::Tensor &input, const torch::Tensor &weight, const torch::Tensor &bias,
    const torch::Tensor &pre_norm_scale1, const torch::Tensor &pre_norm_scale2,
    const torch::Tensor &post_norm_scale, const torch::Tensor &post_norm_bias_scale, double eps,
    double quant_eps, int64_t group_size, bool is_elementwise_affine, bool use_pre_norm_scale,
    bool use_post_norm_scale) {
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
  TORCH_CHECK(input.device().is_cuda(), "input tensor must be on cuda device.");
  TORCH_CHECK(input.is_contiguous(), "input tensor must be contiguous.");
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "input must be bfloat16.");
  if (is_elementwise_affine) {
    TORCH_CHECK(weight.device().is_cuda(), "weight tensor must be on cuda device.");
    TORCH_CHECK(bias.device().is_cuda(), "bias tensor must be on cuda device.");
    TORCH_CHECK(weight.is_contiguous(), "weight tensor must be contiguous.");
    TORCH_CHECK(bias.is_contiguous(), "bias tensor must be contiguous.");
    TORCH_CHECK(weight.scalar_type() == torch::kBFloat16, "weight must be bfloat16.");
    TORCH_CHECK(bias.scalar_type() == torch::kBFloat16, "bias must be bfloat16.");
  }
  if (use_pre_norm_scale) {
    TORCH_CHECK(pre_norm_scale1.device().is_cuda(),
                "pre_norm_scale1 tensor must be on cuda device.");
    TORCH_CHECK(pre_norm_scale2.device().is_cuda(),
                "pre_norm_scale2 tensor must be on cuda device.");
    TORCH_CHECK(pre_norm_scale1.is_contiguous(), "pre_norm_scale1 tensor must be contiguous.");
    TORCH_CHECK(pre_norm_scale2.is_contiguous(), "pre_norm_scale2 tensor must be contiguous.");
    TORCH_CHECK(pre_norm_scale1.scalar_type() == torch::kBFloat16,
                "pre_norm_scale1 must be bfloat16.");
    TORCH_CHECK(pre_norm_scale2.scalar_type() == torch::kBFloat16,
                "pre_norm_scale2 must be bfloat16.");
  }
  if (use_post_norm_scale) {
    TORCH_CHECK(post_norm_scale.device().is_cuda(),
                "post_norm_scale tensor must be on cuda device.");
    TORCH_CHECK(post_norm_bias_scale.device().is_cuda(),
                "post_norm_bias_scale tensor must be on cuda device.");
    TORCH_CHECK(post_norm_scale.is_contiguous(), "post_norm_scale tensor must be contiguous.");
    TORCH_CHECK(post_norm_bias_scale.is_contiguous(),
                "post_norm_bias_scale tensor must be contiguous.");
    TORCH_CHECK(post_norm_scale.scalar_type() == torch::kBFloat16,
                "post_norm_scale must be bfloat16.");
    TORCH_CHECK(post_norm_bias_scale.scalar_type() == torch::kBFloat16,
                "post_norm_bias_scale must be bfloat16.");
  }

  int hidden_state = input.size(input.dim() - 1);
  int batch_size = input.numel() / hidden_state;
  int num_groups = (hidden_state + group_size - 1) / group_size;

  torch::Tensor output = torch::empty_like(input, torch::kFloat8_e4m3fn);
  torch::Tensor quant_scale =
      torch::empty({batch_size, num_groups},
                   torch::TensorOptions().device(input.device()).dtype(torch::kFloat32));
  torch::Tensor output_x = torch::empty_like(input, torch::kBFloat16);

  // Get FP8 E4M3 format range using numeric_limits
  using FP8_E4M3 = cutlass::float_e4m3_t;
  const float fp8_e4m3_max = static_cast<float>(std::numeric_limits<FP8_E4M3>::max());
  const float fp8_e4m3_min = static_cast<float>(std::numeric_limits<FP8_E4M3>::lowest());
  const auto *input_ptr = input.const_data_ptr();
  const auto *weight_ptr = weight.const_data_ptr();
  const auto *bias_ptr = bias.const_data_ptr();
  const auto *pre_norm_scale1_ptr = pre_norm_scale1.const_data_ptr();
  const auto *pre_norm_scale2_ptr = pre_norm_scale2.const_data_ptr();
  const auto *post_norm_scale_ptr = post_norm_scale.const_data_ptr();
  const auto *post_norm_bias_scale_ptr = post_norm_bias_scale.const_data_ptr();
  auto *output_ptr = output.mutable_data_ptr();
  auto *quant_scale_ptr = quant_scale.mutable_data_ptr();
  auto *output_x_ptr = output_x.mutable_data_ptr();

  auto running = fused_layer_norm_with_scale_quant_async(
      input_ptr, weight_ptr, bias_ptr, output_ptr, pre_norm_scale1_ptr, pre_norm_scale2_ptr,
      post_norm_scale_ptr, post_norm_bias_scale_ptr, quant_scale_ptr, output_x_ptr, eps, quant_eps,
      batch_size, hidden_state, group_size, fp8_e4m3_max, fp8_e4m3_min, is_elementwise_affine,
      use_pre_norm_scale, use_post_norm_scale, stream);

  TORCH_CHECK(running, "fused_layer_norm_with_scale_quant_async launch failed!");

  return std::make_tuple(output, quant_scale, output_x);
}

}  // namespace normalization
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "fused_rms_norm_with_scale(Tensor input, Tensor weight, Tensor scale, float eps, bool "
      "is_moe) -> (Tensor, Tensor, Tensor)");
  m.impl("fused_rms_norm_with_scale", torch::kCUDA,
         &hpc::normalization::fused_rms_norm_with_scale_entry);

  m.def(
      "fused_layer_norm_with_scale_quant(Tensor input, Tensor weight, Tensor bias, Tensor "
      "pre_norm_scale1, Tensor pre_norm_scale2, Tensor post_norm_scale, Tensor "
      "post_norm_bias_scale, float eps, float quant_eps, int group_size, bool "
      "is_elementwise_affine, bool use_pre_norm_scale,bool use_post_norm_scale) -> (Tensor, "
      "Tensor, Tensor)");
  m.impl("fused_layer_norm_with_scale_quant", torch::kCUDA,
         &hpc::normalization::fused_layer_norm_with_scale_quant_entry);
}
