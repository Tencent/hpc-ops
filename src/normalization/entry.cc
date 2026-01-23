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
#include "src/normalization/fused_rmsnorm_blockwise_quant.h"
#include "src/normalization/fused_rmsnorm_rope.h"

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

std::tuple<torch::Tensor, std::optional<torch::Tensor>> fused_rmsnorm_blockwise_quant_entry(
    const torch::Tensor &input, const torch::Tensor &weight, double eps,
    const bool with_blockwise_quant, const int64_t quant_size) {
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

  TORCH_CHECK(input.dim() == 2, "input dim must be 2");
  TORCH_CHECK(input.size(-1) == 128 || input.size(-1) == 512 || input.size(-1) == 1024 ||
                  input.size(-1) == 4096,
              "now only support dim 128/512/1024/4096");
  TORCH_CHECK(input.dtype() == torch::kBFloat16, "input dtype must be bfloat16");
  TORCH_CHECK(weight.dtype() == torch::kBFloat16, "weight dtype must be bfloat16");
  TORCH_CHECK(weight.size(-1) == input.size(-1), "weight.size(-1) == input.size(-1) must be true");

  const auto *input_ptr = input.const_data_ptr();
  const auto *weight_ptr = weight.const_data_ptr();

  int m = input.size(0);
  int hidden_size = input.size(1);
  TORCH_CHECK(hidden_size % 128 == 0, "hidden_size % 128 == 0 must be true");

  auto options = input.options();

  if (with_blockwise_quant) {
    auto y = torch::empty({m, hidden_size}, options.dtype(torch::kFloat8_e4m3fn));
    auto y_scale = torch::empty({m, hidden_size / 128}, options.dtype(torch::kFloat32));
    auto *y_ptr = y.mutable_data_ptr();
    auto *y_scale_ptr = y_scale.mutable_data_ptr();
    fused_rmsnorm_blockwise_quant_async(y_ptr, y_scale_ptr, input_ptr, weight_ptr, m, hidden_size,
                                        eps, with_blockwise_quant, stream);
    return std::make_tuple(y, y_scale);
  } else {
    auto y = torch::empty({m, hidden_size}, options);
    auto *y_ptr = y.mutable_data_ptr();
    fused_rmsnorm_blockwise_quant_async(y_ptr, nullptr, input_ptr, weight_ptr, m, hidden_size, eps,
                                        with_blockwise_quant, stream);
    return std::make_tuple(y, std::nullopt);
  }
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>> fused_rmsnorm_rope_entry(
    const torch::Tensor &positions, const torch::Tensor &q, std::optional<torch::Tensor> q_weight,
    std::optional<torch::Tensor> k, std::optional<torch::Tensor> k_weight,
    const torch::Tensor &cos_sin_cache, const double eps) {
  auto stream = at::cuda::getCurrentCUDAStream(q.get_device());

  int num_tokens = q.size(0);
  int num_q_heads = q.size(1);
  int dim = q.size(2);
  int num_k_heads = 1;
  TORCH_CHECK(positions.dim() == 1, "position.dim() must be true");
  TORCH_CHECK(positions.size(0) == q.size(0), "positions.size(0) == q.size(0) must be true");
  TORCH_CHECK(q.dim() == 3, "q.dim() == 3 must be true");
  TORCH_CHECK(q.dtype() == torch::kBFloat16, "now only support bfloat16 for q")
  TORCH_CHECK(positions.dtype() == torch::kInt64, "positions dtype must be int64");
  if (q_weight.has_value()) {
    TORCH_CHECK(q_weight.value().size(-1) == q.size(-1), "q_weight must has same dim with q");
    TORCH_CHECK(q_weight.value().dtype() == q.dtype(), "q_weight must has same dtype with q");
  }
  if (k.has_value()) {
    TORCH_CHECK(k.value().dtype() == torch::kBFloat16, "now only support bfloat16 for k")
    TORCH_CHECK(k.value().dim() == 3, "k.dim() must be 3");
    TORCH_CHECK(k.value().size(-1) == q.size(-1), "k must has same dim with q");
    TORCH_CHECK(k.value().size(0) == q.size(0), "k.value().size(0) == q.size(0) must be true");
    num_k_heads = k.value().size(1);
    TORCH_CHECK(num_q_heads >= num_k_heads, "now only support num_q_heads >= num_k_heads");
  }
  if (k_weight.has_value()) {
    TORCH_CHECK(k.has_value(), "when k_weight is given, k must be provided");
    TORCH_CHECK(k_weight.value().size(-1) == k.value().size(-1), "k_weight must has dim with k");
    TORCH_CHECK(k_weight.value().dtype() == k.value().dtype(),
                "k_weight must has same dtype with k");
  }

  TORCH_CHECK(cos_sin_cache.dtype() == torch::kBFloat16,
              "cos_sim_cache must has same dtype with q");

  TORCH_CHECK(dim == 128 || dim == 512, "now only support dim 128/512");
  int rope_dim = cos_sin_cache.size(-1);

  auto options = q.options();
  auto y_q = torch::empty({num_tokens, num_q_heads, dim}, options);

  const auto *q_ptr = q.const_data_ptr();
  const auto *pos_ptr = positions.const_data_ptr();
  auto *y_q_ptr = y_q.mutable_data_ptr();
  void *y_k_ptr = nullptr;
  const void *q_weight_ptr = nullptr;
  const void *k_ptr = nullptr;
  const void *k_weight_ptr = nullptr;
  const auto *cos_sin_ptr = cos_sin_cache.const_data_ptr();

  if (q_weight.has_value()) {
    q_weight_ptr = q_weight.value().const_data_ptr();
  }
  if (k.has_value()) {
    if (k_weight.has_value()) {
      k_weight_ptr = k_weight.value().const_data_ptr();
    }
    k_ptr = k.value().const_data_ptr();
    auto y_k = torch::empty({num_tokens, num_k_heads, dim}, options);
    auto *y_k_ptr = y_k.mutable_data_ptr();
    fused_rmsnorm_rope_async(y_q_ptr, y_k_ptr, q_ptr, q_weight_ptr, k_ptr, k_weight_ptr, pos_ptr,
                             cos_sin_ptr, num_tokens, dim, rope_dim, num_q_heads, num_k_heads, eps,
                             stream);
    return std::make_tuple(y_q, y_k);
  }
  fused_rmsnorm_rope_async(y_q_ptr, y_k_ptr, q_ptr, q_weight_ptr, k_ptr, k_weight_ptr, pos_ptr,
                           cos_sin_ptr, num_tokens, dim, rope_dim, num_q_heads, num_k_heads, eps,
                           stream);
  return std::make_tuple(y_q, std::nullopt);
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

  m.def(
      "fused_rmsnorm_blockwise_quant(Tensor input, Tensor weight,"
      "float eps, bool with_blockwise_quant, int block_size) -> (Tensor, Tensor ?)");
  m.impl("fused_rmsnorm_blockwise_quant", torch::kCUDA,
         &hpc::normalization::fused_rmsnorm_blockwise_quant_entry);

  m.def(
      "fused_rmsnorm_rope(Tensor positions, Tensor q, Tensor ? q_weight,"
      "Tensor ? k, Tensor ? k_weight, Tensor cos_sin_cache, float eps) -> (Tensor, Tensor ?)");
  m.impl("fused_rmsnorm_rope", torch::kCUDA, &hpc::normalization::fused_rmsnorm_rope_entry);
}
