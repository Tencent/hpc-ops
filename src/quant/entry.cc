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
#include "src/quant/per_token_group_quant.h"

namespace hpc {
namespace quant {

std::tuple<torch::Tensor, torch::Tensor> per_token_group_quant_entry(const torch::Tensor& input,
                                                                     int64_t group_size,
                                                                     double quant_eps) {
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

  TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "input must be bfloat16.");
  TORCH_CHECK(input.device().is_cuda(), "input must be on cuda device.");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous.");

  int hidden_state = input.size(input.dim() - 1);
  int batch_size = input.numel() / hidden_state;
  int num_groups = (hidden_state + group_size - 1) / group_size;

  torch::Tensor output = torch::empty_like(input, torch::kFloat8_e4m3fn);
  torch::Tensor quant_scale =
      torch::empty({batch_size, num_groups},
                   torch::TensorOptions().device(input.device()).dtype(torch::kFloat32));

  const auto* input_ptr = input.const_data_ptr();
  auto* output_ptr = output.mutable_data_ptr();
  auto* quant_scale_ptr = quant_scale.mutable_data_ptr();
  // Get FP8 E4M3 format range using numeric_limits
  using FP8_E4M3 = cutlass::float_e4m3_t;
  const float fp8_e4m3_max = static_cast<float>(std::numeric_limits<FP8_E4M3>::max());
  const float fp8_e4m3_min = static_cast<float>(std::numeric_limits<FP8_E4M3>::lowest());

  auto running =
      per_token_group_quant_async(input_ptr, output_ptr, quant_scale_ptr, group_size, quant_eps,
                                  hidden_state, batch_size, fp8_e4m3_max, fp8_e4m3_min, stream);
  TORCH_CHECK(running, "per_token_group_quant_async launch failed!");

  return std::make_tuple(output, quant_scale);
}

}  // namespace quant
}  // namespace hpc
TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "per_token_group_quant(Tensor input, int group_size, float quant_eps) -> (Tensor, "
      "Tensor)");
  m.impl("per_token_group_quant", torch::kCUDA, &hpc::quant::per_token_group_quant_entry);
}
