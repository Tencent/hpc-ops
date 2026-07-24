// Copyright 2025 hpc-ops authors

// ROCm PyTorch exposes the current stream through the HIP ATen headers. The
// "MasqueradingAsCUDA" stream is torch's official CUDA->HIP bridge (it is what
// hipify rewrites at::cuda::getCurrentCUDAStream into on ROCm).
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include <limits>
#include <tuple>

#include "src/amd/normalization/fused_rmsnorm_with_scale.h"

namespace hpc {
namespace normalization {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fused_rmsnorm_with_scale_entry(
    const torch::Tensor &input, const torch::Tensor &weight, const torch::Tensor &scale, double eps,
    bool is_moe) {
  auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA(input.get_device());

  TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && weight.scalar_type() == torch::kBFloat16,
              "input and weight must be bfloat16.");

  torch::Tensor output = torch::empty_like(input, torch::kFloat8_e4m3fn);

  // output_fp32 / output_scale2 are only produced (and only read by the Python
  // wrapper) on the MoE path; on the non-MoE path they are unused, so allocate
  // zero-element placeholders instead of full input-sized buffers (the fp32 one
  // is 4x the input bytes) to drop that per-call allocation cost.
  torch::Tensor output_fp32;
  torch::Tensor output_scale2;
  void *output_fp32_ptr = nullptr;
  void *output_scale2_ptr = nullptr;
  if (is_moe) {
    output_fp32 = torch::empty_like(input, torch::kFloat32);
    output_scale2 = torch::empty_like(input, torch::kFloat8_e4m3fn);
    output_fp32_ptr = output_fp32.mutable_data_ptr();
    output_scale2_ptr = output_scale2.mutable_data_ptr();
  } else {
    auto opts = input.options();
    output_fp32 = torch::empty({0}, opts.dtype(torch::kFloat32));
    output_scale2 = torch::empty({0}, opts.dtype(torch::kFloat8_e4m3fn));
  }

  int hidden_state = input.size(input.dim() - 1);
  int batch_size = input.numel() / hidden_state;

  const auto *input_ptr = input.const_data_ptr();
  const auto *weight_ptr = weight.const_data_ptr();
  const auto *scale_ptr = scale.const_data_ptr();
  auto *output_ptr = output.mutable_data_ptr();

  auto running = fused_rmsnorm_with_scale_async(input_ptr, weight_ptr, output_ptr, output_fp32_ptr,
                                                output_scale2_ptr, scale_ptr, eps, batch_size,
                                                hidden_state, is_moe, stream);

  TORCH_CHECK(running, "fused_rmsnorm_with_scale_async launch failed!");

  return std::make_tuple(output, output_fp32, output_scale2);
}

}  // namespace normalization
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "fused_rmsnorm_with_scale(Tensor input, Tensor weight, Tensor scale, float eps, bool "
      "is_moe) -> (Tensor, Tensor, Tensor)");
  m.impl("fused_rmsnorm_with_scale", torch::kCUDA,
         &hpc::normalization::fused_rmsnorm_with_scale_entry);
}
