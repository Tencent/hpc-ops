// Copyright 2025 hpc-ops authors

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include "src/allreduce/fuse_allreduce_rmsnorm.h"
#include "src/allreduce/fuse_allreduce_rmsnorm_with_scale.h"

namespace hpc {

namespace allreduce {
void fuse_allreduce_rmsnorm_entry(const torch::Tensor &input,     // [..., hidden_size]
                                  const torch::Tensor &mc_input,  // [..., hidden_size] multimem_ptr
                                  const torch::Tensor &in_residual,  // [..., hidden_size]
                                  const torch::Tensor &weight,       // [hidden_size]
                                  torch::Tensor &signal,             // [world_size] signal ptrs
                                  int64_t rank, int64_t world_size, int64_t num_max_blocks,
                                  double rms_norm_eps,
                                  torch::Tensor &output,          // [..., hidden_size]
                                  torch::Tensor &mc_output,       // [..., hidden_size] multimem_ptr
                                  torch::Tensor &out_residual) {  // [..., hidden_size]
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

  TORCH_CHECK(input.is_contiguous(), "input tensor must be contigous");
  TORCH_CHECK(mc_input.is_contiguous(), "mc_input tensor must be contigous");
  TORCH_CHECK(in_residual.is_contiguous(), "input residual tensor must be contigous");
  TORCH_CHECK(output.is_contiguous(), "output tensor must be contigous");
  TORCH_CHECK(mc_output.is_contiguous(), "mc_output tensor must be contigous");
  TORCH_CHECK(out_residual.is_contiguous(), "output residual tensor must be contigous");
  TORCH_CHECK(weight.is_contiguous(), "weight tensor must be contigous");
  TORCH_CHECK(signal.is_contiguous(), "signal tensor must be contigous");

  TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "input tensor data type must be bfloat16");
  TORCH_CHECK(mc_input.scalar_type() == torch::kBFloat16,
              "mc_input tensor data type must be bfloat16");
  TORCH_CHECK(in_residual.scalar_type() == torch::kBFloat16,
              "residual tensor data type must be bfloat16");
  TORCH_CHECK(output.scalar_type() == torch::kBFloat16, "output tensor data type must be bfloat16");
  TORCH_CHECK(mc_output.scalar_type() == torch::kBFloat16,
              "mc_output tensor data type must be bfloat16");
  TORCH_CHECK(out_residual.scalar_type() == torch::kBFloat16,
              "output residual tensor data type must be bfloat16");
  TORCH_CHECK(weight.scalar_type() == torch::kBFloat16, "weight tensor data type must be bfloat16");
  TORCH_CHECK(signal.scalar_type() == torch::kInt64, "signal tensor data type must be int64");

  const auto *input_ptr = input.const_data_ptr();
  const auto *mc_input_ptr = mc_input.const_data_ptr();
  const auto *in_res_ptr = in_residual.const_data_ptr();
  const auto *weight_ptr = weight.const_data_ptr();

  auto *output_ptr = output.mutable_data_ptr();
  auto *mc_output_ptr = mc_output.mutable_data_ptr();
  auto *out_res_ptr = out_residual.mutable_data_ptr();
  auto *signal_ptr = signal.mutable_data_ptr();

  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  TORCH_CHECK(hidden_size == 4096 || hidden_size == 5120, "unsupported hidden_size");
  bool ptrs_are_aligned = (reinterpret_cast<int64_t>(input_ptr) % 16 == 0 &&
                           reinterpret_cast<int64_t>(mc_input_ptr) % 16 == 0 &&
                           reinterpret_cast<int64_t>(in_res_ptr) % 16 == 0 &&
                           reinterpret_cast<int64_t>(output_ptr) % 16 == 0 &&
                           reinterpret_cast<int64_t>(mc_output_ptr) % 16 == 0 &&
                           reinterpret_cast<int64_t>(out_res_ptr) % 16 == 0 &&
                           reinterpret_cast<int64_t>(weight_ptr) % 16 == 0 &&
                           reinterpret_cast<int64_t>(signal_ptr) % 16 == 0);
  TORCH_CHECK(ptrs_are_aligned, "pointer must be aligned to 16");

  fuse_allreduce_rmsnorm_async(input_ptr, mc_input_ptr, in_res_ptr, weight_ptr, output_ptr,
                               mc_output_ptr, out_res_ptr, signal_ptr, rank, world_size,
                               num_max_blocks, rms_norm_eps, num_tokens, hidden_size, stream);
}

void fuse_allreduce_rmsnorm_with_scale_entry(
    torch::Tensor &input,          // [num_tokens, hidden_size]
    torch::Tensor &mc_input,       // [num_tokens, hidden_size] multimem_ptr
    torch::Tensor &in_residual,    // [num_tokens, hidden_size]
    torch::Tensor &weight,         // [hidden_size]
    torch::Tensor &scale,          // [1]
    torch::Tensor &mc_fp8_output,  // [num_tokens, hidden_size] multimem_ptr
    torch::Tensor &signal,         // [world_size] signal pads ptrs
    int64_t rank, int64_t world_size, int64_t num_max_blocks, double rms_norm_eps, bool is_moe,
    torch::Tensor &out_residual,                    // [num_tokens, hidden_size]
    std::optional<torch::Tensor> scale2,            // [1]
    std::optional<torch::Tensor> mc_fp8_output2,    // [num_tokens, hidden_size] multimem_ptr
    std::optional<torch::Tensor> mc_fp32_output) {  // [num_tokens, hidden_size] multimem_ptr)
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

  TORCH_CHECK(mc_input.is_contiguous(), "mc_input tensor must be contigous");
  TORCH_CHECK(in_residual.is_contiguous(), "input residual tensor must be contigous");
  TORCH_CHECK(out_residual.is_contiguous(), "output residual tensor must be contigous");
  TORCH_CHECK(weight.is_contiguous(), "weight tensor must be contigous");
  TORCH_CHECK(scale.is_contiguous(), "scale tensor must be contigous");

  TORCH_CHECK(mc_fp8_output.is_contiguous(), "mc_fp8_output tensor must be contigous");
  TORCH_CHECK(signal.is_contiguous(), "signal tensor must be contigous");
  TORCH_CHECK(mc_input.scalar_type() == torch::kBFloat16,
              "mc_input tensor data type must be bfloat16");
  TORCH_CHECK(in_residual.scalar_type() == torch::kBFloat16,
              "residual tensor data type must be bfloat16");
  TORCH_CHECK(out_residual.scalar_type() == torch::kBFloat16,
              "output residual tensor data type must be bfloat16");
  TORCH_CHECK(weight.scalar_type() == torch::kBFloat16, "weight tensor data type must be bfloat16");
  TORCH_CHECK(scale.scalar_type() == torch::kFloat, "scale tensor data type must be float32");
  TORCH_CHECK(mc_fp8_output.scalar_type() == torch::kFloat8_e4m3fn,
              "mc_fp8_output tensor data type must be fp8");
  if (is_moe) {
    TORCH_CHECK(scale2.value().scalar_type() == torch::kFloat,
                "scale2 tensor data type must be float32");
    TORCH_CHECK(scale2.value().is_contiguous(), "scale2 tensor must be contigous");
    TORCH_CHECK(mc_fp8_output2.value().scalar_type() == torch::kFloat8_e4m3fn,
                "if is_moe is true, mc_fp8_output2 tensor data type must be fp8");
    TORCH_CHECK(mc_fp32_output.value().scalar_type() == torch::kFloat,
                "if is_moe is true, mc_fp32_output tensor data type must be float32");
  }
  TORCH_CHECK(signal.scalar_type() == torch::kInt64, "signal tensor data type must be int64");

  auto *mc_input_ptr = mc_input.data_ptr();
  auto *in_res_ptr = in_residual.data_ptr();
  auto *out_res_ptr = out_residual.mutable_data_ptr();
  auto *weight_ptr = weight.data_ptr();
  auto *scale_ptr = scale.data_ptr();
  auto *mc_fp8_output_ptr = mc_fp8_output.data_ptr();
  void *mc_fp8_output2_ptr = nullptr;
  void *mc_fp32_output_ptr = nullptr;
  void *scale2_ptr = nullptr;
  if (is_moe) {
    scale2_ptr = scale2.value().data_ptr();
    mc_fp8_output2_ptr = mc_fp8_output2.value().data_ptr();
    mc_fp32_output_ptr = mc_fp32_output.value().data_ptr();
  }
  auto *signal_ptr = signal.data_ptr();

  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  TORCH_CHECK(hidden_size == 4096 || hidden_size == 5120, "unsupported hidden_size");
  bool ptrs_are_aligned = (reinterpret_cast<int64_t>(mc_input_ptr) % 16 == 0 &&
                                   reinterpret_cast<int64_t>(in_res_ptr) % 16 == 0 &&
                                   reinterpret_cast<int64_t>(out_res_ptr) % 16 == 0 &&
                                   reinterpret_cast<int64_t>(weight_ptr) % 16 == 0 &&
                                   reinterpret_cast<int64_t>(scale_ptr) % 16 == 0 &&
                                   reinterpret_cast<int64_t>(mc_fp8_output_ptr) % 16 == 0 && is_moe
                               ? reinterpret_cast<int64_t>(scale2_ptr) % 16 == 0
                           : 1 && is_moe ? reinterpret_cast<int64_t>(mc_fp8_output2_ptr) % 16 == 0
                           : 1 && is_moe ? reinterpret_cast<int64_t>(mc_fp32_output_ptr) % 16 == 0
                                         : 1 && reinterpret_cast<int64_t>(weight_ptr) % 16 == 0 &&
                                               reinterpret_cast<int64_t>(signal_ptr) % 16 == 0);
  TORCH_CHECK(ptrs_are_aligned, "pointer must be aligned to 16");
  fused_allreduce_rmsnorm_with_scale_async(
      mc_input_ptr, in_res_ptr, out_res_ptr, weight_ptr, scale_ptr, scale2_ptr, mc_fp8_output_ptr,
      mc_fp8_output2_ptr, mc_fp32_output_ptr, signal_ptr, rank, world_size, num_max_blocks,
      rms_norm_eps, num_tokens, hidden_size, is_moe, stream);
}
}  // namespace allreduce

}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "fuse_allreduce_rmsnorm(Tensor input, Tensor mc_input, Tensor in_residual, Tensor weight, "
      "Tensor signal, "
      "int rank, int world_size, int num_max_blocks, float rms_norm_eps, Tensor output, Tensor "
      "mc_output, Tensor out_residual) -> ()");
  m.impl("fuse_allreduce_rmsnorm", torch::kCUDA, &hpc::allreduce::fuse_allreduce_rmsnorm_entry);

  m.def(
      "fuse_allreduce_rmsnorm_with_scale(Tensor input, Tensor mc_input, Tensor in_residual, Tensor "
      "weight, Tensor scale, Tensor mc_fp8_output, Tensor signal, "
      "int rank, int world_size, int num_max_blocks, float rms_norm_eps, bool is_moe, Tensor "
      "out_residual, Tensor ? scale2, Tensor ? mc_fp8_output2, Tensor ? mc_fp32_output) -> ()");
  m.impl("fuse_allreduce_rmsnorm_with_scale", torch::kCUDA,
         &hpc::allreduce::fuse_allreduce_rmsnorm_with_scale_entry);
}
