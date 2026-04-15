// Copyright 2025 hpc-ops authors

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cmath>
#include <tuple>

#include "src/hadamard/hadamard.h"

namespace hpc {
namespace hadamard {

torch::Tensor hadamard_transform_entry(const torch::Tensor& input) {
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

  TORCH_CHECK(input.scalar_type() == torch::kBFloat16 || input.scalar_type() == torch::kFloat32,
              "input must be bf16 or fp32.");
  TORCH_CHECK(input.device().is_cuda(), "input must be on cuda device.");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous.");
  TORCH_CHECK(input.dim() >= 2, "input must have at least 2 dimensions.");

  int n = input.size(input.dim() - 1);
  int num_rows = input.numel() / n;
  int input_elem_size = 2;
  if (input.scalar_type() == torch::kFloat32) {
    input_elem_size = 4;
  }

  TORCH_CHECK(n == 64, "currently only n=64 is supported for hadamard transform.");

  float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(n));

  torch::Tensor output = torch::empty_like(input);

  const auto* input_ptr = input.const_data_ptr();
  auto* output_ptr = output.mutable_data_ptr();

  auto running = hadamard_transform_async(input_ptr, output_ptr, inv_sqrt_d, n, num_rows,
                                          input_elem_size, stream);
  TORCH_CHECK(running, "hadamard_transform_async launch failed!");

  return output;
}

std::tuple<torch::Tensor, torch::Tensor> act_mul_hadamard_blockwise_quant_entry(
    const torch::Tensor& gate_up, double upper_max, int64_t block_size, bool use_pdl) {
  TORCH_CHECK(gate_up.scalar_type() == torch::kBFloat16, "gate_up must be bf16.");
  TORCH_CHECK(gate_up.device().is_cuda(), "gate_up must be on cuda device.");
  TORCH_CHECK(gate_up.is_contiguous(), "gate_up must be contiguous.");
  TORCH_CHECK(gate_up.dim() == 2, "gate_up must be 2-dimensional [num_rows, 2*num_col].");

  int num_rows = gate_up.size(0);
  int full_col = gate_up.size(1);  // 2 * num_col
  TORCH_CHECK(full_col % 2 == 0, "gate_up last dim must be even.");
  int num_col = full_col / 2;
  TORCH_CHECK(num_col % block_size == 0,
              "num_col (gate_up.size(1)/2) must be a multiple of block_size.");

  int num_col_blocks = num_col / static_cast<int>(block_size);  // number of scale blocks per row

  auto options = gate_up.options();
  torch::Tensor output = torch::empty({num_rows, num_col}, options.dtype(torch::kFloat8_e4m3fn));
  // Scale layout [num_col_blocks, num_rows]: N-major (same convention as act.py blockwise quant)
  torch::Tensor output_scale =
      torch::empty({num_col_blocks, num_rows}, options.dtype(torch::kFloat32));

  auto stream = at::cuda::getCurrentCUDAStream(gate_up.get_device());

  using Tin = __nv_bfloat16;
  using Tout = __nv_fp8_e4m3;

  const auto* gate_up_ptr = reinterpret_cast<const Tin*>(gate_up.const_data_ptr());
  auto* output_ptr = reinterpret_cast<Tout*>(output.mutable_data_ptr());
  auto* output_scale_ptr = output_scale.mutable_data_ptr<float>();
  const int* valid_row_range_ptr = nullptr;

  auto running = act_mul_hadamard_blockwise_quant_async(
      gate_up_ptr, output_ptr, output_scale_ptr, valid_row_range_ptr, num_rows, num_col,
      static_cast<float>(upper_max), static_cast<int>(block_size), use_pdl, stream);
  TORCH_CHECK(running, "act_mul_hadamard_blockwise_quant_async launch failed!");

  return std::make_tuple(output, output_scale);
}

torch::Tensor act_mul_hadamard_per_tensor_quant_entry(const torch::Tensor& gate_up,
                                                      const torch::Tensor& scale_inv,
                                                      bool use_pdl) {
  TORCH_CHECK(gate_up.scalar_type() == torch::kBFloat16, "gate_up must be bf16.");
  TORCH_CHECK(gate_up.device().is_cuda(), "gate_up must be on cuda device.");
  TORCH_CHECK(gate_up.is_contiguous(), "gate_up must be contiguous.");
  TORCH_CHECK(gate_up.dim() == 2, "gate_up must be 2-dimensional [num_rows, 2*num_col].");

  TORCH_CHECK(scale_inv.scalar_type() == torch::kFloat32, "scale_inv must be float32.");
  TORCH_CHECK(scale_inv.device().is_cuda(), "scale_inv must be on cuda device.");
  TORCH_CHECK(scale_inv.numel() == 1, "scale_inv must contain exactly one element.");

  int num_rows = gate_up.size(0);
  int full_col = gate_up.size(1);
  TORCH_CHECK(full_col % 2 == 0, "gate_up last dim must be even.");
  int num_col = full_col / 2;
  TORCH_CHECK(num_col % 64 == 0, "num_col (gate_up.size(1)/2) must be a multiple of 64.");

  auto options = gate_up.options();
  torch::Tensor output = torch::empty({num_rows, num_col}, options.dtype(torch::kFloat8_e4m3fn));

  auto stream = at::cuda::getCurrentCUDAStream(gate_up.get_device());

  using Tin = __nv_bfloat16;
  using Tout = __nv_fp8_e4m3;

  const auto* gate_up_ptr = reinterpret_cast<const Tin*>(gate_up.const_data_ptr());
  auto* output_ptr = reinterpret_cast<Tout*>(output.mutable_data_ptr());
  const auto* scale_inv_ptr = scale_inv.const_data_ptr<float>();
  const int* valid_row_range_ptr = nullptr;

  auto running = act_mul_hadamard_per_tensor_quant_async(gate_up_ptr, output_ptr, scale_inv_ptr,
                                                         valid_row_range_ptr, num_rows, num_col,
                                                         use_pdl, stream);
  TORCH_CHECK(running, "act_mul_hadamard_per_tensor_quant_async launch failed!");

  return output;
}

}  // namespace hadamard
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def("hadamard_transform(Tensor input) -> Tensor");
  m.impl("hadamard_transform", torch::kCUDA, &hpc::hadamard::hadamard_transform_entry);

  m.def(
      "act_mul_hadamard_blockwise_quant(Tensor gate_up, float upper_max, int block_size, bool "
      "use_pdl) -> (Tensor output, Tensor output_scale)");
  m.impl("act_mul_hadamard_blockwise_quant", torch::kCUDA,
         &hpc::hadamard::act_mul_hadamard_blockwise_quant_entry);

  m.def(
      "act_mul_hadamard_per_tensor_quant(Tensor gate_up, Tensor scale_inv, bool "
      "use_pdl) -> Tensor");
  m.impl("act_mul_hadamard_per_tensor_quant", torch::kCUDA,
         &hpc::hadamard::act_mul_hadamard_per_tensor_quant_entry);
}
