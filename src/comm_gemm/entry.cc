// Copyright 2025 hpc-ops authors

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include "src/comm_gemm/comm_gemm.h"
#include "src/gemm/gemm.h"

namespace hpc {
namespace comm_gemm {

torch::Tensor fuse_gemm_reduce_scatter_entry(
    const torch::Tensor &x, const torch::Tensor &weight, const torch::Tensor &x_scale,
    const torch::Tensor &weight_scale, bool trans_xscale, std::optional<torch::Tensor> bias,
    torch::Tensor &output, torch::Tensor &signal, torch::Tensor &multimem_output,
    torch::Tensor &multimem_signal, int64_t num_comp_sm, int64_t num_comm_sm, int64_t rank,
    int64_t world_size) {
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  TORCH_CHECK(x.is_contiguous(), "x tensor must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight tensor must be contiguous");
  TORCH_CHECK(x_scale.is_contiguous(), "x_scale tensor must be contiguous");
  TORCH_CHECK(weight_scale.is_contiguous(), "weight_scale tensor must be contiguous");
  TORCH_CHECK(output.is_contiguous(), "output tensor must be contiguous");
  TORCH_CHECK(signal.is_contiguous(), "signal tensor must be contiguous");
  TORCH_CHECK(multimem_output.is_contiguous(), "multimem_output tensor must be contiguous");
  TORCH_CHECK(multimem_signal.is_contiguous(), "multimem_signal tensor must be contiguous");

  TORCH_CHECK(x.dtype() == torch::kFloat8_e4m3fn, "x dtype must be float8_e4m3");
  TORCH_CHECK(weight.dtype() == torch::kFloat8_e4m3fn, "weight dtype must be float8_e4m3");
  TORCH_CHECK(x_scale.dtype() == torch::kFloat32, "x_scale dtype must be float32");
  TORCH_CHECK(weight_scale.dtype() == torch::kFloat32, "weight_scale dtype must be float32");
  TORCH_CHECK(output.dtype() == torch::kBFloat16, "output dtype must be bfloat16");
  TORCH_CHECK(signal.dtype() == torch::kUInt64, "signal dtype must be uint64");
  TORCH_CHECK(multimem_output.dtype() == torch::kBFloat16,
              "multimem_output dtype must be bfloat16");
  TORCH_CHECK(multimem_signal.dtype() == torch::kUInt64, "multimem_signal dtype must be uint64");

  int m = x.size(0);
  int k = x.size(1);
  int n = weight.size(0);
  int num_block_n = weight_scale.size(0);
  int num_block_k = weight_scale.size(1);

  TORCH_CHECK(world_size == 8, "only world_size == 8 is supported, got ", world_size);
  TORCH_CHECK(rank >= 0 && rank < world_size, "rank must be in [0, world_size), got rank = ", rank,
              ", world_size = ", world_size);
  TORCH_CHECK(m % (64 * world_size) == 0, "m must be divisible by 64 * world_size (",
              64 * world_size, "), got m = ", m);
  TORCH_CHECK(k % 128 == 0, "k % 128 == 0 must be true");
  TORCH_CHECK(n % 8 == 0, "n % 8 == 0 must be true");
  TORCH_CHECK(num_block_n == (n + 127) / 128, "num_block_n == (n + 127) / 128 must be true");
  TORCH_CHECK(num_block_k == ((k + 127) / 128 + 3) / 4 * 4,
              "num_block_k == ((k + 127) / 128 + 3) / 4 * 4 must be true");
  TORCH_CHECK(num_block_k % 4 == 0, "weight_scale.size(1) % 4 must be 0");
  TORCH_CHECK(k <= 16384, "only support k <= 16384");

  TORCH_CHECK(x.size(1) == weight.size(1), "x and weight must have same k");

  const void *bias_ptr = nullptr;
  if (bias.has_value()) {
    TORCH_CHECK(bias.value().dtype() == torch::kFloat32, "bias dtype must be float32");
    TORCH_CHECK(bias.value().is_contiguous(), "bias tensor must be contiguous");
    TORCH_CHECK(bias.value().size(0) == n, "bias shape must be n");
    bias_ptr = bias.value().const_data_ptr();
  }

  auto options = x.options();
  torch::Tensor y = torch::empty({m, n}, options.dtype(torch::kBFloat16));

  const auto *x_ptr = x.const_data_ptr();
  const auto *weight_ptr = weight.const_data_ptr();
  const auto *x_scale_ptr = x_scale.const_data_ptr();
  const auto *weight_scale_ptr = weight_scale.const_data_ptr();
  auto *output_ptr = output.mutable_data_ptr();
  auto *multimem_output_ptr = multimem_output.mutable_data_ptr();
  auto *signal_ptr = signal.mutable_data_ptr();
  auto *multimem_signal_ptr = multimem_signal.mutable_data_ptr();

  if (trans_xscale) {
    int m_pad = (m + 3) / 4 * 4;
    auto scale_options = x_scale.options();
    torch::Tensor new_x_scale = torch::empty({k / 128, m_pad}, scale_options);
    auto *new_x_scale_ptr = new_x_scale.mutable_data_ptr();
    hpc::gemm::pad_and_transpose_async(new_x_scale_ptr, x_scale_ptr, m, x_scale.size(1), m_pad,
                                       stream);
    fuse_gemm_reduce_scatter_fp8_async(
        output_ptr, x_ptr, weight_ptr, new_x_scale_ptr, weight_scale_ptr, bias_ptr, m, n, k, m_pad,
        num_block_k, num_block_n, stream, signal_ptr, multimem_output_ptr, multimem_signal_ptr,
        num_comp_sm, num_comm_sm, rank, world_size);
  } else {
    int m_pad = x_scale.size(1);
    TORCH_CHECK(x_scale.size(0) == k / 128, "x_scale dim 0 must be k / 128");
    TORCH_CHECK(m_pad == (m + 3) / 4 * 4, "x_scale dim 1 must aligned to 4");
    fuse_gemm_reduce_scatter_fp8_async(output_ptr, x_ptr, weight_ptr, x_scale_ptr, weight_scale_ptr,
                                       bias_ptr, m, n, k, m_pad, num_block_k, num_block_n, stream,
                                       signal_ptr, multimem_output_ptr, multimem_signal_ptr,
                                       num_comp_sm, num_comm_sm, rank, world_size);
  }
  return output;
}

}  // namespace comm_gemm
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "fuse_gemm_reduce_scatter(Tensor x, Tensor weight, Tensor x_scale, Tensor weight_scale, bool "
      "trans_xscale, Tensor? bias, Tensor! output, Tensor! signal, Tensor! multimem_output, "
      "Tensor! multimem_signal, int num_comp_sm, int num_comm_sm, int rank, int world_size) "
      "-> (Tensor)");
  m.impl("fuse_gemm_reduce_scatter", torch::kCUDA, &hpc::comm_gemm::fuse_gemm_reduce_scatter_entry);
}
