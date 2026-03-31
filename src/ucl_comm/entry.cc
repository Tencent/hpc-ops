// Copyright 2025 hpc-ops authors

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include "src/ucl_comm/fuse_allreduce_combine.h"
#include "src/ucl_comm/fuse_allreduce_dispatch.h"

namespace hpc {

namespace ucl_comm {
void fuse_allreduce_dispatch_entry(const torch::Tensor &input, const torch::Tensor &mc_input,
                                   const torch::Tensor &mn_input, torch::Tensor &signal,
                                   int64_t rank, int64_t local_size, int64_t world_size,
                                   int64_t attn_dp_size, int64_t attn_tp_size, int64_t moe_ep_size,
                                   int64_t moe_tp_size, int64_t num_max_blocks,
                                   torch::Tensor &output, torch::Tensor &mc_output,
                                   torch::Tensor &mn_output, torch::Tensor &output_multinode_signal,
                                   int64_t world_rank, int64_t batch_size, int64_t num_qp) {
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

  TORCH_CHECK(input.is_contiguous(), "input tensor must be contigous");
  TORCH_CHECK(mc_input.is_contiguous(), "mc_input tensor must be contigous");
  TORCH_CHECK(mn_input.is_contiguous(), "mn_input tensor must be contigous");
  TORCH_CHECK(output.is_contiguous(), "output tensor must be contigous");
  TORCH_CHECK(mc_output.is_contiguous(), "mc_output tensor must be contigous");
  TORCH_CHECK(mn_output.is_contiguous(), "mn_output tensor must be contigous");
  TORCH_CHECK(signal.is_contiguous(), "signal tensor must be contigous");
  TORCH_CHECK(output_multinode_signal.is_contiguous(),
              "output_multinode_signal tensor must be contigous");

  TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "input tensor data type must be bfloat16");
  TORCH_CHECK(mc_input.scalar_type() == torch::kBFloat16,
              "mc_input tensor data type must be bfloat16");
  TORCH_CHECK(mn_input.scalar_type() == torch::kBFloat16,
              "mn_input tensor data type must be bfloat16");
  TORCH_CHECK(output.scalar_type() == torch::kBFloat16, "output tensor data type must be bfloat16");
  TORCH_CHECK(mc_output.scalar_type() == torch::kBFloat16,
              "mc_output tensor data type must be bfloat16");
  TORCH_CHECK(mn_output.scalar_type() == torch::kBFloat16,
              "mn_output tensor data type must be bfloat16");
  TORCH_CHECK(signal.scalar_type() == torch::kInt64, "signal tensor data type must be int64");
  TORCH_CHECK(output_multinode_signal.scalar_type() == torch::kInt64,
              "output_multinode_signal tensor data type must be int64");

  const auto *input_ptr = input.const_data_ptr();
  const auto *mc_input_ptr = mc_input.const_data_ptr();
  const auto *mn_input_ptr = mn_input.const_data_ptr();

  auto *output_ptr = output.mutable_data_ptr();
  auto *mc_output_ptr = mc_output.mutable_data_ptr();
  auto *mn_output_ptr = mn_output.mutable_data_ptr();

  auto *signal_ptr = signal.mutable_data_ptr();
  auto *output_multinode_signal_ptr = output_multinode_signal.mutable_data_ptr();

  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  TORCH_CHECK(hidden_size == 4096 || hidden_size == 7168, "unsupported hidden_size");
  bool ptrs_are_aligned = (reinterpret_cast<int64_t>(input_ptr) % 16 == 0 &&
                           reinterpret_cast<int64_t>(mc_input_ptr) % 16 == 0 &&
                           reinterpret_cast<int64_t>(mn_input_ptr) % 16 == 0 &&
                           reinterpret_cast<int64_t>(output_ptr) % 16 == 0 &&
                           reinterpret_cast<int64_t>(mc_output_ptr) % 16 == 0 &&
                           reinterpret_cast<int64_t>(mn_output_ptr) % 16 == 0 &&
                           reinterpret_cast<int64_t>(signal_ptr) % 16 == 0 &&
                           reinterpret_cast<int64_t>(output_multinode_signal_ptr) % 16 == 0);
  TORCH_CHECK(ptrs_are_aligned, "pointer must be aligned to 16");

  fuse_allreduce_dispatch_async(input_ptr, mc_input_ptr, mn_input_ptr, output_ptr, mc_output_ptr,
                                mn_output_ptr, signal_ptr, output_multinode_signal_ptr, rank,
                                local_size, world_size, attn_dp_size, attn_tp_size, moe_ep_size,
                                moe_tp_size, num_max_blocks, num_tokens, hidden_size, world_rank,
                                batch_size, num_qp, stream);
}

void fuse_allreduce_combine_entry(const torch::Tensor &input, const torch::Tensor &mc_input,
                                  const torch::Tensor &mn_input, torch::Tensor &signal,
                                  int64_t rank, int64_t local_size, int64_t world_size,
                                  int64_t attn_dp_size, int64_t attn_tp_size, int64_t moe_ep_size,
                                  int64_t moe_tp_size, int64_t num_max_blocks,
                                  torch::Tensor &output, torch::Tensor &mc_output,
                                  torch::Tensor &mn_output, torch::Tensor &output_multinode_signal,
                                  int64_t world_rank, int64_t batch_size, int64_t num_qp) {
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

  TORCH_CHECK(input.is_contiguous(), "input tensor must be contigous");
  TORCH_CHECK(mc_input.is_contiguous(), "mc_input tensor must be contigous");
  TORCH_CHECK(mn_input.is_contiguous(), "mn_input tensor must be contigous");
  TORCH_CHECK(output.is_contiguous(), "output tensor must be contigous");
  TORCH_CHECK(mc_output.is_contiguous(), "mc_output tensor must be contigous");
  TORCH_CHECK(mn_output.is_contiguous(), "mn_output tensor must be contigous");
  TORCH_CHECK(signal.is_contiguous(), "signal tensor must be contigous");
  TORCH_CHECK(output_multinode_signal.is_contiguous(),
              "output_multinode_signal tensor must be contigous");

  TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "input tensor data type must be bfloat16");
  TORCH_CHECK(mc_input.scalar_type() == torch::kBFloat16,
              "mc_input tensor data type must be bfloat16");
  TORCH_CHECK(mn_input.scalar_type() == torch::kBFloat16,
              "mn_input tensor data type must be bfloat16");
  TORCH_CHECK(output.scalar_type() == torch::kBFloat16, "output tensor data type must be bfloat16");
  TORCH_CHECK(mc_output.scalar_type() == torch::kBFloat16,
              "mc_output tensor data type must be bfloat16");
  TORCH_CHECK(mn_output.scalar_type() == torch::kBFloat16,
              "mn_output tensor data type must be bfloat16");
  TORCH_CHECK(signal.scalar_type() == torch::kInt64, "signal tensor data type must be int64");
  TORCH_CHECK(output_multinode_signal.scalar_type() == torch::kInt64,
              "output_multinode_signal tensor data type must be int64");

  const auto *input_ptr = input.const_data_ptr();
  const auto *mc_input_ptr = mc_input.const_data_ptr();
  const auto *mn_input_ptr = mn_input.const_data_ptr();

  auto *output_ptr = output.mutable_data_ptr();
  auto *mc_output_ptr = mc_output.mutable_data_ptr();
  auto *mn_output_ptr = mn_output.mutable_data_ptr();

  auto *signal_ptr = signal.mutable_data_ptr();
  auto *output_multinode_signal_ptr = output_multinode_signal.mutable_data_ptr();

  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  TORCH_CHECK(hidden_size == 4096 || hidden_size == 7168, "unsupported hidden_size");
  bool ptrs_are_aligned = (reinterpret_cast<int64_t>(input_ptr) % 16 == 0 &&
                           reinterpret_cast<int64_t>(mc_input_ptr) % 16 == 0 &&
                           reinterpret_cast<int64_t>(mn_input_ptr) % 16 == 0 &&
                           reinterpret_cast<int64_t>(output_ptr) % 16 == 0 &&
                           reinterpret_cast<int64_t>(mc_output_ptr) % 16 == 0 &&
                           reinterpret_cast<int64_t>(mn_output_ptr) % 16 == 0 &&
                           reinterpret_cast<int64_t>(signal_ptr) % 16 == 0 &&
                           reinterpret_cast<int64_t>(output_multinode_signal_ptr) % 16 == 0);
  TORCH_CHECK(ptrs_are_aligned, "pointer must be aligned to 16");

  fuse_allreduce_combine_async(input_ptr, mc_input_ptr, mn_input_ptr, output_ptr, mc_output_ptr,
                               mn_output_ptr, signal_ptr, output_multinode_signal_ptr, rank,
                               local_size, world_size, attn_dp_size, attn_tp_size, moe_ep_size,
                               moe_tp_size, num_max_blocks, num_tokens, hidden_size, world_rank,
                               batch_size, num_qp, stream);
}

}  // namespace ucl_comm

}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "fuse_allreduce_dispatch(Tensor input, Tensor mc_input, Tensor mn_input, "
      "Tensor signal, int rank, int local_size, int world_size, "
      "int attn_dp_size, int attn_tp_size, int moe_ep_size, int moe_tp_size, "
      "int num_max_blocks, Tensor! output, Tensor! "
      "mc_output, Tensor! mn_output, Tensor! output_multinode_signal, int world_rank, int "
      "batch_size, int num_qp) -> ()");
  m.impl("fuse_allreduce_dispatch", torch::kCUDA, &hpc::ucl_comm::fuse_allreduce_dispatch_entry);
}

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "fuse_allreduce_combine(Tensor input, Tensor mc_input, Tensor mn_input, "
      "Tensor signal, int rank, int local_size, int world_size, "
      "int attn_dp_size, int attn_tp_size, int moe_ep_size, int moe_tp_size, "
      "int num_max_blocks, Tensor! output, Tensor! "
      "mc_output, Tensor! mn_output, Tensor! output_multinode_signal, int world_rank, int "
      "batch_size, int num_qp) -> ()");
  m.impl("fuse_allreduce_combine", torch::kCUDA, &hpc::ucl_comm::fuse_allreduce_combine_entry);
}
