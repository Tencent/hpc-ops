// Copyright 2025 hpc-ops authors

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include "src/allreduce/fuse_allreduce_rmsnorm_high_throughput.h"
#include "src/allreduce/fuse_allreduce_rmsnorm_low_latency.h"

namespace hpc {

namespace allreduce {
void fuse_allreduce_rmsnorm_high_throughput_entry(
    const torch::Tensor &input,        // [..., hidden_size]
    const torch::Tensor &mc_input,     // [..., hidden_size] multimem_ptr
    const torch::Tensor &in_residual,  // [..., hidden_size]
    const torch::Tensor &weight,       // [hidden_size]
    torch::Tensor &signal,             // [world_size] signal ptrs
    int64_t rank, int64_t world_size, int64_t num_max_blocks, double rms_norm_eps,
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

  TORCH_CHECK(hidden_size == 4096 || hidden_size == 5120 || hidden_size == 7168,
              "unsupported hidden_size");
  bool ptrs_are_aligned = (reinterpret_cast<int64_t>(input_ptr) % 16 == 0 &&
                           reinterpret_cast<int64_t>(mc_input_ptr) % 16 == 0 &&
                           reinterpret_cast<int64_t>(in_res_ptr) % 16 == 0 &&
                           reinterpret_cast<int64_t>(output_ptr) % 16 == 0 &&
                           reinterpret_cast<int64_t>(mc_output_ptr) % 16 == 0 &&
                           reinterpret_cast<int64_t>(out_res_ptr) % 16 == 0 &&
                           reinterpret_cast<int64_t>(weight_ptr) % 16 == 0 &&
                           reinterpret_cast<int64_t>(signal_ptr) % 16 == 0);
  TORCH_CHECK(ptrs_are_aligned, "pointer must be aligned to 16");

  fuse_allreduce_rmsnorm_high_throughput_async(
      input_ptr, mc_input_ptr, in_res_ptr, weight_ptr, output_ptr, mc_output_ptr, out_res_ptr,
      signal_ptr, rank, world_size, num_max_blocks, rms_norm_eps, num_tokens, hidden_size, stream);
}

void fuse_allreduce_rmsnorm_low_latency_entry(
    const torch::Tensor &input_x,           // [num_tokens, token_dim]
    const torch::Tensor &multicast_x,       // [num_tokens, token_dim] multimem_ptr tensor
    const torch::Tensor &data_buffer_ptrs,  // [world_size] int64 pointers to remote buffers
    torch::Tensor &multinode_x,             // local lamport buffer tensor
    const torch::Tensor &buffer_flags,      // lamport flag buffer
    int64_t world_size, int64_t rank, bool rmsnorm_fusion, bool launch_with_pdl, bool use_two_shot,
    torch::Tensor &output_x,            // [num_tokens, token_dim]
    torch::Tensor &residual_out,        // [num_tokens, token_dim]
    const torch::Tensor &residual_in,   // [num_tokens, token_dim]
    const torch::Tensor &weight_gamma,  // [token_dim]
    double rms_norm_eps) {
  auto stream = at::cuda::getCurrentCUDAStream(input_x.get_device());

  // Contiguous checks
  TORCH_CHECK(input_x.is_contiguous(), "input_x tensor must be contigous");
  TORCH_CHECK(multicast_x.is_contiguous(), "multicast_x tensor must be contigous");
  TORCH_CHECK(data_buffer_ptrs.is_contiguous(), "data_buffer_ptrs tensor must be contigous");
  TORCH_CHECK(multinode_x.is_contiguous(), "multinode_x tensor must be contigous");
  TORCH_CHECK(buffer_flags.is_contiguous(), "buffer_flags tensor must be contigous");
  TORCH_CHECK(output_x.is_contiguous(), "output_x tensor must be contigous");
  TORCH_CHECK(residual_in.is_contiguous(), "residual_in tensor must be contigous");
  TORCH_CHECK(residual_out.is_contiguous(), "residual_out tensor must be contigous");
  TORCH_CHECK(weight_gamma.is_contiguous(), "weight_gamma tensor must be contigous");

  // Dtype checks
  TORCH_CHECK(input_x.scalar_type() == torch::kBFloat16,
              "input_x tensor data type must be bfloat16");
  TORCH_CHECK(multicast_x.scalar_type() == torch::kBFloat16,
              "multicast_x tensor data type must be bfloat16");
  TORCH_CHECK(output_x.scalar_type() == torch::kBFloat16,
              "output_x tensor data type must be bfloat16");
  TORCH_CHECK(residual_in.scalar_type() == torch::kBFloat16,
              "residual_in tensor data type must be bfloat16");
  TORCH_CHECK(residual_out.scalar_type() == torch::kBFloat16,
              "residual_out tensor data type must be bfloat16");
  TORCH_CHECK(weight_gamma.scalar_type() == torch::kBFloat16,
              "weight_gamma tensor data type must be bfloat16");
  TORCH_CHECK(data_buffer_ptrs.scalar_type() == torch::kInt64,
              "data_buffer_ptrs tensor data type must be int64");
  TORCH_CHECK(
      buffer_flags.scalar_type() == torch::kUInt32 || buffer_flags.scalar_type() == torch::kInt32,
      "buffer_flags tensor data type must be uint32/int32");

  // Shape & value checks
  TORCH_CHECK(input_x.dim() == 2, "input_x must be 2D [num_tokens, token_dim]");
  int64_t num_tokens = input_x.size(0);
  int64_t token_dim = input_x.size(1);
  using c_type = __nv_bfloat16;
  TORCH_CHECK(token_dim % (sizeof(float4) / sizeof(c_type)) == 0, "token_dim must be divisible by ",
              sizeof(float4) / sizeof(c_type));
  TORCH_CHECK(output_x.size(0) == num_tokens && output_x.size(1) == token_dim,
              "output_x shape mismatch: expected (", num_tokens, ", ", token_dim, ") but got (",
              output_x.size(0), ", ", output_x.size(1), ")");
  TORCH_CHECK(world_size >= 2 && world_size <= 64, "world_size must be between 2 and 64, got ",
              world_size);
  TORCH_CHECK(rank >= 0 && rank < world_size, "rank must be between 0 and world_size-1, got ",
              rank);

  if (rmsnorm_fusion) {
    TORCH_CHECK(residual_in.size(0) == num_tokens && residual_in.size(1) == token_dim,
                "residual_in shape mismatch: expected (", num_tokens, ", ", token_dim,
                ") but got (", residual_in.size(0), ", ", residual_in.size(1), ")");
    TORCH_CHECK(residual_out.size(0) == num_tokens && residual_out.size(1) == token_dim,
                "residual_out shape mismatch: expected (", num_tokens, ", ", token_dim,
                ") but got (", residual_out.size(0), ", ", residual_out.size(1), ")");
    TORCH_CHECK(weight_gamma.dim() == 1 && weight_gamma.size(0) == token_dim,
                "weight_gamma must have the same shape as token dimension (", token_dim,
                ") but got (", weight_gamma.size(0), ")");
  }

  // Acquire tensor pointers
  const auto *input_ptr = input_x.const_data_ptr();
  const auto *multicast_ptr = multicast_x.const_data_ptr();
  const auto *data_buffer_ptrs_ptr = data_buffer_ptrs.const_data_ptr();
  const auto *residual_in_ptr = residual_in.const_data_ptr();
  const auto *gamma_ptr = weight_gamma.const_data_ptr();

  auto *output_ptr = output_x.mutable_data_ptr();
  auto *residual_out_ptr = residual_out.mutable_data_ptr();
  auto *multinode_ptr = multinode_x.mutable_data_ptr();
  auto *buffer_flags_ptr = buffer_flags.mutable_data_ptr();

  // Create the parameters struct
  AllReduceFusionParams params;

  // Aux Information
  params.nRanks = static_cast<int>(world_size);
  params.rank = static_cast<int>(rank);
  params.numTokens = static_cast<int>(num_tokens);
  params.tokenDim = static_cast<int>(token_dim);
  params.bufferPtrsDev = reinterpret_cast<void **>(const_cast<void *>(data_buffer_ptrs_ptr));
  params.bufferPtrLocal = multinode_ptr;
  params.multicastPtr = const_cast<void *>(multicast_ptr);
  params.bufferFlags = reinterpret_cast<uint32_t *>(buffer_flags_ptr);
  params.rmsNormFusion = rmsnorm_fusion;
  params.launchWithPdl = launch_with_pdl;

  // input data
  params.input = input_ptr;
  params.residualIn = rmsnorm_fusion ? residual_in_ptr : nullptr;
  params.gamma = rmsnorm_fusion ? gamma_ptr : nullptr;
  params.epsilon = rms_norm_eps;

  // output data
  params.output = output_ptr;
  params.residualOut = rmsnorm_fusion ? residual_out_ptr : nullptr;
  params.stream = stream;

  cudaError_t status = fuse_allreduce_rmsnorm_low_latency_async<c_type>(params);
  TORCH_CHECK(status == cudaSuccess,
              "fuse_allreduce_rmsnorm_low_latency failed with error: ", cudaGetErrorString(status));
}

}  // namespace allreduce

}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "fuse_allreduce_rmsnorm_high_throughput(Tensor input, Tensor mc_input, Tensor in_residual, "
      "Tensor weight, Tensor signal, "
      "int rank, int world_size, int num_max_blocks, float rms_norm_eps, Tensor output, Tensor "
      "mc_output, Tensor out_residual) -> ()");
  m.impl("fuse_allreduce_rmsnorm_high_throughput", torch::kCUDA,
         &hpc::allreduce::fuse_allreduce_rmsnorm_high_throughput_entry);

  m.def(
      "fuse_allreduce_rmsnorm_low_latency(Tensor input_x, Tensor multicast_x, "
      "Tensor data_buffer_ptrs, "
      "Tensor! multinode_x, Tensor buffer_flags, "
      "int world_size, int rank, "
      "bool rmsnorm_fusion, bool launch_with_pdl, bool use_two_shot, "
      "Tensor! output_x, Tensor! residual_out, Tensor residual_in, "
      "Tensor weight_gamma, float rms_norm_eps) -> ()");
  m.impl("fuse_allreduce_rmsnorm_low_latency", torch::kCUDA,
         &hpc::allreduce::fuse_allreduce_rmsnorm_low_latency_entry);
}
