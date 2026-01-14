// Copyright 2025 hpc-ops authors

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include <optional>
#include <tuple>

#include "src/kv_compressor/kv_compressor.h"

namespace hpc {
namespace kv_compressor {

torch::Tensor kv_compressor_entry(const torch::Tensor &kv, const torch::Tensor &score,
                                  const torch::Tensor &cu_seqlens,
                                  const torch::Tensor &cu_compressed_seqlens,
                                  int64_t total_compressed_seqlen, torch::Tensor &kv_states,
                                  torch::Tensor &score_states, const torch::Tensor &state_index,
                                  const torch::Tensor &start_pos, const torch::Tensor &ape,
                                  int64_t ratio, bool overlap, int64_t head_dim, bool is_prefill) {
  auto stream = at::cuda::getCurrentCUDAStream(kv.get_device());
  // kcache and vcache maybe not contiguous, we access them by stride
  TORCH_CHECK(kv.is_contiguous(), "kv tensor must be contiguous");
  TORCH_CHECK(score.is_contiguous(), "score tensor must be contiguous");
  TORCH_CHECK(cu_seqlens.is_contiguous(), "cu_seqlens tensor must be contiguous");
  TORCH_CHECK(cu_compressed_seqlens.is_contiguous(),
              "cu_compressed_seqlens tensor must be contiguous");
  TORCH_CHECK(ape.is_contiguous(), "ape tensor must be contiguous");

  // type check
  using DType = float;
  TORCH_CHECK(kv.scalar_type() == torch::kFloat32, "kv tensor data type must be float32");
  TORCH_CHECK(score.scalar_type() == torch::kFloat32, "score tensor data type must be float32");
  TORCH_CHECK(cu_seqlens.scalar_type() == torch::kInt32,
              "cu_seqlens tensor data type must be int32");
  TORCH_CHECK(cu_compressed_seqlens.scalar_type() == torch::kInt32,
              "cu_compressed_seqlens tensor data type must be int32");
  TORCH_CHECK(kv_states.scalar_type() == torch::kFloat32,
              "kv_states tensor data type must be float32");
  TORCH_CHECK(score_states.scalar_type() == torch::kFloat32,
              "score_states tensor data type must be float32");
  TORCH_CHECK(ape.scalar_type() == torch::kFloat32, "ape tensor data type must be float32");
  TORCH_CHECK(state_index.scalar_type() == torch::kInt32,
              "state_index tensor data type must be int32");
  TORCH_CHECK(start_pos.scalar_type() == torch::kInt32, "start_pos tensor data type must be int32");

  // Get dimensions from input tensors
  int num_batch = cu_seqlens.size(0) - 1;
  int total_seqlen = kv.size(0);
  int hidden_size = kv.size(1);

  if (overlap) {
    TORCH_CHECK(hidden_size == head_dim * 2, "if overlap, kv.size(1) must be head_dim*2");
    TORCH_CHECK(ratio == 4, "if overlap, ratio must be 4");
  }

  // Create output tensors or use provided ones
  torch::Tensor compressed_kv = torch::empty({total_compressed_seqlen, head_dim},
                                             torch::dtype(kv.dtype()).device(kv.device()));

  // Prepare pointers
  auto *compressed_kv_ptr = compressed_kv.mutable_data_ptr<DType>();
  const auto *kv_ptr = kv.const_data_ptr<DType>();
  const auto *score_ptr = score.const_data_ptr<DType>();
  const auto *ape_ptr = ape.const_data_ptr<DType>();
  const auto *cu_seqlens_ptr = cu_seqlens.const_data_ptr<int>();
  const auto *cu_compressed_seqlens_ptr = cu_compressed_seqlens.const_data_ptr<int>();
  const auto *state_index_ptr = state_index.const_data_ptr<int>();
  const auto *start_pos_ptr = start_pos.const_data_ptr<int>();
  auto *kv_states_ptr = kv_states.mutable_data_ptr<DType>();
  auto *score_states_ptr = score_states.mutable_data_ptr<DType>();

  // Launch kernel
  bool running = kv_compressor_fp32_async(
      compressed_kv_ptr, kv_ptr, score_ptr, cu_seqlens_ptr, cu_compressed_seqlens_ptr,
      kv_states_ptr, score_states_ptr, state_index_ptr, start_pos_ptr, ape_ptr, num_batch,
      total_seqlen, ratio, overlap, head_dim, is_prefill, stream);
  TORCH_CHECK(running, "kv compressor fp32 launch failed!");
  return compressed_kv;
}

}  // namespace kv_compressor
}  // namespace hpc

// Register the function with optional output tensors
TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "kv_compressor(Tensor kv, Tensor score, Tensor cu_seqlens, Tensor cu_compressed_seqlens, int "
      "total_compressed_seqlen, "
      "Tensor! kv_states, Tensor! score_states, Tensor state_index, Tensor start_pos, Tensor ape, "
      "int ratio, bool overlap, int head_dim, bool is_prefill) -> (Tensor)");
  m.impl("kv_compressor", torch::kCUDA, &hpc::kv_compressor::kv_compressor_entry);
}
