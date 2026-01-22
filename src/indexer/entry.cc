// Copyright 2025 hpc-ops authors

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include "src/indexer/indexer.h"

namespace hpc {
namespace indexer {

torch::Tensor mqa_indexer_logits_entry(const torch::Tensor &q, const torch::Tensor &kvcache,
                                       const torch::Tensor &weight, const torch::Tensor &block_ids,
                                       const torch::Tensor &cu_seqlens_q,
                                       const torch::Tensor &seqlens_kv, const int64_t &ratio,
                                       const int64_t &max_context_len,
                                       std::optional<torch::Tensor> output) {
  auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
  TORCH_CHECK(q.device().is_cuda(), "q tensor must be cuda");
  TORCH_CHECK(kvcache.device().is_cuda(), "kvcache tensor must be cuda");
  TORCH_CHECK(cu_seqlens_q.device().is_cuda(), "cu_seqlens_q tensor must be cuda");
  TORCH_CHECK(block_ids.device().is_cuda(), "block_ids tensor must be cuda");
  TORCH_CHECK(seqlens_kv.device().is_cuda(), "seqlens_kv tensor must be cuda");

  int total_seq_q = q.size(0);
  int num_head_q = q.size(1);
  int head_dim = q.size(2);

  int num_batch = cu_seqlens_q.size(0) - 1;

  int num_kvcache_blocks = kvcache.size(0);
  int block_size = kvcache.size(1);

  int num_seq_max_blocks = block_ids.stride(0);

  auto options = q.options().dtype(torch::kFloat32);
  torch::Tensor y;
  if (output.has_value()) {
    y = output.value();
  } else {
    y = torch::zeros({total_seq_q, max_context_len}, options);
  }

  int num_split = 1;

  if (total_seq_q <= 16) {
    num_split = 8;
  } else if (total_seq_q <= 32) {
    num_split = 4;
  } else if (total_seq_q <= 78) {
    num_split = 2;
  }

  const auto *q_ptr = q.const_data_ptr();
  const auto *kvcache_ptr = kvcache.const_data_ptr();
  const auto *weight_ptr = weight.const_data_ptr();

  const auto *cu_seqlens_q_ptr = cu_seqlens_q.const_data_ptr();
  const auto *block_ids_ptr = block_ids.const_data_ptr();
  const auto *seqlens_kv_ptr = seqlens_kv.const_data_ptr();

  auto *y_ptr = y.mutable_data_ptr();

  if (q.scalar_type() == torch::kBFloat16) {
    mqa_indexer_logits_bf16_async(y_ptr, q_ptr, kvcache_ptr, weight_ptr, cu_seqlens_q_ptr,
                                  seqlens_kv_ptr, block_ids_ptr, num_batch, total_seq_q, num_head_q,
                                  head_dim, num_kvcache_blocks, block_size, num_seq_max_blocks,
                                  max_context_len, ratio, num_split, stream);
  } else if (q.scalar_type() == torch::kFloat8_e4m3fn) {
    mqa_indexer_logits_fp8_async(y_ptr, q_ptr, kvcache_ptr, weight_ptr, cu_seqlens_q_ptr,
                                 seqlens_kv_ptr, block_ids_ptr, num_batch, total_seq_q, num_head_q,
                                 head_dim, num_kvcache_blocks, block_size, num_seq_max_blocks,
                                 max_context_len, ratio, num_split, stream);
  }

  return y;
}

}  // namespace indexer
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "mqa_indexer_logits(Tensor q, Tensor kvcache,"
      "Tensor weight,  "
      "Tensor block_ids, Tensor cu_seqlens_q, Tensor seqlens_kv, int ratio, int max_context_len, "
      "Tensor? output) "
      "-> "
      "(Tensor)");
  m.impl("mqa_indexer_logits", torch::kCUDA, &hpc::indexer::mqa_indexer_logits_entry);
}
