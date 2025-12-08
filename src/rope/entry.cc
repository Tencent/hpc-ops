// Copyright 2025 hpc-ops authors

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include <optional>
#include <tuple>

#include "src/rope/rope.h"

namespace hpc {
namespace rope {

std::tuple<torch::Tensor, torch::Tensor> rope_norm_blocked_kvcache_entry(
    torch::Tensor &kcache, torch::Tensor &vcache, const torch::Tensor &qkv,
    const torch::Tensor &cos_sin, const torch::Tensor &num_seqlen_per_req,
    const torch::Tensor &q_index, const torch::Tensor &kvcache_indices, bool is_prefill,
    bool use_qk_norm, std::optional<torch::Tensor> q_norm_weight_opt,
    std::optional<torch::Tensor> k_norm_weight_opt, std::optional<torch::Tensor> out_q_opt,
    std::optional<torch::Tensor> out_k_opt) {
  auto stream = at::cuda::getCurrentCUDAStream(qkv.get_device());
  // kcache and vcache maybe not contiguous, we access them by stride
  TORCH_CHECK(qkv.is_contiguous(), "qkv tensor must be contiguous");
  TORCH_CHECK(cos_sin.is_contiguous(), "cos_sin tensor must be contiguous");
  TORCH_CHECK(num_seqlen_per_req.is_contiguous(), "num_seqlen_per_req tensor must be contiguous");
  TORCH_CHECK(kvcache_indices.is_contiguous(), "kvcache_indices tensor must be contiguous");

  // Get dimensions from input tensors
  int num_batch = num_seqlen_per_req.size(0);
  int num_rows = qkv.size(0);
  int num_kv_heads = kcache.size(2);
  int qk_head_dim = kcache.size(3);
  int v_head_dim = vcache.size(3);
  int hidden_size = qkv.size(1);
  int num_q_heads =
      (hidden_size - num_kv_heads * qk_head_dim - num_kv_heads * v_head_dim) / qk_head_dim;
  int kv_block_size = kcache.size(1);
  int max_num_kv_block_per_batch = kvcache_indices.size(1);
  int kcache_block_offset =
      kcache.stride(0);  // [num_blocks, block_size, num_kv_heads, qk_head_dim]
  int vcache_block_offset = vcache.stride(0);  // [num_blocks, block_size, num_kv_heads, v_head_dim]

  // Create output tensors or use provided ones
  torch::Tensor out_q;
  torch::Tensor out_k;

  if (out_q_opt.has_value()) {
    out_q = out_q_opt.value();
    TORCH_CHECK(
        out_q.size(0) == num_rows && out_q.size(1) == num_q_heads && out_q.size(2) == qk_head_dim,
        "out_q tensor shape mismatch");
    TORCH_CHECK(out_q.is_contiguous(), "out_q tensor must be contiguous");
  } else {
    out_q = torch::empty({num_rows, num_q_heads, qk_head_dim},
                         torch::dtype(qkv.dtype()).device(qkv.device()));
  }

  if (out_k_opt.has_value()) {
    out_k = out_k_opt.value();
    TORCH_CHECK(
        out_k.size(0) == num_rows && out_k.size(1) == num_kv_heads && out_k.size(2) == qk_head_dim,
        "out_k tensor shape mismatch");
    TORCH_CHECK(out_k.is_contiguous(), "out_k tensor must be contiguous");
  } else {
    out_k = torch::empty({num_rows, num_kv_heads, qk_head_dim},
                         torch::dtype(qkv.dtype()).device(qkv.device()));
  }

  // Prepare pointers
  using T = __nv_bfloat16;
  auto *out_q_ptr = reinterpret_cast<T *>(out_q.mutable_data_ptr());
  auto *out_k_ptr = reinterpret_cast<T *>(out_k.mutable_data_ptr());
  auto *kcache_ptr = reinterpret_cast<T *>(kcache.mutable_data_ptr());
  auto *vcache_ptr = reinterpret_cast<T *>(vcache.mutable_data_ptr());
  const auto *qkv_ptr = reinterpret_cast<const T *>(qkv.const_data_ptr());
  const auto *cos_sin_ptr = cos_sin.const_data_ptr<float>();
  const auto *num_tokens_per_batch_ptr = num_seqlen_per_req.const_data_ptr<int>();
  const auto *q_index_ptr = q_index.const_data_ptr<int>();
  const auto *kvcache_indices_ptr = kvcache_indices.const_data_ptr<int>();
  const float *q_norm_weight_ptr = nullptr;
  const float *k_norm_weight_ptr = nullptr;
  if (q_norm_weight_opt.has_value()) {
    q_norm_weight_ptr = q_norm_weight_opt.value().const_data_ptr<float>();
  }

  if (k_norm_weight_opt.has_value()) {
    k_norm_weight_ptr = k_norm_weight_opt.value().const_data_ptr<float>();
  }

  // Launch kernel
  apply_rotary_pos_emb_blocked_kvcache_bf16_async(
      out_q_ptr, out_k_ptr, kcache_ptr, vcache_ptr, qkv_ptr, cos_sin_ptr, num_tokens_per_batch_ptr,
      q_index_ptr, kvcache_indices_ptr, q_norm_weight_ptr, k_norm_weight_ptr, kcache_block_offset,
      vcache_block_offset, num_batch, max_num_kv_block_per_batch, kv_block_size, num_rows,
      num_q_heads, num_kv_heads, qk_head_dim, v_head_dim, is_prefill, use_qk_norm, stream);

  return std::make_tuple(out_q, out_k);  // Return both outputs as a tuple
}

}  // namespace rope
}  // namespace hpc

// Register the function with optional output tensors
TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "rope_norm_blocked_kvcache(Tensor! kcache, Tensor! vcache, Tensor qkv, Tensor cos_sin, "
      "Tensor num_seqlen_per_req, Tensor q_index, Tensor kvcache_indices, bool is_prefill, bool "
      "use_qk_norm, "
      "Tensor? q_norm_weight, Tensor? k_norm_weight, Tensor? out_q=None, Tensor? out_k=None) -> "
      "(Tensor, Tensor)");
  m.impl("rope_norm_blocked_kvcache", torch::kCUDA, &hpc::rope::rope_norm_blocked_kvcache_entry);
}
