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
  int num_req = num_seqlen_per_req.size(0);
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
  }

  // Prepare pointers
  using DType = __nv_bfloat16;
  auto *out_q_ptr = reinterpret_cast<DType *>(out_q.mutable_data_ptr());
  auto *kcache_ptr = reinterpret_cast<DType *>(kcache.mutable_data_ptr());
  auto *vcache_ptr = reinterpret_cast<DType *>(vcache.mutable_data_ptr());
  const auto *qkv_ptr = reinterpret_cast<const DType *>(qkv.const_data_ptr());
  const auto *cos_sin_ptr = cos_sin.const_data_ptr<float>();
  const auto *num_tokens_per_batch_ptr = num_seqlen_per_req.const_data_ptr<int>();
  const auto *q_index_ptr = q_index.const_data_ptr<int>();
  const auto *kvcache_indices_ptr = kvcache_indices.const_data_ptr<int>();
  const float *q_norm_weight_ptr = nullptr;
  const float *k_norm_weight_ptr = nullptr;
  if (q_norm_weight_opt.has_value()) {
    TORCH_CHECK(q_norm_weight_opt.value().scalar_type() == torch::kFloat,
                "q_norm_weight tensor data type must be float");
    q_norm_weight_ptr = q_norm_weight_opt.value().const_data_ptr<float>();
  }

  if (k_norm_weight_opt.has_value()) {
    TORCH_CHECK(k_norm_weight_opt.value().scalar_type() == torch::kFloat,
                "k_norm_weight tensor data type must be float");
    k_norm_weight_ptr = k_norm_weight_opt.value().const_data_ptr<float>();
  }

  // Launch kernel
  apply_rotary_pos_emb_blocked_kvcache_bf16_async(
      out_q_ptr, kcache_ptr, vcache_ptr, qkv_ptr, cos_sin_ptr, num_tokens_per_batch_ptr,
      q_index_ptr, kvcache_indices_ptr, q_norm_weight_ptr, k_norm_weight_ptr, kcache_block_offset,
      vcache_block_offset, num_req, max_num_kv_block_per_batch, kv_block_size, num_rows,
      num_q_heads, num_kv_heads, qk_head_dim, v_head_dim, is_prefill, use_qk_norm, stream);

  return std::make_tuple(out_q, out_k);  // Return both outputs as a tuple
}

// @upper_max is used for scale to a suitable range, default should be fp8_max
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rope_norm_blocked_kvcache_w8c8_dqskv_entry(
    torch::Tensor &kcache, torch::Tensor &vcache, const torch::Tensor &qkv,
    const torch::Tensor &cos_sin, const torch::Tensor &num_seqlen_per_req,
    const torch::Tensor &q_index, const torch::Tensor &kvcache_indices, bool is_prefill,
    bool use_qk_norm, int64_t max_seqlens, const torch::Tensor &k_scale,
    const torch::Tensor &v_scale, std::optional<torch::Tensor> q_norm_weight_opt,
    std::optional<torch::Tensor> k_norm_weight_opt, std::optional<double> upper_max_double,
    std::optional<torch::Tensor> out_q_opt, std::optional<torch::Tensor> out_k_opt,
    std::optional<torch::Tensor> out_attention_opt) {
  auto stream = at::cuda::getCurrentCUDAStream(qkv.get_device());
  // kcache and vcache maybe not contiguous, we access them by stride
  TORCH_CHECK(qkv.is_contiguous(), "qkv tensor must be contiguous");
  TORCH_CHECK(cos_sin.is_contiguous(), "cos_sin tensor must be contiguous");
  TORCH_CHECK(num_seqlen_per_req.is_contiguous(), "num_seqlen_per_req tensor must be contiguous");
  TORCH_CHECK(kvcache_indices.is_contiguous(), "kvcache_indices tensor must be contiguous");
  TORCH_CHECK(k_scale.dim() == 1 && k_scale.size(0) == 1,
              "k_scale tensor must contain 1 element");  // per tensor
  TORCH_CHECK(v_scale.dim() == 1 && v_scale.size(0) == 1,
              "v_scale tensor must contain 1 element");  // per tensor
  TORCH_CHECK(
      use_qk_norm == true,
      "use_qk_norm must be true in rope quant op, because we need to reduce max in norm step");

  // dtype check
  TORCH_CHECK(qkv.scalar_type() == torch::kBFloat16, "qkv tensor data type must be bfloat16");
  TORCH_CHECK(cos_sin.scalar_type() == torch::kFloat, "cos_sin tensor data type must be float");
  TORCH_CHECK(kcache.dtype().itemsize() == 1,
              "kcache tensor element type size must be 1 byte (e.g., int8, fp8, etc.)");
  TORCH_CHECK(vcache.dtype().itemsize() == 1,
              "vcache tensor element type size must be 1 byte (e.g., int8, fp8, etc.)");
  TORCH_CHECK(k_scale.scalar_type() == torch::kFloat, "k_scale tensor data type must be float");
  TORCH_CHECK(v_scale.scalar_type() == torch::kFloat, "v_scale tensor data type must be float");

  using DType = __nv_bfloat16;
  using QType = __nv_fp8_e4m3;

  // Get dimensions from input tensors
  int num_req = num_seqlen_per_req.size(0);
  int num_rows = qkv.size(0);
  int num_kv_heads = kcache.size(2);
  int qk_head_dim = kcache.size(3);
  int v_head_dim = vcache.size(3);
  int hidden_size = qkv.size(1);
  int num_q_heads =
      (hidden_size - num_kv_heads * qk_head_dim - num_kv_heads * v_head_dim) / qk_head_dim;
  int kv_block_size = kcache.size(1);
  int max_num_kv_block_per_batch = kvcache_indices.size(1);
  int max_seqlens_pad128 = 0;
  int kcache_block_offset =
      kcache.stride(0);  // [num_blocks, block_size, num_kv_heads, qk_head_dim]
  int vcache_block_offset = vcache.stride(0);  // [num_blocks, block_size, num_kv_heads, v_head_dim]
  float upper_max = static_cast<float>(QType(1000.f));  // auto saturate 1000.f to fp8_max(≈448)
  if (upper_max_double.has_value()) {
    float in_upper_max = static_cast<float>(upper_max_double.value());
    TORCH_CHECK(!(in_upper_max > upper_max), "upper_max should not be larger than fp8_max");
    upper_max = in_upper_max;
  }

  // Create output tensors or use provided ones
  torch::Tensor out_q;
  torch::Tensor out_k;
  torch::Tensor q_scale;
  torch::Tensor out_attention;
  torch::Tensor split_k_flag;
  torch::Tensor tma_tensor;

  split_k_flag =
      torch::empty({num_req, num_kv_heads}, torch::dtype(torch::kInt32).device(qkv.device()));

  if (is_prefill) {
    max_seqlens_pad128 = ((max_seqlens + 127) / 128) * 128;
    q_scale = torch::empty({num_req, num_q_heads, max_seqlens_pad128},
                           torch::dtype(k_scale.dtype()).device(k_scale.device()));
  } else {
    q_scale = torch::empty({num_rows, num_q_heads},
                           torch::dtype(k_scale.dtype()).device(k_scale.device()));
  }

  if (out_q_opt.has_value()) {
    out_q = out_q_opt.value();
    TORCH_CHECK(
        out_q.size(0) == num_rows && out_q.size(1) == num_q_heads && out_q.size(2) == qk_head_dim,
        "out_q tensor shape mismatch");
    TORCH_CHECK(out_q.is_contiguous(), "out_q tensor must be contiguous");
    TORCH_CHECK(out_q.scalar_type() == torch::kFloat8_e4m3fn,
                "out_q tensor data type must be float8_e4m3fn");
  } else {
    out_q = torch::empty({num_rows, num_q_heads, qk_head_dim},
                         torch::dtype(torch::kFloat8_e4m3fn).device(qkv.device()));
  }

  if (out_k_opt.has_value()) {
    out_k = out_k_opt.value();
    TORCH_CHECK(
        out_k.size(0) == num_rows && out_k.size(1) == num_kv_heads && out_k.size(2) == qk_head_dim,
        "out_k tensor shape mismatch");
    TORCH_CHECK(out_k.is_contiguous(), "out_k tensor must be contiguous");
    TORCH_CHECK(out_k.scalar_type() == torch::kFloat8_e4m3fn,
                "out_k tensor data type must be float8_e4m3fn");
  }

  if (out_attention_opt.has_value()) {
    out_attention = out_attention_opt.value();
    TORCH_CHECK(
        out_attention.size(0) == num_rows && out_attention.size(1) == num_q_heads &&
            out_attention.size(2) == v_head_dim,
        "out_attention tensor shape should be [num_rows, num_q_heads, v_head_dim], but got shape=[",
        out_attention.size(0), ", ", out_attention.size(1), ", ", out_attention.size(2), "]");
    TORCH_CHECK(out_attention.is_contiguous(), "out_attention tensor must be contiguous");
    TORCH_CHECK(out_attention.scalar_type() == torch::kBFloat16,
                "out_attention tensor data type must be bfloat16");
  }

  // Prepare pointers
  auto *out_q_ptr = reinterpret_cast<QType *>(out_q.mutable_data_ptr());
  auto *kcache_ptr = reinterpret_cast<QType *>(kcache.mutable_data_ptr());
  auto *vcache_ptr = reinterpret_cast<QType *>(vcache.mutable_data_ptr());
  const auto *qkv_ptr = reinterpret_cast<const DType *>(qkv.const_data_ptr());
  const auto *cos_sin_ptr = cos_sin.const_data_ptr<float>();
  const auto *num_tokens_per_batch_ptr = num_seqlen_per_req.const_data_ptr<int>();
  const auto *q_index_ptr = q_index.const_data_ptr<int>();
  const auto *kvcache_indices_ptr = kvcache_indices.const_data_ptr<int>();
  const float *q_norm_weight_ptr = nullptr;
  const float *k_norm_weight_ptr = nullptr;
  auto *q_scale_ptr = q_scale.mutable_data_ptr<float>();
  const auto *k_scale_ptr = k_scale.const_data_ptr<float>();
  const auto *v_scale_ptr = v_scale.const_data_ptr<float>();
  auto *split_k_flag_ptr = split_k_flag.mutable_data_ptr<int>();

  if (q_norm_weight_opt.has_value()) {
    TORCH_CHECK(q_norm_weight_opt.value().scalar_type() == torch::kFloat,
                "q_norm_weight tensor data type must be float");
    q_norm_weight_ptr = q_norm_weight_opt.value().const_data_ptr<float>();
  }

  if (k_norm_weight_opt.has_value()) {
    TORCH_CHECK(k_norm_weight_opt.value().scalar_type() == torch::kFloat,
                "k_norm_weight tensor data type must be float");
    k_norm_weight_ptr = k_norm_weight_opt.value().const_data_ptr<float>();
  }

  // Launch kernel
  apply_rotary_pos_emb_blocked_kvcache_bf16_to_fp8_async(
      out_q_ptr, kcache_ptr, vcache_ptr, qkv_ptr, cos_sin_ptr, num_tokens_per_batch_ptr,
      q_index_ptr, kvcache_indices_ptr, q_norm_weight_ptr, k_norm_weight_ptr, q_scale_ptr,
      k_scale_ptr, v_scale_ptr, split_k_flag_ptr, upper_max, kcache_block_offset,
      vcache_block_offset, num_req, max_num_kv_block_per_batch, kv_block_size, num_rows,
      num_q_heads, num_kv_heads, qk_head_dim, v_head_dim, is_prefill, use_qk_norm,
      max_seqlens_pad128, stream);

  return std::make_tuple(out_q, out_k, q_scale, split_k_flag, out_attention,
                         tma_tensor);  // Return both outputs as a tuple
}

torch::Tensor rope_interleave_entry(torch::Tensor &input, const torch::Tensor &cos_sin_cache,
                                    const torch::Tensor &cu_seqlen_q,
                                    const torch::Tensor &seqlen_kv,
                                    std::optional<torch::Tensor> output) {
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

  torch::Tensor y;
  if (output.has_value()) {
    y = output.value();
  } else {
    y = torch::empty_like(input);
  }

  int num_tokens = input.size(0);
  int num_batch = seqlen_kv.size(0);
  int num_heads = input.size(1);
  int dim = input.size(2);
  int ldX = input.stride(0);
  int ldXHead = input.stride(1);
  int ldYHead = y.stride(1);
  int ldCache = cos_sin_cache.stride(0);
  int ldY = y.stride(0);

  TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "input tensor data type must be bfloat16");
  TORCH_CHECK(cos_sin_cache.scalar_type() == torch::kFloat,
              "cos_sin_cache tensor data type must be float");
  TORCH_CHECK(dim == 64, "rope_interleave dim only support dim 64.");

  auto *x_ptr = input.mutable_data_ptr();
  const auto *cos_sin_cache_ptr = cos_sin_cache.data_ptr();
  const auto *cu_seqlen_q_ptr = cu_seqlen_q.data_ptr<int>();
  const auto *seqlen_kv_ptr = seqlen_kv.data_ptr<int>();
  auto *y_ptr = y.mutable_data_ptr();

  bool running = rope_interleave_bf16_async(y_ptr, x_ptr, cos_sin_cache_ptr, cu_seqlen_q_ptr,
                                            seqlen_kv_ptr, num_batch, num_tokens, num_heads, dim,
                                            ldX, ldCache, ldY, ldXHead, ldYHead, stream);

  TORCH_CHECK(running, "rope_interleave_async running failed");

  return y;
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

  m.def(
      "rope_norm_blocked_kvcache_w8c8_dqskv(Tensor! kcache, Tensor! vcache, Tensor qkv, Tensor "
      "cos_sin, "
      "Tensor num_seqlen_per_req, Tensor q_index, Tensor kvcache_indices, bool is_prefill, bool "
      "use_qk_norm, int max_seqlens, Tensor k_scale, Tensor v_scale,"
      "Tensor? q_norm_weight, Tensor? k_norm_weight, float? upper_max, Tensor? out_q=None, Tensor? "
      "out_k=None, Tensor? out_attention=None) -> "
      "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.impl("rope_norm_blocked_kvcache_w8c8_dqskv", torch::kCUDA,
         &hpc::rope::rope_norm_blocked_kvcache_w8c8_dqskv_entry);

  m.def(
      "rope_interleave(Tensor! input, Tensor cos_sin_cache, Tensor cu_seqlen_q, Tensor seqlen_kv, "
      " Tensor? output) -> (Tensor)");
  m.impl("rope_interleave", torch::kCUDA, &hpc::rope::rope_interleave_entry);
}
