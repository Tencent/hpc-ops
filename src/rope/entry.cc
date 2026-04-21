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
    int64_t qk_norm_policy, std::optional<torch::Tensor> q_norm_weight_opt,
    std::optional<torch::Tensor> k_norm_weight_opt, std::optional<torch::Tensor> out_q_opt,
    std::optional<torch::Tensor> out_k_opt) {
  auto stream = at::cuda::getCurrentCUDAStream(qkv.get_device());
  // kcache and vcache maybe not contiguous, we access them by stride
  TORCH_CHECK(qkv.is_contiguous(), "qkv tensor must be contiguous");
  TORCH_CHECK(cos_sin.is_contiguous(), "cos_sin tensor must be contiguous");
  TORCH_CHECK(num_seqlen_per_req.is_contiguous(), "num_seqlen_per_req tensor must be contiguous");
  TORCH_CHECK(kvcache_indices.is_contiguous(), "kvcache_indices tensor must be contiguous");
  TORCH_CHECK(qk_norm_policy >= 0 && qk_norm_policy <= 2, "qk_norm_policy must be 0, 1 or 2");

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
      num_q_heads, num_kv_heads, qk_head_dim, v_head_dim, is_prefill, qk_norm_policy, stream);

  return std::make_tuple(out_q, out_k);  // Return both outputs as a tuple
}

// @upper_max is used for scale to a suitable range, default should be fp8_max
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
rope_norm_w8c8_entry(const torch::Tensor &q, const torch::Tensor &k, const torch::Tensor &v,
                     const torch::Tensor &cos_sin, const torch::Tensor &num_seqlen_per_req,
                     const torch::Tensor &q_index, bool is_prefill, int64_t max_seqlens,
                     const torch::Tensor &k_scale, const torch::Tensor &v_scale,
                     int64_t qk_norm_policy, std::optional<torch::Tensor> q_norm_weight_opt,
                     std::optional<torch::Tensor> k_norm_weight_opt,
                     std::optional<double> upper_max_double, std::optional<torch::Tensor> out_q_opt,
                     std::optional<torch::Tensor> out_k_opt, std::optional<torch::Tensor> out_v_opt,
                     std::optional<torch::Tensor> out_attention_opt) {
  auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
  // kcache and vcache maybe not contiguous, we access them by stride
  TORCH_CHECK(cos_sin.is_contiguous(), "cos_sin tensor must be contiguous");
  TORCH_CHECK(num_seqlen_per_req.is_contiguous(), "num_seqlen_per_req tensor must be contiguous");
  TORCH_CHECK(k_scale.dim() == 1 && k_scale.size(0) == 1,
              "k_scale tensor must contain 1 element");  // per tensor
  TORCH_CHECK(v_scale.dim() == 1 && v_scale.size(0) == 1,
              "v_scale tensor must contain 1 element");  // per tensor
  TORCH_CHECK(qk_norm_policy >= 0 && qk_norm_policy <= 2, "qk_norm_policy must be 0, 1 or 2");

  // dtype check
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "q tensor data type must be bfloat16");
  TORCH_CHECK(k.scalar_type() == torch::kBFloat16, "k tensor data type must be bfloat16");
  TORCH_CHECK(v.scalar_type() == torch::kBFloat16, "v tensor data type must be bfloat16");
  TORCH_CHECK(cos_sin.scalar_type() == torch::kFloat, "cos_sin tensor data type must be float");
  TORCH_CHECK(k_scale.scalar_type() == torch::kFloat, "k_scale tensor data type must be float");
  TORCH_CHECK(v_scale.scalar_type() == torch::kFloat, "v_scale tensor data type must be float");

  using DType = __nv_bfloat16;
  using QType = __nv_fp8_e4m3;

  // Get dimensions from input tensors
  int num_req = num_seqlen_per_req.size(0);
  int num_rows = q.size(0);  // [ num_rows, num_q_heads, qk_head_dim ]
  int num_q_heads = q.size(1);
  int q_stride = q.stride(0);
  int num_kv_heads = k.size(1);  // [ num_rows, num_kv_heads, qk_head_dim ]
  int qk_head_dim = k.size(2);
  int k_stride = k.stride(0);
  int v_head_dim = v.size(2);  // [ num_rows, num_kv_heads, v_head_dim ]
  int v_stride = v.stride(0);
  int max_seqlens_pad128 = 0;
  float upper_max = static_cast<float>(QType(1000.f));  // auto saturate 1000.f to fp8_max(448)
  // the fp8 kernel only support qkv combined
  TORCH_CHECK(q_stride == k_stride && k_stride == v_stride,
              "q, k, v must be sliced from one contiguous tensor");
  if (upper_max_double.has_value()) {
    float in_upper_max = static_cast<float>(upper_max_double.value());
    TORCH_CHECK(!(in_upper_max > upper_max), "upper_max should not be larger than fp8_max");
    upper_max = in_upper_max;
  }

  // Create output tensors or use provided ones
  torch::Tensor out_q;
  torch::Tensor out_k;
  torch::Tensor out_v;
  torch::Tensor q_scale;
  torch::Tensor out_attention;
  torch::Tensor split_k_flag;
  torch::Tensor tma_tensor;

  split_k_flag =
      torch::empty({num_req, num_kv_heads}, torch::dtype(torch::kInt32).device(q.device()));

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
                         torch::dtype(torch::kFloat8_e4m3fn).device(q.device()));
  }

  if (out_k_opt.has_value()) {
    out_k = out_k_opt.value();
    TORCH_CHECK(
        out_k.size(0) == num_rows && out_k.size(1) == num_kv_heads && out_k.size(2) == qk_head_dim,
        "out_k tensor shape mismatch");
    TORCH_CHECK(out_k.scalar_type() == torch::kFloat8_e4m3fn,
                "out_k tensor data type must be float8_e4m3fn");
  } else {
    out_k = torch::empty({num_rows, num_kv_heads, qk_head_dim},
                         torch::dtype(torch::kFloat8_e4m3fn).device(q.device()));
  }

  if (out_v_opt.has_value()) {
    out_v = out_v_opt.value();
    TORCH_CHECK(
        out_v.size(0) == num_rows && out_v.size(1) == num_kv_heads && out_v.size(2) == v_head_dim,
        "out_v tensor shape mismatch");
    TORCH_CHECK(out_v.scalar_type() == torch::kFloat8_e4m3fn,
                "out_v tensor data type must be float8_e4m3fn");
  } else {
    out_v = torch::empty({num_rows, num_kv_heads, v_head_dim},
                         torch::dtype(torch::kFloat8_e4m3fn).device(q.device()));
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

  int out_q_stride = out_q.stride(0);
  int out_k_stride = out_k.stride(0);
  int out_v_stride = out_v.stride(0);

  // Prepare pointers
  auto *out_q_ptr = reinterpret_cast<QType *>(out_q.mutable_data_ptr());
  auto *out_k_ptr = reinterpret_cast<QType *>(out_k.mutable_data_ptr());
  auto *out_v_ptr = reinterpret_cast<QType *>(out_v.mutable_data_ptr());
  const auto *in_q_ptr = reinterpret_cast<const DType *>(q.const_data_ptr());
  const auto *in_k_ptr = reinterpret_cast<const DType *>(k.const_data_ptr());
  const auto *in_v_ptr = reinterpret_cast<const DType *>(v.const_data_ptr());
  const auto *cos_sin_ptr = cos_sin.const_data_ptr<float>();
  const auto *num_tokens_per_batch_ptr = num_seqlen_per_req.const_data_ptr<int>();
  const auto *q_index_ptr = q_index.const_data_ptr<int>();
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
  } else {
    TORCH_CHECK(qk_norm_policy == 0, "q_norm_weight is required when qk_norm_policy is 1 or 2");
  }

  if (k_norm_weight_opt.has_value()) {
    TORCH_CHECK(k_norm_weight_opt.value().scalar_type() == torch::kFloat,
                "k_norm_weight tensor data type must be float");
    k_norm_weight_ptr = k_norm_weight_opt.value().const_data_ptr<float>();
  } else {
    TORCH_CHECK(qk_norm_policy == 0, "k_norm_weight is required when qk_norm_policy is 1 or 2");
  }

  // Launch kernel
  apply_rotary_pos_emb_bf16_to_fp8_async(
      out_q_ptr, out_k_ptr, out_v_ptr, in_q_ptr, in_k_ptr, in_v_ptr, q_stride, k_stride, v_stride,
      out_q_stride, out_k_stride, out_v_stride, cos_sin_ptr, num_tokens_per_batch_ptr, q_index_ptr,
      q_norm_weight_ptr, k_norm_weight_ptr, q_scale_ptr, k_scale_ptr, v_scale_ptr, split_k_flag_ptr,
      upper_max, num_req, num_rows, num_q_heads, num_kv_heads, qk_head_dim, v_head_dim, is_prefill,
      qk_norm_policy, max_seqlens_pad128, stream);

  return std::make_tuple(out_q, out_k, out_v, q_scale, split_k_flag, out_attention,
                         tma_tensor);  // Return both outputs as a tuple
}

// @upper_max is used for scale to a suitable range, default should be fp8_max
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rope_norm_blocked_kvcache_w8c8_dqskv_entry(
    torch::Tensor &kcache, torch::Tensor &vcache, const torch::Tensor &qkv,
    const torch::Tensor &cos_sin, const torch::Tensor &num_seqlen_per_req,
    const torch::Tensor &q_index, const torch::Tensor &kvcache_indices, bool is_prefill,
    int64_t qk_norm_policy, int64_t max_seqlens, const torch::Tensor &k_scale,
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
  TORCH_CHECK(qk_norm_policy >= 0 && qk_norm_policy <= 2, "qk_norm_policy must be 0, 1 or 2");

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
      num_q_heads, num_kv_heads, qk_head_dim, v_head_dim, is_prefill, qk_norm_policy,
      max_seqlens_pad128, stream);

  return std::make_tuple(out_q, out_k, q_scale, split_k_flag, out_attention,
                         tma_tensor);  // Return both outputs as a tuple
}

torch::Tensor rope_interleave_entry(torch::Tensor &input, const torch::Tensor &cos_sin_cache,
                                    const torch::Tensor &position,
                                    std::optional<torch::Tensor> output) {
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

  torch::Tensor y;
  if (output.has_value()) {
    y = output.value();
  } else {
    y = torch::empty_like(input);
  }

  int num_tokens = input.size(0);
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
  const auto *position_ptr = position.data_ptr<int64_t>();
  auto *y_ptr = y.mutable_data_ptr();

  bool running =
      rope_interleave_bf16_async(y_ptr, x_ptr, cos_sin_cache_ptr, position_ptr, num_tokens,
                                 num_heads, dim, ldX, ldCache, ldY, ldXHead, ldYHead, stream);

  TORCH_CHECK(running, "rope_interleave_async running failed");

  return y;
}

}  // namespace rope

namespace rope_v2 {

torch::Tensor rope_norm_store_kv_entry(
    torch::Tensor &kcache, torch::Tensor &vcache, const torch::Tensor &qkv,
    const torch::Tensor &cos_sin, const torch::Tensor &num_seqlen_per_req,
    const torch::Tensor &q_index, const torch::Tensor &kvcache_indices, bool is_prefill,
    std::optional<torch::Tensor> q_norm_weight_opt, std::optional<torch::Tensor> k_norm_weight_opt,
    std::optional<torch::Tensor> out_q_opt, std::optional<torch::Tensor> out_k_opt,
    std::optional<torch::Tensor> out_v_opt, int64_t qk_norm_policy) {
  auto stream = at::cuda::getCurrentCUDAStream(qkv.get_device());
  TORCH_CHECK(qkv.is_contiguous(), "qkv tensor must be contiguous");
  TORCH_CHECK(cos_sin.is_contiguous(), "cos_sin tensor must be contiguous");
  TORCH_CHECK(num_seqlen_per_req.is_contiguous(), "num_seqlen_per_req tensor must be contiguous");
  TORCH_CHECK(kvcache_indices.is_contiguous(), "kvcache_indices tensor must be contiguous");

  TORCH_CHECK(qk_norm_policy >= 0 && qk_norm_policy <= 2, "qk_norm_policy must be 0, 1 or 2");

  // Get dimensions
  int num_req = num_seqlen_per_req.size(0);
  int num_rows = qkv.size(0);
  int num_kv_heads = kcache.size(2);
  int qk_head_dim = kcache.size(3);
  int v_head_dim = vcache.size(3);
  int hidden_size = qkv.size(1);
  int num_q_heads =
      (hidden_size - num_kv_heads * qk_head_dim - num_kv_heads * v_head_dim) / qk_head_dim;
  int kv_block_size = kcache.size(1);

  TORCH_CHECK(kvcache_indices.stride(1) == 1,
              "kvcache_indices dim-1 must be contiguous (stride(1)==1)");

  Strides4D kc_strides{kcache.stride(0), kcache.stride(1), kcache.stride(2)};
  Strides4D vc_strides{vcache.stride(0), vcache.stride(1), vcache.stride(2)};
  Strides2D ki_strides{kvcache_indices.stride(0)};

  // Create output tensors
  using DType = __nv_bfloat16;
  torch::Tensor out_q;
  if (out_q_opt.has_value()) {
    out_q = out_q_opt.value();
    TORCH_CHECK(out_q.is_contiguous(), "out_q tensor must be contiguous");
  } else {
    out_q = torch::empty({num_rows, num_q_heads, qk_head_dim},
                         torch::dtype(qkv.dtype()).device(qkv.device()));
  }

  DType *out_k_ptr = nullptr;
  if (out_k_opt.has_value()) {
    TORCH_CHECK(out_k_opt.value().is_contiguous(), "out_k tensor must be contiguous");
    out_k_ptr = reinterpret_cast<DType *>(out_k_opt.value().mutable_data_ptr());
  }

  DType *out_v_ptr = nullptr;
  if (out_v_opt.has_value()) {
    auto out_v = out_v_opt.value();
    TORCH_CHECK(out_v.is_contiguous(), "out_v tensor must be contiguous");
    out_v_ptr = reinterpret_cast<DType *>(out_v.mutable_data_ptr());
  }

  const float *q_norm_weight_ptr = nullptr;
  const float *k_norm_weight_ptr = nullptr;
  if (q_norm_weight_opt.has_value()) {
    TORCH_CHECK(q_norm_weight_opt.value().scalar_type() == torch::kFloat);
    q_norm_weight_ptr = q_norm_weight_opt.value().const_data_ptr<float>();
  }
  if (k_norm_weight_opt.has_value()) {
    TORCH_CHECK(k_norm_weight_opt.value().scalar_type() == torch::kFloat);
    k_norm_weight_ptr = k_norm_weight_opt.value().const_data_ptr<float>();
  }

  rope_norm_store_kv_async(
      reinterpret_cast<DType *>(out_q.mutable_data_ptr()),
      reinterpret_cast<DType *>(kcache.mutable_data_ptr()),
      reinterpret_cast<DType *>(vcache.mutable_data_ptr()), out_k_ptr, out_v_ptr,
      reinterpret_cast<const DType *>(qkv.const_data_ptr()), cos_sin.const_data_ptr<float>(),
      num_seqlen_per_req.const_data_ptr<int>(), q_index.const_data_ptr<int>(),
      kvcache_indices.const_data_ptr<int>(), q_norm_weight_ptr, k_norm_weight_ptr, kc_strides,
      vc_strides, ki_strides, num_req, kv_block_size, num_rows, num_q_heads, num_kv_heads,
      qk_head_dim, v_head_dim, is_prefill, qk_norm_policy, stream);

  return out_q;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> rope_norm_store_kv_fp8_entry(
    torch::Tensor &kcache, torch::Tensor &vcache, const torch::Tensor &qkv,
    const torch::Tensor &cos_sin, const torch::Tensor &num_seqlen_per_req,
    const torch::Tensor &q_index, const torch::Tensor &kvcache_indices, bool is_prefill,
    torch::Tensor &k_scale, const torch::Tensor &v_scale, int64_t quant_policy, int64_t max_seqlens,
    std::optional<double> upper_max_double, std::optional<torch::Tensor> q_scale_inv_opt,
    std::optional<torch::Tensor> q_norm_weight_opt, std::optional<torch::Tensor> k_norm_weight_opt,
    std::optional<torch::Tensor> out_q_opt, std::optional<torch::Tensor> out_k_opt,
    std::optional<torch::Tensor> out_v_opt, int64_t qk_norm_policy) {
  auto stream = at::cuda::getCurrentCUDAStream(qkv.get_device());
  TORCH_CHECK(qkv.is_contiguous(), "qkv tensor must be contiguous");
  TORCH_CHECK(cos_sin.is_contiguous(), "cos_sin tensor must be contiguous");
  TORCH_CHECK(num_seqlen_per_req.is_contiguous(), "num_seqlen_per_req tensor must be contiguous");
  TORCH_CHECK(kvcache_indices.stride(1) == 1, "kvcache_indices must stride(1) == 1");
  TORCH_CHECK(qkv.scalar_type() == torch::kBFloat16, "qkv must be bfloat16");
  TORCH_CHECK(kcache.dtype().itemsize() == 1, "kcache must be 1-byte dtype");
  TORCH_CHECK(vcache.dtype().itemsize() == 1, "vcache must be 1-byte dtype");
  TORCH_CHECK(quant_policy >= 0 && quant_policy <= 2, "quant_policy must be 0, 1 or 2");
  TORCH_CHECK(qk_norm_policy >= 0 && qk_norm_policy <= 2, "qk_norm_policy must be 0, 1 or 2");

  using DType = __nv_bfloat16;
  using QType = __nv_fp8_e4m3;

  int num_req = num_seqlen_per_req.size(0);
  int num_rows = qkv.size(0);
  int num_kv_heads = kcache.size(2);
  int qk_head_dim = kcache.size(3);
  int v_head_dim = vcache.size(3);
  int hidden_size = qkv.size(1);
  int num_q_heads =
      (hidden_size - num_kv_heads * qk_head_dim - num_kv_heads * v_head_dim) / qk_head_dim;
  int kv_block_size = kcache.size(1);

  Strides4D kc_strides{kcache.stride(0), kcache.stride(1), kcache.stride(2)};
  Strides4D vc_strides{vcache.stride(0), vcache.stride(1), vcache.stride(2)};
  Strides2D ki_strides{kvcache_indices.stride(0)};

  // Validate scale tensor shapes based on quant_policy
  Strides4D ks_strides{0, 0, 0};
  if (quant_policy == 0) {
    // Dynamic per-head per-token: k_scale is [num_block, R, num_kv_heads, L] (output)
    int L = qk_head_dim * static_cast<int>(sizeof(QType)) / static_cast<int>(sizeof(float));
    int R = kv_block_size / L;
    TORCH_CHECK(k_scale.dim() == 4, "k_scale must be 4D for quant_policy=0");
    TORCH_CHECK(k_scale.size(1) == R, "k_scale dim 1 must equal block_size/L=", R, ", got ",
                k_scale.size(1));
    TORCH_CHECK(k_scale.size(2) == num_kv_heads,
                "k_scale dim 2 must equal num_kv_heads=", num_kv_heads, ", got ", k_scale.size(2));
    TORCH_CHECK(k_scale.size(3) == L, "k_scale dim 3 must equal L=", L, ", got ", k_scale.size(3));
    // V scale: per-head [num_kv_heads]
    TORCH_CHECK(v_scale.dim() == 1 && v_scale.size(0) == num_kv_heads,
                "v_scale must be [num_kv_heads] for quant_policy=0");
    ks_strides = Strides4D{k_scale.stride(0), k_scale.stride(1), k_scale.stride(2)};
  } else {
    // Static per-tensor: k_scale and v_scale are [1]
    TORCH_CHECK(k_scale.dim() == 1 && k_scale.size(0) == 1, "k_scale must contain 1 element");
    TORCH_CHECK(v_scale.dim() == 1 && v_scale.size(0) == 1, "v_scale must contain 1 element");
  }

  float upper_max = static_cast<float>(QType(1000.f));
  if (upper_max_double.has_value()) {
    float in_upper_max = static_cast<float>(upper_max_double.value());
    TORCH_CHECK(!(in_upper_max > upper_max), "upper_max should not be larger than fp8_max");
    upper_max = in_upper_max;
  }

  // out_q
  torch::Tensor out_q;
  if (out_q_opt.has_value()) {
    out_q = out_q_opt.value();
    TORCH_CHECK(out_q.is_contiguous() && out_q.scalar_type() == torch::kFloat8_e4m3fn);
  } else {
    out_q = torch::empty({num_rows, num_q_heads, qk_head_dim},
                         torch::dtype(torch::kFloat8_e4m3fn).device(qkv.device()));
  }

  // q_scale: dqskv allocates real storage, sqskv gets an empty tensor
  torch::Tensor q_scale;
  float *q_scale_ptr = nullptr;
  int max_seqlens_pad128 = 0;
  if (quant_policy == 0 || quant_policy == 1) {
    if (is_prefill) {
      max_seqlens_pad128 = ((max_seqlens + 127) / 128) * 128;
      q_scale = torch::empty({num_req, num_q_heads, max_seqlens_pad128},
                             torch::dtype(torch::kFloat).device(qkv.device()));
    } else {
      q_scale =
          torch::empty({num_rows, num_q_heads}, torch::dtype(torch::kFloat).device(qkv.device()));
    }
    q_scale_ptr = q_scale.mutable_data_ptr<float>();
  }

  // split_k_flag
  torch::Tensor split_k_flag =
      torch::empty({num_req, num_kv_heads}, torch::dtype(torch::kInt32).device(qkv.device()));

  // out_k, out_v (nullable bypass)
  QType *out_k_ptr = nullptr;
  QType *out_v_ptr = nullptr;
  if (out_k_opt.has_value()) {
    auto out_k = out_k_opt.value();
    TORCH_CHECK(out_k.is_contiguous() && out_k.scalar_type() == torch::kFloat8_e4m3fn);
    out_k_ptr = reinterpret_cast<QType *>(out_k.mutable_data_ptr());
  }
  if (out_v_opt.has_value()) {
    auto out_v = out_v_opt.value();
    TORCH_CHECK(out_v.is_contiguous() && out_v.scalar_type() == torch::kFloat8_e4m3fn);
    out_v_ptr = reinterpret_cast<QType *>(out_v.mutable_data_ptr());
  }

  const float *q_norm_weight_ptr = nullptr;
  const float *k_norm_weight_ptr = nullptr;
  if (q_norm_weight_opt.has_value()) {
    TORCH_CHECK(q_norm_weight_opt.value().scalar_type() == torch::kFloat);
    q_norm_weight_ptr = q_norm_weight_opt.value().const_data_ptr<float>();
  }
  if (k_norm_weight_opt.has_value()) {
    TORCH_CHECK(k_norm_weight_opt.value().scalar_type() == torch::kFloat);
    k_norm_weight_ptr = k_norm_weight_opt.value().const_data_ptr<float>();
  }

  const float *q_scale_inv_ptr = nullptr;
  if (quant_policy == 2) {
    TORCH_CHECK(q_scale_inv_opt.has_value(), "q_scale_inv required for quant_policy=2");
    TORCH_CHECK(q_scale_inv_opt.value().scalar_type() == torch::kFloat);
    q_scale_inv_ptr = q_scale_inv_opt.value().const_data_ptr<float>();
  }

  rope_norm_store_kv_fp8_async(
      reinterpret_cast<QType *>(out_q.mutable_data_ptr()),
      reinterpret_cast<QType *>(kcache.mutable_data_ptr()),
      reinterpret_cast<QType *>(vcache.mutable_data_ptr()), out_k_ptr, out_v_ptr,
      split_k_flag.mutable_data_ptr<int32_t>(), q_scale_ptr,
      reinterpret_cast<const DType *>(qkv.const_data_ptr()), cos_sin.const_data_ptr<float>(),
      num_seqlen_per_req.const_data_ptr<int>(), q_index.const_data_ptr<int>(),
      kvcache_indices.const_data_ptr<int>(), q_norm_weight_ptr, k_norm_weight_ptr,
      k_scale.mutable_data_ptr<float>(), v_scale.const_data_ptr<float>(), q_scale_inv_ptr,
      upper_max, max_seqlens, kc_strides, ks_strides, vc_strides, ki_strides, num_req,
      kv_block_size, num_rows, num_q_heads, num_kv_heads, qk_head_dim, v_head_dim, is_prefill,
      qk_norm_policy, quant_policy, stream);

  return std::make_tuple(out_q, q_scale, split_k_flag);
}
}  // namespace rope_v2

}  // namespace hpc

// Register the function with optional output tensors
TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "rope_norm_blocked_kvcache(Tensor! kcache, Tensor! vcache, Tensor qkv, Tensor cos_sin, "
      "Tensor num_seqlen_per_req, Tensor q_index, Tensor kvcache_indices, bool is_prefill, int "
      "qk_norm_policy, "
      "Tensor? q_norm_weight, Tensor? k_norm_weight, Tensor? out_q=None, Tensor? out_k=None) -> "
      "(Tensor, Tensor)");
  m.impl("rope_norm_blocked_kvcache", torch::kCUDA, &hpc::rope::rope_norm_blocked_kvcache_entry);

  m.def(
      "rope_norm_blocked_kvcache_w8c8_dqskv(Tensor! kcache, Tensor! vcache, Tensor qkv, Tensor "
      "cos_sin, "
      "Tensor num_seqlen_per_req, Tensor q_index, Tensor kvcache_indices, bool is_prefill, int "
      "qk_norm_policy, int max_seqlens, Tensor k_scale, Tensor v_scale,"
      "Tensor? q_norm_weight, Tensor? k_norm_weight, float? upper_max, Tensor? out_q=None, Tensor? "
      "out_k=None, Tensor? out_attention=None) -> "
      "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.impl("rope_norm_blocked_kvcache_w8c8_dqskv", torch::kCUDA,
         &hpc::rope::rope_norm_blocked_kvcache_w8c8_dqskv_entry);

  m.def(
      "rope_interleave(Tensor! input, Tensor cos_sin_cache, Tensor position, "
      " Tensor? output) -> (Tensor)");
  m.impl("rope_interleave", torch::kCUDA, &hpc::rope::rope_interleave_entry);

  m.def(
      "rope_norm_w8c8(Tensor q, Tensor k, Tensor v, Tensor cos_sin, Tensor num_seqlen_per_req, "
      "Tensor q_index, bool is_prefill, int max_seqlens, Tensor "
      "k_scale, Tensor v_scale, int qk_norm_policy, Tensor? q_norm_weight, Tensor? k_norm_weight, "
      "float? upper_max, "
      "Tensor? out_q, Tensor? out_k, Tensor? out_v, Tensor? out_attention) -> (Tensor, Tensor, "
      "Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.impl("rope_norm_w8c8", torch::kCUDA, &hpc::rope::rope_norm_w8c8_entry);

  m.def(
      "rope_norm_store_kv(Tensor! kcache, Tensor! vcache, Tensor qkv, Tensor cos_sin, "
      "Tensor num_seqlen_per_req, Tensor q_index, Tensor kvcache_indices, bool is_prefill, "
      "Tensor? q_norm_weight, Tensor? k_norm_weight, "
      "Tensor? out_q=None, Tensor? out_k=None, Tensor? out_v=None, int qk_norm_policy=0) -> "
      "Tensor");
  m.impl("rope_norm_store_kv", torch::kCUDA, &hpc::rope_v2::rope_norm_store_kv_entry);

  m.def(
      "rope_norm_store_kv_fp8(Tensor! kcache, Tensor! vcache, Tensor qkv, "
      "Tensor cos_sin, Tensor num_seqlen_per_req, Tensor q_index, Tensor kvcache_indices, "
      "bool is_prefill, Tensor! k_scale, Tensor v_scale, "
      "int quant_policy, int max_seqlens, float? upper_max, Tensor? q_scale_inv, "
      "Tensor? q_norm_weight, Tensor? k_norm_weight, "
      "Tensor? out_q=None, Tensor? out_k=None, Tensor? out_v=None, int qk_norm_policy=0) -> "
      "(Tensor, Tensor, Tensor)");
  m.impl("rope_norm_store_kv_fp8", torch::kCUDA, &hpc::rope_v2::rope_norm_store_kv_fp8_entry);
}
