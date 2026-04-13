// Copyright (C) 2026 Tencent.

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include <optional>
#include <tuple>

#include "src/rope/rope.h"

namespace hpc {
namespace rope {

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
  int max_num_kv_block_per_batch = kvcache_indices.size(1);
  int kcache_block_offset = kcache.stride(0);
  int vcache_block_offset = vcache.stride(0);

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
      kvcache_indices.const_data_ptr<int>(), q_norm_weight_ptr, k_norm_weight_ptr,
      kcache_block_offset, vcache_block_offset, num_req, max_num_kv_block_per_batch, kv_block_size,
      num_rows, num_q_heads, num_kv_heads, qk_head_dim, v_head_dim, is_prefill, qk_norm_policy,
      stream);

  return out_q;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> rope_norm_store_kv_fp8_entry(
    torch::Tensor &kcache, torch::Tensor &vcache, const torch::Tensor &qkv,
    const torch::Tensor &cos_sin, const torch::Tensor &num_seqlen_per_req,
    const torch::Tensor &q_index, const torch::Tensor &kvcache_indices, bool is_prefill,
    const torch::Tensor &k_scale, const torch::Tensor &v_scale, int64_t quant_policy,
    int64_t max_seqlens, std::optional<double> upper_max_double,
    std::optional<torch::Tensor> q_scale_inv_opt, std::optional<torch::Tensor> q_norm_weight_opt,
    std::optional<torch::Tensor> k_norm_weight_opt, std::optional<torch::Tensor> out_q_opt,
    std::optional<torch::Tensor> out_k_opt, std::optional<torch::Tensor> out_v_opt,
    int64_t qk_norm_policy) {
  auto stream = at::cuda::getCurrentCUDAStream(qkv.get_device());
  TORCH_CHECK(qkv.is_contiguous(), "qkv tensor must be contiguous");
  TORCH_CHECK(cos_sin.is_contiguous(), "cos_sin tensor must be contiguous");
  TORCH_CHECK(num_seqlen_per_req.is_contiguous(), "num_seqlen_per_req tensor must be contiguous");
  TORCH_CHECK(kvcache_indices.is_contiguous(), "kvcache_indices tensor must be contiguous");
  TORCH_CHECK(k_scale.dim() == 1 && k_scale.size(0) == 1, "k_scale must contain 1 element");
  TORCH_CHECK(v_scale.dim() == 1 && v_scale.size(0) == 1, "v_scale must contain 1 element");
  TORCH_CHECK(quant_policy == 1 || quant_policy == 2, "quant_policy must be 1 or 2");
  TORCH_CHECK(qkv.scalar_type() == torch::kBFloat16, "qkv must be bfloat16");
  TORCH_CHECK(kcache.dtype().itemsize() == 1, "kcache must be 1-byte dtype");
  TORCH_CHECK(vcache.dtype().itemsize() == 1, "vcache must be 1-byte dtype");

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
  int max_num_kv_block_per_batch = kvcache_indices.size(1);
  int kcache_block_offset = kcache.stride(0);
  int vcache_block_offset = vcache.stride(0);

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
  if (quant_policy == 1) {
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
      k_scale.const_data_ptr<float>(), v_scale.const_data_ptr<float>(), q_scale_inv_ptr, upper_max,
      max_seqlens, kcache_block_offset, vcache_block_offset, num_req, max_num_kv_block_per_batch,
      kv_block_size, num_rows, num_q_heads, num_kv_heads, qk_head_dim, v_head_dim, is_prefill,
      qk_norm_policy, quant_policy, stream);

  return std::make_tuple(out_q, q_scale, split_k_flag);
}

}  // namespace rope
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "rope_norm_store_kv(Tensor! kcache, Tensor! vcache, Tensor qkv, Tensor cos_sin, "
      "Tensor num_seqlen_per_req, Tensor q_index, Tensor kvcache_indices, bool is_prefill, "
      "Tensor? q_norm_weight, Tensor? k_norm_weight, "
      "Tensor? out_q=None, Tensor? out_k=None, Tensor? out_v=None, int qk_norm_policy=0) -> "
      "Tensor");
  m.impl("rope_norm_store_kv", torch::kCUDA, &hpc::rope::rope_norm_store_kv_entry);

  m.def(
      "rope_norm_store_kv_fp8(Tensor! kcache, Tensor! vcache, Tensor qkv, "
      "Tensor cos_sin, Tensor num_seqlen_per_req, Tensor q_index, Tensor kvcache_indices, "
      "bool is_prefill, Tensor k_scale, Tensor v_scale, "
      "int quant_policy, int max_seqlens, float? upper_max, Tensor? q_scale_inv, "
      "Tensor? q_norm_weight, Tensor? k_norm_weight, "
      "Tensor? out_q=None, Tensor? out_k=None, Tensor? out_v=None, int qk_norm_policy=0) -> "
      "(Tensor, Tensor, Tensor)");
  m.impl("rope_norm_store_kv_fp8", torch::kCUDA, &hpc::rope::rope_norm_store_kv_fp8_entry);
}
