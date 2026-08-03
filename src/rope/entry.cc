// Copyright (C) 2026 Tencent.

#include <ATen/MemoryOverlap.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cstdint>
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

std::tuple<torch::Tensor, torch::Tensor> multimodal_rope_impl(
    const torch::Tensor &q, const torch::Tensor &k, const torch::Tensor &cos_sin_cache,
    const torch::Tensor &positions, bool is_neox, std::optional<torch::Tensor> q_out_opt,
    std::optional<torch::Tensor> k_out_opt) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && cos_sin_cache.is_cuda() && positions.is_cuda(),
              "q/k/cos_sin_cache/positions must be CUDA tensors");
  TORCH_CHECK(q.device() == k.device() && q.device() == cos_sin_cache.device() &&
                  q.device() == positions.device(),
              "q/k/cos_sin_cache/positions must be on the same CUDA device");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4, "q/k must be [B,H,S,D]");
  TORCH_CHECK(cos_sin_cache.dim() == 2, "cos_sin_cache must be [max_position, head_dim]");
  TORCH_CHECK(positions.dim() == 2, "positions must be [B, S]");
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16 && k.scalar_type() == torch::kBFloat16,
              "q/k must be bfloat16");
  TORCH_CHECK(cos_sin_cache.scalar_type() == torch::kFloat, "cos_sin_cache must be float32");
  TORCH_CHECK(positions.scalar_type() == torch::kInt64, "positions must be int64");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(2) == k.size(2) && q.size(3) == k.size(3),
              "q/k shape mismatch");
  TORCH_CHECK(cos_sin_cache.size(1) == q.size(3),
              "cos_sin_cache last dimension must equal head_dim (first half cos, second half "
              "sin; not duplicated/expanded per batch or sequence)");
  TORCH_CHECK(positions.size(0) == q.size(0) && positions.size(1) == q.size(2),
              "positions shape mismatch, expected [B, S]");
  const int64_t q_heads = q.size(1);
  const int64_t kv_heads = k.size(1);
  const int64_t head_dim = q.size(3);
  const bool neox_profile =
      is_neox && (head_dim == 128 || head_dim == 512) &&
      ((q_heads == 16 && kv_heads == 8) || (q_heads == 28 && kv_heads == 4) ||
       (q_heads == 32 && kv_heads == 8) || (q_heads == 64 && (kv_heads == 4 || kv_heads == 8)));
  const bool interleaved_profile =
      !is_neox && head_dim == 64 && (q_heads == 32 || q_heads == 64) && kv_heads == 1;
  TORCH_CHECK(neox_profile || interleaved_profile,
              "multimodal_rope supports NeoX D128/D512 Hq/Hkv 16/8, 28/4, 32/8, "
              "64/4, 64/8 and interleaved D64 Hq/Hkv 32/1, 64/1");
  TORCH_CHECK(q.stride(3) == 1 && k.stride(3) == 1, "q/k last dimension must be contiguous");
  TORCH_CHECK(cos_sin_cache.stride(1) == 1, "cos_sin_cache last dimension must be contiguous");
  at::assert_no_internal_overlap(q);
  at::assert_no_internal_overlap(k);
  for (int64_t dim = 0; dim < q.dim(); ++dim) {
    TORCH_CHECK(q.size(dim) <= 1 || q.stride(dim) > 0,
                "q must have positive strides for non-singleton dimensions");
    TORCH_CHECK(k.size(dim) <= 1 || k.stride(dim) > 0,
                "k must have positive strides for non-singleton dimensions");
  }

  auto prepare_output = [](const std::optional<torch::Tensor> &output_opt,
                           const torch::Tensor &input, const char *name) {
    if (!output_opt.has_value()) {
      return torch::empty_strided(input.sizes(), input.strides(), input.options());
    }
    torch::Tensor output = output_opt.value();
    TORCH_CHECK(output.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(output.device() == input.device(), name, " must be on the input device");
    TORCH_CHECK(output.scalar_type() == input.scalar_type(), name, " must match input dtype");
    TORCH_CHECK(output.sizes() == input.sizes(), name, " must match input shape");
    TORCH_CHECK(output.stride(3) == 1, name, " last dimension must be contiguous");
    at::assert_no_internal_overlap(output);
    for (int64_t dim = 0; dim < output.dim(); ++dim) {
      TORCH_CHECK(output.size(dim) <= 1 || output.stride(dim) > 0, name,
                  " must have positive strides for non-singleton dimensions");
    }
    return output;
  };
  torch::Tensor q_out = prepare_output(q_out_opt, q, "q_out");
  torch::Tensor k_out = prepare_output(k_out_opt, k, "k_out");
  const std::uintptr_t tensor_alignment = is_neox ? 4 : 16;
  const int64_t stride_alignment = is_neox ? 2 : 8;
  TORCH_CHECK(
      reinterpret_cast<std::uintptr_t>(q.const_data_ptr()) % tensor_alignment == 0 &&
          reinterpret_cast<std::uintptr_t>(k.const_data_ptr()) % tensor_alignment == 0 &&
          reinterpret_cast<std::uintptr_t>(q_out.const_data_ptr()) % tensor_alignment == 0 &&
          reinterpret_cast<std::uintptr_t>(k_out.const_data_ptr()) % tensor_alignment == 0 &&
          reinterpret_cast<std::uintptr_t>(cos_sin_cache.const_data_ptr()) % 16 == 0,
      "multimodal_rope inputs and outputs do not satisfy vector alignment");
  TORCH_CHECK(
      q.stride(0) % stride_alignment == 0 && q.stride(1) % stride_alignment == 0 &&
          q.stride(2) % stride_alignment == 0 && k.stride(0) % stride_alignment == 0 &&
          k.stride(1) % stride_alignment == 0 && k.stride(2) % stride_alignment == 0 &&
          q_out.stride(0) % stride_alignment == 0 && q_out.stride(1) % stride_alignment == 0 &&
          q_out.stride(2) % stride_alignment == 0 && k_out.stride(0) % stride_alignment == 0 &&
          k_out.stride(1) % stride_alignment == 0 && k_out.stride(2) % stride_alignment == 0 &&
          cos_sin_cache.stride(0) % 4 == 0,
      "multimodal_rope inputs and outputs do not satisfy vector stride alignment");
  if (q_out_opt.has_value()) {
    at::assert_no_overlap(q_out, q);
    at::assert_no_overlap(q_out, k);
    at::assert_no_overlap(q_out, cos_sin_cache);
    at::assert_no_overlap(q_out, positions);
  }
  if (k_out_opt.has_value()) {
    at::assert_no_overlap(k_out, q);
    at::assert_no_overlap(k_out, k);
    at::assert_no_overlap(k_out, cos_sin_cache);
    at::assert_no_overlap(k_out, positions);
  }
  if (q_out_opt.has_value() && k_out_opt.has_value()) {
    at::assert_no_overlap(q_out, k_out);
  }
  if (q.size(0) == 0 || q.size(2) == 0) {
    return std::make_tuple(q_out, k_out);
  }
  MultimodalRopeParams params{q.size(0),           q.size(2),           q.size(1),
                              k.size(1),           q.size(3),           q.stride(0),
                              q.stride(1),         q.stride(2),         k.stride(0),
                              k.stride(1),         k.stride(2),         q_out.stride(0),
                              q_out.stride(1),     q_out.stride(2),     k_out.stride(0),
                              k_out.stride(1),     k_out.stride(2),     cos_sin_cache.stride(0),
                              positions.stride(0), positions.stride(1), is_neox};
  auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
  multimodal_rope_async(reinterpret_cast<__nv_bfloat16 *>(q_out.mutable_data_ptr()),
                        reinterpret_cast<__nv_bfloat16 *>(k_out.mutable_data_ptr()),
                        reinterpret_cast<const __nv_bfloat16 *>(q.const_data_ptr()),
                        reinterpret_cast<const __nv_bfloat16 *>(k.const_data_ptr()),
                        reinterpret_cast<const float *>(cos_sin_cache.const_data_ptr()),
                        positions.const_data_ptr<int64_t>(), params, stream);
  const cudaError_t launch_error = cudaGetLastError();
  TORCH_CHECK(launch_error == cudaSuccess,
              "multimodal_rope kernel launch failed: ", cudaGetErrorString(launch_error));
  return std::make_tuple(q_out, k_out);
}

std::tuple<torch::Tensor, torch::Tensor> multimodal_rope_entry(const torch::Tensor &q,
                                                               const torch::Tensor &k,
                                                               const torch::Tensor &cos_sin_cache,
                                                               const torch::Tensor &positions,
                                                               bool is_neox) {
  return multimodal_rope_impl(q, k, cos_sin_cache, positions, is_neox, std::nullopt, std::nullopt);
}

std::tuple<torch::Tensor, torch::Tensor> multimodal_rope_out_entry(
    const torch::Tensor &q, const torch::Tensor &k, const torch::Tensor &cos_sin_cache,
    const torch::Tensor &positions, bool is_neox, torch::Tensor q_out, torch::Tensor k_out) {
  return multimodal_rope_impl(q, k, cos_sin_cache, positions, is_neox, q_out, k_out);
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

  m.def(
      "multimodal_rope(Tensor q, Tensor k, Tensor cos_sin_cache, Tensor positions, "
      "bool is_neox=True) "
      "-> (Tensor, Tensor)");
  m.impl("multimodal_rope", torch::kCUDA, &hpc::rope::multimodal_rope_entry);
  m.def(
      "multimodal_rope.out(Tensor q, Tensor k, Tensor cos_sin_cache, Tensor positions, "
      "bool is_neox=True, *, Tensor(a!) out_q, Tensor(b!) out_k) -> (Tensor(a!), Tensor(b!))");
  m.impl("multimodal_rope.out", torch::kCUDA, &hpc::rope::multimodal_rope_out_entry);
}
