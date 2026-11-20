// Copyright (C) 2026 Tencent.
#include <torch/library.h>

#include <optional>

#include "src/rope/amd/rope.h"
#include "src/utils/amd/runtime.h"

namespace hpc::rope::amd {

at::Tensor rope_entry(at::Tensor& kcache, at::Tensor& vcache, const at::Tensor& qkv,
                      const at::Tensor& cos_sin, const at::Tensor& seqlens,
                      const at::Tensor& q_index, const at::Tensor& block_ids, bool is_prefill,
                      std::optional<at::Tensor> q_weight, std::optional<at::Tensor> k_weight,
                      std::optional<at::Tensor> out_q, std::optional<at::Tensor> out_k,
                      std::optional<at::Tensor> out_v, int64_t norm_policy) {
  hpc::amd::check_tensor(qkv, qkv, at::kBFloat16, "qkv");
  hpc::amd::check_tensor(cos_sin, qkv, at::kFloat, "cos_sin");
  hpc::amd::check_tensor(seqlens, qkv, at::kInt, "num_seqlen_per_req");
  hpc::amd::check_tensor(q_index, qkv, at::kInt, "q_index");
  hpc::amd::check_tensor(block_ids, qkv, at::kInt, "kvcache_indices");
  TORCH_CHECK(qkv.dim() == 2 && seqlens.dim() == 1 && q_index.dim() == 1 && block_ids.dim() == 2 &&
                  cos_sin.dim() == 2 && cos_sin.size(1) == 128,
              "invalid RoPE input dimensions; head_dim must be 128");
  TORCH_CHECK(kcache.dim() == 4 && vcache.dim() == 4, "KV caches must be 4-D");
  for (const auto* cache : {&kcache, &vcache}) {
    TORCH_CHECK(cache->is_cuda() && cache->device() == qkv.device() &&
                    cache->scalar_type() == at::kBFloat16,
                "KV caches must be BF16 on the input GPU");
    TORCH_CHECK(cache->size(3) == 128 && cache->stride(3) == 1 && cache->stride(2) == 128 &&
                    cache->stride(1) == cache->size(2) * 128 &&
                    cache->stride(0) >= cache->size(1) * cache->stride(1),
                "KV cache rows must be contiguous and non-overlapping");
  }
  TORCH_CHECK(kcache.sizes() == vcache.sizes() && kcache.size(1) > 0,
              "K/V cache shapes must match and block_size must be positive");
  const int requests = seqlens.size(0);
  TORCH_CHECK(q_index.numel() == requests + 1 && block_ids.size(0) == requests &&
                  (requests == 0 || block_ids.size(1) > 0),
              "request metadata shapes do not match");
  const int kv_heads = kcache.size(2);
  TORCH_CHECK(kv_heads > 0 && qkv.size(1) % 128 == 0, "invalid QKV width");
  const int q_heads = qkv.size(1) / 128 - 2 * kv_heads;
  TORCH_CHECK((q_heads == 8 && kv_heads == 1) || (q_heads == 64 && kv_heads == 8),
              "supported (Q heads, KV heads) are (8,1) and (64,8)");
  TORCH_CHECK(norm_policy >= 0 && norm_policy <= 2, "qk_norm_policy must be 0, 1 or 2");
  if (norm_policy != 0) {
    TORCH_CHECK(q_weight && k_weight, "Q/K norm weights are required for qk_norm_policy != 0");
    for (const auto* weight : {&*q_weight, &*k_weight}) {
      hpc::amd::check_tensor(*weight, qkv, at::kFloat, "norm weight");
      TORCH_CHECK(weight->dim() == 1 && weight->size(0) == 128,
                  "norm weight must have shape [128]");
    }
  }
  const c10::DeviceGuard guard(qkv.device());
  auto q = out_q ? *out_q : at::empty({qkv.size(0), q_heads, 128}, qkv.options());
  auto check_output = [&](const at::Tensor& out, int heads) {
    hpc::amd::check_tensor(out, qkv, at::kBFloat16, "output");
    TORCH_CHECK(
        out.dim() == 3 && out.size(0) == qkv.size(0) && out.size(1) == heads && out.size(2) == 128,
        "output has an invalid shape");
  };
  check_output(q, q_heads);
  if (out_k) check_output(*out_k, kv_heads);
  if (out_v) check_output(*out_v, kv_heads);
  // q_index describes both prefill and decode/MTP rows; no device-to-host synchronization.
  rope_norm_store_kv_async(
      hpc::amd::bf16_data(q), hpc::amd::bf16_data(kcache), hpc::amd::bf16_data(vcache),
      out_k ? hpc::amd::bf16_data(*out_k) : nullptr, out_v ? hpc::amd::bf16_data(*out_v) : nullptr,
      hpc::amd::bf16_data(qkv), cos_sin.data_ptr<float>(), seqlens.data_ptr<int>(),
      q_index.data_ptr<int>(), block_ids.data_ptr<int>(),
      norm_policy ? q_weight->data_ptr<float>() : nullptr,
      norm_policy ? k_weight->data_ptr<float>() : nullptr, qkv.size(0), requests, q_heads, kv_heads,
      kcache.size(1), block_ids.size(1), kcache.stride(0), vcache.stride(0), norm_policy,
      hpc::amd::current_stream(qkv));
  C10_HIP_CHECK(hipGetLastError());
  return q;
}

}  // namespace hpc::rope::amd

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "rope_norm_store_kv(Tensor! kcache, Tensor! vcache, Tensor qkv, Tensor cos_sin, "
      "Tensor num_seqlen_per_req, Tensor q_index, Tensor kvcache_indices, bool is_prefill, "
      "Tensor? q_norm_weight, Tensor? k_norm_weight, Tensor? out_q=None, "
      "Tensor? out_k=None, Tensor? out_v=None, int qk_norm_policy=0) -> Tensor");
  m.impl("rope_norm_store_kv", c10::DispatchKey::CUDA, &hpc::rope::amd::rope_entry);
  // TODO(amd): rope_norm_store_kv_fp8 is registered as an explicit FP8 TODO in unsupported.cc.
}
