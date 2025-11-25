// Copyright 2025 hpc-ops authors

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include "src/attention/decode/decode.h"
#include "src/attention/prefill/prefill.h"

namespace hpc {
namespace attention {

torch::Tensor attention_prefill_bf16_entry(const torch::Tensor &q, const torch::Tensor &k,
                                           const torch::Tensor &v, const torch::Tensor &seqlens_q,
                                           const torch::Tensor &cu_seqlens_q, int64_t max_seqlens_q,
                                           std::optional<torch::Tensor> output) {
  auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
  TORCH_CHECK(q.device().is_cuda(), "q tensor must be cuda");
  TORCH_CHECK(k.device().is_cuda(), "k tensor must be cuda");
  TORCH_CHECK(v.device().is_cuda(), "v tensor must be cuda");
  TORCH_CHECK(seqlens_q.device().is_cuda(), "seqlens_q tensor must be cuda");
  TORCH_CHECK(cu_seqlens_q.device().is_cuda(), "cu_seqlens_q tensor must be cuda");

  int total_seq_q = q.size(0);
  int num_head_q = q.size(1);
  int num_dim_qk = q.size(2);

  int num_head_kv = v.size(1);
  int num_dim_v = v.size(2);

  int num_batch = seqlens_q.size(0);

  auto options = q.options().dtype(torch::kBFloat16);
  torch::Tensor y;
  if (output.has_value()) {
    y = output.value();
  } else {
    y = torch::empty({total_seq_q, num_head_q, num_dim_v}, options);
  }

  int num_tmas = 4 * num_batch;
  torch::Tensor tmas = torch::empty({num_tmas, 64}, options);

  const auto *q_ptr = q.const_data_ptr();
  const auto *k_ptr = k.const_data_ptr();
  const auto *v_ptr = v.const_data_ptr();
  const auto *seqlens_q_ptr = seqlens_q.const_data_ptr();
  const auto *cu_seqlens_q_ptr = cu_seqlens_q.const_data_ptr();
  void *tmas_ptr = tmas.mutable_data_ptr();

  using T = __nv_bfloat16;
  auto *y_ptr = reinterpret_cast<T *>(y.mutable_data_ptr());

  int ldQ = q.stride(0);  // num_head_q * num_dim_qk;
  int ldK = k.stride(0);  // num_head_kv * num_dim_qk;
  int ldV = v.stride(0);  // num_head_kv * num_dim_v;
  int ldY = y.stride(0);  // num_head_q * num_dim_v;

  attention_prefill_bf16_async(y_ptr, q_ptr, k_ptr, v_ptr, seqlens_q_ptr, cu_seqlens_q_ptr,
                               tmas_ptr, num_batch, total_seq_q, max_seqlens_q, num_dim_qk,
                               num_dim_v, num_head_q, num_head_kv, ldY, ldQ, ldK, ldV, stream);

  return y;
}

torch::Tensor attention_decode_bf16_entry(const torch::Tensor &q, torch::Tensor &kcache,
                                          torch::Tensor &vcache, const torch::Tensor &block_ids,
                                          const torch::Tensor &num_seq_kvcache,
                                          bool new_kv_included, bool use_splitk,
                                          std::optional<torch::Tensor> output) {
  auto stream = at::cuda::getCurrentCUDAStream(q.get_device());

  TORCH_CHECK(q.device().is_cuda(), "q tensor must be cuda");
  TORCH_CHECK(kcache.device().is_cuda(), "v tensor must be cuda");
  TORCH_CHECK(vcache.device().is_cuda(), "v tensor must be cuda");
  TORCH_CHECK(block_ids.device().is_cuda(), "v tensor must be cuda");
  TORCH_CHECK(block_ids.is_contiguous(), "block_ids tensor must be contiguous");
  TORCH_CHECK(num_seq_kvcache.is_contiguous(), "num_seq_kvcache tensor must be contiguous");
  TORCH_CHECK(block_ids.scalar_type() == torch::kInt32, "block_ids dtype must be int32");
  TORCH_CHECK(num_seq_kvcache.scalar_type() == torch::kInt32,
              "num_seq_kvcache dtype must be int32");

  int num_batch = num_seq_kvcache.size(0);
  int num_seq_q = q.size(0) / num_batch;
  TORCH_CHECK(num_seq_q == 1, "num_seq_q must be 1");
  int num_head_q = q.size(1);
  int num_dim_qk = q.size(2);

  int num_kvcache_blocks = kcache.size(0);
  int block_size = kcache.size(1);

  int num_head_k = kcache.size(2);
  int num_head_v = vcache.size(2);
  int num_dim_v = vcache.size(3);

  int num_seq_max_blocks = block_ids.size(1);

  const auto *q_ptr = q.const_data_ptr();
  auto *kcache_ptr = kcache.mutable_data_ptr();
  auto *vcache_ptr = vcache.mutable_data_ptr();
  const int *block_ids_ptr = block_ids.const_data_ptr<int>();
  const int *num_seq_kvcache_ptr = num_seq_kvcache.const_data_ptr<int>();

  auto options = q.options().dtype(torch::kBFloat16);
  torch::Tensor y;
  if (output.has_value()) {
    y = output.value();
  } else {
    y = torch::empty({num_batch * num_seq_q, num_head_q, num_dim_v}, options);
  }

  torch::Tensor lse;
  torch::Tensor split_out;

  int splitk = 0;
  // small batch increase splitk number to maximize sm usage.
  // 1. batch <= 32. split one request seqlenk to 16 parts.
  // 2. batch > 32. split one request seqlenk to 4 parts.
  if (use_splitk) {
    if (num_batch <= 32) {
      splitk = 16;
    } else {
      splitk = 4;
    }
  }

  if (splitk > 0) {
    lse = torch::empty({num_batch, splitk, num_head_q}, q.options().dtype(torch::kFloat32));
    split_out = torch::empty({num_batch, splitk, num_head_q, num_dim_v},
                             q.options().dtype(torch::kFloat32));
  }

  auto *lse_ptr = splitk > 0 ? lse.mutable_data_ptr() : nullptr;
  auto *split_out_ptr = splitk > 0 ? split_out.mutable_data_ptr() : nullptr;

  auto *y_ptr = y.mutable_data_ptr();

  int ldQ = q.stride(0);  // num_head_q * num_dim_qk;
  int ldK = kcache.stride(0);
  int ldV = vcache.stride(0);
  int ldY = y.stride(0);  // num_head_q * num_dim_v;

  bool running = attention_decode_bf16_async(
      y_ptr, lse_ptr, split_out_ptr, q_ptr, kcache_ptr, vcache_ptr, block_ids_ptr,
      num_seq_kvcache_ptr, new_kv_included, splitk, num_batch, num_head_q, num_head_k, num_head_v,
      num_dim_qk, num_dim_v, num_kvcache_blocks, block_size, num_seq_max_blocks, ldY, ldQ, ldK, ldV,
      stream);

  TORCH_CHECK(running, "attn decode kernel launch failed!");

  return y;
}

}  // namespace attention
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "attention_prefill_bf16(Tensor q, Tensor k, Tensor v, Tensor seqlens_q, Tensor cu_seqlens_q, "
      "int max_seqlens_q, Tensor? output) -> (Tensor)");
  m.impl("attention_prefill_bf16", torch::kCUDA, &hpc::attention::attention_prefill_bf16_entry);

  m.def(
      "attention_decode_bf16(Tensor q, Tensor! kcache, Tensor! vcache, Tensor block_ids, Tensor "
      "num_seq_kvcache, bool new_kv_included, bool use_splitk, Tensor? output) -> (Tensor)");
  m.impl("attention_decode_bf16", torch::kCUDA, &hpc::attention::attention_decode_bf16_entry);
}
