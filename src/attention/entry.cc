// Copyright 2025 hpc-ops authors

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include "src/attention/attention.h"

namespace hpc {
namespace attention {

torch::Tensor attention_prefill_bf16_entry(const torch::Tensor &q, const torch::Tensor &k,
                                           const torch::Tensor &v) {
  auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
  TORCH_CHECK(q.device().is_cuda(), "q tensor must be cuda");
  TORCH_CHECK(k.device().is_cuda(), "k tensor must be cuda");
  TORCH_CHECK(v.device().is_cuda(), "v tensor must be cuda");

  int num_batch = q.size(0);
  int num_seq_q = q.size(1);
  int num_head_q = q.size(2);
  int num_dim_qk = q.size(3);

  int num_seq_kv = v.size(1);
  int num_head_kv = v.size(2);
  int num_dim_v = v.size(3);

  const auto *q_ptr = q.const_data_ptr();
  const auto *k_ptr = k.const_data_ptr();
  const auto *v_ptr = v.const_data_ptr();

  auto options = q.options().dtype(torch::kBFloat16);
  torch::Tensor y = torch::empty({num_batch, num_seq_q, num_head_q, num_dim_v}, options);

  using T = __nv_bfloat16;
  auto *y_ptr = reinterpret_cast<T *>(y.mutable_data_ptr());

  int ldQ = q.stride(1);  // num_head_q * num_dim_qk;
  int ldK = k.stride(1);  // num_head_kv * num_dim_qk;
  int ldV = v.stride(1);  // num_head_kv * num_dim_v;
  int ldY = y.stride(1);  // num_head_q * num_dim_v;

  attention_prefill_bf16_async(y_ptr, q_ptr, k_ptr, v_ptr, num_batch, num_seq_q, num_seq_kv,
                               num_dim_qk, num_dim_v, num_head_q, num_head_kv, ldY, ldQ, ldK, ldV,
                               stream);

  return y;
}

torch::Tensor attention_decode_bf16_entry(const torch::Tensor &q, torch::Tensor &kvcache,
                                          const torch::Tensor &block_ids,
                                          const torch::Tensor &cache_lens,
                                          std::optional<torch::Tensor> output) {
  auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
  TORCH_CHECK(q.device().is_cuda(), "q tensor must be cuda");
  TORCH_CHECK(kvcache.device().is_cuda(), "v tensor must be cuda");
  TORCH_CHECK(block_ids.device().is_cuda(), "v tensor must be cuda");
  TORCH_CHECK(kvcache.is_contiguous(), "kvcache tensor must be contiguous");
  TORCH_CHECK(block_ids.is_contiguous(), "block_ids tensor must be contiguous");
  TORCH_CHECK(cache_lens.is_contiguous(), "cache_lens tensor must be contiguous");
  TORCH_CHECK(block_ids.scalar_type() == torch::kInt32, "block_ids dtype must be int32");
  TORCH_CHECK(cache_lens.scalar_type() == torch::kInt32, "cache_lens dtype must be int32");

  int num_batch = cache_lens.size(0);
  int num_seq_q = q.size(0) / num_batch;
  int num_head_q = q.size(1);
  int num_dim_qk = q.size(2);

  int num_blocks = kvcache.size(0);
  int block_size = kvcache.size(2);
  int num_head_kv = kvcache.size(3);
  int num_dim_v = kvcache.size(4);

  int max_num_blocks = block_ids.size(1);

  int head_per_group = num_head_q / num_head_kv;

  const auto *q_ptr = q.const_data_ptr();
  auto *kvcache_ptr = kvcache.mutable_data_ptr();
  const int *block_ids_ptr = block_ids.const_data_ptr<int>();
  const int *cache_lens_ptr = cache_lens.const_data_ptr<int>();

  auto options = q.options().dtype(torch::kBFloat16);
  torch::Tensor y = torch::empty({num_batch * num_seq_q, num_head_q, num_dim_v}, options);
  if (output.has_value()) {
    y = output.value();
  }

  auto *y_ptr = y.mutable_data_ptr();

  int ldQ = q.stride(0);  // num_head_q * num_dim_qk;
  int ldY = y.stride(0);  // num_head_q * num_dim_v;

  bool running = false;
  if (num_dim_qk == 80) {
    running = attention_decode_bf16_headdim80_async(
        y_ptr, q_ptr, kvcache_ptr, block_ids_ptr, cache_lens_ptr, num_batch, num_seq_q, num_head_q,
        num_head_kv, head_per_group, num_dim_qk, num_dim_v, num_blocks, block_size, max_num_blocks,
        ldY, ldQ, stream);
  } else if (num_dim_qk == 128) {
    running = attention_decode_bf16_headdim128_async(
        y_ptr, q_ptr, kvcache_ptr, block_ids_ptr, cache_lens_ptr, num_batch, num_seq_q, num_head_q,
        num_head_kv, head_per_group, num_dim_qk, num_dim_v, num_blocks, block_size, max_num_blocks,
        ldY, ldQ, stream);
  }

  TORCH_CHECK(running, "attn decode kernel launch failed!");

  return y;
}

}  // namespace attention
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def("attention_prefill_bf16", &hpc::attention::attention_prefill_bf16_entry);
  m.def("attention_decode_bf16", &hpc::attention::attention_decode_bf16_entry);
}
