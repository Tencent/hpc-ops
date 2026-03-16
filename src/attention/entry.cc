// Copyright 2025 hpc-ops authors

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include "src/attention/decode/decode.h"
#include "src/attention/mla/mla.h"
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

torch::Tensor attention_with_kvcache_prefill_bf16_entry(
    const torch::Tensor &q, const torch::Tensor &kcache, const torch::Tensor &vcache,
    const torch::Tensor &cu_seqlens_q, const torch::Tensor block_ids,
    const torch::Tensor seqlens_kvcache, int64_t max_seqlens_q,
    std::optional<torch::Tensor> output) {
  auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
  TORCH_CHECK(q.device().is_cuda(), "q tensor must be cuda");
  TORCH_CHECK(kcache.device().is_cuda(), "kcache tensor must be cuda");
  TORCH_CHECK(vcache.device().is_cuda(), "vcache tensor must be cuda");
  TORCH_CHECK(cu_seqlens_q.device().is_cuda(), "cu_seqlens_q tensor must be cuda");
  TORCH_CHECK(block_ids.device().is_cuda(), "block_ids tensor must be cuda");
  TORCH_CHECK(seqlens_kvcache.device().is_cuda(), "seqlens_kvcache tensor must be cuda");

  int total_seq_q = q.size(0);
  int num_head_q = q.size(1);
  int num_dim_qk = q.size(2);

  int num_batch = cu_seqlens_q.size(0) - 1;

  int num_kvcache_blocks = kcache.size(0);
  int block_size = kcache.size(1);

  int num_head_kv = kcache.size(2);
  int num_dim_v = vcache.size(3);

  int num_seq_max_blocks = block_ids.size(1);

  auto options = q.options().dtype(torch::kBFloat16);
  torch::Tensor y;
  if (output.has_value()) {
    y = output.value();
  } else {
    y = torch::empty({total_seq_q, num_head_q, num_dim_v}, options);
  }

  int num_tmas = 2 * num_batch;
  torch::Tensor tmas = torch::empty({num_tmas, 64}, options);

  const auto *q_ptr = q.const_data_ptr();
  const auto *kcache_ptr = kcache.const_data_ptr();
  const auto *vcache_ptr = vcache.const_data_ptr();
  const auto *cu_seqlens_q_ptr = cu_seqlens_q.const_data_ptr();
  const auto *block_ids_ptr = block_ids.const_data_ptr();
  const auto *seqlens_kvcache_ptr = seqlens_kvcache.const_data_ptr();
  void *tmas_ptr = tmas.mutable_data_ptr();

  using T = __nv_bfloat16;
  auto *y_ptr = reinterpret_cast<T *>(y.mutable_data_ptr());

  int ldQ = q.stride(0);       // num_head_q * num_dim_qk;
  int ldK = kcache.stride(0);  // num_head_kv * num_dim_qk;
  int ldV = vcache.stride(0);  // num_head_kv * num_dim_v;
  int ldY = y.stride(0);       // num_head_q * num_dim_v;

  attention_with_kvcache_prefill_bf16_async(
      y_ptr, q_ptr, kcache_ptr, vcache_ptr, cu_seqlens_q_ptr, block_ids_ptr, seqlens_kvcache_ptr,
      tmas_ptr, num_batch, total_seq_q, max_seqlens_q, num_dim_qk, num_dim_v, num_head_q,
      num_head_kv, num_kvcache_blocks, block_size, num_seq_max_blocks, ldY, ldQ, ldK, ldV, stream);

  return y;
}

torch::Tensor attention_with_kvcache_prefill_fp8_entry(
    const torch::Tensor &q, const torch::Tensor &kcache, const torch::Tensor &vcache,
    const torch::Tensor &qkscale, const torch::Tensor &vscale, const torch::Tensor &cu_seqlens_q,
    const torch::Tensor block_ids, const torch::Tensor seqlens_kvcache, int64_t max_seqlens_q,
    std::optional<torch::Tensor> output) {
  auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
  TORCH_CHECK(q.device().is_cuda(), "q tensor must be cuda");
  TORCH_CHECK(kcache.device().is_cuda(), "kcache tensor must be cuda");
  TORCH_CHECK(vcache.device().is_cuda(), "vcache tensor must be cuda");
  TORCH_CHECK(qkscale.device().is_cuda(), "qkscale tensor must be cuda");
  TORCH_CHECK(vscale.device().is_cuda(), "vscale tensor must be cuda");
  TORCH_CHECK(cu_seqlens_q.device().is_cuda(), "cu_seqlens_q tensor must be cuda");
  TORCH_CHECK(block_ids.device().is_cuda(), "block_ids tensor must be cuda");
  TORCH_CHECK(seqlens_kvcache.device().is_cuda(), "seqlens_kvcache tensor must be cuda");

  int total_seq_q = q.size(0);
  int num_head_q = q.size(1);
  int num_dim_qk = q.size(2);

  int num_batch = cu_seqlens_q.size(0) - 1;

  int num_kvcache_blocks = kcache.size(0);
  int block_size = kcache.size(1);

  int num_head_kv = kcache.size(2);
  int num_dim_v = vcache.size(3);

  int num_seq_max_blocks = block_ids.size(1);

  int max_seqlens_q_pad = qkscale.size(2);

  auto options = q.options().dtype(torch::kBFloat16);
  torch::Tensor y;
  if (output.has_value()) {
    y = output.value();
  } else {
    y = torch::empty({total_seq_q, num_head_q, num_dim_v}, options);
  }

  int num_tmas = 2 * num_batch;
  torch::Tensor tmas = torch::empty({num_tmas, 64}, options);

  const auto *q_ptr = q.const_data_ptr();
  const auto *kcache_ptr = kcache.const_data_ptr();
  const auto *vcache_ptr = vcache.const_data_ptr();
  const auto *qkscale_ptr = qkscale.const_data_ptr();
  const auto *vscale_ptr = vscale.const_data_ptr();
  const auto *cu_seqlens_q_ptr = cu_seqlens_q.const_data_ptr();
  const auto *block_ids_ptr = block_ids.const_data_ptr();
  const auto *seqlens_kvcache_ptr = seqlens_kvcache.const_data_ptr();
  void *tmas_ptr = tmas.mutable_data_ptr();

  using T = __nv_bfloat16;
  auto *y_ptr = reinterpret_cast<T *>(y.mutable_data_ptr());

  int ldQ = q.stride(0);       // num_head_q * num_dim_qk;
  int ldK = kcache.stride(0);  // num_head_kv * num_dim_qk;
  int ldV = vcache.stride(0);  // num_head_kv * num_dim_v;
  int ldY = y.stride(0);       // num_head_q * num_dim_v;

  attention_with_kvcache_prefill_fp8_async(
      y_ptr, q_ptr, kcache_ptr, vcache_ptr, qkscale_ptr, vscale_ptr, cu_seqlens_q_ptr,
      block_ids_ptr, seqlens_kvcache_ptr, tmas_ptr, num_batch, total_seq_q, max_seqlens_q,
      max_seqlens_q_pad, num_dim_qk, num_dim_v, num_head_q, num_head_kv, num_kvcache_blocks,
      block_size, num_seq_max_blocks, ldY, ldQ, ldK, ldV, stream);

  return y;
}

torch::Tensor attention_decode_bf16_entry(const torch::Tensor &q, torch::Tensor &kcache,
                                          torch::Tensor &vcache, const torch::Tensor &block_ids,
                                          const torch::Tensor &num_seq_kvcache, int64_t mtp,
                                          bool new_kv_included, bool use_splitk,
                                          std::optional<torch::Tensor> split_flag,
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
  TORCH_CHECK((mtp == 0 || mtp == 1 || mtp == 2), "we only support mtp 0, 1, 2.");

  int num_batch = num_seq_kvcache.size(0);
  int num_seq_q = q.size(0) / num_batch;
  TORCH_CHECK(num_seq_q == mtp + 1, "every request num_seq_q must be mtp + 1");
  int num_head_q = q.size(1);
  int num_dim_qk = q.size(2);

  TORCH_CHECK((num_dim_qk == 128 || num_dim_qk == 80), "we only support head dim 128 and 80.");

  int num_kvcache_blocks = kcache.size(0);
  int block_size = kcache.size(1);

  TORCH_CHECK((block_size == 32 || block_size == 64), "kvcache paged blocksize must be 32 and 64.");

  int num_head_k = kcache.size(2);
  int num_head_v = vcache.size(2);
  int num_dim_v = vcache.size(3);

  int num_seq_max_blocks = block_ids.size(1);

  int heads_per_group = num_head_q / num_head_k;
  TORCH_CHECK(heads_per_group == 4 || heads_per_group == 8,
              "we only support num_head_q / num_head_k == 4 or 8.");

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

  int splitk = 1;
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

  torch::Tensor split_flag_tensor;

  if (splitk > 1) {
    int pad_heads_per_group = ((heads_per_group + 7) / 8) * 8;
    lse = torch::empty({num_batch, splitk, num_head_k, num_seq_q, pad_heads_per_group},
                       q.options().dtype(torch::kFloat32));
    split_out = torch::empty({num_batch, splitk, num_seq_q, num_head_q, num_dim_v},
                             q.options().dtype(torch::kFloat32));
    if (split_flag.has_value()) {
      split_flag_tensor = split_flag.value();
    } else {
      split_flag_tensor = torch::zeros({num_batch, num_head_k}, q.options().dtype(torch::kInt32));
    }
  }

  auto *lse_ptr = splitk > 1 ? lse.mutable_data_ptr() : nullptr;
  auto *split_out_ptr = splitk > 1 ? split_out.mutable_data_ptr() : nullptr;
  auto *split_flag_ptr = splitk > 1 ? split_flag_tensor.mutable_data_ptr<int>() : nullptr;

  auto *y_ptr = y.mutable_data_ptr();

  int ldQ = q.stride(0);  // num_head_q * num_dim_qk;
  int ldK = kcache.stride(0);
  int ldV = vcache.stride(0);
  int ldY = y.stride(0);  // num_head_q * num_dim_v;

  bool running = attention_decode_bf16_async(
      y_ptr, lse_ptr, split_out_ptr, q_ptr, kcache_ptr, vcache_ptr, block_ids_ptr,
      num_seq_kvcache_ptr, split_flag_ptr, new_kv_included, splitk, num_batch, num_seq_q,
      num_head_q, num_head_k, num_head_v, num_dim_qk, num_dim_v, num_kvcache_blocks, block_size,
      num_seq_max_blocks, ldY, ldQ, ldK, ldV, stream);

  TORCH_CHECK(running, "attn decode kernel launch failed!");

  return y;
}

torch::Tensor attention_decode_fp8_entry(const torch::Tensor &q, torch::Tensor &kcache,
                                         torch::Tensor &vcache, const torch::Tensor &block_ids,
                                         const torch::Tensor &num_seq_kvcache,
                                         const torch::Tensor &qscale, const torch::Tensor &kscale,
                                         const torch::Tensor &vscale, int64_t mtp,
                                         bool new_kv_included, bool use_splitk,
                                         std::optional<torch::Tensor> split_flag,
                                         std::optional<torch::Tensor> output) {
  auto stream = at::cuda::getCurrentCUDAStream(q.get_device());

  TORCH_CHECK(q.device().is_cuda(), "q tensor must be cuda");
  TORCH_CHECK(kcache.device().is_cuda(), "v tensor must be cuda");
  TORCH_CHECK(vcache.device().is_cuda(), "v tensor must be cuda");
  TORCH_CHECK(block_ids.device().is_cuda(), "v tensor must be cuda");
  TORCH_CHECK(block_ids.is_contiguous(), "block_ids tensor must be contiguous");
  TORCH_CHECK(num_seq_kvcache.is_contiguous(), "num_seq_kvcache tensor must be contiguous");
  TORCH_CHECK(q.scalar_type() == torch::kFloat8_e4m3fn, "q dtype must be fp8_e4m3fn");
  TORCH_CHECK(kcache.dtype().itemsize() == 1, "kcache tensor element type size must be fp8_e4m3");
  TORCH_CHECK(vcache.dtype().itemsize() == 1, "vcache tensor element type size must be fp8_e4m3");
  TORCH_CHECK(block_ids.scalar_type() == torch::kInt32, "block_ids dtype must be int32");
  TORCH_CHECK(num_seq_kvcache.scalar_type() == torch::kInt32,
              "num_seq_kvcache dtype must be int32");
  TORCH_CHECK((mtp == 0 || mtp == 1 || mtp == 2), "we only support mtp 0, 1, 2.");

  int num_batch = num_seq_kvcache.size(0);
  int num_seq_q = q.size(0) / num_batch;
  TORCH_CHECK(num_seq_q == mtp + 1, "every request num_seq_q must be mtp + 1");
  int num_head_q = q.size(1);
  int num_dim_qk = q.size(2);

  TORCH_CHECK(num_dim_qk == 128, "we only support head dim 128.");

  int num_kvcache_blocks = kcache.size(0);
  int block_size = kcache.size(1);

  TORCH_CHECK(block_size == 64, "kvcache paged blocksize must be 64.");

  int num_head_k = kcache.size(2);
  int num_head_v = vcache.size(2);
  int num_dim_v = vcache.size(3);

  int num_seq_max_blocks = block_ids.size(1);
  int qscale_pad_stride = qscale.stride(0);

  int heads_per_group = num_head_q / num_head_k;
  TORCH_CHECK(heads_per_group == 4 || heads_per_group == 8,
              "we only support num_head_q / num_head_k == 4 or 8.");

  const auto *q_ptr = q.const_data_ptr();
  auto *kcache_ptr = kcache.mutable_data_ptr();
  auto *vcache_ptr = vcache.mutable_data_ptr();
  const int *block_ids_ptr = block_ids.const_data_ptr<int>();
  const int *num_seq_kvcache_ptr = num_seq_kvcache.const_data_ptr<int>();
  const float *qscale_ptr = qscale.const_data_ptr<float>();
  const float *kscale_ptr = kscale.const_data_ptr<float>();
  const float *vscale_ptr = vscale.const_data_ptr<float>();

  auto options = q.options().dtype(torch::kBFloat16);
  torch::Tensor y;
  if (output.has_value()) {
    y = output.value();
  } else {
    y = torch::empty({num_batch * num_seq_q, num_head_q, num_dim_v}, options);
  }

  torch::Tensor lse;
  torch::Tensor split_out;

  int splitk = 1;
  int splitk_min_len = 0;

  // small batch increase splitk number to maximize sm usage.
  if (use_splitk) {
    if (num_batch <= 32) {
      splitk = 4;
      splitk_min_len = 512;
    } else {
      splitk = 4;
      splitk_min_len = 4096;
    }
  }

  int consumers = 2;
  if (num_batch <= 156) {
    consumers = 2;
  } else if (num_batch <= 234) {
    consumers = 1;
  } else {
    consumers = 2;
  }

  torch::Tensor split_flag_tensor;

  bool use_split = splitk > 1 || consumers > 1;

  if (use_split) {
    int pad_heads_per_group = ((heads_per_group + 7) / 8) * 8;
    lse = torch::empty({num_batch, splitk * consumers, num_head_k, num_seq_q, pad_heads_per_group},
                       q.options().dtype(torch::kFloat32));
    split_out = torch::empty({num_batch, splitk * consumers, num_seq_q, num_head_q, num_dim_v},
                             q.options().dtype(torch::kFloat32));
    if (split_flag.has_value()) {
      split_flag_tensor = split_flag.value();
    } else {
      split_flag_tensor = torch::zeros({num_batch, num_head_k}, q.options().dtype(torch::kInt32));
    }
  }

  auto *lse_ptr = use_split ? lse.mutable_data_ptr() : nullptr;
  auto *split_out_ptr = use_split ? split_out.mutable_data_ptr() : nullptr;
  auto *split_flag_ptr = use_split ? split_flag_tensor.mutable_data_ptr<int>() : nullptr;

  auto *y_ptr = y.mutable_data_ptr();

  int ldQ = q.stride(0);  // num_head_q * num_dim_qk;
  int ldK = kcache.stride(0);
  int ldV = vcache.stride(0);
  int ldY = y.stride(0);  // num_head_q * num_dim_v;

  bool running = attention_decode_fp8_async(
      y_ptr, lse_ptr, split_out_ptr, q_ptr, kcache_ptr, vcache_ptr, block_ids_ptr,
      num_seq_kvcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr, split_flag_ptr, new_kv_included,
      splitk, splitk_min_len, consumers, num_batch, num_seq_q, num_head_q, num_head_k, num_head_v,
      num_dim_qk, num_dim_v, num_kvcache_blocks, block_size, num_seq_max_blocks, qscale_pad_stride,
      ldY, ldQ, ldK, ldV, stream);

  TORCH_CHECK(running, "attn decode kernel launch failed!");

  return y;
}

torch::Tensor attention_mla_with_kvcache_bf16_entry(const torch::Tensor &q,
                                                    const torch::Tensor &kvcache,
                                                    const torch::Tensor &block_ids,
                                                    const torch::Tensor &cu_seqlens_q,
                                                    const torch::Tensor &num_seq_kv,
                                                    std::optional<torch::Tensor> output) {
  auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
  TORCH_CHECK(q.device().is_cuda(), "q tensor must be cuda");
  TORCH_CHECK(kvcache.device().is_cuda(), "kvcache tensor must be cuda");
  TORCH_CHECK(block_ids.device().is_cuda(), "kvcache tensor must be cuda");
  TORCH_CHECK(cu_seqlens_q.device().is_cuda(), "kvcache tensor must be cuda");
  TORCH_CHECK(num_seq_kv.device().is_cuda(), "kvcache tensor must be cuda");

  int num_batch = num_seq_kv.size(0);
  int total_seq_q = q.size(0);
  int num_head_q = q.size(1);
  int head_dim = q.size(2);

  int num_kvcache_blocks = kvcache.size(0);

  TORCH_CHECK(block_size == 64, "we only support kvcache block size 64.");

  int num_seq_max_blocks = block_ids.size(1);

  const auto *q_ptr = q.const_data_ptr();
  auto *kvcache_ptr = kvcache.mutable_data_ptr();
  const int *block_ids_ptr = block_ids.const_data_ptr<int>();

  const int *cu_seqlens_q_ptr = cu_seqlens_q.const_data_ptr<int>();
  const int *num_seq_kv_ptr = num_seq_kv.const_data_ptr<int>();

  auto options = q.options().dtype(torch::kBFloat16);
  torch::Tensor y;
  if (output.has_value()) {
    y = output.value();
  } else {
    y = torch::empty({total_seq_q, num_head_q, head_dim}, options);
  }
  auto *y_ptr = y.data_ptr();

  int ldQ = q.stride(0);  // num_head_q * num_dim_qk;
  int ldKV = kvcache.stride(0);
  int ldY = y.stride(0);  // num_head_q * num_dim_v;

  bool running = attention_mla_with_kvcache_bf16_async(
      y_ptr, q_ptr, kvcache_ptr, block_ids_ptr, cu_seqlens_q_ptr, num_seq_kv_ptr, num_batch,
      total_seq_q, num_head_q, head_dim, num_kvcache_blocks, num_seq_max_blocks, ldY, ldQ, ldKV,
      stream);

  TORCH_CHECK(running, "attn decode kernel launch failed!");

  return y;
}

torch::Tensor attention_sparse_mla_with_kvcache_bf16_entry(
    const torch::Tensor &q, const torch::Tensor &win_kvcache, const torch::Tensor &win_block_ids,
    const torch::Tensor &win_topk_ids, const torch::Tensor &compress_kvcache,
    const torch::Tensor &compress_block_ids, const torch::Tensor &compress_topk_ids,
    const torch::Tensor &cu_seqlens_q, const torch::Tensor &sink_weight, double softmax_scale,
    std::optional<torch::Tensor> output) {
  auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
  TORCH_CHECK(q.device().is_cuda(), "q tensor must be cuda");
  TORCH_CHECK(win_kvcache.device().is_cuda(), "win_kvcache tensor must be cuda");
  TORCH_CHECK(win_block_ids.device().is_cuda(), "win_block_ids tensor must be cuda");
  TORCH_CHECK(win_topk_ids.device().is_cuda(), "win_topk_ids tensor must be cuda");

  TORCH_CHECK(compress_kvcache.device().is_cuda(), "compress_kvcache tensor must be cuda");
  TORCH_CHECK(compress_block_ids.device().is_cuda(), "compress_block_ids tensor must be cuda");
  TORCH_CHECK(compress_topk_ids.device().is_cuda(), "compress_topk_ids tensor must be cuda");

  TORCH_CHECK(cu_seqlens_q.device().is_cuda(), "cu_seqlens_q tensor must be cuda");
  TORCH_CHECK(sink_weight.device().is_cuda(), "sink_weight tensor must be cuda");

  int num_batch = cu_seqlens_q.size(0) - 1;
  int total_seq_q = q.size(0);
  int num_head_q = q.size(1);
  int head_dim = q.size(2);

  int num_win_kvcache_blocks = win_kvcache.size(0);
  int win_block_size = win_kvcache.size(1);

  int num_compress_kvcache_blocks = compress_kvcache.size(0);
  int compress_block_size = compress_kvcache.size(1);

  int num_win_seq_max_blocks = win_block_ids.size(1);
  int num_compress_seq_max_blocks = compress_block_ids.size(1);

  int num_win_max_topk = win_topk_ids.size(1);
  int num_compress_max_topk = compress_topk_ids.size(1);

  TORCH_CHECK(win_block_size == compress_block_size,
              "compress_block_size should equal win_block_size");

  const auto *q_ptr = q.const_data_ptr();
  auto *win_kvcache_ptr = win_kvcache.mutable_data_ptr();
  const int *win_block_ids_ptr = win_block_ids.const_data_ptr<int>();
  const int *win_topk_ids_ptr = win_topk_ids.const_data_ptr<int>();

  auto *compress_kvcache_ptr = compress_kvcache.mutable_data_ptr();
  const int *compress_block_ids_ptr = compress_block_ids.const_data_ptr<int>();
  const int *compress_topk_ids_ptr = compress_topk_ids.const_data_ptr<int>();

  const int *cu_seqlens_q_ptr = cu_seqlens_q.const_data_ptr<int>();
  const auto *sink_weight_ptr = sink_weight.const_data_ptr();

  auto options = q.options().dtype(torch::kBFloat16);
  torch::Tensor y;
  if (output.has_value()) {
    y = output.value();
  } else {
    y = torch::empty({total_seq_q, num_head_q, head_dim}, options);
  }
  auto *y_ptr = y.data_ptr();

  int ldQ = q.stride(0);  // num_head_q * num_dim_qk;
  int ldWinKV = win_kvcache.stride(0);
  int ldCompressKV = compress_kvcache.stride(0);
  int ldY = y.stride(0);  // num_head_q * num_dim_v;

  bool running = attention_sparse_mla_with_kvcache_bf16_async(
      y_ptr, q_ptr, win_kvcache_ptr, win_block_ids_ptr, win_topk_ids_ptr, compress_kvcache_ptr,
      compress_block_ids_ptr, compress_topk_ids_ptr, cu_seqlens_q_ptr, sink_weight_ptr,
      softmax_scale, num_batch, total_seq_q, num_head_q, head_dim, num_win_kvcache_blocks,
      num_compress_kvcache_blocks, num_win_seq_max_blocks, num_compress_seq_max_blocks,
      win_block_size, num_win_max_topk, num_compress_max_topk, ldY, ldQ, ldWinKV, ldCompressKV,
      stream);

  TORCH_CHECK(running, "sparse_mla_with_kvcache_bf16 kernel launch failed!");

  return y;
}

torch::Tensor mla_prefill_bf16_entry(const torch::Tensor &q, const torch::Tensor &kv,
                                     const torch::Tensor &seqlens_q,
                                     const torch::Tensor &cu_seqlens_q, int64_t num_dim_qk,
                                     int64_t num_dim_v, int64_t max_seqlens_q,
                                     std::optional<torch::Tensor> output) {
  auto stream = at::cuda::getCurrentCUDAStream(q.get_device());

  TORCH_CHECK(num_dim_qk == 192 && num_dim_v == 128,
              "now mla only support num_dim_qk == 192 and num_dim_v == 128");
  TORCH_CHECK(q.dtype() == torch::kBFloat16, "q dtype must be bfloat16");
  TORCH_CHECK(kv.dtype() == torch::kBFloat16, "kv dtype must be bfloat16");
  TORCH_CHECK(seqlens_q.dtype() == torch::kInt32, "kv dtype must be int32");
  TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "kv dtype must be int32");
  TORCH_CHECK(q.device().is_cuda(), "q tensor must be cuda");
  TORCH_CHECK(kv.device().is_cuda(), "kv tensor must be cuda");
  TORCH_CHECK(seqlens_q.device().is_cuda(), "seqlens_q tensor must be cuda");
  TORCH_CHECK(cu_seqlens_q.device().is_cuda(), "cu_seqlens_q tensor must be cuda");
  TORCH_CHECK(kv.size(-1) == num_dim_qk, "kv.size(-1) == num_dim_qk must be true");
  TORCH_CHECK(num_dim_v <= num_dim_qk, "num_dim_v <= num_dim_qk must be true");

  int total_seq_q = q.size(0);
  int num_head_q = q.size(1);
  int num_head_kv = kv.size(1);

  int num_batch = seqlens_q.size(0);

  auto options = q.options().dtype(torch::kBFloat16);
  torch::Tensor y;
  if (output.has_value()) {
    y = output.value();
  } else {
    y = torch::empty({total_seq_q, num_head_q, num_dim_v}, options);
  }

  int num_tmas = 3 * num_batch;
  torch::Tensor tmas = torch::empty({num_tmas, 64}, options);

  const auto *q_ptr = q.const_data_ptr();
  const auto *kv_ptr = kv.const_data_ptr();
  const auto *seqlens_q_ptr = seqlens_q.const_data_ptr();
  const auto *cu_seqlens_q_ptr = cu_seqlens_q.const_data_ptr();
  void *tmas_ptr = tmas.mutable_data_ptr();

  using T = __nv_bfloat16;
  auto *y_ptr = reinterpret_cast<T *>(y.mutable_data_ptr());

  int ldQ = q.stride(0);    // num_head_q * num_dim_qk;
  int ldKV = kv.stride(0);  // num_head_kv * num_dim_qk;
  int ldY = y.stride(0);    // num_head_q * num_dim_v;

  mla_prefill_bf16_async(y_ptr, q_ptr, kv_ptr, seqlens_q_ptr, cu_seqlens_q_ptr, tmas_ptr, num_batch,
                         total_seq_q, max_seqlens_q, num_dim_qk, num_dim_v, num_head_q, num_head_kv,
                         ldY, ldQ, ldKV, stream);

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
      "attention_with_kvcache_prefill_bf16(Tensor q, Tensor kcache, Tensor vcache,"
      "Tensor cu_seqlens_q, "
      "Tensor block_ids, Tensor num_seq_kvcache, int max_seqlens_q, Tensor? output) -> (Tensor)");
  m.impl("attention_with_kvcache_prefill_bf16", torch::kCUDA,
         &hpc::attention::attention_with_kvcache_prefill_bf16_entry);

  m.def(
      "attention_with_kvcache_prefill_fp8(Tensor q, Tensor kcache, Tensor vcache,"
      "Tensor qkscale, Tensor vscale, Tensor cu_seqlens_q,"
      "Tensor block_ids, Tensor num_seq_kvcache, int max_seqlens_q, Tensor? output) -> (Tensor)");
  m.impl("attention_with_kvcache_prefill_fp8", torch::kCUDA,
         &hpc::attention::attention_with_kvcache_prefill_fp8_entry);

  m.def(
      "attention_decode_bf16(Tensor q, Tensor! kcache, Tensor! vcache, Tensor block_ids, Tensor "
      "num_seq_kvcache, int mtp, bool new_kv_included, bool use_splitk, Tensor? split_flag, "
      "Tensor? output) -> "
      "(Tensor)");
  m.impl("attention_decode_bf16", torch::kCUDA, &hpc::attention::attention_decode_bf16_entry);

  m.def(
      "attention_decode_fp8(Tensor q, Tensor! kcache, Tensor! vcache, Tensor block_ids, Tensor "
      "num_seq_kvcache, Tensor qscale, Tensor kscale, Tensor vscale, int mtp, bool "
      "new_kv_included, bool "
      "use_splitk, Tensor? split_flag, Tensor? output) -> (Tensor)");
  m.impl("attention_decode_fp8", torch::kCUDA, &hpc::attention::attention_decode_fp8_entry);

  m.def(
      "attention_mla_with_kvcache_bf16(Tensor q, Tensor kvcache, Tensor block_ids, Tensor "
      "cu_seqlens_q, Tensor "
      "num_seq_kv, Tensor? output) -> (Tensor)");
  m.impl("attention_mla_with_kvcache_bf16", torch::kCUDA,
         &hpc::attention::attention_mla_with_kvcache_bf16_entry);

  m.def(
      "attention_sparse_mla_with_kvcache_bf16("
      " Tensor q, Tensor win_kvcache, Tensor win_block_ids, Tensor win_topk_ids,"
      " Tensor compress_kvcache, Tensor compress_block_ids, Tensor compress_topk_ids,"
      " Tensor cu_seqlens_q, Tensor sink_weight, float softmax_scale, Tensor? output) -> (Tensor)");
  m.impl("attention_sparse_mla_with_kvcache_bf16", torch::kCUDA,
         &hpc::attention::attention_sparse_mla_with_kvcache_bf16_entry);

  m.def(
      "mla_prefill_bf16(Tensor q, Tensor kv, Tensor seqlens_q, Tensor cu_seqlens_q, "
      "int num_dim_qk, int num_dim_v, int max_seqlens_q, Tensor? output) -> (Tensor)");
  m.impl("mla_prefill_bf16", torch::kCUDA, &hpc::attention::mla_prefill_bf16_entry);
}
