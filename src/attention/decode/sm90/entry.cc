// Copyright 2025 hpc-ops authors

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include "src/attention/decode/decode.h"

namespace hpc {
namespace attention {

torch::Tensor attention_decode_fp8_entry(
    const torch::Tensor &q, torch::Tensor &kcache, torch::Tensor &vcache,
    const torch::Tensor &block_ids, const torch::Tensor &num_seq_kvcache,
    const torch::Tensor &qscale, const torch::Tensor &kscale, const torch::Tensor &vscale,
    int64_t mtp, bool new_kv_included, int64_t quant_type, bool use_splitk,
    std::optional<torch::Tensor> task_map, std::optional<torch::Tensor> split_flag,
    std::optional<torch::Tensor> p_scale, std::optional<torch::Tensor> p_scale_inv,
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
  const float *kscale_ptr = reinterpret_cast<const float *>(kscale.data_ptr());
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
  int kcache_block_stride = kcache.stride(0);
  int vcache_block_stride = vcache.stride(0);

  int kcache_token_stride = kcache.stride(1);
  int vcache_token_stride = vcache.stride(1);

  int kcache_head_stride = kcache.stride(2);
  int vcache_head_stride = vcache.stride(2);

  int ldY = y.stride(0);  // num_head_q * num_dim_v;

  const float *p_scale_ptr = nullptr;
  const float *p_scale_inv_ptr = nullptr;
  if (p_scale.has_value() || p_scale_inv.has_value()) {
    TORCH_CHECK(p_scale.has_value() && p_scale_inv.has_value(),
                "p_scale and p_scale_inv must be provided together");
    const auto &ps = p_scale.value();
    const auto &psi = p_scale_inv.value();
    TORCH_CHECK(ps.device() == q.device() && psi.device() == q.device(),
                "p_scale/p_scale_inv must share q's device");
    TORCH_CHECK(ps.scalar_type() == torch::kFloat32 && psi.scalar_type() == torch::kFloat32,
                "p_scale/p_scale_inv dtype must be float32");
    TORCH_CHECK(ps.is_contiguous() && psi.is_contiguous(),
                "p_scale/p_scale_inv must be contiguous");
    TORCH_CHECK(ps.numel() == num_head_q && psi.numel() == num_head_q,
                "p_scale/p_scale_inv must have shape [num_head_q=", num_head_q, "]");
    p_scale_ptr = ps.const_data_ptr<float>();
    p_scale_inv_ptr = psi.const_data_ptr<float>();
  }

  bool running = false;
  if (quant_type == 0) {
    running = attention_decode_fp8_qkpertoken_perhead_vperhead_async(
        y_ptr, lse_ptr, split_out_ptr, nullptr, q_ptr, kcache_ptr, vcache_ptr, block_ids_ptr,
        num_seq_kvcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr, split_flag_ptr, new_kv_included,
        splitk, splitk_min_len, consumers, num_batch, num_seq_q, num_head_q, num_head_k, num_head_v,
        num_dim_qk, num_dim_v, num_kvcache_blocks, block_size, num_seq_max_blocks,
        qscale_pad_stride, ldY, ldQ, kcache_block_stride, kcache_token_stride, kcache_head_stride,
        vcache_block_stride, vcache_token_stride, vcache_head_stride, p_scale_ptr, p_scale_inv_ptr,
        stream);
  } else if (quant_type == 1) {
    running = attention_decode_fp8_qpertoken_perhead_kvpertensor_async(
        y_ptr, lse_ptr, split_out_ptr, nullptr, q_ptr, kcache_ptr, vcache_ptr, block_ids_ptr,
        num_seq_kvcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr, split_flag_ptr, new_kv_included,
        splitk, splitk_min_len, consumers, num_batch, num_seq_q, num_head_q, num_head_k, num_head_v,
        num_dim_qk, num_dim_v, num_kvcache_blocks, block_size, num_seq_max_blocks,
        qscale_pad_stride, ldY, ldQ, kcache_block_stride, kcache_token_stride, kcache_head_stride,
        vcache_block_stride, vcache_token_stride, vcache_head_stride, p_scale_ptr, p_scale_inv_ptr,
        stream);
  }

  TORCH_CHECK(running, "attn decode kernel launch failed!");

  return y;
}

}  // namespace attention
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "attention_decode_fp8(Tensor q, Tensor! kcache, Tensor! vcache, Tensor block_ids, Tensor "
      "num_seq_kvcache, Tensor qscale, Tensor kscale, Tensor vscale, int mtp, bool "
      "new_kv_included, int quant_type, bool "
      "use_splitk, Tensor? task_map, Tensor? split_flag, Tensor? p_scale, Tensor? p_scale_inv, "
      "Tensor? output) -> (Tensor)");
  m.impl("attention_decode_fp8", torch::kCUDA, &hpc::attention::attention_decode_fp8_entry);
}
