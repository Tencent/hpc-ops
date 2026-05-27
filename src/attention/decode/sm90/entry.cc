// Copyright 2025 hpc-ops authors

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include "src/attention/decode/decode.h"
#include "src/attention/decode/sm90/dynamic/decode_dynamic.h"

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
  TORCH_CHECK((mtp == 0 || mtp == 1 || mtp == 2 || mtp == 3), "we only support mtp 0, 1, 2, 3.");

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

  // Dynamic task_map path (sm90): activated when caller passes a task_map
  // tensor. Works for both quant_type=0 (qkpertoken_perhead_vperhead) and
  // quant_type=1 (qpertoken_perhead_kvpertensor). In this path we split-k up
  // to decode::dynamic::kMaxSplitK chunks at runtime; the bucketing kernel in
  // the dynamic launcher decides how many are actually used per request.
  bool use_dynamic = task_map.has_value();

  bool use_split = splitk > 1 || consumers > 1 || use_dynamic;

  if (use_split) {
    int pad_heads_per_group = ((heads_per_group + 7) / 8) * 8;
    int split_chunks = use_dynamic ? decode::dynamic::kMaxSplitK : splitk * consumers;
    lse = torch::empty({num_batch, split_chunks, num_head_k, num_seq_q, pad_heads_per_group},
                       q.options().dtype(torch::kFloat32));
    split_out = torch::empty({num_batch, split_chunks, num_seq_q, num_head_q, num_dim_v},
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
    if (use_dynamic) {
      int *task_map_ptr = reinterpret_cast<int *>(task_map.value().mutable_data_ptr());
      running = decode::dynamic::smallm_dim128_fp8_qkpertoken_perhead_vperhead_dynamic_async(
          y_ptr, lse_ptr, split_out_ptr, task_map_ptr, q_ptr, kcache_ptr, vcache_ptr, block_ids_ptr,
          num_seq_kvcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr, new_kv_included, num_batch,
          num_seq_q, num_head_q, num_head_k, num_head_v, num_dim_qk, num_dim_v, num_kvcache_blocks,
          block_size, num_seq_max_blocks, qscale_pad_stride, ldY, ldQ, kcache_block_stride,
          kcache_token_stride, kcache_head_stride, vcache_block_stride, vcache_token_stride,
          vcache_head_stride, p_scale_ptr, p_scale_inv_ptr, stream);
    } else {
      running = attention_decode_fp8_qkpertoken_perhead_vperhead_async(
          y_ptr, lse_ptr, split_out_ptr, nullptr, q_ptr, kcache_ptr, vcache_ptr, block_ids_ptr,
          num_seq_kvcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr, split_flag_ptr, new_kv_included,
          splitk, splitk_min_len, consumers, num_batch, num_seq_q, num_head_q, num_head_k,
          num_head_v, num_dim_qk, num_dim_v, num_kvcache_blocks, block_size, num_seq_max_blocks,
          qscale_pad_stride, ldY, ldQ, kcache_block_stride, kcache_token_stride, kcache_head_stride,
          vcache_block_stride, vcache_token_stride, vcache_head_stride, p_scale_ptr,
          p_scale_inv_ptr, stream);
    }
  } else if (quant_type == 1) {
    if (use_dynamic) {
      int *task_map_ptr = reinterpret_cast<int *>(task_map.value().mutable_data_ptr());
      running = decode::dynamic::smallm_dim128_fp8_qpertoken_perhead_kvpertensor_dynamic_async(
          y_ptr, lse_ptr, split_out_ptr, task_map_ptr, q_ptr, kcache_ptr, vcache_ptr, block_ids_ptr,
          num_seq_kvcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr, new_kv_included, num_batch,
          num_seq_q, num_head_q, num_head_k, num_head_v, num_dim_qk, num_dim_v, num_kvcache_blocks,
          block_size, num_seq_max_blocks, qscale_pad_stride, ldY, ldQ, kcache_block_stride,
          kcache_token_stride, kcache_head_stride, vcache_block_stride, vcache_token_stride,
          vcache_head_stride, p_scale_ptr, p_scale_inv_ptr, stream);
    } else {
      running = attention_decode_fp8_qpertoken_perhead_kvpertensor_async(
          y_ptr, lse_ptr, split_out_ptr, nullptr, q_ptr, kcache_ptr, vcache_ptr, block_ids_ptr,
          num_seq_kvcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr, split_flag_ptr, new_kv_included,
          splitk, splitk_min_len, consumers, num_batch, num_seq_q, num_head_q, num_head_k,
          num_head_v, num_dim_qk, num_dim_v, num_kvcache_blocks, block_size, num_seq_max_blocks,
          qscale_pad_stride, ldY, ldQ, kcache_block_stride, kcache_token_stride, kcache_head_stride,
          vcache_block_stride, vcache_token_stride, vcache_head_stride, p_scale_ptr,
          p_scale_inv_ptr, stream);
    }
  }

  TORCH_CHECK(running, "attn decode kernel launch failed!");

  return y;
}

// Populate a task_map buffer for the sm90 dynamic path (kTileN=64). The buffer
// must have been allocated with matching kTileN=64 sizing (see the Python
// helper get_attention_decode_task_workspace_sm90_dynamic).
torch::Tensor assign_attention_decode_task_sm90_dynamic_entry(
    const torch::Tensor &num_seq_kvcache, int64_t num_head_kv, int64_t num_seq_q,
    bool new_kv_included, int64_t min_process_len, std::optional<torch::Tensor> task_map) {
  TORCH_CHECK(num_seq_kvcache.device().is_cuda(), "num_seq_kvcache tensor must be cuda");
  TORCH_CHECK(task_map.has_value(),
              "assign_attention_decode_task_sm90_dynamic (CUDA) must be given a task_map output");
  auto task_map_tensor = task_map.value();
  TORCH_CHECK(task_map_tensor.device().is_cuda(), "task_map tensor must be cuda");

  int num_batch = num_seq_kvcache.size(0);
  auto stream = at::cuda::getCurrentCUDAStream(num_seq_kvcache.get_device());

  constexpr int kMaxNumBatch = 2048;
  TORCH_CHECK(num_batch <= kMaxNumBatch,
              "assign_attention_decode_task_sm90_dynamic only supports batch_size <= 2048");

  const int *num_seq_kvcache_ptr = num_seq_kvcache.const_data_ptr<int>();
  int *task_map_ptr = reinterpret_cast<int *>(task_map_tensor.mutable_data_ptr());

  auto running = decode::dynamic::assign_attention_decode_task_sm90_dynamic_async(
      task_map_ptr, num_seq_kvcache_ptr, num_batch, num_head_kv, num_seq_q, new_kv_included,
      min_process_len, stream);

  TORCH_CHECK(running, "launch assign_attention_decode_task_sm90_dynamic_async failed");

  return task_map_tensor;
}

// CPU-side twin of assign_attention_decode_task_sm90_dynamic_entry. Returns a
// newly-allocated int8 tensor on CPU with the same task_map layout as the CUDA
// path (kTileN=64, SM90DynamicTaskInfo-sized tasks = 48B each). The optional
// `task_map` arg is ignored on CPU (schema symmetry with the CUDA variant).
// Used by tests that allclose the CUDA task_map against a host reference.
torch::Tensor assign_attention_decode_task_sm90_dynamic_cpu_entry(
    const torch::Tensor &num_seq_kvcache, int64_t num_head_kv, int64_t num_seq_q,
    bool new_kv_included, int64_t min_process_len, std::optional<torch::Tensor> /*placehold*/) {
  TORCH_CHECK(num_seq_kvcache.device().is_cpu(), "num_seq_kvcache tensor must be cpu");
  int num_batch = num_seq_kvcache.size(0);

  const int *num_seq_kvcache_ptr = num_seq_kvcache.const_data_ptr<int>();

  auto tasks_pair = decode::dynamic::assign_attention_decode_task_sm90_dynamic_sync(
      num_seq_kvcache_ptr, num_batch, num_head_kv, num_seq_q, new_kv_included, min_process_len);
  auto tasks = tasks_pair.first;
  auto num_chunks = tasks_pair.second;
  // num_chunks layout: [num_head_kv * num_batch] per-(head, batch) entries,
  // then a trailing slot that stores num_tile_per_cta+1 (CPU-entry contract).
  int num_tile_per_cta = num_chunks[num_head_kv * num_batch];

  constexpr int kTaskInfoSize = sizeof(decode::dynamic::SM90DynamicTaskInfo);  // 48
  int num_task = tasks.size();
  auto options = num_seq_kvcache.options().dtype(torch::kInt8);

  // Task_map byte layout on CPU mirror:
  //   header (kTaskInfoSize B, first 4 = num_tile_per_cta+1)
  // + per-CTA tasks (num_task × kTaskInfoSize B)
  // + num_chunks region for combine (ceil(num_head_kv*num_batch*sizeof(int) / kTaskInfoSize)
  //                                   blocks of kTaskInfoSize)
  int num_chunks_bytes = num_head_kv * num_batch * sizeof(int);
  int task_map_shape0 = 1 + num_task + (num_chunks_bytes + kTaskInfoSize - 1) / kTaskInfoSize;
  auto task_map = torch::zeros({task_map_shape0, kTaskInfoSize}, options);
  uint8_t *task_map_ptr = reinterpret_cast<uint8_t *>(task_map.mutable_data_ptr());

  memcpy(task_map_ptr, &num_tile_per_cta, sizeof(int));
  memcpy(task_map_ptr + kTaskInfoSize, tasks.data(), kTaskInfoSize * num_task);
  memcpy(task_map_ptr + kTaskInfoSize * (num_task + 1), num_chunks.data(), num_chunks_bytes);

  return task_map;
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

  m.def(
      "assign_attention_decode_task_sm90_dynamic(Tensor num_seq_kvcache, int num_head_kv, int mtp, "
      "bool new_kv_included, int min_process_len, Tensor? task_map) -> (Tensor)");
  m.impl("assign_attention_decode_task_sm90_dynamic", torch::kCUDA,
         &hpc::attention::assign_attention_decode_task_sm90_dynamic_entry);
  m.impl("assign_attention_decode_task_sm90_dynamic", torch::kCPU,
         &hpc::attention::assign_attention_decode_task_sm90_dynamic_cpu_entry);
}
