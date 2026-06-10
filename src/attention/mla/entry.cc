// Copyright 2025 hpc-ops authors

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include "src/attention/mla/mla.h"
#include "src/attention/mla/smallm_mla_dim576_persistent.h"
#include "src/attention/mla/smallm_sparse_mla_dim576_persistent.h"
#include "src/utils/utils.h"

namespace hpc {
namespace attention {

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
  int block_size = kvcache.size(1);

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

namespace {

torch::Tensor resolve_dim576_task_tensor(std::optional<torch::Tensor> task_tensor,
                                         int64_t expected_elems, int total_seq_q,
                                         torch::TensorOptions i32_options) {
  if (task_tensor.has_value()) {
    auto t = task_tensor.value();
    TORCH_CHECK(t.device().is_cuda(), "task_tensor must be cuda");
    TORCH_CHECK(t.scalar_type() == torch::kInt32, "task_tensor must be int32");
    TORCH_CHECK(t.is_contiguous(), "task_tensor must be contiguous");
    TORCH_CHECK(t.numel() >= expected_elems, "task_tensor too small: have ", t.numel(), ", need ≥ ",
                expected_elems, " (total_seq_q=", total_seq_q, ")");
    return t;
  }
  return torch::empty({expected_elems}, i32_options);
}

}  // namespace

torch::Tensor mla_decode_with_kvcache_bf16_entry(
    const torch::Tensor &q, const torch::Tensor &kvcache, const torch::Tensor &block_ids,
    const torch::Tensor &cu_seqlens_q, const torch::Tensor &num_seq_kv,
    std::optional<torch::Tensor> sink_weight, double softmax_scale,
    std::optional<torch::Tensor> task_tensor, std::optional<torch::Tensor> output, bool splitk) {
  auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
  TORCH_CHECK(q.device().is_cuda(), "q tensor must be cuda");
  TORCH_CHECK(kvcache.device().is_cuda(), "kvcache tensor must be cuda");
  TORCH_CHECK(block_ids.device().is_cuda(), "block_ids tensor must be cuda");
  TORCH_CHECK(cu_seqlens_q.device().is_cuda(), "cu_seqlens_q tensor must be cuda");
  TORCH_CHECK(num_seq_kv.device().is_cuda(), "num_seq_kv tensor must be cuda");

  int num_batch = num_seq_kv.size(0);
  int total_seq_q = q.size(0);
  int num_head_q = q.size(1);
  int head_dim = q.size(2);

  int num_kvcache_blocks = kvcache.size(0);
  int block_size = kvcache.size(1);

  TORCH_CHECK(block_size == 64, "mla_decode path requires kvcache block_size == 64, got ",
              block_size);
  TORCH_CHECK(head_dim == 576, "mla_decode path requires head_dim == 576, got ", head_dim);
  TORCH_CHECK(head_dim == kvcache.size(2), "q and kvcache must share the same last-dim (", head_dim,
              " vs ", kvcache.size(2), ").");
  TORCH_CHECK(num_head_q >= 1 && num_head_q <= 64,
              "mla_decode path requires 1 <= num_head_q <= 64, got ", num_head_q);
  TORCH_CHECK(total_seq_q >= num_batch,
              "mla_decode path requires total_seq_q >= num_batch; got total_seq_q=", total_seq_q,
              ", num_batch=", num_batch);

  const int v_dim = mla::kDim576VDim;
  int num_seq_max_blocks = block_ids.size(1);

  const auto *q_ptr = q.const_data_ptr();
  auto *kvcache_ptr = kvcache.mutable_data_ptr();
  const int *block_ids_ptr = block_ids.const_data_ptr<int>();
  const int *cu_seqlens_q_ptr = cu_seqlens_q.const_data_ptr<int>();
  const int *num_seq_kv_ptr = num_seq_kv.const_data_ptr<int>();

  auto bf16_options = q.options().dtype(torch::kBFloat16);
  torch::Tensor y;
  if (output.has_value()) {
    y = output.value();
    TORCH_CHECK(y.device().is_cuda(), "output tensor must be cuda");
    TORCH_CHECK(y.scalar_type() == torch::kBFloat16, "output dtype must be bfloat16");
    TORCH_CHECK(y.is_contiguous(), "output tensor must be contiguous");
    TORCH_CHECK(
        y.dim() == 3 && y.size(0) == total_seq_q && y.size(1) == num_head_q && y.size(2) == v_dim,
        "output must have shape [total_seq_q, num_head_q, ", v_dim, "]");
  } else {
    y = torch::empty({total_seq_q, num_head_q, v_dim}, bf16_options);
  }

  int ldQ = q.stride(0);
  int ldKV = kvcache.stride(0);
  int ldY = y.stride(0);

  const float *sink_weight_ptr = nullptr;
  if (sink_weight.has_value()) {
    const auto &sw = sink_weight.value();
    TORCH_CHECK(sw.device().is_cuda(), "sink_weight must be cuda");
    TORCH_CHECK(sw.scalar_type() == torch::kFloat32, "sink_weight must be float32");
    TORCH_CHECK(sw.dim() == 1 && sw.size(0) == num_head_q,
                "sink_weight must have shape [num_head_q]");
    sink_weight_ptr = sw.const_data_ptr<float>();
  }

  auto fp32_options = q.options().dtype(torch::kFloat32);
  auto i32_options = q.options().dtype(torch::kInt32);

  int num_sm = hpc::get_sm_count();
  TORCH_CHECK(num_sm <= mla::kDim576PersistentMaxNumSm,
              "mla_decode_with_kvcache_bf16: runtime num_sm=", num_sm,
              " exceeds kDim576PersistentMaxNumSm=", mla::kDim576PersistentMaxNumSm,
              "; bump the constant in smallm_mla_dim576_persistent.h.");
  bool task_tensor_prebuilt = task_tensor.has_value();
  int64_t expected_task_elems =
      static_cast<int64_t>(mla::dim576_persistent_task_tensor_elems(total_seq_q, num_sm));
  auto resolved_task_tensor =
      resolve_dim576_task_tensor(task_tensor, expected_task_elems, total_seq_q, i32_options);

  auto y_partial = torch::empty({static_cast<int64_t>(mla::dim576_persistent_y_partial_elems(
                                    total_seq_q, num_head_q, v_dim, num_sm))},
                                fp32_options);
  auto lse = torch::empty(
      {static_cast<int64_t>(mla::dim576_persistent_lse_elems(total_seq_q, num_head_q, num_sm))},
      fp32_options);

  bool running = mla::smallm_mla_dim576_persistent_async(
      y.data_ptr(), q_ptr, kvcache_ptr, y_partial.mutable_data_ptr<float>(),
      lse.mutable_data_ptr<float>(), resolved_task_tensor.mutable_data_ptr<int>(), block_ids_ptr,
      cu_seqlens_q_ptr, num_seq_kv_ptr, sink_weight_ptr, num_batch, total_seq_q, num_head_q,
      /*qk_dim=*/head_dim, v_dim, num_kvcache_blocks, num_seq_max_blocks, ldY, ldQ, ldKV,
      static_cast<float>(softmax_scale), stream, task_tensor_prebuilt, splitk);

  TORCH_CHECK(running, "mla_decode_with_kvcache_bf16_entry kernel launch failed!");
  return y;
}

torch::Tensor sparse_mla_dsa_with_kvcache_bf16_entry(
    const torch::Tensor &q, const torch::Tensor &kvcache, const torch::Tensor &block_ids,
    const torch::Tensor &topk_ids, const torch::Tensor &cu_seqlens_q,
    std::optional<torch::Tensor> sink_weight, double softmax_scale,
    std::optional<torch::Tensor> task_tensor, std::optional<torch::Tensor> output, bool splitk) {
  auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
  TORCH_CHECK(q.device().is_cuda(), "q tensor must be cuda");
  TORCH_CHECK(kvcache.device().is_cuda(), "kvcache tensor must be cuda");
  TORCH_CHECK(block_ids.device().is_cuda(), "block_ids tensor must be cuda");
  TORCH_CHECK(topk_ids.device().is_cuda(), "topk_ids tensor must be cuda");
  TORCH_CHECK(cu_seqlens_q.device().is_cuda(), "cu_seqlens_q tensor must be cuda");

  TORCH_CHECK(topk_ids.scalar_type() == torch::kInt32, "topk_ids must be int32");
  TORCH_CHECK(topk_ids.is_contiguous(), "topk_ids must be contiguous");
  TORCH_CHECK(topk_ids.dim() == 2,
              "topk_ids must be 2-D [num_topk_rows, num_max_topk], got dim=", topk_ids.dim());
  TORCH_CHECK(block_ids.scalar_type() == torch::kInt32, "block_ids must be int32");
  TORCH_CHECK(cu_seqlens_q.scalar_type() == torch::kInt32, "cu_seqlens_q must be int32");
  TORCH_CHECK(cu_seqlens_q.dim() == 1, "cu_seqlens_q must be 1-D");

  int num_batch = cu_seqlens_q.size(0) - 1;
  int total_seq_q = q.size(0);
  int num_head_q = q.size(1);
  int head_dim = q.size(2);

  bool prefill = (total_seq_q != num_batch);
  int num_topk_rows = topk_ids.size(0);

  int num_kvcache_blocks = kvcache.size(0);
  int block_size = kvcache.size(1);

  TORCH_CHECK(block_size == 64, "sparse_mla_dsa requires kvcache block_size == 64, got ",
              block_size);
  TORCH_CHECK(head_dim == 576, "sparse_mla_dsa requires head_dim == 576, got ", head_dim);
  TORCH_CHECK(head_dim == kvcache.size(2), "q and kvcache must share the same last-dim (", head_dim,
              " vs ", kvcache.size(2), ").");
  TORCH_CHECK(num_head_q >= 1 && num_head_q <= 64,
              "sparse_mla_dsa requires 1 <= num_head_q <= 64, got ", num_head_q);

  if (!prefill) {
    TORCH_CHECK(num_topk_rows == num_batch, "sparse_mla_dsa decode-mode: topk_ids.size(0) (",
                num_topk_rows, ") must equal num_batch (", num_batch, ")");
  } else {
    TORCH_CHECK(num_topk_rows == total_seq_q, "sparse_mla_dsa prefill-mode: topk_ids.size(0) (",
                num_topk_rows, ") must equal total_seq_q (", total_seq_q, ")");
  }

  int num_max_topk = topk_ids.size(1);
  TORCH_CHECK(num_max_topk > 0 && num_max_topk <= mla::kSparseDim576MaxNumTopk,
              "sparse_mla_dsa requires 0 < num_max_topk <= ", mla::kSparseDim576MaxNumTopk,
              ", got ", num_max_topk);
  TORCH_CHECK(num_max_topk % 64 == 0,
              "sparse_mla_dsa requires num_max_topk to be a multiple of 64 (kTileN), got ",
              num_max_topk);

  const int v_dim = mla::kDim576VDim;
  int num_seq_max_blocks = block_ids.size(1);

  const auto *q_ptr = q.const_data_ptr();
  auto *kvcache_ptr = kvcache.mutable_data_ptr();
  const int *block_ids_ptr = block_ids.const_data_ptr<int>();
  const int *topk_ids_ptr = topk_ids.const_data_ptr<int>();
  const int *cu_seqlens_q_ptr = cu_seqlens_q.const_data_ptr<int>();

  auto bf16_options = q.options().dtype(torch::kBFloat16);
  torch::Tensor y;
  if (output.has_value()) {
    y = output.value();
    TORCH_CHECK(y.device().is_cuda(), "output tensor must be cuda");
    TORCH_CHECK(y.scalar_type() == torch::kBFloat16, "output dtype must be bfloat16");
    TORCH_CHECK(y.is_contiguous(), "output tensor must be contiguous");
    TORCH_CHECK(
        y.dim() == 3 && y.size(0) == total_seq_q && y.size(1) == num_head_q && y.size(2) == v_dim,
        "output must have shape [total_seq_q, num_head_q, ", v_dim, "]");
  } else {
    y = torch::empty({total_seq_q, num_head_q, v_dim}, bf16_options);
  }

  int ldQ = q.stride(0);
  int ldKV = kvcache.stride(0);
  int ldY = y.stride(0);

  const float *sink_weight_ptr = nullptr;
  if (sink_weight.has_value()) {
    const auto &sw = sink_weight.value();
    TORCH_CHECK(sw.device().is_cuda(), "sink_weight must be cuda");
    TORCH_CHECK(sw.scalar_type() == torch::kFloat32, "sink_weight must be float32");
    TORCH_CHECK(sw.dim() == 1 && sw.size(0) == num_head_q,
                "sink_weight must have shape [num_head_q]");
    sink_weight_ptr = sw.const_data_ptr<float>();
  }

  auto fp32_options = q.options().dtype(torch::kFloat32);
  auto i32_options = q.options().dtype(torch::kInt32);

  int num_sm = hpc::get_sm_count();
  TORCH_CHECK(num_sm <= mla::kDim576PersistentMaxNumSm,
              "sparse_mla_dsa_with_kvcache_bf16: runtime num_sm=", num_sm,
              " exceeds kDim576PersistentMaxNumSm=", mla::kDim576PersistentMaxNumSm,
              "; bump the constant in smallm_mla_dim576_persistent.h.");
  bool task_tensor_prebuilt = task_tensor.has_value();
  int64_t expected_task_elems =
      static_cast<int64_t>(mla::dim576_persistent_task_tensor_elems(total_seq_q, num_sm));
  auto resolved_task_tensor =
      resolve_dim576_task_tensor(task_tensor, expected_task_elems, total_seq_q, i32_options);

  auto y_partial = torch::empty({static_cast<int64_t>(mla::dim576_persistent_y_partial_elems(
                                    total_seq_q, num_head_q, v_dim, num_sm))},
                                fp32_options);
  auto lse = torch::empty(
      {static_cast<int64_t>(mla::dim576_persistent_lse_elems(total_seq_q, num_head_q, num_sm))},
      fp32_options);

  bool running = mla::smallm_sparse_mla_dim576_persistent_async(
      y.data_ptr(), q_ptr, kvcache_ptr, y_partial.mutable_data_ptr<float>(),
      lse.mutable_data_ptr<float>(), resolved_task_tensor.mutable_data_ptr<int>(), block_ids_ptr,
      topk_ids_ptr, cu_seqlens_q_ptr, sink_weight_ptr, num_batch, total_seq_q, num_head_q,
      /*qk_dim=*/head_dim, v_dim, num_kvcache_blocks, num_seq_max_blocks, num_max_topk, block_size,
      ldY, ldQ, ldKV, static_cast<float>(softmax_scale), stream, task_tensor_prebuilt, splitk,
      prefill);

  TORCH_CHECK(running, "sparse_mla_dsa_with_kvcache_bf16 kernel launch failed!");
  return y;
}

torch::Tensor get_mla_scheduler_map_entry(const torch::Tensor &num_seq_kv,
                                          const torch::Tensor &cu_seqlens_q,
                                          int64_t num_actual_tokens, int64_t index_topk,
                                          bool splitk, std::optional<torch::Tensor> task_tensor) {
  auto stream = at::cuda::getCurrentCUDAStream(num_seq_kv.get_device());
  TORCH_CHECK(num_seq_kv.device().is_cuda(), "num_seq_kv must be cuda");
  TORCH_CHECK(num_seq_kv.scalar_type() == torch::kInt32, "num_seq_kv must be int32");
  TORCH_CHECK(num_seq_kv.dim() == 1, "num_seq_kv must be 1-D");
  int num_batch = num_seq_kv.size(0);
  TORCH_CHECK(num_actual_tokens >= num_batch, "num_actual_tokens must be >= num_batch");

  int total_seq_q = num_actual_tokens;

  int num_sm = hpc::get_sm_count();
  TORCH_CHECK(num_sm <= mla::kDim576PersistentMaxNumSm,
              "get_mla_scheduler_map: runtime num_sm=", num_sm,
              " exceeds kDim576PersistentMaxNumSm=", mla::kDim576PersistentMaxNumSm,
              "; bump the constant in smallm_mla_dim576_persistent.h.");
  bool sparse = (index_topk > 0);
  // Unified self-describing 8-int task layout for all paths (dense / sparse,
  // decode / prefill). No iquery_to_ibatch_map segment.
  int64_t expected =
      static_cast<int64_t>(mla::dim576_persistent_task_tensor_elems(total_seq_q, num_sm));
  auto i32_options = num_seq_kv.options().dtype(torch::kInt32);
  auto out = resolve_dim576_task_tensor(task_tensor, expected, total_seq_q, i32_options);

  int *base = out.mutable_data_ptr<int>();
  int *task_list = base + mla::dim576_persistent_task_list_offset();
  int *cu_tasks = base + mla::dim576_persistent_cu_tasks_offset(total_seq_q, num_sm);
  int *cu_splits = base + mla::dim576_persistent_cu_splits_offset(total_seq_q, num_sm);

  // Unified decode/prefill: cu_seqlens_q is always required. Decode passes the
  // identity map cu_seqlens_q = [0,1,...,num_batch]; the scheduler resolves
  // ibatch by binary search in both modes (no nullptr decode branch).
  TORCH_CHECK(cu_seqlens_q.device().is_cuda(), "cu_seqlens_q must be cuda");
  TORCH_CHECK(cu_seqlens_q.scalar_type() == torch::kInt32, "cu_seqlens_q must be int32");
  TORCH_CHECK(cu_seqlens_q.dim() == 1, "cu_seqlens_q must be 1-D");
  TORCH_CHECK(cu_seqlens_q.size(0) == num_batch + 1,
              "get_mla_scheduler_map requires cu_seqlens_q of shape [num_batch + 1]; got ",
              cu_seqlens_q.size(0), " for num_batch=", num_batch);
  const int *cu_seqlens_q_ptr = cu_seqlens_q.const_data_ptr<int>();

  if (sparse) {
    bool running = mla::dim576_persistent_get_scheduler_map_sparse_async(
        task_list, cu_tasks, cu_splits, cu_seqlens_q_ptr, num_batch, total_seq_q,
        static_cast<int>(index_topk), num_sm, stream, splitk);
    TORCH_CHECK(running, "get_mla_scheduler_map_entry (sparse) launch failed!");
    return out;
  } else {
    bool running = mla::dim576_persistent_get_scheduler_map_async(
        task_list, cu_tasks, cu_splits, cu_seqlens_q_ptr, num_seq_kv.const_data_ptr<int>(),
        num_batch, total_seq_q, num_sm, stream, splitk);
    TORCH_CHECK(running, "get_mla_scheduler_map_entry launch failed!");
    return out;
  }
}

}  // namespace attention
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
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
      "mla_decode_with_kvcache_bf16(Tensor q, Tensor kvcache, Tensor block_ids, "
      "Tensor cu_seqlens_q, Tensor num_seq_kv, Tensor? sink_weight, float softmax_scale, "
      "Tensor? task_tensor, Tensor? output, bool splitk=True) -> (Tensor)");
  m.impl("mla_decode_with_kvcache_bf16", torch::kCUDA,
         &hpc::attention::mla_decode_with_kvcache_bf16_entry);

  m.def(
      "sparse_mla_dsa_with_kvcache_bf16(Tensor q, Tensor kvcache, Tensor block_ids, "
      "Tensor topk_ids, Tensor cu_seqlens_q, Tensor? sink_weight, "
      "float softmax_scale, Tensor? task_tensor, Tensor? output, bool splitk=True) -> (Tensor)");
  m.impl("sparse_mla_dsa_with_kvcache_bf16", torch::kCUDA,
         &hpc::attention::sparse_mla_dsa_with_kvcache_bf16_entry);

  m.def(
      "get_mla_scheduler_map(Tensor num_seq_kv, Tensor cu_seqlens_q, int num_actual_tokens, "
      "int index_topk=0, bool splitk=True, Tensor? task_tensor=None) -> (Tensor)");
  m.impl("get_mla_scheduler_map", torch::kCUDA, &hpc::attention::get_mla_scheduler_map_entry);
}
