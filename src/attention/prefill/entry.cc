// Copyright 2025 hpc-ops authors

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

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
    TORCH_CHECK(y.device().is_cuda(), "output tensor must be cuda");
    TORCH_CHECK(y.get_device() == q.get_device(), "output tensor must be on the same device as q");
    TORCH_CHECK(y.scalar_type() == torch::kBFloat16, "output dtype must be bfloat16");
    TORCH_CHECK(y.is_contiguous(), "output tensor must be contiguous");
    TORCH_CHECK(y.dim() == 3 && y.size(0) == total_seq_q && y.size(1) == num_head_q &&
                    y.size(2) == num_dim_v,
                "output must have shape [total_seq_q, num_head_q, num_dim_v]");
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
  TORCH_CHECK(num_dim_qk == 128 && num_dim_v == 128,
              "attention_with_kvcache_prefill_bf16: expected dim_qk=128 and dim_v=128, got dim_qk=",
              num_dim_qk, " dim_v=", num_dim_v);

  int num_seq_max_blocks = block_ids.size(1);

  auto options = q.options().dtype(torch::kBFloat16);
  torch::Tensor y;
  if (output.has_value()) {
    y = output.value();
    TORCH_CHECK(y.device().is_cuda(), "output tensor must be cuda");
    TORCH_CHECK(y.get_device() == q.get_device(), "output tensor must be on the same device as q");
    TORCH_CHECK(y.scalar_type() == torch::kBFloat16, "output dtype must be bfloat16");
    TORCH_CHECK(y.is_contiguous(), "output tensor must be contiguous");
    TORCH_CHECK(y.dim() == 3 && y.size(0) == total_seq_q && y.size(1) == num_head_q &&
                    y.size(2) == num_dim_v,
                "output must have shape [total_seq_q, num_head_q, num_dim_v]");
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

  int ldQ = q.stride(0);  // num_head_q * num_dim_qk;
  int ldK = kcache.stride(0);
  int ldK1 = kcache.stride(1);
  int ldK2 = kcache.stride(2);
  int ldV = vcache.stride(0);
  int ldV1 = vcache.stride(1);
  int ldV2 = vcache.stride(2);
  int ldY = y.stride(0);  // num_head_q * num_dim_v;

  attention_with_kvcache_prefill_bf16_async(
      y_ptr, q_ptr, kcache_ptr, vcache_ptr, cu_seqlens_q_ptr, block_ids_ptr, seqlens_kvcache_ptr,
      tmas_ptr, num_batch, total_seq_q, max_seqlens_q, num_dim_qk, num_dim_v, num_head_q,
      num_head_kv, num_kvcache_blocks, block_size, num_seq_max_blocks, ldY, ldQ, ldK, ldK1, ldK2,
      ldV, ldV1, ldV2, stream);

  return y;
}

torch::Tensor attention_with_kvcache_prefill_fp8_entry(
    const torch::Tensor &q, const torch::Tensor &kcache, const torch::Tensor &vcache,
    const torch::Tensor &qscale, const torch::Tensor &kscale, const torch::Tensor &vscale,
    const torch::Tensor &cu_seqlens_q, const torch::Tensor block_ids,
    const torch::Tensor seqlens_kvcache, int64_t max_seqlens_q, int64_t quant_type,
    std::optional<torch::Tensor> output) {
  auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
  TORCH_CHECK(q.device().is_cuda(), "q tensor must be cuda");
  TORCH_CHECK(kcache.device().is_cuda(), "kcache tensor must be cuda");
  TORCH_CHECK(vcache.device().is_cuda(), "vcache tensor must be cuda");
  TORCH_CHECK(qscale.device().is_cuda(), "qscale tensor must be cuda");
  TORCH_CHECK(kscale.device().is_cuda(), "kscale tensor must be cuda");
  TORCH_CHECK(vscale.device().is_cuda(), "vscale tensor must be cuda");
  TORCH_CHECK(cu_seqlens_q.device().is_cuda(), "cu_seqlens_q tensor must be cuda");
  TORCH_CHECK(block_ids.device().is_cuda(), "block_ids tensor must be cuda");
  TORCH_CHECK(seqlens_kvcache.device().is_cuda(), "seqlens_kvcache tensor must be cuda");
  TORCH_CHECK((quant_type == 0 || quant_type == 1), "quant_type only support 0/1");
  TORCH_CHECK((kscale.dtype().itemsize() == 4 || kscale.dtype().itemsize() == 1),
              "kscale dtype must be float or fp8");

  int total_seq_q = q.size(0);
  int num_head_q = q.size(1);
  int num_dim_qk = q.size(2);

  int num_batch = cu_seqlens_q.size(0) - 1;

  int num_kvcache_blocks = kcache.size(0);
  int block_size = kcache.size(1);

  int num_head_kv = kcache.size(2);
  int num_dim_v = vcache.size(3);
  TORCH_CHECK(num_dim_qk == 128 && num_dim_v == 128,
              "attention_with_kvcache_prefill_fp8: expected dim_qk=128 and dim_v=128, got dim_qk=",
              num_dim_qk, " dim_v=", num_dim_v);

  int num_seq_max_blocks = block_ids.size(1);

  int max_seqlens_q_pad = qscale.size(2);

  auto options = q.options().dtype(torch::kBFloat16);
  torch::Tensor y;
  if (output.has_value()) {
    y = output.value();
    TORCH_CHECK(y.device().is_cuda(), "output tensor must be cuda");
    TORCH_CHECK(y.get_device() == q.get_device(), "output tensor must be on the same device as q");
    TORCH_CHECK(y.scalar_type() == torch::kBFloat16, "output dtype must be bfloat16");
    TORCH_CHECK(y.is_contiguous(), "output tensor must be contiguous");
    TORCH_CHECK(y.dim() == 3 && y.size(0) == total_seq_q && y.size(1) == num_head_q &&
                    y.size(2) == num_dim_v,
                "output must have shape [total_seq_q, num_head_q, num_dim_v]");
  } else {
    y = torch::empty({total_seq_q, num_head_q, num_dim_v}, options);
  }

  int num_tmas = 2 * num_batch;
  torch::Tensor tmas = torch::empty({num_tmas, 64}, options);

  const auto *q_ptr = q.const_data_ptr();
  const auto *kcache_ptr = kcache.const_data_ptr();
  const auto *vcache_ptr = vcache.const_data_ptr();
  const auto *qscale_ptr = qscale.const_data_ptr();
  const auto *kscale_ptr = kscale.const_data_ptr();
  const auto *vscale_ptr = vscale.const_data_ptr();
  const auto *cu_seqlens_q_ptr = cu_seqlens_q.const_data_ptr();
  const auto *block_ids_ptr = block_ids.const_data_ptr();
  const auto *seqlens_kvcache_ptr = seqlens_kvcache.const_data_ptr();
  void *tmas_ptr = tmas.mutable_data_ptr();

  using T = __nv_bfloat16;
  auto *y_ptr = reinterpret_cast<T *>(y.mutable_data_ptr());

  int ldQ = q.stride(0);
  int ldK = kcache.stride(0);
  int ldK1 = kcache.stride(1);
  int ldK2 = kcache.stride(2);
  int ldV = vcache.stride(0);
  int ldV1 = vcache.stride(1);
  int ldV2 = vcache.stride(2);
  int ldY = y.stride(0);

  if (quant_type == 1) {
    attention_with_kvcache_prefill_qpertoken_perhead_kvpertensor_fp8_async(
        y_ptr, q_ptr, kcache_ptr, vcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr, cu_seqlens_q_ptr,
        block_ids_ptr, seqlens_kvcache_ptr, tmas_ptr, num_batch, total_seq_q, max_seqlens_q,
        max_seqlens_q_pad, num_dim_qk, num_dim_v, num_head_q, num_head_kv, num_kvcache_blocks,
        block_size, num_seq_max_blocks, ldY, ldQ, ldK, ldK1, ldK2, ldV, ldV1, ldV2, stream);
  } else if (quant_type == 0) {
    int ldKS = 0;
    int ldKS1 = 0;
    int ldKS2 = 0;
    if (kscale.dtype().itemsize() == 4) {
      ldKS = kscale.stride(0);
      ldKS1 = kscale.stride(1);
      ldKS2 = kscale.stride(2);
    } else if (kscale.dtype().itemsize() == 1) {
      ldKS = kscale.stride(0) / sizeof(float);
      ldKS1 = kscale.stride(1) / sizeof(float);
      ldKS2 = kscale.stride(2) / sizeof(float);
    }
    int scale_block_size = kscale.size(1);
    attention_with_kvcache_prefill_qkpertoken_perhead_vperhead_fp8_async(
        y_ptr, q_ptr, kcache_ptr, vcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr, cu_seqlens_q_ptr,
        block_ids_ptr, seqlens_kvcache_ptr, tmas_ptr, num_batch, total_seq_q, max_seqlens_q,
        max_seqlens_q_pad, num_dim_qk, num_dim_v, num_head_q, num_head_kv, num_kvcache_blocks,
        block_size, scale_block_size, num_seq_max_blocks, ldY, ldQ, ldK, ldK1, ldK2, ldV, ldV1,
        ldV2, ldKS, ldKS1, ldKS2, stream);
  }

  return y;
}

torch::Tensor attention_with_kvcache_blocksparse_prefill_fp8_entry(
    const torch::Tensor &q, const torch::Tensor &kcache, const torch::Tensor &vcache,
    const torch::Tensor &qscale, const torch::Tensor &kscale, const torch::Tensor &vscale,
    const torch::Tensor &cu_seqlens_q, const torch::Tensor block_ids,
    const torch::Tensor seqlens_kvcache, int64_t max_seqlens_q, int64_t quant_type,
    std::optional<torch::Tensor> block_mask, std::optional<torch::Tensor> output) {
  auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
  TORCH_CHECK(q.device().is_cuda(), "q tensor must be cuda");
  TORCH_CHECK(kcache.device().is_cuda(), "kcache tensor must be cuda");
  TORCH_CHECK(vcache.device().is_cuda(), "vcache tensor must be cuda");
  TORCH_CHECK(qscale.device().is_cuda(), "qscale tensor must be cuda");
  TORCH_CHECK(kscale.device().is_cuda(), "kscale tensor must be cuda");
  TORCH_CHECK(vscale.device().is_cuda(), "vscale tensor must be cuda");
  TORCH_CHECK(cu_seqlens_q.device().is_cuda(), "cu_seqlens_q tensor must be cuda");
  TORCH_CHECK(block_ids.device().is_cuda(), "block_ids tensor must be cuda");
  TORCH_CHECK(seqlens_kvcache.device().is_cuda(), "seqlens_kvcache tensor must be cuda");
  TORCH_CHECK((quant_type == 0 || quant_type == 1), "quant_type only support 0/1");
  TORCH_CHECK((kscale.dtype().itemsize() == 4 || kscale.dtype().itemsize() == 1),
              "kscale dtype must be float or fp8");

  bool has_block_mask = block_mask.has_value();

  int total_seq_q = q.size(0);
  int num_head_q = q.size(1);
  int num_dim_qk = q.size(2);

  int num_batch = cu_seqlens_q.size(0) - 1;

  int num_kvcache_blocks = kcache.size(0);
  int block_size = kcache.size(1);

  int num_head_kv = kcache.size(2);
  int num_dim_v = vcache.size(3);
  TORCH_CHECK(
      num_dim_qk == 128 && num_dim_v == 128,
      "attention_with_kvcache_blocksparse_prefill_fp8: expected dim_qk=128 and dim_v=128, got "
      "dim_qk=",
      num_dim_qk, " dim_v=", num_dim_v);

  int num_seq_max_blocks = block_ids.size(1);

  int max_seqlens_q_pad = qscale.size(2);

  constexpr int kTileM = 128;
  constexpr int kTileN = 128;
  TORCH_CHECK(kTileN % block_size == 0, "unsupported block_size for FP8 blocksparse prefill");
  int expected_max_num_tile_m = (max_seqlens_q + kTileM - 1) / kTileM;

  int num_tile_kv_in_mask = 0;
  if (has_block_mask) {
    const auto &block_mask_tensor = block_mask.value();
    TORCH_CHECK(block_mask_tensor.device() == q.device(),
                "block_mask tensor must be on the same device as q");
    TORCH_CHECK(block_mask_tensor.scalar_type() == torch::kUInt8, "block_mask dtype must be uint8");
    TORCH_CHECK(block_mask_tensor.is_contiguous(), "block_mask tensor must be contiguous");
    TORCH_CHECK(block_mask_tensor.dim() == 4 && block_mask_tensor.size(0) == num_batch &&
                    block_mask_tensor.size(1) == num_head_q &&
                    block_mask_tensor.size(2) == expected_max_num_tile_m,
                "block_mask must have shape [", num_batch, ", ", num_head_q, ", ",
                expected_max_num_tile_m, ", Kb] where Kb = ceil(max_kv_len / kTileN=", kTileN, ")");
    num_tile_kv_in_mask = block_mask_tensor.size(3);

    TORCH_CHECK(num_tile_kv_in_mask > 0, "block_mask Kb dim must be > 0");
  }

  auto options = q.options().dtype(torch::kBFloat16);
  torch::Tensor y;
  if (output.has_value()) {
    y = output.value();
    TORCH_CHECK(y.device().is_cuda(), "output tensor must be cuda");
    TORCH_CHECK(y.get_device() == q.get_device(), "output tensor must be on the same device as q");
    TORCH_CHECK(y.scalar_type() == torch::kBFloat16, "output dtype must be bfloat16");
    TORCH_CHECK(y.is_contiguous(), "output tensor must be contiguous");
    TORCH_CHECK(y.dim() == 3 && y.size(0) == total_seq_q && y.size(1) == num_head_q &&
                    y.size(2) == num_dim_v,
                "output must have shape [total_seq_q, num_head_q, num_dim_v]");
  } else {
    y = torch::empty({total_seq_q, num_head_q, num_dim_v}, options);
  }

  int num_tmas = 2 * num_batch;
  torch::Tensor tmas = torch::empty({num_tmas, 64}, options);

  const auto *q_ptr = q.const_data_ptr();
  const auto *kcache_ptr = kcache.const_data_ptr();
  const auto *vcache_ptr = vcache.const_data_ptr();
  const auto *qscale_ptr = qscale.const_data_ptr();
  const auto *kscale_ptr = kscale.const_data_ptr();
  const auto *vscale_ptr = vscale.const_data_ptr();
  const auto *cu_seqlens_q_ptr = cu_seqlens_q.const_data_ptr();
  const auto *block_ids_ptr = block_ids.const_data_ptr();
  const auto *seqlens_kvcache_ptr = seqlens_kvcache.const_data_ptr();
  const void *block_mask_ptr = nullptr;
  if (has_block_mask) {
    block_mask_ptr = block_mask.value().const_data_ptr();
  }
  void *tmas_ptr = tmas.mutable_data_ptr();

  using T = __nv_bfloat16;
  auto *y_ptr = reinterpret_cast<T *>(y.mutable_data_ptr());

  int ldQ = q.stride(0);  // num_head_q * num_dim_qk;
  int ldK = kcache.stride(0);
  int ldK1 = kcache.stride(1);
  int ldK2 = kcache.stride(2);
  int ldV = vcache.stride(0);
  int ldV1 = vcache.stride(1);
  int ldV2 = vcache.stride(2);
  int ldY = y.stride(0);  // num_head_q * num_dim_v;

  if (quant_type == 1) {
    attention_with_kvcache_blocksparse_prefill_qpertoken_perhead_kvpertensor_fp8_async(
        y_ptr, q_ptr, kcache_ptr, vcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr, cu_seqlens_q_ptr,
        block_ids_ptr, seqlens_kvcache_ptr, tmas_ptr, num_batch, total_seq_q, max_seqlens_q,
        max_seqlens_q_pad, num_dim_qk, num_dim_v, num_head_q, num_head_kv, num_kvcache_blocks,
        block_size, num_seq_max_blocks, ldY, ldQ, ldK, ldK1, ldK2, ldV, ldV1, ldV2, block_mask_ptr,
        num_tile_kv_in_mask, stream);
  } else if (quant_type == 0) {
    int ldKS = 0;
    int ldKS1 = 0;
    int ldKS2 = 0;
    if (kscale.dtype().itemsize() == 4) {
      ldKS = kscale.stride(0);
      ldKS1 = kscale.stride(1);
      ldKS2 = kscale.stride(2);
    } else if (kscale.dtype().itemsize() == 1) {
      ldKS = kscale.stride(0) / sizeof(float);
      ldKS1 = kscale.stride(1) / sizeof(float);
      ldKS2 = kscale.stride(2) / sizeof(float);
    }
    int scale_block_size = kscale.size(1);
    attention_with_kvcache_blocksparse_prefill_qkpertoken_perhead_vperhead_fp8_async(
        y_ptr, q_ptr, kcache_ptr, vcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr, cu_seqlens_q_ptr,
        block_ids_ptr, seqlens_kvcache_ptr, tmas_ptr, num_batch, total_seq_q, max_seqlens_q,
        max_seqlens_q_pad, num_dim_qk, num_dim_v, num_head_q, num_head_kv, num_kvcache_blocks,
        block_size, scale_block_size, num_seq_max_blocks, ldY, ldQ, ldK, ldK1, ldK2, ldV, ldV1,
        ldV2, ldKS, ldKS1, ldKS2, block_mask_ptr, num_tile_kv_in_mask, stream);
  }

  return y;
}

torch::Tensor attention_blocksparse_prefill_fp8_dim192_entry(
    const torch::Tensor &q, const torch::Tensor &k, const torch::Tensor &v,
    const torch::Tensor &cu_seqlens_q, const torch::Tensor &cu_seqlens_kv, int64_t max_seqlens_q,
    int64_t max_seqlens_kv, std::optional<torch::Tensor> block_mask, const torch::Tensor &q_scale,
    const torch::Tensor &k_scale, const torch::Tensor &v_scale, double softmax_scale,
    std::optional<torch::Tensor> output) {
  auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
  TORCH_CHECK(q.device().is_cuda(), "q tensor must be cuda");
  TORCH_CHECK(k.device().is_cuda(), "k tensor must be cuda");
  TORCH_CHECK(v.device().is_cuda(), "v tensor must be cuda");
  TORCH_CHECK(cu_seqlens_q.device().is_cuda(), "cu_seqlens_q tensor must be cuda");
  TORCH_CHECK(cu_seqlens_kv.device().is_cuda(), "cu_seqlens_kv tensor must be cuda");
  TORCH_CHECK(q.dim() == 3, "q tensor must be 3D [total_q, num_head_q, dim_qk]");
  TORCH_CHECK(k.dim() == 3, "k tensor must be 3D [total_kv, num_head_kv, dim_qk]");
  TORCH_CHECK(v.dim() == 3, "v tensor must be 3D [total_kv, num_head_kv, dim_v]");
  TORCH_CHECK(q.is_contiguous(), "q tensor must be contiguous");
  TORCH_CHECK(k.is_contiguous(), "k tensor must be contiguous");
  TORCH_CHECK(v.is_contiguous(), "v tensor must be contiguous");
  TORCH_CHECK(q.size(2) == k.size(2), "q and k must have same dim_qk, got ", q.size(2), " vs ",
              k.size(2));
  TORCH_CHECK(q.scalar_type() == torch::kFloat8_e4m3fn, "q dtype must be float8_e4m3fn");
  TORCH_CHECK(k.scalar_type() == torch::kFloat8_e4m3fn, "k dtype must be float8_e4m3fn");
  TORCH_CHECK(v.scalar_type() == torch::kFloat8_e4m3fn, "v dtype must be float8_e4m3fn");
  TORCH_CHECK(q_scale.numel() == 1, "q_scale must be a scalar tensor with numel()==1, got ",
              q_scale.numel());
  TORCH_CHECK(k_scale.numel() == 1, "k_scale must be a scalar tensor with numel()==1, got ",
              k_scale.numel());
  TORCH_CHECK(v_scale.numel() == 1, "v_scale must be a scalar tensor with numel()==1, got ",
              v_scale.numel());

  float softmax_qkscale =
      static_cast<float>(softmax_scale) * q_scale.item<float>() * k_scale.item<float>();
  float vscale = v_scale.item<float>();

  bool has_block_mask = block_mask.has_value();

  int total_seq_q = q.size(0);
  int num_head_q = q.size(1);
  int num_dim_qk = q.size(2);

  int num_head_kv = k.size(1);
  int num_dim_v = v.size(2);
  TORCH_CHECK(
      num_dim_qk == 192 && num_dim_v == 128,
      "attention_blocksparse_prefill_fp8_dim192: expected dim_qk=192 and dim_v=128, got dim_qk=",
      num_dim_qk, " dim_v=", num_dim_v);

  int num_batch = cu_seqlens_q.size(0) - 1;

  constexpr int kTileM = 128;
  constexpr int kTileN = 128;
  int expected_max_num_tile_m = (max_seqlens_q + kTileM - 1) / kTileM;
  int expected_min_num_tile_n = (max_seqlens_kv + kTileN - 1) / kTileN;

  int num_tile_kv_in_mask = 0;
  if (has_block_mask) {
    const auto &block_mask_tensor = block_mask.value();
    TORCH_CHECK(block_mask_tensor.device() == q.device(),
                "block_mask tensor must be on the same device as q");
    TORCH_CHECK(block_mask_tensor.scalar_type() == torch::kUInt8, "block_mask dtype must be uint8");
    TORCH_CHECK(block_mask_tensor.is_contiguous(), "block_mask tensor must be contiguous");
    TORCH_CHECK(block_mask_tensor.dim() == 4 && block_mask_tensor.size(0) == num_batch &&
                    block_mask_tensor.size(1) == num_head_q &&
                    block_mask_tensor.size(2) == expected_max_num_tile_m,
                "block_mask must have shape [", num_batch, ", ", num_head_q, ", ",
                expected_max_num_tile_m, ", Kb] where Kb >= ceil(max_seqlens_kv / kTileN=", kTileN,
                ")");
    num_tile_kv_in_mask = block_mask_tensor.size(3);
    TORCH_CHECK(num_tile_kv_in_mask >= expected_min_num_tile_n, "block_mask Kb dim (",
                num_tile_kv_in_mask, ") must be >= ceil(max_seqlens_kv=", max_seqlens_kv,
                " / kTileN=", kTileN, ") = ", expected_min_num_tile_n,
                "; otherwise BSA tail-padding (kernels.cuh:4071-4076) silently drops K-tiles");
  }

  auto options = q.options().dtype(torch::kBFloat16);
  torch::Tensor y;
  if (output.has_value()) {
    y = output.value();
    TORCH_CHECK(y.device().is_cuda(), "output tensor must be cuda");
    TORCH_CHECK(y.get_device() == q.get_device(), "output tensor must be on the same device as q");
    TORCH_CHECK(y.scalar_type() == torch::kBFloat16, "output dtype must be bfloat16");
    TORCH_CHECK(y.is_contiguous(), "output tensor must be contiguous");
    TORCH_CHECK(y.dim() == 3 && y.size(0) == total_seq_q && y.size(1) == num_head_q &&
                    y.size(2) == num_dim_v,
                "output must have shape [total_seq_q, num_head_q, num_dim_v]");
  } else {
    y = torch::empty({total_seq_q, num_head_q, num_dim_v}, options);
  }

  int num_tmas = 4 * num_batch;
  torch::Tensor tmas = torch::empty({num_tmas, 64}, options);

  const auto *q_ptr = q.const_data_ptr();
  const auto *k_ptr = k.const_data_ptr();
  const auto *v_ptr = v.const_data_ptr();
  const auto *cu_seqlens_q_ptr = cu_seqlens_q.const_data_ptr();
  const auto *cu_seqlens_kv_ptr = cu_seqlens_kv.const_data_ptr();
  const void *block_mask_ptr = nullptr;
  if (has_block_mask) {
    block_mask_ptr = block_mask.value().const_data_ptr();
  }
  void *tmas_ptr = tmas.mutable_data_ptr();

  using T = __nv_bfloat16;
  auto *y_ptr = reinterpret_cast<T *>(y.mutable_data_ptr());

  int ldQ = q.stride(0);
  int ldK = k.stride(0);
  int ldV = v.stride(0);
  int ldY = y.stride(0);

  attention_blocksparse_prefill_fp8_dim192_async(
      y_ptr, q_ptr, k_ptr, v_ptr, cu_seqlens_q_ptr, cu_seqlens_kv_ptr, tmas_ptr, num_batch,
      total_seq_q, max_seqlens_q, max_seqlens_kv, num_dim_qk, num_dim_v, num_head_q, num_head_kv,
      ldY, ldQ, ldK, ldV, block_mask_ptr, num_tile_kv_in_mask, softmax_qkscale, vscale, stream);

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
      "Tensor qscale, Tensor kscale, Tensor vscale, Tensor cu_seqlens_q,"
      "Tensor block_ids, Tensor num_seq_kvcache, int max_seqlens_q, int quant_type,"
      "Tensor? output) -> (Tensor)");
  m.impl("attention_with_kvcache_prefill_fp8", torch::kCUDA,
         &hpc::attention::attention_with_kvcache_prefill_fp8_entry);

  m.def(
      "attention_with_kvcache_blocksparse_prefill_fp8(Tensor q, Tensor kcache, Tensor vcache,"
      "Tensor qscale, Tensor kscale, Tensor vscale, Tensor cu_seqlens_q,"
      "Tensor block_ids, Tensor num_seq_kvcache, int max_seqlens_q, int quant_type,"
      "Tensor? block_mask, Tensor? output) -> (Tensor)");
  m.impl("attention_with_kvcache_blocksparse_prefill_fp8", torch::kCUDA,
         &hpc::attention::attention_with_kvcache_blocksparse_prefill_fp8_entry);

  m.def(
      "attention_blocksparse_prefill_fp8_dim192(Tensor q, Tensor k, Tensor v,"
      "Tensor cu_seqlens_q, Tensor cu_seqlens_kv,"
      "int max_seqlens_q, int max_seqlens_kv,"
      "Tensor? block_mask, Tensor q_scale, Tensor k_scale, Tensor v_scale,"
      "float softmax_scale, Tensor? output) -> (Tensor)");
  m.impl("attention_blocksparse_prefill_fp8_dim192", torch::kCUDA,
         &hpc::attention::attention_blocksparse_prefill_fp8_dim192_entry);

  m.def(
      "mla_prefill_bf16(Tensor q, Tensor kv, Tensor seqlens_q, Tensor cu_seqlens_q, "
      "int num_dim_qk, int num_dim_v, int max_seqlens_q, Tensor? output) -> (Tensor)");
  m.impl("mla_prefill_bf16", torch::kCUDA, &hpc::attention::mla_prefill_bf16_entry);
}
