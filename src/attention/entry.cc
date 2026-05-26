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

  TORCH_CHECK((num_dim_qk == 128), "we only support head dim 128.");

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
  int kcache_block_stride = kcache.stride(0);
  int vcache_block_stride = vcache.stride(0);

  int kcache_token_stride = kcache.stride(1);
  int vcache_token_stride = vcache.stride(1);

  int kcache_head_stride = kcache.stride(2);
  int vcache_head_stride = vcache.stride(2);
  int ldY = y.stride(0);  // num_head_q * num_dim_v;

  bool running = attention_decode_bf16_async(
      y_ptr, lse_ptr, split_out_ptr, q_ptr, kcache_ptr, vcache_ptr, block_ids_ptr,
      num_seq_kvcache_ptr, split_flag_ptr, new_kv_included, splitk, num_batch, num_seq_q,
      num_head_q, num_head_k, num_head_v, num_dim_qk, num_dim_v, num_kvcache_blocks, block_size,
      num_seq_max_blocks, ldY, ldQ, kcache_block_stride, kcache_token_stride, kcache_head_stride,
      vcache_block_stride, vcache_token_stride, vcache_head_stride, stream);

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

torch::Tensor assign_attention_decode_task_cpu_entry(const torch::Tensor &num_seq_kvcache,
                                                     int64_t num_head_kv, int64_t num_seq_q,
                                                     bool new_kv_included, int64_t min_process_len,
                                                     std::optional<torch::Tensor> placehold) {
  TORCH_CHECK(num_seq_kvcache.device().is_cpu(), "num_seq_kvcache tensor must be cpu");
  int num_batch = num_seq_kvcache.size(0);

  const int *num_seq_kvcache_ptr = num_seq_kvcache.const_data_ptr<int>();

  auto tasks_pair = assign_attention_decode_task_sync(num_seq_kvcache_ptr, num_batch, num_head_kv,
                                                      num_seq_q, new_kv_included, min_process_len);
  auto tasks = tasks_pair.first;
  auto num_chunks = tasks_pair.second;
  int num_tile_per_sm = num_chunks[num_batch];

  constexpr int kTaskInfoSize = sizeof(TaskScheduleInfo);
  int num_task = tasks.size();
  auto options = num_seq_kvcache.options().dtype(torch::kInt8);

  int task_map_shape0 =
      1 + num_task + (num_batch * sizeof(int) + kTaskInfoSize - 1) / kTaskInfoSize;
  auto task_map = torch::zeros({task_map_shape0, kTaskInfoSize}, options);
  uint8_t *task_map_ptr = reinterpret_cast<uint8_t *>(task_map.mutable_data_ptr());

  memcpy(task_map_ptr, &num_tile_per_sm, sizeof(int));
  memcpy(task_map_ptr + kTaskInfoSize, tasks.data(), kTaskInfoSize * num_task);
  memcpy(task_map_ptr + kTaskInfoSize * (num_task + 1), num_chunks.data(), sizeof(int) * num_batch);

  return task_map;
}

torch::Tensor assign_attention_decode_task_cuda_entry(const torch::Tensor &num_seq_kvcache,
                                                      int64_t num_head_kv, int64_t num_seq_q,
                                                      bool new_kv_included, int64_t min_process_len,
                                                      std::optional<torch::Tensor> task_map) {
  TORCH_CHECK(num_seq_kvcache.device().is_cuda(), "num_seq_kvcache tensor must be cuda");
  int num_batch = num_seq_kvcache.size(0);

  auto stream = at::cuda::getCurrentCUDAStream(num_seq_kvcache.get_device());
  constexpr int kMaxNumBatch = 2048;

  TORCH_CHECK(num_batch <= kMaxNumBatch,
              "assign_attention_decode_task_cuda only support batch_size <= 2048");

  const int *num_seq_kvcache_ptr = num_seq_kvcache.const_data_ptr<int>();
  TORCH_CHECK(task_map.has_value(), "assign_attention_decode_task_cuda must use task_map output.")

  auto task_map_tensor = task_map.value();

  int *task_map_ptr = reinterpret_cast<int *>(task_map_tensor.mutable_data_ptr());

  auto running =
      assign_attention_decode_task_async(task_map_ptr, num_seq_kvcache_ptr, num_batch, num_head_kv,
                                         num_seq_q, new_kv_included, min_process_len, stream);

  TORCH_CHECK(running, "launch assign_attention_decode_task_async failed");

  return task_map_tensor;
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
      "attention_decode_bf16(Tensor q, Tensor! kcache, Tensor! vcache, Tensor block_ids, Tensor "
      "num_seq_kvcache, int mtp, bool new_kv_included, bool use_splitk, Tensor? split_flag, "
      "Tensor? output) -> "
      "(Tensor)");
  m.impl("attention_decode_bf16", torch::kCUDA, &hpc::attention::attention_decode_bf16_entry);

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

  m.def(
      "assign_attention_decode_task(Tensor num_seq_kvcache, int num_head_kv, int num_seq_q, bool "
      "new_kv_included, "
      "int min_process_len, Tensor? task_map) -> "
      "(Tensor)");
  m.impl("assign_attention_decode_task", torch::kCPU,
         &hpc::attention::assign_attention_decode_task_cpu_entry);
  m.impl("assign_attention_decode_task", torch::kCUDA,
         &hpc::attention::assign_attention_decode_task_cuda_entry);
}
