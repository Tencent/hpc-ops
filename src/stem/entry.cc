// Copyright (C) 2026 Tencent.

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include <algorithm>
#include <tuple>

#include "src/stem/stem_oam_gemm_dim128.h"
#include "src/stem/stem_oam_prep_paged_kv_dim128.h"
#include "src/stem/stem_oam_prep_varlen_q_dim128.h"
#include "src/stem/stem_tpd.h"

namespace hpc {
namespace stem {

// Returns (kflat, vbias) from paged KV cache.
// kflat: [num_batch, num_head_kv, max_Kb, stem_stride * dim_qk]  BF16
// vbias: [num_batch, num_head_kv, max_Kb]                        FP32
std::tuple<torch::Tensor, torch::Tensor> stem_oam_prep_paged_kv_entry(
    const torch::Tensor &kcache, const torch::Tensor &vcache, const torch::Tensor &kscale,
    const torch::Tensor &vscale, const torch::Tensor &kv_indices, const torch::Tensor &kv_seq_lens,
    double lambda_mag, int64_t stem_block_size, int64_t stem_stride, int64_t quant_type) {
  auto stream = at::cuda::getCurrentCUDAStream(kcache.get_device());

  TORCH_CHECK(kcache.device().is_cuda(), "kcache must be CUDA tensor");
  TORCH_CHECK(vcache.device().is_cuda(), "vcache must be CUDA tensor");
  TORCH_CHECK(kscale.device().is_cuda(), "kscale must be CUDA tensor");
  TORCH_CHECK(vscale.device().is_cuda(), "vscale must be CUDA tensor");
  TORCH_CHECK(kv_indices.device().is_cuda(), "kv_indices must be CUDA tensor");
  TORCH_CHECK(kv_seq_lens.device().is_cuda(), "kv_seq_lens must be CUDA tensor");
  TORCH_CHECK((quant_type == 0 || quant_type == 1), "quant_type only support 0/1");
  TORCH_CHECK((kscale.dtype().itemsize() == 4 || kscale.dtype().itemsize() == 1),
              "kscale dtype must be float or fp8");
  TORCH_CHECK(stem_block_size == 128 && stem_stride == 16,
              "stem_oam_prep_paged_kv: only stem_block_size=128, stem_stride=16 is supported");

  // kcache: [num_kvcache_blocks, kv_block_size, num_head_kv, dim_qk]
  int num_kvcache_blocks = kcache.size(0);
  int block_size = kcache.size(1);
  TORCH_CHECK(block_size == 32 || block_size == 64,
              "stem_oam_prep_paged_kv: kv_block_size must be 32 or 64, got ", block_size);
  int num_head_kv = kcache.size(2);
  int num_dim_qk = kcache.size(3);
  int num_dim_v = vcache.size(3);

  int num_batch = kv_seq_lens.size(0);
  int num_seq_max_blocks = kv_indices.size(1);

  int ldK = kcache.stride(0);
  int ldV = vcache.stride(0);

  int64_t max_kv_len = kv_seq_lens.max().item<int64_t>();
  int64_t max_kv_padded = ((max_kv_len + stem_block_size - 1) / stem_block_size) * stem_block_size;
  int max_num_stem_blocks = static_cast<int>(max_kv_padded / stem_block_size);
  int max_k_down_len = static_cast<int>(max_kv_padded / stem_stride);

  int kflat_inner = static_cast<int>(stem_stride * num_dim_qk);

  auto kflat = torch::empty({num_batch, num_head_kv, max_num_stem_blocks, kflat_inner},
                            kcache.options().dtype(torch::kBFloat16));

  auto vbias = torch::empty({num_batch, num_head_kv, max_num_stem_blocks},
                            kcache.options().dtype(torch::kFloat32));

  auto v_norm_down = torch::empty({num_batch, num_head_kv, max_k_down_len},
                                  kcache.options().dtype(torch::kFloat32));

  TORCH_CHECK(num_dim_qk == 128 && num_dim_v == 128,
              "stem_oam_prep_paged_kv: unsupported dim_qk=", num_dim_qk, " dim_v=", num_dim_v,
              " (expected dim_qk=128, dim_v=128)");

  if (quant_type == 1) {
    stem_oam_prep_paged_kv_qpertoken_perhead_kvpertensor_dim128_async(
        kflat.data_ptr(), vbias.data_ptr(), kcache.const_data_ptr(), vcache.const_data_ptr(),
        kscale.const_data_ptr(), vscale.const_data_ptr(), kv_indices.const_data_ptr(),
        kv_seq_lens.const_data_ptr(), num_batch, num_dim_qk, num_dim_v, num_head_kv,
        num_kvcache_blocks, block_size, num_seq_max_blocks, static_cast<int>(stem_block_size),
        static_cast<int>(stem_stride), static_cast<float>(lambda_mag), max_num_stem_blocks,
        max_k_down_len, ldK, ldV, v_norm_down.data_ptr(), stream);
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
    int scale_block_size = static_cast<int>(kscale.size(1));
    stem_oam_prep_paged_kv_qkpertoken_perhead_vperhead_dim128_async(
        kflat.data_ptr(), vbias.data_ptr(), kcache.const_data_ptr(), vcache.const_data_ptr(),
        kscale.const_data_ptr(), vscale.const_data_ptr(), kv_indices.const_data_ptr(),
        kv_seq_lens.const_data_ptr(), num_batch, num_dim_qk, num_dim_v, num_head_kv,
        num_kvcache_blocks, block_size, scale_block_size, num_seq_max_blocks,
        static_cast<int>(stem_block_size), static_cast<int>(stem_stride),
        static_cast<float>(lambda_mag), max_num_stem_blocks, max_k_down_len, ldK, ldV, ldKS, ldKS1,
        ldKS2, v_norm_down.data_ptr(), stream);
  }

  return std::make_tuple(kflat, vbias);
}

// Returns (kflat, vbias) from ragged varlen FP8 KV.

// Returns dim128 qflat [num_batch, num_head_q, max_Qb, stem_stride * dim_qk] BF16.
torch::Tensor stem_oam_prep_varlen_q_entry(const torch::Tensor &q_fp8, const torch::Tensor &qscale,
                                           const torch::Tensor &q_seq_lens,
                                           const torch::Tensor &cu_seqlens_q,
                                           int64_t stem_block_size, int64_t stem_stride) {
  auto stream = at::cuda::getCurrentCUDAStream(q_fp8.get_device());

  TORCH_CHECK(q_fp8.device().is_cuda(), "q_fp8 must be CUDA tensor");
  TORCH_CHECK(q_fp8.is_contiguous(), "q_fp8 must be contiguous");
  TORCH_CHECK(qscale.device().is_cuda(), "qscale must be CUDA tensor");
  TORCH_CHECK(q_seq_lens.device().is_cuda(), "q_seq_lens must be CUDA tensor");
  TORCH_CHECK(cu_seqlens_q.device().is_cuda(), "cu_seqlens_q must be CUDA tensor");
  TORCH_CHECK(stem_block_size == 128 && stem_stride == 16,
              "stem_oam_prep_varlen_q: only stem_block_size=128, stem_stride=16 is supported");

  // q_fp8: [total_q, Hq, Dqk]
  int num_head_q = q_fp8.size(1);
  int num_dim_qk = q_fp8.size(2);
  int num_batch = q_seq_lens.size(0);
  int ldQ = q_fp8.stride(0);
  TORCH_CHECK(num_dim_qk == 128, "stem_oam_prep_varlen_q: expected dim_qk=128, got ", num_dim_qk);
  TORCH_CHECK(qscale.dim() == 3, "stem_oam_prep_varlen_q: qscale must be [B, Hq, max_q_pad]");

  int64_t max_q_len = q_seq_lens.max().item<int64_t>();
  int64_t max_q_padded = ((max_q_len + stem_block_size - 1) / stem_block_size) * stem_block_size;
  int max_num_q_blocks = static_cast<int>(max_q_padded / stem_block_size);

  int qflat_inner = static_cast<int>(stem_stride) * num_dim_qk;

  auto qflat = torch::empty({num_batch, num_head_q, max_num_q_blocks, qflat_inner},
                            q_fp8.options().dtype(torch::kBFloat16));

  stem_oam_prep_varlen_q_dim128_async(
      qflat.data_ptr(), q_fp8.const_data_ptr(), qscale.const_data_ptr(),
      q_seq_lens.const_data_ptr(), cu_seqlens_q.const_data_ptr(), num_batch, num_head_q, ldQ,
      static_cast<int>(stem_block_size), static_cast<int>(stem_stride), max_num_q_blocks, stream);

  return qflat;
}

// Returns dim128 block_logits [num_batch, num_head_q, max_Qb, max_Kb] BF16.
// All invalid positions (causal / OOB / padding) are set to -inf.
torch::Tensor stem_oam_gemm_entry(const torch::Tensor &qflat, const torch::Tensor &kflat,
                                  const torch::Tensor &vbias, const torch::Tensor &q_seq_lens,
                                  const torch::Tensor &kv_seq_lens, int64_t stem_block_size,
                                  int64_t stem_stride, bool causal) {
  auto stream = at::cuda::getCurrentCUDAStream(qflat.get_device());

  TORCH_CHECK(qflat.device().is_cuda(), "qflat must be CUDA tensor");
  TORCH_CHECK(kflat.device().is_cuda(), "kflat must be CUDA tensor");
  TORCH_CHECK(vbias.device().is_cuda(), "vbias must be CUDA tensor");
  TORCH_CHECK(q_seq_lens.device().is_cuda(), "q_seq_lens must be CUDA tensor");
  TORCH_CHECK(kv_seq_lens.device().is_cuda(), "kv_seq_lens must be CUDA tensor");
  TORCH_CHECK(stem_block_size == 128 && stem_stride == 16,
              "stem_oam_gemm: only stem_block_size=128, stem_stride=16 is supported");
  TORCH_CHECK(kv_seq_lens.device().is_cuda(), "kv_seq_lens must be CUDA tensor");

  // qflat:  [B, Hq,  max_num_qb, qFlatDim]
  // kflat:  [B, Hkv, max_num_kb, kFlatDim]
  // vbias:  [B, Hkv, max_num_kb]
  int num_batch = qflat.size(0);
  int num_head_q = qflat.size(1);
  int max_num_qb = qflat.size(2);
  int kflat_inner = qflat.size(3);

  int num_head_kv = kflat.size(1);
  int max_num_kb = kflat.size(2);

  TORCH_CHECK(qflat.size(0) == kflat.size(0),
              "stem_oam_gemm: batch size mismatch between qflat and kflat");
  TORCH_CHECK(qflat.size(3) == kflat.size(3), "stem_oam_gemm: kFlatDim mismatch between qflat (",
              qflat.size(3), ") and kflat (", kflat.size(3), ")");
  TORCH_CHECK(num_head_q % num_head_kv == 0, "stem_oam_gemm: num_head_q (", num_head_q,
              ") must be divisible by num_head_kv (", num_head_kv, ")");
  TORCH_CHECK(
      vbias.size(0) == num_batch && vbias.size(1) == num_head_kv && vbias.size(2) == max_num_kb,
      "stem_oam_gemm: vbias shape mismatch, expected [", num_batch, ", ", num_head_kv, ", ",
      max_num_kb, "]");

  int num_dim_qk = kflat_inner / static_cast<int>(stem_stride);
  TORCH_CHECK(num_dim_qk == 128, "stem_oam_gemm: expected dim_qk=128, got ", num_dim_qk);

  // qflat/kflat are used directly — TMA Load zero-fills OOB reads.
  // Only block_logits needs padding for TMA Store (copy box [64, 128]):
  //   dim2 >= 64 and multiple of 64
  //   dim3 >= 128 and multiple of 64 (for 128B stride alignment)
  int logits_qb = std::max(((max_num_qb + 63) / 64) * 64, 64);
  int logits_kb = std::max(((max_num_kb + 63) / 64) * 64, 128);

  auto block_logits =
      torch::full({num_batch, num_head_q, logits_qb, logits_kb},
                  -std::numeric_limits<float>::infinity(), qflat.options().dtype(torch::kBFloat16));

  stem_oam_gemm_dim128_async(
      block_logits.data_ptr(), qflat.const_data_ptr(), kflat.const_data_ptr(),
      vbias.const_data_ptr(), q_seq_lens.const_data_ptr(), kv_seq_lens.const_data_ptr(), num_batch,
      num_head_q, num_head_kv, max_num_qb, max_num_kb, static_cast<int>(stem_block_size),
      static_cast<int>(stem_stride), causal, stream);

  // Slice back to real shape. No-op when no padding; cheap copy for short sequences.
  if (logits_qb == max_num_qb && logits_kb == max_num_kb) {
    return block_logits;
  }
  return block_logits.slice(2, 0, max_num_qb).slice(3, 0, max_num_kb).contiguous();
}

// Returns u8_mask [num_batch, num_heads, max_Qb, max_Kb] uint8.
// `num_prompt_tokens` ([num_batch] int32) is the full prompt KV-token count
// per request (chunked-invariant; pass kv_seq_lens for normal prefill).
torch::Tensor stem_tpd_entry(const torch::Tensor &block_logits, const torch::Tensor &q_seq_lens,
                             const torch::Tensor &kv_seq_lens,
                             const torch::Tensor &num_prompt_tokens, int64_t block_size,
                             double alpha, int64_t initial_blocks, int64_t window_size,
                             double k_block_num_rate_medium, int64_t k_block_num_bias_medium,
                             double k_block_num_rate_large, int64_t k_block_num_bias_large) {
  auto stream = at::cuda::getCurrentCUDAStream(block_logits.get_device());

  TORCH_CHECK(block_logits.device().is_cuda(), "block_logits must be CUDA tensor");
  TORCH_CHECK(q_seq_lens.device().is_cuda(), "q_seq_lens must be CUDA tensor");
  TORCH_CHECK(kv_seq_lens.device().is_cuda(), "kv_seq_lens must be CUDA tensor");
  TORCH_CHECK(num_prompt_tokens.device().is_cuda(), "num_prompt_tokens must be CUDA tensor");
  TORCH_CHECK(block_logits.dtype() == torch::kBFloat16, "block_logits must be bfloat16");
  TORCH_CHECK(block_logits.is_contiguous(), "block_logits must be contiguous");
  TORCH_CHECK(q_seq_lens.dtype() == torch::kInt32, "q_seq_lens must be int32, got ",
              q_seq_lens.dtype());
  TORCH_CHECK(kv_seq_lens.dtype() == torch::kInt32, "kv_seq_lens must be int32, got ",
              kv_seq_lens.dtype());
  TORCH_CHECK(num_prompt_tokens.dtype() == torch::kInt32, "num_prompt_tokens must be int32, got ",
              num_prompt_tokens.dtype());

  // block_logits: [B, Hq, max_Qb, max_Kb]
  int num_batch = block_logits.size(0);
  int num_heads = block_logits.size(1);
  int max_Qb = block_logits.size(2);
  int max_Kb = block_logits.size(3);

  TORCH_CHECK(num_prompt_tokens.dim() == 1 && num_prompt_tokens.size(0) == num_batch,
              "num_prompt_tokens must have shape [num_batch=", num_batch, "], got ",
              num_prompt_tokens.sizes());

  // Maximum number of blocks (4M tokens) supported by TPD kernel
  TORCH_CHECK(max_Kb <= 32768, "stem_tpd: max_Kb=", max_Kb, " exceeds 32768 limit");

  auto mask = torch::zeros({num_batch, num_heads, max_Qb, max_Kb},
                           block_logits.options().dtype(torch::kUInt8));

  stem_tpd_async(
      mask.data_ptr(), block_logits.const_data_ptr(), q_seq_lens.const_data_ptr(),
      kv_seq_lens.const_data_ptr(), num_prompt_tokens.const_data_ptr(), num_batch, num_heads,
      max_Qb, max_Kb, static_cast<int>(block_size), static_cast<float>(alpha),
      static_cast<int>(initial_blocks), static_cast<int>(window_size),
      static_cast<float>(k_block_num_rate_medium), static_cast<int>(k_block_num_bias_medium),
      static_cast<float>(k_block_num_rate_large), static_cast<int>(k_block_num_bias_large), stream);

  return mask;
}

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "stem_oam_prep_paged_kv(Tensor kcache, Tensor vcache, "
      "Tensor kscale, Tensor vscale, Tensor kv_indices, Tensor kv_seq_lens, "
      "float lambda_mag, int stem_block_size, int stem_stride, int quant_type) "
      "-> (Tensor, Tensor)");
  m.impl("stem_oam_prep_paged_kv", torch::kCUDA, &hpc::stem::stem_oam_prep_paged_kv_entry);

  m.def(
      "stem_oam_prep_varlen_q(Tensor q_fp8, Tensor qscale, Tensor q_seq_lens, "
      "Tensor cu_seqlens_q, int stem_block_size, int stem_stride) -> Tensor");
  m.impl("stem_oam_prep_varlen_q", torch::kCUDA, &hpc::stem::stem_oam_prep_varlen_q_entry);

  m.def(
      "stem_oam_gemm(Tensor qflat, Tensor kflat, Tensor vbias, "
      "Tensor q_seq_lens, Tensor kv_seq_lens, "
      "int stem_block_size, int stem_stride, bool causal) -> Tensor");
  m.impl("stem_oam_gemm", torch::kCUDA, &hpc::stem::stem_oam_gemm_entry);

  m.def(
      "stem_tpd(Tensor block_logits, Tensor q_seq_lens, Tensor kv_seq_lens, "
      "Tensor num_prompt_tokens, "
      "int block_size, float alpha, int initial_blocks, int window_size, "
      "float k_block_num_rate_medium, int k_block_num_bias_medium, "
      "float k_block_num_rate_large, int k_block_num_bias_large) -> Tensor");
  m.impl("stem_tpd", torch::kCUDA, &hpc::stem::stem_tpd_entry);
}

}  // namespace stem
}  // namespace hpc
