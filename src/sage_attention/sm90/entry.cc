// Copyright 2025 hpc-ops authors

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cmath>

#include "src/sage_attention/sm90/sage_attn_int8_dim128.h"
#include "src/sage_attention/sm90/sage_quant.h"  // k_mean_v_scale_fused_cuda + qkv_fused_sm90_cuda

namespace hpc {
namespace sage_attention {

// SM90 fused SageAttention2 entry.  Public schema:
//   sage_attn_fused(Tensor q, Tensor k, Tensor v, Tensor? output,
//                   int tensor_layout, int is_causal) -> Tensor
// Called from `hpc.sm90_sage_attention.sageattn_qk_int8_pv_fp8` (Python wrapper).
torch::Tensor sage_attn_fused_entry(const torch::Tensor &q, const torch::Tensor &k,
                                    const torch::Tensor &v, std::optional<torch::Tensor> output,
                                    int64_t tensor_layout, int64_t is_causal) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q/k/v must be on CUDA");
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16 && k.scalar_type() == torch::kBFloat16 &&
                  v.scalar_type() == torch::kBFloat16,
              "SM90 sage_attn_fused only supports bfloat16 (see qkv_fused_sm90_cuda / "
              "k_mean_v_scale_fused_cuda dtype checks).");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q/k/v must be 4D");
  TORCH_CHECK(q.size(-1) == 128, "SM90 sage_attn_fused only supports head_dim=128");
  TORCH_CHECK(q.stride(-1) == 1 && k.stride(-1) == 1 && v.stride(-1) == 1,
              "q/k/v last dim must be contiguous");

  const int batch_size = q.size(0);
  const int head_dim = q.size(3);
  int qo_len, kv_len, num_head_q, num_head_kv;
  if (tensor_layout == 0) {  // NHD
    qo_len = q.size(1);
    num_head_q = q.size(2);
    kv_len = k.size(1);
    num_head_kv = k.size(2);
  } else {  // HND
    num_head_q = q.size(1);
    qo_len = q.size(2);
    num_head_kv = k.size(1);
    kv_len = k.size(2);
  }

  auto device = q.device();
  auto opts_bf16 = torch::dtype(torch::kBFloat16).device(device);
  auto opts_fp32 = torch::dtype(torch::kFloat32).device(device);
  auto opts_i8 = torch::dtype(torch::kInt8).device(device);
  auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(device);

  // Middle-state allocation (SM90-specific layouts: q_scale per-thread 32/block
  // for CTA_Q=64; k_scale 4/block for CTA_K=128; v_fp8 padded to 128 to match
  // CTA_K).
  const int num_block_q = (qo_len + 63) / 64;
  const int num_block_k = (kv_len + 127) / 128;
  const int padded_kv_len = num_block_k * 128;

  torch::Tensor km, v_fp8;
  if (tensor_layout == 0) {  // NHD
    km = torch::empty({batch_size, 1, num_head_kv, head_dim}, opts_fp32);
    v_fp8 = torch::empty({batch_size, head_dim, num_head_kv, padded_kv_len}, opts_fp8);
  } else {  // HND
    km = torch::empty({batch_size, num_head_kv, 1, head_dim}, opts_fp32);
    v_fp8 = torch::empty({batch_size, num_head_kv, head_dim, padded_kv_len}, opts_fp8);
  }
  auto v_scale = torch::empty({batch_size, num_head_kv, head_dim}, opts_fp32);
  auto q_int8 = torch::empty(q.sizes(), opts_i8);
  auto k_int8 = torch::empty(k.sizes(), opts_i8);
  auto q_scale = torch::empty({batch_size, num_head_q, num_block_q * 32}, opts_fp32);
  auto k_scale = torch::empty({batch_size, num_head_kv, num_block_k * 4}, opts_fp32);

  torch::Tensor y;
  if (output.has_value()) {
    y = output.value();
    TORCH_CHECK(y.sizes() == q.sizes() && y.scalar_type() == torch::kBFloat16,
                "output must have same shape as q and dtype bf16");
  } else {
    y = torch::empty(q.sizes(), opts_bf16);
  }

  const float sm_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  k_mean_v_scale_fused_cuda(k, v, km, v_scale, tensor_layout);
  qkv_fused_sm90_cuda(q, k, v, km, v_scale, q_int8, q_scale, k_int8, k_scale, v_fp8, tensor_layout);

  auto stream = at::cuda::getCurrentCUDAStream(q.get_device());

  int stride_bz_y = y.stride(0), stride_seq_y, stride_h_y;
  int stride_bz_q = q_int8.stride(0), stride_seq_q, stride_h_q;
  int stride_bz_k = k_int8.stride(0), stride_seq_k, stride_h_k;
  if (tensor_layout == 0) {
    stride_seq_y = y.stride(1);
    stride_h_y = y.stride(2);
    stride_seq_q = q_int8.stride(1);
    stride_h_q = q_int8.stride(2);
    stride_seq_k = k_int8.stride(1);
    stride_h_k = k_int8.stride(2);
  } else {
    stride_h_y = y.stride(1);
    stride_seq_y = y.stride(2);
    stride_h_q = q_int8.stride(1);
    stride_seq_q = q_int8.stride(2);
    stride_h_k = k_int8.stride(1);
    stride_seq_k = k_int8.stride(2);
  }

  sage_attn_int8_fp8_dim128_sm90_async(
      y.mutable_data_ptr(), q_int8.const_data_ptr(), k_int8.const_data_ptr(),
      v_fp8.const_data_ptr(), q_scale.const_data_ptr<float>(), k_scale.const_data_ptr<float>(),
      v_scale.const_data_ptr<float>(), batch_size, qo_len, kv_len, head_dim, num_head_q,
      num_head_kv, stride_bz_y, stride_seq_y, stride_h_y, stride_bz_q, stride_seq_q, stride_h_q,
      stride_bz_k, stride_seq_k, stride_h_k, tensor_layout, is_causal, sm_scale, stream);

  return y;
}

}  // namespace sage_attention
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "sage_attn_fused(Tensor q, Tensor k, Tensor v, "
      "Tensor? output, int tensor_layout, int is_causal) -> (Tensor)");
  m.impl("sage_attn_fused", torch::kCUDA, &hpc::sage_attention::sage_attn_fused_entry);
}
