// Copyright 2025 hpc-ops authors

#ifndef SRC_SAGE_ATTENTION_SM90_SAGE_QUANT_H_
#define SRC_SAGE_ATTENTION_SM90_SAGE_QUANT_H_

#include <torch/all.h>

namespace hpc {
namespace sage_attention {

// SM90 SageAttention2 quantization (Q INT8 / K INT8 / V FP8).
//
// Layouts (matching SageAttention SM90 official kernel + our cute kernel):
//   - q_int8     : same shape as q (bf16 -> int8).
//   - q_scale    : [batch, num_head_q,  num_block_q * 32]
//                  num_block_q = ceil(qo_len, 64) ; CTA_Q = 64.
//                  scale at offset (warp_idx*8 + lane_id/4) covers wgmma rows
//                  warp_idx*16 + lane_id/4 AND warp_idx*16 + lane_id/4 + 8.
//   - k_int8     : same shape as k (bf16 - km -> int8).
//   - k_scale    : [batch, num_head_kv, num_block_k * 4]
//                  num_block_k = ceil(kv_len, 128) ; CTA_K = 128.
//                  scale s in {0..3} covers the 32 K-token positions visited by
//                  lane%4 == s in wgmma m64n128k32 C: {atom*8 + s*2 + 0/1} for
//                  atom in 0..15 within the block.
//   - v_fp8      : transposed + per-channel FP8, padded along the kv axis to a
//                  multiple of 128 with explicit zero-fill, NO token permutation.
//                  HND : [batch, num_head_kv, head_dim, padded_kv_len_128]
//                  NHD : [batch, head_dim, num_head_kv, padded_kv_len_128]
//   - v_scale    : [batch, num_head_kv, head_dim]   (per-channel, fp32; produced
//                  by k_mean_v_scale_fused_cuda before this call).
//   - km         : [batch, *, num_head_kv, head_dim] mean of K along seq dim;
//                  subtracted before INT8 quant (smooth-K).  Required.
void qkv_fused_sm90_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor km,
                         torch::Tensor v_scale, torch::Tensor q_int8, torch::Tensor q_scale,
                         torch::Tensor k_int8, torch::Tensor k_scale, torch::Tensor v_fp8,
                         int tensor_layout);

// One-pass fused kernel that computes K mean (smooth-K) and per-channel V
// amax / 448 (per-channel FP8 scale).  Outputs:
//   - km       : fp32, shape [batch, *, num_head_kv, head_dim] (per kv head).
//   - v_scale  : fp32, shape [batch, num_head_kv, head_dim] (per channel).
// Called before `qkv_fused_sm90_cuda`, which consumes both.
void k_mean_v_scale_fused_cuda(torch::Tensor k, torch::Tensor v, torch::Tensor km,
                               torch::Tensor v_scale, int tensor_layout);

}  // namespace sage_attention
}  // namespace hpc

#endif  // SRC_SAGE_ATTENTION_SM90_SAGE_QUANT_H_
