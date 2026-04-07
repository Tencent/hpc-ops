// Copyright 2025 hpc-ops authors

#ifndef SRC_ROPE_ROPE_H_
#define SRC_ROPE_ROPE_H_

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>
#include <stdint.h>

#include <vector>

namespace hpc {
namespace rope {

void apply_rotary_pos_emb_bf16_to_fp8_async(
    __nv_fp8_e4m3 *out_q_ptr, __nv_fp8_e4m3 *out_k_ptr, __nv_fp8_e4m3 *out_v_ptr,
    const __nv_bfloat16 *in_q_ptr, const __nv_bfloat16 *in_k_ptr, const __nv_bfloat16 *in_v_ptr,
    const int q_stride, const int k_stride, const int v_stride, const int out_q_stride,
    const int out_k_stride, const int out_v_stride, const float *cos_sin_ptr,
    const int *num_tokens_per_batch_ptr, const int *q_index_ptr, const float *q_norm_weight_ptr,
    const float *k_norm_weight_ptr, float *q_scale_ptr, const float *k_scale_ptr,
    const float *v_scale_ptr, int *split_k_flag_ptr, float upper_max, int num_batch, int num_rows,
    int num_q_heads, int num_kv_heads, int qk_head_dim, int v_head_dim, bool is_prefill,
    int qk_norm_policy, int max_seqlens_pad128, cudaStream_t stream);

void apply_rotary_pos_emb_blocked_kvcache_bf16_async(
    __nv_bfloat16 *out_q_ptr, __nv_bfloat16 *kcache_ptr, __nv_bfloat16 *vcache_ptr,
    const __nv_bfloat16 *in_qkv_ptr, const float *cos_sin_ptr, const int *num_tokens_per_batch_ptr,
    const int *q_index_ptr, const int *kv_block_indices_ptr, const float *q_norm_weight_ptr,
    const float *k_norm_weight_ptr, int kcache_block_offset, int vcache_block_offset, int num_batch,
    int max_num_kv_block_per_batch, int kv_block_size, int num_rows, int num_q_heads,
    int num_kv_heads, int qk_head_dim, int v_head_dim, bool is_prefill, int qk_norm_policy,
    cudaStream_t stream);

void apply_rotary_pos_emb_blocked_kvcache_bf16_to_fp8_async(
    __nv_fp8_e4m3 *out_q_ptr, __nv_fp8_e4m3 *kcache_ptr, __nv_fp8_e4m3 *vcache_ptr,
    const __nv_bfloat16 *in_qkv_ptr, const float *cos_sin_ptr, const int *num_tokens_per_batch_ptr,
    const int *q_index_ptr, const int *kv_block_indices_ptr, const float *q_norm_weight_ptr,
    const float *k_norm_weight_ptr, float *q_scale_ptr, const float *k_scale_ptr,
    const float *v_scale_ptr, int *split_k_flag_ptr, float upper_max, int kcache_block_offset,
    int vcache_block_offset, int num_batch, int max_num_kv_block_per_batch, int kv_block_size,
    int num_rows, int num_q_heads, int num_kv_heads, int qk_head_dim, int v_head_dim,
    bool is_prefill, int qk_norm_policy, int max_seqlens_pad128, cudaStream_t stream);

bool rope_interleave_bf16_async(void *y_ptr, void *x_ptr, const void *cos_sin_cache_ptr,
                                const int64_t *position_ptr, int num_tokens, int num_heads, int dim,
                                int ldX, int ldCache, int ldY, int ldXHead, int ldYHead,
                                cudaStream_t stream);

}  // namespace rope

namespace rope_v2 {
void rope_norm_store_kv_async(__nv_bfloat16 *out_q_ptr, __nv_bfloat16 *kcache_ptr,
                              __nv_bfloat16 *vcache_ptr, __nv_bfloat16 *out_k_ptr,
                              __nv_bfloat16 *out_v_ptr, const __nv_bfloat16 *in_qkv_ptr,
                              const float *cos_sin_ptr, const int *num_seqlen_per_req_ptr,
                              const int *q_index_ptr, const int *kvcache_indices_ptr,
                              const float *q_norm_weight_ptr, const float *k_norm_weight_ptr,
                              int kcache_block_offset, int vcache_block_offset, int num_batch,
                              int max_num_kv_block_per_batch, int kv_block_size, int num_rows,
                              int num_q_heads, int num_kv_heads, int qk_head_dim, int v_head_dim,
                              bool is_prefill, int qk_norm_policy, cudaStream_t stream);
void rope_norm_store_kv_fp8_async(
    __nv_fp8_e4m3 *out_q_ptr, __nv_fp8_e4m3 *kcache_ptr, __nv_fp8_e4m3 *vcache_ptr,
    __nv_fp8_e4m3 *out_k_ptr, __nv_fp8_e4m3 *out_v_ptr, int32_t *split_k_flag_ptr,
    float *q_scale_ptr, const __nv_bfloat16 *in_qkv_ptr, const float *cos_sin_ptr,
    const int *num_seqlen_per_req_ptr, const int *q_index_ptr, const int *kvcache_indices_ptr,
    const float *q_norm_weight_ptr, const float *k_norm_weight_ptr, const float *k_scale_ptr,
    const float *v_scale_ptr, const float *q_scale_inv_ptr, float upper_max, int max_seqlens,
    int kcache_block_offset, int vcache_block_offset, int num_batch, int max_num_kv_block_per_batch,
    int kv_block_size, int num_rows, int num_q_heads, int num_kv_heads, int qk_head_dim,
    int v_head_dim, bool is_prefill, int qk_norm_policy, int quant_policy, cudaStream_t stream);
}  // namespace rope_v2
}  // namespace hpc

#endif  // SRC_ROPE_ROPE_H_
