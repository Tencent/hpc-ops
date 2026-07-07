// Copyright 2025 hpc-ops authors

#ifndef SRC_ATTENTION_PREFILL_PREFILL_H_
#define SRC_ATTENTION_PREFILL_PREFILL_H_

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace hpc {
namespace attention {

void attention_prefill_bf16_async(void *y_ptr, const void *q_ptr, const void *k_ptr,
                                  const void *v_ptr, const void *seqlens_q_ptr,
                                  const void *cu_seqlens_q_ptr, void *tmas_ptr, int num_batch,
                                  int total_seq_q, int max_seq_q, int num_dim_qk, int num_dim_v,
                                  int num_head_q, int num_head_kv, int ldY, int ldQ, int ldK,
                                  int ldV, cudaStream_t stream);

void attention_with_kvcache_prefill_bf16_async(
    void *y_ptr, const void *q_ptr, const void *kcache_ptr, const void *vcache_ptr,
    const void *cu_seqlens_q_ptr, const void *block_ids_ptr, const void *num_seq_kvcache_ptr,
    void *tmas_ptr, int num_batch, int total_seq_q, int max_seq_q, int num_dim_qk, int num_dim_v,
    int num_head_q, int num_head_kv, int num_kvcache_blocks, int block_size, int num_seq_max_blocks,
    int ldY, int ldQ, int ldK, int ldK1, int ldK2, int ldV, int ldV1, int ldV2,
    cudaStream_t stream);

void attention_with_kvcache_prefill_bf16_hybrid_mask_async(
    void *y_ptr, const void *q_ptr, const void *kcache_ptr, const void *vcache_ptr,
    const void *cu_seqlens_q_ptr, const void *block_ids_ptr, const void *num_seq_kvcache_ptr,
    const void *mm_prefix_range_ptr, int max_spans, void *tmas_ptr, int num_batch, int total_seq_q,
    int max_seq_q, int num_dim_qk, int num_dim_v, int num_head_q, int num_head_kv,
    int num_kvcache_blocks, int block_size, int num_seq_max_blocks, int ldY, int ldQ, int ldK,
    int ldK1, int ldK2, int ldV, int ldV1, int ldV2, cudaStream_t stream);

// Hybrid a8c8-fp16pv prefill: Q/K/V all FP8; FP8 QK WGMMA + FP16 PV WGMMA.
// quant_type 20 = K per-token+head / V per-head; 21 = K/V per-tensor. qscale
// (per-token+head, fp32 [num_batch, num_head_q, max_seq_q_pad]) is required.
void attention_with_kvcache_prefill_fp8_kv_fp16_pv_compute_async(
    void *y_ptr, const void *q_ptr, const void *kcache_ptr, const void *vcache_ptr,
    const void *qscale_ptr, const void *kscale_ptr, const void *vscale_ptr,
    const void *cu_seqlens_q_ptr, const void *block_ids_ptr, const void *seqlens_kvcache_ptr,
    void *tmas_ptr, int num_batch, int total_seq_q, int max_seq_q, int max_seq_q_pad,
    int num_dim_qk, int num_dim_v, int num_head_q, int num_head_kv, int num_kvcache_blocks,
    int block_size, int num_seq_max_blocks, int ldY, int ldQ, int ldK, int ldK1, int ldK2, int ldV,
    int ldV1, int ldV2, int quant_type, cudaStream_t stream);

void attention_with_kvcache_prefill_qpertoken_perhead_kvpertensor_fp8_async(
    void *y_ptr, const void *q_ptr, const void *kcache_ptr, const void *vcache_ptr,
    const void *qscale_ptr, const void *kscale_ptr, const void *vscale_ptr,
    const void *cu_seqlens_q_ptr, const void *block_ids_ptr, const void *seqlens_kvcache_ptr,
    void *tmas_ptr, int num_batch, int total_seq_q, int max_seq_q, int max_seq_q_pad,
    int num_dim_qk, int num_dim_v, int num_head_q, int num_head_kv, int num_kvcache_blocks,
    int block_size, int num_seq_max_blocks, int ldY, int ldQ, int ldK, int ldK1, int ldK2, int ldV,
    int ldV1, int ldV2, cudaStream_t stream);

void attention_with_kvcache_prefill_qkpertoken_perhead_vperhead_fp8_async(
    void *y_ptr, const void *q_ptr, const void *kcache_ptr, const void *vcache_ptr,
    const void *qscale_ptr, const void *kscale_ptr, const void *vscale_ptr,
    const void *cu_seqlens_q_ptr, const void *block_ids_ptr, const void *seqlens_kvcache_ptr,
    void *tmas_ptr, int num_batch, int total_seq_q, int max_seq_q, int max_seq_q_pad,
    int num_dim_qk, int num_dim_v, int num_head_q, int num_head_kv, int num_kvcache_blocks,
    int block_size, int scale_block_size, int num_seq_max_blocks, int ldY, int ldQ, int ldK,
    int ldK1, int ldK2, int ldV, int ldV1, int ldV2, int ldKS, int ldKS1, int ldKS2,
    cudaStream_t stream);

void attention_with_kvcache_prefill_qpertoken_perhead_kvpertensor_fp8_hybrid_mask_async(
    void *y_ptr, const void *q_ptr, const void *kcache_ptr, const void *vcache_ptr,
    const void *qscale_ptr, const void *kscale_ptr, const void *pscale_ptr, const void *vscale_ptr,
    const void *cu_seqlens_q_ptr, const void *block_ids_ptr, const void *seqlens_kvcache_ptr,
    const void *mm_prefix_range_ptr, int max_spans, void *tmas_ptr, int num_batch, int total_seq_q,
    int max_seq_q, int max_seq_q_pad, int num_dim_qk, int num_dim_v, int num_head_q,
    int num_head_kv, int num_kvcache_blocks, int block_size, int num_seq_max_blocks, int ldY,
    int ldQ, int ldK, int ldK1, int ldK2, int ldV, int ldV1, int ldV2, cudaStream_t stream);

void attention_with_kvcache_prefill_qkpertoken_perhead_vperhead_fp8_hybrid_mask_async(
    void *y_ptr, const void *q_ptr, const void *kcache_ptr, const void *vcache_ptr,
    const void *qscale_ptr, const void *kscale_ptr, const void *pscale_ptr, const void *vscale_ptr,
    const void *cu_seqlens_q_ptr, const void *block_ids_ptr, const void *seqlens_kvcache_ptr,
    const void *mm_prefix_range_ptr, int max_spans, void *tmas_ptr, int num_batch, int total_seq_q,
    int max_seq_q, int max_seq_q_pad, int num_dim_qk, int num_dim_v, int num_head_q,
    int num_head_kv, int num_kvcache_blocks, int block_size, int scale_block_size,
    int num_seq_max_blocks, int ldY, int ldQ, int ldK, int ldK1, int ldK2, int ldV, int ldV1,
    int ldV2, int ldKS, int ldKS1, int ldKS2, cudaStream_t stream);

void mla_prefill_bf16_async(void *y_ptr, const void *q_ptr, const void *kv_ptr,
                            const void *seqlens_q_ptr, const void *cu_seqlens_q_ptr, void *tmas_ptr,
                            int num_batch, int total_seq_q, int max_seq_q, int num_dim_qk,
                            int num_dim_v, int num_head_q, int num_head_kv, int ldY, int ldQ,
                            int ldKV, cudaStream_t stream);

void attention_with_kvcache_blocksparse_prefill_qpertoken_perhead_kvpertensor_fp8_async(
    void *y_ptr, const void *q_ptr, const void *kcache_ptr, const void *vcache_ptr,
    const void *qscale_ptr, const void *kscale_ptr, const void *vscale_ptr,
    const void *cu_seqlens_q_ptr, const void *block_ids_ptr, const void *seqlens_kvcache_ptr,
    void *tmas_ptr, int num_batch, int total_seq_q, int max_seq_q, int max_seq_q_pad,
    int num_dim_qk, int num_dim_v, int num_head_q, int num_head_kv, int num_kvcache_blocks,
    int block_size, int num_seq_max_blocks, int ldY, int ldQ, int ldK, int ldK1, int ldK2, int ldV,
    int ldV1, int ldV2, const void *block_mask_ptr, int num_tile_kv_in_mask, cudaStream_t stream);

void attention_with_kvcache_blocksparse_prefill_qkpertoken_perhead_vperhead_fp8_async(
    void *y_ptr, const void *q_ptr, const void *kcache_ptr, const void *vcache_ptr,
    const void *qscale_ptr, const void *kscale_ptr, const void *vscale_ptr,
    const void *cu_seqlens_q_ptr, const void *block_ids_ptr, const void *seqlens_kvcache_ptr,
    void *tmas_ptr, int num_batch, int total_seq_q, int max_seq_q, int max_seq_q_pad,
    int num_dim_qk, int num_dim_v, int num_head_q, int num_head_kv, int num_kvcache_blocks,
    int block_size, int scale_block_size, int num_seq_max_blocks, int ldY, int ldQ, int ldK,
    int ldK1, int ldK2, int ldV, int ldV1, int ldV2, int ldKS, int ldKS1, int ldKS2,
    const void *block_mask_ptr, int num_tile_kv_in_mask, cudaStream_t stream);

void attention_blocksparse_prefill_fp8_dim192_async(
    void *y_ptr, const void *q_ptr, const void *k_ptr, const void *v_ptr,
    const void *cu_seqlens_q_ptr, const void *cu_seqlens_kv_ptr, void *tmas_ptr, int num_batch,
    int total_seq_q, int max_seq_q, int max_seq_kv, int num_dim_qk, int num_dim_v, int num_head_q,
    int num_head_kv, int ldY, int ldQ, int ldK, int ldV, const void *block_mask_ptr,
    int num_tile_kv_in_mask, float softmax_qkscale, float vscale, cudaStream_t stream);

}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_PREFILL_PREFILL_H_
