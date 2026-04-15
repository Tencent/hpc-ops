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
    int ldY, int ldQ, int ldK, int ldV, cudaStream_t stream);

void attention_with_kvcache_prefill_Qpertoken_KVpertensor_fp8_async(
    void *y_ptr, const void *q_ptr, const void *kcache_ptr, const void *vcache_ptr,
    const void *qscale_ptr, const void *kscale_ptr, const void *vscale_ptr,
    const void *cu_seqlens_q_ptr, const void *block_ids_ptr, const void *seqlens_kvcache_ptr,
    void *tmas_ptr, int num_batch, int total_seq_q, int max_seq_q, int max_seq_q_pad,
    int num_dim_qk, int num_dim_v, int num_head_q, int num_head_kv, int num_kvcache_blocks,
    int block_size, int num_seq_max_blocks, int ldY, int ldQ, int ldK, int ldV,
    cudaStream_t stream);

void attention_with_kvcache_prefill_QKpertoken_Vpertensor_fp8_async(
    void *y_ptr, const void *q_ptr, const void *kcache_ptr, const void *vcache_ptr,
    const void *qscale_ptr, const void *kscale_ptr, const void *vscale_ptr,
    const void *cu_seqlens_q_ptr, const void *block_ids_ptr, const void *seqlens_kvcache_ptr,
    void *tmas_ptr, int num_batch, int total_seq_q, int max_seq_q, int max_seq_q_pad,
    int num_dim_qk, int num_dim_v, int num_head_q, int num_head_kv, int num_kvcache_blocks,
    int block_size, int scale_block_size, int num_seq_max_blocks, int ldY, int ldQ, int ldK,
    int ldV, int ldKS, cudaStream_t stream);

void mla_prefill_bf16_async(void *y_ptr, const void *q_ptr, const void *kv_ptr,
                            const void *seqlens_q_ptr, const void *cu_seqlens_q_ptr, void *tmas_ptr,
                            int num_batch, int total_seq_q, int max_seq_q, int num_dim_qk,
                            int num_dim_v, int num_head_q, int num_head_kv, int ldY, int ldQ,
                            int ldKV, cudaStream_t stream);

void attention_with_kvcache_blocksparse_prefill_fp8_async(
    void *y_ptr, const void *q_ptr, const void *kcache_ptr, const void *vcache_ptr,
    const void *qscale_ptr, const void *kscale_ptr, const void *vscale_ptr,
    const void *cu_seqlens_q_ptr, const void *block_ids_ptr, const void *seqlens_kvcache_ptr,
    void *tmas_ptr, int num_batch, int total_seq_q, int max_seq_q, int max_seq_q_pad,
    int num_dim_qk, int num_dim_v, int num_head_q, int num_head_kv, int num_kvcache_blocks,
    int block_size, int num_seq_max_blocks, int ldY, int ldQ, int ldK, int ldV,
    const void *block_mask_ptr, int num_tile_kv_in_mask, cudaStream_t stream);

void attention_blocksparse_prefill_fp8_async(
    void *y_ptr, const void *q_ptr, const void *k_ptr, const void *v_ptr,
    const void *cu_seqlens_q_ptr, const void *cu_seqlens_kv_ptr, void *tmas_ptr, int num_batch,
    int total_seq_q, int max_seq_q, int max_seq_kv, int num_dim_qk, int num_dim_v, int num_head_q,
    int num_head_kv, int ldY, int ldQ, int ldK, int ldV, const void *block_mask_ptr,
    int num_tile_kv_in_mask, float softmax_qkscale, float vscale, cudaStream_t stream);

}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_PREFILL_PREFILL_H_
