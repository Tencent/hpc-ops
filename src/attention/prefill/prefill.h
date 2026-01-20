// Copyright (C) 2026 Tencent.

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

void attention_with_kvcache_prefill_fp8_async(
    void *y_ptr, const void *q_ptr, const void *kcache_ptr, const void *vcache_ptr,
    const void *qkscale_ptr, const void *vscale_ptr, const void *cu_seqlens_q_ptr,
    const void *block_ids_ptr, const void *seqlens_kvcache_ptr, void *tmas_ptr, int num_batch,
    int total_seq_q, int max_seq_q, int max_seq_q_pad, int num_dim_qk, int num_dim_v,
    int num_head_q, int num_head_kv, int num_kvcache_blocks, int block_size, int num_seq_max_blocks,
    int ldY, int ldQ, int ldK, int ldV, cudaStream_t stream);

}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_PREFILL_PREFILL_H_
