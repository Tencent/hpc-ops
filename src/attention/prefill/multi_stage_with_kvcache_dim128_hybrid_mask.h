// Copyright 2025 hpc-ops authors

#ifndef SRC_ATTENTION_PREFILL_MULTI_STAGE_WITH_KVCACHE_DIM128_HYBRID_MASK_H_
#define SRC_ATTENTION_PREFILL_MULTI_STAGE_WITH_KVCACHE_DIM128_HYBRID_MASK_H_

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace hpc {
namespace attention {
namespace prefill {

void multi_stage_with_kvcache_dim128_hybrid_mask_async(
    void *y_ptr, const void *q_ptr, const void *kcache_ptr, const void *vcache_ptr,
    const void *cu_seqlens_q_ptr, const void *block_ids_ptr, const void *seqlens_kvcache_ptr,
    const void *mm_prefix_range_ptr, int max_spans, void *tmas_ptr, int num_batch, int total_seq_q,
    int max_seq_q, int num_dim_qk, int num_dim_v, int num_head_q, int num_head_kv,
    int num_kvcache_blocks, int block_size, int num_seq_max_blocks, int ldY, int ldQ, int ldK,
    int ldK1, int ldK2, int ldV, int ldV1, int ldV2, cudaStream_t stream);

}  // namespace prefill
}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_PREFILL_MULTI_STAGE_WITH_KVCACHE_DIM128_HYBRID_MASK_H_
