// Copyright 2025 hpc-ops authors

#ifndef SRC_ATTENTION_MLA_SMALLM_MLA_DIM512_H_
#define SRC_ATTENTION_MLA_SMALLM_MLA_DIM512_H_

#include <cuda_runtime_api.h>

namespace hpc {
namespace attention {
namespace mla {

bool smallm_mla_dim512_async(void *y_ptr, const void *q_ptr, const void *kvcache_ptr,
                             const int *block_ids_ptr, const int *cu_seqlens_q_ptr,
                             const int *num_seq_kv_ptr, int num_batch, int total_seq_q,
                             int num_head_q, int head_dim, int num_kvcache_blocks,
                             int num_seq_max_blocks, int ldY, int ldQ, int ldKV,
                             cudaStream_t stream);

}  // namespace mla
}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_MLA_SMALLM_MLA_DIM512_H_
