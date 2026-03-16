// Copyright 2025 hpc-ops authors

#ifndef SRC_ATTENTION_PREFILL_WARP_SPEC_MLA_H_
#define SRC_ATTENTION_PREFILL_WARP_SPEC_MLA_H_

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace hpc {
namespace attention {
namespace prefill {

void warp_spec_mla_async(void *y_ptr, const void *q_ptr, const void *kv_ptr,
                         const void *seqlens_q_ptr, const void *cu_seqlens_q_ptr, void *tmas_ptr,
                         int num_batch, int total_seq_q, int max_seq_q, int num_dim_qk,
                         int num_dim_v, int num_head_q, int num_head_kv, int ldY, int ldQ, int ldKV,
                         cudaStream_t stream);

}  // namespace prefill
}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_PREFILL_WARP_SPEC_MLA_H_
