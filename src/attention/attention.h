// Copyright 2025 hpc-ops authors

#ifndef SRC_ATTENTION_ATTENTION_H_
#define SRC_ATTENTION_ATTENTION_H_

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace hpc {
namespace attention {

void attention_prefill_bf16_async(void *y_ptr, const void *q_ptr, const void *k_ptr,
                                  const void *v_ptr, int num_batch, int num_seq_q, int num_seq_kv,
                                  int num_dim_qk, int num_dim_v, int num_head_q, int num_head_kv,
                                  int ldY, int ldQ, int ldK, int ldV, cudaStream_t stream);

}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_ATTENTION_H_
