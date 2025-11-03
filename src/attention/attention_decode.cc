// Copyright 2025 hpc-ops authors

#include "src/attention/attention_decode.h"

#include <cuda.h>
#include <stdio.h>

#include <algorithm>
#include <iostream>

#include "src/attention/attention.h"

namespace hpc {
namespace attention {

bool attention_decode_bf16_async(void *y_ptr, const void *q_ptr, void *kcache_ptr, void *vcache_ptr,
                                 const int *block_ids_ptr, const int *num_seq_kvcache_ptr,
                                 int num_batch, int num_head_q, int num_head_k, int num_head_v,
                                 int num_dim_qk, int num_dim_v, int num_kvcache_blocks,
                                 int block_size, int num_seq_max_blocks, int ldY, int ldQ, int ldK,
                                 int ldV, cudaStream_t stream) {
  if (num_dim_qk == 80) {
    return attention_decode_bf16_headdim80_async(
        y_ptr, q_ptr, kcache_ptr, vcache_ptr, block_ids_ptr, num_seq_kvcache_ptr, num_batch,
        num_head_q, num_head_k, num_head_v, num_dim_qk, num_dim_v, num_kvcache_blocks, block_size,
        num_seq_max_blocks, ldY, ldQ, ldK, ldV, stream);
  } else if (num_dim_qk == 128) {
    return attention_decode_bf16_headdim128_smallm_async(
        y_ptr, q_ptr, kcache_ptr, vcache_ptr, block_ids_ptr, num_seq_kvcache_ptr, num_batch,
        num_head_q, num_head_k, num_head_v, num_dim_qk, num_dim_v, num_kvcache_blocks, block_size,
        num_seq_max_blocks, ldY, ldQ, ldK, ldV, stream);
  }
  return false;
}
}  // namespace attention
}  // namespace hpc
