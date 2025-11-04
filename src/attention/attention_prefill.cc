// Copyright 2025 hpc-ops authors

#include "src/attention/attention_prefill.h"

#include <cuda.h>
#include <stdio.h>

#include <algorithm>

#include "src/utils/utils.h"

namespace hpc {
namespace attention {

void attention_prefill_bf16_async(void *y_ptr, const void *q_ptr, const void *k_ptr,
                                  const void *v_ptr, const void *seqlens_q_ptr,
                                  const void *cu_seqlens_q_ptr, void *tmas_ptr, int num_batch,
                                  int total_seq_q, int max_seq_q, int num_dim_qk, int num_dim_v,
                                  int num_head_q, int num_head_kv, int ldY, int ldQ, int ldK,
                                  int ldV, cudaStream_t stream) {
  constexpr int kTileM = 64;
  int max_total_blocks = (max_seq_q + kTileM - 1) / kTileM * num_batch * num_head_q;
  if (max_total_blocks < get_sm_count() * 2) {
    attention_prefill_bf16_multi_stage_async(y_ptr, q_ptr, k_ptr, v_ptr, seqlens_q_ptr,
                                             cu_seqlens_q_ptr, tmas_ptr, num_batch, total_seq_q,
                                             max_seq_q, num_dim_qk, num_dim_v, num_head_q,
                                             num_head_kv, ldY, ldQ, ldK, ldV, stream);
  } else {
    attention_prefill_bf16_warp_specialization_async(
        y_ptr, q_ptr, k_ptr, v_ptr, seqlens_q_ptr, cu_seqlens_q_ptr, tmas_ptr, num_batch,
        total_seq_q, max_seq_q, num_dim_qk, num_dim_v, num_head_q, num_head_kv, ldY, ldQ, ldK, ldV,
        stream);
  }
}
}  // namespace attention
}  // namespace hpc
