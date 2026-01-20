// Copyright (C) 2026 Tencent.

#include "src/attention/decode/decode.h"

#include <cuda_runtime_api.h>

#include <algorithm>

#include "src/attention/decode/m64_dim80.h"
#include "src/attention/decode/smallm_dim128.h"

namespace hpc {
namespace attention {

bool attention_decode_bf16_async(void *y_ptr, void *lse_ptr, void *split_out_ptr, const void *q_ptr,
                                 void *kcache_ptr, void *vcache_ptr, const int *block_ids_ptr,
                                 const int *num_seq_kvcache_ptr, bool new_kv_included, int splitk,
                                 int num_batch, int num_head_q, int num_head_k, int num_head_v,
                                 int num_dim_qk, int num_dim_v, int num_kvcache_blocks,
                                 int block_size, int num_seq_max_blocks, int ldY, int ldQ, int ldK,
                                 int ldV, cudaStream_t stream) {
  if (num_dim_qk == 128) {
    if (splitk <= 0) {
      return decode::smallm_dim128_async(
          y_ptr, q_ptr, kcache_ptr, vcache_ptr, block_ids_ptr, num_seq_kvcache_ptr, new_kv_included,
          num_batch, num_head_q, num_head_k, num_head_v, num_dim_qk, num_dim_v, num_kvcache_blocks,
          block_size, num_seq_max_blocks, ldY, ldQ, ldK, ldV, stream);
    } else {
      return decode::smallm_splitk_dim128_async(
          y_ptr, lse_ptr, split_out_ptr, q_ptr, kcache_ptr, vcache_ptr, block_ids_ptr,
          num_seq_kvcache_ptr, new_kv_included, splitk, num_batch, num_head_q, num_head_k,
          num_head_v, num_dim_qk, num_dim_v, num_kvcache_blocks, block_size, num_seq_max_blocks,
          ldY, ldQ, ldK, ldV, stream);
    }
  } else if (num_dim_qk == 80) {
    return decode::m64_dim80_async(
        y_ptr, q_ptr, kcache_ptr, vcache_ptr, block_ids_ptr, num_seq_kvcache_ptr, new_kv_included,
        num_batch, num_head_q, num_head_k, num_head_v, num_dim_qk, num_dim_v, num_kvcache_blocks,
        block_size, num_seq_max_blocks, ldY, ldQ, ldK, ldV, stream);
  }
  return false;
}

bool attention_decode_fp8_async(void *y_ptr, void *lse_ptr, void *split_out_ptr, const void *q_ptr,
                                void *kcache_ptr, void *vcache_ptr, const int *block_ids_ptr,
                                const int *num_seq_kvcache_ptr, const float *qscale_ptr,
                                const float *kscale_ptr, const float *vscale_ptr,
                                int *split_flag_ptr, bool new_kv_included, int splitk,
                                int splitk_min_len, int consumers, int num_batch, int num_head_q,
                                int num_head_k, int num_head_v, int num_dim_qk, int num_dim_v,
                                int num_kvcache_blocks, int block_size, int num_seq_max_blocks,
                                int qscale_pad_stride, int ldY, int ldQ, int ldK, int ldV,
                                cudaStream_t stream) {
  if (num_dim_qk == 128) {
    return decode::smallm_splitk_dim128_fp8_async(
        y_ptr, lse_ptr, split_out_ptr, q_ptr, kcache_ptr, vcache_ptr, block_ids_ptr,
        num_seq_kvcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr, split_flag_ptr, new_kv_included,
        splitk, splitk_min_len, consumers, num_batch, num_head_q, num_head_k, num_head_v,
        num_dim_qk, num_dim_v, num_kvcache_blocks, block_size, num_seq_max_blocks,
        qscale_pad_stride, ldY, ldQ, ldK, ldV, stream);
  }
  return false;
}

}  // namespace attention
}  // namespace hpc
