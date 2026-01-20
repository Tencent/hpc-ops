// Copyright (C) 2026 Tencent.

#ifndef SRC_ATTENTION_DECODE_DECODE_H_
#define SRC_ATTENTION_DECODE_DECODE_H_

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace hpc {
namespace attention {

bool attention_decode_bf16_async(void *y_ptr, void *lse_ptr, void *split_out_ptr, const void *q_ptr,
                                 void *kcache_ptr, void *vcache_ptr, const int *block_ids_ptr,
                                 const int *num_seq_kvcache_ptr, bool new_kv_included, int splitk,
                                 int num_batch, int num_head_q, int num_head_k, int num_head_v,
                                 int num_dim_qk, int num_dim_v, int num_kvcache_blocks,
                                 int block_size, int num_seq_max_blocks, int ldY, int ldQ, int ldK,
                                 int ldV, cudaStream_t stream);

bool attention_decode_fp8_async(void *y_ptr, void *lse_ptr, void *split_out_ptr, const void *q_ptr,
                                void *kcache_ptr, void *vcache_ptr, const int *block_ids_ptr,
                                const int *num_seq_kvcache_ptr, const float *qscale_ptr,
                                const float *kscale_ptr, const float *vscale_ptr,
                                int *split_flag_ptr, bool new_kv_included, int splitk,
                                int splitk_min_len, int consumers, int num_batch, int num_head_q,
                                int num_head_k, int num_head_v, int num_dim_qk, int num_dim_v,
                                int num_kvcache_blocks, int block_size, int num_seq_max_blocks,
                                int qscale_pad_stride, int ldY, int ldQ, int ldK, int ldV,
                                cudaStream_t stream);

}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_DECODE_DECODE_H_
