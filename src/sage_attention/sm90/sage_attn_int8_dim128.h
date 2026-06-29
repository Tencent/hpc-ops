// Copyright 2025 hpc-ops authors

#ifndef SRC_SAGE_ATTENTION_SM90_SAGE_ATTN_INT8_DIM128_H_
#define SRC_SAGE_ATTENTION_SM90_SAGE_ATTN_INT8_DIM128_H_

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace hpc {
namespace sage_attention {

void sage_attn_int8_fp8_dim128_sm90_async(void *y_ptr, const void *q_ptr, const void *k_ptr,
                                          const void *v_ptr, const float *q_scale_ptr,
                                          const float *k_scale_ptr, const float *v_scale_ptr,
                                          int num_batch, int qo_len, int kv_len, int head_dim,
                                          int num_head_q, int num_head_kv, int stride_bz_y,
                                          int stride_seq_y, int stride_h_y, int stride_bz_q,
                                          int stride_seq_q, int stride_h_q, int stride_bz_k,
                                          int stride_seq_k, int stride_h_k, int tensor_layout,
                                          int is_causal, float sm_scale, cudaStream_t stream);

}  // namespace sage_attention
}  // namespace hpc

#endif  // SRC_SAGE_ATTENTION_SM90_SAGE_ATTN_INT8_DIM128_H_
