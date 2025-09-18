// Copyright 2025 hpc-ops authors

#ifndef SRC_MAMBA_CONV1D_STATE_H_
#define SRC_MAMBA_CONV1D_STATE_H_

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace hpc {
namespace mamba {

void causal_conv1d_update_async(__nv_bfloat16 *zxbcdt_ptr, __nv_bfloat16 *conv_state_ptr,
                                const __nv_bfloat16 *weight_ptr, const __nv_bfloat16 *bias_ptr,
                                const int *indices_ptr, int num_batch, int state_len, int conv_dim,
                                int d_conv, int d_inner, int num_head, int num_spec_tokens,
                                const int *num_accept_tokens_ptr, cudaStream_t stream);

bool causal_conv1d_prefill_async(__nv_bfloat16 *middle_y_ptr, __nv_bfloat16 *y_ptr,
                                 __nv_bfloat16 *zxbcdt_ptr, __nv_bfloat16 *conv_state_ptr,
                                 const __nv_bfloat16 *weight_ptr, const __nv_bfloat16 *bias_ptr,
                                 const int *indices_ptr, const int *split_metadata_ptr,
                                 const float *x_scale_ptr, const float *y_scale_ptr, int num_batch,
                                 int total_chunks, int total_padded_seqlen, int state_len,
                                 int conv_dim, int d_inner, int num_head, int d_conv,
                                 int chunk_size, int tileL, cudaStream_t stream);
}  // namespace mamba
}  // namespace hpc

#endif  // SRC_MAMBA_CONV1D_STATE_H_
