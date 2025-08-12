// Copyright 2025 hpc-ops authors

#ifndef SRC_MAMBA_SELECTIVE_STATE_H_
#define SRC_MAMBA_SELECTIVE_STATE_H_

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace hpc {
namespace mamba {

void selective_state_update_async(__nv_bfloat16 *out_ptr, float *ssm_states_ptr,
                                  const int *indices_ptr, const __nv_bfloat16 *zxbcdt_ptr,
                                  const float *AD_ptr, const float *bias_ptr, int num_batch,
                                  int num_head, int head_dim, int num_group, int state_dim,
                                  int num_sp_tokens, const int *num_accepted_tokens_ptr,
                                  cudaStream_t stream);

}  // namespace mamba
}  // namespace hpc

#endif  // SRC_MAMBA_SELECTIVE_STATE_H_
