#ifndef SRC_MAMBA_CONV1D_STATE_H_
#define SRC_MAMBA_CONV1D_STATE_H_

#include <stdint.h> 
#include <cuda_runtime_api.h>

namespace hpc {
namespace mamba {

void causal_conv1d_update_async(
    __nv_bfloat16* zxbcdt_ptr, __nv_bfloat16* conv_state_ptr,
    const __nv_bfloat16* weight_ptr, const __nv_bfloat16* bias_ptr,
    const int *indices_ptr,int num_batch, int state_len, int conv_dim,
    int d_conv, int d_inner, int num_head, cudaStream_t stream);

}  // namespace mamba
}  // namespace hpc

#endif  // SRC_MAMBA_CONV1D_STATE_H_

