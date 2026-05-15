// Copyright 2025 hpc-ops authors

#ifndef SRC_ALLREDUCE_FUSE_ALLREDUCE_RMSNORM_V2_H_
#define SRC_ALLREDUCE_FUSE_ALLREDUCE_RMSNORM_V2_H_

#include <cuda_runtime_api.h>

namespace hpc {
namespace allreduce {

// Only __nv_bfloat16 is supported.
void fuse_allreduce_rmsnorm_v2_async(const void* input, const void* mc_input,
                                     void** buffer_ptrs_dev, void* buffer_ptr_local,
                                     uint32_t* buffer_flags, const void* in_residual,
                                     const void* gamma, void* output, void* out_residual, int rank,
                                     int world_size, double rms_norm_eps, int num_tokens,
                                     int hidden_size, bool launch_with_pdl, cudaStream_t stream);

}  // namespace allreduce
}  // namespace hpc

#endif  // SRC_ALLREDUCE_FUSE_ALLREDUCE_RMSNORM_V2_H_
