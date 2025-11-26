// Copyright 2025 hpc-ops authors

#ifndef SRC_ALLREDUCE_FUSE_ALLREDUCE_RMSNORM_WITH_SCALE_H_
#define SRC_ALLREDUCE_FUSE_ALLREDUCE_RMSNORM_WITH_SCALE_H_

#include <cuda_runtime_api.h>

namespace hpc {
namespace allreduce {
void fused_allreduce_rmsnorm_with_scale_async(void *mc_input_ptr, void *in_residual_ptr,
                                              void *out_residual_ptr, void *weight_ptr, void *scale,
                                              void *scales2, void *mc_fp8_output_ptr,
                                              void *mc_fp8_output2_ptr, void *mc_fp32_output_ptr,
                                              void *signal_ptr, int64_t rank, int64_t world_size,
                                              int64_t num_max_blocks, double rms_norm_eps,
                                              int num_tokens, int hidden_size, bool is_moe,
                                              cudaStream_t stream);
}  // namespace allreduce
}  // namespace hpc

#endif  // SRC_ALLREDUCE_FUSE_ALLREDUCE_RMSNORM_WITH_SCALE_H_
