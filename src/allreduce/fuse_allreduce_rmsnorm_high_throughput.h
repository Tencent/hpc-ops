// Copyright 2025 hpc-ops authors

#ifndef SRC_ALLREDUCE_FUSE_ALLREDUCE_RMSNORM_HIGH_THROUGHPUT_H_
#define SRC_ALLREDUCE_FUSE_ALLREDUCE_RMSNORM_HIGH_THROUGHPUT_H_

#include <cuda_runtime_api.h>

namespace hpc {
namespace allreduce {

void fuse_allreduce_rmsnorm_high_throughput_async(const void *input_ptr, const void *mc_input_ptr,
                                                  const void *in_res_ptr, const void *weight_ptr,
                                                  void *output_ptr, void *mc_output_ptr,
                                                  void *out_res_ptr, void *signal_ptr, int64_t rank,
                                                  int64_t world_size, int64_t num_max_blocks,
                                                  double rms_norm_eps, int num_tokens,
                                                  int hidden_size, cudaStream_t stream);

}  // namespace allreduce
}  // namespace hpc

#endif  // SRC_ALLREDUCE_FUSE_ALLREDUCE_RMSNORM_HIGH_THROUGHPUT_H_
