// Copyright 2025 hpc-ops authors

#ifndef SRC_ALLREDUCE_REDUCE_SCATTER_H_
#define SRC_ALLREDUCE_REDUCE_SCATTER_H_

#include <cuda_runtime_api.h>

namespace hpc {
namespace allreduce {

void reduce_scatter_async(const void *input_ptr, const void *mc_input_ptr, void *output_ptr,
                          void *mc_output_ptr, void *signal_ptr, int64_t rank, int64_t world_size,
                          int64_t num_max_blocks, int num_tokens, int hidden_size,
                          cudaStream_t stream);

}  // namespace allreduce
}  // namespace hpc

#endif  // SRC_ALLREDUCE_REDUCE_SCATTER_H_
