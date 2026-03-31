// Copyright 2025 hpc-ops authors

#ifndef SRC_UCL_COMM_FUSE_ALLREDUCE_DISPATCH_H_
#define SRC_UCL_COMM_FUSE_ALLREDUCE_DISPATCH_H_

#include <cuda_runtime_api.h>

namespace hpc {
namespace ucl_comm {

void fuse_allreduce_dispatch_async(const void *input_ptr, const void *mc_input_ptr,
                                   const void *mn_input_ptr, void *output_ptr, void *mc_output_ptr,
                                   void *mn_output_ptr, void *signal_ptr,
                                   void *output_multinode_signal_ptr, int64_t rank,
                                   int64_t local_size, int64_t world_size, int64_t attn_dp_size,
                                   int64_t attn_tp_size, int64_t moe_ep_size, int64_t moe_tp_size,
                                   int64_t num_max_blocks, int num_tokens, int hidden_size,
                                   int world_rank, int batch_szie, int num_qp, cudaStream_t stream);

}  // namespace ucl_comm
}  // namespace hpc

#endif  // SRC_UCL_COMM_FUSE_ALLREDUCE_DISPATCH_H_
