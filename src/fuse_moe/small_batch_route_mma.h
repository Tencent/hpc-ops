// Copyright (C) 2026 Tencent.

#ifndef SRC_FUSE_MOE_SMALL_BATCH_ROUTE_MMA_H_
#define SRC_FUSE_MOE_SMALL_BATCH_ROUTE_MMA_H_

#include <cuda_runtime_api.h>

namespace hpc {
namespace fuse_moe {

// SM90 route-direct implementation: Gate/Up WGMMA can split K for small-batch
// occupancy, then one kernel reduces the partials, applies SwiGLU, and
// requantizes for route-direct Down WGMMA. intermediate_ptr is a byte workspace
// containing BF16 Gate/Up partials, FP8 Down input, and BF16 route output.
void fuse_moe_small_batch_route_mma_async(
    void *output_ptr, const void *input_ptr, void *intermediate_ptr,
    const void *gate_up_weight_ptr, const void *gate_up_scale_ptr,
    const void *act_and_mul_scale_ptr, const void *down_weight_ptr,
    const void *down_scale_ptr, const void *topk_ids_ptr, const void *topk_scale_ptr,
    const void *shared_output_ptr, int num_seq, int hidden_size, int intermediate_size,
    int num_topk, int num_splits, int num_expert_local, int rank_ep, bool use_bf16_mul,
    cudaStream_t stream);

void fuse_moe_blockwise_small_batch_route_mma_async(
    void *output_ptr, const void *input_ptr, const void *input_scale_ptr,
    void *workspace_ptr, const void *gate_up_weight_ptr,
    const void *gate_up_weight_scale_ptr, const void *down_weight_ptr,
    const void *down_weight_scale_ptr, const void *topk_ids_ptr,
    const void *topk_scale_ptr, const void *shared_output_ptr, int num_tokens,
    int hidden_size, int intermediate_size, int num_topk, int num_splits,
    int num_expert_local, int gate_up_weight_scale_lastdim_pad4,
    int down_weight_scale_lastdim_pad4, int rank_ep, cudaStream_t stream);

}  // namespace fuse_moe
}  // namespace hpc

#endif  // SRC_FUSE_MOE_SMALL_BATCH_ROUTE_MMA_H_
