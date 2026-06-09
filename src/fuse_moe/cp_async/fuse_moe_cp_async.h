// Copyright (C) 2026 Tencent.

#ifndef SRC_FUSE_MOE_CP_ASYNC_FUSE_MOE_CP_ASYNC_H_
#define SRC_FUSE_MOE_CP_ASYNC_FUSE_MOE_CP_ASYNC_H_

#include <cuda_runtime_api.h>

namespace hpc {
namespace fuse_moe_cp_async {

// Count tokens per expert and build auxiliary index arrays for the cp.async path.
// Optional task maps use int4 {igroup, itile_m, itile_n, 0}; unused entries use igroup = -1.
void count_and_build_indices_async(const void *topk_ids_ptr, void *row_indices_ptr,
                                   void *topk_pos_ptr, void *seqlens_ptr, void *cu_seqlens_ptr,
                                   void *tiles_ptr, void *cu_tiles_ptr, void *gateup_task_map_ptr,
                                   void *down_task_map_ptr, int gate_up_num_tile_n,
                                   int down_num_tile_n, int gateup_task_map_len,
                                   int down_task_map_len, int num_seq, int num_topk, int num_expert,
                                   int eprank, int num_seq_per_group_avg, cudaStream_t stream);

// Full fused MoE forward pass using cp.async-based group GEMM kernels.
void fuse_moe_cp_async(void *output_ptr, const void *input_ptr, void *gate_up_output_ptr,
                       const void *gate_up_weight_ptr, const void *gate_up_scale_ptr,
                       const void *act_and_mul_scale_ptr, void *down_input_ptr,
                       void *down_output_ptr, const void *down_weight_ptr,
                       const void *down_scale_ptr, const void *topk_ids_ptr,
                       const void *topk_scale_ptr, void *topk_pos_ptr, void *row_indices_ptr,
                       void *seqlens_ptr, void *cu_seqlens_ptr, void *tiles_ptr, void *cu_tiles_ptr,
                       void *gateup_task_map_ptr, void *down_task_map_ptr, int gateup_task_map_len,
                       int down_task_map_len, const void *shared_output_ptr, int num_seq,
                       int hidden_size, int intermediate_size, int num_topk, int num_expert_total,
                       int num_expert_local, int rank_ep, bool use_bf16_mul, cudaStream_t stream);

}  // namespace fuse_moe_cp_async
}  // namespace hpc

#endif  // SRC_FUSE_MOE_CP_ASYNC_FUSE_MOE_CP_ASYNC_H_
