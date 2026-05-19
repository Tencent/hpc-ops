// Copyright 2026 hpc-ops authors

#ifndef SRC_FUSE_MOE_SM90_CP_ASYNC_FUSE_MOE_CP_ASYNC_H_
#define SRC_FUSE_MOE_SM90_CP_ASYNC_FUSE_MOE_CP_ASYNC_H_

#include <cuda_runtime_api.h>
#include <stdint.h>

#include "src/activation/activation.h"
#include "src/group_gemm/sm90/cp_async/group_gemm.h"

namespace hpc {
namespace fuse_moe_cp_async {

// Count tokens per expert and build auxiliary index arrays.
// Outputs:
//   seqlens_ptr      : int32[num_expert]          - tokens per expert (zeroed after use)
//   cu_seqlens_ptr   : int32[num_expert+1]         - exclusive prefix sum of seqlens
//   tiles_ptr        : int32[num_expert]           - tile count per expert (ceil(seqlens/kTileM))
//   cu_tiles_ptr     : int32[num_expert+1]         - exclusive prefix sum of tiles
//   row_indices_ptr  : int32[total_num_topk]       - original token row for each sorted slot
//                      row_indices[cu_seqlens[e]+k] = original token index for k-th token at expert
//                      e
//   topk_pos_ptr     : int32[num_seq * num_topk]   - sorted position for each (token, topk) pair
//                      topk_pos[i*num_topk+j] = position in sorted buffer for token i's j-th expert
//
// task_map outputs (optional — pass nullptr to skip):
//   gateup_task_map_ptr : int4[ceil(total_tokens/kTileM + num_expert) * gate_up_num_tile_n]
//   down_task_map_ptr   : int4[ceil(total_tokens/kTileM + num_expert) * down_num_tile_n]
//   Each int4 = {igroup, itile_m, itile_n, 0}.  Tail entries are sentinel igroup=-1.
void count_and_build_indices_async(const void *topk_ids_ptr, void *row_indices_ptr,
                                   void *topk_pos_ptr, void *seqlens_ptr, void *cu_seqlens_ptr,
                                   void *tiles_ptr, void *cu_tiles_ptr, void *gateup_task_map_ptr,
                                   void *down_task_map_ptr, int gate_up_num_tile_n,
                                   int down_num_tile_n, int gateup_task_map_len,
                                   int down_task_map_len, int num_seq, int num_topk, int num_expert,
                                   int eprank, int num_seq_per_group_avg, cudaStream_t stream);

// Full fused MoE forward pass using cp.async-based group GEMM kernels (SM80+).
// Pipeline:
//   1. count_and_build_indices: count tokens per expert, build row_indices, topk_pos,
//      and (if task_map pointers are provided) the per-gemm task_maps.
//   2. group_gemm_fp8_scatter_async (gate_up): x[row_indices] @ gate_up_weight → gate_up_output
//   3. act_mul_and_quant_async: silu(gate) * up → fp8 down_input
//   4. group_gemm_fp8_multistage_async (down): down_input @ down_weight → down_output
//   5. reduce_async: weighted sum by topk_scale → output
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

#endif  // SRC_FUSE_MOE_SM90_CP_ASYNC_FUSE_MOE_CP_ASYNC_H_
