// Copyright (C) 2026 Tencent.

#ifndef SRC_FUSE_MOE_FUSE_MOE_H_
#define SRC_FUSE_MOE_FUSE_MOE_H_

#include <cuda_runtime_api.h>
#include <stdint.h>

#include "src/activation/activation.h"
#include "src/group_gemm/group_gemm.h"
#include "src/utils/utils.h"

namespace hpc {
namespace fuse_moe {

void count_and_gather_async(void *gate_up_input_ptr, void *gate_up_output_ptr, void *down_input_ptr,
                            void *down_output_ptr, const void *x_ptr, const void *topk_ids_ptr,
                            void *topk_pos_ptr, void *seqlens_ptr, void *cu_seqlens_ptr,
                            void *gate_up_tmas_ptr, void *down_tmas_ptr, void *tiles_ptr,
                            void *cu_tiles_ptr, void *gateup_task_map_ptr, void *down_task_map_ptr,
                            int num_seq, int hidden_size, int intermediate_size, int num_topk,
                            int num_expert, int eprank, int num_seq_per_group_avg,
                            cudaStream_t stream);

void blockwise_count_and_gather_async(
    const void *input_ptr, const void *input_scale_ptr, void *gate_up_input_ptr,
    void *gate_up_output_ptr, void *gate_up_input_scale_ptr, void *down_input_ptr,
    void *down_output_ptr, const void *topk_ids_ptr, void *topk_pos_ptr,
    void *num_tokens_per_group_ptr, void *cu_num_tokens_per_group_ptr, void *gate_up_tmas_ptr,
    void *down_tmas_ptr, void *tiles_ptr, void *cu_tiles_ptr, void *gateup_task_map_ptr,
    void *down_task_map_ptr, int num_tokens, int num_padded_tokens, int hidden_size,
    int intermediate_size, int num_topk, int num_expert_local, int eprank,
    int num_tokens_per_group_avg, bool use_pdl, cudaStream_t stream);

void reduce_async(void *y_ptr, const void *x_ptr, const void *topk_pos_ptr,
                  const void *topk_scale_ptr, const void *shared_output_ptr, int total_num_seq,
                  int num_seq, int hidden_size, int num_topk, bool use_pdl, cudaStream_t stream);

void fuse_moe_async(void *output_ptr, const void *input_ptr, void *gate_up_input_ptr,
                    void *gate_up_output_ptr, const void *gate_up_weight_ptr,
                    const void *gate_up_scale_ptr, void *gate_up_tmas_ptr,
                    const void *act_and_mul_scale_ptr, void *down_input_ptr, void *down_output_ptr,
                    const void *down_weight_ptr, const void *down_scale_ptr, void *down_tmas_ptr,
                    const void *topk_ids_ptr, const void *topk_scale_ptr, void *topk_pos_ptr,
                    void *seqlens_ptr, void *cu_seqlens_ptr, void *tiles_ptr, void *cu_tiles_ptr,
                    const void *shared_output_ptr, void *gateup_task_map_ptr,
                    void *down_task_map_ptr, int num_gateup_waves, int num_down_waves, int num_seq,
                    int hidden_size, int intermediate_size, int num_topk, int num_expert_total,
                    int num_expert_local, int rank_ep, bool use_bf16_mul, cudaStream_t stream);

void fuse_moe_blockwise_async(
    void *output_ptr, const void *input_ptr, const void *input_scale_ptr, void *gate_up_input_ptr,
    void *gate_up_input_scale_ptr, void *gate_up_output_ptr, const void *gate_up_weight_ptr,
    const void *gate_up_weight_scale_ptr, void *gate_up_tmas_ptr, void *down_input_ptr,
    void *down_input_scale_ptr, void *down_output_ptr, const void *down_weight_ptr,
    const void *down_weight_scale_ptr, void *down_tmas_ptr, const void *topk_ids_ptr,
    const void *topk_scale_ptr, void *topk_pos_ptr, void *num_tokens_per_group_ptr,
    void *cu_num_tokens_per_group_ptr, void *tiles_ptr, void *cu_tiles_ptr,
    const void *shared_output_ptr, void *gateup_task_map_ptr, void *down_task_map_ptr,
    int num_gateup_waves, int num_down_waves, int num_tokens, int num_padded_tokens,
    int hidden_size, int intermediate_size, int num_topk, int num_expert_total,
    int num_expert_local, int gate_up_weight_scale_lastdim_pad4, int down_weight_scale_lastdim_pad4,
    int rank_ep, cudaStream_t stream);

// MX fused MoE: MXFP4 or MXFP8 weights with UE8M0 scales in the packed int32
// layout, activations quantized to MXFP8. fuse_moe_mx_plan picks the launch
// configuration and the workspace extents for fuse_moe_mx_async.
struct FuseMoeMxPlan {
  int tile_m;
  int gate_up_tile_n;
  int down_tile_n;
  int gate_up_num_ctas;
  // Routed rows the workspaces hold, each expert padded to tile_m.
  int max_padded_rows;
  // Rows of the packed activation scales.
  int scale_m_padded;
  // The routing accumulates into route_counts, which must start at zero.
  bool zero_route_counts;
  // The gate_up GEMM zeroes the output instead of the routing.
  bool zero_output_in_gate_up_gemm;
  // Output bytes each gate_up tile zeroes when zero_output_in_gate_up_gemm.
  int64_t zero_output_chunk;
  // TMA L2 cache hints for both GEMMs' activation and weight loads.
  uint64_t cache_hint_activation;
  uint64_t cache_hint_weight;
};

FuseMoeMxPlan fuse_moe_mx_plan(int num_tokens, int num_topk, int num_expert_local,
                               int num_expert_total, int hidden_size, int down_input_size,
                               bool mxfp4_weight, bool has_shared_output);

// input_scale_ptr is null for BF16 input. Otherwise the input is E4M3 with uint8
// UE8M0 scales per 32 columns, or with block128_scale float32 powers of two per
// 128 columns. When init_output, the routing first sets the output to a copy of
// shared_output_ptr, or to zeros when that is null.
void fuse_moe_mx_async(void *output_ptr, const void *input_ptr, const void *input_scale_ptr,
                       bool block128_scale, const void *gate_up_weight_ptr,
                       const void *gate_up_weight_scale_ptr, const void *down_weight_ptr,
                       const void *down_weight_scale_ptr, const void *topk_ids_ptr,
                       const void *topk_scale_ptr, const void *shared_output_ptr, bool init_output,
                       void *topk_pos_ptr, void *row_to_route_ptr, void *num_tokens_per_group_ptr,
                       void *cu_padded_tokens_per_group_ptr, void *cu_tiles_ptr,
                       void *route_counts_ptr, void *gate_up_input_ptr,
                       void *gate_up_input_scale_ptr, void *down_input_ptr,
                       void *down_input_scale_ptr, int num_tokens, int num_topk, int hidden_size,
                       int down_input_size, int num_expert_local, int rank_ep, bool mxfp4_weight,
                       float swiglu_limit, const FuseMoeMxPlan &plan, cudaStream_t stream);

}  // namespace fuse_moe
}  // namespace hpc

#endif  // SRC_FUSE_MOE_FUSE_MOE_H_
