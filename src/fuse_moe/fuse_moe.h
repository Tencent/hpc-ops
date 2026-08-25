// Copyright (C) 2026 Tencent.

#ifndef SRC_FUSE_MOE_FUSE_MOE_H_
#define SRC_FUSE_MOE_FUSE_MOE_H_

#include <cuda_runtime_api.h>
#include <stdint.h>

#include "src/activation/activation.h"
#include "src/group_gemm/group_gemm.h"

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
    int rank_ep, float swiglu_limit, cudaStream_t stream);

void fuse_moe_blockwise_indexed_direct_async(
    void *output_ptr, const void *input_ptrs_dev, const void *input_scale_ptrs_dev,
    int rows_per_shard, int num_input_ptrs, const void *source_rows_ptr, int64_t input_row_stride,
    int64_t input_scale_row_stride, void *row_indices_ptr, void *gate_up_output_ptr,
    const void *gate_up_weight_ptr, const void *gate_up_weight_scale_ptr, void *down_input_ptr,
    void *down_input_scale_ptr, void *down_output_ptr, const void *down_weight_ptr,
    const void *down_weight_scale_ptr, void *down_tmas_ptr, const void *topk_ids_ptr,
    const void *topk_scale_ptr, void *topk_pos_ptr, void *num_tokens_per_group_ptr,
    void *cu_num_tokens_per_group_ptr, void *tiles_ptr, void *cu_tiles_ptr,
    void *gateup_task_map_ptr, int num_gateup_waves, int num_tokens, int num_padded_tokens,
    int hidden_size, int intermediate_size, int num_topk, int num_expert_total,
    int num_expert_local, int gate_up_weight_scale_lastdim_pad4, int down_weight_scale_lastdim_pad4,
    int rank_ep, float swiglu_limit, cudaStream_t stream);

void pull_indexed_rows_async(const void *input_ptrs_dev, const void *input_scale_ptrs_dev,
                             int rows_per_shard, int num_input_ptrs, int64_t input_row_stride,
                             int64_t input_scale_row_stride, void *row_indices_ptr,
                             const void *cu_num_tokens_per_group_ptr, const void *cu_tiles_ptr,
                             void *output_ptr, void *output_scale_ptr, int max_rows,
                             int hidden_size, int scale_groups, int num_padded_tokens,
                             int num_expert_local, int tile_m, cudaStream_t stream);

void fuse_moe_blockwise_indexed_pull_async(
    void *output_ptr, const void *input_ptrs_dev, const void *input_scale_ptrs_dev,
    int rows_per_shard, int num_input_ptrs, const void *source_rows_ptr, int64_t input_row_stride,
    int64_t input_scale_row_stride, void *row_indices_ptr, void *gate_up_input_ptr,
    void *gate_up_input_scale_ptr, void *gate_up_output_ptr, const void *gate_up_weight_ptr,
    const void *gate_up_weight_scale_ptr, void *gate_up_tmas_ptr, void *down_input_ptr,
    void *down_input_scale_ptr, void *down_output_ptr, const void *down_weight_ptr,
    const void *down_weight_scale_ptr, void *down_tmas_ptr, const void *topk_ids_ptr,
    const void *topk_scale_ptr, void *topk_pos_ptr, void *num_tokens_per_group_ptr,
    void *cu_num_tokens_per_group_ptr, void *tiles_ptr, void *cu_tiles_ptr,
    void *gateup_task_map_ptr, int num_gateup_waves, int num_tokens, int num_padded_tokens,
    int aligned_size, int hidden_size, int intermediate_size, int num_topk, int num_expert_total,
    int num_expert_local, int gate_up_weight_scale_lastdim_pad4, int down_weight_scale_lastdim_pad4,
    int rank_ep, float swiglu_limit, cudaStream_t stream);

}  // namespace fuse_moe
}  // namespace hpc

#endif  // SRC_FUSE_MOE_FUSE_MOE_H_
