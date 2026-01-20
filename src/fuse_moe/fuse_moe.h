// Copyright (C) 2026 Tencent.

#ifndef SRC_FUSE_MOE_FUSE_MOE_H_
#define SRC_FUSE_MOE_FUSE_MOE_H_

#include <cuda_runtime_api.h>
#include <stdint.h>

#include "src/activation/activation.h"
#include "src/group_gemm/group_gemm.h"

namespace hpc {
namespace fuse_moe {

void count_and_gather_async(void *gate_up_input_ptr, void *gata_up_output_ptr, void *down_input_ptr,
                            void *down_output_ptr, const void *x_ptr, const void *topk_ids_ptr,
                            void *topk_pos_ptr, void *seqlens_ptr, void *cu_seqlens_ptr,
                            void *gate_up_tmas_ptr, void *down_tmas_ptr, void *tiles_ptr,
                            void *cu_tiles_ptr, int num_seq, int hidden_size, int intermediate_size,
                            int num_topk, int num_expert, int rank_ep, int num_seq_per_group_avg,
                            cudaStream_t stream);

void blockwise_count_and_gather_async(
    const void *input_ptr, const void *input_scale_ptr, void *gate_up_input_ptr,
    void *gate_up_output_ptr, void *gate_up_input_scale_ptr, void *down_input_ptr,
    void *down_output_ptr, const void *topk_ids_ptr, void *topk_pos_ptr,
    void *num_tokens_per_group_ptr, void *cu_num_tokens_per_group_ptr, void *gate_up_tmas_ptr,
    void *down_tmas_ptr, void *tiles_ptr, void *cu_tiles_ptr, int num_tokens, int num_padded_tokens,
    int hidden_size, int intermediate_size, int num_topk, int num_expert_local, int eprank,
    int num_tokens_per_group_avg, cudaStream_t stream);

void reduce_async(void *y_ptr, const void *x_ptr, const void *topk_pos_ptr,
                  const void *topk_scale_ptr, const void *shared_output_ptr, int total_num_seq,
                  int num_seq, int hidden_size, int num_topk, cudaStream_t stream);

void fuse_moe_pertensor_fp8_async(
    void *output_ptr, const void *input_ptr, void *gate_up_input_ptr, void *gate_up_output_ptr,
    const void *gate_up_weight_ptr, const void *gate_up_scale_ptr, void *gate_up_tmas_ptr,
    const void *act_and_mul_scale_ptr, void *down_input_ptr, void *down_output_ptr,
    const void *down_weight_ptr, const void *down_scale_ptr, void *down_tmas_ptr,
    const void *topk_ids_ptr, const void *topk_scale_ptr, void *topk_pos_ptr, void *seqlens_ptr,
    void *cu_seqlens_ptr, void *tiles_ptr, void *cu_tiles_ptr, const void *shared_output_ptr,
    int num_seq, int hidden_size, int intermediate_size, int num_topk, int num_expert_total,
    int num_expert_local, int rank_ep, bool use_bf16_mul, cudaStream_t stream);

void fuse_moe_blockwise_fp8_async(
    void *output_ptr, const void *input_ptr, const void *input_scale_ptr, void *gate_up_input_ptr,
    void *gate_up_input_scale_ptr, void *gate_up_output_ptr, const void *gate_up_weight_ptr,
    const void *gate_up_weight_scale_ptr, void *gate_up_tmas_ptr, void *down_input_ptr,
    void *down_input_scale_ptr, void *down_output_ptr, const void *down_weight_ptr,
    const void *down_weight_scale_ptr, void *down_tmas_ptr, const void *topk_ids_ptr,
    const void *topk_scale_ptr, void *topk_pos_ptr, void *num_tokens_per_group_ptr,
    void *cu_num_tokens_per_group_ptr, void *tiles_ptr, void *cu_tiles_ptr,
    const void *shared_output_ptr, int num_tokens, int num_padded_tokens, int hidden_size,
    int intermediate_size, int num_topk, int num_expert_total, int num_expert_local,
    int gate_up_weight_scale_lastdim_pad4, int down_weight_scale_lastdim_pad4, int rank_ep,
    cudaStream_t stream);

}  // namespace fuse_moe
}  // namespace hpc

#endif  // SRC_FUSE_MOE_FUSE_MOE_H_
