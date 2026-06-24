// Copyright 2026 hpc-ops authors

#ifndef SRC_FUSE_MOE_SM100_MXFP8_FUSE_MOE_MXFP8_H_
#define SRC_FUSE_MOE_SM100_MXFP8_FUSE_MOE_MXFP8_H_

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace hpc {
namespace fuse_moe {

void count_and_route_row_mxfp8_async(
    void *gate_up_input_ptr, void *gate_up_output_ptr, void *down_input_ptr, void *down_output_ptr,
    const void *topk_ids_ptr, void *topk_pos_ptr, const void *topk_scale_ptr,
    void *gateup_x_row_map_ptr, void *seqlens_ptr, void *cu_seqlens_ptr, void *gateup_tiles_ptr,
    void *gateup_cu_tiles_ptr, void *down_tiles_ptr, void *down_cu_tiles_ptr,
    void *gate_up_tmas_ptr, void *down_tmas_ptr, int num_seq, int num_topk, int hidden_size,
    int intermediate_size, int num_expert_local, int rank_ep, int num_seq_per_group_avg,
    cudaStream_t stream);

void act_mul_and_mxfp8_quant_async(void *out_ptr, void *out_scale_ptr, const void *gate_up_ptr,
                                   const void *valid_row_range_ptr, int total_num_seq,
                                   int intermediate_size, cudaStream_t stream,
                                   bool use_pdl = false);

void fuse_moe_mxfp8_async(void *output_ptr,
                          // input
                          const void *x_ptr, const void *x_scale_ptr,
                          // gateup
                          void *gate_up_input_ptr, void *gate_up_output_ptr,
                          const void *gate_up_weight_ptr,
                          const void *gate_up_weight_scale_packed_ptr, void *gate_up_tmas_ptr,
                          // down
                          void *down_input_ptr, void *down_input_scale_ptr, void *down_output_ptr,
                          const void *down_weight_ptr, const void *down_weight_scale_packed_ptr,
                          void *down_tmas_ptr,
                          // routing / scratch
                          const void *topk_ids_ptr, const void *topk_scale_ptr, void *topk_pos_ptr,
                          void *gateup_x_row_map_ptr, void *seqlens_ptr, void *cu_seqlens_ptr,
                          void *tiles_ptr, void *cu_tiles_ptr, const void *shared_output_ptr,
                          int num_seq, int hidden_size, int intermediate_size, int num_topk,
                          int num_expert_total, int num_expert_local, int rank_ep, bool is_fp4,
                          cudaStream_t stream);

}  // namespace fuse_moe
}  // namespace hpc

#endif  // SRC_FUSE_MOE_SM100_MXFP8_FUSE_MOE_MXFP8_H_
