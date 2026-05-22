// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "src/fuse_moe/sm100/fuse_moe.h"

namespace hpc {
namespace fuse_moe {

void fuse_moe_async(void *output_ptr, const void *input_ptr, void *gate_up_input_ptr,
                    void *gate_up_output_ptr, const void *gate_up_weight_ptr,
                    const void *gate_up_scale_ptr, void *gate_up_tmas_ptr,
                    const void *act_and_mul_scale_ptr, void *down_input_ptr, void *down_output_ptr,
                    const void *down_weight_ptr, const void *down_scale_ptr, void *down_tmas_ptr,
                    const void *topk_ids_ptr, const void *topk_scale_ptr, void *topk_pos_ptr,
                    void *seqlens_ptr, void *cu_seqlens_ptr, void *tiles_ptr, void *cu_tiles_ptr,
                    const void *shared_output_ptr, int num_gateup_waves, int num_down_waves,
                    int num_seq, int hidden_size, int intermediate_size, int num_topk,
                    int num_expert_total, int num_expert_local, int rank_ep, bool use_bf16_mul,
                    cudaStream_t stream, void *x_row_map_ptr) {
  int total_num_seq = num_seq * num_topk;
  int num_seq_per_group_avg = total_num_seq / num_expert_total;
  using T1 = __nv_bfloat16;
  using T2 = __nv_fp8_e4m3;

  bool use_pdl = true;
  const bool skip_token_copy = (x_row_map_ptr != nullptr);
  const bool fuse_act = true;

  const void *gate_up_x_ptr = skip_token_copy ? input_ptr : gate_up_input_ptr;

  int gateup_k = hidden_size;
  int gateup_n = fuse_act ? intermediate_size / 2 : intermediate_size;

  int down_k = intermediate_size / 2;
  int down_n = hidden_size;

  // 0. call count_and_gather_async (produces topk_pos and optionally x_row_map)
  count_and_gather_async(topk_ids_ptr, topk_pos_ptr, topk_scale_ptr, seqlens_ptr, cu_seqlens_ptr,
                         tiles_ptr, cu_tiles_ptr, gate_up_tmas_ptr, down_tmas_ptr, gate_up_x_ptr,
                         fuse_act ? down_input_ptr : gate_up_output_ptr, down_input_ptr,
                         down_output_ptr, gateup_k, gateup_n, down_k, down_n, num_seq, num_topk,
                         num_expert_local, rank_ep, num_seq_per_group_avg, stream, x_row_map_ptr,
                         fuse_act);

  void *gateup_tiles_ptr = reinterpret_cast<void *>(reinterpret_cast<int32_t *>(tiles_ptr));
  void *down_tiles_ptr =
      reinterpret_cast<void *>(reinterpret_cast<int32_t *>(tiles_ptr) + num_expert_local);
  void *gateup_cu_tiles_ptr = reinterpret_cast<void *>(reinterpret_cast<int32_t *>(cu_tiles_ptr));
  void *down_cu_tiles_ptr =
      reinterpret_cast<void *>(reinterpret_cast<int32_t *>(cu_tiles_ptr) + num_expert_local + 1);
  void *topk_scale_row_map_ptr =
      reinterpret_cast<void *>(reinterpret_cast<float *>(x_row_map_ptr) + total_num_seq);
  if (fuse_act) {
    // 1. && 2. call gate_up linear FUSED with act_mul_and_quant.
    group_gemm::group_gemm_cp_async_fp8_act_mul_async(
        down_input_ptr, gate_up_x_ptr, gate_up_weight_ptr, seqlens_ptr, cu_seqlens_ptr,
        gate_up_scale_ptr, act_and_mul_scale_ptr, gate_up_tmas_ptr, gateup_tiles_ptr,
        gateup_cu_tiles_ptr, num_expert_local, total_num_seq, intermediate_size, hidden_size,
        num_seq_per_group_avg,
        /*update_tma=*/false, use_pdl, stream, skip_token_copy ? x_row_map_ptr : nullptr, num_seq);
  } else {
    // 1. call gate_up linear (group_gemm still takes task_map_ptr in its signature; pass nullptr
    // since the sm100 fuse_moe path no longer produces a task map).
    group_gemm::group_gemm_cp_async_fp8_async(
        gate_up_output_ptr, gate_up_x_ptr, gate_up_weight_ptr, seqlens_ptr, cu_seqlens_ptr,
        gate_up_scale_ptr, gate_up_tmas_ptr, gateup_tiles_ptr, gateup_cu_tiles_ptr,
        /*task_map=*/nullptr, num_gateup_waves, num_expert_local, total_num_seq, intermediate_size,
        hidden_size, num_seq_per_group_avg, false, use_pdl, stream,
        skip_token_copy ? x_row_map_ptr : nullptr, num_seq);

    // 2. call act and mul
    const int *valid_row_range_ptr =
        (int *)cu_seqlens_ptr + num_expert_local;  // get last number as valid row
    activation::act_mul_and_quant_async((T2 *)down_input_ptr, (const T1 *)gate_up_output_ptr,
                                        (const float *)act_and_mul_scale_ptr, valid_row_range_ptr,
                                        total_num_seq, intermediate_size, use_bf16_mul, stream);
  }

  // 3. call down linear
  if (shared_output_ptr && (num_seq_per_group_avg >= 128 || num_seq_per_group_avg <= 32)) {
    group_gemm::group_gemm_fp8_with_reduce_async(
        output_ptr, down_input_ptr, down_weight_ptr, seqlens_ptr, cu_seqlens_ptr, down_scale_ptr,
        down_tmas_ptr, down_tiles_ptr, down_cu_tiles_ptr, /*task_map=*/nullptr, x_row_map_ptr,
        topk_scale_row_map_ptr, num_down_waves, num_expert_local, total_num_seq, hidden_size,
        intermediate_size / 2, num_seq_per_group_avg, false, use_pdl, stream);
  } else {
    group_gemm::group_gemm_fp8_async(
        output_ptr, down_input_ptr, down_weight_ptr, seqlens_ptr, cu_seqlens_ptr, down_scale_ptr,
        down_tmas_ptr, down_tiles_ptr, down_cu_tiles_ptr, /*task_map=*/nullptr, num_down_waves,
        num_expert_local, total_num_seq, hidden_size, intermediate_size / 2, num_seq_per_group_avg,
        false, use_pdl, stream);

    // 4. call reduce
    reduce_async(output_ptr, down_output_ptr, topk_pos_ptr, topk_scale_ptr, shared_output_ptr,
                 total_num_seq, num_seq, hidden_size, num_topk, use_pdl, stream);
  }
}

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
    int rank_ep, cudaStream_t stream) {}

}  // namespace fuse_moe
}  // namespace hpc
