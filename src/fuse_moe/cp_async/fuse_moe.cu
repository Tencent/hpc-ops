// Copyright (C) 2026 Tencent.

#include <cuda.h>

#include "src/activation/activation.h"
#include "src/fuse_moe/cp_async/fuse_moe_cp_async.h"
#include "src/fuse_moe/fuse_moe.h"  // for reduce_async
#include "src/group_gemm/cp_async/group_gemm.h"

namespace hpc {
namespace fuse_moe_cp_async {

// kTileN of the main gemm (fixed at 64 in group_gemm_cp_async).
static constexpr int kGemmTileN = 64;

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
                       int num_expert_local, int rank_ep, bool use_bf16_mul, cudaStream_t stream) {
  using T1 = __nv_bfloat16;
  using T2 = __nv_fp8_e4m3;

  int total_num_seq = num_seq * num_topk;
  int num_seq_per_group_avg = total_num_seq / num_expert_total;

  int gate_up_num_tile_n = (intermediate_size + kGemmTileN - 1) / kGemmTileN;
  int down_num_tile_n = (hidden_size + kGemmTileN - 1) / kGemmTileN;

  // When task_map pointers are non-null, this call also fills both task_maps
  // and writes sentinel -1 into their unused tail slots.
  count_and_build_indices_async(topk_ids_ptr, row_indices_ptr, topk_pos_ptr, seqlens_ptr,
                                cu_seqlens_ptr, tiles_ptr, cu_tiles_ptr, gateup_task_map_ptr,
                                down_task_map_ptr, gate_up_num_tile_n, down_num_tile_n,
                                gateup_task_map_len, down_task_map_len, num_seq, num_topk,
                                num_expert_local, rank_ep, num_seq_per_group_avg, stream);

  group_gemm_cp_async::group_gemm_fp8_scatter_async(
      gate_up_output_ptr, input_ptr, gate_up_weight_ptr, gate_up_scale_ptr, row_indices_ptr,
      seqlens_ptr, cu_seqlens_ptr, tiles_ptr, cu_tiles_ptr, gateup_task_map_ptr,
      gateup_task_map_len, total_num_seq, intermediate_size, hidden_size, num_expert_local,
      num_seq_per_group_avg, /*use_pdl=*/true, stream);

  const int *valid_row_range_ptr = (const int *)cu_seqlens_ptr + num_expert_local;
  activation::act_mul_and_quant_async((T2 *)down_input_ptr, (const T1 *)gate_up_output_ptr,
                                      (const float *)act_and_mul_scale_ptr, valid_row_range_ptr,
                                      total_num_seq, intermediate_size, use_bf16_mul, stream);

  group_gemm_cp_async::group_gemm_fp8_multistage_async(
      down_output_ptr, down_input_ptr, down_weight_ptr, down_scale_ptr, seqlens_ptr, cu_seqlens_ptr,
      tiles_ptr, cu_tiles_ptr, down_task_map_ptr, down_task_map_len, total_num_seq, hidden_size,
      intermediate_size / 2, num_expert_local, num_seq_per_group_avg, /*use_pdl=*/true, stream);

  fuse_moe::reduce_async(output_ptr, down_output_ptr, topk_pos_ptr, topk_scale_ptr,
                         shared_output_ptr, total_num_seq, num_seq, hidden_size, num_topk,
                         /*use_pdl=*/true, stream);
}

}  // namespace fuse_moe_cp_async
}  // namespace hpc
