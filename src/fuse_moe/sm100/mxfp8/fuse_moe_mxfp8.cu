// Copyright 2026 hpc-ops authors

#include <cuda.h>

#include "src/fuse_moe/sm100/fuse_moe.h"  // for reduce_async (reused from fp8 path)
#include "src/fuse_moe/sm100/mxfp8/fuse_moe_mxfp8.h"
#include "src/group_gemm/sm100/group_gemm.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace fuse_moe {

void fuse_moe_mxfp8_async(void *output_ptr, const void *x_ptr, const void *x_scale_ptr,
                          void *gate_up_input_ptr, void *gate_up_output_ptr,
                          const void *gate_up_weight_ptr,
                          const void *gate_up_weight_scale_packed_ptr, void *gate_up_tmas_ptr,
                          void *down_input_ptr, void *down_input_scale_ptr, void *down_output_ptr,
                          const void *down_weight_ptr, const void *down_weight_scale_packed_ptr,
                          void *down_tmas_ptr, const void *topk_ids_ptr, const void *topk_scale_ptr,
                          void *topk_pos_ptr, void *gateup_x_row_map_ptr, void *seqlens_ptr,
                          void *cu_seqlens_ptr, void *tiles_ptr, void *cu_tiles_ptr,
                          const void *shared_output_ptr, int num_seq, int hidden_size,
                          int intermediate_size, int num_topk, int num_expert_total,
                          int num_expert_local, int rank_ep, bool is_fp4, cudaStream_t stream) {
  const int total_num_seq = num_seq * num_topk;
  const int num_seq_per_group_avg = total_num_seq / num_expert_total;

  // tiles / cu_tiles workspaces are split in half for gateup and down GEMMs.
  void *gateup_tiles_ptr = static_cast<int *>(tiles_ptr);
  void *down_tiles_ptr = static_cast<int *>(tiles_ptr) + num_expert_local;
  void *gateup_cu_tiles_ptr = static_cast<int *>(cu_tiles_ptr);
  void *down_cu_tiles_ptr = static_cast<int *>(cu_tiles_ptr) + (num_expert_local + 1);

  bool fuse_gate_up_with_act_mul = false;
  // 1. Route tokens: write topk_pos and gateup_x_row_map; update per-group X/Y TMA
  //    descriptors; compute seqlens / cu_seqlens / tiles / cu_tiles.
  count_and_route_row_mxfp8_async(
      gate_up_input_ptr, gate_up_output_ptr, down_input_ptr, down_output_ptr, topk_ids_ptr,
      topk_pos_ptr, topk_scale_ptr, gateup_x_row_map_ptr, seqlens_ptr, cu_seqlens_ptr,
      gateup_tiles_ptr, gateup_cu_tiles_ptr, down_tiles_ptr, down_cu_tiles_ptr, gate_up_tmas_ptr,
      down_tmas_ptr, num_seq, num_topk, hidden_size, intermediate_size, num_expert_local, rank_ep,
      num_seq_per_group_avg, stream);

  if (fuse_gate_up_with_act_mul) {
    // 2&3. Fused Gateup GEMM with act_mul_and_mxfp8_quant
    group_gemm::group_gemm_cp_async_mxfp8_act_mul_async(
        gate_up_output_ptr,  // unused
        down_input_ptr,      // y_fp8_ptr
        x_ptr,
        gate_up_weight_ptr,               // interleave
        x_scale_ptr,                      // raw sfx
        gate_up_weight_scale_packed_ptr,  // interleave
        seqlens_ptr, cu_seqlens_ptr, gate_up_tmas_ptr, gateup_tiles_ptr, gateup_cu_tiles_ptr,
        down_input_scale_ptr,  // out_scale_ptr
        num_expert_local, total_num_seq,
        intermediate_size * 2,  // n
        hidden_size,            // k
        num_seq_per_group_avg,
        /*update_tma=*/false, stream, gateup_x_row_map_ptr, num_seq,  // x_row_map + x_num_rows
        is_fp4);
  } else {
    // 2. Gateup GEMM via cp_async (N = intermediate_size * 2, K = hidden_size)
    //    SFA is loaded inline by the kernel via cp.async with x_row_map indirection.
    group_gemm::group_gemm_cp_async_mxfp8_async(
        gate_up_output_ptr, x_ptr, gate_up_weight_ptr, x_scale_ptr, gate_up_weight_scale_packed_ptr,
        seqlens_ptr, cu_seqlens_ptr, gate_up_tmas_ptr, gateup_tiles_ptr, gateup_cu_tiles_ptr,
        num_expert_local, total_num_seq, intermediate_size * 2, hidden_size, num_seq_per_group_avg,
        /*update_tma=*/false, stream, gateup_x_row_map_ptr, num_seq,
        /*is_fp4=*/is_fp4);

    // 3. Activation + mxfp8 quant: silu(gate) * up → fp8, writes row-major scale.
    //    Down GEMM kernel does inline prepack of SFA via cp.async.
    const int *valid_row_range_ptr = static_cast<const int *>(cu_seqlens_ptr) + num_expert_local;
    act_mul_and_mxfp8_quant_async(down_input_ptr, down_input_scale_ptr, gate_up_output_ptr,
                                  valid_row_range_ptr, total_num_seq, intermediate_size, stream,
                                  true);
  }

  // 4. Down GEMM (N = hidden_size, K = intermediate_size) + reduce.
  //    x_row_map and topk_scale_row_map are packed in gateup_x_row_map_ptr buffer:
  //      [x_row_map: int[total_num_seq]] [topk_scale_row_map: float[total_num_seq]] [...]
  void *x_row_map_ptr = gateup_x_row_map_ptr;
  void *topk_scale_row_map_ptr =
      reinterpret_cast<void *>(reinterpret_cast<float *>(gateup_x_row_map_ptr) + total_num_seq);

  //    Down GEMM + reduce: atomic scatter-add with topk_scale weights into output.
  //    Caller guarantees output_ptr is already initialized:
  //      - When shared_output is provided, output_ptr == shared_output_ptr (accumulate on top).
  //      - Otherwise, caller zeros output_ptr before calling fuse_moe_mxfp8_async.
  group_gemm::group_gemm_mxfp8_with_reduce_async(
      output_ptr, down_input_ptr, down_weight_ptr, down_input_scale_ptr,
      down_weight_scale_packed_ptr, seqlens_ptr, cu_seqlens_ptr, down_tmas_ptr, down_tiles_ptr,
      down_cu_tiles_ptr, x_row_map_ptr, topk_scale_row_map_ptr, num_expert_local, total_num_seq,
      hidden_size, intermediate_size, num_seq_per_group_avg, /*update_tma=*/false, stream,
      /*is_fp4=*/is_fp4);
}

}  // namespace fuse_moe
}  // namespace hpc
