// Copyright (C) 2026 Tencent.

#include <cuda.h>

#include "src/fuse_moe/fuse_moe_mxfp8.h"

namespace hpc {
namespace fuse_moe {

bool fuse_moe_mxfp8_async(void *output_ptr, const void *x_ptr, const void *x_scale_ptr,
                          const void *gate_up_weight_ptr, const void *gate_up_weight_scale_ptr,
                          void *down_input_ptr, void *down_input_scale_ptr,
                          const void *down_weight_ptr, const void *down_weight_scale_ptr,
                          const void *topk_ids_ptr, const void *topk_scale_ptr,
                          void *permuted_idx_to_raw_idx_ptr, void *seqlens_ptr,
                          void *cu_seqlens_ptr, void *tiles_ptr, int num_seq, int hidden_size,
                          int intermediate_size, int num_topk, int num_expert_local, int rank_ep,
                          int max_tiles, int tile_m, float limit, bool is_fp4,
                          cudaStream_t stream) {
  // 1. Route tokens; gate_up and down share tile_m, so they share the per-expert tile counts.
  if (!count_and_route_mxfp8_async(topk_ids_ptr, permuted_idx_to_raw_idx_ptr, seqlens_ptr,
                                   cu_seqlens_ptr, tiles_ptr, num_seq, num_topk, num_expert_local,
                                   rank_ep, tile_m, stream)) {
    return false;
  }

  // 2. gate_up GEMM + act_mul + mxfp8 quant (N = 2 * intermediate_size, K = hidden_size)
  if (!gate_up_fuse_act_mul_quant_async(down_input_ptr, down_input_scale_ptr, x_ptr, x_scale_ptr,
                                        gate_up_weight_ptr, gate_up_weight_scale_ptr, seqlens_ptr,
                                        cu_seqlens_ptr, tiles_ptr, permuted_idx_to_raw_idx_ptr,
                                        num_expert_local, max_tiles, intermediate_size * 2,
                                        hidden_size, tile_m, num_topk, limit, is_fp4, stream)) {
    return false;
  }

  // 3. down GEMM (N = hidden_size, K = intermediate_size); its epilogue applies topk_scale and
  //    atomically adds into output_ptr, so the pipeline ends here.
  return down_fuse_reduce_async(output_ptr, down_input_ptr, down_input_scale_ptr, down_weight_ptr,
                                down_weight_scale_ptr, topk_scale_ptr, seqlens_ptr, cu_seqlens_ptr,
                                tiles_ptr, permuted_idx_to_raw_idx_ptr, num_expert_local, max_tiles,
                                hidden_size, intermediate_size, tile_m, num_topk, is_fp4, stream);
}

}  // namespace fuse_moe
}  // namespace hpc
