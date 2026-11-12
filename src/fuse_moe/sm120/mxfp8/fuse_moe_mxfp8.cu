// Copyright (C) 2026 Tencent.

#include <cuda.h>

#include "src/fuse_moe/fuse_moe_mxfp8.h"

namespace hpc {
namespace fuse_moe {

void fuse_moe_mxfp8_async(void *output_ptr, const void *x_ptr, const void *x_scale_ptr,
                          const void *gate_up_weight_ptr, const void *gate_up_weight_scale_ptr,
                          void *down_input_ptr, void *down_input_scale_ptr,
                          const void *down_weight_ptr, const void *down_weight_scale_ptr,
                          const void *topk_ids_ptr, const void *topk_scale_ptr,
                          void *permuted_idx_to_raw_idx_ptr, void *seqlens_ptr,
                          void *cu_seqlens_ptr, void *tiles_ptr, void *permuted_topk_scale_ptr,
                          const void *init_src_ptr, void *init_dst_ptr, int64_t init_bytes,
                          int num_seq, int hidden_size, int intermediate_size, int num_topk,
                          int num_expert_local, int rank_ep, int max_tiles, int tile_m, float limit,
                          bool is_fp4, bool stream_weights, cudaStream_t stream) {
  void *gateup_tiles_ptr = static_cast<int *>(tiles_ptr);
  void *down_tiles_ptr = static_cast<int *>(tiles_ptr) + num_expert_local;

  // 1. Route tokens (and init output)
  count_and_route_mxfp8_async(
      topk_ids_ptr, topk_scale_ptr, permuted_idx_to_raw_idx_ptr, permuted_topk_scale_ptr,
      seqlens_ptr, cu_seqlens_ptr, gateup_tiles_ptr, down_tiles_ptr, init_src_ptr, init_dst_ptr,
      init_bytes, num_seq, num_topk, num_expert_local, rank_ep, tile_m, stream);

  // 2. gate_up GEMM + act_mul + mxfp8 quant (N = 2 * intermediate_size, K = hidden_size)
  gate_up_fuse_act_mul_quant_async(down_input_ptr, down_input_scale_ptr, x_ptr, x_scale_ptr,
                                   gate_up_weight_ptr, gate_up_weight_scale_ptr, seqlens_ptr,
                                   cu_seqlens_ptr, gateup_tiles_ptr, permuted_idx_to_raw_idx_ptr,
                                   num_expert_local, max_tiles, intermediate_size * 2, hidden_size,
                                   tile_m, num_topk, limit, is_fp4, stream_weights, stream);

  // 3. down GEMM (N = hidden_size, K = intermediate_size); its epilogue applies topk_scale and
  //    atomically adds into output_ptr, so the pipeline ends here.
  down_fuse_reduce_async(output_ptr, down_input_ptr, down_input_scale_ptr, down_weight_ptr,
                         down_weight_scale_ptr, permuted_topk_scale_ptr, seqlens_ptr,
                         cu_seqlens_ptr, down_tiles_ptr, permuted_idx_to_raw_idx_ptr,
                         num_expert_local, max_tiles, hidden_size, intermediate_size, tile_m,
                         num_topk, is_fp4, stream_weights, stream);
}

}  // namespace fuse_moe
}  // namespace hpc
