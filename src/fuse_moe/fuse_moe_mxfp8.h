// Copyright (C) 2026 Tencent.

#ifndef SRC_FUSE_MOE_FUSE_MOE_MXFP8_H_
#define SRC_FUSE_MOE_FUSE_MOE_MXFP8_H_

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace hpc {
namespace fuse_moe {

// init_dst_ptr (optional) is set to init_src_ptr, or zeros; both 16B aligned.
void count_and_route_mxfp8_async(const void *topk_ids_ptr, const void *topk_scale_ptr,
                                 void *permuted_idx_to_raw_idx_ptr, void *permuted_topk_scale_ptr,
                                 void *seqlens_ptr, void *cu_seqlens_ptr, void *gateup_tiles_ptr,
                                 void *down_tiles_ptr, const void *init_src_ptr, void *init_dst_ptr,
                                 int64_t init_bytes, int num_seq, int num_topk,
                                 int num_expert_local, int rank_ep, int tile_m,
                                 cudaStream_t stream);

void gate_up_fuse_act_mul_quant_async(void *y_ptr, void *y_scale_ptr, const void *x_ptr,
                                      const void *x_scale_ptr, const void *w_ptr,
                                      const void *w_scale_ptr, const void *seqlens_ptr,
                                      const void *cu_seqlens_ptr, const void *tiles_ptr,
                                      const void *permuted_idx_to_raw_idx_ptr, int num_group,
                                      int max_tiles, int n, int k, int tile_m, int num_topk,
                                      float limit, bool is_fp4, bool stream_weights,
                                      cudaStream_t stream);

void down_fuse_reduce_async(void *y_ptr, const void *a_ptr, const void *a_scale_ptr,
                            const void *w_ptr, const void *w_scale_ptr,
                            const void *permuted_topk_scale_ptr, const void *seqlens_ptr,
                            const void *cu_seqlens_ptr, const void *tiles_ptr,
                            const void *permuted_idx_to_raw_idx_ptr, int num_group, int max_tiles,
                            int n, int k, int tile_m, int num_topk, bool is_fp4,
                            bool stream_weights, cudaStream_t stream);

// With is_fp4 both the gate_up and the down weights are mxfp4 (two e2m1 per byte, the even index
// in the low 4 bits); their scales are still one ue8m0 per 32 in layout_128x4, and the
// activations stay mxfp8. stream_weights loads the weights L2 evict-first.
void fuse_moe_mxfp8_async(void *output_ptr,
                          // input
                          const void *x_ptr, const void *x_scale_ptr,
                          // gateup
                          const void *gate_up_weight_ptr, const void *gate_up_weight_scale_ptr,
                          // down
                          void *down_input_ptr, void *down_input_scale_ptr,
                          const void *down_weight_ptr, const void *down_weight_scale_ptr,
                          // routing / scratch
                          const void *topk_ids_ptr, const void *topk_scale_ptr,
                          void *permuted_idx_to_raw_idx_ptr, void *seqlens_ptr,
                          void *cu_seqlens_ptr, void *tiles_ptr, void *permuted_topk_scale_ptr,
                          // output init
                          const void *init_src_ptr, void *init_dst_ptr, int64_t init_bytes,
                          int num_seq, int hidden_size, int intermediate_size, int num_topk,
                          int num_expert_local, int rank_ep, int max_tiles, int tile_m, float limit,
                          bool is_fp4, bool stream_weights, cudaStream_t stream);

}  // namespace fuse_moe
}  // namespace hpc

#endif  // SRC_FUSE_MOE_FUSE_MOE_MXFP8_H_
