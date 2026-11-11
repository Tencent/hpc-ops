// Copyright (C) 2026 Tencent.
#ifndef SRC_GROUP_GEMM_GROUP_GEMM_W4A8_H_
#define SRC_GROUP_GEMM_GROUP_GEMM_W4A8_H_

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace hpc {
namespace group_gemm_cp_async {

struct W4A8GemmPlan {
  int tile_m;
  int tile_n;
  int stages;
};

// No GPU contents read. Both MoE GEMMs share tile_m for the same expanded_rows.
// Unsupported dimensions return {0,0,0}; routing consumes the returned tile_m.
W4A8GemmPlan group_gemm_w4a8_plan(int expanded_rows, int n, int k);

// x: FP8 [source_rows,K]; x_scale: FP32 [source_rows,K/128].
// w4: native packed-K int32 [E,K/64,8*N]; relative: opaque uint8 [E,K/32,N],
// physically [E,K128,N64,warp4,lane_group8,K32_4,rowpair2].
// common_scale: FP32 [E], from strict preparation, with no implicit /64 factor.
// row_indices: optional int32 [output_rows]; nullptr means already-grouped x.
// seqlens/cu_seqlens and int4 task_map use the original hpc format. E is 256.
// Each task is (expert, local_m_tile, n_tile,0); tail tasks have expert=-1.
// Output is grouped BF16 [output_rows,N], except paired gate/up plans
// (tile_n>=256,N=1024): output is FP8 [output_rows,512] and output_scale must
// point to FP32 [output_rows,4]. Other plans require output_scale=nullptr.
// Routing/weight preparation are external.
bool group_gemm_w4a8_async(void *output, const void *x, const float *x_scale, const void *w4,
                           const uint8_t *relative, const float *common_scale,
                           const int *row_indices, const int *seqlens, const int *cu_seqlens,
                           const void *task_map, int task_map_len, int source_rows, int output_rows,
                           int n, int k, cudaStream_t stream, float *output_scale = nullptr);

}  // namespace group_gemm_cp_async
}  // namespace hpc
#endif  // SRC_GROUP_GEMM_GROUP_GEMM_W4A8_H_
