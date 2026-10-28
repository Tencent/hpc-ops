// Copyright (C) 2026 Tencent.

#ifndef SRC_FUSE_MOE_SM120_DOWN_GEMM_CUH_
#define SRC_FUSE_MOE_SM120_DOWN_GEMM_CUH_

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace hpc::fuse_moe::sm120 {

// Down GEMM: MXFP8 E4M3 activations A[M, K] x expert weights, packed MXFP4 E2M1
// [E, N, K/2] when mxfp4_weight, else MXFP8 E4M3 [E, N, K]. UE8M0 block scales
// use vector size 32 in the compact int32 layout.
//
// cu_padded_tokens_per_group: int32[num_experts + 1] cumulative padded row offsets (with leading
// zero), matching prepare_routes; row_counts the routed rows per expert.
//
// Each row r of the product is scaled by topk_scale[row_to_route[r]] and added into the
// BF16 output row row_to_route[r] / num_topk with BF16 atomics; rows mapped to -1 are
// skipped. output is row-major [num_tokens, N].
void down_gemm_async(bool mxfp4_weight, const void *down_input, const void *down_weight,
                     void *down_input_scale, const void *down_weight_scale, void *output,
                     const int32_t *cu_padded_tokens_per_group, const int32_t *row_counts,
                     const int32_t *row_to_route, const float *topk_scale, int num_topk,
                     int max_padded_rows, int hidden_size, int down_input_size, int num_experts,
                     int tile_m, int tile_n, int num_ctas, uint64_t cache_hint_activation,
                     uint64_t cache_hint_weight, cudaStream_t stream);

}  // namespace hpc::fuse_moe::sm120

#endif  // SRC_FUSE_MOE_SM120_DOWN_GEMM_CUH_
