// Copyright (C) 2026 Tencent.

#ifndef SRC_FUSE_MOE_SM120_GATE_UP_GEMM_CUH_
#define SRC_FUSE_MOE_SM120_GATE_UP_GEMM_CUH_

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace hpc::fuse_moe::sm120 {

// Gate-up GEMM: MXFP8 E4M3 activations x gate+up weights, fused SwiGLU and MXFP8
// requant (UE8M0, GranK=32). Weights are packed MXFP4 E2M1 [E, 2N, K/2] when
// mxfp4_weight, else MXFP8 E4M3 [E, 2N, K], with gate in [0:N) and up in
// [N:2N). tile_m is 32 or 64 and tile_n is 64 or 128.
//
// gate_up_input is token-major MXFP8 [num_tokens, K]; the kernel gathers each padded row's
// token, row_to_route[row] / num_topk, with TMA gather4 (-1 rows zero fill).
// gate_up_input_scale/gate_up_weight_scale are compact int32 UE8M0 packs (four scales per word,
// GranK=32). cu_padded_tokens_per_group is int32[E+1] cumulative padded row offsets and
// row_counts the routed rows per expert.
//
// down_input is MXFP8 E4M3 [M, N] plus compact int32 down_input_scale addressed with
// packed_scale_row(global_row, expert_begin, expert). Padding rows inside an
// expert tile are not written. swiglu_limit < 0 disables the DeepSeek clamp.
//
// zero_dst: zero_bytes (whole 16-byte vectors, 16-byte aligned) the kernel
// zeroes from a warp that does no GEMM work, zero_chunk bytes per CTA per tile,
// for the down GEMM to add into.
void gate_up_gemm_async(bool mxfp4_weight, const void *gate_up_input, const void *gate_up_weight,
                        void *gate_up_input_scale, const void *gate_up_weight_scale,
                        void *down_input, void *down_input_scale,
                        const int32_t *cu_padded_tokens_per_group, const int32_t *row_counts,
                        const int32_t *row_to_route, int num_tokens, int num_topk,
                        int max_padded_rows, int down_input_size, int hidden_size, int num_experts,
                        int tile_m, int tile_n, int num_ctas, float swiglu_limit, void *zero_dst,
                        int64_t zero_bytes, int64_t zero_chunk, uint64_t cache_hint_activation,
                        uint64_t cache_hint_weight, cudaStream_t stream);

}  // namespace hpc::fuse_moe::sm120

#endif  // SRC_FUSE_MOE_SM120_GATE_UP_GEMM_CUH_
