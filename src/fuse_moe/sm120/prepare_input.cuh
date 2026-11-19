// Copyright (C) 2026 Tencent.

#ifndef SRC_FUSE_MOE_SM120_PREPARE_INPUT_CUH_
#define SRC_FUSE_MOE_SM120_PREPARE_INPUT_CUH_

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace hpc::fuse_moe::sm120 {

// Token-major activations for the gate-up GEMM. BF16 input is quantized once per
// token to MXFP8 E4M3 [num_tokens, K]; each token's UE8M0 scales are scattered to
// its route rows in the packed SFA layout. The gate-up GEMM gathers A rows with
// TMA gather4 (see gemm/gate_up/tma_gather4.cuh); this stage does not copy A per
// route. Each local route maps to one GEMM row, grouped by expert and padded;
// the down GEMM uses the same route rows.
//
// Host side (prepare_input.cu):
//   quantize_tokens_and_scatter_scales_async quantizes BF16 [num_tokens, hidden]
//   (hidden % 128 == 0, num_topk <= kMaxTopK) into token-major MXFP8 and scatters
//   each token's UE8M0 scales to its route rows: byte
//   (packed_scale_row(row, cu_padded_tokens_per_group[expert], expert) + pack_col * scale_m_padded)
//   * 4 + byte_in_pack is the scale of K block pack_col * 4 + byte_in_pack.
//   row_to_route[padded_row] is the route, or -1 for padding.
//   scatter_token_scales_async handles existing E4M3 input with uint8 UE8M0 or
//   block128 float32 scales: the input stays in place; only the scales are prepared
//   and scattered.
//
//   Without finalize_route_rows, routing has already stored each local route's final
//   padded row in topk_pos[route] and set row_to_route[row] = route. When it is set,
//   topk_pos[route] holds the route's position within its expert; the kernel adds
//   cu_padded_tokens_per_group[expert], stores the final row in topk_pos[route], and
//   records the inverse mapping as row_to_route[row] = route.

void quantize_tokens_and_scatter_scales_async(const void *bf16_input, const int *topk_ids,
                                              const int *cu_padded_tokens_per_group, int *topk_pos,
                                              int *row_to_route, void *quantized_tokens,
                                              int32_t *packed_scale, int num_tokens, int num_topk,
                                              int hidden_size, int first_expert, int last_expert,
                                              int scale_m_padded, bool finalize_route_rows,
                                              cudaStream_t stream);

// Token-count-one fast path: prepare routing, set up the output, and scatter the
// scales in one 256-thread kernel. A null input_scale means BF16 input, which is
// quantized into quantized_tokens; otherwise input is E4M3 and the gate-up GEMM reads it in
// place. input_scale is then uint8 UE8M0 [num_tokens, hidden / 32], or with
// block128_scale float32 powers of two [num_tokens, hidden / 128]. init_dst,
// when not null, is set to a copy of init_src, or to zeros when that is null;
// both are 16-byte aligned and init_bytes is a multiple of 16.
void prepare_single_token_input_async(
    const void *input, const void *input_scale, bool block128_scale, const int *topk_ids,
    int *topk_pos, int *row_to_route, int *num_tokens_per_expert, int *cu_padded_tokens_per_group,
    int *cu_m_tiles, void *quantized_tokens, int32_t *packed_scale, int num_topk, int hidden_size,
    int num_experts, int first_expert, int tile_m, int max_padded_rows, int scale_m_padded,
    const void *init_src, void *init_dst, int64_t init_bytes, cudaStream_t stream);

// Token-major E4M3: no A copy. Scatter the token's scales into packed SFA at
// each local route row; input_scale is laid out as above. The gate-up GEMM TMA-gathers the
// input directly.
void scatter_token_scales_async(const void *input_scale, bool block128_scale, const int *topk_ids,
                                const int *cu_padded_tokens_per_group, int *topk_pos,
                                int *row_to_route, int32_t *packed_scale, int num_tokens,
                                int num_topk, int hidden_size, int first_expert, int last_expert,
                                int scale_m_padded, bool finalize_route_rows, cudaStream_t stream);

}  // namespace hpc::fuse_moe::sm120

#endif  // SRC_FUSE_MOE_SM120_PREPARE_INPUT_CUH_
