// Copyright (C) 2026 Tencent.

#ifndef SRC_FUSE_MOE_SM120_ROUTING_CUH_
#define SRC_FUSE_MOE_SM120_ROUTING_CUH_

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace hpc::fuse_moe::sm120 {

// Up to kSmallRoutingMaxTokens tokens, topk_pos and row_to_route come out final.
// Past that, a local route's topk_pos is its position within its expert, -1 for
// other routes, and prepare_input with finalize_route_rows adds the expert's
// cu_padded_tokens_per_group entry to make it a row.
// init_dst, when not null, is set to a copy of init_src, or to zeros when that is
// null, before any later kernel on the stream runs. Both are 16-byte aligned and
// init_bytes is a multiple of 16.
void prepare_routes_async(const int *topk_ids, int *topk_pos, int *row_to_route,
                          int *num_tokens_per_expert, int *cu_padded_tokens_per_group,
                          int *cu_m_tiles, int *route_counts, int num_tokens, int num_topk,
                          int num_experts, int first_expert, int tile_m, int max_padded_rows,
                          int num_sms, const void *init_src, void *init_dst, int64_t init_bytes,
                          cudaStream_t stream);

}  // namespace hpc::fuse_moe::sm120

#endif  // SRC_FUSE_MOE_SM120_ROUTING_CUH_
