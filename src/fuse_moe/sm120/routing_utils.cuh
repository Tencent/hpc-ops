// Copyright (C) 2026 Tencent.

#ifndef SRC_FUSE_MOE_SM120_ROUTING_UTILS_CUH_
#define SRC_FUSE_MOE_SM120_ROUTING_UTILS_CUH_

#include "src/fuse_moe/sm120/common/utils.h"

namespace hpc::fuse_moe::sm120 {

constexpr int kRouteWarps = kRouteThreads / kWarpSize;
static_assert(2 * kRouteThreads >= kMaxExperts, "each routing thread scans two experts");
static_assert(kMaxTopK <= kRouteThreads, "one routing thread per route of a token");

// Exclusive prefix sum of the experts' M tiles, run by all kRouteThreads threads
// of the block: cu_m_tiles[e] is expert e's first tile and cu_padded_tokens_per_group[e]
// its first padded row, both with num_experts + 1 entries. warp_totals is shared
// memory of kRouteWarps ints.
__device__ __forceinline__ void scan_expert_tiles(const int* counts, int num_experts, int tile_m,
                                                  int* cu_padded_tokens_per_group, int* cu_m_tiles,
                                                  int* warp_totals) {
  int tid = threadIdx.x;
  int lane = tid % kWarpSize;
  int warp = tid / kWarpSize;
  int slot0 = tid * 2;
  int count0 = slot0 < num_experts ? counts[slot0] : 0;
  int count1 = slot0 + 1 < num_experts ? counts[slot0 + 1] : 0;
  int tiles0 = (count0 + tile_m - 1) / tile_m;
  int tiles1 = (count1 + tile_m - 1) / tile_m;
  int thread_total = tiles0 + tiles1;

  int inclusive = thread_total;
#pragma unroll
  for (int offset = 1; offset < kWarpSize; offset *= 2) {
    int preceding = __shfl_up_sync(0xffffffff, inclusive, offset);
    if (lane >= offset) {
      inclusive += preceding;
    }
  }
  if (lane == kWarpSize - 1) {
    warp_totals[warp] = inclusive;
  }
  __syncthreads();
  if (warp == 0 && lane < kRouteWarps) {
    int prefix = warp_totals[lane];
#pragma unroll
    for (int offset = 1; offset < kRouteWarps; offset *= 2) {
      int preceding = __shfl_up_sync((1u << kRouteWarps) - 1, prefix, offset);
      if (lane >= offset) {
        prefix += preceding;
      }
    }
    warp_totals[lane] = prefix;
  }
  __syncthreads();

  int prefix = (warp == 0 ? 0 : warp_totals[warp - 1]) + inclusive - thread_total;
  if (slot0 < num_experts) {
    cu_m_tiles[slot0] = prefix;
    cu_padded_tokens_per_group[slot0] = prefix * tile_m;
  }
  prefix += tiles0;
  if (slot0 + 1 < num_experts) {
    cu_m_tiles[slot0 + 1] = prefix;
    cu_padded_tokens_per_group[slot0 + 1] = prefix * tile_m;
  }
  if (tid == 0) {
    int total_tiles = warp_totals[kRouteWarps - 1];
    cu_m_tiles[num_experts] = total_tiles;
    cu_padded_tokens_per_group[num_experts] = total_tiles * tile_m;
  }
}

// Maps every route with this block alone, run by all kRouteThreads threads:
// counts the local routes per expert, scans them into cu_padded_tokens_per_group /
// cu_m_tiles, stores each local route's final row in topk_pos[route], and records
// row_to_route[row] = route. Nonlocal routes get topk_pos[route] = -1; unused padded
// rows get row_to_route[row] = -1. Callers sync before reading the outputs.
// kOneRoutePerThread needs total_routes <= kRouteThreads; dropping the route
// loop there keeps the single-token kernel about 0.15 us faster.
template <bool kOneRoutePerThread = false>
__device__ __forceinline__ void map_routes_in_block(const int* topk_ids, int* topk_pos,
                                                    int* row_to_route, int* num_tokens_per_expert,
                                                    int* cu_padded_tokens_per_group,
                                                    int* cu_m_tiles, int total_routes,
                                                    int num_experts, int first_expert, int tile_m,
                                                    int max_padded_rows) {
  __shared__ int counts[kMaxExperts];
  __shared__ int row_counters[kMaxExperts];
  __shared__ int warp_totals[kRouteWarps];
  int last_expert = first_expert + num_experts;
  auto for_each_route = [&](auto&& fn) {
    if constexpr (kOneRoutePerThread) {
      if (threadIdx.x < total_routes) {
        fn(static_cast<int>(threadIdx.x));
      }
    } else {
      for (int route = threadIdx.x; route < total_routes; route += blockDim.x) {
        fn(route);
      }
    }
  };

  for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x) {
    counts[expert] = 0;
    row_counters[expert] = 0;
  }
  for (int row = threadIdx.x; row < max_padded_rows; row += blockDim.x) {
    row_to_route[row] = -1;
  }
  __syncthreads();

  for_each_route([&](int route) {
    int expert = topk_ids[route];
    topk_pos[route] = -1;
    if (expert >= first_expert && expert < last_expert) {
      atomicAdd(counts + expert - first_expert, 1);
    }
  });
  __syncthreads();
  for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x) {
    num_tokens_per_expert[expert] = counts[expert];
  }
  scan_expert_tiles(counts, num_experts, tile_m, cu_padded_tokens_per_group, cu_m_tiles,
                    warp_totals);
  __syncthreads();

  for_each_route([&](int route) {
    int global_expert = topk_ids[route];
    if (global_expert >= first_expert && global_expert < last_expert) {
      int expert = global_expert - first_expert;
      int row = cu_padded_tokens_per_group[expert] + atomicAdd(row_counters + expert, 1);
      topk_pos[route] = row;
      row_to_route[row] = route;
    }
  });
}

}  // namespace hpc::fuse_moe::sm120

#endif  // SRC_FUSE_MOE_SM120_ROUTING_UTILS_CUH_
