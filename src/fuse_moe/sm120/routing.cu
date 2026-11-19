// Copyright (C) 2026 Tencent.

#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cute/int_tuple.hpp>

#include "src/fuse_moe/sm120/common/launch.cuh"
#include "src/fuse_moe/sm120/common/utils.h"
#include "src/fuse_moe/sm120/routing.cuh"
#include "src/fuse_moe/sm120/routing_utils.cuh"

namespace hpc::fuse_moe::sm120 {
namespace {

// Counts are summed per block in shared memory first: the same few hundred
// global counters taking one atomic per route serialize in L2. The block's
// global add returns where its routes start within each expert, so topk_pos of
// a local route ends up as its position within the expert, and input preparation
// (finalize_route_rows) turns that into a padded row with no atomics of its own.
// The blocks also set up the scattered output, which saves its own launch: a copy
// of init_src, or zeros without one.
__global__ void count_routes_kernel(const int *topk_ids, int *topk_pos, int *route_counts,
                                    int *row_to_route, int total_routes, int max_padded_rows,
                                    int first_expert, int last_expert, const int4 *init_src,
                                    int4 *init_dst, int64_t init_vectors) {
  __shared__ int counts[kMaxExperts];
  int num_experts = last_expert - first_expert;
  for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x) {
    counts[expert] = 0;
  }
  __syncthreads();
  cudaGridDependencySynchronize();

  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (int row = thread; row < max_padded_rows; row += stride) {
    row_to_route[row] = -1;
  }
  for (int route = thread; route < total_routes; route += stride) {
    int expert = topk_ids[route];
    int position = -1;
    if (expert >= first_expert && expert < last_expert) {
      position = atomicAdd(counts + expert - first_expert, 1);
    }
    topk_pos[route] = position;
  }
  for (int64_t i = thread; i < init_vectors; i += stride) {
    init_dst[i] = init_src != nullptr ? init_src[i] : make_int4(0, 0, 0, 0);
  }
  __syncthreads();
  for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x) {
    if (counts[expert] != 0) {
      counts[expert] = atomicAdd(route_counts + expert, counts[expert]);
    }
  }
  __syncthreads();
  for (int route = thread; route < total_routes; route += stride) {
    int expert = topk_ids[route];
    if (expert >= first_expert && expert < last_expert) {
      topk_pos[route] += counts[expert - first_expert];
    }
  }
  cudaTriggerProgrammaticLaunchCompletion();
}

__global__ void prefix_routes_kernel(int *route_counts, int *num_tokens_per_expert,
                                     int *cu_padded_tokens_per_group, int *cu_m_tiles,
                                     int num_experts, int tile_m) {
  __shared__ int warp_totals[kRouteWarps];
  cudaGridDependencySynchronize();

  for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x) {
    int count = route_counts[expert];
    num_tokens_per_expert[expert] = count;
  }
  scan_expert_tiles(route_counts, num_experts, tile_m, cu_padded_tokens_per_group, cu_m_tiles,
                    warp_totals);
  __syncthreads();
  for (int expert = threadIdx.x; expert < num_experts; expert += blockDim.x) {
    route_counts[expert] = 0;
  }
  cudaTriggerProgrammaticLaunchCompletion();
}

// Block 0 maps the routes. Every block takes a slice of the output setup, as in
// count_routes_kernel: a memset or memcpy node would end the PDL chain.
__global__ void prepare_small_routes_kernel(const int *topk_ids, int *topk_pos, int *row_to_route,
                                            int *num_tokens_per_expert,
                                            int *cu_padded_tokens_per_group, int *cu_m_tiles,
                                            int total_routes, int num_experts, int first_expert,
                                            int tile_m, int max_padded_rows, const int4 *init_src,
                                            int4 *init_dst, int64_t init_vectors) {
  cudaGridDependencySynchronize();

  int64_t thread = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  for (int64_t i = thread; i < init_vectors; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    init_dst[i] = init_src != nullptr ? init_src[i] : make_int4(0, 0, 0, 0);
  }
  if (blockIdx.x != 0) {
    return;
  }
  map_routes_in_block(topk_ids, topk_pos, row_to_route, num_tokens_per_expert,
                      cu_padded_tokens_per_group, cu_m_tiles, total_routes, num_experts,
                      first_expert, tile_m, max_padded_rows);
  cudaTriggerProgrammaticLaunchCompletion();
}

}  // namespace

void prepare_routes_async(const int *topk_ids, int *topk_pos, int *row_to_route,
                          int *num_tokens_per_expert, int *cu_padded_tokens_per_group,
                          int *cu_m_tiles, int *route_counts, int num_tokens, int num_topk,
                          int num_experts, int first_expert, int tile_m, int max_padded_rows,
                          int num_sms, const void *init_src, void *init_dst, int64_t init_bytes,
                          cudaStream_t stream) {
  int total_routes = num_tokens * num_topk;
  int last_expert = first_expert + num_experts;
  int64_t init_vectors = init_dst != nullptr ? init_bytes / static_cast<int64_t>(sizeof(int4)) : 0;
  if (num_tokens <= kSmallRoutingMaxTokens) {
    constexpr int kVectorsPerThread = 4;
    int blocks = std::clamp(
        cute::ceil_div(static_cast<int>(init_vectors), kRouteThreads * kVectorsPerThread), 1,
        2 * num_sms);
    launch_pdl(prepare_small_routes_kernel, blocks, kRouteThreads, stream, topk_ids, topk_pos,
               row_to_route, num_tokens_per_expert, cu_padded_tokens_per_group, cu_m_tiles,
               total_routes, num_experts, first_expert, tile_m, max_padded_rows,
               static_cast<const int4 *>(init_src), static_cast<int4 *>(init_dst), init_vectors);
    return;
  }

  // Enough routes per block to keep the global atomics few, and enough blocks
  // to stream the output initialization at full bandwidth.
  constexpr int kRoutesPerBlock = 4 * kRouteThreads;
  int blocks = cute::ceil_div(total_routes, kRoutesPerBlock);
  if (init_dst != nullptr) {
    blocks = std::max(blocks, 4 * num_sms);
  }
  launch_pdl(count_routes_kernel, blocks, kRouteThreads, stream, topk_ids, topk_pos, route_counts,
             row_to_route, total_routes, max_padded_rows, first_expert, last_expert,
             static_cast<const int4 *>(init_src), static_cast<int4 *>(init_dst), init_vectors);
  launch_pdl(prefix_routes_kernel, 1, kRouteThreads, stream, route_counts, num_tokens_per_expert,
             cu_padded_tokens_per_group, cu_m_tiles, num_experts, tile_m);
}

}  // namespace hpc::fuse_moe::sm120
