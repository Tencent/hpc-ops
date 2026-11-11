// Copyright (C) 2026 Tencent.

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cub/cub.cuh>  // NOLINT
#include <stdexcept>
#include <string>

#include "src/fuse_moe/fuse_moe_mxfp8.h"
#include "src/fuse_moe/sm120/mxfp8/config.h"
#include "src/fuse_moe/sm120/mxfp8/count_and_routing.cuh"
#include "src/utils/utils.h"

namespace hpc {
namespace fuse_moe {

// gate_up and down share one kTileM, so one tile count per expert serves both.
template <int kTileM>
static bool launch_count_and_route_mxfp8(const void *topk_ids_ptr,
                                         void *permuted_idx_to_raw_idx_ptr, void *seqlens_ptr,
                                         void *cu_seqlens_ptr, void *tiles_ptr, int num_seq,
                                         int num_topk, int num_expert_local, int rank_ep,
                                         cudaStream_t stream) {
  int total_num_topk = num_seq * num_topk;
  int start_expert = rank_ep * num_expert_local;
  int end_expert = (rank_ep + 1) * num_expert_local;

  int num_sm = get_sm_count();

  constexpr int kThreadPerBlock = 256;
  constexpr std::array<int, 3> kRoutesPerThreadTactics = {8, 16, 32};
  int max_blocks = std::max(1, num_sm - 8);

  int routes_per_thread = kRoutesPerThreadTactics.back();
  for (int cand : kRoutesPerThreadTactics) {
    int blocks_needed = (total_num_topk + kThreadPerBlock * cand - 1) / (kThreadPerBlock * cand);
    if (blocks_needed <= max_blocks) {
      routes_per_thread = cand;
      break;
    }
  }

  // Every route keeps its slot in a register until the emit stage, so one launch covers at most
  // max_blocks * kThreadPerBlock * 32 routes; past that the rest would be silently dropped.
  int64_t max_routes =
      static_cast<int64_t>(max_blocks) * kThreadPerBlock * kRoutesPerThreadTactics.back();
  if (total_num_topk > max_routes) {
    throw std::invalid_argument(
        "fuse_moe_mxfp8: num_seq * num_topk = " + std::to_string(total_num_topk) +
        " exceeds the routing limit " + std::to_string(max_routes) + " on this device");
  }

  int route_blocks = (total_num_topk + kThreadPerBlock * routes_per_thread - 1) /
                     (kThreadPerBlock * routes_per_thread);
  int blocks = std::max(1, std::min(max_blocks, route_blocks));

  cudaLaunchAttribute attrs[2];
  attrs[0].id = cudaLaunchAttributeCooperative;
  attrs[0].val.cooperative = 1;
  attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[1].val.programmaticStreamSerializationAllowed = 1;

  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(blocks);
  cfg.blockDim = dim3(kThreadPerBlock);

  cfg.dynamicSmemBytes = 2 * num_expert_local * sizeof(int);
  cfg.stream = stream;
  cfg.attrs = attrs;
  cfg.numAttrs = 2;

  auto launch_fused = [&](auto kernel) {
    int blocks_per_sm = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, kThreadPerBlock,
                                                      cfg.dynamicSmemBytes) != cudaSuccess) {
      return false;
    }
    if (blocks_per_sm < 1 || blocks > blocks_per_sm * num_sm) {
      throw std::runtime_error("fuse_moe_mxfp8: routing grid too large for a cooperative launch");
    }
    return cudaLaunchKernelEx(&cfg, kernel, reinterpret_cast<const int *>(topk_ids_ptr),
                              reinterpret_cast<int *>(seqlens_ptr),
                              reinterpret_cast<int *>(cu_seqlens_ptr),
                              reinterpret_cast<int *>(tiles_ptr),
                              reinterpret_cast<int *>(permuted_idx_to_raw_idx_ptr), total_num_topk,
                              num_expert_local, start_expert, end_expert) == cudaSuccess;
  };
  switch (routes_per_thread) {
    case 8:
      return launch_fused(
          kernels::mxfp8::fused_count_scan_route_kernel<kThreadPerBlock, kTileM, 8>);
    case 16:
      return launch_fused(
          kernels::mxfp8::fused_count_scan_route_kernel<kThreadPerBlock, kTileM, 16>);
    default:
      return launch_fused(
          kernels::mxfp8::fused_count_scan_route_kernel<kThreadPerBlock, kTileM, 32>);
  }
}

bool count_and_route_mxfp8_async(const void *topk_ids_ptr, void *permuted_idx_to_raw_idx_ptr,
                                 void *seqlens_ptr, void *cu_seqlens_ptr, void *tiles_ptr,
                                 int num_seq, int num_topk, int num_expert_local, int rank_ep,
                                 int tile_m, cudaStream_t stream) {
  bool launched = false;
  dispatch_tile_m(tile_m, [&](auto kTileM) {
    launched = launch_count_and_route_mxfp8<decltype(kTileM)::value>(
        topk_ids_ptr, permuted_idx_to_raw_idx_ptr, seqlens_ptr, cu_seqlens_ptr, tiles_ptr, num_seq,
        num_topk, num_expert_local, rank_ep, stream);
  });
  return launched;
}

}  // namespace fuse_moe
}  // namespace hpc
