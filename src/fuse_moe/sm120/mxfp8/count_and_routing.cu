// Copyright (C) 2026 Tencent.

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cub/cub.cuh>  // NOLINT

#include "src/fuse_moe/fuse_moe_mxfp8.h"
#include "src/fuse_moe/sm120/mxfp8/config.h"
#include "src/fuse_moe/sm120/mxfp8/count_and_routing.cuh"
#include "src/utils/utils.h"

namespace hpc {
namespace fuse_moe {

// gate_up and down share one kTileM.
template <int kTileM>
static void launch_count_and_route_mxfp8(
    const void *topk_ids_ptr, const void *topk_scale_ptr, void *permuted_idx_to_raw_idx_ptr,
    void *permuted_topk_scale_ptr, void *seqlens_ptr, void *cu_seqlens_ptr, void *gateup_tiles_ptr,
    void *down_tiles_ptr, const void *init_src_ptr, void *init_dst_ptr, int64_t init_bytes,
    int num_seq, int num_topk, int num_expert_local, int rank_ep, cudaStream_t stream) {
  constexpr int kTileM_GateUp = kTileM;
  constexpr int kTileM_Down = kTileM;

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

  assert(total_num_topk <=
             static_cast<int64_t>(max_blocks) * kThreadPerBlock * kRoutesPerThreadTactics.back() &&
         "fused_count_scan_route_kernel: total_num_topk exceeds the routes_per_thread=32 bound");

  int route_blocks = (total_num_topk + kThreadPerBlock * routes_per_thread - 1) /
                     (kThreadPerBlock * routes_per_thread);
  // Extra blocks for the output init own no routes.
  constexpr int kInitVectorsPerThread = 2;
  int64_t init_vectors = init_dst_ptr != nullptr ? init_bytes / 16 : 0;
  int64_t init_blocks = (init_vectors + kThreadPerBlock * kInitVectorsPerThread - 1) /
                        (kThreadPerBlock * kInitVectorsPerThread);
  int blocks = std::max(
      1, std::min<int>(max_blocks, std::max(static_cast<int64_t>(route_blocks), init_blocks)));

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
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, kThreadPerBlock,
                                                  cfg.dynamicSmemBytes);
    assert(blocks_per_sm >= 1 && blocks <= blocks_per_sm * num_sm &&
           "fused_count_scan_route_kernel: grid too large for a cooperative launch");
    cudaLaunchKernelEx(
        &cfg, kernel, reinterpret_cast<const int *>(topk_ids_ptr),
        reinterpret_cast<const float *>(topk_scale_ptr), reinterpret_cast<int *>(seqlens_ptr),
        reinterpret_cast<int *>(cu_seqlens_ptr), reinterpret_cast<int *>(gateup_tiles_ptr),
        reinterpret_cast<int *>(down_tiles_ptr),
        reinterpret_cast<int *>(permuted_idx_to_raw_idx_ptr),
        reinterpret_cast<float *>(permuted_topk_scale_ptr),
        reinterpret_cast<const int4 *>(init_src_ptr), reinterpret_cast<int4 *>(init_dst_ptr),
        init_vectors, total_num_topk, num_expert_local, start_expert, end_expert);
  };
  switch (routes_per_thread) {
    case 8:
      launch_fused(kernels::mxfp8::fused_count_scan_route_kernel<kThreadPerBlock, kTileM_GateUp,
                                                                 kTileM_Down, 8>);
      break;
    case 16:
      launch_fused(kernels::mxfp8::fused_count_scan_route_kernel<kThreadPerBlock, kTileM_GateUp,
                                                                 kTileM_Down, 16>);
      break;
    default:
      launch_fused(kernels::mxfp8::fused_count_scan_route_kernel<kThreadPerBlock, kTileM_GateUp,
                                                                 kTileM_Down, 32>);
      break;
  }
}

void count_and_route_mxfp8_async(const void *topk_ids_ptr, const void *topk_scale_ptr,
                                 void *permuted_idx_to_raw_idx_ptr, void *permuted_topk_scale_ptr,
                                 void *seqlens_ptr, void *cu_seqlens_ptr, void *gateup_tiles_ptr,
                                 void *down_tiles_ptr, const void *init_src_ptr, void *init_dst_ptr,
                                 int64_t init_bytes, int num_seq, int num_topk,
                                 int num_expert_local, int rank_ep, int tile_m,
                                 cudaStream_t stream) {
  dispatch_tile_m(tile_m, [&](auto kTileM) {
    launch_count_and_route_mxfp8<decltype(kTileM)::value>(
        topk_ids_ptr, topk_scale_ptr, permuted_idx_to_raw_idx_ptr, permuted_topk_scale_ptr,
        seqlens_ptr, cu_seqlens_ptr, gateup_tiles_ptr, down_tiles_ptr, init_src_ptr, init_dst_ptr,
        init_bytes, num_seq, num_topk, num_expert_local, rank_ep, stream);
  });
}

}  // namespace fuse_moe
}  // namespace hpc
