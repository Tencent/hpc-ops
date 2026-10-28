// Copyright (C) 2026 Tencent.

#pragma once

// clang-format off
#include <cutlass/arch/barrier.h>

#include <cstdint>
// clang-format on

namespace hpc::fuse_moe::sm120::gemm::down {

// Scatter epilogue rows: the scheduler warp resolves each published tile's
// rows to (token, routing weight) so the math warps' epilogue reads SMEM
// instead of waiting on two dependent global loads.
template <int Stages, int TileM>
struct MoeScatterRowsStorage {
  int2 rows[Stages][TileM];
  alignas(8) uint64_t full_mbar[Stages];
  alignas(8) uint64_t empty_mbar[Stages];

  __device__ __forceinline__ cutlass::arch::ClusterBarrier* full(int stage) {
    return reinterpret_cast<cutlass::arch::ClusterBarrier*>(&full_mbar[stage]);
  }
  __device__ __forceinline__ cutlass::arch::ClusterBarrier* empty(int stage) {
    return reinterpret_cast<cutlass::arch::ClusterBarrier*>(&empty_mbar[stage]);
  }
  __device__ __forceinline__ void init_mbars(int num_consumers) {
#pragma unroll
    for (int i = 0; i < Stages; ++i) {
      full(i)->init(32);
      empty(i)->init(num_consumers);
    }
  }
};

// One tile's slot as seen by a math warp; consumed once by the epilogue.
struct MoeScatterRowsSlot {
  int2 const* rows;
  cutlass::arch::ClusterBarrier* full;
  cutlass::arch::ClusterBarrier* empty;
  uint32_t phase;
};

// Called by all 32 lanes of the scheduler warp after it publishes the tile.
template <int Stages, int TileM, typename Params>
__device__ __forceinline__ void produce_scatter_rows(MoeScatterRowsStorage<Stages, TileM>& storage,
                                                     uint32_t iter, Params const& params,
                                                     int32_t m_offset, int32_t m_boundary,
                                                     int32_t m_block_idx) {
  int stage = iter % Stages;
  storage.empty(stage)->wait(((iter / Stages) & 1) ^ 1);
  int lane = threadIdx.x % 32;
  int32_t row_base = m_offset + m_block_idx * TileM;
  int32_t residue_m = m_boundary - row_base;
#pragma unroll
  for (int r = lane; r < TileM; r += 32) {
    int route = r < residue_m ? params.row_to_route[row_base + r] : -1;
    float weight = route >= 0 ? params.topk_scale[route] : 0.f;
    storage.rows[stage][r] =
        make_int2(route >= 0 ? route / params.num_topk : -1, __float_as_int(weight));
  }
  storage.full(stage)->arrive();
}

}  // namespace hpc::fuse_moe::sm120::gemm::down
