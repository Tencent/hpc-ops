// Copyright (C) 2026 Tencent.

#pragma once
// clang-format off
#include <cutlass/arch/barrier.h>

#include <cstdint>
// clang-format on

namespace hpc::fuse_moe::sm120::gemm {
namespace common {

// Warp-cooperative tile scheduler for MGroupedContiguousWithZeroPadding.
// cu_padded_tokens_per_group is token_offset[E+1] int32 cumsum with a leading 0.
// Each get_next_block resolves the owning group with one coalesced 32-group
// window per round (lane-parallel prefix sum + ballot). All 32 lanes of the
// calling warp must be active and converged.
template <int BlockM, int BlockN>
struct MoeScheduler {
  int32_t shape_m;
  int32_t shape_n;
  int32_t num_groups;
  int32_t num_n_blocks;
  int32_t const* cu_padded_tokens_per_group;
  int32_t const* row_counts;

  int32_t current_iter = -1;
  int32_t current_group_idx = 0;
  int32_t prev_psum_m = 0;
  int32_t current_psum_m = 0;
  int32_t current_valid_boundary = 0;

  int32_t window_base = 0;
  int32_t window_start_block = 0;
  int32_t lane_off = 0;
  int32_t lane_next = 0;
  int32_t lane_count = 0;

  __device__ __forceinline__ explicit MoeScheduler(int shape_m_, int shape_n_, int num_groups_,
                                                   int32_t const* cu_padded_tokens_per_group_,
                                                   int32_t const* row_counts_ = nullptr)
      : shape_m(shape_m_),
        shape_n(shape_n_),
        num_groups(num_groups_),
        num_n_blocks((shape_n_ + BlockN - 1) / BlockN),
        cu_padded_tokens_per_group(cu_padded_tokens_per_group_),
        row_counts(row_counts_) {
    load_window();
  }

  __device__ __forceinline__ void load_window() {
    int32_t lane = threadIdx.x & 31;
    int32_t g = window_base + lane;
    lane_off = cu_padded_tokens_per_group[g < num_groups ? g : num_groups];
    lane_next = cu_padded_tokens_per_group[g + 1 < num_groups ? g + 1 : num_groups];
    lane_count = row_counts != nullptr && g < num_groups ? row_counts[g] : lane_next - lane_off;
  }

  __device__ __forceinline__ bool get_next_block(int32_t& m_block_idx, int32_t& n_block_idx) {
    int64_t next_block_idx = static_cast<int64_t>(++current_iter) * gridDim.x + blockIdx.x;
    int64_t target_m_block = next_block_idx / num_n_blocks;
    n_block_idx = static_cast<int32_t>(next_block_idx % num_n_blocks);

    int32_t lane = threadIdx.x & 31;
    while (true) {
      int32_t my_blocks = (lane_next - lane_off + BlockM - 1) / BlockM;
      int32_t incl = my_blocks;
#pragma unroll
      for (int d = 1; d < 32; d <<= 1) {
        int32_t up = __shfl_up_sync(0xffffffffu, incl, d);
        if (lane >= d) {
          incl += up;
        }
      }
      int32_t window_total = __shfl_sync(0xffffffffu, incl, 31);

      if (target_m_block < static_cast<int64_t>(window_start_block) + window_total) {
        int32_t local = static_cast<int32_t>(target_m_block - window_start_block);
        uint32_t owners = __ballot_sync(0xffffffffu, incl > local);
        int32_t src = __ffs(owners) - 1;
        current_group_idx = window_base + src;
        prev_psum_m = __shfl_sync(0xffffffffu, lane_off, src);
        current_psum_m = __shfl_sync(0xffffffffu, lane_next, src);
        current_valid_boundary = prev_psum_m + __shfl_sync(0xffffffffu, lane_count, src);
        int32_t group_start = __shfl_sync(0xffffffffu, incl - my_blocks, src);
        m_block_idx = local - group_start;
        return true;
      }
      if (window_base + 32 >= num_groups) {
        return false;
      }
      window_start_block += window_total;
      window_base += 32;
      load_window();
    }
  }

  __device__ __forceinline__ int32_t get_m_offset() const { return prev_psum_m; }
  __device__ __forceinline__ int32_t get_m_boundary() const { return current_valid_boundary; }
  __device__ __forceinline__ int32_t get_expert_idx(int32_t = 0) const { return current_group_idx; }
};

// Dedicated-scheduler-warp handoff: one warp runs MoeScheduler and publishes
// every resolved tile into this SMEM pipeline; the consumer warps carry no
// scheduler and drain stages until the valid == 0 end sentinel.
struct MoeWorkTile {
  int32_t m_block, n_block, group, m_offset, m_boundary, valid;
};

template <int Stages>
struct MoeSchedStorage {
  MoeWorkTile work[Stages];
  alignas(8) uint64_t sched_full_mbar[Stages];
  alignas(8) uint64_t sched_empty_mbar[Stages];

  __device__ __forceinline__ cutlass::arch::ClusterBarrier* full_mbar(int sched_stage) {
    return reinterpret_cast<cutlass::arch::ClusterBarrier*>(&sched_full_mbar[sched_stage]);
  }
  __device__ __forceinline__ cutlass::arch::ClusterBarrier* empty_mbar(int sched_stage) {
    return reinterpret_cast<cutlass::arch::ClusterBarrier*>(&sched_empty_mbar[sched_stage]);
  }
  __device__ __forceinline__ void init_mbars(int num_consumers) {
#pragma unroll
    for (int i = 0; i < Stages; ++i) {
      full_mbar(i)->init(1);
      empty_mbar(i)->init(num_consumers);
    }
  }
};

// Producer view of the sched pipeline: the scheduler warp publishes one
// MoeWorkTile per stage and closes the stream with the valid == 0 sentinel.
template <int Stages>
struct MoeSchedProducer {
  MoeSchedStorage<Stages>& storage;
  int lane_predicate;
  uint32_t sched_iter = 0;

  __device__ __forceinline__ void publish(MoeWorkTile const& tile) {
    int sched_stage = sched_iter % Stages;
    storage.empty_mbar(sched_stage)->wait(((sched_iter / Stages) & 1) ^ 1);
    if (lane_predicate) {
      storage.work[sched_stage] = tile;
      storage.full_mbar(sched_stage)->arrive();
    }
    ++sched_iter;
  }
  __device__ __forceinline__ void publish_sentinel() {
    int sched_stage = sched_iter % Stages;
    storage.empty_mbar(sched_stage)->wait(((sched_iter / Stages) & 1) ^ 1);
    if (lane_predicate) {
      storage.work[sched_stage].valid = 0;
      storage.full_mbar(sched_stage)->arrive();
    }
  }
};

// Consumer view: waits a stage, copies the tile into registers, then the
// elected lane releases the stage before the caller starts working on it.
// Returns false on the sentinel (which is never released).
template <int Stages>
struct MoeSchedConsumer {
  MoeSchedStorage<Stages>& storage;
  int lane_predicate;
  uint32_t sched_iter = 0;

  __device__ __forceinline__ bool get_next_tile(MoeWorkTile& tile) {
    int sched_stage = sched_iter % Stages;
    storage.full_mbar(sched_stage)->wait((sched_iter / Stages) & 1);
    tile = storage.work[sched_stage];
    // Every lane must own its register copy before the elected lane lets
    // the producer overwrite the stage.
    __syncwarp();
    if (!tile.valid) {
      return false;
    }
    if (lane_predicate) {
      storage.empty_mbar(sched_stage)->arrive();
    }
    ++sched_iter;
    return true;
  }
};

}  // namespace common
}  // namespace hpc::fuse_moe::sm120::gemm
