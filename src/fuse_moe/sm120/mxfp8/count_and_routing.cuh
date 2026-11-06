// Copyright (C) 2026 Tencent.

#ifndef SRC_FUSE_MOE_SM120_MXFP8_COUNT_AND_ROUTING_CUH_
#define SRC_FUSE_MOE_SM120_MXFP8_COUNT_AND_ROUTING_CUH_

#include <cooperative_groups.h>
#include <cuda.h>

#include <cub/cub.cuh>  // NOLINT

#include "cute/tensor.hpp"
#include "cutlass/fast_math.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace fuse_moe {
namespace kernels {
namespace mxfp8 {

// Counting, the prefix sums and the routing table in one kernel, with a single grid sync.
// Outputs:
//   seqlens / cu_seqlens        token count per local expert and its prefix sum
//   gateup_tiles                M-direction tile count of the GateUp GEMM
//   down_tiles                  M-direction tile count of the Down GEMM
//   permuted_idx_to_raw_idx     GEMM row -> raw flat index (reverse map)
template <int kThreadPerBlock, int kTileM_Gateup, int kTileM_Down, int kRoutesPerThread>
__global__ void fused_count_scan_route_kernel(const int *topk_ids_ptr, int *seqlens_ptr,
                                              int *cu_seqlens_ptr, int *gateup_tiles_ptr,
                                              int *down_tiles_ptr, int *permuted_idx_to_raw_idx_ptr,
                                              int total_num_topk, int num_expert, int start_expert,
                                              int end_expert) {
  int tid = threadIdx.x;
  int idx = tid + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;

  // The launcher enables programmatic stream serialization, so without this call PDL grants no
  // ordering at all against the producer of topk_ids. Placed before the shared-memory init so
  // that init overlaps the wait.
  cudaGridDependencySynchronize();

  extern __shared__ int seqlens_shm[];
  int *block_base_shm = seqlens_shm + num_expert;

  using BlockScan = cub::BlockScan<int, kThreadPerBlock>;
  __shared__ typename BlockScan::TempStorage temp_storage_seqs;

  for (int i = tid; i < num_expert; i += blockDim.x) {
    seqlens_shm[i] = 0;
  }
  __syncthreads();

  // ---- Stage A: per-block histogram, remembering each route's slot ----
  // local_expert/local_slot are per-slot register arrays (NOT a scalar reused across iterations)
  // because every route this thread owns must survive to the emit stage below.
  int local_expert[kRoutesPerThread];
  int local_slot[kRoutesPerThread];

  // 4-way lookahead: if the LAST index of a group of 4 is in bounds, all 4 are, so the tail check
  // collapses to one test per group.
  static_assert(kRoutesPerThread % 4 == 0, "kRoutesPerThread must be a multiple of 4");
#pragma unroll
  for (int rb = 0; rb < kRoutesPerThread; rb += 4) {
    int i_last = idx + (rb + 3) * stride;
    if (i_last < total_num_topk) {
#pragma unroll
      for (int d = 0; d < 4; d++) {
        int r = rb + d;
        int iexpert = topk_ids_ptr[idx + r * stride];
        if ((iexpert >= start_expert) && (iexpert < end_expert)) {
          local_expert[r] = iexpert - start_expert;
          local_slot[r] = atomicAdd(&seqlens_shm[local_expert[r]], 1);
        } else {
          local_expert[r] = -1;
        }
      }
    } else {
#pragma unroll
      for (int d = 0; d < 4; d++) {
        int r = rb + d;
        local_expert[r] = -1;
        int i = idx + r * stride;
        if (i < total_num_topk) {
          int iexpert = topk_ids_ptr[i];
          if ((iexpert >= start_expert) && (iexpert < end_expert)) {
            local_expert[r] = iexpert - start_expert;
            local_slot[r] = atomicAdd(&seqlens_shm[local_expert[r]], 1);
          }
        }
      }
    }
  }
  __syncthreads();

  // ---- Stage B: publish this block's counts, claim its slice per expert ----
  for (int i = tid; i < num_expert; i += blockDim.x) {
    int count = seqlens_shm[i];
    block_base_shm[i] = (count > 0) ? atomicAdd(&seqlens_ptr[i], count) : 0;
  }

  // The only grid-wide sync: after this, seqlens_ptr holds the final per-expert totals and is
  // visible to every block. Legal because the launcher uses a cooperative launch with a
  // co-resident grid.
  cooperative_groups::this_grid().sync();

  // ---- Stage C: every block redundantly scans (num_expert <= 256) ----
  // Recomputing this tiny block scan in all blocks is cheaper than a second grid sync to
  // broadcast block 0's result. Only cu_seqlens needs a scan: the GEMMs consume the per-expert
  // tile counts directly, since get_next_tile_horizon accumulates them as it walks.
  int iseq = (tid < num_expert) ? seqlens_ptr[tid] : 0;
  int itiles_gate_up = (iseq + kTileM_Gateup - 1) / kTileM_Gateup;
  int itiles_down = (iseq + kTileM_Down - 1) / kTileM_Down;

  int seq_total;
  int cu_seqlen;
  BlockScan(temp_storage_seqs).ExclusiveSum(iseq, cu_seqlen, seq_total);

  // Only block 0 publishes the metadata the downstream GEMMs consume; every block computed
  // identical values, so the rest would be redundant traffic.
  if (blockIdx.x == 0) {
    if (tid < num_expert) {
      gateup_tiles_ptr[tid] = itiles_gate_up;
      down_tiles_ptr[tid] = itiles_down;
      cu_seqlens_ptr[tid] = cu_seqlen;
    }
    if (tid == 0) {
      cu_seqlens_ptr[num_expert] = seq_total;
    }
  }

  // Broadcast cu_seqlen into smem (seqlens_shm is dead after Stage B) so the emit stage can look
  // up any local_expert[r], not just tid's own expert.
  if (tid < num_expert) {
    seqlens_shm[tid] = cu_seqlen;
  }
  __syncthreads();

// ---- Stage D: emit permuted_idx_to_raw_idx ----
// Row = global expert base + this block's slice + slot within the block. It stores the raw flat
// index (token_idx*num_topk + topk_idx), which is the dense row space the GEMMs read; they decode
// token_idx via a compile-time divide and read topk_scale[i].
#pragma unroll
  for (int r = 0; r < kRoutesPerThread; r++) {
    int le = local_expert[r];
    if (le >= 0) {
      permuted_idx_to_raw_idx_ptr[seqlens_shm[le] + block_base_shm[le] + local_slot[r]] =
          idx + r * stride;
    }
  }

  cudaTriggerProgrammaticLaunchCompletion();
}

}  // namespace mxfp8
}  // namespace kernels
}  // namespace fuse_moe
}  // namespace hpc

#endif  // SRC_FUSE_MOE_SM120_MXFP8_COUNT_AND_ROUTING_CUH_
