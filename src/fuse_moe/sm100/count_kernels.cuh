// Copyright 2026 hpc-ops authors
//
// Shared routing kernels used by both fp8 and mxfp8 fuse_moe paths.
// fuse_moe paths.

#ifndef SRC_FUSE_MOE_SM100_COUNT_KERNELS_CUH_
#define SRC_FUSE_MOE_SM100_COUNT_KERNELS_CUH_

#include <cooperative_groups.h>
#include <cuda.h>

#include <cub/cub.cuh>

#include "cute/tensor.hpp"
#include "cutlass/fast_math.h"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace fuse_moe {
namespace kernels {

template <int kThreadPerBlock, bool kUsePDL = false>
__global__ void count_seq_kernel(const int *topk_ids_ptr, int *topk_pos_ptr, int *seqlens_ptr,
                                 int total_num_topk, int num_expert, int start_expert,
                                 int end_expert) {
  namespace cg = cooperative_groups;
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;

  extern __shared__ int seqlens_shm[];

  for (int i = idx; i < num_expert; i += stride) {
    seqlens_ptr[i] = 0;
  }

  for (int i = threadIdx.x; i < num_expert; i += blockDim.x) {
    seqlens_shm[i] = 0;
  }
  __syncthreads();

  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }

  cg::this_grid().sync();

  for (int i = idx; i < total_num_topk; i += stride) {
    int iexpert = topk_ids_ptr[i];
    if ((iexpert >= start_expert) && (iexpert < end_expert)) {
      atomicAdd(&seqlens_shm[iexpert - start_expert], 1);
    } else {
      topk_pos_ptr[i] = -1;
    }
  }

  __syncthreads();

  for (int i = threadIdx.x; i < num_expert; i += blockDim.x) {
    int count = seqlens_shm[i];
    if (count > 0) {
      atomicAdd(&seqlens_ptr[i], count);
    }
  }
  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

template <int kThreadPerBlock, int kGroupPerThread, int kTileM_Gateup, int kTileM_Down,
          bool kUsePDL = false>
__global__ void count_cuseq_kernel(const int *topk_ids_ptr, int *topk_pos_ptr, int *seqlens_ptr,
                                   int *cu_seqlens_ptr, int *gateup_tiles_ptr,
                                   int *gateup_cu_tiles_ptr, int *down_tiles_ptr,
                                   int *down_cu_tiles_ptr, int *x_row_map_race_pos_ptr, int num_seq,
                                   int num_topk, int total_num_topk, int num_expert,
                                   int start_expert, int end_expert) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  int thread_seqs[kGroupPerThread];
  int gateup_thread_tiles[kGroupPerThread];
  int down_thread_tiles[kGroupPerThread];
  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }
#pragma unroll
  for (int i = 0; i < kGroupPerThread; i++) {
    int igroup = idx * kGroupPerThread + i;
    if (igroup < num_expert) {
      int iseq = seqlens_ptr[igroup];
      int itile_num_gateup = (iseq + kTileM_Gateup - 1) / kTileM_Gateup;
      int itile_num_down = (iseq + kTileM_Down - 1) / kTileM_Down;
      thread_seqs[i] = iseq;
      gateup_thread_tiles[i] = itile_num_gateup;
      down_thread_tiles[i] = itile_num_down;
      gateup_tiles_ptr[igroup] = itile_num_gateup;
      down_tiles_ptr[igroup] = itile_num_down;
    } else {
      thread_seqs[i] = 0;
      gateup_thread_tiles[i] = 0;
      down_thread_tiles[i] = 0;
    }
  }
  using BlockScan = cub::BlockScan<int, kThreadPerBlock>;
  __shared__ typename BlockScan::TempStorage temp_storage1;
  __shared__ typename BlockScan::TempStorage temp_storage2;
  int seqs_aggregate, gateup_tiles_aggregate, down_tiles_aggregate;
  BlockScan(temp_storage1).ExclusiveSum(thread_seqs, thread_seqs, seqs_aggregate);
  BlockScan(temp_storage2)
      .ExclusiveSum(gateup_thread_tiles, gateup_thread_tiles, gateup_tiles_aggregate);
  BlockScan(temp_storage2).ExclusiveSum(down_thread_tiles, down_thread_tiles, down_tiles_aggregate);

  for (int i = idx; i < num_expert; i += blockDim.x) {
    x_row_map_race_pos_ptr[i] = 0;
  }

#pragma unroll
  for (int i = 0; i < kGroupPerThread; i++) {
    int igroup = idx * kGroupPerThread + i;
    if (igroup < num_expert) {
      cu_seqlens_ptr[igroup] = thread_seqs[i];
      gateup_cu_tiles_ptr[igroup] = gateup_thread_tiles[i];
      down_cu_tiles_ptr[igroup] = down_thread_tiles[i];
    }
  }
  if (idx == 0) {
    cu_seqlens_ptr[num_expert] = seqs_aggregate;
    gateup_cu_tiles_ptr[num_expert] = gateup_tiles_aggregate;
    down_cu_tiles_ptr[num_expert] = down_tiles_aggregate;
  }
  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

template <int kThreadPerBlock, int kMaxNumTopk, typename Tin, typename Tout, typename Tout_Gateup,
          typename GateUpTmaX, typename GateUpTmaY, typename DownTmaX, typename DownTmaY,
          bool kUsePDL = false>
__global__ void route_row_map_kernel(
    const int *topk_ids_ptr, int *topk_pos_ptr, const float *topk_scale_ptr, int *seqlens_ptr,
    const int *cu_seqlens_ptr, const vec_t<cute::TmaDescriptor, 4> td_xy,
    cute::TmaDescriptor *gate_up_tma_xy, cute::TmaDescriptor *down_tma_xy,
    const Tin *gate_up_input_ptr, Tout_Gateup *gate_up_output_ptr, Tin *down_input_ptr,
    Tout *down_output_ptr, int *row_map_race_pos_ptr, int gateup_k, int gateup_n, int down_k,
    int down_n, int total_num_topk, int num_topk, int num_expert, int start_expert, int end_expert,
    cutlass::FastDivmod topk_divider, int *x_row_map_ptr, float *weight_scale_row_map_ptr) {
  const int tid = threadIdx.x;
  // int ilane = tid % 32;
  int iwarp = tid / 32;
  bool elected = cute::elect_one_sync();
  const int gid = blockIdx.x * blockDim.x + tid;
  const int num_seq = topk_divider.divide(total_num_topk);  // total_num_topk / num_topk
  const bool active = (gid < num_seq);
  const int iseq = gid;

  const int route_blocks = gridDim.x - num_expert;
  const int igroup = blockIdx.x - route_blocks;

  extern __shared__ int smem_raw[];
  int *counter_shm = smem_raw;
  int *cu_seqlens_shm = smem_raw + num_expert;

  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }

#pragma unroll 1
  for (int i = tid; i < num_expert; i += kThreadPerBlock) {
    counter_shm[i] = 0;
    cu_seqlens_shm[i] = cu_seqlens_ptr[i];
  }
  __syncthreads();

  if (blockIdx.x >= route_blocks) {
    using namespace cute;  // NOLINT
    // update tma blocks
    __shared__ cute::TmaDescriptor smem_tma_desc[4];

    int num_seq = seqlens_ptr[igroup];
    uint64_t cu_seqlen = cu_seqlens_ptr[igroup];

    if (tid < 4) {
      smem_tma_desc[tid] = td_xy[tid];
    }
    __syncthreads();

    if (iwarp == 0) {
      auto *x_gate_up_ibatch_ptr = gate_up_input_ptr + cu_seqlen * gateup_k;
      auto gX = make_tensor(make_gmem_ptr(x_gate_up_ibatch_ptr), make_shape(num_seq, gateup_k),
                            make_stride(gateup_k, Int<1>{}));
      if (elected) {
        update_tma_gtensor<GateUpTmaX, decltype(gX), true, true>(smem_tma_desc[0], gX);
      }
      __syncwarp();
      if (elected) {
        tma_desc_commit_group();
        tma_desc_wait_group();
      }
      tma_descriptor_cp_fence_release(gate_up_tma_xy + igroup * 2, smem_tma_desc[0]);
    }

    if (iwarp == 1) {
      auto *y_gate_up_ibatch_ptr = gate_up_output_ptr + cu_seqlen * gateup_n;
      auto gY = make_tensor(make_gmem_ptr(y_gate_up_ibatch_ptr), make_shape(gateup_n, num_seq),
                            make_stride(Int<1>{}, gateup_n));
      if (elected) {
        update_tma_gtensor<GateUpTmaY, decltype(gY), true, true>(smem_tma_desc[1], gY);
      }
      __syncwarp();
      if (elected) {
        tma_desc_commit_group();
        tma_desc_wait_group();
      }
      tma_descriptor_cp_fence_release(gate_up_tma_xy + igroup * 2 + 1, smem_tma_desc[1]);
    }

    if (iwarp == 2) {
      auto *x_down_ibatch_ptr = down_input_ptr + cu_seqlen * down_k;
      auto gX = make_tensor(make_gmem_ptr(x_down_ibatch_ptr), make_shape(num_seq, down_k),
                            make_stride(down_k, Int<1>{}));
      if (elected) {
        update_tma_gtensor<DownTmaX, decltype(gX), true, true>(smem_tma_desc[2], gX);
      }
      __syncwarp();
      if (elected) {
        tma_desc_commit_group();
        tma_desc_wait_group();
      }
      tma_descriptor_cp_fence_release(down_tma_xy + igroup * 2, smem_tma_desc[2]);
    }

    if (iwarp == 3) {
      auto *y_down_ibatch_ptr = down_output_ptr + cu_seqlen * down_n;
      auto gY = make_tensor(make_gmem_ptr(y_down_ibatch_ptr), make_shape(down_n, num_seq),
                            make_stride(Int<1>{}, down_n));
      if (elected) {
        update_tma_gtensor<DownTmaY, decltype(gY), true, true>(smem_tma_desc[3], gY);
      }
      __syncwarp();
      if (elected) {
        tma_desc_commit_group();
        tma_desc_wait_group();
      }
      tma_descriptor_cp_fence_release(down_tma_xy + igroup * 2 + 1, smem_tma_desc[3]);
    }
  } else {
    // route row map blocks
    int local_expert_k[kMaxNumTopk];
    float local_expert_scale[kMaxNumTopk];
    int local_pos_k[kMaxNumTopk];

    // Stage A: shared-memory atomicAdd.
    if (active) {
      const int itopk_base = iseq * num_topk;
#pragma unroll
      for (int k = 0; k < kMaxNumTopk; ++k) {
        if (k >= num_topk) {
          break;
        }
        const int iexpert = topk_ids_ptr[itopk_base + k];
        if ((iexpert >= start_expert) && (iexpert < end_expert)) {
          const int local_expert = iexpert - start_expert;
          local_expert_k[k] = local_expert;
          local_expert_scale[k] = topk_scale_ptr[itopk_base + k];
          local_pos_k[k] = atomicAdd(&counter_shm[local_expert], 1);
        } else {
          local_expert_k[k] = -1;
          local_expert_scale[k] = 0;
          local_pos_k[k] = 0;
        }
      }
    }

    __syncthreads();

    // Stage B: promote per-block counts to global bases. For every expert with count > 0,
    // exactly one thread does a single global atomicAdd and overwrites counter_shm[e] with
    // the base. Spreading experts across threads parallelizes the global-atomic traffic.
#pragma unroll 1
    for (int e = tid; e < num_expert; e += kThreadPerBlock) {
      int cnt = counter_shm[e];
      if (cnt > 0) {
        int base = atomicAdd(&row_map_race_pos_ptr[e], cnt);
        counter_shm[e] = base;  // reinterpret: counter_shm[e] now holds this block's base for e.
      }
    }

    __syncthreads();

    // Stage C: emit topk_pos and x_row_map using (base + local_pos). No atomics here.
    if (active) {
      const int itopk_base = iseq * num_topk;
#pragma unroll
      for (int k = 0; k < kMaxNumTopk; ++k) {
        if (k >= num_topk) {
          break;
        }
        const int local_expert = local_expert_k[k];
        if (local_expert >= 0) {
          const int itopk = itopk_base + k;
          const uint64_t irow = static_cast<uint64_t>(cu_seqlens_shm[local_expert]) +
                                static_cast<uint64_t>(counter_shm[local_expert]) +
                                static_cast<uint64_t>(local_pos_k[k]);
          topk_pos_ptr[itopk] = irow;
          x_row_map_ptr[irow] = iseq;
          weight_scale_row_map_ptr[irow] = local_expert_scale[k];
        } else {
          topk_pos_ptr[itopk_base + k] = -1;
        }
      }
    }
  }

  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

template <int kThreadPerBlock, int kMaxNumTopk, typename Tin, typename Tout, typename Tout_Gateup,
          typename GateUpTmaX, typename GateUpTmaY, typename DownTmaX, typename DownTmaY,
          bool kUsePDL = false>
__global__ void route_row_map_kernel_mxfp8(
    const int *topk_ids_ptr, int *topk_pos_ptr, const float *topk_scale_ptr, int *seqlens_ptr,
    const int *cu_seqlens_ptr, const vec_t<cute::TmaDescriptor, 4> td_xy,
    cute::TmaDescriptor *gate_up_tma_xy, cute::TmaDescriptor *down_tma_xy,
    const Tin *gate_up_input_ptr, Tout_Gateup *gate_up_output_ptr, Tin *down_input_ptr,
    Tout *down_output_ptr, int *x_row_map_race_pos_ptr, int gateup_k, int gateup_n, int down_k,
    int down_n, int total_num_topk, int num_topk, int num_expert_local, int start_expert,
    int end_expert, cutlass::FastDivmod topk_divider, int *gateup_x_row_map_ptr,
    float *weight_scale_row_map_ptr) {
  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }
  const int tid = threadIdx.x;
  bool elected = cute::elect_one_sync();
  const int gid = blockIdx.x * kThreadPerBlock + tid;
  const int num_seq = topk_divider.divide(total_num_topk);
  const bool active = (gid < num_seq);
  const int iseq = gid;

  const int route_blocks = gridDim.x - num_expert_local;
  const int igroup = blockIdx.x - route_blocks;

  extern __shared__ int smem_raw[];
  int *counter_shm = smem_raw;
  int *cu_seqlens_shm = smem_raw + num_expert_local;

#pragma unroll 1
  for (int i = tid; i < num_expert_local; i += kThreadPerBlock) {
    counter_shm[i] = 0;
    cu_seqlens_shm[i] = cu_seqlens_ptr[i];
  }
  __syncthreads();

  if (blockIdx.x >= route_blocks) {
    // TMA descriptor update blocks (one block per expert)
    using namespace cute;  // NOLINT
    __shared__ cute::TmaDescriptor smem_tma_desc[4];

    int num_seq_grp = cu_seqlens_ptr[igroup + 1] - cu_seqlens_ptr[igroup];
    uint64_t cu_seqlen = cu_seqlens_ptr[igroup];

    auto *x_gate_up_ibatch = gate_up_input_ptr + cu_seqlen * gateup_k;
    auto *y_gate_up_ibatch = gate_up_output_ptr + cu_seqlen * gateup_n;
    auto *x_down_ibatch = down_input_ptr + cu_seqlen * down_k;
    auto *y_down_ibatch = down_output_ptr + cu_seqlen * down_n;

    if (tid < 4) {
      smem_tma_desc[tid] = td_xy[tid];
    }
    __syncwarp();

    if (tid == 0) {
      auto gX = make_tensor(make_gmem_ptr(x_gate_up_ibatch), make_shape(num_seq_grp, gateup_k),
                            make_stride(gateup_k, Int<1>{}));
      update_tma_gtensor<GateUpTmaX, decltype(gX), true, true>(smem_tma_desc[0], gX);
    } else if (tid == 1) {
      auto gY = make_tensor(make_gmem_ptr(y_gate_up_ibatch), make_shape(gateup_n, num_seq_grp),
                            make_stride(Int<1>{}, gateup_n));
      update_tma_gtensor<GateUpTmaY, decltype(gY), true, true>(smem_tma_desc[1], gY);
    } else if (tid == 2) {
      auto gX = make_tensor(make_gmem_ptr(x_down_ibatch), make_shape(num_seq_grp, down_k),
                            make_stride(down_k, Int<1>{}));
      update_tma_gtensor<DownTmaX, decltype(gX), true, true>(smem_tma_desc[2], gX);
    } else if (tid == 3) {
      auto gY = make_tensor(make_gmem_ptr(y_down_ibatch), make_shape(down_n, num_seq_grp),
                            make_stride(Int<1>{}, down_n));
      update_tma_gtensor<DownTmaY, decltype(gY), true, true>(smem_tma_desc[3], gY);
    }

#pragma unroll
    for (int i = 0; i < 2; ++i) {
      __syncwarp();
      if (cute::elect_one_sync()) {
        cute::tma_desc_commit_group();
        cute::tma_desc_wait_group();
      }
      tma_descriptor_cp_fence_release(gate_up_tma_xy + igroup * 2 + i, smem_tma_desc[i]);
      tma_descriptor_cp_fence_release(down_tma_xy + igroup * 2 + i, smem_tma_desc[i + 2]);
    }
  } else {
    // Route blocks: 3-stage routing
    int local_expert_k[kMaxNumTopk];
    int local_pos_k[kMaxNumTopk];

    // Stage A: block-local shared-memory atomicAdd
    if (active) {
      const int itopk_base = iseq * num_topk;
#pragma unroll
      for (int k = 0; k < kMaxNumTopk; ++k) {
        if (k >= num_topk) {
          break;
        }
        const int iexpert = topk_ids_ptr[itopk_base + k];
        if (iexpert >= start_expert && iexpert < end_expert) {
          const int local_expert = iexpert - start_expert;
          local_expert_k[k] = local_expert;
          local_pos_k[k] = atomicAdd(&counter_shm[local_expert], 1);
        } else {
          local_expert_k[k] = -1;
          local_pos_k[k] = 0;
        }
      }
    }

    __syncthreads();

    // Stage B: promote per-block counts to global bases
#pragma unroll 1
    for (int e = tid; e < num_expert_local; e += kThreadPerBlock) {
      int cnt = counter_shm[e];
      if (cnt > 0) {
        int base = atomicAdd(&x_row_map_race_pos_ptr[e], cnt);
        counter_shm[e] = base;
      }
    }

    __syncthreads();

    // Stage C: emit topk_pos + x_row_map (SFX prepack is done separately)
    if (active) {
      const int itopk_base = iseq * num_topk;
#pragma unroll
      for (int k = 0; k < kMaxNumTopk; ++k) {
        if (k >= num_topk) {
          break;
        }
        const int local_expert = local_expert_k[k];
        if (local_expert >= 0) {
          const int itopk = itopk_base + k;
          const uint64_t irow = static_cast<uint64_t>(cu_seqlens_shm[local_expert]) +
                                static_cast<uint64_t>(counter_shm[local_expert]) +
                                static_cast<uint64_t>(local_pos_k[k]);
          topk_pos_ptr[itopk] = static_cast<int>(irow);
          gateup_x_row_map_ptr[irow] = iseq;
        } else {
          topk_pos_ptr[itopk_base + k] = -1;
        }
      }
    }
  }
  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

template <int kThreadPerBlock, int kClusterSize, int kGroupPerThread, int kTileM_Gateup,
          int kTileM_Down, int kTmaUpdateExpertsPerBlock, typename Tin, typename Tout,
          typename Tout_Gateup, typename GateUpTmaX, typename GateUpTmaY, typename DownTmaX,
          typename DownTmaY, bool kUsePDL = false>
__global__ void fused_count_cuseq_route_kernel(
    const int *__restrict__ topk_ids_ptr, int *__restrict__ topk_pos_ptr,
    const float *__restrict__ topk_scale_ptr, int *__restrict__ seqlens_ptr,
    int *__restrict__ cu_seqlens_ptr, int *__restrict__ gateup_tiles_ptr,
    int *__restrict__ gateup_cu_tiles_ptr, int *__restrict__ down_tiles_ptr,
    int *__restrict__ down_cu_tiles_ptr, int *__restrict__ x_row_map_ptr,
    float *__restrict__ weight_scale_row_map_ptr, const vec_t<cute::TmaDescriptor, 4> td_xy,
    cute::TmaDescriptor *gate_up_tma_xy, cute::TmaDescriptor *down_tma_xy,
    const Tin *gate_up_input_ptr, Tout_Gateup *gate_up_output_ptr, Tin *down_input_ptr,
    Tout *down_output_ptr, int gateup_k, int gateup_n, int down_k, int down_n, int num_seq,
    int num_topk, int total_num_topk, int num_expert, int start_expert, int end_expert,
    int update_tma_blocks, cutlass::FastDivmod topk_divider) {
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();

  const int tid = threadIdx.x;
  const int block_rank = cluster.block_rank();                 // 0 .. kClusterSize-1
  const int cluster_tid = block_rank * kThreadPerBlock + tid;  // 0 .. cluster_threads-1
  constexpr int kClusterThreads = kThreadPerBlock * kClusterSize;

  int igroup_updata_tma = (blockIdx.x - kClusterSize) * kTmaUpdateExpertsPerBlock + tid / 128;
  int iupdata_tma = tid / 128;
  int idx_updata_tma = tid % 128;
  int iwarp_updata_tma = idx_updata_tma / 32;
  bool elected = cute::elect_one_sync();

  extern __shared__ int smem_raw[];
  int *counter_shm = smem_raw;
  int *cu_seqlens_shm = smem_raw + num_expert;
  int *block_base_shm = smem_raw + 2 * num_expert + 1;  // size: kClusterSize * num_expert

  // Stage 0: every block zeroes its OWN counter_shm.  This is purely local
  // shared memory; no DSMEM involved.
#pragma unroll 1
  for (int i = tid; i < num_expert; i += kThreadPerBlock) {
    counter_shm[i] = 0;
  }
  __syncthreads();

  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }

  // -----------------------------------------------------------------------
  // Stage A: count tokens into this block's own counter_shm.  One flat
  // topk_ids position per thread-iter; blocks stride by kClusterThreads over
  // the total_num_topk positions.  Block-local atomicAdd — no DSMEM atomic.
  // -----------------------------------------------------------------------
#pragma unroll 1
  for (int pos = cluster_tid; pos < total_num_topk; pos += kClusterThreads) {
    int iexpert = topk_ids_ptr[pos];
    if ((iexpert >= start_expert) && (iexpert < end_expert)) {
      atomicAdd(&counter_shm[iexpert - start_expert], 1);
    }
  }
  __syncthreads();  // make all Stage A writes visible within the block
  cluster.sync();   // make all blocks' Stage A writes visible cluster-wide

  // -----------------------------------------------------------------------
  // Stage B (rank-0 only): gather each block's per-expert count via DSMEM
  // loads, accumulate the total per-expert count, compute the per-block
  // prefix base per expert, do a block-wide exclusive scan for cu_seqlens
  // and cu_tiles, and publish all the global metadata.
  // -----------------------------------------------------------------------

  int thread_seqs[kGroupPerThread];
  int gateup_thread_tiles[kGroupPerThread];
  int down_thread_tiles[kGroupPerThread];

#pragma unroll
  for (int i = 0; i < kGroupPerThread; i++) {
    int igroup = tid * kGroupPerThread + i;
    if (igroup < num_expert) {
      // Sum per-expert counts across all kClusterSize blocks via DSMEM reads,
      // and simultaneously build the per-block exclusive-prefix base into
      // block_base_shm[r][igroup] (stored as block_base_shm[r*num_expert+igroup]).
      int acc = 0;
#pragma unroll
      for (int r = 0; r < kClusterSize; ++r) {
        int v = cluster.map_shared_rank(counter_shm, r)[igroup];
        block_base_shm[r * num_expert + igroup] = acc;
        acc += v;
      }
      int itile_num_gateup = (acc + kTileM_Gateup - 1) / kTileM_Gateup;
      int itile_num_down = (acc + kTileM_Down - 1) / kTileM_Down;
      thread_seqs[i] = acc;
      gateup_thread_tiles[i] = itile_num_gateup;
      down_thread_tiles[i] = itile_num_down;
      // Publish the per-expert seqlen and tile counts to global.
      if (blockIdx.x == 0) {
        seqlens_ptr[igroup] = acc;
        gateup_tiles_ptr[igroup] = itile_num_gateup;
        down_tiles_ptr[igroup] = itile_num_down;
      }
    } else {
      thread_seqs[i] = 0;
      gateup_thread_tiles[i] = 0;
      down_thread_tiles[i] = 0;
    }
  }

  using BlockScan = cub::BlockScan<int, kThreadPerBlock>;
  __shared__ typename BlockScan::TempStorage temp_storage_seqs;
  __shared__ typename BlockScan::TempStorage temp_storage_tiles;
  int seqs_aggregate, gateup_tiles_aggregate, down_tiles_aggregate;
  BlockScan(temp_storage_seqs).ExclusiveSum(thread_seqs, thread_seqs, seqs_aggregate);
  BlockScan(temp_storage_tiles)
      .ExclusiveSum(gateup_thread_tiles, gateup_thread_tiles, gateup_tiles_aggregate);
  BlockScan(temp_storage_tiles)
      .ExclusiveSum(down_thread_tiles, down_thread_tiles, down_tiles_aggregate);

#pragma unroll
  for (int i = 0; i < kGroupPerThread; i++) {
    int igroup = tid * kGroupPerThread + i;
    if (igroup < num_expert) {
      if (blockIdx.x == 0) {
        cu_seqlens_ptr[igroup] = thread_seqs[i];
        gateup_cu_tiles_ptr[igroup] = gateup_thread_tiles[i];
        down_cu_tiles_ptr[igroup] = down_thread_tiles[i];
      }
      cu_seqlens_shm[igroup] = thread_seqs[i];  // for Stage C DSMEM reads
    }
  }
  if (tid == 0) {
    if (blockIdx.x == 0) {
      cu_seqlens_ptr[num_expert] = seqs_aggregate;
      gateup_cu_tiles_ptr[num_expert] = gateup_tiles_aggregate;
      down_cu_tiles_ptr[num_expert] = down_tiles_aggregate;
    }
    cu_seqlens_shm[num_expert] = seqs_aggregate;
  }

  // Make sure rank-0 has finished all DSMEM reads from every block's
  // counter_shm before any block resets its counter_shm for Stage C.
  cluster.sync();

  if (blockIdx.x >= kClusterSize && blockIdx.x < kClusterSize + update_tma_blocks) {
    // updata tma
    using namespace cute;  // NOLINT
    __shared__ cute::TmaDescriptor smem_tma_desc[4 * kTmaUpdateExpertsPerBlock];

    int num_seq = 0;
    uint64_t cu_seqlen = 0;
    if (igroup_updata_tma < num_expert) {
      num_seq = cu_seqlens_shm[igroup_updata_tma + 1] - cu_seqlens_shm[igroup_updata_tma];
      cu_seqlen = cu_seqlens_shm[igroup_updata_tma];
    }

    if (idx_updata_tma < 4) {
      smem_tma_desc[4 * iupdata_tma + idx_updata_tma] = td_xy[idx_updata_tma];
    }
    __syncthreads();

    if (igroup_updata_tma < num_expert) {
      if (iwarp_updata_tma == 0) {
        auto *x_gate_up_ibatch_ptr = gate_up_input_ptr + cu_seqlen * gateup_k;
        auto gX = make_tensor(make_gmem_ptr(x_gate_up_ibatch_ptr), make_shape(num_seq, gateup_k),
                              make_stride(gateup_k, Int<1>{}));
        if (elected) {
          update_tma_gtensor<GateUpTmaX, decltype(gX), true, true>(smem_tma_desc[4 * iupdata_tma],
                                                                   gX);
        }
        __syncwarp();
        if (elected) {
          tma_desc_commit_group();
          tma_desc_wait_group();
        }
        tma_descriptor_cp_fence_release(gate_up_tma_xy + igroup_updata_tma * 2,
                                        smem_tma_desc[4 * iupdata_tma]);
      }

      if (iwarp_updata_tma == 1) {
        auto *y_gate_up_ibatch_ptr = gate_up_output_ptr + cu_seqlen * gateup_n;
        auto gY = make_tensor(make_gmem_ptr(y_gate_up_ibatch_ptr), make_shape(gateup_n, num_seq),
                              make_stride(Int<1>{}, gateup_n));
        if (elected) {
          update_tma_gtensor<GateUpTmaY, decltype(gY), true, true>(
              smem_tma_desc[4 * iupdata_tma + 1], gY);
        }
        __syncwarp();
        if (elected) {
          tma_desc_commit_group();
          tma_desc_wait_group();
        }
        tma_descriptor_cp_fence_release(gate_up_tma_xy + igroup_updata_tma * 2 + 1,
                                        smem_tma_desc[4 * iupdata_tma + 1]);
      }

      if (iwarp_updata_tma == 2) {
        auto *x_down_ibatch_ptr = down_input_ptr + cu_seqlen * down_k;
        auto gX = make_tensor(make_gmem_ptr(x_down_ibatch_ptr), make_shape(num_seq, down_k),
                              make_stride(down_k, Int<1>{}));
        if (elected) {
          update_tma_gtensor<DownTmaX, decltype(gX), true, true>(smem_tma_desc[4 * iupdata_tma + 2],
                                                                 gX);
        }
        __syncwarp();
        if (elected) {
          tma_desc_commit_group();
          tma_desc_wait_group();
        }
        tma_descriptor_cp_fence_release(down_tma_xy + igroup_updata_tma * 2,
                                        smem_tma_desc[4 * iupdata_tma + 2]);
      }

      if (iwarp_updata_tma == 3) {
        auto *y_down_ibatch_ptr = down_output_ptr + cu_seqlen * down_n;
        auto gY = make_tensor(make_gmem_ptr(y_down_ibatch_ptr), make_shape(down_n, num_seq),
                              make_stride(Int<1>{}, down_n));
        if (elected) {
          update_tma_gtensor<DownTmaY, decltype(gY), true, true>(smem_tma_desc[4 * iupdata_tma + 3],
                                                                 gY);
        }
        __syncwarp();
        if (elected) {
          tma_desc_commit_group();
          tma_desc_wait_group();
        }
        tma_descriptor_cp_fence_release(down_tma_xy + igroup_updata_tma * 2 + 1,
                                        smem_tma_desc[4 * iupdata_tma + 3]);
      }
    }
  } else if (blockIdx.x < kClusterSize) {
    // Reset this block's counter_shm to 0 — in Stage C it becomes this block's
    // per-expert write-offset counter (block-local atomicAdd).
#pragma unroll 1
    for (int i = tid; i < num_expert; i += kThreadPerBlock) {
      counter_shm[i] = 0;
    }
    __syncthreads();
    cluster.sync();

    // -----------------------------------------------------------------------
    // Stage C: route.  One flat position per thread-iter, partitioned exactly
    // like Stage A so per-block counts line up with per-block writes.  For each
    // position, decode (iseq, k) via FastDivmod; for in-range experts, compute
    // irow = cu_seqlens[e] + my_base[e] + local_pos, then write topk_pos and
    // x_row_map.  For out-of-range experts write topk_pos = -1.
    // -----------------------------------------------------------------------
    {
      int *block_base_shm_rank0 = cluster.map_shared_rank(block_base_shm, 0);
      int *cu_seqlens_shm_rank0_src = cluster.map_shared_rank(cu_seqlens_shm, 0);
      // Reuse our own block_base_shm row (block_rank row) as the local cache of
      // "cu + my_base" merged offsets: cache[e] = cu_seqlens[e] + block_rank's base[e].
      // Also repurpose our cu_seqlens_shm as the local cu_seqlens cache.
#pragma unroll 1
      for (int e = tid; e < num_expert; e += kThreadPerBlock) {
        cu_seqlens_shm[e] = cu_seqlens_shm_rank0_src[e];
        block_base_shm[block_rank * num_expert + e] =
            block_base_shm_rank0[block_rank * num_expert + e];
      }
      cluster.sync();
    }
    const int *my_base = block_base_shm + block_rank * num_expert;  // now local
    const int *cu_seqlens_local = cu_seqlens_shm;                   // now local

#pragma unroll 1
    for (int pos = cluster_tid; pos < total_num_topk; pos += kClusterThreads) {
      int iseq, k;
      topk_divider(iseq, k, pos);
      const int iexpert = topk_ids_ptr[pos];
      if ((iexpert >= start_expert) && (iexpert < end_expert)) {
        const int local_expert = iexpert - start_expert;
        const int local_pos = atomicAdd(&counter_shm[local_expert], 1);
        const uint64_t irow = static_cast<uint64_t>(cu_seqlens_local[local_expert]) +
                              static_cast<uint64_t>(my_base[local_expert]) +
                              static_cast<uint64_t>(local_pos);
        topk_pos_ptr[pos] = static_cast<int>(irow);
        x_row_map_ptr[irow] = iseq;
        if (weight_scale_row_map_ptr && topk_scale_ptr) {
          weight_scale_row_map_ptr[irow] = topk_scale_ptr[pos];
        }
      } else {
        topk_pos_ptr[pos] = -1;
      }
    }
  }

  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}
}  // namespace kernels
}  // namespace fuse_moe
}  // namespace hpc

#endif  // SRC_FUSE_MOE_SM100_COUNT_KERNELS_CUH_
