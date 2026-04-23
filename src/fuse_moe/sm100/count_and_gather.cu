// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include <cub/cub.cuh>

#include "cutlass/fast_math.h"
#include "src/fuse_moe/sm100/fuse_moe.h"
#include "src/utils/utils.cuh"
#include "src/utils/utils.h"

namespace hpc {
namespace fuse_moe {

namespace kernels {

template <int kThreadPerBlock, bool kUsePDL = false>
__global__ void count_seq_kernel(const int *topk_ids_ptr, int *topk_pos_ptr, int *seqlens_ptr,
                                 int total_num_topk, int num_expert, int start_expert,
                                 int end_expert) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;

  extern __shared__ int seqlens_shm[];

  for (int i = threadIdx.x; i < num_expert; i += blockDim.x) {
    seqlens_shm[i] = 0;
  }
  __syncthreads();

  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }

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

template <int kThreadPerBlock, int kGroupPerThread, int kTileM, bool kUsePDL = false>
__global__ void count_cuseq_kernel(const int *topk_ids_ptr, int *topk_pos_ptr, int *seqlens_ptr,
                                   int *cu_seqlens_ptr, int *tiles_ptr, int *cu_tiles_ptr,
                                   int num_seq, int num_topk, int total_num_topk, int num_expert,
                                   int start_expert, int end_expert) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  // cusum
  int thread_seqs[kGroupPerThread];
  int thread_tiles[kGroupPerThread];
  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }
#pragma unroll
  for (int i = 0; i < kGroupPerThread; i++) {
    int igroup = idx * kGroupPerThread + i;
    if (igroup < num_expert) {
      int iseq = seqlens_ptr[igroup];
      int itile_num = (iseq + kTileM - 1) / kTileM;
      thread_seqs[i] = iseq;
      thread_tiles[i] = itile_num;
      tiles_ptr[igroup] = itile_num;
    } else {
      thread_seqs[i] = 0;
      thread_tiles[i] = 0;
    }
  }
  using BlockScan = cub::BlockScan<int, kThreadPerBlock>;
  __shared__ typename BlockScan::TempStorage temp_storage1;
  __shared__ typename BlockScan::TempStorage temp_storage2;
  int seqs_aggregate, tiles_aggregate;
  BlockScan(temp_storage1).ExclusiveSum(thread_seqs, thread_seqs, seqs_aggregate);
  BlockScan(temp_storage2).ExclusiveSum(thread_tiles, thread_tiles, tiles_aggregate);

  // store
  // fill seqlens with zero
  for (int i = idx; i < num_expert; i += blockDim.x) {
    seqlens_ptr[i] = 0;  // seqlens_shm[i];
  }

#pragma unroll
  for (int i = 0; i < kGroupPerThread; i++) {
    int igroup = idx * kGroupPerThread + i;
    if (igroup < num_expert) {
      cu_seqlens_ptr[igroup] = thread_seqs[i];
      cu_tiles_ptr[igroup] = thread_tiles[i];
    }
  }
  if (idx == 0) {
    cu_seqlens_ptr[num_expert] = seqs_aggregate;
    cu_tiles_ptr[num_expert] = tiles_aggregate;
  }
  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

template <int kThreadPerBlock, int kGroupPerThread, int kTileM, bool kUsePDL = false>
__global__ void count_seq_and_cuseq_kernel(const int *topk_ids_ptr, int *topk_pos_ptr,
                                           int *seqlens_ptr, int *cu_seqlens_ptr, int *tiles_ptr,
                                           int *cu_tiles_ptr, int num_seq, int num_topk,
                                           int total_num_topk, int num_expert, int start_expert,
                                           int end_expert) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  extern __shared__ int seqlens_shm[];

  for (int i = idx; i < num_expert; i += blockDim.x) {
    seqlens_shm[i] = 0;
  }
  __syncthreads();

  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }

  for (int i = idx; i < total_num_topk; i += blockDim.x) {
    int iexpert = topk_ids_ptr[i];
    if ((iexpert >= start_expert) && (iexpert < end_expert)) {
      atomicAdd(&seqlens_shm[iexpert - start_expert], 1);
    }
    topk_pos_ptr[i] = -1;
  }

  __syncthreads();

  // cusum
  int thread_seqs[kGroupPerThread];
  int thread_tiles[kGroupPerThread];
#pragma unroll
  for (int i = 0; i < kGroupPerThread; i++) {
    int igroup = idx * kGroupPerThread + i;
    if (igroup < num_expert) {
      int iseq = seqlens_shm[igroup];
      int itile_num = (iseq + kTileM - 1) / kTileM;
      thread_seqs[i] = iseq;
      thread_tiles[i] = itile_num;
      tiles_ptr[igroup] = itile_num;
    } else {
      thread_seqs[i] = 0;
      thread_tiles[i] = 0;
    }
  }
  using BlockScan = cub::BlockScan<int, kThreadPerBlock>;
  __shared__ typename BlockScan::TempStorage temp_storage1;
  __shared__ typename BlockScan::TempStorage temp_storage2;
  int seqs_aggregate, tiles_aggregate;
  BlockScan(temp_storage1).ExclusiveSum(thread_seqs, thread_seqs, seqs_aggregate);
  BlockScan(temp_storage2).ExclusiveSum(thread_tiles, thread_tiles, tiles_aggregate);

  // store
  // fill seqlens with zero
  for (int i = idx; i < num_expert; i += blockDim.x) {
    seqlens_ptr[i] = 0;  // seqlens_shm[i];
  }

#pragma unroll
  for (int i = 0; i < kGroupPerThread; i++) {
    int igroup = idx * kGroupPerThread + i;
    if (igroup < num_expert) {
      cu_seqlens_ptr[igroup] = thread_seqs[i];
      cu_tiles_ptr[igroup] = thread_tiles[i];
    }
  }
  if (idx == 0) {
    cu_seqlens_ptr[num_expert] = seqs_aggregate;
    cu_tiles_ptr[num_expert] = tiles_aggregate;
  }
  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

// Lightweight routing kernel that replaces the per-itopk warp in `gather_kernel`'s copy phase
// when we don't need to physically copy tokens (i.e. the row-map fast path is enabled).
//
// Each thread handles one itopk:
//   * Look up topk_ids[itopk]
//   * Plain atomicAdd(&seqlens[e], 1) per thread (no warp-level coalescing).
//   * Write topk_pos[itopk] and x_row_map[row] in one go.
//
// Compared to the legacy gather_kernel routing phase (1 warp / itopk, 31 lanes idle),
// this brings every lane into useful work and shrinks the grid from
//   ceil(total_num_topk / 4)  ->  ceil(total_num_topk / kThreadPerBlock)
// (e.g. 128k -> ~1k blocks for batch=64k, topk=8), removing most of the launch overhead.
//
// Note: this variant intentionally does NOT use __match_any_sync / __popc to coalesce atomics,
// so it isolates the launch-shape change from the atomic-contention reduction.
template <int kThreadPerBlock, bool kUsePDL = false>
__global__ void route_row_map_kernel(const int *topk_ids_ptr, int *topk_pos_ptr, int *seqlens_ptr,
                                     const int *cu_seqlens_ptr, int total_num_topk, int num_topk,
                                     int num_expert, int start_expert, int end_expert,
                                     cutlass::FastDivmod topk_divider, int *x_row_map_ptr) {
  const int tid = threadIdx.x;
  const int gid = blockIdx.x * blockDim.x + tid;
  const int gstride = blockDim.x * gridDim.x;

  // Cache cu_seqlens in shared memory; the hot loop reads it once per itopk.
  extern __shared__ int cu_seqlens_shm[];
#pragma unroll 1
  for (int i = tid; i < num_expert; i += blockDim.x) {
    cu_seqlens_shm[i] = cu_seqlens_ptr[i];
  }
  __syncthreads();

  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }

  // Carry (iseq, irem) instead of recomputing topk_divider every iteration.
  int iseq, irem;
  topk_divider(iseq, irem, gid);
  int stride_seq, stride_rem;
  topk_divider(stride_seq, stride_rem, gstride);

#pragma unroll 1
  for (int itopk = gid; itopk < total_num_topk; itopk += gstride) {
    int iexpert = topk_ids_ptr[itopk];
    if ((iexpert >= start_expert) && (iexpert < end_expert)) {
      int local_expert = iexpert - start_expert;
      int pos_in_expert = atomicAdd(&seqlens_ptr[local_expert], 1);
      uint64_t irow = cu_seqlens_shm[local_expert] + pos_in_expert;
      topk_pos_ptr[itopk] = irow;
      x_row_map_ptr[irow] = iseq;
    }

    iseq += stride_seq;
    irem += stride_rem;
    if (irem >= num_topk) {
      irem -= num_topk;
      ++iseq;
    }
  }

  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

// Legacy gather_kernel for the non-fast-path callers (kept for w4a8 and very small num_seq
// routing where the row-map fast path isn't taken). Produces topk_pos and optionally
// x_row_map via warp-per-itopk atomicAdd routing.
//
// Historical TMA-descriptor-update and task-map-population branches have been removed: the
// downstream group_gemm kernels (sm100 fp8 / cp_async_fp8 / w4a8) run their own TMA descriptor
// update and ignore task_map_ptr, so producing either from here was pure dead work.
template <int kWarpPerBlock, bool kUsePDL = false>
__global__ void gather_kernel(const int *topk_ids_ptr, int *topk_pos_ptr, int *seqlens_ptr,
                              const int *cu_seqlens_ptr, int total_num_topk, int start_expert,
                              int end_expert, cutlass::FastDivmod topk_divider,
                              int *x_row_map_ptr = nullptr) {
  constexpr int kThreadPerWarp = 32;
  int idx = threadIdx.x;
  int iblock = blockIdx.x;
  int iwarp = idx / kThreadPerWarp;
  int ilane = idx % kThreadPerWarp;
  int itopk = iblock * kWarpPerBlock + iwarp;

  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }

  if (itopk < total_num_topk) {
    int iexpert = topk_ids_ptr[itopk];
    if ((iexpert >= start_expert) && (iexpert < end_expert)) {
      int pos_in_expert;
      if (ilane == 0) {
        pos_in_expert = atomicAdd(&seqlens_ptr[iexpert - start_expert], 1);
      }
      pos_in_expert = __shfl_sync(0xFFFFFFFF, pos_in_expert, 0);

      uint64_t irow = cu_seqlens_ptr[iexpert - start_expert] + pos_in_expert;
      int iseq, res;
      topk_divider(iseq, res, itopk);

      if (ilane == 0) {
        topk_pos_ptr[itopk] = irow;
        if (x_row_map_ptr != nullptr) {
          x_row_map_ptr[irow] = iseq;
        }
      }
    }
  }

  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

}  // namespace kernels

template <int kTileM, bool kUsePDL = false>
void launch_count_and_gather(const void *topk_ids_ptr, void *topk_pos_ptr, void *seqlens_ptr,
                             void *cu_seqlens_ptr, void *tiles_ptr, void *cu_tiles_ptr, int num_seq,
                             int num_topk, int num_expert, int eprank, cudaStream_t stream,
                             void *x_row_map_ptr = nullptr) {
  int total_num_topk = num_seq * num_topk;
  int start_expert = eprank * num_expert;
  int end_expert = (eprank + 1) * num_expert;
  int num_sm_count = get_sm_count();

  // 0. count tokens
  {
    constexpr int kThreadPerBlock = 256;
    constexpr int kGroupPerThread = 2;

    if (num_seq <= 128) {
      dim3 block(kThreadPerBlock);
      dim3 grid(1);
      cudaLaunchAttribute attribute[1];
      attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attribute[0].val.programmaticStreamSerializationAllowed = 1;

      // Set the attribute in a kernel launch configuration
      cudaLaunchConfig_t config{};

      // Base launch configuration
      config.gridDim = grid;
      config.blockDim = block;
      config.dynamicSmemBytes = num_expert * 4;
      config.stream = stream;

      // Add special attribute for PDL
      config.attrs = attribute;
      config.numAttrs = 1;
      auto kernel =
          kernels::count_seq_and_cuseq_kernel<kThreadPerBlock, kGroupPerThread, kTileM, kUsePDL>;
      cudaLaunchKernelEx(&config, kernel, (const int *)topk_ids_ptr, (int *)topk_pos_ptr,
                         (int *)seqlens_ptr, (int *)cu_seqlens_ptr, (int *)tiles_ptr,
                         (int *)cu_tiles_ptr, num_seq, num_topk, total_num_topk, num_expert,
                         start_expert, end_expert);
    } else {
      // 0.1 count seqlens
      {
        dim3 block(kThreadPerBlock);
        int num_block = (total_num_topk + kThreadPerBlock - 1) / kThreadPerBlock;
        int max_num_block = num_sm_count * 8;
        if (num_block > max_num_block) {
          num_block = max_num_block;
        }
        dim3 grid(num_block);

        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute[0].val.programmaticStreamSerializationAllowed = 1;

        // Set the attribute in a kernel launch configuration
        cudaLaunchConfig_t config{};

        // Base launch configuration
        config.gridDim = grid;
        config.blockDim = block;
        config.dynamicSmemBytes = num_expert * sizeof(int);
        config.stream = stream;

        // Add special attribute for PDL
        config.attrs = attribute;
        config.numAttrs = 1;
        auto kernel = kernels::count_seq_kernel<kThreadPerBlock, kUsePDL>;
        cudaLaunchKernelEx(&config, kernel, (const int *)topk_ids_ptr, (int *)topk_pos_ptr,
                           (int *)seqlens_ptr, total_num_topk, num_expert, start_expert,
                           end_expert);
      }
      // 0.2 count cu_seqlens
      {
        dim3 block(kThreadPerBlock);
        dim3 grid(1);

        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute[0].val.programmaticStreamSerializationAllowed = 1;

        // Set the attribute in a kernel launch configuration
        cudaLaunchConfig_t config{};

        // Base launch configuration
        config.gridDim = grid;
        config.blockDim = block;
        config.dynamicSmemBytes = num_expert * 4;
        config.stream = stream;

        // Add special attribute for PDL
        config.attrs = attribute;
        config.numAttrs = 1;
        auto kernel =
            kernels::count_cuseq_kernel<kThreadPerBlock, kGroupPerThread, kTileM, kUsePDL>;
        cudaLaunchKernelEx(&config, kernel, (const int *)topk_ids_ptr, (int *)topk_pos_ptr,
                           (int *)seqlens_ptr, (int *)cu_seqlens_ptr, (int *)tiles_ptr,
                           (int *)cu_tiles_ptr, num_seq, num_topk, total_num_topk, num_expert,
                           start_expert, end_expert);
      }
    }
  }

  // 1. route tokens (topk_pos, and optionally x_row_map).
  //
  // TMA descriptors are intentionally NOT written here: every downstream group_gemm kernel
  // (gate_up via group_gemm_cp_async_fp8_async, down via group_gemm_fp8_async, and the w4a8
  // path via group_gemm_groupwise_w4a8_mma_async which doesn't use TMA descriptors at all)
  // either runs its own update_grouped_tma kernel that fully overwrites the descriptor buffer,
  // or doesn't consume it. So any TMA-update work here is dead work.
  {
    cutlass::FastDivmod topk_divider(num_topk);

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = 1;

    const bool use_route_fast_path = x_row_map_ptr != nullptr;

    if (use_route_fast_path) {
      constexpr int kRouteThreadPerBlock = 256;
      int route_blocks = (total_num_topk + kRouteThreadPerBlock - 1) / kRouteThreadPerBlock;
      int max_route_blocks = num_sm_count * 8;
      if (route_blocks > max_route_blocks) {
        route_blocks = max_route_blocks;
      }
      if (route_blocks < 1) {
        route_blocks = 1;
      }

      cudaLaunchConfig_t route_config{};
      route_config.gridDim = dim3(route_blocks);
      route_config.blockDim = dim3(kRouteThreadPerBlock);
      route_config.dynamicSmemBytes = num_expert * sizeof(int);
      route_config.stream = stream;
      route_config.attrs = attribute;
      route_config.numAttrs = 1;

      auto route_kernel = kernels::route_row_map_kernel<kRouteThreadPerBlock, kUsePDL>;
      cudaLaunchKernelEx(&route_config, route_kernel, (const int *)topk_ids_ptr,
                         (int *)topk_pos_ptr, (int *)seqlens_ptr, (const int *)cu_seqlens_ptr,
                         total_num_topk, num_topk, num_expert, start_expert, end_expert,
                         topk_divider, (int *)x_row_map_ptr);
    } else {
      // Legacy warp-per-itopk gather_kernel.
      constexpr int kWarpPerBlock = 4;
      int num_block_for_copy = (total_num_topk + kWarpPerBlock - 1) / kWarpPerBlock;
      dim3 block(kWarpPerBlock * 32);
      dim3 grid(num_block_for_copy);

      cudaLaunchConfig_t config{};
      config.gridDim = grid;
      config.blockDim = block;
      config.dynamicSmemBytes = 0;
      config.stream = stream;
      config.attrs = attribute;
      config.numAttrs = 1;

      auto kernel = kernels::gather_kernel<kWarpPerBlock, kUsePDL>;
      cudaLaunchKernelEx(&config, kernel, (const int *)topk_ids_ptr, (int *)topk_pos_ptr,
                         (int *)seqlens_ptr, (const int *)cu_seqlens_ptr, total_num_topk,
                         start_expert, end_expert, topk_divider, (int *)x_row_map_ptr);
    }
  }
}

void count_and_gather_async(const void *topk_ids_ptr, void *topk_pos_ptr, void *seqlens_ptr,
                            void *cu_seqlens_ptr, void *tiles_ptr, void *cu_tiles_ptr, int num_seq,
                            int num_topk, int num_expert, int eprank, int num_seq_per_group_avg,
                            cudaStream_t stream, void *x_row_map_ptr) {
  constexpr bool kUsePDL = true;

  // kTileM is the M-tile size used by the downstream group GEMM; count_cuseq_kernel
  // ceil-divides seqlen by kTileM when filling tiles_ptr / cu_tiles_ptr.
  if (num_seq_per_group_avg <= 8) {
    launch_count_and_gather</*kTileM=*/8, kUsePDL>(
        topk_ids_ptr, topk_pos_ptr, seqlens_ptr, cu_seqlens_ptr, tiles_ptr, cu_tiles_ptr, num_seq,
        num_topk, num_expert, eprank, stream, x_row_map_ptr);
  } else if (num_seq_per_group_avg <= 16) {
    launch_count_and_gather</*kTileM=*/16, kUsePDL>(
        topk_ids_ptr, topk_pos_ptr, seqlens_ptr, cu_seqlens_ptr, tiles_ptr, cu_tiles_ptr, num_seq,
        num_topk, num_expert, eprank, stream, x_row_map_ptr);
  } else if (num_seq_per_group_avg <= 32) {
    launch_count_and_gather</*kTileM=*/32, kUsePDL>(
        topk_ids_ptr, topk_pos_ptr, seqlens_ptr, cu_seqlens_ptr, tiles_ptr, cu_tiles_ptr, num_seq,
        num_topk, num_expert, eprank, stream, x_row_map_ptr);
  } else if (num_seq_per_group_avg <= 48) {
    launch_count_and_gather</*kTileM=*/48, kUsePDL>(
        topk_ids_ptr, topk_pos_ptr, seqlens_ptr, cu_seqlens_ptr, tiles_ptr, cu_tiles_ptr, num_seq,
        num_topk, num_expert, eprank, stream, x_row_map_ptr);
  } else {
    launch_count_and_gather</*kTileM=*/64, kUsePDL>(
        topk_ids_ptr, topk_pos_ptr, seqlens_ptr, cu_seqlens_ptr, tiles_ptr, cu_tiles_ptr, num_seq,
        num_topk, num_expert, eprank, stream, x_row_map_ptr);
  }
}

}  // namespace fuse_moe
}  // namespace hpc
