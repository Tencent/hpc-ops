// Copyright 2026 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include <cub/cub.cuh>

#include "src/fuse_moe/sm90/cp_async/fuse_moe_cp_async.h"
#include "src/group_gemm/sm90/cp_async/build_task_map.h"
#include "src/utils/utils.cuh"
#include "src/utils/utils.h"

namespace hpc {
namespace fuse_moe_cp_async {

namespace kernels {

// Kernel 1 (large num_seq path): count tokens per expert and initialize topk_pos to -1.
// One thread per topk element.
template <bool kUsePDL = false>
__global__ void count_seq_kernel(const int *topk_ids_ptr, int *topk_pos_ptr, int *seqlens_ptr,
                                 int total_num_topk, int start_expert, int end_expert) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }
  if (idx < total_num_topk) {
    int iexpert = topk_ids_ptr[idx];
    if ((iexpert >= start_expert) && (iexpert < end_expert)) {
      atomicAdd(&seqlens_ptr[iexpert - start_expert], 1);
    }
    topk_pos_ptr[idx] = -1;
  }
  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

// Kernel 2a (large num_seq path): compute cu_seqlens, cu_tiles from seqlens.
// Single block, uses CUB BlockScan.  Also resets seqlens_ptr to 0 for reuse as per-expert
// atomic counters in the index-build kernel.
template <int kThreadPerBlock, int kGroupPerThread, int kTileM, bool kUsePDL = false>
__global__ void count_cuseq_kernel(int *seqlens_ptr, int *cu_seqlens_ptr, int *tiles_ptr,
                                   int *cu_tiles_ptr, int num_expert) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

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

  // Reset seqlens to 0 so the subsequent index-build kernel can reuse them as counters.
  for (int i = idx; i < num_expert; i += blockDim.x) {
    seqlens_ptr[i] = 0;
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

// Kernel 2b (large num_seq path): build row_indices and topk_pos.
// Reuses seqlens_ptr (must be zeroed) as per-expert atomic counters.
//   row_indices[cu_seqlens[e] + slot] = original token row index
//   topk_pos[token_idx * num_topk + topk_j] = cu_seqlens[e] + slot
//
// When gateup_task_map / down_task_map are non-null, extra tail blocks (one per expert)
// fill the per-gemm task_map slices as a fused side-effect.  The grid layout is:
//   [0 .. num_topk_blocks)           : token-index build (per-thread topk work)
//   [0 .. num_topk_blocks)           : row_indices / topk_pos build
//   [num_topk_blocks .. num_topk_blocks + num_expert)
//                                    : task_map build (blockIdx - num_topk_blocks == igroup)
//   [num_topk_blocks + num_expert]   : task_map tail: write sentinel -1 into slots past
//                                      cu_tiles_ptr[num_expert] * num_tile_n, so callers
//                                      can skip cudaMemsetAsync(0xFF) of the buffers.
template <bool kUsePDL = false>
__global__ void build_indices_kernel(const int *topk_ids_ptr, int *row_indices_ptr,
                                     int *topk_pos_ptr, int *seqlens_ptr,  // used as counters
                                     const int *cu_seqlens_ptr, const int *cu_tiles_ptr,
                                     const int *tiles_ptr, int4 *gateup_task_map,
                                     int4 *down_task_map, int gate_up_num_tile_n,
                                     int down_num_tile_n, int gateup_task_map_len,
                                     int down_task_map_len, int total_num_topk, int num_topk,
                                     int num_expert, int num_topk_blocks, int start_expert,
                                     int end_expert) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }
  if (blockIdx.x < num_topk_blocks) {
    if (idx < total_num_topk) {
      int iexpert = topk_ids_ptr[idx];
      if ((iexpert >= start_expert) && (iexpert < end_expert)) {
        int local_expert = iexpert - start_expert;
        int slot = atomicAdd(&seqlens_ptr[local_expert], 1);
        int pos = cu_seqlens_ptr[local_expert] + slot;
        // token index and topk sub-index
        int token_idx = idx / num_topk;
        int topk_j = idx % num_topk;
        row_indices_ptr[pos] = token_idx;
        topk_pos_ptr[token_idx * num_topk + topk_j] = pos;
      }
      // topk_pos was already set to -1 in count_seq_kernel for tokens outside this EP rank
    }
  } else {
    int task_idx = blockIdx.x - num_topk_blocks;
    if (task_idx < num_expert) {
      // task_map build: one block per expert
      int igroup = task_idx;
      if (gateup_task_map != nullptr || down_task_map != nullptr) {
        int start = cu_tiles_ptr[igroup];
        int num_tm = tiles_ptr[igroup];
        if (gateup_task_map != nullptr) {
          int total = num_tm * gate_up_num_tile_n;
          for (int i = threadIdx.x; i < total; i += blockDim.x) {
            int itm = i / gate_up_num_tile_n;
            int itn = i - itm * gate_up_num_tile_n;
            int off = (start + itm) * gate_up_num_tile_n + itn;
            int4 v;
            v.x = igroup;
            v.y = itm;
            v.z = itn;
            v.w = 0;
            gateup_task_map[off] = v;
          }
        }
        if (down_task_map != nullptr) {
          int total = num_tm * down_num_tile_n;
          for (int i = threadIdx.x; i < total; i += blockDim.x) {
            int itm = i / down_num_tile_n;
            int itn = i - itm * down_num_tile_n;
            int off = (start + itm) * down_num_tile_n + itn;
            int4 v;
            v.x = igroup;
            v.y = itm;
            v.z = itn;
            v.w = 0;
            down_task_map[off] = v;
          }
        }
      }
    } else {
      // Tail blocks cooperatively write sentinel -1 to the unused task-map
      // suffix. `tail_id` is in [0, tail_blocks).
      int tail_id = task_idx - num_expert;
      int tail_blocks = gridDim.x - num_topk_blocks - num_expert;
      int total_used = cu_tiles_ptr[num_expert];
      int4 sentinel;
      sentinel.x = -1;
      sentinel.y = 0;
      sentinel.z = 0;
      sentinel.w = 0;
      int stride = tail_blocks * blockDim.x;
      if (gateup_task_map != nullptr) {
        int used = total_used * gate_up_num_tile_n;
        for (int i = used + tail_id * blockDim.x + threadIdx.x; i < gateup_task_map_len;
             i += stride) {
          gateup_task_map[i] = sentinel;
        }
      }
      if (down_task_map != nullptr) {
        int used = total_used * down_num_tile_n;
        for (int i = used + tail_id * blockDim.x + threadIdx.x; i < down_task_map_len;
             i += stride) {
          down_task_map[i] = sentinel;
        }
      }
    }
  }
  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

// Combined kernel (small num_seq path): count, prefix-sum, build row_indices and topk_pos
// in a single kernel using shared memory for counting. Single block only.
// task_map pointers are accepted for signature symmetry but ignored; task_map
// build is performed by `build_two_task_maps_kernel` instead.
template <int kThreadPerBlock, int kGroupPerThread, int kTileM, bool kUsePDL = false>
__global__ void count_and_build_kernel(const int *topk_ids_ptr, int *row_indices_ptr,
                                       int *topk_pos_ptr, int *seqlens_ptr, int *cu_seqlens_ptr,
                                       int *tiles_ptr, int *cu_tiles_ptr, int4 *gateup_task_map,
                                       int4 *down_task_map, int gate_up_num_tile_n,
                                       int down_num_tile_n, int num_seq, int num_topk,
                                       int total_num_topk, int num_expert, int start_expert,
                                       int end_expert) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  // Phase 1: count tokens per expert in shared memory
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
    topk_pos_ptr[i] = -1;  // default: token not handled by this EP rank
  }
  __syncthreads();

  // Phase 2: prefix sum over seqlens (single block CUB scan)
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

  // Store cu_seqlens and cu_tiles
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

  // Reset shared counters for phase 3
  for (int i = idx; i < num_expert; i += blockDim.x) {
    seqlens_shm[i] = 0;
  }
  __syncthreads();

  // Phase 3: build row_indices and topk_pos using shared counter atomics
  for (int i = idx; i < total_num_topk; i += blockDim.x) {
    int iexpert = topk_ids_ptr[i];
    if ((iexpert >= start_expert) && (iexpert < end_expert)) {
      int local_expert = iexpert - start_expert;
      int slot = atomicAdd(&seqlens_shm[local_expert], 1);
      int pos = cu_seqlens_ptr[local_expert] + slot;
      int token_idx = i / num_topk;
      int topk_j = i % num_topk;
      row_indices_ptr[pos] = token_idx;
      topk_pos_ptr[token_idx * num_topk + topk_j] = pos;
    }
  }
  __syncthreads();

  // Write back seqlens from shared memory to global memory so that
  // downstream GEMM kernels can read the correct per-expert token counts.
  for (int i = idx; i < num_expert; i += blockDim.x) {
    seqlens_ptr[i] = seqlens_shm[i];
  }

  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

}  // namespace kernels

// ---------------------------------------------------------------------------
// Helper: launch config with PDL attribute
// ---------------------------------------------------------------------------
static cudaLaunchConfig_t make_pdl_config(dim3 grid, dim3 block, size_t smem, cudaStream_t stream,
                                          cudaLaunchAttribute *attr_out) {
  attr_out[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr_out[0].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cfg.attrs = attr_out;
  cfg.numAttrs = 1;
  return cfg;
}

// ---------------------------------------------------------------------------
// Template dispatcher: selects kTileM based on average tokens per expert
// ---------------------------------------------------------------------------
template <int kTileM, bool kUsePDL>
void launch_count_and_build(const int *topk_ids_ptr, int *row_indices_ptr, int *topk_pos_ptr,
                            int *seqlens_ptr, int *cu_seqlens_ptr, int *tiles_ptr,
                            int *cu_tiles_ptr, int4 *gateup_task_map, int4 *down_task_map,
                            int gate_up_num_tile_n, int down_num_tile_n, int gateup_task_map_len,
                            int down_task_map_len, int num_seq, int num_topk, int num_expert,
                            int start_expert, int end_expert, cudaStream_t stream) {
  constexpr int kThreadPerBlock = 256;
  constexpr int kGroupPerThread = 2;
  int total_num_topk = num_seq * num_topk;

  cudaLaunchAttribute attr[1];

  if (num_seq <= 128) {
    // Small-batch path: count/prefix-sum/build_indices in a single block,
    // then launch build_two_task_maps separately to fill both task_maps
    // in parallel.  task_map pointers are passed as nullptr to the fused
    // kernel so it skips the task_map stage.
    dim3 block(kThreadPerBlock);
    dim3 grid(1);
    auto cfg = make_pdl_config(grid, block, num_expert * sizeof(int), stream, attr);
    auto kernel =
        kernels::count_and_build_kernel<kThreadPerBlock, kGroupPerThread, kTileM, kUsePDL>;
    cudaLaunchKernelEx(&cfg, kernel, topk_ids_ptr, row_indices_ptr, topk_pos_ptr, seqlens_ptr,
                       cu_seqlens_ptr, tiles_ptr, cu_tiles_ptr,
                       /*gateup_task_map=*/nullptr, /*down_task_map=*/nullptr, gate_up_num_tile_n,
                       down_num_tile_n, num_seq, num_topk, total_num_topk, num_expert, start_expert,
                       end_expert);

    // Fill both task_maps with one kernel launch. Extra blocks write
    // sentinel -1 into unused tail entries.
    if (gateup_task_map != nullptr || down_task_map != nullptr) {
      ::hpc::group_gemm_cp_async::launch_build_two_task_maps(
          gateup_task_map, down_task_map, cu_tiles_ptr, tiles_ptr, num_expert, gate_up_num_tile_n,
          down_num_tile_n, gateup_task_map_len, down_task_map_len,
          /*use_pdl=*/kUsePDL, stream);
    }
  } else {
    // Step 0.1: count tokens per expert
    {
      dim3 block(kThreadPerBlock);
      dim3 grid((total_num_topk + kThreadPerBlock - 1) / kThreadPerBlock);
      auto cfg = make_pdl_config(grid, block, 0, stream, attr);
      auto kernel = kernels::count_seq_kernel<kUsePDL>;
      cudaLaunchKernelEx(&cfg, kernel, topk_ids_ptr, topk_pos_ptr, seqlens_ptr, total_num_topk,
                         start_expert, end_expert);
    }

    // Step 0.2: prefix sum → cu_seqlens, cu_tiles; also resets seqlens to 0
    {
      dim3 block(kThreadPerBlock);
      dim3 grid(1);
      auto cfg = make_pdl_config(grid, block, 0, stream, attr);
      auto kernel = kernels::count_cuseq_kernel<kThreadPerBlock, kGroupPerThread, kTileM, kUsePDL>;
      cudaLaunchKernelEx(&cfg, kernel, seqlens_ptr, cu_seqlens_ptr, tiles_ptr, cu_tiles_ptr,
                         num_expert);
    }

    // Step 0.3: build row_indices, topk_pos, expert task-map blocks, and
    // tail blocks for sentinel fill.
    {
      int num_topk_blocks = (total_num_topk + kThreadPerBlock - 1) / kThreadPerBlock;
      bool need_tm = (gateup_task_map != nullptr || down_task_map != nullptr);
      int tail_blocks = 0;
      if (need_tm) {
        // Approximate sentinel bytes = (gateup_task_map_len + down_task_map_len) * 16.
        int64_t total_bytes =
            (static_cast<int64_t>(gateup_task_map_len) + static_cast<int64_t>(down_task_map_len)) *
            16;
        int bytes_per_block = kThreadPerBlock * 16 * 8;  // ~32 KB
        tail_blocks = static_cast<int>((total_bytes + bytes_per_block - 1) / bytes_per_block);
        if (tail_blocks < 1) tail_blocks = 1;
        if (tail_blocks > 32) tail_blocks = 32;  // cap at 32 for small kernel
      }
      int task_map_blocks = need_tm ? (num_expert + tail_blocks) : 0;
      dim3 block(kThreadPerBlock);
      dim3 grid(num_topk_blocks + task_map_blocks);
      auto cfg = make_pdl_config(grid, block, 0, stream, attr);
      auto kernel = kernels::build_indices_kernel<kUsePDL>;
      cudaLaunchKernelEx(&cfg, kernel, topk_ids_ptr, row_indices_ptr, topk_pos_ptr, seqlens_ptr,
                         (const int *)cu_seqlens_ptr, (const int *)cu_tiles_ptr,
                         (const int *)tiles_ptr, gateup_task_map, down_task_map, gate_up_num_tile_n,
                         down_num_tile_n, gateup_task_map_len, down_task_map_len, total_num_topk,
                         num_topk, num_expert, num_topk_blocks, start_expert, end_expert);
    }
  }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
void count_and_build_indices_async(const void *topk_ids_ptr, void *row_indices_ptr,
                                   void *topk_pos_ptr, void *seqlens_ptr, void *cu_seqlens_ptr,
                                   void *tiles_ptr, void *cu_tiles_ptr, void *gateup_task_map_ptr,
                                   void *down_task_map_ptr, int gate_up_num_tile_n,
                                   int down_num_tile_n, int gateup_task_map_len,
                                   int down_task_map_len, int num_seq, int num_topk, int num_expert,
                                   int eprank, int num_seq_per_group_avg, cudaStream_t stream) {
  constexpr bool kUsePDL = true;

  int start_expert = eprank * num_expert;
  int end_expert = (eprank + 1) * num_expert;

  int4 *gate_up_tm = reinterpret_cast<int4 *>(gateup_task_map_ptr);
  int4 *down_tm = reinterpret_cast<int4 *>(down_task_map_ptr);

  // Variable kTileM dispatch based on average tokens per expert
  if (num_seq_per_group_avg <= 8) {
    constexpr int kTileM = 8;
    launch_count_and_build<kTileM, kUsePDL>(
        (const int *)topk_ids_ptr, (int *)row_indices_ptr, (int *)topk_pos_ptr, (int *)seqlens_ptr,
        (int *)cu_seqlens_ptr, (int *)tiles_ptr, (int *)cu_tiles_ptr, gate_up_tm, down_tm,
        gate_up_num_tile_n, down_num_tile_n, gateup_task_map_len, down_task_map_len, num_seq,
        num_topk, num_expert, start_expert, end_expert, stream);
  } else if (num_seq_per_group_avg <= 16) {
    constexpr int kTileM = 16;
    launch_count_and_build<kTileM, kUsePDL>(
        (const int *)topk_ids_ptr, (int *)row_indices_ptr, (int *)topk_pos_ptr, (int *)seqlens_ptr,
        (int *)cu_seqlens_ptr, (int *)tiles_ptr, (int *)cu_tiles_ptr, gate_up_tm, down_tm,
        gate_up_num_tile_n, down_num_tile_n, gateup_task_map_len, down_task_map_len, num_seq,
        num_topk, num_expert, start_expert, end_expert, stream);
  } else if (num_seq_per_group_avg <= 32) {
    constexpr int kTileM = 32;
    launch_count_and_build<kTileM, kUsePDL>(
        (const int *)topk_ids_ptr, (int *)row_indices_ptr, (int *)topk_pos_ptr, (int *)seqlens_ptr,
        (int *)cu_seqlens_ptr, (int *)tiles_ptr, (int *)cu_tiles_ptr, gate_up_tm, down_tm,
        gate_up_num_tile_n, down_num_tile_n, gateup_task_map_len, down_task_map_len, num_seq,
        num_topk, num_expert, start_expert, end_expert, stream);
  } else if (num_seq_per_group_avg <= 48) {
    constexpr int kTileM = 48;
    launch_count_and_build<kTileM, kUsePDL>(
        (const int *)topk_ids_ptr, (int *)row_indices_ptr, (int *)topk_pos_ptr, (int *)seqlens_ptr,
        (int *)cu_seqlens_ptr, (int *)tiles_ptr, (int *)cu_tiles_ptr, gate_up_tm, down_tm,
        gate_up_num_tile_n, down_num_tile_n, gateup_task_map_len, down_task_map_len, num_seq,
        num_topk, num_expert, start_expert, end_expert, stream);
  } else {
    constexpr int kTileM = 64;
    launch_count_and_build<kTileM, kUsePDL>(
        (const int *)topk_ids_ptr, (int *)row_indices_ptr, (int *)topk_pos_ptr, (int *)seqlens_ptr,
        (int *)cu_seqlens_ptr, (int *)tiles_ptr, (int *)cu_tiles_ptr, gate_up_tm, down_tm,
        gate_up_num_tile_n, down_num_tile_n, gateup_task_map_len, down_task_map_len, num_seq,
        num_topk, num_expert, start_expert, end_expert, stream);
  }
}

}  // namespace fuse_moe_cp_async
}  // namespace hpc
