// Copyright 2025 hpc-ops authors

#ifndef SRC_ATTENTION_MLA_SMALLM_MLA_DIM576_PERSISTENT_GET_SCHEDULER_MAP_CUH_
#define SRC_ATTENTION_MLA_SMALLM_MLA_DIM576_PERSISTENT_GET_SCHEDULER_MAP_CUH_

#include <cuda.h>
#include <stdint.h>

#include <algorithm>
#include <cub/cub.cuh>
#include <limits>

namespace hpc {
namespace attention {
namespace kernels {

// scheduler-map regions:
//   task_list[max_num_jobs,  4]   = (ibatch, isplit_in_batch,
//                                    ikv_tile_start, ikv_tile_end)
//   cu_tasks [num_sm + 1]
//   cu_splits[num_batch + 1]

constexpr int kDim576PersistentSchedulerMapThreads = 256;
constexpr int kDim576PersistentSchedulerMapItemsPerThread = 4;
constexpr int kDim576PersistentMaxBatch =
    kDim576PersistentSchedulerMapThreads * kDim576PersistentSchedulerMapItemsPerThread;

__device__ __forceinline__ int dim576_persistent_sm_of_pos(int t, int cap_lo, int cap_hi,
                                                           int n_hi) {
  int hi_boundary = n_hi * cap_hi;
  if (t < hi_boundary) {
    return t / cap_hi;
  }
  return n_hi + (t - hi_boundary) / cap_lo;
}

template <int kTileN>
__global__ void __launch_bounds__(kDim576PersistentSchedulerMapThreads)
    get_scheduler_map_kernel(int4* task_list, int* cu_tasks, int* cu_splits,
                             const int* num_seq_kv_ptr, int num_batch, int num_sm) {
  constexpr int kThreads = kDim576PersistentSchedulerMapThreads;
  constexpr int kI = kDim576PersistentSchedulerMapItemsPerThread;

  int tid = threadIdx.x;

  using BlockScanI32 = cub::BlockScan<int, kThreads, cub::BLOCK_SCAN_RAKING>;
  __shared__ typename BlockScanI32::TempStorage temp_i32;
  __shared__ int sh_first_sm[kDim576PersistentMaxBatch];
  __shared__ int sh_splits[kDim576PersistentMaxBatch];

  // scan T_i (number of KV tiles per batch).
  int my_T_arr[kI];
#pragma unroll
  for (int i = 0; i < kI; ++i) {
    int ibatch = tid * kI + i;
    int t = 0;
    if (ibatch < num_batch) {
      int n = num_seq_kv_ptr[ibatch];
      t = (n + kTileN - 1) / kTileN;
      if (t < 0) {
        t = 0;
      }
    }
    my_T_arr[i] = t;
  }

  int excl_T_arr[kI];
  int T_total;
  BlockScanI32(temp_i32).ExclusiveSum(my_T_arr, excl_T_arr, T_total);

  // Empty grid (all batches have num_seq_kv == 0).
  if (T_total <= 0) {
#pragma unroll 1
    for (int i = tid; i <= num_batch; i += kThreads) {
      cu_splits[i] = 0;
    }
#pragma unroll 1
    for (int i = tid; i <= num_sm; i += kThreads) {
      cu_tasks[i] = 0;
    }
    // No tasks to emit: zero the entire task_list region
    int max_num_jobs = num_batch + num_sm;
#pragma unroll 1
    for (int t = tid; t < max_num_jobs; t += kThreads) {
      task_list[t] = make_int4(0, 0, 0, 0);
    }
    return;
  }

  int cap_lo = T_total / num_sm;
  int cap_hi = cap_lo + 1;
  int n_hi = T_total - cap_lo * num_sm;

  // per-batch (splits_i, first_sm_i).
  int splits_arr[kI];
  int first_sm_arr[kI];
#pragma unroll
  for (int i = 0; i < kI; ++i) {
    int ibatch = tid * kI + i;
    int my_T = my_T_arr[i];
    int splits_i = 0;
    int first_sm = 0;
    if (ibatch < num_batch && my_T > 0) {
      int b_lo = excl_T_arr[i];
      int b_hi = b_lo + my_T;
      first_sm = dim576_persistent_sm_of_pos(b_lo, cap_lo, cap_hi, n_hi);
      int last_sm = dim576_persistent_sm_of_pos(b_hi - 1, cap_lo, cap_hi, n_hi);
      splits_i = last_sm - first_sm + 1;
    }
    splits_arr[i] = splits_i;
    first_sm_arr[i] = first_sm;
    if (ibatch < num_batch) {
      sh_first_sm[ibatch] = first_sm;
      sh_splits[ibatch] = splits_i;
    }
  }

  // build cu_splits
  __syncthreads();

  int excl_splits_arr[kI];
  int total_splits;
  BlockScanI32(temp_i32).ExclusiveSum(splits_arr, excl_splits_arr, total_splits);

  if (tid == 0) {
    cu_splits[num_batch] = total_splits;
  }
#pragma unroll
  for (int i = 0; i < kI; ++i) {
    int ibatch = tid * kI + i;
    if (ibatch < num_batch) {
      cu_splits[ibatch] = excl_splits_arr[i];
    }
  }

  // build cu_tasks
#pragma unroll 1
  for (int sm = tid; sm < num_sm; sm += kThreads) {
    int cu = 0;
#pragma unroll 1
    for (int ib = 0; ib < num_batch; ++ib) {
      int fs = sh_first_sm[ib];
      if (fs >= sm) {
        break;
      }

      int sp = sh_splits[ib];
      int contrib = sm - fs;
      if (contrib > sp) {
        contrib = sp;
      }
      cu += contrib;
    }
    cu_tasks[sm] = cu;
  }
  if (tid == 0) {
    cu_tasks[num_sm] = total_splits;
  }

  // emit task_list entries.
#pragma unroll
  for (int i = 0; i < kI; ++i) {
    int ibatch = tid * kI + i;
    if (ibatch >= num_batch) {
      continue;
    }
    int my_T = my_T_arr[i];
    if (my_T <= 0) {
      continue;
    }

    int b_lo_index = excl_T_arr[i];
    int b_hi_index = b_lo_index + my_T;
    int first_sm = first_sm_arr[i];
    int splits_i = splits_arr[i];
    int isplit_globa_base = excl_splits_arr[i];

    for (int j = 0; j < splits_i; ++j) {
      int sm = first_sm + j;
      int start_index_in_sm = (sm < n_hi) ? sm * cap_hi : n_hi * cap_hi + (sm - n_hi) * cap_lo;
      int end_index_in_sm = start_index_in_sm + ((sm < n_hi) ? cap_hi : cap_lo);
      int s_lo = max(b_lo_index, start_index_in_sm);
      int s_hi = min(b_hi_index, end_index_in_sm);
      int kv_tile_start = s_lo - b_lo_index;
      int kv_tile_end = s_hi - b_lo_index;
      task_list[isplit_globa_base + j] = make_int4(ibatch, j, kv_tile_start, kv_tile_end);
    }
  }

  // zero the unused task_list tail.
  int max_num_jobs = num_batch + num_sm;
#pragma unroll 1
  for (int t = total_splits + tid; t < max_num_jobs; t += kThreads) {
    task_list[t] = make_int4(0, 0, 0, 0);
  }
}

}  // namespace kernels
}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_MLA_SMALLM_MLA_DIM576_PERSISTENT_GET_SCHEDULER_MAP_CUH_
