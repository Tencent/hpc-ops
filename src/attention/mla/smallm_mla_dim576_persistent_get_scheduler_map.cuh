// Copyright 2025 hpc-ops authors

#ifndef SRC_ATTENTION_MLA_SMALLM_MLA_DIM576_PERSISTENT_GET_SCHEDULER_MAP_CUH_
#define SRC_ATTENTION_MLA_SMALLM_MLA_DIM576_PERSISTENT_GET_SCHEDULER_MAP_CUH_

#include <cuda.h>
#include <stdint.h>

#include <algorithm>
#include <cub/cub.cuh>
#include <limits>

#include "src/attention/mla/smallm_mla_dim576_persistent.h"  // kDim576PersistentMaxNumSm

namespace hpc {
namespace attention {
namespace kernels {

// scheduler-map regions (self-describing 8-int tasks, logical batch == query token):
//   task_list[max_num_jobs, kDim576IntsPerTask]
//     = (itoken, ibatch, ikv_tile_start, ikv_tile_end, kv_len, isplit_in_token, _, _)
//   cu_tasks [num_sm + 1]
//   cu_splits[total_seq_q + 1]

constexpr int kDim576PersistentSchedulerMapThreads = 256;
constexpr int kDim576PersistentSchedulerMapItemsPerThread = 4;

__device__ __forceinline__ int dim576_persistent_sm_of_pos(int t, int cap_lo, int cap_hi,
                                                           int n_hi) {
  int hi_boundary = n_hi * cap_hi;
  if (t < hi_boundary) {
    return t / cap_hi;
  }
  return n_hi + (t - hi_boundary) / cap_lo;
}

template <int kTileN, bool kSplitK = true>
__global__ void __launch_bounds__(kDim576PersistentSchedulerMapThreads)
    get_scheduler_map_kernel(int* task_list, int* cu_tasks, int* cu_splits,
                             const int* cu_seqlens_q_ptr, const int* num_seq_kv_ptr, int num_batch,
                             int total_seq_q, int num_sm) {
  constexpr int kThreads = kDim576PersistentSchedulerMapThreads;
  constexpr int kI = kDim576PersistentSchedulerMapItemsPerThread;
  constexpr int kChunk = kThreads * kI;
  constexpr int kW = mla::kDim576IntsPerTask;  // 8

  int tid = threadIdx.x;

  using BlockScanI32 = cub::BlockScan<int, kThreads, cub::BLOCK_SCAN_RAKING>;
  using BlockReduceI32 = cub::BlockReduce<int, kThreads>;
  __shared__ union {
    typename BlockScanI32::TempStorage scan;
    typename BlockReduceI32::TempStorage reduce;
  } temp;
  __shared__ int sh_cu_tasks_hist[mla::kDim576PersistentMaxNumSm + 1];

  auto resolve_token = [&](int iquery, int* ibatch_out) -> int {
    int lo = 0, hi = num_batch;
    while (lo < hi) {
      int mid = (lo + hi) >> 1;
      if (cu_seqlens_q_ptr[mid] <= iquery) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    int b = lo - 1;
    *ibatch_out = b;
    int seq_q_start = cu_seqlens_q_ptr[b];
    int len_q = cu_seqlens_q_ptr[b + 1] - seq_q_start;
    int cache_len = num_seq_kv_ptr[b] - len_q;
    int j = iquery - seq_q_start;
    return cache_len + j + 1;  // kv_len
  };

  auto tiles_of = [&](int kv_len) -> int {
    return kv_len > 0 ? (kv_len + kTileN - 1) / kTileN : 0;
  };
  auto T_of = [&](int kv_len) -> int {
    if constexpr (kSplitK) {
      return tiles_of(kv_len);
    } else {
      return kv_len > 0 ? 1 : 0;
    }
  };

  // Zero histogram.
#pragma unroll 1
  for (int s = tid; s <= num_sm; s += kThreads) {
    sh_cu_tasks_hist[s] = 0;
  }

  // ---- Pass A: T_total = sum_tok T(tok).
  int my_T_sum = 0;
#pragma unroll 1
  for (int base = 0; base < total_seq_q; base += kThreads) {
    int iquery = base + tid;
    if (iquery < total_seq_q) {
      int ib;
      my_T_sum += T_of(resolve_token(iquery, &ib));
    }
  }
  int T_total = BlockReduceI32(temp.reduce).Sum(my_T_sum);
  __shared__ int sh_T_total;
  if (tid == 0) {
    sh_T_total = T_total;
  }
  __syncthreads();
  T_total = sh_T_total;

  // Empty grid.
  if (T_total <= 0) {
#pragma unroll 1
    for (int i = tid; i <= total_seq_q; i += kThreads) {
      cu_splits[i] = 0;
    }
#pragma unroll 1
    for (int s = tid; s <= num_sm; s += kThreads) {
      cu_tasks[s] = 0;
    }
    int max_num_jobs = total_seq_q + num_sm;
#pragma unroll 1
    for (int t = tid; t < max_num_jobs; t += kThreads) {
#pragma unroll
      for (int w = 0; w < kW; ++w) {
        task_list[t * kW + w] = 0;
      }
    }
    return;
  }

  int cap_lo = T_total / num_sm;
  int cap_hi = cap_lo + 1;
  int n_hi = T_total - cap_lo * num_sm;

  // ---- Pass B: chunked. running_T carries the global tile prefix; running_splits
  // carries the global split prefix.
  int running_T = 0;
  int running_splits = 0;

  for (int chunk_base = 0; chunk_base < total_seq_q; chunk_base += kChunk) {
    // (a) per-token T and metadata.
    int my_T[kI];
    int my_ibatch[kI];
    int my_kv_len[kI];
#pragma unroll
    for (int i = 0; i < kI; ++i) {
      int iquery = chunk_base + tid * kI + i;
      int Ti = 0, ib = 0, kvl = 0;
      if (iquery < total_seq_q) {
        kvl = resolve_token(iquery, &ib);
        Ti = T_of(kvl);
      }
      my_T[i] = Ti;
      my_ibatch[i] = ib;
      my_kv_len[i] = kvl;
    }

    // (b) exclusive prefix of T within chunk -> global tile base b_lo.
    int excl_T[kI];
    int chunk_T_total;
    BlockScanI32(temp.scan).ExclusiveSum(my_T, excl_T, chunk_T_total);
    __syncthreads();

    // (c) splits_i and first_sm per token from global tile base.
    int my_splits[kI];
    int my_first_sm[kI];
#pragma unroll
    for (int i = 0; i < kI; ++i) {
      int splits_i = 0;
      int first_sm = 0;
      if (my_T[i] > 0) {
        int b_lo = running_T + excl_T[i];
        int b_hi = b_lo + my_T[i];
        first_sm = dim576_persistent_sm_of_pos(b_lo, cap_lo, cap_hi, n_hi);
        int last_sm = dim576_persistent_sm_of_pos(b_hi - 1, cap_lo, cap_hi, n_hi);
        splits_i = last_sm - first_sm + 1;
      }
      my_splits[i] = splits_i;
      my_first_sm[i] = first_sm;
    }

    // (d) exclusive prefix of splits within chunk -> global split base.
    int excl_splits[kI];
    int chunk_splits_total;
    BlockScanI32(temp.scan).ExclusiveSum(my_splits, excl_splits, chunk_splits_total);

    // (e) emit cu_splits, histogram, task_list.
#pragma unroll
    for (int i = 0; i < kI; ++i) {
      int iquery = chunk_base + tid * kI + i;
      if (iquery >= total_seq_q) {
        continue;
      }
      int splits_i = my_splits[i];
      int first_sm = my_first_sm[i];
      int isplit_global_base = running_splits + excl_splits[i];
      cu_splits[iquery] = isplit_global_base;

      int b_lo_index = running_T + excl_T[i];
      int b_hi_index = b_lo_index + my_T[i];
      int ibatch = my_ibatch[i];
      int kv_len = my_kv_len[i];
      int real_tiles = tiles_of(kv_len);

#pragma unroll 1
      for (int js = 0; js < splits_i; ++js) {
        int sm = first_sm + js;
        atomicAdd(&sh_cu_tasks_hist[sm], 1);

        int kv_tile_start;
        int kv_tile_end;
        if constexpr (kSplitK) {
          int start_index_in_sm = (sm < n_hi) ? sm * cap_hi : n_hi * cap_hi + (sm - n_hi) * cap_lo;
          int end_index_in_sm = start_index_in_sm + ((sm < n_hi) ? cap_hi : cap_lo);
          int s_lo = max(b_lo_index, start_index_in_sm);
          int s_hi = min(b_hi_index, end_index_in_sm);
          kv_tile_start = s_lo - b_lo_index;
          kv_tile_end = s_hi - b_lo_index;
        } else {
          kv_tile_start = 0;
          kv_tile_end = real_tiles;
        }
        int pos = (isplit_global_base + js) * kW;
        task_list[pos + 0] = iquery;  // itoken
        task_list[pos + 1] = ibatch;  // phys batch
        task_list[pos + 2] = kv_tile_start;
        task_list[pos + 3] = kv_tile_end;
        task_list[pos + 4] = kv_len;
        task_list[pos + 5] = js;  // isplit_in_token
        task_list[pos + 6] = 0;
        task_list[pos + 7] = 0;
      }
    }

    running_T += chunk_T_total;
    running_splits += chunk_splits_total;
    __syncthreads();  // histogram + temp.scan reuse barrier
  }

  if (tid == 0) {
    cu_splits[total_seq_q] = running_splits;
  }

  // ---- Build cu_tasks from histogram via in-block exclusive scan.
  int my_hist[kI];
#pragma unroll
  for (int i = 0; i < kI; ++i) {
    int s = tid * kI + i;
    my_hist[i] = (s < num_sm) ? sh_cu_tasks_hist[s] : 0;
  }
  int excl_hist[kI];
  int hist_total;
  BlockScanI32(temp.scan).ExclusiveSum(my_hist, excl_hist, hist_total);
#pragma unroll
  for (int i = 0; i < kI; ++i) {
    int s = tid * kI + i;
    if (s < num_sm) {
      cu_tasks[s] = excl_hist[i];
    }
  }
  if (tid == 0) {
    cu_tasks[num_sm] = hist_total;  // == running_splits
  }

  // Zero unused task_list tail.
  int max_num_jobs = total_seq_q + num_sm;
#pragma unroll 1
  for (int t = running_splits + tid; t < max_num_jobs; t += kThreads) {
#pragma unroll
    for (int w = 0; w < kW; ++w) {
      task_list[t * kW + w] = 0;
    }
  }
}

// T is uniform, so geometry is closed-form: token i owns global tiles
// [i*T, (i+1)*T). Chunked BlockScan over splits handles arbitrary total_seq_q.
template <int kTileN, bool kSplitK = true>
__global__ void __launch_bounds__(kDim576PersistentSchedulerMapThreads)
    get_scheduler_map_sparse_kernel(int* task_list, int* cu_tasks, int* cu_splits,
                                    const int* cu_seqlens_q_ptr, int num_batch, int total_seq_q,
                                    int num_max_topk, int num_sm) {
  constexpr int kThreads = kDim576PersistentSchedulerMapThreads;
  constexpr int kI = kDim576PersistentSchedulerMapItemsPerThread;
  constexpr int kChunk = kThreads * kI;
  constexpr int kW = mla::kDim576IntsPerTask;  // 8

  int tid = threadIdx.x;

  using BlockScanI32 = cub::BlockScan<int, kThreads, cub::BLOCK_SCAN_RAKING>;
  __shared__ typename BlockScanI32::TempStorage temp_i32;
  __shared__ int sh_cu_tasks_hist[mla::kDim576PersistentMaxNumSm + 1];

  auto resolve_ibatch = [&](int iquery) -> int {
    int lo = 0, hi = num_batch;
    while (lo < hi) {
      int mid = (lo + hi) >> 1;
      if (cu_seqlens_q_ptr[mid] <= iquery) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    return lo - 1;
  };

  int real_tiles = (num_max_topk + kTileN - 1) / kTileN;
  int T;
  if constexpr (kSplitK) {
    T = real_tiles;
  } else {
    T = (num_max_topk > 0) ? 1 : 0;
  }
  int T_total = total_seq_q * T;

  // Zero histogram.
#pragma unroll 1
  for (int s = tid; s <= num_sm; s += kThreads) {
    sh_cu_tasks_hist[s] = 0;
  }
  __syncthreads();

  // Empty grid.
  if (T_total <= 0) {
#pragma unroll 1
    for (int i = tid; i <= total_seq_q; i += kThreads) {
      cu_splits[i] = 0;
    }
#pragma unroll 1
    for (int s = tid; s <= num_sm; s += kThreads) {
      cu_tasks[s] = 0;
    }
    int max_num_jobs = total_seq_q + num_sm;
#pragma unroll 1
    for (int t = tid; t < max_num_jobs; t += kThreads) {
#pragma unroll
      for (int w = 0; w < kW; ++w) {
        task_list[t * kW + w] = 0;
      }
    }
    return;
  }

  int cap_lo = T_total / num_sm;
  int cap_hi = cap_lo + 1;
  int n_hi = T_total - cap_lo * num_sm;

  int running_splits = 0;
  for (int chunk_base = 0; chunk_base < total_seq_q; chunk_base += kChunk) {
    int my_splits[kI];
    int my_first_sm[kI];
#pragma unroll
    for (int i = 0; i < kI; ++i) {
      int iquery = chunk_base + tid * kI + i;
      int splits_i = 0;
      int first_sm = 0;
      if (iquery < total_seq_q && T > 0) {
        int b_lo = iquery * T;
        int b_hi = b_lo + T;
        first_sm = dim576_persistent_sm_of_pos(b_lo, cap_lo, cap_hi, n_hi);
        int last_sm = dim576_persistent_sm_of_pos(b_hi - 1, cap_lo, cap_hi, n_hi);
        splits_i = last_sm - first_sm + 1;
      }
      my_splits[i] = splits_i;
      my_first_sm[i] = first_sm;
    }

    int excl_splits[kI];
    int chunk_total;
    BlockScanI32(temp_i32).ExclusiveSum(my_splits, excl_splits, chunk_total);

#pragma unroll
    for (int i = 0; i < kI; ++i) {
      int iquery = chunk_base + tid * kI + i;
      if (iquery >= total_seq_q) {
        continue;
      }
      int splits_i = my_splits[i];
      int first_sm = my_first_sm[i];
      int isplit_global_base = running_splits + excl_splits[i];
      cu_splits[iquery] = isplit_global_base;

      int ibatch = resolve_ibatch(iquery);
      int b_lo_index = iquery * T;
      int b_hi_index = b_lo_index + T;

#pragma unroll 1
      for (int js = 0; js < splits_i; ++js) {
        int sm = first_sm + js;
        atomicAdd(&sh_cu_tasks_hist[sm], 1);

        int kv_tile_start;
        int kv_tile_end;
        if constexpr (kSplitK) {
          int start_index_in_sm = (sm < n_hi) ? sm * cap_hi : n_hi * cap_hi + (sm - n_hi) * cap_lo;
          int end_index_in_sm = start_index_in_sm + ((sm < n_hi) ? cap_hi : cap_lo);
          int s_lo = max(b_lo_index, start_index_in_sm);
          int s_hi = min(b_hi_index, end_index_in_sm);
          kv_tile_start = s_lo - b_lo_index;
          kv_tile_end = s_hi - b_lo_index;
        } else {
          kv_tile_start = 0;
          kv_tile_end = real_tiles;
        }
        int pos = (isplit_global_base + js) * kW;
        task_list[pos + 0] = iquery;  // itoken
        task_list[pos + 1] = ibatch;  // phys batch
        task_list[pos + 2] = kv_tile_start;
        task_list[pos + 3] = kv_tile_end;
        task_list[pos + 4] = num_max_topk;  // kv_len slot (unused by sparse kernel)
        task_list[pos + 5] = js;            // isplit_in_token
        task_list[pos + 6] = 0;
        task_list[pos + 7] = 0;
      }
    }

    running_splits += chunk_total;
    __syncthreads();
  }

  if (tid == 0) {
    cu_splits[total_seq_q] = running_splits;
  }

  int my_hist[kI];
#pragma unroll
  for (int i = 0; i < kI; ++i) {
    int s = tid * kI + i;
    my_hist[i] = (s < num_sm) ? sh_cu_tasks_hist[s] : 0;
  }
  int excl_hist[kI];
  int hist_total;
  BlockScanI32(temp_i32).ExclusiveSum(my_hist, excl_hist, hist_total);
#pragma unroll
  for (int i = 0; i < kI; ++i) {
    int s = tid * kI + i;
    if (s < num_sm) {
      cu_tasks[s] = excl_hist[i];
    }
  }
  if (tid == 0) {
    cu_tasks[num_sm] = hist_total;
  }

  int max_num_jobs = total_seq_q + num_sm;
#pragma unroll 1
  for (int t = running_splits + tid; t < max_num_jobs; t += kThreads) {
#pragma unroll
    for (int w = 0; w < kW; ++w) {
      task_list[t * kW + w] = 0;
    }
  }
}

}  // namespace kernels
}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_MLA_SMALLM_MLA_DIM576_PERSISTENT_GET_SCHEDULER_MAP_CUH_
