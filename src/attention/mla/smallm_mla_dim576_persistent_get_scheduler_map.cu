// Copyright 2025 hpc-ops authors

#include <cuda.h>

#include "src/attention/mla/smallm_mla_dim576_persistent.h"
#include "src/attention/mla/smallm_mla_dim576_persistent_get_scheduler_map.cuh"

namespace hpc {
namespace attention {
namespace mla {

bool dim576_persistent_get_scheduler_map_async(int* task_list, int* cu_tasks, int* cu_splits,
                                               const int* cu_seqlens_q_ptr,
                                               const int* num_seq_kv_ptr, int num_batch,
                                               int total_seq_q, int num_sm, cudaStream_t stream,
                                               bool splitk) {
  constexpr int kTileN = 64;
  if (total_seq_q <= 0) {
    return true;
  }
  if (splitk) {
    auto kernel = kernels::get_scheduler_map_kernel<kTileN, /*kSplitK=*/true>;
    kernel<<<1, kernels::kDim576PersistentSchedulerMapThreads, 0, stream>>>(
        task_list, cu_tasks, cu_splits, cu_seqlens_q_ptr, num_seq_kv_ptr, num_batch, total_seq_q,
        num_sm);
  } else {
    auto kernel = kernels::get_scheduler_map_kernel<kTileN, /*kSplitK=*/false>;
    kernel<<<1, kernels::kDim576PersistentSchedulerMapThreads, 0, stream>>>(
        task_list, cu_tasks, cu_splits, cu_seqlens_q_ptr, num_seq_kv_ptr, num_batch, total_seq_q,
        num_sm);
  }
  return true;
}

bool dim576_persistent_get_scheduler_map_sparse_async(int* task_list, int* cu_tasks, int* cu_splits,
                                                      const int* cu_seqlens_q_ptr, int num_batch,
                                                      int total_seq_q, int num_max_topk, int num_sm,
                                                      cudaStream_t stream, bool splitk) {
  constexpr int kTileN = 64;
  if (total_seq_q <= 0) {
    return true;
  }
  if (splitk) {
    auto kernel = kernels::get_scheduler_map_sparse_kernel<kTileN, /*kSplitK=*/true>;
    kernel<<<1, kernels::kDim576PersistentSchedulerMapThreads, 0, stream>>>(
        task_list, cu_tasks, cu_splits, cu_seqlens_q_ptr, num_batch, total_seq_q, num_max_topk,
        num_sm);
  } else {
    auto kernel = kernels::get_scheduler_map_sparse_kernel<kTileN, /*kSplitK=*/false>;
    kernel<<<1, kernels::kDim576PersistentSchedulerMapThreads, 0, stream>>>(
        task_list, cu_tasks, cu_splits, cu_seqlens_q_ptr, num_batch, total_seq_q, num_max_topk,
        num_sm);
  }
  return true;
}

}  // namespace mla
}  // namespace attention
}  // namespace hpc
