// Copyright 2025 hpc-ops authors

#ifndef SRC_ATTENTION_MLA_SMALLM_MLA_DIM576_PERSISTENT_H_
#define SRC_ATTENTION_MLA_SMALLM_MLA_DIM576_PERSISTENT_H_

#include <cuda_runtime_api.h>
#include <stddef.h>

namespace hpc {
namespace attention {
namespace mla {

// Upper bound on physical SM count assumed by the persistent dim576 path.
// The actual SM count is queried at launch time via `hpc::get_sm_count()`;
// this constant only sizes the combine kernel's compile-time `kMaxSplits`
// template parameter (per-lane register array).
//
// Cover sm90 (H20=78, H100=132, H200=132) and sm100 (B200=148). Bumping
// this only costs a couple of fp32 registers per combine-kernel lane.
constexpr int kDim576PersistentMaxNumSm = 148;

// Latent V dimension for dim576 MLA
constexpr int kDim576VDim = 512;

// Self-describing task width (int32 slots per task_list entry), shared by the
// dense and sparse persistent paths:
//   (itoken, ibatch, ikv_tile_start, ikv_tile_end, kv_len, isplit_in_token, _, _)
constexpr int kDim576IntsPerTask = 8;

// max_num_jobs cap: B + num_sm worst case (each SM may straddle one extra
// batch boundary).
inline int dim576_persistent_max_num_jobs(int total_seq_q, int num_sm) {
  return total_seq_q + num_sm;
}

// Task tensor layout (logical batch == query token):
//   [ task_list (max_num_jobs * kDim576IntsPerTask) | cu_tasks (num_sm+1) | cu_splits (L+1) ]
inline size_t dim576_persistent_task_tensor_elems(int total_seq_q, int num_sm) {
  size_t list_elems =
      static_cast<size_t>(dim576_persistent_max_num_jobs(total_seq_q, num_sm)) * kDim576IntsPerTask;
  size_t cu_tasks_elems = static_cast<size_t>(num_sm) + 1;
  size_t cu_splits_elems = static_cast<size_t>(total_seq_q) + 1;
  return list_elems + cu_tasks_elems + cu_splits_elems;
}

inline size_t dim576_persistent_task_list_offset() { return 0; }
inline size_t dim576_persistent_cu_tasks_offset(int total_seq_q, int num_sm) {
  return static_cast<size_t>(dim576_persistent_max_num_jobs(total_seq_q, num_sm)) *
         kDim576IntsPerTask;
}
inline size_t dim576_persistent_cu_splits_offset(int total_seq_q, int num_sm) {
  return dim576_persistent_cu_tasks_offset(total_seq_q, num_sm) + static_cast<size_t>(num_sm) + 1;
}

inline size_t dim576_persistent_y_partial_elems(int total_seq_q, int num_head_q, int v_dim,
                                                int num_sm) {
  size_t max_splits = static_cast<size_t>(num_sm) + total_seq_q;
  return max_splits * static_cast<size_t>(num_head_q) * static_cast<size_t>(v_dim);
}
inline size_t dim576_persistent_lse_elems(int total_seq_q, int num_head_q, int num_sm) {
  size_t max_splits = static_cast<size_t>(num_sm) + total_seq_q;
  return max_splits * static_cast<size_t>(num_head_q);
}

bool smallm_mla_dim576_persistent_async(void* y_ptr, const void* q_ptr, const void* kvcache_ptr,
                                        float* y_partial_ptr, float* lse_ptr, int* task_tensor_ptr,
                                        const int* block_ids_ptr, const int* cu_seqlens_q_ptr,
                                        const int* num_seq_kv_ptr, const float* sink_weight_ptr,
                                        int num_batch, int total_seq_q, int num_head_q, int qk_dim,
                                        int v_dim, int num_kvcache_blocks, int num_seq_max_blocks,
                                        int ldY, int ldQ, int ldKV, float softmax_scale,
                                        cudaStream_t stream, bool task_tensor_prebuilt = false,
                                        bool splitk = true);

bool dim576_persistent_get_scheduler_map_async(int* task_list, int* cu_tasks, int* cu_splits,
                                               const int* cu_seqlens_q_ptr,
                                               const int* num_seq_kv_ptr, int num_batch,
                                               int total_seq_q, int num_sm, cudaStream_t stream,
                                               bool splitk = true);

}  // namespace mla
}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_MLA_SMALLM_MLA_DIM576_PERSISTENT_H_
