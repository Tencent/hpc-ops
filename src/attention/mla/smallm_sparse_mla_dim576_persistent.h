// Copyright 2025 hpc-ops authors

#ifndef SRC_ATTENTION_MLA_SMALLM_SPARSE_MLA_DIM576_PERSISTENT_H_
#define SRC_ATTENTION_MLA_SMALLM_SPARSE_MLA_DIM576_PERSISTENT_H_

#include <cuda_runtime_api.h>
#include <stddef.h>

#include "src/attention/mla/smallm_mla_dim576_persistent.h"  // task tensor / lse / partial sizing

namespace hpc {
namespace attention {
namespace mla {

constexpr int kSparseDim576MaxNumTopk = 2048;

bool smallm_sparse_mla_dim576_persistent_async(
    void* y_ptr, const void* q_ptr, const void* kvcache_ptr, float* y_partial_ptr, float* lse_ptr,
    int* task_tensor_ptr, const int* block_ids_ptr, const int* topk_ids_ptr,
    const int* cu_seqlens_q_ptr, const float* sink_weight_ptr, int num_batch, int total_seq_q,
    int num_head_q, int qk_dim, int v_dim, int num_kvcache_blocks, int num_seq_max_blocks,
    int num_max_topk, int block_size, int ldY, int ldQ, int ldKV, float softmax_scale,
    cudaStream_t stream, bool task_tensor_prebuilt = false, bool splitk = true,
    bool prefill = false);

bool dim576_persistent_get_scheduler_map_sparse_async(int* task_list, int* cu_tasks, int* cu_splits,
                                                      const int* cu_seqlens_q_ptr, int num_batch,
                                                      int total_seq_q, int num_max_topk, int num_sm,
                                                      cudaStream_t stream, bool splitk = true);

}  // namespace mla
}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_MLA_SMALLM_SPARSE_MLA_DIM576_PERSISTENT_H_
