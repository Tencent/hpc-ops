// Copyright 2025 hpc-ops authors

#ifndef SRC_ATTENTION_MLA_SMALLM_MLA_DIM576_PERSISTENT_LAUNCH_H_
#define SRC_ATTENTION_MLA_SMALLM_MLA_DIM576_PERSISTENT_LAUNCH_H_

#include <cuda_runtime_api.h>

namespace hpc {
namespace attention {
namespace mla {

template <int kTileM, int kNumMathWG, bool kUseSink>
void run_dim576_persistent(void* y_ptr, const void* q_ptr, const void* kvcache_ptr,
                           float* y_partial_ptr, float* lse_ptr, int* task_tensor_ptr,
                           const int* block_ids_ptr, const int* cu_seqlens_q_ptr,
                           const int* num_seq_kv_ptr, const float* sink_weight_ptr, int num_batch,
                           int total_seq_q, int num_head_q, int qk_dim, int v_dim,
                           int num_kvcache_blocks, int num_seq_max_blocks, int ldY, int ldQ,
                           int ldKV, float softmax_scale, cudaStream_t stream,
                           bool task_tensor_prebuilt, bool splitk);

}  // namespace mla
}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_MLA_SMALLM_MLA_DIM576_PERSISTENT_LAUNCH_H_
