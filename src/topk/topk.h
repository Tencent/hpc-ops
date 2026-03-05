// Copyright 2025 hpc-ops authors

#ifndef SRC_TOPK_TOPK_H_
#define SRC_TOPK_TOPK_H_

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace hpc {
namespace topk {

bool topk_per_row_async(int *topk_indices, const float *logits, const int *seqlens, int topk,
                        int num_sp_tokens, int num_rows, int row_stride, cudaStream_t stream);

bool topk_per_row_varlen_async(int *topk_indices, const float *logits_ptr,
                               const int *cu_seqlens_q_ptr, const int *seqlens_kv_ptr, int topk,
                               int compress_ratio, int num_batch, int num_rows, int row_stride,
                               bool deterministic, cudaStream_t stream);

bool grouped_topk_async(float *topk_weights_ptr, int *topk_ids_ptr, const float *scores_ptr,
                        const float *bias_ptr, float scale, int num_tokens, int topk,
                        int topk_group, int num_experts, int num_expert_group, bool renormalize,
                        cudaStream_t stream);

}  // namespace topk
}  // namespace hpc

#endif  // SRC_TOPK_TOPK_H_
