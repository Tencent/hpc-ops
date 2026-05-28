// Copyright 2025 hpc-ops authors

#ifndef SRC_ATTENTION_DECODE_DECODE_H_
#define SRC_ATTENTION_DECODE_DECODE_H_

#include <cuda_runtime_api.h>
#include <stdint.h>

#include <utility>
#include <vector>

#include "src/attention/decode/sched_task_info.h"

namespace hpc {
namespace attention {
namespace decode {
bool attention_decode_bf16_async(
    void *y_ptr, void *lse_ptr, void *split_out_ptr, const void *q_ptr, void *kcache_ptr,
    void *vcache_ptr, const int *block_ids_ptr, const int *num_seq_kvcache_ptr, int *split_flag_ptr,
    bool new_kv_included, int splitk, int num_batch, int num_seq_q, int num_head_q, int num_head_k,
    int num_head_v, int num_dim_qk, int num_dim_v, int num_kvcache_blocks, int block_size,
    int num_seq_max_blocks, int ldY, int ldQ, int64_t kcache_block_stride,
    int64_t kcache_token_stride, int64_t kcache_head_stride, int64_t vcache_block_stride,
    int64_t vcache_token_stride, int64_t vcache_head_stride, cudaStream_t stream);

bool attention_decode_fp8_async(
    void *y_ptr, void *lse_ptr, void *split_out_ptr, const int *task_map_ptr, const void *q_ptr,
    void *kcache_ptr, void *vcache_ptr, const int *block_ids_ptr, const int *num_seq_kvcache_ptr,
    const float *qscale_ptr, const float *kscale_ptr, const float *vscale_ptr, int *split_flag_ptr,
    bool new_kv_included, int splitk, int splitk_min_len, int consumers, int quant_type,
    int num_batch, int num_seq_q, int num_head_q, int num_head_k, int num_head_v, int num_dim_qk,
    int num_dim_v, int num_kvcache_blocks, int block_size, int num_seq_max_blocks,
    int qscale_pad_stride, int ldY, int ldQ, int64_t kcache_block_stride,
    int64_t kcache_token_stride, int64_t kcache_head_stride, int64_t vcache_block_stride,
    int64_t vcache_token_stride, int64_t vcache_head_stride, cudaStream_t stream);

std::pair<std::vector<decode::dynamic::TaskScheduleInfo>, std::vector<int>>
assign_attention_decode_task_sync(const int *num_seq_kvcache, int num_total_ctas, int num_batch,
                                  int num_head_kv, int num_seq_q, int tilen, bool new_kv_included,
                                  int min_process_len);

bool assign_attention_decode_task_async(int *task_map_ptr, const int *num_seq_kvcache,
                                        int num_total_ctas, int num_batch, int num_head_kv,
                                        int num_seq_q, int tilen, bool new_kv_included,
                                        int min_process_len, cudaStream_t stream);

}  // namespace decode
}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_DECODE_DECODE_H_
