// Copyright 2025 hpc-ops authors

#ifndef SRC_ATTENTION_MLA_MLA_H_
#define SRC_ATTENTION_MLA_MLA_H_

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace hpc {
namespace attention {

bool attention_mla_with_kvcache_bf16_async(void* y_ptr, const void* q_ptr, const void* kvcache_ptr,
                                           const int* block_ids_ptr, const int* cu_seqlens_q_ptr,
                                           const int* num_seq_kv_ptr, int num_batch,
                                           int total_seq_q, int num_head_q, int head_dim,
                                           int num_kvcache_blocks, int num_seq_max_blocks, int ldY,
                                           int ldQ, int ldKV, cudaStream_t stream);

bool attention_sparse_mla_with_kvcache_bf16_async(
    void* y_ptr, const void* q_ptr, void* win_kvcache_ptr, const int* win_block_ids_ptr,
    const int* win_topk_ids_ptr, void* compress_kvcache_ptr, const int* compress_block_ids_ptr,
    const int* compress_topk_ids_ptr, const int* cu_seqlens_q_ptr, const void* sink_weight_ptr,
    float softmax_scale, int num_batch, int total_seq_q, int num_head_q, int head_dim,
    int num_win_kvcache_blocks, int num_compress_kvcache_blocks, int num_win_seq_max_blocks,
    int num_compress_seq_max_blocks, int block_size, int num_win_max_topk,
    int num_compress_max_topk, int ldY, int ldQ, int ldWinKV, int ldCompressKV,
    cudaStream_t stream);

}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_MLA_MLA_H_
