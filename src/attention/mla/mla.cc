// Copyright 2025 hpc-ops authors

#include "src/attention/mla/mla.h"

#include <cuda_runtime_api.h>

#include <algorithm>

#include "src/attention/mla/smallm_mla_dim512.h"
#include "src/attention/mla/smallm_sparse_mla_dim512.h"

namespace hpc {
namespace attention {

bool attention_mla_with_kvcache_bf16_async(void* y_ptr, const void* q_ptr, const void* kvcache_ptr,
                                           const int* block_ids_ptr, const int* cu_seqlens_q_ptr,
                                           const int* num_seq_kv_ptr, int num_batch,
                                           int total_seq_q, int num_head_q, int head_dim,
                                           int num_kvcache_blocks, int num_seq_max_blocks, int ldY,
                                           int ldQ, int ldKV, cudaStream_t stream) {
  if (head_dim == 512) {
    return mla::smallm_mla_dim512_async(y_ptr, q_ptr, kvcache_ptr, block_ids_ptr, cu_seqlens_q_ptr,
                                        num_seq_kv_ptr, num_batch, total_seq_q, num_head_q,
                                        head_dim, num_kvcache_blocks, num_seq_max_blocks, ldY, ldQ,
                                        ldKV, stream);
  }
  return false;
}

bool attention_sparse_mla_with_kvcache_bf16_async(
    void* y_ptr, const void* q_ptr, void* win_kvcache_ptr, const int* win_block_ids_ptr,
    const int* win_topk_ids_ptr, void* compress_kvcache_ptr, const int* compress_block_ids_ptr,
    const int* compress_topk_ids_ptr, const int* cu_seqlens_q_ptr, const void* sink_weight_ptr,
    float softmax_scale, int num_batch, int total_seq_q, int num_head_q, int head_dim,
    int num_win_kvcache_blocks, int num_compress_kvcache_blocks, int num_win_seq_max_blocks,
    int num_compress_seq_max_blocks, int block_size, int num_win_max_topk,
    int num_compress_max_topk, int ldY, int ldQ, int ldWinKV, int ldCompressKV,
    cudaStream_t stream) {
  if (head_dim == 512) {
    return mla::smallm_sparse_mla_dim512_async(
        y_ptr, q_ptr, win_kvcache_ptr, win_block_ids_ptr, win_topk_ids_ptr, compress_kvcache_ptr,
        compress_block_ids_ptr, compress_topk_ids_ptr, cu_seqlens_q_ptr, sink_weight_ptr,
        softmax_scale, num_batch, total_seq_q, num_head_q, head_dim, num_win_kvcache_blocks,
        num_compress_kvcache_blocks, num_win_seq_max_blocks, num_compress_seq_max_blocks,
        block_size, num_win_max_topk, num_compress_max_topk, ldY, ldQ, ldWinKV, ldCompressKV,
        stream);
  }

  return false;
}

}  // namespace attention
}  // namespace hpc
