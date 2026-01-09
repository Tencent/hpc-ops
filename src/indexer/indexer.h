// Copyright 2025 hpc-ops authors

#ifndef SRC_INDEXER_INDEXER_H_
#define SRC_INDEXER_INDEXER_H_

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace hpc {
namespace indexer {

void mqa_indexer_logits_bf16_async(void* y_ptr, const void* q_ptr, const void* kvcache_ptr,
                                   const void* w_ptr, const void* cu_seqlens_q_ptr,
                                   const void* seqlens_kv_ptr, const void* block_ids_ptr,
                                   const int& num_batch, const int& total_seq_q,
                                   const int& num_head_q, const int& head_dim,
                                   const int& num_kvcache_blocks, const int& block_size,
                                   const int& num_seq_max_blocks, const int& max_context_len,
                                   const int& ratio, const int& num_split, cudaStream_t stream);

void mqa_indexer_logits_fp8_async(void* y_ptr, const void* q_ptr, const void* kvcache_ptr,
                                  const void* w_ptr, const void* cu_seqlens_q_ptr,
                                  const void* seqlens_kv_ptr, const void* block_ids_ptr,
                                  const int& num_batch, const int& total_seq_q,
                                  const int& num_head_q, const int& head_dim,
                                  const int& num_kvcache_blocks, const int& block_size,
                                  const int& num_seq_max_blocks, const int& max_context_len,
                                  const int& ratio, const int& num_split, cudaStream_t stream);

}  // namespace indexer
}  // namespace hpc

#endif  // SRC_INDEXER_INDEXER_H_
