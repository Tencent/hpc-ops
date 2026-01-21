// Copyright 2025 hpc-ops authors

#ifndef SRC_COMPRESSOR_COMPRESSOR_H_
#define SRC_COMPRESSOR_COMPRESSOR_H_

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>
#include <stdint.h>

#include <vector>

namespace hpc {
namespace compressor {

bool kv_compressor_fp32_async(float* compressed_kv_ptr, const float* kv_ptr, const float* score_ptr,
                              const int* cu_seqlens_ptr, const int* cu_compressed_seqlens_ptr,
                              float* kv_states_ptr, float* score_states_ptr,
                              const int* state_index_ptr, const int* start_pos_ptr,
                              const float* ape_ptr, int num_batch, int total_seqlen, int ratio,
                              bool overlap, int head_dim, bool is_prefill, cudaStream_t stream);
void kv_compressor_decode_async(void* y_ptr, const void* kv_ptr, const void* score_ptr,
                                const void* ape_ptr, void* kv_states_ptr, void* score_states_ptr,
                                const void* state_idx_ptr, const void* start_pos_ptr,
                                const void* cu_compress_seqlens_ptr, int batch_size, int head_dim,
                                int ratio, int mtp, cudaStream_t stream);
}  // namespace compressor
}  // namespace hpc

#endif  // SRC_COMPRESSOR_COMPRESSOR_H_
