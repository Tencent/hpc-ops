// Copyright 2025 hpc-ops authors

#ifndef SRC_KV_COMPRESSOR_KV_COMPRESSOR_H_
#define SRC_KV_COMPRESSOR_KV_COMPRESSOR_H_

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>
#include <stdint.h>

#include <vector>

namespace hpc {
namespace kv_compressor {

bool kv_compressor_fp32_async(float* compressed_kv_ptr, const float* kv_ptr, const float* score_ptr,
                              const int* cu_seqlens_ptr, const int* cu_compressed_seqlens_ptr,
                              float* kv_states_ptr, float* score_states_ptr,
                              const int* state_index_ptr, const int* start_pos_ptr,
                              const float* ape_ptr, int num_batch, int total_seqlen, int ratio,
                              bool overlap, int head_dim, bool is_prefill, cudaStream_t stream);
}  // namespace kv_compressor
}  // namespace hpc

#endif  // SRC_KV_COMPRESSOR_KV_COMPRESSOR_H_
