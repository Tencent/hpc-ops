// Copyright 2025 hpc-ops authors

#ifndef SRC_KV_COMPRESSOR_DECODE_KV_COMPRESSOR_DECODE_H_
#define SRC_KV_COMPRESSOR_DECODE_KV_COMPRESSOR_DECODE_H_

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace hpc {
namespace kv_compressor_decode {

void kv_compressor_decode_async(void *y_ptr, const void *kv_ptr, const void *score_ptr,
                                const void *ape_ptr, void *kv_states_ptr, void *score_states_ptr,
                                const void *state_idx_ptr, const void *start_pos_ptr,
                                const void *cu_compress_seqlens_ptr, int batch_size, int head_dim,
                                int ratio, int mtp, cudaStream_t stream);

}  // namespace kv_compressor_decode
}  // namespace hpc

#endif  // SRC_KV_COMPRESSOR_DECODE_KV_COMPRESSOR_DECODE_H_
