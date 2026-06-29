// Copyright 2025 hpc-ops authors

#ifndef SRC_NORMALIZATION_FUSED_QK_RMSNORM_MROPE_H_
#define SRC_NORMALIZATION_FUSED_QK_RMSNORM_MROPE_H_

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace hpc {
namespace normalization {

void fused_qk_rmsnorm_mrope_async(
    __nv_bfloat16* out_q_ptr, __nv_bfloat16* out_k_ptr, __nv_bfloat16* out_v_ptr,
    const __nv_bfloat16* und_qkv_ptr, const __nv_bfloat16* gen_qkv_ptr,
    const __nv_bfloat16* und_q_weight_ptr, const __nv_bfloat16* und_k_weight_ptr,
    const __nv_bfloat16* gen_q_weight_ptr, const __nv_bfloat16* gen_k_weight_ptr,
    const int64_t* positions_ptr, const float* cos_sin_cache_ptr, const int64_t* cat_indices_ptr,
    int und_len, int total_tokens, int num_q_heads, int num_k_heads, int num_v_heads, int head_dim,
    int mrope_section_h, int mrope_section_w, float eps, int cos_sin_cache_stride,
    cudaStream_t stream);

}  // namespace normalization
}  // namespace hpc

#endif  // SRC_NORMALIZATION_FUSED_QK_RMSNORM_MROPE_H_
