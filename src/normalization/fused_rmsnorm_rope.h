// Copyright 2025 hpc-ops authors

#ifndef SRC_NORMALIZATION_FUSED_RMSNORM_ROPE_H_
#define SRC_NORMALIZATION_FUSED_RMSNORM_ROPE_H_

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

#include <tuple>

namespace hpc {
namespace normalization {
void fused_rmsnorm_rope_async(void* y_q_ptr, void* y_k_ptr, const void* q_ptr,
                              const void* q_weight_ptr, const void* k_ptr, const void* k_weight_ptr,
                              const void* pos_ptr, const void* sin_cos_ptr, const int num_tokens,
                              const int dim, const int rope_dim, const int num_q_heads,
                              const int num_k_heads, const float eps, const int dtype,
                              cudaStream_t stream);
}  // namespace normalization
}  // namespace hpc

#endif  // SRC_NORMALIZATION_FUSED_RMSNORM_ROPE_H_
