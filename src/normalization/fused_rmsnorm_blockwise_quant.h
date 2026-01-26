// Copyright 2025 hpc-ops authors

#ifndef SRC_NORMALIZATION_FUSED_RMSNORM_BLOCKWISE_QUANT_H_
#define SRC_NORMALIZATION_FUSED_RMSNORM_BLOCKWISE_QUANT_H_

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

#include <tuple>

namespace hpc {
namespace normalization {
void fused_rmsnorm_blockwise_quant_async(void* y_fp8_ptr, void* y_bf16_ptr, void* y_scale_ptr,
                                         const void* input_ptr, const void* weight_ptr, const int m,
                                         const int hidden_size, const float eps,
                                         bool with_blockwise_quant, cudaStream_t stream);
}  // namespace normalization
}  // namespace hpc

#endif  // SRC_NORMALIZATION_FUSED_RMSNORM_BLOCKWISE_QUANT_H_
