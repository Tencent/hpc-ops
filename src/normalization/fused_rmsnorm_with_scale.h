// Copyright 2025 hpc-ops authors

#ifndef SRC_NORMALIZATION_FUSED_RMSNORM_WITH_SCALE_H_
#define SRC_NORMALIZATION_FUSED_RMSNORM_WITH_SCALE_H_

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include <tuple>

namespace hpc {
namespace normalization {

bool fused_rmsnorm_with_scale_async(const void* input_ptr, const void* weight_ptr, void* output_ptr,
                                    void* output_fp32_ptr, void* output_fp8_scale2_ptr,
                                    const void* scale, float eps, int batch_size, int hidden_state,
                                    bool is_moe, cudaStream_t stream);

}  // namespace normalization
}  // namespace hpc

#endif  // SRC_NORMALIZATION_FUSED_RMSNORM_WITH_SCALE_H_
