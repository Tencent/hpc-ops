// Copyright 2025 hpc-ops authors

#ifndef SRC_NORMALIZATION_FUSED_RMS_NORM_WITH_SCALE_FUSED_RMS_NORM_WITH_SCALE_H_
#define SRC_NORMALIZATION_FUSED_RMS_NORM_WITH_SCALE_FUSED_RMS_NORM_WITH_SCALE_H_

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
namespace fused_rms_norm_with_scale {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> entry(torch::Tensor& input,
                                                              torch::Tensor& weight,
                                                              torch::Tensor& scale, double eps,
                                                              bool output_high_precise);

template <typename Tin, typename Tout>
void fused_rms_norm_with_scale_async(void* input_ptr, void* weight_ptr, void* output_ptr,
                                     void* output_fp32_ptr, void* output_fp8_scale2_ptr,
                                     void* scale, float eps, int batch_size, int hidden_state,
                                     bool is_moe, cudaStream_t stream);

}  // namespace fused_rms_norm_with_scale
}  // namespace normalization
}  // namespace hpc

#endif  // SRC_NORMALIZATION_FUSED_RMS_NORM_WITH_SCALE_FUSED_RMS_NORM_WITH_SCALE_H_
