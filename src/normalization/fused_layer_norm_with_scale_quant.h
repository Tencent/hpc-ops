// Copyright 2025 hpc-ops authors

#ifndef SRC_NORMALIZATION_FUSED_LAYER_NORM_WITH_SCALE_QUANT_H_
#define SRC_NORMALIZATION_FUSED_LAYER_NORM_WITH_SCALE_QUANT_H_

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

#include <tuple>

namespace hpc {
namespace normalization {

bool fused_layer_norm_with_scale_quant_async(
    const void* input_ptr, const void* weight_ptr, const void* bias_ptr, void* output_ptr,
    const void* pre_norm_scale1_ptr, const void* pre_norm_scale2_ptr,
    const void* post_norm_scale_ptr, const void* post_norm_bias_scale_ptr, void* quant_scale,
    void* output_x_ptr, float eps, float quant_eps, int batch_size, int hidden_state,
    int group_size, float fp8_e4m3_max, float fp8_e4m3_min, bool is_elementwise_affine,
    bool use_pre_norm_scale, bool use_post_norm_scale, cudaStream_t stream);

}  // namespace normalization
}  // namespace hpc

#endif  // SRC_NORMALIZATION_FUSED_LAYER_NORM_WITH_SCALE_QUANT_H_
