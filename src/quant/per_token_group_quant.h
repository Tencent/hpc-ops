// Copyright 2025 hpc-ops authors

#ifndef SRC_QUANT_PER_TOKEN_GROUP_QUANT_H_
#define SRC_QUANT_PER_TOKEN_GROUP_QUANT_H_

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

#include <tuple>

namespace hpc {
namespace quant {

bool per_token_group_quant_async(const void* input_ptr, void* output_ptr, void* quant_scale,
                                 int group_size, float quant_eps, int hidden_states, int batch_size,
                                 float fp8_e4m3_max, float fp8_e4m3_min, cudaStream_t stream);

}  // namespace quant
}  // namespace hpc

#endif  // SRC_QUANT_PER_TOKEN_GROUP_QUANT_H_
