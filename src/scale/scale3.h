// Copyright 2025 hpc-ops authors

#ifndef SRC_SCALE_SCALE3_H_
#define SRC_SCALE_SCALE3_H_

#include <cuda_runtime_api.h>

namespace hpc {
namespace scale {

void scale3_async(void* input_ptr, void* scale_ptr, void* scale2_ptr, void* output_ptr,
                  void* output_fp8_scale2_ptr, void* output_fp32_ptr, int num_tokens,
                  int hidden_state, bool is_moe, cudaStream_t stream);

}  // namespace scale
}  // namespace hpc

#endif  // SRC_SCALE_SCALE3_H_
