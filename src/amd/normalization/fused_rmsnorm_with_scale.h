// Copyright 2025 hpc-ops authors
// HIP port: torch/ATen includes dropped (only entry.cc needs torch, and it
// includes those itself). Kernel launcher declaration only needs hipStream_t.

#ifndef SRC_AMD_NORMALIZATION_FUSED_RMSNORM_WITH_SCALE_H_
#define SRC_AMD_NORMALIZATION_FUSED_RMSNORM_WITH_SCALE_H_

#include <hip/hip_runtime_api.h>  // hipStream_t only; device types live in the .cu

#include <tuple>

namespace hpc {
namespace normalization {

bool fused_rmsnorm_with_scale_async(const void* input_ptr, const void* weight_ptr, void* output_ptr,
                                    void* output_fp32_ptr, void* output_fp8_scale2_ptr,
                                    const void* scale, float eps, int batch_size, int hidden_state,
                                    bool is_moe, hipStream_t stream);

}  // namespace normalization
}  // namespace hpc

#endif  // SRC_AMD_NORMALIZATION_FUSED_RMSNORM_WITH_SCALE_H_
