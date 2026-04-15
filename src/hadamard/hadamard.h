// Copyright 2025 hpc-ops authors

#ifndef SRC_HADAMARD_HADAMARD_H_
#define SRC_HADAMARD_HADAMARD_H_

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

namespace hpc {
namespace hadamard {

bool hadamard_transform_async(const void* input_ptr, void* output_ptr, float inv_sqrt_d, int n,
                              int num_rows, int input_elem_size, cudaStream_t stream);

bool act_mul_hadamard_blockwise_quant_async(const __nv_bfloat16* gate_up_ptr,
                                            __nv_fp8_e4m3* output_ptr, float* output_scale_ptr,
                                            const int* valid_row_range_ptr, int num_rows,
                                            int num_col, float upper_max, int block_size,
                                            bool use_pdl, cudaStream_t stream);

bool act_mul_hadamard_per_tensor_quant_async(const __nv_bfloat16* gate_up_ptr,
                                             __nv_fp8_e4m3* output_ptr, const float* scale_inv_ptr,
                                             const int* valid_row_range_ptr, int num_rows,
                                             int num_col, bool use_pdl, cudaStream_t stream);

}  // namespace hadamard
}  // namespace hpc

#endif  // SRC_HADAMARD_HADAMARD_H_
