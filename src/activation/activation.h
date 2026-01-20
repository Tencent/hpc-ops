// Copyright (C) 2026 Tencent.

#ifndef SRC_ACTIVATION_ACTIVATION_H_
#define SRC_ACTIVATION_ACTIVATION_H_

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>
#include <stdint.h>

namespace hpc {
namespace activation {

void act_mul_and_quant_async(__nv_fp8_e4m3 *y_ptr, const __nv_bfloat16 *x_ptr,
                             const float *scale_ptr, const int num_row, const int num_col,
                             bool use_bf16_mul, cudaStream_t stream);

void act_mul_and_quant_async(__nv_fp8_e4m3 *out_ptr, const __nv_bfloat16 *gate_up_ptr,
                             const float *scale_ptr, const int *valid_row_range, const int num_row,
                             const int num_col, bool use_bf16_mul, cudaStream_t stream);

void masked_act_mul_and_quant_async(__nv_fp8_e4m3 *output_ptr, const __nv_bfloat16 *input_ptr,
                                    const float *scale_ptr, const int *num_per_expert_ptr,
                                    int num_total_tokens, int num_intermediate_size,
                                    int num_tokens_per_expert, cudaStream_t stream);

void act_mul_and_blockwise_quant_async(void *output_ptr, void *output_scale_ptr,
                                       const void *input_ptr,
                                       const void *cu_num_tokens_per_group_ptr,
                                       const void *cu_tiles_ptr, const int num_row,
                                       const int num_row_padded_size, const int num_col,
                                       const int num_group, const int num_tokens_per_group_avg,
                                       cudaStream_t stream);

void masked_act_mul_and_blockwise_quant_async(__nv_fp8_e4m3 *output_ptr, float *output_scale_ptr,
                                              const __nv_bfloat16 *input_ptr,
                                              const int *num_per_expert_ptr, int num_total_tokens,
                                              int num_intermediate_size, int num_tokens_per_expert,
                                              cudaStream_t stream);

}  // namespace activation
}  // namespace hpc

#endif  // SRC_ACTIVATION_ACTIVATION_H_
