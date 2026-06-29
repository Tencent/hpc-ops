// Copyright 2025 hpc-ops authors

#ifndef SRC_GEMM_GEMM_H_
#define SRC_GEMM_GEMM_H_

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace hpc {
namespace gemm {

bool pad_and_transpose_async(void *new_scale_ptr, const void *scale_ptr, int m, int n, int m_pad,
                             cudaStream_t stream);

bool gemm_blockwise_fp8_async(void *y_ptr, void *split_y_ptr, void *split_flag_ptr,
                              const void *x_ptr, const void *weight_ptr, const void *x_scale_ptr,
                              const void *weight_scale_ptr, const void *bias_ptr, int m, int n,
                              int k, int m_pad, int num_block_k, int num_block_n, int splitk,
                              cudaStream_t stream);

bool gemm_blockwise_fp8_low_latency_async(void *y_ptr, const void *x_ptr, const void *w_ptr,
                                          const void *x_scale_ptr, const void *weight_scale_ptr,
                                          int m, int n, int k, int x_scale_stride,
                                          int w_scale_stride, cudaStream_t stream);

bool gemm_fp8_async(void *y_ptr, const void *x_ptr, const void *w_ptr, int m, int n, int k,
                    cudaStream_t stream);

bool gemm_bf16_async(void *y_ptr, const void *x_ptr, const void *w_ptr, int m, int n, int k,
                     cudaStream_t stream);

bool gemm_bf16xfp32_async(void *y_ptr, void *splitk_y_ptr, void *split_flag_ptr, const void *x_ptr,
                          const void *w_high_ptr, const void *w_low_ptr, int m, int n, int k,
                          float scale, bool use_fp32_output, int splitk, cudaStream_t stream);
}  // namespace gemm
}  // namespace hpc

#endif  // SRC_GEMM_GEMM_H_
