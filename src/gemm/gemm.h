// Copyright 2025 hpc-ops authors

#ifndef SRC_GEMM_GEMM_H_
#define SRC_GEMM_GEMM_H_

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace hpc {
namespace gemm {

void pad_and_transpose_async(void *new_scale_ptr, const void *scale_ptr, int m, int n, int m_pad,
                             cudaStream_t stream);

void gemm_blockwise_async(void *y_ptr, const void *x_ptr, const void *weight_ptr,
                          const void *x_scale_ptr, const void *weight_scale_ptr,
                          const void *bias_ptr, int m, int n, int k, int m_pad,
                          cudaStream_t stream);

bool gemm_bf16xfp32_async(void *y_ptr, const void *x_ptr, const void *w_high_ptr,
                          const void *w_low_ptr, int m, int n, int k, float scale,
                          bool use_fp32_output, cudaStream_t stream);
}  // namespace gemm
}  // namespace hpc

#endif  // SRC_GEMM_GEMM_H_
