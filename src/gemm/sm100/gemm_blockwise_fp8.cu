// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "cutlass/arch/reg_reconfig.h"
#include "src/gemm/gemm.h"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace gemm {

bool gemm_blockwise_fp8_async(void *y_ptr, void *split_y_ptr, void *split_flag_ptr,
                              const void *x_ptr, const void *w_ptr, const void *x_scale_ptr,
                              const void *weight_scale_ptr, const void *bias_ptr, int m, int n,
                              int k, int m_pad, int num_block_k, int num_block_n, int splitk,
                              cudaStream_t stream) {
  return false;
}

bool pad_and_transpose_async(void *new_scale_ptr, const void *scale_ptr, int m, int n, int m_pad,
                             cudaStream_t stream) {
  return false;
}

}  // namespace gemm
}  // namespace hpc
