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

bool gemm_fp8_async(void *y_ptr, const void *x_ptr, const void *w_ptr, int m, int n, int k,
                    cudaStream_t stream) {
  return false;
}

}  // namespace gemm
}  // namespace hpc
