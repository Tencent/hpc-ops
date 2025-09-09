// Copyright 2025 hpc-ops authors

#ifndef SRC_GEM3_GEMM_H_
#define SRC_GEM3_GEMM_H_

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace hpc {
namespace gem3 {

void gemm_async(void *y_ptr, const void *x_ptr, const void *weight_ptr, int m, int n, int k,
                cudaStream_t stream);

}  // namespace gem3
}  // namespace hpc

#endif  // SRC_GEM3_GEMM_H_
