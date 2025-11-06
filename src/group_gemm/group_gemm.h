// Copyright 2025 hpc-ops authors

#ifndef SRC_GROUP_GEMM_GROUP_GEMM_H_
#define SRC_GROUP_GEMM_GROUP_GEMM_H_

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace hpc {
namespace group_gemm {

void group_gemm_fp8_async(void *y_ptr, const void *x_ptr, const void *w_ptr,
                          const void *seqlens_ptr, const void *cu_seqlens_ptr, const void *y_scale,
                          void *tmas_ptr, void *tiles_ptr, void *cu_tiles_ptr, int num_group, int m,
                          int n, int k, cudaStream_t stream);

}  // namespace group_gemm
}  // namespace hpc

#endif  // SRC_GROUP_GEMM_GROUP_GEMM_H_
