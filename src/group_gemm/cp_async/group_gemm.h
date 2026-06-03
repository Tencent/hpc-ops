// Copyright (C) 2026 Tencent.

#ifndef SRC_GROUP_GEMM_CP_ASYNC_GROUP_GEMM_H_
#define SRC_GROUP_GEMM_CP_ASYNC_GROUP_GEMM_H_

#include <cuda_runtime_api.h>

namespace hpc {
namespace group_gemm_cp_async {

void group_gemm_fp8_multistage_async(void *y_ptr, const void *x_ptr, const void *w_ptr,
                                     const void *y_scale_ptr, const void *seqlens_ptr,
                                     const void *cu_seqlens_ptr, const void *tiles_ptr,
                                     const void *cu_tiles_ptr, const void *task_map_ptr,
                                     int task_map_len, int m, int n, int k, int num_group,
                                     int num_seq_per_group_avg, bool use_pdl, cudaStream_t stream);

void group_gemm_fp8_scatter_async(void *y_ptr, const void *x_ptr, const void *w_ptr,
                                  const void *y_scale_ptr, const void *row_indices_ptr,
                                  const void *seqlens_ptr, const void *cu_seqlens_ptr,
                                  const void *tiles_ptr, const void *cu_tiles_ptr,
                                  const void *task_map_ptr, int task_map_len, int m, int n, int k,
                                  int num_group, int num_seq_per_group_avg, bool use_pdl,
                                  cudaStream_t stream);

}  // namespace group_gemm_cp_async
}  // namespace hpc

#endif  // SRC_GROUP_GEMM_CP_ASYNC_GROUP_GEMM_H_
