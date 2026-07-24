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

// Route-direct small-M GEMM. Each logical row selects its expert from topk_ids;
// no expert sorting, prefix sum, or task map is required. When input_is_token
// is true, route r reads input row r / num_topk (Gate/Up); otherwise it reads
// route row r directly (Down).
void group_gemm_fp8_route_async(void *y_ptr, const void *x_ptr, const void *w_ptr,
                                const void *y_scale_ptr, const void *topk_ids_ptr,
                                int num_routes, int num_topk, int n, int k,
                                int num_expert_local, int rank_ep, bool input_is_token,
                                cudaStream_t stream);

// Route-direct blockwise-scaled GEMM. Input and weight scales describe
// 128-element K blocks; weight_scale_stride is the padded K-block dimension.
void group_gemm_fp8_route_blockwise_async(
    void *y_ptr, const void *x_ptr, const void *x_scale_ptr,
    const void *w_ptr, const void *w_scale_ptr, const void *topk_ids_ptr,
    int num_routes, int num_topk, int n, int k, int num_splits,
    int num_expert_local, int rank_ep, bool input_is_token, int weight_scale_stride,
    cudaStream_t stream);

// Route-direct Gate/Up projection split along K. Output is BF16 with layout
// [num_routes, num_splits, n], consumed by the activation kernel.
void group_gemm_fp8_route_splitk_async(void *partial_ptr, const void *x_ptr,
                                       const void *w_ptr, const void *y_scale_ptr,
                                       const void *topk_ids_ptr, int num_routes,
                                       int num_topk, int n, int k, int num_splits,
                                       int num_expert_local, int rank_ep,
                                       cudaStream_t stream);

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
