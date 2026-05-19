// Copyright 2026 hpc-ops authors

#ifndef SRC_GROUP_GEMM_SM90_CP_ASYNC_GROUP_GEMM_H_
#define SRC_GROUP_GEMM_SM90_CP_ASYNC_GROUP_GEMM_H_

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace hpc {
namespace group_gemm_cp_async {

// `task_map_ptr`:
//   - nullptr → fallback: kernel uses runtime `get_next_tile_horizon` scan
//   - non-null → kernel reads pre-computed int4 entries (igroup, itile_m, itile_n, 0);
//                entries past the end must be igroup=-1 (sentinel).
// `task_map_len`: length of the task_map (in int4 entries). Unused when task_map_ptr == nullptr.
// `use_pdl`: when true, the kernel uses `cudaGridDependencySynchronize` at entry
//   and `cudaTriggerProgrammaticLaunchCompletion` at exit; the launch is via
//   `cudaLaunchKernelEx` with `ProgrammaticStreamSerialization` so that this
//   kernel can be chained with PDL-aware upstream/downstream kernels.
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

#endif  // SRC_GROUP_GEMM_SM90_CP_ASYNC_GROUP_GEMM_H_
