// Copyright (C) 2026 Tencent.

#ifndef SRC_GROUP_GEMM_CP_ASYNC_BUILD_TASK_MAP_H_
#define SRC_GROUP_GEMM_CP_ASYNC_BUILD_TASK_MAP_H_

#include <cuda_runtime_api.h>

namespace hpc {
namespace group_gemm_cp_async {

// Build one task map. The caller preinitializes unused entries with igroup = -1.
void launch_build_task_map(void *task_map_ptr, const void *cu_tiles_ptr, const void *tiles_ptr,
                           int num_group, int num_tile_n, bool use_pdl, cudaStream_t stream);

// Build gateup and down task maps. Tail entries are written with igroup = -1.
void launch_build_two_task_maps(void *gateup_task_map_ptr, void *down_task_map_ptr,
                                const void *cu_tiles_ptr, const void *tiles_ptr, int num_group,
                                int gate_up_num_tile_n, int down_num_tile_n,
                                int gateup_task_map_len, int down_task_map_len, bool use_pdl,
                                cudaStream_t stream);

}  // namespace group_gemm_cp_async
}  // namespace hpc

#endif  // SRC_GROUP_GEMM_CP_ASYNC_BUILD_TASK_MAP_H_
