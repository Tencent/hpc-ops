// Copyright 2026 hpc-ops authors

#ifndef SRC_GROUP_GEMM_SM90_CP_ASYNC_BUILD_TASK_MAP_H_
#define SRC_GROUP_GEMM_SM90_CP_ASYNC_BUILD_TASK_MAP_H_

#include <cuda_runtime_api.h>

namespace hpc {
namespace group_gemm_cp_async {

// Host wrapper that launches the task_map build kernel.
//   task_map_ptr    : pre-allocated int4 buffer (must be memset to 0xFF before this call)
//   cu_tiles_ptr    : exclusive prefix sum of tile_m per group (length = num_group+1)
//   tiles_ptr       : tile_m per group (length = num_group)
//   num_group       : number of groups
//   num_tile_n      : number of N-direction tiles (= ceil(n / kTileN))
//   use_pdl         : whether to emit PDL sync at kernel boundaries
//   stream          : CUDA stream
void launch_build_task_map(void *task_map_ptr, const void *cu_tiles_ptr, const void *tiles_ptr,
                           int num_group, int num_tile_n, bool use_pdl, cudaStream_t stream);

// Fused variant: fill BOTH gateup and down task_maps in a single kernel
// launch. Each block handles one expert and shares the per-expert
// (cu_tiles, tiles) loads across both maps. Either map_ptr may be nullptr
// (the corresponding map is skipped).
//
// The grid has `num_group + 1` blocks — the extra block writes the
// sentinel value (igroup = -1) into the unused tail of each task_map, so
// callers do NOT need to cudaMemsetAsync the buffers before this launch.
// `gateup_task_map_len` / `down_task_map_len` give the full allocated
// size (in int4 entries) of each buffer.
void launch_build_two_task_maps(void *gateup_task_map_ptr, void *down_task_map_ptr,
                                const void *cu_tiles_ptr, const void *tiles_ptr, int num_group,
                                int gate_up_num_tile_n, int down_num_tile_n,
                                int gateup_task_map_len, int down_task_map_len, bool use_pdl,
                                cudaStream_t stream);

}  // namespace group_gemm_cp_async
}  // namespace hpc

#endif  // SRC_GROUP_GEMM_SM90_CP_ASYNC_BUILD_TASK_MAP_H_
