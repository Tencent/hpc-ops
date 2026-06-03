// Copyright (C) 2026 Tencent.

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdint.h>

#include "src/group_gemm/cp_async/build_task_map.h"

namespace hpc {
namespace group_gemm_cp_async {
namespace kernels {

// Fill each group's task_map slice. Uncovered entries must already contain igroup = -1.
template <bool kUsePDL>
__global__ void build_task_map_kernel(int4 *task_map, const int *cu_tiles_ptr, const int *tiles_ptr,
                                      int num_tile_n) {
  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }
  int igroup = blockIdx.x;
  int start = cu_tiles_ptr[igroup];  // exclusive prefix of tile_m
  int num_tm = tiles_ptr[igroup];
  int total = num_tm * num_tile_n;
  for (int i = threadIdx.x; i < total; i += blockDim.x) {
    int itm = i / num_tile_n;
    int itn = i - itm * num_tile_n;
    int off = (start + itm) * num_tile_n + itn;
    int4 v;
    v.x = igroup;
    v.y = itm;
    v.z = itn;
    v.w = 0;
    task_map[off] = v;
  }
  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

// Fill gateup and down task maps; tail blocks mark unused entries with igroup = -1.
template <bool kUsePDL>
__global__ void build_two_task_maps_kernel(int4 *gateup_task_map, int4 *down_task_map,
                                           const int *cu_tiles_ptr, const int *tiles_ptr,
                                           int num_group, int gate_up_num_tile_n,
                                           int down_num_tile_n, int gateup_task_map_len,
                                           int down_task_map_len) {
  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }
  int igroup = blockIdx.x;
  if (igroup < num_group) {
    int start = cu_tiles_ptr[igroup];
    int num_tm = tiles_ptr[igroup];

    int total_gu = (gateup_task_map != nullptr) ? num_tm * gate_up_num_tile_n : 0;
    int total_dn = (down_task_map != nullptr) ? num_tm * down_num_tile_n : 0;
    int total = total_gu + total_dn;

    for (int i = threadIdx.x; i < total; i += blockDim.x) {
      if (i < total_gu) {
        int itm = i / gate_up_num_tile_n;
        int itn = i - itm * gate_up_num_tile_n;
        int off = (start + itm) * gate_up_num_tile_n + itn;
        int4 v;
        v.x = igroup;
        v.y = itm;
        v.z = itn;
        v.w = 0;
        gateup_task_map[off] = v;
      } else {
        int j = i - total_gu;
        int itm = j / down_num_tile_n;
        int itn = j - itm * down_num_tile_n;
        int off = (start + itm) * down_num_tile_n + itn;
        int4 v;
        v.x = igroup;
        v.y = itm;
        v.z = itn;
        v.w = 0;
        down_task_map[off] = v;
      }
    }
  } else {
    int tail_id = blockIdx.x - num_group;
    int tail_blocks = gridDim.x - num_group;
    int total_used = cu_tiles_ptr[num_group];
    int4 sentinel;
    sentinel.x = -1;
    sentinel.y = 0;
    sentinel.z = 0;
    sentinel.w = 0;
    int stride = tail_blocks * blockDim.x;
    if (gateup_task_map != nullptr) {
      int used = total_used * gate_up_num_tile_n;
      for (int i = used + tail_id * blockDim.x + threadIdx.x; i < gateup_task_map_len;
           i += stride) {
        gateup_task_map[i] = sentinel;
      }
    }
    if (down_task_map != nullptr) {
      int used = total_used * down_num_tile_n;
      for (int i = used + tail_id * blockDim.x + threadIdx.x; i < down_task_map_len; i += stride) {
        down_task_map[i] = sentinel;
      }
    }
  }
  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

}  // namespace kernels

void launch_build_two_task_maps(void *gateup_task_map_ptr, void *down_task_map_ptr,
                                const void *cu_tiles_ptr, const void *tiles_ptr, int num_group,
                                int gate_up_num_tile_n, int down_num_tile_n,
                                int gateup_task_map_len, int down_task_map_len, bool use_pdl,
                                cudaStream_t stream) {
  if (num_group <= 0) {
    return;
  }
  if (gateup_task_map_ptr == nullptr && down_task_map_ptr == nullptr) {
    return;
  }
  auto *gu = reinterpret_cast<int4 *>(gateup_task_map_ptr);
  auto *dn = reinterpret_cast<int4 *>(down_task_map_ptr);
  // Extra CTAs initialize the unused task-map suffix with sentinel entries.
  constexpr int kBlockSize = 128;
  int64_t total_bytes =
      (static_cast<int64_t>(gateup_task_map_len) + static_cast<int64_t>(down_task_map_len)) * 16;
  int bytes_per_block = kBlockSize * 16 * 8;  // ~16 KB / block
  int tail_blocks = static_cast<int>((total_bytes + bytes_per_block - 1) / bytes_per_block);
  if (tail_blocks < 1) {
    tail_blocks = 1;
  }
  if (tail_blocks > 32) {
    tail_blocks = 32;
  }
  int grid = num_group + tail_blocks;
  if (use_pdl) {
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(grid);
    cfg.blockDim = dim3(kBlockSize);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, kernels::build_two_task_maps_kernel</*kUsePDL=*/true>, gu, dn,
                       reinterpret_cast<const int *>(cu_tiles_ptr),
                       reinterpret_cast<const int *>(tiles_ptr), num_group, gate_up_num_tile_n,
                       down_num_tile_n, gateup_task_map_len, down_task_map_len);
  } else {
    kernels::build_two_task_maps_kernel</*kUsePDL=*/false><<<grid, kBlockSize, 0, stream>>>(
        gu, dn, reinterpret_cast<const int *>(cu_tiles_ptr),
        reinterpret_cast<const int *>(tiles_ptr), num_group, gate_up_num_tile_n, down_num_tile_n,
        gateup_task_map_len, down_task_map_len);
  }
}

void launch_build_task_map(void *task_map_ptr, const void *cu_tiles_ptr, const void *tiles_ptr,
                           int num_group, int num_tile_n, bool use_pdl, cudaStream_t stream) {
  if (num_group <= 0) {
    return;
  }
  if (task_map_ptr == nullptr) {
    return;
  }
  if (use_pdl) {
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(num_group);
    cfg.blockDim = dim3(128);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, kernels::build_task_map_kernel</*kUsePDL=*/true>,
                       reinterpret_cast<int4 *>(task_map_ptr),
                       reinterpret_cast<const int *>(cu_tiles_ptr),
                       reinterpret_cast<const int *>(tiles_ptr), num_tile_n);
  } else {
    kernels::build_task_map_kernel</*kUsePDL=*/false><<<num_group, 128, 0, stream>>>(
        reinterpret_cast<int4 *>(task_map_ptr), reinterpret_cast<const int *>(cu_tiles_ptr),
        reinterpret_cast<const int *>(tiles_ptr), num_tile_n);
  }
}

}  // namespace group_gemm_cp_async
}  // namespace hpc
