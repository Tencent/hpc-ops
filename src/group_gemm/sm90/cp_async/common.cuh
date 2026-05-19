// Copyright 2026 hpc-ops authors
#ifndef SRC_GROUP_GEMM_SM90_CP_ASYNC_COMMON_CUH_
#define SRC_GROUP_GEMM_SM90_CP_ASYNC_COMMON_CUH_

// Shared helpers for group_gemm_fp8.cu (multistage Down GEMM) and
// group_gemm_fp8_scatter.cu (scatter Gate-Up GEMM).  Both kernels use the
// same persistent-block scheduler; the device helper get_next_tile_horizon
// and the host helper grid_multiplier are defined here and included by both.

#include "cute/tensor.hpp"
#include "cutlass/fast_math.h"

namespace hpc {
namespace group_gemm_cp_async {
namespace kernels {

// Persistent-block scheduler: given a flat iblock index, resolve which
// expert-group (igroup), which M-tile within that group (itile_m), and
// which N-tile (itile_n) this CTA should process next.  `sum_tile_m` is a
// running prefix-sum kept across calls to amortize the walk over groups.
//
// On return:
//   igroup = -1         -> this iblock is past the end; CTA should exit.
//   igroup, itile_m,
//   itile_n, sum_tile_m -> valid tile for this persistent iteration.
__device__ __forceinline__ void get_next_tile_horizon(const int *tiles_ptr, int iblock,
                                                      int num_group, int &igroup, int &itile_m,
                                                      int &itile_n, int &sum_tile_m,
                                                      cutlass::FastDivmod flat_divider) {
  int num_tile_m, itile_m_total;

  flat_divider(itile_m_total, itile_n, iblock);
  for (int i = igroup; i < num_group; i++) {
    num_tile_m = tiles_ptr[i];
    sum_tile_m += num_tile_m;
    if (itile_m_total < sum_tile_m) {
      igroup = i;
      sum_tile_m = sum_tile_m - num_tile_m;
      itile_m = itile_m_total - sum_tile_m;
      return;
    }
  }
  igroup = -1;
}

}  // namespace kernels

// Grid multiplier for persistent-block launch: grid.x = num_sm * grid_mul.
// The multiplier is tuned per kTileM to balance SM occupancy against the
// per-CTA scheduler walk cost.
static constexpr int grid_multiplier(int tile_m) {
  if (tile_m <= 8) return 8;
  if (tile_m <= 16) return 9;
  if (tile_m <= 32) return 8;
  if (tile_m <= 48) return 7;
  return 5;
}

}  // namespace group_gemm_cp_async
}  // namespace hpc

#endif  // SRC_GROUP_GEMM_SM90_CP_ASYNC_COMMON_CUH_
