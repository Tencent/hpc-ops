// Copyright (C) 2026 Tencent.
#ifndef SRC_GROUP_GEMM_CP_ASYNC_COMMON_CUH_
#define SRC_GROUP_GEMM_CP_ASYNC_COMMON_CUH_

#include "cute/tensor.hpp"
#include "cutlass/fast_math.h"

namespace hpc {
namespace group_gemm_cp_async {
namespace kernels {

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

static constexpr int grid_multiplier(int tile_m) {
  if (tile_m <= 8) {
    return 8;
  }
  if (tile_m <= 16) {
    return 9;
  }
  if (tile_m <= 32) {
    return 8;
  }
  if (tile_m <= 48) {
    return 7;
  }
  return 5;
}

}  // namespace group_gemm_cp_async
}  // namespace hpc

#endif  // SRC_GROUP_GEMM_CP_ASYNC_COMMON_CUH_
