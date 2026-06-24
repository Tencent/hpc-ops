// Copyright 2026 hpc-ops authors

#ifndef SRC_GROUP_GEMM_SM100_MXFP8_DISPATCH_CUH_
#define SRC_GROUP_GEMM_SM100_MXFP8_DISPATCH_CUH_

#include "src/group_gemm/sm100/mxfp8/config.h"

namespace hpc {
namespace group_gemm {

template <typename T>
struct type_id {
  using type = T;
};

template <typename Tin, typename Tout, typename Tsf, typename TinB, typename Fn>
auto group_gemm_1sm_mxfp8_dispatch_selector(int kTileM_dispatch, Fn &&fn) {
  constexpr int kTileN = 128;
  constexpr int kTileK = 128;
  constexpr int kMmaSM = 1;
  switch (kTileM_dispatch) {
    case 16:
      return fn(type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 16, kTileN, kTileK, 16, 8, 4, kMmaSM,
                                             4, TinB>>{});
    case 32:
      return fn(type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 32, kTileN, kTileK, 32, 6, 4, kMmaSM,
                                             4, TinB>>{});
    case 64:
      return fn(type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 64, kTileN, kTileK, 64, 6, 4, kMmaSM,
                                             4, TinB>>{});
    case 96:
      return fn(type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 96, kTileN, kTileK, 32, 4, 4, kMmaSM,
                                             4, TinB>>{});
    case 128:
      return fn(type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 128, kTileN, kTileK, 64, 3, 3, kMmaSM,
                                             3, TinB>>{});
    case 160:
      return fn(type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 160, kTileN, kTileK, 32, 3, 2, kMmaSM,
                                             2, TinB>>{});
    case 192:
      return fn(type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 192, kTileN, kTileK, 32, 3, 2, kMmaSM,
                                             2, TinB>>{});
    default:
      return fn(type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 256, kTileN, kTileK, 64, 3, 2, kMmaSM,
                                             1, TinB>>{});
  }
}

inline int mxfp8_dispatch_kTileM(int num_seq_per_group_avg) {
  if (num_seq_per_group_avg <= 16) {
    return 16;
  }
  if (num_seq_per_group_avg <= 32) {
    return 32;
  }
  if (num_seq_per_group_avg <= 64) {
    return 64;
  }
  if (num_seq_per_group_avg <= 96) {
    return 96;
  }
  if (num_seq_per_group_avg <= 128) {
    return 128;
  }
  if (num_seq_per_group_avg <= 160) {
    return 160;
  }
  if (num_seq_per_group_avg <= 192) {
    return 192;
  }
  return 256;
}
}  // namespace group_gemm
}  // namespace hpc

#endif  // SRC_GROUP_GEMM_SM100_MXFP8_DISPATCH_CUH_
