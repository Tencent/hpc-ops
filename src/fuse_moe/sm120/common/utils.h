// Copyright (C) 2026 Tencent.

#ifndef SRC_FUSE_MOE_SM120_COMMON_UTILS_H_
#define SRC_FUSE_MOE_SM120_COMMON_UTILS_H_

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace hpc::fuse_moe::sm120 {

constexpr int kMaxExperts = 512;
constexpr int kMaxTopK = 128;
constexpr int kRouteThreads = 256;
// Up to this many tokens one block maps every route; past it the routing
// counts in several blocks and accumulates into route_counts.
constexpr int kSmallRoutingMaxTokens = 128;
constexpr int kScaleVectorSize = 32;
constexpr int kWarpSize = 32;
constexpr int kIntsPerVector = 4;
constexpr int kBf16PerLane = 8;
constexpr int kLanesPerMxBlock = kScaleVectorSize / kBf16PerLane;

// Host launch switch for the four GEMM tiles. fn is called as fn.template
// operator()<TileM, TileN>(). Returns false if the pair is not one of them.
template <typename Fn>
inline bool dispatch_tile_mn(int tile_m, int tile_n, Fn&& fn) {
  if (tile_m == 32 && tile_n == 64) {
    fn.template operator()<32, 64>();
  } else if (tile_m == 32 && tile_n == 128) {
    fn.template operator()<32, 128>();
  } else if (tile_m == 64 && tile_n == 64) {
    fn.template operator()<64, 64>();
  } else if (tile_m == 64 && tile_n == 128) {
    fn.template operator()<64, 128>();
  } else {
    return false;
  }
  return true;
}

// Packed activation scale layout. Two producers write it: prepare_input writes the
// gate-up GEMM's activation scales, and the gate-up epilogue writes the down GEMM's.
// Both GEMMs' TMA loads read it, so a layout change touches all four places. Each
// int32 word holds the UE8M0 scales of four consecutive 32-column K blocks; word
// pack_col of scale row r sits at r + pack_col * scale_m_padded, so rows are
// contiguous within a word column. Each expert's scale rows start at a four-row
// boundary: expert e's first padded row expert_row_begin maps to
// packed_scale_expert_base(expert_row_begin, e).
// T may be wider than int for the intermediate result, but the plan and the TMA
// alignment helper (SfLoadConfig::get_tma_aligned_size) take int extents, so the
// path as a whole supports int extents only.
template <typename T, typename U>
__host__ __device__ constexpr T packed_scale_expert_base(T expert_row_begin, U expert) {
  return (expert_row_begin + static_cast<T>(expert) * 3) / 4 * 4;
}

__host__ __device__ constexpr int packed_scale_row(int row, int expert_row_begin, int expert) {
  return packed_scale_expert_base(expert_row_begin, expert) + row - expert_row_begin;
}

}  // namespace hpc::fuse_moe::sm120

#endif  // SRC_FUSE_MOE_SM120_COMMON_UTILS_H_
