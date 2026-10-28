// Copyright (C) 2026 Tencent.

#pragma once

#include <cute/tensor.hpp>

namespace hpc::fuse_moe::sm120::gemm {
namespace common {

// Fewest row blocks a warp with kBlocks of them runs. A warp that owns a single
// block runs it even when it is all padding: idling it does not shorten the tile,
// and measured 3-5% slower on small batches. With two blocks per warp the idle
// warp saves power that shows up under sustained load (2-3% on half-empty tiles).
template <int kBlocks>
constexpr int kMinLiveRowBlocks = kBlocks == 1 ? 1 : 0;

// How many of this thread's MMA row blocks hold a row below valid_rows, at least
// kMinLiveRowBlocks. Rows from valid_rows on are the padding that rounds an
// expert up to a whole tile. The blocks run in row order, so the live ones come
// first, and all rows of an MMA atom belong to one warp, so the count is
// warp-uniform.
template <class ThrMma, class TileMN, class Accum>
CUTE_DEVICE int live_row_blocks(ThrMma const& thr_mma, TileMN const& tile_mn, Accum const& accum,
                                int valid_rows) {
  constexpr int kAtomM = decltype(cute::size<0>(typename ThrMma::AtomShape_MNK{}))::value;
  constexpr int kBlocks = decltype(cute::size<1>(accum))::value;
  auto coords = thr_mma.partition_C(cute::make_identity_tensor(tile_mn));
  int live = kMinLiveRowBlocks<kBlocks>;
  CUTE_UNROLL
  for (int m = kMinLiveRowBlocks<kBlocks>; m < kBlocks; ++m) {
    live += cute::get<0>(coords(0, m, 0)) / kAtomM * kAtomM < valid_rows;
  }
  return live;
}

// Calls body(Int<live>{}). Each count gets its own copy of the body, so the
// branch is taken once per tile and not inside the mainloop.
template <int kBlocks, int kLive = kBlocks, class Body>
CUTE_DEVICE void with_live_row_blocks(int live, Body&& body) {
  if constexpr (kLive == kMinLiveRowBlocks<kBlocks>) {
    body(cute::Int<kLive>{});
  } else if (live == kLive) {
    body(cute::Int<kLive>{});
  } else {
    with_live_row_blocks<kBlocks, kLive - 1>(live, body);
  }
}

}  // namespace common
}  // namespace hpc::fuse_moe::sm120::gemm
