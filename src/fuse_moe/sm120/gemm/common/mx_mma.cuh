// Copyright (C) 2026 Tencent.

#pragma once

// clang-format off
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm120.hpp>
#include <cute/layout.hpp>
// clang-format on

namespace hpc::fuse_moe::sm120::gemm {
namespace common {

using namespace cute;  // NOLINT

// SM120 allows 99 KB of dynamic SMEM per block; the barriers and the scheduler
// pipeline take well under the 2 KB left over.
constexpr int kTensorSmemBudget = 99 * 1024 - 2048;

template <typename ElementB, int kTileM, int kTileN>
struct MxMmaConfig {
  using MMA_Atom =
      cute::MMA_Atom<SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<cute::float_e4m3_t, ElementB, float,
                                                             cute::float_ue8m0_t, 32>>;

  static_assert(kTileN >= 8 && (kTileN % 8) == 0,
                "kTileN must be >= 8 and multiple of MMA atom_N=8");
  static constexpr int kNumMathWarpN = (kTileN >= 32) ? 4 : (kTileN / 8);
  static constexpr int kNumMathWarpM = 8 / kNumMathWarpN;
  static constexpr int kNumMathWarps = kNumMathWarpM * kNumMathWarpN;
  static constexpr int kNumMathThreads = kNumMathWarps * 32;
  static_assert(kNumMathWarps == 8, "Total math warps must be 8 (256 threads = 2 WG)");
  static_assert(kTileM >= kNumMathWarpM * 16,
                "kTileM must be >= kNumMathWarpM * 16 (ThrLayout M-direction lower bound).");

  using PermMmaTileM = Int<kTileM>;
  using PermMmaTileN = Int<kTileN>;

  using TiledMma = TiledMMA<
      MMA_Atom,
      Layout<Shape<Int<kNumMathWarpM>, Int<kNumMathWarpN>, _1>, Stride<_1, Int<kNumMathWarpM>, _0>>,
      Tile<PermMmaTileM, PermMmaTileN, Underscore>>;

  static_assert(size<2>(typename MMA_Atom::Shape_MNK{}) == 32,
                "MMA atom K-dim must be 32 to match SFVecSize");
};

}  // namespace common
}  // namespace hpc::fuse_moe::sm120::gemm
