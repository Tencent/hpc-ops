// Copyright (C) 2026 Tencent.
#ifndef SRC_FUSE_MOE_SM120_MXFP8_CONFIG_H_
#define SRC_FUSE_MOE_SM120_MXFP8_CONFIG_H_

#include <cuda.h>

#include <stdexcept>
#include <string>
#include <type_traits>

#include "cute/arch/copy_sm100.hpp"
#include "cute/atom/copy_traits_sm100.hpp"
#include "cute/atom/mma_traits_sm120.hpp"
#include "cute/tensor.hpp"
#include "cutlass/float_subbyte.h"
#include "src/utils/tma.cuh"

namespace hpc {
namespace fuse_moe {

using namespace cute;  // NOLINT

// The TiledMMA that partitions a block-scaled TiledMMA's scales: same atom shape, warps and
// permutation, with the atom's SFALayout/SFBLayout as its A/B layouts, so partition_A/B hand each
// thread the scales it gives the MMA (what cutlass' partition_fragment_SFA/SFB compute).
template <typename Atom, typename ThrLayout, typename Permutation>
CUTE_HOST_DEVICE constexpr auto make_tiled_sf(TiledMMA<Atom, ThrLayout, Permutation> const &) {
  struct SFAtom : Atom {
    using LayoutA_TV = typename Atom::Traits::SFALayout;
    using LayoutB_TV = typename Atom::Traits::SFBLayout;
  };
  return TiledMMA<SFAtom, ThrLayout, Permutation>{};
}

template <typename TC_, typename TW_, int kTileM_, int kTileN_, int kTileK_>
struct GroupGemmMxfp8Config {
  using TA = uint8_t;
  // B's shared-memory element type; an fp4 B is expanded to one byte per element too, see TW.
  using TB = uint8_t;
  // B (weight) gmem type: uint8_t for mxfp8, float_e2m1_unpacksmem_t for mxfp4.
  // The latter makes make_tma_copy pick CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,
  // which expands every 16 fp4 (8 gmem bytes) to 16 shm bytes, so the shm layout, the
  // swizzle and TiledCopyB's partitioning all stay those of fp8.
  using TW = TW_;
  static constexpr bool kIsFp4W = sizeof_bits_v<TW> == 4;
  // Epilogue write-back type: fp8 (uint8) for gate_up, bf16 for down.
  using TC = TC_;
  using TSF = float_ue8m0_t;

  static constexpr bool kIsGateUp = sizeof(TC) == 1;
  static constexpr int kTileM = kTileM_;  // 32/64/128
  static constexpr int kTileN = kTileN_;  // 128
  static constexpr int kTileK = kTileK_;  // 128
  static constexpr int kSFVecSize = 32;
  static constexpr int kSFPerTile = kTileK / kSFVecSize;  // 4

  // Scales use the flashinfer layout_128x4.
  static constexpr int kSFBlockM = 128;
  static constexpr int kSFBlockBytes = kSFBlockM * kSFPerTile * sizeof(TSF);
  static constexpr int kSFABytesPerStage = kIsGateUp ? kTileM * kSFPerTile : kSFBlockBytes;
  static constexpr int kTileNC = kIsGateUp ? kTileN / 2 : kTileN;

  static constexpr int kStageABMax = 4;
  static constexpr int kShmBudget = 98 * 1024;
  static constexpr int kCBytes = sizeof(TC) * kTileM * kTileNC;
  static constexpr int kStageBytes = sizeof(TA) * kTileM * kTileK + sizeof(TB) * kTileN * kTileK +
                                     kSFABytesPerStage + kSFBlockBytes;
  static constexpr int kStageABFit = (kShmBudget - kCBytes) / kStageBytes;
  static constexpr int kStageAB = kStageABFit > kStageABMax ? kStageABMax : kStageABFit;
  static_assert(kStageAB >= 2, "the pipeline needs at least 2 stages");
  static constexpr int kStageTask = 2;

  // shm A/B/C
  using SLayoutA = decltype(tile_to_shape(
      UMMA::Layout_K_SW128_Atom<TA>{}, make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStageAB>{})));
  using SLayoutB = decltype(tile_to_shape(
      UMMA::Layout_K_SW128_Atom<TB>{}, make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStageAB>{})));
  using CAtom =
      std::conditional_t<kIsGateUp, UMMA::Layout_K_SW64_Atom<TC>, UMMA::Layout_K_SW128_Atom<TC>>;
  using SLayoutC = decltype(tile_to_shape(CAtom{}, make_shape(Int<kTileM>{}, Int<kTileNC>{})));

  // copybox
  using CopyBoxA = decltype(tile_to_shape(UMMA::Layout_K_SW128_Atom<TA>{},
                                          make_shape(Int<kTileM>{}, Int<kTileK>{})));
  using CopyBoxB = decltype(tile_to_shape(UMMA::Layout_K_SW128_Atom<TB>{},
                                          make_shape(Int<kTileN>{}, Int<kTileK>{})));
  using CopyBoxC = SLayoutC;

  // shm SFA/SFB
  // gate_up's SFA rows go to one warp's lanes kSFARowsPerLane at a time.
  static constexpr int kSFARowsPerLane = 4;
  static_assert(!kIsGateUp || (kTileM % kSFARowsPerLane == 0 && kTileM / kSFARowsPerLane <= 32),
                "one warp's lanes must cover A's kTileM scale rows");
  static_assert(kTileN == kSFBlockM && kSFPerTile == 4,
                "one tile of B's scales is exactly one 128x4 block");
  using SLayoutSF128x4 =
      Layout<Shape<_4, _128, Int<kStageAB>>, Stride<_128, _1, Int<kSFBlockBytes>>>;
  using CopyBoxSF = Layout<Shape<_4, _128>, Stride<_128, _1>>;

  using SLayoutSFARow = Layout<Shape<Int<kTileM>, Int<kSFPerTile>, Int<kStageAB>>,
                               Stride<Int<kSFPerTile>, _1, Int<kTileM * kSFPerTile>>>;
  using SLayoutSFA = std::conditional_t<kIsGateUp, SLayoutSFARow, SLayoutSF128x4>;
  using SLayoutSFB = SLayoutSF128x4;

  // A 128x4 block keeps row r at (r % 32) * 16 + r / 32 * 4.
  using SLayoutSF128x4MK = Layout<Shape<Shape<_32, _4>, Shape<_32, Int<kSFPerTile>>, Int<kStageAB>>,
                                  Stride<Stride<_16, _4>, Stride<_0, _1>, Int<kSFBlockBytes>>>;
  // down's A tile is kTileM of a block's rows, so the block holds kTilePerSFBlock of them:
  // (M, K, stage, tile in block), tile t starting at t * kTileM / 32 * 4.
  static constexpr int kTilePerSFBlock = kSFBlockM / kTileM;
  using SLayoutSF128x4TileMK =
      Layout<Shape<Shape<_32, Int<kTileM / 32>>, Shape<_32, Int<kSFPerTile>>, Int<kStageAB>,
                   Int<kTilePerSFBlock>>,
             Stride<Stride<_16, _4>, Stride<_0, _1>, Int<kSFBlockBytes>, Int<kTileM / 32 * 4>>>;
  using SLayoutSFARowMK = Layout<Shape<Int<kTileM>, Shape<_32, Int<kSFPerTile>>, Int<kStageAB>>,
                                 Stride<Int<kSFPerTile>, Stride<_0, _1>, Int<kTileM * kSFPerTile>>>;
  using SLayoutSFAMK = std::conditional_t<kIsGateUp, SLayoutSFARowMK, SLayoutSF128x4TileMK>;
  using SLayoutSFBMK = SLayoutSF128x4MK;

  // mma
  static constexpr int kNumMmaWarps = 8;
  static constexpr int kMathThreads = 32 * kNumMmaWarps;
  static constexpr int kNumThreads = kMathThreads + 128;
  static constexpr int kMathRegisters = 232;
  static constexpr int kLoadRegisters = 40;
  static_assert(2 * kMathRegisters + kLoadRegisters <= 512, "CTA register pool exhausted");

  using WarpLayout = Layout<Shape<_2, _4, _1>, Stride<_4, _1, _0>>;
  // (coordinate within the atom, warp coordinate, repeated block within the warp)
  using PermM = Layout<Shape<_16, _2, Int<kTileM / 32>>, Stride<_1, _16, _32>>;
  using PermN = Layout<Shape<_8, _4, _4>, Stride<_1, _32, _8>>;
  // cute's block-scaled atom takes one UE8M0 per 32-K step (byte-id fixed at 0), fed through
  // zipped A/SFA and B/SFB tensors.
  using MmaOp = SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
      float_e4m3_t, std::conditional_t<kIsFp4W, cutlass::float_e2m1_t, float_e4m3_t>, float,
      float_ue8m0_t, kSFVecSize>;
  using TiledMma = decltype(make_tiled_mma(MmaOp{}, WarpLayout{}, Tile<PermM, PermN, _32>{}));
  using TiledSF = decltype(make_tiled_sf(TiledMma{}));

  // LDSM/STSM
  using TiledCopyA = decltype(make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, TA>{}, TiledMma{}));
  // fp4 uses ldmatrix .b8x16.b4x16_p64: it skips the padding and puts each fp4 in a byte's low 4
  // bits
  using LdsmOpB = std::conditional_t<kIsFp4W, SM100_SU4_DU8x16_x4_LDSM_N, SM75_U32x4_LDSM_N>;
  using TiledCopyB = decltype(make_tiled_copy_B(Copy_Atom<LdsmOpB, TB>{}, TiledMma{}));
  using StsmOp = std::conditional_t<(kTileM / 32) % 2 == 0, SM90_U32x4_STSM_N, SM90_U32x2_STSM_N>;
  using TiledCopyC = decltype(make_tiled_copy_C(Copy_Atom<StsmOp, TC>{}, TiledMma{}));

  static_assert((kTileM == 32 || kTileM == 64 || kTileM == 128) && kTileN == 128,
                "MMA permutation requires kTileM in {32, 64, 128} and kTileN = 128");
  static_assert(size(TiledMma{}) == kNumMmaWarps * 32, "MMA must use 8 warps");

  static constexpr int kShmABytes = sizeof_bits_v<TA> * cosize(SLayoutA{}) / 8;
  static constexpr int kShmBBytes = sizeof_bits_v<TB> * cosize(SLayoutB{}) / 8;
  static constexpr int kShmCBytes = sizeof_bits_v<TC> * cosize(SLayoutC{}) / 8;
  static constexpr int kShmSFABytes = sizeof_bits_v<TSF> * cosize(SLayoutSFA{}) / 8;
  static constexpr int kShmSFBBytes = sizeof_bits_v<TSF> * cosize(SLayoutSFB{}) / 8;
  static constexpr int kShmSize =
      kShmABytes + kShmBBytes + kShmCBytes + kShmSFABytes + kShmSFBBytes;
  static_assert(kShmSize <= kShmBudget, "shm overflow");
};

template <int kTileM, typename TW>
using GateUpConfig = GroupGemmMxfp8Config<uint8_t, TW, kTileM, 128, 128>;
template <int kTileM, typename TW>
using DownConfig = GroupGemmMxfp8Config<cute::bfloat16_t, TW, kTileM, 128, 128>;

template <typename F>
void dispatch_topk(int num_topk, F &&f) {
  switch (num_topk) {
    case 8:
      f(std::integral_constant<int, 8>{});
      break;
    case 16:
      f(std::integral_constant<int, 16>{});
      break;
    default:
      throw std::invalid_argument("fuse_moe_mxfp8: unsupported num_topk " +
                                  std::to_string(num_topk) + ", supported: 8, 16");
  }
}

// W's gmem type: float_e2m1_unpacksmem_t for mxfp4 weights, uint8_t otherwise.
template <typename F>
void dispatch_w_type(bool is_fp4, F &&f) {
  if (is_fp4) {
    f(cutlass::detail::float_e2m1_unpacksmem_t{});
  } else {
    f(uint8_t{});
  }
}

template <typename F>
void dispatch_tile_m(int tile_m, F &&f) {
  switch (tile_m) {
    case 32:
      f(std::integral_constant<int, 32>{});
      break;
    case 64:
      f(std::integral_constant<int, 64>{});
      break;
    default:
      f(std::integral_constant<int, 128>{});
      break;
  }
}

}  // namespace fuse_moe
}  // namespace hpc

#endif  // SRC_FUSE_MOE_SM120_MXFP8_CONFIG_H_
