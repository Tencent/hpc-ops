// Copyright 2026 hpc-ops authors

#ifndef SRC_GROUP_GEMM_SM100_MXFP8_CONFIG_H_
#define SRC_GROUP_GEMM_SM100_MXFP8_CONFIG_H_

#include "cute/algorithm/gemm.hpp"
#include "cute/arch/copy_sm100.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/arch/copy_sm90_tma.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/atom/copy_traits_sm100.hpp"
#include "cute/atom/copy_traits_sm90.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_sm100.hpp"
#include "cute/tensor.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/float8.h"
#include "cutlass/float_subbyte.h"
#include "cutlass/numeric_types.h"

namespace hpc {
namespace group_gemm {

using namespace cute;  // NOLINT

template <int kMmaSM_, typename Tin, typename TinB, typename Tsf, int kTileM, int kTileN>
struct MxFp8MmaAtomSelector;

// swapAB: MMA-A operand = B(weight, TinB), MMA-B operand = A(activation, Tin)
template <typename Tin, typename TinB, typename Tsf, int kTileM, int kTileN>
struct MxFp8MmaAtomSelector<1, Tin, TinB, Tsf, kTileM, kTileN> {
  using type = decltype(make_tiled_mma(SM100_MMA_MXF8F6F4_SS<TinB, Tin, float, Tsf, kTileN, kTileM,
                                                             UMMA::Major::K, UMMA::Major::K>{}));
};

template <typename Tin, typename TinB, typename Tsf, int kTileM, int kTileN>
struct MxFp8MmaAtomSelector<2, Tin, TinB, Tsf, kTileM, kTileN> {
  using type =
      decltype(make_tiled_mma(SM100_MMA_MXF8F6F4_2x1SM_SS<TinB, Tin, float, Tsf, kTileN, kTileM,
                                                          UMMA::Major::K, UMMA::Major::K>{}));
};

// kStageTile: TMEM C-accumulator depth. Higher values overlap more MMA with epi store,
// but each extra slot costs kTileM TMEM cols.
// kStageTile * kTileM + kScaleColsPerTile ≤ 512.
template <typename Tin_, typename Tout_, typename Tsf_, int kTileM_, int kTileN_, int kTileK_,
          int kEpiTileM_, int kStage_, int kStageTMA_, int kMmaSM_ = 1, int kStageTile_ = 2,
          typename TinB_ = Tin_>
struct GroupGEMMMxFp8Config {
  using Tin = Tin_;
  using TinB = TinB_;  // B(weight) gmem element type: fp8 (== Tin) or fp4 (float_e2m1_unpacksmem_t)
  using Tout = Tout_;
  using Tsf = Tsf_;
  using Tacc = float;

  static constexpr int kTileM = kTileM_;
  static constexpr int kTileN = kTileN_;
  static constexpr int kTileK = kTileK_;
  static constexpr int kEpiTileM = kEpiTileM_;
  static constexpr int kStage = kStage_;
  static constexpr int kStageTMA = kStageTMA_;
  static constexpr int kStageTile = kStageTile_;

  static constexpr int kMmaSM = kMmaSM_;
  static constexpr int kCtaTileM = kTileM / kMmaSM;
  static constexpr int kCtaTileN = kTileN / kMmaSM;
  static constexpr int kClusterM = kMmaSM;
  static constexpr int kClusterN = 1;
  static constexpr int kClusterK = 1;
  static constexpr int kClusters = kClusterM * kClusterN * kClusterK;

  static constexpr int kSfVec = 32;
  static constexpr bool kSmallTM = (kTileM <= 128);
  static constexpr int kSfxRows = kSmallTM ? 32 : 64;
  static constexpr int kSFAlignM = kSmallTM ? 128 : 256;
  static constexpr int kScaleColsPerTile = kSmallTM ? 8 : 12;

  // B(weight) is sub-byte (fp4)?
  static constexpr bool kBSubByte = (cute::sizeof_bits_v<TinB> < 8);
  static constexpr bool kNeedPreZeroB = kBSubByte;

  static_assert(kMmaSM == 1 || kMmaSM == 2, "kMmaSM must be 1 or 2");
  static_assert(kTileN == 128 * kMmaSM, "kTileN must be 128 (1SM) or 256 (2SM)");
  static_assert(kTileM >= 8 * kMmaSM && kTileM <= 256 && kTileM % (8 * kMmaSM) == 0,
                "kTileM must be in [8*kMmaSM, 256] step 8*kMmaSM");
  static_assert(kTileK == 64 || kTileK == 128, "kTileK must be 64 or 128");
  static_assert(kStageTile >= 1 && kStageTile <= 4, "kStageTile must be in [1, 4]");
  static_assert(kStageTile * kTileM + kScaleColsPerTile <= 512,
                "TMEM exceeds 512 cols: reduce kStageTile, "
                "(formula: kStageTile * kTileM + kScaleColsPerTile)");

  // kABPerSF: how many AB K-tiles fit in one SF 128-K packed tile.
  // When kTileK=64, each SF load covers 2 AB tiles; when kTileK=128, it's 1:1.
  static constexpr int kABPerSF = 128 / kTileK;

  // AB SMEM atom: SW128 for kTileK=128 (128B = 128 fp8 elems), SW64 for kTileK=64.
  using SLayoutABAtom = std::conditional_t<kTileK == 128, UMMA::Layout_K_SW128_Atom<Tin>,
                                           UMMA::Layout_K_SW64_Atom<Tin>>;
  using SLayoutA = decltype(tile_to_shape(
      SLayoutABAtom{}, make_shape(Int<kCtaTileM>{}, Int<kTileK>{}, Int<kStage>{})));
  using SLayoutB = decltype(tile_to_shape(
      SLayoutABAtom{}, make_shape(Int<kCtaTileN>{}, Int<kTileK>{}, Int<kStage>{})));
  using SLayoutSFA = decltype(make_layout(make_shape(Int<kSfxRows>{}, Int<16>{}, Int<kStage>{}),
                                          make_stride(Int<16>{}, Int<1>{}, Int<kSfxRows * 16>{})));
  using SLayoutSFB = decltype(make_layout(make_shape(Int<32>{}, Int<16>{}, Int<kStage>{}),
                                          make_stride(Int<16>{}, Int<1>{}, Int<32 * 16>{})));

  using SLayoutYAtom =
      std::conditional_t<(kEpiTileM == 8), UMMA::Layout_K_INTER_Atom<Tout>,
                         std::conditional_t<(kEpiTileM == 16), UMMA::Layout_K_SW32_Atom<Tout>,
                                            UMMA::Layout_K_SW64_Atom<Tout>>>;
  using SLayoutY = decltype(tile_to_shape(
      SLayoutYAtom{}, make_shape(Int<kCtaTileN>{}, Int<kEpiTileM>{}, Int<kStageTMA>{})));
  using SLayoutYT =
      decltype(tile_to_shape(UMMA::Layout_MN_SW128_Atom<Tout>{},
                             make_shape(Int<kCtaTileN>{}, Int<kEpiTileM>{}, Int<kStageTMA>{})));

  using TiledMma = typename MxFp8MmaAtomSelector<kMmaSM, Tin, TinB, Tsf, kTileM, kTileN>::type;

  // cp.async copy for A (activation): used by group_gemm_cp_async_mxfp8 to
  // load X from gmem directly into SMEM (with optional row_map indirection),
  // bypassing the gather + TMA path. 16x8 thread grid × 16-elem vec = 16-row
  // × 128-col = (kCtaTileM up to 16 per warp-iter, kTileK=128) — matches the
  // fp8 cp_async kernel's geometry. Driven by 128 threads (warps 8-11).
  using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>;
  using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
  using g2s_copy_atom = Copy_Atom<g2s_copy_traits, Tin>;
  using G2SCopy = decltype(make_tiled_copy(
      g2s_copy_atom{},
      make_layout(make_shape(Int<16>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<16>{}))));

  // Epilogue atoms picked by kEpiTileM:
  //   kEpiTileM=8  -> 16dp256b1x + STSM_T_x2
  //   kEpiTileM=16 -> 16dp256b2x + STSM_T_x4
  //   kEpiTileM=32 -> 16dp256b4x + STSM_T_x8
  //   kEpiTileM=64 -> 16dp256b4x + STSM_T_x8 (atom stays b4x; TiledCopy auto-tiles)
  using TmemLoadAtom = std::conditional_t<
      kEpiTileM == 8, SM100_TMEM_LOAD_16dp256b1x,
      std::conditional_t<kEpiTileM == 16, SM100_TMEM_LOAD_16dp256b2x, SM100_TMEM_LOAD_16dp256b4x>>;
  using StsmTAtom =
      std::conditional_t<kEpiTileM == 8, SM90_U16x2_STSM_T,
                         std::conditional_t<kEpiTileM == 16, SM90_U16x4_STSM_T, SM90_U16x8_STSM_T>>;

  template <typename TX, typename TW, typename TY, typename TSFX, typename TSFW>
  auto get_tma(TX x, TW w, TY y, TSFX sfx, TSFW sfw) {
    if constexpr (kMmaSM == 2) {
      // cluster via the kMmaSM partition arg:
      //   * A / X (kTileM, kTileK):
      //       box = (kTileM, kTileK), 2SM_LOAD with kMmaSM partition.
      //   * B / W (kTileN, kTileK):
      //       box = (kTileN, kTileK), 2SM_LOAD with kMmaSM partition.
      //   * SFB (split on N):                box = (64, 16), 2SM_LOAD.
      //   * SFA (multicast, half-tile/issue): box = (32, 16), MULTICAST.
      auto copybox_a = tile_to_shape(SLayoutABAtom{}, make_shape(Int<kTileM>{}, Int<kTileK>{}));
      auto copybox_b = tile_to_shape(SLayoutABAtom{}, make_shape(Int<kTileN>{}, Int<kTileK>{}));
      auto copybox_sfb =
          make_layout(make_shape(Int<64>{}, Int<16>{}), make_stride(Int<16>{}, Int<1>{}));
      auto copybox_sfa_half =
          make_layout(make_shape(Int<32>{}, Int<16>{}), make_stride(Int<16>{}, Int<1>{}));
      auto tma_a = make_tma_copy(SM100_TMA_2SM_LOAD{}, x, copybox_a, Int<kMmaSM>{});
      auto tma_b = make_tma_copy(SM100_TMA_2SM_LOAD{}, w, copybox_b, Int<kMmaSM>{});
      auto tma_sfa = make_tma_copy(SM90_TMA_LOAD_MULTICAST{}, sfx, copybox_sfa_half);
      auto tma_sfb = make_tma_copy(SM100_TMA_2SM_LOAD{}, sfw, copybox_sfb, Int<kMmaSM>{});
      auto tma_y = make_tma_copy(SM90_TMA_STORE{}, y, SLayoutYT{}(_, _, 0));  // NOLINT
      return std::make_tuple(tma_a, tma_b, tma_y, tma_sfa, tma_sfb);
    } else {
      auto tma_a = make_tma_copy(SM90_TMA_LOAD{}, x, SLayoutA{}(_, _, 0));        // NOLINT
      auto tma_b = make_tma_copy(SM90_TMA_LOAD{}, w, SLayoutB{}(_, _, 0));        // NOLINT
      auto tma_y = make_tma_copy(SM90_TMA_STORE{}, y, SLayoutYT{}(_, _, 0));      // NOLINT
      auto tma_sfa = make_tma_copy(SM90_TMA_LOAD{}, sfx, SLayoutSFA{}(_, _, 0));  // NOLINT
      auto tma_sfb = make_tma_copy(SM90_TMA_LOAD{}, sfw, SLayoutSFB{}(_, _, 0));  // NOLINT
      return std::make_tuple(tma_a, tma_b, tma_y, tma_sfa, tma_sfb);
    }
  }

  // get_tma variant that skips SFA TMA construction (for cp_async kernels where
  // SFA is loaded via cp.async inline prepack, not TMA).
  template <typename TX, typename TW, typename TY, typename TSFW>
  auto get_tma_without_sfa(TX x, TW w, TY y, TSFW sfw) {
    if constexpr (kMmaSM == 2) {
      auto copybox_a = tile_to_shape(SLayoutABAtom{}, make_shape(Int<kTileM>{}, Int<kTileK>{}));
      auto copybox_b = tile_to_shape(SLayoutABAtom{}, make_shape(Int<kTileN>{}, Int<kTileK>{}));
      auto copybox_sfb =
          make_layout(make_shape(Int<64>{}, Int<16>{}), make_stride(Int<16>{}, Int<1>{}));
      auto tma_a = make_tma_copy(SM100_TMA_2SM_LOAD{}, x, copybox_a, Int<kMmaSM>{});
      auto tma_b = make_tma_copy(SM100_TMA_2SM_LOAD{}, w, copybox_b, Int<kMmaSM>{});
      auto tma_sfb = make_tma_copy(SM100_TMA_2SM_LOAD{}, sfw, copybox_sfb, Int<kMmaSM>{});
      auto tma_y = make_tma_copy(SM90_TMA_STORE{}, y, SLayoutYT{}(_, _, 0));  // NOLINT
      return std::make_tuple(tma_a, tma_b, tma_y, tma_sfb);
    } else {
      auto tma_a = make_tma_copy(SM90_TMA_LOAD{}, x, SLayoutA{}(_, _, 0));        // NOLINT
      auto tma_b = make_tma_copy(SM90_TMA_LOAD{}, w, SLayoutB{}(_, _, 0));        // NOLINT
      auto tma_y = make_tma_copy(SM90_TMA_STORE{}, y, SLayoutYT{}(_, _, 0));      // NOLINT
      auto tma_sfb = make_tma_copy(SM90_TMA_LOAD{}, sfw, SLayoutSFB{}(_, _, 0));  // NOLINT
      return std::make_tuple(tma_a, tma_b, tma_y, tma_sfb);
    }
  }

  static constexpr int shm_ab = (cosize(SLayoutA{}) + cosize(SLayoutB{})) * sizeof(Tin);
  static constexpr int shm_sf = (cosize(SLayoutSFA{}) + cosize(SLayoutSFB{})) * sizeof(Tsf);
  static constexpr int shm_y = cosize(SLayoutYT{}) * sizeof(Tout);
  static constexpr int shm_size = shm_ab + shm_sf + shm_y;

  static_assert(shm_size <= 228 * 1024,
                "Per-CTA dynamic SHM exceeds 228 KB: reduce kStage / kStageTMA / kEpiTileM");

  // TMA transaction bytes (per AB stage):
  static constexpr uint32_t kExpectedBytesA = (cosize(SLayoutA{}) / kStage) * sizeof(Tin);
  static constexpr uint32_t kExpectedBytesB = kCtaTileN * kTileK * cute::sizeof_bits_v<TinB> / 8;
  static constexpr uint32_t kExpectedBytesAB = kExpectedBytesA + kExpectedBytesB;

  auto get_shm_size() { return shm_size; }
};

}  // namespace group_gemm
}  // namespace hpc

#endif  // SRC_GROUP_GEMM_SM100_MXFP8_CONFIG_H_
