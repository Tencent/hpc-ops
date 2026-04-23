// Copyright 2025 hpc-ops authors

#ifndef SRC_GROUP_GEMM_SM100_CONFIG_H_
#define SRC_GROUP_GEMM_SM100_CONFIG_H_

#include "cute/tensor.hpp"

namespace hpc {
namespace group_gemm {

using namespace cute;  // NOLINT

template <int kSwizzle, typename T, bool kKmajor = true>
static constexpr auto slayout_selector() {
  if constexpr (kSwizzle == 128) {
    if constexpr (kKmajor) {
      return cute::UMMA::Layout_K_SW128_Atom<T>{};
    } else {
      return cute::UMMA::Layout_MN_SW128_Atom<T>{};
    }
  } else if constexpr (kSwizzle == 64) {
    if constexpr (kKmajor) {
      return cute::UMMA::Layout_K_SW64_Atom<T>{};
    } else {
      return cute::UMMA::Layout_MN_SW64_Atom<T>{};
    }
  } else if constexpr (kSwizzle == 32) {
    if constexpr (kKmajor) {
      return cute::UMMA::Layout_K_SW32_Atom<T>{};
    } else {
      return cute::UMMA::Layout_MN_SW32_Atom<T>{};
    }
  } else {
    if constexpr (kKmajor) {
      return cute::UMMA::Layout_K_INTER_Atom<T>{};
    } else {
      return cute::UMMA::Layout_MN_INTER_Atom<T>{};
    }
  }
}

template <int kMmaSM>
static constexpr auto mma_selector() {
  if constexpr (kMmaSM == 2) {
    return cute::SM100_MMA_F8F6F4_2x1SM_SS{};
  } else if constexpr (kMmaSM == 1) {
    return cute::SM100_MMA_F8F6F4_SS{};
  }
}

template <typename Tin_, typename Tout_, typename Tacc_, int kTileM_, int kTileN_, int kTileK_,
          int kStage_, int kClusterM_ = 2, int kClusterN_ = 1, int kClusterK_ = 1, int kMmaSM_ = 2,
          int kEpiTileN_ = 32, int kStageTile_ = 4, int kStageTMA_ = 4, int kSwizzleX_ = 128,
          int kSwizzleW_ = 128, int kSwizzleY_ = 128, int kSwizzleYInner_ = 32>
struct GroupGEMMFp8Config {
  using Tin = Tin_;
  using Tout = Tout_;
  using Tacc = Tacc_;

  static constexpr int kTileM = kTileM_;
  static constexpr int kTileN = kTileN_;
  static constexpr int kTileK = kTileK_;
  static constexpr int kStage = kStage_;
  static constexpr int kClusterM = kClusterM_;
  static constexpr int kClusterN = kClusterN_;
  static constexpr int kClusterK = kClusterK_;
  static constexpr int kSwizzleX = kSwizzleX_;
  static constexpr int kSwizzleW = kSwizzleW_;
  static constexpr int kSwizzleY = kSwizzleY_;
  static constexpr int kMmaSM = kMmaSM_;
  static constexpr int kEpiTileN = kEpiTileN_;
  static constexpr int kStageTile = kStageTile_;
  static constexpr int kStageTMA = kStageTMA_;

  static constexpr int kCtaTileM = kTileM / kMmaSM;
  static constexpr int kCtaTileN = kTileN / kMmaSM;
  static constexpr int kClusters = kClusterM * kClusterN * kClusterK;

  using SLayoutXAtom = decltype(slayout_selector<kSwizzleX, Tin>());
  using SLayoutWAtom = decltype(slayout_selector<kSwizzleW, Tin>());
  using SLayoutYAtom = decltype(slayout_selector<kSwizzleYInner_, Tout>());
  using SLayoutYTAtom = decltype(slayout_selector<kSwizzleY, Tout, false>());

  using SLayoutX = decltype(tile_to_shape(
      SLayoutXAtom{}, make_shape(Int<kCtaTileM>{}, Int<kTileK>{}, Int<kStage>{})));
  using SLayoutW = decltype(tile_to_shape(
      SLayoutWAtom{}, make_shape(Int<kCtaTileN>{}, Int<kTileK>{}, Int<kStage>{})));
  using SLayoutY = decltype(tile_to_shape(
      SLayoutYAtom{}, make_shape(Int<kCtaTileM>{}, Int<kEpiTileN>{}, Int<kStageTMA>{})));
  using SLayoutYT = decltype(tile_to_shape(
      SLayoutYTAtom{}, make_shape(Int<kCtaTileM>{}, Int<kEpiTileN>{}, Int<kStageTMA>{})));

  using CopyBoxX =
      decltype(tile_to_shape(SLayoutXAtom{}, make_shape(Int<kTileM>{}, Int<kTileK>{})));
  using CopyBoxW =
      decltype(tile_to_shape(SLayoutWAtom{}, make_shape(Int<kTileN>{}, Int<kTileK>{})));
  using CopyBoxYT =
      decltype(tile_to_shape(SLayoutYTAtom{}, make_shape(Int<kCtaTileM>{}, Int<kEpiTileN>{})));

  // copy A using cp_async
  using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>;
  using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
  using g2s_copy_atom = Copy_Atom<g2s_copy_traits, Tin>;
  using G2SCopy = decltype(make_tiled_copy(
      g2s_copy_atom{},
      make_layout(make_shape(Int<16>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<16>{}))));

  template <typename TX, typename TW, typename TY>
  auto get_tma(TX x, TW w, TY y) {
    if constexpr (kMmaSM == 2) {
      auto tma_x = make_tma_copy(SM100_TMA_2SM_LOAD{}, x, CopyBoxX{}, Int<kClusterN * kMmaSM>{});
      auto tma_w = make_tma_copy(SM100_TMA_2SM_LOAD{}, w, CopyBoxW{}, Int<kClusterM>{});
      auto tma_y = make_tma_copy(SM90_TMA_STORE{}, y, CopyBoxYT{});
      // auto tma_y = make_tma_copy(SM90_TMA_REDUCE_ADD{}, y, CopyBoxYT{});
      return std::make_tuple(tma_x, tma_w, tma_y);
    } else if constexpr (kMmaSM == 1) {
      auto tma_x = make_tma_copy(SM90_TMA_LOAD{}, x, CopyBoxX{});
      auto tma_w = make_tma_copy(SM90_TMA_LOAD{}, w, CopyBoxW{});
      auto tma_y = make_tma_copy(SM90_TMA_STORE{}, y, CopyBoxYT{});
      // auto tma_y = make_tma_copy(SM90_TMA_REDUCE_ADD{}, y, CopyBoxYT{});
      return std::make_tuple(tma_x, tma_w, tma_y);
    }
  }

  static constexpr int shm_xw = (cosize(SLayoutX{}) + cosize(SLayoutW{})) * sizeof(Tin);
  static constexpr int shm_y = cosize(SLayoutY{}) * sizeof(Tout);
  static constexpr int shm_size = shm_xw + shm_y;

  static_assert((shm_size < 228 * 1024), "shared memory request is large than 228KB");

  auto get_shm_size() { return shm_size; }
};

}  // namespace group_gemm
}  // namespace hpc

#endif  // SRC_GROUP_GEMM_SM100_CONFIG_H_
