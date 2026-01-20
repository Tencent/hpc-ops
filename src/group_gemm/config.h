// Copyright (C) 2026 Tencent.

#ifndef SRC_GROUP_GEMM_CONFIG_H_
#define SRC_GROUP_GEMM_CONFIG_H_

#include "cute/tensor.hpp"

namespace hpc {
namespace group_gemm {

using namespace cute;  // NOLINT

template <int kSwizzle, typename T, bool kKmajor = true>
static constexpr auto slayout_selector() {
  if constexpr (kSwizzle == 128) {
    if constexpr (kKmajor) {
      return cute::GMMA::Layout_K_SW128_Atom<T>{};
    } else {
      return cute::GMMA::Layout_MN_SW128_Atom<T>{};
    }
  } else if constexpr (kSwizzle == 64) {
    if constexpr (kKmajor) {
      return cute::GMMA::Layout_K_SW64_Atom<T>{};
    } else {
      return cute::GMMA::Layout_MN_SW64_Atom<T>{};
    }
  } else if constexpr (kSwizzle == 32) {
    if constexpr (kKmajor) {
      return cute::GMMA::Layout_K_SW32_Atom<T>{};
    } else {
      return cute::GMMA::Layout_MN_SW32_Atom<T>{};
    }
  } else {
    if constexpr (kKmajor) {
      return cute::GMMA::Layout_K_INTER_Atom<T>{};
    } else {
      return cute::GMMA::Layout_MN_INTER_Atom<T>{};
    }
  }
}

template <int kTileM>
static constexpr auto mma_selector() {
  if constexpr (kTileM == 8) {
    return cute::SM90_64x8x32_F32E4M3E4M3_SS_TN<>{};
  } else if constexpr (kTileM == 16) {
    return cute::SM90_64x16x32_F32E4M3E4M3_SS_TN<>{};
  } else if constexpr (kTileM == 32) {
    return cute::SM90_64x32x32_F32E4M3E4M3_SS_TN<>{};
  } else if constexpr (kTileM == 48) {
    return cute::SM90_64x48x32_F32E4M3E4M3_SS_TN<>{};
  } else if constexpr (kTileM == 64) {
    return cute::SM90_64x64x32_F32E4M3E4M3_SS_TN<>{};
  } else if constexpr (kTileM == 96) {
    return cute::SM90_64x96x32_F32E4M3E4M3_SS_TN<>{};
  } else if constexpr (kTileM == 128) {
    return cute::SM90_64x128x32_F32E4M3E4M3_SS_TN<>{};
  } else {
    return cute::SM90_64x64x32_F32E4M3E4M3_SS_TN<>{};
  }
}

template <typename Tin_, typename Tout_, int kTileM_, int kTileN_, int kTileK_, int kStage_,
          int kWarpgroupM_ = 2, int kWarpgroupN_ = 1, int kSwizzleX = 128, int kSwizzleW = 128,
          int kSwizzleY = 128>
struct GroupGEMMFp8Config {
  using Tin = Tin_;
  using Tout = Tout_;

  static constexpr int kTileM = kTileM_;
  static constexpr int kTileN = kTileN_;
  static constexpr int kTileK = kTileK_;
  static constexpr int kStage = kStage_;
  static constexpr int kWarpgroupM = kWarpgroupM_;
  static constexpr int kWarpgroupN = kWarpgroupN_;

  using SLayoutXAtom = decltype(slayout_selector<kSwizzleX, Tin>());
  using SLayoutWAtom = decltype(slayout_selector<kSwizzleW, Tin>());
  using SLayoutYAtom = decltype(slayout_selector<kSwizzleY, Tout, false>());

  using SLayoutX = decltype(tile_to_shape(SLayoutXAtom{},
                                          make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{})));
  using SLayoutW = decltype(tile_to_shape(SLayoutWAtom{},
                                          make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{})));
  using SLayoutY =
      decltype(tile_to_shape(SLayoutYAtom{}, make_shape(Int<kTileN>{}, Int<kTileM>{})));
  using CopyBoxY = decltype(tile_to_shape(SLayoutYAtom{},
                                          make_shape(Int<kTileN / kWarpgroupM>{}, Int<kTileM>{})));

  template <typename TX, typename TW, typename TY>
  auto get_tma(TX x, TW w, TY y) {
    auto tma_x = make_tma_copy(SM90_TMA_LOAD{}, x, take<0, 2>(SLayoutX{}));
    auto tma_w = make_tma_copy(SM90_TMA_LOAD{}, w, take<0, 2>(SLayoutW{}));
    auto tma_y = make_tma_copy(SM90_TMA_STORE{}, y, CopyBoxY{});
    return std::make_tuple(tma_x, tma_w, tma_y);
  }

  using WarpgroupLayout =
      decltype(make_layout(make_shape(Int<kWarpgroupM>{}, Int<kWarpgroupN>{}, Int<1>{})));
  using TiledMma = decltype(make_tiled_mma(mma_selector<kTileM>(), WarpgroupLayout{}));

  static constexpr int shm_xw = (cosize(SLayoutX{}) + cosize(SLayoutW{})) * sizeof(Tin);
  static constexpr int shm_y = cosize(SLayoutY{}) * sizeof(Tout);
  static constexpr int shm_size = shm_xw + shm_y;

  auto get_shm_size() { return shm_size; }
};

template <typename Tin_, typename Tout_, typename TS_, int kTileM_, int kTileN_, int kTileK_,
          int kTileS_, int kStage_, int kWarpgroupM_ = 2, int kWarpgroupN_ = 1, int kSwizzleX = 128,
          int kSwizzleW = 128, int kSwizzleY = 128>
struct GroupGEMMBlockWiseFp8Config {
  using Tin = Tin_;
  using Tout = Tout_;
  using TS = TS_;

  static constexpr int kTileM = kTileM_;
  static constexpr int kTileN = kTileN_;
  static constexpr int kTileK = kTileK_;
  static constexpr int kTileS = kTileS_;
  static constexpr int kStage = kStage_;
  static constexpr int kWarpgroupM = kWarpgroupM_;
  static constexpr int kWarpgroupN = kWarpgroupN_;

  using SLayoutXAtom = decltype(slayout_selector<kSwizzleX, Tin>());
  using SLayoutWAtom = decltype(slayout_selector<kSwizzleW, Tin>());
  using SLayoutYAtom = decltype(slayout_selector<kSwizzleY, Tout, false>());

  using SLayoutX = decltype(tile_to_shape(SLayoutXAtom{},
                                          make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{})));
  using SLayoutW = decltype(tile_to_shape(SLayoutWAtom{},
                                          make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{})));
  using SLayoutY =
      decltype(tile_to_shape(SLayoutYAtom{}, make_shape(Int<kTileN>{}, Int<kTileM>{})));
  using SLayoutXS = decltype(make_layout(make_shape(Int<kStage>{}, Int<kTileS>{}),
                                         make_stride(Int<kTileS>{}, Int<1>{})));
  using SLayoutWS = decltype(make_layout(make_shape(Int<kStage>{}, Int<kTileS>{}),
                                         make_stride(Int<kTileS>{}, Int<1>{})));
  using CopyBoxY = decltype(tile_to_shape(SLayoutYAtom{},
                                          make_shape(Int<kTileN / kWarpgroupM>{}, Int<kTileM>{})));
  using CopyBoxXS = decltype(make_layout(make_shape(Int<1>{}, Int<kTileM>{}),
                                         make_stride(Int<kTileM>{}, Int<1>{})));
  using CopyBoxWS =
      decltype(make_layout(make_shape(Int<1>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})));

  template <typename TX, typename TW, typename TY, typename TXS, typename TWS>
  auto get_tma(TX x, TW w, TY y, TXS xs, TWS ws) {
    auto tma_x = make_tma_copy(SM90_TMA_LOAD{}, x, take<0, 2>(SLayoutX{}));
    auto tma_w = make_tma_copy(SM90_TMA_LOAD{}, w, take<0, 2>(SLayoutW{}));
    auto tma_y = make_tma_copy(SM90_TMA_STORE{}, y, CopyBoxY{});
    auto tma_xs = make_tma_copy(SM90_TMA_LOAD{}, xs, CopyBoxXS{});
    auto tma_ws = make_tma_copy(SM90_TMA_LOAD{}, ws, CopyBoxWS{});
    return std::make_tuple(tma_x, tma_w, tma_y, tma_xs, tma_ws);
  }

  using WarpgroupLayout =
      decltype(make_layout(make_shape(Int<kWarpgroupM>{}, Int<kWarpgroupN>{}, Int<1>{})));
  using TiledMma = decltype(make_tiled_mma(mma_selector<kTileM>(), WarpgroupLayout{}));

  static constexpr int shm_xw = (cosize(SLayoutX{}) + cosize(SLayoutW{})) * sizeof(Tin);
  static constexpr int shm_y = cosize(SLayoutY{}) * sizeof(Tout);
  static constexpr int shm_xws = (cosize(SLayoutXS{}) + cosize(SLayoutWS{})) * sizeof(TS);
  static constexpr int shm_size = shm_xw + shm_y + shm_xws;

  auto get_shm_size() { return shm_size; }
};

}  // namespace group_gemm
}  // namespace hpc

#endif  // SRC_GROUP_GEMM_CONFIG_H_
