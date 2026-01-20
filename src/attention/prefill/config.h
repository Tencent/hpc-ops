// Copyright (C) 2026 Tencent.

#ifndef SRC_ATTENTION_PREFILL_CONFIG_H_
#define SRC_ATTENTION_PREFILL_CONFIG_H_

#include "cute/tensor.hpp"

namespace hpc {
namespace attention {
namespace prefill {

using namespace cute;  // NOLINT

template <typename Tin_, typename Tout_, typename TiledMmaQKAtom, typename TiledMmaPVAtom,
          int kTileM_, int kTileN_, int kTileK_, int kTileV_, int kStage_, int kWarpgroupM_ = 2,
          int kWarpgroupN_ = 1, int kSwizzleQ = 128, int kSwizzleK = 128, int kSwizzleV = 128,
          int kSwizzleY = 128>
struct AttentionPrefillConfig {
  using Tin = Tin_;
  using Tout = Tout_;

  static constexpr int kTileM = kTileM_;
  static constexpr int kTileN = kTileN_;
  static constexpr int kTileK = kTileK_;
  static constexpr int kTileV = kTileV_;
  static constexpr int kStage = kStage_;
  static constexpr int kWarpgroupM = kWarpgroupM_;

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

  using SLayoutQAtom = decltype(slayout_selector<kSwizzleQ, Tin>());
  using SLayoutKAtom = decltype(slayout_selector<kSwizzleK, Tin>());
  using SLayoutVAtom = decltype(slayout_selector<kSwizzleV, Tin, false>());
  using SLayoutYAtom = decltype(slayout_selector<kSwizzleY, Tout>());

  using SLayoutQ =
      decltype(tile_to_shape(SLayoutQAtom{}, make_shape(Int<kTileM>{}, Int<kTileK>{})));
  using SLayoutK = decltype(tile_to_shape(SLayoutKAtom{},
                                          make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{})));
  using SLayoutV = decltype(tile_to_shape(SLayoutVAtom{},
                                          make_shape(Int<kTileV>{}, Int<kTileN>{}, Int<kStage>{})));
  using SLayoutY =
      decltype(tile_to_shape(SLayoutYAtom{}, make_shape(Int<kTileM>{}, Int<kTileV>{})));
  using CopyBoxY = decltype(tile_to_shape(SLayoutYAtom{},
                                          make_shape(Int<kTileM / kWarpgroupM>{}, Int<kTileV>{})));

  template <typename TQ, typename TK, typename TV, typename TY>
  auto get_tma(TQ q, TK k, TV v, TY y) {
    auto tma_q = make_tma_copy(SM90_TMA_LOAD{}, q, SLayoutQ{});
    auto tma_k = make_tma_copy(SM90_TMA_LOAD{}, k, take<0, 2>(SLayoutK{}));
    auto tma_v = make_tma_copy(SM90_TMA_LOAD{}, v, take<0, 2>(SLayoutV{}));
    auto tma_y = make_tma_copy(SM90_TMA_STORE{}, y, CopyBoxY{});
    return std::make_tuple(tma_q, tma_k, tma_v, tma_y);
  }

  using WarpgroupLayout =
      decltype(make_layout(make_shape(Int<kWarpgroupM_>{}, Int<kWarpgroupN_>{}, Int<1>{})));
  using TiledMmaQK = decltype(make_tiled_mma(TiledMmaQKAtom{}, WarpgroupLayout{}));
  using TiledMmaPV = decltype(make_tiled_mma(TiledMmaPVAtom{}, WarpgroupLayout{}));

  static constexpr int shm_qkv =
      (cosize(SLayoutQ{}) + cosize(SLayoutK{}) + cosize(SLayoutV{})) * sizeof(Tin);
  static constexpr int shm_y = cosize(SLayoutY{}) * sizeof(Tout);
  static constexpr int shm_size = shm_qkv + shm_y;

  auto get_shm_size() { return shm_size; }
};

template <typename Tin_, typename Tout_, typename TiledMmaQKAtom, typename TiledMmaPVAtom,
          int kTileM_, int kTileN_, int kTileK_, int kTileV_, int kBlockSize_, int kStage_,
          int kWarpgroupM_ = 2, int kWarpgroupN_ = 1, int kSwizzleQ = 128, int kSwizzleK = 128,
          int kSwizzleV = 128, int kSwizzleY = 128>
struct AttentionKVCachePrefillConfig {
  using Tin = Tin_;
  using Tout = Tout_;

  static constexpr int kTileM = kTileM_;
  static constexpr int kTileN = kTileN_;
  static constexpr int kTileK = kTileK_;
  static constexpr int kTileV = kTileV_;
  static constexpr int kBlockSize = kBlockSize_;
  static constexpr int kStage = kStage_;
  static constexpr int kWarpgroupM = kWarpgroupM_;

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

  using SLayoutQAtom = decltype(slayout_selector<kSwizzleQ, Tin>());
  using SLayoutKAtom = decltype(slayout_selector<kSwizzleK, Tin>());
  using SLayoutVAtom = decltype(slayout_selector<kSwizzleV, Tin, false>());
  using SLayoutYAtom = decltype(slayout_selector<kSwizzleY, Tout>());

  using SLayoutQ =
      decltype(tile_to_shape(SLayoutQAtom{}, make_shape(Int<kTileM>{}, Int<kTileK>{})));
  using SLayoutK = decltype(tile_to_shape(SLayoutKAtom{},
                                          make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{})));
  using SLayoutV = decltype(tile_to_shape(SLayoutVAtom{},
                                          make_shape(Int<kTileV>{}, Int<kTileN>{}, Int<kStage>{})));
  using SLayoutY =
      decltype(tile_to_shape(SLayoutYAtom{}, make_shape(Int<kTileM>{}, Int<kTileV>{})));

  using CopyBoxK =
      decltype(tile_to_shape(SLayoutKAtom{}, make_shape(Int<kBlockSize>{}, Int<kTileK>{})));
  using CopyBoxV =
      decltype(tile_to_shape(SLayoutVAtom{}, make_shape(Int<kTileV>{}, Int<kBlockSize>{})));
  using CopyBoxY = decltype(tile_to_shape(SLayoutYAtom{},
                                          make_shape(Int<kTileM / kWarpgroupM>{}, Int<kTileV>{})));

  template <typename TQ, typename TK, typename TV, typename TY>
  auto get_tma(TQ q, TK k, TV v, TY y) {
    auto tma_q = make_tma_copy(SM90_TMA_LOAD{}, q, SLayoutQ{});
    auto tma_k = make_tma_copy(SM90_TMA_LOAD{}, k, CopyBoxK{});
    auto tma_v = make_tma_copy(SM90_TMA_LOAD{}, v, CopyBoxV{});
    auto tma_y = make_tma_copy(SM90_TMA_STORE{}, y, CopyBoxY{});
    return std::make_tuple(tma_q, tma_k, tma_v, tma_y);
  }

  using WarpgroupLayout =
      decltype(make_layout(make_shape(Int<kWarpgroupM_>{}, Int<kWarpgroupN_>{}, Int<1>{})));
  using TiledMmaQK = decltype(make_tiled_mma(TiledMmaQKAtom{}, WarpgroupLayout{}));
  using TiledMmaPV = decltype(make_tiled_mma(TiledMmaPVAtom{}, WarpgroupLayout{}));

  static constexpr int shm_qkv =
      (cosize(SLayoutQ{}) + cosize(SLayoutK{}) + cosize(SLayoutV{})) * sizeof(Tin);
  static constexpr int shm_y = cosize(SLayoutY{}) * sizeof(Tout);
  static constexpr int shm_size = shm_qkv + shm_y;

  auto get_shm_size() { return shm_size; }
};

template <typename Tin_, typename Tout_, typename TiledMmaQKAtom, typename TiledMmaPVAtom,
          int kTileM_, int kTileN_, int kTileK_, int kTileV_, int kBlockSize_, int kStage_,
          int kWarpgroupM_ = 2, int kWarpgroupN_ = 1, int kSwizzleQ = 128, int kSwizzleK = 128,
          int kSwizzleV = 128, int kSwizzleY = 128>
struct AttentionKVCachePrefillFp8Config {
  using Tin = Tin_;
  using Tout = Tout_;

  static constexpr int kTileM = kTileM_;
  static constexpr int kTileN = kTileN_;
  static constexpr int kTileK = kTileK_;
  static constexpr int kTileV = kTileV_;
  static constexpr int kBlockSize = kBlockSize_;
  static constexpr int kStage = kStage_;
  static constexpr int kWarpgroupM = kWarpgroupM_;

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

  using SLayoutQAtom = decltype(slayout_selector<kSwizzleQ, Tin>());
  using SLayoutKAtom = decltype(slayout_selector<kSwizzleK, Tin>());
  using SLayoutVAtom = decltype(slayout_selector<kSwizzleV, Tin, false>());
  using SLayoutVTAtom = decltype(slayout_selector<kSwizzleV, Tin>());
  using SLayoutYAtom = decltype(slayout_selector<kSwizzleY, Tout>());

  using SLayoutQ =
      decltype(tile_to_shape(SLayoutQAtom{}, make_shape(Int<kTileM>{}, Int<kTileK>{})));
  using SLayoutK = decltype(tile_to_shape(SLayoutKAtom{},
                                          make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{})));
  using SLayoutV = decltype(tile_to_shape(SLayoutVAtom{},
                                          make_shape(Int<kTileV>{}, Int<kTileN>{}, Int<kStage>{})));
  using SLayoutVT = decltype(tile_to_shape(
      SLayoutVTAtom{}, make_shape(Int<kTileV>{}, Int<kTileN>{}, Int<kStage * kWarpgroupM>{})));
  using SLayoutY =
      decltype(tile_to_shape(SLayoutYAtom{}, make_shape(Int<kTileM>{}, Int<kTileV>{})));
  using SLayoutQKS = decltype(make_layout(make_shape(Int<kTileM>{}), make_stride(Int<1>{})));

  using CopyBoxK =
      decltype(tile_to_shape(SLayoutKAtom{}, make_shape(Int<kBlockSize>{}, Int<kTileK>{})));
  using CopyBoxV =
      decltype(tile_to_shape(SLayoutVAtom{}, make_shape(Int<kTileV>{}, Int<kBlockSize>{})));
  using CopyBoxY = decltype(tile_to_shape(SLayoutYAtom{},
                                          make_shape(Int<kTileM / kWarpgroupM>{}, Int<kTileV>{})));

  template <typename TQ, typename TK, typename TV, typename TY, typename TQKS>
  auto get_tma(TQ q, TK k, TV v, TY y, TQKS qks) {
    auto tma_q = make_tma_copy(SM90_TMA_LOAD{}, q, SLayoutQ{});
    auto tma_k = make_tma_copy(SM90_TMA_LOAD{}, k, CopyBoxK{});
    auto tma_v = make_tma_copy(SM90_TMA_LOAD{}, v, CopyBoxV{});
    auto tma_y = make_tma_copy(SM90_TMA_STORE{}, y, CopyBoxY{});
    auto tma_qks = make_tma_copy(SM90_TMA_LOAD{}, qks, SLayoutQKS{});
    return std::make_tuple(tma_q, tma_k, tma_v, tma_y, tma_qks);
  }

  using WarpgroupLayout =
      decltype(make_layout(make_shape(Int<kWarpgroupM_>{}, Int<kWarpgroupN_>{}, Int<1>{})));
  using TiledMmaQK = decltype(make_tiled_mma(TiledMmaQKAtom{}, WarpgroupLayout{}));
  using TiledMmaPV = decltype(make_tiled_mma(TiledMmaPVAtom{}, WarpgroupLayout{}));

  static constexpr int shm_qkv =
      (cosize(SLayoutQ{}) + cosize(SLayoutK{}) + cosize(SLayoutV{}) + cosize(SLayoutVT{})) *
      sizeof(Tin);
  static constexpr int shm_y = cosize(SLayoutY{}) * sizeof(Tout);
  static constexpr int shm_qks = cosize(SLayoutQKS{}) * sizeof(float);
  static constexpr int shm_size = shm_qkv + shm_y + shm_qks;

  auto get_shm_size() { return shm_size; }
};

}  // namespace prefill
}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_PREFILL_CONFIG_H_
