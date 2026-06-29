// Copyright 2025 hpc-ops authors

#ifndef SRC_SAGE_ATTENTION_SM90_SAGE_ATTN_INT8_CONFIG_H_
#define SRC_SAGE_ATTENTION_SM90_SAGE_ATTN_INT8_CONFIG_H_

#include "cute/tensor.hpp"

namespace hpc {
namespace sage_attention {

using namespace cute;  // NOLINT

template <typename TiledMmaQKAtom_, typename TiledMmaPVAtom_, int kTileM_, int kTileN_, int kTileK_,
          int kTileV_, int kStage_, int kWarpgroupM_ = 1, int kWarpgroupN_ = 1, int kSwizzleQ = 128,
          int kSwizzleK = 128, int kSwizzleV = 128, int kSwizzleY = 128>
struct SageAttentionInt8Sm90Config {
  using TinQK = int8_t;
  using TinV = cute::float_e4m3_t;
  using Tout = cute::bfloat16_t;

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

  // Q, K: int8, K-major swizzle
  using SLayoutQAtom = decltype(slayout_selector<kSwizzleQ, TinQK>());
  using SLayoutKAtom = decltype(slayout_selector<kSwizzleK, TinQK>());
  // V: fp8, K-major swizzle (data is already transposed in global memory)
  using SLayoutVAtom = decltype(slayout_selector<kSwizzleV, TinV>());
  // Y: bf16 output
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
  using TiledMmaQK = decltype(make_tiled_mma(TiledMmaQKAtom_{}, WarpgroupLayout{}));
  using TiledMmaPV = decltype(make_tiled_mma(TiledMmaPVAtom_{}, WarpgroupLayout{}));

  static constexpr int shm_size_q = cosize(SLayoutQ{}) * sizeof(TinQK);
  static constexpr int shm_size_k = cosize(SLayoutK{}) * sizeof(TinQK);
  static constexpr int shm_size_v = cosize(SLayoutV{}) * sizeof(TinV);
  static constexpr int shm_size_y = cosize(SLayoutY{}) * sizeof(Tout);
  // The Y output buffer aliases the Q/K/V region (reused in the epilogue), so
  // the arena only needs to be as large as the larger of the two.  This saves
  // the dedicated 16KB output region and raises occupancy from 2 to 3 CTAs/SM.
  static constexpr int shm_size_qkv = shm_size_q + shm_size_k + shm_size_v;
  static constexpr int shm_size = shm_size_qkv > shm_size_y ? shm_size_qkv : shm_size_y;

  auto get_shm_size() { return shm_size; }
};

}  // namespace sage_attention
}  // namespace hpc

#endif  // SRC_SAGE_ATTENTION_SM90_SAGE_ATTN_INT8_CONFIG_H_
