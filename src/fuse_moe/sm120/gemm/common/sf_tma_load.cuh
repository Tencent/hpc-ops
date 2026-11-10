// Copyright (C) 2026 Tencent.

#pragma once

// clang-format off
#include <cutlass/numeric_size.h>

#include <cstdint>

#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/copy_sm90.hpp>
#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/layout.hpp>

#include "src/fuse_moe/sm120/common/utils.h"
// clang-format on

namespace hpc::fuse_moe::sm120::gemm {
namespace common {

using namespace cute;  // NOLINT

// Packed UE8M0 scales as every launch uses them: one scale per 32 K elements,
// four of them per int32 word, so one word covers a 128-wide K tile and the SF
// ring advances with the AB ring.
template <int TileM_, int TileN_, int Stages_>
struct SfLoadConfig {
  using ElementSFLoad = int32_t;
  using ElementSFCompute = cute::float_ue8m0_t;

  static constexpr int kTileM = TileM_;
  static constexpr int kTileN = TileN_;
  static constexpr int kTileK = 128;
  static constexpr int kGranK = 32;
  static constexpr int kStageSF = Stages_;

  static constexpr int SFVecSize = kGranK;
  static constexpr int PACK_NSF = sizeof(ElementSFLoad) / sizeof(ElementSFCompute);
  static constexpr int PACK_NK = kGranK * PACK_NSF;
  static constexpr int kNumSFPerTileK = kTileK / kGranK;
  static constexpr int kTileSF = 1;
  static_assert(PACK_NK == kTileK, "one packed scale word per K tile");

  using ScaleTileShape = Shape<Int<kTileM>, Int<kTileN>, Int<kTileSF>>;
  using ProblemShape = Shape<int, int, int, int>;

  // Extents are int throughout: M is the plan's max_padded_rows, an int.
  CUTE_HOST_DEVICE
  static auto get_tma_aligned_size(const int& x) {
    constexpr int kNumTMAAlignmentBytes = 16;
    CUTE_STATIC_ASSERT(kNumTMAAlignmentBytes % sizeof(ElementSFLoad) == 0,
                       "element_size must be a multiple of 16");
    auto alignment = kNumTMAAlignmentBytes / sizeof(ElementSFLoad);
    return cute::round_up(x, static_cast<int>(alignment));
  }

  CUTE_HOST_DEVICE
  static auto deduce_sfa_layout(ProblemShape const& problem_shape, int num_experts_for_sf = 0) {
    auto [M, N, K, L] = problem_shape;
    constexpr int pack_nk = PACK_NK;
    int64_t scale_m =
        static_cast<int64_t>(get_tma_aligned_size(packed_scale_expert_base(M, num_experts_for_sf)));
    int64_t scale_k = static_cast<int64_t>(cute::ceil_div(K, pack_nk));
    return make_layout(make_shape(scale_m, scale_k, L),
                       make_stride(Int<1>{}, scale_m, scale_m * scale_k));
  }

  CUTE_HOST_DEVICE
  static auto deduce_sfb_layout(ProblemShape const& problem_shape, int = 0) {
    auto [M, N, K, L] = problem_shape;
    constexpr int pack_nk = PACK_NK;
    int64_t scale_n = static_cast<int64_t>(get_tma_aligned_size(N));
    int64_t scale_k = static_cast<int64_t>(cute::ceil_div(K, pack_nk));
    return make_layout(make_shape(scale_n, scale_k, L),
                       make_stride(Int<1>{}, scale_n, scale_n * scale_k));
  }

  template <class SFATensor, class Atom, class TiledThr, class TiledPerm>
  CUTE_HOST_DEVICE static constexpr auto thrfrg_SFA(SFATensor&& sfatensor,
                                                    TiledMMA<Atom, TiledThr, TiledPerm>& mma) {
    CUTE_STATIC_ASSERT_V(rank(sfatensor) >= Int<2>{});

    auto permutation_mnk = TiledPerm{};
    auto t_tile = make_tile(get<0>(permutation_mnk), _1{});
    auto tiled_sfa = logical_divide(sfatensor, t_tile);

    using AtomShape_MNK = typename Atom::Shape_MNK;
    auto atom_tile = make_tile(make_layout(size<0>(AtomShape_MNK{})), make_layout(_1{}));
    auto tiled_atom_sfa = zipped_divide(tiled_sfa, atom_tile);
    using AtomLayoutSFA_TV = Layout<Shape<Shape<_2, _2, _8>, _1>, Stride<Stride<_8, _0, _1>, _16>>;
    auto tv_atom_sfa = tiled_atom_sfa.compose(AtomLayoutSFA_TV{}, _);

    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
    auto thr_tile = make_tile(
        _, make_tile(make_layout(size<1>(thr_layout_vmnk)), make_layout(size<3>(thr_layout_vmnk))));
    auto thr_tensor = zipped_divide(tv_atom_sfa, thr_tile);
    return thr_tensor;
  }

  template <class SFATensor, class ThrMma>
  CUTE_HOST_DEVICE static constexpr auto partition_fragment_SFA(SFATensor&& sfatensor,
                                                                ThrMma& thread_mma) {
    auto thr_tensor = make_tensor(static_cast<SFATensor&&>(sfatensor).data(),
                                  thrfrg_SFA(sfatensor.layout(), thread_mma));
    auto thr_vmnk = thread_mma.thr_vmnk_;
    auto thr_vmk = make_coord(get<0>(thr_vmnk), make_coord(get<1>(thr_vmnk), get<3>(thr_vmnk)));
    auto partition_SFA = thr_tensor(thr_vmk, make_coord(_, repeat<rank<1, 1>(thr_tensor)>(_)));
    auto frg_SFA = make_fragment_like<ElementSFLoad>(partition_SFA);
    return frg_SFA;
  }

  template <class TiledMma>
  CUTE_HOST_DEVICE static constexpr auto get_layoutSFA_TV(TiledMma& mma) {
    auto tile_shape_mnk = tile_shape(mma);
    auto ref_A = make_layout(make_shape(size<0>(tile_shape_mnk), _1{}));
    auto thr_tensor = thrfrg_SFA(ref_A, mma);
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
    auto atile = make_tile(
        _, make_tile(make_layout(make_shape(size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                                 make_stride(Int<1>{}, Int<0>{})),
                     _));
    auto tv_sfa = thr_tensor.compose(atile, _);
    auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
    auto tv_layout = tv_sfa.compose(thridx_2_thrid, _);
    return tv_layout;
  }

  template <class SFBTensor, class Atom, class TiledThr, class TiledPerm>
  CUTE_HOST_DEVICE static constexpr auto thrfrg_SFB(SFBTensor&& sfbtensor,
                                                    TiledMMA<Atom, TiledThr, TiledPerm>& mma) {
    CUTE_STATIC_ASSERT_V(rank(sfbtensor) >= Int<2>{});

    auto permutation_mnk = TiledPerm{};
    auto t_tile = make_tile(get<1>(permutation_mnk), _1{});
    auto tiled_sfb = logical_divide(sfbtensor, t_tile);

    using AtomShape_MNK = typename Atom::Shape_MNK;
    auto atom_tile = make_tile(make_layout(size<1>(AtomShape_MNK{})), make_layout(_1{}));
    auto tiled_atom_sfb = zipped_divide(tiled_sfb, atom_tile);
    using AtomLayoutSFB_TV = Layout<Shape<Shape<_4, _8>, _1>, Stride<Stride<_0, _1>, _8>>;
    auto tv_atom_sfb = tiled_atom_sfb.compose(AtomLayoutSFB_TV{}, _);

    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
    auto thr_tile = make_tile(
        _, make_tile(make_layout(size<2>(thr_layout_vmnk)), make_layout(size<3>(thr_layout_vmnk))));
    auto thr_tensor = zipped_divide(tv_atom_sfb, thr_tile);
    return thr_tensor;
  }

  template <class SFBTensor, class ThrMma>
  CUTE_HOST_DEVICE static constexpr auto partition_fragment_SFB(SFBTensor&& sfbtensor,
                                                                ThrMma& thread_mma) {
    auto thr_tensor = make_tensor(static_cast<SFBTensor&&>(sfbtensor).data(),
                                  thrfrg_SFB(sfbtensor.layout(), thread_mma));
    auto thr_vmnk = thread_mma.thr_vmnk_;
    auto thr_vnk = make_coord(get<0>(thr_vmnk), make_coord(get<1>(thr_vmnk), get<3>(thr_vmnk)));
    auto partition_SFB = thr_tensor(thr_vnk, make_coord(_, repeat<rank<1, 1>(thr_tensor)>(_)));
    auto frg_SFB = make_fragment_like<ElementSFLoad>(partition_SFB);
    return frg_SFB;
  }

  template <class TiledMma>
  CUTE_HOST_DEVICE static constexpr auto get_layoutSFB_TV(TiledMma& mma) {
    auto tile_shape_mnk = tile_shape(mma);
    auto ref_B = make_layout(make_shape(size<1>(tile_shape_mnk), _1{}));
    auto thr_tensor = thrfrg_SFB(ref_B, mma);
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
    auto btile = make_tile(
        _, make_tile(make_layout(make_shape(size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                                 make_stride(Int<0>{}, Int<1>{})),
                     _));
    auto tv_sfb = thr_tensor.compose(btile, _);
    auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
    auto tv_layout = tv_sfb.compose(thridx_2_thrid, _);
    return tv_layout;
  }

  template <class Tensor>
  CUTE_HOST_DEVICE static constexpr auto transform_fragment_for_qmma(Tensor&& tensor) {
    CUTE_STATIC_ASSERT_V(rank(tensor) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<2>(shape(tensor)) == Int<1>{});
    auto old_ptr = tensor.data();
    auto new_ptr = recast_ptr<ElementSFCompute>(old_ptr);
    auto old_layout = tensor.layout();
    auto num_mn = size<1>(shape(old_layout));
    // Every 32-element MMA K atom of the tile picks byte k of the thread's word.
    auto new_layout =
        make_layout(make_shape(Int<SFVecSize>{}, num_mn, make_shape(_1{}, Int<kNumSFPerTileK>{}),
                               make_shape(_1{}, _1{})),
                    make_stride(_0{}, Int<PACK_NSF>{}, make_stride(_0{}, _1{}),
                                make_stride(_0{}, Int<kNumSFPerTileK>{})));
    auto new_tensor = make_tensor(new_ptr, new_layout);
    return new_tensor;
  }

  using SmemCopyAtomSF = Copy_Atom<AutoVectorizingCopy, ElementSFLoad>;

  using SmemLayoutAtomSFA =
      decltype(make_ordered_layout(select<0, 2>(ScaleTileShape{}), Step<_1, _2>{}));

  using SmemLayoutAtomSFB =
      decltype(make_ordered_layout(select<1, 2>(ScaleTileShape{}), Step<_1, _2>{}));

  static constexpr int kSfStageStrideElems = 128 / static_cast<int>(sizeof(ElementSFLoad));
  static constexpr int kSfaStageElems =
      (static_cast<int>(cute::cosize_v<SmemLayoutAtomSFA>) < kSfStageStrideElems)
          ? kSfStageStrideElems
          : static_cast<int>(cute::cosize_v<SmemLayoutAtomSFA>);
  static constexpr int kSfbStageElems =
      (static_cast<int>(cute::cosize_v<SmemLayoutAtomSFB>) < kSfStageStrideElems)
          ? kSfStageStrideElems
          : static_cast<int>(cute::cosize_v<SmemLayoutAtomSFB>);

  using SmemLayoutSFA = Layout<Shape<decltype(shape<0>(ScaleTileShape{})),
                                     decltype(shape<2>(ScaleTileShape{})), Int<kStageSF>>,
                               Stride<_1, _0, Int<kSfaStageElems>>>;

  using SmemLayoutSFB = Layout<Shape<decltype(shape<1>(ScaleTileShape{})),
                                     decltype(shape<2>(ScaleTileShape{})), Int<kStageSF>>,
                               Stride<_1, _0, Int<kSfbStageElems>>>;

  using StrideSFA = Stride<Int<1>, int64_t, int64_t>;
  using StrideSFB = Stride<Int<1>, int64_t, int64_t>;

  using TMA_SFA = decltype(make_tma_copy(
      SM90_TMA_LOAD{},
      make_tensor(recast_ptr<ElementSFLoad>(nullptr),
                  repeat_like(StrideSFA{}, static_cast<int64_t>(0)), StrideSFA{}),
      SmemLayoutSFA{}(_, _, cute::Int<0>{}),  // NOLINT
      make_shape(shape<0>(ScaleTileShape{}), shape<2>(ScaleTileShape{})), _1{}));

  using TMA_SFB = decltype(make_tma_copy(
      SM90_TMA_LOAD{},
      make_tensor(recast_ptr<ElementSFLoad>(nullptr),
                  repeat_like(StrideSFB{}, static_cast<int64_t>(0)), StrideSFB{}),
      SmemLayoutSFB{}(_, _, cute::Int<0>{}),  // NOLINT
      make_shape(shape<1>(ScaleTileShape{}), shape<2>(ScaleTileShape{})), _1{}));

  static constexpr uint32_t TmaTransactionBytesSFA = static_cast<uint32_t>(cutlass::bits_to_bytes(
      cosize(take<0, 2>(SmemLayoutSFA{})) * cute::sizeof_bits_v<ElementSFLoad>));
  static constexpr uint32_t TmaTransactionBytesSFB = static_cast<uint32_t>(cutlass::bits_to_bytes(
      cosize(take<0, 2>(SmemLayoutSFB{})) * cute::sizeof_bits_v<ElementSFLoad>));
  static constexpr uint32_t TmaSFTransactionBytes = TmaTransactionBytesSFA + TmaTransactionBytesSFB;
};

template <typename KT>
static auto make_sf_tma_descriptors(typename KT::SFConfig::ElementSFLoad* ptr_SFA,
                                    typename KT::SFConfig::ElementSFLoad* ptr_SFB, int M, int N,
                                    int K, int num_experts) {
  int sfa_src_M = M;
  int sfa_src_L = 1;
  int sfb_src_N = N;
  int sfb_src_L = num_experts;

  auto sfa_shape = make_shape(sfa_src_M, sfb_src_N, K, sfa_src_L);
  auto sfa_layout = KT::SFConfig::deduce_sfa_layout(sfa_shape, num_experts);
  auto tensor_sfa = make_tensor(make_gmem_ptr(ptr_SFA), sfa_layout);
  typename KT::SFConfig::TMA_SFA tma_load_sfa =
      make_tma_copy(SM90_TMA_LOAD{}, tensor_sfa,
                    typename KT::SFConfig::SmemLayoutSFA{}(_, _, Int<0>{}),  // NOLINT
                    make_shape(shape<0>(typename KT::SFConfig::ScaleTileShape{}),
                               shape<2>(typename KT::SFConfig::ScaleTileShape{})),
                    _1{});

  auto sfb_shape = make_shape(sfa_src_M, sfb_src_N, K, sfb_src_L);
  auto sfb_layout = KT::SFConfig::deduce_sfb_layout(sfb_shape, num_experts);
  auto tensor_sfb = make_tensor(make_gmem_ptr(ptr_SFB), sfb_layout);
  typename KT::SFConfig::TMA_SFB tma_load_sfb =
      make_tma_copy(SM90_TMA_LOAD{}, tensor_sfb,
                    typename KT::SFConfig::SmemLayoutSFB{}(_, _, Int<0>{}),  // NOLINT
                    make_shape(shape<1>(typename KT::SFConfig::ScaleTileShape{}),
                               shape<2>(typename KT::SFConfig::ScaleTileShape{})),
                    _1{});

  return cute::make_tuple(tma_load_sfa, tma_load_sfb);
}

template <typename KT, typename BlkCoord>
CUTE_DEVICE auto tma_sf_partition(typename KT::SFConfig::TMA_SFA const& tma_load_sfa,
                                  typename KT::SFConfig::TMA_SFB const& tma_load_sfb, int M, int N,
                                  int K, int num_experts, BlkCoord const& blk_coord,
                                  int32_t m_offset) {
  auto [m_block_idx, n_block_idx, expert_idx] = blk_coord;
  using X = Underscore;
  int sfa_src_M = M;
  int sfa_src_L = 1;
  int sfb_src_N = N;
  int sfb_src_L = num_experts;
  int32_t sfa_tile_idx = m_block_idx;
  int32_t sfa_batch_idx = 0;
  int32_t sfb_tile_idx = n_block_idx;
  int32_t sfb_batch_idx = expert_idx;

  auto sfa_shape = make_shape(sfa_src_M, sfb_src_N, K, sfa_src_L);
  auto mSFA_full =
      tma_load_sfa.get_tma_tensor(shape(KT::SFConfig::deduce_sfa_layout(sfa_shape, num_experts)));

  auto sfb_shape = make_shape(sfa_src_M, sfb_src_N, K, sfb_src_L);
  auto mSFB_full =
      tma_load_sfb.get_tma_tensor(shape(KT::SFConfig::deduce_sfb_layout(sfb_shape, num_experts)));

  auto sf_m_offset = packed_scale_expert_base(m_offset, expert_idx);
  auto mSFA = cute::domain_offset(make_coord(sf_m_offset, 0, 0), mSFA_full);
  auto mSFB = mSFB_full;

  auto gSFA_mkl = local_tile(mSFA, typename KT::SFConfig::ScaleTileShape{}, make_coord(_, _, _),
                             Step<_1, X, _1>{});
  auto gSFB_nkl = local_tile(mSFB, typename KT::SFConfig::ScaleTileShape{}, make_coord(_, _, _),
                             Step<X, _1, _1>{});

  auto block_tma_sfa = tma_load_sfa.get_slice(0);
  auto block_tma_sfb = tma_load_sfb.get_slice(0);

  auto gSFA = gSFA_mkl(_, _, sfa_tile_idx, _, sfa_batch_idx);
  auto gSFB = gSFB_nkl(_, _, sfb_tile_idx, _, sfb_batch_idx);

  auto tAgSFA = block_tma_sfa.partition_S(gSFA);
  auto tBgSFB = block_tma_sfb.partition_S(gSFB);

  return cute::make_tuple(tAgSFA, tBgSFB);
}

}  // namespace common
}  // namespace hpc::fuse_moe::sm120::gemm
