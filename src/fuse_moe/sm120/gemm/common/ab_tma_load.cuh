// Copyright (C) 2026 Tencent.

#pragma once

// clang-format off
#include <cutlass/float_subbyte.h>
#include <cutlass/numeric_size.h>

#include <cstdint>

#include <cute/arch/copy_sm100.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/copy_sm90.hpp>
#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/layout.hpp>
#include <cute/pointer.hpp>
// clang-format on

namespace hpc::fuse_moe::sm120::gemm {
namespace common {

using namespace cute;  // NOLINT

template <int kTileK, typename Element>
CUTE_HOST_DEVICE constexpr auto select_smem_layout_atom_k() {
  static_assert(kTileK == 128);
  return GMMA::Layout_K_SW128_Atom<Element>{};
}

template <typename KT, typename BlkCoord>
CUTE_DEVICE auto tma_ab_partition(typename KT::ABLoadConfig::TMA_A const& tma_load_a,
                                  typename KT::ABLoadConfig::TMA_B const& tma_load_b, int M, int N,
                                  int K, int num_experts, BlkCoord const& blk_coord,
                                  int32_t m_offset) {
  auto [m_block_idx, n_block_idx, expert_idx] = blk_coord;
  using X = Underscore;

  int a_src_M = M;
  int a_src_L = 1;
  int b_src_N = N;
  int b_src_L = num_experts;
  int32_t a_tile_idx = m_block_idx;
  int32_t a_batch_idx = 0;
  int32_t b_tile_idx = n_block_idx;
  int32_t b_batch_idx = expert_idx;

  auto mA_full = tma_load_a.get_tma_tensor(make_shape(a_src_M, K, a_src_L));
  auto mB_full = tma_load_b.get_tma_tensor(make_shape(b_src_N, K, b_src_L));

  auto mA = cute::domain_offset(make_coord(m_offset, 0, 0), mA_full);
  auto mB = mB_full;

  auto gA_mkl = local_tile(mA, typename KT::TileShape{}, make_coord(_, _, _), Step<_1, X, _1>{});
  auto gB_nkl = local_tile(mB, typename KT::TileShape{}, make_coord(_, _, _), Step<X, _1, _1>{});

  auto block_tma_a = tma_load_a.get_slice(0);
  auto block_tma_b = tma_load_b.get_slice(0);

  auto gA = gA_mkl(_, _, a_tile_idx, _, a_batch_idx);
  auto gB = gB_nkl(_, _, b_tile_idx, _, b_batch_idx);

  auto tAgA = block_tma_a.partition_S(gA);
  auto tBgB = block_tma_b.partition_S(gB);

  return cute::make_tuple(tAgA, tBgB);
}

template <typename ElementB>
struct MxOperandB;

// MXFP4 E2M1 weights are packed two to a byte in global memory and TMA unpacks
// them to one per byte in SMEM. GEMM K counts FP4 elements, so the TMA recast
// starts from the sub-byte type.
template <>
struct MxOperandB<cute::float_e2m1_t> {
  using ElementSmem = uint8_t;
  using TmaElement = cutlass::detail::float_e2m1_unpacksmem_t;
  using SmemCopyAtom = Copy_Atom<SM100_SU4_DU8x16_x4_LDSM_N, ElementSmem>;
};

template <>
struct MxOperandB<cute::float_e4m3_t> {
  using ElementSmem = cute::float_e4m3_t;
  using TmaElement = cute::float_e4m3_t;
  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, ElementSmem>;
};

// MXFP8 E4M3 A with MXFP4 E2M1 or MXFP8 E4M3 B. Both B types take one byte per
// element in SMEM, so the SMEM layout and stage size do not depend on B.
template <typename ElementB_, int kTileM, int kTileN, int kTileK, int kStages>
struct AbLoadConfig {
  using ElementA = cute::float_e4m3_t;
  using ElementB = ElementB_;
  using ElementBGmem = ElementB;
  using ElementBSmem = typename MxOperandB<ElementB>::ElementSmem;
  using TmaElementB = typename MxOperandB<ElementB>::TmaElement;

  using TileShape = Shape<Int<kTileM>, Int<kTileN>, Int<kTileK>>;

  using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, ElementA>;
  using SmemCopyAtomB = typename MxOperandB<ElementB>::SmemCopyAtom;

  using SmemLayoutAtomA = decltype(select_smem_layout_atom_k<kTileK, ElementA>());
  using SmemLayoutAtomB = decltype(select_smem_layout_atom_k<kTileK, ElementBSmem>());

  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{}, make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<kStages>{}),
      Step<_1, _2, _3>{}));

  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{}, make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<kStages>{}),
      Step<_1, _2, _3>{}));

  using StrideA = Stride<int64_t, Int<1>, int64_t>;
  using StrideB = Stride<int64_t, Int<1>, int64_t>;

  using TMA_A = decltype(make_tma_copy(
      SM90_TMA_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<ElementA const*>(nullptr)),
                  repeat_like(StrideA{}, static_cast<int64_t>(0)), StrideA{}),
      SmemLayoutA{}(_, _, Int<0>{}),  // NOLINT
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})), _1{}));

  using TMA_B = decltype(make_tma_copy(
      SM90_TMA_LOAD{},
      make_tensor(recast_ptr<TmaElementB>(static_cast<ElementB const*>(nullptr)),
                  repeat_like(StrideB{}, static_cast<int64_t>(0)), StrideB{}),
      SmemLayoutB{}(_, _, Int<0>{}),  // NOLINT
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})), _1{}));

  static constexpr uint32_t TmaTransactionBytesA = static_cast<uint32_t>(
      cutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutA{})) * cute::sizeof_bits_v<ElementA>));
  static constexpr uint32_t TmaTransactionBytesB = static_cast<uint32_t>(
      cutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutB{})) * cute::sizeof_bits_v<ElementB>));
  static constexpr uint32_t TmaABTransactionBytes = TmaTransactionBytesA + TmaTransactionBytesB;
};

template <typename KT>
static auto make_b_tma_descriptor(typename KT::ElementBGmem const* ptr_B,
                                  typename KT::ABLoadConfig::StrideB dB, int N, int K,
                                  int num_experts) {
  using TmaElementB = typename KT::ABLoadConfig::TmaElementB;
  auto tensor_B = make_tensor(make_gmem_ptr(recast_ptr<TmaElementB>(ptr_B)),
                              make_layout(make_shape(N, K, num_experts), dB));
  typename KT::ABLoadConfig::TMA_B tma_load_b = make_tma_copy(
      SM90_TMA_LOAD{}, tensor_B,
      typename KT::ABLoadConfig::SmemLayoutB{}(_, _, Int<0>{}),  // NOLINT
      make_shape(shape<1>(typename KT::TileShape{}), shape<2>(typename KT::TileShape{})), _1{});
  return tma_load_b;
}

// Route-major A [M, K] and expert weights B [num_experts, N, K], as the down GEMM reads them.
template <typename KT>
static auto make_ab_tma_descriptors(typename KT::ElementA const* ptr_A,
                                    typename KT::ABLoadConfig::StrideA dA,
                                    typename KT::ElementBGmem const* ptr_B,
                                    typename KT::ABLoadConfig::StrideB dB, int M, int N, int K,
                                    int num_experts) {
  auto tensor_A = make_tensor(make_gmem_ptr(ptr_A), make_layout(make_shape(M, K, 1), dA));
  typename KT::ABLoadConfig::TMA_A tma_load_a = make_tma_copy(
      SM90_TMA_LOAD{}, tensor_A,
      typename KT::ABLoadConfig::SmemLayoutA{}(_, _, Int<0>{}),  // NOLINT
      make_shape(shape<0>(typename KT::TileShape{}), shape<2>(typename KT::TileShape{})), _1{});
  return cute::make_tuple(tma_load_a, make_b_tma_descriptor<KT>(ptr_B, dB, N, K, num_experts));
}

}  // namespace common
}  // namespace hpc::fuse_moe::sm120::gemm
