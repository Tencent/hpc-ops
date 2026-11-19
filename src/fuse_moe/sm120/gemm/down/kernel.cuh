// Copyright (C) 2026 Tencent.

#pragma once

// clang-format off
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cutlass/arch/barrier.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>

#include <type_traits>

#include <cute/atom/mma_traits_sm120.hpp>

#include "src/fuse_moe/sm120/common/utils.h"
#include "src/fuse_moe/sm120/gemm/common/ab_tma_load.cuh"
#include "src/fuse_moe/sm120/gemm/common/live_rows.cuh"
#include "src/fuse_moe/sm120/gemm/common/pipe_state.cuh"
#include "src/fuse_moe/sm120/gemm/common/qmma.cuh"
#include "src/fuse_moe/sm120/gemm/common/scheduler.cuh"
#include "src/fuse_moe/sm120/gemm/common/sf_tma_load.cuh"
#include "src/fuse_moe/sm120/gemm/down/config.cuh"
#include "src/fuse_moe/sm120/gemm/down/scatter_rows.cuh"
// clang-format on

namespace hpc::fuse_moe::sm120::gemm {
namespace down {

using namespace cute;  // NOLINT

template <typename KT, typename Accum>
CUTE_DEVICE auto convert_accum(Accum const& accum) {
  auto out = make_fragment_like<typename KT::ElementD>(accum);
  auto accum_frg = recast<cutlass::Array<typename KT::ElementAccum, 2>>(accum);
  auto out_frg = recast<cutlass::Array<typename KT::ElementD, 2>>(out);
  cutlass::NumericArrayConverter<typename KT::ElementD, typename KT::ElementAccum, 2> converter;
  cute::for_each(cute::make_int_sequence<cute::size(out_frg)>{},
                 [&](auto i) { out_frg(i) = converter(accum_frg(i)); });
  return out;
}

template <typename Config>
CUTE_HOST_DEVICE constexpr auto scatter_tiled_copy() {
  return typename Config::ScatterTiledCopy{};
}

// ptr must be aligned to the vector: 16 bytes for four bf16x2.
template <int kPairs>
CUTE_DEVICE void red_add_bf16x2(__nv_bfloat16* ptr, uint32_t const (&v)[kPairs]) {
  static_assert(kPairs == 1 || kPairs == 2 || kPairs == 4);
  if constexpr (kPairs == 4) {
    asm volatile("red.global.add.noftz.v4.bf16x2 [%0], {%1, %2, %3, %4};"
                 :
                 : "l"(ptr), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3])
                 : "memory");
  } else if constexpr (kPairs == 2) {
    asm volatile("red.global.add.noftz.v2.bf16x2 [%0], {%1, %2};"
                 :
                 : "l"(ptr), "r"(v[0]), "r"(v[1])
                 : "memory");
  } else {
    asm volatile("red.global.add.noftz.bf16x2 [%0], %1;" : : "l"(ptr), "r"(v[0]) : "memory");
  }
}

template <typename KT>
CUTE_DEVICE auto scatter_coords(int thread_idx) {
  decltype(scatter_tiled_copy<typename KT::StoreConfig>()) tiled_copy;
  auto cD = cute::make_identity_tensor(
      cute::make_shape(cute::Int<KT::kTileM>{}, cute::Int<KT::kTileN>{}));
  return tiled_copy.get_slice(thread_idx).partition_D(cD);
}

template <int kRows>
struct ScatterRows {
  int tokens[kRows];
  float weights[kRows];
};

template <typename KT, typename Slot>
CUTE_DEVICE auto load_scatter_rows(Slot const& slot, int thread_idx) {
  auto coords = scatter_coords<KT>(thread_idx);
  constexpr int kRows = decltype(cute::size<1>(coords))::value;
  ScatterRows<kRows> rows;
  slot.full->wait(slot.phase);
  CUTE_UNROLL
  for (int m = 0; m < kRows; ++m) {
    int2 row = slot.rows[cute::get<0>(coords(0, m, 0))];
    rows.tokens[m] = row.x;
    rows.weights[m] = __int_as_float(row.y);
  }
  __syncwarp();
  if (thread_idx % 32 == 0) {
    slot.empty->arrive();
  }
  return rows;
}

template <typename KT, typename Params, typename SharedStorage, typename Accum, typename Slot,
          typename Barrier>
CUTE_DEVICE void scatter_store(Params const& params, SharedStorage& shared_storage,
                               Accum const& accum, int thread_idx, int32_t n_block_idx,
                               Slot const& slot, Barrier* store_empty_mbar) {
  typename KT::MMAConfig::TiledMma mma;
  auto out = convert_accum<KT>(accum);
  auto sD_ = cute::make_tensor(cute::make_smem_ptr(shared_storage.tensors.store.smem_O.begin()),
                               typename KT::StoreConfig::SmemLayoutO{});
  auto sD = as_position_independent_swizzle_tensor(sD_);
  auto rows = load_scatter_rows<KT>(slot, thread_idx);

  decltype(scatter_tiled_copy<typename KT::StoreConfig>()) tiled_copy;
  auto thr_copy = tiled_copy.get_slice(thread_idx);
  auto coords = scatter_coords<KT>(thread_idx);
  constexpr int kRows = decltype(cute::size<1>(coords))::value;
  int32_t residue_n = params.N - n_block_idx * KT::kTileN;
  auto const& tokens = rows.tokens;
  auto const& weights = rows.weights;

  auto smem_store = cute::make_tiled_copy_C(typename KT::StoreConfig::SmemCopyAtom{}, mma);
  auto thr_smem_store = smem_store.get_slice(thread_idx);
  cute::copy(smem_store, thr_smem_store.retile_S(out), thr_smem_store.partition_D(sD));
  cutlass::arch::NamedBarrier::sync(KT::MMAConfig::kNumMathThreads, 0);

  auto t_sD = thr_copy.partition_S(sD);
  auto t_rD = cute::make_tensor<typename KT::ElementD>(cute::shape(t_sD));
  cute::copy(tiled_copy, t_sD, t_rD);
  cutlass::arch::NamedBarrier::sync(KT::MMAConfig::kNumMathThreads, 0);
  if constexpr (KT::kUnionSmem) {
    store_empty_mbar[0].arrive();
  }

  auto rD = thr_copy.retile_S(t_rD);
  constexpr int kVec = decltype(cute::size<0>(rD))::value;
  CUTE_UNROLL
  for (int m = 0; m < kRows; ++m) {
    if (tokens[m] >= 0) {
      auto* row = reinterpret_cast<__nv_bfloat16*>(params.ptr_D) +
                  static_cast<int64_t>(tokens[m]) * params.N + n_block_idx * KT::kTileN;
      CUTE_UNROLL
      for (int n = 0; n < cute::size<2>(rD); ++n) {
        int n_in = cute::get<1>(coords(0, m, n));
        if (n_in + kVec <= residue_n) {
          uint32_t packed[kVec / 2];
          CUTE_UNROLL
          for (int v = 0; v < kVec; v += 2) {
            __nv_bfloat162 value =
                __floats2bfloat162_rn(static_cast<float>(rD(v, m, n)) * weights[m],
                                      static_cast<float>(rD(v + 1, m, n)) * weights[m]);
            packed[v / 2] = *reinterpret_cast<uint32_t const*>(&value);
          }
          red_add_bf16x2(row + n_in, packed);
        }
      }
    }
  }
}

template <typename KT, bool kHinted = false>
struct DownGemmKernel {
  // One packed scale load covers one K tile, so the SF and AB rings advance together.
  static_assert(KT::SFConfig::kStageSF == KT::kStageAB);

  static constexpr int kNumTMAThreads = 128;
  static constexpr int kNumMathThreads = KT::MMAConfig::kNumMathThreads;
  static constexpr int MaxThreadsPerBlock = kNumTMAThreads + kNumMathThreads;
  // Bounds the launch register count; setmaxnreg then moves registers from the
  // load warps to the MMA warps.
  static constexpr int MinBlocksPerMultiprocessor = 1;

  using ProblemShape = typename KT::ProblemShape;

  struct Params {
    typename KT::ABLoadConfig::TMA_A tma_load_a;
    typename KT::ABLoadConfig::TMA_B tma_load_b;
    typename KT::SFConfig::TMA_SFA tma_load_sfa;
    typename KT::SFConfig::TMA_SFB tma_load_sfb;
    typename KT::ElementD* ptr_D;
    int M;
    int N;
    int K;
    int num_experts;
    int32_t const* cu_padded_tokens_per_group;
    int32_t const* row_counts;
    int32_t const* row_to_route;
    float const* topk_scale;
    int num_topk;
    uint64_t cache_hint_a;
    uint64_t cache_hint_b;
  };

  struct Arguments {
    typename KT::ElementA* ptr_A;
    typename KT::ABLoadConfig::StrideA dA;
    typename KT::ElementB* ptr_B;
    typename KT::ABLoadConfig::StrideB dB;
    typename KT::SFConfig::ElementSFLoad* ptr_SFA;
    typename KT::SFConfig::ElementSFLoad* ptr_SFB;
    typename KT::ElementD* ptr_D;
    int32_t const* cu_padded_tokens_per_group;
    int32_t const* row_counts = nullptr;
    int32_t const* row_to_route = nullptr;
    float const* topk_scale = nullptr;
    int num_topk = 0;
    uint64_t cache_hint_a = static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL);
    uint64_t cache_hint_b = static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL);
  };

  static Params to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args) {
    auto [M, N, K, num_experts] = problem_shape;
    auto [tma_load_a, tma_load_b] = common::make_ab_tma_descriptors<KT>(
        args.ptr_A, args.dA, args.ptr_B, args.dB, M, N, K, num_experts);
    auto [tma_load_sfa, tma_load_sfb] =
        common::make_sf_tma_descriptors<KT>(args.ptr_SFA, args.ptr_SFB, M, N, K, num_experts);

    return {tma_load_a,
            tma_load_b,
            tma_load_sfa,
            tma_load_sfb,
            args.ptr_D,
            M,
            N,
            K,
            num_experts,
            args.cu_padded_tokens_per_group,
            args.row_counts,
            args.row_to_route,
            args.topk_scale,
            args.num_topk,
            args.cache_hint_a,
            args.cache_hint_b};
  }

  static dim3 get_grid_shape(int num_sms) { return dim3(num_sms, 1, 1); }

  static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }

  CUTE_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
    cute::prefetch_tma_descriptor(params.tma_load_a.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_b.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_sfa.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_sfb.get_tma_descriptor());
  }

  using TensorStorage = std::conditional_t<KT::kUnionSmem, typename KT::TensorStorageUnion,
                                           typename KT::TensorStorageSplit>;
  using BarrierStorage = typename KT::BarrierStorage;
  using Scheduler = common::MoeScheduler<KT::kTileM, KT::kTileN>;

  static constexpr int kNumSchedStages = 2;
  static constexpr int kNumSchedConsumers = KT::MMAConfig::kNumMathWarps + 2;
  // One more than the sched stages: the scheduler fills a tile's rows after
  // publishing it and must not stall the loaders on an epilogue.
  static constexpr int kNumScatterStages = kNumSchedStages + 1;

  struct SharedStorage {
    TensorStorage tensors;
    alignas(16) BarrierStorage barriers;
    alignas(8) common::MoeSchedStorage<kNumSchedStages> sched;
    alignas(16) MoeScatterRowsStorage<kNumScatterStages, KT::kTileM> scatter;
  };
  static constexpr int kSmemSize = static_cast<int>(sizeof(SharedStorage));

  using FullBarrier = typename KT::FullBarrier;
  using EmptyBarrier = typename KT::EmptyBarrier;
  using ProducerBarrierType = typename FullBarrier::ValueType;

  CUTE_DEVICE
  static auto get_mbarriers(SharedStorage& shared_storage) {
    auto* ab_full_mbar = recast_ptr<FullBarrier>(&shared_storage.barriers.ab_full_mbar[0]);
    auto* ab_empty_mbar = recast_ptr<EmptyBarrier>(&shared_storage.barriers.ab_empty_mbar[0]);
    auto* sf_full_mbar = recast_ptr<FullBarrier>(&shared_storage.barriers.sf_full_mbar[0]);
    auto* sf_empty_mbar = recast_ptr<EmptyBarrier>(&shared_storage.barriers.sf_empty_mbar[0]);
    auto* store_empty_mbar = recast_ptr<EmptyBarrier>(&shared_storage.barriers.store_empty_mbar[0]);
    return cute::make_tuple(ab_full_mbar, ab_empty_mbar, sf_full_mbar, sf_empty_mbar,
                            store_empty_mbar);
  }

  template <typename BlkCoord>
  CUTE_DEVICE static void load_sf(Params const& params, SharedStorage& shared_storage,
                                  BlkCoord const& blk_coord, int32_t m_offset, PipeState& sf_pipe,
                                  uint32_t& store_phase) {
    auto [tAgSFA, tBgSFB] =
        common::tma_sf_partition<KT>(params.tma_load_sfa, params.tma_load_sfb, params.M, params.N,
                                     params.K, params.num_experts, blk_coord, m_offset);

    auto block_tma_sfa = params.tma_load_sfa.get_slice(0);
    auto block_tma_sfb = params.tma_load_sfb.get_slice(0);

    auto sSFA_ = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFA.begin()),
                             typename KT::SFConfig::SmemLayoutSFA{});
    auto sSFB_ = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFB.begin()),
                             typename KT::SFConfig::SmemLayoutSFB{});
    auto sSFA = as_position_independent_swizzle_tensor(sSFA_);
    auto sSFB = as_position_independent_swizzle_tensor(sSFB_);

    auto tAsSFA = block_tma_sfa.partition_D(sSFA);
    auto tBsSFB = block_tma_sfb.partition_D(sSFB);

    auto [ab_full_mbar, ab_empty_mbar, sf_full_mbar, sf_empty_mbar, store_empty_mbar] =
        get_mbarriers(shared_storage);
    if constexpr (KT::kUnionSmem) {
      store_empty_mbar[0].wait(store_phase);
      store_phase ^= 1;
    }

    int32_t k_tiles = cute::ceil_div(params.K, static_cast<int>(KT::kTileK));
    for (int32_t k_tile = 0; k_tile < k_tiles; ++k_tile) {
      int stage = sf_pipe.stage;
      sf_empty_mbar[stage].wait(sf_pipe.phase);
      auto& sf_full_barrier = sf_full_mbar[stage];
      auto tma_copy_sfa =
          params.tma_load_sfa.with(*recast_ptr<ProducerBarrierType>(&sf_full_barrier));
      cute::copy(tma_copy_sfa, tAgSFA(_, _, _, k_tile), tAsSFA(_, _, _, stage));
      auto tma_copy_sfb =
          params.tma_load_sfb.with(*recast_ptr<ProducerBarrierType>(&sf_full_barrier));
      cute::copy(tma_copy_sfb, tBgSFB(_, _, _, k_tile), tBsSFB(_, _, _, stage));
      sf_full_barrier.arrive_and_expect_tx(KT::SFConfig::TmaSFTransactionBytes);
      sf_pipe.advance<KT::SFConfig::kStageSF>();
    }
  }

  template <typename BlkCoord>
  CUTE_DEVICE static void load_ab(Params const& params, SharedStorage& shared_storage,
                                  BlkCoord const& blk_coord, int32_t m_offset, PipeState& ab_pipe,
                                  uint32_t& store_phase) {
    auto [tAgA, tBgB] =
        common::tma_ab_partition<KT>(params.tma_load_a, params.tma_load_b, params.M, params.N,
                                     params.K, params.num_experts, blk_coord, m_offset);

    auto block_tma_a = params.tma_load_a.get_slice(0);
    auto block_tma_b = params.tma_load_b.get_slice(0);

    auto sA_ = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_A.begin()),
                           typename KT::ABLoadConfig::SmemLayoutA{});
    auto sB_ = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_B.begin()),
                           typename KT::ABLoadConfig::SmemLayoutB{});
    auto sA = as_position_independent_swizzle_tensor(sA_);
    auto sB = as_position_independent_swizzle_tensor(sB_);

    auto tAsA = block_tma_a.partition_D(sA);
    auto tBsB = block_tma_b.partition_D(sB);

    auto [ab_full_mbar, ab_empty_mbar, sf_full_mbar, sf_empty_mbar, store_empty_mbar] =
        get_mbarriers(shared_storage);
    if constexpr (KT::kUnionSmem) {
      store_empty_mbar[0].wait(store_phase);
      store_phase ^= 1;
    }

    int32_t k_tiles = cute::ceil_div(params.K, static_cast<int>(KT::kTileK));
    for (int32_t k_tile = 0; k_tile < k_tiles; ++k_tile) {
      int stage = ab_pipe.stage;
      ab_empty_mbar[stage].wait(ab_pipe.phase);
      auto& ab_full_barrier = ab_full_mbar[stage];
      auto tma_copy_a = params.tma_load_a.with(
          *recast_ptr<ProducerBarrierType>(&ab_full_barrier), 0,
          kHinted ? static_cast<cute::TMA::CacheHintSm90>(params.cache_hint_a)
                  : cute::TMA::CacheHintSm90::EVICT_NORMAL);
      cute::copy(tma_copy_a, tAgA(_, _, _, k_tile), tAsA(_, _, _, stage));
      auto tma_copy_b = params.tma_load_b.with(
          *recast_ptr<ProducerBarrierType>(&ab_full_barrier), 0,
          kHinted ? static_cast<cute::TMA::CacheHintSm90>(params.cache_hint_b)
                  : cute::TMA::CacheHintSm90::EVICT_NORMAL);
      cute::copy(tma_copy_b, tBgB(_, _, _, k_tile), tBsB(_, _, _, stage));
      ab_full_barrier.arrive_and_expect_tx(KT::ABLoadConfig::TmaABTransactionBytes);
      ab_pipe.advance<KT::kStageAB>();
    }
  }

  CUTE_DEVICE
  static void mma(Params const& params, SharedStorage& shared_storage, int32_t m_offset,
                  int32_t m_boundary, int32_t m_block_idx, int32_t n_block_idx, PipeState& pipe,
                  MoeScatterRowsSlot const& scatter_slot) {
    int thread_idx = static_cast<int>(threadIdx.x);

    auto sA_ = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_A.begin()),
                           typename KT::ABLoadConfig::SmemLayoutA{});
    auto sB_ = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_B.begin()),
                           typename KT::ABLoadConfig::SmemLayoutB{});
    auto sSFA_ = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFA.begin()),
                             typename KT::SFConfig::SmemLayoutSFA{});
    auto sSFB_ = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFB.begin()),
                             typename KT::SFConfig::SmemLayoutSFB{});
    auto sA = as_position_independent_swizzle_tensor(sA_);
    auto sB = as_position_independent_swizzle_tensor(sB_);
    auto sSFA = as_position_independent_swizzle_tensor(sSFA_);
    auto sSFB = as_position_independent_swizzle_tensor(sSFB_);

    typename KT::MMAConfig::TiledMma mma;
    auto tile_shape_mnk = tile_shape(mma);
    auto thr_mma = mma.get_thread_slice(thread_idx);
    auto accum = partition_fragment_C(mma, cute::take<0, 2>(typename KT::TileShape{}));
    auto tCrA = thr_mma.partition_fragment_A(sA(_, _, Int<0>{}));
    auto tCrB = thr_mma.partition_fragment_B(sB(_, _, Int<0>{}));

    auto s2r_copy_A = make_tiled_copy_A(typename KT::ABLoadConfig::SmemCopyAtomA{}, mma);
    auto s2r_thr_copy_A = s2r_copy_A.get_thread_slice(thread_idx);
    auto tXsA = s2r_thr_copy_A.partition_S(sA);
    auto tXrA = s2r_thr_copy_A.retile_D(tCrA);
    auto s2r_copy_B = make_tiled_copy_B(typename KT::ABLoadConfig::SmemCopyAtomB{}, mma);
    auto s2r_thr_copy_B = s2r_copy_B.get_thread_slice(thread_idx);
    auto tXsB = s2r_thr_copy_B.partition_S(sB);
    auto tXrB = s2r_thr_copy_B.retile_D(tCrB);

    auto s2r_copy_SFA = make_tiled_copy_impl(typename KT::SFConfig::SmemCopyAtomSF{},
                                             KT::SFConfig::get_layoutSFA_TV(mma),
                                             make_shape(size<0>(tile_shape(mma)), _1{}));
    auto s2r_thr_copy_SFA = s2r_copy_SFA.get_thread_slice(thread_idx);
    auto tXsSFA = s2r_thr_copy_SFA.partition_S(sSFA);
    auto tCrSFA = KT::SFConfig::partition_fragment_SFA(sSFA(_, _, Int<0>{}), thr_mma);
    auto tXrSFA = s2r_thr_copy_SFA.retile_D(tCrSFA);
    auto tCrSFA_frg = KT::SFConfig::transform_fragment_for_qmma(tCrSFA);

    auto s2r_copy_SFB = make_tiled_copy_impl(typename KT::SFConfig::SmemCopyAtomSF{},
                                             KT::SFConfig::get_layoutSFB_TV(mma),
                                             make_shape(size<1>(tile_shape(mma)), _1{}));
    auto s2r_thr_copy_SFB = s2r_copy_SFB.get_thread_slice(thread_idx);
    auto tXsSFB = s2r_thr_copy_SFB.partition_S(sSFB);
    auto tCrSFB = KT::SFConfig::partition_fragment_SFB(sSFB(_, _, Int<0>{}), thr_mma);
    auto tXrSFB = s2r_thr_copy_SFB.retile_D(tCrSFB);
    auto tCrSFB_frg = KT::SFConfig::transform_fragment_for_qmma(tCrSFB);

    using ElementB = typename KT::ElementBMma;
    constexpr bool kFp4B = cute::is_same_v<ElementB, cute::float_e2m1_t>;
    auto prepare_b_fragment = [&](auto&& b_regs) {
      // ldsm.b4x16_p64 leaves each unpacked FP4 nibble in the low bits of
      // an 8-bit lane. The MMA consumes it two bits higher. Recast to the
      // actual 32-bit register words explicitly; dispatching fp4_shift_B
      // through the CuTe scalar alias can silently select the no-op path.
      if constexpr (kFp4B) {
        auto b_words = cute::recast<uint32_t>(b_regs);
        cute::transform(b_words, [](uint32_t& value) { return value << 2; });
      }
    };

    cute::clear(accum);
    auto [ab_full_mbar, ab_empty_mbar, sf_full_mbar, sf_empty_mbar, store_empty_mbar] =
        get_mbarriers(shared_storage);
    auto* ab_full = ab_full_mbar;
    auto* ab_empty = ab_empty_mbar;
    auto* sf_full = sf_full_mbar;
    auto* sf_empty = sf_empty_mbar;

    int32_t k_tiles = cute::ceil_div(params.K, static_cast<int>(KT::kTileK));
    // Each thread's packed scale word per row (column) block holds the four
    // k-blocks of the tile, so the MMA can pick the byte itself.
    static_assert(cute::is_same_v<typename KT::ElementA, cute::float_e4m3_t> &&
                  KT::SFConfig::kTileK == KT::kTileK);
    using SfaBytes = decltype(tCrSFA_frg.layout());
    using SfbBytes = decltype(tCrSFB_frg.layout());
    static_assert(decltype(cute::size(tCrSFA))::value == decltype(cute::size<1>(accum))::value &&
                  decltype(cute::size(tCrSFB))::value == decltype(cute::size<2>(accum))::value);
    static_assert(SfaBytes{}(0, 1, 0, 0) == 4 && SfaBytes{}(0, 0, 1, 0) == 1 &&  // NOLINT
                  SfbBytes{}(0, 1, 0, 0) == 4 && SfbBytes{}(0, 0, 1, 0) == 1);   // NOLINT
    auto mainloop = [&](auto live) {
      constexpr int kLive = decltype(live)::value;
      for (int32_t k_tile = 0; k_tile < k_tiles; ++k_tile) {
        int stage = pipe.stage;
        sf_full[stage].wait(pipe.phase);
        if constexpr (kLive > 0) {
          cute::copy(s2r_copy_SFA, tXsSFA(_, _, _, stage), tXrSFA);
          cute::copy(s2r_copy_SFB, tXsSFB(_, _, _, stage), tXrSFB);
        }
        sf_empty[stage].arrive();

        ab_full[stage].wait(pipe.phase);
        if constexpr (kLive > 0) {
          cute::copy(s2r_copy_A, tXsA(_, _, _, stage), tXrA);
          cute::copy(s2r_copy_B, tXsB(_, _, _, stage), tXrB);
          prepare_b_fragment(tXrB);
        }
        ab_empty[stage].arrive();
        pipe.advance<KT::kStageAB>();

        if constexpr (KT::kUnionSmem) {
          if (k_tile == k_tiles - 1) {
            cutlass::arch::NamedBarrier::sync(KT::MMAConfig::kNumMathThreads, 0);
          }
        }

        if constexpr (kLive > 0) {
          common::qmma_live_rows_bytesel<kLive, ElementB>(tCrA, tCrSFA.data(), tCrB, tCrSFB.data(),
                                                          accum);
        }
      }
    };
    constexpr int kRowBlocks = decltype(cute::size<1>(accum))::value;
    int live_blocks =
        common::live_row_blocks(thr_mma, cute::take<0, 2>(typename KT::TileShape{}), accum,
                                m_boundary - m_offset - m_block_idx * KT::kTileM);
    common::with_live_row_blocks<kRowBlocks>(live_blocks, mainloop);

    scatter_store<KT>(params, shared_storage, accum, thread_idx, n_block_idx, scatter_slot,
                      store_empty_mbar);
  }

  CUTE_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    cudaGridDependencySynchronize();

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_predicate = cute::elect_one_sync();
    bool is_tma_thread = warp_idx == 0 && lane_predicate;

    if (is_tma_thread) {
      prefetch_tma_descriptors(params);
    }
    __syncthreads();

    auto [ab_full_mbar, ab_empty_mbar, sf_full_mbar, sf_empty_mbar, store_empty_mbar] =
        get_mbarriers(shared_storage);
    if (is_tma_thread) {
#pragma unroll
      for (uint32_t i = 0; i < KT::SFConfig::kStageSF; ++i) {
        sf_full_mbar[i].init(1);
        sf_empty_mbar[i].init(KT::MMAConfig::kNumMathThreads);
      }
#pragma unroll
      for (uint32_t i = 0; i < KT::kStageAB; ++i) {
        ab_full_mbar[i].init(1);
        ab_empty_mbar[i].init(KT::MMAConfig::kNumMathThreads);
      }
      store_empty_mbar[0].init(KT::MMAConfig::kNumMathThreads);
      shared_storage.sched.init_mbars(kNumSchedConsumers);
      shared_storage.scatter.init_mbars(KT::MMAConfig::kNumMathWarps);
      cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    if (warp_idx >= KT::MMAConfig::kNumMathWarps) {
      constexpr int sched_warp_idx = KT::MMAConfig::kNumMathWarps;
      constexpr int ab_warp_idx = sched_warp_idx + 1;
      constexpr int sf_warp_idx = ab_warp_idx + 1;

      if (warp_idx == sched_warp_idx) {
        Scheduler scheduler(params.M, params.N, params.num_experts,
                            params.cu_padded_tokens_per_group, params.row_counts);
        common::MoeSchedProducer<kNumSchedStages> sched_pipeline{shared_storage.sched,
                                                                 lane_predicate};
        uint32_t scatter_iter = 0;
        int32_t m_block_idx, n_block_idx;
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
          sched_pipeline.publish(
              common::MoeWorkTile{m_block_idx, n_block_idx, scheduler.get_expert_idx(m_block_idx),
                                  scheduler.get_m_offset(), scheduler.get_m_boundary(), 1});
          produce_scatter_rows(shared_storage.scatter, scatter_iter++, params,
                               scheduler.get_m_offset(), scheduler.get_m_boundary(), m_block_idx);
        }
        sched_pipeline.publish_sentinel();
      }
      if (warp_idx == ab_warp_idx) {
        PipeState ab_pipe{0, 1};
        uint32_t store_phase = 1;
        common::MoeSchedConsumer<kNumSchedStages> sched_pipeline{shared_storage.sched,
                                                                 lane_predicate};
        common::MoeWorkTile tile;
        while (sched_pipeline.get_next_tile(tile)) {
          if (lane_predicate) {
            auto blk_coord = cute::make_coord(tile.m_block, tile.n_block, tile.group);
            load_ab(params, shared_storage, blk_coord, tile.m_offset, ab_pipe, store_phase);
          }
        }
      }
      if (warp_idx == sf_warp_idx) {
        PipeState sf_pipe{0, 1};
        uint32_t store_phase = 1;
        common::MoeSchedConsumer<kNumSchedStages> sched_pipeline{shared_storage.sched,
                                                                 lane_predicate};
        common::MoeWorkTile tile;
        while (sched_pipeline.get_next_tile(tile)) {
          if (lane_predicate) {
            auto blk_coord = cute::make_coord(tile.m_block, tile.n_block, tile.group);
            load_sf(params, shared_storage, blk_coord, tile.m_offset, sf_pipe, store_phase);
          }
        }
      }
    } else {
      PipeState pipe{0, 0};
      common::MoeSchedConsumer<kNumSchedStages> sched_pipeline{shared_storage.sched,
                                                               lane_predicate};
      uint32_t scatter_iter = 0;
      common::MoeWorkTile tile;
      while (sched_pipeline.get_next_tile(tile)) {
        int scatter_stage = scatter_iter % kNumScatterStages;
        MoeScatterRowsSlot slot{
            shared_storage.scatter.rows[scatter_stage], shared_storage.scatter.full(scatter_stage),
            shared_storage.scatter.empty(scatter_stage), (scatter_iter / kNumScatterStages) & 1};
        ++scatter_iter;
        mma(params, shared_storage, tile.m_offset, tile.m_boundary, tile.m_block, tile.n_block,
            pipe, slot);
      }
    }
  }
};

}  // namespace down
}  // namespace hpc::fuse_moe::sm120::gemm
