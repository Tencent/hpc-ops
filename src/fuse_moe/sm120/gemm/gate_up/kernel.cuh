// Copyright (C) 2026 Tencent.

#pragma once

// clang-format off
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/cutlass.h>

#include <type_traits>

#include <cute/algorithm/tensor_algorithms.hpp>
#include <cute/atom/mma_traits_sm120.hpp>

#include "src/fuse_moe/sm120/common/quant.cuh"
#include "src/fuse_moe/sm120/gemm/common/ab_tma_load.cuh"
#include "src/fuse_moe/sm120/gemm/common/live_rows.cuh"
#include "src/fuse_moe/sm120/gemm/common/qmma.cuh"
#include "src/fuse_moe/sm120/gemm/common/pipe_state.cuh"
#include "src/fuse_moe/sm120/gemm/common/scheduler.cuh"
#include "src/fuse_moe/sm120/gemm/gate_up/config.cuh"
#include "src/fuse_moe/sm120/gemm/gate_up/tma_gather4.cuh"
#include "src/utils/utils.cuh"
// clang-format on

namespace hpc::fuse_moe::sm120::gemm {
namespace gate_up {

using namespace cute;  // NOLINT

namespace fused_epi {

using hpc::fuse_moe::sm120::pack_e4m3_x4;
using hpc::fuse_moe::sm120::quantization_scale;
using hpc::fuse_moe::sm120::store_packed_scale;

constexpr int kFloatsPerLane = 8;
constexpr int kLanesPerMxBlock = hpc::fuse_moe::sm120::kScaleVectorSize / kFloatsPerLane;

CUTE_DEVICE float bf16_round_trip(float value) {
  return __bfloat162float(__float2bfloat16_rn(value));
}

CUTE_DEVICE float swiglu_pair(float gate, float up, float swiglu_limit) {
  gate = bf16_round_trip(gate);
  up = bf16_round_trip(up);
  if (swiglu_limit >= 0.0f) {
    gate = fminf(gate, swiglu_limit);
    up = fminf(fmaxf(-swiglu_limit, up), swiglu_limit);
  }
  return hpc::silu(gate) * up;
}

template <class Tensor>
CUTE_DEVICE void shift_fp4_b_regs(Tensor&& b_regs) {
  auto b_words = cute::recast<uint32_t>(b_regs);
  cute::transform(b_words, [](uint32_t& value) { return value << 2; });
}

}  // namespace fused_epi

template <typename KT, bool kHinted = false>
struct GateUpGemmKernel {
  static_assert(KT::SFConfig::kStageSF == KT::kStageAB, "one SF stage per AB stage");

  using ElementB = typename KT::ElementBMma;
  static constexpr bool kFp4B = cute::is_same_v<ElementB, cute::float_e2m1_t>;

  static constexpr int kNumTMAThreads = 128;
  static constexpr int kNumGatherThreads = tma_gather4::kGather4Warps * kWarpSize;
  static constexpr int kNumMathThreads = KT::MMAConfig::kNumMathThreads;
  static constexpr int MaxThreadsPerBlock = kNumTMAThreads + kNumGatherThreads + kNumMathThreads;
  static constexpr int MinBlocksPerMultiprocessor = 1;
  static constexpr int kNumSchedStages = 2;
  static constexpr int kNumSchedConsumers =
      KT::MMAConfig::kNumMathWarps + 3 + tma_gather4::kGather4Warps;

  using Scheduler = common::MoeScheduler<KT::kTileM, KT::kTileN>;
  using ProblemShape = typename KT::ProblemShape;
  using FullBarrier = typename KT::FullBarrier;
  using EmptyBarrier = typename KT::EmptyBarrier;
  using ProducerBarrierType = typename FullBarrier::ValueType;

  static constexpr uint32_t TmaTransactionBytesAB =
      KT::ABLoadConfig::TmaTransactionBytesA + 2 * KT::ABLoadConfig::TmaTransactionBytesB;
  static constexpr uint32_t TmaTransactionBytesSF =
      KT::SFConfig::TmaTransactionBytesSFA + 2 * KT::SFConfig::TmaTransactionBytesSFB;

  struct Params {
    typename KT::ABLoadConfig::TMA_B tma_load_b;
    typename KT::SFConfig::TMA_SFA tma_load_sfa;
    typename KT::SFConfig::TMA_SFB tma_load_sfb;
    tma_gather4::TmaGather4A tma_gather_a;
    typename KT::ElementD* ptr_D;
    int32_t* ptr_SFD;
    int M;
    int N;
    int K;
    int num_experts;
    int scale_m_padded;
    float swiglu_limit;
    int32_t const* cu_padded_tokens_per_group;
    int32_t const* row_counts;
    int32_t const* row_to_route;
    int num_tokens;
    int num_topk;
    int4* zero_dst;
    int64_t zero_vectors;
    int64_t zero_chunk_vectors;
    uint64_t cache_hint_a;
    uint64_t cache_hint_b;
  };

  // A is token-major E4M3 [num_tokens, K], gathered row by row through
  // row_to_route, which must not be null.
  struct Arguments {
    typename KT::ElementA* ptr_A;
    typename KT::ElementB* ptr_B;
    typename KT::ABLoadConfig::StrideB dB;
    typename KT::SFConfig::ElementSFLoad* ptr_SFA;
    typename KT::SFConfig::ElementSFLoad* ptr_SFB;
    typename KT::ElementD* ptr_D;
    int32_t* ptr_SFD;
    int32_t const* cu_padded_tokens_per_group;
    int32_t const* row_counts;
    int32_t const* row_to_route;
    int num_tokens;
    int num_topk;
    float swiglu_limit;
    int4* zero_dst;
    int64_t zero_vectors;
    int64_t zero_chunk_vectors;
    uint64_t cache_hint_a = static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL);
    uint64_t cache_hint_b = static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL);
  };

  struct SharedStorage {
    typename KT::TensorStorage tensors;
    alignas(16) typename KT::BarrierStorage barriers;
    alignas(8) common::MoeSchedStorage<kNumSchedStages> sched;
  };
  static constexpr int kSmemSize = static_cast<int>(sizeof(SharedStorage));

  static Params to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args) {
    auto [M, N, K, num_experts] = problem_shape;
    auto tma_load_b = common::make_b_tma_descriptor<KT>(args.ptr_B, args.dB, 2 * N, K, num_experts);
    auto [tma_load_sfa, tma_load_sfb] =
        common::make_sf_tma_descriptors<KT>(args.ptr_SFA, args.ptr_SFB, M, 2 * N, K, num_experts);
    auto tma_gather_a = tma_gather4::make_tma_gather4_a_copy(args.ptr_A, args.num_tokens, K);
    int scale_m_padded =
        KT::SFConfig::get_tma_aligned_size(packed_scale_expert_base(M, num_experts));
    return {tma_load_b,
            tma_load_sfa,
            tma_load_sfb,
            tma_gather_a,
            args.ptr_D,
            args.ptr_SFD,
            M,
            N,
            K,
            num_experts,
            scale_m_padded,
            args.swiglu_limit,
            args.cu_padded_tokens_per_group,
            args.row_counts,
            args.row_to_route,
            args.num_tokens,
            args.num_topk,
            args.zero_dst,
            args.zero_vectors,
            args.zero_chunk_vectors,
            args.cache_hint_a,
            args.cache_hint_b};
  }

  static dim3 get_grid_shape(int num_sms) { return dim3(num_sms, 1, 1); }

  static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }

  CUTE_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
    cute::prefetch_tma_descriptor(params.tma_load_b.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_sfa.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_sfb.get_tma_descriptor());
    tma_gather4::prefetch_tma_gather4_descriptor(params.tma_gather_a.get_tma_descriptor());
  }

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

  // Loads the gate and up B slabs and arms the AB barrier for the A bytes the
  // gather warps add through gather4.
  template <typename BlkCoord>
  CUTE_DEVICE static void load_b(Params const& params, SharedStorage& shared_storage,
                                 BlkCoord const& blk_coord, int32_t k_tiles, PipeState& ab_pipe,
                                 uint32_t& store_phase) {
    auto [m_block_idx, n_block_idx, expert_idx] = blk_coord;
    (void)m_block_idx;
    using X = Underscore;

    auto mB =
        params.tma_load_b.get_tma_tensor(make_shape(2 * params.N, params.K, params.num_experts));
    auto mB_gate = mB;
    auto mB_up = domain_offset(make_coord(params.N, 0, 0), mB);
    auto gB_gate_nkl =
        local_tile(mB_gate, typename KT::TileShape{}, make_coord(_, _, _), Step<X, _1, _1>{});
    auto gB_up_nkl =
        local_tile(mB_up, typename KT::TileShape{}, make_coord(_, _, _), Step<X, _1, _1>{});
    auto gB_gate = gB_gate_nkl(_, _, n_block_idx, _, expert_idx);
    auto gB_up = gB_up_nkl(_, _, n_block_idx, _, expert_idx);

    auto block_tma_b = params.tma_load_b.get_slice(0);
    auto tBgB_gate = block_tma_b.partition_S(gB_gate);
    auto tBgB_up = block_tma_b.partition_S(gB_up);

    auto sB_gate = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_B_gate.begin()),
                    typename KT::ABLoadConfig::SmemLayoutB{}));
    auto sB_up = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_B_up.begin()),
                    typename KT::ABLoadConfig::SmemLayoutB{}));
    auto tBsB_gate = block_tma_b.partition_D(sB_gate);
    auto tBsB_up = block_tma_b.partition_D(sB_up);

    auto [ab_full_mbar, ab_empty_mbar, sf_full_mbar, sf_empty_mbar, store_empty_mbar] =
        get_mbarriers(shared_storage);
    (void)sf_full_mbar;
    (void)sf_empty_mbar;

    if constexpr (KT::kUnionSmem) {
      store_empty_mbar[0].wait(store_phase);
      store_phase ^= 1;
    }

    for (int32_t k_tile = 0; k_tile < k_tiles; ++k_tile) {
      int stage = ab_pipe.stage;
      ab_empty_mbar[stage].wait(ab_pipe.phase);
      auto& ab_full_barrier = ab_full_mbar[stage];
      ab_full_barrier.arrive_and_expect_tx(TmaTransactionBytesAB);
      auto tma_copy_b = params.tma_load_b.with(
          *recast_ptr<ProducerBarrierType>(&ab_full_barrier), 0,
          kHinted ? static_cast<cute::TMA::CacheHintSm90>(params.cache_hint_b)
                  : cute::TMA::CacheHintSm90::EVICT_NORMAL);
      cute::copy(tma_copy_b, tBgB_gate(_, _, _, k_tile), tBsB_gate(_, _, _, stage));
      cute::copy(tma_copy_b, tBgB_up(_, _, _, k_tile), tBsB_up(_, _, _, stage));
      ab_pipe.advance<KT::kStageAB>();
    }
  }

  template <typename BlkCoord>
  CUTE_DEVICE static void load_sf(Params const& params, SharedStorage& shared_storage,
                                  BlkCoord const& blk_coord, int32_t m_offset, int32_t k_tiles,
                                  PipeState& sf_pipe, uint32_t& store_phase) {
    auto [m_block_idx, n_block_idx, expert_idx] = blk_coord;
    using X = Underscore;

    auto sf_shape = make_shape(params.M, 2 * params.N, params.K, 1);
    auto sfa_layout = KT::SFConfig::deduce_sfa_layout(sf_shape, params.num_experts);
    auto mSFA_full = params.tma_load_sfa.get_tma_tensor(shape(sfa_layout));
    int32_t sf_m_offset = packed_scale_expert_base(m_offset, expert_idx);
    auto mSFA = domain_offset(make_coord(sf_m_offset, 0, 0), mSFA_full);
    auto gSFA_mkl = local_tile(mSFA, typename KT::SFConfig::ScaleTileShape{}, make_coord(_, _, _),
                               Step<_1, X, _1>{});
    auto gSFA = gSFA_mkl(_, _, m_block_idx, _, Int<0>{});

    auto sfb_shape = make_shape(params.M, 2 * params.N, params.K, params.num_experts);
    auto sfb_layout = KT::SFConfig::deduce_sfb_layout(sfb_shape, params.num_experts);
    auto mSFB = params.tma_load_sfb.get_tma_tensor(shape(sfb_layout));
    auto mSFB_gate = mSFB;
    auto mSFB_up = domain_offset(make_coord(params.N, 0, 0), mSFB);
    auto gSFB_gate_nkl = local_tile(mSFB_gate, typename KT::SFConfig::ScaleTileShape{},
                                    make_coord(_, _, _), Step<X, _1, _1>{});
    auto gSFB_up_nkl = local_tile(mSFB_up, typename KT::SFConfig::ScaleTileShape{},
                                  make_coord(_, _, _), Step<X, _1, _1>{});
    auto gSFB_gate = gSFB_gate_nkl(_, _, n_block_idx, _, expert_idx);
    auto gSFB_up = gSFB_up_nkl(_, _, n_block_idx, _, expert_idx);

    auto block_tma_sfa = params.tma_load_sfa.get_slice(0);
    auto block_tma_sfb = params.tma_load_sfb.get_slice(0);
    auto tAgSFA = block_tma_sfa.partition_S(gSFA);
    auto tBgSFB_gate = block_tma_sfb.partition_S(gSFB_gate);
    auto tBgSFB_up = block_tma_sfb.partition_S(gSFB_up);

    auto sSFA = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFA.begin()),
                    typename KT::SFConfig::SmemLayoutSFA{}));
    auto sSFB_gate = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFB_gate.begin()),
                    typename KT::SFConfig::SmemLayoutSFB{}));
    auto sSFB_up = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFB_up.begin()),
                    typename KT::SFConfig::SmemLayoutSFB{}));
    auto tAsSFA = block_tma_sfa.partition_D(sSFA);
    auto tBsSFB_gate = block_tma_sfb.partition_D(sSFB_gate);
    auto tBsSFB_up = block_tma_sfb.partition_D(sSFB_up);

    auto [ab_full_mbar, ab_empty_mbar, sf_full_mbar, sf_empty_mbar, store_empty_mbar] =
        get_mbarriers(shared_storage);
    (void)ab_full_mbar;
    (void)ab_empty_mbar;

    if constexpr (KT::kUnionSmem) {
      store_empty_mbar[0].wait(store_phase);
      store_phase ^= 1;
    }

    for (int32_t k_tile = 0; k_tile < k_tiles; ++k_tile) {
      int stage = sf_pipe.stage;
      sf_empty_mbar[stage].wait(sf_pipe.phase);
      auto& sf_full_barrier = sf_full_mbar[stage];
      sf_full_barrier.arrive_and_expect_tx(TmaTransactionBytesSF);
      auto tma_copy_sfa =
          params.tma_load_sfa.with(*recast_ptr<ProducerBarrierType>(&sf_full_barrier));
      auto tma_copy_sfb =
          params.tma_load_sfb.with(*recast_ptr<ProducerBarrierType>(&sf_full_barrier));
      cute::copy(tma_copy_sfa, tAgSFA(_, _, _, k_tile), tAsSFA(_, _, _, stage));
      cute::copy(tma_copy_sfb, tBgSFB_gate(_, _, _, k_tile), tBsSFB_gate(_, _, _, stage));
      cute::copy(tma_copy_sfb, tBgSFB_up(_, _, _, k_tile), tBsSFB_up(_, _, _, stage));
      sf_pipe.advance<KT::SFConfig::kStageSF>();
    }
  }

  // SwiGLU staged through a float tile in SMEM, then one MXFP8 block per lane group.
  template <class Storage, class ThrMma, class Accum>
  CUTE_DEVICE static void smem_quant_epilogue(Params const& params, Storage& shared_storage,
                                              ThrMma const& thr_mma, Accum const& accum_gate,
                                              Accum const& accum_up, int thread_idx,
                                              int32_t m_offset, int32_t m_boundary,
                                              int32_t m_block_idx, int32_t n_block_idx,
                                              int32_t expert_idx) {
    using namespace fused_epi;  // NOLINT
    {
      using EpiLayout =
          Layout<Shape<Int<KT::kTileM>, Int<KT::kTileN>>, Stride<Int<KT::kTileN>, _1>>;
      auto sC =
          make_tensor(make_smem_ptr(shared_storage.tensors.store.smem_C.begin()), EpiLayout{});
      auto tCsC = thr_mma.partition_C(sC);
      CUTE_UNROLL
      for (int i = 0; i < cute::size(accum_gate); ++i) {
        tCsC(i) = swiglu_pair(accum_gate(i), accum_up(i), params.swiglu_limit);
      }
      cutlass::arch::NamedBarrier::sync(KT::MMAConfig::kNumMathThreads, 0);
    }
    constexpr int kTileM = KT::kTileM;
    constexpr int kTileN = KT::kTileN;
    constexpr int kMxBlocks = kTileN / hpc::fuse_moe::sm120::kScaleVectorSize;
    constexpr int kNumMathWarps = KT::MMAConfig::kNumMathWarps;

    float* sC = shared_storage.tensors.store.smem_C.begin();
    int warp = thread_idx / kWarpSize;
    int lane = thread_idx % kWarpSize;

    int32_t valid_rows = m_boundary - m_offset - m_block_idx * kTileM;
    valid_rows = valid_rows < 0 ? 0 : valid_rows;
    valid_rows = valid_rows > kTileM ? kTileM : valid_rows;
    int32_t valid_cols = params.N - n_block_idx * kTileN;
    valid_cols = valid_cols < 0 ? 0 : valid_cols;
    valid_cols = valid_cols > kTileN ? kTileN : valid_cols;

    uint8_t* scale_bytes = reinterpret_cast<uint8_t*>(params.ptr_SFD);
    int* output_words = reinterpret_cast<int*>(params.ptr_D);

    for (int row = warp; row < kTileM; row += kNumMathWarps) {
      int mx_block = lane / kLanesPerMxBlock;
      int lane_in_block = lane % kLanesPerMxBlock;
      int column =
          mx_block * hpc::fuse_moe::sm120::kScaleVectorSize + lane_in_block * kFloatsPerLane;

      float values[kFloatsPerLane];
#pragma unroll
      for (int i = 0; i < kFloatsPerLane; ++i) {
        values[i] = 0.0f;
      }
      if (mx_block < kMxBlocks && column < kTileN) {
        *reinterpret_cast<float4*>(&values[0]) =
            *reinterpret_cast<float4 const*>(sC + row * kTileN + column);
        *reinterpret_cast<float4*>(&values[4]) =
            *reinterpret_cast<float4 const*>(sC + row * kTileN + column + 4);
      }

      float block_max = 0.0f;
#pragma unroll
      for (int i = 0; i < kFloatsPerLane; ++i) {
        block_max = fmaxf(block_max, fabsf(values[i]));
      }
      block_max = hpc::warp_4lane_reduce_max_xor(block_max);

      uint8_t storage = 0;
      float inv_scale = 0.0f;
      quantization_scale(block_max, storage, inv_scale);

      bool live = row < valid_rows && mx_block < kMxBlocks && column < valid_cols;
      if (live) {
        int packed[2];
#pragma unroll
        for (int half = 0; half < kFloatsPerLane / 4; ++half) {
          packed[half] =
              pack_e4m3_x4(values[4 * half] * inv_scale, values[4 * half + 1] * inv_scale,
                           values[4 * half + 2] * inv_scale, values[4 * half + 3] * inv_scale);
        }
        int global_row = m_offset + m_block_idx * kTileM + row;
        int global_col = n_block_idx * kTileN + column;
        int* destination =
            output_words + (static_cast<int64_t>(global_row) * params.N + global_col) / 4;
        *reinterpret_cast<int2*>(destination) = *reinterpret_cast<int2 const*>(packed);
        if (lane_in_block == 0) {
          int scale_row = hpc::fuse_moe::sm120::packed_scale_row(global_row, m_offset, expert_idx);
          store_packed_scale(scale_bytes, params.scale_m_padded, scale_row, global_col, storage);
        }
      }
    }
  }

  // SwiGLU in registers, then MXFP8 requant straight to global memory. A row's
  // 32-column block spans the four N warps (8 columns each), so only their
  // partial maxima go through shared memory.
  template <class Storage, class ThrMma, class Accum>
  CUTE_DEVICE static void register_quant_epilogue(Params const& params, Storage& shared_storage,
                                                  ThrMma const& thr_mma, Accum& accum_gate,
                                                  Accum const& accum_up, int32_t m_offset,
                                                  int32_t m_boundary, int32_t m_block_idx,
                                                  int32_t n_block_idx, int32_t expert_idx) {
    using namespace fused_epi;  // NOLINT
    constexpr int kTileM = KT::kTileM;
    constexpr int kTileN = KT::kTileN;
    constexpr int kSlots = kTileN / 8;
    static_assert(decltype(cute::size<0>(accum_gate))::value == 4, "16x8 MMA C fragment");

    float* partial_max = shared_storage.tensors.store.smem_max.begin();
    auto tCcC = thr_mma.partition_C(
        cute::make_identity_tensor(cute::make_shape(cute::Int<kTileM>{}, cute::Int<kTileN>{})));
    int lane = static_cast<int>(threadIdx.x) % kWarpSize;

    CUTE_UNROLL
    for (int i = 0; i < cute::size(accum_gate); ++i) {
      accum_gate(i) = swiglu_pair(accum_gate(i), accum_up(i), params.swiglu_limit);
    }
    // Fragment value (c, h): column pair c of row half h of one 16x8 atom.
    CUTE_UNROLL
    for (int mi = 0; mi < cute::size<1>(accum_gate); ++mi) {
      CUTE_UNROLL
      for (int ni = 0; ni < cute::size<2>(accum_gate); ++ni) {
        CUTE_UNROLL
        for (int h = 0; h < 2; ++h) {
          float m = fmaxf(fabsf(accum_gate(2 * h, mi, ni)), fabsf(accum_gate(2 * h + 1, mi, ni)));
          m = hpc::warp_4lane_reduce_max_xor(m);
          auto coord = tCcC(2 * h, mi, ni);
          if (lane % 4 == 0) {
            partial_max[cute::get<0>(coord) * kSlots + cute::get<1>(coord) / 8] = m;
          }
        }
      }
    }
    cutlass::arch::NamedBarrier::sync(KT::MMAConfig::kNumMathThreads, 0);

    int32_t valid_rows = m_boundary - m_offset - m_block_idx * kTileM;
    int32_t valid_cols = params.N - n_block_idx * kTileN;
    uint8_t* scale_bytes = reinterpret_cast<uint8_t*>(params.ptr_SFD);
    auto* output = reinterpret_cast<__nv_fp8x2_storage_t*>(params.ptr_D);
    CUTE_UNROLL
    for (int mi = 0; mi < cute::size<1>(accum_gate); ++mi) {
      CUTE_UNROLL
      for (int ni = 0; ni < cute::size<2>(accum_gate); ++ni) {
        CUTE_UNROLL
        for (int h = 0; h < 2; ++h) {
          auto coord = tCcC(2 * h, mi, ni);
          int row = cute::get<0>(coord);
          int col = cute::get<1>(coord);
          float4 parts = *reinterpret_cast<float4 const*>(partial_max + row * kSlots +
                                                          col / kScaleVectorSize * 4);
          float block_max = fmaxf(fmaxf(parts.x, parts.y), fmaxf(parts.z, parts.w));
          uint8_t storage = 0;
          float inv_scale = 0.0f;
          quantization_scale(block_max, storage, inv_scale);
          if (row < valid_rows && col < valid_cols) {
            int global_row = m_offset + m_block_idx * kTileM + row;
            int global_col = n_block_idx * kTileN + col;
            output[(static_cast<int64_t>(global_row) * params.N + global_col) / 2] =
                __nv_cvt_float2_to_fp8x2(make_float2(accum_gate(2 * h, mi, ni) * inv_scale,
                                                     accum_gate(2 * h + 1, mi, ni) * inv_scale),
                                         __NV_SATFINITE, __NV_E4M3);
            if (col % kScaleVectorSize == 0) {
              int scale_row =
                  hpc::fuse_moe::sm120::packed_scale_row(global_row, m_offset, expert_idx);
              store_packed_scale(scale_bytes, params.scale_m_padded, scale_row, global_col,
                                 storage);
            }
          }
        }
      }
    }
  }

  CUTE_DEVICE
  static void mma(Params const& params, SharedStorage& shared_storage, int32_t k_tiles,
                  int32_t m_offset, int32_t m_boundary, int32_t m_block_idx, int32_t n_block_idx,
                  int32_t expert_idx, PipeState& pipe) {
    int thread_idx = static_cast<int>(threadIdx.x);

    auto sA = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_A.begin()),
                    typename KT::ABLoadConfig::SmemLayoutA{}));
    auto sB_gate = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_B_gate.begin()),
                    typename KT::ABLoadConfig::SmemLayoutB{}));
    auto sB_up = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_B_up.begin()),
                    typename KT::ABLoadConfig::SmemLayoutB{}));
    auto sSFA = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFA.begin()),
                    typename KT::SFConfig::SmemLayoutSFA{}));
    auto sSFB_gate = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFB_gate.begin()),
                    typename KT::SFConfig::SmemLayoutSFB{}));
    auto sSFB_up = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFB_up.begin()),
                    typename KT::SFConfig::SmemLayoutSFB{}));

    typename KT::MMAConfig::TiledMma mma;
    auto thr_mma = mma.get_thread_slice(thread_idx);
    auto accum_gate = partition_fragment_C(mma, take<0, 2>(typename KT::TileShape{}));
    auto accum_up = partition_fragment_C(mma, take<0, 2>(typename KT::TileShape{}));
    auto tCrA = thr_mma.partition_fragment_A(sA(_, _, Int<0>{}));
    auto tCrB_gate = thr_mma.partition_fragment_B(sB_gate(_, _, Int<0>{}));
    auto tCrB_up = thr_mma.partition_fragment_B(sB_up(_, _, Int<0>{}));

    auto s2r_copy_A = make_tiled_copy_A(typename KT::ABLoadConfig::SmemCopyAtomA{}, mma);
    auto s2r_thr_copy_A = s2r_copy_A.get_thread_slice(thread_idx);
    auto tXsA = s2r_thr_copy_A.partition_S(sA);
    auto tXrA = s2r_thr_copy_A.retile_D(tCrA);
    auto s2r_copy_B = make_tiled_copy_B(typename KT::ABLoadConfig::SmemCopyAtomB{}, mma);
    auto s2r_thr_copy_B = s2r_copy_B.get_thread_slice(thread_idx);
    auto tXsB_gate = s2r_thr_copy_B.partition_S(sB_gate);
    auto tXsB_up = s2r_thr_copy_B.partition_S(sB_up);
    auto tXrB_gate = s2r_thr_copy_B.retile_D(tCrB_gate);
    auto tXrB_up = s2r_thr_copy_B.retile_D(tCrB_up);

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
    auto tXsSFB_gate = s2r_thr_copy_SFB.partition_S(sSFB_gate);
    auto tXsSFB_up = s2r_thr_copy_SFB.partition_S(sSFB_up);
    auto tCrSFB_gate = KT::SFConfig::partition_fragment_SFB(sSFB_gate(_, _, Int<0>{}), thr_mma);
    auto tCrSFB_up = KT::SFConfig::partition_fragment_SFB(sSFB_up(_, _, Int<0>{}), thr_mma);
    auto tXrSFB_gate = s2r_thr_copy_SFB.retile_D(tCrSFB_gate);
    auto tXrSFB_up = s2r_thr_copy_SFB.retile_D(tCrSFB_up);
    auto tCrSFB_gate_frg = KT::SFConfig::transform_fragment_for_qmma(tCrSFB_gate);
    auto tCrSFB_up_frg = KT::SFConfig::transform_fragment_for_qmma(tCrSFB_up);

    cute::clear(accum_gate);
    cute::clear(accum_up);
    auto [ab_full_mbar, ab_empty_mbar, sf_full_mbar, sf_empty_mbar, store_empty_mbar] =
        get_mbarriers(shared_storage);
    auto* ab_full = ab_full_mbar;
    auto* ab_empty = ab_empty_mbar;
    auto* sf_full = sf_full_mbar;
    auto* sf_empty = sf_empty_mbar;

    int live_blocks =
        common::live_row_blocks(thr_mma, take<0, 2>(typename KT::TileShape{}), accum_gate,
                                m_boundary - m_offset - m_block_idx * KT::kTileM);
    constexpr int kRowBlocks = decltype(cute::size<1>(accum_gate))::value;
    // Each thread's packed scale word per row (column) block holds the four
    // k-blocks of the tile, so the MMA can pick the byte itself.
    static_assert(cute::is_same_v<typename KT::ElementA, cute::float_e4m3_t> &&
                  KT::SFConfig::kTileK == KT::kTileK);
    using SfaBytes = decltype(tCrSFA_frg.layout());
    using SfbBytes = decltype(tCrSFB_gate_frg.layout());
    static_assert(decltype(cute::size(tCrSFA))::value == kRowBlocks &&
                  decltype(cute::size(tCrSFB_gate))::value ==
                      decltype(cute::size<2>(accum_gate))::value);
    static_assert(SfaBytes{}(0, 1, 0, 0) == 4 && SfaBytes{}(0, 0, 1, 0) == 1 &&  // NOLINT
                  SfbBytes{}(0, 1, 0, 0) == 4 && SfbBytes{}(0, 0, 1, 0) == 1);   // NOLINT
    common::with_live_row_blocks<kRowBlocks>(live_blocks, [&](auto live) {
      constexpr int kLive = decltype(live)::value;
      for (int32_t k_tile = 0; k_tile < k_tiles; ++k_tile) {
        int stage = pipe.stage;
        sf_full[stage].wait(pipe.phase);
        if constexpr (kLive > 0) {
          cute::copy(s2r_copy_SFA, tXsSFA(_, _, _, stage), tXrSFA);
          cute::copy(s2r_copy_SFB, tXsSFB_gate(_, _, _, stage), tXrSFB_gate);
          cute::copy(s2r_copy_SFB, tXsSFB_up(_, _, _, stage), tXrSFB_up);
        }
        sf_empty[stage].arrive();

        ab_full[stage].wait(pipe.phase);
        if constexpr (kLive > 0) {
          cute::copy(s2r_copy_A, tXsA(_, _, _, stage), tXrA);
          cute::copy(s2r_copy_B, tXsB_gate(_, _, _, stage), tXrB_gate);
          cute::copy(s2r_copy_B, tXsB_up(_, _, _, stage), tXrB_up);
          if constexpr (kFp4B) {
            fused_epi::shift_fp4_b_regs(tXrB_gate);
            fused_epi::shift_fp4_b_regs(tXrB_up);
          }
        }
        ab_empty[stage].arrive();
        pipe.advance<KT::kStageAB>();

        // Also orders the previous tile's epilogue reads of smem_C or smem_max before this
        // tile's writes when the epilogue has its own buffer.
        if (k_tile == k_tiles - 1) {
          cutlass::arch::NamedBarrier::sync(KT::MMAConfig::kNumMathThreads, 0);
        }

        if constexpr (kLive > 0) {
          common::qmma_live_rows_bytesel<kLive, ElementB>(tCrA, tCrSFA.data(), tCrB_gate,
                                                          tCrSFB_gate.data(), accum_gate);
          common::qmma_live_rows_bytesel<kLive, ElementB>(tCrA, tCrSFA.data(), tCrB_up,
                                                          tCrSFB_up.data(), accum_up);
        }
      }
    });

    if constexpr (KT::kRegisterEpilogue) {
      register_quant_epilogue(params, shared_storage, thr_mma, accum_gate, accum_up, m_offset,
                              m_boundary, m_block_idx, n_block_idx, expert_idx);
    } else {
      smem_quant_epilogue(params, shared_storage, thr_mma, accum_gate, accum_up, thread_idx,
                          m_offset, m_boundary, m_block_idx, n_block_idx, expert_idx);
    }

    if constexpr (KT::kUnionSmem) {
      store_empty_mbar[0].arrive();
    }
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
      cute::for_each(cute::make_int_sequence<KT::kStageAB>{}, [&](auto stage) {
        ab_full_mbar[stage].init(1);
        ab_empty_mbar[stage].init(KT::MMAConfig::kNumMathThreads);
      });
      cute::for_each(cute::make_int_sequence<KT::SFConfig::kStageSF>{}, [&](auto stage) {
        sf_full_mbar[stage].init(1);
        sf_empty_mbar[stage].init(KT::MMAConfig::kNumMathThreads);
      });
      store_empty_mbar[0].init(KT::MMAConfig::kNumMathThreads);
      shared_storage.sched.init_mbars(kNumSchedConsumers);
      cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    int32_t k_tiles = cute::ceil_div(params.K, static_cast<int>(KT::kTileK));

    if (warp_idx >= KT::MMAConfig::kNumMathWarps) {
      constexpr int sched_warp_idx = KT::MMAConfig::kNumMathWarps;
      constexpr int ab_warp_idx = sched_warp_idx + 1;
      constexpr int sf_warp_idx = ab_warp_idx + 1;
      constexpr int zero_warp_idx = sf_warp_idx + 1;
      constexpr int gather_warp_begin = KT::MMAConfig::kNumMathWarps + 4;

      if (warp_idx >= gather_warp_begin) {
        cutlass::arch::warpgroup_reg_dealloc<KT::GatherRegisterRequirement>();
        PipeState ab_pipe{0, 1};
        uint32_t store_phase = 1;
        int gather_warp = warp_idx - gather_warp_begin;
        common::MoeSchedConsumer<kNumSchedStages> sched_pipeline{shared_storage.sched,
                                                                 lane_predicate};
        common::MoeWorkTile tile;
        while (sched_pipeline.get_next_tile(tile)) {
          int32_t valid_rows = tma_gather4::gather4_valid_rows(tile.m_offset, tile.m_block,
                                                               tile.m_boundary, KT::kTileM);
          tma_gather4::gather_warp_load_a<kHinted, KT::kTileM, KT::kStageAB>(
              params.tma_gather_a.get_tma_descriptor(), shared_storage.tensors.load.smem_A.begin(),
              ab_full_mbar, ab_empty_mbar, KT::kUnionSmem ? store_empty_mbar : nullptr, ab_pipe,
              store_phase, params.row_to_route, params.num_topk, tile.m_offset, tile.m_block,
              k_tiles, gather_warp, params.cache_hint_a, valid_rows);
        }
      } else {
        cutlass::arch::warpgroup_reg_dealloc<KT::LoadRegisterRequirement>();
      }

      if (warp_idx == sched_warp_idx) {
        Scheduler scheduler(params.M, params.N, params.num_experts,
                            params.cu_padded_tokens_per_group, params.row_counts);
        common::MoeSchedProducer<kNumSchedStages> sched_pipeline{shared_storage.sched,
                                                                 lane_predicate};
        int32_t m_block_idx;
        int32_t n_block_idx;
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
          sched_pipeline.publish(
              common::MoeWorkTile{m_block_idx, n_block_idx, scheduler.get_expert_idx(m_block_idx),
                                  scheduler.get_m_offset(), scheduler.get_m_boundary(), 1});
        }
        sched_pipeline.publish_sentinel();
      } else if (warp_idx == ab_warp_idx) {
        PipeState ab_pipe{0, 1};
        uint32_t store_phase = 1;
        common::MoeSchedConsumer<kNumSchedStages> sched_pipeline{shared_storage.sched,
                                                                 lane_predicate};
        common::MoeWorkTile tile;
        while (sched_pipeline.get_next_tile(tile)) {
          if (lane_predicate) {
            auto blk_coord = cute::make_coord(tile.m_block, tile.n_block, tile.group);
            load_b(params, shared_storage, blk_coord, k_tiles, ab_pipe, store_phase);
          }
        }
      } else if (warp_idx == zero_warp_idx) {
        // This CTA's share of down GEMM's output, zeroed while the
        // tensor pipe is busy instead of in a bandwidth-bound pass before.
        int64_t per_cta = (params.zero_vectors + gridDim.x - 1) / gridDim.x;
        int64_t next = cute::min(params.zero_vectors, static_cast<int64_t>(blockIdx.x) * per_cta);
        int64_t end = cute::min(params.zero_vectors, next + per_cta);
        int lane = static_cast<int>(threadIdx.x) % 32;
        auto zero_until = [&](int64_t stop) {
          for (int64_t i = next + lane; i < stop; i += 32) {
            __stcs(params.zero_dst + i, make_int4(0, 0, 0, 0));
          }
          next = stop;
        };
        common::MoeSchedConsumer<kNumSchedStages> sched_pipeline{shared_storage.sched,
                                                                 lane_predicate};
        common::MoeWorkTile tile;
        while (sched_pipeline.get_next_tile(tile)) {
          zero_until(cute::min(end, next + params.zero_chunk_vectors));
        }
        zero_until(end);
      } else if (warp_idx == sf_warp_idx) {
        PipeState sf_pipe{0, 1};
        uint32_t store_phase = 1;
        common::MoeSchedConsumer<kNumSchedStages> sched_pipeline{shared_storage.sched,
                                                                 lane_predicate};
        common::MoeWorkTile tile;
        while (sched_pipeline.get_next_tile(tile)) {
          if (lane_predicate) {
            auto blk_coord = cute::make_coord(tile.m_block, tile.n_block, tile.group);
            load_sf(params, shared_storage, blk_coord, tile.m_offset, k_tiles, sf_pipe,
                    store_phase);
          }
        }
      }
    } else {
      cutlass::arch::warpgroup_reg_alloc<KT::MmaRegisterRequirement>();
      PipeState pipe{0, 0};
      common::MoeSchedConsumer<kNumSchedStages> sched_pipeline{shared_storage.sched,
                                                               lane_predicate};
      common::MoeWorkTile tile;
      while (sched_pipeline.get_next_tile(tile)) {
        auto blk_coord = cute::make_coord(tile.m_block, tile.n_block, tile.group);
        mma(params, shared_storage, k_tiles, tile.m_offset, tile.m_boundary,
            cute::get<0>(blk_coord), cute::get<1>(blk_coord), cute::get<2>(blk_coord), pipe);
      }
      cudaTriggerProgrammaticLaunchCompletion();
    }
  }
};

}  // namespace gate_up
}  // namespace hpc::fuse_moe::sm120::gemm
