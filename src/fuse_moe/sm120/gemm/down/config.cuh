// Copyright (C) 2026 Tencent.

#pragma once

// clang-format off
#include <cutlass/arch/barrier.h>
#include <cutlass/numeric_size.h>

#include <cstdint>
#include <type_traits>

#include <cute/arch/copy_sm90.hpp>
#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/layout.hpp>

#include "src/fuse_moe/sm120/gemm/common/ab_tma_load.cuh"
#include "src/fuse_moe/sm120/gemm/common/mx_mma.cuh"
#include "src/fuse_moe/sm120/gemm/common/sf_tma_load.cuh"
// clang-format on

namespace hpc::fuse_moe::sm120::gemm {
namespace down {

using namespace cute;  // NOLINT

// MXFP8 E4M3 activations x MXFP4 E2M1 (byte-packed) or MXFP8 E4M3 weights,
// UE8M0 block scales (vector 32). D is BF16 scatter.
template <typename ElementB_, int TileM_ = 32, int TileN_ = 128, int Stages_ = 4>
struct DownGemmConfig {
  using ElementA = cute::float_e4m3_t;
  // FP4 keeps the logical sub-byte type in the TMA API so K strides are in elements.
  using ElementBMma = ElementB_;
  using ElementBGmem = ElementBMma;
  using ElementB = ElementBGmem;
  using ElementAccum = float;
  using ElementD = cute::bfloat16_t;

  static constexpr int kStageAB = Stages_;

  static constexpr int kTileM = TileM_;
  static constexpr int kTileN = TileN_;
  static constexpr int kTileK = 128;
  using TileShape = Shape<Int<kTileM>, Int<kTileN>, Int<kTileK>>;
  using ProblemShape = Shape<int, int, int, int>;

  using SFConfig = common::SfLoadConfig<kTileM, kTileN, Stages_>;
  using MMAConfig = common::MxMmaConfig<ElementB, kTileM, kTileN>;
  using ABLoadConfig = common::AbLoadConfig<ElementB, kTileM, kTileN, kTileK, kStageAB>;

  struct StoreConfig {
    static constexpr int kSmemON = kTileN;

    using SmemAtomLayoutO = decltype(composition(
        Swizzle<3, 3, 3>{}, Layout<Shape<_8, Shape<_8, _8>>, Stride<_8, Stride<_1, _64>>>{}));
    using SmemLayoutO =
        decltype(tile_to_shape(SmemAtomLayoutO{}, Shape<Int<kTileM>, Int<kSmemON>>{}));
    using SmemCopyAtom = Copy_Atom<AutoVectorizingCopy, ElementD>;
    // Eight columns per thread, so each thread adds them with one 16-byte atomic.
    static constexpr int kScatterThreadsN = kSmemON / 8;
    using ScatterTiledCopy =
        decltype(make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, ElementD>{},
                                 Layout<Shape<Int<256 / kScatterThreadsN>, Int<kScatterThreadsN>>,
                                        Stride<Int<kScatterThreadsN>, _1>>{},
                                 Layout<Shape<_1, _8>>{}));

    struct SharedStorage : cute::aligned_struct<128, _0> {
      alignas(1024) cute::ArrayEngine<ElementD, cute::cosize_v<SmemLayoutO>> smem_O;
    };
  };

  struct SharedStorageLoad : cute::aligned_struct<128, _0> {
    alignas(1024)
        cute::ArrayEngine<ElementA, cute::cosize_v<typename ABLoadConfig::SmemLayoutA>> smem_A;
    alignas(1024) cute::ArrayEngine<typename ABLoadConfig::ElementBSmem,
                                    cute::cosize_v<typename ABLoadConfig::SmemLayoutB>> smem_B;
    cute::ArrayEngine<typename SFConfig::ElementSFLoad,
                      cute::cosize_v<typename SFConfig::SmemLayoutSFA>>
        smem_SFA;
    cute::ArrayEngine<typename SFConfig::ElementSFLoad,
                      cute::cosize_v<typename SFConfig::SmemLayoutSFB>>
        smem_SFB;
  };

  using FullBarrier = cutlass::arch::ClusterTransactionBarrier;
  using EmptyBarrier = cutlass::arch::ClusterBarrier;
  using ProducerBarrierType = FullBarrier::ValueType;

  struct BarrierStorage {
    FullBarrier ab_full_mbar[kStageAB];
    EmptyBarrier ab_empty_mbar[kStageAB];
    FullBarrier sf_full_mbar[SFConfig::kStageSF];
    EmptyBarrier sf_empty_mbar[SFConfig::kStageSF];
    EmptyBarrier store_empty_mbar[1];
  };

  union TensorStorageUnion {
    SharedStorageLoad load;
    typename StoreConfig::SharedStorage store;
  };

  struct TensorStorageSplit {
    SharedStorageLoad load;
    typename StoreConfig::SharedStorage store;
  };

  // A separate store buffer lets the next tile's loads run under this tile's
  // store. Only the union fits when the pipeline already fills the SMEM.
  static constexpr bool kUnionSmem = sizeof(TensorStorageSplit) > common::kTensorSmemBudget;
};

}  // namespace down
}  // namespace hpc::fuse_moe::sm120::gemm
