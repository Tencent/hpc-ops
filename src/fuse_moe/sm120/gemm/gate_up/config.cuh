// Copyright (C) 2026 Tencent.

#pragma once

// clang-format off
#include <cutlass/arch/barrier.h>

#include <cstdint>
#include <type_traits>

#include <cute/tensor.hpp>
#include <cute/layout.hpp>

#include "src/fuse_moe/sm120/gemm/common/ab_tma_load.cuh"
#include "src/fuse_moe/sm120/gemm/common/mx_mma.cuh"
#include "src/fuse_moe/sm120/gemm/common/sf_tma_load.cuh"
// clang-format on

namespace hpc::fuse_moe::sm120::gemm {
namespace gate_up {

using namespace cute;  // NOLINT

// gate-up GEMM: MXFP8 A x MXFP4/MXFP8 B, SwiGLU + MXFP8 D. Gate/up each have a B and
// SFB slab. Does not use down GEMM's scatter store.
template <typename ElementB_, int TileM_ = 32, int TileN_ = 64, int Stages_ = 2>
struct GateUpGemmConfig {
  static_assert(TileM_ == 32 || TileM_ == 64, "gate-up GEMM TileM is 32 or 64");
  static_assert(TileN_ == 64 || TileN_ == 128, "gate-up GEMM TileN is 64 or 128");
  static_assert((TileN_ % 32) == 0, "gate-up GEMM quant block is 32 columns");

  using ElementA = cute::float_e4m3_t;
  using ElementBMma = ElementB_;
  using ElementBGmem = ElementBMma;
  using ElementB = ElementBGmem;
  using ElementAccum = float;
  using ElementD = cute::float_e4m3_t;

  static constexpr int kStageAB = Stages_;
  static constexpr int kTileM = TileM_;
  static constexpr int kTileN = TileN_;
  static constexpr int kTileK = 128;
  using TileShape = Shape<Int<kTileM>, Int<kTileN>, Int<kTileK>>;
  using ProblemShape = Shape<int, int, int, int>;

  using SFConfig = common::SfLoadConfig<kTileM, kTileN, Stages_>;
  using MMAConfig = common::MxMmaConfig<ElementB, kTileM, kTileN>;
  using ABLoadConfig = common::AbLoadConfig<ElementB, kTileM, kTileN, kTileK, kStageAB>;

  using FullBarrier = cutlass::arch::ClusterTransactionBarrier;
  using EmptyBarrier = cutlass::arch::ClusterBarrier;
  using ProducerBarrierType = FullBarrier::ValueType;

  // Lean gather and load warps leave the MMA warps enough of the register file
  // for the 512-thread CTA.
  static constexpr uint32_t LoadRegisterRequirement = 40;
  static constexpr uint32_t GatherRegisterRequirement = 32;
  static constexpr uint32_t MmaRegisterRequirement = 216;

  struct SharedStorageLoad : cute::aligned_struct<128, _0> {
    alignas(1024)
        cute::ArrayEngine<ElementA, cute::cosize_v<typename ABLoadConfig::SmemLayoutA>> smem_A;
    alignas(1024) cute::ArrayEngine<typename ABLoadConfig::ElementBSmem,
                                    cute::cosize_v<typename ABLoadConfig::SmemLayoutB>> smem_B_gate;
    alignas(1024) cute::ArrayEngine<typename ABLoadConfig::ElementBSmem,
                                    cute::cosize_v<typename ABLoadConfig::SmemLayoutB>> smem_B_up;
    cute::ArrayEngine<typename SFConfig::ElementSFLoad,
                      cute::cosize_v<typename SFConfig::SmemLayoutSFA>>
        smem_SFA;
    cute::ArrayEngine<typename SFConfig::ElementSFLoad,
                      cute::cosize_v<typename SFConfig::SmemLayoutSFB>>
        smem_SFB_gate;
    cute::ArrayEngine<typename SFConfig::ElementSFLoad,
                      cute::cosize_v<typename SFConfig::SmemLayoutSFB>>
        smem_SFB_up;
  };

  struct SharedStorageSmemEpi : cute::aligned_struct<128, _0> {
    alignas(1024) cute::ArrayEngine<float, kTileM * kTileN> smem_C;
  };

  struct SharedStorageRegisterEpi : cute::aligned_struct<128, _0> {
    alignas(16) cute::ArrayEngine<float, kTileM * kTileN / 8> smem_max;
  };

  struct SmemEpiSplit {
    SharedStorageLoad load;
    SharedStorageSmemEpi store;
  };

  // Staging the whole tile in SMEM is cheapest when it fits beside the
  // pipeline. Otherwise the epilogue works from registers and keeps only the
  // per-warp block maxima in SMEM, so the next tile's loads still run under it.
  static constexpr bool kRegisterEpilogue = sizeof(SmemEpiSplit) > common::kTensorSmemBudget;
  using SharedStorageEpi =
      std::conditional_t<kRegisterEpilogue, SharedStorageRegisterEpi, SharedStorageSmemEpi>;

  struct BarrierStorage {
    FullBarrier ab_full_mbar[kStageAB];
    EmptyBarrier ab_empty_mbar[kStageAB];
    FullBarrier sf_full_mbar[SFConfig::kStageSF];
    EmptyBarrier sf_empty_mbar[SFConfig::kStageSF];
    EmptyBarrier store_empty_mbar[1];
  };

  union TensorStorageUnion {
    SharedStorageLoad load;
    SharedStorageEpi store;
  };

  struct TensorStorageSplit {
    SharedStorageLoad load;
    SharedStorageEpi store;
  };

  // A separate epilogue buffer lets the next tile's loads run under this tile's
  // epilogue. Only the union fits when the pipeline already fills the SMEM.
  static constexpr bool kUnionSmem = sizeof(TensorStorageSplit) > common::kTensorSmemBudget;
  using TensorStorage = std::conditional_t<kUnionSmem, TensorStorageUnion, TensorStorageSplit>;
};

}  // namespace gate_up
}  // namespace hpc::fuse_moe::sm120::gemm
