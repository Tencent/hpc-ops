// Copyright (C) 2026 Tencent.
#ifndef SRC_FUSE_MOE_SM120_MXFP8_CONFIG_H_
#define SRC_FUSE_MOE_SM120_MXFP8_CONFIG_H_

#include <cuda.h>

#include <stdexcept>
#include <string>
#include <type_traits>

#include "cute/arch/copy_sm100.hpp"
#include "cute/atom/copy_traits_sm100.hpp"
#include "cute/atom/mma_traits_sm120.hpp"
#include "cute/tensor.hpp"
#include "cutlass/float_subbyte.h"
#include "src/utils/tma.cuh"

namespace hpc {
namespace fuse_moe {
namespace kernels {
namespace mxfp8 {

// BType is B's MMA type: float_e4m3_t (mxfp8) or float_e2m1_t (mxfp4). It only changes B's type in
// the instruction, same as cute's SM120_16x8x32_TN_VS<AType, BType, ...>.
template <typename BType>
struct SM120_16x8x32_TN_VS_BYTESEL {
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void fma(float &d0, float &d1, float &d2, float &d3, uint32_t const &a0,
                                   uint32_t const &a1, uint32_t const &a2, uint32_t const &a3,
                                   uint32_t const &b0, uint32_t const &b1, float const &c0,
                                   float const &c1, float const &c2, float const &c3,
                                   uint32_t const &sfa, uint32_t const &sfb,
                                   uint16_t const &byte_id) {
#if defined(CUTE_ARCH_MXF8F6F4_MMA_ENABLED)
    static constexpr uint16_t tid = 0;

    if constexpr (cute::is_same_v<BType, cutlass::float_e2m1_t>) {
      asm volatile(
          "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3."
          "e2m1.f32.ue8m0 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13},"
          "{%14},"
          "{%15, %16},"
          "{%17},"
          "{%18, %19};\n"
          : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2),
            "f"(c3), "r"(sfa), "h"(byte_id), "h"(tid), "r"(sfb), "h"(byte_id), "h"(tid));
    } else {
      asm volatile(
          "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3."
          "e4m3.f32.ue8m0 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13},"
          "{%14},"
          "{%15, %16},"
          "{%17},"
          "{%18, %19};\n"
          : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2),
            "f"(c3), "r"(sfa), "h"(byte_id), "h"(tid), "r"(sfb), "h"(byte_id), "h"(tid));
    }
#else
    CUTE_INVALID_CONTROL_PATH(
        "Attempting to use SM120_16x8x32_TN_VS_BYTESEL without CUTE_ARCH_MXF8F6F4_MMA_ENABLED");
#endif
  }
};

}  // namespace mxfp8
}  // namespace kernels
}  // namespace fuse_moe
}  // namespace hpc

namespace cute {

// An fp4 B still holds one element per byte in registers (mxf8f6f4's 8-bit container), so the
// A/B/C thread partitioning is fp8's SM120_16x8x32_TN either way.
template <typename BType>
struct MMA_Traits<hpc::fuse_moe::kernels::mxfp8::SM120_16x8x32_TN_VS_BYTESEL<BType>>
    : MMA_Traits<SM120_16x8x32_TN<float_e4m3_t, float_e4m3_t, float>> {
  uint32_t sfa_ = 0;
  uint32_t sfb_ = 0;
  uint16_t byte_id_ = 0;

  template <class TD, class DLayout, class TA, class ALayout, class TB, class BLayout, class TC,
            class CLayout>
  CUTE_HOST_DEVICE constexpr friend void mma_unpack(MMA_Traits const &traits,
                                                    Tensor<TD, DLayout> &D,
                                                    Tensor<TA, ALayout> const &A,
                                                    Tensor<TB, BLayout> const &B,
                                                    Tensor<TC, CLayout> const &C) {
    static_assert(is_rmem<TD>::value, "Expected registers in MMA_Atom::call");
    static_assert(is_rmem<TA>::value, "Expected registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected registers in MMA_Atom::call");
    static_assert(is_rmem<TC>::value, "Expected registers in MMA_Atom::call");

    Tensor rA = recast<uint32_t>(A);
    Tensor rB = recast<uint32_t>(B);
    Tensor rD = recast<float>(D);
    Tensor rC = recast<float>(C);
    CUTE_STATIC_ASSERT_V(size(rA) == Int<4>{});
    CUTE_STATIC_ASSERT_V(size(rB) == Int<2>{});
    CUTE_STATIC_ASSERT_V(size(rD) == Int<4>{});
    CUTE_STATIC_ASSERT_V(size(rC) == Int<4>{});

    hpc::fuse_moe::kernels::mxfp8::SM120_16x8x32_TN_VS_BYTESEL<BType>::fma(
        rD(0), rD(1), rD(2), rD(3), rA(0), rA(1), rA(2), rA(3), rB(0), rB(1), rC(0), rC(1), rC(2),
        rC(3), traits.sfa_, traits.sfb_, traits.byte_id_);
  }

  // Set the scales (u32) and the byte id for this MMA.
  CUTE_HOST_DEVICE constexpr MMA_Traits with(uint32_t sfa, uint32_t sfb, int byte_id) const {
    MMA_Traits traits = *this;
    traits.sfa_ = sfa;
    traits.sfb_ = sfb;
    traits.byte_id_ = static_cast<uint16_t>(byte_id);
    return traits;
  }
};

}  // namespace cute

namespace hpc {
namespace fuse_moe {

using namespace cute;  // NOLINT

template <typename TC_, typename TW_, int kTileM_, int kTileN_, int kTileK_>
struct GroupGemmMxfp8Config {
  using TA = uint8_t;
  // B's shared-memory element type; an fp4 B is expanded to one byte per element too, see TW.
  using TB = uint8_t;
  // B (weight) gmem type: uint8_t for mxfp8, float_e2m1_unpacksmem_t for mxfp4.
  // The latter makes make_tma_copy pick CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,
  // which expands every 16 fp4 (8 gmem bytes) to 16 shm bytes, so the shm layout, the
  // swizzle and TiledCopyB's partitioning all stay those of fp8.
  using TW = TW_;
  static constexpr bool kIsFp4W = sizeof_bits_v<TW> == 4;
  // Epilogue write-back type: fp8 (uint8) for gate_up, bf16 for down.
  using TC = TC_;
  using TSF = float_ue8m0_t;

  static constexpr bool kIsGateUp = sizeof(TC) == 1;
  static constexpr int kTileM = kTileM_;  // 32/64/128
  static constexpr int kTileN = kTileN_;  // 128, or 256 for gate_up
  static constexpr int kTileK = kTileK_;  // 128
  static constexpr int kSFVecSize = 32;
  static constexpr int kSFPerTile = kTileK / kSFVecSize;  // 4

  // Scales use the flashinfer layout_128x4.
  static constexpr int kSFBlockM = 128;
  static constexpr int kSFBlockBytes = kSFBlockM * kSFPerTile * sizeof(TSF);
  static constexpr int kNumSFB = kTileN / kSFBlockM;
  static constexpr int kSFABytesPerStage = kIsGateUp ? kTileM * kSFPerTile : kSFBlockBytes;
  static constexpr int kTileNC = kIsGateUp ? kTileN / 2 : kTileN;

  static constexpr int kStageABMax = 4;
  static constexpr int kShmBudget = 98 * 1024;
  // down adds into y from registers.
  static constexpr int kCBytes = kIsGateUp ? sizeof(TC) * kTileM * kTileNC : 0;
  static constexpr int kStageBytes = sizeof(TA) * kTileM * kTileK + sizeof(TB) * kTileN * kTileK +
                                     kSFABytesPerStage + kNumSFB * kSFBlockBytes;
  static constexpr int kStageABFit = (kShmBudget - kCBytes) / kStageBytes;
  static constexpr int kStageAB = kStageABFit > kStageABMax ? kStageABMax : kStageABFit;
  static_assert(kStageAB >= 2, "the pipeline needs at least 2 stages");
  static constexpr int kStageTask = 2;

  // shm A/B/C
  using SLayoutA = decltype(tile_to_shape(
      UMMA::Layout_K_SW128_Atom<TA>{}, make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStageAB>{})));
  using SLayoutB = decltype(tile_to_shape(
      UMMA::Layout_K_SW128_Atom<TB>{}, make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStageAB>{})));
  using CAtom = std::conditional_t<sizeof(TC) * kTileNC >= 128, UMMA::Layout_K_SW128_Atom<TC>,
                                   UMMA::Layout_K_SW64_Atom<TC>>;
  using SLayoutC = decltype(tile_to_shape(CAtom{}, make_shape(Int<kTileM>{}, Int<kTileNC>{})));

  // copybox
  using CopyBoxA = decltype(tile_to_shape(UMMA::Layout_K_SW128_Atom<TA>{},
                                          make_shape(Int<kTileM>{}, Int<kTileK>{})));
  using CopyBoxB = decltype(tile_to_shape(UMMA::Layout_K_SW128_Atom<TB>{},
                                          make_shape(Int<kTileN>{}, Int<kTileK>{})));
  using CopyBoxC = SLayoutC;

  // shm SFA/SFB
  // gate_up's SFA rows go to one warp's lanes kSFARowsPerLane at a time.
  static constexpr int kSFARowsPerLane = 4;
  static_assert(!kIsGateUp || (kTileM % kSFARowsPerLane == 0 && kTileM / kSFARowsPerLane <= 32),
                "one warp's lanes must cover A's kTileM scale rows");
  static_assert(kTileN % kSFBlockM == 0 && kSFPerTile == 4,
                "one tile of B's scales is a whole number of 128x4 blocks");
  using SLayoutSF128x4 =
      Layout<Shape<_4, _128, Int<kStageAB>>, Stride<_128, _1, Int<kSFBlockBytes>>>;
  using SLayoutSF128x4Read =
      Layout<Shape<_32, _16, Int<kStageAB>>, Stride<_16, _1, Int<kSFBlockBytes>>>;
  using CopyBoxSF = Layout<Shape<_4, _128>, Stride<_128, _1>>;

  using SLayoutSFARow = Layout<Shape<Int<kTileM>, Int<kSFPerTile>, Int<kStageAB>>,
                               Stride<Int<kSFPerTile>, _1, Int<kTileM * kSFPerTile>>>;
  using SLayoutSFA = std::conditional_t<kIsGateUp, SLayoutSFARow, SLayoutSF128x4>;
  using SLayoutSFARead = std::conditional_t<kIsGateUp, SLayoutSFARow, SLayoutSF128x4Read>;
  // (.., 128x4 block within the tile, stage)
  using SLayoutSFB = Layout<Shape<_4, _128, Int<kNumSFB>, Int<kStageAB>>,
                            Stride<_128, _1, Int<kSFBlockBytes>, Int<kNumSFB * kSFBlockBytes>>>;
  using SLayoutSFBRead = Layout<Shape<_32, _16, Int<kNumSFB>, Int<kStageAB>>,
                                Stride<_16, _1, Int<kSFBlockBytes>, Int<kNumSFB * kSFBlockBytes>>>;

  // mma
  static constexpr int kNumMmaWarps = 8;
  static constexpr int kMathThreads = 32 * kNumMmaWarps;
  static constexpr int kNumThreads = kMathThreads + 128;
  static constexpr int kMathRegisters = 232;
  static constexpr int kLoadRegisters = 40;
  static_assert(2 * kMathRegisters + kLoadRegisters <= 512, "CTA register pool exhausted");

  using WarpLayout = Layout<Shape<_2, _4, _1>, Stride<_4, _1, _0>>;
  // (coordinate within the atom, warp coordinate, repeated block within the warp)
  using PermM = Layout<Shape<_16, _2, Int<kTileM / 32>>, Stride<_1, _16, _32>>;
  using PermN = Layout<Shape<_8, _4, Int<kTileN / 32>>, Stride<_1, Int<kTileN / 4>, _8>>;
  using MmaOp = kernels::mxfp8::SM120_16x8x32_TN_VS_BYTESEL<
      std::conditional_t<kIsFp4W, cutlass::float_e2m1_t, float_e4m3_t>>;
  using TiledMma = decltype(make_tiled_mma(MmaOp{}, WarpLayout{}, Tile<PermM, PermN, _32>{}));

  // LDSM/STSM
  using TiledCopyA = decltype(make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, TA>{}, TiledMma{}));
  // fp4 uses ldmatrix .b8x16.b4x16_p64: it skips the padding and puts each fp4 in a byte's low 4
  // bits
  using LdsmOpB = std::conditional_t<kIsFp4W, SM100_SU4_DU8x16_x4_LDSM_N, SM75_U32x4_LDSM_N>;
  using TiledCopyB = decltype(make_tiled_copy_B(Copy_Atom<LdsmOpB, TB>{}, TiledMma{}));
  using StsmOp = std::conditional_t<(kTileM / 32) % 2 == 0, SM90_U32x4_STSM_N, SM90_U32x2_STSM_N>;
  using TiledCopyC = decltype(make_tiled_copy_C(Copy_Atom<StsmOp, TC>{}, TiledMma{}));

  static_assert((kTileM == 32 || kTileM == 64 || kTileM == 128) &&
                    (kTileN == 128 || (kIsGateUp && kTileN == 256)),
                "MMA permutation requires kTileM in {32, 64, 128} and kTileN = 128 (or 256 for "
                "gate_up)");
  static_assert(size(TiledMma{}) == kNumMmaWarps * 32, "MMA must use 8 warps");

  static constexpr int kShmABytes = sizeof_bits_v<TA> * cosize(SLayoutA{}) / 8;
  static constexpr int kShmBBytes = sizeof_bits_v<TB> * cosize(SLayoutB{}) / 8;
  static constexpr int kShmCBytes = kIsGateUp ? sizeof_bits_v<TC> * cosize(SLayoutC{}) / 8 : 0;
  static constexpr int kShmSFABytes = sizeof_bits_v<TSF> * cosize(SLayoutSFA{}) / 8;
  static constexpr int kShmSFBBytes = sizeof_bits_v<TSF> * cosize(SLayoutSFB{}) / 8;
  static constexpr int kShmSize =
      kShmABytes + kShmBBytes + kShmCBytes + kShmSFABytes + kShmSFBBytes;
  static_assert(kShmSize <= kShmBudget, "shm overflow");
};

// 64-row gate_up tiles go 256 wide, so each warp still holds 32x64.
template <int kTileM, typename TW>
using GateUpConfig = GroupGemmMxfp8Config<uint8_t, TW, kTileM, kTileM == 64 ? 256 : 128, 128>;
template <int kTileM, typename TW>
using DownConfig = GroupGemmMxfp8Config<cute::bfloat16_t, TW, kTileM, 128, 128>;

// W's gmem type: float_e2m1_unpacksmem_t for mxfp4 weights, uint8_t otherwise.
template <typename F>
void dispatch_w_type(bool is_fp4, F &&f) {
  if (is_fp4) {
    f(cutlass::detail::float_e2m1_unpacksmem_t{});
  } else {
    f(uint8_t{});
  }
}

// Calls f(Int<n>) for n clamped to [0, kMax], so each count gets straight-line code.
template <int kMax, typename F>
CUTE_DEVICE void dispatch_num_valid_m(int n, F &&f) {
  if constexpr (kMax > 0) {
    if (n < kMax) {
      dispatch_num_valid_m<kMax - 1>(n, f);
      return;
    }
  }
  f(Int<kMax>{});
}

template <typename F>
void dispatch_tile_m(int tile_m, F &&f) {
  switch (tile_m) {
    case 32:
      f(std::integral_constant<int, 32>{});
      break;
    case 64:
      f(std::integral_constant<int, 64>{});
      break;
    default:
      f(std::integral_constant<int, 128>{});
      break;
  }
}

}  // namespace fuse_moe
}  // namespace hpc

#endif  // SRC_FUSE_MOE_SM120_MXFP8_CONFIG_H_
