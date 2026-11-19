// Copyright (C) 2026 Tencent.
// decode_fp4x8 is adapted from vllm-project/humming bb9f595d3,
// include/humming/datatype/dequant_fused.cuh (Apache-2.0).
// Only the local MXFP4-to-E4M3 LUT decode is retained; no Humming runtime is used.

#ifndef SRC_GROUP_GEMM_SM90_CP_ASYNC_W4A8_OPERAND_CUH_
#define SRC_GROUP_GEMM_SM90_CP_ASYNC_W4A8_OPERAND_CUH_

#include <cuda_runtime.h>
#include <stdint.h>

#include "src/group_gemm/sm90/cp_async/config.h"

namespace hpc {
namespace group_gemm_cp_async {
namespace w4a8_detail {

template <int kTileM>
CUTE_HOST_DEVICE constexpr auto select_rs_mma() {
  if constexpr (kTileM == 8) {
    return cute::SM90_64x8x32_F32E4M3E4M3_RS_TN<>{};
  } else if constexpr (kTileM == 16) {
    return cute::SM90_64x16x32_F32E4M3E4M3_RS_TN<>{};
  } else if constexpr (kTileM == 32) {
    return cute::SM90_64x32x32_F32E4M3E4M3_RS_TN<>{};
  } else {
    static_assert(kTileM == 40);
    return cute::SM90_64x40x32_F32E4M3E4M3_RS_TN<>{};
  }
}

// A is the activation in the operator's naming. SwapAB puts weights into
// WGMMA register operand A, and the existing swizzled activation into operand B.
template <typename Tout, int kTileM, int kTileN = 64, int kTileK = 128, int kStages = 3>
struct W4A8GemmConfig
    : config::FP8GemmConfig<cute::float_e4m3_t, Tout, kTileM, kTileN, kTileK, kStages> {
  static_assert(kTileK == 128 && kTileN % 64 == 0);
  using Base = config::FP8GemmConfig<cute::float_e4m3_t, Tout, kTileM, kTileN, kTileK, kStages>;
  static constexpr int kWarpGroups = kTileN >= 256 || (kTileN == 128 && kTileM == 40) ? 2 : 1;
  static constexpr int kThreads = 128 * kWarpGroups;
  static constexpr int kPanelsPerWarpGroup = kTileN / (64 * kWarpGroups);
  static_assert(kTileN % (64 * kWarpGroups) == 0);
  using WarpGroupLayout = decltype(cute::make_layout(
      cute::make_shape(cute::Int<kWarpGroups>{}, cute::Int<1>{}, cute::Int<1>{})));
  using TiledMMA = decltype(cute::make_tiled_mma(select_rs_mma<kTileM>(), WarpGroupLayout{}));
  // M40 has five groups of eight values and uses the four-value store atom.
  using R2SCopyAtomC =
      std::conditional_t<kTileM == 40, cute::Copy_Atom<cute::SM90_U16x4_STSM_T, Tout>,
                         typename Base::R2SCopyAtomC>;
  static constexpr int kPackedStageBytes = kTileN * kTileK / 2;
  static constexpr int kRelativeStageBytes = kTileN * (kTileK / 32);
  static constexpr int kScaleStageBytes = kTileM * sizeof(float);
  static constexpr int shm_xw =
      cute::cosize(typename Base::SmemLayoutA{}) +
      kStages * (kPackedStageBytes + kRelativeStageBytes + kScaleStageBytes);
  static constexpr int kShmSize = shm_xw > Base::shm_y ? shm_xw : Base::shm_y;
};

__device__ __forceinline__ void copy_weight_16(void *destination, const void *source) {
  const uint32_t shared = static_cast<uint32_t>(__cvta_generic_to_shared(destination));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" : : "r"(shared), "l"(source) : "memory");
}

// Global W: int32[E,K/64,8*N], relative: uint8[E,K/32,N]. n_base and
// k_base are element coordinates. Tile N/K have no tails in the H20 contract.
// Paired gate/up uses n_base=tile_n*TileN/2 and virtual [gate(TileN/2),up(TileN/2)];
// only global loads are remapped. Shared layout and local decode are unchanged.
// Shared W: [2][TileN/64][128][uint4]. Relative physical layout is
// [E,K128,N64,warp4,lane_group8,K32_4,rowpair2]; shared keeps one K128 tile.
// The caller commits/waits cp.async together with its activation/AS stage.
template <class Config, bool kPairedGateUp = false>
__device__ __forceinline__ void issue_weight_stage(uint8_t *shared_weight, uint8_t *shared_relative,
                                                   const int32_t *weight, const uint8_t *relative,
                                                   int expert, int n_base, int k_base, int n, int k,
                                                   int thread) {
  static_assert(!kPairedGateUp || Config::kTileN == 256 || Config::kTileN == 512);
  const int local_thread = thread % 128;
  const int warpgroup = thread / 128;
#pragma unroll
  for (int plane = 0; plane < 2; ++plane) {
#pragma unroll
    for (int panel = 0; panel < Config::kPanelsPerWarpGroup; ++panel) {
      const int tile = warpgroup + panel * Config::kWarpGroups;
      const int64_t row = static_cast<int64_t>(expert) * (k / 64) + k_base / 64 + plane;
      int column = tile * 64;
      if constexpr (kPairedGateUp)
        column = (tile % (Config::kTileN / 128)) * 64 + (tile / (Config::kTileN / 128)) * 512;
      const auto *source = weight + row * (n * 8) + (n_base + column) * 8 + local_thread * 4;
      const int offset = ((plane * (Config::kTileN / 64) + tile) * 128 + local_thread) * 16;
      copy_weight_16(shared_weight + offset, source);
    }
  }
#pragma unroll
  for (int unit = thread; unit < Config::kRelativeStageBytes / 16; unit += Config::kThreads) {
    const int byte = unit * 16;
    const int source_byte =
        kPairedGateUp ? byte % (2 * Config::kTileN) + (byte / (2 * Config::kTileN)) * 2048 : byte;
    const int64_t row = static_cast<int64_t>(expert) * (k / 128) + k_base / 128;
    copy_weight_16(shared_relative + byte, relative + row * (n * 4) + n_base * 4 + source_byte);
  }
}

// The magnitude nibbles remain in fragment order. Their signs were interleaved
// offline so two PRMT and two sign merges form eight E4M3 values, including
// signed zero. exponent is the strict relative E8M0 byte in [1,12].
__device__ __forceinline__ uint2 decode_fp4x8(uint32_t packed, uint32_t exponent) {
  const uint32_t table_lo = exponent * 0x08080800u + 0x0c080000u;
  const uint32_t table_hi = exponent * 0x08080808u + 0x1c181410u;
  uint2 magnitude;
  asm("{ .reg .b32 s, t, a, b;\n"
      "  and.b32 s, %2, 0x77777777;\n"
      "  shr.u32 t, s, 16;\n"
      "  prmt.b32 a, %3, %4, s;\n"
      "  prmt.b32 b, %3, %4, t;\n"
      "  mov.b32 %0, a;\n"
      "  mov.b32 %1, b;\n"
      "}"
      : "=r"(magnitude.x), "=r"(magnitude.y)
      : "r"(packed), "r"(table_lo), "r"(table_hi));
  return {magnitude.x | ((packed << 4) & 0x80808080u), magnitude.y | (packed & 0x80808080u)};
}

// One logical N64 panel for the calling warpgroup, across the complete K128.
// Fragment shape is (16,1,4), so each K32 slice is a rank-2 MMA A operand.
// kPanel is the compile-time fragment repetition, not the physical N64 index.
template <class Config, int kPanel, bool kCached = false, class Fragment>
__device__ __forceinline__ void decode_weight_panel(const uint8_t *shared_weight,
                                                    const uint8_t *shared_relative,
                                                    Fragment &fragment, int thread,
                                                    const uint32_t *cached = nullptr) {
  static_assert(kPanel >= 0 && kPanel < Config::kPanelsPerWarpGroup);
  auto registers = cute::recast<uint32_t>(fragment);
  const int local_thread = thread % 128;
  const int tile = thread / 128 + kPanel * Config::kWarpGroups;
  const int scale_pair = (local_thread / 32) * 8 + (local_thread % 32) / 4;
  const uint32_t scale_address = static_cast<uint32_t>(
      __cvta_generic_to_shared(shared_relative + tile * 256 + scale_pair * 8));
  [[maybe_unused]] uint2 scale_words;
  if constexpr (Config::kTileM == 40) {
    asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];"
                 : "=r"(scale_words.x), "=r"(scale_words.y)
                 : "r"(scale_address)
                 : "memory");
  }
#pragma unroll
  for (int plane = 0; plane < 2; ++plane) {
    uint4 packed;
    if constexpr (kCached)
      packed = {cached[plane * 4], cached[plane * 4 + 1], cached[plane * 4 + 2],
                cached[plane * 4 + 3]};
    else
      packed = reinterpret_cast<const uint4 *>(
          shared_weight)[(plane * (Config::kTileN / 64) + tile) * 128 + local_thread];
    const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int ik = plane * 2 + half;
      uint32_t scales;
      if constexpr (Config::kTileM == 40) {
        scales = ((plane == 0 ? scale_words.x : scale_words.y) >> (half * 16)) & 0xffffu;
      } else {
        // Same packed bytes; keep small-M scale lifetime local to this K32 slice.
        uint16_t scale_pair_value;
        asm volatile("ld.shared.u16 %0, [%1];"
                     : "=h"(scale_pair_value)
                     : "r"(scale_address + ik * 2)
                     : "memory");
        scales = scale_pair_value;
      }
      const uint2 first = decode_fp4x8(words[half * 2], scales & 255u);
      const uint2 second = decode_fp4x8(words[half * 2 + 1], scales >> 8);
      registers(0, 0, ik) = first.x;
      registers(1, 0, ik) = second.x;
      registers(2, 0, ik) = first.y;
      registers(3, 0, ik) = second.y;
    }
  }
}

}  // namespace w4a8_detail
}  // namespace group_gemm_cp_async
}  // namespace hpc

#endif  // SRC_GROUP_GEMM_SM90_CP_ASYNC_W4A8_OPERAND_CUH_
