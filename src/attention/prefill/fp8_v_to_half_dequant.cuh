// Copyright 2025 hpc-ops authors

#ifndef SRC_ATTENTION_PREFILL_FP8_V_TO_HALF_DEQUANT_CUH_
#define SRC_ATTENTION_PREFILL_FP8_V_TO_HALF_DEQUANT_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#include "cute/tensor.hpp"

namespace hpc {
namespace attention {
namespace prefill {

// MN-major fp8 -> half (fp16) raw dequant for V (PV WGMMA consumes V as fp16).
// Cooperative dequant in one CTA, vectorized at 16-element granularity along
// the row (MN) axis: 16B fp8 LDS -> 32B half STS. Source and dest are both
// 128B-swizzled MN-major layouts; a 16-element MN strip is contiguous in smem
// (swizzle XOR only touches col offsets).
template <int kRows, int kCols, int kThreads, typename FP8Tensor, typename HalfTensor>
__device__ __forceinline__ void fp8_smem_to_half_smem_tile_raw_vec16_mn(FP8Tensor const &sX_fp8,
                                                                        HalfTensor &sX_half,
                                                                        int tid) {
  constexpr int kStrip = 16;
  static_assert(kRows % kStrip == 0, "kRows must be a multiple of 16 for raw vec16 (MN) dequant");
  constexpr int kStripsPerCol = kRows / kStrip;
  constexpr int kTotalStrips = kCols * kStripsPerCol;

  for (int s = tid; s < kTotalStrips; s += kThreads) {
    int col = s / kStripsPerCol;
    int row = (s % kStripsPerCol) * kStrip;

    auto fp8_smem_ptr = &sX_fp8(row, col);
    auto fp8_raw = cute::raw_pointer_cast(fp8_smem_ptr);

    uint4 src = *reinterpret_cast<uint4 const *>(fp8_raw);
    // 16 fp8 -> 16 half via hardware cvt.rn.f16x2.e4m3x2 on the 8 packed uint16
    // lanes of src directly (no 64-bit shift/or recombination, no intrinsic).
    uint4 dst_lo, dst_hi;
    const uint16_t *sp = reinterpret_cast<const uint16_t *>(&src);
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;\n"
                   : "=r"(reinterpret_cast<uint32_t *>(&dst_lo)[p])
                   : "h"(sp[p]));
    }
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;\n"
                   : "=r"(reinterpret_cast<uint32_t *>(&dst_hi)[p])
                   : "h"(sp[p + 4]));
    }

    auto half_lo_ptr = &sX_half(row, col);
    auto half_hi_ptr = &sX_half(row + 8, col);
    auto half_lo_raw = cute::raw_pointer_cast(half_lo_ptr);
    auto half_hi_raw = cute::raw_pointer_cast(half_hi_ptr);
    *reinterpret_cast<uint4 *>(half_lo_raw) = dst_lo;
    *reinterpret_cast<uint4 *>(half_hi_raw) = dst_hi;
  }
}

}  // namespace prefill
}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_PREFILL_FP8_V_TO_HALF_DEQUANT_CUH_
