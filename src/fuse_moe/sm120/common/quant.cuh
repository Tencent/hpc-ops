// Copyright (C) 2026 Tencent.

#ifndef SRC_FUSE_MOE_SM120_COMMON_QUANT_CUH_
#define SRC_FUSE_MOE_SM120_COMMON_QUANT_CUH_

#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "src/fuse_moe/sm120/common/utils.h"

namespace hpc::fuse_moe::sm120 {

constexpr float kFp8Max = 448.0f;
constexpr float kMinScale = 0x1.0p-127f;

__device__ __forceinline__ uint8_t to_ue8m0_storage(float value) {
  int bits = __float_as_int(value);
  int exponent = (bits >> 23) & 0xFF;
  int mantissa = bits & 0x7FFFFF;
  bool tiny = (exponent == 0) && (mantissa <= 0x00400000);
  bool round_up = (mantissa > 0) && (exponent != 0xFE) && !tiny;
  return static_cast<uint8_t>(exponent + static_cast<int>(round_up));
}

__device__ __forceinline__ float ue8m0_to_float(uint8_t storage) {
  if (storage == 0) {
    return kMinScale;
  }
  return __int_as_float(static_cast<int>(storage) << 23);
}

__device__ __forceinline__ void quantization_scale(float max_abs, uint8_t& storage,
                                                   float& inv_scale) {
  float requested = fmaxf(max_abs / kFp8Max, kMinScale);
  storage = to_ue8m0_storage(requested);
  inv_scale = 1.0f / ue8m0_to_float(storage);
}

__device__ __forceinline__ int pack_e4m3_x4(float v0, float v1, float v2, float v3) {
  float4 values = make_float4(v0, v1, v2, v3);
  __nv_fp8x4_e4m3 packed(values);
  return *reinterpret_cast<int*>(&packed);
}

__device__ __forceinline__ void store_packed_scale(uint8_t* packed_scale_bytes, int scale_m_padded,
                                                   int scale_row, int column, uint8_t storage) {
  int block_k = column / kScaleVectorSize;
  int pack_col = block_k / 4;
  int byte_in_pack = block_k % 4;
  int64_t byte_offset =
      (static_cast<int64_t>(scale_row) + static_cast<int64_t>(pack_col) * scale_m_padded) * 4 +
      byte_in_pack;
  packed_scale_bytes[byte_offset] = storage;
}

}  // namespace hpc::fuse_moe::sm120

#endif  // SRC_FUSE_MOE_SM120_COMMON_QUANT_CUH_
