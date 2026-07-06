// Copyright 2025 hpc-ops authors
#include <cuda.h>

#include <iostream>
#include <type_traits>

#include "src/quant/per_token_group_fp8_quant.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace quant {
namespace kernels {

// ---------------------------------------------------------------------------
// Small helpers.
// ---------------------------------------------------------------------------
constexpr float kInvFp8E4m3Max = 1.f / 448.f;
constexpr unsigned int kFullMask = 0xffffffffu;

// ---------------------------------------------------------------------------
// Fused bf16x2 * fp32_inv_scale -> fp8x2 (E4M3) cvt, saturating.
//
// The bf16 -> fp32 cast is exact; the multiply happens in fp32.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint16_t bf16x2_mul_to_fp8x2_e4m3_satfinite_f32(__nv_bfloat162 v,
                                                                           float inv_scale_f32) {
  float2 f = __bfloat1622float2(v);
  f.x *= inv_scale_f32;
  f.y *= inv_scale_f32;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
  uint16_t out;
  asm volatile("{cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;}\n" : "=h"(out) : "f"(f.x), "f"(f.y));
  return out;
#else
  __nv_fp8x2_e4m3 packed = __nv_fp8x2_e4m3(f);
  uint16_t out;
  static_assert(sizeof(packed) == sizeof(out), "fp8x2 must be 16-bit");
  __builtin_memcpy(&out, &packed, sizeof(out));
  return out;
#endif
}

template <int Width>
__device__ __forceinline__ float warp_reduce_max_xor(float x) {
  static_assert((Width & (Width - 1)) == 0 && Width <= 32, "Width must be 1,2,4,8,16,32");
#pragma unroll
  for (int offset = Width / 2; offset >= 1; offset >>= 1) {
    x = fmaxf(x, __shfl_xor_sync(kFullMask, x, offset, Width));
  }
  return x;
}

// ---------------------------------------------------------------------------
// Core kernel (templated on the static tile shape).
// ---------------------------------------------------------------------------
template <int kHiddenSize, int kWarpsPerRow, int kRowsPerBlock, int kN, int kBlocksPerRow>
__global__ void per_token_group_fp8_quant_v2(__nv_bfloat16 const *__restrict__ input_ptr,
                                             __nv_fp8_e4m3 *__restrict__ y_fp8_ptr,
                                             float *__restrict__ y_scale_ptr, float const quant_eps,
                                             int const batch_size) {
  constexpr int kGroupSize = 128;
  constexpr int kLanesPerGroup = kGroupSize / kN;      // 16 (kN=8) | 8 (kN=16)
  constexpr int kGroupsPerWarp = 32 / kLanesPerGroup;  // 2  | 4
  constexpr int kElementsPerWarp = 32 * kN;            // 256 | 512
  constexpr int kColsPerBlockX = kHiddenSize / kBlocksPerRow;
  constexpr int kColsPerSweep = kWarpsPerRow * kElementsPerWarp;
  constexpr int kIterPerRow = (kColsPerBlockX + kColsPerSweep - 1) / kColsPerSweep;
  constexpr int kHalfPairs = kN / 2;

  static_assert(kN == 8 || kN == 16, "kN must be 8 or 16");
  static_assert(kHiddenSize % kGroupSize == 0, "hidden must be multiple of 128");
  static_assert(kHiddenSize % kBlocksPerRow == 0, "row split must divide hidden");
  static_assert(kColsPerBlockX % kGroupSize == 0, "row split must keep group alignment");
  static_assert(kRowsPerBlock * kWarpsPerRow == 8, "block has exactly 8 warps");

  int const tid = threadIdx.x;
  int const iwarp = tid >> 5;
  int const ilane = tid & 31;
  int const ilane_in_group = ilane & (kLanesPerGroup - 1);
  int const igroup_in_warp = ilane / kLanesPerGroup;

  int const irow = blockIdx.x * kRowsPerBlock + iwarp / kWarpsPerRow;
  if (irow >= batch_size) {
    return;
  }

  int const split = blockIdx.y;
  int const iwarp_in_row = iwarp % kWarpsPerRow;
  int const col_split_base = split * kColsPerBlockX;
  int const col_thread_base = (iwarp_in_row * 32 + ilane) * kN;

  __nv_bfloat16 const *in_row =
      input_ptr + static_cast<size_t>(irow) * kHiddenSize + col_split_base;
  __nv_fp8_e4m3 *fp8_row = y_fp8_ptr + static_cast<size_t>(irow) * kHiddenSize + col_split_base;
  float *sc_row = y_scale_ptr + static_cast<size_t>(irow) * (kHiddenSize / kGroupSize) +
                  col_split_base / kGroupSize;

#pragma unroll
  for (int iter = 0; iter < kIterPerRow; iter++) {
    int const col = col_thread_base + iter * kColsPerSweep;
    bool const valid = (col + kN <= kColsPerBlockX);

    __nv_bfloat162 in[kHalfPairs];
    if constexpr (kN == 8) {
      uint4 v0{};
      if (valid) {
        v0 = *reinterpret_cast<uint4 const *>(in_row + col);
      }
#pragma unroll
      for (int k = 0; k < 4; k++) {
        unsigned u = reinterpret_cast<unsigned const *>(&v0)[k];
        __builtin_memcpy(&in[k], &u, sizeof(in[k]));
      }
    } else {  // kN == 16
      uint4 v0{};
      uint4 v1{};
      if (valid) {
        v0 = *reinterpret_cast<uint4 const *>(in_row + col);
        v1 = *reinterpret_cast<uint4 const *>(in_row + col + 8);
      }
#pragma unroll
      for (int k = 0; k < 4; k++) {
        unsigned u0 = reinterpret_cast<unsigned const *>(&v0)[k];
        unsigned u1 = reinterpret_cast<unsigned const *>(&v1)[k];
        __builtin_memcpy(&in[k], &u0, sizeof(in[k]));
        __builtin_memcpy(&in[k + 4], &u1, sizeof(in[k]));
      }
    }

    __nv_bfloat162 amax2 = __habs2(in[0]);
#pragma unroll
    for (int k = 1; k < kHalfPairs; k++) {
      amax2 = __hmax2(amax2, __habs2(in[k]));
    }
    float amax = __bfloat162float(__hmax(amax2.x, amax2.y));
    amax = warp_reduce_max_xor<kLanesPerGroup>(amax);

    // -------------------- Scale + fp32 quant ------------------------
    // inv_scale stays fp32, and the quant multiply runs in fp32 inside the bf16x2->fp8x2 converter.
    float const scale = amax * kInvFp8E4m3Max;
    float const inv_scale_f32 = (scale > 0.f) ? rcpf_ftz(scale + quant_eps) : 0.f;

    if (valid) {
      if constexpr (kN == 8) {
        uint2 out;
        uint16_t *o = reinterpret_cast<uint16_t *>(&out);
#pragma unroll
        for (int k = 0; k < 4; k++) {
          o[k] = bf16x2_mul_to_fp8x2_e4m3_satfinite_f32(in[k], inv_scale_f32);
        }
        *reinterpret_cast<uint2 *>(fp8_row + col) = out;
      } else {  // kN == 16
        uint4 out;
        uint16_t *o = reinterpret_cast<uint16_t *>(&out);
#pragma unroll
        for (int k = 0; k < 8; k++) {
          o[k] = bf16x2_mul_to_fp8x2_e4m3_satfinite_f32(in[k], inv_scale_f32);
        }
        *reinterpret_cast<uint4 *>(fp8_row + col) = out;
      }
    }

    if (valid && ilane_in_group == 0) {
      constexpr int kScalesPerIter = kWarpsPerRow * kGroupsPerWarp;
      int const pos = iter * kScalesPerIter + iwarp_in_row * kGroupsPerWarp + igroup_in_warp;
      sc_row[pos] = scale;
    }
  }
}

// ---------------------------------------------------------------------------
// Host-side launchers
// ---------------------------------------------------------------------------
template <int kHiddenSize, int kWarpsPerRow, int kN, int kBlocksPerRow>
static void launch_v2(void const *input_ptr, void *output_ptr, void *quant_scale,
                      float const quant_eps, int const batch_size, cudaStream_t stream) {
  constexpr int kWarpCount = 8;
  constexpr int kWarpSize = 32;
  constexpr int kRowsPerBlock = kWarpCount / kWarpsPerRow;

  dim3 const block(kWarpSize * kWarpCount);
  dim3 const grid((batch_size + kRowsPerBlock - 1) / kRowsPerBlock, kBlocksPerRow);

  per_token_group_fp8_quant_v2<kHiddenSize, kWarpsPerRow, kRowsPerBlock, kN, kBlocksPerRow>
      <<<grid, block, 0, stream>>>(static_cast<__nv_bfloat16 const *>(input_ptr),
                                   static_cast<__nv_fp8_e4m3 *>(output_ptr),
                                   static_cast<float *>(quant_scale), quant_eps, batch_size);
}

template <int kHiddenSize, int kWarpsPerRow, int kN, int kBlocksPerRow>
static void maybe_launch_v2(void const *input_ptr, void *output_ptr, void *quant_scale,
                            float const quant_eps, int const batch_size, cudaStream_t stream) {
  constexpr int kSweepCols = kWarpsPerRow * 32 * kN;
  constexpr int kColsPerBlock = kHiddenSize / kBlocksPerRow;
  constexpr bool kValid =
      (kBlocksPerRow == 1) || ((kHiddenSize % kBlocksPerRow == 0) && (kColsPerBlock % 128 == 0) &&
                               (kColsPerBlock % kSweepCols == 0));

  if constexpr (kValid) {
    launch_v2<kHiddenSize, kWarpsPerRow, kN, kBlocksPerRow>(input_ptr, output_ptr, quant_scale,
                                                            quant_eps, batch_size, stream);
  } else {
    launch_v2<kHiddenSize, kWarpsPerRow, kN, 1>(input_ptr, output_ptr, quant_scale, quant_eps,
                                                batch_size, stream);
  }
}

static int pick_blocks_per_row(int batch_size, int rows_per_block, int hidden_size, int max_split,
                               int sweep_cols) {
  constexpr int kTargetBlocks = 256;
  int const row_blocks = (batch_size + rows_per_block - 1) / rows_per_block;
  int splits = 1;
  while (splits * 2 <= max_split && row_blocks * splits < kTargetBlocks &&
         hidden_size % (splits * 2) == 0 && (hidden_size / (splits * 2)) % sweep_cols == 0) {
    splits *= 2;
  }
  return splits;
}

template <int kHiddenSize, int kWarpsPerRow, int kRowsPerBlock>
__global__ void per_token_group_fp8_quant(const __nv_bfloat16 *input_ptr, __nv_fp8_e4m3 *y_fp8_ptr,
                                          float *y_scale_ptr, float quant_eps, int batch_size) {
  constexpr int kN = 16 / sizeof(__nv_bfloat16);
  constexpr int kElementsPerWarp = 32 * kN;
  constexpr int kIerPerRow =
      (kHiddenSize + kWarpsPerRow * kElementsPerWarp - 1) / (kWarpsPerRow * kElementsPerWarp);
  constexpr float kInvFp8Max = 1.f / 448.f;

  vec_t<float, kN> input[kIerPerRow];
  int idx = threadIdx.x;
  int iwarp = idx / 32;
  int ilane = idx % 32;

  int irow = blockIdx.x * kRowsPerBlock + iwarp / kWarpsPerRow;
  if (irow >= batch_size) {
    return;
  }

  auto *input_row_ptr = input_ptr + irow * kHiddenSize;
  auto *y_scale_row_ptr = y_scale_ptr + irow * kHiddenSize / 128;
  auto *y_fp8_row_ptr = y_fp8_ptr + irow * kHiddenSize;

  int icol = ((iwarp % kWarpsPerRow) * 32 + ilane) * kN;
#pragma unroll
  for (int iter = 0; iter < kIerPerRow; iter++) {
    int col = icol + iter * kWarpsPerRow * kElementsPerWarp;
    if (col < kHiddenSize) {
      // load input
      input[iter] = to<float>(load<__nv_bfloat162, kN / 2>(input_row_ptr + col));

      // cal max
      float max = 0.0f;
#pragma unroll
      for (int i = 0; i < kN; i++) {
        max = fmaxf(max, fabsf(input[iter][i]));
      }
      max = half_warp_reduce_max_down(max);
      float scale = max * kInvFp8Max;
      float inv_scale = rcpf_ftz(scale + quant_eps);

      // quant
#pragma unroll
      for (int i = 0; i < kN; i++) {
        input[iter][i] *= inv_scale;
      }

      // store scale
      int pos = iwarp % kWarpsPerRow * 2 + iter * kWarpsPerRow * 2;
      if (ilane == 0) {
        store(y_scale_row_ptr + pos, scale);
      } else if (ilane == 16) {
        store(y_scale_row_ptr + pos + 1, scale);
      }

      // store output
      store(y_fp8_row_ptr + col, to<__nv_fp8x4_e4m3>(input[iter]));
    }
  }
}

}  // namespace kernels

template <int kHiddenSize, int kWarpsPerRow>
void launch_per_token_group_fp8_quant(const void *input_ptr, void *output_ptr, void *quant_scale,
                                      int group_size, float quant_eps, int hidden_size,
                                      int batch_size, cudaStream_t stream) {
  constexpr int kWarpCount = 8;
  constexpr int kWarpSize = 32;
  constexpr int kRowsPerBlock = kWarpCount / kWarpsPerRow;
  dim3 block(kWarpSize * kWarpCount);
  dim3 grid((batch_size + kRowsPerBlock - 1) / kRowsPerBlock);
  kernels::per_token_group_fp8_quant<kHiddenSize, kWarpsPerRow, kRowsPerBlock>
      <<<grid, block, 0, stream>>>((__nv_bfloat16 *)(input_ptr), (__nv_fp8_e4m3 *)(output_ptr),
                                   (float *)(quant_scale), quant_eps, batch_size);
}

bool per_token_group_fp8_quant_async(const void *input_ptr, void *output_ptr, void *quant_scale,
                                     int group_size, float quant_eps, int hidden_size,
                                     int batch_size, cudaStream_t stream) {
  // Expand a static (hidden, warps-per-row, elems-per-thread, max-blocks-per-row) tile choice
  // into a runtime row-split selection while keeping launch template arguments compile-time.
  auto dispatch_v2 = [&](auto hidden_size_tag, auto warps_per_row_tag, auto elems_per_thread_tag,
                         auto max_split_tag) {
    constexpr int kHiddenSize = decltype(hidden_size_tag)::value;
    constexpr int kWarpsPerRow = decltype(warps_per_row_tag)::value;
    constexpr int kElemsPerThread = decltype(elems_per_thread_tag)::value;
    constexpr int kMaxSplit = decltype(max_split_tag)::value;
    constexpr int kRowsPerBlock = 8 / kWarpsPerRow;
    constexpr int kSweepCols = kWarpsPerRow * 32 * kElemsPerThread;
    int const splits =
        kernels::pick_blocks_per_row(batch_size, kRowsPerBlock, kHiddenSize, kMaxSplit, kSweepCols);
    switch (splits) {
      case 1:
        kernels::maybe_launch_v2<kHiddenSize, kWarpsPerRow, kElemsPerThread, 1>(
            input_ptr, output_ptr, quant_scale, quant_eps, batch_size, stream);
        break;
      case 2:
        kernels::maybe_launch_v2<kHiddenSize, kWarpsPerRow, kElemsPerThread, 2>(
            input_ptr, output_ptr, quant_scale, quant_eps, batch_size, stream);
        break;
      case 4:
        kernels::maybe_launch_v2<kHiddenSize, kWarpsPerRow, kElemsPerThread, 4>(
            input_ptr, output_ptr, quant_scale, quant_eps, batch_size, stream);
        break;
      case 8:
        kernels::maybe_launch_v2<kHiddenSize, kWarpsPerRow, kElemsPerThread, 8>(
            input_ptr, output_ptr, quant_scale, quant_eps, batch_size, stream);
        break;
      default:
        kernels::maybe_launch_v2<kHiddenSize, kWarpsPerRow, kElemsPerThread, 1>(
            input_ptr, output_ptr, quant_scale, quant_eps, batch_size, stream);
        break;
    }
  };

  if (hidden_size == 128) {
    constexpr int kHiddenSize = 128;
    constexpr int kWarpsPerRow = 1;
    launch_per_token_group_fp8_quant<kHiddenSize, kWarpsPerRow>(
        input_ptr, output_ptr, quant_scale, group_size, quant_eps, hidden_size, batch_size, stream);
  } else if (hidden_size == 256) {
    constexpr int kHiddenSize = 256;
    constexpr int kWarpsPerRow = 1;
    launch_per_token_group_fp8_quant<kHiddenSize, kWarpsPerRow>(
        input_ptr, output_ptr, quant_scale, group_size, quant_eps, hidden_size, batch_size, stream);
  } else if (hidden_size == 384) {
    constexpr int kHiddenSize = 384;
    constexpr int kWarpsPerRow = 2;
    launch_per_token_group_fp8_quant<kHiddenSize, kWarpsPerRow>(
        input_ptr, output_ptr, quant_scale, group_size, quant_eps, hidden_size, batch_size, stream);
  } else if (hidden_size == 512) {
    constexpr int kHiddenSize = 512;
    constexpr int kWarpsPerRow = 2;
    launch_per_token_group_fp8_quant<kHiddenSize, kWarpsPerRow>(
        input_ptr, output_ptr, quant_scale, group_size, quant_eps, hidden_size, batch_size, stream);
  } else if (hidden_size == 768) {
    constexpr int kHiddenSize = 768;
    constexpr int kWarpsPerRow = 4;
    launch_per_token_group_fp8_quant<kHiddenSize, kWarpsPerRow>(
        input_ptr, output_ptr, quant_scale, group_size, quant_eps, hidden_size, batch_size, stream);
  } else if (hidden_size == 1024) {
    constexpr int kHiddenSize = 1024;
    constexpr int kWarpsPerRow = 4;
    launch_per_token_group_fp8_quant<kHiddenSize, kWarpsPerRow>(
        input_ptr, output_ptr, quant_scale, group_size, quant_eps, hidden_size, batch_size, stream);
  } else if (hidden_size == 1536) {
    constexpr int kHiddenSize = 1536;
    constexpr int kWarpsPerRow = 8;
    launch_per_token_group_fp8_quant<kHiddenSize, kWarpsPerRow>(
        input_ptr, output_ptr, quant_scale, group_size, quant_eps, hidden_size, batch_size, stream);
  } else if (hidden_size == 2048) {
    dispatch_v2(std::integral_constant<int, 2048>{}, std::integral_constant<int, 1>{},
                std::integral_constant<int, 16>{}, std::integral_constant<int, 1>{});
  } else if (hidden_size == 3072) {
    dispatch_v2(std::integral_constant<int, 3072>{}, std::integral_constant<int, 1>{},
                std::integral_constant<int, 16>{}, std::integral_constant<int, 1>{});
  } else if (hidden_size == 3200) {
    dispatch_v2(std::integral_constant<int, 3200>{}, std::integral_constant<int, 1>{},
                std::integral_constant<int, 8>{}, std::integral_constant<int, 1>{});
  } else if (hidden_size == 3456) {
    dispatch_v2(std::integral_constant<int, 3456>{}, std::integral_constant<int, 1>{},
                std::integral_constant<int, 8>{}, std::integral_constant<int, 1>{});
  } else if (hidden_size == 4096) {
    dispatch_v2(std::integral_constant<int, 4096>{}, std::integral_constant<int, 1>{},
                std::integral_constant<int, 16>{}, std::integral_constant<int, 1>{});
  } else if (hidden_size == 5120) {
    dispatch_v2(std::integral_constant<int, 5120>{}, std::integral_constant<int, 1>{},
                std::integral_constant<int, 16>{}, std::integral_constant<int, 1>{});
  } else if (hidden_size == 5504) {
    dispatch_v2(std::integral_constant<int, 5504>{}, std::integral_constant<int, 1>{},
                std::integral_constant<int, 16>{}, std::integral_constant<int, 1>{});
  } else if (hidden_size == 6144) {
    dispatch_v2(std::integral_constant<int, 6144>{}, std::integral_constant<int, 1>{},
                std::integral_constant<int, 16>{}, std::integral_constant<int, 1>{});
  } else if (hidden_size == 6912) {
    dispatch_v2(std::integral_constant<int, 6912>{}, std::integral_constant<int, 1>{},
                std::integral_constant<int, 16>{}, std::integral_constant<int, 1>{});
  } else if (hidden_size == 7168) {
    dispatch_v2(std::integral_constant<int, 7168>{}, std::integral_constant<int, 1>{},
                std::integral_constant<int, 16>{}, std::integral_constant<int, 1>{});
  } else if (hidden_size == 8192) {
    dispatch_v2(std::integral_constant<int, 8192>{}, std::integral_constant<int, 1>{},
                std::integral_constant<int, 16>{}, std::integral_constant<int, 1>{});
  } else if (hidden_size == 13824) {
    dispatch_v2(std::integral_constant<int, 13824>{}, std::integral_constant<int, 1>{},
                std::integral_constant<int, 16>{}, std::integral_constant<int, 1>{});
  } else {
    std::cout << "not supported hidden_size for per_token_group_fp8_quant_async:" << hidden_size
              << std::endl;
    return false;
  }

  // Surface launch/config failures instead of silently leaving the output
  // uninitialized (which downstream reads back as NaN).
  cudaError_t const launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    std::cout << "per_token_group_fp8_quant_async launch failed (hidden_size=" << hidden_size
              << ", batch_size=" << batch_size << "): " << cudaGetErrorString(launch_err)
              << std::endl;
    return false;
  }

  return true;
}
}  // namespace quant
}  // namespace hpc
