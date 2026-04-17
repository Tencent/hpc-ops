// Copyright 2025 hpc-ops authors
#include <cuda.h>

#include <algorithm>
#include <cstdio>

#include "cutlass/arch/cache_operation.h"
#include "cutlass/arch/memory.h"
#include "cutlass/arch/mma.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/array.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"
#include "src/group_gemm/group_gemm.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace group_gemm {
namespace groupwise_w4a8_mma {

using Tw = cutlass::int4b_t;
using Tx = cutlass::float_e4m3_t;
using Ts = cutlass::bfloat16_t;  // scale
using Tout = cutlass::bfloat16_t;

using MMA_16x8x16_F32F16F16 = cutlass::arch::Mma<cutlass::gemm::GemmShape<16, 8, 16>,  // MMA Shape
                                                 32,  // Number of threads participating
                                                 cutlass::half_t,                // A type
                                                 cutlass::layout::RowMajor,      // A layout
                                                 cutlass::half_t,                // B type
                                                 cutlass::layout::ColumnMajor,   // B layout
                                                 float,                          // accum type
                                                 cutlass::layout::RowMajor,      // accum layout
                                                 cutlass::arch::OpMultiplyAdd>;  // operator

constexpr int kElementsPerAccess =
    128 / cutlass::sizeof_bits<Tx>::value;  // each thread load 16B = 16 fp8
constexpr int kPackedElements = 8 / cutlass::sizeof_bits<Tw>::value;  // 1B = 2 int4
constexpr int kThreadPerRow = 4;  // fixed to 4 for mma.sync.aligned.m16n8k32...
constexpr int kInterleaveBlockSizeKMode = kThreadPerRow * kElementsPerAccess;  // 64 elem
constexpr int kInterleaveBlockSizeMMode =
    4;  // 64 int4 = 32B, so fold 4 block into one row for 128B cache line in load global.
constexpr int kWarpSize = 32;
constexpr int kWarpCount = 4;
constexpr int kThreadsPerBlock = kWarpSize * kWarpCount;  // 128
constexpr int kTileM = 8;
constexpr int kTileN = 16 * kWarpCount;  // 64
constexpr int kTileK = kInterleaveBlockSizeKMode;
constexpr int kScalesPerAccess =
    128 / cutlass::sizeof_bits<Ts>::value;  // each thread once load 16B = 8 bf16
// real size is 2 * kThreadsPerBlock * 16 = 2 * 128 * 16B = 4KB
// for avoid bank conflict, set shm like tensor which shape is [2 * 16, 8 + 1] and dtype is float4
constexpr int kSharedMemorySize = 2 * 16 * (8 + 1) * 16;
constexpr int kScaleInterleaveBlockSizeMMode =
    2;  // one thread is 16B, four threads per row , so fold 2 row into one row for 128B cache
        // line in load share memory.

using FragmentA = cutlass::Array<Tw, kElementsPerAccess>;
using FragmentB = cutlass::Array<Tx, kElementsPerAccess>;
using FragmentS = cutlass::Array<Ts, kScalesPerAccess>;
constexpr int kUnroll = 2;
constexpr int kUnrollTileK = kUnroll * kTileK;
using FragmentArrayA = cutlass::Array<FragmentA, kUnroll>;
using FragmentArrayB = cutlass::Array<FragmentB, kUnroll>;
using FragmentArrayC = cutlass::Array<float, 4>;
using FragmentArrayOut = cutlass::Array<Tout, 4>;

constexpr cutlass::FloatRoundStyle Round = cutlass::FloatRoundStyle::round_to_nearest;

}  // namespace groupwise_w4a8_mma

namespace kernels {

__device__ cutlass::Array<cutlass::half_t, 8> convert_interleaved_int4_to_fp16(uint32_t src_reg) {
  static constexpr uint32_t immLut = (0xf0 & 0xcc) ^ 0xaa;
  static constexpr uint32_t bottom_and_mask = 0x000F000F;
  static constexpr uint32_t top_and_mask = 0x00F000F0;
  static constexpr uint32_t bottom_xor_mask = 0x64086408;
  static constexpr uint32_t top_xor_mask = 0x64806480;

  // This is the half2 {1032, 1032} represented as an integer.
  static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64086408;
  // This is the half2 {1 / 16, 1 / 16} represented as an integer.
  static constexpr uint32_t ONE_SIXTEENTH = 0x2c002c00;
  // This is the half2 {-72, -72} represented as an integer.
  static constexpr uint32_t NEG_72 = 0xd480d480;

  cutlass::Array<cutlass::half_t, 8> res;

  // #pragma unroll
  //   for (int i = 0; i < groupwise_w4a8_mma::kElementsPerAccess / 8; ++i) {
  // uint32_t src_reg = *(reinterpret_cast<const uint32_t *>(&src) + i);
  uint32_t *r = reinterpret_cast<uint32_t *>(&res);

  asm volatile(
      "{\n"
      "  lop3.b32 %0, %1, %2, %3, %4;\n"
      "}\n"
      : "=r"(r[0])
      : "r"(src_reg), "n"(bottom_and_mask), "n"(bottom_xor_mask), "n"(immLut));
  asm volatile(
      "{\n"
      "  lop3.b32 %0, %1, %2, %3, %4;\n"
      "}\n"
      : "=r"(r[1])
      : "r"(src_reg), "n"(top_and_mask), "n"(top_xor_mask), "n"(immLut));

  src_reg >>= 8;
  asm volatile(
      "{\n"
      "  lop3.b32 %0, %1, %2, %3, %4;\n"
      "}\n"
      : "=r"(r[2])
      : "r"(src_reg), "n"(bottom_and_mask), "n"(bottom_xor_mask), "n"(immLut));
  asm volatile(
      "{\n"
      "  lop3.b32 %0, %1, %2, %3, %4;\n"
      "}\n"
      : "=r"(r[3])
      : "r"(src_reg), "n"(top_and_mask), "n"(top_xor_mask), "n"(immLut));

  asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(r[0]) : "r"(r[0]), "r"(FP16_TOP_MAGIC_NUM));
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
               : "=r"(r[1])
               : "r"(r[1]), "r"(ONE_SIXTEENTH), "r"(NEG_72));
  asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(r[2]) : "r"(r[2]), "r"(FP16_TOP_MAGIC_NUM));
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
               : "=r"(r[3])
               : "r"(r[3]), "r"(ONE_SIXTEENTH), "r"(NEG_72));
  // }

  return res;
}

template <int kGroupSize>
__global__ void group_gemm_groupwise_w4a8_mma_kernel(
    groupwise_w4a8_mma::Tout *y_ptr, const groupwise_w4a8_mma::Tx *x_ptr,
    const groupwise_w4a8_mma::Tw *weight_ptr, const int32_t *seqlens_ptr,
    const int32_t *cu_seqlens_ptr, const groupwise_w4a8_mma::Ts *yscale_ptr, int n, int k) {
  using namespace groupwise_w4a8_mma;  // NOLINT

  extern __shared__ uint8_t shm_data[] alignas(16);  // scale aligned 16

  int group_idx = blockIdx.y;

  cudaGridDependencySynchronize();

  int m = seqlens_ptr[group_idx];
  int64_t m_offset = static_cast<int64_t>(cu_seqlens_ptr[group_idx]);

#pragma unroll 1
  for (int mtile_idx = 0; mtile_idx < m; mtile_idx += kTileM) {
    // block base ptr
    auto const *x_row_ptr = x_ptr + m_offset * k;
    int ntile_idx = blockIdx.x;
    int64_t n_offset = static_cast<int64_t>(group_idx * n + ntile_idx * kTileN);
    auto const *weight_row_ptr = weight_ptr + n_offset * k / kPackedElements;
    int pad_s_k = (k / kGroupSize + kScalesPerAccess - 1) / kScalesPerAccess * kScalesPerAccess;
    auto const *yscale_row_ptr = yscale_ptr + n_offset * pad_s_k;

    // thread base ptr
    int row_idx = threadIdx.y;
    int col_idx = threadIdx.x;
    int x_m_idx = mtile_idx + row_idx % 8;  // each warp load same x tile
    x_row_ptr += col_idx * kElementsPerAccess;
    if (x_m_idx < m) {
      x_row_ptr += x_m_idx * k;
    }

    weight_row_ptr += (row_idx / kInterleaveBlockSizeMMode * kInterleaveBlockSizeMMode * k +
                       row_idx % kInterleaveBlockSizeMMode * kInterleaveBlockSizeKMode +
                       col_idx * kElementsPerAccess) /
                      kPackedElements;  // each warp load diff w tile

    // scale is small, so just load 64B cache line, do not k mode interleave
    yscale_row_ptr += row_idx * pad_s_k + col_idx * kScalesPerAccess;

    // each thread in one row store diff scale
    auto *shm_scale_row_store_ptr =
        shm_data + row_idx / kScaleInterleaveBlockSizeMMode * (8 + 1) * 16 +
        row_idx % kScaleInterleaveBlockSizeMMode * 4 * 16 + col_idx * 16;

    // fragment
    FragmentArrayC frag_mma_c;
    frag_mma_c.clear();

    FragmentArrayA frag_array_A_row0;
    FragmentArrayA frag_array_A_row1;
    FragmentArrayB frag_array_B;

    constexpr int kElementsCorrespondPerScalesAccess =
        kThreadPerRow * kScalesPerAccess * kGroupSize;  // 4 * 8 * 64/128 = 2048/4096

#pragma unroll 1
    // share memory can only store scales of 2048/4096 elem in k dimension
    for (int ksplit_idx = 0; ksplit_idx < (k + kElementsCorrespondPerScalesAccess - 1) /
                                              kElementsCorrespondPerScalesAccess;
         ++ksplit_idx) {
      // load scales
      int k_scale_offset = ksplit_idx * kThreadPerRow * kScalesPerAccess;
      if (col_idx * kScalesPerAccess + k_scale_offset < pad_s_k) {
        cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Global>(
            shm_scale_row_store_ptr,
            reinterpret_cast<uint8_t const *>(yscale_row_ptr + k_scale_offset));
        cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Global>(
            shm_scale_row_store_ptr + kSharedMemorySize / 2,
            reinterpret_cast<uint8_t const *>(yscale_row_ptr + k_scale_offset +
                                              kTileN / 2 * pad_s_k));
      }

      bool wait_cp_async = false;
      // each thread in one row load same scale
      Ts *shm_scale_row0_load_ptr = reinterpret_cast<Ts *>(
          shm_data + row_idx / kScaleInterleaveBlockSizeMMode * (8 + 1) * 16 +
          row_idx % kScaleInterleaveBlockSizeMMode * 4 * 16);

#pragma unroll 1
      for (int k_base_offset = ksplit_idx * kElementsCorrespondPerScalesAccess;
           k_base_offset < min(k, ksplit_idx * kElementsCorrespondPerScalesAccess +
                                      kElementsCorrespondPerScalesAccess);
           k_base_offset += kUnrollTileK) {
        FragmentArrayC frag_mma_accum[kUnrollTileK / kGroupSize];
#pragma unroll
        for (int i = 0; i < kUnrollTileK / kGroupSize; ++i) {
          frag_mma_accum[i].clear();
        }

        cutlass::NumericArrayConverter<cutlass::half_t, Tx, 4, Round> srcB_converter;
        MMA_16x8x16_F32F16F16 mma_op;

#pragma unroll
        for (int unroll_idx = 0; unroll_idx < kUnroll; ++unroll_idx) {
          int k_offset = k_base_offset + unroll_idx * kTileK;
          // swapAB
          // fetch fragment A from weight matrix
          cutlass::arch::global_load<FragmentA, sizeof(FragmentA),
                                     cutlass::arch::CacheOperation::LastUse>(
              frag_array_A_row0[unroll_idx],
              (weight_row_ptr + k_offset * kInterleaveBlockSizeMMode / kPackedElements), true);
          cutlass::arch::global_load<FragmentA, sizeof(FragmentA),
                                     cutlass::arch::CacheOperation::LastUse>(
              frag_array_A_row1[unroll_idx],
              (weight_row_ptr + k_offset * kInterleaveBlockSizeMMode / kPackedElements +
               kTileN / 2 / kPackedElements * k),
              true);

          // fetch fragment B from x matrix
          cutlass::arch::global_load<FragmentB, sizeof(FragmentB),
                                     cutlass::arch::CacheOperation::Always>(
              frag_array_B[unroll_idx], (x_row_ptr + k_offset), true);
        }

#pragma unroll
        for (int unroll_idx = 0; unroll_idx < kUnroll; ++unroll_idx) {
          for (int i = 0; i < 2; ++i) {
            cutlass::Array<cutlass::half_t, 8> fragA_compute_row0 =
                convert_interleaved_int4_to_fp16(
                    *(reinterpret_cast<uint32_t *>(&frag_array_A_row0[unroll_idx]) + i));
            cutlass::Array<cutlass::half_t, 8> fragA_compute_row1 =
                convert_interleaved_int4_to_fp16(
                    *(reinterpret_cast<uint32_t *>(&frag_array_A_row1[unroll_idx]) + i));
            for (int j = 0; j < 2; ++j) {
              cutlass::Array<cutlass::half_t, 4> frag_mma_b = srcB_converter(
                  *(reinterpret_cast<cutlass::Array<Tx, 4> *>(&frag_array_B[unroll_idx]) + 2 * i +
                    j));
              cutlass::Array<cutlass::half_t, 8> frag_mma_a;

              uint32_t *mma_2xfp16_A = reinterpret_cast<uint32_t *>(&frag_mma_a);

              uint32_t const *frag_2xfp16_A_row0 =
                  reinterpret_cast<uint32_t const *>(&(fragA_compute_row0.data()[j * 4]));
              uint32_t const *frag_2xfp16_A_row1 =
                  reinterpret_cast<uint32_t const *>(&(fragA_compute_row1.data()[j * 4]));

              mma_2xfp16_A[0] = frag_2xfp16_A_row0[0];
              mma_2xfp16_A[1] = frag_2xfp16_A_row1[0];
              mma_2xfp16_A[2] = frag_2xfp16_A_row0[1];
              mma_2xfp16_A[3] = frag_2xfp16_A_row1[1];

              if constexpr (kGroupSize == 64) {
                mma_op(frag_mma_accum[unroll_idx], frag_mma_a, frag_mma_b,
                       frag_mma_accum[unroll_idx]);
              } else if constexpr (kGroupSize == 128) {
                mma_op(frag_mma_accum[0], frag_mma_a, frag_mma_b, frag_mma_accum[0]);
              }
            }
          }
        }

        if (!wait_cp_async) {
          cutlass::arch::cp_async_fence();
          cutlass::arch::cp_async_wait<0>();
          // each warp only use scale which load by the warp
          __syncwarp();
          wait_cp_async = true;
        }

        Ts SFA_row0 = *(shm_scale_row0_load_ptr);
        Ts SFA_row1 = *(shm_scale_row0_load_ptr + kSharedMemorySize / 4);
        ++shm_scale_row0_load_ptr;

        frag_mma_c[0] += frag_mma_accum[0][0] * float(SFA_row0);
        frag_mma_c[1] += frag_mma_accum[0][1] * float(SFA_row0);
        frag_mma_c[2] += frag_mma_accum[0][2] * float(SFA_row1);
        frag_mma_c[3] += frag_mma_accum[0][3] * float(SFA_row1);

        if constexpr (kGroupSize == 64) {
          Ts SFA_row0 = *(shm_scale_row0_load_ptr);
          Ts SFA_row1 = *(shm_scale_row0_load_ptr + kSharedMemorySize / 4);
          ++shm_scale_row0_load_ptr;

          frag_mma_c[0] += frag_mma_accum[1][0] * float(SFA_row0);
          frag_mma_c[1] += frag_mma_accum[1][1] * float(SFA_row0);
          frag_mma_c[2] += frag_mma_accum[1][2] * float(SFA_row1);
          frag_mma_c[3] += frag_mma_accum[1][3] * float(SFA_row1);
        }
      }
    }

    // Epilogue: convert float mma output to bfloat16
    cutlass::NumericArrayConverter<Tout, float, 4, Round> output_converter;
    int y_m_idx = mtile_idx + col_idx * 2;
    if (y_m_idx < m) {
      // block base ptr
      auto *y_row_ptr = y_ptr + m_offset * n + ntile_idx * kTileN;
      // thread base ptr
      // swap ab
      y_row_ptr += y_m_idx * n + row_idx * 2;
      FragmentArrayOut frag_out = output_converter(frag_mma_c);
      vec_t<__nv_bfloat16, 2> vec;
      vec[0] = static_cast<Tout>(frag_out[0]);
      vec[1] = static_cast<Tout>(frag_out[2]);
      store(y_row_ptr, vec);
      if (y_m_idx + 1 < m) {
        vec[0] = static_cast<Tout>(frag_out[1]);
        vec[1] = static_cast<Tout>(frag_out[3]);
        store(y_row_ptr + n, vec);
      }
    }
  }

  cudaTriggerProgrammaticLaunchCompletion();
}
}  // namespace kernels

void group_gemm_groupwise_w4a8_mma_async(void *y_ptr, const void *x_ptr, const void *weight_ptr,
                                         const void *seqlens_ptr, const void *cu_seqlens_ptr,
                                         const void *yscale_ptr, int num_group, int m, int n, int k,
                                         int group_size, cudaStream_t stream) {
  using namespace groupwise_w4a8_mma;  // NOLINT

  dim3 grid(n / kTileN, num_group);  // n must be divided by kTileN
  dim3 block(kThreadPerRow, kThreadsPerBlock / kThreadPerRow);

  if (group_size == 128) {
    // pdl attr
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = 1;

    // Set the attribute in a kernel launch configuration
    cudaLaunchConfig_t config{};

    // Base launch configuration
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = kSharedMemorySize;
    config.stream = stream;

    // Add special attribute for PDL
    config.attrs = attribute;
    config.numAttrs = 1;

    auto kernel = kernels::group_gemm_groupwise_w4a8_mma_kernel<128>;

    cudaLaunchKernelEx(
        &config, kernel, reinterpret_cast<Tout *>(y_ptr), reinterpret_cast<const Tx *>(x_ptr),
        reinterpret_cast<const Tw *>(weight_ptr), reinterpret_cast<const int32_t *>(seqlens_ptr),
        reinterpret_cast<const int32_t *>(cu_seqlens_ptr), reinterpret_cast<const Ts *>(yscale_ptr),
        n, k);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("kernel launch error: %s - %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
    }
  } else if (group_size == 64) {
    // pdl attr
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = 1;

    // Set the attribute in a kernel launch configuration
    cudaLaunchConfig_t config{};

    // Base launch configuration
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = kSharedMemorySize;
    config.stream = stream;

    // Add special attribute for PDL
    config.attrs = attribute;
    config.numAttrs = 1;

    auto kernel = kernels::group_gemm_groupwise_w4a8_mma_kernel<64>;

    cudaLaunchKernelEx(
        &config, kernel, reinterpret_cast<Tout *>(y_ptr), reinterpret_cast<const Tx *>(x_ptr),
        reinterpret_cast<const Tw *>(weight_ptr), reinterpret_cast<const int32_t *>(seqlens_ptr),
        reinterpret_cast<const int32_t *>(cu_seqlens_ptr), reinterpret_cast<const Ts *>(yscale_ptr),
        n, k);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("kernel launch error: %s - %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
    }
  }
}

}  // namespace group_gemm
}  // namespace hpc
