// Copyright 2025 hpc-ops authors
#include <cuda.h>
#include <cuda_fp8.h>

#include <iostream>

#include "src/hadamard/hadamard.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace hadamard {
namespace kernels {

// ============================
//   Primitive Hadamard Ops
// ============================

template <int kStride, int kBaseSize>
__device__ __forceinline__ void unit_hadamard(vec_t<float, kBaseSize>& data, int ilane) {
  const float sign = (ilane & kStride) ? -1.f : 1.f;
#pragma unroll
  for (int i = 0; i < kBaseSize; i++) {
    float other = __shfl_xor_sync(0xFFFFFFFF, data[i], kStride);
    data[i] = sign * data[i] + other;
  }
}

template <int kBaseSize>
__device__ __forceinline__ void base_hadamard(vec_t<float, kBaseSize>& x);

template <>
__device__ __forceinline__ void base_hadamard<4>(vec_t<float, 4>& x) {
  float y0 = x[0] + x[1] + x[2] + x[3];
  float y1 = x[0] - x[1] + x[2] - x[3];
  float y2 = x[0] + x[1] - x[2] - x[3];
  float y3 = x[0] - x[1] - x[2] + x[3];
  x[0] = y0;
  x[1] = y1;
  x[2] = y2;
  x[3] = y3;
}

// n=64 device Hadamard Compute
// 1 warp for 2 rows of n=64 hadamard
__device__ __forceinline__ void hadamard_n64_warp(vec_t<float, 4>& data, int ilane) {
  // 4 rounds of butterfly
  unit_hadamard<1, 4>(data, ilane);
  unit_hadamard<2, 4>(data, ilane);
  unit_hadamard<4, 4>(data, ilane);
  unit_hadamard<8, 4>(data, ilane);
  // 4-point intra-thread Hadamard
  base_hadamard<4>(data);
}

// ============================
//   Optimized Hadamard Kernel for n=64
// ============================
// For n=64: kBaseSize=4, kActiveThreads=16, kUnitIters=4
// So each warp processes TWO rows simultaneously:
//   lanes  0-15 → row (2*iwarp + 0)
//   lanes 16-31 → row (2*iwarp + 1)
// DType: input/output data type (__nv_bfloat16 or float)
template <typename DType, int kNumWarps>
__global__ void hadamard_transform_n64(const DType* input_ptr, DType* output_ptr, float inv_sqrt_d,
                                       int num_rows) {
  constexpr int kWarpSize = 32;
  constexpr int kN = 64;
  constexpr int kBaseSize = 4;
  constexpr int kRowsPerWarp = 2;  // each warp handles 2 rows in parallel

  const int iwarp = threadIdx.x / kWarpSize;
  const int ilane = threadIdx.x % kWarpSize;
  const int local_lane = ilane % 16;    // position within the 16-thread half
  const int irow_in_warp = ilane / 16;  // 0 for lanes 0-15, 1 for lanes 16-31

  const int ibatch = blockIdx.x * (kNumWarps * kRowsPerWarp) + iwarp * kRowsPerWarp + irow_in_warp;
  const bool is_valid_row = (ibatch < num_rows);

  // Initialize to 0; inis_valid_row lanes still participate in shfl (required by __shfl_xor_sync)
  vec_t<float, kBaseSize> data = {};

  // Step 1: load from global memory
  if (is_valid_row) {
    auto v = load<DType, kBaseSize>(input_ptr + ibatch * kN + local_lane * kBaseSize);
    if constexpr (std::is_same_v<DType, __nv_bfloat16>) {
#pragma unroll
      for (int i = 0; i < kBaseSize; i++) {
        data[i] = __bfloat162float(v[i]);
      }
    } else {
#pragma unroll
      for (int i = 0; i < kBaseSize; i++) {
        data[i] = static_cast<float>(v[i]);
      }
    }
  }

  // Step 2: compute Hadamard transform on reg
  hadamard_n64_warp(data, ilane);

  // Step 3: scale and store directly to global memory
  if (is_valid_row) {
    vec_t<DType, kBaseSize> v;
#pragma unroll
    for (int i = 0; i < kBaseSize; i++) {
      float val = data[i] * inv_sqrt_d;
      if constexpr (std::is_same_v<DType, __nv_bfloat16>) {
        v[i] = __float2bfloat16(val);
      } else {
        v[i] = static_cast<DType>(val);
      }
    }
    store(output_ptr + ibatch * kN + local_lane * kBaseSize, v);
  }
}

constexpr float kActMulHadamardEpsilon = 1e-8f;

template <int kNumWarps, bool kUsePDL = false>
__global__ void act_mul_hadamard_64_blockwise_quant_kernel(
    const __nv_bfloat16* gate_up_ptr, __nv_fp8_e4m3* output_ptr, float* output_scale_ptr,
    const int* valid_row_range_ptr, int num_rows, int num_col, float upper_max) {
  constexpr int kWarpSize = 32;
  constexpr int kN = 64;        // one scale block width
  constexpr int kBaseSize = 4;  // elements per thread
  constexpr int kRowsPerWarp = 2;
  constexpr float kInvSqrt64 = 0.125f;  // 1/sqrt(64)

  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }

  const int iwarp = threadIdx.x / kWarpSize;
  const int ilane = threadIdx.x % kWarpSize;
  const int local_lane = ilane % 16;    // lane within each 16-thread half
  const int irow_in_warp = ilane / 16;  // 0 → lanes 0-15, 1 → lanes 16-31

  // gridDim.x = num_col_blocks: each block owns one 64-wide column block
  const int icol_block = blockIdx.x;
  const int icol_start = icol_block * kN;

  // gridDim.y covers rows: each block handles kNumWarps*2 rows
  const int irow = blockIdx.y * (kNumWarps * kRowsPerWarp) + iwarp * kRowsPerWarp + irow_in_warp;

  const bool is_valid_row =
      (irow < num_rows) && (valid_row_range_ptr == nullptr || irow < valid_row_range_ptr[0]);

  vec_t<float, kBaseSize> data = {0};

  // Step 1: load gate and up, compute silu(gate)*up
  if (is_valid_row) {
    const __nv_bfloat16* gate_row = gate_up_ptr + irow * (num_col * 2);
    const __nv_bfloat16* up_row = gate_row + num_col;
    int elem_off = icol_start + local_lane * kBaseSize;

    auto gate_v = load<__nv_bfloat16, kBaseSize>(gate_row + elem_off);
    auto up_v = load<__nv_bfloat16, kBaseSize>(up_row + elem_off);
#pragma unroll
    for (int i = 0; i < kBaseSize; i++) {
      float g = __bfloat162float(gate_v[i]);
      float u = __bfloat162float(up_v[i]);
      data[i] = silu(g) * u;
    }
  }

  // Step 2: Hadamard transform
  hadamard_n64_warp(data, ilane);

  // Step 3: Hadamard scale by 1/sqrt(64)
#pragma unroll
  for (int i = 0; i < kBaseSize; i++) {
    data[i] *= kInvSqrt64;
  }

  // Step 4: blockwise quantization — reduce max over the 16-lane half-warp
  float thread_max = 0.f;
  if (is_valid_row) {
#pragma unroll
    for (int i = 0; i < kBaseSize; i++) {
      thread_max = fmaxf(thread_max, fabsf(data[i]));
    }
  }
  // Reduce within each 16-lane group
  thread_max = half_warp_reduce_max_down(thread_max);

  float scale = thread_max / upper_max;
  float inv_scale = rcpf_ftz(scale + kActMulHadamardEpsilon);

  // Step 5: quantize and store fp8 output + scale
  if (is_valid_row) {
    vec_t<float, kBaseSize> out_f;
#pragma unroll
    for (int i = 0; i < kBaseSize; i++) {
      out_f[i] = data[i] * inv_scale;
    }
    auto out_fp8 = to<__nv_fp8x4_e4m3>(out_f);

    __nv_fp8_e4m3* out_row = output_ptr + irow * num_col;
    store(out_row + icol_start + local_lane * kBaseSize, out_fp8);

    // scale layout: [num_col_blocks, num_rows]  (N-major)
    // lane 0 of each 16-lane group writes the scale
    if (local_lane == 0) {
      output_scale_ptr[icol_block * num_rows + irow] = scale;
    }
  }

  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

template <int kNumWarps, bool kUsePDL = false>
__global__ void act_mul_hadamard_64_per_tensor_quant_kernel(const __nv_bfloat16* gate_up_ptr,
                                                            __nv_fp8_e4m3* output_ptr,
                                                            const float* scale_inv_ptr,
                                                            const int* valid_row_range_ptr,
                                                            int num_rows, int num_col) {
  constexpr int kWarpSize = 32;
  constexpr int kN = 64;
  constexpr int kBaseSize = 4;
  constexpr int kRowsPerWarp = 2;
  constexpr float kInvSqrt64 = 0.125f;

  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }

  const float scale_inv = *scale_inv_ptr;

  const int iwarp = threadIdx.x / kWarpSize;
  const int ilane = threadIdx.x % kWarpSize;
  const int local_lane = ilane % 16;
  const int irow_in_warp = ilane / 16;

  const int icol_block = blockIdx.x;
  const int icol_start = icol_block * kN;

  const int irow = blockIdx.y * (kNumWarps * kRowsPerWarp) + iwarp * kRowsPerWarp + irow_in_warp;

  const bool is_valid_row =
      (irow < num_rows) && (valid_row_range_ptr == nullptr || irow < valid_row_range_ptr[0]);

  vec_t<float, kBaseSize> data = {0};

  // Step 1: load gate and up, compute silu(gate)*up
  if (is_valid_row) {
    const __nv_bfloat16* gate_row = gate_up_ptr + irow * (num_col * 2);
    const __nv_bfloat16* up_row = gate_row + num_col;
    int elem_off = icol_start + local_lane * kBaseSize;

    auto gate_v = load<__nv_bfloat16, kBaseSize>(gate_row + elem_off);
    auto up_v = load<__nv_bfloat16, kBaseSize>(up_row + elem_off);
#pragma unroll
    for (int i = 0; i < kBaseSize; i++) {
      float g = __bfloat162float(gate_v[i]);
      float u = __bfloat162float(up_v[i]);
      data[i] = silu(g) * u;
    }
  }

  // Step 2: Hadamard transform
  hadamard_n64_warp(data, ilane);

  // Step 3: scale by 1/sqrt(64) then quantize with per-tensor scale_inv
  if (is_valid_row) {
    vec_t<float, kBaseSize> out_f;
#pragma unroll
    for (int i = 0; i < kBaseSize; i++) {
      out_f[i] = data[i] * kInvSqrt64 * scale_inv;
    }
    auto out_fp8 = to<__nv_fp8x4_e4m3>(out_f);

    __nv_fp8_e4m3* out_row = output_ptr + irow * num_col;
    store(out_row + icol_start + local_lane * kBaseSize, out_fp8);
  }

  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

// ============================
//   General Hadamard Kernel
// ============================
// DType: input/output data type (e.g., __nv_bfloat16)
// kNumWarps: number of warps per block (each warp handles one row)
// kN: dimension of each row (e.g., 64)
// kBaseSize: base Hadamard size (e.g., 4)
// kUnitIters: number of unit Hadamard iterations (e.g., 4 for n=64)
template <typename DType, int kNumWarps, int kN, int kBaseSize, int kUnitIters>
__global__ void hadamard_transform(const DType* input_ptr, DType* output_ptr, float inv_sqrt_d,
                                   int num_rows) {
  constexpr int kWarpSize = 32;
  constexpr int kActiveThreads = kN / kBaseSize;   // threads needed per row
  constexpr int kItemPer16B = 16 / sizeof(DType);  // elements per 16-byte load
  constexpr int kTotalElements = kNumWarps * kN;   // total elements per block in smem
  // Number of 16-byte loads needed to fill shared memory for all rows in one block
  constexpr int kLoadsTotal = kTotalElements / kItemPer16B;
  constexpr int kBlockSize = kNumWarps * kWarpSize;
  // Each thread may need to do multiple 16-byte loads
  constexpr int kLoadsPerThread = (kLoadsTotal + kBlockSize - 1) / kBlockSize;

  static_assert(kActiveThreads <= kWarpSize, "kActiveThreads must be <= warp size");
  static_assert(kN % kItemPer16B == 0, "Dim of each row should be divisable by 16Byte");

  const int iwarp = threadIdx.x / kWarpSize;
  const int ilane = threadIdx.x % kWarpSize;
  const int ibatch = blockIdx.x * kNumWarps + iwarp;

  using Vec16B = vec_t<DType, kItemPer16B>;

  // Shared memory for all rows in this block
  extern __shared__ char smem_raw[];
  DType* smem = reinterpret_cast<DType*>(smem_raw);

  // Step 1: Cooperatively load all rows from global memory to shared memory
  // using maximum bandwidth (16-byte loads) with loop unrolling
#pragma unroll
  for (int i = 0; i < kLoadsPerThread; i++) {
    int load_idx = threadIdx.x + i * kBlockSize;
    if (load_idx < kLoadsTotal) {
      int elem_offset = load_idx * kItemPer16B;
      int row = elem_offset / kN;
      int col = elem_offset % kN;
      int global_row = blockIdx.x * kNumWarps + row;
      if (global_row < num_rows) {
        Vec16B v = load<DType, kItemPer16B>(input_ptr + global_row * kN + col);
        store(smem + elem_offset, v);
      }
    }
  }
  __syncthreads();

  // Step 2: Each warp reads its row from shared memory into registers
  vec_t<float, kBaseSize> data;
  if (ibatch < num_rows) {
    if (ilane < kActiveThreads) {
#pragma unroll
      for (int i = 0; i < kBaseSize; i++) {
        int col = ilane * kBaseSize + i;
        if constexpr (std::is_same_v<DType, __nv_bfloat16>) {
          data[i] = __bfloat162float(smem[iwarp * kN + col]);
        } else {
          data[i] = static_cast<float>(smem[iwarp * kN + col]);
        }
      }
    }

    // Step 3: Perform kUnitIters levels of unit Hadamard transform (inter-thread butterfly)
    if constexpr (kUnitIters >= 1) unit_hadamard<1, kBaseSize>(data, ilane);
    if constexpr (kUnitIters >= 2) unit_hadamard<2, kBaseSize>(data, ilane);
    if constexpr (kUnitIters >= 3) unit_hadamard<4, kBaseSize>(data, ilane);
    if constexpr (kUnitIters >= 4) unit_hadamard<8, kBaseSize>(data, ilane);
    if constexpr (kUnitIters >= 5) unit_hadamard<16, kBaseSize>(data, ilane);

    // Step 4: Perform base Hadamard transform (intra-thread)
    base_hadamard<kBaseSize>(data);

    // Step 5: Scale by 1/sqrt(d) and write back to shared memory
    if (ilane < kActiveThreads) {
#pragma unroll
      for (int i = 0; i < kBaseSize; i++) {
        float val = data[i] * inv_sqrt_d;
        int col = ilane * kBaseSize + i;
        if constexpr (std::is_same_v<DType, __nv_bfloat16>) {
          smem[iwarp * kN + col] = __float2bfloat16(val);
        } else {
          smem[iwarp * kN + col] = static_cast<DType>(val);
        }
      }
    }
  }
  __syncthreads();

  // Step 6: Cooperatively store all rows from shared memory to global memory
#pragma unroll
  for (int i = 0; i < kLoadsPerThread; i++) {
    int load_idx = threadIdx.x + i * kBlockSize;
    if (load_idx < kLoadsTotal) {
      int elem_offset = load_idx * kItemPer16B;
      int row = elem_offset / kN;
      int col = elem_offset % kN;
      int global_row = blockIdx.x * kNumWarps + row;
      if (global_row < num_rows) {
        Vec16B v = load<DType, kItemPer16B>(smem + elem_offset);
        store(output_ptr + global_row * kN + col, v);
      }
    }
  }
}

}  // namespace kernels

bool hadamard_transform_async(const void* input_ptr, void* output_ptr, float inv_sqrt_d, int n,
                              int num_rows, int input_elem_size, cudaStream_t stream) {
  constexpr int kNumWarps = 4;
  constexpr int kWarpSize = 32;
  constexpr int kBlockSize = kNumWarps * kWarpSize;

  dim3 block(kBlockSize);

  if (input_elem_size == 2) {
    using DType = __nv_bfloat16;
    if (n == 64) {
      constexpr int kRowsPerBlock64 = kNumWarps * 2;
      dim3 grid((num_rows + kRowsPerBlock64 - 1) / kRowsPerBlock64);
      kernels::hadamard_transform_n64<DType, kNumWarps>
          <<<grid, block, 0, stream>>>(reinterpret_cast<const DType*>(input_ptr),
                                       reinterpret_cast<DType*>(output_ptr), inv_sqrt_d, num_rows);
    } else {
      std::cout << "not supported dimension for hadamard_transform_async: " << n << std::endl;
      return false;
    }
  } else if (input_elem_size == 4) {
    using DType = float;
    if (n == 64) {
      constexpr int kRowsPerBlock64 = kNumWarps * 2;
      dim3 grid((num_rows + kRowsPerBlock64 - 1) / kRowsPerBlock64);
      kernels::hadamard_transform_n64<DType, kNumWarps>
          <<<grid, block, 0, stream>>>(reinterpret_cast<const DType*>(input_ptr),
                                       reinterpret_cast<DType*>(output_ptr), inv_sqrt_d, num_rows);
    } else {
      std::cout << "not supported dimension for hadamard_transform_async: " << n << std::endl;
      return false;
    }
  } else {
    std::cout << "not supported elem size for hadamard_transform_async: " << input_elem_size
              << std::endl;
    return false;
  }
  return true;
}

bool act_mul_hadamard_blockwise_quant_async(const __nv_bfloat16* gate_up_ptr,
                                            __nv_fp8_e4m3* output_ptr, float* output_scale_ptr,
                                            const int* valid_row_range_ptr, int num_rows,
                                            int num_col, float upper_max, int block_size,
                                            bool use_pdl, cudaStream_t stream) {
  if (block_size != 64) {
    std::cout << "act_mul_hadamard_blockwise_quant_async: block_size=" << block_size
              << " not impl (only 64 supported)" << std::endl;
    return false;
  }
  if (num_col % block_size != 0) {
    std::cout << "act_mul_hadamard_blockwise_quant_async: num_col must be a multiple of block_size="
              << block_size << ", got " << num_col << std::endl;
    return false;
  }

  const int num_col_blocks = num_col / block_size;

  constexpr int kNumWarps = 4;
  constexpr int kWarpSize = 32;
  constexpr int kBlockSize = kNumWarps * kWarpSize;
  constexpr int kRowsPerBlock = kNumWarps * 2;

  // gridDim.x = num_col_blocks: one block per 64-wide column group
  // gridDim.y: each block handles kNumWarps*2 rows
  dim3 grid(num_col_blocks, (num_rows + kRowsPerBlock - 1) / kRowsPerBlock);
  dim3 block(kBlockSize);

  if (use_pdl) {
    constexpr bool kUsePDL = true;

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    config.attrs = attribute;
    config.numAttrs = 1;

    auto kernel = kernels::act_mul_hadamard_64_blockwise_quant_kernel<kNumWarps, kUsePDL>;
    cudaLaunchKernelEx(&config, kernel, gate_up_ptr, output_ptr, output_scale_ptr,
                       valid_row_range_ptr, num_rows, num_col, upper_max);
  } else {
    constexpr bool kUsePDL = false;
    kernels::act_mul_hadamard_64_blockwise_quant_kernel<kNumWarps, kUsePDL>
        <<<grid, block, 0, stream>>>(gate_up_ptr, output_ptr, output_scale_ptr, valid_row_range_ptr,
                                     num_rows, num_col, upper_max);
  }
  return true;
}

bool act_mul_hadamard_blockwise_quant_async(const __nv_bfloat16* gate_up_ptr,
                                            __nv_fp8_e4m3* output_ptr, float* output_scale_ptr,
                                            const int* valid_row_range_ptr, int num_rows,
                                            int num_col, cudaStream_t stream) {
  return act_mul_hadamard_blockwise_quant_async(gate_up_ptr, output_ptr, output_scale_ptr,
                                                valid_row_range_ptr, num_rows, num_col, 448.0f, 64,
                                                true, stream);
}

bool act_mul_hadamard_per_tensor_quant_async(const __nv_bfloat16* gate_up_ptr,
                                             __nv_fp8_e4m3* output_ptr, const float* scale_inv_ptr,
                                             const int* valid_row_range_ptr, int num_rows,
                                             int num_col, bool use_pdl, cudaStream_t stream) {
  constexpr int kHadamardWidth = 64;
  if (num_col % kHadamardWidth != 0) {
    std::cout << "act_mul_hadamard_per_tensor_quant_async: num_col must be a multiple of "
                 "kHadamardWidth, got "
              << num_col << std::endl;
    return false;
  }

  const int num_col_blocks = num_col / kHadamardWidth;

  constexpr int kNumWarps = 4;
  constexpr int kWarpSize = 32;
  constexpr int kBlockSize = kNumWarps * kWarpSize;
  constexpr int kRowsPerBlock = kNumWarps * 2;

  dim3 grid(num_col_blocks, (num_rows + kRowsPerBlock - 1) / kRowsPerBlock);
  dim3 block(kBlockSize);

  if (use_pdl) {
    constexpr bool kUsePDL = true;

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    config.attrs = attribute;
    config.numAttrs = 1;

    auto kernel = kernels::act_mul_hadamard_64_per_tensor_quant_kernel<kNumWarps, kUsePDL>;
    cudaLaunchKernelEx(&config, kernel, gate_up_ptr, output_ptr, scale_inv_ptr, valid_row_range_ptr,
                       num_rows, num_col);
  } else {
    constexpr bool kUsePDL = false;
    kernels::act_mul_hadamard_64_per_tensor_quant_kernel<kNumWarps, kUsePDL>
        <<<grid, block, 0, stream>>>(gate_up_ptr, output_ptr, scale_inv_ptr, valid_row_range_ptr,
                                     num_rows, num_col);
  }
  return true;
}

bool act_mul_hadamard_per_tensor_quant_async(const __nv_bfloat16* gate_up_ptr,
                                             __nv_fp8_e4m3* output_ptr, const float* scale_inv_ptr,
                                             const int* valid_row_range_ptr, int num_rows,
                                             int num_col, cudaStream_t stream) {
  return act_mul_hadamard_per_tensor_quant_async(
      gate_up_ptr, output_ptr, scale_inv_ptr, valid_row_range_ptr, num_rows, num_col, true, stream);
}

}  // namespace hadamard
}  // namespace hpc
