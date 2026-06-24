// Copyright 2026 hpc-ops authors

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdio.h>

#include <type_traits>

#include "src/fuse_moe/sm100/mxfp8/fuse_moe_mxfp8.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace fuse_moe {
namespace kernels_mxfp8 {

// UE8M0 scale calculation: SF = 2^k where k = ceil(log2(absmax) - log2(448)).
__device__ __forceinline__ uint8_t fp32_absmax_to_ue8m0(float absmax) {
  if (absmax == 0.f) {
    return 0;
  }
  uint32_t bits = __float_as_uint(absmax);
  int exp_biased = (bits >> 23) & 0xFF;
  uint32_t mant = bits & 0x7FFFFF;
  int sf_bits = exp_biased - 8 + (mant > 0x600000u ? 1 : 0);
  if (sf_bits < 0) {
    sf_bits = 0;
  }
  if (sf_bits > 255) {
    sf_bits = 255;
  }
  return static_cast<uint8_t>(sf_bits);
}

template <int kThreadPerBlock, bool kUsePDL = false>
__global__ void act_mul_mxfp8_kernel(const __nv_bfloat16 *__restrict__ gate_up_ptr,
                                     __nv_fp8_e4m3 *__restrict__ out_ptr,
                                     uint8_t *__restrict__ out_scale_packed_ptr,
                                     const int *__restrict__ valid_row_range_ptr,
                                     int intermediate_size) {
  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }
  constexpr int kSfVec = 32;
  constexpr int kElemsPerIter = 8;
  constexpr int kNumIter = kSfVec / kElemsPerIter;  // = 4

  const int K_sf = intermediate_size / kSfVec;

  int irow = blockIdx.y;
  int ikblock = blockIdx.x * kThreadPerBlock + threadIdx.x;

  if (ikblock >= K_sf) {
    return;
  }

  int valid_rows = *valid_row_range_ptr;
  if (irow >= valid_rows) {
    return;
  }

  const int k_base = ikblock * kSfVec;
  const __nv_bfloat16 *row_gate = gate_up_ptr + static_cast<uint64_t>(irow) * intermediate_size * 2;
  const __nv_bfloat16 *row_up = row_gate + intermediate_size;
  __nv_fp8_e4m3 *out_row = out_ptr + static_cast<uint64_t>(irow) * intermediate_size + k_base;

  using T = __nv_bfloat162;

  // Pass 1: compute y = silu(gate) * up, find absmax over 32 elements
  float y_buf[kSfVec];
  float absmax = 0.f;

#pragma unroll
  for (int iter = 0; iter < kNumIter; ++iter) {
    int offset = k_base + iter * kElemsPerIter;
    auto gate = to<float>(load<T, 4>(row_gate + offset));
    auto up = to<float>(load<T, 4>(row_up + offset));
#pragma unroll
    for (int i = 0; i < kElemsPerIter; ++i) {
      float v = silu(gate[i]) * up[i];
      y_buf[iter * kElemsPerIter + i] = v;
      absmax = fmaxf(absmax, fabsf(v));
    }
  }

  // Compute UE8M0 scale from absmax
  uint8_t sf_bits = fp32_absmax_to_ue8m0(absmax);

  float inv_sf;
  if (sf_bits == 0) {
    inv_sf = 0.f;
  } else {
    float sf = exp2f_ftz(static_cast<float>(sf_bits) - 127.f);
    inv_sf = 1.f / sf;
  }

  // Pass 2: quantize 32 fp8 elements and store as 2 x 16-byte writes
  vec_t<__nv_fp8_e4m3, 16> out_fp8_lo, out_fp8_hi;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    out_fp8_lo[i] = __nv_fp8_e4m3(y_buf[i] * inv_sf);
  }
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    out_fp8_hi[i] = __nv_fp8_e4m3(y_buf[16 + i] * inv_sf);
  }
  store<__nv_fp8_e4m3, 16>(out_row, out_fp8_lo);
  store<__nv_fp8_e4m3, 16>(out_row + 16, out_fp8_hi);

  // Store scale in row-major layout: sfx_ptr[irow * K_sf + ikblock]
  // The downstream GEMM kernel does inline prepack via cp.async.
  out_scale_packed_ptr[static_cast<uint64_t>(irow) * K_sf + ikblock] = sf_bits;

  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

}  // namespace kernels_mxfp8

void act_mul_and_mxfp8_quant_async(void *out_ptr, void *out_scale_ptr, const void *gate_up_ptr,
                                   const void *valid_row_range_ptr, int total_num_seq,
                                   int intermediate_size, cudaStream_t stream, bool use_pdl) {
  constexpr int kSfVec = 32;
  constexpr int kThreadPerBlock = 128;

  int K_sf = intermediate_size / kSfVec;
  int gridx = (K_sf + kThreadPerBlock - 1) / kThreadPerBlock;

  dim3 grid(gridx, total_num_seq);
  dim3 block(kThreadPerBlock);

  if (use_pdl) {
    auto kernel = kernels_mxfp8::act_mul_mxfp8_kernel<kThreadPerBlock, true>;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, kernel, reinterpret_cast<const __nv_bfloat16 *>(gate_up_ptr),
                       reinterpret_cast<__nv_fp8_e4m3 *>(out_ptr),
                       reinterpret_cast<uint8_t *>(out_scale_ptr),
                       reinterpret_cast<const int *>(valid_row_range_ptr), intermediate_size);
  } else {
    auto kernel = kernels_mxfp8::act_mul_mxfp8_kernel<kThreadPerBlock, false>;
    kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16 *>(gate_up_ptr),
        reinterpret_cast<__nv_fp8_e4m3 *>(out_ptr), reinterpret_cast<uint8_t *>(out_scale_ptr),
        reinterpret_cast<const int *>(valid_row_range_ptr), intermediate_size);
  }
}

}  // namespace fuse_moe
}  // namespace hpc
