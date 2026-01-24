// Copyright 2025 hpc-ops authors
#include <cuda.h>

#include <iostream>

#include "src/normalization/fused_rmsnorm_rope.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace normalization {

namespace kernels {
template <typename Tin, int kWarpsPerRow, int kDim>
__global__ void fused_rmsnorm_rope_kernel(__nv_bfloat16* y_q_ptr, __nv_bfloat16* y_k_ptr,
                                          const Tin* q_ptr, const Tin* q_weight_ptr,
                                          const Tin* k_ptr, const Tin* k_weight_ptr,
                                          const int64_t* pos_ptr, const __nv_bfloat16* cos_sin_ptr,
                                          const int num_tokens, const int dim, const int rope_dim,
                                          const int num_q_heads, const int num_k_heads,
                                          const float eps) {
  constexpr int kN = 16 / sizeof(Tin);
  constexpr int kRowsPerBlock = 4 / kWarpsPerRow;
  constexpr float kInvHiddenSize = 1.0f / kDim;
  constexpr int kWarpsPerBlock = 4;

  __shared__ float smem_sum[2][kWarpsPerBlock];

  int idx = threadIdx.x;
  int iwarp = idx / 32;
  int ilane = idx % 32;

  const int irow = blockIdx.x * kRowsPerBlock + iwarp / kWarpsPerRow;
  const int col = ((iwarp % kWarpsPerRow) * 32 + ilane) * kN;
  if (irow >= num_tokens * num_q_heads || col >= kDim) {
    return;
  }
  auto* q_row_ptr = q_ptr + irow * kDim;
  auto* y_q_row_ptr = y_q_ptr + irow * kDim;

  // process q
  // 1.rmsnorm
  float rms = 0.0f;
  vec_t<float, kN> q_in;
  if constexpr (std::is_same_v<Tin, __nv_bfloat16>) {
    q_in = to<float>(load<__nv_bfloat162, kN / 2>(q_row_ptr + col));
  }
  if constexpr (std::is_same_v<Tin, float>) {
    q_in = load<float, kN>(q_row_ptr + col);
  }
#pragma unroll
  for (int i = 0; i < kN; i++) {
    rms += q_in[i] * q_in[i];
  }
  rms = warp_reduce_sum_xor(rms);
  if (ilane == 0) {
    smem_sum[0][iwarp] = rms;
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < kWarpsPerRow; i++) {
    if (iwarp % kWarpsPerRow != i) {
      rms += smem_sum[0][iwarp / kWarpsPerRow * kWarpsPerRow + i];
    }
  }
  rms = rsqrtf_ftz(rms * kInvHiddenSize + eps);
  if (q_weight_ptr) {
    vec_t<float, kN> q_w;
    if constexpr (std::is_same_v<Tin, __nv_bfloat16>) {
      q_w = to<float>(load<__nv_bfloat162, kN / 2>(q_weight_ptr + col));
    }
    if constexpr (std::is_same_v<Tin, float>) {
      q_w = load<float, kN>(q_weight_ptr + col);
    }
#pragma unroll
    for (int i = 0; i < kN; i++) {
      q_in[i] *= rms * q_w[i];
    }
  } else {
#pragma unroll
    for (int i = 0; i < kN; i++) {
      q_in[i] *= rms;
    }
  }
  // 2.store nope
  if (col < kDim - rope_dim) {
    store(y_q_row_ptr + col, to<__nv_bfloat16>(q_in));
  }

  // 3.rope
  if (col >= kDim - rope_dim) {
    int itoken = irow / num_q_heads;
    int pos = pos_ptr[itoken];
    // cos_sin is always bfloat16
    auto cos_sin = to<float>(
        load<__nv_bfloat162, kN / 2>(cos_sin_ptr + pos * rope_dim + col - (kDim - rope_dim)));
#pragma unroll
    for (int i = 0; i < kN / 2; i++) {
      float a = q_in[2 * i];
      float b = q_in[2 * i + 1];
      float cos = cos_sin[2 * i];
      float sin = cos_sin[2 * i + 1];
      q_in[2 * i] = a * cos - b * sin;      // a*cos - b*sin
      q_in[2 * i + 1] = a * sin + b * cos;  // a*sin + b*cos
    }
    // store rope
    store(y_q_row_ptr + col, to<__nv_bfloat16>(q_in));
  }

  // process k, same as q
  if (k_ptr) {
    if (irow >= num_tokens * num_k_heads) {
      return;
    }
    auto y_k_row_ptr = y_k_ptr + irow * kDim;
    auto* k_row_ptr = k_ptr + irow * kDim;
    float rms = 0.0f;
    vec_t<float, kN> k_in;
    if constexpr (std::is_same_v<Tin, __nv_bfloat16>) {
      k_in = to<float>(load<__nv_bfloat162, kN / 2>(k_row_ptr + col));
    }
    if constexpr (std::is_same_v<Tin, float>) {
      k_in = load<float, kN>(k_row_ptr + col);
    }
#pragma unroll
    for (int i = 0; i < kN; i++) {
      rms += k_in[i] * k_in[i];
    }
    rms = warp_reduce_sum_xor(rms);
    if (ilane == 0) {
      smem_sum[1][iwarp] = rms;
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < kWarpsPerRow; i++) {
      if (iwarp % kWarpsPerRow != i) {
        rms += smem_sum[1][iwarp / kWarpsPerRow * kWarpsPerRow + i];
      }
    }
    rms = rsqrtf_ftz(rms * kInvHiddenSize + eps);
    if (k_weight_ptr) {
      vec_t<float, kN> k_w;
      if constexpr (std::is_same_v<Tin, __nv_bfloat16>) {
        k_w = to<float>(load<__nv_bfloat162, kN / 2>(k_weight_ptr + col));
      }
      if constexpr (std::is_same_v<Tin, float>) {
        k_w = load<float, kN>(k_weight_ptr + col);
      }
#pragma unroll
      for (int i = 0; i < kN; i++) {
        k_in[i] *= rms * k_w[i];
      }
    } else {
#pragma unroll
      for (int i = 0; i < kN; i++) {
        k_in[i] *= rms;
      }
    }
    // 2.store nope
    if (col < kDim - rope_dim) {
      store(y_k_row_ptr + col, to<__nv_bfloat16>(k_in));
    }
    if (col >= kDim - rope_dim) {
      int itoken = irow / num_k_heads;
      int pos = pos_ptr[itoken];
      // cos_sin is always bfloat16
      auto cos_sin = to<float>(
          load<__nv_bfloat162, kN / 2>(cos_sin_ptr + pos * rope_dim + col - (kDim - rope_dim)));
#pragma unroll
      for (int i = 0; i < kN / 2; i++) {
        float a = k_in[2 * i];
        float b = k_in[2 * i + 1];
        float cos = cos_sin[2 * i];
        float sin = cos_sin[2 * i + 1];
        k_in[2 * i] = a * cos - b * sin;  // a*cos - b*sin
        k_in

            [2 * i + 1] = a * sin + b * cos;  // a*sin + b*cos
      }
      store(y_k_row_ptr + col, to<__nv_bfloat16>(k_in));
    }
  }
}

}  // namespace kernels

void fused_rmsnorm_rope_async(void* y_q_ptr, void* y_k_ptr, const void* q_ptr,
                              const void* q_weight_ptr, const void* k_ptr, const void* k_weight_ptr,
                              const void* pos_ptr, const void* cos_sin_ptr, const int num_tokens,
                              const int dim, const int rope_dim, const int num_q_heads,
                              const int num_k_heads, const float eps, const int dtype,
                              cudaStream_t stream) {
  constexpr int kWarpCount = 4;
  constexpr int kWarpSize = 32;
  dim3 block(kWarpSize * kWarpCount);
  if (dim == 128) {
    constexpr int kDim = 128;
    constexpr int kWarpsPerRow = 1;
    constexpr int kRowsPerBlock = kWarpCount / kWarpsPerRow;
    dim3 grid((num_tokens * num_q_heads + kRowsPerBlock - 1) / kRowsPerBlock);
    if (dtype == 0) {
      using Tin = __nv_bfloat16;
      kernels::fused_rmsnorm_rope_kernel<Tin, kWarpsPerRow, kDim><<<grid, block, 0, stream>>>(
          (__nv_bfloat16*)y_q_ptr, (__nv_bfloat16*)y_k_ptr, (Tin*)q_ptr, (Tin*)q_weight_ptr,
          (Tin*)k_ptr, (Tin*)k_weight_ptr, (int64_t*)pos_ptr, (__nv_bfloat16*)cos_sin_ptr,
          num_tokens, dim, rope_dim, num_q_heads, num_k_heads, eps);
    } else if (dtype == 1) {
      using Tin = float;
      kernels::fused_rmsnorm_rope_kernel<Tin, kWarpsPerRow, kDim><<<grid, block, 0, stream>>>(
          (__nv_bfloat16*)y_q_ptr, (__nv_bfloat16*)y_k_ptr, (Tin*)q_ptr, (Tin*)q_weight_ptr,
          (Tin*)k_ptr, (Tin*)k_weight_ptr, (int64_t*)pos_ptr, (__nv_bfloat16*)cos_sin_ptr,
          num_tokens, dim, rope_dim, num_q_heads, num_k_heads, eps);
    }
  }
  if (dim == 512) {
    if (dtype == 0) {
      using Tin = __nv_bfloat16;
      constexpr int kDim = 512;
      constexpr int kWarpsPerRow = 2;
      constexpr int kRowsPerBlock = kWarpCount / kWarpsPerRow;
      dim3 grid((num_tokens * num_q_heads + kRowsPerBlock - 1) / kRowsPerBlock);
      kernels::fused_rmsnorm_rope_kernel<Tin, kWarpsPerRow, kDim><<<grid, block, 0, stream>>>(
          (__nv_bfloat16*)y_q_ptr, (__nv_bfloat16*)y_k_ptr, (Tin*)q_ptr, (Tin*)q_weight_ptr,
          (Tin*)k_ptr, (Tin*)k_weight_ptr, (int64_t*)pos_ptr, (__nv_bfloat16*)cos_sin_ptr,
          num_tokens, dim, rope_dim, num_q_heads, num_k_heads, eps);
    } else if (dtype == 1) {
      using Tin = float;
      constexpr int kDim = 512;
      constexpr int kWarpsPerRow = 4;
      constexpr int kRowsPerBlock = kWarpCount / kWarpsPerRow;
      dim3 grid((num_tokens * num_q_heads + kRowsPerBlock - 1) / kRowsPerBlock);
      kernels::fused_rmsnorm_rope_kernel<Tin, kWarpsPerRow, kDim><<<grid, block, 0, stream>>>(
          (__nv_bfloat16*)y_q_ptr, (__nv_bfloat16*)y_k_ptr, (Tin*)q_ptr, (Tin*)q_weight_ptr,
          (Tin*)k_ptr, (Tin*)k_weight_ptr, (int64_t*)pos_ptr, (__nv_bfloat16*)cos_sin_ptr,
          num_tokens, dim, rope_dim, num_q_heads, num_k_heads, eps);
    }
  }
}

}  // namespace normalization
}  // namespace hpc
