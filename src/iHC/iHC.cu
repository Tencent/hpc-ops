// Copyright 2026 Tencent
// Copyright 2026, The FlagOS Contributors.

#include <cooperative_groups.h>
#include <cuda.h>
#include <stdio.h>

#include <algorithm>
#include <cstdlib>
#include <type_traits>

#include "cute/arch/simd_sm100.hpp"
#include "cutlass/fast_math.h"
#include "src/iHC/iHC.h"
#include "src/utils/utils.cuh"

#if HPC_TARGET_ARCH != 90
#include "cute/arch/copy_sm100.hpp"
#include "cute/arch/tmem_allocator_sm100.hpp"
#include "cute/atom/mma_traits_sm100.hpp"
#include "cutlass/arch/barrier.h"
#endif

namespace hpc {
namespace iHC {

// CuTe uses scalar FP32 fallbacks before SM100.
__device__ __forceinline__ float2 ihc_fma2(float2 a, float2 b, float2 c) {
  float2 d;
  cute::fma(d, a, b, c);
  return d;
}

__device__ __forceinline__ float2 ihc_mul2(float2 a, float2 b) {
  float2 c;
  cute::mul(c, a, b);
  return c;
}

template <bool kWeightsAligned32 = false>
__device__ __forceinline__ vec_t<float, 8> ihc_load_weight8(const float* weight_ptr) {
  vec_t<float, 8> weights;
#if defined(CUTE_ARCH_LOAD256_SM100A_ENABLED)
  if constexpr (kWeightsAligned32) {
    asm volatile("ld.global.v8.f32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8];"
                 : "=f"(weights[0]), "=f"(weights[1]), "=f"(weights[2]), "=f"(weights[3]),
                   "=f"(weights[4]), "=f"(weights[5]), "=f"(weights[6]), "=f"(weights[7])
                 : "l"(weight_ptr));
    return weights;
  }
#endif
  *reinterpret_cast<vec_t<float, 4>*>(&weights[0]) = load<float, 4>(weight_ptr);
  *reinterpret_cast<vec_t<float, 4>*>(&weights[4]) = load<float, 4>(weight_ptr + 4);
  return weights;
}

template <int kElementsPerThread>
__device__ __forceinline__ void ihc_dot_pair(float& acc0, float& acc1,
                                             const vec_t<float, kElementsPerThread>& values,
                                             const vec_t<float, kElementsPerThread>& weights0,
                                             const vec_t<float, kElementsPerThread>& weights1) {
  float2 acc = make_float2(acc0, acc1);
#pragma unroll
  for (int j = 0; j < kElementsPerThread; ++j) {
    acc.x += values[j] * weights0[j];
    acc.y += values[j] * weights1[j];
  }
  acc0 = acc.x;
  acc1 = acc.y;
}

template <int kHCMult, int kHiddenDim, int kElementsPerThread, int kRowsPerBlock,
          int kThreadsPerBlock, int kThreadsPerRow, bool kFuseRmsNorm, bool kCastBfloatForNorm>
__device__ __forceinline__ void ihc_mix_epilogue(__nv_bfloat16* z_ptr, const __nv_bfloat16* y_ptr,
                                                 const float* pre_gates_ptr, int row_base,
                                                 int first_y_row, int valid_rows,
                                                 const __nv_bfloat16* rms_weight_ptr, float rms_eps,
                                                 float* norm_partial) {
  static_assert(kThreadsPerRow % 32 == 0 && kThreadsPerBlock % kThreadsPerRow == 0,
                "each row group must contain whole warps");
  static_assert(kHiddenDim % (kThreadsPerRow * kElementsPerThread) == 0,
                "row span must divide hidden size");
  constexpr int kRowGroups = kThreadsPerBlock / kThreadsPerRow;
  constexpr int kWarpsPerRow = kThreadsPerRow / 32;
  constexpr int kChunksPerRow = kHiddenDim / (kThreadsPerRow * kElementsPerThread);
  const int group = threadIdx.x / kThreadsPerRow;
  const int tid = threadIdx.x % kThreadsPerRow;

#pragma unroll 1
  for (int wave = 0; wave < (kRowsPerBlock + kRowGroups - 1) / kRowGroups; ++wave) {
    const int r = wave * kRowGroups + group;
    const bool valid = r < valid_rows;
    using SavedOutput =
        std::conditional_t<kFuseRmsNorm && kCastBfloatForNorm, __nv_bfloat16, float>;
    vec_t<SavedOutput, kElementsPerThread> saved[kChunksPerRow];
    float output_sqsum = 0.f;
    if (valid) {
      const auto h = load<float, kHCMult>(pre_gates_ptr + r * kHCMult);
      const auto* y_row_ptr = y_ptr + static_cast<int64_t>(first_y_row + r) * kHCMult * kHiddenDim;
      auto* z_row_ptr = z_ptr + static_cast<int64_t>(row_base + r) * kHiddenDim;
#pragma unroll
      for (int ch = 0; ch < kChunksPerRow; ++ch) {
        const int col = (ch * kThreadsPerRow + tid) * kElementsPerThread;
        vec_t<float, kElementsPerThread> y;
#pragma unroll
        for (int j = 0; j < kElementsPerThread; ++j) {
          y[j] = 0.f;
        }
        auto* y2 = reinterpret_cast<float2*>(&y[0]);
#pragma unroll
        for (int c = 0; c < kHCMult; ++c) {
          const auto v =
              to<float>(load<__nv_bfloat16, kElementsPerThread>(y_row_ptr + c * kHiddenDim + col));
          const auto* v2 = reinterpret_cast<const float2*>(&v[0]);
#pragma unroll
          for (int j = 0; j < kElementsPerThread / 2; ++j) {
            y2[j] = ihc_fma2(make_float2(h[c], h[c]), v2[j], y2[j]);
          }
        }
        if constexpr (kFuseRmsNorm) {
          if constexpr (kCastBfloatForNorm) {
            saved[ch] = to<__nv_bfloat16>(y);
            y = to<float>(saved[ch]);
          } else {
            saved[ch] = y;
          }
#pragma unroll
          for (int j = 0; j < kElementsPerThread; ++j) {
            output_sqsum += y[j] * y[j];
          }
        } else {
          store(z_row_ptr + col, to<__nv_bfloat16>(y));
        }
      }
    }

    if constexpr (kFuseRmsNorm) {
      const float v = warp_reduce_sum_xor(output_sqsum);
      if ((tid % 32) == 0 && r < kRowsPerBlock) {
        norm_partial[r * kWarpsPerRow + tid / 32] = v;
      }
      __syncthreads();
      if (valid) {
        float total = 0.f;
#pragma unroll
        for (int w = 0; w < kWarpsPerRow; ++w) {
          total += norm_partial[r * kWarpsPerRow + w];
        }
        const float scale = rsqrtf_ftz(total * (1.f / kHiddenDim) + rms_eps);
        auto* z_row_ptr = z_ptr + static_cast<int64_t>(row_base + r) * kHiddenDim;
#pragma unroll
        for (int ch = 0; ch < kChunksPerRow; ++ch) {
          const int col = (ch * kThreadsPerRow + tid) * kElementsPerThread;
          const auto g = to<float>(load<__nv_bfloat16, kElementsPerThread>(rms_weight_ptr + col));
          auto y = to<float>(saved[ch]);
          auto* y2 = reinterpret_cast<float2*>(&y[0]);
          const auto* g2 = reinterpret_cast<const float2*>(&g[0]);
#pragma unroll
          for (int j = 0; j < kElementsPerThread / 2; ++j) {
            y2[j] = ihc_mul2(ihc_mul2(y2[j], make_float2(scale, scale)), g2[j]);
          }
          store(z_row_ptr + col, to<__nv_bfloat16>(y));
        }
      }
    }
  }
}

namespace kernels {

constexpr int ihc_post_pre_num_blocks(int hidden_dim) { return hidden_dim == 3072 ? 6 : 8; }

template <int kHiddenDim, int kBlocksPerCluster, bool kFuseRmsNorm, bool kCastBfloatForNorm,
          bool kUsePDL = false>
__device__ __forceinline__ void ihc_post_pre_cluster_device(
    __nv_bfloat16* y_ptr, __nv_bfloat16* z_ptr, float* H_post_out_ptr, const __nv_bfloat16* xa_ptr,
    const __nv_bfloat16* residual_ptr, const float* H_post_in_ptr, const float* w_ptr,
    const float* hc_scale_ptr, const float* hc_base_ptr, int num_batch, float norm_eps,
    float hc_eps, float magnitude, const __nv_bfloat16* rms_weight_ptr, float rms_eps) {
  constexpr int kElementsPerThread = 8,
                kThreadsPerBlock = kHiddenDim / (kBlocksPerCluster * kElementsPerThread),
                kHiddenPerBlock = kHiddenDim / kBlocksPerCluster,
                kWarpsPerBlock = kThreadsPerBlock / 32;
  static_assert(
      kHiddenDim % (kBlocksPerCluster * kElementsPerThread) == 0 && kThreadsPerBlock % 32 == 0,
      "shards must contain whole warps");
  static_assert(kBlocksPerCluster <= 8, "use a portable cluster size");
  const auto cluster = cooperative_groups::this_cluster();
  const int rank = cluster.block_rank(), row = blockIdx.x / kBlocksPerCluster;
  const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32,
            col = rank * kHiddenPerBlock + tid * kElementsPerThread;
  struct SharedStorage {
    float partial[9][kWarpsPerBlock];
    float sums[9], pre_gates[4], norm[kWarpsPerBlock], norm_sum, norm_scale;
  };
  __shared__ SharedStorage smem;
  if constexpr (kUsePDL) cudaGridDependencySynchronize();
  float acc_dot[8] = {}, acc_sqsum = 0.f;
  vec_t<__nv_bfloat16, kElementsPerThread> post_output_cache[4];
  const auto a = to<float>(load<__nv_bfloat16, kElementsPerThread>(
      xa_ptr + static_cast<int64_t>(row) * kHiddenDim + col));
  const auto h = load<float, 4>(H_post_in_ptr + row * 4);
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const int64_t flat = static_cast<int64_t>(row) * 4 * kHiddenDim + c * kHiddenDim + col;
    const auto b = to<float>(load<__nv_bfloat16, kElementsPerThread>(residual_ptr + flat));
    vec_t<float, kElementsPerThread> v;
    auto* v2 = reinterpret_cast<float2*>(&v[0]);
    const auto* a2 = reinterpret_cast<const float2*>(&a[0]);
    const auto* b2 = reinterpret_cast<const float2*>(&b[0]);
#pragma unroll
    for (int j = 0; j < kElementsPerThread / 2; ++j)
      v2[j] = ihc_fma2(make_float2(h[c], h[c]), a2[j], b2[j]);
    const auto bf = to<__nv_bfloat16>(v);
    store(y_ptr + flat, bf);
    post_output_cache[c] = bf;
    v = to<float>(bf);
#pragma unroll
    for (int j = 0; j < kElementsPerThread; ++j) acc_sqsum += v[j] * v[j];
#pragma unroll
    for (int p = 0; p < 8; ++p) {
      vec_t<float, kElementsPerThread> weights;
      const float* wp = w_ptr + static_cast<int64_t>(p) * 4 * kHiddenDim + c * kHiddenDim + col;
      *reinterpret_cast<vec_t<float, 4>*>(&weights[0]) = load<float, 4>(wp);
      *reinterpret_cast<vec_t<float, 4>*>(&weights[4]) = load<float, 4>(wp + 4);
#pragma unroll
      for (int j = 0; j < kElementsPerThread; ++j) acc_dot[p] += v[j] * weights[j];
    }
  }
#pragma unroll
  for (int p = 0; p < 8; ++p) {
    const float v = warp_reduce_sum_xor(acc_dot[p]);
    if (lane == 0) smem.partial[p][warp] = v;
  }
  const float v = warp_reduce_sum_xor(acc_sqsum);
  if (lane == 0) smem.partial[8][warp] = v;
  __syncthreads();
  if (tid < 9) {
    float total = 0;
#pragma unroll
    for (int k = 0; k < kWarpsPerBlock; ++k) total += smem.partial[tid][k];
    smem.sums[tid] = total;
  }
  // Publish all projection and input-norm partials before any DSM reads.
  cluster.sync();
  if (tid < 4) {
    float pre = 0, post = 0, total = 0;
#pragma unroll
    for (int k = 0; k < kBlocksPerCluster; ++k) {
      const auto* peer = cluster.map_shared_rank(&smem, k);
      pre += peer->sums[tid];
      post += peer->sums[tid + 4];
      total += peer->sums[8];
    }
    const float r = rsqrtf_ftz(total * (1.f / (4 * kHiddenDim)) + norm_eps);
    smem.pre_gates[tid] = sigmoid(hc_scale_ptr[0] * (pre * r) + hc_base_ptr[tid]) + hc_eps;
    if (rank == 0)
      H_post_out_ptr[row * 4 + tid] =
          magnitude * sigmoid(hc_scale_ptr[1] * (post * r) + hc_base_ptr[tid + 4]) + hc_eps;
  }
  __syncthreads();
  vec_t<float, kElementsPerThread> out;
#pragma unroll
  for (int j = 0; j < kElementsPerThread; ++j) out[j] = 0;
  auto* o2 = reinterpret_cast<float2*>(&out[0]);
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const auto in = to<float>(post_output_cache[c]);
    const auto* i2 = reinterpret_cast<const float2*>(&in[0]);
#pragma unroll
    for (int j = 0; j < kElementsPerThread / 2; ++j)
      o2[j] = ihc_fma2(make_float2(smem.pre_gates[c], smem.pre_gates[c]), i2[j], o2[j]);
  }
  if constexpr (kFuseRmsNorm) {
    if constexpr (kCastBfloatForNorm) out = to<float>(to<__nv_bfloat16>(out));
    float output_sqsum = 0;
#pragma unroll
    for (int j = 0; j < kElementsPerThread; ++j) output_sqsum += out[j] * out[j];
    const float t = warp_reduce_sum_xor(output_sqsum);
    if (lane == 0) smem.norm[warp] = t;
    __syncthreads();
    if (tid == 0) {
      float total = 0;
#pragma unroll
      for (int k = 0; k < kWarpsPerBlock; ++k) total += smem.norm[k];
      smem.norm_sum = total;
    }
    // Publish output-norm partials before reading peer CTAs.
    cluster.sync();
    if (tid == 0) {
      float total = 0;
#pragma unroll
      for (int k = 0; k < kBlocksPerCluster; ++k)
        total += cluster.map_shared_rank(&smem, k)->norm_sum;
      smem.norm_scale = rsqrtf_ftz(total * (1.f / kHiddenDim) + rms_eps);
    }
    __syncthreads();
    // Overlap the final cluster barrier with gamma loads and stores.
    cluster.barrier_arrive();
    const auto g = to<float>(load<__nv_bfloat16, kElementsPerThread>(rms_weight_ptr + col));
    const auto* g2 = reinterpret_cast<const float2*>(&g[0]);
#pragma unroll
    for (int j = 0; j < kElementsPerThread / 2; ++j)
      o2[j] = ihc_mul2(ihc_mul2(o2[j], make_float2(smem.norm_scale, smem.norm_scale)), g2[j]);
  } else {
    cluster.barrier_arrive();
  }
  store(z_ptr + static_cast<int64_t>(row) * kHiddenDim + col, to<__nv_bfloat16>(out));
  // Remote DSM readers must finish before any owning CTA exits.
  cluster.barrier_wait();
  if constexpr (kUsePDL) cudaTriggerProgrammaticLaunchCompletion();
}

template <int kHiddenDim, int kThreadsPerBlock, bool kFuseRmsNorm, bool kCastBfloatForNorm,
          bool kUsePDL = false, bool kWeightsAligned32 = false, bool kUsePairFma = false,
          bool kRetainPostOutput = true>
__device__ __forceinline__ void ihc_post_pre_single_row_device(
    __nv_bfloat16* y_ptr, __nv_bfloat16* z_ptr, float* H_post_out_ptr, const __nv_bfloat16* xa_ptr,
    const __nv_bfloat16* residual_ptr, const float* H_post_in_ptr, const float* w_ptr,
    const float* hc_scale_ptr, const float* hc_base_ptr, int num_batch, float norm_eps,
    float hc_eps, float magnitude, const __nv_bfloat16* rms_weight_ptr, float rms_eps) {
  constexpr int kElementsPerThread = 8,
                kChunksPerRow = kHiddenDim / (kThreadsPerBlock * kElementsPerThread),
                kWarpsPerBlock = kThreadsPerBlock / 32;
  static_assert(
      kThreadsPerBlock % 32 == 0 && kHiddenDim % (kThreadsPerBlock * kElementsPerThread) == 0,
      "whole warps and complete column tiles");
  const int row = blockIdx.x, tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
  if constexpr (kUsePDL) cudaGridDependencySynchronize();
  __shared__ float smem_partial[9][kWarpsPerBlock];
  __shared__ float smem_pre_gates[4], smem_norm_partial[kWarpsPerBlock], smem_rms_inv;
  vec_t<__nv_bfloat16, kElementsPerThread> post_output_cache[kChunksPerRow][4];
  float acc_dot[8] = {}, acc_sqsum = 0;
  auto h = load<float, 4>(H_post_in_ptr + row * 4);
#pragma unroll
  for (int ch = 0; ch < kChunksPerRow; ++ch) {
    int col = (ch * kThreadsPerBlock + tid) * kElementsPerThread;
    auto a = to<float>(load<__nv_bfloat16, kElementsPerThread>(
        xa_ptr + static_cast<int64_t>(row) * kHiddenDim + col));
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      auto b = to<float>(load<__nv_bfloat16, kElementsPerThread>(
          residual_ptr + static_cast<int64_t>(row) * 4 * kHiddenDim + i * kHiddenDim + col));
      vec_t<float, kElementsPerThread> v;
      auto* v2 = reinterpret_cast<float2*>(&v[0]);
      const auto* a2 = reinterpret_cast<const float2*>(&a[0]);
      const auto* b2 = reinterpret_cast<const float2*>(&b[0]);
#pragma unroll
      for (int j = 0; j < kElementsPerThread / 2; ++j)
        v2[j] = ihc_fma2(make_float2(h[i], h[i]), a2[j], b2[j]);
      auto bf = to<__nv_bfloat16>(v);
      if constexpr (kRetainPostOutput) {
        post_output_cache[ch][i] = bf;
      }
      store(y_ptr + static_cast<int64_t>(row) * 4 * kHiddenDim + i * kHiddenDim + col, bf);
      v = to<float>(bf);
#pragma unroll
      for (int j = 0; j < kElementsPerThread; ++j) acc_sqsum += v[j] * v[j];
      if constexpr (kUsePairFma) {
#pragma unroll
        for (int p = 0; p < 8; p += 2) {
          const auto* wp = w_ptr + static_cast<int64_t>(p) * 4 * kHiddenDim + i * kHiddenDim + col;
          const auto w0 = ihc_load_weight8<kWeightsAligned32>(wp);
          const auto w1 = ihc_load_weight8<kWeightsAligned32>(wp + 4 * kHiddenDim);
          auto acc = make_float2(acc_dot[p], acc_dot[p + 1]);
#pragma unroll
          for (int j = 0; j < kElementsPerThread; ++j)
            acc = ihc_fma2(make_float2(v[j], v[j]), make_float2(w0[j], w1[j]), acc);
          acc_dot[p] = acc.x;
          acc_dot[p + 1] = acc.y;
        }
      } else {
#pragma unroll
        for (int p = 0; p < 8; ++p) {
          vec_t<float, kElementsPerThread> ww;
          auto wp = w_ptr + static_cast<int64_t>(p) * 4 * kHiddenDim + i * kHiddenDim + col;
          ww = ihc_load_weight8<kWeightsAligned32>(wp);
#pragma unroll
          for (int j = 0; j < kElementsPerThread; ++j) acc_dot[p] += v[j] * ww[j];
        }
      }
    }
  }
#pragma unroll
  for (int p = 0; p < 8; ++p) {
    float v = warp_reduce_sum_xor(acc_dot[p]);
    if (lane == 0) smem_partial[p][warp] = v;
  }
  float v = warp_reduce_sum_xor(acc_sqsum);
  if (lane == 0) smem_partial[8][warp] = v;
  __syncthreads();
  if (tid < 4) {
    float pre = 0, post = 0, total = 0;
#pragma unroll
    for (int k = 0; k < kWarpsPerBlock; ++k) {
      pre += smem_partial[tid][k];
      post += smem_partial[tid + 4][k];
      total += smem_partial[8][k];
    }
    float r = rsqrtf_ftz(total * (1.f / (4 * kHiddenDim)) + norm_eps);
    smem_pre_gates[tid] = sigmoid(hc_scale_ptr[0] * (pre * r) + hc_base_ptr[tid]) + hc_eps;
    H_post_out_ptr[row * 4 + tid] =
        magnitude * sigmoid(hc_scale_ptr[1] * (post * r) + hc_base_ptr[tid + 4]) + hc_eps;
  }
  __syncthreads();
  using SavedOutput = std::conditional_t<kFuseRmsNorm && kCastBfloatForNorm, __nv_bfloat16, float>;
  vec_t<SavedOutput, kElementsPerThread> saved[kChunksPerRow];
  float output_sqsum = 0;
#pragma unroll
  for (int ch = 0; ch < kChunksPerRow; ++ch) {
    vec_t<float, kElementsPerThread> out;
    for (int j = 0; j < kElementsPerThread; ++j) out[j] = 0;
    auto* o2 = reinterpret_cast<float2*>(&out[0]);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      vec_t<__nv_bfloat16, kElementsPerThread> values;
      if constexpr (kRetainPostOutput) {
        values = post_output_cache[ch][i];
      } else {
        const int col = (ch * kThreadsPerBlock + tid) * kElementsPerThread;
        values = load<__nv_bfloat16, kElementsPerThread>(
            y_ptr + static_cast<int64_t>(row) * 4 * kHiddenDim + i * kHiddenDim + col);
      }
      auto in = to<float>(values);
      auto* i2 = reinterpret_cast<const float2*>(&in[0]);
#pragma unroll
      for (int j = 0; j < kElementsPerThread / 2; ++j)
        o2[j] = ihc_fma2(make_float2(smem_pre_gates[i], smem_pre_gates[i]), i2[j], o2[j]);
    }
    if constexpr (kFuseRmsNorm) {
      if constexpr (kCastBfloatForNorm) {
        saved[ch] = to<__nv_bfloat16>(out);
        out = to<float>(saved[ch]);
      } else {
        saved[ch] = out;
      }
#pragma unroll
      for (int j = 0; j < kElementsPerThread; ++j) output_sqsum += out[j] * out[j];
    } else {
      store(z_ptr + static_cast<int64_t>(row) * kHiddenDim +
                (ch * kThreadsPerBlock + tid) * kElementsPerThread,
            to<__nv_bfloat16>(out));
    }
  }
  if constexpr (kFuseRmsNorm) {
    float ss = warp_reduce_sum_xor(output_sqsum);
    if (lane == 0) smem_norm_partial[warp] = ss;
    __syncthreads();
    if (tid == 0) {
      float total = 0;
      for (int k = 0; k < kWarpsPerBlock; ++k) total += smem_norm_partial[k];
      smem_rms_inv = rsqrtf_ftz(total * (1.f / kHiddenDim) + rms_eps);
    }
    __syncthreads();
#pragma unroll
    for (int ch = 0; ch < kChunksPerRow; ++ch) {
      int col = (ch * kThreadsPerBlock + tid) * kElementsPerThread;
      auto g = to<float>(load<__nv_bfloat16, kElementsPerThread>(rms_weight_ptr + col));
      auto out = to<float>(saved[ch]);
      auto* o2 = reinterpret_cast<float2*>(&out[0]);
      auto* g2 = reinterpret_cast<const float2*>(&g[0]);
#pragma unroll
      for (int j = 0; j < kElementsPerThread / 2; ++j)
        o2[j] = ihc_mul2(ihc_mul2(o2[j], make_float2(smem_rms_inv, smem_rms_inv)), g2[j]);
      store(z_ptr + static_cast<int64_t>(row) * kHiddenDim + col, to<__nv_bfloat16>(out));
    }
  }
  if constexpr (kUsePDL) cudaTriggerProgrammaticLaunchCompletion();
}

template <int kHiddenDim, int kRowsPerBlock, int kThreadsPerBlock, int kEpilogueThreadsPerRow,
          bool kFuseRmsNorm, bool kCastBfloatForNorm, bool kUsePDL = false,
          bool kWeightsAligned32 = false, bool kSharedPost = false>
__device__ __forceinline__ void ihc_post_pre_rows_device(
    __nv_bfloat16* y_ptr, __nv_bfloat16* z_ptr, float* H_post_out_ptr, const __nv_bfloat16* xa_ptr,
    const __nv_bfloat16* residual_ptr, const float* H_post_in_ptr, const float* w_ptr,
    const float* hc_scale_ptr, const float* hc_base_ptr, int num_batch, float norm_eps,
    float hc_eps, float magnitude, const __nv_bfloat16* rms_weight_ptr, float rms_eps) {
  constexpr int kHCMult = 4;
  constexpr int kProjRows = 8;
  constexpr int kElementsPerThread = 8;
  constexpr int kWarpsPerBlock = kThreadsPerBlock / 32;
  constexpr int kChunksPerRow = kHiddenDim / (kThreadsPerBlock * kElementsPerThread);
  constexpr bool kSm90 = HPC_TARGET_ARCH == 90;
  constexpr bool kSingleRow = kSm90 && kRowsPerBlock == 1;
  static_assert(kHiddenDim % (kThreadsPerBlock * kElementsPerThread) == 0,
                "block span must divide hidden size");
  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }
  const int row_base = blockIdx.x * kRowsPerBlock;
  const int valid_rows = kSingleRow ? 1 : min(kRowsPerBlock, num_batch - row_base);
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  __shared__ float smem_partial[kRowsPerBlock][kProjRows + 1][kWarpsPerBlock];
  __shared__ __align__(16) float smem_pre_gates[kRowsPerBlock][kHCMult];
  __shared__ float norm_partial[kRowsPerBlock * (kEpilogueThreadsPerRow / 32)];
  extern __shared__ __align__(16) __nv_bfloat16 shared_post[];
  float acc_dot[kRowsPerBlock][kProjRows];
  float acc_sqsum[kRowsPerBlock];
#pragma unroll
  for (int r = 0; r < kRowsPerBlock; ++r) {
    acc_sqsum[r] = 0.f;
#pragma unroll
    for (int c = 0; c < kProjRows; ++c) {
      acc_dot[r][c] = 0.f;
    }
  }

#pragma unroll 1
  for (int ch = 0; ch < kChunksPerRow; ++ch) {
    const int col = (ch * kThreadsPerBlock + threadIdx.x) * kElementsPerThread;
    vec_t<__nv_bfloat16, kElementsPerThread> reg_xa[kRowsPerBlock];
#pragma unroll
    for (int r = 0; r < kRowsPerBlock; ++r) {
      if (r < valid_rows) {
        reg_xa[r] = load<__nv_bfloat16, kElementsPerThread>(
            xa_ptr + static_cast<int64_t>(row_base + r) * kHiddenDim + col);
      }
    }
#pragma unroll(kSingleRow ? kHCMult : 1)
    for (int i = 0; i < kHCMult; ++i) {
      const int flat = i * kHiddenDim + col;
      vec_t<float, kElementsPerThread> reg_y[kRowsPerBlock];
#pragma unroll
      for (int r = 0; r < kRowsPerBlock; ++r) {
        if (r < valid_rows) {
          const int64_t row = row_base + r;
          const auto a = to<float>(reg_xa[r]);
          const auto b = to<float>(load<__nv_bfloat16, kElementsPerThread>(
              residual_ptr + row * kHCMult * kHiddenDim + flat));
          const float h = H_post_in_ptr[row * kHCMult + i];
          const auto* a2 = reinterpret_cast<const float2*>(&a[0]);
          const auto* b2 = reinterpret_cast<const float2*>(&b[0]);
          auto* y2 = reinterpret_cast<float2*>(&reg_y[r][0]);
#pragma unroll
          for (int j = 0; j < kElementsPerThread / 2; ++j) {
            y2[j] = ihc_fma2(make_float2(h, h), a2[j], b2[j]);
          }
          const auto bf = to<__nv_bfloat16>(reg_y[r]);
          store(y_ptr + row * kHCMult * kHiddenDim + flat, bf);
          if constexpr (kSharedPost) {
            store(shared_post + r * kHCMult * kHiddenDim + flat, bf);
          }
          reg_y[r] = to<float>(bf);
#pragma unroll
          for (int j = 0; j < kElementsPerThread; ++j) {
            acc_sqsum[r] += reg_y[r][j] * reg_y[r][j];
          }
        }
      }
#pragma unroll
      for (int c = 0; c < kProjRows; c += 2) {
        vec_t<float, kElementsPerThread> w0, w1;
        const float* p0 = w_ptr + static_cast<int64_t>(c) * kHCMult * kHiddenDim + flat;
        const float* p1 = p0 + kHCMult * kHiddenDim;
        if constexpr (kHiddenDim != 4096) {
          w0 = ihc_load_weight8<kWeightsAligned32>(p0);
          w1 = ihc_load_weight8<kWeightsAligned32>(p1);
        } else {
          *reinterpret_cast<vec_t<float, kElementsPerThread / 2>*>(&w0[0]) =
              load<float, kElementsPerThread / 2>(p0);
          *reinterpret_cast<vec_t<float, kElementsPerThread / 2>*>(&w0[kElementsPerThread / 2]) =
              load<float, kElementsPerThread / 2>(p0 + kElementsPerThread / 2);
          *reinterpret_cast<vec_t<float, kElementsPerThread / 2>*>(&w1[0]) =
              load<float, kElementsPerThread / 2>(p1);
          *reinterpret_cast<vec_t<float, kElementsPerThread / 2>*>(&w1[kElementsPerThread / 2]) =
              load<float, kElementsPerThread / 2>(p1 + kElementsPerThread / 2);
        }
#pragma unroll
        for (int r = 0; r < kRowsPerBlock; ++r) {
          if (r < valid_rows) {
            ihc_dot_pair<kElementsPerThread>(acc_dot[r][c], acc_dot[r][c + 1], reg_y[r], w0, w1);
          }
        }
      }
    }
  }

#pragma unroll
  for (int r = 0; r < kRowsPerBlock; ++r) {
#pragma unroll
    for (int c = 0; c < kProjRows; ++c) {
      const float v = warp_reduce_sum_xor(acc_dot[r][c]);
      if (lane == 0) {
        smem_partial[r][c][warp] = v;
      }
    }
    const float v = warp_reduce_sum_xor(acc_sqsum[r]);
    if (lane == 0) {
      smem_partial[r][kProjRows][warp] = v;
    }
  }
  __syncthreads();
  if constexpr (!kSm90) {
    for (int idx = threadIdx.x; idx < kRowsPerBlock * (kProjRows + 1); idx += kThreadsPerBlock) {
      const int r = idx / (kProjRows + 1);
      const int c = idx % (kProjRows + 1);
      float total = 0.f;
#pragma unroll
      for (int k = 0; k < kWarpsPerBlock; ++k) {
        total += smem_partial[r][c][k];
      }
      smem_partial[r][c][0] = total;
    }
    __syncthreads();
  }
  for (int idx = threadIdx.x; idx < valid_rows * kHCMult; idx += kThreadsPerBlock) {
    const int r = idx / kHCMult;
    const int i = idx % kHCMult;
    float pre = 0.f, post = 0.f, sq = 0.f;
    if constexpr (kSm90) {
#pragma unroll
      for (int k = 0; k < kWarpsPerBlock; ++k) {
        pre += smem_partial[r][i][k];
        post += smem_partial[r][kHCMult + i][k];
        sq += smem_partial[r][kProjRows][k];
      }
    } else {
      pre = smem_partial[r][i][0];
      post = smem_partial[r][kHCMult + i][0];
      sq = smem_partial[r][kProjRows][0];
    }
    const float scale = rsqrtf_ftz(sq * (1.f / (kHCMult * kHiddenDim)) + norm_eps);
    smem_pre_gates[r][i] = sigmoid(hc_scale_ptr[0] * (pre * scale) + hc_base_ptr[i]) + hc_eps;
    H_post_out_ptr[(row_base + r) * kHCMult + i] =
        magnitude * sigmoid(hc_scale_ptr[1] * (post * scale) + hc_base_ptr[kHCMult + i]) + hc_eps;
  }
  __syncthreads();
  ihc_mix_epilogue<kHCMult, kHiddenDim, kElementsPerThread, kRowsPerBlock, kThreadsPerBlock,
                   kEpilogueThreadsPerRow, kFuseRmsNorm, kCastBfloatForNorm>(
      z_ptr, kSharedPost ? shared_post : y_ptr, &smem_pre_gates[0][0], row_base,
      kSharedPost ? 0 : row_base, valid_rows, rms_weight_ptr, rms_eps, norm_partial);
  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

template <int kHCMult, int kHiddenDim, int kElementsPerThread, int kRowsPerBlock,
          int kThreadsPerBlock, int kProjRows, int kActiveRows>
__device__ __forceinline__ void ihc_pre_accumulate(const __nv_bfloat16* x_ptr, const float* w_ptr,
                                                   int row_base,
                                                   float (&acc_dot)[kRowsPerBlock][kProjRows],
                                                   float (&acc_sqsum)[kRowsPerBlock],
                                                   int row_offset) {
  constexpr int kHCDim = kHCMult * kHiddenDim;
  constexpr int kSpan = kThreadsPerBlock * kElementsPerThread;
  constexpr int kSteps = kHCDim / kSpan;
  constexpr int kHalf = kElementsPerThread / 2;

  static_assert(kHCDim % kSpan == 0, "the block's span must divide hc_mult * hidden_dim");

  for (int step = 0; step < kSteps; ++step) {
    const int iflat = step * kSpan + threadIdx.x * kElementsPerThread;

    vec_t<float, kElementsPerThread> reg_w[kProjRows];
#pragma unroll
    for (int c = 0; c < kProjRows; ++c) {
      const float* w_row_ptr = w_ptr + static_cast<int64_t>(c) * kHCDim + iflat;
      *reinterpret_cast<vec_t<float, kHalf>*>(&reg_w[c][0]) = load<float, kHalf>(w_row_ptr);
      *reinterpret_cast<vec_t<float, kHalf>*>(&reg_w[c][kHalf]) =
          load<float, kHalf>(w_row_ptr + kHalf);
    }

#pragma unroll
    for (int rr = 0; rr < kActiveRows; ++rr) {
      const int r = row_offset + rr;
      auto reg_x = to<float>(load<__nv_bfloat16, kElementsPerThread>(
          x_ptr + static_cast<int64_t>(row_base + r) * kHCDim + iflat));
      const auto* x2 = reinterpret_cast<const float2*>(&reg_x[0]);
#pragma unroll
      for (int j = 0; j < kElementsPerThread / 2; ++j) {
        acc_sqsum[r] += x2[j].x * x2[j].x;
        acc_sqsum[r] += x2[j].y * x2[j].y;
      }
#pragma unroll
      for (int c = 0; c < kProjRows; ++c) {
        const auto* w2 = reinterpret_cast<const float2*>(&reg_w[c][0]);
#pragma unroll
        for (int j = 0; j < kElementsPerThread / 2; ++j) {
          acc_dot[r][c] += x2[j].x * w2[j].x;
          acc_dot[r][c] += x2[j].y * w2[j].y;
        }
      }
    }
  }
}

template <int kHCMult, int kHiddenDim, int kElementsPerThread, int kRowsPerBlock,
          int kThreadsPerBlock, bool kHasPost, bool kUsePDL = false, bool kFuseRmsNorm = false,
          bool kCastBfloatForNorm = false>
__global__ void __launch_bounds__(kThreadsPerBlock)
    fuse_ihc_pre_or_head_kernel(__nv_bfloat16* output_y_ptr, float* output_H_post_ptr,
                                const __nv_bfloat16* x_ptr, const float* w_ptr,
                                const float* hc_scale_ptr, const float* hc_base_ptr, int num_batch,
                                float norm_eps, float hc_eps, float magnitude,
                                const __nv_bfloat16* rms_weight_ptr = nullptr,
                                float rms_eps = 0.f) {
  constexpr int kProjRows = kHasPost ? 2 * kHCMult : kHCMult;
  constexpr int kHCDim = kHCMult * kHiddenDim;
  constexpr int kChunks = kHiddenDim / (kThreadsPerBlock * kElementsPerThread);
  constexpr int kWarpSize = 32;
  constexpr int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;
  constexpr float kInvHCDim = 1.0f / kHCDim;
  constexpr float kInvHiddenDim = 1.0f / kHiddenDim;

  static_assert(kHiddenDim % (kThreadsPerBlock * kElementsPerThread) == 0,
                "kThreadsPerBlock * kElementsPerThread must divide kHiddenDim");
  static_assert(kThreadsPerBlock % kWarpSize == 0, "block must be whole warps");

  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }

  const int row_base = blockIdx.x * kRowsPerBlock;
  const int num_valid_rows = min(kRowsPerBlock, num_batch - row_base);

  const int iwarp = threadIdx.x / kWarpSize;
  const int ilane = threadIdx.x % kWarpSize;

  __shared__ float smem_partial[kRowsPerBlock][kProjRows + 1][kWarpsPerBlock];
  __shared__ __align__(sizeof(float) * kHCMult) float smem_H_pre[kRowsPerBlock][kHCMult];

  float acc_dot[kRowsPerBlock][kProjRows];
  float acc_sqsum[kRowsPerBlock];
#pragma unroll
  for (int r = 0; r < kRowsPerBlock; ++r) {
    acc_sqsum[r] = 0.f;
#pragma unroll
    for (int c = 0; c < kProjRows; ++c) {
      acc_dot[r][c] = 0.f;
    }
  }

  if (num_valid_rows == kRowsPerBlock) {
    ihc_pre_accumulate<kHCMult, kHiddenDim, kElementsPerThread, kRowsPerBlock, kThreadsPerBlock,
                       kProjRows, kRowsPerBlock>(x_ptr, w_ptr, row_base, acc_dot, acc_sqsum, 0);
  } else {
    for (int r = 0; r < num_valid_rows; ++r) {
      ihc_pre_accumulate<kHCMult, kHiddenDim, kElementsPerThread, kRowsPerBlock, kThreadsPerBlock,
                         kProjRows, 1>(x_ptr, w_ptr, row_base, acc_dot, acc_sqsum, r);
    }
  }

#pragma unroll
  for (int r = 0; r < kRowsPerBlock; ++r) {
#pragma unroll
    for (int c = 0; c < kProjRows; ++c) {
      float v = warp_reduce_sum_xor(acc_dot[r][c]);
      if (ilane == 0) {
        smem_partial[r][c][iwarp] = v;
      }
    }
    float v = warp_reduce_sum_xor(acc_sqsum[r]);
    if (ilane == 0) {
      smem_partial[r][kProjRows][iwarp] = v;
    }
  }
  __syncthreads();

  const cutlass::FastDivmod proj_divmod(kProjRows + 1);
  for (int idx = threadIdx.x; idx < kRowsPerBlock * (kProjRows + 1); idx += kThreadsPerBlock) {
    int r, c;
    proj_divmod(r, c, idx);  // r = idx / (kProjRows+1), c = idx % (kProjRows+1)
    float sum = 0.f;
#pragma unroll
    for (int w = 0; w < kWarpsPerBlock; ++w) {
      sum += smem_partial[r][c][w];
    }
    smem_partial[r][c][0] = sum;
  }
  __syncthreads();

  const cutlass::FastDivmod hc_divmod(kHCMult);
  for (int idx = threadIdx.x; idx < num_valid_rows * kHCMult; idx += kThreadsPerBlock) {
    int r, i;
    hc_divmod(r, i, idx);  // r = idx / kHCMult, i = idx % kHCMult
    const float rms_scale = rsqrtf_ftz(smem_partial[r][kProjRows][0] * kInvHCDim + norm_eps);

    const float mixes_pre = smem_partial[r][i][0] * rms_scale;
    smem_H_pre[r][i] = sigmoid(hc_scale_ptr[0] * mixes_pre + hc_base_ptr[i]) + hc_eps;

    if constexpr (kHasPost) {
      const float mixes_post = smem_partial[r][kHCMult + i][0] * rms_scale;
      output_H_post_ptr[(row_base + r) * kHCMult + i] =
          magnitude * sigmoid(hc_scale_ptr[1] * mixes_post + hc_base_ptr[kHCMult + i]) + hc_eps;
    }
  }
  __syncthreads();

  for (int r = 0; r < num_valid_rows; ++r) {
    vec_t<float, kHCMult> reg_H_pre = load<float, kHCMult>(&smem_H_pre[r][0]);
    const __nv_bfloat16* x_row_ptr = x_ptr + (row_base + r) * kHCDim;
    __nv_bfloat16* y_row_ptr = output_y_ptr + (row_base + r) * kHiddenDim;

    vec_t<float, kElementsPerThread> reg_y[kChunks];
    float y_sqsum = 0.f;

#pragma unroll
    for (int ch = 0; ch < kChunks; ++ch) {
      const int icol = (ch * kThreadsPerBlock + threadIdx.x) * kElementsPerThread;

#pragma unroll
      for (int j = 0; j < kElementsPerThread; ++j) {
        reg_y[ch][j] = 0.f;
      }
      auto* y2 = reinterpret_cast<float2*>(&reg_y[ch][0]);

#pragma unroll
      for (int i = 0; i < kHCMult; ++i) {
        auto reg_x =
            to<float>(load<__nv_bfloat16, kElementsPerThread>(x_row_ptr + i * kHiddenDim + icol));
        const float h = reg_H_pre[i];
        const auto* x2 = reinterpret_cast<const float2*>(&reg_x[0]);
#pragma unroll
        for (int j = 0; j < kElementsPerThread / 2; ++j) {
          y2[j].x += h * x2[j].x;
          y2[j].y += h * x2[j].y;
        }
      }

      if constexpr (kFuseRmsNorm) {
#pragma unroll
        for (int j = 0; j < kElementsPerThread / 2; ++j) {
          if constexpr (kCastBfloatForNorm) {
            y2[j].x = __bfloat162float(__float2bfloat16(y2[j].x));
            y2[j].y = __bfloat162float(__float2bfloat16(y2[j].y));
          }
          y_sqsum += y2[j].x * y2[j].x;
          y_sqsum += y2[j].y * y2[j].y;
        }
      } else {
        store(y_row_ptr + icol, to<__nv_bfloat16>(reg_y[ch]));
      }
    }

    if constexpr (kFuseRmsNorm) {
      float v = warp_reduce_sum_xor(y_sqsum);
      if (ilane == 0) {
        smem_partial[r][0][iwarp] = v;
      }
      __syncthreads();
      float total = 0.f;
#pragma unroll
      for (int w = 0; w < kWarpsPerBlock; ++w) {
        total += smem_partial[r][0][w];
      }
      const float rms_scale = rsqrtf_ftz(total * kInvHiddenDim + rms_eps);

#pragma unroll
      for (int ch = 0; ch < kChunks; ++ch) {
        const int icol = (ch * kThreadsPerBlock + threadIdx.x) * kElementsPerThread;
        auto reg_g = to<float>(load<__nv_bfloat16, kElementsPerThread>(rms_weight_ptr + icol));
        auto* y2 = reinterpret_cast<float2*>(&reg_y[ch][0]);
        const auto* g2 = reinterpret_cast<const float2*>(&reg_g[0]);
#pragma unroll
        for (int j = 0; j < kElementsPerThread / 2; ++j) {
          y2[j].x = y2[j].x * rms_scale * g2[j].x;
          y2[j].y = y2[j].y * rms_scale * g2[j].y;
        }
        store(y_row_ptr + icol, to<__nv_bfloat16>(reg_y[ch]));
      }
      __syncthreads();
    }
  }

  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

template <int kHCMult, int kHiddenDim, int kElementsPerThread, int kThreadsPerBlock,
          bool kUsePDL = false>
__global__ void __launch_bounds__(kThreadsPerBlock)
    fuse_ihc_post_kernel(__nv_bfloat16* output_ptr, const __nv_bfloat16* x_ptr,
                         const __nv_bfloat16* residual_ptr, const float* H_post_ptr) {
  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }
  const int irow = blockIdx.x;
  auto reg_H_post = load<float, kHCMult>(H_post_ptr + irow * kHCMult);

  const int icol = (blockIdx.y * blockDim.x + threadIdx.x) * kElementsPerThread;
  // The launch tiles the hidden dimension exactly; no column bounds check is needed.
  const __nv_bfloat16* x_row_ptr = x_ptr + irow * kHiddenDim + icol;
  const __nv_bfloat16* residual_row_ptr = residual_ptr + irow * kHCMult * kHiddenDim + icol;
  __nv_bfloat16* output_row_ptr = output_ptr + irow * kHCMult * kHiddenDim + icol;

  auto reg_x = to<float>(load<__nv_bfloat16, kElementsPerThread>(x_row_ptr));

#pragma unroll
  for (int i = 0; i < kHCMult; ++i) {
    auto reg_residual =
        to<float>(load<__nv_bfloat16, kElementsPerThread>(residual_row_ptr + i * kHiddenDim));

    vec_t<float, kElementsPerThread> reg_output;
    auto* o2 = reinterpret_cast<float2*>(&reg_output[0]);
    const auto* x2 = reinterpret_cast<const float2*>(&reg_x[0]);
    const auto* r2 = reinterpret_cast<const float2*>(&reg_residual[0]);
    const float h = reg_H_post[i];
#pragma unroll
    for (int j = 0; j < kElementsPerThread / 2; ++j) {
      o2[j].x = h * x2[j].x + r2[j].x;
      o2[j].y = h * x2[j].y + r2[j].y;
    }

    store(output_row_ptr + i * kHiddenDim, to<__nv_bfloat16>(reg_output));
  }

  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

#if HPC_TARGET_ARCH == 90
enum class PostPrePath { Rows, Retained, Cluster, ReloadPostOutput, SharedPostOutput };

template <int kHiddenDim, int kThreadsPerBlock, PostPrePath kPath, int kRowsPerBlock,
          int kClusterBlocks, int kMinBlocksPerSm, bool kUsePDL, bool kFuseRmsNorm,
          bool kCastBfloatForNorm, int kEpilogueThreadsPerRow = kThreadsPerBlock>
__global__ void __launch_bounds__(kThreadsPerBlock, kMinBlocksPerSm)
    fuse_ihc_post_pre_sm90_kernel(__nv_bfloat16* y_ptr, __nv_bfloat16* z_ptr, float* H_post_out_ptr,
                                  const __nv_bfloat16* xa_ptr, const __nv_bfloat16* residual_ptr,
                                  const float* H_post_in_ptr, const float* w_ptr,
                                  const float* hc_scale_ptr, const float* hc_base_ptr,
                                  int num_batch, float norm_eps, float hc_eps, float magnitude,
                                  const __nv_bfloat16* rms_weight_ptr, float rms_eps) {
  if constexpr (kPath == PostPrePath::Cluster) {
    ihc_post_pre_cluster_device<kHiddenDim, kClusterBlocks, kFuseRmsNorm, kCastBfloatForNorm,
                                kUsePDL>(y_ptr, z_ptr, H_post_out_ptr, xa_ptr, residual_ptr,
                                         H_post_in_ptr, w_ptr, hc_scale_ptr, hc_base_ptr, num_batch,
                                         norm_eps, hc_eps, magnitude, rms_weight_ptr, rms_eps);
  } else if constexpr (kPath == PostPrePath::Retained || kPath == PostPrePath::ReloadPostOutput) {
    constexpr bool kRetainPostOutput = kPath == PostPrePath::Retained;
    ihc_post_pre_single_row_device<kHiddenDim, kThreadsPerBlock, kFuseRmsNorm, kCastBfloatForNorm,
                                   kUsePDL, false, false, kRetainPostOutput>(
        y_ptr, z_ptr, H_post_out_ptr, xa_ptr, residual_ptr, H_post_in_ptr, w_ptr, hc_scale_ptr,
        hc_base_ptr, num_batch, norm_eps, hc_eps, magnitude, rms_weight_ptr, rms_eps);
  } else {
    constexpr bool kSharedPost = kPath == PostPrePath::SharedPostOutput;
    ihc_post_pre_rows_device<kHiddenDim, kRowsPerBlock, kThreadsPerBlock, kEpilogueThreadsPerRow,
                             kFuseRmsNorm, kCastBfloatForNorm, kUsePDL, false, kSharedPost>(
        y_ptr, z_ptr, H_post_out_ptr, xa_ptr, residual_ptr, H_post_in_ptr, w_ptr, hc_scale_ptr,
        hc_base_ptr, num_batch, norm_eps, hc_eps, magnitude, rms_weight_ptr, rms_eps);
  }
}
#else
template <int kHCMult, int kHiddenDim, int kElementsPerThread, int kThreadsPerBlock,
          int kRowsPerBlock, bool kUsePDL = false, bool kFuseRmsNorm = false,
          bool kCastBfloatForNorm = false, bool kWeightsAligned32 = true>
__global__ void __launch_bounds__(kThreadsPerBlock)
    fuse_ihc_post_pre_kernel(__nv_bfloat16* y_ptr, __nv_bfloat16* z_ptr, float* H_post_out_ptr,
                             const __nv_bfloat16* xa_ptr, const __nv_bfloat16* residual_ptr,
                             const float* H_post_in_ptr, const float* w_ptr,
                             const float* hc_scale_ptr, const float* hc_base_ptr, int num_batch,
                             float norm_eps, float hc_eps, float magnitude,
                             const __nv_bfloat16* rms_weight_ptr = nullptr, float rms_eps = 0.f) {
  static_assert(kHCMult == 4 && kElementsPerThread == 8, "supported fused layout");
  static_assert(kHiddenDim == 3072 || kHiddenDim == 4096 || kHiddenDim == 6144,
                "supported hidden dimensions");
  ihc_post_pre_rows_device<kHiddenDim, kRowsPerBlock, kThreadsPerBlock, kThreadsPerBlock,
                           kFuseRmsNorm, kCastBfloatForNorm, kUsePDL, kWeightsAligned32>(
      y_ptr, z_ptr, H_post_out_ptr, xa_ptr, residual_ptr, H_post_in_ptr, w_ptr, hc_scale_ptr,
      hc_base_ptr, num_batch, norm_eps, hc_eps, magnitude, rms_weight_ptr, rms_eps);
}
#endif

template <int kHCMult, int kHiddenDim, int kElementsPerThread, int kBlocksPerCluster,
          int kThreadsPerBlock, bool kUsePDL = false, bool kFuseRmsNorm = false,
          bool kCastBfloatForNorm = false>
__global__ void __launch_bounds__(kThreadsPerBlock)
    fuse_ihc_post_pre_cluster_kernel(__nv_bfloat16* y_ptr, __nv_bfloat16* z_ptr,
                                     float* H_post_out_ptr, const __nv_bfloat16* xa_ptr,
                                     const __nv_bfloat16* residual_ptr, const float* H_post_in_ptr,
                                     const float* w_ptr, const float* hc_scale_ptr,
                                     const float* hc_base_ptr, int num_batch, float norm_eps,
                                     float hc_eps, float magnitude,
                                     const __nv_bfloat16* rms_weight_ptr = nullptr,
                                     float rms_eps = 0.f) {
  static_assert(kHCMult == 4 && kElementsPerThread == 8, "supported fused layout");
  static_assert(kThreadsPerBlock * kElementsPerThread * kBlocksPerCluster == kHiddenDim,
                "cluster blocks must cover the hidden dimension");
  ihc_post_pre_cluster_device<kHiddenDim, kBlocksPerCluster, kFuseRmsNorm, kCastBfloatForNorm,
                              kUsePDL>(y_ptr, z_ptr, H_post_out_ptr, xa_ptr, residual_ptr,
                                       H_post_in_ptr, w_ptr, hc_scale_ptr, hc_base_ptr, num_batch,
                                       norm_eps, hc_eps, magnitude, rms_weight_ptr, rms_eps);
}

template <int kHCMult, int kHiddenDim, int kElementsPerThread, int kThreadsPerBlock,
          bool kUsePDL = false, bool kFuseRmsNorm = false, bool kCastBfloatForNorm = false,
          bool kWeightsAligned32 = true>
__global__ void __launch_bounds__(kThreadsPerBlock)
    fuse_ihc_post_pre_retained_kernel(__nv_bfloat16* y_ptr, __nv_bfloat16* z_ptr,
                                      float* H_post_out_ptr, const __nv_bfloat16* xa_ptr,
                                      const __nv_bfloat16* residual_ptr, const float* H_post_in_ptr,
                                      const float* w_ptr, const float* hc_scale_ptr,
                                      const float* hc_base_ptr, int num_batch, float norm_eps,
                                      float hc_eps, float magnitude,
                                      const __nv_bfloat16* rms_weight_ptr = nullptr,
                                      float rms_eps = 0.f) {
  static_assert(kHCMult == 4 && kElementsPerThread == 8, "supported fused layout");
  ihc_post_pre_single_row_device<kHiddenDim, kThreadsPerBlock, kFuseRmsNorm, kCastBfloatForNorm,
                                 kUsePDL, kWeightsAligned32>(
      y_ptr, z_ptr, H_post_out_ptr, xa_ptr, residual_ptr, H_post_in_ptr, w_ptr, hc_scale_ptr,
      hc_base_ptr, num_batch, norm_eps, hc_eps, magnitude, rms_weight_ptr, rms_eps);
}

#if HPC_TARGET_ARCH != 90
__device__ __forceinline__ void ihc_tmem_fence_before_sync() {
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void ihc_tmem_fence_after_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

template <int kBlockK, int kParts>
struct IhcPostPreUmmaStorage {
  using SmemLayoutA =
      decltype(cute::tile_to_shape(cute::UMMA::Layout_K_SW128_Atom<cute::bfloat16_t>{},
                                   cute::Shape<cute::_64, cute::Int<kBlockK>>{}));
  using SmemLayoutB =
      decltype(cute::tile_to_shape(cute::UMMA::Layout_K_SW128_Atom<cute::bfloat16_t>{},
                                   cute::Shape<cute::Int<8 * kParts>, cute::Int<kBlockK>>{}));
  __align__(1024) __nv_bfloat16 a[2][cute::cosize_v<SmemLayoutA>];
  __align__(1024) __nv_bfloat16 b[2][cute::cosize_v<SmemLayoutB>];
  __align__(8) uint64_t barrier[2];
  uint32_t tmem;
  float sq[64];
  float projection[64][8];
  float normsum[64];
  __align__(16) float gate[64][4];
};

// M64 accumulators use half the TMEM datapaths. Interleave independent
// accumulators across both halves to shorten the FP32 summation chains.
template <int kHCMult, int kHiddenDim, int kElementsPerThread, int kThreadsPerBlock, bool kUsePDL,
          bool kFuseRmsNorm, bool kCastBfloatForNorm, bool kWeightsAligned32,
          int kAccumulatorGroups>
__global__ void fuse_ihc_post_pre_umma_kernel(
    __nv_bfloat16* y_ptr, __nv_bfloat16* z_ptr, float* H_post_out_ptr, const __nv_bfloat16* xa_ptr,
    const __nv_bfloat16* residual_ptr, const float* H_post_in_ptr, const float* w_ptr,
    const float* hc_scale_ptr, const float* hc_base_ptr, int num_batch, float norm_eps,
    float hc_eps, float magnitude, const __nv_bfloat16* rms_weight_ptr, float rms_eps) {
  static_assert((kHiddenDim == 3072 || kHiddenDim == 4096 || kHiddenDim == 6144) && kHCMult == 4 &&
                    kElementsPerThread == 8 && kThreadsPerBlock == 256 && kWeightsAligned32,
                "supported UMMA layout");
  static_assert(!kFuseRmsNorm || kCastBfloatForNorm, "RMSNorm requires BF16 intermediate");
  static_assert(kAccumulatorGroups == 8, "supported TMEM layout");
  constexpr int kBlockK = 128, kParts = 3;
  constexpr int kBlocks = ihc_post_pre_num_blocks(kHiddenDim);
  constexpr int kVecsPerTile = kBlockK / kElementsPerThread;
  constexpr int kRowsPerThread = 64 * kVecsPerTile / kThreadsPerBlock;
  constexpr int kTmemColumns = 128;
  using Storage = IhcPostPreUmmaStorage<kBlockK, kParts>;
  using SmemLayoutA = typename Storage::SmemLayoutA;
  using SmemLayoutB = typename Storage::SmemLayoutB;
  extern __shared__ __align__(1024) unsigned char storage[];
  auto& smem = *reinterpret_cast<Storage*>(storage);
  auto cluster = cooperative_groups::this_cluster();
  const int rank = cluster.block_rank();
  constexpr int kSlice = kHiddenDim / kBlocks;
  static_assert(kSlice % kBlockK == 0 && kSlice % (32 * kElementsPerThread) == 0,
                "slice must hold whole K tiles");
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, base = (blockIdx.x / kBlocks) * 64;
  const int valid = min(64, num_batch - base);
  if constexpr (kUsePDL) cudaGridDependencySynchronize();
  if (tid < 32) {
    cute::TMEM::Allocator1Sm alloc;
    alloc.allocate(kTmemColumns, &smem.tmem);
    alloc.release_allocation_lock();
    ihc_tmem_fence_before_sync();
  }
  // Only the MMA warp signals each slot's barrier.
  if (tid == 0) {
    cutlass::arch::ClusterBarrier::init(&smem.barrier[0], 1);
    cutlass::arch::ClusterBarrier::init(&smem.barrier[1], 1);
  }
  float sq[kRowsPerThread] = {};
  __syncthreads();
  ihc_tmem_fence_after_sync();
  using TiledMma = decltype(cute::make_tiled_mma(
      cute::SM100_MMA_F16BF16_SS<cute::bfloat16_t, cute::bfloat16_t, float, 64, 8 * kParts,
                                 cute::UMMA::Major::K, cute::UMMA::Major::K>{}));
  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_slice(0);
  const auto tAccr_layout = thr_mma
                                .partition_fragment_C(cute::make_tensor(
                                    cute::make_gmem_ptr(static_cast<float*>(nullptr)),
                                    cute::make_shape(cute::Int<64>{}, cute::Int<8 * kParts>{}),
                                    cute::make_stride(cute::Int<8 * kParts>{}, cute::Int<1>{})))
                                .layout();
  auto sA0 = cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<cute::bfloat16_t*>(smem.a[0])),
                               SmemLayoutA{});
  auto sA1 = cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<cute::bfloat16_t*>(smem.a[1])),
                               SmemLayoutA{});
  auto sB0 = cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<cute::bfloat16_t*>(smem.b[0])),
                               SmemLayoutB{});
  auto sB1 = cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<cute::bfloat16_t*>(smem.b[1])),
                               SmemLayoutB{});
  constexpr uint32_t kTmemAccWidth = 8 * kParts;
  constexpr uint32_t kTmemPartitionStride = 1u << 20;  // high/low 32K partition selector
  int phase[2] = {0, 0};
  constexpr int kTilesPerChannel = kSlice / kBlockK;
  // Reuse each XA tile across channels for hidden size 6144.
#pragma unroll 1
  for (int outer = 0; outer < (kHiddenDim == 6144 ? kTilesPerChannel : 4); ++outer) {
#pragma unroll 1
    for (int inner = 0; inner < (kHiddenDim == 6144 ? 4 : kTilesPerChannel); ++inner) {
      const int ch = kHiddenDim == 6144 ? inner : outer;
      const int k0 = (kHiddenDim == 6144 ? outer : inner) * kBlockK;
      const int tile = kHiddenDim == 6144 ? outer * 4 + inner : outer * kTilesPerChannel + inner;
      const int slot = tile & 1;
      if (tile >= 2) {
        // Wait for the previous MMA to finish reading this slot before refilling it.
        if (tid < 32) cutlass::arch::ClusterBarrier::wait(&smem.barrier[slot], phase[slot]);
        phase[slot] ^= 1;
        __syncthreads();
      }
      // Select the slot outside the element loops to keep swizzled addresses static.
      auto fill_smem = [&](auto& sA, auto& sB) {
#pragma unroll
        for (int j = 0; j < kRowsPerThread; ++j) {
          const int idx = j * kThreadsPerBlock + tid, row = idx / kVecsPerTile,
                    k = (idx % kVecsPerTile) * kElementsPerThread, col = rank * kSlice + k0 + k;
          vec_t<__nv_bfloat16, kElementsPerThread> bf{};
          if (row < valid) {
            const auto a = to<float>(load<__nv_bfloat16, kElementsPerThread>(
                xa_ptr + int64_t(base + row) * kHiddenDim + col));
            const auto b = to<float>(load<__nv_bfloat16, kElementsPerThread>(
                residual_ptr + int64_t(base + row) * 4 * kHiddenDim + ch * kHiddenDim + col));
            const float h = H_post_in_ptr[(base + row) * 4 + ch];
            vec_t<float, kElementsPerThread> v;
            auto* v2 = reinterpret_cast<float2*>(&v[0]);
            const auto* a2 = reinterpret_cast<const float2*>(&a[0]);
            const auto* b2 = reinterpret_cast<const float2*>(&b[0]);
#pragma unroll
            for (int q = 0; q < kElementsPerThread / 2; ++q)
              v2[q] = ihc_fma2(make_float2(h, h), a2[q], b2[q]);
            bf = to<__nv_bfloat16>(v);
            store(y_ptr + int64_t(base + row) * 4 * kHiddenDim + ch * kHiddenDim + col, bf);
            v = to<float>(bf);
#pragma unroll
            for (int q = 0; q < kElementsPerThread; ++q) sq[j] += v[q] * v[q];
          }
          store(reinterpret_cast<__nv_bfloat16*>(&sA(row, k)), bf);
        }
        // Split FP32 weights into BF16 residual terms for UMMA.
        for (int idx = tid; idx < 8 * kVecsPerTile; idx += kThreadsPerBlock) {
          const int p = idx / kVecsPerTile, k = (idx % kVecsPerTile) * kElementsPerThread;
          auto ww = ihc_load_weight8<true>(w_ptr + int64_t(p) * 4 * kHiddenDim + ch * kHiddenDim +
                                           rank * kSlice + k0 + k);
#pragma unroll
          for (int part = 0; part < kParts; ++part) {
            const auto bf = to<__nv_bfloat16>(ww);
            store(reinterpret_cast<__nv_bfloat16*>(&sB(part * 8 + p, k)), bf);
            const auto rounded = to<float>(bf);
#pragma unroll
            for (int q = 0; q < kElementsPerThread; ++q) ww[q] -= rounded[q];
          }
        }
      };
      if (slot == 0) {
        fill_smem(sA0, sB0);
      } else {
        fill_smem(sA1, sB1);
      }
      cutlass::arch::fence_view_async_shared();
      __syncthreads();
      if (tid < 32) {
        auto tAs = thr_mma.make_fragment_A(thr_mma.partition_A(slot == 0 ? sA0 : sA1));
        auto tBs = thr_mma.make_fragment_B(thr_mma.partition_B(slot == 0 ? sB0 : sB1));
        auto tAccr = cute::make_tensor(
            cute::make_tmem_ptr<float>(smem.tmem +
                                       ((tile % kAccumulatorGroups) / 2) * kTmemAccWidth +
                                       ((tile % kAccumulatorGroups) & 1) * kTmemPartitionStride),
            tAccr_layout);
        // The first tile in each group initializes its TMEM accumulator.
        tiled_mma.accumulate_ =
            tile < kAccumulatorGroups ? cute::UMMA::ScaleOut::Zero : cute::UMMA::ScaleOut::One;
#pragma unroll
        for (int ik = 0; ik < cute::size<2>(tAs); ++ik) {
          cute::gemm(tiled_mma, tAs(cute::_, cute::_, ik), tBs(cute::_, cute::_, ik), tAccr);
          tiled_mma.accumulate_ = cute::UMMA::ScaleOut::One;
        }
        cutlass::arch::umma_arrive(&smem.barrier[slot]);
      }
      __syncthreads();
    }
  }
  if (tid < 32) {
    for (int slot = 0; slot < 2; ++slot)
      cutlass::arch::ClusterBarrier::wait(&smem.barrier[slot], phase[slot]);
    ihc_tmem_fence_before_sync();
  }
  __syncthreads();
  ihc_tmem_fence_after_sync();
#pragma unroll
  for (int j = 0; j < kRowsPerThread; ++j) {
    constexpr int Width = kVecsPerTile < 32 ? kVecsPerTile : 32;
    float total = sq[j];
#pragma unroll
    for (int mask = Width / 2; mask > 0; mask /= 2)
      total += __shfl_xor_sync(0xffffffff, total, mask);
    if (tid % kVecsPerTile == 0) smem.sq[(j * kThreadsPerBlock + tid) / kVecsPerTile] = total;
    if constexpr (kVecsPerTile > 32) {
      if (tid % kVecsPerTile == 32)
        smem.gate[(j * kThreadsPerBlock + tid) / kVecsPerTile][0] = total;
    }
  }
  __syncthreads();
  if constexpr (kVecsPerTile > 32) {
    if (tid < 64) smem.sq[tid] += smem.gate[tid][0];
    __syncthreads();
  }
  if (tid < 128) {
    // Each warp loads one 32-datapath TMEM subpartition.
    auto tAccAll = cute::make_tensor(
        cute::make_tmem_ptr<float>(smem.tmem),
        cute::make_layout(
            cute::make_shape(cute::Int<128>{}, cute::Int<kAccumulatorGroups / 2 * 8 * kParts>{}),
            cute::make_stride(cute::TMEM::DP<float>{}, cute::_1{})));
    auto tiled_t2r = cute::make_tmem_copy(cute::SM100_TMEM_LOAD_32dp32b8x{}, tAccAll);
    auto thr_t2r = tiled_t2r.get_slice(tid);
    auto tSrc = thr_t2r.partition_S(tAccAll);
    auto tDst = cute::make_fragment_like<float>(thr_t2r.partition_D(tAccAll));
    cute::copy(tiled_t2r, tSrc, tDst);
    cutlass::arch::fence_view_async_tmem_load();
    // TMEM columns repeat the eight projection channels for each part and group.
    float projection[8] = {};
#pragma unroll
    for (int n = 0; n < cute::size<2>(tDst); ++n) {
#pragma unroll
      for (int v = 0; v < 8; ++v) {
        projection[v] += tDst(v, 0, n);
      }
    }
#pragma unroll
    for (int p = 0; p < 8; ++p) projection[p] += __shfl_xor_sync(0xffffffff, projection[p], 16);
    const int row = warp * 16 + lane;
    if (lane < 16) {
#pragma unroll
      for (int p = 0; p < 8; ++p) smem.projection[row][p] = projection[p];
    }
    ihc_tmem_fence_before_sync();
  }
  __syncthreads();
  if (tid < 32) {
    ihc_tmem_fence_after_sync();
    cute::TMEM::Allocator1Sm alloc;
    alloc.free(smem.tmem, kTmemColumns);
  }
  cluster.sync();
  if (rank == 0 && tid < valid) {
    float projection[8] = {}, total = 0;
#pragma unroll
    for (int b = 0; b < kBlocks; ++b) {
      const auto* remote = cluster.map_shared_rank(&smem, b);
      total += remote->sq[tid];
#pragma unroll
      for (int p = 0; p < 8; ++p) projection[p] += remote->projection[tid][p];
    }
    const float inv = rsqrtf_ftz(total * (1.f / (4 * kHiddenDim)) + norm_eps);
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      smem.gate[tid][p] =
          sigmoid(hc_scale_ptr[0] * (projection[p] * inv) + hc_base_ptr[p]) + hc_eps;
      H_post_out_ptr[(base + tid) * 4 + p] =
          magnitude * sigmoid(hc_scale_ptr[1] * (projection[p + 4] * inv) + hc_base_ptr[p + 4]) +
          hc_eps;
    }
  }
  cluster.sync();
  const auto* root = cluster.map_shared_rank(&smem, 0);
#pragma unroll 1
  for (int row = warp; row < valid; row += 8) {
    constexpr int Chunks = kSlice / (32 * kElementsPerThread);
    const auto gates = load<float, 4>(&root->gate[row][0]);
    float total = 0;
#pragma unroll
    for (int chunk = 0; chunk < Chunks; ++chunk) {
      const int col = rank * kSlice + (chunk * 32 + lane) * kElementsPerThread;
      vec_t<float, kElementsPerThread> v{};
      auto* v2 = reinterpret_cast<float2*>(&v[0]);
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        const auto in = to<float>(load<__nv_bfloat16, kElementsPerThread>(
            y_ptr + int64_t(base + row) * 4 * kHiddenDim + p * kHiddenDim + col));
        const auto* in2 = reinterpret_cast<const float2*>(&in[0]);
#pragma unroll
        for (int q = 0; q < kElementsPerThread / 2; ++q)
          v2[q] = ihc_fma2(make_float2(gates[p], gates[p]), in2[q], v2[q]);
      }
      const auto bf = to<__nv_bfloat162>(v);
      store(z_ptr + int64_t(base + row) * kHiddenDim + col, bf);
      if constexpr (kFuseRmsNorm) {
        v = to<float>(bf);
#pragma unroll
        for (int q = 0; q < kElementsPerThread; ++q) total += v[q] * v[q];
      }
    }
    if constexpr (kFuseRmsNorm) {
      total = warp_reduce_sum_xor(total);
      if (lane == 0) smem.normsum[row] = total;
    }
  }
  if constexpr (kFuseRmsNorm) {
    cluster.sync();
    if (rank == 0 && tid < valid) {
      float total = 0;
#pragma unroll
      for (int b = 0; b < kBlocks; ++b) total += cluster.map_shared_rank(&smem, b)->normsum[tid];
      smem.normsum[tid] = rsqrtf_ftz(total * (1.f / kHiddenDim) + rms_eps);
    }
    cluster.sync();
#pragma unroll 1
    for (int row = warp; row < valid; row += 8) {
      const float inv = root->normsum[row];
#pragma unroll
      for (int chunk = 0; chunk < kSlice / (32 * kElementsPerThread); ++chunk) {
        const int col = rank * kSlice + (chunk * 32 + lane) * kElementsPerThread;
        auto v = to<float>(load<__nv_bfloat16, kElementsPerThread>(
            z_ptr + int64_t(base + row) * kHiddenDim + col));
        const auto g = to<float>(load<__nv_bfloat16, kElementsPerThread>(rms_weight_ptr + col));
        auto* v2 = reinterpret_cast<float2*>(&v[0]);
        const auto* g2 = reinterpret_cast<const float2*>(&g[0]);
#pragma unroll
        for (int q = 0; q < kElementsPerThread / 2; ++q)
          v2[q] = ihc_mul2(ihc_mul2(v2[q], make_float2(inv, inv)), g2[q]);
        store(z_ptr + int64_t(base + row) * kHiddenDim + col, to<__nv_bfloat16>(v));
      }
    }
  }
  cluster.sync();

  if constexpr (kUsePDL) cudaTriggerProgrammaticLaunchCompletion();
}

// Per chunk, WG0 publishes BF16 post output before WG1/WG2 read it.
// Chunks use disjoint shared-memory offsets and stay live for the final mix.
__device__ __forceinline__ void ihc_three_wg_barrier() {
  constexpr int kBarrierId = 1;
  constexpr int kParticipants = 384;
  asm volatile("barrier.sync %0, %1;" ::"n"(kBarrierId), "n"(kParticipants) : "memory");
}

template <int kHiddenDim>
struct IhcPostPreThreeWarpGroupStorage {
  __align__(16) __nv_bfloat16 post_output[1][4][kHiddenDim];
  float partial[1][9][4];
  float pre_gates[1][4];
  float norm_partial[1][4];
};

template <int kHCMult, int kHiddenDim, int kElementsPerThread, int kThreadsPerBlock, bool kUsePDL,
          bool kFuseRmsNorm, bool kCastBfloatForNorm, bool kWeightsAligned32>
__global__ void __launch_bounds__(kThreadsPerBlock, kHiddenDim == 6144 ? 1 : 2)
    fuse_ihc_post_pre_three_wg_kernel(__nv_bfloat16* y_ptr, __nv_bfloat16* z_ptr,
                                      float* H_post_out_ptr, const __nv_bfloat16* xa_ptr,
                                      const __nv_bfloat16* residual_ptr, const float* H_post_in_ptr,
                                      const float* w_ptr, const float* hc_scale_ptr,
                                      const float* hc_base_ptr, int num_batch, float norm_eps,
                                      float hc_eps, float magnitude,
                                      const __nv_bfloat16* rms_weight_ptr, float rms_eps) {
  static_assert(
      kHCMult == 4 && kElementsPerThread == 8 && kThreadsPerBlock == 384 && kWeightsAligned32,
      "supported three-warp-group layout");
  static_assert(!kFuseRmsNorm || kCastBfloatForNorm, "RMSNorm requires BF16 intermediate");
  if constexpr (kHiddenDim == 6144 || kHiddenDim == 3072) {
    ihc_post_pre_single_row_device<kHiddenDim, kThreadsPerBlock, kFuseRmsNorm, kCastBfloatForNorm,
                                   kUsePDL, kWeightsAligned32, true>(
        y_ptr, z_ptr, H_post_out_ptr, xa_ptr, residual_ptr, H_post_in_ptr, w_ptr, hc_scale_ptr,
        hc_base_ptr, num_batch, norm_eps, hc_eps, magnitude, rms_weight_ptr, rms_eps);
  } else {
    static_assert(kHiddenDim == 4096, "supported hidden dimension");
    constexpr int kRowsPerBlock = 1;
    constexpr int kThreadsPerWarpGroup = 128,
                  kChunksPerRow = kHiddenDim / (kThreadsPerWarpGroup * kElementsPerThread);
    extern __shared__ __align__(16) unsigned char three_warp_storage[];
    auto& smem =
        *reinterpret_cast<IhcPostPreThreeWarpGroupStorage<kHiddenDim>*>(three_warp_storage);
    const int wg = threadIdx.x / kThreadsPerWarpGroup, tid = threadIdx.x % kThreadsPerWarpGroup;
    const int lane = tid % 32, warp = tid / 32;
    const int base = blockIdx.x * kRowsPerBlock;
    const int valid = min(kRowsPerBlock, num_batch - base);
    if constexpr (kUsePDL) cudaGridDependencySynchronize();
    if (wg == 0) {
      float sq[kRowsPerBlock] = {};
#pragma unroll 1
      for (int ch = 0; ch < kChunksPerRow; ++ch) {
        const int col = (ch * kThreadsPerWarpGroup + tid) * kElementsPerThread;
        vec_t<float, kElementsPerThread> a[kRowsPerBlock];
#pragma unroll
        for (int r = 0; r < kRowsPerBlock; ++r) {
          if (r < valid)
            a[r] = to<float>(load<__nv_bfloat16, kElementsPerThread>(
                xa_ptr + int64_t(base + r) * kHiddenDim + col));
        }
#pragma unroll 1
        for (int c = 0; c < 4; ++c) {
#pragma unroll
          for (int r = 0; r < kRowsPerBlock; ++r) {
            if (r < valid) {
              const int64_t flat = int64_t(base + r) * 4 * kHiddenDim + c * kHiddenDim + col;
              const auto b =
                  to<float>(load<__nv_bfloat16, kElementsPerThread>(residual_ptr + flat));
              const float h = H_post_in_ptr[(base + r) * 4 + c];
              vec_t<float, kElementsPerThread> v;
              auto* v2 = reinterpret_cast<float2*>(&v[0]);
              const auto* a2 = reinterpret_cast<const float2*>(&a[r][0]);
              const auto* b2 = reinterpret_cast<const float2*>(&b[0]);
#pragma unroll
              for (int j = 0; j < kElementsPerThread / 2; ++j)
                v2[j] = ihc_fma2(make_float2(h, h), a2[j], b2[j]);
              const auto bf = to<__nv_bfloat16>(v);
              store(y_ptr + flat, bf);
              store(&smem.post_output[r][c][col], bf);
              const auto rounded = to<float>(bf);
#pragma unroll
              for (int j = 0; j < kElementsPerThread; ++j) sq[r] += rounded[j] * rounded[j];
            }
          }
        }
        ihc_three_wg_barrier();
      }
#pragma unroll
      for (int r = 0; r < kRowsPerBlock; ++r) {
        const float sum = warp_reduce_sum_xor(sq[r]);
        if (lane == 0) smem.partial[r][8][warp] = sum;
      }
    } else {
      float acc_dot[kRowsPerBlock][4] = {};
#pragma unroll 1
      for (int ch = 0; ch < kChunksPerRow; ++ch) {
        const int col = (ch * kThreadsPerWarpGroup + tid) * kElementsPerThread;
        ihc_three_wg_barrier();
#pragma unroll 1
        for (int c = 0; c < 4; ++c) {
          vec_t<float, kElementsPerThread> v[kRowsPerBlock];
#pragma unroll
          for (int r = 0; r < kRowsPerBlock; ++r)
            if (r < valid)
              v[r] =
                  to<float>(load<__nv_bfloat16, kElementsPerThread>(&smem.post_output[r][c][col]));
#pragma unroll
          for (int p = 0; p < 4; ++p) {
            const int proj = (wg - 1) * 4 + p;
            const float* wp = w_ptr + int64_t(proj) * 4 * kHiddenDim + c * kHiddenDim + col;
            const auto weights = ihc_load_weight8<true>(wp);
#pragma unroll
            for (int r = 0; r < kRowsPerBlock; ++r) {
              if (r < valid) {
#pragma unroll
                for (int j = 0; j < kElementsPerThread; ++j) acc_dot[r][p] += v[r][j] * weights[j];
              }
            }
          }
        }
      }
#pragma unroll
      for (int r = 0; r < kRowsPerBlock; ++r) {
#pragma unroll
        for (int p = 0; p < 4; ++p) {
          const float sum = warp_reduce_sum_xor(acc_dot[r][p]);
          if (lane == 0) smem.partial[r][(wg - 1) * 4 + p][warp] = sum;
        }
      }
    }
    __syncthreads();
    if (threadIdx.x < kRowsPerBlock * 4) {
      const int r = threadIdx.x / 4, i = threadIdx.x % 4;
      if (r < valid) {
        float pre = 0, post = 0, sq = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          pre += smem.partial[r][i][k];
          post += smem.partial[r][i + 4][k];
          sq += smem.partial[r][8][k];
        }
        const float inv = rsqrtf_ftz(sq * (1.f / (4 * kHiddenDim)) + norm_eps);
        smem.pre_gates[r][i] = sigmoid(hc_scale_ptr[0] * (pre * inv) + hc_base_ptr[i]) + hc_eps;
        H_post_out_ptr[(base + r) * 4 + i] =
            magnitude * sigmoid(hc_scale_ptr[1] * (post * inv) + hc_base_ptr[i + 4]) + hc_eps;
      }
    }
    __syncthreads();
#pragma unroll 1
    for (int wave = 0; wave < (kRowsPerBlock + 2) / 3; ++wave) {
      const int r = wave * 3 + wg;
      const bool active = r < valid;
      vec_t<__nv_bfloat16, kElementsPerThread> saved[kChunksPerRow];
      float sq = 0;
      if (active) {
#pragma unroll
        for (int ch = 0; ch < kChunksPerRow; ++ch) {
          const int col = (ch * kThreadsPerWarpGroup + tid) * kElementsPerThread;
          vec_t<float, kElementsPerThread> out = {};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const auto v =
                to<float>(load<__nv_bfloat16, kElementsPerThread>(&smem.post_output[r][c][col]));
            auto* o2 = reinterpret_cast<float2*>(&out[0]);
            const auto* v2 = reinterpret_cast<const float2*>(&v[0]);
#pragma unroll
            for (int j = 0; j < kElementsPerThread / 2; ++j)
              o2[j] =
                  ihc_fma2(make_float2(smem.pre_gates[r][c], smem.pre_gates[r][c]), v2[j], o2[j]);
          }
          saved[ch] = to<__nv_bfloat16>(out);
          if constexpr (kFuseRmsNorm) {
            out = to<float>(saved[ch]);
#pragma unroll
            for (int j = 0; j < kElementsPerThread; ++j) sq += out[j] * out[j];
          } else {
            store(z_ptr + int64_t(base + r) * kHiddenDim + col, saved[ch]);
          }
        }
      }
      if constexpr (kFuseRmsNorm) {
        const float sum = warp_reduce_sum_xor(sq);
        if (lane == 0 && r < kRowsPerBlock) smem.norm_partial[r][warp] = sum;
        __syncthreads();
        if (active) {
          float total = 0;
#pragma unroll
          for (int k = 0; k < 4; ++k) total += smem.norm_partial[r][k];
          const float inv = rsqrtf_ftz(total * (1.f / kHiddenDim) + rms_eps);
#pragma unroll
          for (int ch = 0; ch < kChunksPerRow; ++ch) {
            const int col = (ch * kThreadsPerWarpGroup + tid) * kElementsPerThread;
            const auto g = to<float>(load<__nv_bfloat16, kElementsPerThread>(rms_weight_ptr + col));
            auto out = to<float>(saved[ch]);
            auto* o2 = reinterpret_cast<float2*>(&out[0]);
            const auto* g2 = reinterpret_cast<const float2*>(&g[0]);
#pragma unroll
            for (int j = 0; j < kElementsPerThread / 2; ++j)
              o2[j] = ihc_mul2(ihc_mul2(o2[j], make_float2(inv, inv)), g2[j]);
            store(z_ptr + int64_t(base + r) * kHiddenDim + col, to<__nv_bfloat16>(out));
          }
        }
      }
    }
    if constexpr (kUsePDL) cudaTriggerProgrammaticLaunchCompletion();
  }
}

#endif  // HPC_TARGET_ARCH != 90

}  // namespace kernels

namespace {

constexpr int kElementsPerThread = 8;
constexpr int kPostThreadsPerBlock = 128;

struct PreConfig {
  int rows;
  int threads;
};

#if HPC_TARGET_ARCH == 90
constexpr int pre_rows_per_block(int num_batch, bool fuse_rms_norm) {
  if (num_batch < (fuse_rms_norm ? 192 : 64)) {
    return 1;
  }
  if (num_batch < 320) {
    return 2;
  }
  return 4;
}

constexpr int head_rows_per_block(int num_batch, int hidden_dim) {
  if (num_batch < 320) {
    return 1;
  }
  if (hidden_dim == 6144 && num_batch >= 1024) {
    return 4;
  }
  return 2;
}

constexpr PreConfig pre_config(int num_batch, int hidden_dim) {
  if (hidden_dim == 6144) {
    if (num_batch < 192) {
      return {2, 256};
    }
    if (num_batch < 320) {
      return {4, 384};
    }
    if (num_batch < 512) {
      return {6, 256};
    }
    if (num_batch < 768) {
      return {4, 256};
    }
    if (num_batch >= 896 && num_batch < 1152) {
      return {8, 192};
    }
    if (num_batch < 12288) {
      return {6, 192};
    }
    return {8, 128};
  }
  if (num_batch < 384) {
    return {2, 256};
  }
  if (num_batch < 768) {
    return {4, 256};
  }
  if (num_batch >= 896 && num_batch < 1152) {
    return {8, 128};
  }
  if (num_batch < 12288) {
    return {6, 128};
  }
  return {4, 256};
}
#else
constexpr PreConfig sm100_config(int num_batch, int hidden_dim, bool has_post, bool fuse_rms_norm,
                                 int sm_count) {
  if (hidden_dim == 3072) {
    if (num_batch <= sm_count) {
      return {1, 384};
    }
    if (num_batch < 320) {
      return has_post ? PreConfig{2, 384} : PreConfig{1, 384};
    }
    if (has_post) {
      return num_batch <= 8 * sm_count ? PreConfig{4, 192} : PreConfig{4, 128};
    }
    return num_batch < 1024 ? PreConfig{2, 192} : PreConfig{4, 128};
  }
  if (hidden_dim == 6144) {
    if (num_batch <= sm_count) {
      return {1, 768};
    }
    if (num_batch < 320) {
      return {2, 384};
    }
    if (num_batch < 768) {
      return has_post ? PreConfig{4, 384} : PreConfig{2, 384};
    }
    if (num_batch < 1024) {
      return {4, 256};
    }
    if (!fuse_rms_norm) {
      return {4, 256};
    }
    if (has_post) {
      return num_batch < 2048 ? PreConfig{4, 256} : PreConfig{4, 128};
    }
    return (num_batch >= 2048 && num_batch < 8192) ? PreConfig{4, 192} : PreConfig{4, 256};
  }
  // The block span (threads * 8) must divide 4096.
  if (num_batch <= sm_count) {
    return {1, 512};
  }
  if (num_batch < 320) {
    if (!has_post && fuse_rms_norm) {
      return {1, 256};
    }
    return {2, 512};
  }
  if (num_batch < 1024) {
    return has_post ? PreConfig{4, 256} : PreConfig{2, 256};
  }
  if (has_post) {
    const int narrow_from = fuse_rms_norm ? 2048 : 4096;
    return num_batch < narrow_from ? PreConfig{4, 256} : PreConfig{4, 128};
  }
  if (num_batch < 4096) {
    return {4, 128};
  }
  if (!fuse_rms_norm) {
    return {2, 256};
  }
  return num_batch < 8192 ? PreConfig{2, 128} : PreConfig{4, 128};
}
#endif  // HPC_TARGET_ARCH == 90

template <int kHiddenDim, int kRowsPerBlock, bool kHasPost, bool kFuseRmsNorm,
          bool kCastBfloatForNorm, int kThreads = 256>
void launch_pre_head_rows(__nv_bfloat16* output_y_ptr, float* output_H_post_ptr,
                          const __nv_bfloat16* x_ptr, const float* w_ptr, const float* hc_scale_ptr,
                          const float* hc_base_ptr, int num_batch, float norm_eps, float hc_eps,
                          float magnitude, const __nv_bfloat16* rms_weight_ptr, float rms_eps,
                          cudaStream_t stream, bool use_pdl) {
  const int num_block = (num_batch + kRowsPerBlock - 1) / kRowsPerBlock;
  auto launch = [&](auto pdl_tag) {
    constexpr bool kUsePDL = decltype(pdl_tag)::value;
    auto kernel = kernels::fuse_ihc_pre_or_head_kernel<4, kHiddenDim, kElementsPerThread,
                                                       kRowsPerBlock, kThreads, kHasPost, kUsePDL,
                                                       kFuseRmsNorm, kCastBfloatForNorm>;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(num_block);
    config.blockDim = dim3(kThreads);
    config.stream = stream;
    cudaLaunchAttribute attr[1];
    config.numAttrs = 0;
    if constexpr (kUsePDL) {
      attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attr[0].val.programmaticStreamSerializationAllowed = 1;
      config.attrs = attr;
      config.numAttrs = 1;
    }
    cudaLaunchKernelEx(&config, kernel, output_y_ptr, output_H_post_ptr, x_ptr, w_ptr, hc_scale_ptr,
                       hc_base_ptr, num_batch, norm_eps, hc_eps, magnitude, rms_weight_ptr,
                       rms_eps);
  };
  if (use_pdl) {
    launch(std::true_type{});
  } else {
    launch(std::false_type{});
  }
}

template <int kHiddenDim, bool kHasPost, bool kFuseRmsNorm, bool kCastBfloatForNorm>
void launch_pre_head(__nv_bfloat16* output_y_ptr, float* output_H_post_ptr,
                     const __nv_bfloat16* x_ptr, const float* w_ptr, const float* hc_scale_ptr,
                     const float* hc_base_ptr, int num_batch, float norm_eps, float hc_eps,
                     float magnitude, const __nv_bfloat16* rms_weight_ptr, float rms_eps,
                     cudaStream_t stream, bool use_pdl) {
  constexpr int kRowThreads = kHiddenDim == 3072 ? 128 : 256;
  auto launch = [&](auto rows_tag, auto threads_tag) {
    launch_pre_head_rows<kHiddenDim, decltype(rows_tag)::value, kHasPost, kFuseRmsNorm,
                         kCastBfloatForNorm, decltype(threads_tag)::value>(
        output_y_ptr, output_H_post_ptr, x_ptr, w_ptr, hc_scale_ptr, hc_base_ptr, num_batch,
        norm_eps, hc_eps, magnitude, rms_weight_ptr, rms_eps, stream, use_pdl);
  };

#if HPC_TARGET_ARCH == 90
  if constexpr (kHasPost) {
    const PreConfig cfg = pre_config(num_batch, kHiddenDim);
    if constexpr (kHiddenDim == 6144) {
      if (cfg.rows == 2 && cfg.threads == 256) {
        launch(std::integral_constant<int, 2>{}, std::integral_constant<int, 256>{});
        return;
      }
      if (cfg.rows == 4 && cfg.threads == 384) {
        launch(std::integral_constant<int, 4>{}, std::integral_constant<int, 384>{});
        return;
      }
      if (cfg.rows == 6 && cfg.threads == 256) {
        launch(std::integral_constant<int, 6>{}, std::integral_constant<int, 256>{});
        return;
      }
      if (cfg.rows == 4 && cfg.threads == 256) {
        launch(std::integral_constant<int, 4>{}, std::integral_constant<int, 256>{});
        return;
      }
      if (cfg.rows == 8 && cfg.threads == 192) {
        launch(std::integral_constant<int, 8>{}, std::integral_constant<int, 192>{});
        return;
      }
      if (cfg.rows == 6 && cfg.threads == 192) {
        launch(std::integral_constant<int, 6>{}, std::integral_constant<int, 192>{});
        return;
      }
      if (cfg.rows == 8 && cfg.threads == 128) {
        launch(std::integral_constant<int, 8>{}, std::integral_constant<int, 128>{});
        return;
      }
    } else if constexpr (kHiddenDim == 4096) {
      if (cfg.rows == 2 && cfg.threads == 256) {
        launch(std::integral_constant<int, 2>{}, std::integral_constant<int, 256>{});
        return;
      }
      if (cfg.rows == 4 && cfg.threads == 256) {
        launch(std::integral_constant<int, 4>{}, std::integral_constant<int, 256>{});
        return;
      }
      if (cfg.rows == 8 && cfg.threads == 128) {
        launch(std::integral_constant<int, 8>{}, std::integral_constant<int, 128>{});
        return;
      }
      if (cfg.rows == 6 && cfg.threads == 128) {
        launch(std::integral_constant<int, 6>{}, std::integral_constant<int, 128>{});
        return;
      }
    }
  }

  if constexpr (kHasPost) {
    switch (pre_rows_per_block(num_batch, kFuseRmsNorm)) {
      case 1:
        launch(std::integral_constant<int, 1>{}, std::integral_constant<int, kRowThreads>{});
        break;
      case 2:
        launch(std::integral_constant<int, 2>{}, std::integral_constant<int, kRowThreads>{});
        break;
      default:
        launch(std::integral_constant<int, 4>{}, std::integral_constant<int, kRowThreads>{});
        break;
    }
  } else {
    switch (head_rows_per_block(num_batch, kHiddenDim)) {
      case 1:
        launch(std::integral_constant<int, 1>{}, std::integral_constant<int, kRowThreads>{});
        break;
      case 2:
        launch(std::integral_constant<int, 2>{}, std::integral_constant<int, kRowThreads>{});
        break;
      default:
        if constexpr (kHiddenDim == 6144) {
          launch(std::integral_constant<int, 4>{}, std::integral_constant<int, 256>{});
        }
        break;
    }
  }
#else
  const PreConfig cfg = sm100_config(num_batch, kHiddenDim, kHasPost, kFuseRmsNorm, get_sm_count());
  if constexpr (kHiddenDim == 6144) {
    if (cfg.rows == 1 && cfg.threads == 768) {
      launch(std::integral_constant<int, 1>{}, std::integral_constant<int, 768>{});
      return;
    }
    if (cfg.rows == 2 && cfg.threads == 384) {
      launch(std::integral_constant<int, 2>{}, std::integral_constant<int, 384>{});
      return;
    }
    if constexpr (kHasPost) {
      if (cfg.rows == 4 && cfg.threads == 384) {
        launch(std::integral_constant<int, 4>{}, std::integral_constant<int, 384>{});
        return;
      }
      if (cfg.rows == 4 && cfg.threads == 128) {
        launch(std::integral_constant<int, 4>{}, std::integral_constant<int, 128>{});
        return;
      }
    }
    if (cfg.rows == 4 && cfg.threads == 256) {
      launch(std::integral_constant<int, 4>{}, std::integral_constant<int, 256>{});
      return;
    }
    if constexpr (!kHasPost) {
      if (cfg.rows == 4 && cfg.threads == 192) {
        launch(std::integral_constant<int, 4>{}, std::integral_constant<int, 192>{});
        return;
      }
    }
  } else if constexpr (kHiddenDim == 3072) {
    if (cfg.rows == 1 && cfg.threads == 384) {
      launch(std::integral_constant<int, 1>{}, std::integral_constant<int, 384>{});
      return;
    }
    if (cfg.rows == 2 && cfg.threads == 384) {
      launch(std::integral_constant<int, 2>{}, std::integral_constant<int, 384>{});
      return;
    }
    if constexpr (kHasPost) {
      if (cfg.rows == 4 && cfg.threads == 192) {
        launch(std::integral_constant<int, 4>{}, std::integral_constant<int, 192>{});
        return;
      }
    } else {
      if (cfg.rows == 2 && cfg.threads == 192) {
        launch(std::integral_constant<int, 2>{}, std::integral_constant<int, 192>{});
        return;
      }
    }
    if (cfg.rows == 4 && cfg.threads == 128) {
      launch(std::integral_constant<int, 4>{}, std::integral_constant<int, 128>{});
      return;
    }
  } else if constexpr (kHiddenDim == 4096) {
    if (cfg.rows == 1 && cfg.threads == 512) {
      launch(std::integral_constant<int, 1>{}, std::integral_constant<int, 512>{});
      return;
    }
    if (cfg.rows == 2 && cfg.threads == 512) {
      launch(std::integral_constant<int, 2>{}, std::integral_constant<int, 512>{});
      return;
    }
    if (cfg.rows == 2 && cfg.threads == 256) {
      launch(std::integral_constant<int, 2>{}, std::integral_constant<int, 256>{});
      return;
    }
    if constexpr (kHasPost) {
      if (cfg.rows == 4 && cfg.threads == 256) {
        launch(std::integral_constant<int, 4>{}, std::integral_constant<int, 256>{});
        return;
      }
    } else {
      if (cfg.rows == 1 && cfg.threads == 256) {
        launch(std::integral_constant<int, 1>{}, std::integral_constant<int, 256>{});
        return;
      }
      if (cfg.rows == 2 && cfg.threads == 128) {
        launch(std::integral_constant<int, 2>{}, std::integral_constant<int, 128>{});
        return;
      }
    }
    if (cfg.rows == 4 && cfg.threads == 128) {
      launch(std::integral_constant<int, 4>{}, std::integral_constant<int, 128>{});
      return;
    }
  }
  launch(std::integral_constant<int, 1>{}, std::integral_constant<int, kRowThreads>{});
#endif  // HPC_TARGET_ARCH == 90
}

template <int kHiddenDim, bool kFuseRmsNorm, bool kCastBfloatForNorm, bool kUsePDL>
cudaError_t launch_fuse_ihc_post_pre(int num_batch, cudaStream_t stream, __nv_bfloat16* y_ptr,
                                     __nv_bfloat16* z_ptr, float* H_post_out_ptr,
                                     const __nv_bfloat16* xa_ptr, const __nv_bfloat16* residual_ptr,
                                     const float* H_post_in_ptr, const float* w_ptr,
                                     const float* hc_scale_ptr, const float* hc_base_ptr,
                                     float norm_eps, float hc_eps, float magnitude,
                                     const __nv_bfloat16* rms_weight_ptr, float rms_eps) {
  constexpr int kThreads = kHiddenDim == 3072 ? 192 : 256;
#if HPC_TARGET_ARCH != 90
  constexpr int k6144NonClusterBatchMin = 80;
  constexpr int kRows2BatchMin = 297;
  const bool weights_aligned32 = (reinterpret_cast<uintptr_t>(w_ptr) & 31) == 0;
  if constexpr (kUsePDL && (!kFuseRmsNorm || kCastBfloatForNorm)) {
    constexpr int kUmmaMin = kHiddenDim == 3072 ? 8960 : 8192;
    if (num_batch >= kUmmaMin && weights_aligned32) {
      constexpr int kUmmaThreads = 256;
      constexpr int kUmmaBlocks = kernels::ihc_post_pre_num_blocks(kHiddenDim);
      auto kernel =
          kernels::fuse_ihc_post_pre_umma_kernel<4, kHiddenDim, 8, kUmmaThreads, kUsePDL,
                                                 kFuseRmsNorm, kCastBfloatForNorm, true, 8>;
      constexpr size_t kShared = sizeof(kernels::IhcPostPreUmmaStorage<128, 3>);
      const cudaError_t setup =
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kShared);
      if (setup != cudaSuccess) return setup;
      cudaLaunchConfig_t umma_config{};
      umma_config.gridDim = dim3(((num_batch + 63) / 64) * kUmmaBlocks);
      umma_config.blockDim = dim3(kUmmaThreads);
      umma_config.dynamicSmemBytes = kShared;
      umma_config.stream = stream;
      cudaLaunchAttribute umma_attrs[2]{};
      umma_attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      umma_attrs[0].val.programmaticStreamSerializationAllowed = 1;
      umma_attrs[1].id = cudaLaunchAttributeClusterDimension;
      umma_attrs[1].val.clusterDim.x = kUmmaBlocks;
      umma_attrs[1].val.clusterDim.y = 1;
      umma_attrs[1].val.clusterDim.z = 1;
      umma_config.attrs = umma_attrs;
      umma_config.numAttrs = 2;
      return cudaLaunchKernelEx(&umma_config, kernel, y_ptr, z_ptr, H_post_out_ptr, xa_ptr,
                                residual_ptr, H_post_in_ptr, w_ptr, hc_scale_ptr, hc_base_ptr,
                                num_batch, norm_eps, hc_eps, magnitude, rms_weight_ptr, rms_eps);
    }
    if constexpr (kHiddenDim != 4096 || !kFuseRmsNorm) {
      constexpr int kThreeWarpGroupMin = kHiddenDim == 6144   ? k6144NonClusterBatchMin
                                         : kHiddenDim == 3072 ? (kFuseRmsNorm ? 48 : 72)
                                                              : 149;
      const int three_warp_group_max = kHiddenDim == 3072 ? 4 * get_sm_count() : kRows2BatchMin - 1;
      if (num_batch >= kThreeWarpGroupMin && num_batch <= three_warp_group_max &&
          weights_aligned32) {
        constexpr int kThreeWarpGroupThreads = 384;
        auto kernel =
            kernels::fuse_ihc_post_pre_three_wg_kernel<4, kHiddenDim, 8, kThreeWarpGroupThreads,
                                                       kUsePDL, kFuseRmsNorm, kCastBfloatForNorm,
                                                       true>;
        constexpr size_t kShared =
            kHiddenDim == 4096 ? sizeof(kernels::IhcPostPreThreeWarpGroupStorage<4096>) : 0;
        if constexpr (kShared > 0) {
          const cudaError_t setup =
              cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kShared);
          if (setup != cudaSuccess) return setup;
        }
        cudaLaunchConfig_t three_warp_group_config{};
        three_warp_group_config.gridDim = dim3(num_batch);
        three_warp_group_config.blockDim = dim3(kThreeWarpGroupThreads);
        three_warp_group_config.dynamicSmemBytes = kShared;
        three_warp_group_config.stream = stream;
        cudaLaunchAttribute attr{};
        attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr.val.programmaticStreamSerializationAllowed = 1;
        three_warp_group_config.attrs = &attr;
        three_warp_group_config.numAttrs = 1;
        return cudaLaunchKernelEx(&three_warp_group_config, kernel, y_ptr, z_ptr, H_post_out_ptr,
                                  xa_ptr, residual_ptr, H_post_in_ptr, w_ptr, hc_scale_ptr,
                                  hc_base_ptr, num_batch, norm_eps, hc_eps, magnitude,
                                  rms_weight_ptr, rms_eps);
      }
    }
  }
#endif
  cudaLaunchConfig_t config{};
  config.stream = stream;
  cudaLaunchAttribute attrs[2]{};
  config.attrs = attrs;
  if constexpr (kUsePDL) {
    attrs[config.numAttrs].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[config.numAttrs++].val.programmaticStreamSerializationAllowed = 1;
  }
#if HPC_TARGET_ARCH == 90
  if (num_batch == 0) {
    return cudaSuccess;
  }
  using kernels::PostPrePath;
  // Zero leaves the minimum blocks per SM unspecified for ptxas.
  auto launch = [&]<int kBlockThreads, PostPrePath kPath = PostPrePath::Rows, int kRowsPerBlock = 1,
                    int kClusterBlocks = 1, int kMinBlocksPerSm = 0,
                    int kEpilogueThreads = kBlockThreads>() {
    config.gridDim = dim3(((num_batch + kRowsPerBlock - 1) / kRowsPerBlock) * kClusterBlocks);
    config.blockDim = dim3(kBlockThreads);
    if constexpr (kClusterBlocks > 1) {
      auto& attr = attrs[config.numAttrs++];
      attr.id = cudaLaunchAttributeClusterDimension;
      attr.val.clusterDim.x = kClusterBlocks;
      attr.val.clusterDim.y = 1;
      attr.val.clusterDim.z = 1;
    }
    auto kernel =
        kernels::fuse_ihc_post_pre_sm90_kernel<kHiddenDim, kBlockThreads, kPath, kRowsPerBlock,
                                               kClusterBlocks, kMinBlocksPerSm, kUsePDL,
                                               kFuseRmsNorm, kCastBfloatForNorm, kEpilogueThreads>;
    if constexpr (kPath == PostPrePath::SharedPostOutput) {
      constexpr size_t kShared = kRowsPerBlock * 4 * kHiddenDim * sizeof(__nv_bfloat16);
      const cudaError_t setup =
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kShared);
      if (setup != cudaSuccess) return setup;
      config.dynamicSmemBytes = kShared;
    }
    return cudaLaunchKernelEx(&config, kernel, y_ptr, z_ptr, H_post_out_ptr, xa_ptr, residual_ptr,
                              H_post_in_ptr, w_ptr, hc_scale_ptr, hc_base_ptr, num_batch, norm_eps,
                              hc_eps, magnitude, rms_weight_ptr, rms_eps);
  };

  // Cluster and specialized single-row kernels vectorize the four input gates.
  if (reinterpret_cast<uintptr_t>(H_post_in_ptr) % (4 * sizeof(float)) != 0) {
    return launch.template operator()<kThreads>();
  }
  const int sm_count = get_sm_count();
  constexpr int kHiddenThreads = kHiddenDim / kElementsPerThread;
  const int waves = 1 + (num_batch - 1) / sm_count;
  if constexpr (kHiddenDim == 3072) {
    if (waves == 4 || (waves == 5 && 16 * int64_t(num_batch) <= 65 * sm_count)) {
      return launch.template operator()<kHiddenThreads, PostPrePath::SharedPostOutput, 2, 1, 2,
                                        kHiddenThreads / 2>();
    }
    if ((waves == 6 && 2 * int64_t(num_batch) <= 11 * sm_count) || waves == 8 || waves == 16 ||
        num_batch >= 3072) {
      return launch.template operator()<128, PostPrePath::SharedPostOutput, 2, 1, 3, 64>();
    }
  } else if constexpr (kHiddenDim == 4096) {
    if (num_batch >= 3 * sm_count && waves <= 4) {
      return launch.template operator()<kThreads, PostPrePath::SharedPostOutput, 2, 1, 2, 128>();
    }
  } else if (waves == 4 && ((kFuseRmsNorm && !kCastBfloatForNorm) ||
                            8 * num_batch >= 31 * sm_count)) {
    return launch.template operator()<kThreads, PostPrePath::SharedPostOutput, 2, 1, 2, 128>();
  }
  if constexpr (kHiddenDim != 3072) {
    // Split the hidden dimension across CTAs when there are few tokens to schedule.
    const int cluster_limit = kHiddenDim == 6144 ? 3 * sm_count / 5 : sm_count / 2;
    if (num_batch <= cluster_limit) {
      constexpr int kClusterBlocks = kHiddenDim == 6144 ? 8 : 4;
      constexpr int kClusterThreads = kHiddenThreads / kClusterBlocks;
      return launch.template operator()<kClusterThreads, PostPrePath::Cluster, 1, kClusterBlocks>();
    }
  }
  if (num_batch <= sm_count) {
    return launch.template operator()<kHiddenThreads, PostPrePath::Retained>();
  }
  if (waves <= 2) {
    constexpr int kMinBlocks = kHiddenDim == 6144 ? 0 : 1;
    return launch.template operator()<kHiddenThreads, PostPrePath::Rows, 2, 1, kMinBlocks>();
  }
  if constexpr (kHiddenDim == 6144) {
    if (waves <= 4) {
      return launch.template operator()<kThreads>();
    }
    if (waves == 5) {
      if constexpr (kFuseRmsNorm) {
        return launch.template operator()<96>();
      } else {
        return launch.template operator()<kThreads, PostPrePath::Rows, 2>();
      }
    }
    return launch.template operator()<kHiddenThreads / 2, PostPrePath::Rows, 1, 1, 2>();
  } else if constexpr (kHiddenDim == 4096) {
    if (waves <= 3) {
      return launch.template operator()<kThreads>();
    }
    if (waves == 4) {
      return launch.template operator()<kThreads, PostPrePath::Rows, 2, 1, 2>();
    }
    if (waves == 5 && kFuseRmsNorm && kCastBfloatForNorm) {
      return launch.template operator()<kThreads, PostPrePath::Rows, 2>();
    }
    return launch.template operator()<kHiddenThreads / 2, PostPrePath::Retained, 1, 1, 2>();
  } else {
    static_assert(kHiddenDim == 3072);
    if (waves <= 4 || waves == 8 || (kFuseRmsNorm && waves >= 11 && waves <= 13)) {
      return launch.template operator()<kHiddenThreads, PostPrePath::Rows, 2, 1, 2>();
    }
    if ((waves <= 6 && 2 * num_batch <= (2 * waves - 1) * sm_count) || waves == 7) {
      // Reread y with fewer registers so more CTAs can be resident.
      return launch.template operator()<128, PostPrePath::ReloadPostOutput>();
    }
    if (waves <= 5) {
      return launch.template operator()<kThreads>();
    }
    if (waves == 9 || num_batch >= 3072) {
      return launch.template operator()<kHiddenThreads>();
    }
    return launch.template operator()<kHiddenThreads / 2, PostPrePath::Retained, 1, 1, 2>();
  }
#else
  if constexpr (kHiddenDim == 6144) {
    if (num_batch < k6144NonClusterBatchMin) {
      constexpr int kBlocks = kernels::ihc_post_pre_num_blocks(kHiddenDim);
      constexpr int kClusterThreads = kHiddenDim / (kBlocks * kElementsPerThread);
      config.gridDim = dim3(num_batch * kBlocks);
      config.blockDim = dim3(kClusterThreads);
      auto& cattr = attrs[config.numAttrs++];
      cattr.id = cudaLaunchAttributeClusterDimension;
      cattr.val.clusterDim = {kBlocks, 1, 1};
      auto kernel = kernels::fuse_ihc_post_pre_cluster_kernel<4, kHiddenDim, kElementsPerThread,
                                                              kBlocks, kClusterThreads, kUsePDL,
                                                              kFuseRmsNorm, kCastBfloatForNorm>;
      return cudaLaunchKernelEx(&config, kernel, y_ptr, z_ptr, H_post_out_ptr, xa_ptr, residual_ptr,
                                H_post_in_ptr, w_ptr, hc_scale_ptr, hc_base_ptr, num_batch,
                                norm_eps, hc_eps, magnitude, rms_weight_ptr, rms_eps);
    }
    if (num_batch < kRows2BatchMin) {
      constexpr int kRetainedThreads = kHiddenDim / kElementsPerThread;
      config.gridDim = dim3(num_batch);
      config.blockDim = dim3(kRetainedThreads);
      auto& cattr = attrs[config.numAttrs++];
      cattr.id = cudaLaunchAttributeClusterDimension;
      cattr.val.clusterDim = {1, 1, 1};
      auto launch = [&](auto aligned32_tag) {
        constexpr bool kWeightsAligned32 = decltype(aligned32_tag)::value;
        auto kernel =
            kernels::fuse_ihc_post_pre_retained_kernel<4, kHiddenDim, kElementsPerThread,
                                                       kRetainedThreads, kUsePDL, kFuseRmsNorm,
                                                       kCastBfloatForNorm, kWeightsAligned32>;
        return cudaLaunchKernelEx(&config, kernel, y_ptr, z_ptr, H_post_out_ptr, xa_ptr,
                                  residual_ptr, H_post_in_ptr, w_ptr, hc_scale_ptr, hc_base_ptr,
                                  num_batch, norm_eps, hc_eps, magnitude, rms_weight_ptr, rms_eps);
      };
      return weights_aligned32 ? launch(std::true_type{}) : launch(std::false_type{});
    }
    constexpr int kRowsPerBlock = 2;
    constexpr int kRowsThreads = 384;
    config.gridDim = dim3((num_batch + kRowsPerBlock - 1) / kRowsPerBlock);
    config.blockDim = dim3(kRowsThreads);
    auto& cattr = attrs[config.numAttrs++];
    cattr.id = cudaLaunchAttributeClusterDimension;
    cattr.val.clusterDim = {1, 1, 1};
    auto launch = [&](auto aligned32_tag) {
      constexpr bool kWeightsAligned32 = decltype(aligned32_tag)::value;
      auto kernel =
          kernels::fuse_ihc_post_pre_kernel<4, kHiddenDim, kElementsPerThread, kRowsThreads,
                                            kRowsPerBlock, kUsePDL, kFuseRmsNorm,
                                            kCastBfloatForNorm, kWeightsAligned32>;
      return cudaLaunchKernelEx(&config, kernel, y_ptr, z_ptr, H_post_out_ptr, xa_ptr, residual_ptr,
                                H_post_in_ptr, w_ptr, hc_scale_ptr, hc_base_ptr, num_batch,
                                norm_eps, hc_eps, magnitude, rms_weight_ptr, rms_eps);
    };
    return weights_aligned32 ? launch(std::true_type{}) : launch(std::false_type{});
  } else {
    static_assert(kHiddenDim == 4096 || kHiddenDim == 3072, "supported hidden dimensions");
    constexpr int kClusterBlocks = kernels::ihc_post_pre_num_blocks(kHiddenDim);
    constexpr int kClusterThreads = kHiddenDim / (kClusterBlocks * kElementsPerThread);
    if (num_batch <= 128) {
      config.gridDim = dim3(num_batch * kClusterBlocks);
      config.blockDim = dim3(kClusterThreads);
      auto& cattr = attrs[config.numAttrs++];
      cattr.id = cudaLaunchAttributeClusterDimension;
      cattr.val.clusterDim = {kClusterBlocks, 1, 1};
      auto kernel =
          kernels::fuse_ihc_post_pre_cluster_kernel<4, kHiddenDim, kElementsPerThread,
                                                    kClusterBlocks, kClusterThreads, kUsePDL,
                                                    kFuseRmsNorm, kCastBfloatForNorm>;
      return cudaLaunchKernelEx(&config, kernel, y_ptr, z_ptr, H_post_out_ptr, xa_ptr, residual_ptr,
                                H_post_in_ptr, w_ptr, hc_scale_ptr, hc_base_ptr, num_batch,
                                norm_eps, hc_eps, magnitude, rms_weight_ptr, rms_eps);
    }
    const int rows = num_batch >= kRows2BatchMin ? 2 : 1;
    config.gridDim = dim3((num_batch + rows - 1) / rows);
    config.blockDim = dim3(kThreads);
    auto& cattr = attrs[config.numAttrs++];
    cattr.id = cudaLaunchAttributeClusterDimension;
    cattr.val.clusterDim = {1, 1, 1};
    if (rows == 2) {
      auto launch = [&](auto aligned32_tag) {
        constexpr bool kWeightsAligned32 = decltype(aligned32_tag)::value;
        auto kernel =
            kernels::fuse_ihc_post_pre_kernel<4, kHiddenDim, 8, kThreads, 2, kUsePDL, kFuseRmsNorm,
                                              kCastBfloatForNorm, kWeightsAligned32>;
        return cudaLaunchKernelEx(&config, kernel, y_ptr, z_ptr, H_post_out_ptr, xa_ptr,
                                  residual_ptr, H_post_in_ptr, w_ptr, hc_scale_ptr, hc_base_ptr,
                                  num_batch, norm_eps, hc_eps, magnitude, rms_weight_ptr, rms_eps);
      };
      return weights_aligned32 ? launch(std::true_type{}) : launch(std::false_type{});
    }
    auto launch = [&](auto aligned32_tag) {
      constexpr bool kWeightsAligned32 = decltype(aligned32_tag)::value;
      auto kernel =
          kernels::fuse_ihc_post_pre_kernel<4, kHiddenDim, 8, kThreads, 1, kUsePDL, kFuseRmsNorm,
                                            kCastBfloatForNorm, kWeightsAligned32>;
      return cudaLaunchKernelEx(&config, kernel, y_ptr, z_ptr, H_post_out_ptr, xa_ptr, residual_ptr,
                                H_post_in_ptr, w_ptr, hc_scale_ptr, hc_base_ptr, num_batch,
                                norm_eps, hc_eps, magnitude, rms_weight_ptr, rms_eps);
    };
    return weights_aligned32 ? launch(std::true_type{}) : launch(std::false_type{});
  }
#endif
}

template <int kHiddenDim, bool kFuseRmsNorm, bool kCastBfloatForNorm>
void launch_post_pre(__nv_bfloat16* y_ptr, __nv_bfloat16* z_ptr, float* H_post_out_ptr,
                     const __nv_bfloat16* xa_ptr, const __nv_bfloat16* residual_ptr,
                     const float* H_post_in_ptr, const float* w_ptr, const float* hc_scale_ptr,
                     const float* hc_base_ptr, int num_batch, float norm_eps, float hc_eps,
                     float magnitude, const __nv_bfloat16* rms_weight_ptr, float rms_eps,
                     cudaStream_t stream, bool use_pdl) {
  auto launch = [&](auto pdl_tag) {
    constexpr bool kUsePDL = decltype(pdl_tag)::value;
    launch_fuse_ihc_post_pre<kHiddenDim, kFuseRmsNorm, kCastBfloatForNorm, kUsePDL>(
        num_batch, stream, y_ptr, z_ptr, H_post_out_ptr, xa_ptr, residual_ptr, H_post_in_ptr, w_ptr,
        hc_scale_ptr, hc_base_ptr, norm_eps, hc_eps, magnitude, rms_weight_ptr, rms_eps);
  };
  if (use_pdl) {
    launch(std::true_type{});
  } else {
    launch(std::false_type{});
  }
}

template <int kHiddenDim, int kThreads>
void launch_post_shape(__nv_bfloat16* output_ptr, const __nv_bfloat16* x_ptr,
                       const __nv_bfloat16* residual_ptr, const float* H_post_ptr, int num_batch,
                       int block_per_row, cudaStream_t stream, bool use_pdl) {
  dim3 block(kThreads);
  dim3 grid(num_batch, block_per_row, 1);

  auto launch = [&](auto pdl_tag) {
    constexpr bool kUsePDL = decltype(pdl_tag)::value;
    auto kernel =
        kernels::fuse_ihc_post_kernel<4, kHiddenDim, kElementsPerThread, kThreads, kUsePDL>;
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = block;
    config.stream = stream;
    cudaLaunchAttribute attr[1];
    config.numAttrs = 0;
    if constexpr (kUsePDL) {
      attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attr[0].val.programmaticStreamSerializationAllowed = 1;
      config.attrs = attr;
      config.numAttrs = 1;
    }
    cudaLaunchKernelEx(&config, kernel, output_ptr, x_ptr, residual_ptr, H_post_ptr);
  };
  if (use_pdl) {
    launch(std::true_type{});
  } else {
    launch(std::false_type{});
  }
}

struct PostConfig {
  int threads;
  int splits;
};

constexpr PostConfig post_config(int num_batch, int hidden_dim) {
  int threads = kPostThreadsPerBlock;
#if HPC_TARGET_ARCH != 90
  if (hidden_dim == 4096 && num_batch >= 4096) {
    threads = 256;
  }
#endif

  const int max_split = hidden_dim / (threads * kElementsPerThread);
  return {threads, max_split > 1 ? max_split : 1};
}

template <int kHiddenDim>
void launch_post(__nv_bfloat16* output_ptr, const __nv_bfloat16* x_ptr,
                 const __nv_bfloat16* residual_ptr, const float* H_post_ptr, int num_batch,
                 cudaStream_t stream, bool use_pdl) {
  const PostConfig cfg = post_config(num_batch, kHiddenDim);

  auto go = [&](auto threads_tag) {
    launch_post_shape<kHiddenDim, decltype(threads_tag)::value>(
        output_ptr, x_ptr, residual_ptr, H_post_ptr, num_batch, cfg.splits, stream, use_pdl);
  };

  if constexpr (kHiddenDim == 4096) {
    if (cfg.threads == 256) {
      go(std::integral_constant<int, 256>{});
      return;
    }
  }
  go(std::integral_constant<int, kPostThreadsPerBlock>{});
}

}  // namespace

void fuse_ihc_pre_async(__nv_bfloat16* output_y_ptr, float* output_H_post_ptr,
                        const __nv_bfloat16* x_ptr, const float* w_ptr, const float* hc_scale_ptr,
                        const float* hc_base_ptr, int num_batch, int hc_mult, int hidden_dim,
                        float norm_eps, float hc_eps, float magnitude, cudaStream_t stream,
                        const __nv_bfloat16* rms_weight_ptr, float rms_eps,
                        bool cast_bfloat_for_norm, bool use_pdl) {
  if (hc_mult != 4) {
    return;
  }
  auto by_hidden = [&](auto hidden_tag) {
    constexpr int kHidden = decltype(hidden_tag)::value;
    if (rms_weight_ptr == nullptr) {
      launch_pre_head<kHidden, true, false, false>(
          output_y_ptr, output_H_post_ptr, x_ptr, w_ptr, hc_scale_ptr, hc_base_ptr, num_batch,
          norm_eps, hc_eps, magnitude, nullptr, 0.f, stream, use_pdl);
    } else if (cast_bfloat_for_norm) {
      launch_pre_head<kHidden, true, true, true>(
          output_y_ptr, output_H_post_ptr, x_ptr, w_ptr, hc_scale_ptr, hc_base_ptr, num_batch,
          norm_eps, hc_eps, magnitude, rms_weight_ptr, rms_eps, stream, use_pdl);
    } else {
      launch_pre_head<kHidden, true, true, false>(
          output_y_ptr, output_H_post_ptr, x_ptr, w_ptr, hc_scale_ptr, hc_base_ptr, num_batch,
          norm_eps, hc_eps, magnitude, rms_weight_ptr, rms_eps, stream, use_pdl);
    }
  };

  if (hidden_dim == 6144) {
    by_hidden(std::integral_constant<int, 6144>{});
  } else if (hidden_dim == 4096) {
    by_hidden(std::integral_constant<int, 4096>{});
  } else if (hidden_dim == 3072) {
    by_hidden(std::integral_constant<int, 3072>{});
  } else {
    return;
  }
}

void fuse_ihc_post_pre_async(__nv_bfloat16* y_ptr, __nv_bfloat16* z_ptr, float* H_post_out_ptr,
                             const __nv_bfloat16* xa_ptr, const __nv_bfloat16* residual_ptr,
                             const float* H_post_in_ptr, const float* w_ptr,
                             const float* hc_scale_ptr, const float* hc_base_ptr, int num_batch,
                             int hc_mult, int hidden_dim, float norm_eps, float hc_eps,
                             float magnitude, cudaStream_t stream,
                             const __nv_bfloat16* rms_weight_ptr, float rms_eps,
                             bool cast_bfloat_for_norm, bool use_pdl) {
  if (hc_mult != 4) {
    return;
  }
  auto by_hidden = [&](auto hidden_tag) {
    constexpr int kHidden = decltype(hidden_tag)::value;
    if (rms_weight_ptr == nullptr) {
      launch_post_pre<kHidden, false, false>(
          y_ptr, z_ptr, H_post_out_ptr, xa_ptr, residual_ptr, H_post_in_ptr, w_ptr, hc_scale_ptr,
          hc_base_ptr, num_batch, norm_eps, hc_eps, magnitude, nullptr, 0.f, stream, use_pdl);
    } else if (cast_bfloat_for_norm) {
      launch_post_pre<kHidden, true, true>(y_ptr, z_ptr, H_post_out_ptr, xa_ptr, residual_ptr,
                                           H_post_in_ptr, w_ptr, hc_scale_ptr, hc_base_ptr,
                                           num_batch, norm_eps, hc_eps, magnitude, rms_weight_ptr,
                                           rms_eps, stream, use_pdl);
    } else {
      launch_post_pre<kHidden, true, false>(y_ptr, z_ptr, H_post_out_ptr, xa_ptr, residual_ptr,
                                            H_post_in_ptr, w_ptr, hc_scale_ptr, hc_base_ptr,
                                            num_batch, norm_eps, hc_eps, magnitude, rms_weight_ptr,
                                            rms_eps, stream, use_pdl);
    }
  };

  if (hidden_dim == 6144) {
    by_hidden(std::integral_constant<int, 6144>{});
  } else if (hidden_dim == 4096) {
    by_hidden(std::integral_constant<int, 4096>{});
  } else if (hidden_dim == 3072) {
    by_hidden(std::integral_constant<int, 3072>{});
  } else {
    return;
  }
}

void fuse_ihc_post_async(__nv_bfloat16* output_ptr, const __nv_bfloat16* x_ptr,
                         const __nv_bfloat16* residual_ptr, const float* H_post_ptr, int num_batch,
                         int hc_mult, int hidden_dim, cudaStream_t stream, bool use_pdl) {
  if (hc_mult != 4) {
    return;
  }
  if (hidden_dim == 6144) {
    launch_post<6144>(output_ptr, x_ptr, residual_ptr, H_post_ptr, num_batch, stream, use_pdl);
  } else if (hidden_dim == 4096) {
    launch_post<4096>(output_ptr, x_ptr, residual_ptr, H_post_ptr, num_batch, stream, use_pdl);
  } else if (hidden_dim == 3072) {
    launch_post<3072>(output_ptr, x_ptr, residual_ptr, H_post_ptr, num_batch, stream, use_pdl);
  } else {
    return;
  }
}

void fuse_ihc_head_async(__nv_bfloat16* output_ptr, const __nv_bfloat16* x_ptr, const float* w_ptr,
                         const float* hc_scale_ptr, const float* hc_base_ptr, int num_batch,
                         int hc_mult, int hidden_dim, float norm_eps, float hc_eps,
                         cudaStream_t stream, const __nv_bfloat16* rms_weight_ptr, float rms_eps,
                         bool cast_bfloat_for_norm, bool use_pdl) {
  if (hc_mult != 4) {
    return;
  }
  auto by_hidden = [&](auto hidden_tag) {
    constexpr int kHidden = decltype(hidden_tag)::value;
    if (rms_weight_ptr == nullptr) {
      launch_pre_head<kHidden, false, false, false>(output_ptr, nullptr, x_ptr, w_ptr, hc_scale_ptr,
                                                    hc_base_ptr, num_batch, norm_eps, hc_eps, 0.f,
                                                    nullptr, 0.f, stream, use_pdl);
    } else if (cast_bfloat_for_norm) {
      launch_pre_head<kHidden, false, true, true>(output_ptr, nullptr, x_ptr, w_ptr, hc_scale_ptr,
                                                  hc_base_ptr, num_batch, norm_eps, hc_eps, 0.f,
                                                  rms_weight_ptr, rms_eps, stream, use_pdl);
    } else {
      launch_pre_head<kHidden, false, true, false>(output_ptr, nullptr, x_ptr, w_ptr, hc_scale_ptr,
                                                   hc_base_ptr, num_batch, norm_eps, hc_eps, 0.f,
                                                   rms_weight_ptr, rms_eps, stream, use_pdl);
    }
  };

  if (hidden_dim == 6144) {
    by_hidden(std::integral_constant<int, 6144>{});
  } else if (hidden_dim == 4096) {
    by_hidden(std::integral_constant<int, 4096>{});
  } else if (hidden_dim == 3072) {
    by_hidden(std::integral_constant<int, 3072>{});
  } else {
    return;
  }
}

}  // namespace iHC
}  // namespace hpc
