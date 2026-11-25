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

// Pack independent elementwise outputs, preserving each output's FP32 FMA
// semantics. CuTe provides scalar fallbacks on architectures before SM100.
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

template <bool Aligned32 = false>
__device__ __forceinline__ vec_t<float, 8> ihc_load_weight8(const float* p) {
  vec_t<float, 8> v;
#if defined(CUTE_ARCH_LOAD256_SM100A_ENABLED)
  if constexpr (Aligned32) {
    asm volatile("ld.global.v8.f32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8];"
                 : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]), "=f"(v[4]), "=f"(v[5]),
                   "=f"(v[6]), "=f"(v[7])
                 : "l"(p));
    return v;
  }
#endif
  *reinterpret_cast<vec_t<float, 4>*>(&v[0]) = load<float, 4>(p);
  *reinterpret_cast<vec_t<float, 4>*>(&v[4]) = load<float, 4>(p + 4);
  return v;
}

// Accumulate two projection rows in FP32, reusing each input vector.
template <int N>
__device__ __forceinline__ void ihc_dot_pair(float& acc0, float& acc1, const vec_t<float, N>& x,
                                             const vec_t<float, N>& w0, const vec_t<float, N>& w1) {
  float2 acc = make_float2(acc0, acc1);
#pragma unroll
  for (int j = 0; j < N; ++j) {
    acc.x += x[j] * w0[j];
    acc.y += x[j] * w1[j];
  }
  acc0 = acc.x;
  acc1 = acc.y;
}

template <int HC, int D, int E, int Rows, int Threads, int ThreadsPerRow, bool FuseNorm,
          bool CastBfloat>
__device__ __forceinline__ void ihc_mix_epilogue(__nv_bfloat16* out, const __nv_bfloat16* x,
                                                 const float* gates, int row_base, int valid_rows,
                                                 const __nv_bfloat16* gamma, float eps,
                                                 float* norm_partial) {
  static_assert(ThreadsPerRow % 32 == 0 && Threads % ThreadsPerRow == 0,
                "each row group must contain whole warps");
  static_assert(D % (ThreadsPerRow * E) == 0, "row span must divide hidden size");
  constexpr int Groups = Threads / ThreadsPerRow;
  constexpr int WarpsPerRow = ThreadsPerRow / 32;
  constexpr int Chunks = D / (ThreadsPerRow * E);
  const int group = threadIdx.x / ThreadsPerRow;
  const int tid = threadIdx.x % ThreadsPerRow;

#pragma unroll 1
  for (int wave = 0; wave < (Rows + Groups - 1) / Groups; ++wave) {
    const int r = wave * Groups + group;
    const bool valid = r < valid_rows;
    using Saved = std::conditional_t<FuseNorm && CastBfloat, __nv_bfloat16, float>;
    vec_t<Saved, E> saved[Chunks];
    float sqsum = 0.f;
    if (valid) {
      const auto h = load<float, HC>(gates + r * HC);
      const auto* xr = x + static_cast<int64_t>(row_base + r) * HC * D;
      auto* yr = out + static_cast<int64_t>(row_base + r) * D;
#pragma unroll
      for (int ch = 0; ch < Chunks; ++ch) {
        const int col = (ch * ThreadsPerRow + tid) * E;
        vec_t<float, E> y;
#pragma unroll
        for (int j = 0; j < E; ++j) {
          y[j] = 0.f;
        }
        auto* y2 = reinterpret_cast<float2*>(&y[0]);
#pragma unroll
        for (int c = 0; c < HC; ++c) {
          const auto v = to<float>(load<__nv_bfloat16, E>(xr + c * D + col));
          const auto* v2 = reinterpret_cast<const float2*>(&v[0]);
#pragma unroll
          for (int j = 0; j < E / 2; ++j) {
            y2[j] = ihc_fma2(make_float2(h[c], h[c]), v2[j], y2[j]);
          }
        }
        if constexpr (FuseNorm) {
          if constexpr (CastBfloat) {
            saved[ch] = to<__nv_bfloat16>(y);
            y = to<float>(saved[ch]);
          } else {
            saved[ch] = y;
          }
#pragma unroll
          for (int j = 0; j < E; ++j) {
            sqsum += y[j] * y[j];
          }
        } else {
          store(yr + col, to<__nv_bfloat16>(y));
        }
      }
    }

    if constexpr (FuseNorm) {
      const float v = warp_reduce_sum_xor(sqsum);
      if ((tid % 32) == 0 && r < Rows) {
        norm_partial[r * WarpsPerRow + tid / 32] = v;
      }
      __syncthreads();
      if (valid) {
        float total = 0.f;
#pragma unroll
        for (int w = 0; w < WarpsPerRow; ++w) {
          total += norm_partial[r * WarpsPerRow + w];
        }
        const float scale = rsqrtf_ftz(total * (1.f / D) + eps);
        auto* yr = out + static_cast<int64_t>(row_base + r) * D;
#pragma unroll
        for (int ch = 0; ch < Chunks; ++ch) {
          const int col = (ch * ThreadsPerRow + tid) * E;
          const auto g = to<float>(load<__nv_bfloat16, E>(gamma + col));
          auto y = to<float>(saved[ch]);
          auto* y2 = reinterpret_cast<float2*>(&y[0]);
          const auto* g2 = reinterpret_cast<const float2*>(&g[0]);
#pragma unroll
          for (int j = 0; j < E / 2; ++j) {
            y2[j] = ihc_mul2(ihc_mul2(y2[j], make_float2(scale, scale)), g2[j]);
          }
          store(yr + col, to<__nv_bfloat16>(y));
        }
      }
    }
  }
}

namespace kernels {

constexpr int ihc_post_pre_cluster_blocks(int hidden_dim) { return hidden_dim == 3072 ? 6 : 8; }

constexpr int ihc_post_pre_umma_blocks(int hidden_dim) { return hidden_dim == 3072 ? 6 : 8; }

template <int D, int Blocks, bool Norm, bool Cast, bool PDL = false>
__device__ __forceinline__ void ihc_post_pre_cluster_device(
    __nv_bfloat16* y, __nv_bfloat16* z, float* hp, const __nv_bfloat16* xa,
    const __nv_bfloat16* residual, const float* hi, const float* w, const float* scale,
    const float* bias, int m, float eps, float hc_eps, float magnitude, const __nv_bfloat16* gamma,
    float rms_eps) {
  constexpr int E = 8, T = D / (Blocks * E), Slice = D / Blocks, Warps = T / 32;
  static_assert(D % (Blocks * E) == 0 && T % 32 == 0, "shards must contain whole warps");
  static_assert(Blocks <= 8, "use a portable cluster size");
  const auto cluster = cooperative_groups::this_cluster();
  const int rank = cluster.block_rank(), row = blockIdx.x / Blocks;
  const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32, col = rank * Slice + tid * E;
  struct Smem {
    float partial[9][Warps];
    float sums[9], gates[4], norm[Warps], normsum, normscale;
  };
  __shared__ Smem s;
  if constexpr (PDL) cudaGridDependencySynchronize();
  float dot[8] = {}, sq = 0.f;
  vec_t<__nv_bfloat16, E> cache[4];
  const auto a = to<float>(load<__nv_bfloat16, E>(xa + static_cast<int64_t>(row) * D + col));
  const auto h = load<float, 4>(hi + row * 4);
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const int64_t flat = static_cast<int64_t>(row) * 4 * D + c * D + col;
    const auto b = to<float>(load<__nv_bfloat16, E>(residual + flat));
    vec_t<float, E> v;
    auto* v2 = reinterpret_cast<float2*>(&v[0]);
    const auto* a2 = reinterpret_cast<const float2*>(&a[0]);
    const auto* b2 = reinterpret_cast<const float2*>(&b[0]);
#pragma unroll
    for (int j = 0; j < E / 2; ++j) v2[j] = ihc_fma2(make_float2(h[c], h[c]), a2[j], b2[j]);
    const auto bf = to<__nv_bfloat16>(v);
    store(y + flat, bf);
    cache[c] = bf;
    v = to<float>(bf);
#pragma unroll
    for (int j = 0; j < E; ++j) sq += v[j] * v[j];
#pragma unroll
    for (int p = 0; p < 8; ++p) {
      vec_t<float, E> weights;
      const float* wp = w + static_cast<int64_t>(p) * 4 * D + c * D + col;
      *reinterpret_cast<vec_t<float, 4>*>(&weights[0]) = load<float, 4>(wp);
      *reinterpret_cast<vec_t<float, 4>*>(&weights[4]) = load<float, 4>(wp + 4);
#pragma unroll
      for (int j = 0; j < E; ++j) dot[p] += v[j] * weights[j];
    }
  }
#pragma unroll
  for (int p = 0; p < 8; ++p) {
    const float v = warp_reduce_sum_xor(dot[p]);
    if (lane == 0) s.partial[p][warp] = v;
  }
  const float v = warp_reduce_sum_xor(sq);
  if (lane == 0) s.partial[8][warp] = v;
  __syncthreads();
  if (tid < 9) {
    float total = 0;
#pragma unroll
    for (int k = 0; k < Warps; ++k) total += s.partial[tid][k];
    s.sums[tid] = total;
  }
  // Publish all projection and input-norm partials before any DSM reads.
  cluster.sync();
  if (tid < 4) {
    float pre = 0, post = 0, total = 0;
#pragma unroll
    for (int k = 0; k < Blocks; ++k) {
      const auto* peer = cluster.map_shared_rank(&s, k);
      pre += peer->sums[tid];
      post += peer->sums[tid + 4];
      total += peer->sums[8];
    }
    const float r = rsqrtf_ftz(total * (1.f / (4 * D)) + eps);
    s.gates[tid] = sigmoid(scale[0] * (pre * r) + bias[tid]) + hc_eps;
    if (rank == 0)
      hp[row * 4 + tid] = magnitude * sigmoid(scale[1] * (post * r) + bias[tid + 4]) + hc_eps;
  }
  __syncthreads();
  vec_t<float, E> out;
#pragma unroll
  for (int j = 0; j < E; ++j) out[j] = 0;
  auto* o2 = reinterpret_cast<float2*>(&out[0]);
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const auto in = to<float>(cache[c]);
    const auto* i2 = reinterpret_cast<const float2*>(&in[0]);
#pragma unroll
    for (int j = 0; j < E / 2; ++j)
      o2[j] = ihc_fma2(make_float2(s.gates[c], s.gates[c]), i2[j], o2[j]);
  }
  if constexpr (Norm) {
    if constexpr (Cast) out = to<float>(to<__nv_bfloat16>(out));
    float osq = 0;
#pragma unroll
    for (int j = 0; j < E; ++j) osq += out[j] * out[j];
    const float t = warp_reduce_sum_xor(osq);
    if (lane == 0) s.norm[warp] = t;
    __syncthreads();
    if (tid == 0) {
      float total = 0;
#pragma unroll
      for (int k = 0; k < Warps; ++k) total += s.norm[k];
      s.normsum = total;
    }
    // Publish output-norm partials; gate reads have already completed.
    cluster.sync();
    if (tid == 0) {
      float total = 0;
#pragma unroll
      for (int k = 0; k < Blocks; ++k) total += cluster.map_shared_rank(&s, k)->normsum;
      s.normscale = rsqrtf_ftz(total * (1.f / D) + rms_eps);
    }
    __syncthreads();
    // Retire remote reads while gamma loads and final stores are in flight.
    cluster.barrier_arrive();
    const auto g = to<float>(load<__nv_bfloat16, E>(gamma + col));
    const auto* g2 = reinterpret_cast<const float2*>(&g[0]);
#pragma unroll
    for (int j = 0; j < E / 2; ++j)
      o2[j] = ihc_mul2(ihc_mul2(o2[j], make_float2(s.normscale, s.normscale)), g2[j]);
  } else {
    cluster.barrier_arrive();
  }
  store(z + static_cast<int64_t>(row) * D + col, to<__nv_bfloat16>(out));
  // Remote DSM readers must finish before any owning CTA exits.
  cluster.barrier_wait();
  if constexpr (PDL) cudaTriggerProgrammaticLaunchCompletion();
}

template <int D, int T, bool Norm, bool Cast, bool PDL = false, bool Aligned32 = false,
          bool PairFma = false>
__device__ __forceinline__ void ihc_post_pre_retained_device(
    __nv_bfloat16* y, __nv_bfloat16* z, float* hp, const __nv_bfloat16* xa,
    const __nv_bfloat16* res, const float* hi, const float* w, const float* scale,
    const float* bias, int m, float eps, float hc_eps, float mag, const __nv_bfloat16* gamma,
    float rms_eps) {
  constexpr int E = 8, C = D / (T * E), Warps = T / 32;
  static_assert(T % 32 == 0 && D % (T * E) == 0, "whole warps and complete column tiles");
  const int row = blockIdx.x, tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
  if constexpr (PDL) cudaGridDependencySynchronize();
  __shared__ float part[9][Warps];
  __shared__ float gates[4], norm[Warps], rn;
  vec_t<__nv_bfloat16, E> cache[C][4];
  float dot[8] = {}, sq = 0;
  auto h = load<float, 4>(hi + row * 4);
#pragma unroll
  for (int ch = 0; ch < C; ++ch) {
    int col = (ch * T + tid) * E;
    auto a = to<float>(load<__nv_bfloat16, E>(xa + static_cast<int64_t>(row) * D + col));
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      auto b =
          to<float>(load<__nv_bfloat16, E>(res + static_cast<int64_t>(row) * 4 * D + i * D + col));
      vec_t<float, E> v;
      auto* v2 = reinterpret_cast<float2*>(&v[0]);
      const auto* a2 = reinterpret_cast<const float2*>(&a[0]);
      const auto* b2 = reinterpret_cast<const float2*>(&b[0]);
#pragma unroll
      for (int j = 0; j < E / 2; ++j) v2[j] = ihc_fma2(make_float2(h[i], h[i]), a2[j], b2[j]);
      auto bf = to<__nv_bfloat16>(v);
      cache[ch][i] = bf;
      store(y + static_cast<int64_t>(row) * 4 * D + i * D + col, bf);
      v = to<float>(bf);
#pragma unroll
      for (int j = 0; j < E; ++j) sq += v[j] * v[j];
      if constexpr (PairFma) {
#pragma unroll
        for (int p = 0; p < 8; p += 2) {
          const auto* wp = w + static_cast<int64_t>(p) * 4 * D + i * D + col;
          const auto w0 = ihc_load_weight8<Aligned32>(wp);
          const auto w1 = ihc_load_weight8<Aligned32>(wp + 4 * D);
          auto acc = make_float2(dot[p], dot[p + 1]);
#pragma unroll
          for (int j = 0; j < E; ++j)
            acc = ihc_fma2(make_float2(v[j], v[j]), make_float2(w0[j], w1[j]), acc);
          dot[p] = acc.x;
          dot[p + 1] = acc.y;
        }
      } else {
#pragma unroll
        for (int p = 0; p < 8; ++p) {
          vec_t<float, E> ww;
          auto wp = w + static_cast<int64_t>(p) * 4 * D + i * D + col;
          ww = ihc_load_weight8<Aligned32>(wp);
#pragma unroll
          for (int j = 0; j < E; ++j) dot[p] += v[j] * ww[j];
        }
      }
    }
  }
#pragma unroll
  for (int p = 0; p < 8; ++p) {
    float v = warp_reduce_sum_xor(dot[p]);
    if (lane == 0) part[p][warp] = v;
  }
  float v = warp_reduce_sum_xor(sq);
  if (lane == 0) part[8][warp] = v;
  __syncthreads();
  if (tid < 4) {
    float pre = 0, post = 0, total = 0;
#pragma unroll
    for (int k = 0; k < Warps; ++k) {
      pre += part[tid][k];
      post += part[tid + 4][k];
      total += part[8][k];
    }
    float r = rsqrtf_ftz(total * (1.f / (4 * D)) + eps);
    gates[tid] = sigmoid(scale[0] * (pre * r) + bias[tid]) + hc_eps;
    hp[row * 4 + tid] = mag * sigmoid(scale[1] * (post * r) + bias[tid + 4]) + hc_eps;
  }
  __syncthreads();
  using Saved = std::conditional_t<Norm && Cast, __nv_bfloat16, float>;
  vec_t<Saved, E> saved[C];
  float osq = 0;
#pragma unroll
  for (int ch = 0; ch < C; ++ch) {
    vec_t<float, E> out;
    for (int j = 0; j < E; ++j) out[j] = 0;
    auto* o2 = reinterpret_cast<float2*>(&out[0]);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      auto in = to<float>(cache[ch][i]);
      auto* i2 = reinterpret_cast<const float2*>(&in[0]);
#pragma unroll
      for (int j = 0; j < E / 2; ++j)
        o2[j] = ihc_fma2(make_float2(gates[i], gates[i]), i2[j], o2[j]);
    }
    if constexpr (Norm) {
      if constexpr (Cast) {
        saved[ch] = to<__nv_bfloat16>(out);
        out = to<float>(saved[ch]);
      } else {
        saved[ch] = out;
      }
#pragma unroll
      for (int j = 0; j < E; ++j) osq += out[j] * out[j];
    } else {
      store(z + static_cast<int64_t>(row) * D + (ch * T + tid) * E, to<__nv_bfloat16>(out));
    }
  }
  if constexpr (Norm) {
    float ss = warp_reduce_sum_xor(osq);
    if (lane == 0) norm[warp] = ss;
    __syncthreads();
    if (tid == 0) {
      float total = 0;
      for (int k = 0; k < Warps; ++k) total += norm[k];
      rn = rsqrtf_ftz(total * (1.f / D) + rms_eps);
    }
    __syncthreads();
#pragma unroll
    for (int ch = 0; ch < C; ++ch) {
      int col = (ch * T + tid) * E;
      auto g = to<float>(load<__nv_bfloat16, E>(gamma + col));
      auto out = to<float>(saved[ch]);
      auto* o2 = reinterpret_cast<float2*>(&out[0]);
      auto* g2 = reinterpret_cast<const float2*>(&g[0]);
#pragma unroll
      for (int j = 0; j < E / 2; ++j) o2[j] = ihc_mul2(ihc_mul2(o2[j], make_float2(rn, rn)), g2[j]);
      store(z + static_cast<int64_t>(row) * D + col, to<__nv_bfloat16>(out));
    }
  }
  if constexpr (PDL) cudaTriggerProgrammaticLaunchCompletion();
}

template <int D, int Rows, int Threads, int EpilogueThreads, bool FuseNorm, bool CastBfloat,
          bool UsePDL = false, bool Aligned32 = false>
__device__ __forceinline__ void ihc_post_pre_rows_device(
    __nv_bfloat16* y, __nv_bfloat16* z, float* h_out, const __nv_bfloat16* xa,
    const __nv_bfloat16* residual, const float* h_in, const float* w, const float* hc_scale,
    const float* hc_base, int m, float norm_eps, float hc_eps, float magnitude,
    const __nv_bfloat16* gamma, float rms_eps) {
  constexpr int HC = 4;
  constexpr int Proj = 8;
  constexpr int E = 8;
  constexpr int Warps = Threads / 32;
  constexpr int Chunks = D / (Threads * E);
  static_assert(D % (Threads * E) == 0, "block span must divide hidden size");
  if constexpr (UsePDL) {
    cudaGridDependencySynchronize();
  }
  const int row_base = blockIdx.x * Rows;
  const int valid_rows = min(Rows, m - row_base);
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  __shared__ float partial[Rows][Proj + 1][Warps];
  __shared__ __align__(16) float h_pre[Rows][HC];
  __shared__ float norm_partial[Rows * (EpilogueThreads / 32)];
  float dot[Rows][Proj];
  float sqsum[Rows];
#pragma unroll
  for (int r = 0; r < Rows; ++r) {
    sqsum[r] = 0.f;
#pragma unroll
    for (int c = 0; c < Proj; ++c) {
      dot[r][c] = 0.f;
    }
  }

#pragma unroll 1
  for (int ch = 0; ch < Chunks; ++ch) {
    const int col = (ch * Threads + threadIdx.x) * E;
    // Keep xa packed while all four residual channels are processed.
    vec_t<__nv_bfloat16, E> reg_xa[Rows];
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
      if (r < valid_rows) {
        reg_xa[r] = load<__nv_bfloat16, E>(xa + static_cast<int64_t>(row_base + r) * D + col);
      }
    }
#pragma unroll 1
    for (int i = 0; i < HC; ++i) {
      const int flat = i * D + col;
      vec_t<float, E> reg_y[Rows];
#pragma unroll
      for (int r = 0; r < Rows; ++r) {
        if (r < valid_rows) {
          const int64_t row = row_base + r;
          const auto a = to<float>(reg_xa[r]);
          const auto b = to<float>(load<__nv_bfloat16, E>(residual + row * HC * D + flat));
          const float h = h_in[row * HC + i];
          const auto* a2 = reinterpret_cast<const float2*>(&a[0]);
          const auto* b2 = reinterpret_cast<const float2*>(&b[0]);
          auto* y2 = reinterpret_cast<float2*>(&reg_y[r][0]);
#pragma unroll
          for (int j = 0; j < E / 2; ++j) {
            y2[j] = ihc_fma2(make_float2(h, h), a2[j], b2[j]);
          }
          const auto bf = to<__nv_bfloat16>(reg_y[r]);
          store(y + row * HC * D + flat, bf);
          reg_y[r] = to<float>(bf);
#pragma unroll
          for (int j = 0; j < E; ++j) {
            sqsum[r] += reg_y[r][j] * reg_y[r][j];
          }
        }
      }
#pragma unroll
      for (int c = 0; c < Proj; c += 2) {
        vec_t<float, E> w0, w1;
        const float* p0 = w + static_cast<int64_t>(c) * HC * D + flat;
        const float* p1 = p0 + HC * D;
        if constexpr (D != 4096) {
          w0 = ihc_load_weight8<Aligned32>(p0);
          w1 = ihc_load_weight8<Aligned32>(p1);
        } else {
          *reinterpret_cast<vec_t<float, E / 2>*>(&w0[0]) = load<float, E / 2>(p0);
          *reinterpret_cast<vec_t<float, E / 2>*>(&w0[E / 2]) = load<float, E / 2>(p0 + E / 2);
          *reinterpret_cast<vec_t<float, E / 2>*>(&w1[0]) = load<float, E / 2>(p1);
          *reinterpret_cast<vec_t<float, E / 2>*>(&w1[E / 2]) = load<float, E / 2>(p1 + E / 2);
        }
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
          if (r < valid_rows) {
            ihc_dot_pair<E>(dot[r][c], dot[r][c + 1], reg_y[r], w0, w1);
          }
        }
      }
    }
  }

#pragma unroll
  for (int r = 0; r < Rows; ++r) {
#pragma unroll
    for (int c = 0; c < Proj; ++c) {
      const float v = warp_reduce_sum_xor(dot[r][c]);
      if (lane == 0) {
        partial[r][c][warp] = v;
      }
    }
    const float v = warp_reduce_sum_xor(sqsum[r]);
    if (lane == 0) {
      partial[r][Proj][warp] = v;
    }
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < Rows * (Proj + 1); idx += Threads) {
    const int r = idx / (Proj + 1);
    const int c = idx % (Proj + 1);
    float total = 0.f;
#pragma unroll
    for (int k = 0; k < Warps; ++k) {
      total += partial[r][c][k];
    }
    partial[r][c][0] = total;
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < valid_rows * HC; idx += Threads) {
    const int r = idx / HC;
    const int i = idx % HC;
    const float scale = rsqrtf_ftz(partial[r][Proj][0] * (1.f / (HC * D)) + norm_eps);
    h_pre[r][i] = sigmoid(hc_scale[0] * (partial[r][i][0] * scale) + hc_base[i]) + hc_eps;
    h_out[(row_base + r) * HC + i] =
        magnitude * sigmoid(hc_scale[1] * (partial[r][HC + i][0] * scale) + hc_base[HC + i]) +
        hc_eps;
  }
  __syncthreads();
  ihc_mix_epilogue<HC, D, E, Rows, Threads, EpilogueThreads, FuseNorm, CastBfloat>(
      z, y, &h_pre[0][0], row_base, valid_rows, gamma, rms_eps, norm_partial);
  if constexpr (UsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

// Projection/sqrsum accumulation for one block's rows.
// kActiveRows is a compile-time constant so the compiler can unroll and hoist loads.
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

// kThreadsPerBlock is templated so __launch_bounds__ matches the launched block.
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
  const int icol_stride = gridDim.y * blockDim.x * kElementsPerThread;
  const int num_iter = (kHiddenDim - icol + icol_stride - 1) / icol_stride;

  const __nv_bfloat16* x_row_ptr = x_ptr + irow * kHiddenDim + icol;
  const __nv_bfloat16* residual_row_ptr = residual_ptr + irow * kHCMult * kHiddenDim + icol;
  __nv_bfloat16* output_row_ptr = output_ptr + irow * kHCMult * kHiddenDim + icol;

  for (int iter = 0; iter < num_iter; ++iter) {
    auto reg_x = to<float>(load<__nv_bfloat16, kElementsPerThread>(x_row_ptr + iter * icol_stride));

#pragma unroll
    for (int i = 0; i < kHCMult; ++i) {
      auto reg_residual = to<float>(load<__nv_bfloat16, kElementsPerThread>(
          residual_row_ptr + i * kHiddenDim + iter * icol_stride));

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

      store(output_row_ptr + i * kHiddenDim + iter * icol_stride, to<__nv_bfloat16>(reg_output));
    }
  }

  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

// One global entry completes post, pre and optional RMSNorm. M is uniform across
// the grid; each branch calls an inline device helper with its own fixed layout.
template <int kHCMult, int kHiddenDim, int kElementsPerThread, int kThreadsPerBlock,
          bool kUsePDL = false, bool kFuseRmsNorm = false, bool kCastBfloatForNorm = false,
          bool kWeightsAligned = true>
__global__ void __launch_bounds__(HPC_TARGET_ARCH == 90 ? kThreadsPerBlock : 768)
    fuse_ihc_post_pre_kernel(__nv_bfloat16* y_ptr, __nv_bfloat16* z_ptr, float* H_post_out_ptr,
                             const __nv_bfloat16* xa_ptr, const __nv_bfloat16* residual_ptr,
                             const float* H_post_in_ptr, const float* w_ptr,
                             const float* hc_scale_ptr, const float* hc_base_ptr, int num_batch,
                             float norm_eps, float hc_eps, float magnitude,
                             const __nv_bfloat16* rms_weight_ptr = nullptr, float rms_eps = 0.f) {
  static_assert(kHCMult == 4 && kElementsPerThread == 8, "supported fused layout");
  static_assert(kHiddenDim == 3072 || kHiddenDim == 4096 || kHiddenDim == 6144,
                "supported hidden dimensions");
#if HPC_TARGET_ARCH == 90
  ihc_post_pre_rows_device<kHiddenDim, 1, kThreadsPerBlock, kThreadsPerBlock, kFuseRmsNorm,
                           kCastBfloatForNorm, kUsePDL, kWeightsAligned>(
      y_ptr, z_ptr, H_post_out_ptr, xa_ptr, residual_ptr, H_post_in_ptr, w_ptr, hc_scale_ptr,
      hc_base_ptr, num_batch, norm_eps, hc_eps, magnitude, rms_weight_ptr, rms_eps);
#else
  if constexpr (kHiddenDim == 6144) {
    if (num_batch < 80) {
      ihc_post_pre_cluster_device<kHiddenDim, 8, kFuseRmsNorm, kCastBfloatForNorm, kUsePDL>(
          y_ptr, z_ptr, H_post_out_ptr, xa_ptr, residual_ptr, H_post_in_ptr, w_ptr, hc_scale_ptr,
          hc_base_ptr, num_batch, norm_eps, hc_eps, magnitude, rms_weight_ptr, rms_eps);
    } else if (num_batch <= 296) {
      ihc_post_pre_retained_device<kHiddenDim, 768, kFuseRmsNorm, kCastBfloatForNorm, kUsePDL,
                                   kWeightsAligned>(
          y_ptr, z_ptr, H_post_out_ptr, xa_ptr, residual_ptr, H_post_in_ptr, w_ptr, hc_scale_ptr,
          hc_base_ptr, num_batch, norm_eps, hc_eps, magnitude, rms_weight_ptr, rms_eps);
    } else {
      ihc_post_pre_rows_device<kHiddenDim, 2, 384, 384, kFuseRmsNorm, kCastBfloatForNorm, kUsePDL,
                               kWeightsAligned>(
          y_ptr, z_ptr, H_post_out_ptr, xa_ptr, residual_ptr, H_post_in_ptr, w_ptr, hc_scale_ptr,
          hc_base_ptr, num_batch, norm_eps, hc_eps, magnitude, rms_weight_ptr, rms_eps);
    }
  } else {
    if (num_batch <= 128) {
      ihc_post_pre_cluster_device<kHiddenDim, ihc_post_pre_cluster_blocks(kHiddenDim), kFuseRmsNorm,
                                  kCastBfloatForNorm, kUsePDL>(
          y_ptr, z_ptr, H_post_out_ptr, xa_ptr, residual_ptr, H_post_in_ptr, w_ptr, hc_scale_ptr,
          hc_base_ptr, num_batch, norm_eps, hc_eps, magnitude, rms_weight_ptr, rms_eps);
    } else if (num_batch < 297) {
      ihc_post_pre_rows_device<kHiddenDim, 1, kThreadsPerBlock, kThreadsPerBlock, kFuseRmsNorm,
                               kCastBfloatForNorm, kUsePDL, kWeightsAligned>(
          y_ptr, z_ptr, H_post_out_ptr, xa_ptr, residual_ptr, H_post_in_ptr, w_ptr, hc_scale_ptr,
          hc_base_ptr, num_batch, norm_eps, hc_eps, magnitude, rms_weight_ptr, rms_eps);
    } else {
      ihc_post_pre_rows_device<kHiddenDim, 2, kThreadsPerBlock, kThreadsPerBlock, kFuseRmsNorm,
                               kCastBfloatForNorm, kUsePDL, kWeightsAligned>(
          y_ptr, z_ptr, H_post_out_ptr, xa_ptr, residual_ptr, H_post_in_ptr, w_ptr, hc_scale_ptr,
          hc_base_ptr, num_batch, norm_eps, hc_eps, magnitude, rms_weight_ptr, rms_eps);
    }
  }
#endif
}

#if HPC_TARGET_ARCH != 90
__device__ __forceinline__ void ihc_tmem_fence_before_sync() {
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void ihc_tmem_fence_after_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

template <int BK, int Parts>
struct IhcPostPreUmmaStorage {
  using LA = decltype(cute::tile_to_shape(cute::UMMA::Layout_K_SW128_Atom<cute::bfloat16_t>{},
                                          cute::Shape<cute::_64, cute::Int<BK>>{}));
  using LB = decltype(cute::tile_to_shape(cute::UMMA::Layout_K_SW128_Atom<cute::bfloat16_t>{},
                                          cute::Shape<cute::Int<8 * Parts>, cute::Int<BK>>{}));
  __align__(1024) __nv_bfloat16 a[2][cute::cosize_v<LA>];
  __align__(1024) __nv_bfloat16 b[2][cute::cosize_v<LB>];
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
          bool kFuseRmsNorm, bool kCastBfloatForNorm, bool kWeightsAligned, int kAccumulatorGroups>
__global__ void __maxnreg__(64)
    fuse_ihc_post_pre_kernel(__nv_bfloat16* y, __nv_bfloat16* z, float* hp, const __nv_bfloat16* xa,
                             const __nv_bfloat16* residual, const float* hi, const float* w,
                             const float* scale, const float* bias, int m, float norm_eps,
                             float eps, float magnitude, const __nv_bfloat16* gamma,
                             float rms_eps) {
  static_assert((kHiddenDim == 3072 || kHiddenDim == 4096 || kHiddenDim == 6144) && kHCMult == 4 &&
                    kElementsPerThread == 8 && kThreadsPerBlock == 256 && kWeightsAligned,
                "supported UMMA layout");
  static_assert(!kFuseRmsNorm || kCastBfloatForNorm, "RMSNorm requires BF16 intermediate");
  static_assert(kAccumulatorGroups == 8, "supported TMEM layout");
  constexpr int D = kHiddenDim, BK = 128, Parts = 3, Groups = kAccumulatorGroups;
  constexpr int Blocks = ihc_post_pre_umma_blocks(D);
  constexpr bool Norm = kFuseRmsNorm;
  constexpr int T = 256, E = 8, Vecs = BK / E, RowsPerThread = 64 * Vecs / T;
  constexpr int Columns = 128;
  using Storage = IhcPostPreUmmaStorage<BK, Parts>;
  using LA = typename Storage::LA;
  using LB = typename Storage::LB;
  using MMA = cute::SM100_MMA_F16BF16_SS<cute::bfloat16_t, cute::bfloat16_t, float, 64, 8 * Parts,
                                         cute::UMMA::Major::K, cute::UMMA::Major::K>;
  extern __shared__ __align__(1024) unsigned char storage[];
  auto& s = *reinterpret_cast<Storage*>(storage);
  auto cluster = cooperative_groups::this_cluster();
  const int rank = cluster.block_rank();
  constexpr int Slice = D / Blocks;
  static_assert(Slice % BK == 0 && Slice % (32 * E) == 0, "slice must hold whole K tiles");
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, base = (blockIdx.x / Blocks) * 64;
  const int valid = min(64, m - base);
  if constexpr (kUsePDL) cudaGridDependencySynchronize();
  if (tid < 32) {
    cute::TMEM::Allocator1Sm alloc;
    alloc.allocate(Columns, &s.tmem);
    alloc.release_allocation_lock();
    ihc_tmem_fence_before_sync();
  }
  if (tid == 0) {
    cutlass::arch::ClusterBarrier::init(&s.barrier[0], 1);
    cutlass::arch::ClusterBarrier::init(&s.barrier[1], 1);
  }
  float sq[RowsPerThread] = {};
  __syncthreads();
  ihc_tmem_fence_after_sync();
  using LDA = cute::ComposedLayout<
      cute::Swizzle<3, 4, 3>, cute::smem_ptr_flag_bits<16>,
      cute::Layout<cute::Shape<cute::_64, cute::_64>, cute::Stride<cute::_64, cute::_1>>>;
  using LDB = cute::ComposedLayout<cute::Swizzle<3, 4, 3>, cute::smem_ptr_flag_bits<16>,
                                   cute::Layout<cute::Shape<cute::Int<8 * Parts>, cute::_64>,
                                                cute::Stride<cute::_64, cute::_1>>>;
  auto ta =
      cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<cute::bfloat16_t*>(s.a[0])), LDA{});
  auto tb =
      cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<cute::bfloat16_t*>(s.b[0])), LDB{});
  const uint64_t ad = cute::UMMA::make_umma_desc<cute::UMMA::Major::K>(ta);
  const uint64_t bd = cute::UMMA::make_umma_desc<cute::UMMA::Major::K>(tb);
  const auto id =
      cute::UMMA::make_runtime_instr_desc<cute::bfloat16_t, cute::bfloat16_t, float, 64, 8 * Parts,
                                          cute::UMMA::Major::K, cute::UMMA::Major::K>();
  int phase[2] = {0, 0};
  constexpr int TilesPerChannel = Slice / BK;
  // Keep each XA tile hot while visiting all channels for D6144.
#pragma unroll 1
  for (int outer = 0; outer < (D == 6144 ? TilesPerChannel : 4); ++outer) {
#pragma unroll 1
    for (int inner = 0; inner < (D == 6144 ? 4 : TilesPerChannel); ++inner) {
      const int ch = D == 6144 ? inner : outer;
      const int k0 = (D == 6144 ? outer : inner) * BK;
      const int tile = D == 6144 ? outer * 4 + inner : outer * TilesPerChannel + inner;
      const int slot = tile & 1;
      if (tile >= 2) {
        if (tid < 32) cutlass::arch::ClusterBarrier::wait(&s.barrier[slot], phase[slot]);
        phase[slot] ^= 1;
        __syncthreads();
      }
#pragma unroll
      for (int j = 0; j < RowsPerThread; ++j) {
        const int idx = j * T + tid, row = idx / Vecs, k = (idx % Vecs) * E,
                  col = rank * Slice + k0 + k;
        vec_t<__nv_bfloat16, E> bf{};
        if (row < valid) {
          const auto a = to<float>(load<__nv_bfloat16, E>(xa + int64_t(base + row) * D + col));
          const auto b = to<float>(
              load<__nv_bfloat16, E>(residual + int64_t(base + row) * 4 * D + ch * D + col));
          const float h = hi[(base + row) * 4 + ch];
          vec_t<float, E> v;
          auto* v2 = reinterpret_cast<float2*>(&v[0]);
          const auto* a2 = reinterpret_cast<const float2*>(&a[0]);
          const auto* b2 = reinterpret_cast<const float2*>(&b[0]);
#pragma unroll
          for (int q = 0; q < E / 2; ++q) v2[q] = ihc_fma2(make_float2(h, h), a2[q], b2[q]);
          bf = to<__nv_bfloat16>(v);
          store(y + int64_t(base + row) * 4 * D + ch * D + col, bf);
          v = to<float>(bf);
#pragma unroll
          for (int q = 0; q < E; ++q) sq[j] += v[q] * v[q];
        }
        store(s.a[slot] + cute::Swizzle<3, 4, 3>{}(LA{}.layout_b()(row, k) * 2) / 2, bf);
      }
      // Preserve the FP32 weight residuals in separate BF16 projection terms.
      for (int idx = tid; idx < 8 * Vecs; idx += T) {
        const int p = idx / Vecs, k = (idx % Vecs) * E;
        auto ww = ihc_load_weight8<true>(w + int64_t(p) * 4 * D + ch * D + rank * Slice + k0 + k);
#pragma unroll
        for (int part = 0; part < Parts; ++part) {
          const auto bf = to<__nv_bfloat16>(ww);
          store(s.b[slot] + cute::Swizzle<3, 4, 3>{}(LB{}.layout_b()(part * 8 + p, k) * 2) / 2, bf);
          const auto rounded = to<float>(bf);
#pragma unroll
          for (int q = 0; q < E; ++q) ww[q] -= rounded[q];
        }
      }
      cutlass::arch::fence_view_async_shared();
      __syncthreads();
      if (tid < 32) {
#pragma unroll
        for (int k = 0; k < BK; k += 16)
          MMA::fma(ad + (sizeof(s.a[0]) / 16) * slot + LA{}.layout_b()(0, k) / 8,
                   bd + (sizeof(s.b[0]) / 16) * slot + LB{}.layout_b()(0, k) / 8,
                   s.tmem + ((tile % Groups) / 2) * (8 * Parts) + (((tile % Groups) & 1) << 20),
                   (tile >= Groups || k != 0), id);
        cutlass::arch::umma_arrive(&s.barrier[slot]);
      }
      __syncthreads();
    }
  }
  if (tid < 32) {
    for (int slot = 0; slot < 2; ++slot)
      cutlass::arch::ClusterBarrier::wait(&s.barrier[slot], phase[slot]);
    ihc_tmem_fence_before_sync();
  }
  __syncthreads();
  ihc_tmem_fence_after_sync();
#pragma unroll
  for (int j = 0; j < RowsPerThread; ++j) {
    constexpr int Width = Vecs < 32 ? Vecs : 32;
    float total = sq[j];
#pragma unroll
    for (int mask = Width / 2; mask > 0; mask /= 2)
      total += __shfl_xor_sync(0xffffffff, total, mask);
    if (tid % Vecs == 0) s.sq[(j * T + tid) / Vecs] = total;
    if constexpr (Vecs > 32) {
      if (tid % Vecs == 32) s.gate[(j * T + tid) / Vecs][0] = total;
    }
  }
  __syncthreads();
  if constexpr (Vecs > 32) {
    if (tid < 64) s.sq[tid] += s.gate[tid][0];
    __syncthreads();
  }
  if (tid < 128) {
    float projection[8] = {};
#pragma unroll
    for (int group = 0; group < Groups / 2; ++group) {
      float partial[8] = {};
#pragma unroll
      for (int part = Parts - 1; part >= 0; --part) {
        uint32_t v[8];
        cute::SM100_TMEM_LOAD_32dp32b8x::copy(
            s.tmem + (warp * 32 << 16) + group * (8 * Parts) + part * 8, v[0], v[1], v[2], v[3],
            v[4], v[5], v[6], v[7]);
        cutlass::arch::fence_view_async_tmem_load();
#pragma unroll
        for (int p = 0; p < 8; ++p) partial[p] += __uint_as_float(v[p]);
      }
#pragma unroll
      for (int p = 0; p < 8; ++p) projection[p] += partial[p];
    }
#pragma unroll
    for (int p = 0; p < 8; ++p) projection[p] += __shfl_xor_sync(0xffffffff, projection[p], 16);
    const int row = warp * 16 + lane;
    if (lane < 16) {
#pragma unroll
      for (int p = 0; p < 8; ++p) s.projection[row][p] = projection[p];
    }
    ihc_tmem_fence_before_sync();
  }
  __syncthreads();
  if (tid < 32) {
    ihc_tmem_fence_after_sync();
    cute::TMEM::Allocator1Sm alloc;
    alloc.free(s.tmem, Columns);
  }
  cluster.sync();
  if (rank == 0 && tid < valid) {
    float projection[8] = {}, total = 0;
#pragma unroll
    for (int b = 0; b < Blocks; ++b) {
      const auto* remote = cluster.map_shared_rank(&s, b);
      total += remote->sq[tid];
#pragma unroll
      for (int p = 0; p < 8; ++p) projection[p] += remote->projection[tid][p];
    }
    const float inv = rsqrtf_ftz(total * (1.f / (4 * D)) + norm_eps);
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      s.gate[tid][p] = sigmoid(scale[0] * (projection[p] * inv) + bias[p]) + eps;
      hp[(base + tid) * 4 + p] =
          magnitude * sigmoid(scale[1] * (projection[p + 4] * inv) + bias[p + 4]) + eps;
    }
  }
  cluster.sync();
  const auto* root = cluster.map_shared_rank(&s, 0);
#pragma unroll 1
  for (int row = warp; row < valid; row += 8) {
    constexpr int Chunks = Slice / (32 * E);
    const auto gates = load<float, 4>(&root->gate[row][0]);
    float total = 0;
#pragma unroll
    for (int chunk = 0; chunk < Chunks; ++chunk) {
      const int col = rank * Slice + (chunk * 32 + lane) * E;
      vec_t<float, E> v{};
      auto* v2 = reinterpret_cast<float2*>(&v[0]);
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        const auto in =
            to<float>(load<__nv_bfloat16, E>(y + int64_t(base + row) * 4 * D + p * D + col));
        const auto* in2 = reinterpret_cast<const float2*>(&in[0]);
#pragma unroll
        for (int q = 0; q < E / 2; ++q)
          v2[q] = ihc_fma2(make_float2(gates[p], gates[p]), in2[q], v2[q]);
      }
      const auto bf = to<__nv_bfloat162>(v);
      store(z + int64_t(base + row) * D + col, bf);
      if constexpr (Norm) {
        v = to<float>(bf);
#pragma unroll
        for (int q = 0; q < E; ++q) total += v[q] * v[q];
      }
    }
    if constexpr (Norm) {
      total = warp_reduce_sum_xor(total);
      if (lane == 0) s.normsum[row] = total;
    }
  }
  if constexpr (Norm) {
    cluster.sync();
    if (rank == 0 && tid < valid) {
      float total = 0;
#pragma unroll
      for (int b = 0; b < Blocks; ++b) total += cluster.map_shared_rank(&s, b)->normsum[tid];
      s.normsum[tid] = rsqrtf_ftz(total * (1.f / D) + rms_eps);
    }
    cluster.sync();
#pragma unroll 1
    for (int row = warp; row < valid; row += 8) {
      const float inv = root->normsum[row];
#pragma unroll
      for (int chunk = 0; chunk < Slice / (32 * E); ++chunk) {
        const int col = rank * Slice + (chunk * 32 + lane) * E;
        auto v = to<float>(load<__nv_bfloat16, E>(z + int64_t(base + row) * D + col));
        const auto g = to<float>(load<__nv_bfloat16, E>(gamma + col));
        auto* v2 = reinterpret_cast<float2*>(&v[0]);
        const auto* g2 = reinterpret_cast<const float2*>(&g[0]);
#pragma unroll
        for (int q = 0; q < E / 2; ++q)
          v2[q] = ihc_mul2(ihc_mul2(v2[q], make_float2(inv, inv)), g2[q]);
        store(z + int64_t(base + row) * D + col, to<__nv_bfloat16>(v));
      }
    }
  }
  cluster.sync();

  if constexpr (kUsePDL) cudaTriggerProgrammaticLaunchCompletion();
}

// WG0 produces rounded Y; WG1/2 project four outputs each. The shared Y
// remains live until the fused mixing and optional normalization complete.
struct IhcPostPreThreeWarpGroups {};

template <int D>
struct IhcPostPreThreeWarpStorage {
  __align__(16) __nv_bfloat16 y[1][4][D];
  float part[1][9][4];
  float gates[1][4];
  float norm[1][4];
  float rn[1];
};

template <int kHCMult, int kHiddenDim, int kElementsPerThread, int kThreadsPerBlock, bool kUsePDL,
          bool kFuseRmsNorm, bool kCastBfloatForNorm, bool kWeightsAligned, typename Layout>
__global__ void __launch_bounds__(384, kHiddenDim == 6144 ? 1 : 2)
    fuse_ihc_post_pre_kernel(__nv_bfloat16* y, __nv_bfloat16* z, float* hp, const __nv_bfloat16* xa,
                             const __nv_bfloat16* res, const float* hi, const float* w,
                             const float* scale, const float* bias, int m, float norm_eps,
                             float hc_eps, float magnitude, const __nv_bfloat16* gamma,
                             float rms_eps) {
  static_assert(std::is_same_v<Layout, IhcPostPreThreeWarpGroups>, "supported warp layout");
  static_assert(
      kHCMult == 4 && kElementsPerThread == 8 && kThreadsPerBlock == 384 && kWeightsAligned,
      "supported three-warp-group layout");
  static_assert(!kFuseRmsNorm || kCastBfloatForNorm, "RMSNorm requires BF16 intermediate");
  if constexpr (kHiddenDim == 6144 || kHiddenDim == 3072) {
    ihc_post_pre_retained_device<kHiddenDim, 384, kFuseRmsNorm, kCastBfloatForNorm, kUsePDL,
                                 kWeightsAligned, true>(
        y, z, hp, xa, res, hi, w, scale, bias, m, norm_eps, hc_eps, magnitude, gamma, rms_eps);
  } else {
    static_assert(kHiddenDim == 4096, "supported hidden dimension");
    constexpr int D = kHiddenDim, Rows = 1;
    constexpr bool Norm = kFuseRmsNorm;
    constexpr int T = 128, E = 8, C = D / (T * E);
    extern __shared__ __align__(16) unsigned char three_warp_storage[];
    auto& s = *reinterpret_cast<IhcPostPreThreeWarpStorage<D>*>(three_warp_storage);
    const int wg = threadIdx.x / T, tid = threadIdx.x % T;
    const int lane = tid % 32, warp = tid / 32;
    const int base = blockIdx.x * Rows;
    const int valid = min(Rows, m - base);
    if constexpr (kUsePDL) cudaGridDependencySynchronize();
    if (wg == 0) {
      float sq[Rows] = {};
#pragma unroll 1
      for (int ch = 0; ch < C; ++ch) {
        const int col = (ch * T + tid) * E;
        vec_t<float, E> a[Rows];
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
          if (r < valid) a[r] = to<float>(load<__nv_bfloat16, E>(xa + int64_t(base + r) * D + col));
        }
#pragma unroll 1
        for (int c = 0; c < 4; ++c) {
#pragma unroll
          for (int r = 0; r < Rows; ++r) {
            if (r < valid) {
              const int64_t flat = int64_t(base + r) * 4 * D + c * D + col;
              const auto b = to<float>(load<__nv_bfloat16, E>(res + flat));
              const float h = hi[(base + r) * 4 + c];
              vec_t<float, E> v;
              auto* v2 = reinterpret_cast<float2*>(&v[0]);
              const auto* a2 = reinterpret_cast<const float2*>(&a[r][0]);
              const auto* b2 = reinterpret_cast<const float2*>(&b[0]);
#pragma unroll
              for (int j = 0; j < E / 2; ++j) v2[j] = ihc_fma2(make_float2(h, h), a2[j], b2[j]);
              const auto bf = to<__nv_bfloat16>(v);
              store(y + flat, bf);
              store(&s.y[r][c][col], bf);
              const auto rounded = to<float>(bf);
#pragma unroll
              for (int j = 0; j < E; ++j) sq[r] += rounded[j] * rounded[j];
            }
          }
        }
        asm volatile("barrier.sync 1, 384;" ::: "memory");
      }
#pragma unroll
      for (int r = 0; r < Rows; ++r) {
        const float sum = warp_reduce_sum_xor(sq[r]);
        if (lane == 0) s.part[r][8][warp] = sum;
      }
    } else {
      float dot[Rows][4] = {};
#pragma unroll 1
      for (int ch = 0; ch < C; ++ch) {
        const int col = (ch * T + tid) * E;
        asm volatile("barrier.sync 1, 384;" ::: "memory");
#pragma unroll 1
        for (int c = 0; c < 4; ++c) {
          vec_t<float, E> v[Rows];
#pragma unroll
          for (int r = 0; r < Rows; ++r)
            if (r < valid) v[r] = to<float>(load<__nv_bfloat16, E>(&s.y[r][c][col]));
#pragma unroll
          for (int p = 0; p < 4; ++p) {
            const int proj = (wg - 1) * 4 + p;
            const float* wp = w + int64_t(proj) * 4 * D + c * D + col;
            const auto weights = ihc_load_weight8<true>(wp);
#pragma unroll
            for (int r = 0; r < Rows; ++r) {
              if (r < valid) {
#pragma unroll
                for (int j = 0; j < E; ++j) dot[r][p] += v[r][j] * weights[j];
              }
            }
          }
        }
      }
#pragma unroll
      for (int r = 0; r < Rows; ++r) {
#pragma unroll
        for (int p = 0; p < 4; ++p) {
          const float sum = warp_reduce_sum_xor(dot[r][p]);
          if (lane == 0) s.part[r][(wg - 1) * 4 + p][warp] = sum;
        }
      }
    }
    __syncthreads();
    if (threadIdx.x < Rows * 4) {
      const int r = threadIdx.x / 4, i = threadIdx.x % 4;
      if (r < valid) {
        float pre = 0, post = 0, sq = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          pre += s.part[r][i][k];
          post += s.part[r][i + 4][k];
          sq += s.part[r][8][k];
        }
        const float inv = rsqrtf_ftz(sq * (1.f / (4 * D)) + norm_eps);
        s.gates[r][i] = sigmoid(scale[0] * (pre * inv) + bias[i]) + hc_eps;
        hp[(base + r) * 4 + i] =
            magnitude * sigmoid(scale[1] * (post * inv) + bias[i + 4]) + hc_eps;
      }
    }
    __syncthreads();
#pragma unroll 1
    for (int wave = 0; wave < (Rows + 2) / 3; ++wave) {
      const int r = wave * 3 + wg;
      const bool active = r < valid;
      vec_t<__nv_bfloat16, E> saved[C];
      float sq = 0;
      if (active) {
#pragma unroll
        for (int ch = 0; ch < C; ++ch) {
          const int col = (ch * T + tid) * E;
          vec_t<float, E> out = {};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const auto v = to<float>(load<__nv_bfloat16, E>(&s.y[r][c][col]));
            auto* o2 = reinterpret_cast<float2*>(&out[0]);
            const auto* v2 = reinterpret_cast<const float2*>(&v[0]);
#pragma unroll
            for (int j = 0; j < E / 2; ++j)
              o2[j] = ihc_fma2(make_float2(s.gates[r][c], s.gates[r][c]), v2[j], o2[j]);
          }
          saved[ch] = to<__nv_bfloat16>(out);
          if constexpr (Norm) {
            out = to<float>(saved[ch]);
#pragma unroll
            for (int j = 0; j < E; ++j) sq += out[j] * out[j];
          } else {
            store(z + int64_t(base + r) * D + col, saved[ch]);
          }
        }
      }
      if constexpr (Norm) {
        const float sum = warp_reduce_sum_xor(sq);
        if (lane == 0 && r < Rows) s.norm[r][warp] = sum;
        __syncthreads();
        if (active) {
          float total = 0;
#pragma unroll
          for (int k = 0; k < 4; ++k) total += s.norm[r][k];
          const float inv = rsqrtf_ftz(total * (1.f / D) + rms_eps);
#pragma unroll
          for (int ch = 0; ch < C; ++ch) {
            const int col = (ch * T + tid) * E;
            const auto g = to<float>(load<__nv_bfloat16, E>(gamma + col));
            auto out = to<float>(saved[ch]);
            auto* o2 = reinterpret_cast<float2*>(&out[0]);
            const auto* g2 = reinterpret_cast<const float2*>(&g[0]);
#pragma unroll
            for (int j = 0; j < E / 2; ++j)
              o2[j] = ihc_mul2(ihc_mul2(o2[j], make_float2(inv, inv)), g2[j]);
            store(z + int64_t(base + r) * D + col, to<__nv_bfloat16>(out));
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

// Launch geometry for the pre/head row kernel.
struct PreConfig {
  int rows;
  int threads;
};

#if HPC_TARGET_ARCH == 90
// Pre row counts at 256 threads.
constexpr int pre_rows_per_block(int num_batch, bool fuse_rms_norm) {
  if (num_batch < (fuse_rms_norm ? 192 : 64)) {
    return 1;
  }
  if (num_batch < 320) {
    return 2;
  }
  return 4;
}

// Head projects half as many rows as pre and uses separate batch thresholds.
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
// One CTA per token for batches up to the SM count.
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
  // hidden 4096: threads * 8 must divide 4096, so the wide arm is 512 not 768.
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

// Selects launch shape and forwards to launch_pre_head_rows.
// sm90 uses pre_config(), other targets use sm100_config().
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
  // SM90 head uses head_rows_per_block() instead of pre_config().
  if constexpr (kHasPost) {
    const PreConfig cfg = pre_config(num_batch, kHiddenDim);
    // Only instantiate pairs pre_config() can return.
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
    // Anything pre_config() does not cover falls through below.
  }

  // Keep pre-only row configurations out of the head specialization.
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
  // Only instantiate pairs sm100_config() can return.
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
  // Fallback for any pair not covered above.
  launch(std::integral_constant<int, 1>{}, std::integral_constant<int, kRowThreads>{});
#endif  // HPC_TARGET_ARCH == 90
}

// Match grid, block and cluster dimensions to the kernel's uniform M branch.
template <int kHiddenDim, bool kFuseRmsNorm, bool kCastBfloatForNorm, bool kUsePDL>
cudaError_t launch_fuse_ihc_post_pre(int num_batch, cudaStream_t stream, __nv_bfloat16* y_ptr,
                                     __nv_bfloat16* z_ptr, float* H_post_out_ptr,
                                     const __nv_bfloat16* xa_ptr, const __nv_bfloat16* residual_ptr,
                                     const float* H_post_in_ptr, const float* w_ptr,
                                     const float* hc_scale_ptr, const float* hc_base_ptr,
                                     float norm_eps, float hc_eps, float magnitude,
                                     const __nv_bfloat16* rms_weight_ptr, float rms_eps) {
  constexpr int kThreads = kHiddenDim == 3072 ? 192 : 256;
  const bool weights_aligned = (reinterpret_cast<uintptr_t>(w_ptr) & 31) == 0;
#if HPC_TARGET_ARCH != 90
  if constexpr (kUsePDL && (!kFuseRmsNorm || kCastBfloatForNorm)) {
    constexpr int kUmmaMin = kHiddenDim == 3072 ? 8960 : 8192;
    if (num_batch >= kUmmaMin && weights_aligned) {
      constexpr int kUmmaThreads = 256;
      constexpr int kUmmaBlocks = kernels::ihc_post_pre_umma_blocks(kHiddenDim);
      auto kernel = kernels::fuse_ihc_post_pre_kernel<4, kHiddenDim, 8, kUmmaThreads, kUsePDL,
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
      constexpr int kThreeWarpMin =
          kHiddenDim == 6144 ? 80 : kHiddenDim == 3072 ? (kFuseRmsNorm ? 48 : 72) : 149;
      const int three_warp_max = kHiddenDim == 3072 ? 4 * get_sm_count() : 296;
      if (num_batch >= kThreeWarpMin && num_batch <= three_warp_max && weights_aligned) {
        auto kernel = kernels::fuse_ihc_post_pre_kernel<4, kHiddenDim, 8, 384, kUsePDL,
                                                        kFuseRmsNorm, kCastBfloatForNorm, true,
                                                        kernels::IhcPostPreThreeWarpGroups>;
        constexpr size_t kShared =
            kHiddenDim == 4096 ? sizeof(kernels::IhcPostPreThreeWarpStorage<4096>) : 0;
        if constexpr (kShared > 0) {
          const cudaError_t setup =
              cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kShared);
          if (setup != cudaSuccess) return setup;
        }
        cudaLaunchConfig_t three_warp_config{};
        three_warp_config.gridDim = dim3(num_batch);
        three_warp_config.blockDim = dim3(384);
        three_warp_config.dynamicSmemBytes = kShared;
        three_warp_config.stream = stream;
        cudaLaunchAttribute attr{};
        attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr.val.programmaticStreamSerializationAllowed = 1;
        three_warp_config.attrs = &attr;
        three_warp_config.numAttrs = 1;
        return cudaLaunchKernelEx(&three_warp_config, kernel, y_ptr, z_ptr, H_post_out_ptr, xa_ptr,
                                  residual_ptr, H_post_in_ptr, w_ptr, hc_scale_ptr, hc_base_ptr,
                                  num_batch, norm_eps, hc_eps, magnitude, rms_weight_ptr, rms_eps);
      }
    }
  }
#endif
  cudaLaunchConfig_t config{};
  config.stream = stream;
  config.gridDim = dim3(num_batch);
  config.blockDim = dim3(kThreads);
  cudaLaunchAttribute attrs[2]{};
  config.attrs = attrs;
  if constexpr (kUsePDL) {
    attrs[config.numAttrs].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[config.numAttrs++].val.programmaticStreamSerializationAllowed = 1;
  }
#if HPC_TARGET_ARCH != 90
  int blocks = 1;
  if constexpr (kHiddenDim == 6144) {
    blocks = num_batch < 80 ? 8 : 1;
    const int threads = num_batch < 80 ? 96 : num_batch <= 296 ? 768 : 384;
    const int rows = num_batch >= 297 ? 2 : 1;
    config.gridDim = dim3(((num_batch + rows - 1) / rows) * blocks);
    config.blockDim = dim3(threads);
  } else {
    static_assert(kHiddenDim == 4096 || kHiddenDim == 3072, "supported hidden dimensions");
    constexpr int kClusterBlocks = kernels::ihc_post_pre_cluster_blocks(kHiddenDim);
    blocks = num_batch <= 128 ? kClusterBlocks : 1;
    const int rows = num_batch >= 297 ? 2 : 1;
    config.gridDim = dim3(((num_batch + rows - 1) / rows) * blocks);
    config.blockDim = dim3(blocks > 1 ? kHiddenDim / (kClusterBlocks * 8) : kThreads);
  }
  auto& attr = attrs[config.numAttrs++];
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim.x = blocks;
  attr.val.clusterDim.y = 1;
  attr.val.clusterDim.z = 1;
#endif
  auto launch = [&](auto aligned_tag) {
    constexpr bool kWeightsAligned = decltype(aligned_tag)::value;
    auto kernel =
        kernels::fuse_ihc_post_pre_kernel<4, kHiddenDim, 8, kThreads, kUsePDL, kFuseRmsNorm,
                                          kCastBfloatForNorm, kWeightsAligned>;
    return cudaLaunchKernelEx(&config, kernel, y_ptr, z_ptr, H_post_out_ptr, xa_ptr, residual_ptr,
                              H_post_in_ptr, w_ptr, hc_scale_ptr, hc_base_ptr, num_batch, norm_eps,
                              hc_eps, magnitude, rms_weight_ptr, rms_eps);
  };
  return weights_aligned ? launch(std::true_type{}) : launch(std::false_type{});
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

// post's launch shape: use enough splits to cover each token's row in one pass.
struct PostConfig {
  int threads;
  int splits;
};

constexpr PostConfig post_config(int num_batch, int hidden_dim) {
  (void)num_batch;
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
      launch_post_pre<kHidden, false, false>(y_ptr, z_ptr, H_post_out_ptr, xa_ptr, residual_ptr,
                                             H_post_in_ptr, w_ptr, hc_scale_ptr, hc_base_ptr,
                                             num_batch, norm_eps, hc_eps, magnitude, nullptr, 0.f,
                                             stream, use_pdl);
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
