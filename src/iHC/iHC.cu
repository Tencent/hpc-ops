// Copyright 2026 Tencent

#include <cuda.h>
#include <stdio.h>

#include <algorithm>
#include <cstdlib>
#include <type_traits>

#include "cutlass/fast_math.h"
#include "src/iHC/iHC.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace iHC {
namespace kernels {

// The projection/sqrsum accumulation for one block's rows.
//
// kActiveRows being a compile-time trip count is the point of factoring this
// out: under a runtime bound the compiler can neither unroll the row loop nor
// hoist the next weight loads above the current FFMAs, so the whole W tile (64
// registers) stays live with every load on the critical path. On H20 at n=2048
// / hidden 6144 / rows=4 that put 77% of stall cycles on long-scoreboard (8.32
// of 10.76) at 22-53% throughput -- waiting on loads, not short of bandwidth.
// Fixing the count drops the stall to 3.26 and runs 1.13x faster.
//
// The caller passes kRowsPerBlock for a full tile and 1 for the grid's partial
// tail, so the tail stays exact without instantiating every row count.
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

  // One flat pass over hc_mult * hidden_dim. The original nested the head and
  // chunk loops, but `i * kHiddenDim + icol` is contiguous across them anyway.
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

  // Full tiles take the specialized trip count; only the last block of the grid
  // falls back to one row at a time. See ihc_pre_accumulate() for why the
  // compile-time bound matters.
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

// kThreadsPerBlock is a template parameter because __launch_bounds__ must match
// the block launched: a hardcoded 128 made every other size an invalid launch.
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

template <int kHCMult, int kHiddenDim, int kElementsPerThread, int kThreadsPerBlock,
          bool kUsePDL = false, bool kFuseRmsNorm = false, bool kCastBfloatForNorm = false>
__global__ void __launch_bounds__(kThreadsPerBlock)
    fuse_ihc_post_pre_kernel(__nv_bfloat16* y_ptr, __nv_bfloat16* z_ptr, float* H_post_out_ptr,
                             const __nv_bfloat16* xa_ptr, const __nv_bfloat16* residual_ptr,
                             const float* H_post_in_ptr, const float* w_ptr,
                             const float* hc_scale_ptr, const float* hc_base_ptr, int num_batch,
                             float norm_eps, float hc_eps, float magnitude,
                             const __nv_bfloat16* rms_weight_ptr = nullptr, float rms_eps = 0.f) {
  constexpr int kProjRows = 2 * kHCMult;
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

  const int irow = blockIdx.x;
  if (irow >= num_batch) {
    return;
  }
  const int iwarp = threadIdx.x / kWarpSize;
  const int ilane = threadIdx.x % kWarpSize;

  __shared__ float smem_partial[kProjRows + 1][kWarpsPerBlock];
  __shared__ __align__(sizeof(float) * kHCMult) float smem_H_pre[kHCMult];

  const auto reg_H_post_in = load<float, kHCMult>(H_post_in_ptr + irow * kHCMult);
  const __nv_bfloat16* xa_row_ptr = xa_ptr + irow * kHiddenDim;
  const __nv_bfloat16* res_row_ptr = residual_ptr + irow * kHCDim;
  __nv_bfloat16* y_row_ptr = y_ptr + irow * kHCDim;

  float acc_dot[kProjRows];
  float acc_sqsum = 0.f;
#pragma unroll
  for (int c = 0; c < kProjRows; ++c) {
    acc_dot[c] = 0.f;
  }

#pragma unroll
  for (int ch = 0; ch < kChunks; ++ch) {
    const int icol = (ch * kThreadsPerBlock + threadIdx.x) * kElementsPerThread;
    const auto reg_xa = to<float>(load<__nv_bfloat16, kElementsPerThread>(xa_row_ptr + icol));

#pragma unroll
    for (int i = 0; i < kHCMult; ++i) {
      const int iflat = i * kHiddenDim + icol;
      const auto reg_res = to<float>(load<__nv_bfloat16, kElementsPerThread>(res_row_ptr + iflat));

      vec_t<float, kElementsPerThread> reg_y;
      auto* y2 = reinterpret_cast<float2*>(&reg_y[0]);
      const auto* xa2 = reinterpret_cast<const float2*>(&reg_xa[0]);
      const auto* res2 = reinterpret_cast<const float2*>(&reg_res[0]);
      const float h = reg_H_post_in[i];
#pragma unroll
      for (int j = 0; j < kElementsPerThread / 2; ++j) {
        y2[j].x = h * xa2[j].x + res2[j].x;
        y2[j].y = h * xa2[j].y + res2[j].y;
      }
      store(y_row_ptr + iflat, to<__nv_bfloat16>(reg_y));

      const auto reg_y_bf = to<float>(to<__nv_bfloat16>(reg_y));
      const auto* yb2 = reinterpret_cast<const float2*>(&reg_y_bf[0]);

#pragma unroll
      for (int j = 0; j < kElementsPerThread / 2; ++j) {
        acc_sqsum += yb2[j].x * yb2[j].x;
        acc_sqsum += yb2[j].y * yb2[j].y;
      }
      constexpr int kHalf = kElementsPerThread / 2;
#pragma unroll
      for (int c = 0; c < kProjRows; ++c) {
        const float* w_row_ptr = w_ptr + c * kHCDim + iflat;
        vec_t<float, kElementsPerThread> reg_w;
        *reinterpret_cast<vec_t<float, kHalf>*>(&reg_w[0]) = load<float, kHalf>(w_row_ptr);
        *reinterpret_cast<vec_t<float, kHalf>*>(&reg_w[kHalf]) =
            load<float, kHalf>(w_row_ptr + kHalf);
        const auto* w2 = reinterpret_cast<const float2*>(&reg_w[0]);
#pragma unroll
        for (int j = 0; j < kElementsPerThread / 2; ++j) {
          acc_dot[c] += yb2[j].x * w2[j].x;
          acc_dot[c] += yb2[j].y * w2[j].y;
        }
      }
    }
  }

#pragma unroll
  for (int c = 0; c < kProjRows; ++c) {
    const float v = warp_reduce_sum_xor(acc_dot[c]);
    if (ilane == 0) {
      smem_partial[c][iwarp] = v;
    }
  }
  {
    const float v = warp_reduce_sum_xor(acc_sqsum);
    if (ilane == 0) {
      smem_partial[kProjRows][iwarp] = v;
    }
  }
  __syncthreads();

  for (int c = threadIdx.x; c < kProjRows + 1; c += kThreadsPerBlock) {
    float sum = 0.f;
#pragma unroll
    for (int w = 0; w < kWarpsPerBlock; ++w) {
      sum += smem_partial[c][w];
    }
    smem_partial[c][0] = sum;
  }
  __syncthreads();

  for (int i = threadIdx.x; i < kHCMult; i += kThreadsPerBlock) {
    const float rms_scale = rsqrtf_ftz(smem_partial[kProjRows][0] * kInvHCDim + norm_eps);
    const float mixes_pre = smem_partial[i][0] * rms_scale;
    smem_H_pre[i] = sigmoid(hc_scale_ptr[0] * mixes_pre + hc_base_ptr[i]) + hc_eps;

    const float mixes_post = smem_partial[kHCMult + i][0] * rms_scale;
    H_post_out_ptr[irow * kHCMult + i] =
        magnitude * sigmoid(hc_scale_ptr[1] * mixes_post + hc_base_ptr[kHCMult + i]) + hc_eps;
  }
  __syncthreads();

  const auto reg_H_pre = load<float, kHCMult>(&smem_H_pre[0]);
  __nv_bfloat16* z_row_ptr = z_ptr + irow * kHiddenDim;

  vec_t<float, kElementsPerThread> reg_z[kChunks];
  float z_sqsum = 0.f;

#pragma unroll
  for (int ch = 0; ch < kChunks; ++ch) {
    const int icol = (ch * kThreadsPerBlock + threadIdx.x) * kElementsPerThread;
#pragma unroll
    for (int j = 0; j < kElementsPerThread; ++j) {
      reg_z[ch][j] = 0.f;
    }
    auto* z2 = reinterpret_cast<float2*>(&reg_z[ch][0]);
#pragma unroll
    for (int i = 0; i < kHCMult; ++i) {
      const auto reg_y =
          to<float>(load<__nv_bfloat16, kElementsPerThread>(y_row_ptr + i * kHiddenDim + icol));
      const float h = reg_H_pre[i];
      const auto* y2 = reinterpret_cast<const float2*>(&reg_y[0]);
#pragma unroll
      for (int j = 0; j < kElementsPerThread / 2; ++j) {
        z2[j].x += h * y2[j].x;
        z2[j].y += h * y2[j].y;
      }
    }
    if constexpr (kFuseRmsNorm) {
#pragma unroll
      for (int j = 0; j < kElementsPerThread / 2; ++j) {
        if constexpr (kCastBfloatForNorm) {
          z2[j].x = __bfloat162float(__float2bfloat16(z2[j].x));
          z2[j].y = __bfloat162float(__float2bfloat16(z2[j].y));
        }
        z_sqsum += z2[j].x * z2[j].x;
        z_sqsum += z2[j].y * z2[j].y;
      }
    } else {
      store(z_row_ptr + icol, to<__nv_bfloat16>(reg_z[ch]));
    }
  }

  if constexpr (kFuseRmsNorm) {
    const float v = warp_reduce_sum_xor(z_sqsum);
    if (ilane == 0) {
      smem_partial[0][iwarp] = v;
    }
    __syncthreads();
    float total = 0.f;
#pragma unroll
    for (int w = 0; w < kWarpsPerBlock; ++w) {
      total += smem_partial[0][w];
    }
    const float rms_scale = rsqrtf_ftz(total * kInvHiddenDim + rms_eps);
#pragma unroll
    for (int ch = 0; ch < kChunks; ++ch) {
      const int icol = (ch * kThreadsPerBlock + threadIdx.x) * kElementsPerThread;
      const auto reg_g = to<float>(load<__nv_bfloat16, kElementsPerThread>(rms_weight_ptr + icol));
      auto* z2 = reinterpret_cast<float2*>(&reg_z[ch][0]);
      const auto* g2 = reinterpret_cast<const float2*>(&reg_g[0]);
#pragma unroll
      for (int j = 0; j < kElementsPerThread / 2; ++j) {
        z2[j].x = z2[j].x * rms_scale * g2[j].x;
        z2[j].y = z2[j].y * rms_scale * g2[j].y;
      }
      store(z_row_ptr + icol, to<__nv_bfloat16>(reg_z[ch]));
    }
  }

  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

template <int kHCMult, int kHiddenDim, int kElementsPerThread, int kThreadsPerBlock,
          bool kUsePDL = false>
__global__ void __launch_bounds__(kThreadsPerBlock)
    fuse_ihc_post_pre_stage1_kernel(__nv_bfloat16* y_ptr, float* scratch_ptr,
                                    const __nv_bfloat16* xa_ptr, const __nv_bfloat16* residual_ptr,
                                    const float* H_post_in_ptr, const float* w_ptr, int num_batch) {
  constexpr int kProjRows = 2 * kHCMult;
  constexpr int kHCDim = kHCMult * kHiddenDim;
  constexpr int kColsPerBlock = kThreadsPerBlock * kElementsPerThread;
  constexpr int kWarpSize = 32;
  constexpr int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;
  constexpr int kNumChunks = kHiddenDim / kColsPerBlock;
  static_assert(kHiddenDim % kColsPerBlock == 0, "block cols must divide hidden dim");

  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }
  const int ch = blockIdx.x;
  const int irow = blockIdx.y;
  const int iwarp = threadIdx.x / kWarpSize;
  const int ilane = threadIdx.x % kWarpSize;
  const int icol = ch * kColsPerBlock + threadIdx.x * kElementsPerThread;

  __shared__ float smem_partial[kProjRows + 1][kWarpsPerBlock];

  const auto reg_H_post_in = load<float, kHCMult>(H_post_in_ptr + irow * kHCMult);
  const __nv_bfloat16* xa_row_ptr = xa_ptr + irow * kHiddenDim;
  const __nv_bfloat16* res_row_ptr = residual_ptr + irow * kHCDim;
  __nv_bfloat16* y_row_ptr = y_ptr + irow * kHCDim;

  float acc_dot[kProjRows];
  float acc_sqsum = 0.f;
#pragma unroll
  for (int c = 0; c < kProjRows; ++c) {
    acc_dot[c] = 0.f;
  }

  const auto reg_xa = to<float>(load<__nv_bfloat16, kElementsPerThread>(xa_row_ptr + icol));
#pragma unroll
  for (int i = 0; i < kHCMult; ++i) {
    const int iflat = i * kHiddenDim + icol;
    const auto reg_res = to<float>(load<__nv_bfloat16, kElementsPerThread>(res_row_ptr + iflat));
    vec_t<float, kElementsPerThread> reg_y;
    auto* y2 = reinterpret_cast<float2*>(&reg_y[0]);
    const auto* xa2 = reinterpret_cast<const float2*>(&reg_xa[0]);
    const auto* res2 = reinterpret_cast<const float2*>(&reg_res[0]);
    const float h = reg_H_post_in[i];
#pragma unroll
    for (int j = 0; j < kElementsPerThread / 2; ++j) {
      y2[j].x = h * xa2[j].x + res2[j].x;
      y2[j].y = h * xa2[j].y + res2[j].y;
    }
    store(y_row_ptr + iflat, to<__nv_bfloat16>(reg_y));
    const auto reg_y_bf = to<float>(to<__nv_bfloat16>(reg_y));
    const auto* yb2 = reinterpret_cast<const float2*>(&reg_y_bf[0]);
#pragma unroll
    for (int j = 0; j < kElementsPerThread / 2; ++j) {
      acc_sqsum += yb2[j].x * yb2[j].x;
      acc_sqsum += yb2[j].y * yb2[j].y;
    }
    constexpr int kHalf = kElementsPerThread / 2;
#pragma unroll
    for (int c = 0; c < kProjRows; ++c) {
      const float* w_row_ptr = w_ptr + c * kHCDim + iflat;
      vec_t<float, kElementsPerThread> reg_w;
      *reinterpret_cast<vec_t<float, kHalf>*>(&reg_w[0]) = load<float, kHalf>(w_row_ptr);
      *reinterpret_cast<vec_t<float, kHalf>*>(&reg_w[kHalf]) =
          load<float, kHalf>(w_row_ptr + kHalf);
      const auto* w2 = reinterpret_cast<const float2*>(&reg_w[0]);
#pragma unroll
      for (int j = 0; j < kElementsPerThread / 2; ++j) {
        acc_dot[c] += yb2[j].x * w2[j].x;
        acc_dot[c] += yb2[j].y * w2[j].y;
      }
    }
  }

#pragma unroll
  for (int c = 0; c < kProjRows; ++c) {
    const float v = warp_reduce_sum_xor(acc_dot[c]);
    if (ilane == 0) {
      smem_partial[c][iwarp] = v;
    }
  }
  {
    const float v = warp_reduce_sum_xor(acc_sqsum);
    if (ilane == 0) {
      smem_partial[kProjRows][iwarp] = v;
    }
  }
  __syncthreads();
  float* scratch_row = scratch_ptr + (irow * kNumChunks + ch) * (kProjRows + 1);
  for (int c = threadIdx.x; c < kProjRows + 1; c += kThreadsPerBlock) {
    float sum = 0.f;
#pragma unroll
    for (int w = 0; w < kWarpsPerBlock; ++w) {
      sum += smem_partial[c][w];
    }
    scratch_row[c] = sum;
  }

  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

template <int kHCMult, int kHiddenDim, int kElementsPerThread, int kThreadsPerBlock,
          bool kUsePDL = false, bool kFuseRmsNorm = false, bool kCastBfloatForNorm = false>
__global__ void __launch_bounds__(kThreadsPerBlock)
    fuse_ihc_post_pre_stage2_kernel(__nv_bfloat16* z_ptr, float* H_post_out_ptr,
                                    const __nv_bfloat16* y_ptr, const float* scratch_ptr,
                                    const float* hc_scale_ptr, const float* hc_base_ptr,
                                    int num_batch, float norm_eps, float hc_eps, float magnitude,
                                    const __nv_bfloat16* rms_weight_ptr = nullptr,
                                    float rms_eps = 0.f) {
  constexpr int kProjRows = 2 * kHCMult;
  constexpr int kHCDim = kHCMult * kHiddenDim;
  constexpr int kChunks = kHiddenDim / (kThreadsPerBlock * kElementsPerThread);
  constexpr int kWarpSize = 32;
  constexpr int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;
  constexpr float kInvHCDim = 1.0f / kHCDim;
  constexpr float kInvHiddenDim = 1.0f / kHiddenDim;

  const int irow = blockIdx.x;
  const int iwarp = threadIdx.x / kWarpSize;
  const int ilane = threadIdx.x % kWarpSize;

  __shared__ float smem_partial[kWarpsPerBlock];
  __shared__ __align__(sizeof(float) * kHCMult) float smem_H_pre[kHCMult];

  float full[kProjRows + 1];
#pragma unroll
  for (int c = 0; c < kProjRows + 1; ++c) {
    full[c] = 0.f;
  }
  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }
#pragma unroll
  for (int ch = 0; ch < kChunks; ++ch) {
    const float* scratch_row = scratch_ptr + (irow * kChunks + ch) * (kProjRows + 1);
#pragma unroll
    for (int c = 0; c < kProjRows + 1; ++c) {
      full[c] += scratch_row[c];
    }
  }

  const float rms_scale = rsqrtf_ftz(full[kProjRows] * kInvHCDim + norm_eps);
  for (int i = threadIdx.x; i < kHCMult; i += kThreadsPerBlock) {
    const float mixes_pre = full[i] * rms_scale;
    smem_H_pre[i] = sigmoid(hc_scale_ptr[0] * mixes_pre + hc_base_ptr[i]) + hc_eps;
    const float mixes_post = full[kHCMult + i] * rms_scale;
    H_post_out_ptr[irow * kHCMult + i] =
        magnitude * sigmoid(hc_scale_ptr[1] * mixes_post + hc_base_ptr[kHCMult + i]) + hc_eps;
  }
  __syncthreads();

  const auto reg_H_pre = load<float, kHCMult>(&smem_H_pre[0]);
  const __nv_bfloat16* y_row_ptr = y_ptr + irow * kHCDim;
  __nv_bfloat16* z_row_ptr = z_ptr + irow * kHiddenDim;

  vec_t<float, kElementsPerThread> reg_z[kChunks];
  float z_sqsum = 0.f;
#pragma unroll
  for (int ch = 0; ch < kChunks; ++ch) {
    const int icol = (ch * kThreadsPerBlock + threadIdx.x) * kElementsPerThread;
#pragma unroll
    for (int j = 0; j < kElementsPerThread; ++j) {
      reg_z[ch][j] = 0.f;
    }
    auto* z2 = reinterpret_cast<float2*>(&reg_z[ch][0]);
#pragma unroll
    for (int i = 0; i < kHCMult; ++i) {
      const auto reg_y =
          to<float>(load<__nv_bfloat16, kElementsPerThread>(y_row_ptr + i * kHiddenDim + icol));
      const float h = reg_H_pre[i];
      const auto* y2 = reinterpret_cast<const float2*>(&reg_y[0]);
#pragma unroll
      for (int j = 0; j < kElementsPerThread / 2; ++j) {
        z2[j].x += h * y2[j].x;
        z2[j].y += h * y2[j].y;
      }
    }
    if constexpr (kFuseRmsNorm) {
#pragma unroll
      for (int j = 0; j < kElementsPerThread / 2; ++j) {
        if constexpr (kCastBfloatForNorm) {
          z2[j].x = __bfloat162float(__float2bfloat16(z2[j].x));
          z2[j].y = __bfloat162float(__float2bfloat16(z2[j].y));
        }
        z_sqsum += z2[j].x * z2[j].x;
        z_sqsum += z2[j].y * z2[j].y;
      }
    } else {
      store(z_row_ptr + icol, to<__nv_bfloat16>(reg_z[ch]));
    }
  }

  if constexpr (kFuseRmsNorm) {
    const float v = warp_reduce_sum_xor(z_sqsum);
    if (ilane == 0) {
      smem_partial[iwarp] = v;
    }
    __syncthreads();
    float total = 0.f;
#pragma unroll
    for (int w = 0; w < kWarpsPerBlock; ++w) {
      total += smem_partial[w];
    }
    const float z_scale = rsqrtf_ftz(total * kInvHiddenDim + rms_eps);
#pragma unroll
    for (int ch = 0; ch < kChunks; ++ch) {
      const int icol = (ch * kThreadsPerBlock + threadIdx.x) * kElementsPerThread;
      const auto reg_g = to<float>(load<__nv_bfloat16, kElementsPerThread>(rms_weight_ptr + icol));
      auto* z2 = reinterpret_cast<float2*>(&reg_z[ch][0]);
      const auto* g2 = reinterpret_cast<const float2*>(&reg_g[0]);
#pragma unroll
      for (int j = 0; j < kElementsPerThread / 2; ++j) {
        z2[j].x = z2[j].x * z_scale * g2[j].x;
        z2[j].y = z2[j].y * z_scale * g2[j].y;
      }
      store(z_row_ptr + icol, to<__nv_bfloat16>(reg_z[ch]));
    }
  }

  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

}  // namespace kernels

namespace {

constexpr int kElementsPerThread = 8;
constexpr int kPostThreadsPerBlock = 128;

// Rows per block for pre (kHasPost = true) at a fixed 256 threads. A block
// reads all of W once, so more rows amortize it until accumulators spill.
//
// The first boundary depends on the RMSNorm fold: on B200 an unfolded request
// below ~176 tokens goes to the split-K path (src/iHC/ihc_splitk.cu) and never
// reaches here, while a folded one does, its second pass over y already
// filling a block at one row. B200, hidden 6144, 256 threads (us per call):
//
//     tokens   plain r=1   plain r=2   folded r=1   folded r=2
//         64       17.80       13.72        12.60        15.25
//        128       18.40       13.85        13.00        15.25
//        192       18.75       14.03        17.83        15.45
//        320       24.59       19.13        25.31        20.16
//
// sm100/103 use sm100_config() instead; this stays as the fallback elsewhere,
// gated because -Werror=all-warnings rejects an unreferenced function (#177-D).
#if HPC_TARGET_ARCH != 100 && HPC_TARGET_ARCH != 103
constexpr int pre_rows_per_block(int num_batch, bool fuse_rms_norm) {
  if (num_batch < (fuse_rms_norm ? 192 : 64)) {
    return 1;
  }
  if (num_batch < 320) {
    return 2;
  }
  return 4;
}

// Rows per block for the head kernel (kHasPost = false). Head needs its own
// table: kHasPost halves kProjRows, so there is less work per row to pay for a
// wider tile, and its register use is not a subset of pre's -- two pairs the
// pre tables pick spill here (sm100 and sm_90a alike), <4096, rows=4, 256thr>
// by 2640B and <6144, rows=4, 384thr> by 1336B.
//
// B200, each pair instantiated directly, 256 threads, us per call (* = choice):
//
//   hidden 6144                      hidden 4096
//   tokens   rows=1  rows=2  rows=4  tokens   rows=1  rows=2   rows=4
//      128    8.88*  10.94       -      320    10.43   9.39*    27.98 spill
//      320   14.13   12.17*  13.06     1024    25.49  15.54*    45.77 spill
//     1024   37.55   21.58   17.00*   16384   347.88 209.20*  1887.76 spill
//    16384  933.59  519.26* 513.53
//
// At 4096 the spilling pair also loses to non-spilling rows=4 at 128 threads
// (13.10 vs 9.49 at 512 tokens), so capping at rows=2 is the tuning answer and
// not just spill avoidance. At 6144 rows=4 takes over from 1024 tokens; the
// 16384 cell is near parity and left on rows=2.
constexpr int head_rows_per_block(int num_batch, int hidden_dim) {
  if (num_batch < 320) {
    return 1;
  }
  if (hidden_dim == 6144 && num_batch >= 1024) {
    return 4;
  }
  return 2;
}
#endif  // HPC_TARGET_ARCH != 100 && HPC_TARGET_ARCH != 103

// Rows and block size as a pair, for sm90's pre path. The two interact (192
// threads wins at several token counts and rows alone cannot reach it), and
// only a block whose span divides hc_mult * hidden_dim is legal -- 128/192/256/384
// at hidden 6144 but only 128/256 at 4096, which is why the tables differ.
//
// H20, against the best of rows=1/2/4 at 256 threads (us/call):
//
//   hidden 6144            hidden 4096
//   tokens  base   this    tokens  base   this
//      512  30.9   26.1      512  20.6   17.8
//     2048 140.3  112.0     2048  94.3   69.9
//    32768 1886.4 1657.1   32768 1243.1 1008.3
//
// Gated on the arch it was measured for: the only caller sits in
// launch_pre_head()'s HPC_TARGET_ARCH == 90 branch, so an ungated definition is
// unreferenced elsewhere, which -Werror=all-warnings rejects (#177-D).
struct PreConfig {
  int rows;
  int threads;
};

// Rows and block size as a pair for sm100/sm103; 256 threads alone is best at
// neither end. Swept per pair in bench_ihc/sweep_rowpath.cu on B200.
//
// The first boundary is sm_count, not a token constant: rows=1 gives a grid of
// exactly num_batch, and crossing one wave costs 1.7x (6.50 -> 11.20us at 148).
#if HPC_TARGET_ARCH == 100 || HPC_TARGET_ARCH == 103
constexpr PreConfig sm100_config(int num_batch, int hidden_dim, bool has_post, bool fuse_rms_norm,
                                 int sm_count) {
  if (hidden_dim == 6144) {
    // One wave of one-row blocks, each as wide as the row.
    if (num_batch <= sm_count) {
      return {1, 768};
    }
    if (num_batch < 320) {
      return {2, 384};
    }
    if (num_batch < 768) {
      // head stays at rows=2: <6144, rows=4, 384thr> spills 1336B, and the
      // spilling pair's 0.5us edge is not worth 2648B of spill loads.
      return has_post ? PreConfig{4, 384} : PreConfig{2, 384};
    }
    if (num_batch < 1024) {
      // No rows=6 arm: it wins only the 768 cell (22.33 vs 24.14) and loses
      // 1.3-2.2x elsewhere. An arm for it measured 0.52x end to end.
      return {4, 256};
    }
    // Past ~1k tokens only the block size is left. Unfolded, 256 wins every
    // cell but head's 2048 (37.55 at 192 vs 38.74), which neighbours contradict.
    if (!fuse_rms_norm) {
      return {4, 256};
    }
    // Folded, the two go opposite ways: pre crosses to 128 from 2048 and stays,
    // head takes 192 only in 2048..4096 (36.64 vs 42.98) and returns to 256.
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
    // Folded head is the one exception to the wide block here: 8.05/8.10 at
    // rows=1/256 against 8.17/8.35 for rows=2/512.
    if (!has_post && fuse_rms_norm) {
      return {1, 256};
    }
    return {2, 512};
  }
  if (num_batch < 1024) {
    // head caps at rows=2: every rows=4 pair at 256 threads spills here, and
    // rows=4/128 is behind until 1024 -- cutting at 768 measured 0.85x.
    return has_post ? PreConfig{4, 256} : PreConfig{2, 256};
  }
  // No rows=6 arm here either, for the reason given above. pre takes the narrow
  // block once ahead: from 2048 folded, from 4096 unfolded (67.93 -> 63.89).
  if (has_post) {
    const int narrow_from = fuse_rms_norm ? 2048 : 4096;
    return num_batch < narrow_from ? PreConfig{4, 256} : PreConfig{4, 128};
  }
  // head's only rows=4 band here is 1024..4096 (14.31 vs 15.30 unfolded).
  if (num_batch < 4096) {
    return {4, 128};
  }
  if (!fuse_rms_norm) {
    // 4096: 51.33 at rows=2/256 against 56.56 for rows=4/128; 8192: 96.86
    // against 97.70. The rows=4 pairs at 256 threads all spill here.
    return {2, 256};
  }
  // Folded, 4096 and 8192 disagree by enough to each get an arm: 53.02 vs
  // 57.24 at 4096, then 97.57 vs 102.04 at 8192.
  return num_batch < 8192 ? PreConfig{2, 128} : PreConfig{4, 128};
}
#endif  // HPC_TARGET_ARCH == 100 || HPC_TARGET_ARCH == 103

#if HPC_TARGET_ARCH == 90

constexpr PreConfig pre_config(int num_batch, int hidden_dim) {
  if (hidden_dim == 6144) {
    // Under 192 tokens the grid is too small for a wide block to pay: at 128
    // tokens rows=4/384 threads measured 17.59us against 13.84us for
    // rows=2/256, which also matches what the shared table already picked
    // there. From 192 up the wide block wins (17.87 vs 19.85 at 192, 18.24 vs
    // 19.90 at 256).
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
    // A notch around 1024 tokens from wave quantization against 78 SMs:
    // rows=6 gives 171 blocks (2.2 waves, the last barely occupied) where
    // rows=8 gives 128. Measured at n=1024, rows=6 85.3us against rows=8
    // 56.4us; the neighbours do not share it (768: 38.6 vs 50.7, 1280: 75.1
    // vs 93.7), so the exception is bounded instead of moving the threshold.
    if (num_batch >= 896 && num_batch < 1152) {
      return {8, 192};
    }
    if (num_batch < 12288) {
      return {6, 192};
    }
    return {8, 128};
  }
  // hidden 4096.
  if (num_batch < 384) {
    return {2, 256};
  }
  if (num_batch < 768) {
    return {4, 256};
  }
  // The same notch as above: rows=6 measured 43.5us at n=1024 against 35.7us
  // for rows=8, while 768 and 1280 both prefer rows=6 (24.5 vs 31.4, 40.3 vs
  // 44.8).
  if (num_batch >= 896 && num_batch < 1152) {
    return {8, 128};
  }
  if (num_batch < 12288) {
    return {6, 128};
  }
  // Past ~12k tokens the narrow blocks lose to 256 threads: at 16384, rows=4
  // at 256 threads measured 496us against 524us (rows=6/128) and 520us
  // (rows=8/128).
  return {4, 256};
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

// Picks the launch shape and forwards; same kernel in every case, the split is
// over constants. sm90 uses pre_config(), sm100/103 sm100_config().
template <int kHiddenDim, bool kHasPost, bool kFuseRmsNorm, bool kCastBfloatForNorm>
void launch_pre_head(__nv_bfloat16* output_y_ptr, float* output_H_post_ptr,
                     const __nv_bfloat16* x_ptr, const float* w_ptr, const float* hc_scale_ptr,
                     const float* hc_base_ptr, int num_batch, float norm_eps, float hc_eps,
                     float magnitude, const __nv_bfloat16* rms_weight_ptr, float rms_eps,
                     cudaStream_t stream, bool use_pdl) {
  // Tag-dispatched: <rows, threads> as integral_constant pairs rather than a
  // macro, so the launch keeps its types and the arms stay greppable.
  auto launch = [&](auto rows_tag, auto threads_tag) {
    launch_pre_head_rows<kHiddenDim, decltype(rows_tag)::value, kHasPost, kFuseRmsNorm,
                         kCastBfloatForNorm, decltype(threads_tag)::value>(
        output_y_ptr, output_H_post_ptr, x_ptr, w_ptr, hc_scale_ptr, hc_base_ptr, num_batch,
        norm_eps, hc_eps, magnitude, rms_weight_ptr, rms_eps, stream, use_pdl);
  };

#if HPC_TARGET_ARCH == 100 || HPC_TARGET_ARCH == 103
  {
    const PreConfig cfg =
        sm100_config(num_batch, kHiddenDim, kHasPost, kFuseRmsNorm, get_sm_count());
    // Only the pairs sm100_config() can return here; threads * kElem must
    // divide kHiddenDim, which is why the two hidden sizes list different ones.
    if constexpr (kHiddenDim == 6144) {
      if (cfg.rows == 1 && cfg.threads == 768) {
        launch(std::integral_constant<int, 1>{}, std::integral_constant<int, 768>{});
        return;
      }
      if (cfg.rows == 2 && cfg.threads == 384) {
        launch(std::integral_constant<int, 2>{}, std::integral_constant<int, 384>{});
        return;
      }
      // `if constexpr`, not a plain `if`: an arm no call reaches is still
      // instantiated, and these spill up to 1336B in the head specialization.
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
      // head-only: sm100_config() returns 192 threads only for the folded head.
      if constexpr (!kHasPost) {
        if (cfg.rows == 4 && cfg.threads == 192) {
          launch(std::integral_constant<int, 4>{}, std::integral_constant<int, 192>{});
          return;
        }
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
      // head-only below 4096 tokens, and pre-only at rows=4/256: that pair
      // spills 2640B as a head kernel, the largest in the operator.
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
    // Reaching here means the table grew a pair without an arm. Take the one
    // shape every hidden size can express rather than fall through silently.
    launch(std::integral_constant<int, 1>{}, std::integral_constant<int, 256>{});
    return;
  }
#endif  // HPC_TARGET_ARCH == 100 || HPC_TARGET_ARCH == 103

  // Head skips pre_config(): two of its pairs spill in the head instantiation,
  // per -Xptxas=-v under sm_90a (a compile-time fact, not an sm100 extrapolation):
  //
  //     instantiation            pre              head
  //     <4096, rows=4, 256thr>   128 regs, 0B     255 regs, 156B spill
  //     <6144, rows=4, 384thr>   152 regs, 0B     168 regs, 524B spill
  //
  // pre_config() returns both inside windows head reaches (4096 at 384..768
  // tokens, 6144 at 192..320). The sm90 spill's runtime cost is unmeasured and
  // a tenth the sm100 one, so head falls through to head_rows_per_block()
  // rather than get an sm90 table invented for it.
#if HPC_TARGET_ARCH == 90
  if constexpr (kHasPost) {
    const PreConfig cfg = pre_config(num_batch, kHiddenDim);
    // Only the pairs pre_config() can return are instantiated. A block size is
    // legal only if threads * kElementsPerThread divides hc_mult * hidden_dim,
    // which is why the two hidden sizes list different pairs.
    if constexpr (kHiddenDim == 6144) {
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
    // Any shape pre_config() does not cover falls through to the table below.
  }
#endif

  // One table per gate group; see head_rows_per_block() for why they differ.
  // The rows=4 arm is `if constexpr`-guarded rather than left as unreachable
  // runtime code because a template arm no call reaches is still compiled: the
  // spilling <4096, rows=4, kHasPost=false> instantiation would otherwise stay
  // in the module and in the -Xptxas=-v output, hiding the next real spill.
  //
  // sm100/103 returned above; compiling this there would instantiate six pairs
  // the architecture never launches.
#if HPC_TARGET_ARCH != 100 && HPC_TARGET_ARCH != 103
  if constexpr (kHasPost) {
    switch (pre_rows_per_block(num_batch, kFuseRmsNorm)) {
      case 1:
        launch(std::integral_constant<int, 1>{}, std::integral_constant<int, 256>{});
        break;
      case 2:
        launch(std::integral_constant<int, 2>{}, std::integral_constant<int, 256>{});
        break;
      default:
        launch(std::integral_constant<int, 4>{}, std::integral_constant<int, 256>{});
        break;
    }
  } else {
    switch (head_rows_per_block(num_batch, kHiddenDim)) {
      case 1:
        launch(std::integral_constant<int, 1>{}, std::integral_constant<int, 256>{});
        break;
      case 2:
        launch(std::integral_constant<int, 2>{}, std::integral_constant<int, 256>{});
        break;
      default:
        // head_rows_per_block() returns 4 only at hidden 6144, where the pair
        // does not spill; at hidden 4096 it caps at 2, so this arm is not
        // instantiated there.
        if constexpr (kHiddenDim == 6144) {
          launch(std::integral_constant<int, 4>{}, std::integral_constant<int, 256>{});
        }
        break;
    }
  }
#endif  // HPC_TARGET_ARCH != 100 && HPC_TARGET_ARCH != 103
}

template <int kHiddenDim, bool kFuseRmsNorm, bool kCastBfloatForNorm>
void launch_post_pre(__nv_bfloat16* y_ptr, __nv_bfloat16* z_ptr, float* H_post_out_ptr,
                     const __nv_bfloat16* xa_ptr, const __nv_bfloat16* residual_ptr,
                     const float* H_post_in_ptr, const float* w_ptr, const float* hc_scale_ptr,
                     const float* hc_base_ptr, int num_batch, float norm_eps, float hc_eps,
                     float magnitude, const __nv_bfloat16* rms_weight_ptr, float rms_eps,
                     float* scratch_ptr, cudaStream_t stream, bool use_pdl) {
  constexpr int kThreads = 256;
  constexpr int kClusterSize = kHiddenDim / (kThreads * kElementsPerThread);

  const bool use_twostage = kClusterSize >= 2 && scratch_ptr != nullptr;

  if (use_twostage) {
    float* scratch = scratch_ptr;
    dim3 grid1(kClusterSize, num_batch, 1);
    dim3 grid2(num_batch, 1, 1);
    dim3 block(kThreads);
    auto launch = [&](auto pdl_tag) {
      constexpr bool kUsePDL = decltype(pdl_tag)::value;
      auto k1 = kernels::fuse_ihc_post_pre_stage1_kernel<4, kHiddenDim, kElementsPerThread,
                                                         kThreads, kUsePDL>;
      auto k2 =
          kernels::fuse_ihc_post_pre_stage2_kernel<4, kHiddenDim, kElementsPerThread, kThreads,
                                                   kUsePDL, kFuseRmsNorm, kCastBfloatForNorm>;
      cudaLaunchConfig_t c1{};
      c1.gridDim = grid1;
      c1.blockDim = block;
      c1.stream = stream;
      cudaLaunchAttribute a1[1];
      c1.numAttrs = 0;
      if constexpr (kUsePDL) {
        a1[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        a1[0].val.programmaticStreamSerializationAllowed = 1;
        c1.attrs = a1;
        c1.numAttrs = 1;
      }
      cudaLaunchKernelEx(&c1, k1, y_ptr, scratch, xa_ptr, residual_ptr, H_post_in_ptr, w_ptr,
                         num_batch);
      cudaLaunchConfig_t c2{};
      c2.gridDim = grid2;
      c2.blockDim = block;
      c2.stream = stream;
      cudaLaunchAttribute a2[1];
      c2.numAttrs = 0;
      if constexpr (kUsePDL) {
        a2[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        a2[0].val.programmaticStreamSerializationAllowed = 1;
        c2.attrs = a2;
        c2.numAttrs = 1;
      }
      cudaLaunchKernelEx(&c2, k2, z_ptr, H_post_out_ptr, y_ptr, scratch, hc_scale_ptr, hc_base_ptr,
                         num_batch, norm_eps, hc_eps, magnitude, rms_weight_ptr, rms_eps);
    };
    if (use_pdl) {
      launch(std::true_type{});
    } else {
      launch(std::false_type{});
    }
    return;
  }

  auto launch = [&](auto pdl_tag) {
    constexpr bool kUsePDL = decltype(pdl_tag)::value;
    auto kernel = kernels::fuse_ihc_post_pre_kernel<4, kHiddenDim, kElementsPerThread, kThreads,
                                                    kUsePDL, kFuseRmsNorm, kCastBfloatForNorm>;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(num_batch);
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
    cudaLaunchKernelEx(&config, kernel, y_ptr, z_ptr, H_post_out_ptr, xa_ptr, residual_ptr,
                       H_post_in_ptr, w_ptr, hc_scale_ptr, hc_base_ptr, num_batch, norm_eps, hc_eps,
                       magnitude, rms_weight_ptr, rms_eps);
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

// post's launch shape: block size, and how many hidden slices cover one token.
//
// post is pure streaming (1 read x reused across 4 channels, 4 read residual,
// 4 write out), so the launch shape is the entire tuning surface.
//
// The old policy derived splits as ceil(sm_count / num_batch), so it collapsed
// to 1 past the SM count. Covering the row exactly wins at every count swept.
//
// B200, 148 SM, hidden 6144, us per call (bench_ihc/sweep_post.cu):
//
//   hidden 6144, 128 threads: 192 tokens measured 6.91us at 1 split (what the
//   old policy picked) against 3.44us at 6. bench_ihc/sweep_post.cu.
//
// 6 x 128 x 8 = 6144: one block per token covers the whole row in one pass. The
// 2.01x gap is one the SM-count formula creates, not one it was working around.
//
// A block that does not cover its row loops with a grid-wide stride, so each
// iteration's loads are hidden/span apart and never coalesce with the next's.
//
// The block size then barely matters: at hidden 6144 every count lands within
// 2% of the best at its own full-coverage split, so 128 stays.
struct PostConfig {
  int threads;
  int splits;
};

constexpr PostConfig post_config(int num_batch, int hidden_dim) {
  (void)num_batch;
  int threads = kPostThreadsPerBlock;
#if HPC_TARGET_ARCH == 100 || HPC_TARGET_ARCH == 103
  // The one block-size arm: 256x2 measured 47.87/92.11/180.54 at 4096/8192/
  // 16384 tokens against 48.14/93.19/181.71 for 128x4. hidden 6144 needs none.
  if (hidden_dim == 4096 && num_batch >= 4096) {
    threads = 256;
  }
#endif

  // Cover the row: one pass, no grid-stride loop. Among shapes that do, the
  // thread count varies by a percent or two and not monotonically.
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

  // Only the sizes post_config() can return. 256 is reachable only at hidden
  // 4096, so the arm is `if constexpr`-guarded.
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
  } else {
    return;
  }
}

void fuse_ihc_post_pre_async(__nv_bfloat16* y_ptr, __nv_bfloat16* z_ptr, float* H_post_out_ptr,
                             const __nv_bfloat16* xa_ptr, const __nv_bfloat16* residual_ptr,
                             const float* H_post_in_ptr, const float* w_ptr,
                             const float* hc_scale_ptr, const float* hc_base_ptr, int num_batch,
                             int hc_mult, int hidden_dim, float norm_eps, float hc_eps,
                             float magnitude, float* scratch_ptr, cudaStream_t stream,
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
                                             scratch_ptr, stream, use_pdl);
    } else if (cast_bfloat_for_norm) {
      launch_post_pre<kHidden, true, true>(y_ptr, z_ptr, H_post_out_ptr, xa_ptr, residual_ptr,
                                           H_post_in_ptr, w_ptr, hc_scale_ptr, hc_base_ptr,
                                           num_batch, norm_eps, hc_eps, magnitude, rms_weight_ptr,
                                           rms_eps, scratch_ptr, stream, use_pdl);
    } else {
      launch_post_pre<kHidden, true, false>(y_ptr, z_ptr, H_post_out_ptr, xa_ptr, residual_ptr,
                                            H_post_in_ptr, w_ptr, hc_scale_ptr, hc_base_ptr,
                                            num_batch, norm_eps, hc_eps, magnitude, rms_weight_ptr,
                                            rms_eps, scratch_ptr, stream, use_pdl);
    }
  };

  if (hidden_dim == 6144) {
    by_hidden(std::integral_constant<int, 6144>{});
  } else if (hidden_dim == 4096) {
    by_hidden(std::integral_constant<int, 4096>{});
  } else {
    return;
  }
}

size_t ihc_post_pre_scratch_floats(int num_batch, int hc_mult, int hidden_dim) {
  if (hc_mult != 4 || (hidden_dim != 6144 && hidden_dim != 4096)) {
    return 0;
  }
  constexpr int kColsPerBlock = 256 * kElementsPerThread;
  const int chunks = hidden_dim / kColsPerBlock;
  if (chunks < 2 || num_batch <= 0 || num_batch >= get_sm_count() * 2) {
    return 0;
  }
  const int proj_rows = 2 * hc_mult;
  return static_cast<size_t>(num_batch) * chunks * (proj_rows + 1);
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
  } else {
    return;
  }
}

}  // namespace iHC
}  // namespace hpc
