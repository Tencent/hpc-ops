// Copyright 2025 hpc-ops authors

#ifndef SRC_ATTENTION_MLA_SMALLM_MLA_DIM576_PERSISTENT_COMBINE_KERNELS_CUH_
#define SRC_ATTENTION_MLA_SMALLM_MLA_DIM576_PERSISTENT_COMBINE_KERNELS_CUH_

#include <cuda.h>
#include <cuda_bf16.h>

#include <limits>

#include "src/utils/utils.cuh"

namespace hpc {
namespace attention {
namespace kernels {

// Layout assumptions:
//   y_partial : fp32 [total_splits, num_head_q, v_dim]
//   lse       : fp32 [total_splits, num_head_q]
//   y         : Tout [total_seq_q, num_head_q, v_dim]   strided by ldY on axis-0
//   sink_weight: fp32[num_head_q] (optional)
//
// Each WARP reduces one kVChunk-wide V slice
template <typename Tout, int kVChunk, int kMaxSplits, bool kUseSink, bool kUsePDL = false,
          int kWarpsPerBlock = 1>
__global__ void __launch_bounds__(32 * kWarpsPerBlock)
    attention_mla_dim576_persistent_combine_kernel(Tout* y_ptr, const float* y_partial_ptr,
                                                   const float* lse_ptr,
                                                   const float* sink_weight_ptr,
                                                   const int* cu_splits_ptr, int num_head_q,
                                                   int v_dim, int total_seq_q, int ldY) {
  int ibatch = blockIdx.x;
  int ihead = blockIdx.y;
  int iwarp_in_block = threadIdx.x / 32;
  int iv_chunk = blockIdx.z * kWarpsPerBlock + iwarp_in_block;

  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }

  constexpr int kItemsPerThread = 4;
  constexpr int kThreadsPerWarp = 32;
  constexpr int kVecsPerChunk = kVChunk / kItemsPerThread;
  constexpr int kSplitsPerThread = (kMaxSplits + kThreadsPerWarp - 1) / kThreadsPerWarp;
  static_assert(kVChunk % kItemsPerThread == 0, "kVChunk must be a multiple of kItemsPerThread");
  static_assert(kVecsPerChunk <= kThreadsPerWarp, "kVChunk must fit one warp (<= 32*items)");
  static_assert(kMaxSplits <= kThreadsPerWarp * kSplitsPerThread,
                "kSplitsPerThread * 32 must cover kMaxSplits");

  int s_begin = cu_splits_ptr[ibatch];
  int s_end = cu_splits_ptr[ibatch + 1];
  int num_splits_local = s_end - s_begin;

  int lane = threadIdx.x % 32;

  using YPVec = vec_t<float, kItemsPerThread>;
  using YVec = vec_t<Tout, kItemsPerThread>;

  const YPVec* yp_base =
      reinterpret_cast<const YPVec*>(y_partial_ptr + static_cast<size_t>(ihead) * v_dim);
  size_t yp_stride_split = static_cast<size_t>(num_head_q) * v_dim / kItemsPerThread;
  bool lane_active = (lane < kVecsPerChunk);
  int my_v_vec = iv_chunk * kVecsPerChunk + lane;
  YVec* y_base = reinterpret_cast<YVec*>(y_ptr + static_cast<size_t>(ibatch) * ldY +
                                         static_cast<size_t>(ihead) * v_dim);

  // Empty batch: zero output and bail.
  if (num_splits_local <= 0) {
    if (lane_active) {
      YVec out;
#pragma unroll
      for (int j = 0; j < kItemsPerThread; ++j) {
        out[j] = static_cast<Tout>(0.f);
      }
      y_base[my_v_vec] = out;
    }
    return;
  }

  // Single-split fast path
  if (num_splits_local == 1) {
    return;
  }

  // Load LSE for this batch's splits into per-thread regs.
  float rLSE[kSplitsPerThread];
#pragma unroll
  for (int i = 0; i < kSplitsPerThread; ++i) {
    int s = lane + i * kThreadsPerWarp;
    if (s < num_splits_local) {
      rLSE[i] = lse_ptr[(s_begin + s) * num_head_q + ihead];
    } else {
      rLSE[i] = -std::numeric_limits<float>::infinity();
    }
  }

  // Compute m_global and sum_w (log2 domain).
  constexpr float kLog2e = 1.4426950408889634f;
  float sink_log2 = 0.f;
  if constexpr (kUseSink) {
    sink_log2 = sink_weight_ptr[ihead] * kLog2e;
  }

  // max_lse
  float max_lse = -std::numeric_limits<float>::infinity();
#pragma unroll
  for (int i = 0; i < kSplitsPerThread; ++i) {
    max_lse = fmaxf(max_lse, rLSE[i]);
  }
  if constexpr (kUseSink) {
    max_lse = fmaxf(max_lse, sink_log2);
  }
#pragma unroll
  for (int mask = kThreadsPerWarp / 2; mask > 0; mask >>= 1) {
    max_lse = fmaxf(max_lse, __shfl_xor_sync(0xffffffff, max_lse, mask));
  }

  float sum_w = 0.f;
#pragma unroll
  for (int i = 0; i < kSplitsPerThread; ++i) {
    float lse = rLSE[i];
    if (lse != -std::numeric_limits<float>::infinity()) {
      sum_w += exp2f_ftz(lse - max_lse);
    }
  }
#pragma unroll
  for (int mask = kThreadsPerWarp / 2; mask > 0; mask >>= 1) {
    sum_w += __shfl_xor_sync(0xffffffff, sum_w, mask);
  }
  if constexpr (kUseSink) {
    sum_w += exp2f_ftz(sink_log2 - max_lse);
  }

  float lse_norm = (sum_w > 0.f) ? log2f_ftz(sum_w) : 0.f;

  // y_acc = Σ_s w_s · y_partial[s][head][v].
  int my_v_vec_eff = lane_active ? my_v_vec : (iv_chunk * kVecsPerChunk);

  float rAcc[kItemsPerThread];
#pragma unroll
  for (int j = 0; j < kItemsPerThread; ++j) {
    rAcc[j] = 0.f;
  }

  auto issue_lse = [&](int s) -> float {
    int slot = s / kThreadsPerWarp;
    int src_lane = s % kThreadsPerWarp;
    return __shfl_sync(0xffffffff, rLSE[slot], src_lane);
  };

  int s = 0;
  int paired_end = (num_splits_local / 2) * 2;
  for (; s < paired_end; s += 2) {
    float lse_a = issue_lse(s);
    float lse_b = issue_lse(s + 1);
    YPVec ya = yp_base[(s_begin + s) * yp_stride_split + my_v_vec_eff];
    YPVec yb = yp_base[(s_begin + s + 1) * yp_stride_split + my_v_vec_eff];
    float wa = (lse_a == -std::numeric_limits<float>::infinity())
                   ? 0.f
                   : exp2f_ftz(lse_a - max_lse - lse_norm);
    float wb = (lse_b == -std::numeric_limits<float>::infinity())
                   ? 0.f
                   : exp2f_ftz(lse_b - max_lse - lse_norm);
#pragma unroll
    for (int j = 0; j < kItemsPerThread; ++j) {
      rAcc[j] += wa * ya[j] + wb * yb[j];
    }
  }
  if (s < num_splits_local) {
    float lse_s = issue_lse(s);
    if (lse_s != -std::numeric_limits<float>::infinity()) {
      float w = exp2f_ftz(lse_s - max_lse - lse_norm);
      YPVec y_s = yp_base[(s_begin + s) * yp_stride_split + my_v_vec_eff];
#pragma unroll
      for (int j = 0; j < kItemsPerThread; ++j) {
        rAcc[j] += w * y_s[j];
      }
    }
  }

  if (!lane_active) {
    return;
  }

  YVec out;
#pragma unroll
  for (int j = 0; j < kItemsPerThread; ++j) {
    out[j] = static_cast<Tout>(rAcc[j]);
  }
  y_base[my_v_vec] = out;
}

}  // namespace kernels
}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_MLA_SMALLM_MLA_DIM576_PERSISTENT_COMBINE_KERNELS_CUH_
