// Copyright 2026 Tencent

// sm90 decode-shaped iHC pre: a K-split translation of the FlagGems-vllm Triton
// kernels (flaggems_vllm/ops/ihc/ihc_pre.py).
//
// Why a second path exists on sm90. The shared fuse_ihc_pre_or_head_kernel uses
// kPreRowsPerBlock = 1, so its grid is exactly num_batch. On an H20 (78 SMs)
// that leaves 77 SMs idle at one token, and its latency is flat at ~12-13us from
// 1 to 64 tokens -- measured 0.36x of the Triton baseline at a single token.
// Splitting K instead of rows exposes num_batch * num_splits blocks, which fills
// the machine at decode sizes; above the crossover the row-per-block kernel wins
// again and this path is not used.
//
// Two kernels, like the Triton original:
//   stage1  one block per (token, K-chunk): fp32 sqrsum + the 2*hc projection
//           rows for its chunk, written to its own partial slot (no atomics).
//   stage2  reduces the slots for one token, forms the gates, and emits the
//           pre-gated channel sum. Folding the slot reduction in here keeps the
//           op at two launches; a separate reduction kernel would add a third,
//           which is ~4.5us and a large fraction of the total at these sizes.
//
// fp32 accumulation throughout, no tensor cores: K is hc_mult * hidden_dim
// (24576 at hidden 6144), and a bf16/tf32 dot would blow the 2e-5 tolerance the
// post gates are held to.

#ifndef SRC_IHC_SM90_IHC_DECODE_CUH_
#define SRC_IHC_SM90_IHC_DECODE_CUH_

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "src/utils/utils.cuh"

namespace hpc {
namespace iHC {
namespace sm90 {

// Partial slot layout per (token, split): [sqrsum, mix[0 .. 2*hc-1]].
template <int kHCMult>
struct PartialSlot {
  static constexpr int kStride = 1 + 2 * kHCMult;
};

// stage1: one block owns one (token, K-chunk). Threads stride over the chunk,
// each accumulating into fp32 registers; a warp reduction then a small shared
// reduction combine them, and lane 0 of warp 0 writes the slot.
//
// kElementsPerThread is 8 bf16 (16B) for x. The weight rows are fp32, so the
// same element count is two 16B loads per row -- the loop reads x once and
// reuses it across all 2*hc rows, which is the point: the baseline Triton
// kernel's L1TEX bottleneck was re-reading x once per projection row.
template <int kHCMult, int kHiddenDim, int kElementsPerThread, int kThreadsPerBlock, int kBlockK,
          bool kUsePDL = false>
__global__ void __launch_bounds__(kThreadsPerBlock)
    ihc_pre_stage1_decode_kernel(float* partial_ptr, const __nv_bfloat16* x_ptr, const float* w_ptr,
                                 int num_batch) {
  constexpr int kProjRows = 2 * kHCMult;
  constexpr int kHCDim = kHCMult * kHiddenDim;
  constexpr int kWarpSize = 32;
  constexpr int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;
  constexpr int kStride = PartialSlot<kHCMult>::kStride;
  constexpr int kNumSplits = (kHCDim + kBlockK - 1) / kBlockK;
  constexpr int kPerIter = kThreadsPerBlock * kElementsPerThread;

  static_assert(kBlockK % kPerIter == 0, "kBlockK must be a multiple of the per-iteration span");
  static_assert(kHCDim % kBlockK == 0, "kBlockK must divide hc_mult * hidden_dim");
  static_assert(kThreadsPerBlock % kWarpSize == 0, "block must be whole warps");
  static_assert(kElementsPerThread % 2 == 0, "need pairs for float2 accumulation");

  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }

  const int token_idx = blockIdx.x;
  const int split_idx = blockIdx.y;
  if (token_idx >= num_batch) {
    return;
  }

  const int iwarp = threadIdx.x / kWarpSize;
  const int ilane = threadIdx.x % kWarpSize;
  const int k_start = split_idx * kBlockK;

  float acc_sqsum = 0.f;
  float acc_dot[kProjRows];
#pragma unroll
  for (int c = 0; c < kProjRows; ++c) {
    acc_dot[c] = 0.f;
  }

  const __nv_bfloat16* x_row = x_ptr + static_cast<int64_t>(token_idx) * kHCDim;

#pragma unroll
  for (int iter = 0; iter < kBlockK / kPerIter; ++iter) {
    const int iflat = k_start + iter * kPerIter + threadIdx.x * kElementsPerThread;

    auto reg_x = to<float>(load<__nv_bfloat16, kElementsPerThread>(x_row + iflat));
    const auto* x2 = reinterpret_cast<const float2*>(&reg_x[0]);

#pragma unroll
    for (int j = 0; j < kElementsPerThread / 2; ++j) {
      acc_sqsum += x2[j].x * x2[j].x;
      acc_sqsum += x2[j].y * x2[j].y;
    }

    // x is in registers; walk the projection rows against it.
#pragma unroll
    for (int c = 0; c < kProjRows; ++c) {
      constexpr int kHalf = kElementsPerThread / 2;
      const float* w_row_ptr = w_ptr + c * kHCDim + iflat;
      vec_t<float, kElementsPerThread> reg_w;
      *reinterpret_cast<vec_t<float, kHalf>*>(&reg_w[0]) = load<float, kHalf>(w_row_ptr);
      *reinterpret_cast<vec_t<float, kHalf>*>(&reg_w[kHalf]) =
          load<float, kHalf>(w_row_ptr + kHalf);

      const auto* w2 = reinterpret_cast<const float2*>(&reg_w[0]);
#pragma unroll
      for (int j = 0; j < kElementsPerThread / 2; ++j) {
        acc_dot[c] += x2[j].x * w2[j].x;
        acc_dot[c] += x2[j].y * w2[j].y;
      }
    }
  }

  // Warp-level reduction, then combine the per-warp results through shared
  // memory. kProjRows + 1 values per block: the projections plus the sqrsum.
  __shared__ float smem[kProjRows + 1][kWarpsPerBlock];
#pragma unroll
  for (int c = 0; c < kProjRows; ++c) {
    const float v = warp_reduce_sum_xor(acc_dot[c]);
    if (ilane == 0) {
      smem[c][iwarp] = v;
    }
  }
  {
    const float v = warp_reduce_sum_xor(acc_sqsum);
    if (ilane == 0) {
      smem[kProjRows][iwarp] = v;
    }
  }
  __syncthreads();

  // One thread per output value finishes the cross-warp sum and stores the slot.
  float* slot = partial_ptr + (static_cast<int64_t>(token_idx) * kNumSplits + split_idx) * kStride;
  if (threadIdx.x < kProjRows + 1) {
    float sum = 0.f;
#pragma unroll
    for (int w = 0; w < kWarpsPerBlock; ++w) {
      sum += smem[threadIdx.x][w];
    }
    // Slot is [sqrsum, mix...]; smem holds [mix..., sqrsum].
    const int out_idx = (threadIdx.x == kProjRows) ? 0 : static_cast<int>(threadIdx.x) + 1;
    slot[out_idx] = sum;
  }

  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

// stage2: one block per (token, hidden-slice). Reduces that token's partial
// slots, forms the pre/post gates, and writes the pre-gated channel sum for its
// slice.
//
// The hidden dim must be split across blocks, not walked by one block: at a
// single token the channel sum alone is 4 * hidden bf16 reads, and giving it to
// one block reproduces exactly the occupancy problem this path exists to avoid
// (measured 4.9us for one block vs 1.7us across six). The slot reduction and
// the gate math are repeated per slice -- a few dozen flops against a
// kHiddenDim-sized streaming load, which is the same trade the Triton
// implementation makes. Only slice 0 stores the post gates.
template <int kHCMult, int kHiddenDim, int kElementsPerThread, int kThreadsPerBlock, int kNumSplits,
          bool kUsePDL = false>
__global__ void __launch_bounds__(kThreadsPerBlock)
    ihc_pre_stage2_decode_kernel(__nv_bfloat16* output_y_ptr, float* output_H_post_ptr,
                                 const __nv_bfloat16* x_ptr, const float* partial_ptr,
                                 const float* hc_scale_ptr, const float* hc_base_ptr, int num_batch,
                                 float norm_eps, float hc_eps, float magnitude) {
  constexpr int kProjRows = 2 * kHCMult;
  constexpr int kHCDim = kHCMult * kHiddenDim;
  constexpr int kStride = PartialSlot<kHCMult>::kStride;
  // One block covers kBlockD contiguous hidden elements.
  constexpr int kBlockD = kThreadsPerBlock * kElementsPerThread;
  constexpr float kInvHCDim = 1.0f / kHCDim;

  static_assert(kHiddenDim % kBlockD == 0,
                "kThreadsPerBlock * kElementsPerThread must divide kHiddenDim");

  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }

  const int token_idx = blockIdx.x;
  if (token_idx >= num_batch) {
    return;
  }

  // Reduce this token's slots. Each of the first kProjRows+1 threads owns one
  // column of the slot array and walks the splits; the result goes to shared so
  // every thread can read the gates.
  __shared__ float smem_reduced[kProjRows + 1];
  __shared__ float smem_H_pre[kHCMult];

  const float* slots = partial_ptr + static_cast<int64_t>(token_idx) * kNumSplits * kStride;
  if (threadIdx.x < kProjRows + 1) {
    float sum = 0.f;
    for (int s = 0; s < kNumSplits; ++s) {
      sum += slots[s * kStride + threadIdx.x];
    }
    smem_reduced[threadIdx.x] = sum;
  }
  __syncthreads();

  const float rms_scale = rsqrtf_ftz(smem_reduced[0] * kInvHCDim + norm_eps);
  if (threadIdx.x < kHCMult) {
    const int i = threadIdx.x;
    // smem_reduced is [sqrsum, mix...]: pre logits at 1+i, post at 1+hc+i.
    const float mixes_pre = smem_reduced[1 + i] * rms_scale;
    smem_H_pre[i] = sigmoid(hc_scale_ptr[0] * mixes_pre + hc_base_ptr[i]) + hc_eps;

    if (blockIdx.y == 0) {
      const float mixes_post = smem_reduced[1 + kHCMult + i] * rms_scale;
      output_H_post_ptr[static_cast<int64_t>(token_idx) * kHCMult + i] =
          magnitude * sigmoid(hc_scale_ptr[1] * mixes_post + hc_base_ptr[kHCMult + i]) + hc_eps;
    }
  }
  __syncthreads();

  float reg_H_pre[kHCMult];
#pragma unroll
  for (int i = 0; i < kHCMult; ++i) {
    reg_H_pre[i] = smem_H_pre[i];
  }

  // y[token, slice] = sum_i H_pre[i] * x[token, i, slice]
  const __nv_bfloat16* x_row = x_ptr + static_cast<int64_t>(token_idx) * kHCDim;
  __nv_bfloat16* y_row = output_y_ptr + static_cast<int64_t>(token_idx) * kHiddenDim;
  const int icol = blockIdx.y * kBlockD + threadIdx.x * kElementsPerThread;

  vec_t<float, kElementsPerThread> acc;
#pragma unroll
  for (int j = 0; j < kElementsPerThread; ++j) {
    acc[j] = 0.f;
  }

#pragma unroll
  for (int i = 0; i < kHCMult; ++i) {
    auto reg_x = to<float>(load<__nv_bfloat16, kElementsPerThread>(x_row + i * kHiddenDim + icol));
#pragma unroll
    for (int j = 0; j < kElementsPerThread; ++j) {
      acc[j] += reg_H_pre[i] * reg_x[j];
    }
  }

  store<__nv_bfloat16, kElementsPerThread>(y_row + icol, to<__nv_bfloat16>(acc));

  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

}  // namespace sm90
}  // namespace iHC
}  // namespace hpc

#endif  // SRC_IHC_SM90_IHC_DECODE_CUH_
