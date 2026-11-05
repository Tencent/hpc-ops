// Copyright 2026 Tencent

// Split-K fuse_ihc_pre and fuse_ihc_head, for batches too small to fill the
// device. From the FlagGems-vllm Triton kernels, parameterized by gate group.
//
// fuse_ihc_pre_or_head_kernel gives each block one row, so its grid is
// num_batch and one token leaves all but one SM idle. Splitting K fixes that.
//
// kHasPost picks the operator: pre projects 2*hc rows and emits post gates,
// head projects hc and emits none. The host tabulates the two separately.
//
// The token count is the whole criterion -- decode is the common source of such
// a batch but neither necessary nor sufficient, so nothing here tests the phase.
//
// At the operator root because it runs on every KNOWN_ARCHS: fp32 FFMA, warp
// shuffles and PDL, no wgmma/TMA/cluster. Only ihc_splitk.cu's tables are arch-specific.
//
// stage1 one block per (token, K-chunk) -> its own partial slot, no atomics.
// stage2 reduces the slots, forms the gates, emits the pre-gated channel sum.
//
// fp32 throughout, no tensor cores: K is 24576 at hidden 6144, and a bf16/tf32
// dot would blow the 2e-5 tolerance the post gates are held to.

#ifndef SRC_IHC_IHC_SPLITK_CUH_
#define SRC_IHC_IHC_SPLITK_CUH_

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "src/utils/utils.cuh"

namespace hpc {
namespace iHC {
namespace splitk {

// Partial slot per (token, split): [sqrsum, mix...]. Head's is hc floats
// narrower, and stage2 reads it whole per split, so it is real traffic.
template <int kHCMult, bool kHasPost>
struct PartialSlot {
  static constexpr int kProjRows = kHasPost ? 2 * kHCMult : kHCMult;
  static constexpr int kStride = 1 + kProjRows;
};

// stage1: one block owns one (token, K-chunk), reduced through warp shuffles
// then shared memory.
//
// The loop reads x once into registers and reuses it across all weight rows:
// the baseline Triton kernel's L1TEX bottleneck was re-reading x per row.
template <int kHCMult, int kHiddenDim, int kElementsPerThread, int kThreadsPerBlock, int kBlockK,
          bool kUsePDL = false, bool kHasPost = true>
__global__ void __launch_bounds__(kThreadsPerBlock)
    ihc_pre_stage1_splitk_kernel(float* partial_ptr, const __nv_bfloat16* x_ptr, const float* w_ptr,
                                 int num_batch) {
  constexpr int kProjRows = PartialSlot<kHCMult, kHasPost>::kProjRows;
  constexpr int kHCDim = kHCMult * kHiddenDim;
  constexpr int kWarpSize = 32;
  constexpr int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;
  constexpr int kStride = PartialSlot<kHCMult, kHasPost>::kStride;
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

// stage2: one block per (token, hidden-slice). Reduces that token's slots,
// forms the gates, writes the pre-gated channel sum. Only slice 0 stores post.
//
// Slicing the hidden dim beats walking it in one block (4.9us vs 1.7us across
// six): one block would reproduce the occupancy problem this path exists to
// avoid. The gate math is redone per slice -- a few dozen flops against a
// kHiddenDim streaming load.
//
// kFuseRmsNorm gives that up: sum(y^2) spans the row, and both alternatives
// cost more than the slicing saves (a third kernel ~4.5us; cross-block atomics
// plus a grid barrier worse still). So a folded launch runs one wide block per
// token, keeping two launches and a block-local reduction. The static_assert
// below enforces the pairing; the host picks the thread count to match.
template <int kHCMult, int kHiddenDim, int kElementsPerThread, int kThreadsPerBlock, int kNumSplits,
          bool kUsePDL = false, bool kFuseRmsNorm = false, bool kCastBfloatForNorm = false,
          bool kHasPost = true>
__global__ void __launch_bounds__(kThreadsPerBlock)
    ihc_pre_stage2_splitk_kernel(__nv_bfloat16* output_y_ptr, float* output_H_post_ptr,
                                 const __nv_bfloat16* x_ptr, const float* partial_ptr,
                                 const float* hc_scale_ptr, const float* hc_base_ptr, int num_batch,
                                 float norm_eps, float hc_eps, float magnitude,
                                 const __nv_bfloat16* rms_weight_ptr = nullptr,
                                 float rms_eps = 0.f) {
  constexpr int kProjRows = PartialSlot<kHCMult, kHasPost>::kProjRows;
  constexpr int kHCDim = kHCMult * kHiddenDim;
  constexpr int kStride = PartialSlot<kHCMult, kHasPost>::kStride;
  // One block covers kBlockD contiguous hidden elements.
  constexpr int kBlockD = kThreadsPerBlock * kElementsPerThread;
  constexpr float kInvHCDim = 1.0f / kHCDim;
  constexpr float kInvHiddenDim = 1.0f / kHiddenDim;
  constexpr int kWarpSize = 32;
  constexpr int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;

  static_assert(!kFuseRmsNorm || kBlockD == kHiddenDim,
                "the RMSNorm fold needs the whole row in one block, so a folded launch must "
                "size the block as kHiddenDim / kElementsPerThread");

  static_assert(kHiddenDim % kBlockD == 0,
                "kThreadsPerBlock * kElementsPerThread must divide kHiddenDim");

  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }

  const int token_idx = blockIdx.x;
  if (token_idx >= num_batch) {
    return;
  }

  // Each of the first kProjRows+1 threads owns one slot column and walks the
  // splits; the result goes to shared so every thread can read the gates.
  //
  // A warp-parallel rewrite must fix the summation order, not let it fall out
  // of scheduling: fuse_ihc_pre has a bitwise-repeatability test.
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

    // Compile-time, not a null check: head's tensors have no hc_scale[1].
    if constexpr (kHasPost) {
      if (blockIdx.y == 0) {
        const float mixes_post = smem_reduced[1 + kHCMult + i] * rms_scale;
        output_H_post_ptr[static_cast<int64_t>(token_idx) * kHCMult + i] =
            magnitude * sigmoid(hc_scale_ptr[1] * mixes_post + hc_base_ptr[kHCMult + i]) + hc_eps;
      }
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

  if constexpr (!kFuseRmsNorm) {
    store<__nv_bfloat16, kElementsPerThread>(y_row + icol, to<__nv_bfloat16>(acc));
  } else {
    // y never reaches HBM unnormalized: it stays in acc while the row's sqsum
    // is reduced block-locally (kBlockD == kHiddenDim), then goes out scaled.
    //
    // kCastBfloatForNorm rounds y to bf16 before squaring, matching what the
    // unfused path would read back from HBM; without it the norm is more
    // accurate but not bit-identical. The caller picks.
    float y_sqsum = 0.f;
    if constexpr (kCastBfloatForNorm) {
      auto acc_bf = to<__nv_bfloat16>(acc);
      acc = to<float>(acc_bf);
    }
#pragma unroll
    for (int j = 0; j < kElementsPerThread; ++j) {
      y_sqsum += acc[j] * acc[j];
    }

    __shared__ float smem_sqsum[kWarpsPerBlock];
    const float warp_sum = warp_reduce_sum_xor(y_sqsum);
    if (threadIdx.x % kWarpSize == 0) {
      smem_sqsum[threadIdx.x / kWarpSize] = warp_sum;
    }
    __syncthreads();

    float total = 0.f;
#pragma unroll
    for (int w = 0; w < kWarpsPerBlock; ++w) {
      total += smem_sqsum[w];
    }
    const float y_scale = rsqrtf_ftz(total * kInvHiddenDim + rms_eps);

    const auto reg_g = to<float>(load<__nv_bfloat16, kElementsPerThread>(rms_weight_ptr + icol));
#pragma unroll
    for (int j = 0; j < kElementsPerThread; ++j) {
      acc[j] = acc[j] * y_scale * reg_g[j];
    }
    store<__nv_bfloat16, kElementsPerThread>(y_row + icol, to<__nv_bfloat16>(acc));
  }

  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

}  // namespace splitk
}  // namespace iHC
}  // namespace hpc

#endif  // SRC_IHC_IHC_SPLITK_CUH_
