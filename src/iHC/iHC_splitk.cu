// Copyright 2026 Tencent
// Copyright 2026, The FlagOS Contributors.

// Split-K fuse_ihc_pre and fuse_ihc_head for small batches.
// kHasPost picks the operator: pre projects 2*hc rows and emits post gates,
// head projects hc and emits none.
//
// stage1: one block per (token, K-chunk) → partial slot, no atomics.
// stage2: reduces slots, forms gates, emits the pre-gated channel sum.
//
// fp32 throughout: a bf16/tf32 dot would violate the post-gate tolerance.
//
// Host dispatch for the split-K fuse_ihc_pre / fuse_ihc_head implementation.
// Routes by token count; defers to the row-per-block kernel once its grid is large enough.

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "src/iHC/iHC.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace iHC {
namespace splitk {

// Partial slot per (token, split): [sqrsum, mix...].
template <int kHCMult, bool kHasPost>
struct PartialSlot {
  static constexpr int kProjRows = kHasPost ? 2 * kHCMult : kHCMult;
  static constexpr int kStride = 1 + kProjRows;
};

// stage1: one block owns one (token, K-chunk). Reads x into registers once
// and reuses it across all weight rows.
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

  // Warp-level reduction then cross-warp through shared memory.
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

// stage2: one block per (token, hidden-slice). Reduces slots, forms gates,
// writes the pre-gated channel sum. Only slice 0 stores post gates.
// kFuseRmsNorm disables slicing: sum(y^2) spans the row, so a folded launch
// uses one wide block per token with a block-local reduction.
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

  // First kProjRows+1 threads each own one slot column and reduce across splits.
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

    // Compile-time check: head has no hc_scale[1].
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
    // y stays in registers; kCastBfloatForNorm rounds to bf16 before squaring
    // to match what the unfused path reads back from HBM.
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


namespace {

constexpr int kStage1Threads = 128;
constexpr int kStage2Threads = 256;
constexpr int kElementsPerThread = 8;

// Force controls: 0=automatic dispatch, 1=split-K, 2=row kernel.
#ifndef HPC_IHC_SPLITK_FORCE
#define HPC_IHC_SPLITK_FORCE 0
#endif

#ifndef HPC_IHC_HEAD_SPLITK_FORCE
#define HPC_IHC_HEAD_SPLITK_FORCE 0
#endif

// Split-K applies below a device- and shape-dependent token threshold.
int splitk_token_threshold(int hidden_dim, bool fuse_rms_norm) {
#if HPC_IHC_SPLITK_FORCE == 1
  (void)hidden_dim;
  (void)fuse_rms_norm;
  return 1 << 30;
#elif HPC_IHC_SPLITK_FORCE == 2
  (void)hidden_dim;
  (void)fuse_rms_norm;
  return 0;
#else
  constexpr float kSmPerBlockRow = 1.2f;
  constexpr float kFoldScale = 1.2f;
  constexpr int kHiddenRef = 6144;

  const int sm_count = get_sm_count();
  if (sm_count <= 0) {
    return 0;
  }
  if (hidden_dim == 3072) {
    return sm_count * 2 / 3;
  }
  const float hidden_scale = sqrtf(static_cast<float>(hidden_dim) / kHiddenRef);
  const float fold_scale = fuse_rms_norm ? kFoldScale : 1.0f;
  return static_cast<int>(sm_count * kSmPerBlockRow * hidden_scale * fold_scale);
#endif
}

constexpr int splitk_block_k(int num_batch) {
#if HPC_TARGET_ARCH == 90
  if (num_batch < 2) {
    return 1024;
  }
  if (num_batch < 48) {
    return 2048;
  }
  return 4096;
#else
  return num_batch < 48 ? 1024 : 4096;
#endif
}

constexpr int head_splitk_block_k(int num_batch, int hidden_dim) {
#if HPC_TARGET_ARCH == 90
  (void)hidden_dim;
  if (num_batch < 2) {
    return 1024;
  }
  if (num_batch < 48) {
    return 2048;
  }
  return 4096;
#else
  if (hidden_dim == 6144) {
    if (num_batch < 16) {
      return 1024;
    }
    return num_batch < 96 ? 2048 : 4096;
  }
  if (num_batch < 48) {
    return 1024;
  }
  return num_batch < 160 ? 2048 : 4096;
#endif
}

int head_splitk_token_threshold(int hidden_dim, bool fuse_rms_norm) {
#if HPC_IHC_HEAD_SPLITK_FORCE == 1
  (void)hidden_dim;
  (void)fuse_rms_norm;
  return 1 << 30;
#elif HPC_IHC_HEAD_SPLITK_FORCE == 2
  (void)hidden_dim;
  (void)fuse_rms_norm;
  return 0;
#else
  constexpr float kSmPerBlockRow = 0.75f;
  constexpr float kFoldScale = 0.8f;
  constexpr int kHiddenRef = 6144;

  const int sm_count = get_sm_count();
  if (sm_count <= 0) {
    return 0;
  }
  if (hidden_dim == 3072 && !fuse_rms_norm) {
    return sm_count / 3;
  }
  const float hidden_scale = static_cast<float>(hidden_dim) / kHiddenRef;
  const float fold_scale = fuse_rms_norm ? kFoldScale : 1.0f;
  return static_cast<int>(sm_count * kSmPerBlockRow * hidden_scale * fold_scale);
#endif
}

template <int kHiddenDim, int kBlockK, bool kFuseRmsNorm = false, bool kCastBfloatForNorm = false,
          bool kHasPost = true>
void launch_splitk(__nv_bfloat16* output_y_ptr, float* output_H_post_ptr,
                   const __nv_bfloat16* x_ptr, const float* w_ptr, const float* hc_scale_ptr,
                   const float* hc_base_ptr, int num_batch, float norm_eps, float hc_eps,
                   float magnitude, float* scratch_ptr, cudaStream_t stream, bool use_pdl,
                   const __nv_bfloat16* rms_weight_ptr = nullptr, float rms_eps = 0.f) {
  constexpr int kHCMult = 4;
  constexpr int kHCDim = kHCMult * kHiddenDim;
  constexpr int kNumSplits = kHCDim / kBlockK;

  auto launch = [&](auto pdl_tag) {
    constexpr bool kUsePDL = decltype(pdl_tag)::value;

    auto stage1 = ihc_pre_stage1_splitk_kernel<kHCMult, kHiddenDim, kElementsPerThread,
                                               kStage1Threads, kBlockK, kUsePDL, kHasPost>;
    cudaLaunchConfig_t cfg1{};
    cfg1.gridDim = dim3(num_batch, kNumSplits);
    cfg1.blockDim = dim3(kStage1Threads);
    cfg1.stream = stream;
    cudaLaunchAttribute attr1[1];
    cfg1.numAttrs = 0;
    if constexpr (kUsePDL) {
      attr1[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attr1[0].val.programmaticStreamSerializationAllowed = 1;
      cfg1.attrs = attr1;
      cfg1.numAttrs = 1;
    }
    cudaLaunchKernelEx(&cfg1, stage1, scratch_ptr, x_ptr, w_ptr, num_batch);

    // RMSNorm requires one block to cover the full hidden row.
    constexpr int kStage2ThreadsHere =
        kFuseRmsNorm ? kHiddenDim / kElementsPerThread
        : kHiddenDim % (kStage2Threads * kElementsPerThread) == 0 ? kStage2Threads
                                                                   : 192;
    constexpr int kBlockD = kStage2ThreadsHere * kElementsPerThread;
    constexpr int kHiddenBlocks = kHiddenDim / kBlockD;
    static_assert(kStage2ThreadsHere <= 1024, "stage2 block exceeds the CUDA limit");

    auto stage2 = ihc_pre_stage2_splitk_kernel<kHCMult, kHiddenDim, kElementsPerThread,
                                               kStage2ThreadsHere, kNumSplits, kUsePDL,
                                               kFuseRmsNorm, kCastBfloatForNorm, kHasPost>;
    cudaLaunchConfig_t cfg2{};
    cfg2.gridDim = dim3(num_batch, kHiddenBlocks);
    cfg2.blockDim = dim3(kStage2ThreadsHere);
    cfg2.stream = stream;
    cudaLaunchAttribute attr2[1];
    cfg2.numAttrs = 0;
    if constexpr (kUsePDL) {
      attr2[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attr2[0].val.programmaticStreamSerializationAllowed = 1;
      cfg2.attrs = attr2;
      cfg2.numAttrs = 1;
    }
    cudaLaunchKernelEx(&cfg2, stage2, output_y_ptr, output_H_post_ptr, x_ptr,
                       const_cast<const float*>(scratch_ptr), hc_scale_ptr, hc_base_ptr, num_batch,
                       norm_eps, hc_eps, magnitude, rms_weight_ptr, rms_eps);
  };

  if (use_pdl) {
    launch(std::true_type{});
  } else {
    launch(std::false_type{});
  }
}

template <int kHiddenDim, bool kFuseRmsNorm, bool kCastBfloatForNorm, bool kHasPost = true>
void launch_splitk_by_block_k(__nv_bfloat16* output_y_ptr, float* output_H_post_ptr,
                              const __nv_bfloat16* x_ptr, const float* w_ptr,
                              const float* hc_scale_ptr, const float* hc_base_ptr, int num_batch,
                              float norm_eps, float hc_eps, float magnitude, float* scratch_ptr,
                              cudaStream_t stream, bool use_pdl,
                              const __nv_bfloat16* rms_weight_ptr, float rms_eps) {
  auto launch = [&](auto block_k_tag) {
    launch_splitk<kHiddenDim, decltype(block_k_tag)::value, kFuseRmsNorm, kCastBfloatForNorm,
                  kHasPost>(output_y_ptr, output_H_post_ptr, x_ptr, w_ptr, hc_scale_ptr,
                            hc_base_ptr, num_batch, norm_eps, hc_eps, magnitude, scratch_ptr,
                            stream, use_pdl, rms_weight_ptr, rms_eps);
  };

  switch (kHasPost ? splitk_block_k(num_batch) : head_splitk_block_k(num_batch, kHiddenDim)) {
    case 1024:
      launch(std::integral_constant<int, 1024>{});
      break;
    case 2048:
      launch(std::integral_constant<int, 2048>{});
      break;
    default:
      launch(std::integral_constant<int, 4096>{});
      break;
  }
}

}  // namespace

bool ihc_pre_splitk_supported(int num_batch, int hc_mult, int hidden_dim, bool fuse_rms_norm) {
  if (hc_mult != 4) {
    return false;
  }
  // The folded stage2 puts the whole row in one block, so the thread count
  // must be a legal block size.
  if (fuse_rms_norm && hidden_dim / kElementsPerThread > 1024) {
    return false;
  }
  if (hidden_dim != 3072 && hidden_dim != 4096 && hidden_dim != 6144) {
    return false;
  }
  // Check shape before threshold: threshold scales with hidden_dim.
  return num_batch > 0 && num_batch < splitk_token_threshold(hidden_dim, fuse_rms_norm);
}

size_t ihc_pre_splitk_scratch_floats(int num_batch, int hc_mult, int hidden_dim,
                                     bool fuse_rms_norm) {
  if (!ihc_pre_splitk_supported(num_batch, hc_mult, hidden_dim, fuse_rms_norm)) {
    return 0;
  }
  const int num_splits = (hc_mult * hidden_dim) / splitk_block_k(num_batch);
  return static_cast<size_t>(num_batch) * num_splits * (1 + 2 * hc_mult);
}

void fuse_ihc_pre_splitk_async(__nv_bfloat16* output_y_ptr, float* output_H_post_ptr,
                               const __nv_bfloat16* x_ptr, const float* w_ptr,
                               const float* hc_scale_ptr, const float* hc_base_ptr, int num_batch,
                               int hc_mult, int hidden_dim, float norm_eps, float hc_eps,
                               float magnitude, float* scratch_ptr, cudaStream_t stream,
                               bool use_pdl, const __nv_bfloat16* rms_weight_ptr, float rms_eps,
                               bool cast_bfloat_for_norm) {
  auto by_hidden = [&](auto hidden_tag) {
    constexpr int kHidden = decltype(hidden_tag)::value;
    if (rms_weight_ptr == nullptr) {
      launch_splitk_by_block_k<kHidden, false, false>(
          output_y_ptr, output_H_post_ptr, x_ptr, w_ptr, hc_scale_ptr, hc_base_ptr, num_batch,
          norm_eps, hc_eps, magnitude, scratch_ptr, stream, use_pdl, rms_weight_ptr, rms_eps);
    } else if (cast_bfloat_for_norm) {
      launch_splitk_by_block_k<kHidden, true, true>(
          output_y_ptr, output_H_post_ptr, x_ptr, w_ptr, hc_scale_ptr, hc_base_ptr, num_batch,
          norm_eps, hc_eps, magnitude, scratch_ptr, stream, use_pdl, rms_weight_ptr, rms_eps);
    } else {
      launch_splitk_by_block_k<kHidden, true, false>(
          output_y_ptr, output_H_post_ptr, x_ptr, w_ptr, hc_scale_ptr, hc_base_ptr, num_batch,
          norm_eps, hc_eps, magnitude, scratch_ptr, stream, use_pdl, rms_weight_ptr, rms_eps);
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

bool ihc_head_splitk_supported(int num_batch, int hc_mult, int hidden_dim, bool fuse_rms_norm) {
  if (hc_mult != 4) {
    return false;
  }
  if (fuse_rms_norm && hidden_dim / kElementsPerThread > 1024) {
    return false;
  }
  if (hidden_dim != 3072 && hidden_dim != 4096 && hidden_dim != 6144) {
    return false;
  }
  return num_batch > 0 && num_batch < head_splitk_token_threshold(hidden_dim, fuse_rms_norm);
}

size_t ihc_head_splitk_scratch_floats(int num_batch, int hc_mult, int hidden_dim,
                                      bool fuse_rms_norm) {
  if (!ihc_head_splitk_supported(num_batch, hc_mult, hidden_dim, fuse_rms_norm)) {
    return 0;
  }
  const int num_splits = (hc_mult * hidden_dim) / head_splitk_block_k(num_batch, hidden_dim);
  // One gate group, so the slot is 1 + hc floats rather than pre's 1 + 2*hc.
  return static_cast<size_t>(num_batch) * num_splits * (1 + hc_mult);
}

void fuse_ihc_head_splitk_async(__nv_bfloat16* output_ptr, const __nv_bfloat16* x_ptr,
                                const float* w_ptr, const float* hc_scale_ptr,
                                const float* hc_base_ptr, int num_batch, int hc_mult,
                                int hidden_dim, float norm_eps, float hc_eps, float* scratch_ptr,
                                cudaStream_t stream, bool use_pdl,
                                const __nv_bfloat16* rms_weight_ptr, float rms_eps,
                                bool cast_bfloat_for_norm) {
  // kHasPost = false: head has no post gates; the specialization drops the write.
  auto by_hidden = [&](auto hidden_tag) {
    constexpr int kHidden = decltype(hidden_tag)::value;
    if (rms_weight_ptr == nullptr) {
      launch_splitk_by_block_k<kHidden, false, false, false>(
          output_ptr, nullptr, x_ptr, w_ptr, hc_scale_ptr, hc_base_ptr, num_batch, norm_eps, hc_eps,
          0.f, scratch_ptr, stream, use_pdl, rms_weight_ptr, rms_eps);
    } else if (cast_bfloat_for_norm) {
      launch_splitk_by_block_k<kHidden, true, true, false>(
          output_ptr, nullptr, x_ptr, w_ptr, hc_scale_ptr, hc_base_ptr, num_batch, norm_eps, hc_eps,
          0.f, scratch_ptr, stream, use_pdl, rms_weight_ptr, rms_eps);
    } else {
      launch_splitk_by_block_k<kHidden, true, false, false>(
          output_ptr, nullptr, x_ptr, w_ptr, hc_scale_ptr, hc_base_ptr, num_batch, norm_eps, hc_eps,
          0.f, scratch_ptr, stream, use_pdl, rms_weight_ptr, rms_eps);
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

}  // namespace splitk
}  // namespace iHC
}  // namespace hpc
