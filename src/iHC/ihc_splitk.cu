// Copyright 2026 Tencent

// Host dispatch for the split-K fuse_ihc_pre implementation.
//
// Routes by token count, deferring to the row-per-block kernel once its grid is
// large enough. The kernels are portable (see ihc_splitk.cuh); the threshold
// and BLOCK_K tables below are not.
//
// H20, hidden 6144, bf16, CUDA Graph, us per call vs the Triton versions:
//
//     tokens    vllm triton    flaggems triton    hpc (split-K)
//          1           4.20               3.87               3.18
//         16           6.64               5.28               5.10
//         64          16.91               8.52              10.51

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "src/iHC/iHC.h"
#include "src/iHC/ihc_splitk.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace iHC {
namespace splitk {

namespace {

constexpr int kStage1Threads = 128;
constexpr int kStage2Threads = 256;
constexpr int kElementsPerThread = 8;

// Token count at or above which this path declines to the row-per-block kernel.
//
// Modelled from the device, not tabulated: the row path cannot fill the machine
// until num_batch/rows passes the SM count, so the crossing tracks that count.
//
//     threshold ~ sm_count * k * sqrt(hidden/6144) * (fold ? 1.2 : 1)
//
// Fitted, not guessed -- measured by building twice with HPC_IHC_SPLITK_FORCE
// (1 = always, 2 = never), the only way to compare paths the operator otherwise
// routes between by this very threshold:
//
//     device         SMs   hidden   fold   crossover   implied k
//     H20 (sm90)      78     6144     no          96        1.23
//     B200 (sm100)   148     6144     no         176        1.19
//     B200           148     4096     no         160        1.32
//     B200           148     6144    yes         208           -
//     B200           148     4096    yes         192           -
//
// k = 1.2 covers the no-fold rows within 9%. The fold gets its own factor
// because it slows the row path, not this one, and both hidden sizes agree on
// how much (208/176 = 1.18, 192/160 = 1.20). The fit lands within the sweep's
// own spacing everywhere but H20, which predates the row path's retuning.
//
// A new architecture therefore gets a usable boundary with no table edit.
//
// Defaulted so a misspelled -D is a compile error, not a silently ignored
// override -- `#if HPC_IHC_SPLITK_FORCE == 1` would read an undefined macro as
// 0 and quietly take the normal path, fatal to a calibration run.
#ifndef HPC_IHC_SPLITK_FORCE
#define HPC_IHC_SPLITK_FORCE 0
#endif

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
  // Declared here rather than at namespace scope so a forced build does not
  // carry unreferenced constants, which -Werror=all-warnings rejects (#177-D).
  constexpr float kSmPerBlockRow = 1.2f;
  constexpr float kFoldScale = 1.2f;
  constexpr int kHiddenRef = 6144;

  const int sm_count = get_sm_count();
  if (sm_count <= 0) {
    return 0;
  }
  // sqrt rather than a linear term: the row path's cost per block grows with
  // the row width while its block count does not, so a wider row moves the
  // crossing out by less than proportionally.
  const float hidden_scale = sqrtf(static_cast<float>(hidden_dim) / kHiddenRef);
  const float fold_scale = fuse_rms_norm ? kFoldScale : 1.0f;
  return static_cast<int>(sm_count * kSmPerBlockRow * hidden_scale * fold_scale);
#endif
}

// BLOCK_K: how finely stage1 cuts K. A table rather than a model because the
// measurements do not support one -- B200 wants the finest chunk at 32 tokens
// (5.2 waves) and the coarsest at 64 (2.6), so the preference is not monotonic
// in occupancy and no single threshold on it can be right.
//
// Measured at hidden 6144, forcing each BLOCK_K (us per call, best marked):
//
//   H20 (sm90, 78 SM)                B200 (sm100, 148 SM)
//   tokens   1024   2048   4096      tokens   1024   2048   4096
//        1   3.76*  4.00   5.37           1   4.06*  4.78   5.72
//        4   4.24   4.20*  5.55          16   4.99*  5.40   5.94
//       32   7.85   7.08*  7.88          32   6.24*  6.34   7.07
//       63  12.47  11.63  11.09*         64   8.59   8.06   7.96*
//                                       128  13.12  11.84  11.52*
//
// B200 holds 1024 to 32 tokens where H20 only wins below 2, and never picks
// 2048. Carrying H20's table over costs 1.17x at one token, hence per-arch.
//
// An unmeasured architecture runs the sm90 table while the threshold model
// already lets it onto this path -- deliberate, since the path itself is worth
// 2.9x against the tables' 1.17x spread, so defaulting in with an imperfect
// BLOCK_K beats defaulting out. Calibrating adds a branch, not a code path.
// Candidates must divide hc_mult * hidden_dim and be a multiple of 1024.
constexpr int splitk_block_k(int num_batch) {
#if HPC_TARGET_ARCH == 100
  return num_batch < 48 ? 1024 : 4096;
#else
  if (num_batch < 2) {
    return 1024;
  }
  if (num_batch < 48) {
    return 2048;
  }
  return 4096;
#endif
}

template <int kHiddenDim, int kBlockK, bool kFuseRmsNorm = false, bool kCastBfloatForNorm = false>
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
                                               kStage1Threads, kBlockK, kUsePDL>;
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

    // Slice the hidden dim across blocks too: at one token a single block over
    // all 6144 lanes measured 4.9us against 1.7us spread over six. A folded
    // launch cannot slice (sum(y^2) spans the row) and uses one wide block per
    // token, which is why kStage2Threads is not a constant here.
    constexpr int kStage2ThreadsHere =
        kFuseRmsNorm ? kHiddenDim / kElementsPerThread : kStage2Threads;
    constexpr int kBlockD = kStage2ThreadsHere * kElementsPerThread;
    constexpr int kHiddenBlocks = kHiddenDim / kBlockD;
    static_assert(kStage2ThreadsHere <= 1024, "stage2 block exceeds the CUDA limit");

    auto stage2 =
        ihc_pre_stage2_splitk_kernel<kHCMult, kHiddenDim, kElementsPerThread, kStage2ThreadsHere,
                                     kNumSplits, kUsePDL, kFuseRmsNorm, kCastBfloatForNorm>;
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

template <int kHiddenDim, bool kFuseRmsNorm, bool kCastBfloatForNorm>
void launch_splitk_by_block_k(__nv_bfloat16* output_y_ptr, float* output_H_post_ptr,
                              const __nv_bfloat16* x_ptr, const float* w_ptr,
                              const float* hc_scale_ptr, const float* hc_base_ptr, int num_batch,
                              float norm_eps, float hc_eps, float magnitude, float* scratch_ptr,
                              cudaStream_t stream, bool use_pdl,
                              const __nv_bfloat16* rms_weight_ptr, float rms_eps) {
#define HPC_IHC_LAUNCH_SPLITK(block_k)                                                     \
  launch_splitk<kHiddenDim, block_k, kFuseRmsNorm, kCastBfloatForNorm>(                    \
      output_y_ptr, output_H_post_ptr, x_ptr, w_ptr, hc_scale_ptr, hc_base_ptr, num_batch, \
      norm_eps, hc_eps, magnitude, scratch_ptr, stream, use_pdl, rms_weight_ptr, rms_eps)

  switch (splitk_block_k(num_batch)) {
    case 1024:
      HPC_IHC_LAUNCH_SPLITK(1024);
      break;
#if HPC_TARGET_ARCH != 100
    // Only instantiated where the table can return it: B200's table never
    // picks 2048, and an arm no call reaches is still compiled into the module.
    case 2048:
      HPC_IHC_LAUNCH_SPLITK(2048);
      break;
#endif
    default:
      HPC_IHC_LAUNCH_SPLITK(4096);
      break;
  }
#undef HPC_IHC_LAUNCH_SPLITK
}

}  // namespace

bool ihc_pre_splitk_supported(int num_batch, int hc_mult, int hidden_dim, bool fuse_rms_norm) {
  if (hc_mult != 4) {
    return false;
  }
  // The folded stage2 puts the whole row in one block, so hidden_dim /
  // kElementsPerThread threads must be a legal block size.
  if (fuse_rms_norm && hidden_dim / kElementsPerThread > 1024) {
    return false;
  }
  if (hidden_dim != 4096 && hidden_dim != 6144) {
    return false;
  }
  // Shape first: the threshold scales with hidden_dim, so it means nothing
  // until the size is known to be one this path supports.
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
#define HPC_IHC_SPLITK_BY_HIDDEN(hidden, fuse, cast)                                       \
  launch_splitk_by_block_k<hidden, fuse, cast>(                                            \
      output_y_ptr, output_H_post_ptr, x_ptr, w_ptr, hc_scale_ptr, hc_base_ptr, num_batch, \
      norm_eps, hc_eps, magnitude, scratch_ptr, stream, use_pdl, rms_weight_ptr, rms_eps)

  const bool fuse_norm = rms_weight_ptr != nullptr;
  if (hidden_dim == 6144) {
    if (!fuse_norm) {
      HPC_IHC_SPLITK_BY_HIDDEN(6144, false, false);
    } else if (cast_bfloat_for_norm) {
      HPC_IHC_SPLITK_BY_HIDDEN(6144, true, true);
    } else {
      HPC_IHC_SPLITK_BY_HIDDEN(6144, true, false);
    }
  } else if (hidden_dim == 4096) {
    if (!fuse_norm) {
      HPC_IHC_SPLITK_BY_HIDDEN(4096, false, false);
    } else if (cast_bfloat_for_norm) {
      HPC_IHC_SPLITK_BY_HIDDEN(4096, true, true);
    } else {
      HPC_IHC_SPLITK_BY_HIDDEN(4096, true, false);
    }
  }
#undef HPC_IHC_SPLITK_BY_HIDDEN
}

}  // namespace splitk
}  // namespace iHC
}  // namespace hpc
