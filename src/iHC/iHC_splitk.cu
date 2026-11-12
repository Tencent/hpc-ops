// Copyright 2026 Tencent

// Host dispatch for the split-K fuse_ihc_pre / fuse_ihc_head implementation.
// Routes by token count; defers to the row-per-block kernel once its grid is large enough.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "src/iHC/iHC.h"
#include "src/iHC/iHC_splitk.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace iHC {
namespace splitk {

namespace {

constexpr int kStage1Threads = 128;
constexpr int kStage2Threads = 256;
constexpr int kElementsPerThread = 8;

// Token count at or above which this path declines to the row-per-block kernel.
#ifndef HPC_IHC_SPLITK_FORCE
#define HPC_IHC_SPLITK_FORCE 0
#endif

// Forced builds override head's threshold too, for calibration runs.
#ifndef HPC_IHC_HEAD_SPLITK_FORCE
#define HPC_IHC_HEAD_SPLITK_FORCE 0
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
  // Constants declared here so forced builds don't carry unreferenced vars (#177-D).
  constexpr float kSmPerBlockRow = 1.2f;
  constexpr float kFoldScale = 1.2f;
  constexpr int kHiddenRef = 6144;

  const int sm_count = get_sm_count();
  if (sm_count <= 0) {
    return 0;
  }
  // sqrt rather than linear: row path's cost per block grows with row width, so
  // a wider row moves the crossing out sub-proportionally.
  const float hidden_scale = sqrtf(static_cast<float>(hidden_dim) / kHiddenRef);
  const float fold_scale = fuse_rms_norm ? kFoldScale : 1.0f;
  return static_cast<int>(sm_count * kSmPerBlockRow * hidden_scale * fold_scale);
#endif
}

constexpr int splitk_block_k(int num_batch) {
#if HPC_TARGET_ARCH == 100 || HPC_TARGET_ARCH == 103
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

constexpr int head_splitk_block_k(int num_batch, int hidden_dim) {
#if HPC_TARGET_ARCH == 100 || HPC_TARGET_ARCH == 103
  if (hidden_dim == 6144) {
    if (num_batch < 16) {
      return 1024;
    }
    return num_batch < 96 ? 2048 : 4096;
  }
  // hidden 4096.
  if (num_batch < 48) {
    return 1024;
  }
  return num_batch < 160 ? 2048 : 4096;
#else
  (void)hidden_dim;
  if (num_batch < 2) {
    return 1024;
  }
  if (num_batch < 48) {
    return 2048;
  }
  return 4096;
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
  // Fold moves the crossing in; 0.8 errs toward the row path.
  constexpr float kFoldScale = 0.8f;
  constexpr int kHiddenRef = 6144;

  const int sm_count = get_sm_count();
  if (sm_count <= 0) {
    return 0;
  }
  // Linear in hidden, not sqrt as pre's is.
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

    // Slice the hidden dim across blocks; folded launches cannot slice
    constexpr int kStage2ThreadsHere =
        kFuseRmsNorm ? kHiddenDim / kElementsPerThread : kStage2Threads;
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
  // Tag-dispatched over BLOCK_K.
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
  if (hidden_dim != 4096 && hidden_dim != 6144) {
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
  // The fold/cast nesting is written once and reused per hidden, rather than
  // once per hidden as a macro expansion.
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
  if (hidden_dim != 4096 && hidden_dim != 6144) {
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
  } else {
    return;
  }
}

}  // namespace splitk
}  // namespace iHC
}  // namespace hpc
