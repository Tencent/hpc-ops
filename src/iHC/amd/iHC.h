// Copyright (C) 2026 Tencent.
#ifndef SRC_IHC_AMD_IHC_H_
#define SRC_IHC_AMD_IHC_H_

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

namespace hpc::iHC::amd {

void pre_head_async(hip_bfloat16* y, float* post, const hip_bfloat16* x, const float* w,
                    const float* scale, const float* base, int batch, int hidden, float norm_eps,
                    float hc_eps, float magnitude, const hip_bfloat16* rms_weight, float rms_eps,
                    bool cast_bfloat_for_norm, hipStream_t stream);
void post_async(hip_bfloat16* y, const hip_bfloat16* x, const hip_bfloat16* residual,
                const float* post, int batch, int hidden, hipStream_t stream);

// TODO(amd): fuse_ihc_post_pre's single-kernel and two-stage implementations.
// This is a non-FP8 SIMT path, deferred from the first batch rather than unsupported hardware.

}  // namespace hpc::iHC::amd
#endif  // SRC_IHC_AMD_IHC_H_
