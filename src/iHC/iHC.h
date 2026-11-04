// Copyright 2026 Tencent

#ifndef SRC_IHC_IHC_H_
#define SRC_IHC_IHC_H_

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

namespace hpc {
namespace iHC {

void fuse_ihc_pre_async(__nv_bfloat16* output_y_ptr, float* output_H_post_ptr,
                        const __nv_bfloat16* x_ptr, const float* w_ptr, const float* hc_scale_ptr,
                        const float* hc_base_ptr, int num_batch, int hc_mult, int hidden_dim,
                        float norm_eps, float hc_eps, float magnitude, cudaStream_t stream,
                        const __nv_bfloat16* rms_weight_ptr = nullptr, float rms_eps = 0.f,
                        bool cast_bfloat_for_norm = false, bool use_pdl = true);

void fuse_ihc_post_pre_async(__nv_bfloat16* y_ptr, __nv_bfloat16* z_ptr, float* H_post_out_ptr,
                             const __nv_bfloat16* xa_ptr, const __nv_bfloat16* residual_ptr,
                             const float* H_post_in_ptr, const float* w_ptr,
                             const float* hc_scale_ptr, const float* hc_base_ptr, int num_batch,
                             int hc_mult, int hidden_dim, float norm_eps, float hc_eps,
                             float magnitude, float* scratch_ptr, cudaStream_t stream,
                             const __nv_bfloat16* rms_weight_ptr = nullptr, float rms_eps = 0.f,
                             bool cast_bfloat_for_norm = false, bool use_pdl = true);

size_t ihc_post_pre_scratch_floats(int num_batch, int hc_mult, int hidden_dim);

void fuse_ihc_post_async(__nv_bfloat16* output_ptr, const __nv_bfloat16* x_ptr,
                         const __nv_bfloat16* residual_ptr, const float* H_post_ptr, int num_batch,
                         int hc_mult, int hidden_dim, cudaStream_t stream, bool use_pdl = true);

void fuse_ihc_head_async(__nv_bfloat16* output_ptr, const __nv_bfloat16* x_ptr, const float* w_ptr,
                         const float* hc_scale_ptr, const float* hc_base_ptr, int num_batch,
                         int hc_mult, int hidden_dim, float norm_eps, float hc_eps,
                         cudaStream_t stream, const __nv_bfloat16* rms_weight_ptr = nullptr,
                         float rms_eps = 0.f, bool cast_bfloat_for_norm = false,
                         bool use_pdl = true);

// A split-K pre path for batches with too few tokens to fill the device, where
// fuse_ihc_pre_async's grid of num_batch leaves most SMs idle. The selector is
// the token count against an SM-count-scaled threshold, not the inference
// phase -- a short prefill or a speculative-decode verify step qualifies on the
// same terms as a decode step.
//
// The kernels are portable across KNOWN_ARCHS but the crossover is not, so the
// threshold is derived per device; ihc_pre_splitk_supported() answers for the
// module it was built into, so the caller needs no #if of its own.
namespace splitk {

bool ihc_pre_splitk_supported(int num_batch, int hc_mult, int hidden_dim, bool fuse_rms_norm);

size_t ihc_pre_splitk_scratch_floats(int num_batch, int hc_mult, int hidden_dim,
                                     bool fuse_rms_norm);

void fuse_ihc_pre_splitk_async(__nv_bfloat16* output_y_ptr, float* output_H_post_ptr,
                               const __nv_bfloat16* x_ptr, const float* w_ptr,
                               const float* hc_scale_ptr, const float* hc_base_ptr, int num_batch,
                               int hc_mult, int hidden_dim, float norm_eps, float hc_eps,
                               float magnitude, float* scratch_ptr, cudaStream_t stream,
                               bool use_pdl = true, const __nv_bfloat16* rms_weight_ptr = nullptr,
                               float rms_eps = 0.f, bool cast_bfloat_for_norm = false);

}  // namespace splitk

}  // namespace iHC
}  // namespace hpc

#endif  // SRC_IHC_IHC_H_
