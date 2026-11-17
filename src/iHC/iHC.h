// Copyright 2026 Tencent
// Copyright 2026, The FlagOS Contributors.

#ifndef SRC_IHC_IHC_H_
#define SRC_IHC_IHC_H_

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

namespace hpc {
namespace iHC {

// Callers must validate hc_mult and hidden_dim; unsupported shapes launch nothing.
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

// Split-K pre dispatch uses a device-specific token threshold.
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

// Head shares the split-K kernels (kHasPost=false) with its own dispatch threshold.
bool ihc_head_splitk_supported(int num_batch, int hc_mult, int hidden_dim, bool fuse_rms_norm);

size_t ihc_head_splitk_scratch_floats(int num_batch, int hc_mult, int hidden_dim,
                                      bool fuse_rms_norm);

void fuse_ihc_head_splitk_async(__nv_bfloat16* output_ptr, const __nv_bfloat16* x_ptr,
                                const float* w_ptr, const float* hc_scale_ptr,
                                const float* hc_base_ptr, int num_batch, int hc_mult,
                                int hidden_dim, float norm_eps, float hc_eps, float* scratch_ptr,
                                cudaStream_t stream, bool use_pdl = true,
                                const __nv_bfloat16* rms_weight_ptr = nullptr, float rms_eps = 0.f,
                                bool cast_bfloat_for_norm = false);

}  // namespace splitk

}  // namespace iHC
}  // namespace hpc

#endif  // SRC_IHC_IHC_H_
