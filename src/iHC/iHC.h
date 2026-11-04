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

// sm90 only: a K-split pre path for decode-shaped batches. The shared
// row-per-block kernel above has a grid of exactly num_batch, which cannot fill
// an sm90 device at small token counts (latency-flat ~12-13us from 1 to 64
// tokens on H20); splitting K instead exposes enough blocks to scale down.
// ihc_pre_decode_supported() reports whether this path claims a given shape --
// it declines at and above the measured crossover, where the shared kernel wins.
namespace sm90 {

bool ihc_pre_decode_supported(int num_batch, int hc_mult, int hidden_dim);

size_t ihc_pre_decode_scratch_floats(int num_batch, int hc_mult, int hidden_dim);

void fuse_ihc_pre_decode_async(__nv_bfloat16* output_y_ptr, float* output_H_post_ptr,
                               const __nv_bfloat16* x_ptr, const float* w_ptr,
                               const float* hc_scale_ptr, const float* hc_base_ptr, int num_batch,
                               int hc_mult, int hidden_dim, float norm_eps, float hc_eps,
                               float magnitude, float* scratch_ptr, cudaStream_t stream,
                               bool use_pdl = true);

}  // namespace sm90

}  // namespace iHC
}  // namespace hpc

#endif  // SRC_IHC_IHC_H_
