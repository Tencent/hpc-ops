// Copyright 2025 hpc-ops authors

#ifndef SRC_MHC_MHC_H_
#define SRC_MHC_MHC_H_

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

namespace hpc {
namespace mHC {

void reciprocal_mean_square_root_norm_async(__nv_bfloat16* output_ptr,
                                            const __nv_bfloat16* input_ptr, int num_batch,
                                            int hidden_dim, float norm_eps, cudaStream_t stream);

void fuse_cal_three_H_async(float* output_H_pre_ptr, float* output_H_post_ptr,
                            float* output_H_res_ptr, const float* mixes_hat_hat_H_ptr,
                            const float* hc_scale_ptr, const float* hc_base_ptr, int num_batch,
                            int hc_mult, int hc_sinkhorn_iters, float hc_eps, cudaStream_t stream);

void fuse_hc_pre_mapping_async(__nv_bfloat16* output_ptr, const __nv_bfloat16* x_ptr,
                               const float* H_pre_ptr, int num_batch, int hc_mult, int hidden_dim,
                               cudaStream_t stream);

void fuse_H_post_mapping_H_res_mapping_and_residual_add_async(
    __nv_bfloat16* output_ptr, const __nv_bfloat16* x_ptr, const __nv_bfloat16* residual_ptr,
    const float* H_post_ptr, const float* H_res_ptr, int num_batch, int hc_mult, int hidden_dim,
    cudaStream_t stream);

}  // namespace mHC
}  // namespace hpc

#endif  // SRC_MHC_MHC_H_
