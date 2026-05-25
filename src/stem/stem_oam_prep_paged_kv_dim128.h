// Copyright (C) 2026 Tencent.

#ifndef SRC_STEM_STEM_OAM_PREP_PAGED_KV_DIM128_H_
#define SRC_STEM_STEM_OAM_PREP_PAGED_KV_DIM128_H_

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace hpc {
namespace stem {

// Legacy quant_type=1 path: per-tensor K-scale + per-tensor V-scale.
void stem_oam_prep_paged_kv_qpertoken_perhead_kvpertensor_dim128_async(
    void *kflat_ptr, void *vbias_ptr, const void *kcache_ptr, const void *vcache_ptr,
    const void *kscale_ptr, const void *vscale_ptr, const void *block_ids_ptr,
    const void *kv_seq_lens_ptr, int num_batch, int num_dim_qk, int num_dim_v, int num_head_kv,
    int num_kvcache_blocks, int block_size, int num_seq_max_blocks, int stem_block_size,
    int stem_stride, float lambda_mag, int max_num_stem_blocks, int max_k_down_len, int ldK,
    int ldV, void *v_norm_down_ptr, cudaStream_t stream);

// New quant_type=0 path: per-token K-scale (4D fp32, strides ldKS/ldKS1/ldKS2 in fp32 elements)
// + per-head V-scale (vscale_ptr is a vector of length num_head_kv).
void stem_oam_prep_paged_kv_qkpertoken_perhead_vperhead_dim128_async(
    void *kflat_ptr, void *vbias_ptr, const void *kcache_ptr, const void *vcache_ptr,
    const void *kscale_ptr, const void *vscale_ptr, const void *block_ids_ptr,
    const void *kv_seq_lens_ptr, int num_batch, int num_dim_qk, int num_dim_v, int num_head_kv,
    int num_kvcache_blocks, int block_size, int scale_block_size, int num_seq_max_blocks,
    int stem_block_size, int stem_stride, float lambda_mag, int max_num_stem_blocks,
    int max_k_down_len, int ldK, int ldV, int ldKS, int ldKS1, int ldKS2, void *v_norm_down_ptr,
    cudaStream_t stream);

}  // namespace stem
}  // namespace hpc

#endif  // SRC_STEM_STEM_OAM_PREP_PAGED_KV_DIM128_H_
