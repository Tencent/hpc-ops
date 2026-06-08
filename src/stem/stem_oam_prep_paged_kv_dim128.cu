// Copyright (C) 2026 Tencent.

#include <cuda.h>

#include "cute/tensor.hpp"
#include "src/stem/stem_kernels.cuh"
#include "src/stem/stem_oam_prep_paged_kv_dim128.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace stem {

// Legacy qpertoken_perhead_kvpertensor path (quant_type=1).
template <int kBlockSize, int kStemBlockSize, int kStride, int kDimQK, int kDimV>
void launch_stem_oam_prep_paged_kv_qpertoken_perhead_kvpertensor_dim128(
    void *kflat_ptr, void *vbias_ptr, const void *kcache_ptr, const void *vcache_ptr,
    const void *kscale_ptr, const void *vscale_ptr, const void *block_ids_ptr,
    const void *kv_seq_lens_ptr, int num_batch, int num_dim_qk, int num_dim_v, int num_head_kv,
    int num_kvcache_blocks, int num_seq_max_blocks, float lambda_mag, int max_num_stem_blocks,
    int max_k_down_len, int ldK, int ldV, void *v_norm_down_ptr, cudaStream_t stream) {
  using namespace cute;  // NOLINT

  using Tfp8 = cute::float_e4m3_t;
  using Tbf16 = cute::bfloat16_t;

  // Sub-kernel 1: paged KV prep (K group-sum + V L2 norm)
  {
    constexpr int kSamplePerBlock = kStemBlockSize / kStride;
    dim3 grid(max_num_stem_blocks, num_batch * num_head_kv);
    dim3 block(kSamplePerBlock * 32);

    kernels::stem_prep_paged_kv_qpertoken_perhead_kvpertensor_kernel<
        kBlockSize, kStemBlockSize, kStride, kDimQK, kDimV><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_fp8_e4m3 *>(kcache_ptr),
        reinterpret_cast<const __nv_fp8_e4m3 *>(vcache_ptr),
        reinterpret_cast<const float *>(kscale_ptr), reinterpret_cast<const float *>(vscale_ptr),
        reinterpret_cast<const int *>(block_ids_ptr),
        reinterpret_cast<const int *>(kv_seq_lens_ptr),
        reinterpret_cast<__nv_bfloat16 *>(kflat_ptr), reinterpret_cast<float *>(v_norm_down_ptr),
        num_head_kv, num_seq_max_blocks, ldK, ldV, max_num_stem_blocks, max_k_down_len);
  }

  // Sub-kernel 2: Vbias reduce (shared between both quant_type paths).
  {
    dim3 grid(num_batch * num_head_kv);
    dim3 block(256);

    kernels::vbias_reduce_kernel<kStemBlockSize, kStride><<<grid, block, 0, stream>>>(
        reinterpret_cast<const float *>(v_norm_down_ptr),
        reinterpret_cast<const int *>(kv_seq_lens_ptr), reinterpret_cast<float *>(vbias_ptr),
        num_head_kv, max_k_down_len, max_num_stem_blocks, lambda_mag);
  }
}

void stem_oam_prep_paged_kv_qpertoken_perhead_kvpertensor_dim128_async(
    void *kflat_ptr, void *vbias_ptr, const void *kcache_ptr, const void *vcache_ptr,
    const void *kscale_ptr, const void *vscale_ptr, const void *block_ids_ptr,
    const void *kv_seq_lens_ptr, int num_batch, int num_dim_qk, int num_dim_v, int num_head_kv,
    int num_kvcache_blocks, int block_size, int num_seq_max_blocks, int stem_block_size,
    int stem_stride, float lambda_mag, int max_num_stem_blocks, int max_k_down_len, int ldK,
    int ldV, void *v_norm_down_ptr, cudaStream_t stream) {
  if (block_size == 32) {
    launch_stem_oam_prep_paged_kv_qpertoken_perhead_kvpertensor_dim128<32, 128, 16, 128, 128>(
        kflat_ptr, vbias_ptr, kcache_ptr, vcache_ptr, kscale_ptr, vscale_ptr, block_ids_ptr,
        kv_seq_lens_ptr, num_batch, num_dim_qk, num_dim_v, num_head_kv, num_kvcache_blocks,
        num_seq_max_blocks, lambda_mag, max_num_stem_blocks, max_k_down_len, ldK, ldV,
        v_norm_down_ptr, stream);
  } else if (block_size == 64) {
    launch_stem_oam_prep_paged_kv_qpertoken_perhead_kvpertensor_dim128<64, 128, 16, 128, 128>(
        kflat_ptr, vbias_ptr, kcache_ptr, vcache_ptr, kscale_ptr, vscale_ptr, block_ids_ptr,
        kv_seq_lens_ptr, num_batch, num_dim_qk, num_dim_v, num_head_kv, num_kvcache_blocks,
        num_seq_max_blocks, lambda_mag, max_num_stem_blocks, max_k_down_len, ldK, ldV,
        v_norm_down_ptr, stream);
  }
}

// New qkpertoken_perhead_vperhead path (quant_type=0).
template <int kBlockSize, int kStemBlockSize, int kStride, int kDimQK, int kDimV>
void launch_stem_oam_prep_paged_kv_qkpertoken_perhead_vperhead_dim128(
    void *kflat_ptr, void *vbias_ptr, const void *kcache_ptr, const void *vcache_ptr,
    const void *kscale_ptr, const void *vscale_ptr, const void *block_ids_ptr,
    const void *kv_seq_lens_ptr, int num_batch, int num_dim_qk, int num_dim_v, int num_head_kv,
    int num_kvcache_blocks, int num_seq_max_blocks, float lambda_mag, int max_num_stem_blocks,
    int max_k_down_len, int ldK, int ldV, int ldKS, int ldKS1, int ldKS2, void *v_norm_down_ptr,
    cudaStream_t stream) {
  using namespace cute;  // NOLINT

  using Tfp8 = cute::float_e4m3_t;
  using Tbf16 = cute::bfloat16_t;

  // Sub-kernel 1: paged KV prep with per-token K-scale + per-head V-scale.
  {
    constexpr int kSamplePerBlock = kStemBlockSize / kStride;
    dim3 grid(max_num_stem_blocks, num_batch * num_head_kv);
    dim3 block(kSamplePerBlock * 32);

    kernels::stem_prep_paged_kv_qkpertoken_perhead_vperhead_kernel<kBlockSize, kStemBlockSize,
                                                                   kStride, kDimQK, kDimV>
        <<<grid, block, 0, stream>>>(reinterpret_cast<const __nv_fp8_e4m3 *>(kcache_ptr),
                                     reinterpret_cast<const __nv_fp8_e4m3 *>(vcache_ptr),
                                     reinterpret_cast<const float *>(kscale_ptr),
                                     reinterpret_cast<const float *>(vscale_ptr),
                                     reinterpret_cast<const int *>(block_ids_ptr),
                                     reinterpret_cast<const int *>(kv_seq_lens_ptr),
                                     reinterpret_cast<__nv_bfloat16 *>(kflat_ptr),
                                     reinterpret_cast<float *>(v_norm_down_ptr), num_head_kv,
                                     num_seq_max_blocks, ldK, ldV, ldKS, ldKS1, ldKS2,
                                     max_num_stem_blocks, max_k_down_len);
  }

  // Sub-kernel 2: Vbias reduce (shared between both quant_type paths; log normalize
  // is invariant to the global per-head V-scale).
  {
    dim3 grid(num_batch * num_head_kv);
    dim3 block(256);

    kernels::vbias_reduce_kernel<kStemBlockSize, kStride><<<grid, block, 0, stream>>>(
        reinterpret_cast<const float *>(v_norm_down_ptr),
        reinterpret_cast<const int *>(kv_seq_lens_ptr), reinterpret_cast<float *>(vbias_ptr),
        num_head_kv, max_k_down_len, max_num_stem_blocks, lambda_mag);
  }
}

void stem_oam_prep_paged_kv_qkpertoken_perhead_vperhead_dim128_async(
    void *kflat_ptr, void *vbias_ptr, const void *kcache_ptr, const void *vcache_ptr,
    const void *kscale_ptr, const void *vscale_ptr, const void *block_ids_ptr,
    const void *kv_seq_lens_ptr, int num_batch, int num_dim_qk, int num_dim_v, int num_head_kv,
    int num_kvcache_blocks, int block_size, int scale_block_size, int num_seq_max_blocks,
    int stem_block_size, int stem_stride, float lambda_mag, int max_num_stem_blocks,
    int max_k_down_len, int ldK, int ldV, int ldKS, int ldKS1, int ldKS2, void *v_norm_down_ptr,
    cudaStream_t stream) {
  if (block_size == 32) {
    launch_stem_oam_prep_paged_kv_qkpertoken_perhead_vperhead_dim128<32, 128, 16, 128, 128>(
        kflat_ptr, vbias_ptr, kcache_ptr, vcache_ptr, kscale_ptr, vscale_ptr, block_ids_ptr,
        kv_seq_lens_ptr, num_batch, num_dim_qk, num_dim_v, num_head_kv, num_kvcache_blocks,
        num_seq_max_blocks, lambda_mag, max_num_stem_blocks, max_k_down_len, ldK, ldV, ldKS, ldKS1,
        ldKS2, v_norm_down_ptr, stream);
  } else if (block_size == 64) {
    launch_stem_oam_prep_paged_kv_qkpertoken_perhead_vperhead_dim128<64, 128, 16, 128, 128>(
        kflat_ptr, vbias_ptr, kcache_ptr, vcache_ptr, kscale_ptr, vscale_ptr, block_ids_ptr,
        kv_seq_lens_ptr, num_batch, num_dim_qk, num_dim_v, num_head_kv, num_kvcache_blocks,
        num_seq_max_blocks, lambda_mag, max_num_stem_blocks, max_k_down_len, ldK, ldV, ldKS, ldKS1,
        ldKS2, v_norm_down_ptr, stream);
  }
}

}  // namespace stem
}  // namespace hpc
