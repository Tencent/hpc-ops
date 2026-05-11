// Copyright 2025 hpc-ops authors

#include <cuda.h>

#include "src/stem/stem_kernels.cuh"
#include "src/stem/stem_oam_prep_varlen_kv_dim192.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace stem {

template <int kStemBlockSize, int kStride, int kDimQK, int kDimV>
void launch_stem_oam_prep_varlen_kv_dim192(void *kflat_ptr, void *vbias_ptr, const void *k_fp8_ptr,
                                           const void *v_fp8_ptr, const void *kscale_ptr,
                                           const void *vscale_ptr, const void *kv_seq_lens_ptr,
                                           const void *cu_seqlens_kv_ptr, int num_batch,
                                           int num_head_kv, int ldK, int ldV, float lambda_mag,
                                           int max_num_stem_blocks, int max_k_down_len,
                                           void *v_norm_down_ptr, cudaStream_t stream) {
  // Sub-kernel 1: varlen KV prep (K group-sum + V L2 norm)
  {
    constexpr int kSamplePerBlock = kStemBlockSize / kStride;
    dim3 grid(max_num_stem_blocks, num_batch * num_head_kv);
    dim3 block(kSamplePerBlock * 32);

    kernels::stem_prep_varlen_kv_kernel<kStemBlockSize, kStride, kDimQK, kDimV>
        <<<grid, block, 0, stream>>>(reinterpret_cast<const __nv_fp8_e4m3 *>(k_fp8_ptr),
                                     reinterpret_cast<const __nv_fp8_e4m3 *>(v_fp8_ptr),
                                     reinterpret_cast<const float *>(kscale_ptr),
                                     reinterpret_cast<const float *>(vscale_ptr),
                                     reinterpret_cast<const int *>(kv_seq_lens_ptr),
                                     reinterpret_cast<const int *>(cu_seqlens_kv_ptr),
                                     reinterpret_cast<__nv_bfloat16 *>(kflat_ptr),
                                     reinterpret_cast<float *>(v_norm_down_ptr), num_head_kv, ldK,
                                     ldV, max_num_stem_blocks, max_k_down_len);
  }

  // Sub-kernel 2: Vbias reduce
  {
    dim3 grid(num_batch * num_head_kv);
    dim3 block(256);

    kernels::vbias_reduce_kernel<kStemBlockSize, kStride><<<grid, block, 0, stream>>>(
        reinterpret_cast<const float *>(v_norm_down_ptr),
        reinterpret_cast<const int *>(kv_seq_lens_ptr), reinterpret_cast<float *>(vbias_ptr),
        num_head_kv, max_k_down_len, max_num_stem_blocks, lambda_mag);
  }
}

void stem_oam_prep_varlen_kv_dim192_async(void *kflat_ptr, void *vbias_ptr, const void *k_fp8_ptr,
                                          const void *v_fp8_ptr, const void *kscale_ptr,
                                          const void *vscale_ptr, const void *kv_seq_lens_ptr,
                                          const void *cu_seqlens_kv_ptr, int num_batch,
                                          int num_head_kv, int ldK, int ldV, float lambda_mag,
                                          int max_num_stem_blocks, int max_k_down_len,
                                          void *v_norm_down_ptr, cudaStream_t stream) {
  launch_stem_oam_prep_varlen_kv_dim192<128, 16, 192, 128>(
      kflat_ptr, vbias_ptr, k_fp8_ptr, v_fp8_ptr, kscale_ptr, vscale_ptr, kv_seq_lens_ptr,
      cu_seqlens_kv_ptr, num_batch, num_head_kv, ldK, ldV, lambda_mag, max_num_stem_blocks,
      max_k_down_len, v_norm_down_ptr, stream);
}

}  // namespace stem
}  // namespace hpc
