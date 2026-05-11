// Copyright 2025 hpc-ops authors

#ifndef SRC_STEM_STEM_OAM_PREP_VARLEN_KV_DIM192_H_
#define SRC_STEM_STEM_OAM_PREP_VARLEN_KV_DIM192_H_

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace hpc {
namespace stem {

void stem_oam_prep_varlen_kv_dim192_async(void *kflat_ptr, void *vbias_ptr, const void *k_fp8_ptr,
                                          const void *v_fp8_ptr, const void *kscale_ptr,
                                          const void *vscale_ptr, const void *kv_seq_lens_ptr,
                                          const void *cu_seqlens_kv_ptr, int num_batch,
                                          int num_head_kv, int ldK, int ldV, float lambda_mag,
                                          int max_num_stem_blocks, int max_k_down_len,
                                          void *v_norm_down_ptr, cudaStream_t stream);

}  // namespace stem
}  // namespace hpc

#endif  // SRC_STEM_STEM_OAM_PREP_VARLEN_KV_DIM192_H_
