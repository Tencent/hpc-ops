// Copyright 2025 hpc-ops authors

#ifndef SRC_STEM_STEM_OAM_GEMM_DIM192_H_
#define SRC_STEM_STEM_OAM_GEMM_DIM192_H_

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace hpc {
namespace stem {

void stem_oam_gemm_dim192_async(void *block_logits_ptr, const void *qflat_ptr,
                                const void *kflat_ptr, const void *vbias_ptr,
                                const void *q_seq_lens_ptr, const void *kv_seq_lens_ptr,
                                int num_batch, int num_head_q, int num_head_kv, int max_num_qb,
                                int max_num_kb, int stem_block_size, int stem_stride, bool causal,
                                cudaStream_t stream);

}  // namespace stem
}  // namespace hpc

#endif  // SRC_STEM_STEM_OAM_GEMM_DIM192_H_
