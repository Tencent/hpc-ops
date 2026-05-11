// Copyright 2025 hpc-ops authors

#ifndef SRC_STEM_STEM_OAM_PREP_VARLEN_Q_DIM128_H_
#define SRC_STEM_STEM_OAM_PREP_VARLEN_Q_DIM128_H_

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace hpc {
namespace stem {

void stem_oam_prep_varlen_q_dim128_async(void *qflat_ptr, const void *q_fp8_ptr,
                                         const void *qscale_ptr, const void *q_seq_lens_ptr,
                                         const void *cu_seqlens_q_ptr, int num_batch,
                                         int num_head_q, int ldQ, int stem_block_size,
                                         int stem_stride, int max_num_q_blocks,
                                         cudaStream_t stream);

}  // namespace stem
}  // namespace hpc

#endif  // SRC_STEM_STEM_OAM_PREP_VARLEN_Q_DIM128_H_
