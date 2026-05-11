// Copyright 2025 hpc-ops authors

#include <cuda.h>

#include "src/stem/stem_kernels.cuh"
#include "src/stem/stem_oam_prep_varlen_q_dim192.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace stem {

template <int kStemBlockSize, int kStride, int kDimQK, bool kIsPerTensorQscale>
void launch_stem_oam_prep_varlen_q_dim192(void *qflat_ptr, const void *q_fp8_ptr,
                                          const void *qscale_ptr, const void *q_seq_lens_ptr,
                                          const void *cu_seqlens_q_ptr, int num_batch,
                                          int num_head_q, int ldQ, int max_num_q_blocks,
                                          cudaStream_t stream) {
  constexpr int kSamplePerBlock = kStemBlockSize / kStride;
  dim3 grid(max_num_q_blocks, num_batch * num_head_q);
  dim3 block(kSamplePerBlock * 32);

  kernels::stem_prep_varlen_q_kernel<kStemBlockSize, kStride, kDimQK, kIsPerTensorQscale>
      <<<grid, block, 0, stream>>>(reinterpret_cast<const __nv_fp8_e4m3 *>(q_fp8_ptr),
                                   reinterpret_cast<const float *>(qscale_ptr),
                                   reinterpret_cast<const int *>(q_seq_lens_ptr),
                                   reinterpret_cast<const int *>(cu_seqlens_q_ptr),
                                   reinterpret_cast<__nv_bfloat16 *>(qflat_ptr), num_head_q, ldQ,
                                   max_num_q_blocks);
}

void stem_oam_prep_varlen_q_dim192_async(void *qflat_ptr, const void *q_fp8_ptr,
                                         const void *qscale_ptr, const void *q_seq_lens_ptr,
                                         const void *cu_seqlens_q_ptr, int num_batch,
                                         int num_head_q, int ldQ, int stem_block_size,
                                         int stem_stride, int max_num_q_blocks,
                                         cudaStream_t stream) {
  // dim192 MLA model uses scalar (per-tensor) qscale [1].
  launch_stem_oam_prep_varlen_q_dim192<128, 16, 192, true>(
      qflat_ptr, q_fp8_ptr, qscale_ptr, q_seq_lens_ptr, cu_seqlens_q_ptr, num_batch, num_head_q,
      ldQ, max_num_q_blocks, stream);
}

}  // namespace stem
}  // namespace hpc
