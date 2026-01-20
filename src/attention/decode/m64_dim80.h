// Copyright (C) 2026 Tencent.

#ifndef SRC_ATTENTION_DECODE_M64_DIM80_H_
#define SRC_ATTENTION_DECODE_M64_DIM80_H_

#include <cuda_runtime_api.h>

namespace hpc {
namespace attention {
namespace decode {

bool m64_dim80_async(void *y_ptr, const void *q_ptr, void *kcache_ptr, void *vcache_ptr,
                     const int *block_ids_ptr, const int *num_seq_kvcache_ptr, bool new_kv_included,
                     int num_batch, int num_head_q, int num_head_k, int num_head_v, int num_dim_qk,
                     int num_dim_v, int num_kvcache_blocks, int block_size, int num_seq_max_blocks,
                     int ldY, int ldQ, int ldK, int ldV, cudaStream_t stream);

}  // namespace decode
}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_DECODE_M64_DIM80_H_
