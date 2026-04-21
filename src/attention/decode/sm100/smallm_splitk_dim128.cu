// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include <algorithm>
#include <iostream>

#include "cute/tensor.hpp"
#include "src/attention/decode/smallm_dim128.h"

namespace hpc {
namespace attention {
namespace decode {

bool smallm_splitk_dim128_async(
    void *y_ptr, void *lse_ptr, void *splitk_out_ptr, const void *q_ptr, void *kcache_ptr,
    void *vcache_ptr, const int *block_ids_ptr, const int *num_seq_kvcache_ptr, int *split_flag_ptr,
    bool new_kv_included, int splitk, int num_batch, int num_seq_q, int num_head_q, int num_head_k,
    int num_head_v, int num_dim_qk, int num_dim_v, int num_kvcache_blocks, int block_size,
    int num_seq_max_blocks, int ldY, int ldQ, int64_t kcache_block_stride,
    int64_t kcache_token_stride, int64_t kcache_head_stride, int64_t vcache_block_stride,
    int64_t vcache_token_stride, int64_t vcache_head_stride, cudaStream_t stream) {
  return false;
}

}  // namespace decode
}  // namespace attention
}  // namespace hpc
