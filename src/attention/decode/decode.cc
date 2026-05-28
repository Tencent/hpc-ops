// Copyright 2025 hpc-ops authors

#include "src/attention/decode/decode.h"

#include <cuda_runtime_api.h>

#include <algorithm>

#include "src/attention/decode/smallm_dim128.h"

namespace hpc {
namespace attention {
namespace decode {
bool attention_decode_bf16_async(
    void *y_ptr, void *lse_ptr, void *split_out_ptr, const void *q_ptr, void *kcache_ptr,
    void *vcache_ptr, const int *block_ids_ptr, const int *num_seq_kvcache_ptr, int *split_flag_ptr,
    bool new_kv_included, int splitk, int num_batch, int num_seq_q, int num_head_q, int num_head_k,
    int num_head_v, int num_dim_qk, int num_dim_v, int num_kvcache_blocks, int block_size,
    int num_seq_max_blocks, int ldY, int ldQ, int64_t kcache_block_stride,
    int64_t kcache_token_stride, int64_t kcache_head_stride, int64_t vcache_block_stride,
    int64_t vcache_token_stride, int64_t vcache_head_stride, cudaStream_t stream) {
  if (num_dim_qk == 128) {
    return smallm_bf16_dim128_static_async(
        y_ptr, lse_ptr, split_out_ptr, q_ptr, kcache_ptr, vcache_ptr, block_ids_ptr,
        num_seq_kvcache_ptr, split_flag_ptr, new_kv_included, splitk, num_batch, num_seq_q,
        num_head_q, num_head_k, num_head_v, num_dim_qk, num_dim_v, num_kvcache_blocks, block_size,
        num_seq_max_blocks, ldY, ldQ, kcache_block_stride, kcache_token_stride, kcache_head_stride,
        vcache_block_stride, vcache_token_stride, vcache_head_stride, stream);
  }
  return false;
}

bool attention_decode_fp8_async(
    void *y_ptr, void *lse_ptr, void *split_out_ptr, const int *task_map_ptr, const void *q_ptr,
    void *kcache_ptr, void *vcache_ptr, const int *block_ids_ptr, const int *num_seq_kvcache_ptr,
    const float *qscale_ptr, const float *kscale_ptr, const float *vscale_ptr, int *split_flag_ptr,
    bool new_kv_included, int splitk, int splitk_min_len, int consumers, int quant_type,
    int num_batch, int num_seq_q, int num_head_q, int num_head_k, int num_head_v, int num_dim_qk,
    int num_dim_v, int num_kvcache_blocks, int block_size, int num_seq_max_blocks,
    int qscale_pad_stride, int ldY, int ldQ, int64_t kcache_block_stride,
    int64_t kcache_token_stride, int64_t kcache_head_stride, int64_t vcache_block_stride,
    int64_t vcache_token_stride, int64_t vcache_head_stride, cudaStream_t stream) {
  if (num_dim_qk == 128) {
    if (quant_type == 0) {
      if (task_map_ptr) {
        return smallm_fp8_qkpertoken_perhead_vperhead_dim128_dynamic_async(
            y_ptr, lse_ptr, split_out_ptr, task_map_ptr, q_ptr, kcache_ptr, vcache_ptr,
            block_ids_ptr, num_seq_kvcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr, split_flag_ptr,
            new_kv_included, splitk, splitk_min_len, consumers, num_batch, num_seq_q, num_head_q,
            num_head_k, num_head_v, num_dim_qk, num_dim_v, num_kvcache_blocks, block_size,
            num_seq_max_blocks, qscale_pad_stride, ldY, ldQ, kcache_block_stride,
            kcache_token_stride, kcache_head_stride, vcache_block_stride, vcache_token_stride,
            vcache_head_stride, stream);
      } else {
        return smallm_fp8_qkpertoken_perhead_vperhead_dim128_static_async(
            y_ptr, lse_ptr, split_out_ptr, task_map_ptr, q_ptr, kcache_ptr, vcache_ptr,
            block_ids_ptr, num_seq_kvcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr, split_flag_ptr,
            new_kv_included, splitk, splitk_min_len, consumers, num_batch, num_seq_q, num_head_q,
            num_head_k, num_head_v, num_dim_qk, num_dim_v, num_kvcache_blocks, block_size,
            num_seq_max_blocks, qscale_pad_stride, ldY, ldQ, kcache_block_stride,
            kcache_token_stride, kcache_head_stride, vcache_block_stride, vcache_token_stride,
            vcache_head_stride, stream);
      }
    } else if (quant_type == 1) {
      if (task_map_ptr) {
        return smallm_fp8_qpertoken_perhead_kvpertensor_dim128_dynamic_async(
            y_ptr, lse_ptr, split_out_ptr, task_map_ptr, q_ptr, kcache_ptr, vcache_ptr,
            block_ids_ptr, num_seq_kvcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr, split_flag_ptr,
            new_kv_included, splitk, splitk_min_len, consumers, num_batch, num_seq_q, num_head_q,
            num_head_k, num_head_v, num_dim_qk, num_dim_v, num_kvcache_blocks, block_size,
            num_seq_max_blocks, qscale_pad_stride, ldY, ldQ, kcache_block_stride,
            kcache_token_stride, kcache_head_stride, vcache_block_stride, vcache_token_stride,
            vcache_head_stride, stream);
      } else {
        return smallm_fp8_qpertoken_perhead_kvpertensor_dim128_static_async(
            y_ptr, lse_ptr, split_out_ptr, task_map_ptr, q_ptr, kcache_ptr, vcache_ptr,
            block_ids_ptr, num_seq_kvcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr, split_flag_ptr,
            new_kv_included, splitk, splitk_min_len, consumers, num_batch, num_seq_q, num_head_q,
            num_head_k, num_head_v, num_dim_qk, num_dim_v, num_kvcache_blocks, block_size,
            num_seq_max_blocks, qscale_pad_stride, ldY, ldQ, kcache_block_stride,
            kcache_token_stride, kcache_head_stride, vcache_block_stride, vcache_token_stride,
            vcache_head_stride, stream);
      }
    }
  }
  return false;
}
}  // namespace decode
}  // namespace attention
}  // namespace hpc
