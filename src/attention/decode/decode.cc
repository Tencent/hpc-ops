// Copyright 2025 hpc-ops authors

#include "src/attention/decode/decode.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>

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

bool attention_decode_bf16_adaptive_async(
    void *y_ptr, void *lse_ptr, void *split_out_ptr, const int *task_map_ptr, const void *q_ptr,
    void *kcache_ptr, void *vcache_ptr, const int *block_ids_ptr, const int *num_seq_kvcache_ptr,
    int *split_flag_ptr, bool new_kv_included, int splitk, int num_batch, int num_seq_q,
    int num_head_q, int num_head_k, int num_head_v, int num_dim_qk, int num_dim_v,
    int num_kvcache_blocks, int block_size, int num_seq_max_blocks, int ldY, int ldQ,
    int64_t kcache_block_stride, int64_t kcache_token_stride, int64_t kcache_head_stride,
    int64_t vcache_block_stride, int64_t vcache_token_stride, int64_t vcache_head_stride,
    cudaStream_t stream) {
  if (num_dim_qk == 128) {
    return smallm_bf16_dim128_dynamic_adaptive_async(
        y_ptr, lse_ptr, split_out_ptr, task_map_ptr, q_ptr, kcache_ptr, vcache_ptr, block_ids_ptr,
        splitk, num_batch, num_seq_q, num_head_q, num_head_k, num_head_v, num_dim_qk, num_dim_v,
        num_kvcache_blocks, block_size, num_seq_max_blocks, ldQ, kcache_block_stride,
        kcache_token_stride, kcache_head_stride, vcache_block_stride, vcache_token_stride,
        vcache_head_stride, stream);
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

bool attention_decode_fp8_adaptive_async(
    void *y_ptr, void *lse_ptr, void *split_out_ptr, const int *task_map_ptr, const void *q_ptr,
    void *kcache_ptr, void *vcache_ptr, const int *block_ids_ptr, const int *num_seq_kvcache_ptr,
    const float *qscale_ptr, const float *kscale_ptr, const float *pscale_ptr,
    const float *vscale_ptr, int *split_flag_ptr, bool new_kv_included, int splitk,
    int splitk_min_len, int consumers, int quant_type, int num_batch, int num_seq_q, int num_head_q,
    int num_head_k, int num_head_v, int num_dim_qk, int num_dim_v, int num_kvcache_blocks,
    int block_size, int num_seq_max_blocks, int qscale_pad_stride, int ldY, int ldQ,
    int64_t kcache_block_stride, int64_t kcache_token_stride, int64_t kcache_head_stride,
    int64_t vcache_block_stride, int64_t vcache_token_stride, int64_t vcache_head_stride,
    cudaStream_t stream) {
  if (num_dim_qk == 128) {
    if (quant_type == 0) {
      return smallm_fp8_qkpertoken_perhead_vperhead_dim128_dynamic_adaptive_async(
          y_ptr, lse_ptr, split_out_ptr, task_map_ptr, q_ptr, kcache_ptr, vcache_ptr, block_ids_ptr,
          num_seq_kvcache_ptr, qscale_ptr, kscale_ptr, pscale_ptr, vscale_ptr, split_flag_ptr,
          new_kv_included, splitk, splitk_min_len, consumers, num_batch, num_seq_q, num_head_q,
          num_head_k, num_head_v, num_dim_qk, num_dim_v, num_kvcache_blocks, block_size,
          num_seq_max_blocks, qscale_pad_stride, ldY, ldQ, kcache_block_stride, kcache_token_stride,
          kcache_head_stride, vcache_block_stride, vcache_token_stride, vcache_head_stride, stream);
    } else if (quant_type == 1) {
      return smallm_fp8_qpertoken_perhead_kvpertensor_dim128_dynamic_adaptive_async(
          y_ptr, lse_ptr, split_out_ptr, task_map_ptr, q_ptr, kcache_ptr, vcache_ptr, block_ids_ptr,
          num_seq_kvcache_ptr, qscale_ptr, kscale_ptr, pscale_ptr, vscale_ptr, split_flag_ptr,
          new_kv_included, splitk, splitk_min_len, consumers, num_batch, num_seq_q, num_head_q,
          num_head_k, num_head_v, num_dim_qk, num_dim_v, num_kvcache_blocks, block_size,
          num_seq_max_blocks, qscale_pad_stride, ldY, ldQ, kcache_block_stride, kcache_token_stride,
          kcache_head_stride, vcache_block_stride, vcache_token_stride, vcache_head_stride, stream);
    }
  }
  return false;
}
}  // namespace decode

// Hybrid a8c8-fp16pv decode dispatcher (Q fp8 with per-token+head qscale).
namespace decode {
// Hybrid a8c8-fp16pv decode launchers (declared here, defined in the sm90
// static/dynamic .cu files).
void launch_attention_decode_qfp8_kpertensor_vpertensor_fp16_pv_compute_dim128_smallm_splitk(
    void *y_ptr, void *lse_ptr, void *split_out_ptr, const void *q_ptr, void *kcache_ptr,
    void *vcache_ptr, const void *qscale_ptr, const void *kscale_ptr, const void *vscale_ptr,
    const int *block_ids_ptr, const int *num_seq_kvcache_ptr, int *split_flag_ptr,
    bool new_kv_included, int splitk, int num_batch, int num_seq_q, int num_head_q, int num_head_k,
    int num_head_v, int num_dim_qk, int num_dim_v, int num_kvcache_blocks, int block_size,
    int num_seq_max_blocks, int qscale_pad_stride, int ldY, int ldQ, int64_t kcache_block_stride,
    int64_t kcache_token_stride, int64_t kcache_head_stride, int64_t vcache_block_stride,
    int64_t vcache_token_stride, int64_t vcache_head_stride, cudaStream_t stream);

void launch_attention_decode_qfp8_kpertoken_perhead_vperhead_fp16_pv_compute_dim128_smallm_splitk(
    void *y_ptr, void *lse_ptr, void *split_out_ptr, const void *q_ptr, void *kcache_ptr,
    void *vcache_ptr, const void *qscale_ptr, const void *kscale_ptr, const void *vscale_ptr,
    const int *block_ids_ptr, const int *num_seq_kvcache_ptr, int *split_flag_ptr,
    bool new_kv_included, int splitk, int num_batch, int num_seq_q, int num_head_q, int num_head_k,
    int num_head_v, int num_dim_qk, int num_dim_v, int num_kvcache_blocks, int block_size,
    int num_seq_max_blocks, int qscale_pad_stride, int ldY, int ldQ, int64_t kcache_block_stride,
    int64_t kcache_token_stride, int64_t kcache_head_stride, int64_t vcache_block_stride,
    int64_t vcache_token_stride, int64_t vcache_head_stride, cudaStream_t stream);

bool attention_decode_qfp8_kpertensor_vpertensor_fp16_pv_compute_dynamic_async(
    void *y_ptr, void *lse_ptr, void *splitk_out_ptr, const int *task_map_ptr, int num_total_ctas,
    const void *q_ptr, void *kcache_ptr, void *vcache_ptr, const float *qscale_ptr,
    const float *kscale_ptr, const float *vscale_ptr, const int *block_ids_ptr,
    bool new_kv_included, int splitk, int num_batch, int num_seq_q, int num_head_q, int num_head_k,
    int num_head_v, int num_dim_qk, int num_dim_v, int num_kvcache_blocks, int block_size,
    int num_seq_max_blocks, int ldQ, int64_t kcache_block_stride, int64_t kcache_token_stride,
    int64_t kcache_head_stride, int64_t vcache_block_stride, int64_t vcache_token_stride,
    int64_t vcache_head_stride, cudaStream_t stream);

bool attention_decode_qfp8_kpertoken_perhead_vperhead_fp16_pv_compute_dynamic_async(
    void *y_ptr, void *lse_ptr, void *splitk_out_ptr, const int *task_map_ptr, int num_total_ctas,
    const void *q_ptr, void *kcache_ptr, void *vcache_ptr, const float *qscale_ptr,
    const float *kscale_ptr, const float *vscale_ptr, const int *block_ids_ptr,
    bool new_kv_included, int splitk, int num_batch, int num_seq_q, int num_head_q, int num_head_k,
    int num_head_v, int num_dim_qk, int num_dim_v, int num_kvcache_blocks, int block_size,
    int num_seq_max_blocks, int ldQ, int64_t kcache_block_stride, int64_t kcache_token_stride,
    int64_t kcache_head_stride, int64_t vcache_block_stride, int64_t vcache_token_stride,
    int64_t vcache_head_stride, cudaStream_t stream);

bool attention_decode_fp8_kv_fp16_pv_compute_async(
    void *y_ptr, void *lse_ptr, void *split_out_ptr, const void *q_ptr, void *kcache_ptr,
    void *vcache_ptr, const void *qscale_ptr, const void *kscale_ptr, const void *vscale_ptr,
    const int *block_ids_ptr, const int *num_seq_kvcache_ptr, int *split_flag_ptr,
    bool new_kv_included, int splitk, int num_batch, int num_seq_q, int num_head_q, int num_head_k,
    int num_head_v, int num_dim_qk, int num_dim_v, int num_kvcache_blocks, int block_size,
    int num_seq_max_blocks, int qscale_pad_stride, int ldY, int ldQ, int64_t kcache_block_stride,
    int64_t kcache_token_stride, int64_t kcache_head_stride, int64_t vcache_block_stride,
    int64_t vcache_token_stride, int64_t vcache_head_stride, int quant_type,
    const int *task_map_ptr, int num_total_ctas, cudaStream_t stream) {
  if (task_map_ptr != nullptr && num_dim_qk == 128) {
    const float *qs_f32 = reinterpret_cast<const float *>(qscale_ptr);
    const float *ks_f32 = reinterpret_cast<const float *>(kscale_ptr);
    const float *vs_f32 = reinterpret_cast<const float *>(vscale_ptr);
    if (quant_type == 21) {
      return decode::attention_decode_qfp8_kpertensor_vpertensor_fp16_pv_compute_dynamic_async(
          y_ptr, lse_ptr, split_out_ptr, task_map_ptr, num_total_ctas, q_ptr, kcache_ptr,
          vcache_ptr, qs_f32, ks_f32, vs_f32, block_ids_ptr, new_kv_included, splitk, num_batch,
          num_seq_q, num_head_q, num_head_k, num_head_v, num_dim_qk, num_dim_v, num_kvcache_blocks,
          block_size, num_seq_max_blocks, ldQ, kcache_block_stride, kcache_token_stride,
          kcache_head_stride, vcache_block_stride, vcache_token_stride, vcache_head_stride, stream);
    }
    if (quant_type == 20) {
      return decode::attention_decode_qfp8_kpertoken_perhead_vperhead_fp16_pv_compute_dynamic_async(
          y_ptr, lse_ptr, split_out_ptr, task_map_ptr, num_total_ctas, q_ptr, kcache_ptr,
          vcache_ptr, qs_f32, ks_f32, vs_f32, block_ids_ptr, new_kv_included, splitk, num_batch,
          num_seq_q, num_head_q, num_head_k, num_head_v, num_dim_qk, num_dim_v, num_kvcache_blocks,
          block_size, num_seq_max_blocks, ldQ, kcache_block_stride, kcache_token_stride,
          kcache_head_stride, vcache_block_stride, vcache_token_stride, vcache_head_stride, stream);
    }
    return false;
  }

  if (quant_type == 21 && num_dim_qk == 128) {
    decode::launch_attention_decode_qfp8_kpertensor_vpertensor_fp16_pv_compute_dim128_smallm_splitk(
        y_ptr, lse_ptr, split_out_ptr, q_ptr, kcache_ptr, vcache_ptr, qscale_ptr, kscale_ptr,
        vscale_ptr, block_ids_ptr, num_seq_kvcache_ptr, split_flag_ptr, new_kv_included, splitk,
        num_batch, num_seq_q, num_head_q, num_head_k, num_head_v, num_dim_qk, num_dim_v,
        num_kvcache_blocks, block_size, num_seq_max_blocks, qscale_pad_stride, ldY, ldQ,
        kcache_block_stride, kcache_token_stride, kcache_head_stride, vcache_block_stride,
        vcache_token_stride, vcache_head_stride, stream);
    return true;
  }
  if (quant_type == 20 && num_dim_qk == 128) {
    decode::
        launch_attention_decode_qfp8_kpertoken_perhead_vperhead_fp16_pv_compute_dim128_smallm_splitk(  // NOLINT(whitespace/line_length)
            y_ptr, lse_ptr, split_out_ptr, q_ptr, kcache_ptr, vcache_ptr, qscale_ptr, kscale_ptr,
            vscale_ptr, block_ids_ptr, num_seq_kvcache_ptr, split_flag_ptr, new_kv_included, splitk,
            num_batch, num_seq_q, num_head_q, num_head_k, num_head_v, num_dim_qk, num_dim_v,
            num_kvcache_blocks, block_size, num_seq_max_blocks, qscale_pad_stride, ldY, ldQ,
            kcache_block_stride, kcache_token_stride, kcache_head_stride, vcache_block_stride,
            vcache_token_stride, vcache_head_stride, stream);
    return true;
  }
  return false;
}

}  // namespace decode
}  // namespace attention
}  // namespace hpc
