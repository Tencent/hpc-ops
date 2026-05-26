// Copyright 2025 hpc-ops authors

#include "src/attention/prefill/prefill.h"

#include <cuda.h>

#include <algorithm>

#include "src/attention/prefill/multi_stage_dim128.h"
#include "src/attention/prefill/multi_stage_with_kvcache_dim128.h"
#include "src/attention/prefill/warp_spec_blocksparse_fp8_dim192.h"
#include "src/attention/prefill/warp_spec_dim128.h"
#include "src/attention/prefill/warp_spec_mla.h"
#include "src/attention/prefill/warp_spec_with_kvcache_blocksparse_fp8_dim128.h"
#include "src/attention/prefill/warp_spec_with_kvcache_dim128.h"
#include "src/attention/prefill/warp_spec_with_kvcache_fp8_dim128.h"
#include "src/utils/utils.h"

namespace hpc {
namespace attention {

void attention_prefill_bf16_async(void *y_ptr, const void *q_ptr, const void *k_ptr,
                                  const void *v_ptr, const void *seqlens_q_ptr,
                                  const void *cu_seqlens_q_ptr, void *tmas_ptr, int num_batch,
                                  int total_seq_q, int max_seq_q, int num_dim_qk, int num_dim_v,
                                  int num_head_q, int num_head_kv, int ldY, int ldQ, int ldK,
                                  int ldV, cudaStream_t stream) {
  constexpr int kTileM = 64;
  int max_total_blocks = (max_seq_q + kTileM - 1) / kTileM * num_batch * num_head_q;
  if (max_total_blocks < get_sm_count() * 2) {
    if (num_dim_qk == 128 && num_dim_v == 128) {
      prefill::multi_stage_dim128_async(y_ptr, q_ptr, k_ptr, v_ptr, seqlens_q_ptr, cu_seqlens_q_ptr,
                                        tmas_ptr, num_batch, total_seq_q, max_seq_q, num_dim_qk,
                                        num_dim_v, num_head_q, num_head_kv, ldY, ldQ, ldK, ldV,
                                        stream);
    }
  } else {
    if (num_dim_qk == 128 && num_dim_v == 128) {
      prefill::warp_spec_dim128_async(y_ptr, q_ptr, k_ptr, v_ptr, seqlens_q_ptr, cu_seqlens_q_ptr,
                                      tmas_ptr, num_batch, total_seq_q, max_seq_q, num_dim_qk,
                                      num_dim_v, num_head_q, num_head_kv, ldY, ldQ, ldK, ldV,
                                      stream);
    }
  }
}

void attention_with_kvcache_prefill_bf16_async(
    void *y_ptr, const void *q_ptr, const void *kcache_ptr, const void *vcache_ptr,
    const void *cu_seqlens_q_ptr, const void *block_ids_ptr, const void *seqlens_kvcache_ptr,
    void *tmas_ptr, int num_batch, int total_seq_q, int max_seq_q, int num_dim_qk, int num_dim_v,
    int num_head_q, int num_head_kv, int num_kvcache_blocks, int block_size, int num_seq_max_blocks,
    int ldY, int ldQ, int ldK, int ldK1, int ldK2, int ldV, int ldV1, int ldV2,
    cudaStream_t stream) {
  constexpr int kTileM = 64;
  int max_total_blocks = (max_seq_q + kTileM - 1) / kTileM * num_batch * num_head_q;
  if (max_total_blocks < get_sm_count() * 2) {
    if (num_dim_qk == 128 && num_dim_v == 128) {
      prefill::multi_stage_with_kvcache_dim128_async(
          y_ptr, q_ptr, kcache_ptr, vcache_ptr, cu_seqlens_q_ptr, block_ids_ptr,
          seqlens_kvcache_ptr, tmas_ptr, num_batch, total_seq_q, max_seq_q, num_dim_qk, num_dim_v,
          num_head_q, num_head_kv, num_kvcache_blocks, block_size, num_seq_max_blocks, ldY, ldQ,
          ldK, ldK1, ldK2, ldV, ldV1, ldV2, stream);
    }
  } else {
    if (num_dim_qk == 128 && num_dim_v == 128) {
      prefill::warp_spec_with_kvcache_dim128_async(
          y_ptr, q_ptr, kcache_ptr, vcache_ptr, cu_seqlens_q_ptr, block_ids_ptr,
          seqlens_kvcache_ptr, tmas_ptr, num_batch, total_seq_q, max_seq_q, num_dim_qk, num_dim_v,
          num_head_q, num_head_kv, num_kvcache_blocks, block_size, num_seq_max_blocks, ldY, ldQ,
          ldK, ldK1, ldK2, ldV, ldV1, ldV2, stream);
    }
  }
}

void attention_with_kvcache_prefill_qpertoken_perhead_kvpertensor_fp8_async(
    void *y_ptr, const void *q_ptr, const void *kcache_ptr, const void *vcache_ptr,
    const void *qscale_ptr, const void *kscale_ptr, const void *vscale_ptr,
    const void *cu_seqlens_q_ptr, const void *block_ids_ptr, const void *seqlens_kvcache_ptr,
    void *tmas_ptr, int num_batch, int total_seq_q, int max_seq_q, int max_seq_q_pad,
    int num_dim_qk, int num_dim_v, int num_head_q, int num_head_kv, int num_kvcache_blocks,
    int block_size, int num_seq_max_blocks, int ldY, int ldQ, int ldK, int ldK1, int ldK2, int ldV,
    int ldV1, int ldV2, cudaStream_t stream) {
  prefill::warp_spec_with_kvcache_qpertoken_perhead_kvpertensor_fp8_dim128_async(
      y_ptr, q_ptr, kcache_ptr, vcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr, cu_seqlens_q_ptr,
      block_ids_ptr, seqlens_kvcache_ptr, tmas_ptr, num_batch, total_seq_q, max_seq_q,
      max_seq_q_pad, num_dim_qk, num_dim_v, num_head_q, num_head_kv, num_kvcache_blocks, block_size,
      num_seq_max_blocks, ldY, ldQ, ldK, ldK1, ldK2, ldV, ldV1, ldV2, stream);
}

void attention_with_kvcache_prefill_qkpertoken_perhead_vperhead_fp8_async(
    void *y_ptr, const void *q_ptr, const void *kcache_ptr, const void *vcache_ptr,
    const void *qscale_ptr, const void *kscale_ptr, const void *vscale_ptr,
    const void *cu_seqlens_q_ptr, const void *block_ids_ptr, const void *seqlens_kvcache_ptr,
    void *tmas_ptr, int num_batch, int total_seq_q, int max_seq_q, int max_seq_q_pad,
    int num_dim_qk, int num_dim_v, int num_head_q, int num_head_kv, int num_kvcache_blocks,
    int block_size, int scale_block_size, int num_seq_max_blocks, int ldY, int ldQ, int ldK,
    int ldK1, int ldK2, int ldV, int ldV1, int ldV2, int ldKS, int ldKS1, int ldKS2,
    cudaStream_t stream) {
  prefill::warp_spec_with_kvcache_qkpertoken_perhead_vperhead_fp8_dim128_async(
      y_ptr, q_ptr, kcache_ptr, vcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr, cu_seqlens_q_ptr,
      block_ids_ptr, seqlens_kvcache_ptr, tmas_ptr, num_batch, total_seq_q, max_seq_q,
      max_seq_q_pad, num_dim_qk, num_dim_v, num_head_q, num_head_kv, num_kvcache_blocks, block_size,
      scale_block_size, num_seq_max_blocks, ldY, ldQ, ldK, ldK1, ldK2, ldV, ldV1, ldV2, ldKS, ldKS1,
      ldKS2, stream);
}

void mla_prefill_bf16_async(void *y_ptr, const void *q_ptr, const void *kv_ptr,
                            const void *seqlens_q_ptr, const void *cu_seqlens_q_ptr, void *tmas_ptr,
                            int num_batch, int total_seq_q, int max_seq_q, int num_dim_qk,
                            int num_dim_v, int num_head_q, int num_head_kv, int ldY, int ldQ,
                            int ldKV, cudaStream_t stream) {
  prefill::warp_spec_mla_async(y_ptr, q_ptr, kv_ptr, seqlens_q_ptr, cu_seqlens_q_ptr, tmas_ptr,
                               num_batch, total_seq_q, max_seq_q, num_dim_qk, num_dim_v, num_head_q,
                               num_head_kv, ldY, ldQ, ldKV, stream);
}

void attention_with_kvcache_blocksparse_prefill_qpertoken_perhead_kvpertensor_fp8_async(
    void *y_ptr, const void *q_ptr, const void *kcache_ptr, const void *vcache_ptr,
    const void *qscale_ptr, const void *kscale_ptr, const void *vscale_ptr,
    const void *cu_seqlens_q_ptr, const void *block_ids_ptr, const void *seqlens_kvcache_ptr,
    void *tmas_ptr, int num_batch, int total_seq_q, int max_seq_q, int max_seq_q_pad,
    int num_dim_qk, int num_dim_v, int num_head_q, int num_head_kv, int num_kvcache_blocks,
    int block_size, int num_seq_max_blocks, int ldY, int ldQ, int ldK, int ldK1, int ldK2, int ldV,
    int ldV1, int ldV2, const void *block_mask_ptr, int num_tile_kv_in_mask, cudaStream_t stream) {
  prefill::warp_spec_with_kvcache_blocksparse_qpertoken_perhead_kvpertensor_fp8_dim128_async(
      y_ptr, q_ptr, kcache_ptr, vcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr, cu_seqlens_q_ptr,
      block_ids_ptr, seqlens_kvcache_ptr, tmas_ptr, num_batch, total_seq_q, max_seq_q,
      max_seq_q_pad, num_dim_qk, num_dim_v, num_head_q, num_head_kv, num_kvcache_blocks, block_size,
      num_seq_max_blocks, ldY, ldQ, ldK, ldK1, ldK2, ldV, ldV1, ldV2, block_mask_ptr,
      num_tile_kv_in_mask, stream);
}

void attention_with_kvcache_blocksparse_prefill_qkpertoken_perhead_vperhead_fp8_async(
    void *y_ptr, const void *q_ptr, const void *kcache_ptr, const void *vcache_ptr,
    const void *qscale_ptr, const void *kscale_ptr, const void *vscale_ptr,
    const void *cu_seqlens_q_ptr, const void *block_ids_ptr, const void *seqlens_kvcache_ptr,
    void *tmas_ptr, int num_batch, int total_seq_q, int max_seq_q, int max_seq_q_pad,
    int num_dim_qk, int num_dim_v, int num_head_q, int num_head_kv, int num_kvcache_blocks,
    int block_size, int scale_block_size, int num_seq_max_blocks, int ldY, int ldQ, int ldK,
    int ldK1, int ldK2, int ldV, int ldV1, int ldV2, int ldKS, int ldKS1, int ldKS2,
    const void *block_mask_ptr, int num_tile_kv_in_mask, cudaStream_t stream) {
  prefill::warp_spec_with_kvcache_blocksparse_qkpertoken_perhead_vperhead_fp8_dim128_async(
      y_ptr, q_ptr, kcache_ptr, vcache_ptr, qscale_ptr, kscale_ptr, vscale_ptr, cu_seqlens_q_ptr,
      block_ids_ptr, seqlens_kvcache_ptr, tmas_ptr, num_batch, total_seq_q, max_seq_q,
      max_seq_q_pad, num_dim_qk, num_dim_v, num_head_q, num_head_kv, num_kvcache_blocks, block_size,
      scale_block_size, num_seq_max_blocks, ldY, ldQ, ldK, ldK1, ldK2, ldV, ldV1, ldV2, ldKS, ldKS1,
      ldKS2, block_mask_ptr, num_tile_kv_in_mask, stream);
}

void attention_blocksparse_prefill_fp8_dim192_async(
    void *y_ptr, const void *q_ptr, const void *k_ptr, const void *v_ptr,
    const void *cu_seqlens_q_ptr, const void *cu_seqlens_kv_ptr, void *tmas_ptr, int num_batch,
    int total_seq_q, int max_seq_q, int max_seq_kv, int num_dim_qk, int num_dim_v, int num_head_q,
    int num_head_kv, int ldY, int ldQ, int ldK, int ldV, const void *block_mask_ptr,
    int num_tile_kv_in_mask, float softmax_qkscale, float vscale, cudaStream_t stream) {
  prefill::warp_spec_blocksparse_fp8_dim192_async(
      y_ptr, q_ptr, k_ptr, v_ptr, cu_seqlens_q_ptr, cu_seqlens_kv_ptr, tmas_ptr, num_batch,
      total_seq_q, max_seq_q, max_seq_kv, num_dim_qk, num_dim_v, num_head_q, num_head_kv, ldY, ldQ,
      ldK, ldV, block_mask_ptr, num_tile_kv_in_mask, softmax_qkscale, vscale, stream);
}

}  // namespace attention
}  // namespace hpc
