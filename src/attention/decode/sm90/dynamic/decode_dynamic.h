// Copyright 2025 hpc-ops authors

#ifndef SRC_ATTENTION_DECODE_SM90_DYNAMIC_DECODE_DYNAMIC_H_
#define SRC_ATTENTION_DECODE_SM90_DYNAMIC_DECODE_DYNAMIC_H_

#include <cuda_runtime_api.h>
#include <stdint.h>

#include <utility>
#include <vector>

#include "src/attention/decode/sm90/dynamic/dynamic_sched_task_info.h"

namespace hpc {
namespace attention {
namespace decode {
namespace dynamic {

// Number of CTAs launched per SM for the sm90 dynamic attention kernels.
constexpr int kCTAPerSM = 4;

// Maximum number of split-k chunks a single (ihead_kv, ibatch) can be cut into
// by the dynamic assigner.
constexpr int kMaxSplitK = 78 * kCTAPerSM;

// sm90-only assigner that buckets attention-decode tasks at kTileN=64 (the
// sm90 GEMM tile size), with a **flat** launcher: each CTA pulls its own bin
// of (ihead_kv, ibatch, chunk) tasks
//
// task_map_ptr layout (int units, 12 ints per SM90DynamicTaskInfo):
//   offset 0                                              : num_tile_per_cta + 1  (header)
//   [1 .. 1 + (num_tile_per_cta+1)*num_cta_count*12)      : per-CTA task list
//                                                            (first slot with ihead_kv<0
//                                                            i.e. ibatch < 0 marks end;
//                                                            trailing slots same)
//   + max_num_batch_pad  ints                             : num_chunks[ibatch] for combine
//                                                            (per-batch; same across heads)
//   + num_cta_count_pad  ints                             : per-CTA finish flag
//   + num_cta_count_pad  ints                             : per-CTA actual task count
bool assign_attention_decode_task_sm90_dynamic_async(int *task_map_ptr, const int *num_seq_kvcache,
                                                     int num_batch, int num_head_kv, int num_seq_q,
                                                     bool new_kv_included, int min_process_len,
                                                     cudaStream_t stream);

// CPU reference implementation of the sm90 dynamic assigner at kTileN=64.
// Emits SM90DynamicTaskInfo-layout tasks so tests can allclose a CUDA-
// populated task_map against a host-computed reference.
std::pair<std::vector<SM90DynamicTaskInfo>, std::vector<int>>
assign_attention_decode_task_sm90_dynamic_sync(const int *num_seq_kvcache, int num_batch,
                                               int num_head_kv, int num_seq_q, bool new_kv_included,
                                               int min_process_len);

// Drop-in replacement for
// hpc::attention::decode::smallm_splitk_dim128_fp8_qpertoken_perhead_kvpertensor_async
// but running the sm90 "task_map dynamic scheduling" path:
//   1) bucket tasks across SMs via assign_attention_decode_task_dynamic_async
//      (kTileN=64 granularity)
//   2) launch a persistent attention kernel that consumes its per-SM task list
//   3) launch a combine kernel that merges split-k partials using LSE
//
// `task_map_ptr` must point to a pre-allocated int buffer large enough to hold
// the task map layout (see assign_task_dynamic.cuh for the precise layout).
bool smallm_dim128_fp8_qpertoken_perhead_kvpertensor_dynamic_async(
    void *y_ptr, void *lse_ptr, void *splitk_out_ptr, int *task_map_ptr, const void *q_ptr,
    void *kcache_ptr, void *vcache_ptr, const int *block_ids_ptr, const int *num_seq_kvcache_ptr,
    const float *qscale_ptr, const float *kscale_ptr, const float *vscale_ptr, bool new_kv_included,
    int num_batch, int num_seq_q, int num_head_q, int num_head_k, int num_head_v, int num_dim_qk,
    int num_dim_v, int num_kvcache_blocks, int block_size, int num_seq_max_blocks,
    int qscale_pad_stride, int ldY, int ldQ, int64_t kcache_block_stride,
    int64_t kcache_token_stride, int64_t kcache_head_stride, int64_t vcache_block_stride,
    int64_t vcache_token_stride, int64_t vcache_head_stride, cudaStream_t stream);

// Drop-in replacement for attention_decode_fp8_qkpertoken_perhead_vperhead_async
// using the same task_map dynamic scheduling path as its kvpertensor sibling.
// Same expectations around `task_map_ptr` layout.
bool smallm_dim128_fp8_qkpertoken_perhead_vperhead_dynamic_async(
    void *y_ptr, void *lse_ptr, void *splitk_out_ptr, int *task_map_ptr, const void *q_ptr,
    void *kcache_ptr, void *vcache_ptr, const int *block_ids_ptr, const int *num_seq_kvcache_ptr,
    const float *qscale_ptr, const float *kscale_ptr, const float *vscale_ptr, bool new_kv_included,
    int num_batch, int num_seq_q, int num_head_q, int num_head_k, int num_head_v, int num_dim_qk,
    int num_dim_v, int num_kvcache_blocks, int block_size, int num_seq_max_blocks,
    int qscale_pad_stride, int ldY, int ldQ, int64_t kcache_block_stride,
    int64_t kcache_token_stride, int64_t kcache_head_stride, int64_t vcache_block_stride,
    int64_t vcache_token_stride, int64_t vcache_head_stride, cudaStream_t stream);

}  // namespace dynamic
}  // namespace decode
}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_DECODE_SM90_DYNAMIC_DECODE_DYNAMIC_H_
