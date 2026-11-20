// Copyright (C) 2026 Tencent.
#ifndef SRC_ROPE_AMD_ROPE_H_
#define SRC_ROPE_AMD_ROPE_H_

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace hpc::rope::amd {

void rope_norm_store_kv_async(hip_bfloat16* q, hip_bfloat16* kcache, hip_bfloat16* vcache,
                              hip_bfloat16* out_k, hip_bfloat16* out_v, const hip_bfloat16* qkv,
                              const float* cos_sin, const int* seqlens, const int* q_index,
                              const int* block_ids, const float* q_weight, const float* k_weight,
                              int rows, int requests, int q_heads, int kv_heads, int block_size,
                              int max_blocks, int64_t k_stride, int64_t v_stride, int norm_policy,
                              hipStream_t stream);

// TODO(amd): rope_norm_store_kv_fp8_kernel and the FP8 entry; excluded from this batch.

}  // namespace hpc::rope::amd
#endif  // SRC_ROPE_AMD_ROPE_H_
