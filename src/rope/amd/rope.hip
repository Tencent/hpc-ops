// Copyright (C) 2026 Tencent.
#include "src/rope/amd/rope.h"
#include "src/utils/amd/device.cuh"

namespace hpc::rope::amd {
namespace kernels {

// One CTA per token/head, plus one tail-clear CTA per request/KV head.
// A full CTA reduction avoids assuming a wave32 execution mode.
template <int kNormPolicy>
__global__ void rope_norm_store_kv_kernel(hip_bfloat16* q, hip_bfloat16* kcache,
                                          hip_bfloat16* vcache, hip_bfloat16* out_k,
                                          hip_bfloat16* out_v, const hip_bfloat16* qkv,
                                          const float* cos_sin, const int* seqlens,
                                          const int* q_index, const int* block_ids,
                                          const float* q_weight, const float* k_weight, int rows,
                                          int requests, int q_heads, int kv_heads, int block_size,
                                          int max_blocks, int64_t k_stride, int64_t v_stride) {
  constexpr int kDim = 128;
  constexpr int kThreads = 64;
  const int row = blockIdx.x;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;
  const bool is_q = head < q_heads;
  const int kv_head = head - q_heads;
  if (row >= rows) {
    if (is_q) return;
    const int req = row - rows;
    const int length = seqlens[req];
    if (length <= 0) return;
    const int last = length - 1;
    const int physical = block_ids[static_cast<int64_t>(req) * max_blocks + last / block_size];
    for (int i = tid; i < (block_size - 1 - last % block_size) * kDim; i += kThreads) {
      const int tail_row = last % block_size + 1 + i / kDim;
      const int64_t offset =
          (static_cast<int64_t>(tail_row) * kv_heads + kv_head) * kDim + i % kDim;
      kcache[static_cast<int64_t>(physical) * k_stride + offset] = hip_bfloat16(0.0f);
      vcache[static_cast<int64_t>(physical) * v_stride + offset] = hip_bfloat16(0.0f);
    }
    return;
  }

  // upper_bound(q_index, row), including empty requests and padded graph rows.
  int lo = 0;
  int hi = requests + 1;
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    if (q_index[mid] <= row)
      lo = mid + 1;
    else
      hi = mid;
  }
  const int req = lo - 1;
  if (req < 0 || req >= requests || seqlens[req] <= 0) return;
  const int position = row + seqlens[req] - q_index[req + 1];
  if (position < 0) return;
  const int physical = block_ids[static_cast<int64_t>(req) * max_blocks + position / block_size];
  const int64_t kv_offset =
      (static_cast<int64_t>(position % block_size) * kv_heads + kv_head) * kDim;
  const int64_t qkv_row = static_cast<int64_t>(row) * (q_heads + 2 * kv_heads) * kDim;
  const int input_head = is_q ? head : q_heads + kv_head;
  float first = static_cast<float>(qkv[qkv_row + input_head * kDim + tid]);
  float second = static_cast<float>(qkv[qkv_row + input_head * kDim + tid + kDim / 2]);
  const float* weight = is_q ? q_weight : k_weight;
  __shared__ float scratch[32];
  if constexpr (kNormPolicy == 2) {
    const float sum = hpc::amd::block_sum<kThreads>(first * first + second * second, scratch);
    const float inv = rsqrtf(sum / kDim + 1e-6f);
    first = first * inv * weight[tid];
    second = second * inv * weight[tid + kDim / 2];
  }
  const float cosine = cos_sin[static_cast<int64_t>(position) * kDim + tid];
  const float sine = cos_sin[static_cast<int64_t>(position) * kDim + tid + kDim / 2];
  const float rotated_first = first * cosine - second * sine;
  second = second * cosine + first * sine;
  first = rotated_first;
  if constexpr (kNormPolicy == 1) {
    const float sum = hpc::amd::block_sum<kThreads>(first * first + second * second, scratch);
    const float inv = rsqrtf(sum / kDim + 1e-6f);
    first = first * inv * weight[tid];
    second = second * inv * weight[tid + kDim / 2];
  }
  hip_bfloat16* dst = is_q    ? q + (static_cast<int64_t>(row) * q_heads + head) * kDim
                      : out_k ? out_k + (static_cast<int64_t>(row) * kv_heads + kv_head) * kDim
                              : kcache + static_cast<int64_t>(physical) * k_stride + kv_offset;
  dst[tid] = hip_bfloat16(first);
  dst[tid + kDim / 2] = hip_bfloat16(second);
  if (!is_q) {
    const int64_t v_input = qkv_row + (q_heads + kv_heads + kv_head) * kDim;
    hip_bfloat16* v_dst = out_v ? out_v + (static_cast<int64_t>(row) * kv_heads + kv_head) * kDim
                                : vcache + static_cast<int64_t>(physical) * v_stride + kv_offset;
    v_dst[tid] = qkv[v_input + tid];
    v_dst[tid + kDim / 2] = qkv[v_input + tid + kDim / 2];
  }
}

}  // namespace kernels

void rope_norm_store_kv_async(hip_bfloat16* q, hip_bfloat16* kcache, hip_bfloat16* vcache,
                              hip_bfloat16* out_k, hip_bfloat16* out_v, const hip_bfloat16* qkv,
                              const float* cos_sin, const int* seqlens, const int* q_index,
                              const int* block_ids, const float* q_weight, const float* k_weight,
                              int rows, int requests, int q_heads, int kv_heads, int block_size,
                              int max_blocks, int64_t k_stride, int64_t v_stride, int norm_policy,
                              hipStream_t stream) {
  if (requests == 0) return;
#define HPC_LAUNCH_ROPE(POLICY)                                                                 \
  hipLaunchKernelGGL((kernels::rope_norm_store_kv_kernel<POLICY>),                              \
                     dim3(rows + requests, q_heads + kv_heads), dim3(64), 0, stream, q, kcache, \
                     vcache, out_k, out_v, qkv, cos_sin, seqlens, q_index, block_ids, q_weight, \
                     k_weight, rows, requests, q_heads, kv_heads, block_size, max_blocks,       \
                     k_stride, v_stride)
  if (norm_policy == 0) {
    HPC_LAUNCH_ROPE(0);
  } else if (norm_policy == 1) {
    HPC_LAUNCH_ROPE(1);
  } else {
    HPC_LAUNCH_ROPE(2);
  }
#undef HPC_LAUNCH_ROPE
}

}  // namespace hpc::rope::amd
