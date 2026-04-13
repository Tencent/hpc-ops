// Copyright (C) 2026 Tencent.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdio.h>

#include <string>

#include "cutlass/fast_math.h"
#include "src/rope/rope.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace rope {

namespace kernels {

template <int kNumerator, int kDenominator>
__device__ __forceinline__ constexpr int ceil_div() {
  static_assert(kDenominator > 0, "denominator must >0");
  return (kNumerator + kDenominator - 1) / kDenominator;
}

constexpr float kEps = 1e-6f;

/// In-place rotate a pair of RoPE elements (NeoX version)
__device__ __forceinline__ void rope_rotate_pair(float &x1, float &x2, float cos_val,
                                                 float sin_val) {
  float y1 = x1 * cos_val - x2 * sin_val;
  float y2 = x2 * cos_val + x1 * sin_val;
  x1 = y1;
  x2 = y2;
}

/// RMSNorm in-place: compute RMS over register values, apply weight from shared memory
template <int kNumItemPerThread, int kHeadDim, typename T, int N, int kWarpSize = 32>
__device__ __forceinline__ void rms_norm_apply(vec_t<T, N> &data, const float *smem_weight,
                                               int ilane) {
  float sum_sq = 0.f;
#pragma unroll
  for (int i = 0; i < kNumItemPerThread; ++i) sum_sq += data[i] * data[i];
  sum_sq = warp_reduce_sum_xor(sum_sq);
  float inv_rms = rsqrtf(sum_sq / kHeadDim + kEps);
  constexpr int kRoundsHalf = (kHeadDim / 2 + kWarpSize - 1) / kWarpSize;
#pragma unroll
  for (int r = 0; r < kRoundsHalf; ++r) {
    int i = r * kWarpSize + ilane;
    if (i < kHeadDim / 2) {
      data[r * 2] *= inv_rms * smem_weight[i];
      data[r * 2 + 1] *= inv_rms * smem_weight[i + kHeadDim / 2];
    }
  }
}

/// Warp-level max absolute value
template <int kN, typename T, int N>
__device__ __forceinline__ float warp_abs_max(vec_t<T, N> &data) {
  float m = kEps;
#pragma unroll
  for (int i = 0; i < kN; ++i) m = fmaxf(m, fabsf(data[i]));
  return warp_reduce_max_xor(m);
}

/// Zero rows [from_row, to_row) of a KV cache block
template <typename CacheT, int kElemPerRow, int kWarpSize = 32>
__device__ __forceinline__ void zero_kv_rows(CacheT *block_start, int from_row, int to_row,
                                             int ilane) {
  constexpr int kItemPerThread = 16 / sizeof(CacheT);
  vec_t<CacheT, kItemPerThread> zero_vec;
#pragma unroll
  for (int i = 0; i < kItemPerThread; ++i) zero_vec[i] = CacheT(0);
  for (int row = from_row; row < to_row; ++row) {
    CacheT *row_ptr = block_start + row * kElemPerRow;
    for (int idx = ilane * kItemPerThread; idx < kElemPerRow; idx += kWarpSize * kItemPerThread)
      store(row_ptr + idx, zero_vec);
  }
}

template <int kWarpsPerBlock, int kNumQHeads, int kNumKVHeads, int kQKHeadDim, int kVHeadDim,
          int kNormPolicy>
__global__ void rope_norm_store_kv_kernel(
    __nv_bfloat16 *out_q_ptr, __nv_bfloat16 *kcache_ptr, __nv_bfloat16 *vcache_ptr,
    __nv_bfloat16 *out_k_ptr, __nv_bfloat16 *out_v_ptr, const __nv_bfloat16 *in_qkv_ptr,
    const float *cos_sin_ptr, const int *num_seqlen_per_req_ptr, const int *q_index_ptr,
    const int *kvcache_indices_ptr, const float *q_norm_weight_ptr, const float *k_norm_weight_ptr,
    int kcache_block_offset, int vcache_block_offset, int num_batch, int max_num_kv_block_per_batch,
    cutlass::FastDivmod kv_block_size_divider, int num_rows, int num_compute_blocks) {
  using DType = __nv_bfloat16;

  constexpr int kWarpSize = 32;
  constexpr int kNumElemPerRow =
      kNumQHeads * kQKHeadDim + kNumKVHeads * kQKHeadDim + kNumKVHeads * kVHeadDim;
  constexpr int kNumRoundsHalf = ceil_div<kQKHeadDim / 2, kWarpSize>();
  constexpr int kNumItemPerThread = kNumRoundsHalf * 2;

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int iwarp = tid / kWarpSize;
  int ilane = tid % kWarpSize;

  __shared__ float smem_cos_sin[kWarpsPerBlock][kQKHeadDim];
  __shared__ float smem_q_norm_w[kQKHeadDim];
  __shared__ float smem_k_norm_w[kQKHeadDim];
  __shared__ int smem_batch_id[kWarpsPerBlock];
  __shared__ int smem_token_pos[kWarpsPerBlock];

  // ---- Clear blocks: bid >= num_compute_blocks → one block per request -----
  if (bid >= num_compute_blocks) {
    int req_id = bid - num_compute_blocks;
    if (req_id >= num_batch) return;

    // Last token of this request determines the clear range
    int last_token_pos = num_seqlen_per_req_ptr[req_id] - 1;
    if (last_token_pos < 0) return;

    int block_idx_in_batch, pos_in_block;
    kv_block_size_divider(block_idx_in_batch, pos_in_block, last_token_pos);
    int phys_block_id =
        kvcache_indices_ptr[req_id * max_num_kv_block_per_batch + block_idx_in_batch];

    int zero_from = pos_in_block + 1;
    int zero_to = kv_block_size_divider.divisor;
    if (zero_from < zero_to) {
      // Use all kWarpsPerBlock warps cooperatively to zero rows
      for (int row = zero_from + iwarp; row < zero_to; row += kWarpsPerBlock) {
        DType *k_row = kcache_ptr + (int64_t)phys_block_id * (int64_t)kcache_block_offset +
                       row * (kNumKVHeads * kQKHeadDim);
        DType *v_row = vcache_ptr + (int64_t)phys_block_id * (int64_t)vcache_block_offset +
                       row * (kNumKVHeads * kVHeadDim);
        constexpr int kKItemPerThread = 16 / sizeof(DType);
        vec_t<DType, kKItemPerThread> zero_vec;
#pragma unroll
        for (int z = 0; z < kKItemPerThread; ++z) zero_vec[z] = DType(0);
        for (int idx = ilane * kKItemPerThread; idx < kNumKVHeads * kQKHeadDim;
             idx += kWarpSize * kKItemPerThread)
          store(k_row + idx, zero_vec);
        for (int idx = ilane * kKItemPerThread; idx < kNumKVHeads * kVHeadDim;
             idx += kWarpSize * kKItemPerThread)
          store(v_row + idx, zero_vec);
      }
    }
    return;
  }

  // Search q_index to find batch_id
  int batch_id = 0;
  int token_id = 0;
  int irow = bid * kWarpsPerBlock + iwarp;

  // First kWarpsPerBlock threads do the q_index search for the whole block
  if (tid < kWarpsPerBlock) {
    int global_row = bid * kWarpsPerBlock + tid;
    if (global_row < num_rows) {
      int b = -1;
      for (int i = 0; i < num_batch; ++i) {
        if (global_row < q_index_ptr[i + 1]) {
          b = i;
          break;
        }
      }
      if (b >= 0) {
        smem_batch_id[tid] = b;
        smem_token_pos[tid] = global_row + num_seqlen_per_req_ptr[b] - q_index_ptr[b + 1];
      } else {
        // Padding row: global_row >= q_index[num_batch] (CUDA graph padding)
        smem_batch_id[tid] = -1;
        smem_token_pos[tid] = -1;
      }
    } else {
      smem_batch_id[tid] = -1;
      smem_token_pos[tid] = -1;
    }
  }

  // Load norm weights into shared memory (once per block)
  if constexpr (kNormPolicy > 0) {
    constexpr int kItemPerThread = 16 / sizeof(float);
    constexpr int kNumPacks = kQKHeadDim / kItemPerThread;
    static_assert(kQKHeadDim % kItemPerThread == 0,
                  "kQKHeadDim must be divisible by kItemPerThread");
    static_assert(kItemPerThread * kWarpSize >= kQKHeadDim, "otherwise here should loop");
    if (tid < kNumPacks) {
      int ioffset = tid * kItemPerThread;
      store(smem_q_norm_w + ioffset, load<float, kItemPerThread>(q_norm_weight_ptr + ioffset));
      store(smem_k_norm_w + ioffset, load<float, kItemPerThread>(k_norm_weight_ptr + ioffset));
    }
  }

  __syncthreads();

  //  Early-exit for invalid rows
  if (irow >= num_rows) return;
  batch_id = smem_batch_id[iwarp];
  token_id = smem_token_pos[iwarp];
  if (token_id < 0) return;

  //  Load cos_sin
  {
    constexpr int kItemPerThread = 16 / sizeof(float);
    constexpr int kNumPacks = kQKHeadDim / kItemPerThread;
    static_assert(kQKHeadDim % kItemPerThread == 0, "");
    static_assert(kNumPacks <= kWarpSize, "");
    const float *cos_sin_row = cos_sin_ptr + token_id * kQKHeadDim;
    if (ilane < kNumPacks) {
      int ioffset = ilane * kItemPerThread;
      store(&smem_cos_sin[iwarp][0] + ioffset, load<float, kItemPerThread>(cos_sin_row + ioffset));
    }
    __syncwarp();
  }

  //  KV cache block addressing
  int block_idx_in_batch, block_row;
  kv_block_size_divider(block_idx_in_batch, block_row, token_id);
  int phys_block_id =
      kvcache_indices_ptr[batch_id * max_num_kv_block_per_batch + block_idx_in_batch];

  DType *k_cache_row_start = kcache_ptr + (int64_t)phys_block_id * (int64_t)kcache_block_offset +
                             block_row * (kNumKVHeads * kQKHeadDim);
  DType *v_cache_row_start = vcache_ptr + (int64_t)phys_block_id * (int64_t)vcache_block_offset +
                             block_row * (kNumKVHeads * kVHeadDim);

  const DType *qkv_row = in_qkv_ptr + irow * kNumElemPerRow;

  // Process Q heads – load from global, optional norm, RoPE, optional norm, store
#pragma unroll
  for (int q_head = 0; q_head < kNumQHeads; ++q_head) {
    const DType *q_src = qkv_row + q_head * kQKHeadDim;
    DType *q_dst = out_q_ptr + irow * kNumQHeads * kQKHeadDim + q_head * kQKHeadDim;

    vec_t<float, kNumItemPerThread> data = {0};

    // Load Q head from global memory directly into registers
#pragma unroll
    for (int r = 0; r < kNumRoundsHalf; ++r) {
      int i = r * kWarpSize + ilane;
      if (i < kQKHeadDim / 2) {
        data[r * 2] = __bfloat162float(q_src[i]);
        data[r * 2 + 1] = __bfloat162float(q_src[i + kQKHeadDim / 2]);
      }
    }

    // norm-then-rope
    if constexpr (kNormPolicy == 2) {
      rms_norm_apply<kNumItemPerThread, kQKHeadDim>(data, smem_q_norm_w, ilane);
    }

    // RoPE rotation
#pragma unroll
    for (int r = 0; r < kNumRoundsHalf; ++r) {
      int i = r * kWarpSize + ilane;
      if (i < kQKHeadDim / 2) {
        rope_rotate_pair(data[r * 2], data[r * 2 + 1], smem_cos_sin[iwarp][i],
                         smem_cos_sin[iwarp][i + kQKHeadDim / 2]);
      }
    }

    // rope-then-norm
    if constexpr (kNormPolicy == 1) {
      rms_norm_apply<kNumItemPerThread, kQKHeadDim>(data, smem_q_norm_w, ilane);
    }

    // Store Q output
#pragma unroll
    for (int r = 0; r < kNumRoundsHalf; ++r) {
      int i = r * kWarpSize + ilane;
      if (i < kQKHeadDim / 2) {
        q_dst[i] = __float2bfloat16(data[r * 2]);
        q_dst[i + kQKHeadDim / 2] = __float2bfloat16(data[r * 2 + 1]);
      }
    }
  }

  // Process K heads – load from global, optional norm, RoPE, optional norm,
  // write to KV cache (or out_k_ptr if non-null)
#pragma unroll
  for (int kv_head = 0; kv_head < kNumKVHeads; ++kv_head) {
    const DType *k_src = qkv_row + kNumQHeads * kQKHeadDim + kv_head * kQKHeadDim;

    vec_t<float, kNumItemPerThread> data = {0};

#pragma unroll
    for (int r = 0; r < kNumRoundsHalf; ++r) {
      int i = r * kWarpSize + ilane;
      if (i < kQKHeadDim / 2) {
        data[r * 2] = __bfloat162float(k_src[i]);
        data[r * 2 + 1] = __bfloat162float(k_src[i + kQKHeadDim / 2]);
      }
    }

    if constexpr (kNormPolicy == 2) {
      rms_norm_apply<kNumItemPerThread, kQKHeadDim>(data, smem_k_norm_w, ilane);
    }

#pragma unroll
    for (int r = 0; r < kNumRoundsHalf; ++r) {
      int i = r * kWarpSize + ilane;
      if (i < kQKHeadDim / 2) {
        rope_rotate_pair(data[r * 2], data[r * 2 + 1], smem_cos_sin[iwarp][i],
                         smem_cos_sin[iwarp][i + kQKHeadDim / 2]);
      }
    }

    if constexpr (kNormPolicy == 1) {
      rms_norm_apply<kNumItemPerThread, kQKHeadDim>(data, smem_k_norm_w, ilane);
    }

    // Write K output
    DType *k_dst = (out_k_ptr != nullptr)
                       ? out_k_ptr + irow * kNumKVHeads * kQKHeadDim + kv_head * kQKHeadDim
                       : k_cache_row_start + kv_head * kQKHeadDim;

#pragma unroll
    for (int r = 0; r < kNumRoundsHalf; ++r) {
      int i = r * kWarpSize + ilane;
      if (i < kQKHeadDim / 2) {
        k_dst[i] = __float2bfloat16(data[r * 2]);
        k_dst[i + kQKHeadDim / 2] = __float2bfloat16(data[r * 2 + 1]);
      }
    }
  }

  // Process V heads – no RoPE
  {
    constexpr int kNumVElemPerRow = kNumKVHeads * kVHeadDim;
    constexpr int kItemPerThread = 16 / sizeof(DType);
    static_assert(kNumVElemPerRow % kItemPerThread == 0,
                  "kNumKVHeads * kVHeadDim must be multiple of kItemPerThread");
    constexpr int kNumPackPerRow = kNumVElemPerRow / kItemPerThread;

    const DType *v_src = qkv_row + (kNumQHeads + kNumKVHeads) * kQKHeadDim;
    DType *v_dst =
        (out_v_ptr != nullptr) ? out_v_ptr + irow * kNumKVHeads * kVHeadDim : v_cache_row_start;

    constexpr int kNumLoadRound = ceil_div<kNumPackPerRow, kWarpSize>();
#pragma unroll
    for (int r = 0; r < kNumLoadRound; ++r) {
      int ioffset = (r * kWarpSize + ilane) * kItemPerThread;
      if (ioffset < kNumVElemPerRow) {
        store(v_dst + ioffset, load<DType, kItemPerThread>(v_src + ioffset));
      }
    }
  }
}

template <int kQuantPolicy, int kWarpsPerBlock, int kNumQHeads, int kNumKVHeads, int kQKHeadDim,
          int kVHeadDim, int kNormPolicy>
__global__ void rope_norm_store_kv_fp8_kernel(
    __nv_fp8_e4m3 *out_q_ptr, __nv_fp8_e4m3 *kcache_ptr, __nv_fp8_e4m3 *vcache_ptr,
    __nv_fp8_e4m3 *out_k_ptr, __nv_fp8_e4m3 *out_v_ptr, int32_t *split_k_flag_ptr,
    float *q_scale_ptr, const __nv_bfloat16 *in_qkv_ptr, const float *cos_sin_ptr,
    const int *num_seqlen_per_req_ptr, const int *q_index_ptr, const int *kvcache_indices_ptr,
    const float *q_norm_weight_ptr, const float *k_norm_weight_ptr, const float *k_scale_ptr,
    const float *v_scale_ptr, const float *q_scale_inv_ptr, float upper_max, int max_seqlen_aligned,
    int kcache_block_offset, int vcache_block_offset, int num_batch, int max_num_kv_block_per_batch,
    cutlass::FastDivmod kv_block_size_divider, int num_rows, int num_compute_blocks,
    bool is_prefill) {
  using DType = __nv_bfloat16;
  using QType = __nv_fp8_e4m3;

  constexpr int kWarpSize = 32;
  constexpr int kNumElemPerRow =
      kNumQHeads * kQKHeadDim + kNumKVHeads * kQKHeadDim + kNumKVHeads * kVHeadDim;
  constexpr int kNumRoundsHalf = ceil_div<kQKHeadDim / 2, kWarpSize>();
  constexpr int kNumItemPerThread = kNumRoundsHalf * 2;

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int iwarp = tid / kWarpSize;
  int ilane = tid % kWarpSize;

  // Shared memory
  __shared__ float smem_cos_sin[kWarpsPerBlock][kQKHeadDim];
  __shared__ float smem_q_norm_w[kQKHeadDim];
  __shared__ float smem_k_norm_w[kQKHeadDim];
  __shared__ int smem_batch_id[kWarpsPerBlock];
  __shared__ int smem_token_pos[kWarpsPerBlock];

  // ---- Clear blocks: bid >= num_compute_blocks → one block per request -----
  if (bid >= num_compute_blocks) {
    int req_id = bid - num_compute_blocks;
    if (req_id >= num_batch) return;

    int last_token_pos = num_seqlen_per_req_ptr[req_id] - 1;
    if (last_token_pos < 0) return;

    int block_idx_in_batch, pos_in_block;
    kv_block_size_divider(block_idx_in_batch, pos_in_block, last_token_pos);
    int phys_block_id =
        kvcache_indices_ptr[req_id * max_num_kv_block_per_batch + block_idx_in_batch];

    int zero_from = pos_in_block + 1;
    int zero_to = kv_block_size_divider.divisor;
    if (zero_from < zero_to) {
      for (int row = zero_from + iwarp; row < zero_to; row += kWarpsPerBlock) {
        QType *k_row = kcache_ptr + (int64_t)phys_block_id * (int64_t)kcache_block_offset +
                       row * (kNumKVHeads * kQKHeadDim);
        QType *v_row = vcache_ptr + (int64_t)phys_block_id * (int64_t)vcache_block_offset +
                       row * (kNumKVHeads * kVHeadDim);
        constexpr int kKItemPerThread = 16 / sizeof(QType);
        vec_t<QType, kKItemPerThread> zero_vec;
#pragma unroll
        for (int z = 0; z < kKItemPerThread; ++z) zero_vec[z] = QType(0);
        for (int idx = ilane * kKItemPerThread; idx < kNumKVHeads * kQKHeadDim;
             idx += kWarpSize * kKItemPerThread)
          store(k_row + idx, zero_vec);
        for (int idx = ilane * kKItemPerThread; idx < kNumKVHeads * kVHeadDim;
             idx += kWarpSize * kKItemPerThread)
          store(v_row + idx, zero_vec);
      }
    }
    return;
  }

  // Determine batch_id and token position — unified for prefill and decode
  int batch_id = 0;
  int token_id = 0;
  int irow = bid * kWarpsPerBlock + iwarp;

  if (tid < kWarpsPerBlock) {
    int global_row = bid * kWarpsPerBlock + tid;
    if (global_row < num_rows) {
      int b = -1;
      for (int i = 0; i < num_batch; ++i) {
        if (global_row < q_index_ptr[i + 1]) {
          b = i;
          break;
        }
      }
      if (b >= 0) {
        smem_batch_id[tid] = b;
        smem_token_pos[tid] = global_row + num_seqlen_per_req_ptr[b] - q_index_ptr[b + 1];
      } else {
        smem_batch_id[tid] = -1;
        smem_token_pos[tid] = -1;
      }
    } else {
      smem_batch_id[tid] = -1;
      smem_token_pos[tid] = -1;
    }
  }

  // Load norm weights
  if constexpr (kNormPolicy > 0) {
    constexpr int kItemPerThread = 16 / sizeof(float);
    constexpr int kNumPacks = kQKHeadDim / kItemPerThread;
    static_assert(kQKHeadDim % kItemPerThread == 0, "");
    static_assert(kItemPerThread * kWarpSize >= kQKHeadDim, "");
    if (tid < kNumPacks) {
      int ioffset = tid * kItemPerThread;
      store(smem_q_norm_w + ioffset, load<float, kItemPerThread>(q_norm_weight_ptr + ioffset));
      store(smem_k_norm_w + ioffset, load<float, kItemPerThread>(k_norm_weight_ptr + ioffset));
    }
  }

  // Single barrier: makes batch_id, token_pos, and norm weights visible
  __syncthreads();

  // Early-exit for invalid/padding rows
  if (irow >= num_rows) return;
  batch_id = smem_batch_id[iwarp];
  token_id = smem_token_pos[iwarp];
  if (token_id < 0) return;

  // Load cos/sin (per-warp, needs __syncwarp for intra-warp visibility)
  {
    constexpr int kItemPerThread = 16 / sizeof(float);
    constexpr int kNumPacks = kQKHeadDim / kItemPerThread;
    const float *cos_sin_row = cos_sin_ptr + token_id * kQKHeadDim;
    if (ilane < kNumPacks) {
      int ioffset = ilane * kItemPerThread;
      store(&smem_cos_sin[iwarp][0] + ioffset, load<float, kItemPerThread>(cos_sin_row + ioffset));
    }
    __syncwarp();
  }

  // KV cache block addressing
  int block_idx_in_batch, block_row;
  kv_block_size_divider(block_idx_in_batch, block_row, token_id);
  int phys_block_id =
      kvcache_indices_ptr[batch_id * max_num_kv_block_per_batch + block_idx_in_batch];

  QType *k_cache_row_start = kcache_ptr + (int64_t)phys_block_id * (int64_t)kcache_block_offset +
                             block_row * (kNumKVHeads * kQKHeadDim);
  QType *v_cache_row_start = vcache_ptr + (int64_t)phys_block_id * (int64_t)vcache_block_offset +
                             block_row * (kNumKVHeads * kVHeadDim);

  const DType *qkv_row = in_qkv_ptr + irow * kNumElemPerRow;

  // ========= Process Q heads =========
#pragma unroll
  for (int q_head = 0; q_head < kNumQHeads; ++q_head) {
    const DType *q_src = qkv_row + q_head * kQKHeadDim;
    QType *q_dst = out_q_ptr + irow * kNumQHeads * kQKHeadDim + q_head * kQKHeadDim;

    vec_t<float, kNumItemPerThread> data = {0};

#pragma unroll
    for (int r = 0; r < kNumRoundsHalf; ++r) {
      int i = r * kWarpSize + ilane;
      if (i < kQKHeadDim / 2) {
        data[r * 2] = __bfloat162float(q_src[i]);
        data[r * 2 + 1] = __bfloat162float(q_src[i + kQKHeadDim / 2]);
      }
    }

    if constexpr (kNormPolicy == 2) {
      rms_norm_apply<kNumItemPerThread, kQKHeadDim>(data, smem_q_norm_w, ilane);
    }

#pragma unroll
    for (int r = 0; r < kNumRoundsHalf; ++r) {
      int i = r * kWarpSize + ilane;
      if (i < kQKHeadDim / 2) {
        rope_rotate_pair(data[r * 2], data[r * 2 + 1], smem_cos_sin[iwarp][i],
                         smem_cos_sin[iwarp][i + kQKHeadDim / 2]);
      }
    }

    if constexpr (kNormPolicy == 1) {
      rms_norm_apply<kNumItemPerThread, kQKHeadDim>(data, smem_q_norm_w, ilane);
    }

    // Q quantization
    float q_mult;
    if constexpr (kQuantPolicy == 1) {
      // dqskv: dynamic per-token per-head
      float max_abs = warp_abs_max<kNumItemPerThread>(data);
      float q_scale_val = max_abs / upper_max;
      if (ilane == 0) {
        if (is_prefill) {
          // Prefill layout: [batch_id, q_head, tok_in_chunk]
          int tok_in_chunk = irow - q_index_ptr[batch_id];
          q_scale_ptr[batch_id * kNumQHeads * max_seqlen_aligned + q_head * max_seqlen_aligned +
                      tok_in_chunk] = q_scale_val;
        } else {
          // Decode layout: [irow, q_head]
          q_scale_ptr[irow * kNumQHeads + q_head] = q_scale_val;
        }
      }
      q_mult = __frcp_rn(q_scale_val);
    } else if constexpr (kQuantPolicy == 2) {
      // sqskv: static per-tensor
      q_mult = q_scale_inv_ptr[0];
    }

    // Store FP8 Q
#pragma unroll
    for (int r = 0; r < kNumRoundsHalf; ++r) {
      int i = r * kWarpSize + ilane;
      if (i < kQKHeadDim / 2) {
        q_dst[i] = QType(data[r * 2] * q_mult);
        q_dst[i + kQKHeadDim / 2] = QType(data[r * 2 + 1] * q_mult);
      }
    }
  }

  // ========= Process K heads =========
  float k_scale_inv = __frcp_rn(k_scale_ptr[0]);
#pragma unroll
  for (int kv_head = 0; kv_head < kNumKVHeads; ++kv_head) {
    const DType *k_src = qkv_row + kNumQHeads * kQKHeadDim + kv_head * kQKHeadDim;

    // Zero split_k_flag inside K loop
    if (ilane == 0) {
      split_k_flag_ptr[batch_id * kNumKVHeads + kv_head] = 0;
    }

    vec_t<float, kNumItemPerThread> data = {0};

#pragma unroll
    for (int r = 0; r < kNumRoundsHalf; ++r) {
      int i = r * kWarpSize + ilane;
      if (i < kQKHeadDim / 2) {
        data[r * 2] = __bfloat162float(k_src[i]);
        data[r * 2 + 1] = __bfloat162float(k_src[i + kQKHeadDim / 2]);
      }
    }

    if constexpr (kNormPolicy == 2) {
      rms_norm_apply<kNumItemPerThread, kQKHeadDim>(data, smem_k_norm_w, ilane);
    }

#pragma unroll
    for (int r = 0; r < kNumRoundsHalf; ++r) {
      int i = r * kWarpSize + ilane;
      if (i < kQKHeadDim / 2) {
        rope_rotate_pair(data[r * 2], data[r * 2 + 1], smem_cos_sin[iwarp][i],
                         smem_cos_sin[iwarp][i + kQKHeadDim / 2]);
      }
    }

    if constexpr (kNormPolicy == 1) {
      rms_norm_apply<kNumItemPerThread, kQKHeadDim>(data, smem_k_norm_w, ilane);
    }

    QType *k_dst = (out_k_ptr != nullptr)
                       ? out_k_ptr + irow * kNumKVHeads * kQKHeadDim + kv_head * kQKHeadDim
                       : k_cache_row_start + kv_head * kQKHeadDim;

#pragma unroll
    for (int r = 0; r < kNumRoundsHalf; ++r) {
      int i = r * kWarpSize + ilane;
      if (i < kQKHeadDim / 2) {
        k_dst[i] = QType(data[r * 2] * k_scale_inv);
        k_dst[i + kQKHeadDim / 2] = QType(data[r * 2 + 1] * k_scale_inv);
      }
    }
  }

  // ========= Process V heads (no RoPE, bf16→fp8) =========
  {
    float v_scale_inv = __frcp_rn(v_scale_ptr[0]);
    using LoadDType = __nv_bfloat162;
    using PackQType = __nv_fp8x4_e4m3;
    constexpr int kNumVElemPerRow = kNumKVHeads * kVHeadDim;
    constexpr int kItemPerThread = 16 / sizeof(DType);
    static_assert(kNumVElemPerRow % kItemPerThread == 0, "");
    constexpr int kNumPackPerRow = kNumVElemPerRow / kItemPerThread;

    const DType *v_src = qkv_row + (kNumQHeads + kNumKVHeads) * kQKHeadDim;
    QType *v_dst = (out_v_ptr != nullptr) ? out_v_ptr + irow * kNumKVHeads * kVHeadDim
                                          : reinterpret_cast<QType *>(v_cache_row_start);

    constexpr int kNumLoadRound = ceil_div<kNumPackPerRow, kWarpSize>();
#pragma unroll
    for (int r = 0; r < kNumLoadRound; ++r) {
      int ioffset = (r * kWarpSize + ilane) * kItemPerThread;
      if (ioffset < kNumVElemPerRow) {
        auto vec_bf162 = load<LoadDType, kItemPerThread / 2>(v_src + ioffset);
        auto vec_float = to<float>(vec_bf162);
#pragma unroll
        for (int i = 0; i < size(vec_float); i++) {
          vec_float[i] = vec_float[i] * v_scale_inv;
        }
        store(v_dst + ioffset, to<PackQType>(vec_float));
      }
    }
  }
}

}  // namespace kernels

template <int kNumQHeads, int kNumKVHeads, int kQKHeadDim, int kVHeadDim>
void launch_rope_norm_store_kv(__nv_bfloat16 *out_q_ptr, __nv_bfloat16 *kcache_ptr,
                               __nv_bfloat16 *vcache_ptr, __nv_bfloat16 *out_k_ptr,
                               __nv_bfloat16 *out_v_ptr, const __nv_bfloat16 *in_qkv_ptr,
                               const float *cos_sin_ptr, const int *num_seqlen_per_req_ptr,
                               const int *q_index_ptr, const int *kvcache_indices_ptr,
                               const float *q_norm_weight_ptr, const float *k_norm_weight_ptr,
                               int kcache_block_offset, int vcache_block_offset, int num_batch,
                               int max_num_kv_block_per_batch,
                               cutlass::FastDivmod kv_block_size_divider, int num_rows,
                               int qk_norm_policy, cudaStream_t stream) {
  constexpr int kWarpsPerBlock = 4;
  constexpr int kWarpSize = 32;

  int num_compute_blocks = (num_rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
  dim3 block(kWarpsPerBlock * kWarpSize);
  dim3 grid(num_compute_blocks + num_batch);  // compute blocks + 1 clear block per request

  auto launch = [&](auto norm_tag) {
    constexpr int kNP = decltype(norm_tag)::value;
    kernels::rope_norm_store_kv_kernel<kWarpsPerBlock, kNumQHeads, kNumKVHeads, kQKHeadDim,
                                       kVHeadDim, kNP><<<grid, block, 0, stream>>>(
        out_q_ptr, kcache_ptr, vcache_ptr, out_k_ptr, out_v_ptr, in_qkv_ptr, cos_sin_ptr,
        num_seqlen_per_req_ptr, q_index_ptr, kvcache_indices_ptr, q_norm_weight_ptr,
        k_norm_weight_ptr, kcache_block_offset, vcache_block_offset, num_batch,
        max_num_kv_block_per_batch, kv_block_size_divider, num_rows, num_compute_blocks);
  };

  if (qk_norm_policy == 1) {
    launch(std::integral_constant<int, 1>{});
  } else if (qk_norm_policy == 2) {
    launch(std::integral_constant<int, 2>{});
  } else {
    launch(std::integral_constant<int, 0>{});
  }
}

void rope_norm_store_kv_async(__nv_bfloat16 *out_q_ptr, __nv_bfloat16 *kcache_ptr,
                              __nv_bfloat16 *vcache_ptr, __nv_bfloat16 *out_k_ptr,
                              __nv_bfloat16 *out_v_ptr, const __nv_bfloat16 *in_qkv_ptr,
                              const float *cos_sin_ptr, const int *num_seqlen_per_req_ptr,
                              const int *q_index_ptr, const int *kvcache_indices_ptr,
                              const float *q_norm_weight_ptr, const float *k_norm_weight_ptr,
                              int kcache_block_offset, int vcache_block_offset, int num_batch,
                              int max_num_kv_block_per_batch, int kv_block_size, int num_rows,
                              int num_q_heads, int num_kv_heads, int qk_head_dim, int v_head_dim,
                              bool is_prefill, int qk_norm_policy, cudaStream_t stream) {
  cutlass::FastDivmod kv_block_size_divider(kv_block_size);

  if (num_q_heads == 8 && num_kv_heads == 1 && qk_head_dim == 128 && v_head_dim == 128) {
    launch_rope_norm_store_kv<8, 1, 128, 128>(
        out_q_ptr, kcache_ptr, vcache_ptr, out_k_ptr, out_v_ptr, in_qkv_ptr, cos_sin_ptr,
        num_seqlen_per_req_ptr, q_index_ptr, kvcache_indices_ptr, q_norm_weight_ptr,
        k_norm_weight_ptr, kcache_block_offset, vcache_block_offset, num_batch,
        max_num_kv_block_per_batch, kv_block_size_divider, num_rows, qk_norm_policy, stream);
  } else if (num_q_heads == 64 && num_kv_heads == 8 && qk_head_dim == 128 && v_head_dim == 128) {
    launch_rope_norm_store_kv<64, 8, 128, 128>(
        out_q_ptr, kcache_ptr, vcache_ptr, out_k_ptr, out_v_ptr, in_qkv_ptr, cos_sin_ptr,
        num_seqlen_per_req_ptr, q_index_ptr, kvcache_indices_ptr, q_norm_weight_ptr,
        k_norm_weight_ptr, kcache_block_offset, vcache_block_offset, num_batch,
        max_num_kv_block_per_batch, kv_block_size_divider, num_rows, qk_norm_policy, stream);
  } else {
    throw std::invalid_argument("rope_norm_store_kv_async: unsupported config, got: q_heads=" +
                                std::to_string(num_q_heads) +
                                ", kv_heads=" + std::to_string(num_kv_heads) +
                                ", qk_head_dim=" + std::to_string(qk_head_dim) +
                                ", v_head_dim=" + std::to_string(v_head_dim));
  }
}

// Launch helper – dispatches kQuantPolicy + kNormPolicy at compile time
template <int kNumQHeads, int kNumKVHeads, int kQKHeadDim, int kVHeadDim>
void launch_rope_norm_store_kv_fp8(
    __nv_fp8_e4m3 *out_q_ptr, __nv_fp8_e4m3 *kcache_ptr, __nv_fp8_e4m3 *vcache_ptr,
    __nv_fp8_e4m3 *out_k_ptr, __nv_fp8_e4m3 *out_v_ptr, int32_t *split_k_flag_ptr,
    float *q_scale_ptr, const __nv_bfloat16 *in_qkv_ptr, const float *cos_sin_ptr,
    const int *num_seqlen_per_req_ptr, const int *q_index_ptr, const int *kvcache_indices_ptr,
    const float *q_norm_weight_ptr, const float *k_norm_weight_ptr, const float *k_scale_ptr,
    const float *v_scale_ptr, const float *q_scale_inv_ptr, float upper_max, int max_seqlen_aligned,
    int kcache_block_offset, int vcache_block_offset, int num_batch, int max_num_kv_block_per_batch,
    cutlass::FastDivmod kv_block_size_divider, int num_rows, int qk_norm_policy, int quant_policy,
    bool is_prefill, cudaStream_t stream) {
  constexpr int kWarpsPerBlock = 4;
  constexpr int kWarpSize = 32;

  int num_compute_blocks = (num_rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
  dim3 block(kWarpsPerBlock * kWarpSize);
  dim3 grid(num_compute_blocks + num_batch);

  auto launch = [&](auto quant_tag, auto norm_tag) {
    constexpr int kQP = decltype(quant_tag)::value;
    constexpr int kNP = decltype(norm_tag)::value;
    kernels::rope_norm_store_kv_fp8_kernel<kQP, kWarpsPerBlock, kNumQHeads, kNumKVHeads, kQKHeadDim,
                                           kVHeadDim, kNP><<<grid, block, 0, stream>>>(
        out_q_ptr, kcache_ptr, vcache_ptr, out_k_ptr, out_v_ptr, split_k_flag_ptr, q_scale_ptr,
        in_qkv_ptr, cos_sin_ptr, num_seqlen_per_req_ptr, q_index_ptr, kvcache_indices_ptr,
        q_norm_weight_ptr, k_norm_weight_ptr, k_scale_ptr, v_scale_ptr, q_scale_inv_ptr, upper_max,
        max_seqlen_aligned, kcache_block_offset, vcache_block_offset, num_batch,
        max_num_kv_block_per_batch, kv_block_size_divider, num_rows, num_compute_blocks,
        is_prefill);
  };

  auto dispatch_norm = [&](auto quant_tag) {
    if (qk_norm_policy == 1) {
      launch(quant_tag, std::integral_constant<int, 1>{});
    } else if (qk_norm_policy == 2) {
      launch(quant_tag, std::integral_constant<int, 2>{});
    } else {
      launch(quant_tag, std::integral_constant<int, 0>{});
    }
  };

  if (quant_policy == 1) {
    dispatch_norm(std::integral_constant<int, 1>{});
  } else {
    dispatch_norm(std::integral_constant<int, 2>{});
  }
}

void rope_norm_store_kv_fp8_async(
    __nv_fp8_e4m3 *out_q_ptr, __nv_fp8_e4m3 *kcache_ptr, __nv_fp8_e4m3 *vcache_ptr,
    __nv_fp8_e4m3 *out_k_ptr, __nv_fp8_e4m3 *out_v_ptr, int32_t *split_k_flag_ptr,
    float *q_scale_ptr, const __nv_bfloat16 *in_qkv_ptr, const float *cos_sin_ptr,
    const int *num_seqlen_per_req_ptr, const int *q_index_ptr, const int *kvcache_indices_ptr,
    const float *q_norm_weight_ptr, const float *k_norm_weight_ptr, const float *k_scale_ptr,
    const float *v_scale_ptr, const float *q_scale_inv_ptr, float upper_max, int max_seqlens,
    int kcache_block_offset, int vcache_block_offset, int num_batch, int max_num_kv_block_per_batch,
    int kv_block_size, int num_rows, int num_q_heads, int num_kv_heads, int qk_head_dim,
    int v_head_dim, bool is_prefill, int qk_norm_policy, int quant_policy, cudaStream_t stream) {
  cutlass::FastDivmod kv_block_size_divider(kv_block_size);
  int max_seqlen_aligned = ((max_seqlens + 127) / 128) * 128;

  if (num_q_heads == 8 && num_kv_heads == 1 && qk_head_dim == 128 && v_head_dim == 128) {
    launch_rope_norm_store_kv_fp8<8, 1, 128, 128>(
        out_q_ptr, kcache_ptr, vcache_ptr, out_k_ptr, out_v_ptr, split_k_flag_ptr, q_scale_ptr,
        in_qkv_ptr, cos_sin_ptr, num_seqlen_per_req_ptr, q_index_ptr, kvcache_indices_ptr,
        q_norm_weight_ptr, k_norm_weight_ptr, k_scale_ptr, v_scale_ptr, q_scale_inv_ptr, upper_max,
        max_seqlen_aligned, kcache_block_offset, vcache_block_offset, num_batch,
        max_num_kv_block_per_batch, kv_block_size_divider, num_rows, qk_norm_policy, quant_policy,
        is_prefill, stream);
  } else if (num_q_heads == 64 && num_kv_heads == 8 && qk_head_dim == 128 && v_head_dim == 128) {
    launch_rope_norm_store_kv_fp8<64, 8, 128, 128>(
        out_q_ptr, kcache_ptr, vcache_ptr, out_k_ptr, out_v_ptr, split_k_flag_ptr, q_scale_ptr,
        in_qkv_ptr, cos_sin_ptr, num_seqlen_per_req_ptr, q_index_ptr, kvcache_indices_ptr,
        q_norm_weight_ptr, k_norm_weight_ptr, k_scale_ptr, v_scale_ptr, q_scale_inv_ptr, upper_max,
        max_seqlen_aligned, kcache_block_offset, vcache_block_offset, num_batch,
        max_num_kv_block_per_batch, kv_block_size_divider, num_rows, qk_norm_policy, quant_policy,
        is_prefill, stream);
  } else {
    throw std::invalid_argument("rope_norm_store_kv_fp8_async: unsupported config, got: q_heads=" +
                                std::to_string(num_q_heads) +
                                ", kv_heads=" + std::to_string(num_kv_heads) +
                                ", qk_head_dim=" + std::to_string(qk_head_dim) +
                                ", v_head_dim=" + std::to_string(v_head_dim));
  }
}

}  // namespace rope
}  // namespace hpc
