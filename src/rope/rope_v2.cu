// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdio.h>

#include <string>

#include "cutlass/fast_math.h"
#include "src/hadamard/hadamard_device.cuh"
#include "src/rope/rope.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace rope_v2 {

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
  for (int i = 0; i < kNumItemPerThread; ++i) {
    sum_sq += data[i] * data[i];
  }
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
  for (int i = 0; i < kN; ++i) {
    m = fmaxf(m, fabsf(data[i]));
  }
  return warp_reduce_max_xor(m);
}

template <typename T, int N>
__device__ __forceinline__ void hadamard_128(vec_t<T, N> &data, int ilane) {
  constexpr float kInvSqrt128 = 0.08838834764831845f;  // 1/sqrt(128)
  hpc::hadamard::device::hadamard_n128_warp(data, ilane);
#pragma unroll
  for (int i = 0; i < N; ++i) {
    data[i] *= kInvSqrt128;
  }
}

template <int kWarpsPerBlock, int kNumQHeads, int kNumKVHeads, int kQKHeadDim, int kVHeadDim,
          int kNormPolicy>
__global__ void rope_norm_store_kv_kernel(
    __nv_bfloat16 *out_q_ptr, __nv_bfloat16 *kcache_ptr, __nv_bfloat16 *vcache_ptr,
    __nv_bfloat16 *out_k_ptr, __nv_bfloat16 *out_v_ptr, const __nv_bfloat16 *in_qkv_ptr,
    const float *cos_sin_ptr, const int *num_seqlen_per_req_ptr, const int *q_index_ptr,
    const int *kvcache_indices_ptr, const float *q_norm_weight_ptr, const float *k_norm_weight_ptr,
    Strides4D kc, Strides4D vc, Strides2D ki, int num_batch,
    cutlass::FastDivmod kv_block_size_divider, int num_rows, int num_compute_blocks) {
  using DType = __nv_bfloat16;

  // PDL: wait for the predecessor kernel's writes to be visible.
  cudaGridDependencySynchronize();

  // bidy = kNumKVHeads: each CTA handles one KV head and its kQPerKV Q heads
  constexpr int kQPerKV = kNumQHeads / kNumKVHeads;
  const int bidy = blockIdx.y;  // kv_head index [0, kNumKVHeads)

  constexpr int kWarpSize = 32;
  constexpr int kNumElemPerRow =
      kNumQHeads * kQKHeadDim + kNumKVHeads * kQKHeadDim + kNumKVHeads * kVHeadDim;
  constexpr int kNumRoundsHalf = ceil_div<kQKHeadDim / 2, kWarpSize>();
  constexpr int kNumItemPerThread = kNumRoundsHalf * 2;

  int tid = threadIdx.x;
  int bidx = blockIdx.x;
  int iwarp = tid / kWarpSize;
  int ilane = tid % kWarpSize;

  __shared__ float smem_cos_sin[kWarpsPerBlock][kQKHeadDim];
  __shared__ float smem_q_norm_w[kQKHeadDim];
  __shared__ float smem_k_norm_w[kQKHeadDim];
  __shared__ int smem_batch_id[kWarpsPerBlock];
  __shared__ int smem_token_pos[kWarpsPerBlock];

  // ---- Clear blocks: bidx >= num_compute_blocks → one block per request -----
  if (bidx >= num_compute_blocks) {
    int req_id = bidx - num_compute_blocks;
    if (req_id >= num_batch) {
      return;
    }

    // Last token of this request determines the clear range
    int last_token_pos = num_seqlen_per_req_ptr[req_id] - 1;
    if (last_token_pos < 0) {
      return;
    }

    int block_idx_in_batch, pos_in_block;
    kv_block_size_divider(block_idx_in_batch, pos_in_block, last_token_pos);
    int phys_block_id = kvcache_indices_ptr[req_id * ki.s0 + block_idx_in_batch];

    int zero_from = pos_in_block + 1;
    int zero_to = kv_block_size_divider.divisor;
    if (zero_from < zero_to) {
      constexpr int kKItemPerThread = 16 / sizeof(DType);
      vec_t<DType, kKItemPerThread> zero_vec;
#pragma unroll
      for (int z = 0; z < kKItemPerThread; ++z) {
        zero_vec[z] = DType(0);
      }
      // Each grid.y clears only its own kv_head = bidy; warps cooperate across rows
      for (int row = zero_from + iwarp; row < zero_to; row += kWarpsPerBlock) {
        DType *k_row = kcache_ptr + phys_block_id * kc.s0 + row * kc.s1 + bidy * kc.s2;
        for (int idx = ilane * kKItemPerThread; idx < kQKHeadDim;
             idx += kWarpSize * kKItemPerThread) {
          store(k_row + idx, zero_vec);
        }
        DType *v_row = vcache_ptr + phys_block_id * vc.s0 + row * vc.s1 + bidy * vc.s2;
        for (int idx = ilane * kKItemPerThread; idx < kVHeadDim;
             idx += kWarpSize * kKItemPerThread) {
          store(v_row + idx, zero_vec);
        }
      }
    }
  } else {
    // Search q_index to find batch_id
    int batch_id = 0;
    int token_id = 0;
    int irow = bidx * kWarpsPerBlock + iwarp;

    // First kWarpsPerBlock threads do the q_index search for the whole block
    if (tid < kWarpsPerBlock) {
      smem_batch_id[tid] = -1;
      smem_token_pos[tid] = -1;
    }
    __syncthreads();

    // use one block to find row<->batch_id
    constexpr int kBatchSize = kWarpSize * kWarpsPerBlock;
    int n_find_batch = (num_batch + kBatchSize - 1) / (kBatchSize);
    for (int round = 0; round < n_find_batch; round++) {
      int b = round * kBatchSize + tid;
      if (b < num_batch) {
        int l = q_index_ptr[b];
        int r = q_index_ptr[b + 1];
#pragma unroll
        for (int iw = 0; iw < kWarpsPerBlock; iw++) {
          int global_row = bidx * kWarpsPerBlock + iw;
          if (l <= global_row && global_row < r) {
            smem_batch_id[iw] = b;
            smem_token_pos[iw] = global_row + num_seqlen_per_req_ptr[b] - q_index_ptr[b + 1];
          }
        }
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
    if (irow >= num_rows) {
      return;
    }
    batch_id = smem_batch_id[iwarp];
    token_id = smem_token_pos[iwarp];
    if (token_id < 0) {
      return;
    }

    //  Load cos_sin
    {
      constexpr int kItemPerThread = 16 / sizeof(float);
      constexpr int kNumPacks = kQKHeadDim / kItemPerThread;
      static_assert(kQKHeadDim % kItemPerThread == 0, "");
      static_assert(kNumPacks <= kWarpSize, "");
      const float *cos_sin_row = cos_sin_ptr + token_id * kQKHeadDim;
      if (ilane < kNumPacks) {
        int ioffset = ilane * kItemPerThread;
        store(&smem_cos_sin[iwarp][0] + ioffset,
              load<float, kItemPerThread>(cos_sin_row + ioffset));
      }
      __syncwarp();
    }

    //  KV cache block addressing
    int block_idx_in_batch, block_row;
    kv_block_size_divider(block_idx_in_batch, block_row, token_id);
    int phys_block_id = kvcache_indices_ptr[batch_id * ki.s0 + block_idx_in_batch];

    // Pre-compute block+token offset; head offset added inside the per-head loop
    int64_t kc_tok_offset = phys_block_id * kc.s0 + block_row * kc.s1;
    int64_t vc_tok_offset = phys_block_id * vc.s0 + block_row * vc.s1;

    const DType *qkv_row = in_qkv_ptr + irow * kNumElemPerRow;

    // Process Q heads for this group: [bidy*kQPerKV, (bidy+1)*kQPerKV)
#pragma unroll
    for (int dq = 0; dq < kQPerKV; ++dq) {
      int q_head = bidy * kQPerKV + dq;
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

    // Process K head bidy – norm, RoPE, write to KV cache (or out_k_ptr if non-null)
    {
      const DType *k_src = qkv_row + kNumQHeads * kQKHeadDim + bidy * kQKHeadDim;

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

      DType *k_dst = (out_k_ptr != nullptr)
                         ? out_k_ptr + irow * kNumKVHeads * kQKHeadDim + bidy * kQKHeadDim
                         : kcache_ptr + kc_tok_offset + bidy * kc.s2;

#pragma unroll
      for (int r = 0; r < kNumRoundsHalf; ++r) {
        int i = r * kWarpSize + ilane;
        if (i < kQKHeadDim / 2) {
          k_dst[i] = __float2bfloat16(data[r * 2]);
          k_dst[i + kQKHeadDim / 2] = __float2bfloat16(data[r * 2 + 1]);
        }
      }
    }

    // Process V head bidy – no RoPE; out_v_ptr is a fallback path, use per-head loop uniformly
    {
      constexpr int kItemPerThread = 16 / sizeof(DType);
      static_assert(kVHeadDim % kItemPerThread == 0,
                    "kVHeadDim must be multiple of kItemPerThread");
      constexpr int kNumPackPerHead = kVHeadDim / kItemPerThread;
      constexpr int kNumLoadRound = ceil_div<kNumPackPerHead, kWarpSize>();

      const DType *v_src_head =
          qkv_row + (kNumQHeads + kNumKVHeads) * kQKHeadDim + bidy * kVHeadDim;
      DType *v_dst_head = (out_v_ptr != nullptr)
                              ? out_v_ptr + irow * kNumKVHeads * kVHeadDim + bidy * kVHeadDim
                              : vcache_ptr + vc_tok_offset + bidy * vc.s2;
#pragma unroll
      for (int r = 0; r < kNumLoadRound; ++r) {
        int ioffset = (r * kWarpSize + ilane) * kItemPerThread;
        if (ioffset < kVHeadDim) {
          store(v_dst_head + ioffset, load<DType, kItemPerThread>(v_src_head + ioffset));
        }
      }
    }
  }

  // PDL: signal that the dependent kernel can begin launching.
  cudaTriggerProgrammaticLaunchCompletion();
}

template <int kQuantPolicy, bool kIsPrefill, int kWarpsPerBlock, int kNumQHeads, int kNumKVHeads,
          int kQKHeadDim, int kVHeadDim, int kNormPolicy>
__global__ void rope_norm_store_kv_fp8_kernel(
    __nv_fp8_e4m3 *out_q_ptr, __nv_fp8_e4m3 *kcache_ptr, __nv_fp8_e4m3 *vcache_ptr,
    __nv_fp8_e4m3 *out_k_ptr, __nv_fp8_e4m3 *out_v_ptr, int32_t *split_k_flag_ptr,
    float *q_scale_ptr, const __nv_bfloat16 *in_qkv_ptr, const float *cos_sin_ptr,
    const int *num_seqlen_per_req_ptr, const int *q_index_ptr, const int *kvcache_indices_ptr,
    const float *q_norm_weight_ptr, const float *k_norm_weight_ptr, float *k_scale_ptr,
    const float *v_scale_ptr, const float *q_scale_inv_ptr, float upper_max, int max_seqlen_aligned,
    Strides4D kc, Strides4D ks, Strides4D vc, Strides2D ki, int num_batch,
    cutlass::FastDivmod kv_block_size_divider, int num_rows, int num_compute_blocks) {
  using DType = __nv_bfloat16;
  using QType = __nv_fp8_e4m3;

  // PDL: wait for the predecessor kernel's writes to be visible.
  cudaGridDependencySynchronize();

  // grid.y = kNumKVHeads: each CTA handles one KV head and its kQPerKV Q heads
  constexpr int kQPerKV = kNumQHeads / kNumKVHeads;
  const int bidy = blockIdx.y;  // kv_head index [0, kNumKVHeads)

  constexpr int kWarpSize = 32;
  constexpr int kNumElemPerRow =
      kNumQHeads * kQKHeadDim + kNumKVHeads * kQKHeadDim + kNumKVHeads * kVHeadDim;
  // Per-block smem stages only the slice this CTA actually consumes:
  //   - kQPerKV Q heads (reused across the inner Q-head loop)
  //   - 1 K head (single read but RoPE wants stride-64 paired access)
  // V head is read directly from global later (no RoPE, no reuse).
  constexpr int kSmemQPerRow = kQPerKV * kQKHeadDim;
  constexpr int kSmemKPerRow = kQKHeadDim;
  constexpr int kSmemElemPerRow = kSmemQPerRow + kSmemKPerRow;
  constexpr int kNumRoundsHalf = ceil_div<kQKHeadDim / 2, kWarpSize>();
  constexpr int kNumItemPerThread = kNumRoundsHalf * 2;

  int tid = threadIdx.x;
  int bidx = blockIdx.x;
  int iwarp = tid / kWarpSize;
  int ilane = tid % kWarpSize;

  // Shared memory
  __shared__ float smem_cos_sin[kWarpsPerBlock][kQKHeadDim];
  __shared__ float smem_q_norm_w[kQKHeadDim];
  __shared__ float smem_k_norm_w[kQKHeadDim];
  __shared__ int smem_batch_id[kWarpsPerBlock];
  __shared__ int smem_token_pos[kWarpsPerBlock];
  // Dynamic shared memory: kWarpsPerBlock * kNumElemPerRow DType elements
  extern __shared__ DType smem_qkv[];

  // ---- Clear blocks: bidx >= num_compute_blocks → one warp per request -----
  if (bidx >= num_compute_blocks) {
    int req_id = (bidx - num_compute_blocks) * kWarpsPerBlock + iwarp;
    if (req_id >= num_batch) {
      return;
    }

    int last_token_pos = num_seqlen_per_req_ptr[req_id] - 1;
    if (last_token_pos < 0) {
      return;
    }

    int block_idx_in_batch, pos_in_block;
    kv_block_size_divider(block_idx_in_batch, pos_in_block, last_token_pos);
    int phys_block_id = kvcache_indices_ptr[req_id * ki.s0 + block_idx_in_batch];

    if constexpr (!kIsPrefill) {
      if (pos_in_block > 0) {
        return;
      }
    }
    int zero_from = pos_in_block + 1;
    const int zero_to = kv_block_size_divider.divisor;
    constexpr int kKItemPerThread = 16 / sizeof(QType);
    constexpr int kKItemPerRound = kWarpSize * kKItemPerThread;
    constexpr int kKRounds = (kQKHeadDim + kKItemPerRound - 1) / kKItemPerRound;
    constexpr int kVRounds = (kVHeadDim + kKItemPerRound - 1) / kKItemPerRound;
    vec_t<QType, kKItemPerThread> zero_vec;
#pragma unroll
    for (int z = 0; z < kKItemPerThread; ++z) {
      zero_vec[z] = QType(0);
    }
    for (int row = zero_from; row < zero_to; ++row) {
      // Each grid.y clears only its own kv_head = bidy
      QType *k_row = kcache_ptr + phys_block_id * kc.s0 + row * kc.s1 + bidy * kc.s2;
#pragma unroll
      for (int round = 0; round < kKRounds; round++) {
        int idx = ilane * kKItemPerThread + round * kKItemPerRound;
        if (idx < kQKHeadDim) {
          store(k_row + idx, zero_vec);
        }
      }
      QType *v_row = vcache_ptr + phys_block_id * vc.s0 + row * vc.s1 + bidy * vc.s2;
#pragma unroll
      for (int round = 0; round < kVRounds; round++) {
        int idx = ilane * kKItemPerThread + round * kKItemPerRound;
        if (idx < kVHeadDim) {
          store(v_row + idx, zero_vec);
        }
      }
    }
  } else {
    // Determine batch_id and token position — unified for prefill and decode
    int batch_id = 0;
    int token_id = 0;
    int irow = bidx * kWarpsPerBlock + iwarp;

    if (tid < kWarpsPerBlock) {
      smem_batch_id[tid] = -1;
      smem_token_pos[tid] = -1;
    }
    __syncthreads();

    // use one block to find row<->batch_id
    constexpr int kBatchSize = kWarpSize * kWarpsPerBlock;
    int n_find_batch = (num_batch + kBatchSize - 1) / (kBatchSize);
    for (int round = 0; round < n_find_batch; round++) {
      int b = round * kBatchSize + tid;
      if (b < num_batch) {
        int l = q_index_ptr[b];
        int r = q_index_ptr[b + 1];
#pragma unroll
        for (int iw = 0; iw < kWarpsPerBlock; iw++) {
          int global_row = bidx * kWarpsPerBlock + iw;
          if (l <= global_row && global_row < r) {
            smem_batch_id[iw] = b;
            smem_token_pos[iw] = global_row + num_seqlen_per_req_ptr[b] - q_index_ptr[b + 1];
          }
        }
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

    // Preload only this block's Q-slice (kQPerKV heads) + this block's 1 K head.
    // V head is read directly from global later (no RoPE, no reuse, linear access).
    {
      constexpr int kElemPerThread = 16 / sizeof(DType);
      static_assert(kSmemQPerRow % kElemPerThread == 0, "Q slice must align to packed load width");
      static_assert(kSmemKPerRow % kElemPerThread == 0, "K slice must align to packed load width");
      constexpr int kQPacks = kSmemQPerRow / kElemPerThread;
      constexpr int kKPacks = kSmemKPerRow / kElemPerThread;
      constexpr int kQRounds = ceil_div<kQPacks, kWarpSize>();
      int irow_local = bidx * kWarpsPerBlock + iwarp;
      if (irow_local < num_rows) {
        const DType *src_q = in_qkv_ptr + irow_local * kNumElemPerRow + bidy * kSmemQPerRow;
        const DType *src_k =
            in_qkv_ptr + irow_local * kNumElemPerRow + kNumQHeads * kQKHeadDim + bidy * kQKHeadDim;
        DType *dst_q = smem_qkv + iwarp * kSmemElemPerRow;
        DType *dst_k = dst_q + kSmemQPerRow;
#pragma unroll
        for (int i = 0; i < kQRounds; ++i) {
          int ioffset = (i * kWarpSize + ilane) * kElemPerThread;
          if (ioffset < kSmemQPerRow) {
            store(dst_q + ioffset, load<DType, kElemPerThread>(src_q + ioffset));
          }
        }
        // K slice: single head, fits in one round (ilane < kKPacks)
        if (ilane < kKPacks) {
          int ioffset = ilane * kElemPerThread;
          store(dst_k + ioffset, load<DType, kElemPerThread>(src_k + ioffset));
        }
      }
    }

    // Single barrier: makes batch_id, token_pos, and norm weights visible
    __syncthreads();

    // Early-exit for invalid/padding rows
    if (irow >= num_rows) {
      return;
    }
    batch_id = smem_batch_id[iwarp];
    token_id = smem_token_pos[iwarp];
    if (token_id < 0) {
      return;
    }

    // Load cos/sin (per-warp, needs __syncwarp for intra-warp visibility)
    {
      constexpr int kItemPerThread = 16 / sizeof(float);
      constexpr int kNumPacks = kQKHeadDim / kItemPerThread;
      const float *cos_sin_row = cos_sin_ptr + token_id * kQKHeadDim;
      if (ilane < kNumPacks) {
        int ioffset = ilane * kItemPerThread;
        store(&smem_cos_sin[iwarp][0] + ioffset,
              load<float, kItemPerThread>(cos_sin_row + ioffset));
      }
      __syncwarp();
    }

    // KV cache block addressing
    int block_idx_in_batch, block_row;
    kv_block_size_divider(block_idx_in_batch, block_row, token_id);
    int phys_block_id = kvcache_indices_ptr[batch_id * ki.s0 + block_idx_in_batch];

    // Pre-compute block+token offset; head offset added inside per-head loops
    int64_t kc_tok_offset = phys_block_id * kc.s0 + block_row * kc.s1;
    int64_t vc_tok_offset = phys_block_id * vc.s0 + block_row * vc.s1;

    const DType *qkv_row = smem_qkv + iwarp * kSmemElemPerRow;

    // ========= Process Q heads for this group: [bidy*kQPerKV, (bidy+1)*kQPerKV) =========
#pragma unroll
    for (int dq = 0; dq < kQPerKV; ++dq) {
      int q_head = bidy * kQPerKV + dq;
      // Q heads are stored compactly in smem starting at offset 0 (only this group's Q heads)
      const DType *q_src = qkv_row + dq * kQKHeadDim;
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

      if constexpr (kQuantPolicy == 3) {
        static_assert(kQKHeadDim == 128, "kQuantPolicy=3 (hadamard) requires kQKHeadDim == 128");
        hadamard_128(data, ilane);
      }

      // Q quantization
      float q_mult;
      if constexpr (kQuantPolicy == 0 || kQuantPolicy == 1 || kQuantPolicy == 3) {
        // dqskv (and dqksv+hadamard): dynamic per-token per-head
        float max_abs = warp_abs_max<kNumItemPerThread>(data);
        float q_scale_val = max_abs / upper_max;
        if (ilane == 0) {
          if constexpr (kIsPrefill) {
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

    // ========= Process K head bidy =========
    {
      // K head sits right after the Q slice in smem
      const DType *k_src = qkv_row + kSmemQPerRow;

      // Zero split_k_flag for this kv_head
      if (ilane == 0) {
        split_k_flag_ptr[batch_id * kNumKVHeads + bidy] = 0;
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

      if constexpr (kQuantPolicy == 3) {
        static_assert(kQKHeadDim == 128, "kQuantPolicy=3 (hadamard) requires kQKHeadDim == 128");
        hadamard_128(data, ilane);
      }

      QType *k_dst = (out_k_ptr != nullptr)
                         ? out_k_ptr + irow * kNumKVHeads * kQKHeadDim + bidy * kQKHeadDim
                         : kcache_ptr + kc_tok_offset + bidy * kc.s2;

      // K quantization
      float k_mult;
      if constexpr (kQuantPolicy == 0 || kQuantPolicy == 3) {
        // Dynamic per-token per-head quantization (with optional hadamard)
        float max_abs = warp_abs_max<kNumItemPerThread>(data);
        float k_scale_val = max_abs / upper_max;
        if (ilane == 0) {
          // k_scale layout: [num_block, R, num_kv_heads, L]
          // where L = head_dim * sizeof(kv_type) / sizeof(float), R = block_size / L
          constexpr int L = kQKHeadDim * sizeof(QType) / sizeof(float);
          int r = block_row / L;
          int l = block_row % L;
          k_scale_ptr[phys_block_id * ks.s0 + r * ks.s1 + bidy * ks.s2 + l] = k_scale_val;
        }
        k_mult = __frcp_rn(k_scale_val);
      } else if constexpr (kQuantPolicy == 1 || kQuantPolicy == 2) {
        // Static per-tensor quantization
        k_mult = __frcp_rn(k_scale_ptr[0]);
      }

#pragma unroll
      for (int r = 0; r < kNumRoundsHalf; ++r) {
        int i = r * kWarpSize + ilane;
        if (i < kQKHeadDim / 2) {
          k_dst[i] = QType(data[r * 2] * k_mult);
          k_dst[i + kQKHeadDim / 2] = QType(data[r * 2 + 1] * k_mult);
        }
      }
    }

    // ========= Process V head bidy (no RoPE, bf16→fp8, unified path) =========
    {
      float v_mult;
      using LoadDType = __nv_bfloat162;
      using PackQType = __nv_fp8x4_e4m3;
      constexpr int kItemPerThread = 16 / sizeof(DType);
      static_assert(kVHeadDim % kItemPerThread == 0,
                    "kVHeadDim must be multiple of kItemPerThread");
      constexpr int kNumPackPerHead = kVHeadDim / kItemPerThread;
      constexpr int kNumLoadRound = ceil_div<kNumPackPerHead, kWarpSize>();

      // V is not staged: read directly from global to register
      // (no RoPE, no reuse, linear vectorized access)
      const DType *v_src_head = in_qkv_ptr + irow * kNumElemPerRow +
                                (kNumQHeads + kNumKVHeads) * kQKHeadDim + bidy * kVHeadDim;
      QType *v_dst_head = (out_v_ptr != nullptr)
                              ? out_v_ptr + irow * kNumKVHeads * kVHeadDim + bidy * kVHeadDim
                              : vcache_ptr + vc_tok_offset + bidy * vc.s2;

      if constexpr (kQuantPolicy == 0 || kQuantPolicy == 3) {
        v_mult = __frcp_rn(v_scale_ptr[bidy]);
      } else {
        v_mult = __frcp_rn(v_scale_ptr[0]);
      }
#pragma unroll
      for (int r = 0; r < kNumLoadRound; ++r) {
        int ioffset = (r * kWarpSize + ilane) * kItemPerThread;
        if (ioffset < kVHeadDim) {
          auto vec_bf162 = load<LoadDType, kItemPerThread / 2>(v_src_head + ioffset);
          auto vec_float = to<float>(vec_bf162);
#pragma unroll
          for (int i = 0; i < size(vec_float); i++) {
            vec_float[i] = vec_float[i] * v_mult;
          }
          store(v_dst_head + ioffset, to<PackQType>(vec_float));
        }
      }
    }
  }

  // PDL: signal that the dependent kernel can begin launching.
  cudaTriggerProgrammaticLaunchCompletion();
}
}  // namespace kernels

template <int kNumQHeads, int kNumKVHeads, int kQKHeadDim, int kVHeadDim>
void launch_rope_norm_store_kv(__nv_bfloat16 *out_q_ptr, __nv_bfloat16 *kcache_ptr,
                               __nv_bfloat16 *vcache_ptr, __nv_bfloat16 *out_k_ptr,
                               __nv_bfloat16 *out_v_ptr, const __nv_bfloat16 *in_qkv_ptr,
                               const float *cos_sin_ptr, const int *num_seqlen_per_req_ptr,
                               const int *q_index_ptr, const int *kvcache_indices_ptr,
                               const float *q_norm_weight_ptr, const float *k_norm_weight_ptr,
                               Strides4D kc, Strides4D vc, Strides2D ki, int num_batch,
                               cutlass::FastDivmod kv_block_size_divider, int num_rows,
                               int qk_norm_policy, cudaStream_t stream) {
  constexpr int kWarpsPerBlock = 4;
  constexpr int kWarpSize = 32;

  int num_compute_blocks = (num_rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
  dim3 block(kWarpsPerBlock * kWarpSize);
  dim3 grid(num_compute_blocks + num_batch,
            kNumKVHeads);  // compute blocks + 1 clear block per request

  auto launch = [&](auto norm_tag) {
    constexpr int kNP = decltype(norm_tag)::value;
    auto *kernel = kernels::rope_norm_store_kv_kernel<kWarpsPerBlock, kNumQHeads, kNumKVHeads,
                                                      kQKHeadDim, kVHeadDim, kNP>;

    // PDL is always on for rope_v2 kernels.
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    config.attrs = attribute;
    config.numAttrs = 1;

    cudaLaunchKernelEx(&config, kernel, out_q_ptr, kcache_ptr, vcache_ptr, out_k_ptr, out_v_ptr,
                       in_qkv_ptr, cos_sin_ptr, num_seqlen_per_req_ptr, q_index_ptr,
                       kvcache_indices_ptr, q_norm_weight_ptr, k_norm_weight_ptr, kc, vc, ki,
                       num_batch, kv_block_size_divider, num_rows, num_compute_blocks);
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
                              Strides4D kc_strides, Strides4D vc_strides, Strides2D ki_strides,
                              int num_batch, int kv_block_size, int num_rows, int num_q_heads,
                              int num_kv_heads, int qk_head_dim, int v_head_dim, bool is_prefill,
                              int qk_norm_policy, cudaStream_t stream) {
  cutlass::FastDivmod kv_block_size_divider(kv_block_size);

  if (qk_head_dim == 128 && v_head_dim == 128) {
    constexpr int kQKHeadDim = 128;
    constexpr int kVHeadDim = 128;
    auto dispatch_heads = [&](auto q_heads, auto kv_heads) {
      constexpr int kNumQHeads = decltype(q_heads)::value;
      constexpr int kNumKVHeads = decltype(kv_heads)::value;
      launch_rope_norm_store_kv<kNumQHeads, kNumKVHeads, kQKHeadDim, kVHeadDim>(
          out_q_ptr, kcache_ptr, vcache_ptr, out_k_ptr, out_v_ptr, in_qkv_ptr, cos_sin_ptr,
          num_seqlen_per_req_ptr, q_index_ptr, kvcache_indices_ptr, q_norm_weight_ptr,
          k_norm_weight_ptr, kc_strides, vc_strides, ki_strides, num_batch, kv_block_size_divider,
          num_rows, qk_norm_policy, stream);
    };
    if (num_q_heads == 8 && num_kv_heads == 1) {
      dispatch_heads(std::integral_constant<int, 8>{}, std::integral_constant<int, 1>{});
    } else if (num_q_heads == 16 && num_kv_heads == 2) {
      dispatch_heads(std::integral_constant<int, 16>{}, std::integral_constant<int, 2>{});
    } else if (num_q_heads == 32 && num_kv_heads == 4) {
      dispatch_heads(std::integral_constant<int, 32>{}, std::integral_constant<int, 4>{});
    } else if (num_q_heads == 64 && num_kv_heads == 8) {
      dispatch_heads(std::integral_constant<int, 64>{}, std::integral_constant<int, 8>{});
    } else if (num_q_heads == 4 && num_kv_heads == 1) {
      dispatch_heads(std::integral_constant<int, 4>{}, std::integral_constant<int, 1>{});
    } else {
      throw std::invalid_argument("rope_norm_store_kv_async: unsupported config, got: q_heads=" +
                                  std::to_string(num_q_heads) +
                                  ", kv_heads=" + std::to_string(num_kv_heads) +
                                  ", qk_head_dim=" + std::to_string(qk_head_dim) +
                                  ", v_head_dim=" + std::to_string(v_head_dim));
    }
  } else {
    throw std::invalid_argument("rope_norm_store_kv_async: unsupported config, got: q_heads=" +
                                std::to_string(num_q_heads) +
                                ", kv_heads=" + std::to_string(num_kv_heads) +
                                ", qk_head_dim=" + std::to_string(qk_head_dim) +
                                ", v_head_dim=" + std::to_string(v_head_dim));
  }
}

// Launch helper – dispatches kQuantPolicy + kNormPolicy + kIsPrefill at compile time
template <int kNumQHeads, int kNumKVHeads, int kQKHeadDim, int kVHeadDim>
void launch_rope_norm_store_kv_fp8(
    __nv_fp8_e4m3 *out_q_ptr, __nv_fp8_e4m3 *kcache_ptr, __nv_fp8_e4m3 *vcache_ptr,
    __nv_fp8_e4m3 *out_k_ptr, __nv_fp8_e4m3 *out_v_ptr, int32_t *split_k_flag_ptr,
    float *q_scale_ptr, const __nv_bfloat16 *in_qkv_ptr, const float *cos_sin_ptr,
    const int *num_seqlen_per_req_ptr, const int *q_index_ptr, const int *kvcache_indices_ptr,
    const float *q_norm_weight_ptr, const float *k_norm_weight_ptr, float *k_scale_ptr,
    const float *v_scale_ptr, const float *q_scale_inv_ptr, float upper_max, int max_seqlen_aligned,
    Strides4D kc, Strides4D ks, Strides4D vc, Strides2D ki, int num_batch,
    cutlass::FastDivmod kv_block_size_divider, int num_rows, int qk_norm_policy, int quant_policy,
    bool is_prefill, cudaStream_t stream) {
  constexpr int kWarpsPerBlock = 4;
  constexpr int kWarpSize = 32;

  int num_compute_blocks = (num_rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
  int num_clear_blocks = (num_batch + kWarpsPerBlock - 1) / kWarpsPerBlock;
  dim3 block(kWarpsPerBlock * kWarpSize);
  dim3 grid(num_compute_blocks + num_clear_blocks, kNumKVHeads);

  auto launch = [&](auto quant_tag, auto norm_tag, auto prefill_tag) {
    constexpr int kQP = decltype(quant_tag)::value;
    constexpr int kNP = decltype(norm_tag)::value;
    constexpr bool kIP = decltype(prefill_tag)::value;
    // Matches kernel-side staging: only (kQPerKV Q heads + 1 K head) per warp
    constexpr int kQPerKV = kNumQHeads / kNumKVHeads;
    constexpr int kSmemElemPerRow = (kQPerKV + 1) * kQKHeadDim;
    constexpr int kShmQkv = kWarpsPerBlock * kSmemElemPerRow * sizeof(__nv_bfloat16);
    auto *kernel = kernels::rope_norm_store_kv_fp8_kernel<kQP, kIP, kWarpsPerBlock, kNumQHeads,
                                                          kNumKVHeads, kQKHeadDim, kVHeadDim, kNP>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kShmQkv);

    // PDL is always on for rope_v2 kernels.
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = kShmQkv;
    config.stream = stream;
    config.attrs = attribute;
    config.numAttrs = 1;

    cudaLaunchKernelEx(&config, kernel, out_q_ptr, kcache_ptr, vcache_ptr, out_k_ptr, out_v_ptr,
                       split_k_flag_ptr, q_scale_ptr, in_qkv_ptr, cos_sin_ptr,
                       num_seqlen_per_req_ptr, q_index_ptr, kvcache_indices_ptr, q_norm_weight_ptr,
                       k_norm_weight_ptr, k_scale_ptr, v_scale_ptr, q_scale_inv_ptr, upper_max,
                       max_seqlen_aligned, kc, ks, vc, ki, num_batch, kv_block_size_divider,
                       num_rows, num_compute_blocks);
  };

  auto dispatch_prefill = [&](auto quant_tag, auto norm_tag) {
    if (is_prefill) {
      launch(quant_tag, norm_tag, std::true_type{});
    } else {
      launch(quant_tag, norm_tag, std::false_type{});
    }
  };

  auto dispatch_norm = [&](auto quant_tag) {
    if (qk_norm_policy == 1) {
      dispatch_prefill(quant_tag, std::integral_constant<int, 1>{});
    } else if (qk_norm_policy == 2) {
      dispatch_prefill(quant_tag, std::integral_constant<int, 2>{});
    } else {
      dispatch_prefill(quant_tag, std::integral_constant<int, 0>{});
    }
  };

  if (quant_policy == 0) {
    dispatch_norm(std::integral_constant<int, 0>{});
  } else if (quant_policy == 1) {
    dispatch_norm(std::integral_constant<int, 1>{});
  } else if (quant_policy == 2) {
    dispatch_norm(std::integral_constant<int, 2>{});
  } else {
    dispatch_norm(std::integral_constant<int, 3>{});
  }
}

void rope_norm_store_kv_fp8_async(
    __nv_fp8_e4m3 *out_q_ptr, __nv_fp8_e4m3 *kcache_ptr, __nv_fp8_e4m3 *vcache_ptr,
    __nv_fp8_e4m3 *out_k_ptr, __nv_fp8_e4m3 *out_v_ptr, int32_t *split_k_flag_ptr,
    float *q_scale_ptr, const __nv_bfloat16 *in_qkv_ptr, const float *cos_sin_ptr,
    const int *num_seqlen_per_req_ptr, const int *q_index_ptr, const int *kvcache_indices_ptr,
    const float *q_norm_weight_ptr, const float *k_norm_weight_ptr, float *k_scale_ptr,
    const float *v_scale_ptr, const float *q_scale_inv_ptr, float upper_max, int max_seqlens,
    Strides4D kc_strides, Strides4D ks_strides, Strides4D vc_strides, Strides2D ki_strides,
    int num_batch, int kv_block_size, int num_rows, int num_q_heads, int num_kv_heads,
    int qk_head_dim, int v_head_dim, bool is_prefill, int qk_norm_policy, int quant_policy,
    cudaStream_t stream) {
  cutlass::FastDivmod kv_block_size_divider(kv_block_size);
  int max_seqlen_aligned = ((max_seqlens + 127) / 128) * 128;

  if (qk_head_dim == 128 && v_head_dim == 128) {
    constexpr int kQKHeadDim = 128;
    constexpr int kVHeadDim = 128;
    auto dispatch_heads = [&](auto q_heads, auto kv_heads) {
      constexpr int kNumQHeads = decltype(q_heads)::value;
      constexpr int kNumKVHeads = decltype(kv_heads)::value;
      launch_rope_norm_store_kv_fp8<kNumQHeads, kNumKVHeads, kQKHeadDim, kVHeadDim>(
          out_q_ptr, kcache_ptr, vcache_ptr, out_k_ptr, out_v_ptr, split_k_flag_ptr, q_scale_ptr,
          in_qkv_ptr, cos_sin_ptr, num_seqlen_per_req_ptr, q_index_ptr, kvcache_indices_ptr,
          q_norm_weight_ptr, k_norm_weight_ptr, k_scale_ptr, v_scale_ptr, q_scale_inv_ptr,
          upper_max, max_seqlen_aligned, kc_strides, ks_strides, vc_strides, ki_strides, num_batch,
          kv_block_size_divider, num_rows, qk_norm_policy, quant_policy, is_prefill, stream);
    };
    if (num_q_heads == 8 && num_kv_heads == 1) {
      dispatch_heads(std::integral_constant<int, 8>{}, std::integral_constant<int, 1>{});
    } else if (num_q_heads == 16 && num_kv_heads == 2) {
      dispatch_heads(std::integral_constant<int, 16>{}, std::integral_constant<int, 2>{});
    } else if (num_q_heads == 32 && num_kv_heads == 4) {
      dispatch_heads(std::integral_constant<int, 32>{}, std::integral_constant<int, 4>{});
    } else if (num_q_heads == 64 && num_kv_heads == 8) {
      dispatch_heads(std::integral_constant<int, 64>{}, std::integral_constant<int, 8>{});
    } else if (num_q_heads == 4 && num_kv_heads == 1) {
      // for hy v2
      dispatch_heads(std::integral_constant<int, 4>{}, std::integral_constant<int, 1>{});
    } else {
      throw std::invalid_argument(
          "rope_norm_store_kv_fp8_async: unsupported config, got: q_heads=" +
          std::to_string(num_q_heads) + ", kv_heads=" + std::to_string(num_kv_heads) +
          ", qk_head_dim=" + std::to_string(qk_head_dim) +
          ", v_head_dim=" + std::to_string(v_head_dim));
    }

  } else {
    throw std::invalid_argument("rope_norm_store_kv_fp8_async: unsupported config, got: q_heads=" +
                                std::to_string(num_q_heads) +
                                ", kv_heads=" + std::to_string(num_kv_heads) +
                                ", qk_head_dim=" + std::to_string(qk_head_dim) +
                                ", v_head_dim=" + std::to_string(v_head_dim));
  }
}

}  // namespace rope_v2
}  // namespace hpc
