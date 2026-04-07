// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <cuda_bf16.h>
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

// Compute RMS norm for a vector of length head_dim using warp reduction
// and apply norm weight
template <int kNumItemPerThread, int kQKHeadDim, int kWarpSize = 32>
__device__ __forceinline__ void compute_rms_norm(float *thread_data, const float *norm_weight,
                                                 int ilane) {
  // Compute sum of squares
  float sum_sq = 0.0f;
#pragma unroll
  for (int iround = 0; iround < kNumItemPerThread; ++iround) {
    sum_sq += thread_data[iround] * thread_data[iround];
  }

  // Warp reduction to get total sum
  sum_sq = warp_reduce_sum_xor(sum_sq);

  // Compute RMS normalization factor
  float inv_rms = rsqrtf(sum_sq / kQKHeadDim + kEps);  // faster instruction for 1/sqrt(x)

  // Apply normalization and weight
  constexpr int kNumRoundsHalf = (kQKHeadDim / 2 + kWarpSize - 1) / kWarpSize;
#pragma unroll
  for (int iround = 0; iround < kNumRoundsHalf; ++iround) {
    int i = iround * kWarpSize + ilane;
    if (i < kQKHeadDim / 2) {
      float weight1 = norm_weight[i];
      float weight2 = norm_weight[i + kQKHeadDim / 2];
      thread_data[iround * 2] *= inv_rms * weight1;
      thread_data[iround * 2 + 1] *= inv_rms * weight2;
    }
  }
}

// Prefill Kernel: Each warp processes one row (one token)
template <typename DType = __nv_bfloat16, int kNumWarpsPerBlock = 4, int kNumQHeads = 8,
          int kNumKVHeads = 1, int kQKHeadDim = 80, int kVHeadDim = 80, int kQKNormPolicy = 0>
__global__ void apply_rotary_pos_emb_prefill_kernel(
    DType *out_q_ptr, DType *kcache_ptr, DType *vcache_ptr, const DType *in_qkv_ptr,
    const float *cos_sin_ptr, const int *num_tokens_per_batch_ptr, const int *q_index_ptr,
    const int *kv_block_indices_ptr, const float *q_norm_weight_ptr, const float *k_norm_weight_ptr,
    int kcache_block_offset, int vcache_block_offset, int num_batch, int max_num_kv_block_per_batch,
    cutlass::FastDivmod kv_block_size_divider, int num_rows) {
  constexpr int kNumElemPerRow =
      kNumQHeads * kQKHeadDim + kNumKVHeads * kQKHeadDim + kNumKVHeads * kVHeadDim;

  constexpr int kWarpSize = 32;

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int iwarp = threadIdx.x / kWarpSize;
  int ilane = threadIdx.x % kWarpSize;

  __shared__ DType load_buffer[kNumWarpsPerBlock][kNumElemPerRow];
  __shared__ float load_buffer_cos_sin[kNumWarpsPerBlock][kQKHeadDim];
  __shared__ float load_buffer_q_norm_weight[kQKHeadDim];
  __shared__ float load_buffer_k_norm_weight[kQKHeadDim];
  __shared__ int batch_id_shm[kNumWarpsPerBlock];
  __shared__ int token_id_in_batch_shm[kNumWarpsPerBlock];

  // Load data from global memory to shared memory
  const DType *in_qkv_this_block_ptr = in_qkv_ptr + bid * kNumWarpsPerBlock * kNumElemPerRow;
  {
    constexpr int kItemPerThread = 16 / sizeof(DType);
    constexpr int kNumLoadRound = ceil_div<kNumElemPerRow * kNumWarpsPerBlock,
                                           kItemPerThread * kWarpSize * kNumWarpsPerBlock>();
    int valid_rows_this_block = num_rows - bid * kNumWarpsPerBlock;
    valid_rows_this_block =
        valid_rows_this_block >= kNumWarpsPerBlock ? kNumWarpsPerBlock : valid_rows_this_block;
    int valid_elem_this_block = valid_rows_this_block * kNumElemPerRow;

#pragma unroll
    for (int iround = 0; iround < kNumLoadRound; ++iround) {
      int load_offset_in_block = (iround * kWarpSize * kNumWarpsPerBlock + tid) * kItemPerThread;
      if (load_offset_in_block < valid_elem_this_block) {
        auto load_data = load<DType, kItemPerThread>(in_qkv_this_block_ptr + load_offset_in_block);
        store(&load_buffer[0][0] + load_offset_in_block, load_data);
      }
    }
  }

  int irow = bid * kNumWarpsPerBlock + iwarp;
  if (irow >= num_rows) return;

  // Compute batch_id and token_id_in_batch for each warp
  if (tid < kNumWarpsPerBlock) {
    int global_row = bid * kNumWarpsPerBlock + tid;
    if (global_row < num_rows) {
      int batch_id = 0;
      for (int i = 0; i < num_batch; ++i) {
        if (global_row < q_index_ptr[i + 1]) {
          batch_id = i;
          break;
        }
      }
      batch_id_shm[tid] = batch_id;
      // Compute cumsum up to batch_id
      token_id_in_batch_shm[tid] =
          global_row + num_tokens_per_batch_ptr[batch_id] - q_index_ptr[batch_id + 1];
    }
  }

  __syncthreads();

  int token_id_in_batch = token_id_in_batch_shm[iwarp];

  // Load norm weights into shared memory (only once per block)
  if constexpr (kQKNormPolicy > 0) {
    constexpr int kItemPerThread = 16 / sizeof(float);
    constexpr int kNumPacks = kQKHeadDim / kItemPerThread;
    static_assert(kQKHeadDim % kItemPerThread == 0,
                  "kQKHeadDim must be divisible by kItemPerThread");
    static_assert(kItemPerThread * kWarpSize >= kQKHeadDim, "otherwise here should loop");
    if (tid < kNumPacks) {
      int ioffset = tid * kItemPerThread;
      store(load_buffer_q_norm_weight + ioffset,
            load<float, kItemPerThread>(q_norm_weight_ptr + ioffset));
      store(load_buffer_k_norm_weight + ioffset,
            load<float, kItemPerThread>(k_norm_weight_ptr + ioffset));
    }
  }

  // Load cos sin data into shared memory

  {
    constexpr int kItemPerThread = 16 / sizeof(float);
    constexpr int kNumTotalElems = kQKHeadDim;
    constexpr int kNumPacks = kNumTotalElems / kItemPerThread;
    static_assert(kNumTotalElems % kItemPerThread == 0,
                  "kNumTotalElems must be divide by sizeof(PackType)/sizeof(DType) to maximum "
                  "float load of cos_sin_ptr");
    static_assert(kNumPacks <= kWarpSize,
                  "kNumPacks must be less than total threads, otherwise here must be looped");
    const float *cos_sin_this_row_ptr = cos_sin_ptr + token_id_in_batch * kQKHeadDim;
    if (ilane < kNumPacks) {
      int ioffset = ilane * kItemPerThread;
      store(&load_buffer_cos_sin[iwarp][0] + ioffset,
            load<float, kItemPerThread>(cos_sin_this_row_ptr + ioffset));
    }
  }

  __syncthreads();

  // Each warp processes one row (one token)

  int batch_id = batch_id_shm[iwarp];

  int block_idx_in_batch;
  int pos_in_block;
  // Compute KV cache block addressing
  kv_block_size_divider(block_idx_in_batch, pos_in_block, token_id_in_batch);  // (q,r,s) -> s=q*d+r
  int cache_block_idx =
      kv_block_indices_ptr[batch_id * max_num_kv_block_per_batch + block_idx_in_batch];

  // KV cache layout: [cache_block_idx][pos_in_block][kv_head][head_dim]
  // [warning] : here we assume the cache block is allocated when the token is the first of this
  // new cache block
  DType *k_cache_row_start = kcache_ptr + (int64_t)cache_block_idx * (int64_t)kcache_block_offset +
                             pos_in_block * (kNumKVHeads * kQKHeadDim);
  DType *v_cache_row_start = vcache_ptr + (int64_t)cache_block_idx * (int64_t)vcache_block_offset +
                             pos_in_block * (kNumKVHeads * kVHeadDim);

  DType *row_data = load_buffer[iwarp];

  // Process Q heads
#pragma unroll
  for (int q_head = 0; q_head < kNumQHeads; ++q_head) {
    DType *q_head_data = row_data + q_head * kQKHeadDim;
    DType *out_q_head_ptr = out_q_ptr + irow * kNumQHeads * kQKHeadDim + q_head * kQKHeadDim;

    // Apply RoPE transformation (neox version)
    constexpr int kNumRoundsHalf = (kQKHeadDim / 2 + kWarpSize - 1) / kWarpSize;
    constexpr int kNumItemPerThread = kNumRoundsHalf * 2;
    float q_float_buffer_reg[kNumItemPerThread] = {0};

#pragma unroll
    for (int iround = 0; iround < kNumRoundsHalf; ++iround) {
      int i = iround * kWarpSize + ilane;
      if (i < kQKHeadDim / 2) {
        static_assert(std::is_same<DType, __nv_bfloat16>::value, "DType must be __nv_bfloat16");
        q_float_buffer_reg[iround * 2] = __bfloat162float(q_head_data[i]);
        q_float_buffer_reg[iround * 2 + 1] = __bfloat162float(q_head_data[i + kQKHeadDim / 2]);
      }
    }

    // kQKNormPolicy==2 means norm first, then rope
    if constexpr (kQKNormPolicy == 2) {
      compute_rms_norm<kNumItemPerThread, kQKHeadDim, kWarpSize>(q_float_buffer_reg,
                                                                 load_buffer_q_norm_weight, ilane);
    }

#pragma unroll
    for (int iround = 0; iround < kNumRoundsHalf; ++iround) {
      int i = iround * kWarpSize + ilane;
      if (i < kQKHeadDim / 2) {
        static_assert(std::is_same<DType, __nv_bfloat16>::value, "DType must be __nv_bfloat16");
        float x1 = q_float_buffer_reg[iround * 2];
        float x2 = q_float_buffer_reg[iround * 2 + 1];
        float cos_val = load_buffer_cos_sin[iwarp][i];
        float sin_val = load_buffer_cos_sin[iwarp][i + kQKHeadDim / 2];
        q_float_buffer_reg[iround * 2] = x1 * cos_val - x2 * sin_val;
        q_float_buffer_reg[iround * 2 + 1] = x2 * cos_val + x1 * sin_val;
      }
    }

    // kQKNormPolicy==1 means rope first, then norm
    if constexpr (kQKNormPolicy == 1) {
      compute_rms_norm<kNumItemPerThread, kQKHeadDim, kWarpSize>(q_float_buffer_reg,
                                                                 load_buffer_q_norm_weight, ilane);
    }

    // store output value
#pragma unroll
    for (int iround = 0; iround < kNumRoundsHalf; ++iround) {
      int i = iround * kWarpSize + ilane;
      if (i < kQKHeadDim / 2) {
        out_q_head_ptr[i] = __float2bfloat16(q_float_buffer_reg[iround * 2]);
        out_q_head_ptr[i + kQKHeadDim / 2] = __float2bfloat16(q_float_buffer_reg[iround * 2 + 1]);
      }
    }
  }

  // Process K heads
#pragma unroll
  for (int kv_head = 0; kv_head < kNumKVHeads; ++kv_head) {
    DType *k_head_data = row_data + kNumQHeads * kQKHeadDim + kv_head * kQKHeadDim;

    // Apply RoPE transformation (neox version)
    constexpr int kNumRoundsHalf = (kQKHeadDim / 2 + kWarpSize - 1) / kWarpSize;
    constexpr int kNumItemPerThread = kNumRoundsHalf * 2;
    float k_float_buffer_reg[kNumItemPerThread] = {0};

#pragma unroll
    for (int iround = 0; iround < kNumRoundsHalf; ++iround) {
      int i = iround * kWarpSize + ilane;
      if (i < kQKHeadDim / 2) {
        static_assert(std::is_same<DType, __nv_bfloat16>::value, "DType must be __nv_bfloat16");
        k_float_buffer_reg[iround * 2] = __bfloat162float(k_head_data[i]);
        k_float_buffer_reg[iround * 2 + 1] = __bfloat162float(k_head_data[i + kQKHeadDim / 2]);
      }
    }

    // kQKNormPolicy==2 means norm first, then rope
    if constexpr (kQKNormPolicy == 2) {
      compute_rms_norm<kNumItemPerThread, kQKHeadDim, kWarpSize>(k_float_buffer_reg,
                                                                 load_buffer_k_norm_weight, ilane);
    }

#pragma unroll
    for (int iround = 0; iround < kNumRoundsHalf; ++iround) {
      int i = iround * kWarpSize + ilane;
      if (i < kQKHeadDim / 2) {
        float x1 = k_float_buffer_reg[iround * 2];
        float x2 = k_float_buffer_reg[iround * 2 + 1];
        float cos_val = load_buffer_cos_sin[iwarp][i];
        float sin_val = load_buffer_cos_sin[iwarp][i + kQKHeadDim / 2];
        k_float_buffer_reg[iround * 2] = x1 * cos_val - x2 * sin_val;
        k_float_buffer_reg[iround * 2 + 1] = x2 * cos_val + x1 * sin_val;
      }
    }

    // kQKNormPolicy==1 means rope first, then norm
    if constexpr (kQKNormPolicy == 1) {
      compute_rms_norm<kNumItemPerThread, kQKHeadDim, kWarpSize>(k_float_buffer_reg,
                                                                 load_buffer_k_norm_weight, ilane);
    }

    // store output value and write to KV cache
#pragma unroll
    for (int iround = 0; iround < kNumRoundsHalf; ++iround) {
      int i = iround * kWarpSize + ilane;
      if (i < kQKHeadDim / 2) {
        DType k_out_1 = __float2bfloat16(k_float_buffer_reg[iround * 2]);
        DType k_out_2 = __float2bfloat16(k_float_buffer_reg[iround * 2 + 1]);

        // Write K to KV cache
        DType *kvcache_k_ptr = k_cache_row_start + kv_head * kQKHeadDim;
        kvcache_k_ptr[i] = k_out_1;
        kvcache_k_ptr[i + kQKHeadDim / 2] = k_out_2;
      }
    }
  }

  // Process V heads (no RoPE, just copy)
  {
    constexpr int kNumVElemPerRow = kNumKVHeads * kVHeadDim;
    constexpr int kItemPerThread = 16 / sizeof(DType);
    static_assert(kNumVElemPerRow % kItemPerThread == 0,
                  "kNumKVHeads * kVHeadDim must be multiple of kItemPerThread\n");
    constexpr int kNumPackPerRow = kNumVElemPerRow / kItemPerThread;

    DType *v_head_data = row_data + (kNumQHeads + kNumKVHeads) * kVHeadDim;
    constexpr int kNumLoadRound = ceil_div<kNumPackPerRow, kWarpSize>();
#pragma unroll
    for (int iround = 0; iround < kNumLoadRound; ++iround) {
      int ioffset = (iround * kWarpSize + ilane) * kItemPerThread;
      if (ioffset < kNumVElemPerRow) {
        store(v_cache_row_start + ioffset, load<DType, kItemPerThread>(v_head_data + ioffset));
      }
    }
  }
}

// Decoding Kernel: Each warp processes one batch (one token per batch)
// Uses double warps: first half for computation, second half for zeroing unused cache blocks
template <typename DType = __nv_bfloat16, int kNumWarpsPerBlock = 4, int kNumQHeads = 8,
          int kNumKVHeads = 1, int kQKHeadDim = 80, int kVHeadDim = 80, int kQKNormPolicy = 0>
__global__ void apply_rotary_pos_emb_decoding_kernel(
    DType *out_q_ptr, DType *kcache_ptr, DType *vcache_ptr, const DType *in_qkv_ptr,
    const float *cos_sin_ptr, const int *num_tokens_per_batch_ptr, const int *kv_block_indices_ptr,
    const float *q_norm_weight_ptr, const float *k_norm_weight_ptr, int kcache_block_offset,
    int vcache_block_offset, int num_batch, int max_num_kv_block_per_batch,
    cutlass::FastDivmod kv_block_size_divider, int num_compute_block) {
  constexpr int kNumElemPerRow =
      kNumQHeads * kQKHeadDim + kNumKVHeads * kQKHeadDim + kNumKVHeads * kVHeadDim;

  constexpr int kWarpSize = 32;

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int iwarp = threadIdx.x / kWarpSize;
  int ilane = threadIdx.x % kWarpSize;

  // Shared memory for data prefetching
  __shared__ DType load_buffer_qkv[kNumWarpsPerBlock][kNumElemPerRow];
  __shared__ float load_buffer_cos_sin[kNumWarpsPerBlock][kQKHeadDim];
  __shared__ float load_buffer_q_norm_weight[kQKHeadDim];
  __shared__ float load_buffer_k_norm_weight[kQKHeadDim];

  if (bid < num_compute_block) {
    int batch_id = bid * kNumWarpsPerBlock + iwarp;

    // num_tokens_per_batch_ptr is the number of tokens, but we need the index of current token, so
    // here should -1
    int token_id_in_batch = batch_id < num_batch ? num_tokens_per_batch_ptr[batch_id] - 1 : -1;

    // Load norm weights into shared memory (only once per block)
    if constexpr (kQKNormPolicy > 0) {
      constexpr int kItemPerThread = 16 / sizeof(float);
      constexpr int kNumPacks = kQKHeadDim / kItemPerThread;
      static_assert(kQKHeadDim % kItemPerThread == 0,
                    "kQKHeadDim must be divisible by kItemPerThread");
      static_assert(kItemPerThread * kWarpSize >= kQKHeadDim, "otherwise here should loop");
      if (tid < kNumPacks) {
        int ioffset = tid * kItemPerThread;
        store(load_buffer_q_norm_weight + ioffset,
              load<float, kItemPerThread>(q_norm_weight_ptr + ioffset));
        store(load_buffer_k_norm_weight + ioffset,
              load<float, kItemPerThread>(k_norm_weight_ptr + ioffset));
      }
    }

    // if token_id_in_batch == 0, this row is a padding row, skip
    if (token_id_in_batch <= 0) {
      return;
    }

    if (batch_id < num_batch) {
      // Load QKV data to shared memory using vectorized loads
      {
        constexpr int kElemPerThread = 16 / sizeof(DType);
        static_assert(kNumElemPerRow % kElemPerThread == 0,
                      "kNumElemPerRow must be divisible by kElemPerThread");
        constexpr int kNumPacksPerRow = kNumElemPerRow / kElemPerThread;

        const DType *in_qkv_this_row_ptr = in_qkv_ptr + batch_id * kNumElemPerRow;
        constexpr int kNumLoadRound = ceil_div<kNumPacksPerRow, kWarpSize>();
#pragma unroll
        for (int i = 0; i < kNumLoadRound; ++i) {
          int ioffset = (i * kWarpSize + ilane) * kElemPerThread;
          if (ioffset < kNumElemPerRow) {
            store(load_buffer_qkv[iwarp] + ioffset,
                  load<DType, kElemPerThread>(in_qkv_this_row_ptr + ioffset));
          }
        }
      }

      // Load cos_sin data to shared memory using vectorized loads
      {
        constexpr int kItemPerThread = 16 / sizeof(float);
        constexpr int kNumTotalElems = kQKHeadDim;
        constexpr int kNumPacks = kNumTotalElems / kItemPerThread;
        static_assert(kNumTotalElems % kItemPerThread == 0,
                      "kNumTotalElems must be divide by sizeof(PackType)/sizeof(DType) to maximum "
                      "float load of cos_sin_ptr");
        static_assert(kNumPacks <= kWarpSize,
                      "kNumPacks must be less than total threads, otherwise here must be looped");
        const float *cos_sin_this_row_ptr = cos_sin_ptr + token_id_in_batch * kQKHeadDim;
        if (ilane < kNumPacks) {
          int ioffset = ilane * kItemPerThread;
          store(&load_buffer_cos_sin[iwarp][0] + ioffset,
                load<float, kItemPerThread>(cos_sin_this_row_ptr + ioffset));
        }
      }
    }

    __syncthreads();

    if (batch_id < num_batch) {
      int block_idx_in_batch;
      int pos_in_block;
      // Compute KV cache block addressing
      kv_block_size_divider(block_idx_in_batch, pos_in_block,
                            token_id_in_batch);  // (q,r,s) -> s=q*d+r
      int cache_block_idx =
          kv_block_indices_ptr[batch_id * max_num_kv_block_per_batch + block_idx_in_batch];

      // KV cache layout: [cache_block_idx][pos_in_block][kv_head][head_dim]
      DType *k_cache_row_start = kcache_ptr +
                                 (int64_t)cache_block_idx * (int64_t)kcache_block_offset +
                                 pos_in_block * (kNumKVHeads * kQKHeadDim);
      DType *v_cache_row_start = vcache_ptr +
                                 (int64_t)cache_block_idx * (int64_t)vcache_block_offset +
                                 pos_in_block * (kNumKVHeads * kVHeadDim);

      DType *row_data = load_buffer_qkv[iwarp];
      float *cos_sin_data = load_buffer_cos_sin[iwarp];

      // Process Q heads
#pragma unroll
      for (int q_head = 0; q_head < kNumQHeads; ++q_head) {
        DType *q_head_data = row_data + q_head * kQKHeadDim;
        DType *out_q_head_ptr =
            out_q_ptr + batch_id * kNumQHeads * kQKHeadDim + q_head * kQKHeadDim;

        // Apply RoPE transformation (neox version)
        constexpr int kNumRoundsHalf = (kQKHeadDim / 2 + kWarpSize - 1) / kWarpSize;
        constexpr int kNumItemPerThread = kNumRoundsHalf * 2;
        float q_float_buffer_reg[kNumItemPerThread] = {0};

#pragma unroll
        for (int iround = 0; iround < kNumRoundsHalf; ++iround) {
          int i = iround * kWarpSize + ilane;
          if (i < kQKHeadDim / 2) {
            static_assert(std::is_same<DType, __nv_bfloat16>::value, "DType must be __nv_bfloat16");
            q_float_buffer_reg[iround * 2] = __bfloat162float(q_head_data[i]);
            q_float_buffer_reg[iround * 2 + 1] = __bfloat162float(q_head_data[i + kQKHeadDim / 2]);
          }
        }

        // kQKNormPolicy==2 means norm first, then rope
        if constexpr (kQKNormPolicy == 2) {
          compute_rms_norm<kNumItemPerThread, kQKHeadDim, kWarpSize>(
              q_float_buffer_reg, load_buffer_q_norm_weight, ilane);
        }

#pragma unroll
        for (int iround = 0; iround < kNumRoundsHalf; ++iround) {
          int i = iround * kWarpSize + ilane;
          if (i < kQKHeadDim / 2) {
            float x1 = q_float_buffer_reg[iround * 2];
            float x2 = q_float_buffer_reg[iround * 2 + 1];
            float cos_val = cos_sin_data[i];
            float sin_val = cos_sin_data[i + kQKHeadDim / 2];
            q_float_buffer_reg[iround * 2] = x1 * cos_val - x2 * sin_val;
            q_float_buffer_reg[iround * 2 + 1] = x2 * cos_val + x1 * sin_val;
          }
        }

        // kQKNormPolicy==1 means rope first, then norm
        if constexpr (kQKNormPolicy == 1) {
          compute_rms_norm<kNumItemPerThread, kQKHeadDim, kWarpSize>(
              q_float_buffer_reg, load_buffer_q_norm_weight, ilane);
        }

        // store output value
#pragma unroll
        for (int iround = 0; iround < kNumRoundsHalf; ++iround) {
          int i = iround * kWarpSize + ilane;
          if (i < kQKHeadDim / 2) {
            out_q_head_ptr[i] = __float2bfloat16(q_float_buffer_reg[iround * 2]);
            out_q_head_ptr[i + kQKHeadDim / 2] =
                __float2bfloat16(q_float_buffer_reg[iround * 2 + 1]);
          }
        }
      }

      // Process K heads
#pragma unroll
      for (int kv_head = 0; kv_head < kNumKVHeads; ++kv_head) {
        DType *k_head_data = row_data + kNumQHeads * kQKHeadDim + kv_head * kQKHeadDim;

        // Apply RoPE transformation (neox version)
        constexpr int kNumRoundsHalf = (kQKHeadDim / 2 + kWarpSize - 1) / kWarpSize;
        constexpr int kNumItemPerThread = kNumRoundsHalf * 2;
        float k_float_buffer_reg[kNumItemPerThread] = {0};

#pragma unroll
        for (int iround = 0; iround < kNumRoundsHalf; ++iround) {
          int i = iround * kWarpSize + ilane;
          if (i < kQKHeadDim / 2) {
            static_assert(std::is_same<DType, __nv_bfloat16>::value, "DType must be __nv_bfloat16");
            k_float_buffer_reg[iround * 2] = __bfloat162float(k_head_data[i]);
            k_float_buffer_reg[iround * 2 + 1] = __bfloat162float(k_head_data[i + kQKHeadDim / 2]);
          }
        }

        // kQKNormPolicy==2 means norm first, then rope
        if constexpr (kQKNormPolicy == 2) {
          compute_rms_norm<kNumItemPerThread, kQKHeadDim, kWarpSize>(
              k_float_buffer_reg, load_buffer_k_norm_weight, ilane);
        }

#pragma unroll
        for (int iround = 0; iround < kNumRoundsHalf; ++iround) {
          int i = iround * kWarpSize + ilane;
          if (i < kQKHeadDim / 2) {
            float x1 = k_float_buffer_reg[iround * 2];
            float x2 = k_float_buffer_reg[iround * 2 + 1];
            float cos_val = cos_sin_data[i];
            float sin_val = cos_sin_data[i + kQKHeadDim / 2];
            k_float_buffer_reg[iround * 2] = x1 * cos_val - x2 * sin_val;
            k_float_buffer_reg[iround * 2 + 1] = x2 * cos_val + x1 * sin_val;
          }
        }

        // kQKNormPolicy==1 means rope first, then norm
        if constexpr (kQKNormPolicy == 1) {
          compute_rms_norm<kNumItemPerThread, kQKHeadDim, kWarpSize>(
              k_float_buffer_reg, load_buffer_k_norm_weight, ilane);
        }

        // store output value and write to KV cache
#pragma unroll
        for (int iround = 0; iround < kNumRoundsHalf; ++iround) {
          int i = iround * kWarpSize + ilane;
          if (i < kQKHeadDim / 2) {
            DType k_out_1 = __float2bfloat16(k_float_buffer_reg[iround * 2]);
            DType k_out_2 = __float2bfloat16(k_float_buffer_reg[iround * 2 + 1]);

            // No need to write K to contiguous output in decoding

            // Write K to KV cache
            DType *kvcache_k_ptr = k_cache_row_start + kv_head * kQKHeadDim;
            kvcache_k_ptr[i] = k_out_1;
            kvcache_k_ptr[i + kQKHeadDim / 2] = k_out_2;
          }
        }
      }

      // Process V heads (no RoPE, just copy)
      {
        constexpr int kNumVElemPerRow = kNumKVHeads * kVHeadDim;
        constexpr int kItemPerThread = 16 / sizeof(DType);
        static_assert(kNumVElemPerRow % kItemPerThread == 0,
                      "kNumKVHeads * kVHeadDim must be multiple of kItemPerThread\n");
        constexpr int kNumPackPerRow = kNumVElemPerRow / kItemPerThread;

        DType *v_head_data = row_data + (kNumQHeads + kNumKVHeads) * kVHeadDim;
        constexpr int kNumLoadRound = ceil_div<kNumPackPerRow, kWarpSize>();
#pragma unroll
        for (int iround = 0; iround < kNumLoadRound; ++iround) {
          int ioffset = (iround * kWarpSize + ilane) * kItemPerThread;
          if (ioffset < kNumVElemPerRow) {
            store(v_cache_row_start + ioffset, load<DType, kItemPerThread>(v_head_data + ioffset));
          }
        }
      }
    }
  } else {
    int batch_id = (bid - num_compute_block) * kNumWarpsPerBlock + iwarp;
    if (batch_id < num_batch) {
      int token_id_in_batch = num_tokens_per_batch_ptr[batch_id] - 1;
      // Second half warps: zero out remaining rows in cache block if this is the first token
      int block_idx_in_batch;
      int pos_in_block;
      kv_block_size_divider(block_idx_in_batch, pos_in_block,
                            token_id_in_batch);  // (q,r,s) -> s=q*d+r
      int cache_block_idx =
          kv_block_indices_ptr[batch_id * max_num_kv_block_per_batch + block_idx_in_batch];

      // if token_id_in_batch == 0, this row is a padding row, skip
      if (token_id_in_batch <= 0) {
        return;
      }

      // Only set kv block to zero if this is the first token in the block (pos_in_block == 0)
      if (pos_in_block == 0) {
        // Zero out rows from pos 1 to kv_block_size_divider.divisor-1
        using PackType = int4;
        constexpr int kItemPerThread = 16 / sizeof(DType);
        vec_t<DType, kItemPerThread> zero_vec;
#pragma unroll
        for (int i = 0; i < kItemPerThread; ++i) {
          zero_vec[i] = {0};
        }

        // Zero K cache
        {
          constexpr int kNumElemPerRow = kNumKVHeads * kQKHeadDim;
          DType *k_cache_block_start =
              kcache_ptr + (int64_t)cache_block_idx * (int64_t)kcache_block_offset;
          // Start from row 1 (skip row 0 as it's being written by compute warps)
          for (int row = 1; row < kv_block_size_divider.divisor; ++row) {
            DType *k_row_pack_ptr = k_cache_block_start + row * kNumKVHeads * kQKHeadDim;
            for (int idx = ilane * kItemPerThread; idx < kNumElemPerRow;
                 idx += kWarpSize * kItemPerThread) {
              store(k_row_pack_ptr + idx, zero_vec);
            }
          }
        }

        // Zero V cache
        {
          constexpr int kNumElemPerRow = kNumKVHeads * kVHeadDim;
          DType *v_cache_block_start =
              vcache_ptr + (int64_t)cache_block_idx * (int64_t)vcache_block_offset;
          // Start from row 1 (skip row 0 as it's being written by compute warps)
          for (int row = 1; row < kv_block_size_divider.divisor; ++row) {
            DType *v_row_pack_ptr = v_cache_block_start + row * kNumKVHeads * kVHeadDim;
            for (int idx = ilane * kItemPerThread; idx < kNumElemPerRow;
                 idx += kWarpSize * kItemPerThread) {
              store(v_row_pack_ptr + idx, zero_vec);
            }
          }
        }
      }
    }
  }
}

}  // namespace kernels

void apply_rotary_pos_emb_blocked_kvcache_bf16_async(
    __nv_bfloat16 *out_q_ptr, __nv_bfloat16 *kcache_ptr, __nv_bfloat16 *vcache_ptr,
    const __nv_bfloat16 *in_qkv_ptr, const float *cos_sin_ptr, const int *num_tokens_per_batch_ptr,
    const int *q_index_ptr, const int *kv_block_indices_ptr, const float *q_norm_weight_ptr,
    const float *k_norm_weight_ptr, int kcache_block_offset, int vcache_block_offset, int num_batch,
    int max_num_kv_block_per_batch, int kv_block_size, int num_rows, int num_q_heads,
    int num_kv_heads, int qk_head_dim, int v_head_dim, bool is_prefill, int qk_norm_policy,
    cudaStream_t stream) {
  cutlass::FastDivmod kv_block_size_divider(kv_block_size);

  // Dispatch based on head configuration and mode
  if (num_q_heads == 8 && num_kv_heads == 1 && qk_head_dim == 80 && v_head_dim == 80) {
    constexpr int kNumQHeads = 8;
    constexpr int kNumKVHeads = 1;
    constexpr int kQKHeadDim = 80;
    constexpr int kVHeadDim = 80;
    constexpr int kWarpSize = 32;
    if (is_prefill) {
      constexpr int kNumWarpsPerBlock = 4;
      dim3 block(kNumWarpsPerBlock * kWarpSize);
      dim3 grid((num_rows + kNumWarpsPerBlock - 1) / kNumWarpsPerBlock);

      if (qk_norm_policy == 1) {  // rope first
        constexpr int kQKNormPolicy = 1;
        kernels::apply_rotary_pos_emb_prefill_kernel<__nv_bfloat16, kNumWarpsPerBlock, kNumQHeads,
                                                     kNumKVHeads, kQKHeadDim, kVHeadDim,
                                                     kQKNormPolicy><<<grid, block, 0, stream>>>(
            out_q_ptr, kcache_ptr, vcache_ptr, in_qkv_ptr, cos_sin_ptr, num_tokens_per_batch_ptr,
            q_index_ptr, kv_block_indices_ptr, q_norm_weight_ptr, k_norm_weight_ptr,
            kcache_block_offset, vcache_block_offset, num_batch, max_num_kv_block_per_batch,
            kv_block_size_divider, num_rows);
      } else if (qk_norm_policy == 2) {  // norm first
        constexpr int kQKNormPolicy = 2;
        kernels::apply_rotary_pos_emb_prefill_kernel<__nv_bfloat16, kNumWarpsPerBlock, kNumQHeads,
                                                     kNumKVHeads, kQKHeadDim, kVHeadDim,
                                                     kQKNormPolicy><<<grid, block, 0, stream>>>(
            out_q_ptr, kcache_ptr, vcache_ptr, in_qkv_ptr, cos_sin_ptr, num_tokens_per_batch_ptr,
            q_index_ptr, kv_block_indices_ptr, q_norm_weight_ptr, k_norm_weight_ptr,
            kcache_block_offset, vcache_block_offset, num_batch, max_num_kv_block_per_batch,
            kv_block_size_divider, num_rows);
      } else {  // no normalization
        constexpr int kQKNormPolicy = 0;
        kernels::apply_rotary_pos_emb_prefill_kernel<__nv_bfloat16, kNumWarpsPerBlock, kNumQHeads,
                                                     kNumKVHeads, kQKHeadDim, kVHeadDim,
                                                     kQKNormPolicy><<<grid, block, 0, stream>>>(
            out_q_ptr, kcache_ptr, vcache_ptr, in_qkv_ptr, cos_sin_ptr, num_tokens_per_batch_ptr,
            q_index_ptr, kv_block_indices_ptr, q_norm_weight_ptr, k_norm_weight_ptr,
            kcache_block_offset, vcache_block_offset, num_batch, max_num_kv_block_per_batch,
            kv_block_size_divider, num_rows);
      }

    } else {
      // Decoding mode: use 8 warps per block (4 for compute, 4 for zeroing)
      constexpr int kWarpsPerBlock = 4;
      dim3 block(kWarpsPerBlock * 32);
      // addtional batch is used for clearing the kv cache
      dim3 grid(2 * ((num_batch + kWarpsPerBlock - 1) / kWarpsPerBlock));
      int num_compute_blocks = (num_batch + kWarpsPerBlock - 1) / kWarpsPerBlock;
      if (qk_norm_policy == 1) {  // rope first
        constexpr int kQKNormPolicy = 1;
        kernels::apply_rotary_pos_emb_decoding_kernel<__nv_bfloat16, kWarpsPerBlock, kNumQHeads,
                                                      kNumKVHeads, kQKHeadDim, kVHeadDim,
                                                      kQKNormPolicy><<<grid, block, 0, stream>>>(
            out_q_ptr, kcache_ptr, vcache_ptr, in_qkv_ptr, cos_sin_ptr, num_tokens_per_batch_ptr,
            kv_block_indices_ptr, q_norm_weight_ptr, k_norm_weight_ptr, kcache_block_offset,
            vcache_block_offset, num_batch, max_num_kv_block_per_batch, kv_block_size_divider,
            num_compute_blocks);
      } else if (qk_norm_policy == 2) {  // norm first
        constexpr int kQKNormPolicy = 2;
        kernels::apply_rotary_pos_emb_decoding_kernel<__nv_bfloat16, kWarpsPerBlock, kNumQHeads,
                                                      kNumKVHeads, kQKHeadDim, kVHeadDim,
                                                      kQKNormPolicy><<<grid, block, 0, stream>>>(
            out_q_ptr, kcache_ptr, vcache_ptr, in_qkv_ptr, cos_sin_ptr, num_tokens_per_batch_ptr,
            kv_block_indices_ptr, q_norm_weight_ptr, k_norm_weight_ptr, kcache_block_offset,
            vcache_block_offset, num_batch, max_num_kv_block_per_batch, kv_block_size_divider,
            num_compute_blocks);
      } else {  // no normalization
        constexpr int kQKNormPolicy = 0;
        kernels::apply_rotary_pos_emb_decoding_kernel<__nv_bfloat16, kWarpsPerBlock, kNumQHeads,
                                                      kNumKVHeads, kQKHeadDim, kVHeadDim,
                                                      kQKNormPolicy><<<grid, block, 0, stream>>>(
            out_q_ptr, kcache_ptr, vcache_ptr, in_qkv_ptr, cos_sin_ptr, num_tokens_per_batch_ptr,
            kv_block_indices_ptr, q_norm_weight_ptr, k_norm_weight_ptr, kcache_block_offset,
            vcache_block_offset, num_batch, max_num_kv_block_per_batch, kv_block_size_divider,
            num_compute_blocks);
      }
    }

  } else if (num_q_heads == 8 && num_kv_heads == 1 && qk_head_dim == 128 && v_head_dim == 128) {
    constexpr int kNumQHeads = 8;
    constexpr int kNumKVHeads = 1;
    constexpr int kQKHeadDim = 128;
    constexpr int kVHeadDim = 128;
    constexpr int kWarpSize = 32;
    if (is_prefill) {
      constexpr int kNumWarpsPerBlock = 4;
      dim3 block(kNumWarpsPerBlock * kWarpSize);
      dim3 grid((num_rows + kNumWarpsPerBlock - 1) / kNumWarpsPerBlock);

      if (qk_norm_policy == 1) {  // rope first
        constexpr int kQKNormPolicy = 1;
        kernels::apply_rotary_pos_emb_prefill_kernel<__nv_bfloat16, kNumWarpsPerBlock, kNumQHeads,
                                                     kNumKVHeads, kQKHeadDim, kVHeadDim,
                                                     kQKNormPolicy><<<grid, block, 0, stream>>>(
            out_q_ptr, kcache_ptr, vcache_ptr, in_qkv_ptr, cos_sin_ptr, num_tokens_per_batch_ptr,
            q_index_ptr, kv_block_indices_ptr, q_norm_weight_ptr, k_norm_weight_ptr,
            kcache_block_offset, vcache_block_offset, num_batch, max_num_kv_block_per_batch,
            kv_block_size_divider, num_rows);
      } else if (qk_norm_policy == 2) {  // norm first
        constexpr int kQKNormPolicy = 2;
        kernels::apply_rotary_pos_emb_prefill_kernel<__nv_bfloat16, kNumWarpsPerBlock, kNumQHeads,
                                                     kNumKVHeads, kQKHeadDim, kVHeadDim,
                                                     kQKNormPolicy><<<grid, block, 0, stream>>>(
            out_q_ptr, kcache_ptr, vcache_ptr, in_qkv_ptr, cos_sin_ptr, num_tokens_per_batch_ptr,
            q_index_ptr, kv_block_indices_ptr, q_norm_weight_ptr, k_norm_weight_ptr,
            kcache_block_offset, vcache_block_offset, num_batch, max_num_kv_block_per_batch,
            kv_block_size_divider, num_rows);
      } else {  // no normalization
        constexpr int kQKNormPolicy = 0;
        kernels::apply_rotary_pos_emb_prefill_kernel<__nv_bfloat16, kNumWarpsPerBlock, kNumQHeads,
                                                     kNumKVHeads, kQKHeadDim, kVHeadDim,
                                                     kQKNormPolicy><<<grid, block, 0, stream>>>(
            out_q_ptr, kcache_ptr, vcache_ptr, in_qkv_ptr, cos_sin_ptr, num_tokens_per_batch_ptr,
            q_index_ptr, kv_block_indices_ptr, q_norm_weight_ptr, k_norm_weight_ptr,
            kcache_block_offset, vcache_block_offset, num_batch, max_num_kv_block_per_batch,
            kv_block_size_divider, num_rows);
      }

    } else {
      constexpr int kWarpsPerBlock = 4;
      dim3 block(kWarpsPerBlock * kWarpSize);
      // addtional batch is used for clearing the kv cache
      dim3 grid(2 * ((num_batch + kWarpsPerBlock - 1) / kWarpsPerBlock));
      int num_compute_blocks = (num_batch + kWarpsPerBlock - 1) / kWarpsPerBlock;

      if (qk_norm_policy == 1) {  // rope first
        constexpr int kQKNormPolicy = 1;
        kernels::apply_rotary_pos_emb_decoding_kernel<__nv_bfloat16, kWarpsPerBlock, kNumQHeads,
                                                      kNumKVHeads, kQKHeadDim, kVHeadDim,
                                                      kQKNormPolicy><<<grid, block, 0, stream>>>(
            out_q_ptr, kcache_ptr, vcache_ptr, in_qkv_ptr, cos_sin_ptr, num_tokens_per_batch_ptr,
            kv_block_indices_ptr, q_norm_weight_ptr, k_norm_weight_ptr, kcache_block_offset,
            vcache_block_offset, num_batch, max_num_kv_block_per_batch, kv_block_size_divider,
            num_compute_blocks);
      } else if (qk_norm_policy == 2) {  // rope first
        constexpr int kQKNormPolicy = 2;
        kernels::apply_rotary_pos_emb_decoding_kernel<__nv_bfloat16, kWarpsPerBlock, kNumQHeads,
                                                      kNumKVHeads, kQKHeadDim, kVHeadDim,
                                                      kQKNormPolicy><<<grid, block, 0, stream>>>(
            out_q_ptr, kcache_ptr, vcache_ptr, in_qkv_ptr, cos_sin_ptr, num_tokens_per_batch_ptr,
            kv_block_indices_ptr, q_norm_weight_ptr, k_norm_weight_ptr, kcache_block_offset,
            vcache_block_offset, num_batch, max_num_kv_block_per_batch, kv_block_size_divider,
            num_compute_blocks);
      } else {  // no normalization
        constexpr int kQKNormPolicy = 0;
        kernels::apply_rotary_pos_emb_decoding_kernel<__nv_bfloat16, kWarpsPerBlock, kNumQHeads,
                                                      kNumKVHeads, kQKHeadDim, kVHeadDim,
                                                      kQKNormPolicy><<<grid, block, 0, stream>>>(
            out_q_ptr, kcache_ptr, vcache_ptr, in_qkv_ptr, cos_sin_ptr, num_tokens_per_batch_ptr,
            kv_block_indices_ptr, q_norm_weight_ptr, k_norm_weight_ptr, kcache_block_offset,
            vcache_block_offset, num_batch, max_num_kv_block_per_batch, kv_block_size_divider,
            num_compute_blocks);
      }
    }

  } else if (num_q_heads == 4 && num_kv_heads == 1 && qk_head_dim == 128 && v_head_dim == 128) {
    constexpr int kNumQHeads = 4;
    constexpr int kNumKVHeads = 1;
    constexpr int kQKHeadDim = 128;
    constexpr int kVHeadDim = 128;
    constexpr int kWarpSize = 32;
    if (is_prefill) {
      constexpr int kNumWarpsPerBlock = 4;
      dim3 block(kNumWarpsPerBlock * kWarpSize);
      dim3 grid((num_rows + kNumWarpsPerBlock - 1) / kNumWarpsPerBlock);

      if (qk_norm_policy == 1) {  // rope first
        constexpr int kQKNormPolicy = 1;
        kernels::apply_rotary_pos_emb_prefill_kernel<__nv_bfloat16, kNumWarpsPerBlock, kNumQHeads,
                                                     kNumKVHeads, kQKHeadDim, kVHeadDim,
                                                     kQKNormPolicy><<<grid, block, 0, stream>>>(
            out_q_ptr, kcache_ptr, vcache_ptr, in_qkv_ptr, cos_sin_ptr, num_tokens_per_batch_ptr,
            q_index_ptr, kv_block_indices_ptr, q_norm_weight_ptr, k_norm_weight_ptr,
            kcache_block_offset, vcache_block_offset, num_batch, max_num_kv_block_per_batch,
            kv_block_size_divider, num_rows);
      } else if (qk_norm_policy == 2) {  // norm first
        constexpr int kQKNormPolicy = 2;
        kernels::apply_rotary_pos_emb_prefill_kernel<__nv_bfloat16, kNumWarpsPerBlock, kNumQHeads,
                                                     kNumKVHeads, kQKHeadDim, kVHeadDim,
                                                     kQKNormPolicy><<<grid, block, 0, stream>>>(
            out_q_ptr, kcache_ptr, vcache_ptr, in_qkv_ptr, cos_sin_ptr, num_tokens_per_batch_ptr,
            q_index_ptr, kv_block_indices_ptr, q_norm_weight_ptr, k_norm_weight_ptr,
            kcache_block_offset, vcache_block_offset, num_batch, max_num_kv_block_per_batch,
            kv_block_size_divider, num_rows);
      } else {  // no normalization
        constexpr int kQKNormPolicy = 0;
        kernels::apply_rotary_pos_emb_prefill_kernel<__nv_bfloat16, kNumWarpsPerBlock, kNumQHeads,
                                                     kNumKVHeads, kQKHeadDim, kVHeadDim,
                                                     kQKNormPolicy><<<grid, block, 0, stream>>>(
            out_q_ptr, kcache_ptr, vcache_ptr, in_qkv_ptr, cos_sin_ptr, num_tokens_per_batch_ptr,
            q_index_ptr, kv_block_indices_ptr, q_norm_weight_ptr, k_norm_weight_ptr,
            kcache_block_offset, vcache_block_offset, num_batch, max_num_kv_block_per_batch,
            kv_block_size_divider, num_rows);
      }

    } else {
      // Decoding mode: use 8 warps per block (4 for compute, 4 for zeroing)
      constexpr int kWarpsPerBlock = 4;
      dim3 block(kWarpsPerBlock * kWarpSize);
      // addtional batch is used for clearing the kv cache
      dim3 grid(2 * ((num_batch + kWarpsPerBlock - 1) / kWarpsPerBlock));
      int num_compute_blocks = (num_batch + kWarpsPerBlock - 1) / kWarpsPerBlock;

      if (qk_norm_policy == 1) {  // rope first
        constexpr int kQKNormPolicy = 1;
        kernels::apply_rotary_pos_emb_decoding_kernel<__nv_bfloat16, kWarpsPerBlock, kNumQHeads,
                                                      kNumKVHeads, kQKHeadDim, kVHeadDim,
                                                      kQKNormPolicy><<<grid, block, 0, stream>>>(
            out_q_ptr, kcache_ptr, vcache_ptr, in_qkv_ptr, cos_sin_ptr, num_tokens_per_batch_ptr,
            kv_block_indices_ptr, q_norm_weight_ptr, k_norm_weight_ptr, kcache_block_offset,
            vcache_block_offset, num_batch, max_num_kv_block_per_batch, kv_block_size_divider,
            num_compute_blocks);
      } else if (qk_norm_policy == 2) {  // rope first
        constexpr int kQKNormPolicy = 2;
        kernels::apply_rotary_pos_emb_decoding_kernel<__nv_bfloat16, kWarpsPerBlock, kNumQHeads,
                                                      kNumKVHeads, kQKHeadDim, kVHeadDim,
                                                      kQKNormPolicy><<<grid, block, 0, stream>>>(
            out_q_ptr, kcache_ptr, vcache_ptr, in_qkv_ptr, cos_sin_ptr, num_tokens_per_batch_ptr,
            kv_block_indices_ptr, q_norm_weight_ptr, k_norm_weight_ptr, kcache_block_offset,
            vcache_block_offset, num_batch, max_num_kv_block_per_batch, kv_block_size_divider,
            num_compute_blocks);
      } else {  // no normalization
        constexpr int kQKNormPolicy = 0;
        kernels::apply_rotary_pos_emb_decoding_kernel<__nv_bfloat16, kWarpsPerBlock, kNumQHeads,
                                                      kNumKVHeads, kQKHeadDim, kVHeadDim,
                                                      kQKNormPolicy><<<grid, block, 0, stream>>>(
            out_q_ptr, kcache_ptr, vcache_ptr, in_qkv_ptr, cos_sin_ptr, num_tokens_per_batch_ptr,
            kv_block_indices_ptr, q_norm_weight_ptr, k_norm_weight_ptr, kcache_block_offset,
            vcache_block_offset, num_batch, max_num_kv_block_per_batch, kv_block_size_divider,
            num_compute_blocks);
      }
    }
  } else {
    // throw an error if the configuration is not supported
    std::string msg;
    msg = "Unsupported configuration, got: q_heads=" + std::to_string(num_q_heads) +
          ", kv_heads=" + std::to_string(num_kv_heads) +
          ", qk_head_dim=" + std::to_string(qk_head_dim) +
          ", v_head_dim=" + std::to_string(v_head_dim);
    throw std::invalid_argument(msg);
  }
}

}  // namespace rope
}  // namespace hpc
