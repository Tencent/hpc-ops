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

// Compute RMS norm for a vector of length head_dim using warp reduction
// and apply norm weight, additionally compute dynamic fp8 quant scale
template <int kNumItemPerThread, int kQKHeadDim, int kWarpSize = 32>
__device__ __forceinline__ void compute_rms_norm_dynamic_fp8(float *thread_data, float *max_abs,
                                                             const float *norm_weight, int ilane) {
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

  float my_max_abs = kEps;  // incase all elements are zero
#pragma unroll
  for (int iround = 0; iround < kNumItemPerThread; ++iround) {
    my_max_abs = fmaxf(my_max_abs, fabsf(thread_data[iround]));
  }
  my_max_abs = warp_reduce_max_xor(my_max_abs);
  *max_abs = my_max_abs;
}

template <int kNumItemPerThread>
__device__ void compute_abs_max_warp(float *thread_data, float *max_abs) {
  float my_max_abs = kEps;  // incase all elements are zero
#pragma unroll
  for (int iround = 0; iround < kNumItemPerThread; ++iround) {
    my_max_abs = fmaxf(my_max_abs, fabsf(thread_data[iround]));
  }
  my_max_abs = warp_reduce_max_xor(my_max_abs);
  *max_abs = my_max_abs;
}

// Prefill Kernel: Each warp processes one row (one token)
// @upper_max is used for scale to a suitable range, default is fp8_max
template <typename QType = __nv_fp8_e4m3, typename DType = __nv_bfloat16, int kNumWarpsPerBlock = 4,
          int kNumQHeads = 8, int kNumKVHeads = 1, int kQKHeadDim = 80, int kVHeadDim = 80,
          int kQKNormPolicy = 0>
__global__ void apply_rotary_pos_emb_prefill_fp8_kernel(
    QType *out_q_ptr, QType *out_k_ptr, QType *out_v_ptr, const DType *in_q_ptr,
    const DType *in_k_ptr, const DType *in_v_ptr, int q_stride, int k_stride, int v_stride,
    int out_q_stride, int out_k_stride, int out_v_stride, const float *cos_sin_ptr,
    const int *num_tokens_per_batch_ptr, const int *q_index_ptr, const float *q_norm_weight_ptr,
    const float *k_norm_weight_ptr, float *q_scale_ptr, const float *k_scale_ptr,
    const float *v_scale_ptr, float upper_max, int num_batch, int num_rows,
    int max_seqlens_pad128) {
  constexpr int kNumElemPerRow =
      kNumQHeads * kQKHeadDim + kNumKVHeads * kQKHeadDim + kNumKVHeads * kVHeadDim;

  constexpr int kWarpSize = 32;

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int iwarp = threadIdx.x / kWarpSize;
  int ilane = threadIdx.x % kWarpSize;

  extern __shared__ DType load_buffer[];  // [kNumWarpsPerBlock][kNumElemPerRow]
  __shared__ float load_buffer_cos_sin[kNumWarpsPerBlock][kQKHeadDim];
  __shared__ float load_buffer_q_norm_weight[kQKHeadDim];
  __shared__ float load_buffer_k_norm_weight[kQKHeadDim];
  __shared__ int batch_id_shm[kNumWarpsPerBlock];
  __shared__ int token_id_in_batch_shm[kNumWarpsPerBlock];

  // Load data from global memory to shared memory, use in_q_ptr as the qkv pointer for now
  const DType *in_qkv_this_block_ptr = in_q_ptr + bid * kNumWarpsPerBlock * kNumElemPerRow;
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
        store(load_buffer + load_offset_in_block, load_data);
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
                  "kNumTotalElems must be divide by sizeof(PackType)/sizeof(float) to maximum "
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

  DType *row_data = load_buffer + iwarp * kNumElemPerRow;

  // Process Q heads
  float k_scale = k_scale_ptr[0];  // for pre compute
#pragma unroll
  for (int q_head = 0; q_head < kNumQHeads; ++q_head) {
    DType *q_head_data = row_data + q_head * kQKHeadDim;
    QType *out_q_head_ptr = out_q_ptr + irow * out_q_stride + q_head * kQKHeadDim;

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
        static_assert(std::is_same<DType, __nv_bfloat16>::value,
                      "DType must be __nv_bfloat16 (input data type)");
        float x1 = q_float_buffer_reg[iround * 2];
        float x2 = q_float_buffer_reg[iround * 2 + 1];
        float cos_val = load_buffer_cos_sin[iwarp][i];
        float sin_val = load_buffer_cos_sin[iwarp][i + kQKHeadDim / 2];
        q_float_buffer_reg[iround * 2] = x1 * cos_val - x2 * sin_val;
        q_float_buffer_reg[iround * 2 + 1] = x2 * cos_val + x1 * sin_val;
      }
    }

    float q_scale_this_head = -1.0f;

    // kQKNormPolicy==1 means rope first, then norm
    if constexpr (kQKNormPolicy == 1) {
      compute_rms_norm<kNumItemPerThread, kQKHeadDim, kWarpSize>(q_float_buffer_reg,
                                                                 load_buffer_q_norm_weight, ilane);
    }

    float max_abs;
    compute_abs_max_warp<kNumItemPerThread>(q_float_buffer_reg, &max_abs);
    q_scale_this_head = max_abs / upper_max;

    int token_id_in_this_chunk = irow - q_index_ptr[batch_id];
    if (ilane == 0) {
      q_scale_ptr[batch_id * kNumQHeads * max_seqlens_pad128 + q_head * max_seqlens_pad128 +
                  token_id_in_this_chunk] = q_scale_this_head * k_scale;
    }

    q_scale_this_head = __frcp_rn(q_scale_this_head);

    // store output value
#pragma unroll
    for (int iround = 0; iround < kNumRoundsHalf; ++iround) {
      int i = iround * kWarpSize + ilane;
      if (i < kQKHeadDim / 2) {
        out_q_head_ptr[i] =
            QType(q_float_buffer_reg[iround * 2] *
                  q_scale_this_head);  // constructor QType() will do saturate (__NV_SATFINITE)
        out_q_head_ptr[i + kQKHeadDim / 2] =
            QType(q_float_buffer_reg[iround * 2 + 1] * q_scale_this_head);
      }
    }
  }

  // Process K heads
  k_scale = __frcp_rn(k_scale);
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
    QType *out_k_head_ptr = out_k_ptr + irow * out_k_stride + kv_head * kQKHeadDim;
#pragma unroll
    for (int iround = 0; iround < kNumRoundsHalf; ++iround) {
      int i = iround * kWarpSize + ilane;
      if (i < kQKHeadDim / 2) {
        QType k_out_1 = QType(k_float_buffer_reg[iround * 2] * k_scale);
        QType k_out_2 = QType(k_float_buffer_reg[iround * 2 + 1] * k_scale);

        out_k_head_ptr[i] = k_out_1;
        out_k_head_ptr[i + kQKHeadDim / 2] = k_out_2;
      }
    }
  }

  // Process V heads (no RoPE, just copy)
  {
    float v_scale = v_scale_ptr[0];
    v_scale = __frcp_rn(v_scale);
    static_assert(std::is_same_v<DType, __nv_bfloat16>,
                  "for type convert below ,here only support DType=bf16, otherwise the LoadDType "
                  "should be changed");
    using LoadDType = __nv_bfloat162;
    using PackQType = __nv_fp8x4_e4m3;
    constexpr int kNumVElemPerRow = kNumKVHeads * kVHeadDim;
    constexpr int kItemPerThread = 16 / sizeof(DType);
    static_assert(kNumVElemPerRow % kItemPerThread == 0,
                  "kNumKVHeads * kVHeadDim must be multiple of kItemPerThread\n");
    constexpr int kNumPackPerRow = kNumVElemPerRow / kItemPerThread;

    QType *out_v_row_ptr = out_v_ptr + irow * out_v_stride;
    DType *v_head_data = row_data + (kNumQHeads + kNumKVHeads) * kVHeadDim;
    constexpr int kNumLoadRound = ceil_div<kNumPackPerRow, kWarpSize>();
#pragma unroll
    for (int iround = 0; iround < kNumLoadRound; ++iround) {
      int ioffset = (iround * kWarpSize + ilane) * kItemPerThread;
      if (ioffset < kNumVElemPerRow) {
        auto vec_of_bf162_data = load<LoadDType, kItemPerThread / 2>(v_head_data + ioffset);
        auto vec_of_float_data = to<float>(vec_of_bf162_data);
#pragma unroll
        for (int i = 0; i < size(vec_of_float_data); i++) {
          vec_of_float_data[i] = vec_of_float_data[i] * v_scale;
        }
        store(out_v_row_ptr + ioffset, to<PackQType>(vec_of_float_data));
      }
    }
  }
}

// Prefill Kernel: Each warp processes one row (one token)
// @upper_max is used for scale to a suitable range, default is fp8_max
template <typename QType = __nv_fp8_e4m3, typename DType = __nv_bfloat16, int kNumWarpsPerBlock = 4,
          int kNumQHeads = 8, int kNumKVHeads = 1, int kQKHeadDim = 80, int kVHeadDim = 80,
          int kQKNormPolicy = 0>
__global__ void apply_rotary_pos_emb_blocked_prefill_fp8_kernel(
    QType *out_q_ptr, QType *kcache_ptr, QType *vcache_ptr, const DType *in_qkv_ptr,
    const float *cos_sin_ptr, const int *num_tokens_per_batch_ptr, const int *q_index_ptr,
    const int *kv_block_indices_ptr, const float *q_norm_weight_ptr, const float *k_norm_weight_ptr,
    float *q_scale_ptr, const float *k_scale_ptr, const float *v_scale_ptr, float upper_max,
    int kcache_block_offset, int vcache_block_offset, int num_batch, int max_num_kv_block_per_batch,
    cutlass::FastDivmod kv_block_size_divider, int num_rows, int max_seqlens_pad128) {
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
                  "kNumTotalElems must be divide by sizeof(PackType)/sizeof(float) to maximum "
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
  QType *k_cache_row_start = kcache_ptr + cache_block_idx * kcache_block_offset +
                             pos_in_block * (kNumKVHeads * kQKHeadDim);
  QType *v_cache_row_start =
      vcache_ptr + cache_block_idx * vcache_block_offset + pos_in_block * (kNumKVHeads * kVHeadDim);

  DType *row_data = load_buffer[iwarp];

  // Process Q heads
  float k_scale = k_scale_ptr[0];  // for pre compute
#pragma unroll
  for (int q_head = 0; q_head < kNumQHeads; ++q_head) {
    DType *q_head_data = row_data + q_head * kQKHeadDim;
    QType *out_q_head_ptr = out_q_ptr + irow * kNumQHeads * kQKHeadDim + q_head * kQKHeadDim;

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
        static_assert(std::is_same<DType, __nv_bfloat16>::value,
                      "DType must be __nv_bfloat16 (input data type)");
        float x1 = q_float_buffer_reg[iround * 2];
        float x2 = q_float_buffer_reg[iround * 2 + 1];
        float cos_val = load_buffer_cos_sin[iwarp][i];
        float sin_val = load_buffer_cos_sin[iwarp][i + kQKHeadDim / 2];
        q_float_buffer_reg[iround * 2] = x1 * cos_val - x2 * sin_val;
        q_float_buffer_reg[iround * 2 + 1] = x2 * cos_val + x1 * sin_val;
      }
    }

    float q_scale_this_head = -1.0f;

    // kQKNormPolicy==1 means rope first, then norm
    if constexpr (kQKNormPolicy == 1) {
      compute_rms_norm<kNumItemPerThread, kQKHeadDim, kWarpSize>(q_float_buffer_reg,
                                                                 load_buffer_q_norm_weight, ilane);
    }

    float max_abs;
    compute_abs_max_warp<kNumItemPerThread>(q_float_buffer_reg, &max_abs);
    q_scale_this_head = max_abs / upper_max;

    int token_id_in_this_chunk = irow - q_index_ptr[batch_id];
    if (ilane == 0) {
      q_scale_ptr[batch_id * kNumQHeads * max_seqlens_pad128 + q_head * max_seqlens_pad128 +
                  token_id_in_this_chunk] = q_scale_this_head * k_scale;
    }

    q_scale_this_head = __frcp_rn(q_scale_this_head);

    // store output value
#pragma unroll
    for (int iround = 0; iround < kNumRoundsHalf; ++iround) {
      int i = iround * kWarpSize + ilane;
      if (i < kQKHeadDim / 2) {
        out_q_head_ptr[i] =
            QType(q_float_buffer_reg[iround * 2] *
                  q_scale_this_head);  // constructor QType() will do saturate (__NV_SATFINITE)
        out_q_head_ptr[i + kQKHeadDim / 2] =
            QType(q_float_buffer_reg[iround * 2 + 1] * q_scale_this_head);
      }
    }
  }

  // Process K heads
  k_scale = __frcp_rn(k_scale);
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
        QType k_out_1 = QType(k_float_buffer_reg[iround * 2] * k_scale);
        QType k_out_2 = QType(k_float_buffer_reg[iround * 2 + 1] * k_scale);

        // Write K to KV cache
        QType *kvcache_k_ptr = k_cache_row_start + kv_head * kQKHeadDim;
        kvcache_k_ptr[i] = k_out_1;
        kvcache_k_ptr[i + kQKHeadDim / 2] = k_out_2;
      }
    }
  }

  // Process V heads (no RoPE, just copy)
  {
    float v_scale = v_scale_ptr[0];
    v_scale = __frcp_rn(v_scale);
    static_assert(std::is_same_v<DType, __nv_bfloat16>,
                  "for type convert below ,here only support DType=bf16, otherwise the LoadDType "
                  "should be changed");
    using LoadDType = __nv_bfloat162;
    using PackQType = __nv_fp8x4_e4m3;
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
        auto vec_of_bf162_data = load<LoadDType, kItemPerThread / 2>(v_head_data + ioffset);
        auto vec_of_float_data = to<float>(vec_of_bf162_data);
#pragma unroll
        for (int i = 0; i < size(vec_of_float_data); i++) {
          vec_of_float_data[i] = vec_of_float_data[i] * v_scale;
        }
        store(v_cache_row_start + ioffset, to<PackQType>(vec_of_float_data));
      }
    }
  }
}

// Decoding Kernel: Each warp processes one batch (one token per batch)
// Uses double warps: first half for computation, second half for zeroing unused cache blocks
template <typename QType = __nv_fp8_e4m3, typename DType = __nv_bfloat16, int kNumWarpsPerBlock = 4,
          int kNumQHeads = 8, int kNumKVHeads = 1, int kQKHeadDim = 80, int kVHeadDim = 80,
          int kQKNormPolicy = 0>
__global__ void apply_rotary_pos_emb_decoding_fp8_kernel(
    QType *out_q_ptr, QType *kcache_ptr, QType *vcache_ptr, const DType *in_qkv_ptr,
    const float *cos_sin_ptr, const int *num_tokens_per_batch_ptr, const int *kv_block_indices_ptr,
    const float *q_norm_weight_ptr, const float *k_norm_weight_ptr, float *q_scale_ptr,
    const float *k_scale_ptr, const float *v_scale_ptr, int *split_k_flag_ptr, float upper_max,
    int kcache_block_offset, int vcache_block_offset, int num_batch, int max_num_kv_block_per_batch,
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
    if (token_id_in_batch < 0) {
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
      QType *k_cache_row_start = kcache_ptr + cache_block_idx * kcache_block_offset +
                                 pos_in_block * (kNumKVHeads * kQKHeadDim);
      QType *v_cache_row_start = vcache_ptr + cache_block_idx * vcache_block_offset +
                                 pos_in_block * (kNumKVHeads * kVHeadDim);

      DType *row_data = load_buffer_qkv[iwarp];
      float *cos_sin_data = load_buffer_cos_sin[iwarp];

      // Process Q heads
#pragma unroll
      for (int q_head = 0; q_head < kNumQHeads; ++q_head) {
        DType *q_head_data = row_data + q_head * kQKHeadDim;
        QType *out_q_head_ptr =
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

        float q_scale_this_head = -1.0f;

        // kQKNormPolicy==1 means rope first, then norm
        if constexpr (kQKNormPolicy == 1) {
          compute_rms_norm<kNumItemPerThread, kQKHeadDim, kWarpSize>(
              q_float_buffer_reg, load_buffer_q_norm_weight, ilane);
        }

        float max_abs;
        compute_abs_max_warp<kNumItemPerThread>(q_float_buffer_reg, &max_abs);
        q_scale_this_head = max_abs / upper_max;

        if (ilane == 0) {
          q_scale_ptr[batch_id * kNumQHeads + q_head] = q_scale_this_head;
        }

        q_scale_this_head = __frcp_rn(q_scale_this_head);
        // store output value
#pragma unroll
        for (int iround = 0; iround < kNumRoundsHalf; ++iround) {
          int i = iround * kWarpSize + ilane;
          if (i < kQKHeadDim / 2) {
            out_q_head_ptr[i] =
                QType(q_float_buffer_reg[iround * 2] *
                      q_scale_this_head);  // constructor QType() will do saturate (__NV_SATFINITE)
            out_q_head_ptr[i + kQKHeadDim / 2] =
                QType(q_float_buffer_reg[iround * 2 + 1] * q_scale_this_head);
          }
        }
      }

      // Process K heads
      float k_scale = k_scale_ptr[0];
      k_scale = __frcp_rn(k_scale);
#pragma unroll
      for (int kv_head = 0; kv_head < kNumKVHeads; ++kv_head) {
        DType *k_head_data = row_data + kNumQHeads * kQKHeadDim + kv_head * kQKHeadDim;
        split_k_flag_ptr[batch_id * kNumKVHeads + kv_head] = 0;
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
            QType k_out_1 = QType(k_float_buffer_reg[iround * 2] * k_scale);
            QType k_out_2 = QType(k_float_buffer_reg[iround * 2 + 1] * k_scale);

            // No need to write K to contiguous output in decoding

            // Write K to KV cache
            QType *kvcache_k_ptr = k_cache_row_start + kv_head * kQKHeadDim;
            kvcache_k_ptr[i] = k_out_1;
            kvcache_k_ptr[i + kQKHeadDim / 2] = k_out_2;
          }
        }
      }

      // Process V heads (no RoPE, just copy)
      {
        float v_scale = v_scale_ptr[0];
        v_scale = __frcp_rn(v_scale);
        static_assert(
            std::is_same_v<DType, __nv_bfloat16>,
            "for type convert below ,here only support DType=bf16, otherwise the LoadDType "
            "should be changed");
        using LoadDType = __nv_bfloat162;
        using PackQType = __nv_fp8x4_e4m3;
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
            auto vec_of_bf162_data = load<LoadDType, kItemPerThread / 2>(v_head_data + ioffset);
            auto vec_of_float_data = to<float>(vec_of_bf162_data);
#pragma unroll
            for (int i = 0; i < size(vec_of_float_data); i++) {
              vec_of_float_data[i] = vec_of_float_data[i] * v_scale;
            }
            store(v_cache_row_start + ioffset, to<PackQType>(vec_of_float_data));
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
      if (token_id_in_batch < 0) {
        return;
      }

      // Only set kv block to zero if this is the first token in the block (pos_in_block == 0)
      if (pos_in_block == 0) {
        // Zero out rows from pos 1 to kv_block_size_divider.divisor-1
        using PackType = int4;
        constexpr int kItemPerThread = 16 / sizeof(DType);
        vec_t<QType, kItemPerThread> zero_vec;
#pragma unroll
        for (int i = 0; i < kItemPerThread; ++i) {
          zero_vec[i] = QType(0);
        }

        // Zero K cache
        {
          constexpr int kNumElemPerRow = kNumKVHeads * kQKHeadDim;
          QType *k_cache_block_start = kcache_ptr + cache_block_idx * kcache_block_offset;
          // Start from row 1 (skip row 0 as it's being written by compute warps)
          for (int row = 1; row < kv_block_size_divider.divisor; ++row) {
            QType *k_row_pack_ptr = k_cache_block_start + row * kNumKVHeads * kQKHeadDim;
            for (int idx = ilane * kItemPerThread; idx < kNumElemPerRow;
                 idx += kWarpSize * kItemPerThread) {
              store(k_row_pack_ptr + idx, zero_vec);
            }
          }
        }

        // Zero V cache
        {
          constexpr int kNumElemPerRow = kNumKVHeads * kVHeadDim;
          QType *v_cache_block_start = vcache_ptr + cache_block_idx * vcache_block_offset;
          // Start from row 1 (skip row 0 as it's being written by compute warps)
          for (int row = 1; row < kv_block_size_divider.divisor; ++row) {
            QType *v_row_pack_ptr = v_cache_block_start + row * kNumKVHeads * kVHeadDim;
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

void apply_rotary_pos_emb_blocked_kvcache_bf16_to_fp8_async(
    __nv_fp8_e4m3 *out_q_ptr, __nv_fp8_e4m3 *kcache_ptr, __nv_fp8_e4m3 *vcache_ptr,
    const __nv_bfloat16 *in_qkv_ptr, const float *cos_sin_ptr, const int *num_tokens_per_batch_ptr,
    const int *q_index_ptr, const int *kv_block_indices_ptr, const float *q_norm_weight_ptr,
    const float *k_norm_weight_ptr, float *q_scale_ptr, const float *k_scale_ptr,
    const float *v_scale_ptr, int *split_k_flag_ptr, float upper_max, int kcache_block_offset,
    int vcache_block_offset, int num_batch, int max_num_kv_block_per_batch, int kv_block_size,
    int num_rows, int num_q_heads, int num_kv_heads, int qk_head_dim, int v_head_dim,
    bool is_prefill, int qk_norm_policy, int max_seqlens_pad128, cudaStream_t stream) {
  cutlass::FastDivmod kv_block_size_divider(kv_block_size);
  using QType = __nv_fp8_e4m3;
  using DType = __nv_bfloat16;

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

      if (qk_norm_policy == 0) {
        constexpr int kQKNormPolicy = 0;
        kernels::apply_rotary_pos_emb_blocked_prefill_fp8_kernel<
            QType, DType, kNumWarpsPerBlock, kNumQHeads, kNumKVHeads, kQKHeadDim, kVHeadDim,
            kQKNormPolicy><<<grid, block, 0, stream>>>(
            out_q_ptr, kcache_ptr, vcache_ptr, in_qkv_ptr, cos_sin_ptr, num_tokens_per_batch_ptr,
            q_index_ptr, kv_block_indices_ptr, q_norm_weight_ptr, k_norm_weight_ptr, q_scale_ptr,
            k_scale_ptr, v_scale_ptr, upper_max, kcache_block_offset, vcache_block_offset,
            num_batch, max_num_kv_block_per_batch, kv_block_size_divider, num_rows,
            max_seqlens_pad128);
      } else if (qk_norm_policy == 1) {
        constexpr int kQKNormPolicy = 1;
        kernels::apply_rotary_pos_emb_blocked_prefill_fp8_kernel<
            QType, DType, kNumWarpsPerBlock, kNumQHeads, kNumKVHeads, kQKHeadDim, kVHeadDim,
            kQKNormPolicy><<<grid, block, 0, stream>>>(
            out_q_ptr, kcache_ptr, vcache_ptr, in_qkv_ptr, cos_sin_ptr, num_tokens_per_batch_ptr,
            q_index_ptr, kv_block_indices_ptr, q_norm_weight_ptr, k_norm_weight_ptr, q_scale_ptr,
            k_scale_ptr, v_scale_ptr, upper_max, kcache_block_offset, vcache_block_offset,
            num_batch, max_num_kv_block_per_batch, kv_block_size_divider, num_rows,
            max_seqlens_pad128);
      } else if (qk_norm_policy == 2) {
        constexpr int kQKNormPolicy = 2;
        kernels::apply_rotary_pos_emb_blocked_prefill_fp8_kernel<
            QType, DType, kNumWarpsPerBlock, kNumQHeads, kNumKVHeads, kQKHeadDim, kVHeadDim,
            kQKNormPolicy><<<grid, block, 0, stream>>>(
            out_q_ptr, kcache_ptr, vcache_ptr, in_qkv_ptr, cos_sin_ptr, num_tokens_per_batch_ptr,
            q_index_ptr, kv_block_indices_ptr, q_norm_weight_ptr, k_norm_weight_ptr, q_scale_ptr,
            k_scale_ptr, v_scale_ptr, upper_max, kcache_block_offset, vcache_block_offset,
            num_batch, max_num_kv_block_per_batch, kv_block_size_divider, num_rows,
            max_seqlens_pad128);
      }

    } else {
      // Decoding mode: use 8 warps per block (4 for compute, 4 for zeroing)
      constexpr int kWarpsPerBlock = 4;
      dim3 block(kWarpsPerBlock * 32);
      // addtional batch is used for clearing the kv cache
      dim3 grid(2 * ((num_batch + kWarpsPerBlock - 1) / kWarpsPerBlock));
      int num_compute_blocks = (num_batch + kWarpsPerBlock - 1) / kWarpsPerBlock;
      if (qk_norm_policy == 0) {
        constexpr int kQKNormPolicy = 0;
        kernels::apply_rotary_pos_emb_decoding_fp8_kernel<QType, DType, kWarpsPerBlock, kNumQHeads,
                                                          kNumKVHeads, kQKHeadDim, kVHeadDim,
                                                          kQKNormPolicy>
            <<<grid, block, 0, stream>>>(
                out_q_ptr, kcache_ptr, vcache_ptr, in_qkv_ptr, cos_sin_ptr,
                num_tokens_per_batch_ptr, kv_block_indices_ptr, q_norm_weight_ptr,
                k_norm_weight_ptr, q_scale_ptr, k_scale_ptr, v_scale_ptr, split_k_flag_ptr,
                upper_max, kcache_block_offset, vcache_block_offset, num_batch,
                max_num_kv_block_per_batch, kv_block_size_divider, num_compute_blocks);
      } else if (qk_norm_policy == 1) {
        constexpr int kQKNormPolicy = 1;
        kernels::apply_rotary_pos_emb_decoding_fp8_kernel<QType, DType, kWarpsPerBlock, kNumQHeads,
                                                          kNumKVHeads, kQKHeadDim, kVHeadDim,
                                                          kQKNormPolicy>
            <<<grid, block, 0, stream>>>(
                out_q_ptr, kcache_ptr, vcache_ptr, in_qkv_ptr, cos_sin_ptr,
                num_tokens_per_batch_ptr, kv_block_indices_ptr, q_norm_weight_ptr,
                k_norm_weight_ptr, q_scale_ptr, k_scale_ptr, v_scale_ptr, split_k_flag_ptr,
                upper_max, kcache_block_offset, vcache_block_offset, num_batch,
                max_num_kv_block_per_batch, kv_block_size_divider, num_compute_blocks);
      } else if (qk_norm_policy == 2) {
        constexpr int kQKNormPolicy = 2;
        kernels::apply_rotary_pos_emb_decoding_fp8_kernel<QType, DType, kWarpsPerBlock, kNumQHeads,
                                                          kNumKVHeads, kQKHeadDim, kVHeadDim,
                                                          kQKNormPolicy>
            <<<grid, block, 0, stream>>>(
                out_q_ptr, kcache_ptr, vcache_ptr, in_qkv_ptr, cos_sin_ptr,
                num_tokens_per_batch_ptr, kv_block_indices_ptr, q_norm_weight_ptr,
                k_norm_weight_ptr, q_scale_ptr, k_scale_ptr, v_scale_ptr, split_k_flag_ptr,
                upper_max, kcache_block_offset, vcache_block_offset, num_batch,
                max_num_kv_block_per_batch, kv_block_size_divider, num_compute_blocks);
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
      if (qk_norm_policy == 0) {
        constexpr int kQKNormPolicy = 0;
        kernels::apply_rotary_pos_emb_blocked_prefill_fp8_kernel<
            QType, DType, kNumWarpsPerBlock, kNumQHeads, kNumKVHeads, kQKHeadDim, kVHeadDim,
            kQKNormPolicy><<<grid, block, 0, stream>>>(
            out_q_ptr, kcache_ptr, vcache_ptr, in_qkv_ptr, cos_sin_ptr, num_tokens_per_batch_ptr,
            q_index_ptr, kv_block_indices_ptr, q_norm_weight_ptr, k_norm_weight_ptr, q_scale_ptr,
            k_scale_ptr, v_scale_ptr, upper_max, kcache_block_offset, vcache_block_offset,
            num_batch, max_num_kv_block_per_batch, kv_block_size_divider, num_rows,
            max_seqlens_pad128);
      } else if (qk_norm_policy == 1) {
        constexpr int kQKNormPolicy = 1;
        kernels::apply_rotary_pos_emb_blocked_prefill_fp8_kernel<
            QType, DType, kNumWarpsPerBlock, kNumQHeads, kNumKVHeads, kQKHeadDim, kVHeadDim,
            kQKNormPolicy><<<grid, block, 0, stream>>>(
            out_q_ptr, kcache_ptr, vcache_ptr, in_qkv_ptr, cos_sin_ptr, num_tokens_per_batch_ptr,
            q_index_ptr, kv_block_indices_ptr, q_norm_weight_ptr, k_norm_weight_ptr, q_scale_ptr,
            k_scale_ptr, v_scale_ptr, upper_max, kcache_block_offset, vcache_block_offset,
            num_batch, max_num_kv_block_per_batch, kv_block_size_divider, num_rows,
            max_seqlens_pad128);
      } else if (qk_norm_policy == 2) {
        constexpr int kQKNormPolicy = 2;
        kernels::apply_rotary_pos_emb_blocked_prefill_fp8_kernel<
            QType, DType, kNumWarpsPerBlock, kNumQHeads, kNumKVHeads, kQKHeadDim, kVHeadDim,
            kQKNormPolicy><<<grid, block, 0, stream>>>(
            out_q_ptr, kcache_ptr, vcache_ptr, in_qkv_ptr, cos_sin_ptr, num_tokens_per_batch_ptr,
            q_index_ptr, kv_block_indices_ptr, q_norm_weight_ptr, k_norm_weight_ptr, q_scale_ptr,
            k_scale_ptr, v_scale_ptr, upper_max, kcache_block_offset, vcache_block_offset,
            num_batch, max_num_kv_block_per_batch, kv_block_size_divider, num_rows,
            max_seqlens_pad128);
      }

    } else {
      constexpr int kWarpsPerBlock = 4;
      dim3 block(kWarpsPerBlock * kWarpSize);
      // addtional batch is used for clearing the kv cache
      dim3 grid(2 * ((num_batch + kWarpsPerBlock - 1) / kWarpsPerBlock));
      int num_compute_blocks = (num_batch + kWarpsPerBlock - 1) / kWarpsPerBlock;

      if (qk_norm_policy == 0) {
        constexpr int kQKNormPolicy = 0;
        kernels::apply_rotary_pos_emb_decoding_fp8_kernel<QType, DType, kWarpsPerBlock, kNumQHeads,
                                                          kNumKVHeads, kQKHeadDim, kVHeadDim,
                                                          kQKNormPolicy>
            <<<grid, block, 0, stream>>>(
                out_q_ptr, kcache_ptr, vcache_ptr, in_qkv_ptr, cos_sin_ptr,
                num_tokens_per_batch_ptr, kv_block_indices_ptr, q_norm_weight_ptr,
                k_norm_weight_ptr, q_scale_ptr, k_scale_ptr, v_scale_ptr, split_k_flag_ptr,
                upper_max, kcache_block_offset, vcache_block_offset, num_batch,
                max_num_kv_block_per_batch, kv_block_size_divider, num_compute_blocks);
      } else if (qk_norm_policy == 1) {
        constexpr int kQKNormPolicy = 1;
        kernels::apply_rotary_pos_emb_decoding_fp8_kernel<QType, DType, kWarpsPerBlock, kNumQHeads,
                                                          kNumKVHeads, kQKHeadDim, kVHeadDim,
                                                          kQKNormPolicy>
            <<<grid, block, 0, stream>>>(
                out_q_ptr, kcache_ptr, vcache_ptr, in_qkv_ptr, cos_sin_ptr,
                num_tokens_per_batch_ptr, kv_block_indices_ptr, q_norm_weight_ptr,
                k_norm_weight_ptr, q_scale_ptr, k_scale_ptr, v_scale_ptr, split_k_flag_ptr,
                upper_max, kcache_block_offset, vcache_block_offset, num_batch,
                max_num_kv_block_per_batch, kv_block_size_divider, num_compute_blocks);
      } else if (qk_norm_policy == 2) {
        constexpr int kQKNormPolicy = 2;
        kernels::apply_rotary_pos_emb_decoding_fp8_kernel<QType, DType, kWarpsPerBlock, kNumQHeads,
                                                          kNumKVHeads, kQKHeadDim, kVHeadDim,
                                                          kQKNormPolicy>
            <<<grid, block, 0, stream>>>(
                out_q_ptr, kcache_ptr, vcache_ptr, in_qkv_ptr, cos_sin_ptr,
                num_tokens_per_batch_ptr, kv_block_indices_ptr, q_norm_weight_ptr,
                k_norm_weight_ptr, q_scale_ptr, k_scale_ptr, v_scale_ptr, split_k_flag_ptr,
                upper_max, kcache_block_offset, vcache_block_offset, num_batch,
                max_num_kv_block_per_batch, kv_block_size_divider, num_compute_blocks);
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
      if (qk_norm_policy == 0) {
        constexpr int kQKNormPolicy = 0;
        kernels::apply_rotary_pos_emb_blocked_prefill_fp8_kernel<
            QType, DType, kNumWarpsPerBlock, kNumQHeads, kNumKVHeads, kQKHeadDim, kVHeadDim,
            kQKNormPolicy><<<grid, block, 0, stream>>>(
            out_q_ptr, kcache_ptr, vcache_ptr, in_qkv_ptr, cos_sin_ptr, num_tokens_per_batch_ptr,
            q_index_ptr, kv_block_indices_ptr, q_norm_weight_ptr, k_norm_weight_ptr, q_scale_ptr,
            k_scale_ptr, v_scale_ptr, upper_max, kcache_block_offset, vcache_block_offset,
            num_batch, max_num_kv_block_per_batch, kv_block_size_divider, num_rows,
            max_seqlens_pad128);
      } else if (qk_norm_policy == 1) {
        constexpr int kQKNormPolicy = 1;
        kernels::apply_rotary_pos_emb_blocked_prefill_fp8_kernel<
            QType, DType, kNumWarpsPerBlock, kNumQHeads, kNumKVHeads, kQKHeadDim, kVHeadDim,
            kQKNormPolicy><<<grid, block, 0, stream>>>(
            out_q_ptr, kcache_ptr, vcache_ptr, in_qkv_ptr, cos_sin_ptr, num_tokens_per_batch_ptr,
            q_index_ptr, kv_block_indices_ptr, q_norm_weight_ptr, k_norm_weight_ptr, q_scale_ptr,
            k_scale_ptr, v_scale_ptr, upper_max, kcache_block_offset, vcache_block_offset,
            num_batch, max_num_kv_block_per_batch, kv_block_size_divider, num_rows,
            max_seqlens_pad128);
      } else if (qk_norm_policy == 2) {
        constexpr int kQKNormPolicy = 2;
        kernels::apply_rotary_pos_emb_blocked_prefill_fp8_kernel<
            QType, DType, kNumWarpsPerBlock, kNumQHeads, kNumKVHeads, kQKHeadDim, kVHeadDim,
            kQKNormPolicy><<<grid, block, 0, stream>>>(
            out_q_ptr, kcache_ptr, vcache_ptr, in_qkv_ptr, cos_sin_ptr, num_tokens_per_batch_ptr,
            q_index_ptr, kv_block_indices_ptr, q_norm_weight_ptr, k_norm_weight_ptr, q_scale_ptr,
            k_scale_ptr, v_scale_ptr, upper_max, kcache_block_offset, vcache_block_offset,
            num_batch, max_num_kv_block_per_batch, kv_block_size_divider, num_rows,
            max_seqlens_pad128);
      }

    } else {
      // Decoding mode: use 8 warps per block (4 for compute, 4 for zeroing)
      constexpr int kWarpsPerBlock = 4;
      dim3 block(kWarpsPerBlock * kWarpSize);
      // addtional batch is used for clearing the kv cache
      dim3 grid(2 * ((num_batch + kWarpsPerBlock - 1) / kWarpsPerBlock));
      int num_compute_blocks = (num_batch + kWarpsPerBlock - 1) / kWarpsPerBlock;

      if (qk_norm_policy == 0) {
        constexpr int kQKNormPolicy = 0;
        kernels::apply_rotary_pos_emb_decoding_fp8_kernel<QType, DType, kWarpsPerBlock, kNumQHeads,
                                                          kNumKVHeads, kQKHeadDim, kVHeadDim,
                                                          kQKNormPolicy>
            <<<grid, block, 0, stream>>>(
                out_q_ptr, kcache_ptr, vcache_ptr, in_qkv_ptr, cos_sin_ptr,
                num_tokens_per_batch_ptr, kv_block_indices_ptr, q_norm_weight_ptr,
                k_norm_weight_ptr, q_scale_ptr, k_scale_ptr, v_scale_ptr, split_k_flag_ptr,
                upper_max, kcache_block_offset, vcache_block_offset, num_batch,
                max_num_kv_block_per_batch, kv_block_size_divider, num_compute_blocks);
      } else if (qk_norm_policy == 1) {
        constexpr int kQKNormPolicy = 1;
        kernels::apply_rotary_pos_emb_decoding_fp8_kernel<QType, DType, kWarpsPerBlock, kNumQHeads,
                                                          kNumKVHeads, kQKHeadDim, kVHeadDim,
                                                          kQKNormPolicy>
            <<<grid, block, 0, stream>>>(
                out_q_ptr, kcache_ptr, vcache_ptr, in_qkv_ptr, cos_sin_ptr,
                num_tokens_per_batch_ptr, kv_block_indices_ptr, q_norm_weight_ptr,
                k_norm_weight_ptr, q_scale_ptr, k_scale_ptr, v_scale_ptr, split_k_flag_ptr,
                upper_max, kcache_block_offset, vcache_block_offset, num_batch,
                max_num_kv_block_per_batch, kv_block_size_divider, num_compute_blocks);
      } else if (qk_norm_policy == 2) {
        constexpr int kQKNormPolicy = 2;
        kernels::apply_rotary_pos_emb_decoding_fp8_kernel<QType, DType, kWarpsPerBlock, kNumQHeads,
                                                          kNumKVHeads, kQKHeadDim, kVHeadDim,
                                                          kQKNormPolicy>
            <<<grid, block, 0, stream>>>(
                out_q_ptr, kcache_ptr, vcache_ptr, in_qkv_ptr, cos_sin_ptr,
                num_tokens_per_batch_ptr, kv_block_indices_ptr, q_norm_weight_ptr,
                k_norm_weight_ptr, q_scale_ptr, k_scale_ptr, v_scale_ptr, split_k_flag_ptr,
                upper_max, kcache_block_offset, vcache_block_offset, num_batch,
                max_num_kv_block_per_batch, kv_block_size_divider, num_compute_blocks);
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

void apply_rotary_pos_emb_bf16_to_fp8_async(
    __nv_fp8_e4m3 *out_q_ptr, __nv_fp8_e4m3 *out_k_ptr, __nv_fp8_e4m3 *out_v_ptr,
    const __nv_bfloat16 *in_q_ptr, const __nv_bfloat16 *in_k_ptr, const __nv_bfloat16 *in_v_ptr,
    const int q_stride, const int k_stride, const int v_stride, const int out_q_stride,
    const int out_k_stride, const int out_v_stride, const float *cos_sin_ptr,
    const int *num_tokens_per_batch_ptr, const int *q_index_ptr, const float *q_norm_weight_ptr,
    const float *k_norm_weight_ptr, float *q_scale_ptr, const float *k_scale_ptr,
    const float *v_scale_ptr, int *split_k_flag_ptr, float upper_max, int num_batch, int num_rows,
    int num_q_heads, int num_kv_heads, int qk_head_dim, int v_head_dim, bool is_prefill,
    int qk_norm_policy, int max_seqlens_pad128, cudaStream_t stream) {
  using QType = __nv_fp8_e4m3;
  using DType = __nv_bfloat16;

  // Dispatch based on head configuration and mode
  if (num_q_heads == 8 && num_kv_heads == 1 && qk_head_dim == 128 && v_head_dim == 128) {
    constexpr int kNumQHeads = 8;
    constexpr int kNumKVHeads = 1;
    constexpr int kQKHeadDim = 128;
    constexpr int kVHeadDim = 128;
    constexpr int kWarpSize = 32;
    constexpr int kNumElemPerRow =
        kNumQHeads * kQKHeadDim + kNumKVHeads * kQKHeadDim + kNumKVHeads * kVHeadDim;
    if (is_prefill) {
      constexpr int kNumWarpsPerBlock = 4;
      dim3 block(kNumWarpsPerBlock * kWarpSize);
      dim3 grid((num_rows + kNumWarpsPerBlock - 1) / kNumWarpsPerBlock);
      if (qk_norm_policy == 0) {
        constexpr int kQKNormPolicy = 0;
        auto kernel =
            kernels::apply_rotary_pos_emb_prefill_fp8_kernel<QType, DType, kNumWarpsPerBlock,
                                                             kNumQHeads, kNumKVHeads, kQKHeadDim,
                                                             kVHeadDim, kQKNormPolicy>;
        int shm_size = kNumWarpsPerBlock * kNumElemPerRow * sizeof(DType);
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
        kernel<<<grid, block, shm_size, stream>>>(
            out_q_ptr, out_k_ptr, out_v_ptr, in_q_ptr, in_k_ptr, in_v_ptr, q_stride, k_stride,
            v_stride, out_q_stride, out_k_stride, out_v_stride, cos_sin_ptr,
            num_tokens_per_batch_ptr, q_index_ptr, q_norm_weight_ptr, k_norm_weight_ptr,
            q_scale_ptr, k_scale_ptr, v_scale_ptr, upper_max, num_batch, num_rows,
            max_seqlens_pad128);
      } else if (qk_norm_policy == 1) {
        constexpr int kQKNormPolicy = 1;

        auto kernel =
            kernels::apply_rotary_pos_emb_prefill_fp8_kernel<QType, DType, kNumWarpsPerBlock,
                                                             kNumQHeads, kNumKVHeads, kQKHeadDim,
                                                             kVHeadDim, kQKNormPolicy>;
        int shm_size = kNumWarpsPerBlock * kNumElemPerRow * sizeof(DType);
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
        kernel<<<grid, block, shm_size, stream>>>(
            out_q_ptr, out_k_ptr, out_v_ptr, in_q_ptr, in_k_ptr, in_v_ptr, q_stride, k_stride,
            v_stride, out_q_stride, out_k_stride, out_v_stride, cos_sin_ptr,
            num_tokens_per_batch_ptr, q_index_ptr, q_norm_weight_ptr, k_norm_weight_ptr,
            q_scale_ptr, k_scale_ptr, v_scale_ptr, upper_max, num_batch, num_rows,
            max_seqlens_pad128);
      } else if (qk_norm_policy == 2) {
        constexpr int kQKNormPolicy = 2;
        auto kernel =
            kernels::apply_rotary_pos_emb_prefill_fp8_kernel<QType, DType, kNumWarpsPerBlock,
                                                             kNumQHeads, kNumKVHeads, kQKHeadDim,
                                                             kVHeadDim, kQKNormPolicy>;
        int shm_size = kNumWarpsPerBlock * kNumElemPerRow * sizeof(DType);
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
        kernel<<<grid, block, shm_size, stream>>>(
            out_q_ptr, out_k_ptr, out_v_ptr, in_q_ptr, in_k_ptr, in_v_ptr, q_stride, k_stride,
            v_stride, out_q_stride, out_k_stride, out_v_stride, cos_sin_ptr,
            num_tokens_per_batch_ptr, q_index_ptr, q_norm_weight_ptr, k_norm_weight_ptr,
            q_scale_ptr, k_scale_ptr, v_scale_ptr, upper_max, num_batch, num_rows,
            max_seqlens_pad128);
      }

    } else {
      // throw an error if the configuration is not supported
      std::string msg;
      msg = "Unsupported decode for rope_norm_w8c8 currently";
      throw std::invalid_argument(msg);
    }

  } else if (num_q_heads == 64 && num_kv_heads == 8 && qk_head_dim == 128 && v_head_dim == 128) {
    constexpr int kNumQHeads = 64;
    constexpr int kNumKVHeads = 8;
    constexpr int kQKHeadDim = 128;
    constexpr int kVHeadDim = 128;
    constexpr int kWarpSize = 32;
    constexpr int kNumElemPerRow =
        kNumQHeads * kQKHeadDim + kNumKVHeads * kQKHeadDim + kNumKVHeads * kVHeadDim;
    if (is_prefill) {
      constexpr int kNumWarpsPerBlock = 4;
      dim3 block(kNumWarpsPerBlock * kWarpSize);
      dim3 grid((num_rows + kNumWarpsPerBlock - 1) / kNumWarpsPerBlock);
      if (qk_norm_policy == 0) {
        constexpr int kQKNormPolicy = 0;
        auto kernel =
            kernels::apply_rotary_pos_emb_prefill_fp8_kernel<QType, DType, kNumWarpsPerBlock,
                                                             kNumQHeads, kNumKVHeads, kQKHeadDim,
                                                             kVHeadDim, kQKNormPolicy>;
        int shm_size = kNumWarpsPerBlock * kNumElemPerRow * sizeof(DType);
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
        kernel<<<grid, block, shm_size, stream>>>(
            out_q_ptr, out_k_ptr, out_v_ptr, in_q_ptr, in_k_ptr, in_v_ptr, q_stride, k_stride,
            v_stride, out_q_stride, out_k_stride, out_v_stride, cos_sin_ptr,
            num_tokens_per_batch_ptr, q_index_ptr, q_norm_weight_ptr, k_norm_weight_ptr,
            q_scale_ptr, k_scale_ptr, v_scale_ptr, upper_max, num_batch, num_rows,
            max_seqlens_pad128);
      } else if (qk_norm_policy == 1) {
        constexpr int kQKNormPolicy = 1;
        auto kernel =
            kernels::apply_rotary_pos_emb_prefill_fp8_kernel<QType, DType, kNumWarpsPerBlock,
                                                             kNumQHeads, kNumKVHeads, kQKHeadDim,
                                                             kVHeadDim, kQKNormPolicy>;
        int shm_size = kNumWarpsPerBlock * kNumElemPerRow * sizeof(DType);
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
        kernel<<<grid, block, shm_size, stream>>>(
            out_q_ptr, out_k_ptr, out_v_ptr, in_q_ptr, in_k_ptr, in_v_ptr, q_stride, k_stride,
            v_stride, out_q_stride, out_k_stride, out_v_stride, cos_sin_ptr,
            num_tokens_per_batch_ptr, q_index_ptr, q_norm_weight_ptr, k_norm_weight_ptr,
            q_scale_ptr, k_scale_ptr, v_scale_ptr, upper_max, num_batch, num_rows,
            max_seqlens_pad128);
      } else if (qk_norm_policy == 2) {
        constexpr int kQKNormPolicy = 2;
        auto kernel =
            kernels::apply_rotary_pos_emb_prefill_fp8_kernel<QType, DType, kNumWarpsPerBlock,
                                                             kNumQHeads, kNumKVHeads, kQKHeadDim,
                                                             kVHeadDim, kQKNormPolicy>;
        int shm_size = kNumWarpsPerBlock * kNumElemPerRow * sizeof(DType);
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
        kernel<<<grid, block, shm_size, stream>>>(
            out_q_ptr, out_k_ptr, out_v_ptr, in_q_ptr, in_k_ptr, in_v_ptr, q_stride, k_stride,
            v_stride, out_q_stride, out_k_stride, out_v_stride, cos_sin_ptr,
            num_tokens_per_batch_ptr, q_index_ptr, q_norm_weight_ptr, k_norm_weight_ptr,
            q_scale_ptr, k_scale_ptr, v_scale_ptr, upper_max, num_batch, num_rows,
            max_seqlens_pad128);
      }
    } else {
      // throw an error if the configuration is not supported
      std::string msg;
      msg = "Unsupported decode for rope_norm_w8c8 currently";
      throw std::invalid_argument(msg);
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
