// Copyright 2025 hpc-ops authors

#ifndef SRC_ATTENTION_DECODE_SPLITK_COMBINE_KERNELS_DIMTILED_CUH_
#define SRC_ATTENTION_DECODE_SPLITK_COMBINE_KERNELS_DIMTILED_CUH_

#include <cuda.h>
#include <stdio.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "cutlass/fast_math.h"
#include "src/attention/decode/sched_task_info.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace attention {
namespace decode {
namespace kernels {

template <typename T, int kWarpCount, int kMaxSplitK, int kDimTiles = 1>
__global__ void attention_decode_dynamic_splitk_combine_dimtiled_kernel(
    T *y_ptr, const float *split_input_ptr, const float *lse_ptr, const int *task_map_ptr,
    int num_sm_count, int num_batch, int num_seq_q, int num_head_q, int num_head_k,
    int pad_heads_per_group, int num_dim_v, cutlass::FastDivmod heads_per_group_divider) {
  static_assert(kDimTiles == 1 || kDimTiles == 2 || kDimTiles == 4, "kDimTiles must divide 4");
  constexpr int kItemsPerThread = 4 / kDimTiles;
  constexpr int kColsPerTile = 32 * kItemsPerThread;
  constexpr int kPrefetch = 32 / kItemsPerThread;

  int iseq = blockIdx.y;
  int ibatch = blockIdx.z;
  int ilinear = blockIdx.x * kWarpCount + threadIdx.x / 32;
  int idimtile = ilinear % kDimTiles;
  int ihead_q = ilinear / kDimTiles;
  int ihead_kv, ihead;
  heads_per_group_divider(ihead_kv, ihead, ihead_q);
  int iwarp = threadIdx.x / 32;
  int ilane = threadIdx.x % 32;
  int icol = idimtile * kColsPerTile + ilane * kItemsPerThread;

  int num_tiles_per_sm = task_map_ptr[0];
  constexpr int kTaskStride = dynamic::kTaskScheduleInfoStride;
  int num_chunks = task_map_ptr[(num_tiles_per_sm * num_sm_count + 1) * kTaskStride +
                                ihead_kv * num_batch + ibatch];

  const int lse_kv_head_stride = num_seq_q * pad_heads_per_group;
  const int lse_chunk_stride = num_head_k * lse_kv_head_stride;
  const int lse_batch_stride = kMaxSplitK * lse_chunk_stride;

  const int input_seq_stride = num_head_q * num_dim_v;
  const int input_chunk_stride = num_seq_q * input_seq_stride;
  const int input_batch_stride = kMaxSplitK * input_chunk_stride;

  const float *lse_batch = lse_ptr + ibatch * lse_batch_stride + ihead_kv * lse_kv_head_stride +
                           iseq * pad_heads_per_group + ihead;
  const float *split_input = split_input_ptr + ibatch * input_batch_stride +
                             iseq * input_seq_stride + ihead_q * num_dim_v + icol;
  T *out_row = y_ptr + ibatch * num_seq_q * input_seq_stride + iseq * input_seq_stride +
               ihead_q * num_dim_v + icol;

  __shared__ float lse[kWarpCount][kMaxSplitK];
  vec_t<float, kItemsPerThread> output;
#pragma unroll
  for (int i = 0; i < kItemsPerThread; i++) {
    output[i] = 0.f;
  }
  float max_lse = -std::numeric_limits<float>::infinity();
  float sum_lse = 0.f;

  cudaGridDependencySynchronize();

  vec_t<float, kItemsPerThread> buf[kPrefetch];
  const int first_group = num_chunks < kPrefetch ? num_chunks : kPrefetch;
#pragma unroll
  for (int s = 0; s < kPrefetch; s++) {
    if (s < first_group) {
      buf[s] = load<float, kItemsPerThread>(split_input + s * input_chunk_stride);
    }
  }

#pragma unroll 1
  for (int ichunk = ilane; ichunk < num_chunks; ichunk += 32) {
    lse[iwarp][ichunk] = lse_batch[ichunk * lse_chunk_stride];
    max_lse = max(max_lse, lse[iwarp][ichunk]);
  }

  max_lse = warp_reduce_max_xor(max_lse);

#pragma unroll 1
  for (int ichunk = ilane; ichunk < num_chunks; ichunk += 32) {
    float w = exp2f_ftz(lse[iwarp][ichunk] - max_lse);
    lse[iwarp][ichunk] = w;
    sum_lse += w;
  }

  const float sum = warp_reduce_sum_xor(sum_lse);
  const float inv_sum = sum > 0.f ? 1.f / sum : 0.f;

  __syncwarp();

  // kPrefetch-deep software pipeline so per-chunk loads overlap.
  int ichunk = 0;
  for (; ichunk + kPrefetch <= num_chunks; ichunk += kPrefetch) {
    if (ichunk != 0) {
#pragma unroll
      for (int s = 0; s < kPrefetch; s++) {
        buf[s] = load<float, kItemsPerThread>(split_input + (ichunk + s) * input_chunk_stride);
      }
    }
#pragma unroll
    for (int s = 0; s < kPrefetch; s++) {
      float weight = lse[iwarp][ichunk + s];
#pragma unroll
      for (int i = 0; i < kItemsPerThread; i++) {
        output[i] += weight * buf[s][i];
      }
    }
  }

  int rem = num_chunks - ichunk;
  if (ichunk != 0) {
#pragma unroll
    for (int s = 0; s < kPrefetch; s++) {
      if (s < rem) {
        buf[s] = load<float, kItemsPerThread>(split_input + (ichunk + s) * input_chunk_stride);
      }
    }
  }
#pragma unroll
  for (int s = 0; s < kPrefetch; s++) {
    if (s < rem) {
      float weight = lse[iwarp][ichunk + s];
#pragma unroll
      for (int i = 0; i < kItemsPerThread; i++) {
        output[i] += weight * buf[s][i];
      }
    }
  }

#pragma unroll
  for (int i = 0; i < kItemsPerThread; i++) {
    output[i] *= inv_sum;
  }

  store(out_row, to<T>(output));

  cudaTriggerProgrammaticLaunchCompletion();
}

template <typename T, int kWarpCount, int kMaxSplitK, int kDimTiles, int kChunkSplit>
__global__ void attention_decode_dynamic_splitk_combine_chunksplit_kernel(
    T *y_ptr, const float *split_input_ptr, const float *lse_ptr, const int *task_map_ptr,
    int num_sm_count, int num_batch, int num_seq_q, int num_head_q, int num_head_k,
    int pad_heads_per_group, int num_dim_v, cutlass::FastDivmod heads_per_group_divider) {
  static_assert(kDimTiles == 1 || kDimTiles == 2 || kDimTiles == 4, "kDimTiles must divide 4");
  static_assert(kChunkSplit >= 2, "chunksplit kernel needs kChunkSplit >= 2");
  static_assert(kWarpCount % kChunkSplit == 0, "kWarpCount must be a multiple of kChunkSplit");
  constexpr int kItemsPerThread = 4 / kDimTiles;
  constexpr int kColsPerTile = 32 * kItemsPerThread;
  constexpr int kSlotsPerBlock = kWarpCount / kChunkSplit;
  constexpr int kPrefetch = (8 / kItemsPerThread) > 2 ? (8 / kItemsPerThread) : 2;

  const int iwarp = threadIdx.x / 32;
  const int ilane = threadIdx.x % 32;
  const int islot_in_block = iwarp / kChunkSplit;
  const int igroup = iwarp % kChunkSplit;
  const int islot = blockIdx.x * kSlotsPerBlock + islot_in_block;

  const int iseq = blockIdx.y;
  const int ibatch = blockIdx.z;
  const int idimtile = islot % kDimTiles;
  const int ihead_q = islot / kDimTiles;
  int ihead_kv, ihead;
  heads_per_group_divider(ihead_kv, ihead, ihead_q);
  const int icol = idimtile * kColsPerTile + ilane * kItemsPerThread;

  const int num_tiles_per_sm = task_map_ptr[0];
  constexpr int kTaskStride = dynamic::kTaskScheduleInfoStride;
  const int num_chunks = task_map_ptr[(num_tiles_per_sm * num_sm_count + 1) * kTaskStride +
                                      ihead_kv * num_batch + ibatch];

  const int lse_kv_head_stride = num_seq_q * pad_heads_per_group;
  const int lse_chunk_stride = num_head_k * lse_kv_head_stride;
  const int lse_batch_stride = kMaxSplitK * lse_chunk_stride;

  const int input_seq_stride = num_head_q * num_dim_v;
  const int input_chunk_stride = num_seq_q * input_seq_stride;
  const int input_batch_stride = kMaxSplitK * input_chunk_stride;

  const float *lse_batch = lse_ptr + ibatch * lse_batch_stride + ihead_kv * lse_kv_head_stride +
                           iseq * pad_heads_per_group + ihead;
  const float *split_group = split_input_ptr + ibatch * input_batch_stride +
                             iseq * input_seq_stride + ihead_q * num_dim_v + icol +
                             igroup * input_chunk_stride;
  const int group_stride = kChunkSplit * input_chunk_stride;
  T *out_row = y_ptr + ibatch * num_seq_q * input_seq_stride + iseq * input_seq_stride +
               ihead_q * num_dim_v + icol;

  const int stripe_len = igroup < num_chunks ? (num_chunks - 1 - igroup) / kChunkSplit + 1 : 0;

  __shared__ float sout[kSlotsPerBlock][kChunkSplit][kColsPerTile];
  __shared__ float sweight[kWarpCount][kMaxSplitK];

  vec_t<float, kItemsPerThread> output;
#pragma unroll
  for (int i = 0; i < kItemsPerThread; i++) {
    output[i] = 0.f;
  }

  cudaGridDependencySynchronize();

  vec_t<float, kItemsPerThread> buf[kPrefetch];
  const int first_group = stripe_len < kPrefetch ? stripe_len : kPrefetch;
#pragma unroll
  for (int s = 0; s < kPrefetch; s++) {
    if (s < first_group) {
      buf[s] = load<float, kItemsPerThread>(split_group + s * group_stride);
    }
  }

  float max_lse = -std::numeric_limits<float>::infinity();
#pragma unroll 1
  for (int ichunk = ilane; ichunk < num_chunks; ichunk += 32) {
    float v = lse_batch[ichunk * lse_chunk_stride];
    sweight[iwarp][ichunk] = v;
    max_lse = max(max_lse, v);
  }
  max_lse = warp_reduce_max_xor(max_lse);

  float sum_lse = 0.f;
#pragma unroll 1
  for (int ichunk = ilane; ichunk < num_chunks; ichunk += 32) {
    float w = exp2f_ftz(sweight[iwarp][ichunk] - max_lse);
    sweight[iwarp][ichunk] = w;
    sum_lse += w;
  }
  sum_lse = warp_reduce_sum_xor(sum_lse);
  const float inv_sum = sum_lse > 0.f ? 1.f / sum_lse : 0.f;
  __syncwarp();

  int j = 0;
  for (; j + kPrefetch <= stripe_len; j += kPrefetch) {
    if (j != 0) {
#pragma unroll
      for (int s = 0; s < kPrefetch; s++) {
        buf[s] = load<float, kItemsPerThread>(split_group + (j + s) * group_stride);
      }
    }
#pragma unroll
    for (int s = 0; s < kPrefetch; s++) {
      float weight = sweight[iwarp][igroup + (j + s) * kChunkSplit];
#pragma unroll
      for (int i = 0; i < kItemsPerThread; i++) {
        output[i] += weight * buf[s][i];
      }
    }
  }
  int rem = stripe_len - j;
  if (j != 0) {
#pragma unroll
    for (int s = 0; s < kPrefetch; s++) {
      if (s < rem) {
        buf[s] = load<float, kItemsPerThread>(split_group + (j + s) * group_stride);
      }
    }
  }
#pragma unroll
  for (int s = 0; s < kPrefetch; s++) {
    if (s < rem) {
      float weight = sweight[iwarp][igroup + (j + s) * kChunkSplit];
#pragma unroll
      for (int i = 0; i < kItemsPerThread; i++) {
        output[i] += weight * buf[s][i];
      }
    }
  }

#pragma unroll
  for (int i = 0; i < kItemsPerThread; i++) {
    sout[islot_in_block][igroup][ilane * kItemsPerThread + i] = output[i];
  }
  __syncthreads();

  if (igroup == 0) {
    vec_t<float, kItemsPerThread> acc;
#pragma unroll
    for (int i = 0; i < kItemsPerThread; i++) {
      acc[i] = 0.f;
    }
#pragma unroll
    for (int g = 0; g < kChunkSplit; g++) {
#pragma unroll
      for (int i = 0; i < kItemsPerThread; i++) {
        acc[i] += sout[islot_in_block][g][ilane * kItemsPerThread + i];
      }
    }
#pragma unroll
    for (int i = 0; i < kItemsPerThread; i++) {
      acc[i] *= inv_sum;
    }
    store(out_row, to<T>(acc));
  }

  cudaTriggerProgrammaticLaunchCompletion();
}

}  // namespace kernels
}  // namespace decode
}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_DECODE_SPLITK_COMBINE_KERNELS_DIMTILED_CUH_
