// Copyright 2025 hpc-ops authors

#ifndef SRC_ATTENTION_DECODE_SM100_SMALLM_SPLITK_COMBINE_KERNELS_CUH_
#define SRC_ATTENTION_DECODE_SM100_SMALLM_SPLITK_COMBINE_KERNELS_CUH_

#include <cuda.h>
#include <stdio.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "src/attention/decode/sched_task_info.h"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace attention {
namespace kernels {

template <typename T, int kWarpCount, int kMaxSplitK>
__global__ void attention_decode_smallm_splitk_combine_kernel(
    T *y_ptr, const float *split_input_ptr, const float *lse_ptr, const int *task_map_ptr,
    int num_sm_count, int num_seq_q, int num_head_q, int num_head_k, int pad_heads_per_group,
    int num_dim_v, cutlass::FastDivmod heads_per_group_divider) {
  constexpr int kItemsPerThread = 4;
  int iseq = blockIdx.y;
  int ibatch = blockIdx.z;
  int ihead_q = blockIdx.x * kWarpCount + threadIdx.x / 32;
  int ihead_kv, ihead;
  heads_per_group_divider(ihead_kv, ihead, ihead_q);
  int ilane = threadIdx.x % 32;
  int icol = ilane * kItemsPerThread;

  int num_tiles_per_sm = task_map_ptr[0];
  constexpr int kTaskStride = sizeof(TaskScheduleInfo) / sizeof(int);
  int num_chunks = task_map_ptr[(num_tiles_per_sm * num_sm_count + 1) * kTaskStride + ibatch];

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

  vec_t<float, kMaxSplitK> lse;
  vec_t<float, kItemsPerThread> output;
#pragma unroll
  for (int i = 0; i < kItemsPerThread; i++) {
    output[i] = 0.f;
  }
  float max_lse = 0.f;
  float sum_lse = 0.f;

  cudaGridDependencySynchronize();

  if (num_chunks > 1) {
#pragma unroll
    for (int ichunk = 0; ichunk < kMaxSplitK; ichunk++) {
      if (ichunk < num_chunks) {
        lse[ichunk] = lse_batch[ichunk * lse_chunk_stride];
        max_lse = max(max_lse, lse[ichunk]);
      }
    }

#pragma unroll
    for (int ichunk = 0; ichunk < kMaxSplitK; ichunk++) {
      if (ichunk < num_chunks) {
        sum_lse += exp2f_ftz(lse[ichunk] - max_lse);
      }
    }

    sum_lse = log2f_ftz(sum_lse) + max_lse;

#pragma unroll
    for (int ichunk = 0; ichunk < kMaxSplitK; ichunk++) {
      if (ichunk < num_chunks) {
        auto y = load<float, kItemsPerThread>(split_input + ichunk * input_chunk_stride);
        float scale = exp2f_ftz(lse[ichunk] - sum_lse);

#pragma unroll
        for (int i = 0; i < kItemsPerThread; i++) {
          output[i] += scale * y[i];
        }
      }
    }
  } else {
    output = load<float, kItemsPerThread>(split_input);
  }

  store(out_row, to<T>(output));

  cudaTriggerProgrammaticLaunchCompletion();
}

}  // namespace kernels
}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_DECODE_SM100_SMALLM_SPLITK_COMBINE_KERNELS_CUH_
