// Copyright (C) 2026 Tencent.

#ifndef SRC_ATTENTION_DECODE_SMALLM_SPLITK_COMBINE_KERNELS_CUH_
#define SRC_ATTENTION_DECODE_SMALLM_SPLITK_COMBINE_KERNELS_CUH_

#include <cuda.h>
#include <stdio.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace attention {
namespace kernels {

template <typename T, int kTileM, int kTileV, int kSplitK, int kSplitMinLen, int kConsumers>
__global__ void attention_decode_bf16_smallm_splitk_combine_kernel(
    T *y_ptr, const float *split_input_ptr, const float *lse_ptr, const int *num_seq_kvcache_ptr,
    bool new_kv_included, int num_head_q) {
  int ibatch = blockIdx.x;
  int ihead = threadIdx.x / 32;
  int ilane = threadIdx.x % 32;

  constexpr int kItemsPerThread = 4;
  constexpr int kSeqlenQ = 1;

  int icol = ilane * kItemsPerThread;
  int num_seq_kvcache = num_seq_kvcache_ptr[ibatch];
  if (new_kv_included) {
    num_seq_kvcache -= kSeqlenQ;
  }
  int num_seq_kv = kSeqlenQ + num_seq_kvcache;

  if (num_seq_kv <= 0) {
    return;
  }

  int num_seq_per_chunk = (num_seq_kv + kSplitK - 1) / kSplitK;
  num_seq_per_chunk = (num_seq_per_chunk + kTileM - 1) / kTileM * kTileM;
  num_seq_per_chunk = max(num_seq_per_chunk, kSplitMinLen);

  bool is_split = false;
  if (num_seq_per_chunk < num_seq_kv) {
    is_split = true;
  }

  if (!is_split && kConsumers == 1) {
    return;
  }

  constexpr int kSplitKConsumsers = kSplitK * kConsumers;
  const float *lse_batch = lse_ptr + ibatch * kSplitKConsumsers * num_head_q + ihead;
  const float *split_input =
      split_input_ptr + ibatch * kSplitKConsumsers * num_head_q * kTileV + ihead * kTileV + icol;
  T *out_row = y_ptr + ibatch * num_head_q * kTileV + ihead * kTileV + icol;

  int num_chunks = kConsumers * ((num_seq_kv + num_seq_per_chunk - 1) / num_seq_per_chunk);

  vec_t<float, kSplitKConsumsers> lse;
  vec_t<float, kItemsPerThread> output;
#pragma unroll
  for (int i = 0; i < kItemsPerThread; i++) {
    output[i] = 0.f;
  }
  float max_lse = 0.f;
  float sum_lse = 0.f;

#pragma unroll
  for (int ichunk = 0; ichunk < kSplitKConsumsers; ichunk++) {
    if (ichunk < num_chunks) {
      lse[ichunk] = lse_batch[ichunk * num_head_q];
      max_lse = max(max_lse, lse[ichunk]);
    }
  }

#pragma unroll
  for (int ichunk = 0; ichunk < kSplitKConsumsers; ichunk++) {
    if (ichunk < num_chunks) {
      sum_lse += exp2f_ftz(lse[ichunk] - max_lse);
    }
  }

  sum_lse = log2f_ftz(sum_lse) + max_lse;

#pragma unroll
  for (int ichunk = 0; ichunk < kSplitKConsumsers; ichunk++) {
    if (ichunk < num_chunks) {
      auto y = load<float, kItemsPerThread>(split_input + ichunk * num_head_q * kTileV);
      float scale = exp2f_ftz(lse[ichunk] - sum_lse);

#pragma unroll
      for (int i = 0; i < kItemsPerThread; i++) {
        output[i] += scale * y[i];
      }
    }
  }

  store(out_row, to<T>(output));
}

}  // namespace kernels
}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_DECODE_SMALLM_SPLITK_COMBINE_KERNELS_CUH_
