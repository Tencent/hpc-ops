// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <math.h>
#include <stdio.h>

#include <cub/cub.cuh>
#include <limits>

#include "src/topk/topk.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace topk {

namespace kernels {

__device__ bool is_float_nan_fast(float f) {
  const uint32_t u = *reinterpret_cast<const uint32_t*>(&f);
  const uint32_t exponent = u & 0x7F800000;
  const uint32_t mantissa = u & 0x007FFFFF;
  return (exponent == 0x7F800000) && (mantissa != 0);
}

template <bool kUseGroup, bool kReNorm>
__global__ void grouped_topk_kernel(float* topk_weights_ptr, int* topk_ids_ptr,
                                    const float* scores_ptr, const float* bias_ptr, float scale,
                                    int topk, int topk_group, int num_experts,
                                    int num_expert_group) {
  int idx = threadIdx.x;
  int irow = blockIdx.x;
  int icol = idx * 4;

  const auto* scores_row = scores_ptr + irow * num_experts;
  auto* topk_ids_row = topk_ids_ptr + irow * topk;
  auto* topk_weights_row = topk_weights_ptr + irow * topk;

  __shared__ float smem_scores[256];
  __shared__ float smem_bias[256];

  if (icol < num_experts) {
    auto scores = load<float, 4>(scores_row + icol);

    // 1. sigmoid
#pragma unroll
    for (int i = 0; i < 4; i++) {
      scores[i] = sigmoid(is_float_nan_fast(scores[i]) ? 0.f : scores[i]);
    }

    // 2. add bias
    if (bias_ptr) {
      auto bias = load<float, 4>(bias_ptr + icol);
#pragma unroll
      for (int i = 0; i < 4; i++) {
        scores[i] += bias[i];
      }
      store(smem_bias + icol, bias);
    }

    // 3. store to smem
    store(smem_scores + icol, scores);
  }

  __syncthreads();

  // 4. remap task to thread
  if constexpr (kUseGroup) {
  } else {
    float score = 0.f;
    float bias = 0.f;
    int count = topk;
    if (idx < num_experts) {
      count = 0;
      score = smem_scores[idx];
      bias = smem_bias[idx];
      for (int i = 0; i < num_experts; i += 4) {
        auto other_score = load<float, 4>(smem_scores + i);
        for (int j = 0; j < 4; j++) {
          if (other_score[j] > score || (score == other_score[j] && (i + j < idx))) {
            count++;
          }
        }
      }
    }

    __syncthreads();
    if (count < topk) {
      score -= bias;
      smem_scores[count] = score;
      topk_ids_row[count] = idx;
    }

    __syncthreads();
    if (count < topk) {
      if constexpr (kReNorm) {
        float sum = 0;
        for (int i = 0; i < topk; i++) {
          sum += smem_scores[i];
        }
        score *= rcpf_ftz(sum);
      }
      score *= scale;
      topk_weights_row[count] = score;
    }
  }
}

}  // namespace kernels

bool grouped_topk_async(float* topk_weights_ptr, int* topk_ids_ptr, const float* scores_ptr,
                        const float* bias_ptr, float scale, int num_tokens, int topk,
                        int topk_group, int num_experts, int num_expert_group, bool renormalize,
                        cudaStream_t stream) {
  dim3 block(256);
  dim3 grid(num_tokens);

  if (num_expert_group == 1) {
    if (renormalize) {
      kernels::grouped_topk_kernel<false, true>
          <<<grid, block, 0, stream>>>(topk_weights_ptr, topk_ids_ptr, scores_ptr, bias_ptr, scale,
                                       topk, topk_group, num_experts, num_expert_group);
    } else {
      kernels::grouped_topk_kernel<false, false>
          <<<grid, block, 0, stream>>>(topk_weights_ptr, topk_ids_ptr, scores_ptr, bias_ptr, scale,
                                       topk, topk_group, num_experts, num_expert_group);
    }
  } else {
    return false;
  }

  return true;
}

}  // namespace topk
}  // namespace hpc
