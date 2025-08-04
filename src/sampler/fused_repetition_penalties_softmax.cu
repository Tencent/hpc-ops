#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <stdio.h>

#include "src/sampler/fused_repetition_penalties_softmax.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace sampler {
namespace fused_repetition_penalties_softmax {
namespace kernels {

template <int kVocabSize, int kThreadsPerBlock = 1024, int kClusterSize = 8,
          bool kOnlySoftmax = false>
__global__ void cluster_fused_repetition_penalties_softmax_kernel(
    float* out, const float* logits, const uint8_t** penalties_masks_ptrs,
    const float repetition_penalties, const float inv_temperature) {
  constexpr int kItemPer16B = 4;
  constexpr int kItemsPerIter = kItemPer16B * kThreadsPerBlock * kClusterSize;
  constexpr int kIters = (kVocabSize + kItemsPerIter - 1) / kItemsPerIter;
  constexpr int kWarpCount = kThreadsPerBlock / 32;

  extern __shared__ float smem[];

  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();

  int ibatch = blockIdx.y;
  int iwarp = threadIdx.x / 32;
  int ilane = threadIdx.x % 32;
  int idx = cluster.thread_rank();
  int iblock_in_cluster = cluster.block_rank();

  float* out_batch = out + ibatch * kVocabSize;
  const float* logits_batch = logits + ibatch * kVocabSize;

  const uint8_t* penalties_masks_batch = nullptr;
  float inv_repetition_penalties = 0;
  if constexpr (!kOnlySoftmax) {
    penalties_masks_batch = penalties_masks_ptrs[ibatch];
    inv_repetition_penalties = rcpf_ftz(repetition_penalties);
  }
  vec_t<float, kItemPer16B> local_logits[kIters];

  float local_max = 0;
  float local_sum = 0;

  // step 1.
  //   (1) Load logits from global memory, apply repetition_penalties according
  //   to mask. (2) apply temperature and exp. (3) get max value in thread
  //   level.
#pragma unroll
  for (int iter = 0; iter < kIters; iter++) {
    int64_t icol = iter * kItemsPerIter + idx * kItemPer16B;
    if (icol + kItemPer16B <= kVocabSize) {
      local_logits[iter] = load<float, kItemPer16B>(logits_batch + icol);
      uint8_t mask = 0;
      if constexpr (!kOnlySoftmax) {
        mask = penalties_masks_batch[icol / 8];
        if (icol % 8 >= 4) {
          mask >>= 4;
        }
      }

#pragma unroll
      for (int i = 0; i < kItemPer16B; i++) {
        if constexpr (!kOnlySoftmax) {
          if (mask & (1 << i)) {
            local_logits[iter][i] *=
                (local_logits[iter][i] > 0) ? inv_repetition_penalties : repetition_penalties;
          }
          local_logits[iter][i] *= inv_temperature;
        }
        local_max = fmaxf(local_max, local_logits[iter][i]);
      }
    }
  }

  // step 2. Reduce max value
  //   (1) warp reduce
  //   (2) block reduce
  //   (3) cluster reduce

  local_max = warp_reduce_max_down(local_max);
  if (ilane == 0) {
    smem[iwarp] = local_max;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    local_max = 0;
#pragma unroll
    for (int i = 0; i < kWarpCount; i++) {
      local_max = fmaxf(smem[i], local_max);
    }
    smem[0] = local_max;
  }
  cluster.sync();
  if (idx == 0) {
    local_max = smem[0];
#pragma unroll
    for (int i = 1; i < kClusterSize; i++) {
      local_max = fmaxf(cluster.map_shared_rank(smem, i)[0], local_max);
    }
    smem[0] = local_max;
  }
  cluster.sync();
  if (iblock_in_cluster > 0) {
    local_max = cluster.map_shared_rank(smem, 0)[0];
  } else {
    local_max = smem[0];
  }
  cluster.sync();

  // step 3. calculate exp(x - max) and local_sum
#pragma unroll
  for (int iter = 0; iter < kIters; iter++) {
    int64_t icol = iter * kItemsPerIter + idx * kItemPer16B;
    if (icol + kItemPer16B <= kVocabSize) {
#pragma unroll
      for (int i = 0; i < kItemPer16B; i++) {
        local_logits[iter][i] = expf_ftz(local_logits[iter][i] - local_max);
        local_sum += local_logits[iter][i];
      }
    }
  }

  // step 4. Reduce sum value
  // warp reduce
  local_sum = warp_reduce_sum_down(local_sum);
  // block reduce
  if (ilane == 0) {
    smem[iwarp] = local_sum;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    local_sum = 0;
#pragma unroll
    for (int i = 0; i < kWarpCount; i++) {
      local_sum += smem[i];
    }
    smem[0] = local_sum;
  }
  // cluster reduce
  cluster.sync();
  if (idx == 0) {
    local_sum = smem[0];
#pragma unroll
    for (int i = 1; i < kClusterSize; i++) {
      local_sum += cluster.map_shared_rank(smem, i)[0];
    }
    smem[0] = local_sum;
  }
  cluster.sync();
  if (iblock_in_cluster > 0) {
    local_sum = cluster.map_shared_rank(smem, 0)[0];
  } else {
    local_sum = smem[0];
  }
  cluster.sync();

  float inv_local_sum = rcpf_ftz(local_sum);

  // step 5. store outputs
#pragma unroll
  for (int iter = 0; iter < kIters; iter++) {
    int64_t icol = iter * kItemsPerIter + idx * kItemPer16B;
    if (icol + kItemPer16B <= kVocabSize) {
#pragma unroll
      for (int i = 0; i < kItemPer16B; i++) {
        local_logits[iter][i] *= inv_local_sum;
      }
      store<float, kItemPer16B>(out_batch + icol, local_logits[iter]);
    }
  }
}

template <int kThreadsPerBlock = 1024, bool kOnlySoftmax = false>
__global__ void block_fused_repetition_penalties_softmax_kernel(
    float* out, const float* logits, const uint8_t** penalties_masks_ptrs,
    const float repetition_penalties, const float inv_temperature, const int64_t vocab_size) {
  constexpr int kItemPer16B = 4;
  constexpr int kItemsPerIter = kItemPer16B * kThreadsPerBlock;
  const int kIters = (vocab_size + kItemsPerIter - 1) / kItemsPerIter;
  constexpr int kWarpCount = kThreadsPerBlock / 32;

  __shared__ float smem[kWarpCount];

  int ibatch = blockIdx.x;
  int iwarp = threadIdx.x / 32;
  int ilane = threadIdx.x % 32;
  int idx = threadIdx.x;

  float* out_batch = out + ibatch * vocab_size;
  const float* logits_batch = logits + ibatch * vocab_size;

  const uint8_t* penalties_masks_batch = nullptr;
  float inv_repetition_penalties = 0;
  if constexpr (!kOnlySoftmax) {
    penalties_masks_batch = penalties_masks_ptrs[ibatch];
    inv_repetition_penalties = rcpf_ftz(repetition_penalties);
  }

  vec_t<float, kItemPer16B> local_logits;

  float local_max = 0;
  float local_sum = 0;

  // step 1. Load logits and calculate local max.
  for (int iter = 0; iter < kIters; iter++) {
    int64_t icol = iter * kItemsPerIter + idx * kItemPer16B;
    if (icol + kItemPer16B <= vocab_size) {
      local_logits = load<float, kItemPer16B>(logits_batch + icol);
      uint8_t mask = 0;
      if constexpr (!kOnlySoftmax) {
        mask = penalties_masks_batch[icol / 8];
        if (icol % 8 >= 4) {
          mask >>= 4;
        }
      }

#pragma unroll
      for (int i = 0; i < kItemPer16B; i++) {
        if constexpr (!kOnlySoftmax) {
          if (mask & (1 << i)) {
            local_logits[i] *=
                (local_logits[i] > 0) ? inv_repetition_penalties : repetition_penalties;
          }
          local_logits[i] *= inv_temperature;
        }
        local_max = fmaxf(local_max, local_logits[i]);
      }
      if constexpr (!kOnlySoftmax) {
        store<float, kItemPer16B>(out_batch + icol, local_logits);
      }
    }
  }

  // step 2. Reduce max value.
  local_max = warp_reduce_max_down(local_max);
  if (ilane == 0) {
    smem[iwarp] = local_max;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    local_max = 0;
#pragma unroll
    for (int i = 0; i < kWarpCount; i++) {
      local_max = fmaxf(smem[i], local_max);
    }
    smem[0] = local_max;
  }
  __syncthreads();
  local_max = smem[0];

  // step 3. calculate exp(x - max) and local_sum
  for (int iter = 0; iter < kIters; iter++) {
    int64_t icol = iter * kItemsPerIter + idx * kItemPer16B;
    if (icol + kItemPer16B <= vocab_size) {
      if constexpr (!kOnlySoftmax) {
        local_logits = load<float, kItemPer16B>(out_batch + icol);
      } else {
        local_logits = load<float, kItemPer16B>(logits_batch + icol);
      }
#pragma unroll
      for (int i = 0; i < kItemPer16B; i++) {
        local_logits[i] = expf_ftz(local_logits[i] - local_max);
        local_sum += local_logits[i];
      }
    }
  }

  __syncthreads();
  // step 4. Reduce sum value
  // warp reduce
  local_sum = warp_reduce_sum_down(local_sum);
  // block reduce
  if (ilane == 0) {
    smem[iwarp] = local_sum;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    local_sum = 0;
#pragma unroll
    for (int i = 0; i < kWarpCount; i++) {
      local_sum += smem[i];
    }
    smem[0] = local_sum;
  }
  __syncthreads();
  local_sum = smem[0];

  float inv_local_sum = rcpf_ftz(local_sum);

  // step 5. store outputs
  for (int iter = 0; iter < kIters; iter++) {
    int64_t icol = iter * kItemsPerIter + idx * kItemPer16B;
    if (icol + kItemPer16B <= vocab_size) {
      local_logits = load<float, kItemPer16B>(out_batch + icol);
#pragma unroll
      for (int i = 0; i < kItemPer16B; i++) {
        local_logits[i] = inv_local_sum * expf_ftz(local_logits[i] - local_max);
      }
      store<float, kItemPer16B>(out_batch + icol, local_logits);
    }
  }
}

}  // namespace kernels
}  // namespace fused_repetition_penalties_softmax

void fused_repetition_penalties_softmax_async(float* out_ptr, const float* logits_ptr,
                                              const uint8_t** penalties_masks_ptrs,
                                              const float repetition_penalties,
                                              const float temperature, const int num_batch,
                                              const int vocab_size, cudaStream_t stream) {
  constexpr int kThreadsPerBlock = 1024;
  constexpr int kClusterSize = 8;

  if (num_batch <= 32) {
    cudaLaunchConfig_t config;
    memset(&config, 0, sizeof(config));
    dim3 grid(kClusterSize, num_batch);
    config.gridDim = grid;
    config.blockDim = kThreadsPerBlock;
    config.dynamicSmemBytes = (kThreadsPerBlock / 32) * sizeof(float);

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = kClusterSize;
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;

    config.numAttrs = 1;
    config.attrs = attribute;

    config.stream = stream;

    cudaLaunchKernelEx(
        &config,
        fused_repetition_penalties_softmax::kernels::
            cluster_fused_repetition_penalties_softmax_kernel<129024, kThreadsPerBlock,
                                                              kClusterSize>,
        out_ptr, logits_ptr, penalties_masks_ptrs, repetition_penalties, 1.0f / temperature);
  } else {
    dim3 grid(num_batch);
    dim3 block(kThreadsPerBlock);

    fused_repetition_penalties_softmax::kernels::block_fused_repetition_penalties_softmax_kernel<
        kThreadsPerBlock><<<grid, block, 0, stream>>>(out_ptr, logits_ptr, penalties_masks_ptrs,
                                                      repetition_penalties, 1.0f / temperature,
                                                      vocab_size);
  }
}

}  // namespace sampler
}  // namespace hpc
