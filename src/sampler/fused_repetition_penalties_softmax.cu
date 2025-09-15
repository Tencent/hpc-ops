// Copyright 2025 hpc-ops authors

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

template <int kVocabSize, int kThreadsPerBlock = 1024, int kClusterSize = 8>
__global__ void cluster_fused_repetition_penalties_softmax_kernel(
    float* out, const float* logits, const uint8_t** penalties_masks_ptrs,
    const float* repetition_penalties_arr, float repetition_penalties_val,
    const float* temperature_arr, float temperature_val) {
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
  if (penalties_masks_ptrs) {
    penalties_masks_batch = penalties_masks_ptrs[ibatch];
  }
  float temperature = temperature_arr ? temperature_arr[ibatch] : temperature_val;
  float inv_temperature = temperature <= 0.f ? 0.f : rcpf_ftz(temperature);
  float repetition_penalties =
      repetition_penalties_arr ? repetition_penalties_arr[ibatch] : repetition_penalties_val;
  float inv_repetition_penalties =
      repetition_penalties <= 0.f ? 0.f : rcpf_ftz(repetition_penalties);
  vec_t<float, kItemPer16B> local_logits[kIters];

  float local_max = 0.f;
  float local_sum = 0.f;

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
      if (penalties_masks_batch != nullptr) {
        mask = penalties_masks_batch[icol / 8];
        if (icol % 8 >= 4) {
          mask >>= 4;
        }
      }

#pragma unroll
      for (int i = 0; i < kItemPer16B; i++) {
        if (repetition_penalties > 0.f) {
          if (mask & (1 << i)) {
            local_logits[iter][i] *=
                (local_logits[iter][i] > 0.f) ? inv_repetition_penalties : repetition_penalties;
          }
        }
        if (temperature > 0.f) {
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

template <int kThreadsPerBlock = 1024, int kItemPerLoad = 4, int kStage = 1>
__global__ void block_fused_repetition_penalties_softmax_kernel(
    float* out, const float* logits, const uint8_t** penalties_masks_ptrs,
    const float* repetition_penalties_arr, float repetition_penalties_val,
    const float* temperature_arr, float temperature_val, const int64_t vocab_size) {
  constexpr int kItemsPerStage = kItemPerLoad * kThreadsPerBlock;
  constexpr int kItemsPerIter = kItemsPerStage * kStage;

  const int kIters = vocab_size / kItemsPerIter;

  constexpr int kWarpCount = kThreadsPerBlock / 32;

  __shared__ float smem[kWarpCount];

  int ibatch = blockIdx.x;
  int iwarp = threadIdx.x / 32;
  int ilane = threadIdx.x % 32;
  int idx = threadIdx.x;

  float* out_batch = out + ibatch * vocab_size;
  const float* logits_batch = logits + ibatch * vocab_size;

  const uint8_t* penalties_masks_batch = nullptr;
  if (penalties_masks_ptrs) {
    penalties_masks_batch = penalties_masks_ptrs[ibatch];
  }
  float temperature = temperature_arr ? temperature_arr[ibatch] : temperature_val;
  float inv_temperature = temperature <= 0.f ? 0.f : rcpf_ftz(temperature);
  float repetition_penalties =
      repetition_penalties_arr ? repetition_penalties_arr[ibatch] : repetition_penalties_val;
  float inv_repetition_penalties =
      repetition_penalties <= 0.f ? 0.f : rcpf_ftz(repetition_penalties);

  vec_t<float, kItemPerLoad> local_logits[kStage];
  vec_t<uint8_t, kStage> masks;
  float local_max = 0;
  float local_sum = 0;

  // seperate iter loops to two parts:
  // 1. first part deal with data within [0: vocab_size / kItemsPerIter * kItemsPerIter] ,
  //    so we don't need add bound checking.
  // 2. second part deal with data within [vocab_size / kItemsPerIter * kItemsPerIter:],
  //    which not all threads participat in, so we add extra 'if' instruction.

  // step 1. Load logits and calculate local max.
  for (int iter = 0; iter < kIters; iter++) {
#pragma unroll
    for (int istage = 0; istage < kStage; istage++) {
      int64_t icol = iter * kItemsPerIter + istage * kItemsPerStage + idx * kItemPerLoad;
      local_logits[istage] = load<float, kItemPerLoad>(logits_batch + icol);
    }

    if (repetition_penalties > 0.f && penalties_masks_batch != nullptr) {
#pragma unroll
      for (int istage = 0; istage < kStage; istage++) {
        int64_t icol = iter * kItemsPerIter + istage * kItemsPerStage + idx * kItemPerLoad;
        int ibit = icol % 8;
        masks[istage] = penalties_masks_batch[icol / 8];
        masks[istage] >>= ibit;
#pragma unroll
        for (int i = 0; i < kItemPerLoad; i++) {
          if (masks[istage] & (1 << i)) {
            local_logits[istage][i] *=
                (local_logits[istage][i] > 0.f) ? inv_repetition_penalties : repetition_penalties;
          }
        }
      }
    }

#pragma unroll
    for (int istage = 0; istage < kStage; istage++) {
      int64_t icol = iter * kItemsPerIter + istage * kItemsPerStage + idx * kItemPerLoad;
#pragma unroll
      for (int i = 0; i < kItemPerLoad; i++) {
        if (temperature > 0.f) {
          local_logits[istage][i] *= inv_temperature;
        }
        local_max = fmaxf(local_max, local_logits[istage][i]);
      }
      if (repetition_penalties > 0.f || temperature > 0.f) {
        store<float, kItemPerLoad>(out_batch + icol, local_logits[istage]);
      }
    }
  }

  {
#pragma unroll
    for (int istage = 0; istage < kStage; istage++) {
      int64_t icol = kIters * kItemsPerIter + istage * kItemsPerStage + idx * kItemPerLoad;
      if (icol + kItemPerLoad <= vocab_size) {
        local_logits[istage] = load<float, kItemPerLoad>(logits_batch + icol);
      } else {
        break;
      }
    }

    if (repetition_penalties > 0.f && penalties_masks_batch != nullptr) {
#pragma unroll
      for (int istage = 0; istage < kStage; istage++) {
        int64_t icol = kIters * kItemsPerIter + istage * kItemsPerStage + idx * kItemPerLoad;
        if (icol + kItemPerLoad > vocab_size) break;
        int ibit = icol % 8;
        masks[istage] = penalties_masks_batch[icol / 8];
        masks[istage] >>= ibit;
#pragma unroll
        for (int i = 0; i < kItemPerLoad; i++) {
          if (masks[istage] & (1 << i)) {
            local_logits[istage][i] *=
                (local_logits[istage][i] > 0.f) ? inv_repetition_penalties : repetition_penalties;
          }
        }
      }
    }

#pragma unroll
    for (int istage = 0; istage < kStage; istage++) {
      int64_t icol = kIters * kItemsPerIter + istage * kItemsPerStage + idx * kItemPerLoad;
      if (icol + kItemPerLoad > vocab_size) break;
#pragma unroll
      for (int i = 0; i < kItemPerLoad; i++) {
        if (temperature > 0.f) {
          local_logits[istage][i] *= inv_temperature;
        }
        local_max = fmaxf(local_max, local_logits[istage][i]);
      }
      if (repetition_penalties > 0.f || temperature > 0.f) {
        store<float, kItemPerLoad>(out_batch + icol, local_logits[istage]);
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
#pragma unroll
    for (int istage = 0; istage < kStage; istage++) {
      int64_t icol = iter * kItemsPerIter + istage * kItemsPerStage + idx * kItemPerLoad;
      if (repetition_penalties > 0. || temperature > 0.) {
        local_logits[istage] = load<float, kItemPerLoad>(out_batch + icol);
      } else {
        local_logits[istage] = load<float, kItemPerLoad>(logits_batch + icol);
      }
    }

#pragma unroll
    for (int istage = 0; istage < kStage; istage++) {
      int64_t icol = iter * kItemsPerIter + istage * kItemsPerStage + idx * kItemPerLoad;
#pragma unroll
      for (int i = 0; i < kItemPerLoad; i++) {
        local_logits[istage][i] = expf_ftz(local_logits[istage][i] - local_max);
        local_sum += local_logits[istage][i];
      }
    }
  }

  {
#pragma unroll
    for (int istage = 0; istage < kStage; istage++) {
      int64_t icol = kIters * kItemsPerIter + istage * kItemsPerStage + idx * kItemPerLoad;
      if (icol + kItemPerLoad <= vocab_size) {
        if (repetition_penalties > 0. || temperature > 0.) {
          local_logits[istage] = load<float, kItemPerLoad>(out_batch + icol);
        } else {
          local_logits[istage] = load<float, kItemPerLoad>(logits_batch + icol);
        }
      } else {
        break;
      }
    }

#pragma unroll
    for (int istage = 0; istage < kStage; istage++) {
      int64_t icol = kIters * kItemsPerIter + istage * kItemsPerStage + idx * kItemPerLoad;
#pragma unroll
      if (icol + kItemPerLoad <= vocab_size) {
        for (int i = 0; i < kItemPerLoad; i++) {
          local_logits[istage][i] = expf_ftz(local_logits[istage][i] - local_max);
          local_sum += local_logits[istage][i];
        }
      } else {
        break;
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
#pragma unroll
    for (int istage = 0; istage < kStage; istage++) {
      int64_t icol = iter * kItemsPerIter + istage * kItemsPerStage + idx * kItemPerLoad;
      if (repetition_penalties > 0.f || temperature > 0.f) {
        local_logits[istage] = load<float, kItemPerLoad>(out_batch + icol);
      } else {
        local_logits[istage] = load<float, kItemPerLoad>(logits_batch + icol);
      }
    }

#pragma unroll
    for (int istage = 0; istage < kStage; istage++) {
      int64_t icol = iter * kItemsPerIter + istage * kItemsPerStage + idx * kItemPerLoad;
#pragma unroll
      for (int i = 0; i < kItemPerLoad; i++) {
        local_logits[istage][i] = inv_local_sum * expf_ftz(local_logits[istage][i] - local_max);
      }
      store<float, kItemPerLoad>(out_batch + icol, local_logits[istage]);
    }
  }

  {
#pragma unroll
    for (int istage = 0; istage < kStage; istage++) {
      int64_t icol = kIters * kItemsPerIter + istage * kItemsPerStage + idx * kItemPerLoad;
      if (icol + kItemPerLoad <= vocab_size) {
        if (repetition_penalties > 0.f || temperature > 0.f) {
          local_logits[istage] = load<float, kItemPerLoad>(out_batch + icol);
        } else {
          local_logits[istage] = load<float, kItemPerLoad>(logits_batch + icol);
        }
      } else {
        break;
      }
    }

#pragma unroll
    for (int istage = 0; istage < kStage; istage++) {
      int64_t icol = kIters * kItemsPerIter + istage * kItemsPerStage + idx * kItemPerLoad;
      if (icol + kItemPerLoad <= vocab_size) {
#pragma unroll
        for (int i = 0; i < kItemPerLoad; i++) {
          local_logits[istage][i] = inv_local_sum * expf_ftz(local_logits[istage][i] - local_max);
        }
        store<float, kItemPerLoad>(out_batch + icol, local_logits[istage]);
      } else {
        break;
      }
    }
  }
}

}  // namespace kernels
}  // namespace fused_repetition_penalties_softmax

void fused_repetition_penalties_softmax_async(
    float* out_ptr, const float* logits_ptr, const uint8_t** penalties_masks_ptrs,
    const float* repetition_penalties, float repetition_penalties_val, const float* temperature,
    float temperature_val, const int num_batch, const int vocab_size, cudaStream_t stream) {
  constexpr int kThreadsPerBlock = 1024;
  constexpr int kClusterSize = 8;

  if (num_batch <= 32 && (vocab_size == 129024 || vocab_size == 128512 || vocab_size == 129280)) {
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

    if (vocab_size == 129024) {
      cudaLaunchKernelEx(
          &config,
          fused_repetition_penalties_softmax::kernels::
              cluster_fused_repetition_penalties_softmax_kernel<129024, kThreadsPerBlock,
                                                                kClusterSize>,
          out_ptr, logits_ptr, penalties_masks_ptrs, repetition_penalties, repetition_penalties_val,
          temperature, temperature_val);
    } else if (vocab_size == 128512) {
      cudaLaunchKernelEx(
          &config,
          fused_repetition_penalties_softmax::kernels::
              cluster_fused_repetition_penalties_softmax_kernel<128512, kThreadsPerBlock,
                                                                kClusterSize>,
          out_ptr, logits_ptr, penalties_masks_ptrs, repetition_penalties, repetition_penalties_val,
          temperature, temperature_val);
    } else {  // vocab_size == 129280
      cudaLaunchKernelEx(
          &config,
          fused_repetition_penalties_softmax::kernels::
              cluster_fused_repetition_penalties_softmax_kernel<129280, kThreadsPerBlock,
                                                                kClusterSize>,
          out_ptr, logits_ptr, penalties_masks_ptrs, repetition_penalties, repetition_penalties_val,
          temperature, temperature_val);
    }
  } else {
    dim3 grid(num_batch);
    dim3 block(kThreadsPerBlock);

    if (vocab_size % 4 == 0) {
      constexpr int kItemsPerLoad = 4;
      constexpr int kStage = 4;
      fused_repetition_penalties_softmax::kernels::block_fused_repetition_penalties_softmax_kernel<
          kThreadsPerBlock, kItemsPerLoad, kStage><<<grid, block, 0, stream>>>(
          out_ptr, logits_ptr, penalties_masks_ptrs, repetition_penalties, repetition_penalties_val,
          temperature, temperature_val, vocab_size);
    } else if (vocab_size % 2 == 0) {
      constexpr int kItemsPerLoad = 2;
      constexpr int kStage = 7;
      fused_repetition_penalties_softmax::kernels::block_fused_repetition_penalties_softmax_kernel<
          kThreadsPerBlock, kItemsPerLoad, kStage><<<grid, block, 0, stream>>>(
          out_ptr, logits_ptr, penalties_masks_ptrs, repetition_penalties, repetition_penalties_val,
          temperature, temperature_val, vocab_size);
    } else {
      constexpr int kItemsPerLoad = 1;
      constexpr int kStage = 8;
      fused_repetition_penalties_softmax::kernels::block_fused_repetition_penalties_softmax_kernel<
          kThreadsPerBlock, kItemsPerLoad, kStage><<<grid, block, 0, stream>>>(
          out_ptr, logits_ptr, penalties_masks_ptrs, repetition_penalties, repetition_penalties_val,
          temperature, temperature_val, vocab_size);
    }
  }
}

}  // namespace sampler
}  // namespace hpc
