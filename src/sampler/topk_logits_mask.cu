// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <curand_kernel.h>
#include <stdio.h>

#include <cub/cub.cuh>
#include <limits>

#include "src/sampler/sampler.h"
#include "src/utils/utils.cuh"
namespace hpc {
namespace sampler {
namespace kernels {

template <typename KType, int kThreadsPerBlock, int kBlockPerBatch, int kItemsPerThread,
          int kSortItemsPerThread, int kMaxTopK, bool kUseVec16B>
__global__ void topk_stage1(float* output_logits, float* mid_logits, int* mid_tokens,
                            const float* logits, KType* topk, float* reject_threshold,
                            float reject_threshold_val, int vocab_size, int vocab_size_padded) {
  constexpr int kWarpSize = 32;
  constexpr int kWarpCount = kThreadsPerBlock / kWarpSize;
  constexpr int kKeepTopKTheads = kMaxTopK / kSortItemsPerThread;
  constexpr int kLoadDataTheads = kWarpSize - kKeepTopKTheads;
  constexpr float kNegInf = -std::numeric_limits<float>::infinity();

  int ibatch = blockIdx.y;
  int iwarp = threadIdx.x / 32;
  int ilane = threadIdx.x % 32;

  auto* logits_batch = logits + ibatch * vocab_size_padded;
  auto* output_logits_batch = output_logits + ibatch * vocab_size;
  auto* mid_logits_batch = mid_logits + ibatch * kBlockPerBatch * kMaxTopK;
  auto* mid_tokens_batch = mid_tokens + ibatch * kBlockPerBatch * kMaxTopK;
  float reject_threshold_batch = reject_threshold ? reject_threshold[ibatch] : reject_threshold_val;

  bool is_load_thread = ilane >= kKeepTopKTheads;

  int iload = (blockIdx.x * kLoadDataTheads * kWarpCount + iwarp * kLoadDataTheads +
               (ilane - kKeepTopKTheads)) *
              kItemsPerThread;
  int icol =
      is_load_thread
          ? iload
          : (blockIdx.x * kLoadDataTheads * kWarpCount + iwarp * kLoadDataTheads) * kItemsPerThread;

  using WarpMergeSort = cub::WarpMergeSort<float, kSortItemsPerThread, 32, int>;
  __shared__ typename WarpMergeSort::TempStorage temp_storage[kWarpCount];
  extern __shared__ float smem[];
  float* warp_reduce_logits = smem;
  int* warp_reduce_tokens = (int*)smem + kWarpCount * kMaxTopK;

  WarpMergeSort warp_merge_sort(temp_storage[iwarp]);

  vec_t<float, kSortItemsPerThread> logits_local;
  vec_t<int, kSortItemsPerThread> tokens_local;
  auto& logits_load_vec =
      reshape<kSortItemsPerThread / kItemsPerThread, kItemsPerThread>(logits_local);
  auto& tokens_local_vec =
      reshape<kSortItemsPerThread / kItemsPerThread, kItemsPerThread>(tokens_local);

  int load_idx = 0;
  int sort_idx = 0;

  int need_sort = 0;
  for (; icol < (vocab_size + kLoadDataTheads * kItemsPerThread - 1) /
                    (kLoadDataTheads * kItemsPerThread) * kLoadDataTheads * kItemsPerThread;
       icol += kBlockPerBatch * kLoadDataTheads * kWarpCount * kItemsPerThread) {
    if (is_load_thread && icol < vocab_size) {
      if constexpr (kUseVec16B && kItemsPerThread == 4) {
        logits_load_vec[load_idx] = load<float, kItemsPerThread>(logits_batch + icol);
#pragma unroll
        for (int i = 0; i < kItemsPerThread; i++) {
          if (icol + i >= vocab_size) {
            logits_load_vec[load_idx][i] = kNegInf;
          }
        }
        store(output_logits_batch + icol, kNegInf, kNegInf, kNegInf, kNegInf);
      } else {
#pragma unroll
        for (int i = 0; i < kItemsPerThread; i++) {
          int idata = icol + i;
          logits_load_vec[load_idx][i] = idata >= vocab_size ? kNegInf : logits_batch[icol + i];
          if (idata < vocab_size_padded) {
            output_logits_batch[icol + i] = kNegInf;
          }
        }
      }

      // set output to float -inf first, we will update it later in stage2

      float warp_local_max_logits = 0;
#pragma unroll
      for (int i = 0; i < kItemsPerThread; i++) {
        tokens_local_vec[load_idx][i] = icol + i;
        if (logits_load_vec[load_idx][i] > warp_local_max_logits) {
          warp_local_max_logits = logits_load_vec[load_idx][i];
        }
      }
      need_sort = warp_local_max_logits >= reject_threshold_batch;
    } else if (sort_idx == 0 || icol >= vocab_size) {
#pragma unroll
      for (int i = 0; i < kItemsPerThread; i++) {
        logits_load_vec[load_idx][i] = 0;
        tokens_local_vec[load_idx][i] = -1;
      }
      need_sort = 0;
    }

    need_sort = warp_reduce_sum_xor(need_sort);
    load_idx++;

    if (load_idx == kSortItemsPerThread / kItemsPerThread) {
      if (need_sort) {
        warp_merge_sort.Sort(logits_local.data, tokens_local.data, std::greater());
      }
      load_idx = 0;
      need_sort = 0;
      sort_idx++;
    }
  }

  if (load_idx != 0) {
    warp_merge_sort.Sort(logits_local.data, tokens_local.data, std::greater());
  }

  constexpr int kBlockReduceKeepTopKTheads = kMaxTopK / kItemsPerThread;
  constexpr int kBlockReduceWarpCount = kWarpCount * kMaxTopK / kItemsPerThread / kWarpSize;  // 8
  using BlockReduceSort = cub::WarpMergeSort<float, kItemsPerThread, 32, int>;
  __shared__ typename BlockReduceSort::TempStorage block_reduce_temp_storage[kBlockReduceWarpCount];
  BlockReduceSort block_reduce_sort(block_reduce_temp_storage[iwarp]);

  if (!is_load_thread) {
#pragma unroll
    for (int i = 0; i < kSortItemsPerThread; i++) {
      warp_reduce_logits[i * kWarpCount * kKeepTopKTheads + iwarp * kKeepTopKTheads + ilane] =
          logits_local[i];
      warp_reduce_tokens[i * kWarpCount * kKeepTopKTheads + iwarp * kKeepTopKTheads + ilane] =
          tokens_local[i];
    }
  }

  __syncthreads();

  if (iwarp < kBlockReduceWarpCount) {
    float warp_local_max_logits = 0;

    logits_load_vec[0] =
        load<float, kItemsPerThread>(&warp_reduce_logits[threadIdx.x * kItemsPerThread]);
    tokens_local_vec[0] =
        load<int, kItemsPerThread>(&warp_reduce_tokens[threadIdx.x * kItemsPerThread]);

#pragma unroll
    for (int i = 0; i < kSortItemsPerThread; i++) {
      if (logits_local[i] > warp_local_max_logits) {
        warp_local_max_logits = logits_local[i];
      }
    }

    need_sort = warp_local_max_logits >= reject_threshold_batch;
    need_sort = warp_reduce_sum_xor(need_sort);

    // if (need_sort) { // modified: we do sort for all warps
    block_reduce_sort.Sort(logits_load_vec[0].data, tokens_local_vec[0].data, std::greater());
    // }
  }
  __syncthreads();
  if (iwarp < kBlockReduceWarpCount && ilane < kBlockReduceKeepTopKTheads) {
    store(&warp_reduce_logits[(iwarp * kBlockReduceKeepTopKTheads + ilane) * kItemsPerThread],
          logits_load_vec[0]);
    store(&warp_reduce_tokens[(iwarp * kBlockReduceKeepTopKTheads + ilane) * kItemsPerThread],
          tokens_local_vec[0]);
  }
  __syncthreads();

  constexpr int kBlockReduceWarpCount2 =
      kBlockReduceWarpCount * kMaxTopK / kItemsPerThread / kWarpSize;  // 2
  if (iwarp < kBlockReduceWarpCount2) {
    int ismem_load_store = threadIdx.x * kItemsPerThread;
    logits_load_vec[0] = load<float, kItemsPerThread>(&warp_reduce_logits[ismem_load_store]);
    tokens_local_vec[0] = load<int, kItemsPerThread>(&warp_reduce_tokens[ismem_load_store]);
    block_reduce_sort.Sort(logits_load_vec[0].data, tokens_local_vec[0].data, std::greater());
  }
  __syncthreads();
  if (iwarp < kBlockReduceWarpCount2 && ilane < kBlockReduceKeepTopKTheads) {
    store(&warp_reduce_logits[(iwarp * kBlockReduceKeepTopKTheads + ilane) * kItemsPerThread],
          logits_load_vec[0]);
    store(&warp_reduce_tokens[(iwarp * kBlockReduceKeepTopKTheads + ilane) * kItemsPerThread],
          tokens_local_vec[0]);
  }

  __syncthreads();

  if (iwarp == 0) {
    int ismem_load_store = threadIdx.x * kItemsPerThread;
    if (threadIdx.x < kBlockReduceWarpCount2 * kMaxTopK / kItemsPerThread) {
      logits_load_vec[0] = load<float, kItemsPerThread>(&warp_reduce_logits[ismem_load_store]);
      tokens_local_vec[0] = load<int, kItemsPerThread>(&warp_reduce_tokens[ismem_load_store]);
    } else {
#pragma unroll
      for (int i = 0; i < kItemsPerThread; ++i) {
        logits_load_vec[0][i] = -1.f;
        tokens_local_vec[0][i] = -1;
      }
    }

    block_reduce_sort.Sort(logits_load_vec[0].data, tokens_local_vec[0].data, std::greater());

    if (ilane < kBlockReduceKeepTopKTheads) {
      store(mid_logits_batch + blockIdx.x * kMaxTopK + ilane * kItemsPerThread, logits_load_vec[0]);
      store(mid_tokens_batch + blockIdx.x * kMaxTopK + ilane * kItemsPerThread,
            tokens_local_vec[0]);
    }
  }
}

template <typename KType, int kThreadsPerBlock, int kBlockPerBatch, int kItemsPerThread,
          int kSortItemsPerThread, int kMaxTopK>
__global__ void topk_stage2(float* output_logits, int* out, float* middle_logits,
                            int* middle_tokens, KType* topk, int topk_val, int vocab_size) {
  constexpr int kWarpSize = 32;
  constexpr int kKeepTopKTheads = kMaxTopK / kSortItemsPerThread;
  using BlockRadixSort = cub::BlockRadixSort<float, kThreadsPerBlock, kSortItemsPerThread, int>;
  __shared__ typename BlockRadixSort::TempStorage temp_storage;

  int ibatch = blockIdx.x;
  int iwarp = threadIdx.x / kWarpSize;
  int ilane = threadIdx.x % kWarpSize;
  int icol = threadIdx.x * kItemsPerThread;

  auto* middle_logits_batch = middle_logits + ibatch * kBlockPerBatch * kMaxTopK;
  auto* middle_tokens_batch = middle_tokens + ibatch * kBlockPerBatch * kMaxTopK;
  int topk_batch = topk ? int(topk[ibatch]) : topk_val;

  vec_t<float, kSortItemsPerThread> logits_local;
  vec_t<int, kSortItemsPerThread> tokens_local;

  logits_local = load<float, kItemsPerThread>(middle_logits_batch + icol);
  tokens_local = load<int, kItemsPerThread>(middle_tokens_batch + icol);

  BlockRadixSort(temp_storage).SortDescending(logits_local.data, tokens_local.data);

  if (iwarp == 0) {
    if (ilane < kKeepTopKTheads) {
#pragma unroll
      for (int ik = 0; ik < kSortItemsPerThread; ik++) {
        int real_idx = ilane * kSortItemsPerThread + ik;
        if (real_idx < topk_batch) {
          output_logits[ibatch * vocab_size + tokens_local[ik]] = logits_local[ik];
        }
      }
    }
  }
}
}  // namespace kernels

void topk_mask_logits_async(void* output_logits, void* out, void* middle_logits,
                            void* middle_tokens, void* logits, void* topk, int topk_val,
                            void* reject_threshold, float reject_threshold_val, int batch_size,
                            int vocab_size, int vocab_size_padded, int int_bytes,
                            cudaStream_t stream) {
  // Fixed size for current implementation
  constexpr int kThreadsPerBlock = 1024;
  constexpr int kItemsPerThread = 4;
  constexpr int kWarpSize = 32;
  constexpr int kBlockPerBatch = 8;
  constexpr int kSortItemsPerThread = 4;
  constexpr int kMaxTopK = 32;

  dim3 block1(kThreadsPerBlock);
  dim3 grid1(kBlockPerBatch, batch_size);

  constexpr int topk_stage1_smem_size = kThreadsPerBlock / kWarpSize * kMaxTopK * 2 * sizeof(float);

  if (int_bytes == 4) {
    using KType = int;
    // stage 1 will load and sort topk logits in every warp, and do block-level topk and store final
    // 32 floats per block for each batch
    if (vocab_size_padded % 4 == 0) {
      constexpr bool kUseVec16B = true;
      kernels::topk_stage1<KType, kThreadsPerBlock, kBlockPerBatch, kItemsPerThread,
                           kSortItemsPerThread, kMaxTopK, kUseVec16B>
          <<<grid1, block1, topk_stage1_smem_size, stream>>>(
              reinterpret_cast<float*>(output_logits), reinterpret_cast<float*>(middle_logits),
              reinterpret_cast<int*>(middle_tokens), reinterpret_cast<float*>(logits),
              reinterpret_cast<KType*>(topk), reinterpret_cast<float*>(reject_threshold),
              reject_threshold_val, vocab_size, vocab_size_padded);
    } else {
      constexpr bool kUseVec16B = false;
      kernels::topk_stage1<KType, kThreadsPerBlock, kBlockPerBatch, kItemsPerThread,
                           kSortItemsPerThread, kMaxTopK, kUseVec16B>
          <<<grid1, block1, topk_stage1_smem_size, stream>>>(
              reinterpret_cast<float*>(output_logits), reinterpret_cast<float*>(middle_logits),
              reinterpret_cast<int*>(middle_tokens), reinterpret_cast<float*>(logits),
              reinterpret_cast<KType*>(topk), reinterpret_cast<float*>(reject_threshold),
              reject_threshold_val, vocab_size, vocab_size_padded);
    }

    // stage 2 will load 32*num_block floats and do topk again, and store final k value for each
    // batch
    constexpr int kSortItemsPerThread2 = 4;
    constexpr int kThreadsPerBlock2 = kBlockPerBatch * kMaxTopK / kSortItemsPerThread2;
    dim3 block2(kThreadsPerBlock2);
    dim3 grid2(batch_size);
    kernels::topk_stage2<KType, kThreadsPerBlock2, kBlockPerBatch, kItemsPerThread,
                         kSortItemsPerThread2, kMaxTopK><<<grid2, block2, 0, stream>>>(
        reinterpret_cast<float*>(output_logits), reinterpret_cast<int*>(out),
        reinterpret_cast<float*>(middle_logits), reinterpret_cast<int*>(middle_tokens),
        reinterpret_cast<KType*>(topk), topk_val, vocab_size);
  } else if (int_bytes == 8) {
    using KType = int64_t;
    // stage 1 will load and sort topk logits in every warp, and do block-level topk and store final
    // 32 floats per block for each batch
    if (vocab_size_padded % 4 == 0) {
      constexpr bool kUseVec16B = true;
      kernels::topk_stage1<KType, kThreadsPerBlock, kBlockPerBatch, kItemsPerThread,
                           kSortItemsPerThread, kMaxTopK, kUseVec16B>
          <<<grid1, block1, topk_stage1_smem_size, stream>>>(
              reinterpret_cast<float*>(output_logits), reinterpret_cast<float*>(middle_logits),
              reinterpret_cast<int*>(middle_tokens), reinterpret_cast<float*>(logits),
              reinterpret_cast<KType*>(topk), reinterpret_cast<float*>(reject_threshold),
              reject_threshold_val, vocab_size, vocab_size_padded);
    } else {
      constexpr bool kUseVec16B = false;
      kernels::topk_stage1<KType, kThreadsPerBlock, kBlockPerBatch, kItemsPerThread,
                           kSortItemsPerThread, kMaxTopK, kUseVec16B>
          <<<grid1, block1, topk_stage1_smem_size, stream>>>(
              reinterpret_cast<float*>(output_logits), reinterpret_cast<float*>(middle_logits),
              reinterpret_cast<int*>(middle_tokens), reinterpret_cast<float*>(logits),
              reinterpret_cast<KType*>(topk), reinterpret_cast<float*>(reject_threshold),
              reject_threshold_val, vocab_size, vocab_size_padded);
    }

    // stage 2 will load 32*num_block floats and do topk again, and store final k value for each
    // batch
    constexpr int kSortItemsPerThread2 = 4;
    constexpr int kThreadsPerBlock2 = kBlockPerBatch * kMaxTopK / kSortItemsPerThread2;
    dim3 block2(kThreadsPerBlock2);
    dim3 grid2(batch_size);
    kernels::topk_stage2<KType, kThreadsPerBlock2, kBlockPerBatch, kItemsPerThread,
                         kSortItemsPerThread2, kMaxTopK><<<grid2, block2, 0, stream>>>(
        reinterpret_cast<float*>(output_logits), reinterpret_cast<int*>(out),
        reinterpret_cast<float*>(middle_logits), reinterpret_cast<int*>(middle_tokens),
        reinterpret_cast<KType*>(topk), topk_val, vocab_size);
  }
}

}  // namespace sampler
}  // namespace hpc
