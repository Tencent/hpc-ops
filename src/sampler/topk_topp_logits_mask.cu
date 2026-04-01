// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <curand_kernel.h>
#include <stdio.h>

#include <cub/cub.cuh>
#include <limits>
#include <string>

#include "src/sampler/sampler.h"
#include "src/utils/utils.cuh"
namespace hpc {
namespace sampler {
namespace kernels {

template <typename T, typename DType>
__device__ __forceinline__ void fill_vec(T& vec, DType val) {
#pragma unroll
  for (int i = 0; i < size(vec); i++) {
    vec[i] = val;
  }
}

template <typename KType, int kThreadsPerBlock, int kBlockPerBatch, int kItemsPerThread,
          int kSortItemsPerThread, int kMaxTopK, bool kUseVec16B>
__global__ void topk_topp_stage1(float* output_logits, float* mid_logits, int* mid_tokens,
                                 const float* logits, KType* topk, float* reject_threshold,
                                 float reject_threshold_val, int vocab_size,
                                 int vocab_size_padded) {
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

template <typename KType, typename PType, int kThreadsPerBlock, int kBlockPerBatch,
          int kItemsPerThread, int kSortItemsPerThread, int kMaxTopK, int kEnableTopP>
__global__ void topk_topp_stage2(float* output_logits, int* out, float* middle_logits,
                                 int* middle_tokens, KType* topk, int topk_val, PType* topp,
                                 float topp_val, int vocab_size) {
  constexpr int kWarpSize = 32;
  constexpr int kKeepTopKTheads = kMaxTopK / kSortItemsPerThread;
  using BlockRadixSort = cub::BlockRadixSort<float, kThreadsPerBlock, kSortItemsPerThread, int>;
  __shared__ typename BlockRadixSort::TempStorage temp_storage;
  __shared__ float thread_probs_sum[kKeepTopKTheads];
  __shared__ float max_logits;

  int ibatch = blockIdx.x;
  int iwarp = threadIdx.x / kWarpSize;
  int ilane = threadIdx.x % kWarpSize;
  int icol = threadIdx.x * kItemsPerThread;

  auto* middle_logits_batch = middle_logits + ibatch * kBlockPerBatch * kMaxTopK;
  auto* middle_tokens_batch = middle_tokens + ibatch * kBlockPerBatch * kMaxTopK;
  int topk_batch = topk ? int(topk[ibatch]) : topk_val;
  float topp_batch = topp ? float(topp[ibatch]) : topp_val;

  vec_t<float, kSortItemsPerThread> logits_local;
  vec_t<int, kSortItemsPerThread> tokens_local;
  vec_t<float, kSortItemsPerThread> probs_local;

  logits_local = load<float, kItemsPerThread>(middle_logits_batch + icol);
  tokens_local = load<int, kItemsPerThread>(middle_tokens_batch + icol);

  BlockRadixSort(temp_storage).SortDescending(logits_local.data, tokens_local.data);
  float partial_sum = 0.f;

  if constexpr (kEnableTopP) {
    vec_t<float, kSortItemsPerThread> exp_logits_local;
    // do softmax first
    if (iwarp == 0) {
      static_assert(kMaxTopK <= kWarpSize * kItemsPerThread,
                    "otherwise should take topk in multi warps");
      if (threadIdx.x == 0) {
        max_logits = logits_local[0];
      }
      __syncwarp();

      float exp_sum = 0.f;
#pragma unroll
      for (int ik = 0; ik < kSortItemsPerThread; ik++) {
        int real_idx = ilane * kSortItemsPerThread + ik;
        if (real_idx < topk_batch) {
          exp_logits_local[ik] = expf_ftz(logits_local[ik] - max_logits);
          exp_sum += exp_logits_local[ik];
        }
      }
      exp_sum = warp_reduce_sum_xor(exp_sum);
      float inv_exp_sum = rcpf_ftz(exp_sum);

#pragma unroll
      for (int ik = 0; ik < kSortItemsPerThread; ik++) {
        int real_idx = ilane * kSortItemsPerThread + ik;
        if (real_idx < topk_batch) {
          // output_logits[ibatch * vocab_size + tokens_local[ik]] = logits_local[ik];
          probs_local[ik] = exp_logits_local[ik] * inv_exp_sum;
        }
      }

      // cum sum logits for top-p
#pragma unroll
      for (int ik = 0; ik < kSortItemsPerThread; ik++) {
        int real_idx = ilane * kSortItemsPerThread + ik;
        if (real_idx < topk_batch) {
          partial_sum += probs_local[ik];
        }
      }
      if (ilane < kKeepTopKTheads) {
        thread_probs_sum[ilane] = partial_sum;
      }
      __syncwarp();
      if (ilane < kKeepTopKTheads) {
        partial_sum = 0.f;
        for (int i = 0; i < ilane; i++) {
          partial_sum += thread_probs_sum[i];
        }
      }
    }
  }

  // select logits
  if (iwarp == 0) {
#pragma unroll
    for (int ik = 0; ik < kSortItemsPerThread; ik++) {
      int real_idx = ilane * kSortItemsPerThread + ik;
      if constexpr (kEnableTopP) {
        if (real_idx < topk_batch) {
          if (real_idx == 0 || partial_sum < topp_batch) {  // at least select one
            output_logits[ibatch * vocab_size + tokens_local[ik]] = logits_local[ik];
          }
          partial_sum += probs_local[ik];
        }
      } else {
        if (real_idx < topk_batch) {
          output_logits[ibatch * vocab_size + tokens_local[ik]] = logits_local[ik];
        }
      }
    }
  }
}

// output mode : 0 means output to whole logits, 1 means output to token id
template <typename KType, int kThreadsPerBlock, int kBlockPerBatch, int kItemsPerThread,
          int kMaxTopK, bool kUseVec16B, int kOutputMode = 0>
__global__ void topk_topp_blocksort_stage1(float* output_logits, float* mid_logits, int* mid_tokens,
                                           const float* logits, KType* topk,
                                           float* reject_threshold, float reject_threshold_val,
                                           int vocab_size, int vocab_size_padded,
                                           int num_load_loop) {
  static_assert(kMaxTopK % kItemsPerThread == 0);
  constexpr int kKeeperThreads =
      kMaxTopK / kItemsPerThread;  // Num of threads to keep topk elem per block
  constexpr int kLoaderThreads = kThreadsPerBlock - kKeeperThreads;
  // Each loop, all blocks' loader threads collectively consume this many elements
  constexpr float kNegInf = -std::numeric_limits<float>::infinity();

  using BlockRadixSort = cub::BlockRadixSort<float, kThreadsPerBlock, kItemsPerThread, int>;
  __shared__ typename BlockRadixSort::TempStorage temp_storage;

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int ibatch = blockIdx.y;

  bool is_keeper = tid < kKeeperThreads;
  int loader_id = tid - kKeeperThreads;  // index among loader threads (valid when !is_keeper)

  auto* logits_batch = logits + ibatch * vocab_size_padded;
  auto* output_logits_batch = output_logits + ibatch * vocab_size;
  auto* mid_logits_batch = mid_logits + ibatch * kBlockPerBatch * kMaxTopK;
  auto* mid_tokens_batch = mid_tokens + ibatch * kBlockPerBatch * kMaxTopK;

  // Each thread holds kItemsPerThread logits and their token ids
  vec_t<float, kItemsPerThread> thread_logits;
  vec_t<int, kItemsPerThread> thread_tokens;

  // Keeper threads start with -inf; loader threads will overwrite every loop
  fill_vec(thread_logits, kNegInf);
  fill_vec(thread_tokens, -1);

  for (int iloop = 0; iloop < num_load_loop; iloop++) {
    if (!is_keeper) {
      // Loader thread: compute global offset and vectorized load
      int ivec_offset =
          (iloop * kBlockPerBatch * kLoaderThreads + bid * kLoaderThreads + loader_id) *
          kItemsPerThread;

      if (ivec_offset + kItemsPerThread <= vocab_size) {
        // Fast path: all elements in range
        if constexpr (kUseVec16B) {
          thread_logits = load<float, kItemsPerThread>(logits_batch + ivec_offset);
        } else {
#pragma unroll
          for (int i = 0; i < kItemsPerThread; i++) {
            thread_logits[i] = logits_batch[ivec_offset + i];
          }
        }
#pragma unroll
        for (int i = 0; i < kItemsPerThread; i++) {
          thread_tokens[i] = ivec_offset + i;
        }
        if constexpr (kOutputMode == 0) {
          if constexpr (kUseVec16B) {
            vec_t<float, kItemsPerThread> neg_inf_vec;
            fill_vec(neg_inf_vec, kNegInf);
            store(output_logits_batch + ivec_offset, neg_inf_vec);
          } else {
#pragma unroll
            for (int i = 0; i < kItemsPerThread; i++) {
              output_logits_batch[ivec_offset + i] = kNegInf;
            }
          }
        }
      } else {
        // Boundary: load element by element
#pragma unroll
        for (int i = 0; i < kItemsPerThread; i++) {
          int idx = ivec_offset + i;
          if (idx < vocab_size) {
            thread_logits[i] = logits_batch[idx];
            thread_tokens[i] = idx;
            if constexpr (kOutputMode == 0) {
              output_logits_batch[idx] = kNegInf;
            }
          } else {
            thread_logits[i] = kNegInf;
            thread_tokens[i] = -1;
          }
        }
      }
    }
    // Block-level radix sort descending on (logits, tokens)
    __syncthreads();
    BlockRadixSort(temp_storage).SortDescending(thread_logits.data, thread_tokens.data);
    __syncthreads();
  }

  // Write out: only keeper threads write results to mid buffer
  if (is_keeper) {
    store(mid_logits_batch + bid * kMaxTopK + tid * kItemsPerThread, thread_logits);
    store(mid_tokens_batch + bid * kMaxTopK + tid * kItemsPerThread, thread_tokens);
  }
}

}  // namespace kernels

template <typename KType, typename PType, bool kEnableTopP>
void launch_topk_topp(float* output_logits, int* out, float* mid_logits, int* mid_tokens,
                      const float* logits, KType* topk, int topk_val, PType* topp, float topp_val,
                      float* reject_threshold, float reject_threshold_val, int batch_size,
                      int vocab_size, int vocab_size_padded, int max_topk_val,
                      cudaStream_t stream) {
  constexpr int kThreadsPerBlock = 1024;
  constexpr int kItemsPerThread = 4;
  constexpr int kBlockPerBatch = 8;

  if (max_topk_val <= 32) {
    // ---- Path A: topk <= 32, warp merge sort stage1 ----
    constexpr int kWarpSize = 32;
    constexpr int kMaxTopK = 32;
    constexpr int topk_stage1_smem_size =
        kThreadsPerBlock / kWarpSize * kMaxTopK * 2 * sizeof(float);

    constexpr int kSortItemsPerThread = 4;

    dim3 grid1(kBlockPerBatch, batch_size);
    dim3 block1(kThreadsPerBlock);

    if (vocab_size_padded % 4 == 0) {
      constexpr bool kUseVec16B = true;
      kernels::topk_topp_stage1<KType, kThreadsPerBlock, kBlockPerBatch, kItemsPerThread,
                                kSortItemsPerThread, kMaxTopK, kUseVec16B>
          <<<grid1, block1, topk_stage1_smem_size, stream>>>(
              output_logits, mid_logits, mid_tokens, logits, topk, reject_threshold,
              reject_threshold_val, vocab_size, vocab_size_padded);
    } else {
      constexpr bool kUseVec16B = false;
      kernels::topk_topp_stage1<KType, kThreadsPerBlock, kBlockPerBatch, kItemsPerThread,
                                kSortItemsPerThread, kMaxTopK, kUseVec16B>
          <<<grid1, block1, topk_stage1_smem_size, stream>>>(
              output_logits, mid_logits, mid_tokens, logits, topk, reject_threshold,
              reject_threshold_val, vocab_size, vocab_size_padded);
    }

    // stage 2
    constexpr int kSortItemsPerThread2 = 4;
    constexpr int kThreadsPerBlock2 = kBlockPerBatch * kMaxTopK / kSortItemsPerThread2;
    dim3 grid2(batch_size);
    dim3 block2(kThreadsPerBlock2);
    kernels::topk_topp_stage2<KType, PType, kThreadsPerBlock2, kBlockPerBatch, kItemsPerThread,
                              kSortItemsPerThread2, kMaxTopK, kEnableTopP>
        <<<grid2, block2, 0, stream>>>(output_logits, out, mid_logits, mid_tokens, topk, topk_val,
                                       topp, topp_val, vocab_size);
  } else if (max_topk_val <= 64) {
    // ---- Path B: 32 < topk <= 64, block radix sort stage1 ----
    constexpr int kMaxTopK = 64;
    constexpr int kKeeperThreads = kMaxTopK / kItemsPerThread;
    constexpr int kLoaderThreads = kThreadsPerBlock - kKeeperThreads;
    constexpr int kElemtsPerLoadLoop = kBlockPerBatch * kLoaderThreads * kItemsPerThread;

    int num_load_loop = (vocab_size + kElemtsPerLoadLoop - 1) / kElemtsPerLoadLoop;

    dim3 grid1(kBlockPerBatch, batch_size);
    dim3 block1(kThreadsPerBlock);

    if (vocab_size_padded % 4 == 0) {
      constexpr bool kUseVec16B = true;
      kernels::topk_topp_blocksort_stage1<KType, kThreadsPerBlock, kBlockPerBatch, kItemsPerThread,
                                          kMaxTopK, kUseVec16B><<<grid1, block1, 0, stream>>>(
          output_logits, mid_logits, mid_tokens, logits, topk, reject_threshold,
          reject_threshold_val, vocab_size, vocab_size_padded, num_load_loop);
    } else {
      constexpr bool kUseVec16B = false;
      kernels::topk_topp_blocksort_stage1<KType, kThreadsPerBlock, kBlockPerBatch, kItemsPerThread,
                                          kMaxTopK, kUseVec16B><<<grid1, block1, 0, stream>>>(
          output_logits, mid_logits, mid_tokens, logits, topk, reject_threshold,
          reject_threshold_val, vocab_size, vocab_size_padded, num_load_loop);
    }

    // stage 2 with kMaxTopK=64
    constexpr int kSortItemsPerThread2 = 4;
    constexpr int kThreadsPerBlock2 = kBlockPerBatch * kMaxTopK / kSortItemsPerThread2;
    dim3 grid2(batch_size);
    dim3 block2(kThreadsPerBlock2);
    kernels::topk_topp_stage2<KType, PType, kThreadsPerBlock2, kBlockPerBatch, kItemsPerThread,
                              kSortItemsPerThread2, kMaxTopK, kEnableTopP>
        <<<grid2, block2, 0, stream>>>(output_logits, out, mid_logits, mid_tokens, topk, topk_val,
                                       topp, topp_val, vocab_size);
  } else {
    throw std::invalid_argument("max_topk_val=" + std::to_string(max_topk_val) +
                                " is not supported for now, contact hpc team.");
  }
}

void topk_topp_mask_logits_async(void* output_logits, void* out, void* middle_logits,
                                 void* middle_tokens, void* logits, void* topk, int topk_val,
                                 void* topp, float topp_val, void* reject_threshold,
                                 float reject_threshold_val, int batch_size, int vocab_size,
                                 int vocab_size_padded, int int_bytes, int max_topk_val,
                                 cudaStream_t stream) {
  bool enable_topp = !(topp == nullptr && (topp_val <= 0 || topp_val > 1));

  if (!enable_topp) {
    if (int_bytes == 4) {
      launch_topk_topp<int, float, false>(
          reinterpret_cast<float*>(output_logits), reinterpret_cast<int*>(out),
          reinterpret_cast<float*>(middle_logits), reinterpret_cast<int*>(middle_tokens),
          reinterpret_cast<const float*>(logits), reinterpret_cast<int*>(topk), topk_val,
          reinterpret_cast<float*>(topp), topp_val, reinterpret_cast<float*>(reject_threshold),
          reject_threshold_val, batch_size, vocab_size, vocab_size_padded, max_topk_val, stream);
    } else if (int_bytes == 8) {
      launch_topk_topp<int64_t, float, false>(
          reinterpret_cast<float*>(output_logits), reinterpret_cast<int*>(out),
          reinterpret_cast<float*>(middle_logits), reinterpret_cast<int*>(middle_tokens),
          reinterpret_cast<const float*>(logits), reinterpret_cast<int64_t*>(topk), topk_val,
          reinterpret_cast<float*>(topp), topp_val, reinterpret_cast<float*>(reject_threshold),
          reject_threshold_val, batch_size, vocab_size, vocab_size_padded, max_topk_val, stream);
    }
  } else {
    if (int_bytes == 4) {
      launch_topk_topp<int, float, true>(
          reinterpret_cast<float*>(output_logits), reinterpret_cast<int*>(out),
          reinterpret_cast<float*>(middle_logits), reinterpret_cast<int*>(middle_tokens),
          reinterpret_cast<const float*>(logits), reinterpret_cast<int*>(topk), topk_val,
          reinterpret_cast<float*>(topp), topp_val, reinterpret_cast<float*>(reject_threshold),
          reject_threshold_val, batch_size, vocab_size, vocab_size_padded, max_topk_val, stream);
    } else if (int_bytes == 8) {
      launch_topk_topp<int64_t, float, true>(
          reinterpret_cast<float*>(output_logits), reinterpret_cast<int*>(out),
          reinterpret_cast<float*>(middle_logits), reinterpret_cast<int*>(middle_tokens),
          reinterpret_cast<const float*>(logits), reinterpret_cast<int64_t*>(topk), topk_val,
          reinterpret_cast<float*>(topp), topp_val, reinterpret_cast<float*>(reject_threshold),
          reject_threshold_val, batch_size, vocab_size, vocab_size_padded, max_topk_val, stream);
    }
  }
}

}  // namespace sampler
}  // namespace hpc
