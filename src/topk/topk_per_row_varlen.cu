// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <math.h>
#include <stdio.h>

#include <algorithm>
#include <cub/cub.cuh>  // NOLINT
#include <iostream>
#include <limits>

#include "src/topk/topk.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace topk {

namespace kernels {

template <int kStep>
__forceinline__ __device__ uint32_t extract_bin(float x) {
  if constexpr (kStep == 0) {
    __half hx = __float2half(x);
    uint16_t bits = __half_as_ushort(hx);
    bits = (bits & 0x8000) ? bits : ~bits & 0x7fff;
    return bits >> 5;
  } else {
    uint32_t bits = __float_as_uint(x);
    bits = (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;

    if constexpr (kStep == 1) {
      return bits >> 21;
    } else if constexpr (kStep == 2) {
      return (bits >> 10) & 0x7ff;
    } else if constexpr (kStep == 3) {
      return bits & 0x3ff;
    }
  }
}

template <int kShift>
__forceinline__ __device__ bool partial_match(float x, uint32_t pattern) {
  if constexpr (kShift == 0) {
    return true;
  }
  uint32_t bits = __float_as_uint(x);
  bits = (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;
  return (bits ^ pattern) >> kShift == 0;
}

template <int kStep, int kShift>
__forceinline__ __device__ void distribute_to_bins(float logit, const uint32_t& logit_pattern,
                                                   int* smem_histogram) {
  if (partial_match<kShift>(logit, logit_pattern)) {
    uint32_t bin = extract_bin<kStep>(logit);
    atomicAdd(&smem_histogram[bin], 1);
  }
}

template <int kStep, int kShift, int kTopK, int kFinalBinItems, bool kDeterministic,
          typename SmemFinalSort>
__forceinline__ __device__ void process_bins(float logit, const uint32_t& logit_pattern, int ipos,
                                             int threshold_bin, int* smem_found_topk,
                                             int* smem_histogram, SmemFinalSort& smem_sort,
                                             int final_count, int* smem_final_actual_count,
                                             int* smem_indices) {
  if (partial_match<kShift>(logit, logit_pattern)) {
    uint32_t bin = extract_bin<kStep>(logit);

    if (bin < threshold_bin) {
      int smem_idx = atomicAdd(&smem_found_topk[0], 1);
      smem_indices[smem_idx] = ipos;
    }

    if constexpr (kStep < 3) {
      if constexpr (kDeterministic) {
        if (bin == threshold_bin && final_count <= kFinalBinItems) {
          int smem_idx = atomicAdd(&smem_final_actual_count[0], 1);
          smem_sort.items.logits[smem_idx] = logit;
          smem_sort.items.indices[smem_idx] = ipos;
        }
      } else {
        if (bin == threshold_bin) {
          int smem_idx = atomicAdd(&smem_final_actual_count[0], 1);
          if (smem_idx < kFinalBinItems) {
            smem_sort.items.logits[smem_idx] = logit;
            smem_sort.items.indices[smem_idx] = ipos;
          }
        }
      }
    } else {
      if (bin == threshold_bin) {
        int smem_idx = atomicAdd(&smem_histogram[bin], 1);

        if (smem_idx < kTopK) {
          smem_indices[smem_idx] = ipos;
        }
      }
    }
  }
}

template <int kStep, int kTopK, int kBins, int kFinalBinItems, bool kDeterministic,
          int kThreadsPerBlock, int kItemPerLoad, typename BinPrefixSumScan, typename SmemFinalSort>
__device__ bool process_histogram_step(int* smem_indices, const float* logits_row,
                                       int* smem_histogram, int* smem_found_topk,
                                       int* smem_threshold_bin, BinPrefixSumScan& scan,
                                       int* smem_final_count, int* smem_final_actual_count,
                                       SmemFinalSort& smem_sort, uint32_t& logit_pattern,
                                       int& threshold_bin, int seqlen, int idx) {
  for (int i = idx; i < kBins; i += kThreadsPerBlock) {
    smem_histogram[i] = 0;
  }
  __syncthreads();

  // Update pattern
  constexpr auto kShift = kStep < 2 ? 0 : kStep == 2 ? 21 : 10;
  if constexpr (kStep == 2) {
    logit_pattern = static_cast<uint32_t>(threshold_bin & 0x7ff) << kShift;
  } else if constexpr (kStep == 3) {
    logit_pattern |= static_cast<uint32_t>(threshold_bin & 0x7ff) << kShift;
  }

  for (int icol = idx * kItemPerLoad; icol < seqlen; icol += kThreadsPerBlock * kItemPerLoad) {
    auto logits = load<float, kItemPerLoad>(logits_row + icol);
#pragma unroll
    for (int i = 0; i < kItemPerLoad; i++) {
      int ipos = icol + i;
      if (ipos < seqlen) {
        distribute_to_bins<kStep, kShift>(logits[i], logit_pattern, smem_histogram);
      }
    }
  }
  __syncthreads();

  int last_sum = smem_found_topk[0];

  for (int ibin = idx; ibin < kBins; ibin += kThreadsPerBlock) {
    int bin_count = smem_histogram[ibin];
    __syncthreads();

    int prefix_sum = 0;
    int sum = 0;

    scan.ExclusiveSum(bin_count, prefix_sum, sum);

    prefix_sum += last_sum;
    sum += last_sum;

    smem_histogram[ibin] = prefix_sum;
    __syncthreads();

    // find the bin threshold which topk in.
    bool found_threshold = false;
    if (prefix_sum < kTopK) {
      int next_prefix_sum = (idx == kThreadsPerBlock - 1) ? sum : smem_histogram[ibin + 1];
      if (next_prefix_sum >= kTopK) {
        smem_threshold_bin[0] = ibin;
        smem_final_count[0] = next_prefix_sum - prefix_sum;
        found_threshold = true;
      }
    }

    if (__syncthreads_or(found_threshold)) {
      break;
    }

    last_sum = sum;
  }

  __syncthreads();

  threshold_bin = smem_threshold_bin[0];
  int final_count = smem_final_count[0];

  // perform final sort in threshold bin
  for (int icol = idx * kItemPerLoad; icol < seqlen; icol += kThreadsPerBlock * kItemPerLoad) {
    auto logits = load<float, kItemPerLoad>(logits_row + icol);
#pragma unroll
    for (int i = 0; i < kItemPerLoad; i++) {
      int ipos = icol + i;
      if (ipos < seqlen) {
        process_bins<kStep, kShift, kTopK, kFinalBinItems, kDeterministic>(
            logits[i], logit_pattern, ipos, threshold_bin, smem_found_topk, smem_histogram,
            smem_sort, final_count, smem_final_actual_count, smem_indices);
      }
    }
  }

  __syncthreads();

  return smem_final_count[0] > kFinalBinItems;
}

template <int kTopK, int kThreadsPerBlock, int kFinalItemsPerThread, typename FinalBinSort,
          typename SmemFinalSort>
__forceinline__ __device__ void final_radix_sort(int* smem_indices, FinalBinSort& final_sort,
                                                 SmemFinalSort& smem_sort, int ibase, int idx,
                                                 int final_actual_count) {
  vec_t<float, kFinalItemsPerThread> final_logits;
  vec_t<int, kFinalItemsPerThread> final_indices;

#pragma unroll
  for (int i = 0; i < kFinalItemsPerThread; i++) {
    final_logits[i] = std::numeric_limits<float>::lowest();
  }

#pragma unroll
  for (int i = 0; i < kFinalItemsPerThread; i++) {
    int final_idx = i * kThreadsPerBlock + idx;
    if (final_idx < final_actual_count) {
      final_logits[i] = smem_sort.items.logits[final_idx];
      final_indices[i] = smem_sort.items.indices[final_idx];
    }
  }
  __syncthreads();

  final_sort.SortDescendingBlockedToStriped(final_logits.data, final_indices.data);

#pragma unroll
  for (int i = 0; i < kFinalItemsPerThread; i++) {
    int ifinal = i * kThreadsPerBlock + idx;
    int smem_idx = ibase + ifinal;
    if (smem_idx < kTopK) {
      smem_indices[smem_idx] = final_indices[i];
    }
  }
}

template <int kTopK, int kThreadsPerBlock, typename SmemFinalSort>
__forceinline__ __device__ void final_insert_sort(int* smem_indices, SmemFinalSort& smem_sort,
                                                  int ibase, int idx, int final_actual_count) {
  for (int i = idx; i < final_actual_count; i += kThreadsPerBlock) {
    int iout = 0;
    auto logit = smem_sort.items.logits[i];
    for (int j = 0; j < final_actual_count; j++) {
      auto logit2 = smem_sort.items.logits[j];
      if (logit < logit2 || (logit == logit2 && i < j)) {
        iout++;
      }
    }

    if (iout + ibase < kTopK) {
      smem_indices[iout + ibase] = smem_sort.items.indices[i];
    }
  }
}

template <int kTopK, int kFinalBinItems, int kThreadsPerBlock, bool kDeterministic>
__global__ void topk_per_row_varlen_kernel(int* output_indices, const float* logits_ptr,
                                           const int* cu_seqlens_q_ptr, const int* seqlens_kv_ptr,
                                           cutlass::FastDivmod compress_ratio_divider,
                                           const int num_batch, const int row_stride) {
  constexpr int kItemPerLoad = 4;
  constexpr int kBins = 2048;
  constexpr int kFinalItemsPerThread = kFinalBinItems / kThreadsPerBlock;

  struct Items {
    int indices[kFinalBinItems];
    float logits[kFinalBinItems];
  };

  using FinalBinSort = cub::BlockRadixSort<float, kThreadsPerBlock, kFinalItemsPerThread, int>;
  using BinPrefixSumScan = cub::BlockScan<int, kThreadsPerBlock>;

  __shared__ typename BinPrefixSumScan::TempStorage smem_scan;

  __shared__ union {
    Items items;
    typename FinalBinSort::TempStorage final_sort;
  } smem_sort;

  __shared__ int smem_histogram[kBins];
  __shared__ __align__(16) int smem_indices[kTopK];
  __shared__ int smem_threshold_bin[1];
  __shared__ int smem_final_count[1];
  __shared__ int smem_final_actual_count[1];
  __shared__ int smem_found_topk[1];

  BinPrefixSumScan scan(smem_scan);
  FinalBinSort final_sort(smem_sort.final_sort);

  int idx = threadIdx.x;
  int irow = blockIdx.x;

  int ibatch = 0;
  int itoken_in_batch = irow;

  for (int i = 1; i < num_batch + 1; i++) {
    int cu_seqlenq = cu_seqlens_q_ptr[i];
    if (irow < cu_seqlenq) {
      ibatch = i - 1;
      itoken_in_batch = irow - cu_seqlens_q_ptr[ibatch];
      break;
    }
  }

  int seqlen_q = cu_seqlens_q_ptr[ibatch + 1] - cu_seqlens_q_ptr[ibatch];
  int seqlen_kv = seqlens_kv_ptr[ibatch];
  int seqlen_kvcache = seqlen_kv - seqlen_q;

  int seqlen = 0;
  int compress_remain = 0;
  compress_ratio_divider(seqlen, compress_remain, seqlen_kvcache + itoken_in_batch + 1);

  const auto* logits_row = logits_ptr + irow * static_cast<int64_t>(row_stride);
  auto* output_indices_row = output_indices + static_cast<int64_t>(irow) * kTopK;

  if (seqlen < kTopK) {
    vec_t<int, kItemPerLoad> indices;
#pragma unroll
    for (int icol = idx * kItemPerLoad; icol < kTopK; icol += kThreadsPerBlock * kItemPerLoad) {
#pragma unroll
      for (int i = 0; i < kItemPerLoad; i++) {
        int ipos = icol + i;
        if (ipos >= seqlen) {
          indices[i] = -1;
        } else {
          indices[i] = ipos;
        }
      }
      store(output_indices_row + icol, indices);
    }
    return;
  }

  if (idx == 0) {
    smem_final_actual_count[0] = 0;
    smem_found_topk[0] = 0;
  }

  __syncthreads();
  int threshold_bin = -1;
  uint32_t logit_pattern = 0;

  // Step 0: Process first 11 bits of half representation
  constexpr int kStepFp16 = 0;
  bool continue_next_step = process_histogram_step<kStepFp16, kTopK, kBins, kFinalBinItems,
                                                   kDeterministic, kThreadsPerBlock, kItemPerLoad>(
      smem_indices, logits_row, smem_histogram, smem_found_topk, smem_threshold_bin, scan,
      smem_final_count, smem_final_actual_count, smem_sort, logit_pattern, threshold_bin, seqlen,
      idx);

  if constexpr (kDeterministic) {
    // Step 1: Process next 11 bits
    if (continue_next_step) {
      constexpr int kStepFp32 = 1;
      continue_next_step = process_histogram_step<kStepFp32, kTopK, kBins, kFinalBinItems,
                                                  kDeterministic, kThreadsPerBlock, kItemPerLoad>(
          smem_indices, logits_row, smem_histogram, smem_found_topk, smem_threshold_bin, scan,
          smem_final_count, smem_final_actual_count, smem_sort, logit_pattern, threshold_bin,
          seqlen, idx);
    }

    // Step 2: Process next 11 bits
    if (continue_next_step) {
      constexpr int kStepFp32 = 2;
      continue_next_step = process_histogram_step<kStepFp32, kTopK, kBins, kFinalBinItems,
                                                  kDeterministic, kThreadsPerBlock, kItemPerLoad>(
          smem_indices, logits_row, smem_histogram, smem_found_topk, smem_threshold_bin, scan,
          smem_final_count, smem_final_actual_count, smem_sort, logit_pattern, threshold_bin,
          seqlen, idx);
    }

    // Step 2: Process last 10 bits
    if (continue_next_step) {
      constexpr int kStepFp32 = 3;
      continue_next_step = process_histogram_step<kStepFp32, kTopK, kBins, kFinalBinItems,
                                                  kDeterministic, kThreadsPerBlock, kItemPerLoad>(
          smem_indices, logits_row, smem_histogram, smem_found_topk, smem_threshold_bin, scan,
          smem_final_count, smem_final_actual_count, smem_sort, logit_pattern, threshold_bin,
          seqlen, idx);
    }
  } else {
    continue_next_step = false;
  }

  if (!continue_next_step) {
    int final_actual_count = smem_final_actual_count[0];
    if (final_actual_count <= 256) {
      final_insert_sort<kTopK, kThreadsPerBlock>(smem_indices, smem_sort, smem_found_topk[0], idx,
                                                 final_actual_count);
    } else {
      final_radix_sort<kTopK, kThreadsPerBlock, kFinalItemsPerThread>(
          smem_indices, final_sort, smem_sort, smem_found_topk[0], idx, final_actual_count);
    }
  }

  __syncthreads();
  // store result
#pragma unroll
  for (int icol = idx * kItemPerLoad; icol < kTopK; icol += kItemPerLoad * kThreadsPerBlock) {
    auto result = load<int, kItemPerLoad>(smem_indices + icol);
    store(output_indices_row + icol, result);
  }
}

}  // namespace kernels

bool topk_per_row_varlen_async(int* topk_indices, const float* logits_ptr,
                               const int* cu_seqlens_q_ptr, const int* seqlens_kv_ptr, int topk,
                               int compress_ratio, int num_batch, int num_rows, int row_stride,
                               bool deterministic, cudaStream_t stream) {
  constexpr int kTopK = 512;
  constexpr int kThreadsPerBlock = 256;
  constexpr int kFinalBinItems = 2048;

  if (topk != kTopK) {
    std::cout << "topk_per_row_varlen_async only support topk 512, not support topk: " << topk
              << std::endl;
    return false;
  }

  dim3 block(kThreadsPerBlock);
  dim3 grid(num_rows);

  cutlass::FastDivmod compress_ratio_divider(compress_ratio);

  if (deterministic) {
    constexpr bool kDeterministic = true;
    auto kernel = kernels::topk_per_row_varlen_kernel<kTopK, kFinalBinItems, kThreadsPerBlock,
                                                      kDeterministic>;
    kernel<<<grid, block, 0, stream>>>(topk_indices, logits_ptr, cu_seqlens_q_ptr, seqlens_kv_ptr,
                                       compress_ratio_divider, num_batch, row_stride);
  } else {
    constexpr bool kDeterministic = false;
    auto kernel = kernels::topk_per_row_varlen_kernel<kTopK, kFinalBinItems, kThreadsPerBlock,
                                                      kDeterministic>;
    kernel<<<grid, block, 0, stream>>>(topk_indices, logits_ptr, cu_seqlens_q_ptr, seqlens_kv_ptr,
                                       compress_ratio_divider, num_batch, row_stride);
  }

  return true;
}

}  // namespace topk
}  // namespace hpc
