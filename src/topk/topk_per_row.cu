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

__forceinline__ __device__ uint16_t extract_bin(float x) {
  union {
    __half h;
    uint16_t u16;
  } tmp;
  tmp.h = __float2half_rn(x);
  tmp.u16 = (x < 0.f) ? (~tmp.u16 & 0xffff) : (tmp.u16 | 0x8000);
  return 511 - (tmp.u16 >> 7);
}

template <int kTopK, int kFinalBinItems, int kThreadsPerBlock>
__global__ void topk_per_row_kernel(int *output_indices, const float *logits_ptr,
                                    const int *seqlens_ptr, int num_sp_tokens,
                                    cutlass::FastDivmod block2batch, const int row_stride) {
  constexpr int kItemPerLoad = 4;
  constexpr int kBins = 512;
  constexpr int kFinalThreads = 512;
  constexpr int kFinalItemsPerThread = kFinalBinItems / kFinalThreads;
  constexpr int kStoreItemsPerIter = kItemPerLoad * kThreadsPerBlock;

  struct Items {
    int indices[kFinalBinItems];
    float logits[kFinalBinItems];
  };

  using FinalBinSort = cub::BlockRadixSort<float, kFinalThreads, kFinalItemsPerThread, int>;
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

  BinPrefixSumScan scan(smem_scan);
  FinalBinSort final_sort(smem_sort.final_sort);

  int ibatch;
  int iseq;
  int idx = threadIdx.x;
  int irow = blockIdx.x;
  block2batch(ibatch, iseq, irow);

  int seqlen = seqlens_ptr[ibatch] - num_sp_tokens + iseq + 1;
  const auto *logits_row = logits_ptr + irow * row_stride;
  auto *output_indices_row = output_indices + irow * kTopK;

  if (seqlen < kTopK) {
    vec_t<int, kItemPerLoad> indices;
#pragma unroll
    for (int icol = idx * kItemPerLoad; icol < kTopK; icol += kStoreItemsPerIter) {
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

  if (idx < kBins) {
    smem_histogram[idx] = 0;
  }
  __syncthreads();

  for (int icol = idx * kItemPerLoad; icol < seqlen; icol += kStoreItemsPerIter) {
    auto logits = load<float, kItemPerLoad>(logits_row + icol);
#pragma unroll
    for (int i = 0; i < kItemPerLoad; i++) {
      int ipos = icol + i;
      if (ipos < seqlen) {
        uint16_t bin = extract_bin(logits[i]);
        atomicAdd(&smem_histogram[bin], 1);
      }
    }
  }
  __syncthreads();

  int bin_count = 0;
  if (idx < kBins) {
    bin_count = smem_histogram[idx];
  }
  __syncthreads();

  int prefix_sum = 0;
  int sum = 0;

  scan.ExclusiveSum(bin_count, prefix_sum, sum);

  if (idx < kBins) {
    smem_histogram[idx] = prefix_sum;
  }
  __syncthreads();

  // find the bin threshold which topk in.
  if (idx < kBins) {
    bool is_last_bin = (idx == kBins - 1);
    int next_prefix_sum = is_last_bin ? sum : smem_histogram[idx + 1];
    if (prefix_sum < kTopK && next_prefix_sum >= kTopK) {
      smem_threshold_bin[0] = idx;
    }
  }

  if (idx == 0) {
    smem_final_count[0] = 0;
  }
  __syncthreads();

  int threshold_bin = smem_threshold_bin[0];

  // perform final sort in threshold bin
  for (int icol = idx * kItemPerLoad; icol < seqlen; icol += kStoreItemsPerIter) {
    auto logits = load<float, kItemPerLoad>(logits_row + icol);
#pragma unroll
    for (int i = 0; i < kItemPerLoad; i++) {
      uint16_t bin = extract_bin(logits[i]);
      int ipos = icol + i;
      if (ipos < seqlen) {
        if (bin < threshold_bin) {
          int smem_idx = atomicAdd(&smem_histogram[bin], 1);
          smem_indices[smem_idx] = ipos;
        } else if (bin == threshold_bin) {
          int smem_idx = atomicAdd(&smem_final_count[0], 1);
          if (smem_idx < kFinalBinItems) {
            smem_sort.items.logits[smem_idx] = logits[i];
            smem_sort.items.indices[smem_idx] = ipos;
          }
        }
      }
    }
  }
  __syncthreads();

  if (idx < kFinalThreads) {
    vec_t<float, kFinalItemsPerThread> final_logits;
    vec_t<int, kFinalItemsPerThread> final_indices;

#pragma unroll
    for (int i = 0; i < kFinalItemsPerThread; i++) {
      final_logits[i] = std::numeric_limits<float>::lowest();
    }

    int final_count = smem_final_count[0];

#pragma unroll
    for (int i = 0; i < kFinalItemsPerThread; i++) {
      int final_idx = i * kFinalThreads + idx;
      if (final_idx < final_count) {
        final_logits[i] = smem_sort.items.logits[final_idx];
        final_indices[i] = smem_sort.items.indices[final_idx];
      }
    }
    __syncthreads();

    final_sort.SortDescendingBlockedToStriped(final_logits.data, final_indices.data);

    int ibase = threshold_bin > 0 ? smem_histogram[threshold_bin - 1] : 0;
#pragma unroll
    for (int i = 0; i < kFinalItemsPerThread; i++) {
      int ifinal = i * kFinalThreads + idx;
      int smem_idx = ibase + ifinal;
      if (smem_idx < kTopK) {
        smem_indices[smem_idx] = final_indices[i];
      }
    }

    __syncthreads();
    // store result

#pragma unroll
    for (int icol = idx * kItemPerLoad; icol < kTopK; icol += kItemPerLoad * kFinalThreads) {
      auto result = load<int, kItemPerLoad>(smem_indices + icol);
      store(output_indices_row + icol, result);
    }
  } else {
    // make sure cub sort __syncthreads to work
    return;
  }
}

}  // namespace kernels

bool topk_per_row_async(int *topk_indices, const float *logits, const int *seqlens, int topk,
                        int num_sp_tokens, int num_rows, int row_stride, cudaStream_t stream) {
  constexpr int kTopK = 2048;
  constexpr int kThreadsPerBlock = 1024;
  constexpr int kFinalBinItems = 3072;

  cutlass::FastDivmod Row2Batch(num_sp_tokens);

  dim3 block(kThreadsPerBlock);
  dim3 grid(num_rows);

  auto kernel = kernels::topk_per_row_kernel<kTopK, kFinalBinItems, kThreadsPerBlock>;
  kernel<<<grid, block, 0, stream>>>(topk_indices, logits, seqlens, num_sp_tokens, Row2Batch,
                                     row_stride);

  return true;
}

}  // namespace topk
}  // namespace hpc
