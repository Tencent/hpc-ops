// Copyright (C) 2026 Tencent.

#include <cuda.h>
#include <stdio.h>

#include <cub/cub.cuh>

#include "cute/tensor.hpp"
#include "src/fuse_moe/fuse_moe.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace fuse_moe {

namespace kernels {

template <typename T, int kThreadPerBlock, int kNumItemPer16B, int kNumTopkMax = 128>
__global__ void reduce_kernel(T *y_ptr, const T *x_ptr, const int *topk_pos_ptr,
                              const float *topk_scale_ptr, const T *shared_output_ptr,
                              int total_num_seq, int num_seq, int hidden_size, int num_topk,
                              cutlass::FastDivmod block_divider) {
  int idx = threadIdx.x;
  int iblock = blockIdx.x;

  int iblockx;
  int iblocky;

  block_divider(iblocky, iblockx, iblock);
  int irow = iblocky;
  int icol = (threadIdx.x + iblockx * kThreadPerBlock) * kNumItemPer16B;

  __shared__ int pos_shm[kNumTopkMax];
  __shared__ float scale_shm[kNumTopkMax];

#pragma unroll 1
  for (int i = idx; i < num_topk; i += blockDim.x) {
    pos_shm[i] = topk_pos_ptr[irow * num_topk + i];
    scale_shm[i] = topk_scale_ptr[irow * num_topk + i];
  }
  __syncthreads();

  if (icol < hidden_size) {
    vec_t<float, kNumItemPer16B> y_fp32;
#pragma unroll
    for (int i = 0; i < kNumItemPer16B; i++) {
      y_fp32[i] = 0.f;
    }

    auto y_irow_ptr = y_ptr + irow * hidden_size;
    for (int i = 0; i < num_topk; i++) {
      int ipos = pos_shm[i];
      float iscale = scale_shm[i];
      if (ipos >= 0) {
        auto x_irow_ptr = x_ptr + ipos * hidden_size;
        auto x_fp32 = to<float>(load<T, kNumItemPer16B>(x_irow_ptr + icol));
#pragma unroll
        for (int j = 0; j < kNumItemPer16B; j++) {
          y_fp32[j] += x_fp32[j] * iscale;
        }
      }
    }
    if (shared_output_ptr) {
      auto shared_irow_ptr = shared_output_ptr + irow * hidden_size;
      auto shared_fp32 = to<float>(load<T, kNumItemPer16B>(shared_irow_ptr + icol));
#pragma unroll
      for (int j = 0; j < kNumItemPer16B; j++) {
        y_fp32[j] += shared_fp32[j];
      }
    }
    auto y_bf16 = to<T>(y_fp32);
    store(y_irow_ptr + icol, y_bf16);
  }
}

}  // namespace kernels

void reduce_async(void *y_ptr, const void *x_ptr, const void *topk_pos_ptr,
                  const void *topk_scale_ptr, const void *shared_output_ptr, int total_num_seq,
                  int num_seq, int hidden_size, int num_topk, cudaStream_t stream) {
  using T = __nv_bfloat16;

  // reduce tokens
  {
    constexpr int kThreadPerBlock = 256;
    constexpr int kNumItemPer16B = 16 / sizeof(T);
    int num_block_col = (hidden_size / kNumItemPer16B + kThreadPerBlock - 1) / kThreadPerBlock;
    int num_block_total = num_seq * num_block_col;

    cutlass::FastDivmod block_divider(num_block_col);

    dim3 block(kThreadPerBlock);
    dim3 grid(num_block_total);
    kernels::reduce_kernel<T, kThreadPerBlock, kNumItemPer16B><<<grid, block, 0, stream>>>(
        (T *)y_ptr, (const T *)x_ptr, (const int *)topk_pos_ptr, (const float *)topk_scale_ptr,
        (const T *)shared_output_ptr, total_num_seq, num_seq, hidden_size, num_topk, block_divider);
  }
}

}  // namespace fuse_moe
}  // namespace hpc
