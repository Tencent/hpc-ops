// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <cuda_bf16.h>
#include <stdio.h>

#include <string>

#include "cutlass/fast_math.h"
#include "src/rope/rope.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace rope {

namespace kernels {

template <typename T, int kRowsPerBlock, int kThreadsPerRow, int kItemsPerThread>
__global__ void rope_interleave_kernels(T* y_ptr, T* x_ptr, const float* cos_sin_cache_ptr,
                                        const int* cu_seqlens_q_ptr, const int* seqlen_kv_ptr,
                                        int num_batch, int num_tokens, int ldX, int ldCache,
                                        int ldY) {
  int itoken = blockIdx.x * kRowsPerBlock + threadIdx.x / kThreadsPerRow;

  int ibatch = 0;
  int itoken_in_batch = itoken;
  int seqlenq = 0;

  for (int i = 1; i < num_batch + 1; i++) {
    int cu_seqlenq = cu_seqlens_q_ptr[i];
    if (itoken < cu_seqlenq) {
      ibatch = i - 1;
      itoken_in_batch = itoken - cu_seqlens_q_ptr[ibatch];
      seqlenq = cu_seqlenq - cu_seqlens_q_ptr[ibatch];
      break;
    }
  }

  int ipos = (seqlen_kv_ptr[ibatch] - seqlenq) + itoken_in_batch;
  int icol = (threadIdx.x % kThreadsPerRow) * kItemsPerThread;

  if (itoken >= num_tokens) {
    return;
  }

  T* y_row = y_ptr + itoken * ldY;
  T* x_row = x_ptr + itoken * ldX;
  const float* cos_sin_cache_row = cos_sin_cache_ptr + ipos * ldCache;

  auto x = to<float>(load<T, kItemsPerThread>(x_row + icol));
  auto cos_sin = load<float, kItemsPerThread>(cos_sin_cache_row + icol);

#pragma unroll
  for (int i = 0; i < kItemsPerThread / 2; i++) {
    auto x1 = x[2 * i];
    auto x2 = x[2 * i + 1];
    auto cos = cos_sin[2 * i];
    auto sin = cos_sin[2 * i + 1];

    x[2 * i] = x1 * cos - x2 * sin;
    x[2 * i + 1] = x1 * sin + x2 * cos;
  }

  store(y_row + icol, to<T>(x));
}

}  // namespace kernels

bool rope_interleave_bf16_async(void* y_ptr, void* x_ptr, const void* cos_sin_cache_ptr,
                                const void* cu_seqlens_q_ptr, const void* seqlen_kv_ptr,
                                int num_batch, int num_tokens, int dim, int ldX, int ldCache,
                                int ldY, cudaStream_t stream) {
  constexpr int kDim = 64;
  constexpr int kItemsPerThread = 4;
  constexpr int kThreadsPerRow = kDim / kItemsPerThread;
  constexpr int kRowsPerWarp = 32 / kThreadsPerRow;
  constexpr int kWarpCount = 4;
  constexpr int kRowsPerBlock = kRowsPerWarp * kWarpCount;

  dim3 grid((num_tokens + kRowsPerBlock - 1) / kRowsPerBlock);
  dim3 block(kWarpCount * 32);

  using T = __nv_bfloat16;
  kernels::rope_interleave_kernels<T, kRowsPerBlock, kThreadsPerRow, kItemsPerThread>
      <<<grid, block, 0, stream>>>(reinterpret_cast<T*>(y_ptr), reinterpret_cast<T*>(x_ptr),
                                   reinterpret_cast<const float*>(cos_sin_cache_ptr),
                                   reinterpret_cast<const int*>(cu_seqlens_q_ptr),
                                   reinterpret_cast<const int*>(seqlen_kv_ptr), num_batch,
                                   num_tokens, ldX, ldCache, ldY);

  return true;
}

}  // namespace rope
}  // namespace hpc
