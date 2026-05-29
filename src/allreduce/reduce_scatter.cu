// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <cuda_bf16.h>

#include <algorithm>

#include "src/allreduce/reduce_scatter.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace allreduce {
namespace kernels {

template <int kVecSize = 8, int kHiddenSize, int kNumThreadPerBlcok>
__global__ void reduce_scatter_kernel(const __nv_bfloat16 *__restrict__ input_ptr,
                                      const __nv_bfloat16 *__restrict__ mc_input_ptr,
                                      __nv_bfloat16 *__restrict__ output_ptr,
                                      __nv_bfloat16 *__restrict__ mc_output_ptr, uint32_t **signal,
                                      int rank, int world_size, const int num_tokens) {
  using T = __nv_bfloat162;
  constexpr int kN = kVecSize / 2;

  const int idx = threadIdx.x;
  const int itoken_start = blockIdx.x;

  // sync remote blocks
  if (idx < world_size) {
    auto target_rank = idx;
    auto bid = blockIdx.x;
    put_signal_relaxed(signal[target_rank] + bid * world_size + rank);
    wait_signal_relaxed(signal[rank] + bid * world_size + target_rank);
  }
  __syncthreads();

#pragma unroll 1
  for (int itoken = itoken_start; itoken < num_tokens; itoken += gridDim.x) {
    const int offset = itoken * kHiddenSize + idx * kVecSize;
    auto *mcptr_in = mc_input_ptr + offset;
    auto *out = output_ptr + offset;

    // 1. reduce sum input
    auto in_sum = multi_load_reduce_add<T, kN>(mcptr_in);

    // 2. store sum to output
    store(out, in_sum);
  }

  __syncthreads();
  // sync remote blocks
  if (idx < world_size) {
    auto target_rank = threadIdx.x;
    auto bid = blockIdx.x;
    put_signal_release(signal[target_rank] + bid * world_size + rank);
    wait_signal_acquire(signal[rank] + bid * world_size + target_rank);
  }
}

}  // namespace kernels

void reduce_scatter_async(const void *input_ptr, const void *mc_input_ptr, void *output_ptr,
                          void *mc_output_ptr, void *signal_ptr, int64_t rank, int64_t world_size,
                          int64_t num_max_blocks, int num_tokens, int hidden_size,
                          cudaStream_t stream) {
  constexpr int kVecSize = 8;
  if (hidden_size == 7168) {
    constexpr int kNumThreadPerBlcok = 896;
    constexpr int kHiddenSize = 7168;
    dim3 grid(num_max_blocks);
    dim3 block(kNumThreadPerBlcok);
    kernels::reduce_scatter_kernel<kVecSize, kHiddenSize, kNumThreadPerBlcok>
        <<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16 *>(input_ptr),
            static_cast<const __nv_bfloat16 *>(mc_input_ptr),
            static_cast<__nv_bfloat16 *>(output_ptr), static_cast<__nv_bfloat16 *>(mc_output_ptr),
            reinterpret_cast<uint32_t **>(signal_ptr), static_cast<int>(rank),
            static_cast<int>(world_size), num_tokens);
  } else if (hidden_size == 4096) {
    constexpr int kNumThreadPerBlcok = 512;
    constexpr int kHiddenSize = 4096;
    dim3 grid(num_max_blocks);
    dim3 block(kNumThreadPerBlcok);
    kernels::reduce_scatter_kernel<kVecSize, kHiddenSize, kNumThreadPerBlcok>
        <<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16 *>(input_ptr),
            static_cast<const __nv_bfloat16 *>(mc_input_ptr),
            static_cast<__nv_bfloat16 *>(output_ptr), static_cast<__nv_bfloat16 *>(mc_output_ptr),
            reinterpret_cast<uint32_t **>(signal_ptr), static_cast<int>(rank),
            static_cast<int>(world_size), num_tokens);
  }
}

}  // namespace allreduce
}  // namespace hpc
