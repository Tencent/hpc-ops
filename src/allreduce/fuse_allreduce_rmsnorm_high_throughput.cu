// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <cuda_bf16.h>

#include <algorithm>

#include "src/allreduce/fuse_allreduce_rmsnorm_high_throughput.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace allreduce {
namespace kernels {

template <int kVecSize = 8, int kHiddenSize, int kNumThreadPerBlcok>
__global__ void fused_allreduce_rmsnorm_kernel(
    const __nv_bfloat16 *__restrict__ input_ptr, const __nv_bfloat16 *__restrict__ mc_input_ptr,
    const __nv_bfloat16 *__restrict__ in_res_ptr, const __nv_bfloat16 *__restrict__ weight_ptr,
    __nv_bfloat16 *__restrict__ output_ptr, __nv_bfloat16 *__restrict__ mc_output_ptr,
    __nv_bfloat16 *__restrict__ out_res_ptr, uint32_t **signal, int rank, int world_size,
    const float rms_norm_eps, const int num_tokens) {
  constexpr int kNumWarp = kNumThreadPerBlcok / 32;
  constexpr float kInvHiddenStates = 1.0f / kHiddenSize;

  using T = __nv_bfloat162;
  constexpr int kN = kVecSize / 2;

  __shared__ float smem_sum2[32];
  __shared__ float smem_var;

  const int ilane = threadIdx.x % 32;
  const int iwarp = threadIdx.x / 32;
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
    auto *residual_in = in_res_ptr + offset;
    auto *mcptr_out = mc_output_ptr + offset;
    auto *residual_out = out_res_ptr + offset;

    // 1. reduce sum input and residual
    auto in_sum = to<float>(multi_load_reduce_add<T, kN>(mcptr_in));
    auto res = to<float>(load<T, kN>(residual_in));
    auto weight = to<float>(load<T, kN>(weight_ptr + idx * kVecSize));

    float sum2 = 0.f;
#pragma unroll
    for (int i = 0; i < size(in_sum); i++) {
      in_sum[i] += res[i];
      sum2 += in_sum[i] * in_sum[i];
    }
    store(residual_out, to<T>(in_sum));

    // 2. block reduce sum variance
    sum2 = warp_reduce_sum_down(sum2);
    if (ilane == 0) {
      smem_sum2[iwarp] = sum2;
    }
    __syncthreads();
    if (iwarp == 0) {
      bool mask = ilane >= kNumWarp;
      sum2 = mask ? 0.f : smem_sum2[ilane];
      sum2 = warp_reduce_sum_down(sum2);
      if (ilane == 0) {
        smem_var = rsqrtf_ftz(sum2 * kInvHiddenStates + rms_norm_eps);
      }
    }
    __syncthreads();
    float var = smem_var;

    // 3. normalize and apply weight
#pragma unroll
    for (int i = 0; i < size(in_sum); i++) {
      in_sum[i] = in_sum[i] * var * weight[i];
    }
    multi_store(mcptr_out, to<T>(in_sum));
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

void fuse_allreduce_rmsnorm_high_throughput_async(const void *input_ptr, const void *mc_input_ptr,
                                                  const void *in_res_ptr, const void *weight_ptr,
                                                  void *output_ptr, void *mc_output_ptr,
                                                  void *out_res_ptr, void *signal_ptr, int64_t rank,
                                                  int64_t world_size, int64_t num_max_blocks,
                                                  double rms_norm_eps, int num_tokens,
                                                  int hidden_size, cudaStream_t stream) {
  constexpr int kVecSize = 8;
  if (hidden_size == 4096) {
    constexpr int kNumThreadPerBlcok = 512;
    constexpr int kHiddenSize = 4096;
    dim3 grid(num_max_blocks);
    dim3 block(kNumThreadPerBlcok);
    kernels::fused_allreduce_rmsnorm_kernel<kVecSize, kHiddenSize, kNumThreadPerBlcok>
        <<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16 *>(input_ptr),
            static_cast<const __nv_bfloat16 *>(mc_input_ptr),
            static_cast<const __nv_bfloat16 *>(in_res_ptr),
            static_cast<const __nv_bfloat16 *>(weight_ptr),
            static_cast<__nv_bfloat16 *>(output_ptr), static_cast<__nv_bfloat16 *>(mc_output_ptr),
            static_cast<__nv_bfloat16 *>(out_res_ptr), reinterpret_cast<uint32_t **>(signal_ptr),
            static_cast<int>(rank), static_cast<int>(world_size), rms_norm_eps, num_tokens);
  } else if (hidden_size == 5120) {
    constexpr int kNumThreadPerBlcok = 640;
    constexpr int kHiddenSize = 5120;
    dim3 grid(num_max_blocks);
    dim3 block(kNumThreadPerBlcok);
    kernels::fused_allreduce_rmsnorm_kernel<kVecSize, kHiddenSize, kNumThreadPerBlcok>
        <<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16 *>(input_ptr),
            static_cast<const __nv_bfloat16 *>(mc_input_ptr),
            static_cast<const __nv_bfloat16 *>(in_res_ptr),
            static_cast<const __nv_bfloat16 *>(weight_ptr),
            static_cast<__nv_bfloat16 *>(output_ptr), static_cast<__nv_bfloat16 *>(mc_output_ptr),
            static_cast<__nv_bfloat16 *>(out_res_ptr), reinterpret_cast<uint32_t **>(signal_ptr),
            static_cast<int>(rank), static_cast<int>(world_size), rms_norm_eps, num_tokens);
  } else if (hidden_size == 7168) {
    constexpr int kNumThreadPerBlcok = 896;
    constexpr int kHiddenSize = 7168;
    dim3 grid(num_max_blocks);
    dim3 block(kNumThreadPerBlcok);
    kernels::fused_allreduce_rmsnorm_kernel<kVecSize, kHiddenSize, kNumThreadPerBlcok>
        <<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16 *>(input_ptr),
            static_cast<const __nv_bfloat16 *>(mc_input_ptr),
            static_cast<const __nv_bfloat16 *>(in_res_ptr),
            static_cast<const __nv_bfloat16 *>(weight_ptr),
            static_cast<__nv_bfloat16 *>(output_ptr), static_cast<__nv_bfloat16 *>(mc_output_ptr),
            static_cast<__nv_bfloat16 *>(out_res_ptr), reinterpret_cast<uint32_t **>(signal_ptr),
            static_cast<int>(rank), static_cast<int>(world_size), rms_norm_eps, num_tokens);
  }
}

}  // namespace allreduce
}  // namespace hpc
