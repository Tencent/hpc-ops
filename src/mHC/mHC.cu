// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include <algorithm>

#include "src/mHC/mHC.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace mHC {
namespace kernels {

// one block per row
template <int kElementsPerThread, int kIter, int kHiddenDim>
__global__ void reciprocal_mean_square_root_norm_kernel(__nv_bfloat16* output_ptr,
                                                        const __nv_bfloat16* input_ptr,
                                                        float norm_eps) {
  int irow = blockIdx.x;
  int icol = threadIdx.x * kElementsPerThread;
  const int iwarp = threadIdx.x / 32;
  const int ilane = threadIdx.x % 32;
  __shared__ float smem_sum[32];

  constexpr float kInvHiddenDim = 1.0f / kHiddenDim;
  __nv_bfloat16* output_row_ptr = output_ptr + irow * kHiddenDim;
  const __nv_bfloat16* input_row_ptr = input_ptr + irow * kHiddenDim;
  vec_t<float, kElementsPerThread> reg_input[kIter];
  float local_sum = 0.0f;

#pragma unroll
  for (int i = 0; i < kIter; ++i) {
    reg_input[i] = to<float>(load<__nv_bfloat162, kElementsPerThread / 2>(
        input_row_ptr + i * blockDim.x * kElementsPerThread + icol));
#pragma unroll
    for (int j = 0; j < kElementsPerThread; ++j) {
      local_sum += reg_input[i][j] * reg_input[i][j];
    }
  }

  // warp reduce
  local_sum = warp_reduce_sum_xor(local_sum);
  if (ilane == 0) {
    smem_sum[iwarp] = local_sum;
  }
  __syncthreads();

  // block reduce
  if (iwarp == 0) {
    local_sum = smem_sum[ilane];
    local_sum = warp_reduce_sum_xor(local_sum);
    if (ilane == 0) {
      smem_sum[0] = local_sum;
    }
  }
  __syncthreads();

  float local_mean = rsqrtf_ftz(smem_sum[0] * kInvHiddenDim + norm_eps);

#pragma unroll
  for (int i = 0; i < kIter; ++i) {
#pragma unroll
    for (int j = 0; j < kElementsPerThread; ++j) {
      reg_input[i][j] *= local_mean;
    }

    store(output_row_ptr + i * blockDim.x * kElementsPerThread + icol,
          to<__nv_bfloat162>(reg_input[i]));
  }
}

// one thread per row
template <int kHCMult>
__global__ void fuse_cal_three_H_kernel(float* output_H_pre_ptr, float* output_H_post_ptr,
                                        float* output_H_res_ptr, const float* mixes_hat_hat_H_ptr,
                                        const float* hc_scale_ptr, const float* hc_base_ptr,
                                        int num_batch, int hc_sinkhorn_iters, float hc_eps) {
  const int irow = blockIdx.x * blockDim.x + threadIdx.x;
  if (irow >= num_batch) return;

  constexpr int kHCDim = 2 * kHCMult + kHCMult * kHCMult;
  const float* mixes_hat_hat_H_row_ptr = mixes_hat_hat_H_ptr + irow * kHCDim;
  vec_t<float, kHCMult> reg_H_pre_post;  // also use as row max, row sum, col sum for H_res
  vec_t<float, kHCMult * kHCMult> reg_H_res;

  // 1. H_pre = sigmoid(scale_pre * hat_hat_H_pre + base_pre) + hc_eps
  float* output_H_pre_row_ptr = output_H_pre_ptr + irow * kHCMult;
#pragma unroll
  for (int i = 0; i < kHCMult; ++i) {
    reg_H_pre_post[i] =
        sigmoid(hc_scale_ptr[0] * mixes_hat_hat_H_row_ptr[i] + hc_base_ptr[i]) + hc_eps;
  }
  store(output_H_pre_row_ptr, reg_H_pre_post);

  // 2. H_post = 2 * sigmoid(scale_post * hat_hat_H_post + base_post)
  float* output_H_post_row_ptr = output_H_post_ptr + irow * kHCMult;
#pragma unroll
  for (int i = 0; i < kHCMult; ++i) {
    reg_H_pre_post[i] = 2.f * sigmoid(hc_scale_ptr[1] * mixes_hat_hat_H_row_ptr[kHCMult + i] +
                                      hc_base_ptr[kHCMult + i]);
  }
  store(output_H_post_row_ptr, reg_H_pre_post);

  // 3. hat_H_res = scale_res * hat_hat_H_res + base_res
  float* output_H_res_row_ptr = output_H_res_ptr + irow * kHCMult * kHCMult;
#pragma unroll
  for (int i = 0; i < kHCMult * kHCMult; ++i) {
    reg_H_res[i] =
        hc_scale_ptr[2] * mixes_hat_hat_H_row_ptr[2 * kHCMult + i] + hc_base_ptr[2 * kHCMult + i];
  }

  // 4. H_res = Sinkhorn-Knopp(har_H_res, hc_sinkhorn_iters, hc_eps)

  // 4.1 H_res = hat_H_res.softmax(-1) + eps
  // 4.1.1 row_max = H_res.row_max()
#pragma unroll
  for (int i = 0; i < kHCMult; ++i) {
    reg_H_pre_post[i] = -1e30f;
#pragma unroll
    for (int j = 0; j < kHCMult; ++j) {
      reg_H_pre_post[i] = max(reg_H_pre_post[i], reg_H_res[i * kHCMult + j]);
    }
  }

  // 4.1.2 H_res = exp(H_res - row_max)
#pragma unroll
  for (int i = 0; i < kHCMult; ++i) {
#pragma unroll
    for (int j = 0; j < kHCMult; ++j) {
      reg_H_res[i * kHCMult + j] = expf_ftz(reg_H_res[i * kHCMult + j] - reg_H_pre_post[i]);
    }
  }

  // 4.1.3 row_sum = H_res.row_sum()
#pragma unroll
  for (int i = 0; i < kHCMult; ++i) {
    reg_H_pre_post[i] = 0.f;
#pragma unroll
    for (int j = 0; j < kHCMult; ++j) {
      reg_H_pre_post[i] += reg_H_res[i * kHCMult + j];
    }
  }

  // 4.1.4 H_res = H_res / row_sum + eps
#pragma unroll
  for (int i = 0; i < kHCMult; ++i) {
#pragma unroll
    for (int j = 0; j < kHCMult; ++j) {
      reg_H_res[i * kHCMult + j] =
          reg_H_res[i * kHCMult + j] * rcpf_ftz(reg_H_pre_post[i]) + hc_eps;
    }
  }

  // 4.2 H_res = H_res / (col_sum + eps)

  // 4.2.1 col_sum = H_res.col_sum()
#pragma unroll
  for (int j = 0; j < kHCMult; ++j) {
    reg_H_pre_post[j] = 0.f;
#pragma unroll
    for (int i = 0; i < kHCMult; ++i) {
      reg_H_pre_post[j] += reg_H_res[i * kHCMult + j];
    }
  }

  // 4.2.2 H_res = H_res / (col_sum + eps)
#pragma unroll
  for (int j = 0; j < kHCMult; ++j) {
#pragma unroll
    for (int i = 0; i < kHCMult; ++i) {
      reg_H_res[i * kHCMult + j] =
          reg_H_res[i * kHCMult + j] * rcpf_ftz(reg_H_pre_post[j] + hc_eps);
    }
  }

  // 5 H_res = Sinkhorn-Knopp(har_H_res, hc_sinkhorn_iters-1 , hc_eps)
  for (int k = 0; k < hc_sinkhorn_iters - 1; ++k) {
    // H_res = H_res / (row_sum + eps)
#pragma unroll
    for (int i = 0; i < kHCMult; ++i) {
      reg_H_pre_post[i] = 0.f;
#pragma unroll
      for (int j = 0; j < kHCMult; ++j) {
        reg_H_pre_post[i] += reg_H_res[i * kHCMult + j];
      }
    }
#pragma unroll
    for (int i = 0; i < kHCMult; ++i) {
#pragma unroll
      for (int j = 0; j < kHCMult; ++j) {
        reg_H_res[i * kHCMult + j] =
            reg_H_res[i * kHCMult + j] * rcpf_ftz(reg_H_pre_post[i]) + hc_eps;
      }
    }

    // H_res = H_res / (row_sum + eps)
#pragma unroll
    for (int j = 0; j < kHCMult; ++j) {
      reg_H_pre_post[j] = 0.f;
#pragma unroll
      for (int i = 0; i < kHCMult; ++i) {
        reg_H_pre_post[j] += reg_H_res[i * kHCMult + j];
      }
    }

    // 4.2.2 H_res = H_res / (col_sum + eps)
#pragma unroll
    for (int j = 0; j < kHCMult; ++j) {
#pragma unroll
      for (int i = 0; i < kHCMult; ++i) {
        reg_H_res[i * kHCMult + j] =
            reg_H_res[i * kHCMult + j] * rcpf_ftz(reg_H_pre_post[j] + hc_eps);
      }
    }
  }

#pragma unroll
  for (int i = 0; i < kHCMult; ++i) {
    vec_t<float, 4>& v = *reinterpret_cast<vec_t<float, 4>*>(&reg_H_res[i * kHCMult]);
    store(output_H_res_row_ptr + i * kHCMult, v);
  }
}

// one block per row
template <int kHCMult, int kHiddenDim, int kElementsPerThread>
__global__ void fuse_hc_pre_mapping_kernel(__nv_bfloat16* output_ptr, const __nv_bfloat16* x_ptr,
                                           const float* H_pre_ptr) {
  const int irow = blockIdx.x;
  const float* H_pre_row_ptr = H_pre_ptr + irow * kHCMult;
  auto reg_H_pre = load<float, kHCMult>(H_pre_row_ptr);

  const int icol = threadIdx.x * kElementsPerThread;
  const __nv_bfloat16* x_row_ptr = x_ptr + irow * kHCMult * kHiddenDim + icol;

  vec_t<float, kElementsPerThread> reg_output;
#pragma unroll
  for (int i = 0; i < kElementsPerThread; ++i) {
    reg_output[i] = 0.f;
  }

#pragma unroll
  for (int i = 0; i < kHCMult; ++i) {
    auto reg_x = to<float>(load<__nv_bfloat16, kElementsPerThread>(x_row_ptr + i * kHiddenDim));
#pragma unroll
    for (int j = 0; j < kElementsPerThread; ++j) {
      reg_output[j] += reg_x[j] * reg_H_pre[i];
    }
  }

  store(output_ptr + irow * kHiddenDim + icol, to<__nv_bfloat16>(reg_output));
}

template <int kHCMult, int kHiddenDim, int kElementsPerThread>
__global__ void fuse_H_post_mapping_H_res_mapping_and_residual_add_kernel(
    __nv_bfloat16* output_ptr, const __nv_bfloat16* x_ptr, const __nv_bfloat16* residual_ptr,
    const float* H_post_ptr, const float* H_res_ptr) {
  const int irow = blockIdx.x;
  const float* H_post_row_ptr = H_post_ptr + irow * kHCMult;
  auto reg_H_post = load<float, kHCMult>(H_post_row_ptr);

  const int icol = threadIdx.x * kElementsPerThread;
  const __nv_bfloat16* x_row_ptr = x_ptr + irow * kHiddenDim + icol;
  auto reg_x = to<float>(load<__nv_bfloat16, kElementsPerThread>(x_row_ptr));

  const float* H_res_row_ptr = H_res_ptr + irow * kHCMult * kHCMult;
  const __nv_bfloat16* residual_row_ptr = residual_ptr + irow * kHCMult * kHiddenDim + icol;
  __nv_bfloat16* output_row_ptr = output_ptr + irow * kHCMult * kHiddenDim + icol;

#pragma unroll
  for (int i = 0; i < kHCMult; ++i) {
    vec_t<float, kElementsPerThread> reg_output;

#pragma unroll
    for (int j = 0; j < kElementsPerThread; ++j) {
      reg_output[j] = reg_H_post[i] * reg_x[j];
    }

    auto reg_H_res = load<float, kHCMult>(H_res_row_ptr + i * kHCMult);
    auto reg_residual =
        to<float>(load<__nv_bfloat16, kElementsPerThread>(residual_row_ptr + i * kHiddenDim));
#pragma unroll
    for (int j = 0; j < kHCMult; ++j) {
#pragma unroll
      for (int k = 0; k < kElementsPerThread; ++k) {
        reg_output[k] += reg_H_res[j] * reg_residual[k];
      }
    }

    store(output_row_ptr + i * kHiddenDim, to<__nv_bfloat16>(reg_output));
  }
}

}  // namespace kernels

void reciprocal_mean_square_root_norm_async(__nv_bfloat16* output_ptr,
                                            const __nv_bfloat16* input_ptr, int num_batch,
                                            int hidden_dim, float norm_eps, cudaStream_t stream) {
  if (hidden_dim == 16384) {
    dim3 grid(num_batch);
    dim3 block(1024);
    constexpr int kElementsPerThread = 8;
    constexpr int kIter = 2;  // 16384 / 1024 / 8

    kernels::reciprocal_mean_square_root_norm_kernel<kElementsPerThread, kIter, 16384>
        <<<grid, block, 0, stream>>>(output_ptr, input_ptr, norm_eps);
  }
}

void fuse_cal_three_H_async(float* output_H_pre_ptr, float* output_H_post_ptr,
                            float* output_H_res_ptr, const float* mixes_hat_hat_H_ptr,
                            const float* hc_scale_ptr, const float* hc_base_ptr, int num_batch,
                            int hc_mult, int hc_sinkhorn_iters, float hc_eps, cudaStream_t stream) {
  if (hc_mult == 4) {
    constexpr int kWarpSize = 32;
    constexpr int kWarpPerBlock = 4;
    constexpr int kThreadPerBlock = kWarpPerBlock * kWarpSize;
    const int num_block = (num_batch + kThreadPerBlock - 1) / kThreadPerBlock;
    kernels::fuse_cal_three_H_kernel<4><<<num_block, kThreadPerBlock, 0, stream>>>(
        output_H_pre_ptr, output_H_post_ptr, output_H_res_ptr, mixes_hat_hat_H_ptr, hc_scale_ptr,
        hc_base_ptr, num_batch, hc_sinkhorn_iters, hc_eps);
  }
}

void fuse_hc_pre_mapping_async(__nv_bfloat16* output_ptr, const __nv_bfloat16* x_ptr,
                               const float* H_pre_ptr, int num_batch, int hc_mult, int hidden_dim,
                               cudaStream_t stream) {
  if (hc_mult == 4 && hidden_dim == 4096) {
    constexpr int kElementsPerThread = 8;
    constexpr int kThreadsPerBlock = 4096 / kElementsPerThread;

    kernels::fuse_hc_pre_mapping_kernel<4, 4096, kElementsPerThread>
        <<<num_batch, kThreadsPerBlock, 0, stream>>>(output_ptr, x_ptr, H_pre_ptr);
  }
}

void fuse_H_post_mapping_H_res_mapping_and_residual_add_async(
    __nv_bfloat16* output_ptr, const __nv_bfloat16* x_ptr, const __nv_bfloat16* residual_ptr,
    const float* H_post_ptr, const float* H_res_ptr, int num_batch, int hc_mult, int hidden_dim,
    cudaStream_t stream) {
  if (hc_mult == 4 && hidden_dim == 4096) {
    constexpr int kElementsPerThread = 8;
    constexpr int kThreadsPerBlock = 4096 / kElementsPerThread;

    kernels::fuse_H_post_mapping_H_res_mapping_and_residual_add_kernel<4, 4096, kElementsPerThread>
        <<<num_batch, kThreadsPerBlock, 0, stream>>>(output_ptr, x_ptr, residual_ptr, H_post_ptr,
                                                     H_res_ptr);
  }
}

}  // namespace mHC
}  // namespace hpc
