// Copyright (C) 2026 Tencent.

#include <cuda.h>
#include <stdio.h>

#include "cutlass/fast_math.h"
#include "src/activation/activation.h"
#include "src/utils/utils.cuh"
#include "src/utils/utils.h"

namespace hpc {
namespace activation {
namespace kernels {

template <bool kUseBFloat16PrecisionMultiply = true>
__global__ void act_mul_and_quant_kernel(__nv_fp8_e4m3 *out_ptr, const __nv_bfloat16 *gate_up_ptr,
                                         const float *scale_ptr, const int *valid_row_range,
                                         const int num_row, const int num_col,
                                         cutlass::FastDivmod block1D22D) {
  int iblockx;
  int iblocky;

  block1D22D(iblocky, iblockx, blockIdx.x);
  int it = threadIdx.x + iblockx * blockDim.x;

  int irow = iblocky;
  int my_valid_row_end_exclusive = valid_row_range ? valid_row_range[0] : num_row;
  if (irow >= my_valid_row_end_exclusive) {
    return;
  }

  using T = __nv_bfloat162;

  float scale = scale_ptr[0];

  const auto *gate_row_ptr = gate_up_ptr + irow * num_col * 2;
  const auto *up_row_ptr = gate_row_ptr + num_col;
  auto *out_row_ptr = out_ptr + irow * num_col;

  int icol = it * 8;

  if (icol < num_col) {
    auto gate = to<float>(load<T, 4>(gate_row_ptr + icol));
    auto up = to<float>(load<T, 4>(up_row_ptr + icol));

    vec_t<float, 8> out;
#pragma unroll
    for (int i = 0; i < size(out); ++i) {
      auto g = gate[i];
      auto m = [&] {
        if constexpr (kUseBFloat16PrecisionMultiply) {
          auto u = __float2bfloat16_rn(up[i]);
          return __bfloat162float(__float2bfloat16_rn(silu(g)) * u);
        } else {
          auto u = up[i];
          return silu(g) * u;
        }
      }();
      out[i] = m * scale;
    }

    auto out_fp8 = to<__nv_fp8x4_e4m3>(out);
    store(out_row_ptr + icol, out_fp8);
  }
}

// input : gate + up
__global__ void masked_act_mul_and_quant_kernel(
    __nv_fp8_e4m3 *output_ptr, const __nv_bfloat16 *input_ptr, const float *scale_ptr,
    const int *num_per_expert_ptr, int num_total_tokens, int num_intermediate_size,
    int num_tokens_per_expert, cutlass::FastDivmod Block2YX, cutlass::FastDivmod Row2EandT,
    int num_block_row) {
  constexpr int kRows = 4;

  int iblockx;
  int iblocky;

  Block2YX(iblocky, iblockx, blockIdx.x);

  float scale = scale_ptr[0];

#pragma unroll 1
  for (int irow0 = iblocky * kRows; irow0 < num_total_tokens; irow0 += num_block_row * kRows) {
    int it = threadIdx.x + iblockx * blockDim.x;

#pragma unroll
    for (int i = 0; i < kRows; ++i) {
      int iexpert;
      int itoken;

      int irow = irow0 + i;

      Row2EandT(iexpert, itoken, irow);
      int num_tokens_curr_expert = num_per_expert_ptr[iexpert];
      if (itoken >= num_tokens_curr_expert) {
        continue;
      }

      const auto *gate_row_ptr = input_ptr + irow * (num_intermediate_size * 2);
      const auto *up_row_ptr = gate_row_ptr + num_intermediate_size;
      auto *output_row_ptr = output_ptr + irow * num_intermediate_size;

      int icol = it * 8;
      if (icol < num_intermediate_size) {
        auto gate = to<float>(load<__nv_bfloat162, 4>(gate_row_ptr + icol));
        auto up = to<float>(load<__nv_bfloat162, 4>(up_row_ptr + icol));
        decltype(gate) out;

#pragma unroll
        for (int i = 0; i < decltype(gate)::kNum; ++i) {
          out[i] = silu(gate[i]) * up[i] * scale;
        }

        auto out_fp8 = to<__nv_fp8x4_e4m3>(out);

        store(output_row_ptr + icol, out_fp8);
      }
    }  // for
  }  // irow0
}

__device__ __forceinline__ void get_group_id(int irow, const int *cu_num_tokens_per_group_ptr,
                                             int num_group, int &igroup, int &offset) {
  int left = 0;
  int right = num_group;
  while (left <= right) {
    int mid = left + (right - left) / 2;
    if (cu_num_tokens_per_group_ptr[mid] > irow) {
      right = mid - 1;
    } else {
      left = mid + 1;
    }
  }
  offset = irow - cu_num_tokens_per_group_ptr[right];
  igroup = right;
}

__global__ void act_mul_and_blockwise_quant_kernel(
    const __nv_bfloat16 *gate_up_output_ptr, __nv_fp8_e4m3 *output_ptr, float *output_scale_ptr,
    const int *cu_num_tokens_per_group_ptr, const int *cu_tiles_ptr, const int num_row,
    const int num_row_padded_size, const int num_col, const int num_group, const int ktile_m,
    cutlass::FastDivmod block1D22D) {
  int iblockx;
  int iblocky;

  block1D22D(iblocky, iblockx, blockIdx.x);
  int it = threadIdx.x + iblockx * blockDim.x;
  int irow = iblocky;
  if (irow >= cu_num_tokens_per_group_ptr[num_group]) {
    return;
  }
  int lane_id = threadIdx.x % 32;

  using T = __nv_bfloat162;

  const auto *gate_row_ptr = gate_up_output_ptr + irow * num_col * 2;
  const auto *up_row_ptr = gate_row_ptr + num_col;
  auto *out_row_ptr = output_ptr + irow * num_col;

  int icol = it * 8;

  if (icol < num_col) {
    auto gate = to<float>(load<T, 4>(gate_row_ptr + icol));
    auto up = to<float>(load<T, 4>(up_row_ptr + icol));

    vec_t<float, 8> out;
#pragma unroll
    for (int i = 0; i < size(out); ++i) {
      out[i] = silu(gate[i]) * up[i];
    }

    // get max value per 128 elements and cal scale
    float thread_max = 0.f;
#pragma unroll
    for (int i = 0; i < size(out); i++) {
      if (fabsf(out[i]) > thread_max) {
        thread_max = fabsf(out[i]);
      }
    }
    float max = half_warp_reduce_max_down(thread_max);
    float scale = max / 448.0f;
    float inv_scale = 1.0f / (scale + 1e-8f);

    // quant
#pragma unroll
    for (int i = 0; i < size(out); ++i) {
      out[i] *= inv_scale;
    }

    // store output
    auto out_fp8 = to<__nv_fp8x4_e4m3>(out);
    store(out_row_ptr + icol, out_fp8);

    // store scale
    if (lane_id == 0 || lane_id == 16) {
      int igroup = -1;
      int offset = -1;
      get_group_id(irow, cu_num_tokens_per_group_ptr, num_group, igroup, offset);
      if (igroup >= 0 && offset >= 0) {
        auto *scale_addr = output_scale_ptr + cu_tiles_ptr[igroup] * ktile_m + offset +
                           icol / 128 * num_row_padded_size;
        store(scale_addr, scale);
      }
    }
  }
}

// input : gate + up
__global__ void masked_act_mul_and_blockwise_quant_kernel(
    __nv_fp8_e4m3 *output_ptr, float *output_scale_ptr, const __nv_bfloat16 *input_ptr,
    const int *num_per_expert_ptr, int num_total_tokens, int num_intermediate_size,
    int num_scale_size, int num_tokens_per_expert, cutlass::FastDivmod Block2YX,
    cutlass::FastDivmod Row2EandT, int num_block_row) {
  constexpr int kRows = 4;

  int iblockx;
  int iblocky;
  Block2YX(iblocky, iblockx, blockIdx.x);
  int lane_id = threadIdx.x % 32;

#pragma unroll 1
  for (int irow0 = iblocky * kRows; irow0 < num_total_tokens; irow0 += num_block_row * kRows) {
    int it = threadIdx.x + iblockx * blockDim.x;

#pragma unroll
    for (int i = 0; i < kRows; ++i) {
      int iexpert;
      int itoken;

      int irow = irow0 + i;

      Row2EandT(iexpert, itoken, irow);
      int num_tokens_curr_expert = num_per_expert_ptr[iexpert];
      if (itoken >= num_tokens_curr_expert) {
        continue;
      }

      const auto *gate_row_ptr = input_ptr + irow * (num_intermediate_size * 2);
      const auto *up_row_ptr = gate_row_ptr + num_intermediate_size;
      auto *output_row_ptr = output_ptr + irow * num_intermediate_size;
      auto *output_scale_row_ptr = output_scale_ptr + irow * num_scale_size;

      int icol = it * 8;
      if (icol < num_intermediate_size) {
        // 1. load gate an up
        auto gate = to<float>(load<__nv_bfloat162, 4>(gate_row_ptr + icol));
        auto up = to<float>(load<__nv_bfloat162, 4>(up_row_ptr + icol));
        decltype(gate) out;

        // 2. silu
#pragma unroll
        for (int i = 0; i < decltype(gate)::kNum; ++i) {
          out[i] = silu(gate[i]) * up[i];
        }

        // 3. get max value per 128 elements and cal scale
        float thread_max = 0.f;
#pragma unroll
        for (int i = 0; i < decltype(gate)::kNum; ++i) {
          if (fabsf(out[i]) > thread_max) {
            thread_max = fabsf(out[i]);
          }
        }
        float max = half_warp_reduce_max_down(thread_max);
        float scale = max / 448.0f;
        float inv_scale = 1.0f / (scale + 1e-8f);

        // 4. quant
#pragma unroll
        for (int i = 0; i < decltype(gate)::kNum; ++i) {
          out[i] *= inv_scale;
        }

        // 5. store output
        auto out_fp8 = to<__nv_fp8x4_e4m3>(out);
        store(output_row_ptr + icol, out_fp8);

        // 6. store output scale
        if (lane_id == 0 || lane_id == 16) {
          store(output_scale_row_ptr + icol / 128, scale);
        }
      }
    }  // for
  }  // irow0
}

}  // namespace kernels

void act_mul_and_quant_async(__nv_fp8_e4m3 *out_ptr, const __nv_bfloat16 *gate_up_ptr,
                             const float *scale_ptr, const int num_row, const int num_col,
                             bool use_bf16_mul, cudaStream_t stream) {
  act_mul_and_quant_async(out_ptr, gate_up_ptr, scale_ptr, nullptr, num_row, num_col, use_bf16_mul,
                          stream);
}

void act_mul_and_quant_async(__nv_fp8_e4m3 *out_ptr, const __nv_bfloat16 *gate_up_ptr,
                             const float *scale_ptr, const int *valid_row_range, const int num_row,
                             const int num_col, bool use_bf16_mul, cudaStream_t stream) {
  // num_col == 2128 x 2
  // gate + up

  int intermediate_size = num_col / 2;

  dim3 block(128);
  int num_col_block = (intermediate_size / 8 + block.x - 1) / block.x;
  cutlass::FastDivmod block1D22D(num_col_block);
  dim3 grid(num_row * num_col_block);

  if (use_bf16_mul) {
    kernels::act_mul_and_quant_kernel<true><<<grid, block, 0, stream>>>(
        out_ptr, gate_up_ptr, scale_ptr, valid_row_range, num_row, intermediate_size, block1D22D);
  } else {
    kernels::act_mul_and_quant_kernel<false><<<grid, block, 0, stream>>>(
        out_ptr, gate_up_ptr, scale_ptr, valid_row_range, num_row, intermediate_size, block1D22D);
  }
}

void masked_act_mul_and_quant_async(__nv_fp8_e4m3 *output_ptr, const __nv_bfloat16 *input_ptr,
                                    const float *scale_ptr, const int *num_per_expert_ptr,
                                    int num_total_tokens, int num_intermediate_size,
                                    int num_tokens_per_expert, cudaStream_t stream) {
  dim3 block(256);
  int num_block_col = (num_intermediate_size / 8 + block.x - 1) / block.x;

  int num_sm = get_sm_count();
  int num_block_hard = num_sm * 8;

  int num_block_row = num_block_hard / num_block_col;
  int num_block = num_block_row * num_block_col;
  dim3 grid(num_block);

  cutlass::FastDivmod Block2YX(num_block_col);
  cutlass::FastDivmod Row2EandT(num_tokens_per_expert);

  kernels::masked_act_mul_and_quant_kernel<<<grid, block, 0, stream>>>(
      output_ptr, input_ptr, scale_ptr, num_per_expert_ptr, num_total_tokens, num_intermediate_size,
      num_tokens_per_expert, Block2YX, Row2EandT, num_block_row);
}

void act_mul_and_blockwise_quant_async(void *output_ptr, void *output_scale_ptr,
                                       const void *input_ptr,
                                       const void *cu_num_tokens_per_group_ptr,
                                       const void *cu_tiles_ptr, const int num_row,
                                       const int num_row_padded_size, const int num_col,
                                       const int num_group, const int num_tokens_per_group_avg,
                                       cudaStream_t stream) {
  using Tin = __nv_bfloat16;
  using Tout = __nv_fp8_e4m3;
  int intermediate_size = num_col / 2;
  dim3 block(128);
  int num_block_per_row = (intermediate_size / 8 + block.x - 1) / block.x;
  cutlass::FastDivmod block1D22D(num_block_per_row);
  dim3 grid(num_row * num_block_per_row);
  int ktile_m = 0;
  if (num_tokens_per_group_avg <= 16) {
    ktile_m = 16;
  } else if (num_tokens_per_group_avg <= 32) {
    ktile_m = 32;
  } else if (num_tokens_per_group_avg <= 48) {
    ktile_m = 48;
  } else {
    ktile_m = 64;
  }
  kernels::act_mul_and_blockwise_quant_kernel<<<grid, block, 0, stream>>>(
      (Tin *)input_ptr, (Tout *)output_ptr, (float *)output_scale_ptr,
      (int *)cu_num_tokens_per_group_ptr, (int *)cu_tiles_ptr, num_row, num_row_padded_size,
      intermediate_size, num_group, ktile_m, block1D22D);
}

void masked_act_mul_and_blockwise_quant_async(__nv_fp8_e4m3 *output_ptr, float *output_scale_ptr,
                                              const __nv_bfloat16 *input_ptr,
                                              const int *num_per_expert_ptr, int num_total_tokens,
                                              int num_intermediate_size, int num_tokens_per_expert,
                                              cudaStream_t stream) {
  dim3 block(256);
  int num_block_col = (num_intermediate_size / 8 + block.x - 1) / block.x;

  int num_sm = get_sm_count();
  int num_block_hard = num_sm * 8;

  int num_block_row = num_block_hard / num_block_col;
  int num_block = num_block_row * num_block_col;
  dim3 grid(num_block);

  cutlass::FastDivmod Block2YX(num_block_col);
  cutlass::FastDivmod Row2EandT(num_tokens_per_expert);

  kernels::masked_act_mul_and_blockwise_quant_kernel<<<grid, block, 0, stream>>>(
      output_ptr, output_scale_ptr, input_ptr, num_per_expert_ptr, num_total_tokens,
      num_intermediate_size, num_intermediate_size / 128, num_tokens_per_expert, Block2YX,
      Row2EandT, num_block_row);
}

}  // namespace activation
}  // namespace hpc
