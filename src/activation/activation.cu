// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include "cutlass/fast_math.h"
#include "src/activation/activation.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace activation {

namespace kernels {

__global__ void act_mul_and_quant_kernel(__nv_fp8_e4m3 *out_ptr, const __nv_bfloat16 *gate_up_ptr,
                                         const float *scale_ptr, const int num_row,
                                         const int num_col, cutlass::FastDivmod block1D22D) {
  int iblockx;
  int iblocky;

  block1D22D(iblocky, iblockx, blockIdx.x);
  int it = threadIdx.x + iblockx * blockDim.x;

  int irow = iblocky;

  __nv_bfloat162 gate[4];
  __nv_bfloat162 up[4];

  float scale = scale_ptr[0];

  const auto *gate_row_ptr = gate_up_ptr + irow * num_col * 2;
  const auto *up_row_ptr = gate_row_ptr + num_col;
  const auto *out_row_ptr = out_ptr + irow * num_col;

  int icol = it * 8;

  if (icol < num_col) {
    *((int4 *)&gate) = *((int4 *)(gate_row_ptr + icol));
    *((int4 *)&up) = *((int4 *)(up_row_ptr + icol));

    float2 out[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float2 g2 = __bfloat1622float2(gate[i]);
      float2 u2 = __bfloat1622float2(up[i]);

      out[i].x = silu(g2.x) * u2.x;
      out[i].y = silu(g2.y) * u2.y;
    }

    float4 f1 = make_float4(out[0].x * scale, out[0].y * scale, out[1].x * scale, out[1].y * scale);
    float4 f2 = make_float4(out[2].x * scale, out[2].y * scale, out[3].x * scale, out[3].y * scale);

    __nv_fp8x4_e4m3 o1{f1};
    __nv_fp8x4_e4m3 o2{f2};

    int2 out_2i;

    out_2i.x = *(int *)(&o1);
    out_2i.y = *(int *)(&o2);

    *((int2 *)(out_row_ptr + icol)) = out_2i;
  }
}

// input : gate + up
__global__ void masked_act_mul_and_quant_kernel(
    __nv_fp8_e4m3 *output_ptr, const __nv_bfloat16 *input_ptr, const __nv_bfloat16 *scale_ptr,
    const int *num_per_expert_ptr, int num_total_tokens, int num_intermediate_size,
    int num_tokens_per_expert, cutlass::FastDivmod Block2YX, cutlass::FastDivmod Row2EandT,
    int num_block_row) {
  constexpr int kRows = 4;

  int iblockx;
  int iblocky;

  Block2YX(iblocky, iblockx, blockIdx.x);

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
      const auto *scale_row_ptr = scale_ptr;
      auto *output_row_ptr = output_ptr + irow * num_intermediate_size;

      int icol = it * 8;
      if (icol < num_intermediate_size) {
        auto gate = to<float>(load<__nv_bfloat162, 4>(gate_row_ptr + icol));
        auto up = to<float>(load<__nv_bfloat162, 4>(up_row_ptr + icol));
        auto scale = to<float>(load<__nv_bfloat162, 4>(scale_row_ptr + icol));
        decltype(gate) out;

#pragma unroll
        for (int i = 0; i < decltype(gate)::kNum; ++i) {
          out[i] = silu(gate[i]) * up[i] * scale[i];
        }

        auto out_fp8 = to<__nv_fp8x4_e4m3>(out);

        store(output_row_ptr + icol, out_fp8);
      }
    }  // for
  }  // irow0
}

}  // namespace kernels

void act_mul_and_quant_async(__nv_fp8_e4m3 *out_ptr, const __nv_bfloat16 *gate_up_ptr,
                             const float *scale_ptr, const int num_row, const int num_col,
                             cudaStream_t stream) {
  // num_col == 2128 x 2
  // gate + up

  int intermediate_size = num_col / 2;

  dim3 block(128);
  int num_col_block = (intermediate_size / 8 + block.x - 1) / block.x;
  cutlass::FastDivmod block1D22D(num_col_block);
  dim3 grid(num_row * num_col_block);

  kernels::act_mul_and_quant_kernel<<<grid, block, 0, stream>>>(
      out_ptr, gate_up_ptr, scale_ptr, num_row, intermediate_size, block1D22D);
}

void masked_act_mul_and_quant_async(__nv_fp8_e4m3 *output_ptr, const __nv_bfloat16 *input_ptr,
                                    const __nv_bfloat16 *scale_ptr, const int *num_per_expert_ptr,
                                    int num_total_tokens, int num_intermediate_size,
                                    int num_tokens_per_expert, cudaStream_t stream) {
  dim3 block(256);
  int num_block_col = (num_intermediate_size / 8 + block.x - 1) / block.x;

  int num_sm = 78;
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

}  // namespace activation
}  // namespace hpc
