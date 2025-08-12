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

}  // namespace activation
}  // namespace hpc
