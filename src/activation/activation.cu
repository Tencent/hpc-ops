#include <cuda.h>
#include <stdio.h>

#include "src/activation/activation.h"

namespace hpc {
namespace activation {

namespace kernels {

__device__ __forceinline__ float expf_ftz(float x) {
  // e = 2^m
  // m = 1.4426950408889634

  const float m = 1.4426950408889634f;
  float r;
  asm volatile("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(x * m));
  return r;
}

__device__ __forceinline__ float rcpf_ftz(float x) {
  float r;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(x));
  return r;
}

// x / (1 + e^(-x))
__device__ __forceinline__ float silu(float x) {
  return x * rcpf_ftz(1.f + expf_ftz(-x));
}

__global__ void act_mul_and_quant_kernel(__nv_fp8_e4m3 *out_ptr,
                                         const __nv_bfloat16 *gate_ptr,
                                         const __nv_bfloat16 *up_ptr,
                                         const float *scale_ptr,
                                         const int num_row, const int num_col) {
  int it = threadIdx.x + blockIdx.x * blockDim.x;
  int irow = blockIdx.y;

  __nv_bfloat162 gate[4];
  __nv_bfloat162 up[4];

  float scale = scale_ptr[0];
  const auto *gate_row_ptr = gate_ptr + irow * num_col;
  const auto *up_row_ptr = up_ptr + irow * num_col;
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

    *((int2 *)(out_row_ptr + icol)) = out_2i;  // *((int2*)(&out_2i));
  }
}

}  // namespace kernels

void act_mul_and_quant_async(__nv_fp8_e4m3 *out_ptr,
                             const __nv_bfloat16 *gate_up_ptr,
                             const float *scale_ptr, const int num_row,
                             const int num_col, cudaStream_t stream) {
  // num_col == 2128 x 2
  // gate + up


  int intermediate_size = num_col / 2;
  auto gate_ptr = gate_up_ptr;
  auto up_ptr = gate_up_ptr + intermediate_size;

  dim3 block(128);
  dim3 grid((intermediate_size / 8 + block.x - 1) / block.x, num_row);

  printf("num_row = %d, intermediate_size = %d\n", num_row, intermediate_size);

  kernels::act_mul_and_quant_kernel<<<grid, block, 0, stream>>>(
      out_ptr, gate_ptr, up_ptr, scale_ptr, num_row, intermediate_size);
}

}  // namespace activation
}  // namespace hpc
