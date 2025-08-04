#include <cuda.h>
#include <cuda_bf16.h>
#include <stdio.h>

#include "src/mamba/selective_state.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace mamba {

namespace kernels {

/*
 * one warp per row
 *
 * */

__global__ void selective_state_update_kernel(
    __nv_bfloat16 *out_ptr, float *ssm_states_ptr, const int *indices_ptr,
    const __nv_bfloat16 *z_ptr, const __nv_bfloat16 *x_ptr, const __nv_bfloat16 *b_ptr,
    const __nv_bfloat16 *c_ptr, const __nv_bfloat16 *dt_ptr, const float *AD_ptr,
    const float *bias_ptr, int batch_stride_zxbcdt, int num_head, int head_dim, int state_dim,
    int num_headxhead_dimxstate_dim, int head_dimxstate_dim, int num_headxhead_dim,
    int num_groupxstate_dim) {
  int idx = threadIdx.x;

  int ilane = idx % 32;
  int iwarp = idx / 32;

  int ibatch = blockIdx.z;
  int ihead = blockIdx.y;
  int igroup = ihead / 4;  // hard code: heads per group = 4
  int irow = iwarp + blockIdx.x * 4;
  int icol = ilane;

  if (irow >= head_dim) {
    return;
  }

  int islot = indices_ptr[ibatch];

  auto *ssm_row = ssm_states_ptr + (islot * (num_headxhead_dimxstate_dim) +
                                    ihead * (head_dimxstate_dim) + irow * state_dim + icol * 4);
  auto *out_row = out_ptr + (ibatch * num_headxhead_dim + ihead * head_dim + irow);

  const auto *z_row = z_ptr + (ibatch * batch_stride_zxbcdt + ihead * head_dim + irow);
  const auto *x_row = x_ptr + (ibatch * batch_stride_zxbcdt + ihead * head_dim + irow);
  const auto *b_row = b_ptr + (ibatch * batch_stride_zxbcdt + igroup * state_dim + icol * 4);
  const auto *c_row = c_ptr + (ibatch * batch_stride_zxbcdt + igroup * state_dim + icol * 4);
  const auto *dt_row = dt_ptr + (ibatch * batch_stride_zxbcdt + ihead);

  auto AD = load<float, 2>(AD_ptr + ihead * 2);
  float A = AD[0];
  float D = AD[1];

  float bias = bias_ptr[ihead];

  float x = __bfloat162float(x_row[0]);

  auto B = to<float>(load<__nv_bfloat162, 2>(b_row));
  float dt = __bfloat162float(dt_row[0]);
  auto C = to<float>(load<__nv_bfloat162, 2>(c_row));

  auto curr_ssm = load<float, 4>(ssm_row);
  float z = __bfloat162float(z_row[0]);

  vec_t<float, 4> next_ssm;

  float s = softplus(dt + bias);
  float xs = x * s;

  // xb[head_dim, state_dim] =  x[head_dim, 1] x dB[1, state_dim]
  next_ssm[0] = xs * B[0];
  next_ssm[1] = xs * B[1];
  next_ssm[2] = xs * B[2];
  next_ssm[3] = xs * B[3];

  float dA = expf_ftz(A * s);

  // new_ssm = xb + curr_ssm * dA;
  next_ssm[0] = next_ssm[0] + curr_ssm[0] * dA;
  next_ssm[1] = next_ssm[1] + curr_ssm[1] * dA;
  next_ssm[2] = next_ssm[2] + curr_ssm[2] * dA;
  next_ssm[3] = next_ssm[3] + curr_ssm[3] * dA;

  // output ssm
  store(ssm_row, next_ssm);

  // reduce
  float sum = 0.f;
  sum += next_ssm[0] * C[0];
  sum += next_ssm[1] * C[1];
  sum += next_ssm[2] * C[2];
  sum += next_ssm[3] * C[3];

  sum = warp_reduce_sum_down(sum);

  // only one thread work
  if (icol == 0) {
    float out = sum + x * D;
    out = out * silu(z);

    // output y
    out_row[0] = __float2bfloat16(out);
  }
}

}  // namespace kernels

void selective_state_update_async(__nv_bfloat16 *out_ptr, float *ssm_states_ptr,
                                  const int *indices_ptr, const __nv_bfloat16 *zxbcdt_ptr,
                                  const float *AD_ptr, const float *bias_ptr, int num_batch,
                                  int num_head, int head_dim, int num_group, int state_dim,
                                  cudaStream_t stream) {
  dim3 block(128);
  int num_row = (head_dim + 3) / 4;
  dim3 grid(num_row, num_head, num_batch);

  int num_headxhead_dimxstate_dim = num_head * head_dim * state_dim;
  int head_dimxstate_dim = head_dim * state_dim;
  int num_headxhead_dim = num_head * head_dim;
  int num_groupxstate_dim = num_group * state_dim;

  // zxbcdt
  //      z            x                b               c          dt
  // -----------|-------------|=================|==============|--------|
  //   head_dim     head_dim       state_dim       state_dim

  auto *z_ptr = zxbcdt_ptr;
  auto *x_ptr = zxbcdt_ptr + num_headxhead_dim;
  auto *b_ptr = x_ptr + num_headxhead_dim;
  auto *c_ptr = b_ptr + num_groupxstate_dim;
  auto *dt_ptr = c_ptr + num_groupxstate_dim;

  int batch_stride_zxbcdt = num_headxhead_dim * 2 + num_groupxstate_dim * 2 + num_head;

  kernels::selective_state_update_kernel<<<grid, block, 0, stream>>>(
      out_ptr, ssm_states_ptr, indices_ptr, z_ptr, x_ptr, b_ptr, c_ptr, dt_ptr, AD_ptr, bias_ptr,
      batch_stride_zxbcdt, num_head, head_dim, state_dim, num_headxhead_dimxstate_dim,
      head_dimxstate_dim, num_headxhead_dim, num_groupxstate_dim);
}

}  // namespace mamba
}  // namespace hpc
