#include <cuda.h>
#include <cuda_bf16.h>

#include "src/mamba/conv1d_state.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace mamba {

namespace kernels {

template <int kNumElementsPerThread = 8, int kNumThreadsPerBlock = 128>
__global__ void causal_conv1d_update_kernel(
    __nv_bfloat16 *zxbcdt_ptr, __nv_bfloat16 *conv_state_ptr,
    const __nv_bfloat16 *weight_ptr, const __nv_bfloat16 *bias_ptr,
    const int *indices_ptr, int state_len, int d_conv, int conv_dim,
    int d_inner, int num_head) {
  int tid = threadIdx.x;
  int bidy = blockIdx.y;

  constexpr int kNumElementsPerBlock =
      kNumElementsPerThread * kNumThreadsPerBlock;

  int icol = bidy * kNumElementsPerBlock + tid * kNumElementsPerThread;
  if (icol >= conv_dim) {
    return;
  }

  int ibatch = blockIdx.x;
  int istate = indices_ptr[ibatch];

  auto *cur_batch_conv_state_ptr =
      conv_state_ptr + istate * (state_len * conv_dim);
  auto *cur_batch_zxbcdt_ptr =
      zxbcdt_ptr + ibatch * (d_inner + conv_dim + num_head) + d_inner;

  vec_t<float, kNumElementsPerThread> sum;
#pragma unroll
  for (int i = 0; i < kNumElementsPerThread; i++) {
    sum[i] = 0;
  }

  // past_conv_states * weight
  auto *pre_conv_state_ptr = cur_batch_conv_state_ptr;
  for (int i = 0; i < state_len; i++) {
    auto *cur_weight_ptr = weight_ptr + i * conv_dim;
    auto *cur_conv_state_ptr = cur_batch_conv_state_ptr + i * conv_dim;

    vec_t<float, 8> weight =
        to<float>(load<__nv_bfloat162, 4>(&cur_weight_ptr[icol]));
    vec_t<float, 8> conv_states =
        to<float>(load<__nv_bfloat162, 4>(&cur_conv_state_ptr[icol]));

    if (i > 0) {
      // update conv_state
      store(&pre_conv_state_ptr[icol], to<__nv_bfloat16>(conv_states));
      pre_conv_state_ptr = cur_conv_state_ptr;
    }
#pragma unroll
    for (int e = 0; e < kNumElementsPerThread; e++) {
      sum[e] += conv_states[e] * weight[e];
    }
  }

  auto *cur_weight_ptr = weight_ptr + state_len * conv_dim;
  vec_t<float, 8> weight =
      to<float>(load<__nv_bfloat162, 4>(&cur_weight_ptr[icol]));
  vec_t<float, 8> zxbcdt =
      to<float>(load<__nv_bfloat162, 4>(&cur_batch_zxbcdt_ptr[icol]));

  // update conv_state
  store(&pre_conv_state_ptr[icol], to<__nv_bfloat16>(zxbcdt));

// xbc * weight
#pragma unroll
  for (int e = 0; e < kNumElementsPerThread; e++) {
    sum[e] += zxbcdt[e] * weight[e];
  }

  // add bias + silu
  vec_t<float, 8> bias = to<float>(load<__nv_bfloat162, 4>(&bias_ptr[icol]));

#pragma unroll
  for (int e = 0; e < kNumElementsPerThread; e++) {
    sum[e] += bias[e];
    sum[e] = silu(sum[e]);
  }

  // store output
  store(&cur_batch_zxbcdt_ptr[icol], to<__nv_bfloat16>(sum));
}
}  // namespace kernels

void causal_conv1d_update_async(
    __nv_bfloat16 *zxbcdt_ptr, __nv_bfloat16 *conv_state_ptr,
    const __nv_bfloat16 *weight_ptr, const __nv_bfloat16 *bias_ptr,
    const int *indices_ptr, int num_batch, int state_len, int d_conv,
    int conv_dim, int d_inner, int num_head, cudaStream_t stream) {
  constexpr int kNumElementsPerThread = 8;
  constexpr int kNumThreadsPerBlock = 128;
  constexpr int kNumElementsPerBlock =
      kNumElementsPerThread * kNumThreadsPerBlock;

  dim3 block(kNumThreadsPerBlock);

  int num_blocks = (conv_dim + kNumElementsPerBlock - 1) / kNumElementsPerBlock;
  dim3 grid(num_batch, num_blocks);

  kernels::causal_conv1d_update_kernel<kNumElementsPerThread,
                                       kNumThreadsPerBlock>
      <<<grid, block, 0, stream>>>(zxbcdt_ptr, conv_state_ptr, weight_ptr,
                                   bias_ptr, indices_ptr, state_len, d_conv,
                                   conv_dim, d_inner, num_head);
}

}  // namespace mamba
}  // namespace hpc
