// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <cuda_bf16.h>
#include <stdio.h>

#include "src/mamba/conv1d_state.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace mamba {
namespace kernels {

template <int kNumElementsPerThread, int kNumThreadsPerBlock>
__global__ void causal_conv1d_update_kernel(__nv_bfloat16 *zxbcdt_ptr,
                                            __nv_bfloat16 *conv_state_ptr,
                                            const __nv_bfloat16 *weight_ptr,
                                            const __nv_bfloat16 *bias_ptr, const int *indices_ptr,
                                            int state_len, int d_conv, int conv_dim, int d_inner,
                                            int num_head) {
  int tid = threadIdx.x;
  int bidy = blockIdx.y;

  constexpr int kNumElementsPerBlock = kNumElementsPerThread * kNumThreadsPerBlock;

  int icol = bidy * kNumElementsPerBlock + tid * kNumElementsPerThread;
  if (icol >= conv_dim) {
    return;
  }

  int ibatch = blockIdx.x;
  int istate = indices_ptr[ibatch];

  auto *cur_batch_conv_state_ptr = conv_state_ptr + istate * (state_len * conv_dim);
  auto *cur_batch_zxbcdt_ptr = zxbcdt_ptr + ibatch * (d_inner + conv_dim + num_head) + d_inner;

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

    vec_t<float, kNumElementsPerThread> weight =
        to<float>(load<__nv_bfloat162, kNumElementsPerThread / 2>(&cur_weight_ptr[icol]));
    vec_t<float, kNumElementsPerThread> conv_states =
        to<float>(load<__nv_bfloat162, kNumElementsPerThread / 2>(&cur_conv_state_ptr[icol]));

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
  vec_t<float, kNumElementsPerThread> weight =
      to<float>(load<__nv_bfloat162, kNumElementsPerThread / 2>(&cur_weight_ptr[icol]));
  vec_t<float, kNumElementsPerThread> zxbcdt =
      to<float>(load<__nv_bfloat162, kNumElementsPerThread / 2>(&cur_batch_zxbcdt_ptr[icol]));

  // update conv_state
  store(&pre_conv_state_ptr[icol], to<__nv_bfloat16>(zxbcdt));

// xbc * weight
#pragma unroll
  for (int e = 0; e < kNumElementsPerThread; e++) {
    sum[e] += zxbcdt[e] * weight[e];
  }

  // add bias + silu
  vec_t<float, kNumElementsPerThread> bias =
      to<float>(load<__nv_bfloat162, kNumElementsPerThread / 2>(&bias_ptr[icol]));

#pragma unroll
  for (int e = 0; e < kNumElementsPerThread; e++) {
    sum[e] += bias[e];
    sum[e] = silu(sum[e]);
  }

  // store output
  store(&cur_batch_zxbcdt_ptr[icol], to<__nv_bfloat16>(sum));
}

template <int kNumElementsPerThread, int kNumThreadsPerBlock>
__global__ void causal_conv1d_update_with_spec_kernel(
    __nv_bfloat16 *zxbcdt_ptr, __nv_bfloat16 *conv_state_ptr, const __nv_bfloat16 *weight_ptr,
    const __nv_bfloat16 *bias_ptr, const int *indices_ptr, int state_len, int d_conv, int conv_dim,
    int d_inner, int num_head, int num_spec_tokens, const int *num_accept_tokens_ptr) {
  int tid = threadIdx.x;
  int bidy = blockIdx.y;

  constexpr int kNumElementsPerBlock = kNumElementsPerThread * kNumThreadsPerBlock;

  int icol = bidy * kNumElementsPerBlock + tid * kNumElementsPerThread;
  if (icol >= conv_dim) {
    return;
  }

  int ibatch = blockIdx.x;
  int i_state = indices_ptr[ibatch];
  int num_accept_tokens = num_accept_tokens_ptr[ibatch];

  auto *cur_batch_conv_state_ptr = conv_state_ptr + i_state * (state_len * conv_dim);
  auto *cur_zxbcdt_ptr =
      zxbcdt_ptr + ibatch * (d_inner + conv_dim + num_head) * num_spec_tokens + d_inner;

  vec_t<float, kNumElementsPerThread> sum;

  auto *pre_conv_state_ptr = cur_batch_conv_state_ptr;
  auto *cur_conv_state_ptr = cur_batch_conv_state_ptr + (num_accept_tokens - 1) * conv_dim;
  for (int s = 0; s < num_spec_tokens; s++) {
// init regsiter
#pragma unroll
    for (int i = 0; i < kNumElementsPerThread; i++) {
      sum[i] = 0;
    }

    // past_conv_states * weight
    for (int i = 0; i < d_conv - 1; i++) {
      auto *cur_weight_ptr = weight_ptr + i * conv_dim;
      vec_t<float, kNumElementsPerThread> weight =
          to<float>(load<__nv_bfloat162, kNumElementsPerThread / 2>(&cur_weight_ptr[icol]));
      vec_t<float, kNumElementsPerThread> conv_states =
          to<float>(load<__nv_bfloat162, kNumElementsPerThread / 2>(&cur_conv_state_ptr[icol]));

      if (s == 0 && i > 0) {
        // update conv_state
        store(&pre_conv_state_ptr[icol], to<__nv_bfloat16>(conv_states));
        pre_conv_state_ptr += conv_dim;
      }
#pragma unroll
      for (int e = 0; e < kNumElementsPerThread; e++) {
        sum[e] += conv_states[e] * weight[e];
      }
      cur_conv_state_ptr += conv_dim;
    }

    auto *cur_weight_ptr = weight_ptr + (d_conv - 1) * conv_dim;
    vec_t<float, kNumElementsPerThread> weight =
        to<float>(load<__nv_bfloat162, kNumElementsPerThread / 2>(&cur_weight_ptr[icol]));
    vec_t<float, kNumElementsPerThread> zxbcdt =
        to<float>(load<__nv_bfloat162, kNumElementsPerThread / 2>(&cur_zxbcdt_ptr[icol]));

    // update conv_state
    store(&pre_conv_state_ptr[icol], to<__nv_bfloat16>(zxbcdt));
    pre_conv_state_ptr += conv_dim;

// xbc * weight
#pragma unroll
    for (int e = 0; e < kNumElementsPerThread; e++) {
      sum[e] += zxbcdt[e] * weight[e];
    }

    // add bias + silu
    vec_t<float, kNumElementsPerThread> bias =
        to<float>(load<__nv_bfloat162, kNumElementsPerThread / 2>(&bias_ptr[icol]));
#pragma unroll
    for (int e = 0; e < kNumElementsPerThread; e++) {
      sum[e] += bias[e];
      sum[e] = silu(sum[e]);
    }

    // store output
    store(&cur_zxbcdt_ptr[icol], to<__nv_bfloat16>(sum));

    // restore ptr
    cur_weight_ptr = weight_ptr;
    cur_conv_state_ptr = cur_batch_conv_state_ptr;
    cur_zxbcdt_ptr += d_inner + conv_dim + num_head;
  }
}

template <int kNumElementsPerThread, int kNumThreadsPerBlock, int kDConv, int kChunkSize,
          int kTileL>
__global__ void causal_conv1d_prefill_stage1_kernel(
    __nv_bfloat16 *middle_y_ptr, __nv_bfloat16 *y_ptr, __nv_bfloat16 *zxbcdt_ptr,
    __nv_bfloat16 *conv_state_ptr, const __nv_bfloat16 *weight_ptr, const __nv_bfloat16 *bias_ptr,
    const int *indices_ptr, const int *split_metadata, const float *x_scale_ptr,
    const float *y_scale_ptr, int batch_size, int total_padded_seqlen, int state_len, int conv_dim,
    int d_inner, int num_head) {
  int tid = threadIdx.x;
  int itile = blockIdx.y;
  int bidy = blockIdx.z;
  constexpr int kTilePerChunk = kChunkSize / kTileL;
  const int *cu_seqlens = split_metadata;
  const int *cu_padded_seqlens = split_metadata + batch_size + 1;
  const int *cu_chunks = split_metadata + 2 * (batch_size + 1);
  const int *seqlens = split_metadata + 3 * (batch_size + 1);
  const int *nchunks = split_metadata + 5 * (batch_size + 1);
  int ichunk = blockIdx.x;
  int ibatch = 0;
  int ichunk_in_batch = 0;
  for (int i = 1; i < batch_size + 1; i++) {
    if (ichunk < cu_chunks[i]) {
      ibatch = i - 1;
      ichunk_in_batch = ichunk - cu_chunks[i - 1];
      break;
    }
  }

  constexpr int kNumElementsPerBlock = kNumElementsPerThread * kNumThreadsPerBlock;

  int icol = bidy * kNumElementsPerBlock + tid * kNumElementsPerThread;
  if (icol >= conv_dim) {
    return;
  }

  int istate = indices_ptr[ibatch];
  int seqlen = seqlens[ibatch];
  int cu_seqlen = cu_seqlens[ibatch];
  int head_dim = d_inner / num_head;
  int ihead = icol / head_dim;
  int nchunk = nchunks[ibatch];
  int conv_state_stride = state_len * conv_dim;
  int itile_in_batch = ichunk_in_batch * kTilePerChunk + itile;
  int ntile = nchunk * kTilePerChunk;
  int nheadsxheaddim = num_head * head_dim;

  auto *cur_batch_conv_state_ptr = conv_state_ptr + istate * conv_state_stride;
  const int &row_stride = (d_inner + conv_dim + num_head);
  auto *xbc_tile_ptr = zxbcdt_ptr + (cu_seqlen + itile_in_batch * kTileL) * row_stride + d_inner;
  auto *y_tile_ptr = y_ptr + (cu_seqlen + itile_in_batch * kTileL) * nheadsxheaddim;
  auto *middle_y_tile_ptr =
      middle_y_ptr + (ichunk * kTilePerChunk + itile) * (kDConv - 1) * conv_dim;
  auto *x_scale_tile_ptr = icol < d_inner ? x_scale_ptr + ihead * total_padded_seqlen +
                                                cu_padded_seqlens[ibatch] + itile_in_batch * kTileL
                                          : nullptr;
  auto *y_scale_chunk_ptr = icol < d_inner
                                ? y_scale_ptr + ihead * total_padded_seqlen +
                                      cu_padded_seqlens[ibatch] + ichunk_in_batch * kChunkSize
                                : nullptr;
  auto *y_scale_tile_ptr = icol < d_inner ? y_scale_chunk_ptr + itile * kTileL : nullptr;
  int last_ys_pos = umin(kChunkSize, seqlen - ichunk_in_batch * kChunkSize) - 1;
  float last_y_scale = y_scale_tile_ptr ? y_scale_chunk_ptr[last_ys_pos] : 0.f;
  constexpr int kStage = 2;
  constexpr int kXbcSize = kDConv + kStage - 1;
  vec_t<float, kNumElementsPerThread> xbc[kXbcSize];
  vec_t<float, kNumElementsPerThread> weight[kDConv];
  vec_t<float, kNumElementsPerThread> bias =
      to<float>(load<__nv_bfloat162, kNumElementsPerThread / 2>(&bias_ptr[icol]));

  // load weight
#pragma unroll
  for (int i = 0; i < kDConv; i++) {
    weight[i] = to<float>(
        load<__nv_bfloat162, kNumElementsPerThread / 2>(weight_ptr + i * conv_dim + icol));
  }

  vec_t<float, kNumElementsPerThread> sum;

  if (itile_in_batch > 0) {
#pragma unroll
    for (int irow = -kDConv + 1; irow < 0; irow++) {
      int iglobal_row = itile_in_batch * kTileL + irow;
      if (iglobal_row >= seqlen) {
        break;
      }
      xbc[irow + kDConv - 1] = to<float>(
          load<__nv_bfloat162, kNumElementsPerThread / 2>(xbc_tile_ptr + irow * row_stride + icol));
    }
  } else {
#pragma unroll
    for (int e = 0; e < kDConv; e++) {
#pragma unroll
      for (int i = 0; i < kNumElementsPerThread; i++) {
        xbc[e][i] = 0;
      }
    }
  }

#pragma unroll
  for (int irow = 0; irow < kTileL; irow += kStage) {
#pragma unroll
    for (int istage = 0; istage < kStage; istage++) {
      int iglobal_row = itile_in_batch * kTileL + irow + istage;
      if (iglobal_row >= seqlen) {
        break;
      }
      xbc[(irow + kDConv - 1 + istage) % kXbcSize] =
          to<float>(load<__nv_bfloat162, kNumElementsPerThread / 2>(
              xbc_tile_ptr + (irow + istage) * row_stride + icol));
    }

    vec_t<float, kNumElementsPerThread> x[kStage];
#pragma unroll
    for (int istage = 0; istage < kStage; istage++) {
      int iglobal_row = itile_in_batch * kTileL + irow + istage;
      if (iglobal_row >= seqlen) {
        break;
      }
      if (iglobal_row >= seqlen - kDConv + 1) {
        store(cur_batch_conv_state_ptr + (iglobal_row + kDConv - 1 - seqlen) * conv_dim + icol,
              to<__nv_bfloat16>(xbc[(irow + istage + kDConv - 1) % kXbcSize]));
      }
#pragma unroll
      for (int i = 0; i < kNumElementsPerThread; i++) {
        sum[i] = 0;
#pragma unroll
        for (int e = 0; e < kDConv; e++) {
          sum[i] += xbc[(irow + istage + e) % kXbcSize][i] * weight[e][i];
        }
        x[istage][i] = silu(sum[i] + bias[i]);
      }
      if (itile_in_batch == ntile - 1 || irow + istage < kTileL - kDConv + 1) {
        store(xbc_tile_ptr + (irow + istage) * row_stride + icol, to<__nv_bfloat16>(x[istage]));
      } else {
        store(middle_y_tile_ptr + (irow + istage - kTileL + kDConv - 1) * conv_dim + icol,
              to<__nv_bfloat16>(x[istage]));
      }
      if (x_scale_tile_ptr && y_scale_tile_ptr) {
#pragma unroll
        for (int i = 0; i < kNumElementsPerThread; i++) {
          x[istage][i] = x[istage][i] * x_scale_tile_ptr[istage + irow];
          x[istage][i] *= expf_ftz(last_y_scale - y_scale_tile_ptr[istage + irow]);
        }
        store(y_tile_ptr + (irow + istage) * nheadsxheaddim + icol, to<__nv_bfloat16>(x[istage]));
      }
    }
  }
}

// NOLINTBEGIN
template <int kNumElementsPerThread, int kNumThreadsPerBlock, int kDConv, int kChunkSize,
          int kTileL>
// NOLINTEND
__global__ void causal_conv1d_prefill_stage2_kernel(__nv_bfloat16 *middle_y_ptr,
                                                    __nv_bfloat16 *zxbcdt_ptr,
                                                    const int *split_metadata, int batch_size,
                                                    int conv_dim, int d_inner, int num_head) {
  int tid = threadIdx.x;
  int bidy = blockIdx.z;
  int itile = blockIdx.y;
  const int *cu_seqlens = split_metadata;
  const int *cu_chunks = split_metadata + 2 * (batch_size + 1);
  const int *seqlens = split_metadata + 3 * (batch_size + 1);
  const int *nchunks = split_metadata + 5 * (batch_size + 1);

  int ichunk = blockIdx.x;
  int ibatch = 0;
  int ichunk_in_batch = 0;
  for (int i = 1; i < batch_size + 1; i++) {
    if (ichunk < cu_chunks[i]) {
      ibatch = i - 1;
      ichunk_in_batch = ichunk - cu_chunks[i - 1];
      break;
    }
  }

  constexpr int kNumElementsPerBlock = kNumElementsPerThread * kNumThreadsPerBlock;

  int icol = bidy * kNumElementsPerBlock + tid * kNumElementsPerThread;
  if (icol >= conv_dim) {
    return;
  }

  constexpr int kTilePerChunk = kChunkSize / kTileL;
  int seqlen = seqlens[ibatch];
  int cu_seqlen = cu_seqlens[ibatch];
  int nchunk = nchunks[ibatch];
  int itile_in_batch = ichunk_in_batch * kTilePerChunk + itile;
  int ntile = nchunk * kTilePerChunk;

  const int &row_stride = (d_inner + conv_dim + num_head);
  int iglobal_row = cu_seqlen + itile_in_batch * kTileL;
  auto *xbc_tile_ptr = zxbcdt_ptr + iglobal_row * row_stride + d_inner;
  auto *middle_y_tile_ptr =
      middle_y_ptr + (ichunk * kTilePerChunk + itile) * (kDConv - 1) * conv_dim;

  vec_t<__nv_bfloat162, kNumElementsPerThread / 2> xbc[kDConv - 1];
  if (itile_in_batch != ntile - 1) {
#pragma unroll
    for (int e = 0; e < kDConv - 1; e++) {
      xbc[e] =
          load<__nv_bfloat162, kNumElementsPerThread / 2>(middle_y_tile_ptr + e * conv_dim + icol);
    }

#pragma unroll
    for (int e = 0; e < kDConv - 1; e++) {
      if (iglobal_row + kTileL - kDConv + 1 + e < cu_seqlen + seqlen) {
        store(xbc_tile_ptr + (kTileL - kDConv + 1 + e) * row_stride + icol, xbc[e]);
      }
    }
  }
}

}  // namespace kernels

void causal_conv1d_update_async(__nv_bfloat16 *zxbcdt_ptr, __nv_bfloat16 *conv_state_ptr,
                                const __nv_bfloat16 *weight_ptr, const __nv_bfloat16 *bias_ptr,
                                const int *indices_ptr, int num_batch, int state_len, int d_conv,
                                int conv_dim, int d_inner, int num_head, int num_spec_tokens,
                                const int *num_accept_tokens_ptr, cudaStream_t stream) {
  if (num_accept_tokens_ptr) {
    if (num_head == 4) {
      constexpr int kNumElementsPerThread = 4;
      constexpr int kNumThreadsPerBlock = 128;
      constexpr int kNumElementsPerBlock = kNumElementsPerThread * kNumThreadsPerBlock;

      dim3 block(kNumThreadsPerBlock);

      int num_blocks = (conv_dim + kNumElementsPerBlock - 1) / kNumElementsPerBlock;
      dim3 grid(num_batch, num_blocks);

      kernels::causal_conv1d_update_with_spec_kernel<kNumElementsPerThread, kNumThreadsPerBlock>
          <<<grid, block, 0, stream>>>(zxbcdt_ptr, conv_state_ptr, weight_ptr, bias_ptr,
                                       indices_ptr, state_len, d_conv, conv_dim, d_inner, num_head,
                                       num_spec_tokens, num_accept_tokens_ptr);
    } else if (num_head == 8) {
      constexpr int kNumElementsPerThread = 8;
      constexpr int kNumThreadsPerBlock = 128;
      constexpr int kNumElementsPerBlock = kNumElementsPerThread * kNumThreadsPerBlock;

      dim3 block(kNumThreadsPerBlock);

      int num_blocks = (conv_dim + kNumElementsPerBlock - 1) / kNumElementsPerBlock;
      dim3 grid(num_batch, num_blocks);

      kernels::causal_conv1d_update_with_spec_kernel<kNumElementsPerThread, kNumThreadsPerBlock>
          <<<grid, block, 0, stream>>>(zxbcdt_ptr, conv_state_ptr, weight_ptr, bias_ptr,
                                       indices_ptr, state_len, d_conv, conv_dim, d_inner, num_head,
                                       num_spec_tokens, num_accept_tokens_ptr);
    }

  } else {
    if (num_head == 4) {
      constexpr int kNumElementsPerThread = 4;
      constexpr int kNumThreadsPerBlock = 128;
      constexpr int kNumElementsPerBlock = kNumElementsPerThread * kNumThreadsPerBlock;

      dim3 block(kNumThreadsPerBlock);

      int num_blocks = (conv_dim + kNumElementsPerBlock - 1) / kNumElementsPerBlock;
      dim3 grid(num_batch, num_blocks);

      kernels::causal_conv1d_update_kernel<kNumElementsPerThread, kNumThreadsPerBlock>
          <<<grid, block, 0, stream>>>(zxbcdt_ptr, conv_state_ptr, weight_ptr, bias_ptr,
                                       indices_ptr, state_len, d_conv, conv_dim, d_inner, num_head);
    } else if (num_head == 8) {
      constexpr int kNumElementsPerThread = 8;
      constexpr int kNumThreadsPerBlock = 128;
      constexpr int kNumElementsPerBlock = kNumElementsPerThread * kNumThreadsPerBlock;

      dim3 block(kNumThreadsPerBlock);

      int num_blocks = (conv_dim + kNumElementsPerBlock - 1) / kNumElementsPerBlock;
      dim3 grid(num_batch, num_blocks);

      kernels::causal_conv1d_update_kernel<kNumElementsPerThread, kNumThreadsPerBlock>
          <<<grid, block, 0, stream>>>(zxbcdt_ptr, conv_state_ptr, weight_ptr, bias_ptr,
                                       indices_ptr, state_len, d_conv, conv_dim, d_inner, num_head);
    }
  }
}

bool causal_conv1d_prefill_async(__nv_bfloat16 *middle_y_ptr, __nv_bfloat16 *y_ptr,
                                 __nv_bfloat16 *zxbcdt_ptr, __nv_bfloat16 *conv_state_ptr,
                                 const __nv_bfloat16 *weight_ptr, const __nv_bfloat16 *bias_ptr,
                                 const int *indices_ptr, const int *split_metadata_ptr,
                                 const float *x_scale_ptr, const float *y_scale_ptr, int num_batch,
                                 int total_chunks, int total_padded_seqlen, int state_len,
                                 int conv_dim, int d_inner, int num_head, int d_conv,
                                 int chunk_size, int tileL, cudaStream_t stream) {
  constexpr int kNumElementsPerThread = 8;
  constexpr int kNumThreadsPerBlock = 128;
  constexpr int kNumElementsPerBlock = kNumElementsPerThread * kNumThreadsPerBlock;

  constexpr int kDConv = 4;
  constexpr int kChunkSize = 256;
  constexpr int kTileL = 32;

  if (d_conv != kDConv || chunk_size != kChunkSize || tileL != kTileL) {
    printf("causal_conv1d_prefill_async Only support d_conv:%d, chunk_size:%d, tileL:%d\n", kDConv,
           kChunkSize, kTileL);
    return false;
  }

  dim3 block(kNumThreadsPerBlock);

  int num_blocks = (conv_dim + kNumElementsPerBlock - 1) / kNumElementsPerBlock;
  dim3 grid(total_chunks, kChunkSize / kTileL, num_blocks);

  kernels::causal_conv1d_prefill_stage1_kernel<kNumElementsPerThread, kNumThreadsPerBlock, kDConv,
                                               kChunkSize, kTileL><<<grid, block, 0, stream>>>(
      middle_y_ptr, y_ptr, zxbcdt_ptr, conv_state_ptr, weight_ptr, bias_ptr, indices_ptr,
      split_metadata_ptr, x_scale_ptr, y_scale_ptr, num_batch, total_padded_seqlen, state_len,
      conv_dim, d_inner, num_head);
  kernels::causal_conv1d_prefill_stage2_kernel<kNumElementsPerThread, kNumThreadsPerBlock, kDConv,
                                               kChunkSize, kTileL><<<grid, block, 0, stream>>>(
      middle_y_ptr, zxbcdt_ptr, split_metadata_ptr, num_batch, conv_dim, d_inner, num_head);

  return true;
}

}  // namespace mamba
}  // namespace hpc
