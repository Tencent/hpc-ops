// Copyright 2025 hpc-ops authors

#include <cuda.h>

#include "src/compressor/compressor.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace compressor {

namespace kernels {

template <int kRatio = 4, int kHeadDim = 128, int kTokensPerBatch = 1>
__global__ void c4_kv_compressor_kernel(float* y_ptr, const float* kv_ptr, const float* score_ptr,
                                        const float* ape_ptr, float* kv_states_ptr,
                                        float* score_state_ptr, const int* state_idx_ptr,
                                        const int* start_pos_ptr,
                                        const int* cu_compress_seqlens_ptr, int stride) {
  constexpr int kN = 16 / sizeof(float);
  constexpr int kDoubleRatio = kRatio * 2;
  constexpr int kDoubleHeadDim = kHeadDim * 2;

  int idx = threadIdx.x;
  int bid = blockIdx.x;
  int batch = bid;

  int state_idx = state_idx_ptr[batch];

  auto cur_kv_state_ptr = kv_states_ptr + state_idx * kDoubleRatio * kDoubleHeadDim;
  auto cur_score_state_ptr = score_state_ptr + state_idx * kDoubleRatio * kDoubleHeadDim;
  auto cur_kv_ptr = kv_ptr + (batch * kTokensPerBatch) * stride;
  auto cur_score_ptr = score_ptr + (batch * kTokensPerBatch) * stride;

  int start_pos = start_pos_ptr[batch];

#pragma unroll
  for (int i = 0; i < kTokensPerBatch; i++) {
    // 1.store state
    start_pos += i;
    int pos = start_pos % kRatio;
    bool should_compress = (start_pos + 1) % kRatio == 0;

    for (int col = idx * kN; col < kDoubleHeadDim; col += blockDim.x * kN) {
      auto kv = load<float, kN>(cur_kv_ptr + i * stride + col);
      auto score = load<float, kN>(cur_score_ptr + i * stride + col);
      auto ape = load<float, kN>(ape_ptr + pos * kDoubleHeadDim + col);
#pragma unroll
      for (int i = 0; i < kN; i++) {
        score[i] += ape[i];
      }
      store(cur_kv_state_ptr + (kRatio + pos) * kDoubleHeadDim + col, kv);
      store(cur_score_state_ptr + (kRatio + pos) * kDoubleHeadDim + col, score);
    }

    // 2.compress and move states
    if (should_compress) {
      // 2.1.compress
      const int col = idx * kN;
      vec_t<float, kN> exp_sum;
      vec_t<float, kN> sum;
      vec_t<float, kN> max;
      // init
#pragma unroll
      for (int i = 0; i < kN; i++) {
        exp_sum[i] = 0;
        sum[i] = 0;
        max[i] = -std::numeric_limits<float>::infinity();
      }

#pragma unroll
      // online softmax
      for (int i = 0; i < kRatio; i++) {
        auto kv = load<float, kN>(cur_kv_state_ptr + i * kDoubleHeadDim + col);
        auto score = load<float, kN>(cur_score_state_ptr + i * kDoubleHeadDim + col);
#pragma unroll
        for (int i = 0; i < kN; i++) {
          float new_max = fmaxf(max[i], score[i]);
          float scale = expf_ftz(max[i] - new_max);
          float w = expf_ftz(score[i] - new_max);
          exp_sum[i] = exp_sum[i] * scale + w;
          sum[i] = sum[i] * scale + kv[i] * w;
          max[i] = new_max;
        }
      }
#pragma unroll
      for (int i = 0; i < kRatio; i++) {
        auto kv = load<float, kN>(cur_kv_state_ptr + kRatio * kDoubleHeadDim + kHeadDim +
                                  i * kDoubleHeadDim + col);
        auto score = load<float, kN>(cur_score_state_ptr + kRatio * kDoubleHeadDim + kHeadDim +
                                     i * kDoubleHeadDim + col);
#pragma unroll
        for (int i = 0; i < kN; i++) {
          float new_max = fmaxf(max[i], score[i]);
          float scale = expf_ftz(max[i] - new_max);
          float w = expf_ftz(score[i] - new_max);
          exp_sum[i] = exp_sum[i] * scale + w;
          sum[i] = sum[i] * scale + kv[i] * w;
          max[i] = new_max;
        }
      }
#pragma unroll
      for (int i = 0; i < kN; i++) {
        float inv_exp_sum = 1.f / exp_sum[i];
        sum[i] = sum[i] * inv_exp_sum;
      }

      // store output
      const int store_pos = cu_compress_seqlens_ptr[batch];
      auto b_y_ptr = y_ptr + store_pos * kHeadDim;
      store(b_y_ptr + col, sum);

      // 2.2.move states
      auto src_kv_ptr = cur_kv_state_ptr + kRatio * kDoubleHeadDim;
      auto src_score_ptr = cur_score_state_ptr + kRatio * kDoubleHeadDim;
#pragma unroll
      for (int i = 0; i < kRatio; i++) {
        auto src_kv = load<float, kN>(src_kv_ptr + i * kDoubleHeadDim + col);
        auto src_score = load<float, kN>(src_score_ptr + i * kDoubleHeadDim + col);
        store(cur_kv_state_ptr + i * kDoubleHeadDim + col, src_kv);
        store(cur_score_state_ptr + i * kDoubleHeadDim + col, src_score);
      }
    }  // should compress
  }
}

template <int kRatio = 128, int kHeadDim = 512, int kTokensPerBatch = 1>
__global__ void c128_kv_compressor_kernel(float* y_ptr, const float* kv_ptr, const float* score_ptr,
                                          const float* ape_ptr, float* kv_states_ptr,
                                          float* score_state_ptr, const int* state_idx_ptr,
                                          const int* start_pos_ptr,
                                          const int* cu_compress_seqlens_ptr, int stride) {
  constexpr int kN = 16 / sizeof(float);

  int idx = threadIdx.x;
  int bid = blockIdx.x;
  int batch = bid;

  int state_idx = state_idx_ptr[batch];

  auto cur_kv_state_ptr = kv_states_ptr + state_idx * kRatio * kHeadDim;
  auto cur_score_state_ptr = score_state_ptr + state_idx * kRatio * kHeadDim;
  auto cur_kv_ptr = kv_ptr + (batch * kTokensPerBatch) * stride;
  auto cur_score_ptr = score_ptr + (batch * kTokensPerBatch) * stride;

  int col = idx * kN;
  int start_pos = start_pos_ptr[batch];

#pragma unroll
  for (int i = 0; i < kTokensPerBatch; i++) {
    // 1.store state
    start_pos += i;
    int pos = start_pos % kRatio;
    bool should_compress = (start_pos + 1) % kRatio == 0;

    auto kv = load<float, kN>(cur_kv_ptr + i * stride + col);
    auto score = load<float, kN>(cur_score_ptr + i * stride + col);
    auto ape = load<float, kN>(ape_ptr + pos * kHeadDim + col);
#pragma unroll
    for (int k = 0; k < kN; k++) {
      score[k] += ape[k];
    }
    store(cur_kv_state_ptr + pos * kHeadDim + col, kv);
    store(cur_score_state_ptr + pos * kHeadDim + col, score);

    // 2.compress
    if (should_compress) {
      vec_t<float, kN> exp_sum;
      vec_t<float, kN> sum;
      vec_t<float, kN> max;
      // init
#pragma unroll
      for (int i = 0; i < kN; i++) {
        exp_sum[i] = 0;
        sum[i] = 0;
        max[i] = -std::numeric_limits<float>::infinity();
      }

#pragma unroll
      for (int i = 0; i < kRatio; i++) {
        // online softmax
        auto kv = load<float, kN>(cur_kv_state_ptr + i * kHeadDim + col);
        auto score = load<float, kN>(cur_score_state_ptr + i * kHeadDim + col);
#pragma unroll
        for (int i = 0; i < kN; i++) {
          float new_max = fmaxf(max[i], score[i]);
          float scale = expf_ftz(max[i] - new_max);
          float w = expf_ftz(score[i] - new_max);
          exp_sum[i] = exp_sum[i] * scale + w;
          sum[i] = sum[i] * scale + kv[i] * w;
          max[i] = new_max;
        }
      }
      __syncthreads();
#pragma unroll
      for (int i = 0; i < kN; i++) {
        float inv_exp_sum = 1.f / exp_sum[i];
        sum[i] = sum[i] * inv_exp_sum;
      }

      // store output
      const int store_pos = cu_compress_seqlens_ptr[batch];
      auto b_y_ptr = y_ptr + store_pos * kHeadDim;
      store(b_y_ptr + col, sum);
    }
  }
}
}  // namespace kernels

void kv_compressor_decode_async(void* y_ptr, const void* kv_ptr, const void* score_ptr,
                                const void* ape_ptr, void* kv_states_ptr, void* score_state_ptr,
                                const void* state_idx_ptr, const void* start_pos_ptr,
                                const void* cu_compress_seqlens_ptr, int batch_size, int head_dim,
                                int stride, int ratio, int mtp, cudaStream_t stream) {
  if (mtp == 1) {
    // mtp decode
    if (ratio == 4) {
      if (head_dim == 128) {
        // head_dim == 128,
        dim3 grid(batch_size);
        dim3 block(32);
        auto decode_kernel = kernels::c4_kv_compressor_kernel<4, 128, 2>;
        decode_kernel<<<grid, block, 0, stream>>>(
            (float*)y_ptr, (const float*)kv_ptr, (const float*)score_ptr, (const float*)ape_ptr,
            (float*)kv_states_ptr, (float*)score_state_ptr, (const int*)state_idx_ptr,
            (const int*)start_pos_ptr, (const int*)cu_compress_seqlens_ptr, stride);
      } else {
        // head_dim == 512
        dim3 grid(batch_size);
        dim3 block(128);
        auto decode_kernel = kernels::c4_kv_compressor_kernel<4, 512, 2>;
        decode_kernel<<<grid, block, 0, stream>>>(
            (float*)y_ptr, (const float*)kv_ptr, (const float*)score_ptr, (const float*)ape_ptr,
            (float*)kv_states_ptr, (float*)score_state_ptr, (const int*)state_idx_ptr,
            (const int*)start_pos_ptr, (const int*)cu_compress_seqlens_ptr, stride);
      }
    } else {
      // ratio = 128, head_dim == 512
      dim3 grid(batch_size);
      dim3 block(128);
      auto decode_kernel = kernels::c128_kv_compressor_kernel<128, 512, 2>;
      decode_kernel<<<grid, block, 0, stream>>>(
          (float*)y_ptr, (const float*)kv_ptr, (const float*)score_ptr, (const float*)ape_ptr,
          (float*)kv_states_ptr, (float*)score_state_ptr, (const int*)state_idx_ptr,
          (const int*)start_pos_ptr, (const int*)cu_compress_seqlens_ptr, stride);
    }
  } else {
    // normal decode
    if (ratio == 4) {
      if (head_dim == 128) {
        // head_dim == 128
        dim3 grid(batch_size);
        dim3 block(32);
        auto decode_kernel = kernels::c4_kv_compressor_kernel<4, 128, 1>;
        decode_kernel<<<grid, block, 0, stream>>>(
            (float*)y_ptr, (const float*)kv_ptr, (const float*)score_ptr, (const float*)ape_ptr,
            (float*)kv_states_ptr, (float*)score_state_ptr, (const int*)state_idx_ptr,
            (const int*)start_pos_ptr, (const int*)cu_compress_seqlens_ptr, stride);
      } else {
        // head_dim == 512
        dim3 grid(batch_size);
        dim3 block(128);
        auto decode_kernel = kernels::c4_kv_compressor_kernel<4, 512, 1>;
        decode_kernel<<<grid, block, 0, stream>>>(
            (float*)y_ptr, (const float*)kv_ptr, (const float*)score_ptr, (const float*)ape_ptr,
            (float*)kv_states_ptr, (float*)score_state_ptr, (const int*)state_idx_ptr,
            (const int*)start_pos_ptr, (const int*)cu_compress_seqlens_ptr, stride);
      }
    } else {
      // ratio = 128, head_dim == 512
      dim3 grid(batch_size);
      dim3 block(128);
      auto decode_kernel = kernels::c128_kv_compressor_kernel<128, 512, 1>;
      decode_kernel<<<grid, block, 0, stream>>>(
          (float*)y_ptr, (const float*)kv_ptr, (const float*)score_ptr, (const float*)ape_ptr,
          (float*)kv_states_ptr, (float*)score_state_ptr, (const int*)state_idx_ptr,
          (const int*)start_pos_ptr, (const int*)cu_compress_seqlens_ptr, stride);
    }
  }
}

}  // namespace compressor
}  // namespace hpc
