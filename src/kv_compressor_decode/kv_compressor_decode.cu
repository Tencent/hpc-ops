// Copyright 2025 hpc-ops authors

#include <cuda.h>

#include "src/kv_compressor_decode/kv_compressor_decode.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace kv_compressor_decode {

namespace kernels {

template <int kRatio = 4, int kHeadDim = 128, int kNumTokensPerBatch = 1>
__global__ void c4_kv_compressor_decode_kernel(float* y_ptr, const float* kv_ptr,
                                               const float* score_ptr, const float* ape_ptr,
                                               float* kv_states_ptr, float* score_states_ptr,
                                               const int* state_idx_ptr, const int* start_pos_ptr,
                                               const int* cu_compress_seqlens_ptr) {
  constexpr int kNumElementsPerThread = 16 / sizeof(float);
  constexpr int kDoubleRatio = kRatio * 2;
  constexpr int kDoubleHeadDim = kHeadDim * 2;

  const int idx = threadIdx.x;
  const int bid = blockIdx.x;
  const int batch = bid;

  const int state_idx = state_idx_ptr[batch];

  auto const b_kv_states_ptr = kv_states_ptr + state_idx * kDoubleRatio * kDoubleHeadDim;
  auto const b_score_states_ptr = score_states_ptr + state_idx * kDoubleRatio * kDoubleHeadDim;
  int start_pos = start_pos_ptr[batch];

#pragma unroll
  for (int i = 0; i < kNumTokensPerBatch; i++) {
    // 1.store state
    start_pos += i;
    const int pos = start_pos % kRatio;
    const bool should_compress = (start_pos + 1) % kRatio == 0;

    for (int col = idx * kNumElementsPerThread; col < kDoubleHeadDim;
         col += blockDim.x * kNumElementsPerThread) {
      auto b_kv_ptr = kv_ptr + (batch * kNumTokensPerBatch + i) * kDoubleHeadDim;
      auto b_score_ptr = score_ptr + (batch * kNumTokensPerBatch + i) * kDoubleHeadDim;

      auto kv = load<float, kNumElementsPerThread>(b_kv_ptr + col);
      auto score = load<float, kNumElementsPerThread>(b_score_ptr + col);
      auto ape = load<float, kNumElementsPerThread>(ape_ptr + pos * kDoubleHeadDim + col);

#pragma unroll
      for (int i = 0; i < kNumElementsPerThread; i++) {
        score[i] += ape[i];
      }
      auto store_kv_pos_ptr = b_kv_states_ptr + (kRatio + pos) * kDoubleHeadDim;
      auto store_score_pos_ptr = b_score_states_ptr + (kRatio + pos) * kDoubleHeadDim;
      store(store_kv_pos_ptr + col, kv);
      store(store_score_pos_ptr + col, score);
    }

    // 2.compress and move states
    if (should_compress) {
      // 2.1.compress
      const int col = idx * kNumElementsPerThread;
      vec_t<float, kNumElementsPerThread> score_exp_sum;
      vec_t<float, kNumElementsPerThread> elemul_acc_sum;
      vec_t<float, kNumElementsPerThread> max;
      // init
#pragma unroll
      for (int i = 0; i < kNumElementsPerThread; i++) {
        score_exp_sum[i] = 0;
        elemul_acc_sum[i] = 0;
        max[i] = -std::numeric_limits<float>::infinity();
      }

#pragma unroll
      // online softmax
      for (int i = 0; i < kRatio; i++) {
        auto i_kv_state_pos_ptr = b_kv_states_ptr + i * kDoubleHeadDim;
        auto i_score_state_pos_ptr = b_score_states_ptr + i * kDoubleHeadDim;
        auto kv = load<float, kNumElementsPerThread>(i_kv_state_pos_ptr + col);
        auto score = load<float, kNumElementsPerThread>(i_score_state_pos_ptr + col);
#pragma unroll
        for (int i = 0; i < kNumElementsPerThread; i++) {
          if (score[i] > max[i]) {
            float scale = exp(max[i] - score[i]);
            score_exp_sum[i] = score_exp_sum[i] * scale + 1.f;
            elemul_acc_sum[i] = elemul_acc_sum[i] * scale + kv[i] * 1.f;
            max[i] = score[i];
          } else {
            score[i] = exp(score[i] - max[i]);
            score_exp_sum[i] += score[i];
            elemul_acc_sum[i] += kv[i] * score[i];
          }
        }
      }
      auto const b_kv_states_shift_ptr = b_kv_states_ptr + kRatio * kDoubleHeadDim + kHeadDim;
      auto const b_score_states_shift_ptr = b_score_states_ptr + kRatio * kDoubleHeadDim + kHeadDim;
#pragma unroll
      for (int i = 0; i < kRatio; i++) {
        auto i_kv_state_pos_ptr = b_kv_states_shift_ptr + i * kDoubleHeadDim;
        auto i_score_state_pos_ptr = b_score_states_shift_ptr + i * kDoubleHeadDim;
        auto kv = load<float, kNumElementsPerThread>(i_kv_state_pos_ptr + col);
        auto score = load<float, kNumElementsPerThread>(i_score_state_pos_ptr + col);
#pragma unroll
        for (int i = 0; i < kNumElementsPerThread; i++) {
          if (score[i] > max[i]) {
            float scale = exp(max[i] - score[i]);
            score_exp_sum[i] = score_exp_sum[i] * scale + 1.f;
            elemul_acc_sum[i] = elemul_acc_sum[i] * scale + kv[i] * 1.f;
            max[i] = score[i];
          } else {
            score[i] = exp(score[i] - max[i]);
            score_exp_sum[i] += score[i];
            elemul_acc_sum[i] += kv[i] * score[i];
          }
        }
      }
#pragma unroll
      for (int i = 0; i < kNumElementsPerThread; i++) {
        float inv_exp_sum = 1.f / score_exp_sum[i];
        elemul_acc_sum[i] = elemul_acc_sum[i] * inv_exp_sum;
      }

      // store output
      const int store_pos = cu_compress_seqlens_ptr[batch];
      auto b_y_ptr = y_ptr + store_pos * kHeadDim;
      store(b_y_ptr + col, elemul_acc_sum);

      // 2.2.move states
      auto s_kv_states_start_ptr = b_kv_states_ptr + kRatio * kDoubleHeadDim;
      auto d_kv_states_start_ptr = b_kv_states_ptr;
      auto s_score_states_start_ptr = b_score_states_ptr + kRatio * kDoubleHeadDim;
      auto d_score_states_start_ptr = b_score_states_ptr;
#pragma unroll
      for (int i = 0; i < kRatio; i++) {
        auto s_kv_states_ptr = s_kv_states_start_ptr + i * kDoubleHeadDim;
        auto d_kv_states_ptr = d_kv_states_start_ptr + i * kDoubleHeadDim;
        auto s_score_states_ptr = s_score_states_start_ptr + i * kDoubleHeadDim;
        auto d_score_states_ptr = d_score_states_start_ptr + i * kDoubleHeadDim;
        auto s_kv = load<float, kNumElementsPerThread>(s_kv_states_ptr + col);
        auto s_score = load<float, kNumElementsPerThread>(s_score_states_ptr + col);
        store(d_kv_states_ptr + col, s_kv);
        store(d_score_states_ptr + col, s_score);
      }
    }  // should compress
  }
}

template <int kRatio = 128, int kHeadDim = 512, int kNumTokensPerBatch = 1>
__global__ void c128_kv_compressor_decode_kernel(float* y_ptr, const float* kv_ptr,
                                                 const float* score_ptr, const float* ape_ptr,
                                                 float* kv_states_ptr, float* score_states_ptr,
                                                 const int* state_idx_ptr, const int* start_pos_ptr,
                                                 const int* cu_compress_seqlens_ptr) {
  constexpr int kNumElementsPerThread = 16 / sizeof(float);

  const int idx = threadIdx.x;
  const int bid = blockIdx.x;
  const int batch = bid;

  const int state_idx = state_idx_ptr[batch];

  auto const b_kv_states_ptr = kv_states_ptr + state_idx * kRatio * kHeadDim;
  auto const b_score_states_ptr = score_states_ptr + state_idx * kRatio * kHeadDim;

  const int col = idx * kNumElementsPerThread;
  int start_pos = start_pos_ptr[batch];

#pragma unroll
  for (int i = 0; i < kNumTokensPerBatch; i++) {
    // 1.store state
    start_pos += i;
    const int pos = start_pos % kRatio;
    const bool should_compress = (start_pos + 1) % kRatio == 0;

    const auto b_kv_ptr = kv_ptr + (batch * kNumTokensPerBatch + i) * kHeadDim;
    const auto b_score_ptr = score_ptr + (batch * kNumTokensPerBatch + i) * kHeadDim;

    auto kv = load<float, kNumElementsPerThread>(b_kv_ptr + col);
    auto score = load<float, kNumElementsPerThread>(b_score_ptr + col);
    auto ape = load<float, kNumElementsPerThread>(ape_ptr + pos * kHeadDim + col);
#pragma unroll
    for (int k = 0; k < kNumElementsPerThread; k++) {
      score[k] += ape[k];
    }

    auto store_kv_pos_ptr = b_kv_states_ptr + pos * kHeadDim;
    auto store_score_pos_ptr = b_score_states_ptr + pos * kHeadDim;
    store(store_kv_pos_ptr + col, kv);
    store(store_score_pos_ptr + col, score);

    // 2.compress
    if (should_compress) {
      vec_t<float, kNumElementsPerThread> score_exp_sum;
      vec_t<float, kNumElementsPerThread> elemul_acc_sum;
      vec_t<float, kNumElementsPerThread> max;
      // init
#pragma unroll
      for (int i = 0; i < kNumElementsPerThread; i++) {
        score_exp_sum[i] = 0;
        elemul_acc_sum[i] = 0;
        max[i] = -std::numeric_limits<float>::infinity();
      }

#pragma unroll
      for (int i = 0; i < kRatio; i++) {
        // online softmax
        auto i_kv_states_ptr = b_kv_states_ptr + i * kHeadDim;
        auto i_score_states_ptr = b_score_states_ptr + i * kHeadDim;
        auto kv = load<float, kNumElementsPerThread>(i_kv_states_ptr + col);
        auto score = load<float, kNumElementsPerThread>(i_score_states_ptr + col);
#pragma unroll
        for (int i = 0; i < kNumElementsPerThread; i++) {
          if (score[i] > max[i]) {
            float scale = exp(max[i] - score[i]);
            score_exp_sum[i] = score_exp_sum[i] * scale + 1.f;
            elemul_acc_sum[i] = elemul_acc_sum[i] * scale + kv[i] * 1.f;
            max[i] = score[i];
          } else {
            score[i] = exp(score[i] - max[i]);
            score_exp_sum[i] += score[i];
            elemul_acc_sum[i] += kv[i] * score[i];
          }
        }
      }
      __syncthreads();
#pragma unroll
      for (int i = 0; i < kNumElementsPerThread; i++) {
        float inv_exp_sum = 1.f / score_exp_sum[i];
        elemul_acc_sum[i] = elemul_acc_sum[i] * inv_exp_sum;
      }

      // store output
      const int store_pos = cu_compress_seqlens_ptr[batch];
      auto b_y_ptr = y_ptr + store_pos * kHeadDim;
      store(b_y_ptr + col, elemul_acc_sum);
    }
  }
}
}  // namespace kernels

void kv_compressor_decode_async(void* y_ptr, const void* kv_ptr, const void* score_ptr,
                                const void* ape_ptr, void* kv_states_ptr, void* score_states_ptr,
                                const void* state_idx_ptr, const void* start_pos_ptr,
                                const void* cu_compress_seqlens_ptr, int batch_size, int head_dim,
                                int ratio, int mtp, cudaStream_t stream) {
  if (mtp == 1) {
    // mtp decode
    if (ratio == 4) {
      // head_dim == 128, overlap is true
      dim3 grid(batch_size);
      dim3 block(32);
      auto decode_kernel = kernels::c4_kv_compressor_decode_kernel<4, 128, 2>;
      decode_kernel<<<grid, block, 0, stream>>>(
          (float*)y_ptr, (const float*)kv_ptr, (const float*)score_ptr, (const float*)ape_ptr,
          (float*)kv_states_ptr, (float*)score_states_ptr, (const int*)state_idx_ptr,
          (const int*)start_pos_ptr, (const int*)cu_compress_seqlens_ptr);
    } else {
      // head_dim == 512, overlap is false
      dim3 grid(batch_size);
      dim3 block(128);
      auto decode_kernel = kernels::c128_kv_compressor_decode_kernel<128, 512, 2>;
      decode_kernel<<<grid, block, 0, stream>>>(
          (float*)y_ptr, (const float*)kv_ptr, (const float*)score_ptr, (const float*)ape_ptr,
          (float*)kv_states_ptr, (float*)score_states_ptr, (const int*)state_idx_ptr,
          (const int*)start_pos_ptr, (const int*)cu_compress_seqlens_ptr);
    }
  } else {
    // normal decode
    if (ratio == 4) {
      // head_dim == 128, overlap is true
      dim3 grid(batch_size);
      dim3 block(32);
      auto decode_kernel = kernels::c4_kv_compressor_decode_kernel<4, 128, 1>;
      decode_kernel<<<grid, block, 0, stream>>>(
          (float*)y_ptr, (const float*)kv_ptr, (const float*)score_ptr, (const float*)ape_ptr,
          (float*)kv_states_ptr, (float*)score_states_ptr, (const int*)state_idx_ptr,
          (const int*)start_pos_ptr, (const int*)cu_compress_seqlens_ptr);
    } else {
      // head_dim == 512, overlap is false
      dim3 grid(batch_size);
      dim3 block(128);
      auto decode_kernel = kernels::c128_kv_compressor_decode_kernel<128, 512, 1>;
      decode_kernel<<<grid, block, 0, stream>>>(
          (float*)y_ptr, (const float*)kv_ptr, (const float*)score_ptr, (const float*)ape_ptr,
          (float*)kv_states_ptr, (float*)score_states_ptr, (const int*)state_idx_ptr,
          (const int*)start_pos_ptr, (const int*)cu_compress_seqlens_ptr);
    }
  }
}

}  // namespace kv_compressor_decode
}  // namespace hpc
