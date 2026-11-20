// Copyright (C) 2026 Tencent.
#include "src/iHC/amd/iHC.h"
#include "src/utils/amd/device.cuh"

namespace hpc::iHC::amd {
namespace kernels {

// Small projection using ordinary FP32 multiply/add, with no WMMA/MFMA instructions.
template <int kHidden, bool kHasPost, bool kNorm>
__global__ void fuse_ihc_pre_or_head_kernel(hip_bfloat16* y, float* post, const hip_bfloat16* x,
                                            const float* w, const float* scale, const float* base,
                                            float norm_eps, float hc_eps, float magnitude,
                                            const hip_bfloat16* rms_weight, float rms_eps,
                                            bool cast_bfloat_for_norm) {
  constexpr int kThreads = 256;
  constexpr int kChannels = 4;
  constexpr int kProjection = kHasPost ? 8 : 4;
  constexpr int kFlat = kChannels * kHidden;
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  __shared__ float scratch[32];
  __shared__ float gate[kChannels];
  float dots[kProjection] = {};
  float square = 0.0f;
  for (int col = tid; col < kFlat; col += kThreads) {
    const float value = static_cast<float>(x[static_cast<int64_t>(row) * kFlat + col]);
    square += value * value;
#pragma unroll
    for (int p = 0; p < kProjection; ++p) {
      dots[p] += value * w[p * kFlat + col];
    }
  }
  const float inv_rms = rsqrtf(hpc::amd::block_sum<kThreads>(square, scratch) / kFlat + norm_eps);
#pragma unroll
  for (int p = 0; p < kProjection; ++p) {
    const float dot = hpc::amd::block_sum<kThreads>(dots[p], scratch);
    if (tid == 0) {
      const float value = dot * inv_rms;
      if (p < kChannels) {
        gate[p] = hpc::amd::sigmoid(scale[0] * value + base[p]) + hc_eps;
      } else {
        post[row * kChannels + p - kChannels] =
            magnitude * hpc::amd::sigmoid(scale[1] * value + base[p]) + hc_eps;
      }
    }
  }
  __syncthreads();

  float values[kHidden / kThreads];
  float y_square = 0.0f;
#pragma unroll
  for (int i = 0; i < kHidden / kThreads; ++i) {
    const int col = i * kThreads + tid;
    float value = 0.0f;
#pragma unroll
    for (int c = 0; c < kChannels; ++c) {
      value +=
          gate[c] * static_cast<float>(x[static_cast<int64_t>(row) * kFlat + c * kHidden + col]);
    }
    if constexpr (kNorm) {
      if (cast_bfloat_for_norm) value = static_cast<float>(hip_bfloat16(value));
      values[i] = value;
      y_square += value * value;
    } else {
      y[static_cast<int64_t>(row) * kHidden + col] = hip_bfloat16(value);
    }
  }
  if constexpr (kNorm) {
    const float inv_y_rms =
        rsqrtf(hpc::amd::block_sum<kThreads>(y_square, scratch) / kHidden + rms_eps);
#pragma unroll
    for (int i = 0; i < kHidden / kThreads; ++i) {
      const int col = i * kThreads + tid;
      y[static_cast<int64_t>(row) * kHidden + col] =
          hip_bfloat16(values[i] * inv_y_rms * static_cast<float>(rms_weight[col]));
    }
  }
}

__global__ void fuse_ihc_post_kernel(hip_bfloat16* y, const hip_bfloat16* x,
                                     const hip_bfloat16* residual, const float* post, int hidden) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y;
  if (col >= hidden) return;
  const float value = static_cast<float>(x[static_cast<int64_t>(row) * hidden + col]);
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const int64_t offset = (static_cast<int64_t>(row) * 4 + c) * hidden + col;
    const float product = post[row * 4 + c] * value;
    y[offset] = hip_bfloat16(product + static_cast<float>(residual[offset]));
  }
}

}  // namespace kernels

template <int kHidden, bool kHasPost>
void launch_pre_head(hip_bfloat16* y, float* post, const hip_bfloat16* x, const float* w,
                     const float* scale, const float* base, int batch, float norm_eps, float hc_eps,
                     float magnitude, const hip_bfloat16* rms_weight, float rms_eps,
                     bool cast_bfloat_for_norm, hipStream_t stream) {
  if (rms_weight) {
    hipLaunchKernelGGL((kernels::fuse_ihc_pre_or_head_kernel<kHidden, kHasPost, true>), dim3(batch),
                       dim3(256), 0, stream, y, post, x, w, scale, base, norm_eps, hc_eps,
                       magnitude, rms_weight, rms_eps, cast_bfloat_for_norm);
  } else {
    hipLaunchKernelGGL((kernels::fuse_ihc_pre_or_head_kernel<kHidden, kHasPost, false>),
                       dim3(batch), dim3(256), 0, stream, y, post, x, w, scale, base, norm_eps,
                       hc_eps, magnitude, rms_weight, rms_eps, cast_bfloat_for_norm);
  }
}

void pre_head_async(hip_bfloat16* y, float* post, const hip_bfloat16* x, const float* w,
                    const float* scale, const float* base, int batch, int hidden, float norm_eps,
                    float hc_eps, float magnitude, const hip_bfloat16* rms_weight, float rms_eps,
                    bool cast_bfloat_for_norm, hipStream_t stream) {
  if (batch == 0) return;
#define HPC_LAUNCH_PRE_HEAD(DIM, HAS_POST)                                                       \
  launch_pre_head<DIM, HAS_POST>(y, post, x, w, scale, base, batch, norm_eps, hc_eps, magnitude, \
                                 rms_weight, rms_eps, cast_bfloat_for_norm, stream)
  if (hidden == 4096) {
    if (post) {
      HPC_LAUNCH_PRE_HEAD(4096, true);
    } else {
      HPC_LAUNCH_PRE_HEAD(4096, false);
    }
  } else {
    if (post) {
      HPC_LAUNCH_PRE_HEAD(6144, true);
    } else {
      HPC_LAUNCH_PRE_HEAD(6144, false);
    }
  }
#undef HPC_LAUNCH_PRE_HEAD
}

void post_async(hip_bfloat16* y, const hip_bfloat16* x, const hip_bfloat16* residual,
                const float* post, int batch, int hidden, hipStream_t stream) {
  if (batch == 0) return;
  hipLaunchKernelGGL(kernels::fuse_ihc_post_kernel, dim3((hidden + 255) / 256, batch), dim3(256), 0,
                     stream, y, x, residual, post, hidden);
}

}  // namespace hpc::iHC::amd
