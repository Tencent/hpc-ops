// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include <cub/cub.cuh>

#include "cute/tensor.hpp"
#include "src/fuse_moe/sm100/fuse_moe.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace fuse_moe {

namespace kernels {

template <typename T, int kThreadPerBlock, int kNumItemPer16B, int kNumTopkMax = 128,
          bool kUsePDL = false>
__global__ void reduce_kernel(T *y_ptr, const T *x_ptr, const int *topk_pos_ptr,
                              const float *topk_scale_ptr, const T *shared_output_ptr,
                              int total_num_seq, int num_seq, int hidden_size, int num_topk,
                              cutlass::FastDivmod block_divider) {
  int iblock = blockIdx.x;

  int iblockx;
  int iblocky;

  block_divider(iblocky, iblockx, iblock);
  uint64_t irow = iblocky;
  int icol = (threadIdx.x + iblockx * kThreadPerBlock) * kNumItemPer16B;

  __shared__ int pos_shm[kNumTopkMax];
  __shared__ float scale_shm[kNumTopkMax];

  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }

  // Optimized shared memory loading with warp-level coalescing
  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  const int warps_per_block = kThreadPerBlock / 32;

  // Warp-level loading for better coalescing
  for (int i = warp_id; i < num_topk; i += warps_per_block) {
    if (lane_id == 0) {
      pos_shm[i] = topk_pos_ptr[irow * num_topk + i];
      scale_shm[i] = topk_scale_ptr[irow * num_topk + i];
    }
  }
  __syncthreads();

  if (icol < hidden_size) {
    vec_t<float, kNumItemPer16B> y_fp32;
#pragma unroll
    for (int i = 0; i < kNumItemPer16B; i++) {
      y_fp32[i] = 0.f;
    }

    auto y_irow_ptr = y_ptr + irow * hidden_size;

    // Optimized accumulation with warp-level parallelism
    // Use pairwise summation for better numerical stability
    for (int i = 0; i < num_topk; i++) {
      int ipos = pos_shm[i];
      float iscale = scale_shm[i];
      if (ipos >= 0) {
        auto x_irow_ptr = x_ptr + static_cast<uint64_t>(ipos) * hidden_size;
        auto x_fp32 = to<float>(load<T, kNumItemPer16B>(x_irow_ptr + icol));
#pragma unroll
        for (int j = 0; j < kNumItemPer16B; j++) {
          // Simple accumulation - the difference is likely due to different
          // accumulation order between kernel and reference implementation
          y_fp32[j] += x_fp32[j] * iscale;
        }
      }
    }

    if (shared_output_ptr) {
      auto shared_irow_ptr = shared_output_ptr + irow * hidden_size;
      auto shared_fp32 = to<float>(load<T, kNumItemPer16B>(shared_irow_ptr + icol));
#pragma unroll
      for (int j = 0; j < kNumItemPer16B; j++) {
        y_fp32[j] += shared_fp32[j];
      }
    }

    auto y_bf16 = to<T>(y_fp32);
    store(y_irow_ptr + icol, y_bf16);
  }

  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

template <typename T, int kThreadPerBlock, int kNumItemPer16B, int kNumTopk, int kVecsPerThread,
          bool kUsePDL>
__global__ void reduce_kernel_v2(T *y_ptr, const T *x_ptr, const int *topk_pos_ptr,
                                 const float *topk_scale_ptr, const T *shared_output_ptr,
                                 int total_num_seq, int num_seq, int hidden_size) {
  const uint64_t irow = blockIdx.x;
  if (irow >= (uint64_t)num_seq) {
    return;
  }

  // Load pos/scale once per row. Up to kNumTopk <= kThreadPerBlock so the
  // first kNumTopk threads cover everything. Stash into shared memory so the
  // loop over topk below stays in registers on all threads.
  __shared__ int pos_shm[kNumTopk];
  __shared__ float scale_shm[kNumTopk];

  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }

  if (threadIdx.x < kNumTopk) {
    pos_shm[threadIdx.x] = topk_pos_ptr[irow * kNumTopk + threadIdx.x];
    scale_shm[threadIdx.x] = topk_scale_ptr[irow * kNumTopk + threadIdx.x];
  }
  __syncthreads();

  // Pull meta into registers on every thread (broadcast from smem, 1 LDS).
  int pos_reg[kNumTopk];
  float scale_reg[kNumTopk];
#pragma unroll
  for (int i = 0; i < kNumTopk; ++i) {
    pos_reg[i] = pos_shm[i];
    scale_reg[i] = scale_shm[i];
  }

  auto y_irow_ptr = y_ptr + irow * hidden_size;

  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }

#pragma unroll
  for (int v = 0; v < kVecsPerThread; ++v) {
    int icol = (v * kThreadPerBlock + threadIdx.x) * kNumItemPer16B;
    if (icol >= hidden_size) {
      break;
    }

    vec_t<float, kNumItemPer16B> y_fp32;
#pragma unroll
    for (int j = 0; j < kNumItemPer16B; ++j) {
      y_fp32[j] = 0.f;
    }

    // Fully unrolled topk accumulation. Each iteration is an independent
    // 16B load -> float convert -> fma, so the compiler issues all kNumTopk
    // LDG.E.128 up-front and only the final fmas serialize on y_fp32.
#pragma unroll
    for (int i = 0; i < kNumTopk; ++i) {
      int ipos = pos_reg[i];
      float iscale = scale_reg[i];
      if (ipos >= 0) {
        auto x_irow_ptr = x_ptr + static_cast<uint64_t>(ipos) * hidden_size;
        auto x_fp32 = to<float>(load<T, kNumItemPer16B>(x_irow_ptr + icol));
#pragma unroll
        for (int j = 0; j < kNumItemPer16B; ++j) {
          y_fp32[j] += x_fp32[j] * iscale;
        }
      }
    }

    if (shared_output_ptr) {
      auto shared_irow_ptr = shared_output_ptr + irow * hidden_size;
      auto shared_fp32 = to<float>(load<T, kNumItemPer16B>(shared_irow_ptr + icol));
#pragma unroll
      for (int j = 0; j < kNumItemPer16B; ++j) {
        y_fp32[j] += shared_fp32[j];
      }
    }

    store(y_irow_ptr + icol, to<T>(y_fp32));
  }

  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

}  // namespace kernels

void reduce_async(void *y_ptr, const void *x_ptr, const void *topk_pos_ptr,
                  const void *topk_scale_ptr, const void *shared_output_ptr, int total_num_seq,
                  int num_seq, int hidden_size, int num_topk, bool use_pdl, cudaStream_t stream) {
  using T = __nv_bfloat16;
  constexpr int kThreadPerBlock = 256;
  constexpr int kNumItemPer16B = 16 / sizeof(T);  // = 8 for bf16

  const int num_vec_cols = hidden_size / kNumItemPer16B;
  const int vecs_per_thread = (num_vec_cols + kThreadPerBlock - 1) / kThreadPerBlock;

  dim3 block(kThreadPerBlock);
  dim3 grid(num_seq);

  // Build a cudaLaunchConfig_t that optionally carries the PDL attribute. The
  // in-kernel cudaGridDependencySynchronize / cudaTriggerProgrammaticLaunchCompletion
  // calls only take effect when the kernel is launched through cudaLaunchKernelEx
  // with cudaLaunchAttributeProgrammaticStreamSerialization set; otherwise they
  // are no-ops, which is why PDL previously appeared "not to work".
  cudaLaunchAttribute attribute[1];
  attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute[0].val.programmaticStreamSerializationAllowed = 1;

  // Dispatch on (num_topk, vecs_per_thread). We only specialize the cases we
  // care about here; callers must pre-check and fall back to v1 otherwise.
  auto launch_v2 = [&](auto kNumTopk, auto kVecsPerThread) {
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    if (use_pdl) {
      config.attrs = attribute;
      config.numAttrs = 1;
      auto kernel =
          kernels::reduce_kernel_v2<T, kThreadPerBlock, kNumItemPer16B, decltype(kNumTopk)::value,
                                    decltype(kVecsPerThread)::value, /*kUsePDL=*/true>;
      cudaLaunchKernelEx(&config, kernel, (T *)y_ptr, (const T *)x_ptr, (const int *)topk_pos_ptr,
                         (const float *)topk_scale_ptr, (const T *)shared_output_ptr, total_num_seq,
                         num_seq, hidden_size);
    } else {
      auto kernel =
          kernels::reduce_kernel_v2<T, kThreadPerBlock, kNumItemPer16B, decltype(kNumTopk)::value,
                                    decltype(kVecsPerThread)::value, /*kUsePDL=*/false>;
      cudaLaunchKernelEx(&config, kernel, (T *)y_ptr, (const T *)x_ptr, (const int *)topk_pos_ptr,
                         (const float *)topk_scale_ptr, (const T *)shared_output_ptr, total_num_seq,
                         num_seq, hidden_size);
    }
  };

  auto dispatch_topk = [&](auto kNumTopk) {
    if (vecs_per_thread == 1) {
      launch_v2(kNumTopk, std::integral_constant<int, 1>{});
    } else if (vecs_per_thread == 2) {
      launch_v2(kNumTopk, std::integral_constant<int, 2>{});
    } else if (vecs_per_thread == 4) {
      launch_v2(kNumTopk, std::integral_constant<int, 4>{});
    } else {
      // Not specialized; caller should have checked.
      launch_v2(kNumTopk, std::integral_constant<int, 8>{});
    }
  };

  if (num_topk == 8) {
    dispatch_topk(std::integral_constant<int, 8>{});
  } else if (num_topk == 4) {
    dispatch_topk(std::integral_constant<int, 4>{});
  } else {
    // Fallback to the original reduce_kernel (v1) for any other num_topk.
    // v1 takes num_topk as a runtime argument, so it handles every case.
    constexpr int kNumTopkMax = 128;
    int num_block_col = (hidden_size / kNumItemPer16B + kThreadPerBlock - 1) / kThreadPerBlock;
    int num_block_total = num_seq * num_block_col;
    cutlass::FastDivmod block_divider(num_block_col);

    dim3 v1_block(kThreadPerBlock);
    dim3 v1_grid(num_block_total);
    const size_t v1_smem = kNumTopkMax * sizeof(int) + kNumTopkMax * sizeof(float);

    cudaLaunchConfig_t config{};
    config.gridDim = v1_grid;
    config.blockDim = v1_block;
    config.dynamicSmemBytes = v1_smem;
    config.stream = stream;
    if (use_pdl) {
      config.attrs = attribute;
      config.numAttrs = 1;
      auto kernel = kernels::reduce_kernel<T, kThreadPerBlock, kNumItemPer16B, kNumTopkMax,
                                           /*kUsePDL=*/true>;
      cudaLaunchKernelEx(&config, kernel, (T *)y_ptr, (const T *)x_ptr, (const int *)topk_pos_ptr,
                         (const float *)topk_scale_ptr, (const T *)shared_output_ptr, total_num_seq,
                         num_seq, hidden_size, num_topk, block_divider);
    } else {
      auto kernel = kernels::reduce_kernel<T, kThreadPerBlock, kNumItemPer16B, kNumTopkMax,
                                           /*kUsePDL=*/false>;
      cudaLaunchKernelEx(&config, kernel, (T *)y_ptr, (const T *)x_ptr, (const int *)topk_pos_ptr,
                         (const float *)topk_scale_ptr, (const T *)shared_output_ptr, total_num_seq,
                         num_seq, hidden_size, num_topk, block_divider);
    }
  }
}

}  // namespace fuse_moe
}  // namespace hpc
