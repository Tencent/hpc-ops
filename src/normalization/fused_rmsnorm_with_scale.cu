// Copyright 2025 hpc-ops authors
#include <cuda.h>

#include <iostream>

#include "src/normalization/fused_rmsnorm_with_scale.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace normalization {

namespace kernels {

template <int kHiddenStates, int kWarpPerBatch, int kBatchPerBlock, bool kIsMoe>
__global__ void fused_rmsnorm_with_scale(const __nv_bfloat16 *input_ptr,
                                         const __nv_bfloat16 *weight_ptr, float *output_ptr_fp32,
                                         __nv_fp8_e4m3 *output_ptr_fp8,
                                         __nv_fp8_e4m3 *output_ptr_fp8_scale2, const float *scale,
                                         float eps, int batch_size) {
  constexpr int kWarpSize = 32;
  constexpr int kItemPer16B = 8;
  constexpr float kInvHiddenStates = 1.0f / kHiddenStates;
  constexpr int kIterPerBatch = (kHiddenStates + kWarpPerBatch * kWarpSize * kItemPer16B - 1) /
                                (kWarpPerBatch * kWarpSize * kItemPer16B);
  float inv_scale = rcpf_ftz(scale[0]);
  float inv_scale2 = 1;
  if constexpr (kIsMoe) {
    inv_scale2 = rcpf_ftz(scale[1]);
    inv_scale *= scale[1];
  }

  const int iwarp = threadIdx.x / kWarpSize;
  const int ilane = threadIdx.x % kWarpSize;

  const uint64_t ibatch = blockIdx.x * kBatchPerBlock + iwarp / kWarpPerBatch;
  const uint64_t icol_thr_offset = ((iwarp % kWarpPerBatch) * kWarpSize + ilane) * kItemPer16B;

  __shared__ float smem_sum[kWarpPerBatch];

  vec_t<float, kItemPer16B> reg_input[kIterPerBatch];
  vec_t<float, kItemPer16B> reg_weight[kIterPerBatch];

#pragma unroll
  for (int i = 0; i < kIterPerBatch; i++) {
#pragma unroll
    for (int j = 0; j < kItemPer16B; j++) {
      reg_input[i][j] = 0;
      reg_weight[i][j] = 0;
    }
  }

#pragma unroll
  for (int iter = 0; iter < kIterPerBatch; iter++) {
    uint64_t icol = icol_thr_offset + iter * kWarpPerBatch * kWarpSize * kItemPer16B;

    if (icol < kHiddenStates) {
      reg_weight[iter] = to<float>(load<__nv_bfloat162, kItemPer16B / 2>(&weight_ptr[icol]));
    }
  }

  float local_mean = 0.0f;

  if (ibatch < batch_size) {
#pragma unroll
    for (int iter = 0; iter < kIterPerBatch; iter++) {
      uint64_t icol = icol_thr_offset + iter * kWarpPerBatch * kWarpSize * kItemPer16B;

      if (icol < kHiddenStates) {
        reg_input[iter] = to<float>(
            load<__nv_bfloat162, kItemPer16B / 2>(&input_ptr[ibatch * kHiddenStates + icol]));
#pragma unroll
        for (int i = 0; i < kItemPer16B; i++) {
          local_mean += reg_input[iter][i] * reg_input[iter][i];
        }
      }
    }
  }

  // warp reduce
  local_mean = warp_reduce_sum_xor(local_mean);

  if (ilane == 0) {
    smem_sum[iwarp] = local_mean;
  }
  __syncthreads();

  int first_warp_in_batch = (iwarp / kWarpPerBatch) * kWarpPerBatch;
#pragma unroll
  for (int iwarp_in_batch = 0; iwarp_in_batch < kWarpPerBatch; iwarp_in_batch++) {
    int reduce_warp = first_warp_in_batch + iwarp_in_batch;
    if (iwarp != reduce_warp) {
      local_mean += smem_sum[reduce_warp];
    }
  }

  local_mean = rsqrtf_ftz(local_mean * kInvHiddenStates + eps);

  if constexpr (!kIsMoe) {
    local_mean *= inv_scale;
  }

  if (ibatch < batch_size) {
#pragma unroll
    for (int iter = 0; iter < kIterPerBatch; iter++) {
      uint64_t icol = icol_thr_offset + iter * kWarpPerBatch * kWarpSize * kItemPer16B;
      uint64_t istore = ibatch * kHiddenStates + icol;

      if (icol < kHiddenStates) {
#pragma unroll
        for (int i = 0; i < kItemPer16B; i++) {
          reg_input[iter][i] = reg_input[iter][i] * reg_weight[iter][i] * local_mean;
        }

        auto &split_input = reshape<2, kItemPer16B / 2>(reg_input[iter]);
        if constexpr (kIsMoe) {
          store(&output_ptr_fp32[istore], split_input[0]);
          store(&output_ptr_fp32[istore + kItemPer16B / 2], split_input[1]);
#pragma unroll
          for (int i = 0; i < kItemPer16B; i++) {
            reg_input[iter][i] *= inv_scale2;
          }

          store(&output_ptr_fp8_scale2[istore], to<__nv_fp8x4_e4m3>(split_input[0]));
          store(&output_ptr_fp8_scale2[istore + kItemPer16B / 2],
                to<__nv_fp8x4_e4m3>(split_input[1]));
#pragma unroll
          for (int i = 0; i < kItemPer16B; i++) {
            reg_input[iter][i] *= inv_scale;
          }
        }

        store(&output_ptr_fp8[istore], to<__nv_fp8x4_e4m3>(split_input[0]));
        store(&output_ptr_fp8[istore + kItemPer16B / 2], to<__nv_fp8x4_e4m3>(split_input[1]));
      }
    }
  }
}

}  // namespace kernels

bool fused_rmsnorm_with_scale_async(const void *input_ptr, const void *weight_ptr, void *output_ptr,
                                    void *output_fp32_ptr, void *output_fp8_scale2_ptr,
                                    const void *scale, float eps, int batch_size, int hidden_states,
                                    bool is_moe, cudaStream_t stream) {
  constexpr int kWarpCount = 4;
  constexpr int kWarpSize = 32;

  using Tin = const __nv_bfloat16;
  using Tout = __nv_fp8_e4m3;

  dim3 block(kWarpSize * kWarpCount);
  if (hidden_states == 5120) {
    constexpr int kHiddenStates = 5120;
    constexpr int kWarpPerBatch = 4;
    constexpr int kBatchPerBlock = kWarpCount / kWarpPerBatch;
    dim3 grid((batch_size + kBatchPerBlock - 1) / kBatchPerBlock);
    if (is_moe) {
      constexpr bool kIsMoe = true;
      kernels::fused_rmsnorm_with_scale<kHiddenStates, kWarpPerBatch, kBatchPerBlock, kIsMoe>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<Tin *>(input_ptr), reinterpret_cast<Tin *>(weight_ptr),
              reinterpret_cast<float *>(output_fp32_ptr), reinterpret_cast<Tout *>(output_ptr),
              reinterpret_cast<Tout *>(output_fp8_scale2_ptr),
              reinterpret_cast<const float *>(scale), eps, batch_size);
    } else {
      constexpr bool kIsMoe = false;
      kernels::fused_rmsnorm_with_scale<kHiddenStates, kWarpPerBatch, kBatchPerBlock, kIsMoe>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<Tin *>(input_ptr), reinterpret_cast<Tin *>(weight_ptr),
              reinterpret_cast<float *>(output_fp32_ptr), reinterpret_cast<Tout *>(output_ptr),
              reinterpret_cast<Tout *>(output_fp8_scale2_ptr),
              reinterpret_cast<const float *>(scale), eps, batch_size);
    }
  } else if (hidden_states == 4096) {
    constexpr int kHiddenStates = 4096;
    constexpr int kWarpPerBatch = 4;
    constexpr int kBatchPerBlock = kWarpCount / kWarpPerBatch;
    dim3 grid((batch_size + kBatchPerBlock - 1) / kBatchPerBlock);
    if (is_moe) {
      constexpr bool kIsMoe = true;
      kernels::fused_rmsnorm_with_scale<kHiddenStates, kWarpPerBatch, kBatchPerBlock, kIsMoe>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<Tin *>(input_ptr), reinterpret_cast<Tin *>(weight_ptr),
              reinterpret_cast<float *>(output_fp32_ptr), reinterpret_cast<Tout *>(output_ptr),
              reinterpret_cast<Tout *>(output_fp8_scale2_ptr),
              reinterpret_cast<const float *>(scale), eps, batch_size);
    } else {
      constexpr bool kIsMoe = false;
      kernels::fused_rmsnorm_with_scale<kHiddenStates, kWarpPerBatch, kBatchPerBlock, kIsMoe>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<Tin *>(input_ptr), reinterpret_cast<Tin *>(weight_ptr),
              reinterpret_cast<float *>(output_fp32_ptr), reinterpret_cast<Tout *>(output_ptr),
              reinterpret_cast<Tout *>(output_fp8_scale2_ptr),
              reinterpret_cast<const float *>(scale), eps, batch_size);
    }
  } else if (hidden_states == 320) {
    constexpr int kHiddenStates = 320;
    constexpr int kWarpPerBatch = 1;
    constexpr int kBatchPerBlock = kWarpCount / kWarpPerBatch;
    dim3 grid((batch_size + kBatchPerBlock - 1) / kBatchPerBlock);
    if (is_moe) {
      constexpr bool kIsMoe = true;
      kernels::fused_rmsnorm_with_scale<kHiddenStates, kWarpPerBatch, kBatchPerBlock, kIsMoe>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<Tin *>(input_ptr), reinterpret_cast<Tin *>(weight_ptr),
              reinterpret_cast<float *>(output_fp32_ptr), reinterpret_cast<Tout *>(output_ptr),
              reinterpret_cast<Tout *>(output_fp8_scale2_ptr),
              reinterpret_cast<const float *>(scale), eps, batch_size);
    } else {
      constexpr bool kIsMoe = false;
      kernels::fused_rmsnorm_with_scale<kHiddenStates, kWarpPerBatch, kBatchPerBlock, kIsMoe>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<Tin *>(input_ptr), reinterpret_cast<Tin *>(weight_ptr),
              reinterpret_cast<float *>(output_fp32_ptr), reinterpret_cast<Tout *>(output_ptr),
              reinterpret_cast<Tout *>(output_fp8_scale2_ptr),
              reinterpret_cast<const float *>(scale), eps, batch_size);
    }
  } else {
    std::cout << "not supported hidden_size for fused_rmsnorm_with_scale_async:" << hidden_states
              << std::endl;
    return false;
  }

  return true;
}

}  // namespace normalization
}  // namespace hpc
