// Copyright 2025 hpc-ops authors
#include <cuda.h>

#include <iostream>

#include "src/normalization/fused_layer_norm_with_scale_quant.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace normalization {

namespace kernels {

template <int kHiddenStates, int kWarpPerBatch, int kBatchPerBlock, bool kIsAffine,
          bool kUsePreScale, bool kUsePostScale, int kGroupSize>
__global__ void fused_layer_norm_with_scale_quant(
    const __nv_bfloat16 *input_ptr, const __nv_bfloat16 *weight_ptr, const __nv_bfloat16 *bias_ptr,
    __nv_fp8_e4m3 *output_ptr, float *quant_scale, __nv_bfloat16 *output_x_ptr,
    const __nv_bfloat16 *pre_norm_scale1_ptr, const __nv_bfloat16 *pre_norm_scale2_ptr,
    const __nv_bfloat16 *post_norm_scale_ptr, const __nv_bfloat16 *post_norm_bias_scale_ptr,
    float eps, float quant_eps, float fp8_e4m3_max, float fp8_e4m3_min, int batch_size) {
  using bf162 = __nv_bfloat162;
  constexpr int kWarpSize = 32;
  constexpr int kItemPer16B = 8;
  constexpr int kReadPerIter = 2;
  constexpr float kInvHiddenStates = 1.0f / kHiddenStates;
  constexpr int kIterPerBatch = (kHiddenStates + kWarpPerBatch * kWarpSize * kItemPer16B - 1) /
                                (kWarpPerBatch * kWarpSize * kItemPer16B);

  const int iwarp = threadIdx.x / kWarpSize;
  const int ilane = threadIdx.x % kWarpSize;

  const int ibatch = blockIdx.x * kBatchPerBlock + iwarp / kWarpPerBatch;
  const int icol_thr_offset = ((iwarp % kWarpPerBatch) * kWarpSize + ilane) * kItemPer16B;

  __shared__ float smem_sum[kWarpPerBatch * kBatchPerBlock];
  __shared__ float smem_sum_2[kWarpPerBatch * kBatchPerBlock];

  vec_t<bf162, kItemPer16B / 2> input_bf162[kReadPerIter];
  vec_t<bf162, kItemPer16B / 2> weight_bf162[kReadPerIter];
  vec_t<bf162, kItemPer16B / 2> bias_bf162[kReadPerIter];
  vec_t<bf162, kItemPer16B / 2> pre_norm_scale1_bf162[kReadPerIter];
  vec_t<bf162, kItemPer16B / 2> pre_norm_scale2_bf162[kReadPerIter];
  vec_t<bf162, kItemPer16B / 2> post_norm_scale_bf162[kReadPerIter];
  vec_t<bf162, kItemPer16B / 2> post_norm_bias_scale_bf162[kReadPerIter];
  vec_t<float, kItemPer16B> input[kIterPerBatch];
  vec_t<float, kItemPer16B> weight[kReadPerIter];
  vec_t<float, kItemPer16B> bias[kReadPerIter];
  vec_t<float, kItemPer16B> pre_norm_scale1[kReadPerIter];
  vec_t<float, kItemPer16B> pre_norm_scale2[kReadPerIter];
  vec_t<float, kItemPer16B> post_norm_scale[kReadPerIter];
  vec_t<float, kItemPer16B> post_norm_bias_scale[kReadPerIter];

  float local_mean = 0.0f;
  float local_mean_2 = 0.0f;

  if (ibatch < batch_size) {
#pragma unroll
    for (int iter = 0; iter < kIterPerBatch; iter += kReadPerIter) {
      for (int load_idx = 0; load_idx < kReadPerIter; load_idx++) {
        int icol = icol_thr_offset + (iter + load_idx) * kWarpPerBatch * kWarpSize * kItemPer16B;
        if (icol < kHiddenStates) {
          if constexpr (kUsePreScale) {
            pre_norm_scale1_bf162[load_idx] =
                load<bf162, kItemPer16B / 2>(pre_norm_scale1_ptr + (ibatch * kHiddenStates + icol));
            pre_norm_scale2_bf162[load_idx] =
                load<bf162, kItemPer16B / 2>(pre_norm_scale2_ptr + (ibatch * kHiddenStates + icol));
          }
          input_bf162[load_idx] =
              load<bf162, kItemPer16B / 2>(input_ptr + (ibatch * kHiddenStates + icol));
        }
      }
#pragma unroll
      for (int read_idx = 0; read_idx < kReadPerIter; read_idx++) {
        int icol = icol_thr_offset + (iter + read_idx) * kWarpPerBatch * kWarpSize * kItemPer16B;
        if (icol < kHiddenStates) {
          input[iter + read_idx] = to<float>(input_bf162[read_idx]);
          if constexpr (kUsePreScale) {
            pre_norm_scale1[read_idx] = to<float>(pre_norm_scale1_bf162[read_idx]);
            pre_norm_scale2[read_idx] = to<float>(pre_norm_scale2_bf162[read_idx]);
          }
#pragma unroll
          for (int i = 0; i < kItemPer16B; i++) {
            if constexpr (kUsePreScale) {
              input[iter + read_idx][i] +=
                  pre_norm_scale1[read_idx][i] * pre_norm_scale2[read_idx][i];
            }
            local_mean += input[iter + read_idx][i];
            local_mean_2 += input[iter + read_idx][i] * input[iter + read_idx][i];
          }
          store(output_x_ptr + (ibatch * kHiddenStates + icol),
                to<__nv_bfloat16>(input[iter + read_idx]));
        }
      }
    }
  }

  // warp reduce
  local_mean = warp_reduce_sum_xor(local_mean);
  local_mean_2 = warp_reduce_sum_xor(local_mean_2);

  if (ilane == 0) {
    smem_sum[iwarp] = local_mean;
    smem_sum_2[iwarp] = local_mean_2;
  }
  __syncthreads();

  int first_warp_in_batch = (iwarp / kWarpPerBatch) * kWarpPerBatch;
#pragma unroll
  for (int iwarp_in_batch = 0; iwarp_in_batch < kWarpPerBatch; iwarp_in_batch++) {
    int reduce_warp = first_warp_in_batch + iwarp_in_batch;
    if (iwarp != reduce_warp) {
      local_mean += smem_sum[reduce_warp];
      local_mean_2 += smem_sum_2[reduce_warp];
    }
  }

  local_mean = local_mean * kInvHiddenStates;
  local_mean_2 = local_mean_2 * kInvHiddenStates;

  float local_var = 0.0f;
  local_var = local_mean_2 - local_mean * local_mean;
  local_var = rsqrtf_ftz(local_var + eps);

  if (ibatch < batch_size) {
#pragma unroll
    for (int iter = 0; iter < kIterPerBatch; iter += kReadPerIter) {
#pragma unroll
      for (int write_idx = 0; write_idx < kReadPerIter; write_idx++) {
        int icol = icol_thr_offset + (iter + write_idx) * kWarpPerBatch * kWarpSize * kItemPer16B;
        if (icol < kHiddenStates) {
          if constexpr (kIsAffine) {
            weight_bf162[write_idx] = load<bf162, kItemPer16B / 2>(weight_ptr + icol);
            bias_bf162[write_idx] = load<bf162, kItemPer16B / 2>(bias_ptr + icol);
          }
          if constexpr (kUsePostScale) {
            post_norm_scale_bf162[write_idx] =
                load<bf162, kItemPer16B / 2>(post_norm_scale_ptr + (ibatch * kHiddenStates + icol));
            post_norm_bias_scale_bf162[write_idx] = load<bf162, kItemPer16B / 2>(
                post_norm_bias_scale_ptr + (ibatch * kHiddenStates + icol));
          }
        }
      }
#pragma unroll
      for (int load_idx = 0; load_idx < kReadPerIter; load_idx++) {
        int icol = icol_thr_offset + (iter + load_idx) * kWarpPerBatch * kWarpSize * kItemPer16B;
        if (icol < kHiddenStates) {
#pragma unroll
          for (int i = 0; i < kItemPer16B; i++) {
            input[iter + load_idx][i] = (input[iter + load_idx][i] - local_mean) * local_var;
            if constexpr (kIsAffine) {
              weight[load_idx] = to<float>(weight_bf162[load_idx]);
              bias[load_idx] = to<float>(bias_bf162[load_idx]);
              input[iter + load_idx][i] =
                  input[iter + load_idx][i] * weight[load_idx][i] + bias[load_idx][i];
            }
            if constexpr (kUsePostScale) {
              post_norm_scale[load_idx] = to<float>(post_norm_scale_bf162[load_idx]);
              post_norm_bias_scale[load_idx] = to<float>(post_norm_bias_scale_bf162[load_idx]);
              input[iter + load_idx][i] = input[iter + load_idx][i] * post_norm_scale[load_idx][i] +
                                          post_norm_bias_scale[load_idx][i];
            }
          }
        }
      }
    }

    // quant part
    // per thread read kIterPerBatch discontinuous group's data
    vec_t<float, kIterPerBatch> local_max;
    // kGroupSize = 128, kItemPer16B = 8, therefore each warp processes 8 * 32 / 128 = 2 consecutive
    // groups, and there are kIterPerBatch of such consecutive groups.
    constexpr int kHalfWarpSize = 16;

#pragma unroll
    for (int i = 0; i < kIterPerBatch; i++) {
      local_max[i] = 0;
    }

// find max abs value in kIterPerBatch group for each thread
#pragma unroll
    for (int iter = 0; iter < kIterPerBatch; iter++) {
      int icol = icol_thr_offset + iter * kWarpPerBatch * kWarpSize * kItemPer16B;
      if (icol < kHiddenStates) {
#pragma unroll
        for (int i = 0; i < kItemPer16B; i++) {
          local_max[iter] = fmaxf(local_max[iter], fabsf(input[iter][i]));
        }
      }
    }

// thread in 0~15 read data idx 0~127, thread in 16~31 read data idx 128~255
#pragma unroll
    for (int iter = 0; iter < kIterPerBatch; iter++) {
      local_max[iter] = half_warp_reduce_max_down(local_max[iter]);
    }
    __syncthreads();

#pragma unroll
    for (int iter = 0; iter < kIterPerBatch; iter++) {
      local_max[iter] = fmaxf(local_max[iter] * rcpf_ftz(fp8_e4m3_max), quant_eps);
      int icol = icol_thr_offset + iter * kWarpPerBatch * kWarpSize * kItemPer16B;
      int istore = ibatch * kHiddenStates + icol;
      int igroup_store =
          ibatch * ((kHiddenStates + kGroupSize - 1) / kGroupSize) + icol / kGroupSize;
      if (icol < kHiddenStates) {
        if (ilane % kHalfWarpSize == 0) {
          store(quant_scale + igroup_store, local_max[iter]);
        }
#pragma unroll
        for (int i = 0; i < kItemPer16B; i++) {
          input[iter][i] = fminf(fmaxf((input[iter][i] * rcpf_ftz(local_max[iter])), fp8_e4m3_min),
                                 fp8_e4m3_max);
        }
        auto split_input = reshape<2, kItemPer16B / 2>(input[iter]);
        store(output_ptr + istore, to<__nv_fp8x4_e4m3>(split_input[0]));
        store(output_ptr + istore + kItemPer16B / 2, to<__nv_fp8x4_e4m3>(split_input[1]));
      }
    }
  }
}

}  // namespace kernels

bool fused_layer_norm_with_scale_quant_async(
    const void *input_ptr, const void *weight_ptr, const void *bias_ptr, void *output_ptr,
    const void *pre_norm_scale1_ptr, const void *pre_norm_scale2_ptr,
    const void *post_norm_scale_ptr, const void *post_norm_bias_scale_ptr, void *quant_scale,
    void *output_x_ptr, float eps, float quant_eps, int batch_size, int hidden_states,
    int group_size, float fp8_e4m3_max, float fp8_e4m3_min, bool is_elementwise_affine,
    bool use_pre_norm_scale, bool use_post_norm_scale, cudaStream_t stream) {
  constexpr int kWarpCount = 8;
  constexpr int kWarpSize = 32;

  using Tin = const __nv_bfloat16;
  using Tout = __nv_fp8_e4m3;

  dim3 block(kWarpSize * kWarpCount);
  if (hidden_states == 5120 && group_size == 128) {
    constexpr int kHiddenStates = 5120;
    constexpr int kWarpPerBatch = 4;
    constexpr int kBatchPerBlock = kWarpCount / kWarpPerBatch;
    constexpr int kGroupSize = 128;
    dim3 grid((batch_size + kBatchPerBlock - 1) / kBatchPerBlock);
    if (is_elementwise_affine && use_pre_norm_scale && use_post_norm_scale) {
      constexpr bool kIsAffine = true;
      constexpr bool kUsePreScale = true;
      constexpr bool kUsePostScale = true;
      kernels::fused_layer_norm_with_scale_quant<kHiddenStates, kWarpPerBatch, kBatchPerBlock,
                                                 kIsAffine, kUsePreScale, kUsePostScale, kGroupSize>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<Tin *>(input_ptr), reinterpret_cast<Tin *>(weight_ptr),
              reinterpret_cast<Tin *>(bias_ptr), reinterpret_cast<Tout *>(output_ptr),
              reinterpret_cast<float *>(quant_scale),
              reinterpret_cast<__nv_bfloat16 *>(output_x_ptr),
              reinterpret_cast<Tin *>(pre_norm_scale1_ptr),
              reinterpret_cast<Tin *>(pre_norm_scale2_ptr),
              reinterpret_cast<Tin *>(post_norm_scale_ptr),
              reinterpret_cast<Tin *>(post_norm_bias_scale_ptr), eps, quant_eps, fp8_e4m3_max,
              fp8_e4m3_min, batch_size);
    } else if (is_elementwise_affine && !use_pre_norm_scale && use_post_norm_scale) {
      constexpr bool kIsAffine = true;
      constexpr bool kUsePreScale = false;
      constexpr bool kUsePostScale = true;
      kernels::fused_layer_norm_with_scale_quant<kHiddenStates, kWarpPerBatch, kBatchPerBlock,
                                                 kIsAffine, kUsePreScale, kUsePostScale, kGroupSize>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<Tin *>(input_ptr), reinterpret_cast<Tin *>(weight_ptr),
              reinterpret_cast<Tin *>(bias_ptr), reinterpret_cast<Tout *>(output_ptr),
              reinterpret_cast<float *>(quant_scale),
              reinterpret_cast<__nv_bfloat16 *>(output_x_ptr),
              reinterpret_cast<Tin *>(pre_norm_scale1_ptr),
              reinterpret_cast<Tin *>(pre_norm_scale2_ptr),
              reinterpret_cast<Tin *>(post_norm_scale_ptr),
              reinterpret_cast<Tin *>(post_norm_bias_scale_ptr), eps, quant_eps, fp8_e4m3_max,
              fp8_e4m3_min, batch_size);
    } else if (!is_elementwise_affine && use_pre_norm_scale && use_post_norm_scale) {
      constexpr bool kIsAffine = false;
      constexpr bool kUsePreScale = true;
      constexpr bool kUsePostScale = true;
      kernels::fused_layer_norm_with_scale_quant<kHiddenStates, kWarpPerBatch, kBatchPerBlock,
                                                 kIsAffine, kUsePreScale, kUsePostScale, kGroupSize>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<Tin *>(input_ptr), reinterpret_cast<Tin *>(weight_ptr),
              reinterpret_cast<Tin *>(bias_ptr), reinterpret_cast<Tout *>(output_ptr),
              reinterpret_cast<float *>(quant_scale),
              reinterpret_cast<__nv_bfloat16 *>(output_x_ptr),
              reinterpret_cast<Tin *>(pre_norm_scale1_ptr),
              reinterpret_cast<Tin *>(pre_norm_scale2_ptr),
              reinterpret_cast<Tin *>(post_norm_scale_ptr),
              reinterpret_cast<Tin *>(post_norm_bias_scale_ptr), eps, quant_eps, fp8_e4m3_max,
              fp8_e4m3_min, batch_size);
    } else if (!is_elementwise_affine && !use_pre_norm_scale && use_post_norm_scale) {
      constexpr bool kIsAffine = false;
      constexpr bool kUsePreScale = false;
      constexpr bool kUsePostScale = true;
      kernels::fused_layer_norm_with_scale_quant<kHiddenStates, kWarpPerBatch, kBatchPerBlock,
                                                 kIsAffine, kUsePreScale, kUsePostScale, kGroupSize>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<Tin *>(input_ptr), reinterpret_cast<Tin *>(weight_ptr),
              reinterpret_cast<Tin *>(bias_ptr), reinterpret_cast<Tout *>(output_ptr),
              reinterpret_cast<float *>(quant_scale),
              reinterpret_cast<__nv_bfloat16 *>(output_x_ptr),
              reinterpret_cast<Tin *>(pre_norm_scale1_ptr),
              reinterpret_cast<Tin *>(pre_norm_scale2_ptr),
              reinterpret_cast<Tin *>(post_norm_scale_ptr),
              reinterpret_cast<Tin *>(post_norm_bias_scale_ptr), eps, quant_eps, fp8_e4m3_max,
              fp8_e4m3_min, batch_size);
    } else if (is_elementwise_affine && use_pre_norm_scale && !use_post_norm_scale) {
      constexpr bool kIsAffine = true;
      constexpr bool kUsePreScale = true;
      constexpr bool kUsePostScale = false;
      kernels::fused_layer_norm_with_scale_quant<kHiddenStates, kWarpPerBatch, kBatchPerBlock,
                                                 kIsAffine, kUsePreScale, kUsePostScale, kGroupSize>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<Tin *>(input_ptr), reinterpret_cast<Tin *>(weight_ptr),
              reinterpret_cast<Tin *>(bias_ptr), reinterpret_cast<Tout *>(output_ptr),
              reinterpret_cast<float *>(quant_scale),
              reinterpret_cast<__nv_bfloat16 *>(output_x_ptr),
              reinterpret_cast<Tin *>(pre_norm_scale1_ptr),
              reinterpret_cast<Tin *>(pre_norm_scale2_ptr),
              reinterpret_cast<Tin *>(post_norm_scale_ptr),
              reinterpret_cast<Tin *>(post_norm_bias_scale_ptr), eps, quant_eps, fp8_e4m3_max,
              fp8_e4m3_min, batch_size);
    } else if (is_elementwise_affine && !use_pre_norm_scale && !use_post_norm_scale) {
      constexpr bool kIsAffine = true;
      constexpr bool kUsePreScale = false;
      constexpr bool kUsePostScale = false;
      kernels::fused_layer_norm_with_scale_quant<kHiddenStates, kWarpPerBatch, kBatchPerBlock,
                                                 kIsAffine, kUsePreScale, kUsePostScale, kGroupSize>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<Tin *>(input_ptr), reinterpret_cast<Tin *>(weight_ptr),
              reinterpret_cast<Tin *>(bias_ptr), reinterpret_cast<Tout *>(output_ptr),
              reinterpret_cast<float *>(quant_scale),
              reinterpret_cast<__nv_bfloat16 *>(output_x_ptr),
              reinterpret_cast<Tin *>(pre_norm_scale1_ptr),
              reinterpret_cast<Tin *>(pre_norm_scale2_ptr),
              reinterpret_cast<Tin *>(post_norm_scale_ptr),
              reinterpret_cast<Tin *>(post_norm_bias_scale_ptr), eps, quant_eps, fp8_e4m3_max,
              fp8_e4m3_min, batch_size);
    } else if (!is_elementwise_affine && use_pre_norm_scale && !use_post_norm_scale) {
      constexpr bool kIsAffine = false;
      constexpr bool kUsePreScale = true;
      constexpr bool kUsePostScale = false;
      kernels::fused_layer_norm_with_scale_quant<kHiddenStates, kWarpPerBatch, kBatchPerBlock,
                                                 kIsAffine, kUsePreScale, kUsePostScale, kGroupSize>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<Tin *>(input_ptr), reinterpret_cast<Tin *>(weight_ptr),
              reinterpret_cast<Tin *>(bias_ptr), reinterpret_cast<Tout *>(output_ptr),
              reinterpret_cast<float *>(quant_scale),
              reinterpret_cast<__nv_bfloat16 *>(output_x_ptr),
              reinterpret_cast<Tin *>(pre_norm_scale1_ptr),
              reinterpret_cast<Tin *>(pre_norm_scale2_ptr),
              reinterpret_cast<Tin *>(post_norm_scale_ptr),
              reinterpret_cast<Tin *>(post_norm_bias_scale_ptr), eps, quant_eps, fp8_e4m3_max,
              fp8_e4m3_min, batch_size);
    } else {
      constexpr bool kIsAffine = false;
      constexpr bool kUsePreScale = false;
      constexpr bool kUsePostScale = false;
      kernels::fused_layer_norm_with_scale_quant<kHiddenStates, kWarpPerBatch, kBatchPerBlock,
                                                 kIsAffine, kUsePreScale, kUsePostScale, kGroupSize>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<Tin *>(input_ptr), reinterpret_cast<Tin *>(weight_ptr),
              reinterpret_cast<Tin *>(bias_ptr), reinterpret_cast<Tout *>(output_ptr),
              reinterpret_cast<float *>(quant_scale),
              reinterpret_cast<__nv_bfloat16 *>(output_x_ptr),
              reinterpret_cast<Tin *>(pre_norm_scale1_ptr),
              reinterpret_cast<Tin *>(pre_norm_scale2_ptr),
              reinterpret_cast<Tin *>(post_norm_scale_ptr),
              reinterpret_cast<Tin *>(post_norm_bias_scale_ptr), eps, quant_eps, fp8_e4m3_max,
              fp8_e4m3_min, batch_size);
    }
  } else if (hidden_states == 4096 && group_size == 128) {
    constexpr int kHiddenStates = 4096;
    constexpr int kWarpPerBatch = 4;
    constexpr int kBatchPerBlock = kWarpCount / kWarpPerBatch;
    constexpr int kGroupSize = 128;
    dim3 grid((batch_size + kBatchPerBlock - 1) / kBatchPerBlock);
    if (is_elementwise_affine && use_pre_norm_scale && use_post_norm_scale) {
      constexpr bool kIsAffine = true;
      constexpr bool kUsePreScale = true;
      constexpr bool kUsePostScale = true;
      kernels::fused_layer_norm_with_scale_quant<kHiddenStates, kWarpPerBatch, kBatchPerBlock,
                                                 kIsAffine, kUsePreScale, kUsePostScale, kGroupSize>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<Tin *>(input_ptr), reinterpret_cast<Tin *>(weight_ptr),
              reinterpret_cast<Tin *>(bias_ptr), reinterpret_cast<Tout *>(output_ptr),
              reinterpret_cast<float *>(quant_scale),
              reinterpret_cast<__nv_bfloat16 *>(output_x_ptr),
              reinterpret_cast<Tin *>(pre_norm_scale1_ptr),
              reinterpret_cast<Tin *>(pre_norm_scale2_ptr),
              reinterpret_cast<Tin *>(post_norm_scale_ptr),
              reinterpret_cast<Tin *>(post_norm_bias_scale_ptr), eps, quant_eps, fp8_e4m3_max,
              fp8_e4m3_min, batch_size);
    } else if (is_elementwise_affine && !use_pre_norm_scale && use_post_norm_scale) {
      constexpr bool kIsAffine = true;
      constexpr bool kUsePreScale = false;
      constexpr bool kUsePostScale = true;
      kernels::fused_layer_norm_with_scale_quant<kHiddenStates, kWarpPerBatch, kBatchPerBlock,
                                                 kIsAffine, kUsePreScale, kUsePostScale, kGroupSize>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<Tin *>(input_ptr), reinterpret_cast<Tin *>(weight_ptr),
              reinterpret_cast<Tin *>(bias_ptr), reinterpret_cast<Tout *>(output_ptr),
              reinterpret_cast<float *>(quant_scale),
              reinterpret_cast<__nv_bfloat16 *>(output_x_ptr),
              reinterpret_cast<Tin *>(pre_norm_scale1_ptr),
              reinterpret_cast<Tin *>(pre_norm_scale2_ptr),
              reinterpret_cast<Tin *>(post_norm_scale_ptr),
              reinterpret_cast<Tin *>(post_norm_bias_scale_ptr), eps, quant_eps, fp8_e4m3_max,
              fp8_e4m3_min, batch_size);
    } else if (!is_elementwise_affine && use_pre_norm_scale && use_post_norm_scale) {
      constexpr bool kIsAffine = false;
      constexpr bool kUsePreScale = true;
      constexpr bool kUsePostScale = true;
      kernels::fused_layer_norm_with_scale_quant<kHiddenStates, kWarpPerBatch, kBatchPerBlock,
                                                 kIsAffine, kUsePreScale, kUsePostScale, kGroupSize>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<Tin *>(input_ptr), reinterpret_cast<Tin *>(weight_ptr),
              reinterpret_cast<Tin *>(bias_ptr), reinterpret_cast<Tout *>(output_ptr),
              reinterpret_cast<float *>(quant_scale),
              reinterpret_cast<__nv_bfloat16 *>(output_x_ptr),
              reinterpret_cast<Tin *>(pre_norm_scale1_ptr),
              reinterpret_cast<Tin *>(pre_norm_scale2_ptr),
              reinterpret_cast<Tin *>(post_norm_scale_ptr),
              reinterpret_cast<Tin *>(post_norm_bias_scale_ptr), eps, quant_eps, fp8_e4m3_max,
              fp8_e4m3_min, batch_size);
    } else if (!is_elementwise_affine && !use_pre_norm_scale && use_post_norm_scale) {
      constexpr bool kIsAffine = false;
      constexpr bool kUsePreScale = false;
      constexpr bool kUsePostScale = true;
      kernels::fused_layer_norm_with_scale_quant<kHiddenStates, kWarpPerBatch, kBatchPerBlock,
                                                 kIsAffine, kUsePreScale, kUsePostScale, kGroupSize>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<Tin *>(input_ptr), reinterpret_cast<Tin *>(weight_ptr),
              reinterpret_cast<Tin *>(bias_ptr), reinterpret_cast<Tout *>(output_ptr),
              reinterpret_cast<float *>(quant_scale),
              reinterpret_cast<__nv_bfloat16 *>(output_x_ptr),
              reinterpret_cast<Tin *>(pre_norm_scale1_ptr),
              reinterpret_cast<Tin *>(pre_norm_scale2_ptr),
              reinterpret_cast<Tin *>(post_norm_scale_ptr),
              reinterpret_cast<Tin *>(post_norm_bias_scale_ptr), eps, quant_eps, fp8_e4m3_max,
              fp8_e4m3_min, batch_size);
    } else if (is_elementwise_affine && use_pre_norm_scale && !use_post_norm_scale) {
      constexpr bool kIsAffine = true;
      constexpr bool kUsePreScale = true;
      constexpr bool kUsePostScale = false;
      kernels::fused_layer_norm_with_scale_quant<kHiddenStates, kWarpPerBatch, kBatchPerBlock,
                                                 kIsAffine, kUsePreScale, kUsePostScale, kGroupSize>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<Tin *>(input_ptr), reinterpret_cast<Tin *>(weight_ptr),
              reinterpret_cast<Tin *>(bias_ptr), reinterpret_cast<Tout *>(output_ptr),
              reinterpret_cast<float *>(quant_scale),
              reinterpret_cast<__nv_bfloat16 *>(output_x_ptr),
              reinterpret_cast<Tin *>(pre_norm_scale1_ptr),
              reinterpret_cast<Tin *>(pre_norm_scale2_ptr),
              reinterpret_cast<Tin *>(post_norm_scale_ptr),
              reinterpret_cast<Tin *>(post_norm_bias_scale_ptr), eps, quant_eps, fp8_e4m3_max,
              fp8_e4m3_min, batch_size);
    } else if (is_elementwise_affine && !use_pre_norm_scale && !use_post_norm_scale) {
      constexpr bool kIsAffine = true;
      constexpr bool kUsePreScale = false;
      constexpr bool kUsePostScale = false;
      kernels::fused_layer_norm_with_scale_quant<kHiddenStates, kWarpPerBatch, kBatchPerBlock,
                                                 kIsAffine, kUsePreScale, kUsePostScale, kGroupSize>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<Tin *>(input_ptr), reinterpret_cast<Tin *>(weight_ptr),
              reinterpret_cast<Tin *>(bias_ptr), reinterpret_cast<Tout *>(output_ptr),
              reinterpret_cast<float *>(quant_scale),
              reinterpret_cast<__nv_bfloat16 *>(output_x_ptr),
              reinterpret_cast<Tin *>(pre_norm_scale1_ptr),
              reinterpret_cast<Tin *>(pre_norm_scale2_ptr),
              reinterpret_cast<Tin *>(post_norm_scale_ptr),
              reinterpret_cast<Tin *>(post_norm_bias_scale_ptr), eps, quant_eps, fp8_e4m3_max,
              fp8_e4m3_min, batch_size);
    } else if (!is_elementwise_affine && use_pre_norm_scale && !use_post_norm_scale) {
      constexpr bool kIsAffine = false;
      constexpr bool kUsePreScale = true;
      constexpr bool kUsePostScale = false;
      kernels::fused_layer_norm_with_scale_quant<kHiddenStates, kWarpPerBatch, kBatchPerBlock,
                                                 kIsAffine, kUsePreScale, kUsePostScale, kGroupSize>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<Tin *>(input_ptr), reinterpret_cast<Tin *>(weight_ptr),
              reinterpret_cast<Tin *>(bias_ptr), reinterpret_cast<Tout *>(output_ptr),
              reinterpret_cast<float *>(quant_scale),
              reinterpret_cast<__nv_bfloat16 *>(output_x_ptr),
              reinterpret_cast<Tin *>(pre_norm_scale1_ptr),
              reinterpret_cast<Tin *>(pre_norm_scale2_ptr),
              reinterpret_cast<Tin *>(post_norm_scale_ptr),
              reinterpret_cast<Tin *>(post_norm_bias_scale_ptr), eps, quant_eps, fp8_e4m3_max,
              fp8_e4m3_min, batch_size);
    } else {
      constexpr bool kIsAffine = false;
      constexpr bool kUsePreScale = false;
      constexpr bool kUsePostScale = false;
      kernels::fused_layer_norm_with_scale_quant<kHiddenStates, kWarpPerBatch, kBatchPerBlock,
                                                 kIsAffine, kUsePreScale, kUsePostScale, kGroupSize>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<Tin *>(input_ptr), reinterpret_cast<Tin *>(weight_ptr),
              reinterpret_cast<Tin *>(bias_ptr), reinterpret_cast<Tout *>(output_ptr),
              reinterpret_cast<float *>(quant_scale),
              reinterpret_cast<__nv_bfloat16 *>(output_x_ptr),
              reinterpret_cast<Tin *>(pre_norm_scale1_ptr),
              reinterpret_cast<Tin *>(pre_norm_scale2_ptr),
              reinterpret_cast<Tin *>(post_norm_scale_ptr),
              reinterpret_cast<Tin *>(post_norm_bias_scale_ptr), eps, quant_eps, fp8_e4m3_max,
              fp8_e4m3_min, batch_size);
    }
  } else {
    std::cout << "not supported hidden_size for fused_layer_norm_with_quant_async:" << hidden_states
              << std::endl;
    return false;
  }
  return true;
}

}  // namespace normalization
}  // namespace hpc
