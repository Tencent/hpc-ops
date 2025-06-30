#include <cuda.h>

#include "src/normalization/fused_rms_norm_with_scale/fused_rms_norm_with_scale.h"
#include "src/ptx_wrapper.cuh"
#include "src/utils.cuh"

namespace hpc {
namespace normalization {
namespace fused_rms_norm_with_scale {

namespace kernels {

template <int kHiddenStates, int kWarpPerBatch, int kBatchPerBlock, bool kIsMoe>
__global__ void fused_rms_norm_with_scale(
    __nv_bfloat16 *input_ptr, __nv_bfloat16 *weight_ptr, float *output_ptr_fp32,
    __nv_fp8_e4m3 *output_ptr_fp8, __nv_fp8_e4m3 *output_ptr_fp8_scale2,
    float *scale, float eps, int batch_size) {
  constexpr int kItemPer16B = 8;
  constexpr float kInvHiddenStates = 1.0f / kHiddenStates;
  constexpr int kIterPerBatch =
      UP_DIV(kHiddenStates, kWarpPerBatch * kWarpSize * kItemPer16B);
  float inv_scale = ptx::rcp_ftz(scale[0]);
  float inv_scale2 = 1;
  if constexpr (kIsMoe) {
    inv_scale2 = ptx::rcp_ftz(scale[1]);
    inv_scale *= scale[1];
  }

  const int iwarp = threadIdx.x / kWarpSize;
  const int ilane = threadIdx.x % kWarpSize;

  const uint64_t ibatch = blockIdx.x * kBatchPerBlock + iwarp / kWarpPerBatch;
  const uint64_t icol_thr_offset =
      ((iwarp % kWarpPerBatch) * kWarpSize + ilane) * kItemPer16B;

  __shared__ float smem_sum[kWarpPerBatch];

  float reg_input[kIterPerBatch * kItemPer16B];
  float reg_weight[kIterPerBatch * kItemPer16B];

#pragma unroll
  for (int i = 0; i < kIterPerBatch * kItemPer16B; i++) {
    reg_input[i] = 0;
    reg_weight[i] = 0;
  }

#pragma unroll
  for (int iter = 0; iter < kIterPerBatch; iter++) {
    uint64_t icol =
        icol_thr_offset + iter * kWarpPerBatch * kWarpSize * kItemPer16B;

    if (icol < kHiddenStates) {
      utils::load_16B_from_bf16_to_float(&reg_weight[iter * kItemPer16B],
                                         &weight_ptr[icol]);
    }
  }

  float local_mean = 0.0f;

  if (ibatch < batch_size) {
#pragma unroll
    for (int iter = 0; iter < kIterPerBatch; iter++) {
      uint64_t icol =
          icol_thr_offset + iter * kWarpPerBatch * kWarpSize * kItemPer16B;

      if (icol < kHiddenStates) {
        utils::load_16B_from_bf16_to_float(
            &reg_input[iter * kItemPer16B],
            &input_ptr[ibatch * kHiddenStates + icol]);
#pragma unroll
        for (int i = 0; i < kItemPer16B; i++) {
          local_mean += reg_input[iter * kItemPer16B + i] *
                        reg_input[iter * kItemPer16B + i];
        }
      }
    }
  }

  // warp reduce
#pragma unroll
  for (int mask = 16; mask >= 1; mask /= 2) {
    local_mean += __shfl_xor_sync((uint32_t)-1, local_mean, mask);
  }

  if (ilane == 0) {
    smem_sum[iwarp] = local_mean;
  }
  __syncthreads();

  int first_warp_in_batch = (iwarp / kWarpPerBatch) * kWarpPerBatch;
#pragma unroll
  for (int iwarp_in_batch = 0; iwarp_in_batch < kWarpPerBatch;
       iwarp_in_batch++) {
    int reduce_warp = first_warp_in_batch + iwarp_in_batch;
    if (iwarp != reduce_warp) {
      local_mean += smem_sum[reduce_warp];
    }
  }

  local_mean = ptx::rsqrt_ftz(local_mean * kInvHiddenStates + eps);

  if constexpr (!kIsMoe) {
    local_mean *= inv_scale;
  }

  if (ibatch < batch_size) {
#pragma unroll
    for (int iter = 0; iter < kIterPerBatch; iter++) {
      uint64_t icol =
          icol_thr_offset + iter * kWarpPerBatch * kWarpSize * kItemPer16B;
      uint64_t istore = ibatch * kHiddenStates + icol;

      if (icol < kHiddenStates) {
#pragma unroll
        for (int i = 0; i < kItemPer16B; i++) {
          reg_input[iter * kItemPer16B + i] =
              reg_input[iter * kItemPer16B + i] *
              reg_weight[iter * kItemPer16B + i] * local_mean;
        }

        if constexpr (kIsMoe) {
          utils::store_16B_to_float_from_float(&output_ptr_fp32[istore],
                                               &reg_input[iter * kItemPer16B]);
          utils::store_16B_to_float_from_float(
              &output_ptr_fp32[istore + kItemPer16B / 2],
              &reg_input[iter * kItemPer16B + kItemPer16B / 2]);
#pragma unroll
          for (int i = 0; i < kItemPer16B; i++) {
            reg_input[iter * kItemPer16B + i] *= inv_scale2;
          }
          utils::store_8B_to_fp8e4m3_from_float(&output_ptr_fp8_scale2[istore],
                                                &reg_input[iter * kItemPer16B]);
#pragma unroll
          for (int i = 0; i < kItemPer16B; i++) {
            reg_input[iter * kItemPer16B + i] *= inv_scale;
          }
        }

        utils::store_8B_to_fp8e4m3_from_float(&output_ptr_fp8[istore],
                                              &reg_input[iter * kItemPer16B]);
      }
    }
  }
}

}  // namespace kernels

#define LAUNCH_KERNELS()                                                  \
  {                                                                       \
    auto kernel =                                                         \
        &kernels::fused_rms_norm_with_scale<kHiddenStates, kWarpPerBatch, \
                                            kBatchPerBlock, kIsMoe>;      \
    dim3 grid(UP_DIV(batch_size, kBatchPerBlock));                        \
    kernel<<<grid, block, 0, stream>>>(                                   \
        reinterpret_cast<Tin *>(input_ptr),                               \
        reinterpret_cast<Tin *>(weight_ptr),                              \
        reinterpret_cast<float *>(output_fp32_ptr),                       \
        reinterpret_cast<Tout *>(output_ptr),                             \
        reinterpret_cast<Tout *>(output_fp8_scale2_ptr),                  \
        reinterpret_cast<float *>(scale), eps, batch_size);               \
  }

#define LAUNCH_KERNELS_SWITCH_OUTPUT_HIGH() \
  {                                         \
    if (is_moe) {                           \
      constexpr bool kIsMoe = true;         \
      LAUNCH_KERNELS()                      \
    } else {                                \
      constexpr bool kIsMoe = false;        \
      LAUNCH_KERNELS()                      \
    }                                       \
  }

#define LAUNCH_KERNELS_SWITCH_HIDDEN_STATES()                      \
  {                                                                \
    switch (hidden_states) {                                       \
      case 5120: {                                                 \
        constexpr int kHiddenStates = 5120;                        \
        constexpr int kWarpPerBatch = 4;                           \
        constexpr int kBatchPerBlock = kWarpCount / kWarpPerBatch; \
        LAUNCH_KERNELS_SWITCH_OUTPUT_HIGH()                        \
        break;                                                     \
      }                                                            \
      case 320: {                                                  \
        constexpr int kHiddenStates = 320;                         \
        constexpr int kWarpPerBatch = 1;                           \
        constexpr int kBatchPerBlock = kWarpCount / kWarpPerBatch; \
        LAUNCH_KERNELS_SWITCH_OUTPUT_HIGH()                        \
        break;                                                     \
      }                                                            \
      default: {                                                   \
        throw std::invalid_argument("not support hidden_states!"); \
        break;                                                     \
      }                                                            \
    }                                                              \
  }

template <typename Tin, typename Tout>
void fused_rms_norm_with_scale_async(void *input_ptr, void *weight_ptr,
                                     void *output_ptr, void *output_fp32_ptr,
                                     void *output_fp8_scale2_ptr, void *scale,
                                     float eps, int batch_size,
                                     int hidden_states, bool is_moe,
                                     cudaStream_t stream) {
  constexpr int kWarpCount = 4;
  dim3 block(kWarpSize * kWarpCount);
  LAUNCH_KERNELS_SWITCH_HIDDEN_STATES();
}

template void fused_rms_norm_with_scale_async<__nv_bfloat16, __nv_fp8_e4m3>(
    void *, void *, void *, void *, void *, void *, float, int, int, bool,
    cudaStream_t);

}  // namespace fused_rms_norm_with_scale
}  // namespace normalization
}  // namespace hpc
