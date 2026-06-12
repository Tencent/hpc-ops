// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include "cutlass/fast_math.h"
#include "src/activation/activation.h"
#include "src/utils/utils.cuh"
#include "src/utils/utils.h"

namespace hpc {
namespace activation {
namespace kernels {

template <bool kUseBFloat16PrecisionMultiply = true, bool kUsePDL = false>
__global__ void act_mul_and_quant_kernel(__nv_fp8_e4m3 *out_ptr, const __nv_bfloat16 *gate_up_ptr,
                                         const float *scale_ptr, const int *valid_row_range,
                                         const int num_row, const int num_col,
                                         cutlass::FastDivmod block1D22D) {
  int iblockx;
  int iblocky;

  block1D22D(iblocky, iblockx, blockIdx.x);
  int it = threadIdx.x + iblockx * blockDim.x;

  uint64_t irow = iblocky;
  int my_valid_row_end_exclusive = valid_row_range ? valid_row_range[0] : num_row;
  if (irow >= my_valid_row_end_exclusive) {
    return;
  }

  using T = __nv_bfloat162;

  float scale = scale_ptr[0];

  const auto *gate_row_ptr = gate_up_ptr + irow * num_col * 2;
  const auto *up_row_ptr = gate_row_ptr + num_col;
  auto *out_row_ptr = out_ptr + irow * num_col;

  int icol = it * 8;
  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }
  if (icol < num_col) {
    auto gate = to<float>(load<T, 4>(gate_row_ptr + icol));
    auto up = to<float>(load<T, 4>(up_row_ptr + icol));

    vec_t<float, 8> out;
#pragma unroll
    for (int i = 0; i < size(out); ++i) {
      auto g = gate[i];
      auto m = [&] {
        if constexpr (kUseBFloat16PrecisionMultiply) {
          auto u = __float2bfloat16_rn(up[i]);
          return __bfloat162float(__float2bfloat16_rn(silu(g)) * u);
        } else {
          auto u = up[i];
          return silu(g) * u;
        }
      }();
      out[i] = m * scale;
    }

    auto out_fp8 = to<__nv_fp8x4_e4m3>(out);
    store(out_row_ptr + icol, out_fp8);
  }
  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

template <int kNumWarpPerBlock, int kRowPerWarp = 1, bool kUseBFloat16PrecisionMultiply = true,
          bool kUsePDL = false>
__global__ void act_mul_and_quant_warp_per_row_kernel(__nv_fp8_e4m3 *out_ptr,
                                                      const __nv_bfloat16 *gate_up_ptr,
                                                      const float *scale_ptr,
                                                      const int *valid_row_range, const int num_row,
                                                      const int num_col) {
  int idx = threadIdx.x;
  int iblock = blockIdx.x;
  int ilane = idx % 32;
  int iwarp = idx / 32;

  int my_valid_row_end_exclusive = valid_row_range ? valid_row_range[0] : num_row;
  using T = __nv_bfloat162;

  float scale = scale_ptr[0];

  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }

  // Each warp now processes kRowPerWarp rows, collapsing the 128000-block
  // grid to 128000/kRowPerWarp blocks (launch + PDL overhead savings).
#pragma unroll
  for (int irow_off = 0; irow_off < kRowPerWarp; ++irow_off) {
    uint64_t irow = (uint64_t)iblock * kNumWarpPerBlock * kRowPerWarp +
                    (uint64_t)iwarp * kRowPerWarp + irow_off;
    if (irow >= (uint64_t)my_valid_row_end_exclusive) {
      continue;
    }

    const auto *gate_row_ptr = gate_up_ptr + irow * num_col * 2;
    const auto *up_row_ptr = gate_row_ptr + num_col;
    auto *out_row_ptr = out_ptr + irow * num_col;

    for (int icol = ilane * 8; icol < num_col; icol += 32 * 8) {
      auto gate = to<float>(load<T, 4>(gate_row_ptr + icol));
      auto up = to<float>(load<T, 4>(up_row_ptr + icol));

      vec_t<float, 8> out;
#pragma unroll
      for (int i = 0; i < size(out); ++i) {
        auto g = gate[i];
        auto m = [&] {
          if constexpr (kUseBFloat16PrecisionMultiply) {
            auto u = __float2bfloat16_rn(up[i]);
            return __bfloat162float(__float2bfloat16_rn(silu(g)) * u);
          } else {
            auto u = up[i];
            return silu(g) * u;
          }
        }();
        out[i] = m * scale;
      }

      auto out_fp8 = to<__nv_fp8x4_e4m3>(out);
      store(out_row_ptr + icol, out_fp8);
    }
  }
  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

template <bool kUsePDL = false>
__global__ void act_mul_and_blockwise_quant_kernel(const __nv_bfloat16 *gate_up_output_ptr,
                                                   __nv_fp8_e4m3 *output_ptr,
                                                   float *output_scale_ptr, const int num_row,
                                                   const int num_col,
                                                   cutlass::FastDivmod block1D22D) {
  int iblockx;
  int iblocky;

  block1D22D(iblocky, iblockx, blockIdx.x);
  int it = threadIdx.x + iblockx * blockDim.x;
  int irow = iblocky;
  int lane_id = threadIdx.x % 32;

  using T = __nv_bfloat162;

  const auto *gate_row_ptr = gate_up_output_ptr + irow * num_col * 2;
  const auto *up_row_ptr = gate_row_ptr + num_col;
  auto *out_row_ptr = output_ptr + irow * num_col;

  int icol = it * 8;

  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }
  if (icol < num_col) {
    auto gate = to<float>(load<T, 4>(gate_row_ptr + icol));
    auto up = to<float>(load<T, 4>(up_row_ptr + icol));

    vec_t<float, 8> out;
#pragma unroll
    for (int i = 0; i < size(out); ++i) {
      out[i] = silu(gate[i]) * up[i];
    }

    // get max value per 128 elements and cal scale
    float thread_max = 0.f;
#pragma unroll
    for (int i = 0; i < size(out); i++) {
      if (fabsf(out[i]) > thread_max) {
        thread_max = fabsf(out[i]);
      }
    }
    float max = half_warp_reduce_max_down(thread_max);
    float scale = max / 448.0f;
    float inv_scale = 1.0f / (scale + 1e-8f);

    // quant
#pragma unroll
    for (int i = 0; i < size(out); ++i) {
      out[i] *= inv_scale;
    }

    // store output
    auto out_fp8 = to<__nv_fp8x4_e4m3>(out);
    store(out_row_ptr + icol, out_fp8);

    // store scale
    if (lane_id == 0 || lane_id == 16) {
      auto *scale_addr = output_scale_ptr + irow * 1 + icol / 128 * num_row;
      store(scale_addr, scale);
    }
  }

  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

// Fused silu * mul * scale * fp8 quant on a vec_t<__nv_bfloat162, 4> chunk.
template <bool kUseBFloat16PrecisionMultiply>
__device__ __forceinline__ vec_t<__nv_fp8x4_e4m3, 2> silu_mul_quant_chunk(
    const vec_t<__nv_bfloat162, 4> &gate, const vec_t<__nv_bfloat162, 4> &up, float scale) {
  auto gate_f = to<float>(gate);
  vec_t<float, 8> out_f;
  if constexpr (kUseBFloat16PrecisionMultiply) {
    // silu(gate) and up are rounded to bf16 before the multiply to match the
    // bf16 matmul path; up is already bf16, so only silu is rounded.
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      __nv_bfloat162 s = __floats2bfloat162_rn(silu(gate_f[2 * i]), silu(gate_f[2 * i + 1]));
      float2 prod = __bfloat1622float2(__hmul2(s, up[i]));
      out_f[2 * i + 0] = prod.x * scale;
      out_f[2 * i + 1] = prod.y * scale;
    }
  } else {
    auto up_f = to<float>(up);
#pragma unroll
    for (int i = 0; i < size(out_f); ++i) {
      out_f[i] = silu(gate_f[i]) * up_f[i] * scale;
    }
  }
  return to<__nv_fp8x4_e4m3>(out_f);
}

// Per-expert persistent kernel: block b of an expert strides rows b, b + P, ...
// and stops at num_per_expert[e], so padded rows are skipped.
template <bool kUseBFloat16PrecisionMultiply = true, bool kUsePDL = false>
__global__ void masked_act_mul_and_quant_kernel(
    __nv_fp8_e4m3 *out_ptr, const __nv_bfloat16 *gate_up_ptr, const float *scale_ptr,
    const int *num_per_expert_ptr, const int num_intermediate_size, const int num_tokens_per_expert,
    const int num_block_per_expert, cutlass::FastDivmod Persist2EP,
    cutlass::FastDivmod Persist2PC) {
  using T = __nv_bfloat162;
  using LoadVec = vec_t<T, 4>;

  int rest_ep;
  int iblock_col;
  Persist2PC(rest_ep, iblock_col, blockIdx.x);

  int iexpert;
  int iblock_persist;
  Persist2EP(iexpert, iblock_persist, rest_ep);

  int n_tokens = num_per_expert_ptr[iexpert];
  if (n_tokens > num_tokens_per_expert) {
    n_tokens = num_tokens_per_expert;
  }
  if (n_tokens < 0) {
    n_tokens = 0;
  }
  if (iblock_persist >= n_tokens) {
    return;
  }

  const int icol = (iblock_col * blockDim.x + threadIdx.x) * 8;
  const bool has_col_work = icol < num_intermediate_size;

  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }

  if (has_col_work) {
    const float scale = scale_ptr[0];
    const uint64_t expert_row_base = (uint64_t)iexpert * num_tokens_per_expert;

    for (int t = iblock_persist; t < n_tokens; t += num_block_per_expert) {
      const auto *gate_row_ptr = gate_up_ptr + (expert_row_base + t) * (num_intermediate_size * 2);
      auto *out_row_ptr = out_ptr + (expert_row_base + t) * num_intermediate_size;
      LoadVec gate = load<T, 4>(gate_row_ptr + icol);
      LoadVec up = load<T, 4>(gate_row_ptr + num_intermediate_size + icol);
      auto out_fp8 = silu_mul_quant_chunk<kUseBFloat16PrecisionMultiply>(gate, up, scale);
      store(out_row_ptr + icol, out_fp8);
    }
  }

  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

__device__ __forceinline__ void get_group_id(int irow, const int *cu_num_tokens_per_group_ptr,
                                             int num_group, int &igroup, int &offset) {
  int left = 0;
  int right = num_group;
  while (left <= right) {
    int mid = left + (right - left) / 2;
    if (cu_num_tokens_per_group_ptr[mid] > irow) {
      right = mid - 1;
    } else {
      left = mid + 1;
    }
  }
  offset = irow - cu_num_tokens_per_group_ptr[right];
  igroup = right;
}

template <bool kUsePDL = false>
__global__ void act_mul_and_blockwise_quant_fusemoe_kernel(
    const __nv_bfloat16 *gate_up_output_ptr, __nv_fp8_e4m3 *output_ptr, float *output_scale_ptr,
    const int *cu_num_tokens_per_group_ptr, const int *cu_tiles_ptr, const int num_row,
    const int num_row_padded_size, const int num_col, const int num_group, const int ktile_m,
    cutlass::FastDivmod block1D22D) {
  int iblockx;
  int iblocky;

  block1D22D(iblocky, iblockx, blockIdx.x);
  int it = threadIdx.x + iblockx * blockDim.x;
  int irow = iblocky;

  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }

  if (irow >= cu_num_tokens_per_group_ptr[num_group]) {
    return;
  }
  int lane_id = threadIdx.x % 32;

  using T = __nv_bfloat162;

  const auto *gate_row_ptr = gate_up_output_ptr + irow * num_col * 2;
  const auto *up_row_ptr = gate_row_ptr + num_col;
  auto *out_row_ptr = output_ptr + irow * num_col;

  int icol = it * 8;

  if (icol < num_col) {
    auto gate = to<float>(load<T, 4>(gate_row_ptr + icol));
    auto up = to<float>(load<T, 4>(up_row_ptr + icol));

    vec_t<float, 8> out;
#pragma unroll
    for (int i = 0; i < size(out); ++i) {
      out[i] = silu(gate[i]) * up[i];
    }

    // get max value per 128 elements and cal scale
    float thread_max = 0.f;
#pragma unroll
    for (int i = 0; i < size(out); i++) {
      if (fabsf(out[i]) > thread_max) {
        thread_max = fabsf(out[i]);
      }
    }
    float max = half_warp_reduce_max_down(thread_max);
    float scale = max / 448.0f;
    float inv_scale = 1.0f / (scale + 1e-8f);

    // quant
#pragma unroll
    for (int i = 0; i < size(out); ++i) {
      out[i] *= inv_scale;
    }

    // store output
    auto out_fp8 = to<__nv_fp8x4_e4m3>(out);
    store(out_row_ptr + icol, out_fp8);

    // store scale
    if (lane_id == 0 || lane_id == 16) {
      int igroup = -1;
      int offset = -1;
      get_group_id(irow, cu_num_tokens_per_group_ptr, num_group, igroup, offset);
      if (igroup >= 0 && offset >= 0) {
        auto *scale_addr = output_scale_ptr + cu_tiles_ptr[igroup] * ktile_m + offset +
                           icol / 128 * num_row_padded_size;
        store(scale_addr, scale);
      }
    }
  }

  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

// input : gate + up
__global__ void masked_act_mul_and_blockwise_quant_kernel(
    __nv_fp8_e4m3 *output_ptr, float *output_scale_ptr, const __nv_bfloat16 *input_ptr,
    const int *num_per_expert_ptr, int num_total_tokens, int num_intermediate_size,
    int num_scale_size, int num_tokens_per_expert, cutlass::FastDivmod Block2YX,
    cutlass::FastDivmod Row2EandT, int num_block_row) {
  constexpr int kRows = 4;

  int iblockx;
  int iblocky;
  Block2YX(iblocky, iblockx, blockIdx.x);
  int lane_id = threadIdx.x % 32;

#pragma unroll 1
  for (int irow0 = iblocky * kRows; irow0 < num_total_tokens; irow0 += num_block_row * kRows) {
    int it = threadIdx.x + iblockx * blockDim.x;

#pragma unroll
    for (int i = 0; i < kRows; ++i) {
      int iexpert;
      int itoken;

      int irow = irow0 + i;

      Row2EandT(iexpert, itoken, irow);
      int num_tokens_curr_expert = num_per_expert_ptr[iexpert];
      if (itoken >= num_tokens_curr_expert) {
        continue;
      }

      const auto *gate_row_ptr = input_ptr + irow * (num_intermediate_size * 2);
      const auto *up_row_ptr = gate_row_ptr + num_intermediate_size;
      auto *output_row_ptr = output_ptr + irow * num_intermediate_size;
      auto *output_scale_row_ptr = output_scale_ptr + irow * num_scale_size;

      int icol = it * 8;
      if (icol < num_intermediate_size) {
        // 1. load gate an up
        auto gate = to<float>(load<__nv_bfloat162, 4>(gate_row_ptr + icol));
        auto up = to<float>(load<__nv_bfloat162, 4>(up_row_ptr + icol));
        decltype(gate) out;

        // 2. silu
#pragma unroll
        for (int i = 0; i < decltype(gate)::kNum; ++i) {
          out[i] = silu(gate[i]) * up[i];
        }

        // 3. get max value per 128 elements and cal scale
        float thread_max = 0.f;
#pragma unroll
        for (int i = 0; i < decltype(gate)::kNum; ++i) {
          if (fabsf(out[i]) > thread_max) {
            thread_max = fabsf(out[i]);
          }
        }
        float max = half_warp_reduce_max_down(thread_max);
        float scale = max / 448.0f;
        float inv_scale = 1.0f / (scale + 1e-8f);

        // 4. quant
#pragma unroll
        for (int i = 0; i < decltype(gate)::kNum; ++i) {
          out[i] *= inv_scale;
        }

        // 5. store output
        auto out_fp8 = to<__nv_fp8x4_e4m3>(out);
        store(output_row_ptr + icol, out_fp8);

        // 6. store output scale
        if (lane_id == 0 || lane_id == 16) {
          store(output_scale_row_ptr + icol / 128, scale);
        }
      }
    }  // for
  }  // irow0
}

}  // namespace kernels

void act_mul_and_quant_async(__nv_fp8_e4m3 *out_ptr, const __nv_bfloat16 *gate_up_ptr,
                             const float *scale_ptr, const int num_row, const int num_col,
                             bool use_bf16_mul, cudaStream_t stream) {
  act_mul_and_quant_async(out_ptr, gate_up_ptr, scale_ptr, nullptr, num_row, num_col, use_bf16_mul,
                          stream);
}

void act_mul_and_quant_async(__nv_fp8_e4m3 *out_ptr, const __nv_bfloat16 *gate_up_ptr,
                             const float *scale_ptr, const int *valid_row_range, const int num_row,
                             const int num_col, bool use_bf16_mul, cudaStream_t stream) {
  // num_col == 2128 x 2
  // gate + up

  int intermediate_size = num_col / 2;
  constexpr bool kUsePDL = true;

  cudaLaunchAttribute attribute[1];
  attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute[0].val.programmaticStreamSerializationAllowed = 1;

  // Set the attribute in a kernel launch configuration
  cudaLaunchConfig_t config{};

  if (intermediate_size >= 1024) {
    dim3 block(128);
    int num_col_block = (intermediate_size / 8 + block.x - 1) / block.x;
    cutlass::FastDivmod block1D22D(num_col_block);
    dim3 grid(num_row * num_col_block);

    // Base launch configuration
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 0;
    config.stream = stream;

    // Add special attribute for PDL
    config.attrs = attribute;
    config.numAttrs = 1;

    if (use_bf16_mul) {
      constexpr bool kUseBFloat16PrecisionMultiply = true;
      auto kernel = kernels::act_mul_and_quant_kernel<kUseBFloat16PrecisionMultiply, kUsePDL>;
      cudaLaunchKernelEx(&config, kernel, out_ptr, gate_up_ptr, scale_ptr, valid_row_range, num_row,
                         intermediate_size, block1D22D);
    } else {
      constexpr bool kUseBFloat16PrecisionMultiply = false;
      auto kernel = kernels::act_mul_and_quant_kernel<kUseBFloat16PrecisionMultiply, kUsePDL>;
      cudaLaunchKernelEx(&config, kernel, out_ptr, gate_up_ptr, scale_ptr, valid_row_range, num_row,
                         intermediate_size, block1D22D);
    }
  } else {
    // Scale down the grid by `kRowPerWarp` so we launch fewer blocks; each
    // warp still emits the same per-row access pattern but now handles
    // kRowPerWarp rows sequentially.  2D sweep over
    //   intermediate ∈ {64, 128, 192, 256, 512, 768}
    //   num_row     ∈ {384 .. 512000}
    // on L20A (SM100) showed:
    //   * intermediate ≤ 256 → K=2 wins (tiny perf loss at small num_row,
    //     ~1.5× speedup vs K=1 at large num_row, >5 TB/s bw).
    //   * intermediate ∈ [512, 1024) → K=1 wins (each warp already has
    //     ≥16 vec of work, further row-serialisation saturates ILP and
    //     hurts L2 locality).
    constexpr int kNumWarpPerBlock = 4;
    if (intermediate_size <= 256) {
      constexpr int kRowPerWarp = 2;
      constexpr int kRowPerBlock = kNumWarpPerBlock * kRowPerWarp;
      dim3 block(32 * kNumWarpPerBlock);
      dim3 grid((num_row + kRowPerBlock - 1) / kRowPerBlock);

      config.gridDim = grid;
      config.blockDim = block;
      config.dynamicSmemBytes = 0;
      config.stream = stream;
      config.attrs = attribute;
      config.numAttrs = 1;

      if (use_bf16_mul) {
        constexpr bool kUseBFloat16PrecisionMultiply = true;
        auto kernel =
            kernels::act_mul_and_quant_warp_per_row_kernel<kNumWarpPerBlock, kRowPerWarp,
                                                           kUseBFloat16PrecisionMultiply, kUsePDL>;
        cudaLaunchKernelEx(&config, kernel, out_ptr, gate_up_ptr, scale_ptr, valid_row_range,
                           num_row, intermediate_size);
      } else {
        constexpr bool kUseBFloat16PrecisionMultiply = false;
        auto kernel =
            kernels::act_mul_and_quant_warp_per_row_kernel<kNumWarpPerBlock, kRowPerWarp,
                                                           kUseBFloat16PrecisionMultiply, kUsePDL>;
        cudaLaunchKernelEx(&config, kernel, out_ptr, gate_up_ptr, scale_ptr, valid_row_range,
                           num_row, intermediate_size);
      }
    } else {
      // intermediate in [257, 1023]: per-warp workload is already high,
      // K=1 gives best ILP and L2 locality.
      constexpr int kRowPerWarp = 1;
      constexpr int kRowPerBlock = kNumWarpPerBlock * kRowPerWarp;
      dim3 block(32 * kNumWarpPerBlock);
      dim3 grid((num_row + kRowPerBlock - 1) / kRowPerBlock);

      config.gridDim = grid;
      config.blockDim = block;
      config.dynamicSmemBytes = 0;
      config.stream = stream;
      config.attrs = attribute;
      config.numAttrs = 1;

      if (use_bf16_mul) {
        constexpr bool kUseBFloat16PrecisionMultiply = true;
        auto kernel =
            kernels::act_mul_and_quant_warp_per_row_kernel<kNumWarpPerBlock, kRowPerWarp,
                                                           kUseBFloat16PrecisionMultiply, kUsePDL>;
        cudaLaunchKernelEx(&config, kernel, out_ptr, gate_up_ptr, scale_ptr, valid_row_range,
                           num_row, intermediate_size);
      } else {
        constexpr bool kUseBFloat16PrecisionMultiply = false;
        auto kernel =
            kernels::act_mul_and_quant_warp_per_row_kernel<kNumWarpPerBlock, kRowPerWarp,
                                                           kUseBFloat16PrecisionMultiply, kUsePDL>;
        cudaLaunchKernelEx(&config, kernel, out_ptr, gate_up_ptr, scale_ptr, valid_row_range,
                           num_row, intermediate_size);
      }
    }
  }
}

// Tunables for the per-expert persistent masked dispatcher.
namespace masked_act_dispatch {
constexpr int kNumWarpPerBlock = 8;  // reference block width for the grid budget
constexpr int kSmLoadFactor = 4;     // target ~num_sm * 4 active blocks
constexpr int kMinSparseP = 4;
constexpr int kSparsePAlignment = 8;

int div_round_up(int value, int divisor) { return (value + divisor - 1) / divisor; }

int round_up_to_multiple(int value, int multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

int clamp_int(int value, int lo, int hi) { return value < lo ? lo : (value > hi ? hi : value); }

// Smallest integer r with r*r >= value.  sqrt then correct for fp rounding.
int ceil_sqrt_int64(int64_t value) {
  if (value <= 0) {
    return 0;
  }
  int64_t r = static_cast<int64_t>(sqrtf(static_cast<float>(value)));
  while (r * r < value) {
    ++r;
  }
  while (r > 1 && (r - 1) * (r - 1) >= value) {
    --r;
  }
  return static_cast<int>(r);
}

// One lane handles 8 bf16 columns, so a row needs ceil(C / 8) lanes.
int pick_block_threads(int num_intermediate_size) {
  int vec_cols = (num_intermediate_size + 7) / 8;
  if (vec_cols <= 32) {
    return 32;
  }
  if (vec_cols <= 64) {
    return 64;
  }
  if (vec_cols <= 128) {
    return 128;
  }
  if (vec_cols <= 192) {
    return 192;
  }
  return 256;
}

// Persistent blocks per expert: P ~ avg for balanced routing, raised toward
// sqrt(avg * max) when a max hint marks the load as skewed, then clamped to the
// real token count and the partition budget.
int pick_sparse_p(int avg_tokens_per_expert, int max_tokens_per_expert, int num_tokens_per_expert,
                  int num_parallel_tokens, int sparse_col_tiles) {
  int target = avg_tokens_per_expert <= kMinSparseP
                   ? kMinSparseP
                   : round_up_to_multiple(avg_tokens_per_expert, kSparsePAlignment);

  if (max_tokens_per_expert > avg_tokens_per_expert) {
    const int capped_max = max_tokens_per_expert < num_tokens_per_expert ? max_tokens_per_expert
                                                                         : num_tokens_per_expert;
    const int skew_p = round_up_to_multiple(
        ceil_sqrt_int64((int64_t)avg_tokens_per_expert * capped_max), kSparsePAlignment);
    if (skew_p > target) {
      target = skew_p;
    }
  }

  int grid_budget = num_parallel_tokens * 3 / (2 * sparse_col_tiles);
  if (grid_budget < 1) {
    grid_budget = 1;
  }
  const int upper = num_tokens_per_expert < grid_budget ? num_tokens_per_expert : grid_budget;
  return clamp_int(target, 1, upper);
}

template <bool kUseBFloat16PrecisionMultiply, bool kUsePDL>
void launch_masked_act(cudaLaunchConfig_t *cfg, __nv_fp8_e4m3 *output_ptr,
                       const __nv_bfloat16 *input_ptr, const float *scale_ptr,
                       const int *num_per_expert_ptr, int num_intermediate_size,
                       int num_tokens_per_expert, int sparse_p, cutlass::FastDivmod Persist2EP,
                       cutlass::FastDivmod Persist2PC) {
  cudaLaunchKernelEx(
      cfg, kernels::masked_act_mul_and_quant_kernel<kUseBFloat16PrecisionMultiply, kUsePDL>,
      output_ptr, input_ptr, scale_ptr, num_per_expert_ptr, num_intermediate_size,
      num_tokens_per_expert, sparse_p, Persist2EP, Persist2PC);
}
}  // namespace masked_act_dispatch

void masked_act_mul_and_quant_async(__nv_fp8_e4m3 *output_ptr, const __nv_bfloat16 *input_ptr,
                                    const float *scale_ptr, const int *num_per_expert_ptr,
                                    int num_total_tokens, int num_intermediate_size,
                                    int num_tokens_per_expert, bool use_bf16_mul,
                                    int num_seq_per_group_avg, int num_seq_per_group_max,
                                    cudaStream_t stream) {
  using namespace masked_act_dispatch;  // NOLINT

  // Empty-input guard: avoids `dim3 grid(0)` and division by zero below.
  if (num_total_tokens <= 0 || num_tokens_per_expert <= 0 || num_seq_per_group_avg <= 0) {
    return;
  }

  const int num_experts = num_total_tokens / num_tokens_per_expert;
  const int num_sm = get_sm_count();
  const int block_threads = pick_block_threads(num_intermediate_size);
  const int col_tiles = div_round_up(num_intermediate_size, block_threads * 8);

  // Partition budget a work-saturating launch would use per expert; pick_sparse_p
  // caps P to this so the grid never overshoots.
  const int min_blocks_per_expert = div_round_up(num_sm * kSmLoadFactor, num_experts);
  int num_parallel_tokens = div_round_up(num_tokens_per_expert, kNumWarpPerBlock);
  if (num_parallel_tokens < min_blocks_per_expert) {
    num_parallel_tokens = min_blocks_per_expert;
  }

  const int sparse_p = pick_sparse_p(num_seq_per_group_avg, num_seq_per_group_max,
                                     num_tokens_per_expert, num_parallel_tokens, col_tiles);

  constexpr bool kUsePDL = true;
  cudaLaunchAttribute attribute[1];
  attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute[0].val.programmaticStreamSerializationAllowed = 1;

  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(num_experts * sparse_p * col_tiles);
  cfg.blockDim = dim3(block_threads);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cfg.attrs = attribute;
  cfg.numAttrs = 1;

  cutlass::FastDivmod Persist2EP(sparse_p);
  cutlass::FastDivmod Persist2PC(col_tiles);

  if (use_bf16_mul) {
    launch_masked_act<true, kUsePDL>(&cfg, output_ptr, input_ptr, scale_ptr, num_per_expert_ptr,
                                     num_intermediate_size, num_tokens_per_expert, sparse_p,
                                     Persist2EP, Persist2PC);
  } else {
    launch_masked_act<false, kUsePDL>(&cfg, output_ptr, input_ptr, scale_ptr, num_per_expert_ptr,
                                      num_intermediate_size, num_tokens_per_expert, sparse_p,
                                      Persist2EP, Persist2PC);
  }
}

void act_mul_and_blockwise_quant_async(void *output_ptr, void *output_scale_ptr,
                                       const void *input_ptr, const int num_row, const int num_col,
                                       bool use_pdl, cudaStream_t stream) {
  using Tin = __nv_bfloat16;
  using Tout = __nv_fp8_e4m3;
  int intermediate_size = num_col / 2;

  dim3 block(128);
  int num_block_per_row = (intermediate_size / 8 + block.x - 1) / block.x;
  cutlass::FastDivmod block1D22D(num_block_per_row);
  dim3 grid(num_row * num_block_per_row);

  if (use_pdl) {
    constexpr bool kUsePDL = true;

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = 1;

    // Set the attribute in a kernel launch configuration
    cudaLaunchConfig_t config{};

    // Base launch configuration
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 0;
    config.stream = stream;

    // Add special attribute for PDL
    config.attrs = attribute;
    config.numAttrs = 1;

    auto kernel = kernels::act_mul_and_blockwise_quant_kernel<kUsePDL>;

    cudaLaunchKernelEx(&config, kernel, (Tin *)input_ptr, (Tout *)output_ptr,
                       (float *)output_scale_ptr, num_row, intermediate_size, block1D22D);
  } else {
    constexpr bool kUsePDL = false;
    kernels::act_mul_and_blockwise_quant_kernel<kUsePDL><<<grid, block, 0, stream>>>(
        (Tin *)input_ptr, (Tout *)output_ptr, (float *)output_scale_ptr, num_row, intermediate_size,
        block1D22D);
  }
}

void act_mul_and_blockwise_quant_async(void *output_ptr, void *output_scale_ptr,
                                       const void *input_ptr,
                                       const void *cu_num_tokens_per_group_ptr,
                                       const void *cu_tiles_ptr, const int num_row,
                                       const int num_row_padded_size, const int num_col,
                                       const int num_group, const int num_tokens_per_group_avg,
                                       bool use_pdl, cudaStream_t stream) {
  using Tin = __nv_bfloat16;
  using Tout = __nv_fp8_e4m3;
  int intermediate_size = num_col / 2;

  dim3 block(128);
  int num_block_per_row = (intermediate_size / 8 + block.x - 1) / block.x;
  cutlass::FastDivmod block1D22D(num_block_per_row);
  dim3 grid(num_row * num_block_per_row);
  int ktile_m = 0;
  if (num_tokens_per_group_avg <= 8) {
    ktile_m = 8;
  } else if (num_tokens_per_group_avg <= 16) {
    ktile_m = 16;
  } else if (num_tokens_per_group_avg <= 32) {
    ktile_m = 32;
  } else if (num_tokens_per_group_avg <= 48) {
    ktile_m = 48;
  } else {
    ktile_m = 64;
  }

  if (use_pdl) {
    constexpr bool kUsePDL = true;

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = 1;

    // Set the attribute in a kernel launch configuration
    cudaLaunchConfig_t config{};

    // Base launch configuration
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 0;
    config.stream = stream;

    // Add special attribute for PDL
    config.attrs = attribute;
    config.numAttrs = 1;

    auto kernel = kernels::act_mul_and_blockwise_quant_fusemoe_kernel<kUsePDL>;

    cudaLaunchKernelEx(&config, kernel, (Tin *)input_ptr, (Tout *)output_ptr,
                       (float *)output_scale_ptr, (int *)cu_num_tokens_per_group_ptr,
                       (int *)cu_tiles_ptr, num_row, num_row_padded_size, intermediate_size,
                       num_group, ktile_m, block1D22D);
  } else {
    constexpr bool kUsePDL = false;
    kernels::act_mul_and_blockwise_quant_fusemoe_kernel<kUsePDL><<<grid, block, 0, stream>>>(
        (Tin *)input_ptr, (Tout *)output_ptr, (float *)output_scale_ptr,
        (int *)cu_num_tokens_per_group_ptr, (int *)cu_tiles_ptr, num_row, num_row_padded_size,
        intermediate_size, num_group, ktile_m, block1D22D);
  }
}

void masked_act_mul_and_blockwise_quant_async(__nv_fp8_e4m3 *output_ptr, float *output_scale_ptr,
                                              const __nv_bfloat16 *input_ptr,
                                              const int *num_per_expert_ptr, int num_total_tokens,
                                              int num_intermediate_size, int num_tokens_per_expert,
                                              cudaStream_t stream) {
  dim3 block(256);
  int num_block_col = (num_intermediate_size / 8 + block.x - 1) / block.x;

  int num_sm = get_sm_count();
  int num_block_hard = num_sm * 8;

  int num_block_row = num_block_hard / num_block_col;
  int num_block = num_block_row * num_block_col;
  dim3 grid(num_block);

  cutlass::FastDivmod Block2YX(num_block_col);
  cutlass::FastDivmod Row2EandT(num_tokens_per_expert);

  kernels::masked_act_mul_and_blockwise_quant_kernel<<<grid, block, 0, stream>>>(
      output_ptr, output_scale_ptr, input_ptr, num_per_expert_ptr, num_total_tokens,
      num_intermediate_size, num_intermediate_size / 128, num_tokens_per_expert, Block2YX,
      Row2EandT, num_block_row);
}

}  // namespace activation
}  // namespace hpc
