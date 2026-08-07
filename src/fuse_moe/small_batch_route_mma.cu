// Copyright (C) 2026 Tencent.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <type_traits>

#include "src/fuse_moe/fuse_moe.h"
#include "src/fuse_moe/small_batch_route_mma.h"
#include "src/group_gemm/cp_async/group_gemm.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace fuse_moe {
namespace kernels {

template <bool kUseBFloat16PrecisionMultiply>
__global__ void split_partial_act_quant_kernel(
    __nv_fp8_e4m3 *__restrict__ output_ptr,
    const __nv_bfloat16 *__restrict__ partial_ptr,
    const float *__restrict__ scale_ptr, int num_routes, int num_splits,
    int intermediate_size) {
  cudaGridDependencySynchronize();

  constexpr int kVectorSize = 8;
  const int num_vectors = num_routes * intermediate_size / kVectorSize;
  for (int vector_idx = blockIdx.x * blockDim.x + threadIdx.x;
       vector_idx < num_vectors; vector_idx += blockDim.x * gridDim.x) {
    const int element_idx = vector_idx * kVectorSize;
    const int route = element_idx / intermediate_size;
    const int col = element_idx - route * intermediate_size;
    float gate[kVectorSize] = {};
    float up[kVectorSize] = {};

#pragma unroll 1
    for (int split = 0; split < num_splits; ++split) {
      const uint64_t base =
          (static_cast<uint64_t>(route) * num_splits + split) *
          (2 * intermediate_size);
#pragma unroll
      for (int i = 0; i < kVectorSize; ++i) {
        gate[i] += __bfloat162float(partial_ptr[base + col + i]);
        up[i] += __bfloat162float(
            partial_ptr[base + intermediate_size + col + i]);
      }
    }

    float activated[kVectorSize];
#pragma unroll
    for (int i = 0; i < kVectorSize; ++i) {
      gate[i] = __bfloat162float(__float2bfloat16_rn(gate[i]));
      up[i] = __bfloat162float(__float2bfloat16_rn(up[i]));
      if constexpr (kUseBFloat16PrecisionMultiply) {
        auto silu_bf16 = __float2bfloat16_rn(silu(gate[i]));
        auto up_bf16 = __float2bfloat16_rn(up[i]);
        activated[i] = __bfloat162float(silu_bf16 * up_bf16) * scale_ptr[0];
      } else {
        activated[i] = silu(gate[i]) * up[i] * scale_ptr[0];
      }
    }
    auto *output = output_ptr + element_idx;
    *reinterpret_cast<__nv_fp8x4_e4m3 *>(output) =
        __nv_fp8x4_e4m3(*reinterpret_cast<float4 *>(&activated[0]));
    *reinterpret_cast<__nv_fp8x4_e4m3 *>(output + 4) =
        __nv_fp8x4_e4m3(*reinterpret_cast<float4 *>(&activated[4]));
  }

  // Some threads can have no vector to write. Let natural CTA completion
  // satisfy PDL instead of allowing an idle thread to trigger it early.
}

}  // namespace kernels

void split_partial_act_quant_async(
    void *output_ptr, const void *partial_ptr, const void *scale_ptr,
    int num_routes, int num_splits, int intermediate_size,
    bool use_bf16_mul, cudaStream_t stream) {
  constexpr int kThreads = 256;
  constexpr int kVectorSize = 8;
  const int num_vectors = num_routes * intermediate_size / kVectorSize;
  const int grid = (num_vectors + kThreads - 1) / kThreads;

  auto launch = [&](auto bf16_mul_tag) {
    constexpr bool kUseBFloat16PrecisionMultiply = decltype(bf16_mul_tag)::value;
    auto kernel =
        kernels::split_partial_act_quant_kernel<kUseBFloat16PrecisionMultiply>;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(grid);
    cfg.blockDim = dim3(kThreads);
    cfg.stream = stream;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(
        &cfg, kernel, static_cast<__nv_fp8_e4m3 *>(output_ptr),
        static_cast<const __nv_bfloat16 *>(partial_ptr),
        static_cast<const float *>(scale_ptr), num_routes, num_splits,
        intermediate_size);
  };
  if (use_bf16_mul) {
    launch(std::true_type{});
  } else {
    launch(std::false_type{});
  }
}

void fuse_moe_small_batch_route_mma_async(
    void *output_ptr, const void *input_ptr, void *intermediate_ptr,
    const void *gate_up_weight_ptr, const void *gate_up_scale_ptr,
    const void *act_and_mul_scale_ptr, const void *down_weight_ptr,
    const void *down_scale_ptr, const void *topk_ids_ptr, const void *topk_scale_ptr,
    const void *shared_output_ptr, int num_seq, int hidden_size, int intermediate_size,
    int num_topk, int num_splits, int num_expert_local, int rank_ep, bool use_bf16_mul,
    cudaStream_t stream) {
  const int num_routes = num_seq * num_topk;
  auto *workspace = static_cast<uint8_t *>(intermediate_ptr);
  auto *gate_up_partials = reinterpret_cast<__nv_bfloat16 *>(workspace);
  workspace += static_cast<size_t>(num_routes) * num_splits * 2 * intermediate_size *
               sizeof(__nv_bfloat16);
  auto *down_input = reinterpret_cast<__nv_fp8_e4m3 *>(workspace);
  workspace += static_cast<size_t>(num_routes) * intermediate_size *
               sizeof(__nv_fp8_e4m3);
  auto *down_output = reinterpret_cast<__nv_bfloat16 *>(workspace);

  group_gemm_cp_async::group_gemm_fp8_route_splitk_async(
      gate_up_partials, input_ptr, gate_up_weight_ptr, gate_up_scale_ptr,
      topk_ids_ptr, num_routes, num_topk, 2 * intermediate_size, hidden_size,
      num_splits, num_expert_local, rank_ep, stream);

  split_partial_act_quant_async(
      down_input, gate_up_partials, act_and_mul_scale_ptr, num_routes,
      num_splits, intermediate_size, use_bf16_mul, stream);

  group_gemm_cp_async::group_gemm_fp8_route_async(
      down_output, down_input, down_weight_ptr, down_scale_ptr, topk_ids_ptr,
      num_routes, num_topk, hidden_size, intermediate_size, num_expert_local,
      rank_ep, /*input_is_token=*/false, stream);

  reduce_async(output_ptr, down_output, /*topk_pos_ptr=*/nullptr, topk_scale_ptr,
               shared_output_ptr, num_routes, num_seq, hidden_size, num_topk,
               /*use_pdl=*/true, stream);
}

namespace kernels {

constexpr int kBlockwiseQuantSize = 128;

__global__ void blockwise_act_quant_kernel(
    __nv_fp8_e4m3 *__restrict__ output_ptr,
    float *__restrict__ output_scale_ptr,
    const __nv_bfloat16 *__restrict__ gate_up_ptr, int num_splits,
    int intermediate_size) {
  cudaGridDependencySynchronize();

  const int route = blockIdx.y;
  const int block = blockIdx.x;
  const int col = block * kBlockwiseQuantSize + threadIdx.x;
  float activated = 0.0f;
  if (col < intermediate_size) {
    float gate = 0.0f;
    float up = 0.0f;
#pragma unroll 1
    for (int split = 0; split < num_splits; ++split) {
      const uint64_t base =
          (static_cast<uint64_t>(route) * num_splits + split) *
          2 * intermediate_size;
      gate += __bfloat162float(gate_up_ptr[base + col]);
      up += __bfloat162float(
          gate_up_ptr[base + intermediate_size + col]);
    }
    gate = __bfloat162float(__float2bfloat16_rn(gate));
    up = __bfloat162float(__float2bfloat16_rn(up));
    activated = silu(gate) * up;
  }

  float max_value = warp_reduce_max_xor(fabsf(activated));
  __shared__ float warp_max[4];
  if ((threadIdx.x & 31) == 0) {
    warp_max[threadIdx.x / 32] = max_value;
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    max_value = threadIdx.x < 4 ? warp_max[threadIdx.x] : 0.0f;
    max_value = warp_reduce_max_xor(max_value);
  }

  __shared__ float inverse_scale;
  const int num_blocks =
      (intermediate_size + kBlockwiseQuantSize - 1) /
      kBlockwiseQuantSize;
  if (threadIdx.x == 0) {
    const float scale = max_value / 448.0f;
    output_scale_ptr[static_cast<uint64_t>(route) * num_blocks + block] =
        scale;
    inverse_scale = 1.0f / (scale + 1e-8f);
  }
  __syncthreads();
  if (col < intermediate_size) {
    output_ptr[static_cast<uint64_t>(route) * intermediate_size + col] =
        __nv_fp8_e4m3(activated * inverse_scale);
  }
}

}  // namespace kernels

void blockwise_act_quant_async(
    void *output_ptr, void *output_scale_ptr, const void *gate_up_ptr,
    int num_routes, int num_splits, int intermediate_size,
    cudaStream_t stream) {
  constexpr int kThreads = kernels::kBlockwiseQuantSize;
  const int num_blocks =
      (intermediate_size + kernels::kBlockwiseQuantSize - 1) /
      kernels::kBlockwiseQuantSize;
  auto kernel = kernels::blockwise_act_quant_kernel;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(num_blocks, num_routes);
  cfg.blockDim = dim3(kThreads);
  cfg.stream = stream;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  cudaLaunchKernelEx(
      &cfg, kernel, static_cast<__nv_fp8_e4m3 *>(output_ptr),
      static_cast<float *>(output_scale_ptr),
      static_cast<const __nv_bfloat16 *>(gate_up_ptr), num_splits,
      intermediate_size);
}

void fuse_moe_blockwise_small_batch_route_mma_async(
    void *output_ptr, const void *input_ptr, const void *input_scale_ptr,
    void *workspace_ptr, const void *gate_up_weight_ptr,
    const void *gate_up_weight_scale_ptr, const void *down_weight_ptr,
    const void *down_weight_scale_ptr, const void *topk_ids_ptr,
    const void *topk_scale_ptr, const void *shared_output_ptr, int num_tokens,
    int hidden_size, int intermediate_size, int num_topk, int num_splits,
    int num_expert_local, int gate_up_weight_scale_lastdim_pad4,
    int down_weight_scale_lastdim_pad4, int rank_ep, cudaStream_t stream) {
  const int num_routes = num_tokens * num_topk;
  const int num_intermediate_blocks =
      (intermediate_size + kernels::kBlockwiseQuantSize - 1) /
      kernels::kBlockwiseQuantSize;
  auto *workspace = static_cast<uint8_t *>(workspace_ptr);
  auto *gate_up_partials = reinterpret_cast<__nv_bfloat16 *>(workspace);
  workspace += static_cast<size_t>(num_routes) * num_splits *
               2 * intermediate_size *
               sizeof(__nv_bfloat16);
  auto *down_input = reinterpret_cast<__nv_fp8_e4m3 *>(workspace);
  workspace += static_cast<size_t>(num_routes) * intermediate_size;
  auto *down_input_scale = reinterpret_cast<float *>(workspace);
  workspace += static_cast<size_t>(num_routes) * num_intermediate_blocks *
               sizeof(float);
  auto *down_output = reinterpret_cast<__nv_bfloat16 *>(workspace);

  group_gemm_cp_async::group_gemm_fp8_route_blockwise_async(
      gate_up_partials, input_ptr, input_scale_ptr, gate_up_weight_ptr,
      gate_up_weight_scale_ptr, topk_ids_ptr, num_routes, num_topk,
      2 * intermediate_size, hidden_size, num_splits, num_expert_local, rank_ep,
      /*input_is_token=*/true, gate_up_weight_scale_lastdim_pad4, stream);

  blockwise_act_quant_async(down_input, down_input_scale, gate_up_partials,
                            num_routes, num_splits, intermediate_size, stream);

  group_gemm_cp_async::group_gemm_fp8_route_blockwise_async(
      down_output, down_input, down_input_scale, down_weight_ptr,
      down_weight_scale_ptr, topk_ids_ptr, num_routes, num_topk, hidden_size,
      intermediate_size, /*num_splits=*/1, num_expert_local, rank_ep,
      /*input_is_token=*/false, down_weight_scale_lastdim_pad4, stream);

  reduce_async(output_ptr, down_output, /*topk_pos_ptr=*/nullptr, topk_scale_ptr,
               shared_output_ptr, num_routes, num_tokens, hidden_size, num_topk,
               /*use_pdl=*/true, stream);
}

}  // namespace fuse_moe
}  // namespace hpc
