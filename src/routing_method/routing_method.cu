// Copyright 2025 hpc-ops authors

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>

#include "src/routing_method/routing_method.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace routing_method {

static constexpr int kWarpSize = 32;

namespace kernels {

namespace cg = cooperative_groups;

static constexpr uint32_t POS_MASK = ((uint32_t)(1)) << (sizeof(float) * 8 - 1);
static constexpr uint32_t NEG_MASK = ((uint32_t)(-1));
static constexpr int kMoveBits = 32;
static constexpr int kMaxIdx = 65535;  // must be larger than max index, currently max index is 255

__forceinline__ __device__ uint64_t pack(float score, int index) {
  uint32_t u32_score = __float_as_uint(score);
  uint32_t mask = (u32_score & POS_MASK) ? NEG_MASK : POS_MASK;
  // Use 65535 minus index to give higher priority to elements with smaller indices.
  return (static_cast<uint64_t>(u32_score ^ mask) << kMoveBits) |
         static_cast<uint64_t>(kMaxIdx - index);
}

__forceinline__ __device__ void unpack(uint64_t pack_value, float& score, int& index) {
  index = kMaxIdx - static_cast<int32_t>((pack_value & 0xFFFF));

  uint32_t u32_score = (uint32_t)(pack_value >> kMoveBits);
  uint32_t mask = (u32_score & POS_MASK) ? POS_MASK : NEG_MASK;
  score = __uint_as_float(u32_score ^ mask);
}

__forceinline__ __device__ void cmp_and_swap(uint64_t& x, uint64_t& y) {
  auto pair_max = max(x, y);
  auto pair_min = min(x, y);
  x = pair_max;
  y = pair_min;
}

template <int N>
__forceinline__ __device__ void naive_sort(vec_t<uint64_t, N>& arr);

template <int N = 8>
__forceinline__ __device__ void naive_sort(vec_t<uint64_t, N>& arr) {
  cmp_and_swap(arr[0], arr[1]);
  cmp_and_swap(arr[2], arr[3]);
  cmp_and_swap(arr[4], arr[5]);
  cmp_and_swap(arr[6], arr[7]);

  cmp_and_swap(arr[0], arr[2]);
  cmp_and_swap(arr[1], arr[3]);
  cmp_and_swap(arr[4], arr[6]);
  cmp_and_swap(arr[5], arr[7]);

  cmp_and_swap(arr[1], arr[2]);
  cmp_and_swap(arr[5], arr[6]);

  cmp_and_swap(arr[0], arr[4]);
  cmp_and_swap(arr[1], arr[5]);
  cmp_and_swap(arr[2], arr[6]);
  cmp_and_swap(arr[3], arr[7]);

  cmp_and_swap(arr[2], arr[4]);
  cmp_and_swap(arr[3], arr[5]);

  cmp_and_swap(arr[1], arr[2]);
  cmp_and_swap(arr[3], arr[4]);
  cmp_and_swap(arr[5], arr[6]);
}

template <int kTopK, int kExpertsPerThread>
__forceinline__ __device__ void reduceTopK(cg::thread_block_tile<kWarpSize> const& warp,
                                           vec_t<float, kTopK>& topk_scores,
                                           vec_t<int32_t, kTopK>& topk_indices,
                                           vec_t<float, kExpertsPerThread>& scores,
                                           vec_t<int32_t, kExpertsPerThread>& indices) {
  vec_t<uint64_t, kExpertsPerThread> pack_values;
#pragma unroll
  for (int e = 0; e < kExpertsPerThread; ++e) {
    pack_values[e] = pack(scores[e], indices[e]);
  }

  naive_sort<kExpertsPerThread>(pack_values);

  uint64_t pack_max{};
  int pos_id = 0;
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    pack_max = cg::reduce(warp, pack_values[pos_id], cg::greater<uint64_t>{});
    // Since kExpertsPerThread is greater than kTopK, there's no need to
    // consider the case where pos_id exceeds the boundary.
    if (pack_max == pack_values[pos_id]) {
      pos_id += 1;
    }
    unpack(pack_max, topk_scores[k], topk_indices[k]);
  }
}

template <int kNExpert, int kTopK, bool is_hash>
__global__ void deepseekv4_routing_method_kernel(float* __restrict__ out_weights_ptr,
                                                 int32_t* __restrict__ out_indices_ptr,
                                                 const __nv_bfloat16* __restrict__ score_ptr,
                                                 const float* __restrict__ bias_ptr,
                                                 const int32_t* __restrict__ input_ids_ptr,
                                                 const int32_t* __restrict__ tid2eid_ptr,
                                                 int batch_size, float route_scale) {
  // Compile-time constants
  constexpr int kExpertsPerThread = kNExpert / kWarpSize;
  constexpr int kWarpsPerBlock = 4;  // Adjust based on occupancy needs

  // Shared memory for original scores (one array per warp in the block)
  __shared__ float smem_score[kWarpsPerBlock][kNExpert];

  // One warp per batch element
  const int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / kWarpSize;
  const int local_warp_id = threadIdx.x / kWarpSize;
  const int lane_id = threadIdx.x % kWarpSize;

  if (global_warp_id >= batch_size) {
    return;
  }

  auto warp = cg::tiled_partition<kWarpSize>(cg::this_thread_block());

  // Pointer to this warp's shared memory and input scores
  float* my_smem_score = smem_score[local_warp_id];
  const __nv_bfloat16* score_row_ptr = score_ptr + global_warp_id * kNExpert;

  // Load scores, apply score function (softplus + sqrt), and store to shared memory
  vec_t<float, kExpertsPerThread> scores;
#pragma unroll
  for (int i = 0; i < kExpertsPerThread / 4; ++i) {
    vec_t<float, 4>& part_scores = *reinterpret_cast<vec_t<float, 4>*>(&scores[i * 4]);
    int col = i * kWarpSize * 4 + lane_id * 4;
    part_scores = to<float>(load<__nv_bfloat16, 4>(score_row_ptr + col));
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      // Using the approximate ptx instruction will cause an underflow,
      // which will result in a NaN being generated by dividing by zero.
      // part_scores[j] = sqrt_ftz(softplus(part_scores[j]));
      part_scores[j] = sqrtf(log1pf(expf(part_scores[j])));
    }
    store(my_smem_score + col, part_scores);
  }
  __syncwarp();  // Ensure all scores are written before reading

  // Output: each of first K lanes holds one value
  float my_topk_value = 0.0f;
  int my_topk_index = 0;

  if constexpr (is_hash) {
    // Hash mode: directly read from shared memory
    int token_id = input_ids_ptr[global_warp_id];
    if (lane_id < kTopK) {
      int expert_id = tid2eid_ptr[token_id * kTopK + lane_id];
      my_topk_index = expert_id;
      my_topk_value = my_smem_score[expert_id];  // Direct lookup from shared memory
    }
  } else {
    // non Hash mode: add bias to registers for topk
    vec_t<int32_t, kExpertsPerThread> indices;
#pragma unroll
    for (int i = 0; i < kExpertsPerThread / 4; ++i) {
      vec_t<float, 4>& part_scores = *reinterpret_cast<vec_t<float, 4>*>(&scores[i * 4]);
      vec_t<int32_t, 4>& part_indices = *reinterpret_cast<vec_t<int32_t, 4>*>(&indices[i * 4]);
      int col = i * kWarpSize * 4 + lane_id * 4;
      auto bias = load<float, 4>(bias_ptr + col);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        part_indices[j] = col + j;
        part_scores[j] += bias[j];  // Add bias for topk selection
      }
    }

    // Use reduceTopK to find top-k experts
    vec_t<float, kTopK> topk_scores;
    vec_t<int32_t, kTopK> topk_indices;
    reduceTopK<kTopK, kExpertsPerThread>(warp, topk_scores, topk_indices, scores, indices);

    // Gather original weights (without bias) from shared memory
    if (lane_id < kTopK) {
      int expert_id = topk_indices[lane_id];
      my_topk_index = expert_id;
      my_topk_value = my_smem_score[expert_id];  // Read original score (no bias)
    }
  }

  // Reduce to get sum (first K lanes have values, others have 0)
  float weight_sum = cg::reduce(warp, my_topk_value, cg::plus<float>{});

  // Normalize weights and write output (first K lanes)
  if (lane_id < kTopK) {
    out_weights_ptr[global_warp_id * kTopK + lane_id] = (my_topk_value / weight_sum) * route_scale;
    out_indices_ptr[global_warp_id * kTopK + lane_id] = my_topk_index;
  }
}

}  // namespace kernels

void deepseekv4_routing_method_async(float* out_weights_ptr, int32_t* out_indices_ptr,
                                     const __nv_bfloat16* score_ptr, const float* bias_ptr,
                                     const int32_t* input_ids_ptr, const int32_t* tid2eid_ptr,
                                     int batch_size, int num_expert, int topk, float route_scale,
                                     bool is_hash, cudaStream_t stream) {
  if (num_expert == 256 && topk == 6) {
    constexpr int kWarpPerBlock = 4;
    constexpr int kThreadPerBlock = kWarpPerBlock * kWarpSize;
    // one warp per row
    const int blocks = (batch_size + kWarpPerBlock - 1) / kWarpPerBlock;

    if (is_hash) {
      kernels::deepseekv4_routing_method_kernel<256, 6, true>
          <<<blocks, kThreadPerBlock, 0, stream>>>(out_weights_ptr, out_indices_ptr, score_ptr,
                                                   nullptr, input_ids_ptr, tid2eid_ptr, batch_size,
                                                   route_scale);
    } else {
      kernels::deepseekv4_routing_method_kernel<256, 6, false>
          <<<blocks, kThreadPerBlock, 0, stream>>>(out_weights_ptr, out_indices_ptr, score_ptr,
                                                   bias_ptr, nullptr, nullptr, batch_size,
                                                   route_scale);
    }
  }
}

}  // namespace routing_method
}  // namespace hpc
