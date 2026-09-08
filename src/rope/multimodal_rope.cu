// Copyright (C) 2026 Tencent.

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <stdint.h>

#include "src/rope/rope.h"

namespace hpc {
namespace rope {
namespace qk_rope_kernels {

constexpr int kWarpSize = 32;

template <int kHeadDim>
struct NeoxBf16x2Packs {
  static constexpr int kPacksPerHalf = kHeadDim / 4;
  static constexpr int kPacksPerLane = (kPacksPerHalf + kWarpSize - 1) / kWarpSize;
  float2 first[kPacksPerLane];
  float2 second[kPacksPerLane];
};

template <int kHeadDim>
struct NeoxCachePacks {
  static constexpr int kPacksPerHalf = kHeadDim / 4;
  static constexpr int kPacksPerLane = (kPacksPerHalf + kWarpSize - 1) / kWarpSize;
  float2 cos[kPacksPerLane];
  float2 sin[kPacksPerLane];
};

template <int kHeadDim>
__device__ __forceinline__ NeoxBf16x2Packs<kHeadDim> load_neox_bf16x2(const __nv_bfloat16 *input,
                                                                      int lane) {
  static_assert(kHeadDim % 4 == 0);
  constexpr int kHalfDim = kHeadDim / 2;
  constexpr int kPacksPerHalf = NeoxBf16x2Packs<kHeadDim>::kPacksPerHalf;
  constexpr int kPacksPerLane = NeoxBf16x2Packs<kHeadDim>::kPacksPerLane;
  NeoxBf16x2Packs<kHeadDim> values;
#pragma unroll
  for (int round = 0; round < kPacksPerLane; ++round) {
    const int pack = round * kWarpSize + lane;
    if (pack < kPacksPerHalf) {
      const int offset = pack * 2;
      values.first[round] =
          __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(input + offset));
      values.second[round] =
          __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(input + offset + kHalfDim));
    }
  }
  return values;
}

template <int kHeadDim>
__device__ __forceinline__ NeoxCachePacks<kHeadDim> load_neox_cache_bf16x2(const float *cache,
                                                                           int lane) {
  constexpr int kHalfDim = kHeadDim / 2;
  constexpr int kPacksPerHalf = NeoxCachePacks<kHeadDim>::kPacksPerHalf;
  constexpr int kPacksPerLane = NeoxCachePacks<kHeadDim>::kPacksPerLane;
  NeoxCachePacks<kHeadDim> values;
#pragma unroll
  for (int round = 0; round < kPacksPerLane; ++round) {
    const int pack = round * kWarpSize + lane;
    if (pack < kPacksPerHalf) {
      const int offset = pack * 2;
      values.cos[round] = *reinterpret_cast<const float2 *>(cache + offset);
      values.sin[round] = *reinterpret_cast<const float2 *>(cache + offset + kHalfDim);
    }
  }
  return values;
}

template <int kHeadDim>
__device__ __forceinline__ void rotate_store_neox_bf16x2(__nv_bfloat16 *output,
                                                         const NeoxBf16x2Packs<kHeadDim> &value,
                                                         const NeoxCachePacks<kHeadDim> &cache,
                                                         int lane) {
  constexpr int kHalfDim = kHeadDim / 2;
  constexpr int kPacksPerHalf = NeoxBf16x2Packs<kHeadDim>::kPacksPerHalf;
  constexpr int kPacksPerLane = NeoxBf16x2Packs<kHeadDim>::kPacksPerLane;
#pragma unroll
  for (int round = 0; round < kPacksPerLane; ++round) {
    const int pack = round * kWarpSize + lane;
    if (pack < kPacksPerHalf) {
      const int offset = pack * 2;
      const float2 first = make_float2(__fmaf_rn(-value.second[round].x, cache.sin[round].x,
                                                 value.first[round].x * cache.cos[round].x),
                                       __fmaf_rn(-value.second[round].y, cache.sin[round].y,
                                                 value.first[round].y * cache.cos[round].y));
      const float2 second = make_float2(__fmaf_rn(value.first[round].x, cache.sin[round].x,
                                                  value.second[round].x * cache.cos[round].x),
                                        __fmaf_rn(value.first[round].y, cache.sin[round].y,
                                                  value.second[round].y * cache.cos[round].y));
      *reinterpret_cast<__nv_bfloat162 *>(output + offset) = __float22bfloat162_rn(first);
      *reinterpret_cast<__nv_bfloat162 *>(output + offset + kHalfDim) =
          __float22bfloat162_rn(second);
    }
  }
}

template <int kNumQHeads, int kNumKvHeads, int kHeadDim, int kWarpsPerBlock,
          bool kUsePrefetch = false>
__global__ void qk_rope_gqa_token_kernel(__nv_bfloat16 *q_output, __nv_bfloat16 *k_output,
                                         const __nv_bfloat16 *q_input, const __nv_bfloat16 *k_input,
                                         const float *cos_sin_cache, const int64_t *positions,
                                         MultimodalRopeParams params) {
  static_assert(kNumQHeads % kNumKvHeads == 0);
  static_assert(kWarpsPerBlock == kNumKvHeads);
  static_assert(kHeadDim % 4 == 0);

  const int64_t batch = blockIdx.y;
  const int64_t sequence = blockIdx.x;
  const int kv_head = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  constexpr int kQHeadsPerKvHead = kNumQHeads / kNumKvHeads;

  if constexpr (kUsePrefetch) {
    static_assert(kQHeadsPerKvHead >= 16);
    const int64_t q_group_stride = static_cast<int64_t>(kNumKvHeads) * params.q_s1;
    const int64_t qo_group_stride = static_cast<int64_t>(kNumKvHeads) * params.qo_s1;
    const __nv_bfloat16 *q_row =
        q_input + batch * params.q_s0 + kv_head * params.q_s1 + sequence * params.q_s2;
    __nv_bfloat16 *q_out_row =
        q_output + batch * params.qo_s0 + kv_head * params.qo_s1 + sequence * params.qo_s2;
    NeoxBf16x2Packs<kHeadDim> current = load_neox_bf16x2<kHeadDim>(q_row, lane);

    const int64_t position = positions[batch * params.pos_s0 + sequence * params.pos_s1];
    const NeoxCachePacks<kHeadDim> cache_values =
        load_neox_cache_bf16x2<kHeadDim>(cos_sin_cache + position * params.cache_s0, lane);

#pragma unroll 1
    for (int q_group = 1; q_group < kQHeadsPerKvHead; ++q_group) {
      q_row += q_group_stride;
      const NeoxBf16x2Packs<kHeadDim> next = load_neox_bf16x2<kHeadDim>(q_row, lane);
      rotate_store_neox_bf16x2<kHeadDim>(q_out_row, current, cache_values, lane);
      q_out_row += qo_group_stride;
      current = next;
    }
    rotate_store_neox_bf16x2<kHeadDim>(q_out_row, current, cache_values, lane);

    const __nv_bfloat16 *k_row =
        k_input + batch * params.k_s0 + kv_head * params.k_s1 + sequence * params.k_s2;
    __nv_bfloat16 *k_out_row =
        k_output + batch * params.ko_s0 + kv_head * params.ko_s1 + sequence * params.ko_s2;
    const NeoxBf16x2Packs<kHeadDim> k_value = load_neox_bf16x2<kHeadDim>(k_row, lane);
    rotate_store_neox_bf16x2<kHeadDim>(k_out_row, k_value, cache_values, lane);
  } else {
    NeoxBf16x2Packs<kHeadDim> q_values[kQHeadsPerKvHead];
    __nv_bfloat16 *q_out_rows[kQHeadsPerKvHead];
#pragma unroll
    for (int q_group = 0; q_group < kQHeadsPerKvHead; ++q_group) {
      const int q_head = kv_head + q_group * kNumKvHeads;
      const __nv_bfloat16 *q_row =
          q_input + batch * params.q_s0 + q_head * params.q_s1 + sequence * params.q_s2;
      q_out_rows[q_group] =
          q_output + batch * params.qo_s0 + q_head * params.qo_s1 + sequence * params.qo_s2;
      q_values[q_group] = load_neox_bf16x2<kHeadDim>(q_row, lane);
    }

    const __nv_bfloat16 *k_row =
        k_input + batch * params.k_s0 + kv_head * params.k_s1 + sequence * params.k_s2;
    __nv_bfloat16 *k_out_row =
        k_output + batch * params.ko_s0 + kv_head * params.ko_s1 + sequence * params.ko_s2;
    const NeoxBf16x2Packs<kHeadDim> k_value = load_neox_bf16x2<kHeadDim>(k_row, lane);

    const int64_t position = positions[batch * params.pos_s0 + sequence * params.pos_s1];
    const NeoxCachePacks<kHeadDim> cache_values =
        load_neox_cache_bf16x2<kHeadDim>(cos_sin_cache + position * params.cache_s0, lane);

#pragma unroll
    for (int q_group = 0; q_group < kQHeadsPerKvHead; ++q_group) {
      rotate_store_neox_bf16x2<kHeadDim>(q_out_rows[q_group], q_values[q_group], cache_values,
                                         lane);
    }
    rotate_store_neox_bf16x2<kHeadDim>(k_out_row, k_value, cache_values, lane);
  }
}

template <int kNumQHeads, int kNumKvHeads, int kHeadDim, int kWarpsPerBlock>
__global__ void qk_rope_gqa_split_warp_token_kernel(
    __nv_bfloat16 *q_output, __nv_bfloat16 *k_output, const __nv_bfloat16 *q_input,
    const __nv_bfloat16 *k_input, const float *cos_sin_cache, const int64_t *positions,
    MultimodalRopeParams params) {
  static_assert(kNumQHeads % kNumKvHeads == 0);
  static_assert(kWarpsPerBlock % kNumKvHeads == 0);
  static_assert(kHeadDim % 4 == 0);
  constexpr int kWarpsPerKvHead = kWarpsPerBlock / kNumKvHeads;
  constexpr int kQHeadsPerKvHead = kNumQHeads / kNumKvHeads;
  static_assert(kQHeadsPerKvHead % kWarpsPerKvHead == 0);
  constexpr int kQHeadsPerWarp = kQHeadsPerKvHead / kWarpsPerKvHead;
  static_assert(kWarpsPerKvHead > 1);

  const int64_t batch = blockIdx.y;
  const int64_t sequence = blockIdx.x;
  const int warp = threadIdx.x / kWarpSize;
  const int kv_head = warp % kNumKvHeads;
  const int warp_in_group = warp / kNumKvHeads;
  const int lane = threadIdx.x % kWarpSize;

  NeoxBf16x2Packs<kHeadDim> q_values[kQHeadsPerWarp];
  __nv_bfloat16 *q_out_rows[kQHeadsPerWarp];
#pragma unroll
  for (int local_head = 0; local_head < kQHeadsPerWarp; ++local_head) {
    const int q_group = warp_in_group + local_head * kWarpsPerKvHead;
    const int q_head = kv_head + q_group * kNumKvHeads;
    const __nv_bfloat16 *q_row =
        q_input + batch * params.q_s0 + q_head * params.q_s1 + sequence * params.q_s2;
    q_out_rows[local_head] =
        q_output + batch * params.qo_s0 + q_head * params.qo_s1 + sequence * params.qo_s2;
    q_values[local_head] = load_neox_bf16x2<kHeadDim>(q_row, lane);
  }

  const bool writes_k = warp_in_group == 0;
  NeoxBf16x2Packs<kHeadDim> k_value;
  __nv_bfloat16 *k_out_row = nullptr;
  if (writes_k) {
    const __nv_bfloat16 *k_row =
        k_input + batch * params.k_s0 + kv_head * params.k_s1 + sequence * params.k_s2;
    k_out_row = k_output + batch * params.ko_s0 + kv_head * params.ko_s1 + sequence * params.ko_s2;
    k_value = load_neox_bf16x2<kHeadDim>(k_row, lane);
  }

  const int64_t position = positions[batch * params.pos_s0 + sequence * params.pos_s1];
  const NeoxCachePacks<kHeadDim> cache_values =
      load_neox_cache_bf16x2<kHeadDim>(cos_sin_cache + position * params.cache_s0, lane);

#pragma unroll
  for (int local_head = 0; local_head < kQHeadsPerWarp; ++local_head) {
    rotate_store_neox_bf16x2<kHeadDim>(q_out_rows[local_head], q_values[local_head], cache_values,
                                       lane);
  }
  if (writes_k) {
    rotate_store_neox_bf16x2<kHeadDim>(k_out_row, k_value, cache_values, lane);
  }
}

template <int kNumQHeads, int kNumKvHeads, int kHeadDim>
__global__ void qk_rope_gqa_neox_group_parallel_kernel(
    __nv_bfloat16 *q_output, __nv_bfloat16 *k_output, const __nv_bfloat16 *q_input,
    const __nv_bfloat16 *k_input, const float *cos_sin_cache, const int64_t *positions,
    MultimodalRopeParams params) {
  static_assert(kNumQHeads % kNumKvHeads == 0);
  static_assert(kHeadDim % (kWarpSize * 4) == 0);
  constexpr int kQHeadsPerKvHead = kNumQHeads / kNumKvHeads;
  constexpr int kHalfDim = kHeadDim / 2;
  constexpr int kPacksPerHalf = kHeadDim / 4;
  constexpr int kPacksPerTile = kWarpSize;
  constexpr int kTilesPerHead = kPacksPerHalf / kPacksPerTile;

  const int64_t sequence = blockIdx.x;
  const int64_t batch = blockIdx.y;
  const int kv_head = blockIdx.z / kTilesPerHead;
  const int tile = blockIdx.z % kTilesPerHead;
  const int head_in_group = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int pack = tile * kPacksPerTile + lane;
  const int offset = pack * 2;

  const __nv_bfloat16 *input_row;
  __nv_bfloat16 *output_row;
  if (head_in_group < kQHeadsPerKvHead) {
    const int q_head = kv_head + head_in_group * kNumKvHeads;
    input_row = q_input + batch * params.q_s0 + q_head * params.q_s1 + sequence * params.q_s2;
    output_row = q_output + batch * params.qo_s0 + q_head * params.qo_s1 + sequence * params.qo_s2;
  } else {
    input_row = k_input + batch * params.k_s0 + kv_head * params.k_s1 + sequence * params.k_s2;
    output_row = k_output + batch * params.ko_s0 + kv_head * params.ko_s1 + sequence * params.ko_s2;
  }
  const float2 first =
      __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(input_row + offset));
  const float2 second =
      __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(input_row + kHalfDim + offset));

  const int64_t position = positions[batch * params.pos_s0 + sequence * params.pos_s1];
  const float *cache = cos_sin_cache + position * params.cache_s0;
  const float2 cos = *reinterpret_cast<const float2 *>(cache + offset);
  const float2 sin = *reinterpret_cast<const float2 *>(cache + kHalfDim + offset);

  const float2 first_out = make_float2(__fmaf_rn(-second.x, sin.x, first.x * cos.x),
                                       __fmaf_rn(-second.y, sin.y, first.y * cos.y));
  const float2 second_out = make_float2(__fmaf_rn(first.x, sin.x, second.x * cos.x),
                                        __fmaf_rn(first.y, sin.y, second.y * cos.y));
  *reinterpret_cast<__nv_bfloat162 *>(output_row + offset) = __float22bfloat162_rn(first_out);
  *reinterpret_cast<__nv_bfloat162 *>(output_row + kHalfDim + offset) =
      __float22bfloat162_rn(second_out);
}

template <typename T, int kSize>
struct alignas(sizeof(T) * kSize) AlignedVector {
  T values[kSize];
};

template <int kNumQHeads, int kNumKvHeads, int kHeadDim, int kRowsPerBlock, int kThreadsPerRow,
          int kItemsPerThread>
__global__ __launch_bounds__(kRowsPerBlock *kThreadsPerRow) void qk_rope_interleaved_rows_kernel(
    __nv_bfloat16 *q_output, __nv_bfloat16 *k_output, const __nv_bfloat16 *q_input,
    const __nv_bfloat16 *k_input, const float *cos_sin_cache, const int64_t *positions,
    MultimodalRopeParams params) {
  static_assert(kHeadDim % 2 == 0);
  static_assert(kHeadDim == kThreadsPerRow * kItemsPerThread);
  static_assert(kItemsPerThread % 2 == 0);
  static_assert(kRowsPerBlock * kThreadsPerRow <= 1024);
  static_assert(kNumQHeads > 0 && kNumKvHeads > 0);
  constexpr int kHalfDim = kHeadDim / 2;
  constexpr int kPairsPerThread = kItemsPerThread / 2;
  using Bf16Vector = AlignedVector<__nv_bfloat162, kPairsPerThread>;
  using CacheVector = AlignedVector<float, kPairsPerThread>;

  const int row = threadIdx.y;
  const int lane = threadIdx.x;
  const int64_t batch = static_cast<int64_t>(blockIdx.z) * kRowsPerBlock + row;
  if (batch >= params.batch_size) {
    return;
  }
  const int64_t sequence = blockIdx.x;
  const int64_t position = positions[batch * params.pos_s0 + sequence * params.pos_s1];
  const float *cache = cos_sin_cache + position * params.cache_s0;

  const __nv_bfloat16 *input_row;
  __nv_bfloat16 *output_row;
  if (blockIdx.y < kNumQHeads) {
    const int q_head = blockIdx.y;
    input_row = q_input + batch * params.q_s0 + q_head * params.q_s1 + sequence * params.q_s2;
    output_row = q_output + batch * params.qo_s0 + q_head * params.qo_s1 + sequence * params.qo_s2;
  } else {
    const int kv_head = blockIdx.y - kNumQHeads;
    input_row = k_input + batch * params.k_s0 + kv_head * params.k_s1 + sequence * params.k_s2;
    output_row = k_output + batch * params.ko_s0 + kv_head * params.ko_s1 + sequence * params.ko_s2;
  }
  const int pair_offset = lane * kPairsPerThread;
  union ValueStorage {
    Bf16Vector vector;
    int4 words;
  } values;
  values.words = *reinterpret_cast<const int4 *>(input_row + lane * kItemsPerThread);
  const CacheVector cos_values = *reinterpret_cast<const CacheVector *>(cache + pair_offset);
  const CacheVector sin_values =
      *reinterpret_cast<const CacheVector *>(cache + kHalfDim + pair_offset);
#pragma unroll
  for (int item = 0; item < kPairsPerThread; ++item) {
    const float2 value = __bfloat1622float2(values.vector.values[item]);
    const float2 rotated =
        make_float2(__fmaf_rn(-value.y, sin_values.values[item], value.x * cos_values.values[item]),
                    __fmaf_rn(value.x, sin_values.values[item], value.y * cos_values.values[item]));
    values.vector.values[item] = __float22bfloat162_rn(rotated);
  }
  asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(output_row + lane * kItemsPerThread), "r"(values.words.x), "r"(values.words.y),
                 "r"(values.words.z), "r"(values.words.w));
}

}  // namespace qk_rope_kernels

namespace {

bool is_aligned_4(const void *pointer) {
  return (reinterpret_cast<uintptr_t>(pointer) & 0x3U) == 0;
}

bool is_aligned_16(const void *pointer) {
  return (reinterpret_cast<uintptr_t>(pointer) & 0xFU) == 0;
}

bool is_even(int64_t value) { return (value & 1) == 0; }

template <int kNumQHeads, int kNumKvHeads, int kHeadDim, bool kIsNeox>
bool can_use_qk_rope_profile(const __nv_bfloat16 *q_output, const __nv_bfloat16 *k_output,
                             const __nv_bfloat16 *q_input, const __nv_bfloat16 *k_input,
                             const float *cos_sin_cache, const MultimodalRopeParams &params) {
  return params.head_dim == kHeadDim && params.num_q_heads == kNumQHeads &&
         params.num_kv_heads == kNumKvHeads && params.is_neox == kIsNeox &&
         is_aligned_4(q_output) && is_aligned_4(k_output) && is_aligned_4(q_input) &&
         is_aligned_4(k_input) && is_aligned_16(cos_sin_cache) && is_even(params.q_s0) &&
         is_even(params.q_s1) && is_even(params.q_s2) && is_even(params.k_s0) &&
         is_even(params.k_s1) && is_even(params.k_s2) && is_even(params.qo_s0) &&
         is_even(params.qo_s1) && is_even(params.qo_s2) && is_even(params.ko_s0) &&
         is_even(params.ko_s1) && is_even(params.ko_s2) && params.cache_s0 % 4 == 0;
}

template <int kNumQHeads, int kNumKvHeads, int kHeadDim, bool kUsePrefetch = false>
void launch_qk_rope_gqa_neox(__nv_bfloat16 *q_output, __nv_bfloat16 *k_output,
                             const __nv_bfloat16 *q_input, const __nv_bfloat16 *k_input,
                             const float *cos_sin_cache, const int64_t *positions,
                             const MultimodalRopeParams &params, cudaStream_t stream) {
  static_assert(kNumQHeads % kNumKvHeads == 0);
  constexpr int kWarpsPerBlock = kNumKvHeads;
  dim3 grid(static_cast<uint32_t>(params.seq_len), static_cast<uint32_t>(params.batch_size));
  dim3 block(kWarpsPerBlock * qk_rope_kernels::kWarpSize);
  qk_rope_kernels::qk_rope_gqa_token_kernel<kNumQHeads, kNumKvHeads, kHeadDim, kWarpsPerBlock,
                                            kUsePrefetch><<<grid, block, 0, stream>>>(
      q_output, k_output, q_input, k_input, cos_sin_cache, positions, params);
}

template <int kNumQHeads, int kNumKvHeads, int kHeadDim, int kWarpsPerBlock>
void launch_qk_rope_gqa_neox_split_warps(__nv_bfloat16 *q_output, __nv_bfloat16 *k_output,
                                         const __nv_bfloat16 *q_input, const __nv_bfloat16 *k_input,
                                         const float *cos_sin_cache, const int64_t *positions,
                                         const MultimodalRopeParams &params, cudaStream_t stream) {
  static_assert(kNumQHeads % kNumKvHeads == 0);
  static_assert(kWarpsPerBlock % kNumKvHeads == 0);
  dim3 grid(static_cast<uint32_t>(params.seq_len), static_cast<uint32_t>(params.batch_size));
  dim3 block(kWarpsPerBlock * qk_rope_kernels::kWarpSize);
  qk_rope_kernels::qk_rope_gqa_split_warp_token_kernel<kNumQHeads, kNumKvHeads, kHeadDim,
                                                       kWarpsPerBlock><<<grid, block, 0, stream>>>(
      q_output, k_output, q_input, k_input, cos_sin_cache, positions, params);
}

template <int kNumQHeads, int kNumKvHeads, int kHeadDim>
void launch_qk_rope_gqa_neox_group_parallel(__nv_bfloat16 *q_output, __nv_bfloat16 *k_output,
                                            const __nv_bfloat16 *q_input,
                                            const __nv_bfloat16 *k_input,
                                            const float *cos_sin_cache, const int64_t *positions,
                                            const MultimodalRopeParams &params,
                                            cudaStream_t stream) {
  static_assert(kNumQHeads % kNumKvHeads == 0);
  static_assert(kHeadDim % (qk_rope_kernels::kWarpSize * 4) == 0);
  constexpr int kTilesPerHead = kHeadDim / (qk_rope_kernels::kWarpSize * 4);
  constexpr int kQHeadsPerKvHead = kNumQHeads / kNumKvHeads;
  constexpr int kHeadsPerGroup = kQHeadsPerKvHead + 1;
  dim3 grid(static_cast<uint32_t>(params.seq_len), static_cast<uint32_t>(params.batch_size),
            kNumKvHeads * kTilesPerHead);
  dim3 block(kHeadsPerGroup * qk_rope_kernels::kWarpSize);
  qk_rope_kernels::qk_rope_gqa_neox_group_parallel_kernel<kNumQHeads, kNumKvHeads, kHeadDim>
      <<<grid, block, 0, stream>>>(q_output, k_output, q_input, k_input, cos_sin_cache, positions,
                                   params);
}

template <int kNumQHeads, int kNumKvHeads, int kHeadDim, int kRowsPerBlock, int kItemsPerThread>
void launch_qk_rope_interleaved(__nv_bfloat16 *q_output, __nv_bfloat16 *k_output,
                                const __nv_bfloat16 *q_input, const __nv_bfloat16 *k_input,
                                const float *cos_sin_cache, const int64_t *positions,
                                const MultimodalRopeParams &params, cudaStream_t stream) {
  static_assert(kHeadDim % kItemsPerThread == 0);
  constexpr int kThreadsPerRow = kHeadDim / kItemsPerThread;
  const uint32_t row_blocks =
      static_cast<uint32_t>((params.batch_size + kRowsPerBlock - 1) / kRowsPerBlock);
  dim3 grid(static_cast<uint32_t>(params.seq_len), kNumQHeads + kNumKvHeads, row_blocks);
  dim3 block(kThreadsPerRow, kRowsPerBlock);
  qk_rope_kernels::qk_rope_interleaved_rows_kernel<kNumQHeads, kNumKvHeads, kHeadDim, kRowsPerBlock,
                                                   kThreadsPerRow, kItemsPerThread>
      <<<grid, block, 0, stream>>>(q_output, k_output, q_input, k_input, cos_sin_cache, positions,
                                   params);
}

template <int kNumQHeads, int kNumKvHeads, int kHeadDim>
bool try_launch_grouped_neox(__nv_bfloat16 *q_output, __nv_bfloat16 *k_output,
                             const __nv_bfloat16 *q_input, const __nv_bfloat16 *k_input,
                             const float *cos_sin_cache, const int64_t *positions,
                             const MultimodalRopeParams &params, cudaStream_t stream) {
  if (!can_use_qk_rope_profile<kNumQHeads, kNumKvHeads, kHeadDim, true>(
          q_output, k_output, q_input, k_input, cos_sin_cache, params)) {
    return false;
  }
  constexpr int64_t kGroupParallelMaxTokens = 32;
  const int64_t num_tokens = params.batch_size * params.seq_len;
  if (num_tokens <= kGroupParallelMaxTokens) {
    launch_qk_rope_gqa_neox_group_parallel<kNumQHeads, kNumKvHeads, kHeadDim>(
        q_output, k_output, q_input, k_input, cos_sin_cache, positions, params, stream);
  } else {
    launch_qk_rope_gqa_neox<kNumQHeads, kNumKvHeads, kHeadDim>(
        q_output, k_output, q_input, k_input, cos_sin_cache, positions, params, stream);
  }
  return true;
}

template <int kNumQHeads, int kNumKvHeads, int kHeadDim>
bool try_launch_parallel_neox(__nv_bfloat16 *q_output, __nv_bfloat16 *k_output,
                              const __nv_bfloat16 *q_input, const __nv_bfloat16 *k_input,
                              const float *cos_sin_cache, const int64_t *positions,
                              const MultimodalRopeParams &params, cudaStream_t stream) {
  if (!can_use_qk_rope_profile<kNumQHeads, kNumKvHeads, kHeadDim, true>(
          q_output, k_output, q_input, k_input, cos_sin_cache, params)) {
    return false;
  }
  launch_qk_rope_gqa_neox_group_parallel<kNumQHeads, kNumKvHeads, kHeadDim>(
      q_output, k_output, q_input, k_input, cos_sin_cache, positions, params, stream);
  return true;
}

}  // namespace

void multimodal_rope_async(__nv_bfloat16 *q_output, __nv_bfloat16 *k_output,
                           const __nv_bfloat16 *q_input, const __nv_bfloat16 *k_input,
                           const float *cos_sin_cache, const int64_t *positions,
                           const MultimodalRopeParams &params, cudaStream_t stream) {
  const int64_t num_tokens = params.batch_size * params.seq_len;
  constexpr int kNeoxHeadDim = 128;
  if (can_use_qk_rope_profile<16, 8, kNeoxHeadDim, true>(q_output, k_output, q_input, k_input,
                                                         cos_sin_cache, params)) {
    launch_qk_rope_gqa_neox<16, 8, kNeoxHeadDim>(q_output, k_output, q_input, k_input,
                                                 cos_sin_cache, positions, params, stream);
    return;
  }

  if (try_launch_grouped_neox<28, 4, kNeoxHeadDim>(q_output, k_output, q_input, k_input,
                                                   cos_sin_cache, positions, params, stream)) {
    return;
  }
  if (try_launch_grouped_neox<32, 8, kNeoxHeadDim>(q_output, k_output, q_input, k_input,
                                                   cos_sin_cache, positions, params, stream)) {
    return;
  }
  if (try_launch_grouped_neox<64, 8, kNeoxHeadDim>(q_output, k_output, q_input, k_input,
                                                   cos_sin_cache, positions, params, stream)) {
    return;
  }

  if (can_use_qk_rope_profile<64, 4, kNeoxHeadDim, true>(q_output, k_output, q_input, k_input,
                                                         cos_sin_cache, params)) {
    constexpr int64_t kGroupParallelMaxTokens = 32;
    constexpr int64_t kSplitWarpMaxTokens = 256;
    constexpr int64_t kPrefetchMaxTokens = 512;
    if (num_tokens <= kGroupParallelMaxTokens) {
      launch_qk_rope_gqa_neox_group_parallel<64, 4, kNeoxHeadDim>(
          q_output, k_output, q_input, k_input, cos_sin_cache, positions, params, stream);
    } else if (num_tokens <= kSplitWarpMaxTokens) {
      launch_qk_rope_gqa_neox_split_warps<64, 4, kNeoxHeadDim, 8>(
          q_output, k_output, q_input, k_input, cos_sin_cache, positions, params, stream);
    } else if (num_tokens <= kPrefetchMaxTokens) {
      launch_qk_rope_gqa_neox<64, 4, kNeoxHeadDim, true>(q_output, k_output, q_input, k_input,
                                                         cos_sin_cache, positions, params, stream);
    } else {
      launch_qk_rope_gqa_neox<64, 4, kNeoxHeadDim>(q_output, k_output, q_input, k_input,
                                                   cos_sin_cache, positions, params, stream);
    }
    return;
  }

  constexpr int kNeoxHeadDim512 = 512;
  if (try_launch_parallel_neox<16, 8, kNeoxHeadDim512>(q_output, k_output, q_input, k_input,
                                                       cos_sin_cache, positions, params, stream) ||
      try_launch_parallel_neox<28, 4, kNeoxHeadDim512>(q_output, k_output, q_input, k_input,
                                                       cos_sin_cache, positions, params, stream) ||
      try_launch_parallel_neox<32, 8, kNeoxHeadDim512>(q_output, k_output, q_input, k_input,
                                                       cos_sin_cache, positions, params, stream) ||
      try_launch_parallel_neox<64, 8, kNeoxHeadDim512>(q_output, k_output, q_input, k_input,
                                                       cos_sin_cache, positions, params, stream) ||
      try_launch_parallel_neox<64, 4, kNeoxHeadDim512>(q_output, k_output, q_input, k_input,
                                                       cos_sin_cache, positions, params, stream)) {
    return;
  }

  constexpr int kInterleavedHeadDim = 64;
  if (can_use_qk_rope_profile<64, 1, kInterleavedHeadDim, false>(q_output, k_output, q_input,
                                                                 k_input, cos_sin_cache, params)) {
    constexpr int kItemsPerThread = 8;
    if (params.batch_size <= 8) {
      constexpr int kRowsPerBlock = 8;
      launch_qk_rope_interleaved<64, 1, kInterleavedHeadDim, kRowsPerBlock, kItemsPerThread>(
          q_output, k_output, q_input, k_input, cos_sin_cache, positions, params, stream);
    } else {
      constexpr int kRowsPerBlock = 16;
      launch_qk_rope_interleaved<64, 1, kInterleavedHeadDim, kRowsPerBlock, kItemsPerThread>(
          q_output, k_output, q_input, k_input, cos_sin_cache, positions, params, stream);
    }
    return;
  }
  if (can_use_qk_rope_profile<32, 1, kInterleavedHeadDim, false>(q_output, k_output, q_input,
                                                                 k_input, cos_sin_cache, params)) {
    constexpr int kRowsPerBlock = 16;
    constexpr int kItemsPerThread = 8;
    launch_qk_rope_interleaved<32, 1, kInterleavedHeadDim, kRowsPerBlock, kItemsPerThread>(
        q_output, k_output, q_input, k_input, cos_sin_cache, positions, params, stream);
    return;
  }
}

}  // namespace rope
}  // namespace hpc
