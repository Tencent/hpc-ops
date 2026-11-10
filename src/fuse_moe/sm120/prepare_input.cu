// Copyright (C) 2026 Tencent.

#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cute/int_tuple.hpp>

#include "src/fuse_moe/sm120/common/launch.cuh"
#include "src/fuse_moe/sm120/common/quant.cuh"
#include "src/fuse_moe/sm120/prepare_input.cuh"
#include "src/fuse_moe/sm120/routing_utils.cuh"
#include "src/utils/utils.cuh"

namespace hpc::fuse_moe::sm120 {
namespace {

constexpr int kWarpsPerBlock = 4;
constexpr int kBf16PassColumns = kWarpSize * kBf16PerLane;
// A lane holds routes lane, lane + 32, ... of its token, one per group. Up to 32
// routes take one group, which keeps the scale scatter free of the group loop.
constexpr int kMaxRouteGroups = kMaxTopK / kWarpSize;

// Packed scale word i of a token: four UE8M0 bytes, or one power-of-two float32
// scale per 128 columns repeated over its four 32-column slots.
template <bool kBlock128Scale>
__device__ __forceinline__ uint32_t token_scale_word(const void *input_scale, int64_t i) {
  if constexpr (kBlock128Scale) {
    return static_cast<uint32_t>(to_ue8m0_storage(static_cast<const float *>(input_scale)[i])) *
           0x01010101u;
  } else {
    return static_cast<const uint32_t *>(input_scale)[i];
  }
}

// scale_rows[g] of lane j holds the packed scale row of the token's route
// g * 32 + j (-1 when the route is not local). The lanes of an MX block take turns
// storing its scale for each route. Must be called by the whole warp.
template <int kGroups>
__device__ __forceinline__ void scatter_packed_scale_token_routes(
    uint8_t *packed_scale_bytes, int scale_m_padded, const int (&scale_rows)[kGroups], int num_rows,
    int column, uint8_t storage, bool active, int lane) {
#pragma unroll
  for (int g = 0; g < kGroups; ++g) {
    int group_rows = min(num_rows - g * kWarpSize, kWarpSize);
    if (group_rows <= 0) {
      break;
    }
#pragma unroll 1
    for (int first = 0; first < group_rows; first += kLanesPerMxBlock) {
      int j = first + lane % kLanesPerMxBlock;
      int target = __shfl_sync(0xffffffff, scale_rows[g], j % kWarpSize);
      if (active && j < group_rows && target >= 0) {
        store_packed_scale(packed_scale_bytes, scale_m_padded, target, column, storage);
      }
    }
  }
}

// Returns the padded GEMM row of a local route. With kFinalizeRouteRows, topk_pos holds the
// route's position within its expert on entry; this adds cu_padded_tokens_per_group[expert],
// stores the final padded row in topk_pos[route], and records row_to_route[row] = route.
// Without it, routing has already stored both mappings; only topk_pos[route] is read here.
template <bool kFinalizeRouteRows>
__device__ __forceinline__ int resolve_route_row(int route, int expert,
                                                 const int *cu_padded_tokens_per_group,
                                                 int *topk_pos, int *row_to_route) {
  if constexpr (kFinalizeRouteRows) {
    int row = cu_padded_tokens_per_group[expert] + topk_pos[route];
    topk_pos[route] = row;
    row_to_route[row] = route;
    return row;
  }
  return topk_pos[route];
}

// Quantizes one BF16 token row to MXFP8 and scatters each block's scale to the
// token's routes. A lane takes 8 columns, four lanes one MX block; the warps of a
// token split its columns as passes first_pass, first_pass + pass_stride, ...
template <int kGroups>
__device__ __forceinline__ void quantize_bf16_token_packed(
    const int *input_words, int64_t source_base, int *output_words, int64_t destination_base,
    uint8_t *packed_scale_bytes, const int (&scale_rows)[kGroups], int num_rows, int hidden_size,
    int scale_m_padded, int lane, int first_pass, int pass_stride) {
  for (int base = first_pass * kBf16PassColumns; base < hidden_size;
       base += pass_stride * kBf16PassColumns) {
    int column = base + lane * kBf16PerLane;
    hpc::vec_t<__nv_bfloat162, kIntsPerVector> raw = {};
    if (column < hidden_size) {
      raw = hpc::load<__nv_bfloat162, kIntsPerVector>(input_words + source_base + column / 2);
    }
    auto values = hpc::to<float>(raw);
    float block_max = 0.0f;
#pragma unroll
    for (int item = 0; item < kBf16PerLane; ++item) {
      block_max = fmaxf(block_max, fabsf(values[item]));
    }
    block_max = hpc::warp_4lane_reduce_max_xor(block_max);
    uint8_t storage = 0;
    float inv_scale = 0.0f;
    quantization_scale(block_max, storage, inv_scale);
    if (column < hidden_size) {
      int packed[2];
#pragma unroll
      for (int half = 0; half < kBf16PerLane / 4; ++half) {
        packed[half] =
            pack_e4m3_x4(values[4 * half] * inv_scale, values[4 * half + 1] * inv_scale,
                         values[4 * half + 2] * inv_scale, values[4 * half + 3] * inv_scale);
      }
      int *destination = output_words + destination_base + column / 4;
      *reinterpret_cast<int2 *>(destination) = *reinterpret_cast<const int2 *>(packed);
    }
    scatter_packed_scale_token_routes(packed_scale_bytes, scale_m_padded, scale_rows, num_rows,
                                      column, storage, column < hidden_size, lane);
  }
}

// scale_rows follows scatter_packed_scale_token_routes. Must be called by the whole warp.
// A packed word holds four consecutive k-block scales in byte order, so the token's
// scales copy over a word at a time; input_scale must be 4-byte aligned.
template <bool kBlock128Scale, int kGroups>
__device__ __forceinline__ void scatter_token_packed_scales(
    const void *input_scale, int token, uint8_t *packed_scale_bytes,
    const int (&scale_rows)[kGroups], int num_rows, int hidden_size, int scale_m_padded, int lane,
    int first_pass = 0, int pass_stride = 1) {
  int num_packs = hidden_size / (4 * kScaleVectorSize);
  int64_t token_base = static_cast<int64_t>(token) * num_packs;
  uint32_t *packed_scale = reinterpret_cast<uint32_t *>(packed_scale_bytes);
  int num_items = num_rows * num_packs;
#pragma unroll 1
  for (int first = first_pass * kWarpSize; first < num_items; first += pass_stride * kWarpSize) {
    int item = first + lane;
    int j = min(item / num_packs, num_rows - 1);
    int target = -1;
#pragma unroll
    for (int g = 0; g < kGroups; ++g) {
      if (g * kWarpSize < num_rows) {
        int group_target = __shfl_sync(0xffffffff, scale_rows[g], j % kWarpSize);
        target = j / kWarpSize == g ? group_target : target;
      }
    }
    if (item < num_items && target >= 0) {
      int pack = item - j * num_packs;
      packed_scale[target + static_cast<int64_t>(pack) * scale_m_padded] =
          token_scale_word<kBlock128Scale>(input_scale, token_base + pack);
    }
  }
}

// Returns the packed scale row of the lane's route, or -1 when the lane holds no local route.
template <bool kFinalizeRouteRows>
__device__ __forceinline__ int map_token_route(const int *topk_ids,
                                               const int *cu_padded_tokens_per_group, int *topk_pos,
                                               int *row_to_route, int token, int active_topk,
                                               int first_expert, int last_expert, int lane) {
  if (lane >= active_topk) {
    return -1;
  }
  int route = token * active_topk + lane;
  int global_expert = topk_ids[route];
  if (global_expert < first_expert || global_expert >= last_expert) {
    return -1;
  }
  int expert = global_expert - first_expert;
  int row = resolve_route_row<kFinalizeRouteRows>(route, expert, cu_padded_tokens_per_group,
                                                  topk_pos, row_to_route);
  return row >= 0 ? packed_scale_row(row, cu_padded_tokens_per_group[expert], expert) : -1;
}

template <bool kFinalizeRouteRows, int kGroups>
__device__ __forceinline__ void map_token_routes(const int *topk_ids,
                                                 const int *cu_padded_tokens_per_group,
                                                 int *topk_pos, int *row_to_route, int token,
                                                 int active_topk, int first_expert, int last_expert,
                                                 int lane, int (&scale_rows)[kGroups]) {
#pragma unroll
  for (int g = 0; g < kGroups; ++g) {
    scale_rows[g] = g * kWarpSize < active_topk
                        ? map_token_route<kFinalizeRouteRows>(
                              topk_ids, cu_padded_tokens_per_group, topk_pos, row_to_route, token,
                              active_topk, first_expert, last_expert, g * kWarpSize + lane)
                        : -1;
  }
}

template <bool kFinalizeRouteRows, int kGroups>
__global__ void quantize_tokens_and_scatter_scales_kernel(
    const int *input_words, const int *topk_ids, const int *cu_padded_tokens_per_group,
    int *topk_pos, int *row_to_route, int *quantized_words, uint8_t *packed_scale_bytes,
    int num_tokens, int num_topk, int hidden_size, int first_expert, int last_expert,
    int scale_m_padded, int parts) {
  cudaGridDependencySynchronize();

  int tidx = threadIdx.x;
  int warp = tidx / kWarpSize;
  int lane = tidx % kWarpSize;
  int item = blockIdx.x * kWarpsPerBlock + warp;
  int token = item / parts;
  int part = item % parts;
  int scale_rows[kGroups];

  if constexpr (kFinalizeRouteRows) {
    // topk_pos turns from position into row in place, so only one warp may map
    // the routes; it maps them for the whole block and the warps split the
    // token's columns.
    __shared__ int shared_scale_rows[kGroups][kWarpSize];
    token = blockIdx.x;
    part = warp;
    if (warp == 0) {
      map_token_routes<true>(topk_ids, cu_padded_tokens_per_group, topk_pos, row_to_route, token,
                             num_topk, first_expert, last_expert, lane, scale_rows);
#pragma unroll
      for (int g = 0; g < kGroups; ++g) {
        shared_scale_rows[g][lane] = scale_rows[g];
      }
    }
    __syncthreads();
#pragma unroll
    for (int g = 0; g < kGroups; ++g) {
      scale_rows[g] = shared_scale_rows[g][lane];
    }
  } else {
    map_token_routes<false>(topk_ids, cu_padded_tokens_per_group, topk_pos, row_to_route, token,
                            token < num_tokens ? num_topk : 0, first_expert, last_expert, lane,
                            scale_rows);
  }
  if (token < num_tokens) {
    quantize_bf16_token_packed(input_words, static_cast<int64_t>(token) * (hidden_size / 2),
                               quantized_words, static_cast<int64_t>(token) * (hidden_size / 4),
                               packed_scale_bytes, scale_rows, num_topk, hidden_size,
                               scale_m_padded, lane, part, parts);
  }

  cudaTriggerProgrammaticLaunchCompletion();
}

enum class TokenInput { kBf16, kMxfp8, kFp8Block128 };

template <TokenInput kInput, int kGroups>
__global__ void prepare_single_token_input_kernel(
    const int *input_words, const void *input_scale, const int *topk_ids, int *topk_pos,
    int *row_to_route, int *num_tokens_per_expert, int *cu_padded_tokens_per_group, int *cu_m_tiles,
    int *quantized_words, uint8_t *packed_scale_bytes, int num_topk, int hidden_size,
    int num_experts, int first_expert, int tile_m, int max_padded_rows, int scale_m_padded,
    const int4 *init_src, int4 *init_dst, int init_vectors) {
  cudaGridDependencySynchronize();

  for (int i = threadIdx.x; i < init_vectors; i += blockDim.x) {
    init_dst[i] = init_src != nullptr ? init_src[i] : make_int4(0, 0, 0, 0);
  }
  map_routes_in_block<true>(topk_ids, topk_pos, row_to_route, num_tokens_per_expert,
                            cu_padded_tokens_per_group, cu_m_tiles, num_topk, num_experts,
                            first_expert, tile_m, max_padded_rows);
  __syncthreads();

  int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
  int part = static_cast<int>(threadIdx.x) / kWarpSize;
  int scale_rows[kGroups];
  map_token_routes<false>(topk_ids, cu_padded_tokens_per_group, topk_pos, row_to_route, 0, num_topk,
                          first_expert, first_expert + num_experts, lane, scale_rows);
  if constexpr (kInput != TokenInput::kBf16) {
    scatter_token_packed_scales<kInput == TokenInput::kFp8Block128>(
        input_scale, 0, packed_scale_bytes, scale_rows, num_topk, hidden_size, scale_m_padded, lane,
        part, kRouteThreads / kWarpSize);
  } else {
    quantize_bf16_token_packed(input_words, 0, quantized_words, 0, packed_scale_bytes, scale_rows,
                               num_topk, hidden_size, scale_m_padded, lane, part,
                               kRouteThreads / kWarpSize);
  }
  cudaTriggerProgrammaticLaunchCompletion();
}

template <bool kFinalizeRouteRows, bool kBlock128Scale, int kGroups>
__global__ void scatter_token_scales_kernel(const void *input_scale, const int *topk_ids,
                                            const int *cu_padded_tokens_per_group, int *topk_pos,
                                            int *row_to_route, uint8_t *packed_scale_bytes,
                                            int num_tokens, int num_topk, int hidden_size,
                                            int first_expert, int last_expert, int scale_m_padded) {
  cudaGridDependencySynchronize();

  int tidx = threadIdx.x;
  int warp = tidx / kWarpSize;
  int lane = tidx % kWarpSize;
  int token = blockIdx.x * kWarpsPerBlock + warp;

  if (token < num_tokens) {
    int scale_rows[kGroups];
    map_token_routes<kFinalizeRouteRows>(topk_ids, cu_padded_tokens_per_group, topk_pos,
                                         row_to_route, token, num_topk, first_expert, last_expert,
                                         lane, scale_rows);
    scatter_token_packed_scales<kBlock128Scale>(input_scale, token, packed_scale_bytes, scale_rows,
                                                num_topk, hidden_size, scale_m_padded, lane);
  }

  cudaTriggerProgrammaticLaunchCompletion();
}

int token_warp_parts(bool finalize_route_rows, int num_topk, int hidden_size) {
  if (num_topk <= 0) {
    return 1;
  }
  // Finalizing route rows takes a block per token, one part per warp.
  if (finalize_route_rows) {
    return kWarpsPerBlock;
  }
  return cute::ceil_div(hidden_size, kBf16PassColumns);
}

template <bool kFinalizeRouteRows, int kGroups>
void launch_quantize_tokens_and_scatter_scales(const int *input_words, const int *topk_ids,
                                               const int *cu_padded_tokens_per_group, int *topk_pos,
                                               int *row_to_route, int *quantized_words,
                                               uint8_t *packed_scale_bytes, int num_tokens,
                                               int num_topk, int hidden_size, int first_expert,
                                               int last_expert, int scale_m_padded,
                                               cudaStream_t stream) {
  int parts = token_warp_parts(kFinalizeRouteRows, num_topk, hidden_size);
  int blocks = kFinalizeRouteRows ? num_tokens : cute::ceil_div(num_tokens * parts, kWarpsPerBlock);
  launch_pdl(quantize_tokens_and_scatter_scales_kernel<kFinalizeRouteRows, kGroups>, blocks,
             kWarpsPerBlock * kWarpSize, stream, input_words, topk_ids, cu_padded_tokens_per_group,
             topk_pos, row_to_route, quantized_words, packed_scale_bytes, num_tokens, num_topk,
             hidden_size, first_expert, last_expert, scale_m_padded, parts);
}

template <bool kFinalizeRouteRows, bool kBlock128Scale, int kGroups>
void launch_scatter_token_scales(const void *input_scale, const int *topk_ids,
                                 const int *cu_padded_tokens_per_group, int *topk_pos,
                                 int *row_to_route, uint8_t *packed_scale_bytes, int num_tokens,
                                 int num_topk, int hidden_size, int first_expert, int last_expert,
                                 int scale_m_padded, cudaStream_t stream) {
  int blocks = cute::ceil_div(num_tokens, kWarpsPerBlock);
  launch_pdl(scatter_token_scales_kernel<kFinalizeRouteRows, kBlock128Scale, kGroups>, blocks,
             kWarpsPerBlock * kWarpSize, stream, input_scale, topk_ids, cu_padded_tokens_per_group,
             topk_pos, row_to_route, packed_scale_bytes, num_tokens, num_topk, hidden_size,
             first_expert, last_expert, scale_m_padded);
}

// Calls fn.template operator()<kGroups>() with the route groups num_topk needs.
template <typename Fn>
void dispatch_route_groups(int num_topk, Fn &&fn) {
  num_topk <= kWarpSize ? fn.template operator()<1>() : fn.template operator()<kMaxRouteGroups>();
}

}  // namespace

void quantize_tokens_and_scatter_scales_async(const void *bf16_input, const int *topk_ids,
                                              const int *cu_padded_tokens_per_group, int *topk_pos,
                                              int *row_to_route, void *quantized_tokens,
                                              int32_t *packed_scale, int num_tokens, int num_topk,
                                              int hidden_size, int first_expert, int last_expert,
                                              int scale_m_padded, bool finalize_route_rows,
                                              cudaStream_t stream) {
  const int *input_words = static_cast<const int *>(bf16_input);
  int *quantized_words = static_cast<int *>(quantized_tokens);
  uint8_t *scale_bytes = reinterpret_cast<uint8_t *>(packed_scale);
  dispatch_route_groups(num_topk, [&]<int Groups>() {
    auto launch = [&]<bool FinalizeRouteRows>() {
      launch_quantize_tokens_and_scatter_scales<FinalizeRouteRows, Groups>(
          input_words, topk_ids, cu_padded_tokens_per_group, topk_pos, row_to_route,
          quantized_words, scale_bytes, num_tokens, num_topk, hidden_size, first_expert,
          last_expert, scale_m_padded, stream);
    };  // NOLINT
    finalize_route_rows ? launch.template operator()<true>() : launch.template operator()<false>();
  });
}

void prepare_single_token_input_async(
    const void *input, const void *input_scale, bool block128_scale, const int *topk_ids,
    int *topk_pos, int *row_to_route, int *num_tokens_per_expert, int *cu_padded_tokens_per_group,
    int *cu_m_tiles, void *quantized_tokens, int32_t *packed_scale, int num_topk, int hidden_size,
    int num_experts, int first_expert, int tile_m, int max_padded_rows, int scale_m_padded,
    const void *init_src, void *init_dst, int64_t init_bytes, cudaStream_t stream) {
  const int *input_words = static_cast<const int *>(input);
  int *quantized_words = static_cast<int *>(quantized_tokens);
  uint8_t *scale_bytes = reinterpret_cast<uint8_t *>(packed_scale);
  int init_vectors =
      init_dst != nullptr ? static_cast<int>(init_bytes / static_cast<int64_t>(sizeof(int4))) : 0;
  auto launch = [&]<TokenInput Input, int Groups>() {
    launch_pdl(prepare_single_token_input_kernel<Input, Groups>, 1, kRouteThreads, stream,
               input_words, input_scale, topk_ids, topk_pos, row_to_route, num_tokens_per_expert,
               cu_padded_tokens_per_group, cu_m_tiles, quantized_words, scale_bytes, num_topk,
               hidden_size, num_experts, first_expert, tile_m, max_padded_rows, scale_m_padded,
               static_cast<const int4 *>(init_src), static_cast<int4 *>(init_dst), init_vectors);
  };  // NOLINT
  dispatch_route_groups(num_topk, [&]<int Groups>() {
    if (input_scale == nullptr) {
      launch.template operator()<TokenInput::kBf16, Groups>();
    } else if (block128_scale) {
      launch.template operator()<TokenInput::kFp8Block128, Groups>();
    } else {
      launch.template operator()<TokenInput::kMxfp8, Groups>();
    }
  });
}

void scatter_token_scales_async(const void *input_scale, bool block128_scale, const int *topk_ids,
                                const int *cu_padded_tokens_per_group, int *topk_pos,
                                int *row_to_route, int32_t *packed_scale, int num_tokens,
                                int num_topk, int hidden_size, int first_expert, int last_expert,
                                int scale_m_padded, bool finalize_route_rows, cudaStream_t stream) {
  uint8_t *scale_bytes = reinterpret_cast<uint8_t *>(packed_scale);
  dispatch_route_groups(num_topk, [&]<int Groups>() {
    auto launch = [&]<bool FinalizeRouteRows, bool Block128Scale>() {
      launch_scatter_token_scales<FinalizeRouteRows, Block128Scale, Groups>(
          input_scale, topk_ids, cu_padded_tokens_per_group, topk_pos, row_to_route, scale_bytes,
          num_tokens, num_topk, hidden_size, first_expert, last_expert, scale_m_padded, stream);
    };  // NOLINT
    auto by_finalize_route_rows = [&]<bool Block128Scale>() {
      finalize_route_rows ? launch.template operator()<true, Block128Scale>()
                          : launch.template operator()<false, Block128Scale>();
    };  // NOLINT
    block128_scale ? by_finalize_route_rows.template operator()<true>()
                   : by_finalize_route_rows.template operator()<false>();
  });
}

}  // namespace hpc::fuse_moe::sm120
