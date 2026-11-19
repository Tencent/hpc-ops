// Copyright (C) 2026 Tencent.

#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/int_tuple.hpp>

#include "src/fuse_moe/fuse_moe.h"
#include "src/fuse_moe/sm120/common/utils.h"
#include "src/fuse_moe/sm120/down_gemm.cuh"
#include "src/fuse_moe/sm120/gate_up_gemm.cuh"
#include "src/fuse_moe/sm120/prepare_input.cuh"
#include "src/fuse_moe/sm120/routing.cuh"
#include "src/utils/utils.h"

namespace hpc {
namespace fuse_moe {

namespace {

constexpr int kGateDeepPipelineMinRowBytes = 1024;
// Expected rows per live expert from which tile_m = 64, by whether the live
// weights overflow L2.
constexpr double kTileM64RowsStreamed = 28.0;
constexpr double kTileM64RowsResident = 96.0;

// Measured on six MoE shapes: zeroing inside the gate-up GEMM gains 0.5-1.8% from
// 16k tokens on four and ties on two, but costs up to 1.5% at 512-1k tokens.
constexpr int kZeroInGateUpGemmMinTokens = 16384;
constexpr uint64_t kEvictNormal = static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL);
constexpr uint64_t kEvictFirst = static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_FIRST);
constexpr uint64_t kEvictLast = static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_LAST);

int l2_cache_bytes() {
  static const int bytes = [] {
    int device = 0;
    int value = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&value, cudaDevAttrL2CacheSize, device));
    return value;
  }();
  return bytes;
}

// Experts that receive at least one of the routes under uniformly random routing.
double expected_live_experts(int routes, int num_experts) {
  return num_experts * (1.0 - std::pow(1.0 - 1.0 / num_experts, routes));
}

// Expected M tiles for uniformly random routing: one partial tile per live expert.
int expected_m_tiles(int total_routes, int num_experts, int tile_m) {
  const double live = expected_live_experts(total_routes, num_experts);
  return std::max(1,
                  static_cast<int>(std::lround(live + static_cast<double>(total_routes) / tile_m)));
}

int choose_tile_n(int total_routes, int num_experts, int tile_m, int n, int num_sms) {
  const int m_tiles = expected_m_tiles(total_routes, num_experts, tile_m);
  auto occupancy = [&](int tile_n) {
    const int tiles = m_tiles * (n / tile_n);
    const int waves = cute::ceil_div(tiles, num_sms);
    return static_cast<double>(tiles) / (waves * num_sms);
  };
  return occupancy(64) > occupancy(128) * 1.05 ? 64 : 128;
}

// down GEMM's K is short, so a tile's fixed cost (its store and one more pass
// over A per extra N slab) outweighs wave fill once 128-wide tiles alone
// cover every SM.
int choose_down_tile_n(int total_routes, int num_experts, int tile_m, int n, int num_sms) {
  if (expected_m_tiles(total_routes, num_experts, tile_m) * (n / 128) >= num_sms) {
    return 128;
  }
  return choose_tile_n(total_routes, num_experts, tile_m, n, num_sms);
}

}  // namespace

FuseMoeMxPlan fuse_moe_mx_plan(int num_tokens, int num_topk, int num_expert_local,
                               int num_expert_total, int hidden_size, int down_input_size,
                               bool mxfp4_weight, bool has_shared_output) {
  TORCH_CHECK(num_expert_local >= 1 && num_expert_local <= sm120::kMaxExperts,
              "num_expert_local must be in [1, ", sm120::kMaxExperts, "]");
  TORCH_CHECK(num_topk <= sm120::kMaxTopK, "num_topk must be at most ", sm120::kMaxTopK);
  const int num_experts = num_expert_local;
  const int num_sms = get_sm_count();
  const int total_routes = num_tokens * num_topk;
  // Under EP only this rank's share of the routes lands on its experts; the tile
  // choice follows that share, while the buffers still fit every route.
  const int local_routes =
      num_expert_total > num_experts
          ? std::max(1, static_cast<int>(std::lround(static_cast<double>(total_routes) *
                                                     num_experts / num_expert_total)))
          : total_routes;
  const double expected_live = expected_live_experts(local_routes, num_experts);
  const double expected_rows = static_cast<double>(local_routes) / std::max(expected_live, 1.0);

  FuseMoeMxPlan plan;
  // A 64-row tile reads each expert's weights once instead of once per 32 rows.
  // That pays early when the live weights overflow L2 and every re-read goes to
  // DRAM, and late when they stay resident and padding dominates (measured on
  // five shapes in both formats: 7168/2048 takes 64 at 32 rows, 1024/512 takes
  // 32 at 64 rows).
  const int weight_row_bytes = mxfp4_weight ? hidden_size / 2 : hidden_size;
  const bool weights_overflow_l2 =
      expected_live * 3.0 * weight_row_bytes * down_input_size >= 1.25 * l2_cache_bytes();
  const double tile_m64_rows = weights_overflow_l2 ? kTileM64RowsStreamed : kTileM64RowsResident;
  plan.tile_m = expected_rows >= tile_m64_rows ? 64 : 32;
  // Once most experts hold a row in about one tile, the 32-row gate-up GEMM streams B and
  // the 64-wide slab's deeper pipeline beats the 128-wide slab's better wave fill.
  // A short weight row has too little to stream for the depth to pay for the
  // extra tiles (measured: MXFP4 H=1024 takes 128; MXFP4 H=2560 and FP8 H=1024
  // take 64). With 64-row tiles the MXFP4 gate-up GEMM always takes the 128-wide slab
  // (2-3% at 512-2k tokens), while MXFP8 follows wave fill.
  plan.gate_up_tile_n =
      plan.tile_m == 32 && local_routes * 2 >= num_experts &&
              weight_row_bytes >= kGateDeepPipelineMinRowBytes && expected_rows <= 1.5 * plan.tile_m
          ? 64
      : plan.tile_m == 64 && mxfp4_weight
          ? 128
          : choose_tile_n(local_routes, num_experts, plan.tile_m, down_input_size, num_sms);
  plan.down_tile_n =
      choose_down_tile_n(local_routes, num_experts, plan.tile_m, hidden_size, num_sms);
  plan.gate_up_num_ctas =
      std::min(num_sms, expected_m_tiles(total_routes, num_experts, plan.tile_m) *
                            cute::ceil_div(down_input_size, plan.gate_up_tile_n));

  plan.max_padded_rows =
      cute::round_up(total_routes + (plan.tile_m - 1) * std::min(num_experts, total_routes), 128);
  plan.scale_m_padded =
      cute::round_up(sm120::packed_scale_expert_base(plan.max_padded_rows, num_experts), 4);
  plan.zero_route_counts = num_tokens > sm120::kSmallRoutingMaxTokens;

  // down GEMM adds every route into the output, so it starts as shared_output or
  // zeros. From kZeroInGateUpGemmMinTokens, plain zeros come from an idle warp of
  // gate-up GEMM, which then has DRAM to spare; below that gate-up GEMM needs it and one
  // bandwidth-bound pass in routing is cheaper.
  plan.zero_output_in_gate_up_gemm = !has_shared_output && num_tokens >= kZeroInGateUpGemmMinTokens;
  const int64_t output_bytes = static_cast<int64_t>(num_tokens) * hidden_size * 2;
  const int64_t gate_up_tiles = std::max<int64_t>(
      1, static_cast<int64_t>(expected_m_tiles(local_routes, num_experts, plan.tile_m)) *
             cute::ceil_div(down_input_size, plan.gate_up_tile_n));
  plan.zero_output_chunk = plan.zero_output_in_gate_up_gemm
                               ? (output_bytes / 16 + gate_up_tiles - 1) / gate_up_tiles * 16
                               : 0;

  // L2 policy for both GEMMs' TMA loads. Weights stream through L2 once when each
  // live expert fits one M tile, so they go first and stop evicting the
  // activations, which every N tile re-reads; a weight set that fits L2 stays
  // resident instead. Activations stay last while an expert holds few rows.
  // Measured on six MoE shapes: 3-10% from 8 tokens to 2k, while weight reuse
  // across M tiles loses up to 47% under evict-first.
  const bool stream_weights = weights_overflow_l2 && expected_rows <= 0.8 * plan.tile_m;
  const bool keep_activations = num_tokens > 1 && expected_rows <= 100.0;
  plan.cache_hint_activation = keep_activations ? kEvictLast : kEvictNormal;
  plan.cache_hint_weight = stream_weights ? kEvictFirst : kEvictNormal;
  return plan;
}

void fuse_moe_mx_async(void *output_ptr, const void *input_ptr, const void *input_scale_ptr,
                       bool block128_scale, const void *gate_up_weight_ptr,
                       const void *gate_up_weight_scale_ptr, const void *down_weight_ptr,
                       const void *down_weight_scale_ptr, const void *topk_ids_ptr,
                       const void *topk_scale_ptr, const void *shared_output_ptr, bool init_output,
                       void *topk_pos_ptr, void *row_to_route_ptr, void *num_tokens_per_group_ptr,
                       void *cu_padded_tokens_per_group_ptr, void *cu_tiles_ptr,
                       void *route_counts_ptr, void *gate_up_input_ptr,
                       void *gate_up_input_scale_ptr, void *down_input_ptr,
                       void *down_input_scale_ptr, int num_tokens, int num_topk, int hidden_size,
                       int down_input_size, int num_expert_local, int rank_ep, bool mxfp4_weight,
                       float swiglu_limit, const FuseMoeMxPlan &plan, cudaStream_t stream) {
  const int num_experts = num_expert_local;
  const int first_expert = rank_ep * num_experts;
  const int last_expert = first_expert + num_experts;
  const bool bf16_input = input_scale_ptr == nullptr;
  const auto *topk_ids = static_cast<const int *>(topk_ids_ptr);
  auto *topk_pos = static_cast<int *>(topk_pos_ptr);
  auto *row_to_route = static_cast<int *>(row_to_route_ptr);
  auto *counts = static_cast<int *>(num_tokens_per_group_ptr);
  auto *cu_padded_tokens_per_group = static_cast<int *>(cu_padded_tokens_per_group_ptr);
  auto *cu_tiles = static_cast<int *>(cu_tiles_ptr);
  auto *gate_up_input_scale = static_cast<int32_t *>(gate_up_input_scale_ptr);
  // gate-up GEMM reads E4M3 input in place; BF16 input is quantized token-major first.
  const void *gate_up_input = bf16_input ? gate_up_input_ptr : input_ptr;

  const int64_t output_bytes = static_cast<int64_t>(num_tokens) * hidden_size * 2;
  void *init_dst = init_output ? output_ptr : nullptr;
  const int64_t init_bytes = init_output ? output_bytes : 0;
  const int num_sms = get_sm_count();

  if (num_tokens == 1) {
    sm120::prepare_single_token_input_async(
        input_ptr, input_scale_ptr, block128_scale, topk_ids, topk_pos, row_to_route, counts,
        cu_padded_tokens_per_group, cu_tiles, gate_up_input_ptr, gate_up_input_scale, num_topk,
        hidden_size, num_experts, first_expert, plan.tile_m, plan.max_padded_rows,
        plan.scale_m_padded, shared_output_ptr, init_dst, init_bytes, stream);
  } else {
    sm120::prepare_routes_async(topk_ids, topk_pos, row_to_route, counts,
                                cu_padded_tokens_per_group, cu_tiles,
                                static_cast<int *>(route_counts_ptr), num_tokens, num_topk,
                                num_experts, first_expert, plan.tile_m, plan.max_padded_rows,
                                num_sms, shared_output_ptr, init_dst, init_bytes, stream);
    // The routing leaves each route's position within its expert in topk_pos
    // once it counts in several blocks; input preparation turns it into a padded row.
    const bool finalize_route_rows = plan.zero_route_counts;
    if (bf16_input) {
      sm120::quantize_tokens_and_scatter_scales_async(
          input_ptr, topk_ids, cu_padded_tokens_per_group, topk_pos, row_to_route,
          gate_up_input_ptr, gate_up_input_scale, num_tokens, num_topk, hidden_size, first_expert,
          last_expert, plan.scale_m_padded, finalize_route_rows, stream);
    } else {
      sm120::scatter_token_scales_async(
          input_scale_ptr, block128_scale, topk_ids, cu_padded_tokens_per_group, topk_pos,
          row_to_route, gate_up_input_scale, num_tokens, num_topk, hidden_size, first_expert,
          last_expert, plan.scale_m_padded, finalize_route_rows, stream);
    }
  }

  sm120::gate_up_gemm_async(
      mxfp4_weight, gate_up_input, gate_up_weight_ptr, gate_up_input_scale,
      gate_up_weight_scale_ptr, down_input_ptr, down_input_scale_ptr, cu_padded_tokens_per_group,
      counts, row_to_route, num_tokens, num_topk, plan.max_padded_rows, down_input_size,
      hidden_size, num_experts, plan.tile_m, plan.gate_up_tile_n, plan.gate_up_num_ctas,
      swiglu_limit, plan.zero_output_in_gate_up_gemm ? output_ptr : nullptr,
      plan.zero_output_in_gate_up_gemm ? output_bytes : 0, plan.zero_output_chunk,
      plan.cache_hint_activation, plan.cache_hint_weight, stream);
  sm120::down_gemm_async(mxfp4_weight, down_input_ptr, down_weight_ptr, down_input_scale_ptr,
                         down_weight_scale_ptr, output_ptr, cu_padded_tokens_per_group, counts,
                         row_to_route, static_cast<const float *>(topk_scale_ptr), num_topk,
                         plan.max_padded_rows, hidden_size, down_input_size, num_experts,
                         plan.tile_m, plan.down_tile_n, num_sms, plan.cache_hint_activation,
                         plan.cache_hint_weight, stream);
}

}  // namespace fuse_moe
}  // namespace hpc
