// Copyright (C) 2026 Tencent.

#include <c10/util/Exception.h>

#include <cute/config.hpp>

#include "src/fuse_moe/sm120/common/launch.cuh"
#include "src/fuse_moe/sm120/common/utils.h"
#include "src/fuse_moe/sm120/gate_up_gemm.cuh"
#include "src/fuse_moe/sm120/gemm/gate_up/kernel.cuh"

namespace hpc::fuse_moe::sm120 {
namespace {

using namespace hpc::fuse_moe::sm120::gemm::gate_up;  // NOLINT

using ElementA = cute::float_e4m3_t;
using ElementD = cute::float_e4m3_t;

template <typename ElementB, int TileM, int TileN, bool Hinted>
void launch_gate_up_gemm(const void *gate_up_input, const void *gate_up_weight,
                         void *gate_up_input_scale, const void *gate_up_weight_scale,
                         void *down_input, void *down_input_scale,
                         const int32_t *cu_padded_tokens_per_group, const int32_t *row_counts,
                         const int32_t *row_to_route, int num_tokens, int num_topk,
                         int max_padded_rows, int down_input_size, int hidden_size, int num_experts,
                         int num_ctas, float swiglu_limit, void *zero_dst, int64_t zero_bytes,
                         int64_t zero_chunk, uint64_t cache_hint_activation,
                         uint64_t cache_hint_weight, cudaStream_t stream) {
  // Deepest pipelines that still leave the epilogue its own SMEM: a stage holds A
  // and both B slabs, 21 KB at 32x64 and 25 KB at 64x64. The 128-wide slabs only
  // fit two stages. FP4 B unpacks to a byte per element in SMEM, so FP8 B stages
  // are the same size.
  constexpr int kStages = TileN == 128 ? 2 : TileM == 64 ? 3 : 4;
  using KT = GateUpGemmConfig<ElementB, TileM, TileN, kStages>;
  using Kernel = GateUpGemmKernel<KT, Hinted>;
  using ElementSFLoad = typename KT::SFConfig::ElementSFLoad;

  typename Kernel::Arguments args{
      reinterpret_cast<ElementA *>(const_cast<void *>(gate_up_input)),
      reinterpret_cast<ElementB *>(const_cast<void *>(gate_up_weight)),
      typename KT::ABLoadConfig::StrideB{static_cast<int64_t>(hidden_size), cute::Int<1>{},
                                         static_cast<int64_t>(2 * down_input_size) * hidden_size},
      reinterpret_cast<ElementSFLoad *>(gate_up_input_scale),
      reinterpret_cast<ElementSFLoad *>(const_cast<void *>(gate_up_weight_scale)),
      reinterpret_cast<ElementD *>(down_input),
      reinterpret_cast<int32_t *>(down_input_scale),
      cu_padded_tokens_per_group,
      row_counts,
      row_to_route,
      num_tokens,
      num_topk,
      swiglu_limit,
      static_cast<int4 *>(zero_dst),
      zero_bytes / static_cast<int64_t>(sizeof(int4)),
      zero_chunk / static_cast<int64_t>(sizeof(int4)),
      cache_hint_activation,
      cache_hint_weight};
  auto problem_shape = cute::make_shape(max_padded_rows, down_input_size, hidden_size, num_experts);
  launch_cutlass_kernel<Kernel>(Kernel::to_underlying_arguments(problem_shape, args), num_ctas,
                                stream);
}

}  // namespace

void gate_up_gemm_async(bool mxfp4_weight, const void *gate_up_input, const void *gate_up_weight,
                        void *gate_up_input_scale, const void *gate_up_weight_scale,
                        void *down_input, void *down_input_scale,
                        const int32_t *cu_padded_tokens_per_group, const int32_t *row_counts,
                        const int32_t *row_to_route, int num_tokens, int num_topk,
                        int max_padded_rows, int down_input_size, int hidden_size, int num_experts,
                        int tile_m, int tile_n, int num_ctas, float swiglu_limit, void *zero_dst,
                        int64_t zero_bytes, int64_t zero_chunk, uint64_t cache_hint_activation,
                        uint64_t cache_hint_weight, cudaStream_t stream) {
  constexpr uint64_t kNormal = static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL);
  const bool hinted = cache_hint_activation != kNormal || cache_hint_weight != kNormal;
  auto run = [&]<typename ElementB, int TileM, int TileN, bool Hinted>() {
    launch_gate_up_gemm<ElementB, TileM, TileN, Hinted>(
        gate_up_input, gate_up_weight, gate_up_input_scale, gate_up_weight_scale, down_input,
        down_input_scale, cu_padded_tokens_per_group, row_counts, row_to_route, num_tokens,
        num_topk, max_padded_rows, down_input_size, hidden_size, num_experts, num_ctas,
        swiglu_limit, zero_dst, zero_bytes, zero_chunk, cache_hint_activation, cache_hint_weight,
        stream);
  };  // NOLINT
  auto by_tile = [&]<typename ElementB>() {
    return dispatch_tile_mn(tile_m, tile_n, [&]<int TileM, int TileN>() {
      hinted ? run.template operator()<ElementB, TileM, TileN, true>()
             : run.template operator()<ElementB, TileM, TileN, false>();
    });
  };  // NOLINT
  const bool launched = mxfp4_weight ? by_tile.template operator()<cute::float_e2m1_t>()
                                     : by_tile.template operator()<cute::float_e4m3_t>();
  TORCH_CHECK(launched, "unsupported gate-up GEMM tile_m/tile_n pair: ", tile_m, " x ", tile_n);
}

}  // namespace hpc::fuse_moe::sm120
