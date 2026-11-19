// Copyright (C) 2026 Tencent.

#include <c10/util/Exception.h>

#include <cute/config.hpp>

#include "src/fuse_moe/sm120/common/launch.cuh"
#include "src/fuse_moe/sm120/common/utils.h"
#include "src/fuse_moe/sm120/down_gemm.cuh"
#include "src/fuse_moe/sm120/gemm/down/config.cuh"
#include "src/fuse_moe/sm120/gemm/down/kernel.cuh"

namespace hpc::fuse_moe::sm120 {
namespace {

using namespace hpc::fuse_moe::sm120::gemm::down;  // NOLINT

using ElementA = cute::float_e4m3_t;
using ElementD = cute::bfloat16_t;

template <typename ElementB, int TileM, int TileN, bool Hinted>
void launch_down_gemm(const void *down_input, const void *down_weight, void *down_input_scale,
                      const void *down_weight_scale, void *output,
                      const int32_t *cu_padded_tokens_per_group, const int32_t *row_counts,
                      const int32_t *row_to_route, const float *topk_scale, int num_topk,
                      int max_padded_rows, int hidden_size, int down_input_size, int num_experts,
                      int num_ctas, uint64_t cache_hint_activation, uint64_t cache_hint_weight,
                      cudaStream_t stream) {
  // Deepest pipelines that still leave the epilogue its own SMEM: 64x128 stages
  // are 25 KB, the others at most 21 KB. FP4 B unpacks to a byte per element in
  // SMEM, so FP8 B stages are the same size.
  constexpr int kStages = TileM == 64 && TileN == 128 ? 3 : 4;
  using KT = DownGemmConfig<ElementB, TileM, TileN, kStages>;
  using Kernel = DownGemmKernel<KT, Hinted>;
  using ElementSFLoad = typename KT::SFConfig::ElementSFLoad;

  typename Kernel::Arguments args{
      reinterpret_cast<ElementA *>(const_cast<void *>(down_input)),
      typename KT::ABLoadConfig::StrideA{static_cast<int64_t>(down_input_size), cute::Int<1>{},
                                         static_cast<int64_t>(max_padded_rows) * down_input_size},
      reinterpret_cast<ElementB *>(const_cast<void *>(down_weight)),
      typename KT::ABLoadConfig::StrideB{static_cast<int64_t>(down_input_size), cute::Int<1>{},
                                         static_cast<int64_t>(hidden_size) * down_input_size},
      reinterpret_cast<ElementSFLoad *>(down_input_scale),
      reinterpret_cast<ElementSFLoad *>(const_cast<void *>(down_weight_scale)),
      reinterpret_cast<ElementD *>(output),
      cu_padded_tokens_per_group,
      row_counts,
      row_to_route,
      topk_scale,
      num_topk,
      cache_hint_activation,
      cache_hint_weight};
  auto problem_shape = cute::make_shape(max_padded_rows, hidden_size, down_input_size, num_experts);
  launch_cutlass_kernel<Kernel>(Kernel::to_underlying_arguments(problem_shape, args), num_ctas,
                                stream);
}

}  // namespace

void down_gemm_async(bool mxfp4_weight, const void *down_input, const void *down_weight,
                     void *down_input_scale, const void *down_weight_scale, void *output,
                     const int32_t *cu_padded_tokens_per_group, const int32_t *row_counts,
                     const int32_t *row_to_route, const float *topk_scale, int num_topk,
                     int max_padded_rows, int hidden_size, int down_input_size, int num_experts,
                     int tile_m, int tile_n, int num_ctas, uint64_t cache_hint_activation,
                     uint64_t cache_hint_weight, cudaStream_t stream) {
  constexpr uint64_t kNormal = static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL);
  const bool hinted = cache_hint_activation != kNormal || cache_hint_weight != kNormal;
  auto run = [&]<typename ElementB, int TileM, int TileN, bool Hinted>() {
    launch_down_gemm<ElementB, TileM, TileN, Hinted>(
        down_input, down_weight, down_input_scale, down_weight_scale, output,
        cu_padded_tokens_per_group, row_counts, row_to_route, topk_scale, num_topk, max_padded_rows,
        hidden_size, down_input_size, num_experts, num_ctas, cache_hint_activation,
        cache_hint_weight, stream);
  };  // NOLINT
  auto by_tile = [&]<typename ElementB>() {
    return dispatch_tile_mn(tile_m, tile_n, [&]<int TileM, int TileN>() {
      hinted ? run.template operator()<ElementB, TileM, TileN, true>()
             : run.template operator()<ElementB, TileM, TileN, false>();
    });
  };  // NOLINT
  const bool launched = mxfp4_weight ? by_tile.template operator()<cute::float_e2m1_t>()
                                     : by_tile.template operator()<cute::float_e4m3_t>();
  TORCH_CHECK(launched, "unsupported down GEMM tile_m/tile_n pair: ", tile_m, " x ", tile_n);
}

}  // namespace hpc::fuse_moe::sm120
