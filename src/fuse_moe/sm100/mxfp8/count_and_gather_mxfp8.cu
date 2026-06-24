// Copyright 2026 hpc-ops authors

#include <cooperative_groups.h>
#include <cuda.h>

#include <algorithm>
#include <cub/cub.cuh>  // NOLINT(build/include_order)
#include <type_traits>

#include "cute/tensor.hpp"
#include "cutlass/fast_math.h"
#include "src/fuse_moe/sm100/count_kernels.cuh"
#include "src/fuse_moe/sm100/mxfp8/fuse_moe_mxfp8.h"
#include "src/group_gemm/sm100/group_gemm.h"
#include "src/group_gemm/sm100/mxfp8/dispatch.cuh"
#include "src/group_gemm/sm100/mxfp8/group_gemm_mxfp8.cuh"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"
#include "src/utils/utils.h"

namespace hpc {
namespace fuse_moe {

template <typename GateUpConfig, typename DownConfig>
void launch_count_and_route_row_mxfp8(
    void *gate_up_input_ptr, void *gate_up_output_ptr, void *down_input_ptr, void *down_output_ptr,
    const void *topk_ids_ptr, void *topk_pos_ptr, const void *topk_scale_ptr,
    void *gateup_x_row_map_ptr, void *seqlens_ptr, void *cu_seqlens_ptr, void *gateup_tiles_ptr,
    void *gateup_cu_tiles_ptr, void *down_tiles_ptr, void *down_cu_tiles_ptr,
    void *gate_up_tmas_ptr, void *down_tmas_ptr, int num_seq, int num_topk, int hidden_size,
    int intermediate_size, int num_expert_local, int rank_ep, cudaStream_t stream) {
  using namespace cute;  // NOLINT
  using Tin = cute::float_e4m3_t;
  using Tout = cute::bfloat16_t;
  using Tsf = uint8_t;

  // Tile-M for the tile-counting kernels — taken from each GEMM's config so the
  // tile partition matches what the downstream GEMMs consume.
  constexpr int kTileM = GateUpConfig::kTileM;
  constexpr int kTileM_Down = DownConfig::kTileM;

  int total_num_topk = num_seq * num_topk;
  int start_expert = rank_ep * num_expert_local;
  int end_expert = (rank_ep + 1) * num_expert_local;

  auto *weight_scale_row_map_ptr = reinterpret_cast<float *>(gateup_x_row_map_ptr) + total_num_topk;
  auto *x_row_map_race_pos_ptr = reinterpret_cast<int *>(gateup_x_row_map_ptr) + 2 * total_num_topk;

  // Build TMA descriptors
  int M_total = total_num_topk;
  auto X_gateup = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(gate_up_input_ptr)),
                              make_shape(M_total, hidden_size), make_stride(hidden_size, Int<1>{}));
  auto Y_gateup = make_tensor(make_gmem_ptr(reinterpret_cast<Tout *>(gate_up_output_ptr)),
                              make_shape(intermediate_size * 2, M_total),
                              make_stride(Int<1>{}, intermediate_size * 2));
  auto X_down =
      make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(down_input_ptr)),
                  make_shape(M_total, intermediate_size), make_stride(intermediate_size, Int<1>{}));
  auto Y_down = make_tensor(make_gmem_ptr(reinterpret_cast<Tout *>(down_output_ptr)),
                            make_shape(hidden_size, M_total), make_stride(Int<1>{}, hidden_size));

  // Descriptors are derived from the GEMM configs (both 1SM, kMmaSM=1) so they
  // are byte-identical to what each kernel builds. A box/op mismatch between the
  // descriptor stage-1 publishes and the one a kernel consumes deadlocks the
  // TMA mbarrier (e.g. the down GEMM consumes down_X via 1SM SM90_TMA_LOAD).
  //
  //   * gateup_X : unused by the GateUp GEMM (loads X via cp.async) but built
  //                from GateUpConfig for a consistent descriptor type.
  //   * gateup_Y : consumed by the GateUp epilogue TMA store.
  //   * down_X   : consumed by the Down GEMM A-TMA load — the box must match
  //                DownConfig::SLayoutA exactly.
  //   * down_Y   : unused by the Down GEMM (atomic scatter-add reduce) but built
  //                from DownConfig for a consistent descriptor type.
  static_assert(GateUpConfig::kMmaSM == 1 && DownConfig::kMmaSM == 1,
                "fuse_moe stage-1 descriptors assume 1SM GateUp/Down GEMM configs");

  typename GateUpConfig::SLayoutA gateup_a_layout{};
  typename GateUpConfig::SLayoutYT gateup_yt_layout{};
  typename DownConfig::SLayoutA down_a_layout{};
  typename DownConfig::SLayoutYT down_yt_layout{};
  auto gateup_x_layout = gateup_a_layout(_, _, 0);
  auto gateup_y_layout = gateup_yt_layout(_, _, 0);
  auto down_x_layout = down_a_layout(_, _, 0);
  auto down_y_layout = down_yt_layout(_, _, 0);
  auto tma_gateup_x = make_tma_copy(SM90_TMA_LOAD{}, X_gateup, gateup_x_layout);
  auto tma_gateup_y = make_tma_copy(SM90_TMA_STORE{}, Y_gateup, gateup_y_layout);
  auto tma_down_x = make_tma_copy(SM90_TMA_LOAD{}, X_down, down_x_layout);
  auto tma_down_y = make_tma_copy(SM90_TMA_STORE{}, Y_down, down_y_layout);

  vec_t<cute::TmaDescriptor, 4> td_xy{
      *tma_gateup_x.get_tma_descriptor(),
      *tma_gateup_y.get_tma_descriptor(),
      *tma_down_x.get_tma_descriptor(),
      *tma_down_y.get_tma_descriptor(),
  };

  auto *gate_up_tma_xy = static_cast<cute::TmaDescriptor *>(gate_up_tmas_ptr);
  auto *down_tma_xy = static_cast<cute::TmaDescriptor *>(down_tmas_ptr);

  cutlass::FastDivmod topk_divider(num_topk);

  if (num_seq <= 2048) {
    // Fused kernel path: single launch with cluster DSMEM
    constexpr int kFusedThreadPerBlock = 512;
    constexpr int kFusedClusterSize = 4;
    constexpr int kFusedGroupPerThread = 2;
    constexpr int kTmaUpdateExpertsPerBlock = (kFusedThreadPerBlock / 32) / 4;

    const int update_tma_blocks =
        ((num_expert_local + kTmaUpdateExpertsPerBlock - 1) / kTmaUpdateExpertsPerBlock +
         kFusedClusterSize - 1) /
        kFusedClusterSize * kFusedClusterSize;

    cudaLaunchAttribute attrs[2];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    attrs[1].id = cudaLaunchAttributeClusterDimension;
    attrs[1].val.clusterDim = {kFusedClusterSize, 1, 1};

    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(kFusedClusterSize + update_tma_blocks);
    cfg.blockDim = dim3(kFusedThreadPerBlock);
    cfg.dynamicSmemBytes =
        (2 * num_expert_local + 1 + num_expert_local * kFusedClusterSize) * sizeof(int);
    cfg.stream = stream;
    cfg.attrs = attrs;
    cfg.numAttrs = 2;

    auto kernel = kernels::fused_count_cuseq_route_kernel<
        kFusedThreadPerBlock, kFusedClusterSize, kFusedGroupPerThread, kTileM, kTileM_Down,
        kTmaUpdateExpertsPerBlock, Tin, Tout, Tout, decltype(tma_gateup_x), decltype(tma_gateup_y),
        decltype(tma_down_x), decltype(tma_down_y), true>;
    cudaLaunchKernelEx(
        &cfg, kernel, reinterpret_cast<const int *>(topk_ids_ptr),
        reinterpret_cast<int *>(topk_pos_ptr), reinterpret_cast<const float *>(topk_scale_ptr),
        reinterpret_cast<int *>(seqlens_ptr), reinterpret_cast<int *>(cu_seqlens_ptr),
        reinterpret_cast<int *>(gateup_tiles_ptr), reinterpret_cast<int *>(gateup_cu_tiles_ptr),
        reinterpret_cast<int *>(down_tiles_ptr), reinterpret_cast<int *>(down_cu_tiles_ptr),
        reinterpret_cast<int *>(gateup_x_row_map_ptr),
        reinterpret_cast<float *>(weight_scale_row_map_ptr), td_xy, gate_up_tma_xy, down_tma_xy,
        reinterpret_cast<const Tin *>(gate_up_input_ptr),
        reinterpret_cast<Tout *>(gate_up_output_ptr), reinterpret_cast<Tin *>(down_input_ptr),
        reinterpret_cast<Tout *>(down_output_ptr), hidden_size, intermediate_size * 2,
        intermediate_size, hidden_size, num_seq, num_topk, total_num_topk, num_expert_local,
        start_expert, end_expert, update_tma_blocks, topk_divider);
  } else {
    // 3-kernel path for large m
    int num_sm = get_sm_count();

    // 1. count_seq_kernel (cooperative grid)
    {
      constexpr int kThreadPerBlock = 256;
      int blocks =
          std::max(1, std::min(64, (total_num_topk + kThreadPerBlock - 1) / kThreadPerBlock));

      cudaLaunchAttribute attr[1];
      attr[0].id = cudaLaunchAttributeCooperative;
      attr[0].val.cooperative = 1;
      cudaLaunchConfig_t cfg{};
      cfg.gridDim = dim3(blocks);
      cfg.blockDim = dim3(kThreadPerBlock);
      cfg.dynamicSmemBytes = num_expert_local * sizeof(int);
      cfg.stream = stream;
      cfg.attrs = attr;
      cfg.numAttrs = 1;

      auto kernel = kernels::count_seq_kernel<kThreadPerBlock>;
      cudaLaunchKernelEx(&cfg, kernel, reinterpret_cast<const int *>(topk_ids_ptr),
                         reinterpret_cast<int *>(topk_pos_ptr),
                         reinterpret_cast<int *>(seqlens_ptr), total_num_topk, num_expert_local,
                         start_expert, end_expert);
    }

    // 2. count_cuseq_kernel
    {
      constexpr int kThreadPerBlock = 128;
      constexpr int kGroupPerThread = 8;

      auto kernel =
          kernels::count_cuseq_kernel<kThreadPerBlock, kGroupPerThread, kTileM, kTileM_Down>;
      kernel<<<1, kThreadPerBlock, 0, stream>>>(
          reinterpret_cast<const int *>(topk_ids_ptr), reinterpret_cast<int *>(topk_pos_ptr),
          reinterpret_cast<int *>(seqlens_ptr), reinterpret_cast<int *>(cu_seqlens_ptr),
          reinterpret_cast<int *>(gateup_tiles_ptr), reinterpret_cast<int *>(gateup_cu_tiles_ptr),
          reinterpret_cast<int *>(down_tiles_ptr), reinterpret_cast<int *>(down_cu_tiles_ptr),
          x_row_map_race_pos_ptr, num_seq, num_topk, total_num_topk, num_expert_local, start_expert,
          end_expert);
    }

    // 3. route_row_map_kernel_mxfp8
    {
      constexpr int kRouteThreadPerBlock = 256;
      constexpr int kMaxNumTopk = 8;
      int route_blocks = (num_seq + kRouteThreadPerBlock - 1) / kRouteThreadPerBlock;
      int total_blocks = route_blocks + num_expert_local;

      cudaLaunchAttribute route_attr[1];
      route_attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      route_attr[0].val.programmaticStreamSerializationAllowed = 1;
      cudaLaunchConfig_t route_cfg{};
      route_cfg.gridDim = dim3(total_blocks);
      route_cfg.blockDim = dim3(kRouteThreadPerBlock);
      route_cfg.dynamicSmemBytes = 2 * num_expert_local * static_cast<int>(sizeof(int));
      route_cfg.stream = stream;
      route_cfg.attrs = route_attr;
      route_cfg.numAttrs = 1;

      auto kernel =
          kernels::route_row_map_kernel<kRouteThreadPerBlock, kMaxNumTopk, Tin, Tout, Tout,
                                        decltype(tma_gateup_x), decltype(tma_gateup_y),
                                        decltype(tma_down_x), decltype(tma_down_y), true>;
      cudaLaunchKernelEx(
          &route_cfg, kernel, (const int *)topk_ids_ptr, (int *)topk_pos_ptr,
          (const float *)topk_scale_ptr, (int *)seqlens_ptr, (const int *)cu_seqlens_ptr, td_xy,
          gate_up_tma_xy, down_tma_xy, reinterpret_cast<Tin *>(gate_up_input_ptr),
          reinterpret_cast<Tout *>(gate_up_output_ptr), reinterpret_cast<Tin *>(down_input_ptr),
          reinterpret_cast<Tout *>(down_output_ptr), x_row_map_race_pos_ptr, hidden_size,
          intermediate_size * 2, intermediate_size, hidden_size, total_num_topk, num_topk,
          num_expert_local, start_expert, end_expert, topk_divider, (int *)gateup_x_row_map_ptr,
          (float *)weight_scale_row_map_ptr);
    }
  }
}

void count_and_route_row_mxfp8_async(
    void *gate_up_input_ptr, void *gate_up_output_ptr, void *down_input_ptr, void *down_output_ptr,
    const void *topk_ids_ptr, void *topk_pos_ptr, const void *topk_scale_ptr,
    void *gateup_x_row_map_ptr, void *seqlens_ptr, void *cu_seqlens_ptr, void *gateup_tiles_ptr,
    void *gateup_cu_tiles_ptr, void *down_tiles_ptr, void *down_cu_tiles_ptr,
    void *gate_up_tmas_ptr, void *down_tmas_ptr, int num_seq, int num_topk, int hidden_size,
    int intermediate_size, int num_expert_local, int rank_ep, int num_seq_per_group_avg,
    cudaStream_t stream) {
  using GemmTin = cute::float_e4m3_t;
  using GemmTout = cute::bfloat16_t;
  using GemmTsf = cutlass::float_ue8m0_t;

  // GateUp and Down GEMMs share the same kTileM ladder (same avg, both 1SM), so a
  // single selector instance yields both configs. We derive the stage-1 TMA
  // descriptors from these exact config types so they match the descriptors the
  // GateUp (cp_async) and Down (with_reduce) kernels consume.
  int kTileM = group_gemm::mxfp8_dispatch_kTileM(num_seq_per_group_avg);

  group_gemm::group_gemm_1sm_mxfp8_dispatch_selector<GemmTin, GemmTout, GemmTsf, GemmTin>(
      kTileM, [&](auto cfg_tag) {
        using Cfg = typename decltype(cfg_tag)::type;
        launch_count_and_route_row_mxfp8<Cfg, Cfg>(
            gate_up_input_ptr, gate_up_output_ptr, down_input_ptr, down_output_ptr, topk_ids_ptr,
            topk_pos_ptr, topk_scale_ptr, gateup_x_row_map_ptr, seqlens_ptr, cu_seqlens_ptr,
            gateup_tiles_ptr, gateup_cu_tiles_ptr, down_tiles_ptr, down_cu_tiles_ptr,
            gate_up_tmas_ptr, down_tmas_ptr, num_seq, num_topk, hidden_size, intermediate_size,
            num_expert_local, rank_ep, stream);
      });
}

}  // namespace fuse_moe
}  // namespace hpc
