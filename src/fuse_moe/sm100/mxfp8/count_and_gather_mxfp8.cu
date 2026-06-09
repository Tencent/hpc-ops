// Copyright 2026 hpc-ops authors

#include <cooperative_groups.h>
#include <cuda.h>

#include <algorithm>
#include <cub/cub.cuh>  // NOLINT(build/include_order)

#include "cute/tensor.hpp"
#include "cutlass/fast_math.h"
#include "src/fuse_moe/sm100/count_kernels.cuh"
#include "src/fuse_moe/sm100/mxfp8/fuse_moe_mxfp8.h"
#include "src/group_gemm/sm100/group_gemm.h"
#include "src/group_gemm/sm100/mxfp8/group_gemm_mxfp8.cuh"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"
#include "src/utils/utils.h"

namespace hpc {
namespace fuse_moe {

template <typename GateupCfg, typename DownCfg>
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

  // Gateup config (cp_async 1SM path)
  constexpr int kTileM = GateupCfg::kTileM;
  constexpr int kTileN = GateupCfg::kTileN;
  constexpr int kTileK = GateupCfg::kTileK;
  constexpr int kStage = GateupCfg::kStage;
  constexpr int kEpiTileM = GateupCfg::kEpiTileM;
  constexpr int kStageTMA = GateupCfg::kStageTMA;

  // Down config (may be 1SM or 2SM)
  constexpr int kTileM_Down = DownCfg::kTileM;
  constexpr int kTileN_Down = DownCfg::kTileN;
  constexpr int kMmaSM_Down = DownCfg::kMmaSM;

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

  // Gateup TMA (always 1SM cp_async path)
  auto slayout_gateup_x = tile_to_shape(UMMA::Layout_K_SW128_Atom<Tin>{},
                                        make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{}));
  auto slayout_gateup_yt =
      tile_to_shape(UMMA::Layout_MN_SW128_Atom<Tout>{},
                    make_shape(Int<kTileN>{}, Int<kEpiTileM>{}, Int<kStageTMA>{}));
  auto tma_gateup_x = make_tma_copy(SM90_TMA_LOAD{}, X_gateup, slayout_gateup_x(_, _, 0));
  auto tma_gateup_y = make_tma_copy(SM90_TMA_STORE{}, Y_gateup, slayout_gateup_yt(_, _, 0));

  // Down TMA — use DownCfg's get_tma() for correct tile shapes (handles 1SM/2SM/kTileK)
  // We create dummy B/SFA/SFB tensors since get_tma() needs them but we only use tma_a/tma_y.
  int down_k = intermediate_size;
  int down_n = hidden_size;
  constexpr int kSfVec_Down = DownCfg::kSfVec;
  constexpr int kSfxRows_Down = DownCfg::kSfxRows;
  int down_k_sf_tiles = (down_k / kSfVec_Down + 3) / 4;
  int down_n_padded = ((down_n + kTileN_Down - 1) / kTileN_Down) * kTileN_Down;
  int down_sfx_max_tiles = M_total / kTileM_Down + num_expert_local;

  auto B_down_dummy = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(down_input_ptr)),
                                  make_shape(down_n, down_k, num_expert_local),
                                  make_stride(down_k, Int<1>{}, down_n * down_k));
  auto SFA_down_dummy = make_tensor(
      make_gmem_ptr(reinterpret_cast<const Tsf *>(down_input_ptr)),
      make_shape(Int<kSfxRows_Down>{}, Int<16>{}, down_sfx_max_tiles, down_k_sf_tiles),
      make_stride(Int<16>{}, Int<1>{}, down_k_sf_tiles * kSfxRows_Down * 16, kSfxRows_Down * 16));

  constexpr int kSfbRows_Down = (kMmaSM_Down == 1) ? 32 : 64;
  auto SFB_down_dummy = make_tensor(
      make_gmem_ptr(reinterpret_cast<const Tsf *>(down_input_ptr)),
      make_shape(Int<kSfbRows_Down>{}, Int<16>{}, num_expert_local * (down_n_padded / kTileN_Down),
                 down_k_sf_tiles),
      make_stride(Int<16>{}, Int<1>{}, down_k_sf_tiles * kSfbRows_Down * 16, kSfbRows_Down * 16));

  DownCfg down_config;
  auto [tma_down_x, tma_down_b_unused, tma_down_y, tma_down_sfa_unused, tma_down_sfb_unused] =
      down_config.get_tma(X_down, B_down_dummy, Y_down, SFA_down_dummy, SFB_down_dummy);

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

namespace {
template <typename T>
struct _ti {
  using type = T;
};
}  // namespace

void count_and_route_row_mxfp8_async(
    void *gate_up_input_ptr, void *gate_up_output_ptr, void *down_input_ptr, void *down_output_ptr,
    const void *topk_ids_ptr, void *topk_pos_ptr, const void *topk_scale_ptr,
    void *gateup_x_row_map_ptr, void *seqlens_ptr, void *cu_seqlens_ptr, void *gateup_tiles_ptr,
    void *gateup_cu_tiles_ptr, void *down_tiles_ptr, void *down_cu_tiles_ptr,
    void *gate_up_tmas_ptr, void *down_tmas_ptr, int num_seq, int num_topk, int hidden_size,
    int intermediate_size, int num_expert_local, int rank_ep, int num_seq_per_group_avg,
    cudaStream_t stream) {
  using Tin = cute::float_e4m3_t;
  using Tout = cute::bfloat16_t;
  using Tsf = cutlass::float_ue8m0_t;

  // Down GEMM config dispatch: same logic as group_gemm_mxfp8_async
  bool use_2sm_down = (hidden_size % 256 == 0) && (num_seq_per_group_avg > 32);
  int kTileM_down = group_gemm::mxfp8_dispatch_kTileM(num_seq_per_group_avg, hidden_size);

  auto do_launch = [&](auto gateup_tag, auto down_tag) {
    using GateupCfg = typename decltype(gateup_tag)::type;
    using DownCfg = typename decltype(down_tag)::type;
    launch_count_and_route_row_mxfp8<GateupCfg, DownCfg>(
        gate_up_input_ptr, gate_up_output_ptr, down_input_ptr, down_output_ptr, topk_ids_ptr,
        topk_pos_ptr, topk_scale_ptr, gateup_x_row_map_ptr, seqlens_ptr, cu_seqlens_ptr,
        gateup_tiles_ptr, gateup_cu_tiles_ptr, down_tiles_ptr, down_cu_tiles_ptr, gate_up_tmas_ptr,
        down_tmas_ptr, num_seq, num_topk, hidden_size, intermediate_size, num_expert_local, rank_ep,
        stream);
  };

  // kTileM, kTileN, kTileK, kEpiTileM, kStage, kStageTM, kMmaSM, kStageTile
  auto dispatch_down = [&](auto gateup_tag) {
    if (use_2sm_down) {
      constexpr int kTileN = 256;
      constexpr int kTileK = 128;
      switch (kTileM_down) {
        case 32:
          return do_launch(gateup_tag,
                           _ti<group_gemm::GroupGEMMMxFp8Config<Tin, Tout, Tsf, 32, kTileN, kTileK,
                                                                32, 6, 4, 2, 4>>{});
        case 64:
          return do_launch(gateup_tag,
                           _ti<group_gemm::GroupGEMMMxFp8Config<Tin, Tout, Tsf, 64, kTileN, kTileK,
                                                                32, 6, 4, 2, 4>>{});
        case 96:
          return do_launch(gateup_tag,
                           _ti<group_gemm::GroupGEMMMxFp8Config<Tin, Tout, Tsf, 96, kTileN, kTileK,
                                                                32, 6, 4, 2, 4>>{});
        case 128:
          return do_launch(gateup_tag,
                           _ti<group_gemm::GroupGEMMMxFp8Config<Tin, Tout, Tsf, 128, kTileN, kTileK,
                                                                32, 6, 4, 2, 3>>{});
        case 160:
          return do_launch(gateup_tag,
                           _ti<group_gemm::GroupGEMMMxFp8Config<Tin, Tout, Tsf, 160, kTileN, kTileK,
                                                                32, 6, 4, 2, 2>>{});
        case 192:
          return do_launch(gateup_tag,
                           _ti<group_gemm::GroupGEMMMxFp8Config<Tin, Tout, Tsf, 192, kTileN, kTileK,
                                                                32, 6, 2, 2, 2>>{});
        default:
          return do_launch(gateup_tag,
                           _ti<group_gemm::GroupGEMMMxFp8Config<Tin, Tout, Tsf, 256, kTileN, kTileK,
                                                                64, 5, 2, 2, 1>>{});
      }
    } else {
      constexpr int kTileN = 128;
      constexpr int kTileK = 128;
      switch (kTileM_down) {
        case 16:
          return do_launch(gateup_tag,
                           _ti<group_gemm::GroupGEMMMxFp8Config<Tin, Tout, Tsf, 16, kTileN, kTileK,
                                                                16, 2, 4, 1, 4>>{});
        case 32:
          return do_launch(gateup_tag,
                           _ti<group_gemm::GroupGEMMMxFp8Config<Tin, Tout, Tsf, 32, kTileN, kTileK,
                                                                32, 2, 4, 1, 4>>{});
        case 48:
          return do_launch(gateup_tag,
                           _ti<group_gemm::GroupGEMMMxFp8Config<Tin, Tout, Tsf, 48, kTileN, kTileK,
                                                                16, 6, 4, 1, 4>>{});
        case 64:
          return do_launch(gateup_tag,
                           _ti<group_gemm::GroupGEMMMxFp8Config<Tin, Tout, Tsf, 64, kTileN, kTileK,
                                                                64, 6, 4, 1, 4>>{});
        case 128:
          return do_launch(gateup_tag,
                           _ti<group_gemm::GroupGEMMMxFp8Config<Tin, Tout, Tsf, 128, kTileN, kTileK,
                                                                64, 5, 3, 1, 3>>{});
        default:
          return do_launch(gateup_tag,
                           _ti<group_gemm::GroupGEMMMxFp8Config<Tin, Tout, Tsf, 256, kTileN, kTileK,
                                                                64, 3, 2, 1, 1>>{});
      }
    }
  };

  // Dispatch GateupCfg (always 1SM cp_async, kTileN=128, kTileK=128, kStageTile=1)
  // kTileM, kTileN, kTileK, kEpiTileM, kStage, kStageTM, kMmaSM, kStageTile
  if (num_seq_per_group_avg <= 16) {
    return dispatch_down(
        _ti<group_gemm::GroupGEMMMxFp8Config<Tin, Tout, Tsf, 16, 128, 128, 16, 8, 4, 1, 1>>{});
  }
  if (num_seq_per_group_avg <= 32) {
    return dispatch_down(
        _ti<group_gemm::GroupGEMMMxFp8Config<Tin, Tout, Tsf, 32, 128, 128, 32, 6, 4, 1, 1>>{});
  }
  if (num_seq_per_group_avg <= 48) {
    return dispatch_down(
        _ti<group_gemm::GroupGEMMMxFp8Config<Tin, Tout, Tsf, 48, 128, 128, 16, 6, 4, 1, 1>>{});
  }
  if (num_seq_per_group_avg <= 64) {
    return dispatch_down(
        _ti<group_gemm::GroupGEMMMxFp8Config<Tin, Tout, Tsf, 64, 128, 128, 64, 6, 4, 1, 1>>{});
  }
  if (num_seq_per_group_avg <= 128) {
    return dispatch_down(
        _ti<group_gemm::GroupGEMMMxFp8Config<Tin, Tout, Tsf, 128, 128, 128, 64, 5, 3, 1, 1>>{});
  }
  return dispatch_down(
      _ti<group_gemm::GroupGEMMMxFp8Config<Tin, Tout, Tsf, 256, 128, 128, 64, 3, 2, 1, 1>>{});
}

}  // namespace fuse_moe
}  // namespace hpc
