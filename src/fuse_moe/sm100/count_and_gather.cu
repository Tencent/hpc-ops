// Copyright 2025 hpc-ops authors

#include <cooperative_groups.h>
#include <cuda.h>
#include <stdio.h>

#include <cub/cub.cuh>

#include "cutlass/fast_math.h"
#include "src/fuse_moe/sm100/count_kernels.cuh"
#include "src/fuse_moe/sm100/fuse_moe.h"
#include "src/group_gemm/sm100/gemm_config.h"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"
#include "src/utils/utils.h"

namespace hpc {
namespace fuse_moe {

namespace kernels {

template <int kThreadPerBlock, int kGroupPerThread, int kTileM_Gateup, int kTileM_Down,
          bool kUsePDL = false>
__global__ void count_seq_and_cuseq_kernel(const int *topk_ids_ptr, int *topk_pos_ptr,
                                           int *seqlens_ptr, int *cu_seqlens_ptr,
                                           int *gateup_tiles_ptr, int *gateup_cu_tiles_ptr,
                                           int *down_tiles_ptr, int *down_cu_tiles_ptr, int num_seq,
                                           int num_topk, int total_num_topk, int num_expert,
                                           int start_expert, int end_expert) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  extern __shared__ int seqlens_shm[];

  for (int i = idx; i < num_expert; i += blockDim.x) {
    seqlens_shm[i] = 0;
  }
  __syncthreads();

  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }

  for (int i = idx; i < total_num_topk; i += blockDim.x) {
    int iexpert = topk_ids_ptr[i];
    if ((iexpert >= start_expert) && (iexpert < end_expert)) {
      atomicAdd(&seqlens_shm[iexpert - start_expert], 1);
    }
    topk_pos_ptr[i] = -1;
  }

  __syncthreads();

  // cusum
  int thread_seqs[kGroupPerThread];
  int gateup_thread_tiles[kGroupPerThread];
  int down_thread_tiles[kGroupPerThread];
#pragma unroll
  for (int i = 0; i < kGroupPerThread; i++) {
    int igroup = idx * kGroupPerThread + i;
    if (igroup < num_expert) {
      int iseq = seqlens_shm[igroup];
      int itile_num_gateup = (iseq + kTileM_Gateup - 1) / kTileM_Gateup;
      int itile_num_down = (iseq + kTileM_Down - 1) / kTileM_Down;
      thread_seqs[i] = iseq;
      gateup_thread_tiles[i] = itile_num_gateup;
      down_thread_tiles[i] = itile_num_down;
      gateup_tiles_ptr[igroup] = itile_num_gateup;
      down_tiles_ptr[igroup] = itile_num_down;
    } else {
      thread_seqs[i] = 0;
      gateup_thread_tiles[i] = 0;
      down_thread_tiles[i] = 0;
    }
  }
  using BlockScan = cub::BlockScan<int, kThreadPerBlock>;
  __shared__ typename BlockScan::TempStorage temp_storage1;
  __shared__ typename BlockScan::TempStorage temp_storage2;
  int seqs_aggregate, gateup_tiles_aggregate, down_tiles_aggregate;
  BlockScan(temp_storage1).ExclusiveSum(thread_seqs, thread_seqs, seqs_aggregate);
  BlockScan(temp_storage2)
      .ExclusiveSum(gateup_thread_tiles, gateup_thread_tiles, gateup_tiles_aggregate);
  BlockScan(temp_storage2).ExclusiveSum(down_thread_tiles, down_thread_tiles, down_tiles_aggregate);

  // store
  // fill seqlens with zero
  for (int i = idx; i < num_expert; i += blockDim.x) {
    seqlens_ptr[i] = 0;  // seqlens_shm[i];
  }

#pragma unroll
  for (int i = 0; i < kGroupPerThread; i++) {
    int igroup = idx * kGroupPerThread + i;
    if (igroup < num_expert) {
      cu_seqlens_ptr[igroup] = thread_seqs[i];
      gateup_cu_tiles_ptr[igroup] = gateup_thread_tiles[i];
      down_cu_tiles_ptr[igroup] = down_thread_tiles[i];
    }
  }
  if (idx == 0) {
    cu_seqlens_ptr[num_expert] = seqs_aggregate;
    gateup_cu_tiles_ptr[num_expert] = gateup_tiles_aggregate;
    down_cu_tiles_ptr[num_expert] = down_tiles_aggregate;
  }
  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

}  // namespace kernels

template <typename GateupGemmConfig, typename DownGemmConfig, bool kUsePDL = false>
void launch_count_and_gather(const void *topk_ids_ptr, void *topk_pos_ptr,
                             const void *topk_scale_ptr, void *seqlens_ptr, void *cu_seqlens_ptr,
                             void *gateup_tiles_ptr, void *gateup_cu_tiles_ptr,
                             void *down_tiles_ptr, void *down_cu_tiles_ptr, void *gateup_tmas_ptr,
                             void *down_tmas_ptr, const void *gate_up_input_ptr,
                             void *gate_up_output_ptr, void *down_input_ptr, void *down_output_ptr,
                             int gateup_k, int gateup_n, int down_k, int down_n, int num_seq,
                             int num_topk, int num_expert, int eprank, cudaStream_t stream,
                             void *x_row_map_ptr = nullptr) {
  using namespace cute;  // NOLINT
  int total_num_topk = num_seq * num_topk;
  int start_expert = eprank * num_expert;
  int end_expert = (eprank + 1) * num_expert;
  int num_sm_count = get_sm_count();

  auto *gate_up_tma_xy = static_cast<cute::TmaDescriptor *>(gateup_tmas_ptr);
  auto *down_tma_xy = static_cast<cute::TmaDescriptor *>(down_tmas_ptr);

  auto *weight_scale_row_map_ptr = reinterpret_cast<float *>(x_row_map_ptr) + num_seq * num_topk;
  auto *x_row_map_race_pos_ptr = reinterpret_cast<int *>(x_row_map_ptr) + 2 * num_seq * num_topk;

  auto X_gateup = make_tensor(
      make_gmem_ptr(reinterpret_cast<const typename GateupGemmConfig::Tin *>(gate_up_input_ptr)),
      make_shape(total_num_topk, gateup_k), make_stride(gateup_k, Int<1>{}));
  auto W_gateup = make_tensor(
      make_gmem_ptr(reinterpret_cast<const typename GateupGemmConfig::Tin *>(gate_up_input_ptr)),
      make_shape(gateup_n, gateup_k, num_expert),
      make_stride(gateup_k, Int<1>{}, gateup_n * gateup_k));
  auto Y_gateup = make_tensor(
      make_gmem_ptr(reinterpret_cast<typename GateupGemmConfig::Tout *>(gate_up_output_ptr)),
      make_shape(gateup_n, total_num_topk), make_stride(Int<1>{}, gateup_n));

  auto X_down = make_tensor(
      make_gmem_ptr(reinterpret_cast<const typename DownGemmConfig::Tin *>(down_input_ptr)),
      make_shape(total_num_topk, down_k), make_stride(down_k, Int<1>{}));
  auto W_down = make_tensor(
      make_gmem_ptr(reinterpret_cast<const typename DownGemmConfig::Tin *>(down_input_ptr)),
      make_shape(down_n, down_k, num_expert), make_stride(down_k, Int<1>{}, down_n * down_k));
  auto Y_down =
      make_tensor(make_gmem_ptr(reinterpret_cast<typename DownGemmConfig::Tout *>(down_output_ptr)),
                  make_shape(down_n, total_num_topk), make_stride(Int<1>{}, down_n));

  GateupGemmConfig gateup_config;
  DownGemmConfig down_config;

  auto [gata_up_tma_x, tma_w1, gata_up_tma_y] = gateup_config.get_tma(X_gateup, W_gateup, Y_gateup);
  auto [down_tma_x, tma_w2, down_tma_y] = down_config.get_tma(X_down, W_down, Y_down);

  vec_t<cute::TmaDescriptor, 4> td_xy{
      *gata_up_tma_x.get_tma_descriptor(),
      *gata_up_tma_y.get_tma_descriptor(),
      *down_tma_x.get_tma_descriptor(),
      *down_tma_y.get_tma_descriptor(),
  };

  if (num_seq <= 2048) {
    constexpr int kFusedThreadPerBlock = 512;
    // 4-block cluster: 8-block clusters on sm100 produced flaky
    // "unspecified launch failure" aborts in Stage C for num_seq >= ~1280.
    constexpr int kFusedClusterSize = 4;
    constexpr int kFusedGroupPerThread = 2;

    cutlass::FastDivmod topk_divider(num_topk);

    cudaLaunchAttribute attrs[2];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    attrs[1].id = cudaLaunchAttributeClusterDimension;
    attrs[1].val.clusterDim = {kFusedClusterSize, 1, 1};

    // update tma blocks: each block has 8 warp, 4 warp deal one expert
    // thus need num_expert / 2 blocks, and align to kFusedClusterSize
    constexpr int kTmaUpdateExpertsPerBlock = (kFusedThreadPerBlock / 32) / 4;
    const int updata_tma_blocks =
        ((num_expert / kTmaUpdateExpertsPerBlock) + kFusedClusterSize - 1) / kFusedClusterSize *
        kFusedClusterSize;
    cudaLaunchConfig_t config{};
    // One cluster = kFusedClusterSize blocks; launch exactly one cluster.
    config.gridDim = dim3(kFusedClusterSize + updata_tma_blocks);
    config.blockDim = dim3(kFusedThreadPerBlock);
    // Per-block dynamic smem:
    //   counter_shm    : num_expert ints  (local count + reused Stage C cnt)
    //   cu_seqlens_shm : num_expert + 1 ints  (rank-0 writes; others cache via DSMEM)
    //   block_base_shm : kClusterSize * num_expert ints (rank-0 computes row r,
    //                                                    block r reads its row)
    config.dynamicSmemBytes = (2 * num_expert + 1 + num_expert * kFusedClusterSize) * sizeof(int);
    config.stream = stream;
    config.attrs = attrs;
    config.numAttrs = 2;

    auto kernel = kernels::fused_count_cuseq_route_kernel<
        kFusedThreadPerBlock, kFusedClusterSize, kFusedGroupPerThread, GateupGemmConfig::kTileM,
        DownGemmConfig::kTileM, kTmaUpdateExpertsPerBlock, typename GateupGemmConfig::Tin,
        typename DownGemmConfig::Tout, typename GateupGemmConfig::Tout, decltype(gata_up_tma_x),
        decltype(gata_up_tma_y), decltype(down_tma_x), decltype(down_tma_y), kUsePDL>;
    cudaLaunchKernelEx(&config, kernel, (const int *)topk_ids_ptr, (int *)topk_pos_ptr,
                       (const float *)topk_scale_ptr, (int *)seqlens_ptr, (int *)cu_seqlens_ptr,
                       (int *)gateup_tiles_ptr, (int *)gateup_cu_tiles_ptr, (int *)down_tiles_ptr,
                       (int *)down_cu_tiles_ptr, (int *)x_row_map_ptr,
                       (float *)weight_scale_row_map_ptr, td_xy, gate_up_tma_xy, down_tma_xy,
                       reinterpret_cast<const typename GateupGemmConfig::Tin *>(gate_up_input_ptr),
                       reinterpret_cast<typename GateupGemmConfig::Tout *>(gate_up_output_ptr),
                       reinterpret_cast<typename DownGemmConfig::Tin *>(down_input_ptr),
                       reinterpret_cast<typename DownGemmConfig::Tout *>(down_output_ptr), gateup_k,
                       gateup_n, down_k, down_n, num_seq, num_topk, total_num_topk, num_expert,
                       start_expert, end_expert, num_expert / kTmaUpdateExpertsPerBlock,
                       topk_divider);
  } else {
    constexpr int kThreadPerBlock = 256;
    constexpr int kGroupPerThread = 2;

    // 0. count_seq_kernel
    {
      int num_block = (total_num_topk + kThreadPerBlock - 1) / kThreadPerBlock;
      const int coop_max_block = num_sm_count * 2;
      if (num_block > coop_max_block) {
        num_block = coop_max_block;
      }

      cudaLaunchAttribute attrs[2];
      attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attrs[0].val.programmaticStreamSerializationAllowed = 1;
      attrs[1].id = cudaLaunchAttributeCooperative;
      attrs[1].val.cooperative = 1;

      cudaLaunchConfig_t config{};
      config.gridDim = dim3(num_block);
      config.blockDim = dim3(kThreadPerBlock);
      config.dynamicSmemBytes = num_expert * sizeof(int);
      config.stream = stream;
      config.attrs = attrs;
      config.numAttrs = 2;

      auto kernel = kernels::count_seq_kernel<kThreadPerBlock, kUsePDL>;
      cudaLaunchKernelEx(&config, kernel, (const int *)topk_ids_ptr, (int *)topk_pos_ptr,
                         (int *)seqlens_ptr, total_num_topk, num_expert, start_expert, end_expert);
    }

    // 1. count_cuseq_kernel
    {
      cudaLaunchAttribute attrs[1];
      attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attrs[0].val.programmaticStreamSerializationAllowed = 1;

      cudaLaunchConfig_t config{};
      config.gridDim = dim3(1);
      config.blockDim = dim3(kThreadPerBlock);
      config.dynamicSmemBytes = num_expert * sizeof(int);
      config.stream = stream;
      config.attrs = attrs;
      config.numAttrs = 1;

      auto kernel =
          kernels::count_cuseq_kernel<kThreadPerBlock, kGroupPerThread, GateupGemmConfig::kTileM,
                                      DownGemmConfig::kTileM, kUsePDL>;
      cudaLaunchKernelEx(&config, kernel, (const int *)topk_ids_ptr, (int *)topk_pos_ptr,
                         (int *)seqlens_ptr, (int *)cu_seqlens_ptr, (int *)gateup_tiles_ptr,
                         (int *)gateup_cu_tiles_ptr, (int *)down_tiles_ptr,
                         (int *)down_cu_tiles_ptr, x_row_map_race_pos_ptr, num_seq, num_topk,
                         total_num_topk, num_expert, start_expert, end_expert);
    }

    // 2. route tokens -> topk_pos (+ optional x_row_map).
    {
      cutlass::FastDivmod topk_divider(num_topk);

      cudaLaunchAttribute attrs[1];
      attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attrs[0].val.programmaticStreamSerializationAllowed = 1;

      // route_row_map_kernel: 1 thread per iseq, warp-coalesced atomicAdd
      // into global seqlens_ptr (relies on seqlens_ptr being 0 coming in).
      constexpr int kRouteThreadPerBlock = 256;
      constexpr int kMaxNumTopk = 8;  // matches MoE configs exercised in-repo.
      int route_blocks = (num_seq + kRouteThreadPerBlock - 1) / kRouteThreadPerBlock;
      if (route_blocks < 1) {
        route_blocks = 1;
      }

      cudaLaunchConfig_t config{};
      config.gridDim = dim3(route_blocks + num_expert);
      config.blockDim = dim3(kRouteThreadPerBlock);
      // counter_shm + cu_seqlens_shm, each num_expert ints.
      config.dynamicSmemBytes = 2 * num_expert * sizeof(int);
      config.stream = stream;
      config.attrs = attrs;
      config.numAttrs = 1;

      auto kernel = kernels::route_row_map_kernel<
          kRouteThreadPerBlock, kMaxNumTopk, typename GateupGemmConfig::Tin,
          typename DownGemmConfig::Tout, typename GateupGemmConfig::Tout, decltype(gata_up_tma_x),
          decltype(gata_up_tma_y), decltype(down_tma_x), decltype(down_tma_y), kUsePDL>;
      cudaLaunchKernelEx(
          &config, kernel, (const int *)topk_ids_ptr, (int *)topk_pos_ptr,
          (const float *)topk_scale_ptr, (int *)seqlens_ptr, (const int *)cu_seqlens_ptr, td_xy,
          gate_up_tma_xy, down_tma_xy,
          reinterpret_cast<const typename GateupGemmConfig::Tin *>(gate_up_input_ptr),
          reinterpret_cast<typename GateupGemmConfig::Tout *>(gate_up_output_ptr),
          reinterpret_cast<typename DownGemmConfig::Tin *>(down_input_ptr),
          reinterpret_cast<typename DownGemmConfig::Tout *>(down_output_ptr),
          x_row_map_race_pos_ptr, gateup_k, gateup_n, down_k, down_n, total_num_topk, num_topk,
          num_expert, start_expert, end_expert, topk_divider, (int *)x_row_map_ptr,
          (float *)weight_scale_row_map_ptr);
    }
  }
}

void count_and_gather_async(const void *topk_ids_ptr, void *topk_pos_ptr,
                            const void *topk_scale_ptr, void *seqlens_ptr, void *cu_seqlens_ptr,
                            void *tiles_ptr, void *cu_tiles_ptr, void *gateup_tmas_ptr,
                            void *down_tmas_ptr, const void *gate_up_input_ptr,
                            void *gate_up_output_ptr, void *down_input_ptr, void *down_output_ptr,
                            int gateup_k, int gateup_n, int down_k, int down_n, int num_seq,
                            int num_topk, int num_expert, int eprank, int num_seq_per_group_avg,
                            cudaStream_t stream, void *x_row_map_ptr, bool fuse_act) {
  constexpr bool kUsePDL = true;

  void *gateup_tiles_ptr = reinterpret_cast<void *>(reinterpret_cast<int32_t *>(tiles_ptr));
  void *down_tiles_ptr =
      reinterpret_cast<void *>(reinterpret_cast<int32_t *>(tiles_ptr) + num_expert);
  void *gateup_cu_tiles_ptr = reinterpret_cast<void *>(reinterpret_cast<int32_t *>(cu_tiles_ptr));
  void *down_cu_tiles_ptr =
      reinterpret_cast<void *>(reinterpret_cast<int32_t *>(cu_tiles_ptr) + num_expert + 1);

  auto launch = [&](auto num_seq_per_group_avg_tag, auto gateup_func_tag, auto down_func_tag) {
    constexpr int kNumSeqPerGroupAvg = decltype(num_seq_per_group_avg_tag)::value;
    constexpr group_gemm::GroupGemmFunc kGateupGemmFunc =
        static_cast<group_gemm::GroupGemmFunc>(decltype(gateup_func_tag)::value);
    constexpr group_gemm::GroupGemmFunc kDownGemmFunc =
        static_cast<group_gemm::GroupGemmFunc>(decltype(down_func_tag)::value);
    using GateupGemmConfig =
        decltype(group_gemm::get_group_gemm_config<kGateupGemmFunc, kNumSeqPerGroupAvg>());
    using DownGemmConfig =
        decltype(group_gemm::get_group_gemm_config<kDownGemmFunc, kNumSeqPerGroupAvg>());

    launch_count_and_gather<GateupGemmConfig, DownGemmConfig, kUsePDL>(
        topk_ids_ptr, topk_pos_ptr, topk_scale_ptr, seqlens_ptr, cu_seqlens_ptr, gateup_tiles_ptr,
        gateup_cu_tiles_ptr, down_tiles_ptr, down_cu_tiles_ptr, gateup_tmas_ptr, down_tmas_ptr,
        gate_up_input_ptr, gate_up_output_ptr, down_input_ptr, down_output_ptr, gateup_k, gateup_n,
        down_k, down_n, num_seq, num_topk, num_expert, eprank, stream, x_row_map_ptr);
  };

  auto dispatch_down = [&](auto num_seq_per_group_avg_tag, auto gateup_func_tag) {
    if (down_n % 256 == 0 && num_seq_per_group_avg > 32) {
      launch(num_seq_per_group_avg_tag, gateup_func_tag,
             std::integral_constant<int, static_cast<int>(
                                             group_gemm::GroupGemmFunc::GROUP_GEMM_2SM_FP8)>());
    } else {
      launch(num_seq_per_group_avg_tag, gateup_func_tag,
             std::integral_constant<int, static_cast<int>(
                                             group_gemm::GroupGemmFunc::GROUP_GEMM_1SM_FP8)>());
    }
  };

  auto dispatch_gateup = [&](auto num_seq_per_group_avg_tag) {
    if (fuse_act) {
      dispatch_down(
          num_seq_per_group_avg_tag,
          std::integral_constant<
              int, static_cast<int>(group_gemm::GroupGemmFunc::GROUP_GEMM_CP_ASYNC_FP8_ACT_MUL)>());
    } else {
      dispatch_down(
          num_seq_per_group_avg_tag,
          std::integral_constant<int, static_cast<int>(
                                          group_gemm::GroupGemmFunc::GROUP_GEMM_CP_ASYNC_FP8)>());
    }
  };

  if (num_seq_per_group_avg <= 16) {
    dispatch_gateup(std::integral_constant<int, 16>{});
  } else if (num_seq_per_group_avg <= 32) {
    dispatch_gateup(std::integral_constant<int, 32>{});
  } else if (num_seq_per_group_avg <= 48) {
    dispatch_gateup(std::integral_constant<int, 48>{});
  } else if (num_seq_per_group_avg <= 64) {
    dispatch_gateup(std::integral_constant<int, 64>{});
  } else if (num_seq_per_group_avg <= 96) {
    dispatch_gateup(std::integral_constant<int, 96>{});
  } else if (num_seq_per_group_avg <= 128) {
    dispatch_gateup(std::integral_constant<int, 128>{});
  } else if (num_seq_per_group_avg <= 160) {
    dispatch_gateup(std::integral_constant<int, 160>{});
  } else if (num_seq_per_group_avg <= 192) {
    dispatch_gateup(std::integral_constant<int, 192>{});
  } else {
    dispatch_gateup(std::integral_constant<int, 256>{});
  }
}

}  // namespace fuse_moe
}  // namespace hpc
