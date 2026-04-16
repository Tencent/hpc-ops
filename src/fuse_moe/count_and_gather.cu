// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include <cub/cub.cuh>

#include "cute/tensor.hpp"
#include "cutlass/arch/reg_reconfig.h"
#include "src/fuse_moe/fuse_moe.h"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"
#include "src/utils/utils.h"

namespace hpc {
namespace fuse_moe {

namespace kernels {

template <bool kUsePDL = false>
__global__ void count_seq_kernel(const int *topk_ids_ptr, int *topk_pos_ptr, int *seqlens_ptr,
                                 int total_num_topk, int start_expert, int end_expert) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }
  if ((idx < total_num_topk)) {
    int iexpert = topk_ids_ptr[idx];
    if ((iexpert >= start_expert) && (iexpert < end_expert)) {
      atomicAdd(&seqlens_ptr[iexpert - start_expert], 1);
    }
    topk_pos_ptr[idx] = -1;
  }
  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

template <int kThreadPerBlock, int kGroupPerThread, int kTileM, bool kUsePDL = false>
__global__ void count_cuseq_kernel(const int *topk_ids_ptr, int *topk_pos_ptr, int *seqlens_ptr,
                                   int *cu_seqlens_ptr, int *tiles_ptr, int *cu_tiles_ptr,
                                   int num_seq, int num_topk, int total_num_topk, int num_expert,
                                   int start_expert, int end_expert) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  // cusum
  int thread_seqs[kGroupPerThread];
  int thread_tiles[kGroupPerThread];
  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }
#pragma unroll
  for (int i = 0; i < kGroupPerThread; i++) {
    int igroup = idx * kGroupPerThread + i;
    if (igroup < num_expert) {
      int iseq = seqlens_ptr[igroup];
      int itile_num = (iseq + kTileM - 1) / kTileM;
      thread_seqs[i] = iseq;
      thread_tiles[i] = itile_num;
      tiles_ptr[igroup] = itile_num;
    } else {
      thread_seqs[i] = 0;
      thread_tiles[i] = 0;
    }
  }
  using BlockScan = cub::BlockScan<int, kThreadPerBlock>;
  __shared__ typename BlockScan::TempStorage temp_storage1;
  __shared__ typename BlockScan::TempStorage temp_storage2;
  int seqs_aggregate, tiles_aggregate;
  BlockScan(temp_storage1).ExclusiveSum(thread_seqs, thread_seqs, seqs_aggregate);
  BlockScan(temp_storage2).ExclusiveSum(thread_tiles, thread_tiles, tiles_aggregate);

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
      cu_tiles_ptr[igroup] = thread_tiles[i];
    }
  }
  if (idx == 0) {
    cu_seqlens_ptr[num_expert] = seqs_aggregate;
    cu_tiles_ptr[num_expert] = tiles_aggregate;
  }
  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

template <int kThreadPerBlock, int kGroupPerThread, int kTileM, bool kUsePDL = false>
__global__ void count_seq_and_cuseq_kernel(const int *topk_ids_ptr, int *topk_pos_ptr,
                                           int *seqlens_ptr, int *cu_seqlens_ptr, int *tiles_ptr,
                                           int *cu_tiles_ptr, int num_seq, int num_topk,
                                           int total_num_topk, int num_expert, int start_expert,
                                           int end_expert) {
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
  int thread_tiles[kGroupPerThread];
#pragma unroll
  for (int i = 0; i < kGroupPerThread; i++) {
    int igroup = idx * kGroupPerThread + i;
    if (igroup < num_expert) {
      int iseq = seqlens_shm[igroup];
      int itile_num = (iseq + kTileM - 1) / kTileM;
      thread_seqs[i] = iseq;
      thread_tiles[i] = itile_num;
      tiles_ptr[igroup] = itile_num;
    } else {
      thread_seqs[i] = 0;
      thread_tiles[i] = 0;
    }
  }
  using BlockScan = cub::BlockScan<int, kThreadPerBlock>;
  __shared__ typename BlockScan::TempStorage temp_storage1;
  __shared__ typename BlockScan::TempStorage temp_storage2;
  int seqs_aggregate, tiles_aggregate;
  BlockScan(temp_storage1).ExclusiveSum(thread_seqs, thread_seqs, seqs_aggregate);
  BlockScan(temp_storage2).ExclusiveSum(thread_tiles, thread_tiles, tiles_aggregate);

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
      cu_tiles_ptr[igroup] = thread_tiles[i];
    }
  }
  if (idx == 0) {
    cu_seqlens_ptr[num_expert] = seqs_aggregate;
    cu_tiles_ptr[num_expert] = tiles_aggregate;
  }
  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

template <typename T1, typename T2, typename GateUpTmaX, typename GateUpTmaY, typename DownTmaX,
          typename DownTmaY, int kWarpPerBlock, int kTileM, bool kAssignTask, bool kUsePDL = false>
__global__ void gather_kernel(const vec_t<cute::TmaDescriptor, 4> td_xy,
                              cute::TmaDescriptor *gate_up_tma_xy, cute::TmaDescriptor *down_tma_xy,
                              T1 *gate_up_input_ptr, T2 *gate_up_output_ptr, T1 *down_input_ptr,
                              T2 *down_output_ptr, const T1 *x_ptr, const int *topk_ids_ptr,
                              int *topk_pos_ptr, int *seqlens_ptr, int *cu_seqlens_ptr,
                              int4 *gateup_task_map_ptr, int4 *down_task_map_ptr, int *cu_tiles_ptr,
                              int total_num_topk, int hidden_size, int intermediate_size,
                              int start_expert, int end_expert, int num_block_for_copy,
                              int num_sm_count, cutlass::FastDivmod topk_divider) {
  constexpr int kThreadPerWarp = 32;
  int idx = threadIdx.x;
  int iblock = blockIdx.x;
  int iwarp = idx / kThreadPerWarp;
  int ilane = idx % kThreadPerWarp;
  int itopk = iblock * kWarpPerBlock + iwarp;

  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }

  if (iblock < num_block_for_copy) {
    if (itopk < total_num_topk) {
      int iexpert = topk_ids_ptr[itopk];
      if ((iexpert >= start_expert) && (iexpert < end_expert)) {
        int pos_in_expert;
        if (ilane == 0) {
          pos_in_expert = atomicAdd(&seqlens_ptr[iexpert - start_expert], 1);
        }

        pos_in_expert = __shfl_sync(0xFFFFFFFF, pos_in_expert, 0);

        uint64_t irow = cu_seqlens_ptr[iexpert - start_expert] + pos_in_expert;
        int iseq, res;
        topk_divider(iseq, res, itopk);

        auto y_irow_ptr = gate_up_input_ptr + irow * hidden_size;
        auto x_irow_ptr = x_ptr + iseq * hidden_size;

        constexpr int kNumItemPer16B = 16 / sizeof(T1);
        int total_items = hidden_size / kNumItemPer16B;

        for (int i = ilane; i < total_items; i += kThreadPerWarp) {
          store<T1, kNumItemPer16B>(y_irow_ptr + i * kNumItemPer16B,
                                    load<T1, kNumItemPer16B>(x_irow_ptr + i * kNumItemPer16B));
        }
        topk_pos_ptr[itopk] = irow;
      }
    }
  } else {
    if (idx < 32) {
      using namespace cute;  // NOLINT

      int igroup = iblock - num_block_for_copy;

      __shared__ cute::TmaDescriptor smem_tma_desc[4];

      int num_seq = cu_seqlens_ptr[igroup + 1] - cu_seqlens_ptr[igroup];
      uint64_t cu_seqlen = cu_seqlens_ptr[igroup];
      auto *x_gate_up_ibatch_ptr = gate_up_input_ptr + cu_seqlen * hidden_size;
      auto *y_gate_up_ibatch_ptr = gate_up_output_ptr + cu_seqlen * intermediate_size;
      auto *x_down_ibatch_ptr = down_input_ptr + cu_seqlen * intermediate_size / 2;
      auto *y_down_ibatch_ptr = down_output_ptr + cu_seqlen * hidden_size;

      int k = hidden_size;
      int n = intermediate_size;

      if (idx < 4) {
        smem_tma_desc[idx] = td_xy[idx];
      }
      __syncwarp();

      // X_gate_up
      if (idx == 0) {
        auto gX = make_tensor(make_gmem_ptr(x_gate_up_ibatch_ptr), make_shape(num_seq, k),
                              make_stride(k, Int<1>{}));
        update_tma_gtensor<GateUpTmaX>(smem_tma_desc[idx], gX);
      }

      // Y_gate_up
      if (idx == 1) {
        auto gY = make_tensor(make_gmem_ptr(y_gate_up_ibatch_ptr), make_shape(n, num_seq),
                              make_stride(Int<1>{}, n));
        update_tma_gtensor<GateUpTmaY>(smem_tma_desc[idx], gY);
      }

      // X_down
      if (idx == 2) {
        auto gX = make_tensor(make_gmem_ptr(x_down_ibatch_ptr), make_shape(num_seq, n / 2),
                              make_stride(n / 2, Int<1>{}));
        update_tma_gtensor<DownTmaX>(smem_tma_desc[idx], gX);
      }

      // Y_down
      if (idx == 3) {
        auto gY = make_tensor(make_gmem_ptr(y_down_ibatch_ptr), make_shape(k, num_seq),
                              make_stride(Int<1>{}, k));
        update_tma_gtensor<DownTmaY>(smem_tma_desc[idx], gY);
      }

#pragma unroll
      for (int i = 0; i < 2; i++) {
        __syncwarp();
        if (cute::elect_one_sync()) {
          cute::tma_desc_commit_group();
          cute::tma_desc_wait_group();
        }
        tma_descriptor_cp_fence_release(gate_up_tma_xy + igroup * 2 + i, smem_tma_desc[i]);
        tma_descriptor_cp_fence_release(down_tma_xy + igroup * 2 + i, smem_tma_desc[i + 2]);
      }
    } else if (iwarp == 1) {
      if constexpr (kAssignTask) {
        if constexpr (kTileM == 8) {
          constexpr int kTileN = 128;
          int num_group = gridDim.x - num_block_for_copy;
          int igroup = iblock - num_block_for_copy;
          int num_tile_n = (intermediate_size + kTileN - 1) / kTileN;
          int cu_tile_m = cu_tiles_ptr[igroup];
          int num_tile_m = cu_tiles_ptr[igroup + 1] - cu_tile_m;
          int cu_tiles = cu_tile_m * num_tile_n;

          for (int im = 0; im < num_tile_m; im++) {
            for (int in = ilane; in < num_tile_n; in += 32) {
              int itile = cu_tiles + im * num_tile_n + in;
              int4 task;
              task.x = im;
              task.y = in;
              task.z = igroup;
              gateup_task_map_ptr[itile] = task;
            }
          }

          if (igroup == num_group - 1) {
            int num_gateup_tiles = (num_tile_m + cu_tile_m) * num_tile_n;
            for (int i = ilane; i < num_sm_count; i += 32) {
              int4 task;
              task.z = -1;
              gateup_task_map_ptr[num_gateup_tiles + i] = task;
            }
          }
        }
      }

    } else if (iwarp == 2) {
      if constexpr (kAssignTask) {
        if constexpr (kTileM == 8) {
          constexpr int kTileN = 128;
          int num_group = gridDim.x - num_block_for_copy;
          int igroup = iblock - num_block_for_copy;
          int num_tile_n = (hidden_size + kTileN - 1) / kTileN;
          int cu_tile_m = cu_tiles_ptr[igroup];
          int num_tile_m = cu_tiles_ptr[igroup + 1] - cu_tile_m;
          int cu_tiles = cu_tile_m * num_tile_n;

          for (int im = 0; im < num_tile_m; im++) {
            for (int in = ilane; in < num_tile_n; in += 32) {
              int itile = cu_tiles + im * num_tile_n + in;
              int4 task;
              task.x = im;
              task.y = in;
              task.z = igroup;
              down_task_map_ptr[itile] = task;
            }
          }

          if (igroup == num_group - 1) {
            int num_down_tiles = (num_tile_m + cu_tile_m) * num_tile_n;
            for (int i = ilane; i < num_sm_count; i += 32) {
              int4 task;
              task.z = -1;
              down_task_map_ptr[num_down_tiles + i] = task;
            }
          }
        }
      }
    }
  }
  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

}  // namespace kernels

template <int kTileM, int kTileN, int kTileK, int kStage, bool kUsePDL = false,
          int kDownTileK = kTileK>
void launch_count_and_gather(void *gate_up_input_ptr, void *gate_up_output_ptr,
                             void *down_input_ptr, void *down_output_ptr, const void *x_ptr,
                             const void *topk_ids_ptr, void *topk_pos_ptr, void *seqlens_ptr,
                             void *cu_seqlens_ptr, void *gate_up_tmas_ptr, void *down_tmas_ptr,
                             void *tiles_ptr, void *cu_tiles_ptr, void *gateup_task_map_ptr,
                             void *down_task_map_ptr, int num_seq, int hidden_size,
                             int intermediate_size, int num_topk, int num_expert, int eprank,
                             int num_seq_per_group_avg, cudaStream_t stream) {
  using namespace cute;  // NOLINT

  using Tin = cute::float_e4m3_t;
  using Tout = cute::bfloat16_t;

  int m = num_seq;
  int n = intermediate_size;
  int k = hidden_size;
  auto X_gate_up = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(gate_up_input_ptr)),
                               make_shape(m * num_topk, k), make_stride(k, Int<1>{}));
  auto Y_gate_up = make_tensor(make_gmem_ptr(reinterpret_cast<Tout *>(gate_up_output_ptr)),
                               make_shape(n, m * num_topk), make_stride(Int<1>{}, n));

  auto X_down = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(down_input_ptr)),
                            make_shape(m * num_topk, n / 2), make_stride(n / 2, Int<1>{}));
  auto Y_down = make_tensor(make_gmem_ptr(reinterpret_cast<Tout *>(down_output_ptr)),
                            make_shape(k, m * num_topk), make_stride(Int<1>{}, k));

  // gate_up TMA X
  auto slayout_x = tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{},
                                 make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{}));

  auto cpbox_yt = tile_to_shape(GMMA::Layout_MN_SW64_Atom<Tout>{},
                                make_shape(Int<kTileN / 2>{}, Int<kTileM>{}));

  auto gata_up_tma_x = make_tma_copy(SM90_TMA_LOAD{}, X_gate_up, slayout_x(_, _, 0));
  auto gata_up_tma_y = make_tma_copy(SM90_TMA_STORE{}, Y_gate_up, cpbox_yt);

  // down TMA X: use SW64 when kDownTileK <= 64, otherwise SW128
  constexpr bool kDownUseSW64 = (kDownTileK <= 64);
  auto down_slayout_x = [&]() {
    if constexpr (kDownUseSW64) {
      return tile_to_shape(GMMA::Layout_K_SW64_Atom<Tin>{},
                           make_shape(Int<kTileM>{}, Int<kDownTileK>{}, Int<kStage>{}));
    } else {
      return tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{},
                           make_shape(Int<kTileM>{}, Int<kDownTileK>{}, Int<kStage>{}));
    }
  }();
  auto down_tma_x = make_tma_copy(SM90_TMA_LOAD{}, X_down, down_slayout_x(_, _, 0));
  auto down_tma_y = make_tma_copy(SM90_TMA_STORE{}, Y_down, cpbox_yt);

  auto *gate_up_tma_xy = static_cast<cute::TmaDescriptor *>(gate_up_tmas_ptr);
  auto *down_tma_xy = static_cast<cute::TmaDescriptor *>(down_tmas_ptr);

  int total_num_topk = num_seq * num_topk;
  int start_expert = eprank * num_expert;
  int end_expert = (eprank + 1) * num_expert;

  // 0. count tokens
  {
    constexpr int kThreadPerBlock = 256;
    constexpr int kGroupPerThread = 2;

    if (num_seq <= 128) {
      dim3 block(kThreadPerBlock);
      dim3 grid(1);
      cudaLaunchAttribute attribute[1];
      attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attribute[0].val.programmaticStreamSerializationAllowed = 1;

      // Set the attribute in a kernel launch configuration
      cudaLaunchConfig_t config{};

      // Base launch configuration
      config.gridDim = grid;
      config.blockDim = block;
      config.dynamicSmemBytes = num_expert * 4;
      config.stream = stream;

      // Add special attribute for PDL
      config.attrs = attribute;
      config.numAttrs = 1;
      auto kernel =
          kernels::count_seq_and_cuseq_kernel<kThreadPerBlock, kGroupPerThread, kTileM, kUsePDL>;
      cudaLaunchKernelEx(&config, kernel, (const int *)topk_ids_ptr, (int *)topk_pos_ptr,
                         (int *)seqlens_ptr, (int *)cu_seqlens_ptr, (int *)tiles_ptr,
                         (int *)cu_tiles_ptr, num_seq, num_topk, total_num_topk, num_expert,
                         start_expert, end_expert);
    } else {
      // 0.1 count seqlens
      {
        dim3 block(kThreadPerBlock);
        dim3 grid((total_num_topk + kThreadPerBlock - 1) / kThreadPerBlock);

        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute[0].val.programmaticStreamSerializationAllowed = 1;

        // Set the attribute in a kernel launch configuration
        cudaLaunchConfig_t config{};

        // Base launch configuration
        config.gridDim = grid;
        config.blockDim = block;
        config.dynamicSmemBytes = 0;
        config.stream = stream;

        // Add special attribute for PDL
        config.attrs = attribute;
        config.numAttrs = 1;
        auto kernel = kernels::count_seq_kernel<kUsePDL>;
        cudaLaunchKernelEx(&config, kernel, (const int *)topk_ids_ptr, (int *)topk_pos_ptr,
                           (int *)seqlens_ptr, total_num_topk, start_expert, end_expert);
      }
      // 0.2 count cu_seqlens
      {
        dim3 block(kThreadPerBlock);
        dim3 grid(1);

        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute[0].val.programmaticStreamSerializationAllowed = 1;

        // Set the attribute in a kernel launch configuration
        cudaLaunchConfig_t config{};

        // Base launch configuration
        config.gridDim = grid;
        config.blockDim = block;
        config.dynamicSmemBytes = num_expert * 4;
        config.stream = stream;

        // Add special attribute for PDL
        config.attrs = attribute;
        config.numAttrs = 1;
        auto kernel =
            kernels::count_cuseq_kernel<kThreadPerBlock, kGroupPerThread, kTileM, kUsePDL>;
        cudaLaunchKernelEx(&config, kernel, (const int *)topk_ids_ptr, (int *)topk_pos_ptr,
                           (int *)seqlens_ptr, (int *)cu_seqlens_ptr, (int *)tiles_ptr,
                           (int *)cu_tiles_ptr, num_seq, num_topk, total_num_topk, num_expert,
                           start_expert, end_expert);
      }
    }
  }

  // 1. gather token and update tmas
  {
    vec_t<cute::TmaDescriptor, 4> td_xy{
        *gata_up_tma_x.get_tma_descriptor(),
        *gata_up_tma_y.get_tma_descriptor(),
        *down_tma_x.get_tma_descriptor(),
        *down_tma_y.get_tma_descriptor(),
    };

    constexpr int kWarpPerBlock = 4;
    int num_block_for_copy = (total_num_topk + kWarpPerBlock - 1) / kWarpPerBlock;
    int num_sm_count = get_sm_count();

    cutlass::FastDivmod topk_divider(num_topk);

    dim3 block(kWarpPerBlock * 32);
    dim3 grid(num_block_for_copy + num_expert);

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = 1;

    // Set the attribute in a kernel launch configuration
    cudaLaunchConfig_t config{};

    // Base launch configuration
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 0;
    config.stream = stream;

    // Add special attribute for PDL
    config.attrs = attribute;
    config.numAttrs = 1;

    if (gateup_task_map_ptr && down_task_map_ptr) {
      constexpr int kAssignTask = true;
      auto kernel =
          kernels::gather_kernel<Tin, Tout, decltype(gata_up_tma_x), decltype(gata_up_tma_y),
                                 decltype(down_tma_x), decltype(down_tma_y), kWarpPerBlock, kTileM,
                                 kAssignTask, kUsePDL>;
      cudaLaunchKernelEx(
          &config, kernel, td_xy, gate_up_tma_xy, down_tma_xy, (Tin *)gate_up_input_ptr,
          (Tout *)gate_up_output_ptr, (Tin *)down_input_ptr, (Tout *)down_output_ptr,
          (const Tin *)x_ptr, (const int *)topk_ids_ptr, (int *)topk_pos_ptr, (int *)seqlens_ptr,
          (int *)cu_seqlens_ptr, (int4 *)gateup_task_map_ptr, (int4 *)down_task_map_ptr,
          (int *)cu_tiles_ptr, total_num_topk, hidden_size, intermediate_size, start_expert,
          end_expert, num_block_for_copy, num_sm_count, topk_divider);
    } else {
      constexpr int kAssignTask = false;
      auto kernel =
          kernels::gather_kernel<Tin, Tout, decltype(gata_up_tma_x), decltype(gata_up_tma_y),
                                 decltype(down_tma_x), decltype(down_tma_y), kWarpPerBlock, kTileM,
                                 kAssignTask, kUsePDL>;
      cudaLaunchKernelEx(
          &config, kernel, td_xy, gate_up_tma_xy, down_tma_xy, (Tin *)gate_up_input_ptr,
          (Tout *)gate_up_output_ptr, (Tin *)down_input_ptr, (Tout *)down_output_ptr,
          (const Tin *)x_ptr, (const int *)topk_ids_ptr, (int *)topk_pos_ptr, (int *)seqlens_ptr,
          (int *)cu_seqlens_ptr, (int4 *)gateup_task_map_ptr, (int4 *)down_task_map_ptr,
          (int *)cu_tiles_ptr, total_num_topk, hidden_size, intermediate_size, start_expert,
          end_expert, num_block_for_copy, num_sm_count, topk_divider);
    }
  }
}

void count_and_gather_async(void *gate_up_input_ptr, void *gate_up_output_ptr, void *down_input_ptr,
                            void *down_output_ptr, const void *x_ptr, const void *topk_ids_ptr,
                            void *topk_pos_ptr, void *seqlens_ptr, void *cu_seqlens_ptr,
                            void *gate_up_tmas_ptr, void *down_tmas_ptr, void *tiles_ptr,
                            void *cu_tiles_ptr, void *gateup_task_map_ptr, void *down_task_map_ptr,
                            int num_seq, int hidden_size, int intermediate_size, int num_topk,
                            int num_expert, int eprank, int num_seq_per_group_avg,
                            cudaStream_t stream) {
  constexpr int kTileN = 128;
  constexpr int kTileK = 128;

  constexpr bool kUsePDL = true;

  if (num_seq_per_group_avg <= 8) {
    constexpr int kTileM = 8;
    constexpr int kStage = 8;
    launch_count_and_gather<kTileM, kTileN, kTileK, kStage, kUsePDL>(
        gate_up_input_ptr, gate_up_output_ptr, down_input_ptr, down_output_ptr, x_ptr, topk_ids_ptr,
        topk_pos_ptr, seqlens_ptr, cu_seqlens_ptr, gate_up_tmas_ptr, down_tmas_ptr, tiles_ptr,
        cu_tiles_ptr, gateup_task_map_ptr, down_task_map_ptr, num_seq, hidden_size,
        intermediate_size, num_topk, num_expert, eprank, num_seq_per_group_avg, stream);
  } else if (num_seq_per_group_avg <= 16) {
    constexpr int kTileM = 16;
    constexpr int kStage = 8;
    launch_count_and_gather<kTileM, kTileN, kTileK, kStage, kUsePDL>(
        gate_up_input_ptr, gate_up_output_ptr, down_input_ptr, down_output_ptr, x_ptr, topk_ids_ptr,
        topk_pos_ptr, seqlens_ptr, cu_seqlens_ptr, gate_up_tmas_ptr, down_tmas_ptr, tiles_ptr,
        cu_tiles_ptr, gateup_task_map_ptr, down_task_map_ptr, num_seq, hidden_size,
        intermediate_size, num_topk, num_expert, eprank, num_seq_per_group_avg, stream);
  } else if (num_seq_per_group_avg <= 32) {
    constexpr int kTileM = 32;
    constexpr int kStage = 8;
    launch_count_and_gather<kTileM, kTileN, kTileK, kStage, kUsePDL>(
        gate_up_input_ptr, gate_up_output_ptr, down_input_ptr, down_output_ptr, x_ptr, topk_ids_ptr,
        topk_pos_ptr, seqlens_ptr, cu_seqlens_ptr, gate_up_tmas_ptr, down_tmas_ptr, tiles_ptr,
        cu_tiles_ptr, gateup_task_map_ptr, down_task_map_ptr, num_seq, hidden_size,
        intermediate_size, num_topk, num_expert, eprank, num_seq_per_group_avg, stream);
  } else if (num_seq_per_group_avg <= 48) {
    constexpr int kTileM = 48;
    constexpr int kStage = 8;
    launch_count_and_gather<kTileM, kTileN, kTileK, kStage, kUsePDL>(
        gate_up_input_ptr, gate_up_output_ptr, down_input_ptr, down_output_ptr, x_ptr, topk_ids_ptr,
        topk_pos_ptr, seqlens_ptr, cu_seqlens_ptr, gate_up_tmas_ptr, down_tmas_ptr, tiles_ptr,
        cu_tiles_ptr, gateup_task_map_ptr, down_task_map_ptr, num_seq, hidden_size,
        intermediate_size, num_topk, num_expert, eprank, num_seq_per_group_avg, stream);
  } else {
    constexpr int kTileM = 64;
    constexpr int kStage = 8;
    // use kTileK=64 + SW64 for down TMA when K is small
    if (intermediate_size / 2 <= 192) {
      constexpr int kDownTileK = 64;
      launch_count_and_gather<kTileM, kTileN, kTileK, kStage, kUsePDL, kDownTileK>(
          gate_up_input_ptr, gate_up_output_ptr, down_input_ptr, down_output_ptr, x_ptr,
          topk_ids_ptr, topk_pos_ptr, seqlens_ptr, cu_seqlens_ptr, gate_up_tmas_ptr, down_tmas_ptr,
          tiles_ptr, cu_tiles_ptr, gateup_task_map_ptr, down_task_map_ptr, num_seq, hidden_size,
          intermediate_size, num_topk, num_expert, eprank, num_seq_per_group_avg, stream);
    } else {
      launch_count_and_gather<kTileM, kTileN, kTileK, kStage, kUsePDL>(
          gate_up_input_ptr, gate_up_output_ptr, down_input_ptr, down_output_ptr, x_ptr,
          topk_ids_ptr, topk_pos_ptr, seqlens_ptr, cu_seqlens_ptr, gate_up_tmas_ptr, down_tmas_ptr,
          tiles_ptr, cu_tiles_ptr, gateup_task_map_ptr, down_task_map_ptr, num_seq, hidden_size,
          intermediate_size, num_topk, num_expert, eprank, num_seq_per_group_avg, stream);
    }
  }
}

}  // namespace fuse_moe
}  // namespace hpc
