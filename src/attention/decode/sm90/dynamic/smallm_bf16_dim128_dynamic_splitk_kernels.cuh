// Copyright 2025 hpc-ops authors

#ifndef SRC_ATTENTION_DECODE_SM90_DYNAMIC_SMALLM_BF16_DIM128_DYNAMIC_SPLITK_KERNELS_CUH_
#define SRC_ATTENTION_DECODE_SM90_DYNAMIC_SMALLM_BF16_DIM128_DYNAMIC_SPLITK_KERNELS_CUH_

#include <cuda.h>
#include <stdio.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "src/attention/decode/sched_task_info.h"
#include "src/attention/decode/sm90/util_kernels.cuh"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace attention {
namespace decode {
namespace kernels {

template <typename Tin, typename Tout, int kTileM, int kTileN, int kTileK, int kTileV,
          int kHeadsPerGroup, typename TiledMmaQK, typename TiledMmaSV, typename TmaQ,
          typename TmaK, typename TmaV, typename TmaSplitY, typename SLayoutQ, typename SLayoutK,
          typename SLayoutP, typename SLayoutS, typename SLayoutV, typename SLayoutSplitY,
          int kBlockSize, int kStage>
__global__ void smallm_attention_decode_bf16_dynamic_splitk_kernel(
    const __grid_constant__ TmaQ tma_q, const __grid_constant__ TmaK tma_k,
    const __grid_constant__ TmaV tma_v, const __grid_constant__ TmaSplitY tma_splity,
    float* split_y_ptr, float* lse_ptr, const int* task_map_ptr, const int* block_ids_ptr,
    int num_batch, int num_seq_q, int num_dim_qk, int num_dim_v, int num_head_q, int num_head_k,
    int num_head_v, int heads_per_group, int lse_pad_heads_per_group, int num_kvcache_blocks,
    int num_seq_max_blocks, float one_over_dk_log2e, int max_splitk) {
  using namespace cute;  // NOLINT

  constexpr int kMathThreads = size(TiledMmaQK{});
  constexpr int kWarpsPerWrapGroup = 4;
  constexpr int kTaskStride = dynamic::kTaskScheduleInfoStride;

  int idx = threadIdx.x;
  int elected = cute::elect_one_sync();
  int iwarp = idx / 32;
  int iblock = blockIdx.x;

  cudaGridDependencySynchronize();

  int num_tiles_per_cta = task_map_ptr[0];

  __shared__ uint64_t q_readable;
  __shared__ uint64_t q_writable;
  __shared__ uint64_t k_writable[kStage];
  __shared__ uint64_t v_writable[kStage];
  __shared__ uint64_t k_readable[kStage];
  __shared__ uint64_t v_readable[kStage];
  extern __shared__ uint8_t shm_data[] alignas(128);

  auto* shm_q = reinterpret_cast<Tin*>(shm_data);
  auto* shm_k = shm_q + cosize(SLayoutQ{});
  auto* shm_v = shm_k + cosize(SLayoutK{});
  auto* shm_p = shm_v + cosize(SLayoutV{});
  auto* shm_max = reinterpret_cast<float*>(shm_p + cosize(SLayoutP{}));
  auto* shm_kvblk_ids = reinterpret_cast<int*>(shm_max + kTileM * kWarpsPerWrapGroup);
  auto* shm_splity = reinterpret_cast<float*>(shm_data);  // reuse shm_q/k/v/p for epilogue.

  auto gQ = tma_q.get_tma_tensor(
      make_shape(heads_per_group, num_dim_qk, num_head_k, num_seq_q, num_batch));
  auto gK =
      tma_k.get_tma_tensor(make_shape(kBlockSize, num_dim_qk, num_head_k, num_kvcache_blocks));
  auto gV = tma_v.get_tma_tensor(make_shape(num_dim_v, kBlockSize, num_head_v, num_kvcache_blocks));
  auto gSplitY = tma_splity.get_tma_tensor(
      make_shape(num_dim_v, heads_per_group, num_head_k, num_seq_q, max_splitk, num_batch));

  auto gAtt =
      make_tensor(make_gmem_ptr(static_cast<float*>(nullptr)),
                  make_shape(Int<kTileN>{}, Int<kTileM>{}), make_stride(Int<kTileM>{}, Int<1>{}));
  auto gYY =
      make_tensor(make_gmem_ptr(static_cast<float*>(nullptr)),
                  make_shape(Int<kTileV>{}, Int<kTileM>{}), make_stride(Int<1>{}, Int<kTileV>{}));

  auto sQ = make_tensor(make_smem_ptr(shm_q), SLayoutQ{});
  auto sK = make_tensor(make_smem_ptr(shm_k), SLayoutK{});
  auto sP = make_tensor(make_smem_ptr(shm_p), SLayoutP{});
  auto sS = make_tensor(make_smem_ptr(shm_p), SLayoutS{});
  auto sV = make_tensor(make_smem_ptr(shm_v), SLayoutV{});
  auto sSplitY = make_tensor(make_smem_ptr(shm_splity), SLayoutSplitY{});

  auto btma_q = tma_q.get_slice(0);
  auto btma_k = tma_k.get_slice(0);
  auto btma_v = tma_v.get_slice(0);

  auto tQg = btma_q.partition_S(gQ);
  auto tKg = btma_k.partition_S(gK);
  auto tVg = btma_v.partition_S(gV);

  auto tQs = btma_q.partition_D(sQ);
  auto tKs = btma_k.partition_D(sK);
  auto tVs = btma_v.partition_D(sV);

  if (iwarp == 0 && elected) {
    initialize_barrier(q_readable, 1);
    initialize_barrier(q_writable, 1);
    cutlass::arch::fence_barrier_init();
  } else if (iwarp == 1 && elected) {
#pragma unroll
    for (int i = 0; i < kStage; i++) {
      initialize_barrier(k_readable[i], 1);
      initialize_barrier(k_writable[i], 1);
    }
    cutlass::arch::fence_barrier_init();
  } else if (iwarp == 2 && elected) {
#pragma unroll
    for (int i = 0; i < kStage; i++) {
      initialize_barrier(v_readable[i], 1);
      initialize_barrier(v_writable[i], 1);
    }
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();

  const int* block_task_ptr = task_map_ptr + (1 + iblock * num_tiles_per_cta) * kTaskStride;

  if (idx >= kMathThreads) {
    // ====================== Load warp (32 threads) ======================
    const bool is_leader_in_load = ((iwarp == kMathThreads / 32) && elected);
    const int ilane = idx & 31;

    int phase_q = 1;
    int istage_kv = 0;
    int phase_kv = 1;

    while (true) {
      TaskInfo task;
      if (!get_task<kBlockSize>(task, block_task_ptr)) {
        break;
      }

      const int ihead_kv = task.ihead_kv;
      const int ibatch = task.ibatch;
      const int num_blocks = task.num_blocks;
      const int num_tile_kv = task.num_tile_kv;
      const int num_tile_full = task.num_tile_full;

      const int* block_ids =
          block_ids_ptr + ibatch * num_seq_max_blocks + task.num_blocks_per_chunk;
      for (int i = ilane; i < num_blocks; i += 32) {
        shm_kvblk_ids[i] = block_ids[i];
      }
      __syncwarp();

      if (is_leader_in_load) {
        wait_barrier(q_writable, phase_q);
        phase_q ^= 1;

        for (int iseqq = 0; iseqq < num_seq_q; iseqq++) {
          cute::copy(tma_q.with(q_readable), tQg(_, 0, _, ihead_kv, iseqq, ibatch),
                     tQs(_, iseqq, _));
        }
        set_barrier_transaction_bytes(q_readable, sizeof(Tin) * cosize(SLayoutQ{}));

        constexpr int kBlockPerTileN = kTileN / kBlockSize;
        int iload_tile = 0;

#pragma unroll 1
        for (int itile_seq_kv = num_tile_full; itile_seq_kv < num_tile_kv; ++itile_seq_kv) {
          load_paged_kv<true, kBlockPerTileN, kBlockSize, kStage, Tin>(
              tma_k, tma_v, k_writable, v_writable, k_readable, v_readable, tKg, tKs, tVg, tVs,
              ihead_kv, num_dim_qk, num_dim_v, shm_kvblk_ids, num_blocks, itile_seq_kv, istage_kv,
              phase_kv);
          advance_stage<kStage>(istage_kv, phase_kv);
        }

#pragma unroll 1
        for (int itile_seq_kv = -kStage + 1; itile_seq_kv < num_tile_full; ++itile_seq_kv) {
          if (iload_tile < num_tile_full) {
            load_paged_kv<false, kBlockPerTileN, kBlockSize, kStage, Tin>(
                tma_k, tma_v, k_writable, v_writable, k_readable, v_readable, tKg, tKs, tVg, tVs,
                ihead_kv, num_dim_qk, num_dim_v, shm_kvblk_ids, num_blocks, iload_tile++, istage_kv,
                phase_kv);
            advance_stage<kStage>(istage_kv, phase_kv);
          }
        }
      }

      __syncwarp();
      block_task_ptr += kTaskStride;
    }
  } else {
    // ====================== Math warpgroup (128 threads) ======================
    const int ilane = idx & 31;
    const bool is_leader = (iwarp == 0) && elected;
    constexpr int kIWarpgroup = 0;

    TiledMmaQK tiled_mma_qk;
    TiledMmaSV tiled_mma_sv;

    auto thr_mma_qk = tiled_mma_qk.get_slice(idx);
    auto thr_mma_sv = tiled_mma_sv.get_slice(idx);

    auto tKs4r = thr_mma_qk.partition_A(sK);
    auto tQs4r = thr_mma_qk.partition_B(sQ);
    auto tVs4r = thr_mma_sv.partition_A(sV);
    auto tSs4r = thr_mma_sv.partition_B(sS);

    auto tKr = thr_mma_qk.make_fragment_A(tKs4r);
    auto tQr = thr_mma_qk.make_fragment_B(tQs4r);
    auto tVr = thr_mma_sv.make_fragment_A(tVs4r);
    auto tSr = thr_mma_sv.make_fragment_B(tSs4r);

    auto tAttr = thr_mma_qk.partition_fragment_C(gAtt);
    auto tAttAbf16 = make_tensor_like<Tin>(tAttr);
    auto tYr = thr_mma_sv.partition_fragment_C(gYY);

    auto gI = make_identity_tensor(gAtt.shape());
    auto tI = thr_mma_qk.partition_C(gI);

    auto tI_nm = retile_fragment(tI);
    auto tYr_nm = retile_fragment(tYr);

    auto tAttr_nm = retile_fragment(tAttr);
    constexpr int kN = size<0>(tAttr_nm);
    constexpr int kM = size<1>(tAttr_nm);
    Tensor gMax = make_tensor<float>(Int<kM>{});
    Tensor gSum = make_tensor<float>(Int<kM>{});
    Tensor gSoftmaxScale = make_tensor<float>(Int<kM>{});

    using R2SCopyAtomP = conditional_t<(kTileM % 16 == 0), Copy_Atom<SM90_U16x8_STSM_T, Tin>,
                                       Copy_Atom<SM90_U16x4_STSM_T, Tin>>;
    auto tiled_copy_P_r2s = make_tiled_copy_C(R2SCopyAtomP{}, tiled_mma_qk);
    auto thr_copy_P_r2s = tiled_copy_P_r2s.get_slice(idx);
    auto tPr4s = thr_copy_P_r2s.retile_S(tAttAbf16);
    auto tPs4r = thr_copy_P_r2s.partition_D(sP);

    using R2SCopyAtomSplitY = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, float>;
    auto tiled_copy_SplitY_r2s = make_tiled_copy_C(R2SCopyAtomSplitY{}, tiled_mma_sv);

    int istage_kv = 0;
    int phase_kv = 0;
    int phase_q = 0;

    while (true) {
      TaskInfo task;
      if (!get_task<kBlockSize>(task, block_task_ptr)) {
        break;
      }

      const int ihead_kv = task.ihead_kv;
      const int ibatch = task.ibatch;
      const int ichunk = task.ichunk;
      const int num_seq_kvcache = task.num_seq_kvcache;
      const int num_seq_kv = task.num_seq_kv;
      const int num_tile_kv = task.num_tile_kv;
      const int num_tile_full = task.num_tile_full;

      float* lse_batch = lse_ptr +
                         ibatch * max_splitk * num_head_k * lse_pad_heads_per_group * num_seq_q +
                         ichunk * num_head_k * lse_pad_heads_per_group * num_seq_q +
                         ihead_kv * lse_pad_heads_per_group * num_seq_q;

      clear(gSum);
      fill(gMax, -std::numeric_limits<float>::infinity());
      fill(gSoftmaxScale, one_over_dk_log2e);

      clear(tYr);
      tiled_mma_sv.accumulate_ = GMMA::ScaleOut::One;

      wait_barrier(q_readable, phase_q);
      phase_q ^= 1;

#pragma unroll 1
      for (int itile_seq_kv = num_tile_full; itile_seq_kv < num_tile_kv; ++itile_seq_kv) {
        wait_barrier(k_readable[istage_kv], phase_kv);

        qk_gemm(tiled_mma_qk, tQr, tKr, tAttr, istage_kv);

        if (is_leader) {
          arrive_barrier(k_writable[istage_kv]);
        }

        apply_casual_mask<kTileN, kHeadsPerGroup>(tAttr_nm, tI_nm, itile_seq_kv, num_seq_kvcache,
                                                  num_seq_kv);

        online_softmax<true, kTileM>(tAttr_nm, gMax, gSum, tYr_nm, gSoftmaxScale, shm_max,
                                     kIWarpgroup, iwarp, ilane);

        cast_fp32reg<Tin>(tAttr, tAttAbf16);
        cute::copy(tiled_copy_P_r2s, tPr4s, tPs4r);

        wait_barrier(v_readable[istage_kv], phase_kv);
        cutlass::arch::fence_view_async_shared();
        syncwarpgroup(kIWarpgroup);

        sv_gemm(tiled_mma_sv, tSr, tVr, tYr, istage_kv);

        if (is_leader) {
          arrive_barrier(v_writable[istage_kv]);
        }

        advance_stage<kStage>(istage_kv, phase_kv);
      }

#pragma unroll 1
      for (int itile_seq_kv = 0; itile_seq_kv < num_tile_full; ++itile_seq_kv) {
        wait_barrier(k_readable[istage_kv], phase_kv);

        qk_gemm(tiled_mma_qk, tQr, tKr, tAttr, istage_kv);

        if (is_leader) {
          arrive_barrier(k_writable[istage_kv]);
        }

        online_softmax<false, kTileM>(tAttr_nm, gMax, gSum, tYr_nm, gSoftmaxScale, shm_max,
                                      kIWarpgroup, iwarp, ilane);

        cast_fp32reg<Tin>(tAttr, tAttAbf16);
        cute::copy(tiled_copy_P_r2s, tPr4s, tPs4r);

        wait_barrier(v_readable[istage_kv], phase_kv);
        cutlass::arch::fence_view_async_shared();
        syncwarpgroup(kIWarpgroup);

        sv_gemm(tiled_mma_sv, tSr, tVr, tYr, istage_kv);

        if (is_leader) {
          arrive_barrier(v_writable[istage_kv]);
        }

        advance_stage<kStage>(istage_kv, phase_kv);
      }

      final_online_softmax<kTileM>(tYr_nm, gSum, shm_max, kIWarpgroup, iwarp, ilane);

      store_output<true, 1>(tiled_copy_SplitY_r2s, tma_splity, tYr, sSplitY, gSplitY, ihead_kv,
                            ibatch, ichunk, num_seq_q, idx, kIWarpgroup, is_leader);
      store_lse(lse_batch, gMax, gSum, heads_per_group, ilane, iwarp);

      tma_store_wait<0>();

      if (is_leader) {
        arrive_barrier(q_writable);
      }

      block_task_ptr += kTaskStride;
    }
  }

  cudaTriggerProgrammaticLaunchCompletion();
}

}  // namespace kernels
}  // namespace decode
}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_DECODE_SM90_DYNAMIC_SMALLM_BF16_DIM128_DYNAMIC_SPLITK_KERNELS_CUH_
