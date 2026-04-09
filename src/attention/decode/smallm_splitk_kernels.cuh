// Copyright (C) 2026 Tencent.
#ifndef SRC_ATTENTION_DECODE_SMALLM_SPLITK_KERNELS_CUH_
#define SRC_ATTENTION_DECODE_SMALLM_SPLITK_KERNELS_CUH_

#include <cuda.h>
#include <stdio.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "src/attention/decode/util_kernels.cuh"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace attention {
namespace kernels {

template <typename Tout, typename Tin, int kTileM, int kTileN, int kTileK, int kTileV,
          int kHeadsPerGroup, typename TiledMmaQK, typename TiledMmaSV, typename TmaQ,
          typename TmaK, typename TmaV, typename TmaY, typename TmaSplitY, typename SLayoutQ,
          typename SLayoutK, typename SLayoutP, typename SLayoutS, typename SLayoutV,
          typename SLayoutY, typename SLayoutSplitY, int kBlockSize, int kStage, int kSplitK,
          int kSplitMinLen>
__global__ void attention_decode_bf16_multistage_ws_smallm_splitk_kernel(
    const __grid_constant__ TmaQ tma_q, const __grid_constant__ TmaK tma_k,
    const __grid_constant__ TmaV tma_v, const __grid_constant__ TmaY tma_y,
    const __grid_constant__ TmaSplitY tma_splity, Tout* y_ptr, float* split_y_ptr, float* lse_ptr,
    const int* block_ids_ptr, const int* num_seq_kvcache_ptr, int* split_flag_ptr,
    bool new_kv_included, int num_batch, int num_seq_q, int num_dim_qk, int num_dim_v,
    int num_head_q, int num_head_k, int num_head_v, int heads_per_group,
    int lse_pad_heads_per_group, int num_kvcache_blocks, int num_seq_max_blocks,
    float one_over_dk_log2e) {
  using namespace cute;  // NOLINT

  int idx = threadIdx.x;
  int ihead_kv = blockIdx.x;
  int ibatch = blockIdx.y;
  int ichunk = blockIdx.z;

  constexpr int kMathThreads = size(TiledMmaQK{});
  constexpr int kMathWarps = kMathThreads / 32;
  constexpr int kWarpsPerWrapGroup = 4;

  int elected = cute::elect_one_sync();
  int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
  bool is_leader_in_block = (iwarp == 0) && elected;

  int num_seq_kvcache, num_seq_kv, num_blocks, num_blocks_per_chunk, num_chunks;
  int num_tile_kv, num_tile_full, num_tile_causal;
  bool is_split, is_last_chunk;

  if (!get_task<kTileN, kBlockSize, kSplitK, kSplitMinLen>(
          num_seq_kvcache_ptr, new_kv_included, num_seq_q, ibatch, ichunk, num_seq_kvcache,
          num_seq_kv, num_chunks, is_split, is_last_chunk, num_blocks, num_blocks_per_chunk,
          num_tile_kv, num_tile_full, num_tile_causal)) {
    return;
  }

  float* lse_batch = lse_ptr + ibatch * kSplitK * num_head_k * lse_pad_heads_per_group * num_seq_q +
                     ichunk * num_head_k * lse_pad_heads_per_group * num_seq_q +
                     ihead_kv * lse_pad_heads_per_group * num_seq_q;

  const int* block_ids =
      block_ids_ptr + ibatch * num_seq_max_blocks + ichunk * num_blocks_per_chunk;

  __shared__ uint64_t q_readable;
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
  int* shm_kvblk_ids = reinterpret_cast<int*>(shm_max + kTileM * kWarpsPerWrapGroup);
  auto* shm_y = reinterpret_cast<Tout*>(shm_data);        // Reuse All
  auto* shm_splity = reinterpret_cast<float*>(shm_data);  // Reuse All

  // Tensor Q/K/V/Y
  auto gQ = tma_q.get_tma_tensor(
      make_shape(heads_per_group, num_dim_qk, num_head_k, num_seq_q, num_batch));
  auto gK =
      tma_k.get_tma_tensor(make_shape(kBlockSize, num_dim_qk, num_head_k, num_kvcache_blocks));
  auto gV = tma_v.get_tma_tensor(make_shape(num_dim_v, kBlockSize, num_head_v, num_kvcache_blocks));
  auto gY = tma_y.get_tma_tensor(
      make_shape(num_dim_v, heads_per_group, num_head_k, num_seq_q, num_batch));
  auto gSplitY = tma_splity.get_tma_tensor(
      make_shape(num_dim_v, heads_per_group, num_head_k, num_seq_q, kSplitK, num_batch));

  auto gAtt =
      make_tensor(make_gmem_ptr(static_cast<float*>(nullptr)),
                  make_shape(Int<kTileN>{}, Int<kTileM>{}), make_stride(Int<kTileM>{}, Int<1>{}));
  auto gYY =
      make_tensor(make_gmem_ptr(static_cast<float*>(nullptr)),
                  make_shape(Int<kTileV>{}, Int<kTileM>{}), make_stride(Int<1>{}, Int<kTileV>{}));

  // Tensor sQ/sK/sV
  auto sQ = make_tensor(make_smem_ptr(shm_q), SLayoutQ{});
  auto sK = make_tensor(make_smem_ptr(shm_k), SLayoutK{});
  auto sP = make_tensor(make_smem_ptr(shm_p), SLayoutP{});
  auto sS = make_tensor(make_smem_ptr(shm_p), SLayoutS{});
  auto sV = make_tensor(make_smem_ptr(shm_v), SLayoutV{});
  auto sY = make_tensor(make_smem_ptr(shm_y), SLayoutY{});
  auto sSplitY = make_tensor(make_smem_ptr(shm_splity), SLayoutSplitY{});

  // Block Level tma
  auto btma_q = tma_q.get_slice(0);
  auto btma_k = tma_k.get_slice(0);
  auto btma_v = tma_v.get_slice(0);

  // Thread Level Tensor
  auto tQg = btma_q.partition_S(gQ);  // (TMA, TMA_M, TMA_K, seqlenq, head_kv, batch)
  auto tKg = btma_k.partition_S(gK);  // (TMA, TMA_N, TMA_K, head_kv, batch)
  auto tVg = btma_v.partition_S(gV);  // (TMA, TMA_V, TMA_N, head_kv, batch)

  auto tQs = btma_q.partition_D(sQ);  // (TMA, _1, _1)
  auto tKs = btma_k.partition_D(sK);  // (TMA, _1, _1)
  auto tVs = btma_v.partition_D(sV);  // (TMA, _1, _1)

  // init bar
  if (is_leader_in_block) {
    initialize_barrier(q_readable, 1);
#pragma unroll
    for (int istage = 0; istage < kStage; istage++) {
      initialize_barrier(k_writable[istage], 1);
      initialize_barrier(v_writable[istage], 1);
      initialize_barrier(k_readable[istage], 1);
      initialize_barrier(v_readable[istage], 1);
    }
  }

  // sync to avoid ahead thread use(wait) readable when it is not initizlized yet
  __syncthreads();

  // load warpgroup
  if (idx >= kMathThreads) {
    // cutlass::arch::warpgroup_reg_dealloc<24>();
    bool is_leader_in_load = ((iwarp == kMathThreads / 32) && elected);

    if (is_leader_in_load) {
      // Load Q
      for (int iseqq = 0; iseqq < num_seq_q; iseqq++) {
        cute::copy(tma_q.with(q_readable), tQg(_, 0, _, ihead_kv, iseqq, ibatch), tQs(_, iseqq, _));
      }
      set_barrier_transaction_bytes(q_readable, sizeof(Tin) * cosize(SLayoutQ{}));
    }
  }

  // Load BlockIds
  for (int i = idx; i < num_blocks; i += blockDim.x) {
    shm_kvblk_ids[i] = block_ids[i];
  }
  __syncthreads();

  if (idx >= kMathThreads) {
    idx -= kMathThreads;
    iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);

    constexpr int kBlockPerTileN = kTileN / kBlockSize;

    bool is_leader_in_load = ((iwarp == 0) && elected);
    int phase = 1;
    int iload_tile = 0;

    if (is_leader_in_load) {
      int istage_write = 0;
      // Load Causal KV
#pragma unroll 1
      for (int itile_seq_kv = num_tile_full; itile_seq_kv < num_tile_kv; ++itile_seq_kv) {
        // load k/scale/v
        load_paged_kv<true, kBlockPerTileN, kBlockSize, kStage, Tin>(
            tma_k, tma_v, k_writable, v_writable, k_readable, v_readable, tKg, tKs, tVg, tVs,
            ihead_kv, num_dim_qk, num_dim_v, shm_kvblk_ids, num_blocks, itile_seq_kv, istage_write,
            phase);
        advance_stage<kStage>(istage_write, phase);
      }

      // Load Full KV
#pragma unroll 1
      for (int itile_seq_kv = -kStage + 1; itile_seq_kv < num_tile_full; ++itile_seq_kv) {
        if (iload_tile < num_tile_full) {
          load_paged_kv<false, kBlockPerTileN, kBlockSize, kStage, Tin>(
              tma_k, tma_v, k_writable, v_writable, k_readable, v_readable, tKg, tKs, tVg, tVs,
              ihead_kv, num_dim_qk, num_dim_v, shm_kvblk_ids, num_blocks, iload_tile++,
              istage_write, phase);
          advance_stage<kStage>(istage_write, phase);
        }
      }
    }
  } else {
    // cutlass::arch::warpgroup_reg_alloc<232>();
    // math warpgroup
    int idx_in_warpgroup = idx % 128;
    int iwarpgroup = idx / 128;
    int iwarp_in_warpgroup = idx_in_warpgroup / 32;
    int ilane_in_warpgroup = idx_in_warpgroup % 32;
    int elected_idx_in_warpgroup = ((iwarp_in_warpgroup == 0) && elected);
    bool is_leader_in_warpgroup = ((iwarp % 4) == 0) && elected;

    TiledMmaQK tiled_mma_qk;
    TiledMmaSV tiled_mma_sv;

    auto thr_mma_qk = tiled_mma_qk.get_slice(idx);
    auto thr_mma_sv = tiled_mma_sv.get_slice(idx);

    auto tKs4r = thr_mma_qk.partition_A(sK);
    auto tQs4r = thr_mma_qk.partition_B(sQ);
    auto tVs4r = thr_mma_sv.partition_A(sV);
    auto tSs4r = thr_mma_sv.partition_B(sS);

    auto tKr = thr_mma_qk.make_fragment_A(tKs4r);  // (MMA, MMA_N, MMA_K)
    auto tQr = thr_mma_qk.make_fragment_B(tQs4r);  // (MMA, MMA_M, MMA_K)
    auto tVr = thr_mma_sv.make_fragment_A(tVs4r);  // (MMA, MMA_V, MMA_N)
    auto tSr = thr_mma_sv.make_fragment_B(tSs4r);  // (MMA, MMA_V, MMA_N)

    auto tAttr = thr_mma_qk.partition_fragment_C(gAtt);
    auto tAttAbf16 = make_tensor_like<cute::bfloat16_t>(tAttr);
    auto tYr = thr_mma_sv.partition_fragment_C(gYY);

    auto gI = make_identity_tensor(gAtt.shape());
    auto tI = thr_mma_qk.partition_C(gI);

    auto tAttr_nm = retile_fragment(tAttr);
    auto tI_nm = retile_fragment(tI);
    auto tYr_nm = retile_fragment(tYr);

    constexpr int kN = size<0>(tAttr_nm);
    constexpr int kM = size<1>(tAttr_nm);
    Tensor gMax = make_tensor<float>(Int<kM>{});
    Tensor gSum = make_tensor<float>(Int<kM>{});
    Tensor gSoftmaxScale = make_tensor<float>(Int<kM>{});

    clear(gSum);
    fill(gMax, -std::numeric_limits<float>::infinity());
    fill(gSoftmaxScale, one_over_dk_log2e);

    using STSM_ATOM =
        std::conditional_t<kTileM % 16 == 0, cute::SM90_U16x8_STSM_T, cute::SM90_U16x4_STSM_T>;
    using R2SCopyAtomP = Copy_Atom<STSM_ATOM, Tin>;
    auto tiled_copy_P_r2s = make_tiled_copy_C(R2SCopyAtomP{}, tiled_mma_qk);
    auto thr_copy_P_r2s = tiled_copy_P_r2s.get_slice(idx);
    auto tPr4s = thr_copy_P_r2s.retile_S(tAttAbf16);
    auto tPs4r = thr_copy_P_r2s.partition_D(sP);

    using R2SCopyAtomY = Copy_Atom<STSM_ATOM, Tout>;
    auto tiled_copy_Y_r2s = make_tiled_copy_C(R2SCopyAtomY{}, tiled_mma_sv);

    using R2SCopyAtomSplitY = Copy_Atom<UniversalCopy<int>, float>;
    auto tiled_copy_SplitY_r2s = make_tiled_copy_C(R2SCopyAtomSplitY{}, tiled_mma_sv);

    clear(tYr);

    tiled_mma_sv.accumulate_ = GMMA::ScaleOut::One;

    wait_barrier(q_readable, 0);

    int phase = 0;
    int istage_read = 0;
    // compute casual
#pragma unroll 1
    for (int itile_seq_kv = num_tile_full; itile_seq_kv < num_tile_kv; ++itile_seq_kv) {
      wait_barrier(k_readable[istage_read], phase);

      // P = QK
      qk_gemm(tiled_mma_qk, tQr, tKr, tAttr, istage_read);

      if (elected_idx_in_warpgroup) {
        arrive_barrier(k_writable[istage_read]);
      }

      // do causal mask
      apply_casual_mask<kTileN, kHeadsPerGroup>(tAttr_nm, tI_nm, itile_seq_kv, num_seq_kvcache,
                                                num_seq_kv);

      // online softmax
      online_softmax<true, kTileM>(tAttr_nm, gMax, gSum, tYr_nm, gSoftmaxScale, shm_max, iwarpgroup,
                                   iwarp_in_warpgroup, ilane_in_warpgroup);

      // tAttfp32 => tAttbf16
      cast_fp32reg<Tin>(tAttr, tAttAbf16);

      // P reg to smem
      cute::copy(tiled_copy_P_r2s, tPr4s, tPs4r);

      wait_barrier(v_readable[istage_read], phase);
      cutlass::arch::fence_view_async_shared();
      syncwarpgroup(iwarpgroup);

      // Y = PV
      sv_gemm(tiled_mma_sv, tSr, tVr, tYr, istage_read);

      if (elected_idx_in_warpgroup) {
        arrive_barrier(v_writable[istage_read]);
      }

      advance_stage<kStage>(istage_read, phase);
    }

    // compute full
#pragma unroll 1
    for (int itile_seq_kv = 0; itile_seq_kv < num_tile_full; ++itile_seq_kv) {
      wait_barrier(k_readable[istage_read], phase);

      // P = QK
      qk_gemm(tiled_mma_qk, tQr, tKr, tAttr, istage_read);

      if (elected_idx_in_warpgroup) {
        arrive_barrier(k_writable[istage_read]);
      }

      // online softmax
      online_softmax<false, kTileM>(tAttr_nm, gMax, gSum, tYr_nm, gSoftmaxScale, shm_max,
                                    iwarpgroup, iwarp_in_warpgroup, ilane_in_warpgroup);

      // tAttfp32 => tAttbf16
      cast_fp32reg<Tin>(tAttr, tAttAbf16);

      // P reg to smem
      cute::copy(tiled_copy_P_r2s, tPr4s, tPs4r);

      wait_barrier(v_readable[istage_read], phase);
      cutlass::arch::fence_view_async_shared();
      syncwarpgroup(iwarpgroup);

      // Y = PV
      sv_gemm(tiled_mma_sv, tSr, tVr, tYr, istage_read);

      if (elected_idx_in_warpgroup) {
        arrive_barrier(v_writable[istage_read]);
      }

      advance_stage<kStage>(istage_read, phase);
    }

    // final online softmax
    final_online_softmax<kTileM>(tYr_nm, gSum, shm_max, iwarpgroup, iwarp_in_warpgroup,
                                 ilane_in_warpgroup);

    // Epilogue: write register-C to global memory
    if (!is_split) {
      auto tYr_bf16 = make_tensor_like<Tout>(tYr);
      // to bfloat16
      cast_fp32reg<Tout>(tYr, tYr_bf16);

      store_output<false, 1>(tiled_copy_Y_r2s, tma_y, tYr_bf16, sY, gY, ihead_kv, ibatch, 0,
                             num_seq_q, idx, iwarpgroup, is_leader_in_warpgroup);
    } else {
      store_output<true, 1>(tiled_copy_SplitY_r2s, tma_splity, tYr, sSplitY, gSplitY, ihead_kv,
                            ibatch, ichunk, num_seq_q, idx, iwarpgroup, is_leader_in_warpgroup);

      int ilane = idx % 32;
      store_lse(lse_batch, gMax, gSum, heads_per_group, ilane, iwarp);

      auto* split_flag = split_flag_ptr + ibatch * num_head_k + ihead_kv;

      tma_store_wait<0>();
      __threadfence();
      syncwarpgroup(iwarpgroup);
      if (idx == 0) {
        atomicAdd(split_flag, 1);
      }

      if (is_last_chunk) {
        while (load_global_volatile(split_flag) != (ichunk + 1)) {
        }
        splitk_reduce<__nv_bfloat16, kTileV, kSplitK, kMathWarps>(
            y_ptr, lse_ptr, split_y_ptr, num_chunks, num_seq_q, num_head_q, num_head_k,
            heads_per_group, lse_pad_heads_per_group, ihead_kv, ibatch, iwarp, ilane);
      }
    }
  }
}

}  // namespace kernels
}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_DECODE_SMALLM_SPLITK_KERNELS_CUH_
