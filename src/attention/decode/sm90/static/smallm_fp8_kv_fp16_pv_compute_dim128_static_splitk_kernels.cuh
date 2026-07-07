// Copyright 2025 hpc-ops authors

#ifndef SRC_ATTENTION_DECODE_SM90_STATIC_SMALLM_FP8_KV_FP16_PV_COMPUTE_DIM128_STATIC_SPLITK_KERNELS_CUH_
#define SRC_ATTENTION_DECODE_SM90_STATIC_SMALLM_FP8_KV_FP16_PV_COMPUTE_DIM128_STATIC_SPLITK_KERNELS_CUH_

#include <cuda.h>
#include <stdio.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "src/attention/decode/sm90/util_kernels.cuh"
#include "src/attention/prefill/fp8_v_to_half_dequant.cuh"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace attention {
namespace decode {
namespace kernels {

// Hybrid a8c8-fp16pv decode: Q/K/V all fp8. Q@Kᵀ runs as FP8 WGMMA (K read
// directly from its fp8 TMA stage — no dequant), P@V runs as FP16 WGMMA. This is
// the a16c8 decode kernel with the QK half swapped to fp8 + a per-(token,head)
// qscale folded into the per-row softmax scale (pure-fp8 style). V is still
// dequantized fp8->fp16 in-place and consumed by the F32F16F16 SV MMA.
//
// kDecodePScale scales P before the fp16 cast; epilogue folds the reciprocal
// into vscale.
constexpr float kFp16PvDecodePScale = 256.0f;
constexpr float kFp16PvDecodePScaleInv = 1.0f / kFp16PvDecodePScale;

// =============================================================================
// Mode 20 decode kernel: Q fp8 per-(token,head) + K per-(K-tok, head) + V
// per-head. qscale per-row folds into gSoftmaxScale; kscale per-column applied
// on the fp32 S accumulator; V per-head scale post-muls tYr.
// =============================================================================
template <typename Tout, typename Tin, int kTileM, int kTileN, int kTileK, int kTileV,
          int kHeadsPerGroup, typename TiledMmaQK, typename TiledMmaSV, typename TmaQ,
          typename TmaKFp8, typename TmaVFp8, typename TmaY, typename TmaSplitY, typename TmaKS,
          typename SLayoutQ, typename SLayoutK, typename SLayoutP, typename SLayoutS,
          typename SLayoutV, typename SLayoutVFp8, typename SLayoutKSC, typename SLayoutY,
          typename SLayoutSplitY, int kBlockSize, int kStage, int kSplitK, int kSplitMinLen>
__global__ void
attention_decode_qfp8_kpertoken_perhead_vperhead_fp16_pv_compute_multistage_ws_smallm_splitk_kernel(
    const __grid_constant__ TmaQ tma_q, const __grid_constant__ TmaKFp8 tma_k,
    const __grid_constant__ TmaVFp8 tma_v, const __grid_constant__ TmaY tma_y,
    const __grid_constant__ TmaSplitY tma_splity, const __grid_constant__ TmaKS tma_ks, Tout* y_ptr,
    float* split_y_ptr, float* lse_ptr, const int* block_ids_ptr, const int* num_seq_kvcache_ptr,
    const float* qscale_ptr, const float* kscale_ptr, const float* vscale_ptr, int* split_flag_ptr,
    bool new_kv_included, int num_batch, int num_seq_q, int num_dim_qk, int num_dim_v,
    int num_head_q, int num_head_k, int num_head_v, int heads_per_group,
    int lse_pad_heads_per_group, int num_kvcache_blocks, int num_seq_max_blocks,
    float one_over_dk_log2e) {
  using namespace cute;  // NOLINT

  // V dequantized to fp16, P cast to fp16; PV WGMMA runs F32F16F16.
  using TinKV = cute::float_e4m3_t;
  using TinPV = cute::half_t;

  constexpr int kTileScale = kTileK / 4;  // sizeof(float) == 4
  (void)kscale_ptr;

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

  float vscale_perhead = vscale_ptr[ihead_kv] * kFp16PvDecodePScaleInv;
  float* lse_batch = lse_ptr + ibatch * kSplitK * num_head_k * lse_pad_heads_per_group * num_seq_q +
                     ichunk * num_head_k * lse_pad_heads_per_group * num_seq_q +
                     ihead_kv * lse_pad_heads_per_group * num_seq_q;

  const int* block_ids =
      block_ids_ptr + ibatch * num_seq_max_blocks + ichunk * num_blocks_per_chunk;

  // qscale: per-(token, head), laid out [num_batch, num_seq_q, num_head_q].
  const float* qscales_batch =
      qscale_ptr + ibatch * num_head_q * num_seq_q + ihead_kv * heads_per_group;

  __shared__ uint64_t q_readable;
  __shared__ uint64_t k_writable[kStage];
  __shared__ uint64_t v_writable[kStage];
  __shared__ uint64_t k_readable[kStage];
  __shared__ uint64_t v_readable[kStage];
  extern __shared__ uint8_t shm_data[] alignas(128);

  // Layout: fp8 Q | fp8 K stage (WGMMA operand) | fp16 V | fp16 P | shm_max |
  //         fp8 V stage | fp32 KS stage | shm_kvblk_ids
  auto* shm_q = reinterpret_cast<Tin*>(shm_data);
  auto* shm_k = shm_q + cosize(SLayoutQ{});
  auto* shm_v = reinterpret_cast<TinPV*>(shm_k + cosize(SLayoutK{}));
  auto* shm_p = shm_v + cosize(SLayoutV{});
  auto* shm_max = reinterpret_cast<float*>(shm_p + cosize(SLayoutP{}));
  auto* shm_v_fp8 = reinterpret_cast<TinKV*>(shm_max + kTileM * kWarpsPerWrapGroup);
  auto* shm_ks = reinterpret_cast<float*>(shm_v_fp8 + cosize(SLayoutVFp8{}));
  int* shm_kvblk_ids = reinterpret_cast<int*>(shm_ks + cosize(SLayoutKSC{}));
  auto* shm_y = reinterpret_cast<Tout*>(shm_data);
  auto* shm_splity = reinterpret_cast<float*>(shm_data);

  auto gQ = tma_q.get_tma_tensor(
      make_shape(heads_per_group, num_dim_qk, num_head_k, num_seq_q, num_batch));
  auto gK =
      tma_k.get_tma_tensor(make_shape(kBlockSize, num_dim_qk, num_head_k, num_kvcache_blocks));
  auto gV = tma_v.get_tma_tensor(make_shape(num_dim_v, kBlockSize, num_head_v, num_kvcache_blocks));
  auto gKS = tma_ks.get_tma_tensor(make_shape(Int<kBlockSize / kTileScale>{}, Int<kTileScale>{},
                                              num_head_k, num_kvcache_blocks));
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

  auto sQ = make_tensor(make_smem_ptr(shm_q), SLayoutQ{});
  auto sK = make_tensor(make_smem_ptr(shm_k), SLayoutK{});  // fp8
  auto sP = make_tensor(make_smem_ptr(shm_p), SLayoutP{});
  auto sS = make_tensor(make_smem_ptr(shm_p), SLayoutS{});
  auto sV = make_tensor(make_smem_ptr(shm_v), SLayoutV{});
  auto sY = make_tensor(make_smem_ptr(shm_y), SLayoutY{});
  auto sSplitY = make_tensor(make_smem_ptr(shm_splity), SLayoutSplitY{});

  auto sV_fp8 = make_tensor(make_smem_ptr(shm_v_fp8), SLayoutVFp8{});
  auto sKS_C = make_tensor(make_smem_ptr(shm_ks), SLayoutKSC{});
  auto sKS_flat = make_tensor(sKS_C.data(), make_layout(make_shape(Int<kTileN>{}, Int<kStage>{})));

  auto btma_q = tma_q.get_slice(0);
  auto btma_k = tma_k.get_slice(0);
  auto btma_v = tma_v.get_slice(0);
  auto btma_ks = tma_ks.get_slice(0);

  auto tQg = btma_q.partition_S(gQ);
  auto tKg = btma_k.partition_S(gK);
  auto tVg = btma_v.partition_S(gV);
  auto tKSg = btma_ks.partition_S(gKS);

  auto tQs = btma_q.partition_D(sQ);
  auto tKs = btma_k.partition_D(sK);  // TMA -> fp8 K (also WGMMA operand)
  auto tVs = btma_v.partition_D(sV_fp8);
  auto tKSs = btma_ks.partition_D(sKS_C);

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

  __syncthreads();

  if (idx >= kMathThreads) {
    bool is_leader_in_load = ((iwarp == kMathThreads / 32) && elected);
    if (is_leader_in_load) {
      for (int iseqq = 0; iseqq < num_seq_q; iseqq++) {
        cute::copy(tma_q.with(q_readable), tQg(_, 0, _, ihead_kv, iseqq, ibatch), tQs(_, iseqq, _));
      }
      set_barrier_transaction_bytes(q_readable, sizeof(Tin) * cosize(SLayoutQ{}));
    }
  }

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
      // K stage carries fp8 K + fp32 KS payloads jointly.
      constexpr int kTransactionBytesK = sizeof(TinKV) * kBlockPerTileN * kBlockSize * kTileK +
                                         sizeof(float) * kBlockPerTileN * kBlockSize;
      constexpr int kTransactionBytesV = sizeof(TinKV) * kBlockPerTileN * kBlockSize * kTileV;

      auto load_one_tile = [&](bool check_bound, int itile, int istage) {
        wait_barrier(k_writable[istage], phase);
#pragma unroll
        for (int ikvblock = 0; ikvblock < kBlockPerTileN; ikvblock++) {
          int kvblk_id = itile * kBlockPerTileN + ikvblock;
          int blk_id = -1;
          if (check_bound) {
            if (kvblk_id < num_blocks)
              blk_id = shm_kvblk_ids[kvblk_id];
          } else {
            blk_id = shm_kvblk_ids[kvblk_id];
          }
          cute::copy(tma_k.with(k_readable[istage]), tKg(_, 0, _, ihead_kv, blk_id),
                     tKs(_, ikvblock, _, istage));
          cute::copy(tma_ks.with(k_readable[istage]), tKSg(_, 0, _, ihead_kv, blk_id),
                     tKSs(_, ikvblock, _, istage));
        }
        set_barrier_transaction_bytes(k_readable[istage], kTransactionBytesK);

        wait_barrier(v_writable[istage], phase);
#pragma unroll
        for (int ikvblock = 0; ikvblock < kBlockPerTileN; ikvblock++) {
          int kvblk_id = itile * kBlockPerTileN + ikvblock;
          int blk_id = -1;
          if (check_bound) {
            if (kvblk_id < num_blocks)
              blk_id = shm_kvblk_ids[kvblk_id];
          } else {
            blk_id = shm_kvblk_ids[kvblk_id];
          }
          cute::copy(tma_v.with(v_readable[istage]), tVg(_, _, 0, ihead_kv, blk_id),
                     tVs(_, _, ikvblock, istage));
        }
        set_barrier_transaction_bytes(v_readable[istage], kTransactionBytesV);
      };

#pragma unroll 1
      for (int itile_seq_kv = num_tile_full; itile_seq_kv < num_tile_kv; ++itile_seq_kv) {
        load_one_tile(true, itile_seq_kv, istage_write);
        advance_stage<kStage>(istage_write, phase);
      }
#pragma unroll 1
      for (int itile_seq_kv = -kStage + 1; itile_seq_kv < num_tile_full; ++itile_seq_kv) {
        if (iload_tile < num_tile_full) {
          load_one_tile(false, iload_tile++, istage_write);
          advance_stage<kStage>(istage_write, phase);
        }
      }
    }
  } else {
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

    auto tKs4r = thr_mma_qk.partition_A(sK);  // fp8 K operand
    auto tQs4r = thr_mma_qk.partition_B(sQ);  // fp8 Q operand
    auto tVs4r = thr_mma_sv.partition_A(sV);
    auto tSs4r = thr_mma_sv.partition_B(sS);

    auto tKr = thr_mma_qk.make_fragment_A(tKs4r);
    auto tQr = thr_mma_qk.make_fragment_B(tQs4r);
    auto tVr = thr_mma_sv.make_fragment_A(tVs4r);
    auto tSr = thr_mma_sv.make_fragment_B(tSs4r);

    auto tAttr = thr_mma_qk.partition_fragment_C(gAtt);
    auto tAttApv = make_tensor_like<TinPV>(tAttr);
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
    Tensor kscales = make_tensor<float>(Int<kN>{});

    // qscale per-row folds into the softmax scale (× one_over_dk_log2e).
#pragma unroll
    for (int i = 0; i < kM; i++) {
      int im = get<1>(tI_nm(0, i));
      int iseqq = im / kHeadsPerGroup;
      int iqhead = im % kHeadsPerGroup;
      float qs = (iqhead < heads_per_group) ? qscales_batch[iseqq * num_head_q + iqhead] : 1.f;
      gSoftmaxScale(i) = one_over_dk_log2e * qs;
    }

    clear(gSum);
    fill(gMax, -std::numeric_limits<float>::infinity());

    using STSM_ATOM =
        std::conditional_t<kTileM % 16 == 0, cute::SM90_U16x8_STSM_T, cute::SM90_U16x4_STSM_T>;
    using R2SCopyAtomP = Copy_Atom<STSM_ATOM, TinPV>;
    auto tiled_copy_P_r2s = make_tiled_copy_C(R2SCopyAtomP{}, tiled_mma_qk);
    auto thr_copy_P_r2s = tiled_copy_P_r2s.get_slice(idx);
    auto tPr4s = thr_copy_P_r2s.retile_S(tAttApv);
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
#pragma unroll 1
    for (int itile_seq_kv = num_tile_full; itile_seq_kv < num_tile_kv; ++itile_seq_kv) {
      wait_barrier(k_readable[istage_read], phase);

#pragma unroll
      for (int in = 0; in < kN; in++) {
        kscales(in) = sKS_flat(get<0>(tI_nm(in, 0)), istage_read);
      }

      qk_gemm(tiled_mma_qk, tQr, tKr, tAttr, istage_read);

      if (elected_idx_in_warpgroup) {
        arrive_barrier(k_writable[istage_read]);
      }

      apply_casual_mask_with_kscale<kTileN, kHeadsPerGroup>(tAttr_nm, tI_nm, kscales, itile_seq_kv,
                                                            num_seq_kvcache, num_seq_kv);

      online_softmax<true, kTileM>(tAttr_nm, gMax, gSum, tYr_nm, gSoftmaxScale, shm_max, iwarpgroup,
                                   iwarp_in_warpgroup, ilane_in_warpgroup);

#pragma unroll
      for (int i = 0; i < size(tAttr); ++i) {
        tAttr(i) *= kFp16PvDecodePScale;
      }
      cast_fp32reg<TinPV>(tAttr, tAttApv);

      cute::copy(tiled_copy_P_r2s, tPr4s, tPs4r);

      wait_barrier(v_readable[istage_read], phase);

      {
        auto sVfp8_stage = sV_fp8(_, _, istage_read);
        auto sV_stage = sV(_, _, istage_read);
        hpc::attention::prefill::fp8_smem_to_half_smem_tile_raw_vec16_mn<kTileV, kTileN,
                                                                         /*kThreads=*/kMathThreads>(
            sVfp8_stage, sV_stage, idx_in_warpgroup);
      }

      cutlass::arch::fence_view_async_shared();
      syncwarpgroup(iwarpgroup);

      sv_gemm(tiled_mma_sv, tSr, tVr, tYr, istage_read);

      if (elected_idx_in_warpgroup) {
        arrive_barrier(v_writable[istage_read]);
      }

      advance_stage<kStage>(istage_read, phase);
    }

#pragma unroll 1
    for (int itile_seq_kv = 0; itile_seq_kv < num_tile_full; ++itile_seq_kv) {
      wait_barrier(k_readable[istage_read], phase);

#pragma unroll
      for (int in = 0; in < kN; in++) {
        kscales(in) = sKS_flat(get<0>(tI_nm(in, 0)), istage_read);
      }

      qk_gemm(tiled_mma_qk, tQr, tKr, tAttr, istage_read);

      if (elected_idx_in_warpgroup) {
        arrive_barrier(k_writable[istage_read]);
      }

      // Full (non-causal) tile: apply per-(K-token) scale on every S column.
#pragma unroll
      for (int im = 0; im < kM; ++im) {
#pragma unroll
        for (int in = 0; in < kN; ++in) {
          tAttr_nm(in, im) *= kscales(in);
        }
      }

      online_softmax<false, kTileM>(tAttr_nm, gMax, gSum, tYr_nm, gSoftmaxScale, shm_max,
                                    iwarpgroup, iwarp_in_warpgroup, ilane_in_warpgroup);

#pragma unroll
      for (int i = 0; i < size(tAttr); ++i) {
        tAttr(i) *= kFp16PvDecodePScale;
      }
      cast_fp32reg<TinPV>(tAttr, tAttApv);

      cute::copy(tiled_copy_P_r2s, tPr4s, tPs4r);

      wait_barrier(v_readable[istage_read], phase);

      {
        auto sVfp8_stage = sV_fp8(_, _, istage_read);
        auto sV_stage = sV(_, _, istage_read);
        hpc::attention::prefill::fp8_smem_to_half_smem_tile_raw_vec16_mn<kTileV, kTileN,
                                                                         /*kThreads=*/kMathThreads>(
            sVfp8_stage, sV_stage, idx_in_warpgroup);
      }

      cutlass::arch::fence_view_async_shared();
      syncwarpgroup(iwarpgroup);

      sv_gemm(tiled_mma_sv, tSr, tVr, tYr, istage_read);

      if (elected_idx_in_warpgroup) {
        arrive_barrier(v_writable[istage_read]);
      }

      advance_stage<kStage>(istage_read, phase);
    }

    final_online_softmax<kTileM>(tYr_nm, gSum, shm_max, iwarpgroup, iwarp_in_warpgroup,
                                 ilane_in_warpgroup);

#pragma unroll
    for (int i = 0; i < size(tYr); ++i) {
      tYr(i) *= vscale_perhead;
    }

    if (!is_split) {
      auto tYr_bf16 = make_tensor_like<Tout>(tYr);
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

// =============================================================================
// Mode 21 decode kernel: Q fp8 per-(token,head), K/V per-tensor fp8.
// kscale (scalar) and qscale (per-row) both fold into gSoftmaxScale; vscale
// post-muls the fp32 PV accumulator.
// =============================================================================
template <typename Tout, typename Tin, int kTileM, int kTileN, int kTileK, int kTileV,
          int kHeadsPerGroup, typename TiledMmaQK, typename TiledMmaSV, typename TmaQ,
          typename TmaKFp8, typename TmaVFp8, typename TmaY, typename TmaSplitY, typename SLayoutQ,
          typename SLayoutK, typename SLayoutP, typename SLayoutS, typename SLayoutV,
          typename SLayoutVFp8, typename SLayoutY, typename SLayoutSplitY, int kBlockSize,
          int kStage, int kSplitK, int kSplitMinLen>
__global__ void
attention_decode_qfp8_kpertensor_vpertensor_fp16_pv_compute_multistage_ws_smallm_splitk_kernel(
    const __grid_constant__ TmaQ tma_q, const __grid_constant__ TmaKFp8 tma_k,
    const __grid_constant__ TmaVFp8 tma_v, const __grid_constant__ TmaY tma_y,
    const __grid_constant__ TmaSplitY tma_splity, Tout* y_ptr, float* split_y_ptr, float* lse_ptr,
    const int* block_ids_ptr, const int* num_seq_kvcache_ptr, const float* qscale_ptr,
    const float* kscale_ptr, const float* vscale_ptr, int* split_flag_ptr, bool new_kv_included,
    int num_batch, int num_seq_q, int num_dim_qk, int num_dim_v, int num_head_q, int num_head_k,
    int num_head_v, int heads_per_group, int lse_pad_heads_per_group, int num_kvcache_blocks,
    int num_seq_max_blocks, float one_over_dk_log2e) {
  using namespace cute;  // NOLINT

  using TinKV = cute::float_e4m3_t;
  using TinPV = cute::half_t;

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

  float kscale_pertensor = kscale_ptr[0];
  float vscale_pertensor = vscale_ptr[0] * kFp16PvDecodePScaleInv;

  float* lse_batch = lse_ptr + ibatch * kSplitK * num_head_k * lse_pad_heads_per_group * num_seq_q +
                     ichunk * num_head_k * lse_pad_heads_per_group * num_seq_q +
                     ihead_kv * lse_pad_heads_per_group * num_seq_q;

  const int* block_ids =
      block_ids_ptr + ibatch * num_seq_max_blocks + ichunk * num_blocks_per_chunk;

  const float* qscales_batch =
      qscale_ptr + ibatch * num_head_q * num_seq_q + ihead_kv * heads_per_group;

  __shared__ uint64_t q_readable;
  __shared__ uint64_t k_writable[kStage];
  __shared__ uint64_t v_writable[kStage];
  __shared__ uint64_t k_readable[kStage];
  __shared__ uint64_t v_readable[kStage];
  extern __shared__ uint8_t shm_data[] alignas(128);

  // Layout: fp8 Q | fp8 K stage | fp16 V | fp16 P | shm_max | fp8 V stage |
  //         shm_kvblk_ids
  auto* shm_q = reinterpret_cast<Tin*>(shm_data);
  auto* shm_k = shm_q + cosize(SLayoutQ{});
  auto* shm_v = reinterpret_cast<TinPV*>(shm_k + cosize(SLayoutK{}));
  auto* shm_p = shm_v + cosize(SLayoutV{});
  auto* shm_max = reinterpret_cast<float*>(shm_p + cosize(SLayoutP{}));
  auto* shm_v_fp8 = reinterpret_cast<TinKV*>(shm_max + kTileM * kWarpsPerWrapGroup);
  int* shm_kvblk_ids = reinterpret_cast<int*>(shm_v_fp8 + cosize(SLayoutVFp8{}));
  auto* shm_y = reinterpret_cast<Tout*>(shm_data);
  auto* shm_splity = reinterpret_cast<float*>(shm_data);

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

  auto sQ = make_tensor(make_smem_ptr(shm_q), SLayoutQ{});
  auto sK = make_tensor(make_smem_ptr(shm_k), SLayoutK{});  // fp8
  auto sP = make_tensor(make_smem_ptr(shm_p), SLayoutP{});
  auto sS = make_tensor(make_smem_ptr(shm_p), SLayoutS{});
  auto sV = make_tensor(make_smem_ptr(shm_v), SLayoutV{});
  auto sY = make_tensor(make_smem_ptr(shm_y), SLayoutY{});
  auto sSplitY = make_tensor(make_smem_ptr(shm_splity), SLayoutSplitY{});

  auto sV_fp8 = make_tensor(make_smem_ptr(shm_v_fp8), SLayoutVFp8{});

  auto btma_q = tma_q.get_slice(0);
  auto btma_k = tma_k.get_slice(0);
  auto btma_v = tma_v.get_slice(0);

  auto tQg = btma_q.partition_S(gQ);
  auto tKg = btma_k.partition_S(gK);
  auto tVg = btma_v.partition_S(gV);

  auto tQs = btma_q.partition_D(sQ);
  auto tKs = btma_k.partition_D(sK);
  auto tVs = btma_v.partition_D(sV_fp8);

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

  __syncthreads();

  if (idx >= kMathThreads) {
    bool is_leader_in_load = ((iwarp == kMathThreads / 32) && elected);
    if (is_leader_in_load) {
      for (int iseqq = 0; iseqq < num_seq_q; iseqq++) {
        cute::copy(tma_q.with(q_readable), tQg(_, 0, _, ihead_kv, iseqq, ibatch), tQs(_, iseqq, _));
      }
      set_barrier_transaction_bytes(q_readable, sizeof(Tin) * cosize(SLayoutQ{}));
    }
  }

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
#pragma unroll 1
      for (int itile_seq_kv = num_tile_full; itile_seq_kv < num_tile_kv; ++itile_seq_kv) {
        load_paged_kv<true, kBlockPerTileN, kBlockSize, kStage, TinKV>(
            tma_k, tma_v, k_writable, v_writable, k_readable, v_readable, tKg, tKs, tVg, tVs,
            ihead_kv, num_dim_qk, num_dim_v, shm_kvblk_ids, num_blocks, itile_seq_kv, istage_write,
            phase);
        advance_stage<kStage>(istage_write, phase);
      }
#pragma unroll 1
      for (int itile_seq_kv = -kStage + 1; itile_seq_kv < num_tile_full; ++itile_seq_kv) {
        if (iload_tile < num_tile_full) {
          load_paged_kv<false, kBlockPerTileN, kBlockSize, kStage, TinKV>(
              tma_k, tma_v, k_writable, v_writable, k_readable, v_readable, tKg, tKs, tVg, tVs,
              ihead_kv, num_dim_qk, num_dim_v, shm_kvblk_ids, num_blocks, iload_tile++,
              istage_write, phase);
          advance_stage<kStage>(istage_write, phase);
        }
      }
    }
  } else {
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

    auto tKr = thr_mma_qk.make_fragment_A(tKs4r);
    auto tQr = thr_mma_qk.make_fragment_B(tQs4r);
    auto tVr = thr_mma_sv.make_fragment_A(tVs4r);
    auto tSr = thr_mma_sv.make_fragment_B(tSs4r);

    auto tAttr = thr_mma_qk.partition_fragment_C(gAtt);
    auto tAttApv = make_tensor_like<TinPV>(tAttr);
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

    // qscale per-row × kscale (scalar) fold into the softmax scale.
#pragma unroll
    for (int i = 0; i < kM; i++) {
      int im = get<1>(tI_nm(0, i));
      int iseqq = im / kHeadsPerGroup;
      int iqhead = im % kHeadsPerGroup;
      float qs = (iqhead < heads_per_group) ? qscales_batch[iseqq * num_head_q + iqhead] : 1.f;
      gSoftmaxScale(i) = one_over_dk_log2e * qs * kscale_pertensor;
    }

    clear(gSum);
    fill(gMax, -std::numeric_limits<float>::infinity());

    using STSM_ATOM =
        std::conditional_t<kTileM % 16 == 0, cute::SM90_U16x8_STSM_T, cute::SM90_U16x4_STSM_T>;
    using R2SCopyAtomP = Copy_Atom<STSM_ATOM, TinPV>;
    auto tiled_copy_P_r2s = make_tiled_copy_C(R2SCopyAtomP{}, tiled_mma_qk);
    auto thr_copy_P_r2s = tiled_copy_P_r2s.get_slice(idx);
    auto tPr4s = thr_copy_P_r2s.retile_S(tAttApv);
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
#pragma unroll 1
    for (int itile_seq_kv = num_tile_full; itile_seq_kv < num_tile_kv; ++itile_seq_kv) {
      wait_barrier(k_readable[istage_read], phase);

      qk_gemm(tiled_mma_qk, tQr, tKr, tAttr, istage_read);

      if (elected_idx_in_warpgroup) {
        arrive_barrier(k_writable[istage_read]);
      }

      apply_casual_mask<kTileN, kHeadsPerGroup>(tAttr_nm, tI_nm, itile_seq_kv, num_seq_kvcache,
                                                num_seq_kv);

      online_softmax<true, kTileM>(tAttr_nm, gMax, gSum, tYr_nm, gSoftmaxScale, shm_max, iwarpgroup,
                                   iwarp_in_warpgroup, ilane_in_warpgroup);

#pragma unroll
      for (int i = 0; i < size(tAttr); ++i) {
        tAttr(i) *= kFp16PvDecodePScale;
      }
      cast_fp32reg<TinPV>(tAttr, tAttApv);

      cute::copy(tiled_copy_P_r2s, tPr4s, tPs4r);

      wait_barrier(v_readable[istage_read], phase);

      {
        auto sVfp8_stage = sV_fp8(_, _, istage_read);
        auto sV_stage = sV(_, _, istage_read);
        hpc::attention::prefill::fp8_smem_to_half_smem_tile_raw_vec16_mn<kTileV, kTileN,
                                                                         /*kThreads=*/kMathThreads>(
            sVfp8_stage, sV_stage, idx_in_warpgroup);
      }

      cutlass::arch::fence_view_async_shared();
      syncwarpgroup(iwarpgroup);

      sv_gemm(tiled_mma_sv, tSr, tVr, tYr, istage_read);

      if (elected_idx_in_warpgroup) {
        arrive_barrier(v_writable[istage_read]);
      }

      advance_stage<kStage>(istage_read, phase);
    }

#pragma unroll 1
    for (int itile_seq_kv = 0; itile_seq_kv < num_tile_full; ++itile_seq_kv) {
      wait_barrier(k_readable[istage_read], phase);

      qk_gemm(tiled_mma_qk, tQr, tKr, tAttr, istage_read);

      if (elected_idx_in_warpgroup) {
        arrive_barrier(k_writable[istage_read]);
      }

      online_softmax<false, kTileM>(tAttr_nm, gMax, gSum, tYr_nm, gSoftmaxScale, shm_max,
                                    iwarpgroup, iwarp_in_warpgroup, ilane_in_warpgroup);

#pragma unroll
      for (int i = 0; i < size(tAttr); ++i) {
        tAttr(i) *= kFp16PvDecodePScale;
      }
      cast_fp32reg<TinPV>(tAttr, tAttApv);

      cute::copy(tiled_copy_P_r2s, tPr4s, tPs4r);

      wait_barrier(v_readable[istage_read], phase);

      {
        auto sVfp8_stage = sV_fp8(_, _, istage_read);
        auto sV_stage = sV(_, _, istage_read);
        hpc::attention::prefill::fp8_smem_to_half_smem_tile_raw_vec16_mn<kTileV, kTileN,
                                                                         /*kThreads=*/kMathThreads>(
            sVfp8_stage, sV_stage, idx_in_warpgroup);
      }

      cutlass::arch::fence_view_async_shared();
      syncwarpgroup(iwarpgroup);

      sv_gemm(tiled_mma_sv, tSr, tVr, tYr, istage_read);

      if (elected_idx_in_warpgroup) {
        arrive_barrier(v_writable[istage_read]);
      }

      advance_stage<kStage>(istage_read, phase);
    }

    final_online_softmax<kTileM>(tYr_nm, gSum, shm_max, iwarpgroup, iwarp_in_warpgroup,
                                 ilane_in_warpgroup);

#pragma unroll
    for (int i = 0; i < size(tYr); ++i) {
      tYr(i) *= vscale_pertensor;
    }

    if (!is_split) {
      auto tYr_bf16 = make_tensor_like<Tout>(tYr);
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
}  // namespace decode
}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_DECODE_SM90_STATIC_SMALLM_FP8_KV_FP16_PV_COMPUTE_DIM128_STATIC_SPLITK_KERNELS_CUH_
