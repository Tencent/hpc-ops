// Copyright 2025 hpc-ops authors

#ifndef SRC_ATTENTION_MLA_SMALLM_SPARSE_MLA_DIM576_PERSISTENT_KERNELS_CUH_
#define SRC_ATTENTION_MLA_SMALLM_SPARSE_MLA_DIM576_PERSISTENT_KERNELS_CUH_

#include <cuda.h>
#include <cuda_bf16.h>
#include <stdio.h>

#include <algorithm>
#include <limits>

#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "src/attention/mla/smallm_mla_dim576_persistent_kernels.cuh"
#include "src/attention/mla/smallm_mla_kernels.cuh"  // online_softmax
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace attention {
namespace kernels {

template <typename Tin, typename Tout, int kTileM, int kTileN, int kTileNope, int kTileRope,
          int kTileV, typename TiledMmaQK, typename TiledMmaSV, typename TmaQNope,
          typename TmaQRope, typename SLayoutQNope, typename SLayoutQRope, typename SLayoutKNope,
          typename SLayoutKRope, typename SLayoutP, typename SLayoutS, typename SLayoutV,
          int kBlockSize, int kStage, bool kUseSink = false, int kNumMathWG = 1>
__global__ void __launch_bounds__(kNumMathWG * 128 + 128, 1)
    attention_sparse_mla_dim576_persistent_kernel(
        const __grid_constant__ TmaQNope tma_q_nope, const __grid_constant__ TmaQRope tma_q_rope,
        const Tin *kvcache_ptr, float *y_partial_ptr, float *lse_ptr, Tout *y_ptr,
        const float *sink_weight_ptr, const int *block_ids_ptr, const int *topk_ids_ptr,
        const int *cu_tasks, const int4 *task_list, const int *cu_splits, int num_batch,
        int total_seq_q, int num_head_q, int qk_dim, int v_dim, int num_kvcache_blocks,
        int num_seq_max_blocks, int num_max_topk, int ldKV, int ldY, float one_over_dk_log2e) {
  using namespace cute;  // NOLINT

  int isM = blockIdx.x;
  int idx = threadIdx.x;

  constexpr int kMathThreads = size(TiledMmaQK{}) * kNumMathWG;
  constexpr int kConsumers = kNumMathWG;
  constexpr int kHeadsTotal = kTileM * kNumMathWG;
  constexpr int kProducerBarrier = 3;  // bar.sync ID for producer-WG sync (math uses 1/2)
  static_assert(size(TiledMmaQK{}) == 128, "expects 1 warpgroup per math WG");
  static_assert(kNumMathWG == 1 || kNumMathWG == 2, "only 1 or 2 math WGs supported");
  static_assert(kTileN == kBlockSize,
                "sparse path assumes kTileN == kBlockSize so each tile has 64 topk rows");

  int elected = cute::elect_one_sync();
  int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
  bool is_leader_in_block = (iwarp == 0) && elected;

  __shared__ uint64_t q_readable;
  __shared__ uint64_t q_writable;
  __shared__ uint64_t kv_writable[kStage];
  __shared__ uint64_t kv_readable[kStage];
  __shared__ int shm_task_start;
  __shared__ int shm_task_end;
  __shared__ int shm_kv_offset[kStage][kTileN];
  __shared__ bool shm_kv_valid[kStage][kTileN];
  extern __shared__ uint8_t shm_data[] alignas(128);

  // sV aliases sK_nope (same trick as the splitk kernel).
  auto *shm_q_nope = reinterpret_cast<Tin *>(shm_data);
  auto *shm_q_rope = shm_q_nope + cosize(SLayoutQNope{});
  auto *shm_k_nope = shm_q_rope + cosize(SLayoutQRope{});
  auto *shm_k_rope = shm_k_nope + cosize(SLayoutKNope{});
  auto *shm_p = shm_k_rope + cosize(SLayoutKRope{});
  auto *shm_max = reinterpret_cast<float *>(shm_p + cosize(SLayoutP{}) * kNumMathWG);

  auto gQ_nope = tma_q_nope.get_tma_tensor(make_shape(num_head_q, Int<kTileNope>{}, total_seq_q));
  auto gQ_rope = tma_q_rope.get_tma_tensor(make_shape(num_head_q, Int<kTileRope>{}, total_seq_q));

  auto gAtt =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileN>{}, Int<kTileM>{}), make_stride(Int<kTileM>{}, Int<1>{}));
  auto gYY =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileV>{}, Int<kTileM>{}), make_stride(Int<1>{}, Int<kTileV>{}));

  auto sQ_nope = make_tensor(make_smem_ptr(shm_q_nope), SLayoutQNope{});
  auto sQ_rope = make_tensor(make_smem_ptr(shm_q_rope), SLayoutQRope{});
  auto sK_nope = make_tensor(make_smem_ptr(shm_k_nope), SLayoutKNope{});
  auto sK_rope = make_tensor(make_smem_ptr(shm_k_rope), SLayoutKRope{});
  auto sV = make_tensor(make_smem_ptr(shm_k_nope), SLayoutV{});

  auto btma_q_nope = tma_q_nope.get_slice(0);
  auto btma_q_rope = tma_q_rope.get_slice(0);

  auto tQng = btma_q_nope.partition_S(gQ_nope);
  auto tQrg = btma_q_rope.partition_S(gQ_rope);

  auto tQns = btma_q_nope.partition_D(sQ_nope);
  auto tQrs = btma_q_rope.partition_D(sQ_rope);

  if (is_leader_in_block) {
    initialize_barrier(q_readable, 1);
    initialize_barrier(q_writable, kNumMathWG);
#pragma unroll
    for (int istage = 0; istage < kStage; ++istage) {
      initialize_barrier(kv_writable[istage], kConsumers);
      // 128 producer threads each call cpasync_barrier_arrive_noinc once per stage.
      initialize_barrier(kv_readable[istage], 128);
    }
  }

  if (idx == 0) {
    shm_task_start = cu_tasks[isM];
    shm_task_end = cu_tasks[isM + 1];
  }
  __syncthreads();

  int task_start = shm_task_start;
  int task_end = shm_task_end;

  // Empty SM: no tasks.
  if (task_start >= task_end) {
    return;
  }

  // Producer (load) warp: idx >= kMathThreads. All 128 producer threads
  // cooperate on cp.async; only the leader issues the (dense) Q TMAs.
  // Per kNumMathWG, balance regs against the 64K SM budget so neither WG
  // spills. Math-side budget is dominated by kTileM=32 (most accumulator
  // tiles); producer needs enough for the cute partition tensors.
  //   kNumMathWG=1: 240 + 88 = 328 → 328*128 = 41984 ≤ 65536  ✓
  //   kNumMathWG=2: 224 + 56  = 504 → 224*256 + 56*128 = 64512 ≤ 65536  ✓
  constexpr int kLoadRegs = (kNumMathWG == 1) ? 88 : 56;
  constexpr int kMathRegs = (kNumMathWG == 1) ? 240 : 224;
  if (idx >= kMathThreads) {
    cutlass::arch::warpgroup_reg_dealloc<kLoadRegs>();

    int load_idx = idx - kMathThreads;
    int load_iwarp = __shfl_sync(0xFFFFFFFF, load_idx / 32, 0);
    bool is_leader_in_load = (load_iwarp == 0) && elected;

    // Single thread layout (16, 8) val (1, 8) reused for nope (kTileNope=512)
    // and rope (kTileRope=64); per-row index extraction is therefore identical
    // across both copies, and we only need one identity tensor.
    auto kv_g2s_tiled_copy = make_tiled_copy(
        Copy_Atom<Copy_Traits<SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>>, Tin>{},
        make_layout(make_shape(Int<16>{}, Int<8>{}), make_stride(Int<1>{}, Int<16>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{})));

    auto thr_copy = kv_g2s_tiled_copy.get_slice(load_idx);

    auto IKV = make_identity_tensor(make_shape(Int<kTileN>{}, Int<kTileNope>{}));
    auto tIKV = thr_copy.partition_S(IKV);  // row indices shared across nope/rope

    // Source-base tensors: row stride = qk_dim, col stride = 1. Per-row source
    // is base + shm_kv_offset[stage][ikv] (element offset).
    auto gKV_nope_base =
        make_tensor(make_gmem_ptr(static_cast<const Tin *>(kvcache_ptr)),
                    make_shape(Int<kTileN>{}, Int<kTileNope>{}), make_stride(qk_dim, Int<1>{}));
    auto gKV_rope_base =
        make_tensor(make_gmem_ptr(static_cast<const Tin *>(kvcache_ptr) + kTileNope),
                    make_shape(Int<kTileN>{}, Int<kTileRope>{}), make_stride(qk_dim, Int<1>{}));

    auto tKVgKV_nope = thr_copy.partition_S(gKV_nope_base);
    auto tKVgKV_rope = thr_copy.partition_S(gKV_rope_base);
    auto tKVsKV_nope = thr_copy.partition_D(sK_nope);
    auto tKVsKV_rope = thr_copy.partition_D(sK_rope);

    constexpr int kKVCopyN = size<1>(tKVgKV_nope);
    constexpr int kKVCopyK_nope = size<2>(tKVgKV_nope);
    constexpr int kKVCopyK_rope = size<2>(tKVgKV_rope);
    static_assert(size<1>(tKVgKV_rope) == kKVCopyN, "row-pass count must match for nope/rope");

    int q_phase_w = 1;
    int kv_phase = 1;
    int kv_istage = 0;

#pragma unroll 1
    for (int task_offset = task_start; task_offset < task_end; ++task_offset) {
      int4 t0 = task_list[task_offset * 2 + 0];
      int itoken = t0.x;
      if (itoken < 0) {
        break;  // sentinel
      }
      int ibatch = t0.y;
      int ikv_tile_start = t0.z;
      int ikv_tile_end = t0.w;

      // Q TMA: leader-only (Q is dense per query token).
      if (is_leader_in_load) {
        wait_barrier(q_writable, q_phase_w);

        cute::copy(tma_q_nope.with(q_readable, 0, TMA::CacheHintSm90::EVICT_FIRST),
                   tQng(_, 0, _, itoken), tQns(_, 0, _));
        cute::copy(tma_q_rope.with(q_readable, 0, TMA::CacheHintSm90::EVICT_FIRST),
                   tQrg(_, 0, _, itoken), tQrs(_, 0, _));
        set_barrier_transaction_bytes(
            q_readable, sizeof(Tin) * (cosize(SLayoutQNope{}) + cosize(SLayoutQRope{})));
      }
      q_phase_w ^= 1;

      const int *batch_block_ids = block_ids_ptr + ibatch * num_seq_max_blocks;
      const int *batch_topk_ids = topk_ids_ptr + itoken * num_max_topk;

#pragma unroll 1
      for (int itile_seq_kv = ikv_tile_end - 1; itile_seq_kv >= ikv_tile_start; --itile_seq_kv) {
        wait_barrier(kv_writable[kv_istage], kv_phase);

        // First 64 producer lanes resolve topk → (kv_offset_in_elements, valid).
        // ZFILL on cp.async pads invalid rows with zero in smem; the math WG
        // additionally masks them to -inf at softmax time.
        if (load_idx < kTileN) {
          int itopk = itile_seq_kv * kTileN + load_idx;
          int t_id = (itopk < num_max_topk) ? batch_topk_ids[itopk] : -1;
          bool is_valid = (t_id >= 0);
          int row_delta = 0;
          if (is_valid) {
            int iblk = t_id / kBlockSize;
            int islot = t_id - iblk * kBlockSize;
            int page = batch_block_ids[iblk];
            int real_row = page * kBlockSize + islot;
            row_delta = real_row - load_idx;
          }
          shm_kv_offset[kv_istage][load_idx] = row_delta;
          shm_kv_valid[kv_istage][load_idx] = is_valid;
        }
        bar_sync<128>(kProducerBarrier);  // publish kv_offset / valid to all producer threads

#pragma unroll
        for (int in = 0; in < kKVCopyN; ++in) {
          int ikv = get<0>(tIKV(0, in, 0));
          int64_t kv_off = static_cast<int64_t>(shm_kv_offset[kv_istage][ikv]) * qk_dim;
          bool valid = shm_kv_valid[kv_istage][ikv];
#pragma unroll
          for (int ik = 0; ik < kKVCopyK_nope; ++ik) {
            auto src = make_tensor(tKVgKV_nope(_, in, ik).data() + kv_off,
                                   tKVgKV_nope(_, in, ik).layout());
            cute::copy(kv_g2s_tiled_copy.with(valid), src, tKVsKV_nope(_, in, ik, kv_istage));
          }
#pragma unroll
          for (int ik = 0; ik < kKVCopyK_rope; ++ik) {
            auto src = make_tensor(tKVgKV_rope(_, in, ik).data() + kv_off,
                                   tKVgKV_rope(_, in, ik).layout());
            cute::copy(kv_g2s_tiled_copy.with(valid), src, tKVsKV_rope(_, in, ik, kv_istage));
          }
        }

        cpasync_barrier_arrive_noinc(reinterpret_cast<uint64_t *>(&kv_readable[kv_istage]));

        kv_istage += 1;
        if (kv_istage == kStage) {
          kv_istage = 0;
          kv_phase ^= 1;
        }
      }
    }
  } else {
    cutlass::arch::warpgroup_reg_alloc<kMathRegs>();

    int idx_in_warpgroup = idx % 128;
    int iwarpgroup = idx / 128;
    int iwarp_in_warpgroup = idx_in_warpgroup / 32;
    int ilane_in_warpgroup = idx_in_warpgroup % 32;
    int elected_idx_in_warpgroup = ((iwarp_in_warpgroup == 0) && elected);

    // Per-WG views into the shared smem regions. For kNumMathWG=1 the slice
    // is the whole tensor / single allocation, identical to the original
    // single-WG layout.
    auto sQ_nope_wg =
        local_tile(sQ_nope, make_shape(Int<kTileM>{}, Int<kTileNope>{}), make_coord(iwarpgroup, 0));
    auto sQ_rope_wg =
        local_tile(sQ_rope, make_shape(Int<kTileM>{}, Int<kTileRope>{}), make_coord(iwarpgroup, 0));
    auto *shm_p_wg = shm_p + iwarpgroup * cosize(SLayoutP{});
    auto sP = make_tensor(make_smem_ptr(shm_p_wg), SLayoutP{});
    auto sS = make_tensor(make_smem_ptr(shm_p_wg), SLayoutS{});

    // Each math WG owns a fixed kTileM-stride head slot; the last WG may have
    // < kTileM real heads when num_head_q is not a multiple of kTileM (or is
    // < kHeadsTotal). Epilogue masks m_local >= actual_num_heads_wg, and TMA
    // OOB rows zero-fill so out-of-range head slots contribute nothing.
    int head_offset = iwarpgroup * kTileM;
    int rem_heads = num_head_q - head_offset;
    int actual_num_heads_wg = rem_heads < 0 ? 0 : (rem_heads > kTileM ? kTileM : rem_heads);

    TiledMmaQK tiled_mma_qk;
    TiledMmaSV tiled_mma_sv;

    auto thr_mma_qk = tiled_mma_qk.get_slice(idx_in_warpgroup);
    auto thr_mma_sv = tiled_mma_sv.get_slice(idx_in_warpgroup);

    auto tKns4r = thr_mma_qk.partition_A(sK_nope);
    auto tKrs4r = thr_mma_qk.partition_A(sK_rope);
    auto tQns4r = thr_mma_qk.partition_B(sQ_nope_wg);
    auto tQrs4r = thr_mma_qk.partition_B(sQ_rope_wg);
    auto tVs4r = thr_mma_sv.partition_A(sV);
    auto tSs4r = thr_mma_sv.partition_B(sS);

    auto tKnr = thr_mma_qk.make_fragment_A(tKns4r);
    auto tKrr = thr_mma_qk.make_fragment_A(tKrs4r);
    auto tQnr = thr_mma_qk.make_fragment_B(tQns4r);
    auto tQrr = thr_mma_qk.make_fragment_B(tQrs4r);
    auto tVr = thr_mma_sv.make_fragment_A(tVs4r);
    auto tSr = thr_mma_sv.make_fragment_B(tSs4r);

    auto tAttr = thr_mma_qk.partition_fragment_C(gAtt);
    auto tYr = thr_mma_sv.partition_fragment_C(gYY);

    auto gI = make_identity_tensor(gAtt.shape());
    auto tI = thr_mma_qk.partition_C(gI);
    auto gIY = make_identity_tensor(gYY.shape());
    auto tIY = thr_mma_sv.partition_C(gIY);

    auto tAttr_nm = retile_fragment(tAttr);
    auto tI_nm = retile_fragment(tI);
    auto tYr_nm = retile_fragment(tYr);
    auto tIY_nm = retile_fragment(tIY);

    constexpr int kN = size<0>(tAttr_nm);
    constexpr int kM = size<1>(tAttr_nm);
    constexpr int kVN = size<0>(tYr_nm);
    Tensor gMax = make_tensor<float>(Int<kM>{});
    Tensor gSum = make_tensor<float>(Int<kM>{});

    using R2SCopyAtomP = Copy_Atom<typename Dim576PersistentR2SCopyAtomSelector<kTileM>::type, Tin>;
    auto tiled_copy_P_r2s = make_tiled_copy_C(R2SCopyAtomP{}, tiled_mma_qk);
    auto thr_copy_P_r2s = tiled_copy_P_r2s.get_slice(idx_in_warpgroup);
    auto tPs4r = thr_copy_P_r2s.partition_D(sP);

    int q_phase_r = 0;
    int kv_phase = 0;
    int kv_istage = 0;

#pragma unroll 1
    for (int task_offset = task_start; task_offset < task_end; ++task_offset) {
      int4 t0 = task_list[task_offset * 2 + 0];
      int itoken = t0.x;
      if (itoken < 0) {
        break;
      }
      int ikv_tile_start = t0.z;
      int ikv_tile_end = t0.w;
      int isplit_in_token = task_list[task_offset * 2 + 1].y;

      wait_barrier(q_readable, q_phase_r);
      q_phase_r ^= 1;

      clear(gSum);
      fill(gMax, -std::numeric_limits<float>::infinity());
      clear(tYr);
      tiled_mma_sv.accumulate_ = GMMA::ScaleOut::One;

#pragma unroll 1
      for (int itile_seq_kv = ikv_tile_end - 1; itile_seq_kv >= ikv_tile_start; --itile_seq_kv) {
        wait_barrier(kv_readable[kv_istage], kv_phase);

        tiled_mma_qk.accumulate_ = GMMA::ScaleOut::Zero;
        warpgroup_fence_operand(tAttr);
        warpgroup_arrive();
#pragma unroll
        for (int ik = 0; ik < size<2>(tQnr); ++ik) {
          cute::gemm(tiled_mma_qk, tKnr(_, _, ik, kv_istage), tQnr(_, _, ik), tAttr(_, _, _));
          tiled_mma_qk.accumulate_ = GMMA::ScaleOut::One;
        }
#pragma unroll
        for (int ik = 0; ik < size<2>(tQrr); ++ik) {
          cute::gemm(tiled_mma_qk, tKrr(_, _, ik, kv_istage), tQrr(_, _, ik), tAttr(_, _, _));
        }
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(tAttr);

        // Sparse validity mask: invalid topk rows (-1 / out-of-range) get -inf
        // so they contribute nothing to softmax. ZFILL on cp.async already
        // zeros their K, but Q·0 = 0 ≠ -inf — explicit mask is required.
        const int *batch_topk_ids = topk_ids_ptr + itoken * num_max_topk;
#pragma unroll
        for (int in = 0; in < kN; ++in) {
#pragma unroll
          for (int im = 0; im < kM; ++im) {
            int row_in_tile = get<0>(tI_nm(in, im));
            int itopk = itile_seq_kv * kTileN + row_in_tile;
            int t_id = (itopk < num_max_topk) ? batch_topk_ids[itopk] : -1;
            if (t_id < 0) {
              tAttr_nm(in, im) = -std::numeric_limits<float>::infinity();
            }
          }
        }

        online_softmax<true, kNumMathWG, kHeadsTotal, kM, kN>(
            tAttr_nm, gMax, gSum, tYr_nm, one_over_dk_log2e, shm_max, iwarpgroup,
            iwarp_in_warpgroup, ilane_in_warpgroup);

        auto tAttAbf16 = make_tensor_like<cute::bfloat16_t>(tAttr);
#pragma unroll
        for (int i = 0; i < size(tAttr); ++i) {
          tAttAbf16(i) = (cute::bfloat16_t)(tAttr(i));
        }
        auto tPr4s = thr_copy_P_r2s.retile_S(tAttAbf16);
        cute::copy(tiled_copy_P_r2s, tPr4s, tPs4r);

        cutlass::arch::fence_view_async_shared();
        syncwarpgroup(iwarpgroup);

        warpgroup_fence_operand(tYr);
        warpgroup_arrive();
        cute::gemm(tiled_mma_sv, tVr(_, _, _, kv_istage), tSr, tYr);
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(tYr);

        if (elected_idx_in_warpgroup) {
          arrive_barrier(kv_writable[kv_istage]);
        }
        kv_istage += 1;
        if (kv_istage == kStage) {
          kv_istage = 0;
          kv_phase ^= 1;
        }
        syncwarpgroup(iwarpgroup);
      }

      float *shm_sum = shm_max + kHeadsTotal * 4;
      float *shm_sum_wg = shm_sum + iwarpgroup * (kTileM * 4);
      int isplit_global = cu_splits[itoken] + isplit_in_token;
      int num_splits_for_batch = cu_splits[itoken + 1] - cu_splits[itoken];

      if (num_splits_for_batch == 1) {
        // y row index == itoken (query-token index).
        single_split_epilogue_dim576<kTileM, kTileV, kM, kVN, kUseSink, Tout>(
            tYr_nm, gMax, gSum, tIY_nm, shm_sum_wg, y_ptr, sink_weight_ptr, itoken, ldY, v_dim,
            iwarp_in_warpgroup, ilane_in_warpgroup,
            /*actual_num_heads=*/actual_num_heads_wg, /*iwarpgroup=*/iwarpgroup,
            /*head_offset=*/head_offset);
      } else {
        splitk_epilogue_dim576<kTileM, kTileV, kM, kVN>(
            tYr_nm, gMax, gSum, tIY_nm, shm_sum_wg, y_partial_ptr, lse_ptr,
            /*row_base=*/isplit_global * num_head_q + head_offset, v_dim, iwarp_in_warpgroup,
            ilane_in_warpgroup,
            /*actual_num_heads=*/actual_num_heads_wg, /*iwarpgroup=*/iwarpgroup);
      }

      if (elected_idx_in_warpgroup) {
        arrive_barrier(q_writable);
      }
    }
  }
}

}  // namespace kernels
}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_MLA_SMALLM_SPARSE_MLA_DIM576_PERSISTENT_KERNELS_CUH_
