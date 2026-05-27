// Copyright 2025 hpc-ops authors

#ifndef SRC_ATTENTION_DECODE_SM90_DYNAMIC_SMALLM_FP8_QPERTOKEN_PERHEAD_KVPERTENSOR_DYNAMIC_KERNELS_CUH_  // NOLINT
#define SRC_ATTENTION_DECODE_SM90_DYNAMIC_SMALLM_FP8_QPERTOKEN_PERHEAD_KVPERTENSOR_DYNAMIC_KERNELS_CUH_  // NOLINT

#include <cuda.h>
#include <stdio.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "src/attention/decode/sm90/dynamic/dynamic_sched_task_info.h"
#include "src/attention/decode/sm90/util_kernels.cuh"  // sm90 gemm/softmax/copy utils
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace attention {
namespace decode {
namespace dynamic {
namespace kernels {

// Minimal task descriptor local to the dynamic sm90 path (runtime view of
// SM90DynamicTaskInfo after parse_task decodes it).
struct DynamicTaskInfo {
  int ihead_kv;
  int ibatch;
  int ichunk;
  int num_seq_kvcache;
  int num_seq_kv;
  int num_blocks;
  int num_blocks_per_chunk;
  int num_tile_kv;
  int num_tile_full;
  int num_tile_causal;
};

// Parse one SM90DynamicTaskInfo (12 ints / 48 B) from the sm90 dynamic
// task_map. Three int4 loads — the extra load vs the legacy 8-int layout is
// the cost of carrying ihead_kv in-task, which lets the launcher be a flat
// `dim3(num_sm × kCTAPerSM)` rather than encoding head via blockIdx.x.
//
// Returns false when the slot is a terminator (ihead_kv<0 or ibatch<0).
template <int kBlockSize>
__device__ __forceinline__ bool parse_task(DynamicTaskInfo& task_info, const int* task_map_ptr) {
  using namespace cute;  // NOLINT
  auto v1 = load<int, 4>(task_map_ptr);
  auto v2 = load<int, 4>(task_map_ptr + 4);
  auto v3 = load<int, 4>(task_map_ptr + 8);

  int ihead_kv = v1[0];
  int ibatch = v1[1];
  if (ihead_kv < 0 || ibatch < 0) {
    return false;
  }
  int ichunk = v1[2];
  int iseq_start = v1[3];

  int num_seq_kv = v2[0];
  int num_seq_kvcache = v2[1];
  int num_tile_kv = v2[2];
  int num_tile_full = v2[3];

  int is_casual_chunk = v3[0];
  // v3[1..3] are pad ints — unused.

  task_info.ihead_kv = ihead_kv;
  task_info.ibatch = ibatch;
  task_info.ichunk = ichunk;
  task_info.num_seq_kvcache = num_seq_kvcache;
  task_info.num_seq_kv = num_seq_kv;
  task_info.num_blocks = (num_seq_kv + kBlockSize - 1) / kBlockSize;
  task_info.num_blocks_per_chunk = (iseq_start + kBlockSize - 1) / kBlockSize;
  task_info.num_tile_kv = num_tile_kv;
  task_info.num_tile_full = num_tile_full;
  task_info.num_tile_causal = is_casual_chunk ? (num_tile_kv - num_tile_full) : 0;
  return true;
}

template <typename Tin, typename Tout, int kTileM, int kTileN, int kTileK, int kTileV,
          int kHeadsPerGroup, typename TiledMmaQK, typename TiledMmaSV, typename TmaQ,
          typename TmaK, typename TmaV, typename TmaSplitY, typename SLayoutQ, typename SLayoutK,
          typename SLayoutP, typename SLayoutS, typename SLayoutVTma, typename SLayoutSplitY,
          int kBlockSize, int kStage, int kMaxSplitK>
__global__ void attention_decode_fp8_dynamic_smallm_qpertoken_perhead_kvpertensor_kernel(
    const __grid_constant__ TmaQ tma_q, const __grid_constant__ TmaK tma_k,
    const __grid_constant__ TmaV tma_v, const __grid_constant__ TmaSplitY tma_splity,
    float* split_y_ptr, float* lse_ptr, const int* task_map_ptr, const int* block_ids_ptr,
    const float* qscale_ptr, const float* kscale_ptr, const float* vscale_ptr, int num_batch,
    int num_seq_q, int num_dim_qk, int num_dim_v, int num_head_q, int num_head_k, int num_head_v,
    int heads_per_group, int lse_pad_heads_per_group, int num_kvcache_blocks,
    int num_seq_max_blocks, int qscale_pad_stride, float one_over_dk_log2e) {
  using namespace cute;  // NOLINT
  using TaskInfo = DynamicTaskInfo;

  constexpr int kWarpGroupN = 1;
  constexpr int kMathThreads = size(TiledMmaQK{}) * kWarpGroupN;  // = 128
  constexpr int kWarpsPerWrapGroup = 4;
  constexpr int kTaskStride = kSM90DynamicTaskStride;  // = 12

  int idx = threadIdx.x;
  int elected = cute::elect_one_sync();
  int iwarp = idx / 32;

  int cta_id = blockIdx.x;

  // PDL
  cudaGridDependencySynchronize();

  __shared__ uint64_t q_readable;
  __shared__ uint64_t q_writable;
  __shared__ uint64_t k_writable[kStage];
  __shared__ uint64_t v_writable[kStage];
  __shared__ uint64_t k_readable[kStage];
  __shared__ uint64_t v_readable[kStage];
  extern __shared__ uint8_t shm_data[] alignas(128);

  auto* shm_q = reinterpret_cast<Tin*>(shm_data);
  auto* shm_k = shm_q + cosize(SLayoutQ{});
  auto* shm_vtma = shm_k + cosize(SLayoutK{});
  auto* shm_p = shm_vtma + cosize(SLayoutVTma{});
  auto* shm_max = reinterpret_cast<float*>(shm_p + cosize(SLayoutP{}));
  auto* shm_kvblk_ids = reinterpret_cast<int*>(shm_max + kTileM * kWarpsPerWrapGroup * kWarpGroupN);
  auto* shm_splity = reinterpret_cast<float*>(shm_data);  // reuse shm_q/k/v/p for epilogue

  auto gQ = tma_q.get_tma_tensor(
      make_shape(heads_per_group, num_dim_qk, num_head_k, num_seq_q, num_batch));
  auto gK =
      tma_k.get_tma_tensor(make_shape(kBlockSize, num_dim_qk, num_head_k, num_kvcache_blocks));
  auto gV = tma_v.get_tma_tensor(make_shape(num_dim_v, kBlockSize, num_head_v, num_kvcache_blocks));
  auto gSplitY = tma_splity.get_tma_tensor(
      make_shape(num_dim_v, heads_per_group, num_head_k, num_seq_q, kMaxSplitK, num_batch));

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
  auto sVTma = make_tensor(make_smem_ptr(shm_vtma), SLayoutVTma{});
  auto sSplitY = make_tensor(make_smem_ptr(shm_splity), SLayoutSplitY{});

  auto btma_q = tma_q.get_slice(0);
  auto btma_k = tma_k.get_slice(0);
  auto btma_v = tma_v.get_slice(0);

  auto tQg = btma_q.partition_S(gQ);
  auto tKg = btma_k.partition_S(gK);
  auto tVg = btma_v.partition_S(gV);

  auto tQs = btma_q.partition_D(sQ);
  auto tKs = btma_k.partition_D(sK);
  auto tVs = btma_v.partition_D(sVTma);

  const float kscale = kscale_ptr[0];
  const float vscale = vscale_ptr[0] / 256;

  // ==========================================================================
  // One-time mbarrier init (sm100 pattern). Each warp inits its own group of
  // mbarriers and emits an async-proxy fence; the trailing __syncthreads()
  // makes them visible to all warps.
  // ==========================================================================
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

  int num_tiles_per_cta = task_map_ptr[0];
  const int* my_task_ptr = task_map_ptr + (1 + cta_id * num_tiles_per_cta) * kTaskStride;

  if (idx >= kMathThreads) {
    // ====================== Load warp (32 threads) ========================
    const bool is_leader_in_load = ((iwarp == kMathThreads / 32) && elected);
    const int lane = idx & 31;

    // Per-task phase trackers (carry across tasks):
    //   phase_qw      : q_writable consumer-side. Initial barrier phase = 0,
    //                   so wait_barrier(q_writable, 1) on the FIRST task
    //                   returns immediately (no previous user of shm_q).
    //   kv_istage_w   : producer-side stage cursor for K / V.
    //   kv_phase_w    : producer-side phase cursor; initial 1 because the
    //                   freshly-initialized k_writable[i] / v_writable[i]
    //                   start at phase 0.
    int phase_qw = 1;
    int kv_istage_w = 0;
    int kv_phase_w = 1;

    while (true) {
      TaskInfo task;
      if (!parse_task<kBlockSize>(task, my_task_ptr)) {
        break;
      }

      const int ihead_kv = task.ihead_kv;
      const int ibatch = task.ibatch;
      const int num_blocks = task.num_blocks;
      const int num_tile_kv = task.num_tile_kv;
      const int num_tile_full = task.num_tile_full;

      // Cooperative block_ids load (32 threads). shm_kvblk_ids does NOT
      // overlap shm_splity, so this is safe to do concurrently with the
      // previous task's epilogue.
      const int* block_ids =
          block_ids_ptr + ibatch * num_seq_max_blocks + task.num_blocks_per_chunk;
      for (int i = lane; i < num_blocks; i += 32) {
        shm_kvblk_ids[i] = block_ids[i];
      }
      __syncwarp();

      if (is_leader_in_load) {
        wait_barrier(q_writable, phase_qw);
        phase_qw ^= 1;

        // Q TMA — q_readable's transaction count is set per task.
        for (int iseqq = 0; iseqq < num_seq_q; iseqq++) {
          cute::copy(tma_q.with(q_readable), tQg(_, 0, _, ihead_kv, iseqq, ibatch),
                     tQs(_, iseqq, _));
        }
        set_barrier_transaction_bytes(q_readable, sizeof(Tin) * cosize(SLayoutQ{}));

        // K/V TMAs: causal first (high indices), then full (low indices).
        // Pipeline across kStage stages.
        constexpr int kBlockPerTileN = kTileN / kBlockSize;
        int iload_tile = 0;

#pragma unroll 1
        for (int itile_seq_kv = num_tile_full; itile_seq_kv < num_tile_kv; ++itile_seq_kv) {
          hpc::attention::kernels::load_paged_kv<true, kBlockPerTileN, kBlockSize, kStage, Tin>(
              tma_k, tma_v, k_writable, v_writable, k_readable, v_readable, tKg, tKs, tVg, tVs,
              ihead_kv, num_dim_qk, num_dim_v, shm_kvblk_ids, num_blocks, itile_seq_kv, kv_istage_w,
              kv_phase_w);
          hpc::attention::kernels::advance_stage<kStage>(kv_istage_w, kv_phase_w);
        }

#pragma unroll 1
        for (int itile_seq_kv = -kStage + 1; itile_seq_kv < num_tile_full; ++itile_seq_kv) {
          if (iload_tile < num_tile_full) {
            hpc::attention::kernels::load_paged_kv<false, kBlockPerTileN, kBlockSize, kStage, Tin>(
                tma_k, tma_v, k_writable, v_writable, k_readable, v_readable, tKg, tKs, tVg, tVs,
                ihead_kv, num_dim_qk, num_dim_v, shm_kvblk_ids, num_blocks, iload_tile++,
                kv_istage_w, kv_phase_w);
            hpc::attention::kernels::advance_stage<kStage>(kv_istage_w, kv_phase_w);
          }
        }
      }

      // Keep all 32 threads in lockstep before the next iter overwrites
      // shm_kvblk_ids. Leader's load_paged_kv reads shm_kvblk_ids when issuing
      // K/V TMAs (the smem→reg load happens at issue time, not async); the
      // other 31 threads must not start the next task's coop block_ids load
      // until leader has finished all of its TMA issues for this task.
      __syncwarp();

      my_task_ptr += kTaskStride;
    }
  } else {
    // ===================== Math warpgroup (128 threads) =====================
    const int idx_in_warpgroup = idx % 128;
    const int iwarp_in_warpgroup = idx_in_warpgroup / 32;
    const int ilane_in_warpgroup = idx_in_warpgroup % 32;
    const bool elected_idx_in_warpgroup = ((iwarp_in_warpgroup == 0) && elected);
    const bool is_leader_in_warpgroup = ((iwarp % 4) == 0) && elected;
    constexpr int kIWarpgroup = 0;

    // ---- once-only setup: hoisted out of the per-task while loop ----
    TiledMmaQK tiled_mma_qk;
    TiledMmaSV tiled_mma_sv;

    auto thr_mma_qk = tiled_mma_qk.get_slice(idx_in_warpgroup);
    auto thr_mma_sv = tiled_mma_sv.get_slice(idx_in_warpgroup);

    auto tKs4r = thr_mma_qk.partition_A(sK);
    auto tQs4r = thr_mma_qk.partition_B(sQ);
    auto tVs4r = thr_mma_sv.partition_A(sVTma(_, _, 0));
    auto tSs4r = thr_mma_sv.partition_B(sS);

    auto tKr = thr_mma_qk.make_fragment_A(tKs4r);
    auto tQr = thr_mma_qk.make_fragment_B(tQs4r);
    auto tVr = thr_mma_sv.make_fragment_A(tVs4r);
    auto tSr = thr_mma_sv.make_fragment_B(tSs4r);

    auto tAttr = thr_mma_qk.partition_fragment_C(gAtt);
    auto tAttAfp8 = make_tensor_like<Tin>(tAttr);
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
    Tensor qkscales = make_tensor<float>(Int<kM>{});
    Tensor gSoftmaxScale = make_tensor<float>(Int<kM>{});

    auto tiled_copy_P_r2s = hpc::attention::kernels::make_tiled_copy_P_interleave<kTileM, Tin>();
    auto thr_copy_P_r2s = tiled_copy_P_r2s.get_slice(idx_in_warpgroup);
    auto tPr4s = thr_copy_P_r2s.retile_S(tAttAfp8);
    auto tPs4r = thr_copy_P_r2s.partition_D(sP(_, _, kIWarpgroup));

    auto tiled_copy_VT_s2r = hpc::attention::kernels::make_tiled_copy_V_interleave_trans<Tin>();
    auto thr_copy_VT_s2r = tiled_copy_VT_s2r.get_slice(idx_in_warpgroup);
    auto tVTs4r = thr_copy_VT_s2r.partition_S(sVTma);
    auto tVTr4s = make_fragment_like(thr_copy_VT_s2r.partition_D(sVTma(_, _, 0)));

    auto v_for_trans = recast<uint32_t>(tVTr4s);
    auto vt_for_trans = recast<uint32_t>(
        make_tensor(tVr.data(), left_inverse(tVr.layout()).compose(tVTr4s.layout())));

    using R2SCopyAtomSplitY = Copy_Atom<UniversalCopy<uint32_t>, float>;
    auto tiled_copy_SplitY_r2s =
        hpc::attention::kernels::make_tiled_copy_Y_interleave<kTileM>(R2SCopyAtomSplitY{});

    float* shm_max_wg = shm_max + kIWarpgroup * kTileM * kWarpsPerWrapGroup;

    int kv_istage_r = kIWarpgroup;
    int kv_phase_r = 0;
    int phase_q = 0;

    while (true) {
      TaskInfo task;
      if (!parse_task<kBlockSize>(task, my_task_ptr)) {
        break;
      }

      const int ihead_kv = task.ihead_kv;
      const int ibatch = task.ibatch;
      const int ichunk = task.ichunk;
      const int num_seq_kvcache = task.num_seq_kvcache;
      const int num_seq_kv = task.num_seq_kv;
      const int num_tile_kv = task.num_tile_kv;
      const int num_tile_full = task.num_tile_full;
      const int num_tile_causal = task.num_tile_causal;

      float* lse_batch = lse_ptr +
                         ibatch * kMaxSplitK * num_head_k * lse_pad_heads_per_group * num_seq_q +
                         ichunk * num_head_k * lse_pad_heads_per_group * num_seq_q +
                         ihead_kv * lse_pad_heads_per_group * num_seq_q;

      const float* qscales =
          qscale_ptr + ibatch * num_head_q * num_seq_q + ihead_kv * heads_per_group;

#pragma unroll
      for (int i = 0; i < kM; i++) {
        int im = get<1>(tI_nm(0, i));
        int iseqq = im / kHeadsPerGroup;
        int iqhead = im % kHeadsPerGroup;
        if (iqhead < heads_per_group) {
          qkscales(i) = qscales[iseqq * num_head_q + iqhead];
        } else {
          qkscales(i) = 1;
        }
      }

      clear(gSum);
      fill(gMax, -std::numeric_limits<float>::infinity());
      fill(gSoftmaxScale, one_over_dk_log2e);

      clear(tYr);
      tiled_mma_sv.accumulate_ = GMMA::ScaleOut::One;

      // Wait for load warp's Q TMA (per-task phase toggle).
      wait_barrier(q_readable, phase_q);
      phase_q ^= 1;

      bool warpgroup_computed = false;

#pragma unroll
      for (int i = 0; i < kM; i++) {
        qkscales(i) *= kscale;
      }

      // Causal tiles.
#pragma unroll 1
      for (int itile_seq_kv = num_tile_full + kIWarpgroup; itile_seq_kv < num_tile_kv;
           itile_seq_kv += kWarpGroupN) {
        warpgroup_computed = true;
        wait_barrier(k_readable[kv_istage_r], kv_phase_r);

        hpc::attention::kernels::qk_gemm(tiled_mma_qk, tQr, tKr, tAttr, kv_istage_r);

        if (elected_idx_in_warpgroup) {
          arrive_barrier(k_writable[kv_istage_r]);
        }

        hpc::attention::kernels::apply_casual_mask_with_scale<kTileN, kHeadsPerGroup>(
            tAttr_nm, tI_nm, qkscales, itile_seq_kv, num_seq_kvcache, num_seq_kv);

        hpc::attention::kernels::online_softmax<true, kTileM>(
            tAttr_nm, gMax, gSum, tYr_nm, gSoftmaxScale, shm_max_wg, kIWarpgroup,
            iwarp_in_warpgroup, ilane_in_warpgroup);

#pragma unroll
        for (int im = 0; im < kM; ++im) {
#pragma unroll
          for (int in = 0; in < kN; ++in) {
            tAttr_nm(in, im) *= 256;
          }
        }

        hpc::attention::kernels::cast_fp32reg<Tin>(tAttr_nm, tAttAfp8);
        hpc::attention::kernels::permute_p(tAttAfp8, 0x7520);
        cute::copy(tiled_copy_P_r2s, tPr4s, tPs4r);

        wait_barrier(v_readable[kv_istage_r], kv_phase_r);
        cutlass::arch::fence_view_async_shared();
        hpc::syncwarpgroup(kIWarpgroup);

        cute::copy(tiled_copy_VT_s2r, tVTs4r(_, _, _, kv_istage_r), tVTr4s);
        hpc::attention::kernels::permute_v_sv_gemm(tiled_mma_sv, tSr, tVr, tYr, v_for_trans,
                                                   vt_for_trans, kIWarpgroup);

        if (elected_idx_in_warpgroup) {
          arrive_barrier(v_writable[kv_istage_r]);
        }

        hpc::attention::kernels::advance_stage<kStage, kWarpGroupN>(kv_istage_r, kv_phase_r);

        warpgroup_wait<0>();
        warpgroup_fence_operand(tYr);
      }

#pragma unroll
      for (int i = 0; i < kM; i++) {
        gSoftmaxScale(i) *= qkscales(i);
      }

      // Full (non-causal) tiles.
#pragma unroll 1
      for (int itile_seq_kv = (kWarpGroupN - num_tile_causal + kIWarpgroup) % kWarpGroupN;
           itile_seq_kv < num_tile_full; itile_seq_kv += kWarpGroupN) {
        warpgroup_computed = true;
        wait_barrier(k_readable[kv_istage_r], kv_phase_r);

        hpc::attention::kernels::qk_gemm(tiled_mma_qk, tQr, tKr, tAttr, kv_istage_r);

        if (elected_idx_in_warpgroup) {
          arrive_barrier(k_writable[kv_istage_r]);
        }

        hpc::attention::kernels::online_softmax<false, kTileM>(
            tAttr_nm, gMax, gSum, tYr_nm, gSoftmaxScale, shm_max_wg, kIWarpgroup,
            iwarp_in_warpgroup, ilane_in_warpgroup);

#pragma unroll
        for (int im = 0; im < kM; ++im) {
#pragma unroll
          for (int in = 0; in < kN; ++in) {
            tAttr_nm(in, im) *= 256;
          }
        }

        hpc::attention::kernels::cast_fp32reg<Tin>(tAttr_nm, tAttAfp8);
        hpc::attention::kernels::permute_p(tAttAfp8, 0x7520);
        cute::copy(tiled_copy_P_r2s, tPr4s, tPs4r);

        wait_barrier(v_readable[kv_istage_r], kv_phase_r);
        cutlass::arch::fence_view_async_shared();
        hpc::syncwarpgroup(kIWarpgroup);

        cute::copy(tiled_copy_VT_s2r, tVTs4r(_, _, _, kv_istage_r), tVTr4s);
        hpc::attention::kernels::permute_v_sv_gemm(tiled_mma_sv, tSr, tVr, tYr, v_for_trans,
                                                   vt_for_trans, kIWarpgroup);

        if (elected_idx_in_warpgroup) {
          arrive_barrier(v_writable[kv_istage_r]);
        }

        hpc::attention::kernels::advance_stage<kStage, kWarpGroupN>(kv_istage_r, kv_phase_r);

        warpgroup_wait<0>();
        warpgroup_fence_operand(tYr);
      }

      if (warpgroup_computed) {
        hpc::attention::kernels::final_online_softmax<kTileM>(
            tYr_nm, gSum, shm_max_wg, kIWarpgroup, iwarp_in_warpgroup, ilane_in_warpgroup);
      }

#pragma unroll
      for (int i = 0; i < size(tYr); ++i) {
        tYr(i) *= vscale;
      }

      hpc::bar_sync<kWarpGroupN * 128>(kWarpGroupN);

      hpc::attention::kernels::store_output<true, kWarpGroupN>(
          tiled_copy_SplitY_r2s, tma_splity, tYr, sSplitY, gSplitY, ihead_kv, ibatch, ichunk,
          num_seq_q, idx_in_warpgroup, kIWarpgroup, is_leader_in_warpgroup);
      hpc::attention::kernels::store_lse(lse_batch, gMax, gSum, heads_per_group, ilane_in_warpgroup,
                                         iwarp_in_warpgroup);

      tma_store_wait<0>();

      if (elected_idx_in_warpgroup) {
        arrive_barrier(q_writable);
      }

      my_task_ptr += kTaskStride;
    }
  }

  // PDL
  cudaTriggerProgrammaticLaunchCompletion();
}

}  // namespace kernels
}  // namespace dynamic
}  // namespace decode
}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_DECODE_SM90_DYNAMIC_SMALLM_FP8_QPERTOKEN_PERHEAD_KVPERTENSOR_DYNAMIC_KERNELS_CUH_
        // // NOLINT
