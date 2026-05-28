// Copyright 2025 hpc-ops authors

#ifndef SRC_ATTENTION_MLA_SMALLM_MLA_DIM576_PERSISTENT_KERNELS_CUH_
#define SRC_ATTENTION_MLA_SMALLM_MLA_DIM576_PERSISTENT_KERNELS_CUH_

#include <cuda.h>
#include <stdio.h>

#include <algorithm>
#include <limits>

#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "src/attention/mla/smallm_mla_kernels.cuh"  // online_softmax
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace attention {
namespace kernels {

template <int kTileM>
struct Dim576PersistentR2SCopyAtomSelector;

template <>
struct Dim576PersistentR2SCopyAtomSelector<32> {
  using type = cute::SM90_U16x8_STSM_T;
};

template <>
struct Dim576PersistentR2SCopyAtomSelector<16> {
  using type = cute::SM90_U16x8_STSM_T;
};

template <>
struct Dim576PersistentR2SCopyAtomSelector<8> {
  using type = cute::SM90_U16x4_STSM_T;
};

// per-split epilogue (kM in {2,4}, kConsumers == 1)
template <int kTileM, int kTileV, int kM, int kVN, typename TensorY, typename TensorMax,
          typename TensorSum, typename TensorIY>
__device__ __forceinline__ void splitk_epilogue_dim576(
    TensorY &tYr_nm, TensorMax &gMax, TensorSum &gSum, TensorIY &tIY_nm, float *shm_sum,
    float *y_partial_ptr, float *lse_ptr, int row_base, int v_dim, int iwarp_in_warpgroup,
    int ilane_in_warpgroup, int actual_num_heads) {
  using namespace cute;  // NOLINT
  constexpr int kWarpsPerWarpGroup = 4;

  // warp-cross sum reduction (matches final_online_softmax)
  vec_t<float, kM> warp_sum;
#pragma unroll
  for (int im = 0; im < kM; ++im) {
    warp_sum[im] = warp_8lane_stride4_reduce_sum_xor(gSum(im));
  }

  if (ilane_in_warpgroup < 4) {
    if constexpr (kM == 8) {
      auto &warp_sum_16B = reshape<kM / 4, 4>(warp_sum);
      store(shm_sum + iwarp_in_warpgroup * kTileM + ilane_in_warpgroup * 4, warp_sum_16B[0]);
      store(shm_sum + iwarp_in_warpgroup * kTileM + 16 + ilane_in_warpgroup * 4, warp_sum_16B[1]);
    } else {
      store(shm_sum + iwarp_in_warpgroup * kTileM + ilane_in_warpgroup * kM, warp_sum);
    }
  }

  syncwarpgroup(0);

  if (ilane_in_warpgroup < 4) {
#pragma unroll
    for (int im = 0; im < kM; ++im) {
      warp_sum[im] = 0.f;
    }
#pragma unroll
    for (int i = 0; i < kWarpsPerWarpGroup; i++) {
      vec_t<float, kM> reduce_sum;
      if constexpr (kM == 8) {
        auto &reduce_sum_16B = reshape<kM / 4, 4>(reduce_sum);
        reduce_sum_16B[0] = load<float, 4>(shm_sum + i * kTileM + ilane_in_warpgroup * 4);
        reduce_sum_16B[1] = load<float, 4>(shm_sum + i * kTileM + 16 + ilane_in_warpgroup * 4);
      } else {
        reduce_sum = load<float, kM>(shm_sum + i * kTileM + ilane_in_warpgroup * kM);
      }
#pragma unroll
      for (int im = 0; im < kM; ++im) {
        warp_sum[im] += reduce_sum[im];
      }
    }
  }

#pragma unroll
  for (int im = 0; im < kM; ++im) {
    warp_sum[im] = __shfl_sync(0xFFFFFFFF, warp_sum[im], ilane_in_warpgroup % 4);
  }

  // normalise tYr_nm by warp_sum
#pragma unroll
  for (int im = 0; im < kM; ++im) {
    float one_over_gsum = rcpf_ftz(warp_sum[im]);
#pragma unroll
    for (int iv = 0; iv < kVN; ++iv) {
      tYr_nm(iv, im) = tYr_nm(iv, im) * one_over_gsum;
    }
  }

  // write fp32 partial Y
#pragma unroll
  for (int im = 0; im < kM; ++im) {
    int m_local = get<1>(tIY_nm(0, im));
    if (m_local >= actual_num_heads) {
      continue;
    }
    int row = (row_base + m_local) * v_dim;
#pragma unroll
    for (int iv = 0; iv < kVN; ++iv) {
      int v_local = get<0>(tIY_nm(iv, im));
      y_partial_ptr[row + v_local] = tYr_nm(iv, im);
    }
  }

  // write LSE per (m_local) row
#pragma unroll
  for (int im = 0; im < kM; ++im) {
    int v_local0 = get<0>(tIY_nm(0, im));
    int m_local = get<1>(tIY_nm(0, im));
    if (m_local >= actual_num_heads) {
      continue;
    }
    if (v_local0 == 0) {
      float lse_val;
      if (gMax(im) == -std::numeric_limits<float>::infinity() || warp_sum[im] == 0.f) {
        lse_val = -std::numeric_limits<float>::infinity();
      } else {
        lse_val = gMax(im) + log2f_ftz(warp_sum[im]);
      }
      lse_ptr[row_base + m_local] = lse_val;
    }
  }
}

template <typename Tin, int kTileM, int kTileN, int kTileNope, int kTileRope, int kTileV,
          typename TiledMmaQK, typename TiledMmaSV, typename TmaQNope, typename TmaQRope,
          typename TmaKNope, typename TmaKRope, typename SLayoutQNope, typename SLayoutQRope,
          typename SLayoutKNope, typename SLayoutKRope, typename SLayoutP, typename SLayoutS,
          typename SLayoutV, int kBlockSize, int kStage>
__global__ void __launch_bounds__(256) attention_mla_dim576_persistent_kernel(
    const __grid_constant__ TmaQNope tma_q_nope, const __grid_constant__ TmaQRope tma_q_rope,
    const __grid_constant__ TmaKNope tma_k_nope, const __grid_constant__ TmaKRope tma_k_rope,
    float *y_partial_ptr, float *lse_ptr, const int *block_ids_ptr, const int *cu_seqlens_q_ptr,
    const int *num_seq_kv_ptr, const int *cu_tasks, const int4 *task_list, const int *cu_splits,
    int num_batch, int total_seq_q, int num_head_q, int qk_dim, int v_dim, int num_kvcache_blocks,
    int num_seq_max_blocks, float one_over_dk_log2e) {
  using namespace cute;  // NOLINT

  int isM = blockIdx.x;
  int idx = threadIdx.x;

  constexpr int kMathThreads = size(TiledMmaQK{});
  constexpr int kConsumers = kMathThreads / 128;
  static_assert(kConsumers == 1, "persistent dim576 expects 1 math warpgroup");

  int elected = cute::elect_one_sync();
  int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
  bool is_leader_in_block = (iwarp == 0) && elected;

  __shared__ uint64_t q_readable;
  __shared__ uint64_t q_writable;
  __shared__ uint64_t kv_writable[kStage];
  __shared__ uint64_t kv_readable[kStage];
  __shared__ int shm_task_start;
  __shared__ int shm_task_end;
  extern __shared__ uint8_t shm_data[] alignas(128);

  // sV aliases sK_nope (same trick as the splitk kernel).
  auto *shm_q_nope = reinterpret_cast<Tin *>(shm_data);
  auto *shm_q_rope = shm_q_nope + cosize(SLayoutQNope{});
  auto *shm_k_nope = shm_q_rope + cosize(SLayoutQRope{});
  auto *shm_k_rope = shm_k_nope + cosize(SLayoutKNope{});
  auto *shm_p = shm_k_rope + cosize(SLayoutKRope{});
  auto *shm_max = reinterpret_cast<float *>(shm_p + cosize(SLayoutP{}));

  auto gQ_nope = tma_q_nope.get_tma_tensor(make_shape(num_head_q, Int<kTileNope>{}, total_seq_q));
  auto gQ_rope = tma_q_rope.get_tma_tensor(make_shape(num_head_q, Int<kTileRope>{}, total_seq_q));
  auto gK_nope =
      tma_k_nope.get_tma_tensor(make_shape(kBlockSize, Int<kTileNope>{}, 1, num_kvcache_blocks));
  auto gK_rope =
      tma_k_rope.get_tma_tensor(make_shape(kBlockSize, Int<kTileRope>{}, 1, num_kvcache_blocks));

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
  auto sP = make_tensor(make_smem_ptr(shm_p), SLayoutP{});
  auto sS = make_tensor(make_smem_ptr(shm_p), SLayoutS{});
  auto sV = make_tensor(make_smem_ptr(shm_k_nope), SLayoutV{});

  auto btma_q_nope = tma_q_nope.get_slice(0);
  auto btma_q_rope = tma_q_rope.get_slice(0);
  auto btma_k_nope = tma_k_nope.get_slice(0);
  auto btma_k_rope = tma_k_rope.get_slice(0);

  auto tQng = btma_q_nope.partition_S(gQ_nope);
  auto tQrg = btma_q_rope.partition_S(gQ_rope);
  auto tKng = btma_k_nope.partition_S(gK_nope);
  auto tKrg = btma_k_rope.partition_S(gK_rope);

  auto tQns = btma_q_nope.partition_D(sQ_nope);
  auto tQrs = btma_q_rope.partition_D(sQ_rope);
  auto tKns = btma_k_nope.partition_D(sK_nope);
  auto tKrs = btma_k_rope.partition_D(sK_rope);

  constexpr int kBlockPerTileN = kTileN / kBlockSize;

  if (is_leader_in_block) {
    initialize_barrier(q_readable, 1);
    initialize_barrier(q_writable, 1);
#pragma unroll
    for (int istage = 0; istage < kStage; ++istage) {
      initialize_barrier(kv_writable[istage], kConsumers);
      initialize_barrier(kv_readable[istage], 1);
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

  // Producer (load) warp: idx >= kMathThreads.
  if (idx >= kMathThreads) {
    cutlass::arch::warpgroup_reg_dealloc<24>();

    int load_idx = idx - kMathThreads;
    int load_iwarp = __shfl_sync(0xFFFFFFFF, load_idx / 32, 0);
    bool is_leader_in_load = (load_iwarp == 0) && elected;

    if (is_leader_in_load) {
      int q_phase_w = 1;
      int kv_phase = 1;
      int kv_istage = 0;

#pragma unroll 1
      for (int task_offset = task_start; task_offset < task_end; ++task_offset) {
        int4 t = task_list[task_offset];
        int ibatch = t.x;
        if (ibatch < 0) {
          break;  // sentinel
        }
        int ikv_tile_start = t.z;
        int ikv_tile_end = t.w;

        wait_barrier(q_writable, q_phase_w);
        q_phase_w ^= 1;

        int itoken = cu_seqlens_q_ptr[ibatch];  // decode: == ibatch.

        cute::copy(tma_q_nope.with(q_readable, 0, TMA::CacheHintSm90::EVICT_FIRST),
                   tQng(_, 0, _, itoken), tQns(_, 0, _));
        cute::copy(tma_q_rope.with(q_readable, 0, TMA::CacheHintSm90::EVICT_FIRST),
                   tQrg(_, 0, _, itoken), tQrs(_, 0, _));
        set_barrier_transaction_bytes(
            q_readable, sizeof(Tin) * (cosize(SLayoutQNope{}) + cosize(SLayoutQRope{})));

        const int *batch_block_ids = block_ids_ptr + ibatch * num_seq_max_blocks;
        int num_seq_kvc = num_seq_kv_ptr[ibatch];
        int num_blocks_total = (num_seq_kvc + kBlockSize - 1) / kBlockSize;

#pragma unroll 1
        for (int itile_seq_kv = ikv_tile_end - 1; itile_seq_kv >= ikv_tile_start; --itile_seq_kv) {
          vec_t<int, kBlockPerTileN> kv_block_ids;
#pragma unroll
          for (int ikvblock = 0; ikvblock < kBlockPerTileN; ++ikvblock) {
            int kvblk_id = itile_seq_kv * kBlockPerTileN + ikvblock;
            kv_block_ids[ikvblock] = (kvblk_id < num_blocks_total) ? batch_block_ids[kvblk_id] : -1;
          }

          wait_barrier(kv_writable[kv_istage], kv_phase);
#pragma unroll
          for (int ikvblock = 0; ikvblock < kBlockPerTileN; ++ikvblock) {
            cute::copy(tma_k_nope.with(kv_readable[kv_istage]),
                       tKng(_, 0, _, 0, kv_block_ids[ikvblock]), tKns(_, ikvblock, _, kv_istage));
            cute::copy(tma_k_rope.with(kv_readable[kv_istage]),
                       tKrg(_, 0, _, 0, kv_block_ids[ikvblock]), tKrs(_, ikvblock, _, kv_istage));
          }
          set_barrier_transaction_bytes(kv_readable[kv_istage],
                                        sizeof(Tin) * (kTileN * kTileNope + kTileN * kTileRope));

          kv_istage += 1;
          if (kv_istage == kStage) {
            kv_istage = 0;
            kv_phase ^= 1;
          }
        }
      }
    }
  } else {
    // Math warpgroup: idx < kMathThreads. 128 threads (kConsumers == 1).
    cutlass::arch::warpgroup_reg_alloc<240>();

    int idx_in_warpgroup = idx % 128;
    int iwarpgroup = idx / 128;
    int iwarp_in_warpgroup = idx_in_warpgroup / 32;
    int ilane_in_warpgroup = idx_in_warpgroup % 32;
    int elected_idx_in_warpgroup = ((iwarp_in_warpgroup == 0) && elected);

    TiledMmaQK tiled_mma_qk;
    TiledMmaSV tiled_mma_sv;

    auto thr_mma_qk = tiled_mma_qk.get_slice(idx);
    auto thr_mma_sv = tiled_mma_sv.get_slice(idx);

    auto tKns4r = thr_mma_qk.partition_A(sK_nope);
    auto tKrs4r = thr_mma_qk.partition_A(sK_rope);
    auto tQns4r = thr_mma_qk.partition_B(sQ_nope);
    auto tQrs4r = thr_mma_qk.partition_B(sQ_rope);
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
    auto thr_copy_P_r2s = tiled_copy_P_r2s.get_slice(idx);
    auto tPs4r = thr_copy_P_r2s.partition_D(sP);

    int q_phase_r = 0;
    int kv_phase = 0;
    int kv_istage = 0;

#pragma unroll 1
    for (int task_offset = task_start; task_offset < task_end; ++task_offset) {
      int4 t = task_list[task_offset];
      int ibatch = t.x;
      if (ibatch < 0) {
        break;
      }
      int isplit_in_batch = t.y;
      int ikv_tile_start = t.z;
      int ikv_tile_end = t.w;

      wait_barrier(q_readable, q_phase_r);
      q_phase_r ^= 1;

      clear(gSum);
      fill(gMax, -std::numeric_limits<float>::infinity());
      clear(tYr);
      tiled_mma_sv.accumulate_ = GMMA::ScaleOut::One;

      int num_seq_kvc = num_seq_kv_ptr[ibatch];
      int num_seq_q_local = cu_seqlens_q_ptr[ibatch + 1] - cu_seqlens_q_ptr[ibatch];
      int num_seq_kvcache = num_seq_kvc - num_seq_q_local;
      int itoken_in_batch = 0;
      int num_tile_kv_total = (num_seq_kvcache + itoken_in_batch + 1 + kTileN - 1) / kTileN;
      int iposq = num_seq_kvcache + itoken_in_batch;

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

        // Causal mask only on the global last KV tile.
        if (itile_seq_kv == num_tile_kv_total - 1) {
#pragma unroll
          for (int in = 0; in < kN; ++in) {
#pragma unroll
            for (int im = 0; im < kM; ++im) {
              int iposk = (num_tile_kv_total - 1) * kTileN + get<0>(tI_nm(in, im));
              if (iposk > iposq) {
                tAttr_nm(in, im) = -std::numeric_limits<float>::infinity();
              }
            }
          }
        }

        online_softmax<true, kConsumers, kTileM, kM, kN>(tAttr_nm, gMax, gSum, tYr_nm,
                                                         one_over_dk_log2e, shm_max, iwarpgroup,
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

      // Epilogue
      int isplit_global = cu_splits[ibatch] + isplit_in_batch;
      splitk_epilogue_dim576<kTileM, kTileV, kM, kVN>(
          tYr_nm, gMax, gSum, tIY_nm, shm_max, y_partial_ptr, lse_ptr,
          /*row_base=*/isplit_global * num_head_q, v_dim, iwarp_in_warpgroup, ilane_in_warpgroup,
          /*actual_num_heads=*/num_head_q);

      if (elected_idx_in_warpgroup) {
        arrive_barrier(q_writable);
      }
    }
  }
}

}  // namespace kernels
}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_MLA_SMALLM_MLA_DIM576_PERSISTENT_KERNELS_CUH_
