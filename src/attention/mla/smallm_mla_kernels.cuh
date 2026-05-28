// Copyright 2025 hpc-ops authors

#ifndef SRC_ATTENTION_MLA_SMALLM_MLA_KERNELS_CUH_
#define SRC_ATTENTION_MLA_SMALLM_MLA_KERNELS_CUH_

#include <cuda.h>
#include <stdio.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace attention {
namespace kernels {

template <int kConsumers, int kTileM, int kM, typename TensorY, typename TensorS>
__device__ __forceinline__ void final_online_softmax(TensorY &tYr_nm, TensorS &gSum,
                                                     float *smem_sum, int iwarpgroup, int iwarp,
                                                     int ilane) {
  vec_t<float, kM> warp_sum;

  constexpr int kMaxItemsPerWarpGroup = kTileM / kConsumers;
  constexpr int kWarpsPerWarpGroup = 4;

#pragma unroll
  for (int im = 0; im < kM; ++im) {
    warp_sum[im] = warp_8lane_stride4_reduce_sum_xor(gSum(im));
  }

  if (ilane < 4) {
    if constexpr (kM == 8) {
      auto &warp_sum_16B = reshape<kM / 4, 4>(warp_sum);
      store(smem_sum + iwarp * kTileM + iwarpgroup * kMaxItemsPerWarpGroup + ilane * 4,
            warp_sum_16B[0]);
      store(smem_sum + iwarp * kTileM + iwarpgroup * kMaxItemsPerWarpGroup + 16 + ilane * 4,
            warp_sum_16B[1]);
    } else {
      store(smem_sum + iwarp * kTileM + ilane * kM, warp_sum);
    }
  }

  syncwarpgroup(iwarpgroup);

  if (ilane < 4) {
#pragma unroll
    for (int im = 0; im < kM; ++im) {
      warp_sum[im] = 0.f;
    }
#pragma unroll
    for (int i = 0; i < kWarpsPerWarpGroup; i++) {
      vec_t<float, kM> reduce_sum;
      if constexpr (kM == 8) {
        auto &reduce_sum_16B = reshape<kM / 4, 4>(reduce_sum);
        reduce_sum_16B[0] =
            load<float, 4>(smem_sum + i * kTileM + iwarpgroup * kMaxItemsPerWarpGroup + ilane * 4);
        reduce_sum_16B[1] = load<float, 4>(smem_sum + i * kTileM +
                                           iwarpgroup * kMaxItemsPerWarpGroup + 16 + ilane * 4);
      } else {
        reduce_sum = load<float, kM>(smem_sum + i * kTileM + ilane * kM);
      }
#pragma unroll
      for (int im = 0; im < kM; ++im) {
        warp_sum[im] += reduce_sum[im];
      }
    }
  }

#pragma unroll
  for (int im = 0; im < kM; ++im) {
    warp_sum[im] = __shfl_sync(0xFFFFFFFF, warp_sum[im], ilane % 4);
  }

#pragma unroll
  for (int im = 0; im < kM; ++im) {
    float one_over_gsum = rcpf_ftz(warp_sum[im]);
#pragma unroll
    for (int iv = 0; iv < cute::size<0>(tYr_nm); ++iv) {
      tYr_nm(iv, im) = tYr_nm(iv, im) * one_over_gsum;
    }
  }
}

template <bool kCheckInf, int kConsumers, int kTileM, int kM, int kN, typename TensorA,
          typename TensorM, typename TensorS, typename TensorY>
__device__ __forceinline__ void online_softmax(TensorA &tAttr_nm, TensorM &gMax, TensorS &gSum,
                                               TensorY &tYr_nm, float one_over_dk_log2e,
                                               float *smem_max, int iwarpgroup, int iwarp,
                                               int ilane) {
  vec_t<float, kM> warp_max;

  constexpr int kMaxItemsPerWarpGroup = kTileM / kConsumers;
  constexpr int kWarpsPerWarpGroup = 4;

#pragma unroll
  for (int im = 0; im < kM; ++im) {
    float row_max = tAttr_nm(0, im);

#pragma unroll
    for (int in = 1; in < kN; ++in) {
      row_max = fmaxf(row_max, tAttr_nm(in, im));
    }

    warp_max[im] = warp_8lane_stride4_reduce_max_xor(row_max) * one_over_dk_log2e;
  }

  if (ilane < 4) {
    if constexpr (kM == 8) {
      auto &warp_max_16B = reshape<kM / 4, 4>(warp_max);
      store(smem_max + iwarp * kTileM + iwarpgroup * kMaxItemsPerWarpGroup + ilane * 4,
            warp_max_16B[0]);
      store(smem_max + iwarp * kTileM + iwarpgroup * kMaxItemsPerWarpGroup + 16 + ilane * 4,
            warp_max_16B[1]);
    } else {
      store(smem_max + iwarp * kTileM + ilane * kM, warp_max);
    }
  }

  syncwarpgroup(iwarpgroup);

  if (ilane < 4) {
#pragma unroll
    for (int i = 0; i < kWarpsPerWarpGroup; i++) {
      vec_t<float, kM> reduce_max;
      if constexpr (kM == 8) {
        auto &reduce_max_16B = reshape<kM / 4, 4>(reduce_max);
        reduce_max_16B[0] =
            load<float, 4>(smem_max + i * kTileM + iwarpgroup * kMaxItemsPerWarpGroup + ilane * 4);
        reduce_max_16B[1] = load<float, 4>(smem_max + i * kTileM +
                                           iwarpgroup * kMaxItemsPerWarpGroup + 16 + ilane * 4);
      } else {
        reduce_max = load<float, kM>(smem_max + i * kTileM + ilane * kM);
      }
#pragma unroll
      for (int im = 0; im < kM; ++im) {
        warp_max[im] = fmax(reduce_max[im], warp_max[im]);
      }
    }
  }
#pragma unroll
  for (int im = 0; im < kM; ++im) {
    warp_max[im] = __shfl_sync(0xFFFFFFFF, warp_max[im], ilane % 4);
  }

#pragma unroll
  for (int im = 0; im < kM; ++im) {
    float last_max = gMax(im);
    float row_max = fmaxf(last_max, warp_max[im]);
    float row_sum = 0.f;

    gMax(im) = row_max;

    if constexpr (kCheckInf) {
      if (gMax(im) == -std::numeric_limits<float>::infinity()) {
#pragma unroll
        for (int in = 0; in < kN; ++in) {
          tAttr_nm(in, im) = 0.f;
        }
        continue;
      }
    }

#pragma unroll
    for (int in = 0; in < kN; ++in) {
      tAttr_nm(in, im) = exp2f_ftz(tAttr_nm(in, im) * one_over_dk_log2e - gMax(im));
      row_sum += tAttr_nm(in, im);
    }

    float scale = exp2f_ftz(last_max - gMax(im));
    gSum(im) = gSum(im) * scale + row_sum;

#pragma unroll
    for (int iv = 0; iv < cute::size<0>(tYr_nm); ++iv) {
      tYr_nm(iv, im) = tYr_nm(iv, im) * scale;
    }
  }
}

template <typename Tout, typename Tin, int kTileM, int kTileN, int kTileK, int kTileV,
          typename TiledMmaQK, typename TiledMmaSV, typename TmaQ, typename TmaKV, typename TmaY,
          typename SLayoutQ, typename SLayoutK, typename SLayoutP, typename SLayoutS,
          typename SLayoutV, typename SLayoutY, int kBlockSize, int kStage>
__global__ void __launch_bounds__(384, 1)
    attention_mla_with_kvcache_bf16_multistage_ws_smallm_kernel(
        const __grid_constant__ TmaQ tma_q, const __grid_constant__ TmaKV tma_kv,
        const __grid_constant__ TmaY tma_y, const int *block_ids_ptr, const int *cu_seqlens_q_ptr,
        const int *num_seq_kv_ptr, int num_batch, int total_seq_q, int num_head_q, int head_dim,
        int num_kvcache_blocks, int num_seq_max_blocks, float one_over_dk_log2e) {
  using namespace cute;  // NOLINT

  int idx = threadIdx.x;
  int itile_head = blockIdx.x;
  int itoken = blockIdx.y;

  int ibatch = 0;
  int itoken_in_batch = itoken;

  for (int i = 1; i < num_batch + 1; i++) {
    int cu_seqlenq = cu_seqlens_q_ptr[i];
    if (itoken < cu_seqlenq) {
      ibatch = i - 1;
      itoken_in_batch = itoken - cu_seqlens_q_ptr[ibatch];
      break;
    }
  }

  constexpr int kMathThreads = size(TiledMmaQK{});
  constexpr int kConsumers = kMathThreads / 128;
  constexpr int kWarpsPerWrapGroup = 4;

  int elected = cute::elect_one_sync();
  int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
  bool is_leader_in_block = (iwarp == 0) && elected;

  int num_seq_kv = num_seq_kv_ptr[ibatch];
  int num_seq_q = cu_seqlens_q_ptr[ibatch + 1] - cu_seqlens_q_ptr[ibatch];
  int num_seq_kvcache = num_seq_kv - num_seq_q;

  if (num_seq_kv <= 0) {
    return;
  }

  int num_blocks = (num_seq_kv + kBlockSize - 1) / kBlockSize;

  const int *block_ids = block_ids_ptr + ibatch * num_seq_max_blocks;

  __shared__ uint64_t q_readable;
  __shared__ uint64_t kv_writable[kStage];
  __shared__ uint64_t kv_readable[kStage];
  extern __shared__ uint8_t shm_data[] alignas(128);

  auto *shm_q = reinterpret_cast<Tin *>(shm_data);
  auto *shm_kv = shm_q + cosize(SLayoutQ{});
  auto *shm_p = shm_kv + cosize(SLayoutV{});
  auto *shm_max = reinterpret_cast<float *>(shm_p + cosize(SLayoutP{}));
  int *shm_kvblk_ids = reinterpret_cast<int *>(shm_max + kTileM * kWarpsPerWrapGroup);
  auto *shm_y = reinterpret_cast<Tout *>(shm_data);  // Reuse All

  // Tensor Q/K/V/Y
  auto gQ = tma_q.get_tma_tensor(make_shape(num_head_q, head_dim, total_seq_q));
  auto gKV = tma_kv.get_tma_tensor(make_shape(kBlockSize, head_dim, 1, num_kvcache_blocks));
  auto gY = tma_y.get_tma_tensor(make_shape(head_dim, num_head_q, total_seq_q));

  auto gAtt =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileN>{}, Int<kTileM>{}), make_stride(Int<kTileM>{}, Int<1>{}));
  auto gYY =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileV>{}, Int<kTileM>{}), make_stride(Int<1>{}, Int<kTileV>{}));

  // Tensor sQ/sK/sV
  auto sQ = make_tensor(make_smem_ptr(shm_q), SLayoutQ{});
  auto sK = make_tensor(make_smem_ptr(shm_kv), SLayoutK{});
  auto sP = make_tensor(make_smem_ptr(shm_p), SLayoutP{});
  auto sS = make_tensor(make_smem_ptr(shm_p), SLayoutS{});
  auto sV = make_tensor(make_smem_ptr(shm_kv), SLayoutV{});
  auto sY = make_tensor(make_smem_ptr(shm_y), SLayoutY{});

  // Block Level tma
  auto btma_q = tma_q.get_slice(0);
  auto btma_kv = tma_kv.get_slice(0);
  auto btma_y = tma_y.get_slice(0);

  // Thread Level Tensor
  auto tQg = btma_q.partition_S(gQ);     // (TMA, TMA_M, TMA_K, seqlenq, head_kv, batch)
  auto tKVg = btma_kv.partition_S(gKV);  // (TMA, TMA_N, TMA_K, head_kv, batch)

  auto tQs = btma_q.partition_D(sQ);    // (TMA, _1, _1)
  auto tKVs = btma_kv.partition_D(sK);  // (TMA, _1, _1)

  int num_tile_kv = (num_seq_kvcache + itoken_in_batch + 1 + kTileN - 1) / kTileN;
  // int num_tile_kv = (640 + kTileM - 1) / kTileM;

  constexpr int kBlockPerTileN = kTileN / kBlockSize;

  // init bar
  if (is_leader_in_block) {
    initialize_barrier(q_readable, 1);
#pragma unroll
    for (int istage = 0; istage < kStage; istage++) {
      initialize_barrier(kv_writable[istage], kConsumers);
      initialize_barrier(kv_readable[istage], 1);
    }
  }

  // sync to avoid ahead thread use(wait) readable when it is not initizlized yet
  __syncthreads();
  // load warpgroup
  if (idx >= kMathThreads) {
    cutlass::arch::warpgroup_reg_dealloc<24>();
    bool is_leader_in_load = ((iwarp == kMathThreads / 32) && elected);

    if (is_leader_in_load) {
      // Load Q
      cute::copy(tma_q.with(q_readable, 0, TMA::CacheHintSm90::EVICT_FIRST),
                 tQg(_, itile_head, _, itoken), tQs(_, 0, _));
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

    bool is_leader_in_load = ((iwarp == 0) && elected);
    int phase = 1;

    if (is_leader_in_load) {
      int istage = 0;

      // Load KV
#pragma unroll 1
      for (int itile_seq_kv = num_tile_kv - 1; itile_seq_kv >= 0; --itile_seq_kv) {
        vec_t<int, kBlockPerTileN> kv_block_ids;
#pragma unroll
        for (int ikvblock = 0; ikvblock < kBlockPerTileN; ikvblock++) {
          int kvblk_id = itile_seq_kv * kBlockPerTileN + ikvblock;
          if (kvblk_id < num_blocks) {
            kv_block_ids[ikvblock] = shm_kvblk_ids[kvblk_id];
          } else {
            kv_block_ids[ikvblock] = -1;
          }
        }

        wait_barrier(kv_writable[istage], phase);
#pragma unroll
        for (int ikvblock = 0; ikvblock < kBlockPerTileN; ikvblock++) {
          cute::copy(tma_kv.with(kv_readable[istage]), tKVg(_, 0, _, 0, kv_block_ids[ikvblock]),
                     tKVs(_, ikvblock, _, istage));
        }
        set_barrier_transaction_bytes(kv_readable[istage], sizeof(Tin) * kTileN * kTileK);

        istage++;

        if (istage == kStage) {
          istage = 0;
          phase ^= 1;
        }
      }
    }
  } else {
    cutlass::arch::warpgroup_reg_alloc<232>();
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

    clear(gSum);
    fill(gMax, -std::numeric_limits<float>::infinity());

    using R2SCopyAtomP = Copy_Atom<cute::SM90_U16x8_STSM_T, Tin>;
    auto tiled_copy_P_r2s = make_tiled_copy_C(R2SCopyAtomP{}, tiled_mma_qk);
    auto thr_copy_P_r2s = tiled_copy_P_r2s.get_slice(idx);
    auto tPs4r = thr_copy_P_r2s.partition_D(sP);

    clear(tYr);

    tiled_mma_sv.accumulate_ = GMMA::ScaleOut::One;

    wait_barrier(q_readable, 0);

    int phase = 0;
    int istage_read = 0;
    int iposq = num_seq_kvcache + itoken_in_batch;
    // compute casual
#pragma unroll 1
    for (int itile_seq_kv = num_tile_kv - 1; itile_seq_kv >= 0; --itile_seq_kv) {
      wait_barrier(kv_readable[istage_read], phase);

      // P = QK
      tiled_mma_qk.accumulate_ = GMMA::ScaleOut::Zero;
      warpgroup_fence_operand(tAttr);
      warpgroup_arrive();
#pragma unroll
      for (int ik = 0; ik < size<2>(tQr); ++ik) {
        cute::gemm(tiled_mma_qk, tKr(_, _, ik, istage_read), tQr(_, _, ik), tAttr(_, _, _));
        tiled_mma_qk.accumulate_ = GMMA::ScaleOut::One;
      }
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      warpgroup_fence_operand(tAttr);

      if (itile_seq_kv == num_tile_kv - 1) {
        // do causal mask
#pragma unroll
        for (int in = 0; in < kN; ++in) {
#pragma unroll
          for (int im = 0; im < kM; ++im) {
            int iposk = (num_tile_kv - 1) * kTileN + get<0>(tI_nm(in, im));

            if (iposk > iposq) {
              tAttr_nm(in, im) = -std::numeric_limits<float>::infinity();
            }
          }
        }
      }

      // online softmax
      online_softmax<false, kConsumers, kTileM, kM, kN>(tAttr_nm, gMax, gSum, tYr_nm,
                                                        one_over_dk_log2e, shm_max, iwarpgroup,
                                                        iwarp_in_warpgroup, ilane_in_warpgroup);
      // Y = PV
      auto tAttAbf16 = make_tensor_like<cute::bfloat16_t>(tAttr);
#pragma unroll
      for (int i = 0; i < size(tAttr); ++i) {
        tAttAbf16(i) = static_cast<cute::bfloat16_t>(tAttr(i));
      }

      auto tPr4s = thr_copy_P_r2s.retile_S(tAttAbf16);
      cute::copy(tiled_copy_P_r2s, tPr4s, tPs4r);

      cutlass::arch::fence_view_async_shared();
      syncwarpgroup(iwarpgroup);

      warpgroup_fence_operand(tYr);
      warpgroup_arrive();
      cute::gemm(tiled_mma_sv, tVr(_, _, _, istage_read), tSr, tYr);
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      warpgroup_fence_operand(tYr);

      if (elected_idx_in_warpgroup) {
        arrive_barrier(kv_writable[istage_read]);
      }
      istage_read++;
      if (istage_read == kStage) {
        istage_read = 0;
        phase ^= 1;
      }
      syncwarpgroup(iwarpgroup);
    }

    // final online softmax
    final_online_softmax<kConsumers, kTileM, kM>(tYr_nm, gSum, shm_max, iwarpgroup,
                                                 iwarp_in_warpgroup, ilane_in_warpgroup);
    // to bfloat16

    auto tYr_bf16 = make_tensor_like<Tout>(tYr);

#pragma unroll
    for (int i = 0; i < size(tYr); ++i) {
      Tout v{tYr(i)};
      tYr_bf16(i) = v;
    }

    // Epilogue: write register-C to global memory
    using R2SCopyAtomC = Copy_Atom<cute::SM90_U16x8_STSM_T, Tout>;
    auto r2s_tiled_copy = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma_sv);
    auto r2s_thr_copy = r2s_tiled_copy.get_slice(idx);

    auto tYr4s = r2s_thr_copy.retile_S(tYr_bf16);
    auto tYs4r = r2s_thr_copy.partition_D(sY);

    bar_sync<kConsumers * 128>(kConsumers);
    cute::copy(r2s_tiled_copy, tYr4s, tYs4r);
    tma_store_fence();
    bar_sync<kConsumers * 128>(kConsumers);
    // using TMA to store
    if (is_leader_in_warpgroup && iwarpgroup == 0) {
      auto tYss = btma_y.partition_S(sY);  // (TMA, TMA_M, TMA_N)
      auto tYgg = btma_y.partition_D(gY);  // (TMA, TMA_M, TMA_N, b)

      cute::copy(tma_y, tYss(_, _, 0), tYgg(_, _, itile_head, itoken));
    }
  }
}

}  // namespace kernels
}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_MLA_SMALLM_MLA_KERNELS_CUH_
