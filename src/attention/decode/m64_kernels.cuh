// Copyright (C) 2026 Tencent.

#ifndef SRC_ATTENTION_DECODE_M64_KERNELS_CUH_
#define SRC_ATTENTION_DECODE_M64_KERNELS_CUH_

#include <cuda.h>
#include <stdio.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "cutlass/arch/reg_reconfig.h"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace attention {
namespace decode {
namespace kernels {

template <typename TensorY, typename TensorS>
__device__ __forceinline__ void final_online_softmax(TensorY &tYr_mn, TensorS &gSum, int kM) {
#pragma unroll
  for (int im = 0; im < kM; ++im) {
    gSum(im) = warp_4lane_reduce_sum_xor(gSum(im));

    float one_over_gsum = rcpf_ftz(gSum(im));
#pragma unroll
    for (int in = 0; in < cute::size<1>(tYr_mn); ++in) {
      tYr_mn(im, in) = tYr_mn(im, in) * one_over_gsum;
    }
  }
}

template <bool kCheckInf, typename TensorA, typename TensorM, typename TensorS, typename TensorY>
__device__ __forceinline__ void online_softmax(TensorA &tAttr_mn, TensorM &gMax, TensorS &gSum,
                                               TensorY &tYr_mn, int kM, int kN,
                                               float one_over_dk_log2e) {
#pragma unroll
  for (int im = 0; im < kM; ++im) {
    float row_max = tAttr_mn(im, 0);
    float row_sum = 0.f;

#pragma unroll
    for (int in = 1; in < kN; ++in) {
      row_max = fmaxf(row_max, tAttr_mn(im, in));
    }

    row_max = warp_4lane_reduce_max_xor(row_max) * one_over_dk_log2e;
    float last_max = gMax(im);
    gMax(im) = fmaxf(last_max, row_max);

    if constexpr (kCheckInf) {
      if (gMax(im) == -std::numeric_limits<float>::infinity()) {
#pragma unroll
        for (int in = 0; in < kN; ++in) {
          tAttr_mn(im, in) = 0.f;
        }
        continue;
      }
    }

#pragma unroll
    for (int in = 0; in < kN; ++in) {
      tAttr_mn(im, in) = exp2f_ftz(tAttr_mn(im, in) * one_over_dk_log2e - gMax(im));
      row_sum += tAttr_mn(im, in);
    }

    float scale = exp2f_ftz(last_max - gMax(im));
    gSum(im) = gSum(im) * scale + row_sum;

#pragma unroll
    for (int in = 0; in < cute::size<1>(tYr_mn); ++in) {
      tYr_mn(im, in) = tYr_mn(im, in) * scale;
    }
  }
}

template <bool kCheckBound, int kBlockPerTileN, int kBlockSize, int kStage, typename Tin,
          typename TmaK, typename TmaV, typename TensorGK, typename TensorSK, typename TensorGV,
          typename TensorSV>
__device__ __forceinline__ void load_paged_kv(TmaK &tma_k, TmaV &tma_v, uint64_t *bar_k_writable,
                                              uint64_t *bar_v_writable, uint64_t *bar_k_readable,
                                              uint64_t *bar_v_readable, TensorGK &tKg,
                                              TensorSK &tKs, TensorGV &tVg, TensorSV &tVs,
                                              int ihead_kv, int num_dim_qk, int num_dim_v,
                                              int *block_ids, int nblocks, int itile,
                                              int istage_write, int phase) {
  using namespace cute;  // NOLINT

  int load_blocks = kBlockPerTileN;
  int istage = istage_write;

  wait_barrier(bar_k_writable[istage], phase);
#pragma unroll
  for (int ikvblock = 0; ikvblock < kBlockPerTileN; ikvblock++) {
    int kvblk_id = itile * kBlockPerTileN + ikvblock;
    int blk_id = -1;
    if constexpr (kCheckBound) {
      if (kvblk_id < nblocks) {
        blk_id = block_ids[kvblk_id];
      }
    } else {
      blk_id = block_ids[kvblk_id];
    }
    cute::copy(tma_k.with(bar_k_readable[istage]), tKg(_, 0, _, ihead_kv, blk_id),
               tKs(_, ikvblock, _, istage));
  }
  set_barrier_transaction_bytes(bar_k_readable[istage],
                                sizeof(Tin) * load_blocks * kBlockSize * num_dim_qk);

  wait_barrier(bar_v_writable[istage], phase);
  // v
#pragma unroll
  for (int ikvblock = 0; ikvblock < kBlockPerTileN; ikvblock++) {
    int kvblk_id = itile * kBlockPerTileN + ikvblock;
    int blk_id = -1;
    if constexpr (kCheckBound) {
      if (kvblk_id < nblocks) {
        blk_id = block_ids[kvblk_id];
      }
    } else {
      blk_id = block_ids[kvblk_id];
    }
    cute::copy(tma_v.with(bar_v_readable[istage]), tVg(_, _, 0, ihead_kv, blk_id),
               tVs(_, _, ikvblock, istage));
  }
  set_barrier_transaction_bytes(bar_v_readable[istage],
                                sizeof(Tin) * load_blocks * kBlockSize * num_dim_v);
}

template <typename Tout, typename Tin, int kTileM, int kTileN, int kTileK, int kTileV,
          typename TiledMmaQK, typename TiledMmaPV, typename TmaQ, typename TmaK, typename TmaV,
          typename TmaY, typename SLayoutQ, typename SLayoutK, typename SLayoutV, typename SLayoutY,
          int kBlockSize, int kStage>
__global__ void attention_decode_bf16_multistage_ws_kernel(
    const __grid_constant__ TmaQ tma_q, const __grid_constant__ TmaK tma_k,
    const __grid_constant__ TmaV tma_v, const __grid_constant__ TmaY tma_y,
    const int *block_ids_ptr, const int *num_seq_kvcache_ptr, bool new_kv_included, int num_batch,
    int num_dim_qk, int num_dim_v, int num_head_q, int num_head_k, int num_head_v,
    int heads_per_group, int num_kvcache_blocks, int num_seq_max_blocks, float one_over_dk_log2e) {
  using namespace cute;  // NOLINT

  int idx = threadIdx.x;
  int ihead_kv = blockIdx.x;
  int ibatch = blockIdx.y;

  constexpr int kMathThreads = size(TiledMmaQK{});

  int elected = cute::elect_one_sync();
  int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
  bool is_leader_in_block = (iwarp == 0) && elected;

  constexpr int kSeqlenQ = 1;
  int num_seq_kvcache = num_seq_kvcache_ptr[ibatch];
  if (new_kv_included) {
    num_seq_kvcache -= kSeqlenQ;
  }
  int seqlenk = kSeqlenQ + num_seq_kvcache;

  if (seqlenk <= 0) {
    return;
  }

  int nblocks = (seqlenk + kBlockSize - 1) / kBlockSize;

  const int *block_ids = block_ids_ptr + ibatch * num_seq_max_blocks;

  __shared__ uint64_t bar_q;
  __shared__ uint64_t bar_k_writable[kStage];
  __shared__ uint64_t bar_v_writable[kStage];
  __shared__ uint64_t bar_k_readable[kStage];
  __shared__ uint64_t bar_v_readable[kStage];
  extern __shared__ uint8_t shm_data[] alignas(128);

  auto *shm_q = reinterpret_cast<Tin *>(shm_data);
  auto *shm_k = shm_q + cosize(SLayoutQ{});
  auto *shm_v = shm_k + cosize(SLayoutK{});
  int *shm_kvblk_ids = reinterpret_cast<int *>(shm_v + cosize(SLayoutV{}));
  auto *shm_y = reinterpret_cast<Tout *>(shm_data);  // Reuse All

  for (int i = idx; i < nblocks; i += blockDim.x) {
    shm_kvblk_ids[i] = block_ids[i];
  }

  // Tensor Q/K/V/Y
  auto gQ = tma_q.get_tma_tensor(make_shape(num_head_q, num_dim_qk, num_batch));
  auto gK =
      tma_k.get_tma_tensor(make_shape(kBlockSize, num_dim_qk, num_head_k, num_kvcache_blocks));
  auto gV = tma_v.get_tma_tensor(make_shape(num_dim_v, kBlockSize, num_head_v, num_kvcache_blocks));
  auto gY = tma_y.get_tma_tensor(make_shape(num_head_q, num_dim_v, num_batch));

  auto gAtt =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileN>{}), make_stride(Int<kTileN>{}, Int<1>{}));
  auto gYY =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileV>{}), make_stride(Int<kTileV>{}, Int<1>{}));

  // Tensor sQ/sK/sV
  auto sQ = make_tensor(make_smem_ptr(shm_q), SLayoutQ{});
  auto sK = make_tensor(make_smem_ptr(shm_k), SLayoutK{});
  auto sV = make_tensor(make_smem_ptr(shm_v), SLayoutV{});
  auto sY = make_tensor(make_smem_ptr(shm_y), SLayoutY{});

  // Block Level tma
  auto btma_q = tma_q.get_slice(0);
  auto btma_k = tma_k.get_slice(0);
  auto btma_v = tma_v.get_slice(0);
  auto btma_y = tma_y.get_slice(0);

  // Thread Level Tensor
  auto tQg = btma_q.partition_S(gQ);  // (TMA, TMA_M, TMA_K, seqlenq, head_kv, batch)
  auto tKg = btma_k.partition_S(gK);  // (TMA, TMA_N, TMA_K, head_kv, batch)
  auto tVg = btma_v.partition_S(gV);  // (TMA, TMA_V, TMA_N, head_kv, batch)

  auto tQs = btma_q.partition_D(sQ);  // (TMA, _1, _1)
  auto tKs = btma_k.partition_D(sK);  // (TMA, _1, _1)
  auto tVs = btma_v.partition_D(sV);  // (TMA, _1, _1)

  int num_tile_kv = (seqlenk + kTileN - 1) / kTileN;
  // int num_tile_causal = (seqlenq + kTileN - 1) / kTileN;
  int num_tile_full = (num_seq_kvcache + kTileN - 1) / kTileN;
  int num_tile_causal = num_tile_kv - num_tile_full + (num_seq_kvcache % kTileN != 0);
  num_tile_full = num_tile_kv - num_tile_causal;

  constexpr int kBlockPerTileN = kTileN / kBlockSize;

  // init bar
  if (is_leader_in_block) {
    initialize_barrier(bar_q, 1);
#pragma unroll
    for (int istage = 0; istage < kStage; istage++) {
      initialize_barrier(bar_k_writable[istage], 1);
      initialize_barrier(bar_v_writable[istage], 1);
      initialize_barrier(bar_k_readable[istage], 1);
      initialize_barrier(bar_v_readable[istage], 1);
    }
  }

  // sync to avoid ahead thread use(wait) readable when it is not initizlized yet
  __syncthreads();

  // load warpgroup
  if (idx >= kMathThreads) {
    // cutlass::arch::warpgroup_reg_dealloc<24>();
    idx -= kMathThreads;
    iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
    bool is_leader_in_load = ((iwarp == 0) && elected);

    int phase = 1;
    int iload_tile = 0;

    if (is_leader_in_load) {
      // Load Q
      cute::copy(tma_q.with(bar_q), tQg(_, ihead_kv, _, ibatch), tQs(_, 0, _));
      set_barrier_transaction_bytes(
          bar_q, sizeof(Tin) * max(heads_per_group, size<0, 0, 1>(tQg)) * num_dim_qk);

      int istage_write = 0;
      // Load Causal KV
#pragma unroll 1
      for (int itile_seq_kv = num_tile_full; itile_seq_kv < num_tile_kv; ++itile_seq_kv) {
        // load k/scale/v
        load_paged_kv<true, kBlockPerTileN, kBlockSize, kStage, Tin>(
            tma_k, tma_v, bar_k_writable, bar_v_writable, bar_k_readable, bar_v_readable, tKg, tKs,
            tVg, tVs, ihead_kv, num_dim_qk, num_dim_v, shm_kvblk_ids, nblocks, itile_seq_kv,
            istage_write++, phase);
        if (istage_write == kStage) {
          istage_write = 0;
          phase ^= 1;
        }
      }

      // Load Full KV
#pragma unroll 1
      for (int itile_seq_kv = -kStage + 1; itile_seq_kv < num_tile_full; ++itile_seq_kv) {
        if (iload_tile < num_tile_full) {
          load_paged_kv<false, kBlockPerTileN, kBlockSize, kStage, Tin>(
              tma_k, tma_v, bar_k_writable, bar_v_writable, bar_k_readable, bar_v_readable, tKg,
              tKs, tVg, tVs, ihead_kv, num_dim_qk, num_dim_v, shm_kvblk_ids, nblocks, iload_tile++,
              istage_write++, phase);
          if (istage_write == kStage) {
            istage_write = 0;
            phase ^= 1;
          }
        }
      }
    }
  } else {
    // cutlass::arch::warpgroup_reg_alloc<232>();
    // math warpgroup
    int idx_in_warpgroup = idx % 128;
    int iwarpgroup = idx / 128;
    int iwarp_in_warpgroup = idx_in_warpgroup / 32;
    int elected_idx_in_warpgroup = ((iwarp_in_warpgroup == 0) && elected);
    bool is_leader_in_warpgroup = ((iwarp % 4) == 0) && elected;

    TiledMmaQK tiled_mma_qk;
    TiledMmaPV tiled_mma_pv;

    auto thr_mma_qk = tiled_mma_qk.get_slice(idx);
    auto thr_mma_pv = tiled_mma_pv.get_slice(idx);

    auto tQs4r = thr_mma_qk.partition_A(sQ);
    auto tKs4r = thr_mma_qk.partition_B(sK);
    auto tVs4r = thr_mma_pv.partition_B(sV);

    auto tQr = thr_mma_qk.make_fragment_A(tQs4r);  // (MMA, MMA_M, MMA_K)
    auto tKr = thr_mma_qk.make_fragment_B(tKs4r);  // (MMA, MMA_N, MMA_K)
    auto tVr = thr_mma_pv.make_fragment_B(tVs4r);  // (MMA, MMA_V, MMA_N)

    auto tAttr = thr_mma_qk.partition_fragment_C(gAtt);
    auto tYr = thr_mma_pv.partition_fragment_C(gYY);

    auto gI = make_identity_tensor(gAtt.shape());
    auto tI = thr_mma_qk.partition_C(gI);

    auto tAttr_mn = retile_fragment(tAttr);
    constexpr int kM = size<0>(tAttr_mn);
    constexpr int kN = size<1>(tAttr_mn);
    Tensor gMax = make_tensor<float>(Int<kM>{});
    Tensor gSum = make_tensor<float>(Int<kM>{});
    constexpr int kGmmaK = kTileN / size<2>(tVr);

    clear(gSum);
    fill(gMax, -std::numeric_limits<float>::infinity());

    auto layout_asC = thr_mma_qk.partition_C(gAtt).layout();
    auto layout_asA = thr_mma_pv.partition_A(gAtt).layout();
    auto tAttA = make_tensor(tAttr.data(), left_inverse(layout_asC).compose(layout_asA));

    clear(tYr);

    tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;

    wait_barrier(bar_q, 0);

    int phase = 0;
    int istage_read = 0;
    // compute casual
#pragma unroll 1
    for (int itile_seq_kv = num_tile_full; itile_seq_kv < num_tile_kv; ++itile_seq_kv) {
      wait_barrier(bar_k_readable[istage_read], phase);

      // P = QK
      tiled_mma_qk.accumulate_ = GMMA::ScaleOut::Zero;
      warpgroup_fence_operand(tAttr);
      warpgroup_arrive();
#pragma unroll
      for (int ik = 0; ik < size<2>(tQr); ++ik) {
        cute::gemm(tiled_mma_qk, tQr(_, _, ik), tKr(_, _, ik, istage_read), tAttr(_, _, _));
        tiled_mma_qk.accumulate_ = GMMA::ScaleOut::One;
      }
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      warpgroup_fence_operand(tAttr);

      if (elected_idx_in_warpgroup) {
        arrive_barrier(bar_k_writable[istage_read]);
      }

      // do causal mask
      auto tAttr_mn = retile_fragment(tAttr);
      auto tI_mn = retile_fragment(tI);
#pragma unroll
      for (int im = 0; im < kM; ++im) {
#pragma unroll
        for (int in = 0; in < kN; ++in) {
          int iposq = num_seq_kvcache + get<0>(tI_mn(im, in)) / heads_per_group;
          int iposk = itile_seq_kv * kTileN + get<1>(tI_mn(im, in));

          if ((iposk > iposq) || (iposk >= seqlenk)) {
            tAttr_mn(im, in) = -std::numeric_limits<float>::infinity();
          }
        }
      }

      auto tYr_mn = retile_fragment(tYr);
      // online softmax
      online_softmax<true>(tAttr_mn, gMax, gSum, tYr_mn, kM, kN, one_over_dk_log2e);

      // Y = PV
      auto tAttAbf16 = make_tensor_like<cute::bfloat16_t>(tAttA);
#pragma unroll
      for (int i = 0; i < size(tAttA); ++i) {
        tAttAbf16(i) = (cute::bfloat16_t)(tAttA(i));
      }

      wait_barrier(bar_v_readable[istage_read], phase);

      warpgroup_fence_operand(tYr);
      warpgroup_arrive();
      cute::gemm(tiled_mma_pv, tAttAbf16, tVr(_, _, _, istage_read), tYr);
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      warpgroup_fence_operand(tYr);

      if (elected_idx_in_warpgroup) {
        arrive_barrier(bar_v_writable[istage_read]);
      }
      istage_read++;
      if (istage_read == kStage) {
        istage_read = 0;
        phase ^= 1;
      }
      asm volatile("barrier.sync %0, 128;\n" ::"r"(iwarpgroup) : "memory");
    }

    // compute full
#pragma unroll 1
    for (int itile_seq_kv = 0; itile_seq_kv < num_tile_full; ++itile_seq_kv) {
      wait_barrier(bar_k_readable[istage_read], phase);

      // P = QK
      tiled_mma_qk.accumulate_ = GMMA::ScaleOut::Zero;

      warpgroup_fence_operand(tAttr);
      warpgroup_arrive();
#pragma unroll
      for (int ik = 0; ik < size<2>(tQr); ++ik) {
        cute::gemm(tiled_mma_qk, tQr(_, _, ik), tKr(_, _, ik, istage_read), tAttr(_, _, _));
        tiled_mma_qk.accumulate_ = GMMA::ScaleOut::One;
      }
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      warpgroup_fence_operand(tAttr);

      if (elected_idx_in_warpgroup) {
        arrive_barrier(bar_k_writable[istage_read]);
      }

      auto tYr_mn = retile_fragment(tYr);
      // online softmax
      online_softmax<false>(tAttr_mn, gMax, gSum, tYr_mn, kM, kN, one_over_dk_log2e);
      // Y = PV
      auto tAttAbf16 = make_tensor_like<cute::bfloat16_t>(tAttA);
#pragma unroll
      for (int i = 0; i < size(tAttA); ++i) {
        tAttAbf16(i) = (cute::bfloat16_t)(tAttA(i));
      }

      wait_barrier(bar_v_readable[istage_read], phase);

      warpgroup_fence_operand(tYr);
      warpgroup_arrive();
      cute::gemm(tiled_mma_pv, tAttAbf16, tVr(_, _, _, istage_read), tYr);
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      warpgroup_fence_operand(tYr);

      if (elected_idx_in_warpgroup) {
        arrive_barrier(bar_v_writable[istage_read]);
      }
      istage_read++;
      if (istage_read == kStage) {
        istage_read = 0;
        phase ^= 1;
      }
      asm volatile("barrier.sync %0, 128;\n" ::"r"(iwarpgroup) : "memory");
    }

    auto tYr_mn = retile_fragment(tYr);
    // final online softmax
    final_online_softmax(tYr_mn, gSum, kM);

    // to bfloat16
    auto tYr_bf16 = make_tensor_like<Tout>(tYr);

#pragma unroll
    for (int i = 0; i < size(tYr); ++i) {
      Tout v{tYr(i)};
      tYr_bf16(i) = v;
    }

    // Epilogue: write register-C to global memory
    using R2SCopyAtomC = Copy_Atom<cute::SM90_U32x4_STSM_N, Tout>;
    auto r2s_tiled_copy = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma_pv);
    auto r2s_thr_copy = r2s_tiled_copy.get_slice(idx);

    auto tYr4s = r2s_thr_copy.retile_S(tYr_bf16);
    auto tYs4r = r2s_thr_copy.partition_D(sY);

    cute::copy(r2s_tiled_copy, tYr4s, tYs4r);
    asm volatile("barrier.sync %0, 128;\n" ::"r"(iwarpgroup) : "memory");
    tma_store_fence();
    // using TMA to store
    if (is_leader_in_warpgroup) {
      auto tYss = btma_y.partition_S(sY);  // (TMA, TMA_M, TMA_N)
      auto tYgg = btma_y.partition_D(gY);  // (TMA, TMA_M, TMA_N, b)
      cute::copy(tma_y, tYss(_, 0, _), tYgg(_, ihead_kv, _, ibatch));
    }
  }
}

template <typename Tout, typename Tin, int kTileM, int kTileN, int kTileK, int kTileV,
          typename TiledMmaQK, typename TiledMmaPV, typename TmaQ, typename TmaK, typename TmaV,
          typename TmaY, typename SLayoutQ, typename SLayoutK, typename SLayoutV, typename SLayoutY,
          int kBlockSize, int kStage>
__global__ void attention_decode_bf16_onestage_kernel(
    const __grid_constant__ TmaQ tma_q, const __grid_constant__ TmaK tma_k,
    const __grid_constant__ TmaV tma_v, const __grid_constant__ TmaY tma_y,
    const int *block_ids_ptr, const int *num_seq_kvcache_ptr, int num_batch, int num_dim_qk,
    int num_dim_v, int num_head_q, int num_head_k, int num_head_v, int heads_per_group,
    int num_kvcache_blocks, int num_seq_max_blocks, float one_over_dk_log2e) {
  using namespace cute;  // NOLINT

  int idx = threadIdx.x;
  int ihead_kv = blockIdx.x;
  int ibatch = blockIdx.y;

  int elected = cute::elect_one_sync();
  int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
  bool is_leader_in_block = (iwarp == 0) && elected;

  constexpr int kSeqlenQ = 1;
  int num_seq_kvcache = num_seq_kvcache_ptr[ibatch];
  int seqlenk = kSeqlenQ + num_seq_kvcache;
  int nblocks = (seqlenk + kBlockSize - 1) / kBlockSize;
  const int *block_ids = block_ids_ptr + ibatch * num_seq_max_blocks;

  __shared__ uint64_t bar_q;
  __shared__ uint64_t bar_k;
  __shared__ uint64_t bar_v;
  extern __shared__ uint8_t shm_data[] alignas(128);

  auto *shm_q = reinterpret_cast<Tin *>(shm_data);
  auto *shm_k = shm_q + cosize(SLayoutQ{});
  auto *shm_v = shm_k + cosize(SLayoutK{});
  int *shm_kvblk_ids = reinterpret_cast<int *>(shm_v + cosize(SLayoutV{}));
  auto *shm_y = reinterpret_cast<Tout *>(shm_data);  // Reuse All

  for (int i = idx; i < nblocks; i += blockDim.x) {
    shm_kvblk_ids[i] = block_ids[i];
  }

  // Tensor Q/K/V/Y
  auto gQ = tma_q.get_tma_tensor(make_shape(num_head_q, num_dim_qk, num_batch));
  auto gK =
      tma_k.get_tma_tensor(make_shape(kBlockSize, num_dim_qk, num_head_k, num_kvcache_blocks));
  auto gV = tma_v.get_tma_tensor(make_shape(num_dim_v, kBlockSize, num_head_v, num_kvcache_blocks));
  auto gY = tma_y.get_tma_tensor(make_shape(num_head_q, num_dim_v, num_batch));

  auto gAtt =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileN>{}), make_stride(Int<kTileN>{}, Int<1>{}));
  auto gYY =
      make_tensor(make_gmem_ptr(static_cast<Tout *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileV>{}), make_stride(Int<kTileV>{}, Int<1>{}));

  // Tensor sQ/sK/sV
  auto sQ = make_tensor(make_smem_ptr(shm_q), SLayoutQ{});
  auto sK = make_tensor(make_smem_ptr(shm_k), SLayoutK{});
  auto sV = make_tensor(make_smem_ptr(shm_v), SLayoutV{});
  auto sY = make_tensor(make_smem_ptr(shm_y), SLayoutY{});

  // Block Level tma
  auto btma_q = tma_q.get_slice(0);
  auto btma_k = tma_k.get_slice(0);
  auto btma_v = tma_v.get_slice(0);
  auto btma_y = tma_y.get_slice(0);

  // Thread Level Tensor
  auto tQg = btma_q.partition_S(gQ);  // (TMA, TMA_M, TMA_K, seqlenq, head_kv, batch)
  auto tKg = btma_k.partition_S(gK);  // (TMA, TMA_N, TMA_K, head_kv, batch)
  auto tVg = btma_v.partition_S(gV);  // (TMA, TMA_V, TMA_N, head_kv, batch)

  auto tQs = btma_q.partition_D(sQ);  // (TMA, _1, _1)
  auto tKs = btma_k.partition_D(sK);  // (TMA, _1, _1)
  auto tVs = btma_v.partition_D(sV);  // (TMA, _1, _1)

  TiledMmaQK tiled_mma_qk;
  TiledMmaPV tiled_mma_pv;

  auto thr_mma_qk = tiled_mma_qk.get_slice(idx);
  auto thr_mma_pv = tiled_mma_pv.get_slice(idx);

  auto tQs4r = thr_mma_qk.partition_A(sQ);
  auto tKs4r = thr_mma_qk.partition_B(sK);
  auto tVs4r = thr_mma_pv.partition_B(sV);

  auto tQr = thr_mma_qk.make_fragment_A(tQs4r);  // (MMA, MMA_M, MMA_K)
  auto tKr = thr_mma_qk.make_fragment_B(tKs4r);  // (MMA, MMA_N, MMA_K)
  auto tVr = thr_mma_pv.make_fragment_B(tVs4r);  // (MMA, MMA_V, MMA_N)

  auto tAttr = thr_mma_qk.partition_fragment_C(gAtt);
  auto tYr = thr_mma_pv.partition_fragment_C(gYY);

  auto gI = make_identity_tensor(gAtt.shape());
  auto tI = thr_mma_qk.partition_C(gI);

  auto tAttr_mn = retile_fragment(tAttr);
  constexpr int kM = size<0>(tAttr_mn);
  constexpr int kN = size<1>(tAttr_mn);
  Tensor gMax = make_tensor<float>(Int<kM>{});
  Tensor gSum = make_tensor<float>(Int<kM>{});

  clear(gSum);
  fill(gMax, -std::numeric_limits<float>::infinity());

  // Load Q
  if (is_leader_in_block) {
    initialize_barrier(bar_q, 1);
    cute::copy(tma_q.with(bar_q), tQg(_, ihead_kv, _, ibatch), tQs(_, 0, _));
    set_barrier_transaction_bytes(
        bar_q, sizeof(Tin) * max(heads_per_group, size<0, 0, 1>(tQg)) * num_dim_qk);
  }

  // init k/v barrier
  if (is_leader_in_block) {
    initialize_barrier(bar_k, 1);
    initialize_barrier(bar_v, 1);
  }

  // sync to avoid ahead thread use(wait) readable when it is not initizlized yet
  __syncthreads();
  wait_barrier(bar_q, 0);

  auto layout_asC = thr_mma_qk.partition_C(gAtt).layout();
  auto layout_asA = thr_mma_pv.partition_A(gAtt).layout();
  auto tAttA = make_tensor(tAttr.data(), left_inverse(layout_asC).compose(layout_asA));

  clear(tYr);

  tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;

  int num_tile_kv = (seqlenk + kTileN - 1) / kTileN;
  // int num_tile_causal = (seqlenq + kTileN - 1) / kTileN;
  int num_tile_full = (num_seq_kvcache + kTileN - 1) / kTileN;
  int num_tile_causal = num_tile_kv - num_tile_full + (num_seq_kvcache % kTileN != 0);
  num_tile_full = num_tile_kv - num_tile_causal;

  constexpr int kBlockPerTileN = kTileN / kBlockSize;
  constexpr int kGmmaK = kTileN / size<2>(tVr);

  int phase = 1;
#pragma unroll 1
  for (int itile_seq_kv = num_tile_full; itile_seq_kv < num_tile_kv; ++itile_seq_kv) {
    // load k/scale/v
    if (is_leader_in_block) {
      // k
      load_paged_kv<true, kBlockPerTileN, kBlockSize, kStage, Tin>(
          tma_k, tma_v, &bar_k, &bar_v, &bar_k, &bar_v, tKg, tKs, tVg, tVs, ihead_kv, num_dim_qk,
          num_dim_v, shm_kvblk_ids, nblocks, itile_seq_kv, 0, phase);
    }
    phase ^= 1;
    wait_barrier(bar_k, phase);

    // P = QK
    tiled_mma_qk.accumulate_ = GMMA::ScaleOut::Zero;
    warpgroup_fence_operand(tAttr);
    warpgroup_arrive();
#pragma unroll
    for (int ik = 0; ik < size<2>(tQr); ++ik) {
      cute::gemm(tiled_mma_qk, tQr(_, _, ik), tKr(_, _, ik, 0), tAttr(_, _, _));
      tiled_mma_qk.accumulate_ = GMMA::ScaleOut::One;
    }
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(tAttr);

    // do causal mask
    auto tAttr_mn = retile_fragment(tAttr);
    auto tI_mn = retile_fragment(tI);
#pragma unroll
    for (int im = 0; im < kM; ++im) {
#pragma unroll
      for (int in = 0; in < kN; ++in) {
        int iposq = num_seq_kvcache + get<0>(tI_mn(im, in)) / heads_per_group;
        int iposk = itile_seq_kv * kTileN + get<1>(tI_mn(im, in));

        if ((iposk > iposq) || (iposk >= seqlenk)) {
          tAttr_mn(im, in) = -std::numeric_limits<float>::infinity();
        }
      }
    }

    auto tYr_mn = retile_fragment(tYr);
    // online softmax
    online_softmax<true>(tAttr_mn, gMax, gSum, tYr_mn, kM, kN, one_over_dk_log2e);

    // Y = PV
    auto tAttAbf16 = make_tensor_like<cute::bfloat16_t>(tAttA);
#pragma unroll
    for (int i = 0; i < size(tAttA); ++i) {
      tAttAbf16(i) = (cute::bfloat16_t)(tAttA(i));
    }

    wait_barrier(bar_v, phase);

    warpgroup_fence_operand(tYr);
    warpgroup_arrive();
    cute::gemm(tiled_mma_pv, tAttAbf16, tVr(_, _, _, 0), tYr);
    warpgroup_commit_batch();
    warpgroup_wait<0>();

    warpgroup_fence_operand(tYr);
    __syncthreads();
  }

#pragma unroll 1
  for (int itile_seq_kv = 0; itile_seq_kv < num_tile_full; ++itile_seq_kv) {
    // load k/scale/v
    if (is_leader_in_block) {
      load_paged_kv<false, kBlockPerTileN, kBlockSize, kStage, Tin>(
          tma_k, tma_v, &bar_k, &bar_v, &bar_k, &bar_v, tKg, tKs, tVg, tVs, ihead_kv, num_dim_qk,
          num_dim_v, shm_kvblk_ids, nblocks, itile_seq_kv, 0, phase);
    }

    phase ^= 1;
    wait_barrier(bar_k, phase);

    // P = QK
    tiled_mma_qk.accumulate_ = GMMA::ScaleOut::Zero;
    warpgroup_fence_operand(tAttr);
    warpgroup_arrive();
#pragma unroll
    for (int ik = 0; ik < size<2>(tQr); ++ik) {
      cute::gemm(tiled_mma_qk, tQr(_, _, ik), tKr(_, _, ik, 0), tAttr(_, _, _));
      tiled_mma_qk.accumulate_ = GMMA::ScaleOut::One;
    }
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(tAttr);

    auto tYr_mn = retile_fragment(tYr);
    // online softmax
    online_softmax<false>(tAttr_mn, gMax, gSum, tYr_mn, kM, kN, one_over_dk_log2e);
    // Y = PV
    auto tAttAbf16 = make_tensor_like<cute::bfloat16_t>(tAttA);
#pragma unroll
    for (int i = 0; i < size(tAttA); ++i) {
      tAttAbf16(i) = (cute::bfloat16_t)(tAttA(i));
    }

    wait_barrier(bar_v, phase);

    warpgroup_fence_operand(tYr);
    warpgroup_arrive();
    cute::gemm(tiled_mma_pv, tAttAbf16, tVr(_, _, _, 0), tYr);
    warpgroup_commit_batch();
    warpgroup_wait<0>();

    warpgroup_fence_operand(tYr);
    __syncthreads();
  }

  auto tYr_mn = retile_fragment(tYr);
  // final online softmax
  final_online_softmax(tYr_mn, gSum, kM);

  // to bfloat16
  auto tYr_bf16 = make_tensor_like<Tout>(tYr);

#pragma unroll
  for (int i = 0; i < size(tYr); ++i) {
    Tout v{tYr(i)};
    tYr_bf16(i) = v;
  }

  // Epilogue: write register-C to global memory
  using R2SCopyAtomC = Copy_Atom<cute::SM90_U32x4_STSM_N, Tout>;
  auto r2s_tiled_copy = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma_pv);
  auto r2s_thr_copy = r2s_tiled_copy.get_slice(idx);

  auto tYr4s = r2s_thr_copy.retile_S(tYr_bf16);
  auto tYs4r = r2s_thr_copy.partition_D(sY);

  cute::copy(r2s_tiled_copy, tYr4s, tYs4r);
  __syncthreads();
  tma_store_fence();
  // using TMA to store
  if (is_leader_in_block) {
    auto tYss = btma_y.partition_S(sY);  // (TMA, TMA_M, TMA_N)
    auto tYgg = btma_y.partition_D(gY);  // (TMA, TMA_M, TMA_N, b)
    cute::copy(tma_y, tYss(_, 0, _), tYgg(_, ihead_kv, _, ibatch));
  }
}

}  // namespace kernels
}  // namespace decode
}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_DECODE_M64_KERNELS_CUH_
