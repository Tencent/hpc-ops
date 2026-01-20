// Copyright (C) 2026 Tencent.

#ifndef SRC_ATTENTION_DECODE_SMALLM_SPLITK_FP8_KERNELS_CUH_
#define SRC_ATTENTION_DECODE_SMALLM_SPLITK_FP8_KERNELS_CUH_

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

template <int kTileN, int kN, typename TensorY, typename TensorS>
__device__ __forceinline__ void final_online_softmax(TensorY &tYr_mn, TensorS &gSum,
                                                     float *smem_sum, int iwarpgroup, int iwarp,
                                                     int ilane) {
  vec_t<float, kN> warp_sum;

#pragma unroll
  for (int in = 0; in < kN; ++in) {
    warp_sum[in] = warp_8lane_stride4_reduce_sum_xor(gSum(in));
  }

  if (ilane < 4) {
    store(smem_sum + iwarp * kTileN + ilane * kN, warp_sum);
  }

  syncwarpgroup(iwarpgroup);

  if (ilane < 4) {
#pragma unroll
    for (int in = 0; in < kN; ++in) {
      warp_sum[in] = 0.f;
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
      auto reduce_sum = load<float, kN>(smem_sum + i * kTileN + ilane * kN);
#pragma unroll
      for (int in = 0; in < kN; ++in) {
        warp_sum[in] += reduce_sum[in];
      }
    }
  }

#pragma unroll
  for (int in = 0; in < kN; ++in) {
    warp_sum[in] = __shfl_sync(0xFFFFFFFF, warp_sum[in], ilane % 4);
  }

#pragma unroll
  for (int in = 0; in < kN; ++in) {
    gSum(in) = warp_sum[in];
    float one_over_gsum = rcpf_ftz(warp_sum[in]);
#pragma unroll
    for (int im = 0; im < cute::size<0>(tYr_mn); ++im) {
      tYr_mn(im, in) = tYr_mn(im, in) * one_over_gsum;
    }
  }
}

template <bool kCheckInf, int kTileN, int kM, int kN, typename TensorA, typename TensorM,
          typename TensorS, typename TensorY, typename TensorScale>
__device__ __forceinline__ void online_softmax(TensorA &tAttr_mn, TensorM &gMax, TensorS &gSum,
                                               TensorY &tYr_mn, TensorScale &one_over_dk_log2e,
                                               float *smem_max, int iwarpgroup, int iwarp,
                                               int ilane) {
  vec_t<float, kN> warp_max;
#pragma unroll
  for (int in = 0; in < kN; ++in) {
    float row_max = tAttr_mn(0, in);

#pragma unroll
    for (int im = 1; im < kM; ++im) {
      row_max = fmaxf(row_max, tAttr_mn(im, in));
    }

    warp_max[in] = warp_8lane_stride4_reduce_max_xor(row_max) * one_over_dk_log2e(in);
  }

  if (ilane < 4) {
    store(smem_max + iwarp * kTileN + ilane * kN, warp_max);
  }

  syncwarpgroup(iwarpgroup);

  if (ilane < 4) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
      auto reduce_max = load<float, kN>(smem_max + i * kTileN + ilane * kN);
#pragma unroll
      for (int in = 0; in < kN; ++in) {
        warp_max[in] = fmax(reduce_max[in], warp_max[in]);
      }
    }
  }

#pragma unroll
  for (int in = 0; in < kN; ++in) {
    warp_max[in] = __shfl_sync(0xFFFFFFFF, warp_max[in], ilane % 4);
  }

#pragma unroll
  for (int in = 0; in < kN; ++in) {
    float last_max = gMax(in);
    float row_max = fmaxf(last_max, warp_max[in]);
    float row_sum = 0.f;

    gMax(in) = row_max;

    if constexpr (kCheckInf) {
      if (gMax(in) == -std::numeric_limits<float>::infinity()) {
#pragma unroll
        for (int im = 0; im < kM; ++im) {
          tAttr_mn(im, in) = 0.f;
        }
        continue;
      }
    }

#pragma unroll
    for (int im = 0; im < kM; ++im) {
      tAttr_mn(im, in) = exp2f_ftz(tAttr_mn(im, in) * one_over_dk_log2e(in) - gMax(in));
      row_sum += tAttr_mn(im, in);
    }

    float scale = exp2f_ftz(last_max - gMax(in));
    gSum(in) = gSum(in) * scale + row_sum;

#pragma unroll
    for (int im = 0; im < cute::size<0>(tYr_mn); ++im) {
      tYr_mn(im, in) = tYr_mn(im, in) * scale;
    }
  }
}

template <bool kCheckBound, int kBlockPerTileM, int kBlockSize, int kStage, typename Tin,
          typename TmaK, typename TmaV, typename TensorGK, typename TensorSK, typename TensorGV,
          typename TensorSV>
__device__ __forceinline__ void load_paged_kv(TmaK &tma_k, TmaV &tma_v, uint64_t *k_writable,
                                              uint64_t *v_writable, uint64_t *k_readable,
                                              uint64_t *v_readable, TensorGK &tKg, TensorSK &tKs,
                                              TensorGV &tVg, TensorSV &tVs, int ihead_kv,
                                              int num_dim_qk, int num_dim_v, int *block_ids,
                                              int num_blocks, int itile, int istage_write,
                                              int phase) {
  using namespace cute;  // NOLINT

  int load_blocks = kBlockPerTileM;
  int istage = istage_write;

  vec_t<int, kBlockPerTileM> blk_ids;
#pragma unroll
  for (int ikvblock = 0; ikvblock < kBlockPerTileM; ikvblock++) {
    int kvblk_id = itile * kBlockPerTileM + ikvblock;
    int blk_id = -1;
    if constexpr (kCheckBound) {
      if (kvblk_id < num_blocks) {
        blk_id = block_ids[kvblk_id];
      }
    } else {
      blk_id = block_ids[kvblk_id];
    }
    blk_ids[ikvblock] = blk_id;
  }

  wait_barrier(k_writable[istage], phase);
#pragma unroll
  for (int ikvblock = 0; ikvblock < kBlockPerTileM; ikvblock++) {
    int blk_id = blk_ids[ikvblock];
    cute::copy(tma_k.with(k_readable[istage]), tKg(_, 0, _, ihead_kv, blk_id),
               tKs(_, ikvblock, _, istage));
  }
  set_barrier_transaction_bytes(k_readable[istage],
                                sizeof(Tin) * load_blocks * kBlockSize * num_dim_qk);

  wait_barrier(v_writable[istage], phase);
  // v
#pragma unroll
  for (int ikvblock = 0; ikvblock < kBlockPerTileM; ikvblock++) {
    int blk_id = blk_ids[ikvblock];
    cute::copy(tma_v.with(v_readable[istage]), tVg(_, _, 0, ihead_kv, blk_id),
               tVs(_, _, ikvblock, istage));
  }
  set_barrier_transaction_bytes(v_readable[istage],
                                sizeof(Tin) * load_blocks * kBlockSize * num_dim_v);
}

template <typename Tout, typename Tin, int kTileM, int kTileN, int kTileK, int kTileV,
          typename TiledMmaQK, typename TiledMmaSV, typename TmaQ, typename TmaK, typename TmaV,
          typename TmaY, typename TmaSplitY, typename S2RCopyP, typename R2SCopySplitY,
          typename SLayoutQ, typename SLayoutK, typename SLayoutP, typename SLayoutS,
          typename SLayoutVTma, typename SLayoutY, typename SLayoutSplitY, int kHeadsPerGroup,
          int kBlockSize, int kStage, int kConsumers, int kSplitK, int kSplitMinLen>
__global__ void attention_decode_fp8_multistage_ws_smallm_splitk_kernel(
    const __grid_constant__ TmaQ tma_q, const __grid_constant__ TmaK tma_k,
    const __grid_constant__ TmaV tma_v, const __grid_constant__ TmaY tma_y,
    const __grid_constant__ TmaSplitY tma_splity, Tout *y_ptr, float *split_y_ptr, float *lse_ptr,
    const int *block_ids_ptr, const int *num_seq_kvcache_ptr, const float *qscale_ptr,
    const float *kscale_ptr, const float *vscale_ptr, int *split_flag_ptr, bool new_kv_included,
    int num_batch, int num_dim_qk, int num_dim_v, int num_head_q, int num_head_k, int num_head_v,
    int heads_per_group, int num_kvcache_blocks, int num_seq_max_blocks, int qscale_pad_stride,
    float one_over_dk_log2e) {
  using namespace cute;  // NOLINT

  int idx = threadIdx.x;
  int ihead_kv = blockIdx.x;
  int ibatch = blockIdx.y;
  int ichunk = blockIdx.z;

  constexpr int kMathThreads = size(TiledMmaQK{}) * kConsumers;
  constexpr int kWarpsPerWrapGroup = 4;

  int elected = cute::elect_one_sync();
  int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
  bool is_leader_in_block = (iwarp == 0) && elected;

  constexpr int kSeqlenQ = 1;
  int num_seq_kvcache = num_seq_kvcache_ptr[ibatch];
  if (new_kv_included) {
    num_seq_kvcache -= kSeqlenQ;
  }
  int num_seq_kv = kSeqlenQ + num_seq_kvcache;

  if (num_seq_kv <= 0) {
    return;
  }

  int num_seq_per_chunk = (num_seq_kv + kSplitK - 1) / kSplitK;
  num_seq_per_chunk = (num_seq_per_chunk + kTileM - 1) / kTileM * kTileM;
  num_seq_per_chunk = max(num_seq_per_chunk, kSplitMinLen);

  int iseq_start = ichunk * num_seq_per_chunk;
  if (iseq_start >= num_seq_kv) {
    return;
  }

  bool is_last_chunk = false;
  if (iseq_start + num_seq_per_chunk >= num_seq_kv) {
    is_last_chunk = true;
  }

  bool is_split = false;
  if (num_seq_per_chunk < num_seq_kv) {
    is_split = true;
  }

  int num_chunks = kConsumers * ((num_seq_kv + num_seq_per_chunk - 1) / num_seq_per_chunk);

  num_seq_kv = min(num_seq_kv - iseq_start, num_seq_per_chunk);
  num_seq_kvcache = num_seq_kv - kSeqlenQ;

  int num_blocks = (num_seq_kv + kBlockSize - 1) / kBlockSize;
  int num_blocks_per_chunk = (num_seq_per_chunk + kBlockSize - 1) / kBlockSize;

  float *lse_batch = lse_ptr + ibatch * kSplitK * kConsumers * num_head_q +
                     ichunk * num_head_q * kConsumers + ihead_kv * heads_per_group;

  const int *block_ids =
      block_ids_ptr + ibatch * num_seq_max_blocks + ichunk * num_blocks_per_chunk;

  float kscale = kscale_ptr[0];
  float vscale = vscale_ptr[0];
  auto *qscales = qscale_ptr + ibatch * num_head_q + ihead_kv * heads_per_group;

  __shared__ uint64_t q_readable;
  __shared__ uint64_t k_writable[kStage];
  __shared__ uint64_t v_writable[kStage];
  __shared__ uint64_t k_readable[kStage];
  __shared__ uint64_t v_readable[kStage];
  extern __shared__ uint8_t shm_data[] alignas(128);

  auto *shm_q = reinterpret_cast<Tin *>(shm_data);
  auto *shm_k = shm_q + cosize(SLayoutQ{});
  auto *shm_vtma = shm_k + cosize(SLayoutK{});
  auto *shm_p = shm_vtma + cosize(SLayoutVTma{});
  auto *shm_max = reinterpret_cast<float *>(shm_p + cosize(SLayoutP{}));
  auto *shm_kvblk_ids = reinterpret_cast<int *>(shm_max + kTileN * kWarpsPerWrapGroup * kConsumers);
  auto *shm_y = reinterpret_cast<Tout *>(shm_data);        // Reuse All
  auto *shm_splity = reinterpret_cast<float *>(shm_data);  // Reuse All

  // Tensor Q/K/V/Y
  auto gQ = tma_q.get_tma_tensor(make_shape(num_head_q, num_dim_qk, num_batch));
  auto gK =
      tma_k.get_tma_tensor(make_shape(kBlockSize, num_dim_qk, num_head_k, num_kvcache_blocks));
  auto gV = tma_v.get_tma_tensor(make_shape(num_dim_v, kBlockSize, num_head_v, num_kvcache_blocks));
  auto gY = tma_y.get_tma_tensor(make_shape(num_dim_v, num_head_q, num_batch));
  auto gSplitY =
      tma_splity.get_tma_tensor(make_shape(num_dim_v, num_head_q, kSplitK * kConsumers, num_batch));

  auto gAtt =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileN>{}), make_stride(Int<kTileN>{}, Int<1>{}));
  auto gYY =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileV>{}, Int<kTileN>{}), make_stride(Int<1>{}, Int<kTileV>{}));

  // Tensor sQ/sK/sV
  auto sQ = make_tensor(make_smem_ptr(shm_q), SLayoutQ{});
  auto sK = make_tensor(make_smem_ptr(shm_k), SLayoutK{});
  auto sP = make_tensor(make_smem_ptr(shm_p), SLayoutP{});
  auto sS = make_tensor(make_smem_ptr(shm_p), SLayoutS{});
  auto sVTma = make_tensor(make_smem_ptr(shm_vtma), SLayoutVTma{});
  auto sY = make_tensor(make_smem_ptr(shm_y), SLayoutY{});
  auto sSplitY = make_tensor(make_smem_ptr(shm_splity), SLayoutSplitY{});

  // Block Level tma
  auto btma_q = tma_q.get_slice(0);
  auto btma_k = tma_k.get_slice(0);
  auto btma_v = tma_v.get_slice(0);
  auto btma_y = tma_y.get_slice(0);
  auto btma_splity = tma_splity.get_slice(0);

  // Thread Level Tensor
  auto tQg = btma_q.partition_S(gQ);  // (TMA, TMA_M, TMA_K, seqlenq, head_kv, batch)
  auto tKg = btma_k.partition_S(gK);  // (TMA, TMA_N, TMA_K, head_kv, batch)
  auto tVg = btma_v.partition_S(gV);  // (TMA, TMA_V, TMA_N, head_kv, batch)

  auto tQs = btma_q.partition_D(sQ);     // (TMA, _1, _1)
  auto tKs = btma_k.partition_D(sK);     // (TMA, _1, _1)
  auto tVs = btma_v.partition_D(sVTma);  // (TMA, _1, _1)

  int num_tile_kv = (num_seq_kv + kTileM - 1) / kTileM;
  int num_tile_full = (num_seq_kvcache + kTileM - 1) / kTileM;
  int num_tile_causal = num_tile_kv - num_tile_full + (num_seq_kvcache % kTileM != 0);
  num_tile_full = num_tile_kv - num_tile_causal;

  constexpr int kBlockPerTileM = kTileM / kBlockSize;

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
      cute::copy(tma_q.with(q_readable), tQg(_, ihead_kv, _, ibatch), tQs(_, 0, _));
      set_barrier_transaction_bytes(
          q_readable, sizeof(Tin) * max(heads_per_group, size<0, 0, 1>(tQg)) * num_dim_qk);
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
    int iload_tile = 0;

    if (is_leader_in_load) {
      int istage_write = 0;
      // Load Causal KV
#pragma unroll 1
      for (int itile_seq_kv = num_tile_full; itile_seq_kv < num_tile_kv; ++itile_seq_kv) {
        // load k/scale/v
        load_paged_kv<true, kBlockPerTileM, kBlockSize, kStage, Tin>(
            tma_k, tma_v, k_writable, v_writable, k_readable, v_readable, tKg, tKs, tVg, tVs,
            ihead_kv, num_dim_qk, num_dim_v, shm_kvblk_ids, num_blocks, itile_seq_kv,
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
          load_paged_kv<false, kBlockPerTileM, kBlockSize, kStage, Tin>(
              tma_k, tma_v, k_writable, v_writable, k_readable, v_readable, tKg, tKs, tVg, tVs,
              ihead_kv, num_dim_qk, num_dim_v, shm_kvblk_ids, num_blocks, iload_tile++,
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
    int ilane_in_warpgroup = idx_in_warpgroup % 32;
    int elected_idx_in_warpgroup = ((iwarp_in_warpgroup == 0) && elected);
    bool is_leader_in_warpgroup = ((iwarp % 4) == 0) && elected;

    TiledMmaQK tiled_mma_qk;
    TiledMmaSV tiled_mma_sv;

    auto thr_mma_qk = tiled_mma_qk.get_slice(idx_in_warpgroup);
    auto thr_mma_sv = tiled_mma_sv.get_slice(idx_in_warpgroup);

    auto tKs4r = thr_mma_qk.partition_A(sK);
    auto tQs4r = thr_mma_qk.partition_B(sQ);
    auto tVs4r = thr_mma_sv.partition_A(sVTma(_, _, 0));
    auto tSs4r = thr_mma_sv.partition_B(sS);

    auto tKr = thr_mma_qk.make_fragment_A(tKs4r);  // (MMA, MMA_N, MMA_K)
    auto tQr = thr_mma_qk.make_fragment_B(tQs4r);  // (MMA, MMA_M, MMA_K)
    auto tVr = thr_mma_sv.make_fragment_A(tVs4r);  // (MMA, MMA_V, MMA_N)
    auto tSr = thr_mma_sv.make_fragment_B(tSs4r);  // (MMA, MMA_V, MMA_N)

    auto tAttr = thr_mma_qk.partition_fragment_C(gAtt);
    auto tYr = thr_mma_sv.partition_fragment_C(gYY);

    auto gI = make_identity_tensor(gAtt.shape());
    auto tI = thr_mma_qk.partition_C(gI);
    auto tI_mn = retile_fragment(tI);

    auto tAttr_mn = retile_fragment(tAttr);
    constexpr int kM = size<0>(tAttr_mn);
    constexpr int kN = size<1>(tAttr_mn);
    Tensor gMax = make_tensor<float>(Int<kN>{});
    Tensor gSum = make_tensor<float>(Int<kN>{});
    Tensor qkscales = make_tensor<float>(Int<kN>{});
    Tensor gSoftmaxScale = make_tensor<float>(Int<kN>{});

#pragma unroll
    for (int i = 0; i < kN; i++) {
      int icol = get<1>(tI_mn(0, i));
      qkscales[i] = qscales[icol % kHeadsPerGroup];
    }

    clear(gSum);
    fill(gMax, -std::numeric_limits<float>::infinity());
    fill(gSoftmaxScale, one_over_dk_log2e);

    S2RCopyP tiled_copy_P_r2s;
    auto thr_copy_P_r2s = tiled_copy_P_r2s.get_slice(idx_in_warpgroup);
    auto tPs4r = thr_copy_P_r2s.partition_D(sP);

    auto warp_sVTma =
        local_tile(local_tile(sVTma, make_shape(Int<64>{}, Int<32>{}), make_coord(_, _)),
                   make_shape(Int<16>{}, Int<32>{}), make_coord(iwarp_in_warpgroup, 0));

    auto sVTmaTrans = make_tensor(warp_sVTma.data(), select<1, 0, 3, 2, 4>(warp_sVTma.layout()));

    auto tile_for_trans = make_tile(Int<8>{}, Int<16>{});
    auto sV_tile_for_trans = flat_divide(sVTmaTrans, tile_for_trans);

    auto m_for_trans = size<2>(sV_tile_for_trans);
    auto n_for_trans = size<3>(sV_tile_for_trans);

    auto tiled_copy_VTrans_s2r = make_tiled_copy(
        Copy_Atom<SM75_U16x8_LDSM_T, Tin>{},
        make_layout(make_shape(Int<4>{}, Int<8>{}, Int<1>{}, Int<1>{}),
                    make_stride(Int<1>{}, Int<4>{}, Int<0>{}, Int<0>{})),
        make_layout(make_shape(Int<2>{}, Int<2>{}, m_for_trans, n_for_trans),
                    make_stride(Int<2>{}, Int<1>{}, Int<4>{}, Int<4>{} * m_for_trans)));

    auto thr_copy_VTrans_s2r = tiled_copy_VTrans_s2r.get_slice(ilane_in_warpgroup);
    auto tVTranss4r = thr_copy_VTrans_s2r.partition_S(sV_tile_for_trans);
    auto tVTransr4s =
        make_fragment_like(thr_copy_VTrans_s2r.partition_D(sV_tile_for_trans(_, _, _, _, _, _, 0)));
    auto v_for_trans = coalesce(recast<uint32_t>(tVTransr4s));
    auto layout_asA = tVr.layout();
    auto vt_for_trans = coalesce(recast<uint32_t>(
        make_tensor(tVr.data(), left_inverse(layout_asA).compose(tVTransr4s.layout()))));

    clear(tYr);

    tiled_mma_sv.accumulate_ = GMMA::ScaleOut::One;

    wait_barrier(q_readable, 0);

    int phase = 0;
    int istage_read = iwarpgroup;
    shm_max += iwarpgroup * kTileN * kWarpsPerWrapGroup;
    lse_batch += iwarpgroup * num_head_q;
    bool warpgroup_computed = false;

#pragma unroll
    for (int i = 0; i < kN; i++) {
      qkscales[i] *= kscale;
    }
    // compute casual
#pragma unroll 1
    for (int itile_seq_kv = num_tile_full + iwarpgroup; itile_seq_kv < num_tile_kv;
         itile_seq_kv += kConsumers) {
      warpgroup_computed = true;
      wait_barrier(k_readable[istage_read], phase);

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

      if (elected_idx_in_warpgroup) {
        arrive_barrier(k_writable[istage_read]);
      }
      // do causal mask
      auto tAttr_mn = retile_fragment(tAttr);
#pragma unroll
      for (int im = 0; im < kM; ++im) {
#pragma unroll
        for (int in = 0; in < kN; ++in) {
          int icol = get<1>(tI_mn(im, in));
          int iposq = num_seq_kvcache + icol / kHeadsPerGroup;
          int iposk = itile_seq_kv * kTileM + get<0>(tI_mn(im, in));

          if ((iposk > iposq) || (iposk >= num_seq_kv)) {
            tAttr_mn(im, in) = -std::numeric_limits<float>::infinity();
          } else {
            tAttr_mn(im, in) *= qkscales[in];
          }
        }
      }
      auto tYr_mn = retile_fragment(tYr);

      // online softmax
      online_softmax<true, kTileN, kM, kN>(tAttr_mn, gMax, gSum, tYr_mn, gSoftmaxScale, shm_max,
                                           iwarpgroup, iwarp_in_warpgroup, ilane_in_warpgroup);

      // Y = PV
      auto tAttAfp8 = make_tensor_like<Tin>(tAttr);
      auto *raw_float4_pointer = reinterpret_cast<float4 *>(tAttr.data());
      auto *raw_fp8x4_pointer = reinterpret_cast<__nv_fp8x4_e4m3 *>(tAttAfp8.data());
#pragma unroll
      for (int i = 0; i < size(tAttr) / 4; ++i) {
        raw_fp8x4_pointer[i] = __nv_fp8x4_e4m3(raw_float4_pointer[i]);
      }

      auto tAttAfp8_32 = recast<uint32_t>(tAttAfp8);
#pragma unroll
      for (int i = 0; i < size(tAttAfp8_32); ++i) {
        uint32_t val = tAttAfp8_32(i);
        tAttAfp8_32(i) = __byte_perm(val, val, 0x7520);
      }

      auto tPr4s = thr_copy_P_r2s.retile_S(tAttAfp8);
      cute::copy(tiled_copy_P_r2s, tPr4s, tPs4r(_, _, _, iwarpgroup));

      wait_barrier(v_readable[istage_read], phase);
      cutlass::arch::fence_view_async_shared();
      syncwarpgroup(iwarpgroup);

#pragma unroll
      for (int im = 0; im < size<3>(sVTmaTrans); im++) {
#pragma unroll
        for (int in = 0; in < size<2>(sVTmaTrans); in++) {
          cute::copy(tiled_copy_VTrans_s2r, tVTranss4r(_, _, _, _, _, in, im, istage_read),
                     tVTransr4s(_, _, _, _, _, in, im));
        }
      }

      warpgroup_fence_operand(tYr);
#pragma unroll
      for (int im = 0; im < size<3>(sVTmaTrans); im++) {
#pragma unroll
        for (int in = 0; in < size<2>(sVTmaTrans); in++) {
          vt_for_trans(0, in, im) =
              __byte_perm(v_for_trans(0, in, im), v_for_trans(1, in, im), 0x6240);
          vt_for_trans(1, in, im) =
              __byte_perm(v_for_trans(0, in, im), v_for_trans(1, in, im), 0x7351);

          vt_for_trans(2, in, im) =
              __byte_perm(v_for_trans(2, in, im), v_for_trans(3, in, im), 0x6240);
          vt_for_trans(3, in, im) =
              __byte_perm(v_for_trans(2, in, im), v_for_trans(3, in, im), 0x7351);

          warpgroup_arrive();
          cute::gemm(tiled_mma_sv, tVr(_, im, in), tSr(_, 0, in, iwarpgroup), tYr(_, im, 0));
          warpgroup_commit_batch();
        }
      }

      if (elected_idx_in_warpgroup) {
        arrive_barrier(v_writable[istage_read]);
      }
      istage_read += kConsumers;
      if (istage_read >= kStage) {
        istage_read = istage_read % kStage;
        phase ^= 1;
      }

      warpgroup_wait<0>();
      warpgroup_fence_operand(tYr);

      syncwarpgroup(iwarpgroup);
    }

#pragma unroll
    for (int i = 0; i < kN; i++) {
      gSoftmaxScale[i] *= qkscales[i];
    }
    // compute full
#pragma unroll 1
    for (int itile_seq_kv = (kConsumers - num_tile_causal + iwarpgroup) % kConsumers;
         itile_seq_kv < num_tile_full; itile_seq_kv += kConsumers) {
      warpgroup_computed = true;
      wait_barrier(k_readable[istage_read], phase);

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

      if (elected_idx_in_warpgroup) {
        arrive_barrier(k_writable[istage_read]);
      }

      auto tYr_mn = retile_fragment(tYr);
      // online softmax
      online_softmax<false, kTileN, kM, kN>(tAttr_mn, gMax, gSum, tYr_mn, gSoftmaxScale, shm_max,
                                            iwarpgroup, iwarp_in_warpgroup, ilane_in_warpgroup);

      // Y = PV
      auto tAttAfp8 = make_tensor_like<Tin>(tAttr);
      auto *raw_float4_pointer = reinterpret_cast<float4 *>(tAttr.data());
      auto *raw_fp8x4_pointer = reinterpret_cast<__nv_fp8x4_e4m3 *>(tAttAfp8.data());
#pragma unroll
      for (int i = 0; i < size(tAttr) / 4; ++i) {
        raw_fp8x4_pointer[i] = __nv_fp8x4_e4m3(raw_float4_pointer[i]);
      }

      auto tAttAfp8_32 = recast<uint32_t>(tAttAfp8);
#pragma unroll
      for (int i = 0; i < size(tAttAfp8_32); ++i) {
        uint32_t val = tAttAfp8_32(i);
        tAttAfp8_32(i) = __byte_perm(val, val, 0x7520);
      }

      auto tPr4s = thr_copy_P_r2s.retile_S(tAttAfp8);
      cute::copy(tiled_copy_P_r2s, tPr4s, tPs4r(_, _, _, iwarpgroup));

      wait_barrier(v_readable[istage_read], phase);
      cutlass::arch::fence_view_async_shared();
      syncwarpgroup(iwarpgroup);

#pragma unroll
      for (int im = 0; im < size<3>(sVTmaTrans); im++) {
#pragma unroll
        for (int in = 0; in < size<2>(sVTmaTrans); in++) {
          cute::copy(tiled_copy_VTrans_s2r, tVTranss4r(_, _, _, _, _, in, im, istage_read),
                     tVTransr4s(_, _, _, _, _, in, im));
        }
      }

      warpgroup_fence_operand(tYr);
#pragma unroll
      for (int im = 0; im < size<3>(sVTmaTrans); im++) {
#pragma unroll
        for (int in = 0; in < size<2>(sVTmaTrans); in++) {
          vt_for_trans(0, in, im) =
              __byte_perm(v_for_trans(0, in, im), v_for_trans(1, in, im), 0x6240);
          vt_for_trans(1, in, im) =
              __byte_perm(v_for_trans(0, in, im), v_for_trans(1, in, im), 0x7351);

          vt_for_trans(2, in, im) =
              __byte_perm(v_for_trans(2, in, im), v_for_trans(3, in, im), 0x6240);
          vt_for_trans(3, in, im) =
              __byte_perm(v_for_trans(2, in, im), v_for_trans(3, in, im), 0x7351);

          warpgroup_arrive();
          cute::gemm(tiled_mma_sv, tVr(_, im, in), tSr(_, 0, in, iwarpgroup), tYr(_, im, 0));
          warpgroup_commit_batch();
        }
      }

      if (elected_idx_in_warpgroup) {
        arrive_barrier(v_writable[istage_read]);
      }
      istage_read += kConsumers;
      if (istage_read >= kStage) {
        istage_read = istage_read % kStage;
        phase ^= 1;
      }

      warpgroup_wait<0>();
      warpgroup_fence_operand(tYr);

      syncwarpgroup(iwarpgroup);
    }

    auto tYr_mn = retile_fragment(tYr);
    if (warpgroup_computed) {
      // final online softmax
      final_online_softmax<kTileN, kN>(tYr_mn, gSum, shm_max, iwarpgroup, iwarp_in_warpgroup,
                                       ilane_in_warpgroup);
    }

#pragma unroll
    for (int i = 0; i < size(tYr); ++i) {
      tYr(i) *= vscale;
    }

    // Epilogue: write register-C to global memory
    R2SCopySplitY r2s_tiled_copy;
    auto r2s_thr_copy = r2s_tiled_copy.get_slice(idx_in_warpgroup);

    auto tYr4s = r2s_thr_copy.retile_S(tYr);
    auto tYs4r = r2s_thr_copy.partition_D(sSplitY);

    bar_sync<kConsumers * 128>(kConsumers);
    cute::copy(r2s_tiled_copy, tYr4s, tYs4r(_, _, _, iwarpgroup));
    syncwarpgroup(iwarpgroup);
    tma_store_fence();

    // using TMA to store
    if (is_leader_in_warpgroup) {
      auto tYss = btma_splity.partition_S(sSplitY);  // (TMA, TMA_M, TMA_N)
      auto tYgg = btma_splity.partition_D(gSplitY);  // (TMA, TMA_M, TMA_N, b)

      cute::copy(tma_splity, tYss(_, _, 0, iwarpgroup),
                 tYgg(_, _, ihead_kv, ichunk * kConsumers + iwarpgroup, ibatch));

      tma_store_arrive();
    }

    // write lse
    if (iwarp_in_warpgroup == 0 && ilane_in_warpgroup < heads_per_group / kN) {
      vec_t<float, kN> lse;
#pragma unroll
      for (int in = 0; in < kN; ++in) {
        if (gMax(in) == -std::numeric_limits<float>::infinity()) {
          lse[in] = -std::numeric_limits<float>::infinity();
        } else {
          lse[in] = gMax(in) + log2f_ftz(gSum(in));
        }
      }
      store(lse_batch + ilane_in_warpgroup * kN, lse);
    }

    // do reduce for split k
    auto *split_flag = split_flag_ptr + ibatch * num_head_k + ihead_kv;
    tma_store_wait<0>();

    __threadfence();
    bar_sync<kConsumers * 128>(kConsumers);

    if (idx_in_warpgroup == 0) {
      atomicAdd(split_flag, 1);
    }

    if (is_last_chunk) {
      while (load_global_volatile(split_flag) != (ichunk + 1) * kConsumers) {
      }

      constexpr int kMathWarps = kMathThreads / 32;
      constexpr int kSplitKConsumsers = kSplitK * kConsumers;
      constexpr int kItemsPerThread = 4;
      int icol = ilane_in_warpgroup * kItemsPerThread;

      vec_t<float, kSplitKConsumsers> lse;
      vec_t<float, kItemsPerThread> output;

      lse_batch = lse_ptr + ibatch * kSplitKConsumsers * num_head_q + ihead_kv * heads_per_group;
      auto *split_input = split_y_ptr + ibatch * kSplitKConsumsers * num_head_q * kTileV +
                          ihead_kv * heads_per_group * kTileV + icol;
      auto *out_row = reinterpret_cast<__nv_bfloat16 *>(y_ptr) + ibatch * num_head_q * kTileV +
                      ihead_kv * heads_per_group * kTileV + icol;

      for (int iqhead = iwarp; iqhead < heads_per_group; iqhead += kMathWarps) {
        auto *lse_head = lse_batch + iqhead;
        auto *split_input_head = split_input + iqhead * kTileV;
        auto *out_row_head = out_row + iqhead * kTileV;
#pragma unroll
        for (int i = 0; i < kItemsPerThread; i++) {
          output[i] = 0.f;
        }

        float max_lse = 0.f;
        float sum_lse = 0.f;

#pragma unroll
        for (int i = 0; i < kSplitKConsumsers; i++) {
          if (i < num_chunks) {
            lse[i] = lse_head[i * num_head_q];
            max_lse = max(max_lse, lse[i]);
          }
        }

#pragma unroll
        for (int i = 0; i < kSplitKConsumsers; i++) {
          if (i < num_chunks) {
            sum_lse += exp2f_ftz(lse[i] - max_lse);
          }
        }

        sum_lse = log2f_ftz(sum_lse) + max_lse;

#pragma unroll
        for (int i = 0; i < kSplitKConsumsers; i++) {
          if (i < num_chunks) {
            auto y = load<float, kItemsPerThread>(split_input_head + i * num_head_q * kTileV);
            float scale = exp2f_ftz(lse[i] - sum_lse);
#pragma unroll
            for (int j = 0; j < kItemsPerThread; j++) {
              output[j] += scale * y[j];
            }
          }
        }
        store(out_row_head, to<__nv_bfloat16>(output));
      }
    }
  }
}

}  // namespace kernels
}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_DECODE_SMALLM_SPLITK_FP8_KERNELS_CUH_
