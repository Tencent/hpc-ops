// Copyright (C) 2026 Tencent.

#ifndef SRC_ATTENTION_DECODE_UTIL_KERNELS_CUH_
#define SRC_ATTENTION_DECODE_UTIL_KERNELS_CUH_

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

template <int kTileN, int kBlockSize, int kSplitK, int kSplitMinLen>
__device__ __forceinline__ bool get_task(const int* num_seq_kvcache_ptr, bool new_kv_included,
                                         const int& num_seq_q, const int& ibatch, const int& ichunk,
                                         int& num_seq_kvcache, int& num_seq_kv, int& num_chunks,
                                         bool& is_split, bool& is_last_chunk, int& num_blocks,
                                         int& num_blocks_per_chunk, int& num_tile_kv,
                                         int& num_tile_full, int& num_tile_causal) {
  num_seq_kvcache = num_seq_kvcache_ptr[ibatch];
  if (new_kv_included) {
    num_seq_kvcache -= num_seq_q;
  }
  num_seq_kv = num_seq_q + num_seq_kvcache;

  if (num_seq_kv <= 0) {
    return false;
  }

  int num_seq_per_chunk = (num_seq_kv + kSplitK - 1) / kSplitK;
  num_seq_per_chunk = (num_seq_per_chunk + kTileN - 1) / kTileN * kTileN;
  num_seq_per_chunk = max(num_seq_per_chunk, kSplitMinLen);

  int iseq_start = ichunk * num_seq_per_chunk;
  if (iseq_start >= num_seq_kv) {
    return false;
  }

  is_last_chunk = false;
  if (iseq_start + num_seq_per_chunk >= num_seq_kv) {
    is_last_chunk = true;
  }

  is_split = false;
  if (num_seq_per_chunk < num_seq_kv) {
    is_split = true;
  }

  num_chunks = (num_seq_kv + num_seq_per_chunk - 1) / num_seq_per_chunk;

  num_seq_kv = min(num_seq_kv - iseq_start, num_seq_per_chunk);

  if (is_last_chunk) {
    num_seq_kvcache = num_seq_kv - num_seq_q;
  } else {
    num_seq_kvcache = num_seq_kv;
  }

  num_blocks = (num_seq_kv + kBlockSize - 1) / kBlockSize;
  num_blocks_per_chunk = (num_seq_per_chunk + kBlockSize - 1) / kBlockSize;

  num_tile_kv = (num_seq_kv + kTileN - 1) / kTileN;
  num_tile_full = num_seq_kvcache / kTileN;

  if (is_last_chunk) {
    num_tile_causal = num_tile_kv - num_tile_full;
  } else {
    num_tile_causal = 0;
  }
  num_tile_full = num_tile_kv - num_tile_causal;

  return true;
}

template <bool kCheckBound, int kBlockPerTileN, int kBlockSize, int kStage, typename Tin,
          typename TmaK, typename TmaV, typename TensorGK, typename TensorSK, typename TensorGV,
          typename TensorSV>
__device__ __forceinline__ void load_paged_kv(TmaK& tma_k, TmaV& tma_v, uint64_t* k_writable,
                                              uint64_t* v_writable, uint64_t* k_readable,
                                              uint64_t* v_readable, TensorGK& tKg, TensorSK& tKs,
                                              TensorGV& tVg, TensorSV& tVs, int ihead_kv,
                                              int num_dim_qk, int num_dim_v, int* block_ids,
                                              int num_blocks, int itile, int istage_write,
                                              int phase) {
  using namespace cute;  // NOLINT

  int load_blocks = kBlockPerTileN;
  int istage = istage_write;

  vec_t<int, kBlockPerTileN> blk_ids;

#pragma unroll
  for (int ikvblock = 0; ikvblock < kBlockPerTileN; ikvblock++) {
    int kvblk_id = itile * kBlockPerTileN + ikvblock;
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
  for (int ikvblock = 0; ikvblock < kBlockPerTileN; ikvblock++) {
    int blk_id = blk_ids[ikvblock];
    cute::copy(tma_k.with(k_readable[istage]), tKg(_, 0, _, ihead_kv, blk_id),
               tKs(_, ikvblock, _, istage));
  }
  set_barrier_transaction_bytes(k_readable[istage],
                                sizeof(Tin) * load_blocks * kBlockSize * num_dim_qk);

  wait_barrier(v_writable[istage], phase);
  // v
#pragma unroll
  for (int ikvblock = 0; ikvblock < kBlockPerTileN; ikvblock++) {
    int blk_id = blk_ids[ikvblock];
    cute::copy(tma_v.with(v_readable[istage]), tVg(_, _, 0, ihead_kv, blk_id),
               tVs(_, _, ikvblock, istage));
  }
  set_barrier_transaction_bytes(v_readable[istage],
                                sizeof(Tin) * load_blocks * kBlockSize * num_dim_v);
}

template <typename TiledMmaQK, typename TensorQ, typename TensorK, typename TensorAtt>
__device__ __forceinline__ void qk_gemm(TiledMmaQK& tiled_mma_qk, TensorQ& tQr, TensorK& tKr,
                                        TensorAtt& tAttr, const int& istage) {
  using namespace cute;  // NOLINT
  tiled_mma_qk.accumulate_ = GMMA::ScaleOut::Zero;
  warpgroup_fence_operand(tAttr);
  warpgroup_arrive();
#pragma unroll
  for (int ik = 0; ik < size<2>(tQr); ++ik) {
    cute::gemm(tiled_mma_qk, tKr(_, _, ik, istage), tQr(_, _, ik), tAttr(_, _, _));
    tiled_mma_qk.accumulate_ = GMMA::ScaleOut::One;
  }
  warpgroup_commit_batch();
  warpgroup_wait<0>();
  warpgroup_fence_operand(tAttr);
}

template <int kTileN, int kHeadsPerGroup, typename TensorAtt, typename TensorI>
__device__ __forceinline__ void apply_casual_mask(TensorAtt& tAttr_nm, TensorI& tI_nm,
                                                  const int& itile_seq_kv,
                                                  const int& num_seq_kvcache,
                                                  const int& num_seq_kv) {
  using namespace cute;  // NOLINT
  constexpr int kN = size<0>(TensorAtt{});
  constexpr int kM = size<1>(TensorAtt{});

#pragma unroll
  for (int im = 0; im < kM; ++im) {
#pragma unroll
    for (int in = 0; in < kN; ++in) {
      int iposq = num_seq_kvcache + get<1>(tI_nm(in, im)) / kHeadsPerGroup;
      int iposk = itile_seq_kv * kTileN + get<0>(tI_nm(in, im));

      if ((iposk > iposq) || (iposk >= num_seq_kv)) {
        tAttr_nm(in, im) = -std::numeric_limits<float>::infinity();
      }
    }
  }
}

template <int kTileN, int kHeadsPerGroup, typename TensorAtt, typename TensorI,
          typename TensorScale>
__device__ __forceinline__ void apply_casual_mask_with_scale(TensorAtt& tAttr_nm, TensorI& tI_nm,
                                                             TensorScale& scales,
                                                             const int& itile_seq_kv,
                                                             const int& num_seq_kvcache,
                                                             const int& num_seq_kv) {
  using namespace cute;  // NOLINT
  constexpr int kN = size<0>(TensorAtt{});
  constexpr int kM = size<1>(TensorAtt{});

#pragma unroll
  for (int im = 0; im < kM; ++im) {
#pragma unroll
    for (int in = 0; in < kN; ++in) {
      int iposq = num_seq_kvcache + get<1>(tI_nm(in, im)) / kHeadsPerGroup;
      int iposk = itile_seq_kv * kTileN + get<0>(tI_nm(in, im));

      if ((iposk > iposq) || (iposk >= num_seq_kv)) {
        tAttr_nm(in, im) = -std::numeric_limits<float>::infinity();
      } else {
        tAttr_nm(in, im) *= scales(im);
      }
    }
  }
}

template <bool kCheckInf, int kTileM, typename TensorA, typename TensorM, typename TensorS,
          typename TensorY, typename TensorScale>
__device__ __forceinline__ void online_softmax(TensorA& tAttr_nm, TensorM& gMax, TensorS& gSum,
                                               TensorY& tYr_nm, TensorScale& one_over_dk_log2e,
                                               float* smem_max, int iwarpgroup, int iwarp,
                                               int ilane) {
  using namespace cute;  // NOLINT
  constexpr int kN = size<0>(TensorA{});
  constexpr int kM = size<1>(TensorA{});
  vec_t<float, kM> warp_max;
#pragma unroll
  for (int im = 0; im < kM; ++im) {
    float row_max = tAttr_nm(0, im);

#pragma unroll
    for (int in = 1; in < kN; ++in) {
      row_max = fmaxf(row_max, tAttr_nm(in, im));
    }

    warp_max[im] = warp_8lane_stride4_reduce_max_xor(row_max) * one_over_dk_log2e(im);
  }

  if (ilane < 4) {
    if constexpr (kM == 2 || kM == 4) {
      store(smem_max + iwarp * kTileM + ilane * kM, warp_max);
    } else if constexpr (kM == 6) {
      vec_t<float, 4> warp_max1 = *reinterpret_cast<vec_t<float, 4>*>(&warp_max[0]);
      vec_t<float, 2> warp_max2 = *reinterpret_cast<vec_t<float, 2>*>(&warp_max[4]);
      store(smem_max + iwarp * kTileM + ilane * 4, warp_max1);
      store(smem_max + iwarp * kTileM + ilane * 2 + 16, warp_max2);
    }
  }

  syncwarpgroup(iwarpgroup);

  if (ilane < 4) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
      vec_t<float, kM> reduce_max;
      if constexpr (kM == 2 || kM == 4) {
        reduce_max = load<float, kM>(smem_max + i * kTileM + ilane * kM);
      } else if constexpr (kM == 6) {
        vec_t<float, 4>& reduce_max1 = *reinterpret_cast<vec_t<float, 4>*>(&reduce_max[0]);
        vec_t<float, 2>& reduce_max2 = *reinterpret_cast<vec_t<float, 2>*>(&reduce_max[4]);

        reduce_max1 = load<float, 4>(smem_max + i * kTileM + ilane * 4);
        reduce_max2 = load<float, 2>(smem_max + i * kTileM + ilane * 2 + 16);
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
      tAttr_nm(in, im) = exp2f_ftz(tAttr_nm(in, im) * one_over_dk_log2e(im) - gMax(im));
      row_sum += tAttr_nm(in, im);
    }

    float scale = exp2f_ftz(last_max - gMax(im));
    gSum(im) = gSum(im) * scale + row_sum;

#pragma unroll
    for (int in = 0; in < cute::size<0>(tYr_nm); ++in) {
      tYr_nm(in, im) = tYr_nm(in, im) * scale;
    }
  }
}

template <typename T, typename TensorIn, typename TensorOut>
__device__ __forceinline__ void cast_fp32reg(TensorIn& tFp32r, TensorOut& tTr) {
  if constexpr (std::is_same_v<T, cute::float_e4m3_t>) {
    auto* raw_float4_pointer = reinterpret_cast<float4*>(tFp32r.data());
    auto* raw_fp8x4_pointer = reinterpret_cast<__nv_fp8x4_e4m3*>(tTr.data());
#pragma unroll
    for (int i = 0; i < size(tFp32r) / 4; ++i) {
      raw_fp8x4_pointer[i] = __nv_fp8x4_e4m3(raw_float4_pointer[i]);
    }
  } else {
#pragma unroll
    for (int i = 0; i < cute::size(tFp32r); ++i) {
      tTr(i) = (T)(tFp32r(i));
    }
  }
}

template <typename TensorP>
__device__ __forceinline__ void permute_p(TensorP& tAttr, unsigned int mask) {
  using namespace cute;  // NOLINT
  auto tAttr_view_as_fp32 = recast<uint32_t>(tAttr);
#pragma unroll
  for (int i = 0; i < size(tAttr_view_as_fp32); ++i) {
    uint32_t val = tAttr_view_as_fp32(i);
    tAttr_view_as_fp32(i) = __byte_perm(val, val, mask);
  }
}

template <typename TiledMmaSV, typename TensorS, typename TensorV, typename TensorY,
          typename TensorVV, typename TensorVVt>
__device__ __forceinline__ void permute_v_sv_gemm(TiledMmaSV& tiled_mma_sv, TensorS& tSr,
                                                  TensorV& tVr, TensorY& tYr, TensorVV& v,
                                                  TensorVVt& vt, int iwarpgroup) {
  using namespace cute;  // NOLINT
  warpgroup_fence_operand(tYr);

#pragma unroll
  for (int iv = 0; iv < size<1>(tVr); iv++) {
#pragma unroll
    for (int in = 0; in < size<2>(tVr); in++) {
      vt(0, iv, in) = __byte_perm(v(0, iv, in), v(1, iv, in), 0x6240);
      vt(1, iv, in) = __byte_perm(v(0, iv, in), v(1, iv, in), 0x7351);

      vt(2, iv, in) = __byte_perm(v(2, iv, in), v(3, iv, in), 0x6240);
      vt(3, iv, in) = __byte_perm(v(2, iv, in), v(3, iv, in), 0x7351);

      warpgroup_arrive();
      cute::gemm(tiled_mma_sv, tVr(_, iv, in), tSr(_, 0, in, iwarpgroup), tYr(_, iv, 0));
      warpgroup_commit_batch();
    }
  }
}

template <typename TiledMmaSV, typename TensorS, typename TensorV, typename TensorY>
__device__ __forceinline__ void sv_gemm(TiledMmaSV& tiled_mma_sv, TensorS& tSr, TensorV& tVr,
                                        TensorY& tYr, const int& istage) {
  using namespace cute;  // NOLINT
  warpgroup_fence_operand(tYr);
  warpgroup_arrive();
  cute::gemm(tiled_mma_sv, tVr(_, _, _, istage), tSr, tYr);
  warpgroup_commit_batch();
  warpgroup_wait<0>();
  warpgroup_fence_operand(tYr);
}

template <int kStage, int kStep = 1>
__device__ __forceinline__ void advance_stage(int& istage, int& phase) {
  istage += kStep;
  if (istage >= kStage) {
    istage = istage % kStage;
    phase ^= 1;
  }
}

template <int kTileM, typename TensorY, typename TensorS>
__device__ __forceinline__ void final_online_softmax(TensorY& tYr_nm, TensorS& gSum,
                                                     float* smem_sum, int iwarpgroup, int iwarp,
                                                     int ilane) {
  using namespace cute;  // NOLINT
  constexpr int kM = size(TensorS{});
  vec_t<float, kM> warp_sum;

#pragma unroll
  for (int im = 0; im < kM; ++im) {
    warp_sum[im] = warp_8lane_stride4_reduce_sum_xor(gSum(im));
  }

  if (ilane < 4) {
    if constexpr (kM == 2 || kM == 4) {
      store(smem_sum + iwarp * kTileM + ilane * kM, warp_sum);
    } else if constexpr (kM == 6) {
      vec_t<float, 4> warp_sum1 = *reinterpret_cast<vec_t<float, 4>*>(&warp_sum[0]);
      vec_t<float, 2> warp_sum2 = *reinterpret_cast<vec_t<float, 2>*>(&warp_sum[4]);
      store(smem_sum + iwarp * kTileM + ilane * 4, warp_sum1);
      store(smem_sum + iwarp * kTileM + ilane * 2 + 16, warp_sum2);
    }
  }

  syncwarpgroup(iwarpgroup);

  if (ilane < 4) {
#pragma unroll
    for (int im = 0; im < kM; ++im) {
      warp_sum[im] = 0.f;
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
      vec_t<float, kM> reduce_sum;
      if constexpr (kM == 2 || kM == 4) {
        reduce_sum = load<float, kM>(smem_sum + i * kTileM + ilane * kM);
      } else if constexpr (kM == 6) {
        vec_t<float, 4>& reduce_sum1 = *reinterpret_cast<vec_t<float, 4>*>(&reduce_sum[0]);
        vec_t<float, 2>& reduce_sum2 = *reinterpret_cast<vec_t<float, 2>*>(&reduce_sum[4]);

        reduce_sum1 = load<float, 4>(smem_sum + i * kTileM + ilane * 4);
        reduce_sum2 = load<float, 2>(smem_sum + i * kTileM + ilane * 2 + 16);
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
    gSum(im) = warp_sum[im];
    float one_over_gsum = 0;
    if (warp_sum[im] != 0) {
      one_over_gsum = rcpf_ftz(warp_sum[im]);
    }
#pragma unroll
    for (int in = 0; in < cute::size<0>(tYr_nm); ++in) {
      tYr_nm(in, im) = tYr_nm(in, im) * one_over_gsum;
    }
  }
}

template <bool kSplitK, int kWarpGroupN, typename TiledCopy, typename TmaY, typename TensorRY,
          typename TensorSY, typename TensorGY>
__device__ __forceinline__ void store_output(TiledCopy& tiled_copy, TmaY& tma_y, TensorRY& rY,
                                             TensorSY& sY, TensorGY& gY, const int& ihead_kv,
                                             const int& ibatch, const int& ichunk,
                                             const int& num_seq_q, const int& idx,
                                             const int& iwarpgroup, bool is_leader_in_warpgroup) {
  using namespace cute;  // NOLINT
  auto thr_copy = tiled_copy.get_slice(idx);
  auto tYr4s = thr_copy.retile_S(rY);
  auto tYs4r = thr_copy.partition_D(sY(_, _, iwarpgroup));

  cute::copy(tiled_copy, tYr4s, tYs4r);
  bar_sync<kWarpGroupN * 128>(kWarpGroupN);
  tma_store_fence();

  auto btma_y = tma_y.get_slice(0);
  // using TMA to store
  if (is_leader_in_warpgroup) {
    auto tYss = btma_y.partition_S(sY(_, _, iwarpgroup));  // (TMA, TMA_M, TMA_N)
    auto tYgg = btma_y.partition_D(gY);                    // (TMA, TMA_M, TMA_N, b)

    for (int iseqq = 0; iseqq < num_seq_q; iseqq++) {
      if constexpr (!kSplitK) {
        cute::copy(tma_y, tYss(_, _, iseqq), tYgg(_, _, 0, ihead_kv, iseqq, ibatch));
      } else {
        cute::copy(tma_y, tYss(_, _, iseqq),
                   tYgg(_, _, 0, ihead_kv, iseqq, ichunk * kWarpGroupN + iwarpgroup, ibatch));
      }
    }
    tma_store_arrive();
  }
}

template <typename TensorMax, typename TensorSum>
__device__ __forceinline__ void store_lse(float* lse_batch, TensorMax& gMax, TensorSum& gSum,
                                          const int& heads_per_group, const int& ilane,
                                          const int& iwarp) {
  constexpr int kM = cute::size(TensorMax{});
  // write lse
  if (iwarp == 0 && ilane * 2 < heads_per_group) {
    vec_t<float, kM> lse;
#pragma unroll
    for (int im = 0; im < kM; ++im) {
      if (gMax(im) == -std::numeric_limits<float>::infinity()) {
        lse[im] = -std::numeric_limits<float>::infinity();
      } else {
        lse[im] = gMax(im) + log2f_ftz(gSum(im));
      }
    }
    auto& lse_store = reshape<kM / 2, 2>(lse);
#pragma unroll
    for (int i = 0; i < kM / 2; i++) {
      store(lse_batch + i * 8 + ilane * 2, lse_store[i]);
    }
  }
}

template <typename Tout, int kTileV, int kSplitK, int kWarps, typename CuteT>
__device__ __forceinline__ void splitk_reduce(CuteT* y_ptr, float* lse_ptr, float* split_y_ptr,
                                              const int& num_chunks, const int& num_seq_q,
                                              const int& num_head_q, const int& num_head_k,
                                              const int& heads_per_group,
                                              const int& lse_heads_per_group, const int& ihead_kv,
                                              const int& ibatch, const int& iwarp,
                                              const int& ilane) {
  constexpr int kItemsPerThread = 4;
  int icol = ilane * kItemsPerThread;

  vec_t<float, kSplitK> lse;
  vec_t<float, kItemsPerThread> output;

  auto* lse_batch = lse_ptr + ibatch * kSplitK * num_head_k * lse_heads_per_group * num_seq_q +
                    ihead_kv * lse_heads_per_group * num_seq_q;
  auto* split_input = split_y_ptr + ibatch * kSplitK * num_head_q * num_seq_q * kTileV +
                      ihead_kv * heads_per_group * kTileV + icol;
  auto* out_row = reinterpret_cast<Tout*>(y_ptr) + ibatch * num_head_q * num_seq_q * kTileV +
                  ihead_kv * heads_per_group * kTileV + icol;

  for (int iseqq = 0; iseqq < num_seq_q; iseqq++) {
    auto* lse_seq = lse_batch + iseqq * lse_heads_per_group;
    auto* split_input_seq = split_input + iseqq * num_head_q * kTileV;
    auto* out_row_seq = out_row + iseqq * num_head_q * kTileV;
    for (int iqhead = iwarp; iqhead < heads_per_group; iqhead += kWarps) {
      auto* lse_head = lse_seq + iqhead;
      auto* split_input_head = split_input_seq + iqhead * kTileV;
      auto* out_row_head = out_row_seq + iqhead * kTileV;
#pragma unroll
      for (int i = 0; i < kItemsPerThread; i++) {
        output[i] = 0.f;
      }

      float max_lse = 0.f;
      float sum_lse = 0.f;

#pragma unroll
      for (int i = 0; i < kSplitK; i++) {
        if (i < num_chunks) {
          lse[i] = lse_head[i * lse_heads_per_group * num_seq_q * num_head_k];
          max_lse = max(max_lse, lse[i]);
        }
      }

#pragma unroll
      for (int i = 0; i < kSplitK; i++) {
        if (i < num_chunks) {
          sum_lse += exp2f_ftz(lse[i] - max_lse);
        }
      }

      sum_lse = log2f_ftz(sum_lse) + max_lse;

#pragma unroll
      for (int i = 0; i < kSplitK; i++) {
        if (i < num_chunks) {
          auto y =
              load<float, kItemsPerThread>(split_input_head + i * num_head_q * num_seq_q * kTileV);
          float scale = exp2f_ftz(lse[i] - sum_lse);
#pragma unroll
          for (int j = 0; j < kItemsPerThread; j++) {
            output[j] += scale * y[j];
          }
        }
      }
      store(out_row_head, to<Tout>(output));
    }
  }
}

/*
Fp8 P interleave
｜< --- kTileM --- >|             warp_sP (16, 8):                T0 Reg:
---------------------               -------------------------           (0, 0), (0, 1)
|         |         |               |T0|T0|T1|T1|T2|T2|T3|T3|           (8, 0), (8, 1)
| (16, 8) |         |               |V0|V1|V0|V1|V0|V1|V0|V1|     will store in SMEM P:
|         |         |            |  -------------------------           (0, 0), (0, 1)
|---------|---------|            |  |T4|T4|T5|T5|T6|T6|T7|T7|           (1, 0), (1, 1)
|         |         |   memory   |  |V0|V1|V0|V1|V0|V1|V0|V1|
|         |         | contiguous |  -------------------------
.         .         .    dir     |              .
.         .         .           \|/             .
.         .         .                           .
|         |         |               |T28      ....       T31|
|         |         |               |V0|V1|V0|V1|V0|V1|V0|V1|
---------------------               -------------------------
*/
template <int kTileM, typename T>
__device__ __forceinline__ auto make_tiled_copy_P_interleave() {
  using namespace cute;  // NOLINT
  using STSM_ATOM =
      std::conditional_t<kTileM % 16 == 0, cute::SM90_U16x4_STSM_T, cute::SM90_U16x2_STSM_T>;
  using R2SCopyAtom = Copy_Atom<STSM_ATOM, T>;

  auto thr_layout = make_layout(make_shape(Int<32>{}, Int<4>{}, Int<1>{}, Int<1>{}),
                                make_stride(Int<4>{}, Int<1>{}, Int<0>{}, Int<0>{}));
  auto val_layout = make_layout(make_shape(Int<2>{}, Int<2>{}, Int<1>{}, Int<kTileM / 8>{}),
                                make_stride(Int<1>{}, Int<2>{}, Int<4>{}, Int<4>{}));

  auto tiler = make_tile(Int<64>{}, Int<kTileM>{});

  auto tiled_copy = make_tiled_copy(R2SCopyAtom{}, thr_layout, val_layout, tiler);

  return tiled_copy;
}

// clang-format off
/*
Fp8 V trans and interleave
SMEMV:
｜< --- kTileN --- >|             warp_sVT (16, 32) = (16, 8) x4:                               T0 in Reg:
---------------------               -------------------------     -------------------------         (0, 0), (0, 1), (0, 8), (0, 9)
|         |         |               |T0|T0|T1|T1|T2|T2|T3|T3|     |T0|T0|T1|T1|T2|T2|T3|T3|         (1, 0), (1, 1), (1, 8), (1, 9)
| ======= |         |               |V0|V1|V0|V1|V0|V1|V0|V1|     |V0|V1|V0|V1|V0|V1|V0|V1|
| (16, 32)|         |   /|\         -------------------------     -------------------------
| ======= |(64, 32) |    |          |T4|T4|T5|T5|T6|T6|T7|T7|     |T4|T4|T5|T5|T6|T6|T7|T7|
|         |         |    |          |V0|V1|V0|V1|V0|V1|V0|V1|     |V0|V1|V0|V1|V0|V1|V0|V1|
| ======= |         |  kTileV       -------------------------     -------------------------
|         |         |  memory                   .                             .
| ======= |---------|contiguous                 .                             .
.         .         .   dir                     .                             .
.         .         .    |          |T28      ....       T31|     |T28      ....       T31|
.         .         .   \|/         |V0|V1|V0|V1|V0|V1|V0|V1|     |V0|V1|V0|V1|V0|V1|V0|V1|
|         |         |               -------------------------     -------------------------
|         |         |                         x1                             x2
---------------------
*/
// clang-format on
template <typename T>
__device__ __forceinline__ auto make_tiled_copy_V_interleave_trans() {
  using namespace cute;  // NOLINT

  auto thr_layout = make_layout(make_shape(Int<32>{}, Int<4>{}, Int<1>{}, Int<1>{}),
                                make_stride(Int<4>{}, Int<1>{}, Int<0>{}, Int<0>{}));
  auto val_layout = make_layout(make_shape(Int<2>{}, Int<2>{}, Int<1>{}, Int<4>{}),
                                make_stride(Int<1>{}, Int<2>{}, Int<4>{}, Int<4>{}));

  auto tiler = make_tile(Int<64>{}, Int<32>{});

  auto tiled_copy =
      make_tiled_copy(Copy_Atom<SM75_U16x8_LDSM_T, T>{}, thr_layout, val_layout, tiler);

  return tiled_copy;
}

template <int kTileM, typename R2SCopyAtom>
__device__ __forceinline__ auto make_tiled_copy_Y_interleave(R2SCopyAtom const& copy_atom) {
  using namespace cute;  // NOLINT

  auto thr_layout = make_layout(make_shape(Int<32>{}, Int<4>{}, Int<1>{}, Int<1>{}),
                                make_stride(Int<4>{}, Int<1>{}, Int<0>{}, Int<0>{}));
  auto val_layout = make_layout(make_shape(Int<2>{}, Int<2>{}, Int<1>{}, Int<kTileM / 8>{}),
                                make_stride(Int<2>{}, Int<1>{}, Int<0>{}, Int<4>{}));

  auto tiler = make_tile(Int<64>{}, Int<kTileM>{});

  auto tiled_copy = make_tiled_copy(copy_atom, thr_layout, val_layout, tiler);

  return tiled_copy;
}

}  // namespace kernels
}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_DECODE_UTIL_KERNELS_CUH_
