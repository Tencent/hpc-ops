// Copyright (C) 2026 Tencent.

#ifndef SRC_ATTENTION_PREFILL_KERNELS_CUH_
#define SRC_ATTENTION_PREFILL_KERNELS_CUH_

#include <cuda.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace attention {
namespace kernels {

template <int kTileM>
__device__ __forceinline__ auto get_next_tile(const int *seqlens_q_ptr, int &iblock, int num_head_q,
                                              int num_batch, int max_total_blocks,
                                              int max_num_tile_m, cutlass::FastDivmod head_q_divmod,
                                              cutlass::FastDivmod tile_m_divmod) {
  int itile_m, ihead_q, ibatch;
  int itile_m_res;
  int num_seq_q, num_tile_m, num_tile_m_res;

  int iblock_res = max_total_blocks - iblock - 1;

  tile_m_divmod(itile_m, itile_m_res, iblock_res);
  head_q_divmod(ibatch, ihead_q, itile_m_res);

  num_seq_q = seqlens_q_ptr[ibatch];
  num_tile_m = (num_seq_q + kTileM - 1) / kTileM;
  num_tile_m_res = max_num_tile_m - num_tile_m;
  itile_m = itile_m - num_tile_m_res;

  iblock = iblock + 2 * (gridDim.x - iblock % gridDim.x) - 1;

  return cute::make_tuple(itile_m, ihead_q, ibatch);
}

template <typename YTensor, typename STensor>
__device__ __forceinline__ void final_online_softmax(YTensor &&tYr_mn, STensor &&gSum, int kM) {
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

template <typename ATensor, typename MTensor, typename STensor, typename YTensor>
__device__ __forceinline__ void online_softmax(ATensor &&tAttr_mn, MTensor &&gMax, STensor &&gSum,
                                               YTensor &&tYr_mn, int kM, int kN,
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

template <typename ATensor, typename MTensor, typename STensor, typename YTensor,
          typename QKSTensor>
__device__ __forceinline__ void online_softmax_with_scale(ATensor &&tAttr_mn, MTensor &&gMax,
                                                          STensor &&gSum, YTensor &&tYr_mn,
                                                          QKSTensor &&tQKS, int kM, int kN,
                                                          float one_over_dk_log2e) {
#pragma unroll
  for (int im = 0; im < kM; ++im) {
    float qks_im = tQKS[im];
    tAttr_mn(im, 0) = tAttr_mn(im, 0) * qks_im;
    float row_max = tAttr_mn(im, 0);
    float row_sum = 0.f;

#pragma unroll
    for (int in = 1; in < kN; ++in) {
      float local_max = tAttr_mn(im, in) * qks_im;
      tAttr_mn(im, in) = local_max;
      row_max = fmaxf(row_max, local_max);
    }

    row_max = warp_4lane_reduce_max_xor(row_max) * one_over_dk_log2e;
    float last_max = gMax(im);
    gMax(im) = fmaxf(last_max, row_max);

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

template <typename T, typename TmaQ, typename TmaK, typename TmaV, typename TmaY>
__global__ void update_batched_tma(const vec_t<cute::TmaDescriptor, 4> td_qkvy,
                                   cute::TmaDescriptor *tma_qkvy, const T *q_ptr, const T *k_ptr,
                                   const T *v_ptr, const T *y_ptr, const int *seqlens_q_ptr,
                                   const int *cu_seqlens_q_ptr, int num_batch, int max_seq_q,
                                   int num_dim_qk, int num_dim_v, int num_head_q, int num_head_kv,
                                   int ldQ, int ldK, int ldV, int ldY) {
  using namespace cute;  // NOLINT

  int idx = threadIdx.x;
  int ibatch = blockIdx.x;

  __shared__ cute::TmaDescriptor smem_tma_desc[4];

  int num_seq = seqlens_q_ptr[ibatch];
  int cu_seqlen_q = cu_seqlens_q_ptr[ibatch];
  auto *q_ibatch_ptr = q_ptr + cu_seqlen_q * ldQ;
  auto *k_ibatch_ptr = k_ptr + cu_seqlen_q * ldK;
  auto *v_ibatch_ptr = v_ptr + cu_seqlen_q * ldV;
  auto *y_ibatch_ptr = y_ptr + cu_seqlen_q * ldY;

  if (idx < 4) {
    smem_tma_desc[idx] = td_qkvy[idx];
  }
  __syncwarp();

  // Q
  if (idx == 0) {
    auto gQ = make_tensor(make_gmem_ptr(q_ibatch_ptr), make_shape(num_seq, num_dim_qk, num_head_q),
                          make_stride(ldQ, Int<1>{}, num_dim_qk));
    update_tma_gtensor<TmaQ>(smem_tma_desc[idx], gQ);
  }

  // K
  if (idx == 1) {
    auto gK = make_tensor(make_gmem_ptr(k_ibatch_ptr), make_shape(num_seq, num_dim_qk, num_head_kv),
                          make_stride(ldK, Int<1>{}, num_dim_qk));
    update_tma_gtensor<TmaK>(smem_tma_desc[idx], gK);
  }

  // V
  if (idx == 2) {
    auto gV = make_tensor(make_gmem_ptr(v_ibatch_ptr), make_shape(num_dim_v, num_seq, num_head_kv),
                          make_stride(Int<1>{}, ldV, num_dim_v));
    update_tma_gtensor<TmaV>(smem_tma_desc[idx], gV);
  }

  // Y
  if (idx == 3) {
    auto gY = make_tensor(make_gmem_ptr(y_ibatch_ptr), make_shape(num_seq, num_dim_v, num_head_q),
                          make_stride(ldY, Int<1>{}, num_dim_v));
    update_tma_gtensor<TmaY>(smem_tma_desc[idx], gY);
  }

#pragma unroll
  for (int i = 0; i < 4; i++) {
    __syncwarp();
    if (cute::elect_one_sync()) {
      cute::tma_desc_commit_group();
      cute::tma_desc_wait_group();
    }
    tma_descriptor_cp_fence_release(tma_qkvy + ibatch * 4 + i, smem_tma_desc[i]);
  }
}

template <typename T1, typename T2, typename TmaQ, typename TmaY>
__global__ void update_batched_tma_with_kvcache(const vec_t<cute::TmaDescriptor, 2> td_qy,
                                                cute::TmaDescriptor *tma_qy, const T1 *q_ptr,
                                                const T2 *y_ptr, const int *cu_seqlens_q_ptr,
                                                int num_batch, int max_seq_q, int num_dim_qk,
                                                int num_dim_v, int num_head_q, int ldQ, int ldY) {
  using namespace cute;  // NOLINT

  int idx = threadIdx.x;
  int ibatch = blockIdx.x;

  __shared__ cute::TmaDescriptor smem_tma_desc[2];

  int num_seq = cu_seqlens_q_ptr[ibatch + 1] - cu_seqlens_q_ptr[ibatch];
  int cu_seqlen_q = cu_seqlens_q_ptr[ibatch];
  auto *q_ibatch_ptr = q_ptr + cu_seqlen_q * ldQ;
  auto *y_ibatch_ptr = y_ptr + cu_seqlen_q * ldY;

  if (idx < 2) {
    smem_tma_desc[idx] = td_qy[idx];
  }
  __syncwarp();

  // Q
  if (idx == 0) {
    auto gQ = make_tensor(make_gmem_ptr(q_ibatch_ptr), make_shape(num_seq, num_dim_qk, num_head_q),
                          make_stride(ldQ, Int<1>{}, num_dim_qk));
    update_tma_gtensor<TmaQ>(smem_tma_desc[idx], gQ);
  }

  // Y
  if (idx == 1) {
    auto gY = make_tensor(make_gmem_ptr(y_ibatch_ptr), make_shape(num_seq, num_dim_v, num_head_q),
                          make_stride(ldY, Int<1>{}, num_dim_v));
    update_tma_gtensor<TmaY>(smem_tma_desc[idx], gY);
  }

#pragma unroll
  for (int i = 0; i < 2; i++) {
    __syncwarp();
    if (cute::elect_one_sync()) {
      cute::tma_desc_commit_group();
      cute::tma_desc_wait_group();
    }
    tma_descriptor_cp_fence_release(tma_qy + ibatch * 2 + i, smem_tma_desc[i]);
  }
}

template <typename Config, typename TmaQ, typename TmaK, typename TmaV, typename TmaY>
__global__ void __launch_bounds__(384, 1) attention_prefill_bf16_warp_specialization_kernel(
    cute::TmaDescriptor *td_qkvy, int *seqlens_q_ptr, int num_batch, int max_seq_q, int num_dim_qk,
    int num_dim_v, int num_head_q, int num_head_kv, float one_over_dk_log2e,
    cutlass::FastDivmod head_kv_divmod, cutlass::FastDivmod head_q_divmod,
    cutlass::FastDivmod tile_m_divmod) {
  using namespace cute;  // NOLINT

  using Tin = typename Config::Tin;
  using Tout = typename Config::Tout;
  using TiledMmaQK = typename Config::TiledMmaQK;
  using TiledMmaPV = typename Config::TiledMmaPV;
  using SLayoutQ = typename Config::SLayoutQ;
  using SLayoutK = typename Config::SLayoutK;
  using SLayoutV = typename Config::SLayoutV;
  using SLayoutY = typename Config::SLayoutY;

  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;
  constexpr int kTileV = Config::kTileV;
  constexpr int kStage = Config::kStage;

  int idx = threadIdx.x;
  int iblock = blockIdx.x;

  int elected = cute::elect_one_sync();
  int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
  bool is_leader_in_block = (iwarp == 0) && elected;
  bool is_leader_in_warpgroup = ((iwarp % 4) == 0) && elected;

  __shared__ uint64_t readable_q;
  __shared__ uint64_t writable_q;
  __shared__ uint64_t readable_k[kStage];
  __shared__ uint64_t writable_k[kStage];
  __shared__ uint64_t readable_v[kStage];
  __shared__ uint64_t writable_v[kStage];
  extern __shared__ uint8_t shm_data[] alignas(128);

  auto *shm_q = reinterpret_cast<Tin *>(shm_data);
  auto *shm_k = shm_q + cosize(SLayoutQ{});
  auto *shm_v = shm_k + cosize(SLayoutK{});
  auto *shm_y = reinterpret_cast<Tout *>(shm_v + cosize(SLayoutV{}));

  TmaQ tma_q;
  TmaK tma_k;
  TmaV tma_v;
  TmaY tma_y;

  // Tensor Q/K/V/Y
  auto gQ = tma_q.get_tma_tensor(make_shape(max_seq_q, num_dim_qk, num_head_q));
  auto gK = tma_k.get_tma_tensor(make_shape(max_seq_q, num_dim_qk, num_head_kv));
  auto gV = tma_v.get_tma_tensor(make_shape(num_dim_v, max_seq_q, num_head_kv));
  auto gY = tma_y.get_tma_tensor(make_shape(max_seq_q, num_dim_v, num_head_q));

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
  auto tQg = btma_q.partition_S(gQ);  // (TMA, TMA_M, TMA_K, head, batch)
  auto tKg = btma_k.partition_S(gK);  // (TMA, TMA_N, TMA_K, head, batch)
  auto tVg = btma_v.partition_S(gV);  // (TMA, TMA_V, TMA_N, head, batch)

  auto tQs = btma_q.partition_D(sQ);  // (TMA, _1, _1)
  auto tKs = btma_k.partition_D(sK);  // (TMA, _1, _1, kStage)
  auto tVs = btma_v.partition_D(sV);  // (TMA, _1, _1, kStage)

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

  // init k/v barrier
  if (is_leader_in_block) {
    initialize_barrier(readable_q, 1);
    initialize_barrier(writable_q, 2);
#pragma unroll
    for (int i = 0; i < kStage; ++i) {
      initialize_barrier(readable_k[i], 1);
      initialize_barrier(writable_k[i], 2);
      initialize_barrier(readable_v[i], 1);
      initialize_barrier(writable_v[i], 2);
    }
  }

  // sync to avoid ahead thread use(wait) readable when it is not initizlized yet
  __syncthreads();

  int max_num_tile_m = (max_seq_q + kTileM - 1) / kTileM;
  int max_total_blocks = num_head_q * num_batch * max_num_tile_m;

  if (idx >= 256) {
    cutlass::arch::warpgroup_reg_dealloc<32>();
    idx -= 256;

    int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
    int is_leader_in_load = ((iwarp == 0) && elected);

    if (is_leader_in_load) {
      int phase = 1;    // start with ok
      int phase_q = 1;  // start with ok
      int ismem_write = __shfl_sync(0xFFFFFFFF, 0, 0);

      while (true) {
        if (iblock >= max_total_blocks) {
          break;
        }

        auto [itile_m, ihead_q, ibatch] =
            get_next_tile<kTileM>(seqlens_q_ptr, iblock, num_head_q, num_batch, max_total_blocks,
                                  max_num_tile_m, head_q_divmod, tile_m_divmod);

        if (itile_m < 0) {
          continue;
        }

        int ihead_kv, res;
        head_kv_divmod(ihead_kv, res, ihead_q);

        auto *td_q = td_qkvy + ibatch * 4;
        auto *td_k = td_qkvy + ibatch * 4 + 1;
        auto *td_v = td_qkvy + ibatch * 4 + 2;

        // Load Q
        wait_barrier(writable_q, phase_q);
        cute::copy(tma_q.with(td_q, readable_q), tQg(_, itile_m, _, ihead_q), tQs(_, 0, _));
        set_barrier_transaction_bytes(readable_q, sizeof(Tin) * cosize(SLayoutQ{}));
        phase_q ^= 1;

        int num_tile_kv = ((itile_m + 1) * kTileM + kTileN - 1) / kTileN;
        constexpr int kTransactionBytesK = sizeof(Tin) * kTileN * kTileK;
        constexpr int kTransactionBytesV = sizeof(Tin) * kTileV * kTileN;
        // load KV
#pragma unroll 1
        for (int itile_seq_kv = 0; itile_seq_kv < num_tile_kv; ++itile_seq_kv) {
          // k
          wait_barrier(writable_k[ismem_write], phase);
          cute::copy(tma_k.with(td_k, readable_k[ismem_write]), tKg(_, itile_seq_kv, _, ihead_kv),
                     tKs(_, 0, _, ismem_write));
          set_barrier_transaction_bytes(readable_k[ismem_write], kTransactionBytesK);

          // v
          wait_barrier(writable_v[ismem_write], phase);
          cute::copy(tma_v.with(td_v, readable_v[ismem_write]), tVg(_, _, itile_seq_kv, ihead_kv),
                     tVs(_, _, 0, ismem_write));
          set_barrier_transaction_bytes(readable_v[ismem_write], kTransactionBytesV);

          ++ismem_write;
          if (ismem_write == kStage) {
            ismem_write = 0;
            phase ^= 1;
          }
        }
      }
    }
  } else {
    cutlass::arch::warpgroup_reg_alloc<168>();

    int idx_in_warpgroup = idx % 128;
    int iwarpgroup = idx / 128;
    int iwarp_in_warpgroup = idx_in_warpgroup / 32;
    int elected_idx_in_warpgroup = ((iwarp_in_warpgroup == 0) && elected);

    auto layout_asC = thr_mma_qk.partition_C(gAtt).layout();
    auto layout_asA = thr_mma_pv.partition_A(gAtt).layout();
    auto tAttA = make_tensor(tAttr.data(), left_inverse(layout_asC).compose(layout_asA));

    auto gI = make_identity_tensor(gAtt.shape());
    auto tI = thr_mma_qk.partition_C(gI);

    auto tAttr_mn = retile_fragment(tAttr);
    constexpr int kM = size<0>(tAttr_mn);
    constexpr int kN = size<1>(tAttr_mn);
    Tensor gMax = make_tensor<float>(Int<kM>{});
    Tensor gSum = make_tensor<float>(Int<kM>{});

    tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;

    int ismem_read = 0;
    int phase = 0;
    int phase_q = 0;

    while (true) {
      if (iblock >= max_total_blocks) {
        break;
      }

      auto [itile_m, ihead_q, ibatch] =
          get_next_tile<kTileM>(seqlens_q_ptr, iblock, num_head_q, num_batch, max_total_blocks,
                                max_num_tile_m, head_q_divmod, tile_m_divmod);

      int num_seq_kv = seqlens_q_ptr[ibatch];
      if (itile_m < 0) {
        continue;
      }

      int ihead_kv, res;
      head_kv_divmod(ihead_kv, res, ihead_q);
      int num_tile_kv = ((itile_m + 1) * kTileM + kTileN - 1) / kTileN;
      int num_tile_full = itile_m * kTileM / kTileN;

      clear(tYr);
      clear(gSum);
      fill(gMax, -std::numeric_limits<float>::infinity());

      wait_barrier(readable_q, phase_q);
#pragma unroll 1
      for (int itile_seq_kv = 0; itile_seq_kv < num_tile_kv; ++itile_seq_kv) {
        wait_barrier(readable_k[ismem_read], phase);

        // P = QK
        tiled_mma_qk.accumulate_ = GMMA::ScaleOut::Zero;

        warpgroup_fence_operand(tAttr);
        warpgroup_arrive();
#pragma unroll
        for (int ik = 0; ik < size<2>(tQr); ++ik) {
          cute::gemm(tiled_mma_qk, tQr(_, _, ik), tKr(_, _, ik, ismem_read), tAttr(_, _, _));
          tiled_mma_qk.accumulate_ = GMMA::ScaleOut::One;
        }

        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(tAttr);

        if (elected_idx_in_warpgroup) {
          arrive_barrier(writable_k[ismem_read]);
        }

        if (itile_seq_kv == (num_tile_kv - 1)) {
          if (elected_idx_in_warpgroup) {
            arrive_barrier(writable_q);
          }
          phase_q ^= 1;
        }

        // do causal mask
        auto tI_mn = retile_fragment(tI);

        if (itile_seq_kv >= num_tile_full) {
#pragma unroll
          for (int im = 0; im < kM; ++im) {
#pragma unroll
            for (int in = 0; in < kN; ++in) {
              int irow = itile_m * kTileM + get<0>(tI_mn(im, in));
              int icol = itile_seq_kv * kTileN + get<1>(tI_mn(im, in));

              if ((icol > irow) || (icol >= num_seq_kv)) {
                tAttr_mn(im, in) = -std::numeric_limits<float>::infinity();
              }
            }
          }
        }

        auto tYr_mn = retile_fragment(tYr);
        // online softmax
        online_softmax(tAttr_mn, gMax, gSum, tYr_mn, kM, kN, one_over_dk_log2e);

        // Y = PV
        auto tAttAbf16 = make_tensor_like<cute::bfloat16_t>(tAttA);
#pragma unroll
        for (int i = 0; i < size(tAttA); ++i) {
          tAttAbf16(i) = (cute::bfloat16_t)(tAttA(i));
        }

        wait_barrier(readable_v[ismem_read], phase);

        warpgroup_fence_operand(tYr);
        warpgroup_arrive();
        cute::gemm(tiled_mma_pv, tAttAbf16, tVr(_, _, _, ismem_read), tYr);
        warpgroup_commit_batch();
        warpgroup_wait<0>();

        if (elected_idx_in_warpgroup) {
          arrive_barrier(writable_v[ismem_read]);
        }

        ++ismem_read;
        if (ismem_read == kStage) {
          phase ^= 1;
          ismem_read = 0;
        }
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
        auto cY = tma_y.get_tma_tensor(make_shape(max_seq_q, num_dim_v, num_head_q));
        auto btma_y = tma_y.get_slice(0);

        auto tYss = btma_y.partition_S(sY);  // (TMA, TMA_M, TMA_N)
        auto tYgg = btma_y.partition_D(cY);  // (TMA, TMA_M, TMA_N, b)

        auto *td_y = td_qkvy + ibatch * 4 + 3;
        cute::copy(tma_y.with(td_y), tYss(_, iwarpgroup, 0),
                   tYgg(_, itile_m * 2 + iwarpgroup, 0, ihead_q));
        tma_store_arrive();
      }
    }
  }
}

template <typename Config, typename TmaQ, typename TmaK, typename TmaV, typename TmaY>
__global__ void __launch_bounds__(384, 1)
    attention_with_kvcache_prefill_bf16_warp_specialization_kernel(
        cute::TmaDescriptor *td_qy, const __grid_constant__ TmaK tma_k,
        const __grid_constant__ TmaV tma_v, const int *cu_seqlens_q_ptr,
        const int *seqlens_kvcache_ptr, const int *block_ids_ptr, int num_batch, int max_seq_q,
        int num_dim_qk, int num_dim_v, int num_head_q, int num_head_kv, int num_kvcache_blocks,
        int block_size, int num_seq_max_blocks, float one_over_dk_log2e,
        cutlass::FastDivmod head_kv_divmod, cutlass::FastDivmod head_q_divmod,
        cutlass::FastDivmod tile_m_divmod) {
  using namespace cute;  // NOLINT

  using Tin = typename Config::Tin;
  using Tout = typename Config::Tout;
  using TiledMmaQK = typename Config::TiledMmaQK;
  using TiledMmaPV = typename Config::TiledMmaPV;
  using SLayoutQ = typename Config::SLayoutQ;
  using SLayoutK = typename Config::SLayoutK;
  using SLayoutV = typename Config::SLayoutV;
  using SLayoutY = typename Config::SLayoutY;

  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;
  constexpr int kTileV = Config::kTileV;
  constexpr int kStage = Config::kStage;
  constexpr int kBlockSize = Config::kBlockSize;

  int idx = threadIdx.x;
  int iblock = blockIdx.x;

  int elected = cute::elect_one_sync();
  int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
  bool is_leader_in_block = (iwarp == 0) && elected;
  bool is_leader_in_warpgroup = ((iwarp % 4) == 0) && elected;

  __shared__ uint64_t readable_q;
  __shared__ uint64_t writable_q;
  __shared__ uint64_t readable_k[kStage];
  __shared__ uint64_t writable_k[kStage];
  __shared__ uint64_t readable_v[kStage];
  __shared__ uint64_t writable_v[kStage];
  extern __shared__ uint8_t shm_data[] alignas(128);

  auto *shm_q = reinterpret_cast<Tin *>(shm_data);
  auto *shm_k = shm_q + cosize(SLayoutQ{});
  auto *shm_v = shm_k + cosize(SLayoutK{});
  auto *shm_y = reinterpret_cast<Tout *>(shm_v + cosize(SLayoutV{}));
  auto *shm_seqlens_q = reinterpret_cast<int *>(shm_y + cosize(SLayoutY{}));
  auto *shm_seqlens_kv = shm_seqlens_q + num_batch;
  auto *shm_seqlens_qstart = shm_seqlens_kv + num_batch;

  TmaQ tma_q;
  TmaY tma_y;

  // Tensor Q/K/V/Y
  auto gQ = tma_q.get_tma_tensor(make_shape(max_seq_q, num_dim_qk, num_head_q));
  auto gK =
      tma_k.get_tma_tensor(make_shape(kBlockSize, num_dim_qk, num_head_kv, num_kvcache_blocks));
  auto gV =
      tma_v.get_tma_tensor(make_shape(num_dim_v, kBlockSize, num_head_kv, num_kvcache_blocks));
  auto gY = tma_y.get_tma_tensor(make_shape(max_seq_q, num_dim_v, num_head_q));

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
  auto tQg = btma_q.partition_S(gQ);  // (TMA, TMA_M, TMA_K, head, batch)
  auto tKg = btma_k.partition_S(gK);  // (TMA, TMA_N, TMA_K, head, batch)
  auto tVg = btma_v.partition_S(gV);  // (TMA, TMA_V, TMA_N, head, batch)

  auto tQs = btma_q.partition_D(sQ);  // (TMA, _1, _1)
  auto tKs = btma_k.partition_D(sK);  // (TMA, _1, _1, kStage)
  auto tVs = btma_v.partition_D(sV);  // (TMA, _1, _1, kStage)

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

  // init k/v barrier
  if (is_leader_in_block) {
    initialize_barrier(readable_q, 1);
    initialize_barrier(writable_q, 2);
#pragma unroll
    for (int i = 0; i < kStage; ++i) {
      initialize_barrier(readable_k[i], 1);
      initialize_barrier(writable_k[i], 2);
      initialize_barrier(readable_v[i], 1);
      initialize_barrier(writable_v[i], 2);
    }
  }

  for (int i = idx; i < num_batch; i += blockDim.x) {
    int num_seq_ibatch = cu_seqlens_q_ptr[i + 1] - cu_seqlens_q_ptr[i];
    shm_seqlens_q[i] = num_seq_ibatch;
    shm_seqlens_kv[i] = seqlens_kvcache_ptr[i];
    shm_seqlens_qstart[i] = seqlens_kvcache_ptr[i] - num_seq_ibatch;
  }

  // sync to avoid ahead thread use(wait) readable when it is not initizlized yet
  __syncthreads();

  int max_num_tile_m = (max_seq_q + kTileM - 1) / kTileM;
  int max_total_blocks = num_head_q * num_batch * max_num_tile_m;

  constexpr int kNumBlockPerTileN = kTileN / kBlockSize;

  if (idx >= 256) {
    cutlass::arch::warpgroup_reg_dealloc<24>();
    idx -= 256;

    int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
    int is_leader_in_load = ((iwarp == 0) && elected);

    if (is_leader_in_load) {
      int phase = 1;    // start with ok
      int phase_q = 1;  // start with ok
      int ismem_write = __shfl_sync(0xFFFFFFFF, 0, 0);

      while (true) {
        if (iblock >= max_total_blocks) {
          break;
        }

        auto [itile_m, ihead_q, ibatch] =
            get_next_tile<kTileM>(shm_seqlens_q, iblock, num_head_q, num_batch, max_total_blocks,
                                  max_num_tile_m, head_q_divmod, tile_m_divmod);

        int num_seq_kv = seqlens_kvcache_ptr[ibatch];
        int num_seq_q = shm_seqlens_q[ibatch];
        if (itile_m < 0) {
          continue;
        }

        int ihead_kv, res;
        head_kv_divmod(ihead_kv, res, ihead_q);

        auto *td_q = td_qy + ibatch * 2;

        int start_seq_q = shm_seqlens_qstart[ibatch];
        auto *block_ids_ibatch_ptr = block_ids_ptr + ibatch * num_seq_max_blocks;

        // Load Q
        wait_barrier(writable_q, phase_q);
        cute::copy(tma_q.with(td_q, readable_q), tQg(_, itile_m, _, ihead_q), tQs(_, 0, _));
        set_barrier_transaction_bytes(readable_q, sizeof(Tin) * cosize(SLayoutQ{}));
        phase_q ^= 1;

        int num_tile_kv = (start_seq_q + (itile_m + 1) * kTileM + kTileN - 1) / kTileN;
        constexpr int kTransactionBytesK = sizeof(Tin) * kTileN * kTileK;
        constexpr int kTransactionBytesV = sizeof(Tin) * kTileV * kTileN;
        // load KV
#pragma unroll 1
        for (int itile_seq_kv = 0; itile_seq_kv < num_tile_kv; ++itile_seq_kv) {
          // k
          wait_barrier(writable_k[ismem_write], phase);

          int iblock_ids[kNumBlockPerTileN];
#pragma unroll
          for (int iblock_kv = 0; iblock_kv < kNumBlockPerTileN; iblock_kv++) {
            iblock_ids[iblock_kv] = 0;
            int iblock_id = itile_seq_kv * kNumBlockPerTileN + iblock_kv;
            if (iblock_id < num_seq_max_blocks) {
              iblock_ids[iblock_kv] = block_ids_ibatch_ptr[iblock_id];
            }
          }

#pragma unroll
          for (int iblock_kv = 0; iblock_kv < kNumBlockPerTileN; iblock_kv++) {
            int iblock_true = iblock_ids[iblock_kv];
            cute::copy(tma_k.with(readable_k[ismem_write]), tKg(_, 0, _, ihead_kv, iblock_true),
                       tKs(_, iblock_kv, _, ismem_write));
          }
          set_barrier_transaction_bytes(readable_k[ismem_write], kTransactionBytesK);

          // v
          wait_barrier(writable_v[ismem_write], phase);
#pragma unroll
          for (int iblock_kv = 0; iblock_kv < kNumBlockPerTileN; iblock_kv++) {
            int iblock_true = iblock_ids[iblock_kv];
            cute::copy(tma_v.with(readable_v[ismem_write]), tVg(_, _, 0, ihead_kv, iblock_true),
                       tVs(_, _, iblock_kv, ismem_write));
          }
          set_barrier_transaction_bytes(readable_v[ismem_write], kTransactionBytesV);

          ++ismem_write;
          if (ismem_write == kStage) {
            ismem_write = 0;
            phase ^= 1;
          }
        }
      }
    }
  } else {
    cutlass::arch::warpgroup_reg_alloc<168>();

    int idx_in_warpgroup = idx % 128;
    int iwarpgroup = idx / 128;
    int iwarp_in_warpgroup = idx_in_warpgroup / 32;
    int elected_idx_in_warpgroup = ((iwarp_in_warpgroup == 0) && elected);

    auto layout_asC = thr_mma_qk.partition_C(gAtt).layout();
    auto layout_asA = thr_mma_pv.partition_A(gAtt).layout();
    auto tAttA = make_tensor(tAttr.data(), left_inverse(layout_asC).compose(layout_asA));

    auto gI = make_identity_tensor(gAtt.shape());
    auto tI = thr_mma_qk.partition_C(gI);

    auto tAttr_mn = retile_fragment(tAttr);
    constexpr int kM = size<0>(tAttr_mn);
    constexpr int kN = size<1>(tAttr_mn);
    Tensor gMax = make_tensor<float>(Int<kM>{});
    Tensor gSum = make_tensor<float>(Int<kM>{});

    tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;

    int ismem_read = 0;
    int phase = 0;
    int phase_q = 0;

    while (true) {
      if (iblock >= max_total_blocks) {
        break;
      }

      auto [itile_m, ihead_q, ibatch] =
          get_next_tile<kTileM>(shm_seqlens_q, iblock, num_head_q, num_batch, max_total_blocks,
                                max_num_tile_m, head_q_divmod, tile_m_divmod);

      if (itile_m < 0) {
        continue;
      }

      int num_seq_kv = shm_seqlens_kv[ibatch];
      int start_seq_q = shm_seqlens_qstart[ibatch];

      int ihead_kv, res;
      head_kv_divmod(ihead_kv, res, ihead_q);
      int num_tile_kv = (start_seq_q + (itile_m + 1) * kTileM + kTileN - 1) / kTileN;
      int num_tile_full = (start_seq_q + itile_m * kTileM) / kTileN;

      clear(tYr);
      clear(gSum);
      fill(gMax, -std::numeric_limits<float>::infinity());

      wait_barrier(readable_q, phase_q);
#pragma unroll 1
      for (int itile_seq_kv = 0; itile_seq_kv < num_tile_kv; ++itile_seq_kv) {
        wait_barrier(readable_k[ismem_read], phase);

        // P = QK
        tiled_mma_qk.accumulate_ = GMMA::ScaleOut::Zero;

        warpgroup_fence_operand(tAttr);
        warpgroup_arrive();
#pragma unroll
        for (int ik = 0; ik < size<2>(tQr); ++ik) {
          cute::gemm(tiled_mma_qk, tQr(_, _, ik), tKr(_, _, ik, ismem_read), tAttr(_, _, _));
          tiled_mma_qk.accumulate_ = GMMA::ScaleOut::One;
        }

        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(tAttr);

        if (elected_idx_in_warpgroup) {
          arrive_barrier(writable_k[ismem_read]);
        }

        if (itile_seq_kv == (num_tile_kv - 1)) {
          if (elected_idx_in_warpgroup) {
            arrive_barrier(writable_q);
          }
          phase_q ^= 1;
        }

        // do causal mask
        auto tI_mn = retile_fragment(tI);

        if (itile_seq_kv >= num_tile_full) {
#pragma unroll
          for (int im = 0; im < kM; ++im) {
#pragma unroll
            for (int in = 0; in < kN; ++in) {
              int irow = start_seq_q + itile_m * kTileM + get<0>(tI_mn(im, in));
              int icol = itile_seq_kv * kTileN + get<1>(tI_mn(im, in));

              if ((icol > irow) || (icol >= num_seq_kv)) {
                tAttr_mn(im, in) = -std::numeric_limits<float>::infinity();
              }
            }
          }
        }

        auto tYr_mn = retile_fragment(tYr);
        // online softmax
        online_softmax(tAttr_mn, gMax, gSum, tYr_mn, kM, kN, one_over_dk_log2e);

        // Y = PV
        auto tAttAbf16 = make_tensor_like<cute::bfloat16_t>(tAttA);
#pragma unroll
        for (int i = 0; i < size(tAttA); ++i) {
          tAttAbf16(i) = (cute::bfloat16_t)(tAttA(i));
        }

        wait_barrier(readable_v[ismem_read], phase);

        warpgroup_fence_operand(tYr);
        warpgroup_arrive();
        cute::gemm(tiled_mma_pv, tAttAbf16, tVr(_, _, _, ismem_read), tYr);
        warpgroup_commit_batch();
        warpgroup_wait<0>();

        if (elected_idx_in_warpgroup) {
          arrive_barrier(writable_v[ismem_read]);
        }

        ++ismem_read;
        if (ismem_read == kStage) {
          phase ^= 1;
          ismem_read = 0;
        }
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
        auto cY = tma_y.get_tma_tensor(make_shape(max_seq_q, num_dim_v, num_head_q));
        auto btma_y = tma_y.get_slice(0);

        auto tYss = btma_y.partition_S(sY);  // (TMA, TMA_M, TMA_N)
        auto tYgg = btma_y.partition_D(cY);  // (TMA, TMA_M, TMA_N, b)

        auto *td_y = td_qy + ibatch * 2 + 1;
        cute::copy(tma_y.with(td_y), tYss(_, iwarpgroup, 0),
                   tYgg(_, itile_m * 2 + iwarpgroup, 0, ihead_q));
        tma_store_arrive();
      }
    }
  }
}

template <typename Config, typename TmaQ, typename TmaK, typename TmaV, typename TmaY,
          typename TmaQKS>
__global__ void __launch_bounds__(384, 1)
    attention_with_kvcache_prefill_fp8_warp_specialization_kernel(
        cute::TmaDescriptor *td_qy, const __grid_constant__ TmaK tma_k,
        const __grid_constant__ TmaV tma_v, const __grid_constant__ TmaQKS tma_qks,
        const float *qkscale_ptr, const float *vscale_ptr, const int *cu_seqlens_q_ptr,
        const int *seqlens_kvcache_ptr, const int *block_ids_ptr, int num_batch, int max_seq_q,
        int max_seq_q_pad, int num_dim_qk, int num_dim_v, int num_head_q, int num_head_kv,
        int num_kvcache_blocks, int block_size, int num_seq_max_blocks, float one_over_dk_log2e,
        cutlass::FastDivmod head_kv_divmod, cutlass::FastDivmod head_q_divmod,
        cutlass::FastDivmod tile_m_divmod) {
  using namespace cute;  // NOLINT

  using Tin = typename Config::Tin;
  using Tout = typename Config::Tout;
  using TiledMmaQK = typename Config::TiledMmaQK;
  using TiledMmaPV = typename Config::TiledMmaPV;
  using SLayoutQ = typename Config::SLayoutQ;
  using SLayoutK = typename Config::SLayoutK;
  using SLayoutV = typename Config::SLayoutV;
  using SLayoutVT = typename Config::SLayoutVT;
  using SLayoutY = typename Config::SLayoutY;
  using SLayoutQKS = typename Config::SLayoutQKS;

  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;
  constexpr int kTileV = Config::kTileV;
  constexpr int kStage = Config::kStage;
  constexpr int kBlockSize = Config::kBlockSize;

  int idx = threadIdx.x;
  int iblock = blockIdx.x;

  int elected = cute::elect_one_sync();
  int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
  bool is_leader_in_block = (iwarp == 0) && elected;
  bool is_leader_in_warpgroup = ((iwarp % 4) == 0) && elected;

  __shared__ uint64_t readable_q;
  __shared__ uint64_t writable_q;
  __shared__ uint64_t readable_k[kStage];
  __shared__ uint64_t writable_k[kStage];
  __shared__ uint64_t readable_v[kStage];
  __shared__ uint64_t writable_v[kStage];
  extern __shared__ uint8_t shm_data[] alignas(128);

  auto *shm_q = reinterpret_cast<Tin *>(shm_data);
  auto *shm_k = shm_q + cosize(SLayoutQ{});
  auto *shm_v = shm_k + cosize(SLayoutK{});
  auto *shm_vt = shm_v + cosize(SLayoutV{});
  auto *shm_y = reinterpret_cast<Tout *>(shm_vt + cosize(SLayoutVT{}));
  auto *shm_qks = reinterpret_cast<float *>(shm_y + cosize(SLayoutY{}));
  auto *shm_seqlens_q = reinterpret_cast<int *>(shm_qks + cosize(SLayoutQKS{}));
  auto *shm_seqlens_kv = shm_seqlens_q + num_batch;
  auto *shm_seqlens_qstart = shm_seqlens_kv + num_batch;

  TmaQ tma_q;
  TmaY tma_y;

  // Tensor Q/K/V/Y
  auto gQ = tma_q.get_tma_tensor(make_shape(max_seq_q, num_dim_qk, num_head_q));
  auto gK =
      tma_k.get_tma_tensor(make_shape(kBlockSize, num_dim_qk, num_head_kv, num_kvcache_blocks));
  auto gV =
      tma_v.get_tma_tensor(make_shape(num_dim_v, kBlockSize, num_head_kv, num_kvcache_blocks));
  auto gY = tma_y.get_tma_tensor(make_shape(max_seq_q, num_dim_v, num_head_q));
  auto gQKS = tma_qks.get_tma_tensor(make_shape(max_seq_q_pad, num_head_q, num_batch));

  auto gAtt =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileN>{}), make_stride(Int<kTileN>{}, Int<1>{}));
  auto gAtt_fp8 =
      make_tensor(make_gmem_ptr(static_cast<Tin *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileN>{}), make_stride(Int<kTileN>{}, Int<1>{}));
  auto gYY =
      make_tensor(make_gmem_ptr(static_cast<Tout *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileV>{}), make_stride(Int<kTileV>{}, Int<1>{}));

  // Tensor sQ/sK/sV
  auto sQ = make_tensor(make_smem_ptr(shm_q), SLayoutQ{});
  auto sK = make_tensor(make_smem_ptr(shm_k), SLayoutK{});
  auto sV = make_tensor(make_smem_ptr(shm_v), SLayoutV{});
  auto sVT = make_tensor(make_smem_ptr(shm_vt), SLayoutVT{});
  auto sY = make_tensor(make_smem_ptr(shm_y), SLayoutY{});
  auto sQKS = make_tensor(make_smem_ptr(shm_qks), SLayoutQKS{});

  // Block Level tma
  auto btma_q = tma_q.get_slice(0);
  auto btma_k = tma_k.get_slice(0);
  auto btma_v = tma_v.get_slice(0);
  auto btma_y = tma_y.get_slice(0);
  auto btma_qks = tma_qks.get_slice(0);

  // Thread Level Tensor
  auto tQg = btma_q.partition_S(gQ);        // (TMA, TMA_M, TMA_K, head, batch)
  auto tKg = btma_k.partition_S(gK);        // (TMA, TMA_N, TMA_K, head, batch)
  auto tVg = btma_v.partition_S(gV);        // (TMA, TMA_V, TMA_N, head, batch)
  auto tQKSg = btma_qks.partition_S(gQKS);  // (TMA, TMA_M, head, batch)

  auto tQs = btma_q.partition_D(sQ);        // (TMA, _1, _1)
  auto tKs = btma_k.partition_D(sK);        // (TMA, _1, _1, kStage)
  auto tVs = btma_v.partition_D(sV);        // (TMA, _1, _1, kStage)
  auto tQKSs = btma_qks.partition_D(sQKS);  // (TMA, _1)

  TiledMmaQK tiled_mma_qk;
  TiledMmaPV tiled_mma_pv;

  auto thr_mma_qk = tiled_mma_qk.get_slice(idx);
  auto thr_mma_pv = tiled_mma_pv.get_slice(idx);

  auto tQs4r = thr_mma_qk.partition_A(sQ);
  auto tKs4r = thr_mma_qk.partition_B(sK);
  auto tVTs4r = thr_mma_pv.partition_B(sVT);

  auto tQr = thr_mma_qk.make_fragment_A(tQs4r);    // (MMA, MMA_M, MMA_K)
  auto tKr = thr_mma_qk.make_fragment_B(tKs4r);    // (MMA, MMA_N, MMA_K)
  auto tVTr = thr_mma_pv.make_fragment_B(tVTs4r);  // (MMA, MMA_V, MMA_N, kStage)

  auto tAttr = thr_mma_qk.partition_fragment_C(gAtt);
  auto tYr = thr_mma_pv.partition_fragment_C(gYY);

  // init k/v barrier
  if (is_leader_in_block) {
    initialize_barrier(readable_q, 1);
    initialize_barrier(writable_q, 2);
#pragma unroll
    for (int i = 0; i < kStage; ++i) {
      initialize_barrier(readable_k[i], 1);
      initialize_barrier(writable_k[i], 2);
      initialize_barrier(readable_v[i], 1);
      initialize_barrier(writable_v[i], 2);
    }
  }

  for (int i = idx; i < num_batch; i += blockDim.x) {
    int num_seq_ibatch = cu_seqlens_q_ptr[i + 1] - cu_seqlens_q_ptr[i];
    shm_seqlens_q[i] = num_seq_ibatch;
    shm_seqlens_kv[i] = seqlens_kvcache_ptr[i];
    shm_seqlens_qstart[i] = seqlens_kvcache_ptr[i] - num_seq_ibatch;
  }

  // sync to avoid ahead thread use(wait) readable when it is not initizlized yet
  __syncthreads();

  int max_num_tile_m = (max_seq_q + kTileM - 1) / kTileM;
  int max_total_blocks = num_head_q * num_batch * max_num_tile_m;

  constexpr int kNumBlockPerTileN = kTileN / kBlockSize;

  if (idx >= 256) {
    cutlass::arch::warpgroup_reg_dealloc<24>();
    idx -= 256;

    int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
    int is_leader_in_load = ((iwarp == 0) && elected);

    if (is_leader_in_load) {
      int phase = 1;    // start with ok
      int phase_q = 1;  // start with ok
      int ismem_write = __shfl_sync(0xFFFFFFFF, 0, 0);

      while (true) {
        if (iblock >= max_total_blocks) {
          break;
        }

        auto [itile_m, ihead_q, ibatch] =
            get_next_tile<kTileM>(shm_seqlens_q, iblock, num_head_q, num_batch, max_total_blocks,
                                  max_num_tile_m, head_q_divmod, tile_m_divmod);

        int num_seq_kv = seqlens_kvcache_ptr[ibatch];
        int num_seq_q = shm_seqlens_q[ibatch];
        if (itile_m < 0) {
          continue;
        }

        int ihead_kv, res;
        head_kv_divmod(ihead_kv, res, ihead_q);

        auto *td_q = td_qy + ibatch * 2;

        int start_seq_q = shm_seqlens_qstart[ibatch];
        auto *block_ids_ibatch_ptr = block_ids_ptr + ibatch * num_seq_max_blocks;

        // Load Q
        wait_barrier(writable_q, phase_q);
        cute::copy(tma_q.with(td_q, readable_q), tQg(_, itile_m, _, ihead_q), tQs(_, 0, _));
        cute::copy(tma_qks.with(readable_q), tQKSg(_, itile_m, ihead_q, ibatch), tQKSs(_, 0));
        set_barrier_transaction_bytes(
            readable_q, sizeof(Tin) * cosize(SLayoutQ{}) + sizeof(float) * cosize(SLayoutQKS{}));
        phase_q ^= 1;

        int num_tile_kv = (start_seq_q + (itile_m + 1) * kTileM + kTileN - 1) / kTileN;
        int num_blocks = (num_seq_kv + kBlockSize - 1) / kBlockSize;
        constexpr int kTransactionBytesK = sizeof(Tin) * kTileN * kTileK;
        constexpr int kTransactionBytesV = sizeof(Tin) * kTileV * kTileN;
        // load KV
#pragma unroll 1
        for (int itile_seq_kv = 0; itile_seq_kv < num_tile_kv; ++itile_seq_kv) {
          // k
          wait_barrier(writable_k[ismem_write], phase);

          int iblock_ids[kNumBlockPerTileN];
#pragma unroll
          for (int iblock_kv = 0; iblock_kv < kNumBlockPerTileN; iblock_kv++) {
            iblock_ids[iblock_kv] = 0;
            int iblock_id = itile_seq_kv * kNumBlockPerTileN + iblock_kv;
            if (iblock_id < num_blocks) {
              iblock_ids[iblock_kv] = block_ids_ibatch_ptr[iblock_id];
            } else {
              iblock_ids[iblock_kv] = -1;
            }
          }

#pragma unroll
          for (int iblock_kv = 0; iblock_kv < kNumBlockPerTileN; iblock_kv++) {
            int iblock_true = iblock_ids[iblock_kv];
            cute::copy(tma_k.with(readable_k[ismem_write]), tKg(_, 0, _, ihead_kv, iblock_true),
                       tKs(_, iblock_kv, _, ismem_write));
          }
          set_barrier_transaction_bytes(readable_k[ismem_write], kTransactionBytesK);

          // v
          wait_barrier(writable_v[ismem_write], phase);
#pragma unroll
          for (int iblock_kv = 0; iblock_kv < kNumBlockPerTileN; iblock_kv++) {
            int iblock_true = iblock_ids[iblock_kv];
            cute::copy(tma_v.with(readable_v[ismem_write]), tVg(_, _, 0, ihead_kv, iblock_true),
                       tVs(_, _, iblock_kv, ismem_write));
          }
          set_barrier_transaction_bytes(readable_v[ismem_write], kTransactionBytesV);

          ++ismem_write;
          if (ismem_write == kStage) {
            ismem_write = 0;
            phase ^= 1;
          }
        }
      }
    }
  } else {
    cutlass::arch::warpgroup_reg_alloc<192>();

    int idx_in_warpgroup = idx % 128;
    int idx_in_warp = idx % 32;
    int iwarpgroup = idx / 128;
    int iwarp_in_warpgroup = idx_in_warpgroup / 32;
    int elected_idx_in_warpgroup = ((iwarp_in_warpgroup == 0) && elected);

    auto tAttr_fp8 = make_tensor_like<Tin>(tAttr);
    auto layout_asC = tAttr.layout();
    auto layout_asA = thr_mma_pv.partition_fragment_A(gAtt_fp8).layout();
    // auto tAttA_fp8 = make_tensor(tAttr_fp8.data(), left_inverse(layout_asC).compose(layout_asA));
    auto tAttA_fp8 = make_tensor(tAttr_fp8.data(), layout_asA);

    auto gI = make_identity_tensor(gAtt.shape());
    auto tI = thr_mma_qk.partition_C(gI);

    auto tAttr_mn = retile_fragment(tAttr);
    constexpr int kM = size<0>(tAttr_mn);
    constexpr int kN = size<1>(tAttr_mn);
    Tensor gMax = make_tensor<float>(Int<kM>{});
    Tensor gSum = make_tensor<float>(Int<kM>{});

    constexpr int kTileTransV = 32;
    constexpr int kTileTransN = 16;
    auto sV_tile =
        local_tile(sV, make_tile(Int<kTileTransV>{}, Int<kTileTransN>{}), make_coord(_, _, _));
    auto sVT_tile =
        local_tile(sVT, make_tile(Int<kTileTransV>{}, Int<kTileTransN>{}), make_coord(_, _, _));

    tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;

    int ismem_read = 0;
    int phase = 0;
    int phase_q = 0;

    float tQKS[kM];
    float vscale = vscale_ptr[0];

    while (true) {
      if (iblock >= max_total_blocks) {
        break;
      }

      auto [itile_m, ihead_q, ibatch] =
          get_next_tile<kTileM>(shm_seqlens_q, iblock, num_head_q, num_batch, max_total_blocks,
                                max_num_tile_m, head_q_divmod, tile_m_divmod);

      if (itile_m < 0) {
        continue;
      }

      int num_seq_kv = shm_seqlens_kv[ibatch];
      int start_seq_q = shm_seqlens_qstart[ibatch];

      int ihead_kv, res;
      head_kv_divmod(ihead_kv, res, ihead_q);
      int num_tile_kv = (start_seq_q + (itile_m + 1) * kTileM + kTileN - 1) / kTileN;
      int num_tile_full = (start_seq_q + itile_m * kTileM) / kTileN;

      clear(tYr);
      clear(gSum);
      fill(gMax, -std::numeric_limits<float>::infinity());

      wait_barrier(readable_q, phase_q);

      auto tI_mn = retile_fragment(tI);
#pragma unroll
      for (int im = 0; im < kM; im++) {
        tQKS[im] = sQKS(get<0>(tI_mn(im, 0)));
      }

#pragma unroll 1
      for (int itile_seq_kv = 0; itile_seq_kv < num_tile_kv; ++itile_seq_kv) {
        wait_barrier(readable_k[ismem_read], phase);

        // P = QK
        tiled_mma_qk.accumulate_ = GMMA::ScaleOut::Zero;

        warpgroup_fence_operand(tAttr);
        warpgroup_arrive();
#pragma unroll
        for (int ik = 0; ik < size<2>(tQr); ++ik) {
          cute::gemm(tiled_mma_qk, tQr(_, _, ik), tKr(_, _, ik, ismem_read), tAttr(_, _, _));
          tiled_mma_qk.accumulate_ = GMMA::ScaleOut::One;
        }

        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(tAttr);

        if (elected_idx_in_warpgroup) {
          arrive_barrier(writable_k[ismem_read]);
        }

        if (itile_seq_kv == (num_tile_kv - 1)) {
          if (elected_idx_in_warpgroup) {
            arrive_barrier(writable_q);
          }
          phase_q ^= 1;
        }

        // do causal mask

        if (itile_seq_kv >= num_tile_full) {
#pragma unroll
          for (int im = 0; im < kM; ++im) {
#pragma unroll
            for (int in = 0; in < kN; ++in) {
              int irow = start_seq_q + itile_m * kTileM + get<0>(tI_mn(im, in));
              int icol = itile_seq_kv * kTileN + get<1>(tI_mn(im, in));

              if ((icol > irow) || (icol >= num_seq_kv)) {
                tAttr_mn(im, in) = -std::numeric_limits<float>::infinity();
              }
            }
          }
        }

        auto tYr_mn = retile_fragment(tYr);
        // online softmax
        online_softmax_with_scale(tAttr_mn, gMax, gSum, tYr_mn, tQKS, kM, kN, one_over_dk_log2e);

        // convert P to fp8 and permute for pv gemm
        auto tAttr_float32x4 = recast<float4>(tAttr);
        auto tAttr_fp8x4 = recast<__nv_fp8x4_e4m3>(tAttr_fp8);

#pragma unroll
        for (int i = 0; i < size(tAttr_float32x4); i++) {
          tAttr_fp8x4(i) = __nv_fp8x4_e4m3(tAttr_float32x4(i));
        }

        auto tAttr_fp8_u64 = recast<uint2>(tAttr_fp8);
#pragma unroll
        for (int i = 0; i < size(tAttr_fp8_u64); i++) {
          uint32_t upper = tAttr_fp8_u64(i).x;
          uint32_t lower = tAttr_fp8_u64(i).y;
          tAttr_fp8_u64(i).x = __byte_perm(upper, lower, 0x5410);
          tAttr_fp8_u64(i).y = __byte_perm(upper, lower, 0x7632);
        }

        // Y = PV
        wait_barrier(readable_v[ismem_read], phase);

        auto sV_tile_iwarp = sV_tile(_, _, _, iwarp_in_warpgroup, ismem_read);
        auto sVT_tile_iwarp = sVT_tile(_, _, _, iwarp_in_warpgroup, ismem_read);

        constexpr int kNumTilePerWarp = 32 / kTileTransN;

        warpgroup_fence_operand(tYr);
#pragma unroll
        for (int in = 0; in < size<2>(tVTr); in++) {
#pragma unroll
          for (int iin = 0; iin < kNumTilePerWarp; iin++) {
            smem_trans_and_interleave0189_nm_nm(
                sV_tile(_, _, iwarp_in_warpgroup, in * kNumTilePerWarp + iin, ismem_read),
                sVT_tile(_, _, iwarp_in_warpgroup, in * kNumTilePerWarp + iin,
                         iwarpgroup * kStage + ismem_read),
                idx_in_warp);
          }
          cutlass::arch::fence_view_async_shared();
          // syncwarpgroup(iwarpgroup);

          warpgroup_arrive();
          cute::gemm(tiled_mma_pv, tAttA_fp8(_, _, in),
                     tVTr(_, _, in, iwarpgroup * kStage + ismem_read), tYr);
          warpgroup_commit_batch();
        }

        warpgroup_wait<0>();

        if (elected_idx_in_warpgroup) {
          arrive_barrier(writable_v[ismem_read]);
        }

        ++ismem_read;
        if (ismem_read == kStage) {
          phase ^= 1;
          ismem_read = 0;
        }
      }

      auto tYr_mn = retile_fragment(tYr);
      // final online softmax
      final_online_softmax(tYr_mn, gSum, kM);

      // to bfloat16
      auto tYr_bf16 = make_tensor_like<Tout>(tYr);

#pragma unroll
      for (int i = 0; i < size(tYr); ++i) {
        tYr_bf16(i) = Tout(tYr(i) * vscale);
      }

      // Epilogue: write register-C to global memory
      using R2SCopyAtomC = Copy_Atom<cute::SM90_U32x4_STSM_N, Tout>;
      auto r2s_tiled_copy = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma_pv);
      auto r2s_thr_copy = r2s_tiled_copy.get_slice(idx);

      auto tYr4s = r2s_thr_copy.retile_S(tYr_bf16);
      auto tYs4r = r2s_thr_copy.partition_D(sY);

      cute::copy(r2s_tiled_copy, tYr4s, tYs4r);
      syncwarpgroup(iwarpgroup);
      tma_store_fence();

      // using TMA to store
      if (is_leader_in_warpgroup) {
        auto cY = tma_y.get_tma_tensor(make_shape(max_seq_q, num_dim_v, num_head_q));
        auto btma_y = tma_y.get_slice(0);

        auto tYss = btma_y.partition_S(sY);  // (TMA, TMA_M, TMA_N)
        auto tYgg = btma_y.partition_D(cY);  // (TMA, TMA_M, TMA_N, b)

        auto *td_y = td_qy + ibatch * 2 + 1;
        cute::copy(tma_y.with(td_y), tYss(_, iwarpgroup, 0),
                   tYgg(_, itile_m * 2 + iwarpgroup, 0, ihead_q));
        tma_store_arrive();
      }
    }
  }
}

template <typename Config, typename TmaQ, typename TmaK, typename TmaV, typename TmaY>
__global__ void attention_prefill_bf16_multi_stage_kernel(cute::TmaDescriptor *td_qkvy,
                                                          int *seqlens_q_ptr, int num_batch,
                                                          int max_seq_q, int num_dim_qk,
                                                          int num_dim_v, int num_head_q,
                                                          int num_head_kv, float one_over_dk_log2e,
                                                          cutlass::FastDivmod HeadKV) {
  using namespace cute;  // NOLINT

  using Tout = typename Config::Tout;
  using Tin = typename Config::Tin;
  using TiledMmaQK = typename Config::TiledMmaQK;
  using TiledMmaPV = typename Config::TiledMmaPV;
  using SLayoutQ = typename Config::SLayoutQ;
  using SLayoutK = typename Config::SLayoutK;
  using SLayoutV = typename Config::SLayoutV;
  using SLayoutY = typename Config::SLayoutY;

  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;
  constexpr int kTileV = Config::kTileV;
  constexpr int kStage = Config::kStage;

  int idx = threadIdx.x;
  int itile_m = blockIdx.x;
  int ihead_q = blockIdx.y;
  int ibatch = blockIdx.z;

  int num_seq_q = seqlens_q_ptr[ibatch];
  int num_seq_kv = num_seq_q;
  if (itile_m * kTileM >= num_seq_q) {
    return;
  }

  int ihead_kv, res;
  HeadKV(ihead_kv, res, ihead_q);

  int elected = cute::elect_one_sync();
  int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
  bool is_leader_in_block = (iwarp == 0) && elected;

  __shared__ uint64_t bar_q;
  __shared__ uint64_t bar_k[kStage];
  __shared__ uint64_t bar_v[kStage];
  extern __shared__ uint8_t shm_data[] alignas(128);

  auto *shm_q = reinterpret_cast<Tin *>(shm_data);
  auto *shm_k = shm_q + cosize(SLayoutQ{});
  auto *shm_v = shm_k + cosize(SLayoutK{});
  auto *shm_y = reinterpret_cast<Tout *>(shm_data);  // Reuse All

  TmaQ tma_q;
  TmaK tma_k;
  TmaV tma_v;
  TmaY tma_y;

  auto *td_q = td_qkvy + ibatch * 4;
  auto *td_k = td_qkvy + ibatch * 4 + 1;
  auto *td_v = td_qkvy + ibatch * 4 + 2;
  auto *td_y = td_qkvy + ibatch * 4 + 3;

  // Tensor Q/K/V/Y
  auto gQ = tma_q.get_tma_tensor(make_shape(max_seq_q, num_dim_qk, num_head_q));
  auto gK = tma_k.get_tma_tensor(make_shape(max_seq_q, num_dim_qk, num_head_kv));
  auto gV = tma_v.get_tma_tensor(make_shape(num_dim_v, max_seq_q, num_head_kv));
  auto gY = tma_y.get_tma_tensor(make_shape(max_seq_q, num_dim_v, num_head_q));

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
  auto tQg = btma_q.partition_S(gQ);  // (TMA, TMA_M, TMA_K, head, batch)
  auto tKg = btma_k.partition_S(gK);  // (TMA, TMA_N, TMA_K, head, batch)
  auto tVg = btma_v.partition_S(gV);  // (TMA, TMA_V, TMA_N, head, batch)

  auto tQs = btma_q.partition_D(sQ);  // (TMA, _1, _1)
  auto tKs = btma_k.partition_D(sK);  // (TMA, _1, _1, kStage)
  auto tVs = btma_v.partition_D(sV);  // (TMA, _1, _1, kStage)

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
    cute::copy(tma_q.with(td_q, bar_q), tQg(_, itile_m, _, ihead_q), tQs(_, 0, _));
    set_barrier_transaction_bytes(bar_q, sizeof(Tin) * cosize(SLayoutQ{}));
  }

  // init k/v barrier
  if (is_leader_in_block) {
#pragma unroll
    for (int i = 0; i < kStage; ++i) {
      initialize_barrier(bar_k[i], 1);
      initialize_barrier(bar_v[i], 1);
    }
  }

  // sync to avoid ahead thread use(wait) readable when it is not initizlized yet
  __syncthreads();
  wait_barrier(bar_q, 0);

  constexpr int kTransactionBytesK = sizeof(Tin) * kTileN * kTileK;
  constexpr int kTransactionBytesV = sizeof(Tin) * kTileV * kTileN;
  int ismem_write = 0;
#pragma unroll
  for (int istage = 0; istage < kStage - 1; ++istage) {
    if (is_leader_in_block) {
      // k
      cute::copy(tma_k.with(td_k, bar_k[istage]), tKg(_, istage, _, ihead_kv),
                 tKs(_, 0, _, istage));
      set_barrier_transaction_bytes(bar_k[istage], kTransactionBytesK);

      // v
      cute::copy(tma_v.with(td_v, bar_v[istage]), tVg(_, _, istage, ihead_kv),
                 tVs(_, _, 0, istage));
      set_barrier_transaction_bytes(bar_v[istage], kTransactionBytesV);

      ++ismem_write;
    }
  }

  auto layout_asC = thr_mma_qk.partition_C(gAtt).layout();
  auto layout_asA = thr_mma_pv.partition_A(gAtt).layout();
  auto tAttA = make_tensor(tAttr.data(), left_inverse(layout_asC).compose(layout_asA));

  clear(tYr);

  tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;

  int num_tile_kv = min(((itile_m + 1) * kTileM + kTileN - 1) / kTileN, size<1>(tKg));
  int num_tile_full = min(itile_m * kTileM, num_seq_kv) / kTileN;

  int phase = 0;
  int ismem_read = 0;
  int itile_write = ismem_write;
#pragma unroll 1
  for (int itile_seq_kv = 0; itile_seq_kv < num_tile_kv; ++itile_seq_kv) {
    // load k/scale/v
    if (itile_write < num_tile_kv) {
      if (is_leader_in_block) {
        // k
        cute::copy(tma_k.with(td_k, bar_k[ismem_write]), tKg(_, itile_write, _, ihead_kv),
                   tKs(_, 0, _, ismem_write));
        set_barrier_transaction_bytes(bar_k[ismem_write], kTransactionBytesK);

        // v
        cute::copy(tma_v.with(td_v, bar_v[ismem_write]), tVg(_, _, itile_write, ihead_kv),
                   tVs(_, _, 0, ismem_write));
        set_barrier_transaction_bytes(bar_v[ismem_write], kTransactionBytesV);

        ismem_write = (ismem_write + 1) % kStage;
      }
    }
    ++itile_write;

    wait_barrier(bar_k[ismem_read], phase);

    // P = QK
    tiled_mma_qk.accumulate_ = GMMA::ScaleOut::Zero;

    warpgroup_fence_operand(tAttr);
    warpgroup_arrive();
#pragma unroll
    for (int ik = 0; ik < size<2>(tQr); ++ik) {
      cute::gemm(tiled_mma_qk, tQr(_, _, ik), tKr(_, _, ik, ismem_read), tAttr(_, _, _));
      tiled_mma_qk.accumulate_ = GMMA::ScaleOut::One;
    }

    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(tAttr);

    // do causal mask
    auto tAttr_mn = retile_fragment(tAttr);
    auto tI_mn = retile_fragment(tI);
    if (itile_seq_kv >= num_tile_full) {
#pragma unroll
      for (int im = 0; im < kM; ++im) {
#pragma unroll
        for (int in = 0; in < kN; ++in) {
          int irow = itile_m * kTileM + get<0>(tI_mn(im, in));
          int icol = itile_seq_kv * kTileN + get<1>(tI_mn(im, in));

          if ((icol > irow) || (icol >= num_seq_kv)) {
            tAttr_mn(im, in) = -std::numeric_limits<float>::infinity();
          }
        }
      }
    }

    auto tYr_mn = retile_fragment(tYr);
    // online softmax
    online_softmax(tAttr_mn, gMax, gSum, tYr_mn, kM, kN, one_over_dk_log2e);

    // Y = PV
    auto tAttAbf16 = make_tensor_like<cute::bfloat16_t>(tAttA);
#pragma unroll
    for (int i = 0; i < size(tAttA); ++i) {
      tAttAbf16(i) = (cute::bfloat16_t)(tAttA(i));
    }

    wait_barrier(bar_v[ismem_read], phase);

    warpgroup_fence_operand(tYr);
    warpgroup_arrive();
    cute::gemm(tiled_mma_pv, tAttAbf16, tVr(_, _, _, ismem_read), tYr);
    warpgroup_commit_batch();
    warpgroup_wait<0>();

    ++ismem_read;
    if (ismem_read == kStage) {
      phase ^= 1;
      ismem_read = 0;
    }

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
    auto cY = tma_y.get_tma_tensor(make_shape(max_seq_q, num_dim_v, num_head_q));
    auto btma_y = tma_y.get_slice(0);

    auto tYss = btma_y.partition_S(sY);  // (TMA, TMA_M, TMA_N)
    auto tYgg = btma_y.partition_D(cY);  // (TMA, TMA_M, TMA_N, b)

    cute::copy(tma_y.with(td_y), tYss(_, 0, 0), tYgg(_, itile_m, 0, ihead_q));
  }
}

template <typename Config, typename TmaQ, typename TmaK, typename TmaV, typename TmaY>
__global__ void attention_with_kvcache_prefill_bf16_multi_stage_kernel(
    cute::TmaDescriptor *td_qy, const __grid_constant__ TmaK tma_k,
    const __grid_constant__ TmaV tma_v, const int *cu_seqlens_q_ptr, const int *seqlens_kvcache_ptr,
    const int *block_ids_ptr, int num_batch, int max_seq_q, int num_dim_qk, int num_dim_v,
    int num_head_q, int num_head_kv, int num_kvcache_blocks, int block_size, int num_seq_max_blocks,
    float one_over_dk_log2e, cutlass::FastDivmod HeadKV) {
  using namespace cute;  // NOLINT

  using Tout = typename Config::Tout;
  using Tin = typename Config::Tin;
  using TiledMmaQK = typename Config::TiledMmaQK;
  using TiledMmaPV = typename Config::TiledMmaPV;
  using SLayoutQ = typename Config::SLayoutQ;
  using SLayoutK = typename Config::SLayoutK;
  using SLayoutV = typename Config::SLayoutV;
  using SLayoutY = typename Config::SLayoutY;

  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;
  constexpr int kTileV = Config::kTileV;
  constexpr int kStage = Config::kStage;
  constexpr int kBlockSize = Config::kBlockSize;

  int idx = threadIdx.x;
  int itile_m = blockIdx.x;
  int ihead_q = blockIdx.y;
  int ibatch = blockIdx.z;

  int num_seq_q = cu_seqlens_q_ptr[ibatch + 1] - cu_seqlens_q_ptr[ibatch];
  int num_seq_kv = seqlens_kvcache_ptr[ibatch];
  if (itile_m * kTileM >= num_seq_q) {
    return;
  }

  int ihead_kv, res;
  HeadKV(ihead_kv, res, ihead_q);

  int elected = cute::elect_one_sync();
  int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
  bool is_leader_in_block = (iwarp == 0) && elected;

  __shared__ uint64_t bar_q;
  __shared__ uint64_t bar_k[kStage];
  __shared__ uint64_t bar_v[kStage];
  extern __shared__ uint8_t shm_data[] alignas(128);

  auto *shm_q = reinterpret_cast<Tin *>(shm_data);
  auto *shm_k = shm_q + cosize(SLayoutQ{});
  auto *shm_v = shm_k + cosize(SLayoutK{});
  auto *shm_y = reinterpret_cast<Tout *>(shm_data);  // Reuse All

  TmaQ tma_q;
  TmaY tma_y;

  auto *td_q = td_qy + ibatch * 2;
  auto *td_y = td_qy + ibatch * 2 + 1;

  // Tensor Q/K/V/Y
  auto gQ = tma_q.get_tma_tensor(make_shape(max_seq_q, num_dim_qk, num_head_q));
  auto gK =
      tma_k.get_tma_tensor(make_shape(kBlockSize, num_dim_qk, num_head_kv, num_kvcache_blocks));
  auto gV =
      tma_v.get_tma_tensor(make_shape(num_dim_v, kBlockSize, num_head_kv, num_kvcache_blocks));
  auto gY = tma_y.get_tma_tensor(make_shape(max_seq_q, num_dim_v, num_head_q));

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
  auto tQg = btma_q.partition_S(gQ);  // (TMA, TMA_M, TMA_K, head, batch)
  auto tKg = btma_k.partition_S(gK);  // (TMA, TMA_N, TMA_K, head, batch)
  auto tVg = btma_v.partition_S(gV);  // (TMA, TMA_V, TMA_N, head, batch)

  auto tQs = btma_q.partition_D(sQ);  // (TMA, _1, _1)
  auto tKs = btma_k.partition_D(sK);  // (TMA, _1, _1, kStage)
  auto tVs = btma_v.partition_D(sV);  // (TMA, _1, _1, kStage)

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
    cute::copy(tma_q.with(td_q, bar_q), tQg(_, itile_m, _, ihead_q), tQs(_, 0, _));
    set_barrier_transaction_bytes(bar_q, sizeof(Tin) * cosize(SLayoutQ{}));
  }

  // init k/v barrier
  if (is_leader_in_block) {
#pragma unroll
    for (int i = 0; i < kStage; ++i) {
      initialize_barrier(bar_k[i], 1);
      initialize_barrier(bar_v[i], 1);
    }
  }

  // sync to avoid ahead thread use(wait) readable when it is not initizlized yet
  __syncthreads();
  wait_barrier(bar_q, 0);

  constexpr int kNumBlockPerTileN = kTileN / kBlockSize;
  constexpr int kTransactionBytesK = sizeof(Tin) * kTileN * kTileK;
  constexpr int kTransactionBytesV = sizeof(Tin) * kTileV * kTileN;

  auto *block_ids_ibatch_ptr = block_ids_ptr + ibatch * num_seq_max_blocks;

  int ismem_write = 0;
#pragma unroll
  for (int istage = 0; istage < kStage - 1; ++istage) {
    if (is_leader_in_block) {
      // k

      int iblock_ids[kNumBlockPerTileN];
#pragma unroll
      for (int iblock_kv = 0; iblock_kv < kNumBlockPerTileN; iblock_kv++) {
        iblock_ids[iblock_kv] = 0;
        int iblock_id = istage * kNumBlockPerTileN + iblock_kv;
        if (iblock_id < num_seq_max_blocks) {
          iblock_ids[iblock_kv] = block_ids_ibatch_ptr[iblock_id];
        }
      }
#pragma unroll
      for (int iblock_kv = 0; iblock_kv < kNumBlockPerTileN; iblock_kv++) {
        int iblock_true = iblock_ids[iblock_kv];
        cute::copy(tma_k.with(bar_k[istage]), tKg(_, 0, _, ihead_kv, iblock_true),
                   tKs(_, iblock_kv, _, istage));
      }
      set_barrier_transaction_bytes(bar_k[istage], kTransactionBytesK);

      // v
#pragma unroll
      for (int iblock_kv = 0; iblock_kv < kNumBlockPerTileN; iblock_kv++) {
        int iblock_true = iblock_ids[iblock_kv];
        cute::copy(tma_v.with(bar_v[istage]), tVg(_, _, 0, ihead_kv, iblock_true),
                   tVs(_, _, iblock_kv, istage));
      }
      set_barrier_transaction_bytes(bar_v[istage], kTransactionBytesV);

      ++ismem_write;
    }
  }

  auto layout_asC = thr_mma_qk.partition_C(gAtt).layout();
  auto layout_asA = thr_mma_pv.partition_A(gAtt).layout();
  auto tAttA = make_tensor(tAttr.data(), left_inverse(layout_asC).compose(layout_asA));

  clear(tYr);

  tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;

  int start_seq_q = num_seq_kv - num_seq_q;
  int num_tile_kv = (start_seq_q + (itile_m + 1) * kTileM + kTileN - 1) / kTileN;
  int num_tile_full = (start_seq_q + itile_m * kTileM) / kTileN;

  int phase = 0;
  int ismem_read = 0;
  int itile_write = ismem_write;
#pragma unroll 1
  for (int itile_seq_kv = 0; itile_seq_kv < num_tile_kv; ++itile_seq_kv) {
    // load k/scale/v
    if (itile_write < num_tile_kv) {
      if (is_leader_in_block) {
        // k
        int iblock_ids[kNumBlockPerTileN];
#pragma unroll
        for (int iblock_kv = 0; iblock_kv < kNumBlockPerTileN; iblock_kv++) {
          iblock_ids[iblock_kv] = 0;
          int iblock_id = itile_write * kNumBlockPerTileN + iblock_kv;
          if (iblock_id < num_seq_max_blocks) {
            iblock_ids[iblock_kv] = block_ids_ibatch_ptr[iblock_id];
          }
        }
#pragma unroll
        for (int iblock_kv = 0; iblock_kv < kNumBlockPerTileN; iblock_kv++) {
          int iblock_true = iblock_ids[iblock_kv];
          cute::copy(tma_k.with(bar_k[ismem_write]), tKg(_, 0, _, ihead_kv, iblock_true),
                     tKs(_, iblock_kv, _, ismem_write));
        }
        set_barrier_transaction_bytes(bar_k[ismem_write], kTransactionBytesK);

        // v
#pragma unroll
        for (int iblock_kv = 0; iblock_kv < kNumBlockPerTileN; iblock_kv++) {
          int iblock_true = iblock_ids[iblock_kv];
          cute::copy(tma_v.with(bar_v[ismem_write]), tVg(_, _, 0, ihead_kv, iblock_true),
                     tVs(_, _, iblock_kv, ismem_write));
        }
        set_barrier_transaction_bytes(bar_v[ismem_write], kTransactionBytesV);

        ismem_write = (ismem_write + 1) % kStage;
      }
    }
    ++itile_write;

    wait_barrier(bar_k[ismem_read], phase);

    // P = QK
    tiled_mma_qk.accumulate_ = GMMA::ScaleOut::Zero;

    warpgroup_fence_operand(tAttr);
    warpgroup_arrive();
#pragma unroll
    for (int ik = 0; ik < size<2>(tQr); ++ik) {
      cute::gemm(tiled_mma_qk, tQr(_, _, ik), tKr(_, _, ik, ismem_read), tAttr(_, _, _));
      tiled_mma_qk.accumulate_ = GMMA::ScaleOut::One;
    }

    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(tAttr);

    // do causal mask
    auto tAttr_mn = retile_fragment(tAttr);
    auto tI_mn = retile_fragment(tI);
    if (itile_seq_kv >= num_tile_full) {
#pragma unroll
      for (int im = 0; im < kM; ++im) {
#pragma unroll
        for (int in = 0; in < kN; ++in) {
          int irow = start_seq_q + itile_m * kTileM + get<0>(tI_mn(im, in));
          int icol = itile_seq_kv * kTileN + get<1>(tI_mn(im, in));

          if ((icol > irow) || (icol >= num_seq_kv)) {
            tAttr_mn(im, in) = -std::numeric_limits<float>::infinity();
          }
        }
      }
    }

    auto tYr_mn = retile_fragment(tYr);
    // online softmax
    online_softmax(tAttr_mn, gMax, gSum, tYr_mn, kM, kN, one_over_dk_log2e);

    // Y = PV
    auto tAttAbf16 = make_tensor_like<cute::bfloat16_t>(tAttA);
#pragma unroll
    for (int i = 0; i < size(tAttA); ++i) {
      tAttAbf16(i) = (cute::bfloat16_t)(tAttA(i));
    }

    wait_barrier(bar_v[ismem_read], phase);

    warpgroup_fence_operand(tYr);
    warpgroup_arrive();
    cute::gemm(tiled_mma_pv, tAttAbf16, tVr(_, _, _, ismem_read), tYr);
    warpgroup_commit_batch();
    warpgroup_wait<0>();

    ++ismem_read;
    if (ismem_read == kStage) {
      phase ^= 1;
      ismem_read = 0;
    }

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
    auto cY = tma_y.get_tma_tensor(make_shape(max_seq_q, num_dim_v, num_head_q));
    auto btma_y = tma_y.get_slice(0);

    auto tYss = btma_y.partition_S(sY);  // (TMA, TMA_M, TMA_N)
    auto tYgg = btma_y.partition_D(cY);  // (TMA, TMA_M, TMA_N, b)

    cute::copy(tma_y.with(td_y), tYss(_, 0, 0), tYgg(_, itile_m, 0, ihead_q));
  }
}

}  // namespace kernels
}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_PREFILL_KERNELS_CUH_
