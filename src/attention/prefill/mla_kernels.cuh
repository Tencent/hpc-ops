// Copyright 2025 hpc-ops authors

#ifndef SRC_ATTENTION_PREFILL_MLA_KERNELS_CUH_
#define SRC_ATTENTION_PREFILL_MLA_KERNELS_CUH_

#include <cuda.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "src/attention/prefill/kernels.cuh"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace attention {
namespace kernels {

template <typename T, typename TmaQ, typename TmaKV, typename TmaY>
__global__ void update_batched_tma(const vec_t<cute::TmaDescriptor, 3> td_qkvy,
                                   cute::TmaDescriptor *tma_qkvy, const T *q_ptr, const T *kv_ptr,
                                   const T *y_ptr, const int *seqlens_q_ptr,
                                   const int *cu_seqlens_q_ptr, int num_batch, int max_seq_q,
                                   int num_dim_qk, int num_dim_v, int num_head_q, int num_head_kv,
                                   int ldQ, int ldKV, int ldY) {
  using namespace cute;  // NOLINT

  int idx = threadIdx.x;
  int ibatch = blockIdx.x;

  __shared__ cute::TmaDescriptor smem_tma_desc[4];

  int num_seq = seqlens_q_ptr[ibatch];
  int cu_seqlen_q = cu_seqlens_q_ptr[ibatch];
  auto *q_ibatch_ptr = q_ptr + cu_seqlen_q * ldQ;
  auto *kv_ibatch_ptr = kv_ptr + cu_seqlen_q * ldKV;
  auto *y_ibatch_ptr = y_ptr + cu_seqlen_q * ldY;

  if (idx < 3) {
    smem_tma_desc[idx] = td_qkvy[idx];
  }
  __syncwarp();

  // Q
  if (idx == 0) {
    auto gQ = make_tensor(make_gmem_ptr(q_ibatch_ptr), make_shape(num_seq, num_dim_qk, num_head_q),
                          make_stride(ldQ, Int<1>{}, num_dim_qk));
    update_tma_gtensor<TmaQ>(smem_tma_desc[idx], gQ);
  }

  // KV
  if (idx == 1) {
    auto gKV =
        make_tensor(make_gmem_ptr(kv_ibatch_ptr), make_shape(num_seq, num_dim_qk, num_head_kv),
                    make_stride(ldKV, Int<1>{}, num_dim_qk));
    update_tma_gtensor<TmaKV>(smem_tma_desc[idx], gKV);
  }

  // Y
  if (idx == 2) {
    auto gY = make_tensor(make_gmem_ptr(y_ibatch_ptr), make_shape(num_seq, num_dim_v, num_head_q),
                          make_stride(ldY, Int<1>{}, num_dim_v));
    update_tma_gtensor<TmaY>(smem_tma_desc[idx], gY);
  }

#pragma unroll
  for (int i = 0; i < 3; i++) {
    __syncwarp();
    if (cute::elect_one_sync()) {
      cute::tma_desc_commit_group();
      cute::tma_desc_wait_group();
    }
    tma_descriptor_cp_fence_release(tma_qkvy + ibatch * 3 + i, smem_tma_desc[i]);
  }
}

template <typename Config, typename TmaQ, typename TmaKV, typename TmaY>
__global__ void __launch_bounds__(384, 1)
    mla_prefill_bf16_warp_specialization_kernel(cute::TmaDescriptor *td_qkvy, int *seqlens_q_ptr,
                                                int num_batch, int max_seq_q, int num_dim_qk,
                                                int num_dim_v, int num_head_q, int num_head_kv,
                                                float one_over_dk_log2e,
                                                cutlass::FastDivmod head_kv_divmod,
                                                cutlass::FastDivmod head_q_divmod,
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
  __shared__ uint64_t readable_kv[kStage];
  __shared__ uint64_t writable_kv[kStage];
  extern __shared__ uint8_t shm_data[] alignas(128);

  auto *shm_q = reinterpret_cast<Tin *>(shm_data);
  auto *shm_kv = shm_q + cosize(SLayoutQ{});
  auto *shm_y = reinterpret_cast<Tout *>(shm_kv + cosize(SLayoutK{}));

  TmaQ tma_q;
  TmaKV tma_kv;
  TmaY tma_y;

  // Tensor Q/KV/Y
  auto gQ = tma_q.get_tma_tensor(make_shape(max_seq_q, num_dim_qk, num_head_q));
  auto gKV = tma_kv.get_tma_tensor(make_shape(max_seq_q, num_dim_qk, num_head_kv));
  auto gY = tma_y.get_tma_tensor(make_shape(max_seq_q, num_dim_v, num_head_q));

  auto gAtt =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileN>{}), make_stride(Int<kTileN>{}, Int<1>{}));
  auto gYY =
      make_tensor(make_gmem_ptr(static_cast<Tout *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileV>{}), make_stride(Int<kTileV>{}, Int<1>{}));

  // Tensor sQ/sK/sV
  auto sQ = make_tensor(make_smem_ptr(shm_q), SLayoutQ{});   // (TM, TK)
  auto sK = make_tensor(make_smem_ptr(shm_kv), SLayoutK{});  // (TN, TK, kStage)
  auto sV = make_tensor(make_smem_ptr(shm_kv), SLayoutV{});  // (TV, TN, kStage)
  auto sY = make_tensor(make_smem_ptr(shm_y), SLayoutY{});   // (TM, TN)

  // Block Level tma
  auto btma_q = tma_q.get_slice(0);
  auto btma_kv = tma_kv.get_slice(0);
  auto btma_y = tma_y.get_slice(0);

  // Thread Level Tensor
  auto tQg = btma_q.partition_S(gQ);     // ((TM, TK), num_tile_m, num_tile_k, head)
  auto tKVg = btma_kv.partition_S(gKV);  // ((TN, TK), num_tile_n, num_tile_k, head)

  auto tQs = btma_q.partition_D(sQ);    // ((TM, TK), _1, _1)
  auto tKVs = btma_kv.partition_D(sK);  // ((TN, TK), _1, _1, kStage)

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

  auto tAttr = thr_mma_qk.partition_fragment_C(gAtt);  // (MMA, MMA_M, MMA_N)
  auto tYr = thr_mma_pv.partition_fragment_C(gYY);     // (MMA, MMA_M, MMA_V)

  // init q/kv barrier
  if (is_leader_in_block) {
    initialize_barrier(readable_q, 1);
    initialize_barrier(writable_q, 2);
#pragma unroll
    for (int i = 0; i < kStage; ++i) {
      initialize_barrier(readable_kv[i], 1);
      initialize_barrier(writable_kv[i], 2);
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

        auto *td_q = td_qkvy + ibatch * 3;
        auto *td_kv = td_qkvy + ibatch * 3 + 1;

        // Load Q
        wait_barrier(writable_q, phase_q);
        set_barrier_transaction_bytes(readable_q, sizeof(Tin) * cosize(SLayoutQ{}));
        cute::copy(tma_q.with(td_q, readable_q), tQg(_, itile_m, _, ihead_q), tQs(_, 0, _));
        phase_q ^= 1;

        int num_tile_kv = ((itile_m + 1) * kTileM + kTileN - 1) / kTileN;
        constexpr int kTransactionBytesK = sizeof(Tin) * kTileN * kTileK;
        // load K
#pragma unroll 1
        for (int itile_seq_kv = 0; itile_seq_kv < num_tile_kv; ++itile_seq_kv) {
          // k
          wait_barrier(writable_kv[ismem_write], phase);
          set_barrier_transaction_bytes(readable_kv[ismem_write], kTransactionBytesK);
          cute::copy(tma_kv.with(td_kv, readable_kv[ismem_write]),
                     tKVg(_, itile_seq_kv, _, ihead_kv), tKVs(_, 0, _, ismem_write));
          ++ismem_write;
          if (ismem_write == kStage) {
            ismem_write = 0;
            phase ^= 1;
          }
        }  // for
      }  // while
    }  // if (is_leader_in_load)
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
        wait_barrier(readable_kv[ismem_read], phase);

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

        warpgroup_fence_operand(tYr);
        warpgroup_arrive();
        cute::gemm(tiled_mma_pv, tAttAbf16, tVr(_, _, _, ismem_read), tYr);
        warpgroup_commit_batch();
        warpgroup_wait<0>();

        if (elected_idx_in_warpgroup) {
          arrive_barrier(writable_kv[ismem_read]);
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

        auto *td_y = td_qkvy + ibatch * 3 + 2;
        cute::copy(tma_y.with(td_y), tYss(_, iwarpgroup, 0),
                   tYgg(_, itile_m * 2 + iwarpgroup, 0, ihead_q));
        tma_store_arrive();
      }
    }  // while
  }  // else
}
}  // namespace kernels
}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_PREFILL_MLA_KERNELS_CUH_
