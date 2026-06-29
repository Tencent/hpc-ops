// Copyright 2025 hpc-ops authors

#ifndef SRC_ATTENTION_PREFILL_WARP_SPEC_WITH_KVCACHE_FP8_DIM128_HYBRID_MASK_CUH_
#define SRC_ATTENTION_PREFILL_WARP_SPEC_WITH_KVCACHE_FP8_DIM128_HYBRID_MASK_CUH_

#include <cuda.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "src/attention/prefill/hybrid_mask_common.cuh"
#include "src/attention/prefill/kernels.cuh"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace attention {
namespace kernels {

template <typename Config, typename TmaQ, typename TmaK, typename TmaV, typename TmaY,
          typename TmaQS, bool kHybridMask = false>
__global__ void __launch_bounds__(384, 1)
    attention_with_kvcache_qpertoken_perhead_kvpertensor_prefill_fp8_warp_specialization_hybrid_mask_kernel(  // NOLINT(whitespace/line_length)
        cute::TmaDescriptor *td_qy, const __grid_constant__ TmaK tma_k,
        const __grid_constant__ TmaV tma_v, const __grid_constant__ TmaQS tma_qs,
        const float *qscale_ptr, const float *kscale_ptr, const float *pscale_ptr,
        const float *vscale_ptr, const int *cu_seqlens_q_ptr, const int *seqlens_kvcache_ptr,
        const int *block_ids_ptr, const int *mm_prefix_range_ptr, int max_spans, int num_batch,
        int max_seq_q, int max_seq_q_pad, int num_dim_qk, int num_dim_v, int num_head_q,
        int num_head_kv, int num_kvcache_blocks, int block_size, int num_seq_max_blocks,
        float one_over_dk_log2e, cutlass::FastDivmod head_kv_divmod,
        cutlass::FastDivmod head_q_divmod, cutlass::FastDivmod tile_m_divmod) {
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
  using SLayoutQS = typename Config::SLayoutQS;

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
  auto *shm_qs = reinterpret_cast<float *>(shm_y + cosize(SLayoutY{}));
  auto *shm_seqlens_q = reinterpret_cast<int *>(shm_qs + cosize(SLayoutQS{}));
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
  auto gQS = tma_qs.get_tma_tensor(make_shape(max_seq_q_pad, num_head_q, num_batch));

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
  auto sQS = make_tensor(make_smem_ptr(shm_qs), SLayoutQS{});

  // Block Level tma
  auto btma_q = tma_q.get_slice(0);
  auto btma_k = tma_k.get_slice(0);
  auto btma_v = tma_v.get_slice(0);
  auto btma_y = tma_y.get_slice(0);
  auto btma_qs = tma_qs.get_slice(0);

  // Thread Level Tensor
  auto tQg = btma_q.partition_S(gQ);     // (TMA, TMA_M, TMA_K, head, batch)
  auto tKg = btma_k.partition_S(gK);     // (TMA, TMA_N, TMA_K, head, batch)
  auto tVg = btma_v.partition_S(gV);     // (TMA, TMA_V, TMA_N, head, batch)
  auto tQSg = btma_qs.partition_S(gQS);  // (TMA, TMA_M, head, batch)

  auto tQs = btma_q.partition_D(sQ);     // (TMA, _1, _1)
  auto tKs = btma_k.partition_D(sK);     // (TMA, _1, _1, kStage)
  auto tVs = btma_v.partition_D(sV);     // (TMA, _1, _1, kStage)
  auto tQSs = btma_qs.partition_D(sQS);  // (TMA, _1)

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
        cute::copy(tma_qs.with(readable_q), tQSg(_, itile_m, ihead_q, ibatch), tQSs(_, 0));
        set_barrier_transaction_bytes(
            readable_q, sizeof(Tin) * cosize(SLayoutQ{}) + sizeof(float) * cosize(SLayoutQS{}));
        phase_q ^= 1;

        int num_tile_kv = get_num_tile_kv<kTileM, kTileN, kHybridMask>(
            mm_prefix_range_ptr, max_spans, ibatch, start_seq_q, itile_m, num_seq_q, num_seq_kv);
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

    float tQS[kM];
    float kscale = kscale_ptr[0];
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
      int num_tile_kv = get_num_tile_kv<kTileM, kTileN, kHybridMask>(
          mm_prefix_range_ptr, max_spans, ibatch, start_seq_q, itile_m, shm_seqlens_q[ibatch],
          num_seq_kv);
      int num_tile_full = (start_seq_q + itile_m * kTileM) / kTileN;
      int gM[kM];
      get_bounds<kM, kHybridMask>(mm_prefix_range_ptr, max_spans, ibatch, start_seq_q,
                                  itile_m * kTileM, 1, num_seq_kv, tI, gM);

      float pscale = (pscale_ptr != nullptr) ? pscale_ptr[ihead_q] : 256.0f;

      clear(tYr);
      clear(gSum);
      fill(gMax, -std::numeric_limits<float>::infinity());

      wait_barrier(readable_q, phase_q);

      auto tI_mn = retile_fragment(tI);
#pragma unroll
      for (int im = 0; im < kM; im++) {
        tQS[im] = sQS(get<0>(tI_mn(im, 0))) * kscale;
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

        if (itile_seq_kv >= num_tile_full) {
#pragma unroll
          for (int im = 0; im < kM; ++im) {
#pragma unroll
            for (int in = 0; in < kN; ++in) {
              int icol = itile_seq_kv * kTileN + get<1>(tI_mn(im, in));

              if (icol >= gM[im]) {
                tAttr_mn(im, in) = -std::numeric_limits<float>::infinity();
              } else {
                tAttr_mn(im, in) *= tQS[im];
              }
            }
          }
        } else {
#pragma unroll
          for (int im = 0; im < kM; ++im) {
#pragma unroll
            for (int in = 0; in < kN; ++in) {
              tAttr_mn(im, in) *= tQS[im];
            }
          }
        }

        auto tYr_mn = retile_fragment(tYr);
        // online softmax
        // online_softmax_with_scale(tAttr_mn, gMax, gSum, tYr_mn, tQS, kM, kN, one_over_dk_log2e);
        online_softmax(tAttr_mn, gMax, gSum, tYr_mn, kM, kN, one_over_dk_log2e);

        // Scale P before FP8 quantization; epilogue applies the reciprocal.
#pragma unroll
        for (int i = 0; i < size(tAttr); ++i) {
          tAttr(i) *= pscale;
        }

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

      float vscale_eff = vscale / pscale;
#pragma unroll
      for (int i = 0; i < size(tYr); ++i) {
        tYr_bf16(i) = Tout(tYr(i) * vscale_eff);
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

template <typename Config, typename TmaQ, typename TmaK, typename TmaV, typename TmaY,
          typename TmaQS, typename TmaKS, bool kHybridMask = false>
__global__ void __launch_bounds__(384, 1)
    attention_with_kvcache_qkpertoken_perhead_vperhead_prefill_fp8_warp_specialization_hybrid_mask_kernel(  // NOLINT(whitespace/line_length)
        cute::TmaDescriptor *td_qy, const __grid_constant__ TmaK tma_k,
        const __grid_constant__ TmaV tma_v, const __grid_constant__ TmaQS tma_qs,
        const __grid_constant__ TmaKS tma_ks, const float *qscale_ptr, const float *kscale_ptr,
        const float *pscale_ptr, const float *vscale_ptr, const int *cu_seqlens_q_ptr,
        const int *seqlens_kvcache_ptr, const int *block_ids_ptr, const int *mm_prefix_range_ptr,
        int max_spans, int num_batch, int max_seq_q, int max_seq_q_pad, int num_dim_qk,
        int num_dim_v, int num_dim_scale, int num_head_q, int num_head_kv, int num_kvcache_blocks,
        int num_scale_blocks, int block_size, int num_seq_max_blocks, float one_over_dk_log2e,
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
  using SLayoutQS = typename Config::SLayoutQS;
  using SLayoutKS = typename Config::SLayoutKS;
  using SLayoutKSC = typename Config::SLayoutKSC;

  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;
  constexpr int kTileV = Config::kTileV;
  constexpr int kStage = Config::kStage;
  constexpr int kBlockSize = Config::kBlockSize;
  constexpr int kScaleBlockSize = Config::kScaleBlockSize;

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
  auto *shm_qs = reinterpret_cast<float *>(shm_y + cosize(SLayoutY{}));
  auto *shm_ks = reinterpret_cast<float *>(shm_qs + cosize(SLayoutQS{}));
  auto *shm_seqlens_q = reinterpret_cast<int *>(shm_ks + cosize(SLayoutKSC{}));
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
  auto gQS = tma_qs.get_tma_tensor(make_shape(max_seq_q_pad, num_head_q, num_batch));
  auto gKS = tma_ks.get_tma_tensor(
      make_shape(kScaleBlockSize, num_dim_scale, num_head_kv, num_scale_blocks));

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
  auto sQS = make_tensor(make_smem_ptr(shm_qs), SLayoutQS{});
  auto sKS = make_tensor(make_smem_ptr(shm_ks), SLayoutKS{});
  auto sKSC = make_tensor(make_smem_ptr(shm_ks), SLayoutKSC{});

  // Block Level tma
  auto btma_q = tma_q.get_slice(0);
  auto btma_k = tma_k.get_slice(0);
  auto btma_v = tma_v.get_slice(0);
  auto btma_y = tma_y.get_slice(0);
  auto btma_qs = tma_qs.get_slice(0);
  auto btma_ks = tma_ks.get_slice(0);

  // Thread Level Tensor
  auto tQg = btma_q.partition_S(gQ);     // (TMA, TMA_M, TMA_K, head, batch)
  auto tKg = btma_k.partition_S(gK);     // (TMA, TMA_N, TMA_K, head, batch)
  auto tVg = btma_v.partition_S(gV);     // (TMA, TMA_V, TMA_N, head, batch)
  auto tQSg = btma_qs.partition_S(gQS);  // (TMA, TMA_M, head, batch)
  auto tKSg = btma_ks.partition_S(gKS);  // (TMA, TMA_M, head, batch)

  auto tQs = btma_q.partition_D(sQ);      // (TMA, _1, _1)
  auto tKs = btma_k.partition_D(sK);      // (TMA, _1, _1, kStage)
  auto tVs = btma_v.partition_D(sV);      // (TMA, _1, _1, kStage)
  auto tQSs = btma_qs.partition_D(sQS);   // (TMA, _1)
  auto tKSs = btma_ks.partition_D(sKSC);  // (TMA, _1)

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
        cute::copy(tma_qs.with(readable_q), tQSg(_, itile_m, ihead_q, ibatch), tQSs(_, 0));
        set_barrier_transaction_bytes(
            readable_q, sizeof(Tin) * cosize(SLayoutQ{}) + sizeof(float) * cosize(SLayoutQS{}));
        phase_q ^= 1;

        int num_tile_kv = get_num_tile_kv<kTileM, kTileN, kHybridMask>(
            mm_prefix_range_ptr, max_spans, ibatch, start_seq_q, itile_m, num_seq_q, num_seq_kv);
        int num_blocks = (num_seq_kv + kBlockSize - 1) / kBlockSize;
        constexpr int kTransactionBytesK = sizeof(Tin) * kTileN * kTileK + sizeof(float) * kTileN;
        // constexpr int kTransactionBytesK = sizeof(Tin) * kTileN * kTileK;
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
            cute::copy(tma_ks.with(readable_k[ismem_write]), tKSg(_, 0, _, ihead_kv, iblock_true),
                       tKSs(_, iblock_kv, _, ismem_write));
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
    cutlass::arch::warpgroup_reg_alloc<224>();

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

    float tQS[kM];
    float tKS[kN];
    // float kscale = kscale_ptr[0];
    // float vscale = vscale_ptr[0];

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
      int num_tile_kv = get_num_tile_kv<kTileM, kTileN, kHybridMask>(
          mm_prefix_range_ptr, max_spans, ibatch, start_seq_q, itile_m, shm_seqlens_q[ibatch],
          num_seq_kv);
      int num_tile_full = (start_seq_q + itile_m * kTileM) / kTileN;
      int gM[kM];
      get_bounds<kM, kHybridMask>(mm_prefix_range_ptr, max_spans, ibatch, start_seq_q,
                                  itile_m * kTileM, 1, num_seq_kv, tI, gM);

      float vscale = vscale_ptr[ihead_kv];
      float pscale = (pscale_ptr != nullptr) ? pscale_ptr[ihead_q] : 256.0f;

      clear(tYr);
      clear(gSum);
      fill(gMax, -std::numeric_limits<float>::infinity());

      wait_barrier(readable_q, phase_q);

      auto tI_mn = retile_fragment(tI);
#pragma unroll
      for (int im = 0; im < kM; im++) {
        tQS[im] = sQS(get<0>(tI_mn(im, 0)));
      }

#pragma unroll 1
      for (int itile_seq_kv = 0; itile_seq_kv < num_tile_kv; ++itile_seq_kv) {
        wait_barrier(readable_k[ismem_read], phase);

#pragma unroll
        for (int ins = 0; ins < kN; ins++) {
          tKS[ins] = sKS(get<1>(tI_mn(0, ins)), ismem_read);
        }

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

        if (itile_seq_kv >= num_tile_full) {
#pragma unroll
          for (int im = 0; im < kM; ++im) {
#pragma unroll
            for (int in = 0; in < kN; ++in) {
              int icol = itile_seq_kv * kTileN + get<1>(tI_mn(im, in));

              if (icol >= gM[im]) {
                tAttr_mn(im, in) = -std::numeric_limits<float>::infinity();
              } else {
                tAttr_mn(im, in) *= tQS[im] * tKS[in];
              }
            }
          }
        } else {
#pragma unroll
          for (int im = 0; im < kM; ++im) {
#pragma unroll
            for (int in = 0; in < kN; ++in) {
              tAttr_mn(im, in) *= tQS[im] * tKS[in];
            }
          }
        }

        auto tYr_mn = retile_fragment(tYr);
        // online softmax
        // online_softmax_with_scale(tAttr_mn, gMax, gSum, tYr_mn, tQS, tKS, kM, kN,
        // one_over_dk_log2e);
        online_softmax(tAttr_mn, gMax, gSum, tYr_mn, kM, kN, one_over_dk_log2e);

        // Scale P before FP8 quantization; epilogue applies the reciprocal.
#pragma unroll
        for (int i = 0; i < size(tAttr); ++i) {
          tAttr(i) *= pscale;
        }

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

      float vscale_eff = vscale / pscale;
#pragma unroll
      for (int i = 0; i < size(tYr); ++i) {
        tYr_bf16(i) = Tout(tYr(i) * vscale_eff);
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

}  // namespace kernels
}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_PREFILL_WARP_SPEC_WITH_KVCACHE_FP8_DIM128_HYBRID_MASK_CUH_
