// Copyright 2025 hpc-ops authors

#ifndef SRC_INDEXER_KERNELS_CUH_
#define SRC_INDEXER_KERNELS_CUH_

#include <cuda.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace indexer {
namespace kernels {

__device__ __forceinline__ auto get_next_tile(const int *cu_seqlens_q_ptr, const int &iblock,
                                              const int &start_batch, const int &num_batch,
                                              cutlass::FastDivmod num_split_divider) {
  int itoken = 0;
  int ichunk = 0;

  // (q,r,s) -> s=q*d+r
  num_split_divider(itoken, ichunk, iblock);

  int ibatch = 0;
  int itoken_in_batch = itoken;

  for (int i = start_batch + 1; i < num_batch + 1; i++) {
    int cu_seqlenq = cu_seqlens_q_ptr[i];
    if (itoken < cu_seqlenq) {
      ibatch = i - 1;
      itoken_in_batch = itoken - cu_seqlens_q_ptr[ibatch];
      break;
    }
  }
  return cute::make_tuple(itoken, ichunk, ibatch, itoken_in_batch);
}

template <typename Tin, typename Tout, int kTileM, int kTileN, int kTileK, int kWarpGroupN,
          int kBlockSize, int kStageQ, int kStageK, int kRatio, typename TiledMma, typename TmaQ,
          typename TmaK, typename SLayoutQ, typename SLayoutK>
__global__ void __launch_bounds__(384, 1)
    mqa_indexer_logits_bf16_swap_kernel(const __grid_constant__ TmaQ tma_q,
                                        const __grid_constant__ TmaK tma_k, const Tin *w_ptr,
                                        Tout *output_ptr, const int *cu_seqlens_q_ptr,
                                        const int *seqlens_kv_ptr, const int *block_ids_ptr,
                                        int num_batch, int total_seq_q, int num_head_q,
                                        int head_dim, int num_kvcache_blocks,
                                        int num_seq_max_blocks, int max_context_len,
                                        cutlass::FastDivmod num_split_divider) {
  using namespace cute;  // NOLINT

  int idx = threadIdx.x;
  int iblock = blockIdx.x;

  int elected = cute::elect_one_sync();
  int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
  bool is_leader_in_block = (iwarp == 0) && elected;

  __shared__ uint64_t readable_q[kStageQ];
  __shared__ uint64_t writable_q[kStageQ];

  __shared__ uint64_t readable_k[kStageK];
  __shared__ uint64_t writable_k[kStageK];
  extern __shared__ uint8_t shm_data[] alignas(128);

  auto *shm_q = reinterpret_cast<Tin *>(shm_data);
  auto *shm_k = shm_q + cosize(SLayoutQ{});
  auto *shm_w = shm_k + cosize(SLayoutK{});
  auto *shm_cu_seqlenq = reinterpret_cast<int *>(shm_w + kTileM * kStageQ);
  auto *shm_seqlenkv = shm_cu_seqlenq + num_batch + 1;

  // Tensor Q/K/V/Y
  auto gQ = tma_q.get_tma_tensor(make_shape(num_head_q, head_dim, total_seq_q));
  auto gK = tma_k.get_tma_tensor(make_shape(kBlockSize, head_dim, num_kvcache_blocks));
  auto gYY =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileN>{}, Int<kTileM>{}), make_stride(Int<kTileM>{}, Int<1>{}));

  // Tensor sQ/sK/sV
  auto sQ = make_tensor(make_smem_ptr(shm_q), SLayoutQ{});
  auto sK = make_tensor(make_smem_ptr(shm_k), SLayoutK{});

  // init k/v barrier
  if (is_leader_in_block) {
#pragma unroll
    for (int i = 0; i < kStageQ; ++i) {
      initialize_barrier(readable_q[i], 1);
      initialize_barrier(writable_q[i], kWarpGroupN);
    }
#pragma unroll
    for (int i = 0; i < kStageK; ++i) {
      initialize_barrier(readable_k[i], 1);
      initialize_barrier(writable_k[i], 1);
    }
  }

  for (int i = idx; i < num_batch + 1; i += blockDim.x) {
    shm_cu_seqlenq[i] = cu_seqlens_q_ptr[i];
  }

  for (int i = idx; i < num_batch; i += blockDim.x) {
    shm_seqlenkv[i] = seqlens_kv_ptr[i];
  }

  // sync to avoid ahead thread use(wait) readable when it is not initizlized yet
  __syncthreads();

  if (idx >= kWarpGroupN * 128) {
    cutlass::arch::warpgroup_reg_dealloc<24>();
    idx -= kWarpGroupN * 128;

    // Block Level tma
    auto btma_q = tma_q.get_slice(0);
    auto btma_k = tma_k.get_slice(0);

    // Thread Level Tensor
    auto tQg = btma_q.partition_S(gQ);  // (TMA, TMA_M, TMA_K, head, batch)
    auto tKg = btma_k.partition_S(gK);  // (TMA, TMA_N, TMA_K, head, batch)

    auto tQs = btma_q.partition_D(sQ);  // (TMA, _1, _1)
    auto tKs = btma_k.partition_D(sK);  // (TMA, _1, _1, kStage)

    int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
    int is_leader_in_load = ((iwarp == 0) && elected);

    constexpr int kNumBlockPerTileN = kTileN / kBlockSize;

    if (is_leader_in_load) {
      int phase_k = 1;  // start with ok
      int phase_q = 1;  // start with ok
      int ismemk_write = __shfl_sync(0xFFFFFFFF, 0, 0);
      int ismemq_write = __shfl_sync(0xFFFFFFFF, 0, 0);
      int start_batch = 0;

      while (true) {
        if (iblock >= total_seq_q * num_split_divider.divisor) {
          break;
        }
        auto [itoken, ichunk, ibatch, itoken_in_batch] =
            get_next_tile(shm_cu_seqlenq, iblock, start_batch, num_batch, num_split_divider);

        start_batch = ibatch;
        iblock += gridDim.x;

        int num_seq_q = shm_cu_seqlenq[ibatch + 1] - shm_cu_seqlenq[ibatch];
        int num_seq_kv = shm_seqlenkv[ibatch];
        int num_seq_kv_ratio = ((num_seq_kv - num_seq_q) + itoken_in_batch + 1) / kRatio;
        auto *block_ids_ibatch_ptr = block_ids_ptr + ibatch * num_seq_max_blocks;

        int num_blocks = (num_seq_kv_ratio + kBlockSize - 1) / kBlockSize;
        int num_tile_kv = (num_seq_kv_ratio + kTileN - 1) / kTileN;

        int num_tiles_per_chunk;
        int num_tiles_last_chunk;
        num_split_divider(num_tiles_per_chunk, num_tiles_last_chunk, num_tile_kv);

        int tile_kv_begin = ichunk * num_tiles_per_chunk;
        int tile_kv_end = (ichunk == num_split_divider.divisor - 1)
                              ? num_tile_kv
                              : (tile_kv_begin + num_tiles_per_chunk);

        constexpr int kTransactionBytesQ = sizeof(Tin) * kTileM * (kTileK + 1);
        constexpr int kTransactionBytesK = sizeof(Tin) * kTileN * kTileK;

        if (tile_kv_begin == tile_kv_end) {
          continue;
        }

        // Load Q
        wait_barrier(writable_q[ismemq_write], phase_q);
        cute::copy(tma_q.with(readable_q[ismemq_write]), tQg(_, 0, _, itoken),
                   tQs(_, 0, _, ismemq_write));
        cp_async_g2s(shm_w + ismemq_write * kTileM, w_ptr + itoken * num_head_q,
                     kTileM * sizeof(Tin), &readable_q[ismemq_write]);
        set_barrier_transaction_bytes(readable_q[ismemq_write], kTransactionBytesQ);

        ++ismemq_write;
        if (ismemq_write == kStageQ) {
          ismemq_write = 0;
          phase_q ^= 1;
        }

        // load KV
#pragma unroll 1
        for (int itile_seq_kv = tile_kv_begin; itile_seq_kv < tile_kv_end; ++itile_seq_kv) {
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

          // k
          wait_barrier(writable_k[ismemk_write], phase_k);
#pragma unroll
          for (int iblock_kv = 0; iblock_kv < kNumBlockPerTileN; iblock_kv++) {
            int iblock_true = iblock_ids[iblock_kv];
            cute::copy(tma_k.with(readable_k[ismemk_write]), tKg(_, 0, _, iblock_true),
                       tKs(_, iblock_kv, _, ismemk_write));
          }
          set_barrier_transaction_bytes(readable_k[ismemk_write], kTransactionBytesK);

          ++ismemk_write;
          if (ismemk_write == kStageK) {
            ismemk_write = 0;
            phase_k ^= 1;
          }
        }
      }
    }
  } else {
    cutlass::arch::warpgroup_reg_alloc<168>();

    int iwarpgroup = idx / 128;
    int idx_in_warpgroup = idx % 128;
    int ilane = idx % 32;
    int iwarp_in_warpgroup = idx_in_warpgroup / 32;
    int elected_idx_in_warpgroup = ((iwarp_in_warpgroup == 0) && elected);

    TiledMma tiled_mma;

    auto thr_mma = tiled_mma.get_slice(idx_in_warpgroup);

    auto tKs4r = thr_mma.partition_A(sK);
    auto tQs4r = thr_mma.partition_B(sQ);

    auto tKr = thr_mma.make_fragment_A(tKs4r);  // (MMA, MMA_M, MMA_K)
    auto tQr = thr_mma.make_fragment_B(tQs4r);  // (MMA, MMA_N, MMA_K)

    auto tYr = thr_mma.partition_fragment_C(gYY);

    auto tYr_nm = retile_fragment(tYr);
    auto gI = make_identity_tensor(gYY.shape());
    auto tI = thr_mma.partition_C(gI);
    auto tI_nm = retile_fragment(tI);

    constexpr int kN = size<0>(tYr_nm);
    constexpr int kM = size<1>(tYr_nm);

    Tensor gSum = make_tensor<float>(Int<kN>{});
    Tensor Ws = make_tensor<float>(Int<kM>{});

    int ismemq_read = 0;
    int ismemk_read = iwarpgroup;
    int phase_q = 0;
    int phase_k = 0;

    int start_batch = 0;
    int start_warpgroup = 0;
    while (true) {
      if (iblock >= total_seq_q * num_split_divider.divisor) {
        break;
      }

      auto [itoken, ichunk, ibatch, itoken_in_batch] =
          get_next_tile(shm_cu_seqlenq, iblock, start_batch, num_batch, num_split_divider);

      start_batch = ibatch;
      iblock += gridDim.x;

      auto *output =
          output_ptr + static_cast<uint64_t>(itoken) * static_cast<uint64_t>(max_context_len);
      int num_seq_q = shm_cu_seqlenq[ibatch + 1] - shm_cu_seqlenq[ibatch];
      int num_seq_kv = shm_seqlenkv[ibatch];
      int num_seq_kvcache = num_seq_kv - num_seq_q;
      int num_seq_kv_ratio = (num_seq_kvcache + itoken_in_batch + 1) / kRatio;

      int num_tile_kv = (num_seq_kv_ratio + kTileN - 1) / kTileN;
      int num_tiles_per_chunk;
      int num_tiles_last_chunk;
      num_split_divider(num_tiles_per_chunk, num_tiles_last_chunk, num_tile_kv);

      int tile_kv_begin = ichunk * num_tiles_per_chunk;
      int tile_kv_end = (ichunk == num_split_divider.divisor - 1)
                            ? num_tile_kv
                            : (tile_kv_begin + num_tiles_per_chunk);

      if (tile_kv_begin == tile_kv_end) {
        continue;
      }

      int num_tile_kv_local = tile_kv_end - tile_kv_begin;
      tile_kv_begin += (start_warpgroup + iwarpgroup) % kWarpGroupN;

      wait_barrier(readable_q[ismemq_read], phase_q);
#pragma unroll
      for (int im = 0; im < kM; ++im) {
        int ihead = get<1>(tI_nm(0, im));
        Ws(im) = static_cast<float>(shm_w[ismemq_read * kTileM + ihead]);
      }
#pragma unroll 1
      for (int itile_seq_kv = tile_kv_begin; itile_seq_kv < tile_kv_end;
           itile_seq_kv += kWarpGroupN) {
        wait_barrier(readable_k[ismemk_read], phase_k);
        // P = QK
        tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;

        warpgroup_fence_operand(tYr);
        warpgroup_arrive();
#pragma unroll
        for (int ik = 0; ik < size<2>(tQr); ++ik) {
          cute::gemm(tiled_mma, tKr(_, _, ik, ismemk_read), tQr(_, _, ik, ismemq_read),
                     tYr(_, _, _));
          tiled_mma.accumulate_ = GMMA::ScaleOut::One;
        }

        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(tYr);

        if (elected_idx_in_warpgroup) {
          arrive_barrier(writable_k[ismemk_read]);
        }

        ismemk_read += kWarpGroupN;
        if (ismemk_read >= kStageK) {
          phase_k ^= 1;
          ismemk_read %= kStageK;
        }

        clear(gSum);
#pragma unroll
        for (int in = 0; in < kN; ++in) {
#pragma unroll
          for (int im = 0; im < kM; ++im) {
            gSum(in) += relu(tYr_nm(in, im)) * Ws(im);
          }
          gSum(in) = warp_4lane_reduce_sum_xor(gSum(in));
        }

        // store output
        if (ilane % 4 == 0) {
#pragma unroll
          for (int in = 0; in < kN; ++in) {
            int iseqk = itile_seq_kv * kTileN + get<0>(tI_nm(in, 0));
            if (iseqk < num_seq_kv_ratio) {
              store(output + iseqk, gSum(in));
            }
          }
        }
      }

      start_warpgroup = (start_warpgroup + num_tile_kv_local) % kWarpGroupN;
      if (elected_idx_in_warpgroup) {
        arrive_barrier(writable_q[ismemq_read]);
      }
      ++ismemq_read;
      if (ismemq_read == kStageQ) {
        ismemq_read = 0;
        phase_q ^= 1;
      }
    }
  }
}

template <typename Tin, typename Tout, int kTileM, int kTileN, int kTileK, int kWarpGroupN,
          int kBlockSize, int kStageQ, int kStageK, int kRatio, typename TiledMma, typename TmaQ,
          typename TmaK, typename SLayoutQ, typename SLayoutK>
__global__ void __launch_bounds__(384, 1)
    mqa_indexer_logits_swap_fp8_kernel(const __grid_constant__ TmaQ tma_q,
                                       const __grid_constant__ TmaK tma_k,
                                       const cute::bfloat16_t *w_ptr, Tout *output_ptr,
                                       const int *cu_seqlens_q_ptr, const int *seqlens_kv_ptr,
                                       const int *block_ids_ptr, int num_batch, int total_seq_q,
                                       int num_head_q, int head_dim, int num_kvcache_blocks,
                                       int num_seq_max_blocks, int max_context_len,
                                       cutlass::FastDivmod num_split_divider) {
  using namespace cute;  // NOLINT

  int idx = threadIdx.x;
  int iblock = blockIdx.x;

  int elected = cute::elect_one_sync();
  int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
  bool is_leader_in_block = (iwarp == 0) && elected;

  __shared__ uint64_t readable_q[kStageQ];
  __shared__ uint64_t writable_q[kStageQ];

  __shared__ uint64_t readable_k[kStageK];
  __shared__ uint64_t writable_k[kStageK];
  extern __shared__ uint8_t shm_data[] alignas(128);

  auto *shm_q = reinterpret_cast<Tin *>(shm_data);
  auto *shm_k = shm_q + cosize(SLayoutQ{});
  auto *shm_w = reinterpret_cast<cute::bfloat16_t *>(shm_k + cosize(SLayoutK{}));
  auto *shm_cu_seqlenq = reinterpret_cast<int *>(shm_w + kTileM * kStageQ);
  auto *shm_seqlenkv = shm_cu_seqlenq + num_batch + 1;

  // Tensor Q/K/V/Y
  auto gQ = tma_q.get_tma_tensor(make_shape(num_head_q, head_dim, total_seq_q));
  auto gK = tma_k.get_tma_tensor(make_shape(kBlockSize, head_dim, num_kvcache_blocks));
  auto gYY =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileN>{}, Int<kTileM>{}), make_stride(Int<kTileM>{}, Int<1>{}));

  // Tensor sQ/sK/sV
  auto sQ = make_tensor(make_smem_ptr(shm_q), SLayoutQ{});
  auto sK = make_tensor(make_smem_ptr(shm_k), SLayoutK{});

  // init k/v barrier
  if (is_leader_in_block) {
#pragma unroll
    for (int i = 0; i < kStageQ; ++i) {
      initialize_barrier(readable_q[i], 1);
      initialize_barrier(writable_q[i], kWarpGroupN);
    }
#pragma unroll
    for (int i = 0; i < kStageK; ++i) {
      initialize_barrier(readable_k[i], 1);
      initialize_barrier(writable_k[i], 1);
    }
  }

  for (int i = idx; i < num_batch + 1; i += blockDim.x) {
    shm_cu_seqlenq[i] = cu_seqlens_q_ptr[i];
  }

  for (int i = idx; i < num_batch; i += blockDim.x) {
    shm_seqlenkv[i] = seqlens_kv_ptr[i];
  }

  // sync to avoid ahead thread use(wait) readable when it is not initizlized yet
  __syncthreads();

  if (idx >= kWarpGroupN * 128) {
    cutlass::arch::warpgroup_reg_dealloc<24>();
    idx -= kWarpGroupN * 128;

    // Block Level tma
    auto btma_q = tma_q.get_slice(0);
    auto btma_k = tma_k.get_slice(0);

    // Thread Level Tensor
    auto tQg = btma_q.partition_S(gQ);  // (TMA, TMA_M, TMA_K, head, batch)
    auto tKg = btma_k.partition_S(gK);  // (TMA, TMA_N, TMA_K, head, batch)

    auto tQs = btma_q.partition_D(sQ);  // (TMA, _1, _1)
    auto tKs = btma_k.partition_D(sK);  // (TMA, _1, _1, kStage)

    int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
    int is_leader_in_load = ((iwarp == 0) && elected);

    constexpr int kNumBlockPerTileN = kTileN / kBlockSize;

    if (is_leader_in_load) {
      int phase_k = 1;  // start with ok
      int phase_q = 1;  // start with ok
      int ismemk_write = __shfl_sync(0xFFFFFFFF, 0, 0);
      int ismemq_write = __shfl_sync(0xFFFFFFFF, 0, 0);
      int start_batch = 0;

      while (true) {
        if (iblock >= total_seq_q * num_split_divider.divisor) {
          break;
        }
        auto [itoken, ichunk, ibatch, itoken_in_batch] =
            get_next_tile(shm_cu_seqlenq, iblock, start_batch, num_batch, num_split_divider);

        start_batch = ibatch;
        iblock += gridDim.x;

        int num_seq_q = shm_cu_seqlenq[ibatch + 1] - shm_cu_seqlenq[ibatch];
        int num_seq_kv = shm_seqlenkv[ibatch];
        int num_seq_kv_ratio = ((num_seq_kv - num_seq_q) + itoken_in_batch + 1) / kRatio;
        auto *block_ids_ibatch_ptr = block_ids_ptr + ibatch * num_seq_max_blocks;

        int num_blocks = (num_seq_kv_ratio + kBlockSize - 1) / kBlockSize;
        int num_tile_kv = (num_seq_kv_ratio + kTileN - 1) / kTileN;

        int num_tiles_per_chunk;
        int num_tiles_last_chunk;
        num_split_divider(num_tiles_per_chunk, num_tiles_last_chunk, num_tile_kv);

        int tile_kv_begin = ichunk * num_tiles_per_chunk;
        int tile_kv_end = (ichunk == num_split_divider.divisor - 1)
                              ? num_tile_kv
                              : (tile_kv_begin + num_tiles_per_chunk);

        constexpr int kTransactionBytesQ =
            sizeof(Tin) * kTileM * kTileK + sizeof(cute::bfloat16_t) * kTileM;
        constexpr int kTransactionBytesK = sizeof(Tin) * kTileN * kTileK;

        if (tile_kv_begin == tile_kv_end) {
          continue;
        }

        // Load Q
        wait_barrier(writable_q[ismemq_write], phase_q);
        cute::copy(tma_q.with(readable_q[ismemq_write]), tQg(_, 0, _, itoken),
                   tQs(_, 0, _, ismemq_write));
        cp_async_g2s(shm_w + ismemq_write * kTileM, w_ptr + itoken * num_head_q,
                     kTileM * sizeof(cute::bfloat16_t), &readable_q[ismemq_write]);
        set_barrier_transaction_bytes(readable_q[ismemq_write], kTransactionBytesQ);

        ++ismemq_write;
        if (ismemq_write == kStageQ) {
          ismemq_write = 0;
          phase_q ^= 1;
        }

        // load KV
#pragma unroll 1
        for (int itile_seq_kv = tile_kv_begin; itile_seq_kv < tile_kv_end; ++itile_seq_kv) {
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

          //
          wait_barrier(writable_k[ismemk_write], phase_k);
#pragma unroll
          for (int iblock_kv = 0; iblock_kv < kNumBlockPerTileN; iblock_kv++) {
            int iblock_true = iblock_ids[iblock_kv];
            cute::copy(tma_k.with(readable_k[ismemk_write]), tKg(_, 0, _, iblock_true),
                       tKs(_, iblock_kv, _, ismemk_write));
          }
          set_barrier_transaction_bytes(readable_k[ismemk_write], kTransactionBytesK);

          ++ismemk_write;
          if (ismemk_write == kStageK) {
            ismemk_write = 0;
            phase_k ^= 1;
          }
        }
      }
    }
  } else {
    cutlass::arch::warpgroup_reg_alloc<168>();

    int iwarpgroup = idx / 128;
    int idx_in_warpgroup = idx % 128;
    int ilane = idx % 32;
    int iwarp_in_warpgroup = idx_in_warpgroup / 32;
    int elected_idx_in_warpgroup = ((iwarp_in_warpgroup == 0) && elected);

    TiledMma tiled_mma;

    auto thr_mma = tiled_mma.get_slice(idx_in_warpgroup);

    auto tKs4r = thr_mma.partition_A(sK);
    auto tQs4r = thr_mma.partition_B(sQ);

    auto tKr = thr_mma.make_fragment_A(tKs4r);  // (MMA, MMA_M, MMA_K)
    auto tQr = thr_mma.make_fragment_B(tQs4r);  // (MMA, MMA_N, MMA_K)

    auto tYr = thr_mma.partition_fragment_C(gYY);

    auto tYr_nm = retile_fragment(tYr);
    auto gI = make_identity_tensor(gYY.shape());
    auto tI = thr_mma.partition_C(gI);
    auto tI_nm = retile_fragment(tI);

    constexpr int kN = size<0>(tYr_nm);
    constexpr int kM = size<1>(tYr_nm);

    Tensor gSum = make_tensor<float>(Int<kN>{});
    Tensor Ws = make_tensor<float>(Int<kM>{});

    int ismemq_read = 0;
    int ismemk_read = iwarpgroup;
    int phase_q = 0;
    int phase_k = 0;

    int start_batch = 0;
    int start_warpgroup = 0;
    while (true) {
      if (iblock >= total_seq_q * num_split_divider.divisor) {
        break;
      }

      auto [itoken, ichunk, ibatch, itoken_in_batch] =
          get_next_tile(shm_cu_seqlenq, iblock, start_batch, num_batch, num_split_divider);

      start_batch = ibatch;
      iblock += gridDim.x;

      auto *output =
          output_ptr + static_cast<uint64_t>(itoken) * static_cast<uint64_t>(max_context_len);
      int num_seq_q = shm_cu_seqlenq[ibatch + 1] - shm_cu_seqlenq[ibatch];
      int num_seq_kv = shm_seqlenkv[ibatch];
      int num_seq_kvcache = num_seq_kv - num_seq_q;
      int num_seq_kv_ratio = (num_seq_kvcache + itoken_in_batch + 1) / kRatio;

      int num_tile_kv = (num_seq_kv_ratio + kTileN - 1) / kTileN;
      int num_tiles_per_chunk;
      int num_tiles_last_chunk;
      num_split_divider(num_tiles_per_chunk, num_tiles_last_chunk, num_tile_kv);

      int tile_kv_begin = ichunk * num_tiles_per_chunk;
      int tile_kv_end = (ichunk == num_split_divider.divisor - 1)
                            ? num_tile_kv
                            : (tile_kv_begin + num_tiles_per_chunk);

      if (tile_kv_begin == tile_kv_end) {
        continue;
      }

      int num_tile_kv_local = tile_kv_end - tile_kv_begin;
      tile_kv_begin += (start_warpgroup + iwarpgroup) % kWarpGroupN;

      wait_barrier(readable_q[ismemq_read], phase_q);
#pragma unroll
      for (int im = 0; im < kM; ++im) {
        int ihead = get<1>(tI_nm(0, im));
        Ws(im) = static_cast<float>(shm_w[ismemq_read * kTileM + ihead]);
      }
#pragma unroll 1
      for (int itile_seq_kv = tile_kv_begin; itile_seq_kv < tile_kv_end;
           itile_seq_kv += kWarpGroupN) {
        wait_barrier(readable_k[ismemk_read], phase_k);
        // P = QK
        tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;

        warpgroup_fence_operand(tYr);
        warpgroup_arrive();
#pragma unroll
        for (int ik = 0; ik < size<2>(tQr); ++ik) {
          cute::gemm(tiled_mma, tKr(_, _, ik, ismemk_read), tQr(_, _, ik, ismemq_read),
                     tYr(_, _, _));
          tiled_mma.accumulate_ = GMMA::ScaleOut::One;
        }

        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(tYr);

        if (elected_idx_in_warpgroup) {
          arrive_barrier(writable_k[ismemk_read]);
        }

        ismemk_read += kWarpGroupN;
        if (ismemk_read >= kStageK) {
          phase_k ^= 1;
          ismemk_read %= kStageK;
        }

        clear(gSum);

#pragma unroll
        for (int in = 0; in < kN; ++in) {
#pragma unroll
          for (int im = 0; im < kM; ++im) {
            gSum(in) += relu(tYr_nm(in, im)) * Ws(im);
          }
          gSum(in) = warp_4lane_reduce_sum_xor(gSum(in));
        }

        // store output
        if (ilane % 4 == 0) {
#pragma unroll
          for (int in = 0; in < kN; ++in) {
            int iseqk = itile_seq_kv * kTileN + get<0>(tI_nm(in, 0));
            if (iseqk < num_seq_kv_ratio) {
              store(output + iseqk, gSum(in));
            }
          }
        }
      }

      start_warpgroup = (start_warpgroup + num_tile_kv_local) % kWarpGroupN;
      if (elected_idx_in_warpgroup) {
        arrive_barrier(writable_q[ismemq_read]);
      }
      ++ismemq_read;
      if (ismemq_read == kStageQ) {
        ismemq_read = 0;
        phase_q ^= 1;
      }
    }
  }
}

}  // namespace kernels
}  // namespace indexer
}  // namespace hpc

#endif  // SRC_INDEXER_KERNELS_CUH_
