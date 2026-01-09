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

template <typename Tin, typename Tout, int kTileM, int kTileN, int kTileK, int kBlockSize,
          int kStageK, int kRatio, typename TiledMma, typename TmaQ, typename TmaK,
          typename SLayoutQ, typename SLayoutK>
__global__ void __launch_bounds__(384, 1)
    mqa_indexer_logits_bf16_kernel(const __grid_constant__ TmaQ tma_q,
                                   const __grid_constant__ TmaK tma_k, const Tin *w_ptr,
                                   Tout *output_ptr, const int *cu_seqlens_q_ptr,
                                   const int *seqlens_kv_ptr, const int *block_ids_ptr,
                                   int num_batch, int total_seq_q, int num_head_q, int head_dim,
                                   int num_kvcache_blocks, int num_seq_max_blocks,
                                   int max_context_len, cutlass::FastDivmod num_split_divider) {
  using namespace cute;  // NOLINT

  int idx = threadIdx.x;
  int iblock = blockIdx.x;

  int elected = cute::elect_one_sync();
  int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
  bool is_leader_in_block = (iwarp == 0) && elected;

  __shared__ uint64_t readable_q;
  __shared__ uint64_t writable_q;

  __shared__ uint64_t readable_k[kStageK];
  __shared__ uint64_t writable_k[kStageK];
  extern __shared__ uint8_t shm_data[] alignas(128);

  auto *shm_q = reinterpret_cast<Tin *>(shm_data);
  auto *shm_k = shm_q + cosize(SLayoutQ{});
  auto *shm_w = shm_k + cosize(SLayoutK{});

  // Tensor Q/K/V/Y
  auto gQ = tma_q.get_tma_tensor(make_shape(num_head_q, head_dim, total_seq_q));
  auto gK = tma_k.get_tma_tensor(make_shape(kBlockSize, head_dim, num_kvcache_blocks));
  auto gYY =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileN>{}), make_stride(Int<kTileN>{}, Int<1>{}));

  // Tensor sQ/sK/sV
  auto sQ = make_tensor(make_smem_ptr(shm_q), SLayoutQ{});
  auto sK = make_tensor(make_smem_ptr(shm_k), SLayoutK{});

  // Block Level tma
  auto btma_q = tma_q.get_slice(0);
  auto btma_k = tma_k.get_slice(0);

  // Thread Level Tensor
  auto tQg = btma_q.partition_S(gQ);  // (TMA, TMA_M, TMA_K, head, batch)
  auto tKg = btma_k.partition_S(gK);  // (TMA, TMA_N, TMA_K, head, batch)

  auto tQs = btma_q.partition_D(sQ);  // (TMA, _1, _1)
  auto tKs = btma_k.partition_D(sK);  // (TMA, _1, _1, kStage)

  TiledMma tiled_mma;

  auto thr_mma = tiled_mma.get_slice(idx);

  auto tKs4r = thr_mma.partition_A(sK);
  auto tQs4r = thr_mma.partition_B(sQ);

  auto tKr = thr_mma.make_fragment_A(tKs4r);  // (MMA, MMA_M, MMA_K)
  auto tQr = thr_mma.make_fragment_B(tQs4r);  // (MMA, MMA_N, MMA_K)

  auto tYr = thr_mma.partition_fragment_C(gYY);

  // init k/v barrier
  if (is_leader_in_block) {
    initialize_barrier(readable_q, 1);
    initialize_barrier(writable_q, 2);
#pragma unroll
    for (int i = 0; i < kStageK; ++i) {
      initialize_barrier(readable_k[i], 1);
      initialize_barrier(writable_k[i], 2);
    }
  }

  // sync to avoid ahead thread use(wait) readable when it is not initizlized yet
  __syncthreads();

  constexpr int kNumBlockPerTileM = kTileM / kBlockSize;

  if (idx >= 256) {
    cutlass::arch::warpgroup_reg_dealloc<24>();
    idx -= 256;

    int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
    int is_leader_in_load = ((iwarp == 0) && elected);

    if (is_leader_in_load) {
      int phase = 1;    // start with ok
      int phase_q = 1;  // start with ok
      int ismem_write = __shfl_sync(0xFFFFFFFF, 0, 0);
      int start_batch = 0;

      while (true) {
        if (iblock >= total_seq_q * num_split_divider.divisor) {
          break;
        }
        auto [itoken, ichunk, ibatch, itoken_in_batch] =
            get_next_tile(cu_seqlens_q_ptr, iblock, start_batch, num_batch, num_split_divider);

        start_batch = ibatch;
        iblock += gridDim.x;

        int num_seq_q = cu_seqlens_q_ptr[ibatch + 1] - cu_seqlens_q_ptr[ibatch];
        int num_seq_kv = seqlens_kv_ptr[ibatch];
        int num_seq_kv_ratio = ((num_seq_kv - num_seq_q) + itoken_in_batch + 1) / kRatio;
        auto *block_ids_ibatch_ptr = block_ids_ptr + ibatch * num_seq_max_blocks;

        int num_blocks = (num_seq_kv_ratio + kBlockSize - 1) / kBlockSize;
        int num_tile_kv = (num_seq_kv_ratio + kTileM - 1) / kTileM;

        int num_tiles_per_chunk;
        int num_tiles_last_chunk;
        num_split_divider(num_tiles_per_chunk, num_tiles_last_chunk, num_tile_kv);

        int tile_kv_begin = ichunk * num_tiles_per_chunk;
        int tile_kv_end = (ichunk == num_split_divider.divisor - 1)
                              ? num_tile_kv
                              : (tile_kv_begin + num_tiles_per_chunk);

        constexpr int kTransactionBytesK = sizeof(Tin) * kTileM * kTileK;

        if (tile_kv_begin == tile_kv_end) {
          continue;
        }

        // Load Q
        wait_barrier(writable_q, phase_q);
        cute::copy(tma_q.with(readable_q), tQg(_, 0, _, itoken), tQs(_, 0, _));
        cp_async_g2s(shm_w, w_ptr + itoken * num_head_q, kTileN * sizeof(Tin), &readable_q);
        set_barrier_transaction_bytes(readable_q, sizeof(Tin) * (cosize(SLayoutQ{}) + kTileN));
        phase_q ^= 1;

        // load KV
#pragma unroll 1
        for (int itile_seq_kv = tile_kv_begin; itile_seq_kv < tile_kv_end; ++itile_seq_kv) {
          // k
          wait_barrier(writable_k[ismem_write], phase);

          int iblock_ids[kNumBlockPerTileM];
#pragma unroll
          for (int iblock_kv = 0; iblock_kv < kNumBlockPerTileM; iblock_kv++) {
            iblock_ids[iblock_kv] = 0;
            int iblock_id = itile_seq_kv * kNumBlockPerTileM + iblock_kv;
            if (iblock_id < num_blocks) {
              iblock_ids[iblock_kv] = block_ids_ibatch_ptr[iblock_id];
            } else {
              iblock_ids[iblock_kv] = -1;
            }
          }

#pragma unroll
          for (int iblock_kv = 0; iblock_kv < kNumBlockPerTileM; iblock_kv++) {
            int iblock_true = iblock_ids[iblock_kv];
            cute::copy(tma_k.with(readable_k[ismem_write]), tKg(_, 0, _, iblock_true),
                       tKs(_, iblock_kv, _, ismem_write));
          }
          set_barrier_transaction_bytes(readable_k[ismem_write], kTransactionBytesK);

          ++ismem_write;
          if (ismem_write == kStageK) {
            ismem_write = 0;
            phase ^= 1;
          }
        }
      }
    }
  } else {
    cutlass::arch::warpgroup_reg_alloc<168>();

    int idx_in_warpgroup = idx % 128;
    int ilane = idx % 32;
    int iwarp_in_warpgroup = idx_in_warpgroup / 32;
    int elected_idx_in_warpgroup = ((iwarp_in_warpgroup == 0) && elected);

    auto tYr_mn = retile_fragment(tYr);
    auto gI = make_identity_tensor(gYY.shape());
    auto tI = thr_mma.partition_C(gI);
    auto tI_mn = retile_fragment(tI);

    constexpr int kM = size<0>(tYr_mn);
    constexpr int kN = size<1>(tYr_mn);

    Tensor gSum = make_tensor<float>(Int<kM>{});
    Tensor Ws = make_tensor<float>(Int<kN>{});

    int ismem_read = 0;
    int phase = 0;
    int phase_q = 0;

    int start_batch = 0;
    while (true) {
      if (iblock >= total_seq_q * num_split_divider.divisor) {
        break;
      }

      auto [itoken, ichunk, ibatch, itoken_in_batch] =
          get_next_tile(cu_seqlens_q_ptr, iblock, start_batch, num_batch, num_split_divider);

      start_batch = ibatch;
      iblock += gridDim.x;

      auto *output_warp = output_ptr +
                          static_cast<uint64_t>(itoken) * static_cast<uint64_t>(max_context_len) +
                          iwarp * 16;
      int num_seq_q = cu_seqlens_q_ptr[ibatch + 1] - cu_seqlens_q_ptr[ibatch];
      int num_seq_kv = seqlens_kv_ptr[ibatch];
      int num_seq_kvcache = num_seq_kv - num_seq_q;
      int num_seq_kv_ratio = (num_seq_kvcache + itoken_in_batch + 1) / kRatio;

      int num_tile_kv = (num_seq_kv_ratio + kTileM - 1) / kTileM;
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

      clear(tYr);
      clear(gSum);

      wait_barrier(readable_q, phase_q);
#pragma unroll
      for (int in = 0; in < kN; ++in) {
        int icol = get<1>(tI_mn(0, in));
        Ws(in) = static_cast<float>(shm_w[icol]);
      }
#pragma unroll 1
      for (int itile_seq_kv = tile_kv_begin; itile_seq_kv < tile_kv_end; ++itile_seq_kv) {
        wait_barrier(readable_k[ismem_read], phase);
        // P = QK
        tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;

        warpgroup_fence_operand(tYr);
        warpgroup_arrive();
#pragma unroll
        for (int ik = 0; ik < size<2>(tQr); ++ik) {
          cute::gemm(tiled_mma, tKr(_, _, ik, ismem_read), tQr(_, _, ik), tYr(_, _, _));
          tiled_mma.accumulate_ = GMMA::ScaleOut::One;
        }

        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(tYr);

        if (elected_idx_in_warpgroup) {
          arrive_barrier(writable_k[ismem_read]);
        }

        if (itile_seq_kv == (tile_kv_end - 1)) {
          if (elected_idx_in_warpgroup) {
            arrive_barrier(writable_q);
          }
          phase_q ^= 1;
        }

        ++ismem_read;
        if (ismem_read == kStageK) {
          phase ^= 1;
          ismem_read = 0;
        }
        clear(gSum);

#pragma unroll
        for (int im = 0; im < kM; ++im) {
          int iseqk = (itile_seq_kv * kTileM + get<0>(tI_mn(im, 0)) + 1) * kRatio - 1;

          if (itoken_in_batch + num_seq_kvcache >= iseqk) {
#pragma unroll
            for (int in = 0; in < kN; ++in) {
              int icol = get<1>(tI_mn(im, in));
              gSum(im) += relu(tYr_mn(im, in)) * Ws(in);
            }
          } else {
            gSum(im) = -std::numeric_limits<float>::infinity();
          }
          gSum(im) = warp_4lane_reduce_sum_xor(gSum(im));
        }

        // store output
        if (ilane % 4 == 0) {
          auto *output = output_warp + itile_seq_kv * kTileM + ilane / 4;
#pragma unroll
          for (int im = 0; im < kM; ++im) {
            store(output + im * 8, gSum(im));
          }
        }
      }
    }
  }
}

template <typename Tin, typename Tout, int kTileM, int kTileN, int kTileK, int kBlockSize,
          int kStageK, int kRatio, typename TiledMma, typename TmaQ, typename TmaK,
          typename SLayoutQ, typename SLayoutK>
__global__ void __launch_bounds__(384, 1)
    mqa_indexer_logits_fp8_kernel(const __grid_constant__ TmaQ tma_q,
                                  const __grid_constant__ TmaK tma_k, const cute::bfloat16_t *w_ptr,
                                  Tout *output_ptr, const int *cu_seqlens_q_ptr,
                                  const int *seqlens_kv_ptr, const int *block_ids_ptr,
                                  int num_batch, int total_seq_q, int num_head_q, int head_dim,
                                  int num_kvcache_blocks, int num_seq_max_blocks,
                                  int max_context_len, cutlass::FastDivmod num_split_divider) {
  using namespace cute;  // NOLINT

  int idx = threadIdx.x;
  int iblock = blockIdx.x;

  int elected = cute::elect_one_sync();
  int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
  bool is_leader_in_block = (iwarp == 0) && elected;

  __shared__ uint64_t readable_q;
  __shared__ uint64_t writable_q;

  __shared__ uint64_t readable_k[kStageK];
  __shared__ uint64_t writable_k[kStageK];
  extern __shared__ uint8_t shm_data[] alignas(128);

  auto *shm_q = reinterpret_cast<Tin *>(shm_data);
  auto *shm_k = shm_q + cosize(SLayoutQ{});
  auto *shm_w = reinterpret_cast<cute::bfloat16_t *>(shm_k + cosize(SLayoutK{}));

  // Tensor Q/K/V/Y
  auto gQ = tma_q.get_tma_tensor(make_shape(num_head_q, head_dim, total_seq_q));
  auto gK = tma_k.get_tma_tensor(make_shape(kBlockSize, head_dim, num_kvcache_blocks));
  auto gYY =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileN>{}), make_stride(Int<kTileN>{}, Int<1>{}));

  // Tensor sQ/sK/sV
  auto sQ = make_tensor(make_smem_ptr(shm_q), SLayoutQ{});
  auto sK = make_tensor(make_smem_ptr(shm_k), SLayoutK{});

  // Block Level tma
  auto btma_q = tma_q.get_slice(0);
  auto btma_k = tma_k.get_slice(0);

  // Thread Level Tensor
  auto tQg = btma_q.partition_S(gQ);  // (TMA, TMA_M, TMA_K, head, batch)
  auto tKg = btma_k.partition_S(gK);  // (TMA, TMA_N, TMA_K, head, batch)

  auto tQs = btma_q.partition_D(sQ);  // (TMA, _1, _1)
  auto tKs = btma_k.partition_D(sK);  // (TMA, _1, _1, kStage)

  TiledMma tiled_mma;

  auto thr_mma = tiled_mma.get_slice(idx);

  auto tKs4r = thr_mma.partition_A(sK);
  auto tQs4r = thr_mma.partition_B(sQ);

  auto tKr = thr_mma.make_fragment_A(tKs4r);  // (MMA, MMA_M, MMA_K)
  auto tQr = thr_mma.make_fragment_B(tQs4r);  // (MMA, MMA_N, MMA_K)

  auto tYr = thr_mma.partition_fragment_C(gYY);

  // init k/v barrier
  if (is_leader_in_block) {
    initialize_barrier(readable_q, 1);
    initialize_barrier(writable_q, 2);
#pragma unroll
    for (int i = 0; i < kStageK; ++i) {
      initialize_barrier(readable_k[i], 1);
      initialize_barrier(writable_k[i], 2);
    }
  }

  // sync to avoid ahead thread use(wait) readable when it is not initizlized yet
  __syncthreads();

  constexpr int kNumBlockPerTileM = kTileM / kBlockSize;

  if (idx >= 256) {
    cutlass::arch::warpgroup_reg_dealloc<24>();
    idx -= 256;

    int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
    int is_leader_in_load = ((iwarp == 0) && elected);

    if (is_leader_in_load) {
      int phase = 1;    // start with ok
      int phase_q = 1;  // start with ok
      int ismem_write = __shfl_sync(0xFFFFFFFF, 0, 0);
      int start_batch = 0;

      while (true) {
        if (iblock >= total_seq_q * num_split_divider.divisor) {
          break;
        }
        auto [itoken, ichunk, ibatch, itoken_in_batch] =
            get_next_tile(cu_seqlens_q_ptr, iblock, start_batch, num_batch, num_split_divider);

        start_batch = ibatch;
        iblock += gridDim.x;

        int num_seq_q = cu_seqlens_q_ptr[ibatch + 1] - cu_seqlens_q_ptr[ibatch];
        int num_seq_kv = seqlens_kv_ptr[ibatch];
        int num_seq_kv_ratio = ((num_seq_kv - num_seq_q) + itoken_in_batch + 1) / kRatio;
        auto *block_ids_ibatch_ptr = block_ids_ptr + ibatch * num_seq_max_blocks;

        int num_blocks = (num_seq_kv_ratio + kBlockSize - 1) / kBlockSize;
        int num_tile_kv = (num_seq_kv_ratio + kTileM - 1) / kTileM;

        int num_tiles_per_chunk;
        int num_tiles_last_chunk;
        num_split_divider(num_tiles_per_chunk, num_tiles_last_chunk, num_tile_kv);

        int tile_kv_begin = ichunk * num_tiles_per_chunk;
        int tile_kv_end = (ichunk == num_split_divider.divisor - 1)
                              ? num_tile_kv
                              : (tile_kv_begin + num_tiles_per_chunk);
        constexpr int kTransactionBytesK = sizeof(Tin) * kTileM * kTileK;

        if (tile_kv_begin == tile_kv_end) {
          continue;
        }

        // Load Q
        wait_barrier(writable_q, phase_q);
        cute::copy(tma_q.with(readable_q), tQg(_, 0, _, itoken), tQs(_, 0, _));
        cp_async_g2s(shm_w, w_ptr + itoken * num_head_q, kTileN * sizeof(cute::bfloat16_t),
                     &readable_q);
        set_barrier_transaction_bytes(
            readable_q, sizeof(Tin) * (cosize(SLayoutQ{})) + sizeof(cute::bfloat16_t) * kTileN);
        phase_q ^= 1;

        // load KV
#pragma unroll 1
        for (int itile_seq_kv = tile_kv_begin; itile_seq_kv < tile_kv_end; ++itile_seq_kv) {
          // k
          wait_barrier(writable_k[ismem_write], phase);

          int iblock_ids[kNumBlockPerTileM];
#pragma unroll
          for (int iblock_kv = 0; iblock_kv < kNumBlockPerTileM; iblock_kv++) {
            iblock_ids[iblock_kv] = 0;
            int iblock_id = itile_seq_kv * kNumBlockPerTileM + iblock_kv;
            if (iblock_id < num_blocks) {
              iblock_ids[iblock_kv] = block_ids_ibatch_ptr[iblock_id];
            } else {
              iblock_ids[iblock_kv] = -1;
            }
          }

#pragma unroll
          for (int iblock_kv = 0; iblock_kv < kNumBlockPerTileM; iblock_kv++) {
            int iblock_true = iblock_ids[iblock_kv];
            cute::copy(tma_k.with(readable_k[ismem_write]), tKg(_, 0, _, iblock_true),
                       tKs(_, iblock_kv, _, ismem_write));
          }
          set_barrier_transaction_bytes(readable_k[ismem_write], kTransactionBytesK);

          ++ismem_write;
          if (ismem_write == kStageK) {
            ismem_write = 0;
            phase ^= 1;
          }
        }
      }
    }
  } else {
    cutlass::arch::warpgroup_reg_alloc<168>();

    int idx_in_warpgroup = idx % 128;
    int ilane = idx % 32;
    int iwarp_in_warpgroup = idx_in_warpgroup / 32;
    int elected_idx_in_warpgroup = ((iwarp_in_warpgroup == 0) && elected);

    auto tYr_mn = retile_fragment(tYr);
    auto gI = make_identity_tensor(gYY.shape());
    auto tI = thr_mma.partition_C(gI);
    auto tI_mn = retile_fragment(tI);

    constexpr int kM = size<0>(tYr_mn);
    constexpr int kN = size<1>(tYr_mn);

    Tensor gSum = make_tensor<float>(Int<kM>{});
    Tensor Ws = make_tensor<float>(Int<kN>{});

    int ismem_read = 0;
    int phase = 0;
    int phase_q = 0;

    int start_batch = 0;
    while (true) {
      if (iblock >= total_seq_q * num_split_divider.divisor) {
        break;
      }

      auto [itoken, ichunk, ibatch, itoken_in_batch] =
          get_next_tile(cu_seqlens_q_ptr, iblock, start_batch, num_batch, num_split_divider);

      start_batch = ibatch;
      iblock += gridDim.x;

      auto *output_warp = output_ptr +
                          static_cast<uint64_t>(itoken) * static_cast<uint64_t>(max_context_len) +
                          iwarp * 16;
      int num_seq_q = cu_seqlens_q_ptr[ibatch + 1] - cu_seqlens_q_ptr[ibatch];
      int num_seq_kv = seqlens_kv_ptr[ibatch];
      int num_seq_kvcache = num_seq_kv - num_seq_q;
      int num_seq_kv_ratio = (num_seq_kvcache + itoken_in_batch + 1) / kRatio;

      int num_tile_kv = (num_seq_kv_ratio + kTileM - 1) / kTileM;
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

      clear(tYr);
      clear(gSum);

      wait_barrier(readable_q, phase_q);
#pragma unroll
      for (int in = 0; in < kN; ++in) {
        int icol = get<1>(tI_mn(0, in));
        Ws(in) = static_cast<float>(shm_w[icol]);
      }
#pragma unroll 1
      for (int itile_seq_kv = tile_kv_begin; itile_seq_kv < tile_kv_end; ++itile_seq_kv) {
        wait_barrier(readable_k[ismem_read], phase);
        // P = QK
        tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;

        warpgroup_fence_operand(tYr);
        warpgroup_arrive();
#pragma unroll
        for (int ik = 0; ik < size<2>(tQr); ++ik) {
          cute::gemm(tiled_mma, tKr(_, _, ik, ismem_read), tQr(_, _, ik), tYr(_, _, _));
          tiled_mma.accumulate_ = GMMA::ScaleOut::One;
        }

        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(tYr);

        if (elected_idx_in_warpgroup) {
          arrive_barrier(writable_k[ismem_read]);
        }

        if (itile_seq_kv == (tile_kv_end - 1)) {
          if (elected_idx_in_warpgroup) {
            arrive_barrier(writable_q);
          }
          phase_q ^= 1;
        }

        ++ismem_read;
        if (ismem_read == kStageK) {
          phase ^= 1;
          ismem_read = 0;
        }

        clear(gSum);

#pragma unroll
        for (int im = 0; im < kM; ++im) {
          int iseqk = (itile_seq_kv * kTileM + get<0>(tI_mn(im, 0)) + 1) * kRatio - 1;

          if (itoken_in_batch + num_seq_kvcache >= iseqk) {
#pragma unroll
            for (int in = 0; in < kN; ++in) {
              int icol = get<1>(tI_mn(im, in));
              gSum(im) += relu(tYr_mn(im, in)) * Ws(in);
            }
          } else {
            gSum(im) = -std::numeric_limits<float>::infinity();
          }
          gSum(im) = warp_4lane_reduce_sum_xor(gSum(im));
        }

        // store output
        if (ilane % 4 == 0) {
          auto *output = output_warp + itile_seq_kv * kTileM + ilane / 4;
#pragma unroll
          for (int im = 0; im < kM; ++im) {
            store(output + im * 8, gSum(im));
          }
        }
      }
    }
  }
}

}  // namespace kernels
}  // namespace indexer
}  // namespace hpc

#endif  // SRC_INDEXER_KERNELS_CUH_
