// Copyright 2025 hpc-ops authors
#include <cuda.h>
#include <stdio.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "src/mamba/selective_state_scan.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace mamba {
namespace kernels {

template <typename Tout, typename Tin, int kChunkSize, int kTmaDescCount, int kProbK, int kTileM,
          int kTileN, int kTileK, int kStage, int kHeadsPerBlock, typename TiledMma, typename TmaA,
          typename TmaB, typename TmaC, typename SMemLayoutA, typename SMemLayoutB,
          typename SMemLayoutC>
__global__ void pre_y_bmm(__grid_constant__ const TmaB tma_b, __grid_constant__ const TmaC tma_c,
                          const cute::TmaDescriptor* tensormaps, const int* split_metadata,
                          int batch_size, uint32_t m, uint32_t n, uint32_t k, uint32_t nheads,
                          uint32_t ngroups, uint32_t heads_per_group, uint32_t total_chunks) {
  int idx = threadIdx.x;
  int ichunk = blockIdx.z / (nheads / kHeadsPerBlock);
  const int* cu_chunks = split_metadata + 2 * (batch_size + 1);
  const int* seqlens = split_metadata + 3 * (batch_size + 1);

  int ibatch = 0;
  int ichunk_in_batch = 0;
  for (int i = 1; i < batch_size + 1; i++) {
    if (ichunk < cu_chunks[i] - i) {
      ibatch = i - 1;
      ichunk_in_batch = ichunk - (cu_chunks[ibatch] - ibatch);
      break;
    }
  }

  int ihead_block = blockIdx.z % (nheads / kHeadsPerBlock);
  int ihead_start = ihead_block * kHeadsPerBlock;
  int igroup = ihead_start / heads_per_group;
  int itile_m = blockIdx.y;
  int itile_n = blockIdx.x;

  int elected = cute::elect_one_sync();
  int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
  int seqlen = seqlens[ibatch];

  const int im_chunk_load = (ichunk_in_batch + 1) * kChunkSize / kTileM + itile_m;
  if (im_chunk_load * kTileM >= seqlen) {
    return;
  }

  extern __shared__ float4 shm_data[] alignas(128);
  __shared__ uint64_t barriers[kStage];

  auto* shm_a = (Tin*)shm_data;
  auto* shm_b = ((Tin*)shm_a) + cute::cosize(SMemLayoutA{});

  // barrier
  if ((iwarp == 0) && elected) {
#pragma unroll
    for (int istage = 0; istage < kStage; istage++) {
      cute::initialize_barrier(barriers[istage], 1);
    }
  }
  __syncthreads();

  TmaA tma_a;

  auto gA = tma_a.get_tma_tensor(cute::make_shape(seqlen, k, ngroups));
  auto gB = tma_b.get_tma_tensor(
      cute::make_shape(n, k, kHeadsPerBlock, total_chunks, nheads / kHeadsPerBlock));

  auto C = cute::make_tensor(
      cute::make_gmem_ptr((Tout*)nullptr),
      cute::make_shape(m, n, kHeadsPerBlock, total_chunks, nheads / kHeadsPerBlock),
      cute::make_stride(n, cute::Int<1>{}, m * n, total_chunks * m * n,
                        total_chunks * m * n * kHeadsPerBlock));
  auto gC = cute::local_tile(
      C, cute::make_tile(cute::Int<kTileM>{}, cute::Int<kTileN>{}, cute::Int<kHeadsPerBlock>{}),
      cute::make_coord(itile_m, itile_n, 0, ichunk, ihead_block));

  auto sA = cute::make_tensor(cute::make_smem_ptr(shm_a), SMemLayoutA{});
  auto sB = cute::make_tensor(cute::make_smem_ptr(shm_b), SMemLayoutB{});

  auto btma_a = tma_a.get_slice(0);
  auto btma_b = tma_b.get_slice(0);

  auto tAg = btma_a.partition_S(gA);  // (TMA, TMA_M, TMA_K, batch)
  auto tAs = btma_a.partition_D(sA);  // (TMA, _1, _1)

  auto tBg = btma_b.partition_S(gB);  // (TMA, TMA_M, TMA_K, batch)
  auto tBs = btma_b.partition_D(sB);  // (TMA, _1, _1)

  TiledMma tiled_mma;
  constexpr int nwarps = cute::size(tiled_mma) / 32;
  auto thr_mma = tiled_mma.get_slice(idx);

  auto tAs4r = thr_mma.partition_A(sA);
  auto tBs4r = thr_mma.partition_B(sB);

  auto tAr = thr_mma.make_fragment_A(tAs4r);  // (MMA, MMA_M, MMA_K)
  auto tBr = thr_mma.make_fragment_B(tBs4r);  // (MMA, MMA_N, MMA_K)

  auto tCr = thr_mma.partition_fragment_C(gC);
  tiled_mma.accumulate_ = cute::GMMA::ScaleOut::Zero;

  auto* tdA = &tensormaps[kTmaDescCount * ibatch + 3];
  constexpr int kLoadsPerChunk = kProbK / kTileK;

#pragma unroll
  for (int istage = 0; istage < kStage - 1; istage++) {
    if ((iwarp == 0) && elected) {
      cute::set_barrier_transaction_bytes(
          barriers[istage], sizeof(Tin) * (kTileM + kTileN * kHeadsPerBlock) * (kTileK));
      cute::copy(tma_a.with(tdA, barriers[istage]), tAg(cute::_, im_chunk_load, istage, igroup),
                 tAs(cute::_, 0, 0, istage));
#pragma unroll
      for (int ihead = 0; ihead < kHeadsPerBlock; ihead++) {
        cute::copy(tma_b.with(barriers[istage]),
                   tBg(cute::_, itile_n, istage, ihead, ichunk, ihead_block),
                   tBs(cute::_, 0, 0, istage, ihead));
      }
    }
  }
#pragma unroll
  for (int itile_k = 0; itile_k < kLoadsPerChunk; ++itile_k) {
    // 1. load next stage a/b
    if ((iwarp == 0) && elected && itile_k < kProbK / kTileK - kStage + 1) {
      int iload = (itile_k + kStage - 1) % kStage;
      int iload_phase = itile_k == 0 ? 0 : (itile_k - 1) / kStage + 1;
      cute::set_barrier_transaction_bytes(
          barriers[iload], sizeof(Tin) * (kTileM + kTileN * kHeadsPerBlock) * (kTileK));
      cute::copy(tma_a.with(tdA, barriers[iload]),
                 tAg(cute::_, im_chunk_load, iload_phase * kStage + iload, igroup),
                 tAs(cute::_, 0, 0, iload));
#pragma unroll
      for (int ihead = 0; ihead < kHeadsPerBlock; ihead++) {
        cute::copy(tma_b.with(barriers[iload]),
                   tBg(cute::_, itile_n, iload_phase * kStage + iload, ihead, ichunk, ihead_block),
                   tBs(cute::_, 0, 0, iload, ihead));
      }
    }
    // 2. wait this itile_k ready
    int iwgmma = itile_k % kStage;
    int iphase = (itile_k / kStage) % 2;
    __syncthreads();
    cute::wait_barrier(barriers[iwgmma], iphase);

    // 3. wgmma
    cute::warpgroup_arrive();

#pragma unroll
    for (int ik = 0; ik < cute::size<2>(tAr); ik++) {
      auto tAr_stage = tAr(cute::_, cute::_, ik, iwgmma);
#pragma unroll
      for (int ihead = 0; ihead < kHeadsPerBlock; ihead++) {
        auto tBr_stage = tBr(cute::_, cute::_, ik, iwgmma, ihead);
        cute::gemm(tiled_mma, tAr_stage, tBr_stage, tCr(cute::_, cute::_, cute::_, ihead));
      }
      if (itile_k == 0 && ik == 0) {
        tiled_mma.accumulate_ = cute::GMMA::ScaleOut::One;
      }
    }

    cute::warpgroup_commit_batch();
    cute::warpgroup_wait<0>();
  }

  cute::warpgroup_fence_operand(tCr);
  // Epilogue: write register-C to global memory
  auto sC = cute::make_tensor(cute::make_smem_ptr((Tout*)shm_data), SMemLayoutC{});
  auto tCs4r = thr_mma.partition_C(sC);

  cute::copy(tCr, tCs4r);

  cute::tma_store_fence();
  __syncthreads();

  if ((iwarp == 0) && elected) {
    auto cC = tma_c.get_tma_tensor(
        cute::make_shape(m, n, kHeadsPerBlock, total_chunks, nheads / kHeadsPerBlock));
    auto btma_c = tma_c.get_slice(0);

    auto tCss = btma_c.partition_S(sC);  // (TMA, TMA_M, TMA_N)
    auto tCgg = btma_c.partition_D(cC);  // (TMA, TMA_M, TMA_N, b)

    cute::copy(tma_c, tCss(cute::_, 0, 0, cute::_),
               tCgg(cute::_, itile_m, itile_n, cute::_, ichunk, ihead_block));
  }
}

}  // namespace kernels

bool pre_y_bmm_async(float* out_ptr, const void* zxbcdt_ptr, const void* chunked_states_cumsum,
                     const void* tensormaps, const int* split_metadata, int batch_size,
                     int total_chunks, int nheads, int ngroups, int head_dim, int dstate,
                     int zxbcdt_row_stride, int chunk_size, int tma_desc_count,
                     cudaStream_t stream) {
  using Tin = cute::bfloat16_t;
  using Tout = float;

  constexpr int kProbK = 128;
  constexpr int kTileM = 64;
  constexpr int kTileN = 80;
  constexpr int kTileK = 64;
  constexpr int kStage = 2;
  constexpr int kHeadsPerBlock = 1;
  constexpr int kChunkSize = 256;
  constexpr int kTmaDescCount = 11;

  if (chunk_size != kChunkSize || tma_desc_count != kTmaDescCount) {
    printf("pre_y_bmm_async Only support chunk_size:%d, tma_desc_count:%d\n", kChunkSize,
           kTmaDescCount);
    return false;
  }

  int m = kChunkSize;
  int n = head_dim;
  int k = dstate;

  const Tin* C_ptr =
      reinterpret_cast<const Tin*>(zxbcdt_ptr) + 2 * nheads * head_dim + ngroups * dstate;

  auto A = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<const Tin*>(C_ptr)),
                             cute::make_shape(m, k, ngroups),
                             cute::make_stride(zxbcdt_row_stride, cute::Int<1>{}, dstate));
  auto B = cute::make_tensor(
      cute::make_gmem_ptr(reinterpret_cast<const Tin*>(chunked_states_cumsum)),
      cute::make_shape(n, k, kHeadsPerBlock, total_chunks, nheads / kHeadsPerBlock),
      cute::make_stride(dstate, cute::Int<1>{}, head_dim * dstate * total_chunks, head_dim * dstate,
                        kHeadsPerBlock * total_chunks * head_dim * dstate));
  auto C = cute::make_tensor(
      cute::make_gmem_ptr(out_ptr),
      cute::make_shape(m, n, kHeadsPerBlock, total_chunks, nheads / kHeadsPerBlock),
      cute::make_stride(head_dim, cute::Int<1>{}, m * n * total_chunks, m * n,
                        total_chunks * m * n * kHeadsPerBlock));

  auto sA_layout = cute::tile_to_shape(
      cute::GMMA::Layout_K_SW128_Atom<Tin>{},
      cute::make_shape(cute::Int<kTileM>{}, cute::Int<kTileK>{}, cute::Int<kStage>{}));
  auto sB_layout =
      cute::tile_to_shape(cute::GMMA::Layout_K_SW128_Atom<Tin>{},
                          cute::make_shape(cute::Int<kTileN>{}, cute::Int<kTileK>{},
                                           cute::Int<kStage>{}, cute::Int<kHeadsPerBlock>{}));
  auto sC_layout = cute::tile_to_shape(
      cute::GMMA::Layout_K_SW32_Atom<Tout>{},
      cute::make_shape(cute::Int<kTileM>{}, cute::Int<kTileN>{}, cute::Int<kHeadsPerBlock>{}));

  auto tiled_mma = cute::make_tiled_mma(
      cute::SM90_64x80x16_F32BF16BF16_SS<cute::GMMA::Major::K, cute::GMMA::Major::K>{});

  auto tma_a = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, A, cute::take<0, 2>(sA_layout));
  auto tma_b = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, B, cute::take<0, 2>(sB_layout));
  auto tma_c = cute::make_tma_copy(cute::SM90_TMA_STORE{}, C, sC_layout);

  dim3 block(cute::size(tiled_mma));
  dim3 grid((n + kTileN - 1) / kTileN, (m + kTileM - 1) / kTileM,
            total_chunks * nheads / kHeadsPerBlock);

  int shm_ab = (cute::cosize(sA_layout) + cute::cosize(sB_layout)) * sizeof(Tin);
  int shm_c = cute::cosize(sC_layout) * sizeof(Tout);
  int shm_size = std::max(shm_ab, shm_c);
  auto kernel = kernels::pre_y_bmm<Tout, Tin, kChunkSize, kTmaDescCount, kProbK, kTileM, kTileN,
                                   kTileK, kStage, kHeadsPerBlock, decltype(tiled_mma),
                                   decltype(tma_a), decltype(tma_b), decltype(tma_c),
                                   decltype(sA_layout), decltype(sB_layout), decltype(sC_layout)>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
  kernel<<<grid, block, shm_size, stream>>>(
      tma_b, tma_c, reinterpret_cast<const cute::TmaDescriptor*>(tensormaps), split_metadata,
      batch_size, m, n, k, nheads, ngroups, nheads / ngroups, total_chunks);
  return true;
}

}  // namespace mamba
}  // namespace hpc
