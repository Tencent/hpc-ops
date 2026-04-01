// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace gemm {

namespace kernels {

template <int kBlockSwizzle, int kMmaSM>
__device__ __forceinline__ auto get_next_tile(int iblock, cutlass::FastDivmod flat_divider,
                                              cutlass::FastDivmod swizzle_divider) {
  int itile_m;
  int itile_n;

  if constexpr (kBlockSwizzle > 1) {
    int i_bxm, i_bxm_res;
    swizzle_divider(i_bxm, i_bxm_res, iblock);

    itile_m = (i_bxm * kBlockSwizzle + i_bxm_res % kBlockSwizzle) / kMmaSM;
    itile_n = i_bxm_res / kBlockSwizzle;
  } else {
    flat_divider(itile_n, itile_m, iblock);
    itile_m /= kMmaSM;
  }

  return cute::make_tuple(itile_m, itile_n);
}

template <typename TiledMma, typename TmaA, typename TmaB, typename TmaD, typename SLayoutA,
          typename SLayoutB, typename SLayoutC, typename Tin, typename Tout, int kTileM, int kTileN,
          int kTileK, int kStageK, int kClusterM, int kClusterN, int kClusterK, int kMmaSM,
          int kEpiTileN, int kStageTile, int kBlockSwizzle>
__global__ void __launch_bounds__(256, 1)
    gemm_bf16_kernel(const __grid_constant__ TmaA tma_a, const __grid_constant__ TmaB tma_b,
                     const __grid_constant__ TmaD tma_d, int m, int n, int k, int num_tile_m,
                     int num_tile_n, cutlass::FastDivmod flat_divider,
                     cutlass::FastDivmod swizzle_divider) {
  using namespace cute;  // NOLINT

  int idx = threadIdx.x;
  int iwarp = idx / 32;
  int elected = cute::elect_one_sync();

  int iblock = blockIdx.x;

  constexpr int kStageClc = 2;
  constexpr int kCtaTileM = kTileM / kMmaSM;
  constexpr int kClusterSize = kClusterM * kClusterN * kClusterK;

  __shared__ uint64_t tma_readable[kStageK];
  __shared__ uint64_t tma_writable[kStageK];
  __shared__ uint64_t tmem_readable[kStageTile];
  __shared__ uint64_t tmem_writable[kStageTile];

  __shared__ uint4 clc_resp[kStageClc] alignas(128);
  __shared__ uint64_t clc_readable[kStageClc];
  __shared__ uint64_t clc_writable[kStageClc];

  __shared__ uint32_t tmem_base_ptr;

  extern __shared__ uint8_t shm_data[] alignas(128);
  auto *shm_a = (Tin *)shm_data;
  auto *shm_b = (Tin *)(shm_a + cosize(SLayoutA{}));
  auto *shm_c = (Tin *)(shm_b + cosize(SLayoutB{}));

  auto sA = make_tensor(make_smem_ptr(shm_a), SLayoutA{});
  auto sB = make_tensor(make_smem_ptr(shm_b), SLayoutB{});
  auto sC = make_tensor(make_smem_ptr(shm_c), SLayoutC{});

  auto cluster_layout =
      tiled_divide(make_layout(make_shape(Int<kClusterM>{}, Int<kClusterN>{}, Int<kClusterK>{})),
                   make_tile(Int<kMmaSM>{}, Int<1>{}, Int<1>{}));
  int block_rank_in_cluster = cute::block_rank_in_cluster();
  auto cluster_coord = cluster_layout.get_flat_coord(block_rank_in_cluster);
  bool elected_cta = get<0>(cluster_coord) == Int<0>{};

  auto gA = tma_a.get_tma_tensor(make_shape(m, k));
  auto gB = tma_b.get_tma_tensor(make_shape(n, k));
  auto gC =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileN>{}), make_stride(Int<kTileN>{}, Int<1>{}));

  // TMA partition
  auto btma_ga = tma_a.get_slice(get<2>(cluster_coord) + get<0>(cluster_coord) * kClusterN);
  auto btma_gb =
      tma_b.get_slice(get<1>(cluster_coord) + get<0>(cluster_coord) * kClusterM / kMmaSM);
  auto btma_sa = tma_a.get_slice(get<2>(cluster_coord));
  auto btma_sb = tma_b.get_slice(get<1>(cluster_coord));

  auto tAg = btma_ga.partition_S(gA);  // (TMA, TMA_M, TMA_K)
  auto tAs = btma_sa.partition_D(sA);  // (TMA, _1, _1, kStage)

  auto tBg = btma_gb.partition_S(gB);  // (TMA, TMA_N, TMA_K)
  auto tBs = btma_sb.partition_D(sB);  // (TMA, -1, -1, stage)

  // UMMA partition
  TiledMma tiled_mma;
  auto cta_mma = tiled_mma.get_slice(get<0>(cluster_coord));

  auto tAs4r = cta_mma.partition_A(sA);
  auto tBs4r = cta_mma.partition_B(sB);
  auto tCgC = cta_mma.partition_C(gC);

  auto tAr = cta_mma.make_fragment_A(tAs4r);
  auto tBr = cta_mma.make_fragment_B(tBs4r);
  auto tCt = cta_mma.make_fragment_C(tCgC);

  uint16_t mcast_mask_c = 3;
  using TmemAllocator = TMEM::Allocator2Sm;
  TmemAllocator tmem_allocator{};

  __syncthreads();

  if (iwarp == 0 && elected) {
#pragma unroll
    for (int ik = 0; ik < kStageK; ++ik) {
      initialize_barrier(tma_readable[ik], 2);
      initialize_barrier(tma_writable[ik], 1);
    }

#pragma unroll
    for (int i = 0; i < kStageTile; ++i) {
      initialize_barrier(tmem_readable[i], 1);
      initialize_barrier(tmem_writable[i], 2);
    }

    constexpr int kMmaThreads = 32;
    constexpr int kEpiThreads = 128;
    constexpr int kTmaThreads = 1;
    constexpr int kSchedThreads = 1;
#pragma unroll
    for (int i = 0; i < kStageClc; i++) {
      initialize_barrier(clc_readable[i], 1);
      initialize_barrier(clc_writable[i],
                         kSchedThreads + kMmaThreads + kClusterSize * (kTmaThreads + kEpiThreads));
    }

    cutlass::arch::fence_barrier_init();
  } else if (iwarp == 1) {
    tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &tmem_base_ptr);
  }

  cluster_relaxed_sync();

  int ntile_k = size<2>(tAg);

  constexpr int kTransactionBytes =
      kMmaSM * sizeof(Tin) * (cosize(SLayoutA{}(_, _, 0)) + cosize(SLayoutB{}(_, _, 0)));

  if (iwarp == 0 && elected) {
    // TMA warp
    int phase = 1;
    int istage_k = 0;

    int phase_clc = 0;
    int istage_clc = 0;

    while (true) {
      auto [itile_m, itile_n] =
          get_next_tile<kBlockSwizzle, kMmaSM>(iblock, flat_divider, swizzle_divider);
      if (itile_m >= num_tile_m || itile_n >= num_tile_n) {
        break;
      }

      for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
        wait_barrier(tma_writable[istage_k], phase);
        copy(tma_a.with(tma_readable[istage_k]), tAg(_, itile_m, itile_k), tAs(_, 0, 0, istage_k));
        copy(tma_b.with(tma_readable[istage_k]), tBg(_, itile_n, itile_k), tBs(_, 0, 0, istage_k));
        if (elected_cta) {
          set_barrier_transaction_bytes(tma_readable[istage_k], kTransactionBytes);
        } else {
          arrive_cluster_barrier(tma_readable[istage_k]);
        }

        istage_k++;
        if (istage_k == kStageK) {
          phase ^= 1;
          istage_k = 0;
        }
      }

      wait_barrier(clc_readable[istage_clc], phase_clc);
      auto [icluster_first_cta, valid] = get_next_block(&clc_resp[istage_clc]);

      if (!valid) {
        break;
      }

      arrive_cluster_barrier(clc_writable[istage_clc]);

      istage_clc++;
      if (istage_clc == kStageClc) {
        istage_clc = 0;
        phase_clc ^= 1;
      }

      iblock = icluster_first_cta + block_rank_in_cluster;
    }
  } else if (iwarp == 1 && elected_cta) {
    // UMMA warp
    int phase = 0;
    int phase_tile = 1;
    int phase_clc = 0;
    int istage_k = 0;
    int istage_tile = 0;
    int istage_clc = 0;

    while (true) {
      auto [itile_m, itile_n] =
          get_next_tile<kBlockSwizzle, kMmaSM>(iblock, flat_divider, swizzle_divider);
      if (itile_m >= num_tile_m || itile_n >= num_tile_n) {
        break;
      }

      tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
      wait_barrier(tmem_writable[istage_tile], phase_tile);
      tCt.data() = tmem_base_ptr + istage_tile * kTileN;

      for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
        wait_barrier(tma_readable[istage_k], phase);

        // Execute a MmaTile_M x MmaTile_N x MmaTile_K GEMM
        for (int ik = 0; ik < size<2>(tAr); ++ik) {
          cute::gemm(tiled_mma, tAr(_, _, ik, istage_k), tBr(_, _, ik, istage_k), tCt);
          tiled_mma.accumulate_ = UMMA::ScaleOut::One;
        }

        cutlass::arch::umma_arrive_multicast_2x1SM(&tma_writable[istage_k], mcast_mask_c);

        istage_k++;
        if (istage_k == kStageK) {
          phase ^= 1;
          istage_k = 0;
        }
      }

      cutlass::arch::umma_arrive_multicast_2x1SM(&tmem_readable[istage_tile], mcast_mask_c);

      istage_tile++;
      if (istage_tile == kStageTile) {
        phase_tile ^= 1;
        istage_tile = 0;
      }

      wait_barrier(clc_readable[istage_clc], phase_clc);
      auto [icluster_first_cta, valid] = get_next_block(&clc_resp[istage_clc]);

      if (!valid) {
        break;
      }

      arrive_cluster_barrier(clc_writable[istage_clc]);

      istage_clc++;
      if (istage_clc == kStageClc) {
        istage_clc = 0;
        phase_clc ^= 1;
      }

      iblock = icluster_first_cta + block_rank_in_cluster;
    }
  } else if (iwarp == 2 && (block_rank_in_cluster == 0) && elected) {
    int phase_clc_read = 0;
    int phase_clc_write = 1;
    int istage_clc = 0;
    do {
      wait_barrier(clc_writable[istage_clc], phase_clc_write);
      find_next_block(&clc_resp[istage_clc], &clc_readable[istage_clc]);
      set_barrier_transaction_bytes(clc_readable[istage_clc], sizeof(uint4));
#pragma unroll
      for (int i = 1; i < kClusterSize; i++) {
        set_barrier_transaction_bytes_cluster(clc_readable[istage_clc], sizeof(uint4), i);
      }
      wait_barrier(clc_readable[istage_clc], phase_clc_read);
      auto [icluster_first_cta, valid] = get_next_block(&clc_resp[istage_clc]);

      if (!valid) {
        break;
      }

      arrive_cluster_barrier(clc_writable[istage_clc]);
      istage_clc++;
      if (istage_clc == kStageClc) {
        istage_clc = 0;
        phase_clc_write ^= 1;
        phase_clc_read ^= 1;
      }
    } while (true);
  } else if (idx >= 128) {
    idx -= 128;

    auto epi_tiler = make_tile(Int<kCtaTileM>{}, Int<kEpiTileN>{});
    auto tCt_epi = zipped_divide(tCt, make_tile(epi_tiler));
    auto sC_epi = zipped_divide(sC(_, _, 0), epi_tiler);

    // TiledCopy TMEM -> RMEM
    auto tiled_copy_t2r = make_tmem_copy(SM100_TMEM_LOAD_16dp256b4x{}, tCt_epi(_, _0{}));
    auto thr_copy_t2r = tiled_copy_t2r.get_slice(idx);
    auto tCt4r = thr_copy_t2r.partition_S(tCt_epi);
    auto tCr4t = make_tensor_like<float>(thr_copy_t2r.partition_D(sC_epi(_, 0)));

    // TiledCopy RMEM -> SMEM
    auto tiled_copy_r2s = make_tiled_copy_impl(
        Copy_Atom<cute::SM90_U32x4_STSM_N, Tout>{},
        make_layout(make_shape(make_shape(Int<4>{}, Int<8>{}, Int<4>{}),
                               make_shape(Int<2>{}, Int<2>{}, Int<kEpiTileN / 8>{}, Int<2>{})),
                    make_stride(make_stride(Int<256>{}, Int<1>{}, Int<32>{}),
                                make_stride(Int<128>{}, Int<8>{}, Int<1024>{}, Int<16>{}))),
        make_tile(Int<kCtaTileM>{}, Int<kEpiTileN>{}));

    auto thr_copy_r2s = tiled_copy_r2s.get_slice(idx);
    auto tCr4s = make_tensor_like<Tout>(thr_copy_r2s.partition_S(sC(_, _, 0)));
    auto tCs4r = thr_copy_r2s.partition_D(sC);

    auto &tCr = tCr4t;
    auto &tCr_T = tCr4s;

    auto nepi_tile = size<2>(tCt4r);

    // epi warpgroup
    int phase_tile = 0;
    int phase_clc = 0;

    int istage_tile = 0;
    int istage_clc = 0;

    bool is_leader = elected && (iwarp == 4);

    auto tCt4r_base_ptr = tCt4r.data();

    while (true) {
      auto [itile_m, itile_n] =
          get_next_tile<kBlockSwizzle, kMmaSM>(iblock, flat_divider, swizzle_divider);
      if (itile_m >= num_tile_m || itile_n >= num_tile_n) {
        break;
      }

      tCt4r.data() = tCt4r_base_ptr + istage_tile * kTileN;
      wait_barrier(tmem_readable[istage_tile], phase_tile);
#pragma unroll
      for (int iepi = 0; iepi < nepi_tile; iepi++) {
        // TMEM -> RMEM
        copy(tiled_copy_t2r, tCt4r(_, _, iepi), tCr4t);
        cutlass::arch::fence_view_async_tmem_load();

        auto tCr_fp2 = recast<float2>(tCr);
        auto tCr_bf162 = recast<__nv_bfloat162>(tCr_T);

        // cast
#pragma unroll
        for (int i = 0; i < cute::size(tCr_bf162); i++) {
          tCr_bf162(i) = __float22bfloat162_rn(tCr_fp2(i));
        }

        tma_store_wait<0>();
        cutlass::arch::NamedBarrier::sync(128, 0);

        // RMEM -> SMEM
        copy(tiled_copy_r2s, tCr4s(_, 0, 0), tCs4r(_, 0, 0, istage_tile));

        // SMEM -> GMEM
        tma_store_fence();
        cutlass::arch::NamedBarrier::sync(128, 0);

        if (iwarp == 4) {
          auto gD = tma_d.get_tma_tensor(make_shape(m, n));
          auto btma_d = tma_d.get_slice(0);

          auto tDs = btma_d.partition_S(sC);  // (TMA, _2, _1)
          auto tDg = btma_d.partition_D(gD);  // (TMA, TMA_M, TMA_N)

          cute::copy(tma_d, tDs(_, 0, 0, istage_tile),
                     tDg(_, itile_m * kMmaSM + get<0>(cluster_coord), itile_n * nepi_tile + iepi));
          tma_store_arrive();
        }
      }

      if (is_leader) {
        arrive_cluster_barrier(tmem_writable[istage_tile]);
      }

      istage_tile++;
      if (istage_tile == kStageTile) {
        phase_tile ^= 1;
        istage_tile = 0;
      }

      wait_barrier(clc_readable[istage_clc], phase_clc);
      auto [icluster_first_cta, valid] = get_next_block(&clc_resp[istage_clc]);

      if (!valid) {
        break;
      }

      arrive_cluster_barrier(clc_writable[istage_clc]);

      istage_clc++;
      if (istage_clc == kStageClc) {
        istage_clc = 0;
        phase_clc ^= 1;
      }

      iblock = icluster_first_cta + block_rank_in_cluster;
    }
  }

  __syncthreads();
  cluster_relaxed_sync();
  // Release the right to allocate before deallocations so that the next CTA can rasterize
  // Then deallocate TMEM
  if (iwarp == 1) {
    // tmem_allocator.release_allocation_lock();
    tmem_allocator.free(tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
  }
}

}  // namespace kernels

bool gemm_bf16_async(void *y_ptr, const void *x_ptr, const void *w_ptr, int m, int n, int k,
                     cudaStream_t stream) {
  using namespace cute;  // NOLINT

  using Tin = cute::bfloat16_t;
  using Tout = cute::bfloat16_t;

  constexpr int kTileM = 256;
  constexpr int kTileN = 256;
  constexpr int kTileK = 64;

  constexpr int kMmaSM = 2;

  constexpr int kEpiTileN = 32;
  constexpr int kStageK = 6;

  constexpr int kClusterM = 2;
  constexpr int kClusterN = 1;
  constexpr int kClusterK = 1;
  constexpr int kClusters = kClusterM * kClusterN * kClusterK;

  constexpr int kCtaTileM = kTileM / kMmaSM;
  constexpr int kCtaTileN = kTileN / kMmaSM;

  constexpr int kStageTile = 2;

  auto A = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(x_ptr)), make_shape(m, k),
                       make_stride(k, Int<1>{}));

  auto B = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(w_ptr)), make_shape(n, k),
                       make_stride(k, Int<1>{}));

  auto C = make_tensor(make_gmem_ptr(reinterpret_cast<Tout *>(y_ptr)), make_shape(m, n),
                       make_stride(n, Int<1>{}));

  auto tiled_mma = make_tiled_mma(SM100_MMA_F16BF16_2x1SM_SS<Tin, Tin, float, kTileM, kTileN,
                                                             UMMA::Major::K, UMMA::Major::K>{});

  auto slayout_a = tile_to_shape(UMMA::Layout_K_SW128_Atom<Tin>{},
                                 make_shape(Int<kCtaTileM>{}, Int<kTileK>{}, Int<kStageK>{}));
  auto slayout_b = tile_to_shape(UMMA::Layout_K_SW128_Atom<Tin>{},
                                 make_shape(Int<kCtaTileN>{}, Int<kTileK>{}, Int<kStageK>{}));
  auto slayout_c = tile_to_shape(UMMA::Layout_K_SW64_Atom<Tout>{},
                                 make_shape(Int<kCtaTileM>{}, Int<kEpiTileN>{}, Int<kStageTile>{}));

  auto copybox_a =
      tile_to_shape(UMMA::Layout_K_SW128_Atom<Tin>{}, make_shape(Int<kTileM>{}, Int<kTileK>{}));
  auto copybox_b =
      tile_to_shape(UMMA::Layout_K_SW128_Atom<Tin>{}, make_shape(Int<kTileN>{}, Int<kTileK>{}));

  auto tma_a = make_tma_copy(SM100_TMA_2SM_LOAD{}, A, copybox_a, Int<kClusterN * kMmaSM>{});
  auto tma_b = make_tma_copy(SM100_TMA_2SM_LOAD{}, B, copybox_b, Int<kClusterM>{});
  auto tma_d = make_tma_copy(SM90_TMA_STORE{}, C, slayout_c(_, _, 0));

  static constexpr int shm_ab = (cosize(slayout_a) + cosize(slayout_b)) * sizeof(Tin);
  static constexpr int shm_c = cosize(slayout_c) * sizeof(Tout);
  static constexpr int shm_size = shm_ab + shm_c;

  int num_tile_m = (m + kTileM - 1) / kTileM;
  int num_tile_n = (n + kTileN - 1) / kTileN;

  constexpr int kBlockSwizzle = 1;
  auto kernel = kernels::gemm_bf16_kernel<
      decltype(tiled_mma), decltype(tma_a), decltype(tma_b), decltype(tma_d), decltype(slayout_a),
      decltype(slayout_b), decltype(slayout_c), Tin, Tout, kTileM, kTileN, kTileK, kStageK,
      kClusterM, kClusterN, kClusterK, kMmaSM, kEpiTileN, kStageTile, kBlockSwizzle>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

  dim3 block(256);
  dim3 grid(num_tile_m * kMmaSM * num_tile_n);

  cutlass::FastDivmod swizzle_divider(kBlockSwizzle * num_tile_n);
  cutlass::FastDivmod flat_divider(num_tile_m * kMmaSM);

  cudaLaunchConfig_t config;
  memset(&config, 0, sizeof(config));

  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = shm_size;

  cudaLaunchAttribute attribute[1];
  attribute[0].id = cudaLaunchAttributeClusterDimension;
  attribute[0].val.clusterDim.x = kClusters;
  attribute[0].val.clusterDim.y = 1;
  attribute[0].val.clusterDim.z = 1;

  config.numAttrs = 1;
  config.attrs = attribute;

  config.stream = stream;

  cudaLaunchKernelEx(&config, kernel, tma_a, tma_b, tma_d, m, n, k, num_tile_m, num_tile_n,
                     flat_divider, swizzle_divider);
  return true;
}

}  // namespace gemm
}  // namespace hpc
