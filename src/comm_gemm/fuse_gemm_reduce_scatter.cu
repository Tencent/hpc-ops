// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "cutlass/arch/reg_reconfig.h"
#include "src/comm_gemm/comm_gemm.h"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace comm_gemm {

static constexpr int kWorldSize = 8;

namespace kernels {

static constexpr int kWavesPerSync = 4;

__device__ __forceinline__ auto get_next_tile(int iblock, int num_tile_m, int num_tile_n) {
  int tiles_per_rank = num_tile_m / kWorldSize;
  int itile_m = (iblock % kWorldSize) * tiles_per_rank + (iblock / kWorldSize) / num_tile_n;
  int itile_n = (iblock / kWorldSize) % num_tile_n;
  return cute::make_tuple(itile_m, itile_n);
}

__device__ __forceinline__ int comm_tile_to_gemm_tile(int iblock_comm, int rank) {
  return iblock_comm * kWorldSize + rank;
}

__device__ __forceinline__ auto get_next_comm_tile(int iblock_comm, int rank, int num_tile_m,
                                                   int num_tile_n) {
  int iblock_gemm = comm_tile_to_gemm_tile(iblock_comm, rank);
  auto [itile_m, itile_n] = get_next_tile(iblock_gemm, num_tile_m, num_tile_n);
  return cute::make_tuple(itile_m, itile_n, iblock_gemm);
}

__device__ __forceinline__ void multi_red_add_u64(void *mc_ptr, uint64_t value) {
  asm volatile("multimem.red.release.sys.global.add.u64 [%0], %1;"
               :
               : "l"(mc_ptr), "l"(value)
               : "memory");
}

__device__ uint64_t __forceinline__ load_sys_global_acquire(uint64_t *ptr) {
  uint64_t val;
  asm volatile("ld.acquire.sys.global.u64 {%0}, [%1];\n" : "=l"(val) : "l"(ptr));
  return val;
}

template <typename TiledMma, typename TmaA, typename TmaB, typename TmaAS, typename TmaBS,
          typename TmaBias, typename TmaD, int kTileM, int kTileN, int kStage, typename Tin,
          typename TY, typename Tout, typename SLayoutA, typename SLayoutB, typename SLayoutAS,
          typename SLayoutBS, typename SLayoutBIAS, typename SLayoutCT, bool HasBias>
__global__ void __launch_bounds__(384, 1) fuse_gemm_reduce_scatter_fp8(
    const __grid_constant__ TmaA tma_a, const __grid_constant__ TmaB tma_b,
    const __grid_constant__ TmaAS tma_as, const __grid_constant__ TmaBS tma_bs,
    const __grid_constant__ TmaBias tma_bias, const __grid_constant__ TmaD tma_d, Tout *y_ptr,
    const float *bias_ptr, int m, int n, int k, int m_pad, int num_block_n, int num_block_k,
    uint64_t *signal_ptr, Tout *multimem_output_ptr, uint64_t *multimem_signal_ptr, int num_comm_sm,
    int num_comp_sm, int rank, int world_size, int num_tile_m, int num_tile_n) {
  using namespace cute;  // NOLINT

  int num_tile = num_tile_m * num_tile_n;
  int sm_id = blockIdx.x;

  int num_wave = (num_tile + num_comp_sm - 1) / num_comp_sm;
  int iwave = 0;

  int num_tile_wave = num_comp_sm;
  int num_tile_last_wave = num_tile % num_comp_sm == 0 ? num_comp_sm : num_tile % num_comp_sm;

  if (sm_id < num_comp_sm) {
    int idx = threadIdx.x;

    int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
    int elected = cute::elect_one_sync();
    bool is_leader_in_block = (iwarp == 0) && elected;

    __shared__ uint64_t writable[kStage];
    __shared__ uint64_t readable[kStage];
    __shared__ uint64_t bias_writable;
    __shared__ uint64_t bias_readable;
    __shared__ uint64_t epilogue_ready;  // math → epilogue warp: sCT data ready
    __shared__ uint64_t smem_free;       // epilogue warp → math: sCT can be reused

    extern __shared__ uint8_t shm_data[] alignas(128);
    auto *shm_a = (Tin *)shm_data;
    auto *shm_b = (Tin *)(shm_a + cosize(SLayoutA{}));
    auto *shm_c = (TY *)(shm_b + cosize(SLayoutB{}));
    auto *shm_as = (float *)(shm_c + cosize(SLayoutCT{}));
    auto *shm_bs = (float *)(shm_as + cosize(SLayoutAS{}));
    auto *shm_bias = (float *)(shm_bs + cosize(SLayoutBS{}));

    auto sA = make_tensor(make_smem_ptr(shm_a), SLayoutA{});
    auto sB = make_tensor(make_smem_ptr(shm_b), SLayoutB{});
    auto sAS = make_tensor(make_smem_ptr(shm_as), SLayoutAS{});
    auto sBS = make_tensor(make_smem_ptr(shm_bs), SLayoutBS{});
    auto sBIAS = make_tensor(make_smem_ptr(shm_bias), SLayoutBIAS{});

    auto gA = tma_a.get_tma_tensor(make_shape(m, k));
    auto gB = tma_b.get_tma_tensor(make_shape(n, k));
    auto gC =
        make_tensor(make_gmem_ptr((Tout *)(nullptr)), make_shape(Int<kTileN>{}, Int<kTileM>{}),
                    make_stride(Int<kTileM>{}, Int<1>{}));
    auto gAS = tma_as.get_tma_tensor(make_shape(num_block_k, m_pad));
    auto gBS = tma_bs.get_tma_tensor(make_shape(num_block_n, num_block_k));
    auto gBIAS = tma_bias.get_tma_tensor(make_shape(1, n));

    auto btma_a = tma_a.get_slice(0);
    auto btma_b = tma_b.get_slice(0);
    auto btma_as = tma_as.get_slice(0);
    auto btma_bs = tma_bs.get_slice(0);
    auto btma_bias = tma_bias.get_slice(0);

    auto tAg = btma_a.partition_S(gA);  // (cpbox_x, num_tile_m, num_tile_k)
    auto tAs = btma_a.partition_D(sA);  // (cpbox_x, _1, _1, kStage)

    auto tBg = btma_b.partition_S(gB);  // (cpbox_w, num_tile_n, num_tile_k)
    auto tBs = btma_b.partition_D(sB);  // (cpbox_w, _1, _1, kStage)

    auto tASg = btma_as.partition_S(gAS);  // (cpbox_xs, num_tile_k, num_tile_m)
    auto tASs = btma_as.partition_D(sAS);  // (cpbox_xs, kStage, _1)

    auto tBSg = btma_bs.partition_S(gBS);  // (cpbox_ws, num_tile_n, num_tile_k / 4)
    auto tBSs = btma_bs.partition_D(sBS);  // (cpbox_ws, kStage, 32 / 4)

    auto tBIASg = btma_bias.partition_S(gBIAS);  // (TMA, num_tile_n)
    auto tBIASs = btma_bias.partition_D(sBIAS);  // (TMA, _1)

    int num_tile_m = size<1>(tAg);
    int num_tile_n = size<1>(tBg);

    if (is_leader_in_block) {
      if constexpr (HasBias) {
        initialize_barrier(bias_readable, 1);
        initialize_barrier(bias_writable, size(TiledMma{}) / 128);
      }
#pragma unroll
      for (int i = 0; i < kStage; ++i) {
        initialize_barrier(readable[i], 1);
        initialize_barrier(writable[i], size(TiledMma{}) / 128);
      }
      // epilogue async writeback barriers
      initialize_barrier(epilogue_ready, size(TiledMma{}) / 128);  // math warpgroups arrive
      initialize_barrier(smem_free, 1);                            // epilogue warp arrives
    }
    // sync to avoid ahead thread use(wait) readable when it is not initizlized yet
    __syncthreads();

    constexpr int kNumThreads = size(TiledMma{});
    // load warpgroup
    if (idx >= kNumThreads) {
      cutlass::arch::warpgroup_reg_dealloc<32>();
      idx -= kNumThreads;
      constexpr int kTransactionBytes =
          sizeof(Tin) * cosize(SLayoutA{}(_, _, 0)) + sizeof(Tin) * cosize(SLayoutB{}(_, _, 0)) +
          sizeof(float) * cosize(SLayoutAS{}(0, _)) + sizeof(float) * 4;
      constexpr int kBiasTransactionBytes = sizeof(float) * cosize(SLayoutBIAS{});

      int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
      int is_leader_in_load = ((iwarp == 0) && elected);
      bool is_epilogue_leader = ((iwarp == 1) && elected);

      if (is_leader_in_load) {
        int phase = 1;       // start with ok
        int phase_bias = 1;  // start with ok
        // int ismem_write = __shfl_sync(0xFFFFFFFF, 0, 0);
        int ismem_write = 0;
        int iblock = blockIdx.x;
        int num_tile_k = size<2>(tAg);
        int num_tile_total = num_tile_m * num_tile_n;

        while (true) {
          if (iblock >= num_tile_total) {
            break;
          }
          auto [itile_m, itile_n] = get_next_tile(iblock, num_tile_m, num_tile_n);

          iblock += num_comp_sm;

          // load bias
          if constexpr (HasBias) {
            wait_barrier(bias_writable, phase_bias);
            set_barrier_transaction_bytes(bias_readable, kBiasTransactionBytes);
            cute::copy(tma_bias.with(bias_readable), tBIASg(_, 0, itile_n), tBIASs(_, 0, 0));
            phase_bias ^= 1;
          }

#pragma unroll 1
          for (int itile_k = 0; itile_k < num_tile_k; itile_k++) {
            // load a, b
            wait_barrier(writable[ismem_write], phase);
            set_barrier_transaction_bytes(readable[ismem_write], kTransactionBytes);
            cute::copy(tma_a.with(readable[ismem_write]), tAg(_, itile_m, itile_k),
                       tAs(_, 0, 0, ismem_write));
            cute::copy(tma_b.with(readable[ismem_write]), tBg(_, itile_n, itile_k),
                       tBs(_, 0, 0, ismem_write));
            cute::copy(tma_as.with(readable[ismem_write]), tASg(_, itile_k, itile_m),
                       tASs(_, ismem_write, 0));
            cute::copy(tma_bs.with(readable[ismem_write]), tBSg(_, itile_n, itile_k / 4),
                       tBSs(_, ismem_write, 0));
            ++ismem_write;
            if (ismem_write == kStage) {
              ismem_write = 0;
              phase ^= 1;
            }
          }
        }  // while
      }  // if is_leader_in_load

      // epilogue warp: async TMA store + signal
      if (is_epilogue_leader) {
        auto sCT = make_tensor(make_smem_ptr((TY *)shm_c), SLayoutCT{});
        auto gD = tma_d.get_tma_tensor(make_shape(n, m));
        auto btma_d = tma_d.get_slice(0);
        auto tDs = btma_d.partition_S(sCT);  // (TMA, _2, _1)
        auto tDg = btma_d.partition_D(gD);   // (TMA, TMA_N, TMA_M)

        int epi_phase = 0;
        int epi_iblock = blockIdx.x;
        int epi_iwave = 0;
        int num_tile_total = num_tile_m * num_tile_n;

        while (true) {
          if (epi_iblock >= num_tile_total) {
            break;
          }
          auto [itile_m, itile_n] = get_next_tile(epi_iblock, num_tile_m, num_tile_n);
          epi_iblock += num_comp_sm;

          wait_barrier(epilogue_ready, epi_phase);
          epi_phase ^= 1;

          // TMA store for both warpgroups
          for (int wg = 0; wg < 2; wg++) {
            cute::copy(tma_d, tDs(_, wg, Int<0>{}), tDg(_, itile_n * 2 + wg, itile_m));
          }
          tma_store_arrive();

          // prepare signal logic before TMA finish
          int isync = epi_iwave / kWavesPerSync;
          auto *signal_wave_ptr = signal_ptr + isync * 2;
          auto counter =
              atomicAdd(reinterpret_cast<unsigned long long *>(signal_wave_ptr), 1ULL);  // NOLINT

          int sync_start = isync * kWavesPerSync;
          int sync_end = min(sync_start + kWavesPerSync, num_wave) - 1;
          int total_tiles = 0;
          for (int w = sync_start; w <= sync_end; w++) {
            total_tiles += (w == num_wave - 1) ? num_tile_last_wave : num_tile_wave;
          }

          cute::tma_store_wait<0>();
          // arrive smem_free so math can reuse sCT
          arrive_barrier(smem_free);

          if (counter == total_tiles - 1) {
            auto *mc_signal_wave_ptr = multimem_signal_ptr + isync * 2;
            multi_red_add_u64(mc_signal_wave_ptr + 1, 1);
          }
          epi_iwave += 1;
        }
      }  // if is_epilogue_leader
    } else {
      // math warpgroup
      cutlass::arch::warpgroup_reg_alloc<160>();

      int idx_in_warpgroup = idx % 128;
      int iwarpgroup = idx / 128;
      int iwarp_in_warpgroup = idx_in_warpgroup / 32;
      int elected_idx_in_warpgroup = ((iwarp_in_warpgroup == 0) && elected);

      TiledMma tiled_mma;

      auto thr_mma = tiled_mma.get_slice(idx);
      auto tBs4r = thr_mma.partition_A(sB);
      auto tAs4r = thr_mma.partition_B(sA);

      auto tBr = thr_mma.make_fragment_A(tBs4r);  // (MMA, MMA_N, MMA_K, kStage)
      auto tAr = thr_mma.make_fragment_B(tAs4r);  // (MMA, MMA_M, MMA_K, kStage)

      auto tCr = thr_mma.partition_fragment_C(gC);  // (MMA, MMA_N, MMA_M)

      auto tCr_mn = retile_fragment(tCr);
      constexpr int kM = size<0>(tCr_mn);
      constexpr int kN = size<1>(tCr_mn);

      auto gI = make_identity_tensor(gC.shape());
      auto tI = thr_mma.partition_C(gI);
      auto tI_mn = retile_fragment(tI);

      int ismem_read = 0;
      int phase = 0;
      int phase_bias = 0;

      int iblock = blockIdx.x;
      int num_tile_total = num_tile_m * num_tile_n;

      while (true) {
        if (iblock >= num_tile_total) {
          break;
        }
        auto [itile_m, itile_n] = get_next_tile(iblock, num_tile_m, num_tile_n);

        iblock += num_comp_sm;

        auto tDr = make_tensor_like(tCr);
        clear(tDr);

        int num_tile_k = size<2>(tAg);
#pragma unroll 1
        for (int itile_k = 0; itile_k < num_tile_k; itile_k++) {
          wait_barrier(readable[ismem_read], phase);

          float tCS[kN];
          float wscale = sBS(ismem_read, itile_k % 4);
#pragma unroll
          for (int in = 0; in < kN; in++) {
            float xscale = sAS(ismem_read, get<1>(tI_mn(0, in)));
            tCS[in] = xscale * wscale;
          }

          tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
          // mma
          warpgroup_fence_operand(tCr);
          warpgroup_arrive();
#pragma unroll
          for (int ik = 0; ik < size<2>(tAr); ++ik) {
            cute::gemm(tiled_mma, tBr(_, _, ik, ismem_read), tAr(_, _, ik, ismem_read),
                       tCr(_, _, _));
            tiled_mma.accumulate_ = GMMA::ScaleOut::One;
          }

          warpgroup_commit_batch();
          warpgroup_wait<0>();
          warpgroup_fence_operand(tCr);

          auto tDr_mn = retile_fragment(tDr);
#pragma unroll
          for (int in = 0; in < kN; in++) {
            float yscale = tCS[in];
#pragma unroll
            for (int im = 0; im < kM; im++) {
              tDr_mn(im, in) += tCr_mn(im, in) * yscale;
            }
          }

          if (elected_idx_in_warpgroup) {
            arrive_barrier(writable[ismem_read]);
          }

          ++ismem_read;
          if (ismem_read == kStage) {
            phase ^= 1;
            ismem_read = 0;
          }
        }  // for end
        // add bias
        if constexpr (HasBias) {
          wait_barrier(bias_readable, phase_bias);
          auto tDr_mn = retile_fragment(tDr);
#pragma unroll
          for (int im = 0; im < kM; im++) {
            auto bias = sBIAS(get<0>(tI_mn(im, 0)));
#pragma unroll
            for (int in = 0; in < kN; in++) {
              tDr_mn(im, in) += bias;
            }
          }
        }

        // float32 -> bfloat16
        auto tCrh = make_tensor_like<TY>(tCr);

#pragma unroll
        for (int i = 0; i < size(tCr); ++i) {
          tCrh(i) = (TY)(tDr(i));
        }

        // Epilogue
        auto sCT = make_tensor(make_smem_ptr((TY *)shm_c), SLayoutCT{});  // (TN, TM):(1, TN)
        using STSM_ATOM =
            std::conditional_t<(kTileM >= 16), cute::SM90_U16x8_STSM_T, cute::SM90_U16x4_STSM_T>;
        using STS_ATOM =
            std::conditional_t<std::is_same_v<TY, float>, UniversalCopy<uint32_t>, STSM_ATOM>;
        using R2SCopyAtomY = Copy_Atom<STS_ATOM, TY>;
        auto tiled_copy_c = make_tiled_copy_C(R2SCopyAtomY{}, tiled_mma);
        auto thr_copy_c = tiled_copy_c.get_slice(idx);

        auto tCr4s = thr_copy_c.retile_S(tCrh);
        auto tCs4r = thr_copy_c.partition_D(sCT);

        // wait epilogue warp to finish previous TMA store (sCT reusable)
        if (iwave > 0) {
          wait_barrier(smem_free, (iwave - 1) & 1);
        }
        syncwarpgroup(iwarpgroup);

        // bias arrive must wait all thread in warp have handled bias
        if constexpr (HasBias) {
          if (elected_idx_in_warpgroup) {
            arrive_barrier(bias_writable);
          }
          phase_bias ^= 1;
        }

        cute::copy(tiled_copy_c, tCr4s, tCs4r);
        syncwarpgroup(iwarpgroup);
        cute::tma_store_fence();

        // notify epilogue warp: sCT data is ready
        if (elected_idx_in_warpgroup) {
          arrive_barrier(epilogue_ready);
        }
        iwave += 1;
      }  // while
    }  // else

  } else {
    int idx = threadIdx.x;
    int iblock_comm = blockIdx.x - num_comp_sm;
    int iblock_range = 0;
    int num_comm_tile = (num_tile_m / kWorldSize) * num_tile_n;

    int kWarpCount = (256 + 128) / 32;
    int iwarp = threadIdx.x / 32;
    int ilane = threadIdx.x % 32;
    int iwarp_half = ilane / 16;
    int ilane_half = ilane % 16;
    constexpr int kN = 4;

    int num_sync = (num_wave + kWavesPerSync - 1) / kWavesPerSync;
    for (int isync = 0; isync < num_sync; isync++) {
      auto *signal_wave_ptr = signal_ptr + isync * 2 + 1;
      if (idx == 0) {
        while (load_sys_global_acquire(signal_wave_ptr) != world_size) {
        }
      }
      __syncthreads();

      int sync_start = isync * kWavesPerSync;
      int sync_end = min(sync_start + kWavesPerSync, num_wave);
      for (int w = sync_start; w < sync_end; w++) {
        iblock_range += (w == num_wave - 1) ? num_tile_last_wave : num_tile_wave;
      }

      for (;; iblock_comm += num_comm_sm) {
        if (iblock_comm >= num_comm_tile) {
          break;
        }
        auto [itile_m, itile_n, iblock_comp] =
            get_next_comm_tile(iblock_comm, rank, num_tile_m, num_tile_n);

        if (iblock_comp >= iblock_range) {
          break;
        }

        int row = itile_m * kTileM;
        int col = itile_n * kTileN;

        auto *y_tile = y_ptr + row * n + col;
        auto *multi_y_tile = multimem_output_ptr + row * n + col;

#pragma unroll
        for (int irow = iwarp * 2; irow < kTileM; irow += kWarpCount * 2) {
          if (row + irow + iwarp_half >= m) {
            break;
          }
          int warp_row = row + irow + iwarp_half;
          bool cond = warp_row >= m / 8 * rank && warp_row < m / 8 * (rank + 1);
          if (!cond) {
            break;
          }
          int icol = ilane_half * kN;
          if (col + icol >= n) {
            break;
          }

          auto in_sum = multi_load_reduce_add<__nv_bfloat162, kN>(
              reinterpret_cast<__nv_bfloat162 *>(multi_y_tile + (irow + iwarp_half) * n) + icol);
          store(reinterpret_cast<__nv_bfloat162 *>(y_tile + (irow + iwarp_half) * n) + icol,
                in_sum);
        }
        __syncwarp();
      }
    }
    __syncthreads();
    __shared__ int is_last;
    if (idx == 0) {
      auto old =
          atomicAdd(reinterpret_cast<unsigned long long *>(signal_ptr + 2 * num_wave),  // NOLINT
                    1ULL);
      is_last = (old == num_comm_sm - 1);
    }
    __syncthreads();
    if (is_last) {
      for (int i = idx; i <= 2 * num_wave; i += blockDim.x) {
        signal_ptr[i] = 0;
      }
    }
    // comm SM end
  }
}

}  // namespace kernels

template <int kTileM>
static constexpr auto mma_selector() {
  if constexpr (kTileM == 8) {
    return cute::SM90_64x8x32_F32E4M3E4M3_SS_TN<>{};
  } else if constexpr (kTileM == 16) {
    return cute::SM90_64x16x32_F32E4M3E4M3_SS_TN<>{};
  } else if constexpr (kTileM == 32) {
    return cute::SM90_64x32x32_F32E4M3E4M3_SS_TN<>{};
  } else if constexpr (kTileM == 48) {
    return cute::SM90_64x48x32_F32E4M3E4M3_SS_TN<>{};
  } else if constexpr (kTileM == 64) {
    return cute::SM90_64x64x32_F32E4M3E4M3_SS_TN<>{};
  } else if constexpr (kTileM == 96) {
    return cute::SM90_64x96x32_F32E4M3E4M3_SS_TN<>{};
  } else if constexpr (kTileM == 128) {
    return cute::SM90_64x128x32_F32E4M3E4M3_SS_TN<>{};
  } else {
    return cute::SM90_64x64x32_F32E4M3E4M3_SS_TN<>{};
  }
}

template <int kTileM, int kTileN, int kTileK, int kStage, bool HasBias>
void launch_fuse_gemm_reduce_scatter_fp8(void *y_ptr, const void *x_ptr, const void *w_ptr,
                                         const void *x_scale_ptr, const void *weight_scale_ptr,
                                         const void *bias_ptr, int m, int n, int k, int m_pad,
                                         int num_block_k, int num_block_n, cudaStream_t stream,
                                         void *signal_ptr, void *multimem_output_ptr,
                                         void *multimem_signal_ptr, int num_comp_sm,
                                         int num_comm_sm, int rank, int world_size) {
  using namespace cute;  // NOLINT

  using Tin = cute::float_e4m3_t;
  using Tout = cute::bfloat16_t;
  using TS = float;
  using TBIAS = float;
  using TY = Tout;

  auto X = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(x_ptr)), make_shape(m, k),
                       make_stride(k, Int<1>{}));
  auto W = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(w_ptr)), make_shape(n, k),
                       make_stride(k, Int<1>{}));
  auto Y = make_tensor(make_gmem_ptr(reinterpret_cast<TY *>(y_ptr)), make_shape(n, m),
                       make_stride(Int<1>{}, n));
  auto XS = make_tensor(make_gmem_ptr(reinterpret_cast<const TS *>(x_scale_ptr)),
                        make_shape(num_block_k, m_pad), make_stride(m_pad, Int<1>{}));
  auto WS = make_tensor(make_gmem_ptr(reinterpret_cast<const TS *>(weight_scale_ptr)),
                        make_shape(num_block_n, num_block_k), make_stride(num_block_k, Int<1>{}));
  auto BIAS = make_tensor(make_gmem_ptr(reinterpret_cast<const TBIAS *>(bias_ptr)),
                          make_shape(1, n), make_stride(n, Int<1>{}));

  auto slayout_x = tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{},
                                 make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{}));
  auto slayout_w = tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{},
                                 make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{}));
  static constexpr int kMDim = (kTileM + 31) / 32 * 32;
  auto slayout_xs = make_layout(
      make_shape(Int<kStage>{}, Int<kMDim>{}),  // shared memory address must be 128 byte aligned
      make_stride(Int<kMDim>{}, Int<1>{}));
  auto slayout_ws = make_layout(
      make_shape(Int<kStage>{}, Int<32>{}),
      make_stride(Int<32>{}, Int<1>{}));  // shared memory address must be 128 byte aligned
  auto slayout_bias =
      make_layout(make_shape(Int<1>{}, Int<kTileN>{}), make_stride(Int<kTileN>{}, Int<1>{}));
  auto slayout_yt =
      tile_to_shape(GMMA::Layout_MN_SW64_Atom<TY>{}, make_shape(Int<kTileN>{}, Int<kTileM>{}));

  auto cpbox_xs =
      make_layout(make_shape(Int<1>{}, Int<kMDim>{}), make_stride(Int<kMDim>{}, Int<1>{}));
  auto cpbox_ws = make_layout(
      make_shape(Int<1>{}, Int<4>{}),
      make_stride(Int<4>{}, Int<1>{}));  // size of transfer must be a multiple of 16 bytes.
  auto cpbox_bias =
      make_layout(make_shape(Int<1>{}, Int<kTileN>{}), make_stride(Int<kTileN>{}, Int<1>{}));
  auto cpbox_yt =
      tile_to_shape(GMMA::Layout_MN_SW64_Atom<TY>{}, make_shape(Int<kTileN / 2>{}, Int<kTileM>{}));

  static constexpr int shm_xw = sizeof(Tin) * (cosize(slayout_x) + cosize(slayout_w));
  static constexpr int shm_y = sizeof(TY) * cosize(slayout_yt);
  static constexpr int shm_xws = (cosize(slayout_xs) + cosize(slayout_ws)) * sizeof(TS);
  static constexpr int shm_bias = HasBias ? cosize(slayout_bias) * sizeof(TBIAS) : 0;
  static constexpr int shm_size = shm_xw + shm_y + shm_xws + shm_bias;

  auto tma_x = make_tma_copy(SM90_TMA_LOAD{}, X, slayout_x(_, _, 0));
  auto tma_w = make_tma_copy(SM90_TMA_LOAD{}, W, slayout_w(_, _, 0));
  auto tma_y = make_tma_copy(SM90_TMA_STORE{}, Y, cpbox_yt);
  auto tma_xs = make_tma_copy(SM90_TMA_LOAD{}, XS, cpbox_xs);
  auto tma_ws = make_tma_copy(SM90_TMA_LOAD{}, WS, cpbox_ws);
  auto tma_bias = make_tma_copy(SM90_TMA_LOAD{}, BIAS, cpbox_bias);

  auto warpgroup_layout = make_layout(make_shape(Int<2>{}, Int<1>{}, Int<1>{}));
  auto tiled_mma = make_tiled_mma(mma_selector<kTileM>(), warpgroup_layout);

  auto kernel = kernels::fuse_gemm_reduce_scatter_fp8<
      decltype(tiled_mma), decltype(tma_x), decltype(tma_w), decltype(tma_xs), decltype(tma_ws),
      decltype(tma_bias), decltype(tma_y), kTileM, kTileN, kStage, Tin, TY, Tout,
      decltype(slayout_x), decltype(slayout_w), decltype(slayout_xs), decltype(slayout_ws),
      decltype(slayout_bias), decltype(slayout_yt), HasBias>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

  int num_tile_m = (m + kTileM - 1) / kTileM;
  int num_tile_n = (n + kTileN - 1) / kTileN;
  int num_tile = num_tile_m * num_tile_n;

  assert(num_comp_sm > 0 && num_comm_sm > 0);
  assert(num_comp_sm + num_comm_sm <= get_sm_count());
  num_comp_sm = std::min(num_comp_sm, num_tile);

  dim3 block(size(tiled_mma) + 128);
  dim3 grid(num_comp_sm + num_comm_sm);

  kernel<<<grid, block, shm_size, stream>>>(
      tma_x, tma_w, tma_xs, tma_ws, tma_bias, tma_y, reinterpret_cast<Tout *>(y_ptr),
      reinterpret_cast<const float *>(bias_ptr), m, n, k, m_pad, num_block_n, num_block_k,
      reinterpret_cast<uint64_t *>(signal_ptr), reinterpret_cast<Tout *>(multimem_output_ptr),
      reinterpret_cast<uint64_t *>(multimem_signal_ptr), num_comm_sm, num_comp_sm, rank, world_size,
      num_tile_m, num_tile_n);
}

#define LAUNCH_GEMM_INTERNAL(TILE_M_VAL)                                                          \
  do {                                                                                            \
    constexpr int kTileM = TILE_M_VAL;                                                            \
    if (bias_ptr) {                                                                               \
      launch_fuse_gemm_reduce_scatter_fp8<kTileM, kTileN, kTileK, kStage, true>(                  \
          y_ptr, x_ptr, w_ptr, x_scale_ptr, weight_scale_ptr, bias_ptr, m, n, k, m_pad,           \
          num_block_k, num_block_n, stream, signal_ptr, multimem_output_ptr, multimem_signal_ptr, \
          num_comp_sm, num_comm_sm, rank, world_size);                                            \
    } else {                                                                                      \
      launch_fuse_gemm_reduce_scatter_fp8<kTileM, kTileN, kTileK, kStage, false>(                 \
          y_ptr, x_ptr, w_ptr, x_scale_ptr, weight_scale_ptr, bias_ptr, m, n, k, m_pad,           \
          num_block_k, num_block_n, stream, signal_ptr, multimem_output_ptr, multimem_signal_ptr, \
          num_comp_sm, num_comm_sm, rank, world_size);                                            \
    }                                                                                             \
  } while (0)

void fuse_gemm_reduce_scatter_fp8_async(void *y_ptr, const void *x_ptr, const void *w_ptr,
                                        const void *x_scale_ptr, const void *weight_scale_ptr,
                                        const void *bias_ptr, int m, int n, int k, int m_pad,
                                        int num_block_k, int num_block_n, cudaStream_t stream,
                                        void *signal_ptr, void *multimem_output_ptr,
                                        void *multimem_signal_ptr, int num_comp_sm, int num_comm_sm,
                                        int rank, int world_size) {
  constexpr int kTileN = 128;
  constexpr int kTileK = 128;
  constexpr int kStage = 6;

  if (m <= 8) {
    LAUNCH_GEMM_INTERNAL(8);
  } else if (m <= 16) {
    LAUNCH_GEMM_INTERNAL(16);
  } else if (m <= 32) {
    LAUNCH_GEMM_INTERNAL(32);
  } else if (m <= 48) {
    LAUNCH_GEMM_INTERNAL(48);
  } else {
    LAUNCH_GEMM_INTERNAL(64);
  }
}

#undef LAUNCH_GEMM_INTERNAL
}  // namespace comm_gemm
}  // namespace hpc
