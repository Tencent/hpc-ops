// Copyright 2025 hpc-ops authors

#ifndef SRC_GROUP_GEMM_SM100_GROUP_GEMM_CP_ASYNC_FP8_ACT_MUL_CUH_
#define SRC_GROUP_GEMM_SM100_GROUP_GEMM_CP_ASYNC_FP8_ACT_MUL_CUH_

#include <cuda.h>
#include <stdio.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace group_gemm {

namespace kernels {

template <int kMmaSM>
__device__ __forceinline__ void get_next_tile_horizon(const int *tiles_ptr, int iblock,
                                                      int num_group, int &igroup, int &itile_m,
                                                      int &itile_n, int &sum_tile_n,
                                                      cutlass::FastDivmod flat_divider);

// Config used by the fused kernel. Identical to GroupGEMMFp8Config, with the
// following additions:
//   - Tout is the FP8 output element for the fused activation result.
//   - SLayoutY / SLayoutYT use sizeof(Tout) == 1 byte.
//   - SMEM A is still Tin (fp8). The only extra SMEM cost beyond the
//     original 1SM kernel is that we hold TWO A tiles per stage (gate + up).
//
// TMEM layout (1SM, 2 accumulators per pipeline stage):
//     kStageTile stages * (kTileN for gate + kTileN for up) <= 512 cols
// so kStageTile * kTileN * 2 <= 512.

// The fused kernel itself. Template parameters mirror the existing 1SM
// kernel: Config is GroupGEMMFp8Config (with Tout = fp8).
template <typename Config, typename TiledMma, typename TmaA, typename TmaB, typename TmaDT,
          int kTaskLoopPolicy, bool kUsePDL = false>
__global__ void __launch_bounds__(384, 1) group_gemm_1sm_cp_async_fp8_act_mul_kernel(
    const __grid_constant__ TmaA tma_a_gate, const __grid_constant__ TmaA tma_a_up,
    cute::TmaDescriptor *td_xy, cute::float_e4m3_t *A_gate_ptr, cute::float_e4m3_t *A_up_ptr,
    cute::float_e4m3_t *Bptr, int *cu_seqlens_ptr, int *seqlens_ptr, const float *gate_up_scale_ptr,
    const float *act_mul_scale_ptr, int *tiles_ptr, int *cu_tiles_ptr, int num_group, int m, int n,
    int k, cutlass::FastDivmod flat_divider, const int *x_row_map_ptr = nullptr,
    int x_num_rows = 0) {
  using namespace cute;  // NOLINT

  using Tin = typename Config::Tin;
  using Tout = typename Config::Tout;  // fp8_e4m3_t
  using SLayoutA = typename Config::SLayoutX;
  using SLayoutB = typename Config::SLayoutW;
  using SLayoutC = typename Config::SLayoutY;
  using SLayoutCT = typename Config::SLayoutYT;

  using G2SCopy = typename Config::G2SCopy;

  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;
  constexpr int kStageK = Config::kStage;
  constexpr int kClusterM = Config::kClusterM;
  constexpr int kClusterN = Config::kClusterN;
  constexpr int kClusterK = Config::kClusterK;
  constexpr int kMmaSM = Config::kMmaSM;
  constexpr int kEpiTileN = Config::kEpiTileN;
  constexpr int kStageTile = Config::kStageTile;
  constexpr int kStageTMA = Config::kStageTMA;
  constexpr int kCtaTileM = Config::kCtaTileM;

  int idx = threadIdx.x;
  int iwarp = idx / 32;
  int elected = cute::elect_one_sync();

  int iblock = blockIdx.x;

  constexpr int kStageClc = 5;
  constexpr int kClusterSize = kClusterM * kClusterN * kClusterK;

  // 2 accumulators per tile stage (gate + up) => 2 separate TMEM regions per stage.
  // Each accumulator region occupies kTileN columns, so tmem stride between
  // the gate and up region of the same stage is kTileN columns.
  constexpr int kTmemColsPerStage = 2 * kTileN;

  __shared__ uint64_t cp_async_readable[kStageK];
  __shared__ uint64_t cp_async_writable[kStageK];
  // Single tma_readable barrier tracks BOTH gate and up A loads for a stage
  // (2 * kTransactionBytesA bytes total).
  __shared__ uint64_t tma_readable[kStageK];
  __shared__ uint64_t tma_writable[kStageK];
  __shared__ uint64_t tmem_readable[kStageTile];
  __shared__ uint64_t tmem_writable[kStageTile];

  __shared__ uint64_t task_readable[kStageClc];
  __shared__ uint64_t task_writable[kStageClc];

  __shared__ int task_shm[kStageClc][4];

  __shared__ uint32_t tmem_base_ptr;

  extern __shared__ uint8_t shm_data[] alignas(128);
  auto *shm_a_gate = reinterpret_cast<Tin *>(shm_data);
  auto *shm_a_up = shm_a_gate + cosize(SLayoutA{});
  auto *shm_b = shm_a_up + cosize(SLayoutA{});
  auto *shm_c = reinterpret_cast<Tout *>(shm_b + cosize(SLayoutB{}));

  int *shm_tiles = reinterpret_cast<int *>(shm_c + cosize(SLayoutCT{}));

  TmaB tma_b;
  TmaDT tma_dt;

  auto sA_gate = make_tensor(make_smem_ptr(shm_a_gate), SLayoutA{});
  auto sA_up = make_tensor(make_smem_ptr(shm_a_up), SLayoutA{});
  auto sB = make_tensor(make_smem_ptr(shm_b), SLayoutB{});
  auto sC = make_tensor(make_smem_ptr(shm_c), SLayoutC{});
  auto sCT = make_tensor(make_smem_ptr(shm_c), SLayoutCT{});

  auto cluster_layout =
      tiled_divide(make_layout(make_shape(Int<kClusterM>{}, Int<kClusterN>{}, Int<kClusterK>{})),
                   make_tile(Int<kMmaSM>{}, Int<1>{}, Int<1>{}));
  int block_rank_in_cluster = cute::block_rank_in_cluster();
  auto cluster_coord = cluster_layout.get_flat_coord(block_rank_in_cluster);
  bool elected_cta = get<0>(cluster_coord) == Int<0>{};

  auto gA_gate = tma_a_gate.get_tma_tensor(make_shape(m, k, num_group));
  auto gA_up = tma_a_up.get_tma_tensor(make_shape(m, k, num_group));
  auto gB = tma_b.get_tma_tensor(make_shape(n, k));
  auto gC =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileN>{}), make_stride(Int<kTileN>{}, Int<1>{}));

  auto B = make_tensor(make_gmem_ptr(reinterpret_cast<Tin *>(Bptr)), make_shape(n, k),
                       make_stride(k, Int<1>{}));
  auto gB1 = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(0, _));
  auto IB = make_identity_tensor(make_shape(Int<kTileN>{}, Int<kTileK>{}));

  // TMA partition
  auto btma_a_gate = tma_a_gate.get_slice(0);
  auto btma_a_up = tma_a_up.get_slice(0);
  auto btma_b = tma_b.get_slice(0);

  auto tAg_gate = btma_a_gate.partition_S(gA_gate);  // (TMA, TMA_M, TMA_K, num_group)
  auto tAg_up = btma_a_up.partition_S(gA_up);
  auto tAs_gate = btma_a_gate.partition_D(sA_gate);  // (TMA, _1, _1, kStage)
  auto tAs_up = btma_a_up.partition_D(sA_up);

  auto tBg = btma_b.partition_S(gB);  // (TMA, TMA_N, TMA_K)
  auto tBs = btma_b.partition_D(sB);  // (TMA, _1, _1, stage)

  // UMMA partition
  TiledMma tiled_mma;
  auto cta_mma = tiled_mma.get_slice(0);

  auto tAs4r_gate = cta_mma.partition_A(sA_gate);
  auto tAs4r_up = cta_mma.partition_A(sA_up);
  auto tBs4r = cta_mma.partition_B(sB);
  auto tCgC = cta_mma.partition_C(gC);

  auto tAr_gate = cta_mma.make_fragment_A(tAs4r_gate);
  auto tAr_up = cta_mma.make_fragment_A(tAs4r_up);
  auto tBr = cta_mma.make_fragment_B(tBs4r);
  auto tCt = cta_mma.make_fragment_C(tCgC);

  using TmemAllocator = TMEM::Allocator1Sm;
  TmemAllocator tmem_allocator{};

  if (iwarp == 0 && elected) {
#pragma unroll
    for (int ik = 0; ik < kStageK; ++ik) {
      initialize_barrier(cp_async_readable[ik], 128);
      initialize_barrier(cp_async_writable[ik], 1);
      initialize_barrier(tma_readable[ik], 1);
      initialize_barrier(tma_writable[ik], 1);
    }

#pragma unroll
    for (int i = 0; i < kStageTile; ++i) {
      initialize_barrier(tmem_readable[i], 1);
      initialize_barrier(tmem_writable[i], 1);
    }

    constexpr int kMmaThreads = 32;
    constexpr int kEpiThreads = 128;
    constexpr int kTmaThreads = 128;
#pragma unroll
    for (int i = 0; i < kStageClc; i++) {
      initialize_barrier(task_readable[i], 1);
      initialize_barrier(task_writable[i],
                         kMmaThreads + kClusterSize * (kTmaThreads + kEpiThreads));
    }

    cutlass::arch::fence_barrier_init();
  } else if (iwarp == 1) {
    tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &tmem_base_ptr);
  }

  int ntile_k = size<2>(tAg_gate);

  // constexpr int kTransactionBytesA = kMmaSM * sizeof(Tin) * cosize(SLayoutA{}(_, _, 0));
  constexpr int kTransactionBytesA = kMmaSM * sizeof(Tin) * kCtaTileM * kTileK;
  // Gate + Up A loads share the same tma_readable barrier.
  constexpr int kTransactionBytesA_Both = 2 * kTransactionBytesA;

  int total_m = 0;

  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }

  if constexpr (kTaskLoopPolicy == 1) {
    for (int i = idx; i < num_group; i += blockDim.x) {
      shm_tiles[i] = tiles_ptr[i];
    }
  } else if constexpr (kTaskLoopPolicy == 2) {
    total_m = cu_tiles_ptr[num_group];
    for (int i = idx; i < (num_group + 1); i += blockDim.x) {
      shm_tiles[i] = cu_tiles_ptr[i];
    }
  }

  __syncthreads();

  // =====================================================================
  // Warpgroup 2 (idx 256..383): TMA loader for A (gate+up) + cp.async B
  // =====================================================================
  if (idx >= 256) {
    idx -= 256;

    int phase = 1;
    int istage_k = 0;
    int phase_clc = 0;
    int istage_clc = 0;

    int igroup = 0;
    int sum_tile_n = 0;
    int itile_m, itile_n;

    get_next_tile_horizon<kMmaSM>(shm_tiles, iblock, num_group, igroup, itile_m, itile_n,
                                  sum_tile_n, flat_divider);

    // B partitions only — A uses TMA
    G2SCopy g2s_tiled_copy;
    auto g2s_thr_copy = g2s_tiled_copy.get_slice(idx);
    auto tBg1 = g2s_thr_copy.partition_S(gB1);
    auto tBs1 = g2s_thr_copy.partition_D(sB);
    auto tIB = g2s_thr_copy.partition_S(IB);

    constexpr int kCopyN = size<1>(tBs1);
    int b_row_offsets[kCopyN];

    const bool kUseRowMap = (x_row_map_ptr != nullptr);

    bool is_tma_leader = (idx == 0);

    while (true) {
      if (igroup >= 0) {
        int cu_seq = cu_seqlens_ptr[igroup];
        int tile_base_row = cu_seq + itile_n * kTileN;
        int remaining_n = seqlens_ptr[igroup] - itile_n * kTileN;
#pragma unroll
        for (int ir = 0; ir < kCopyN; ir++) {
          int ir_in_tile = get<0>(tIB(0, ir, 0));
          bool valid = (ir_in_tile < remaining_n);
          int abs_row = tile_base_row + ir_in_tile;
          int src_row;
          if (kUseRowMap) {
            src_row = valid ? x_row_map_ptr[abs_row] : ir_in_tile;
          } else {
            src_row = valid ? abs_row : ir_in_tile;
          }
          b_row_offsets[ir] = (src_row - ir_in_tile) * k;
        }

        for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
          wait_barrier(cp_async_writable[istage_k], phase);

          // A (weight): TMA — single thread issues gate + up back-to-back,
          // both arriving on the same tma_readable barrier.
          if (is_tma_leader) {
            copy(tma_a_gate.with(tma_readable[istage_k]), tAg_gate(_, itile_m, itile_k, igroup),
                 tAs_gate(_, 0, 0, istage_k));
            copy(tma_a_up.with(tma_readable[istage_k]), tAg_up(_, itile_m, itile_k, igroup),
                 tAs_up(_, 0, 0, istage_k));
            set_barrier_transaction_bytes(tma_readable[istage_k], kTransactionBytesA_Both);
          }

          // B (activation): CP.ASYNC — all 128 threads, per-row indirect
          // addressing via b_row_offsets[].
#pragma unroll
          for (int ir = 0; ir < kCopyN; ir++) {
            auto tBg_src = make_tensor(tBg1(_, ir, _, itile_k).data() + b_row_offsets[ir],
                                       tBg1(_, ir, _, itile_k).layout());
            cute::copy(g2s_tiled_copy, tBg_src, tBs1(_, ir, _, istage_k));
          }
          cpasync_barrier_arrive_noinc(reinterpret_cast<uint64_t *>(&cp_async_readable[istage_k]));

          istage_k++;
          if (istage_k == kStageK) {
            phase ^= 1;
            istage_k = 0;
          }
        }
      }

      wait_barrier(task_readable[istage_clc], phase_clc);
      igroup = task_shm[istage_clc][0];
      itile_m = task_shm[istage_clc][1];
      itile_n = task_shm[istage_clc][2];
      arrive_barrier(task_writable[istage_clc]);

      if (igroup < 0) {
        break;
      }

      istage_clc++;
      if (istage_clc == kStageClc) {
        istage_clc = 0;
        phase_clc ^= 1;
      }
    }
  } else if (iwarp == 1) {
    // =====================================================================
    // UMMA warp: 2 sequential MMAs per K-tile (gate acc, then up acc).
    // =====================================================================
    int phase = 0;
    int phase_tile = 1;
    int phase_clc = 0;
    int istage_k = 0;
    int istage_tile = 0;
    int istage_clc = 0;

    int igroup = 0;
    int sum_tile_n = 0;
    int itile_m, itile_n;

    get_next_tile_horizon<kMmaSM>(shm_tiles, iblock, num_group, igroup, itile_m, itile_n,
                                  sum_tile_n, flat_divider);

    auto tCt_gate = tCt;
    auto tCt_up = tCt;

    while (true) {
      if (igroup >= 0) {
        wait_barrier(tmem_writable[istage_tile], phase_tile);
        tCt_gate.data() = tmem_base_ptr + istage_tile * kTmemColsPerStage;
        tCt_up.data() = tmem_base_ptr + istage_tile * kTmemColsPerStage + kTileN;

        for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
          wait_barrier(tma_readable[istage_k], phase);
          wait_barrier(cp_async_readable[istage_k], phase);

          // First K-iteration of the tile zeros the accumulators; subsequent
          // iterations add into them.
          tiled_mma.accumulate_ = (itile_k == 0) ? UMMA::ScaleOut::Zero : UMMA::ScaleOut::One;

          for (int ik = 0; ik < size<2>(tAr_gate); ++ik) {
            gemm(tiled_mma, tAr_gate(_, _, ik, istage_k), tBr(_, _, ik, istage_k), tCt_gate);
            tiled_mma.accumulate_ = UMMA::ScaleOut::One;
          }

          tiled_mma.accumulate_ = (itile_k == 0) ? UMMA::ScaleOut::Zero : UMMA::ScaleOut::One;
          for (int ik = 0; ik < size<2>(tAr_up); ++ik) {
            gemm(tiled_mma, tAr_up(_, _, ik, istage_k), tBr(_, _, ik, istage_k), tCt_up);
            tiled_mma.accumulate_ = UMMA::ScaleOut::One;
          }

          cutlass::arch::umma_arrive(&cp_async_writable[istage_k]);

          istage_k++;
          if (istage_k == kStageK) {
            phase ^= 1;
            istage_k = 0;
          }
        }

        cutlass::arch::umma_arrive(&tmem_readable[istage_tile]);

        istage_tile++;
        if (istage_tile == kStageTile) {
          phase_tile ^= 1;
          istage_tile = 0;
        }
      }

      wait_barrier(task_readable[istage_clc], phase_clc);
      igroup = task_shm[istage_clc][0];
      itile_m = task_shm[istage_clc][1];
      itile_n = task_shm[istage_clc][2];
      arrive_barrier(task_writable[istage_clc]);

      if (igroup < 0) {
        break;
      }

      istage_clc++;
      if (istage_clc == kStageClc) {
        istage_clc = 0;
        phase_clc ^= 1;
      }
    }
  } else if (iwarp == 3 && elected) {
    // =====================================================================
    // CLC warp (task dispatcher)
    // =====================================================================
    int phase_clc_read = 0;
    int phase_clc_write = 1;
    int istage_clc = 0;

    int igroup = 0;
    int sum_tile_n = 0;
    int itile_m, itile_n;

    while (true) {
      wait_barrier(task_writable[istage_clc], phase_clc_write);
      iblock += gridDim.x;
      get_next_tile_horizon<kMmaSM>(shm_tiles, iblock, num_group, igroup, itile_m, itile_n,
                                    sum_tile_n, flat_divider);
      task_shm[istage_clc][0] = igroup;
      task_shm[istage_clc][1] = itile_m;
      task_shm[istage_clc][2] = itile_n;
      task_shm[istage_clc][3] = igroup;
      arrive_barrier(task_readable[istage_clc]);

      if (igroup < 0) {
        break;
      }

      istage_clc++;
      if (istage_clc == kStageClc) {
        istage_clc = 0;
        phase_clc_write ^= 1;
        phase_clc_read ^= 1;
      }
    }
  } else if (idx >= 128 && idx < 256) {
    // =====================================================================
    // Epilogue warpgroup (idx 128..255): load gate+up from TMEM, compute
    // silu(gate)*up*scale, cast to fp8, STSM -> SMEM, TMA STORE -> GMEM.
    // =====================================================================
    idx -= 128;

    auto epi_tiler = make_tile(Int<kCtaTileM>{}, Int<kEpiTileN>{});
    // tCt in the existing kernel has shape (MMA_M, MMA_N); we reuse the same
    // partition for both gate and up accumulators (they have identical shapes).
    auto tCt_epi = zipped_divide(tCt, make_tile(epi_tiler));
    auto sC_epi = zipped_divide(sC, epi_tiler);
    auto sCT_epi = zipped_divide(sCT, epi_tiler);

    // TiledCopy TMEM -> RMEM. Same atom as the bf16-output path so the RMEM
    // fragment ordering matches the existing well-tested code path.
    auto tiled_copy_t2r = make_tmem_copy(SM100_TMEM_LOAD_16dp256b2x{}, tCt_epi(_, _0{}));
    auto thr_copy_t2r = tiled_copy_t2r.get_slice(idx);

    // Partition S on gate and up accumulators. The two tensors share the same
    // TMEM layout but point to different TMEM addresses; we rebind data()
    // per-stage below.
    auto tCt4r_gate = thr_copy_t2r.partition_S(tCt_epi);
    auto tCt4r_up = thr_copy_t2r.partition_S(tCt_epi);
    auto tCr_gate = make_tensor_like<float>(thr_copy_t2r.partition_D(sC_epi(_, 0)));
    auto tCr_up = make_tensor_like<float>(thr_copy_t2r.partition_D(sC_epi(_, 0)));
    auto tCr_out = make_tensor_like<Tout>(thr_copy_t2r.partition_D(sC_epi(_, 0)));

    // TiledCopy RMEM -> SMEM. Build a tiled_copy_d from the TMEM load tiled
    // copy so the destination partitioning matches the RMEM fragment ordering.
    // Use AutoVectorizingCopyWithAssumedAlignment for fp8 so each thread's
    // contiguous block of fp8 values writes directly to its matching SMEM
    // positions without needing an STSM swizzle transpose.
    auto tiled_copy_r2s = make_tiled_copy_D(
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<32>, Tout>{}, tiled_copy_t2r);
    auto thr_copy_r2s = tiled_copy_r2s.get_slice(idx);
    auto tCTs = thr_copy_r2s.partition_D(sCT_epi);

    auto nepi_tile = size<2>(tCt4r_gate);

    int phase_tile = 0;
    int phase_clc = 0;

    int istage_tile = 0;
    int istage_clc = 0;

    int igroup = 0;
    int sum_tile_n = 0;
    int itile_m, itile_n;

    int istage_tma = 0;

    bool is_leader = elected && (iwarp == 4);

    auto tCt4r_base_ptr = tCt4r_gate.data();

    get_next_tile_horizon<kMmaSM>(shm_tiles, iblock, num_group, igroup, itile_m, itile_n,
                                  sum_tile_n, flat_divider);

    while (true) {
      if (igroup >= 0) {
        // Gate accumulator at base + stage*kTmemColsPerStage.
        // Up   accumulator at base + stage*kTmemColsPerStage + kTileN.
        tCt4r_gate.data() = tCt4r_base_ptr + istage_tile * kTmemColsPerStage;
        tCt4r_up.data() = tCt4r_base_ptr + istage_tile * kTmemColsPerStage + kTileN;
        wait_barrier(tmem_readable[istage_tile], phase_tile);
        // Two scales are applied in the unfused pipeline:
        //   1) gate_up_scale_ptr[igroup] scales the GEMM accumulator before
        //      it is cast to bf16 (gate_up_output).
        //   2) act_mul_scale_ptr[0] multiplies silu(gate)*up before fp8 cast.
        // We replicate both so the fused kernel is numerically equivalent.
        float gu_scale = gate_up_scale_ptr[igroup];
        float am_scale = act_mul_scale_ptr[0];
#pragma unroll
        for (int iepi = 0; iepi < nepi_tile; iepi++) {
          // TMEM -> RMEM for gate, then up.
          copy(tiled_copy_t2r, tCt4r_gate(_, _, iepi), tCr_gate);
          copy(tiled_copy_t2r, tCt4r_up(_, _, iepi), tCr_up);
          cutlass::arch::fence_view_async_tmem_load();

          // Match the numerics of the unfused pipeline:
          //   gate_bf = bf16(gate_acc * gu_scale); up_bf = bf16(up_acc * gu_scale);
          //   out = bf16(silu(gate_f32)) * up_bf  (bf16 multiply path)
          //   out_fp32 = fp32(out) * am_scale;  fp8 = cast(out_fp32).
#pragma unroll
          for (int i = 0; i < cute::size(tCr_gate); i++) {
            float g = tCr_gate(i) * gu_scale;
            float u = tCr_up(i) * gu_scale;
            float silu_g = g * rcpf_ftz(1.f + expf_ftz(-g));
            __nv_bfloat16 silu_bf = __float2bfloat16_rn(silu_g);
            __nv_bfloat16 u_bf = __float2bfloat16_rn(u);
            float m = __bfloat162float(silu_bf * u_bf);
            float v = m * am_scale;
            tCr_out(i) = static_cast<Tout>(v);
          }

          tma_store_wait<kStageTMA - 1>();
          cutlass::arch::NamedBarrier::sync(128, 0);

          // RMEM -> SMEM via elementwise vectorized copy; the partition
          // produced by make_tiled_copy_D keeps the thread/value mapping of
          // tiled_copy_t2r, so element i of tCr_out lands at the same logical
          // (m,n) as it did in TMEM.
          copy(tiled_copy_r2s, tCr_out, tCTs(_, _, istage_tma));

          // SMEM -> GMEM (FP8 STORE)
          tma_store_fence();
          cutlass::arch::NamedBarrier::sync(128, 0);

          if (iwarp == 4 && elected) {
            auto gDT = tma_dt.get_tma_tensor(make_shape(m, n));
            auto btma_dt = tma_dt.get_slice(0);

            auto tDs = btma_dt.partition_S(sCT);
            auto tDg = btma_dt.partition_D(gDT);

            auto *td_y = td_xy + igroup * 2 + 1;
            cute::copy(tma_dt.with(td_y), tDs(_, 0, 0, istage_tma),
                       tDg(_, itile_m, itile_n * nepi_tile + iepi));
            tma_store_arrive();
          }

          istage_tma = (istage_tma + 1) % kStageTMA;
        }

        if (is_leader) {
          arrive_barrier(tmem_writable[istage_tile]);
        }

        istage_tile++;
        if (istage_tile == kStageTile) {
          phase_tile ^= 1;
          istage_tile = 0;
        }
      }

      wait_barrier(task_readable[istage_clc], phase_clc);
      igroup = task_shm[istage_clc][0];
      itile_m = task_shm[istage_clc][1];
      itile_n = task_shm[istage_clc][2];
      arrive_barrier(task_writable[istage_clc]);

      if (igroup < 0) {
        break;
      }

      istage_clc++;
      if (istage_clc == kStageClc) {
        istage_clc = 0;
        phase_clc ^= 1;
      }
    }
  }

  __syncthreads();
  if (iwarp == 1) {
    tmem_allocator.free(tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
  }

  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

}  // namespace kernels

}  // namespace group_gemm
}  // namespace hpc

#endif  // SRC_GROUP_GEMM_SM100_GROUP_GEMM_CP_ASYNC_FP8_ACT_MUL_CUH_
