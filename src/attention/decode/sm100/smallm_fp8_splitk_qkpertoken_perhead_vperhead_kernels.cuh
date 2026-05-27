// Copyright 2025 hpc-ops authors

#ifndef SRC_ATTENTION_DECODE_SM100_SMALLM_FP8_SPLITK_QKPERTOKEN_PERHEAD_VPERHEAD_KERNELS_CUH_
#define SRC_ATTENTION_DECODE_SM100_SMALLM_FP8_SPLITK_QKPERTOKEN_PERHEAD_VPERHEAD_KERNELS_CUH_

#include <cuda.h>
#include <stdio.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "src/attention/decode/sched_task_info.h"
#include "src/attention/decode/sm100/util_kernels.cuh"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace attention {
namespace kernels {

template <typename Tout, typename Tin, int kTileM, int kTileN, int kTileK, int kTileV,
          int kHeadsPerGroup, typename TiledMmaQK, typename TiledMmaSV, typename TmaQ,
          typename TmaK, typename TmaV, typename TmaY, typename TmaKS, typename SLayoutQ,
          typename SLayoutK, typename SLayoutP, typename SLayoutS, typename SLayoutV,
          typename SLayoutY, typename SLayoutKS, int kClusterM, int kClusterN, int kClusterK,
          int kMmaSM, int kBlockSize, int kStageQ, int kStageK, int kStageP>
__global__ void __launch_bounds__(512, 1)
    attention_decode_fp8_1sm_smallm_splitk_qkpertoken_perhead_vperhead_kernel(
        const __grid_constant__ TmaQ tma_q, const __grid_constant__ TmaK tma_k,
        const __grid_constant__ TmaV tma_v, const __grid_constant__ TmaY tma_y,
        const __grid_constant__ TmaKS tma_ks, float *lse_ptr, const int *task_map_ptr,
        const int *block_ids_ptr, const int *num_seq_kvcache_ptr, const float *qscale_ptr,
        const float *kscale_ptr, const float *vscale_ptr, bool new_kv_included, int num_batch,
        int num_seq_q, int num_dim_qk, int num_dim_v, int num_head_q, int num_head_k,
        int num_head_v, int heads_per_group, int lse_pad_heads_per_group, int num_kvcache_blocks,
        int num_seq_max_blocks, int qscale_pad_stride, float one_over_dk_log2e,
        cutlass::FastDivmod splitk_head_kv_divider) {
  using namespace cute;  // NOLINT
  constexpr float kDecodePScale = 256.0f;
  constexpr float kDecodePScaleInv = 1.0f / kDecodePScale;

  using TMEM_LOAD_ATOM = std::conditional_t<
      (kTileM == 8), SM100_TMEM_LOAD_16dp256b1x,
      std::conditional_t<(kTileM == 16), SM100_TMEM_LOAD_16dp256b2x, SM100_TMEM_LOAD_16dp256b4x>>;
  using TMEM_STORE_ATOM = std::conditional_t<
      (kTileM == 8), SM100_TMEM_STORE_16dp256b1x,
      std::conditional_t<(kTileM == 16), SM100_TMEM_STORE_16dp256b2x, SM100_TMEM_STORE_16dp256b4x>>;
  using SMEM_STORE_FP8_ATOM =
      std::conditional_t<(kTileM == 8), SM100_U8x4_STSM_T,
                         std::conditional_t<(kTileM == 16), SM100_U8x8_STSM_T, SM100_U8x16_STSM_T>>;

  // two part store q_stage and each contain 4 area. kTileM * 8.
  // 0. qk output. 1. exp input p. 2. sv input s. 3. sv output. kTileM * 4
  constexpr int kTmemColumns = 512;
  constexpr int kClusterSize = kClusterM * kClusterN * kClusterK;
  constexpr int kMaxSplitK = 64;

  int idx = threadIdx.x;
  int iwarp = idx / 32;
  int ilane = idx % 32;
  int elected = cute::elect_one_sync();
  int iblock = blockIdx.y;

  __shared__ float smem_max[kTileM * 4];
  __shared__ float smem_sum[2][kStageQ][kTileM * 4];
  __shared__ float smem_correct_scale[kStageP][kTileM];
  __shared__ TaskInfo smem_task[kStageQ];

  __shared__ uint64_t task_readable[kStageQ];
  __shared__ uint64_t task_writable[kStageQ];

  __shared__ uint64_t q_readable[kStageQ];
  __shared__ uint64_t q_writable[kStageQ];

  __shared__ uint64_t k_readable[kStageK];
  __shared__ uint64_t k_writable[kStageK];

  __shared__ uint64_t v_readable[kStageK];
  __shared__ uint64_t v_writable[kStageK];

  __shared__ uint64_t p_readable[kStageP];
  __shared__ uint64_t p_writable[kStageP];

  __shared__ uint64_t s_readable[kStageP];
  __shared__ uint64_t s_writable[kStageP];

  __shared__ uint64_t exp_readable[kStageP];
  __shared__ uint64_t exp_writable[kStageP];

  __shared__ uint64_t y_readable[kStageQ];
  __shared__ uint64_t y_writable[kStageQ];
  __shared__ uint32_t y_phase[kStageQ];

  __shared__ uint32_t tmem_base_ptr;

  extern __shared__ uint8_t shm_data[] alignas(128);

  auto *shm_q = reinterpret_cast<Tin *>(shm_data);
  auto *shm_k = reinterpret_cast<Tin *>(shm_q + cosize(SLayoutQ{}));
  auto *shm_v = reinterpret_cast<Tin *>(shm_k + cosize(SLayoutK{}));
  auto *shm_p = reinterpret_cast<Tin *>(shm_v + cosize(SLayoutV{}));
  auto *shm_s = shm_p;
  auto *shm_ks = reinterpret_cast<float *>(shm_p + cosize(SLayoutP{}));
  auto *shm_y = reinterpret_cast<Tout *>(shm_ks + cosize(SLayoutKS{}));
  auto *shm_blkids = reinterpret_cast<int *>(shm_y + cosize(SLayoutY{}));

  constexpr int kScaleByteSize = sizeof(float);
  const int num_scale_per_row = num_dim_qk / kScaleByteSize;

  auto gQ = tma_q.get_tma_tensor(
      make_shape(heads_per_group, num_dim_qk, num_head_k, num_seq_q, num_batch));
  auto gK =
      tma_k.get_tma_tensor(make_shape(kBlockSize, num_dim_qk, num_head_k, num_kvcache_blocks));
  auto gV = tma_v.get_tma_tensor(make_shape(num_dim_v, kBlockSize, num_head_v, num_kvcache_blocks));

  auto gAtt =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileN>{}, Int<kTileM>{}), make_stride(Int<kTileM>{}, Int<1>{}));
  auto gY =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileV>{}, Int<kTileM>{}), make_stride(Int<kTileM>{}, Int<1>{}));
  auto gKS = tma_ks.get_tma_tensor(make_shape(kBlockSize / num_scale_per_row, num_scale_per_row,
                                              num_head_k, num_kvcache_blocks));

  // ((_8,_2),(_128,_1),(_1,_2)):((_128,_1024),(_1,_0),(_0,_2048))
  auto sQ = make_tensor(make_smem_ptr(shm_q), SLayoutQ{});
  // ((_8,_16),(_128,_1),(_1,_4)):((_128,_1024),(_1,_0),(_0,_16384))
  auto sK = make_tensor(make_smem_ptr(shm_k), SLayoutK{});
  // ((_128,_1),(_8,_16),(_1,_4)):((_1,_0),(_128,_1024),(_0,_16384))
  auto sV = make_tensor(make_smem_ptr(shm_v), SLayoutV{});
  // ((_128,_1),(_8,_2)):((_1,_0),(_128,_1024))
  auto sP = make_tensor(make_smem_ptr(shm_p), SLayoutP{});
  auto sS = make_tensor(make_smem_ptr(shm_s), SLayoutS{});
  // ((_64,_2),(_8,_2)):((_1,_512),(_64,_1024))
  auto sY = make_tensor(make_smem_ptr(shm_y), SLayoutY{});
  auto sKS = make_tensor(make_smem_ptr(shm_ks), SLayoutKS{});

  auto cluster_layout =
      tiled_divide(make_layout(make_shape(Int<kClusterM>{}, Int<kClusterN>{}, Int<kClusterK>{})),
                   make_tile(Int<kMmaSM>{}, Int<1>{}, Int<1>{}));
  int block_rank_in_cluster = cute::block_rank_in_cluster();
  auto cluster_coord = cluster_layout.get_flat_coord(block_rank_in_cluster);

  using TmemAllocator = TMEM::Allocator1Sm;
  TmemAllocator tmem_allocator{};

  constexpr int kTmaThreads = 1;
  constexpr int kMmaThreads = 32;
  constexpr int kExpThreads = 128;
  constexpr int kRescaleThreads = 128;
  constexpr int kEpiThreads = 128;

  if (iwarp == 0) {
    tmem_allocator.allocate(kTmemColumns, &tmem_base_ptr);
  } else if (iwarp == 1 && elected) {
#pragma unroll
    for (int i = 0; i < kStageQ; i++) {
      initialize_barrier(q_readable[i], 1);
      initialize_barrier(q_writable[i], 1);
    }

    cutlass::arch::fence_barrier_init();
  } else if (iwarp == 2 && elected) {
#pragma unroll
    for (int i = 0; i < kStageQ; i++) {
      initialize_barrier(y_readable[i], 1);
      initialize_barrier(y_writable[i], kEpiThreads);
      y_phase[i] = 1;
    }
    cutlass::arch::fence_barrier_init();
  } else if (iwarp == 5 && elected) {
#pragma unroll
    for (int i = 0; i < kStageK; i++) {
      initialize_barrier(k_readable[i], 1);
      initialize_barrier(k_writable[i], 1 + 128);
    }
    cutlass::arch::fence_barrier_init();
  } else if (iwarp == 6 && elected) {
#pragma unroll
    for (int i = 0; i < kStageK; i++) {
      initialize_barrier(v_readable[i], 1);
      initialize_barrier(v_writable[i], 1);
    }
    cutlass::arch::fence_barrier_init();
  } else if (iwarp == 7 && elected) {
#pragma unroll
    for (int i = 0; i < kStageP; i++) {
      initialize_barrier(p_readable[i], 1);
      initialize_barrier(p_writable[i], kRescaleThreads);
    }
    cutlass::arch::fence_barrier_init();
  } else if (iwarp == 8 && elected) {
#pragma unroll
    for (int i = 0; i < kStageP; i++) {
      initialize_barrier(s_readable[i], kRescaleThreads);
      initialize_barrier(s_writable[i], 1);
    }
    cutlass::arch::fence_barrier_init();
  } else if (iwarp == 9 && elected) {
#pragma unroll
    for (int i = 0; i < kStageP; i++) {
      initialize_barrier(exp_readable[i], kExpThreads);
      initialize_barrier(exp_writable[i], kRescaleThreads);
    }
    cutlass::arch::fence_barrier_init();
  } else if (iwarp == 10 && elected) {
#pragma unroll
    for (int i = 0; i < kStageQ; i++) {
      initialize_barrier(task_readable[i], 1);
      initialize_barrier(task_writable[i], kClusterSize * (kTmaThreads + kMmaThreads + kExpThreads +
                                                           kRescaleThreads + kEpiThreads));
    }
    cutlass::arch::fence_barrier_init();
  }

  constexpr int kSchedTaskInfoSize = sizeof(TaskScheduleInfo) / sizeof(int);
  int num_tiles_per_sm = task_map_ptr[0];
  const int *task_map_num_chunks_ptr =
      task_map_ptr + (gridDim.y * num_tiles_per_sm + 1) * kSchedTaskInfoSize;
  task_map_ptr += (1 + iblock * num_tiles_per_sm) * kSchedTaskInfoSize;
  cluster_relaxed_sync();

  cudaGridDependencySynchronize();

  TaskInfo task_info;
  task_info.ihead_kv = 0;

  TiledMmaQK tiled_mma_qk;
  TiledMmaSV tiled_mma_sv;

  auto cta_mma_qk = tiled_mma_qk.get_slice(0);
  auto cta_mma_sv = tiled_mma_sv.get_slice(0);

  auto tKs4r = cta_mma_qk.partition_A(sK);
  auto tQs4r = cta_mma_qk.partition_B(sQ);
  auto tPgP = cta_mma_qk.partition_C(gAtt);

  // (_1,_1,_4,(_1,_4)):(_0,_0,_2,(_0,_1024))
  auto tKr = cta_mma_qk.make_fragment_A(tKs4r);
  // (_1,_1,_4,(_1,_2)):(_0,_0,_2,(_0,_128))
  auto tQr = cta_mma_qk.make_fragment_B(tQs4r);
  // ((_128,_16),_1,_1):((_65536,_1),_0,_0)
  auto tPt = cta_mma_qk.make_fragment_C(tPgP);

  auto tVs4r = cta_mma_sv.partition_A(sV);
  auto tSs4r = cta_mma_sv.partition_B(sS);
  auto tYgY = cta_mma_sv.partition_C(gY);

  // (_1,_1,_4,(_1,_4)):(_0,_0,_256,(_0,_1024))
  auto tVr = cta_mma_sv.make_fragment_A(tVs4r);
  // (_1,_1,_4):(_0,_0,_2)
  auto tSr = cta_mma_sv.make_fragment_B(tSs4r);
  // ((_128,_16),_1,_1):((_65536,_1),_0,_0)
  auto tYt = cta_mma_sv.make_fragment_C(tYgY);

  uint32_t tmem_p_base_ptr = tmem_base_ptr;
  uint32_t tmem_y_base_ptr = tmem_base_ptr + kStageP * kTileM;

  if (iwarp == 0 && elected) {
    // TMA : issuse Q K1 K2 V1 K3 V2 K4 ... Vn-2 Kn Vn-1 Vn Q K1 K2
    cutlass::arch::warpgroup_reg_dealloc<56>();
    auto btma_q = tma_q.get_slice(0);
    auto btma_k = tma_k.get_slice(0);
    auto btma_v = tma_v.get_slice(0);
    auto btma_ks = tma_ks.get_slice(0);

    // (((_128,_8),_1),1,1,1,1,1):(((_1@0,_1@1),_0),_8@1,_128@0,_1@2,_1@3,_1@4)
    auto tQg = btma_q.partition_S(gQ);
    // (((_128,_64),_1),1,1,1,76):(((_1@0,_1@1),_0),_64@1,_128@0,_1@2,_1@3)
    auto tKg = btma_k.partition_S(gK);
    // (((_128,_64),_1),1,1,1,76):(((_1@0,_1@1),_0),_128@0,_64@1,_1@2,_1@3)
    auto tVg = btma_v.partition_S(gV);
    auto tKSg = btma_ks.partition_S(gKS);

    // (((_128,_8),_1),_2,_1,(_1,_2)):(((_1,_128),_0),_1024,_0,(_0,_2048))
    auto tQs = btma_q.partition_D(sQ);
    // (((_128,_64),_1),_2,_1,(_1,_4)):(((_1,_128),_0),_8192,_0,(_0,_16384))
    auto tKs = btma_k.partition_D(sK);
    // ((_8192,_1),_1,_2,(_1,_4)):((_1,_0),_0,_8192,(_0,_16384))
    auto tVs = btma_v.partition_D(sV);
    auto tKSs = btma_ks.partition_S(sKS);

    int phase_q = 1;
    int istage_q = 0;
    int phase_k = 1;
    int istage_k = 0;
    int phase_v = 1;
    int istage_v = 0;

    task_info.valid =
        get_task_info<kTileN, kBlockSize>(task_info, task_map_ptr, task_map_num_chunks_ptr, iblock,
                                          splitk_head_kv_divider, num_tiles_per_sm, num_seq_q);
    while (true) {
      if (!task_info.valid) {
        break;
      }

      load_q(tma_q, q_readable, q_writable, tQg, tQs, task_info.ihead_kv, task_info.ibatch,
             num_seq_q, kHeadsPerGroup * num_seq_q * num_dim_qk * sizeof(Tin), istage_q, phase_q);

      // auto *block_ids = shm_blkids + istage_q * num_seq_max_blocks;
      auto *block_ids =
          block_ids_ptr + task_info.ibatch * num_seq_max_blocks + task_info.num_blocks_per_chunk;

      int itile_seq_k = task_info.num_tile_kv - 1;
      // load Kn
      load_paged_k_with_scale<kTileN, kBlockSize, kStageK, Tin>(
          tma_k, tma_ks, k_readable, k_writable, tKg, tKs, tKSg, tKSs, task_info.ihead_kv,
          num_dim_qk, block_ids, task_info.num_blocks, itile_seq_k, istage_k, phase_k);
      itile_seq_k--;

#pragma unroll 1
      for (; itile_seq_k >= 0; --itile_seq_k) {
        // load Ki
        load_paged_k_with_scale<kTileN, kBlockSize, kStageK, Tin>(
            tma_k, tma_ks, k_readable, k_writable, tKg, tKs, tKSg, tKSs, task_info.ihead_kv,
            num_dim_qk, block_ids, task_info.num_blocks, itile_seq_k, istage_k, phase_k);

        // load Vi+1
        load_paged_v<kTileN, kBlockSize, kStageK, Tin>(
            tma_v, v_readable, v_writable, tVg, tVs, task_info.ihead_kv, num_dim_v, block_ids,
            task_info.num_blocks, itile_seq_k + 1, istage_v, phase_v);
      }

      // load V0
      load_paged_v<kTileN, kBlockSize, kStageK, Tin>(tma_v, v_readable, v_writable, tVg, tVs,
                                                     task_info.ihead_kv, num_dim_v, block_ids,
                                                     task_info.num_blocks, 0, istage_v, phase_v);

      wait_barrier(task_readable[istage_q], !phase_q);
      TaskInfo::load(&smem_task[istage_q], task_info);
      arrive_barrier(task_writable[istage_q]);
      advance_stage<kStageQ>(istage_q, phase_q);
    }
  } else if (iwarp == 1) {
    // MMA : issuse QK1 QK2 P1V1 QK3 P2V2 QK4 ... Pn-2Vn-2 QKn  Pn-1Vn-1 PnVn
    int phase_q = 0;
    int istage_q = 0;
    int phase_k = 0;
    int istage_k = 0;
    int phase_v = 0;
    int istage_v = 0;
    int phase_p = 1;
    int istage_p = 0;
    int phase_s = 0;
    int istage_s = 0;

    task_info.valid =
        get_task_info<kTileN, kBlockSize>(task_info, task_map_ptr, task_map_num_chunks_ptr, iblock,
                                          splitk_head_kv_divider, num_tiles_per_sm, num_seq_q);

    while (true) {
      if (!task_info.valid) {
        break;
      }

      wait_barrier(q_readable[istage_q], phase_q);

      int itile_seq_k = task_info.num_tile_kv - 1;

      tiled_mma_sv.accumulate_ = UMMA::ScaleOut::Zero;

      // QKn
      wait_barrier(k_readable[istage_k], phase_k);
      wait_barrier(p_writable[istage_p], phase_p);
      qk_gemm<kTileM>(tiled_mma_qk, tQr, tKr, tPt, tmem_p_base_ptr, istage_q, istage_k, istage_p);
      cutlass::arch::umma_arrive(&k_writable[istage_k]);
      cutlass::arch::umma_arrive(&p_readable[istage_p]);
      advance_stage<kStageK>(istage_k, phase_k);
      advance_stage<kStageP>(istage_p, phase_p);
      itile_seq_k--;

      wait_barrier(y_writable[istage_q], !phase_q);
      tYt.data() = tmem_y_base_ptr + istage_q * kTileM;
#pragma unroll 1
      for (; itile_seq_k >= 0; --itile_seq_k) {
        // QKi
        wait_barrier(k_readable[istage_k], phase_k);
        wait_barrier(p_writable[istage_p], phase_p);
        qk_gemm<kTileM>(tiled_mma_qk, tQr, tKr, tPt, tmem_p_base_ptr, istage_q, istage_k, istage_p);
        cutlass::arch::umma_arrive(&k_writable[istage_k]);
        cutlass::arch::umma_arrive(&p_readable[istage_p]);
        advance_stage<kStageK>(istage_k, phase_k);
        advance_stage<kStageP>(istage_p, phase_p);

        // SVi+1
        wait_barrier(v_readable[istage_v], phase_v);
        wait_barrier(s_readable[istage_s], phase_s);
        sv_gemm(tiled_mma_sv, tVr, tSr, tYt, istage_v, istage_s);
        cutlass::arch::umma_arrive(&v_writable[istage_v]);
        cutlass::arch::umma_arrive(&s_writable[istage_s]);
        cutlass::arch::umma_arrive(&y_readable[istage_q]);
        advance_stage<kStageK>(istage_v, phase_v);
        advance_stage<kStageP>(istage_s, phase_s);
      }

      // SV0
      wait_barrier(v_readable[istage_v], phase_v);
      wait_barrier(s_readable[istage_s], phase_s);
      sv_gemm(tiled_mma_sv, tVr, tSr, tYt, istage_v, istage_s);
      cutlass::arch::umma_arrive(&v_writable[istage_v]);
      cutlass::arch::umma_arrive(&s_writable[istage_s]);
      cutlass::arch::umma_arrive(&y_readable[istage_q]);
      advance_stage<kStageK>(istage_v, phase_v);
      advance_stage<kStageP>(istage_s, phase_s);

      cutlass::arch::umma_arrive(&q_writable[istage_q]);

      wait_barrier(task_readable[istage_q], phase_q);
      TaskInfo::load(&smem_task[istage_q], task_info);
      arrive_barrier(task_writable[istage_q]);
      advance_stage<kStageQ>(istage_q, phase_q);
    }
  } else if (iwarp == 2 && (block_rank_in_cluster == 0) && elected) {
    // CLC: find next Q
    int phase_q = 1;
    int istage_q = 0;
    bool has_next_task = (task_map_ptr[0] >= 0);
    task_info.iblock = iblock;

    while (has_next_task) {
      task_map_ptr += kSchedTaskInfoSize;
      task_info.valid = get_task_info<kTileN, kBlockSize>(
          task_info, task_map_ptr, task_map_num_chunks_ptr, iblock, splitk_head_kv_divider,
          num_tiles_per_sm, num_seq_q);

      has_next_task = task_info.valid;

      wait_barrier(task_writable[istage_q], phase_q);
      TaskInfo::store(task_info, &smem_task[istage_q]);
      arrive_barrier(task_readable[istage_q]);
      advance_stage<kStageQ>(istage_q, phase_q);
    }
  } else if (idx >= 128 && idx < 256) {
    // Exp: tP -> rP -> atmoicmax -> exp -> stsm P -> sttm sum
    cutlass::arch::warpgroup_reg_alloc<184>();
    idx -= 128;
    iwarp = idx / 32;
    constexpr int kBarId = 0;
    int phase_q = 1;
    int istage_q = 0;
    int phase_p = 0;
    int istage_p = 0;
    int phase_exp = 1;
    int istage_exp = 0;
    int istage_k = 0;
    int phase_k = 0;

    auto tmem_exp_tiler = make_tile(Int<kTileN>{}, Int<kTileM>{});
    auto tPt_exp = zipped_divide(tPt, make_tile(tmem_exp_tiler));
    auto sP_exp = zipped_divide(gAtt, tmem_exp_tiler);

    auto tiled_copy_t2r = make_tmem_copy(TMEM_LOAD_ATOM{}, tPt_exp(_, 0));
    auto thr_copy_t2r = tiled_copy_t2r.get_slice(idx);
    // (((_16,_16),_1),_2,((_1,_1),_1,_1)):(((_1,_65536),_0),_1048576,((_0,_0),_0,_0))
    auto tPt4r = thr_copy_t2r.partition_S(tPt_exp);
    // (((_2,_2,_2),_1),_2,(_1,_1)):(((_4,_1,_8),_0),_2,(_0,_0))
    auto tPr4t = make_tensor_like<float>(thr_copy_t2r.partition_D(sP_exp));

    auto tiled_copy_r2t = make_tmem_copy(TMEM_STORE_ATOM{}, tPt_exp(_, 0));
    auto thr_copy_r2t = tiled_copy_r2t.get_slice(idx);
    auto &tSr4t = tPr4t;
    // (((_16,_16),_1),_2,((_1,_1),_1,_1)):(((_1,_65536),_0),_1048576,((_0,_0),_0,_0))
    auto tSt4r = thr_copy_r2t.partition_D(tPt_exp);

    auto tPt4r_base_ptr = tPt4r.data();
    auto tSt4r_base_ptr = tSt4r.data();

    auto gI = make_identity_tensor(gAtt.shape());
    auto gI_exp = zipped_divide(gI, tmem_exp_tiler);
    auto tI = thr_copy_t2r.partition_D(gI_exp);
    auto tI_nm = retile_fragment(tI);
    constexpr int kN = size<0>(tI_nm);
    constexpr int kM = size<1>(tI_nm);
    Tensor qscales = make_tensor<float>(Int<kM>{});
    Tensor kscales = make_tensor<float>(Int<kN>{});
    vec_t<float, kM> gMax;
    vec_t<float, kM> gSoftmaxScale;
    vec_t<float, kM> gSum;
    Tensor gCorrectScale = make_tensor<float>(Int<kM>{});

    auto sKS_flatten =
        make_tensor(sKS.data(), make_layout(make_shape(Int<kTileN>{}, Int<kStageK>{})));

    task_info.valid =
        get_task_info<kTileN, kBlockSize>(task_info, task_map_ptr, task_map_num_chunks_ptr, iblock,
                                          splitk_head_kv_divider, num_tiles_per_sm, num_seq_q);

    while (true) {
      if (!task_info.valid) {
        break;
      }

      auto *qscales_batch = qscale_ptr + task_info.ibatch * num_head_q * num_seq_q +
                            task_info.ihead_kv * heads_per_group;
#pragma unroll
      for (int i = 0; i < kM; i++) {
        int im = get<1>(tI_nm(0, i));
        int iseqq = im / kHeadsPerGroup;
        int iqhead = im % kHeadsPerGroup;
        if (iqhead < heads_per_group) {
          qscales(i) = qscales_batch[iseqq * num_head_q + iqhead];
        } else {
          qscales(i) = 1;
        }
      }

#pragma unroll
      for (int i = 0; i < kM; i++) {
        gSoftmaxScale[i] = one_over_dk_log2e;
        gSum[i] = 0;
        gMax[i] = -std::numeric_limits<float>::infinity();
      }

#pragma unroll 1
      for (int itile_seq_k = task_info.num_tile_kv - 1; itile_seq_k >= task_info.num_tile_full;
           --itile_seq_k) {
        tPt4r.data() = tPt4r_base_ptr + istage_p * kTileM;
        tSt4r.data() = tSt4r_base_ptr + istage_exp * kTileM;

        wait_barrier(k_readable[istage_k], phase_k);
#pragma unroll
        for (int in = 0; in < kN; in++) {
          kscales(in) = sKS_flatten(get<0>(tI_nm(in, 0)), istage_k);
        }
        arrive_barrier(k_writable[istage_k]);
        advance_stage<kStageK>(istage_k, phase_k);

        // P from tmem to reg
        wait_barrier(p_readable[istage_p], phase_p);
        fence_tmem_after_thread_sync();
        copy(tiled_copy_t2r, tPt4r, tPr4t);
        cutlass::arch::fence_view_async_tmem_load();
        // get max and do exp
        auto tAttr_nm = retile_fragment(tPr4t);
        apply_casual_mask_with_scale<kTileN, kHeadsPerGroup>(tAttr_nm, tI_nm, qscales, kscales,
                                                             itile_seq_k, task_info.num_seq_kvcache,
                                                             task_info.num_seq_kv);

        online_softmax<true, kTileM>(tAttr_nm, gMax, gSum, gSoftmaxScale, smem_max,
                                     smem_correct_scale[istage_exp], kBarId, iwarp, ilane);

        // Scale P before FP8 quantization; epilogue applies the reciprocal.
#pragma unroll
        for (int im = 0; im < kM; ++im) {
#pragma unroll
          for (int in = 0; in < kN; ++in) {
            tAttr_nm(in, im) *= kDecodePScale;
          }
        }

        if (task_info.num_tile_full == 0) {
#pragma unroll
          for (int im = 0; im < kM; ++im) {
            gSum[im] = warp_8lane_stride4_reduce_sum_xor(gSum[im]);
          }
          wait_barrier(y_writable[istage_q], phase_q);
          if (ilane < 4) {
            store_rC_to_smem<kTileM, kM>(gSum, smem_sum[0][istage_q], iwarp, ilane);
            if (iwarp == 0) {
              store_rC_to_smem<kTileM, kM>(gMax, smem_sum[1][istage_q], 0, ilane);
            }
          }
          __syncwarp();
        }
        // P from reg to tmem
        wait_barrier(exp_writable[istage_exp], phase_exp);
        copy(tiled_copy_r2t, tSr4t, tSt4r);
        cutlass::arch::fence_view_async_tmem_store();
        fence_tmem_before_thread_sync();

        cutlass::arch::fence_view_async_shared();
        arrive_barrier(exp_readable[istage_exp]);
        advance_stage<kStageP>(istage_p, phase_p);
        advance_stage<kStageP>(istage_exp, phase_exp);
      }

#pragma unroll 1
      for (int itile_seq_k = task_info.num_tile_full - 1; itile_seq_k >= 1; --itile_seq_k) {
        tPt4r.data() = tPt4r_base_ptr + istage_p * kTileM;
        tSt4r.data() = tSt4r_base_ptr + istage_exp * kTileM;

        wait_barrier(k_readable[istage_k], phase_k);
#pragma unroll
        for (int in = 0; in < kN; in++) {
          kscales(in) = sKS_flatten(get<0>(tI_nm(in, 0)), istage_k);
        }
        arrive_barrier(k_writable[istage_k]);
        advance_stage<kStageK>(istage_k, phase_k);

        // P from tmem to reg
        wait_barrier(p_readable[istage_p], phase_p);
        fence_tmem_after_thread_sync();
        copy(tiled_copy_t2r, tPt4r, tPr4t);
        cutlass::arch::fence_view_async_tmem_load();
        // get max and do exp
        auto tAttr_nm = retile_fragment(tPr4t);
#pragma unroll
        for (int im = 0; im < kM; ++im) {
#pragma unroll
          for (int in = 0; in < kN; ++in) {
            tAttr_nm(in, im) *= qscales(im) * kscales(in);
          }
        }
        online_softmax<false, kTileM>(tAttr_nm, gMax, gSum, gSoftmaxScale, smem_max,
                                      smem_correct_scale[istage_exp], kBarId, iwarp, ilane);

        // Scale P before FP8 quantization; epilogue applies the reciprocal.
#pragma unroll
        for (int im = 0; im < kM; ++im) {
#pragma unroll
          for (int in = 0; in < kN; ++in) {
            tAttr_nm(in, im) *= kDecodePScale;
          }
        }

        // P from reg to tmem
        wait_barrier(exp_writable[istage_exp], phase_exp);
        copy(tiled_copy_r2t, tSr4t, tSt4r);
        cutlass::arch::fence_view_async_tmem_store();
        fence_tmem_before_thread_sync();

        cutlass::arch::fence_view_async_shared();
        arrive_barrier(exp_readable[istage_exp]);
        advance_stage<kStageP>(istage_p, phase_p);
        advance_stage<kStageP>(istage_exp, phase_exp);
      }

      if (task_info.num_tile_full > 0) {
        tPt4r.data() = tPt4r_base_ptr + istage_p * kTileM;
        tSt4r.data() = tSt4r_base_ptr + istage_exp * kTileM;

        wait_barrier(k_readable[istage_k], phase_k);
#pragma unroll
        for (int in = 0; in < kN; in++) {
          kscales(in) = sKS_flatten(get<0>(tI_nm(in, 0)), istage_k);
        }
        arrive_barrier(k_writable[istage_k]);
        advance_stage<kStageK>(istage_k, phase_k);

        // P from tmem to reg
        wait_barrier(p_readable[istage_p], phase_p);
        fence_tmem_after_thread_sync();
        copy(tiled_copy_t2r, tPt4r, tPr4t);
        cutlass::arch::fence_view_async_tmem_load();
        // get max and do exp
        auto tAttr_nm = retile_fragment(tPr4t);
#pragma unroll
        for (int im = 0; im < kM; ++im) {
#pragma unroll
          for (int in = 0; in < kN; ++in) {
            tAttr_nm(in, im) *= qscales(im) * kscales(in);
          }
        }
        online_softmax<false, kTileM>(tAttr_nm, gMax, gSum, gSoftmaxScale, smem_max,
                                      smem_correct_scale[istage_exp], kBarId, iwarp, ilane);

        // Scale P before FP8 quantization; epilogue applies the reciprocal.
#pragma unroll
        for (int im = 0; im < kM; ++im) {
#pragma unroll
          for (int in = 0; in < kN; ++in) {
            tAttr_nm(in, im) *= kDecodePScale;
          }
        }

        // store gsum to smem_sum
#pragma unroll
        for (int im = 0; im < kM; ++im) {
          gSum[im] = warp_8lane_stride4_reduce_sum_xor(gSum[im]);
        }
        wait_barrier(y_writable[istage_q], phase_q);
        if (ilane < 4) {
          store_rC_to_smem<kTileM, kM>(gSum, smem_sum[0][istage_q], iwarp, ilane);
          if (iwarp == 0) {
            store_rC_to_smem<kTileM, kM>(gMax, smem_sum[1][istage_q], 0, ilane);
          }
        }
        __syncwarp();

        // P from reg to tmem
        wait_barrier(exp_writable[istage_exp], phase_exp);
        copy(tiled_copy_r2t, tSr4t, tSt4r);
        cutlass::arch::fence_view_async_tmem_store();
        fence_tmem_before_thread_sync();

        cutlass::arch::fence_view_async_shared();
        arrive_barrier(exp_readable[istage_exp]);
        advance_stage<kStageP>(istage_p, phase_p);
        advance_stage<kStageP>(istage_exp, phase_exp);
      }

      wait_barrier(task_readable[istage_q], !phase_q);
      TaskInfo::load(&smem_task[istage_q], task_info);
      arrive_barrier(task_writable[istage_q]);
      advance_stage<kStageQ>(istage_q, phase_q);
    }
  } else if (idx >= 256 && idx < 384) {
    // Rescale: tSum -> rSum -> rescale
    cutlass::arch::warpgroup_reg_dealloc<88>();
    idx -= 256;

    constexpr int kBarId = 1;
    int phase_q = 0;
    int istage_q = 0;
    int phase_p = 0;
    int istage_p = 0;
    int phase_s = 1;
    int istage_s = 0;
    int phase_exp = 0;
    int istage_exp = 0;

    tYt.data() = tmem_y_base_ptr;
    auto tmem_exp_tiler = make_tile(Int<kTileN>{}, Int<kTileM>{});
    auto tmem_epi_tiler = make_tile(Int<kTileV>{}, Int<kTileM>{});
    // (((_128,_16)),((_1,_1),_1,_1)):(((_65536,_1)),((_0,_0),_0,_0))
    auto tPt_exp = zipped_divide(tPt, make_tile(tmem_exp_tiler));
    // ((_128,_16),(_1,_1)):((_16,_1),(_0,_0))
    auto rP_exp = zipped_divide(gAtt, tmem_exp_tiler);
    // ((_128,_16),(_1,_2)):((_1,_128),(_0,_2048))
    auto sP_exp = tiled_divide(sP, tmem_exp_tiler)(_, 0, 0, _);
    // (((_128,_16)),((_1,_1),_1,_1)):(((_65536,_1)),((_0,_0),_0,_0))
    auto tYt_epi = zipped_divide(tYt, make_tile(tmem_epi_tiler));
    // ((_128,_16),(_1,_1)):((_1,_128),(_0,_0))
    auto sY_epi = zipped_divide(gY, tmem_epi_tiler);

    auto tiled_copy_P_t2r = make_tmem_copy(TMEM_LOAD_ATOM{}, tPt_exp(_, 0));
    auto thr_copy_P_t2r = tiled_copy_P_t2r.get_slice(idx);
    // (((_16,_16),_1),_2,((_1,_1),_1,_1)):(((_1,_65536),_0),_1048576,((_0,_0),_0,_0))
    auto tPt4r = thr_copy_P_t2r.partition_S(tPt_exp);
    // (((_2,_2,_2),_1),_2,(_1,_1)):(((_1,_4,_2),_0),_8,(_0,_0))
    auto tPr4t = make_tensor_like<float>(thr_copy_P_t2r.partition_D(rP_exp));

    auto tiled_copy_Y_t2r = make_tmem_copy(TMEM_LOAD_ATOM{}, tYt_epi(_, 0));
    auto tiled_copy_Y_r2t = make_tmem_copy(TMEM_STORE_ATOM{}, tYt_epi(_, 0));
    auto thr_copy_Y_t2r = tiled_copy_Y_t2r.get_slice(idx);
    // (((_16,_16),_1),_2,((_1,_1),_1,_1)):(((_1,_65536),_0),_1048576,((_0,_0),_0,_0))
    auto tYt4r = thr_copy_Y_t2r.partition_S(tYt_epi);
    // (((_2,_2,_2),_1),_2,(_1,_1)):(((_4,_1,_8),_0),_2,(_0,_0))
    auto tYr4t = make_tensor_like<float>(thr_copy_Y_t2r.partition_D(sY_epi));

    auto tiled_copy_S_r2s =
        make_tiled_copy_D(Copy_Atom<SMEM_STORE_FP8_ATOM, Tin>{}, tiled_copy_P_t2r);
    auto thr_copy_S_r2s = tiled_copy_S_r2s.get_slice(idx);
    // ((_16,_1),_2,(_1,_2)):((_1,_0),_16,(_0,_2048))
    auto tSs4r = thr_copy_S_r2s.partition_D(sP_exp);
    // (((_2,_2,_2),_1),_2,(_1,_1)):(((_1,_4,_2),_0),_8,(_0,_0))
    auto tSr4s = make_tensor_like<Tin>(thr_copy_S_r2s.partition_S(rP_exp));

    auto tPt4r_base_ptr = tPt4r.data();
    auto tYt4r_base_ptr = tYt4r.data();

    auto tYr_nm = retile_fragment(tYr4t);
    constexpr int kM = size<1>(tYr_nm);

    task_info.valid =
        get_task_info<kTileN, kBlockSize>(task_info, task_map_ptr, task_map_num_chunks_ptr, iblock,
                                          splitk_head_kv_divider, num_tiles_per_sm, num_seq_q);

    while (true) {
      if (!task_info.valid) {
        break;
      }
      int phase_y = y_phase[istage_q];
      tYt4r.data() = tYt4r_base_ptr + istage_q * kTileM;

      // first tile don't need rescale
      {
        tPt4r.data() = tPt4r_base_ptr + istage_exp * kTileM;
        wait_barrier(exp_readable[istage_exp], phase_exp);
        // tmem -> reg
        fence_tmem_after_thread_sync();
        copy(tiled_copy_P_t2r, tPt4r, tPr4t);
        cutlass::arch::fence_view_async_tmem_load();

        arrive_barrier(exp_writable[istage_exp]);
        arrive_barrier(p_writable[istage_p]);
        // cast rS to Tin
        cast<Tin, kTileM>(tPr4t, tSr4s);
        wait_barrier(s_writable[istage_s], phase_s);
        wait_barrier(y_readable[istage_q], phase_y);

        // stsm p -> s, ldtm y, sttm y
        copy(tiled_copy_S_r2s, tSr4s(_, _, 0), tSs4r(_, _, istage_s));
        cutlass::arch::fence_view_async_tmem_load();
        copy(tiled_copy_Y_r2t, tYr4t, tYt4r);
        cutlass::arch::fence_view_async_tmem_store();
        cutlass::arch::fence_view_async_shared();
        fence_tmem_before_thread_sync();
        cutlass::arch::NamedBarrier::sync(128, kBarId);

        arrive_barrier(s_readable[istage_s]);
        phase_y ^= 1;
        advance_stage<kStageP>(istage_p, phase_p);
        advance_stage<kStageP>(istage_s, phase_s);
        advance_stage<kStageP>(istage_exp, phase_exp);
      }

#pragma unroll 1
      for (int itile_seq_k = task_info.num_tile_kv - 2; itile_seq_k >= 0; --itile_seq_k) {
        tPt4r.data() = tPt4r_base_ptr + istage_exp * kTileM;
        wait_barrier(exp_readable[istage_exp], phase_exp);
        // tmem -> reg
        fence_tmem_after_thread_sync();
        copy(tiled_copy_P_t2r, tPt4r, tPr4t);
        cutlass::arch::fence_view_async_tmem_load();

        arrive_barrier(exp_writable[istage_exp]);
        arrive_barrier(p_writable[istage_p]);
        // cast rS to Tin
        cast<Tin, kTileM>(tPr4t, tSr4s);
        wait_barrier(s_writable[istage_s], phase_s);
        wait_barrier(y_readable[istage_q], phase_y);

        // stsm p -> s, ldtm y, rescale, sttm y
        copy(tiled_copy_Y_t2r, tYt4r, tYr4t);
        copy(tiled_copy_S_r2s, tSr4s(_, _, 0), tSs4r(_, _, istage_s));
        cutlass::arch::fence_view_async_tmem_load();
        // rescale
        correct_rescale<kM>(tYr_nm, smem_correct_scale[istage_exp], ilane);
        copy(tiled_copy_Y_r2t, tYr4t, tYt4r);
        cutlass::arch::fence_view_async_tmem_store();
        cutlass::arch::fence_view_async_shared();
        fence_tmem_before_thread_sync();
        cutlass::arch::NamedBarrier::sync(128, kBarId);

        arrive_barrier(s_readable[istage_s]);
        phase_y ^= 1;
        advance_stage<kStageP>(istage_p, phase_p);
        advance_stage<kStageP>(istage_s, phase_s);
        advance_stage<kStageP>(istage_exp, phase_exp);
      }

      if (idx == 0) {
        y_phase[istage_q] = phase_y;
      }
      cutlass::arch::NamedBarrier::sync(128, kBarId);
      wait_barrier(task_readable[istage_q], phase_q);
      TaskInfo::load(&smem_task[istage_q], task_info);
      arrive_barrier(task_writable[istage_q]);
      advance_stage<kStageQ>(istage_q, phase_q);
    }
  } else if (idx >= 384) {
    // Epi: final_softmax -> cast -> stsmy -> splitk_reduce -> tma_store
    cutlass::arch::warpgroup_reg_dealloc<88>();
    idx -= 384;
    iwarp = idx / 32;
    constexpr int kBarId = 2;

    bool leader_warp = (iwarp == 0);
    int istage_q = 0;
    int phase_q = 0;

    tYt.data() = tmem_y_base_ptr;
    auto tmem_epi_tiler = make_tile(Int<kTileV>{}, Int<kTileM>{});
    // (((_128,_16)),((_1,_1),_1,_1)):(((_65536,_1)),((_0,_0),_0,_0))
    auto tYt_epi = zipped_divide(tYt, make_tile(tmem_epi_tiler));
    // ((_128,_16),(_1,_1)):((_1,_128),(_0,_0))
    auto rY_epi = zipped_divide(gY, tmem_epi_tiler);
    auto sY_epi = zipped_divide(sY, tmem_epi_tiler);

    auto tiled_copy_Y_t2r = make_tmem_copy(TMEM_LOAD_ATOM{}, tYt_epi(_, 0));
    auto thr_copy_Y_t2r = tiled_copy_Y_t2r.get_slice(idx);
    auto tYt4r = thr_copy_Y_t2r.partition_S(tYt_epi);
    auto tYr4t = make_tensor_like<float>(thr_copy_Y_t2r.partition_D(rY_epi));

    auto tiled_copy_Y_r2s =
        make_tiled_copy_D(Copy_Atom<UniversalCopy<uint32_t>, Tout>{}, tiled_copy_Y_t2r);
    auto thr_copy_Y_r2s = tiled_copy_Y_r2s.get_slice(idx);
    auto tYs4r = thr_copy_Y_r2s.partition_D(sY_epi);
    auto tYr4s = make_tensor_like<Tout>(thr_copy_Y_r2s.partition_S(rY_epi));

    auto tYt4r_base_ptr = tYt4r.data();

    auto tYr_nm = retile_fragment(tYr4t);
    constexpr int kM = size<1>(tYr_nm);
    vec_t<float, kM> gSum;
    vec_t<float, kM> gMax;

    task_info.valid =
        get_task_info<kTileN, kBlockSize>(task_info, task_map_ptr, task_map_num_chunks_ptr, iblock,
                                          splitk_head_kv_divider, num_tiles_per_sm, num_seq_q);

    const int lse_kv_head_stride = num_seq_q * lse_pad_heads_per_group;
    const int lse_chunk_stride = num_head_k * lse_kv_head_stride;
    const int lse_batch_stride = kMaxSplitK * lse_chunk_stride;

    while (true) {
      if (!task_info.valid) {
        break;
      }
      float vscale_eff = vscale_ptr[task_info.ihead_kv] * kDecodePScaleInv;

      auto *lse_batch = lse_ptr + task_info.ibatch * lse_batch_stride +
                        task_info.ichunk * lse_chunk_stride +
                        task_info.ihead_kv * lse_kv_head_stride;

      tYt4r.data() = tYt4r_base_ptr + istage_q * kTileM;
      wait_barrier(q_writable[istage_q], phase_q);
      // tmem -> reg
      fence_tmem_after_thread_sync();
      copy(tiled_copy_Y_t2r, tYt4r, tYr4t);
      cutlass::arch::fence_view_async_tmem_load();
      final_online_softmax<kTileM>(tYr_nm, gSum, smem_sum[0][istage_q], kBarId, iwarp, ilane);
      if (iwarp == 0 && ilane < 4) {
        load_smem_to_rC<kTileM, kM>(gMax, smem_sum[1][istage_q], 0, ilane);
      }
      arrive_barrier(y_writable[istage_q]);

      store_lse<kM>(lse_batch, gMax, gSum, heads_per_group, num_seq_q, ilane, iwarp);

#pragma unroll
      for (int i = 0; i < size(tYr4t); i++) {
        tYr4t(i) *= vscale_eff;
      }

      // cast to Tout
      cast<Tout, kTileM>(tYr4t, tYr4s);

      // reg -> smem
      tma_store_wait<0>();
      copy(tiled_copy_Y_r2s, tYr4s, tYs4r);
      tma_store_fence();
      cutlass::arch::NamedBarrier::sync(128, kBarId);

      // tma store
      if (leader_warp) {
        auto gYY = tma_y.get_tma_tensor(
            make_shape(num_dim_v, heads_per_group, num_head_k, num_seq_q, kMaxSplitK, num_batch));
        auto btma_y = tma_y.get_slice(0);

        // (((_64,_8),_2),_1,_2):(((_1,_64),_512),_0,_1024)
        auto tYs = btma_y.partition_S(sY);
        // (((_64,_8),_2),1,1,1,1,256):(((_1@0,_1@1),_64@0),_128@0,_8@1,_1@2,_1@3,_1@4)
        auto tYg = btma_y.partition_D(gYY);

        for (int iseqq = 0; iseqq < num_seq_q; iseqq++) {
          cute::copy(tma_y, tYs(_, _, iseqq),
                     tYg(_, _, 0, task_info.ihead_kv, iseqq, task_info.ichunk, task_info.ibatch));
        }
        tma_store_arrive();
      }

      wait_barrier(task_readable[istage_q], phase_q);
      TaskInfo::load(&smem_task[istage_q], task_info);
      arrive_barrier(task_writable[istage_q]);
      advance_stage<kStageQ>(istage_q, phase_q);
    }

    cutlass::arch::NamedBarrier::sync(128, kBarId);
    if (leader_warp) {
      tmem_allocator.release_allocation_lock();
      tmem_allocator.free(tmem_base_ptr, kTmemColumns);
    }
  }

  cudaTriggerProgrammaticLaunchCompletion();
}

}  // namespace kernels
}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_DECODE_SM100_SMALLM_FP8_SPLITK_QKPERTOKEN_PERHEAD_VPERHEAD_KERNELS_CUH_
