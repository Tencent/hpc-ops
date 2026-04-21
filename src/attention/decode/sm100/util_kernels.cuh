// Copyright 2025 hpc-ops authors

#ifndef SRC_ATTENTION_DECODE_SM100_UTIL_KERNELS_CUH_
#define SRC_ATTENTION_DECODE_SM100_UTIL_KERNELS_CUH_

#include <cuda.h>
#include <stdio.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace attention {
namespace kernels {

struct alignas(16) TaskInfo {
  int iblock;
  int ibatch;
  int ihead_kv;
  int ichunk;

  int valid;
  int num_seq_kvcache;
  int num_seq_kv;
  int num_chunks;

  int num_blocks;
  int num_blocks_per_chunk;
  int num_tile_kv;
  int num_tile_full;

  int num_tile_causal;
  int is_last_chunk;
  int pad[2];

  static __device__ __forceinline__ void load(TaskInfo* ptr, TaskInfo& task_info) {
    auto* src_16B = reinterpret_cast<uint4*>(ptr);
    auto* dst_16B = reinterpret_cast<uint4*>(&task_info);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      dst_16B[i] = src_16B[i];
    }
  }

  static __device__ __forceinline__ void store(TaskInfo& task_info, TaskInfo* ptr) {
    auto* dst_16B = reinterpret_cast<uint4*>(ptr);
    auto* src_16B = reinterpret_cast<uint4*>(&task_info);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      dst_16B[i] = src_16B[i];
    }
  }
};

template <int kStage, int kStep = 1>
__device__ __forceinline__ void advance_stage(int& istage, int& phase) {
  istage += kStep;
  if (istage >= kStage) {
    istage = istage % kStage;
    phase ^= 1;
  }
}

template <int kTileN, int kBlockSize, int kSplitK>
__device__ __forceinline__ bool get_task_info(TaskInfo& task_info, const int& iblock,
                                              cutlass::FastDivmod splitk_head_kv_divider,
                                              const int* num_seq_kvcache_ptr, bool new_kv_included,
                                              const int& num_seq_q) {
  int ibatch, isplit_head_kv;
  splitk_head_kv_divider(ibatch, isplit_head_kv, iblock);
  int ihead_kv = isplit_head_kv / kSplitK;
  int ichunk = isplit_head_kv % kSplitK;

  int num_seq_kvcache = num_seq_kvcache_ptr[ibatch];

  if (new_kv_included) {
    num_seq_kvcache -= num_seq_q;
  }
  int num_seq_kv = num_seq_q + num_seq_kvcache;

  if (num_seq_kv <= 0) {
    return false;
  }

  int num_seq_per_chunk = (num_seq_kv + kSplitK - 1) / kSplitK;
  num_seq_per_chunk = (num_seq_per_chunk + kTileN - 1) / kTileN * kTileN;

  int iseq_start = ichunk * num_seq_per_chunk;
  if (iseq_start >= num_seq_kv) {
    return false;
  }

  bool is_last_chunk = false;
  if (iseq_start + num_seq_per_chunk >= num_seq_kv) {
    is_last_chunk = true;
  }

  int num_chunks = (num_seq_kv + num_seq_per_chunk - 1) / num_seq_per_chunk;

  num_seq_kv = min(num_seq_kv - iseq_start, num_seq_per_chunk);

  if (is_last_chunk) {
    num_seq_kvcache = num_seq_kv - num_seq_q;
  } else {
    num_seq_kvcache = num_seq_kv;
  }

  int num_blocks = (num_seq_kv + kBlockSize - 1) / kBlockSize;
  int num_blocks_per_chunk = (num_seq_per_chunk + kBlockSize - 1) / kBlockSize;

  int num_tile_kv = (num_seq_kv + kTileN - 1) / kTileN;
  int num_tile_full = num_seq_kvcache / kTileN;

  int num_tile_causal = 0;
  if (is_last_chunk) {
    num_tile_causal = num_tile_kv - num_tile_full;
  }
  num_tile_full = num_tile_kv - num_tile_causal;

  task_info.iblock = iblock;
  task_info.ibatch = ibatch;
  task_info.ihead_kv = ihead_kv;
  task_info.ichunk = ichunk;
  task_info.num_seq_kvcache = num_seq_kvcache;
  task_info.num_seq_kv = num_seq_kv;
  task_info.num_chunks = num_chunks;
  task_info.num_blocks = num_blocks;
  task_info.num_blocks_per_chunk = num_blocks_per_chunk;
  task_info.num_tile_kv = num_tile_kv;
  task_info.num_tile_full = num_tile_full;
  task_info.num_tile_causal = num_tile_causal;
  task_info.is_last_chunk = is_last_chunk;

  return true;
}

template <int kTileN, int kBlockSize>
__device__ __forceinline__ bool get_task_info(TaskInfo& task_info, const int* task_map_ptr,
                                              const int* task_map_num_chunks_ptr, const int& iblock,
                                              cutlass::FastDivmod splitk_head_kv_divider,
                                              const int& num_tiles_per_sm, const int& num_seq_q) {
  auto v1 = load<int, 4>(task_map_ptr);
  auto v2 = load<int, 4>(task_map_ptr + 4);

  int ibatch = v1[0];
  int ichunk = v1[1];
  int iseq_start = v1[2];
  int num_seq_kv = v1[3];

  int num_seq_kvcache = v2[0];
  int num_tile_kv = v2[1];
  int num_tile_full = v2[2];
  int is_casual_chunk = v2[3];

  if (ibatch < 0) {
    return false;
  }

  int num_blocks = (num_seq_kv + kBlockSize - 1) / kBlockSize;
  int num_blocks_start = (iseq_start + kBlockSize - 1) / kBlockSize;

  int num_tile_causal = 0;
  if (is_casual_chunk) {
    num_tile_causal = num_tile_kv - num_tile_full;
  }

  task_info.iblock = iblock;
  task_info.ibatch = ibatch;
  task_info.ihead_kv = blockIdx.x;
  task_info.ichunk = ichunk;
  task_info.num_seq_kvcache = num_seq_kvcache;
  task_info.num_seq_kv = num_seq_kv;
  task_info.num_blocks = num_blocks;
  task_info.num_blocks_per_chunk = num_blocks_start;
  task_info.num_tile_kv = num_tile_kv;
  task_info.num_tile_full = num_tile_full;
  task_info.num_tile_causal = num_tile_causal;
  task_info.is_last_chunk = is_casual_chunk;

  return true;
}

template <int kTileN, int kBlockSize, int kSplitK>
__device__ __forceinline__ bool get_next_task(TaskInfo& task_info, uint4* clc_resp,
                                              uint64_t* clc_readable, uint64_t* clc_writable,
                                              int& phase_task, const int& block_rank_in_cluster,
                                              const int* num_seq_kvcache_ptr, bool new_kv_included,
                                              const int& num_seq_q,
                                              cutlass::FastDivmod splitk_head_kv_divider) {
  using namespace cute;  // NOLINT

  wait_barrier(clc_readable[0], phase_task);
  auto [icluster_first_cta, valid] = get_next_block(clc_resp);

  if (!valid) {
    task_info.valid = false;
    return false;
  }
  arrive_cluster_barrier(clc_writable[0]);

  int iblock = icluster_first_cta + block_rank_in_cluster;

  task_info.valid = get_task_info<kTileN, kBlockSize, kSplitK>(
      task_info, iblock, splitk_head_kv_divider, num_seq_kvcache_ptr, new_kv_included, num_seq_q);

  if (!task_info.valid) {
    arrive_cluster_barrier(clc_writable[0]);
  }

  phase_task ^= 1;

  return true;
}

template <typename TmaQ, typename TensorGQ, typename TensorSQ>
__device__ __forceinline__ void load_q(TmaQ& tma_q, uint64_t* q_readable, uint64_t* q_writable,
                                       TensorGQ& tQg, TensorSQ& tQs, const int& ihead_kv,
                                       const int& ibatch, const int& num_seq_q,
                                       const int& trans_bytes, const int& istage_q,
                                       const int& phase_q) {
  using namespace cute;  // NOLINT

  wait_barrier(q_writable[istage_q], phase_q);
#pragma unroll 1
  for (int iseqq = 0; iseqq < num_seq_q; iseqq++) {
    cute::copy(tma_q.with(q_readable[istage_q], 0, TMA::CacheHintSm90::EVICT_LAST),
               tQg(_, 0, _, ihead_kv, iseqq, ibatch), tQs(_, iseqq, _, istage_q));
  }

  set_barrier_transaction_bytes(q_readable[istage_q], trans_bytes);
}

__device__ __forceinline__ void load_block_ids(int* shm_blkids, const int* block_ids,
                                               const int& num_blocks, const int& ilane) {
  for (int i = ilane; i < num_blocks; i += 32) {
    cute::SM80_CP_ASYNC_CACHEALWAYS<int>::copy(block_ids[i], shm_blkids[i]);
  }
}

template <int kTileN, int kBlockSize, int kStage, typename Tin, typename Tma, typename TensorGK,
          typename TensorSK>
__device__ __forceinline__ void load_paged_k(Tma& tma, uint64_t* readable, uint64_t* writable,
                                             TensorGK& tKg, TensorSK& tKs, int ihead_kv, int dim,
                                             const int* block_ids, int num_blocks, int itile,
                                             int& istage, int& phase) {
  using namespace cute;  // NOLINT

  constexpr int kBlockPerTileN = kTileN / kBlockSize;

  vec_t<int, kBlockPerTileN> blk_ids;
#pragma unroll
  for (int ikvblock = 0; ikvblock < kBlockPerTileN; ikvblock++) {
    int kvblk_id = itile * kBlockPerTileN + ikvblock;
    int blk_id = -1;
    if (kvblk_id < num_blocks) {
      blk_id = block_ids[kvblk_id];
    }
    blk_ids[ikvblock] = blk_id;
  }

  wait_barrier(writable[istage], phase);
#pragma unroll
  for (int ikvblock = 0; ikvblock < kBlockPerTileN; ikvblock++) {
    cute::copy(tma.with(readable[istage], 0, TMA::CacheHintSm90::EVICT_FIRST),
               tKg(_, 0, _, ihead_kv, blk_ids[ikvblock]), tKs(_, ikvblock, _, istage));
  }
  set_barrier_transaction_bytes(readable[istage], sizeof(Tin) * kTileN * dim);

  advance_stage<kStage>(istage, phase);
}

template <int kTileN, int kBlockSize, int kStage, typename Tin, typename Tma, typename TmaKS,
          typename TensorGK, typename TensorSK, typename TensorGKS, typename TensorSKS>
__device__ __forceinline__ void load_paged_k_with_scale(Tma& tma, TmaKS& tma_ks, uint64_t* readable,
                                                        uint64_t* writable, TensorGK& tKg,
                                                        TensorSK& tKs, TensorGKS& tKSg,
                                                        TensorSKS& tKSs, int ihead_kv, int dim,
                                                        const int* block_ids, int num_blocks,
                                                        int itile, int& istage, int& phase) {
  using namespace cute;  // NOLINT

  constexpr int kBlockPerTileN = kTileN / kBlockSize;

  vec_t<int, kBlockPerTileN> blk_ids;
#pragma unroll
  for (int ikvblock = 0; ikvblock < kBlockPerTileN; ikvblock++) {
    int kvblk_id = itile * kBlockPerTileN + ikvblock;
    int blk_id = -1;
    if (kvblk_id < num_blocks) {
      blk_id = block_ids[kvblk_id];
    }
    blk_ids[ikvblock] = blk_id;
  }

  wait_barrier(writable[istage], phase);
#pragma unroll
  for (int ikvblock = 0; ikvblock < kBlockPerTileN; ikvblock++) {
    cute::copy(tma.with(readable[istage], 0, TMA::CacheHintSm90::EVICT_FIRST),
               tKg(_, 0, _, ihead_kv, blk_ids[ikvblock]), tKs(_, ikvblock, _, istage));
    cute::copy(tma_ks.with(readable[istage]), tKSg(_, 0, _, ihead_kv, blk_ids[ikvblock]),
               tKSs(_, ikvblock, _, istage));
  }
  set_barrier_transaction_bytes(
      readable[istage], sizeof(Tin) * kTileN * dim + sizeof(float) * kBlockPerTileN * kBlockSize);

  advance_stage<kStage>(istage, phase);
}

template <int kTileN, int kBlockSize, int kStage, typename Tin, typename Tma, typename TensorGV,
          typename TensorSV>
__device__ __forceinline__ void load_paged_v(Tma& tma, uint64_t* readable, uint64_t* writable,
                                             TensorGV& tVg, TensorSV& tVs, int ihead_kv, int dim,
                                             const int* block_ids, int num_blocks, int itile,
                                             int& istage, int& phase) {
  using namespace cute;  // NOLINT

  constexpr int kBlockPerTileN = kTileN / kBlockSize;

  vec_t<int, kBlockPerTileN> blk_ids;
#pragma unroll
  for (int ikvblock = 0; ikvblock < kBlockPerTileN; ikvblock++) {
    int kvblk_id = itile * kBlockPerTileN + ikvblock;
    int blk_id = -1;
    if (kvblk_id < num_blocks) {
      blk_id = block_ids[kvblk_id];
    }
    blk_ids[ikvblock] = blk_id;
  }

  wait_barrier(writable[istage], phase);
#pragma unroll
  for (int ikvblock = 0; ikvblock < kBlockPerTileN; ikvblock++) {
    cute::copy(tma.with(readable[istage], 0, TMA::CacheHintSm90::EVICT_FIRST),
               tVg(_, _, 0, ihead_kv, blk_ids[ikvblock]), tVs(_, _, ikvblock, istage));
  }
  set_barrier_transaction_bytes(readable[istage], sizeof(Tin) * kTileN * dim);

  advance_stage<kStage>(istage, phase);
}

template <int kTileM, typename TiledMma, typename TensorRQ, typename TensorRK, typename TensorTP>
__device__ __forceinline__ void qk_gemm(TiledMma& tiled_mma, TensorRQ& tQr, TensorRK& tKr,
                                        TensorTP& tPt, const uint32_t& tmem_p_base_ptr,
                                        const int& istage_q, const int& istage_k,
                                        const int& istage_p) {
  using namespace cute;  // NOLINT

  tPt.data() = tmem_p_base_ptr + istage_p * kTileM;
  tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
#pragma unroll
  for (int ik = 0; ik < size<2>(tQr); ++ik) {
    cute::gemm(tiled_mma, tKr(_, _, ik, istage_k), tQr(_, _, ik, istage_q), tPt);
    tiled_mma.accumulate_ = UMMA::ScaleOut::One;
  }
}

template <typename TiledMma, typename TensorRV, typename TensorRS, typename TensorTY>
__device__ __forceinline__ void sv_gemm(TiledMma& tiled_mma, TensorRV& tVr, TensorRS& tSr,
                                        TensorTY& tYt, const int& istage_v, const int& istage_s) {
  using namespace cute;  // NOLINT
#pragma unroll
  for (int ik = 0; ik < size<2>(tVr); ++ik) {
    cute::gemm(tiled_mma, tVr(_, _, ik, istage_v), tSr(_, _, ik, istage_s), tYt);
    tiled_mma.accumulate_ = UMMA::ScaleOut::One;
  }
}

template <typename T, int kTileM, typename TensorFp32, typename TensorT>
__device__ __forceinline__ void cast(TensorFp32& reg_fp32, TensorT& reg_T) {
  using namespace cute;  // NOLINT

  if constexpr (std::is_same_v<T, cute::float_e4m3_t>) {
    auto reg_fp8x4 = recast<__nv_fp8x4_e4m3>(reg_T);
    constexpr int kIterM = kTileM / 8;
#pragma unroll
    for (int in = 0; in < size(reg_fp8x4) / kIterM; in++) {
#pragma unroll
      for (int im = 0; im < kIterM; im++) {
        float4 fp32x4;
        int ibase = 4 * (in * kIterM + im);
        fp32x4.x = reg_fp32(ibase);
        fp32x4.y = reg_fp32(ibase + 1);
        fp32x4.z = reg_fp32(ibase + 2);
        fp32x4.w = reg_fp32(ibase + 3);
        reg_fp8x4(in * kIterM + im) = __nv_fp8x4_e4m3(fp32x4);
      }
    }
  } else if constexpr (std::is_same_v<T, cute::bfloat16_t>) {
    auto reg_fp32x2 = recast<float2>(reg_fp32);
    auto reg_bf16x2 = recast<__nv_bfloat162>(reg_T);

#pragma unroll
    for (int i = 0; i < size(reg_fp32x2); i++) {
      reg_bf16x2(i) = __float22bfloat162_rn(reg_fp32x2(i));
    }
  } else if constexpr (std::is_same_v<T, float>) {
#pragma unroll
    for (int i = 0; i < size(reg_fp32); i++) {
      reg_T(i) = reg_fp32(i);
    }
  }
}

template <int kTileN, int kHeadsPerGroup, typename TensorAtt, typename TensorI,
          typename TensorScale>
__device__ __forceinline__ void apply_casual_mask_with_scale(TensorAtt& tAttr_nm, TensorI& tI_nm,
                                                             TensorScale& scales,
                                                             const int& itile_seq_kv,
                                                             const int& num_seq_kvcache,
                                                             const int& num_seq_kv) {
  using namespace cute;  // NOLINT
  constexpr int kN = size<0>(TensorAtt{});
  constexpr int kM = size<1>(TensorAtt{});

#pragma unroll
  for (int im = 0; im < kM; ++im) {
#pragma unroll
    for (int in = 0; in < kN; ++in) {
      int iposq = num_seq_kvcache + get<1>(tI_nm(in, im)) / kHeadsPerGroup;
      int iposk = itile_seq_kv * kTileN + get<0>(tI_nm(in, im));

      if ((iposk > iposq) || (iposk >= num_seq_kv)) {
        tAttr_nm(in, im) = -std::numeric_limits<float>::infinity();
      } else {
        tAttr_nm(in, im) *= scales(im);
      }
    }
  }
}

template <int kTileN, int kHeadsPerGroup, typename TensorAtt, typename TensorI, typename TensorQS,
          typename TensorKS>
__device__ __forceinline__ void apply_casual_mask_with_scale(TensorAtt& tAttr_nm, TensorI& tI_nm,
                                                             TensorQS& qscales, TensorKS& kscales,
                                                             const int& itile_seq_kv,
                                                             const int& num_seq_kvcache,
                                                             const int& num_seq_kv) {
  using namespace cute;  // NOLINT
  constexpr int kN = size<0>(TensorAtt{});
  constexpr int kM = size<1>(TensorAtt{});

#pragma unroll
  for (int im = 0; im < kM; ++im) {
#pragma unroll
    for (int in = 0; in < kN; ++in) {
      int iposq = num_seq_kvcache + get<1>(tI_nm(in, im)) / kHeadsPerGroup;
      int iposk = itile_seq_kv * kTileN + get<0>(tI_nm(in, im));

      if ((iposk > iposq) || (iposk >= num_seq_kv)) {
        tAttr_nm(in, im) = -std::numeric_limits<float>::infinity();
      } else {
        tAttr_nm(in, im) *= qscales(im) * kscales(in);
      }
    }
  }
}

template <int kTileM, int kM, typename TensorReg>
__device__ __forceinline__ void store_rC_to_smem(TensorReg& reg, float* smem, int iwarp,
                                                 int ilane) {
  if constexpr (kM == 2 || kM == 4) {
    store(smem + iwarp * kTileM + ilane * kM, reg);
  } else if constexpr (kM == 6) {
    vec_t<float, 4> reg1 = *reinterpret_cast<vec_t<float, 4>*>(&reg[0]);
    vec_t<float, 2> reg2 = *reinterpret_cast<vec_t<float, 2>*>(&reg[4]);
    store(smem + iwarp * kTileM + ilane * 4, reg1);
    store(smem + iwarp * kTileM + ilane * 2 + 16, reg2);
  } else if constexpr (kM == 8) {
    vec_t<float, 4> reg1 = *reinterpret_cast<vec_t<float, 4>*>(&reg[0]);
    vec_t<float, 4> reg2 = *reinterpret_cast<vec_t<float, 4>*>(&reg[4]);
    store(smem + iwarp * kTileM + ilane * 4, reg1);
    store(smem + iwarp * kTileM + ilane * 4 + 16, reg2);
  }
}

template <int kTileM, int kM, typename TensorReg>
__device__ __forceinline__ void load_smem_to_rC(TensorReg& reg, float* smem, int iwarp, int ilane) {
  if constexpr (kM == 2 || kM == 4) {
    reg = load<float, kM>(smem + iwarp * kTileM + ilane * kM);
  } else if constexpr (kM == 6) {
    vec_t<float, 4>& reg1 = *reinterpret_cast<vec_t<float, 4>*>(&reg[0]);
    vec_t<float, 2>& reg2 = *reinterpret_cast<vec_t<float, 2>*>(&reg[4]);

    reg1 = load<float, 4>(smem + iwarp * kTileM + ilane * 4);
    reg2 = load<float, 2>(smem + iwarp * kTileM + ilane * 2 + 16);
  } else if constexpr (kM == 8) {
    vec_t<float, 4>& reg1 = *reinterpret_cast<vec_t<float, 4>*>(&reg[0]);
    vec_t<float, 4>& reg2 = *reinterpret_cast<vec_t<float, 4>*>(&reg[4]);

    reg1 = load<float, 4>(smem + iwarp * kTileM + ilane * 4);
    reg2 = load<float, 4>(smem + iwarp * kTileM + ilane * 4 + 16);
  }
}

template <bool kCheckInf, int kTileM, typename TensorA, typename TensorM, typename TensorS,
          typename TensorScale>
__device__ __forceinline__ void online_softmax(TensorA& tAttr_nm, TensorM& gMax, TensorS& gSum,
                                               TensorScale& one_over_dk_log2e, float* smem_max,
                                               float* smem_correct_scale, int ibar, int iwarp,
                                               int ilane) {
  using namespace cute;  // NOLINT
  constexpr int kN = size<0>(TensorA{});
  constexpr int kM = size<1>(TensorA{});

  vec_t<float, kM> warp_max;
#pragma unroll
  for (int im = 0; im < kM; ++im) {
    float row_max = tAttr_nm(0, im);

#pragma unroll
    for (int in = 1; in < kN; ++in) {
      row_max = fmaxf(row_max, tAttr_nm(in, im));
    }

    warp_max[im] = warp_8lane_stride4_reduce_max_xor(row_max) * one_over_dk_log2e[im];
  }

  if (ilane < 4) {
    store_rC_to_smem<kTileM, kM>(warp_max, smem_max, iwarp, ilane);
  }

  cutlass::arch::NamedBarrier::sync(128, ibar);

  if (ilane < 4) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
      vec_t<float, kM> reduce_max;
      load_smem_to_rC<kTileM, kM>(reduce_max, smem_max, i, ilane);
#pragma unroll
      for (int im = 0; im < kM; ++im) {
        warp_max[im] = fmax(reduce_max[im], warp_max[im]);
      }
    }
  }

#pragma unroll
  for (int im = 0; im < kM; ++im) {
    warp_max[im] = __shfl_sync(0xFFFFFFFF, warp_max[im], ilane % 4);
  }

  vec_t<float, kM> scales;
#pragma unroll
  for (int im = 0; im < kM; ++im) {
    float last_max = gMax[im];
    float row_max = fmaxf(last_max, warp_max[im]);
    float row_sum = 0.f;

    gMax[im] = row_max;

    if constexpr (kCheckInf) {
      if (gMax[im] == -std::numeric_limits<float>::infinity()) {
#pragma unroll
        for (int in = 0; in < kN; ++in) {
          tAttr_nm(in, im) = 0.f;
        }
        continue;
      }
    }

#pragma unroll
    for (int in = 0; in < kN; ++in) {
      tAttr_nm(in, im) = exp2f_ftz(tAttr_nm(in, im) * one_over_dk_log2e[im] - gMax[im]);
      row_sum += tAttr_nm(in, im);
    }

    scales[im] = 1.f;
    if (last_max != row_max) {
      scales[im] = exp2f_ftz(last_max - gMax[im]);
    }
    gSum[im] = gSum[im] * scales[im] + row_sum;
  }

  if (iwarp == 0) {
    if (ilane < 4) {
      store_rC_to_smem<kTileM, kM>(scales, smem_correct_scale, 0, ilane);
    }
    __syncwarp();
  }
}

template <int kM, typename TensorY>
__device__ __forceinline__ void correct_rescale(TensorY& tYr_nm, float* smem_correct_scale,
                                                int ilane) {
  vec_t<float, kM> scales;
  if (ilane < 4) {
    load_smem_to_rC<0, kM>(scales, smem_correct_scale, 0, ilane);
  }

#pragma unroll
  for (int im = 0; im < kM; ++im) {
    float scale = __shfl_sync(0xFFFFFFFF, scales[im], ilane % 4);
    if (scale != 1) {
#pragma unroll
      for (int in = 0; in < cute::size<0>(tYr_nm); ++in) {
        tYr_nm(in, im) = tYr_nm(in, im) * scale;
      }
    }
  }
}

template <int kTileM, typename TensorY, typename TensorS>
__device__ __forceinline__ void final_online_softmax(TensorY& tYr_nm, TensorS& gSum,
                                                     float* smem_sum, int ibar, int iwarp,
                                                     int ilane) {
  using namespace cute;  // NOLINT
  constexpr int kM = size<1>(TensorY{});
  if (ilane < 4) {
#pragma unroll
    for (int im = 0; im < kM; ++im) {
      gSum[im] = 0.f;
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
      vec_t<float, kM> reduce_sum;
      load_smem_to_rC<kTileM, kM>(reduce_sum, smem_sum, i, ilane);
#pragma unroll
      for (int im = 0; im < kM; ++im) {
        gSum[im] += reduce_sum[im];
      }
    }
  }

#pragma unroll
  for (int im = 0; im < kM; ++im) {
    gSum[im] = __shfl_sync(0xFFFFFFFF, gSum[im], ilane % 4);
  }

#pragma unroll
  for (int im = 0; im < kM; ++im) {
    float one_over_gsum = 0.f;
    if (gSum[im] != 0) {
      one_over_gsum = rcpf_ftz(gSum[im]);
    }
#pragma unroll
    for (int in = 0; in < cute::size<0>(tYr_nm); ++in) {
      tYr_nm(in, im) = tYr_nm(in, im) * one_over_gsum;
    }
  }
}

template <int kM, typename TensorMax, typename TensorSum>
__device__ __forceinline__ void store_lse(float* lse_batch, TensorMax& gMax, TensorSum& gSum,
                                          const int& heads_per_group, const int& num_seq_q,
                                          const int& ilane, const int& iwarp) {
  // write lse
  if (iwarp == 0 && ilane * 2 < heads_per_group) {
    vec_t<float, kM> lse;
#pragma unroll
    for (int im = 0; im < kM; ++im) {
      if (gMax[im] == -std::numeric_limits<float>::infinity()) {
        lse[im] = -std::numeric_limits<float>::infinity();
      } else {
        lse[im] = gMax[im] + log2f_ftz(gSum[im]);
      }
    }
    auto& lse_store = reshape<kM / 2, 2>(lse);
#pragma unroll
    for (int i = 0; i < kM / 2; i++) {
      if (i < num_seq_q) {
        store(lse_batch + i * 8 + ilane * 2, lse_store[i]);
      }
    }
  }
}

}  // namespace kernels
}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_DECODE_SM100_UTIL_KERNELS_CUH_
