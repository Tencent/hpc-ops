// Copyright 2025 hpc-ops authors

#ifndef SRC_SAGE_ATTENTION_SM90_SAGE_ATTN_INT8_KERNEL_CUH_
#define SRC_SAGE_ATTENTION_SM90_SAGE_ATTN_INT8_KERNEL_CUH_

#include <cuda.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <limits>

#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace sage_attention {

using namespace cute;  // NOLINT

static constexpr float kSFP8OffsetSm90 = 8.807f;

template <typename Config, int kNumKvGroups, bool kIsCausal, typename TmaQ, typename TmaK,
          typename TmaV, typename TmaY>
__global__ void __launch_bounds__(128, 2)
    sage_attention_int8_kernel_sm90(const __grid_constant__ TmaQ tma_q,
                                    const __grid_constant__ TmaK tma_k,
                                    const __grid_constant__ TmaV tma_v,
                                    const __grid_constant__ TmaY tma_y,
                                    const float *__restrict__ Q_scale,
                                    const float *__restrict__ K_scale,
                                    const float *__restrict__ V_scale, int qo_len, int kv_len,
                                    int num_head_q, int num_head_kv, float sm_scale_log2e) {
  using namespace cute;  // NOLINT

  using TinQK = typename Config::TinQK;
  using TinV = typename Config::TinV;
  using Tout = typename Config::Tout;

  static constexpr int kTileM = Config::kTileM;
  static constexpr int kTileN = Config::kTileN;
  static constexpr int kTileK = Config::kTileK;
  static constexpr int kTileV = Config::kTileV;
  static constexpr int kStage = Config::kStage;
  static constexpr int kWarpgroupM = Config::kWarpgroupM;

  using SLayoutQ = typename Config::SLayoutQ;
  using SLayoutK = typename Config::SLayoutK;
  using SLayoutV = typename Config::SLayoutV;
  // V is already in K-major layout, no transpose needed
  using SLayoutY = typename Config::SLayoutY;

  using TiledMmaQK = typename Config::TiledMmaQK;
  using TiledMmaPV = typename Config::TiledMmaPV;

  int bx = blockIdx.x;
  int head_id = blockIdx.y;
  int batch_id = blockIdx.z;
  int num_batch = gridDim.z;
  int kv_head_id;
  if constexpr (kNumKvGroups > 0) {
    kv_head_id = head_id / kNumKvGroups;
  } else {
    kv_head_id = head_id / (num_head_q / num_head_kv);
  }

  int idx = threadIdx.x;
  int elected = cute::elect_one_sync();
  int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
  bool is_leader_in_block = (iwarp == 0) && elected;
  bool is_leader_in_warpgroup = ((iwarp % 4) == 0) && elected;

  int num_tile_kv;
  if constexpr (kIsCausal) {
    int causal_kv_bound = min(kv_len, (bx + 1) * kTileM);
    num_tile_kv = (causal_kv_bound + kTileN - 1) / kTileN;
  } else {
    num_tile_kv = (kv_len + kTileN - 1) / kTileN;
  }
  int num_tile_kv_full = (kv_len + kTileN - 1) / kTileN;

  int lane_id = idx % 32;
  int iwarpgroup = idx / 128;
  int iwarp_in_warpgroup = (idx % 128) / 32;
  int idx_in_warpgroup = idx % 128;

  constexpr int num_warps_total = kTileM / 16;
  int global_warp_idx = iwarp_in_warpgroup;

  int padded_kv_len = ((kv_len + kTileN - 1) / kTileN) * kTileN;

  // ── Shared memory ──
  extern __shared__ uint8_t shm_data_raw[] alignas(1024);
  uint8_t *shm_ptr = shm_data_raw;

  auto *shm_q = reinterpret_cast<TinQK *>(shm_ptr);
  shm_ptr += Config::shm_size_q;
  auto *shm_k = reinterpret_cast<TinQK *>(shm_ptr);
  shm_ptr += Config::shm_size_k;
  auto *shm_v = reinterpret_cast<TinV *>(shm_ptr);
  shm_ptr += Config::shm_size_v;
  // Output smem aliases the (now-dead) Q/K/V buffers at the base of the arena.
  // By the epilogue all Q/K/V reads (last WGMMA) have completed, so reusing
  // this region instead of a dedicated 16KB buffer drops dynamic smem from
  // 88KB to 72KB -> 3 CTAs/SM instead of 2 (registers already allow 3).
  auto *shm_y = reinterpret_cast<Tout *>(shm_data_raw);

  auto sQ = make_tensor(make_smem_ptr(shm_q), SLayoutQ{});
  auto sK = make_tensor(make_smem_ptr(shm_k), SLayoutK{});
  auto sV = make_tensor(make_smem_ptr(shm_v), SLayoutV{});
  auto sY = make_tensor(make_smem_ptr(shm_y), SLayoutY{});

  // ── TMA partition ──
  auto gQ = tma_q.get_tma_tensor(make_shape(qo_len, kTileK, num_head_q, num_batch));
  auto gK = tma_k.get_tma_tensor(make_shape(kv_len, kTileK, num_head_kv, num_batch));
  auto gV = tma_v.get_tma_tensor(make_shape(kTileV, padded_kv_len, num_head_kv, num_batch));

  auto btma_q = tma_q.get_slice(0);
  auto btma_k = tma_k.get_slice(0);
  auto btma_v = tma_v.get_slice(0);

  auto tQg = btma_q.partition_S(gQ);
  auto tQs = btma_q.partition_D(sQ);
  auto tKg = btma_k.partition_S(gK);
  auto tKs = btma_k.partition_D(sK);
  auto tVg = btma_v.partition_S(gV);
  auto tVs = btma_v.partition_D(sV);

  // ── MMA setup ──
  TiledMmaQK tiled_mma_qk;
  TiledMmaPV tiled_mma_pv;
  auto thr_mma_qk = tiled_mma_qk.get_slice(idx_in_warpgroup);
  auto thr_mma_pv = tiled_mma_pv.get_slice(idx_in_warpgroup);

  // QK: SS (both from smem via GMMA descriptors)
  auto tQs4r = thr_mma_qk.partition_A(sQ);
  auto tKs4r = thr_mma_qk.partition_B(sK);
  auto tQr = thr_mma_qk.make_fragment_A(tQs4r);
  auto tKr = thr_mma_qk.make_fragment_B(tKs4r);

  // PV: RS (P from register, V from smem via GMMA descriptor — V already K-major)
  auto tVs4r = thr_mma_pv.partition_B(sV);
  auto tVr = thr_mma_pv.make_fragment_B(tVs4r);

  // Accumulator tensors
  auto gAtt =
      make_tensor(make_gmem_ptr(static_cast<int32_t *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileN>{}), make_stride(Int<kTileN>{}, Int<1>{}));
  auto gAtt_fp8 =
      make_tensor(make_gmem_ptr(static_cast<TinV *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileN>{}), make_stride(Int<kTileN>{}, Int<1>{}));
  auto gYY =
      make_tensor(make_gmem_ptr(static_cast<float *>(nullptr)),
                  make_shape(Int<kTileM>{}, Int<kTileV>{}), make_stride(Int<kTileV>{}, Int<1>{}));

  auto tAttr = thr_mma_qk.partition_fragment_C(gAtt);
  auto tYr = thr_mma_pv.partition_fragment_C(gYY);

  // ── Barriers ──
  // Single-warpgroup model: math wg issues all TMA via its elected lane and
  // waits on the readable_* barriers.  No producer wg, so writable_* barriers
  // are not needed (the WGMMA wait inside the math wg itself serializes
  // smem reuse).
  __shared__ uint64_t readable_q;
  __shared__ uint64_t readable_k[kStage];
  __shared__ uint64_t readable_v[kStage];

  if (is_leader_in_block) {
    initialize_barrier(readable_q, 1);
#pragma unroll
    for (int i = 0; i < kStage; ++i) {
      initialize_barrier(readable_k[i], 1);
      initialize_barrier(readable_v[i], 1);
    }
  }
  __syncthreads();

  // ===================== Single Math Warpgroup (128 threads) =====================
  {
    // Issue Q TMA (elected lane).
    if (is_leader_in_block) {
      set_barrier_transaction_bytes(readable_q, sizeof(TinQK) * cosize(SLayoutQ{}));
      cute::copy(tma_q.with(readable_q), tQg(_, bx, _, head_id, batch_id), tQs(_, 0, _));
    }

    // Pre-issue first kStage K/V TMA tiles (elected lane).
    if (is_leader_in_block) {
#pragma unroll
      for (int s = 0; s < kStage; ++s) {
        if (s < num_tile_kv) {
          set_barrier_transaction_bytes(readable_k[s], sizeof(TinQK) * kTileN * kTileK);
          cute::copy(tma_k.with(readable_k[s]), tKg(_, s, _, kv_head_id, batch_id),
                     tKs(_, 0, _, s));

          set_barrier_transaction_bytes(readable_v[s], sizeof(TinV) * kTileV * kTileN);
          cute::copy(tma_v.with(readable_v[s]), tVg(_, _, s, kv_head_id, batch_id),
                     tVs(_, _, 0, s));
        }
      }
    }

    // Float buffer for dequantized QK scores (tAttr is int32, softmax needs float)
    auto tAttr_f32 = make_tensor_like<float>(tAttr);

    // FP8 P: layout bridge from QK C-fragment to PV A-fragment
    auto tAttr_fp8 = make_tensor_like<TinV>(tAttr_f32);
    auto layout_asA = thr_mma_pv.partition_fragment_A(gAtt_fp8).layout();
    auto tAttA_fp8 = make_tensor(tAttr_fp8.data(), layout_asA);

    auto gI = make_identity_tensor(gAtt.shape());
    auto tI = thr_mma_qk.partition_C(gI);

    auto tAttr_f32_mn = retile_fragment(tAttr_f32);
    constexpr int kM = size<0>(tAttr_f32_mn);
    constexpr int kN = size<1>(tAttr_f32_mn);

    Tensor gMax = make_tensor<float>(Int<kM>{});
    Tensor gSum = make_tensor<float>(Int<kM>{});

    // Scale pointers
    const float *q_scale_ptr = Q_scale + batch_id * num_head_q * (gridDim.x * num_warps_total * 8) +
                               head_id * (gridDim.x * num_warps_total * 8) +
                               bx * (num_warps_total * 8) + global_warp_idx * 8 + (lane_id >> 2);
    const float *k_scale_ptr = K_scale + batch_id * num_head_kv * (num_tile_kv_full * 4) +
                               kv_head_id * (num_tile_kv_full * 4) + (lane_id & 3);
    const float *v_scale_ptr = V_scale + batch_id * num_head_kv * kTileV + kv_head_id * kTileV;

    float q_scale = *q_scale_ptr;

    tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;

    clear(tYr);
    clear(gSum);
    fill(gMax, -std::numeric_limits<float>::infinity());

    wait_barrier(readable_q, 0);

    auto tI_mn = retile_fragment(tI);

    // ── (d) Hoist the masked tile out of the main loop ────────────────────
    // For non-causal: only the very last KV tile may have OOB cols.
    // For causal:     every tile from `num_tile_full` onward needs masking.
    // Two separate `for` loops below (no helper / no lambda) so the compiler
    // fully inlines each one and keeps the WGMMA fragment registers stable.
    int mask_start;
    if constexpr (kIsCausal) {
      int num_tile_full = (bx * kTileM) / kTileN;
      mask_start = num_tile_full;
    } else {
      mask_start = num_tile_kv - 1;
    }
    if (mask_start < 0) {
      mask_start = 0;
    }
    if (mask_start > num_tile_kv) {
      mask_start = num_tile_kv;
    }

    // ───────────── Main loop: NO mask code ─────────────
#pragma unroll 1
    for (int itile_kv = 0; itile_kv < mask_start; ++itile_kv) {
      int slot = itile_kv % kStage;
      int phase = (itile_kv / kStage) & 1;

      wait_barrier(readable_k[slot], phase);

      // ── QK WGMMA (SS mode) ──
      tiled_mma_qk.accumulate_ = GMMA::ScaleOut::Zero;

      warpgroup_fence_operand(tAttr);
      warpgroup_arrive();
#pragma unroll
      for (int ik = 0; ik < size<2>(tQr); ++ik) {
        cute::gemm(tiled_mma_qk, tQr(_, _, ik), tKr(_, _, ik, slot), tAttr(_, _, _));
        tiled_mma_qk.accumulate_ = GMMA::ScaleOut::One;
      }
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      warpgroup_fence_operand(tAttr);

      int next_iter = itile_kv + kStage;
      if (next_iter < num_tile_kv && is_leader_in_block) {
        set_barrier_transaction_bytes(readable_k[slot], sizeof(TinQK) * kTileN * kTileK);
        cute::copy(tma_k.with(readable_k[slot]), tKg(_, next_iter, _, kv_head_id, batch_id),
                   tKs(_, 0, _, slot));
      }

      float k_scale_cur = k_scale_ptr[itile_kv * 4];
      float dequant_scale = q_scale * k_scale_cur * sm_scale_log2e;

#pragma unroll
      for (int i = 0; i < size(tAttr); ++i) {
        tAttr_f32(i) = __int2float_rz(tAttr(i)) * dequant_scale;
      }

      // (no mask in main loop)

      auto tYr_mn = retile_fragment(tYr);
#pragma unroll
      for (int im = 0; im < kM; ++im) {
        float row_max = tAttr_f32_mn(im, 0);
#pragma unroll
        for (int in = 1; in < kN; ++in) {
          row_max = fmaxf(row_max, tAttr_f32_mn(im, in));
        }
        row_max = warp_4lane_reduce_max_xor(row_max);
        row_max = row_max - kSFP8OffsetSm90;
        float last_max = gMax(im);
        gMax(im) = fmaxf(last_max, row_max);
        float scale = exp2f_ftz(last_max - gMax(im));
        gSum(im) = gSum(im) * scale;

        float row_sum = 0.f;
#pragma unroll
        for (int in = 0; in < kN; ++in) {
          tAttr_f32_mn(im, in) = exp2f_ftz(tAttr_f32_mn(im, in) - gMax(im));
          row_sum += tAttr_f32_mn(im, in);
        }
        gSum(im) += row_sum;

#pragma unroll
        for (int in = 0; in < cute::size<1>(tYr_mn); ++in) {
          tYr_mn(im, in) *= scale;
        }
      }

      {
        auto tAttr_float32x4 = recast<float4>(tAttr_f32);
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
      }

      wait_barrier(readable_v[slot], phase);

      auto tYr_buf = make_fragment_like(tYr);
      tiled_mma_pv.accumulate_ = GMMA::ScaleOut::Zero;

      warpgroup_fence_operand(tYr_buf);
      warpgroup_arrive();
#pragma unroll
      for (int in = 0; in < size<2>(tVr); in++) {
        cute::gemm(tiled_mma_pv, tAttA_fp8(_, _, in), tVr(_, _, in, slot), tYr_buf);
        tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;
      }
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      warpgroup_fence_operand(tYr_buf);

      if (next_iter < num_tile_kv && is_leader_in_block) {
        set_barrier_transaction_bytes(readable_v[slot], sizeof(TinV) * kTileV * kTileN);
        cute::copy(tma_v.with(readable_v[slot]), tVg(_, _, next_iter, kv_head_id, batch_id),
                   tVs(_, _, 0, slot));
      }

#pragma unroll
      for (int i = 0; i < size(tYr); ++i) {
        tYr(i) += tYr_buf(i);
      }
    }

    // ───────────── Tail loop: WITH mask ─────────────
#pragma unroll 1
    for (int itile_kv = mask_start; itile_kv < num_tile_kv; ++itile_kv) {
      int slot = itile_kv % kStage;
      int phase = (itile_kv / kStage) & 1;

      wait_barrier(readable_k[slot], phase);

      tiled_mma_qk.accumulate_ = GMMA::ScaleOut::Zero;

      warpgroup_fence_operand(tAttr);
      warpgroup_arrive();
#pragma unroll
      for (int ik = 0; ik < size<2>(tQr); ++ik) {
        cute::gemm(tiled_mma_qk, tQr(_, _, ik), tKr(_, _, ik, slot), tAttr(_, _, _));
        tiled_mma_qk.accumulate_ = GMMA::ScaleOut::One;
      }
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      warpgroup_fence_operand(tAttr);

      int next_iter = itile_kv + kStage;
      if (next_iter < num_tile_kv && is_leader_in_block) {
        set_barrier_transaction_bytes(readable_k[slot], sizeof(TinQK) * kTileN * kTileK);
        cute::copy(tma_k.with(readable_k[slot]), tKg(_, next_iter, _, kv_head_id, batch_id),
                   tKs(_, 0, _, slot));
      }

      float k_scale_cur = k_scale_ptr[itile_kv * 4];
      float dequant_scale = q_scale * k_scale_cur * sm_scale_log2e;

#pragma unroll
      for (int i = 0; i < size(tAttr); ++i) {
        tAttr_f32(i) = __int2float_rz(tAttr(i)) * dequant_scale;
      }

      // ── Mask (always on in tail loop) ──
#pragma unroll
      for (int im = 0; im < kM; ++im) {
#pragma unroll
        for (int in = 0; in < kN; ++in) {
          int irow = bx * kTileM + get<0>(tI_mn(im, in));
          int icol = itile_kv * kTileN + get<1>(tI_mn(im, in));
          bool oob = (icol >= kv_len);
          if constexpr (kIsCausal) {
            oob = oob || (icol > irow);
          }
          if (oob) {
            tAttr_f32_mn(im, in) = -std::numeric_limits<float>::infinity();
          }
        }
      }

      auto tYr_mn = retile_fragment(tYr);
#pragma unroll
      for (int im = 0; im < kM; ++im) {
        float row_max = tAttr_f32_mn(im, 0);
#pragma unroll
        for (int in = 1; in < kN; ++in) {
          row_max = fmaxf(row_max, tAttr_f32_mn(im, in));
        }
        row_max = warp_4lane_reduce_max_xor(row_max);
        row_max = row_max - kSFP8OffsetSm90;
        float last_max = gMax(im);
        gMax(im) = fmaxf(last_max, row_max);
        float scale = exp2f_ftz(last_max - gMax(im));
        gSum(im) = gSum(im) * scale;

        float row_sum = 0.f;
#pragma unroll
        for (int in = 0; in < kN; ++in) {
          tAttr_f32_mn(im, in) = exp2f_ftz(tAttr_f32_mn(im, in) - gMax(im));
          row_sum += tAttr_f32_mn(im, in);
        }
        gSum(im) += row_sum;

#pragma unroll
        for (int in = 0; in < cute::size<1>(tYr_mn); ++in) {
          tYr_mn(im, in) *= scale;
        }
      }

      {
        auto tAttr_float32x4 = recast<float4>(tAttr_f32);
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
      }

      wait_barrier(readable_v[slot], phase);

      auto tYr_buf = make_fragment_like(tYr);
      tiled_mma_pv.accumulate_ = GMMA::ScaleOut::Zero;

      warpgroup_fence_operand(tYr_buf);
      warpgroup_arrive();
#pragma unroll
      for (int in = 0; in < size<2>(tVr); in++) {
        cute::gemm(tiled_mma_pv, tAttA_fp8(_, _, in), tVr(_, _, in, slot), tYr_buf);
        tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;
      }
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      warpgroup_fence_operand(tYr_buf);

      if (next_iter < num_tile_kv && is_leader_in_block) {
        set_barrier_transaction_bytes(readable_v[slot], sizeof(TinV) * kTileV * kTileN);
        cute::copy(tma_v.with(readable_v[slot]), tVg(_, _, next_iter, kv_head_id, batch_id),
                   tVs(_, _, 0, slot));
      }

#pragma unroll
      for (int i = 0; i < size(tYr); ++i) {
        tYr(i) += tYr_buf(i);
      }
    }

    // ── Normalize softmax ──
    auto tYr_mn_final = retile_fragment(tYr);
#pragma unroll
    for (int im = 0; im < kM; ++im) {
      gSum(im) = warp_4lane_reduce_sum_xor(gSum(im));
      float one_over_gsum = (gSum(im) > 0.f) ? rcpf_ftz(gSum(im)) : 0.f;
#pragma unroll
      for (int in = 0; in < cute::size<1>(tYr_mn_final); ++in) {
        tYr_mn_final(im, in) *= one_over_gsum;
      }
    }

    // ── Apply V scale ──
    {
      auto gYI = make_identity_tensor(make_shape(Int<kTileM>{}, Int<kTileV>{}));
      auto tYI = thr_mma_pv.partition_C(gYI);
      auto tYI_mn = retile_fragment(tYI);

#pragma unroll
      for (int in = 0; in < cute::size<1>(tYr_mn_final); ++in) {
        int col = get<1>(tYI_mn(0, in));
        float vs = v_scale_ptr[col];
#pragma unroll
        for (int im = 0; im < kM; ++im) {
          tYr_mn_final(im, in) *= vs;
        }
      }
    }

    // ── TMA store output ──
    {
      auto sY_wg = local_tile(sY, make_shape(Int<kTileM / kWarpgroupM>{}, Int<kTileV>{}),
                              make_coord(iwarpgroup, _0{}));
      using R2SCopyAtomC = Copy_Atom<SM90_U32x2_STSM_N, Tout>;
      auto r2s_tiled_copy = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma_pv);
      auto r2s_thr_copy = r2s_tiled_copy.get_slice(idx_in_warpgroup);

      auto tYr_bf16 = make_tensor_like<Tout>(tYr);
#pragma unroll
      for (int i = 0; i < size(tYr); ++i) {
        tYr_bf16(i) = Tout(tYr(i));
      }

      tma_store_wait<0>();

      auto tYr4s = r2s_thr_copy.retile_S(tYr_bf16);
      auto tYs4r = r2s_thr_copy.partition_D(sY_wg);
      cute::copy(r2s_tiled_copy, tYr4s, tYs4r);
    }

    syncwarpgroup(iwarpgroup);
    tma_store_fence();

    if (is_leader_in_warpgroup) {
      auto cY = tma_y.get_tma_tensor(make_shape(qo_len, kTileV, num_head_q, num_batch));
      auto btma_y = tma_y.get_slice(0);
      auto tYss = btma_y.partition_S(sY);
      auto tYgg = btma_y.partition_D(cY);
      cute::copy(tma_y, tYss(_, iwarpgroup, 0),
                 tYgg(_, bx * kWarpgroupM + iwarpgroup, 0, head_id, batch_id));
      tma_store_arrive();
    }
  }
}

}  // namespace sage_attention
}  // namespace hpc

#endif  // SRC_SAGE_ATTENTION_SM90_SAGE_ATTN_INT8_KERNEL_CUH_
