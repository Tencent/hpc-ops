// Copyright 2025 hpc-ops authors

#ifndef SRC_ATTENTION_DECODE_SPLITK_ADAPTIVE_COMBINE_KERNELS_CUH_
#define SRC_ATTENTION_DECODE_SPLITK_ADAPTIVE_COMBINE_KERNELS_CUH_

#include <cuda.h>
#include <stdio.h>

#include <algorithm>
#include <limits>

#include "cute/tensor.hpp"
#include "cutlass/fast_math.h"
#include "src/attention/decode/sched_task_info.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace attention {
namespace decode {
namespace kernels {

template <int kCoopWarps, typename TL, typename TW>
__device__ __forceinline__ float softmax_weights(const TL &gL, TW &sW, int num_chunks,
                                                 int coop_warp, int ilane, float2 *sML,
                                                 float *rescale) {
  using namespace cute;  // NOLINT
  float max_lse = -std::numeric_limits<float>::infinity();
#pragma unroll 1
  for (int ichunk = coop_warp + ilane * kCoopWarps; ichunk < num_chunks;
       ichunk += 32 * kCoopWarps) {
    float lse = gL(ichunk);
    sW(ichunk) = lse;
    max_lse = fmaxf(max_lse, lse);
  }
  max_lse = warp_reduce_max_xor(max_lse);
  float sum_lse = 0.f;
#pragma unroll 1
  for (int ichunk = coop_warp + ilane * kCoopWarps; ichunk < num_chunks;
       ichunk += 32 * kCoopWarps) {
    float weight = exp2f_ftz(sW(ichunk) - max_lse);
    sW(ichunk) = weight;
    sum_lse += weight;
  }
  sum_lse = warp_reduce_sum_xor(sum_lse);
  *rescale = 1.f;
  if constexpr (kCoopWarps > 1) {
    if (ilane == 0) {
      sML[coop_warp] = make_float2(max_lse, sum_lse);
    }
    __syncthreads();
    float max_lse_global = -std::numeric_limits<float>::infinity();
#pragma unroll
    for (int w = 0; w < kCoopWarps; w++) {
      max_lse_global = fmaxf(max_lse_global, sML[w].x);
    }
    float sum_lse_global = 0.f;
#pragma unroll
    for (int w = 0; w < kCoopWarps; w++) {
      sum_lse_global += sML[w].y * exp2f_ftz(sML[w].x - max_lse_global);
    }
    *rescale = exp2f_ftz(max_lse - max_lse_global);
    sum_lse = sum_lse_global;
  }
  return sum_lse > 0.f ? rcpf_ftz(sum_lse) : 0.f;
}

template <int kPrefetch, typename TiledCopy, typename TSg, typename TSr>
__device__ __forceinline__ void prefetch_chunks(const TiledCopy &tiled_copy, const TSg &tSg,
                                                TSr &tSr, int num_chunks) {
  using namespace cute;  // NOLINT
  const int first = num_chunks < kPrefetch ? num_chunks : kPrefetch;
#pragma unroll
  for (int s = 0; s < kPrefetch; s++) {
    if (s < first) {
      copy(tiled_copy, tSg(_, _, s), tSr(_, _, s));
    }
  }
}

template <int kPrefetch, typename TiledCopy, typename TSg, typename TSr, typename TYr, typename TW>
__device__ __forceinline__ void accumulate_chunks(const TiledCopy &tiled_copy, const TSg &tSg,
                                                  TSr &tSr, TYr &tYr, const TW &sW,
                                                  int num_chunks) {
  using namespace cute;  // NOLINT

  clear(tYr);

  int idx = 0;
  for (; idx + kPrefetch <= num_chunks; idx += kPrefetch) {
    if (idx != 0) {
#pragma unroll
      for (int s = 0; s < kPrefetch; s++) {
        copy(tiled_copy, tSg(_, _, idx + s), tSr(_, _, s));
      }
    }
#pragma unroll
    for (int s = 0; s < kPrefetch; s++) {
      float weight = sW(idx + s);
      auto input = tSr(_, _, s);
#pragma unroll
      for (int i = 0; i < size(tYr); i++) {
        tYr(i) += weight * input(i);
      }
    }
  }
  const int rem = num_chunks - idx;
  if (idx != 0) {
#pragma unroll
    for (int s = 0; s < kPrefetch; s++) {
      if (s < rem) {
        copy(tiled_copy, tSg(_, _, idx + s), tSr(_, _, s));
      }
    }
  }
#pragma unroll
  for (int s = 0; s < kPrefetch; s++) {
    if (s < rem) {
      float weight = sW(idx + s);
      auto input = tSr(_, _, s);
#pragma unroll
      for (int i = 0; i < size(tYr); i++) {
        tYr(i) += weight * input(i);
      }
    }
  }
}

template <typename TiledCopy, typename TYg, typename TYrOut, typename TYrAcc>
__device__ __forceinline__ void store_row(const TiledCopy &tiled_copy_store, TYg &tYg,
                                          TYrOut &tYr_bf16, const TYrAcc &tYr, float inv_sum) {
  using namespace cute;  // NOLINT
  using TElem = typename TYrOut::value_type;
#pragma unroll
  for (int i = 0; i < size(tYr_bf16); i++) {
    tYr_bf16(i) = static_cast<TElem>(tYr(i) * inv_sum);
  }
  copy(tiled_copy_store, tYr_bf16, tYg);
}

template <typename T, int kTileV, int kWarpCount, int kHeavyThresh>
__global__ void attention_decode_dynamic_splitk_adaptive_combine_kernel(
    T *y_ptr, const float *split_input_ptr, const float *lse_ptr, const int *task_map_ptr,
    int num_total_ctas, int num_batch, int num_seq_q, int num_head_q, int num_head_k,
    int pad_heads_per_group, int max_splitk, cutlass::FastDivmod heads_per_group_divider) {
  using namespace cute;  // NOLINT

  const int iwarp = threadIdx.x / 32;
  const int ilane = threadIdx.x % 32;
  constexpr int kTaskStride = dynamic::kTaskScheduleInfoStride;
  constexpr int kPrefetch = 8;

  // Split input: (Dv, Hq, Sq, C, B)
  Tensor mS = make_tensor(make_gmem_ptr(split_input_ptr),
                          make_shape(Int<kTileV>{}, num_head_q, num_seq_q, max_splitk, num_batch));
  // LSE: (Hg, Sq, Hkv, C, B)
  Tensor mL = make_tensor(make_gmem_ptr(lse_ptr), make_shape(pad_heads_per_group, num_seq_q,
                                                             num_head_k, max_splitk, num_batch));
  // Output: (Dv, Hq, Sq, B)
  Tensor mY = make_tensor(make_gmem_ptr(y_ptr),
                          make_shape(Int<kTileV>{}, num_head_q, num_seq_q, num_batch));
  // NumChunks: (B, Hkv), task_map_ptr[0] = num_tile_per_cta + 1
  Tensor mC = make_tensor(
      make_gmem_ptr(task_map_ptr + (task_map_ptr[0] * num_total_ctas + 1) * kTaskStride),
      make_shape(num_batch, num_head_k));

  extern __shared__ float s_weight_pool[];
  __shared__ float s_out_partial[kWarpCount][32];
  __shared__ float2 s_heavy_ml[kWarpCount];

  cudaGridDependencySynchronize();

  const int max_num_chunks = task_map_ptr[5];
  const bool has_heavy = max_num_chunks > kHeavyThresh;

  // ---- Phase 1: light slots. One warp per slot. ----
  {
    constexpr int kDimPerThread = 4;  // 4 * fp32 -> 128-bit load
    static_assert(kTileV == 32 * kDimPerThread, "light phase covers full kTileV per warp");

    using TiledCopyLoad = decltype(make_tiled_copy(
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, float>{}, Layout<Shape<_32, _1>>{},
        Layout<Shape<Int<kDimPerThread>, _1>>{}));
    using TiledCopyStore =
        decltype(make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, T>{},
                                 Layout<Shape<_32>>{}, Layout<Shape<Int<kDimPerThread>>>{}));

    TiledCopyLoad tiled_copy_load;
    TiledCopyStore tiled_copy_store;

    auto thr_copy_load = tiled_copy_load.get_slice(ilane);
    auto thr_copy_store = tiled_copy_store.get_slice(ilane);

    const int num_slots = num_head_q * num_seq_q * num_batch;
    auto slot_shape = make_shape(num_head_q, num_seq_q, num_batch);

    for (int islot = blockIdx.x * kWarpCount + iwarp; islot < num_slots;
         islot += gridDim.x * kWarpCount) {
      auto crd = idx2crd(islot, slot_shape);
      const int ihead_q = get<0>(crd);
      const int iseq = get<1>(crd);
      const int ibatch = get<2>(crd);

      int ihead_kv, ihead;
      heads_per_group_divider(ihead_kv, ihead, ihead_q);

      const int num_chunks = mC(ibatch, ihead_kv);
      if (num_chunks > kHeavyThresh) {
        continue;
      }

      Tensor gS = mS(_, ihead_q, iseq, _, ibatch);
      Tensor gL = mL(ihead, iseq, ihead_kv, _, ibatch);
      Tensor gY = mY(_, ihead_q, iseq, ibatch);

      Tensor tSg = thr_copy_load.partition_S(gS);
      Tensor tYg = thr_copy_store.partition_D(gY);

      Tensor tSr = make_tensor<float>(make_shape(size<0>(tSg), size<1>(tSg), Int<kPrefetch>{}));
      Tensor tYr = make_fragment_like(tSg(_, _, _0{}));
      Tensor tYr_bf16 = make_fragment_like(tYg);

      Tensor sWr =
          make_tensor(make_smem_ptr(s_weight_pool + iwarp * kHeavyThresh), make_shape(num_chunks));
      float rescale;

      prefetch_chunks<kPrefetch>(tiled_copy_load, tSg, tSr, num_chunks);
      const float inv_sum = softmax_weights<1>(gL, sWr, num_chunks, 0, ilane, s_heavy_ml, &rescale);
      __syncwarp();
      accumulate_chunks<kPrefetch>(tiled_copy_load, tSg, tSr, tYr, sWr, num_chunks);
      store_row(tiled_copy_store, tYg, tYr_bf16, tYr, inv_sum);
    }
  }

  // ---- Phase 2: heavy slots. Block cooperates per dim-tile. ----
  if (has_heavy) {
    __syncthreads();
    constexpr int kDimPerThread = 1;  // 1 * fp32 -> 32-bit load
    constexpr int kDimPerWarp = 32 * kDimPerThread;
    static_assert(kTileV % kDimPerWarp == 0, "kTileV must be divisible by heavy dim-tile");
    constexpr int kDimTiles = kTileV / kDimPerWarp;

    Tensor sY = make_tensor(make_smem_ptr(reinterpret_cast<float *>(s_out_partial)),
                            make_shape(Int<kDimPerWarp>{}, Int<kWarpCount>{}));

    using TiledCopyLoad = decltype(make_tiled_copy(
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, float>{}, Layout<Shape<_32, _1>>{},
        Layout<Shape<Int<kDimPerThread>, _1>>{}));
    using TiledCopyStore =
        decltype(make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, T>{},
                                 Layout<Shape<_32>>{}, Layout<Shape<Int<kDimPerThread>>>{}));

    TiledCopyLoad tiled_copy_load;
    TiledCopyStore tiled_copy_store;

    auto thr_copy_load = tiled_copy_load.get_slice(ilane);
    auto thr_copy_store = tiled_copy_store.get_slice(ilane);

    const int num_slots = kDimTiles * num_head_q * num_seq_q * num_batch;
    auto slot_shape = make_shape(Int<kDimTiles>{}, num_head_q, num_seq_q, num_batch);
    for (int islot = blockIdx.x; islot < num_slots; islot += gridDim.x) {
      auto crd = idx2crd(islot, slot_shape);
      const int idimtile = get<0>(crd);
      const int ihead_q = get<1>(crd);
      const int iseq = get<2>(crd);
      const int ibatch = get<3>(crd);

      int ihead_kv, ihead;
      heads_per_group_divider(ihead_kv, ihead, ihead_q);

      const int num_chunks = mC(ibatch, ihead_kv);
      if (num_chunks <= kHeavyThresh) {
        continue;
      }

      const int stripe_len = iwarp < num_chunks ? (num_chunks - 1 - iwarp) / kWarpCount + 1 : 0;
      Tensor gL = mL(ihead, iseq, ihead_kv, _, ibatch);
      Tensor gS = make_tensor(
          mS(_, ihead_q, iseq, _, ibatch).data() + idimtile * kDimPerWarp + iwarp * stride<3>(mS),
          make_shape(Int<kDimPerWarp>{}, stripe_len),
          make_stride(_1{}, Int<kWarpCount>{} * stride<3>(mS)));
      Tensor gY = make_tensor(mY(_, ihead_q, iseq, ibatch).data() + idimtile * kDimPerWarp,
                              make_shape(Int<kDimPerWarp>{}));

      Tensor tSg = thr_copy_load.partition_S(gS);
      Tensor tSr = make_tensor<float>(make_shape(size<0>(tSg), size<1>(tSg), Int<kPrefetch>{}));
      Tensor tYr = make_fragment_like(tSg(_, _, _0{}));

      Tensor sWrow = make_tensor(make_smem_ptr(s_weight_pool), make_shape(num_chunks));
      Tensor sWg = make_tensor(make_smem_ptr(s_weight_pool + iwarp), make_shape(stripe_len),
                               make_stride(Int<kWarpCount>{}));
      float rescale;

      prefetch_chunks<kPrefetch>(tiled_copy_load, tSg, tSr, stripe_len);
      const float inv_sum =
          softmax_weights<kWarpCount>(gL, sWrow, num_chunks, iwarp, ilane, s_heavy_ml, &rescale);
      accumulate_chunks<kPrefetch>(tiled_copy_load, tSg, tSr, tYr, sWg, stripe_len);

#pragma unroll
      for (int i = 0; i < kDimPerThread; i++) {
        sY(ilane * kDimPerThread + i, iwarp) = tYr(i) * rescale;
      }
      __syncthreads();

      if (iwarp == 0) {
        clear(tYr);
#pragma unroll
        for (int w = 0; w < kWarpCount; w++) {
#pragma unroll
          for (int i = 0; i < kDimPerThread; i++) {
            tYr(i) += sY(ilane * kDimPerThread + i, w);
          }
        }
        Tensor tYg = thr_copy_store.partition_D(gY);
        Tensor tYr_bf16 = make_fragment_like(tYg);
        store_row(tiled_copy_store, tYg, tYr_bf16, tYr, inv_sum);
      }
      __syncthreads();
    }
  }

  cudaTriggerProgrammaticLaunchCompletion();
}

}  // namespace kernels
}  // namespace decode
}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_DECODE_SPLITK_ADAPTIVE_COMBINE_KERNELS_CUH_
