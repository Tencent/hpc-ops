// Copyright (C) 2026 Tencent.

#include <cooperative_groups.h>
#include <cuda.h>

#include <algorithm>

#include "src/topk/topk.h"
#include "src/topk/topk_filtered_boundary.cuh"
#include "src/utils/utils.h"

__device__ __forceinline__ void st_global_s32(int32_t* p, int32_t v) {
  asm volatile("st.global.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

namespace hpc {
namespace topk {
namespace filtered {

constexpr int RADIX = 2048;  // 11-bit fp16 coarse / 11-bit fp32 fine (last round 10-bit)
constexpr int NUM_REFINE_ROUNDS = 3;
// The sampled paths derive a coarse threshold from a fixed systematic view.
constexpr int SAMPLE_STRIDE = 64;
constexpr int SAMPLE_RANK = 48;
constexpr int WIDE_SAMPLE_STRIDE = 128;
constexpr int WIDE_SAMPLE_RANK = 24;
// Segment sampling, shared by the kv-split and cluster-sampled routes:
// each CTA samples only its own segment, publishes the sample histogram,
// and every CTA of the row reduces all segment histograms into one shared
// coarse boundary. One boundary for the whole row means the candidate
// union is exactly {x >= boundary}, so a candidate count past K
// certifies the top-K.
constexpr int SEG_SAMPLE_STRIDE = 16;
constexpr double SEG_SAMPLE_MARGIN = 1.0;
constexpr double SEG_SAMPLE_ZSCORE = 3.0;
constexpr int kSegSampleStride = SEG_SAMPLE_STRIDE;
// Smallest sample rank r with stride*(r - Z*sqrt(r)) >= MARGIN*K: with
// Z-score sampling slack the r-th sample of the global union still
// admits at least MARGIN*K elements, so the derived boundary never
// undershoots the top-K cut.
constexpr int seg_sample_rank(int top_k) {
  const double y = SEG_SAMPLE_MARGIN * static_cast<double>(top_k) / kSegSampleStride;
  for (int r = 1; r < RADIX; ++r) {
    const double d = r - y;
    if (d >= 0.0 && d * d >= SEG_SAMPLE_ZSCORE * SEG_SAMPLE_ZSCORE * r) {
      return r;
    }
  }
  return RADIX - 1;
}

// Bit layouts of the three fp32-ordered fine refinement rounds.
//   round 0 -> bits [21..31]  (top 11 bits)
//   round 1 -> bits [10..20]  (middle 11 bits)
//   round 2 -> bits [0..9]    (low 10 bits, back-fill round)
// 1024-thread CTAs, two resident per SM (2048 threads = 64 warps).
__device__ constexpr int kFineShifts[NUM_REFINE_ROUNDS] = {21, 10, 0};
__device__ constexpr uint32_t kFineMasks[NUM_REFINE_ROUNDS] = {0x7FFu, 0x7FFu, 0x3FFu};
constexpr int THREADS = 1024;
constexpr int MIN_BLOCKS = 2;
// smem candidate capacity per ping-pong slot. The sampled path keeps
// twice as many because its predicted threshold intentionally admits
// more than top_k.
constexpr int COARSE_CANDIDATE_CAPACITY = 2048;
constexpr int SAMPLED_CANDIDATE_CAPACITY = 4096;
// Exact coarse-stage bookkeeping: warp-aggregated candidate counting
// plus a shortcut for boundary bins that hold a single value.
// Persistent grid: a fixed pool of BLOCKS_PER_SM CTAs per SM pulling
// rows from the work queue. Also sizes the gmem spill region
// ([num_blocks, 2, n]).
constexpr int BLOCKS_PER_SM = 3;
// Persistent grid size (num_sms * BLOCKS_PER_SM). Shared by the launch
// and the workspace sizing so the spill buffer always has exactly
// gridDim.x rows.
// The low/high halves hold the descending coarse keys for x/y.
inline int grid_num_blocks() { return get_sm_count() * BLOCKS_PER_SM; }
__device__ __forceinline__ uint32_t to_coarse_key2(float x, float y) {
  return static_cast<uint32_t>(to_coarse_key(x)) | (static_cast<uint32_t>(to_coarse_key(y)) << 16);
}

__device__ __forceinline__ uint32_t to_ordered(float x) {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? bits : ((bits ^ 0xFFFFFFFFu) & 0x7FFFFFFFu);
}

// Inverse of to_ordered: recovers the float bits from an ordered key.
__device__ __forceinline__ float to_ordered_inverse(uint32_t key) {
  return __uint_as_float((key & 0x80000000u) ? key : ((~key) & 0x7FFFFFFFu));
}

__device__ __forceinline__ float4 ld_float4(const float* p) {
  float4 v;
  asm volatile("ld.global.v4.f32 {%0,%1,%2,%3}, [%4];\n"
               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
               : "l"(p));
  return v;
}

constexpr int HIST_SWEEP = RADIX;
constexpr int SAMPLE_TILE_BATCH = 4;
constexpr int SAMPLE_TILE_ELEMS = 32;
// Elements covered by one warp tile of the sample / 32-lane loads per
// tile.
constexpr int SAMPLE_TILE_LOADS = SAMPLE_TILE_ELEMS / 32;
// Per-warp partial sums feeding the block-wide histogram prefix scan.
template <int kBlockThreads>
struct PrefixStorage {
  int warp_sum[kBlockThreads / 32];
};

// Shared-memory layout for one row. The sampled path keeps more
// candidates because its predicted threshold intentionally admits
// more than K.
template <int kTopK, bool kSampled, int kBlockThreads = THREADS>
struct RowSmem {
  static constexpr int CANDIDATE_CAPACITY =
      kSampled ? SAMPLED_CANDIDATE_CAPACITY : COARSE_CANDIDATE_CAPACITY;
  int histogram[RADIX + 1];
  int threshold_bin;
  int num_selected;
  int ties_remaining;
  int num_cand[2];
  int num_spilled_cand[2];
  int32_t indices[kTopK];
  alignas(8) int32_t cand_idx[2][CANDIDATE_CAPACITY];
  PrefixStorage<kBlockThreads> prefix;
};

struct RowCandidate {
  int32_t index;
  uint32_t key;
};

template <int kTopK, int kBlockThreads>
struct RowLocalSmem {
  static constexpr int CANDIDATE_CAPACITY = kBlockThreads;
  int histogram[RADIX + 1];
  int threshold_bin;
  int threshold_above;
  int threshold_count;
  int num_selected;
  int ties_remaining;
  int num_candidates;
  int32_t indices[kTopK];
  alignas(8) RowCandidate candidates[CANDIDATE_CAPACITY];
  PrefixStorage<kBlockThreads> prefix;
};

// Cluster8 reuses the candidate allocation as aligned (index, FP32
// bits) pairs.
__device__ __forceinline__ uint64_t pack_cluster_candidate(int32_t index, uint32_t key) {
  return static_cast<uint64_t>(static_cast<uint32_t>(index)) | (static_cast<uint64_t>(key) << 32);
}

__device__ __forceinline__ int32_t cluster_candidate_index(uint64_t pair) {
  return static_cast<int32_t>(static_cast<uint32_t>(pair));
}

__device__ __forceinline__ uint32_t cluster_candidate_key(uint64_t pair) {
  return static_cast<uint32_t>(pair >> 32);
}

__device__ __forceinline__ int warp_inclusive_sum(int value) {
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const int upper = __shfl_up_sync(0xFFFFFFFFu, value, offset);
    value += ((threadIdx.x & 31) >= offset) ? upper : 0;
  }
  return value;
}

template <int kBins = RADIX, int kTopK, int kBlockThreads>
__device__ __forceinline__ void find_row_local_threshold(RowLocalSmem<kTopK, kBlockThreads>* s,
                                                         int tidx, int remaining) {
  static_assert(kBlockThreads >= 32 && kBlockThreads <= 1024);
  static_assert((kBlockThreads & 31) == 0);
  static_assert(kBins <= RADIX);
  constexpr int kItems = (kBins + kBlockThreads - 1) / kBlockThreads;
  constexpr int kNumWarps = kBlockThreads / 32;
  int local_sum = 0;
#pragma unroll
  for (int item = 0; item < kItems; ++item) {
    const int bin = tidx * kItems + item;
    if (bin < kBins) {
      local_sum += s->histogram[bin];
    }
  }
  const int lane = tidx & 31;
  const int warp = tidx >> 5;
  const int warp_inclusive = warp_inclusive_sum(local_sum);
  const int warp_exclusive = warp_inclusive - local_sum;
  if (lane == 31) {
    s->prefix.warp_sum[warp] = warp_inclusive;
  }
  __syncthreads();
  const int preceding_warp_value = lane < kNumWarps && lane < warp ? s->prefix.warp_sum[lane] : 0;
  const int preceding_warps = __reduce_add_sync(0xFFFFFFFFu, preceding_warp_value);
  int prefix = preceding_warps + warp_exclusive;
#pragma unroll
  for (int item = 0; item < kItems; ++item) {
    const int bin = tidx * kItems + item;
    if (bin >= kBins) {
      continue;
    }
    const int value = s->histogram[bin];
    const int above = prefix;
    prefix += value;
    if (above <= remaining && prefix > remaining) {
      s->threshold_bin = bin;
      s->threshold_above = above;
      s->threshold_count = value;
    }
  }
  __syncthreads();
}

template <int kTopK, bool kSampled, int kSecondaryRemaining = -1, int kBlockThreads = THREADS,
          bool kResetState = true>
__device__ __forceinline__ void prefix_sum_find_threshold(
    RowSmem<kTopK, kSampled, kBlockThreads>* s, int tidx, int remaining, int next_idx,
    int* reset_target, int reset_value) {
  static_assert(kBlockThreads >= 32 && kBlockThreads <= 1024);
  static_assert((kBlockThreads & 31) == 0);
  static_assert(RADIX % kBlockThreads == 0);
  constexpr int kItems = RADIX / kBlockThreads;
  constexpr int kNumWarps = kBlockThreads / 32;
  int values[kItems];
  int local_sum = 0;
#pragma unroll
  for (int item = 0; item < kItems; ++item) {
    const int value = s->histogram[tidx * kItems + item];
    values[item] = value;
    local_sum += value;
  }
  if constexpr (kResetState) {
    if (tidx == 0) {
      s->num_cand[next_idx] = 0;
      s->num_spilled_cand[next_idx] = 0;
      *reset_target = reset_value;
    }
  }
  const int lane = tidx & 31;
  const int warp = tidx >> 5;
  const int warp_inclusive = warp_inclusive_sum(local_sum);
  const int warp_exclusive = warp_inclusive - local_sum;
  if (lane == 31) {
    s->prefix.warp_sum[warp] = warp_inclusive;
  }
  __syncthreads();
  const int preceding_warp_value = lane < kNumWarps && lane < warp ? s->prefix.warp_sum[lane] : 0;
  const int preceding_warps = __reduce_add_sync(0xFFFFFFFFu, preceding_warp_value);
  int prefix = preceding_warps + warp_exclusive;
  int found_bin = -1;
  int secondary_bin = -1;
#pragma unroll
  for (int item = 0; item < kItems; ++item) {
    const int ibin = tidx * kItems + item;
    const int exclusive = prefix;
    prefix += values[item];
    s->histogram[ibin] = prefix;
    if (found_bin < 0 && exclusive <= remaining && prefix > remaining) {
      found_bin = ibin;
    }
    if constexpr (kSecondaryRemaining >= 0) {
      if (secondary_bin < 0 && exclusive <= kSecondaryRemaining && prefix > kSecondaryRemaining) {
        secondary_bin = ibin;
      }
    }
  }
  if (found_bin >= 0) {
    s->threshold_bin = found_bin;
  }
  if constexpr (kSecondaryRemaining >= 0) {
    if (secondary_bin >= 0) {
      s->ties_remaining = secondary_bin;
    }
  }
  __syncthreads();
}

template <int kTopK, int kBlockThreads = THREADS>
__device__ __noinline__ void refine_resident_keys(RowSmem<kTopK, false, kBlockThreads>* s,
                                                  int topk_remaining, int tidx);
// Persistent kernel: resident CTAs claim rows from the work queue and
// run the full coarse/refine pipeline per row; the sampled variant
// derives its coarse threshold from the systematic sample instead of
// an exact histogram.
template <uint32_t VEC, int kTopK, bool kSampled, int kSampleStride = SAMPLE_STRIDE,
          int kSampleRank = SAMPLE_RANK>
__global__ void __launch_bounds__(THREADS, MIN_BLOCKS)
    topk_filtered_kernel(const float* __restrict__ input, int32_t* __restrict__ output,
                         const int32_t* __restrict__ ke, int32_t* __restrict__ extra_buffer,
                         int32_t* __restrict__ work_counter, int32_t* __restrict__ exit_counter,
                         const int32_t* __restrict__ num_valid_rows, uint32_t row_stride,
                         uint32_t out_stride, uint32_t n_cols, int32_t row_capacity) {
  using Smem = RowSmem<kTopK, kSampled>;
  const int tidx = threadIdx.x;
  const int num_valid = *num_valid_rows;
  if (num_valid <= 0) {
    return;
  }
  int32_t* buf0 = extra_buffer + static_cast<size_t>(blockIdx.x) * 2 * n_cols;
  int32_t* buf1 = buf0 + n_cols;
  int32_t* buf[2] = {buf0, buf1};
  extern __shared__ uint8_t smem_raw[];
  Smem* s = reinterpret_cast<Smem*>(smem_raw);
  __shared__ int shared_row;
  __shared__ float shared_coarse_boundary[2];
  bool first_wave = true;
  while (true) {
    if (tidx == 0) {
      if (first_wave) {
        shared_row = static_cast<int>(blockIdx.x);
      } else if (row_capacity <= static_cast<int>(gridDim.x)) {
        shared_row = num_valid;
      } else {
        shared_row = static_cast<int>(gridDim.x) + atomicAdd(work_counter, 1);
      }
    }
    __syncthreads();
    first_wave = false;
    const int row_i = shared_row;
    if (row_i >= num_valid) {
      if (row_capacity <= static_cast<int>(gridDim.x)) {
        break;
      }
      if (tidx == 0 && atomicAdd(exit_counter, 1) == gridDim.x - 1) {
        *work_counter = 0;
        *exit_counter = 0;
      }
      break;
    }
    const uint32_t row = static_cast<uint32_t>(row_i);
    const float* score = input + static_cast<size_t>(row) * row_stride;
    int32_t* dst = output + static_cast<size_t>(row) * out_stride;
    const int length = ke[row];
    if (length <= kTopK) {
      for (int i = tidx; i < kTopK; i += THREADS) {
        dst[i] = (i < length) ? i : -1;
      }
      continue;
    }
    int topk_remaining = kTopK;
    bool need_refine = false;
    bool sampled_ready = false;
    int sampled_threshold = -1;
    if constexpr (kSampled) {
      static_assert((kSampleStride & (kSampleStride - 1)) == 0,
                    "sample stride must be a power of two");
      constexpr bool kUseSecondaryThreshold = kSampleStride == WIDE_SAMPLE_STRIDE;
      constexpr int kSecondarySampleRank = kSampleRank + 4;
      constexpr int kRequiredSampleRank =
          kUseSecondaryThreshold ? kSecondarySampleRank : kSampleRank;
      if (length >= (kRequiredSampleRank + 1) * kSampleStride) {
        for (int i = tidx; i < RADIX + 1; i += THREADS) {
          s->histogram[i] = 0;
        }
        if (tidx == 0) {
          s->num_selected = 0;
        }
        __syncthreads();
        const int warp_id = tidx >> 5;
        const int lane_id = tidx & 31;
        constexpr int kSampleWarps = THREADS >> 5;
        constexpr int kSampleTile = SAMPLE_TILE_ELEMS * kSampleStride;
        const int sample_offset =
            static_cast<int>((row * 17u + 13u) & (kSampleStride - 1)) * SAMPLE_TILE_ELEMS;
        const int sample_step = kSampleWarps * kSampleTile;
        int tile = sample_offset + warp_id * kSampleTile;
        constexpr int kBatchRegs = SAMPLE_TILE_BATCH * SAMPLE_TILE_LOADS;
        for (; tile + (SAMPLE_TILE_BATCH - 1) * sample_step + SAMPLE_TILE_ELEMS <= length;
             tile += SAMPLE_TILE_BATCH * sample_step) {
          float v[kBatchRegs];
#pragma unroll
          for (int b = 0; b < SAMPLE_TILE_BATCH; ++b) {
#pragma unroll
            for (int t = 0; t < SAMPLE_TILE_LOADS; ++t) {
              v[b * SAMPLE_TILE_LOADS + t] = score[tile + b * sample_step + (t << 5) + lane_id];
            }
          }
#pragma unroll
          for (int r = 0; r < kBatchRegs; ++r) {
            atomicAdd(&s->histogram[to_coarse_key(v[r])], 1);
          }
        }
        for (; tile < length; tile += sample_step) {
#pragma unroll
          for (int t = 0; t < SAMPLE_TILE_LOADS; ++t) {
            int i = tile + (t << 5) + lane_id;
            if (i < length) {
              atomicAdd(&s->histogram[to_coarse_key(score[i])], 1);
            }
          }
        }
        __syncthreads();
        if constexpr (kUseSecondaryThreshold) {
          prefix_sum_find_threshold<kTopK, true, kSecondarySampleRank>(s, tidx, kSampleRank, 0,
                                                                       &s->num_selected, 0);
        } else {
          prefix_sum_find_threshold(s, tidx, kSampleRank, 0, &s->num_selected, 0);
        }
        sampled_threshold = s->threshold_bin;
        int secondary_threshold = kUseSecondaryThreshold ? s->ties_remaining : -1;
        if (tidx == 0) {
          shared_coarse_boundary[0] = coarse_boundary(sampled_threshold).value;
          shared_coarse_boundary[1] = coarse_boundary(secondary_threshold).value;
        }
        __syncthreads();
        const float sampled_boundary = shared_coarse_boundary[0];
        const float secondary_boundary = shared_coarse_boundary[1];
        const uint32_t sampled_boundary_bits = __float_as_uint(sampled_boundary);
        const uint32_t secondary_boundary_bits = __float_as_uint(secondary_boundary);
        for (int i = tidx; i < RADIX + 1; i += THREADS) {
          s->histogram[i] = 0;
        }
        __syncthreads();
        auto add_sampled_candidate = [&](float raw, int i) {
          int pos = atomicAdd(&s->num_cand[0], 1);
          if (pos < Smem::CANDIDATE_CAPACITY) {
            s->cand_idx[0][pos] = i;
          } else {
            int gpos = atomicAdd(&s->num_spilled_cand[0], 1);
            st_global_s32(buf[0] + gpos, i);
          }
          int sub = (to_ordered(raw) >> kFineShifts[0]) & kFineMasks[0];
          atomicAdd(&s->histogram[sub], 1);
        };
        auto sampled_split = [&](float raw, int i) {
          if (fp32_boundary_accepts(raw, sampled_boundary, sampled_boundary_bits)) {
            add_sampled_candidate(raw, i);
          }
        };
        if constexpr (VEC == 4) {
          const int n8 = length >> 3;
          const float4* v4 = reinterpret_cast<const float4*>(score);
          if constexpr (kSampleStride == WIDE_SAMPLE_STRIDE) {
            int g = tidx;
            float4 next_a;
            float4 next_b;
            if (g < n8) {
              next_a = ld_float4(reinterpret_cast<const float*>(v4 + 2 * g));
              next_b = ld_float4(reinterpret_cast<const float*>(v4 + 2 * g + 1));
            }
            while (g < n8) {
              const float4 current_a = next_a;
              const float4 current_b = next_b;
              const int base = g << 3;
              g += THREADS;
              if (g < n8) {
                next_a = ld_float4(reinterpret_cast<const float*>(v4 + 2 * g));
                next_b = ld_float4(reinterpret_cast<const float*>(v4 + 2 * g + 1));
              }
              sampled_split(current_a.x, base);
              sampled_split(current_a.y, base + 1);
              sampled_split(current_a.z, base + 2);
              sampled_split(current_a.w, base + 3);
              sampled_split(current_b.x, base + 4);
              sampled_split(current_b.y, base + 5);
              sampled_split(current_b.z, base + 6);
              sampled_split(current_b.w, base + 7);
            }
          } else {
            for (int g = tidx; g < n8; g += THREADS) {
              float4 a = ld_float4(reinterpret_cast<const float*>(v4 + 2 * g));
              float4 b = ld_float4(reinterpret_cast<const float*>(v4 + 2 * g + 1));
              const int base = g << 3;
              sampled_split(a.x, base);
              sampled_split(a.y, base + 1);
              sampled_split(a.z, base + 2);
              sampled_split(a.w, base + 3);
              sampled_split(b.x, base + 4);
              sampled_split(b.y, base + 5);
              sampled_split(b.z, base + 6);
              sampled_split(b.w, base + 7);
            }
          }
          for (int i = (n8 << 3) + tidx; i < length; i += THREADS) {
            float raw = score[i];
            sampled_split(raw, i);
          }
        } else {
          for (int i = tidx; i < length; i += THREADS) {
            float raw = score[i];
            sampled_split(raw, i);
          }
        }
        __syncthreads();
        auto use_sampled_candidates = [&]() {
          int count = s->num_cand[0];
          if (count > kTopK) {
            topk_remaining = kTopK;
            need_refine = true;
            return true;
          }
          if constexpr (kUseSecondaryThreshold) {
            if (count == kTopK) {
              const int n_smem = min(count, Smem::CANDIDATE_CAPACITY);
              for (int i = tidx; i < n_smem; i += THREADS) {
                s->indices[i] = s->cand_idx[0][i];
              }
              const int g_num = s->num_spilled_cand[0];
              for (int i = tidx; i < g_num; i += THREADS) {
                s->indices[n_smem + i] = buf[0][i];
              }
              __syncthreads();
              return true;
            }
          }
          return false;
        };
        sampled_ready = use_sampled_candidates();
        if constexpr (kUseSecondaryThreshold) {
          if (!sampled_ready && secondary_threshold > sampled_threshold) {
            auto append_secondary_band = [&](float raw, int i) {
              const bool in_sample =
                  fp32_boundary_accepts(raw, sampled_boundary, sampled_boundary_bits);
              const bool in_secondary =
                  fp32_boundary_accepts(raw, secondary_boundary, secondary_boundary_bits);
              if (!in_sample && in_secondary) {
                add_sampled_candidate(raw, i);
              }
            };
            if constexpr (VEC == 4) {
              const int n8 = length >> 3;
              const float4* v4 = reinterpret_cast<const float4*>(score);
              if constexpr (kSampleStride == WIDE_SAMPLE_STRIDE) {
                int g = tidx;
                float4 next_a;
                float4 next_b;
                if (g < n8) {
                  next_a = ld_float4(reinterpret_cast<const float*>(v4 + 2 * g));
                  next_b = ld_float4(reinterpret_cast<const float*>(v4 + 2 * g + 1));
                }
                while (g < n8) {
                  const float4 current_a = next_a;
                  const float4 current_b = next_b;
                  const int base = g << 3;
                  g += THREADS;
                  if (g < n8) {
                    next_a = ld_float4(reinterpret_cast<const float*>(v4 + 2 * g));
                    next_b = ld_float4(reinterpret_cast<const float*>(v4 + 2 * g + 1));
                  }
                  append_secondary_band(current_a.x, base);
                  append_secondary_band(current_a.y, base + 1);
                  append_secondary_band(current_a.z, base + 2);
                  append_secondary_band(current_a.w, base + 3);
                  append_secondary_band(current_b.x, base + 4);
                  append_secondary_band(current_b.y, base + 5);
                  append_secondary_band(current_b.z, base + 6);
                  append_secondary_band(current_b.w, base + 7);
                }
              } else {
                for (int g = tidx; g < n8; g += THREADS) {
                  float4 a = ld_float4(reinterpret_cast<const float*>(v4 + 2 * g));
                  float4 b = ld_float4(reinterpret_cast<const float*>(v4 + 2 * g + 1));
                  const int base = g << 3;
                  append_secondary_band(a.x, base);
                  append_secondary_band(a.y, base + 1);
                  append_secondary_band(a.z, base + 2);
                  append_secondary_band(a.w, base + 3);
                  append_secondary_band(b.x, base + 4);
                  append_secondary_band(b.y, base + 5);
                  append_secondary_band(b.z, base + 6);
                  append_secondary_band(b.w, base + 7);
                }
              }
              for (int i = (n8 << 3) + tidx; i < length; i += THREADS) {
                float raw = score[i];
                append_secondary_band(raw, i);
              }
            } else {
              for (int i = tidx; i < length; i += THREADS) {
                float raw = score[i];
                append_secondary_band(raw, i);
              }
            }
            __syncthreads();
            sampled_ready = use_sampled_candidates();
          }
        }
      }
    }
    if (!sampled_ready) {
      topk_remaining = kTopK;
      for (int i = tidx; i < RADIX + 1; i += THREADS) {
        s->histogram[i] = 0;
      }
      if (tidx == 0) {
        s->num_selected = 0;
      }
      __syncthreads();
      auto add_coarse = [&](float raw) { atomicAdd(&s->histogram[to_coarse_key(raw)], 1); };
      if constexpr (VEC == 4) {
        const int n8 = length >> 3;
        const float4* v4 = reinterpret_cast<const float4*>(score);
        for (int g = tidx; g < n8; g += THREADS) {
          float4 a = ld_float4(reinterpret_cast<const float*>(v4 + 2 * g));
          float4 b = ld_float4(reinterpret_cast<const float*>(v4 + 2 * g + 1));
          add_coarse(a.x);
          add_coarse(a.y);
          add_coarse(a.z);
          add_coarse(a.w);
          add_coarse(b.x);
          add_coarse(b.y);
          add_coarse(b.z);
          add_coarse(b.w);
        }
        for (int i = (n8 << 3) + tidx; i < length; i += THREADS) {
          add_coarse(score[i]);
        }
      } else {
        for (int i = tidx; i < length; i += THREADS) {
          add_coarse(score[i]);
        }
      }
      __syncthreads();
      prefix_sum_find_threshold(s, tidx, topk_remaining, 0, &s->num_selected, 0);
      int threshold_bin = s->threshold_bin;
      if (threshold_bin > 0) {
        topk_remaining -= s->histogram[threshold_bin - 1];
      }
      __syncthreads();
      if (topk_remaining == 0) {
        const PackedCoarseBoundary select_boundary =
            pack_coarse_boundary(coarse_boundary(threshold_bin - 1));
        auto collect = [&](float raw, int i) {
          if (coarse_accepts(raw, select_boundary)) {
            int pos = atomicAdd(&s->num_selected, 1);
            s->indices[pos] = i;
          }
        };
        if constexpr (VEC == 4) {
          const int n4 = length >> 2;
          const float4* v4 = reinterpret_cast<const float4*>(score);
          for (int g = tidx; g < n4; g += THREADS) {
            float4 v = ld_float4(reinterpret_cast<const float*>(v4 + g));
            int base = g << 2;
            collect(v.x, base);
            collect(v.y, base + 1);
            collect(v.z, base + 2);
            collect(v.w, base + 3);
          }
          for (int i = (n4 << 2) + tidx; i < length; i += THREADS) {
            collect(score[i], i);
          }
        } else {
          for (int i = tidx; i < length; i += THREADS) {
            collect(score[i], i);
          }
        }
        __syncthreads();
      } else {
        for (int i = tidx; i < RADIX + 1; i += THREADS) {
          s->histogram[i] = 0;
        }
        __syncthreads();
        const PackedCoarseBoundary select_boundary =
            pack_coarse_boundary(coarse_boundary(threshold_bin - 1));
        const PackedCoarseBoundary candidate_boundary =
            pack_coarse_boundary(coarse_boundary(threshold_bin));
        auto split = [&](float raw, int i) {
          const CoarseClass coarse_class =
              coarse_classify(raw, select_boundary, candidate_boundary);
          if (coarse_class == CoarseClass::kSelected) {
            int pos = atomicAdd(&s->num_selected, 1);
            s->indices[pos] = i;
          } else if (coarse_class == CoarseClass::kCandidate) {
            int pos = atomicAdd(&s->num_cand[0], 1);
            const uint32_t ordered = to_ordered(raw);
            if (pos < Smem::CANDIDATE_CAPACITY) {
              s->cand_idx[0][pos] = i;
              if constexpr (!kSampled) {
                s->cand_idx[1][pos] = static_cast<int32_t>(ordered);
              }
            } else {
              int gpos = atomicAdd(&s->num_spilled_cand[0], 1);
              st_global_s32(buf[0] + gpos, i);
            }
            int sub = (ordered >> kFineShifts[0]) & kFineMasks[0];
            atomicAdd(&s->histogram[sub], 1);
          }
        };
        if constexpr (VEC == 4) {
          const int n8 = length >> 3;
          const float4* v4 = reinterpret_cast<const float4*>(score);
          for (int g = tidx; g < n8; g += THREADS) {
            float4 a = ld_float4(reinterpret_cast<const float*>(v4 + 2 * g));
            float4 b = ld_float4(reinterpret_cast<const float*>(v4 + 2 * g + 1));
            int base = g << 3;
            split(a.x, base);
            split(a.y, base + 1);
            split(a.z, base + 2);
            split(a.w, base + 3);
            split(b.x, base + 4);
            split(b.y, base + 5);
            split(b.z, base + 6);
            split(b.w, base + 7);
          }
          for (int i = (n8 << 3) + tidx; i < length; i += THREADS) {
            split(score[i], i);
          }
        } else {
          for (int i = tidx; i < length; i += THREADS) {
            split(score[i], i);
          }
        }
        __syncthreads();
        need_refine = true;
      }
    }
    if (need_refine) {
      bool resident_refined = false;
      if constexpr (!kSampled) {
        if (s->num_spilled_cand[0] == 0) {
          refine_resident_keys(s, topk_remaining, tidx);
          resident_refined = true;
        }
      }
      if (!resident_refined) {
        bool run = true;
        for (int round = 0; round < NUM_REFINE_ROUNDS; ++round) {
          if (!run) {
            break;
          }
          int r = round & 1;
          prefix_sum_find_threshold(s, tidx, topk_remaining, r ^ 1, &s->ties_remaining,
                                    topk_remaining);
          __syncthreads();
          int num_input = min(s->num_cand[r], Smem::CANDIDATE_CAPACITY);
          int g_num = s->num_spilled_cand[r];
          int threshold = s->threshold_bin;
          if (threshold > 0) {
            topk_remaining -= s->histogram[threshold - 1];
          }
          if (tidx == 0) {
            s->ties_remaining = topk_remaining;
          }
          __syncthreads();
          int cur_shift = kFineShifts[round];
          uint32_t cur_mask = kFineMasks[round];
          bool is_last = (round == NUM_REFINE_ROUNDS - 1);
          int next_shift = is_last ? 0 : kFineShifts[round + 1];
          uint32_t next_mask = is_last ? 0 : kFineMasks[round + 1];
          if (topk_remaining == 0) {
            for (int i = tidx; i < num_input; i += THREADS) {
              int idx = s->cand_idx[r][i];
              int b = (to_ordered(score[idx]) >> cur_shift) & cur_mask;
              if (b < threshold) {
                int pos = atomicAdd(&s->num_selected, 1);
                s->indices[pos] = idx;
              }
            }
            for (int i = tidx; i < g_num; i += THREADS) {
              int idx = buf[r][i];
              int b = (to_ordered(score[idx]) >> cur_shift) & cur_mask;
              if (b < threshold) {
                int pos = atomicAdd(&s->num_selected, 1);
                s->indices[pos] = idx;
              }
            }
            __syncthreads();
            run = false;
          } else {
            for (int i = tidx; i < RADIX + 1; i += THREADS) {
              s->histogram[i] = 0;
            }
            __syncthreads();
            for (int i = tidx; i < num_input; i += THREADS) {
              int idx = s->cand_idx[r][i];
              float raw = score[idx];
              uint32_t bits = to_ordered(raw);
              int b = (bits >> cur_shift) & cur_mask;
              if (b < threshold) {
                int pos = atomicAdd(&s->num_selected, 1);
                s->indices[pos] = idx;
              } else if (b == threshold) {
                if (is_last) {
                  int cur = atomicAdd(&s->ties_remaining, -1);
                  if (cur > 0) {
                    s->indices[kTopK - cur] = idx;
                  }
                } else {
                  int cur = atomicAdd(&s->num_cand[r ^ 1], 1);
                  if (cur < Smem::CANDIDATE_CAPACITY) {
                    s->cand_idx[r ^ 1][cur] = idx;
                  } else {
                    int gpos = atomicAdd(&s->num_spilled_cand[r ^ 1], 1);
                    st_global_s32(buf[r ^ 1] + gpos, idx);
                  }
                  int sub = (bits >> next_shift) & next_mask;
                  atomicAdd(&s->histogram[sub], 1);
                }
              }
            }
            for (int i = tidx; i < g_num; i += THREADS) {
              int idx = buf[r][i];
              float raw = score[idx];
              uint32_t bits = to_ordered(raw);
              int b = (bits >> cur_shift) & cur_mask;
              if (b < threshold) {
                int pos = atomicAdd(&s->num_selected, 1);
                s->indices[pos] = idx;
              } else if (b == threshold) {
                if (is_last) {
                  int cur = atomicAdd(&s->ties_remaining, -1);
                  if (cur > 0) {
                    s->indices[kTopK - cur] = idx;
                  }
                } else {
                  int cur = atomicAdd(&s->num_cand[r ^ 1], 1);
                  if (cur < Smem::CANDIDATE_CAPACITY) {
                    s->cand_idx[r ^ 1][cur] = idx;
                  } else {
                    int gpos = atomicAdd(&s->num_spilled_cand[r ^ 1], 1);
                    st_global_s32(buf[r ^ 1] + gpos, idx);
                  }
                  int sub = (bits >> next_shift) & next_mask;
                  atomicAdd(&s->histogram[sub], 1);
                }
              }
            }
            __syncthreads();
          }
        }
      }
    }
    for (int i = tidx; i < kTopK; i += THREADS) {
      dst[i] = static_cast<int32_t>(s->indices[i]);
    }
  }
}

template <int kBlockThreads>
__device__ __forceinline__ bool block_key_range_single(uint32_t lo, uint32_t hi, uint32_t* sh_lo,
                                                       uint32_t* sh_hi, int tidx) {
  if (tidx == 0) {
    *sh_lo = 0xFFFFFFFFu;
    *sh_hi = 0u;
  }
  __syncthreads();
  lo = __reduce_min_sync(0xFFFFFFFFu, lo);
  hi = __reduce_max_sync(0xFFFFFFFFu, hi);
  if ((tidx & 31) == 0 && lo <= hi) {
    atomicMin(sh_lo, lo);
    atomicMax(sh_hi, hi);
  }
  __syncthreads();
  const bool single = *sh_lo == *sh_hi;
  __syncthreads();
  return single;
}

template <uint32_t VEC, int kTopK, bool kSampled, int kBlockThreads = THREADS>
__device__ __noinline__ int run_exact_coarse_stage(RowSmem<kTopK, kSampled, kBlockThreads>* s,
                                                   const float* score, int32_t* buf0, int length,
                                                   int tidx) {
  using Smem = RowSmem<kTopK, kSampled, kBlockThreads>;
  int remaining = kTopK;
  for (int i = tidx; i < RADIX + 1; i += kBlockThreads) {
    s->histogram[i] = 0;
  }
  if (tidx == 0) {
    s->num_selected = 0;
  }
  __syncthreads();
  auto add_coarse = [&](float raw) { atomicAdd(&s->histogram[to_coarse_key(raw)], 1); };
  if constexpr (VEC == 4) {
    const int n8 = length >> 3;
    const float4* v4 = reinterpret_cast<const float4*>(score);
    for (int g = tidx; g < n8; g += kBlockThreads) {
      float4 a = ld_float4(reinterpret_cast<const float*>(v4 + 2 * g));
      float4 b = ld_float4(reinterpret_cast<const float*>(v4 + 2 * g + 1));
      add_coarse(a.x);
      add_coarse(a.y);
      add_coarse(a.z);
      add_coarse(a.w);
      add_coarse(b.x);
      add_coarse(b.y);
      add_coarse(b.z);
      add_coarse(b.w);
    }
    for (int i = (n8 << 3) + tidx; i < length; i += kBlockThreads) {
      add_coarse(score[i]);
    }
  } else {
    for (int i = tidx; i < length; i += kBlockThreads) {
      add_coarse(score[i]);
    }
  }
  __syncthreads();
  prefix_sum_find_threshold(s, tidx, remaining, 0, &s->num_selected, 0);
  int threshold_bin = s->threshold_bin;
  if (threshold_bin > 0) {
    remaining -= s->histogram[threshold_bin - 1];
  }
  __syncthreads();
  if (remaining == 0) {
    const PackedCoarseBoundary select_boundary =
        pack_coarse_boundary(coarse_boundary(threshold_bin - 1));
    auto collect = [&](float raw, int i) {
      if (coarse_accepts(raw, select_boundary)) {
        int pos = atomicAdd(&s->num_selected, 1);
        s->indices[pos] = i;
      }
    };
    if constexpr (VEC == 4) {
      const int n4 = length >> 2;
      const float4* v4 = reinterpret_cast<const float4*>(score);
      for (int g = tidx; g < n4; g += kBlockThreads) {
        float4 v = ld_float4(reinterpret_cast<const float*>(v4 + g));
        int base = g << 2;
        collect(v.x, base);
        collect(v.y, base + 1);
        collect(v.z, base + 2);
        collect(v.w, base + 3);
      }
      for (int i = (n4 << 2) + tidx; i < length; i += kBlockThreads) {
        collect(score[i], i);
      }
    } else {
      for (int i = tidx; i < length; i += kBlockThreads) {
        collect(score[i], i);
      }
    }
    __syncthreads();
    return 0;
  }
  for (int i = tidx; i < RADIX + 1; i += kBlockThreads) {
    s->histogram[i] = 0;
  }
  __syncthreads();
  const PackedCoarseBoundary select_boundary =
      pack_coarse_boundary(coarse_boundary(threshold_bin - 1));
  const PackedCoarseBoundary candidate_boundary =
      pack_coarse_boundary(coarse_boundary(threshold_bin));
  auto split = [&](float raw, int i) {
    const CoarseClass coarse_class = coarse_classify(raw, select_boundary, candidate_boundary);
    if (coarse_class == CoarseClass::kSelected) {
      int pos = atomicAdd(&s->num_selected, 1);
      s->indices[pos] = i;
    } else if (coarse_class == CoarseClass::kCandidate) {
      int pos = atomicAdd(&s->num_cand[0], 1);
      const uint32_t ordered = to_ordered(raw);
      if (pos < Smem::CANDIDATE_CAPACITY) {
        s->cand_idx[0][pos] = i;
        if constexpr (!kSampled) {
          s->cand_idx[1][pos] = static_cast<int32_t>(ordered);
        }
      } else {
        int gpos = atomicAdd(&s->num_spilled_cand[0], 1);
        st_global_s32(buf0 + gpos, i);
      }
      int sub = (ordered >> kFineShifts[0]) & kFineMasks[0];
      atomicAdd(&s->histogram[sub], 1);
    }
  };
  if constexpr (VEC == 4) {
    const int n8 = length >> 3;
    const float4* v4 = reinterpret_cast<const float4*>(score);
    for (int g = tidx; g < n8; g += kBlockThreads) {
      float4 a = ld_float4(reinterpret_cast<const float*>(v4 + 2 * g));
      float4 b = ld_float4(reinterpret_cast<const float*>(v4 + 2 * g + 1));
      int base = g << 3;
      split(a.x, base);
      split(a.y, base + 1);
      split(a.z, base + 2);
      split(a.w, base + 3);
      split(b.x, base + 4);
      split(b.y, base + 5);
      split(b.z, base + 6);
      split(b.w, base + 7);
    }
    for (int i = (n8 << 3) + tidx; i < length; i += kBlockThreads) {
      split(score[i], i);
    }
  } else {
    for (int i = tidx; i < length; i += kBlockThreads) {
      split(score[i], i);
    }
  }
  __syncthreads();
  return remaining;
}

template <int kTopK, bool kSampled, int kBlockThreads = THREADS>
__device__ __noinline__ void refine_rounds(RowSmem<kTopK, kSampled, kBlockThreads>* s,
                                           const float* score, int32_t* buf0, int32_t* buf1,
                                           int topk_remaining, int tidx);
template <int kTopK, int kBlockThreads = THREADS>
__device__ __forceinline__ void refine_rounds_split_keys(RowSmem<kTopK, true, kBlockThreads>* s,
                                                         int32_t* buf0, uint32_t* key0,
                                                         int32_t* buf1, uint32_t* key1,
                                                         int num_input, int topk_remaining,
                                                         int tidx, int first_round = 0) {
  if (first_round == 0) {
    s->num_cand[0] = num_input;
    s->num_cand[1] = 0;
  }
  for (int round = first_round; round < NUM_REFINE_ROUNDS; ++round) {
    const int r = round & 1;
    prefix_sum_find_threshold(s, tidx, topk_remaining, r ^ 1, &s->ties_remaining, topk_remaining);
    __syncthreads();
    const int threshold = s->threshold_bin;
    if (threshold > 0) {
      topk_remaining -= s->histogram[threshold - 1];
    }
    if (tidx == 0) {
      s->ties_remaining = topk_remaining;
    }
    __syncthreads();
    int32_t* cur_ids = r ? buf1 : buf0;
    uint32_t* cur_keys = r ? key1 : key0;
    int32_t* next_ids = r ? buf0 : buf1;
    uint32_t* next_keys = r ? key0 : key1;
    const int cur_num = s->num_cand[r];
    const int cur_shift = kFineShifts[round];
    const uint32_t cur_mask = kFineMasks[round];
    const bool is_last = (round == NUM_REFINE_ROUNDS - 1);
    const int next_shift = is_last ? 0 : kFineShifts[round + 1];
    const uint32_t next_mask = is_last ? 0 : kFineMasks[round + 1];
    if (topk_remaining != 0 && !is_last) {
      for (int i = tidx; i < RADIX + 1; i += kBlockThreads) {
        s->histogram[i] = 0;
      }
      __syncthreads();
    }
    for (int i = tidx; i < cur_num; i += kBlockThreads) {
      const int idx = cur_ids[i];
      const uint32_t bits = cur_keys[i];
      const int bin = (bits >> cur_shift) & cur_mask;
      if (bin < threshold) {
        const int pos = atomicAdd(&s->num_selected, 1);
        s->indices[pos] = idx;
      } else if (bin == threshold) {
        if (is_last || topk_remaining == 0) {
          if (is_last) {
            const int cur = atomicAdd(&s->ties_remaining, -1);
            if (cur > 0) {
              s->indices[kTopK - cur] = idx;
            }
          }
        } else {
          const int pos = atomicAdd(&s->num_cand[r ^ 1], 1);
          next_ids[pos] = idx;
          next_keys[pos] = bits;
          const int sub = (bits >> next_shift) & next_mask;
          atomicAdd(&s->histogram[sub], 1);
        }
      }
    }
    __syncthreads();
    if (topk_remaining == 0 || is_last) {
      break;
    }
  }
}

template <int kTopK, bool kSampled, int kBlockThreads>
__device__ __noinline__ void refine_rounds(RowSmem<kTopK, kSampled, kBlockThreads>* s,
                                           const float* score, int32_t* buf0, int32_t* buf1,
                                           int topk_remaining, int tidx) {
  using Smem = RowSmem<kTopK, kSampled, kBlockThreads>;
  bool run = true;
  for (int round = 0; round < NUM_REFINE_ROUNDS; ++round) {
    if (!run) {
      break;
    }
    int r = round & 1;
    prefix_sum_find_threshold(s, tidx, topk_remaining, r ^ 1, &s->ties_remaining, topk_remaining);
    __syncthreads();
    int num_input = min(s->num_cand[r], Smem::CANDIDATE_CAPACITY);
    int g_num = s->num_spilled_cand[r];
    int threshold = s->threshold_bin;
    if (threshold > 0) {
      topk_remaining -= s->histogram[threshold - 1];
    }
    if (tidx == 0) {
      s->ties_remaining = topk_remaining;
    }
    __syncthreads();
    int cur_shift = kFineShifts[round];
    uint32_t cur_mask = kFineMasks[round];
    bool is_last = (round == NUM_REFINE_ROUNDS - 1);
    int next_shift = is_last ? 0 : kFineShifts[round + 1];
    uint32_t next_mask = is_last ? 0 : kFineMasks[round + 1];
    int32_t* cur_buf = r ? buf1 : buf0;
    int32_t* next_buf = r ? buf0 : buf1;
    if (topk_remaining == 0) {
      for (int i = tidx; i < num_input; i += kBlockThreads) {
        int idx = s->cand_idx[r][i];
        int b = (to_ordered(score[idx]) >> cur_shift) & cur_mask;
        if (b < threshold) {
          int pos = atomicAdd(&s->num_selected, 1);
          s->indices[pos] = idx;
        }
      }
      for (int i = tidx; i < g_num; i += kBlockThreads) {
        int idx = cur_buf[i];
        int b = (to_ordered(score[idx]) >> cur_shift) & cur_mask;
        if (b < threshold) {
          int pos = atomicAdd(&s->num_selected, 1);
          s->indices[pos] = idx;
        }
      }
      __syncthreads();
      run = false;
    } else {
      for (int i = tidx; i < RADIX + 1; i += kBlockThreads) {
        s->histogram[i] = 0;
      }
      __syncthreads();
      auto refine_one = [&](int idx) {
        float raw = score[idx];
        uint32_t bits = to_ordered(raw);
        int b = (bits >> cur_shift) & cur_mask;
        if (b < threshold) {
          int pos = atomicAdd(&s->num_selected, 1);
          s->indices[pos] = idx;
        } else if (b == threshold) {
          if (is_last) {
            int cur = atomicAdd(&s->ties_remaining, -1);
            if (cur > 0) {
              s->indices[kTopK - cur] = idx;
            }
          } else {
            int cur = atomicAdd(&s->num_cand[r ^ 1], 1);
            if (cur < Smem::CANDIDATE_CAPACITY) {
              s->cand_idx[r ^ 1][cur] = idx;
            } else {
              int gpos = atomicAdd(&s->num_spilled_cand[r ^ 1], 1);
              st_global_s32(next_buf + gpos, idx);
            }
            int sub = (bits >> next_shift) & next_mask;
            atomicAdd(&s->histogram[sub], 1);
          }
        }
      };
      for (int i = tidx; i < num_input; i += kBlockThreads) {
        refine_one(s->cand_idx[r][i]);
      }
      for (int i = tidx; i < g_num; i += kBlockThreads) {
        refine_one(cur_buf[i]);
      }
      __syncthreads();
    }
  }
}

template <int kTopK, int kBlockThreads>
__device__ __noinline__ void refine_resident_keys(RowSmem<kTopK, false, kBlockThreads>* s,
                                                  int topk_remaining, int tidx) {
  const int num_input = s->num_cand[0];
  uint32_t active_mask = 0;
  uint32_t active_prefix = 0;
#pragma unroll
  for (int round = 0; round < NUM_REFINE_ROUNDS; ++round) {
    prefix_sum_find_threshold<kTopK, false, -1, kBlockThreads, false>(
        s, tidx, topk_remaining, 0, &s->ties_remaining, topk_remaining);
    const int threshold = s->threshold_bin;
    if (threshold > 0) {
      topk_remaining -= s->histogram[threshold - 1];
    }
    if (tidx == 0) {
      s->ties_remaining = topk_remaining;
    }
    __syncthreads();
    const int cur_shift = kFineShifts[round];
    const uint32_t cur_mask = kFineMasks[round];
    const bool is_last = round == NUM_REFINE_ROUNDS - 1;
    if (topk_remaining != 0 && !is_last) {
      for (int bin = tidx; bin < RADIX + 1; bin += kBlockThreads) {
        s->histogram[bin] = 0;
      }
      __syncthreads();
    }
    for (int pos = tidx; pos < num_input; pos += kBlockThreads) {
      const uint32_t ordered = static_cast<uint32_t>(s->cand_idx[1][pos]);
      const int idx = s->cand_idx[0][pos];
      if ((ordered & active_mask) != active_prefix) {
        continue;
      }
      const int bin = (ordered >> cur_shift) & cur_mask;
      if (bin < threshold) {
        const int output_pos = atomicAdd(&s->num_selected, 1);
        s->indices[output_pos] = idx;
      } else if (bin == threshold && topk_remaining != 0) {
        if (is_last) {
          const int cur = atomicAdd(&s->ties_remaining, -1);
          if (cur > 0) {
            s->indices[kTopK - cur] = idx;
          }
        } else {
          const int next_shift = kFineShifts[round + 1];
          const uint32_t next_mask = kFineMasks[round + 1];
          const int next_bin = (ordered >> next_shift) & next_mask;
          atomicAdd(&s->histogram[next_bin], 1);
        }
      }
    }
    __syncthreads();
    if (topk_remaining == 0 || is_last) {
      break;
    }
    active_mask |= cur_mask << cur_shift;
    active_prefix |= static_cast<uint32_t>(threshold) << cur_shift;
  }
}

template <int kTopK, int kBlockThreads>
__device__ __forceinline__ void refine_row_local_candidates(RowLocalSmem<kTopK, kBlockThreads>* s,
                                                            int topk_remaining, int tidx) {
  const bool valid = tidx < s->num_candidates;
  const RowCandidate candidate = valid ? s->candidates[tidx] : RowCandidate{-1, 0};
  bool active = valid;
#pragma unroll
  for (int round = 0; round < NUM_REFINE_ROUNDS; ++round) {
    find_row_local_threshold(s, tidx, topk_remaining);
    const int threshold = s->threshold_bin;
    topk_remaining -= s->threshold_above;
    if (tidx == 0) {
      s->ties_remaining = topk_remaining;
    }
    const int shift = kFineShifts[round];
    const uint32_t mask = kFineMasks[round];
    const bool is_last = round == NUM_REFINE_ROUNDS - 1;
    if (topk_remaining != 0 && !is_last) {
      for (int bin = tidx; bin < RADIX; bin += kBlockThreads) {
        s->histogram[bin] = 0;
      }
    }
    __syncthreads();
    if (active) {
      const int bin = (candidate.key >> shift) & mask;
      if (bin < threshold) {
        const int pos = atomicAdd(&s->num_selected, 1);
        s->indices[pos] = candidate.index;
        active = false;
      } else if (bin > threshold || topk_remaining == 0) {
        active = false;
      } else if (is_last) {
        const int cur = atomicAdd(&s->ties_remaining, -1);
        if (cur > 0) {
          s->indices[kTopK - cur] = candidate.index;
        }
        active = false;
      } else {
        const int next_shift = kFineShifts[round + 1];
        const uint32_t next_mask = kFineMasks[round + 1];
        const int next_bin = (candidate.key >> next_shift) & next_mask;
        atomicAdd(&s->histogram[next_bin], 1);
      }
    }
    __syncthreads();
    if (topk_remaining == 0 || is_last) {
      break;
    }
  }
}

template <int kTopK, int kBlockThreads>
__device__ __forceinline__ void refine_row_local_rescan(RowLocalSmem<kTopK, kBlockThreads>* s,
                                                        const float* score, int length,
                                                        int threshold_bin, int topk_remaining,
                                                        int tidx) {
  const PackedCoarseBoundary select_boundary =
      pack_coarse_boundary(coarse_boundary(threshold_bin - 1));
  const PackedCoarseBoundary candidate_boundary =
      pack_coarse_boundary(coarse_boundary(threshold_bin));
  uint32_t active_mask = 0;
  uint32_t active_prefix = 0;
  for (int bin = tidx; bin < RADIX; bin += kBlockThreads) {
    s->histogram[bin] = 0;
  }
  if (tidx == 0) {
    s->num_selected = 0;
  }
  __syncthreads();
  for (int i = tidx; i < length; i += kBlockThreads) {
    const float raw = score[i];
    const CoarseClass coarse_class = coarse_classify(raw, select_boundary, candidate_boundary);
    if (coarse_class == CoarseClass::kSelected) {
      const int pos = atomicAdd(&s->num_selected, 1);
      s->indices[pos] = i;
    } else if (coarse_class == CoarseClass::kCandidate) {
      const uint32_t ordered = to_ordered(raw);
      const int bin = (ordered >> kFineShifts[0]) & kFineMasks[0];
      atomicAdd(&s->histogram[bin], 1);
    }
  }
  __syncthreads();
#pragma unroll 1
  for (int round = 0; round < NUM_REFINE_ROUNDS; ++round) {
    find_row_local_threshold(s, tidx, topk_remaining);
    const int threshold = s->threshold_bin;
    topk_remaining -= s->threshold_above;
    if (tidx == 0) {
      s->ties_remaining = topk_remaining;
    }
    const int shift = kFineShifts[round];
    const uint32_t mask = kFineMasks[round];
    const bool is_last = round == NUM_REFINE_ROUNDS - 1;
    if (topk_remaining != 0 && !is_last) {
      for (int bin = tidx; bin < RADIX; bin += kBlockThreads) {
        s->histogram[bin] = 0;
      }
    }
    __syncthreads();
    for (int i = tidx; i < length; i += kBlockThreads) {
      const float raw = score[i];
      if (coarse_classify(raw, select_boundary, candidate_boundary) != CoarseClass::kCandidate) {
        continue;
      }
      const uint32_t ordered = to_ordered(raw);
      if ((ordered & active_mask) != active_prefix) {
        continue;
      }
      const int bin = (ordered >> shift) & mask;
      if (bin < threshold) {
        const int pos = atomicAdd(&s->num_selected, 1);
        s->indices[pos] = i;
      } else if (bin == threshold && topk_remaining != 0) {
        if (is_last) {
          const int cur = atomicAdd(&s->ties_remaining, -1);
          if (cur > 0) {
            s->indices[kTopK - cur] = i;
          }
        } else {
          const int next_shift = kFineShifts[round + 1];
          const uint32_t next_mask = kFineMasks[round + 1];
          const int next_bin = (ordered >> next_shift) & next_mask;
          atomicAdd(&s->histogram[next_bin], 1);
        }
      }
    }
    __syncthreads();
    if (topk_remaining == 0 || is_last) {
      break;
    }
    active_mask |= mask << shift;
    active_prefix |= static_cast<uint32_t>(threshold) << shift;
  }
}

constexpr int SHORT_THREADS = 1024;
constexpr int ROW_LOCAL_SMALL_RANK_CAPACITY = 64;
constexpr int ROW_LOCAL_MEDIUM_RANK_CAPACITY = 128;
__device__ __forceinline__ bool row_candidate_precedes(const RowCandidate& lhs,
                                                       const RowCandidate& rhs) {
  return lhs.key < rhs.key || (lhs.key == rhs.key && lhs.index < rhs.index);
}

template <int kItems, int kTopK, int kBlockThreads>
__device__ __forceinline__ void select_row_local_warp_rank(RowLocalSmem<kTopK, kBlockThreads>* s,
                                                           int topk_remaining, int tidx) {
  static_assert(kItems == 2 || kItems == 4);
  const int count = s->num_candidates;
  const int lane = tidx & 31;
  const int warp = tidx >> 5;
  const RowCandidate invalid{-1, ~0u};
  RowCandidate owned[kItems];
#pragma unroll
  for (int item = 0; item < kItems; ++item) {
    const int pos = lane + item * 32;
    owned[item] = pos < count ? s->candidates[pos] : invalid;
  }
#pragma unroll
  for (int item = 0; item < kItems; ++item) {
    const int target_pos = warp + item * 32;
    if (target_pos >= count) {
      continue;
    }
    const RowCandidate target = s->candidates[target_pos];
    int rank = 0;
#pragma unroll
    for (int other = 0; other < kItems; ++other) {
      rank += __popc(__ballot_sync(0xFFFFFFFFu, row_candidate_precedes(owned[other], target)));
    }
    if (lane == 0 && rank < topk_remaining) {
      s->indices[s->num_selected + rank] = target.index;
    }
  }
  __syncthreads();
}

__device__ __forceinline__ bool claim_streaming_row(int tidx, bool first_wave, int row_capacity,
                                                    int num_valid, int32_t* work_counter,
                                                    int32_t* exit_counter, int* shared_row) {
  if (first_wave) {
    if (static_cast<int>(blockIdx.x) < num_valid) {
      return true;
    }
    if (row_capacity > static_cast<int>(gridDim.x) && tidx == 0 &&
        atomicAdd(exit_counter, 1) == gridDim.x - 1) {
      *work_counter = 0;
      *exit_counter = 0;
    }
    return false;
  }
  if (tidx == 0) {
    if (row_capacity <= static_cast<int>(gridDim.x)) {
      *shared_row = num_valid;
    } else {
      *shared_row = static_cast<int>(gridDim.x) + atomicAdd(work_counter, 1);
    }
  }
  __syncthreads();
  if (*shared_row < num_valid) {
    return true;
  }
  if (row_capacity > static_cast<int>(gridDim.x) && tidx == 0 &&
      atomicAdd(exit_counter, 1) == gridDim.x - 1) {
    *work_counter = 0;
    *exit_counter = 0;
  }
  return false;
}

// Row-local register kernel: one CTA per row keeps the candidate
// working set in registers, sized for the short-row bands.
template <int kVecs, int kTopK, int kMinBlocks>
__global__ void __launch_bounds__(SHORT_THREADS, kMinBlocks)
    topk_filtered_row_local_register_kernel(const float* __restrict__ input,
                                            int32_t* __restrict__ output,
                                            const int32_t* __restrict__ ke,
                                            const int32_t* __restrict__ num_valid_rows,
                                            uint32_t row_stride, uint32_t out_stride) {
  using Smem = RowLocalSmem<kTopK, SHORT_THREADS>;
  const int tidx = threadIdx.x;
  const int num_valid = *num_valid_rows;
  __shared__ Smem smem;
  Smem* s = &smem;
  const int row_i = static_cast<int>(blockIdx.x);
  if (row_i >= num_valid) {
    return;
  }
  const int length = ke[row_i];
  if (length > kVecs * SHORT_THREADS * 4) {
    return;
  }
  if (length <= kTopK) {
    int32_t* dst = output + static_cast<size_t>(row_i) * out_stride;
    for (int i = tidx; i < kTopK; i += SHORT_THREADS) {
      dst[i] = (i < length) ? i : -1;
    }
    return;
  }
  for (int i = tidx; i < RADIX + 1; i += SHORT_THREADS) {
    s->histogram[i] = 0;
  }
  if (tidx == 0) {
    s->num_selected = 0;
  }
  __syncthreads();
  const int n4 = length >> 2;
  float4 local[kVecs];
#pragma unroll
  for (int item = 0; item < kVecs; ++item) {
    const int vi = tidx + item * SHORT_THREADS;
    if (vi < n4) {
      local[item] = ld_float4(input + static_cast<size_t>(row_i) * row_stride + (vi << 2));
    }
  }
  const int tail_start = n4 << 2;
  auto add_coarse = [&](float raw) { atomicAdd(&s->histogram[to_coarse_key(raw)], 1); };
#pragma unroll
  for (int item = 0; item < kVecs; ++item) {
    const int vi = tidx + item * SHORT_THREADS;
    if (vi < n4) {
      add_coarse(local[item].x);
      add_coarse(local[item].y);
      add_coarse(local[item].z);
      add_coarse(local[item].w);
    }
  }
  if (tidx < length - tail_start) {
    add_coarse(input[static_cast<size_t>(row_i) * row_stride + tail_start + tidx]);
  }
  __syncthreads();
  find_row_local_threshold(s, tidx, kTopK);
  const int threshold_bin = s->threshold_bin;
  const int remaining = kTopK - s->threshold_above;
  if (remaining == 0) {
    if (tidx == 0) {
      s->num_selected = 0;
    }
    __syncthreads();
    const PackedCoarseBoundary select_boundary =
        pack_coarse_boundary(coarse_boundary(threshold_bin - 1));
    auto collect = [&](float raw, int idx) {
      if (coarse_accepts(raw, select_boundary)) {
        const int pos = atomicAdd(&s->num_selected, 1);
        s->indices[pos] = idx;
      }
    };
#pragma unroll
    for (int item = 0; item < kVecs; ++item) {
      const int vi = tidx + item * SHORT_THREADS;
      if (vi < n4) {
        const int base = vi << 2;
        collect(local[item].x, base);
        collect(local[item].y, base + 1);
        collect(local[item].z, base + 2);
        collect(local[item].w, base + 3);
      }
    }
    if (tidx < length - tail_start) {
      collect(input[static_cast<size_t>(row_i) * row_stride + tail_start + tidx],
              tail_start + tidx);
    }
    __syncthreads();
  } else if (s->threshold_count > Smem::CANDIDATE_CAPACITY) {
    __shared__ uint32_t tie_lo;
    __shared__ uint32_t tie_hi;
    const PackedCoarseBoundary select_boundary =
        pack_coarse_boundary(coarse_boundary(threshold_bin - 1));
    const PackedCoarseBoundary candidate_boundary =
        pack_coarse_boundary(coarse_boundary(threshold_bin));
    uint32_t lo = 0xFFFFFFFFu;
    uint32_t hi = 0u;
    auto range = [&](float raw) {
      if (coarse_classify(raw, select_boundary, candidate_boundary) == CoarseClass::kCandidate) {
        const uint32_t k = to_ordered(raw);
        lo = min(lo, k);
        hi = max(hi, k);
      }
    };
    if (tidx < n4) range(local[0].x);
    bool tied = block_key_range_single<SHORT_THREADS>(lo, hi, &tie_lo, &tie_hi, tidx);
    if (tied) {
      if (tidx == 0) {
        s->num_selected = 0;
        s->num_candidates = 0;
      }
      __syncthreads();
      const int lane = tidx & 31;
      lo = 0xFFFFFFFFu;
      hi = 0u;
      auto take = [&](float raw, int idx) {
        const CoarseClass c = coarse_classify(raw, select_boundary, candidate_boundary);
        if (c == CoarseClass::kSelected) {
          s->indices[atomicAdd(&s->num_selected, 1)] = idx;
        } else if (c == CoarseClass::kCandidate) {
          const uint32_t k = to_ordered(raw);
          lo = min(lo, k);
          hi = max(hi, k);
          const unsigned act = __activemask();
          const int leader = __ffs(act) - 1;
          int base = 0;
          if (lane == leader) base = atomicAdd(&s->num_candidates, __popc(act));
          base = __shfl_sync(act, base, leader);
          const int pos = base + __popc(act & ((1u << lane) - 1u));
          if (pos < remaining) s->indices[kTopK - remaining + pos] = idx;
        }
      };
#pragma unroll
      for (int item = 0; item < kVecs; ++item) {
        const int vi = tidx + item * SHORT_THREADS;
        if (vi < n4) {
          const int base = vi << 2;
          take(local[item].x, base);
          take(local[item].y, base + 1);
          take(local[item].z, base + 2);
          take(local[item].w, base + 3);
        }
      }
      if (tidx < length - tail_start) {
        take(input[static_cast<size_t>(row_i) * row_stride + tail_start + tidx], tail_start + tidx);
      }
      tied = block_key_range_single<SHORT_THREADS>(lo, hi, &tie_lo, &tie_hi, tidx);
    }
    if (!tied)
      refine_row_local_rescan(s, input + static_cast<size_t>(row_i) * row_stride, length,
                              threshold_bin, remaining, tidx);
  } else {
    if (tidx == 0) {
      s->num_selected = 0;
      s->num_candidates = 0;
    }
    __syncthreads();
    const PackedCoarseBoundary select_boundary =
        pack_coarse_boundary(coarse_boundary(threshold_bin - 1));
    const PackedCoarseBoundary candidate_boundary =
        pack_coarse_boundary(coarse_boundary(threshold_bin));
    auto split = [&](float raw, int idx) {
      const CoarseClass coarse_class = coarse_classify(raw, select_boundary, candidate_boundary);
      if (coarse_class == CoarseClass::kSelected) {
        const int pos = atomicAdd(&s->num_selected, 1);
        s->indices[pos] = idx;
      } else if (coarse_class == CoarseClass::kCandidate) {
        const int pos = atomicAdd(&s->num_candidates, 1);
        const uint32_t ordered = to_ordered(raw);
        s->candidates[pos] = RowCandidate{idx, ordered};
      }
    };
#pragma unroll
    for (int item = 0; item < kVecs; ++item) {
      const int vi = tidx + item * SHORT_THREADS;
      if (vi < n4) {
        const int base = vi << 2;
        split(local[item].x, base);
        split(local[item].y, base + 1);
        split(local[item].z, base + 2);
        split(local[item].w, base + 3);
      }
    }
    if (tidx < length - tail_start) {
      split(input[static_cast<size_t>(row_i) * row_stride + tail_start + tidx], tail_start + tidx);
    }
    __syncthreads();
    if (s->num_candidates <= ROW_LOCAL_SMALL_RANK_CAPACITY) {
      select_row_local_warp_rank<2>(s, remaining, tidx);
    } else if (s->num_candidates <= ROW_LOCAL_MEDIUM_RANK_CAPACITY) {
      select_row_local_warp_rank<4>(s, remaining, tidx);
    } else {
      for (int bin = tidx; bin < RADIX; bin += SHORT_THREADS) {
        s->histogram[bin] = 0;
      }
      __syncthreads();
      if (tidx < s->num_candidates) {
        const uint32_t ordered = s->candidates[tidx].key;
        const int fine = (ordered >> kFineShifts[0]) & kFineMasks[0];
        atomicAdd(&s->histogram[fine], 1);
      }
      __syncthreads();
      refine_row_local_candidates(s, remaining, tidx);
    }
  }
  for (int i = tidx; i < kTopK; i += SHORT_THREADS) {
    output[static_cast<size_t>(row_i) * out_stride + i] = s->indices[i];
  }
}

// Row-local streaming kernel: one CTA per row streams the row through
// the coarse stage once and refines from shared-memory candidates.
template <uint32_t VEC, int kTopK, bool kBounded, int kMinBlocks, int kMinLength = 0>
__global__ void __launch_bounds__(SHORT_THREADS, kMinBlocks)
    topk_filtered_row_local_streaming_kernel(
        const float* __restrict__ input, int32_t* __restrict__ output,
        const int32_t* __restrict__ ke, int32_t* __restrict__ spill_buffer,
        int32_t* __restrict__ work_counter, int32_t* __restrict__ exit_counter,
        const int32_t* __restrict__ num_valid_rows, uint32_t row_stride, uint32_t out_stride,
        uint32_t n_cols, int32_t row_capacity) {
  using Smem = RowSmem<kTopK, false, SHORT_THREADS>;
  const int tidx = threadIdx.x;
  const int num_valid = *num_valid_rows;
  if (num_valid <= 0) {
    return;
  }
  __shared__ Smem smem;
  Smem* s = &smem;
  bool first_wave = true;
  bool have_work;
  if constexpr (kBounded) {
    have_work = claim_streaming_row(tidx, first_wave, row_capacity, num_valid, work_counter,
                                    exit_counter, &s->threshold_bin);
  } else {
    have_work = static_cast<int>(blockIdx.x) < num_valid;
  }
  while (have_work) {
    int row_i;
    if constexpr (kBounded) {
      row_i = first_wave ? static_cast<int>(blockIdx.x) : s->threshold_bin;
      first_wave = false;
    } else {
      row_i = static_cast<int>(blockIdx.x);
    }
    const float* score = input + static_cast<size_t>(row_i) * row_stride;
    const int length = ke[row_i];
    if constexpr (kMinLength > 0) {
      if (length <= kMinLength) {
        if constexpr (!kBounded) {
          break;
        }
        have_work = claim_streaming_row(tidx, false, row_capacity, num_valid, work_counter,
                                        exit_counter, &s->threshold_bin);
        continue;
      }
    }
    if (length <= kTopK) {
      for (int i = tidx; i < kTopK; i += SHORT_THREADS) {
        output[static_cast<size_t>(row_i) * out_stride + i] = (i < length) ? i : -1;
      }
      if constexpr (!kBounded) {
        break;
      }
      have_work = claim_streaming_row(tidx, false, row_capacity, num_valid, work_counter,
                                      exit_counter, &s->threshold_bin);
      continue;
    }
    int32_t* spill0 = spill_buffer + static_cast<size_t>(blockIdx.x) * 2 * n_cols;
    int32_t* spill1 = spill0 + n_cols;
    const int topk_remaining =
        run_exact_coarse_stage<VEC, kTopK, false, SHORT_THREADS>(s, score, spill0, length, tidx);
    if (topk_remaining != 0) {
      if (s->num_spilled_cand[0] == 0) {
        refine_resident_keys<kTopK, SHORT_THREADS>(s, topk_remaining, tidx);
      } else {
        refine_rounds<kTopK, false, SHORT_THREADS>(s, score, spill0, spill1, topk_remaining, tidx);
      }
    }
    for (int i = tidx; i < kTopK; i += SHORT_THREADS) {
      output[static_cast<size_t>(row_i) * out_stride + i] = s->indices[i];
    }
    if constexpr (!kBounded) {
      break;
    }
    have_work = claim_streaming_row(tidx, false, row_capacity, num_valid, work_counter,
                                    exit_counter, &s->threshold_bin);
  }
}

constexpr size_t HEAD_COUNTER_INTS = 2;  // work_counter, exit_counter
inline size_t counters_bytes_for(int m) {
  const size_t rows = static_cast<size_t>(m > 0 ? m : 0);
  return (HEAD_COUNTER_INTS + 6 * rows) * sizeof(int32_t);
}

inline size_t spill_ints(int n) { return static_cast<size_t>(grid_num_blocks()) * 2 * n; }
inline size_t one_cta_workspace_bytes(int n) { return spill_ints(n) * sizeof(int32_t); }
inline size_t split_spill_ints(int m, int n) {
  return static_cast<size_t>(std::max(grid_num_blocks(), m)) * 4 * n;
}

inline size_t split_workspace_bytes(int m, int splits, int n) {
  const size_t split_ints =
      static_cast<size_t>(m) * splits + static_cast<size_t>(m) * splits * RADIX +
      static_cast<size_t>(m) * splits * RADIX + static_cast<size_t>(m) * n * 2;
  return (split_spill_ints(m, n) + split_ints) * sizeof(int32_t);
}

// KV-split kernel: every CTA samples its own segment, exchanges the
// sample histograms over a per-row arrive counter, scans its segment
// under the shared coarse boundary, and the last CTA to arrive
// finishes the row. The exchange spins, so all kSplits CTAs of a row
// must be co-resident; the launcher enforces it.
template <uint32_t VEC, int kTopK, int kSplits, int kSampleStride, int kSampleRank>
__global__ void __launch_bounds__(THREADS, MIN_BLOCKS)
    topk_filtered_split_kernel(const float* __restrict__ input, int32_t* __restrict__ output,
                               const int32_t* __restrict__ ke, int32_t* __restrict__ split_buffer,
                               int32_t* __restrict__ spill_buffer,
                               int32_t* __restrict__ arrive_counters,
                               const int32_t* __restrict__ num_valid_rows, uint32_t row_stride,
                               uint32_t out_stride, uint32_t n_cols, int32_t m_rows) {
  using Smem = RowSmem<kTopK, true>;
  const int tidx = threadIdx.x;
  const int num_valid = *num_valid_rows;
  if (num_valid <= 0) {
    return;
  }
  const int64_t num_tasks = static_cast<int64_t>(num_valid) * kSplits;
  int32_t* const ws_counts = split_buffer;
  int32_t* const ws_hists = ws_counts + static_cast<size_t>(m_rows) * kSplits;
  int32_t* const ws_shists = ws_hists + static_cast<size_t>(m_rows) * kSplits * RADIX;
  int32_t* const ws_cand_ids = ws_shists + static_cast<size_t>(m_rows) * kSplits * RADIX;
  int32_t* const ws_cand_keys = ws_cand_ids + static_cast<size_t>(m_rows) * n_cols;
  extern __shared__ uint8_t smem_raw[];
  Smem* s = reinterpret_cast<Smem*>(smem_raw);
  __shared__ int shared_arrived;
  __shared__ int shared_total;
  __shared__ int shared_seg_offsets[kSplits + 1];
  __shared__ float shared_coarse_boundary;
  {
    const int64_t task = static_cast<int64_t>(blockIdx.x);
    if (task >= num_tasks) {
      return;
    }
    const int row_i = static_cast<int>(task / kSplits);
    const int seg = static_cast<int>(task % kSplits);
    const uint32_t row = static_cast<uint32_t>(row_i);
    const float* score = input + static_cast<size_t>(row) * row_stride;
    int32_t* dst = output + static_cast<size_t>(row) * out_stride;
    const int length = ke[row];
    int32_t* arrive = arrive_counters + row_i;
    int32_t* sample_arrive = arrive_counters + m_rows + row_i;
    int32_t* seg_counts = ws_counts + static_cast<size_t>(row_i) * kSplits;
    int32_t* seg_hists = ws_hists + static_cast<size_t>(row_i) * kSplits * RADIX;
    int32_t* cand_ids = ws_cand_ids + static_cast<size_t>(row_i) * n_cols;
    uint32_t* cand_keys =
        reinterpret_cast<uint32_t*>(ws_cand_keys + static_cast<size_t>(row_i) * n_cols);
    if (length <= kTopK) {
      if (seg == 0) {
        for (int i = tidx; i < kTopK; i += THREADS) {
          dst[i] = (i < length) ? i : -1;
        }
      }
      return;
    }
    const int seg_len = (length + kSplits - 1) / kSplits;
    const int seg_lo = min(seg * seg_len, length);
    const int seg_hi = min(seg_lo + seg_len, length);
    constexpr int kSampleTile = SAMPLE_TILE_ELEMS * kSampleStride;
    const bool sample_usable =
        seg_len >= (kSampleRank + 1) * kSampleStride / kSplits + 2 * kSampleTile;
    if (sample_usable) {
      for (int i = tidx; i < HIST_SWEEP; i += THREADS) {
        s->histogram[i] = 0;
      }
      __syncthreads();
      const int warp_id = tidx >> 5;
      const int lane_id = tidx & 31;
      constexpr int kSampleWarps = THREADS >> 5;
      const int sample_offset =
          static_cast<int>((row * 17u + 13u + static_cast<uint32_t>(seg) * 7u) &
                           (kSampleStride - 1)) *
          SAMPLE_TILE_ELEMS;
      const int sample_step = kSampleWarps * kSampleTile;
      int tile = seg_lo + sample_offset + warp_id * kSampleTile;
      constexpr int kBatchRegs = SAMPLE_TILE_BATCH * SAMPLE_TILE_LOADS;
      for (; tile + (SAMPLE_TILE_BATCH - 1) * sample_step + SAMPLE_TILE_ELEMS <= seg_hi;
           tile += SAMPLE_TILE_BATCH * sample_step) {
        float v[kBatchRegs];
#pragma unroll
        for (int b = 0; b < SAMPLE_TILE_BATCH; ++b) {
#pragma unroll
          for (int t = 0; t < SAMPLE_TILE_LOADS; ++t) {
            v[b * SAMPLE_TILE_LOADS + t] = score[tile + b * sample_step + (t << 5) + lane_id];
          }
        }
#pragma unroll
        for (int r = 0; r < kBatchRegs; ++r) {
          atomicAdd(&s->histogram[to_coarse_key(v[r])], 1);
        }
      }
      for (; tile < seg_hi; tile += sample_step) {
#pragma unroll
        for (int t = 0; t < SAMPLE_TILE_LOADS; ++t) {
          int i = tile + (t << 5) + lane_id;
          if (i < seg_hi) {
            atomicAdd(&s->histogram[to_coarse_key(score[i])], 1);
          }
        }
      }
      __syncthreads();
      {
        int32_t* seg_shists = ws_shists + static_cast<size_t>(row_i) * kSplits * RADIX;
        for (int i = tidx; i < RADIX; i += THREADS) {
          seg_shists[static_cast<size_t>(seg) * RADIX + i] = s->histogram[i];
        }
        __threadfence();
        __syncthreads();
        if (tidx == 0) {
          atomicAdd(sample_arrive, 1);
          while (atomicAdd(sample_arrive, 0) < kSplits) {
          }
        }
        __syncthreads();
        __threadfence();
        for (int i = tidx; i < HIST_SWEEP; i += THREADS) {
          int sum = 0;
          if (i < RADIX) {
            for (int sp = 0; sp < kSplits; ++sp) {
              sum += seg_shists[static_cast<size_t>(sp) * RADIX + i];
            }
          }
          s->histogram[i] = sum;
        }
      }
      __syncthreads();
      prefix_sum_find_threshold(s, tidx, kSampleRank, 0, &s->num_selected, 0);
      if (tidx == 0) {
        shared_coarse_boundary = coarse_boundary(s->threshold_bin).value;
      }
      __syncthreads();
    }
    if (sample_usable) {
      const float sampled_boundary = shared_coarse_boundary;
      const uint32_t sampled_boundary_bits = __float_as_uint(sampled_boundary);
      for (int i = tidx; i < HIST_SWEEP; i += THREADS) {
        s->histogram[i] = 0;
      }
      if (tidx == 0) {
        s->num_cand[0] = 0;
      }
      __syncthreads();
      auto seg_split = [&](float raw, int i) {
        if (raw >= sampled_boundary) {
          const uint32_t bits = to_ordered(raw);
          int pos = seg_lo + atomicAdd(&s->num_cand[0], 1);
          cand_ids[pos] = i;
          cand_keys[pos] = bits;
          int sub = (bits >> kFineShifts[0]) & kFineMasks[0];
          atomicAdd(&s->histogram[sub], 1);
        }
      };
      const bool boundary_is_zero = (sampled_boundary_bits & 0x7FFFFFFFu) == 0u;
      if (boundary_is_zero) {
        for (int i = seg_lo + tidx; i < seg_hi; i += THREADS) {
          const float raw = score[i];
          if (fp32_boundary_accepts(raw, sampled_boundary, sampled_boundary_bits)) {
            const uint32_t bits = to_ordered(raw);
            int pos = seg_lo + atomicAdd(&s->num_cand[0], 1);
            cand_ids[pos] = i;
            cand_keys[pos] = bits;
            atomicAdd(&s->histogram[(bits >> kFineShifts[0]) & kFineMasks[0]], 1);
          }
        }
      } else if constexpr (VEC == 4) {
        const int v_lo = min((seg_lo + 7) & ~7, seg_hi);
        const int v_hi = max(seg_hi & ~7, v_lo);
        for (int i = seg_lo + tidx; i < v_lo; i += THREADS) {
          float raw = score[i];
          seg_split(raw, i);
        }
        const int n4 = (v_hi - v_lo) >> 2;
        int g = tidx;
        for (; g + THREADS < n4; g += 2 * THREADS) {
          const int base_a = v_lo + (g << 2);
          const int base_b = v_lo + ((g + THREADS) << 2);
          const float4 a = ld_float4(score + base_a);
          const float4 b = ld_float4(score + base_b);
          seg_split(a.x, base_a);
          seg_split(a.y, base_a + 1);
          seg_split(a.z, base_a + 2);
          seg_split(a.w, base_a + 3);
          seg_split(b.x, base_b);
          seg_split(b.y, base_b + 1);
          seg_split(b.z, base_b + 2);
          seg_split(b.w, base_b + 3);
        }
        if (g < n4) {
          const int base_a = v_lo + (g << 2);
          const float4 a = ld_float4(score + base_a);
          seg_split(a.x, base_a);
          seg_split(a.y, base_a + 1);
          seg_split(a.z, base_a + 2);
          seg_split(a.w, base_a + 3);
        }
        for (int i = v_hi + tidx; i < seg_hi; i += THREADS) {
          float raw = score[i];
          seg_split(raw, i);
        }
      } else {
        for (int i = seg_lo + tidx; i < seg_hi; i += THREADS) {
          float raw = score[i];
          seg_split(raw, i);
        }
      }
      __syncthreads();
      for (int i = tidx; i < RADIX; i += THREADS) {
        seg_hists[static_cast<size_t>(seg) * RADIX + i] = s->histogram[i];
      }
      if (tidx == 0) {
        seg_counts[seg] = s->num_cand[0];
      }
      __threadfence();
      __syncthreads();
      if (tidx == 0) {
        shared_arrived = atomicAdd(arrive, 1);
      }
      __syncthreads();
      if (shared_arrived != kSplits - 1) {
        return;
      }
      __threadfence();
      if (tidx == 0) {
        *arrive = 0;
        *sample_arrive = 0;
      }
    } else if (seg != 0) {
      return;
    }
    int topk_remaining = kTopK;
    bool need_refine = false;
    bool ready = false;
    int32_t* spill0 = spill_buffer + static_cast<size_t>(row_i) * 4 * n_cols;
    uint32_t* spill_key0 = reinterpret_cast<uint32_t*>(spill0 + n_cols);
    int32_t* spill1 = spill0 + 2 * n_cols;
    uint32_t* spill_key1 = reinterpret_cast<uint32_t*>(spill1 + n_cols);
    if (sample_usable) {
      int local_total = tidx < kSplits ? seg_counts[tidx] : 0;
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        local_total += __shfl_down_sync(0xFFFFFFFFu, local_total, offset);
      }
      if (tidx == 0) {
        shared_total = local_total;
      }
      __syncthreads();
    }
    if (sample_usable && shared_total > kTopK) {
      for (int i = tidx; i < HIST_SWEEP; i += THREADS) {
        int sum = 0;
        if (i < RADIX) {
          for (int sp = 0; sp < kSplits; ++sp) {
            sum += seg_hists[static_cast<size_t>(sp) * RADIX + i];
          }
        }
        s->histogram[i] = sum;
      }
      if (tidx == 0) {
        s->num_selected = 0;
        shared_seg_offsets[0] = 0;
        for (int sp = 0; sp < kSplits; ++sp) {
          shared_seg_offsets[sp + 1] = shared_seg_offsets[sp] + seg_counts[sp];
        }
      }
      __syncthreads();
      prefix_sum_find_threshold(s, tidx, kTopK, 1, &s->ties_remaining, kTopK);
      __syncthreads();
      const int thr = s->threshold_bin;
      int rem = kTopK;
      if (thr > 0) {
        rem -= s->histogram[thr - 1];
      }
      __syncthreads();
      if (rem != 0) {
        for (int i = tidx; i < HIST_SWEEP; i += THREADS) {
          s->histogram[i] = 0;
        }
        __syncthreads();
      }
      for (int flat = tidx; flat < shared_total; flat += THREADS) {
        int sp = 0;
        while (flat >= shared_seg_offsets[sp + 1]) {
          ++sp;
        }
        const int local = flat - shared_seg_offsets[sp];
        const int src = min(sp * seg_len, length) + local;
        const uint32_t key = cand_keys[src];
        const int bin = static_cast<int>((key >> kFineShifts[0]) & kFineMasks[0]);
        if (bin < thr) {
          s->indices[atomicAdd(&s->num_selected, 1)] = cand_ids[src];
        } else if (bin == thr && rem != 0) {
          const int pos = atomicAdd(&s->num_cand[1], 1);
          st_global_s32(spill1 + pos, cand_ids[src]);
          st_global_s32(reinterpret_cast<int32_t*>(spill_key1 + pos), static_cast<int32_t>(key));
          atomicAdd(&s->histogram[(key >> kFineShifts[1]) & kFineMasks[1]], 1);
        }
      }
      __syncthreads();
      ready = true;
      topk_remaining = rem;
      need_refine = rem != 0;
    }
    if (!ready) {
      topk_remaining = run_exact_coarse_stage<VEC, kTopK, true>(s, score, spill0, length, tidx);
      need_refine = topk_remaining != 0;
    }
    if (need_refine) {
      if (ready) {
        refine_rounds_split_keys<kTopK>(s, spill0, spill_key0, spill1, spill_key1, shared_total,
                                        topk_remaining, tidx, 1);
      } else {
        refine_rounds<kTopK, true>(s, score, spill0, spill1, topk_remaining, tidx);
      }
    }
    for (int i = tidx; i < kTopK; i += THREADS) {
      dst[i] = s->indices[i];
    }
  }
}

// Cluster8 finisher path for rows short enough that every CTA sampled
// the whole row redundantly.
template <uint32_t VEC, int kTopK, int kBlockThreads, int kMinBlocks>
__device__ __noinline__ void run_cluster_sampled_short_row(RowSmem<kTopK, true, kBlockThreads>* s,
                                                           const float* __restrict__ score,
                                                           int32_t* __restrict__ dst,
                                                           int32_t* __restrict__ spill0,
                                                           uint32_t n_cols, int length, int tidx) {
  int32_t* spill1 = spill0 + 2 * n_cols;
  const int topk_remaining =
      run_exact_coarse_stage<VEC, kTopK, true, kBlockThreads>(s, score, spill0, length, tidx);
  if (topk_remaining != 0) {
    refine_rounds<kTopK, true, kBlockThreads>(s, score, spill0, spill1, topk_remaining, tidx);
  }
  for (int i = tidx; i < kTopK; i += kBlockThreads) {
    dst[i] = s->indices[i];
  }
}

// mc kernel constants. TARGET_SAMPLES is the sampled elements per row
// (the stride is the power of two closest from above);
// SEGMENT/STAGE/SURVIVOR_CAPACITY size the per-CTA segment list, the
// row's gmem staging list and the smem survivor buffer; ROBUST_STORE_PCT
// switches a CTA to the robust store once its predicted admitted count
// passes this share of the stage.
constexpr int MC_TARGET_SAMPLES = 8192;
constexpr int MC_SEGMENT_CAPACITY = 2048;
constexpr int MC_STAGE_CAPACITY = 4096;
constexpr int MC_ROBUST_STORE_PCT = 50;
constexpr int MC_SURVIVOR_CAPACITY = 512;
// Cluster shape (8 CTAs) and the per-row segment cap.
constexpr int kMcClusterSize = 8;
constexpr int kMcMaxSegs = 512;
// The segment-sample rank bound for a runtime-selected stride.
__device__ __forceinline__ int mc_sample_rank(int top_k, int stride) {
  const float y = static_cast<float>(SEG_SAMPLE_MARGIN) * static_cast<float>(top_k) / stride;
  const float z = static_cast<float>(SEG_SAMPLE_ZSCORE);
  auto ok = [&](int r) {
    const float d = static_cast<float>(r) - y;
    return d >= 0.0f && d * d >= z * z * static_cast<float>(r);
  };
  const float q = 0.5f * (z + sqrtf(z * z + 4.0f * y));
  int r = min(max(static_cast<int>(ceilf(q * q)), 1), RADIX - 1);
  while (r < RADIX - 1 && !ok(r)) ++r;
  while (r > 1 && ok(r - 1)) --r;
  return r;
}

template <int kBlockThreads>
__device__ __forceinline__ void mc_sample_plain(int* hist, const float* __restrict__ score,
                                                int length, int offset, int period, int t0,
                                                int tstep, int tidx) {
  static_assert(kBlockThreads == 1024);
  const int warp = tidx >> 5;
  const int lane = tidx & 31;
  const int step = tstep * 32 * period;
  int pos = offset + (t0 + tstep * warp) * period;
  for (; pos + (SAMPLE_TILE_BATCH - 1) * step + 32 <= length; pos += SAMPLE_TILE_BATCH * step) {
    float v[SAMPLE_TILE_BATCH];
#pragma unroll
    for (int b = 0; b < SAMPLE_TILE_BATCH; ++b) {
      v[b] = score[pos + b * step + lane];
    }
#pragma unroll
    for (int b = 0; b < SAMPLE_TILE_BATCH; ++b) {
      atomicAdd(&hist[to_coarse_key(v[b])], 1);
    }
  }
  for (; pos < length; pos += step) {
    if (pos + lane < length) {
      atomicAdd(&hist[to_coarse_key(score[pos + lane])], 1);
    }
  }
}

template <int kBlockThreads>
__device__ __forceinline__ void mc_sample(int* hist, const float* __restrict__ score, int length,
                                          int offset, int period, int t0, int tstep, int tidx) {
  mc_sample_plain<kBlockThreads>(hist, score, length, offset, period, t0, tstep, tidx);
}

// Tie mode for the mc kernel: when P0's boundary bin holds one exact
// value, P1 admits strictly-greater elements plus a capped number of
// ties per CTA, and the finisher takes the ties from the back.
// TIE_PROBE_ELEMS is the front-of-row probe that decides whether the
// bin is single-valued (taken once the bin holds TIE_MIN_SAMPLES
// samples); TIE_GAIN gates the mode on the predicted win; TIE_CAP_*
// size the per-CTA tie cap.
struct McTieState {
  int probe;
  int tie;
  int cap;
  int n;
  uint32_t kmin;
  uint32_t kmax;
  float in_lo;
  float in_hi;
};

constexpr int MC_TIE_PROBE_ELEMS = 2048;
static_assert(MC_TIE_PROBE_ELEMS % 1024 == 0);
constexpr int MC_TIE_MIN_SAMPLES = 32;
constexpr int MC_TIE_GAIN = 4;
constexpr int MC_TIE_CAP_MULT = 4;
constexpr int MC_TIE_CAP_MIN = 64;
// Exact ordered-key range of the probe samples that fall in the
// boundary bin; warp 0 folds the per-warp ranges into *bin_kmin /
// *bin_kmax.
template <int kProbe, int kBlockThreads>
__device__ __forceinline__ void mc_sample_bin_range(const float* __restrict__ score, int length,
                                                    int offset, int period,
                                                    PackedCoarseBoundary in_lo,
                                                    PackedCoarseBoundary in_hi, uint32_t* scratch,
                                                    uint32_t* bin_kmin, uint32_t* bin_kmax,
                                                    int tidx) {
  static_assert(kBlockThreads == 1024);
  constexpr int kPer = kProbe / kBlockThreads;
  const int lane = tidx & 31;
  const int warp = tidx >> 5;
  float v[kPer];
#pragma unroll
  for (int j = 0; j < kPer; ++j) {
    const int p = offset + (warp + 32 * j) * period + lane;
    v[j] = p < length ? __ldg(score + p) : __int_as_float(0x7FC00000);
  }
  uint32_t lo = 0xFFFFFFFFu;
  uint32_t hi = 0u;
#pragma unroll
  for (int j = 0; j < kPer; ++j) {
    if (coarse_accepts(v[j], in_lo) && !coarse_accepts(v[j], in_hi)) {
      const uint32_t k = to_ordered(v[j]);
      lo = min(lo, k);
      hi = max(hi, k);
    }
  }
  lo = __reduce_min_sync(0xFFFFFFFFu, lo);
  hi = __reduce_max_sync(0xFFFFFFFFu, hi);
  if (lane == 0) {
    scratch[warp] = lo;
    scratch[32 + warp] = hi;
  }
  __syncthreads();
  if (warp == 0) {
    lo = __reduce_min_sync(0xFFFFFFFFu, scratch[lane]);
    hi = __reduce_max_sync(0xFFFFFFFFu, scratch[32 + lane]);
    if (lane == 0) {
      *bin_kmin = lo;
      *bin_kmax = hi;
    }
  }
}

// Runs the front-of-row probe and publishes the boundary bin's key
// range.
template <int kBlockThreads>
__device__ __noinline__ void mc_tie_probe(const float* __restrict__ score, int length, int offset,
                                          int period, float in_lo, float in_hi, uint32_t* scratch,
                                          uint32_t* bin_kmin, uint32_t* bin_kmax, int tidx) {
  mc_sample_bin_range<MC_TIE_PROBE_ELEMS, kBlockThreads>(
      score, length, offset, period, {in_lo, __float_as_uint(in_lo)},
      {in_hi, __float_as_uint(in_hi)}, scratch, bin_kmin, bin_kmax, tidx);
}

template <bool kKeyCmp, uint32_t VEC, int kTopK, int kBlockThreads, int kStageCap>
__device__ __forceinline__ void mc_scan_tie_body(RowSmem<kTopK, true, kBlockThreads>* s,
                                                 uint64_t* stage, uint64_t* ovf,
                                                 const float* __restrict__ score, int seg_lo,
                                                 int seg_hi, uint32_t tie_key, int tie_cap,
                                                 int* tie_n, uint32_t& t_kmin, uint32_t& t_kmax,
                                                 int tidx) {
  const int lane = tidx & 31;
  const uint32_t lt_mask = (1u << lane) - 1u;
  const float tie_val = to_ordered_inverse(tie_key);
  bool open = true;
  auto slow = [&](float raw, int i, bool tie) {
    const uint32_t key = to_ordered(raw);
    const unsigned act = __activemask();
    const int leader = __ffs(act) - 1;
    const unsigned em = __ballot_sync(act, tie);
    int told = 0;
    if (lane == leader && em) told = atomicAdd(tie_n, __popc(em));
    told = __shfl_sync(act, told, leader);
    const bool keep = !tie || told + __popc(em & lt_mask) < tie_cap;
    const unsigned km = __ballot_sync(act, keep);
    if (km == 0u) return;
    const int kl = __ffs(km) - 1;
    int base = 0;
    if (lane == kl) base = atomicAdd(&s->num_cand[0], __popc(km));
    base = __shfl_sync(act, base, kl);
    if (!keep) return;
    const int pos = base + __popc(km & lt_mask);
    uint64_t* slot = pos < kStageCap ? stage + pos : ovf + (pos - kStageCap);
    *slot = pack_cluster_candidate(i, key);
    t_kmin = min(t_kmin, key);
    t_kmax = max(t_kmax, key);
  };
  auto visit = [&](float raw, int i) {
    bool better, tie;
    if constexpr (kKeyCmp) {
      const uint32_t key = to_ordered(raw);
      better = key < tie_key && raw == raw;
      tie = key == tie_key;
    } else {
      better = raw > tie_val;
      tie = raw == tie_val;
    }
    tie = tie && open;
    if (better || tie) slow(raw, i, tie);
  };
  auto refresh = [&]() { open = *reinterpret_cast<volatile int*>(tie_n) < tie_cap; };
  if constexpr (VEC == 4) {
    const int v_hi = max(seg_lo, seg_hi & ~7);
    const int n4 = (v_hi - seg_lo) >> 2;
    int g = tidx;
    for (; g + kBlockThreads < n4; g += 2 * kBlockThreads) {
      const int base_a = seg_lo + (g << 2);
      const int base_b = seg_lo + ((g + kBlockThreads) << 2);
      const float4 a = ld_float4(score + base_a);
      const float4 b = ld_float4(score + base_b);
      refresh();
      visit(a.x, base_a);
      visit(a.y, base_a + 1);
      visit(a.z, base_a + 2);
      visit(a.w, base_a + 3);
      visit(b.x, base_b);
      visit(b.y, base_b + 1);
      visit(b.z, base_b + 2);
      visit(b.w, base_b + 3);
    }
    if (g < n4) {
      const int base_a = seg_lo + (g << 2);
      const float4 a = ld_float4(score + base_a);
      refresh();
      visit(a.x, base_a);
      visit(a.y, base_a + 1);
      visit(a.z, base_a + 2);
      visit(a.w, base_a + 3);
    }
    for (int i = v_hi + tidx; i < seg_hi; i += kBlockThreads) {
      refresh();
      visit(score[i], i);
    }
  } else {
    for (int i = seg_lo + tidx; i < seg_hi; i += kBlockThreads) {
      refresh();
      visit(score[i], i);
    }
  }
}

// P1 scan under tie mode: strictly-greater candidates go straight
// out, ties collect against the per-CTA cap.
template <uint32_t VEC, int kTopK, int kBlockThreads, int kStageCap>
__device__ __noinline__ void mc_scan_tie(RowSmem<kTopK, true, kBlockThreads>* s, uint64_t* stage,
                                         uint64_t* ovf, const float* __restrict__ score, int seg_lo,
                                         int seg_hi, uint32_t tie_key, int tie_cap, int* tie_n,
                                         uint32_t* seg_kmin, uint32_t* seg_kmax, int tidx) {
  uint32_t t_kmin = 0xFFFFFFFFu;
  uint32_t t_kmax = 0u;
  const bool zero = (to_ordered_inverse(tie_key) == 0.0f);
  if (zero) {
    mc_scan_tie_body<true, VEC, kTopK, kBlockThreads, kStageCap>(
        s, stage, ovf, score, seg_lo, seg_hi, tie_key, tie_cap, tie_n, t_kmin, t_kmax, tidx);
  } else {
    mc_scan_tie_body<false, VEC, kTopK, kBlockThreads, kStageCap>(
        s, stage, ovf, score, seg_lo, seg_hi, tie_key, tie_cap, tie_n, t_kmin, t_kmax, tidx);
  }
  t_kmin = __reduce_min_sync(0xFFFFFFFFu, t_kmin);
  t_kmax = __reduce_max_sync(0xFFFFFFFFu, t_kmax);
  if ((tidx & 31) == 0 && t_kmin <= t_kmax) {
    atomicMin(seg_kmin, t_kmin);
    atomicMax(seg_kmax, t_kmax);
  }
}

// mc finisher under tie mode: once the strict prefix is known to
// cover the top-K, the capped ties are taken from the back of the
// candidate list.
template <uint32_t VEC, int kTopK, int kBlockThreads, int kStageCap, int kSurvCap, int kMinBlocks>
__device__ __noinline__ void mc_finish_tie(RowSmem<kTopK, true, kBlockThreads>* s, uint64_t* stage,
                                           int32_t* surv, const float* __restrict__ score,
                                           int32_t* __restrict__ dst, const uint64_t* row_cand,
                                           int32_t* row_total, int32_t* row_nkmin,
                                           int32_t* row_kmax, int32_t* row_ovf,
                                           int32_t* __restrict__ spill0, uint32_t n_cols,
                                           int length, uint32_t tie_key, int tidx) {
  __shared__ int sh_total;
  __shared__ int sh_ovf;
  __shared__ uint32_t sh_kmin;
  __shared__ uint32_t sh_kmax;
  if (tidx == 0) {
    sh_total = __ldcg(row_total);
    sh_ovf = __ldcg(row_ovf);
    sh_kmin = ~static_cast<uint32_t>(__ldcg(row_nkmin));
    sh_kmax = static_cast<uint32_t>(__ldcg(row_kmax));
    *row_total = 0;
    *row_ovf = 0;
    *row_nkmin = 0;
    *row_kmax = 0;
  }
  __syncthreads();
  const int total = sh_total;
  if (total <= kTopK || sh_ovf) {
    run_cluster_sampled_short_row<VEC, kTopK, kBlockThreads, kMinBlocks>(s, score, dst, spill0,
                                                                         n_cols, length, tidx);
    return;
  }
  const bool staged = total <= kStageCap;
  const uint32_t kmin = sh_kmin;
  const uint32_t span = sh_kmax - kmin;
  const int s0 = span < static_cast<uint32_t>(RADIX) ? 0 : (32 - __clz(span)) - 11;
  auto relbin = [&](uint32_t key) { return static_cast<int>((key - kmin) >> s0); };
  for (int i = tidx; i < HIST_SWEEP; i += kBlockThreads) s->histogram[i] = 0;
  __syncthreads();
  constexpr int kBatch = 4;
  for (int j0 = tidx; j0 < total; j0 += kBatch * kBlockThreads) {
    uint64_t pr[kBatch];
#pragma unroll
    for (int b = 0; b < kBatch; ++b) {
      const int j = j0 + b * kBlockThreads;
      pr[b] = j < total ? __ldcg(reinterpret_cast<const uint64_t*>(row_cand + j)) : 0ull;
    }
#pragma unroll
    for (int b = 0; b < kBatch; ++b) {
      const int j = j0 + b * kBlockThreads;
      const uint32_t key = cluster_candidate_key(pr[b]);
      const bool tie = j < total && key == tie_key;
      const unsigned tm = __ballot_sync(__activemask(), tie);
      if (j < total) {
        if (!tie) {
          atomicAdd(&s->histogram[relbin(key)], 1);
        } else if ((tidx & 31) == __ffs(tm) - 1) {
          atomicAdd(&s->histogram[relbin(tie_key)], __popc(tm));
        }
        if (staged) stage[j] = pr[b];
      }
    }
  }
  __syncthreads();
  prefix_sum_find_threshold(s, tidx, kTopK, 1, &s->ties_remaining, kTopK);
  __syncthreads();
  const int thr = s->threshold_bin;
  int rem = kTopK;
  if (thr > 0) rem -= s->histogram[thr - 1];
  const int tie_count = s->histogram[thr] - (thr > 0 ? s->histogram[thr - 1] : 0);
  __syncthreads();
  __shared__ int sh_tie_n;
  if (tidx == 0) {
    s->num_selected = 0;
    s->num_cand[0] = 0;
    sh_tie_n = 0;
  }
  __syncthreads();
  const bool surv_smem = kSurvCap > 0 && tie_count <= kSurvCap;
  int32_t* ids0 = surv_smem ? surv : spill0;
  uint32_t* keys0 = reinterpret_cast<uint32_t*>(surv_smem ? surv + kSurvCap : spill0 + n_cols);
  int32_t* ids1 = surv_smem ? surv + 2 * kSurvCap : spill0 + 2 * n_cols;
  uint32_t* keys1 =
      reinterpret_cast<uint32_t*>(surv_smem ? surv + 3 * kSurvCap : spill0 + 3 * n_cols);
  const int lane = tidx & 31;
  const uint32_t lt_mask = (1u << lane) - 1u;
  auto route_tie = [&](uint64_t pair, bool valid) {
    const uint32_t key = cluster_candidate_key(pair);
    const int bin = relbin(key);
    const bool in_bin = valid && bin == thr && rem != 0;
    const bool tie = in_bin && (s0 == 0 || key == tie_key);
    if (valid && bin < thr) {
      s->indices[atomicAdd(&s->num_selected, 1)] = cluster_candidate_index(pair);
    } else if (in_bin && !tie) {
      const int pos = atomicAdd(&s->num_cand[0], 1);
      ids0[pos] = cluster_candidate_index(pair);
      keys0[pos] = key;
    }
    const unsigned act = __activemask();
    const unsigned tm = __ballot_sync(act, tie);
    if (tm != 0u) {
      const int leader = __ffs(tm) - 1;
      int tbase = 0;
      if (lane == leader) tbase = atomicAdd(&sh_tie_n, __popc(tm));
      tbase = __shfl_sync(act, tbase, leader);
      const int pos = tbase + __popc(tm & lt_mask);
      if (tie && pos < rem) s->indices[kTopK - 1 - pos] = cluster_candidate_index(pair);
    }
  };
  auto sweep = [&](auto&& route) {
    if (staged) {
      for (int i = tidx; i < total; i += kBlockThreads) route(stage[i], true);
    } else {
      for (int j0 = tidx; j0 < total; j0 += kBatch * kBlockThreads) {
        uint64_t pr[kBatch];
#pragma unroll
        for (int b = 0; b < kBatch; ++b) {
          const int j = j0 + b * kBlockThreads;
          pr[b] = j < total ? __ldcg(reinterpret_cast<const uint64_t*>(row_cand + j)) : 0ull;
        }
#pragma unroll
        for (int b = 0; b < kBatch; ++b) route(pr[b], j0 + b * kBlockThreads < total);
      }
    }
  };
  sweep(route_tie);
  __syncthreads();
  const int surv_n = s->num_cand[0];
  if (rem != 0 && surv_n > 0) {
    const int base = s->num_selected;
    if (surv_n <= rem) {
      for (int i = tidx; i < surv_n; i += kBlockThreads) s->indices[base + i] = ids0[i];
      __syncthreads();
    } else if (surv_n <= kBlockThreads) {
      if (tidx < surv_n) {
        const uint32_t mine = keys0[tidx];
        int rank = 0;
        for (int j = 0; j < surv_n; ++j) {
          const uint32_t other = keys0[j];
          rank += (other < mine) || (other == mine && j < tidx);
        }
        if (rank < rem) s->indices[base + rank] = ids0[tidx];
      }
      __syncthreads();
    } else {
      for (int i = tidx; i < HIST_SWEEP; i += kBlockThreads) s->histogram[i] = 0;
      __syncthreads();
      for (int i = tidx; i < surv_n; i += kBlockThreads) {
        atomicAdd(&s->histogram[(keys0[i] >> kFineShifts[0]) & kFineMasks[0]], 1);
      }
      __syncthreads();
      refine_rounds_split_keys<kTopK, kBlockThreads>(s, ids0, keys0, ids1, keys1, surv_n, rem, tidx,
                                                     0);
    }
  }
  for (int i = tidx; i < kTopK; i += kBlockThreads) dst[i] = s->indices[i];
}

// mc finisher: bins the row's staged candidates by their actual key
// range and resolves the boundary survivors; a single-valued bin
// resolves directly.
template <uint32_t VEC, int kTopK, int kBlockThreads, int kStageCap, int kSurvCap, int kMinBlocks>
__device__ __noinline__ void mc_finish(RowSmem<kTopK, true, kBlockThreads>* s, uint64_t* stage,
                                       int32_t* surv, const float* __restrict__ score,
                                       int32_t* __restrict__ dst, const uint64_t* row_cand,
                                       int32_t* row_total, int32_t* row_nkmin, int32_t* row_kmax,
                                       int32_t* row_ovf, int32_t* __restrict__ spill0,
                                       uint32_t n_cols, int length, const McTieState* ts,
                                       int tidx) {
  if (ts->tie) {
    mc_finish_tie<VEC, kTopK, kBlockThreads, kStageCap, kSurvCap, kMinBlocks>(
        s, stage, surv, score, dst, row_cand, row_total, row_nkmin, row_kmax, row_ovf, spill0,
        n_cols, length, ts->kmin, tidx);
    return;
  }
  __shared__ int sh_total;
  __shared__ int sh_ovf;
  __shared__ uint32_t sh_kmin;
  __shared__ uint32_t sh_kmax;
  if (tidx == 0) {
    sh_total = __ldcg(row_total);
    sh_ovf = __ldcg(row_ovf);
    sh_kmin = ~static_cast<uint32_t>(__ldcg(row_nkmin));
    sh_kmax = static_cast<uint32_t>(__ldcg(row_kmax));
    *row_total = 0;
    *row_ovf = 0;
    *row_nkmin = 0;
    *row_kmax = 0;
  }
  __syncthreads();
  const int total = sh_total;
  if (total <= kTopK || sh_ovf) {
    run_cluster_sampled_short_row<VEC, kTopK, kBlockThreads, kMinBlocks>(s, score, dst, spill0,
                                                                         n_cols, length, tidx);
    return;
  }
  const bool staged = total <= kStageCap;
  const uint32_t kmin = sh_kmin;
  const uint32_t span = sh_kmax - kmin;
  const int s0 = span < static_cast<uint32_t>(RADIX) ? 0 : (32 - __clz(span)) - 11;
  auto relbin = [&](uint32_t key) { return static_cast<int>((key - kmin) >> s0); };
  for (int i = tidx; i < HIST_SWEEP; i += kBlockThreads) s->histogram[i] = 0;
  __syncthreads();
  constexpr int kBatch = 4;
  for (int j0 = tidx; j0 < total; j0 += kBatch * kBlockThreads) {
    uint64_t pr[kBatch];
#pragma unroll
    for (int b = 0; b < kBatch; ++b) {
      const int j = j0 + b * kBlockThreads;
      pr[b] = j < total ? __ldcg(reinterpret_cast<const uint64_t*>(row_cand + j)) : 0ull;
    }
#pragma unroll
    for (int b = 0; b < kBatch; ++b) {
      const int j = j0 + b * kBlockThreads;
      if (j < total) {
        atomicAdd(&s->histogram[relbin(cluster_candidate_key(pr[b]))], 1);
        if (staged) stage[j] = pr[b];
      }
    }
  }
  __syncthreads();
  prefix_sum_find_threshold(s, tidx, kTopK, 1, &s->ties_remaining, kTopK);
  __syncthreads();
  const int thr = s->threshold_bin;
  int rem = kTopK;
  if (thr > 0) rem -= s->histogram[thr - 1];
  const int tie_count = s->histogram[thr] - (thr > 0 ? s->histogram[thr - 1] : 0);
  __syncthreads();
  __shared__ uint32_t sh_bkmin;
  __shared__ uint32_t sh_bkmax;
  if (tidx == 0) {
    s->num_selected = 0;
    s->num_cand[0] = 0;
    sh_bkmin = 0xFFFFFFFFu;
    sh_bkmax = 0u;
  }
  __syncthreads();
  const bool surv_smem = kSurvCap > 0 && tie_count <= kSurvCap;
  int32_t* ids0 = surv_smem ? surv : spill0;
  uint32_t* keys0 = reinterpret_cast<uint32_t*>(surv_smem ? surv + kSurvCap : spill0 + n_cols);
  int32_t* ids1 = surv_smem ? surv + 2 * kSurvCap : spill0 + 2 * n_cols;
  uint32_t* keys1 =
      reinterpret_cast<uint32_t*>(surv_smem ? surv + 3 * kSurvCap : spill0 + 3 * n_cols);
  const bool single_key_bin = (s0 == 0);
  auto route = [&](uint64_t pair) {
    const uint32_t key = cluster_candidate_key(pair);
    const int bin = relbin(key);
    if (bin < thr) {
      s->indices[atomicAdd(&s->num_selected, 1)] = cluster_candidate_index(pair);
    } else if (bin == thr && rem != 0) {
      const int pos = atomicAdd(&s->num_cand[0], 1);
      if (single_key_bin) {
        if (pos < rem) s->indices[kTopK - rem + pos] = cluster_candidate_index(pair);
      } else {
        ids0[pos] = cluster_candidate_index(pair);
        keys0[pos] = key;
        if (!surv_smem) {
          const unsigned act = __activemask();
          const uint32_t wlo = __reduce_min_sync(act, key);
          const uint32_t whi = __reduce_max_sync(act, key);
          if ((tidx & 31) == __ffs(act) - 1) {
            atomicMin(&sh_bkmin, wlo);
            atomicMax(&sh_bkmax, whi);
          }
        }
      }
    }
  };
  if (staged) {
    for (int i = tidx; i < total; i += kBlockThreads) route(stage[i]);
  } else {
    for (int j0 = tidx; j0 < total; j0 += kBatch * kBlockThreads) {
      uint64_t pr[kBatch];
#pragma unroll
      for (int b = 0; b < kBatch; ++b) {
        const int j = j0 + b * kBlockThreads;
        pr[b] = j < total ? __ldcg(reinterpret_cast<const uint64_t*>(row_cand + j)) : 0ull;
      }
#pragma unroll
      for (int b = 0; b < kBatch; ++b) {
        if (j0 + b * kBlockThreads < total) route(pr[b]);
      }
    }
  }
  __syncthreads();
  if (rem != 0 && !single_key_bin) {
    const int base = s->num_selected;
    if (!surv_smem && sh_bkmin == sh_bkmax) {
      for (int i = tidx; i < rem; i += kBlockThreads) s->indices[base + i] = ids0[i];
      __syncthreads();
    } else if (tie_count <= kBlockThreads) {
      if (tidx < tie_count) {
        const uint32_t mine = keys0[tidx];
        int rank = 0;
        for (int j = 0; j < tie_count; ++j) {
          const uint32_t other = keys0[j];
          rank += (other < mine) || (other == mine && j < tidx);
        }
        if (rank < rem) s->indices[base + rank] = ids0[tidx];
      }
      __syncthreads();
    } else {
      for (int i = tidx; i < HIST_SWEEP; i += kBlockThreads) s->histogram[i] = 0;
      __syncthreads();
      for (int i = tidx; i < tie_count; i += kBlockThreads) {
        atomicAdd(&s->histogram[(keys0[i] >> kFineShifts[0]) & kFineMasks[0]], 1);
      }
      __syncthreads();
      refine_rounds_split_keys<kTopK, kBlockThreads>(s, ids0, keys0, ids1, keys1, tie_count, rem,
                                                     tidx, 0);
    }
  }
  for (int i = tidx; i < kTopK; i += kBlockThreads) dst[i] = s->indices[i];
}

// Never-truncating P1 scan, called out of line by CTAs whose
// predicted stage occupancy would overflow the lean inline loop.
template <uint32_t VEC, int kTopK, int kBlockThreads, int kStageCap>
__device__ __noinline__ void mc_scan_robust(RowSmem<kTopK, true, kBlockThreads>* s, uint64_t* stage,
                                            uint64_t* ovf, const float* __restrict__ score,
                                            int seg_lo, int seg_hi, float boundary,
                                            uint32_t* seg_kmin, uint32_t* seg_kmax, int tidx) {
  const uint32_t boundary_bits = __float_as_uint(boundary);
  uint32_t t_kmin = 0xFFFFFFFFu;
  uint32_t t_kmax = 0u;
  auto admit = [&](float raw, int i) {
    const uint32_t key = to_ordered(raw);
    const int pos = atomicAdd(&s->num_cand[0], 1);
    uint64_t* slot = pos < kStageCap ? stage + pos : ovf + (pos - kStageCap);
    *slot = pack_cluster_candidate(i, key);
    t_kmin = min(t_kmin, key);
    t_kmax = max(t_kmax, key);
  };
  auto seg_split = [&](float raw, int i) {
    if (raw >= boundary) {
      admit(raw, i);
    }
  };
  const bool boundary_is_zero = (boundary_bits & 0x7FFFFFFFu) == 0u;
  if (boundary_is_zero) {
    for (int i = seg_lo + tidx; i < seg_hi; i += kBlockThreads) {
      const float raw = score[i];
      if (fp32_boundary_accepts(raw, boundary, boundary_bits)) admit(raw, i);
    }
  } else if constexpr (VEC == 4) {
    const int v_hi = max(seg_lo, seg_hi & ~7);
    const int n4 = (v_hi - seg_lo) >> 2;
    int g = tidx;
    for (; g + kBlockThreads < n4; g += 2 * kBlockThreads) {
      const int base_a = seg_lo + (g << 2);
      const int base_b = seg_lo + ((g + kBlockThreads) << 2);
      const float4 a = ld_float4(score + base_a);
      const float4 b = ld_float4(score + base_b);
      seg_split(a.x, base_a);
      seg_split(a.y, base_a + 1);
      seg_split(a.z, base_a + 2);
      seg_split(a.w, base_a + 3);
      seg_split(b.x, base_b);
      seg_split(b.y, base_b + 1);
      seg_split(b.z, base_b + 2);
      seg_split(b.w, base_b + 3);
    }
    if (g < n4) {
      const int base_a = seg_lo + (g << 2);
      const float4 a = ld_float4(score + base_a);
      seg_split(a.x, base_a);
      seg_split(a.y, base_a + 1);
      seg_split(a.z, base_a + 2);
      seg_split(a.w, base_a + 3);
    }
    for (int i = v_hi + tidx; i < seg_hi; i += kBlockThreads) seg_split(score[i], i);
  } else {
    for (int i = seg_lo + tidx; i < seg_hi; i += kBlockThreads) seg_split(score[i], i);
  }
  if (t_kmin <= t_kmax) {
    atomicMin(seg_kmin, t_kmin);
    atomicMax(seg_kmax, t_kmax);
  }
}

// Store variant for CTAs past MC_ROBUST_STORE_PCT of the stage.
template <uint32_t VEC, int kTopK, int kBlockThreads, int kStageCap>
__device__ __noinline__ void mc_scan_fat(RowSmem<kTopK, true, kBlockThreads>* s, uint64_t* stage,
                                         uint64_t* ovf, const float* __restrict__ score, int seg_lo,
                                         int seg_hi, float boundary, McTieState* ts,
                                         uint32_t* seg_kmin, uint32_t* seg_kmax, int tidx) {
  if (ts->tie) {
    mc_scan_tie<VEC, kTopK, kBlockThreads, kStageCap>(
        s, stage, ovf, score, seg_lo, seg_hi, ts->kmin, ts->cap, &ts->n, seg_kmin, seg_kmax, tidx);
  } else {
    mc_scan_robust<VEC, kTopK, kBlockThreads, kStageCap>(s, stage, ovf, score, seg_lo, seg_hi,
                                                         boundary, seg_kmin, seg_kmax, tidx);
  }
}

// mc kernel: each row is cut into S segments, one CTA per segment. P0
// samples the same positions in every segment at a shared runtime
// stride, so every CTA derives the same coarse boundary with no
// synchronization; P1 scans the CTA's own segment into the row's
// staging list; P2 reserves a range of the candidate list with one
// global atomic, and the last CTA to arrive finishes the row without
// spinning, so co-residency is not required.
template <uint32_t VEC, int kTopK, int kBlockThreads, int kSegCap, int kStageCap, int kSurvCap,
          int kMinBlocks>
__global__ void __launch_bounds__(kBlockThreads, kMinBlocks)
    topk_filtered_mc_kernel(const float* __restrict__ input, int32_t* __restrict__ output,
                            const int32_t* __restrict__ ke, int32_t* __restrict__ spill_buffer,
                            uint64_t* __restrict__ ws_cand, int32_t* __restrict__ arrive,
                            int32_t* __restrict__ mc_ctr,
                            const int32_t* __restrict__ num_valid_rows, uint32_t row_stride,
                            uint32_t out_stride, uint32_t n_cols, int num_segs) {
  namespace cg = cooperative_groups;
  using Smem = RowSmem<kTopK, true, kBlockThreads>;
  static_assert(sizeof(Smem) % 8 == 0);
  cg::cluster_group cluster = cg::this_cluster();
  const int tidx = threadIdx.x;
  const int rank = static_cast<int>(cluster.block_rank());
  const int row_i = static_cast<int>(blockIdx.z);
  if (row_i >= *num_valid_rows) return;
  const int seg =
      static_cast<int>(blockIdx.y) * static_cast<int>(gridDim.x) + static_cast<int>(blockIdx.x);
  extern __shared__ uint8_t smem_raw[];
  Smem* s = reinterpret_cast<Smem*>(smem_raw);
  uint64_t* stage = reinterpret_cast<uint64_t*>(smem_raw + sizeof(Smem));
  int32_t* surv = reinterpret_cast<int32_t*>(stage + kStageCap);
  const float* score = input + static_cast<size_t>(row_i) * row_stride;
  int32_t* dst = output + static_cast<size_t>(row_i) * out_stride;
  int32_t* spill0 = spill_buffer + static_cast<size_t>(row_i) * 4 * n_cols;
  const int length = ke[row_i];
  if (length <= kTopK) {
    if (seg == 0) {
      for (int i = tidx; i < kTopK; i += kBlockThreads) dst[i] = (i < length) ? i : -1;
    }
    return;
  }
  int stride_shift = 4;
  while ((length >> stride_shift) > MC_TARGET_SAMPLES) ++stride_shift;
  const int stride = 1 << stride_shift;
  const int period = 32 * stride;
  const int sample_rank = mc_sample_rank(kTopK, stride);
  const bool usable = length >= (sample_rank + 1) * stride + 2 * period;
  if (!usable) {
    if (seg == 0) {
      run_cluster_sampled_short_row<VEC, kTopK, kBlockThreads, kMinBlocks>(s, score, dst, spill0,
                                                                           n_cols, length, tidx);
    }
    return;
  }
  __shared__ float sh_boundary;
  __shared__ int sh_last;
  __shared__ uint32_t sh_seg_kmin;
  __shared__ uint32_t sh_seg_kmax;
  __shared__ int sh_fat;
  __shared__ McTieState ts;
  if (tidx == 0) {
    sh_seg_kmin = 0xFFFFFFFFu;
    sh_seg_kmax = 0u;
    ts.kmin = 0xFFFFFFFFu;
    ts.kmax = 0u;
    ts.n = 0;
  }
  for (int i = tidx; i < HIST_SWEEP; i += kBlockThreads) s->histogram[i] = 0;
  __syncthreads();
  const int offset =
      static_cast<int>((static_cast<uint32_t>(row_i) * 17u + 13u) & (stride - 1)) * 32;
  mc_sample<kBlockThreads>(s->histogram, score, length, offset, period, 0, 1, tidx);
  __syncthreads();
  prefix_sum_find_threshold(s, tidx, sample_rank, 0, &s->num_selected, 0);
  if (tidx == 0) {
    sh_boundary = coarse_boundary(s->threshold_bin).value;
    const int64_t pred = static_cast<int64_t>(s->histogram[s->threshold_bin]) * stride;
    sh_fat = pred / num_segs > (static_cast<int64_t>(kStageCap) * MC_ROBUST_STORE_PCT) / 100;
    const int thr = s->threshold_bin;
    const int bin_samples = s->histogram[thr] - (thr > 0 ? s->histogram[thr - 1] : 0);
    ts.probe = 0;
    if (bin_samples >= MC_TIE_MIN_SAMPLES) {
      const int per_seg = (kTopK + num_segs - 1) / num_segs;
      const int tie_cap = min(kTopK, max(MC_TIE_CAP_MULT * per_seg, MC_TIE_CAP_MIN));
      ts.cap = tie_cap;
      ts.probe = static_cast<int64_t>(bin_samples) * stride >=
                 static_cast<int64_t>(MC_TIE_GAIN) * tie_cap * num_segs;
    }
    if (ts.probe) {
      ts.in_lo = pack_coarse_boundary(coarse_boundary(thr)).value;
      ts.in_hi =
          pack_coarse_boundary(thr > 0 ? coarse_boundary(thr - 1) : CoarseBoundary{0.0f, true})
              .value;
    }
  }
  __syncthreads();
  if (ts.probe) {
    mc_tie_probe<kBlockThreads>(score, length, offset, period, ts.in_lo, ts.in_hi,
                                reinterpret_cast<uint32_t*>(s->histogram), &ts.kmin, &ts.kmax,
                                tidx);
  }
  const float boundary = sh_boundary;
  const uint32_t boundary_bits = __float_as_uint(boundary);
  const int seg_len = (((length + num_segs - 1) / num_segs) + 7) & ~7;
  const int seg_lo = min(seg * seg_len, length);
  const int seg_hi = min(seg_lo + seg_len, length);
  if (tidx == 0) {
    s->num_cand[0] = 0;
    ts.tie = ts.probe && ts.kmin == ts.kmax;
    if (ts.tie) sh_fat = 1;
  }
  __syncthreads();
  const bool fat = sh_fat != 0;
  uint64_t* ovf = reinterpret_cast<uint64_t*>(spill0) + seg_lo;
  if (fat) {
    mc_scan_fat<VEC, kTopK, kBlockThreads, kStageCap>(
        s, stage, ovf, score, seg_lo, seg_hi, boundary, &ts, &sh_seg_kmin, &sh_seg_kmax, tidx);
  } else {
    auto admit = [&](float raw, int i) {
      const uint32_t key = to_ordered(raw);
      const int pos = atomicAdd(&s->num_cand[0], 1);
      if (pos < kStageCap) stage[pos] = pack_cluster_candidate(i, key);
      atomicMin(&sh_seg_kmin, key);
      atomicMax(&sh_seg_kmax, key);
    };
    auto seg_split = [&](float raw, int i) {
      if (raw >= boundary) {
        admit(raw, i);
      }
    };
    const bool boundary_is_zero = (boundary_bits & 0x7FFFFFFFu) == 0u;
    if (boundary_is_zero) {
      for (int i = seg_lo + tidx; i < seg_hi; i += kBlockThreads) {
        const float raw = score[i];
        if (fp32_boundary_accepts(raw, boundary, boundary_bits)) admit(raw, i);
      }
    } else if constexpr (VEC == 4) {
      const int v_hi = max(seg_lo, seg_hi & ~7);
      const int n4 = (v_hi - seg_lo) >> 2;
      int g = tidx;
      for (; g + kBlockThreads < n4; g += 2 * kBlockThreads) {
        const int base_a = seg_lo + (g << 2);
        const int base_b = seg_lo + ((g + kBlockThreads) << 2);
        const float4 a = ld_float4(score + base_a);
        const float4 b = ld_float4(score + base_b);
        seg_split(a.x, base_a);
        seg_split(a.y, base_a + 1);
        seg_split(a.z, base_a + 2);
        seg_split(a.w, base_a + 3);
        seg_split(b.x, base_b);
        seg_split(b.y, base_b + 1);
        seg_split(b.z, base_b + 2);
        seg_split(b.w, base_b + 3);
      }
      if (g < n4) {
        const int base_a = seg_lo + (g << 2);
        const float4 a = ld_float4(score + base_a);
        seg_split(a.x, base_a);
        seg_split(a.y, base_a + 1);
        seg_split(a.z, base_a + 2);
        seg_split(a.w, base_a + 3);
      }
      for (int i = v_hi + tidx; i < seg_hi; i += kBlockThreads) seg_split(score[i], i);
    } else {
      for (int i = seg_lo + tidx; i < seg_hi; i += kBlockThreads) seg_split(score[i], i);
    }
  }
  const int m_rows = static_cast<int>(gridDim.z);
  int32_t* row_total = mc_ctr + row_i;
  int32_t* row_nkmin = mc_ctr + m_rows + row_i;
  int32_t* row_kmax = mc_ctr + 2 * m_rows + row_i;
  int32_t* row_ovf = mc_ctr + 3 * m_rows + row_i;
  const int row_cap = static_cast<int>(n_cols);
  uint64_t* row_cand = ws_cand + static_cast<size_t>(row_i) * n_cols;
  __shared__ int sh_base;
  __syncthreads();
  const int local = sh_fat ? s->num_cand[0] : min(s->num_cand[0], kStageCap);
  if (tidx == 0) {
    const int base = local > 0 ? atomicAdd(row_total, local) : 0;
    sh_base = base;
    if ((!sh_fat && s->num_cand[0] > kStageCap) || base + local > row_cap) atomicOr(row_ovf, 1);
    if (local > 0) {
      atomicMax(reinterpret_cast<unsigned int*>(row_nkmin), ~sh_seg_kmin);
      atomicMax(reinterpret_cast<unsigned int*>(row_kmax), sh_seg_kmax);
    }
  }
  __syncthreads();
  {
    const int base = sh_base;
    const int n_copy = min(local, row_cap - base);
    const int n_stage = min(n_copy, kStageCap);
    for (int i = tidx; i < n_stage; i += kBlockThreads) row_cand[base + i] = stage[i];
    for (int i = n_stage + tidx; i < n_copy; i += kBlockThreads) {
      row_cand[base + i] = __ldcg(reinterpret_cast<const uint64_t*>(ovf + (i - kStageCap)));
    }
  }
  __threadfence();
  __syncthreads();
  if (tidx == 0) {
    const int prev = atomicAdd(arrive + row_i, 1);
    sh_last = (prev == num_segs - 1);
    if (sh_last) arrive[row_i] = 0;
  }
  __syncthreads();
  if (!sh_last) return;
  __threadfence();
  mc_finish<VEC, kTopK, kBlockThreads, kStageCap, kSurvCap, kMinBlocks>(
      s, stage, surv, score, dst, row_cand, row_total, row_nkmin, row_kmax, row_ovf, spill0, n_cols,
      length, &ts, tidx);
}

// Raises the kernel's dynamic smem opt-in limit; false when the
// request exceeds the device maximum.
inline bool opt_in_dynamic_smem(const void* kernel, size_t smem_bytes) {
  static const int max_smem = [] {
    int dev = 0;
    int bytes = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    return bytes;
  }();
  if (smem_bytes > static_cast<size_t>(max_smem)) {
    return false;
  }
  return cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                              static_cast<int>(smem_bytes)) == cudaSuccess;
}

inline uint32_t vector_width(int row_stride) {
  return (row_stride % 4 == 0) ? 4u : (row_stride % 2 == 0) ? 2u : 1u;
}

template <int kTopK, bool kSampled, int kSampleStride = SAMPLE_STRIDE,
          int kSampleRank = SAMPLE_RANK>
bool launch_one_cta_per_row(int* topk_indices, const float* logits, const int* ke,
                            const int* num_valid_rows, int m, int n, int row_stride, int out_stride,
                            void* counters, size_t counters_bytes, void* workspace,
                            size_t workspace_bytes, cudaStream_t stream) {
  if (counters_bytes < HEAD_COUNTER_INTS * sizeof(int32_t) ||
      workspace_bytes < one_cta_workspace_bytes(n)) {
    return false;
  }
  int32_t* work_counter = static_cast<int32_t*>(counters);
  int32_t* exit_counter = work_counter + 1;
  int32_t* spill = static_cast<int32_t*>(workspace);
  const uint32_t vec = vector_width(row_stride);
  auto kernel = (vec == 4)   ? topk_filtered_kernel<4, kTopK, kSampled, kSampleStride, kSampleRank>
                : (vec == 2) ? topk_filtered_kernel<2, kTopK, kSampled, kSampleStride, kSampleRank>
                             : topk_filtered_kernel<1, kTopK, kSampled, kSampleStride, kSampleRank>;
  const size_t smem_bytes = sizeof(RowSmem<kTopK, kSampled>);
  if (!opt_in_dynamic_smem(reinterpret_cast<const void*>(kernel), smem_bytes)) {
    return false;
  }
  const int num_blocks = std::max(1, std::min(m, grid_num_blocks()));
  kernel<<<num_blocks, THREADS, smem_bytes, stream>>>(
      logits, topk_indices, ke, spill, work_counter, exit_counter, num_valid_rows,
      static_cast<uint32_t>(row_stride), static_cast<uint32_t>(out_stride),
      static_cast<uint32_t>(n), m);
  return cudaGetLastError() == cudaSuccess;
}

template <int kTopK, int kMinBlocks>
bool launch_row_local_register(int* topk_indices, const float* logits, const int* ke,
                               const int* num_valid_rows, int m, int n, int row_stride,
                               int out_stride, cudaStream_t stream) {
  constexpr int kRegister2MaxN = SHORT_THREADS * 4 * 2;
  constexpr int kRegister3MaxN = SHORT_THREADS * 4 * 3;
  constexpr int kRegister4MaxN = SHORT_THREADS * 4 * 4;
  if (m <= 0 || n > kRegister4MaxN || vector_width(row_stride) != 4) {
    return false;
  }
  const auto kernel =
      n <= kRegister2MaxN   ? topk_filtered_row_local_register_kernel<2, kTopK, kMinBlocks>
      : n <= kRegister3MaxN ? topk_filtered_row_local_register_kernel<3, kTopK, kMinBlocks>
                            : topk_filtered_row_local_register_kernel<4, kTopK, kMinBlocks>;
  kernel<<<m, SHORT_THREADS, 0, stream>>>(logits, topk_indices, ke, num_valid_rows,
                                          static_cast<uint32_t>(row_stride),
                                          static_cast<uint32_t>(out_stride));
  return cudaGetLastError() == cudaSuccess;
}

template <int kTopK, int kMinBlocks>
bool launch_row_local_register4_if_short(int* topk_indices, const float* logits, const int* ke,
                                         const int* num_valid_rows, int m, int row_stride,
                                         int out_stride, cudaStream_t stream) {
  if (m <= 0 || vector_width(row_stride) != 4) {
    return false;
  }
  topk_filtered_row_local_register_kernel<4, kTopK, kMinBlocks><<<m, SHORT_THREADS, 0, stream>>>(
      logits, topk_indices, ke, num_valid_rows, static_cast<uint32_t>(row_stride),
      static_cast<uint32_t>(out_stride));
  return cudaGetLastError() == cudaSuccess;
}

template <int kTopK, bool kBounded, int kMinBlocks, int kMinLength = 0>
bool launch_row_local_streaming(int* topk_indices, const float* logits, const int* ke,
                                const int* num_valid_rows, int m, int n, int row_stride,
                                int out_stride, void* workspace, size_t workspace_bytes,
                                void* counters, size_t counters_bytes, cudaStream_t stream) {
  if (m <= 0 || workspace_bytes < one_cta_workspace_bytes(n) ||
      counters_bytes < HEAD_COUNTER_INTS * sizeof(int32_t)) {
    return false;
  }
  int32_t* spill = static_cast<int32_t*>(workspace);
  int32_t* work_counter = static_cast<int32_t*>(counters);
  int32_t* exit_counter = work_counter + 1;
  const uint32_t vec = vector_width(row_stride);
  const auto kernel =
      (vec == 4)
          ? topk_filtered_row_local_streaming_kernel<4, kTopK, kBounded, kMinBlocks, kMinLength>
      : (vec == 2)
          ? topk_filtered_row_local_streaming_kernel<2, kTopK, kBounded, kMinBlocks, kMinLength>
          : topk_filtered_row_local_streaming_kernel<1, kTopK, kBounded, kMinBlocks, kMinLength>;
  const int num_blocks = std::max(1, std::min(m, grid_num_blocks()));
  kernel<<<num_blocks, SHORT_THREADS, 0, stream>>>(
      logits, topk_indices, ke, spill, work_counter, exit_counter, num_valid_rows,
      static_cast<uint32_t>(row_stride), static_cast<uint32_t>(out_stride),
      static_cast<uint32_t>(n), m);
  return cudaGetLastError() == cudaSuccess;
}

template <int kTopK>
bool launch_row_local_1024_exact(int* topk_indices, const float* logits, const int* ke,
                                 const int* num_valid_rows, int m, int n, int row_stride,
                                 int out_stride, void* workspace, size_t workspace_bytes,
                                 void* counters, size_t counters_bytes, cudaStream_t stream) {
  if (n <= 16384 && vector_width(row_stride) == 4) {
    return m <= get_sm_count()
               ? launch_row_local_register<kTopK, 1>(topk_indices, logits, ke, num_valid_rows, m, n,
                                                     row_stride, out_stride, stream)
               : launch_row_local_register<kTopK, 2>(topk_indices, logits, ke, num_valid_rows, m, n,
                                                     row_stride, out_stride, stream);
  }
  if (m > grid_num_blocks() && vector_width(row_stride) == 4) {
    if (!launch_row_local_register4_if_short<kTopK, 2>(topk_indices, logits, ke, num_valid_rows, m,
                                                       row_stride, out_stride, stream)) {
      return false;
    }
    return launch_row_local_streaming<kTopK, true, 2, 16384>(
        topk_indices, logits, ke, num_valid_rows, m, n, row_stride, out_stride, workspace,
        workspace_bytes, counters, counters_bytes, stream);
  }
  if (m > grid_num_blocks()) {
    return launch_row_local_streaming<kTopK, true, 2>(
        topk_indices, logits, ke, num_valid_rows, m, n, row_stride, out_stride, workspace,
        workspace_bytes, counters, counters_bytes, stream);
  }
  return m <= get_sm_count()
             ? launch_row_local_streaming<kTopK, false, 1>(
                   topk_indices, logits, ke, num_valid_rows, m, n, row_stride, out_stride,
                   workspace, workspace_bytes, counters, counters_bytes, stream)
             : launch_row_local_streaming<kTopK, false, 2>(
                   topk_indices, logits, ke, num_valid_rows, m, n, row_stride, out_stride,
                   workspace, workspace_bytes, counters, counters_bytes, stream);
}

// KV-split launcher: enforces m*kSplits <= MIN_BLOCKS*sm_count so the
// spin exchange's co-residency precondition holds.
template <int kTopK, int kSplits, int kSampleStride = SAMPLE_STRIDE, int kSampleRank = SAMPLE_RANK>
bool launch_kv_split(int* topk_indices, const float* logits, const int* ke,
                     const int* num_valid_rows, int m, int n, int row_stride, int out_stride,
                     void* counters, size_t counters_bytes, void* workspace, size_t workspace_bytes,
                     cudaStream_t stream) {
  if (counters_bytes < counters_bytes_for(m) ||
      workspace_bytes < split_workspace_bytes(m, kSplits, n)) {
    return false;
  }
  if (static_cast<int64_t>(m) * kSplits > static_cast<int64_t>(MIN_BLOCKS) * get_sm_count()) {
    return false;
  }
  int32_t* arrive = static_cast<int32_t*>(counters) + HEAD_COUNTER_INTS;
  int32_t* spill = static_cast<int32_t*>(workspace);
  int32_t* split_buffer = spill + split_spill_ints(m, n);
  const uint32_t vec = vector_width(row_stride);
  auto kernel =
      (vec == 4)   ? topk_filtered_split_kernel<4, kTopK, kSplits, kSampleStride, kSampleRank>
      : (vec == 2) ? topk_filtered_split_kernel<2, kTopK, kSplits, kSampleStride, kSampleRank>
                   : topk_filtered_split_kernel<1, kTopK, kSplits, kSampleStride, kSampleRank>;
  const size_t smem_bytes = sizeof(RowSmem<kTopK, true>);
  if (!opt_in_dynamic_smem(reinterpret_cast<const void*>(kernel), smem_bytes)) {
    return false;
  }
  const int num_tasks = m * kSplits;
  kernel<<<num_tasks, THREADS, smem_bytes, stream>>>(
      logits, topk_indices, ke, split_buffer, spill, arrive, num_valid_rows,
      static_cast<uint32_t>(row_stride), static_cast<uint32_t>(out_stride),
      static_cast<uint32_t>(n), m);
  return cudaGetLastError() == cudaSuccess;
}

// True on devices that support cluster launches (sm_90 and later).
inline bool cluster_launch_supported() {
  static const bool supported = [] {
    if (get_sm_major_version() < 9) {
      return false;
    }
    int dev = 0;
    int ok = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&ok, cudaDevAttrClusterLaunch, dev);
    return ok != 0;
  }();
  return supported;
}

inline int cluster_wave_rows(const void* kernel, int splits, int threads, size_t smem_bytes) {
  if (!opt_in_dynamic_smem(kernel, smem_bytes)) {
    return 0;
  }
  cudaLaunchAttribute attribute[1]{};
  attribute[0].id = cudaLaunchAttributeClusterDimension;
  attribute[0].val.clusterDim.x = splits;
  attribute[0].val.clusterDim.y = 1;
  attribute[0].val.clusterDim.z = 1;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(splits, 1);
  config.blockDim = dim3(threads);
  config.dynamicSmemBytes = smem_bytes;
  config.attrs = attribute;
  config.numAttrs = 1;
  int clusters = 0;
  if (cudaOccupancyMaxActiveClusters(&clusters, kernel, &config) != cudaSuccess) {
    (void)cudaGetLastError();
    return 0;
  }
  return clusters;
}

inline size_t mc_workspace_bytes(int m, int num_segs, int n) {
  const size_t rows = static_cast<size_t>(m > 0 ? m : 0);
  const size_t segs = rows * num_segs;
  (void)segs;
  return rows * 4 * n * sizeof(int32_t) + rows * n * sizeof(uint64_t);
}

template <int kTopK, int kMinBlocks>
bool launch_mc(int* topk_indices, const float* logits, const int* ke, const int* num_valid_rows,
               int m, int n, int row_stride, int out_stride, int num_segs, void* counters,
               size_t counters_bytes, void* workspace, size_t workspace_bytes,
               cudaStream_t stream) {
  constexpr int kThreads = 1024;
  if (!cluster_launch_supported() || m <= 0 || num_segs <= 0 || num_segs > kMcMaxSegs ||
      counters_bytes < counters_bytes_for(m) ||
      workspace_bytes < mc_workspace_bytes(m, num_segs, n)) {
    return false;
  }
  const uint32_t vec = vector_width(row_stride);
  const auto kernel =
      (vec == 4)   ? topk_filtered_mc_kernel<4, kTopK, kThreads, MC_SEGMENT_CAPACITY,
                                             MC_STAGE_CAPACITY, MC_SURVIVOR_CAPACITY, kMinBlocks>
      : (vec == 2) ? topk_filtered_mc_kernel<2, kTopK, kThreads, MC_SEGMENT_CAPACITY,
                                             MC_STAGE_CAPACITY, MC_SURVIVOR_CAPACITY, kMinBlocks>
                   : topk_filtered_mc_kernel<1, kTopK, kThreads, MC_SEGMENT_CAPACITY,
                                             MC_STAGE_CAPACITY, MC_SURVIVOR_CAPACITY, kMinBlocks>;
  constexpr size_t smem_bytes = sizeof(RowSmem<kTopK, true, kThreads>) +
                                static_cast<size_t>(MC_STAGE_CAPACITY) * sizeof(uint64_t) +
                                static_cast<size_t>(MC_SURVIVOR_CAPACITY) * 4 * sizeof(int32_t);
  static int wave_rows[3] = {-1, -1, -1};
  int& rows = wave_rows[vec == 4 ? 2 : vec == 2 ? 1 : 0];
  if (rows < 0) {
    rows = cluster_wave_rows(reinterpret_cast<const void*>(kernel), kMcClusterSize, kThreads,
                             smem_bytes);
  }
  if (rows == 0) return false;
  int32_t* spill = static_cast<int32_t*>(workspace);
  uint64_t* cand = reinterpret_cast<uint64_t*>(spill + static_cast<size_t>(m) * 4 * n);
  int32_t* arrive = static_cast<int32_t*>(counters) + HEAD_COUNTER_INTS;
  int32_t* mc_ctr = arrive + 2 * m;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(num_segs, 1, m);
  config.blockDim = dim3(kThreads);
  config.dynamicSmemBytes = smem_bytes;
  config.stream = stream;
  return cudaLaunchKernelEx(&config, kernel, logits, topk_indices, ke, spill, cand, arrive, mc_ctr,
                            num_valid_rows, static_cast<uint32_t>(row_stride),
                            static_cast<uint32_t>(out_stride), static_cast<uint32_t>(n),
                            num_segs) == cudaSuccess;
}

constexpr int SPLIT_MIN_N = 128 * 1024 + 1;
constexpr int SPLIT_MAX_ROWS = 192;
// Returning 1 means that row splitting does not pay for this shape.
// Short batches split eight ways while the grid still fits roughly
// one SM-wide wave; larger batches use four.
inline int choose_splits(int m, int n) {
  if (m <= 0 || n < SPLIT_MIN_N || m > SPLIT_MAX_ROWS) {
    return 1;
  }
  return (m <= 16) ? 8 : 4;
}

// Route selected per shape band.
enum class DispatchKind : uint8_t {
  kRowLocal1024Exact,
  kKvSplitSampledExact,
  kPersistentExact,
  kPersistentSampledExact,
  kMultiCluster,
};

// mc route shape policy: CTA_BUDGET(_SHORT) caps the resident CTA
// pool, with short rows (up to SHORT_ROW_MAX_N) taking the quarter
// budget and the smaller segment floor; MIN_N_M32/M64 widen the band
// by row count; MAX_ROWS is the row ceiling.
constexpr int MC_CTA_BUDGET = 19;
constexpr int MC_CTA_BUDGET_SHORT = 8;
constexpr int MC_MIN_SEGMENT_SHORT = 4096;
constexpr int MC_MIN_SEGMENT = 16384;
constexpr int MC_SHORT_ROW_MAX_N = (262144);
constexpr int MC_MIN_N_M32 = (64 * 1024);
constexpr int MC_MIN_N_M64 = (128 * 1024);
constexpr int MC_MAX_ROWS = 64;
// Smallest n for which the mc route wins, by row-count band.
inline int mc_min_n(int m) {
  if (m <= 8) return 32 * 1024;
  if (m <= 16) return 64 * 1024;
  if (m <= 32) return MC_MIN_N_M32;
  return MC_MIN_N_M64;
}

// Segments per row, or 0 when the shape does not take the mc route.
// S is a multiple of the cluster size bounded by the CTA budget and
// the segment floor, capped so the grid fits one resident wave.
inline int mc_segs_per_row(int m, int n) {
  if (!cluster_launch_supported() || m <= 0 || m > MC_MAX_ROWS || n < mc_min_n(m)) {
    return 0;
  }
  const bool short_row = n <= MC_SHORT_ROW_MAX_N;
  const int min_seg = short_row ? MC_MIN_SEGMENT_SHORT : MC_MIN_SEGMENT;
  const int budget = short_row ? MC_CTA_BUDGET_SHORT : MC_CTA_BUDGET;
  const int by_budget = std::max(1, budget / m);
  const int by_len = std::max(1, n / (kMcClusterSize * min_seg));
  const int segs =
      std::min(std::min(by_budget, by_len), kMcMaxSegs / kMcClusterSize) * kMcClusterSize;
  const int wave = MIN_BLOCKS * get_sm_count();
  if (segs * m > wave) return std::max(1, wave / m);
  return segs;
}

// A single shape-only plan drives both launch selection and workspace
// sizing.
struct DispatchPlan {
  DispatchKind kind;
  int splits = 1;
  bool wide_sample = false;
};

// Each interval keeps the row-local grid within two resident waves
// while increasing scan length only after enough rows are available
// to fill the GPU.
inline int row_local_exact_max_n(int m) {
  const int sm_count = get_sm_count();
  if (m <= 0) {
    return 0;
  }
  if (m > 2 * sm_count) {
    return 64 * 1024;
  }
  if (m >= 64) {
    return 262144;
  }
  if (m >= 32) {
    return 196608;
  }
  return 131072;
}

// Chooses the route: the mc band first, then the cluster region, then
// row-local, kv-split, and the persistent sampled/exact tail.
inline DispatchPlan choose_topk2048_plan(int m, int n, int topk = 2048) {
  constexpr int kSampleMinN = 32768;
  constexpr int64_t kWideSampleMinWork = int64_t{32} * 1024 * 1024;
  const int mc_segs = mc_segs_per_row(m, n);
  if (mc_segs > 0) {
    return {DispatchKind::kMultiCluster, mc_segs};
  }
  const int row_local_limit = row_local_exact_max_n(m);
  if (row_local_limit != 0 && n <= row_local_limit) {
    return {DispatchKind::kRowLocal1024Exact};
  }
  const int splits = choose_splits(m, n);
  if (splits > 1) {
    return {DispatchKind::kKvSplitSampledExact, splits};
  }
  if (n < kSampleMinN) {
    return {DispatchKind::kPersistentExact};
  }
  const int64_t work = static_cast<int64_t>(m) * n;
  const bool second_row_wave = m > 2 * get_sm_count();
  const bool second_persistent_wave = m > grid_num_blocks();
  if (second_row_wave &&
      (work >= kWideSampleMinWork || (second_persistent_wave && n >= 48 * 1024))) {
    return {DispatchKind::kPersistentSampledExact, 1, true};
  }
  return {DispatchKind::kPersistentSampledExact};
}
}  // namespace filtered
// Executes the plan; one launch per DispatchKind.
template <int kTopK>
static bool dispatch_planned(int* topk_indices, const float* logits, const int* ke,
                             const int* num_valid_rows, int m, int n, int row_stride,
                             int out_stride, void* counters, size_t counters_bytes, void* workspace,
                             size_t workspace_bytes, cudaStream_t stream) {
  const filtered::DispatchPlan plan = filtered::choose_topk2048_plan(m, n, kTopK);
  switch (plan.kind) {
    case filtered::DispatchKind::kMultiCluster:
      if (filtered::launch_mc<kTopK, filtered::MIN_BLOCKS>(
              topk_indices, logits, ke, num_valid_rows, m, n, row_stride, out_stride, plan.splits,
              counters, counters_bytes, workspace, workspace_bytes, stream)) {
        return true;
      }
      break;
    case filtered::DispatchKind::kRowLocal1024Exact:
      if (filtered::launch_row_local_1024_exact<kTopK>(
              topk_indices, logits, ke, num_valid_rows, m, n, row_stride, out_stride, workspace,
              workspace_bytes, counters, counters_bytes, stream)) {
        return true;
      }
      break;
    case filtered::DispatchKind::kKvSplitSampledExact:
      if (counters_bytes >= filtered::counters_bytes_for(m) &&
          workspace_bytes >= filtered::split_workspace_bytes(m, plan.splits, n)) {
        constexpr int kRank = filtered::seg_sample_rank(kTopK);
        bool launched = false;
        switch (plan.splits) {
          case 8:
            launched = filtered::launch_kv_split<kTopK, 8, filtered::kSegSampleStride, kRank>(
                topk_indices, logits, ke, num_valid_rows, m, n, row_stride, out_stride, counters,
                counters_bytes, workspace, workspace_bytes, stream);
            break;
          case 4:
            launched = filtered::launch_kv_split<kTopK, 4, filtered::kSegSampleStride, kRank>(
                topk_indices, logits, ke, num_valid_rows, m, n, row_stride, out_stride, counters,
                counters_bytes, workspace, workspace_bytes, stream);
            break;
          case 2:
            launched = filtered::launch_kv_split<kTopK, 2, filtered::kSegSampleStride, kRank>(
                topk_indices, logits, ke, num_valid_rows, m, n, row_stride, out_stride, counters,
                counters_bytes, workspace, workspace_bytes, stream);
            break;
          default:
            break;
        }
        if (launched) {
          return true;
        }
      }
      break;
    case filtered::DispatchKind::kPersistentExact:
      return filtered::launch_one_cta_per_row<kTopK, false>(
          topk_indices, logits, ke, num_valid_rows, m, n, row_stride, out_stride, counters,
          counters_bytes, workspace, workspace_bytes, stream);
    case filtered::DispatchKind::kPersistentSampledExact:
      if (plan.wide_sample) {
        return filtered::launch_one_cta_per_row<kTopK, true, filtered::WIDE_SAMPLE_STRIDE,
                                                filtered::WIDE_SAMPLE_RANK>(
            topk_indices, logits, ke, num_valid_rows, m, n, row_stride, out_stride, counters,
            counters_bytes, workspace, workspace_bytes, stream);
      }
      return filtered::launch_one_cta_per_row<kTopK, true>(
          topk_indices, logits, ke, num_valid_rows, m, n, row_stride, out_stride, counters,
          counters_bytes, workspace, workspace_bytes, stream);
  }

  if (n < 32768) {
    return filtered::launch_one_cta_per_row<kTopK, false>(
        topk_indices, logits, ke, num_valid_rows, m, n, row_stride, out_stride, counters,
        counters_bytes, workspace, workspace_bytes, stream);
  }

  if (plan.wide_sample) {
    return filtered::launch_one_cta_per_row<kTopK, true, filtered::WIDE_SAMPLE_STRIDE,
                                            filtered::WIDE_SAMPLE_RANK>(
        topk_indices, logits, ke, num_valid_rows, m, n, row_stride, out_stride, counters,
        counters_bytes, workspace, workspace_bytes, stream);
  }

  return filtered::launch_one_cta_per_row<kTopK, true>(
      topk_indices, logits, ke, num_valid_rows, m, n, row_stride, out_stride, counters,
      counters_bytes, workspace, workspace_bytes, stream);
}
bool topk_filtered_async(int* topk_indices, const float* logits, const int* ke, int topk,
                         const int* num_valid_rows, int m, int n, int row_stride, int out_stride,
                         void* counters, size_t counters_bytes, void* workspace,
                         size_t workspace_bytes, cudaStream_t stream) {
  if (topk == 512) {
    return dispatch_planned<512>(topk_indices, logits, ke, num_valid_rows, m, n, row_stride,
                                 out_stride, counters, counters_bytes, workspace, workspace_bytes,
                                 stream);
  }

  if (topk == 2048) {
    return dispatch_planned<2048>(topk_indices, logits, ke, num_valid_rows, m, n, row_stride,
                                  out_stride, counters, counters_bytes, workspace, workspace_bytes,
                                  stream);
  }

  return false;
}
size_t topk_filtered_counters_bytes(int num_rows) { return filtered::counters_bytes_for(num_rows); }
size_t topk_filtered_workspace_bytes(int num_rows, int max_kv_len) {
  const filtered::DispatchPlan plan = filtered::choose_topk2048_plan(num_rows, max_kv_len);
  if (plan.kind == filtered::DispatchKind::kMultiCluster) {
    return std::max(filtered::mc_workspace_bytes(num_rows, plan.splits, max_kv_len),
                    filtered::one_cta_workspace_bytes(max_kv_len));
  }

  if (plan.kind == filtered::DispatchKind::kKvSplitSampledExact) {
    return filtered::split_workspace_bytes(num_rows, plan.splits, max_kv_len);
  }

  return filtered::one_cta_workspace_bytes(max_kv_len);
}
size_t topk_filtered_min_workspace_bytes(int max_kv_len) {
  return filtered::one_cta_workspace_bytes(max_kv_len);
}
size_t topk_filtered_peak_workspace_bytes(int max_kv_len) {
  size_t peak = filtered::one_cta_workspace_bytes(max_kv_len);
  const int max_split_rows = filtered::SPLIT_MAX_ROWS;
  for (int num_rows = 1; num_rows <= max_split_rows; ++num_rows) {
    peak = std::max(peak, topk_filtered_workspace_bytes(num_rows, max_kv_len));
  }

  return peak;
}
}  // namespace topk
}  // namespace hpc
