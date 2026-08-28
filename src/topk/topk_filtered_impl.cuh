// Copyright (C) 2026 Tencent.

#ifndef SRC_TOPK_TOPK_FILTERED_IMPL_CUH_
#define SRC_TOPK_TOPK_FILTERED_IMPL_CUH_

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_fp16.h>

#include <algorithm>

#include "src/topk/topk.h"
#include "src/topk/topk_filtered_boundary.cuh"
#include "src/utils/utils.h"

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

// Bit layouts of the three fp32-ordered fine refinement rounds.
//   round 0 → bits [21..31]  (top 11 bits)
//   round 1 → bits [10..20]  (middle 11 bits)
//   round 2 → bits [0..9]    (low 10 bits, back-fill round)
__device__ constexpr int kFineShifts[NUM_REFINE_ROUNDS] = {21, 10, 0};
__device__ constexpr uint32_t kFineMasks[NUM_REFINE_ROUNDS] = {0x7FFu, 0x7FFu, 0x3FFu};

constexpr int THREADS = 512;
constexpr int MIN_BLOCKS = 3;

// smem candidate capacity per ping-pong slot. The sampled path keeps twice as many
// because its predicted threshold intentionally admits more than top_k.
constexpr int COARSE_CANDIDATE_CAPACITY = 2048;
constexpr int SAMPLED_CANDIDATE_CAPACITY = 4096;

// Persistent grid: a fixed pool of BLOCKS_PER_SM CTAs per SM pulling rows from
// the work queue. Also sizes the gmem spill region ([num_blocks, 2, n]).
constexpr int BLOCKS_PER_SM = 3;

// Persistent grid size (num_sms * BLOCKS_PER_SM). Shared by the launch and the
// workspace sizing so the spill buffer always has exactly gridDim.x rows.
inline int grid_num_blocks() { return get_sm_count() * BLOCKS_PER_SM; }

// The low/high halves hold the descending coarse keys for x/y.
__device__ __forceinline__ uint32_t to_coarse_key2(float x, float y) {
  return static_cast<uint32_t>(to_coarse_key(x)) | (static_cast<uint32_t>(to_coarse_key(y)) << 16);
}

__device__ __forceinline__ uint32_t to_ordered(float x) {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? bits : ((bits ^ 0xFFFFFFFFu) & 0x7FFFFFFFu);
}

// Vectorized float4 load (cache at global level only).
__device__ __forceinline__ float4 ld_cg_float4(const float* p) {
  float4 v;
  asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];\n"
               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
               : "l"(p));
  return v;
}

// Shared-memory layout for one row. The sampled path keeps more candidates
// because its predicted threshold intentionally admits more than K.
template <int kBlockThreads>
struct PrefixStorage {
  int warp_sum[kBlockThreads / 32];
};

template <int kTopK, bool kSampled, int kBlockThreads = THREADS>
struct RowSmem {
  static constexpr int CANDIDATE_CAPACITY =
      kSampled ? SAMPLED_CANDIDATE_CAPACITY : COARSE_CANDIDATE_CAPACITY;

  int histogram[RADIX + 1];  // coarse / refine histogram (prefix-summed in place)
  int threshold_bin;         // bin holding the top-k boundary
  int num_selected;          // front fill cursor into indices
  int ties_remaining;        // back fill cursor for the last round's ties
  int num_cand[2];           // ping-pong candidate counts (smem portion)
  int num_spilled_cand[2];   // ping-pong candidate counts (gmem spill portion)
  int32_t indices[kTopK];    // accumulated result indices
  alignas(8) int32_t cand_idx[2][CANDIDATE_CAPACITY];  // ping-pong candidate index buffers
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

// Cluster8 reuses the candidate allocation as aligned (index, FP32 bits) pairs.
__device__ __forceinline__ uint64_t pack_cluster_candidate(int32_t index, uint32_t key) {
  return static_cast<uint64_t>(static_cast<uint32_t>(index)) | (static_cast<uint64_t>(key) << 32);
}

__device__ __forceinline__ int32_t cluster_candidate_index(uint64_t pair) {
  return static_cast<int32_t>(static_cast<uint32_t>(pair));
}

__device__ __forceinline__ uint32_t cluster_candidate_key(uint64_t pair) {
  return static_cast<uint32_t>(pair >> 32);
}

template <int kTopK, int kBlockThreads>
__device__ __forceinline__ uint64_t* cluster_candidate_pairs(
    RowSmem<kTopK, false, kBlockThreads>* s) {
  return reinterpret_cast<uint64_t*>(&s->cand_idx[0][0]);
}

__device__ __forceinline__ int warp_inclusive_sum(int value) {
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const int upper = __shfl_up_sync(0xFFFFFFFFu, value, offset);
    if ((threadIdx.x & 31) >= offset) {
      value += upper;
    }
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

// Inclusive prefix sum over the 2048-bin histogram (in place), then find the
// threshold bin: first bin where prefix[b-1] <= remaining < prefix[b]. Each
// thread owns a contiguous run of bins and locates the threshold after
// combining the per-thread totals through warp reductions.
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

template <uint32_t VEC, int kTopK, bool kSampled, int kSampleStride = SAMPLE_STRIDE,
          int kSampleRank = SAMPLE_RANK>
__global__ void __launch_bounds__(THREADS, MIN_BLOCKS)
    topk_filtered_kernel(const float* __restrict__ input, int32_t* __restrict__ output,
                         const int32_t* __restrict__ ke,
                         int32_t* __restrict__ extra_buffer,  // [num_blocks,2,N]
                         int32_t* __restrict__ work_counter, int32_t* __restrict__ exit_counter,
                         const int32_t* __restrict__ num_valid_rows, uint32_t row_stride,
                         uint32_t out_stride, uint32_t n_cols, int32_t row_capacity) {
  using Smem = RowSmem<kTopK, kSampled>;
  const int tidx = threadIdx.x;
  const int num_valid = *num_valid_rows;
  if (num_valid <= 0) {
    return;  // nothing claimed, so both counters keep the zeros they came in with
  }

  // Spill buffer is indexed by CTA (blockIdx.x), not by row: a persistent CTA
  // reuses its own [2, n] slice across every row it processes. A given block id
  // is never resident twice, so gridDim.x slices suffice with no cross-CTA
  // contention. The spill counters live in smem and are re-zeroed per row; stale
  // buffer bytes are harmless (reads are bounded by the counters, writes
  // overwrite from index 0).
  int32_t* buf0 = extra_buffer + static_cast<size_t>(blockIdx.x) * 2 * n_cols;
  int32_t* buf1 = buf0 + n_cols;
  int32_t* buf[2] = {buf0, buf1};

  extern __shared__ uint8_t smem_raw[];
  Smem* s = reinterpret_cast<Smem*>(smem_raw);
  __shared__ int shared_row;  // broadcast the row claimed by tidx==0 to all threads
  __shared__ float shared_coarse_boundary[2];

  // The first scheduling wave is statically mapped: CTA b starts from row b.
  // Only rows beyond that wave use the persistent queue. This preserves the
  // load-balancing behavior for large batches while avoiding queue traffic for
  // one-wave shapes, without a separate direct-map kernel or dispatch kind.
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
      // The CTA that sees exit_counter reach gridDim.x - 1 is the last one out
      // of the queue, so it hands both counters back zeroed for the next launch.
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

    // Rows no longer than top_k use their identity indices.
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

    // Derive a coarse threshold from the systematic sample.
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

        const int sample_offset = static_cast<int>((row * 17u + 13u) & (kSampleStride - 1));
        constexpr int kSampleThreadStride = THREADS * kSampleStride;
        for (int i = sample_offset + tidx * kSampleStride; i < length;
             i += 2 * kSampleThreadStride) {
          float x = score[i];
          int j = i + kSampleThreadStride;
          if (j < length) {
            float y = score[j];
            uint32_t keys = to_coarse_key2(x, y);
            atomicAdd(&s->histogram[static_cast<uint16_t>(keys)], 1);
            atomicAdd(&s->histogram[static_cast<uint16_t>(keys >> 16)], 1);
          } else {
            atomicAdd(&s->histogram[to_coarse_key(x)], 1);
          }
        }
        __syncthreads();

        if constexpr (kUseSecondaryThreshold) {
          prefix_sum_find_threshold<kTopK, true, kSecondarySampleRank>(s, tidx, kSampleRank, 0,
                                                                       &s->num_selected, 0);
        } else {
          prefix_sum_find_threshold(s, tidx, kSampleRank, /*next_idx=*/0, &s->num_selected,
                                    /*reset_value=*/0);
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

        // Build the first exact-fp32 histogram over every item admitted by the
        // sampled fp16 threshold. Include the whole boundary key bin.
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
            buf[0][gpos] = i;
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
              next_a = ld_cg_float4(reinterpret_cast<const float*>(v4 + 2 * g));
              next_b = ld_cg_float4(reinterpret_cast<const float*>(v4 + 2 * g + 1));
            }
            while (g < n8) {
              const float4 current_a = next_a;
              const float4 current_b = next_b;
              const int base = g << 3;
              g += THREADS;
              if (g < n8) {
                next_a = ld_cg_float4(reinterpret_cast<const float*>(v4 + 2 * g));
                next_b = ld_cg_float4(reinterpret_cast<const float*>(v4 + 2 * g + 1));
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
              float4 a = ld_cg_float4(reinterpret_cast<const float*>(v4 + 2 * g));
              float4 b = ld_cg_float4(reinterpret_cast<const float*>(v4 + 2 * g + 1));
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
              for (int i = tidx; i < kTopK; i += THREADS) {
                s->indices[i] = s->cand_idx[0][i];
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
                  next_a = ld_cg_float4(reinterpret_cast<const float*>(v4 + 2 * g));
                  next_b = ld_cg_float4(reinterpret_cast<const float*>(v4 + 2 * g + 1));
                }
                while (g < n8) {
                  const float4 current_a = next_a;
                  const float4 current_b = next_b;
                  const int base = g << 3;
                  g += THREADS;
                  if (g < n8) {
                    next_a = ld_cg_float4(reinterpret_cast<const float*>(v4 + 2 * g));
                    next_b = ld_cg_float4(reinterpret_cast<const float*>(v4 + 2 * g + 1));
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
                  float4 a = ld_cg_float4(reinterpret_cast<const float*>(v4 + 2 * g));
                  float4 b = ld_cg_float4(reinterpret_cast<const float*>(v4 + 2 * g + 1));
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

    // Build the exact coarse histogram when sampling is disabled or undershoots.
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
        // 8-wide per thread (two float4 loads in flight) to match
        // num_copy_bits=256 and hide global-load latency.
        const int n8 = length >> 3;
        const float4* v4 = reinterpret_cast<const float4*>(score);
        for (int g = tidx; g < n8; g += THREADS) {
          float4 a = ld_cg_float4(reinterpret_cast<const float*>(v4 + 2 * g));
          float4 b = ld_cg_float4(reinterpret_cast<const float*>(v4 + 2 * g + 1));
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

      prefix_sum_find_threshold(s, tidx, topk_remaining, /*next_idx=*/0, &s->num_selected,
                                /*reset_value=*/0);

      int threshold_bin = s->threshold_bin;
      if (threshold_bin > 0) {
        topk_remaining -= s->histogram[threshold_bin - 1];
      }
      // All threads must finish reading the prefix value before histogram reset.
      __syncthreads();

      if (topk_remaining == 0) {
        // Boundary bin contributes nothing; just collect bin < threshold_bin.
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
            float4 v = ld_cg_float4(reinterpret_cast<const float*>(v4 + g));
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
        // Reset histogram for the first refine round.
        for (int i = tidx; i < RADIX + 1; i += THREADS) {
          s->histogram[i] = 0;
        }
        __syncthreads();

        // Per-element exact coarse split: bin<thr -> select; bin==thr ->
        // candidate + first refine histogram; bin>thr -> drop.
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
              buf[0][gpos] = i;
            }
            int sub = (ordered >> kFineShifts[0]) & kFineMasks[0];
            atomicAdd(&s->histogram[sub], 1);
          }
        };

        if constexpr (VEC == 4) {
          const int n8 = length >> 3;
          const float4* v4 = reinterpret_cast<const float4*>(score);
          for (int g = tidx; g < n8; g += THREADS) {
            float4 a = ld_cg_float4(reinterpret_cast<const float*>(v4 + 2 * g));
            float4 b = ld_cg_float4(reinterpret_cast<const float*>(v4 + 2 * g + 1));
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

    // Refine the exact FP32 boundary.
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

          // prefix sum + threshold; seeds s->ties_remaining = remaining - prefix[thr-1].
          prefix_sum_find_threshold(s, tidx, topk_remaining, /*next_idx=*/r ^ 1, &s->ties_remaining,
                                    topk_remaining);
          // last_remain reset_value must be remaining-previous, recomputed below.
          __syncthreads();

          int num_input = min(s->num_cand[r], Smem::CANDIDATE_CAPACITY);
          int g_num = s->num_spilled_cand[r];
          int threshold = s->threshold_bin;
          if (threshold > 0) {
            topk_remaining -= s->histogram[threshold - 1];
          }
          // Fix last_remain to the post-subtraction remaining (the # of ties to
          // back-fill in the final round). prefix_sum seeded it with the pre-round
          // remaining; recompute now that we know prefix[threshold-1].
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

            // smem candidates
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
                    buf[r ^ 1][gpos] = idx;
                  }
                  int sub = (bits >> next_shift) & next_mask;
                  atomicAdd(&s->histogram[sub], 1);
                }
              }
            }
            // gmem-spilled candidates
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
                    buf[r ^ 1][gpos] = idx;
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
  }  // for row
}

// KV-split completion keeps these stages out of line so their register demand
// does not affect the distributed scan.

// Exact coarse histogram over the whole row, followed by the coarse split that
// seeds the first refine round. Returns the count still owed by the boundary
// bin: 0 means the bin contributes nothing and no refinement is needed, > 0 is
// the `topk_remaining` to hand to refine_rounds.
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
    // 8-wide per thread (two float4 loads in flight) to match num_copy_bits=256
    // and hide global-load latency.
    const int n8 = length >> 3;
    const float4* v4 = reinterpret_cast<const float4*>(score);
    for (int g = tidx; g < n8; g += kBlockThreads) {
      float4 a = ld_cg_float4(reinterpret_cast<const float*>(v4 + 2 * g));
      float4 b = ld_cg_float4(reinterpret_cast<const float*>(v4 + 2 * g + 1));
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

  prefix_sum_find_threshold(s, tidx, remaining, /*next_idx=*/0, &s->num_selected,
                            /*reset_value=*/0);

  int threshold_bin = s->threshold_bin;
  if (threshold_bin > 0) {
    remaining -= s->histogram[threshold_bin - 1];
  }
  // All threads must finish reading the prefix value before histogram reset.
  __syncthreads();

  if (remaining == 0) {
    // Boundary bin contributes nothing; just collect bin < threshold_bin.
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
        float4 v = ld_cg_float4(reinterpret_cast<const float*>(v4 + g));
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

  // Reset histogram for the first refine round.
  for (int i = tidx; i < RADIX + 1; i += kBlockThreads) {
    s->histogram[i] = 0;
  }
  __syncthreads();

  // Per-element exact coarse split: bin<thr -> select; bin==thr -> candidate +
  // first refine histogram; bin>thr -> drop.
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
        buf0[gpos] = i;
      }
      int sub = (ordered >> kFineShifts[0]) & kFineMasks[0];
      atomicAdd(&s->histogram[sub], 1);
    }
  };

  if constexpr (VEC == 4) {
    const int n8 = length >> 3;
    const float4* v4 = reinterpret_cast<const float4*>(score);
    for (int g = tidx; g < n8; g += kBlockThreads) {
      float4 a = ld_cg_float4(reinterpret_cast<const float*>(v4 + 2 * g));
      float4 b = ld_cg_float4(reinterpret_cast<const float*>(v4 + 2 * g + 1));
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

// Exact fp32 refinement rounds over the round-0 candidate set already staged in
// s->cand_idx[0] / buf0 with s->histogram holding the round-0 fine
// histogram. Fills s->indices with exactly kTopK entries.
template <int kTopK, bool kSampled, int kBlockThreads = THREADS>
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

    // prefix sum + threshold; seeds s->ties_remaining = remaining - prefix[thr-1].
    prefix_sum_find_threshold(s, tidx, topk_remaining, /*next_idx=*/r ^ 1, &s->ties_remaining,
                              topk_remaining);
    // last_remain reset_value must be remaining-previous, recomputed below.
    __syncthreads();

    int num_input = min(s->num_cand[r], Smem::CANDIDATE_CAPACITY);
    int g_num = s->num_spilled_cand[r];
    int threshold = s->threshold_bin;
    if (threshold > 0) {
      topk_remaining -= s->histogram[threshold - 1];
    }
    // Fix last_remain to the post-subtraction remaining (the # of ties to
    // back-fill in the final round). prefix_sum seeded it with the pre-round
    // remaining; recompute now that we know prefix[threshold-1].
    if (tidx == 0) {
      s->ties_remaining = topk_remaining;
    }
    __syncthreads();

    int cur_shift = kFineShifts[round];
    uint32_t cur_mask = kFineMasks[round];
    bool is_last = (round == NUM_REFINE_ROUNDS - 1);
    int next_shift = is_last ? 0 : kFineShifts[round + 1];
    uint32_t next_mask = is_last ? 0 : kFineMasks[round + 1];
    // Select the ping-pong spill slices with ternaries rather than a local
    // pointer array: an array captured by the lambda below would be forced to
    // the stack and spill (sm90 builds gate on -warn-spills).
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

      // Split one candidate: select, promote to the next round, or drop.
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
              next_buf[gpos] = idx;
            }
            int sub = (bits >> next_shift) & next_mask;
            atomicAdd(&s->histogram[sub], 1);
          }
        }
      };

      // smem candidates
      for (int i = tidx; i < num_input; i += kBlockThreads) {
        refine_one(s->cand_idx[r][i]);
      }
      // gmem-spilled candidates
      for (int i = tidx; i < g_num; i += kBlockThreads) {
        refine_one(cur_buf[i]);
      }
      __syncthreads();
    }
  }
}

// Refine a resident exact boundary without rereading scores or ping-ponging
// candidate indices. Sampled and spilled boundaries use refine_rounds.
template <int kTopK, int kBlockThreads>
__device__ __noinline__ void refine_resident_keys(RowSmem<kTopK, false, kBlockThreads>* s,
                                                  int topk_remaining, int tidx) {
  const int num_input = s->num_cand[0];
  uint32_t active_mask = 0;
  uint32_t active_prefix = 0;

#pragma unroll
  for (int round = 0; round < NUM_REFINE_ROUNDS; ++round) {
    prefix_sum_find_threshold<kTopK, false, -1, kBlockThreads,
                              /*kResetState=*/false>(s, tidx, topk_remaining, /*next_idx=*/0,
                                                     &s->ties_remaining, topk_remaining);
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

// One primary thread owns one Cluster8 boundary candidate across 11-11-10.
template <int kTopK, int kBlockThreads>
__device__ __noinline__ void refine_cluster_register_candidate(
    RowSmem<kTopK, false, kBlockThreads>* s, uint64_t raw_pair, bool active, int topk_remaining,
    int tidx, int32_t* direct_output) {
  uint32_t ordered = 0;
  int idx = -1;
  int write_pos = kTopK;
  if (active) {
    idx = cluster_candidate_index(raw_pair);
    ordered = to_ordered(__uint_as_float(cluster_candidate_key(raw_pair)));
  }

  for (int bin = tidx; bin < RADIX + 1; bin += kBlockThreads) {
    s->histogram[bin] = 0;
  }
  __syncthreads();
  if (active) {
    const int sub = (ordered >> kFineShifts[0]) & kFineMasks[0];
    atomicAdd(&s->histogram[sub], 1);
  }
  __syncthreads();

#pragma unroll
  for (int round = 0; round < NUM_REFINE_ROUNDS; ++round) {
    prefix_sum_find_threshold<kTopK, false, -1, kBlockThreads,
                              /*kResetState=*/false>(s, tidx, topk_remaining, /*next_idx=*/0,
                                                     &s->ties_remaining, topk_remaining);
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

    if (active) {
      const int bin = (ordered >> cur_shift) & cur_mask;
      if (bin < threshold) {
        write_pos = atomicAdd(&s->num_selected, 1);
        active = false;
      } else if (bin > threshold) {
        active = false;
      } else if (topk_remaining != 0) {
        if (is_last) {
          const int cur = atomicAdd(&s->ties_remaining, -1);
          if (cur > 0) {
            write_pos = kTopK - cur;
          }
          active = false;
        } else {
          const int next_shift = kFineShifts[round + 1];
          const uint32_t next_mask = kFineMasks[round + 1];
          const int next_bin = (ordered >> next_shift) & next_mask;
          atomicAdd(&s->histogram[next_bin], 1);
        }
      } else {
        active = false;
      }
    }
    __syncthreads();
    if (topk_remaining == 0 || is_last) {
      break;
    }
  }
  if (write_pos < kTopK) {
    direct_output[write_pos] = idx;
  }
}

template <int kTopK, int kBlockThreads>
__device__ __noinline__ void refine_row_local_candidates(RowLocalSmem<kTopK, kBlockThreads>* s,
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
__device__ __noinline__ void refine_row_local_rescan(RowLocalSmem<kTopK, kBlockThreads>* s,
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
__device__ __noinline__ void select_row_local_warp_rank(RowLocalSmem<kTopK, kBlockThreads>* s,
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
      local[item] = ld_cg_float4(input + static_cast<size_t>(row_i) * row_stride + (vi << 2));
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
  int32_t* spill0 = spill_buffer + static_cast<size_t>(blockIdx.x) * 2 * n_cols;
  int32_t* spill1 = spill0 + n_cols;
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
    int row_i = static_cast<int>(blockIdx.x);
    if constexpr (kBounded) {
      row_i = first_wave ? row_i : s->threshold_bin;
      first_wave = false;
    }
    const float* score = input + static_cast<size_t>(row_i) * row_stride;
    int32_t* dst = output + static_cast<size_t>(row_i) * out_stride;
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
        dst[i] = (i < length) ? i : -1;
      }
      if constexpr (!kBounded) {
        break;
      }
      have_work = claim_streaming_row(tidx, false, row_capacity, num_valid, work_counter,
                                      exit_counter, &s->threshold_bin);
      continue;
    }

    const int topk_remaining =
        run_exact_coarse_stage<VEC, kTopK, false, SHORT_THREADS>(s, score, spill0, length, tidx);
    if (topk_remaining != 0) {
      if (s->num_spilled_cand[0] == 0) {
        refine_resident_keys(s, topk_remaining, tidx);
      } else {
        refine_rounds<kTopK, false, SHORT_THREADS>(s, score, spill0, spill1, topk_remaining, tidx);
      }
    }
    for (int i = tidx; i < kTopK; i += SHORT_THREADS) {
      dst[i] = s->indices[i];
    }
    if constexpr (!kBounded) {
      break;
    }
    have_work = claim_streaming_row(tidx, false, row_capacity, num_valid, work_counter,
                                    exit_counter, &s->threshold_bin);
  }
}

// KV-split kernel: kSplits CTAs cooperate on one row.
//
// Every segment derives the same row-dependent sampled threshold, scans a
// disjoint interval, and publishes its candidates and first fine histogram.
// The last-arriving CTA reduces the segment state and completes exact
// refinement. An undershooting sample falls back to the full-row coarse stage.

// Scratch layout, in int32 units, across the two buffers the caller supplies.
//
// counters -- must arrive zeroed, handed back zeroed:
//   work_counter [1]                  one-CTA path queue cursor
//   exit_counter [1]                  one-CTA path exit count
//   arrive       [m]                  direct KV-split per-row arrival counter
//
// workspace -- pure scratch, no initialization required:
//   spill        [max(num_blocks,m),2,n] one slice per persistent CTA or direct row
//   --- KV split only, from here on ---
//   counts       [m, kSplits]         per-segment candidate count
//   hists        [m, kSplits, RADIX]  per-segment round-0 fine histogram
//   cands        [m, n]               candidate global ids, at their KV offset
//
// The split between the two is by lifetime, not by path. Only the counters carry
// state across a launch boundary, and the kernels leave every one of them at zero
// on the way out, which is what lets that buffer be reused with no host-side
// memset. counts/hists/cands are fully written before they are read for a given
// row, and the spill region's reads are bounded by smem counters that are
// re-zeroed per row, so the workspace's incoming contents never matter.
//
// A counters buffer sized for m rows fits either path; the one-CTA path simply
// leaves arrive untouched.
constexpr size_t HEAD_COUNTER_INTS = 2;  // work_counter, exit_counter

inline size_t counters_bytes_for(int m) {
  return (HEAD_COUNTER_INTS + static_cast<size_t>(m > 0 ? m : 0)) * sizeof(int32_t);
}

inline size_t spill_ints(int n) { return static_cast<size_t>(grid_num_blocks()) * 2 * n; }

inline size_t one_cta_workspace_bytes(int n) { return spill_ints(n) * sizeof(int32_t); }

inline size_t split_spill_ints(int m, int n) {
  return static_cast<size_t>(std::max(grid_num_blocks(), m)) * 2 * n;
}

inline size_t split_workspace_bytes(int m, int splits, int n) {
  const size_t split_ints = static_cast<size_t>(m) * splits +          // counts
                            static_cast<size_t>(m) * splits * RADIX +  // hists
                            static_cast<size_t>(m) * n;                // cands
  return (split_spill_ints(m, n) + split_ints) * sizeof(int32_t);
}

template <uint32_t VEC, int kTopK, int kSplits, int kSampleStride, int kSampleRank>
__global__ void __launch_bounds__(THREADS, MIN_BLOCKS)
    topk_filtered_split_kernel(const float* __restrict__ input, int32_t* __restrict__ output,
                               const int32_t* __restrict__ ke, int32_t* __restrict__ split_buffer,
                               int32_t* __restrict__ spill_buffer,
                               int32_t* __restrict__ arrive_counters,
                               const int32_t* __restrict__ num_valid_rows, uint32_t row_stride,
                               uint32_t out_stride, uint32_t n_cols, int32_t m_rows) {
  // The split path always uses the sampled layout: its predicted threshold
  // deliberately admits more than kTopK.
  using Smem = RowSmem<kTopK, true>;
  const int tidx = threadIdx.x;
  const int num_valid = *num_valid_rows;
  if (num_valid <= 0) {
    return;
  }
  const int64_t num_tasks = static_cast<int64_t>(num_valid) * kSplits;

  // Region bases, derived once. Keeping these as three scalars rather than a
  // scratch-layout object keeps the split kernel off the sm90 spill gate.
  int32_t* const ws_counts = split_buffer;
  int32_t* const ws_hists = ws_counts + static_cast<size_t>(m_rows) * kSplits;
  int32_t* const ws_cands = ws_hists + static_cast<size_t>(m_rows) * kSplits * RADIX;

  extern __shared__ uint8_t smem_raw[];
  Smem* s = reinterpret_cast<Smem*>(smem_raw);
  __shared__ int shared_arrived;
  __shared__ int shared_total;
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
    int32_t* seg_counts = ws_counts + static_cast<size_t>(row_i) * kSplits;
    int32_t* seg_hists = ws_hists + static_cast<size_t>(row_i) * kSplits * RADIX;
    int32_t* cand_ids = ws_cands + static_cast<size_t>(row_i) * n_cols;
    // Only the last-arriving CTA continues into spill/refinement, so the row is
    // the stable owner. blockIdx.x can exceed the persistent grid when direct
    // mapping launches m*kSplits tasks and therefore cannot index this pool.
    int32_t* spill0 = spill_buffer + static_cast<size_t>(row_i) * 2 * n_cols;
    int32_t* spill1 = spill0 + n_cols;

    // Only segment 0 writes identity indices for short rows.
    if (length <= kTopK) {
      if (seg == 0) {
        for (int i = tidx; i < kTopK; i += THREADS) {
          dst[i] = (i < length) ? i : -1;
        }
      }
      return;
    }

    // Segment bounds: split the *valid* length so every segment carries work.
    // seg_lo + count <= seg_hi <= length <= n, so segment writes stay in range.
    const int seg_len = (length + kSplits - 1) / kSplits;
    const int seg_lo = min(seg * seg_len, length);
    const int seg_hi = min(seg_lo + seg_len, length);

    // Derive the common sampled threshold.
    const bool sample_usable = length >= (kSampleRank + 1) * kSampleStride;
    int sampled_threshold = -1;
    if (sample_usable) {
      for (int i = tidx; i < RADIX + 1; i += THREADS) {
        s->histogram[i] = 0;
      }
      __syncthreads();

      const int sample_offset = static_cast<int>((row * 17u + 13u) & (kSampleStride - 1));
      constexpr int kSampleThreadStride = THREADS * kSampleStride;
      for (int i = sample_offset + tidx * kSampleStride; i < length; i += 2 * kSampleThreadStride) {
        float x = score[i];
        int j = i + kSampleThreadStride;
        if (j < length) {
          float y = score[j];
          uint32_t keys = to_coarse_key2(x, y);
          atomicAdd(&s->histogram[static_cast<uint16_t>(keys)], 1);
          atomicAdd(&s->histogram[static_cast<uint16_t>(keys >> 16)], 1);
        } else {
          atomicAdd(&s->histogram[to_coarse_key(x)], 1);
        }
      }
      __syncthreads();

      prefix_sum_find_threshold(s, tidx, kSampleRank, /*next_idx=*/0, &s->num_selected,
                                /*reset_value=*/0);
      sampled_threshold = s->threshold_bin;
      if (tidx == 0) {
        shared_coarse_boundary = coarse_boundary(sampled_threshold).value;
      }
      __syncthreads();
    }

    // Scan this segment into disjoint candidate and histogram regions.
    // Candidate ids land at their own KV offset, so segments never collide.
    // s->histogram accumulates this segment's round-0 fine histogram.
    if (sample_usable) {
      const float sampled_boundary = shared_coarse_boundary;
      const uint32_t sampled_boundary_bits = __float_as_uint(sampled_boundary);
      for (int i = tidx; i < RADIX + 1; i += THREADS) {
        s->histogram[i] = 0;
      }
      if (tidx == 0) {
        s->num_cand[0] = 0;
      }
      __syncthreads();

      auto seg_split = [&](float raw, int i) {
        if (fp32_boundary_accepts(raw, sampled_boundary, sampled_boundary_bits)) {
          int pos = seg_lo + atomicAdd(&s->num_cand[0], 1);
          cand_ids[pos] = i;
          int sub = (to_ordered(raw) >> kFineShifts[0]) & kFineMasks[0];
          atomicAdd(&s->histogram[sub], 1);
        }
      };

      if constexpr (VEC == 4) {
        // Vectorize only the 8-aligned interior of this segment.
        const int v_lo = min((seg_lo + 7) & ~7, seg_hi);
        const int v_hi = max(seg_hi & ~7, v_lo);
        for (int i = seg_lo + tidx; i < v_lo; i += THREADS) {
          float raw = score[i];
          seg_split(raw, i);
        }
        const int n8 = (v_hi - v_lo) >> 3;
        for (int g = tidx; g < n8; g += THREADS) {
          const float* p = score + v_lo + (g << 3);
          float4 a = ld_cg_float4(p);
          float4 b = ld_cg_float4(p + 4);
          int base = v_lo + (g << 3);
          seg_split(a.x, base);
          seg_split(a.y, base + 1);
          seg_split(a.z, base + 2);
          seg_split(a.w, base + 3);
          seg_split(b.x, base + 4);
          seg_split(b.y, base + 5);
          seg_split(b.z, base + 6);
          seg_split(b.w, base + 7);
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

      // Publish this segment and elect the finishing CTA.
      for (int i = tidx; i < RADIX; i += THREADS) {
        seg_hists[static_cast<size_t>(seg) * RADIX + i] = s->histogram[i];
      }
      if (tidx == 0) {
        seg_counts[seg] = s->num_cand[0];
      }
      // Release: this segment's data must be visible before the arrive bump.
      __threadfence();
      __syncthreads();
      if (tidx == 0) {
        shared_arrived = atomicAdd(arrive, 1);
      }
      __syncthreads();
      if (shared_arrived != kSplits - 1) {
        return;  // not the finisher
      }
      // Acquire: pair with the peers' releases above.
      __threadfence();
      if (tidx == 0) {
        // All kSplits CTAs have arrived and only this one goes on, so the slot is
        // finished with. Hand the counter back zeroed for the next launch.
        *arrive = 0;
      }
    } else if (seg != 0) {
      // Row too short to sample: segment 0 handles it alone via the plain
      // full-row path; the other segments retire immediately.
      return;
    }

    // The last-arriving CTA completes this row.
    int topk_remaining = kTopK;
    bool need_refine = false;
    bool ready = false;

    if (sample_usable) {
      if (tidx == 0) {
        int sum = 0;
        for (int i = 0; i < kSplits; ++i) {
          sum += seg_counts[i];
        }
        shared_total = sum;
      }
      __syncthreads();
    }

    if (sample_usable && shared_total > kTopK) {
      // Reduce the per-segment round-0 histograms into s->histogram, then gather
      // the segments' candidate ids into the smem/spill candidate buffers.
      for (int i = tidx; i < RADIX + 1; i += THREADS) {
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
        s->num_cand[0] = 0;
        s->num_spilled_cand[0] = 0;
      }
      __syncthreads();

      for (int sp = 0; sp < kSplits; ++sp) {
        const int cnt = seg_counts[sp];
        const int base = min(sp * seg_len, length);
        for (int i = tidx; i < cnt; i += THREADS) {
          int idx = cand_ids[base + i];
          int pos = atomicAdd(&s->num_cand[0], 1);
          if (pos < Smem::CANDIDATE_CAPACITY) {
            s->cand_idx[0][pos] = idx;
          } else {
            int gpos = atomicAdd(&s->num_spilled_cand[0], 1);
            spill0[gpos] = idx;
          }
        }
        __syncthreads();
      }
      need_refine = true;
      ready = true;
    }

    // An undershooting sample restarts from the exact coarse histogram.
    if (!ready) {
      topk_remaining = run_exact_coarse_stage<VEC, kTopK, true>(s, score, spill0, length, tidx);
      need_refine = topk_remaining != 0;
    }

    if (need_refine) {
      refine_rounds<kTopK, true>(s, score, spill0, spill1, topk_remaining, tidx);
    }

    for (int i = tidx; i < kTopK; i += THREADS) {
      dst[i] = s->indices[i];
    }
  }  // direct task
}

// One-shot DSM all-reduce over a power-of-two histogram. Each cluster rank owns
// one contiguous eighth of the bins; groups of eight lanes read the same owned
// bin from all peers, reduce it, and scatter the global sum back to every peer.
template <int kTopK, int kSplits, int kBlockThreads>
__device__ __forceinline__ void cluster_allreduce_histogram(
    RowSmem<kTopK, false, kBlockThreads>* s, cooperative_groups::cluster_group cluster, int rank,
    int tidx) {
  static_assert(kSplits == 8);
  static_assert(kBlockThreads == 1024);
  static_assert(RADIX % kSplits == 0);
  static_assert(RADIX % kBlockThreads == 0);
  constexpr int kBinsPerRank = RADIX / kSplits;
  constexpr int kItems = RADIX / kBlockThreads;

  cluster.sync();
#pragma unroll
  for (int item = 0; item < kItems; ++item) {
    const int linear = tidx + item * kBlockThreads;
    const int bin = rank * kBinsPerRank + linear / kSplits;
    const int peer = linear & (kSplits - 1);
    int* addr = cluster.map_shared_rank(&s->histogram[bin], peer);
    int value = *addr;
#pragma unroll
    for (int offset = 1; offset < kSplits; offset <<= 1) {
      value += __shfl_xor_sync(0xFFFFFFFFu, value, offset);
    }
    *addr = value;
  }
  cluster.sync();
}

// The primary CTA refines boundaries too large for the resident Cluster8 path.
// Peers remain at the cluster rendezvous until the exact continuation finishes.
template <int kTopK, int kBlockThreads>
__device__ __noinline__ void refine_large_boundary_single_cta(
    RowSmem<kTopK, false, kBlockThreads>* s, const float* score, int32_t* dst, int length,
    PackedCoarseBoundary select_boundary, PackedCoarseBoundary candidate_boundary,
    int topk_remaining) {
  const int tidx = threadIdx.x;
  uint32_t active_mask = 0;
  uint32_t active_prefix = 0;

  for (int bin = tidx; bin < RADIX + 1; bin += kBlockThreads) {
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
      if (pos < kTopK) {
        dst[pos] = i;
      }
    } else if (coarse_class == CoarseClass::kCandidate) {
      const uint32_t ordered = to_ordered(raw);
      const int sub = (ordered >> kFineShifts[0]) & kFineMasks[0];
      atomicAdd(&s->histogram[sub], 1);
    }
  }
  __syncthreads();

#pragma unroll 1
  for (int round = 0; round < NUM_REFINE_ROUNDS; ++round) {
    prefix_sum_find_threshold<kTopK, false, -1, kBlockThreads,
                              /*kResetState=*/false>(s, tidx, topk_remaining, /*next_idx=*/0,
                                                     &s->ties_remaining, topk_remaining);

    const int threshold = s->threshold_bin;
    if (threshold > 0) {
      topk_remaining -= s->histogram[threshold - 1];
    }
    const int shift = kFineShifts[round];
    const uint32_t mask = kFineMasks[round];
    const bool is_last = round == NUM_REFINE_ROUNDS - 1;

    if (topk_remaining != 0 && !is_last) {
      for (int bin = tidx; bin < RADIX + 1; bin += kBlockThreads) {
        s->histogram[bin] = 0;
      }
    }
    __syncthreads();

    // Only values in the coarse boundary and the active FP32 prefix remain.
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
        if (pos < kTopK) {
          dst[pos] = i;
        }
      } else if (bin == threshold && topk_remaining != 0) {
        if (is_last) {
          const int pos = atomicAdd(&s->num_selected, 1);
          if (pos < kTopK) {
            dst[pos] = i;
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

// Exact continuation for unaligned Cluster8 rows.
template <int kTopK, int kSplits, int kBlockThreads>
__device__ __noinline__ void run_cluster_unaligned_overflow(RowSmem<kTopK, false, kBlockThreads>* s,
                                                            const float* score, int32_t* dst,
                                                            int32_t* spill0, int32_t* spill1,
                                                            int length, int seg_lo, int seg_len,
                                                            int topk_remaining) {
  namespace cg = cooperative_groups;
  using Smem = RowSmem<kTopK, false, kBlockThreads>;
  cg::cluster_group cluster = cg::this_cluster();
  const int tidx = threadIdx.x;
  const int rank = static_cast<int>(cluster.block_rank());

  for (int bin = tidx; bin < RADIX + 1; bin += kBlockThreads) {
    s->histogram[bin] = 0;
  }
  __syncthreads();
  const int local_resident = min(s->num_cand[0], Smem::CANDIDATE_CAPACITY);
  for (int pos = tidx; pos < local_resident; pos += kBlockThreads) {
    const uint32_t ordered = static_cast<uint32_t>(s->cand_idx[1][pos]);
    const int sub = (ordered >> kFineShifts[0]) & kFineMasks[0];
    atomicAdd(&s->histogram[sub], 1);
  }
  for (int pos = tidx; pos < s->num_spilled_cand[0]; pos += kBlockThreads) {
    const int idx = spill1[seg_lo + pos];
    const uint32_t ordered = to_ordered(score[idx]);
    const int sub = (ordered >> kFineShifts[0]) & kFineMasks[0];
    atomicAdd(&s->histogram[sub], 1);
  }
  __syncthreads();
  cluster_allreduce_histogram<kTopK, kSplits, kBlockThreads>(s, cluster, rank, tidx);

  if (rank == 0) {
    if (tidx == 0) {
      s->num_cand[1] = s->num_spilled_cand[0];
      s->num_selected = s->threshold_bin;
      s->num_cand[0] = min(s->ties_remaining, Smem::CANDIDATE_CAPACITY);
      s->num_spilled_cand[0] = 0;
    }
    __syncthreads();

#pragma unroll
    for (int sp = 1; sp < kSplits; ++sp) {
      Smem* peer = cluster.map_shared_rank(s, sp);
      const int selected = peer->num_selected;
      for (int i = tidx; i < selected; i += kBlockThreads) {
        const int pos = atomicAdd(&s->num_selected, 1);
        s->indices[pos] = peer->indices[i];
      }
      const int candidates = min(peer->num_cand[0], Smem::CANDIDATE_CAPACITY);
      for (int i = tidx; i < candidates; i += kBlockThreads) {
        const int pos = atomicAdd(&s->num_cand[0], 1);
        const int idx = peer->cand_idx[0][i];
        if (pos < Smem::CANDIDATE_CAPACITY) {
          s->cand_idx[0][pos] = idx;
        } else {
          const int gpos = atomicAdd(&s->num_spilled_cand[0], 1);
          spill0[gpos] = idx;
        }
      }
      __syncthreads();
    }

#pragma unroll
    for (int sp = 0; sp < kSplits; ++sp) {
      Smem* peer = cluster.map_shared_rank(s, sp);
      const int overflow = (sp == 0) ? s->num_cand[1] : peer->num_spilled_cand[0];
      const int source_lo = min(sp * seg_len, length);
      for (int i = tidx; i < overflow; i += kBlockThreads) {
        const int pos = atomicAdd(&s->num_cand[0], 1);
        const int idx = spill1[source_lo + i];
        if (pos < Smem::CANDIDATE_CAPACITY) {
          s->cand_idx[0][pos] = idx;
        } else {
          const int gpos = atomicAdd(&s->num_spilled_cand[0], 1);
          spill0[gpos] = idx;
        }
      }
      __syncthreads();
    }
  }
  cluster.sync();

  if (rank != 0) {
    return;
  }
  if (topk_remaining != 0) {
    refine_rounds<kTopK, false, kBlockThreads>(s, score, spill0, spill1, topk_remaining, tidx);
  }
  for (int i = tidx; i < kTopK; i += kBlockThreads) {
    dst[i] = s->indices[i];
  }
}

// Exact cooperative route for very long rows.
// Eight CTAs first build disjoint exact coarse histograms. After the primary
// locates the boundary, they rescan their own segments and publish only the
// definitely selected prefix and the boundary-bin candidates through DSM. The
// primary alone runs the FP32 refinement rounds.
template <uint32_t VEC, int kTopK, int kSplits, int kBlockThreads>
__global__ void __launch_bounds__(kBlockThreads, 2)
    topk_filtered_cluster8_exact_kernel(const float* __restrict__ input,
                                        int32_t* __restrict__ output,
                                        const int32_t* __restrict__ ke,
                                        int32_t* __restrict__ spill_buffer,
                                        const int32_t* __restrict__ num_valid_rows,
                                        uint32_t row_stride, uint32_t out_stride, uint32_t n_cols) {
  namespace cg = cooperative_groups;
  using Smem = RowSmem<kTopK, false, kBlockThreads>;

  cg::cluster_group cluster = cg::this_cluster();
  const int tidx = threadIdx.x;
  const int rank = static_cast<int>(cluster.block_rank());
  const int row_i = static_cast<int>(blockIdx.y);
  if (row_i >= *num_valid_rows) {
    return;
  }

  extern __shared__ uint8_t smem_raw[];
  Smem* s = reinterpret_cast<Smem*>(smem_raw);
  Smem* primary = cluster.map_shared_rank(s, 0);
  const float* score = input + static_cast<size_t>(row_i) * row_stride;
  int32_t* dst = output + static_cast<size_t>(row_i) * out_stride;
  const int length = ke[row_i];
  int32_t* spill0 = spill_buffer + static_cast<size_t>(row_i) * 2 * n_cols;
  int32_t* spill1 = spill0 + n_cols;

  if (length <= kTopK) {
    if (rank == 0) {
      for (int i = tidx; i < kTopK; i += kBlockThreads) {
        dst[i] = (i < length) ? i : -1;
      }
    }
    return;
  }

  const int seg_len = (length + kSplits - 1) / kSplits;
  const int seg_lo = min(rank * seg_len, length);
  const int seg_hi = min(seg_lo + seg_len, length);

  // Pass 1: exact coarse histogram, distributed over the row segments.
  for (int i = tidx; i < RADIX + 1; i += kBlockThreads) {
    s->histogram[i] = 0;
  }
  __syncthreads();

  auto add_coarse = [&](float raw) { atomicAdd(&s->histogram[to_coarse_key(raw)], 1); };
  if constexpr (VEC == 4) {
    const int v_lo = min((seg_lo + 7) & ~7, seg_hi);
    const int v_hi = max(seg_hi & ~7, v_lo);
    for (int i = seg_lo + tidx; i < v_lo; i += kBlockThreads) {
      add_coarse(score[i]);
    }
    const int n8 = (v_hi - v_lo) >> 3;
    for (int g = tidx; g < n8; g += kBlockThreads) {
      const float* p = score + v_lo + (g << 3);
      const float4 a = ld_cg_float4(p);
      const float4 b = ld_cg_float4(p + 4);
      add_coarse(a.x);
      add_coarse(a.y);
      add_coarse(a.z);
      add_coarse(a.w);
      add_coarse(b.x);
      add_coarse(b.y);
      add_coarse(b.z);
      add_coarse(b.w);
    }
    for (int i = v_hi + tidx; i < seg_hi; i += kBlockThreads) {
      add_coarse(score[i]);
    }
  } else {
    for (int i = seg_lo + tidx; i < seg_hi; i += kBlockThreads) {
      add_coarse(score[i]);
    }
  }
  __syncthreads();

  cluster_allreduce_histogram<kTopK, kSplits, kBlockThreads>(s, cluster, rank, tidx);
  // histogram[RADIX] was cleared with the rest of the allocation and is not
  // touched by the all-reduce. The prefix helper resets num_selected when it
  // publishes the threshold, so no intermediate CTA rendezvous is required.
  prefix_sum_find_threshold<kTopK, false, -1, kBlockThreads>(s, tidx, kTopK, /*next_idx=*/0,
                                                             &s->num_selected,
                                                             /*reset_value=*/0);
  if (tidx == 0) {
    const int local_threshold = s->threshold_bin;
    s->ties_remaining = kTopK - (local_threshold == 0 ? 0 : s->histogram[local_threshold - 1]);
  }
  // Every rank owns the same reduced histogram and computes the same boundary.
  // Only the thread-0 remainder write needs publication here; no peer state is
  // consumed until the post-collect cluster barrier.
  __syncthreads();

  const int threshold = s->threshold_bin;
  const int topk_remaining = s->ties_remaining;
  const int coarse_candidate_count =
      s->histogram[threshold] - (threshold == 0 ? 0 : s->histogram[threshold - 1]);
  const bool large_boundary = topk_remaining != 0 && coarse_candidate_count > kBlockThreads;
  const PackedCoarseBoundary select_boundary = pack_coarse_boundary(coarse_boundary(threshold - 1));
  const PackedCoarseBoundary candidate_boundary = pack_coarse_boundary(coarse_boundary(threshold));

  if constexpr (VEC == 4) {
    if (large_boundary) {
      if (rank == 0) {
        refine_large_boundary_single_cta<kTopK, kBlockThreads>(
            s, score, dst, length, select_boundary, candidate_boundary, topk_remaining);
      }
      cluster.sync();
      return;
    }
  }

  // Pass 2: retain only the compact exact upper-tail state from this segment.
  // The cumulative coarse histogram tells us before the scan whether the
  // boundary can use the one-candidate-per-thread tail.  A larger boundary
  // reuses the dead allocation as its first exact histogram instead of
  // materializing a list that must later be rebuilt.
  auto classify = [&](float raw, int i) {
    const CoarseClass coarse_class = coarse_classify(raw, select_boundary, candidate_boundary);
    if (coarse_class == CoarseClass::kSelected) {
      const int pos = atomicAdd(&s->num_selected, 1);
      if constexpr (VEC == 4) {
        if (rank == 0) {
          dst[pos] = i;
        } else {
          s->indices[pos] = i;
        }
      } else {
        s->indices[pos] = i;
      }
    } else if (coarse_class == CoarseClass::kCandidate && topk_remaining != 0) {
      const int pos = atomicAdd(&s->num_cand[0], 1);
      if (pos < Smem::CANDIDATE_CAPACITY) {
        if constexpr (VEC == 4) {
          cluster_candidate_pairs(s)[pos] = pack_cluster_candidate(i, __float_as_uint(raw));
        } else {
          s->cand_idx[0][pos] = i;
          s->cand_idx[1][pos] = static_cast<int32_t>(to_ordered(raw));
        }
      } else {
        if constexpr (VEC != 4) {
          const int gpos = atomicAdd(&s->num_spilled_cand[0], 1);
          spill1[seg_lo + gpos] = i;
        }
      }
    }
  };

  if constexpr (VEC == 4) {
    const int v_lo = min((seg_lo + 3) & ~3, seg_hi);
    const int v_hi = max(seg_hi & ~3, v_lo);
    for (int i = seg_lo + tidx; i < v_lo; i += kBlockThreads) {
      classify(score[i], i);
    }
    const int n4 = (v_hi - v_lo) >> 2;
    int g = tidx;
    float4 next;
    if (g < n4) {
      next = ld_cg_float4(score + v_lo + (g << 2));
    }
    while (g < n4) {
      const float4 current = next;
      const int base = v_lo + (g << 2);
      g += kBlockThreads;
      if (g < n4) {
        next = ld_cg_float4(score + v_lo + (g << 2));
      }
      classify(current.x, base);
      classify(current.y, base + 1);
      classify(current.z, base + 2);
      classify(current.w, base + 3);
    }
    for (int i = v_hi + tidx; i < seg_hi; i += kBlockThreads) {
      classify(score[i], i);
    }
  } else {
    for (int i = seg_lo + tidx; i < seg_hi; i += kBlockThreads) {
      classify(score[i], i);
    }
  }

  // Reuse dead coarse-stage scalars as this CTA's output reservations. The
  // aligned path can merge peer atomics into the primary's live counters while
  // the primary finishes its own collect. The unaligned fallback still stages
  // selected indices through primary DSM and therefore retains its snapshot.
  if constexpr (VEC != 4) {
    __syncthreads();
    if (tidx == 0 && rank == 0) {
      s->threshold_bin = s->num_selected;
      s->ties_remaining = s->num_cand[0];
    }
    cluster.sync();
  }

  // Only peers consume their CTA-local counters and staging buffers here.  For
  // aligned rows, the primary's local and peer DSM atomics may safely overlap:
  // every reservation is atomic and therefore receives a disjoint range.  Let
  // the primary proceed directly to the cluster rendezvous instead of making
  // it pay two block barriers whose data it never reads.
  if constexpr (VEC == 4) {
    if (rank != 0) {
      __syncthreads();
    }
  }
  if (rank != 0 && tidx == 0) {
    s->threshold_bin = atomicAdd(&primary->num_selected, s->num_selected);
    s->ties_remaining = atomicAdd(&primary->num_cand[0], s->num_cand[0]);
    if constexpr (VEC == 4) {
      // Publish the peer's reserved candidate interval in dead coarse
      // histogram storage.  The primary can then load one candidate directly
      // from its owner CTA instead of materializing a second DSM copy.
      primary->histogram[rank] = s->ties_remaining;
      primary->histogram[kSplits + rank] = s->num_cand[0];
    }
  }
  if constexpr (VEC == 4) {
    if (rank != 0) {
      __syncthreads();
    }
  } else {
    __syncthreads();
  }

  if constexpr (VEC == 4) {
    cluster.sync();
    const int aggregate_candidates = primary->num_cand[0];
    if (rank == 0) {
      bool active = tidx < aggregate_candidates;
      uint64_t raw_pair = 0;
      if (active) {
        int owner = 0;
        int local_pos = tidx;
#pragma unroll
        for (int sp = 1; sp < kSplits; ++sp) {
          const int start = primary->histogram[sp];
          const int count = primary->histogram[kSplits + sp];
          if (tidx >= start && tidx < start + count) {
            owner = sp;
            local_pos = tidx - start;
          }
        }
        Smem* source = cluster.map_shared_rank(s, owner);
        raw_pair = cluster_candidate_pairs(source)[local_pos];
      }
      // All metadata reads must finish before refinement recycles the coarse
      // histogram as its first fine histogram.
      __syncthreads();
      if (topk_remaining != 0) {
        refine_cluster_register_candidate<kTopK, kBlockThreads>(s, raw_pair, active, topk_remaining,
                                                                tidx, dst);
      }
    } else {
      const int selected_prefix = s->threshold_bin;
      for (int i = tidx; i < s->num_selected; i += kBlockThreads) {
        dst[selected_prefix + i] = s->indices[i];
      }
    }
    cluster.sync();
    return;
  }

  if (rank != 0) {
    const int candidate_prefix = s->ties_remaining;
    const int selected_prefix = s->threshold_bin;
    for (int i = tidx; i < s->num_selected; i += kBlockThreads) {
      primary->indices[selected_prefix + i] = s->indices[i];
    }
    if (candidate_prefix < Smem::CANDIDATE_CAPACITY) {
      const int copy_candidates = min(s->num_cand[0], Smem::CANDIDATE_CAPACITY - candidate_prefix);
      for (int i = tidx; i < copy_candidates; i += kBlockThreads) {
        primary->cand_idx[0][candidate_prefix + i] = s->cand_idx[0][i];
        primary->cand_idx[1][candidate_prefix + i] = s->cand_idx[1][i];
      }
    }
  }
  cluster.sync();

  if (tidx == 0) {
    // num_cand counts every boundary element, including local overflow, so the
    // aggregate capacity test also proves that no CTA spilled candidates.
    s->num_cand[1] = primary->num_cand[0] <= Smem::CANDIDATE_CAPACITY ? 1 : 0;
  }
  __syncthreads();
  const bool resident = s->num_cand[1] != 0;
  if (resident) {
    if (rank == 0) {
      s->num_spilled_cand[0] = 0;
      if (topk_remaining != 0) {
        for (int bin = tidx; bin < RADIX + 1; bin += kBlockThreads) {
          s->histogram[bin] = 0;
        }
        __syncthreads();
        for (int pos = tidx; pos < s->num_cand[0]; pos += kBlockThreads) {
          const uint32_t ordered = static_cast<uint32_t>(s->cand_idx[1][pos]);
          const int sub = (ordered >> kFineShifts[0]) & kFineMasks[0];
          atomicAdd(&s->histogram[sub], 1);
        }
        __syncthreads();
        refine_resident_keys(s, topk_remaining, tidx);
      }
      for (int i = tidx; i < kTopK; i += kBlockThreads) {
        dst[i] = s->indices[i];
      }
    }
    // Peers publish disjoint selected ranges while the primary refines the
    // boundary. Cluster blocks must still rendezvous before any peer exits;
    // otherwise the cluster's DSM lifetime ends while the primary is active.
    cluster.sync();
    return;
  }

  // Keep shared counters alive until every peer observes the overflow decision.
  cluster.sync();
  run_cluster_unaligned_overflow<kTopK, kSplits, kBlockThreads>(
      s, score, dst, spill0, spill1, length, seg_lo, seg_len, topk_remaining);
}

// Both kernels need more dynamic smem than the 48 KB default, which requires an
// explicit per-kernel opt-in.
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

// Vector width for the row scans, from the row-stride alignment.
inline uint32_t vector_width(int row_stride) {
  return (row_stride % 4 == 0) ? 4u : (row_stride % 2 == 0) ? 2u : 1u;
}

// kSampled picks the stage-1 strategy: false scans the row once into an exact
// coarse histogram; true predicts a threshold from a sparse sample first and only
// falls back to the exact histogram when the prediction undershoots.
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
  // Ragged captures can have short resident rows under a wider static stride.
  // Separate nodes keep the register and streaming worker states independent.
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

template <int kTopK, int kSplits, int kBlockThreads>
bool launch_cluster8_exact(int* topk_indices, const float* logits, const int* ke,
                           const int* num_valid_rows, int m, int n, int row_stride, int out_stride,
                           void* workspace, size_t workspace_bytes, cudaStream_t stream) {
  if (get_sm_major_version() < 9 || m <= 0 || workspace_bytes < one_cta_workspace_bytes(n)) {
    return false;
  }

  int32_t* spill = static_cast<int32_t*>(workspace);
  const uint32_t vec = vector_width(row_stride);
  const auto kernel =
      (vec == 4)   ? topk_filtered_cluster8_exact_kernel<4, kTopK, kSplits, kBlockThreads>
      : (vec == 2) ? topk_filtered_cluster8_exact_kernel<2, kTopK, kSplits, kBlockThreads>
                   : topk_filtered_cluster8_exact_kernel<1, kTopK, kSplits, kBlockThreads>;

  const size_t smem_bytes = sizeof(RowSmem<kTopK, false, kBlockThreads>);
  if (!opt_in_dynamic_smem(reinterpret_cast<const void*>(kernel), smem_bytes)) {
    return false;
  }

  cudaLaunchAttribute attribute[1]{};
  attribute[0].id = cudaLaunchAttributeClusterDimension;
  attribute[0].val.clusterDim.x = kSplits;
  attribute[0].val.clusterDim.y = 1;
  attribute[0].val.clusterDim.z = 1;

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(kSplits, m);
  config.blockDim = dim3(kBlockThreads);
  config.dynamicSmemBytes = smem_bytes;
  config.stream = stream;
  config.attrs = attribute;
  config.numAttrs = 1;

  return cudaLaunchKernelEx(&config, kernel, logits, topk_indices, ke, spill, num_valid_rows,
                            static_cast<uint32_t>(row_stride), static_cast<uint32_t>(out_stride),
                            static_cast<uint32_t>(n)) == cudaSuccess;
}

// kSplits CTAs cooperate per row. Always sampled, with the exact full-row coarse
// stage as the per-row fallback.
template <int kTopK, int kSplits, int kSampleStride = SAMPLE_STRIDE, int kSampleRank = SAMPLE_RANK>
bool launch_kv_split(int* topk_indices, const float* logits, const int* ke,
                     const int* num_valid_rows, int m, int n, int row_stride, int out_stride,
                     void* counters, size_t counters_bytes, void* workspace, size_t workspace_bytes,
                     cudaStream_t stream) {
  if (counters_bytes < counters_bytes_for(m) ||
      workspace_bytes < split_workspace_bytes(m, kSplits, n)) {
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

constexpr int SPLIT_MIN_N = 128 * 1024 + 1;
constexpr int SPLIT_MAX_ROWS = 192;

// Returning 1 means that row splitting does not pay for this shape. The
// shortest batches retain eight segments while their grid fits in roughly one
// SM-wide wave; the remaining bands use four and two segments, respectively.
inline int choose_splits(int m, int n) {
  const int grid = grid_num_blocks();
  if (m <= 0 || n < SPLIT_MIN_N || m > SPLIT_MAX_ROWS) {
    return 1;
  }
  const int eight_way_max_rows = std::max(16, get_sm_count() / 8);
  if (m <= eight_way_max_rows) {
    return 8;
  }
  if (m <= grid / 4) {
    return 4;
  }
  return 2;
}

enum class DispatchKind : uint8_t {
  kRowLocal1024Exact,
  kCluster8Exact,
  kKvSplitSampledExact,
  kPersistentExact,
  kPersistentSampledExact,
};

struct DispatchPlan {
  DispatchKind kind;
  int splits = 1;
  bool wide_sample = false;
};

// Each interval keeps the row-local grid within two resident waves while
// increasing scan length only after enough rows are available to fill the GPU.
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

// A single shape-only plan drives both launch selection and workspace sizing.
// `m` and `n` are captured capacities; no runtime row metadata participates.
inline DispatchPlan choose_topk2048_plan(int m, int n) {
  constexpr int kClusterNarrowMaxRows = 8;
  constexpr int kClusterMaxRows = 16;
  constexpr int kClusterNarrowMinN = 32 * 1024;
  constexpr int kClusterMinN = 64 * 1024;
  constexpr int kSampleMinN = 32768;
  constexpr int64_t kWideSampleMinWork = int64_t{32} * 1024 * 1024;

  const bool use_cluster = get_sm_major_version() >= 9 && m > 0 &&
                           ((m <= kClusterNarrowMaxRows && n >= kClusterNarrowMinN) ||
                            (m <= kClusterMaxRows && n >= kClusterMinN));
  if (use_cluster) {
    return {DispatchKind::kCluster8Exact};
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

bool topk_filtered_async(int* topk_indices, const float* logits, const int* ke, int topk,
                         const int* num_valid_rows, int m, int n, int row_stride, int out_stride,
                         void* counters, size_t counters_bytes, void* workspace,
                         size_t workspace_bytes, cudaStream_t stream) {
  // Path selection reads only the static shape (topk, m, n) and never
  // num_valid_rows, so a captured CUDA graph always replays the kernel it
  // captured. `m` is the static row capacity, not the number of valid rows.
  if (topk == 512) {
    // No sampled variant at 512: k is small enough that the exact coarse
    // histogram already lands on a tight boundary bin, so sampling buys nothing.
    return filtered::launch_one_cta_per_row<512, /*kSampled=*/false>(
        topk_indices, logits, ke, num_valid_rows, m, n, row_stride, out_stride, counters,
        counters_bytes, workspace, workspace_bytes, stream);
  }
  if (topk != 2048) {
    return false;
  }

  constexpr int kClusterSplits = 8;
  const filtered::DispatchPlan plan = filtered::choose_topk2048_plan(m, n);
  switch (plan.kind) {
    case filtered::DispatchKind::kRowLocal1024Exact:
      if (filtered::launch_row_local_1024_exact<2048>(
              topk_indices, logits, ke, num_valid_rows, m, n, row_stride, out_stride, workspace,
              workspace_bytes, counters, counters_bytes, stream)) {
        return true;
      }
      break;
    case filtered::DispatchKind::kCluster8Exact:
      if (filtered::launch_cluster8_exact<2048, kClusterSplits, filtered::SHORT_THREADS>(
              topk_indices, logits, ke, num_valid_rows, m, n, row_stride, out_stride, workspace,
              workspace_bytes, stream)) {
        return true;
      }
      break;
    case filtered::DispatchKind::kKvSplitSampledExact:
      if (counters_bytes >= filtered::counters_bytes_for(m) &&
          workspace_bytes >= filtered::split_workspace_bytes(m, plan.splits, n)) {
        switch (plan.splits) {
          case 8:
            return filtered::launch_kv_split<2048, 8>(
                topk_indices, logits, ke, num_valid_rows, m, n, row_stride, out_stride, counters,
                counters_bytes, workspace, workspace_bytes, stream);
          case 4:
            return filtered::launch_kv_split<2048, 4>(
                topk_indices, logits, ke, num_valid_rows, m, n, row_stride, out_stride, counters,
                counters_bytes, workspace, workspace_bytes, stream);
          case 2:
            return filtered::launch_kv_split<2048, 2>(
                topk_indices, logits, ke, num_valid_rows, m, n, row_stride, out_stride, counters,
                counters_bytes, workspace, workspace_bytes, stream);
          default:
            break;
        }
      }
      break;
    case filtered::DispatchKind::kPersistentExact:
      return filtered::launch_one_cta_per_row<2048, /*kSampled=*/false>(
          topk_indices, logits, ke, num_valid_rows, m, n, row_stride, out_stride, counters,
          counters_bytes, workspace, workspace_bytes, stream);
    case filtered::DispatchKind::kPersistentSampledExact:
      if (plan.wide_sample) {
        return filtered::launch_one_cta_per_row<
            2048, /*kSampled=*/true, filtered::WIDE_SAMPLE_STRIDE, filtered::WIDE_SAMPLE_RANK>(
            topk_indices, logits, ke, num_valid_rows, m, n, row_stride, out_stride, counters,
            counters_bytes, workspace, workspace_bytes, stream);
      }
      return filtered::launch_one_cta_per_row<2048, /*kSampled=*/true>(
          topk_indices, logits, ke, num_valid_rows, m, n, row_stride, out_stride, counters,
          counters_bytes, workspace, workspace_bytes, stream);
  }

  // A caller may intentionally provide only the minimum one-CTA scratch. If a
  // preferred cooperative launch cannot be formed, preserve exactness with the
  // corresponding persistent route.
  if (n < 32768) {
    return filtered::launch_one_cta_per_row<2048, /*kSampled=*/false>(
        topk_indices, logits, ke, num_valid_rows, m, n, row_stride, out_stride, counters,
        counters_bytes, workspace, workspace_bytes, stream);
  }
  if (plan.wide_sample) {
    return filtered::launch_one_cta_per_row<2048, /*kSampled=*/true, filtered::WIDE_SAMPLE_STRIDE,
                                            filtered::WIDE_SAMPLE_RANK>(
        topk_indices, logits, ke, num_valid_rows, m, n, row_stride, out_stride, counters,
        counters_bytes, workspace, workspace_bytes, stream);
  }
  return filtered::launch_one_cta_per_row<2048, /*kSampled=*/true>(
      topk_indices, logits, ke, num_valid_rows, m, n, row_stride, out_stride, counters,
      counters_bytes, workspace, workspace_bytes, stream);
}

size_t topk_filtered_counters_bytes(int num_rows) { return filtered::counters_bytes_for(num_rows); }

size_t topk_filtered_workspace_bytes(int num_rows, int max_kv_len) {
  const filtered::DispatchPlan plan = filtered::choose_topk2048_plan(num_rows, max_kv_len);
  if (plan.kind == filtered::DispatchKind::kKvSplitSampledExact) {
    return filtered::split_workspace_bytes(num_rows, plan.splits, max_kv_len);
  }
  return filtered::one_cta_workspace_bytes(max_kv_len);
}

size_t topk_filtered_min_workspace_bytes(int max_kv_len) {
  return filtered::one_cta_workspace_bytes(max_kv_len);
}

size_t topk_filtered_peak_workspace_bytes(int max_kv_len) {
  // Cooperative plans occur only through one row past half the grid. Their
  // buffers are not monotonic in that interval -- a larger row count can cross
  // into a smaller split count or the row-local envelope -- so take the max
  // rather than evaluating an endpoint. The range is a fixed device property.
  size_t peak = filtered::one_cta_workspace_bytes(max_kv_len);
  const int max_split_rows = filtered::SPLIT_MAX_ROWS;
  for (int num_rows = 1; num_rows <= max_split_rows; ++num_rows) {
    peak = std::max(peak, topk_filtered_workspace_bytes(num_rows, max_kv_len));
  }
  return peak;
}

}  // namespace topk
}  // namespace hpc

#endif  // SRC_TOPK_TOPK_FILTERED_IMPL_CUH_
