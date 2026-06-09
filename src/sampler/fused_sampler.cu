// Copyright (C) 2026 Tencent.
//
// Fused sampler — 2-kernel pipeline. Pipeline:
//   rp -> temp -> [softmax1] -> topk -> [softmax2] -> topp -> Gumbel-max -> penalty writeback
//
//   K1 fused_scan_topk_kernel  grid=(N,B) block=1024: single vocab pass →
//      rp+temp → online max(+sum) → BlockRadixSort top-K per block.
//   K2 stage2_kernel           grid=(B,) block=Nmax*K: merge N blocks' top-K,
//      softmax/topp/Gumbel-max sample, write token + penalty bit.
//
//   N = blocks-per-batch (runtime), Nmax = min(1024/K, 32) compile-time cap.
//   Template axes: DType, kVocabSize, kMaxTopK{32,64}, kSoftmaxPolicy, kHasTopP.
//   Optional features disabled by null/zero: rp, temperature, topk, gumbel_noise.

#include <cuda.h>
#include <cuda_bf16.h>
#include <curand_kernel.h>

#include <algorithm>
#include <cub/cub.cuh>  // NOLINT
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "src/sampler/sampler.h"
#include "src/sampler/sampler_rng.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace sampler {
namespace fused_sampler_kernels {

constexpr int kThreadsPerBlock = 1024;
constexpr int kItemsPerThread = 4;
constexpr int kSoftmaxNone = 0;
constexpr int kSoftmaxBeforeTopk = 1;
constexpr int kSoftmaxAfterTopk = 2;
constexpr float kNegInfF = -std::numeric_limits<float>::infinity();

// Nmax: compile-time blocks-per-batch cap = min(1024/K, 32). 1024/K bounds K2's
// block (Nmax*K ≤ 1024); 32 bounds K2's warp-0 partial reduce (one lane/block).
template <int kMaxTopK>
constexpr int kMaxBlocksPerBatch =
    (kThreadsPerBlock / kMaxTopK) < 32 ? (kThreadsPerBlock / kMaxTopK) : 32;

// ---------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------

template <typename DType>
__device__ __forceinline__ void load_4_as_float(const DType* gmem, float* out) {
  if constexpr (std::is_same_v<DType, float>) {
    vec_t<float, kItemsPerThread> v = load<float, kItemsPerThread>(gmem);
#pragma unroll
    for (int i = 0; i < kItemsPerThread; i++) {
      out[i] = v[i];
    }
  } else {
    vec_t<__nv_bfloat16, kItemsPerThread> v = load<__nv_bfloat16, kItemsPerThread>(gmem);
#pragma unroll
    for (int i = 0; i < kItemsPerThread; i++) {
      out[i] = __bfloat162float(v[i]);
    }
  }
}

template <typename DType, int kVocabSize>
__device__ __forceinline__ void load_logits_safe(const DType* logits_batch, int col_base,
                                                 float* vals) {
  if (col_base + kItemsPerThread <= kVocabSize) {
    load_4_as_float<DType>(logits_batch + col_base, vals);
  } else {
#pragma unroll
    for (int i = 0; i < kItemsPerThread; i++) {
      int col = col_base + i;
      if (col < kVocabSize) {
        if constexpr (std::is_same_v<DType, float>) {
          vals[i] = logits_batch[col];
        } else {
          vals[i] = __bfloat162float(logits_batch[col]);
        }
      } else {
        vals[i] = kNegInfF;
      }
    }
  }
}

__device__ __forceinline__ void apply_rp_temp(float* vals, int col_base, bool rp_active, float rp,
                                              float inv_rp, const uint8_t* penalty_row,
                                              bool temp_active, float inv_t, int vocab_size) {
  if (rp_active) {
#pragma unroll
    for (int i = 0; i < kItemsPerThread; i++) {
      int col = col_base + i;
      if (col < vocab_size) {
        int byte_idx = col >> 3;
        int bit_idx = col & 0x7;
        if ((penalty_row[byte_idx] >> bit_idx) & 1u) {
          vals[i] *= (vals[i] > 0.f) ? inv_rp : rp;
        }
      }
    }
  }
  if (temp_active) {
#pragma unroll
    for (int i = 0; i < kItemsPerThread; i++) {
      vals[i] *= inv_t;
    }
  }
}

__device__ __forceinline__ float block_reduce_max_smem(float v, float* warp_scratch) {
  constexpr int kWarpSize = 32;
  constexpr int kNumWarps = kThreadsPerBlock / kWarpSize;
  int iwarp = threadIdx.x / kWarpSize;
  int ilane = threadIdx.x % kWarpSize;
  v = warp_reduce_max_xor(v);
  if (ilane == 0) {
    warp_scratch[iwarp] = v;
  }
  __syncthreads();
  if (iwarp == 0) {
    float u = (ilane < kNumWarps) ? warp_scratch[ilane] : kNegInfF;
    u = warp_reduce_max_xor(u);
    if (ilane == 0) {
      warp_scratch[0] = u;
    }
  }
  __syncthreads();
  return warp_scratch[0];
}

__device__ __forceinline__ float block_reduce_sum_smem(float v, float* warp_scratch) {
  constexpr int kWarpSize = 32;
  constexpr int kNumWarps = kThreadsPerBlock / kWarpSize;
  int iwarp = threadIdx.x / kWarpSize;
  int ilane = threadIdx.x % kWarpSize;
  v = warp_reduce_sum_xor(v);
  if (ilane == 0) {
    warp_scratch[iwarp] = v;
  }
  __syncthreads();
  if (iwarp == 0) {
    float u = (ilane < kNumWarps) ? warp_scratch[ilane] : 0.f;
    u = warp_reduce_sum_xor(u);
    if (ilane == 0) {
      warp_scratch[0] = u;
    }
  }
  __syncthreads();
  return warp_scratch[0];
}

// ---------------------------------------------------------------------------
// K1: single vocab pass → rp+temp → online max(+sum, BEFORE_TOPK only) →
// per-block BlockRadixSort top-K. grid=(N,B); loop count derives from runtime N.
// ---------------------------------------------------------------------------
template <typename DType, int kVocabSize, int kMaxTopK, int kSoftmaxPolicy>
__global__ __launch_bounds__(kThreadsPerBlock) void fused_scan_topk_kernel(
    float* mid_logits,    // [B, Nmax, kMaxTopK]
    int* mid_tokens,      // [B, Nmax, kMaxTopK]
    float* partial_max,   // [B, Nmax]
    float* partial_sum,   // [B, Nmax]
    const DType* logits,  // [B, V] with stride
    int logits_row_stride, const uint8_t* penalty_mask, const int32_t* slot_id, const float* rp_arr,
    float rp_val, const float* temp_arr, float temp_val, int penalty_row_bytes,
    int n_blocks_per_row,  // dynamic N (∈ [kNMin, Nmax])
    int scratch_stride) {  // output row stride = Nmax (compile-time cap)
  const int ibatch = blockIdx.y;
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;

  // --- Thread role ---
  constexpr int kKeeperThreads = kMaxTopK / kItemsPerThread;
  constexpr int kLoaderThreads = kThreadsPerBlock - kKeeperThreads;
  const bool is_keeper = (tid < kKeeperThreads);
  const int loader_id = tid - kKeeperThreads;  // valid only for loaders

  // --- Per-batch parameters ---
  const DType* logits_batch = logits + static_cast<int64_t>(ibatch) * logits_row_stride;

  const float rp = rp_arr ? rp_arr[ibatch] : rp_val;
  const bool rp_active = (rp > 0.f) && (penalty_mask != nullptr) && (slot_id != nullptr);
  const float inv_rp = rp_active ? rcpf_ftz(rp) : 0.f;
  const uint8_t* penalty_row =
      rp_active ? (penalty_mask + slot_id[ibatch] * penalty_row_bytes) : nullptr;

  const float t = temp_arr ? temp_arr[ibatch] : temp_val;
  const bool temp_active = (t > 0.f);
  const float inv_t = temp_active ? rcpf_ftz(t) : 0.f;

  // --- Sort types ---
  using BlockRadixSortT = cub::BlockRadixSort<float, kThreadsPerBlock, kItemsPerThread, int>;

  __shared__ union {
    typename BlockRadixSortT::TempStorage sort_temp;
    float warp_scratch[kThreadsPerBlock / 32];
  } shared;

  // --- Initialize per-thread state ---
  float thread_logits[kItemsPerThread];
  int thread_tokens[kItemsPerThread];

  // Keepers hold the rolling top-K; initialize to -inf before the load loop.
  if (is_keeper) {
#pragma unroll
    for (int i = 0; i < kItemsPerThread; i++) {
      thread_logits[i] = kNegInfF;
      thread_tokens[i] = -1;
    }
  }

  // Online max+sum accumulators — only used by BEFORE_TOPK loaders.
  // NONE keeps raw logits; AFTER_TOPK normalizes over the surviving top-K in K2.
  [[maybe_unused]] float thread_max = kNegInfF;
  [[maybe_unused]] float thread_sum = 0.f;

  // Load loop: runtime-bounded; `#pragma unroll 1` keeps register pressure flat.
  const int elements_per_load_loop = n_blocks_per_row * kLoaderThreads * kItemsPerThread;
  const int num_load_loops = (kVocabSize + elements_per_load_loop - 1) / elements_per_load_loop;

#pragma unroll 1
  for (int iloop = 0; iloop < num_load_loops; iloop++) {
    if (!is_keeper) {
      int col_base = iloop * elements_per_load_loop + bid * kLoaderThreads * kItemsPerThread +
                     loader_id * kItemsPerThread;

      float vals[kItemsPerThread];
      load_logits_safe<DType, kVocabSize>(logits_batch, col_base, vals);
      apply_rp_temp(vals, col_base, rp_active, rp, inv_rp, penalty_row, temp_active, inv_t,
                    kVocabSize);

      // Populate thread data for sort
#pragma unroll
      for (int i = 0; i < kItemsPerThread; i++) {
        int col = col_base + i;
        if (col < kVocabSize) {
          thread_logits[i] = vals[i];
          thread_tokens[i] = col;

          // Online max (+sum) — only BEFORE_TOPK needs these.
          if constexpr (kSoftmaxPolicy == kSoftmaxBeforeTopk) {
            float x = vals[i];
            if (x > thread_max) {
              thread_sum *= expf(thread_max - x);
              thread_max = x;
            }
            thread_sum += expf(x - thread_max);
          }
        } else {
          thread_logits[i] = kNegInfF;
          thread_tokens[i] = -1;
        }
      }
    }

    __syncthreads();
    BlockRadixSortT(shared.sort_temp).SortDescending(thread_logits, thread_tokens);
    __syncthreads();
  }

  // --- Post-loop: block reduce max+sum (BEFORE_TOPK only) ---
  if constexpr (kSoftmaxPolicy == kSoftmaxBeforeTopk) {
    float reduce_max = is_keeper ? kNegInfF : thread_max;
    float block_max = block_reduce_max_smem(reduce_max, shared.warp_scratch);

    float reduce_sum = is_keeper ? 0.f : thread_sum;
    // Empty tail block (block_max == -inf): contribute 0, else expf(NaN) poisons K2's sum.
    float corrected = (block_max == kNegInfF) ? 0.f : reduce_sum * expf(reduce_max - block_max);
    float block_sum = block_reduce_sum_smem(corrected, shared.warp_scratch);
    if (tid == 0) {
      partial_max[ibatch * scratch_stride + bid] = block_max;
      partial_sum[ibatch * scratch_stride + bid] = block_sum;
    }
  }

  // --- Write top-K outputs ---
  if (is_keeper) {
    int dst_base = ibatch * scratch_stride * kMaxTopK + bid * kMaxTopK + tid * kItemsPerThread;
#pragma unroll
    for (int i = 0; i < kItemsPerThread; i++) {
      mid_logits[dst_base + i] = thread_logits[i];
      mid_tokens[dst_base + i] = thread_tokens[i];
    }
  }
}

// ---------------------------------------------------------------------------
// K2: merge N blocks' top-K candidates → softmax/topp/Gumbel-max sample.
// Block size = Nmax*K (static, cub::BlockRadixSort); slots from blocks ≥ N
// masked to -inf.
// ---------------------------------------------------------------------------
template <int kMaxTopK, int kSoftmaxPolicy, bool kHasTopP>
__global__ void stage2_kernel(int32_t* token_ids_out,    // [B]
                              const float* mid_logits,   // [B, Nmax, kMaxTopK]
                              const int* mid_tokens,     // [B, Nmax, kMaxTopK]
                              const float* partial_max,  // [B, Nmax]
                              const float* partial_sum,  // [B, Nmax]
                              const void* topk_ptr, int topk_int_bytes, int topk_val,
                              const float* topp_ptr, float topp_val, const float* gumbel_noise_ptr,
                              int vocab_size, uint8_t* penalty_mask, const int32_t* slot_id,
                              const float* rp_arr, float rp_val, int penalty_row_bytes,
                              uint64_t rng_seed, uint64_t rng_base_offset, int n_blocks_per_row) {
  constexpr int kNmax = kMaxBlocksPerBatch<kMaxTopK>;
  constexpr int kThreadsPerBlock2 = kNmax * kMaxTopK;
  const int ibatch = blockIdx.x;
  const int tid = threadIdx.x;
  const int ilane = tid % 32;

  // --- Resolve effective top-K for this batch ---
  int effK = kMaxTopK;
  if (topk_ptr != nullptr) {
    int user_k;
    if (topk_int_bytes == 4) {
      user_k = reinterpret_cast<const int32_t*>(topk_ptr)[ibatch];
    } else {
      user_k = static_cast<int>(reinterpret_cast<const int64_t*>(topk_ptr)[ibatch]);
    }
    if (user_k > 0) {
      effK = min(user_k, kMaxTopK);
    }
  } else if (topk_val > 0) {
    effK = min(topk_val, kMaxTopK);
  }

  // --- Phase 0: reduce partial_max/sum → global_max, inv_sum [BEFORE_TOPK only] ---
  // Warp 0 reduces; lanes ≥ N use identity (only first N slots written by K1).
  [[maybe_unused]] float global_max = 0.f;
  [[maybe_unused]] float inv_sum = 0.f;
  if constexpr (kSoftmaxPolicy == kSoftmaxBeforeTopk) {
    __shared__ float smem_global_max;
    __shared__ float smem_inv_sum;

    if (tid < 32) {
      float pm = (ilane < n_blocks_per_row) ? partial_max[ibatch * kNmax + ilane] : kNegInfF;
      float gmax = warp_reduce_max_xor(pm);
      float ps = (ilane < n_blocks_per_row) ? partial_sum[ibatch * kNmax + ilane] : 0.f;
      float corrected = ps * expf(pm - gmax);  // rebase each block's sum to global_max
      float global_sum = warp_reduce_sum_xor(corrected);
      if (ilane == 0) {
        smem_global_max = gmax;
        smem_inv_sum = (global_sum > 0.f) ? rcpf_ftz(global_sum) : 0.f;
      }
    }
    __syncthreads();

    global_max = smem_global_max;
    inv_sum = smem_inv_sum;
  }

  // --- Phase 1: load candidates (mask blocks ≥ N) + probability conversion ---
  using BlockRadixSortT = cub::BlockRadixSort<float, kThreadsPerBlock2, 1, int>;

  __shared__ union {
    typename BlockRadixSortT::TempStorage sort_temp;
    struct {
      float final_logits[kThreadsPerBlock2];
      int final_tokens[kThreadsPerBlock2];
    } data;
  } shared2;

  const int cand_bid = tid / kMaxTopK;
  float key[1];
  int val[1];
  if (cand_bid < n_blocks_per_row) {
    int idx_in = ibatch * kNmax * kMaxTopK + tid;
    key[0] = mid_logits[idx_in];
    val[0] = mid_tokens[idx_in];
  } else {
    key[0] = kNegInfF;
    val[0] = -1;
  }

  // Convert based on softmax policy
  if constexpr (kSoftmaxPolicy == kSoftmaxBeforeTopk) {
    // Full-vocab normalized probability
    if (val[0] >= 0) {
      key[0] = expf(key[0] - global_max) * inv_sum;
    } else {
      key[0] = 0.f;
    }
  }
  // NONE / AFTER_TOPK: keep raw logits for sorting.

  // --- Phase 2: BlockRadixSort merge ---
  __syncthreads();
  BlockRadixSortT(shared2.sort_temp).SortDescending(key, val);
  __syncthreads();

  // Store sorted results to shared memory
  shared2.data.final_logits[tid] = key[0];
  shared2.data.final_tokens[tid] = val[0];
  __syncthreads();

  // === Warp 0 only: phases 2.5-5 ===
  if (tid >= 32) {
    return;
  }

  constexpr int kItemsPerLane = kMaxTopK / 32;
  float w_logit[kItemsPerLane];
  int w_tok[kItemsPerLane];
  float w_prob[kItemsPerLane];

  // Load from shared (top-effK)
#pragma unroll
  for (int i = 0; i < kItemsPerLane; i++) {
    int idx = ilane * kItemsPerLane + i;
    w_logit[i] = shared2.data.final_logits[idx];
    w_tok[i] = shared2.data.final_tokens[idx];
    w_prob[i] = 0.f;
  }

  // --- Phase 2.5: AFTER_TOPK local softmax over top-K ---
  if constexpr (kSoftmaxPolicy == kSoftmaxAfterTopk) {
    float local_max = __shfl_sync(0xffffffff, w_logit[0], 0);  // sorted pos 0
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < kItemsPerLane; i++) {
      int idx = ilane * kItemsPerLane + i;
      if (idx < effK && w_tok[i] >= 0) {
        w_prob[i] = expf(w_logit[i] - local_max);
        sum += w_prob[i];
      }
    }
    sum = warp_reduce_sum_xor(sum);
    float inv = (sum > 0.f) ? rcpf_ftz(sum) : 0.f;
#pragma unroll
    for (int i = 0; i < kItemsPerLane; i++) {
      w_prob[i] *= inv;
    }
  } else if constexpr (kSoftmaxPolicy == kSoftmaxBeforeTopk) {
    // Values are already full-vocab probabilities from Phase 1.
#pragma unroll
    for (int i = 0; i < kItemsPerLane; i++) {
      int idx = ilane * kItemsPerLane + i;
      w_prob[i] = (idx < effK && w_tok[i] >= 0) ? w_logit[i] : 0.f;
    }
  }
  // NONE: w_prob stays 0 (sampling uses raw logits directly).

  // --- Phase 3: Top-P truncation ---
  float lane_run = 0.f;  // exclusive prefix sum at this lane's start
  if constexpr (kHasTopP) {
    float topp_b = topp_ptr ? topp_ptr[ibatch] : topp_val;
    if (topp_b > 0.f) {
      // Compute per-lane sum of probabilities
      float lane_sum = 0.f;
#pragma unroll
      for (int i = 0; i < kItemsPerLane; i++) {
        int idx = ilane * kItemsPerLane + i;
        if (idx < effK) {
          lane_sum += w_prob[i];
        }
      }
      // Warp inclusive prefix scan
      float prefix = lane_sum;
      for (int offset = 1; offset < 32; offset <<= 1) {
        float v = __shfl_up_sync(0xffffffff, prefix, offset);
        if (ilane >= offset) {
          prefix += v;
        }
      }
      lane_run = prefix - lane_sum;  // exclusive prefix for this lane
    }
  }

  // --- Phase 4: Gumbel-max sampling ---
  float best_key = kNegInfF;
  int best_tok = -1;

  // Self-drawn RNG: Philox sequence = ibatch*32+ilane (same contract as
  // fused_sampler_temperature.cu). External gumbel_noise_ptr is used directly.
  static_assert(kItemsPerLane <= 4, "curand_uniform4 supplies at most 4 draws per lane");
  curandStatePhilox4_32_10_t rng;
  float u_vec[kItemsPerLane] = {};
  if (gumbel_noise_ptr == nullptr) {
    uint64_t sequence = static_cast<uint64_t>(ibatch) * 32 + ilane;
    curand_init(rng_seed, sequence, rng_base_offset, &rng);
    float4 r = curand_uniform4(&rng);
    const float rr[4] = {r.x, r.y, r.z, r.w};
#pragma unroll
    for (int i = 0; i < kItemsPerLane; i++) {
      u_vec[i] = rr[i];
    }
  }

  float topp_threshold = 0.f;
  if constexpr (kHasTopP) {
    topp_threshold = topp_ptr ? topp_ptr[ibatch] : topp_val;
  }

  float running_cum = lane_run;
#pragma unroll
  for (int i = 0; i < kItemsPerLane; i++) {
    int idx = ilane * kItemsPerLane + i;
    int tok = w_tok[i];

    bool keep = (idx < effK) && (tok >= 0);
    if constexpr (kHasTopP) {
      if (topp_threshold > 0.f) {
        keep = keep && ((idx == 0) || (running_cum < topp_threshold));
      }
    }

    if (keep) {
      float value;
      if constexpr (kSoftmaxPolicy == kSoftmaxNone) {
        value = w_logit[i];
      } else {
        float p = w_prob[i];
        value = (p > 0.f) ? logf_ftz(p) : kNegInfF;
      }

      // External buffer holds Gumbel(0) directly; self-drawn path applies the
      // Gumbel transform from sampler_rng.cuh.
      float g;
      if (gumbel_noise_ptr != nullptr) {
        g = gumbel_noise_ptr[ibatch * vocab_size + tok];
      } else {
        g = ::hpc::sampler::rng::gumbel_noise_from_uniform(u_vec[i]);
      }

      float k = value + g;
      if (k > best_key || (k == best_key && (best_tok < 0 || tok < best_tok))) {
        best_key = k;
        best_tok = tok;
      }
    }

    if constexpr (kHasTopP) {
      running_cum += w_prob[i];
    }
  }

  // --- Warp argmax with tiebreak ---
  for (int offset = 16; offset > 0; offset >>= 1) {
    float other_key = __shfl_xor_sync(0xffffffff, best_key, offset);
    int other_tok = __shfl_xor_sync(0xffffffff, best_tok, offset);

    bool take;
    if (other_key > best_key) {
      take = true;
    } else if (other_key < best_key) {
      take = false;
    } else {
      // Tiebreak: prefer valid token, then smaller ID
      take = (other_tok >= 0) && (best_tok < 0 || other_tok < best_tok);
    }
    if (take) {
      best_key = other_key;
      best_tok = other_tok;
    }
  }

  // --- Phase 5: Output + penalty writeback ---
  if (ilane == 0) {
    int tok = (best_tok < 0) ? 0 : best_tok;
    token_ids_out[ibatch] = tok;

    // Penalty bit writeback
    if (penalty_mask != nullptr && slot_id != nullptr && tok >= 0) {
      float rp = rp_arr ? rp_arr[ibatch] : rp_val;
      if (rp > 0.f) {
        int slot = slot_id[ibatch];
        uint8_t* row = penalty_mask + slot * penalty_row_bytes;
        int byte_idx = tok >> 3;
        int bit_in_byte = tok & 7;
        uintptr_t byte_addr = reinterpret_cast<uintptr_t>(row + byte_idx);
        uintptr_t word_addr = byte_addr & ~uintptr_t{3};
        int byte_off = static_cast<int>(byte_addr - word_addr);
        unsigned int bit_mask = (1u << bit_in_byte) << (byte_off * 8);
        atomicOr(reinterpret_cast<unsigned int*>(word_addr), bit_mask);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Scratch buffer RAII wrapper (cudaMallocAsync / cudaFreeAsync).
// ---------------------------------------------------------------------------
struct ScratchBuf {
  void* ptr = nullptr;
  cudaStream_t stream = nullptr;

  ScratchBuf() = default;
  ScratchBuf(size_t bytes, cudaStream_t s) : stream(s) {
    if (bytes > 0) {
      cudaMallocAsync(&ptr, bytes, stream);
    }
  }
  ~ScratchBuf() {
    if (ptr) {
      cudaFreeAsync(ptr, stream);
    }
  }
  ScratchBuf(const ScratchBuf&) = delete;
  ScratchBuf& operator=(const ScratchBuf&) = delete;

  template <typename T>
  T* as() {
    return reinterpret_cast<T*>(ptr);
  }
};

// ---------------------------------------------------------------------------
// Dynamic block-count: N = clamp(floor(SMs/B), kNMin, Nmax).
//
// K1 is limited to 1 block/SM, so floor(SMs/B) is the largest N keeping B*N ≤ SMs
// (single wave; ceil would spill into a 2nd wave). kNMin floors N so per-block
// work stays ≤ V/kNMin at large batch.
// ---------------------------------------------------------------------------
constexpr int kNMin = 8;

inline int pick_n_blocks_per_row(int batch_size, int sm_count, int n_max) {
  int n = sm_count / batch_size;  // floor(SMs / B)
  if (n < kNMin) {
    n = kNMin;
  }
  if (n > n_max) {
    n = n_max;
  }
  return n;
}

// ---------------------------------------------------------------------------
// Pipeline launcher.
// ---------------------------------------------------------------------------
template <typename DType, int kVocabSize, int kMaxTopK, int kSoftmaxPolicy, bool kHasTopP>
void launch_pipeline(int32_t* token_ids_out, const void* logits_ptr, uint8_t* penalty_mask_ptr,
                     const int32_t* slot_id_ptr, const float* rp_arr, float rp_val,
                     const float* temp_arr, float temp_val, int softmax_policy,
                     const void* topk_ptr, int topk_int_bytes, int topk_val, const float* topp_ptr,
                     float topp_val, const float* gumbel_noise_ptr, int batch_size, int vocab_size,
                     int logits_row_stride, int max_topk, uint64_t rng_seed, cudaStream_t stream) {
  const int penalty_row_bytes = (vocab_size + 7) / 8;

  // Nmax = compile-time cap; N = runtime value ≤ Nmax.
  constexpr int kNmax = kMaxBlocksPerBatch<kMaxTopK>;
  const int n_blocks_per_row = pick_n_blocks_per_row(batch_size, ::hpc::get_sm_count(), kNmax);

  // Scratch sized by Nmax (static strides); K1 writes first N slots, K2 masks
  // the rest. partial_max/sum only needed for BEFORE_TOPK.
  constexpr bool kNeedPartial = (kSoftmaxPolicy == kSoftmaxBeforeTopk);
  ScratchBuf partial_max_buf(kNeedPartial ? sizeof(float) * batch_size * kNmax : 0, stream);
  ScratchBuf partial_sum_buf(kNeedPartial ? sizeof(float) * batch_size * kNmax : 0, stream);
  ScratchBuf mid_logits_buf(sizeof(float) * batch_size * kNmax * kMaxTopK, stream);
  ScratchBuf mid_tokens_buf(sizeof(int) * batch_size * kNmax * kMaxTopK, stream);

  // --- Launch K1 ---
  dim3 grid1(n_blocks_per_row, batch_size);
  fused_scan_topk_kernel<DType, kVocabSize, kMaxTopK, kSoftmaxPolicy>
      <<<grid1, kThreadsPerBlock, 0, stream>>>(
          mid_logits_buf.as<float>(), mid_tokens_buf.as<int>(), partial_max_buf.as<float>(),
          partial_sum_buf.as<float>(), reinterpret_cast<const DType*>(logits_ptr),
          logits_row_stride, penalty_mask_ptr, slot_id_ptr, rp_arr, rp_val, temp_arr, temp_val,
          penalty_row_bytes, n_blocks_per_row, kNmax);

  // --- Launch K2 ---
  constexpr int kThreadsPerBlock2 = kNmax * kMaxTopK;

  // Per-launch Philox offset for fresh samples; external-gumbel path skips it.
  uint64_t rng_base_offset = 0;
  if (gumbel_noise_ptr == nullptr) {
    rng_base_offset = ::hpc::sampler::rng::next_launch_offset();
  }

  stage2_kernel<kMaxTopK, kSoftmaxPolicy, kHasTopP><<<batch_size, kThreadsPerBlock2, 0, stream>>>(
      token_ids_out, mid_logits_buf.as<float>(), mid_tokens_buf.as<int>(),
      partial_max_buf.as<float>(), partial_sum_buf.as<float>(), topk_ptr, topk_int_bytes, topk_val,
      topp_ptr, topp_val, gumbel_noise_ptr, vocab_size, penalty_mask_ptr, slot_id_ptr, rp_arr,
      rp_val, penalty_row_bytes, rng_seed, rng_base_offset, n_blocks_per_row);
}

// ---------------------------------------------------------------------------
// Template dispatch chain.
// ---------------------------------------------------------------------------

template <typename T>
struct type_tag {
  using type = T;
};

}  // namespace fused_sampler_kernels

// ---------------------------------------------------------------------------
// Public entry point (called from entry.cc).
// ---------------------------------------------------------------------------
void fused_sampler_async(int32_t* token_ids_out, const void* logits_ptr, int logits_dtype,
                         uint8_t* penalty_mask_ptr, const int32_t* slot_id_ptr,
                         const float* repetition_penalty_ptr, float repetition_penalty_val,
                         const float* temperature_ptr, float temperature_val, int softmax_policy,
                         const void* topk_ptr, int topk_int_bytes, int topk_val,
                         const float* topp_ptr, float topp_val, const float* gumbel_noise_ptr,
                         int batch_size, int vocab_size, int logits_row_stride, int max_topk,
                         uint64_t rng_seed, cudaStream_t stream) {
  using fused_sampler_kernels::launch_pipeline;
  using fused_sampler_kernels::type_tag;
  namespace kernels = fused_sampler_kernels;

  const bool has_topp = (topp_ptr != nullptr) || (topp_val > 0.f);

  // Tag-based dispatch: each lambda fixes one runtime axis to a compile-time
  // constant, bottoming out in launch_pipeline.
  auto launch = [&](auto dtype_tag, auto vocab_tag, auto topk_tag, auto policy_tag, auto topp_tag) {
    using DType = typename decltype(dtype_tag)::type;
    constexpr int kVocabSize = decltype(vocab_tag)::value;
    constexpr int kMaxTopK = decltype(topk_tag)::value;
    constexpr int kSoftmaxPolicy = decltype(policy_tag)::value;
    constexpr bool kHasTopP = decltype(topp_tag)::value;
    launch_pipeline<DType, kVocabSize, kMaxTopK, kSoftmaxPolicy, kHasTopP>(
        token_ids_out, logits_ptr, penalty_mask_ptr, slot_id_ptr, repetition_penalty_ptr,
        repetition_penalty_val, temperature_ptr, temperature_val, softmax_policy, topk_ptr,
        topk_int_bytes, topk_val, topp_ptr, topp_val, gumbel_noise_ptr, batch_size, vocab_size,
        logits_row_stride, max_topk, rng_seed, stream);
  };

  auto dispatch_topp = [&](auto dtype_tag, auto vocab_tag, auto topk_tag, auto policy_tag) {
    if (has_topp) {
      launch(dtype_tag, vocab_tag, topk_tag, policy_tag, std::true_type{});
    } else {
      launch(dtype_tag, vocab_tag, topk_tag, policy_tag, std::false_type{});
    }
  };

  auto dispatch_softmax = [&](auto dtype_tag, auto vocab_tag, auto topk_tag) {
    switch (softmax_policy) {
      case kernels::kSoftmaxNone:
        dispatch_topp(dtype_tag, vocab_tag, topk_tag,
                      std::integral_constant<int, kernels::kSoftmaxNone>{});
        break;
      case kernels::kSoftmaxBeforeTopk:
        dispatch_topp(dtype_tag, vocab_tag, topk_tag,
                      std::integral_constant<int, kernels::kSoftmaxBeforeTopk>{});
        break;
      case kernels::kSoftmaxAfterTopk:
        dispatch_topp(dtype_tag, vocab_tag, topk_tag,
                      std::integral_constant<int, kernels::kSoftmaxAfterTopk>{});
        break;
      default:
        throw std::runtime_error("fused_sampler: unsupported softmax_policy " +
                                 std::to_string(softmax_policy));
    }
  };

  auto dispatch_max_topk = [&](auto dtype_tag, auto vocab_tag) {
    if (max_topk == 32) {
      dispatch_softmax(dtype_tag, vocab_tag, std::integral_constant<int, 32>{});
    } else {
      dispatch_softmax(dtype_tag, vocab_tag, std::integral_constant<int, 64>{});
    }
  };

  auto dispatch_vocab = [&](auto dtype_tag) {
    switch (vocab_size) {
      case 120832:
        dispatch_max_topk(dtype_tag, std::integral_constant<int, 120832>{});
        break;
      default:
        throw std::runtime_error("fused_sampler: unsupported vocab_size " +
                                 std::to_string(vocab_size));
    }
  };

  if (logits_dtype == 0) {
    dispatch_vocab(type_tag<float>{});
  } else {
    dispatch_vocab(type_tag<__nv_bfloat16>{});
  }
}

}  // namespace sampler
}  // namespace hpc
