// Copyright (C) 2026 Tencent.
//
// Temperature-only fast-path sampler:
//   token_id[b] = argmax_v ( logits[b, v] / temperature[b] + Gumbel(0)_{b,v} )
//
// Fast path of fused_sampler.cu for the "only temperature" case (skips the
// topk/softmax/penalty machinery).
//
// Geometry: grid=(N, B), block=256. Each block scans a V/N vocab slice, reduces
// to one (score, tok) partial in scratch[ibatch*stride+bid], then atomicAdds
// counter[ibatch]; the last block (counter == N-1) reduces N partials → token.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <stdint.h>

#include <algorithm>
#include <atomic>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "src/sampler/sampler.h"
#include "src/sampler/sampler_rng.cuh"
#include "src/utils/utils.cuh"
#include "src/utils/utils.h"

namespace hpc {
namespace sampler {
namespace fused_sampler_temperature {
namespace kernels {

// Local analogue of `load_as_float` in fused_sampler.cu (own copy per TU).
template <typename DType, int N>
__device__ __forceinline__ void load_as_float_local(const DType* gmem, float* out) {
  if constexpr (std::is_same_v<DType, float>) {
    vec_t<float, N> v = load<float, N>(gmem);
#pragma unroll
    for (int i = 0; i < N; i++) {
      out[i] = v[i];
    }
  } else {
    vec_t<__nv_bfloat16, N> v = load<__nv_bfloat16, N>(gmem);
#pragma unroll
    for (int i = 0; i < N; i++) {
      out[i] = __bfloat162float(v[i]);
    }
  }
}

// Tie-break: larger score wins; on a tie the smaller non-negative token id.
// -1 is the empty-slot sentinel and never beats a real token.
__device__ __forceinline__ bool take_other(float other_score, int other_tok, float my_score,
                                           int my_tok) {
  if (other_score > my_score) {
    return true;
  }
  if (other_score < my_score) {
    return false;
  }
  // Equal scores.
  if (other_tok < 0) {
    return false;
  }
  if (my_tok < 0) {
    return true;
  }
  return other_tok < my_tok;
}

// ============================================================================
// Kernel. Each block scans a V/N slice of one row → (best_score, best_tok) into
// scratch[]; the last block to arrive (counter == N-1) reduces all N partials.
// Draft mask: at most one masked token per row; draft_token_ids[b] == -1 means
// unmasked, else that vocab id's post-temperature logit is treated as -inf.
// ============================================================================

template <typename DType, int kVocabSize, int kThreadsPerBlock, bool kHasExternalGumbel,
          bool kHasDraftMask>
__global__ __launch_bounds__(kThreadsPerBlock, 2) void fused_sampler_temperature_kernel(
    int32_t* token_ids_out,          // [B, 1]
    const DType* logits,             // base pointer, row stride = logits_row_stride
    int logits_row_stride,           // >= kVocabSize
    const float* temperature_arr,    // nullable [B]
    float temperature_val,           //
    const float* gumbel_noise,       // nullable [B, V]
    const int64_t* draft_token_ids,  // nullable [B], -1 = no mask, else token id
    float* scratch_score,            // [B * scratch_stride]
    int32_t* scratch_tok,            // [B * scratch_stride]
    int32_t* counter,                // [B], cache invariant: zero on entry
    int n_blocks_per_row,            // dynamic N from launcher (∈ [kNMin, scratch_stride])
    int scratch_stride,              // row stride for scratch (= max N supported = SM count)
    uint64_t rng_seed,               //
    uint64_t rng_base_offset) {      // per-launch base offset from host counter
  constexpr int kWarpSize = 32;
  constexpr int kWarpCount = kThreadsPerBlock / kWarpSize;
  constexpr float kNegInf = -std::numeric_limits<float>::infinity();

  const int bid = static_cast<int>(blockIdx.x);  // 0..N-1
  const int ibatch = static_cast<int>(blockIdx.y);
  const int tid = static_cast<int>(threadIdx.x);
  const int iwarp = tid / kWarpSize;
  const int ilane = tid % kWarpSize;

  // Per-block vocab slice. Round cols_per_block UP to a multiple of
  // kItemsPerThread (4) so each thread's 16B vec load stays aligned.
  constexpr int kItemsPerThread = 4;  // 16B vec
  const int cols_raw = (kVocabSize + n_blocks_per_row - 1) / n_blocks_per_row;
  const int cols_per_block = (cols_raw + kItemsPerThread - 1) / kItemsPerThread * kItemsPerThread;
  const int col_lo = bid * cols_per_block;
  const int col_hi = min(col_lo + cols_per_block, kVocabSize);

  // ---- Resolve per-batch temperature.
  const float t = temperature_arr ? temperature_arr[ibatch] : temperature_val;

  // ---- RNG init (self-drawn noise only). Unique cuRAND `sequence` per
  // (row, block, thread); rng_base_offset advances per launch.
  curandStatePhilox4_32_10_t rng_state;
  if constexpr (!kHasExternalGumbel) {
    uint64_t sequence =
        (static_cast<uint64_t>(ibatch) * scratch_stride + bid) * kThreadsPerBlock + tid;
    curand_init(rng_seed, sequence, rng_base_offset, &rng_state);
  }

  const DType* row = logits + static_cast<int64_t>(ibatch) * logits_row_stride;
  const float* gumbel_row =
      kHasExternalGumbel ? (gumbel_noise + static_cast<int64_t>(ibatch) * kVocabSize) : nullptr;

  // ---- Resolve per-row draft mask (at most one masked token per row).
  __shared__ int32_t s_mask_tok;
  if constexpr (kHasDraftMask) {
    if (tid == 0) {
      int64_t tok = draft_token_ids[ibatch];
      // -1 / out-of-range → no mask (kVocabSize is unreachable by `col`).
      s_mask_tok = (tok >= 0 && tok < static_cast<int64_t>(kVocabSize))
                       ? static_cast<int32_t>(tok)
                       : static_cast<int32_t>(kVocabSize);
    }
    __syncthreads();
  }
  const int32_t reg_mask = kHasDraftMask ? s_mask_tok : static_cast<int32_t>(kVocabSize);

  // ---- Phase 1: streaming scan over [col_lo, col_hi), per-thread argmax.
  float best_score = kNegInf;
  int best_tok = -1;

  constexpr int kStride = kThreadsPerBlock * kItemsPerThread;  // 1024 items/iter
  // Runtime-bounded iters; `unroll 1` keeps register usage low and stable.
  const int base0 = col_lo + tid * kItemsPerThread;
  const int iters = (cols_per_block + kStride - 1) / kStride;

#pragma unroll 1
  for (int it = 0; it < iters; it++) {
    int col_base = base0 + it * kStride;
    if (col_base >= col_hi) {
      break;
    }

    float tmp[kItemsPerThread];
    const bool full_in_bounds = (col_base + kItemsPerThread) <= col_hi;
    if (full_in_bounds) {
      load_as_float_local<DType, kItemsPerThread>(row + col_base, tmp);
    } else {
#pragma unroll
      for (int i = 0; i < kItemsPerThread; i++) {
        int col = col_base + i;
        if (col < col_hi) {
          if constexpr (std::is_same_v<DType, float>) {
            tmp[i] = row[col];
          } else {
            tmp[i] = __bfloat162float(row[col]);
          }
        } else {
          tmp[i] = kNegInf;
        }
      }
    }

    // Draw 4 uniforms per thread via cuRAND Philox.
    float u_vec[kItemsPerThread];
    if constexpr (!kHasExternalGumbel) {
      float4 r = curand_uniform4(&rng_state);
      u_vec[0] = r.x;
      u_vec[1] = r.y;
      u_vec[2] = r.z;
      u_vec[3] = r.w;
    }

#pragma unroll
    for (int i = 0; i < kItemsPerThread; i++) {
      int col = col_base + i;
      if (col >= col_hi) {
        continue;
      }
      float x = tmp[i] / t;
      if constexpr (kHasDraftMask) {
        if (col == reg_mask) {
          x = kNegInf;
        }
      }
      float g;
      if constexpr (kHasExternalGumbel) {
        g = gumbel_row[col];
      } else {
        g = ::hpc::sampler::rng::gumbel_noise_from_uniform(u_vec[i]);
      }
      float score = x + g;
      if (take_other(score, col, best_score, best_tok)) {
        best_score = score;
        best_tok = col;
      }
    }
  }

  // ---- Phase 2: warp argmax butterfly on (score, tok) pair.
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    float o_s = __shfl_xor_sync(0xffffffff, best_score, offset);
    int o_t = __shfl_xor_sync(0xffffffff, best_tok, offset);
    if (take_other(o_s, o_t, best_score, best_tok)) {
      best_score = o_s;
      best_tok = o_t;
    }
  }

  // ---- Phase 3: block argmax — warp leaders stage to SMEM, warp 0 reduces
  //              into (block_score, block_tok) (held by lane 0 of warp 0).
  __shared__ float s_key[kWarpCount];
  __shared__ int s_tok[kWarpCount];
  if (ilane == 0) {
    s_key[iwarp] = best_score;
    s_tok[iwarp] = best_tok;
  }
  __syncthreads();

  __shared__ float s_block_score;
  __shared__ int s_block_tok;
  if (iwarp == 0) {
    float k;
    int v;
    if (ilane < kWarpCount) {
      k = s_key[ilane];
      v = s_tok[ilane];
    } else {
      k = kNegInf;
      v = -1;
    }
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
      float o_s = __shfl_xor_sync(0xffffffff, k, offset);
      int o_t = __shfl_xor_sync(0xffffffff, v, offset);
      if (take_other(o_s, o_t, k, v)) {
        k = o_s;
        v = o_t;
      }
    }
    if (ilane == 0) {
      s_block_score = k;
      s_block_tok = v;
    }
  }
  __syncthreads();

  // ---- Phase 4: publish partial to scratch, atomicAdd counter to find the
  //              last block.
  __shared__ int s_prev;
  if (tid == 0) {
    scratch_score[ibatch * scratch_stride + bid] = s_block_score;
    scratch_tok[ibatch * scratch_stride + bid] = s_block_tok;
    __threadfence();                          // release scratch before counter increment
    s_prev = atomicAdd(&counter[ibatch], 1);  // 0..N-1; N-1 is the last block
  }
  __syncthreads();
  if (s_prev != n_blocks_per_row - 1) {
    return;
  }

  // ---- Phase 5: last block reduces N partials → final token.
  __threadfence();  // observe other blocks' scratch writes

  float my_s = kNegInf;
  int my_t = -1;
  if (tid < n_blocks_per_row) {  // n_blocks_per_row ≤ scratch_stride ≤ kThreadsPerBlock
    my_s = scratch_score[ibatch * scratch_stride + tid];
    my_t = scratch_tok[ibatch * scratch_stride + tid];
  }

  // Warp butterfly.
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    float o_s = __shfl_xor_sync(0xffffffff, my_s, offset);
    int o_t = __shfl_xor_sync(0xffffffff, my_t, offset);
    if (take_other(o_s, o_t, my_s, my_t)) {
      my_s = o_s;
      my_t = o_t;
    }
  }
  // Cross-warp via SMEM (reuse s_key / s_tok slots).
  if (ilane == 0) {
    s_key[iwarp] = my_s;
    s_tok[iwarp] = my_t;
  }
  __syncthreads();
  if (iwarp == 0) {
    float k = (ilane < kWarpCount) ? s_key[ilane] : kNegInf;
    int v = (ilane < kWarpCount) ? s_tok[ilane] : -1;
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
      float o_s = __shfl_xor_sync(0xffffffff, k, offset);
      int o_t = __shfl_xor_sync(0xffffffff, v, offset);
      if (take_other(o_s, o_t, k, v)) {
        k = o_s;
        v = o_t;
      }
    }
    if (ilane == 0) {
      int tok = (v >= 0) ? v : 0;  // degenerate fallback
      token_ids_out[ibatch] = tok;
      counter[ibatch] = 0;  // self-reset: keeps counter==0 invariant for next launch
    }
  }
}

}  // namespace kernels

// ============================================================================
// Launcher: runtime dispatch over (dtype, vocab_size, kHasExternalGumbel,
// kHasDraftMask).
// ============================================================================
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kItemsPerThread = 4;                           // matches kernel-side vec width
constexpr int kStride = kThreadsPerBlock * kItemsPerThread;  // cols swept per iter
constexpr int kVocabSizeMax = 120832;                        // largest supported vocab
// kNMin: smallest N keeping per-block iters ≤ kMaxItersPerBlock (the cap with
// 0 register spill on the cluster prototype).
constexpr int kMaxItersPerBlock = 16;
constexpr int kNMin =
    (kVocabSizeMax + kStride * kMaxItersPerBlock - 1) / (kStride * kMaxItersPerBlock);

// Shared per-launch Philox advance (sampler_rng.cuh). Assert it covers a
// thread's max draws per launch.
static_assert(::hpc::sampler::rng::kPerLaunchOffsetIncrement >=
                  static_cast<uint64_t>(kMaxItersPerBlock * kItemsPerThread),
              "Philox offset advance must cover all per-thread draws per launch");

// Per-device workspace, lazily grown 2×. Counter is zero-initialized on
// (re)alloc; the kernel maintains counter[*] == 0 across launches via per-row
// self-reset, so no host memset is needed. scratch_score/tok are
// write-before-read each launch. n_max_per_row is captured at first acquire
// (= get_sm_count()) so the device stride matches pick_n_blocks_per_row's cap.
//
// CONCURRENCY CONSTRAINT: this workspace (scratch + counter) is shared per
// device across all streams. The counter==0 invariant and the last-block
// reduction only hold under single-stream serial use. Concurrent invocations of
// this sampler on multiple streams of the SAME device are NOT supported — they
// would race on the shared scratch/counter. Use one stream per device, or
// serialize calls.
struct WorkspaceCache {
  std::mutex mu;
  float* scratch_score = nullptr;  // [max_B * n_max_per_row]
  int32_t* scratch_tok = nullptr;  // [max_B * n_max_per_row]
  int32_t* counter = nullptr;      // [max_B]
  int max_B = 0;
  int n_max_per_row = 0;
  int device_id = -1;
};

constexpr int kMaxDevices = 16;
static WorkspaceCache g_ws[kMaxDevices];

// Acquire the workspace for `device_id`, growing it if needed to hold
// `batch_size` rows at the requested `n_max_per_row` stride. Returned
// pointers are valid until the next grow on the same device.
WorkspaceCache* acquire_workspace(int device_id, int batch_size, int n_max_per_row,
                                  cudaStream_t stream) {
  if (device_id < 0 || device_id >= kMaxDevices) {
    throw std::runtime_error("fused_sampler_temperature: device_id out of range: " +
                             std::to_string(device_id));
  }
  WorkspaceCache& c = g_ws[device_id];
  std::lock_guard<std::mutex> lk(c.mu);
  if (c.scratch_score != nullptr && batch_size <= c.max_B && n_max_per_row == c.n_max_per_row) {
    return &c;
  }
  int new_max = batch_size;
  if (c.max_B * 2 > new_max) {
    new_max = c.max_B * 2;
  }

  if (c.scratch_score) {
    cudaStreamSynchronize(stream);
    cudaFree(c.scratch_score);
    cudaFree(c.scratch_tok);
    cudaFree(c.counter);
  }

  cudaError_t err;
  err = cudaMalloc(&c.scratch_score, static_cast<size_t>(new_max) * n_max_per_row * sizeof(float));
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("fused_sampler_temperature: cudaMalloc(scratch_score) "
                                         "failed: ") +
                             cudaGetErrorString(err));
  }
  err = cudaMalloc(&c.scratch_tok, static_cast<size_t>(new_max) * n_max_per_row * sizeof(int32_t));
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("fused_sampler_temperature: cudaMalloc(scratch_tok) failed: ") +
        cudaGetErrorString(err));
  }
  err = cudaMalloc(&c.counter, static_cast<size_t>(new_max) * sizeof(int32_t));
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("fused_sampler_temperature: cudaMalloc(counter) failed: ") +
        cudaGetErrorString(err));
  }
  // Establish counter[*] == 0 invariant; kernel maintains it from here on.
  err = cudaMemsetAsync(c.counter, 0, static_cast<size_t>(new_max) * sizeof(int32_t), stream);
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("fused_sampler_temperature: cudaMemsetAsync(counter) failed: ") +
        cudaGetErrorString(err));
  }
  c.max_B = new_max;
  c.n_max_per_row = n_max_per_row;
  c.device_id = device_id;
  return &c;
}

// Pick N blocks per row: ceil(SMs / B) clamped to [kNMin, n_max_per_row].
inline int pick_n_blocks_per_row(int batch_size, int n_max_per_row) {
  int n = (n_max_per_row + batch_size - 1) / batch_size;
  if (n < kNMin) {
    n = kNMin;
  }
  if (n > n_max_per_row) {
    n = n_max_per_row;
  }
  return n;
}

template <typename DType, int kVocabSize, bool kHasExternalGumbel, bool kHasDraftMask>
void launch_kernel(int32_t* token_ids_out, const void* logits_ptr, int logits_row_stride,
                   const float* temperature_arr, float temperature_val, const float* gumbel_noise,
                   const int64_t* draft_token_ids, float* scratch_score, int32_t* scratch_tok,
                   int32_t* counter, int batch_size, int n_blocks_per_row, int scratch_stride,
                   uint64_t rng_seed, uint64_t rng_base_offset, cudaStream_t stream) {
  auto kernel = kernels::fused_sampler_temperature_kernel<DType, kVocabSize, kThreadsPerBlock,
                                                          kHasExternalGumbel, kHasDraftMask>;

  // No per-launch counter memset: the counter==0 invariant is maintained by the
  // kernel's per-row self-reset (established once by acquire_workspace).

  dim3 grid(n_blocks_per_row, batch_size);
  dim3 block(kThreadsPerBlock);
  kernel<<<grid, block, 0, stream>>>(
      token_ids_out, reinterpret_cast<const DType*>(logits_ptr), logits_row_stride, temperature_arr,
      temperature_val, gumbel_noise, draft_token_ids, scratch_score, scratch_tok, counter,
      n_blocks_per_row, scratch_stride, rng_seed, rng_base_offset);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("fused_sampler_temperature: kernel launch failed: ") +
                             cudaGetErrorString(err));
  }
}

template <typename T>
struct type_tag {
  using type = T;
};

}  // namespace

void fused_sampler_temperature_async(int32_t* token_ids_out, const void* logits_ptr,
                                     int logits_dtype, int logits_row_stride,
                                     const float* temperature_arr, float temperature_val,
                                     const float* gumbel_noise_ptr,
                                     const int64_t* draft_token_ids_ptr, int batch_size,
                                     int vocab_size, uint64_t rng_seed, cudaStream_t stream) {
  const bool has_external_gumbel = (gumbel_noise_ptr != nullptr);
  const bool has_draft_mask = (draft_token_ids_ptr != nullptr);

  // Advance the global counter to obtain a unique RNG offset for this launch.
  uint64_t rng_base_offset = 0;
  if (!has_external_gumbel) {
    rng_base_offset = ::hpc::sampler::rng::next_launch_offset();
  }

  // n_max_per_row = SM count, also the upper clamp on N. The last-block reduce
  // loads N partials with `if (tid < N)`, so require N ≤ kThreadsPerBlock.
  const int n_max_per_row = ::hpc::get_sm_count();
  if (n_max_per_row > kThreadsPerBlock) {
    throw std::runtime_error("fused_sampler_temperature: device SM count exceeds kThreadsPerBlock");
  }
  const int n_blocks_per_row = pick_n_blocks_per_row(batch_size, n_max_per_row);

  int device_id = -1;
  if (cudaGetDevice(&device_id) != cudaSuccess) {
    device_id = 0;
  }
  WorkspaceCache* ws = acquire_workspace(device_id, batch_size, n_max_per_row, stream);
  float* scratch_score = ws->scratch_score;
  int32_t* scratch_tok = ws->scratch_tok;
  int32_t* counter = ws->counter;

  auto launch = [&](auto dtype_tag, auto vocab_tag, auto gumbel_tag, auto mask_tag) {
    using DType = typename decltype(dtype_tag)::type;
    constexpr int kVocabSize = decltype(vocab_tag)::value;
    constexpr bool kHasExternalGumbel = decltype(gumbel_tag)::value;
    constexpr bool kHasDraftMask = decltype(mask_tag)::value;
    launch_kernel<DType, kVocabSize, kHasExternalGumbel, kHasDraftMask>(
        token_ids_out, logits_ptr, logits_row_stride, temperature_arr, temperature_val,
        gumbel_noise_ptr, draft_token_ids_ptr, scratch_score, scratch_tok, counter, batch_size,
        n_blocks_per_row, n_max_per_row, rng_seed, rng_base_offset, stream);
  };

  auto dispatch_mask = [&](auto dtype_tag, auto vocab_tag, auto gumbel_tag) {
    if (has_draft_mask) {
      launch(dtype_tag, vocab_tag, gumbel_tag, std::true_type{});
    } else {
      launch(dtype_tag, vocab_tag, gumbel_tag, std::false_type{});
    }
  };

  auto dispatch_gumbel = [&](auto dtype_tag, auto vocab_tag) {
    if (has_external_gumbel) {
      dispatch_mask(dtype_tag, vocab_tag, std::true_type{});
    } else {
      dispatch_mask(dtype_tag, vocab_tag, std::false_type{});
    }
  };

  auto dispatch_vocab = [&](auto dtype_tag) {
    switch (vocab_size) {
      case 120832:
        dispatch_gumbel(dtype_tag, std::integral_constant<int, 120832>{});
        break;
      default:
        throw std::invalid_argument("fused_sampler_temperature: unsupported vocab_size=" +
                                    std::to_string(vocab_size) + " (supported: {120832})");
    }
  };

  if (logits_dtype == 0) {
    dispatch_vocab(type_tag<float>{});
  } else if (logits_dtype == 1) {
    dispatch_vocab(type_tag<__nv_bfloat16>{});
  } else {
    throw std::invalid_argument(
        "fused_sampler_temperature: logits_dtype must be 0 (fp32) or 1 (bf16)");
  }
}

}  // namespace fused_sampler_temperature

// Public entry for the header — forwards to the detail namespace.
void fused_sampler_temperature_async(int32_t* token_ids_out, const void* logits_ptr,
                                     int logits_dtype, int logits_row_stride,
                                     const float* temperature_arr, float temperature_val,
                                     const float* gumbel_noise_ptr,
                                     const int64_t* draft_token_ids_ptr, int batch_size,
                                     int vocab_size, uint64_t rng_seed, cudaStream_t stream) {
  fused_sampler_temperature::fused_sampler_temperature_async(
      token_ids_out, logits_ptr, logits_dtype, logits_row_stride, temperature_arr, temperature_val,
      gumbel_noise_ptr, draft_token_ids_ptr, batch_size, vocab_size, rng_seed, stream);
}

}  // namespace sampler
}  // namespace hpc
