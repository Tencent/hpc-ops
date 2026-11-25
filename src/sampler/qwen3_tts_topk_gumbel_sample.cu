// Copyright (C) 2026 Tencent.
//
// Qwen3-TTS short-vocab fused Gumbel-max top-k sampler.
//
// Qwen3-TTS code logits use vocab=2048 and top_k=50.  Reusing the generic
// large-vocab sampler would launch more blocks per row and materialize a masked
// logits tensor that the model never consumes.  This kernel keeps one CUDA block
// per row, materializes only the top-K candidates, then runs Gumbel-max on that
// compact candidate set.

#include <cuda_runtime_api.h>

#include <cub/cub.cuh>  // NOLINT
#include <limits>

#include "src/sampler/sampler.h"

namespace hpc {
namespace sampler {
namespace qwen3_tts_topk_gumbel_sample_kernels {

constexpr int kWarpSize = 32;
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

template <int kThreadsPerBlock, int kMaxTopK, int kFixedTopK = 0>
__global__ void qwen3_tts_topk_gumbel_argmax_kernel(int64_t* out, const float* logits,
                                                    const float* noise, int vocab_size,
                                                    int logits_row_stride, int topk,
                                                    float inv_temperature) {
  static_assert(kMaxTopK <= 64, "kMaxTopK kept small to stay in registers");
  static_assert(kThreadsPerBlock >= 64 && kThreadsPerBlock % kWarpSize == 0, "");

  using KV = cub::KeyValuePair<int, float>;
  using BlockReduce = cub::BlockReduce<KV, kThreadsPerBlock>;
  __shared__ typename BlockReduce::TempStorage reduce_temp;

  constexpr int kNumWarps = kThreadsPerBlock / kWarpSize;
  __shared__ float warp_topk[kNumWarps * kMaxTopK];
  __shared__ int warp_topk_token[kNumWarps * kMaxTopK];
  __shared__ float topk_logits[kMaxTopK];
  __shared__ int topk_tokens[kMaxTopK];

  const int ibatch = blockIdx.x;
  const int tid = threadIdx.x;
  const int iwarp = tid / kWarpSize;
  const int ilane = tid % kWarpSize;
  const float* logits_row = logits + static_cast<int64_t>(ibatch) * logits_row_stride;
  const float* noise_row = noise + static_cast<int64_t>(ibatch) * vocab_size;

  int actual_topk;
  if constexpr (kFixedTopK > 0) {
    actual_topk = kFixedTopK;
  } else {
    actual_topk = topk;
    if (actual_topk > kMaxTopK) actual_topk = kMaxTopK;
    if (actual_topk > vocab_size) actual_topk = vocab_size;
  }

  // === [perf sampler-localtopk 2026-06-25] shrink per-thread register list ===
  // For Qwen3-TTS each thread scans at most ceil(2048/256)=8 tokens.  Keep only
  // the maximum number of elements a thread can own locally, then merge globally.
  constexpr int kMaxItemsPerThread = (2048 + kThreadsPerBlock - 1) / kThreadsPerBlock;
  float per_thread_topk[kMaxItemsPerThread];
  int per_thread_token[kMaxItemsPerThread];
#pragma unroll
  for (int i = 0; i < kMaxItemsPerThread; ++i) {
    per_thread_topk[i] = kNegInf;
    per_thread_token[i] = -1;
  }

  for (int token = tid; token < vocab_size; token += kThreadsPerBlock) {
    const float val = logits_row[token];
    if (val <= per_thread_topk[kMaxItemsPerThread - 1]) continue;
    int j = kMaxItemsPerThread - 1;
    while (j > 0 && val > per_thread_topk[j - 1]) {
      per_thread_topk[j] = per_thread_topk[j - 1];
      per_thread_token[j] = per_thread_token[j - 1];
      --j;
    }
    per_thread_topk[j] = val;
    per_thread_token[j] = token;
  }

  // Warp-level merge to get each warp's top-K logits. Equal logits are popped by
  // the lowest owning lane so only one lane mutates its local list per round.
  for (int k = 0; k < actual_topk; ++k) {
    const float candidate = per_thread_topk[0];
    float best = candidate;
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
      const float other = __shfl_xor_sync(0xffffffff, best, offset);
      if (other > best) best = other;
    }
    const unsigned mask = __ballot_sync(0xffffffff, candidate == best);
    const int popper = __ffs(mask) - 1;
    const int candidate_token = per_thread_token[0];
    const int best_token = __shfl_sync(0xffffffff, candidate_token, popper);
    if (ilane == 0) {
      warp_topk[iwarp * kMaxTopK + k] = best;
      warp_topk_token[iwarp * kMaxTopK + k] = best_token;
    }
    if (ilane == popper) {
#pragma unroll
      for (int i = 0; i < kMaxItemsPerThread - 1; ++i) {
        per_thread_topk[i] = per_thread_topk[i + 1];
        per_thread_token[i] = per_thread_token[i + 1];
      }
      per_thread_topk[kMaxItemsPerThread - 1] = kNegInf;
      per_thread_token[kMaxItemsPerThread - 1] = -1;
    }
    __syncwarp();
  }
  __syncthreads();

  // CTA-level merge to materialize the exact top-K candidate token ids.
  if (tid == 0) {
    int idx_per_warp[kNumWarps];
#pragma unroll
    for (int w = 0; w < kNumWarps; ++w) idx_per_warp[w] = 0;
    for (int k = 0; k < actual_topk; ++k) {
      float best = kNegInf;
      int best_w = 0;
#pragma unroll
      for (int w = 0; w < kNumWarps; ++w) {
        if (idx_per_warp[w] < actual_topk) {
          const float v = warp_topk[w * kMaxTopK + idx_per_warp[w]];
          if (v > best) {
            best = v;
            best_w = w;
          }
        }
      }
      topk_logits[k] = best;
      topk_tokens[k] = warp_topk_token[best_w * kMaxTopK + idx_per_warp[best_w]];
      idx_per_warp[best_w]++;
    }
  }
  __syncthreads();

  // Fused Gumbel-max over the compact top-K candidate set.
  KV thread_best;
  thread_best.key = 0;
  thread_best.value = kNegInf;
  if (tid < actual_topk) {
    const int token = topk_tokens[tid];
    const float val = topk_logits[tid];
    const float u = fminf(fmaxf(noise_row[token], 1.0e-20f), 1.0f - 1.0e-20f);
    thread_best.value = val * inv_temperature - logf(-logf(u));
    thread_best.key = token;
  }
  const KV block_best = BlockReduce(reduce_temp).Reduce(thread_best, cub::ArgMax());

  if (tid == 0) {
    out[ibatch] = static_cast<int64_t>(block_best.key);
  }
}

}  // namespace qwen3_tts_topk_gumbel_sample_kernels

void qwen3_tts_topk_gumbel_sample_async(int64_t* token_ids_out, const float* logits_ptr,
                                        const float* noise_ptr, int batch_size, int vocab_size,
                                        int logits_row_stride, int topk, float inv_temperature,
                                        cudaStream_t stream) {
  constexpr int kThreadsPerBlock = 256;
  dim3 grid(batch_size);
  dim3 block(kThreadsPerBlock);
  if (topk == 50 && vocab_size <= 2048) {
    // === [perf sampler-topk50 2026-06-25] production-specialized path ===
    // Compile top_k as a constant and shrink kMaxTopK 64->50.  The generic path
    // remains available for tests and non-production top_k values.
    constexpr int kMaxTopK = 50;
    constexpr int kFixedTopK = 50;
    qwen3_tts_topk_gumbel_sample_kernels::qwen3_tts_topk_gumbel_argmax_kernel<
        kThreadsPerBlock, kMaxTopK, kFixedTopK>
        <<<grid, block, 0, stream>>>(token_ids_out, logits_ptr, noise_ptr, vocab_size,
                                     logits_row_stride, topk, inv_temperature);
  } else {
    constexpr int kMaxTopK = 64;
    qwen3_tts_topk_gumbel_sample_kernels::qwen3_tts_topk_gumbel_argmax_kernel<
        kThreadsPerBlock, kMaxTopK>
        <<<grid, block, 0, stream>>>(token_ids_out, logits_ptr, noise_ptr, vocab_size,
                                     logits_row_stride, topk, inv_temperature);
  }
}

}  // namespace sampler
}  // namespace hpc
