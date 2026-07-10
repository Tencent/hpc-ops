// Copyright (C) 2026 Tencent.
#ifndef SRC_SAMPLER_SAMPLER_H_
#define SRC_SAMPLER_SAMPLER_H_

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace hpc {
namespace sampler {

// fused_sampler: 2-kernel pipeline — rp / temperature / [softmax1] / topk /
// [softmax2] / topp / Gumbel-max sampling + penalty_mask writeback.
//   logits_dtype: 0 = fp32, 1 = bf16
//   softmax_policy: 0 = NONE, 1 = BEFORE_TOPK, 2 = AFTER_TOPK
// This is a top-`max_topk` (32/64) bounded sampler: the kernel always collapses
// the vocab to the top-`max_topk` candidates before sampling. topk=0 (or unset)
// does NOT mean full-vocab sampling — it only means "do not tighten below
// max_topk". Low-probability tokens outside the top-`max_topk` are never drawn.
void fused_sampler_async(int32_t* token_ids_out, const void* logits_ptr, int logits_dtype,
                         uint8_t* penalty_mask_ptr, const int32_t* slot_id_ptr,
                         const float* repetition_penalty_ptr, float repetition_penalty_val,
                         const float* temperature_ptr, float temperature_val, int softmax_policy,
                         const void* topk_ptr, int topk_int_bytes, int topk_val,
                         const float* topp_ptr, float topp_val, const float* gumbel_noise_ptr,
                         int batch_size, int vocab_size, int logits_row_stride, int max_topk,
                         uint64_t rng_seed, cudaStream_t stream);

// fused_sampler_temperature: temperature-only fast-path. token = argmax_v
// (logit/temperature + Gumbel(0)).
//   logits_dtype: 0 = fp32, 1 = bf16
//   gumbel_noise_ptr: nullable [B, V] fp32. If set, read noise from it
//     (bit-exact tests); else draw via curand.
//   draft_token_ids_ptr: nullable [B] int64 speculative-decoding mask. b's
//     entry != -1 → logits[b, that token] treated as -inf (tensor unmodified);
//     -1 → unmasked. nullptr → zero-overhead no-mask specialization.
// Concurrency: not safe to call concurrently on multiple streams of the same
// device (the per-device scratch workspace is shared across streams).
void fused_sampler_temperature_async(int32_t* token_ids_out, const void* logits_ptr,
                                     int logits_dtype, int logits_row_stride,
                                     const float* temperature_arr, float temperature_val,
                                     const float* gumbel_noise_ptr,
                                     const int64_t* draft_token_ids_ptr, int batch_size,
                                     int vocab_size, uint64_t rng_seed, cudaStream_t stream);

void qwen3_tts_topk_gumbel_sample_async(int64_t* token_ids_out, const float* logits_ptr,
                                        const float* noise_ptr, int batch_size, int vocab_size,
                                        int logits_row_stride, int topk, float inv_temperature,
                                        cudaStream_t stream);

}  // namespace sampler
}  // namespace hpc

#endif  // SRC_SAMPLER_SAMPLER_H_
