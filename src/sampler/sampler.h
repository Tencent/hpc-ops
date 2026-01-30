// Copyright 2025 hpc-ops authors
#ifndef SRC_SAMPLER_SAMPLER_H_
#define SRC_SAMPLER_SAMPLER_H_

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace hpc {
namespace sampler {

void fused_repetition_penalties_softmax_async(
    float* out_ptr, const float* logits_ptr, const uint8_t** penalties_masks_ptrs,
    const float* repetition_penalties, float repetition_penalties_val, const float* temperature,
    float temperature_val, const int num_batch, const int vocab_size, cudaStream_t stream);

void topk_topp_mask_logits_async(void* output_logits, void* out, void* middle_logits,
                                 void* middle_tokens, void* logits, void* topk, int topk_val,
                                 void* topp, float topp_val, void* reject_threshold,
                                 float reject_threshold_val, int batch_size, int vocab_size,
                                 int vocab_size_padded, int int_bytes, cudaStream_t stream);

void argmax_async(void* out, void* logits, int batch_size, int vocab_size, cudaStream_t stream);

}  // namespace sampler
}  // namespace hpc

#endif  // SRC_SAMPLER_SAMPLER_H_
