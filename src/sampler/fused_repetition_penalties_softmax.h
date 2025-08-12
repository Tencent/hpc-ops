#ifndef SRC_SAMPLER_FUSED_REPETITION_PENALTIES_SOFTMAX_H_
#define SRC_SAMPLER_FUSED_REPETITION_PENALTIES_SOFTMAX_H_

#include <stdint.h> 
#include <cuda_runtime_api.h>

namespace hpc {
namespace sampler {

void fused_repetition_penalties_softmax_async(
    float* out_ptr, const float* logits_ptr, const uint8_t** penalties_masks_ptrs,
    const float* repetition_penalties, float repetition_penalties_val,
    const float* temperature, float temperature_val, 
    const int num_batch, const int vocab_size,
    cudaStream_t stream);

}  // namespace sampler
}  // namespace hpc

#endif  // SRC_SAMPLER_FUSED_REPETITION_PENALTIES_SOFTMAX_H_
