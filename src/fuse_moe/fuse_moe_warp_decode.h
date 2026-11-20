// Copyright (C) 2026 Tencent.
// Adapted from Religious-J/warp_decode, commit 1b3fff273d00d4cad4c9f366b0a74dbc80ee2e86.

#ifndef SRC_FUSE_MOE_FUSE_MOE_WARP_DECODE_H_
#define SRC_FUSE_MOE_FUSE_MOE_WARP_DECODE_H_

#include <cuda_runtime_api.h>

namespace hpc {
namespace fuse_moe {
namespace warp_decode {

// FP8 weights are [E,N,K]; FP32 weight scales are [E,N/128,ceil(K/128/4)*4].
// Activation scales are compact [rows,K/128]. Global expert IDs are local only
// in [expert_start, expert_start + num_experts); all other slots contribute zero.
void quantize_async(const void* input, void* out, float* scale, int rows, int cols,
                    cudaStream_t stream);

void gate_up_async(const void* input, const void* weight, const float* input_scale,
                   const float* weight_scale, const int* topk_ids, void* out, int batch_size,
                   int hidden, int inter, int num_experts, int topk, int expert_start,
                   cudaStream_t stream);

void down_async(const void* buffer, const void* weight, const float* buffer_scale,
                const float* weight_scale, const int* topk_ids, const float* topk_scale,
                const void* shared_output, void* out, int batch_size, int hidden, int inter,
                int num_experts, int topk, int expert_start, cudaStream_t stream);

// Inline BF16 activation quantization; selected only for inter=128 or 256.
void down_bf16_async(const void* buffer, const void* weight, const float* weight_scale,
                     const int* topk_ids, const float* topk_scale, const void* shared_output,
                     void* out, int batch_size, int hidden, int inter, int num_experts, int topk,
                     int expert_start, cudaStream_t stream);

// Hunyuan-v3 TP8 per-tensor path: H=4096, I=192 and top-k=8. The first
// kernel writes FP8 [batch*8,192] directly, so no intermediate quantizer is
// needed. Projection scales contain one multiplier per local expert;
// act_and_mul_scale contains a single multiplier before FP8 conversion.
void gate_up_pertensor_async(const void* input, const void* weight, const float* gate_up_scale,
                             const float* act_and_mul_scale, const int* ids, void* buffer,
                             int batch_size, int num_experts, int expert_start, bool use_bf16_mul,
                             cudaStream_t stream);

void down_pertensor_async(const void* buffer, const void* weight, const float* down_scale,
                          const int* ids, const float* topk_scale, const void* shared_output,
                          void* out, int batch_size, int num_experts, int expert_start,
                          cudaStream_t stream);

}  // namespace warp_decode
}  // namespace fuse_moe
}  // namespace hpc

#endif  // SRC_FUSE_MOE_FUSE_MOE_WARP_DECODE_H_
