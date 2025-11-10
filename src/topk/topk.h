// Copyright 2025 hpc-ops authors

#ifndef SRC_TOPK_TOPK_H_
#define SRC_TOPK_TOPK_H_

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace hpc {
namespace topk {

bool topk_per_row_async(int *topk_indices, const float *logits, const int *seqlens, int topk,
                        int num_sp_tokens, int num_rows, int row_stride, cudaStream_t stream);

}  // namespace topk
}  // namespace hpc

#endif  // SRC_TOPK_TOPK_H_
