// Copyright (C) 2026 Tencent.
#ifndef SRC_FUSE_MOE_FUSE_MOE_W4A8_H_
#define SRC_FUSE_MOE_FUSE_MOE_W4A8_H_

#include <cuda_runtime_api.h>
#include <stddef.h>
#include <stdint.h>

namespace hpc {
namespace fuse_moe {

// One per-call workspace, owned by the caller until current-stream completion.
size_t fuse_moe_w4a8_workspace_size(int num_seq, int top_k);
bool fuse_moe_w4a8_async(void *output, const void *x, const void *w13, const uint8_t *s13,
                         const float *ss13, const void *w2, const uint8_t *s2, const float *ss2,
                         const int *ids, const float *weights, void *workspace, int num_seq,
                         int top_k, cudaStream_t stream);

}  // namespace fuse_moe
}  // namespace hpc
#endif  // SRC_FUSE_MOE_FUSE_MOE_W4A8_H_
