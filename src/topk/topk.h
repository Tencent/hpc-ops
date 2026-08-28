// Copyright (C) 2026 Tencent.

#ifndef SRC_TOPK_TOPK_H_
#define SRC_TOPK_TOPK_H_

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace hpc {
namespace topk {

// Exact FP32 Top-K for variable-length rows. Dispatch depends only on the
// captured tensor capacities, while num_valid_rows is read on device.
bool topk_filtered_async(int *topk_indices, const float *logits, const int *ke, int topk,
                         const int *num_valid_rows, int m, int n, int row_stride, int out_stride,
                         void *counters, size_t counters_bytes, void *workspace,
                         size_t workspace_bytes, cudaStream_t stream);

// Persistent state that must be zero-filled before first use. The kernels leave
// this state zero-filled so callers can reuse it without a separate reset.
size_t topk_filtered_counters_bytes(int num_rows);

// Recommended scratch size for the shape-dependent execution mapping.
size_t topk_filtered_workspace_bytes(int num_rows, int max_kv_len);

// Smallest accepted scratch size. This preserves exactness while disabling the
// KV-split fast path when the recommended mapping requires more storage.
size_t topk_filtered_min_workspace_bytes(int max_kv_len);

// Largest scratch request over all row capacities for a fixed padded width.
size_t topk_filtered_peak_workspace_bytes(int max_kv_len);

}  // namespace topk
}  // namespace hpc

#endif  // SRC_TOPK_TOPK_H_
