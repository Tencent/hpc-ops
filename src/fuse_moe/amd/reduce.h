// Copyright (C) 2026 Tencent.
#ifndef SRC_FUSE_MOE_AMD_REDUCE_H_
#define SRC_FUSE_MOE_AMD_REDUCE_H_

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

namespace hpc::fuse_moe::amd {

void reduce_async(hip_bfloat16* output, const hip_bfloat16* input, const int* positions,
                  const float* scales, const hip_bfloat16* shared_output, int rows, int input_rows,
                  int hidden, int topk, hipStream_t stream);

}  // namespace hpc::fuse_moe::amd
#endif  // SRC_FUSE_MOE_AMD_REDUCE_H_
