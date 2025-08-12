// Copyright 2025 hpc-ops authors

#ifndef SRC_ADD_ADD_H_
#define SRC_ADD_ADD_H_

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace hpc {
namespace add {

void add_async(float *cptr, const float *aptr, const float *bptr, int num, cudaStream_t stream);

}  // namespace add
}  // namespace hpc

#endif  // SRC_ADD_ADD_H_
