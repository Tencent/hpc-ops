#ifndef SRC_CAST_CAST_H_
#define SRC_CAST_CAST_H_

#include <cuda_runtime_api.h>
#include <torch/all.h>


namespace hpc {
namespace cast {

void cast_async(void* cptr, const void* aptr, int num, torch::ScalarType tout, torch::ScalarType tin, cudaStream_t stream);


}  // namespace cast
}  // namespace hpc

#endif  // SRC_CAST_CAST_H_
