#include <cuda.h>

#include "src/add/add.h"

namespace hpc {
namespace add {

namespace kernels {
__global__ void add(float *cptr, const float *aptr, const float *bptr,
                    int num) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx >= num) {
    return;
  }

  cptr[idx] = aptr[idx] + bptr[idx];
}
}  // namespace kernels

void add_async(float *cptr, const float *aptr, const float *bptr, int num,
               cudaStream_t stream) {
  dim3 block(128);
  dim3 grid((num + block.x - 1) / block.x);
  kernels::add<<<grid, block, 0, stream>>>(cptr, aptr, bptr, num);
}

}  // namespace add
}  // namespace hpc
