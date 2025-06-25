#include <cuda.h>

namespace hpc {
namespace cast {

namespace kernels {

template <typename Tout, typename Tin>
struct Convertor {
    __device__ __host__ static Tout convert(const Tin& in) {
        return static_cast<Tout>(in);
    }
};

template <>
struct Convertor<half, float> {
    __device__ __host__ static half convert(const float& in) {
      return __float2half(in);
    }
};

template <typename Tout, typename Tin>
__global__ void cast(void *cptr, const void *aptr, int num) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx >= num) {
    return;
  }

  Tin *iptr = (Tin*)aptr;
  Tout *optr = (Tout*)cptr;

  Tin vin = iptr[idx];
  Tout vout = Convertor<Tout, Tin>::convert(vin);

  optr[idx] = vout;
}
}  // namespace kernels

template <typename Tout, typename Tin>
void cast_async(void *cptr, const void *aptr, int num, cudaStream_t stream) {
    dim3 block(128);
    dim3 grid((num + block.x - 1) / block.x);
    kernels::cast<Tout, Tin><<<grid, block, 0, stream>>>(cptr, aptr, num);
}

}  // namespace cast 
}  // namespace hpc 
