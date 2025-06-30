#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <torch/all.h>

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

template <>
struct Convertor<__nv_bfloat16, float> {
  __device__ __host__ static __nv_bfloat16 convert(const float& in) {
    return __float2bfloat16(in);
  }
};

template <>
struct Convertor<__nv_fp8_e4m3, float> {
  __device__ __host__ static __nv_fp8_e4m3 convert(const float& in) {
    __nv_fp8_e4m3 vout{in};
    return vout;
  }
};

template <>
struct Convertor<__nv_fp8_e5m2, float> {
  __device__ __host__ static __nv_fp8_e5m2 convert(const float& in) {
    __nv_fp8_e5m2 vout{in};
    return vout;
  }
};

template <>
struct Convertor<__nv_fp8_e8m0, float> {
  __device__ __host__ static __nv_fp8_e8m0 convert(const float& in) {
    __nv_fp8_e8m0 vout{in};
    return vout;
  }
};

template <typename Tout, typename Tin>
__global__ void cast(void* cptr, const void* aptr, int num) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx >= num) {
    return;
  }

  Tin* iptr = (Tin*)aptr;
  Tout* optr = (Tout*)cptr;

  Tin vin = iptr[idx];
  Tout vout = Convertor<Tout, Tin>::convert(vin);

  optr[idx] = vout;
}
}  // namespace kernels

void cast_async(void* cptr, const void* aptr, int num, torch::ScalarType tout,
                torch::ScalarType tin, cudaStream_t stream) {
  dim3 block(128);
  dim3 grid((num + block.x - 1) / block.x);

  if (tin == torch::kFloat32) {
    switch (tout) {
      case torch::kFloat32: {
        kernels::cast<float, float>
            <<<grid, block, 0, stream>>>(cptr, aptr, num);
        break;
      }
      case torch::kFloat64: {
        kernels::cast<double, float>
            <<<grid, block, 0, stream>>>(cptr, aptr, num);
        break;
      }
      case torch::kFloat16: {
        kernels::cast<half, float><<<grid, block, 0, stream>>>(cptr, aptr, num);
        break;
      }
      case torch::kBFloat16: {
        kernels::cast<__nv_bfloat16, float>
            <<<grid, block, 0, stream>>>(cptr, aptr, num);
        break;
      }
      case torch::kFloat8_e4m3fn: {
        kernels::cast<__nv_fp8_e4m3, float>
            <<<grid, block, 0, stream>>>(cptr, aptr, num);
        break;
      }
      case torch::kFloat8_e5m2: {
        kernels::cast<__nv_fp8_e5m2, float>
            <<<grid, block, 0, stream>>>(cptr, aptr, num);
        break;
      }
      case torch::kFloat8_e8m0fnu: {
        kernels::cast<__nv_fp8_e8m0, float>
            <<<grid, block, 0, stream>>>(cptr, aptr, num);
        break;
      }
      default: {
        throw std::invalid_argument("not support yet!");
        break;
      }
    }  // switch
  }    // if
}

}  // namespace cast
}  // namespace hpc
