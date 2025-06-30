#include <cuda.h>

namespace hpc {
namespace ptx {

__device__ __forceinline__ float rcp_ftz(float in) {
  float out;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(out) : "f"(in));
  return out;
}

__device__ __forceinline__ float rsqrt_ftz(float in) {
  float out;
  asm volatile("rsqrt.approx.ftz.f32 %0, %1;\n" : "=f"(out) : "f"(in));
  return out;
}

}  // namespace ptx
}  // namespace hpc
