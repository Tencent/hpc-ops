#ifndef SRC_UTILS_UTILS_H_
#define SRC_UTILS_UTILS_H_

namespace hpc {

__device__ __forceinline__ float expf_ftz(float x) {
  // e = 2^m
  // m = 1.4426950408889634

  const float m = 1.4426950408889634f;
  float r;
  asm volatile("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(x * m));
  return r;
}

__device__ __forceinline__ float rcpf_ftz(float x) {
  float r;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(x));
  return r;
}

// x / (1 + e^(-x))
__device__ __forceinline__ float silu(float x) {
  return x * rcpf_ftz(1.f + expf_ftz(-x));
}

}  // namespace hpc

#endif  // SRC_UTILS_UTILS_H_
