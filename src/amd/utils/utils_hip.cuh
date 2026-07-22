// Copyright (C) 2026 Tencent.
// Slimmed HIP port of utils.cuh — shared device helpers for AMD gfx950 (wave64).
// Kept verbatim from the original: vec_t / traits_vec_t / size / reshape /
// to<> / load / store. Ported: rcpf_ftz / rsqrtf_ftz (PTX -> HIP intrinsics),
// warp_reduce_sum_xor (wave32 -> wave64). Removed: cute/CUTLASS include and
// attention helpers, multimem/atom/barrier PTX, unused fast-math.

#ifndef SRC_AMD_UTILS_UTILS_HIP_CUH_
#define SRC_AMD_UTILS_UTILS_HIP_CUH_

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

namespace hpc {

// ============================
//    Load/Store(vectorized)
// ============================
template <typename T, int N>
struct vec_t {
  T data[N];

  using type = T;
  static constexpr int num = N;
  static constexpr int kNum = N;

  __device__ __forceinline__ constexpr T &operator[](int idx) { return data[idx]; }

  __device__ __forceinline__ constexpr const T &operator[](int idx) const { return data[idx]; }
};

template <typename T, int N, int... Dims>
struct traits_vec_t;

template <typename T, int N, int Dim>
struct traits_vec_t<T, N, Dim> {
  static_assert(N == Dim, "dimension mismatch");
  using type = vec_t<T, Dim>;
};

template <typename T, int N, int First, int... Rest>
struct traits_vec_t<T, N, First, Rest...> {
  static_assert(N % First == 0, "first dimension must divide total size");
  using inner_type = typename traits_vec_t<T, N / First, Rest...>::type;
  using type = vec_t<inner_type, First>;
};

template <typename T, int N>
__device__ __forceinline__ constexpr int size(vec_t<T, N> &v) {
  return N;
}

template <int... Dims, typename T, int N>
__device__ __forceinline__ constexpr auto &reshape(vec_t<T, N> &v) {
  constexpr int num_elements = (Dims * ...);

  using ResultType = typename traits_vec_t<T, N, Dims...>::type;
  return *reinterpret_cast<ResultType *>(&v);
}

template <typename U, typename T, int N>
__device__ __forceinline__ constexpr auto to(const vec_t<T, N> &v) {
  if constexpr (std::is_same_v<T, float> && std::is_same_v<U, __hip_bfloat16>) {
    using V = vec_t<__hip_bfloat16, N>;
    V o;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      o[i] = __float2bfloat16(v[i]);
    }
    return o;
  } else if constexpr (std::is_same_v<T, float> && std::is_same_v<U, __half2>) {
    using V = vec_t<U, N / 2>;
    V o;
#pragma unroll
    for (int i = 0; i < N / 2; ++i) {
      o[i] = __float22half2_rn(*reinterpret_cast<const float2 *>(&v[2 * i]));
    }
    return o;
  } else if constexpr (std::is_same_v<T, __hip_bfloat16> && std::is_same_v<U, float>) {
    using V = vec_t<float, N>;
    V o;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      o[i] = __bfloat162float(v[i]);
    }
    return o;
  } else if constexpr (std::is_same_v<T, __hip_bfloat162> && std::is_same_v<U, float>) {
    using V = vec_t<float, N * 2>;
    V o;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      auto y = __bfloat1622float2(v[i]);
      o[i * 2 + 0] = y.x;
      o[i * 2 + 1] = y.y;
    }
    return o;
  } else if constexpr (std::is_same_v<T, __half2> && std::is_same_v<U, float>) {
    using V = vec_t<float, N * 2>;
    V o;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      auto y = __half22float2(v[i]);
      o[i * 2 + 0] = y.x;
      o[i * 2 + 1] = y.y;
    }
    return o;
  } else if constexpr (std::is_same_v<T, float> && std::is_same_v<U, __hip_fp8x4_e4m3>) {
    static_assert(N % 4 == 0, "N % 4 must be 0");
    using V = vec_t<__hip_fp8x4_e4m3, N / 4>;
    V o;
#pragma unroll
    for (int i = 0; i < N / 4; ++i) {
      o[i] = __hip_fp8x4_e4m3(*reinterpret_cast<const float4 *>(&v[4 * i]));
    }
    return o;
  } else if constexpr (std::is_same_v<T, __hip_fp8x4_e4m3> && std::is_same_v<U, float>) {
    using V = vec_t<float, N * 4>;
    V o;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      auto y = static_cast<float4>(v[i]);
      o[i * 4 + 0] = y.x;
      o[i * 4 + 1] = y.y;
      o[i * 4 + 2] = y.z;
      o[i * 4 + 3] = y.w;
    }
    return o;
  } else if constexpr (std::is_same_v<T, __hip_fp8_e4m3> && std::is_same_v<U, float>) {
    using V = vec_t<float, N>;
    V o;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      o[i] = static_cast<float>(v[i]);
    }
    return o;
  } else if constexpr (std::is_same_v<T, __hip_bfloat162> && std::is_same_v<U, __hip_fp8x4_e4m3>) {
    static_assert(N % 2 == 0, "N % 2 must be 0");
    using V = vec_t<__hip_fp8x4_e4m3, N / 2>;
    V o;
#pragma unroll
    for (int i = 0; i < N / 2; ++i) {
      o[i] = __hip_fp8x4_e4m3(*(reinterpret_cast<__hip_bfloat162 *>(&v[2 * i])),
                              *(reinterpret_cast<__hip_bfloat162 *>(&v[2 * i + 1])));
    }
    return o;
  } else if constexpr (std::is_same_v<T, float> && std::is_same_v<U, __hip_bfloat162>) {
    static_assert(N % 2 == 0, "N % 2 must be 0");
    using V = vec_t<__hip_bfloat162, N / 2>;
    V o;
#pragma unroll
    for (int i = 0; i < N / 2; ++i) {
      o[i] = __float22bfloat162_rn(*reinterpret_cast<const float2 *>(&v[2 * i]));
    }
    return o;
  } else if constexpr (std::is_same_v<T, U>) {
    return v;
  }
}

template <typename T, int N>
__device__ __forceinline__ constexpr auto load(const void *ptr) {
  using V = vec_t<T, N>;
  V v;

  constexpr int kBytes = sizeof(T) * N;

  static_assert(kBytes == 1 || kBytes == 2 || kBytes == 4 || kBytes == 8 || kBytes == 16,
                "not support for T x N");

  if constexpr (kBytes == 1) {
    using L = uint8_t;
    *reinterpret_cast<L *>(&v) = *reinterpret_cast<const L *>(ptr);
  } else if constexpr (kBytes == 2) {
    using L = uint16_t;
    *reinterpret_cast<L *>(&v) = *reinterpret_cast<const L *>(ptr);
  } else if constexpr (kBytes == 4) {
    using L = uint32_t;
    *reinterpret_cast<L *>(&v) = *reinterpret_cast<const L *>(ptr);
  } else if constexpr (kBytes == 8) {
    using L = uint64_t;
    *reinterpret_cast<L *>(&v) = *reinterpret_cast<const L *>(ptr);
  } else if constexpr (kBytes == 16) {
    using L = uint4;
    *reinterpret_cast<L *>(&v) = *reinterpret_cast<const L *>(ptr);
  }

  return v;
}

template <typename T, int N>
__device__ __forceinline__ constexpr void store(void *ptr, const vec_t<T, N> &v) {
  using V = vec_t<T, N>;

  constexpr int kBytes = sizeof(T) * N;

  static_assert(kBytes == 1 || kBytes == 2 || kBytes == 4 || kBytes == 8 || kBytes == 16,
                "not support for T x N");

  if constexpr (kBytes == 1) {
    using S = uint8_t;
    *reinterpret_cast<S *>(ptr) = *reinterpret_cast<const S *>(&v);
  } else if constexpr (kBytes == 2) {
    using S = uint16_t;
    *reinterpret_cast<S *>(ptr) = *reinterpret_cast<const S *>(&v);
  } else if constexpr (kBytes == 4) {
    using S = uint32_t;
    *reinterpret_cast<S *>(ptr) = *reinterpret_cast<const S *>(&v);
  } else if constexpr (kBytes == 8) {
    using S = uint64_t;
    *reinterpret_cast<S *>(ptr) = *reinterpret_cast<const S *>(&v);
  } else if constexpr (kBytes == 16) {
    using S = uint4;
    *reinterpret_cast<S *>(ptr) = *reinterpret_cast<const S *>(&v);
  }

  return;
}

// Streaming (non-temporal) store — an OPT-IN variant of store() for outputs that
// are written once and never read back within the kernel. The nt/slc hint tells
// the hardware to bypass the L2 write-allocate so a pure output stream does not
// evict live inputs from L2; on bandwidth-bound streaming-write kernels this is
// a net win. It is NOT safe for buffers that are read back in-kernel (in-place
// accumulation, L2-reused scratch), which is why plain store() above keeps the
// default temporal path — callers must request streaming explicitly.
//
// The 16-byte path stays a regular temporal dwordx4 store on purpose: measured
// on the fp32 fused-RMSNorm MoE output (the dominant MoE traffic term) a nt hint
// there REGRESSES because write-allocate helps that wide, fully-coalesced stream.
template <typename T, int N>
__device__ __forceinline__ constexpr void store_streaming(void *__restrict__ ptr,
                                                          const vec_t<T, N> &v) {
  using V = vec_t<T, N>;

  constexpr int kBytes = sizeof(T) * N;

  static_assert(kBytes == 1 || kBytes == 2 || kBytes == 4 || kBytes == 8 || kBytes == 16,
                "not support for T x N");

  if constexpr (kBytes == 1) {
    using S = uint8_t;
    __builtin_nontemporal_store(*reinterpret_cast<const S *>(&v), reinterpret_cast<S *>(ptr));
  } else if constexpr (kBytes == 2) {
    using S = uint16_t;
    __builtin_nontemporal_store(*reinterpret_cast<const S *>(&v), reinterpret_cast<S *>(ptr));
  } else if constexpr (kBytes == 4) {
    using S = uint32_t;
    __builtin_nontemporal_store(*reinterpret_cast<const S *>(&v), reinterpret_cast<S *>(ptr));
  } else if constexpr (kBytes == 8) {
    using S = uint64_t;
    __builtin_nontemporal_store(*reinterpret_cast<const S *>(&v), reinterpret_cast<S *>(ptr));
  } else if constexpr (kBytes == 16) {
    using S = uint4;
    *reinterpret_cast<S *>(ptr) = *reinterpret_cast<const S *>(&v);
  }

  return;
}

template <typename T, typename... Args>
__device__ __forceinline__ constexpr void store(void *ptr, T val, Args... vals) {
  constexpr int N = sizeof...(Args);
  using V = vec_t<T, 1 + N>;

  static_assert((std::is_same_v<Args, T> && ...), "all vals must be type of T");

  V v;
  int idx = 0;
  (reinterpret_cast<T *>(&v))[idx++] = val;
  (((reinterpret_cast<T *>(&v))[idx++] = vals), ...);

  store(ptr, v);
}

// ============================
//       Fast Math API
// ============================

// PTX rcp.approx.ftz.f32 -> HIP round-to-nearest reciprocal intrinsic.
__device__ __forceinline__ float rcpf_ftz(float x) { return __frcp_rn(x); }

// PTX rsqrt.approx.ftz.f32 -> HIP fast inverse square root.
__device__ __forceinline__ float rsqrtf_ftz(float in) { return rsqrtf(in); }

// Butterfly all-reduce across the wavefront. Offset starts at warpSize/2 so it
// covers the full wave (32 lanes on NVIDIA, 64 on AMD CDNA). HIP __shfl_xor
// defaults to width == warpSize.
__device__ __forceinline__ float warp_reduce_sum_xor(float x) {
#pragma unroll
  for (int ioffset = warpSize / 2; ioffset >= 1; ioffset /= 2) {
    x += __shfl_xor(x, ioffset);
  }

  return x;
}

}  // namespace hpc

#endif  // SRC_AMD_UTILS_UTILS_HIP_CUH_
