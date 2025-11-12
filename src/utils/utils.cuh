// Copyright 2025 hpc-ops authors

#ifndef SRC_UTILS_UTILS_CUH_
#define SRC_UTILS_UTILS_CUH_

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "cute/tensor.hpp"
#include "src/utils/utils.h"

namespace hpc {

// ============================
//    Debug Utility
// ============================

// print_type<T> _; // it will generate a compile time error with type T.
template <typename T>
struct print_type;

__device__ __forceinline__ void brkpt() { asm volatile("brkpt;" ::); }

// ============================
//    Load/Store(vectorized)
// ============================

template <typename T, int N>
struct vec_t {
  T data[N];

  using type = T;
  static constexpr int num = N;
  static constexpr int kNum = N;

  __device__ __forceinline__ T &operator[](int idx) { return data[idx]; }

  __device__ __forceinline__ const T &operator[](int idx) const { return data[idx]; }
};

template <int K, typename T, int N>
__device__ __forceinline__ auto &view(vec_t<T, N> &v) {
  using V = vec_t<vec_t<T, N / K>, K>;
  return *reinterpret_cast<V *>(&v);
}

template <typename T, int... Dims>
class vec_view;

template <typename T, int Dim>
class vec_view<T, Dim> {
  T *data_;

 public:
  using value_type = vec_t<T, Dim>;
  static constexpr int size() { return Dim; }

  __device__ __forceinline__ vec_view(T *data) : data_(data) {}

  constexpr __device__ __forceinline__ T &operator[](int i) { return data_[i]; }

  constexpr const __device__ __forceinline__ T &operator[](int i) const { return data_[i]; }

  __device__ __forceinline__ operator vec_t<T, Dim>() const {
    vec_t<T, Dim> ret;
#pragma unroll
    for (int i = 0; i < Dim; ++i) {
      ret.data[i] = data_[i];
    }
    return ret;
  }

  __device__ __forceinline__ vec_view &operator=(const vec_t<T, Dim> &rhs) {
#pragma unroll
    for (int i = 0; i < Dim; ++i) {
      data_[i] = rhs.data[i];
    }
    return *this;
  }
};

template <typename T, int First, int... Rest>
class vec_view<T, First, Rest...> {
  T *data_;

 public:
  static constexpr int inner_size = (Rest * ...);
  using inner_view_type = vec_view<T, Rest...>;
  using inner_value_type = typename inner_view_type::value_type;
  using value_type = vec_t<inner_value_type, First>;
  static constexpr int size() { return First; }
  __device__ __forceinline__ vec_view(T *data) : data_(data) {}

  __device__ __forceinline__ constexpr auto operator[](int i) {
    return vec_view<T, Rest...>(data_ + i * inner_size);
  }

  __device__ __forceinline__ constexpr auto operator[](int i) const {
    return vec_view<T, Rest...>(data_ + i * inner_size);
  }

  __device__ __forceinline__ operator vec_t<vec_t<T, inner_size>, First>() const {
    vec_t<vec_t<T, inner_size>, First> ret;
#pragma unroll
    for (int i = 0; i < First; ++i) {
#pragma unroll
      for (int j = 0; j < inner_size; ++j) {
        ret.data[i].data[j] = data_[i * inner_size + j];
      }
    }
    return ret;
  }

  template <int N>
  __device__ __forceinline__ vec_view &operator=(const vec_t<vec_t<T, N>, First> &rhs) {
    static_assert(N == inner_size, "size mismatch in assignment");
#pragma unroll
    for (int i = 0; i < First; ++i) {
#pragma unroll
      for (int j = 0; j < inner_size; ++j) {
        data_[i * inner_size + j] = rhs.data[i].data[j];
      }
    }
    return *this;
  }
};

template <int... NewDims, typename T, int N>
__device__ __forceinline__ auto reshape(vec_t<T, N> &v) {
  constexpr int total_elements = (NewDims * ...);
  static_assert(total_elements == N, "total elements must match in reshape");
  return vec_view<T, NewDims...>(v.data);
}

template <typename U, typename T, int N>
__device__ __forceinline__ auto to(const vec_t<T, N> &v) {
  if constexpr (std::is_same_v<T, float> && std::is_same_v<U, __nv_bfloat16>) {
    using V = vec_t<__nv_bfloat16, N>;
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
  } else if constexpr (std::is_same_v<T, __nv_bfloat16> && std::is_same_v<U, float>) {
    using V = vec_t<float, N>;
    V o;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      o[i] = __bfloat162float(v[i]);
    }
    return o;
  } else if constexpr (std::is_same_v<T, __nv_bfloat162> && std::is_same_v<U, float>) {
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
  } else if constexpr (std::is_same_v<T, float> && std::is_same_v<U, __nv_fp8x4_e4m3>) {
    static_assert(N % 4 == 0, "N % 4 must be 0");
    using V = vec_t<__nv_fp8x4_e4m3, N / 4>;
    V o;
#pragma unroll
    for (int i = 0; i < N / 4; ++i) {
      o[i] = __nv_fp8x4_e4m3(*reinterpret_cast<const float4 *>(&v[4 * i]));
    }
    return o;
  } else if constexpr (std::is_same_v<T, __nv_bfloat162> && std::is_same_v<U, __nv_fp8x4_e4m3>) {
    static_assert(N % 2 == 0, "N % 2 must be 0");
    using V = vec_t<__nv_fp8x4_e4m3, N / 2>;
    V o;
#pragma unroll
    for (int i = 0; i < N / 2; ++i) {
      o[i] = __nv_fp8x4_e4m3(*(reinterpret_cast<__nv_bfloat162 *>(&v[2 * i])),
                             *(reinterpret_cast<__nv_bfloat162 *>(&v[2 * i + 1])));
    }
    return o;
  } else if constexpr (std::is_same_v<T, float> && std::is_same_v<U, __nv_bfloat162>) {
    static_assert(N % 2 == 0, "N % 2 must be 0");
    using V = vec_t<__nv_bfloat162, N / 2>;
    V o;
#pragma unroll
    for (int i = 0; i < N / 2; ++i) {
      o[i] = __float22bfloat162_rn(*reinterpret_cast<const float2 *>(&v[2 * i]));
    }
    return o;
  }
}

template <typename U, typename T, int... Rest>
__device__ __forceinline__ auto to(const vec_view<T, Rest...> &view) {
  return to<U>(static_cast<typename vec_view<T, Rest...>::value_type>(view));
}

template <typename T, int N>
__device__ __forceinline__ auto load(const void *ptr) {
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
__device__ __forceinline__ void store(void *ptr, const vec_t<T, N> &v) {
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

template <typename T, int... Dims>
__device__ __forceinline__ void store(void *ptr, const vec_view<T, Dims...> &view) {
  store(ptr, static_cast<typename vec_view<T, Dims...>::value_type>(view));
}

template <typename T, typename... Args>
__device__ __forceinline__ void store(void *ptr, T val, Args... vals) {
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

__device__ __forceinline__ float expf_ftz(float x) {
  // e^x = (2^m)^x
  // e = 2^m
  // m = lg2(e)
  // m = 1.4426950408889634

  const float m = 1.4426950408889634f;
  float r;
  asm volatile("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(x * m));
  return r;
}

__device__ __forceinline__ float exp2f_ftz(float x) {
  float r;
  asm volatile("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(x));
  return r;
}

__device__ __forceinline__ float logf_ftz(float x) {
  // log(x) = lg2(x)log(2)
  // m = log(2) = 0.6931471805599453

  const float m = 0.6931471805599453f;
  float r;
  asm volatile("lg2.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(x));
  return r * m;
}

__device__ __forceinline__ float rcpf_ftz(float x) {
  float r;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(x));
  return r;
}

// y = x / (1 + e^(-x))
__device__ __forceinline__ float silu(float x) { return x * rcpf_ftz(1.f + expf_ftz(-x)); }

// y = log(1 + exp(x))
__device__ __forceinline__ float softplus(float x) { return logf_ftz(1.f + expf_ftz(x)); }

__device__ __forceinline__ float rsqrtf_ftz(float in) {
  float out;
  asm volatile("rsqrt.approx.ftz.f32 %0, %1;\n" : "=f"(out) : "f"(in));
  return out;
}

__device__ __forceinline__ float warp_reduce_sum_down(float x) {
#pragma unroll
  for (int ioffset = 16; ioffset >= 1; ioffset /= 2) {
    x += __shfl_down_sync(0xFFFFFFFF, x, ioffset);
  }

  return x;
}

__device__ __forceinline__ float warp_reduce_max_down(float x) {
#pragma unroll
  for (int ioffset = 16; ioffset >= 1; ioffset /= 2) {
    x = fmaxf(x, __shfl_down_sync(0xFFFFFFFF, x, ioffset));
  }

  return x;
}

__device__ __forceinline__ float warp_reduce_sum_xor(float x) {
#pragma unroll
  for (int ioffset = 16; ioffset >= 1; ioffset /= 2) {
    x += __shfl_xor_sync(0xFFFFFFFF, x, ioffset);
  }

  return x;
}

__device__ __forceinline__ float warp_4lane_reduce_max_xor(float x) {
  x = fmaxf(__shfl_xor_sync(0xFFFFFFFF, x, 1), x);
  x = fmaxf(__shfl_xor_sync(0xFFFFFFFF, x, 2), x);

  return x;
}

__device__ __forceinline__ float warp_4lane_reduce_sum_xor(float x) {
  x += __shfl_xor_sync(0xFFFFFFFF, x, 1);
  x += __shfl_xor_sync(0xFFFFFFFF, x, 2);

  return x;
}

__device__ __forceinline__ float warp_8lane_stride4_reduce_max_xor(float x) {
  x = fmaxf(__shfl_xor_sync(0xFFFFFFFF, x, 4), x);
  x = fmaxf(__shfl_xor_sync(0xFFFFFFFF, x, 8), x);
  x = fmaxf(__shfl_xor_sync(0xFFFFFFFF, x, 16), x);

  return x;
}

__device__ __forceinline__ float warp_8lane_stride4_reduce_sum_xor(float x) {
  x += __shfl_xor_sync(0xFFFFFFFF, x, 4);
  x += __shfl_xor_sync(0xFFFFFFFF, x, 8);
  x += __shfl_xor_sync(0xFFFFFFFF, x, 16);

  return x;
}

// ============================
//    Fragment Retile
// ============================

template <typename Tensor>
__device__ __forceinline__ auto retile_fragment(Tensor &&tensor) {
  using namespace cute;  // NOLINT

  constexpr int R = decltype(tensor.layout())::rank;
  static_assert(R == 3, "we only support rank 3 fragment");

  auto thr_vmk = flatten(select<0>(tensor.layout()));
  auto tile_mk = select<1, 2>(tensor.layout());

  auto m_layout =
      coalesce(make_layout(make_shape(get<1>(thr_vmk.shape()), get<0>(tile_mk.shape())),
                           make_stride(get<1>(thr_vmk.stride()), get<0>(tile_mk.stride()))));
  auto k_layout = coalesce(make_layout(
      make_shape(get<0>(thr_vmk.shape()), get<2>(thr_vmk.shape()), get<1>(tile_mk.shape())),
      make_stride(get<0>(thr_vmk.stride()), get<2>(thr_vmk.stride()), get<1>(tile_mk.stride()))));

  return make_tensor(static_cast<Tensor &&>(tensor).data(), make_layout(m_layout, k_layout));
}

}  // namespace hpc

#endif  // SRC_UTILS_UTILS_CUH_
