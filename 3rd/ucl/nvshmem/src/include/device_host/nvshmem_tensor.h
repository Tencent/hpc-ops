/*
 * Copyright (c) 2018-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See License.txt for license information
 */

#ifndef _NVSHMEM_TENSOR_H_
#define _NVSHMEM_TENSOR_H_

#if !defined __CUDACC_RTC__
#include <stdint.h>
#include <limits.h>
#else
#include "cuda/std/cstdint"
#include <cuda/std/climits>
#include <cuda/std/cstddef>
#if !defined SIZE_MAX
#define SIZE_MAX (1ULL << 63)
#endif
#endif
#include <cuda_runtime.h>
#ifdef NVSHMEM_COMPLEX_SUPPORT
#include <complex.h>
#endif
#include "cuda_fp16.h"
#include "cuda_bf16.h"
#include "non_abi/nvshmem_build_options.h"
#include "device_host_transport/nvshmem_common_transport.h"
#include "device_host/nvshmem_types.h"
#include "device_host_transport/nvshmem_constants.h"
#include "cuda/std/tuple"
#include "cuda/std/type_traits"

template <int v>
struct ConstInt : cuda::std::integral_constant<int, v> {
    __host__ __device__ constexpr operator int() const { return v; }
};

namespace nvshmemx {

template <class... Shapes>
using shape = cuda::std::tuple<Shapes...>;

template <class... Strides>
using stride = cuda::std::tuple<Strides...>;

template <class... Ts>
__host__ __device__ constexpr shape<Ts...> make_shape(Ts const&... t) {
    return {t...};
}

template <class... Ts>
__host__ __device__ constexpr stride<Ts...> make_stride(Ts const&... t) {
    return {t...};
}

template <class Shape, class Stride>
struct Layout : private cuda::std::tuple<Shape, Stride> {
    __host__ __device__ constexpr Layout(Shape const& shape = {}, Stride const& stride = {})
        : cuda::std::tuple<Shape, Stride>(shape, stride) {}

    __host__ __device__ constexpr Shape shape() const {
        return cuda::std::get<0>(static_cast<const cuda::std::tuple<Shape, Stride>&>(*this));
    }

    __host__ __device__ constexpr Stride stride() const {
        return cuda::std::get<1>(static_cast<const cuda::std::tuple<Shape, Stride>&>(*this));
    }
};

template <class Shape, class Stride>
__host__ __device__ constexpr Layout<Shape, Stride> make_layout(Shape const& shape,
                                                                Stride const& stride) {
    return Layout<Shape, Stride>(shape, stride);
}

template <typename T, class Layout>
struct Tensor {
    using value_type = T;
    using layout_type = Layout;
    T* _data;
    const Layout _layout;

    __host__ __device__ constexpr Tensor(T* data, Layout layout) : _data(data), _layout(layout) {}
    Tensor(Layout layout) : _data(NULL), _layout(layout) {}

    __host__ __device__ constexpr Layout layout() const { return _layout; }

    __host__ __device__ T* data() { return _data; }

    __host__ __device__ constexpr decltype(_layout.stride()) const stride() const {
        return _layout.stride();
    }

    __host__ __device__ constexpr decltype(_layout.shape()) const shape() const {
        return _layout.shape();
    }
};

typedef enum {
    NVLS_ONE_SHOT_PUSH_NBI = 0,
    NVLS_ONE_SHOT_PULL_NBI = 1,
    NVLS_TWO_SHOT_PUSH_NBI = 2,
    NVLS_TWO_SHOT_PULL_NBI = 3,
    NVSHMEMI_TILE_ALGO_SENTINEL = INT_MAX
} tile_coll_algo_t;

}  // namespace nvshmemx
#endif
