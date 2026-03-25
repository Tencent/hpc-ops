/*
 * Copyright (c) 2016-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */
#ifndef _NVSHMEM_TILE_UTILS_CUH_
#define _NVSHMEM_TILE_UTILS_CUH_

#include <cuda_runtime.h>
#include "cuda/std/tuple"
#include "cuda/std/type_traits"
#include "cuda/std/utility"
#include "host/nvshmem_macros.h"
#include "device_host/nvshmem_tensor.h"

using tuple5Int_t = cuda::std::tuple<int, int, int, int, int>;

/**** Functions to get constant values at compile time ****/

template <typename T, typename Enable = void>
struct is_integral_constant {
    static const bool value = false;
};

// Specialization for integral_constant types
template <typename T>
struct is_integral_constant<
    T, typename cuda::std::enable_if<cuda::std::is_base_of<
           cuda::std::integral_constant<typename T::value_type, T::value>, T>::value>::type> {
    static const bool value = true;
};

// Check if T is an integral constant with specific value
template <typename T, int N, typename Enable = void>
struct is_integral_constant_value {
    static const bool value = false;
};

template <typename T, int N>
struct is_integral_constant_value<
    T, N, typename cuda::std::enable_if<is_integral_constant<T>::value>::type> {
    static const bool value = (T::value == N);
};

// Helper function to get integral constant value at compile time
template <typename T, typename Enable = void>
struct get_constant_value {
    __host__ __device__ __forceinline__ static constexpr int value() { return 0; }  // default case
};

template <typename T>
struct get_constant_value<T, typename cuda::std::enable_if<is_integral_constant<T>::value>::type> {
    __host__ __device__ __forceinline__ static constexpr int value() { return T::value; }
};

// Convenience wrapper function
template <typename T>
__host__ __device__ __forceinline__ constexpr int get_constant() {
    return get_constant_value<T>::value();
}

template <typename T>
__host__ __device__ __forceinline__ constexpr int get_constant(T val) {
    return get_constant_value<T>::value();
}
/**** End of Functions to get constant values at compile time ****/

/***** Helper functions for tuples ******/
// Traits to get tuple size at compile time
template <typename Tuple>
struct tuple_size_traits {
    static constexpr size_t value = cuda::std::tuple_size<Tuple>::value;
};

// Check if index is in bounds
template <int I, typename Tuple>
struct is_index_in_bounds {
    static constexpr bool value = (I >= 0 && I < tuple_size_traits<Tuple>::value);
};
/**** End of Helper functions for tuples ******/

/*** Function to return the size of a tuple at compile time ****/
// TODO: overload for cute::tuple
template <typename... Args>
NVSHMEMI_HOSTDEVICE_PREFIX constexpr size_t get_tuple_size(
    cuda::std::tuple<Args...> const& /*tuple*/) {  // The instance isn't needed
    return cuda::std::tuple_size<cuda::std::tuple<Args...>>::value;
}

template <int I, typename... Args>
/*NVSHMEMI_HOSTDEVICE_PREFIX constexpr*/
__host__ __device__ __forceinline__ constexpr decltype(
    cuda::std::get<I>(cuda::std::declval<const cuda::std::tuple<Args...>&>()))
get_tuple_val(cuda::std::tuple<Args...> const& tuple) {
    return cuda::std::get<I>(tuple);
}

/*** Accessor functions for shape ***/
/*** if index used is out of bounds, return 1 ***/
// Primary template for shape access (out-of-bounds)
template <int I, typename T, class Layout, bool InBounds>
struct tensor_shape_element_impl {
    NVSHMEMI_HOSTDEVICE_PREFIX static constexpr int get(const nvshmemx::Tensor<T, Layout>& tensor) {
        return 1;  // Out of bounds, return 1
    }
};

// Specialization for in-bounds shape access
template <int I, typename T, class Layout>
struct tensor_shape_element_impl<I, T, Layout, true> {
    NVSHMEMI_HOSTDEVICE_PREFIX static constexpr auto get(const nvshmemx::Tensor<T, Layout>& tensor)
        -> decltype(cuda::std::get<I>(tensor.shape())) {
        return cuda::std::get<I>(tensor.shape());
    }
};

// Public interface for shape
template <int I, typename T, class Layout>
NVSHMEMI_HOSTDEVICE_PREFIX constexpr auto get_shape_element(
    const nvshmemx::Tensor<T, Layout>& tensor)
    -> decltype(
        tensor_shape_element_impl<
            I, T, Layout, is_index_in_bounds<I, decltype(tensor.shape())>::value>::get(tensor)) {
    return tensor_shape_element_impl<
        I, T, Layout, is_index_in_bounds<I, decltype(tensor.shape())>::value>::get(tensor);
}

/*** Accessor functions for stride ***/
/*** if index used is out of bounds, return 0 ***/

// Primary template for stride access (out-of-bounds)
template <int I, typename T, class Layout, bool InBounds>
struct tensor_stride_element_impl {
    NVSHMEMI_HOSTDEVICE_PREFIX static constexpr int get(const nvshmemx::Tensor<T, Layout>& tensor) {
        return 0;  // Out of bounds, return 0 for stride
    }
};

// Specialization for in-bounds stride access
template <int I, typename T, class Layout>
struct tensor_stride_element_impl<I, T, Layout, true> {
    NVSHMEMI_HOSTDEVICE_PREFIX static constexpr auto get(const nvshmemx::Tensor<T, Layout>& tensor)
        -> decltype(cuda::std::get<I>(tensor.stride())) {
        return cuda::std::get<I>(tensor.stride());
    }
};

// Public interface for stride
template <int I, typename T, class Layout>
NVSHMEMI_HOSTDEVICE_PREFIX constexpr auto get_stride_element(
    const nvshmemx::Tensor<T, Layout>& tensor)
    -> decltype(
        tensor_stride_element_impl<
            I, T, Layout, is_index_in_bounds<I, decltype(tensor.stride())>::value>::get(tensor)) {
    return tensor_stride_element_impl<
        I, T, Layout, is_index_in_bounds<I, decltype(tensor.stride())>::value>::get(tensor);
}

/**** Safe access functions for tuples ******/
/*** Returns 0 for out-of-bounds access ***/

// Primary template for safe tuple access (out-of-bounds)
template <int I, typename Tuple, bool InBounds>
struct safe_get_impl {
    // Out of bounds access returns int
    NVSHMEMI_HOSTDEVICE_PREFIX
    static constexpr int get(const Tuple&) {
        return 0;  // Default to 0 for out-of-bounds
    }
};

// Specialization for in-bounds access
template <int I, typename Tuple>
struct safe_get_impl<I, Tuple, true> {
    // Define the return type for in-bounds access
    typedef decltype(cuda::std::get<I>(cuda::std::declval<const Tuple&>())) return_type;

    NVSHMEMI_HOSTDEVICE_PREFIX
    static constexpr return_type get(const Tuple& tuple) { return cuda::std::get<I>(tuple); }
};

// Public interface for safe tuple access
template <int I, typename Tuple>
NVSHMEMI_HOSTDEVICE_PREFIX constexpr
    typename cuda::std::conditional<(I < tuple_size_traits<Tuple>::value),
                                    decltype(cuda::std::get<I>(cuda::std::declval<const Tuple&>())),
                                    int>::type
    safe_get(const Tuple& tuple) {
    return safe_get_impl<I, Tuple, (I < tuple_size_traits<Tuple>::value)>::get(tuple);
}

/**** Function to tuple of size 2 accounting for the major dimension */
// TODO:  expand this to support more than 2 dimensions
template <int major_dim>
NVSHMEMI_HOSTDEVICE_PREFIX __forceinline__ tuple5Int_t create_coord_tuple(int coord_major,
                                                                          int coord_minor) {
    if (major_dim == 0) {
        return cuda::std::make_tuple(coord_major, coord_minor, 0, 0, 0);
    } else {
        return cuda::std::make_tuple(coord_minor, coord_major, 0, 0, 0);
    }
}

/*
 * Functionto compare elements of tuples
 */
// TODO make a cute variant of this
// Helper to check bounds for each dimension
// Helper to check if a dimension is valid
template <typename tuple_t>
NVSHMEMI_HOSTDEVICE_PREFIX __forceinline__ bool check_dimension_valid(const tuple_t& start_coord,
                                                                      const tuple5Int_t& elem_coord,
                                                                      const tuple_t& boundary,
                                                                      int dim, int major_dim,
                                                                      int vlen) {
    if (dim == major_dim) {
        return (get_tuple_val<0>(start_coord) + get_tuple_val<0>(elem_coord) + (vlen - 1) <
                get_tuple_val<0>(boundary));
    } else {
        return (get_tuple_val<0>(start_coord) + get_tuple_val<0>(elem_coord) <
                get_tuple_val<0>(boundary));
    }
}

// Primary template for dimension checking
template <int Dim, int MaxDim, typename tuple_t, bool = (Dim < MaxDim)>
struct dimension_checker {
    NVSHMEMI_HOSTDEVICE_PREFIX __forceinline__ static bool check(const tuple_t& start_coord,
                                                                 const tuple5Int_t& elem_coord,
                                                                 const tuple_t& boundary,
                                                                 int major_dim, int vlen) {
        bool current_valid;

        if (Dim == major_dim) {
            current_valid =
                (get_tuple_val<Dim>(start_coord) + get_tuple_val<Dim>(elem_coord) + (vlen - 1) <
                 get_tuple_val<Dim>(boundary));
        } else {
            current_valid = (get_tuple_val<Dim>(start_coord) + get_tuple_val<Dim>(elem_coord) <
                             get_tuple_val<Dim>(boundary));
        }

        return current_valid && dimension_checker<Dim + 1, MaxDim, tuple_t>::check(
                                    start_coord, elem_coord, boundary, major_dim, vlen);
    }
};

// Base case: When Dim == MaxDim
template <int Dim, int MaxDim, typename tuple_t>
struct dimension_checker<Dim, MaxDim, tuple_t, false> {
    NVSHMEMI_HOSTDEVICE_PREFIX __forceinline__ static bool check(const tuple_t&, const tuple5Int_t&,
                                                                 const tuple_t&, int, int) {
        return true;  // End of recursion
    }
};

// Special case for 1D tuples
template <typename tuple_t, int major_dim>
NVSHMEMI_HOSTDEVICE_PREFIX __forceinline__
    typename cuda::std::enable_if<(tuple_size_traits<tuple_t>::value == 1), bool>::type
    is_less_than_impl(const tuple_t& start_coord, const tuple5Int_t& elem_coord,
                      const tuple_t& boundary, int vlen) {
    return check_dimension_valid(start_coord, elem_coord, boundary, 0, major_dim, vlen);
}

// Special case for 2D tuples
template <typename tuple_t, int major_dim>
NVSHMEMI_HOSTDEVICE_PREFIX __forceinline__
    typename cuda::std::enable_if<(tuple_size_traits<tuple_t>::value == 2), bool>::type
    is_less_than_impl(const tuple_t& start_coord, const tuple5Int_t& elem_coord,
                      const tuple_t& boundary, int vlen) {
    return dimension_checker<0, 2, tuple_t>::check(start_coord, elem_coord, boundary, major_dim,
                                                   vlen);
}

// Main function with runtime dispatching based on tuple size
template <typename tuple_t, int major_dim>
NVSHMEMI_HOSTDEVICE_PREFIX __forceinline__ bool is_less_than(const tuple_t& start_coord,
                                                             const tuple5Int_t& elem_coord,
                                                             const tuple_t& boundary,
                                                             int vlen = 1) {
    static_assert(tuple_size_traits<tuple_t>::value <= 2,
                  "Tuples with more than 2 dimensions are not supported");

    return is_less_than_impl<tuple_t, major_dim>(start_coord, elem_coord, boundary, vlen);
}

#endif
