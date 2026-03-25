/*
 * Copyright (c) 2017-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

#ifndef REDUCE_DEVICE_CUH
#define REDUCE_DEVICE_CUH

#include <cuda_runtime.h>
#include "non_abi/device/pt-to-pt/proxy_device.cuh"
#include "non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh"
#include "non_abi/device/common/nvshmemi_common_device.cuh"
#include "non_abi/device/team/nvshmemi_team_defines.cuh"
#include "non_abi/device/common/nvshmemi_tile_utils.cuh"
#include "device_host/nvshmem_tensor.h"
#include "non_abi/nvshmem_build_options.h"
#if defined(NVSHMEM_ENABLE_ALL_DEVICE_INLINING) || defined(__NVSHMEM_NUMBA_SUPPORT__)
#include "non_abi/device/pt-to-pt/transfer_device.cuh"
#else
#include "non_abi/device/pt-to-pt/nvshmemi_transfer_api.cuh"
#endif
#include "non_abi/device/wait/nvshmemi_wait_until_apis.cuh"
#include "utils.cuh"
#include "fcollect.cuh"
#include "broadcast.cuh"

#ifdef __CUDACC__
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;
#endif

#ifdef __CUDA_ARCH__

#define GPU_BITS_COPY_THREADGROUP_DIRECT(TYPENAME, TYPE, dest, src, nelems, myIdx, groupSize) \
    do {                                                                                      \
        int i;                                                                                \
        for (i = myIdx; i < nelems; i += groupSize) {                                         \
            *((TYPE *)dest + i) = *((TYPE *)src + i);                                         \
        }                                                                                     \
    } while (0)

template <typename T, rdxn_ops_t op>
#if !defined __CUDACC_RTC__
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE
    typename std::enable_if<std::is_integral<T>::value, T>::type
    perform_gpu_rdxn(T op1, T op2) {
#else
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE
    typename cuda::std::enable_if<cuda::std::is_integral<T>::value, T>::type
    perform_gpu_rdxn(T op1, T op2) {
#endif
    switch (op) {
        case RDXN_OPS_SUM:
            return op1 + op2;
        case RDXN_OPS_PROD:
            return op1 * op2;
        case RDXN_OPS_AND:
            return op1 & op2;
        case RDXN_OPS_OR:
            return op1 | op2;
        case RDXN_OPS_XOR:
            return op1 ^ op2;
        case RDXN_OPS_MIN:
            return (op1 < op2) ? op1 : op2;
        case RDXN_OPS_MAX:
            return (op1 > op2) ? op1 : op2;
        default:
            printf("Unsupported rdxn op\n");
            assert(0);
            return T();
    }
}

template <typename T, rdxn_ops_t op>
#if !defined __CUDACC_RTC__
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE
    typename std::enable_if<!std::is_integral<T>::value, T>::type
    perform_gpu_rdxn(T op1, T op2) {
#else
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE
    typename cuda::std::enable_if<!cuda::std::is_integral<T>::value, T>::type
    perform_gpu_rdxn(T op1, T op2) {
#endif
    switch (op) {
        case RDXN_OPS_SUM:
            return op1 + op2;
        case RDXN_OPS_PROD:
            return op1 * op2;
        case RDXN_OPS_MIN:
            return (op1 < op2) ? op1 : op2;
        case RDXN_OPS_MAX:
            return (op1 > op2) ? op1 : op2;
        default:
            printf("Unsupported rdxn op\n");
            assert(0);
            return T();
    }
}

template <>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE double2
perform_gpu_rdxn<double2, RDXN_OPS_MAXLOC>(double2 op1, double2 op2) {
    return (op1.x > op2.x) ? op1 : op2;
}

template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE __device__ void gpu_linear_reduce_threadgroup(
    TYPE *x, TYPE *y, TYPE *z, size_t nelems) {
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    int i;
    for (i = myIdx; i < nelems; i += groupSize) {
        z[i] = perform_gpu_rdxn<TYPE, OP>(x[i], y[i]);
    }
}

#define NVSHMEMI_MCAST_PTX_REG_TYPE_u32 "r"
#define NVSHMEMI_MCAST_PTX_REG_TYPE_b32 "r"
#define NVSHMEMI_MCAST_PTX_REG_TYPE_s32 "r"
#define NVSHMEMI_MCAST_PTX_REG_TYPE_f32 "f"
#define NVSHMEMI_MCAST_PTX_REG_TYPE_u64 "l"
#define NVSHMEMI_MCAST_PTX_REG_TYPE_b64 "l"
#define NVSHMEMI_MCAST_PTX_REG_TYPE_s64 "l"
#define NVSHMEMI_MCAST_PTX_REG_TYPE_f64 "d"
#define NVSHMEMI_MCAST_PTX_REG_TYPE_f16x2 "r"
#define NVSHMEMI_MCAST_PTX_REG_TYPE_bf16x2 "r"

#define NVSHMEMI_MCAST_ADD_MIXOP_f16x2 "add.acc::f32"
#define NVSHMEMI_MCAST_ADD_MIXOP_bf16x2 "add.acc::f32"
#define NVSHMEMI_MCAST_ADD_MIXOP_f32 "add"
#define NVSHMEMI_MCAST_MIN_MIXOP_f16x2 "min.acc::f32"
#define NVSHMEMI_MCAST_MIN_MIXOP_bf16x2 "min.acc::f32"
#define NVSHMEMI_MCAST_MIN_MIXOP_f32 "min"
#define NVSHMEMI_MCAST_MAX_MIXOP_f16x2 "max.acc::f32"
#define NVSHMEMI_MCAST_MAX_MIXOP_bf16x2 "max.acc::f32"
#define NVSHMEMI_MCAST_MAX_MIXOP_f32 "max"

// mcast ldreduce+multimem.st of 16B
// The requirement to use these primitives is that nelems % UNROLL == 0
#define NVSHMEMI_MCAST16_REDUCE_THREADGROUP_SUM_V4(PTX_TYPE)                                    \
    template <threadgroup_t SCOPE, int UNROLL, bool ONESHOT>                                    \
    __device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void                                               \
        nvshmemi_##PTX_TYPE##_add_reduce_mcast16_v4_threadgroup(int4 *dest, const int4 *source, \
                                                                size_t nelems) {                \
        int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();                                 \
        int groupSize = nvshmemi_threadgroup_size<SCOPE>();                                     \
        for (size_t j = myIdx * UNROLL; j < nelems; j += groupSize * UNROLL) {                  \
            uint32_t u4[4 * UNROLL];                                                            \
            _Pragma("unroll UNROLL") for (int u = 0; u < UNROLL; u++) {                         \
                asm("multimem.ld_reduce.global." NVSHMEMI_MCAST_ADD_MIXOP_##PTX_TYPE            \
                    ".v4." #PTX_TYPE " {%0, %1, %2, %3}, [%4];"                                 \
                    : "=r"(u4[4 * u]), "=r"(u4[4 * u + 1]), "=r"(u4[4 * u + 2]),                \
                      "=r"(u4[4 * u + 3])                                                       \
                    : "l"(source + j + u));                                                     \
            }                                                                                   \
            _Pragma("unroll UNROLL") for (int u = 0; u < UNROLL; u++) {                         \
                if (ONESHOT) {                                                                  \
                    asm("st.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(dest + j + u),         \
                        "r"(u4[4 * u]), "r"(u4[4 * u + 1]), "r"(u4[4 * u + 2]),                 \
                        "r"(u4[4 * u + 3]));                                                    \
                } else {                                                                        \
                    asm("multimem.st.global.v4." #PTX_TYPE                                      \
                        " [%0], {%1, %2, %3, %4};" ::"l"(dest + j + u),                         \
                        "r"(u4[4 * u]), "r"(u4[4 * u + 1]), "r"(u4[4 * u + 2]),                 \
                        "r"(u4[4 * u + 3]));                                                    \
                }                                                                               \
            }                                                                                   \
        }                                                                                       \
    }
// mcast ldreduce+multimem.st of 8B
#define NVSHMEMI_MCAST8_REDUCE_THREADGROUP_SUM_V2(CXX_TYPE, PTX_TYPE)                      \
    template <threadgroup_t SCOPE, bool ONESHOT>                                           \
    __device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void                                          \
        nvshmemi_##PTX_TYPE##_add_reduce_mcast8_v2_threadgroup(                            \
            uint64_t *dest, const uint64_t *source, size_t nelems) {                       \
        int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();                            \
        int groupSize = nvshmemi_threadgroup_size<SCOPE>();                                \
        for (size_t j = myIdx; j < nelems; j += groupSize) {                               \
            CXX_TYPE val1[2];                                                              \
            asm("multimem.ld_reduce.global." NVSHMEMI_MCAST_ADD_MIXOP_##PTX_TYPE           \
                ".v2." #PTX_TYPE " {%0, %1}, [%2];"                                        \
                : "=" NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1[0]),                     \
                  "=" NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1[1])                      \
                : "l"(source + j));                                                        \
            if (ONESHOT)                                                                   \
                asm("st.global.v2.b32 [%0], {%1, %2};" ::"l"(dest + j),                    \
                    NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1[0]),                       \
                    NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1[1]));                      \
            else                                                                           \
                asm("multimem.st.global.v2." #PTX_TYPE " [%0], {%1, %2};" ::"l"(dest + j), \
                    NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1[0]),                       \
                    NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1[1]));                      \
        }                                                                                  \
    }

#define NVSHMEMI_MCAST4_REDUCE_THREADGROUP(OP, CXX_TYPE, PTX_TYPE)                             \
    template <threadgroup_t SCOPE, bool ONESHOT>                                               \
    __device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void                                              \
        nvshmemi_##PTX_TYPE##_##OP##_reduce_mcast4_threadgroup(                                \
            uint32_t *dest, const uint32_t *source, size_t nelems) {                           \
        int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();                                \
        int groupSize = nvshmemi_threadgroup_size<SCOPE>();                                    \
        for (size_t j = myIdx; j < nelems; j += groupSize) {                                   \
            CXX_TYPE val1;                                                                     \
            asm("multimem.ld_reduce.global." NVSHMEMI_MCAST_ADD_MIXOP_##PTX_TYPE "." #PTX_TYPE \
                                                                                 " %0, [%1];"  \
                : "=" NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1)                             \
                : "l"(source + j));                                                            \
            if (ONESHOT)                                                                       \
                asm("st.global.b32 [%0], %1;" ::"l"(dest + j),                                 \
                    NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1));                             \
            else                                                                               \
                asm("multimem.st.global." #PTX_TYPE " [%0], %1;" ::"l"(dest + j),              \
                    NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1));                             \
        }                                                                                      \
    }

/* nvshmemi_<PTX_TYPE>_add_reduce_mcast16_v4_threadgroup(int4 *dest,const int4 *source, size_t
 * nelems) distributes contiguous "nelems" elements across the threadgroup. For tile collective,
 * input is a tile (often strided along a dimension), calling the above function along the contigous
 * dimension repeatedly will underutilize the threads so using a dedicated function
 */
#define NVSHMEMI_MCAST_TILE_ALLREDUCE_THREADGROUP(CXX_TYPE, PTX_TYPE, OP_TYPE)                     \
    template <typename elemType, threadgroup_t SCOPE, typename tuple_t, int UNROLL, int ONESHOT,   \
              int major_dim, int minor_dim>                                                        \
    __device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void                                                  \
        nvshmemi_##PTX_TYPE##_tile_allreduce##OP_TYPE##_mcast_threadgroup_v4(                      \
            int4 *dest, const int4 *source, const int nelem_major_dim, const int nelem_minor_dim,  \
            const int src_stride_minor_dim, const int dst_stride_minor_dim,                        \
            const int src_stride_major_dim, const int dst_stride_major_dim, tuple_t start_coord,   \
            tuple_t boundary) {                                                                    \
        /*src_stride_major_dim == 1 && dst_stride_major_dim == 1 for vectorized implementation*/   \
        int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();                                    \
        int groupSize = nvshmemi_threadgroup_size<SCOPE>();                                        \
        int nelems = nelem_major_dim * nelem_minor_dim; /* # vec elems*/                           \
        using vtype = int4;                                                                        \
        if constexpr (cuda::std::is_empty<tuple_t>::value) {                                       \
            /* If no predicate, we vectorize the operation */                                      \
            for (size_t j = myIdx * UNROLL; j < nelems; j += groupSize * UNROLL) {                 \
                uint32_t u4[4 * UNROLL];                                                           \
                _Pragma("unroll UNROLL") for (int u = 0; u < UNROLL; u++) {                        \
                    asm("multimem.ld_reduce.global." NVSHMEMI_MCAST_##OP_TYPE##_MIXOP_##PTX_TYPE   \
                        ".v4." #PTX_TYPE " {%0, %1, %2, %3}, [%4];"                                \
                        : "=r"(u4[4 * u]), "=r"(u4[4 * u + 1]), "=r"(u4[4 * u + 2]),               \
                          "=r"(u4[4 * u + 3])                                                      \
                        : "l"(source + ((j + u) % nelem_major_dim) +                               \
                              ((j + u) / nelem_major_dim) * src_stride_minor_dim));                \
                }                                                                                  \
                _Pragma("unroll UNROLL") for (int u = 0; u < UNROLL; u++) {                        \
                    if (ONESHOT) {                                                                 \
                        asm("st.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(                      \
                                dest + ((j + u) % nelem_major_dim) +                               \
                                (((j + u) / nelem_major_dim) * dst_stride_minor_dim)),             \
                            "r"(u4[4 * u]), "r"(u4[4 * u + 1]), "r"(u4[4 * u + 2]),                \
                            "r"(u4[4 * u + 3]));                                                   \
                    } else {                                                                       \
                        asm("multimem.st.global.v4." #PTX_TYPE " [%0], {%1, %2, %3, %4};" ::"l"(   \
                                dest + ((j + u) % nelem_major_dim) +                               \
                                (((j + u) / nelem_major_dim) * dst_stride_minor_dim)),             \
                            "r"(u4[4 * u]), "r"(u4[4 * u + 1]), "r"(u4[4 * u + 2]),                \
                            "r"(u4[4 * u + 3]));                                                   \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
        } else {                                                                                   \
            /* if predicate is provided, we use UNROLL == 1 to prevent repeated computation of     \
             * pred*/                                                                              \
            for (size_t j = myIdx; j < nelems; j += groupSize) {                                   \
                uint32_t u4[4];                                                                    \
                /* nelem_major_dim is in vector units*/                                            \
                uint32_t elem_coord_major =                                                        \
                    (j % nelem_major_dim) * (sizeof(vtype) / sizeof(elemType));                    \
                uint32_t elem_coord_minor = (j / nelem_major_dim);                                 \
                                                                                                   \
                /* start_coord, boundary are in elemType units */                                  \
                /* Check if entire vector is within boundary */                                    \
                /* start_coord_major_dim + elem_coord_major_dim + vector len (in elements) <=      \
                 * boundary_major_dim */                                                           \
                if (is_less_than<tuple_t, major_dim>(                                              \
                        start_coord,                                                               \
                        create_coord_tuple<major_dim>(elem_coord_major, elem_coord_minor),         \
                        boundary, (sizeof(vtype) / sizeof(elemType)))) {                           \
                    /* entire vector is within boundary*/                                          \
                    asm("multimem.ld_reduce.global." NVSHMEMI_MCAST_##OP_TYPE##_MIXOP_##PTX_TYPE   \
                        ".v4." #PTX_TYPE " {%0, %1, %2, %3}, [%4];"                                \
                        : "=r"(u4[0]), "=r"(u4[1]), "=r"(u4[2]), "=r"(u4[3])                       \
                        : "l"(source + ((j) % nelem_major_dim) +                                   \
                              ((j) / nelem_major_dim) * src_stride_minor_dim));                    \
                                                                                                   \
                    if (ONESHOT) {                                                                 \
                        asm("st.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(                      \
                                dest + (j % nelem_major_dim) +                                     \
                                ((j / nelem_major_dim) * dst_stride_minor_dim)),                   \
                            "r"(u4[0]), "r"(u4[1]), "r"(u4[2]), "r"(u4[3]));                       \
                    } else {                                                                       \
                        asm("multimem.st.global.v4." #PTX_TYPE " [%0], {%1, %2, %3, %4};" ::"l"(   \
                                dest + (j % nelem_major_dim) +                                     \
                                ((j / nelem_major_dim) * dst_stride_minor_dim)),                   \
                            "r"(u4[0]), "r"(u4[1]), "r"(u4[2]), "r"(u4[3]));                       \
                    }                                                                              \
                } else {                                                                           \
                    /* vector is partially within boundary, check each element */                  \
                    CXX_TYPE val;                                                                  \
                    /* convert elem_coord_major from elemType to CXX_TYPE units */                 \
                    /* no change to elem_coord_minor */                                            \
                    elem_coord_major = (elem_coord_major * sizeof(elemType)) / sizeof(CXX_TYPE);   \
                    for (int u = 0; u < sizeof(vtype) / sizeof(CXX_TYPE); ++u) {                   \
                        /* check if elem is within boundary, use u & elem_coord_major in elemType  \
                         * units */                                                                \
                        if (is_less_than<tuple_t, major_dim>(                                      \
                                start_coord,                                                       \
                                create_coord_tuple<major_dim>(                                     \
                                    ((elem_coord_major + u) * sizeof(CXX_TYPE) /                   \
                                     sizeof(elemType)),                                            \
                                    elem_coord_minor),                                             \
                                boundary)) {                                                       \
                            /* convert strides from vector to CXX_TYPE units */                    \
                            asm("multimem.ld_reduce."                                              \
                                "global." NVSHMEMI_MCAST_##OP_TYPE##_MIXOP_##PTX_TYPE              \
                                "." #PTX_TYPE " %0, [%1];"                                         \
                                : "=" NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val)                  \
                                : "l"(reinterpret_cast<const CXX_TYPE *>(source) +                 \
                                      (elem_coord_major + u) +                                     \
                                      (elem_coord_minor * src_stride_minor_dim *                   \
                                       (sizeof(vtype) / sizeof(CXX_TYPE)))));                      \
                            if (ONESHOT)                                                           \
                                asm("st.global.b32 [%0], %1;" ::"l"(                               \
                                        reinterpret_cast<CXX_TYPE *>(dest) +                       \
                                        +(elem_coord_major + u) +                                  \
                                        +(elem_coord_minor * dst_stride_minor_dim *                \
                                          (sizeof(vtype) / sizeof(CXX_TYPE)))),                    \
                                    NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val));                  \
                            else                                                                   \
                                asm("multimem.st.global." #PTX_TYPE                                \
                                    " [%0], %1;" ::"l"(reinterpret_cast<CXX_TYPE *>(dest) +        \
                                                       (elem_coord_major + u) +                    \
                                                       (elem_coord_minor * dst_stride_minor_dim *  \
                                                        (sizeof(vtype) / sizeof(CXX_TYPE)))),      \
                                    NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val));                  \
                        }                                                                          \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
        } /*end of if else*/                                                                       \
    }                                                                                              \
                                                                                                   \
    /* **************  Vector len = 2 specialization ************** */                             \
    template <typename elemType, threadgroup_t SCOPE, typename tuple_t, int ONESHOT,               \
              int major_dim, int minor_dim>                                                        \
    __device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void                                                  \
        nvshmemi_##PTX_TYPE##_tile_allreduce##OP_TYPE##_mcast_threadgroup_v2(                      \
            uint64_t *dest, const uint64_t *source, const int nelem_major_dim,                     \
            const int nelem_minor_dim, const int src_stride_minor_dim,                             \
            const int dst_stride_minor_dim, const int src_stride_major_dim,                        \
            const int dst_stride_major_dim, tuple_t start_coord, tuple_t boundary) {               \
        using vtype = uint64_t;                                                                    \
        /*src_stride_major_dim == 0 && dst_stride_major_dim == 0 for vectorized implementation*/   \
        int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();                                    \
        int groupSize = nvshmemi_threadgroup_size<SCOPE>();                                        \
        int nelems = nelem_major_dim * nelem_minor_dim;                                            \
                                                                                                   \
        if constexpr (cuda::std::is_empty<tuple_t>::value) {                                       \
            /* If no predicate, we vectorize the operation */                                      \
            for (size_t j = myIdx; j < nelems; j += groupSize) {                                   \
                CXX_TYPE val1[2];                                                                  \
                asm("multimem.ld_reduce.global." NVSHMEMI_MCAST_##OP_TYPE##_MIXOP_##PTX_TYPE       \
                    ".v2." #PTX_TYPE " {%0, %1}, [%2];"                                            \
                    : "=" NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1[0]),                         \
                      "=" NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1[1])                          \
                    : "l"(source + (j % nelem_major_dim) +                                         \
                          ((j / nelem_major_dim) * src_stride_minor_dim)));                        \
                if (ONESHOT)                                                                       \
                    asm("st.global.v2.b32 [%0], {%1, %2};" ::"l"(                                  \
                            dest + (j % nelem_major_dim) +                                         \
                            ((j / nelem_major_dim) * dst_stride_minor_dim)),                       \
                        NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1[0]),                           \
                        NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1[1]));                          \
                else                                                                               \
                    asm("multimem.st.global.v2." #PTX_TYPE                                         \
                        " [%0], {%1, %2};" ::"l"(dest + (j % nelem_major_dim) +                    \
                                                 ((j / nelem_major_dim) * dst_stride_minor_dim)),  \
                        NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1[0]),                           \
                        NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1[1]));                          \
            }                                                                                      \
        } else {                                                                                   \
            for (size_t j = myIdx; j < nelems; j += groupSize) {                                   \
                CXX_TYPE val1[2];                                                                  \
                /* nelem_major_dim is in vector units, convert to elemType units*/                 \
                uint32_t elem_coord_major =                                                        \
                    (j % nelem_major_dim) * (sizeof(vtype) / sizeof(elemType));                    \
                uint32_t elem_coord_minor = (j / nelem_major_dim);                                 \
                                                                                                   \
                /* start_coord, boundary are in elemType units */                                  \
                /* Check if entire vector is within boundary */                                    \
                /* start_coord_major_dim + elem_coord_major_dim + vector len (in elements) <=      \
                 * boundary_major_dim */                                                           \
                if (is_less_than<tuple_t, major_dim>(                                              \
                        start_coord,                                                               \
                        create_coord_tuple<major_dim>(elem_coord_major, elem_coord_minor),         \
                        boundary, (sizeof(vtype) / sizeof(elemType)))) {                           \
                    asm("multimem.ld_reduce.global." NVSHMEMI_MCAST_##OP_TYPE##_MIXOP_##PTX_TYPE   \
                        ".v2." #PTX_TYPE " {%0, %1}, [%2];"                                        \
                        : "=" NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1[0]),                     \
                          "=" NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1[1])                      \
                        : "l"(source + (j % nelem_major_dim) +                                     \
                              ((j / nelem_major_dim) * src_stride_minor_dim)));                    \
                    if (ONESHOT)                                                                   \
                        asm("st.global.v2.b32 [%0], {%1, %2};" ::"l"(                              \
                                dest + (j % nelem_major_dim) +                                     \
                                ((j / nelem_major_dim) * dst_stride_minor_dim)),                   \
                            NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1[0]),                       \
                            NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1[1]));                      \
                    else                                                                           \
                        asm("multimem.st.global.v2." #PTX_TYPE " [%0], {%1, %2};" ::"l"(           \
                                dest + (j % nelem_major_dim) +                                     \
                                ((j / nelem_major_dim) * dst_stride_minor_dim)),                   \
                            NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1[0]),                       \
                            NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1[1]));                      \
                } else {                                                                           \
                    /* convert elem_coord_major from elemType to CXX_TYPE units */                 \
                    /* no change to elem_coord_minor */                                            \
                    elem_coord_major = (elem_coord_major * sizeof(elemType)) / sizeof(CXX_TYPE);   \
                    /* vector is partially within boundary, check each element */                  \
                    for (int u = 0; u < sizeof(vtype) / sizeof(CXX_TYPE); ++u) {                   \
                        /* check if elem is within boundary, use u & elem_coord_major in elemType  \
                         * units */                                                                \
                        if (is_less_than<tuple_t, major_dim>(                                      \
                                start_coord,                                                       \
                                create_coord_tuple<major_dim>(                                     \
                                    ((elem_coord_major + u) * sizeof(CXX_TYPE) /                   \
                                     sizeof(elemType)),                                            \
                                    elem_coord_minor),                                             \
                                boundary)) {                                                       \
                            /* convert strides from vector to CXX_TYPE units */                    \
                            asm("multimem.ld_reduce."                                              \
                                "global." NVSHMEMI_MCAST_##OP_TYPE##_MIXOP_##PTX_TYPE              \
                                "." #PTX_TYPE " %0, [%1];"                                         \
                                : "=" NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1[0])              \
                                : "l"(reinterpret_cast<const CXX_TYPE *>(source) +                 \
                                      (elem_coord_major + u) +                                     \
                                      (elem_coord_minor * src_stride_minor_dim *                   \
                                       (sizeof(vtype) / sizeof(CXX_TYPE)))));                      \
                            if (ONESHOT)                                                           \
                                asm("st.global.b32 [%0], %1;" ::"l"(                               \
                                        reinterpret_cast<CXX_TYPE *>(dest) +                       \
                                        +(elem_coord_major + u) +                                  \
                                        +(elem_coord_minor * dst_stride_minor_dim *                \
                                          (sizeof(vtype) / sizeof(CXX_TYPE)))),                    \
                                    NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1[0]));              \
                            else                                                                   \
                                asm("multimem.st.global." #PTX_TYPE                                \
                                    " [%0], %1;" ::"l"(reinterpret_cast<CXX_TYPE *>(dest) +        \
                                                       (elem_coord_major + u) +                    \
                                                       (elem_coord_minor * dst_stride_minor_dim *  \
                                                        (sizeof(vtype) / sizeof(CXX_TYPE)))),      \
                                    NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1[0]));              \
                        }                                                                          \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
    }                                                                                              \
    /* **************  Vector len = 1 specialization ************** */                             \
    /* TODO: for majorStride != 1, this will cause error as we do f32 reduce and not f16 */        \
    template <typename elemType, threadgroup_t SCOPE, typename tuple_t, int ONESHOT,               \
              int major_dim, int minor_dim>                                                        \
    __device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void                                                  \
        nvshmemi_##PTX_TYPE##_tile_allreduce##OP_TYPE##_mcast_threadgroup_v1(                      \
            uint32_t *dest, const uint32_t *source, const int nelem_major_dim,                     \
            const int nelem_minor_dim, const int src_stride_minor_dim,                             \
            const int dst_stride_minor_dim, const int src_stride_major_dim,                        \
            const int dst_stride_major_dim, tuple_t start_coord, tuple_t boundary) {               \
        using vtype = uint32_t;                                                                    \
        /* This variant supports strides along both major and minor dimensions */                  \
        int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();                                    \
        int groupSize = nvshmemi_threadgroup_size<SCOPE>();                                        \
        int nelems = nelem_major_dim * nelem_minor_dim;                                            \
        if constexpr (cuda::std::is_empty<tuple_t>::value) {                                       \
            /* Case: no predicate */                                                               \
            for (size_t j = myIdx; j < nelems; j += groupSize) {                                   \
                CXX_TYPE val1;                                                                     \
                asm("multimem.ld_reduce.global." NVSHMEMI_MCAST_##OP_TYPE##_MIXOP_##PTX_TYPE       \
                    "." #PTX_TYPE " %0, [%1];"                                                     \
                    : "=" NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1)                             \
                    : "l"(source + ((j % nelem_major_dim) * src_stride_major_dim) +                \
                          ((j / nelem_major_dim) * src_stride_minor_dim)));                        \
                if (ONESHOT)                                                                       \
                    asm("st.global.b32 [%0], %1;" ::"l"(                                           \
                            dest + ((j % nelem_major_dim) * dst_stride_major_dim) +                \
                            ((j / nelem_major_dim) * dst_stride_minor_dim)),                       \
                        NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1));                             \
                else                                                                               \
                    asm("multimem.st.global." #PTX_TYPE                                            \
                        " [%0], %1;" ::"l"(dest + ((j % nelem_major_dim) * dst_stride_major_dim) + \
                                           ((j / nelem_major_dim) * dst_stride_minor_dim)),        \
                        NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1));                             \
            }                                                                                      \
        } else {                                                                                   \
            for (size_t j = myIdx; j < nelems; j += groupSize) {                                   \
                uint32_t elem_coord_major =                                                        \
                    (j % nelem_major_dim) * (sizeof(vtype) / sizeof(elemType));                    \
                uint32_t elem_coord_minor = (j / nelem_major_dim);                                 \
                CXX_TYPE val1;                                                                     \
                /* check if elem is within boundary, convert u to elemType units */                \
                /* for major stride > 1, we can't do f32 for half/bfloat */                        \
                if (is_less_than<tuple_t, major_dim>(                                              \
                        start_coord,                                                               \
                        create_coord_tuple<major_dim>(elem_coord_major, elem_coord_minor),         \
                        boundary)) {                                                               \
                    /* convert elem_coord_major from elemType to CXX_TYPE units */                 \
                    /* no change to elem_coord_minor */                                            \
                    elem_coord_major = (elem_coord_major * sizeof(elemType)) / sizeof(CXX_TYPE);   \
                                                                                                   \
                    /* convert strides from vector to CXX_TYPE units */                            \
                    asm("multimem.ld_reduce.global." NVSHMEMI_MCAST_##OP_TYPE##_MIXOP_##PTX_TYPE   \
                        "." #PTX_TYPE " %0, [%1];"                                                 \
                        : "=" NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1)                         \
                        : "l"(reinterpret_cast<const CXX_TYPE *>(source) +                         \
                              (elem_coord_major * src_stride_major_dim) +                          \
                              (elem_coord_minor * src_stride_minor_dim *                           \
                               (sizeof(vtype) / sizeof(CXX_TYPE)))));                              \
                    if (ONESHOT)                                                                   \
                        asm("st.global.b32 [%0], %1;" ::"l"(                                       \
                                reinterpret_cast<CXX_TYPE *>(dest) +                               \
                                +(elem_coord_major * dst_stride_major_dim) +                       \
                                +(elem_coord_minor * dst_stride_minor_dim *                        \
                                  (sizeof(vtype) / sizeof(CXX_TYPE)))),                            \
                            NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1));                         \
                    else                                                                           \
                        asm("multimem.st.global." #PTX_TYPE                                        \
                            " [%0], %1;" ::"l"(reinterpret_cast<CXX_TYPE *>(dest) +                \
                                               (elem_coord_major * dst_stride_major_dim) +         \
                                               (elem_coord_minor * dst_stride_minor_dim *          \
                                                (sizeof(vtype) / sizeof(CXX_TYPE)))),              \
                            NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1));                         \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
    }                                                                                              \
    template <typename vtype, typename elemType, threadgroup_t SCOPE, typename tuple_t,            \
              int ONESHOT, int major_dim, int minor_dim>                                           \
    __device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void                                                  \
        nvshmemi_##PTX_TYPE##_tile_allreduce##OP_TYPE##_mcast_threadgroup(                         \
            vtype *dest, const vtype *source, const int nelem_major_dim,                           \
            const int nelem_minor_dim, const int src_stride_minor_dim,                             \
            const int dst_stride_minor_dim, const int src_stride_major_dim,                        \
            const int dst_stride_major_dim, tuple_t start_coord, tuple_t boundary) {               \
        if constexpr (cuda::std::is_same<vtype, int4>::value) {                                    \
            nvshmemi_##PTX_TYPE##_tile_allreduce##OP_TYPE##_mcast_threadgroup_v4<                  \
                elemType, SCOPE, tuple_t, 1, ONESHOT, major_dim, minor_dim>(                       \
                dest, source, nelem_major_dim, nelem_minor_dim, src_stride_minor_dim,              \
                dst_stride_minor_dim, src_stride_major_dim, dst_stride_major_dim, start_coord,     \
                boundary);                                                                         \
        } else if constexpr (cuda::std::is_same<vtype, uint64_t>::value) {                         \
            nvshmemi_##PTX_TYPE##_tile_allreduce##OP_TYPE##_mcast_threadgroup_v2<                  \
                elemType, SCOPE, tuple_t, ONESHOT, major_dim, minor_dim>(                          \
                dest, source, nelem_major_dim, nelem_minor_dim, src_stride_minor_dim,              \
                dst_stride_minor_dim, src_stride_major_dim, dst_stride_major_dim, start_coord,     \
                boundary);                                                                         \
        } else if constexpr (cuda::std::is_same<vtype, uint32_t>::value) {                         \
            nvshmemi_##PTX_TYPE##_tile_allreduce##OP_TYPE##_mcast_threadgroup_v1<                  \
                elemType, SCOPE, tuple_t, ONESHOT, major_dim, minor_dim>(                          \
                dest, source, nelem_major_dim, nelem_minor_dim, src_stride_minor_dim,              \
                dst_stride_minor_dim, src_stride_major_dim, dst_stride_major_dim, start_coord,     \
                boundary);                                                                         \
        } else {                                                                                   \
            if (cuda::std::is_same<vtype, int4>::value) {                                          \
                nvshmemi_##PTX_TYPE##_tile_allreduce##OP_TYPE##_mcast_threadgroup_v4<              \
                    elemType, SCOPE, tuple_t, 1, ONESHOT, major_dim, minor_dim>(                   \
                    dest, source, nelem_major_dim, nelem_minor_dim, src_stride_minor_dim,          \
                    dst_stride_minor_dim, src_stride_major_dim, dst_stride_major_dim, start_coord, \
                    boundary);                                                                     \
            } else if (cuda::std::is_same<vtype, uint64_t>::value) {                               \
                nvshmemi_##PTX_TYPE##_tile_allreduce##OP_TYPE##_mcast_threadgroup_v2<              \
                    elemType, SCOPE, tuple_t, ONESHOT, major_dim, minor_dim>(                      \
                    dest, source, nelem_major_dim, nelem_minor_dim, src_stride_minor_dim,          \
                    dst_stride_minor_dim, src_stride_major_dim, dst_stride_major_dim, start_coord, \
                    boundary);                                                                     \
            } else if (cuda::std::is_same<vtype, uint32_t>::value) {                               \
                nvshmemi_##PTX_TYPE##_tile_allreduce##OP_TYPE##_mcast_threadgroup_v1<              \
                    elemType, SCOPE, tuple_t, ONESHOT, major_dim, minor_dim>(                      \
                    dest, source, nelem_major_dim, nelem_minor_dim, src_stride_minor_dim,          \
                    dst_stride_minor_dim, src_stride_major_dim, dst_stride_major_dim, start_coord, \
                    boundary);                                                                     \
            } else {                                                                               \
                printf("Unsupported vector len %lu\n", sizeof(vtype));                             \
                assert(0 && "Unsupported vtype");                                                  \
            }                                                                                      \
        }                                                                                          \
    }

// mcast ldreduce+st of 16B
// The requirement to use these primitives is that nelems % UNROLL == 0
#define NVSHMEMI_MCAST16_LOCAL_REDUCE_THREADGROUP_SUM_V4(PTX_TYPE)                               \
    template <threadgroup_t SCOPE, int UNROLL>                                                   \
    __device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void                                                \
        nvshmemi_##PTX_TYPE##_add_local_reduce_mcast16_v4_threadgroup(                           \
            int4 *dest, const int4 *source, size_t nelems) {                                     \
        int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();                                  \
        int groupSize = nvshmemi_threadgroup_size<SCOPE>();                                      \
        for (size_t j = myIdx * UNROLL; j < nelems; j += groupSize * UNROLL) {                   \
            uint32_t u4[4 * UNROLL];                                                             \
            _Pragma("unroll UNROLL") for (int u = 0; u < UNROLL; u++) {                          \
                asm("multimem.ld_reduce.global." NVSHMEMI_MCAST_ADD_MIXOP_##PTX_TYPE             \
                    ".v4." #PTX_TYPE " {%0, %1, %2, %3}, [%4];"                                  \
                    : "=r"(u4[4 * u]), "=r"(u4[4 * u + 1]), "=r"(u4[4 * u + 2]),                 \
                      "=r"(u4[4 * u + 3])                                                        \
                    : "l"(source + j + u));                                                      \
            }                                                                                    \
            _Pragma("unroll UNROLL") for (int u = 0; u < UNROLL; u++) {                          \
                asm("st.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(dest + j + u),              \
                    "r"(u4[4 * u]), "r"(u4[4 * u + 1]), "r"(u4[4 * u + 2]), "r"(u4[4 * u + 3])); \
            }                                                                                    \
        }                                                                                        \
    }

// mcast ldreduce+st of 8B
#define NVSHMEMI_MCAST8_LOCAL_REDUCE_THREADGROUP_MINMAX(OP, CXX_TYPE, PTX_TYPE) \
    template <typename TYPE, threadgroup_t SCOPE>                               \
    __device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void                               \
        nvshmemi_##PTX_TYPE##_##OP##_local_reduce_mcast8_threadgroup(           \
            uint64_t *dest, const uint64_t *source, size_t nelems) {            \
        int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();                 \
        int groupSize = nvshmemi_threadgroup_size<SCOPE>();                     \
        for (size_t j = myIdx; j < nelems; j += groupSize) {                    \
            CXX_TYPE val1;                                                      \
            asm("multimem.ld_reduce.global." #OP "." #PTX_TYPE " %0, [%1];"     \
                : "=" NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1)              \
                : "l"(source + j));                                             \
            asm("st.global.b64 [%0], %1;" ::"l"(dest + j),                      \
                NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1));                  \
        }                                                                       \
    }

#define NVSHMEMI_MCAST8_LOCAL_REDUCE_THREADGROUP_SUM(CXX_TYPE, PTX_TYPE) \
    template <typename TYPE, threadgroup_t SCOPE>                        \
    __device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void                        \
        nvshmemi_##PTX_TYPE##_add_local_reduce_mcast8_threadgroup(       \
            uint64_t *dest, const uint64_t *source, size_t nelems) {     \
        int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();          \
        int groupSize = nvshmemi_threadgroup_size<SCOPE>();              \
        for (size_t j = myIdx; j < nelems; j += groupSize) {             \
            CXX_TYPE val1;                                               \
            asm("multimem.ld_reduce.global.add." #PTX_TYPE " %0, [%1];"  \
                : "=" NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1)       \
                : "l"(source + j));                                      \
            asm("st.global.b64 [%0], %1;" ::"l"(dest + j),               \
                NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1));           \
        }                                                                \
    }

#define NVSHMEMI_MCAST8_LOCAL_REDUCE_THREADGROUP(OP, CXX_TYPE, PTX_TYPE)    \
    template <typename TYPE, threadgroup_t SCOPE>                           \
    __device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void                           \
        nvshmemi_##PTX_TYPE##_##OP##_local_reduce_mcast8_threadgroup(       \
            uint64_t *dest, const uint64_t *source, size_t nelems) {        \
        int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();             \
        int groupSize = nvshmemi_threadgroup_size<SCOPE>();                 \
        for (size_t j = myIdx; j < nelems; j += groupSize) {                \
            CXX_TYPE val1;                                                  \
            asm("multimem.ld_reduce.global." #OP "." #PTX_TYPE " %0, [%1];" \
                : "=" NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1)          \
                : "l"(source + j));                                         \
            asm("st.global.b64 [%0], %1;" ::"l"(dest + j),                  \
                NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1));              \
        }                                                                   \
    }

#define NVSHMEMI_MCAST8_LOCAL_REDUCE_THREADGROUP_SUM_V2(CXX_TYPE, PTX_TYPE)      \
    template <typename TYPE, threadgroup_t SCOPE>                                \
    __device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void                                \
        nvshmemi_##PTX_TYPE##_add_local_reduce_mcast8_v2_threadgroup(            \
            uint64_t *dest, const uint64_t *source, size_t nelems) {             \
        int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();                  \
        int groupSize = nvshmemi_threadgroup_size<SCOPE>();                      \
        for (size_t j = myIdx; j < nelems; j += groupSize) {                     \
            CXX_TYPE val1[2];                                                    \
            asm("multimem.ld_reduce.global." NVSHMEMI_MCAST_ADD_MIXOP_##PTX_TYPE \
                ".v2." #PTX_TYPE " {%0, %1}, [%2];"                              \
                : "=" NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1[0]),           \
                  "=" NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1[1])            \
                : "l"(source + j));                                              \
            asm("st.global.v2.b32 [%0], {%1, %2};" ::"l"(dest + j),              \
                NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1[0]),                 \
                NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1[1]));                \
        }                                                                        \
    }

// mcast ldreduce+st of 4B
#define NVSHMEMI_MCAST4_LOCAL_REDUCE_THREADGROUP_SUM(CXX_TYPE, PTX_TYPE)                       \
    template <typename TYPE, threadgroup_t SCOPE>                                              \
    __device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void                                              \
        nvshmemi_##PTX_TYPE##_add_local_reduce_mcast4_threadgroup(                             \
            uint32_t *dest, const uint32_t *source, size_t nelems) {                           \
        int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();                                \
        int groupSize = nvshmemi_threadgroup_size<SCOPE>();                                    \
        for (size_t j = myIdx; j < nelems; j += groupSize) {                                   \
            CXX_TYPE val1;                                                                     \
            asm("multimem.ld_reduce.global." NVSHMEMI_MCAST_ADD_MIXOP_##PTX_TYPE "." #PTX_TYPE \
                                                                                 " %0, [%1];"  \
                : "=" NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1)                             \
                : "l"(source + j));                                                            \
            asm("st.global.b32 [%0], %1;" ::"l"(dest + j),                                     \
                NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1));                                 \
        }                                                                                      \
    }

#define NVSHMEMI_MCAST4_LOCAL_REDUCE_THREADGROUP(OP, CXX_TYPE, PTX_TYPE)    \
    template <typename TYPE, threadgroup_t SCOPE>                           \
    __device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void                           \
        nvshmemi_##PTX_TYPE##_##OP##_local_reduce_mcast4_threadgroup(       \
            uint32_t *dest, const uint32_t *source, size_t nelems) {        \
        int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();             \
        int groupSize = nvshmemi_threadgroup_size<SCOPE>();                 \
        for (size_t j = myIdx; j < nelems; j += groupSize) {                \
            CXX_TYPE val1;                                                  \
            asm("multimem.ld_reduce.global." #OP "." #PTX_TYPE " %0, [%1];" \
                : "=" NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1)          \
                : "l"(source + j));                                         \
            asm("st.global.b32 [%0], %1;" ::"l"(dest + j),                  \
                NVSHMEMI_MCAST_PTX_REG_TYPE_##PTX_TYPE(val1));              \
        }                                                                   \
    }

#define NVSHMEMI_MCAST_RDXN_OP_IS_CAP(OP) \
    (OP != RDXN_OPS_PROD && OP != RDXN_OPS_MAXLOC && OP != RDXN_OPS_sentinel)
#define NVSHMEMI_MCAST_RDXN_OP_IS_CAP_8B(OP) (NVSHMEMI_MCAST_RDXN_OP_IS_CAP(OP))
#define NVSHMEMI_MCAST_RDXN_OP_IS_CAP_16B(OP) (OP == RDXN_OPS_SUM)
#define NVSHMEMI_MCAST_RDXN_OP_IS_CAP_UNTYPED(OP) \
    (OP == RDXN_OPS_SUM || OP == RDXN_OPS_AND || OP == RDXN_OPS_OR || OP == RDXN_OPS_XOR)

/* Supported matrix here:
 * https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-multimem-ld-reduce-multimem-st-multimem-red
 */

/* 4B local reduce primitives */
NVSHMEMI_MCAST4_LOCAL_REDUCE_THREADGROUP(add, uint32_t, u32)
NVSHMEMI_MCAST4_LOCAL_REDUCE_THREADGROUP(add, int32_t, s32)
NVSHMEMI_MCAST4_LOCAL_REDUCE_THREADGROUP_SUM(float, f32)
NVSHMEMI_MCAST4_LOCAL_REDUCE_THREADGROUP_SUM(uint32_t, f16x2)
NVSHMEMI_MCAST4_LOCAL_REDUCE_THREADGROUP_SUM(uint32_t, bf16x2)
NVSHMEMI_MCAST4_LOCAL_REDUCE_THREADGROUP(min, uint32_t, u32)
NVSHMEMI_MCAST4_LOCAL_REDUCE_THREADGROUP(min, int32_t, s32)
NVSHMEMI_MCAST4_LOCAL_REDUCE_THREADGROUP(max, uint32_t, u32)
NVSHMEMI_MCAST4_LOCAL_REDUCE_THREADGROUP(max, int32_t, s32)
NVSHMEMI_MCAST4_LOCAL_REDUCE_THREADGROUP(and, uint32_t, b32)
NVSHMEMI_MCAST4_LOCAL_REDUCE_THREADGROUP(xor, uint32_t, b32)
NVSHMEMI_MCAST4_LOCAL_REDUCE_THREADGROUP(or, uint32_t, b32)

/* 8B local reduce primitives */
NVSHMEMI_MCAST8_LOCAL_REDUCE_THREADGROUP_SUM(uint64_t, u64)
NVSHMEMI_MCAST8_LOCAL_REDUCE_THREADGROUP_SUM(double, f64)
NVSHMEMI_MCAST8_LOCAL_REDUCE_THREADGROUP_SUM_V2(float, f32)
NVSHMEMI_MCAST8_LOCAL_REDUCE_THREADGROUP_SUM_V2(uint32_t, f16x2)
NVSHMEMI_MCAST8_LOCAL_REDUCE_THREADGROUP_SUM_V2(uint32_t, bf16x2)
NVSHMEMI_MCAST8_LOCAL_REDUCE_THREADGROUP_MINMAX(min, uint64_t, u64)
NVSHMEMI_MCAST8_LOCAL_REDUCE_THREADGROUP_MINMAX(min, int64_t, s64)
NVSHMEMI_MCAST8_LOCAL_REDUCE_THREADGROUP_MINMAX(max, uint64_t, u64)
NVSHMEMI_MCAST8_LOCAL_REDUCE_THREADGROUP_MINMAX(max, int64_t, s64)
NVSHMEMI_MCAST8_LOCAL_REDUCE_THREADGROUP(and, uint64_t, b64)
NVSHMEMI_MCAST8_LOCAL_REDUCE_THREADGROUP(xor, uint64_t, b64)
NVSHMEMI_MCAST8_LOCAL_REDUCE_THREADGROUP(or, uint64_t, b64)

/* 16B local reduce primitives */
NVSHMEMI_MCAST16_LOCAL_REDUCE_THREADGROUP_SUM_V4(f32)
NVSHMEMI_MCAST16_LOCAL_REDUCE_THREADGROUP_SUM_V4(f16x2)
NVSHMEMI_MCAST16_LOCAL_REDUCE_THREADGROUP_SUM_V4(bf16x2)

/* 4B-16B reduce (SUM) primitives */
NVSHMEMI_MCAST8_REDUCE_THREADGROUP_SUM_V2(float, f32)
NVSHMEMI_MCAST8_REDUCE_THREADGROUP_SUM_V2(uint32_t, f16x2)
NVSHMEMI_MCAST8_REDUCE_THREADGROUP_SUM_V2(uint32_t, bf16x2)
NVSHMEMI_MCAST4_REDUCE_THREADGROUP(add, float, f32)
NVSHMEMI_MCAST4_REDUCE_THREADGROUP(add, uint32_t, f16x2)
NVSHMEMI_MCAST4_REDUCE_THREADGROUP(add, uint32_t, bf16x2)
NVSHMEMI_MCAST16_REDUCE_THREADGROUP_SUM_V4(f32)
NVSHMEMI_MCAST16_REDUCE_THREADGROUP_SUM_V4(f16x2)
NVSHMEMI_MCAST16_REDUCE_THREADGROUP_SUM_V4(bf16x2)

// ld_reduce errors on using .acc for min, max
#undef NVSHMEMI_MCAST_MIN_MIXOP_f16x2
#undef NVSHMEMI_MCAST_MIN_MIXOP_bf16x2
#undef NVSHMEMI_MCAST_MAX_MIXOP_f16x2
#undef NVSHMEMI_MCAST_MAX_MIXOP_bf16x2
#define NVSHMEMI_MCAST_MIN_MIXOP_f16x2 "min"
#define NVSHMEMI_MCAST_MIN_MIXOP_bf16x2 "min"
#define NVSHMEMI_MCAST_MAX_MIXOP_f16x2 "max"
#define NVSHMEMI_MCAST_MAX_MIXOP_bf16x2 "max"
NVSHMEMI_MCAST_TILE_ALLREDUCE_THREADGROUP(float, f32, ADD)
NVSHMEMI_MCAST_TILE_ALLREDUCE_THREADGROUP(uint32_t, f16x2, ADD)
NVSHMEMI_MCAST_TILE_ALLREDUCE_THREADGROUP(uint32_t, bf16x2, ADD)
NVSHMEMI_MCAST_TILE_ALLREDUCE_THREADGROUP(uint32_t, f16x2, MIN)
NVSHMEMI_MCAST_TILE_ALLREDUCE_THREADGROUP(uint32_t, bf16x2, MIN)
NVSHMEMI_MCAST_TILE_ALLREDUCE_THREADGROUP(uint32_t, f16x2, MAX)
NVSHMEMI_MCAST_TILE_ALLREDUCE_THREADGROUP(uint32_t, bf16x2, MAX)
// resetting defines to ensure rest code can use it as is
#undef NVSHMEMI_MCAST_MIN_MIXOP_f16x2
#undef NVSHMEMI_MCAST_MIN_MIXOP_bf16x2
#undef NVSHMEMI_MCAST_MAX_MIXOP_f16x2
#undef NVSHMEMI_MCAST_MAX_MIXOP_bf16x2
#define NVSHMEMI_MCAST_MIN_MIXOP_f16x2 "min.acc::f32"
#define NVSHMEMI_MCAST_MIN_MIXOP_bf16x2 "min.acc::f32"
#define NVSHMEMI_MCAST_MAX_MIXOP_f16x2 "max.acc::f32"
#define NVSHMEMI_MCAST_MAX_MIXOP_bf16x2 "max.acc::f32"

#if defined(__cplusplus) && __cplusplus >= 201703L
#define IF_CONSTEXPR(expression) if constexpr (expression)
#define ELSE_IF_CONSTEXPR(expression) else if constexpr (expression)
#else
#define IF_CONSTEXPR(expression) if (expression)
#define ELSE_IF_CONSTEXPR(expression) else if (expression)
#endif

template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE int nvshmemi_local_reduce_mcast_threadgroup(
    TYPE *__restrict__ dest, const TYPE *__restrict__ src, size_t nreduce) {
    constexpr bool is_unsigned = std::is_integral<TYPE>::value && std::is_unsigned<TYPE>::value;
    constexpr bool is_signed = std::is_integral<TYPE>::value && std::is_signed<TYPE>::value;
    constexpr bool is_float_v = is_float<TYPE>::value;
    constexpr bool is_double_v = is_double<TYPE>::value;
    constexpr bool is_half_v = is_half<TYPE>::value;
    constexpr bool is_bfloat_v = is_bfloat<TYPE>::value;
    size_t len = nreduce * sizeof(TYPE);

    if ((uintptr_t)dest % sizeof(int4) == 0 && (uintptr_t)src % sizeof(int4) == 0 &&
        len >= sizeof(int4) && NVSHMEMI_MCAST_RDXN_OP_IS_CAP_16B(OP) && !is_double_v) {
        const size_t nelems = len / sizeof(int4);
        int4 *__restrict__ dst_p = (int4 *)dest;
        const int4 *__restrict__ src_p = (const int4 *)src;

        IF_CONSTEXPR(is_unsigned || is_signed || is_float_v) {
            if (len >= 192 && len % 192 == 0)
                nvshmemi_f32_add_local_reduce_mcast16_v4_threadgroup<SCOPE, 12>(dst_p, src_p,
                                                                                nelems);
            else
                nvshmemi_f32_add_local_reduce_mcast16_v4_threadgroup<SCOPE, 1>(dst_p, src_p,
                                                                               nelems);
        }
        ELSE_IF_CONSTEXPR(is_half_v) {
            if (len >= 192 && len % 192 == 0)
                nvshmemi_f16x2_add_local_reduce_mcast16_v4_threadgroup<SCOPE, 12>(dst_p, src_p,
                                                                                  nelems);
            else
                nvshmemi_f16x2_add_local_reduce_mcast16_v4_threadgroup<SCOPE, 1>(dst_p, src_p,
                                                                                 nelems);
        }
        ELSE_IF_CONSTEXPR(is_bfloat_v) {
            if (len >= 192 && len % 192 == 0)
                nvshmemi_bf16x2_add_local_reduce_mcast16_v4_threadgroup<SCOPE, 12>(dst_p, src_p,
                                                                                   nelems);
            else
                nvshmemi_bf16x2_add_local_reduce_mcast16_v4_threadgroup<SCOPE, 1>(dst_p, src_p,
                                                                                  nelems);
        }
        len -= nelems * sizeof(int4);
        if (0 == len) return 0;
        dest = (TYPE *)(dst_p + nelems);
        src = (TYPE *)(src_p + nelems);
    }

    if ((uintptr_t)dest % sizeof(uint64_t) == 0 && (uintptr_t)src % sizeof(uint64_t) == 0 &&
        len >= sizeof(uint64_t) && NVSHMEMI_MCAST_RDXN_OP_IS_CAP_8B(OP)) {
        const size_t nelems = len / sizeof(uint64_t);
        uint64_t *__restrict__ dst_p = (uint64_t *)dest;
        const uint64_t *__restrict__ src_p = (const uint64_t *)src;
        switch (OP) {
            case RDXN_OPS_SUM:
                if (is_unsigned || is_signed)
                    nvshmemi_u64_add_local_reduce_mcast8_threadgroup<TYPE, SCOPE>(dst_p, src_p,
                                                                                  nelems);
                else if (is_float_v)
                    nvshmemi_f32_add_local_reduce_mcast8_v2_threadgroup<TYPE, SCOPE>(dst_p, src_p,
                                                                                     nelems);
                else if (is_double_v)
                    nvshmemi_f64_add_local_reduce_mcast8_threadgroup<TYPE, SCOPE>(dst_p, src_p,
                                                                                  nelems);
                else if (is_half_v)
                    nvshmemi_f16x2_add_local_reduce_mcast8_v2_threadgroup<TYPE, SCOPE>(dst_p, src_p,
                                                                                       nelems);
                else if (is_bfloat_v)
                    nvshmemi_bf16x2_add_local_reduce_mcast8_v2_threadgroup<TYPE, SCOPE>(
                        dst_p, src_p, nelems);
                break;
            case RDXN_OPS_MIN:
                if (is_unsigned)
                    nvshmemi_u64_min_local_reduce_mcast8_threadgroup<TYPE, SCOPE>(dst_p, src_p,
                                                                                  nelems);
                else if (is_signed)
                    nvshmemi_s64_min_local_reduce_mcast8_threadgroup<TYPE, SCOPE>(dst_p, src_p,
                                                                                  nelems);
                break;
            case RDXN_OPS_MAX:
                if (is_unsigned)
                    nvshmemi_u64_max_local_reduce_mcast8_threadgroup<TYPE, SCOPE>(dst_p, src_p,
                                                                                  nelems);
                else if (is_signed)
                    nvshmemi_s64_max_local_reduce_mcast8_threadgroup<TYPE, SCOPE>(dst_p, src_p,
                                                                                  nelems);
                break;
            case RDXN_OPS_AND:
                nvshmemi_b64_and_local_reduce_mcast8_threadgroup<TYPE, SCOPE>(dst_p, src_p, nelems);
                break;
            case RDXN_OPS_XOR:
                nvshmemi_b64_xor_local_reduce_mcast8_threadgroup<TYPE, SCOPE>(dst_p, src_p, nelems);
                break;
            case RDXN_OPS_OR:
                nvshmemi_b64_or_local_reduce_mcast8_threadgroup<TYPE, SCOPE>(dst_p, src_p, nelems);
                break;
            default:
                break;
        }

        len -= nelems * sizeof(uint64_t);
        if (0 == len) return 0;
        dest = (TYPE *)(dst_p + nelems);
        src = (TYPE *)(src_p + nelems);
    }

    if ((uintptr_t)dest % sizeof(uint32_t) == 0 && (uintptr_t)src % sizeof(uint32_t) == 0 &&
        len >= sizeof(uint32_t)) {
        const size_t nelems = len / sizeof(uint32_t);
        uint32_t *__restrict__ dst_p = (uint32_t *)dest;
        const uint32_t *__restrict__ src_p = (const uint32_t *)src;
        switch (OP) {
            case RDXN_OPS_SUM:
                if (is_unsigned)
                    nvshmemi_u32_add_local_reduce_mcast4_threadgroup<TYPE, SCOPE>(dst_p, src_p,
                                                                                  nelems);
                else if (is_signed)
                    nvshmemi_s32_add_local_reduce_mcast4_threadgroup<TYPE, SCOPE>(dst_p, src_p,
                                                                                  nelems);
                else if (is_float_v)
                    nvshmemi_f32_add_local_reduce_mcast4_threadgroup<TYPE, SCOPE>(dst_p, src_p,
                                                                                  nelems);
                else if (is_half_v)
                    nvshmemi_f16x2_add_local_reduce_mcast4_threadgroup<TYPE, SCOPE>(dst_p, src_p,
                                                                                    nelems);
                else if (is_bfloat_v)
                    nvshmemi_bf16x2_add_local_reduce_mcast4_threadgroup<TYPE, SCOPE>(dst_p, src_p,
                                                                                     nelems);
                break;
            case RDXN_OPS_MIN:
                if (is_unsigned)
                    nvshmemi_u32_min_local_reduce_mcast4_threadgroup<TYPE, SCOPE>(dst_p, src_p,
                                                                                  nelems);
                else if (is_signed)
                    nvshmemi_s32_min_local_reduce_mcast4_threadgroup<TYPE, SCOPE>(dst_p, src_p,
                                                                                  nelems);
                break;
            case RDXN_OPS_MAX:
                if (is_unsigned)
                    nvshmemi_u32_max_local_reduce_mcast4_threadgroup<TYPE, SCOPE>(dst_p, src_p,
                                                                                  nelems);
                else if (is_signed)
                    nvshmemi_s32_max_local_reduce_mcast4_threadgroup<TYPE, SCOPE>(dst_p, src_p,
                                                                                  nelems);
                break;
            case RDXN_OPS_XOR:
                nvshmemi_b32_xor_local_reduce_mcast4_threadgroup<TYPE, SCOPE>(dst_p, src_p, nelems);
                break;
            case RDXN_OPS_AND:
                nvshmemi_b32_and_local_reduce_mcast4_threadgroup<TYPE, SCOPE>(dst_p, src_p, nelems);
                break;
            case RDXN_OPS_OR:
                nvshmemi_b32_or_local_reduce_mcast4_threadgroup<TYPE, SCOPE>(dst_p, src_p, nelems);
                break;
            default:
                break;
        }

        len -= nelems * sizeof(uint32_t);
        if (0 == len) return 0;
    }

    /* Return the remainder length, incase the caller wants to retry with unicast */
    return (len);
}

template <typename TYPE, rdxn_ops_t OP>
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void gpu_rdxn_on_demand_2(
    int start, int stride, int size, TYPE *dest, const TYPE *source, size_t nelems, TYPE *pWrk,
    volatile long *pSync, volatile long *sync_counter) {
    int next_rank = -1;
    TYPE *op1 = NULL, *op2 = NULL;
    size_t i;
    volatile TYPE *tmp_operand;
    int my_active_set_pe = ((nvshmemi_device_state_d.mype - start) / stride);
    tmp_operand = (TYPE *)pWrk;
    nvshmemi_put_threadgroup<TYPE, NVSHMEMI_THREADGROUP_THREAD>(dest, source, nelems,
                                                                nvshmemi_device_state_d.mype);
    for (i = 1; i < size; i++) {
        next_rank = start + ((my_active_set_pe + i) % size) * stride;
        nvshmemi_put_nbi_threadgroup<TYPE, NVSHMEMI_THREADGROUP_THREAD>((TYPE *)tmp_operand, source,
                                                                        nelems, next_rank);
        nvshmemi_quiet<NVSHMEMI_THREADGROUP_THREAD>();
        sync_dissem_threadgroup_2<NVSHMEMI_THREADGROUP_THREAD>(start, stride, size, pSync,
                                                               sync_counter);
        op1 = (TYPE *)dest;
        op2 = (TYPE *)tmp_operand;
        gpu_linear_reduce_threadgroup<TYPE, OP, NVSHMEMI_THREADGROUP_THREAD>(op1, op2, op1, nelems);
        sync_dissem_threadgroup_2<NVSHMEMI_THREADGROUP_THREAD>(start, stride, size, pSync,
                                                               sync_counter);
    }
}

template <typename TYPE, rdxn_ops_t OP>
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void gpu_rdxn_on_demand(
    nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nelems) {
    int next_rank = -1;
    TYPE *op1 = NULL, *op2 = NULL;
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    size_t i;
    volatile TYPE *tmp_operand;
    int my_active_set_pe = teami->my_pe;
    tmp_operand = (volatile TYPE *)nvshmemi_team_get_psync(teami, REDUCE);
    nvshmemi_put_threadgroup<TYPE, NVSHMEMI_THREADGROUP_THREAD>(dest, source, nelems,
                                                                nvshmemi_device_state_d.mype);
    for (i = 1 + my_active_set_pe; i < teami->size + my_active_set_pe; i++) {
        int next_rank = nvshmemi_team_translate_pe_to_team_world_wrap(teami, i);
        nvshmemi_put_nbi_threadgroup<TYPE, NVSHMEMI_THREADGROUP_THREAD>((TYPE *)tmp_operand, source,
                                                                        nelems, next_rank);
        nvshmemi_quiet<NVSHMEMI_THREADGROUP_THREAD>();
        sync_dissem_threadgroup<NVSHMEMI_THREADGROUP_THREAD>(team);
        op1 = (TYPE *)dest;
        op2 = (TYPE *)tmp_operand;
        gpu_linear_reduce_threadgroup<TYPE, OP, NVSHMEMI_THREADGROUP_THREAD>(op1, op2, op1, nelems);
        sync_dissem_threadgroup<NVSHMEMI_THREADGROUP_THREAD>(team);
    }
}

/* pWrk usage - (k - 1) * nreduce for step 1
              - k * step2_nphases * nreduce for receiving step 2 data
              - step2_nphases * nreduce for sending data of each phase */
template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void gpu_rdxn_recexch_threadgroup(
    nvshmem_team_t team, TYPE *dst, const TYPE *source, size_t nreduce) {
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    TYPE *pWrk = (TYPE *)nvshmemi_team_get_psync(teami, REDUCE);
    volatile long *pSync = (volatile long *)nvshmemi_team_get_psync(teami, SYNC);
    volatile long *sync_counter = (volatile long *)nvshmemi_team_get_sync_counter(teami);
    const int step1_sendto = teami->reduce_recexch.step1_sendto;
    const int step1_nrecvs = teami->reduce_recexch.step1_nrecvs;
    const int *step1_recvfrom = teami->reduce_recexch.step1_recvfrom;
    const int step2_nphases = teami->reduce_recexch.step2_nphases;
    int **step2_nbrs = teami->reduce_recexch.step2_nbrs;
    const int rank = nvshmemi_device_state_d.mype;
    const int k = nvshmemi_device_state_d.gpu_coll_env_params_var.reduce_recexch_kval;

    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    int in_step2 = (step1_sendto == -1); /* whether this rank participates in Step 2 */

    if (in_step2 == 1) {
        for (size_t i = myIdx; i < nreduce; i += groupSize) {
            dst[i] = source[i];
        }
        nvshmemi_threadgroup_sync<SCOPE>();
    }

    if (in_step2 == 0) {
        size_t offset = (step1_sendto - rank - 1) * nreduce;
        nvshmemi_put_nbi_threadgroup<TYPE, SCOPE>(pWrk + offset, source, nreduce, step1_sendto);
        if (!myIdx) {
            nvshmemi_fence();
            nvshmemi_signal_for_barrier<long>((long *)(pSync + rank), sync_counter[0],
                                              step1_sendto);
        }
    } else if (step1_nrecvs != 0) {
        for (int i = 0; i < step1_nrecvs; i += 1) {
            nvshmemi_wait_until<long>((long *)pSync + step1_recvfrom[i], NVSHMEM_CMP_GE,
                                      sync_counter[0]);
            size_t offset = (rank - step1_recvfrom[i] - 1) * nreduce;
            gpu_linear_reduce_threadgroup<TYPE, OP, SCOPE>(dst, (pWrk + offset), dst, nreduce);
        }
        nvshmemi_threadgroup_sync<SCOPE>();
    }

    /* Step 2 */
    if (in_step2) {
        size_t send_offset = (k - 1) * nreduce + k * step2_nphases * nreduce;
        size_t recv_offset = (k - 1) * nreduce;
        for (int phase = 0; phase < step2_nphases; phase++) {
            int num_small = k - 1;
            for (int i = 0; i < k - 1; i++) {
                if (step2_nbrs[phase][i] > rank) {
                    num_small = i;
                    break;
                }
            }
            /* copy the data to end of pWrk that can be used as source for puts
                while we use dst for reduction */
            for (size_t i = myIdx; i < nreduce; i += groupSize) {
                pWrk[send_offset + phase * nreduce + i] = dst[i];
            }
            nvshmemi_threadgroup_sync<SCOPE>();
            for (int i = 0; i < k - 1; i++) {
                size_t offset = recv_offset + k * phase * nreduce + num_small * nreduce;
                nvshmemi_put_nbi_threadgroup<TYPE, SCOPE>(pWrk + offset,
                                                          pWrk + send_offset + phase * nreduce,
                                                          nreduce, step2_nbrs[phase][i]);
            }
            if (!myIdx) nvshmemi_fence();
            nvshmemi_threadgroup_sync<SCOPE>();
            for (int i = myIdx; i < k - 1; i += groupSize) {
                nvshmemi_signal_for_barrier<long>((long *)(pSync + rank), sync_counter[0],
                                                  step2_nbrs[phase][i]);
            }

            for (int i = 0; i < k - 1; i += 1) {
                nvshmemi_wait_until<uint64_t>((uint64_t *)(pSync + step2_nbrs[phase][i]),
                                              NVSHMEM_CMP_GE, sync_counter[0]);
                int offset = recv_offset + k * phase * nreduce;
                if (step2_nbrs[phase][i] < rank)
                    offset += i * nreduce;
                else
                    offset += (i + 1) * nreduce;
                gpu_linear_reduce_threadgroup<TYPE, OP, SCOPE>(dst, (pWrk + offset), dst, nreduce);
            }
            /*nvshmem_quiet(); */ /*wait for my puts to complete */
        }
    }

    /* Step 3 */
    if (step1_nrecvs > 0) {
        for (int i = 0; i < step1_nrecvs; i++) {
            nvshmemi_put_nbi_threadgroup<TYPE, SCOPE>(dst, dst, nreduce, step1_recvfrom[i]);
        }
        if (!myIdx) nvshmemi_fence();
        nvshmemi_threadgroup_sync<SCOPE>();
        for (int i = myIdx; i < step1_nrecvs; i += groupSize) {
            nvshmemi_signal_for_barrier<long>((long *)(pSync + rank), sync_counter[0],
                                              step1_recvfrom[i]);
        }
    } else if (step1_sendto != -1) {
        if (!myIdx)
            nvshmemi_wait_until<uint64_t>((uint64_t *)(pSync + step1_sendto), NVSHMEM_CMP_GE,
                                          sync_counter[0]);
    }
    nvshmemi_threadgroup_sync<SCOPE>();
    if (!myIdx) sync_counter[0] += 1;
    nvshmemi_threadgroup_sync<SCOPE>();
}

template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE __device__ void
nvshmemi_gpu_rdxn_threadgroup_zcopy_get_bar_direct(nvshmem_team_t team, TYPE *dest,
                                                   const TYPE *source, size_t nreduce) {
    int next_rank = -1;
    int src_offset = -1;
    int next_offset = -1;
    char *base = NULL;
    char *peer_base = NULL;
    char *peer_source = NULL;
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    TYPE *pWrk = (TYPE *)nvshmemi_team_get_psync(teami, REDUCE);
    int i;
    int my_active_set_pe = teami->my_pe;

    base = (char *)((void *)__ldg(
        (const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p +
        nvshmemi_device_state_d.mype));
    src_offset = ((char *)source - base);

    next_rank = nvshmemi_team_translate_pe_to_team_world_wrap(teami, my_active_set_pe + 1);
    next_offset = src_offset;
    peer_base = (char *)((void *)__ldg(
        (const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + next_rank));
    peer_source = peer_base + next_offset;
    gpu_linear_reduce_threadgroup<TYPE, OP, SCOPE>((void *)source, peer_source, dest, nreduce);

    for (i = 2; i < teami->size; i++) {
        next_rank = nvshmemi_team_translate_pe_to_team_world_wrap(teami, my_active_set_pe + i);
        next_offset = src_offset;
        peer_base = (char *)((void *)__ldg(
            (const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + next_rank));
        peer_source = peer_base + next_offset;
        gpu_linear_reduce_threadgroup<TYPE, OP, SCOPE>(dest, peer_source, dest, nreduce);
    }
    nvshmemi_barrier_threadgroup<SCOPE>(team);
}

template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE __device__ void gpu_rdxn_segment_threadgroup(
    nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nelems) {
    int type_size = sizeof(TYPE);
    size_t msg_len = nelems * type_size;
    int next_rank = -1;
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    TYPE *pWrk = (TYPE *)nvshmemi_team_get_psync(teami, REDUCE);
    TYPE *op1 = NULL, *op2 = NULL;
    int i;
    size_t j;
    volatile TYPE *tmp_operand;
    size_t remainder = 0;
    size_t rnds_floor = 0;
    size_t offset = 0;
    int pe_offset = 0;
    int pes_per_round = 0;
    int round = 0;
    size_t exchange_size = 0;
    int my_active_set_pe = teami->my_pe;
    size_t nvshm_gpu_rdxn_seg_size =
        (nvshmemi_device_state_d.gpu_coll_env_params_var.reduce_scratch_size / 2) / sizeof(long);

    tmp_operand = (TYPE *)pWrk;
    nvshmemi_put_nbi_threadgroup<TYPE, SCOPE>((TYPE *)dest, (const TYPE *)source, nelems,
                                              nvshmemi_device_state_d.mype);

    rnds_floor = msg_len / nvshm_gpu_rdxn_seg_size;
    remainder = msg_len % nvshm_gpu_rdxn_seg_size;

    for (j = 0; j < rnds_floor; j++) {
        exchange_size = nvshm_gpu_rdxn_seg_size;
        for (i = 1; i < teami->size; i++) {
            next_rank = nvshmemi_team_translate_pe_to_team_world_wrap(teami, my_active_set_pe + i);
            nvshmemi_put_nbi_threadgroup<TYPE, SCOPE>((TYPE *)tmp_operand,
                                                      (const TYPE *)source + offset,
                                                      (exchange_size / sizeof(TYPE)), next_rank);
            nvshmemi_barrier_threadgroup<SCOPE>(team);
            op1 = (TYPE *)dest + offset;
            op2 = (TYPE *)tmp_operand;
            gpu_linear_reduce_threadgroup<TYPE, OP, SCOPE>(op1, op2, op1,
                                                           (exchange_size / sizeof(TYPE)));
            nvshmemi_sync_threadgroup<SCOPE>(team);
        }
        offset += (exchange_size / sizeof(TYPE));
    }
    if (remainder != 0) {
        exchange_size = remainder;
        pes_per_round = nvshm_gpu_rdxn_seg_size / remainder;
        pe_offset = 1;
        do {
            round = 0;
            for (i = pe_offset; ((round < pes_per_round) && (i < teami->size)); i++) {
                next_rank =
                    nvshmemi_team_translate_pe_to_team_world_wrap(teami, my_active_set_pe + i);
                nvshmemi_put_nbi_threadgroup<TYPE, SCOPE>(
                    (TYPE *)((TYPE *)tmp_operand + (round * (exchange_size / sizeof(TYPE)))),
                    (TYPE *)source + offset, (exchange_size / sizeof(TYPE)), next_rank);
                round++;
                pe_offset++;
            }
            nvshmemi_barrier_threadgroup<SCOPE>(team);
            for (i = 0; i < round; i++) {
                op1 = (TYPE *)dest + offset;
                op2 = (TYPE *)((TYPE *)tmp_operand + (i * (exchange_size / sizeof(TYPE))));
                gpu_linear_reduce_threadgroup<TYPE, OP, SCOPE>(op1, op2, op1,
                                                               (exchange_size / sizeof(TYPE)));
            }
            nvshmemi_sync_threadgroup<SCOPE>(team);
        } while (pe_offset < teami->size);
    }
}

template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_FORCE_INLINE __device__ void
nvshmemi_gpu_rdxn_hierarchical_fcollect_threadgroup(nvshmem_team_t team, TYPE *dest,
                                                    const TYPE *source, size_t nreduce) {
    nvshmemi_team_t *teami_node = nvshmemi_device_state_d.team_pool[NVSHMEM_TEAM_NODE_INDEX];
    nvshmemi_team_t *teami_same_mype_node =
        nvshmemi_device_state_d.team_pool[NVSHMEMX_TEAM_SAME_MYPE_NODE];
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();

    if (!myIdx) { /* Only one thread should increment rdxn_count */
        teami_node->rdxn_count++;
        teami_same_mype_node->rdxn_count++;
    }
    nvshmemi_threadgroup_sync<SCOPE>();

    TYPE *pWrk = (TYPE *)nvshmemi_team_get_psync(teami_node, REDUCE);
    if (teami_node->size >= 2)
        nvshmemi_fcollect_threadgroup<TYPE, SCOPE>(
            NVSHMEMX_TEAM_NODE, pWrk, source, nvshmemi_team_my_pe(NVSHMEMX_TEAM_NODE) * nreduce,
            nreduce);
    else
        nvshmemi_memcpy_threadgroup<SCOPE>(dest, source, nreduce * sizeof(TYPE));

    if (teami_node->size >= 2) {
        for (int j = myIdx; j < nreduce; j += groupSize) {
            gpu_linear_reduce_threadgroup<TYPE, OP, NVSHMEMI_THREADGROUP_THREAD>(
                pWrk + j, pWrk + nreduce + j, dest + j, 1);
            for (int i = 2; i < teami_node->size; i++) {
                gpu_linear_reduce_threadgroup<TYPE, OP, NVSHMEMI_THREADGROUP_THREAD>(
                    pWrk + i * nreduce + j, dest + j, dest + j, 1);
            }
        }
    }
    nvshmemi_threadgroup_sync<SCOPE>();

    if (teami_same_mype_node->size >= 2) {
        pWrk = (TYPE *)nvshmemi_team_get_psync(teami_same_mype_node, REDUCE);
        nvshmemi_fcollect_threadgroup<TYPE, SCOPE>(
            NVSHMEMX_TEAM_SAME_MYPE_NODE, pWrk, dest,
            nvshmemi_team_my_pe(NVSHMEMX_TEAM_SAME_MYPE_NODE) * nreduce, nreduce);
#if CUDART_VERSION >= 12000 && defined(__cplusplus) && __cplusplus >= 201703L
        if constexpr (SCOPE == NVSHMEMI_THREADGROUP_BLOCK && OP == RDXN_OPS_SUM &&
                      sizeof(TYPE) >= 4 && sizeof(TYPE) <= 8) {
            for (int i = myIdx; i < nreduce; i += groupSize) *(dest + i) = 0;
            nvshmemi_threadgroup_sync<SCOPE>();
            auto block = cg::this_thread_block();
            auto tile = cg::tiled_partition<32>(block);
            for (int j = 0; j < nreduce; j++) {
                cg::reduce_update_async(
                    tile, cuda::atomic_ref<TYPE, cuda::thread_scope_block>(dest[j]),
                    (myIdx < teami_same_mype_node->size) ? *((TYPE *)pWrk + myIdx * nreduce + j)
                                                         : (TYPE)0,
                    cg::plus<TYPE>());
            }
        } else
#endif
        {
            for (int j = myIdx; j < nreduce; j += groupSize) {
                gpu_linear_reduce_threadgroup<TYPE, OP, NVSHMEMI_THREADGROUP_THREAD>(
                    (TYPE *)pWrk + j, (TYPE *)pWrk + nreduce + j, dest + j, 1);
                for (int i = 2; i < teami_same_mype_node->size; i++) {
                    gpu_linear_reduce_threadgroup<TYPE, OP, NVSHMEMI_THREADGROUP_THREAD>(
                        (TYPE *)pWrk + i * nreduce + j, dest + j, dest + j, 1);
                }
            }
        }
        nvshmemi_threadgroup_sync<SCOPE>();
    }
}

template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE __device__ void nvshmemi_gpu_rdxn_threadgroup(
    nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nreduce) {
#ifdef NVSHMEM_GPU_COLL_USE_LDST
    nvshmemi_gpu_rdxn_threadgroup_zcopy_get_bar_direct<TYPE, OP, SCOPE>(team, dest, source,
                                                                        nreduce);
#else
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    bool is_team_world =
        nvshmemi_team_is_identical(teami, nvshmemi_device_state_d.team_pool[NVSHMEM_TEAM_WORLD]);
    int k = nvshmemi_device_state_d.gpu_coll_env_params_var.reduce_recexch_kval;
    if (is_team_world && sizeof(TYPE) >= 4 && nreduce % 2 == 0 &&
        nvshmemi_device_state_d.gpu_coll_env_params_var.reduce_scratch_size / 2 >=
            teami->size * nreduce * sizeof(TYPE) &&
        nvshmemi_device_state_d.gpu_coll_env_params_var.fcollect_ll_threshold >=
            nreduce * sizeof(TYPE) &&
        SCOPE == NVSHMEMI_THREADGROUP_BLOCK)
        nvshmemi_gpu_rdxn_hierarchical_fcollect_threadgroup<TYPE, OP, SCOPE>(team, dest, source,
                                                                             nreduce);
    else if (is_team_world &&
             ((nvshmemi_device_state_d.gpu_coll_env_params_var.reduce_scratch_size / 2) /
              sizeof(long)) >=
                 ((k - 1) * nreduce + k * teami->reduce_recexch.step2_nphases * nreduce +
                  teami->reduce_recexch.step2_nphases * nreduce)) {
        gpu_rdxn_recexch_threadgroup<TYPE, OP, SCOPE>(team, dest, source, nreduce);
    } else {
        gpu_rdxn_segment_threadgroup<TYPE, OP, SCOPE>(team, dest, source, nreduce);
    }
#endif
}

#define ALIGNED_UNROLLED_LEN 192 /* 16B (v4.b32) * UNROLL=12 */
#define NVSHMEMI_HALF_ADD_REDUCE_MCAST16_THREADGROUP_UNROLLED(SCOPE, ONESHOT, dst, src, nelems) \
    nvshmemi_f16x2_add_reduce_mcast16_v4_threadgroup<SCOPE, 12, ONESHOT>(dst, src, nelems)
#define NVSHMEMI_BFLOAT_ADD_REDUCE_MCAST16_THREADGROUP_UNROLLED(SCOPE, ONESHOT, dst, src, nelems) \
    nvshmemi_bf16x2_add_reduce_mcast16_v4_threadgroup<SCOPE, 12, ONESHOT>(dst, src, nelems)
#define NVSHMEMI_FLOAT_ADD_REDUCE_MCAST16_THREADGROUP_UNROLLED(SCOPE, ONESHOT, dst, src, nelems) \
    nvshmemi_f32_add_reduce_mcast16_v4_threadgroup<SCOPE, 12, ONESHOT>(dst, src, nelems)

/* Works for inplace and out-of-place reduction for ONESHOT == 0
 * Works for out-of-place reduction for ONESHOT == 1
 */
template <typename TYPE, threadgroup_t SCOPE, bool ONESHOT>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_add_reduce_mcast_threadroup(
    nvshmemi_team_t *teami, TYPE *__restrict__ dst_ptr, const TYPE *__restrict__ src_ptr,
    int nreduce) {
    TYPE *src = (TYPE *)nvshmemi_mc_ptr(teami, src_ptr);
    TYPE *dest;
    if (ONESHOT)
        dest = dst_ptr;
    else
        dest = (TYPE *)nvshmemi_mc_ptr(teami, dst_ptr);
    nvshmemi_threadgroup_sync<SCOPE>();
    size_t len = nreduce * sizeof(TYPE);
    constexpr bool is_half_v = is_half<TYPE>::value;
    constexpr bool is_bfloat_v = is_bfloat<TYPE>::value;

    if ((uintptr_t)dest % sizeof(int4) == 0 && (uintptr_t)src % sizeof(int4) == 0 &&
        len >= sizeof(int4)) {
        const size_t nelems = len / sizeof(int4);
        int4 *__restrict__ dst_p = (int4 *)dest;
        const int4 *__restrict__ src_p = (const int4 *)src;
        IF_CONSTEXPR(is_half_v) {
            if (len >= ALIGNED_UNROLLED_LEN && len % ALIGNED_UNROLLED_LEN == 0)
                NVSHMEMI_HALF_ADD_REDUCE_MCAST16_THREADGROUP_UNROLLED(SCOPE, ONESHOT, dst_p, src_p,
                                                                      nelems);
            else
                nvshmemi_f16x2_add_reduce_mcast16_v4_threadgroup<SCOPE, 1, ONESHOT>(dst_p, src_p,
                                                                                    nelems);
        }
        ELSE_IF_CONSTEXPR(is_bfloat_v) {
            if (len >= ALIGNED_UNROLLED_LEN && len % ALIGNED_UNROLLED_LEN == 0)
                NVSHMEMI_BFLOAT_ADD_REDUCE_MCAST16_THREADGROUP_UNROLLED(SCOPE, ONESHOT, dst_p,
                                                                        src_p, nelems);
            else
                nvshmemi_bf16x2_add_reduce_mcast16_v4_threadgroup<SCOPE, 1, ONESHOT>(dst_p, src_p,
                                                                                     nelems);
        }
        else {
            if (len >= ALIGNED_UNROLLED_LEN && len % ALIGNED_UNROLLED_LEN == 0)
                NVSHMEMI_FLOAT_ADD_REDUCE_MCAST16_THREADGROUP_UNROLLED(SCOPE, ONESHOT, dst_p, src_p,
                                                                       nelems);
            else
                nvshmemi_f32_add_reduce_mcast16_v4_threadgroup<SCOPE, 1, ONESHOT>(dst_p, src_p,
                                                                                  nelems);
        }
        len -= nelems * sizeof(int4);
        if (0 == len) return;
        dest = (TYPE *)(dst_p + nelems);
        src = (TYPE *)(src_p + nelems);
    }

    if ((uintptr_t)dest % sizeof(uint64_t) == 0 && (uintptr_t)src % sizeof(uint64_t) == 0 &&
        len >= sizeof(uint64_t)) {
        const size_t nelems = len / sizeof(uint64_t);
        uint64_t *__restrict__ dst_p = (uint64_t *)dest;
        const uint64_t *__restrict__ src_p = (const uint64_t *)src;
        IF_CONSTEXPR(is_half_v) {
            nvshmemi_f16x2_add_reduce_mcast8_v2_threadgroup<SCOPE, ONESHOT>(dst_p, src_p, nelems);
        }
        ELSE_IF_CONSTEXPR(is_bfloat_v) {
            nvshmemi_bf16x2_add_reduce_mcast8_v2_threadgroup<SCOPE, ONESHOT>(dst_p, src_p, nelems);
        }
        else {
            nvshmemi_f32_add_reduce_mcast8_v2_threadgroup<SCOPE, ONESHOT>(dst_p, src_p, nelems);
        }
        len -= nelems * sizeof(uint64_t);
        if (0 == len) return;
        dest = (TYPE *)(dst_p + nelems);
        src = (TYPE *)(src_p + nelems);
    }

    if ((uintptr_t)dest % sizeof(uint32_t) == 0 && (uintptr_t)src % sizeof(uint32_t) == 0 &&
        len >= sizeof(uint32_t)) {
        const size_t nelems = len / sizeof(uint32_t);
        uint32_t *__restrict__ dst_p = (uint32_t *)dest;
        const uint32_t *__restrict__ src_p = (const uint32_t *)src;
        IF_CONSTEXPR(is_half_v) {
            nvshmemi_f16x2_add_reduce_mcast4_threadgroup<SCOPE, ONESHOT>(dst_p, src_p, nelems);
        }
        ELSE_IF_CONSTEXPR(is_bfloat_v) {
            nvshmemi_bf16x2_add_reduce_mcast4_threadgroup<SCOPE, ONESHOT>(dst_p, src_p, nelems);
        }
        else {
            nvshmemi_f32_add_reduce_mcast4_threadgroup<SCOPE, ONESHOT>(dst_p, src_p, nelems);
        }
        len -= nelems * sizeof(uint32_t);
        if (0 == len) return;
    }

    return;
}

template <typename TYPE, threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_add_reduce_nvls_twoshot_threadgroup(
    nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nreduce) {
#if defined __clang_llvm_bitcode_lib__
    if (__nvvm_reflect("__CUDA_ARCH") >= 900) {
        nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
        int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
        int my_idx_in_active_set = teami->my_pe;
        /* Divide nreduce by team size and handle for the 3 cases */
        int elems_per_pe = nreduce / teami->size;
        int elems_remain = nreduce % teami->size;
        // Case 1: elems_per_pe == 0 => GPU [size-1] does the work on nreduce
        // Case 2: elems_per_pe != 0 and elems_remain != 0 => GPU [0-size-2] does elems_per_pe,
        // GPU[size-1] does elems_per_pe + elems_remain Case 3: elems_per_pe != 0 and elems_remain
        // == 0
        // => all GPUs do work for elems_per_pe
        int my_nelems = elems_per_pe;
        if (my_idx_in_active_set == (teami->size - 1)) {
            my_nelems = elems_per_pe + elems_remain;
        }

        if (my_nelems > 0) {
            nvshmemi_add_reduce_mcast_threadroup<TYPE, SCOPE, 0>(
                teami, dest + elems_per_pe * my_idx_in_active_set,
                source + elems_per_pe * my_idx_in_active_set, my_nelems);
        }

        nvshmemi_barrier_threadgroup<SCOPE>(team);
    } else {
        assert(0 && "Unsupported NVLS on this platform\n");
    }
#else
#if __CUDA_ARCH__ >= 900 && CUDART_VERSION >= 12010
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int my_idx_in_active_set = teami->my_pe;
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    /* Divide nreduce by team size and handle for the 3 cases */
    int elems_per_pe = nreduce / teami->size;
    int elems_remain = nreduce % teami->size;
    // Case 1: elems_per_pe == 0 => GPU [size-1] does the work on nreduce
    // Case 2: elems_per_pe != 0 and elems_remain != 0 => GPU [0-size-2] does elems_per_pe,
    // GPU[size-1] does elems_per_pe + elems_remain Case 3: elems_per_pe != 0 and elems_remain == 0
    // => all GPUs do work for elems_per_pe
    int my_nelems = elems_per_pe;
    if (my_idx_in_active_set == (teami->size - 1)) {
        my_nelems = elems_per_pe + elems_remain;
    }

    if (my_nelems > 0) {
        nvshmemi_add_reduce_mcast_threadroup<TYPE, SCOPE, 0>(
            teami, dest + elems_per_pe * my_idx_in_active_set,
            source + elems_per_pe * my_idx_in_active_set, my_nelems);
    }

    nvshmemi_barrier_threadgroup<SCOPE>(team);
#else
    assert(0 && "Unsupported NVLS on this platform\n");
#endif
#endif
}

template <typename TYPE, threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_add_reduce_nvls_oneshot_threadgroup(
    nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nreduce) {
#if defined __clang_llvm_bitcode_lib__
    if (__nvvm_reflect("__CUDA_ARCH") >= 900) {
        nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
        /* Assign nreduce for all PEs. It may lead to duplicate reduction, but avoid AG stage to
         * communicate partial results as compared to two-shot */
        int elems_per_pe = nreduce;
        // Case 1: elems_per_pe == 0 => no GPUs do any work.
        // Case 2: elems_per_pe != 0 => all GPUs do work for elems_per_pe
        if (elems_per_pe > 0) {
            nvshmemi_add_reduce_mcast_threadroup<TYPE, SCOPE, 1>(teami, dest, source, elems_per_pe);
        }

        /**
         * Using __threadfence_system() is an overkill since we store to local vidmem buffers at the
         * end of ONESHOT add_reducast_mcast The only requirement is not reorder store with sync.
         * Since this code is inlined, this requirement is important (non-inlined function call
         * would automatically guarantee this). Since we use PTX for store, compiler should
         * typically not reorder PTX. So opportunistically, we don't introduce membar.cta PTX here.
         */
        nvshmemi_sync_algo_threadgroup<SCOPE>(team);
    } else {
        assert(0 && "Unsupported NVLS on this platform\n");
    }
#else
#if __CUDA_ARCH__ >= 900 && CUDART_VERSION >= 12010
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    /* Assign nreduce for all PEs. It may lead to duplicate reduction, but avoid AG stage to
     * communicate partial results as compared to two-shot */
    int elems_per_pe = nreduce;
    // Case 1: elems_per_pe == 0 => no GPUs do any work.
    // Case 2: elems_per_pe != 0 => all GPUs do work for elems_per_pe
    if (elems_per_pe > 0) {
        nvshmemi_add_reduce_mcast_threadroup<TYPE, SCOPE, 1>(teami, dest, source, elems_per_pe);
    }

    /**
     * Using __threadfence_system() is an overkill since we store to local vidmem buffers at the end
     * of ONESHOT add_reducast_mcast The only requirement is not reorder store with sync. Since this
     * code is inlined, this requirement is important (non-inlined function call would automatically
     * guarantee this). Since we use PTX for store, compiler should typically not reorder PTX. So
     * opportunistically, we don't introduce membar.cta PTX here.
     */
    nvshmemi_sync_algo_threadgroup<SCOPE>(team);
#else
    assert(0 && "Unsupported NVLS on this platform\n");
#endif
#endif
}

/* This is the entry function for any rdxn collective op - host, on-stream, device
   There is only one exception - nvshmemi_reduce_kernel that is directly calling
   a specific reduction algorithm. That is a special need for team creation */
template <typename TYPE, rdxn_ops_t OP, threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_reduce_threadgroup(nvshmem_team_t team,
                                                                          TYPE *dest,
                                                                          const TYPE *source,
                                                                          size_t nreduce) {
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    if (!myIdx) /* Only one thread should increment rdxn_count */
        nvshmemi_device_state_d.team_pool[team]->rdxn_count += 1;
    nvshmemi_threadgroup_sync<SCOPE>();

    constexpr bool is_rdxn_sum = (OP == RDXN_OPS_SUM);
    constexpr bool is_float_v =
        is_float<TYPE>::value || is_half<TYPE>::value || is_bfloat<TYPE>::value;

    constexpr bool is_half_prec = is_half<TYPE>::value || is_bfloat<TYPE>::value;
    int reduce_algo = nvshmemi_device_state_d.gpu_coll_env_params_var.reduce_algo;

    bool is_nvls_algo_supported =
        is_rdxn_sum && is_float_v &&
        nvshmemi_device_state_d.team_pool[team]->nvls_rsc_base_ptr != NULL &&
        (nreduce * sizeof(TYPE)) % 4 == 0;
    bool is_inplace_op = (source == dest);  // exact overlap
    // inplace ops cannot use 1-shot AR as there can be a race condition due to execution of
    // ld_reduce on a given MC offset on the switch and a posted stg to the same MC offset on any of
    // the GPUs, causing incorrect result.
    bool is_one_shot_supported =
        (is_nvls_algo_supported) && !(is_half_prec && nreduce == 1) && !(is_inplace_op);
    bool is_two_shot_supported =
        (is_nvls_algo_supported) &&
        !(is_half_prec && (nreduce <= nvshmemi_device_state_d.team_pool[team]->size ||
                           (nreduce % nvshmemi_device_state_d.team_pool[team]->size > 0)));

    /* When adding new algorithms in 3.x, start at 5. This is for backward compatibility reasons,
     * see cpu_coll.cpp.
     */
    switch (reduce_algo) {
        case 0: /* NVLS Two Shot or One Shot Allreduce for REDUCE_SUM and float/bfloat/half dtype */
        case 1:
        case 2:
            if (nvshmemi_device_state_d.gpu_coll_env_params_var.reduce_nvls_threshold >= nreduce &&
                is_one_shot_supported)
                reduce_algo = 4;
            else if (is_two_shot_supported)
                reduce_algo = 3;
            else {
                reduce_algo = 0;
            }
            break;
        case 3:
            /* When forcing NVLS algos, legalize for unsupported case and switch back to non-NVLS
             * algos multimem PTX don't support BF16/FP16 single element reduction
             */
            if (!is_two_shot_supported) {
                reduce_algo = 0;
            }

            break;
        case 4:
            /* When forcing NVLS algos, legalize for unsupported case and switch back to non-NVLS
             * algos multimem PTX don't support BF16/FP16 single element reduction
             */
            if (!is_one_shot_supported) {
                reduce_algo = 0;
            }

            break;
        default:
            break;
    }

    switch (reduce_algo) {
        case 3: /* NVLS Two Shot Allreduce (RS + AG) */
            nvshmemi_add_reduce_nvls_twoshot_threadgroup<TYPE, SCOPE>(team, dest, source, nreduce);
            break;
        case 4: /* NVLS One Shot Allreduce (AR) */
            nvshmemi_add_reduce_nvls_oneshot_threadgroup<TYPE, SCOPE>(team, dest, source, nreduce);
            break;
        default:
            nvshmemi_gpu_rdxn_threadgroup<TYPE, OP, SCOPE>(team, dest, source, nreduce);

            break;
    }
}

NVSHMEMI_STATIC __device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void
nvshmemi_double2_maxloc_reduce_alltoall_block(nvshmem_team_t team, double2 *dest,
                                              const double2 *source) {
#define SCOPE NVSHMEMI_THREADGROUP_BLOCK
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    const int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    const int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    if (!myIdx) { /* Only one thread should increment */
        teami->ll_flag++;
        teami->rdxn_count += 1;
    }
    nvshmemi_threadgroup_sync<SCOPE>();
    const uint32_t ll_flag = teami->ll_flag;
    char *pWrk = (char *)nvshmemi_team_get_psync(teami, REDUCE);

    nvshmemi_packLL_naive<double2, SCOPE>((uint64_t *)pWrk, source, 1, ll_flag);
    nvshmemi_threadgroup_sync<SCOPE>();
    const int my_pe = nvshmemi_team_my_pe(team);
    const int n_pes = nvshmemi_team_n_pes(team);
    for (int i = myIdx + 1; i < n_pes; i += groupSize) {
        int peer = (my_pe + i) % n_pes;
        size_t offset = 2 * sizeof(double2) * (my_pe + 2);
        nvshmemi_put_nbi_threadgroup<uint64_t, NVSHMEMI_THREADGROUP_THREAD>(
            (uint64_t *)(pWrk + offset), (uint64_t *)(pWrk), sizeof(double2) / sizeof(uint32_t),
            nvshmemi_team_translate_pe(team, peer, NVSHMEM_TEAM_WORLD));
    }

    if (!myIdx) {
        dest[0] = source[0];
        for (int i = 1; i < n_pes; i++) {
            int peer = (my_pe + i) % n_pes;
            size_t offset = 2 * sizeof(double2) * (peer + 2);
            nvshmemi_recvLL<double2, NVSHMEMI_THREADGROUP_THREAD>(
                (double2 *)(pWrk + 2 * sizeof(double2)), (uint64_t *)(pWrk + offset), 1, ll_flag);
            dest[0] = perform_gpu_rdxn<double2, RDXN_OPS_MAXLOC>(
                dest[0], *((double2 *)(pWrk + 2 * sizeof(double2))));
        }
    }
    nvshmemi_threadgroup_sync<SCOPE>();
#undef SCOPE
}

NVSHMEMI_STATIC __device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void
nvshmemi_double2_maxloc_rooted_reduce_flat_block(nvshmem_team_t team, double2 *dest,
                                                 const double2 *source) {
#define SCOPE NVSHMEMI_THREADGROUP_BLOCK
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    const int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    const int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    if (!myIdx) { /* Only one thread should increment */
        teami->ll_flag++;
        teami->rdxn_count += 1;
    }
    nvshmemi_threadgroup_sync<SCOPE>();
    const uint32_t ll_flag = teami->ll_flag;
    char *pWrk = (char *)nvshmemi_team_get_psync(teami, REDUCE);

    if (nvshmemi_team_my_pe(team) != 0) {
        nvshmemi_packLL_naive<double2, SCOPE>((uint64_t *)pWrk, source, 1, ll_flag);
        size_t offset = 2 * sizeof(double2) * nvshmemi_team_my_pe(team);
        nvshmemi_put_nbi_threadgroup<uint64_t, SCOPE>(
            (uint64_t *)(pWrk + offset), (uint64_t *)(pWrk), sizeof(double2) / sizeof(uint32_t),
            nvshmemi_team_translate_pe(team, 0, NVSHMEM_TEAM_WORLD));
    } else {
        dest[0] = source[0];
        if (!myIdx) {
            for (int i = 1; i < teami->size; i += 1) {
                size_t offset = 2 * sizeof(double2) * i;
                nvshmemi_recvLL<double2, NVSHMEMI_THREADGROUP_THREAD>(
                    (double2 *)pWrk, (uint64_t *)(pWrk + offset), 1, ll_flag);
                dest[0] = perform_gpu_rdxn<double2, RDXN_OPS_MAXLOC>(dest[0], *(double2 *)pWrk);
            }
        }
        nvshmemi_threadgroup_sync<SCOPE>();
    }
#undef SCOPE
}

NVSHMEMI_STATIC __device__ NVSHMEMI_DEVICE_ALWAYS_INLINE int nvshmemi_double2_maxloc_reduce_block(
    nvshmem_team_t team, double2 *dest, const double2 *source, size_t nreduce) {
#ifdef NVSHMEM_DEBUG
    assert(nreduce == 1);
#endif
#define SCOPE NVSHMEMI_THREADGROUP_BLOCK
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    switch (nvshmemi_device_state_d.gpu_coll_env_params_var.reduce_maxloc_algo) {
        case 1: /*  Alltoall algorithm */
            nvshmemi_double2_maxloc_reduce_alltoall_block(team, dest, source);
            break;
        case 2: /* Topo-unaware: Flat reduce + Flat bcast */
            nvshmemi_double2_maxloc_rooted_reduce_flat_block(team, dest, source);
            nvshmemi_bcast_internode_tree_threadgroup<double2, SCOPE>(
                team, dest, dest, 1, 0, nvshmemi_team_n_pes(team) - 1);
            break;
        case 3: /* Topo aware two-level flat reduce + Topo aware two-level tree bcast */
            if (teami->is_team_node || teami->is_team_same_mype_node) {
                nvshmemi_double2_maxloc_rooted_reduce_flat_block(team, dest, source);
            } else {
                nvshmemi_double2_maxloc_rooted_reduce_flat_block(teami->team_node, dest, source);
                if (nvshmemi_team_my_pe(teami->team_node) == 0) {
                    nvshmemi_double2_maxloc_rooted_reduce_flat_block(teami->team_same_mype_node,
                                                                     dest, dest);
                }
            }
            nvshmemi_bcast_hierarchical_threadgroup<double2, SCOPE>(team, dest, dest, 1, 0);
            break;
        case 4: /* Topo aware two-level flat reduce + Topo aware two-level tree bcast */
            if (teami->is_team_node || teami->is_team_same_mype_node) {
                nvshmemi_double2_maxloc_reduce_alltoall_block(team, dest, source);
            } else {
                nvshmemi_double2_maxloc_reduce_alltoall_block(teami->team_node, dest, source);
                nvshmemi_double2_maxloc_reduce_alltoall_block(teami->team_same_mype_node, dest,
                                                              dest);
            }
            break;
        default:
            assert(0);
            break;
    }
#undef SCOPE
    return 0;
}

/******* Tile collective functions ********/

// Select implementation based on the operation, datatype
template <typename vtype, typename T, threadgroup_t scope, typename tuple_t, rdxn_ops_t op,
          int ONESHOT, int major_dim, int minor_dim>
__device__ inline void nvshmemi_tile_allreduce_nvls_thread_vec(
    nvshmem_team_t team, T *src, T *dst,
    const int size_major_dim,        // size along the major dimension in elements
    const int size_minor_dim,        // size along the minor dimension in elements
    const int src_stride_minor_dim,  // src stride along minor dimension in elements
    const int dst_stride_minor_dim,  // dst stride along minor dimension in elements
    const int src_stride_major_dim,  // src stride along major dimension in elements
    const int dst_stride_major_dim,  // dst stride along major dimension in elements
    tuple_t start_coord, tuple_t boundary) {
    vtype *src_v = reinterpret_cast<vtype *>(nvshmemx_mc_ptr(team, src));
    vtype *dst_v =
        reinterpret_cast<vtype *>(dst);  // one-shot does only local stores to destination
    if (ONESHOT == 0) {
        dst_v = reinterpret_cast<vtype *>(nvshmemx_mc_ptr(team, dst));  // two-shot
        assert((dst_v != nullptr) && "Failed to get multicast ptr for destination");
    }

    int src_stride_minor_dim_v = src_stride_minor_dim;
    if (src_stride_minor_dim > 1) {
        src_stride_minor_dim_v = (src_stride_minor_dim * sizeof(T)) / sizeof(vtype);
    }
    int dst_stride_minor_dim_v = dst_stride_minor_dim;
    if (dst_stride_minor_dim > 1) {
        dst_stride_minor_dim_v = (dst_stride_minor_dim * sizeof(T)) / sizeof(vtype);
    }
    int src_stride_major_dim_v = src_stride_major_dim;  // keep stride as is if ==1
    if (src_stride_major_dim > 1) {
        src_stride_major_dim_v = (src_stride_major_dim * sizeof(T)) / sizeof(vtype);
    }
    int dst_stride_major_dim_v = dst_stride_major_dim;
    if (dst_stride_major_dim > 1) {
        dst_stride_major_dim_v = (dst_stride_major_dim * sizeof(T)) / sizeof(vtype);
    }
    int nelem_major_dim = (size_major_dim * sizeof(T)) / sizeof(vtype);
    const int nelem_minor_dim = size_minor_dim;

    if constexpr (is_half<T>::value) {
        if constexpr (op == RDXN_OPS_SUM) {
            nvshmemi_f16x2_tile_allreduceADD_mcast_threadgroup<vtype, T, scope, tuple_t, ONESHOT,
                                                               major_dim, minor_dim>(
                dst_v, src_v, nelem_major_dim, nelem_minor_dim, src_stride_minor_dim_v,
                dst_stride_minor_dim_v, src_stride_major_dim_v, dst_stride_major_dim_v, start_coord,
                boundary);
        } else if constexpr (op == RDXN_OPS_MIN) {
            nvshmemi_f16x2_tile_allreduceMIN_mcast_threadgroup<vtype, T, scope, tuple_t, ONESHOT,
                                                               major_dim, minor_dim>(
                dst_v, src_v, nelem_major_dim, nelem_minor_dim, src_stride_minor_dim_v,
                dst_stride_minor_dim_v, src_stride_major_dim_v, dst_stride_major_dim_v, start_coord,
                boundary);
        } else if constexpr (op == RDXN_OPS_MAX) {
            nvshmemi_f16x2_tile_allreduceMAX_mcast_threadgroup<vtype, T, scope, tuple_t, ONESHOT,
                                                               major_dim, minor_dim>(
                dst_v, src_v, nelem_major_dim, nelem_minor_dim, src_stride_minor_dim_v,
                dst_stride_minor_dim_v, src_stride_major_dim_v, dst_stride_major_dim_v, start_coord,
                boundary);
        }
    } else if constexpr (is_bfloat<T>::value) {
        if constexpr (op == RDXN_OPS_SUM) {
            nvshmemi_bf16x2_tile_allreduceADD_mcast_threadgroup<vtype, T, scope, tuple_t, ONESHOT,
                                                                major_dim, minor_dim>(
                dst_v, src_v, nelem_major_dim, nelem_minor_dim, src_stride_minor_dim_v,
                dst_stride_minor_dim_v, src_stride_major_dim_v, dst_stride_major_dim_v, start_coord,
                boundary);
        } else if constexpr (op == RDXN_OPS_MIN) {
            nvshmemi_bf16x2_tile_allreduceMIN_mcast_threadgroup<vtype, T, scope, tuple_t, ONESHOT,
                                                                major_dim, minor_dim>(
                dst_v, src_v, nelem_major_dim, nelem_minor_dim, src_stride_minor_dim_v,
                dst_stride_minor_dim_v, src_stride_major_dim_v, dst_stride_major_dim_v, start_coord,
                boundary);
        } else if constexpr (op == RDXN_OPS_MAX) {
            nvshmemi_bf16x2_tile_allreduceMAX_mcast_threadgroup<vtype, T, scope, tuple_t, ONESHOT,
                                                                major_dim, minor_dim>(
                dst_v, src_v, nelem_major_dim, nelem_minor_dim, src_stride_minor_dim_v,
                dst_stride_minor_dim_v, src_stride_major_dim_v, dst_stride_major_dim_v, start_coord,
                boundary);
        }
    } else if constexpr (is_float<T>::value) {
        static_assert((op != RDXN_OPS_MIN) && (op != RDXN_OPS_MAX),
                      "NVLS allreduce/reduce min/max not supported for float datatype");
        if constexpr (op == RDXN_OPS_SUM) {
            nvshmemi_f32_tile_allreduceADD_mcast_threadgroup<vtype, T, scope, tuple_t, ONESHOT,
                                                             major_dim, minor_dim>(
                dst_v, src_v, nelem_major_dim, nelem_minor_dim, src_stride_minor_dim_v,
                dst_stride_minor_dim_v, src_stride_major_dim_v, dst_stride_major_dim_v, start_coord,
                boundary);
        } else {
            assert(0 && "unsupported reduce operation");
        }
    }
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t, threadgroup_t scope,
          rdxn_ops_t op, int ONESHOT, int major_dim, int minor_dim>
__device__ inline void nvshmemi_tile_allreduce_nvls_dim(nvshmem_team_t team,
                                                        src_tensor_t src_tensor,
                                                        dst_tensor_t dst_tensor,
                                                        tuple_t start_coord, tuple_t boundary) {
    using T = typename src_tensor_t::value_type;

    // check for vector len == 4
    // Conditions: ptr must be aligned to int4, shape must be a multiple of 16, stride must be a
    // multiple of 16
    if (((size_t)src_tensor.data() % sizeof(int4) == 0) &&
        ((size_t)dst_tensor.data() % sizeof(int4) == 0) &&
        (((get_tuple_val<major_dim>(src_tensor.shape()) * sizeof(T)) % sizeof(int4)) == 0) &&
        (((get_stride_element<minor_dim>(src_tensor) * sizeof(T)) % sizeof(int4)) == 0) &&
        (((get_stride_element<minor_dim>(dst_tensor) * sizeof(T)) % sizeof(int4)) == 0)) {
        nvshmemi_tile_allreduce_nvls_thread_vec<int4, T, scope, tuple_t, op, ONESHOT, major_dim,
                                                minor_dim>(
            team, src_tensor.data(), dst_tensor.data(),
            get_shape_element<major_dim>(src_tensor),   // contiguous size
            get_shape_element<minor_dim>(src_tensor),   // strided size
            get_stride_element<minor_dim>(src_tensor),  // src stride minor_dim
            get_stride_element<minor_dim>(dst_tensor),  // dst stride minor_dim
            get_stride_element<major_dim>(src_tensor),  // src stride major_dim; equal to 1
            get_stride_element<major_dim>(dst_tensor),  // dst stride major_dim; equal to 1
            start_coord, boundary);

    } else if (((size_t)src_tensor.data() % sizeof(uint64_t) == 0) &&
               ((size_t)dst_tensor.data() % sizeof(uint64_t) == 0) &&
               (((get_tuple_val<major_dim>(src_tensor.shape()) * sizeof(T)) % sizeof(uint64_t)) ==
                0) &&
               (((get_stride_element<minor_dim>(src_tensor) * sizeof(T)) % sizeof(uint64_t)) ==
                0) &&
               (((get_stride_element<minor_dim>(dst_tensor) * sizeof(T)) % sizeof(uint64_t)) ==
                0)) {
        // Vector length == 2
        nvshmemi_tile_allreduce_nvls_thread_vec<uint64_t, T, scope, tuple_t, op, ONESHOT, major_dim,
                                                minor_dim>(
            team, src_tensor.data(), dst_tensor.data(),
            get_shape_element<major_dim>(src_tensor),   // contiguous size
            get_shape_element<minor_dim>(src_tensor),   // strided size
            get_stride_element<minor_dim>(src_tensor),  // src stride minor_dim
            get_stride_element<minor_dim>(dst_tensor),  // dst stride minor_dim
            get_stride_element<major_dim>(src_tensor),  // src stride major_dim; equal to 1
            get_stride_element<major_dim>(dst_tensor),  // dst stride major_dim; equal to 1
            start_coord, boundary);

    } else {  // vector len 1
        nvshmemi_tile_allreduce_nvls_thread_vec<uint32_t, T, scope, tuple_t, op, ONESHOT, major_dim,
                                                minor_dim>(
            team, src_tensor.data(), dst_tensor.data(),
            get_shape_element<major_dim>(src_tensor),   // contiguous size
            get_shape_element<minor_dim>(src_tensor),   // strided size
            get_stride_element<minor_dim>(src_tensor),  // src stride minor_dim
            get_stride_element<minor_dim>(dst_tensor),  // dst stride minor_dim
            get_stride_element<major_dim>(src_tensor),  // src stride major_dim; equal to 1
            get_stride_element<major_dim>(dst_tensor),  // dst stride major_dim; equal to 1
            start_coord, boundary);
    }
}
// specialize for the vectorization
template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t, threadgroup_t scope,
          rdxn_ops_t op, int ONESHOT>
__device__ inline void nvshmemi_tile_allreduce_nvls_thread(nvshmem_team_t team,
                                                           src_tensor_t src_tensor,
                                                           dst_tensor_t dst_tensor,
                                                           tuple_t start_coord, tuple_t boundary) {
    using T = typename src_tensor_t::value_type;

    if constexpr ((get_constant(safe_get<0>(decltype(src_tensor.stride()){})) == 1) &&
                  (get_constant(safe_get<0>(decltype(dst_tensor.stride()){})) == 1)) {
        // dim 0 major
        constexpr int major_dim = 0;
        constexpr int minor_dim = 1;

        if constexpr (sizeof(T) < 4) {
            // Shape along major dimension should be divisible by 2, because we operate at fp16x2
            assert(((get_shape_element<major_dim>(src_tensor) % 2) == 0) &&
                   ((get_shape_element<major_dim>(dst_tensor) % 2) == 0) &&
                   "Currently for 16B datatypes, we only support tensors which are 32b aligned "
                   "along their continuous dimension");
        }

        nvshmemi_tile_allreduce_nvls_dim<src_tensor_t, dst_tensor_t, tuple_t, scope, op, ONESHOT,
                                         major_dim, minor_dim>(team, src_tensor, dst_tensor,
                                                               start_coord, boundary);

    } else if constexpr ((get_constant(safe_get<1>(decltype(src_tensor.stride()){})) == 1) &&
                         (get_constant(safe_get<1>(decltype(dst_tensor.stride()){})) == 1)) {
        // dim 1 major
        constexpr int major_dim = 1;
        constexpr int minor_dim = 0;

        if constexpr (sizeof(T) < 4) {
            // Shape along major dimension should be divisible by 2, because we operate at fp16x2
            assert(((get_shape_element<major_dim>(src_tensor) % 2) == 0) &&
                   ((get_shape_element<major_dim>(dst_tensor) % 2) == 0) &&
                   "Currently for 16B datatypes, we only support tensors which are 32b aligned "
                   "along their continuous dimension");
        }

        nvshmemi_tile_allreduce_nvls_dim<src_tensor_t, dst_tensor_t, tuple_t, scope, op, ONESHOT,
                                         major_dim, minor_dim>(team, src_tensor, dst_tensor,
                                                               start_coord, boundary);
    } else {
        // No contiguous dimension found at compile time
        // TODO support when major dimension for src and dst tensors are different
        if ((get_stride_element<1>(src_tensor) == 1) && (get_stride_element<1>(dst_tensor) == 1)) {
            constexpr int major_dim = 1;
            constexpr int minor_dim = 0;

            if constexpr (sizeof(T) < 4) {
                // Shape along major dimension should be divisible by 2, because we operate at
                // fp16x2
                assert(((get_shape_element<major_dim>(src_tensor) % 2) == 0) &&
                       ((get_shape_element<major_dim>(dst_tensor) % 2) == 0) &&
                       "Currently for 16B datatypes, we only support tensors which are 32b aligned "
                       "along their continuous dimension");
            }

            nvshmemi_tile_allreduce_nvls_dim<src_tensor_t, dst_tensor_t, tuple_t, scope, op,
                                             ONESHOT, major_dim, minor_dim>(
                team, src_tensor, dst_tensor, start_coord, boundary);
        } else {
            // setting major_dim to 0, minor_dim to 1
            constexpr int major_dim = 0;
            constexpr int minor_dim = 1;

            if constexpr (sizeof(T) < 4) {
                // Shape along major dimension should be divisible by 2, because we operate at
                // fp16x2
                assert(((get_shape_element<major_dim>(src_tensor) % 2) == 0) &&
                       ((get_shape_element<major_dim>(dst_tensor) % 2) == 0) &&
                       "Currently for 16B datatypes, we only support tensors which are 32b aligned "
                       "along their continuous dimension");
            }
            nvshmemi_tile_allreduce_nvls_dim<src_tensor_t, dst_tensor_t, tuple_t, scope, op,
                                             ONESHOT, major_dim, minor_dim>(
                team, src_tensor, dst_tensor, start_coord, boundary);
        }
    }
}

// Tile allreduce entrypoint
// Call underlying function based on scope and algo
template <nvshmemx::tile_coll_algo_t algo, typename src_tensor_t, typename dst_tensor_t,
          typename tuple_t, threadgroup_t scope, rdxn_ops_t op>
__device__ inline int nvshmemi_tile_allreduce(nvshmem_team_t team, src_tensor_t src_tensor,
                                              dst_tensor_t dst_tensor, tuple_t start_coord,
                                              tuple_t boundary, int root, uint64_t flag) {
    using T = typename src_tensor_t::value_type;
#if defined(__cplusplus) && __cplusplus < 201703L
    assert(0 && "Tile-granular APIs need C++ 17");
#endif

    static_assert(cuda::std::is_same<typename src_tensor_t::value_type,
                                     typename dst_tensor_t::value_type>::value,
                  "Source and destination tensors must have the same type");

    static_assert(cuda::std::is_same<decltype(cuda::std::declval<src_tensor_t>().shape()),
                                     decltype(cuda::std::declval<dst_tensor_t>().shape())>::value,
                  "Source and destination tensors must have same shape");

    static_assert((algo == nvshmemx::tile_coll_algo_t::NVLS_ONE_SHOT_PULL_NBI) ||
                      (algo == nvshmemx::tile_coll_algo_t::NVLS_TWO_SHOT_PUSH_NBI),
                  "Unsupported tile AllReduce algorithm. "
                  "Currently NVLS_TILE_ONE_SHOT_PULL_NBI and NVLS_TILE_TWO_SHOT_PUSH_NBI "
                  "are supported for tile allreduce");

    static_assert((scope == NVSHMEMI_THREADGROUP_THREAD) || (scope == NVSHMEMI_THREADGROUP_WARP) ||
                      (scope == NVSHMEMI_THREADGROUP_WARPGROUP) ||
                      (scope == NVSHMEMI_THREADGROUP_BLOCK),
                  "Unsupported scope");

    static_assert((op == RDXN_OPS_SUM) || (op == RDXN_OPS_MIN) || (op == RDXN_OPS_MAX),
                  "Unsupported operation");

    static_assert(((is_half<T>::value) || (is_bfloat<T>::value) || (is_float<T>::value)),
                  "Unsupported datatype");

    assert((src_tensor.data() != nullptr) && (dst_tensor.data() != nullptr) &&
           "Null pointers passed");

    // check if both src and dst have same continuous dimension
    // TODO relax this constraint
    assert(
        (((get_stride_element<0>(src_tensor) == 1) && (get_stride_element<0>(dst_tensor) == 1)) ||
         ((get_stride_element<1>(src_tensor) == 1) && (get_stride_element<1>(dst_tensor) == 1))) &&
        "Currently we only support cases where source and destination tile are continuous "
        "along one dimension");

    assert(!flag && "Currently non-zero flag value is unsupported");

    if constexpr (algo == nvshmemx::tile_coll_algo_t::NVLS_TWO_SHOT_PUSH_NBI) {
        // check for NVLS support in hardware
#if __CUDA_ARCH__ >= 900 && CUDART_VERSION >= 12010
        assert(__CUDA_ARCH__ >= 900 && CUDART_VERSION >= 12010);

        // As this algo PULLs data from other PEs, we need to ensure src data is ready
        // Ensure all PEs have reached this point and pushed their data to local mem

        __threadfence();  // ensure data is visible in local GPU mem
        nvshmemi_sync_algo_threadgroup<scope>(team);

        // Only root will perform all reduce for two-shot
        if (root == -1) {
            assert(0 && "Root must be specified for NVLS two-shot tile allreduce");
            return 0;
        } else if (root != nvshmem_team_my_pe(team)) {
            return 0;
        }

        nvshmemi_tile_allreduce_nvls_thread<src_tensor_t, dst_tensor_t, tuple_t, scope, op, 0>(
            team, src_tensor, dst_tensor, start_coord, boundary);
#else
        assert(__CUDA_ARCH__ >= 900 && CUDART_VERSION >= 12010 &&
               "Unsupported NVLS on this platform");
#endif
        return 0;
    } else {
        // check for NVLS support in hardware
#if __CUDA_ARCH__ >= 900 && CUDART_VERSION >= 12010

        // As this algo PULLs data from other PEs, we need to ensure src data is ready
        // Ensure all PEs have reached this point and pushed their data to local mem

        __threadfence();  // ensure data is visible in local GPU mem
        nvshmemi_sync_algo_threadgroup<scope>(team);

        // root is not used in one-shot allreduce
        // One-shot allreduce
        nvshmemi_tile_allreduce_nvls_thread<src_tensor_t, dst_tensor_t, tuple_t, scope, op, 1>(
            team, src_tensor, dst_tensor, start_coord, boundary);
#else
        assert(__CUDA_ARCH__ >= 900 && CUDART_VERSION >= 12010 &&
               "Unsupported NVLS on this platform");
#endif
        return 0;
    }
}

// Tile reduce entrypoint
// Call underlying function based on scope and algo
template <nvshmemx::tile_coll_algo_t algo, typename src_tensor_t, typename dst_tensor_t,
          typename tuple_t, threadgroup_t scope, rdxn_ops_t op>
__device__ inline int nvshmemi_tile_reduce(nvshmem_team_t team, src_tensor_t src_tensor,
                                           dst_tensor_t dst_tensor, tuple_t start_coord,
                                           tuple_t boundary, int root, uint64_t flag) {
    using T = typename src_tensor_t::value_type;

#if defined(__cplusplus) && __cplusplus < 201703L
    assert(0 && "Tile-granular APIs need C++ 17");
#endif

    static_assert(cuda::std::is_same<typename src_tensor_t::value_type,
                                     typename dst_tensor_t::value_type>::value,
                  "Source and destination tensors must have the same type");

    static_assert(cuda::std::is_same<decltype(cuda::std::declval<src_tensor_t>().shape()),
                                     decltype(cuda::std::declval<dst_tensor_t>().shape())>::value,
                  "Source and destination tensors must have same shape");

    static_assert((algo == nvshmemx::tile_coll_algo_t::NVLS_ONE_SHOT_PULL_NBI),
                  "Unsupported tile Reduce algorithm. "
                  "Currently NVLS_ONE_SHOT_PULL_NBI is supported for tile reduce");

    static_assert((scope == NVSHMEMI_THREADGROUP_THREAD) || (scope == NVSHMEMI_THREADGROUP_WARP) ||
                      (scope == NVSHMEMI_THREADGROUP_WARPGROUP) ||
                      (scope == NVSHMEMI_THREADGROUP_BLOCK),
                  "Unsupported scope");

    static_assert((op == RDXN_OPS_SUM) || (op == RDXN_OPS_MIN) || (op == RDXN_OPS_MAX),
                  "Unsupported operation");

    static_assert(((is_half<T>::value) || (is_bfloat<T>::value) || (is_float<T>::value)),
                  "Unsupported datatype");

    assert((src_tensor.data() != nullptr) && (dst_tensor.data() != nullptr) &&
           "Null pointers passed");

    // check if both src and dst have same continuous dimension
    // TODO relax this constraint
    assert(
        (((get_stride_element<0>(src_tensor) == 1) && (get_stride_element<0>(dst_tensor) == 1)) ||
         ((get_stride_element<1>(src_tensor) == 1) && (get_stride_element<1>(dst_tensor) == 1))) &&
        "Currently we only support cases where source and destination tile are continuous "
        "along one dimension");

    assert(!flag && "Currently non-zero flag value is unsupported");

    // NVLS Reduce only has one-shot
    if constexpr (algo == nvshmemx::tile_coll_algo_t::NVLS_ONE_SHOT_PULL_NBI) {
        // check for NVLS support in hardware
#if __CUDA_ARCH__ >= 900 && CUDART_VERSION >= 12010
        // As this algo PULLs data from other PEs, we need to ensure src data is ready
        // Ensure all PEs have reached this point and pushed their data to local mem

        __threadfence();  // ensure data is visible in local GPU mem
        nvshmemi_sync_algo_threadgroup<scope>(team);

        // Reduce is implemented as one-shot AllReduce with a root

        // Only root will perform reduce
        if (root == -1) {
            assert(0 && "Root must be specified for NVLS tile reduce");
            return 0;
        } else if (root != nvshmem_team_my_pe(team)) {
            return 0;
        }

        nvshmemi_tile_allreduce_nvls_thread<src_tensor_t, dst_tensor_t, tuple_t, scope, op, 1>(
            team, src_tensor, dst_tensor, start_coord, boundary);
#else
        assert(__CUDA_ARCH__ >= 900 && CUDART_VERSION >= 12010 &&
               "Unsupported NVLS on this platform");
#endif
        return 0;
    } else {
        // Extend as other algorithms are added
        return 0;
    }
}

#endif /* __CUDA_ARCH__ */
#endif /* REDUCE_DEVICE_CUH */
