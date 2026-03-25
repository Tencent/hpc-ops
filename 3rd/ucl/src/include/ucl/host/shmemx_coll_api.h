/*
 * UCL SHMEM API - Host Extended Collective API
 * This file enhances NVSHMEM host extended collective API functions into UCL shmem namespace
 */

#ifndef UCL_SHMEMX_HOST_COLL_API_H
#define UCL_SHMEMX_HOST_COLL_API_H

#include "host/nvshmemx_coll_api.h"
#include "host/nvshmem_macros.h"

namespace ucl {
namespace shmem {

// ===== Standard RMA types for collectives =====
#define UCL_SHMEMX_REPT_FOR_STANDARD_RMA_TYPES_COLL(MACRO) \
    MACRO(float, float) \
    MACRO(double, double) \
    MACRO(char, char) \
    MACRO(schar, signed char) \
    MACRO(short, short) \
    MACRO(int, int) \
    MACRO(long, long) \
    MACRO(longlong, long long) \
    MACRO(uchar, unsigned char) \
    MACRO(ushort, unsigned short) \
    MACRO(uint, unsigned int) \
    MACRO(ulong, unsigned long) \
    MACRO(ulonglong, unsigned long long) \
    MACRO(int8, int8_t) \
    MACRO(int16, int16_t) \
    MACRO(int32, int32_t) \
    MACRO(int64, int64_t) \
    MACRO(uint8, uint8_t) \
    MACRO(uint16, uint16_t) \
    MACRO(uint32, uint32_t) \
    MACRO(uint64, uint64_t) \
    MACRO(size, size_t) \
    MACRO(ptrdiff, ptrdiff_t)

// ===== Alltoall On Stream =====
#define UCL_SHMEMX_DECL_TYPENAME_ALLTOALL_ON_STREAM(TYPENAME, TYPE) \
    static inline int shmemx_##TYPENAME##_alltoall_on_stream(nvshmem_team_t team, TYPE *dest, const TYPE *src, \
                                                              size_t nelem, cudaStream_t stream) { \
        return nvshmemx_##TYPENAME##_alltoall_on_stream(team, dest, src, nelem, stream); \
    }

UCL_SHMEMX_REPT_FOR_STANDARD_RMA_TYPES_COLL(UCL_SHMEMX_DECL_TYPENAME_ALLTOALL_ON_STREAM)

static inline int shmemx_alltoallmem_on_stream(nvshmem_team_t team, void *dest, const void *src, size_t nelem, cudaStream_t stream) {
    return nvshmemx_alltoallmem_on_stream(team, dest, src, nelem, stream);
}

// ===== Barrier On Stream =====
static inline int shmemx_barrier_on_stream(nvshmem_team_t team, cudaStream_t stream) {
    return nvshmemx_barrier_on_stream(team, stream);
}
static inline void shmemx_barrier_all_on_stream(cudaStream_t stream) {
    nvshmemx_barrier_all_on_stream(stream);
}

// ===== Sync On Stream =====
static inline int shmemx_team_sync_on_stream(nvshmem_team_t team, cudaStream_t stream) {
    return nvshmemx_team_sync_on_stream(team, stream);
}
static inline void shmemx_sync_all_on_stream(cudaStream_t stream) {
    nvshmemx_sync_all_on_stream(stream);
}

// ===== Broadcast On Stream =====
#define UCL_SHMEMX_DECL_TYPENAME_BROADCAST_ON_STREAM(TYPENAME, TYPE) \
    static inline int shmemx_##TYPENAME##_broadcast_on_stream(nvshmem_team_t team, TYPE *dest, const TYPE *src, \
                                                               size_t nelem, int PE_root, cudaStream_t stream) { \
        return nvshmemx_##TYPENAME##_broadcast_on_stream(team, dest, src, nelem, PE_root, stream); \
    }

UCL_SHMEMX_REPT_FOR_STANDARD_RMA_TYPES_COLL(UCL_SHMEMX_DECL_TYPENAME_BROADCAST_ON_STREAM)

static inline int shmemx_broadcastmem_on_stream(nvshmem_team_t team, void *dest, const void *src, size_t nelem, int PE_root, cudaStream_t stream) {
    return nvshmemx_broadcastmem_on_stream(team, dest, src, nelem, PE_root, stream);
}

// ===== Fcollect On Stream =====
#define UCL_SHMEMX_DECL_TYPENAME_FCOLLECT_ON_STREAM(TYPENAME, TYPE) \
    static inline int shmemx_##TYPENAME##_fcollect_on_stream(nvshmem_team_t team, TYPE *dest, const TYPE *src, \
                                                              size_t nelem, cudaStream_t stream) { \
        return nvshmemx_##TYPENAME##_fcollect_on_stream(team, dest, src, nelem, stream); \
    }

UCL_SHMEMX_REPT_FOR_STANDARD_RMA_TYPES_COLL(UCL_SHMEMX_DECL_TYPENAME_FCOLLECT_ON_STREAM)

static inline int shmemx_fcollectmem_on_stream(nvshmem_team_t team, void *dest, const void *src, size_t nelem, cudaStream_t stream) {
    return nvshmemx_fcollectmem_on_stream(team, dest, src, nelem, stream);
}

// ===== Reduce types =====
#define UCL_SHMEMX_REPT_FOR_BITWISE_REDUCE_TYPES(MACRO, OP) \
    MACRO(uchar, unsigned char, OP) \
    MACRO(ushort, unsigned short, OP) \
    MACRO(uint, unsigned int, OP) \
    MACRO(ulong, unsigned long, OP) \
    MACRO(ulonglong, unsigned long long, OP) \
    MACRO(int8, int8_t, OP) \
    MACRO(int16, int16_t, OP) \
    MACRO(int32, int32_t, OP) \
    MACRO(int64, int64_t, OP) \
    MACRO(uint8, uint8_t, OP) \
    MACRO(uint16, uint16_t, OP) \
    MACRO(uint32, uint32_t, OP) \
    MACRO(uint64, uint64_t, OP)

#define UCL_SHMEMX_REPT_FOR_STANDARD_REDUCE_TYPES(MACRO, OP) \
    MACRO(char, char, OP) \
    MACRO(schar, signed char, OP) \
    MACRO(short, short, OP) \
    MACRO(int, int, OP) \
    MACRO(long, long, OP) \
    MACRO(longlong, long long, OP) \
    MACRO(uchar, unsigned char, OP) \
    MACRO(ushort, unsigned short, OP) \
    MACRO(uint, unsigned int, OP) \
    MACRO(ulong, unsigned long, OP) \
    MACRO(ulonglong, unsigned long long, OP) \
    MACRO(int8, int8_t, OP) \
    MACRO(int16, int16_t, OP) \
    MACRO(int32, int32_t, OP) \
    MACRO(int64, int64_t, OP) \
    MACRO(uint8, uint8_t, OP) \
    MACRO(uint16, uint16_t, OP) \
    MACRO(uint32, uint32_t, OP) \
    MACRO(uint64, uint64_t, OP) \
    MACRO(size, size_t, OP) \
    MACRO(float, float, OP) \
    MACRO(double, double, OP)

#define UCL_SHMEMX_REPT_FOR_ARITH_REDUCE_TYPES(MACRO, OP) \
    MACRO(char, char, OP) \
    MACRO(schar, signed char, OP) \
    MACRO(short, short, OP) \
    MACRO(int, int, OP) \
    MACRO(long, long, OP) \
    MACRO(longlong, long long, OP) \
    MACRO(uchar, unsigned char, OP) \
    MACRO(ushort, unsigned short, OP) \
    MACRO(uint, unsigned int, OP) \
    MACRO(ulong, unsigned long, OP) \
    MACRO(ulonglong, unsigned long long, OP) \
    MACRO(int8, int8_t, OP) \
    MACRO(int16, int16_t, OP) \
    MACRO(int32, int32_t, OP) \
    MACRO(int64, int64_t, OP) \
    MACRO(uint8, uint8_t, OP) \
    MACRO(uint16, uint16_t, OP) \
    MACRO(uint32, uint32_t, OP) \
    MACRO(uint64, uint64_t, OP) \
    MACRO(size, size_t, OP) \
    MACRO(float, float, OP) \
    MACRO(double, double, OP)

// ===== Reduce On Stream =====
#define UCL_SHMEMX_DECL_REDUCE_ONSTREAM(NAME, TYPE, OP) \
    NVSHMEMI_HOSTDEVICE_PREFIX static inline int shmemx_##NAME##_##OP##_reduce_on_stream(nvshmem_team_t team, TYPE *dest, \
                                                                                          const TYPE *src, size_t nreduce, \
                                                                                          cudaStream_t stream) { \
        return nvshmemx_##NAME##_##OP##_reduce_on_stream(team, dest, src, nreduce, stream); \
    }

UCL_SHMEMX_REPT_FOR_BITWISE_REDUCE_TYPES(UCL_SHMEMX_DECL_REDUCE_ONSTREAM, and)
UCL_SHMEMX_REPT_FOR_BITWISE_REDUCE_TYPES(UCL_SHMEMX_DECL_REDUCE_ONSTREAM, or)
UCL_SHMEMX_REPT_FOR_BITWISE_REDUCE_TYPES(UCL_SHMEMX_DECL_REDUCE_ONSTREAM, xor)

UCL_SHMEMX_REPT_FOR_STANDARD_REDUCE_TYPES(UCL_SHMEMX_DECL_REDUCE_ONSTREAM, max)
UCL_SHMEMX_REPT_FOR_STANDARD_REDUCE_TYPES(UCL_SHMEMX_DECL_REDUCE_ONSTREAM, min)

UCL_SHMEMX_REPT_FOR_ARITH_REDUCE_TYPES(UCL_SHMEMX_DECL_REDUCE_ONSTREAM, sum)
UCL_SHMEMX_REPT_FOR_ARITH_REDUCE_TYPES(UCL_SHMEMX_DECL_REDUCE_ONSTREAM, prod)

// ===== Reducescatter On Stream =====
#define UCL_SHMEMX_DECL_REDUCESCATTER_ONSTREAM(NAME, TYPE, OP) \
    NVSHMEMI_HOSTDEVICE_PREFIX static inline int shmemx_##NAME##_##OP##_reducescatter_on_stream(nvshmem_team_t team, TYPE *dest, \
                                                                                                 const TYPE *src, size_t nreduce, \
                                                                                                 cudaStream_t stream) { \
        return nvshmemx_##NAME##_##OP##_reducescatter_on_stream(team, dest, src, nreduce, stream); \
    }

UCL_SHMEMX_REPT_FOR_BITWISE_REDUCE_TYPES(UCL_SHMEMX_DECL_REDUCESCATTER_ONSTREAM, and)
UCL_SHMEMX_REPT_FOR_BITWISE_REDUCE_TYPES(UCL_SHMEMX_DECL_REDUCESCATTER_ONSTREAM, or)
UCL_SHMEMX_REPT_FOR_BITWISE_REDUCE_TYPES(UCL_SHMEMX_DECL_REDUCESCATTER_ONSTREAM, xor)

UCL_SHMEMX_REPT_FOR_STANDARD_REDUCE_TYPES(UCL_SHMEMX_DECL_REDUCESCATTER_ONSTREAM, max)
UCL_SHMEMX_REPT_FOR_STANDARD_REDUCE_TYPES(UCL_SHMEMX_DECL_REDUCESCATTER_ONSTREAM, min)

UCL_SHMEMX_REPT_FOR_ARITH_REDUCE_TYPES(UCL_SHMEMX_DECL_REDUCESCATTER_ONSTREAM, sum)
UCL_SHMEMX_REPT_FOR_ARITH_REDUCE_TYPES(UCL_SHMEMX_DECL_REDUCESCATTER_ONSTREAM, prod)

// ===== Threadgroup Collectives =====
#ifdef __CUDACC__

// alltoall(s) collectives
#define UCL_SHMEMX_DECL_TYPENAME_ALLTOALL_SCOPE(SCOPE, TYPENAME, TYPE)                       \
    __device__ int shmemx_##TYPENAME##_alltoall_##SCOPE(nvshmem_team_t team, TYPE *dest, \
                                                          const TYPE *src, size_t nelem);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE(UCL_SHMEMX_DECL_TYPENAME_ALLTOALL_SCOPE, warp)
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE(UCL_SHMEMX_DECL_TYPENAME_ALLTOALL_SCOPE, block)
#undef UCL_SHMEMX_DECL_TYPENAME_ALLTOALL_SCOPE

__device__ int shmemx_alltoallmem_warp(nvshmem_team_t team, void *dest, const void *src,
                                         size_t nelem);
__device__ int shmemx_alltoallmem_block(nvshmem_team_t team, void *dest, const void *src,
                                          size_t nelem);

// barrier collectives
__device__ int shmemx_barrier_warp(nvshmem_team_t team);
__device__ void shmemx_barrier_all_warp();
__device__ int shmemx_barrier_warpgroup(nvshmem_team_t team);
__device__ int shmemx_barrier_block(nvshmem_team_t team);
__device__ void shmemx_barrier_all_block();

// sync collectives
__device__ int shmemx_team_sync_warp(nvshmem_team_t team);
__device__ void shmemx_sync_all_warp();
__device__ int shmemx_team_sync_block(nvshmem_team_t team);
__device__ void shmemx_sync_all_block();

// broadcast collectives
#define UCL_SHMEMX_DECL_TYPENAME_BROADCAST_SCOPE(SCOPE, TYPENAME, TYPE) \
    __device__ int shmemx_##TYPENAME##_broadcast_##SCOPE(           \
        nvshmem_team_t team, TYPE *dest, const TYPE *src, size_t nelem, int PE_root);
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE(UCL_SHMEMX_DECL_TYPENAME_BROADCAST_SCOPE, warp)
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE(UCL_SHMEMX_DECL_TYPENAME_BROADCAST_SCOPE, block)
#undef UCL_SHMEMX_DECL_TYPENAME_BROADCAST_SCOPE

__device__ int shmemx_broadcastmem_warp(nvshmem_team_t team, void *dest, const void *src,
                                          size_t nelem);
__device__ int shmemx_broadcastmem_block(nvshmem_team_t team, void *dest, const void *src,
                                           size_t nelem);

// fcollect collectives
#define UCL_SHMEMX_DECL_TYPENAME_FCOLLECT_SCOPE(SCOPE, TYPENAME, TYPE)                       \
    __device__ int shmemx_##TYPENAME##_fcollect_##SCOPE(nvshmem_team_t team, TYPE *dest, \
                                                          const TYPE *src, size_t nelem);
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE(UCL_SHMEMX_DECL_TYPENAME_FCOLLECT_SCOPE, warp)
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE(UCL_SHMEMX_DECL_TYPENAME_FCOLLECT_SCOPE, block)
#undef UCL_SHMEMX_DECL_TYPENAME_FCOLLECT_SCOPE

__device__ int shmemx_fcollectmem_warp(nvshmem_team_t team, void *dest, const void *src,
                                         size_t nelem);
__device__ int shmemx_fcollectmem_block(nvshmem_team_t team, void *dest, const void *src,
                                          size_t nelem);

// reduction collectives
#define UCL_SHMEMX_DECL_TYPENAME_OP_REDUCE_THREADGROUP(SCOPE, TYPENAME, TYPE, OP) \
    NVSHMEMI_HOSTDEVICE_PREFIX int shmemx_##TYPENAME##_##OP##_reduce_##SCOPE( \
        nvshmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce);

#define UCL_SHMEMX_DECL_TYPENAME_OP_REDUCE(SC)                                                      \
    NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE(                                            \
        UCL_SHMEMX_DECL_TYPENAME_OP_REDUCE_THREADGROUP, SC, and)                                    \
    NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE(                                            \
        UCL_SHMEMX_DECL_TYPENAME_OP_REDUCE_THREADGROUP, SC, or)                                     \
    NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE(                                            \
        UCL_SHMEMX_DECL_TYPENAME_OP_REDUCE_THREADGROUP, SC, xor)                                    \
                                                                                                  \
    NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE(                                           \
        UCL_SHMEMX_DECL_TYPENAME_OP_REDUCE_THREADGROUP, SC, max)                                    \
    NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE(                                           \
        UCL_SHMEMX_DECL_TYPENAME_OP_REDUCE_THREADGROUP, SC, min)                                    \
                                                                                                  \
    NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES_WITH_SCOPE(UCL_SHMEMX_DECL_TYPENAME_OP_REDUCE_THREADGROUP, \
                                                    SC, sum)                                      \
    NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES_WITH_SCOPE(UCL_SHMEMX_DECL_TYPENAME_OP_REDUCE_THREADGROUP, \
                                                    SC, prod)

UCL_SHMEMX_DECL_TYPENAME_OP_REDUCE(warp);
UCL_SHMEMX_DECL_TYPENAME_OP_REDUCE(block);

NVSHMEMI_HOSTDEVICE_PREFIX static inline int shmemx_double2_maxloc_reduce_block(nvshmem_team_t team, double2 *dest, const double2 *src, size_t nreduce) {
    return nvshmemx_double2_maxloc_reduce_block(team, dest, src, nreduce);
}

// Reducescatter threadgroup
#define UCL_SHMEMX_DECL_TYPENAME_OP_REDUCESCATTER_THREADGROUP(NAME, TYPE, OP) \
    NVSHMEMI_HOSTDEVICE_PREFIX static inline int shmemx_##NAME##_##OP##_reducescatter_warp(nvshmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce) { \
        return nvshmemx_##NAME##_##OP##_reducescatter_warp(team, dest, src, nreduce); \
    } \
    NVSHMEMI_HOSTDEVICE_PREFIX static inline int shmemx_##NAME##_##OP##_reducescatter_block(nvshmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce) { \
        return nvshmemx_##NAME##_##OP##_reducescatter_block(team, dest, src, nreduce); \
    }

UCL_SHMEMX_REPT_FOR_BITWISE_REDUCE_TYPES(UCL_SHMEMX_DECL_TYPENAME_OP_REDUCESCATTER_THREADGROUP, and)
UCL_SHMEMX_REPT_FOR_BITWISE_REDUCE_TYPES(UCL_SHMEMX_DECL_TYPENAME_OP_REDUCESCATTER_THREADGROUP, or)
UCL_SHMEMX_REPT_FOR_BITWISE_REDUCE_TYPES(UCL_SHMEMX_DECL_TYPENAME_OP_REDUCESCATTER_THREADGROUP, xor)

UCL_SHMEMX_REPT_FOR_STANDARD_REDUCE_TYPES(UCL_SHMEMX_DECL_TYPENAME_OP_REDUCESCATTER_THREADGROUP, max)
UCL_SHMEMX_REPT_FOR_STANDARD_REDUCE_TYPES(UCL_SHMEMX_DECL_TYPENAME_OP_REDUCESCATTER_THREADGROUP, min)

UCL_SHMEMX_REPT_FOR_ARITH_REDUCE_TYPES(UCL_SHMEMX_DECL_TYPENAME_OP_REDUCESCATTER_THREADGROUP, sum)
UCL_SHMEMX_REPT_FOR_ARITH_REDUCE_TYPES(UCL_SHMEMX_DECL_TYPENAME_OP_REDUCESCATTER_THREADGROUP, prod)

#endif // __CUDACC__

} // namespace shmem
} // namespace ucl

#endif // UCL_SHMEMX_HOST_COLL_API_H
