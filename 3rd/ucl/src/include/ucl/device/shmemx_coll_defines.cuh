/*
 * UCL SHMEM API - Device Extended Collective Defines
 * This file enhances NVSHMEM device extended collective API functions into UCL shmem namespace
 */

#ifndef UCL_SHMEMX_DEVICE_COLL_DEFINES_CUH
#define UCL_SHMEMX_DEVICE_COLL_DEFINES_CUH

#include "device/nvshmemx_coll_defines.cuh"

namespace ucl {
namespace shmem {

#ifdef __CUDA_ARCH__

// ===== Standard RMA Types =====
#define UCL_SHMEMX_DEVICE_COLL_REPT_FOR_STANDARD_RMA_TYPES(MACRO) \
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

// ===== Alltoall Threadgroup =====
#define UCL_SHMEMX_DEVICE_COLL_DECL_TYPENAME_ALLTOALL_SCOPE(TYPENAME, TYPE) \
    NVSHMEMI_STATIC NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_INLINE int shmemx_##TYPENAME##_alltoall_warp(nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nelems) { \
        return nvshmemx_##TYPENAME##_alltoall_warp(team, dest, source, nelems); \
    } \
    NVSHMEMI_STATIC NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_INLINE int shmemx_##TYPENAME##_alltoall_block(nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nelems) { \
        return nvshmemx_##TYPENAME##_alltoall_block(team, dest, source, nelems); \
    }

UCL_SHMEMX_DEVICE_COLL_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEMX_DEVICE_COLL_DECL_TYPENAME_ALLTOALL_SCOPE)

NVSHMEMI_STATIC NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_INLINE int shmemx_alltoallmem_warp(nvshmem_team_t team, void *dest, const void *source, size_t nelems) {
    return nvshmemx_alltoallmem_warp(team, dest, source, nelems);
}

NVSHMEMI_STATIC NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_INLINE int shmemx_alltoallmem_block(nvshmem_team_t team, void *dest, const void *source, size_t nelems) {
    return nvshmemx_alltoallmem_block(team, dest, source, nelems);
}

// ===== Barrier Threadgroup =====
NVSHMEMI_STATIC NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_INLINE int shmemx_barrier_warp(nvshmem_team_t team) {
    return nvshmemx_barrier_warp(team);
}

NVSHMEMI_STATIC NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_INLINE void shmemx_barrier_all_warp() {
    nvshmemx_barrier_all_warp();
}

NVSHMEMI_STATIC NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_INLINE int shmemx_barrier_warpgroup(nvshmem_team_t team) {
    return nvshmemx_barrier_warpgroup(team);
}

NVSHMEMI_STATIC NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_INLINE int shmemx_barrier_block(nvshmem_team_t team) {
    return nvshmemx_barrier_block(team);
}

NVSHMEMI_STATIC NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_INLINE void shmemx_barrier_all_block() {
    nvshmemx_barrier_all_block();
}

// ===== Sync Threadgroup =====
NVSHMEMI_STATIC NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_INLINE int shmemx_team_sync_warp(nvshmem_team_t team) {
    return nvshmemx_team_sync_warp(team);
}

NVSHMEMI_STATIC NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_INLINE void shmemx_sync_all_warp() {
    nvshmemx_sync_all_warp();
}

NVSHMEMI_STATIC NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_INLINE int shmemx_team_sync_block(nvshmem_team_t team) {
    return nvshmemx_team_sync_block(team);
}

NVSHMEMI_STATIC NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_INLINE void shmemx_sync_all_block() {
    nvshmemx_sync_all_block();
}

// ===== Broadcast Threadgroup =====
#define UCL_SHMEMX_DEVICE_COLL_DECL_TYPENAME_BROADCAST_SCOPE(TYPENAME, TYPE) \
    NVSHMEMI_STATIC NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_INLINE int shmemx_##TYPENAME##_broadcast_warp(nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nelems, int PE_root) { \
        return nvshmemx_##TYPENAME##_broadcast_warp(team, dest, source, nelems, PE_root); \
    } \
    NVSHMEMI_STATIC NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_INLINE int shmemx_##TYPENAME##_broadcast_block(nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nelems, int PE_root) { \
        return nvshmemx_##TYPENAME##_broadcast_block(team, dest, source, nelems, PE_root); \
    }

UCL_SHMEMX_DEVICE_COLL_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEMX_DEVICE_COLL_DECL_TYPENAME_BROADCAST_SCOPE)

NVSHMEMI_STATIC NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_INLINE int shmemx_broadcastmem_warp(nvshmem_team_t team, void *dest, const void *source, size_t nelems, int PE_root) {
    return nvshmemx_broadcastmem_warp(team, dest, source, nelems, PE_root);
}

NVSHMEMI_STATIC NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_INLINE int shmemx_broadcastmem_block(nvshmem_team_t team, void *dest, const void *source, size_t nelems, int PE_root) {
    return nvshmemx_broadcastmem_block(team, dest, source, nelems, PE_root);
}

// ===== Fcollect Threadgroup =====
#define UCL_SHMEMX_DEVICE_COLL_DECL_TYPENAME_FCOLLECT_SCOPE(TYPENAME, TYPE) \
    NVSHMEMI_STATIC NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_INLINE int shmemx_##TYPENAME##_fcollect_warp(nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nelems) { \
        return nvshmemx_##TYPENAME##_fcollect_warp(team, dest, source, nelems); \
    } \
    NVSHMEMI_STATIC NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_INLINE int shmemx_##TYPENAME##_fcollect_block(nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nelems) { \
        return nvshmemx_##TYPENAME##_fcollect_block(team, dest, source, nelems); \
    }

UCL_SHMEMX_DEVICE_COLL_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEMX_DEVICE_COLL_DECL_TYPENAME_FCOLLECT_SCOPE)

NVSHMEMI_STATIC NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_INLINE int shmemx_fcollectmem_warp(nvshmem_team_t team, void *dest, const void *source, size_t nelems) {
    return nvshmemx_fcollectmem_warp(team, dest, source, nelems);
}

NVSHMEMI_STATIC NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_INLINE int shmemx_fcollectmem_block(nvshmem_team_t team, void *dest, const void *source, size_t nelems) {
    return nvshmemx_fcollectmem_block(team, dest, source, nelems);
}

// ===== Reduce Types =====
#define UCL_SHMEMX_DEVICE_COLL_REPT_FOR_BITWISE_REDUCE_TYPES(MACRO, OP) \
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

#define UCL_SHMEMX_DEVICE_COLL_REPT_FOR_STANDARD_REDUCE_TYPES(MACRO, OP) \
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

#define UCL_SHMEMX_DEVICE_COLL_REPT_FOR_ARITH_REDUCE_TYPES(MACRO, OP) \
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

// ===== Reduce Threadgroup =====
#define UCL_SHMEMX_DEVICE_COLL_DECL_TYPENAME_OP_REDUCE_THREADGROUP(TYPENAME, TYPE, OP) \
    NVSHMEMI_STATIC NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_INLINE int shmemx_##TYPENAME##_##OP##_reduce_warp(nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nreduce) { \
        return nvshmemx_##TYPENAME##_##OP##_reduce_warp(team, dest, source, nreduce); \
    } \
    NVSHMEMI_STATIC NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_INLINE int shmemx_##TYPENAME##_##OP##_reduce_block(nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nreduce) { \
        return nvshmemx_##TYPENAME##_##OP##_reduce_block(team, dest, source, nreduce); \
    }

UCL_SHMEMX_DEVICE_COLL_REPT_FOR_BITWISE_REDUCE_TYPES(UCL_SHMEMX_DEVICE_COLL_DECL_TYPENAME_OP_REDUCE_THREADGROUP, and)
UCL_SHMEMX_DEVICE_COLL_REPT_FOR_BITWISE_REDUCE_TYPES(UCL_SHMEMX_DEVICE_COLL_DECL_TYPENAME_OP_REDUCE_THREADGROUP, or)
UCL_SHMEMX_DEVICE_COLL_REPT_FOR_BITWISE_REDUCE_TYPES(UCL_SHMEMX_DEVICE_COLL_DECL_TYPENAME_OP_REDUCE_THREADGROUP, xor)

UCL_SHMEMX_DEVICE_COLL_REPT_FOR_STANDARD_REDUCE_TYPES(UCL_SHMEMX_DEVICE_COLL_DECL_TYPENAME_OP_REDUCE_THREADGROUP, max)
UCL_SHMEMX_DEVICE_COLL_REPT_FOR_STANDARD_REDUCE_TYPES(UCL_SHMEMX_DEVICE_COLL_DECL_TYPENAME_OP_REDUCE_THREADGROUP, min)

UCL_SHMEMX_DEVICE_COLL_REPT_FOR_ARITH_REDUCE_TYPES(UCL_SHMEMX_DEVICE_COLL_DECL_TYPENAME_OP_REDUCE_THREADGROUP, sum)
UCL_SHMEMX_DEVICE_COLL_REPT_FOR_ARITH_REDUCE_TYPES(UCL_SHMEMX_DEVICE_COLL_DECL_TYPENAME_OP_REDUCE_THREADGROUP, prod)

NVSHMEMI_STATIC NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_INLINE int shmemx_double2_maxloc_reduce_block(nvshmem_team_t team, double2 *dest, const double2 *source, size_t nreduce);

// ===== Reducescatter Threadgroup =====
#define UCL_SHMEMX_DEVICE_COLL_DECL_TYPENAME_OP_REDUCESCATTER_THREADGROUP(TYPENAME, TYPE, OP) \
    NVSHMEMI_STATIC NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_INLINE int shmemx_##TYPENAME##_##OP##_reducescatter_warp(nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nreduce); \
    NVSHMEMI_STATIC NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_INLINE int shmemx_##TYPENAME##_##OP##_reducescatter_block(nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nreduce);

UCL_SHMEMX_DEVICE_COLL_REPT_FOR_BITWISE_REDUCE_TYPES(UCL_SHMEMX_DEVICE_COLL_DECL_TYPENAME_OP_REDUCESCATTER_THREADGROUP, and)
UCL_SHMEMX_DEVICE_COLL_REPT_FOR_BITWISE_REDUCE_TYPES(UCL_SHMEMX_DEVICE_COLL_DECL_TYPENAME_OP_REDUCESCATTER_THREADGROUP, or)
UCL_SHMEMX_DEVICE_COLL_REPT_FOR_BITWISE_REDUCE_TYPES(UCL_SHMEMX_DEVICE_COLL_DECL_TYPENAME_OP_REDUCESCATTER_THREADGROUP, xor)

UCL_SHMEMX_DEVICE_COLL_REPT_FOR_STANDARD_REDUCE_TYPES(UCL_SHMEMX_DEVICE_COLL_DECL_TYPENAME_OP_REDUCESCATTER_THREADGROUP, max)
UCL_SHMEMX_DEVICE_COLL_REPT_FOR_STANDARD_REDUCE_TYPES(UCL_SHMEMX_DEVICE_COLL_DECL_TYPENAME_OP_REDUCESCATTER_THREADGROUP, min)

UCL_SHMEMX_DEVICE_COLL_REPT_FOR_ARITH_REDUCE_TYPES(UCL_SHMEMX_DEVICE_COLL_DECL_TYPENAME_OP_REDUCESCATTER_THREADGROUP, sum)
UCL_SHMEMX_DEVICE_COLL_REPT_FOR_ARITH_REDUCE_TYPES(UCL_SHMEMX_DEVICE_COLL_DECL_TYPENAME_OP_REDUCESCATTER_THREADGROUP, prod)

#endif // __CUDA_ARCH__

} // namespace shmem
} // namespace ucl

#endif // UCL_SHMEMX_DEVICE_COLL_DEFINES_CUH
