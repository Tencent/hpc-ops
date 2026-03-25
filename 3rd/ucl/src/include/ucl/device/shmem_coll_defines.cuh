/*
 * UCL SHMEM API - Device Collective Defines
 * This file enhances NVSHMEM device collective API functions into UCL shmem namespace
 */

#ifndef UCL_SHMEM_DEVICE_COLL_DEFINES_CUH
#define UCL_SHMEM_DEVICE_COLL_DEFINES_CUH

#include "device/nvshmem_coll_defines.cuh"

namespace ucl {
namespace shmem {

#ifdef __CUDA_ARCH__

// ===== Standard RMA Types =====
#define UCL_SHMEM_DEVICE_COLL_REPT_FOR_STANDARD_RMA_TYPES(MACRO) \
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

// ===== Alltoall =====
NVSHMEMI_STATIC NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_INLINE int shmem_alltoallmem(nvshmem_team_t team, void *dest, const void *source, size_t nelems);

#define UCL_SHMEM_DEVICE_COLL_DECL_TYPENAME_ALLTOALL(TYPENAME, TYPE) \
    NVSHMEMI_STATIC NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_INLINE int shmem_##TYPENAME##_alltoall(nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nelems);

UCL_SHMEM_DEVICE_COLL_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEM_DEVICE_COLL_DECL_TYPENAME_ALLTOALL)

// ===== Barrier =====
NVSHMEMI_STATIC NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_INLINE int shmem_barrier(nvshmem_team_t team);

NVSHMEMI_STATIC NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_INLINE void shmem_barrier_all();
// ===== Sync =====
NVSHMEMI_STATIC NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_INLINE int shmem_team_sync(nvshmem_team_t team);

NVSHMEMI_STATIC NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_INLINE void shmem_sync_all();

// ===== Broadcast =====
NVSHMEMI_STATIC NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_INLINE int shmem_broadcastmem(nvshmem_team_t team, void *dest, const void *source, size_t nelems, int PE_root);

#define UCL_SHMEM_DEVICE_COLL_DECL_TYPENAME_BROADCAST(TYPENAME, TYPE) \
    NVSHMEMI_STATIC NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_INLINE int shmem_##TYPENAME##_broadcast(nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nelems, int PE_root);

UCL_SHMEM_DEVICE_COLL_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEM_DEVICE_COLL_DECL_TYPENAME_BROADCAST)

// ===== Fcollect =====
NVSHMEMI_STATIC NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_INLINE int shmem_fcollectmem(nvshmem_team_t team, void *dest, const void *source, size_t nelems);

#define UCL_SHMEM_DEVICE_COLL_DECL_TYPENAME_FCOLLECT(TYPENAME, TYPE) \
    NVSHMEMI_STATIC NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_INLINE int shmem_##TYPENAME##_fcollect(nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nelems);

UCL_SHMEM_DEVICE_COLL_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEM_DEVICE_COLL_DECL_TYPENAME_FCOLLECT)

// ===== Reduce Types =====
#define UCL_SHMEM_DEVICE_COLL_REPT_FOR_BITWISE_REDUCE_TYPES(MACRO, OP) \
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

#define UCL_SHMEM_DEVICE_COLL_REPT_FOR_STANDARD_REDUCE_TYPES(MACRO, OP) \
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

#define UCL_SHMEM_DEVICE_COLL_REPT_FOR_ARITH_REDUCE_TYPES(MACRO, OP) \
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

// ===== Reduce =====
#define UCL_SHMEM_DEVICE_COLL_DECL_TYPENAME_OP_REDUCE(TYPENAME, TYPE, OP) \
    NVSHMEMI_STATIC NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_INLINE int shmem_##TYPENAME##_##OP##_reduce(nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nreduce);

UCL_SHMEM_DEVICE_COLL_REPT_FOR_BITWISE_REDUCE_TYPES(UCL_SHMEM_DEVICE_COLL_DECL_TYPENAME_OP_REDUCE, and)
UCL_SHMEM_DEVICE_COLL_REPT_FOR_BITWISE_REDUCE_TYPES(UCL_SHMEM_DEVICE_COLL_DECL_TYPENAME_OP_REDUCE, or)
UCL_SHMEM_DEVICE_COLL_REPT_FOR_BITWISE_REDUCE_TYPES(UCL_SHMEM_DEVICE_COLL_DECL_TYPENAME_OP_REDUCE, xor)

UCL_SHMEM_DEVICE_COLL_REPT_FOR_STANDARD_REDUCE_TYPES(UCL_SHMEM_DEVICE_COLL_DECL_TYPENAME_OP_REDUCE, max)
UCL_SHMEM_DEVICE_COLL_REPT_FOR_STANDARD_REDUCE_TYPES(UCL_SHMEM_DEVICE_COLL_DECL_TYPENAME_OP_REDUCE, min)

UCL_SHMEM_DEVICE_COLL_REPT_FOR_ARITH_REDUCE_TYPES(UCL_SHMEM_DEVICE_COLL_DECL_TYPENAME_OP_REDUCE, sum)
UCL_SHMEM_DEVICE_COLL_REPT_FOR_ARITH_REDUCE_TYPES(UCL_SHMEM_DEVICE_COLL_DECL_TYPENAME_OP_REDUCE, prod)

// ===== Reducescatter =====
#define UCL_SHMEM_DEVICE_COLL_DECL_TYPENAME_OP_REDUCESCATTER(TYPENAME, TYPE, OP) \
    NVSHMEMI_STATIC NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_INLINE int shmem_##TYPENAME##_##OP##_reducescatter(nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nreduce);

UCL_SHMEM_DEVICE_COLL_REPT_FOR_BITWISE_REDUCE_TYPES(UCL_SHMEM_DEVICE_COLL_DECL_TYPENAME_OP_REDUCESCATTER, and)
UCL_SHMEM_DEVICE_COLL_REPT_FOR_BITWISE_REDUCE_TYPES(UCL_SHMEM_DEVICE_COLL_DECL_TYPENAME_OP_REDUCESCATTER, or)
UCL_SHMEM_DEVICE_COLL_REPT_FOR_BITWISE_REDUCE_TYPES(UCL_SHMEM_DEVICE_COLL_DECL_TYPENAME_OP_REDUCESCATTER, xor)

UCL_SHMEM_DEVICE_COLL_REPT_FOR_STANDARD_REDUCE_TYPES(UCL_SHMEM_DEVICE_COLL_DECL_TYPENAME_OP_REDUCESCATTER, max)
UCL_SHMEM_DEVICE_COLL_REPT_FOR_STANDARD_REDUCE_TYPES(UCL_SHMEM_DEVICE_COLL_DECL_TYPENAME_OP_REDUCESCATTER, min)

UCL_SHMEM_DEVICE_COLL_REPT_FOR_ARITH_REDUCE_TYPES(UCL_SHMEM_DEVICE_COLL_DECL_TYPENAME_OP_REDUCESCATTER, sum)
UCL_SHMEM_DEVICE_COLL_REPT_FOR_ARITH_REDUCE_TYPES(UCL_SHMEM_DEVICE_COLL_DECL_TYPENAME_OP_REDUCESCATTER, prod)

#endif // __CUDA_ARCH__

} // namespace shmem
} // namespace ucl

#endif // UCL_SHMEM_DEVICE_COLL_DEFINES_CUH
