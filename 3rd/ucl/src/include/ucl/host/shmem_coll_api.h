/*
 * UCL SHMEM API - Host Collective API
 * This file enhances NVSHMEM host collective API functions into UCL shmem namespace
 */

#ifndef UCL_SHMEM_HOST_COLL_API_H
#define UCL_SHMEM_HOST_COLL_API_H

#include "host/nvshmem_coll_api.h"

namespace ucl {
namespace shmem {

// ===== Alltoall Collectives =====
#define UCL_SHMEM_DECL_TYPENAME_ALLTOALL(TYPENAME, TYPE) \
    NVSHMEMI_HOSTDEVICE_PREFIX static inline int shmem_##TYPENAME##_alltoall(nvshmem_team_t team, TYPE *dest, \
                                                                              const TYPE *src, size_t nelems) { \
        return nvshmem_##TYPENAME##_alltoall(team, dest, src, nelems); \
    }

#define UCL_SHMEM_DECL_TYPENAME_ALLTOALLS(TYPENAME, TYPE) \
    NVSHMEMI_HOSTDEVICE_PREFIX static inline int shmem_##TYPENAME##_alltoalls(nvshmem_team_t team, TYPE *dest, \
                                                                               const TYPE *src, ptrdiff_t dst, \
                                                                               ptrdiff_t sst, size_t nelems) { \
        return nvshmem_##TYPENAME##_alltoalls(team, dest, src, dst, sst, nelems); \
    }

#define UCL_SHMEM_REPT_FOR_STANDARD_RMA_TYPES_COLL(MACRO) \
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

UCL_SHMEM_REPT_FOR_STANDARD_RMA_TYPES_COLL(UCL_SHMEM_DECL_TYPENAME_ALLTOALL)
UCL_SHMEM_REPT_FOR_STANDARD_RMA_TYPES_COLL(UCL_SHMEM_DECL_TYPENAME_ALLTOALLS)

NVSHMEMI_HOSTDEVICE_PREFIX static inline int shmem_alltoallmem(nvshmem_team_t team, void *dest, const void *src, size_t nelems) {
    return nvshmem_alltoallmem(team, dest, src, nelems);
}

// ===== Barrier Collectives =====
NVSHMEMI_HOSTDEVICE_PREFIX static inline int shmem_barrier(nvshmem_team_t team) {
    return nvshmem_barrier(team);
}
NVSHMEMI_HOSTDEVICE_PREFIX static inline void shmem_barrier_all() {
    nvshmem_barrier_all();
}

// ===== Sync Collectives =====
NVSHMEMI_HOSTDEVICE_PREFIX static inline int shmem_team_sync(nvshmem_team_t team) {
    return nvshmem_team_sync(team);
}
NVSHMEMI_HOSTDEVICE_PREFIX static inline void shmem_sync_all() {
    nvshmem_sync_all();
}
#define shmem_sync shmem_team_sync

// ===== Broadcast Collectives =====
#define UCL_SHMEM_DECL_TYPENAME_BROADCAST(TYPENAME, TYPE) \
    NVSHMEMI_HOSTDEVICE_PREFIX static inline int shmem_##TYPENAME##_broadcast(nvshmem_team_t team, TYPE *dest, \
                                                                               const TYPE *src, size_t nelem, int PE_root) { \
        return nvshmem_##TYPENAME##_broadcast(team, dest, src, nelem, PE_root); \
    }

UCL_SHMEM_REPT_FOR_STANDARD_RMA_TYPES_COLL(UCL_SHMEM_DECL_TYPENAME_BROADCAST)

NVSHMEMI_HOSTDEVICE_PREFIX static inline int shmem_broadcastmem(nvshmem_team_t team, void *dest, const void *src, size_t nelems, int PE_root) {
    return nvshmem_broadcastmem(team, dest, src, nelems, PE_root);
}

// ===== Fcollect Collectives =====
#define UCL_SHMEM_DECL_TYPENAME_FCOLLECT(TYPENAME, TYPE) \
    NVSHMEMI_HOSTDEVICE_PREFIX static inline int shmem_##TYPENAME##_fcollect(nvshmem_team_t team, TYPE *dest, \
                                                                              const TYPE *src, size_t nelem) { \
        return nvshmem_##TYPENAME##_fcollect(team, dest, src, nelem); \
    }

UCL_SHMEM_REPT_FOR_STANDARD_RMA_TYPES_COLL(UCL_SHMEM_DECL_TYPENAME_FCOLLECT)

NVSHMEMI_HOSTDEVICE_PREFIX static inline int shmem_fcollectmem(nvshmem_team_t team, void *dest, const void *src, size_t nelems) {
    return nvshmem_fcollectmem(team, dest, src, nelems);
}

// ===== Reduction Collectives =====
// Bitwise reduce types
#define UCL_SHMEM_REPT_FOR_BITWISE_REDUCE_TYPES(MACRO, OP) \
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

// Standard reduce types
#define UCL_SHMEM_REPT_FOR_STANDARD_REDUCE_TYPES(MACRO, OP) \
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

// Arithmetic reduce types
#define UCL_SHMEM_REPT_FOR_ARITH_REDUCE_TYPES(MACRO, OP) \
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

#define UCL_SHMEM_DECL_TEAM_REDUCE(NAME, TYPE, OP) \
    NVSHMEMI_HOSTDEVICE_PREFIX static inline int shmem_##NAME##_##OP##_reduce(nvshmem_team_t team, TYPE *dest, \
                                                                               const TYPE *src, size_t nreduce) { \
        return nvshmem_##NAME##_##OP##_reduce(team, dest, src, nreduce); \
    }

UCL_SHMEM_REPT_FOR_BITWISE_REDUCE_TYPES(UCL_SHMEM_DECL_TEAM_REDUCE, and)
UCL_SHMEM_REPT_FOR_BITWISE_REDUCE_TYPES(UCL_SHMEM_DECL_TEAM_REDUCE, or)
UCL_SHMEM_REPT_FOR_BITWISE_REDUCE_TYPES(UCL_SHMEM_DECL_TEAM_REDUCE, xor)

UCL_SHMEM_REPT_FOR_STANDARD_REDUCE_TYPES(UCL_SHMEM_DECL_TEAM_REDUCE, max)
UCL_SHMEM_REPT_FOR_STANDARD_REDUCE_TYPES(UCL_SHMEM_DECL_TEAM_REDUCE, min)

UCL_SHMEM_REPT_FOR_ARITH_REDUCE_TYPES(UCL_SHMEM_DECL_TEAM_REDUCE, sum)
UCL_SHMEM_REPT_FOR_ARITH_REDUCE_TYPES(UCL_SHMEM_DECL_TEAM_REDUCE, prod)

// ===== Reducescatter Collectives =====
#define UCL_SHMEM_DECL_TEAM_REDUCESCATTER(NAME, TYPE, OP) \
    NVSHMEMI_HOSTDEVICE_PREFIX static inline int shmem_##NAME##_##OP##_reducescatter(nvshmem_team_t team, TYPE *dest, \
                                                                                      const TYPE *src, size_t nreduce) { \
        return nvshmem_##NAME##_##OP##_reducescatter(team, dest, src, nreduce); \
    }

UCL_SHMEM_REPT_FOR_BITWISE_REDUCE_TYPES(UCL_SHMEM_DECL_TEAM_REDUCESCATTER, and)
UCL_SHMEM_REPT_FOR_BITWISE_REDUCE_TYPES(UCL_SHMEM_DECL_TEAM_REDUCESCATTER, or)
UCL_SHMEM_REPT_FOR_BITWISE_REDUCE_TYPES(UCL_SHMEM_DECL_TEAM_REDUCESCATTER, xor)

UCL_SHMEM_REPT_FOR_STANDARD_REDUCE_TYPES(UCL_SHMEM_DECL_TEAM_REDUCESCATTER, max)
UCL_SHMEM_REPT_FOR_STANDARD_REDUCE_TYPES(UCL_SHMEM_DECL_TEAM_REDUCESCATTER, min)

UCL_SHMEM_REPT_FOR_ARITH_REDUCE_TYPES(UCL_SHMEM_DECL_TEAM_REDUCESCATTER, sum)
UCL_SHMEM_REPT_FOR_ARITH_REDUCE_TYPES(UCL_SHMEM_DECL_TEAM_REDUCESCATTER, prod)

} // namespace shmem
} // namespace ucl

#endif // UCL_SHMEM_HOST_COLL_API_H
