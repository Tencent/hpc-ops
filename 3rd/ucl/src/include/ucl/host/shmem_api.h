/*
 * UCL SHMEM API - Host API
 * This file enhances NVSHMEM host API functions into UCL shmem namespace
 */

#ifndef UCL_SHMEM_HOST_API_H
#define UCL_SHMEM_HOST_API_H

#include "host/nvshmem_api.h"

namespace ucl {
namespace shmem {

// ===== Library Initialization =====
static inline void shmem_init() { nvshmem_init(); }
static inline int shmem_init_thread(int requested, int *provided) {
    return nvshmem_init_thread(requested, provided);
}
static inline void shmem_query_thread(int *provided) { nvshmem_query_thread(provided); }
NVSHMEMI_HOSTDEVICE_PREFIX static inline void shmem_global_exit(int status) { nvshmem_global_exit(status); }
static inline void shmem_finalize() { nvshmem_finalize(); }
static inline int shmemx_init_status() { return nvshmemx_init_status(); }

// ===== PE Info Query =====
NVSHMEMI_HOSTDEVICE_PREFIX static inline int shmem_my_pe() { return nvshmem_my_pe(); }
NVSHMEMI_HOSTDEVICE_PREFIX static inline int shmem_n_pes() { return nvshmem_n_pes(); }
NVSHMEMI_HOSTDEVICE_PREFIX static inline void shmem_info_get_version(int *major, int *minor) {
    nvshmem_info_get_version(major, minor);
}
NVSHMEMI_HOSTDEVICE_PREFIX static inline void shmem_info_get_name(char *name) {
    nvshmem_info_get_name(name);
}
NVSHMEMI_HOSTDEVICE_PREFIX static inline void shmemx_vendor_get_version_info(int *major, int *minor, int *patch) {
    nvshmemx_vendor_get_version_info(major, minor, patch);
}

// ===== Heap Management =====
static inline void *shmem_malloc(size_t size) { return nvshmem_malloc(size); }
static inline void *shmem_calloc(size_t count, size_t size) { return nvshmem_calloc(count, size); }
static inline void *shmem_align(size_t alignment, size_t size) { return nvshmem_align(alignment, size); }
static inline void shmem_free(void *ptr) { nvshmem_free(ptr); }
static inline void *shmem_realloc(void *ptr, size_t size) { return nvshmem_realloc(ptr, size); }
NVSHMEMI_HOSTDEVICE_PREFIX static inline void *shmem_ptr(const void *ptr, int pe) {
    return nvshmem_ptr(ptr, pe);
}
NVSHMEMI_HOSTDEVICE_PREFIX static inline void *shmemx_mc_ptr(nvshmem_team_t team, const void *ptr) {
    return nvshmemx_mc_ptr(team, ptr);
}

// ===== Atomic Operations Macros =====
#define UCL_SHMEM_DECL_ATOMIC_INC(type, TYPE) \
    NVSHMEMI_HOSTDEVICE_PREFIX static inline void shmem_##type##_atomic_inc(TYPE *dest, int pe) { \
        nvshmem_##type##_atomic_inc(dest, pe); \
    }

#define UCL_SHMEM_DECL_ATOMIC_FINC(type, TYPE) \
    NVSHMEMI_HOSTDEVICE_PREFIX static inline TYPE shmem_##type##_atomic_fetch_inc(TYPE *dest, int pe) { \
        return nvshmem_##type##_atomic_fetch_inc(dest, pe); \
    }

#define UCL_SHMEM_DECL_ATOMIC_FETCH(type, TYPE) \
    NVSHMEMI_HOSTDEVICE_PREFIX static inline TYPE shmem_##type##_atomic_fetch(const TYPE *dest, int pe) { \
        return nvshmem_##type##_atomic_fetch(dest, pe); \
    }

#define UCL_SHMEM_DECL_ATOMIC_ADD(type, TYPE) \
    NVSHMEMI_HOSTDEVICE_PREFIX static inline void shmem_##type##_atomic_add(TYPE *dest, TYPE value, int pe) { \
        nvshmem_##type##_atomic_add(dest, value, pe); \
    }

#define UCL_SHMEM_DECL_ATOMIC_SET(type, TYPE) \
    NVSHMEMI_HOSTDEVICE_PREFIX static inline void shmem_##type##_atomic_set(TYPE *dest, TYPE value, int pe) { \
        nvshmem_##type##_atomic_set(dest, value, pe); \
    }

#define UCL_SHMEM_DECL_ATOMIC_FADD(type, TYPE) \
    NVSHMEMI_HOSTDEVICE_PREFIX static inline TYPE shmem_##type##_atomic_fetch_add(TYPE *dest, TYPE value, int pe) { \
        return nvshmem_##type##_atomic_fetch_add(dest, value, pe); \
    }

#define UCL_SHMEM_DECL_ATOMIC_SWAP(type, TYPE) \
    NVSHMEMI_HOSTDEVICE_PREFIX static inline TYPE shmem_##type##_atomic_swap(TYPE *dest, TYPE value, int pe) { \
        return nvshmem_##type##_atomic_swap(dest, value, pe); \
    }

#define UCL_SHMEM_DECL_ATOMIC_CSWAP(type, TYPE) \
    NVSHMEMI_HOSTDEVICE_PREFIX static inline TYPE shmem_##type##_atomic_compare_swap(TYPE *dest, TYPE cond, TYPE value, int pe) { \
        return nvshmem_##type##_atomic_compare_swap(dest, cond, value, pe); \
    }

#define UCL_SHMEM_DECL_ATOMIC_AND(type, TYPE) \
    NVSHMEMI_HOSTDEVICE_PREFIX static inline void shmem_##type##_atomic_and(TYPE *dest, TYPE value, int pe) { \
        nvshmem_##type##_atomic_and(dest, value, pe); \
    }

#define UCL_SHMEM_DECL_ATOMIC_OR(type, TYPE) \
    NVSHMEMI_HOSTDEVICE_PREFIX static inline void shmem_##type##_atomic_or(TYPE *dest, TYPE value, int pe) { \
        nvshmem_##type##_atomic_or(dest, value, pe); \
    }

#define UCL_SHMEM_DECL_ATOMIC_XOR(type, TYPE) \
    NVSHMEMI_HOSTDEVICE_PREFIX static inline void shmem_##type##_atomic_xor(TYPE *dest, TYPE value, int pe) { \
        nvshmem_##type##_atomic_xor(dest, value, pe); \
    }

#define UCL_SHMEM_DECL_ATOMIC_FAND(type, TYPE) \
    NVSHMEMI_HOSTDEVICE_PREFIX static inline TYPE shmem_##type##_atomic_fetch_and(TYPE *dest, TYPE value, int pe) { \
        return nvshmem_##type##_atomic_fetch_and(dest, value, pe); \
    }

#define UCL_SHMEM_DECL_ATOMIC_FOR(type, TYPE) \
    NVSHMEMI_HOSTDEVICE_PREFIX static inline TYPE shmem_##type##_atomic_fetch_or(TYPE *dest, TYPE value, int pe) { \
        return nvshmem_##type##_atomic_fetch_or(dest, value, pe); \
    }

#define UCL_SHMEM_DECL_ATOMIC_FXOR(type, TYPE) \
    NVSHMEMI_HOSTDEVICE_PREFIX static inline TYPE shmem_##type##_atomic_fetch_xor(TYPE *dest, TYPE value, int pe) { \
        return nvshmem_##type##_atomic_fetch_xor(dest, value, pe); \
    }

// Bitwise AMO types
#define UCL_SHMEM_REPT_FOR_BITWISE_AMO(MACRO) \
    MACRO(uint, unsigned int) \
    MACRO(ulong, unsigned long) \
    MACRO(ulonglong, unsigned long long) \
    MACRO(int32, int32_t) \
    MACRO(uint32, uint32_t) \
    MACRO(int64, int64_t) \
    MACRO(uint64, uint64_t)

// Standard AMO types
#define UCL_SHMEM_REPT_FOR_STANDARD_AMO(MACRO) \
    MACRO(int, int) \
    MACRO(long, long) \
    MACRO(longlong, long long) \
    MACRO(size, size_t) \
    MACRO(ptrdiff, ptrdiff_t)

// Extended AMO types
#define UCL_SHMEM_REPT_FOR_EXTENDED_AMO(MACRO) \
    MACRO(float, float) \
    MACRO(double, double)

// Generate atomic operations
UCL_SHMEM_REPT_FOR_BITWISE_AMO(UCL_SHMEM_DECL_ATOMIC_INC)
UCL_SHMEM_REPT_FOR_STANDARD_AMO(UCL_SHMEM_DECL_ATOMIC_INC)

UCL_SHMEM_REPT_FOR_BITWISE_AMO(UCL_SHMEM_DECL_ATOMIC_FINC)
UCL_SHMEM_REPT_FOR_STANDARD_AMO(UCL_SHMEM_DECL_ATOMIC_FINC)

UCL_SHMEM_REPT_FOR_BITWISE_AMO(UCL_SHMEM_DECL_ATOMIC_FETCH)
UCL_SHMEM_REPT_FOR_STANDARD_AMO(UCL_SHMEM_DECL_ATOMIC_FETCH)
UCL_SHMEM_REPT_FOR_EXTENDED_AMO(UCL_SHMEM_DECL_ATOMIC_FETCH)

UCL_SHMEM_REPT_FOR_BITWISE_AMO(UCL_SHMEM_DECL_ATOMIC_ADD)
UCL_SHMEM_REPT_FOR_STANDARD_AMO(UCL_SHMEM_DECL_ATOMIC_ADD)

UCL_SHMEM_REPT_FOR_BITWISE_AMO(UCL_SHMEM_DECL_ATOMIC_SET)
UCL_SHMEM_REPT_FOR_STANDARD_AMO(UCL_SHMEM_DECL_ATOMIC_SET)
UCL_SHMEM_REPT_FOR_EXTENDED_AMO(UCL_SHMEM_DECL_ATOMIC_SET)

UCL_SHMEM_REPT_FOR_BITWISE_AMO(UCL_SHMEM_DECL_ATOMIC_FADD)
UCL_SHMEM_REPT_FOR_STANDARD_AMO(UCL_SHMEM_DECL_ATOMIC_FADD)

UCL_SHMEM_REPT_FOR_BITWISE_AMO(UCL_SHMEM_DECL_ATOMIC_SWAP)
UCL_SHMEM_REPT_FOR_STANDARD_AMO(UCL_SHMEM_DECL_ATOMIC_SWAP)
UCL_SHMEM_REPT_FOR_EXTENDED_AMO(UCL_SHMEM_DECL_ATOMIC_SWAP)

UCL_SHMEM_REPT_FOR_BITWISE_AMO(UCL_SHMEM_DECL_ATOMIC_CSWAP)
UCL_SHMEM_REPT_FOR_STANDARD_AMO(UCL_SHMEM_DECL_ATOMIC_CSWAP)

UCL_SHMEM_REPT_FOR_BITWISE_AMO(UCL_SHMEM_DECL_ATOMIC_AND)
UCL_SHMEM_REPT_FOR_BITWISE_AMO(UCL_SHMEM_DECL_ATOMIC_OR)
UCL_SHMEM_REPT_FOR_BITWISE_AMO(UCL_SHMEM_DECL_ATOMIC_XOR)

UCL_SHMEM_REPT_FOR_BITWISE_AMO(UCL_SHMEM_DECL_ATOMIC_FAND)
UCL_SHMEM_REPT_FOR_BITWISE_AMO(UCL_SHMEM_DECL_ATOMIC_FOR)
UCL_SHMEM_REPT_FOR_BITWISE_AMO(UCL_SHMEM_DECL_ATOMIC_FXOR)

// ===== RMA Put Operations =====
#define UCL_SHMEM_DECL_TYPE_P(NAME, TYPE) \
    NVSHMEMI_HOSTDEVICE_PREFIX static inline void shmem_##NAME##_p(TYPE *dest, const TYPE value, int pe) { \
        nvshmem_##NAME##_p(dest, value, pe); \
    }

#define UCL_SHMEM_DECL_TYPE_PUT(NAME, TYPE) \
    NVSHMEMI_HOSTDEVICE_PREFIX static inline void shmem_##NAME##_put(TYPE *dest, const TYPE *source, size_t nelems, int pe) { \
        nvshmem_##NAME##_put(dest, source, nelems, pe); \
    }

#define UCL_SHMEM_DECL_TYPE_IPUT(NAME, TYPE) \
    NVSHMEMI_HOSTDEVICE_PREFIX static inline void shmem_##NAME##_iput(TYPE *dest, const TYPE *source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe) { \
        nvshmem_##NAME##_iput(dest, source, dst, sst, nelems, pe); \
    }

#define UCL_SHMEM_DECL_TYPE_PUT_NBI(NAME, TYPE) \
    NVSHMEMI_HOSTDEVICE_PREFIX static inline void shmem_##NAME##_put_nbi(TYPE *dest, const TYPE *source, size_t nelems, int pe) { \
        nvshmem_##NAME##_put_nbi(dest, source, nelems, pe); \
    }

// Standard RMA types
#define UCL_SHMEM_REPT_FOR_STANDARD_RMA_TYPES(MACRO) \
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

UCL_SHMEM_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEM_DECL_TYPE_P)
UCL_SHMEM_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEM_DECL_TYPE_PUT)
UCL_SHMEM_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEM_DECL_TYPE_IPUT)
UCL_SHMEM_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEM_DECL_TYPE_PUT_NBI)

// Size-based put operations
#define UCL_SHMEM_DECL_SIZE_PUT(NAME) \
    NVSHMEMI_HOSTDEVICE_PREFIX static inline void shmem_put##NAME(void *dest, const void *source, size_t nelems, int pe) { \
        nvshmem_put##NAME(dest, source, nelems, pe); \
    }

#define UCL_SHMEM_DECL_SIZE_IPUT(NAME) \
    NVSHMEMI_HOSTDEVICE_PREFIX static inline void shmem_iput##NAME(void *dest, const void *source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe) { \
        nvshmem_iput##NAME(dest, source, dst, sst, nelems, pe); \
    }

#define UCL_SHMEM_DECL_SIZE_PUT_NBI(NAME) \
    NVSHMEMI_HOSTDEVICE_PREFIX static inline void shmem_put##NAME##_nbi(void *dest, const void *source, size_t nelems, int pe) { \
        nvshmem_put##NAME##_nbi(dest, source, nelems, pe); \
    }

UCL_SHMEM_DECL_SIZE_PUT(8)
UCL_SHMEM_DECL_SIZE_PUT(16)
UCL_SHMEM_DECL_SIZE_PUT(32)
UCL_SHMEM_DECL_SIZE_PUT(64)
UCL_SHMEM_DECL_SIZE_PUT(128)

UCL_SHMEM_DECL_SIZE_IPUT(8)
UCL_SHMEM_DECL_SIZE_IPUT(16)
UCL_SHMEM_DECL_SIZE_IPUT(32)
UCL_SHMEM_DECL_SIZE_IPUT(64)
UCL_SHMEM_DECL_SIZE_IPUT(128)

UCL_SHMEM_DECL_SIZE_PUT_NBI(8)
UCL_SHMEM_DECL_SIZE_PUT_NBI(16)
UCL_SHMEM_DECL_SIZE_PUT_NBI(32)
UCL_SHMEM_DECL_SIZE_PUT_NBI(64)
UCL_SHMEM_DECL_SIZE_PUT_NBI(128)

NVSHMEMI_HOSTDEVICE_PREFIX static inline void shmem_putmem(void *dest, const void *source, size_t bytes, int pe) {
    nvshmem_putmem(dest, source, bytes, pe);
}

NVSHMEMI_HOSTDEVICE_PREFIX static inline void shmem_putmem_nbi(void *dest, const void *source, size_t bytes, int pe) {
    nvshmem_putmem_nbi(dest, source, bytes, pe);
}

// ===== RMA Get Operations =====
#define UCL_SHMEM_DECL_TYPE_G(NAME, TYPE) \
    NVSHMEMI_HOSTDEVICE_PREFIX static inline TYPE shmem_##NAME##_g(const TYPE *src, int pe) { \
        return nvshmem_##NAME##_g(src, pe); \
    }

#define UCL_SHMEM_DECL_TYPE_GET(NAME, TYPE) \
    NVSHMEMI_HOSTDEVICE_PREFIX static inline void shmem_##NAME##_get(TYPE *dest, const TYPE *source, size_t nelems, int pe) { \
        nvshmem_##NAME##_get(dest, source, nelems, pe); \
    }

#define UCL_SHMEM_DECL_TYPE_IGET(NAME, TYPE) \
    NVSHMEMI_HOSTDEVICE_PREFIX static inline void shmem_##NAME##_iget(TYPE *dest, const TYPE *source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe) { \
        nvshmem_##NAME##_iget(dest, source, dst, sst, nelems, pe); \
    }

#define UCL_SHMEM_DECL_TYPE_GET_NBI(NAME, TYPE) \
    NVSHMEMI_HOSTDEVICE_PREFIX static inline void shmem_##NAME##_get_nbi(TYPE *dest, const TYPE *source, size_t nelems, int pe) { \
        nvshmem_##NAME##_get_nbi(dest, source, nelems, pe); \
    }

UCL_SHMEM_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEM_DECL_TYPE_G)
UCL_SHMEM_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEM_DECL_TYPE_GET)
UCL_SHMEM_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEM_DECL_TYPE_IGET)
UCL_SHMEM_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEM_DECL_TYPE_GET_NBI)

// Size-based get operations
#define UCL_SHMEM_DECL_SIZE_GET(NAME) \
    NVSHMEMI_HOSTDEVICE_PREFIX static inline void shmem_get##NAME(void *dest, const void *source, size_t nelems, int pe) { \
        nvshmem_get##NAME(dest, source, nelems, pe); \
    }

#define UCL_SHMEM_DECL_SIZE_IGET(NAME) \
    NVSHMEMI_HOSTDEVICE_PREFIX static inline void shmem_iget##NAME(void *dest, const void *source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe) { \
        nvshmem_iget##NAME(dest, source, dst, sst, nelems, pe); \
    }

#define UCL_SHMEM_DECL_SIZE_GET_NBI(NAME) \
    NVSHMEMI_HOSTDEVICE_PREFIX static inline void shmem_get##NAME##_nbi(void *dest, const void *source, size_t nelems, int pe) { \
        nvshmem_get##NAME##_nbi(dest, source, nelems, pe); \
    }

UCL_SHMEM_DECL_SIZE_GET(8)
UCL_SHMEM_DECL_SIZE_GET(16)
UCL_SHMEM_DECL_SIZE_GET(32)
UCL_SHMEM_DECL_SIZE_GET(64)
UCL_SHMEM_DECL_SIZE_GET(128)

UCL_SHMEM_DECL_SIZE_IGET(8)
UCL_SHMEM_DECL_SIZE_IGET(16)
UCL_SHMEM_DECL_SIZE_IGET(32)
UCL_SHMEM_DECL_SIZE_IGET(64)
UCL_SHMEM_DECL_SIZE_IGET(128)

UCL_SHMEM_DECL_SIZE_GET_NBI(8)
UCL_SHMEM_DECL_SIZE_GET_NBI(16)
UCL_SHMEM_DECL_SIZE_GET_NBI(32)
UCL_SHMEM_DECL_SIZE_GET_NBI(64)
UCL_SHMEM_DECL_SIZE_GET_NBI(128)

NVSHMEMI_HOSTDEVICE_PREFIX static inline void shmem_getmem(void *dest, const void *source, size_t bytes, int pe) {
    nvshmem_getmem(dest, source, bytes, pe);
}

NVSHMEMI_HOSTDEVICE_PREFIX static inline void shmem_getmem_nbi(void *dest, const void *source, size_t bytes, int pe) {
    nvshmem_getmem_nbi(dest, source, bytes, pe);
}

// ===== Signal API =====
#ifdef __CUDACC__
#define UCL_SHMEM_DECL_PUT_SIGNAL(TYPENAME, TYPE) \
    __device__ void shmem_##TYPENAME##_put_signal(TYPE *dest, const TYPE *source, size_t nelems, \
                                                                 uint64_t *sig_addr, uint64_t signal, \
                                                   int sig_op, int pe);

#define UCL_SHMEM_DECL_PUT_SIGNAL_NBI(TYPENAME, TYPE) \
    __device__ void shmem_##TYPENAME##_put_signal_nbi(TYPE *dest, const TYPE *source, \
                                                                     size_t nelems, uint64_t *sig_addr, \
                                                       uint64_t signal, int sig_op, int pe);

UCL_SHMEM_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEM_DECL_PUT_SIGNAL)
UCL_SHMEM_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEM_DECL_PUT_SIGNAL_NBI)

#define UCL_SHMEM_DECL_SIZE_PUT_SIGNAL(BITS) \
    __device__ void shmem_put##BITS##_signal(void *dest, const void *source, size_t nelems, \
                                              uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);

#define UCL_SHMEM_DECL_SIZE_PUT_SIGNAL_NBI(BITS) \
    __device__ void shmem_put##BITS##_signal_nbi(void *dest, const void *source, size_t nelems, \
                                                  uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);

UCL_SHMEM_DECL_SIZE_PUT_SIGNAL(8)
UCL_SHMEM_DECL_SIZE_PUT_SIGNAL(16)
UCL_SHMEM_DECL_SIZE_PUT_SIGNAL(32)
UCL_SHMEM_DECL_SIZE_PUT_SIGNAL(64)
UCL_SHMEM_DECL_SIZE_PUT_SIGNAL(128)

UCL_SHMEM_DECL_SIZE_PUT_SIGNAL_NBI(8)
UCL_SHMEM_DECL_SIZE_PUT_SIGNAL_NBI(16)
UCL_SHMEM_DECL_SIZE_PUT_SIGNAL_NBI(32)
UCL_SHMEM_DECL_SIZE_PUT_SIGNAL_NBI(64)
UCL_SHMEM_DECL_SIZE_PUT_SIGNAL_NBI(128)

__device__ void shmem_putmem_signal(void *dest, const void *source, size_t bytes,
                                     uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);

__device__ void shmem_putmem_signal_nbi(void *dest, const void *source, size_t bytes,
                                         uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
#endif // __CUDACC__

NVSHMEMI_HOSTDEVICE_PREFIX static inline uint64_t shmem_signal_fetch(uint64_t *sig_addr) {
    return nvshmem_signal_fetch(sig_addr);
}

// ===== Point-to-Point Synchronization =====
NVSHMEMI_HOSTDEVICE_PREFIX static inline void shmem_quiet() { nvshmem_quiet(); }
NVSHMEMI_HOSTDEVICE_PREFIX static inline void shmem_fence() { nvshmem_fence(); }

#ifdef __CUDACC__
__device__ static inline uint64_t shmem_signal_wait_until(uint64_t *sig_addr, int cmp, uint64_t cmp_val) {
    return nvshmem_signal_wait_until(sig_addr, cmp, cmp_val);
}

#define UCL_SHMEM_DECL_WAIT_UNTIL(NAME, TYPE) \
    __device__ void shmem_##NAME##_wait_until(TYPE *ivar, int cmp, TYPE cmp_value);

#define UCL_SHMEM_DECL_WAIT_UNTIL_ALL(NAME, TYPE) \
    __device__ void shmem_##NAME##_wait_until_all(TYPE *ivar, size_t nelems, const int *status, int cmp, TYPE cmp_value);

#define UCL_SHMEM_DECL_WAIT_UNTIL_ANY(NAME, TYPE) \
    __device__ size_t shmem_##NAME##_wait_until_any(TYPE *ivar, size_t nelems, const int *status, int cmp, TYPE cmp_value);

#define UCL_SHMEM_DECL_WAIT_UNTIL_SOME(NAME, TYPE) \
    __device__ size_t shmem_##NAME##_wait_until_some(TYPE *ivar, size_t nelems, size_t *indices, const int *status, int cmp, TYPE cmp_value);

#define UCL_SHMEM_DECL_WAIT_UNTIL_ALL_VECTOR(NAME, TYPE) \
    __device__ void shmem_##NAME##_wait_until_all_vector(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE *cmp_values);

#define UCL_SHMEM_DECL_WAIT_UNTIL_ANY_VECTOR(NAME, TYPE) \
    __device__ size_t shmem_##NAME##_wait_until_any_vector(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE *cmp_values);

#define UCL_SHMEM_DECL_WAIT_UNTIL_SOME_VECTOR(NAME, TYPE) \
    __device__ size_t shmem_##NAME##_wait_until_some_vector(TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp, TYPE *cmp_values);

#define UCL_SHMEM_DECL_TEST(NAME, TYPE) \
    __device__ int shmem_##NAME##_test(TYPE *ivar, int cmp, TYPE cmp_value);

#define UCL_SHMEM_DECL_TEST_ALL(NAME, TYPE) \
    __device__ int shmem_##NAME##_test_all(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE cmp_value);

#define UCL_SHMEM_DECL_TEST_ANY(NAME, TYPE) \
    __device__ size_t shmem_##NAME##_test_any(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE cmp_value);

#define UCL_SHMEM_DECL_TEST_SOME(NAME, TYPE) \
    __device__ size_t shmem_##NAME##_test_some(TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp, TYPE cmp_value);

#define UCL_SHMEM_DECL_TEST_ALL_VECTOR(NAME, TYPE) \
    __device__ int shmem_##NAME##_test_all_vector(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE *cmp_values);

#define UCL_SHMEM_DECL_TEST_ANY_VECTOR(NAME, TYPE) \
    __device__ size_t shmem_##NAME##_test_any_vector(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE *cmp_values);

#define UCL_SHMEM_DECL_TEST_SOME_VECTOR(NAME, TYPE) \
    __device__ size_t shmem_##NAME##_test_some_vector(TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp, TYPE *cmp_values);

// Wait types
#define UCL_SHMEM_REPT_FOR_WAIT_TYPES(MACRO) \
    MACRO(short, short) \
    MACRO(int, int) \
    MACRO(long, long) \
    MACRO(longlong, long long) \
    MACRO(ushort, unsigned short) \
    MACRO(uint, unsigned int) \
    MACRO(ulong, unsigned long) \
    MACRO(ulonglong, unsigned long long) \
    MACRO(int32, int32_t) \
    MACRO(int64, int64_t) \
    MACRO(uint32, uint32_t) \
    MACRO(uint64, uint64_t) \
    MACRO(size, size_t) \
    MACRO(ptrdiff, ptrdiff_t)

UCL_SHMEM_REPT_FOR_WAIT_TYPES(UCL_SHMEM_DECL_WAIT_UNTIL)
UCL_SHMEM_REPT_FOR_WAIT_TYPES(UCL_SHMEM_DECL_WAIT_UNTIL_ALL)
UCL_SHMEM_REPT_FOR_WAIT_TYPES(UCL_SHMEM_DECL_WAIT_UNTIL_ANY)
UCL_SHMEM_REPT_FOR_WAIT_TYPES(UCL_SHMEM_DECL_WAIT_UNTIL_SOME)
UCL_SHMEM_REPT_FOR_WAIT_TYPES(UCL_SHMEM_DECL_WAIT_UNTIL_ALL_VECTOR)
UCL_SHMEM_REPT_FOR_WAIT_TYPES(UCL_SHMEM_DECL_WAIT_UNTIL_ANY_VECTOR)
UCL_SHMEM_REPT_FOR_WAIT_TYPES(UCL_SHMEM_DECL_WAIT_UNTIL_SOME_VECTOR)
UCL_SHMEM_REPT_FOR_WAIT_TYPES(UCL_SHMEM_DECL_TEST)
UCL_SHMEM_REPT_FOR_WAIT_TYPES(UCL_SHMEM_DECL_TEST_ALL)
UCL_SHMEM_REPT_FOR_WAIT_TYPES(UCL_SHMEM_DECL_TEST_ANY)
UCL_SHMEM_REPT_FOR_WAIT_TYPES(UCL_SHMEM_DECL_TEST_SOME)
UCL_SHMEM_REPT_FOR_WAIT_TYPES(UCL_SHMEM_DECL_TEST_ALL_VECTOR)
UCL_SHMEM_REPT_FOR_WAIT_TYPES(UCL_SHMEM_DECL_TEST_ANY_VECTOR)
UCL_SHMEM_REPT_FOR_WAIT_TYPES(UCL_SHMEM_DECL_TEST_SOME_VECTOR)
#endif // __CUDACC__

// ===== Teams API =====
NVSHMEMI_HOSTDEVICE_PREFIX static inline int shmem_team_my_pe(nvshmem_team_t team) {
    return nvshmem_team_my_pe(team);
}
NVSHMEMI_HOSTDEVICE_PREFIX static inline int shmem_team_n_pes(nvshmem_team_t team) {
    return nvshmem_team_n_pes(team);
}
static inline void shmem_team_get_config(nvshmem_team_t team, nvshmem_team_config_t *config) {
    nvshmem_team_get_config(team, config);
}
NVSHMEMI_HOSTDEVICE_PREFIX static inline int shmem_team_translate_pe(nvshmem_team_t src_team, int src_pe, nvshmem_team_t dest_team) {
    return nvshmem_team_translate_pe(src_team, src_pe, dest_team);
}
static inline int shmem_team_split_strided(nvshmem_team_t parent_team, int PE_start, int PE_stride, int PE_size,
                                           const nvshmem_team_config_t *config, long config_mask, nvshmem_team_t *new_team) {
    return nvshmem_team_split_strided(parent_team, PE_start, PE_stride, PE_size, config, config_mask, new_team);
}
static inline int shmemx_team_get_uniqueid(nvshmemx_team_uniqueid_t *uniqueid) {
    return nvshmemx_team_get_uniqueid(uniqueid);
}
static inline int shmemx_team_init(nvshmem_team_t *team, nvshmem_team_config_t *config, long config_mask, int npes, int pe_idx_in_team) {
    return nvshmemx_team_init(team, config, config_mask, npes, pe_idx_in_team);
}
static inline int shmem_team_split_2d(nvshmem_team_t parent_team, int xrange,
                                      const nvshmem_team_config_t *xaxis_config, long xaxis_mask,
                                      nvshmem_team_t *xaxis_team, const nvshmem_team_config_t *yaxis_config,
                                      long yaxis_mask, nvshmem_team_t *yaxis_team) {
    return nvshmem_team_split_2d(parent_team, xrange, xaxis_config, xaxis_mask, xaxis_team, yaxis_config, yaxis_mask, yaxis_team);
}
static inline void shmem_team_destroy(nvshmem_team_t team) {
    nvshmem_team_destroy(team);
}

} // namespace shmem
} // namespace ucl

#endif // UCL_SHMEM_HOST_API_H
