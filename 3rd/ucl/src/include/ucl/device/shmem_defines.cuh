/*
 * UCL SHMEM API - Device Defines
 * This file enhances NVSHMEM device API functions into UCL shmem namespace
 */

#ifndef UCL_SHMEM_DEVICE_DEFINES_CUH
#define UCL_SHMEM_DEVICE_DEFINES_CUH

#include "device/nvshmem_defines.h"

namespace ucl {
namespace shmem {

#ifdef __CUDA_ARCH__

// ===== PE Info Query =====
NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE int shmem_team_my_pe(nvshmem_team_t team);

NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE int shmem_team_n_pes(nvshmem_team_t team);

NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE int shmem_team_translate_pe(nvshmem_team_t src_team, int src_pe, nvshmem_team_t dest_team);

NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE int shmem_my_pe(void);

NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE int shmem_n_pes(void);

NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmem_info_get_name(char *name);

NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmem_info_get_version(int *major, int *minor);

// ===== Standard RMA Types =====
#define UCL_SHMEM_DEVICE_REPT_FOR_STANDARD_RMA_TYPES(MACRO) \
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

// ===== P and G Operations =====
#define UCL_SHMEM_DEVICE_DECL_TYPE_P(TYPENAME, TYPE) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmem_##TYPENAME##_p(TYPE *dest, const TYPE value, int pe);

#define UCL_SHMEM_DEVICE_DECL_TYPE_G(TYPENAME, TYPE) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE TYPE shmem_##TYPENAME##_g(const TYPE *source, int pe);

UCL_SHMEM_DEVICE_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEM_DEVICE_DECL_TYPE_P)
UCL_SHMEM_DEVICE_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEM_DEVICE_DECL_TYPE_G)

// ===== Put Operations =====
#define UCL_SHMEM_DEVICE_DECL_TYPE_PUT(TYPENAME, TYPE) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmem_##TYPENAME##_put(TYPE *dest, const TYPE *source, size_t nelems, int pe);

#define UCL_SHMEM_DEVICE_DECL_TYPE_PUT_SIGNAL(TYPENAME, TYPE) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmem_##TYPENAME##_put_signal(TYPE *dest, const TYPE *source, size_t nelems, \
                                                                                            uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) { \
        nvshmem_##TYPENAME##_put_signal(dest, source, nelems, sig_addr, signal, sig_op, pe); \
    }

#define UCL_SHMEM_DEVICE_DECL_TYPE_PUT_NBI(TYPENAME, TYPE) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmem_##TYPENAME##_put_nbi(TYPE *dest, const TYPE *source, size_t nelems, int pe);

#define UCL_SHMEM_DEVICE_DECL_TYPE_PUT_SIGNAL_NBI(TYPENAME, TYPE) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmem_##TYPENAME##_put_signal_nbi(TYPE *dest, const TYPE *source, size_t nelems, \
                                                                                                uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) { \
        nvshmem_##TYPENAME##_put_signal_nbi(dest, source, nelems, sig_addr, signal, sig_op, pe); \
    }

UCL_SHMEM_DEVICE_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEM_DEVICE_DECL_TYPE_PUT)
UCL_SHMEM_DEVICE_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEM_DEVICE_DECL_TYPE_PUT_SIGNAL)
UCL_SHMEM_DEVICE_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEM_DEVICE_DECL_TYPE_PUT_NBI)
UCL_SHMEM_DEVICE_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEM_DEVICE_DECL_TYPE_PUT_SIGNAL_NBI)

// ===== Get Operations =====
#define UCL_SHMEM_DEVICE_DECL_TYPE_GET(TYPENAME, TYPE) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmem_##TYPENAME##_get(TYPE *dest, const TYPE *source, size_t nelems, int pe);

#define UCL_SHMEM_DEVICE_DECL_TYPE_GET_NBI(TYPENAME, TYPE) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmem_##TYPENAME##_get_nbi(TYPE *dest, const TYPE *source, size_t nelems, int pe);

UCL_SHMEM_DEVICE_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEM_DEVICE_DECL_TYPE_GET)
UCL_SHMEM_DEVICE_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEM_DEVICE_DECL_TYPE_GET_NBI)

// ===== Size-based Put/Get Operations =====
#define UCL_SHMEM_DEVICE_DECL_SIZE_PUT(BITS) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmem_put##BITS(void *dest, const void *source, size_t nelems, int pe);

#define UCL_SHMEM_DEVICE_DECL_SIZE_PUT_NBI(BITS) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmem_put##BITS##_nbi(void *dest, const void *source, size_t nelems, int pe);

#define UCL_SHMEM_DEVICE_DECL_SIZE_PUT_SIGNAL(BITS) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmem_put##BITS##_signal(void *dest, const void *source, size_t nelems, \
                                                                                       uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) { \
        nvshmem_put##BITS##_signal(dest, source, nelems, sig_addr, signal, sig_op, pe); \
    }

#define UCL_SHMEM_DEVICE_DECL_SIZE_PUT_SIGNAL_NBI(BITS) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmem_put##BITS##_signal_nbi(void *dest, const void *source, size_t nelems, \
                                                                                           uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) { \
        nvshmem_put##BITS##_signal_nbi(dest, source, nelems, sig_addr, signal, sig_op, pe); \
    }

#define UCL_SHMEM_DEVICE_DECL_SIZE_GET(BITS) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmem_get##BITS(void *dest, const void *source, size_t nelems, int pe);

#define UCL_SHMEM_DEVICE_DECL_SIZE_GET_NBI(BITS) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmem_get##BITS##_nbi(void *dest, const void *source, size_t nelems, int pe);

UCL_SHMEM_DEVICE_DECL_SIZE_PUT(8)
UCL_SHMEM_DEVICE_DECL_SIZE_PUT(16)
UCL_SHMEM_DEVICE_DECL_SIZE_PUT(32)
UCL_SHMEM_DEVICE_DECL_SIZE_PUT(64)
UCL_SHMEM_DEVICE_DECL_SIZE_PUT(128)

UCL_SHMEM_DEVICE_DECL_SIZE_PUT_NBI(8)
UCL_SHMEM_DEVICE_DECL_SIZE_PUT_NBI(16)
UCL_SHMEM_DEVICE_DECL_SIZE_PUT_NBI(32)
UCL_SHMEM_DEVICE_DECL_SIZE_PUT_NBI(64)
UCL_SHMEM_DEVICE_DECL_SIZE_PUT_NBI(128)

UCL_SHMEM_DEVICE_DECL_SIZE_PUT_SIGNAL(8)
UCL_SHMEM_DEVICE_DECL_SIZE_PUT_SIGNAL(16)
UCL_SHMEM_DEVICE_DECL_SIZE_PUT_SIGNAL(32)
UCL_SHMEM_DEVICE_DECL_SIZE_PUT_SIGNAL(64)
UCL_SHMEM_DEVICE_DECL_SIZE_PUT_SIGNAL(128)

UCL_SHMEM_DEVICE_DECL_SIZE_PUT_SIGNAL_NBI(8)
UCL_SHMEM_DEVICE_DECL_SIZE_PUT_SIGNAL_NBI(16)
UCL_SHMEM_DEVICE_DECL_SIZE_PUT_SIGNAL_NBI(32)
UCL_SHMEM_DEVICE_DECL_SIZE_PUT_SIGNAL_NBI(64)
UCL_SHMEM_DEVICE_DECL_SIZE_PUT_SIGNAL_NBI(128)

UCL_SHMEM_DEVICE_DECL_SIZE_GET(8)
UCL_SHMEM_DEVICE_DECL_SIZE_GET(16)
UCL_SHMEM_DEVICE_DECL_SIZE_GET(32)
UCL_SHMEM_DEVICE_DECL_SIZE_GET(64)
UCL_SHMEM_DEVICE_DECL_SIZE_GET(128)

UCL_SHMEM_DEVICE_DECL_SIZE_GET_NBI(8)
UCL_SHMEM_DEVICE_DECL_SIZE_GET_NBI(16)
UCL_SHMEM_DEVICE_DECL_SIZE_GET_NBI(32)
UCL_SHMEM_DEVICE_DECL_SIZE_GET_NBI(64)
UCL_SHMEM_DEVICE_DECL_SIZE_GET_NBI(128)

// ===== Putmem/Getmem =====
NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmem_putmem(void *dest, const void *source, size_t bytes, int pe);

NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmem_getmem(void *dest, const void *source, size_t bytes, int pe);

NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmem_putmem_nbi(void *dest, const void *source, size_t bytes, int pe);

NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmem_getmem_nbi(void *dest, const void *source, size_t bytes, int pe);

NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmem_putmem_signal(void *dest, const void *source, size_t bytes,
                                                                               uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) {
    nvshmem_putmem_signal(dest, source, bytes, sig_addr, signal, sig_op, pe);
}

NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmem_putmem_signal_nbi(void *dest, const void *source, size_t bytes,
                                                                                   uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) {
    nvshmem_putmem_signal_nbi(dest, source, bytes, sig_addr, signal, sig_op, pe);
}

// ===== IPut/IGet Operations =====
#define UCL_SHMEM_DEVICE_DECL_TYPE_IPUT(TYPENAME, TYPE) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmem_##TYPENAME##_iput(TYPE *dest, const TYPE *source, \
                                                                                       ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe);

#define UCL_SHMEM_DEVICE_DECL_TYPE_IGET(TYPENAME, TYPE) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmem_##TYPENAME##_iget(TYPE *dest, const TYPE *source, \
                                                                                       ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe);

UCL_SHMEM_DEVICE_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEM_DEVICE_DECL_TYPE_IPUT)
UCL_SHMEM_DEVICE_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEM_DEVICE_DECL_TYPE_IGET)

#define UCL_SHMEM_DEVICE_DECL_SIZE_IPUT(BITS) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmem_iput##BITS(void *dest, const void *source, \
                                                                                ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe);

#define UCL_SHMEM_DEVICE_DECL_SIZE_IGET(BITS) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmem_iget##BITS(void *dest, const void *source, \
                                                                                ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe);

UCL_SHMEM_DEVICE_DECL_SIZE_IPUT(8)
UCL_SHMEM_DEVICE_DECL_SIZE_IPUT(16)
UCL_SHMEM_DEVICE_DECL_SIZE_IPUT(32)
UCL_SHMEM_DEVICE_DECL_SIZE_IPUT(64)
UCL_SHMEM_DEVICE_DECL_SIZE_IPUT(128)

UCL_SHMEM_DEVICE_DECL_SIZE_IGET(8)
UCL_SHMEM_DEVICE_DECL_SIZE_IGET(16)
UCL_SHMEM_DEVICE_DECL_SIZE_IGET(32)
UCL_SHMEM_DEVICE_DECL_SIZE_IGET(64)
UCL_SHMEM_DEVICE_DECL_SIZE_IGET(128)

// ===== Wait Types =====
#define UCL_SHMEM_DEVICE_REPT_FOR_WAIT_TYPES(MACRO) \
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

// ===== Test Operations =====
#define UCL_SHMEM_DEVICE_DECL_TEST(Name, Type) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE int shmem_##Name##_test(Type *ivar, int cmp, Type cmp_value) { \
        return nvshmem_##Name##_test(ivar, cmp, cmp_value); \
    }

#define UCL_SHMEM_DEVICE_DECL_TEST_ALL(Name, Type) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE int shmem_##Name##_test_all(Type *ivars, size_t nelems, const int *status, int cmp, Type cmp_value) { \
        return nvshmem_##Name##_test_all(ivars, nelems, status, cmp, cmp_value); \
    }

#define UCL_SHMEM_DEVICE_DECL_TEST_ANY(Name, Type) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE size_t shmem_##Name##_test_any(Type *ivars, size_t nelems, const int *status, int cmp, Type cmp_value) { \
        return nvshmem_##Name##_test_any(ivars, nelems, status, cmp, cmp_value); \
    }

#define UCL_SHMEM_DEVICE_DECL_TEST_SOME(Name, Type) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE size_t shmem_##Name##_test_some(Type *ivars, size_t nelems, size_t *indices, const int *status, int cmp, Type cmp_value) { \
        return nvshmem_##Name##_test_some(ivars, nelems, indices, status, cmp, cmp_value); \
    }

#define UCL_SHMEM_DEVICE_DECL_TEST_ALL_VECTOR(Name, Type) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE int shmem_##Name##_test_all_vector(Type *ivars, size_t nelems, const int *status, int cmp, Type *cmp_values) { \
        return nvshmem_##Name##_test_all_vector(ivars, nelems, status, cmp, cmp_values); \
    }

#define UCL_SHMEM_DEVICE_DECL_TEST_ANY_VECTOR(Name, Type) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE size_t shmem_##Name##_test_any_vector(Type *ivars, size_t nelems, const int *status, int cmp, Type *cmp_values) { \
        return nvshmem_##Name##_test_any_vector(ivars, nelems, status, cmp, cmp_values); \
    }

#define UCL_SHMEM_DEVICE_DECL_TEST_SOME_VECTOR(Name, Type) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE size_t shmem_##Name##_test_some_vector(Type *ivars, size_t nelems, size_t *indices, const int *status, int cmp, Type *cmp_values) { \
        return nvshmem_##Name##_test_some_vector(ivars, nelems, indices, status, cmp, cmp_values); \
    }

UCL_SHMEM_DEVICE_REPT_FOR_WAIT_TYPES(UCL_SHMEM_DEVICE_DECL_TEST)
UCL_SHMEM_DEVICE_REPT_FOR_WAIT_TYPES(UCL_SHMEM_DEVICE_DECL_TEST_ALL)
UCL_SHMEM_DEVICE_REPT_FOR_WAIT_TYPES(UCL_SHMEM_DEVICE_DECL_TEST_ANY)
UCL_SHMEM_DEVICE_REPT_FOR_WAIT_TYPES(UCL_SHMEM_DEVICE_DECL_TEST_SOME)
UCL_SHMEM_DEVICE_REPT_FOR_WAIT_TYPES(UCL_SHMEM_DEVICE_DECL_TEST_ALL_VECTOR)
UCL_SHMEM_DEVICE_REPT_FOR_WAIT_TYPES(UCL_SHMEM_DEVICE_DECL_TEST_ANY_VECTOR)
UCL_SHMEM_DEVICE_REPT_FOR_WAIT_TYPES(UCL_SHMEM_DEVICE_DECL_TEST_SOME_VECTOR)

// ===== Wait Operations =====
#define UCL_SHMEM_DEVICE_DECL_WAIT_UNTIL(Name, Type) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmem_##Name##_wait_until(Type *ivar, int cmp, Type cmp_value) { \
        nvshmem_##Name##_wait_until(ivar, cmp, cmp_value); \
    }

#define UCL_SHMEM_DEVICE_DECL_WAIT_UNTIL_ALL(Name, Type) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmem_##Name##_wait_until_all(Type *ivars, size_t nelems, const int *status, int cmp, Type cmp_value) { \
        nvshmem_##Name##_wait_until_all(ivars, nelems, status, cmp, cmp_value); \
    }

#define UCL_SHMEM_DEVICE_DECL_WAIT_UNTIL_ANY(Name, Type) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE size_t shmem_##Name##_wait_until_any(Type *ivars, size_t nelems, const int *status, int cmp, Type cmp_value) { \
        return nvshmem_##Name##_wait_until_any(ivars, nelems, status, cmp, cmp_value); \
    }

#define UCL_SHMEM_DEVICE_DECL_WAIT_UNTIL_SOME(Name, Type) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE size_t shmem_##Name##_wait_until_some(Type *ivars, size_t nelems, size_t *indices, const int *status, int cmp, Type cmp_value) { \
        return nvshmem_##Name##_wait_until_some(ivars, nelems, indices, status, cmp, cmp_value); \
    }

#define UCL_SHMEM_DEVICE_DECL_WAIT_UNTIL_ALL_VECTOR(Name, Type) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmem_##Name##_wait_until_all_vector(Type *ivars, size_t nelems, const int *status, int cmp, Type *cmp_values) { \
        nvshmem_##Name##_wait_until_all_vector(ivars, nelems, status, cmp, cmp_values); \
    }

#define UCL_SHMEM_DEVICE_DECL_WAIT_UNTIL_ANY_VECTOR(Name, Type) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE size_t shmem_##Name##_wait_until_any_vector(Type *ivars, size_t nelems, const int *status, int cmp, Type *cmp_values) { \
        return nvshmem_##Name##_wait_until_any_vector(ivars, nelems, status, cmp, cmp_values); \
    }

#define UCL_SHMEM_DEVICE_DECL_WAIT_UNTIL_SOME_VECTOR(Name, Type) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE size_t shmem_##Name##_wait_until_some_vector(Type *ivars, size_t nelems, size_t *indices, const int *status, int cmp, Type *cmp_values) { \
        return nvshmem_##Name##_wait_until_some_vector(ivars, nelems, indices, status, cmp, cmp_values); \
    }

UCL_SHMEM_DEVICE_REPT_FOR_WAIT_TYPES(UCL_SHMEM_DEVICE_DECL_WAIT_UNTIL)
UCL_SHMEM_DEVICE_REPT_FOR_WAIT_TYPES(UCL_SHMEM_DEVICE_DECL_WAIT_UNTIL_ALL)
UCL_SHMEM_DEVICE_REPT_FOR_WAIT_TYPES(UCL_SHMEM_DEVICE_DECL_WAIT_UNTIL_ANY)
UCL_SHMEM_DEVICE_REPT_FOR_WAIT_TYPES(UCL_SHMEM_DEVICE_DECL_WAIT_UNTIL_SOME)
UCL_SHMEM_DEVICE_REPT_FOR_WAIT_TYPES(UCL_SHMEM_DEVICE_DECL_WAIT_UNTIL_ALL_VECTOR)
UCL_SHMEM_DEVICE_REPT_FOR_WAIT_TYPES(UCL_SHMEM_DEVICE_DECL_WAIT_UNTIL_ANY_VECTOR)
UCL_SHMEM_DEVICE_REPT_FOR_WAIT_TYPES(UCL_SHMEM_DEVICE_DECL_WAIT_UNTIL_SOME_VECTOR)

// ===== Signal Operations =====
NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE uint64_t shmem_signal_fetch(uint64_t *sig_addr);

NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE uint64_t shmem_signal_wait_until(uint64_t *sig_addr, int cmp, uint64_t cmp_value);

// ===== Synchronization =====
NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmem_quiet();

NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmem_fence();

// ===== Memory Pointer =====
NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void *shmem_ptr(const void *dest, int pe);

#endif // __CUDA_ARCH__

} // namespace shmem
} // namespace ucl

#endif // UCL_SHMEM_DEVICE_DEFINES_CUH
