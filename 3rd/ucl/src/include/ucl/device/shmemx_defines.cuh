/*
 * UCL SHMEM API - Device Extended Defines
 * This file enhances NVSHMEM device extended API functions into UCL shmem namespace
 */

#ifndef UCL_SHMEMX_DEVICE_DEFINES_CUH
#define UCL_SHMEMX_DEVICE_DEFINES_CUH

#include "device/nvshmemx_defines.h"

namespace ucl {
namespace shmem {

#ifdef __CUDA_ARCH__

// ===== Version Info =====
NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmemx_vendor_get_version_info(int *major, int *minor, int *patch);

// ===== Signal Operations =====
NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmemx_signal_op(uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
// ===== MC Pointer =====
NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void *shmemx_mc_ptr(nvshmem_team_t team, const void *ptr);

// ===== Standard RMA Types =====
#define UCL_SHMEMX_DEVICE_REPT_FOR_STANDARD_RMA_TYPES(MACRO) \
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

// ===== Put Threadgroup Operations =====
#define UCL_SHMEMX_DEVICE_DECL_TYPE_PUT_THREADGROUP(Name, Type) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmemx_##Name##_put_warp(Type *dest, const Type *source, size_t nelems, int pe) { \
        nvshmemx_##Name##_put_warp(dest, source, nelems, pe); \
    } \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmemx_##Name##_put_block(Type *dest, const Type *source, size_t nelems, int pe) { \
        nvshmemx_##Name##_put_block(dest, source, nelems, pe); \
    }

UCL_SHMEMX_DEVICE_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEMX_DEVICE_DECL_TYPE_PUT_THREADGROUP)

// ===== Get Threadgroup Operations =====
#define UCL_SHMEMX_DEVICE_DECL_TYPE_GET_THREADGROUP(Name, Type) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmemx_##Name##_get_warp(Type *dest, const Type *source, size_t nelems, int pe) { \
        nvshmemx_##Name##_get_warp(dest, source, nelems, pe); \
    } \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmemx_##Name##_get_block(Type *dest, const Type *source, size_t nelems, int pe) { \
        nvshmemx_##Name##_get_block(dest, source, nelems, pe); \
    }

UCL_SHMEMX_DEVICE_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEMX_DEVICE_DECL_TYPE_GET_THREADGROUP)

// ===== Put Signal Threadgroup Operations =====
#define UCL_SHMEMX_DEVICE_DECL_TYPE_PUT_SIGNAL_THREADGROUP(Name, Type) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmemx_##Name##_put_signal_warp(Type *dest, const Type *source, size_t nelems, \
                                                                                              uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) { \
        nvshmemx_##Name##_put_signal_warp(dest, source, nelems, sig_addr, signal, sig_op, pe); \
    } \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmemx_##Name##_put_signal_block(Type *dest, const Type *source, size_t nelems, \
                                                                                               uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) { \
        nvshmemx_##Name##_put_signal_block(dest, source, nelems, sig_addr, signal, sig_op, pe); \
    }

UCL_SHMEMX_DEVICE_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEMX_DEVICE_DECL_TYPE_PUT_SIGNAL_THREADGROUP)

// ===== Put Signal NBI Threadgroup Operations =====
#define UCL_SHMEMX_DEVICE_DECL_TYPE_PUT_SIGNAL_NBI_THREADGROUP(Name, Type) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmemx_##Name##_put_signal_nbi_warp(Type *dest, const Type *source, size_t nelems, \
                                                                                                  uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) { \
        nvshmemx_##Name##_put_signal_nbi_warp(dest, source, nelems, sig_addr, signal, sig_op, pe); \
    } \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmemx_##Name##_put_signal_nbi_block(Type *dest, const Type *source, size_t nelems, \
                                                                                                   uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) { \
        nvshmemx_##Name##_put_signal_nbi_block(dest, source, nelems, sig_addr, signal, sig_op, pe); \
    }

UCL_SHMEMX_DEVICE_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEMX_DEVICE_DECL_TYPE_PUT_SIGNAL_NBI_THREADGROUP)

// ===== Size-based Threadgroup Operations =====
#define UCL_SHMEMX_DEVICE_DECL_SIZE_PUT_THREADGROUP(BITS) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmemx_put##BITS##_warp(void *dest, const void *source, size_t nelems, int pe) { \
        nvshmemx_put##BITS##_warp(dest, source, nelems, pe); \
    } \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmemx_put##BITS##_block(void *dest, const void *source, size_t nelems, int pe) { \
        nvshmemx_put##BITS##_block(dest, source, nelems, pe); \
    }

#define UCL_SHMEMX_DEVICE_DECL_SIZE_GET_THREADGROUP(BITS) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmemx_get##BITS##_warp(void *dest, const void *source, size_t nelems, int pe) { \
        nvshmemx_get##BITS##_warp(dest, source, nelems, pe); \
    } \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmemx_get##BITS##_block(void *dest, const void *source, size_t nelems, int pe) { \
        nvshmemx_get##BITS##_block(dest, source, nelems, pe); \
    }

UCL_SHMEMX_DEVICE_DECL_SIZE_PUT_THREADGROUP(8)
UCL_SHMEMX_DEVICE_DECL_SIZE_PUT_THREADGROUP(16)
UCL_SHMEMX_DEVICE_DECL_SIZE_PUT_THREADGROUP(32)
UCL_SHMEMX_DEVICE_DECL_SIZE_PUT_THREADGROUP(64)
UCL_SHMEMX_DEVICE_DECL_SIZE_PUT_THREADGROUP(128)

UCL_SHMEMX_DEVICE_DECL_SIZE_GET_THREADGROUP(8)
UCL_SHMEMX_DEVICE_DECL_SIZE_GET_THREADGROUP(16)
UCL_SHMEMX_DEVICE_DECL_SIZE_GET_THREADGROUP(32)
UCL_SHMEMX_DEVICE_DECL_SIZE_GET_THREADGROUP(64)
UCL_SHMEMX_DEVICE_DECL_SIZE_GET_THREADGROUP(128)

// ===== Putmem/Getmem Threadgroup =====
NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmemx_putmem_warp(void *dest, const void *source, size_t bytes, int pe) {
    nvshmemx_putmem_warp(dest, source, bytes, pe);
}

NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmemx_putmem_block(void *dest, const void *source, size_t bytes, int pe) {
    nvshmemx_putmem_block(dest, source, bytes, pe);
}

NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmemx_getmem_warp(void *dest, const void *source, size_t bytes, int pe) {
    nvshmemx_getmem_warp(dest, source, bytes, pe);
}

NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmemx_getmem_block(void *dest, const void *source, size_t bytes, int pe) {
    nvshmemx_getmem_block(dest, source, bytes, pe);
}

// ===== Putmem/Getmem Signal Threadgroup =====
NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmemx_putmem_signal_warp(void *dest, const void *source, size_t nelems,
                                                                                     uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) {
    nvshmemx_putmem_signal_warp(dest, source, nelems, sig_addr, signal, sig_op, pe);
}

NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmemx_putmem_signal_block(void *dest, const void *source, size_t nelems,
                                                                                      uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) {
    nvshmemx_putmem_signal_block(dest, source, nelems, sig_addr, signal, sig_op, pe);
}

NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmemx_putmem_signal_nbi_warp(void *dest, const void *source, size_t nelems,
                                                                                         uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) {
    nvshmemx_putmem_signal_nbi_warp(dest, source, nelems, sig_addr, signal, sig_op, pe);
}

NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmemx_putmem_signal_nbi_block(void *dest, const void *source, size_t nelems,
                                                                                          uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) {
    nvshmemx_putmem_signal_nbi_block(dest, source, nelems, sig_addr, signal, sig_op, pe);
}

// ===== Size-based Signal Threadgroup =====
#define UCL_SHMEMX_DEVICE_DECL_SIZE_PUT_SIGNAL_THREADGROUP(BITS) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmemx_put##BITS##_signal_warp(void *dest, const void *source, size_t nelems, \
                                                                                             uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) { \
        nvshmemx_put##BITS##_signal_warp(dest, source, nelems, sig_addr, signal, sig_op, pe); \
    } \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmemx_put##BITS##_signal_block(void *dest, const void *source, size_t nelems, \
                                                                                              uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) { \
        nvshmemx_put##BITS##_signal_block(dest, source, nelems, sig_addr, signal, sig_op, pe); \
    }

#define UCL_SHMEMX_DEVICE_DECL_SIZE_PUT_SIGNAL_NBI_THREADGROUP(BITS) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmemx_put##BITS##_signal_nbi_warp(void *dest, const void *source, size_t nelems, \
                                                                                                 uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) { \
        nvshmemx_put##BITS##_signal_nbi_warp(dest, source, nelems, sig_addr, signal, sig_op, pe); \
    } \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmemx_put##BITS##_signal_nbi_block(void *dest, const void *source, size_t nelems, \
                                                                                                  uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) { \
        nvshmemx_put##BITS##_signal_nbi_block(dest, source, nelems, sig_addr, signal, sig_op, pe); \
    }

UCL_SHMEMX_DEVICE_DECL_SIZE_PUT_SIGNAL_THREADGROUP(8)
UCL_SHMEMX_DEVICE_DECL_SIZE_PUT_SIGNAL_THREADGROUP(6)
UCL_SHMEMX_DEVICE_DECL_SIZE_PUT_SIGNAL_THREADGROUP(32)
UCL_SHMEMX_DEVICE_DECL_SIZE_PUT_SIGNAL_THREADGROUP(64)
UCL_SHMEMX_DEVICE_DECL_SIZE_PUT_SIGNAL_THREADGROUP(128)

UCL_SHMEMX_DEVICE_DECL_SIZE_PUT_SIGNAL_NBI_THREADGROUP(8)
UCL_SHMEMX_DEVICE_DECL_SIZE_PUT_SIGNAL_NBI_THREADGROUP(6)
UCL_SHMEMX_DEVICE_DECL_SIZE_PUT_SIGNAL_NBI_THREADGROUP(32)
UCL_SHMEMX_DEVICE_DECL_SIZE_PUT_SIGNAL_NBI_THREADGROUP(64)
UCL_SHMEMX_DEVICE_DECL_SIZE_PUT_SIGNAL_NBI_THREADGROUP(128)

// ===== Put NBI Threadgroup =====
#define UCL_SHMEMX_DEVICE_DECL_TYPE_PUT_NBI_THREADGROUP(Name, Type) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmemx_##Name##_put_nbi_warp(Type *dest, const Type *source, size_t nelems, int pe) { \
        nvshmemx_##Name##_put_nbi_warp(dest, source, nelems, pe); \
    } \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmemx_##Name##_put_nbi_block(Type *dest, const Type *source, size_t nelems, int pe) { \
        nvshmemx_##Name##_put_nbi_block(dest, source, nelems, pe); \
    }

UCL_SHMEMX_DEVICE_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEMX_DEVICE_DECL_TYPE_PUT_NBI_THREADGROUP)

// ===== Get NBI Threadgroup =====
#define UCL_SHMEMX_DEVICE_DECL_TYPE_GET_NBI_THREADGROUP(Name, Type) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmemx_##Name##_get_nbi_warp(Type *dest, const Type *source, size_t nelems, int pe) { \
        nvshmemx_##Name##_get_nbi_warp(dest, source, nelems, pe); \
    } \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmemx_##Name##_get_nbi_block(Type *dest, const Type *source, size_t nelems, int pe) { \
        nvshmemx_##Name##_get_nbi_block(dest, source, nelems, pe); \
    }

UCL_SHMEMX_DEVICE_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEMX_DEVICE_DECL_TYPE_GET_NBI_THREADGROUP)

// ===== Size-based NBI Threadgroup =====
#define UCL_SHMEMX_DEVICE_DECL_SIZE_PUT_NBI_THREADGROUP(BITS) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmemx_put##BITS##_nbi_warp(void *dest, const void *source, size_t nelems, int pe) { \
        nvshmemx_put##BITS##_nbi_warp(dest, source, nelems, pe); \
    } \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmemx_put##BITS##_nbi_block(void *dest, const void *source, size_t nelems, int pe) { \
        nvshmemx_put##BITS##_nbi_block(dest, source, nelems, pe); \
    }

#define UCL_SHMEMX_DEVICE_DECL_SIZE_GET_NBI_THREADGROUP(BITS) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmemx_get##BITS##_nbi_warp(void *dest, const void *source, size_t nelems, int pe) { \
        nvshmemx_get##BITS##_nbi_warp(dest, source, nelems, pe); \
    } \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmemx_get##BITS##_nbi_block(void *dest, const void *source, size_t nelems, int pe) { \
        nvshmemx_get##BITS##_nbi_block(dest, source, nelems, pe); \
    }

UCL_SHMEMX_DEVICE_DECL_SIZE_PUT_NBI_THREADGROUP(8)
UCL_SHMEMX_DEVICE_DECL_SIZE_PUT_NBI_THREADGROUP(16)
UCL_SHMEMX_DEVICE_DECL_SIZE_PUT_NBI_THREADGROUP(32)
UCL_SHMEMX_DEVICE_DECL_SIZE_PUT_NBI_THREADGROUP(64)
UCL_SHMEMX_DEVICE_DECL_SIZE_PUT_NBI_THREADGROUP(128)

UCL_SHMEMX_DEVICE_DECL_SIZE_GET_NBI_THREADGROUP(8)
UCL_SHMEMX_DEVICE_DECL_SIZE_GET_NBI_THREADGROUP(16)
UCL_SHMEMX_DEVICE_DECL_SIZE_GET_NBI_THREADGROUP(32)
UCL_SHMEMX_DEVICE_DECL_SIZE_GET_NBI_THREADGROUP(64)
UCL_SHMEMX_DEVICE_DECL_SIZE_GET_NBI_THREADGROUP(128)

NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmemx_putmem_nbi_warp(void *dest, const void *source, size_t bytes, int pe) {
    nvshmemx_putmem_nbi_warp(dest, source, bytes, pe);
}

NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmemx_putmem_nbi_block(void *dest, const void *source, size_t bytes, int pe) {
    nvshmemx_putmem_nbi_block(dest, source, bytes, pe);
}

NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmemx_getmem_nbi_warp(void *dest, const void *source, size_t bytes, int pe) {
    nvshmemx_getmem_nbi_warp(dest, source, bytes, pe);
}

NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmemx_getmem_nbi_block(void *dest, const void *source, size_t bytes, int pe) {
    nvshmemx_getmem_nbi_block(dest, source, bytes, pe);
}

// ===== IPut Threadgroup =====
#define UCL_SHMEMX_DEVICE_DECL_TYPE_IPUT_THREADGROUP(Name, Type) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmemx_##Name##_iput_warp(Type *dest, const Type *source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe) { \
        nvshmemx_##Name##_iput_warp(dest, source, dst, sst, nelems, pe); \
    } \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmemx_##Name##_iput_block(Type *dest, const Type *source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe) { \
        nvshmemx_##Name##_iput_block(dest, source, dst, sst, nelems, pe); \
    }

UCL_SHMEMX_DEVICE_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEMX_DEVICE_DECL_TYPE_IPUT_THREADGROUP)

// ===== IGet Threadgroup =====
#define UCL_SHMEMX_DEVICE_DECL_TYPE_IGET_THREADGROUP(Name, Type) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmemx_##Name##_iget_warp(Type *dest, const Type *source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe) { \
        nvshmemx_##Name##_iget_warp(dest, source, dst, sst, nelems, pe); \
    } \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmemx_##Name##_iget_block(Type *dest, const Type *source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe) { \
        nvshmemx_##Name##_iget_block(dest, source, dst, sst, nelems, pe); \
    }

UCL_SHMEMX_DEVICE_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEMX_DEVICE_DECL_TYPE_IGET_THREADGROUP)

// ===== Size-based IPut/IGet Threadgroup =====
#define UCL_SHMEMX_DEVICE_DECL_SIZE_IPUT_THREADGROUP(BITS) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmemx_iput##BITS##_warp(void *dest, const void *source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe) { \
        nvshmemx_iput##BITS##_warp(dest, source, dst, sst, nelems, pe); \
    } \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmemx_iput##BITS##_block(void *dest, const void *source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe) { \
        nvshmemx_iput##BITS##_block(dest, source, dst, sst, nelems, pe); \
    }

#define UCL_SHMEMX_DEVICE_DECL_SIZE_IGET_THREADGROUP(BITS) \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmemx_iget##BITS##_warp(void *dest, const void *source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe) { \
        nvshmemx_iget##BITS##_warp(dest, source, dst, sst, nelems, pe); \
    } \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void shmemx_iget##BITS##_block(void *dest, const void *source, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe) { \
        nvshmemx_iget##BITS##_block(dest, source, dst, sst, nelems, pe); \
    }

UCL_SHMEMX_DEVICE_DECL_SIZE_IPUT_THREADGROUP(8)
UCL_SHMEMX_DEVICE_DECL_SIZE_IPUT_THREADGROUP(16)
UCL_SHMEMX_DEVICE_DECL_SIZE_IPUT_THREADGROUP(32)
UCL_SHMEMX_DEVICE_DECL_SIZE_IPUT_THREADGROUP(64)
UCL_SHMEMX_DEVICE_DECL_SIZE_IPUT_THREADGROUP(128)

UCL_SHMEMX_DEVICE_DECL_SIZE_IGET_THREADGROUP(8)
UCL_SHMEMX_DEVICE_DECL_SIZE_IGET_THREADGROUP(16)
UCL_SHMEMX_DEVICE_DECL_SIZE_IGET_THREADGROUP(32)
UCL_SHMEMX_DEVICE_DECL_SIZE_IGET_THREADGROUP(64)
UCL_SHMEMX_DEVICE_DECL_SIZE_IGET_THREADGROUP(128)

#endif // __CUDA_ARCH__

} // namespace shmem
} // namespace ucl

#endif // UCL_SHMEMX_DEVICE_DEFINES_CUH
