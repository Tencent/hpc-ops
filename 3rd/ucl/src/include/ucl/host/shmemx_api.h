/*
 * UCL SHMEM API - Host Extended API
 * This file enhances NVSHMEM host extended API functions into UCL shmem namespace
 */

#ifndef UCL_SHMEMX_HOST_API_H
#define UCL_SHMEMX_HOST_API_H

#include "host/nvshmemx_api.h"

namespace ucl {
namespace shmem {

// ===== Init Flags Enum =====
enum flags {
    SHMEMX_INIT_THREAD_PES = NVSHMEMX_INIT_THREAD_PES,
    SHMEMX_INIT_WITH_MPI_COMM = NVSHMEMX_INIT_WITH_MPI_COMM,
    SHMEMX_INIT_WITH_SHMEM = NVSHMEMX_INIT_WITH_SHMEM,
    SHMEMX_INIT_WITH_UNIQUEID = NVSHMEMX_INIT_WITH_UNIQUEID,
    SHMEMX_INIT_MAX = NVSHMEMX_INIT_MAX
}; 

// ===== Heap Management Extensions =====
static inline int shmemx_buffer_register(void *addr, size_t length) {
    return nvshmemx_buffer_register(addr, length);
}
static inline int shmemx_buffer_unregister(void *addr) {
    return nvshmemx_buffer_unregister(addr);
}
static inline void shmemx_buffer_unregister_all() {
    nvshmemx_buffer_unregister_all();
}

// ===== Initialization & Finalization Extensions =====
static inline int shmemx_hostlib_init_attr(unsigned int flags, nvshmemx_init_attr_t *attr) {
    return nvshmemx_hostlib_init_attr(flags, attr);
}
static inline void shmemx_hostlib_finalize() {
    nvshmemx_hostlib_finalize();
}
static inline int shmemx_init_attr(unsigned int flags, nvshmemx_init_attr_t *attributes) {
    return nvshmemx_init_attr(flags, attributes);
}
static inline int shmemx_set_attr_uniqueid_args(const int myrank, const int nranks,
                                                 const nvshmemx_uniqueid_t *uniqueid,
                                                 nvshmemx_init_attr_t *attr) {
    return nvshmemx_set_attr_uniqueid_args(myrank, nranks, uniqueid, attr);
}
static inline int shmemx_set_attr_mpi_comm_args(void *mpi_comm, nvshmemx_init_attr_t *nvshmem_attr) {
    return nvshmemx_set_attr_mpi_comm_args(mpi_comm, nvshmem_attr);
}
static inline int shmemx_get_uniqueid(nvshmemx_uniqueid_t *uniqueid) {
    return nvshmemx_get_uniqueid(uniqueid);
}
static inline int shmemx_cumodule_init(CUmodule module) {
    return nvshmemx_cumodule_init(module);
}
static inline int shmemx_cumodule_finalize(CUmodule module) {
    return nvshmemx_cumodule_finalize(module);
}
static inline void *shmemx_buffer_register_symmetric(void *buf_ptr, size_t size, int flags) {
    return nvshmemx_buffer_register_symmetric(buf_ptr, size, flags);
}
static inline int shmemx_buffer_unregister_symmetric(void *mmap_ptr, size_t size) {
    return nvshmemx_buffer_unregister_symmetric(mmap_ptr, size);
}
static inline int shmemx_culibrary_init(CUlibrary library) {
    return nvshmemx_culibrary_init(library);
}
static inline int shmemx_culibrary_finalize(CUlibrary library) {
    return nvshmemx_culibrary_finalize(library);
}

// ===== Put On Stream =====
#define UCL_SHMEMX_DECL_TYPE_P_ON_STREAM(NAME, TYPE) \
    static inline void shmemx_##NAME##_p_on_stream(TYPE *dest, const TYPE value, int pe, cudaStream_t cstrm) { \
        nvshmemx_##NAME##_p_on_stream(dest, value, pe, cstrm); \
    }

#define UCL_SHMEMX_DECL_TYPE_PUT_ON_STREAM(NAME, TYPE) \
    static inline void shmemx_##NAME##_put_on_stream(TYPE *dest, const TYPE *source, size_t nelems, int pe, cudaStream_t cstrm) { \
        nvshmemx_##NAME##_put_on_stream(dest, source, nelems, pe, cstrm); \
    }

#define UCL_SHMEMX_DECL_TYPE_PUT_SIGNAL_ON_STREAM(NAME, TYPE) \
    static inline void shmemx_##NAME##_put_signal_on_stream(TYPE *dest, const TYPE *source, size_t nelems, \
                                                            uint64_t *sig_addr, uint64_t signal, int sig_op, \
                                                            int pe, cudaStream_t cstrm) { \
        nvshmemx_##NAME##_put_signal_on_stream(dest, source, nelems, sig_addr, signal, sig_op, pe, cstrm); \
    }

#define UCL_SHMEMX_DECL_TYPE_PUT_SIGNAL_NBI_ON_STREAM(NAME, TYPE) \
    static inline void shmemx_##NAME##_put_signal_nbi_on_stream(TYPE *dest, const TYPE *source, size_t nelems, \
                                                                 uint64_t *sig_addr, uint64_t signal, int sig_op, \
                                                                 int pe, cudaStream_t cstrm) { \
        nvshmemx_##NAME##_put_signal_nbi_on_stream(dest, source, nelems, sig_addr, signal, sig_op, pe, cstrm); \
    }

#define UCL_SHMEMX_DECL_TYPE_IPUT_ON_STREAM(NAME, TYPE) \
    static inline void shmemx_##NAME##_iput_on_stream(TYPE *dest, const TYPE *source, ptrdiff_t dst, \
                                                       ptrdiff_t sst, size_t nelems, int pe, cudaStream_t cstrm) { \
        nvshmemx_##NAME##_iput_on_stream(dest, source, dst, sst, nelems, pe, cstrm); \
    }

#define UCL_SHMEMX_DECL_TYPE_PUT_NBI_ON_STREAM(NAME, TYPE) \
    static inline void shmemx_##NAME##_put_nbi_on_stream(TYPE *dest, const TYPE *source, size_t nelems, \
                                                          int pe, cudaStream_t cstrm) { \
        nvshmemx_##NAME##_put_nbi_on_stream(dest, source, nelems, pe, cstrm); \
    }

// Standard RMA types
#define UCL_SHMEMX_REPT_FOR_STANDARD_RMA_TYPES(MACRO) \
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

UCL_SHMEMX_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEMX_DECL_TYPE_P_ON_STREAM)
UCL_SHMEMX_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEMX_DECL_TYPE_PUT_ON_STREAM)
UCL_SHMEMX_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEMX_DECL_TYPE_PUT_SIGNAL_ON_STREAM)
UCL_SHMEMX_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEMX_DECL_TYPE_PUT_SIGNAL_NBI_ON_STREAM)
UCL_SHMEMX_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEMX_DECL_TYPE_IPUT_ON_STREAM)
UCL_SHMEMX_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEMX_DECL_TYPE_PUT_NBI_ON_STREAM)

// Size-based put on stream
#define UCL_SHMEMX_DECL_SIZE_PUT_ON_STREAM(NAME) \
    static inline void shmemx_put##NAME##_on_stream(void *dest, const void *source, size_t nelems, int pe, cudaStream_t cstrm) { \
        nvshmemx_put##NAME##_on_stream(dest, source, nelems, pe, cstrm); \
    }

#define UCL_SHMEMX_DECL_SIZE_PUT_SIGNAL_ON_STREAM(NAME) \
    static inline void shmemx_put##NAME##_signal_on_stream(void *dest, const void *source, size_t nelems, \
                                                           uint64_t *sig_addr, uint64_t signal, int sig_op, \
                                                           int pe, cudaStream_t cstrm) { \
        nvshmemx_put##NAME##_signal_on_stream(dest, source, nelems, sig_addr, signal, sig_op, pe, cstrm); \
    }

#define UCL_SHMEMX_DECL_SIZE_PUT_SIGNAL_NBI_ON_STREAM(NAME) \
    static inline void shmemx_put##NAME##_signal_nbi_on_stream(void *dest, const void *source, size_t nelems, \
                                                                uint64_t *sig_addr, uint64_t signal, int sig_op, \
                                                                int pe, cudaStream_t cstrm) { \
        nvshmemx_put##NAME##_signal_nbi_on_stream(dest, source, nelems, sig_addr, signal, sig_op, pe, cstrm); \
    }

#define UCL_SHMEMX_DECL_SIZE_IPUT_ON_STREAM(NAME) \
    static inline void shmemx_iput##NAME##_on_stream(void *dest, const void *source, ptrdiff_t dst, \
                                                      ptrdiff_t sst, size_t nelems, int pe, cudaStream_t cstrm) { \
        nvshmemx_iput##NAME##_on_stream(dest, source, dst, sst, nelems, pe, cstrm); \
    }

#define UCL_SHMEMX_DECL_SIZE_PUT_NBI_ON_STREAM(NAME) \
    static inline void shmemx_put##NAME##_nbi_on_stream(void *dest, const void *source, size_t nelems, int pe, cudaStream_t cstrm) { \
        nvshmemx_put##NAME##_nbi_on_stream(dest, source, nelems, pe, cstrm); \
    }

UCL_SHMEMX_DECL_SIZE_PUT_ON_STREAM(8)
UCL_SHMEMX_DECL_SIZE_PUT_ON_STREAM(16)
UCL_SHMEMX_DECL_SIZE_PUT_ON_STREAM(32)
UCL_SHMEMX_DECL_SIZE_PUT_ON_STREAM(64)
UCL_SHMEMX_DECL_SIZE_PUT_ON_STREAM(128)

UCL_SHMEMX_DECL_SIZE_PUT_SIGNAL_ON_STREAM(8)
UCL_SHMEMX_DECL_SIZE_PUT_SIGNAL_ON_STREAM(16)
UCL_SHMEMX_DECL_SIZE_PUT_SIGNAL_ON_STREAM(32)
UCL_SHMEMX_DECL_SIZE_PUT_SIGNAL_ON_STREAM(64)
UCL_SHMEMX_DECL_SIZE_PUT_SIGNAL_ON_STREAM(128)

UCL_SHMEMX_DECL_SIZE_PUT_SIGNAL_NBI_ON_STREAM(8)
UCL_SHMEMX_DECL_SIZE_PUT_SIGNAL_NBI_ON_STREAM(16)
UCL_SHMEMX_DECL_SIZE_PUT_SIGNAL_NBI_ON_STREAM(32)
UCL_SHMEMX_DECL_SIZE_PUT_SIGNAL_NBI_ON_STREAM(64)
UCL_SHMEMX_DECL_SIZE_PUT_SIGNAL_NBI_ON_STREAM(128)

UCL_SHMEMX_DECL_SIZE_IPUT_ON_STREAM(8)
UCL_SHMEMX_DECL_SIZE_IPUT_ON_STREAM(16)
UCL_SHMEMX_DECL_SIZE_IPUT_ON_STREAM(32)
UCL_SHMEMX_DECL_SIZE_IPUT_ON_STREAM(64)
UCL_SHMEMX_DECL_SIZE_IPUT_ON_STREAM(128)

UCL_SHMEMX_DECL_SIZE_PUT_NBI_ON_STREAM(8)
UCL_SHMEMX_DECL_SIZE_PUT_NBI_ON_STREAM(16)
UCL_SHMEMX_DECL_SIZE_PUT_NBI_ON_STREAM(32)
UCL_SHMEMX_DECL_SIZE_PUT_NBI_ON_STREAM(64)
UCL_SHMEMX_DECL_SIZE_PUT_NBI_ON_STREAM(128)

static inline void shmemx_putmem_on_stream(void *dest, const void *source, size_t bytes, int pe, cudaStream_t cstrm) {
    nvshmemx_putmem_on_stream(dest, source, bytes, pe, cstrm);
}

static inline void shmemx_putmem_signal_on_stream(void *dest, const void *source, size_t bytes,
                                                   uint64_t *sig_addr, uint64_t signal, int sig_op,
                                                   int pe, cudaStream_t cstrm) {
    nvshmemx_putmem_signal_on_stream(dest, source, bytes, sig_addr, signal, sig_op, pe, cstrm);
}

static inline void shmemx_putmem_signal_nbi_on_stream(void *dest, const void *source, size_t bytes,
                                                       uint64_t *sig_addr, uint64_t signal, int sig_op,
                                                       int pe, cudaStream_t cstrm) {
    nvshmemx_putmem_signal_nbi_on_stream(dest, source, bytes, sig_addr, signal, sig_op, pe, cstrm);
}

static inline void shmemx_putmem_nbi_on_stream(void *dest, const void *source, size_t bytes, int pe, cudaStream_t cstrm) {
    nvshmemx_putmem_nbi_on_stream(dest, source, bytes, pe, cstrm);
}

// ===== Get On Stream =====
#define UCL_SHMEMX_DECL_TYPE_G_ON_STREAM(NAME, TYPE) \
    static inline TYPE shmemx_##NAME##_g_on_stream(const TYPE *src, int pe, cudaStream_t cstrm) { \
        return nvshmemx_##NAME##_g_on_stream(src, pe, cstrm); \
    }

#define UCL_SHMEMX_DECL_TYPE_GET_ON_STREAM(NAME, TYPE) \
    static inline void shmemx_##NAME##_get_on_stream(TYPE *dest, const TYPE *source, size_t nelems, int pe, cudaStream_t cstrm) { \
        nvshmemx_##NAME##_get_on_stream(dest, source, nelems, pe, cstrm); \
    }

#define UCL_SHMEMX_DECL_TYPE_IGET_ON_STREAM(NAME, TYPE) \
    static inline void shmemx_##NAME##_iget_on_stream(TYPE *dest, const TYPE *source, ptrdiff_t dst, \
                                                       ptrdiff_t sst, size_t nelems, int pe, cudaStream_t cstrm) { \
        nvshmemx_##NAME##_iget_on_stream(dest, source, dst, sst, nelems, pe, cstrm); \
    }

#define UCL_SHMEMX_DECL_TYPE_GET_NBI_ON_STREAM(NAME, TYPE) \
    static inline void shmemx_##NAME##_get_nbi_on_stream(TYPE *dest, const TYPE *source, size_t nelems, \
                                                          int pe, cudaStream_t cstrm) { \
        nvshmemx_##NAME##_get_nbi_on_stream(dest, source, nelems, pe, cstrm); \
    }

UCL_SHMEMX_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEMX_DECL_TYPE_G_ON_STREAM)
UCL_SHMEMX_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEMX_DECL_TYPE_GET_ON_STREAM)
UCL_SHMEMX_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEMX_DECL_TYPE_IGET_ON_STREAM)
UCL_SHMEMX_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEMX_DECL_TYPE_GET_NBI_ON_STREAM)

// Size-based get on stream
#define UCL_SHMEMX_DECL_SIZE_GET_ON_STREAM(NAME) \
    static inline void shmemx_get##NAME##_on_stream(void *dest, const void *source, size_t nelems, int pe, cudaStream_t cstrm) { \
        nvshmemx_get##NAME##_on_stream(dest, source, nelems, pe, cstrm); \
    }

#define UCL_SHMEMX_DECL_SIZE_IGET_ON_STREAM(NAME) \
    static inline void shmemx_iget##NAME##_on_stream(void *dest, const void *source, ptrdiff_t dst, \
                                                      ptrdiff_t sst, size_t nelems, int pe, cudaStream_t cstrm) { \
        nvshmemx_iget##NAME##_on_stream(dest, source, dst, sst, nelems, pe, cstrm); \
    }

#define UCL_SHMEMX_DECL_SIZE_GET_NBI_ON_STREAM(NAME) \
    static inline void shmemx_get##NAME##_nbi_on_stream(void *dest, const void *source, size_t nelems, int pe, cudaStream_t cstrm) { \
        nvshmemx_get##NAME##_nbi_on_stream(dest, source, nelems, pe, cstrm); \
    }

UCL_SHMEMX_DECL_SIZE_GET_ON_STREAM(8)
UCL_SHMEMX_DECL_SIZE_GET_ON_STREAM(16)
UCL_SHMEMX_DECL_SIZE_GET_ON_STREAM(32)
UCL_SHMEMX_DECL_SIZE_GET_ON_STREAM(64)
UCL_SHMEMX_DECL_SIZE_GET_ON_STREAM(128)

UCL_SHMEMX_DECL_SIZE_IGET_ON_STREAM(8)
UCL_SHMEMX_DECL_SIZE_IGET_ON_STREAM(16)
UCL_SHMEMX_DECL_SIZE_IGET_ON_STREAM(32)
UCL_SHMEMX_DECL_SIZE_IGET_ON_STREAM(64)
UCL_SHMEMX_DECL_SIZE_IGET_ON_STREAM(128)

UCL_SHMEMX_DECL_SIZE_GET_NBI_ON_STREAM(8)
UCL_SHMEMX_DECL_SIZE_GET_NBI_ON_STREAM(16)
UCL_SHMEMX_DECL_SIZE_GET_NBI_ON_STREAM(32)
UCL_SHMEMX_DECL_SIZE_GET_NBI_ON_STREAM(64)
UCL_SHMEMX_DECL_SIZE_GET_NBI_ON_STREAM(128)

static inline void shmemx_getmem_on_stream(void *dest, const void *source, size_t bytes, int pe, cudaStream_t cstrm) {
    nvshmemx_getmem_on_stream(dest, source, bytes, pe, cstrm);
}

static inline void shmemx_getmem_nbi_on_stream(void *dest, const void *source, size_t bytes, int pe, cudaStream_t cstrm) {
    nvshmemx_getmem_nbi_on_stream(dest, source, bytes, pe, cstrm);
}

// ===== Synchronization On Stream =====
static inline void shmemx_quiet_on_stream(cudaStream_t cstrm) {
    nvshmemx_quiet_on_stream(cstrm);
}

static inline void shmemx_signal_op_on_stream(uint64_t *sig_addr, uint64_t signal, int sig_op, int pe, cudaStream_t cstrm) {
    nvshmemx_signal_op_on_stream(sig_addr, signal, sig_op, pe, cstrm);
}

#define UCL_SHMEMX_DECL_WAIT_UNTIL_ON_STREAM(NAME, TYPE) \
    static inline void shmemx_##NAME##_wait_until_on_stream(TYPE *ivar, int cmp, TYPE cmp_value, cudaStream_t cstream) { \
        nvshmemx_##NAME##_wait_until_on_stream(ivar, cmp, cmp_value, cstream); \
    }

#define UCL_SHMEMX_DECL_WAIT_UNTIL_ALL_ON_STREAM(NAME, TYPE) \
    static inline void shmemx_##NAME##_wait_until_all_on_stream(TYPE *ivars, size_t nelems, const int *status, \
                                                                 int cmp, TYPE cmp_value, cudaStream_t cstream) { \
        nvshmemx_##NAME##_wait_until_all_on_stream(ivars, nelems, status, cmp, cmp_value, cstream); \
    }

#define UCL_SHMEMX_DECL_WAIT_UNTIL_ALL_VECTOR_ON_STREAM(NAME, TYPE) \
    static inline void shmemx_##NAME##_wait_until_all_vector_on_stream(TYPE *ivars, size_t nelems, \
                                                                        const int *status, int cmp, \
                                                                        TYPE *cmp_value, cudaStream_t cstream) { \
        nvshmemx_##NAME##_wait_until_all_vector_on_stream(ivars, nelems, status, cmp, cmp_value, cstream); \
    }

#define UCL_SHMEMX_REPT_FOR_WAIT_TYPES(MACRO) \
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

UCL_SHMEMX_REPT_FOR_WAIT_TYPES(UCL_SHMEMX_DECL_WAIT_UNTIL_ON_STREAM)
UCL_SHMEMX_REPT_FOR_WAIT_TYPES(UCL_SHMEMX_DECL_WAIT_UNTIL_ALL_ON_STREAM)
UCL_SHMEMX_REPT_FOR_WAIT_TYPES(UCL_SHMEMX_DECL_WAIT_UNTIL_ALL_VECTOR_ON_STREAM)

static inline void shmemx_signal_wait_until_on_stream(uint64_t *sig_addr, int cmp, uint64_t cmp_value, cudaStream_t cstream) {
    nvshmemx_signal_wait_until_on_stream(sig_addr, cmp, cmp_value, cstream);
}

// ===== Put on Thread Group =====
#ifdef __CUDACC__
#define UCL_SHMEMX_DECL_TYPE_PUT_THREADGROUP(NAME, TYPE) \
    __device__ void shmemx_##NAME##_put_warp(TYPE *dest, const TYPE *source, size_t nelems, int pe); \
    __device__ void shmemx_##NAME##_put_block(TYPE *dest, const TYPE *source, size_t nelems, int pe);

#define UCL_SHMEMX_DECL_SIZE_PUT_THREADGROUP(NAME) \
    __device__ void shmemx_put##NAME##_warp(void *dest, const void *source, size_t nelems, int pe); \
    __device__ void shmemx_put##NAME##_block(void *dest, const void *source, size_t nelems, int pe);

UCL_SHMEMX_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEMX_DECL_TYPE_PUT_THREADGROUP)

UCL_SHMEMX_DECL_SIZE_PUT_THREADGROUP(8)
UCL_SHMEMX_DECL_SIZE_PUT_THREADGROUP(16)
UCL_SHMEMX_DECL_SIZE_PUT_THREADGROUP(32)
UCL_SHMEMX_DECL_SIZE_PUT_THREADGROUP(64)
UCL_SHMEMX_DECL_SIZE_PUT_THREADGROUP(128)

__device__ void shmemx_putmem_warp(void *dest, const void *source, size_t bytes, int pe);
__device__ void shmemx_putmem_block(void *dest, const void *source, size_t bytes, int pe);

// Put signal warp/block
#define UCL_SHMEMX_DECL_TYPE_PUT_SIGNAL_SCOPE(NAME, TYPE) \
    __device__ void shmemx_##NAME##_put_signal_warp(TYPE *dest, const TYPE *source, size_t nelems, \
                                                    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe); \
    __device__ void shmemx_##NAME##_put_signal_block(TYPE *dest, const TYPE *source, size_t nelems, \
                                                     uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);

#define UCL_SHMEMX_DECL_TYPE_PUT_SIGNAL_NBI_SCOPE(NAME, TYPE) \
    __device__ void shmemx_##NAME##_put_signal_nbi_warp(TYPE *dest, const TYPE *source, size_t nelems, \
                                                        uint64_t *sig_addr, uint64_t signal, int sig_op, int pe); \
    __device__ void shmemx_##NAME##_put_signal_nbi_block(TYPE *dest, const TYPE *source, size_t nelems, \
                                                         uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);

UCL_SHMEMX_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEMX_DECL_TYPE_PUT_SIGNAL_SCOPE)
UCL_SHMEMX_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEMX_DECL_TYPE_PUT_SIGNAL_NBI_SCOPE)

__device__ void shmemx_putmem_signal_warp(void *dest, const void *source, size_t nelems,
                                          uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ void shmemx_putmem_signal_block(void *dest, const void *source, size_t nelems,
                                           uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ void shmemx_putmem_signal_nbi_warp(void *dest, const void *source, size_t nelems,
                                              uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);
__device__ void shmemx_putmem_signal_nbi_block(void *dest, const void *source, size_t nelems,
                                               uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);

#define UCL_SHMEMX_DECL_SIZE_PUT_SIGNAL_SCOPE(NAME) \
    __device__ void shmemx_put##NAME##_signal_warp(void *dest, const void *source, size_t nelems, \
                                                   uint64_t *sig_addr, uint64_t signal, int sig_op, int pe); \
    __device__ void shmemx_put##NAME##_signal_block(void *dest, const void *source, size_t nelems, \
                                                    uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);

#define UCL_SHMEMX_DECL_SIZE_PUT_SIGNAL_NBI_SCOPE(NAME) \
    __device__ void shmemx_put##NAME##_signal_nbi_warp(void *dest, const void *source, size_t nelems, \
                                                       uint64_t *sig_addr, uint64_t signal, int sig_op, int pe); \
    __device__ void shmemx_put##NAME##_signal_nbi_block(void *dest, const void *source, size_t nelems, \
                                                        uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);

UCL_SHMEMX_DECL_SIZE_PUT_SIGNAL_SCOPE(8)
UCL_SHMEMX_DECL_SIZE_PUT_SIGNAL_SCOPE(6)
UCL_SHMEMX_DECL_SIZE_PUT_SIGNAL_SCOPE(32)
UCL_SHMEMX_DECL_SIZE_PUT_SIGNAL_SCOPE(64)
UCL_SHMEMX_DECL_SIZE_PUT_SIGNAL_SCOPE(128)

UCL_SHMEMX_DECL_SIZE_PUT_SIGNAL_NBI_SCOPE(8)
UCL_SHMEMX_DECL_SIZE_PUT_SIGNAL_NBI_SCOPE(6)
UCL_SHMEMX_DECL_SIZE_PUT_SIGNAL_NBI_SCOPE(32)
UCL_SHMEMX_DECL_SIZE_PUT_SIGNAL_NBI_SCOPE(64)
UCL_SHMEMX_DECL_SIZE_PUT_SIGNAL_NBI_SCOPE(128)

// iPut threadgroup
#define UCL_SHMEMX_DECL_TYPE_IPUT_THREADGROUP(NAME, TYPE) \
    __device__ void shmemx_##NAME##_iput_warp(TYPE *dest, const TYPE *source, ptrdiff_t dst, \
                                              ptrdiff_t sst, size_t nelems, int pe); \
    __device__ void shmemx_##NAME##_iput_block(TYPE *dest, const TYPE *source, ptrdiff_t dst, \
                                               ptrdiff_t sst, size_t nelems, int pe);

#define UCL_SHMEMX_DECL_SIZE_IPUT_THREADGROUP(NAME) \
    __device__ void shmemx_iput##NAME##_warp(void *dest, const void *source, ptrdiff_t dst, \
                                             ptrdiff_t sst, size_t nelems, int pe); \
    __device__ void shmemx_iput##NAME##_block(void *dest, const void *source, ptrdiff_t dst, \
                                              ptrdiff_t sst, size_t nelems, int pe);

UCL_SHMEMX_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEMX_DECL_TYPE_IPUT_THREADGROUP)

UCL_SHMEMX_DECL_SIZE_IPUT_THREADGROUP(8)
UCL_SHMEMX_DECL_SIZE_IPUT_THREADGROUP(16)
UCL_SHMEMX_DECL_SIZE_IPUT_THREADGROUP(32)
UCL_SHMEMX_DECL_SIZE_IPUT_THREADGROUP(64)
UCL_SHMEMX_DECL_SIZE_IPUT_THREADGROUP(128)

// Put nbi threadgroup
#define UCL_SHMEMX_DECL_TYPE_PUT_NBI_THREADGROUP(NAME, TYPE) \
    __device__ void shmemx_##NAME##_put_nbi_warp(TYPE *dest, const TYPE *source, size_t nelems, int pe); \
    __device__ void shmemx_##NAME##_put_nbi_block(TYPE *dest, const TYPE *source, size_t nelems, int pe);

#define UCL_SHMEMX_DECL_SIZE_PUT_NBI_THREADGROUP(NAME) \
    __device__ void shmemx_put##NAME##_nbi_warp(void *dest, const void *source, size_t nelems, int pe); \
    __device__ void shmemx_put##NAME##_nbi_block(void *dest, const void *source, size_t nelems, int pe);

UCL_SHMEMX_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEMX_DECL_TYPE_PUT_NBI_THREADGROUP)

UCL_SHMEMX_DECL_SIZE_PUT_NBI_THREADGROUP(8)
UCL_SHMEMX_DECL_SIZE_PUT_NBI_THREADGROUP(16)
UCL_SHMEMX_DECL_SIZE_PUT_NBI_THREADGROUP(32)
UCL_SHMEMX_DECL_SIZE_PUT_NBI_THREADGROUP(64)
UCL_SHMEMX_DECL_SIZE_PUT_NBI_THREADGROUP(128)

__device__ void shmemx_putmem_nbi_warp(void *dest, const void *source, size_t bytes, int pe);
__device__ void shmemx_putmem_nbi_block(void *dest, const void *source, size_t bytes, int pe);

// ===== Get on Thread Group =====
#define UCL_SHMEMX_DECL_TYPE_GET_THREADGROUP(NAME, TYPE) \
    __device__ void shmemx_##NAME##_get_warp(TYPE *dest, const TYPE *source, size_t nelems, int pe); \
    __device__ void shmemx_##NAME##_get_block(TYPE *dest, const TYPE *source, size_t nelems, int pe);

#define UCL_SHMEMX_DECL_TYPE_IGET_THREADGROUP(NAME, TYPE) \
    __device__ void shmemx_##NAME##_iget_warp(TYPE *dest, const TYPE *source, ptrdiff_t dst, \
                                              ptrdiff_t sst, size_t nelems, int pe); \
    __device__ void shmemx_##NAME##_iget_block(TYPE *dest, const TYPE *source, ptrdiff_t dst, \
                                               ptrdiff_t sst, size_t nelems, int pe);

#define UCL_SHMEMX_DECL_SIZE_GET_THREADGROUP(NAME) \
    __device__ void shmemx_get##NAME##_warp(void *dest, const void *source, size_t nelems, int pe); \
    __device__ void shmemx_get##NAME##_block(void *dest, const void *source, size_t nelems, int pe);

#define UCL_SHMEMX_DECL_SIZE_IGET_THREADGROUP(NAME) \
    __device__ void shmemx_iget##NAME##_warp(void *dest, const void *source, ptrdiff_t dst, \
                                             ptrdiff_t sst, size_t nelems, int pe); \
    __device__ void shmemx_iget##NAME##_block(void *dest, const void *source, ptrdiff_t dst, \
                                              ptrdiff_t sst, size_t nelems, int pe);

UCL_SHMEMX_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEMX_DECL_TYPE_GET_THREADGROUP)
UCL_SHMEMX_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEMX_DECL_TYPE_IGET_THREADGROUP)

UCL_SHMEMX_DECL_SIZE_GET_THREADGROUP(8)
UCL_SHMEMX_DECL_SIZE_GET_THREADGROUP(16)
UCL_SHMEMX_DECL_SIZE_GET_THREADGROUP(32)
UCL_SHMEMX_DECL_SIZE_GET_THREADGROUP(64)
UCL_SHMEMX_DECL_SIZE_GET_THREADGROUP(128)

UCL_SHMEMX_DECL_SIZE_IGET_THREADGROUP(8)
UCL_SHMEMX_DECL_SIZE_IGET_THREADGROUP(16)
UCL_SHMEMX_DECL_SIZE_IGET_THREADGROUP(32)
UCL_SHMEMX_DECL_SIZE_IGET_THREADGROUP(64)
UCL_SHMEMX_DECL_SIZE_IGET_THREADGROUP(128)

__device__ void shmemx_getmem_warp(void *dest, const void *source, size_t bytes, int pe);
__device__ void shmemx_getmem_block(void *dest, const void *source, size_t bytes, int pe);

// Get nbi threadgroup
#define UCL_SHMEMX_DECL_TYPE_GET_NBI_THREADGROUP(NAME, TYPE) \
    __device__ void shmemx_##NAME##_get_nbi_warp(TYPE *dest, const TYPE *source, size_t nelems, int pe); \
    __device__ void shmemx_##NAME##_get_nbi_block(TYPE *dest, const TYPE *source, size_t nelems, int pe);

#define UCL_SHMEMX_DECL_SIZE_GET_NBI_THREADGROUP(NAME) \
    __device__ void shmemx_get##NAME##_nbi_warp(void *dest, const void *source, size_t nelems, int pe); \
    __device__ void shmemx_get##NAME##_nbi_block(void *dest, const void *source, size_t nelems, int pe);

UCL_SHMEMX_REPT_FOR_STANDARD_RMA_TYPES(UCL_SHMEMX_DECL_TYPE_GET_NBI_THREADGROUP)

UCL_SHMEMX_DECL_SIZE_GET_NBI_THREADGROUP(8)
UCL_SHMEMX_DECL_SIZE_GET_NBI_THREADGROUP(16)
UCL_SHMEMX_DECL_SIZE_GET_NBI_THREADGROUP(32)
UCL_SHMEMX_DECL_SIZE_GET_NBI_THREADGROUP(64)
UCL_SHMEMX_DECL_SIZE_GET_NBI_THREADGROUP(128)

__device__ void shmemx_getmem_nbi_warp(void *dest, const void *source, size_t bytes, int pe);
__device__ void shmemx_getmem_nbi_block(void *dest, const void *source, size_t bytes, int pe);
#endif // __CUDACC__

// ===== Signal =====
NVSHMEMI_HOSTDEVICE_PREFIX void shmemx_signal_op(uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);

} // namespace shmem
} // namespace ucl

#endif // UCL_SHMEMX_HOST_API_H
