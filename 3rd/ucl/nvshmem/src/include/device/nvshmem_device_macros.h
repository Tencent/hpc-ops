#ifndef _NVSHMEM_DEVICE_MACROS_H_
#define _NVSHMEM_DEVICE_MACROS_H_

#include "non_abi/nvshmem_build_options.h"  // IWYU pragma: keep

/*
 * These macros represent various inlining requirements based on configuration rules.
 * All functions are force inlined in the bitcode library.
 * Macro Key:
 * NVSHMEMI_DEVICE_INLINE - inlined based on NVSHMEM_ENABLE_ALL_DEVICE_INLINING
 * NVSHMEMI_DEVICE_ALWAYS_INLINE - inlined regardless of NVSHMEM_ENABLE_ALL_DEVICE_INLINING
 * NVSHMEMI_DEVICE_ALWAYS_FORCE_INLINE - like above, but uses NVSHMEMI_DEVICE_ALWAYS_FORCE_INLINE
 */

#if !defined __clang_llvm_bitcode_lib__
#define NVSHMEMI_DEVICE_ALWAYS_INLINE inline
#define NVSHMEMI_DEVICE_ALWAYS_FORCE_INLINE __forceinline__
#define NVSHMEM_ALWAYS_STATIC static
#if defined NVSHMEM_ENABLE_ALL_DEVICE_INLINING
#define NVSHMEMI_STATIC static
#define NVSHMEMI_DEVICE_INLINE inline
#else
#define NVSHMEMI_DEVICE_INLINE __noinline__
#define NVSHMEMI_STATIC static
#endif
#else
/* clang llvm ir compilation mangles names of functions marked NVSHMEMI_STATIC
 * even if they are behind extern c guards. */
#define NVSHMEMI_STATIC
#if defined NVSHMEM_ENABLE_ALL_DEVICE_INLINING
#define NVSHMEMI_DEVICE_INLINE __attribute__((always_inline))
#else
#define NVSHMEMI_DEVICE_INLINE __noinline__
#endif
#define NVSHMEMI_DEVICE_ALWAYS_INLINE __attribute__((always_inline))
#define NVSHMEMI_DEVICE_ALWAYS_FORCE_INLINE __attribute__((always_inline))
#endif

#endif