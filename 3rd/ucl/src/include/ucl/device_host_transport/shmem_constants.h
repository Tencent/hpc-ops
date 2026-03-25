/*
 * UCL SHMEM API - Device-Host Transport Constants
 * This file enhances NVSHMEM transport constants into UCL shmem namespace
 *
 * All NVSHMEM/nvshmemx constants and types are wrapped into ucl::shmem namespace
 * with SHMEM_/shmemx_ prefix, e.g.:
 *   ucl::shmem::SHMEM_CMP_EQ
 *   ucl::shmem::SHMEM_THREAD_MULTIPLE
 *   ucl::shmem::shmemx_cmp_type_t
 */

#ifndef UCL_SHMEM_DEVICE_HOST_TRANSPORT_CONSTANTS_H
#define UCL_SHMEM_DEVICE_HOST_TRANSPORT_CONSTANTS_H

#include "device_host_transport/nvshmem_constants.h"

namespace ucl {
namespace shmem {

// ===== OpenSHMEM Spec Version Constants =====
constexpr int SHMEM_MAJOR_VERSION = NVSHMEM_MAJOR_VERSION;
constexpr int SHMEM_MINOR_VERSION = NVSHMEM_MINOR_VERSION;

// ===== Vendor Version =====
constexpr int SHMEM_VENDOR_VERSION = NVSHMEM_VENDOR_VERSION;

// ===== Max Name Length =====
constexpr int SHMEM_MAX_NAME_LEN = NVSHMEM_MAX_NAME_LEN;

// ===== Comparison Operators =====
// Wraps nvshmemi_cmp_type enum values
enum shmem_cmp_type {
    SHMEM_CMP_EQ       = NVSHMEM_CMP_EQ,
    SHMEM_CMP_NE       = NVSHMEM_CMP_NE,
    SHMEM_CMP_GT       = NVSHMEM_CMP_GT,
    SHMEM_CMP_LE       = NVSHMEM_CMP_LE,
    SHMEM_CMP_LT       = NVSHMEM_CMP_LT,
    SHMEM_CMP_GE       = NVSHMEM_CMP_GE,
    SHMEM_CMP_SENTINEL  = NVSHMEM_CMP_SENTINEL,
};
using shmemx_cmp_type_t = nvshmemx_cmp_type_t;

// ===== Thread Support Levels =====
// Wraps nvshmemi_thread_support enum values
enum shmem_thread_support {
    SHMEM_THREAD_SINGLE     = NVSHMEM_THREAD_SINGLE,
    SHMEM_THREAD_FUNNELED   = NVSHMEM_THREAD_FUNNELED,
    SHMEM_THREAD_SERIALIZED = NVSHMEM_THREAD_SERIALIZED,
    SHMEM_THREAD_MULTIPLE   = NVSHMEM_THREAD_MULTIPLE,
    SHMEM_THREAD_TYPE_SENTINEL = NVSHMEM_THREAD_TYPE_SENTINEL,
};

// ===== Initialization Status =====
// Wraps nvshmemx_init_status_t enum values
enum shmemx_init_status {
    SHMEM_STATUS_NOT_INITIALIZED = NVSHMEM_STATUS_NOT_INITIALIZED,
    SHMEM_STATUS_IS_BOOTSTRAPPED = NVSHMEM_STATUS_IS_BOOTSTRAPPED,
    SHMEM_STATUS_IS_INITIALIZED  = NVSHMEM_STATUS_IS_INITIALIZED,
    SHMEM_STATUS_LIMITED_MPG     = NVSHMEM_STATUS_LIMITED_MPG,
    SHMEM_STATUS_FULL_MPG        = NVSHMEM_STATUS_FULL_MPG,
    SHMEM_STATUS_INVALID         = NVSHMEM_STATUS_INVALID,
};
using shmemx_init_status_t = nvshmemx_init_status_t;

} // namespace shmem
} // namespace ucl

#endif // UCL_SHMEM_DEVICE_HOST_TRANSPORT_CONSTANTS_H
