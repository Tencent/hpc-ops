/*
 * UCL SHMEM API - Bootstrap Device-Host UniqueID
 * This file enhances NVSHMEM unique ID types into UCL shmem namespace
 */

#ifndef UCL_SHMEM_BOOTSTRAP_DEVICE_HOST_UNIQUEID_H
#define UCL_SHMEM_BOOTSTRAP_DEVICE_HOST_UNIQUEID_H

#include "bootstrap_device_host/nvshmem_uniqueid.h"

namespace ucl {
namespace shmem {

// ===== Constant Definitions =====
// Enhance NVSHMEM unique ID constants to SHMEM versions (accessible as ucl::shmem::SHMEM_XXX)
constexpr int SHMEM_UNIQUEID_PADDING = UNIQUEID_PADDING;
constexpr int SHMEM_UNIQUEID_ARGS_INVALID = UNIQUEID_ARGS_INVALID;

// ===== Type Aliases for NVSHMEM unique ID types =====
using shmemx_uniqueid_v1 = nvshmemx_uniqueid_v1;
using shmemx_uniqueid_t = nvshmemx_uniqueid_t;
using shmemx_uniqueid_args_v1 = nvshmemx_uniqueid_args_v1;
using shmemx_uniqueid_args_t = nvshmemx_uniqueid_args_t;

// ===== Initializer Constants =====
// Enhance NVSHMEM initializers as inline constants (accessible as ucl::shmem::SHMEMX_XXX)
#if !defined __CUDACC_RTC__
inline constexpr shmemx_uniqueid_t SHMEMX_UNIQUEID_INITIALIZER = NVSHMEMX_UNIQUEID_INITIALIZER;
inline const shmemx_uniqueid_args_t SHMEMX_UNIQUEID_ARGS_INITIALIZER = NVSHMEMX_UNIQUEID_ARGS_INITIALIZER;
#endif

} // namespace shmem
} // namespace ucl

#endif // UCL_SHMEM_BOOTSTRAP_DEVICE_HOST_UNIQUEID_H
