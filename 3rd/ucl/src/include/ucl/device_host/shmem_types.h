/*
 * UCL SHMEM API - Device-Host Types
 * This file enhances NVSHMEM device-host types into UCL shmem namespace
 */

#ifndef UCL_SHMEM_DEVICE_HOST_TYPES_H
#define UCL_SHMEM_DEVICE_HOST_TYPES_H

#include "device_host/nvshmem_types.h"

namespace ucl {
namespace shmem {

// Type aliases for NVSHMEM types
using shmemx_team_uniqueid_t = nvshmemx_team_uniqueid_t;
using shmem_team_t = nvshmem_team_t;
using shmemx_team_t = nvshmemx_team_t;
using shmemx_init_args_t = nvshmemx_init_args_t;
using shmemx_init_attr_t = nvshmemx_init_attr_t;
using shmem_team_config_t = nvshmem_team_config_t;

inline const shmemx_init_args_t SHMEMX_INIT_ARGS_INITIALIZER = NVSHMEMX_INIT_ARGS_INITIALIZER;
inline const shmemx_init_attr_t SHMEMX_INIT_ATTR_INITIALIZER = NVSHMEMX_INIT_ATTR_INITIALIZER;

// Re-export team constants
// Note: The actual constants (NVSHMEM_TEAM_WORLD, NVSHMEM_TEAM_SHARED, etc.)
// are defined elsewhere and should be accessed via nvshmem_* naming

// Macros cannot be aliased, but we re-export them for documentation
// SHMEM_TEAM_WORLD is equivalent to NVSHMEM_TEAM_WORLD
// SHMEM_TEAM_SHARED is equivalent to NVSHMEM_TEAM_SHARED

} // namespace shmem
} // namespace ucl

#endif // UCL_SHMEM_DEVICE_HOST_TYPES_H
