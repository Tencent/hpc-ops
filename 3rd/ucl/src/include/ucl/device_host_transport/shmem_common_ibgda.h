/*
 * UCL SHMEM API - Device-Host IBGDA
 * This file enhances NVSHMEM IBGDA transport types into UCL shmem namespace
 * 
 * Note: This is a transport-level header. Some functionality may overlap
 * with extensions/ directory which should not be modified.
 */

#ifndef UCL_SHMEM_DEVICE_HOST_TRANSPORT_COMMON_IBGDA_H
#define UCL_SHMEM_DEVICE_HOST_TRANSPORT_COMMON_IBGDA_H

#include "device_host_transport/nvshmem_common_ibgda.h"

// Re-export IBGDA types and structures
// The nvshmem_common_ibgda.h defines IBGDA (InfiniBand GPU Direct Async) related types
// These are used for low-level transport operations

#endif // UCL_SHMEM_DEVICE_HOST_TRANSPORT_COMMON_IBGDA_H
