/*
 * UCL SHMEM Main Header File
 * 
 * This is the UCL (Unified Communication Library).
 * 
 * Directory Structure:
 *   - bootstrap_device_host/  : Bootstrap unique ID types and functions
 *   - device_host/            : Types and APIs callable from both host and device
 *   - device_host_transport/  : Transport constants and IBGDA definitions
 *   - host/                   : Host-only APIs (initialization, on_stream operations)
 *   - device/                 : Device-only APIs (signal operations, warp/block level)
 *   - device/tile/            : Tile-level collective operations
 *   - extensions/             : UCL-specific extensions (IBGDA, etc.)
 * 
 * Usage:
 *   #include "shmem.h"
 *   
 *   // Use ucl::shmem::shmem_* or ucl::shmem::shmemx_* to call functions
 *   ucl::shmem::shmem_init();
 *   int pe = ucl::shmem::shmem_my_pe();
 *   ucl::shmem::shmemx_vendor_get_version_info(&major, &minor, &patch);
 *   ucl::shmem::shmem_finalize();
 * 
 */

#ifndef UCL_SHMEM_H
#define UCL_SHMEM_H

//==========================================
// Bootstrap Device-Host headers (unique ID)
//==========================================
#include "bootstrap_device_host/shmem_uniqueid.h"

//==========================================
// Device-Host shared headers (types and APIs callable from both sides)
//==========================================
#include "device_host/shmem_types.h"
#include "device_host/shmem_common.cuh"
#include "device_host/shmem_tensor.h"
#include "device_host/shmem_proxy_channel.h"

//==========================================
// Device-Host Transport headers (constants)
//==========================================
#include "device_host_transport/shmem_constants.h"
#include "device_host_transport/shmem_common_ibgda.h"
#include "device_host_transport/shmem_common_transport.h"

//==========================================
// Host-only headers
//==========================================
#include "host/shmem_macros.h"
#include "host/shmem_api.h"
#include "host/shmemx_api.h"
#include "host/shmem_coll_api.h"
#include "host/shmemx_coll_api.h"

//==========================================
// Device-only headers (requires CUDA compilation)
//==========================================
#ifdef __CUDACC__
#include "device/shmem_device_macros.cuh"
#include "device/shmem_defines.cuh"
#include "device/shmemx_defines.cuh"
#include "device/shmem_coll_defines.cuh"
#include "device/shmemx_coll_defines.cuh"
#include "device/shmemx_collective_launch_apis.cuh"

// UCL-specific extensions (IBGDA, etc.)
#include "shmem_extensions.h"
#endif

// Import shmem namespace contents into ucl namespace
// This allows users to call functions directly using ucl::function() form
namespace ucl {
    using namespace shmem;
}

#endif // UCL_SHMEM_H
