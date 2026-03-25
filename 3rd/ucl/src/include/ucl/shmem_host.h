/*
 * UCL SHMEM Host-Only API Header File
 * 
 * This header provides access to host-only SHMEM APIs.
 * Use this header for host-side code that doesn't need device-side functions.
 * For the complete API including device functions, use shmem.h instead.
 * 
 * Usage:
 *   #include "shmem_host.h"
 *   
 *   // Use ucl::shmem::shmem_* to call host functions
 *   ucl::shmem::shmem_init();
 *   int pe = ucl::shmem::shmem_my_pe();
 *   void *ptr = ucl::shmem::shmem_malloc(size);
 *   ucl::shmem::shmem_finalize();
 */

#ifndef UCL_SHMEM_HOST_H
#define UCL_SHMEM_HOST_H

//==========================================
// Bootstrap Device-Host headers (unique ID)
//==========================================
#include "bootstrap_device_host/shmem_uniqueid.h"

//==========================================
// Device-Host shared headers (types)
//==========================================
#include "device_host/shmem_types.h"

//==========================================
// Device-Host Transport headers (constants)
//==========================================
#include "device_host_transport/shmem_constants.h"

//==========================================
// Host-only headers
//==========================================
#include "host/shmem_macros.h"
#include "host/shmem_api.h"
#include "host/shmemx_api.h"
#include "host/shmem_coll_api.h"
#include "host/shmemx_coll_api.h"

/*
 * UCL SHMEM Host API Summary:
 * 
 * This header provides host-only APIs for SHMEM programming.
 * All functions are under the ucl::shmem namespace.
 * 
 * Host API Categories:
 *   1. Library Initialization/Finalization: shmem_init, shmem_finalize, shmemx_init_attr
 *   2. PE Information Query: shmem_my_pe, shmem_n_pes, shmem_info_get_*
 *   3. Heap Management: shmem_malloc, shmem_free, shmem_calloc, shmem_align, shmem_realloc
 *   4. Buffer Management: shmemx_buffer_register, shmemx_buffer_unregister
 *   5. RMA Operations: shmem_<type>_put, shmem_<type>_get, shmem_putmem, shmem_getmem
 *   6. Atomic Operations: shmem_<type>_atomic_*, shmem_<type>_atomic_fetch_*
 *   7. Synchronization: shmem_quiet, shmem_fence, shmem_barrier, shmem_barrier_all
 *   8. Collective Communication: shmem_<type>_alltoall, shmem_<type>_broadcast, shmem_<type>_reduce
 *   9. Teams API: shmem_team_my_pe, shmem_team_n_pes, shmem_team_split_*
 *   10. On-Stream Operations: shmemx_*_on_stream (put, get, barrier, reduce, etc.)
 * 
 * Note: Device-side functions (warp/block level operations, device signal operations)
 * are not included in this header. Use shmem.h with __CUDACC__ for device-side code.
 * 
 * See shmem.h for the complete API reference including device functions.
 */

// Import shmem namespace contents into ucl namespace
namespace ucl {
    using namespace shmem;
}

#endif // UCL_SHMEM_HOST_H
