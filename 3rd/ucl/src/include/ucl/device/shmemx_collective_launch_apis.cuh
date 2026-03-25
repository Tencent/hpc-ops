/*
 * UCL SHMEM API - Device Collective Launch APIs
 * This file enhances NVSHMEM collective launch APIs into UCL shmem namespace
 */

#ifndef UCL_SHMEMX_DEVICE_COLLECTIVE_LAUNCH_APIS_CUH
#define UCL_SHMEMX_DEVICE_COLLECTIVE_LAUNCH_APIS_CUH

#include "device/nvshmemx_collective_launch_apis.h"

namespace ucl {
namespace shmem {

#if !defined __CUDACC_RTC__
static inline int shmemx_collective_launch(const void *func, dim3 gridDims, dim3 blockDims, void **args,
                                            size_t sharedMem, cudaStream_t stream) {
    return nvshmemx_collective_launch(func, gridDims, blockDims, args, sharedMem, stream);
}

static inline int shmemx_collective_launch_query_gridsize(const void *func, dim3 blockDims, void **args,
                                                           size_t sharedMem, int *gridsize) {
    return nvshmemx_collective_launch_query_gridsize(func, blockDims, args, sharedMem, gridsize);
}
#endif

} // namespace shmem
} // namespace ucl

#endif // UCL_SHMEMX_DEVICE_COLLECTIVE_LAUNCH_APIS_CUH
