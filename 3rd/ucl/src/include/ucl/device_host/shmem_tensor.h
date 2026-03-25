/*
 * UCL SHMEM API - Device-Host Tensor
 * This file enhances NVSHMEM tensor types into UCL shmem namespace
 */

#ifndef UCL_SHMEM_DEVICE_HOST_TENSOR_H
#define UCL_SHMEM_DEVICE_HOST_TENSOR_H

#include "device_host/nvshmem_tensor.h"

namespace ucl {
namespace shmem {

// Re-export nvshmemx tensor types
// The Tensor, Layout, shape, stride types are in nvshmemx namespace
// Users can access them via nvshmemx:: prefix

// Tile collective algorithm types
using tile_coll_algo_t = nvshmemx::tile_coll_algo_t;

// Re-export algorithm constants
// NVLS_ONE_SHOT_PUSH_NBI, NVLS_ONE_SHOT_PULL_NBI, NVLS_TWO_SHOT_PUSH_NBI, NVLS_TWO_SHOT_PULL_NBI

} // namespace shmem
} // namespace ucl

#endif // UCL_SHMEM_DEVICE_HOST_TENSOR_H
