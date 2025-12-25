/*
 * Copyright (c) 2018-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See License.txt for license information
 */

#ifndef _NVSHMEMX_TILE_API_HPP_
#define _NVSHMEMX_TILE_API_HPP_
#include <cuda_runtime.h>
#include "device_host/nvshmem_tensor.h"

namespace nvshmemx {

// Tile AllReduce
#define DECL_NVSHMEMX_OP_TILE_ALLREDUCE_THREADGROUP(SCOPE_SUFFIX, OP)                 \
    template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t,         \
              nvshmemx::tile_coll_algo_t algo>                                        \
    NVSHMEMI_DEVICE_PREFIX int tile_##OP##_allreduce##SCOPE_SUFFIX(                   \
        nvshmem_team_t team, src_tensor_t src, dst_tensor_t dst, tuple_t start_coord, \
        tuple_t boundary, int root, uint64_t flag);                                   \
                                                                                      \
    /* overloaded version for with no root, only for one shot */                      \
    template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t,         \
              nvshmemx::tile_coll_algo_t algo>                                        \
    NVSHMEMI_DEVICE_PREFIX int tile_##OP##_allreduce##SCOPE_SUFFIX(                   \
        nvshmem_team_t team, src_tensor_t src, dst_tensor_t dst, tuple_t start_coord, \
        tuple_t boundary, uint64_t flag);

#define DECL_NVSHMEMX_OP_TILE_ALLREDUCE(SC_suffix)              \
    DECL_NVSHMEMX_OP_TILE_ALLREDUCE_THREADGROUP(SC_suffix, sum) \
    DECL_NVSHMEMX_OP_TILE_ALLREDUCE_THREADGROUP(SC_suffix, max) \
    DECL_NVSHMEMX_OP_TILE_ALLREDUCE_THREADGROUP(SC_suffix, min)

DECL_NVSHMEMX_OP_TILE_ALLREDUCE();            // thread
DECL_NVSHMEMX_OP_TILE_ALLREDUCE(_warp);       // warp
DECL_NVSHMEMX_OP_TILE_ALLREDUCE(_warpgroup);  // warpgroup
DECL_NVSHMEMX_OP_TILE_ALLREDUCE(_block);      // block

// Tile AllGather
#define DECL_NVSHMEMX_TILE_ALLGATHER_THREADGROUP(SCOPE_SUFFIX)                                     \
    template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t,                      \
              nvshmemx::tile_coll_algo_t algo>                                                     \
    NVSHMEMI_DEVICE_PREFIX int tile_allgather##SCOPE_SUFFIX(nvshmem_team_t team, src_tensor_t src, \
                                                            dst_tensor_t dst, tuple_t start_coord, \
                                                            tuple_t boundary, uint64_t flag);

DECL_NVSHMEMX_TILE_ALLGATHER_THREADGROUP();            // thread
DECL_NVSHMEMX_TILE_ALLGATHER_THREADGROUP(_warp);       // warp
DECL_NVSHMEMX_TILE_ALLGATHER_THREADGROUP(_warpgroup);  // warpgroup
DECL_NVSHMEMX_TILE_ALLGATHER_THREADGROUP(_block);      // block

// Tile Reduce
#define DECL_NVSHMEMX_OP_TILE_REDUCE_THREADGROUP(SCOPE_SUFFIX, OP)                    \
    template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t,         \
              nvshmemx::tile_coll_algo_t algo>                                        \
    NVSHMEMI_DEVICE_PREFIX int tile_##OP##_reduce##SCOPE_SUFFIX(                      \
        nvshmem_team_t team, src_tensor_t src, dst_tensor_t dst, tuple_t start_coord, \
        tuple_t boundary, int root, uint64_t flag);

#define DECL_NVSHMEMX_OP_TILE_REDUCE(SC_suffix)              \
    DECL_NVSHMEMX_OP_TILE_REDUCE_THREADGROUP(SC_suffix, sum) \
    DECL_NVSHMEMX_OP_TILE_REDUCE_THREADGROUP(SC_suffix, max) \
    DECL_NVSHMEMX_OP_TILE_REDUCE_THREADGROUP(SC_suffix, min)

DECL_NVSHMEMX_OP_TILE_REDUCE();            // thread
DECL_NVSHMEMX_OP_TILE_REDUCE(_warp);       // warp
DECL_NVSHMEMX_OP_TILE_REDUCE(_warpgroup);  // warpgroup
DECL_NVSHMEMX_OP_TILE_REDUCE(_block);      // block

// Tile Coll Wait
#define DECL_NVSHMEMX_TILE_COLLECTIVE_WAIT_THREADGROUP(SCOPE_SUFFIX)                   \
    template <nvshmemx::tile_coll_algo_t algo>                                         \
    NVSHMEMI_DEVICE_PREFIX int tile_collective_wait##SCOPE_SUFFIX(nvshmem_team_t team, \
                                                                  uint64_t flag);

DECL_NVSHMEMX_TILE_COLLECTIVE_WAIT_THREADGROUP();            // thread
DECL_NVSHMEMX_TILE_COLLECTIVE_WAIT_THREADGROUP(_warp);       // warp
DECL_NVSHMEMX_TILE_COLLECTIVE_WAIT_THREADGROUP(_warpgroup);  // warpgroup
DECL_NVSHMEMX_TILE_COLLECTIVE_WAIT_THREADGROUP(_block);      // block

}  // namespace nvshmemx

#endif
