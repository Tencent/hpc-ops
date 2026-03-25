
#ifndef _NVSHMEMX_TILE_API_DEFINES_CUH_
#define _NVSHMEMX_TILE_API_DEFINES_CUH_

#include <cuda_runtime.h>

#include "device_host/nvshmem_common.cuh"
#include "device/nvshmem_coll_defines.cuh"
#include "device/nvshmem_device_macros.h"

#ifdef __CUDA_ARCH__

namespace nvshmemx {
// Tile AllReduce
#define DEFN_NVSHMEMX_OP_TILE_ALLREDUCE_THREADGROUP(SC, SC_SUFFIX, SC_PREFIX, OP)     \
    template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t,         \
              nvshmemx::tile_coll_algo_t algo>                                        \
    NVSHMEMI_STATIC NVSHMEMI_DEVICE_PREFIX int tile_##OP##_allreduce##SC_SUFFIX(      \
        nvshmem_team_t team, src_tensor_t src, dst_tensor_t dst, tuple_t start_coord, \
        tuple_t boundary, int root, uint64_t flag) {                                  \
        nvshmemi_tile_allreduce<algo, src_tensor_t, dst_tensor_t, tuple_t,            \
                                nvshmemi_threadgroup_##SC, RDXN_OPS_##OP>(            \
            team, src, dst, start_coord, boundary, root, flag);                       \
        return 0;                                                                     \
    }                                                                                 \
    template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t,         \
              nvshmemx::tile_coll_algo_t algo>                                        \
    NVSHMEMI_STATIC NVSHMEMI_DEVICE_PREFIX int tile_##OP##_allreduce##SC_SUFFIX(      \
        nvshmem_team_t team, src_tensor_t src, dst_tensor_t dst, tuple_t start_coord, \
        tuple_t boundary, uint64_t flag) {                                            \
        nvshmemi_tile_allreduce<algo, src_tensor_t, dst_tensor_t, tuple_t,            \
                                nvshmemi_threadgroup_##SC, RDXN_OPS_##OP>(            \
            team, src, dst, start_coord, boundary, -1, flag);                         \
        return 0;                                                                     \
    }

#define DEFN_NVSHMEM_TILE_ALLREDUCE_THREADGROUP(SC, SC_SUFFIX, SC_PREFIX)      \
    DEFN_NVSHMEMX_OP_TILE_ALLREDUCE_THREADGROUP(SC, SC_SUFFIX, SC_PREFIX, sum) \
    DEFN_NVSHMEMX_OP_TILE_ALLREDUCE_THREADGROUP(SC, SC_SUFFIX, SC_PREFIX, max) \
    DEFN_NVSHMEMX_OP_TILE_ALLREDUCE_THREADGROUP(SC, SC_SUFFIX, SC_PREFIX, min)

DEFN_NVSHMEM_TILE_ALLREDUCE_THREADGROUP(thread, , x);
DEFN_NVSHMEM_TILE_ALLREDUCE_THREADGROUP(warp, _warp, x);
DEFN_NVSHMEM_TILE_ALLREDUCE_THREADGROUP(warpgroup, _warpgroup, x);
DEFN_NVSHMEM_TILE_ALLREDUCE_THREADGROUP(block, _block, x);

#undef DEFN_NVSHMEMX_TYPENAME_OP_TILE_ALLREDUCE_THREADGROUP
#undef DEFN_NVSHMEM_TILE_ALLREDUCE_THREADGROUP

// Tile Reduce
#define DEFN_NVSHMEMX_OP_TILE_REDUCE_THREADGROUP(SC, SC_SUFFIX, SC_PREFIX, OP)                     \
    template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t,                      \
              nvshmemx::tile_coll_algo_t algo>                                                     \
    NVSHMEMI_STATIC NVSHMEMI_DEVICE_PREFIX int tile_##OP##_reduce##SC_SUFFIX(                      \
        nvshmem_team_t team, src_tensor_t src, dst_tensor_t dst, tuple_t start_coord,              \
        tuple_t boundary, int root, uint64_t flag) {                                               \
        nvshmemi_tile_reduce<algo, src_tensor_t, dst_tensor_t, tuple_t, nvshmemi_threadgroup_##SC, \
                             RDXN_OPS_##OP>(team, src, dst, start_coord, boundary, root, flag);    \
        return 0;                                                                                  \
    }

#define DEFN_NVSHMEM_TILE_REDUCE_THREADGROUP(SC, SC_SUFFIX, SC_PREFIX)      \
    DEFN_NVSHMEMX_OP_TILE_REDUCE_THREADGROUP(SC, SC_SUFFIX, SC_PREFIX, sum) \
    DEFN_NVSHMEMX_OP_TILE_REDUCE_THREADGROUP(SC, SC_SUFFIX, SC_PREFIX, max) \
    DEFN_NVSHMEMX_OP_TILE_REDUCE_THREADGROUP(SC, SC_SUFFIX, SC_PREFIX, min)

DEFN_NVSHMEM_TILE_REDUCE_THREADGROUP(thread, , x);
DEFN_NVSHMEM_TILE_REDUCE_THREADGROUP(warp, _warp, x);
DEFN_NVSHMEM_TILE_REDUCE_THREADGROUP(warpgroup, _warpgroup, x);
DEFN_NVSHMEM_TILE_REDUCE_THREADGROUP(block, _block, x);

#undef DEFN_NVSHMEMX_TYPENAME_OP_TILE_REDUCE_THREADGROUP
#undef DEFN_NVSHMEM_TILE_REDUCE_THREADGROUP

// Tile AllGather
#define DEFN_NVSHMEMX_TILE_ALLGATHER_THREADGROUP(SC, SC_SUFFIX, SC_PREFIX)                        \
    template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t,                     \
              nvshmemx::tile_coll_algo_t algo>                                                    \
    NVSHMEMI_STATIC NVSHMEMI_DEVICE_PREFIX int tile_allgather##SC_SUFFIX(                         \
        nvshmem_team_t team, src_tensor_t src, dst_tensor_t dst, tuple_t start_coord,             \
        tuple_t boundary, uint64_t flag) {                                                        \
        nvshmemi_tile_allgather<algo, src_tensor_t, dst_tensor_t, tuple_t,                        \
                                nvshmemi_threadgroup_##SC>(team, src, dst, start_coord, boundary, \
                                                           flag);                                 \
        return 0;                                                                                 \
    }

DEFN_NVSHMEMX_TILE_ALLGATHER_THREADGROUP(thread, , x);
DEFN_NVSHMEMX_TILE_ALLGATHER_THREADGROUP(warp, _warp, x);
DEFN_NVSHMEMX_TILE_ALLGATHER_THREADGROUP(warpgroup, _warpgroup, x);
DEFN_NVSHMEMX_TILE_ALLGATHER_THREADGROUP(block, _block, x);

#undef DEFN_NVSHMEMX_TILE_ALLGATHER_THREADGROUP

// Tile Collective Wait
#define DEFN_NVSHMEMX_TILE_COLLECTIVE_WAIT_THREADGROUP(SC, SC_SUFFIX, SC_PREFIX)             \
    template <nvshmemx::tile_coll_algo_t algo>                                               \
    NVSHMEMI_STATIC NVSHMEMI_DEVICE_PREFIX int tile_collective_wait##SC_SUFFIX(              \
        nvshmem_team_t team, uint64_t flag) {                                                \
        int status = 0;                                                                      \
        status = nvshmemi_tile_collective_wait<algo, nvshmemi_threadgroup_##SC>(team, flag); \
        return status;                                                                       \
    }

DEFN_NVSHMEMX_TILE_COLLECTIVE_WAIT_THREADGROUP(thread, , x);
DEFN_NVSHMEMX_TILE_COLLECTIVE_WAIT_THREADGROUP(warp, _warp, x);
DEFN_NVSHMEMX_TILE_COLLECTIVE_WAIT_THREADGROUP(warpgroup, _warpgroup, x);
DEFN_NVSHMEMX_TILE_COLLECTIVE_WAIT_THREADGROUP(block, _block, x);

#undef DEFN_NVSHMEMX_TILE_COLLECTIVE_WAIT_THREADGROUP

}  // namespace nvshmemx
#endif  // __CUDA_ARCH__

#endif
