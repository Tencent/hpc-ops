/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */
#ifndef __REDUCE_COMMON_CUH__
#define __REDUCE_COMMON_CUH__
#include <cuda.h>
#include <cuda_runtime.h>
#include <map>
#include <string>
#include <typeinfo>

#include "internal/host/util.h"
#include "internal/non_abi/nvshmemi_h_to_d_coll_defs.cuh"
#include "host/nvshmem_api.h"

/* NVL4: Threshould is modeled using the following: L + o + g*M where, L is the half of round-trip
 * time, o is GPU processing overhead, g is the transmit time on NVLINK egress and M is # of
 * messages in bytes so threshold for multi-CTA = (1/g)*(25GB/s). Based on practical observations,
 * 0.25MB is where multi-CTA proves to be beneficial in accelerating AR.
 */
#define NVSHMEMI_REDUCE_CTA_THRESHOLD 262144
#define NVSHMEMI_REDUCE_CTA_COUNT_DEFAULT 16

extern std::map<std::string, size_t> nvshmemi_broadcast_maxblocksize;
static std::map<std::pair<std::string, rdxn_ops_t>, size_t> nvshmemi_reduce_maxblocksize;

template <typename TYPE, rdxn_ops_t OP>
void nvshmemi_call_rdxn_on_stream_kernel(nvshmem_team_t team, TYPE *dest, const TYPE *source,
                                         size_t nreduce, cudaStream_t stream) {
    int tmp;
    int in_cuda_graph = 0;
    std::pair<std::string, rdxn_ops_t> map_pair(std::string(typeid(TYPE).name()), OP);
    if (nvshmemi_reduce_maxblocksize.find(map_pair) == nvshmemi_reduce_maxblocksize.end()) {
        CUDA_RUNTIME_CHECK(cudaOccupancyMaxPotentialBlockSize(
            &tmp, (int *)&nvshmemi_reduce_maxblocksize[map_pair], rdxn_on_stream_kernel<TYPE, OP>));
    }
    size_t num_threads_per_block = nvshmemi_reduce_maxblocksize[map_pair];

    /* Use env to override the value */
    if (nvshmemi_options.REDUCE_NTHREADS_provided) {
        num_threads_per_block = nvshmemi_options.REDUCE_NTHREADS;
    }

    cudaStreamCaptureStatus status;
    CUDA_RUNTIME_CHECK(cudaStreamIsCapturing(stream, &status));
    if (status == cudaStreamCaptureStatusActive) in_cuda_graph = 1;

    nvshmemi_team_t *teami = nvshmemi_team_pool[team];
    int num_blocks = 1;

    /* By default for NVLS sharp based algorithms, select num of blocks by size heuristic */
    if (teami->nvls_rsc_base_ptr != NULL) {
        if (nvshmemi_options.MAX_CTAS_provided &&
            nreduce * sizeof(TYPE) >= NVSHMEMI_REDUCE_CTA_THRESHOLD) {
            num_blocks = nvshmemi_options.MAX_CTAS;
        } else if (nreduce * sizeof(TYPE) >= NVSHMEMI_REDUCE_CTA_THRESHOLD) {
            num_blocks = NVSHMEMI_REDUCE_CTA_COUNT_DEFAULT;
        } else {
            num_blocks = 1;
        }
    }

    if (num_blocks > 1 && teami->team_dups[1] == NVSHMEM_TEAM_INVALID) {
        CUDA_RUNTIME_CHECK(cudaStreamSynchronize(
            stream)); /* This is to synchronize with any prior operations submitted on this stream
                         such as barrier that can deadlock with similar ops in split_strided */
        NVSHMEMU_FOR_EACH(block_id, num_blocks - 1) {
            nvshmem_team_split_strided(team, 0, 1, nvshmem_team_n_pes(team), NULL, 0,
                                       &(teami->team_dups[block_id + 1]));
            INFO(NVSHMEM_TEAM, "Duplicate team ID: %d of parent team: %d for CTA %zu\n",
                 teami->team_dups[block_id + 1], teami->team_idx, block_id);
        }

        off_t team_dups_offset = offsetof(nvshmemi_team_t, team_dups);
        nvshmemi_team_t *teami_pool_device_addr;
        CUDA_RUNTIME_CHECK(cudaMemcpy((void **)&teami_pool_device_addr,
                                      &nvshmemi_device_state.team_pool[team],
                                      sizeof(nvshmemi_team_t *), cudaMemcpyDeviceToHost));
        CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
        off_t team_dups_device_addr = (off_t)((char *)teami_pool_device_addr + team_dups_offset);
        CUDA_RUNTIME_CHECK(cudaMemcpy((void *)(team_dups_device_addr), &teami->team_dups[0],
                                      sizeof(nvshmem_team_t) * num_blocks, cudaMemcpyHostToDevice));
        CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
    }

    rdxn_on_stream_kernel<TYPE, OP><<<num_blocks, num_threads_per_block, 0, stream>>>(
        team, dest, source, nreduce, in_cuda_graph);
    CUDA_RUNTIME_CHECK(cudaGetLastError());
}

#define INSTANTIATE_NVSHMEMI_CALL_RDXN_ON_STREAM_KERNEL(TYPE, OP) \
    template void nvshmemi_call_rdxn_on_stream_kernel<TYPE, OP>(  \
        nvshmem_team_t, TYPE *, const TYPE *, size_t, cudaStream_t);

#define REPT_FOR_BITWISE_TYPES(FN, OP) \
    FN(int8_t, RDXN_OPS_##OP)          \
    FN(uint8_t, RDXN_OPS_##OP)         \
    FN(uint16_t, RDXN_OPS_##OP)        \
    FN(int16_t, RDXN_OPS_##OP)         \
    FN(uint32_t, RDXN_OPS_##OP)        \
    FN(int32_t, RDXN_OPS_##OP)         \
    FN(uint64_t, RDXN_OPS_##OP)        \
    FN(int64_t, RDXN_OPS_##OP)         \
    FN(char, RDXN_OPS_##OP)            \
    FN(long long, RDXN_OPS_##OP)       \
    FN(unsigned long long, RDXN_OPS_##OP)

#define REPT_FOR_FLOATING_TYPES(FN, OP) \
    FN(half, RDXN_OPS_##OP)             \
    FN(__nv_bfloat16, RDXN_OPS_##OP)    \
    FN(float, RDXN_OPS_##OP)            \
    FN(double, RDXN_OPS_##OP)

#endif /* __REDUCE_COMMON_CUH__ */
