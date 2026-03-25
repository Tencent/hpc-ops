/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <map>
#include <string>
#include <typeinfo>
#include "internal/host/util.h"
#include "internal/non_abi/nvshmemi_h_to_d_coll_defs.cuh"
#include "host/nvshmem_api.h"

#define NVSHMEMI_FCOLLECT_CTA_THRESHOLD 1048576
#define NVSHMEMI_FCOLLECT_CTA_COUNT_DEFAULT 32 /* # of GPUs x 4 CTA per GPU for DGX */

std::map<std::string, size_t> nvshmemi_fcollect_maxblocksize;

template <typename TYPE>
void nvshmemi_call_fcollect_on_stream_kernel(nvshmem_team_t team, TYPE *dest, const TYPE *source,
                                             size_t nelems, cudaStream_t stream) {
    int tmp;
    std::string type_str(typeid(TYPE).name());
    int in_cuda_graph = 0;

    if (nvshmemi_fcollect_maxblocksize.find(type_str) == nvshmemi_fcollect_maxblocksize.end()) {
        CUDA_RUNTIME_CHECK(cudaOccupancyMaxPotentialBlockSize(
            &tmp, (int *)&nvshmemi_fcollect_maxblocksize[type_str],
            fcollect_on_stream_kernel<TYPE>));
    }

    cudaStreamCaptureStatus status;
    CUDA_RUNTIME_CHECK(cudaStreamIsCapturing(stream, &status));
    if (status == cudaStreamCaptureStatusActive) in_cuda_graph = 1;

    /* By default select min(occupancy, nelems) */
    int num_threads_per_block = nvshmemi_fcollect_maxblocksize[type_str];

    /* Use env to override the value */
    if (nvshmemi_options.FCOLLECT_NTHREADS_provided) {
        num_threads_per_block = nvshmemi_options.FCOLLECT_NTHREADS;
    }

    nvshmemi_team_t *teami = nvshmemi_team_pool[team];
    int num_blocks = 1;

    /* By default for NVLS sharp based algorithms, Select num of blocks by size heuristic */
    if (teami->nvls_rsc_base_ptr != NULL) {
        if (nvshmemi_options.MAX_CTAS_provided &&
            nelems * teami->size * sizeof(TYPE) >= NVSHMEMI_FCOLLECT_CTA_THRESHOLD) {
            num_blocks = nvshmemi_options.MAX_CTAS;
        } else if (nelems * teami->size * sizeof(TYPE) >= NVSHMEMI_FCOLLECT_CTA_THRESHOLD) {
            num_blocks = ((NVSHMEMI_FCOLLECT_CTA_COUNT_DEFAULT / teami->size) > 1
                              ? (NVSHMEMI_FCOLLECT_CTA_COUNT_DEFAULT / teami->size)
                              : 1);
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
            if (teami->team_dups[block_id + 1] == NVSHMEM_TEAM_INVALID) {
                NVSHMEMI_ERROR_EXIT(
                    "Unable to allocate enough teams for fcollect. This will cause significant "
                    "performance degradation. Please increase NVSHMEM_MAX_TEAMS. Exiting\n");
            }
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

    fcollect_on_stream_kernel<TYPE><<<num_blocks, num_threads_per_block, 0, stream>>>(
        team, dest, source, nelems, in_cuda_graph);
    CUDA_RUNTIME_CHECK(cudaGetLastError());
}

#define INSTANTIATE_NVSHMEMI_CALL_FCOLLECT_ON_STREAM_KERNEL(TYPE) \
    template void nvshmemi_call_fcollect_on_stream_kernel<TYPE>(  \
        nvshmem_team_t, TYPE *, const TYPE *, size_t, cudaStream_t);
INSTANTIATE_NVSHMEMI_CALL_FCOLLECT_ON_STREAM_KERNEL(uint8_t)
INSTANTIATE_NVSHMEMI_CALL_FCOLLECT_ON_STREAM_KERNEL(uint16_t)
INSTANTIATE_NVSHMEMI_CALL_FCOLLECT_ON_STREAM_KERNEL(uint32_t)
INSTANTIATE_NVSHMEMI_CALL_FCOLLECT_ON_STREAM_KERNEL(uint64_t)
INSTANTIATE_NVSHMEMI_CALL_FCOLLECT_ON_STREAM_KERNEL(int8_t)
INSTANTIATE_NVSHMEMI_CALL_FCOLLECT_ON_STREAM_KERNEL(int16_t)
INSTANTIATE_NVSHMEMI_CALL_FCOLLECT_ON_STREAM_KERNEL(int32_t)
INSTANTIATE_NVSHMEMI_CALL_FCOLLECT_ON_STREAM_KERNEL(int64_t)
INSTANTIATE_NVSHMEMI_CALL_FCOLLECT_ON_STREAM_KERNEL(half)
INSTANTIATE_NVSHMEMI_CALL_FCOLLECT_ON_STREAM_KERNEL(__nv_bfloat16)
INSTANTIATE_NVSHMEMI_CALL_FCOLLECT_ON_STREAM_KERNEL(float)
INSTANTIATE_NVSHMEMI_CALL_FCOLLECT_ON_STREAM_KERNEL(char)
INSTANTIATE_NVSHMEMI_CALL_FCOLLECT_ON_STREAM_KERNEL(double)
INSTANTIATE_NVSHMEMI_CALL_FCOLLECT_ON_STREAM_KERNEL(long long)
INSTANTIATE_NVSHMEMI_CALL_FCOLLECT_ON_STREAM_KERNEL(unsigned long long)
