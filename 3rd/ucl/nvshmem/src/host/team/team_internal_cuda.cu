/*
 * Copyright (c) 2016-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

#include <cuda.h>
#include <cuda_runtime.h>

#include "device_host/nvshmem_common.cuh"
#include "non_abi/device/coll/utils.cuh"
#include "non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh"
#include "non_abi/device/common/nvshmemi_common_device.cuh"
#include "team_internal.h"
#include "internal/host/util.h"

template <typename TYPE, rdxn_ops_t OP>
extern __global__ void nvshmemi_reduce_kernel(int start, int stride, int size, TYPE *dst,
                                              const TYPE *source, size_t nreduce, TYPE *pWrk,
                                              volatile long *pSync, volatile long *sync_counter);

template <typename T>
__global__ void nvshmemi_init_array_kernel(T *array, int len, T val) {
    for (int i = 0; i < len; i++) array[i] = val;
}

template <typename T>
void nvshmemi_call_init_array_kernel(T *array, int len, T val) {
    nvshmemi_init_array_kernel<T><<<1, 1>>>(array, len, val);
    CUDA_RUNTIME_CHECK(cudaGetLastError());
    CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
}

template void nvshmemi_call_init_array_kernel<nvshmemi_team_t *>(nvshmemi_team_t **, int,
                                                                 nvshmemi_team_t *);
template void nvshmemi_call_init_array_kernel<long>(long *, int, long);
template void nvshmemi_call_init_array_kernel<uint64_t>(uint64_t *, int, uint64_t);
template void nvshmemi_call_init_array_kernel<nvshmemi_team_creation_pe_info_t>(
    nvshmemi_team_creation_pe_info_t *, int, nvshmemi_team_creation_pe_info_t);
template void nvshmemi_call_init_array_kernel<unsigned char>(unsigned char *, int, unsigned char);

template <typename TYPE, rdxn_ops_t OP>
void nvshmemi_call_reduce_kernel(int start, int stride, int size, TYPE *dst, const TYPE *source,
                                 size_t nreduce, TYPE *pWrk, volatile long *pSync,
                                 volatile long *sync_counter) {
    nvshmemi_reduce_kernel<TYPE, OP>
        <<<1, 1>>>(start, stride, size, dst, source, nreduce, pWrk, pSync, sync_counter);
    CUDA_RUNTIME_CHECK(cudaGetLastError());
    CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
}

template void nvshmemi_call_reduce_kernel<unsigned char, (rdxn_ops)0>(
    int, int, int, unsigned char *, unsigned char const *, unsigned long, unsigned char *,
    long volatile *, long volatile *);

template void nvshmemi_call_reduce_kernel<int, (rdxn_ops)4>(int, int, int, int *, int const *,
                                                            unsigned long, int *, long volatile *,
                                                            long volatile *);

#ifdef __CUDA_ARCH__
__device__ void nvshmemi_team_creation_state_barrier(
    nvshmemi_team_t *myteam, volatile nvshmemi_team_creation_pe_info_t *pe_info,
    nvshmemi_team_creation_pe_state state) {
    int num_peers_complete = 0;
    int mype = nvshmemi_device_state_d.mype;

    nvshmemi_quiet<NVSHMEMI_THREADGROUP_THREAD>();
    pe_info[mype].state_idx |= state;
    for (int i = 0; i < myteam->size; i++) {
        int remote_pe = myteam->pe_mapping[i];
        if (remote_pe == mype) {
            continue;
        }
        nvshmemi_signal_for_barrier<int>((int *)&pe_info[mype].state_idx, pe_info[mype].state_idx,
                                         remote_pe);
    }

    do {
        for (int i = 0; i < myteam->size; i++) {
            int global_pe_index = myteam->pe_mapping[i];
            do {
                if (pe_info[global_pe_index].state_idx & state) {
                    num_peers_complete++;
                }
            } while (!(pe_info[global_pe_index].state_idx & state));
        }
    } while (num_peers_complete < myteam->size);

    if (nvshmemi_device_state_d.job_connectivity > NVSHMEMI_JOB_GPU_PROXY)
        nvshmemi_transfer_enforce_consistency_at_target(false);
}
#endif
template <threadgroup_t SCOPE>
__global__ void nvshmemi_team_index_kernel(
    nvshmemi_team_t *myteam, nvshmemi_team_creation_psync_t *nvshmemi_team_creation_psync,
    unsigned char *team_index_array, long N_PSYNC_BYTES) {
#ifdef __CUDA_ARCH__
    unsigned char *local_ptr, *peer_ptr;

    local_ptr =
        nvshmemi_team_creation_psync->pe_info[nvshmemi_device_state_d.mype].team_index_array;

    if (nvshmemi_thread_id_in_threadgroup<SCOPE>() == 0) {
        nvshmemi_team_creation_state_barrier(myteam, nvshmemi_team_creation_psync->pe_info,
                                             NVSHMEMI_TEAM_CREATION_PE_STATE_READ_PE_IN_TEAM);

        for (int i = 0; i < N_PSYNC_BYTES; i++) {
            team_index_array[i] = local_ptr[i];
        }

        for (int i = 0; i < myteam->size; i++) {
            int global_pe_index = myteam->pe_mapping[i];
            if (global_pe_index == nvshmemi_device_state_d.mype) {
                continue;
            }
            nvshmemi_put_nbi_threadgroup<unsigned char, NVSHMEMI_THREADGROUP_THREAD>(
                local_ptr, local_ptr, N_PSYNC_BYTES, global_pe_index);
        }

        nvshmemi_team_creation_state_barrier(myteam, nvshmemi_team_creation_psync->pe_info,
                                             NVSHMEMI_TEAM_CREATION_PE_STATE_WROTE_INDEX);

        for (int i = 0; i < myteam->size; i++) {
            int global_pe_index = myteam->pe_mapping[i];
            peer_ptr = nvshmemi_team_creation_psync->pe_info[global_pe_index].team_index_array;
            for (int j = 0; j < N_PSYNC_BYTES; j++) {
                team_index_array[j] &= peer_ptr[j];
            }
        }

        nvshmemi_team_creation_state_barrier(myteam, nvshmemi_team_creation_psync->pe_info,
                                             NVSHMEMI_TEAM_CREATION_PE_STATE_DONE);

        myteam->team_idx = nvshmemi_bit_1st_nonzero(team_index_array, N_PSYNC_BYTES);
    }

#endif
}

template <threadgroup_t SCOPE>
__global__ void nvshmemi_team_mapping_kernel(
    uint64_t uniqueid, int npes, int *num_peers_found, int *pe_mapping,
    nvshmemi_team_creation_psync_t *nvshmemi_team_creation_psync) {
#ifdef __CUDA_ARCH__
    int mype_in_team =
        nvshmemi_team_creation_psync->pe_info[nvshmemi_device_state_d.mype].pe_in_team;
    int nThreads = nvshmemi_threadgroup_size<SCOPE>();
    int myThread = nvshmemi_thread_id_in_threadgroup<SCOPE>();

    if (myThread == 0) {
        *num_peers_found = 1;
    }

    __syncthreads();

    /* write my psync and pe indexes to all other teams. */
    for (int i = myThread; i < nvshmemi_device_state_d.npes; i += nThreads) {
        if (i == nvshmemi_device_state_d.mype) {
            pe_mapping[mype_in_team] = nvshmemi_device_state_d.mype;
            pe_mapping[npes + nvshmemi_device_state_d.mype] = mype_in_team;
            continue;
        }
        nvshmemi_put_nbi_threadgroup<int, NVSHMEMI_THREADGROUP_THREAD>(
            (int *)&nvshmemi_team_creation_psync->pe_info[nvshmemi_device_state_d.mype].pe_in_team,
            (int *)&nvshmemi_team_creation_psync->pe_info[nvshmemi_device_state_d.mype].pe_in_team,
            1, i);
    }
    nvshmemi_quiet<SCOPE>();
    if (myThread == 0) {
        nvshmemi_team_creation_psync->uniqueid = uniqueid;
        __threadfence_system();
    }
    nvshmemi_threadgroup_sync<SCOPE>();

    /* wait for all other teams to write their psync and pe indexes to me. */
    do {
        for (int i = myThread; i < nvshmemi_device_state_d.npes; i += nThreads) {
            if (i == nvshmemi_device_state_d.mype) {
                continue;
            }

            uint64_t peer_uniqueid =
                nvshmemi_g<uint64_t>(&nvshmemi_team_creation_psync->uniqueid, i);

            if (peer_uniqueid == uniqueid) {
                nvshmemi_transfer_syncapi_update_mem();
                int peer_index_in_team =
                    *(volatile int *)&nvshmemi_team_creation_psync->pe_info[i].pe_in_team;
                if (pe_mapping[npes + i] == -1) {
                    pe_mapping[peer_index_in_team] = i;
                    pe_mapping[npes + i] = peer_index_in_team;
                    atomicAdd(num_peers_found, 1);
                }
            }
        }
        if (myThread == 0) {
            __threadfence_system();
        }
        nvshmemi_threadgroup_sync<SCOPE>();
    } while (*num_peers_found < npes);
#endif
}

void nvshmemi_call_team_mapping_kernel(
    uint64_t uniqueid, int npes, int *pe_mapping,
    nvshmemi_team_creation_psync_t *nvshmemi_team_creation_psync) {
    int *num_peers_found;
    CUDA_RUNTIME_CHECK(cudaMalloc(&num_peers_found, sizeof(int)));
    nvshmemi_team_mapping_kernel<NVSHMEMI_THREADGROUP_WARP>
        <<<1, 32>>>(uniqueid, npes, num_peers_found, pe_mapping, nvshmemi_team_creation_psync);
    CUDA_RUNTIME_CHECK(cudaGetLastError());
    CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
    CUDA_RUNTIME_CHECK(cudaFree(num_peers_found));
}

void nvshmemi_call_team_index_kernel(nvshmemi_team_t *myteam,
                                     nvshmemi_team_creation_psync_t *nvshmemi_team_creation_psync,
                                     long N_PSYNC_BYTES) {
    unsigned char *team_index_array;

    INFO(NVSHMEM_COLL, "in team index kernel mype: %d\n", nvshmemi_state->mype);
    CUDA_RUNTIME_CHECK(cudaMalloc(&team_index_array, N_PSYNC_BYTES));
    nvshmemi_team_index_kernel<NVSHMEMI_THREADGROUP_THREAD>
        <<<1, 1>>>(myteam, nvshmemi_team_creation_psync, team_index_array, N_PSYNC_BYTES);
    CUDA_RUNTIME_CHECK(cudaGetLastError());
    CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
    CUDA_RUNTIME_CHECK(cudaFree(team_index_array));
}
