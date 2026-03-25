/*
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

#ifndef _NVSHMEMI_TEAM_INTERNAL_H_
#define _NVSHMEMI_TEAM_INTERNAL_H_

#include <stddef.h>
#include <cuda_runtime.h>
#include "device_host/nvshmem_common.cuh"

#if !defined __CUDACC_RTC__
#include <limits.h>
#else
#include <cuda/std/climits>
#endif

enum nvshmemi_team_creation_pe_state {
    NVSHMEMI_TEAM_CREATION_PE_STATE_PREINIT = 0x00000000,
    NVSHMEMI_TEAM_CREATION_PE_STATE_READ_PE_IN_TEAM = 0x00000001,
    NVSHMEMI_TEAM_CREATION_PE_STATE_WROTE_INDEX = 0x00000002,
    NVSHMEMI_TEAM_CREATION_PE_STATE_DONE = 0x00000004,
};

struct nvshmemi_team_creation_pe_info {
    int pe_in_team;
    int state_idx;
    unsigned char *team_index_array;
};
typedef struct nvshmemi_team_creation_pe_info nvshmemi_team_creation_pe_info_t;

__host__ __device__ inline size_t nvshmemi_bit_1st_nonzero(const unsigned char *ptr,
                                                           const size_t size) {
    /* The following ignores endianess: */
    for (size_t i = 0; i < size; i++) {
        unsigned char bit_val = ptr[i];
        for (size_t j = 0; bit_val && j < CHAR_BIT; j++) {
            if (bit_val & 1) return i * CHAR_BIT + j;
            bit_val >>= 1;
        }
    }

    return (size_t)-1;
}

struct nvshmemi_team_creation_psync {
    uint64_t uniqueid;
    nvshmemi_team_creation_pe_info_t pe_info[];
};
typedef struct nvshmemi_team_creation_psync nvshmemi_team_creation_psync_t;

template <typename T>
void nvshmemi_call_init_array_kernel(T *array, int len, T val);

template <typename TYPE, rdxn_ops_t OP>
void nvshmemi_call_reduce_kernel(int start, int stride, int size, TYPE *dst, const TYPE *source,
                                 size_t nreduce, TYPE *pWrk, volatile long *pSync,
                                 volatile long *sync_counter);

typedef nvshmemx_team_uniqueid_t nvshmemi_team_uniqueid_t;

void nvshmemi_call_team_mapping_kernel(
    uint64_t uniqueid, int npes, int *pe_mapping,
    nvshmemi_team_creation_psync_t *nvshmemi_team_creation_psync);

void nvshmemi_call_team_index_kernel(nvshmemi_team_t *myteam,
                                     nvshmemi_team_creation_psync_t *nvshmemi_team_creation_psync,
                                     long N_PSYNC_BYTES);

#endif
