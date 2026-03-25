/*
 * Copyright (c) 2016-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

#include "team_internal.h"
#include <assert.h>        // for assert
#include <cuda_runtime.h>  // for cudaMemcpy
#include <driver_types.h>  // for cudaMemcpyHos...
#include <limits.h>        // for CHAR_BIT
#include <random>
#include <stdint.h>                                      // for uint64_t
#include <stdio.h>                                       // for snprintf, printf
#include <stdlib.h>                                      // for free, malloc
#include <string.h>                                      // for memset, memcmp
#include <cmath>                                         // for ceil
#include "../coll/rdxn/rdxn.h"                           // for nvshmemi_call...
#include "device_host/nvshmem_types.h"                   // for nvshmemi_team_t
#include "cpu_coll.h"                                    // for nccl_ftable
#include "bootstrap_device_host/nvshmem_uniqueid.h"      // for nvshmemx_team_uniqueid_t
#include "device_host/nvshmem_common.cuh"                // for nvshmemi_pe_i...
#include "bootstrap_host_transport/env_defs_internal.h"  // for nvshmemi_opti...
#include "host/nvshmem_api.h"                            // for nvshmem_quiet
#include "host/nvshmem_coll_api.h"                       // for nvshmem_team_...
#include "host/nvshmemx_api.h"                           // for nvshmemx_char...
#include "non_abi/nvshmemx_error.h"                      // for NVSHMEMI_NULL...
#include "internal/host/debug.h"                         // for INFO, NVSHMEM...
#include "internal/host/nvshmem_internal.h"              // for nvshmemi_free
#include "internal/host/nvshmemi_coll.h"                 // for nvshmemi_barrier
#include "internal/host/nvshmemi_symmetric_heap.hpp"     // for nvshmemi_symm...
#include "internal/host/nvshmemi_team.h"                 // for N_PSYNC_BYTES
#include "internal/host/nvshmemi_types.h"                // for nvshmemi_state
#include "internal/host/util.h"                          // for CUDA_RUNTIME_...
#include "internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h"  // for nvshmemi_boot...
#include "internal/host_transport/transport.h"                             // for nvshmem_trans...
#include "non_abi/nvshmem_build_options.h"                                 // for NVSHMEM_USE_NCCL
#include "internal/host/nvshmemi_nvls_rsc.hpp"                             // for nvshmemi_nvls...
#ifdef NVSHMEM_USE_NCCL
#include "nccl.h"  // for ncclUniqueId
#endif
#include "internal/host_transport/nvshmemi_transport_defines.h"  // for NVSHMEM_MEM_H...

using namespace nvls;

#define NVSHMEMI_DIAG_STRLEN 1024
#define NVSHMEMI_SYNC_VALUE 0
#define NVSHMEMI_REDUCE_MAX_CTA_COUNT 64
#define NVSHMEMI_DEVICE_TEAM_PE_LOCATION(team) ((int *)(team + 1))

/* 0th entry in team duplicate resources is same team as the encapsulating team. This allows
 * for reuse of the same business logic for nCTA == 1 and nCTA > 1 and minimizes if/else
 */
#define NVSHMEMI_TEAM_DUP_INITIALIZER(teami, team_idx) \
    (teami)->team_dups[0] = (team_idx);                \
    for (int i = 1; i < 128; i++) {                    \
        (teami)->team_dups[i] = NVSHMEM_TEAM_INVALID;  \
    }

#define NVSHMEMI_TEAM_PE_MAPPING_INITIALIZER(team, team_npes)     \
    for (int i = 0; i < nvshmemi_state->npes; i++) {              \
        (team)->pe_mapping[i + team_npes] = NVSHMEM_TEAM_INVALID; \
    }                                                             \
    for (int i = 0; i < team_npes; i++) {                         \
        (team)->pe_mapping[i] = NVSHMEM_TEAM_INVALID;             \
    }

static long *nvshmemi_team_get_sync_counter(nvshmemi_team_t *team);

long nvshmemi_max_teams;
static long N_PSYNC_BYTES = 32; /* N_PSYNC_BYTES * CHAR_NBIT == max_teams supported */

nvshmemi_team_t *nvshmemi_team_world;
nvshmemi_team_t *nvshmemi_team_shared;
nvshmemi_team_t *nvshmemi_team_node;
nvshmemi_team_t *nvshmemi_team_same_mype_node;
nvshmemi_team_t *nvshmemi_team_same_gpu;
nvshmemi_team_t *nvshmemi_team_gpu_leaders;

nvshmemi_team_t *nvshmemi_device_team_world, *nvshmemi_device_team_shared,
    *nvshmemi_device_team_node, *nvshmemi_device_team_same_mype_node,
    *nvshmemi_device_team_same_gpu, *nvshmemi_device_team_gpu_leaders;

nvshmemi_team_t **nvshmemi_team_pool;
long *nvshmemi_psync_pool;
long *nvshmemi_sync_counter;

nvshmemi_team_t **nvshmemi_device_team_pool;

static unsigned char *psync_pool_avail;
static unsigned char *psync_pool_avail_reduced;
static unsigned char *device_psync_pool_avail;
static unsigned char *device_psync_pool_avail_reduced;

static int *team_ret_val;
static int *team_ret_val_reduced;
static int *device_team_ret_val;
static int *device_team_ret_val_reduced;

nvshmemi_team_creation_psync_t *nvshmemi_team_creation_psync = NULL;

static nvshmemi_team_uniqueid_t nvshmemi_team_populate_uniqueid(void) {
    /* 64-bit uniqueid */
    nvshmemi_team_uniqueid_t team_uniqueid = TEAM_ULSCALAR_INVALID;
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dis(0, UINT64_MAX);
    do {
        team_uniqueid = dis(gen);
    } while (team_uniqueid == TEAM_ULSCALAR_INVALID);
    return team_uniqueid;
}

static bool nvshmemi_team_is_identical(nvshmemi_team_t *t1, nvshmemi_team_t *t2) {
    if (t1 == NULL || t2 == NULL) {
        return false;
    }

    if (t1->start != t2->start || t1->size != t2->size) {
        return false;
    }

    /* shortcut for teams with linear stride */
    if (t1->stride > 0 && t2->stride == t1->stride) {
        return true;
    }

    for (int i = 0; i < t1->size; i++) {
        if (t1->pe_mapping[i] != t2->pe_mapping[i]) {
            return false;
        }
    }

    return true;
}

static bool nvshmemi_team_is_subset(nvshmemi_team_t *subset, nvshmemi_team_t *superset) {
    int subset_pe_global_idx = -1;
    int superset_pe_global_idx = -1;
    bool is_subset = true;
    int i, j;

    if (subset->size > superset->size) {
        return false;
    }

    for (i = 0; i < subset->size; i++) {
        subset_pe_global_idx = subset->pe_mapping[i];
        for (j = 0; j < superset->size; j++) {
            superset_pe_global_idx = superset->pe_mapping[j];
            if (subset_pe_global_idx == superset_pe_global_idx) {
                break;
            }
        }
        if (j == superset->size) {
            is_subset = false;
            break;
        }
    }

    return is_subset;
}

static size_t nvshmemi_team_get_total_team_size(size_t npes) {
    return (sizeof(nvshmemi_team_t) + (npes + nvshmemi_state->npes) * sizeof(int));
}

static void copy_team_pe_mapping_to_host(nvshmemi_team_t *device_team, nvshmemi_team_t *host_team) {
    int *device_pe_mapping = (int *)(device_team + 1);
    CUDA_RUNTIME_CHECK(cudaMemcpy(host_team->pe_mapping, device_pe_mapping,
                                  sizeof(int) * host_team->size, cudaMemcpyDeviceToHost));
    CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
}

static void nvshmemi_team_copy_pe_mapping(nvshmemi_team_t *host_team, nvshmemi_team_t *device_team,
                                          int npes) {
    int *device_pe_mapping = (int *)(device_team + 1);
    CUDA_RUNTIME_CHECK(cudaMemcpy(device_pe_mapping, host_team->pe_mapping,
                                  sizeof(int) * (npes + nvshmemi_state->npes),
                                  cudaMemcpyHostToDevice));
    CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
}

static int nvshmemi_team_allocate_team(nvshmemi_team_t **host_ptr, nvshmemi_team_t **device_ptr,
                                       int npes) {
    size_t total_team_size = nvshmemi_team_get_total_team_size(npes);
    CUDA_RUNTIME_CHECK(cudaMalloc((void **)device_ptr, total_team_size));

    (*host_ptr) = (nvshmemi_team_t *)malloc(total_team_size);
    if (host_ptr == NULL) {
        return NVSHMEMX_ERROR_OUT_OF_MEMORY;
    }

    **host_ptr = NVSHMEMI_TEAM_INITIALIZER;

    // Set the pe_mapping pointer to point to the memory right after the struct
    (*host_ptr)->pe_mapping = (int *)((*host_ptr) + 1);

    NVSHMEMI_TEAM_PE_MAPPING_INITIALIZER(*host_ptr, npes);

    nvshmemi_team_copy_pe_mapping(*host_ptr, *device_ptr, npes);

    return NVSHMEMX_SUCCESS;
}

static unsigned char *nvshmemi_get_pe_info_array_ptr(nvshmemi_team_creation_psync_t *psync,
                                                     int pe) {
    return (unsigned char *)((char *)psync + sizeof(nvshmemi_team_creation_psync_t) +
                             sizeof(nvshmemi_team_creation_pe_info_t) * nvshmemi_state->npes +
                             N_PSYNC_BYTES * pe);
}
static void nvshmemi_reset_team_creation_psync() {
    unsigned char *team_index_array_start;
    nvshmemi_team_creation_pe_info_t info;

    info.state_idx = NVSHMEMI_TEAM_CREATION_PE_STATE_PREINIT;
    info.pe_in_team = -1;

    team_index_array_start = nvshmemi_get_pe_info_array_ptr(nvshmemi_team_creation_psync, 0);

    nvshmemi_call_init_array_kernel<nvshmemi_team_uniqueid_t>(
        &nvshmemi_team_creation_psync->uniqueid, 1, TEAM_ULSCALAR_INVALID);

    for (int i = 0; i < nvshmemi_state->npes; i++) {
        info.team_index_array = nvshmemi_get_pe_info_array_ptr(nvshmemi_team_creation_psync, i);
        nvshmemi_call_init_array_kernel<nvshmemi_team_creation_pe_info_t>(
            (nvshmemi_team_creation_pe_info_t *)&nvshmemi_team_creation_psync->pe_info[i], 1, info);
    }
    nvshmemi_call_init_array_kernel<unsigned char>(team_index_array_start,
                                                   N_PSYNC_BYTES * nvshmemi_state->npes, 0x00);
}

static int nvshmemi_init_team_creation_psync() {
    int status = NVSHMEMX_SUCCESS;

    if (nvshmemi_team_creation_psync == NULL) {
        nvshmemi_team_creation_psync = (nvshmemi_team_creation_psync_t *)nvshmemi_malloc(
            sizeof(nvshmemi_team_creation_psync_t) +
            sizeof(nvshmemi_team_creation_pe_info_t) * nvshmemi_state->npes +
            N_PSYNC_BYTES * nvshmemi_state->npes);
        NVSHMEMI_NULL_ERROR_JMP(nvshmemi_team_creation_psync, status, NVSHMEMX_ERROR_OUT_OF_MEMORY,
                                out, "nvshmemi_team_creation_psync is not allocated \n");
    }
    nvshmemi_reset_team_creation_psync();
out:
    return status;
}

static void copy_team_to_device(nvshmemi_team_t *host_team, nvshmemi_team_t *device_team) {
    // Copy the main struct first
    CUDA_RUNTIME_CHECK(
        cudaMemcpy(device_team, host_team, sizeof(nvshmemi_team_t), cudaMemcpyHostToDevice));

    // Calculate the address for the device pe_mapping and set it
    int *device_pe_mapping = (int *)(device_team + 1);
    CUDA_RUNTIME_CHECK(cudaMemcpy(&(device_team->pe_mapping), &device_pe_mapping, sizeof(int *),
                                  cudaMemcpyHostToDevice));

    // Copy the pe_mapping array
    nvshmemi_team_copy_pe_mapping(host_team, device_team, host_team->size);

    // Update the device team pointer pool
    if (host_team->team_idx >= 0 && host_team->team_idx < nvshmemi_max_teams) {
        CUDA_RUNTIME_CHECK(cudaMemcpy(&nvshmemi_device_team_pool[host_team->team_idx], &device_team,
                                      sizeof(nvshmemi_team_t *), cudaMemcpyHostToDevice));
    }

    CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
}

static void nvshmemi_team_update_device(void) {
    copy_team_to_device(nvshmemi_team_world, nvshmemi_device_team_world);

    copy_team_to_device(nvshmemi_team_shared, nvshmemi_device_team_shared);

    copy_team_to_device(nvshmemi_team_node, nvshmemi_device_team_node);

    copy_team_to_device(nvshmemi_team_same_mype_node, nvshmemi_device_team_same_mype_node);

    copy_team_to_device(nvshmemi_team_same_gpu, nvshmemi_device_team_same_gpu);

    if (nvshmemi_team_gpu_leaders != NULL) {
        copy_team_to_device(nvshmemi_team_gpu_leaders, nvshmemi_device_team_gpu_leaders);
    }
}

static void nvshmemi_recexchalgo_get_neighbors(nvshmemi_team_t *teami) {
    int i, j, k;
    int p_of_k = 1, log_p_of_k = 0, rem, T, newpe;
    int step1_sendto, step1_nrecvs, step2_nphases;
    int *step1_recvfrom, **step2_nbrs;
    int *step1_recvfrom_device, **step2_nbrs_device;

    int my_pe = teami->my_pe;
    int num_pes = teami->size;
    INFO(NVSHMEM_COLL, "step 1 nbr calculation started, num_pes = %d", num_pes);

    k = nvshmemi_options.REDUCE_RECEXCH_KVAL;
    assert(k > 1);

    if (num_pes < k) /* If size of the active set is less than k, reduce the value of k */
        k = (num_pes > 2) ? num_pes : 2;

    /* Calculate p_of_k, p_of_k is the largest power of k that is less than num_pes */
    while (p_of_k <= num_pes) {
        p_of_k *= k;
        log_p_of_k++;
    }
    p_of_k /= k;
    /* protect against underflow warnings when asserts are disabled. */
    if (log_p_of_k > 0) {
        log_p_of_k--;
    }

    step2_nphases = log_p_of_k;
    step1_recvfrom = (int *)malloc(sizeof(int) * (k - 1));
    assert(step1_recvfrom);
    step2_nbrs = (int **)malloc(sizeof(int *) * step2_nphases);
    assert(step2_nbrs);

    for (int i = 0; i < step2_nphases; i++) {
        step2_nbrs[i] = (int *)malloc(sizeof(int) * (k - 1));
        assert(step2_nbrs[i]);
        for (int j = 0; j < k - 1; j++) {
            step2_nbrs[i][j] = TEAM_SCALAR_INVALID;
        }
    }

    rem = num_pes - p_of_k;
    /* rem is the number of PEs that do not particpate in Step 2
     * We need to identify these non-participating PEs. This is done in the following way.
     * The first T PEs are divided into sets of k consecutive PEs each.
     * In each of these sets, the first k-1 PEs are the non-participating
     * PEs while the last PE is the participating PE.
     * The non-participating PEs send their data to the participating PE
     * in their corresponding set.
     */
    T = (rem * k) / (k - 1);

    INFO(NVSHMEM_COLL, "step 1 nbr calculation started. T is %d", T);
    step1_nrecvs = 0;
    step1_sendto = -1;

    /* Step 1 */
    if (my_pe < T) {
        if (my_pe % k != (k - 1)) {                     /* I am a non-participating PE */
            step1_sendto = my_pe + (k - 1 - my_pe % k); /* partipating PE to send the data to */
            /* if the corresponding participating PE is not in T,
             * then send to the Tth PE to preserve non-commutativity */
            if (step1_sendto > T - 1) step1_sendto = T;
            newpe = -1; /* tag this PE as non-participating */
        } else {        /* participating PE */
            for (i = 0; i < k - 1; i++) {
                step1_recvfrom[i] = my_pe - i - 1;
            }
            step1_nrecvs = k - 1;
            newpe = my_pe / k; /* this is the new PE amongst the set of participating PEs */
        }
    } else { /* PE >= T */
        newpe = my_pe - rem;

        if (my_pe == T && (T - 1) % k != k - 1 && T >= 1) {
            int nsenders = (T - 1) % k + 1; /* number of PEs sending their data to me in Step 1 */

            for (j = nsenders - 1; j >= 0; j--) {
                step1_recvfrom[nsenders - 1 - j] = T - nsenders + j;
            }
            step1_nrecvs = nsenders;
        }
    }

    INFO(NVSHMEM_COLL, "step 1 nbr computation completed");

    /* Step 2 */
    if (step1_sendto == -1) { /* calulate step2_nbrs only for participating PEs */
        int *digit = (int *)malloc(sizeof(int) * step2_nphases);
        assert(digit != NULL);
        int temppe = newpe;
        int mask = 0x1;
        int phase = 0, cbit, cnt, nbr, power;

        /* calculate the digits in base k representation of newpe */
        for (i = 0; i < log_p_of_k; i++) digit[i] = 0;

        int remainder, i_digit = 0;
        while (temppe != 0) {
            remainder = temppe % k;
            temppe = temppe / k;
            digit[i_digit] = remainder;
            i_digit++;
        }
        while (mask < p_of_k) {
            cbit =
                digit[phase]; /* phase_th digit changes in this phase, obtain its original value */
            cnt = 0;
            for (i = 0; i < k; i++) { /* there are k-1 neighbors */
                if (i != cbit) {      /* do not generate yourself as your nieighbor */
                    digit[phase] = i; /* this gets us the base k representation of the neighbor */

                    /* calculate the base 10 value of the neighbor PE */
                    nbr = 0;
                    power = 1;
                    for (j = 0; j < log_p_of_k; j++) {
                        nbr += digit[j] * power;
                        power *= k;
                    }
                    /* calculate its real PE and store it */
                    step2_nbrs[phase][cnt] =
                        (nbr < rem / (k - 1)) ? (nbr * k) + (k - 1) : nbr + rem;
                    cnt++;
                }
            }
            digit[phase] = cbit; /* reset the digit to original value */
            phase++;
            mask *= k;
        }
        free(digit);
    }
    // Update with global PE numbers
    if (step1_sendto != -1) step1_sendto = teami->pe_mapping[step1_sendto];
    for (int i = 0; i < step1_nrecvs; i++) step1_recvfrom[i] = teami->pe_mapping[step1_recvfrom[i]];
    for (int i = 0; i < step2_nphases; i++) {
        for (int j = 0; j < k - 1; j++) {
            if (step2_nbrs[i][j] != TEAM_SCALAR_INVALID) {
                step2_nbrs[i][j] = teami->pe_mapping[step2_nbrs[i][j]];
            }
        }
    }

    // Copy the data to device memory
    CUDA_RUNTIME_CHECK(cudaMalloc(&step1_recvfrom_device, sizeof(int) * (k - 1)));
    CUDA_RUNTIME_CHECK(cudaMalloc(
        &step2_nbrs_device,
        sizeof(int *) * (step2_nphases + 1))); /* + 1 to make it non-zero otherwise cuMemAlloc
                                                  returns error when step2_nphases is 0 */

    for (int i = 0; i < step2_nphases; i++) {
        void *dev_ptr;
        CUDA_RUNTIME_CHECK(cudaMalloc(&dev_ptr, sizeof(int) * (k - 1)));
        CUDA_RUNTIME_CHECK(cudaMemcpy((int **)step2_nbrs_device + i, &dev_ptr, sizeof(int *),
                                      cudaMemcpyHostToDevice));
    }
    CUDA_RUNTIME_CHECK(cudaMemcpy(step1_recvfrom_device, step1_recvfrom, sizeof(int) * step1_nrecvs,
                                  cudaMemcpyHostToDevice));
    void *dev_ptr, *dev_ptr_2;
    dev_ptr = step2_nbrs_device;
    for (int i = 0; i < step2_nphases; i++) {
        CUDA_RUNTIME_CHECK(
            cudaMemcpy(&dev_ptr_2, (int **)dev_ptr + i, sizeof(int *), cudaMemcpyDeviceToHost));
        CUDA_RUNTIME_CHECK(
            cudaMemcpy(dev_ptr_2, step2_nbrs[i], sizeof(int) * (k - 1), cudaMemcpyHostToDevice));
    }
    teami->reduce_recexch.step1_sendto = step1_sendto;
    teami->reduce_recexch.step1_nrecvs = step1_nrecvs;
    teami->reduce_recexch.step2_nphases = step2_nphases;
    teami->reduce_recexch.step1_recvfrom = step1_recvfrom_device;
    teami->reduce_recexch.step2_nbrs = step2_nbrs_device;

    free(step1_recvfrom);
    for (int i = 0; i < step2_nphases; i++) {
        if (step2_nbrs[i]) {
            free(step2_nbrs[i]);
        }
    }
    free(step2_nbrs);
}

static void nvshmemi_recexchalgo_free_mem(nvshmemi_team_t *teami) {
    if (teami->reduce_recexch.step1_recvfrom != NULL) {
        CUDA_RUNTIME_CHECK(cudaFree(teami->reduce_recexch.step1_recvfrom));
    }
    if (teami->reduce_recexch.step2_nbrs != NULL) {
        if (teami->reduce_recexch.step2_nphases != RED_REC_INVALID_SCALAR) {
            for (int i = 0; i < teami->reduce_recexch.step2_nphases; i++) {
                void *dev_ptr;
                CUDA_RUNTIME_CHECK(cudaMemcpy(&dev_ptr, teami->reduce_recexch.step2_nbrs + i,
                                              sizeof(int *), cudaMemcpyDeviceToHost));
                if (dev_ptr != NULL) {
                    CUDA_RUNTIME_CHECK(cudaFree(dev_ptr));
                }
            }
            CUDA_RUNTIME_CHECK(cudaFree(teami->reduce_recexch.step2_nbrs));
        }
    }
}

static inline void nvshmemi_bit_set(unsigned char *ptr, size_t size, size_t index) {
    assert(size > 0 && (index < size * CHAR_BIT));

    size_t which_byte = index / CHAR_BIT;
    ptr[which_byte] |= (1 << (index % CHAR_BIT));

    return;
}

static inline void nvshmemi_bit_clear(unsigned char *ptr, size_t size, size_t index) {
    assert(size > 0 && (index < size * CHAR_BIT));

    size_t which_byte = index / CHAR_BIT;
    ptr[which_byte] &= ~(1 << (index % CHAR_BIT));

    return;
}

static inline unsigned char nvshmemi_bit_fetch(unsigned char *ptr, size_t index) {
    return (ptr[index / CHAR_BIT] >> (index % CHAR_BIT)) & 1;
}

/* Create a bit string of the format AAAAAAAA.BBBBBBBB into str for the byte
 * array passed via ptr. */
static inline void nvshmemi_bit_to_string(char *str, size_t str_size, unsigned char *ptr,
                                          size_t ptr_size) {
    size_t off = 0;

    for (size_t i = 0; i < ptr_size; i++) {
        for (size_t j = 0; j < CHAR_BIT; j++) {
            off += snprintf(str + off, str_size - off, "%s",
                            (ptr[i] & (1 << (CHAR_BIT - 1 - j))) ? "1" : "0");
            if (off >= str_size) return;
        }
        if (i < ptr_size - 1) {
            off += snprintf(str + off, str_size - off, ".");
            if (off >= str_size) return;
        }
    }
}

/*
 * Checks if the team has a constant stride.
 * This is used to determine whether we can use the older stride based index selection.
 */
static inline int nvshmemi_team_get_stride(nvshmemi_team_t *team) {
    int stride = team->pe_mapping[1] - team->pe_mapping[0];
    for (int i = 2; i < team->size; i++) {
        if (team->pe_mapping[i] - team->pe_mapping[i - 1] != stride) {
            return TEAM_SCALAR_INVALID;
        }
    }
    return stride;
}

/* Checks whether a PE has a consistent stride given (start, stride, size).
 * This function is useful within a loop across PE IDs, and sets 'start',
 * 'stride' and 'size' accordingly upon exiting the loop. It also assumes
 * 'start' and 'stride' are initialized to a negative number and 'size' to 0.
 * If an inconsistent stride is found, returns -1. */
static inline int check_for_linear_stride(int pe, int *start, int *stride, int *size) {
    if (*start < 0) {
        *start = pe;
        (*size)++;
    } else if (*stride < 0) {
        *stride = pe - *start;
        (*size)++;
    } else if ((pe - *start) % *stride != 0) {
        NVSHMEMI_WARN_PRINT("Detected non-uniform stride inserting PE %d into <%d, %d, %d>\n", pe,
                            *start, *stride, *size);
        return -1;
    } else {
        (*size)++;
    }
    return 0;
}

static inline int nvshmemi_pe_in_active_set(int global_pe, int PE_start, int PE_stride,
                                            int PE_size) {
    int n = (global_pe - PE_start) / PE_stride;
    if (global_pe < PE_start || (global_pe - PE_start) % PE_stride || n >= PE_size)
        return -1;
    else {
        return n;
    }
}

int nvshmemi_team_translate_pe_to_team_world_wrap(nvshmemi_team_t *src_team, int src_pe) {
    return src_team->pe_mapping[src_pe % src_team->size];
}

int nvshmemi_team_translate_pe_from_team_world(nvshmemi_team_t *dest_team, int src_pe) {
    return dest_team->pe_mapping[dest_team->size + src_pe];
}

int nvshmemi_team_translate_pe(nvshmemi_team_t *src_team, int src_pe, nvshmemi_team_t *dest_team) {
    int src_pe_world, dest_pe = -1;

    if (src_pe > src_team->size) return -1;

    src_pe_world = nvshmemi_team_translate_pe_to_team_world_wrap(src_team, src_pe);
    assert(src_pe_world >= src_team->start && src_pe_world < nvshmemi_state->npes);
    dest_pe = nvshmemi_team_translate_pe_from_team_world(dest_team, src_pe_world);

    return dest_pe;
}

static inline size_t get_fcollect_psync_len_per_team() {
    size_t fcollect_ll_threshold =
        nvshmemi_device_state.gpu_coll_env_params_var.fcollect_ll_threshold;
    size_t fcollect_sync_size =
        (2 * 2 * nvshmemi_state->npes * fcollect_ll_threshold) / sizeof(long);
    assert(fcollect_ll_threshold % sizeof(long) == 0);

    return fcollect_sync_size;
}

static inline size_t get_fcollect_ll128_psync_len_per_team() {
    size_t fcollect_ll128_threshold =
        nvshmemi_device_state.gpu_coll_env_params_var.fcollect_ll128_threshold;
    size_t fcollect_ll128_sync_size =
        NVSHMEMI_FCOLLECT_LL128_CALC_PSYNC_SIZE(fcollect_ll128_threshold, char);

    /* scale for npes and two separate psyncs */
    fcollect_ll128_sync_size = fcollect_ll128_sync_size * 2 * nvshmemi_state->npes / sizeof(long);

    return fcollect_ll128_sync_size;
}

static inline size_t get_psync_len_per_team() {
    size_t fcollect_sync_size = get_fcollect_psync_len_per_team();
    size_t fcollect_ll128_sync_size = get_fcollect_ll128_psync_len_per_team();
    /* sync: Two buffers are used - one for sync/barrier collective ops, the second one during team
       split operation reduce: Two pWrk's are used alternatively across consecutive reduce calls,
       this is to avoid having to put a barrier in between bcast: The buffer is split to do multiple
       consecutive broadcast, when all buffers are used, a barrier is called and then again we begin
       from the start of the buffer fcollect: Two sets of buffer are used to alternate between -
       same way as in reduce. The other fator of 2 is because when using LL double the space is
       needed to fuse flag with data */

    size_t ans = (2 * NVSHMEMI_SYNC_SIZE +
                  nvshmemi_device_state.gpu_coll_env_params_var.reduce_scratch_size / sizeof(long) +
                  NVSHMEMI_BCAST_SYNC_SIZE + fcollect_sync_size + 2 * NVSHMEMI_ALLTOALL_SYNC_SIZE +
                  fcollect_ll128_sync_size);
    return ans;
}

size_t nvshmemi_get_teams_mem_requirement() {
    size_t psync_size = get_psync_len_per_team();
    size_t teams_mem_req = sizeof(long) * nvshmemi_max_teams * psync_size + /* psync's */
                           2 * N_PSYNC_BYTES +                              /* psync_pool_avail */
                           2 * sizeof(int) +                                /* team_ret_val */
                           2 * sizeof(long) * nvshmemi_max_teams            /* storing counters */
#ifdef NVSHMEM_USE_NCCL
                           + sizeof(ncclUniqueId)
#endif
                           + sizeof(nvshmemi_team_creation_pe_info_t) * nvshmemi_state->npes +
                           N_PSYNC_BYTES * nvshmemi_state->npes;
    INFO(NVSHMEM_INIT, "team psync mem req %ld bytes, team mem total req %d bytes, max teams %ld\n",
         psync_size, teams_mem_req, nvshmemi_max_teams);
    return teams_mem_req;
}

#ifdef NVSHMEM_USE_NCCL
void nvshmemi_team_init_nccl_comm(nvshmemi_team_t *teami) {
    ncclUniqueId Id;
    int size = teami->size;
    /* This is technical debt where we are using the REDUCE op psync as scratchpad for src/dst of
     * broadcast broadcast's psync is used for LL8 and other algorithms, making it non-trivial to
     * share when issued from the host as a src or dest buffer.
     *
     * When reduce coll supports LL8 algorithm, we need to clean this up as a independent scratch
     * space
     */
    long *pWrk = nvshmemi_team_get_psync(teami, REDUCE);
    if (teami->my_pe == 0) {
        NCCL_CHECK(nccl_ftable.GetUniqueId(&Id));
        CUDA_RUNTIME_CHECK(cudaMemcpy(pWrk, &Id, sizeof(ncclUniqueId), cudaMemcpyHostToDevice));
        CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
        for (int i = 0; i < size; i++) {
            nvshmemx_char_put_nbi_on_stream((char *)pWrk, (const char *)pWrk, sizeof(ncclUniqueId),
                                            teami->pe_mapping[i], (cudaStream_t)0);
        }
        CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
        nvshmemi_barrier(teami->team_idx);
    } else {
        nvshmemi_barrier(teami->team_idx);
        CUDA_RUNTIME_CHECK(cudaMemcpy(&Id, pWrk, sizeof(ncclUniqueId), cudaMemcpyDeviceToHost));
    }
    INFO(NVSHMEM_TEAM, "Calling ncclCommInitRank, teami->size = %d, teami->my_pe = %d", teami->size,
         teami->my_pe);
    NCCL_CHECK(
        nccl_ftable.CommInitRank((ncclComm_t *)&teami->nccl_comm, teami->size, Id, teami->my_pe));
}
#endif /* NVSHMEM_USE_NCCL */
void nvshmemi_team_set_p2p_connectivity(nvshmemi_team_t *teami) {
    teami->are_gpus_p2p_connected = nvshmemi_team_is_subset(teami, nvshmemi_team_shared);
}

bool nvshmemi_team_support_nvls(nvshmemi_team_t *team) {
    return ((team->are_gpus_p2p_connected) && (team->nvls_rsc != nullptr));
}

bool nvshmemi_team_is_owner_nvls(nvshmemi_team_t *team) {
    nvshmemi_nvls_rsc *nvls = reinterpret_cast<nvshmemi_nvls_rsc *>(team->nvls_rsc);
    INFO(NVSHMEM_TEAM, "Team ID: %d NVLS Resource Owner ID: %d\n", team->team_idx,
         nvls->get_owner());
    return (nvls->is_owner(team));
}

static bool nvshmemi_team_is_nvls_capable(nvshmemi_team_t *team) {
    return (team->are_gpus_p2p_connected && team->size >= 2 && nvshmemi_state->is_platform_nvls);
}

nvshmemi_team_t *nvshmemi_team_get_same_existing_nvls_team(nvshmemi_team_t *team) {
    if (nvshmemi_options.DISABLE_NVLS_SHARING) {
        return nullptr;
    }
    for (int i = 0; i < team->team_idx; i++) {
        if (nvshmemi_team_is_identical(team, nvshmemi_team_pool[i])) {
            if (nvshmemi_team_support_nvls(nvshmemi_team_pool[i])) {
                return nvshmemi_team_pool[i];
            }
            return nullptr;
        }
    }
    return nullptr;
}

/* NVLS Resource management for teams */
static void nvshmemi_team_destroy_nvls(nvshmemi_team_t *team) {
    if (team->nvls_rsc == nullptr) return; /* NOOP */

    nvshmemi_nvls_rsc *nvls_obj = nullptr;
    nvls_obj = reinterpret_cast<nvshmemi_nvls_rsc *>(team->nvls_rsc);
    if (nvls_obj->get_refcount() == 0) { /* Last reference */
        nvshmemi_state->heap_obj->nvls_unmap_heap_memory_by_team(team);
        nvshmemi_state->heap_obj->nvls_unbind_heap_memory_by_team(team);
        nvls_obj->free_group_mem();
        nvls_obj->release_owner();
        delete nvls_obj;
        cudaFree(team->nvls_rsc_base_ptr);
        team->nvls_rsc = nullptr;
        INFO(NVSHMEM_TEAM, "NVLS Resource Destroyed for Team ID %d\n", team->team_idx);
    } else {
        nvls_obj->del_refcount(); /* Shared nvls resource */
        /* Ownership of NVLS resource is necessary to allow for newly allocated UC memory to be
         * bound and mapped to MC heap */
        if (nvls_obj->is_owner(team)) {
            // Transfer ownership to one of the dup teams
            NVSHMEMU_FOR_EACH_IF(
                i, nvshmemi_max_teams,
                nvshmemi_team_pool[i] != NULL && nvshmemi_team_support_nvls(nvshmemi_team_pool[i]),
                {
                    // Find first duplicate team that shares the nvls rsc and make it the owner
                    if (nvshmemi_team_pool[i]->nvls_rsc == team->nvls_rsc) {
                        nvls_obj->release_owner();
                        nvls_obj->assign_owner(nvshmemi_team_pool[i]);
                        break;
                    }
                });
        }
    }
}

static int nvshmemi_team_create_nvls(nvshmemi_team_t *team) {
    int status = -1;
    uint64_t mc_heap_base;
    nvshmemi_nvls_rsc *nvls_obj = nullptr;

    if (!nvshmemi_team_is_nvls_capable(team)) {
        team->nvls_rsc = nullptr;
        WARN("Skipping NVLINK SHARP resource initialized for team ID: %d\n", team->team_idx);
        return 0;
    }

    try {
        team->nvls_rsc = reinterpret_cast<void *>(new nvshmemi_nvls_rsc(team, nvshmemi_state));
    } catch (nvshmemi_nvls_exception &exp) {
        WARN("NVLINK SHARP resource initialization failed for team ID: %d\n", team->team_idx);
        team->nvls_rsc = nullptr;
        return 0;
    }

    nvls_obj = reinterpret_cast<nvshmemi_nvls_rsc *>(team->nvls_rsc);
    status = nvls_obj->reserve_group_mem();
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, cleanup,
                          "Reserve multicast group mapping failed for pe %d\n", team->my_pe);

    nvls_obj->assign_owner(team);
    CUDA_RUNTIME_CHECK(cudaMalloc((void **)&team->nvls_rsc_base_ptr, sizeof(void *)));
    mc_heap_base = (uint64_t)(nvls_obj->get_mc_base());
    CUDA_RUNTIME_CHECK(
        cudaMemcpy(team->nvls_rsc_base_ptr, &mc_heap_base, sizeof(void *), cudaMemcpyHostToDevice));

    /* Make a MC handle as large as reserved heap size (VA range) */
    status = nvshmemi_state->heap_obj->nvls_create_heap_memory_by_team(team);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, cleanup,
                          "Create multicast groups for UC heap failed for pe %d team ID %d\n",
                          team->my_pe, team->team_idx);

    status = nvshmemi_state->heap_obj->nvls_map_heap_memory_by_team(team);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, cleanup,
                          "Mapping multicast groups for UC heap failed for pe %d team ID %d\n",
                          team->my_pe, team->team_idx);

    INFO(NVSHMEM_TEAM, "NVLS Resource Created for Team ID %d MC VA Base: %llx\n", team->team_idx,
         mc_heap_base);
    return (status);

cleanup:
    (void)nvls_obj->free_group_mem();
    delete nvls_obj;
    team->nvls_rsc = nullptr;
    return (status);
}

static int nvshmemi_team_bind_nvls(nvshmemi_team_t *team) {
    int status = -1;
    nvshmemi_nvls_rsc *nvls_obj = nullptr;
    /* Bind existing UC mem handles to the single MC handle */
    nvls_obj = reinterpret_cast<nvshmemi_nvls_rsc *>(team->nvls_rsc);
    if (!nvls_obj->is_owner(team)) return 0;

    status = nvshmemi_state->heap_obj->nvls_bind_heap_memory_by_team(team);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, cleanup,
                          "Binding multicast groups to UC heap failed for pe %d team ID %d\n",
                          team->my_pe, team->team_idx);
cleanup:
    return (status);
}

static int nvshmemi_team_setup_nvls(nvshmemi_team_t *team) {
    int status = NVSHMEMX_SUCCESS;
    nvshmemi_team_t *identical_team = nullptr;

    /* Check if there is an identical team with NVLS resources */
    identical_team = nvshmemi_team_get_same_existing_nvls_team(team);
    if (identical_team != nullptr) {
        team->nvls_rsc = identical_team->nvls_rsc;
        team->nvls_rsc_base_ptr = identical_team->nvls_rsc_base_ptr;
        assert(team->nvls_rsc != nullptr);
        assert(team->nvls_rsc_base_ptr != nullptr);
        nvshmemi_nvls_rsc *nvls = reinterpret_cast<nvshmemi_nvls_rsc *>(team->nvls_rsc);
        nvls->add_refcount();
        INFO(NVSHMEM_TEAM, "Successful NVLS resource sharing for new team ID: %d (parent ID: %d)\n",
             team->team_idx, identical_team->team_idx);
        goto cleanup;
    }

    /* Initialize NVLS resources for team supporting P2P connected GPUs */
    INFO(NVSHMEM_COLL, "Creating unique NVLS resources for new team ID: %d\n", team->team_idx);
    status = nvshmemi_team_create_nvls(team);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, cleanup,
                          "NVLS resource initialization failed for team ID: %d\n", team->team_idx);

    /* Any prior UC allocations need to bound to this team's MC groups */
    if (team->nvls_rsc != nullptr) {
        status = nvshmemi_team_bind_nvls(team);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, cleanup,
                              "NVLS resource bind and mapping existing UC mappings to MC heap "
                              "failed for team ID: %d\n",
                              team->team_idx);
    }

cleanup:
    return (status);
}

/* Team Management Routines */
int nvshmemi_set_max_teams(void) {
    nvshmemi_max_teams = nvshmemi_options.MAX_TEAMS;
    if (nvshmemi_max_teams < NVSHMEM_TEAMS_MIN) nvshmemi_max_teams = NVSHMEM_TEAMS_MIN;
    /* On NVLS sharp enabled platforms increase default max teams to 64 */
    if (nvshmemi_state->is_platform_nvls) {
        nvshmemi_max_teams = (nvshmemi_max_teams > NVSHMEMI_REDUCE_MAX_CTA_COUNT)
                                 ? nvshmemi_max_teams
                                 : NVSHMEMI_REDUCE_MAX_CTA_COUNT;
    }

    if (nvshmemi_max_teams > N_PSYNC_BYTES * CHAR_BIT) {
        N_PSYNC_BYTES = ((nvshmemi_max_teams + CHAR_BIT - 1) / CHAR_BIT);
    }
    return 0;
}

static void nvshmemi_team_populate_from_world_pe_mapping(nvshmemi_team_t *team) {
    for (int i = 0; i < team->size; i++) {
        int global_pe_index = team->pe_mapping[i];
        team->pe_mapping[global_pe_index + team->size] = nvshmemi_team_world->pe_mapping[i];
    }
}

static void nvshmemi_team_populate_pe_mappings_from_constant_stride(nvshmemi_team_t *team) {
    for (int i = 0; i < team->size; i++) {
        int global_pe_index = team->start + i * team->stride;
        team->pe_mapping[i] = global_pe_index;
    }
    nvshmemi_team_populate_from_world_pe_mapping(team);
}

int nvshmemi_team_init(void) {
    long psync_len;
    int start, stride, size;
    int *scratch = NULL;
    int status = 0;
    uint64_t *hostHash = NULL;
    uint64_t myHostHash = 0;
    nvshmem_transport_pe_info_t *pe_info;
    int i;

    /* Initialize NVSHMEM_TEAM_WORLD */
    if (nvshmemi_team_allocate_team(&nvshmemi_team_world, &nvshmemi_device_team_world,
                                    nvshmemi_state->npes) != NVSHMEMX_SUCCESS) {
        return NVSHMEMX_ERROR_OUT_OF_MEMORY;
    }
    nvshmemi_team_world->team_idx = NVSHMEM_TEAM_WORLD_INDEX;
    NVSHMEMI_TEAM_DUP_INITIALIZER(nvshmemi_team_world, NVSHMEM_TEAM_WORLD_INDEX);
    nvshmemi_team_world->start = 0;
    nvshmemi_team_world->stride = 1;
    nvshmemi_team_world->size = nvshmemi_state->npes;
    nvshmemi_team_world->my_pe = nvshmemi_state->mype;
    nvshmemi_team_world->rdxn_count = 0;
    nvshmemi_team_world->config_mask = 0;
    nvshmemi_team_world->ll_flag = 1;
    nvshmemi_team_world->alltoall_count = 0;
    nvshmemi_team_world->bcast_count = 0;
    nvshmemi_team_world->bcast_sync_offset = 0;
    nvshmemi_team_world->fcollect_count = 0;
    nvshmemi_team_world->is_team_node = false;
    nvshmemi_team_world->is_team_same_mype_node = false;

    nvshmemi_team_populate_pe_mappings_from_constant_stride(nvshmemi_team_world);
    nvshmemi_recexchalgo_get_neighbors(nvshmemi_team_world);

    /* Collect list of p2p connected PEs */
    int *p2p_pe_list = (int *)malloc(nvshmemi_team_world->size * sizeof(int));
    int n_p2p_pes = 0;
    int my_idx_in_p2p_list = 0;
    for (int i = 0; i < nvshmemi_team_world->size; i++) {
        if (nvshmemi_state->heap_obj->get_local_pe_base()[i]) {
            if (i == nvshmemi_team_world->my_pe) my_idx_in_p2p_list = n_p2p_pes;
            p2p_pe_list[n_p2p_pes++] = i;
        }
    }

    std::ostringstream ss;
    for (int i = 0; i < n_p2p_pes; i++) {
        ss << p2p_pe_list[i] << " ";
    }
    INFO(NVSHMEM_INIT, "P2P list: %s", ss.str().c_str());

    /* Make sure that n_p2p_pes is same for all PEs to form TEAM_SHARED */
    int *n_p2p_pes_all = (int *)malloc(nvshmemi_team_world->size * sizeof(int));
    int *p2p_pe_list_all = (int *)malloc(sizeof(int) * n_p2p_pes * nvshmemi_team_world->size);

    nvshmemi_boot_handle.allgather((void *)&n_p2p_pes, (void *)n_p2p_pes_all, sizeof(int),
                                   &nvshmemi_boot_handle);

    for (i = 0; i < nvshmemi_team_world->size; i++) {
        if (n_p2p_pes_all[i] != n_p2p_pes) {
            INFO(NVSHMEM_INIT,
                 "n_p2p_pes is not equal across PEs, setting NVSHMEM_TEAM_SHARED to self");
            goto team_shared_single_pe;
        }
    }

    /* Gather p2p lists of all PEs and ensure they are the same */
    nvshmemi_boot_handle.allgather((void *)p2p_pe_list, (void *)p2p_pe_list_all,
                                   sizeof(int) * n_p2p_pes, &nvshmemi_boot_handle);
    for (i = 0; i < n_p2p_pes; i++) {
        if (memcmp((void *)p2p_pe_list, (void *)&p2p_pe_list_all[p2p_pe_list[i] * n_p2p_pes],
                   sizeof(int) * n_p2p_pes) != 0) {
            INFO(NVSHMEM_INIT, "P2P lists are not symmetric, setting NVSHMEM_TEAM_SHARED to self");
            goto team_shared_single_pe;
        }
    }

    for (int i = 2; i < n_p2p_pes; i++) {
        if (p2p_pe_list[i] - p2p_pe_list[i - 1] != p2p_pe_list[i - 1] - p2p_pe_list[i - 2]) {
            INFO(NVSHMEM_INIT,
                 "P2P list is not of the form (start, stride, size). Cannot form "
                 "NVSHMEM_TEAM_SHARED.");
            goto team_shared_single_pe;
        }
    }

    /* Initialize NVSHMEM_TEAM_SHARED */
    if (nvshmemi_team_allocate_team(&nvshmemi_team_shared, &nvshmemi_device_team_shared,
                                    n_p2p_pes) != NVSHMEMX_SUCCESS) {
        return NVSHMEMX_ERROR_OUT_OF_MEMORY;
    }
    nvshmemi_team_shared->team_idx = NVSHMEM_TEAM_SHARED_INDEX;
    NVSHMEMI_TEAM_DUP_INITIALIZER(nvshmemi_team_shared, NVSHMEM_TEAM_SHARED_INDEX);
    nvshmemi_team_shared->my_pe = my_idx_in_p2p_list;
    nvshmemi_team_shared->start = p2p_pe_list[0];
    nvshmemi_team_shared->stride = n_p2p_pes > 1 ? (p2p_pe_list[1] - p2p_pe_list[0]) : 1;
    nvshmemi_team_shared->size = n_p2p_pes;
    nvshmemi_team_shared->is_team_same_mype_node = false;

    goto team_shared_setup;

team_shared_single_pe:
    nvshmemi_team_shared->my_pe = 0;
    nvshmemi_team_shared->start = nvshmemi_state->mype;
    nvshmemi_team_shared->stride = 1;
    nvshmemi_team_shared->size = 1;
    nvshmemi_team_shared->is_team_node = true;
    nvshmemi_team_shared->is_team_same_mype_node = true;

team_shared_setup:
    free(n_p2p_pes_all);
    free(p2p_pe_list_all);
    free(p2p_pe_list);

    nvshmemi_team_shared->rdxn_count = 0;
    nvshmemi_team_shared->config_mask = 0;

    nvshmemi_team_shared->ll_flag = 1;
    nvshmemi_team_shared->alltoall_count = 0;
    nvshmemi_team_shared->bcast_count = 0;
    nvshmemi_team_shared->bcast_sync_offset = 0;
    nvshmemi_team_shared->fcollect_count = 0;
    nvshmemi_team_shared->are_gpus_p2p_connected = 1;

    nvshmemi_team_populate_pe_mappings_from_constant_stride(nvshmemi_team_shared);
    nvshmemi_recexchalgo_get_neighbors(nvshmemi_team_shared);

    INFO(NVSHMEM_INIT, "NVSHMEM_TEAM_SHARED: start=%d, stride=%d, size=%d",
         nvshmemi_team_shared->start, nvshmemi_team_shared->stride, nvshmemi_team_shared->size);
    nvshmemi_team_set_p2p_connectivity(nvshmemi_team_world);

    /* Search for on-node peer PEs while checking for a consistent stride */
    myHostHash = nvshmemu_getHostHash();
    hostHash = (uint64_t *)malloc(sizeof(uint64_t) * nvshmemi_state->npes);
    NVSHMEMI_NULL_ERROR_JMP(hostHash, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, cleanup,
                            "hostHash allocation failed \n");
    status = nvshmemi_boot_handle.allgather((void *)&myHostHash, (void *)hostHash, sizeof(uint64_t),
                                            &nvshmemi_boot_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, cleanup,
                          "allgather of host hashes failed\n");
    start = -1;
    stride = -1;
    size = 0;

    for (int pe = 0; pe < nvshmemi_state->npes; pe++) {
        if (hostHash[pe] != myHostHash) continue;

        int ret = check_for_linear_stride(pe, &start, &stride, &size);
        if (ret < 0) {
            start = nvshmemi_state->mype;
            stride = 1;
            size = 1;
            break;
        }
    }
    assert(start >= 0 && size > 0);

    /* Initialize NVSHMEMX_TEAM_NODE */
    if (nvshmemi_team_allocate_team(&nvshmemi_team_node, &nvshmemi_device_team_node, size) !=
        NVSHMEMX_SUCCESS) {
        return NVSHMEMX_ERROR_OUT_OF_MEMORY;
    }
    nvshmemi_team_node->team_idx = NVSHMEM_TEAM_NODE_INDEX;
    NVSHMEMI_TEAM_DUP_INITIALIZER(nvshmemi_team_node, NVSHMEM_TEAM_NODE_INDEX);
    nvshmemi_team_world->team_node = nvshmemi_team_node->team_idx;
    nvshmemi_team_node->my_pe = nvshmemi_state->mype_node;
    nvshmemi_team_node->rdxn_count = 0;
    nvshmemi_team_node->config_mask = 0;
    nvshmemi_team_node->ll_flag = 1;
    nvshmemi_team_node->alltoall_count = 0;
    nvshmemi_team_node->bcast_count = 0;
    nvshmemi_team_node->bcast_sync_offset = 0;
    nvshmemi_team_node->fcollect_count = 0;

    nvshmemi_team_node->start = start;
    nvshmemi_team_node->stride = (stride == -1) ? 1 : stride;
    nvshmemi_team_node->size = size;
    if (nvshmemi_team_is_identical(nvshmemi_team_world, nvshmemi_team_node)) {
        nvshmemi_team_world->is_team_node = true;
    }
    if (nvshmemi_team_is_identical(nvshmemi_team_shared, nvshmemi_team_node)) {
        nvshmemi_team_shared->is_team_node = true;
    }

    nvshmemi_team_populate_pe_mappings_from_constant_stride(nvshmemi_team_node);
    nvshmemi_team_set_p2p_connectivity(nvshmemi_team_node);
    nvshmemi_recexchalgo_get_neighbors(nvshmemi_team_node);
    nvshmemi_team_node->is_team_node = true;
    nvshmemi_team_node->is_team_same_mype_node = false;
    INFO(NVSHMEM_INIT, "NVSHMEMX_TEAM_NODE: start=%d, stride=%d, size=%d",
         nvshmemi_team_node->start, nvshmemi_team_node->stride, nvshmemi_team_node->size);

    /* Initialize NVSHMEMX_TEAM_SAME_MYPE_NODE */
    if (nvshmemi_team_allocate_team(
            &nvshmemi_team_same_mype_node, &nvshmemi_device_team_same_mype_node,
            nvshmemi_state->npes / nvshmemi_state->npes_node) != NVSHMEMX_SUCCESS) {
        return NVSHMEMX_ERROR_OUT_OF_MEMORY;
    }
    nvshmemi_team_same_mype_node->team_idx = NVSHMEM_TEAM_SAME_MYPE_NODE_INDEX;
    NVSHMEMI_TEAM_DUP_INITIALIZER(nvshmemi_team_same_mype_node, NVSHMEM_TEAM_SAME_MYPE_NODE_INDEX);
    nvshmemi_team_world->team_same_mype_node = nvshmemi_team_same_mype_node->team_idx;
    nvshmemi_team_same_mype_node->my_pe = nvshmemi_state->mype / nvshmemi_state->npes_node;
    nvshmemi_team_same_mype_node->rdxn_count = 0;
    nvshmemi_team_same_mype_node->config_mask = 0;

    nvshmemi_team_same_mype_node->start = nvshmemi_state->mype_node;
    nvshmemi_team_same_mype_node->stride = nvshmemi_state->npes_node;
    nvshmemi_team_same_mype_node->size = nvshmemi_state->npes / nvshmemi_state->npes_node;
    assert(nvshmemi_state->npes % nvshmemi_state->npes_node == 0);
    nvshmemi_team_same_mype_node->ll_flag = 1;
    nvshmemi_team_same_mype_node->alltoall_count = 0;
    nvshmemi_team_same_mype_node->bcast_count = 0;
    nvshmemi_team_same_mype_node->bcast_sync_offset = 0;
    nvshmemi_team_same_mype_node->fcollect_count = 0;
    nvshmemi_team_same_mype_node->is_team_node = false;
    nvshmemi_team_same_mype_node->is_team_same_mype_node = true;

    INFO(NVSHMEM_INIT, "NVSHMEMX_TEAM_SAME_MYPE_NODE: start=%d, stride=%d, size=%d",
         nvshmemi_team_same_mype_node->start, nvshmemi_team_same_mype_node->stride,
         nvshmemi_team_same_mype_node->size);

    nvshmemi_team_populate_pe_mappings_from_constant_stride(nvshmemi_team_same_mype_node);
    nvshmemi_team_set_p2p_connectivity(nvshmemi_team_same_mype_node);
    nvshmemi_recexchalgo_get_neighbors(nvshmemi_team_same_mype_node);

    /* Initialize team NVSHMEMI_TEAM_SAME_GPU */
    pe_info = nvshmemi_state->pe_info;
    start = -1;
    stride = -1;
    size = 0;
    for (int pe = 0; pe < nvshmemi_state->npes; pe++) {
        if (pe_info[pe].hostHash != pe_info[nvshmemi_state->mype].hostHash ||
            memcmp(&pe_info[pe].gpu_uuid, &pe_info[nvshmemi_state->mype].gpu_uuid,
                   sizeof(cudaUUID_t)) != 0)
            continue;
        int ret = check_for_linear_stride(pe, &start, &stride, &size);
        if (ret < 0) {
            NVSHMEMI_ERROR_EXIT("Could not form NVSHMEMI_TEAM_SAME_GPU\n");
            break;
        }
    }
    assert(start >= 0 && size > 0);

    if (nvshmemi_team_allocate_team(&nvshmemi_team_same_gpu, &nvshmemi_device_team_same_gpu,
                                    size) != NVSHMEMX_SUCCESS) {
        return NVSHMEMX_ERROR_OUT_OF_MEMORY;
    }
    nvshmemi_team_same_gpu->team_idx = NVSHMEM_TEAM_SAME_GPU_INDEX;
    NVSHMEMI_TEAM_DUP_INITIALIZER(nvshmemi_team_same_gpu, NVSHMEM_TEAM_SAME_GPU_INDEX);
    nvshmemi_team_same_gpu->rdxn_count = 0;
    nvshmemi_team_same_gpu->ll_flag = 1;
    nvshmemi_team_same_gpu->alltoall_count = 0;
    nvshmemi_team_same_gpu->bcast_count = 0;
    nvshmemi_team_same_gpu->bcast_sync_offset = 0;
    nvshmemi_team_same_gpu->fcollect_count = 0;
    nvshmemi_team_same_gpu->config_mask = 0;
    nvshmemi_team_same_gpu->my_pe = (nvshmemi_state->mype - start) / stride;
    nvshmemi_team_same_gpu->start = start;
    nvshmemi_team_same_gpu->stride = (stride == -1) ? 1 : stride;
    nvshmemi_team_same_gpu->size = size;
    nvshmemi_team_same_gpu->is_team_node = true;
    nvshmemi_team_same_gpu->is_team_same_mype_node = false;
    INFO(NVSHMEM_INIT, "NVSHMEMI_TEAM_SAME_GPU: start=%d, stride=%d, size=%d",
         nvshmemi_team_same_gpu->start, nvshmemi_team_same_gpu->stride,
         nvshmemi_team_same_gpu->size);

    nvshmemi_team_populate_pe_mappings_from_constant_stride(nvshmemi_team_same_gpu);
    nvshmemi_team_set_p2p_connectivity(nvshmemi_team_same_gpu);
    nvshmemi_recexchalgo_get_neighbors(nvshmemi_team_same_gpu);
    /* All GPUs must have same number of processes (requires for us to form teams) */

    /* Initialize team NVSHMEMI_TEAM_GPU_LEADERS */
    scratch = (int *)malloc(sizeof(int) * nvshmemi_state->npes);
    NVSHMEMI_NULL_ERROR_JMP(scratch, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, cleanup,
                            "Unable to allocate host memory for team creation.\n");
    if (nvshmemi_team_same_gpu->start ==
        nvshmemi_state->mype) { /* Only GPU leaders are part of this team */
        if (nvshmemi_team_allocate_team(
                &nvshmemi_team_gpu_leaders, &nvshmemi_device_team_gpu_leaders,
                nvshmemi_state->npes / nvshmemi_team_same_gpu->size) != NVSHMEMX_SUCCESS) {
            return NVSHMEMX_ERROR_OUT_OF_MEMORY;
        }
        nvshmemi_team_gpu_leaders->team_idx = NVSHMEM_TEAM_GPU_LEADERS_INDEX;
        NVSHMEMI_TEAM_DUP_INITIALIZER(nvshmemi_team_gpu_leaders, NVSHMEM_TEAM_GPU_LEADERS_INDEX);
        nvshmemi_team_gpu_leaders->config_mask = 0;

        nvshmemi_team_gpu_leaders->start = 0;
        nvshmemi_team_gpu_leaders->stride =
            (nvshmemi_team_same_gpu->stride == 1) ? nvshmemi_team_same_gpu->size : 1;
        nvshmemi_team_gpu_leaders->size = nvshmemi_state->npes / nvshmemi_team_same_gpu->size;
        nvshmemi_team_gpu_leaders->my_pe =
            (nvshmemi_state->mype - nvshmemi_team_gpu_leaders->start) /
            nvshmemi_team_gpu_leaders->stride;
        nvshmemi_team_gpu_leaders->rdxn_count = 0;
        nvshmemi_team_gpu_leaders->ll_flag = 1;
        nvshmemi_team_gpu_leaders->alltoall_count = 0;
        nvshmemi_team_gpu_leaders->bcast_count = 0;
        nvshmemi_team_gpu_leaders->bcast_sync_offset = 0;
        nvshmemi_team_gpu_leaders->fcollect_count = 0;

        nvshmemi_team_populate_pe_mappings_from_constant_stride(nvshmemi_team_gpu_leaders);
        nvshmemi_team_set_p2p_connectivity(nvshmemi_team_gpu_leaders);
        nvshmemi_recexchalgo_get_neighbors(nvshmemi_team_gpu_leaders);
        status =
            nvshmemi_boot_handle.allgather((void *)&(nvshmemi_team_gpu_leaders->my_pe),
                                           (void *)scratch, sizeof(int), &nvshmemi_boot_handle);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, cleanup,
                              "allgather of gpu leaders failed\n");
        /* Check whether a valid TEAM_GPU_LEADERS was formed */
        int last_mype = -1;
        for (int i = 0; i < nvshmemi_state->npes; i++) {
            if (scratch[i] != -1) {
                if (scratch[i] != last_mype + 1) {
                    WARN(
                        "NVSHMEMI_TEAM_GPU_LEADERS could not be formed, Limited MPG support will "
                        "not be available\n");
                    break;
                } else {
                    last_mype++;
                }
            }
        }
        /* XXX: Note that we are not setting team_node and team_same_mype_node for
         * nvshmemi_team_gpu_leaders */
        nvshmemi_team_gpu_leaders->is_team_node = false;
        nvshmemi_team_gpu_leaders->is_team_same_mype_node = false;
        INFO(NVSHMEM_INIT, "NVSHMEMI_TEAM_GPU_LEADERS: start=%d, stride=%d, size=%d",
             nvshmemi_team_gpu_leaders->start, nvshmemi_team_gpu_leaders->stride,
             nvshmemi_team_gpu_leaders->size);
    } else {
        int my_pe = -1;
        nvshmemi_team_gpu_leaders = NULL;
        status = nvshmemi_boot_handle.allgather((void *)&my_pe, (void *)scratch, sizeof(int),
                                                &nvshmemi_boot_handle);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, cleanup,
                              "allgather of gpu leaders failed\n");
    }
    if (nvshmemi_max_teams < NVSHMEM_TEAMS_MIN) nvshmemi_max_teams = NVSHMEM_TEAMS_MIN;

    if (nvshmemi_max_teams > N_PSYNC_BYTES * CHAR_BIT) {
        NVSHMEMI_ERROR_EXIT("Requested %ld teams, but only %ld are supported\n", nvshmemi_max_teams,
                            N_PSYNC_BYTES * CHAR_BIT);
        goto cleanup;
    }

    status = nvshmemi_init_team_creation_psync();
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_OUT_OF_MEMORY, cleanup,
                          "Failed to reset team creation psync\n");

    nvshmemi_team_pool = (nvshmemi_team_t **)calloc(nvshmemi_max_teams, sizeof(nvshmemi_team_t *));
    NVSHMEMI_NULL_ERROR_JMP(nvshmemi_team_pool, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, cleanup,
                            "nvshmemi_team_pool allocation failed \n");
    CUDA_RUNTIME_CHECK(cudaMalloc((void **)&nvshmemi_device_team_pool,
                                  nvshmemi_max_teams * sizeof(nvshmemi_team_t *)));
    nvshmemi_device_state.team_pool = nvshmemi_device_team_pool;

    for (long i = 0; i < nvshmemi_max_teams; i++) {
        nvshmemi_team_pool[i] = NULL;
    }

    nvshmemi_call_init_array_kernel<nvshmemi_team_t *>(nvshmemi_device_team_pool,
                                                       nvshmemi_max_teams, NULL);

    nvshmemi_team_pool[NVSHMEM_TEAM_WORLD_INDEX] = nvshmemi_team_world;
    nvshmemi_team_pool[NVSHMEM_TEAM_SHARED_INDEX] = nvshmemi_team_shared;
    nvshmemi_team_pool[NVSHMEM_TEAM_NODE_INDEX] = nvshmemi_team_node;
    nvshmemi_team_pool[NVSHMEM_TEAM_SAME_MYPE_NODE_INDEX] = nvshmemi_team_same_mype_node;
    nvshmemi_team_pool[NVSHMEM_TEAM_SAME_GPU_INDEX] = nvshmemi_team_same_gpu;
    if (nvshmemi_team_same_gpu->start == nvshmemi_state->mype)
        nvshmemi_team_pool[NVSHMEM_TEAM_GPU_LEADERS_INDEX] = nvshmemi_team_gpu_leaders;

    /* Allocate pSync pool, each with the maximum possible size requirement */
    /* Create two pSyncs per team for back-to-back collectives and one for barriers.
     * Array organization:
     *
     * [ (world) (shared) (team 1) (team 2) ...  (world) (shared) (team 1) (team 2) ... ]
     *  <----------- groups 1 & 2-------------->|<------------- group 3 ---------------->
     *  <--- (bcast, collect, reduce, etc.) --->|<------ (barriers and syncs) ---------->
     * */

    psync_len = nvshmemi_max_teams * get_psync_len_per_team();
    nvshmemi_psync_pool = (long *)nvshmemi_malloc(sizeof(long) * psync_len);
    NVSHMEMI_NULL_ERROR_JMP(nvshmemi_psync_pool, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, cleanup,
                            "nvshmemi_psync_pool allocation failed \n");

    nvshmemi_device_state.psync_pool = nvshmemi_psync_pool;

    nvshmemi_call_init_array_kernel<long>(nvshmemi_psync_pool, psync_len, NVSHMEMI_SYNC_VALUE);

    nvshmemi_sync_counter = (long *)nvshmemi_malloc(2 * nvshmemi_max_teams * sizeof(long));
    NVSHMEMI_NULL_ERROR_JMP(nvshmemi_sync_counter, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, cleanup,
                            "nvshmemi_sync_counter allocation failed \n");

    nvshmemi_device_state.sync_counter = nvshmemi_sync_counter;
    nvshmemi_update_device_state();

    nvshmemi_call_init_array_kernel<long>(nvshmemi_sync_counter, 2 * nvshmemi_max_teams, 1);

    /* Convenience pointer to the group-3 pSync array (for barriers and syncs): */
    psync_pool_avail = (unsigned char *)malloc(2 * N_PSYNC_BYTES);
    NVSHMEMI_NULL_ERROR_JMP(psync_pool_avail, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, cleanup,
                            "psync_pool_avail allocation failed \n");
    psync_pool_avail_reduced = &psync_pool_avail[N_PSYNC_BYTES];

    device_psync_pool_avail = (unsigned char *)nvshmemi_malloc(2 * N_PSYNC_BYTES);
    NVSHMEMI_NULL_ERROR_JMP(device_psync_pool_avail, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, cleanup,
                            "device_psync_pool_avail allocation failed \n");
    device_psync_pool_avail_reduced = &device_psync_pool_avail[N_PSYNC_BYTES];
    /* Initialize the psync bits to 1, making all slots available: */
    memset(psync_pool_avail, 0, 2 * N_PSYNC_BYTES);
    for (size_t i = 0; i < (size_t)nvshmemi_max_teams; i++) {
        nvshmemi_bit_set(psync_pool_avail, N_PSYNC_BYTES, i);
    }

    /* Set the bits for NVSHMEM_TEAM_WORLD, NVSHMEM_TEAM_SHARED, NVSHMEMX_TEAM_NODE to 0: */
    nvshmemi_bit_clear(psync_pool_avail, N_PSYNC_BYTES, NVSHMEM_TEAM_WORLD_INDEX);
    nvshmemi_bit_clear(psync_pool_avail, N_PSYNC_BYTES, NVSHMEM_TEAM_SHARED_INDEX);
    nvshmemi_bit_clear(psync_pool_avail, N_PSYNC_BYTES, NVSHMEM_TEAM_NODE_INDEX);
    nvshmemi_bit_clear(psync_pool_avail, N_PSYNC_BYTES, NVSHMEM_TEAM_SAME_MYPE_NODE_INDEX);
    nvshmemi_bit_clear(psync_pool_avail, N_PSYNC_BYTES, NVSHMEM_TEAM_SAME_GPU_INDEX);
    nvshmemi_bit_clear(psync_pool_avail, N_PSYNC_BYTES, NVSHMEM_TEAM_GPU_LEADERS_INDEX);

    /* Initialize an integer used to agree on an equal return value across PEs in team creation: */
    team_ret_val = (int *)malloc(sizeof(int) * 2);
    NVSHMEMI_NULL_ERROR_JMP(team_ret_val, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, cleanup,
                            "team_ret_val allocation failed \n");
    team_ret_val_reduced = &team_ret_val[1];

    device_team_ret_val = (int *)nvshmemi_malloc(sizeof(int) * 2);
    NVSHMEMI_NULL_ERROR_JMP(team_ret_val, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, cleanup,
                            "device_team_ret_val allocation failed \n");
    device_team_ret_val_reduced = &device_team_ret_val[1];

    nvshmemi_boot_handle.barrier(
        &nvshmemi_boot_handle); /* To ensure neccessary setup has been done all PEs */

    nvshmemi_team_update_device();
    nvshmemi_boot_handle.barrier(
        &nvshmemi_boot_handle); /* To ensure neccessary setup has been done all PEs */

#ifdef NVSHMEM_USE_NCCL
    if (nvshmemi_use_nccl) {
        /* Setup NCCL usage */
        nvshmemi_team_init_nccl_comm(nvshmemi_team_world);
        nvshmemi_team_init_nccl_comm(nvshmemi_team_shared);
        nvshmemi_team_init_nccl_comm(nvshmemi_team_node);
        nvshmemi_team_init_nccl_comm(nvshmemi_team_same_mype_node);
        nvshmemi_team_init_nccl_comm(nvshmemi_team_same_gpu);
        if (nvshmemi_team_gpu_leaders != NULL) {
            if (nvshmemi_pe_in_active_set(nvshmemi_state->mype, nvshmemi_team_gpu_leaders->start,
                                          nvshmemi_team_gpu_leaders->stride,
                                          nvshmemi_team_gpu_leaders->size) >= 0) {
                nvshmemi_team_init_nccl_comm(nvshmemi_team_gpu_leaders);
            }
        }
    }
#endif /* NVSHMEM_USE_NCCL */

    /* Setup NVLS resources for all internal p2p connected teams */
    NVSHMEMU_FOR_EACH_IF(
        i, nvshmemi_max_teams,
        nvshmemi_team_pool[i] != NULL && nvshmemi_team_pool[i]->are_gpus_p2p_connected, {
            status = nvshmemi_team_setup_nvls(nvshmemi_team_pool[i]);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, cleanup,
                                  "NVLS resource setup failed for team ID: %d\n",
                                  nvshmemi_team_pool[i]->team_idx);
            if (nvshmemi_team_pool[i]->nvls_rsc) {
                INFO(NVSHMEM_TEAM, "Successful NVLS resource setup for team ID: %d\n",
                     nvshmemi_team_pool[i]->team_idx);
            }
        });

    nvshmemi_boot_handle.barrier(
        &nvshmemi_boot_handle); /* To ensure neccessary setup has been done all PEs */

    nvshmemi_team_update_device();

    nvshmemi_boot_handle.barrier(
        &nvshmemi_boot_handle); /* To ensure neccessary setup has been done all PEs */

#if defined(NVSHMEM_PPC64LE)
    if (nvshmemi_use_nccl) {
        /* Set GPU thread stack size to be max stack size of any kernel invoked by NCCL.
           The value 1256 has been obtained by profiling all NCCL kernels in NCCL 2.8.3-1.
           This value is being set to prevent any memory config during application run
           as that can lead to potential deadlock */
        if (nvshmemi_options.CUDA_LIMIT_STACK_SIZE_provided) {
            CUDA_RUNTIME_CHECK(
                cudaDeviceSetLimit(cudaLimitStackSize, nvshmemi_options.CUDA_LIMIT_STACK_SIZE));
            if (nvshmemi_options.CUDA_LIMIT_STACK_SIZE < 1256)
                NVSHMEMI_WARN_PRINT(
                    "CUDA stack size limit has been set to less than 1256.\n"
                    "This can lead to hangs because a NCCL kernel can need up\n"
                    "to 1256 bytes");
        } else
            CUDA_RUNTIME_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 1256));
    } else if (nvshmemi_options.CUDA_LIMIT_STACK_SIZE_provided) {
        CUDA_RUNTIME_CHECK(
            cudaDeviceSetLimit(cudaLimitStackSize, nvshmemi_options.CUDA_LIMIT_STACK_SIZE));
    }
#endif

cleanup:
    if (scratch) {
        free(scratch);
    }
    if (hostHash) {
        free(hostHash);
    }

    if (status != NVSHMEMX_SUCCESS) {
        if (nvshmemi_team_pool) {
            free(nvshmemi_team_pool);
            nvshmemi_team_pool = NULL;
            cudaFree(nvshmemi_device_team_pool);
            nvshmemi_device_team_pool = NULL;
        }
        if (nvshmemi_psync_pool) {
            nvshmemi_free(nvshmemi_psync_pool);
            nvshmemi_psync_pool = NULL;
        }
        if (psync_pool_avail) {
            free(psync_pool_avail);
            psync_pool_avail = NULL;
        }
        if (device_psync_pool_avail) {
            nvshmemi_free(device_psync_pool_avail);
            device_psync_pool_avail = NULL;
        }
        if (team_ret_val) {
            free(team_ret_val);
            team_ret_val = NULL;
        }
        if (device_team_ret_val) {
            nvshmemi_free(device_team_ret_val);
            device_team_ret_val = NULL;
        }
    }

    return status;
}

int nvshmemi_team_finalize(void) {
    /* Destroy all undestroyed teams */
    for (long i = 0; i < nvshmemi_max_teams; i++) {
        if (nvshmemi_team_pool[i] != NULL) nvshmemi_team_destroy(nvshmemi_team_pool[i]);
    }

    free(nvshmemi_team_pool);
    nvshmemi_team_pool = NULL;
    CUDA_RUNTIME_CHECK(cudaFree(nvshmemi_device_team_pool));

    nvshmemi_free(nvshmemi_psync_pool);
    nvshmemi_free(nvshmemi_sync_counter);

    free(psync_pool_avail);
    nvshmemi_free(device_psync_pool_avail);
    free(team_ret_val);
    nvshmemi_free(device_team_ret_val);
    nvshmemi_free(nvshmemi_team_creation_psync);
    nvshmemi_team_creation_psync = NULL;
    cudaFree(nvshmemi_device_team_world);
    cudaFree(nvshmemi_device_team_shared);
    cudaFree(nvshmemi_device_team_node);
    cudaFree(nvshmemi_device_team_same_mype_node);
    cudaFree(nvshmemi_device_team_same_gpu);
    cudaFree(nvshmemi_device_team_gpu_leaders);

    return 0;
}

/* Begin Team Allocation Functions */
int nvshmemi_team_split_node(nvshmemi_team_t *parent_team, nvshmem_team_t *new_team);
int nvshmemi_team_split_same_mype_node(nvshmemi_team_t *parent_team, nvshmem_team_t *new_team);

int nvshmemi_team_check_collective_error(nvshmem_team_t team_idx) {
    /* This OR reduction assures all PEs return the same value.  */
    CUDA_RUNTIME_CHECK(
        cudaMemcpy(device_team_ret_val, team_ret_val, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
    nvshmemi_call_rdxn_on_stream_kernel<int, RDXN_OPS_MAX>(
        team_idx, device_team_ret_val_reduced, device_team_ret_val, 1, nvshmemi_state->my_stream);
    CUDA_RUNTIME_CHECK(cudaStreamSynchronize(nvshmemi_state->my_stream));
    CUDA_RUNTIME_CHECK(cudaMemcpy(team_ret_val_reduced, device_team_ret_val_reduced, sizeof(int),
                                  cudaMemcpyDeviceToHost));

    return *team_ret_val_reduced;
}

int nvshmemi_team_create_internal_teams(nvshmemi_team_t *myteam) {
    int status = NVSHMEMX_SUCCESS;
    if (myteam->is_team_node || myteam->is_team_same_mype_node) {
        return status;
    }

    if (nvshmemi_team_is_subset(myteam, nvshmemi_team_node)) {
        myteam->is_team_node = true;
    }

    if (nvshmemi_team_is_subset(myteam, nvshmemi_team_same_mype_node)) {
        myteam->is_team_same_mype_node = true;
    }

    if (myteam->is_team_node || myteam->is_team_same_mype_node) {
        return status;
    }

    nvshmemi_barrier(myteam->team_idx);

    if (nvshmemi_team_split_node(myteam, &myteam->team_node) != 0) {
        NVSHMEMI_ERROR_PRINT("Failed to split node for team %d\n", myteam->team_idx);
        status = NVSHMEMX_ERROR_INTERNAL;
    }

    nvshmemi_barrier(myteam->team_idx);

    if (nvshmemi_team_split_same_mype_node(myteam, &myteam->team_same_mype_node) != 0) {
        NVSHMEMI_ERROR_PRINT("Failed to split same mype node for team %d\n", myteam->team_idx);
        status = NVSHMEMX_ERROR_INTERNAL;
    }

    nvshmemi_barrier(myteam->team_idx);

    return status;
}

static void nvshmemi_team_reset_psync(nvshmemi_team_t *myteam) {
    long *psync = &nvshmemi_team_get_psync(myteam, SYNC)[NVSHMEMI_SYNC_SIZE];
    long *sync_counter = &nvshmemi_team_get_sync_counter(myteam)[1];

    nvshmemi_call_init_array_kernel<long>(sync_counter, 1, 1);
    nvshmemi_call_init_array_kernel<long>(psync, NVSHMEMI_SYNC_SIZE, NVSHMEMI_SYNC_VALUE);
    CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
}

int nvshmemi_team_set_team_idx_v1(nvshmemi_team_t *myteam, nvshmemi_team_t *mydeviceteam,
                                  nvshmemi_team_t *parent_team) {
    INFO(NVSHMEM_COLL, "entering nvshmemi_team_set_team_idx_v1\n");
    int status = NVSHMEMX_SUCCESS;
    long *psync_reduce = nvshmemi_team_get_psync(parent_team, REDUCE);
    long *psync = &nvshmemi_team_get_psync(parent_team, SYNC)[NVSHMEMI_SYNC_SIZE];
    long *sync_counter = &nvshmemi_team_get_sync_counter(parent_team)[1];
    int team_idx = TEAM_SCALAR_INVALID;

    char bit_str[NVSHMEMI_DIAG_STRLEN];

    assert(myteam->stride != TEAM_SCALAR_INVALID);

    nvshmemi_bit_to_string(bit_str, NVSHMEMI_DIAG_STRLEN, psync_pool_avail, N_PSYNC_BYTES);

    CUDA_RUNTIME_CHECK(cudaMemcpy(device_psync_pool_avail, psync_pool_avail, N_PSYNC_BYTES,
                                  cudaMemcpyHostToDevice));
    CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
    nvshmemi_call_reduce_kernel<unsigned char, RDXN_OPS_AND>(
        myteam->start, myteam->stride, myteam->size,
        (unsigned char *)device_psync_pool_avail_reduced,
        (const unsigned char *)device_psync_pool_avail, N_PSYNC_BYTES,
        (unsigned char *)psync_reduce, (long *)(psync), sync_counter);

    CUDA_RUNTIME_CHECK(cudaMemcpy(psync_pool_avail_reduced, device_psync_pool_avail_reduced,
                                  N_PSYNC_BYTES, cudaMemcpyDeviceToHost));

    /* We cannot release the psync here, because this reduction may not
     * have been performed on the entire parent team. */
    nvshmemi_bit_to_string(bit_str, NVSHMEMI_DIAG_STRLEN, psync_pool_avail_reduced, N_PSYNC_BYTES);

    /* Select the least signficant nonzero bit, which corresponds to an available pSync. */
    team_idx = nvshmemi_bit_1st_nonzero(psync_pool_avail_reduced, N_PSYNC_BYTES);

    nvshmemi_bit_to_string(bit_str, NVSHMEMI_DIAG_STRLEN, psync_pool_avail_reduced, N_PSYNC_BYTES);
    if (team_idx == -1 || team_idx >= (int)nvshmemi_max_teams) {
        /* No psync was available, but must call barrier across parent team before returning. */
        team_idx = -1;
        *team_ret_val = 1;
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                           "No more teams available (max = %ld), try setting NVSHMEM_MAX_TEAMS "
                           "environment variable\n",
                           nvshmemi_max_teams);
    } else {
        /* Set the selected psync bit to 0, reserving that slot */
        myteam->team_idx = team_idx;
        nvshmemi_bit_clear(psync_pool_avail, N_PSYNC_BYTES, myteam->team_idx);
        NVSHMEMI_TEAM_DUP_INITIALIZER(myteam, myteam->team_idx);
        nvshmemi_team_pool[myteam->team_idx] = myteam;
        copy_team_to_device(myteam, mydeviceteam);
        CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
    }

out:
    INFO(NVSHMEM_COLL, "exiting nvshmemi_team_set_team_idx_v1 for team %d with status %d\n",
         myteam->team_idx, status);
    return status;
}

int nvshmemi_copy_internal_team_pe_info(nvshmemi_team_t *myteam, nvshmemi_team_t *mydeviceteam) {
    int status = NVSHMEMX_SUCCESS;
    nvshmemi_team_creation_pe_info *team_info = NULL;

    team_info = (nvshmemi_team_creation_pe_info *)calloc(1, sizeof(nvshmemi_team_creation_pe_info));
    NVSHMEMI_NULL_ERROR_JMP(team_info, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Failed to allocate team_info\n");

    team_info->pe_in_team = myteam->my_pe;
    team_info->team_index_array =
        nvshmemi_get_pe_info_array_ptr(nvshmemi_team_creation_psync, nvshmemi_state->mype);

    if (nvshmemi_device_state.symmetric_heap_kind == NVSHMEMI_HEAP_KIND_SYSMEM) {
        memcpy(&nvshmemi_team_creation_psync->pe_info[nvshmemi_state->mype], team_info,
               sizeof(nvshmemi_team_creation_pe_info));
    } else {
        CUDA_RUNTIME_CHECK(cudaMemcpy(&nvshmemi_team_creation_psync->pe_info[nvshmemi_state->mype],
                                      team_info, sizeof(nvshmemi_team_creation_pe_info),
                                      cudaMemcpyHostToDevice));
    }

    CUDA_RUNTIME_CHECK(cudaMemcpy(team_info->team_index_array, psync_pool_avail, N_PSYNC_BYTES,
                                  cudaMemcpyHostToDevice));

    CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

out:
    INFO(NVSHMEM_COLL, "exiting nvshmemi_copy_internal_team_pe_info with status %d\n", status);
    free(team_info);
    return status;
}

static int nvshmemi_team_populate_from_uid(nvshmemi_team_t *myteam, nvshmemi_team_t *mydeviceteam,
                                           nvshmemi_team_uniqueid_t team_uniqueid, int npes,
                                           int my_pe_idx_in_team) {
    int status = NVSHMEMX_SUCCESS;

    char bit_str[NVSHMEMI_DIAG_STRLEN];

    nvshmemi_bit_to_string(bit_str, NVSHMEMI_DIAG_STRLEN, psync_pool_avail, N_PSYNC_BYTES);
    INFO(NVSHMEM_COLL, "in nvshmemi_team_populate_from_uid, psync_pool_avail: %s\n", bit_str);

    status = nvshmemi_copy_internal_team_pe_info(myteam, mydeviceteam);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                          "Failed to copy internal team pe info\n");

    nvshmemi_call_team_mapping_kernel(team_uniqueid, npes,
                                      NVSHMEMI_DEVICE_TEAM_PE_LOCATION(mydeviceteam),
                                      nvshmemi_team_creation_psync);

    copy_team_pe_mapping_to_host(mydeviceteam, myteam);
    nvshmemi_team_populate_from_world_pe_mapping(myteam);
    myteam->stride = nvshmemi_team_get_stride(myteam);

    /* Copy team to device */
    copy_team_to_device(myteam, mydeviceteam);

out:
    INFO(NVSHMEM_COLL, "exiting nvshmemi_team_populate_from_uid with status %d\n", status);
    return status;
}

int nvshmemi_team_set_team_idx_v2(nvshmemi_team_t *myteam, nvshmemi_team_t *mydeviceteam) {
    int status = NVSHMEMX_SUCCESS;
    int team_idx = TEAM_SCALAR_INVALID;

    INFO(NVSHMEM_COLL, "entering nvshmemi_team_set_team_idx_v2\n");
    assert(myteam->config.version != NVSHMEMI_TEAM_CONFIG_VERSION_1_IDENTIFIER);

    nvshmemi_team_populate_from_uid(myteam, mydeviceteam, myteam->config.uniqueid, myteam->size,
                                    myteam->my_pe);

    nvshmemi_call_team_index_kernel(mydeviceteam, nvshmemi_team_creation_psync, N_PSYNC_BYTES);

    CUDA_RUNTIME_CHECK(
        cudaMemcpy(&team_idx, &mydeviceteam->team_idx, sizeof(int), cudaMemcpyDeviceToHost));

    if (team_idx >= nvshmemi_max_teams || team_idx < 0) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                           "Team index %d out of bounds\n", team_idx);
    }

    if (nvshmemi_team_pool[team_idx] != NULL) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                           "Team index already in use\n");
    }

    /* Set the selected psync bit to 0, reserving that slot */
    myteam->team_idx = team_idx;
    nvshmemi_bit_clear(psync_pool_avail, N_PSYNC_BYTES, myteam->team_idx);
    NVSHMEMI_TEAM_DUP_INITIALIZER(myteam, myteam->team_idx);
    nvshmemi_team_pool[myteam->team_idx] = myteam;
    copy_team_to_device(myteam, mydeviceteam);
    nvshmemi_barrier(myteam->team_idx);
out:
    nvshmemi_reset_team_creation_psync();
    INFO(NVSHMEM_COLL, "exiting nvshmemi_team_set_team_idx_v2 for team %d with status %d\n",
         myteam->team_idx, status);
    return status;
}

int nvshmemi_team_set_team_idx(nvshmemi_team_t *myteam, nvshmemi_team_t *mydeviceteam,
                               nvshmemi_team_t *parent_team) {
    int status = NVSHMEMX_SUCCESS;
    if (myteam->stride == TEAM_SCALAR_INVALID) {
        status = nvshmemi_team_set_team_idx_v2(myteam, mydeviceteam);
        return status;
    } else {
        return nvshmemi_team_set_team_idx_v1(myteam, mydeviceteam, parent_team);
    }
}

/* This must be called after the team has been populated*/
int nvshmemi_team_allocate_resources(nvshmemi_team_t *myteam, nvshmemi_team_t *mydeviceteam,
                                     nvshmemi_team_t *parent_team, nvshmem_team_config_t *config,
                                     long config_mask) {
    int status = NVSHMEMX_SUCCESS;

    myteam->rdxn_count = 0;
    myteam->ll_flag = 1;
    myteam->alltoall_count = 0;
    myteam->bcast_count = 0;
    myteam->bcast_sync_offset = 0;
    myteam->fcollect_count = 0;

    if (parent_team) {
        if (parent_team->is_team_node) {
            myteam->is_team_node = true;
        }

        if (parent_team->is_team_same_mype_node) {
            myteam->is_team_same_mype_node = true;
        }
    }

    if (nvshmemi_team_set_team_idx(myteam, mydeviceteam, parent_team) != NVSHMEMX_SUCCESS) {
        NVSHMEMI_ERROR_PRINT("Failed to set team index for team %d\n", myteam->team_idx);
        status = NVSHMEMX_ERROR_INTERNAL;
        goto out;
    }
#ifdef NVSHMEM_USE_NCCL
    if (nvshmemi_use_nccl) nvshmemi_team_init_nccl_comm(myteam);
#endif
    /* Set Team Characteristics Start */
    nvshmemi_team_set_p2p_connectivity(myteam);
    nvshmemi_recexchalgo_get_neighbors(myteam);
    /* Set Team Characteristics End */

    /* Allocate NVLS resources Start */
    /*
     * Reuse NVLS resources if teams are identical,
     * else creating a new NVLS resources for p2p connected teams
     */
    if (nvshmemi_team_is_nvls_capable(myteam)) {
        INFO(NVSHMEM_COLL, "Setting up NVLS resources for team %d\n", myteam->team_idx);
        if (nvshmemi_team_setup_nvls(myteam) != 0) {
            NVSHMEMI_WARN_PRINT("NVLS resource setup failed for team ID: %d\n", myteam->team_idx);
            status = NVSHMEMX_ERROR_INTERNAL;
            goto out;
        }
    } else {
        myteam->nvls_rsc = nullptr; /* NVLS not supported, so no resource created/bound */
    }
    /* Allocate NVLS resources End */

    /* Allocate Internal Teams Start */
    if (nvshmemi_team_create_internal_teams(myteam) != NVSHMEMX_SUCCESS) {
        NVSHMEMI_ERROR_PRINT("Failed to create internal teams for team %d\n", myteam->team_idx);
        status = NVSHMEMX_ERROR_INTERNAL;
        goto out;
    }
    /* Allocate Internal Teams End */

    /* Copy team to device */
    copy_team_to_device(myteam, mydeviceteam);
out:
    if (parent_team) {
        nvshmemi_team_reset_psync(parent_team);
    }
    return status;
}

/* TODO: Make this function support arbitrary team creation */
int nvshmemi_team_initialize_from_overlap(nvshmemi_team_t *parent_team,
                                          nvshmemi_team_t *overlapping_team,
                                          nvshmemi_team_t **new_team,
                                          nvshmemi_team_t **mydeviceteam) {
    int pe_in_parent_team_index = -1;
    int pe_in_overlapping_team_index = -1;

    int stride = TEAM_SCALAR_INVALID;
    int new_stride = TEAM_SCALAR_INVALID;
    int start = TEAM_SCALAR_INVALID;
    int size = 0;
    int my_pe = TEAM_SCALAR_INVALID;
    int status = NVSHMEMX_SUCCESS;

    bool stride_is_invalid = false;
    bool is_reversed = false;

    int pes_in_team[parent_team->size];

    /* Get PE Indexes for same mype node. Note, the outer loop is over the parent team to preserve
     * parent team's order which may not match the order of the same mype node team.
     */
    for (int parent_idx = 0; parent_idx < parent_team->size; parent_idx++) {
        pe_in_parent_team_index =
            nvshmemi_team_translate_pe_to_team_world_wrap(parent_team, parent_idx);

        for (int overlapping_idx = 0; overlapping_idx < overlapping_team->size; overlapping_idx++) {
            pe_in_overlapping_team_index =
                nvshmemi_team_translate_pe_to_team_world_wrap(overlapping_team, overlapping_idx);

            if (pe_in_overlapping_team_index == pe_in_parent_team_index) {
                pes_in_team[size] = pe_in_parent_team_index;

                if (nvshmemi_state->mype == pe_in_parent_team_index) {
                    my_pe = size;
                }

                if (stride != TEAM_SCALAR_INVALID) {
                    new_stride = pe_in_parent_team_index - pes_in_team[size - 1];
                    if (is_reversed) {
                        new_stride = new_stride * -1;
                    }
                }
                /* Assign start and stride to new team */
                if (start == TEAM_SCALAR_INVALID) {
                    start = pe_in_parent_team_index;
                } else if (stride == TEAM_SCALAR_INVALID && !stride_is_invalid) {
                    stride = pe_in_parent_team_index - start;
                    if (stride < 0) {
                        is_reversed = true;
                        stride = stride * -1;
                    }
                } else if (stride != new_stride) {
                    /* TODO: Remove when arbitrary team creation is supported */
                    stride = TEAM_SCALAR_INVALID;
                    stride_is_invalid = true;
                }
                size++;
                break;
            }
        }
    }

    if (size == 1) {
        stride = 1;
    }
    if (is_reversed && stride != TEAM_SCALAR_INVALID) {
        stride = stride * -1;
    }
    if (start == TEAM_SCALAR_INVALID || size == 0) {
        NVSHMEMI_ERROR_PRINT("Failed to initialize team from overlap\n");
        return NVSHMEMX_ERROR_INTERNAL;
    }

    if (parent_team->config.version == NVSHMEMI_TEAM_CONFIG_VERSION_1_IDENTIFIER &&
        stride == TEAM_SCALAR_INVALID) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                           "Stride is invalid for version 1 team\n");
        return NVSHMEMX_ERROR_INTERNAL;
    }

    if (nvshmemi_team_allocate_team(new_team, mydeviceteam, size) != NVSHMEMX_SUCCESS) {
        NVSHMEMI_ERROR_PRINT("Failed to allocate team\n");
        return NVSHMEMX_ERROR_OUT_OF_MEMORY;
    }

    (*new_team)->start = start;
    (*new_team)->stride = stride;
    (*new_team)->size = size;
    (*new_team)->my_pe = my_pe;

    /* Follow the proper initialization path based on the parent team's version */
    if (parent_team->config.version == NVSHMEMI_TEAM_CONFIG_VERSION_1_IDENTIFIER) {
        (*new_team)->config.version = NVSHMEMI_TEAM_CONFIG_VERSION_1_IDENTIFIER;
        (*new_team)->config.num_contexts = TEAM_CONFIG_SCALAR_INVALID;
    } else {
        (*new_team)->config = NVSHMEMI_TEAM_CONFIG_INITIALIZER;
        (*new_team)->config.uniqueid = parent_team->config.uniqueid;
    }
    (*new_team)->config.num_contexts = parent_team->config.num_contexts;

    memcpy((*new_team)->pe_mapping, pes_in_team, (size) * sizeof(int));
    nvshmemi_team_populate_from_world_pe_mapping(*new_team);
out:
    if (status != NVSHMEMX_SUCCESS) {
        nvshmemi_team_destroy(*new_team);
        *new_team = NULL;
    }
    return status;
}

int nvshmemi_team_split_same_mype_node(nvshmemi_team_t *parent_team, nvshmem_team_t *new_team) {
    nvshmemi_team_t *myteam, *mydeviceteam;

    int status = NVSHMEMX_SUCCESS;

    *new_team = NVSHMEM_TEAM_INVALID;

    INFO(NVSHMEM_COLL, "entering nvshmemi_team_split_same_mype_node with parent ID %d\n",
         parent_team->team_idx);
    if (nvshmemi_team_initialize_from_overlap(parent_team, nvshmemi_team_same_mype_node, &myteam,
                                              &mydeviceteam) != 0) {
        status = NVSHMEMX_ERROR_INTERNAL;
        goto cleanup;
    }

    myteam->is_team_same_mype_node = true;

    for (int i = 0; i < nvshmemi_state->npes_node; i++) {
        if (nvshmemi_state->mype_node == i) {
            myteam->config.uniqueid = parent_team->config.uniqueid + nvshmemi_state->mype_node;
            if (nvshmemi_team_allocate_resources(myteam, mydeviceteam, parent_team, NULL, 0) != 0) {
                status = NVSHMEMX_ERROR_INTERNAL;
            }
        }
        nvshmem_quiet();
        nvshmem_team_sync(parent_team->team_idx);
    }

cleanup:
    if (status != NVSHMEMX_SUCCESS) {
        if (myteam != NULL) {
            free(myteam);
        }
    } else {
        *new_team = myteam->team_idx;
        parent_team->team_same_mype_node = myteam->team_idx;
    }

    INFO(NVSHMEM_COLL,
         "exiting nvshmemi_team_split_same_mype_node with parent ID %d and new team ID %d and "
         "status %d\n",
         parent_team->team_idx, *new_team, status);

    return status;
}

int nvshmemi_team_split_node(nvshmemi_team_t *parent_team, nvshmem_team_t *new_team) {
    nvshmemi_team_t *myteam, *mydeviceteam;
    int status = NVSHMEMX_SUCCESS;

    INFO(NVSHMEM_COLL, "entering nvshmemi_team_split_node with parent ID %d\n",
         parent_team->team_idx);

    *new_team = NVSHMEM_TEAM_INVALID;

    if (nvshmemi_team_initialize_from_overlap(parent_team, nvshmemi_team_node, &myteam,
                                              &mydeviceteam) != 0) {
        status = NVSHMEMX_ERROR_INTERNAL;
        goto cleanup;
    }

    myteam->is_team_node = true;

    /* Do node creation in sequence to avoid race condition */
    for (int i = 0; i < nvshmemi_state->npes / nvshmemi_state->npes_node; i++) {
        uint64_t my_node_idx = nvshmemi_state->mype / nvshmemi_state->npes_node;
        myteam->config.uniqueid = parent_team->config.uniqueid + (my_node_idx << 32);
        if (nvshmemi_state->mype / nvshmemi_state->npes_node == i) {
            if (nvshmemi_team_allocate_resources(myteam, mydeviceteam, parent_team, NULL, 0) != 0) {
                status = NVSHMEMX_ERROR_INTERNAL;
            }
        }
        nvshmem_quiet();
        nvshmem_team_sync(parent_team->team_idx);
    }

cleanup:
    if (status != NVSHMEMX_SUCCESS) {
        if (myteam != NULL) {
            nvshmemi_team_destroy(myteam);
        }
    } else {
        *new_team = myteam->team_idx;
        parent_team->team_node = myteam->team_idx;
    }

    INFO(NVSHMEM_COLL,
         "exiting nvshmemi_team_split_node with parent ID %d and new team ID %d and status %d\n",
         parent_team->team_idx, *new_team, status);
    return status;
}

int nvshmemi_team_get_uniqueid(nvshmemx_team_uniqueid_t *uniqueid) {
    NVSHMEMI_CHECK_INIT_STATUS();
    NVSHMEM_API_NOT_SUPPORTED_WITH_LIMITED_MPG_RUNS();

    assert(uniqueid != NULL);

    *uniqueid = nvshmemi_team_populate_uniqueid();
    return NVSHMEMX_SUCCESS;
}

int nvshmemi_team_create(nvshmem_team_t *team, nvshmem_team_config_t *config, long config_mask,
                         int npes, int my_pe_idx_in_team) {
    *team = NVSHMEM_TEAM_INVALID;
    int status = NVSHMEMX_SUCCESS;
    nvshmemi_team_t *my_team, *my_device_team;

    if (config == NULL) {
        NVSHMEMI_ERROR_PRINT("Unable to initialize team with NULL config\n");
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    if (config->version == NVSHMEMI_TEAM_CONFIG_VERSION_1_IDENTIFIER) {
        NVSHMEMI_ERROR_PRINT("Version 1 team config is not supported\n");
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    if (config->uniqueid == TEAM_ULSCALAR_INVALID) {
        NVSHMEMI_ERROR_PRINT("Unique ID is not initialized\n");
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    if (nvshmemi_team_allocate_team(&my_team, &my_device_team, npes) != NVSHMEMX_SUCCESS) {
        NVSHMEMI_ERROR_PRINT("Failed to allocate my_team\n");
        return NVSHMEMX_ERROR_OUT_OF_MEMORY;
    }

    my_team->my_pe = my_pe_idx_in_team;
    my_team->size = npes;
    my_team->config = *config;
    my_team->config_mask = config_mask;

    if (nvshmemi_team_allocate_resources(my_team, my_device_team, NULL, NULL, 0) != 0) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                           "Failed to allocate my_team\n");
    }
out:
    if (status != NVSHMEMX_SUCCESS) {
        nvshmemi_team_destroy(my_team);
        CUDA_RUNTIME_CHECK(cudaFree(my_device_team));
    } else {
        nvshmem_quiet();
        nvshmem_team_sync(my_team->team_idx);
        *team = my_team->team_idx;
    }

    return status;
}

int nvshmemi_team_split_from_non_strided_parent(nvshmemi_team_t *parent_team, int PE_start,
                                                int PE_stride, int PE_size,
                                                const nvshmem_team_config_t *config,
                                                long config_mask, nvshmem_team_t *new_team) {
    nvshmemi_team_t *myteam = NULL;
    nvshmemi_team_t *mydeviceteam = NULL;
    int my_pe = TEAM_SCALAR_INVALID;
    *team_ret_val = NVSHMEMX_SUCCESS;
    bool in_team = false;

    INFO(NVSHMEM_COLL, "entering nvshmemi_team_split_from_non_strided_parent with parent ID %d\n",
         parent_team->team_idx);
    if (config == NULL &&
        parent_team->config.version == NVSHMEMI_TEAM_CONFIG_VERSION_2_IDENTIFIER) {
        ;
    } else if (config == NULL) {
        NVSHMEMI_ERROR_PRINT("Unable to initialize unstrided team with NULL config\n");
        *team_ret_val = NVSHMEMX_ERROR_INVALID_VALUE;
        goto out;
    } else if (config->version == NVSHMEMI_TEAM_CONFIG_VERSION_1_IDENTIFIER) {
        NVSHMEMI_ERROR_PRINT("Version 1 team config on unstrided team is not supported\n");
        *team_ret_val = NVSHMEMX_ERROR_INVALID_VALUE;
        goto out;
    } else if (config->uniqueid == TEAM_ULSCALAR_INVALID) {
        NVSHMEMI_ERROR_PRINT("Unique ID not initialized. This is required for unstrided teams\n");
        *team_ret_val = NVSHMEMX_ERROR_INVALID_VALUE;
        goto out;
    }

    for (int i = PE_start; i < PE_start + PE_size; i++) {
        int parent_team_index = PE_start + i * PE_stride;
        int team_world_index = parent_team->pe_mapping[parent_team_index];

        if (team_world_index == nvshmemi_state->mype) {
            in_team = true;
            my_pe = i;
            break;
        }
    }

    if (in_team) {
        *team_ret_val = nvshmemi_team_allocate_team(&myteam, &mydeviceteam, PE_size);
        if (*team_ret_val != NVSHMEMX_SUCCESS) {
            NVSHMEMI_ERROR_PRINT("Failed to allocate myteam\n");
            goto out;
        }

        if (config == NULL) {
            myteam->config = NVSHMEMI_TEAM_CONFIG_INITIALIZER;
            myteam->config.version = NVSHMEMI_TEAM_CONFIG_VERSION_2_IDENTIFIER;
            myteam->config.uniqueid = parent_team->config.uniqueid;
            myteam->config.num_contexts = parent_team->config.num_contexts;
            myteam->config_mask = parent_team->config_mask;
        } else {
            myteam->config = *config;
            myteam->config_mask = config_mask;
        }
        myteam->my_pe = my_pe;
        myteam->size = PE_size;

        if (nvshmemi_team_allocate_resources(myteam, mydeviceteam, parent_team, NULL, 0) != 0) {
            *team_ret_val = NVSHMEMX_ERROR_INTERNAL;
            NVSHMEMI_ERROR_PRINT("Failed to initialize myteam\n");
            goto out;
        }
    }

out:
    nvshmem_quiet();
    nvshmem_team_sync(parent_team->team_idx);

    if (nvshmemi_team_check_collective_error(parent_team->team_idx) != 0) {
        /* If no team was available, print some team triplet info and return nonzero. */
        if (myteam != NULL && myteam->team_idx == -1) {
            NVSHMEMI_WARN_PRINT(
                "Team split strided failed: child <%d, %d, %d>, parent <%d, %d, %d>\n",
                myteam->start, myteam->stride, myteam->size, parent_team->start,
                parent_team->stride, parent_team->size);
        }
    }

    if (*team_ret_val_reduced != NVSHMEMX_SUCCESS) {
        nvshmemi_team_destroy(myteam);
        CUDA_RUNTIME_CHECK(cudaFree(mydeviceteam));
    } else {
        *new_team = myteam->team_idx;
    }
    INFO(NVSHMEM_COLL,
         "exiting nvshmemi_team_split_from_non_strided_parent with parent ID %d and new team ID %d "
         "and status %d\n",
         parent_team->team_idx, *new_team, *team_ret_val_reduced);
    return *team_ret_val_reduced;
}

int nvshmemi_team_split_strided(nvshmemi_team_t *parent_team, int PE_start, int PE_stride,
                                int PE_size, const nvshmem_team_config_t *config, long config_mask,
                                nvshmem_team_t *new_team) {
    int global_PE_start = -1;
    int global_PE_stride = -1;
    int global_PE_end = -1;
    int my_pe = -1;

    *new_team = NVSHMEM_TEAM_INVALID;

    nvshmemi_team_t *myteam = NULL;
    nvshmemi_team_t *mydeviceteam = NULL;
    *team_ret_val = NVSHMEMX_SUCCESS;
    *team_ret_val_reduced = NVSHMEMX_SUCCESS;

    if (PE_start < 0 || PE_start >= parent_team->size || PE_size <= 0 ||
        PE_size > parent_team->size || PE_stride < 1) {
        NVSHMEMI_ERROR_JMP(
            *team_ret_val_reduced, NVSHMEMX_ERROR_INVALID_VALUE, out,
            "Invalid <start, stride, size>: child <%d, %d, %d>, parent <%d, %d, %d>\n", PE_start,
            PE_stride, PE_size, parent_team->start, parent_team->stride, parent_team->size);
    }

    if (parent_team->stride == TEAM_SCALAR_INVALID) {
        my_pe = nvshmemi_team_translate_pe_from_team_world(parent_team, nvshmemi_state->mype);
        *team_ret_val_reduced = nvshmemi_team_split_from_non_strided_parent(
            parent_team, PE_start, PE_stride, PE_size, config, config_mask, new_team);
        if (*team_ret_val_reduced != NVSHMEMX_SUCCESS) {
            NVSHMEMI_ERROR_PRINT("Failed to split from non-strided parent team\n");
        }
        goto out_clean;
    }
    global_PE_start = nvshmemi_team_pe(parent_team, PE_start);
    global_PE_stride = parent_team->stride * PE_stride;
    global_PE_end = global_PE_start + global_PE_stride * (PE_size - 1);

    if (global_PE_start >= nvshmemi_state->npes || global_PE_end >= nvshmemi_state->npes) {
        NVSHMEMI_ERROR_JMP(*team_ret_val, NVSHMEMX_ERROR_INVALID_VALUE, out,
                           "Starting PE (%d) or ending PE (%d) is invalid\n", global_PE_start,
                           global_PE_end);
    }

    /* idx in new team: */
    my_pe =
        nvshmemi_pe_in_active_set(nvshmemi_state->mype, global_PE_start, global_PE_stride, PE_size);

    nvshmemi_barrier(parent_team->team_idx);

    if (my_pe >= 0) {
        if (nvshmemi_team_allocate_team(&myteam, &mydeviceteam, PE_size) != NVSHMEMX_SUCCESS) {
            *team_ret_val = NVSHMEMX_ERROR_OUT_OF_MEMORY;
            goto out;
        }

        myteam->my_pe = my_pe;
        myteam->start = global_PE_start;
        myteam->stride = global_PE_stride;
        myteam->size = PE_size;

        if (config) {
            myteam->config = *config;
            myteam->config_mask = config_mask;
        } else {
            myteam->config.version = NVSHMEMI_TEAM_CONFIG_VERSION_1_IDENTIFIER;
            myteam->config.num_contexts = TEAM_CONFIG_SCALAR_INVALID;
        }

        for (int i = 0; i < PE_size; i++) {
            int pe = global_PE_start + i * global_PE_stride;
            myteam->pe_mapping[i] = pe;
            myteam->pe_mapping[pe + myteam->size] = i;
        }

        if (nvshmemi_team_allocate_resources(myteam, mydeviceteam, parent_team, NULL, 0) != 0) {
            *team_ret_val = NVSHMEMX_ERROR_INTERNAL;
        } else {
            *new_team = myteam->team_idx;
        }
    }

out:
    /* Synchronize Start */
    nvshmem_quiet();
    nvshmem_team_sync(parent_team->team_idx);
    /* Synchronize End */

    /* Check Collective Error Start */
    if (nvshmemi_team_check_collective_error(parent_team->team_idx) != 0) {
        /* If no team was available, print some team triplet info and return nonzero. */
        if (myteam != NULL && myteam->team_idx == -1) {
            NVSHMEMI_WARN_PRINT(
                "Team split strided failed: child <%d, %d, %d>, parent <%d, %d, %d>\n",
                myteam->start, myteam->stride, myteam->size, parent_team->start,
                parent_team->stride, parent_team->size);
        }
    }
    /* Check Collective Error End */

    if (*team_ret_val_reduced != NVSHMEMX_SUCCESS) {
        *new_team = NVSHMEM_TEAM_INVALID;
        nvshmemi_team_destroy(myteam);
        CUDA_RUNTIME_CHECK(cudaFree(mydeviceteam));
    }
out_clean:
    return *team_ret_val_reduced;
}

int nvshmemi_team_split_2d(nvshmemi_team_t *parent_team, int xrange,
                           const nvshmem_team_config_t *xaxis_config, long xaxis_mask,
                           nvshmem_team_t *xaxis_team, const nvshmem_team_config_t *yaxis_config,
                           long yaxis_mask, nvshmem_team_t *yaxis_team) {
    *xaxis_team = NVSHMEM_TEAM_INVALID;
    *yaxis_team = NVSHMEM_TEAM_INVALID;

    if (xrange > parent_team->size) {
        xrange = parent_team->size;
    }

    const int parent_size = parent_team->size;
    const int num_xteams = ceil(parent_size / (float)xrange);
    const int num_yteams = xrange;

    int start = 0;
    int ret = 0;

    for (int i = 0; i < num_xteams; i++) {
        nvshmem_team_t my_xteam;
        int xsize = (i == num_xteams - 1 && parent_size % xrange) ? parent_size % xrange : xrange;
        ret = nvshmemi_team_split_strided(parent_team, start, 1, xsize, xaxis_config, xaxis_mask,
                                          &my_xteam);
        if (ret) {
            NVSHMEMI_ERROR_PRINT("Creation of x-axis team %d of %d failed\n", i + 1, num_xteams);
        }
        start += xrange;

        if (my_xteam != NVSHMEM_TEAM_INVALID) {
            assert(*xaxis_team == NVSHMEM_TEAM_INVALID);
            *xaxis_team = my_xteam;
        }
    }

    start = 0;

    for (int i = 0; i < num_yteams; i++) {
        nvshmem_team_t my_yteam;
        int remainder = parent_size % xrange;
        int yrange = parent_size / xrange;
        int ysize = (remainder && i < remainder) ? yrange + 1 : yrange;

        ret = nvshmemi_team_split_strided(parent_team, start, xrange, ysize, yaxis_config,
                                          yaxis_mask, &my_yteam);
        if (ret) {
            NVSHMEMI_ERROR_PRINT("Creation of y-axis team %d of %d failed\n", i + 1, num_yteams);
        }
        start += 1;

        if (my_yteam != NVSHMEM_TEAM_INVALID) {
            assert(*yaxis_team == NVSHMEM_TEAM_INVALID);
            *yaxis_team = my_yteam;
        }
    }

    nvshmem_quiet();
    nvshmem_team_sync(parent_team->team_idx);

    return 0;
}
/* End Team Allocation Functions */

static bool inline nvshmemi_is_rsvd_teams(nvshmem_team_t team_idx) {
    /* This team resource shouldn't not be deleted as they are used for collectives APIs during
     * init/finalize */
    return ((team_idx == NVSHMEM_TEAM_INVALID) ||
            (team_idx >= NVSHMEM_TEAM_WORLD_INDEX && team_idx < NVSHMEM_TEAMS_MIN));
}

void nvshmemi_team_destroy(nvshmemi_team_t *team) {
    int idx;

    if (team == NULL) {
        return;
    }

    idx = team->team_idx;
    if (idx != NVSHMEM_TEAM_INVALID) {
        if (nvshmemi_bit_fetch(psync_pool_avail, idx)) {
            NVSHMEMI_ERROR_PRINT("Destroying team at index[%d] without an active pSync\n", idx);
        }

        for (int i = 0; i < nvshmemi_max_teams; i++) {
            if (nvshmemi_team_pool[i] == team && i != idx) {
                NVSHMEMI_ERROR_PRINT(
                    "the team at index[%d] is already in use at another index[%d].\n", idx, i);
            }
        }
    }

    if (!team->is_team_node) {
        if (!nvshmemi_is_rsvd_teams(team->team_node) &&
            nvshmemi_team_pool[team->team_node] != NULL) {
            INFO(NVSHMEM_COLL,
                 "Destroy sub-team 1 [%p] at index[%d] for parent-team [%p] at index[%d]",
                 nvshmemi_team_pool[team->team_node], team->team_node, team, idx);
            nvshmemi_team_destroy(nvshmemi_team_pool[team->team_node]);
        }
    }

    if (!team->is_team_same_mype_node) {
        if (!nvshmemi_is_rsvd_teams(team->team_same_mype_node) &&
            nvshmemi_team_pool[team->team_same_mype_node] != NULL) {
            INFO(NVSHMEM_COLL,
                 "Destroy sub-team 2 [%p] at index[%d] for parent-team [%p] at index[%d]",
                 nvshmemi_team_pool[team->team_same_mype_node], team->team_same_mype_node, team,
                 idx);
            nvshmemi_team_destroy(nvshmemi_team_pool[team->team_same_mype_node]);
        }
    }

    if (idx != NVSHMEM_TEAM_INVALID) {
        INFO(NVSHMEM_COLL, "cleaning up team pSync at index[%d]", idx);
        nvshmemi_bit_set(psync_pool_avail, N_PSYNC_BYTES, idx);
        nvshmemi_team_pool[idx] = NULL;
        CUDA_RUNTIME_CHECK(
            cudaMemset(&nvshmemi_device_team_pool[idx], 0, sizeof(nvshmemi_team_t *)));

        nvshmemi_call_init_array_kernel<long>(&nvshmemi_sync_counter[2 * idx], 2, 1);
        nvshmemi_call_init_array_kernel<long>(&nvshmemi_psync_pool[idx * get_psync_len_per_team()],
                                              get_psync_len_per_team(), NVSHMEMI_SYNC_VALUE);
        CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

        nvshmemi_team_t *device_team_addr;
        CUDA_RUNTIME_CHECK(cudaMemcpy((void **)&device_team_addr, &nvshmemi_device_team_pool[idx],
                                      sizeof(nvshmemi_team_t *), cudaMemcpyDeviceToHost));
        CUDA_RUNTIME_CHECK(cudaFree(device_team_addr));
    }

    nvshmemi_team_destroy_nvls(team);
    nvshmemi_recexchalgo_free_mem(team);
#ifdef NVSHMEM_USE_NCCL
    if (nvshmemi_use_nccl) NCCL_CHECK(nccl_ftable.CommDestroy((ncclComm_t)team->nccl_comm));
#endif
    if (team != nvshmemi_team_world && team != nvshmemi_team_shared && team != nvshmemi_team_node &&
        team != nvshmemi_team_same_mype_node && team != nvshmemi_team_same_gpu &&
        team != nvshmemi_team_gpu_leaders) {
        free(team);
    }
}

long *nvshmemi_team_get_psync(nvshmemi_team_t *team, nvshmemi_team_op_t op) {
    long *team_psync;
    size_t psync_fcollect_len;
    psync_fcollect_len = get_fcollect_psync_len_per_team();
    team_psync = &nvshmemi_psync_pool[team->team_idx * get_psync_len_per_team()];
    switch (op) {
        case SYNC:
            return team_psync;
        case REDUCE:
            return &team_psync
                [2 * NVSHMEMI_SYNC_SIZE +
                 (((nvshmemi_device_state.gpu_coll_env_params_var.reduce_scratch_size / 2) /
                   sizeof(long)) *
                  (team->rdxn_count % 2))];
        case BCAST:
            return &team_psync[2 * NVSHMEMI_SYNC_SIZE +
                               nvshmemi_device_state.gpu_coll_env_params_var.reduce_scratch_size /
                                   sizeof(long)];
        case FCOLLECT:
            return &team_psync[2 * NVSHMEMI_SYNC_SIZE +
                               nvshmemi_device_state.gpu_coll_env_params_var.reduce_scratch_size /
                                   sizeof(long) +
                               NVSHMEMI_BCAST_SYNC_SIZE];
        case ALLTOALL:
            return &team_psync[2 * NVSHMEMI_SYNC_SIZE +
                               nvshmemi_device_state.gpu_coll_env_params_var.reduce_scratch_size /
                                   sizeof(long) +
                               NVSHMEMI_BCAST_SYNC_SIZE + psync_fcollect_len +
                               (NVSHMEMI_ALLTOALL_SYNC_SIZE * (team->alltoall_count % 2))];
        case FCOLLECT_128:
            return &team_psync[2 * NVSHMEMI_SYNC_SIZE +
                               nvshmemi_device_state.gpu_coll_env_params_var.reduce_scratch_size /
                                   sizeof(long) +
                               NVSHMEMI_BCAST_SYNC_SIZE + psync_fcollect_len +
                               2 * NVSHMEMI_ALLTOALL_SYNC_SIZE];
        default:
            WARN("Incorrect argument to nvshmemi_team_get_psync\n");
            return NULL;
    }
}

long *nvshmemi_team_get_sync_counter(nvshmemi_team_t *team) {
    return &nvshmemi_sync_counter[2 * team->team_idx];
}
