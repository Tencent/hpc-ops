/*
 * Copyright (c) 2016-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

#ifndef __NVSHMEM_TRANSPORT_COMMON_H
#define __NVSHMEM_TRANSPORT_COMMON_H

#if !defined __CUDACC_RTC__
#include <stdint.h>
#include <limits.h>
#else
#include <cuda/std/cstdint>
#include <cuda/std/climits>
#endif

typedef enum {
    NVSHMEMI_OP_PUT = 1,
    NVSHMEMI_OP_P,
    NVSHMEMI_OP_PUT_SIGNAL,
    NVSHMEMI_OP_GET,
    NVSHMEMI_OP_G,
    NVSHMEMI_OP_FENCE,
    NVSHMEMI_OP_AMO,
    NVSHMEMI_OP_QUIET,
    NVSHMEMI_OP_SENTINEL = INT_MAX,
} nvshmemi_op_t;

typedef enum { NVSHMEM_SIGNAL_SET = 9, NVSHMEM_SIGNAL_ADD = 10 } nvshmemx_signal_op_t;

typedef enum {
    NVSHMEMI_AMO_ACK = 1,
    NVSHMEMI_AMO_INC = 2,
    NVSHMEMI_AMO_SET = 3,
    NVSHMEMI_AMO_ADD = 4,
    NVSHMEMI_AMO_AND = 5,
    NVSHMEMI_AMO_OR = 6,
    NVSHMEMI_AMO_XOR = 7,
    NVSHMEMI_AMO_SIGNAL = 8,
    NVSHMEMI_AMO_SIGNAL_SET = NVSHMEM_SIGNAL_SET,  // Note - NVSHMEM_SIGNAL_SET == 9
    NVSHMEMI_AMO_SIGNAL_ADD = NVSHMEM_SIGNAL_ADD,  // Note - NVSHMEM_SIGNAL_ADD == 10
    NVSHMEMI_AMO_END_OF_NONFETCH = 11,             // end of nonfetch atomics
    NVSHMEMI_AMO_FETCH = 12,
    NVSHMEMI_AMO_FETCH_INC = 13,
    NVSHMEMI_AMO_FETCH_ADD = 14,
    NVSHMEMI_AMO_FETCH_AND = 15,
    NVSHMEMI_AMO_FETCH_OR = 16,
    NVSHMEMI_AMO_FETCH_XOR = 17,
    NVSHMEMI_AMO_SWAP = 18,
    NVSHMEMI_AMO_COMPARE_SWAP = 19,
    NVSHMEMI_AMO_OP_SENTINEL = INT_MAX,
} nvshmemi_amo_t;

typedef struct {
    volatile uint64_t data;
    volatile uint64_t flag;
} g_elem_t;

#endif
