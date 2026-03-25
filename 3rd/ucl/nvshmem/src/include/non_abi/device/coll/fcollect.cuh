/*
 * Copyright (c) 2017-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */
#ifndef FCOLLECT_DEVICE_CUH
#define FCOLLECT_DEVICE_CUH

#if !defined __CUDACC_RTC__
#include <stdint.h>
#include <limits.h>
#else
#include "cuda/std/cstdint"
#include <cuda/std/climits>
#endif

#include <cuda_runtime.h>
#include "device_host/nvshmem_common.cuh"
#include "non_abi/device/common/nvshmemi_common_device.cuh"
#include "non_abi/device/common/nvshmemi_tile_utils.cuh"
#include "non_abi/nvshmem_build_options.h"
#include "device_host/nvshmem_tensor.h"
#if defined(NVSHMEM_ENABLE_ALL_DEVICE_INLINING) || defined(__NVSHMEM_NUMBA_SUPPORT__)
#include "non_abi/device/pt-to-pt/transfer_device.cuh"
#else
#include "non_abi/device/pt-to-pt/nvshmemi_transfer_api.cuh"
#endif
#include "non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh"
#include "non_abi/device/common/nvshmemi_common_device.cuh"
#include "non_abi/device/team/nvshmemi_team_defines.cuh"

#define _FCOLLECT_LL8_PSYNC_SCALE_FACTOR 2

#define _FCOLLECT_MAX(x, y) ((x) > (y) ? (x) : (y))
#define _FCOLLECT_MIN(x, y) ((x) < (y) ? (x) : (y))

typedef enum { LL8 = 0, LL128 } ll_version_t;

#ifdef __CUDA_ARCH__

template <typename T, threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_FORCE_INLINE void nvshmemi_fcollect_nvls_ll_threadgroup(
    nvshmem_team_t team, T *dest, const T *source, size_t nelems) {
#if defined __clang_llvm_bitcode_lib__
    if (__nvvm_reflect("__CUDA_ARCH") >= 900) {
        nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
        const size_t fcollect_ll_threshold =
            nvshmemi_device_state_d.gpu_coll_env_params_var.fcollect_ll_threshold;
        const size_t fcollect_count = teami->fcollect_count;
        const uint32_t ll_flag = teami->fcollect_count;
        char *pWrk = (char *)nvshmemi_team_get_psync(teami, FCOLLECT) +
                     (2 * teami->size * fcollect_ll_threshold *
                      (fcollect_count % 2)); /* same for NVLS in terms of size */
        const size_t pack_offset = (nvshmemi_team_my_pe(team) * nelems * sizeof(T)) /
                                   sizeof(uint32_t); /* offset in pSync space */
        /* Find the multicast ptr for pWrk + pack_offset and do a store to remote pSync */
        void *mcast_pWrk = nvshmemi_mc_ptr(teami, (void *)((uint64_t *)pWrk + pack_offset));
        nvshmemi_mcast_packLL<T, SCOPE>((uint64_t *)mcast_pWrk, source, nelems, ll_flag);
        for (int ii = 0; ii < teami->size; ii += 1) {
            size_t prev_offset = (nelems * ii * sizeof(T)) / sizeof(uint32_t);
            nvshmemi_mcast_recvLL<T, SCOPE>(dest + (ii * nelems), (uint64_t *)pWrk + prev_offset,
                                            nelems, ll_flag);
        }

        nvshmemi_threadgroup_sync<SCOPE>();
    } else {
        assert(0 && "NVLink SHARP is not supported on this platform");
    }
#else
#if __CUDA_ARCH__ >= 900 && CUDART_VERSION >= 12010
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    const size_t fcollect_ll_threshold =
        nvshmemi_device_state_d.gpu_coll_env_params_var.fcollect_ll_threshold;
    const size_t fcollect_count = teami->fcollect_count;
    const uint32_t ll_flag = teami->fcollect_count;
    char *pWrk = (char *)nvshmemi_team_get_psync(teami, FCOLLECT) +
                 (2 * teami->size * fcollect_ll_threshold *
                  (fcollect_count % 2)); /* same for NVLS in terms of size */
    const size_t pack_offset = (nvshmemi_team_my_pe(team) * nelems * sizeof(T)) /
                               sizeof(uint32_t); /* offset in pSync space */
    /* Find the multicast ptr for pWrk + pack_offset and do a store to remote pSync */
    void *mcast_pWrk = nvshmemi_mc_ptr(teami, (void *)((uint64_t *)pWrk + pack_offset));
    nvshmemi_mcast_packLL<T, SCOPE>((uint64_t *)mcast_pWrk, source, nelems, ll_flag);
    for (int ii = 0; ii < teami->size; ii += 1) {
        size_t prev_offset = (nelems * ii * sizeof(T)) / sizeof(uint32_t);
        nvshmemi_mcast_recvLL<T, SCOPE>(dest + (ii * nelems), (uint64_t *)pWrk + prev_offset,
                                        nelems, ll_flag);
    }

    nvshmemi_threadgroup_sync<SCOPE>();
#else
    assert(0 && "NVLink SHARP is not supported on this platform");
#endif
#endif
}

/* This function must not ever call a block-scoped synchronization API.
 * See call-site in nvshmemi_fcollect_threadgroup
 */
template <typename T, threadgroup_t SCOPE, ll_version_t LL_VERSION, bool NODE_SAFE>
__device__ NVSHMEMI_DEVICE_ALWAYS_FORCE_INLINE void nvshmemi_fcollect_allpush_ll_threadgroup(
    nvshmem_team_t team, T *dest, const T *source, size_t nelems) {
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    nvshmemi_team_t *teami_node = NULL;
    const size_t fcollect_count = teami->fcollect_count;
    const uint32_t ll_flag = teami->fcollect_count;
    const int my_pe_in_team = teami->my_pe;
    const int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    const int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    const int myWarpIdx = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();
    size_t data_element_offset;
    size_t psync_element_offset;
    size_t psync_remote_write_elements;
    T *peer_addr;

    T *pWrk;
    size_t pack_offset;
    size_t max_data_elems_per_warp;
    size_t max_psync_elems_per_warp;
    int next_pe, start_pe;
    int num_pes_per_group, remaining_pes;
    int num_warp_groups, num_warps_per_group;
    int warp_id, warp_count, warp_group_id;

    if (NODE_SAFE) {
        /*
         * Really we mean team shared here, but that is not a team that exists today.
         * TODO: Replace team_node with team_shared.
         */
        if (teami->are_gpus_p2p_connected) {
            teami_node = teami;
        } else if (teami->team_node != NVSHMEM_TEAM_INVALID &&
                   nvshmemi_device_state_d.team_pool[NVSHMEM_TEAM_SHARED]->is_team_node) {
            teami_node = nvshmemi_device_state_d.team_pool[teami->team_node];
        }
    }

    warp_count = nvshmemi_threadgroup_size<SCOPE>() / 32;
    warp_id = myIdx / 32;

    if (LL_VERSION == LL8) {
        const size_t fcollect_ll_threshold =
            nvshmemi_device_state_d.gpu_coll_env_params_var.fcollect_ll_threshold / sizeof(T);
        pWrk = (T *)nvshmemi_team_get_psync(teami, FCOLLECT) +
               (_FCOLLECT_LL8_PSYNC_SCALE_FACTOR * teami->size * fcollect_ll_threshold *
                (fcollect_count % 2));
        /* round up to 16 bytes*/
        psync_remote_write_elements =
            NVSHMEMI_TEAM_ROUND_UP((nelems * _FCOLLECT_LL8_PSYNC_SCALE_FACTOR), 16 / sizeof(T));
        pack_offset = my_pe_in_team * psync_remote_write_elements;
        if (!NODE_SAFE || !teami->is_team_node) {
            nvshmemi_packLL<T, SCOPE, 1>((T *)(pWrk + pack_offset), source, nelems, ll_flag, teami,
                                         1, my_pe_in_team);
        }
    } else {
        const size_t fcollect_ll_threshold =
            nvshmemi_device_state_d.gpu_coll_env_params_var.fcollect_ll128_threshold / sizeof(T);
        pWrk = (T *)nvshmemi_team_get_psync(teami, FCOLLECT_128) +
               (teami->size * NVSHMEMI_FCOLLECT_LL128_CALC_PSYNC_SIZE(fcollect_ll_threshold, T) *
                (fcollect_count % 2));
        psync_remote_write_elements = NVSHMEMI_FCOLLECT_LL128_CALC_PSYNC_SIZE(nelems, T);
        pack_offset = my_pe_in_team * psync_remote_write_elements;
        if (!NODE_SAFE || !teami->is_team_node) {
            nvshmemi_packLL128<T, SCOPE, 1>(pWrk + pack_offset, source, nelems, ll_flag, teami, 1,
                                            my_pe_in_team, 0);
        }
    }

    /* send out non blocking puts for all remote PEs */
    if (teami_node != teami) {
        for (uint32_t ii = myIdx + 1 + my_pe_in_team; ii < teami->size + my_pe_in_team;
             ii += groupSize) {
            next_pe = nvshmemi_team_translate_pe_to_team_world_wrap(teami, ii);
            if (NODE_SAFE) {
                if (nvshmemi_ptr(pWrk, next_pe) == NULL) {
                    nvshmemi_put_nbi_threadgroup<T, NVSHMEMI_THREADGROUP_THREAD>(
                        pWrk + pack_offset, pWrk + pack_offset, psync_remote_write_elements,
                        next_pe);
                }
            } else {
                nvshmemi_put_nbi_threadgroup<T, NVSHMEMI_THREADGROUP_THREAD>(
                    pWrk + pack_offset, pWrk + pack_offset, psync_remote_write_elements, next_pe);
            }
        }
        nvshmemi_threadgroup_sync<NVSHMEMI_THREADGROUP_WARP>();
    }

    if (LL_VERSION == LL8) {
        max_data_elems_per_warp = _LL_8_DATA_BYTES_PER_WARP * _LL_MAX_UNROLL / sizeof(T);
        max_psync_elems_per_warp = _LL_8_PSYNC_BYTES_PER_WARP * _LL_MAX_UNROLL / sizeof(T);
        if (nelems < max_data_elems_per_warp) {
            if (teami_node) {
                for (uint32_t ii = warp_id + my_pe_in_team; ii < teami_node->size + my_pe_in_team;
                     ii += warp_count) {
                    next_pe = nvshmemi_team_translate_pe_to_team_world_wrap(teami_node, ii);
                    peer_addr = (T *)nvshmemi_ptr(pWrk, next_pe) + pack_offset;
                    nvshmemi_packLL_naive<T, NVSHMEMI_THREADGROUP_WARP>((uint64_t *)peer_addr,
                                                                        source, nelems, ll_flag);
                }
            }

            for (uint32_t ii = warp_id; ii < teami->size; ii += warp_count) {
                next_pe = ii;
                pack_offset =
                    next_pe * NVSHMEMI_TEAM_ROUND_UP((nelems * _FCOLLECT_LL8_PSYNC_SCALE_FACTOR),
                                                     16 / sizeof(T));
                nvshmemi_recvLL<T, NVSHMEMI_THREADGROUP_WARP>(
                    dest + (next_pe * nelems), (uint64_t *)(pWrk + pack_offset), nelems, ll_flag);
            }
            return;
        }
    } else {
        max_data_elems_per_warp = _LL_128_DATA_BYTES_PER_WARP * _LL_MAX_UNROLL / sizeof(T);
        max_psync_elems_per_warp = _LL_128_PSYNC_BYTES_PER_WARP * _LL_MAX_UNROLL / sizeof(T);
    }

    num_warp_groups =
        _FCOLLECT_MAX(1, warp_count / _FCOLLECT_MAX(1, nelems / max_data_elems_per_warp));
    num_warps_per_group = warp_count / num_warp_groups;
    warp_group_id = warp_id / num_warps_per_group;

    if (teami_node != NULL) {
        /* first n ggroups take on an extra PE in the case of remainder */
        num_pes_per_group = teami_node->size / num_warp_groups;
        remaining_pes = teami_node->size % num_warp_groups;
        num_pes_per_group += warp_group_id < remaining_pes ? 1 : 0;

        start_pe = num_pes_per_group * warp_group_id;
        start_pe += warp_group_id >= remaining_pes ? remaining_pes : 0;

        data_element_offset = warp_id % num_warps_per_group * max_data_elems_per_warp;
        psync_element_offset = warp_id % num_warps_per_group * max_psync_elems_per_warp;
        /* All warps except final one per-pe should be full. */
        for (; data_element_offset + max_data_elems_per_warp < nelems;
             data_element_offset += num_warps_per_group * max_data_elems_per_warp,
             psync_element_offset += num_warps_per_group * max_psync_elems_per_warp) {
            if (LL_VERSION == LL8) {
                nvshmemi_packLL<T, NVSHMEMI_THREADGROUP_WARP, _LL_MAX_UNROLL>(
                    pWrk + pack_offset + psync_element_offset, source + data_element_offset,
                    max_data_elems_per_warp, ll_flag, teami_node, num_pes_per_group, start_pe);
            } else {
                nvshmemi_packLL128<T, NVSHMEMI_THREADGROUP_WARP, _LL_MAX_UNROLL>(
                    pWrk + pack_offset + psync_element_offset, source + data_element_offset,
                    max_data_elems_per_warp, ll_flag, teami_node, num_pes_per_group, start_pe,
                    warp_id % num_warps_per_group);
            }
        }

        if (nelems > data_element_offset) {
            if (LL_VERSION == LL8) {
                nvshmemi_packLL<T, NVSHMEMI_THREADGROUP_WARP, 1>(
                    pWrk + pack_offset + psync_element_offset, source + data_element_offset,
                    nelems - data_element_offset, ll_flag, teami_node, num_pes_per_group, start_pe);
            } else {
                nvshmemi_packLL128<T, NVSHMEMI_THREADGROUP_WARP, 1>(
                    pWrk + pack_offset + psync_element_offset, source + data_element_offset,
                    nelems - data_element_offset, ll_flag, teami_node, num_pes_per_group, start_pe,
                    warp_id % num_warps_per_group);
                ;
            }
        }
    }

    /* todo: also try unrolling in recvLL */
    num_pes_per_group = teami->size / num_warp_groups;
    remaining_pes = teami->size % num_warp_groups;
    /* first n ggroups take on an extra PE in the case of remainder */
    num_pes_per_group += warp_group_id < remaining_pes ? 1 : 0;

    start_pe = num_pes_per_group * warp_group_id;
    start_pe += warp_group_id >= remaining_pes ? remaining_pes : 0;
    data_element_offset = warp_id % num_warps_per_group * max_data_elems_per_warp;
    psync_element_offset = warp_id % num_warps_per_group * max_psync_elems_per_warp;
    for (; data_element_offset + max_data_elems_per_warp < nelems;
         data_element_offset += num_warps_per_group * max_data_elems_per_warp,
         psync_element_offset += num_warps_per_group * max_psync_elems_per_warp) {
        for (next_pe = start_pe; next_pe < start_pe + num_pes_per_group; next_pe++) {
            if (LL_VERSION == LL8) {
                pack_offset =
                    next_pe * NVSHMEMI_TEAM_ROUND_UP((nelems * _FCOLLECT_LL8_PSYNC_SCALE_FACTOR),
                                                     16 / sizeof(T)) +
                    psync_element_offset;
                nvshmemi_recvLL<T, NVSHMEMI_THREADGROUP_WARP>(
                    dest + (next_pe * nelems) + data_element_offset,
                    (uint64_t *)(pWrk + pack_offset), max_data_elems_per_warp, ll_flag);
            } else {
                pack_offset = next_pe * NVSHMEMI_FCOLLECT_LL128_CALC_PSYNC_SIZE(nelems, T) +
                              psync_element_offset;
                nvshmemi_recvLL128<T, _LL_MAX_UNROLL>(
                    dest + (next_pe * nelems) + data_element_offset, pWrk + pack_offset,
                    max_data_elems_per_warp, ll_flag);
            }
        }
    }

    if (nelems > data_element_offset) {
        for (next_pe = start_pe; next_pe < start_pe + num_pes_per_group; next_pe++) {
            if (LL_VERSION == LL8) {
                pack_offset =
                    next_pe * NVSHMEMI_TEAM_ROUND_UP((nelems * _FCOLLECT_LL8_PSYNC_SCALE_FACTOR),
                                                     16 / sizeof(T)) +
                    psync_element_offset;
                nvshmemi_recvLL<T, NVSHMEMI_THREADGROUP_WARP>(
                    dest + (next_pe * nelems) + data_element_offset,
                    (uint64_t *)(pWrk + pack_offset), nelems - data_element_offset, ll_flag);
            } else {
                pack_offset = next_pe * NVSHMEMI_FCOLLECT_LL128_CALC_PSYNC_SIZE(nelems, T) +
                              psync_element_offset;
                nvshmemi_recvLL128<T, 1>(dest + (next_pe * nelems) + data_element_offset,
                                         pWrk + pack_offset, nelems - data_element_offset, ll_flag);
            }
        }
    }
}

template <typename T, threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_fcollect_allpush_threadgroup(
    nvshmem_team_t team, T *dest, const T *source, int dest_offset, size_t nelems) {
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int next_rank;
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();

    // nvshmemi_threadgroup_sync<SCOPE>();
    for (int ii = teami->my_pe; ii < teami->size + teami->my_pe; ii++) {
        next_rank = nvshmemi_team_translate_pe_to_team_world_wrap(teami, ii);
        nvshmemi_put_nbi_threadgroup<T, SCOPE>(dest + dest_offset, source, nelems, next_rank);
    }
    nvshmemi_barrier_threadgroup<SCOPE>(team);
}

template <typename T, threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_fcollect_p2p_allpush_threadgroup(
    nvshmem_team_t team, T *dest, const T *source, int dest_offset, size_t nelems) {
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int next_rank;
    T *dst_ptr;
    nvshmemi_threadgroup_sync<SCOPE>();
    for (int ii = teami->my_pe; ii < teami->size + teami->my_pe; ii++) {
        next_rank = nvshmemi_team_translate_pe_to_team_world_wrap(teami, ii);
        dst_ptr = (T *)nvshmemi_ptr((void *)(dest + dest_offset), next_rank);
        nvshmemi_memcpy_threadgroup<SCOPE>(dst_ptr, source, nelems * sizeof(T));
    }
    nvshmemi_barrier_threadgroup<SCOPE>(team);
}

template <typename T, threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_fcollect_nvls_allpush_threadgroup(
    nvshmem_team_t team, T *dest, const T *source, int dest_offset, size_t nelems) {
#if defined __clang_llvm_bitcode_lib__
    if (__nvvm_reflect("__CUDA_ARCH") >= 900) {
        nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
        nvshmemi_threadgroup_sync<SCOPE>();
        T *dst_ptr = (T *)nvshmemi_mc_ptr(teami, (void *)(dest + dest_offset));
        nvshmemi_mcast_memcpy_threadgroup<T, SCOPE>(dst_ptr, source, nelems * sizeof(T));
        nvshmemi_barrier_threadgroup<SCOPE>(team);
    } else {
        assert(0 && "NVLS is not supported on this platform");
    }
#else
#if __CUDA_ARCH__ >= 900 && CUDART_VERSION >= 12010
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    nvshmemi_threadgroup_sync<SCOPE>();
    T *dst_ptr = (T *)nvshmemi_mc_ptr(teami, (void *)(dest + dest_offset));
    nvshmemi_mcast_memcpy_threadgroup<T, SCOPE>(dst_ptr, source, nelems * sizeof(T));
    nvshmemi_barrier_threadgroup<SCOPE>(team);
#else
    assert(0 && "NVLS is not supported on this platform");
#endif
#endif
}

template <typename T, threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_FORCE_INLINE void nvshmemi_fcollect_threadgroup(
    nvshmem_team_t team, T *dest, const T *source, int dest_offset, size_t nelems) {
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int nthreads = nvshmemi_threadgroup_size<SCOPE>();
    if (!myIdx) /* Only one thread should increment fcollect_count */
        nvshmemi_device_state_d.team_pool[team]->fcollect_count += 1;
    nvshmemi_threadgroup_sync<SCOPE>();
    constexpr bool is_half_prec =
        is_half<T>::value || is_bfloat<T>::value || is_uint16<T>::value || is_int16<T>::value;
    int fcollect_algo = nvshmemi_device_state_d.gpu_coll_env_params_var.fcollect_algo;
    int p2p_direct =
        (nvshmemi_device_state_d.job_connectivity <= NVSHMEMI_JOB_GPU_LDST_REMOTE_ATOMICS);
    const size_t fcollect_ll_threshold =
        nvshmemi_device_state_d.gpu_coll_env_params_var.fcollect_ll_threshold;
    const size_t fcollect_ll128_threshold =
        nvshmemi_device_state_d.gpu_coll_env_params_var.fcollect_ll128_threshold;
    /* NVLS LL performs better with block scoped than thread/warp scoped operations
       due to better efficiency of distributing cvt/pack/unpack ops across threads across GPUs */
    const uint8_t prefer_nvls_ll = (SCOPE == NVSHMEMI_THREADGROUP_BLOCK);
    bool valid_ll_configuration = (SCOPE != NVSHMEMI_THREADGROUP_THREAD &&
                                   ((sizeof(T) >= sizeof(uint32_t) && (nelems % 2 == 0)) ||
                                    (is_half_prec && (nelems % 4 == 0) &&
                                     (nvshmemi_device_state_d.team_pool[team]->size % 2 == 0))));
    /* DISABLE non NVLS LL for hybrid MNNVL configurations. */
    valid_ll_configuration &=
        (nvshmemi_device_state_d.team_pool[NVSHMEM_TEAM_WORLD_INDEX]->are_gpus_p2p_connected ||
         nvshmemi_device_state_d.team_pool[NVSHMEM_TEAM_SHARED]->is_team_node ||
         !nvshmemi_device_state_d.team_pool[NVSHMEM_TEAM_NODE_INDEX]->are_gpus_p2p_connected);
    /* This 2-level selection logic is implemented to reduce code duplication of calling leaf
     * functions on the device code */
    switch (fcollect_algo) {
        case 0: /* default selection */ {
            if (valid_ll_configuration && fcollect_ll_threshold >= (nelems * sizeof(T))) {
                fcollect_algo = FCOLLECT_LL8; /* LL algorithm */
            } else if (valid_ll_configuration && (nelems * sizeof(T)) < fcollect_ll128_threshold) {
                fcollect_algo = FCOLLECT_LL128;
            } else if (sizeof(T) >= sizeof(uint32_t) && (nelems % 2 == 0) &&
                       nvshmemi_device_state_d.gpu_coll_env_params_var.fcollect_ll_threshold >=
                           (nelems * sizeof(T)) &&
                       nvshmemi_device_state_d.team_pool[team]->nvls_rsc_base_ptr != NULL &&
                       prefer_nvls_ll) {
                fcollect_algo = FCOLLECT_NVLS_LL; /* NVLS LL algorithm */
            } else if (nvshmemi_device_state_d.team_pool[team]->nvls_rsc_base_ptr != NULL &&
                       (nelems * sizeof(T)) % 4 == 0) {
                fcollect_algo = FCOLLECT_NVLS; /* NVLS One shot algorithm */
            } else {
                fcollect_algo = FCOLLECT_ONESHOT; /* P2P One shot algorithm */
            }
        } break;
        case FCOLLECT_LL8: /* LL algorithm */
            if (!valid_ll_configuration) {
                fcollect_algo = FCOLLECT_ONESHOT;
            }
            break;
        case FCOLLECT_ONESHOT: /* One shot */
            break;
        case FCOLLECT_NVLS:
            if (nvshmemi_device_state_d.team_pool[team]->nvls_rsc_base_ptr != NULL &&
                (nelems * sizeof(T)) % 4 == 0) {
                /* NVLS simple */
                break;
            } else {
                fcollect_algo = FCOLLECT_ONESHOT; /* One shot */
                break;
            }
        case FCOLLECT_NVLS_LL:
            if (sizeof(T) >= sizeof(uint32_t) && (nelems % 2 == 0) &&
                nvshmemi_device_state_d.gpu_coll_env_params_var.fcollect_ll_threshold >=
                    (nelems * sizeof(T)) &&
                nvshmemi_device_state_d.team_pool[team]->nvls_rsc_base_ptr != NULL) {
                fcollect_algo = FCOLLECT_NVLS_LL; /* Use NVLS LL */
            } else if (nvshmemi_device_state_d.team_pool[team]->nvls_rsc_base_ptr != NULL &&
                       (nelems * sizeof(T)) % 4 == 0) {
                fcollect_algo = FCOLLECT_NVLS; /* Switch to NVLS simple */
            } else {
                fcollect_algo = FCOLLECT_ONESHOT; /* One shot */
            }
            break;
        case FCOLLECT_LL128: /* LL 128 */
            if (!valid_ll_configuration) {
                fcollect_algo = FCOLLECT_ONESHOT;
            }
            break;
        default:
            assert(0 && "Unsupported fcollect algo");
            break;
    }

    switch (fcollect_algo) {
        case FCOLLECT_LL8:
            if (myIdx < nthreads / 32 * 32) {
                if (nvshmemi_device_state_d.team_pool[NVSHMEM_TEAM_NODE_INDEX]
                        ->are_gpus_p2p_connected) {
                    nvshmemi_fcollect_allpush_ll_threadgroup<T, SCOPE, LL8, true>(team, dest,
                                                                                  source, nelems);
                } else {
                    nvshmemi_fcollect_allpush_ll_threadgroup<T, SCOPE, LL8, false>(team, dest,
                                                                                   source, nelems);
                }
            }
            nvshmemi_threadgroup_sync<SCOPE>();
            break;
        case FCOLLECT_ONESHOT:
            if (p2p_direct)
                nvshmemi_fcollect_p2p_allpush_threadgroup<T, SCOPE>(team, dest, source, dest_offset,
                                                                    nelems);
            else
                nvshmemi_fcollect_allpush_threadgroup<T, SCOPE>(team, dest, source, dest_offset,
                                                                nelems);
            break;
        case FCOLLECT_NVLS:
            nvshmemi_fcollect_nvls_allpush_threadgroup<T, SCOPE>(team, dest, source, dest_offset,
                                                                 nelems);
            break;
        case FCOLLECT_LL128:
            if (myIdx < nthreads / 32 * 32) {
                if (nvshmemi_device_state_d.team_pool[NVSHMEM_TEAM_NODE_INDEX]
                        ->are_gpus_p2p_connected) {
                    nvshmemi_fcollect_allpush_ll_threadgroup<T, SCOPE, LL128, true>(team, dest,
                                                                                    source, nelems);
                } else {
                    nvshmemi_fcollect_allpush_ll_threadgroup<T, SCOPE, LL128, false>(
                        team, dest, source, nelems);
                }
            }
            nvshmemi_threadgroup_sync<SCOPE>();
            break;
        case FCOLLECT_NVLS_LL:
            nvshmemi_fcollect_nvls_ll_threadgroup<T, SCOPE>(team, dest, source, nelems);
            break;
        default:
            assert(0);
            break;
    }
}

// ************** Tile allgather **************/

template <typename elemType, threadgroup_t SCOPE, typename tuple_t, int major_dim, int minor_dim>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_tile_allgather_mcast_threadgroup_v4(
    int4 *dest, const int4 *source, const int nelem_major_dim, const int nelem_minor_dim,
    const int src_stride_minor_dim, const int dst_stride_minor_dim, const int src_stride_major_dim,
    const int dst_stride_major_dim, tuple_t start_coord, tuple_t boundary) {
    /*src_stride_major_dim == 0 && dst_stride_major_dim == 0 for vectorized implementation*/
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    int nelems = nelem_major_dim * nelem_minor_dim; /* # vec elems*/
    if constexpr (std::is_empty<tuple_t>::value) {
        /* If no predicate, we vectorize the operation */
        for (size_t j = myIdx; j < nelems; j += groupSize) {
            uint32_t u4[4];
            asm("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                : "=r"(u4[0]), "=r"(u4[1]), "=r"(u4[2]), "=r"(u4[3])
                : "l"(source + ((j) % nelem_major_dim) +
                      ((j) / nelem_major_dim) * src_stride_minor_dim));

            asm("multimem.st.global.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(
                    dest + ((j) % nelem_major_dim) +
                    (((j) / nelem_major_dim) * dst_stride_minor_dim)),
                "r"(u4[0]), "r"(u4[1]), "r"(u4[2]), "r"(u4[3])
                : "memory");
        }
    } else {
        using vtype = int4;
        using cxx_type = uint32_t;
        for (size_t j = myIdx; j < nelems; j += groupSize) {
            uint32_t u4[4];
            /* nelem_major_dim is in vector units*/
            uint32_t elem_coord_major = (j % nelem_major_dim) * (sizeof(vtype) / sizeof(elemType));
            uint32_t elem_coord_minor = (j / nelem_major_dim);

            /* start_coord, boundary are in elemType units */
            /* Check if entire vector is within boundary */
            /* start_coord_major_dim + elem_coord_major_dim + vector len (in elements) <=
             * boundary_major_dim */
            if (is_less_than<tuple_t, major_dim>(
                    start_coord, create_coord_tuple<major_dim>(elem_coord_major, elem_coord_minor),
                    boundary, (sizeof(vtype) / sizeof(elemType)))) {
                asm("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(u4[0]), "=r"(u4[1]), "=r"(u4[2]), "=r"(u4[3])
                    : "l"(source + ((j) % nelem_major_dim) +
                          ((j) / nelem_major_dim) * src_stride_minor_dim));

                asm("multimem.st.global.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(
                        dest + ((j) % nelem_major_dim) +
                        (((j) / nelem_major_dim) * dst_stride_minor_dim)),
                    "r"(u4[0]), "r"(u4[1]), "r"(u4[2]), "r"(u4[3])
                    : "memory");

            } else { /* not all pred elems in vector are 1 */
                     /* perform operations one elem at a time */
                     /* if elem type is < 4B (e.g., f16, bf16), we check at granularity of 4B */

                /* convert elem_coord_major from elemType to cxx_type units */
                /* no change to elem_coord_minor */
                elem_coord_major = (elem_coord_major * sizeof(elemType)) / sizeof(cxx_type);

                /* vector is partially within boundary, check each element */
                cxx_type val;
                for (int u = 0; u < sizeof(vtype) / sizeof(cxx_type); ++u) {
                    /* check if elem is within boundary, use u & elem_coord_major in elemType units
                     */
                    if (is_less_than<tuple_t, major_dim>(
                            start_coord,
                            create_coord_tuple<major_dim>(
                                ((elem_coord_major + u) * sizeof(cxx_type) / sizeof(elemType)),
                                elem_coord_minor),
                            boundary)) {
                        /* convert strides from vector to cxx_type units */
                        asm("ld.global.b32 %0, [%1];"
                            : "=r"(val)
                            : "l"(reinterpret_cast<const cxx_type *>(source) +
                                  (elem_coord_major + u) +
                                  (elem_coord_minor * src_stride_minor_dim *
                                   (sizeof(vtype) / sizeof(cxx_type)))));

                        asm("multimem.st.global.u32 [%0], %1;" ::"l"(
                                reinterpret_cast<cxx_type *>(dest) + (elem_coord_major + u) +
                                (elem_coord_minor * dst_stride_minor_dim *
                                 (sizeof(vtype) / sizeof(cxx_type)))),
                            "r"(val)
                            : "memory");
                    }
                }
            }
        }
    } /*end of if else*/
}

template <typename elemType, threadgroup_t SCOPE, typename tuple_t, int major_dim, int minor_dim>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_tile_allgather_mcast_threadgroup_v2(
    uint64_t *dest, const uint64_t *source, const int nelem_major_dim, const int nelem_minor_dim,
    const int src_stride_minor_dim, const int dst_stride_minor_dim, const int src_stride_major_dim,
    const int dst_stride_major_dim, tuple_t start_coord, tuple_t boundary) {
    /* src_stride_major_dim == 0 && dst_stride_major_dim == 0 for vectorized implementation */
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    int nelems = nelem_major_dim * nelem_minor_dim; /* # vec elems*/

    if constexpr (std::is_empty<tuple_t>::value) {
        /* If no predicate, we vectorize the operation */
        for (size_t j = myIdx; j < nelems; j += groupSize) {
            uint64_t val1;
            asm("ld.global.b64 %0, [%1];"
                : "=l"(val1)
                : "l"(source + ((j) % nelem_major_dim) +
                      ((j) / nelem_major_dim) * src_stride_minor_dim));

            asm("multimem.st.global.u64 [%0], %1;" ::"l"(
                    dest + ((j) % nelem_major_dim) +
                    (((j) / nelem_major_dim) * dst_stride_minor_dim)),
                "l"(val1)
                : "memory");
        }
    } else {
        using vtype = uint64_t;
        using cxx_type = uint32_t;
        for (size_t j = myIdx; j < nelems; j += groupSize) {
            uint64_t val1;
            /* nelem_major_dim is in vector units*/
            /* compute elem_coord_major in elemType units*/
            uint32_t elem_coord_major = (j % nelem_major_dim) * (sizeof(vtype) / sizeof(elemType));
            uint32_t elem_coord_minor = (j / nelem_major_dim);

            /* start_coord, boundary are in elemType units */
            /* Check if entire vector is within boundary */
            /* start_coord_major_dim + elem_coord_major_dim + vector len (in elements) <=
             * boundary_major_dim */
            if (is_less_than<tuple_t, major_dim>(
                    start_coord, create_coord_tuple<major_dim>(elem_coord_major, elem_coord_minor),
                    boundary, (sizeof(vtype) / sizeof(elemType)))) {
                asm("ld.global.b64 %0, [%1];"
                    : "=l"(val1)
                    : "l"(source + ((j) % nelem_major_dim) +
                          ((j) / nelem_major_dim) * src_stride_minor_dim));

                asm("multimem.st.global.u64 [%0], %1;" ::"l"(
                        dest + ((j) % nelem_major_dim) +
                        (((j) / nelem_major_dim) * dst_stride_minor_dim)),
                    "l"(val1)
                    : "memory");

            } else { /* not all pred elems in vector are 1 */
                     /* perform operations one elem at a time */
                     /* if elem type is < 4B (e.g., f16, bf16), we check at granularity of 4B */

                /* convert elem_coord_major from elemType to cxx_type units */
                /* no change to elem_coord_minor */
                elem_coord_major = (elem_coord_major * sizeof(elemType)) / sizeof(cxx_type);

                /* vector is partially within boundary, check each element */
                cxx_type val;
                for (int u = 0; u < sizeof(vtype) / sizeof(cxx_type); ++u) {
                    /* check if elem is within boundary, use u and elem_coord_major in elemType
                     * units */
                    if (is_less_than<tuple_t, major_dim>(
                            start_coord,
                            create_coord_tuple<major_dim>(
                                ((elem_coord_major + u) * sizeof(cxx_type) / sizeof(elemType)),
                                elem_coord_minor),
                            boundary)) {
                        /* convert strides from vector to cxx_type units */
                        asm("ld.global.b32 %0, [%1];"
                            : "=r"(val)
                            : "l"(reinterpret_cast<const cxx_type *>(source) +
                                  (elem_coord_major + u) +
                                  (elem_coord_minor * src_stride_minor_dim *
                                   (sizeof(vtype) / sizeof(cxx_type)))));

                        asm("multimem.st.global.u32 [%0], %1;" ::"l"(
                                reinterpret_cast<cxx_type *>(dest) + (elem_coord_major + u) +
                                (elem_coord_minor * dst_stride_minor_dim *
                                 (sizeof(vtype) / sizeof(cxx_type)))),
                            "r"(val)
                            : "memory");
                    }
                }
            }
        }
    } /*end of if else*/
}

template <typename elemType, threadgroup_t SCOPE, typename tuple_t, int major_dim, int minor_dim>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_tile_allgather_mcast_threadgroup_v1(
    uint32_t *dest, const uint32_t *source, const int nelem_major_dim, const int nelem_minor_dim,
    const int src_stride_minor_dim, const int dst_stride_minor_dim, const int src_stride_major_dim,
    const int dst_stride_major_dim, tuple_t start_coord, tuple_t boundary) {
    int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int groupSize = nvshmemi_threadgroup_size<SCOPE>();
    int nelems = nelem_major_dim * nelem_minor_dim; /* # vec elems*/
    using vtype = uint32_t;
    using cxx_type = uint32_t;
    if constexpr (std::is_empty<tuple_t>::value) {
        cxx_type val;
        for (size_t j = myIdx; j < nelems; j += groupSize) {
            asm("ld.global.b32 %0, [%1];"
                : "=r"(val)
                : "l"(source + ((j % nelem_major_dim) * src_stride_major_dim) +
                      ((j / nelem_major_dim) * src_stride_minor_dim)));

            asm("multimem.st.global.u32 [%0], %1;" ::"l"(
                    dest + ((j % nelem_major_dim) * dst_stride_major_dim) +
                    ((j / nelem_major_dim) * dst_stride_minor_dim)),
                "r"(val)
                : "memory");
        }
    } else {
        for (size_t j = myIdx; j < nelems; j += groupSize) {
            /* nelem_major_dim is in vector units*/
            /* compute elem_coord_major in elemType units*/
            uint32_t elem_coord_major = (j % nelem_major_dim) * (sizeof(vtype) / sizeof(elemType));
            uint32_t elem_coord_minor = (j / nelem_major_dim);
            cxx_type val;

            /* convert elem_coord_major from elemType to cxx_type units */
            /* no change to elem_coord_minor */
            elem_coord_major = (elem_coord_major * sizeof(elemType)) / sizeof(cxx_type);

            for (int u = 0; u < sizeof(vtype) / sizeof(cxx_type); ++u) {
                /* check if elem is within boundary, use u and elem_coord_major in elemType units */
                if (is_less_than<tuple_t, major_dim>(
                        start_coord,
                        create_coord_tuple<major_dim>(
                            ((elem_coord_major + u) * sizeof(cxx_type) / sizeof(elemType)),
                            elem_coord_minor),
                        boundary)) {
                    /* convert strides from vector to cxx_type units */
                    asm("ld.global.b32 %0, [%1];"
                        : "=r"(val)
                        : "l"(reinterpret_cast<const cxx_type *>(source) + (elem_coord_major + u) +
                              (elem_coord_minor * src_stride_minor_dim *
                               (sizeof(vtype) / sizeof(cxx_type)))));

                    asm("multimem.st.global.u32 [%0], %1;" ::"l"(
                            reinterpret_cast<cxx_type *>(dest) + (elem_coord_major + u) +
                            (elem_coord_minor * dst_stride_minor_dim *
                             (sizeof(vtype) / sizeof(cxx_type)))),
                        "r"(val)
                        : "memory");
                }
            }
        }
    } /*end of if else*/
}

// Select implementation based on the operation, datatype
template <typename vtype, typename T, threadgroup_t scope, typename tuple_t, int major_dim,
          int minor_dim>
__device__ inline void nvshmemi_tile_allgather_nvls_threadgroup_vec(
    nvshmem_team_t team, T *src, T *dst,
    const int size_major_dim,        // size along the major dimension in elements
    const int size_minor_dim,        // size along the minor dimension in elements
    const int src_stride_minor_dim,  // src stride along minor dimension in elements
    const int dst_stride_minor_dim,  // dst stride along minor dimension in elements
    const int src_stride_major_dim,  // src stride along major dimension in elements
    const int dst_stride_major_dim,  // dst stride along major dimension in elements
    tuple_t start_coord, tuple_t boundary) {
    // src is local, dst is multicast address
    vtype *src_v = reinterpret_cast<vtype *>(src);
    vtype *dst_v = reinterpret_cast<vtype *>(nvshmemx_mc_ptr(team, dst));
    assert((dst_v != nullptr) && "Failed to get multicast ptr for destination");

    int src_stride_minor_dim_v = src_stride_minor_dim;
    if (src_stride_minor_dim > 1) {
        src_stride_minor_dim_v = (src_stride_minor_dim * sizeof(T)) / sizeof(vtype);
    }
    int dst_stride_minor_dim_v = dst_stride_minor_dim;
    if (dst_stride_minor_dim > 1) {
        dst_stride_minor_dim_v = (dst_stride_minor_dim * sizeof(T)) / sizeof(vtype);
    }
    int src_stride_major_dim_v = src_stride_major_dim;  // keep stride as is if ==1
    if (src_stride_major_dim > 1) {
        src_stride_major_dim_v = (src_stride_major_dim * sizeof(T)) / sizeof(vtype);
    }
    int dst_stride_major_dim_v = dst_stride_major_dim;
    if (dst_stride_major_dim > 1) {
        dst_stride_major_dim_v = (dst_stride_major_dim * sizeof(T)) / sizeof(vtype);
    }

    int nelem_major_dim = (size_major_dim * sizeof(T)) / sizeof(vtype);
    int nelem_minor_dim = size_minor_dim;

    if constexpr (std::is_same<vtype, int4>::value) {
        nvshmemi_tile_allgather_mcast_threadgroup_v4<T, scope, tuple_t, major_dim, minor_dim>(
            dst_v, src_v, nelem_major_dim, nelem_minor_dim, src_stride_minor_dim_v,
            dst_stride_minor_dim_v, src_stride_major_dim_v, dst_stride_major_dim_v, start_coord,
            boundary);

    } else if constexpr (std::is_same<vtype, uint64_t>::value) {
        nvshmemi_tile_allgather_mcast_threadgroup_v2<T, scope, tuple_t, major_dim, minor_dim>(
            dst_v, src_v, nelem_major_dim, nelem_minor_dim, src_stride_minor_dim_v,
            dst_stride_minor_dim_v, src_stride_major_dim_v, dst_stride_major_dim_v, start_coord,
            boundary);

    } else if constexpr (std::is_same<vtype, uint32_t>::value) {
        nvshmemi_tile_allgather_mcast_threadgroup_v1<T, scope, tuple_t, major_dim, minor_dim>(
            dst_v, src_v, nelem_major_dim, nelem_minor_dim, src_stride_minor_dim_v,
            dst_stride_minor_dim_v, src_stride_major_dim_v, dst_stride_major_dim_v, start_coord,
            boundary);

    } else {
        if (std::is_same<vtype, int4>::value) {
            nvshmemi_tile_allgather_mcast_threadgroup_v4<T, scope, tuple_t, major_dim, minor_dim>(
                dst_v, src_v, nelem_major_dim, nelem_minor_dim, src_stride_minor_dim_v,
                dst_stride_minor_dim_v, src_stride_major_dim_v, dst_stride_major_dim_v, start_coord,
                boundary);

        } else if (std::is_same<vtype, uint64_t>::value) {
            nvshmemi_tile_allgather_mcast_threadgroup_v2<T, scope, tuple_t, major_dim, minor_dim>(
                dst_v, src_v, nelem_major_dim, nelem_minor_dim, src_stride_minor_dim_v,
                dst_stride_minor_dim_v, src_stride_major_dim_v, dst_stride_major_dim_v, start_coord,
                boundary);

        } else if (std::is_same<vtype, uint32_t>::value) {
            nvshmemi_tile_allgather_mcast_threadgroup_v1<T, scope, tuple_t, major_dim, minor_dim>(
                dst_v, src_v, nelem_major_dim, nelem_minor_dim, src_stride_minor_dim_v,
                dst_stride_minor_dim_v, src_stride_major_dim_v, dst_stride_major_dim_v, start_coord,
                boundary);
        } else {
            assert(0 && "unsupported vector type");
        }
    }
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t, threadgroup_t scope,
          int major_dim, int minor_dim>
__device__ inline void nvshmemi_tile_allgather_nvls_dim(nvshmem_team_t team,
                                                        src_tensor_t src_tensor,
                                                        dst_tensor_t dst_tensor,
                                                        tuple_t start_coord, tuple_t boundary) {
    using T = typename src_tensor_t::value_type;
    int mype = nvshmem_team_my_pe(team);

    /* The destination tensor is npes * src_tensor size
     *  Assume 4 PEs, with src_tensor as shown below
     *    __    __    __    __
     *   |__|  |__|  |__|  |__|   // src_tensor of 4 PEs
     *
     *  dest_tensor is 4 x src_tensor
     *   __ __
     *  |__|__|   // dest_tensor of a PE (4xsrc_tensor)
     *  |__|__|
     *
     *   Other shapes are possible for dst_tensor
     *   __ __ __ __      __
     *  |__|__|__|__|    |__|
     *                   |__|
     *                   |__|
     *                   |__|
     *
     *  For every src_tensor from a given PE, we need to find the starting offset within
     *  dest_tensor.
     */

    // Compute the offset within dst tile specific for the PE
    T *dst =
        (dst_tensor.data() +
         (((mype * get_shape_element<major_dim>(src_tensor)) %
           get_shape_element<major_dim>(dst_tensor)) *
          get_stride_element<major_dim>(dst_tensor)) +
         (((mype * get_shape_element<major_dim>(src_tensor)) /
           get_shape_element<major_dim>(dst_tensor)) *
          (get_shape_element<minor_dim>(src_tensor) * get_stride_element<minor_dim>(dst_tensor))));

    // Since it is PUSH based implementation, number of elements being copied is same as
    // src_tensor so, elements in each dimension is based on src_tensor but stride is based on
    // dst_tensor

    // check for vector len == 4
    // Conditions: ptr must be aligned to int4, shape must be a multiple of 16, stride must be a
    // multiple of 16
    if (((size_t)src_tensor.data() % sizeof(int4) == 0) &&
        ((size_t)dst_tensor.data() % sizeof(int4) == 0) &&
        (((get_tuple_val<major_dim>(src_tensor.shape()) * sizeof(T)) % sizeof(int4)) == 0) &&
        (((get_stride_element<minor_dim>(src_tensor) * sizeof(T)) % sizeof(int4)) == 0) &&
        (((get_stride_element<minor_dim>(dst_tensor) * sizeof(T)) % sizeof(int4)) == 0)) {
        nvshmemi_tile_allgather_nvls_threadgroup_vec<int4, T, scope, tuple_t, major_dim, minor_dim>(
            team, src_tensor.data(), dst,
            get_shape_element<major_dim>(src_tensor),   // contiguous size
            get_shape_element<minor_dim>(src_tensor),   // strided size
            get_stride_element<minor_dim>(src_tensor),  // src stride minor_dim
            get_stride_element<minor_dim>(dst_tensor),  // dst stride minor_dim
            get_stride_element<major_dim>(src_tensor),  // src stride major_dim; equal to 1
            get_stride_element<major_dim>(dst_tensor),  // dst stride major_dim; equal to 1
            start_coord, boundary);

    } else if (((size_t)src_tensor.data() % sizeof(uint64_t) == 0) &&
               ((size_t)dst_tensor.data() % sizeof(uint64_t) == 0) &&
               (((get_tuple_val<major_dim>(src_tensor.shape()) * sizeof(T)) % sizeof(uint64_t)) ==
                0) &&
               (((get_stride_element<minor_dim>(src_tensor) * sizeof(T)) % sizeof(uint64_t)) ==
                0) &&
               (((get_stride_element<minor_dim>(dst_tensor) * sizeof(T)) % sizeof(uint64_t)) ==
                0)) {
        nvshmemi_tile_allgather_nvls_threadgroup_vec<uint64_t, T, scope, tuple_t, major_dim,
                                                     minor_dim>(
            team, src_tensor.data(), dst,
            get_shape_element<major_dim>(src_tensor),   // contiguous size
            get_shape_element<minor_dim>(src_tensor),   // strided size
            get_stride_element<minor_dim>(src_tensor),  // src stride minor_dim
            get_stride_element<minor_dim>(dst_tensor),  // dst stride minor_dim
            get_stride_element<major_dim>(src_tensor),  // src stride major_dim; equal to 1
            get_stride_element<major_dim>(dst_tensor),  // dst stride major_dim; equal to 1
            start_coord, boundary);

    } else {  // vector len 1
        nvshmemi_tile_allgather_nvls_threadgroup_vec<uint32_t, T, scope, tuple_t, major_dim,
                                                     minor_dim>(
            team, src_tensor.data(), dst,
            get_shape_element<major_dim>(src_tensor),   // contiguous size
            get_shape_element<minor_dim>(src_tensor),   // strided size
            get_stride_element<minor_dim>(src_tensor),  // src stride minor_dim
            get_stride_element<minor_dim>(dst_tensor),  // dst stride minor_dim
            get_stride_element<major_dim>(src_tensor),  // src stride major_dim; equal to 1
            get_stride_element<major_dim>(dst_tensor),  // dst stride major_dim; equal to 1
            start_coord, boundary);
    }
}
// specialize for the vectorization
template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t, threadgroup_t scope>
__device__ inline void nvshmemi_tile_allgather_nvls_threadgroup(nvshmem_team_t team,
                                                                src_tensor_t src_tensor,
                                                                dst_tensor_t dst_tensor,
                                                                tuple_t start_coord,
                                                                tuple_t boundary) {
    using T = typename src_tensor_t::value_type;
    if constexpr ((get_constant(safe_get<0>(decltype(src_tensor.stride()){})) == 1) &&
                  (get_constant(safe_get<0>(decltype(dst_tensor.stride()){})) == 1)) {
        // dim 0 major
        constexpr int major_dim = 0;
        constexpr int minor_dim = 1;

        if constexpr (sizeof(T) < 4) {
            // Shape along major dimension should be divisible by 2, because we operate at fp16x2
            assert(((get_shape_element<major_dim>(src_tensor) % 2) == 0) &&
                   ((get_shape_element<major_dim>(dst_tensor) % 2) == 0) &&
                   "Currently for 16B datatypes, we only support tensors which are 32b aligned "
                   "along their continuous dimension");
        }

        nvshmemi_tile_allgather_nvls_dim<src_tensor_t, dst_tensor_t, tuple_t, scope, major_dim,
                                         minor_dim>(team, src_tensor, dst_tensor, start_coord,
                                                    boundary);
    } else if constexpr ((get_constant(safe_get<1>(decltype(src_tensor.stride()){})) == 1) &&
                         (get_constant(safe_get<1>(decltype(dst_tensor.stride()){})) == 1)) {
        // dim 1 major
        constexpr int major_dim = 1;
        constexpr int minor_dim = 0;

        if constexpr (sizeof(T) < 4) {
            // Shape along major dimension should be divisible by 2, because we operate at fp16x2
            assert(((get_shape_element<major_dim>(src_tensor) % 2) == 0) &&
                   ((get_shape_element<major_dim>(dst_tensor) % 2) == 0) &&
                   "Currently for 16B datatypes, we only support tensors which are 32b aligned "
                   "along their continuous dimension");
        }

        nvshmemi_tile_allgather_nvls_dim<src_tensor_t, dst_tensor_t, tuple_t, scope, major_dim,
                                         minor_dim>(team, src_tensor, dst_tensor, start_coord,
                                                    boundary);
    } else {
        // No contiguous dimension found at compile time
        // TODO support when major dimension for src and tensor are different
        if ((get_stride_element<1>(src_tensor) == 1) && (get_stride_element<1>(dst_tensor) == 1)) {
            constexpr int major_dim = 1;
            constexpr int minor_dim = 0;

            if constexpr (sizeof(T) < 4) {
                // Shape along major dimension should be divisible by 2, because we operate at
                // fp16x2
                assert(((get_shape_element<major_dim>(src_tensor) % 2) == 0) &&
                       ((get_shape_element<major_dim>(dst_tensor) % 2) == 0) &&
                       "Currently for 16B datatypes, we only support tensors which are 32b aligned "
                       "along their continuous dimension");
            }

            nvshmemi_tile_allgather_nvls_dim<src_tensor_t, dst_tensor_t, tuple_t, scope, major_dim,
                                             minor_dim>(team, src_tensor, dst_tensor, start_coord,
                                                        boundary);
        } else {
            // setting major_dim to 0, minor_dim to 1
            constexpr int major_dim = 0;
            constexpr int minor_dim = 1;

            if constexpr (sizeof(T) < 4) {
                // Shape along major dimension should be divisible by 2, because we operate at
                // fp16x2
                assert(((get_shape_element<major_dim>(src_tensor) % 2) == 0) &&
                       ((get_shape_element<major_dim>(dst_tensor) % 2) == 0) &&
                       "Currently for 16B datatypes, we only support tensors which are 32b aligned "
                       "along their continuous dimension");
            }
            nvshmemi_tile_allgather_nvls_dim<src_tensor_t, dst_tensor_t, tuple_t, scope, major_dim,
                                             minor_dim>(team, src_tensor, dst_tensor, start_coord,
                                                        boundary);
        }
    }
}

// Tile allgather entrypoint
// Call underlying function based on scope and algo
template <nvshmemx::tile_coll_algo_t algo, typename src_tensor_t, typename dst_tensor_t,
          typename tuple_t, threadgroup_t scope>
__device__ inline int nvshmemi_tile_allgather(nvshmem_team_t team, src_tensor_t src_tensor,
                                              dst_tensor_t dst_tensor, tuple_t start_coord,
                                              tuple_t boundary, uint64_t flag) {
#if defined(__cplusplus) && __cplusplus < 201703L
    assert(0 && "Tile-granular APIs need C++ 17");
#endif
    using T = typename src_tensor_t::value_type;

    static_assert(
        std::is_same<typename src_tensor_t::value_type, typename dst_tensor_t::value_type>::value,
        "Source and destination tensors must have the same type");

    static_assert((algo == nvshmemx::tile_coll_algo_t::NVLS_ONE_SHOT_PUSH_NBI),
                  "Unsupported tile AllGather algorithm. "
                  "Currently NVLS_ONE_SHOT_PUSH_NBI is supported for tile allgather");

    static_assert((scope == NVSHMEMI_THREADGROUP_THREAD) || (scope == NVSHMEMI_THREADGROUP_WARP) ||
                      (scope == NVSHMEMI_THREADGROUP_WARPGROUP) ||
                      (scope == NVSHMEMI_THREADGROUP_BLOCK),
                  "Unsupported scope");

    assert((src_tensor.data() != nullptr) && (dst_tensor.data() != nullptr) &&
           "Null pointers passed");

    // check shape
    assert((get_shape_element<0>(src_tensor) * get_shape_element<1>(src_tensor) *
            nvshmem_team_n_pes(team)) &&
           (get_shape_element<0>(dst_tensor) * get_shape_element<1>(dst_tensor)));

    // TODO add other data types
    static_assert(((is_half<T>::value) || (is_bfloat<T>::value) || (is_float<T>::value)),
                  "Unsupported datatype");

    // check if both src and dst have same continuous dimension
    // TODO relax this constraint
    assert(
        (((get_stride_element<0>(src_tensor) == 1) && (get_stride_element<0>(dst_tensor) == 1)) ||
         ((get_stride_element<1>(src_tensor) == 1) && (get_stride_element<1>(dst_tensor) == 1))) &&
        "Currently we only support cases where source and destination tile are continuous "
        "along one dimension");

    assert(!flag && "Currently non-zero flag value is unsupported");

    // NVLS Gather only has one-shot push support currently
    if constexpr (algo == nvshmemx::tile_coll_algo_t::NVLS_ONE_SHOT_PUSH_NBI) {
        // check for NVLS support in hardware
#if __CUDA_ARCH__ >= 900 && CUDART_VERSION >= 12010

        // NVLS ONE_SHOT AllGather is PUSH based algo, so we can directly start communicating
        // User should ensure src data is ready

        nvshmemi_tile_allgather_nvls_threadgroup<src_tensor_t, dst_tensor_t, tuple_t, scope>(
            team, src_tensor, dst_tensor, start_coord, boundary);
#else
        assert(__CUDA_ARCH__ >= 900 && CUDART_VERSION >= 12010 &&
               "Unsupported NVLS on this platform");
#endif
        return 0;
    } else {
        // Extend as other algorithms are added
        return 0;
    }
}
#endif /* __CUDA_ARCH__ */
#endif /* FCOLLECT_DEVICE_CUH */
