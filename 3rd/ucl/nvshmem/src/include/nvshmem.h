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

#ifndef _NVSHMEM_H_
#define _NVSHMEM_H_

#include "non_abi/nvshmem_build_options.h"
/* NVRTC only compiles device code. Leave out host headers */
#if !defined __CUDACC_RTC__ && !defined __clang_llvm_bitcode_lib__
#include "nvshmem_host.h"
#endif
/* NVSHMEM4PY hostlib can't parse device headers */
#if !defined NVSHMEM_HOSTLIB_ONLY
#include "device/nvshmem_defines.h"
#include "device/nvshmem_coll_defines.cuh"
#include "device/nvshmemx_defines.h"
#include "device/nvshmemx_coll_defines.cuh"
#endif
#endif
