/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See License.txt for license information
 */

#include "host/nvshmem_macros.h"
#include "device_host/nvshmem_types.h"

#ifndef NVSHMEMI_COLL_H
#define NVSHMEMI_COLL_H

NVSHMEMI_HOSTDEVICE_PREFIX void nvshmemi_barrier(nvshmem_team_t team);

#endif /* NVSHMEMI_COLL_H */
