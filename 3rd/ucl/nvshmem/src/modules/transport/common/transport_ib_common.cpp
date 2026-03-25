/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

#include "transport_ib_common.h"
#include <assert.h>                            // for assert
#include <cuda.h>                              // for CUdeviceptr, CU_MEM_RA...
#include <cuda_runtime.h>                      // for cudaGetLastError, cuda...
#include <dlfcn.h>                             // for dlclose, dlopen, RTLD_...
#include <driver_types.h>                      // for cudaPointerAttributes
#include <errno.h>                             // for errno
#include <infiniband/verbs.h>                  // for IBV_ACCESS_LOCAL_WRITE
#include <stdint.h>                            // for uintptr_t, uint64_t
#include <string.h>                            // for strerror
#include <unistd.h>                            // for access, close, sysconf
#include "internal/host_transport/cudawrap.h"  // for nvshmemi_cuda_fn_table
#include "non_abi/nvshmemx_error.h"            // for NVSHMEMX_ERROR_INTERNAL
#include "non_abi/nvshmem_build_options.h"     // for NVSHMEM_USE_MLX5DV
#include "transport_common.h"                  // for LOAD_SYM, INFO, MAXPAT...

int nvshmemt_ib_common_nv_peer_mem_available() {
    if (access("/sys/kernel/mm/memory_peers/nv_mem/version", F_OK) == 0) {
        return NVSHMEMX_SUCCESS;
    }
    if (access("/sys/kernel/mm/memory_peers/nvidia-peermem/version", F_OK) == 0) {
        return NVSHMEMX_SUCCESS;
    }
    if (access("/sys/module/nvidia_peermem/version", F_OK) == 0) {
        return NVSHMEMX_SUCCESS;
    }

    return NVSHMEMX_ERROR_INTERNAL;
}

int nvshmemt_ib_common_reg_mem_handle(struct nvshmemt_ibv_function_table *ftable,
                                      struct nvshmemt_mlx5dv_function_table *mlx5dv_ftable,
                                      struct ibv_pd *pd, nvshmem_mem_handle_t *mem_handle,
                                      void *buf, size_t length, bool local_only,
                                      bool dmabuf_support, struct nvshmemi_cuda_fn_table *table,
                                      int log_level, bool relaxed_ordering, bool is_data_direct,
                                      void *alias_va_ptr) {
    struct nvshmemt_ib_common_mem_handle *handle =
        (struct nvshmemt_ib_common_mem_handle *)mem_handle;
    struct ibv_mr *mr = NULL;
    int status = 0;
    int ro_flag = 0;
    bool host_memory = false;

    assert(sizeof(struct nvshmemt_ib_common_mem_handle) <= NVSHMEM_MEM_HANDLE_SIZE);

    cudaPointerAttributes attr;
    status = cudaPointerGetAttributes(&attr, buf);
    if (status != cudaSuccess) {
        host_memory = true;
        status = 0;
        cudaGetLastError();
    } else if (attr.type != cudaMemoryTypeDevice) {
        host_memory = true;
    }

#if defined(HAVE_IBV_ACCESS_RELAXED_ORDERING)
#if HAVE_IBV_ACCESS_RELAXED_ORDERING == 1
    // IBV_ACCESS_RELAXED_ORDERING has been introduced to rdma-core since v28.0.
    if (relaxed_ordering) {
        ro_flag = IBV_ACCESS_RELAXED_ORDERING;
    }
#endif
#endif

    if (ftable->reg_dmabuf_mr != NULL && !host_memory && dmabuf_support &&
        CUPFN(table, cuMemGetHandleForAddressRange)) {
        size_t page_size = sysconf(_SC_PAGESIZE);
        size_t size_aligned;
        int handle_flag = is_data_direct ? CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE : 0;
        CUdeviceptr p;
        p = (CUdeviceptr)((uintptr_t)buf & ~(page_size - 1));
        size_aligned =
            ((length + (uintptr_t)buf - (uintptr_t)p + page_size - 1) / page_size) * page_size;

        CUCHECKGOTO(table,
                    cuMemGetHandleForAddressRange(&handle->fd, (CUdeviceptr)p, size_aligned,
                                                  CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, handle_flag),
                    status, out);
        if (is_data_direct) {
            NVSHMEMI_NULL_ERROR_JMP(mlx5dv_ftable, status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                                    "mlx5dv_ftable is NULL with data direct enabled\n");
            mr = mlx5dv_ftable->mlx5dv_internal_reg_dmabuf_mr(
                pd, 0, size_aligned, (uint64_t)p, handle->fd,
                IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ |
                    IBV_ACCESS_REMOTE_ATOMIC | ro_flag,
                MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT);
            if (mr == NULL) {
                close(handle->fd);
                goto reg_dmabuf_failure;
            }
            INFO(log_level, "mlx5dv_reg_dmabuf_mr handle %p mr %p", handle, mr);
        } else {
            mr = ftable->reg_dmabuf_mr(pd, 0, size_aligned, (uint64_t)p, handle->fd,
                                       IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                           IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC |
                                           ro_flag);
            if (mr == NULL) {
                close(handle->fd);
                goto reg_dmabuf_failure;
            }
            INFO(log_level, "ibv_reg_dmabuf_mr handle %p mr %p", handle, mr);
        }
    } else {
    reg_dmabuf_failure:

        handle->fd = 0;
        if (alias_va_ptr) {
            mr = ftable->reg_mr_iova(pd, alias_va_ptr, length, (uint64_t)buf,
                                     IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                         IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC |
                                         ro_flag);
            INFO(log_level, "ibv_reg_mr_iova handle %p mr %p", handle, mr);
        } else {
            mr = ftable->reg_mr(pd, buf, length,
                                IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                    IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC | ro_flag);
            INFO(log_level, "ibv_reg_mr handle %p mr %p", handle, mr);
        }

        NVSHMEMI_NULL_ERROR_JMP(mr, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                                "mem registration failed. Reason: %s\n", strerror(errno));
    }

    handle->buf = buf;
    handle->lkey = mr->lkey;
    handle->rkey = mr->rkey;
    handle->mr = mr;
    handle->local_only = local_only;

out:
    return status;
}

int nvshmemt_ib_common_release_mem_handle(struct nvshmemt_ibv_function_table *ftable,
                                          nvshmem_mem_handle_t *mem_handle, int log_level) {
    int status = 0;
    struct nvshmemt_ib_common_mem_handle *handle =
        (struct nvshmemt_ib_common_mem_handle *)mem_handle;

    INFO(log_level, "ibv_dereg_mr handle %p handle->mr %p", handle, handle->mr);
    if (handle->mr) {
        status = ftable->dereg_mr((struct ibv_mr *)handle->mr);
        if (handle->fd) close(handle->fd);
    }
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_dereg_mr failed \n");

out:
    return status;
}

bool nvshmemt_mlx5dv_dmabuf_capable(ibv_context *context,
                                    struct nvshmemt_ibv_function_table *ftable,
                                    struct nvshmemt_mlx5dv_function_table *mlx5dv_ftable) {
    int status = 0;
    int dev_fail = 0;
    struct ibv_pd *pd = ftable->alloc_pd(context);
    NVSHMEMI_NULL_ERROR_JMP(pd, status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_alloc_pd failed \n");

    if (mlx5dv_ftable->mlx5dv_internal_reg_dmabuf_mr == NULL) {
        errno = EOPNOTSUPP;
    } else {
        mlx5dv_ftable->mlx5dv_internal_reg_dmabuf_mr(pd, 0ULL /*offset*/, 0ULL /*len*/,
                                                     0ULL /*iova*/, -1 /*fd*/, 0 /*flags*/,
                                                     0 /* mlx5 flags*/);
        // mlx5dv_reg_dmabuf_mr() will fail with EOPNOTSUPP/EPROTONOSUPPORT if not supported (EBADF
        // otherwise)
    }

    dev_fail |= (errno == EOPNOTSUPP) || (errno == EPROTONOSUPPORT);

    status = ftable->dealloc_pd(pd);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_dealloc_pd failed \n");
    if (dev_fail) goto out;
    return true;
out:
    return false;
}

int nvshmemt_ib_iface_get_mlx_path(ibv_device *dev, ibv_context *ctx, char **path,
                                   struct nvshmemt_ibv_function_table *ftable,
                                   struct nvshmemt_mlx5dv_function_table *mlx5dv_ftable,
                                   bool *is_data_direct, int log_level) {
    int status;
    char device_path[MAXPATHSIZE];

    *is_data_direct = false;
#ifdef NVSHMEM_USE_MLX5DV
    if (mlx5dv_ftable->mlx5dv_internal_is_supported &&
        mlx5dv_ftable->mlx5dv_internal_is_supported(dev)) {
        snprintf(device_path, MAXPATHSIZE, "/sys");
        if ((nvshmemt_mlx5dv_dmabuf_capable(ctx, ftable, mlx5dv_ftable)) &&
            (!mlx5dv_ftable->mlx5dv_internal_get_data_direct_sysfs_path(ctx, device_path + 4,
                                                                        MAXPATHSIZE - 4))) {
            *is_data_direct = true;
            INFO(log_level, "directNIC features supported and enabled in device %s %s", dev->name,
                 device_path);
            status = NVSHMEMX_SUCCESS;
        }
    }
#endif
    if (!*is_data_direct) {
        status = snprintf(device_path, MAXPATHSIZE, "/sys/class/infiniband/%s/device",
                          (const char *)dev->name);
        if (status < 0 || status >= MAXPATHSIZE) {
            NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                               "Unable to fill in device name.\n");
        } else {
            status = NVSHMEMX_SUCCESS;
        }
    }

    *path = realpath(device_path, NULL);
    NVSHMEMI_NULL_ERROR_JMP(*path, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "realpath failed \n");

out:
    return status;
}

int nvshmemt_ibv_ftable_init(void **ibv_handle, struct nvshmemt_ibv_function_table *ftable,
                             int log_level) {
    *ibv_handle = dlopen("libibverbs.so.1", RTLD_LAZY);
    if (*ibv_handle == NULL) {
        INFO(log_level, "libibverbs not found on the system.");
        return -1;
    }

    LOAD_SYM(*ibv_handle, "ibv_fork_init", ftable->fork_init);
    LOAD_SYM(*ibv_handle, "ibv_create_ah", ftable->create_ah);
    LOAD_SYM(*ibv_handle, "ibv_get_device_list", ftable->get_device_list);
    LOAD_SYM(*ibv_handle, "ibv_get_device_name", ftable->get_device_name);
    LOAD_SYM(*ibv_handle, "ibv_open_device", ftable->open_device);
    LOAD_SYM(*ibv_handle, "ibv_close_device", ftable->close_device);
    LOAD_SYM(*ibv_handle, "ibv_query_port", ftable->query_port);
    LOAD_SYM(*ibv_handle, "ibv_query_device", ftable->query_device);
    LOAD_SYM(*ibv_handle, "ibv_alloc_pd", ftable->alloc_pd);
    LOAD_SYM(*ibv_handle, "ibv_reg_mr", ftable->reg_mr);
    LOAD_SYM(*ibv_handle, "ibv_reg_mr_iova", ftable->reg_mr_iova);
    LOAD_SYM(*ibv_handle, "ibv_reg_dmabuf_mr", ftable->reg_dmabuf_mr);
    LOAD_SYM(*ibv_handle, "ibv_dereg_mr", ftable->dereg_mr);
    LOAD_SYM(*ibv_handle, "ibv_create_cq", ftable->create_cq);
    LOAD_SYM(*ibv_handle, "ibv_create_qp", ftable->create_qp);
    LOAD_SYM(*ibv_handle, "ibv_create_srq", ftable->create_srq);
    LOAD_SYM(*ibv_handle, "ibv_modify_qp", ftable->modify_qp);
    LOAD_SYM(*ibv_handle, "ibv_query_gid", ftable->query_gid);
    LOAD_SYM(*ibv_handle, "ibv_dealloc_pd", ftable->dealloc_pd);
    LOAD_SYM(*ibv_handle, "ibv_destroy_qp", ftable->destroy_qp);
    LOAD_SYM(*ibv_handle, "ibv_destroy_cq", ftable->destroy_cq);
    LOAD_SYM(*ibv_handle, "ibv_destroy_srq", ftable->destroy_srq);
    LOAD_SYM(*ibv_handle, "ibv_destroy_ah", ftable->destroy_ah);

    return 0;
}

int nvshmemt_mlx5dv_ftable_init(void **mlx5dv_handle, struct nvshmemt_mlx5dv_function_table *ftable,
                                int log_level) {
    *mlx5dv_handle = dlopen("libmlx5.so", RTLD_LAZY);
    if (*mlx5dv_handle == NULL) {
        *mlx5dv_handle = dlopen("libmlx5.so.1", RTLD_LAZY);
    }
    if (*mlx5dv_handle == NULL) {
        INFO(log_level, "Failed to open libmlx5.so[.1]");
        ftable->mlx5dv_internal_is_supported = NULL;
        ftable->mlx5dv_internal_get_data_direct_sysfs_path = NULL;
        ftable->mlx5dv_internal_reg_dmabuf_mr = NULL;
        return -1;
    }
    LOAD_SYM_VERSION(*mlx5dv_handle, "mlx5dv_is_supported", ftable->mlx5dv_internal_is_supported,
                     MLX5DV_VERSION);
    LOAD_SYM_VERSION(*mlx5dv_handle, "mlx5dv_get_data_direct_sysfs_path",
                     ftable->mlx5dv_internal_get_data_direct_sysfs_path, "MLX5_1.25");
    LOAD_SYM_VERSION(*mlx5dv_handle, "mlx5dv_reg_dmabuf_mr", ftable->mlx5dv_internal_reg_dmabuf_mr,
                     "MLX5_1.25");

    return 0;
}

void nvshmemt_ibv_ftable_fini(void **ibv_handle) {
    int status;

    if (ibv_handle) {
        status = dlclose(*ibv_handle);
        if (status) {
            NVSHMEMI_ERROR_PRINT("Unable to close libibverbs handle.");
        }
    }
}

void nvshmemt_mlx5dv_ftable_fini(void **mlx5dv_handle) {
    int status;

    if (mlx5dv_handle) {
        status = dlclose(*mlx5dv_handle);
        if (status) {
            NVSHMEMI_ERROR_PRINT("Unable to close libmlx5dv handle.");
        }
    }
}
