/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

#ifndef _TRANSPORT_COMMON_H
#define _TRANSPORT_COMMON_H

#define __STDC_FORMAT_MACROS 1

#include <stdio.h>    // for fprintf, stderr
#include <strings.h>  // for strncasecmp
#include <unordered_map>
#include "bootstrap_host_transport/env_defs_internal.h"  // for nvshmemi_opt...
#include "internal/host_transport/transport.h"           // for nvshmem_tran...

#define MAXPATHSIZE 1024
#define MAX_TRANSPORT_EP_COUNT 1

#define likely(x) __builtin_expect((x), 1)
#define unlikely(x) __builtin_expect((x), 0)

#define TRANSPORT_LOG_NONE 0
#define TRANSPORT_LOG_VERSION 1
#define TRANSPORT_LOG_WARN 2
#define TRANSPORT_LOG_INFO 3
#define TRANSPORT_LOG_ABORT 4
#define TRANSPORT_LOG_TRACE 5

#if defined(NVSHMEM_x86_64)
#define MEM_BARRIER() asm volatile("mfence" ::: "memory")
#define STORE_BARRIER() asm volatile("sfence" ::: "memory")
#define LOAD_BARRIER() asm volatile("lfence" ::: "memory")
#elif defined(NVSHMEM_PPC64LE)
#define MEM_BARRIER() asm volatile("sync" ::: "memory")
#define STORE_BARRIER() MEM_BARRIER()
#define LOAD_BARRIER() MEM_BARRIER()
#elif defined(NVSHMEM_AARCH64)
#define MEM_BARRIER() asm volatile("dmb sy" ::: "memory")
#define STORE_BARRIER() asm volatile("dmb st" ::: "memory")
#define LOAD_BARRIER() MEM_BARRIER()
#else
#define MEM_BARRIER() asm volatile("" ::: "memory")
#define STORE_BARRIER() MEM_BARRIER()
#define LOAD_BARRIER() MEM_BARRIER()
#endif

#define INFO(LOG_LEVEL, fmt, ...)                                                  \
    do {                                                                           \
        if (LOG_LEVEL >= TRANSPORT_LOG_INFO) {                                     \
            fprintf(stderr, "%s %d " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
        }                                                                          \
    } while (0)

#define TRACE(LOG_LEVEL, fmt, ...)                                                 \
    do {                                                                           \
        if (LOG_LEVEL >= TRANSPORT_LOG_TRACE) {                                    \
            fprintf(stderr, "%s %d " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
        }                                                                          \
    } while (0)

#define LOAD_SYM(handle, symbol, funcptr)  \
    do {                                   \
        void **cast = (void **)&funcptr;   \
        void *tmp = dlsym(handle, symbol); \
        *cast = tmp;                       \
    } while (0)

#define LOAD_SYM_VERSION(handle, symbol, funcptr, version) \
    do {                                                   \
        void **cast = (void **)&funcptr;                   \
        void *tmp = dlvsym(handle, symbol, version);       \
        *cast = tmp;                                       \
    } while (0)

static inline int nvshmemt_common_get_log_level(struct nvshmemi_options_s *options) {
    if (!options->DEBUG_provided && !options->DEBUG_SUBSYS_provided) {
        return TRANSPORT_LOG_NONE;
    } else if (strncasecmp(options->DEBUG, "VERSION", 8) == 0) {
        return TRANSPORT_LOG_VERSION;
    } else if (strncasecmp(options->DEBUG, "WARN", 5) == 0) {
        return TRANSPORT_LOG_WARN;
    } else if (strncasecmp(options->DEBUG, "INFO", 5) == 0) {
        return TRANSPORT_LOG_INFO;
    } else if (strncasecmp(options->DEBUG, "ABORT", 6) == 0) {
        return TRANSPORT_LOG_ABORT;
    } else if (strncasecmp(options->DEBUG, "TRACE", 6) == 0) {
        return TRANSPORT_LOG_TRACE;
    }

    return TRANSPORT_LOG_INFO;
}

struct transport_mem_handle_info_cache;  // IWYU pragma: keep

struct nvshmemt_hca_info {
    char name[64];
    int port;
    int count;
    int found;
};

typedef int (*pci_path_cb)(int dev, char **pcipath, struct nvshmem_transport *transport);

int nvshmemt_parse_hca_list(const char *string, struct nvshmemt_hca_info *hca_list, int max_count,
                            int log_level);

int nvshmemt_mem_handle_cache_init(nvshmem_transport_t t,
                                   struct transport_mem_handle_info_cache **cache);
int nvshmemt_mem_handle_cache_add(nvshmem_transport_t t,
                                  struct transport_mem_handle_info_cache *cache, void *addr,
                                  void *mem_handle_info);
void *nvshmemt_mem_handle_cache_get(nvshmem_transport_t t,
                                    struct transport_mem_handle_info_cache *cache, void *addr);
void *nvshmemt_mem_handle_cache_get_by_idx(struct transport_mem_handle_info_cache *cache,
                                           size_t idx);
size_t nvshmemt_mem_handle_cache_get_size(struct transport_mem_handle_info_cache *cache);
int nvshmemt_mem_handle_cache_remove(nvshmem_transport_t t,
                                     struct transport_mem_handle_info_cache *cache, void *addr);
int nvshmemt_mem_handle_cache_fini(struct transport_mem_handle_info_cache *cache);

bool check_egm(void *addr, std::unordered_map<void *, size_t> *egm_map);
extern "C" {
int nvshmemt_init(nvshmem_transport_t *transport, struct nvshmemi_cuda_fn_table *table,
                  int api_version);
}

#endif
