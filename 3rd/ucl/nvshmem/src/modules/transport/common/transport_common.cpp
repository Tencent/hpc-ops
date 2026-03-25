/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

#include "transport_common.h"
#include <stdint.h>  // for uint64_t, uintptr_t
#include <stdlib.h>  // for atoi, calloc, free, realloc
#include <utility>
#include "non_abi/nvshmemx_error.h"  // for NVSHMEMI_ERROR_PRINT, NVSHMEMX_E...

struct transport_mem_handle_info_cache {
    void **cache;
    uint64_t size;
    uint64_t address_granularity;
    uintptr_t address_mask;
};

int nvshmemt_parse_hca_list(const char *string, struct nvshmemt_hca_info *hca_list, int max_count,
                            int log_level) {
    if (!string) return 0;

    const char *ptr = string;
    // Ignore "^" name, will be detected outside of this function
    if (ptr[0] == '^') ptr++;

    int if_num = 0;
    int if_counter = 0;
    int segment_counter = 0;
    char c;
    do {
        c = *ptr;
        if (c == ':') {
            if (segment_counter == 0) {
                if (if_counter > 0) {
                    hca_list[if_num].name[if_counter] = '\0';
                    hca_list[if_num].port = atoi(ptr + 1);
                    hca_list[if_num].found = 0;
                    if_num++;
                    if_counter = 0;
                    segment_counter++;
                }
            } else {
                hca_list[if_num - 1].count = atoi(ptr + 1);
                segment_counter = 0;
            }
            c = *(ptr + 1);
            while (c != ',' && c != ':' && c != '\0') {
                ptr++;
                c = *(ptr + 1);
            }
        } else if (c == ',' || c == '\0') {
            if (if_counter > 0) {
                hca_list[if_num].name[if_counter] = '\0';
                hca_list[if_num].found = 0;
                if_num++;
                if_counter = 0;
            }
            segment_counter = 0;
        } else {
            if (if_counter == 0) {
                hca_list[if_num].port = -1;
                hca_list[if_num].count = 1;
            }
            hca_list[if_num].name[if_counter] = c;
            if_counter++;
        }
        ptr++;
    } while (if_num < max_count && c);

    INFO(log_level, "Begin - Parsed HCA list provided by user - ");
    for (int i = 0; i < if_num; i++) {
        INFO(log_level,
             "Parsed HCA list provided by user - i=%d (of %d), name=%s, port=%d, count=%d", i,
             if_num, hca_list[i].name, hca_list[i].port, hca_list[i].count);
    }
    INFO(log_level, "End - Parsed HCA list provided by user");

    return if_num;
}

int nvshmemt_mem_handle_cache_init(nvshmem_transport_t t,
                                   struct transport_mem_handle_info_cache **cache) {
    struct transport_mem_handle_info_cache *cache_pointer;

    if (cache == NULL) {
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    *cache = (struct transport_mem_handle_info_cache *)calloc(
        1, sizeof(struct transport_mem_handle_info_cache));
    if (!(*cache)) {
        NVSHMEMI_ERROR_PRINT("Unable to allocate mem handle cache in transport code.");
        return NVSHMEMX_ERROR_OUT_OF_MEMORY;
    }

    cache_pointer = *cache;

    cache_pointer->cache = (void **)calloc(1000, sizeof(void *));
    if (!(cache_pointer->cache)) {
        NVSHMEMI_ERROR_PRINT("Unable to allocate mem handle cache in transport code.");
        return NVSHMEMX_ERROR_OUT_OF_MEMORY;
    }
    cache_pointer->size = 1000;
    cache_pointer->address_granularity = 1ULL << t->log2_cumem_granularity;
    cache_pointer->address_mask = (uintptr_t)(~(cache_pointer->address_granularity - 1));

    return NVSHMEMX_SUCCESS;
}

int nvshmemt_mem_handle_cache_add(nvshmem_transport_t t,
                                  struct transport_mem_handle_info_cache *cache, void *addr,
                                  void *mem_handle_info) {
    uint64_t addr_offset;
    uint64_t arr_idx;

    if (addr < t->heap_base) {
        NVSHMEMI_ERROR_PRINT("Unable to process pointers outside of the heap.");
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    addr_offset = (uint64_t)((char *)addr - (char *)t->heap_base);
    if (addr_offset % cache->address_granularity) {
        NVSHMEMI_ERROR_PRINT("Unable to process unaligned pointers.");
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    arr_idx = addr_offset / cache->address_granularity;

    if (arr_idx >= cache->size) {
        size_t new_cache_size = cache->size * 2 > arr_idx ? cache->size * 2 : arr_idx + 1;
        void *new_cache;
        new_cache = realloc(cache->cache, new_cache_size);
        if (new_cache == NULL) {
            NVSHMEMI_ERROR_PRINT("Unable to reallocate larger heap cache.");
            return NVSHMEMX_ERROR_OUT_OF_MEMORY;
        }

        cache->cache = (void **)new_cache;
        cache->size = new_cache_size;
    }

    cache->cache[arr_idx] = mem_handle_info;
    return NVSHMEMX_SUCCESS;
}

void *nvshmemt_mem_handle_cache_get(nvshmem_transport_t t,
                                    struct transport_mem_handle_info_cache *cache, void *addr) {
    uintptr_t addr_offset;
    uintptr_t aligned_addr;
    uint64_t arr_idx;

    if (addr < t->heap_base) {
        NVSHMEMI_ERROR_PRINT("Unable to process pointers outside of the heap.");
        return NULL;
    }

    addr_offset = (uintptr_t)((char *)addr - (char *)t->heap_base);
    aligned_addr = addr_offset & cache->address_mask;
    arr_idx = (uint64_t)aligned_addr / cache->address_granularity;

    if (arr_idx >= cache->size) {
        NVSHMEMI_ERROR_PRINT("Address not registered. Unable to get handle for it.");
        return NULL;
    }

    return cache->cache[arr_idx];
}

void *nvshmemt_mem_handle_cache_get_by_idx(struct transport_mem_handle_info_cache *cache,
                                           size_t idx) {
    if (idx > cache->size) {
        NVSHMEMI_ERROR_PRINT("Index out of bounds. Unable to get handle for it.");
    }
    return cache->cache[idx];
}
size_t nvshmemt_mem_handle_cache_get_size(struct transport_mem_handle_info_cache *cache) {
    return cache->size;
}

int nvshmemt_mem_handle_cache_remove(nvshmem_transport_t t,
                                     struct transport_mem_handle_info_cache *cache, void *addr) {
    uint64_t addr_offset;
    uint64_t arr_idx;

    if (addr < t->heap_base) {
        NVSHMEMI_ERROR_PRINT("Unable to process pointers outside of the heap.");
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    addr_offset = (uint64_t)((char *)addr - (char *)t->heap_base);
    if (addr_offset % cache->address_granularity) {
        NVSHMEMI_ERROR_PRINT("Unable to process unaligned pointers.");
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    arr_idx = addr_offset / cache->address_granularity;

    if (arr_idx >= cache->size) {
        NVSHMEMI_ERROR_PRINT("Address not registered. Unable to unregister it.");
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    cache->cache[arr_idx] = NULL;
    return NVSHMEMX_SUCCESS;
}

int nvshmemt_mem_handle_cache_fini(struct transport_mem_handle_info_cache *cache) {
    if (cache == NULL) {
        NVSHMEMI_ERROR_PRINT("Mem handle cache not initialized, cannot finalize it.");
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    if (cache->cache) {
        free(cache->cache);
    }

    free(cache);

    return NVSHMEMX_SUCCESS;
}

bool check_egm(void *addr, std::unordered_map<void *, size_t> *egm_map) {
    if (egm_map == NULL) {
        return false;
    }
    if (egm_map->count(addr)) {
        return true;
    }

    for (auto iter = egm_map->begin(); iter != egm_map->end(); ++iter) {
        if ((addr >= iter->first) && ((char *)addr < ((char *)iter->first + iter->second))) {
            return true;
        }
    }
    return false;
}
