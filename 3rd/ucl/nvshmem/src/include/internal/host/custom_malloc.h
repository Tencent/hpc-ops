/*
 * Copyright (c) 2016-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

/*
 *   mspace is an opaque type representing an independent
 *     region of space that supports mspace_malloc, etc.
 *     */
#ifndef NVSHMEMI_CUSTOM_MALLOC_H
#define NVSHMEMI_CUSTOM_MALLOC_H

#include <stddef.h>  // for size_t
#include <map>       // for map
#include <utility>   // for pair

#define NVSHMEMI_MALLOC_ALIGNMENT ((size_t)512U)

class mspace {
   private:
    /* free_chunks_start is mapping of start address of each free chunk to size of that chunk */
    /* free_chunks_end is mapping of end address of each free chunk to size of that chunk */
    std::map<void *, size_t> free_chunks_start, free_chunks_end;
    /* in_use_cunks is a mapping of each in use chunks start address to size of the chunk */
    std::map<void *, size_t> inuse_chunks;
    size_t total_size = 0; /* size of total space managed by mspace */
   public:
    mspace() {}
    mspace(void *base, size_t capacity);
    void print();
    void add_free_chunk(char *base, size_t capacity);
    void add_new_chunk(void *base, size_t capacity);
    int track_large_chunks(int enable);
    void *allocate(size_t bytes);
    void deallocate(void *mem);
    void *allocate_zeroed(size_t n_elements, size_t elem_size);
    void *allocate_aligned(size_t alignment, size_t bytes);
    void *reallocate(void *ptr, size_t size);
    bool checkInuse(void *ptr, size_t size);
    void *get_startInusePtr() {
        if (inuse_chunks.empty()) {
            return NULL;
        }
        return inuse_chunks.begin()->first;
    }
    std::map<void *, size_t> *get_inuse_chunks() { return &inuse_chunks; }
};

#endif
