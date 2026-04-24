// Copyright 2025 hpc-ops authors

#include <algorithm>
#include <cstdio>
#include <utility>
#include <vector>

#include "src/attention/decode/decode.h"
#include "src/utils/utils.cuh"
#include "src/utils/utils.h"

namespace hpc {
namespace attention {

namespace kernels {

template <int kSchedTaskViewAsIntCount>
__device__ __forceinline__ void store_task(int* dst_ptr, TaskScheduleInfo& task_info) {
  auto task_info_as_int4 = reinterpret_cast<vec_t<int, 4>*>(&task_info);
#pragma unroll
  for (int i = 0; i < kSchedTaskViewAsIntCount / 4; i++) {
    store(dst_ptr + 4 * i, task_info_as_int4[i]);
  }
}

template <int kSchedTaskViewAsIntCount>
__device__ __forceinline__ auto load_task(int* src_ptr) {
  TaskScheduleInfo task_info;
  auto task_info_as_int4 = reinterpret_cast<vec_t<int, 4>*>(&task_info);
#pragma unroll
  for (int i = 0; i < kSchedTaskViewAsIntCount / 4; i++) {
    task_info_as_int4[i] = load<int, 4>(src_ptr + 4 * i);
  }
  return task_info;
}

template <int kMaxNumBatch, int kMaxSplitK, int kTileN>
__global__ void assign_attention_decode_task_kernel(int* task_map_ptr, const int* num_seq_kvcache,
                                                    int num_batch, int num_seq_q,
                                                    bool new_kv_included, int min_process_len,
                                                    int num_sm_count) {
  __shared__ int num_seqkvs[kMaxNumBatch];
  __shared__ int num_tiles[kMaxNumBatch];
  __shared__ int smem_total_tiles[1];

  int ism = blockIdx.x;
  int idx = threadIdx.x;

  int total_tiles = 0;

  int max_num_batch = task_map_ptr[2];
  if (idx == 0) {
    smem_total_tiles[0] = 0;
  }
  __syncthreads();

  for (int ibatch = idx; ibatch < num_batch; ibatch += blockDim.x) {
    int num_seqkv =
        (new_kv_included ? num_seq_kvcache[ibatch] : num_seq_kvcache[ibatch] + num_seq_q);
    num_seqkvs[ibatch] = num_seqkv;
    num_tiles[ibatch] = (num_seqkv + kTileN - 1) / kTileN;
    total_tiles += num_tiles[ibatch];
  }

  atomicAdd(&smem_total_tiles[0], total_tiles);
  __syncthreads();

  total_tiles = smem_total_tiles[0];

  int num_tile_per_sm =
      std::max((total_tiles + num_sm_count - 1) / num_sm_count, min_process_len / kTileN);

  if (ism == 0 && idx == 1) {
    task_map_ptr[0] = num_tile_per_sm + 1;
  }

  if (idx != 0) {
    return;
  }

  int ibatch = 0;
  int num_chunks = 0;
  int start_tiles = 0;
  int num_tile = num_tiles[ibatch];

  for (int i = 0; i < ism; i++) {
    int bucket = num_tile_per_sm;
    while (bucket > 0 && ibatch < num_batch) {
      int add_tiles = std::min(num_tile, bucket);
      if (num_chunks == kMaxSplitK - 1) {
        add_tiles = num_tile;
      }
      num_chunks++;
      start_tiles += add_tiles;
      num_tile -= add_tiles;
      bucket -= add_tiles;
      if (num_tile <= 0) {
        ibatch++;
        num_tile = num_tiles[ibatch];
        num_chunks = 0;
        start_tiles = 0;
      }
    }
  }

  constexpr int kSchedTaskViewAsIntCount = sizeof(TaskScheduleInfo) / sizeof(int);
  int max_num_batch_pad = (max_num_batch + kSchedTaskViewAsIntCount - 1) /
                          kSchedTaskViewAsIntCount * kSchedTaskViewAsIntCount;
  int num_sm_count_pad = (num_sm_count + kSchedTaskViewAsIntCount - 1) / kSchedTaskViewAsIntCount *
                         kSchedTaskViewAsIntCount;

  auto* task_map_chunk_ptr =
      task_map_ptr + kSchedTaskViewAsIntCount * ((num_tile_per_sm + 1) * num_sm_count + 1);
  auto* task_map_sm_finish_ptr = task_map_chunk_ptr + max_num_batch_pad;
  auto* num_task_map_ptr = task_map_sm_finish_ptr + num_sm_count_pad;
  task_map_ptr += ((num_tile_per_sm + 1) * ism + 1) * kSchedTaskViewAsIntCount;

  int itask = 0;
  int bucket = num_tile_per_sm;

  int wait_batch = -1;
  int wait_batch_num_seqkvcache;

  while (bucket > 0 && ibatch < num_batch) {
    int add_tiles = std::min(num_tile, bucket);

    int num_seqkv = num_seqkvs[ibatch];

    if (num_seqkv <= 0) {
      ibatch++;
      task_map_chunk_ptr[ibatch] = 0;
      continue;
    }

    if (num_chunks == kMaxSplitK - 1) {
      add_tiles = num_tile;
    }

    TaskScheduleInfo task_info;
    task_info.ibatch = ibatch;
    task_info.ichunk = num_chunks;
    task_info.iseq_start = start_tiles * kTileN;
    task_info.num_seqkv = std::min(add_tiles * kTileN, num_seqkv - task_info.iseq_start);

    task_info.num_seqkvcache = task_info.num_seqkv;
    task_info.num_tile_kv = (task_info.num_seqkv + kTileN - 1) / kTileN;
    task_info.num_tile_full = task_info.num_seqkvcache / kTileN;
    task_info.is_casual_chunk = 0;

    num_chunks++;
    start_tiles += add_tiles;
    num_tile -= add_tiles;
    bucket -= add_tiles;

    if (num_tile <= 0) {
      task_info.is_casual_chunk = 1;
      task_info.num_seqkvcache -= num_seq_q;
      task_info.num_tile_full = std::max(task_info.num_seqkvcache / kTileN, 0);
      task_map_chunk_ptr[ibatch] = num_chunks;

      if (task_info.num_seqkvcache < 0) {
        if (itask != 0) {
          TaskScheduleInfo last_task_info = load_task<kSchedTaskViewAsIntCount>(
              task_map_ptr + (itask - 1) * kSchedTaskViewAsIntCount);
          // load last task
          last_task_info.is_casual_chunk = 1;
          last_task_info.num_seqkvcache += task_info.num_seqkvcache;
          last_task_info.num_tile_full = std::max(last_task_info.num_seqkvcache / kTileN, 0);
          // store last task
          store_task<kSchedTaskViewAsIntCount>(
              task_map_ptr + (itask - 1) * kSchedTaskViewAsIntCount, task_info);
        } else {
          // wait last sm finish
          wait_batch = ibatch;
          wait_batch_num_seqkvcache = task_info.num_seqkvcache;
        }
      }

      ibatch++;
      num_tile = num_tiles[ibatch];
      num_chunks = 0;
      start_tiles = 0;
    }

    store_task<kSchedTaskViewAsIntCount>(task_map_ptr + itask * kSchedTaskViewAsIntCount,
                                         task_info);
    itask++;
  }

  task_map_ptr[itask * kSchedTaskViewAsIntCount] = -1;
  num_task_map_ptr[ism] = itask;

  if (wait_batch >= 0) {
    task_map_ptr -= (num_tile_per_sm + 1) * kSchedTaskViewAsIntCount;
    task_map_sm_finish_ptr[ism] = 1;
    // wait process corner
    while (load_global_volatile(task_map_sm_finish_ptr + (ism - 1)) < 1) {
    }
    __threadfence();
    int last_task_id = num_task_map_ptr[ism - 1] - 1;
    TaskScheduleInfo last_task_info =
        load_task<kSchedTaskViewAsIntCount>(task_map_ptr + last_task_id * kSchedTaskViewAsIntCount);
    // load last task
    last_task_info.is_casual_chunk = 1;
    last_task_info.num_seqkvcache += wait_batch_num_seqkvcache;
    last_task_info.num_tile_full = std::max(last_task_info.num_seqkvcache / kTileN, 0);
    store_task<kSchedTaskViewAsIntCount>(task_map_ptr + last_task_id * kSchedTaskViewAsIntCount,
                                         last_task_info);

    task_map_sm_finish_ptr[ism] = 2;
  } else {
    task_map_sm_finish_ptr[ism] = 2;
  }

  if (ism == 0) {
    vec_t<int, 4> zeros;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      zeros[i] = 0;
    }

    // check all finish
    for (int i = 0; i < num_sm_count; i++) {
      while (load_global_volatile(task_map_sm_finish_ptr + i) != 2) {
      }
    }

    // unset flag
    for (int i = 0; i < (num_sm_count + 3) / 4; i++) {
      store(task_map_sm_finish_ptr + 4 * i, zeros);
    }
  }
}

}  // namespace kernels

std::pair<std::vector<TaskScheduleInfo>, std::vector<int>> assign_attention_decode_task_sync(
    const int* num_seq_kvcache, int num_batch, int num_head_kv, int num_seq_q, bool new_kv_included,
    int min_process_len) {
  constexpr int kTileN = 128;
  constexpr int kMaxSplitK = 64;

  int num_sm_count = get_sm_count();

  num_sm_count /= num_head_kv;

  std::vector<int> num_seqkvs(num_batch);
  std::vector<int> num_tiles(num_batch);
  std::vector<int> last_chunk_casual_token(num_batch);
  int total_tiles = 0;
  for (int ibatch = 0; ibatch < num_batch; ibatch++) {
    int num_seqkv =
        (new_kv_included ? num_seq_kvcache[ibatch] : num_seq_kvcache[ibatch] + num_seq_q);
    num_seqkvs[ibatch] = num_seqkv;
    num_tiles[ibatch] = (num_seqkv + kTileN - 1) / kTileN;
    total_tiles += num_tiles[ibatch];
  }

  int num_tile_per_sm =
      std::max((total_tiles + num_sm_count - 1) / num_sm_count, min_process_len / kTileN);

  std::vector<TaskScheduleInfo> tasks(num_sm_count * (num_tile_per_sm + 1));

  std::vector<int> num_chunks(num_batch + 1, 0);
  std::vector<int> start_tiles(num_batch, 0);

  int ibatch = 0;

  int last_sm = 0;
  int last_task = 0;
  for (int ism = 0; ism < num_sm_count; ism++) {
    int bucket = num_tile_per_sm;
    int itask = 0;
    while (bucket > 0 && ibatch < num_batch) {
      int num_tile = num_tiles[ibatch];
      int add_tiles = std::min(num_tile, bucket);

      int num_seqkv = num_seqkvs[ibatch];

      if (num_seqkv <= 0) {
        ibatch++;
        num_chunks[ibatch] = 0;
        continue;
      }

      if (num_chunks[ibatch] == kMaxSplitK - 1) {
        add_tiles = num_tile;
      }

      TaskScheduleInfo task_info;
      task_info.ibatch = ibatch;
      task_info.ichunk = num_chunks[ibatch];
      task_info.iseq_start = start_tiles[ibatch] * kTileN;
      task_info.num_seqkv = std::min(add_tiles * kTileN, num_seqkv - task_info.iseq_start);

      task_info.num_seqkvcache = task_info.num_seqkv;
      task_info.num_tile_kv = (task_info.num_seqkv + kTileN - 1) / kTileN;
      task_info.num_tile_full = task_info.num_seqkvcache / kTileN;
      task_info.is_casual_chunk = 0;

      tasks[ism * (num_tile_per_sm + 1) + itask] = task_info;

      itask++;
      num_chunks[ibatch]++;
      start_tiles[ibatch] += add_tiles;
      num_tiles[ibatch] -= add_tiles;
      bucket -= add_tiles;

      if (num_tiles[ibatch] <= 0) {
        // last chunk
        tasks[ism * (num_tile_per_sm + 1) + itask - 1].is_casual_chunk = 1;
        tasks[ism * (num_tile_per_sm + 1) + itask - 1].num_seqkvcache -= num_seq_q;
        tasks[ism * (num_tile_per_sm + 1) + itask - 1].num_tile_full =
            std::max(tasks[ism * (num_tile_per_sm + 1) + itask - 1].num_seqkvcache / kTileN, 0);

        // last - 1 chunk should be casual.
        if (tasks[ism * (num_tile_per_sm + 1) + itask - 1].num_seqkvcache < 0) {
          tasks[last_sm * (num_tile_per_sm + 1) + last_task].is_casual_chunk = 1;
          tasks[last_sm * (num_tile_per_sm + 1) + last_task].num_seqkvcache +=
              tasks[ism * (num_tile_per_sm + 1) + itask - 1].num_seqkvcache;
          tasks[last_sm * (num_tile_per_sm + 1) + last_task].num_tile_full = std::max(
              tasks[last_sm * (num_tile_per_sm + 1) + last_task].num_seqkvcache / kTileN, 0);
        }

        ibatch++;
      }
      last_task = itask - 1;
    }

    last_sm = ism;

    tasks[ism * (num_tile_per_sm + 1) + itask].ibatch = -1;
  }

  num_chunks[num_batch] = num_tile_per_sm + 1;

  return std::make_pair(tasks, num_chunks);
}

bool assign_attention_decode_task_async(int* task_map_ptr, const int* num_seq_kvcache,
                                        int num_batch, int num_head_kv, int num_seq_q,
                                        bool new_kv_included, int min_process_len,
                                        cudaStream_t stream) {
  int num_sm_count = get_sm_count();
  num_sm_count /= num_head_kv;

  dim3 grid(num_sm_count);
  dim3 block(128);

  constexpr int kMaxNumBatch = 2048;
  constexpr int kMaxSplitK = 64;
  constexpr int kTileN = 128;
  kernels::assign_attention_decode_task_kernel<kMaxNumBatch, kMaxSplitK, kTileN>
      <<<grid, block, 0, stream>>>(task_map_ptr, num_seq_kvcache, num_batch, num_seq_q,
                                   new_kv_included, min_process_len, num_sm_count);

  return true;
}

}  // namespace attention
}  // namespace hpc
