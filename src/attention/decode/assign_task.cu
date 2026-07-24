// Copyright 2025 hpc-ops authors

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

#include "src/attention/decode/decode.h"
#include "src/attention/decode/sched_task_info.h"
#include "src/utils/utils.cuh"
#include "src/utils/utils.h"

namespace hpc {
namespace attention {
namespace decode {
namespace kernels {

template <int kTaskStride>
__device__ __forceinline__ void store_task(int* dst_ptr,
                                           const dynamic::TaskScheduleInfo& task_info) {
  auto task_info_as_int4 = reinterpret_cast<const vec_t<int, 4>*>(&task_info);
#pragma unroll
  for (int i = 0; i < kTaskStride / 4; i++) {
    store(dst_ptr + 4 * i, task_info_as_int4[i]);
  }
}

template <int kTaskStride>
__device__ __forceinline__ dynamic::TaskScheduleInfo load_task(int* src_ptr) {
  dynamic::TaskScheduleInfo task_info;
  auto task_info_as_int4 = reinterpret_cast<vec_t<int, 4>*>(&task_info);
#pragma unroll
  for (int i = 0; i < kTaskStride / 4; i++) {
    task_info_as_int4[i] = load<int, 4>(src_ptr + 4 * i);
  }
  return task_info;
}

// Scheduling loop order: outer over ihead_kv, inner over ibatch.
template <int kMaxNumBatch, int kTileN>
__global__ void assign_attention_decode_task_kernel(int* task_map_ptr, const int* num_seq_kvcache,
                                                    int num_batch, int num_head_kv, int num_seq_q,
                                                    bool new_kv_included, int min_process_len,
                                                    int num_total_ctas, int max_splitk) {
  __shared__ int num_seqkvs[kMaxNumBatch];
  __shared__ int num_tiles[kMaxNumBatch];
  __shared__ int smem_total_tiles[1];

  int icta = blockIdx.x;
  int idx = threadIdx.x;

  int total_tiles_per_head = 0;

  int max_num_batch = task_map_ptr[3];
  if (idx == 0) {
    smem_total_tiles[0] = 0;
  }
  __syncthreads();

  for (int ibatch = idx; ibatch < num_batch; ibatch += blockDim.x) {
    // Skip batches whose total KV length is zero — nothing to attend to.
    int num_seqkv =
        (new_kv_included ? num_seq_kvcache[ibatch] : num_seq_kvcache[ibatch] + num_seq_q);
    num_seqkvs[ibatch] = num_seqkv;
    num_tiles[ibatch] = (num_seqkv + kTileN - 1) / kTileN;
    total_tiles_per_head += num_tiles[ibatch];
  }

  atomicAdd(&smem_total_tiles[0], total_tiles_per_head);
  __syncthreads();

  int total_tiles_all_heads = smem_total_tiles[0] * num_head_kv;

  int num_tile_per_cta = std::max((total_tiles_all_heads + num_total_ctas - 1) / num_total_ctas,
                                  min_process_len / kTileN);

  if (icta == 0 && idx == 1) {
    task_map_ptr[0] = num_tile_per_cta + 1;
    task_map_ptr[1] = num_total_ctas;
  }

  if (idx != 0) {
    return;
  }

  // Fast-forward state to this CTA's starting (ihead_kv, ibatch, chunk)
  int ihead_kv = 0;
  int ibatch = 0;
  int num_chunks = 0;
  int start_tiles = 0;
  int num_tile = num_tiles[0];

  for (int i = 0; i < icta; i++) {
    int bucket = num_tile_per_cta;
    while (bucket > 0 && ihead_kv < num_head_kv) {
      // Skip (ihead_kv, ibatch) with no tiles (e.g. empty KV cache).
      if (num_tile <= 0) {
        ibatch++;
        if (ibatch >= num_batch) {
          ibatch = 0;
          ihead_kv++;
          if (ihead_kv >= num_head_kv) {
            break;
          }
        }
        num_tile = num_tiles[ibatch];
        continue;
      }

      int add_tiles = std::min(num_tile, bucket);
      if (num_chunks == max_splitk - 1) {
        add_tiles = num_tile;
      }

      num_chunks++;
      start_tiles += add_tiles;
      num_tile -= add_tiles;
      bucket -= add_tiles;
      // forward to next batch/head
      if (num_tile <= 0) {
        ibatch++;
        if (ibatch >= num_batch) {
          ibatch = 0;
          ihead_kv++;
          if (ihead_kv >= num_head_kv) {
            break;
          }
        }
        num_tile = num_tiles[ibatch];
        num_chunks = 0;
        start_tiles = 0;
      }
    }
  }

  constexpr int kTaskStride = dynamic::kTaskScheduleInfoStride;
  int max_num_batch_with_head = max_num_batch * num_head_kv;
  int max_num_batch_with_head_pad =
      (max_num_batch_with_head + kTaskStride - 1) / kTaskStride * kTaskStride;
  int num_cta_count_pad = (num_total_ctas + kTaskStride - 1) / kTaskStride * kTaskStride;

  auto* task_map_chunk_ptr =
      task_map_ptr + kTaskStride * ((num_tile_per_cta + 1) * num_total_ctas + 1);
  auto* task_map_sm_finish_ptr = task_map_chunk_ptr + max_num_batch_with_head_pad;
  auto* num_task_map_ptr = task_map_sm_finish_ptr + num_cta_count_pad;
  task_map_ptr += ((num_tile_per_cta + 1) * icta + 1) * kTaskStride;

  int itask = 0;
  int bucket = num_tile_per_cta;

  // When the last chunk of a (head, batch) has num_seqkvcache < 0 after
  // the causal adjustment, Q tokens straddle the prev/cur boundary. We
  // keep cur (clamped to 0) and propagate the overflow to prev so it gets
  // is_casual_chunk=1 with a correctly reduced num_seqkvcache.
  // If prev is in the same CTA we fix it in-place; otherwise we defer
  // via wait_batch and fix it after the CTA finishes (cross-CTA sync).
  int wait_batch = -1;
  int wait_batch_num_seqkvcache_overflow = 0;

  while (bucket > 0 && ihead_kv < num_head_kv) {
    // Skip (ihead_kv, ibatch) with no tiles (e.g. empty KV cache).
    if (num_tile <= 0) {
      ibatch++;
      if (ibatch >= num_batch) {
        ibatch = 0;
        ihead_kv++;
      }
      num_tile = num_tiles[ibatch];
      if (num_tile == 0) {
        task_map_chunk_ptr[ibatch] = 0;
      }
      continue;
    }

    int add_tiles = std::min(num_tile, bucket);
    int num_seqkv = num_seqkvs[ibatch];

    if (num_chunks == max_splitk - 1) {
      add_tiles = num_tile;
    }

    dynamic::TaskScheduleInfo task_info;
    task_info.ihead_kv = ihead_kv;
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
      // int raw_seqkvcache = task_info.num_seqkvcache - num_seq_q;
      // task_info.num_seqkvcache = std::max(raw_seqkvcache, 0);
      // task_info.num_tile_full = task_info.num_seqkvcache / kTileN;

      task_info.num_seqkvcache -= num_seq_q;
      task_info.num_tile_full = std::max(task_info.num_seqkvcache / kTileN, 0);
      task_map_chunk_ptr[ihead_kv * num_batch + ibatch] = num_chunks;

      // When num_seqkvcache < 0, some Q tokens overflow into the previous
      // chunk. Mark prev as causal and reduce its num_seqkvcache accordingly.
      if (task_info.num_seqkvcache < 0) {
        if (itask != 0) {
          dynamic::TaskScheduleInfo last_task_info =
              load_task<kTaskStride>(task_map_ptr + (itask - 1) * kTaskStride);
          last_task_info.is_casual_chunk = 1;
          last_task_info.num_seqkvcache += task_info.num_seqkvcache;
          last_task_info.num_tile_full = std::max(last_task_info.num_seqkvcache / kTileN, 0);
          store_task<kTaskStride>(task_map_ptr + (itask - 1) * kTaskStride, last_task_info);
        } else {
          wait_batch = ibatch;
          wait_batch_num_seqkvcache_overflow = task_info.num_seqkvcache;
        }
      }

      // Advance (ihead_kv, ibatch).
      ibatch++;
      if (ibatch >= num_batch) {
        ibatch = 0;
        ihead_kv++;
      }
      num_tile = num_tiles[ibatch];
      num_chunks = 0;
      start_tiles = 0;
    }

    store_task<kTaskStride>(task_map_ptr + itask * kTaskStride, task_info);
    itask++;
  }

  task_map_ptr[itask * kTaskStride] = -1;
  task_map_ptr[itask * kTaskStride + 1] = -1;
  num_task_map_ptr[icta] = itask;

  // Fill every remaining slot in this bin with a terminator too, so
  // stragglers / early-terminating walkers stay safe on zero-init memory.
  for (int slot = itask + 1; slot <= num_tile_per_cta; slot++) {
    task_map_ptr[slot * kTaskStride] = -1;
    task_map_ptr[slot * kTaskStride + 1] = -1;
  }

  if (wait_batch >= 0) {
    if (icta > 0) {
      task_map_ptr -= (num_tile_per_cta + 1) * kTaskStride;
      __threadfence();
      task_map_sm_finish_ptr[icta] = 1;
      while (load_global_volatile(task_map_sm_finish_ptr + (icta - 1)) < 1) {
      }
      __threadfence();
      int last_task_id = num_task_map_ptr[icta - 1] - 1;
      if (last_task_id >= 0) {
        dynamic::TaskScheduleInfo last_task_info =
            load_task<kTaskStride>(task_map_ptr + last_task_id * kTaskStride);
        last_task_info.is_casual_chunk = 1;
        last_task_info.num_seqkvcache += wait_batch_num_seqkvcache_overflow;
        last_task_info.num_tile_full = std::max(last_task_info.num_seqkvcache / kTileN, 0);
        store_task<kTaskStride>(task_map_ptr + last_task_id * kTaskStride, last_task_info);
      }
      __threadfence();
      task_map_sm_finish_ptr[icta] = 2;
    } else {
      __threadfence();
      task_map_sm_finish_ptr[icta] = 2;
    }
  } else {
    __threadfence();
    task_map_sm_finish_ptr[icta] = 2;
  }

  if (icta == 0) {
    vec_t<int, 4> zeros;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      zeros[i] = 0;
    }

    for (int i = 0; i < num_total_ctas; i++) {
      while (load_global_volatile(task_map_sm_finish_ptr + i) != 2) {
      }
    }

    for (int i = 0; i < (num_total_ctas + 3) / 4; i++) {
      store(task_map_sm_finish_ptr + 4 * i, zeros);
    }
  }
}

}  // namespace kernels

bool assign_attention_decode_task_async(int* task_map_ptr, const int* num_seq_kvcache,
                                        int num_total_ctas, int num_batch, int num_head_kv,
                                        int num_seq_q, int tilen, bool new_kv_included,
                                        int min_process_len, cudaStream_t stream) {
  dim3 grid(num_total_ctas);
  dim3 block(128);

  int num_sm = get_sm_count();

  constexpr int kMaxNumBatch = 2048;

  auto launch = [&](auto tilen_tag) {
    constexpr int kTileN = decltype(tilen_tag)::value;
    int max_splitk = num_total_ctas;

    kernels::assign_attention_decode_task_kernel<kMaxNumBatch, kTileN><<<grid, block, 0, stream>>>(
        task_map_ptr, num_seq_kvcache, num_batch, num_head_kv, num_seq_q, new_kv_included,
        min_process_len, num_total_ctas, max_splitk);
  };

  if (tilen == 64) {
    launch(std::integral_constant<int, 64>{});
  } else if (tilen == 128) {
    launch(std::integral_constant<int, 128>{});
  }

  return true;
}

std::pair<std::vector<dynamic::TaskScheduleInfo>, std::vector<int>>
assign_attention_decode_task_sync(const int* num_seq_kvcache, int num_total_ctas, int num_batch,
                                  int num_head_kv, int num_seq_q, int tilen, bool new_kv_included,
                                  int min_process_len) {
  std::vector<int> num_seqkvs(num_batch);
  std::vector<int> num_tiles(num_batch);

  int total_tiles_per_head = 0;
  for (int ibatch = 0; ibatch < num_batch; ibatch++) {
    // Skip batches whose KV cache is empty
    int num_seqkv =
        (new_kv_included ? num_seq_kvcache[ibatch] : num_seq_kvcache[ibatch] + num_seq_q);
    num_seqkvs[ibatch] = num_seqkv;
    num_tiles[ibatch] = (num_seqkv + tilen - 1) / tilen;
    total_tiles_per_head += num_tiles[ibatch];
  }

  int total_tiles_all_heads = total_tiles_per_head * num_head_kv;

  int num_tile_per_cta = std::max((total_tiles_all_heads + num_total_ctas - 1) / num_total_ctas,
                                  min_process_len / tilen);

  std::vector<dynamic::TaskScheduleInfo> tasks(num_total_ctas * (num_tile_per_cta + 1));

  // num_chunks has one entry per (ihead_kv, ibatch). The trailing slot
  // (num_chunks[num_head_kv * num_batch]) is repurposed to carry
  // num_tile_per_cta+1 for the CPU entry wrapper
  std::vector<int> num_chunks(num_head_kv * num_batch + 1, 0);
  std::vector<int> start_tiles(num_batch * num_head_kv, 0);
  std::vector<int> chunks_in_progress(num_batch * num_head_kv, 0);
  std::vector<int> num_tiles_left(num_batch * num_head_kv, 0);
  for (int h = 0; h < num_head_kv; h++) {
    for (int b = 0; b < num_batch; b++) {
      num_tiles_left[h * num_batch + b] = num_tiles[b];
    }
  }

  int ihead_kv = 0;
  int ibatch = 0;

  int last_cta = 0;
  int last_task = 0;

  for (int icta = 0; icta < num_total_ctas; icta++) {
    int bucket = num_tile_per_cta;
    int itask = 0;
    while (bucket > 0 && ihead_kv < num_head_kv) {
      int idx = ihead_kv * num_batch + ibatch;
      int num_tile = num_tiles_left[idx];

      // Skip (ihead_kv, ibatch) with no tiles
      if (num_tile <= 0) {
        ibatch++;
        if (ibatch >= num_batch) {
          ibatch = 0;
          ihead_kv++;
          if (ihead_kv >= num_head_kv) {
            break;
          }
        }
        continue;
      }

      int add_tiles = std::min(num_tile, bucket);
      int num_seqkv = num_seqkvs[ibatch];

      if (chunks_in_progress[idx] == num_total_ctas - 1) {
        add_tiles = num_tile;
      }

      dynamic::TaskScheduleInfo task_info;
      task_info.ihead_kv = ihead_kv;
      task_info.ibatch = ibatch;
      task_info.ichunk = chunks_in_progress[idx];
      task_info.iseq_start = start_tiles[idx] * tilen;
      task_info.num_seqkv = std::min(add_tiles * tilen, num_seqkv - task_info.iseq_start);
      task_info.num_seqkvcache = task_info.num_seqkv;
      task_info.num_tile_kv = (task_info.num_seqkv + tilen - 1) / tilen;
      task_info.num_tile_full = task_info.num_seqkvcache / tilen;
      task_info.is_casual_chunk = 0;

      tasks[icta * (num_tile_per_cta + 1) + itask] = task_info;

      itask++;
      chunks_in_progress[idx]++;
      start_tiles[idx] += add_tiles;
      num_tiles_left[idx] -= add_tiles;
      bucket -= add_tiles;

      if (num_tiles_left[idx] <= 0) {
        // last chunk
        auto& cur = tasks[icta * (num_tile_per_cta + 1) + itask - 1];
        cur.is_casual_chunk = 1;
        cur.num_seqkvcache -= num_seq_q;
        cur.num_tile_full = std::max(cur.num_seqkvcache / tilen, 0);
        num_chunks[ihead_kv * num_batch + ibatch] = chunks_in_progress[idx];

        // When raw_seqkvcache < 0, some Q tokens overflow into the previous
        // chunk. Mark prev as causal and reduce its num_seqkvcache accordingly.
        if (cur.num_seqkvcache < 0) {
          auto& prev = tasks[last_cta * (num_tile_per_cta + 1) + last_task];
          prev.is_casual_chunk = 1;
          prev.num_seqkvcache += cur.num_seqkvcache;
          prev.num_tile_full = std::max(prev.num_seqkvcache / tilen, 0);
        }

        // advance (ihead_kv, ibatch)
        ibatch++;
        if (ibatch >= num_batch) {
          ibatch = 0;
          ihead_kv++;
        }
      }
      last_task = itask - 1;
    }

    last_cta = icta;

    tasks[icta * (num_tile_per_cta + 1) + itask].ihead_kv = -1;
    tasks[icta * (num_tile_per_cta + 1) + itask].ibatch = -1;
    // Fill unused tail slots with -1 terminators too — consistent with CUDA.
    for (int slot = itask + 1; slot <= num_tile_per_cta; slot++) {
      tasks[icta * (num_tile_per_cta + 1) + slot].ihead_kv = -1;
      tasks[icta * (num_tile_per_cta + 1) + slot].ibatch = -1;
    }
  }

  num_chunks[num_head_kv * num_batch] = num_tile_per_cta + 1;

  return std::make_pair(tasks, num_chunks);
}

}  // namespace decode
}  // namespace attention
}  // namespace hpc
