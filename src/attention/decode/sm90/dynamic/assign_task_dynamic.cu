// Copyright 2025 hpc-ops authors

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

#include "src/attention/decode/sm90/dynamic/decode_dynamic.h"
#include "src/attention/decode/sm90/dynamic/dynamic_sched_task_info.h"
#include "src/utils/utils.cuh"
#include "src/utils/utils.h"

namespace hpc {
namespace attention {
namespace decode {
namespace dynamic {

namespace kernels {

template <int kTaskStride>
__device__ __forceinline__ void store_task(int* dst_ptr, const SM90DynamicTaskInfo& task_info) {
  auto task_info_as_int4 = reinterpret_cast<const vec_t<int, 4>*>(&task_info);
#pragma unroll
  for (int i = 0; i < kTaskStride / 4; i++) {
    store(dst_ptr + 4 * i, task_info_as_int4[i]);
  }
}

template <int kTaskStride>
__device__ __forceinline__ SM90DynamicTaskInfo load_task(int* src_ptr) {
  SM90DynamicTaskInfo task_info;
  auto task_info_as_int4 = reinterpret_cast<vec_t<int, 4>*>(&task_info);
#pragma unroll
  for (int i = 0; i < kTaskStride / 4; i++) {
    task_info_as_int4[i] = load<int, 4>(src_ptr + 4 * i);
  }
  return task_info;
}

// sm90 dynamic attention-decode assigner at kTileN=64.
// Scheduling loop order: outer over ihead_kv, inner over ibatch.
template <int kMaxNumBatch, int kMaxSplitK, int kTileN>
__global__ void assign_attention_decode_task_sm90_dynamic_kernel(
    int* task_map_ptr, const int* num_seq_kvcache, int num_batch, int num_head_kv, int num_seq_q,
    bool new_kv_included, int min_process_len, int num_total_ctas) {
  __shared__ int num_seqkvs[kMaxNumBatch];
  __shared__ int num_tiles[kMaxNumBatch];
  __shared__ int smem_total_tiles[1];

  int icta = blockIdx.x;
  int idx = threadIdx.x;

  int total_tiles_per_head = 0;

  int max_num_batch = task_map_ptr[2];
  if (idx == 0) {
    smem_total_tiles[0] = 0;
  }
  __syncthreads();

  for (int ibatch = idx; ibatch < num_batch; ibatch += blockDim.x) {
    // Skip batches whose total KV length is zero — nothing to attend to.
    if (num_seq_kvcache[ibatch] == 0) {
      num_seqkvs[ibatch] = 0;
      num_tiles[ibatch] = 0;
      continue;
    }
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
  }

  // Clear num_chunks table to prevent stale entries from previous iterations
  constexpr int kTaskStride = kSM90DynamicTaskStride;  // = 12
  int num_chunks_entries = max_num_batch * num_head_kv;
  int max_num_batch_pad_local = (num_chunks_entries + kTaskStride - 1) / kTaskStride * kTaskStride;
  int num_cta_count_pad_local = (num_total_ctas + kTaskStride - 1) / kTaskStride * kTaskStride;
  int* chunk_clear_ptr = task_map_ptr + kTaskStride * ((num_tile_per_cta + 1) * num_total_ctas + 1);
  int* finish_flag_ptr_global = chunk_clear_ptr + max_num_batch_pad_local;
  int* clear_sentinel_ptr = finish_flag_ptr_global + (num_cta_count_pad_local - 1);

  if (icta == 0) {
    for (int i = idx; i < num_chunks_entries; i += blockDim.x) {
      chunk_clear_ptr[i] = 0;
    }
    __syncthreads();
    if (idx == 0) {
      __threadfence();
      *clear_sentinel_ptr = 1;
    }
  } else {
    if (idx == 0) {
      while (load_global_volatile(clear_sentinel_ptr) == 0) {
      }
      __threadfence();
    }
  }
  __syncthreads();

  if (idx != 0) {
    return;
  }

  // Fast-forward state to this CTA's starting (ihead_kv, ibatch, chunk)
  int ihead_kv = 0;
  int ibatch = 0;
  int num_chunks = 0;
  int start_tiles = 0;
  int num_tile = (num_head_kv > 0 && num_batch > 0) ? num_tiles[0] : 0;

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
      if (num_chunks == kMaxSplitK - 1) {
        add_tiles = num_tile;
      }
      num_chunks++;
      start_tiles += add_tiles;
      num_tile -= add_tiles;
      bucket -= add_tiles;
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

  // Compute per-region pointers in the task_map buffer
  // (kTaskStride / num_chunks_entries already computed above before the
  // CTA-0 clear handshake — reuse those.)
  int max_num_batch_pad = max_num_batch_pad_local;
  int num_cta_count_pad = num_cta_count_pad_local;

  auto* task_map_chunk_ptr =
      task_map_ptr + kTaskStride * ((num_tile_per_cta + 1) * num_total_ctas + 1);
  auto* task_map_sm_finish_ptr = task_map_chunk_ptr + max_num_batch_pad;
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
      num_tile = (ihead_kv < num_head_kv) ? num_tiles[ibatch] : 0;
      continue;
    }

    int add_tiles = std::min(num_tile, bucket);
    int num_seqkv = num_seqkvs[ibatch];

    if (num_chunks == kMaxSplitK - 1) {
      add_tiles = num_tile;
    }

    SM90DynamicTaskInfo task_info;
    task_info.ihead_kv = ihead_kv;
    task_info.ibatch = ibatch;
    task_info.ichunk = num_chunks;
    task_info.iseq_start = start_tiles * kTileN;
    task_info.num_seqkv = std::min(add_tiles * kTileN, num_seqkv - task_info.iseq_start);
    task_info.num_seqkvcache = task_info.num_seqkv;
    task_info.num_tile_kv = (task_info.num_seqkv + kTileN - 1) / kTileN;
    task_info.num_tile_full = task_info.num_seqkvcache / kTileN;
    task_info.is_casual_chunk = 0;
    task_info._pad0 = 0;
    task_info._pad1 = 0;
    task_info._pad2 = 0;

    num_chunks++;
    start_tiles += add_tiles;
    num_tile -= add_tiles;
    bucket -= add_tiles;

    if (num_tile <= 0) {
      task_info.is_casual_chunk = 1;
      int raw_seqkvcache = task_info.num_seqkvcache - num_seq_q;
      task_info.num_seqkvcache = std::max(raw_seqkvcache, 0);
      task_info.num_tile_full = task_info.num_seqkvcache / kTileN;
      task_map_chunk_ptr[ihead_kv * max_num_batch + ibatch] = num_chunks;

      // When raw_seqkvcache < 0, some Q tokens overflow into the previous
      // chunk. Mark prev as causal and reduce its num_seqkvcache accordingly.
      if (raw_seqkvcache < 0) {
        if (itask != 0) {
          SM90DynamicTaskInfo last_task_info =
              load_task<kTaskStride>(task_map_ptr + (itask - 1) * kTaskStride);
          last_task_info.is_casual_chunk = 1;
          last_task_info.num_seqkvcache =
              std::max(last_task_info.num_seqkvcache + raw_seqkvcache, 0);
          last_task_info.num_tile_full = last_task_info.num_seqkvcache / kTileN;
          store_task<kTaskStride>(task_map_ptr + (itask - 1) * kTaskStride, last_task_info);
        } else {
          wait_batch = ibatch;
          wait_batch_num_seqkvcache_overflow = raw_seqkvcache;
        }
      }

      // Advance (ihead_kv, ibatch).
      ibatch++;
      if (ibatch >= num_batch) {
        ibatch = 0;
        ihead_kv++;
      }
      num_tile = (ihead_kv < num_head_kv) ? num_tiles[ibatch] : 0;
      num_chunks = 0;
      start_tiles = 0;
    }

    store_task<kTaskStride>(task_map_ptr + itask * kTaskStride, task_info);
    itask++;
  }

  // Write terminator: ihead_kv < 0 at slot 0 marks end of this bin's list.
  // (The kernel's parse_task checks ibatch first, and we also set ibatch < 0,
  //  so consumers keying off either field work.)
  task_map_ptr[itask * kTaskStride] = -1;      // ihead_kv
  task_map_ptr[itask * kTaskStride + 1] = -1;  // ibatch
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
        SM90DynamicTaskInfo last_task_info =
            load_task<kTaskStride>(task_map_ptr + last_task_id * kTaskStride);
        last_task_info.is_casual_chunk = 1;
        last_task_info.num_seqkvcache =
            std::max(last_task_info.num_seqkvcache + wait_batch_num_seqkvcache_overflow, 0);
        last_task_info.num_tile_full = last_task_info.num_seqkvcache / kTileN;
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

    // Reset the clear-done sentinel (sits at the tail of the pad region) so
    // the next call's CTA-0 vs CTA-N>0 handshake starts from 0 again.
    *clear_sentinel_ptr = 0;
  }
}

}  // namespace kernels

bool assign_attention_decode_task_sm90_dynamic_async(int* task_map_ptr, const int* num_seq_kvcache,
                                                     int num_batch, int num_head_kv, int num_seq_q,
                                                     bool new_kv_included, int min_process_len,
                                                     cudaStream_t stream) {
  int num_sm = get_sm_count();
  int num_total_ctas = num_sm * kCTAPerSM;

  dim3 grid(num_total_ctas);
  dim3 block(128);

  constexpr int kMaxNumBatch = 2048;
  constexpr int kTileN = 64;  // sm90 GEMM tile size

  kernels::assign_attention_decode_task_sm90_dynamic_kernel<kMaxNumBatch, kMaxSplitK, kTileN>
      <<<grid, block, 0, stream>>>(task_map_ptr, num_seq_kvcache, num_batch, num_head_kv, num_seq_q,
                                   new_kv_included, min_process_len, num_total_ctas);

  return true;
}

// CPU reference implementation of the sm90 dynamic assigner at kTileN=64.
std::pair<std::vector<SM90DynamicTaskInfo>, std::vector<int>>
assign_attention_decode_task_sm90_dynamic_sync(const int* num_seq_kvcache, int num_batch,
                                               int num_head_kv, int num_seq_q, bool new_kv_included,
                                               int min_process_len) {
  constexpr int kTileN = 64;

  int num_sm = get_sm_count();
  int num_total_ctas = num_sm * kCTAPerSM;

  std::vector<int> num_seqkvs(num_batch);
  std::vector<int> num_tiles(num_batch);
  int total_tiles_per_head = 0;
  for (int ibatch = 0; ibatch < num_batch; ibatch++) {
    // Skip batches whose KV cache is empty
    if (num_seq_kvcache[ibatch] == 0) {
      num_seqkvs[ibatch] = 0;
      num_tiles[ibatch] = 0;
      continue;
    }
    int num_seqkv =
        (new_kv_included ? num_seq_kvcache[ibatch] : num_seq_kvcache[ibatch] + num_seq_q);
    num_seqkvs[ibatch] = num_seqkv;
    num_tiles[ibatch] = (num_seqkv + kTileN - 1) / kTileN;
    total_tiles_per_head += num_tiles[ibatch];
  }

  int total_tiles_all_heads = total_tiles_per_head * num_head_kv;

  int num_tile_per_cta = std::max((total_tiles_all_heads + num_total_ctas - 1) / num_total_ctas,
                                  min_process_len / kTileN);

  std::vector<SM90DynamicTaskInfo> tasks(num_total_ctas * (num_tile_per_cta + 1));
  std::memset(tasks.data(), 0, tasks.size() * sizeof(SM90DynamicTaskInfo));

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

      if (chunks_in_progress[idx] == kMaxSplitK - 1) {
        add_tiles = num_tile;
      }

      SM90DynamicTaskInfo task_info;
      task_info.ihead_kv = ihead_kv;
      task_info.ibatch = ibatch;
      task_info.ichunk = chunks_in_progress[idx];
      task_info.iseq_start = start_tiles[idx] * kTileN;
      task_info.num_seqkv = std::min(add_tiles * kTileN, num_seqkv - task_info.iseq_start);
      task_info.num_seqkvcache = task_info.num_seqkv;
      task_info.num_tile_kv = (task_info.num_seqkv + kTileN - 1) / kTileN;
      task_info.num_tile_full = task_info.num_seqkvcache / kTileN;
      task_info.is_casual_chunk = 0;
      task_info._pad0 = 0;
      task_info._pad1 = 0;
      task_info._pad2 = 0;

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
        int raw_seqkvcache = cur.num_seqkvcache - num_seq_q;
        cur.num_seqkvcache = std::max(raw_seqkvcache, 0);
        cur.num_tile_full = cur.num_seqkvcache / kTileN;
        num_chunks[ihead_kv * num_batch + ibatch] = chunks_in_progress[idx];

        // When raw_seqkvcache < 0, some Q tokens overflow into the previous
        // chunk. Mark prev as causal and reduce its num_seqkvcache accordingly.
        if (raw_seqkvcache < 0) {
          auto& prev = tasks[last_cta * (num_tile_per_cta + 1) + last_task];
          prev.is_casual_chunk = 1;
          prev.num_seqkvcache = std::max(prev.num_seqkvcache + raw_seqkvcache, 0);
          prev.num_tile_full = prev.num_seqkvcache / kTileN;
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

}  // namespace dynamic
}  // namespace decode
}  // namespace attention
}  // namespace hpc
