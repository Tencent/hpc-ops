// Copyright 2025 hpc-ops authors

#ifndef SRC_ATTENTION_DECODE_SCHED_TASK_INFO_H_
#define SRC_ATTENTION_DECODE_SCHED_TASK_INFO_H_

#include <cuda_runtime_api.h>
#include <stdint.h>

#include <utility>
#include <vector>

namespace hpc {
namespace attention {

struct alignas(16) TaskScheduleInfo {
  int ibatch;
  int ichunk;
  int iseq_start;
  int num_seqkv;

  int num_seqkvcache;
  int num_tile_kv;
  int num_tile_full;
  int is_casual_chunk;
};

}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_DECODE_SCHED_TASK_INFO_H_
