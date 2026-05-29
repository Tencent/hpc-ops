// Copyright 2025 hpc-ops authors

#ifndef SRC_ATTENTION_DECODE_SCHED_TASK_INFO_H_
#define SRC_ATTENTION_DECODE_SCHED_TASK_INFO_H_

#include <cuda_runtime_api.h>
#include <stdint.h>

#include <unordered_map>
#include <utility>
#include <vector>

namespace hpc {
namespace attention {
namespace decode {
namespace dynamic {

struct alignas(16) TaskScheduleInfo {
  int ihead_kv;
  int ibatch;
  int ichunk;
  int iseq_start;

  int num_seqkv;
  int num_seqkvcache;
  int num_tile_kv;
  int num_tile_full;

  int is_casual_chunk;
  int pad[3];
};

constexpr int kTaskScheduleInfoStride = sizeof(TaskScheduleInfo) / sizeof(int);

static const std::unordered_map<int, std::vector<int>> kCtaPerSmMap = {{9, {4, 3, 3, 2}},
                                                                       {10, {1, 1, 1, 1}}};

}  // namespace dynamic
}  // namespace decode
}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_DECODE_SCHED_TASK_INFO_H_
