// Copyright 2025 hpc-ops authors

#ifndef SRC_ATTENTION_DECODE_SM90_DYNAMIC_DYNAMIC_SCHED_TASK_INFO_H_
#define SRC_ATTENTION_DECODE_SM90_DYNAMIC_DYNAMIC_SCHED_TASK_INFO_H_

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace hpc {
namespace attention {
namespace decode {
namespace dynamic {

struct alignas(16) SM90DynamicTaskInfo {
  int ihead_kv;
  int ibatch;
  int ichunk;
  int iseq_start;

  int num_seqkv;
  int num_seqkvcache;
  int num_tile_kv;
  int num_tile_full;

  int is_casual_chunk;
  int _pad0;
  int _pad1;
  int _pad2;
};

// Stride (in ints) of one on-disk task entry. All consumers (assigner, attn
// kernel, combine kernel) should reference this constant.
constexpr int kSM90DynamicTaskStride = sizeof(SM90DynamicTaskInfo) / sizeof(int);
static_assert(kSM90DynamicTaskStride == 12, "expected 12 int32 per sm90 dynamic task");

}  // namespace dynamic
}  // namespace decode
}  // namespace attention
}  // namespace hpc

#endif  // SRC_ATTENTION_DECODE_SM90_DYNAMIC_DYNAMIC_SCHED_TASK_INFO_H_
