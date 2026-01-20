// Copyright (C) 2026 Tencent.

#include "src/utils/utils.h"

#include <cuda_runtime_api.h>

#include <atomic>

namespace hpc {

static std::atomic<int> g_num_sm(-1);

int get_sm_count() {
  int num_sm = g_num_sm.load(std::memory_order_relaxed);
  if (num_sm == -1) {
    // Here we assume all the device share the same properity with the device 0.
    int dev = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    num_sm = prop.multiProcessorCount;

    g_num_sm.store(num_sm, std::memory_order_relaxed);
  }
  return num_sm;
}

}  // namespace hpc
