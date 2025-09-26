// Copyright 2025 hpc-ops authors

#include "src/utils/utils.h"

#include <cuda_runtime_api.h>

namespace hpc {

static int g_num_sm = -1;

int get_sm_count() {
  int num_sm = __atomic_load_n(&g_num_sm, __ATOMIC_RELAXED);
  if (num_sm == -1) {
    // Here we assume all the device share the same properity with the device 0.
    int dev = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    num_sm = prop.multiProcessorCount;
    __atomic_store_n(&g_num_sm, num_sm, __ATOMIC_RELAXED);
  }
  return num_sm;
}

}  // namespace hpc
