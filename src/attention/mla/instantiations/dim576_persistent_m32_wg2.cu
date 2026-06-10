// Copyright 2025 hpc-ops authors

#include "src/attention/mla/smallm_mla_dim576_persistent_launch.cuh"

namespace hpc {
namespace attention {
namespace mla {

template void run_dim576_persistent<32, 2, false>(void*, const void*, const void*, float*, float*,
                                                  int*, const int*, const int*, const int*,
                                                  const float*, int, int, int, int, int, int, int,
                                                  int, int, int, float, cudaStream_t, bool, bool);

}  // namespace mla
}  // namespace attention
}  // namespace hpc
