// Copyright (C) 2026 Tencent.

#pragma once

#include <cstdint>

namespace hpc::fuse_moe::sm120 {

// Position in a ring of pipeline stages. Tiles run exactly ceil(K / TileK) K tiles,
// so a tile can start at any stage and the position carries over between tiles.
struct PipeState {
  int stage;
  uint32_t phase;

  template <int Stages>
  __device__ __forceinline__ void advance() {
    if (++stage == Stages) {
      stage = 0;
      phase ^= 1;
    }
  }
};

}  // namespace hpc::fuse_moe::sm120
