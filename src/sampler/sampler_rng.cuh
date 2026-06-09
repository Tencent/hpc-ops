// Copyright (C) 2026 Tencent.
//
// Shared RNG for the sampler kernels' self-drawn Gumbel-max noise (cuRAND
// Philox-4x32-10), so fused_sampler.cu and fused_sampler_temperature.cu stay
// byte-for-byte identical on the self-drawn path:
//   seed   — forwarded from the caller unchanged.
//   offset — host atomic counter advanced once per launch (shared single
//            instance, `inline`); makes the same seed give fresh samples.
//   init   — curand_init(seed, sequence, offset). `sequence` is the only
//            per-kernel difference (a function of block/thread geometry).
//   draw   — curand_uniform4 (4 uniforms/call).
//   noise  — gumbel_noise_from_uniform(u) = -log(-log(U)), inner clamped so
//            U→1 cannot poison the score. Matches the PyTorch test reference.

#ifndef SRC_SAMPLER_SAMPLER_RNG_CUH_
#define SRC_SAMPLER_SAMPLER_RNG_CUH_

#include <curand_kernel.h>
#include <stdint.h>

#include <atomic>

#include "src/utils/utils.cuh"

namespace hpc {
namespace sampler {
namespace rng {

// Per-launch advance in the Philox `offset`. 128 covers every per-thread draw
// on both kernels and stays on a power-of-two boundary.
constexpr uint64_t kPerLaunchOffsetIncrement = 128;

// Gumbel(0) noise from a uniform draw U ~ (0, 1].
//   inner = max(-log(U), 1e-20)        // guard U -> 1
//   g     = -log(inner)                // = -log(-log(U))
__device__ __forceinline__ float gumbel_noise_from_uniform(float u) {
  float inner = fmaxf(-logf_ftz(u), 1e-20f);
  return -logf_ftz(inner);
}

// Host-side per-launch RNG offset counter, shared by all sampler kernels
// (`inline` → single instance program-wide).
inline std::atomic<uint64_t>& launch_offset_counter() {
  static std::atomic<uint64_t> counter{0};
  return counter;
}

// Reserve the Philox `offset` base for one launch, then advance the counter.
inline uint64_t next_launch_offset() {
  return launch_offset_counter().fetch_add(kPerLaunchOffsetIncrement, std::memory_order_relaxed);
}

}  // namespace rng
}  // namespace sampler
}  // namespace hpc

#endif  // SRC_SAMPLER_SAMPLER_RNG_CUH_
