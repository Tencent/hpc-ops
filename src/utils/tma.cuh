// Copyright (C) 2026 Tencent.

#ifndef SRC_UTILS_TMA_CUH_
#define SRC_UTILS_TMA_CUH_

#include "cute/tensor.hpp"

namespace hpc {

__device__ __forceinline__ void tma_descriptor_replace_shapes_in_shared_mem(
    cute::TmaDescriptor &smem_desc, cute::array<uint32_t, 5> const &prob_shape) {
#if (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 3))
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  uint32_t smem_int_desc = cute::cast_smem_ptr_to_uint(&smem_desc);
  uint64_t const smem_int64_desc = 0;
  asm volatile("cvt.u64.u32 %0, %1;" ::"l"(smem_int64_desc), "r"(smem_int_desc));
  asm volatile(
      "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 0, %1;" ::"l"(smem_int64_desc),
      "r"(prob_shape[0]));
  asm volatile(
      "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 1, %1;" ::"l"(smem_int64_desc),
      "r"(prob_shape[1]));
  asm volatile(
      "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 2, %1;" ::"l"(smem_int64_desc),
      "r"(prob_shape[2]));
  asm volatile(
      "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 3, %1;" ::"l"(smem_int64_desc),
      "r"(prob_shape[3]));
  asm volatile(
      "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 4, %1;" ::"l"(smem_int64_desc),
      "r"(prob_shape[4]));
#endif
#endif
}

template <typename Tma, typename GTensor, bool kUpdateShape = true>
__device__ __forceinline__ void update_tma_gtensor(cute::TmaDescriptor &smem_tma_desc,
                                                   const GTensor &gtensor) {
  cute::array<uint32_t, 5> shape{1, 1, 1, 1, 1};
  cute::array<uint64_t, 5> stride{0, 0, 0, 0, 0};

  cute::detail::fill_tma_gmem_shape_stride(Tma{}, gtensor, shape, stride);

  const void *gmem_ptr = gtensor.data().get();
  cute::tma_descriptor_replace_addr_in_shared_mem(smem_tma_desc, gmem_ptr);

  if constexpr (kUpdateShape) {
    tma_descriptor_replace_shapes_in_shared_mem(smem_tma_desc, shape);
  } else {
    // update stride to byte representation
    using T = typename GTensor::value_type;
    for (auto &s : stride) {
      s = (s * cute::sizeof_bits_v<T>) / 8;
    }
    tma_descriptor_replace_dims_strides_in_shared_mem(smem_tma_desc, shape, stride);
  }
}

__device__ __forceinline__ void cp_async_g2s(void *smem_ptr, const void *gmem_ptr, int bytes,
                                             const uint64_t *bar_ptr) {
  int smem_int = cute::cast_smem_ptr_to_uint(smem_ptr);
  int bar_int = cute::cast_smem_ptr_to_uint(bar_ptr);
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(
          smem_int),
      "l"(gmem_ptr), "r"(bytes), "r"(bar_int));
}

}  // namespace hpc

#endif  // SRC_UTILS_TMA_CUH_
