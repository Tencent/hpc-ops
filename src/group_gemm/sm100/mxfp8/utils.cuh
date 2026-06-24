// Copyright 2026 hpc-ops authors

#ifndef SRC_GROUP_GEMM_SM100_MXFP8_UTILS_CUH_
#define SRC_GROUP_GEMM_SM100_MXFP8_UTILS_CUH_

#include <cuda.h>

#include <cub/cub.cuh>

#include "cute/atom/copy_atom.hpp"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace group_gemm {

using namespace cute;  // NOLINT

namespace kernels {

// Grid: dim3(num_group, max_row_tiles)
// Block: dim3(kSfLanes)  (32 or 64 threads)
// Each thread produces one 16B output vector per K_tile iteration:
template <int kRowsPerTile, int kSfLanes, bool kUsePDL = false>
__global__ void __launch_bounds__(kSfLanes)
    prepack_mxfp8_scale_kernel(const uint8_t *__restrict__ src, uint8_t *__restrict__ dst,
                               const int *__restrict__ cu_rows_src, int K_sf, int row_stride,
                               int num_group) {
  static_assert(kSfLanes == 32 || kSfLanes == 64, "kSfLanes must be 32 or 64");
  static_assert(kRowsPerTile == kSfLanes * 4, "kRowsPerTile must equal kSfLanes * 4");

  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }

  int igroup = blockIdx.x;
  int rt = blockIdx.y;  // row tile index within this group

  // Early exit: out-of-range group
  if (igroup >= num_group) {
    return;
  }

  int rows_src_start = cu_rows_src[igroup];
  int rows_valid = cu_rows_src[igroup + 1] - rows_src_start;
  int actual_row_tiles = (rows_valid + row_stride - 1) / row_stride;
  int K_tiles = (K_sf + 3) / 4;

  // Early exit: this CTA's row_tile is out of range
  if (rt >= actual_row_tiles) {
    return;
  }

  // Compute dst offset from cu_rows_src
  int dst_tile_offset = 0;
  for (int i = 0; i < igroup; i++) {
    int sl = cu_rows_src[i + 1] - cu_rows_src[i];
    dst_tile_offset += (sl + row_stride - 1) / row_stride;
  }

  int tid = threadIdx.x;  // tid ∈ [0, kSfLanes)

  // Map thread to 4 source rows via TMA packed layout mapping
  int row_base_in_tile = (tid / 32) * 128 + (tid % 32);

  // Pre-compute row pointers and validity for the 4 rows this thread handles
  const uint8_t *src_base = src + rows_src_start * K_sf;
  const uint8_t *row_ptrs[4];
  bool row_valids[4];
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    int row = rt * row_stride + row_base_in_tile + q * 32;
    if (row < rows_valid) {
      row_ptrs[q] = src_base + row * K_sf;
      row_valids[q] = true;
    } else {
      row_ptrs[q] = nullptr;
      row_valids[q] = false;
    }
  }

  // Precompute dst base for this row_tile
  uint8_t *dst_base = dst + (dst_tile_offset + rt) * kSfLanes * 16 * K_tiles;

  bool use_byte_path = (K_sf & 3);

  for (int kt = 0; kt < K_tiles; ++kt) {
    vec_t<uint8_t, 16> out;
    uint32_t *vu32 = reinterpret_cast<uint32_t *>(&out);
    int k_base = kt * 4;

#pragma unroll
    for (int q = 0; q < 4; ++q) {
      uint32_t v = 0;
      if (row_valids[q]) {
        if (use_byte_path) {
          uint8_t b0 = (k_base + 0 < K_sf) ? row_ptrs[q][k_base + 0] : uint8_t(0);
          uint8_t b1 = (k_base + 1 < K_sf) ? row_ptrs[q][k_base + 1] : uint8_t(0);
          uint8_t b2 = (k_base + 2 < K_sf) ? row_ptrs[q][k_base + 2] : uint8_t(0);
          uint8_t b3 = (k_base + 3 < K_sf) ? row_ptrs[q][k_base + 3] : uint8_t(0);
          v = static_cast<uint32_t>(b0) | (static_cast<uint32_t>(b1) << 8) |
              (static_cast<uint32_t>(b2) << 16) | (static_cast<uint32_t>(b3) << 24);
        } else {
          auto ld = load<uint8_t, 4>(row_ptrs[q] + k_base);
          v = *reinterpret_cast<const uint32_t *>(&ld);
        }
      }
      vu32[q] = v;
    }

    store<uint8_t, 16>(dst_base + (kt * kSfLanes + tid) * 16, out);
  }

  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

// Same as above but for the case where every group has the same number of rows
// (e.g. SFW weight scale prepack). Avoids the need for a cu_rows_src array.
// Grid: dim3(num_group, row_tiles)
// Block: dim3(kSfLanes)  (32 or 64 threads)
template <int kRowsPerTile, int kSfLanes, bool kUsePDL = false>
__global__ void __launch_bounds__(kSfLanes)
    prepack_mxfp8_scale_kernel(const uint8_t *__restrict__ src, uint8_t *__restrict__ dst,
                               int n_per_group, int K_sf) {
  static_assert(kSfLanes == 32 || kSfLanes == 64, "kSfLanes must be 32 or 64");
  static_assert(kRowsPerTile == kSfLanes * 4, "kRowsPerTile must equal kSfLanes * 4");

  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }

  int igroup = blockIdx.x;
  int rt = blockIdx.y;  // row tile index
  int row_tiles = n_per_group / kRowsPerTile;

  // Early exit: this CTA's row_tile is out of range
  if (rt >= row_tiles) {
    return;
  }

  int K_tiles = (K_sf + 3) / 4;
  int tid = threadIdx.x;  // tid ∈ [0, kSfLanes)

  // Regular layout: each group starts at igroup * n_per_group
  int rows_src_start = igroup * n_per_group;
  // dst tile offset: all groups have the same row_tiles, so offset is linear
  int dst_tile_offset = igroup * row_tiles;

  // Map thread to 4 source rows via TMA packed layout mapping
  int row_base_in_tile = (tid / 32) * 128 + (tid % 32);

  // Pre-compute row pointers (all rows are valid since n_per_group is tile-aligned)
  const uint8_t *src_base = src + rows_src_start * K_sf;
  const uint8_t *row_ptrs[4];
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    int row = rt * kRowsPerTile + row_base_in_tile + q * 32;
    row_ptrs[q] = src_base + row * K_sf;
  }

  // Precompute dst base for this row_tile
  uint8_t *dst_base = dst + (dst_tile_offset + rt) * kSfLanes * 16 * K_tiles;

  bool use_byte_path = (K_sf & 3);

  for (int kt = 0; kt < K_tiles; ++kt) {
    vec_t<uint8_t, 16> out;
    uint32_t *vu32 = reinterpret_cast<uint32_t *>(&out);
    int k_base = kt * 4;

#pragma unroll
    for (int q = 0; q < 4; ++q) {
      uint32_t v;
      if (use_byte_path) {
        uint8_t b0 = (k_base + 0 < K_sf) ? row_ptrs[q][k_base + 0] : uint8_t(0);
        uint8_t b1 = (k_base + 1 < K_sf) ? row_ptrs[q][k_base + 1] : uint8_t(0);
        uint8_t b2 = (k_base + 2 < K_sf) ? row_ptrs[q][k_base + 2] : uint8_t(0);
        uint8_t b3 = (k_base + 3 < K_sf) ? row_ptrs[q][k_base + 3] : uint8_t(0);
        v = static_cast<uint32_t>(b0) | (static_cast<uint32_t>(b1) << 8) |
            (static_cast<uint32_t>(b2) << 16) | (static_cast<uint32_t>(b3) << 24);
      } else {
        auto ld = load<uint8_t, 4>(row_ptrs[q] + k_base);
        v = *reinterpret_cast<const uint32_t *>(&ld);
      }
      vu32[q] = v;
    }

    store<uint8_t, 16>(dst_base + (kt * kSfLanes + tid) * 16, out);
  }

  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

template <typename Tin, typename Tout, typename TmaA, typename TmaY, int kTileM,
          int kGroupPerThread, int kThreadPerBlock>
__global__ void update_grouped_tma_mxfp8(const vec_t<cute::TmaDescriptor, 2> td_ay,
                                         cute::TmaDescriptor *tma_ay, const Tin *x_ptr,
                                         const Tout *y_ptr, const int *seqlens_ptr,
                                         const int *cu_seqlens_ptr, int *tiles_ptr,
                                         int *cu_tiles_ptr, int num_group, int m, int n, int k) {
  using namespace cute;  // NOLINT

  int idx = threadIdx.x;
  int igroup = blockIdx.x;

  if (igroup == num_group) {
    // last block: tiles[i] = ceil(seqlens[i] / kTileM); exclusive_sum -> cu_tiles
    int tiles[kGroupPerThread];
#pragma unroll
    for (int i = 0; i < kGroupPerThread; i++) {
      int g = idx * kGroupPerThread + i;
      if (g < num_group) {
        tiles[i] = (seqlens_ptr[g] + kTileM - 1) / kTileM;
        tiles_ptr[g] = tiles[i];
      } else {
        tiles[i] = 0;
      }
    }

    using BlockScan = cub::BlockScan<int, kThreadPerBlock>;
    __shared__ typename BlockScan::TempStorage temp_storage;
    int block_aggregate;
    BlockScan(temp_storage).ExclusiveSum(tiles, tiles, block_aggregate);

#pragma unroll
    for (int i = 0; i < kGroupPerThread; i++) {
      int g = idx * kGroupPerThread + i;
      if (g < num_group) {
        cu_tiles_ptr[g] = tiles[i];
      }
    }
    if (idx == 0) {
      cu_tiles_ptr[num_group] = block_aggregate;
    }
  } else {
    __shared__ cute::TmaDescriptor smem_tma_desc[2];

    int num_seq = seqlens_ptr[igroup];
    uint64_t cu_seqlen = cu_seqlens_ptr[igroup];
    auto *x_ibatch_ptr = x_ptr + cu_seqlen * k;
    auto *y_ibatch_ptr = y_ptr + cu_seqlen * n;

    if (idx < 2) {
      smem_tma_desc[idx] = td_ay[idx];
    }
    __syncwarp();

    // B: shape (num_seq, k), stride (k, 1)
    if (idx == 0) {
      auto gA = make_tensor(make_gmem_ptr(x_ibatch_ptr), make_shape(num_seq, k),
                            make_stride(k, Int<1>{}));
      update_tma_gtensor<TmaA, decltype(gA), true, true>(smem_tma_desc[idx], gA);
    }

    // Y: shape (n, num_seq), stride (1, n) — physical col-major
    if (idx == 1) {
      auto gY = make_tensor(make_gmem_ptr(y_ibatch_ptr), make_shape(n, num_seq),
                            make_stride(Int<1>{}, n));
      update_tma_gtensor<TmaY, decltype(gY), true, true>(smem_tma_desc[idx], gY);
    }

#pragma unroll
    for (int i = 0; i < 2; i++) {
      __syncwarp();
      if (cute::elect_one_sync()) {
        cute::tma_desc_commit_group();
        cute::tma_desc_wait_group();
      }
      tma_descriptor_cp_fence_release(tma_ay + igroup * 2 + i, smem_tma_desc[i]);
    }
  }
}

}  // namespace kernels
}  // namespace group_gemm
}  // namespace hpc

#endif  // SRC_GROUP_GEMM_SM100_MXFP8_UTILS_CUH_
