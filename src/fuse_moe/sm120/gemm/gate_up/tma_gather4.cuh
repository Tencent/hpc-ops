// Copyright (C) 2026 Tencent.

#pragma once

// clang-format off
#include <cstdint>

#include <cute/algorithm/tuple_algorithms.hpp>
#include <cute/arch/copy_sm100.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/copy_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/util.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cute/config.hpp>
#include <cute/layout.hpp>
#include <cute/pointer_flagged.hpp>
#include <cute/swizzle.hpp>
#include <cute/tensor.hpp>

#include "src/fuse_moe/sm120/gemm/common/pipe_state.cuh"
// clang-format on

namespace hpc::fuse_moe::sm120::gemm::gate_up::tma_gather4 {

using namespace cute;  // NOLINT

// One gather4 box is a 128-byte K-major line (SW128 atom row) of E4M3 A.
constexpr int kGather4TileK = 128;
constexpr int kGather4Rows = 4;
constexpr int kGather4RowBytes = 128;
constexpr int kGather4Warps = 4;
constexpr int kGather4OobRow = -1;

constexpr uint32_t kGather4BoxBytes = static_cast<uint32_t>(kGather4Rows * kGather4RowBytes);

using ElementA = float_e4m3_t;

// One-row K-major box with the same SW128 swizzle as GMMA Layout_K_SW128_Atom.
// A 128-byte line is a full swizzle period along K, so four stacked gather4
// rows land at the same smem offsets a plain TMA load of the A tile would use.
using Gather4SmemLayout =
    ComposedLayout<Swizzle<3, 4, 3>, smem_ptr_flag,
                   Layout<Shape<_1, Int<kGather4TileK>>, Stride<Int<kGather4TileK>, _1>>>;

using Gather4StrideA = Stride<int64_t, Int<1>>;

using TmaGather4A = decltype(make_tma_copy(
    SM90_TMA_LOAD{},
    make_tensor(make_gmem_ptr(static_cast<ElementA const *>(nullptr)),
                repeat_like(Gather4StrideA{}, static_cast<int64_t>(0)), Gather4StrideA{}),
    Gather4SmemLayout{}, make_shape(_1{}, Int<kGather4TileK>{}), _1{}));

// Token-major E4M3 [num_tokens, K], K-major, 16-byte aligned. Box is 1 x 128.
// This is a plain one-row TMA atom (SM90_TMA_LOAD); gather4 is only in the
// issue PTX. num_tokens >= 1, K % 16 == 0.
__host__ inline TmaGather4A make_tma_gather4_a_copy(ElementA const *ptr, int num_tokens, int k) {
  auto tensor_a = make_tensor(
      make_gmem_ptr(ptr),
      make_layout(make_shape(num_tokens, k), make_stride(static_cast<int64_t>(k), Int<1>{})));
  return make_tma_copy(SM90_TMA_LOAD{}, tensor_a, Gather4SmemLayout{},
                       make_shape(_1{}, Int<kGather4TileK>{}), _1{});
}

// PTX coordinates are {column, row0, row1, row2, row3}: TMA dim0 is contiguous
// K, dim1 is the gathered token-row axis. A row of -1 is OOB; TMA zeros that
// box row and still completes its bytes on the barrier.
template <bool kHinted>
__device__ __forceinline__ void tma_gather4(cute::TmaDescriptor const *desc, void *smem_dst,
                                            uint64_t *mbar, int32_t column, int32_t row0,
                                            int32_t row1, int32_t row2, int32_t row3,
                                            uint64_t cache_hint) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
  uint32_t smem_int_ptr = cute::cast_smem_ptr_to_uint(smem_dst);
  uint32_t smem_int_mbar = cute::cast_smem_ptr_to_uint(mbar);
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc);
  if constexpr (!kHinted) {
    (void)cache_hint;
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
        ".mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6, %7}], [%2];"
        :
        : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar), "r"(column), "r"(row0),
          "r"(row1), "r"(row2), "r"(row3)
        : "memory");
  } else {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
        ".mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4, %5, %6, %7}], [%2], %8;"
        :
        : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar), "r"(column), "r"(row0),
          "r"(row1), "r"(row2), "r"(row3), "l"(cache_hint)
        : "memory");
  }
#elif defined(__CUDA_ARCH__)
  CUTE_INVALID_CONTROL_PATH("tma_gather4 requires SM120+");
#else
  (void)desc;
  (void)smem_dst;
  (void)mbar;
  (void)column;
  (void)row0;
  (void)row1;
  (void)row2;
  (void)row3;
  (void)cache_hint;
#endif
}

// Count byte_count toward the barrier without moving data. Used for a gather4
// box whose four rows are all padding.
__device__ __forceinline__ void mbarrier_complete_tx(uint64_t *mbar, uint32_t byte_count) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
  uint32_t smem_int_mbar = cute::cast_smem_ptr_to_uint(mbar);
  asm volatile("mbarrier.complete_tx.shared::cta.b64 [%0], %1;"
               :
               : "r"(smem_int_mbar), "r"(byte_count)
               : "memory");
#elif defined(__CUDA_ARCH__)
  CUTE_INVALID_CONTROL_PATH("mbarrier_complete_tx requires SM120+");
#else
  (void)mbar;
  (void)byte_count;
#endif
}

__device__ __forceinline__ void prefetch_tma_gather4_descriptor(cute::TmaDescriptor const *desc) {
#if defined(__CUDA_ARCH__)
  cute::prefetch_tma_descriptor(desc);
#else
  (void)desc;
#endif
}

template <bool kHinted>
__device__ __forceinline__ void tma_gather4_issue_or_skip(cute::TmaDescriptor const *desc,
                                                          void *smem_dst, uint64_t *mbar,
                                                          int32_t column, int32_t const *rows,
                                                          bool live, bool issuing,
                                                          uint64_t cache_hint) {
  if (live) {
    tma_gather4<kHinted>(desc, smem_dst, mbar, column, rows[0], rows[1], rows[2], rows[3],
                         cache_hint);
  } else if (issuing) {
    mbarrier_complete_tx(mbar, kGather4BoxBytes);
  }
}

// Lane state for one gather4 box of a tile:
//   rows_per_warp = tile_m / kGather4Warps
//   row0 = gather_warp * rows_per_warp + lane * kGather4Rows
//   issuing = lane < rows_per_warp / kGather4Rows
struct Gather4Lane {
  int32_t tokens[kGather4Rows];
  int32_t row0;
  bool live;
  bool issuing;
};

// Token index is route // num_topk. Padding (row_to_route < 0) stays -1 so TMA
// OOB-fills that box row. Never divide a -1 route: integer division would
// yield 0 and silently load token 0.
__device__ __forceinline__ int32_t token_from_padded_row(int32_t const *row_to_route,
                                                         int32_t global_row, int num_topk) {
  int32_t route = row_to_route[global_row];
  return route >= 0 ? route / num_topk : kGather4OobRow;
}

// Epilogue / scheduler form: live rows in this tile. Matches
// quant_epilogue's m_boundary math in kernel.cuh.
__device__ __forceinline__ int32_t gather4_valid_rows(int32_t m_offset, int32_t m_block,
                                                      int32_t m_boundary, int tile_m) {
  int32_t valid = m_boundary - m_offset - m_block * tile_m;
  if (valid < 0) {
    return 0;
  }
  if (valid > tile_m) {
    return tile_m;
  }
  return valid;
}

// A box is live when its first row is below valid_rows; rows at or past it
// stay -1 and are zero filled.
__device__ __forceinline__ Gather4Lane map_gather4_lane(int32_t const *row_to_route, int num_topk,
                                                        int32_t m_offset, int32_t m_block,
                                                        int tile_m, int gather_warp, int lane,
                                                        int32_t valid_rows) {
  Gather4Lane mapped{};
  int rows_per_warp = tile_m / kGather4Warps;
  mapped.row0 = gather_warp * rows_per_warp + lane * kGather4Rows;
  mapped.issuing = lane < (rows_per_warp / kGather4Rows);
  mapped.live = mapped.issuing && mapped.row0 < valid_rows;
#pragma unroll
  for (int i = 0; i < kGather4Rows; ++i) {
    mapped.tokens[i] = kGather4OobRow;
  }
  if (mapped.issuing) {
    int32_t base = m_offset + m_block * tile_m + mapped.row0;
#pragma unroll
    for (int i = 0; i < kGather4Rows; ++i) {
      if (mapped.row0 + i < valid_rows) {
        mapped.tokens[i] = token_from_padded_row(row_to_route, base + i, num_topk);
      }
    }
  }
  return mapped;
}

__device__ __forceinline__ int smem_gather4_row_offset(int stage, int tile_m, int row0) {
  return (stage * tile_m + row0) * kGather4TileK;
}

// Pipelined gather-warp A load, lockstep with the AB DMA warp:
//   * wait the same ab_empty phase the DMA warp waits
//   * do not arrive on ab_full; only complete_tx A bytes (via gather4 or skip)
//   * DMA still arrive_and_expect_tx(A+B[+SF]) including TmaTransactionBytesA
//   * skip gather4 boxes with no live rows; complete_tx their 512 bytes
//
// Called by all threads of the four gather warps. kTileK is 128. kTileM is 32
// or 64. store_empty_mbar is the unioned-epilogue empty barrier, or nullptr.
template <bool kHinted, int kTileM, int kABStages, class FullBarrier, class EmptyBarrier>
__device__ void gather_warp_load_a(cute::TmaDescriptor const *desc, cute::float_e4m3_t *smem_a,
                                   FullBarrier *ab_full_mbar, EmptyBarrier *ab_empty_mbar,
                                   EmptyBarrier *store_empty_mbar, PipeState &ab_pipe,
                                   uint32_t &store_phase, int32_t const *row_to_route, int num_topk,
                                   int32_t m_offset, int32_t m_block, int32_t k_tile_count,
                                   int gather_warp, uint64_t cache_hint, int32_t valid_rows) {
  static_assert(kTileM % (kGather4Warps * kGather4Rows) == 0,
                "tile_m must be a multiple of 16 gather4 rows");
  static_assert(kABStages >= 1, "AB pipeline needs at least one stage");

  int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
  Gather4Lane mapped = map_gather4_lane(row_to_route, num_topk, m_offset, m_block, kTileM,
                                        gather_warp, lane, valid_rows);

  if (store_empty_mbar != nullptr) {
    store_empty_mbar[0].wait(store_phase);
    store_phase ^= 1u;
  }

  for (int32_t k_tile_idx = 0; k_tile_idx < k_tile_count; ++k_tile_idx) {
    ab_empty_mbar[ab_pipe.stage].wait(ab_pipe.phase);
    uint64_t *mbar = cute::recast_ptr<uint64_t>(&ab_full_mbar[ab_pipe.stage]);
    cute::float_e4m3_t *dst = smem_a + smem_gather4_row_offset(ab_pipe.stage, kTileM, mapped.row0);
    int32_t column = k_tile_idx * kGather4TileK;
    tma_gather4_issue_or_skip<kHinted>(desc, dst, mbar, column, mapped.tokens, mapped.live,
                                       mapped.issuing, cache_hint);
    ab_pipe.advance<kABStages>();
  }
}

}  // namespace hpc::fuse_moe::sm120::gemm::gate_up::tma_gather4
