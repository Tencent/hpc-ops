// Copyright (C) 2026 Tencent.

#include <c10/util/Exception.h>
#include <cuda.h>

#include <cassert>

#include "cute/tensor.hpp"
#include "src/group_gemm/cp_async/common.cuh"
#include "src/group_gemm/cp_async/config.h"
#include "src/group_gemm/cp_async/group_gemm.h"
#include "src/utils/utils.cuh"
#include "src/utils/utils.h"

namespace hpc {
namespace group_gemm_cp_async {
namespace kernels {

template <typename Config, bool kFullTile, bool kPeerInput, typename TsACopy>
__device__ __forceinline__ void scatter_load_A_tile(
    TsACopy &tsA_copy, const typename Config::Tin *__restrict__ A_pool,
    const void *const *__restrict__ input_ptrs, const uint64_t *__restrict__ shm_input_row_bases,
    const int *__restrict__ shm_row_indices, int num_valid, int k_col_base,
    int64_t input_row_stride, int ismem) {
  // Peer-indexed input changes only the global A-row address. The native
  // cp.async stages, WGMMA work, and epilogue stay on the same CTA/warp-group;
  // resident warps/CTAs let hardware scheduling overlap part of a peer stall.
  using namespace cute;  // NOLINT
  using Tin = typename Config::Tin;
  constexpr int kTileM = Config::kTileM;
  constexpr int kTileK = Config::kTileK;
  constexpr int kElemsPerAtom = 16;  // 16 fp8 = 128 bits
  constexpr int kThreadsPerRow = kTileK / kElemsPerAtom;
  constexpr int kRowsPerIter = 128 / kThreadsPerRow;
  constexpr int kNumIters = (kTileM + kRowsPerIter - 1) / kRowsPerIter;
  constexpr bool kHasVirtualOOB = (kNumIters * kRowsPerIter > kTileM);

  const int row_in_group = threadIdx.x / kThreadsPerRow;
  const int k_thread = threadIdx.x % kThreadsPerRow;
  const int k_col = k_col_base + k_thread * kElemsPerAtom;

  auto do_one = [&](int row_iter, int local_row) {
    void *smem_ptr = (void *)&tsA_copy(cute::Int<0>{}, row_iter, cute::Int<0>{}, ismem);
    int src_size;
    if constexpr (kFullTile) {
      src_size = 16;
    } else {
      const bool valid = local_row < num_valid;
      src_size = valid ? 16 : 0;
    }
    const Tin *input;
    int shard_row;
    if constexpr (kPeerInput) {
      if constexpr (kFullTile) {
        input = reinterpret_cast<const Tin *>(shm_input_row_bases[local_row]);
      } else {
        input = local_row < num_valid
                    ? reinterpret_cast<const Tin *>(shm_input_row_bases[local_row])
                    : static_cast<const Tin *>(input_ptrs[0]);
      }
      shard_row = 0;
    } else {
      input = A_pool;
      if constexpr (kFullTile) {
        shard_row = shm_row_indices[local_row];
      } else {
        shard_row = local_row < num_valid ? shm_row_indices[local_row] : 0;
      }
    }
    const void *gmem_ptr =
        static_cast<const void *>(&input[uint64_t(shard_row) * input_row_stride + k_col]);
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(
                     static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr))),
                 "l"(gmem_ptr), "n"(16), "r"(src_size));
  };

#pragma unroll
  for (int row_iter = 0; row_iter < kNumIters; ++row_iter) {
    const int local_row = row_iter * kRowsPerIter + row_in_group;
    if constexpr (kHasVirtualOOB) {
      if (local_row < kTileM) {
        do_one(row_iter, local_row);
      }
    } else {
      do_one(row_iter, local_row);
    }
  }
}

// Parameterised launch bounds so the host dispatcher can pick the
// __launch_bounds__ kMinBlocks per (kTileM, kTileK, kStage) via
// ScatterLaunchPolicy below.
template <typename Config, bool kUseTaskMap, bool kUsePDL, bool kBlockwiseScale, bool kPeerInput,
          int kMinBlocks = 5>
__global__ void __launch_bounds__(128, kMinBlocks) group_gemm_fp8_scatter_kernel(
    const void *Cptr, const void *Aptr, const void *const *input_ptrs,
    const void *const *input_scale_ptrs, const void *Bptr, const float *y_scale_ptr,
    const float *weight_scale_ptr,
    const int *row_indices_ptr,  // scatter row indices, length = total_tokens
    int64_t input_row_stride, int64_t input_scale_row_stride, int weight_scale_stride,
    const int *seqlens_ptr, const int *cu_seqlens_ptr, const int *tiles_ptr,
    const int *cu_tiles_ptr, const int4 *task_map_ptr, int task_map_len, int m, int n, int k,
    int num_group, int source_row_limit, cutlass::FastDivmod flat_divider,
    cutlass::FastDivmod source_row_divider) {
  using namespace cute;  // NOLINT

  using Tin = typename Config::Tin;
  using Tout = typename Config::Tout;
  using SmemLayoutA = typename Config::SmemLayoutA;
  using SmemLayoutB = typename Config::SmemLayoutB;
  using SmemLayoutCT = typename Config::SmemLayoutCT;
  using G2SCopyA = typename Config::G2SCopyA;
  using G2SCopyB = typename Config::G2SCopyB;
  using S2GCopyC = typename Config::S2GCopyC;
  using TiledMMA = typename Config::TiledMMA;

  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;
  constexpr int kStage = Config::kStage;
  constexpr int kScaleBlock = 128;

  extern __shared__ Tin shm_data[];
  Tin *Ashm = shm_data;
  Tin *Bshm = Ashm + cosize(SmemLayoutA{});
  int *shm_tiles = reinterpret_cast<int *>(Bshm + cosize(SmemLayoutB{}));
  int *shm_row_indices = shm_tiles + num_group;
  uintptr_t peer_scratch_addr = reinterpret_cast<uintptr_t>(shm_row_indices + kTileM);
  peer_scratch_addr = (peer_scratch_addr + alignof(uint64_t) - 1) & ~(alignof(uint64_t) - 1);
  uint64_t *shm_input_row_bases = reinterpret_cast<uint64_t *>(peer_scratch_addr);
  uint64_t *shm_input_scale_row_bases = shm_input_row_bases + (kPeerInput ? kTileM : 0);
  void *scale_scratch_start = kPeerInput ? static_cast<void *>(shm_input_scale_row_bases + kTileM)
                                         : static_cast<void *>(shm_row_indices + kTileM);
  float *shm_input_scales = reinterpret_cast<float *>(scale_scratch_start);
  float *shm_weight_scales = shm_input_scales + kTileM;
  Tout *Cshm = reinterpret_cast<Tout *>(shm_data);

  int idx = threadIdx.x;
  int iblock = blockIdx.x;

  // PDL acquire: wait until the upstream kernel (count/build) has finished
  // writing the inputs we are about to read (x_pool row_indices, tiles,
  // cu_seqlens, task_map, scale).
  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }

  if constexpr (!kUseTaskMap) {
    for (int i = idx; i < num_group; i += blockDim.x) {
      shm_tiles[i] = tiles_ptr[i];
    }
    __syncthreads();
  }

  int igroup = 0;
  int itile_m, itile_n;
  int sum_tile_m = 0;
  int ntile = k / kTileK;
  while (true) {
    if constexpr (kUseTaskMap) {
      if (iblock >= task_map_len) {
        break;
      }
      int4 task = task_map_ptr[iblock];
      igroup = task.x;
      if (igroup < 0) {
        break;
      }
      itile_m = task.y;
      itile_n = task.z;
    } else {
      get_next_tile_horizon(shm_tiles, iblock, num_group, igroup, itile_m, itile_n, sum_tile_m,
                            flat_divider);
      if (igroup < 0) {
        break;
      }
    }

    int start_token = cu_seqlens_ptr[igroup];  // start offset of this group in row_indices
    m = seqlens_ptr[igroup];                   // token count of this group
    iblock += gridDim.x;

    // Global Tensor
    Tensor B = make_tensor(make_gmem_ptr((Tin *)Bptr + uint64_t(igroup) * n * k), make_shape(n, k),
                           make_stride(k, Int<1>{}));
    Tensor C = make_tensor(make_gmem_ptr((Tout *)Cptr + uint64_t(start_token) * n),
                           make_shape(n, m), make_stride(Int<1>{}, n));
    Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(itile_n, _));
    Tensor gCT =
        local_tile(C, make_tile(Int<kTileN>{}, Int<kTileM>{}), make_coord(itile_n, itile_m));

    // Smem Tensor
    auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});  // (kTileM, kTileK, kStage)
    auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});  // (kTileN, kTileK, kStage)
    auto sCT = make_tensor(make_smem_ptr((Tout *)Cshm), SmemLayoutCT{});  // (kTileN, kTileM)

    // G2S for B
    G2SCopyB g2s_tiled_copy_b;
    auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
    auto tgB_copy = g2s_thr_copy_b.partition_S(gB);
    auto tsB_copy = g2s_thr_copy_b.partition_D(sB);

    // G2S partition for A: we use this only to compute the per-thread
    // swizzle-aware smem destination address; the actual cp.async is
    // issued by hand inside scatter_load_A_tile with a custom gmem source.
    G2SCopyA g2s_tiled_copy_a;
    auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
    auto tsA_copy = g2s_thr_copy_a.partition_D(sA);

    // S2G
    S2GCopyC s2g_tiled_copy_c;
    auto s2g_thr_copy_c = s2g_tiled_copy_c.get_slice(idx);
    auto tCs2g = s2g_thr_copy_c.partition_S(sCT);
    auto tCg2s = s2g_thr_copy_c.partition_D(gCT);

    // swapAB TiledMMA: swap A↔B operands
    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(idx);
    auto tBr = thr_mma.make_fragment_A(thr_mma.partition_A(sB));  // (MMA, MMA_N, MMA_K, kStage)
    auto tAr = thr_mma.make_fragment_B(thr_mma.partition_B(sA));  // (MMA, MMA_M, MMA_K, kStage)
    auto tCr = thr_mma.partition_fragment_C(gCT);
    auto tCr_mn = retile_fragment(tCr);
    constexpr int kFragmentN = size<0>(tCr_mn);
    constexpr int kFragmentM = size<1>(tCr_mn);
    auto tI_mn = retile_fragment(thr_mma.partition_C(make_identity_tensor(gCT.shape())));

    // Identity tensor for M boundary predicate.  Same rationale as in the
    // non-scatter kernel: `row < kTileM` masks ThrLayout rows past kTileM
    // (kTileM=8 + ThrLayout_M=16), and `itile_m*kTileM + row < m` masks
    // the global tile-M tail.  UniversalCopy atom has no `.with(bool)` so
    // `copy_if` truly skips OOB threads — no extra idx gate needed.
    auto tIC =
        s2g_thr_copy_c.partition_S(make_identity_tensor(make_shape(Int<kTileN>{}, Int<kTileM>{})));
    auto pred_c = make_tensor<bool>(shape(tIC));
#pragma unroll
    for (int i = 0; i < size(tIC); ++i) {
      pred_c(i) = get<1>(tIC(i)) < kTileM && itile_m * kTileM + get<1>(tIC(i)) < m;
    }

    // Start offset of this tile in row_indices.
    const int token_base_for_tile = start_token + itile_m * kTileM;
    const int remaining = m - itile_m * kTileM;
    const int num_valid = remaining < kTileM ? remaining : kTileM;
    const bool is_full_tile = (num_valid == kTileM);

    // Load this tile's row_indices from global to smem, reused across K-dim iterations
    for (int i = idx; i < num_valid; i += blockDim.x) {
      const int source_row = row_indices_ptr[token_base_for_tile + i];
      shm_row_indices[i] = source_row;
      if constexpr (kPeerInput) {
        const bool source_row_valid = source_row >= 0 && source_row < source_row_limit;
        const int safe_source_row = source_row_valid ? source_row : 0;
        int owner_rank;
        int shard_row;
        source_row_divider(owner_rank, shard_row, safe_source_row);
        const auto *input = static_cast<const Tin *>(input_ptrs[owner_rank]);
        const auto *input_scale = static_cast<const float *>(input_scale_ptrs[owner_rank]);
        shm_input_row_bases[i] =
            reinterpret_cast<uint64_t>(&input[static_cast<uint64_t>(shard_row) * input_row_stride]);
        shm_input_scale_row_bases[i] =
            source_row_valid
                ? reinterpret_cast<uint64_t>(
                      &input_scale[static_cast<uint64_t>(shard_row) * input_scale_row_stride])
                : 0;
      }
    }
    __syncthreads();

    // Prologue
    int itile_to_read = 0;
    int ismem_write = 0;
#pragma unroll
    for (int i = 0; i < kStage - 1; i++) {
      if (i < ntile) {
        if (is_full_tile) {
          scatter_load_A_tile<Config, /*kFullTile=*/true, kPeerInput>(
              tsA_copy, (const Tin *)Aptr, input_ptrs, shm_input_row_bases, shm_row_indices,
              num_valid, i * kTileK, input_row_stride, i);
        } else {
          scatter_load_A_tile<Config, /*kFullTile=*/false, kPeerInput>(
              tsA_copy, (const Tin *)Aptr, input_ptrs, shm_input_row_bases, shm_row_indices,
              num_valid, i * kTileK, input_row_stride, i);
        }
        cute::copy(g2s_tiled_copy_b, tgB_copy(_, _, _, i), tsB_copy(_, _, _, i));
      }
      cp_async_fence();
      ++itile_to_read;
      ++ismem_write;
    }

    // Main loop
    auto tDr = make_tensor_like(tCr);
    clear(tDr);
    float tensor_scale = 0.0f;
    if constexpr (!kBlockwiseScale) {
      tensor_scale = y_scale_ptr[igroup];
    }
    for (int itile = 0; itile < ntile; itile++) {
      // 1. Prefetch: issue next tile (if exists)
      if (itile_to_read < ntile) {
        if (is_full_tile) {
          scatter_load_A_tile<Config, /*kFullTile=*/true, kPeerInput>(
              tsA_copy, (const Tin *)Aptr, input_ptrs, shm_input_row_bases, shm_row_indices,
              num_valid, itile_to_read * kTileK, input_row_stride, ismem_write);
        } else {
          scatter_load_A_tile<Config, /*kFullTile=*/false, kPeerInput>(
              tsA_copy, (const Tin *)Aptr, input_ptrs, shm_input_row_bases, shm_row_indices,
              num_valid, itile_to_read * kTileK, input_row_stride, ismem_write);
        }
        cute::copy(g2s_tiled_copy_b, tgB_copy(_, _, _, itile_to_read),
                   tsB_copy(_, _, _, ismem_write));
        ++itile_to_read;
        ismem_write = (ismem_write + 1) % kStage;
      }
      cp_async_fence();

      if constexpr (kBlockwiseScale) {
        const int k_block = itile * kTileK / kScaleBlock;
        if (idx < num_valid) {
          const float *input_scale;
          if constexpr (kPeerInput) {
            const uint64_t input_scale_row_base = shm_input_scale_row_bases[idx];
            if (input_scale_row_base == 0) {
              shm_input_scales[idx] = 0.0f;
            } else {
              input_scale = reinterpret_cast<const float *>(input_scale_row_base);
              shm_input_scales[idx] = input_scale[k_block];
            }
          } else {
            input_scale = static_cast<const float *>(y_scale_ptr);
            const int local_row = shm_row_indices[idx];
            shm_input_scales[idx] =
                input_scale[static_cast<uint64_t>(local_row) * input_scale_row_stride + k_block];
          }
        }
        constexpr int kWeightScalesPerTile = kTileN / kScaleBlock;
        if (idx < kWeightScalesPerTile) {
          const int num_n_blocks = (n + kScaleBlock - 1) / kScaleBlock;
          const int n_block = itile_n * kWeightScalesPerTile + idx;
          shm_weight_scales[idx] =
              weight_scale_ptr[(static_cast<uint64_t>(igroup) * num_n_blocks + n_block) *
                                   weight_scale_stride +
                               k_block];
        }
      }

      // 2. Wait for the oldest fence group to complete
      cp_async_wait<kStage - 1>();
      __syncthreads();

      // 3. WGMMA on smem stage itile % kStage — swapAB: gemm(B, A, C)
      tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
      warpgroup_fence_operand(tCr);
      warpgroup_arrive();
#pragma unroll
      for (int ik = 0; ik < size<2>(tBr); ++ik) {
        cute::gemm(tiled_mma, tBr(_, _, ik, itile % kStage), tAr(_, _, ik, itile % kStage), tCr);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
      }
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      warpgroup_fence_operand(tCr);
      if constexpr (kBlockwiseScale) {
        auto tDr_mn = retile_fragment(tDr);
#pragma unroll
        for (int im = 0; im < kFragmentM; ++im) {
          const int row = get<1>(tI_mn(0, im));
          const float input_scale = row < num_valid ? shm_input_scales[row] : 0.0f;
#pragma unroll
          for (int in = 0; in < kFragmentN; ++in) {
            const int n_block = get<0>(tI_mn(in, 0)) / kScaleBlock;
            const float scale = input_scale * shm_weight_scales[n_block];
            tDr_mn(in, im) = tCr_mn(in, im) * scale + tDr_mn(in, im);
          }
        }
      } else {
#pragma unroll
        for (int i = 0; i < size(tCr); ++i) {
          tDr(i) = tCr(i) * tensor_scale + tDr(i);
        }
      }
    }

    // Epilogue: float32 → bfloat16 → smem → global
    auto tCrh = make_tensor_like<cute::bfloat16_t>(tCr);
#pragma unroll
    for (int i = 0; i < size(tCr); ++i) {
      tCrh(i) = (Tout)(tDr(i));
    }

    // swapAB: use transpose R2S copy atom
    using R2SCopyAtomC = typename Config::R2SCopyAtomC;
    auto tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
    auto thr_copy_c = tiled_copy_c.get_slice(idx);
    auto tCr4s = thr_copy_c.retile_S(tCrh);
    auto tCs4r = thr_copy_c.partition_D(sCT);

    cute::copy(tiled_copy_c, tCr4s, tCs4r);
    __syncthreads();

    cute::copy_if(s2g_tiled_copy_c, pred_c, tCs2g, tCg2s);
    __syncthreads();
  }

  // PDL release: signal downstream can start.
  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

}  // namespace kernels

// Scatter launch policy. The kTileM=48 specialization uses a tighter
// launch bound to preserve occupancy without register spilling. kStage is
// fixed at 2 to match the dispatcher below.
template <int kTileM, int kTileK, int kStage, bool kUseTaskMap>
struct ScatterLaunchPolicy {
  static constexpr int kMinBlocks = 5;
};
template <int kTileM, int kTileK, bool kUseTaskMap>
struct ScatterLaunchPolicy<kTileM, kTileK, 3, kUseTaskMap> {
  static constexpr int kMinBlocks = 4;
};
template <int kTileK, bool kUseTaskMap>
struct ScatterLaunchPolicy<48, kTileK, 2, kUseTaskMap> {
  static constexpr int kMinBlocks = 7;
};

template <bool kBlockwiseScale, bool kPeerInput>
void launch_group_gemm_fp8_scatter(void *y_ptr, const void *x_ptr, const void *input_ptrs_dev,
                                   const void *input_scale_ptrs_dev, int rows_per_shard,
                                   int num_input_ptrs, const void *w_ptr, const void *y_scale_ptr,
                                   const void *weight_scale_ptr, const void *row_indices_ptr,
                                   int64_t input_row_stride, int64_t input_scale_row_stride,
                                   int weight_scale_stride, const void *seqlens_ptr,
                                   const void *cu_seqlens_ptr, const void *tiles_ptr,
                                   const void *cu_tiles_ptr, const void *task_map_ptr,
                                   int task_map_len, int m, int n, int k, int num_group,
                                   int num_seq_per_group_avg, bool use_pdl, cudaStream_t stream) {
  using namespace cute;  // NOLINT

  using Tin = cute::float_e4m3_t;
  using Tout = cute::bfloat16_t;

  // Precondition: kernel's B load has no N predicate. The 128-wide blockwise
  // tile balances activation reuse with CTA residency for low-M peer loads.
  // See the matching assert in group_gemm_fp8_multistage_async.
  constexpr int kTileN = kBlockwiseScale ? 128 : 64;
  assert(n % kTileN == 0 && "group_gemm_fp8_scatter_async: n must be aligned to kTileN");

  constexpr int kStage = kBlockwiseScale ? 3 : 2;

  int num_tile_n = (n + kTileN - 1) / kTileN;
  cutlass::FastDivmod flat_divider(num_tile_n);
  cutlass::FastDivmod source_row_divider(kPeerInput ? rows_per_shard : 1);
  int source_row_limit = kPeerInput ? rows_per_shard * num_input_ptrs : 0;
  const auto *input_ptrs = static_cast<const void *const *>(input_ptrs_dev);
  const auto *input_scale_ptrs = static_cast<const void *const *>(input_scale_ptrs_dev);

  auto launch_with_tile_m = [&](auto tile_m_tag) {
    constexpr int kTileM = decltype(tile_m_tag)::value;

    auto launch = [&](auto gemm_config) {
      dim3 block(128);
      // sCT aliases sA/sB at the head of SMEM, so total compute SMEM = max(xw, y).
      // After that we append scheduling scratch: shm_tiles + shm_row_indices.
      const int shm_compute = gemm_config.kShmSize;  // = max(shm_xw, shm_y)
      const int peer_row_scratch =
          kPeerInput ? 2 * kTileM * sizeof(uint64_t) + alignof(uint64_t) - 1 : 0;
      const int scale_scratch = kBlockwiseScale ? (kTileM + kTileN / 128) * sizeof(float) : 0;
      const int shm_size = shm_compute + (num_group + 1) * sizeof(int) + kTileM * sizeof(int) +
                           peer_row_scratch + scale_scratch;
      constexpr int kGridMul = grid_multiplier(kTileM);
      dim3 grid(get_sm_count() * kGridMul);
      const bool use_task_map = (task_map_ptr != nullptr);

      auto dispatch = [&](auto use_task_map_tag, auto use_pdl_tag) {
        constexpr bool kUseTaskMap = decltype(use_task_map_tag)::value;
        constexpr bool kUsePDL = decltype(use_pdl_tag)::value;
        // Per-instance launch-bounds hint from ScatterLaunchPolicy.
        using GemmCfgT = std::decay_t<decltype(gemm_config)>;
        constexpr int kScatterMb = ScatterLaunchPolicy<GemmCfgT::kTileM, GemmCfgT::kTileK,
                                                       GemmCfgT::kStage, kUseTaskMap>::kMinBlocks;
        auto kernel =
            kernels::group_gemm_fp8_scatter_kernel<decltype(gemm_config), kUseTaskMap, kUsePDL,
                                                   kBlockwiseScale, kPeerInput, kScatterMb>;
        auto attribute_error =
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
        TORCH_CHECK(attribute_error == cudaSuccess,
                    "grouped GEMM shared-memory configuration failed: ",
                    cudaGetErrorString(attribute_error));
        const int4 *tm_ptr_typed =
            kUseTaskMap ? reinterpret_cast<const int4 *>(task_map_ptr) : nullptr;
        int tm_ub = kUseTaskMap ? task_map_len : 0;

        if constexpr (kUsePDL) {
          cudaLaunchAttribute attr[1];
          attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
          attr[0].val.programmaticStreamSerializationAllowed = 1;
          cudaLaunchConfig_t cfg{};
          cfg.gridDim = grid;
          cfg.blockDim = block;
          cfg.dynamicSmemBytes = shm_size;
          cfg.stream = stream;
          cfg.attrs = attr;
          cfg.numAttrs = 1;
          auto launch_error = cudaLaunchKernelEx(
              &cfg, kernel, y_ptr, x_ptr, input_ptrs, input_scale_ptrs, w_ptr, (float *)y_scale_ptr,
              (float *)weight_scale_ptr, (const int *)row_indices_ptr, input_row_stride,
              input_scale_row_stride, weight_scale_stride, (int *)seqlens_ptr,
              (int *)cu_seqlens_ptr, (int *)tiles_ptr, (int *)cu_tiles_ptr, tm_ptr_typed, tm_ub, m,
              n, k, num_group, source_row_limit, flat_divider, source_row_divider);
          TORCH_CHECK(launch_error == cudaSuccess,
                      "grouped GEMM launch failed: ", cudaGetErrorString(launch_error));
        } else {
          kernel<<<grid, block, shm_size, stream>>>(
              y_ptr, x_ptr, input_ptrs, input_scale_ptrs, w_ptr, (float *)y_scale_ptr,
              (float *)weight_scale_ptr, (const int *)row_indices_ptr, input_row_stride,
              input_scale_row_stride, weight_scale_stride, (int *)seqlens_ptr,
              (int *)cu_seqlens_ptr, (int *)tiles_ptr, (int *)cu_tiles_ptr, tm_ptr_typed, tm_ub, m,
              n, k, num_group, source_row_limit, flat_divider, source_row_divider);
        }
      };

      if (use_task_map) {
        if (use_pdl) {
          dispatch(std::true_type{}, std::true_type{});
        } else {
          dispatch(std::true_type{}, std::false_type{});
        }
      } else {
        if (use_pdl) {
          dispatch(std::false_type{}, std::true_type{});
        } else {
          dispatch(std::false_type{}, std::false_type{});
        }
      }
    };

    if (k % 128 == 0) {
      constexpr int kTileK = 128;
      auto gemm_config = config::FP8GemmConfig<Tin, Tout, kTileM, kTileN, kTileK, kStage>{};
      launch(gemm_config);
    } else if (k % 64 == 0) {
      constexpr int kTileK = 64;
      auto gemm_config = config::FP8GemmConfig<Tin, Tout, kTileM, kTileN, kTileK, kStage>{};
      launch(gemm_config);
    }
  };

  if (num_seq_per_group_avg <= 8) {
    launch_with_tile_m(cute::Int<8>{});
  } else if (num_seq_per_group_avg <= 16) {
    launch_with_tile_m(cute::Int<16>{});
  } else if (num_seq_per_group_avg <= 32) {
    launch_with_tile_m(cute::Int<32>{});
  } else if (num_seq_per_group_avg <= 48) {
    launch_with_tile_m(cute::Int<48>{});
  } else if (num_seq_per_group_avg <= 64) {
    launch_with_tile_m(cute::Int<64>{});
  } else if (num_seq_per_group_avg <= 96) {
    launch_with_tile_m(cute::Int<48>{});
  } else if (num_seq_per_group_avg <= 128) {
    launch_with_tile_m(cute::Int<64>{});
  } else if (num_seq_per_group_avg <= 144) {
    launch_with_tile_m(cute::Int<48>{});
  } else {
    launch_with_tile_m(cute::Int<64>{});
  }
}

void group_gemm_fp8_scatter_async(void *y_ptr, const void *x_ptr, const void *w_ptr,
                                  const void *y_scale_ptr, const void *row_indices_ptr,
                                  const void *seqlens_ptr, const void *cu_seqlens_ptr,
                                  const void *tiles_ptr, const void *cu_tiles_ptr,
                                  const void *task_map_ptr, int task_map_len, int m, int n, int k,
                                  int num_group, int num_seq_per_group_avg, bool use_pdl,
                                  cudaStream_t stream) {
  launch_group_gemm_fp8_scatter<false, false>(
      y_ptr, x_ptr, nullptr, nullptr, m, 0, w_ptr, y_scale_ptr, nullptr, row_indices_ptr, k, 0, 0,
      seqlens_ptr, cu_seqlens_ptr, tiles_ptr, cu_tiles_ptr, task_map_ptr, task_map_len, m, n, k,
      num_group, num_seq_per_group_avg, use_pdl, stream);
}

void group_gemm_blockwise_fp8_scatter_async(
    void *y_ptr, const void *input_ptrs_dev, const void *input_scale_ptrs_dev, int rows_per_shard,
    int num_input_ptrs, const void *w_ptr, const void *w_scale_ptr, const void *row_indices_ptr,
    int64_t input_row_stride, int64_t input_scale_row_stride, int weight_scale_stride,
    const void *seqlens_ptr, const void *cu_seqlens_ptr, const void *tiles_ptr,
    const void *cu_tiles_ptr, const void *task_map_ptr, int task_map_len, int m, int n, int k,
    int num_group, int num_seq_per_group_avg, bool use_pdl, cudaStream_t stream) {
  launch_group_gemm_fp8_scatter<true, true>(
      y_ptr, nullptr, input_ptrs_dev, input_scale_ptrs_dev, rows_per_shard, num_input_ptrs, w_ptr,
      nullptr, w_scale_ptr, row_indices_ptr, input_row_stride, input_scale_row_stride,
      weight_scale_stride, seqlens_ptr, cu_seqlens_ptr, tiles_ptr, cu_tiles_ptr, task_map_ptr,
      task_map_len, m, n, k, num_group, num_seq_per_group_avg, use_pdl, stream);
}

}  // namespace group_gemm_cp_async
}  // namespace hpc
