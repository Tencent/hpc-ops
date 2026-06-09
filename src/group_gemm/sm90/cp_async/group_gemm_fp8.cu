// Copyright 2026 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include <cassert>

#include "cute/tensor.hpp"
#include "src/group_gemm/sm90/cp_async/common.cuh"
#include "src/group_gemm/sm90/cp_async/config.h"
#include "src/group_gemm/sm90/cp_async/group_gemm.h"
#include "src/utils/utils.h"

namespace hpc {
namespace group_gemm_cp_async {
namespace kernels {

// kMinBlocks is a plain compile-time template parameter so __launch_bounds__
// sees a literal integer; the value is chosen by MultistageLaunchPolicy in
// the host dispatcher (see below).
template <typename Config, bool kUseTaskMap, bool kUsePDL, int kMinBlocks>
__global__ void __launch_bounds__(128, kMinBlocks)
    group_gemm_fp8_multistage_kernel(const void *Cptr, const void *Aptr, const void *Bptr,
                                     const float *y_scale_ptr, const int *seqlens_ptr,
                                     const int *cu_seqlens_ptr, const int *tiles_ptr,
                                     const int *cu_tiles_ptr, const int4 *task_map_ptr,
                                     int task_map_len, int m, int n, int k, int num_group,
                                     cutlass::FastDivmod flat_divider) {
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

  extern __shared__ Tin shm_data[];
  Tin *Ashm = shm_data;
  Tin *Bshm = Ashm + cosize(SmemLayoutA{});
  int *shm_tiles = reinterpret_cast<int *>(Bshm + cosize(SmemLayoutB{}));
  Tout *Cshm = (Tout *)(shm_data);

  int idx = threadIdx.x;
  int iblock = blockIdx.x;

  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }

  if constexpr (!kUseTaskMap) {
    for (int i = idx; i < num_group; i += blockDim.x) {
      shm_tiles[i] = tiles_ptr[i];
    }
    __syncthreads();
  }

  int ntile = k / kTileK;
  int igroup = 0;
  int itile_m, itile_n;
  int sum_tile_m = 0;
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

    int start_token = cu_seqlens_ptr[igroup];
    m = seqlens_ptr[igroup];
    iblock += gridDim.x;

    // Global tensors
    Tensor A = make_tensor(make_gmem_ptr((Tin *)Aptr + uint64_t(start_token) * k), make_shape(m, k),
                           make_stride(k, Int<1>{}));
    Tensor B = make_tensor(make_gmem_ptr((Tin *)Bptr + uint64_t(igroup) * n * k), make_shape(n, k),
                           make_stride(k, Int<1>{}));
    Tensor C = make_tensor(make_gmem_ptr((Tout *)Cptr + uint64_t(start_token) * n),
                           make_shape(n, m), make_stride(Int<1>{}, n));

    Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(itile_m, _));
    Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(itile_n, _));
    Tensor gCT =
        local_tile(C, make_tile(Int<kTileN>{}, Int<kTileM>{}), make_coord(itile_n, itile_m));

    // Shared memory tensors
    auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});  // (kTileM, kTileK, kStage)
    auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});  // (kTileN, kTileK, kStage)
    auto sCT = make_tensor(make_smem_ptr((Tout *)Cshm), SmemLayoutCT{});  // (kTileN, kTileM)

    // G2S
    G2SCopyA g2s_tiled_copy_a;
    auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
    auto tgA_copy = g2s_thr_copy_a.partition_S(gA);
    auto tsA_copy = g2s_thr_copy_a.partition_D(sA);

    G2SCopyB g2s_tiled_copy_b;
    auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
    auto tgB_copy = g2s_thr_copy_b.partition_S(gB);
    auto tsB_copy = g2s_thr_copy_b.partition_D(sB);

    // S2G
    S2GCopyC s2g_tiled_copy_c;
    auto s2g_thr_copy_c = s2g_tiled_copy_c.get_slice(idx);
    auto tCs2g = s2g_thr_copy_c.partition_S(sCT);
    auto tCg2s = s2g_thr_copy_c.partition_D(gCT);

    // swapAB
    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(idx);
    auto tBr = thr_mma.make_fragment_A(thr_mma.partition_A(sB));
    auto tAr = thr_mma.make_fragment_B(thr_mma.partition_B(sA));
    auto tCr = thr_mma.partition_fragment_C(gCT);

    // Predicates
    auto tIA =
        g2s_thr_copy_a.partition_S(make_identity_tensor(make_shape(Int<kTileM>{}, Int<kTileK>{})));
    auto tIC =
        s2g_thr_copy_c.partition_S(make_identity_tensor(make_shape(Int<kTileN>{}, Int<kTileM>{})));
    auto pred_a = make_tensor<bool>(shape(tIA));
    auto pred_c = make_tensor<bool>(shape(tIC));
#pragma unroll
    for (int i = 0; i < size(tIA); ++i) {
      pred_a(i) = get<0>(tIA(i)) < kTileM && itile_m * kTileM + get<0>(tIA(i)) < m;
    }
#pragma unroll
    for (int i = 0; i < size(tIC); ++i) {
      pred_c(i) = get<1>(tIC(i)) < kTileM && itile_m * kTileM + get<1>(tIC(i)) < m;
    }

    // Prologue
    int itile_to_read = 0;
    int ismem_write = 0;
#pragma unroll
    for (int i = 0; i < kStage - 1; i++) {
      if (i < ntile) {
        cute::copy_if(g2s_tiled_copy_a, pred_a, tgA_copy(_, _, _, i), tsA_copy(_, _, _, i));
        cute::copy(g2s_tiled_copy_b, tgB_copy(_, _, _, i), tsB_copy(_, _, _, i));
      }
      cp_async_fence();
      ++itile_to_read;
      ++ismem_write;
    }

    // Main loop
    auto tDr = make_tensor_like(tCr);
    clear(tDr);
    float scale = y_scale_ptr[igroup];
    for (int itile = 0; itile < ntile; itile++) {
      if (itile_to_read < ntile) {
        cute::copy_if(g2s_tiled_copy_a, pred_a, tgA_copy(_, _, _, itile_to_read),
                      tsA_copy(_, _, _, ismem_write));
        cute::copy(g2s_tiled_copy_b, tgB_copy(_, _, _, itile_to_read),
                   tsB_copy(_, _, _, ismem_write));
        ++itile_to_read;
        ismem_write = (ismem_write + 1) % kStage;
      }
      cp_async_fence();

      cp_async_wait<kStage - 1>();
      __syncthreads();

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
#pragma unroll
      for (int i = 0; i < size(tCr); ++i) {
        tDr(i) = tCr(i) * scale + tDr(i);
      }
    }

    // Epilogue
    auto tCrh = make_tensor_like<cute::bfloat16_t>(tCr);
#pragma unroll
    for (int i = 0; i < size(tCr); ++i) {
      tCrh(i) = (Tout)(tDr(i));
    }

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

  // PDL release
  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

}  // namespace kernels

// Multistage launch policy. kStage is selected in the dispatcher below
// from the k divisibility, and the kTileM=48 specialization uses a tighter
// launch bound to preserve occupancy without register spilling.
template <int kTileM, int kTileK, int kStage>
struct MultistageLaunchPolicy {
  static constexpr int kMinBlocks = 5;
};
template <int kTileK, int kStage>
struct MultistageLaunchPolicy<48, kTileK, kStage> {
  static constexpr int kMinBlocks = 7;
};

void group_gemm_fp8_multistage_async(void *y_ptr, const void *x_ptr, const void *w_ptr,
                                     const void *y_scale_ptr, const void *seqlens_ptr,
                                     const void *cu_seqlens_ptr, const void *tiles_ptr,
                                     const void *cu_tiles_ptr, const void *task_map_ptr,
                                     int task_map_len, int m, int n, int k, int num_group,
                                     int num_seq_per_group_avg, bool use_pdl, cudaStream_t stream) {
  using namespace cute;  // NOLINT

  using Tin = cute::float_e4m3_t;
  using Tout = cute::bfloat16_t;

  // B-matrix load has no N predicate; n must align to kTileN=64.
  assert(n % 64 == 0 && "group_gemm_fp8_multistage_async: n must be multiple of kTileN=64");

  // Shapes: A [m, k] activations (fp8, sorted by expert),
  //         B [num_group, n, k] per-expert weights (fp8),
  //         C [m, n] output (bf16).
  constexpr int kTileN = 64;
  const int num_tile_n = (n + kTileN - 1) / kTileN;
  cutlass::FastDivmod flat_divider(num_tile_n);

  auto launch = [&](auto tile_m_tag, auto tile_k_tag, auto stage_tag) {
    constexpr int kTileM = decltype(tile_m_tag)::value;
    constexpr int kTileK = decltype(tile_k_tag)::value;
    constexpr int kStage = decltype(stage_tag)::value;
    constexpr int kMinBlocks = MultistageLaunchPolicy<kTileM, kTileK, kStage>::kMinBlocks;

    using GemmConfig = config::FP8GemmConfig<Tin, Tout, kTileM, kTileN, kTileK, kStage>;
    GemmConfig gemm_config;

    dim3 block(128);
    const int shm_size = gemm_config.kShmSize + (num_group + 1) * sizeof(int);
    constexpr int kGridMul = grid_multiplier(kTileM);
    dim3 grid(get_sm_count() * kGridMul);
    const bool use_task_map = (task_map_ptr != nullptr);

    auto dispatch = [&](auto use_task_map_tag, auto use_pdl_tag) {
      constexpr bool kUseTaskMap = decltype(use_task_map_tag)::value;
      constexpr bool kUsePDL = decltype(use_pdl_tag)::value;
      auto kernel =
          kernels::group_gemm_fp8_multistage_kernel<GemmConfig, kUseTaskMap, kUsePDL, kMinBlocks>;
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
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
        cudaLaunchKernelEx(&cfg, kernel, y_ptr, x_ptr, w_ptr, (float *)y_scale_ptr,
                           (int *)seqlens_ptr, (int *)cu_seqlens_ptr, (int *)tiles_ptr,
                           (int *)cu_tiles_ptr, tm_ptr_typed, tm_ub, m, n, k, num_group,
                           flat_divider);
      } else {
        kernel<<<grid, block, shm_size, stream>>>(
            y_ptr, x_ptr, w_ptr, (float *)y_scale_ptr, (int *)seqlens_ptr, (int *)cu_seqlens_ptr,
            (int *)tiles_ptr, (int *)cu_tiles_ptr, tm_ptr_typed, tm_ub, m, n, k, num_group,
            flat_divider);
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

  // Pick kTileK and kStage from k divisibility.
  auto dispatch_k_stage = [&](auto tile_m_tag) {
    if (k % 128 == 0) {
      launch(tile_m_tag, Int<128>{}, Int<2>{});
    } else if (k % 64 == 0) {
      launch(tile_m_tag, Int<64>{}, Int<3>{});
    }
    // k not divisible by 64: caller responsibility; silently skip.
  };

  // Pick kTileM from the average tokens per expert.
  if (num_seq_per_group_avg <= 8) {
    dispatch_k_stage(Int<8>{});
  } else if (num_seq_per_group_avg <= 16) {
    dispatch_k_stage(Int<16>{});
  } else if (num_seq_per_group_avg <= 32) {
    dispatch_k_stage(Int<32>{});
  } else if (num_seq_per_group_avg <= 48) {
    dispatch_k_stage(Int<48>{});
  } else if (num_seq_per_group_avg <= 64) {
    dispatch_k_stage(Int<64>{});
  } else if (num_seq_per_group_avg <= 96) {
    dispatch_k_stage(Int<48>{});
  } else if (num_seq_per_group_avg <= 128) {
    dispatch_k_stage(Int<64>{});
  } else if (num_seq_per_group_avg <= 144) {
    dispatch_k_stage(Int<48>{});
  } else {
    dispatch_k_stage(Int<64>{});
  }
}

}  // namespace group_gemm_cp_async
}  // namespace hpc
