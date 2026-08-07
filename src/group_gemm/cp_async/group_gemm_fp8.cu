// Copyright (C) 2026 Tencent.

#include <cuda.h>

#include <cassert>
#include <type_traits>

#include "cute/tensor.hpp"
#include "src/group_gemm/cp_async/common.cuh"
#include "src/group_gemm/cp_async/config.h"
#include "src/group_gemm/cp_async/group_gemm.h"
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

template <typename Config, typename TsACopy>
__device__ __forceinline__ void route_load_A_tile(
    TsACopy &tsA_copy, const typename Config::Tin *__restrict__ input_ptr,
    int input_row, int k_col_base, int k, int ismem) {
  constexpr int kTileM = Config::kTileM;
  constexpr int kTileK = Config::kTileK;
  constexpr int kElemsPerAtom = 16;
  constexpr int kThreadsPerRow = kTileK / kElemsPerAtom;
  constexpr int kRowsPerIter = 128 / kThreadsPerRow;
  constexpr int kNumIters = (kTileM + kRowsPerIter - 1) / kRowsPerIter;

#pragma unroll
  for (int row_iter = 0; row_iter < kNumIters; ++row_iter) {
    const int local_row = row_iter * kRowsPerIter + threadIdx.x / kThreadsPerRow;
    if (local_row < kTileM) {
      const int k_thread = threadIdx.x % kThreadsPerRow;
      const int k_col = k_col_base + k_thread * kElemsPerAtom;
      void *smem_ptr =
          (void *)&tsA_copy(cute::Int<0>{}, row_iter, cute::Int<0>{}, ismem);
      const void *gmem_ptr =
          (const void *)(input_ptr + static_cast<uint64_t>(input_row) * k + k_col);
      const int src_size = local_row == 0 ? 16 : 0;
      asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n" : :
                   "r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr))),
                   "l"(gmem_ptr), "n"(16), "r"(src_size));
    }
  }
}

// One logical route per WGMMA M tile. SM90 WGMMA has an M=8 minimum here, so
// row 0 carries the route and rows 1..7 are zero-filled by cp.async. This is
// deliberately route-major: the expert is read directly from topk_ids and no
// sorting metadata is needed.
template <typename Config, bool kInputIsToken, bool kBlockwiseScale>
__global__ void __launch_bounds__(128, 5) group_gemm_fp8_route_kernel(
    void *Cptr, const void *Aptr, const void *Bptr, const float *y_scale_ptr,
    const float *input_scale_ptr, const float *weight_scale_ptr,
    const int *topk_ids_ptr, int num_routes, int num_topk, int n, int k,
    int num_expert_local, int start_expert, int num_splits, int tiles_per_split,
    int input_scale_stride, int weight_scale_stride,
    cutlass::FastDivmod flat_divider, cutlass::FastDivmod split_divider,
    cutlass::FastDivmod topk_divider) {
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
  static_assert(kTileM == 8, "route-direct kernel requires the SM90 M=8 tile");

  extern __shared__ Tin shm_data[];
  Tin *Ashm = shm_data;
  Tin *Bshm = Ashm + cosize(SmemLayoutA{});
  Tout *Cshm = reinterpret_cast<Tout *>(shm_data);

  cudaGridDependencySynchronize();

  const int num_tile_n = n / kTileN;
  const int num_tasks = num_routes * num_splits * num_tile_n;
  for (int task_idx = blockIdx.x; task_idx < num_tasks; task_idx += gridDim.x) {
    int route_split, itile_n;
    flat_divider(route_split, itile_n, task_idx);
    int route, split;
    split_divider(route, split, route_split);
    const int local_expert = topk_ids_ptr[route] - start_expert;
    const uint64_t output_row = static_cast<uint64_t>(route) * num_splits + split;

    if (local_expert < 0 || local_expert >= num_expert_local) {
      if (threadIdx.x < kTileN) {
        reinterpret_cast<Tout *>(Cptr)[output_row * n +
                                       itile_n * kTileN + threadIdx.x] = Tout{};
      }
      continue;
    }

    int input_row = route;
    if constexpr (kInputIsToken) {
      int topk_slot;
      topk_divider(input_row, topk_slot, route);
    }

    Tensor B = make_tensor(
        make_gmem_ptr((Tin *)Bptr + static_cast<uint64_t>(local_expert) * n * k),
        make_shape(n, k), make_stride(k, Int<1>{}));
    Tensor C = make_tensor(make_gmem_ptr((Tout *)Cptr + output_row * n),
                           make_shape(n, 1), make_stride(Int<1>{}, n));
    Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}),
                           make_coord(itile_n, _));
    Tensor gCT = local_tile(C, make_tile(Int<kTileN>{}, Int<kTileM>{}),
                            make_coord(itile_n, 0));

    auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});
    auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});
    auto sCT = make_tensor(make_smem_ptr(Cshm), SmemLayoutCT{});

    G2SCopyA g2s_tiled_copy_a;
    auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(threadIdx.x);
    auto tsA_copy = g2s_thr_copy_a.partition_D(sA);

    G2SCopyB g2s_tiled_copy_b;
    auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(threadIdx.x);
    auto tgB_copy = g2s_thr_copy_b.partition_S(gB);
    auto tsB_copy = g2s_thr_copy_b.partition_D(sB);

    S2GCopyC s2g_tiled_copy_c;
    auto s2g_thr_copy_c = s2g_tiled_copy_c.get_slice(threadIdx.x);
    auto tCs2g = s2g_thr_copy_c.partition_S(sCT);
    auto tCg2s = s2g_thr_copy_c.partition_D(gCT);

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    auto tBr = thr_mma.make_fragment_A(thr_mma.partition_A(sB));
    auto tAr = thr_mma.make_fragment_B(thr_mma.partition_B(sA));
    auto tCr = thr_mma.partition_fragment_C(gCT);

    auto tIC = s2g_thr_copy_c.partition_S(
        make_identity_tensor(make_shape(Int<kTileN>{}, Int<kTileM>{})));
    auto pred_c = make_tensor<bool>(shape(tIC));
#pragma unroll
    for (int i = 0; i < size(tIC); ++i) {
      pred_c(i) = get<1>(tIC(i)) == 0;
    }

    const int ntile = tiles_per_split;
    const int first_k_tile = split * tiles_per_split;
    int itile_to_read = 0;
    int ismem_write = 0;
#pragma unroll
    for (int stage = 0; stage < kStage - 1; ++stage) {
      if (stage < ntile) {
        route_load_A_tile<Config>(tsA_copy, static_cast<const Tin *>(Aptr), input_row,
                                  (first_k_tile + stage) * kTileK, k, stage);
        cute::copy(g2s_tiled_copy_b, tgB_copy(_, _, _, first_k_tile + stage),
                   tsB_copy(_, _, _, stage));
      }
      cp_async_fence();
      ++itile_to_read;
      ++ismem_write;
    }

    auto tDr = make_tensor_like(tCr);
    clear(tDr);
    constexpr int kScaleBlock = 128;
    const float *input_scale_row = nullptr;
    const float *weight_scale_row = nullptr;
    float tensor_scale = 0.0f;
    if constexpr (kBlockwiseScale) {
      const int n_block = itile_n * kTileN / kScaleBlock;
      const int num_n_blocks = (n + kScaleBlock - 1) / kScaleBlock;
      input_scale_row = input_scale_ptr +
                        static_cast<uint64_t>(input_row) * input_scale_stride;
      weight_scale_row = weight_scale_ptr +
                         (static_cast<uint64_t>(local_expert) * num_n_blocks +
                          n_block) *
                             weight_scale_stride;
    } else {
      tensor_scale = y_scale_ptr[local_expert];
    }
    for (int itile = 0; itile < ntile; ++itile) {
      if (itile_to_read < ntile) {
        route_load_A_tile<Config>(tsA_copy, static_cast<const Tin *>(Aptr), input_row,
                                  (first_k_tile + itile_to_read) * kTileK, k, ismem_write);
        cute::copy(g2s_tiled_copy_b, tgB_copy(_, _, _, first_k_tile + itile_to_read),
                   tsB_copy(_, _, _, ismem_write));
        ++itile_to_read;
        ismem_write = (ismem_write + 1) % kStage;
      }
      cp_async_fence();

      float scale;
      if constexpr (kBlockwiseScale) {
        const int global_k_tile = first_k_tile + itile;
        const int k_block = global_k_tile * kTileK / kScaleBlock;
        scale = input_scale_row[k_block] * weight_scale_row[k_block];
      } else {
        scale = tensor_scale;
      }

      cp_async_wait<kStage - 1>();
      __syncthreads();

      tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
      warpgroup_fence_operand(tCr);
      warpgroup_arrive();
#pragma unroll
      for (int ik = 0; ik < size<2>(tBr); ++ik) {
        cute::gemm(tiled_mma, tBr(_, _, ik, itile % kStage),
                   tAr(_, _, ik, itile % kStage), tCr);
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

    auto tCrh = make_tensor_like<Tout>(tCr);
#pragma unroll
    for (int i = 0; i < size(tCr); ++i) {
      tCrh(i) = Tout(tDr(i));
    }
    using R2SCopyAtomC = typename Config::R2SCopyAtomC;
    auto tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
    auto thr_copy_c = tiled_copy_c.get_slice(threadIdx.x);
    auto tCr4s = thr_copy_c.retile_S(tCrh);
    auto tCs4r = thr_copy_c.partition_D(sCT);
    cute::copy(tiled_copy_c, tCr4s, tCs4r);
    __syncthreads();
    cute::copy_if(s2g_tiled_copy_c, pred_c, tCs2g, tCg2s);
    __syncthreads();
  }

  // Do not trigger PDL early: route branches can leave different warps at the
  // tail at different times. Natural CTA completion preserves the dependency.
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

void group_gemm_fp8_route_async(void *y_ptr, const void *x_ptr, const void *w_ptr,
                                const void *y_scale_ptr, const void *topk_ids_ptr,
                                int num_routes, int num_topk, int n, int k,
                                int num_expert_local, int rank_ep, bool input_is_token,
                                cudaStream_t stream) {
  using namespace cute;  // NOLINT
  using Tin = cute::float_e4m3_t;
  using Tout = cute::bfloat16_t;

  assert(n % 64 == 0 && "group_gemm_fp8_route_async: n must be a multiple of 64");
  assert(k % 64 == 0 && "group_gemm_fp8_route_async: k must be a multiple of 64");

  constexpr int kTileM = 8;
  constexpr int kTileN = 64;
  const int num_tile_n = n / kTileN;
  const int num_tasks = num_routes * num_tile_n;
  if (num_tasks == 0) {
    return;
  }
  cutlass::FastDivmod flat_divider(num_tile_n);
  cutlass::FastDivmod split_divider(1);
  cutlass::FastDivmod topk_divider(num_topk);
  const int start_expert = rank_ep * num_expert_local;

  auto launch = [&](auto tile_k_tag, auto stage_tag, auto input_is_token_tag) {
    constexpr int kTileK = decltype(tile_k_tag)::value;
    constexpr int kStage = decltype(stage_tag)::value;
    constexpr bool kInputIsToken = decltype(input_is_token_tag)::value;
    using GemmConfig = config::FP8GemmConfig<Tin, Tout, kTileM, kTileN, kTileK, kStage>;
    GemmConfig gemm_config;
    const int shm_size = gemm_config.kShmSize;
    auto kernel =
        kernels::group_gemm_fp8_route_kernel<GemmConfig, kInputIsToken, false>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg{};
    const int max_grid = get_sm_count() * grid_multiplier(kTileM);
    cfg.gridDim = dim3(num_tasks < max_grid ? num_tasks : max_grid);
    cfg.blockDim = dim3(128);
    cfg.dynamicSmemBytes = shm_size;
    cfg.stream = stream;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, kernel, y_ptr, x_ptr, w_ptr,
                       static_cast<const float *>(y_scale_ptr), nullptr, nullptr,
                       static_cast<const int *>(topk_ids_ptr), num_routes, num_topk, n, k,
                       num_expert_local, start_expert, 1, k / kTileK, 0, 0,
                       flat_divider, split_divider, topk_divider);
  };

  auto dispatch_input = [&](auto tile_k_tag, auto stage_tag) {
    if (input_is_token) {
      launch(tile_k_tag, stage_tag, std::true_type{});
    } else {
      launch(tile_k_tag, stage_tag, std::false_type{});
    }
  };
  if (k % 128 == 0) {
    dispatch_input(Int<128>{}, Int<2>{});
  } else {
    dispatch_input(Int<64>{}, Int<3>{});
  }
}

void group_gemm_fp8_route_blockwise_async(
    void *y_ptr, const void *x_ptr, const void *x_scale_ptr,
    const void *w_ptr, const void *w_scale_ptr, const void *topk_ids_ptr,
    int num_routes, int num_topk, int n, int k, int num_splits,
    int num_expert_local, int rank_ep, bool input_is_token, int weight_scale_stride,
    cudaStream_t stream) {
  using namespace cute;  // NOLINT
  using Tin = cute::float_e4m3_t;
  using Tout = cute::bfloat16_t;

  assert(n % 64 == 0 &&
         "group_gemm_fp8_route_blockwise_async: n must be a multiple of 64");
  assert(k % 64 == 0 &&
         "group_gemm_fp8_route_blockwise_async: k must be a multiple of 64");
  assert(num_splits > 0 &&
         "group_gemm_fp8_route_blockwise_async: num_splits must be positive");

  constexpr int kTileM = 8;
  constexpr int kTileN = 64;
  constexpr int kScaleBlock = 128;
  const int num_tile_n = n / kTileN;
  const int num_tasks = num_routes * num_splits * num_tile_n;
  if (num_tasks == 0) {
    return;
  }
  cutlass::FastDivmod flat_divider(num_tile_n);
  cutlass::FastDivmod split_divider(num_splits);
  cutlass::FastDivmod topk_divider(num_topk);
  const int start_expert = rank_ep * num_expert_local;
  const int input_scale_stride = (k + kScaleBlock - 1) / kScaleBlock;

  auto launch = [&](auto tile_k_tag, auto stage_tag,
                    auto input_is_token_tag) {
    constexpr int kTileK = decltype(tile_k_tag)::value;
    constexpr int kStage = decltype(stage_tag)::value;
    constexpr bool kInputIsToken = decltype(input_is_token_tag)::value;
    assert((k / kTileK) % num_splits == 0 &&
           "group_gemm_fp8_route_blockwise_async: split must divide K tiles");
    const int tiles_per_split = (k / kTileK) / num_splits;
    using GemmConfig =
        config::FP8GemmConfig<Tin, Tout, kTileM, kTileN, kTileK, kStage>;
    GemmConfig gemm_config;
    const int shm_size = gemm_config.kShmSize;
    auto kernel =
        kernels::group_gemm_fp8_route_kernel<GemmConfig, kInputIsToken, true>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         shm_size);

    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg{};
    const int max_grid = get_sm_count() * grid_multiplier(kTileM);
    cfg.gridDim = dim3(num_tasks < max_grid ? num_tasks : max_grid);
    cfg.blockDim = dim3(128);
    cfg.dynamicSmemBytes = shm_size;
    cfg.stream = stream;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(
        &cfg, kernel, y_ptr, x_ptr, w_ptr, nullptr,
        static_cast<const float *>(x_scale_ptr),
        static_cast<const float *>(w_scale_ptr),
        static_cast<const int *>(topk_ids_ptr), num_routes, num_topk, n, k,
        num_expert_local, start_expert, num_splits, tiles_per_split, input_scale_stride,
        weight_scale_stride, flat_divider, split_divider, topk_divider);
  };

  auto dispatch_input = [&](auto tile_k_tag, auto stage_tag) {
    if (input_is_token) {
      launch(tile_k_tag, stage_tag, std::true_type{});
    } else {
      launch(tile_k_tag, stage_tag, std::false_type{});
    }
  };
  if (k % 128 == 0 && (k / 128) % num_splits == 0) {
    dispatch_input(Int<128>{}, Int<2>{});
  } else {
    assert((k / 64) % num_splits == 0 &&
           "group_gemm_fp8_route_blockwise_async: split must divide K tiles");
    dispatch_input(Int<64>{}, Int<3>{});
  }
}

void group_gemm_fp8_route_splitk_async(void *partial_ptr, const void *x_ptr,
                                       const void *w_ptr, const void *y_scale_ptr,
                                       const void *topk_ids_ptr, int num_routes,
                                       int num_topk, int n, int k, int num_splits,
                                       int num_expert_local, int rank_ep,
                                       cudaStream_t stream) {
  using namespace cute;  // NOLINT
  using Tin = cute::float_e4m3_t;
  using Tout = cute::bfloat16_t;

  assert(n % 64 == 0 && "group_gemm_fp8_route_splitk_async: n must be a multiple of 64");
  assert(k % 64 == 0 && "group_gemm_fp8_route_splitk_async: k must be a multiple of 64");
  assert(num_splits > 0 && "group_gemm_fp8_route_splitk_async: num_splits must be positive");

  constexpr int kTileM = 8;
  constexpr int kTileN = 64;
  const int num_tile_n = n / kTileN;
  const int num_tasks = num_routes * num_splits * num_tile_n;
  if (num_tasks == 0) {
    return;
  }
  cutlass::FastDivmod flat_divider(num_tile_n);
  cutlass::FastDivmod split_divider(num_splits);
  cutlass::FastDivmod topk_divider(num_topk);
  const int start_expert = rank_ep * num_expert_local;

  auto launch = [&](auto tile_k_tag, auto stage_tag) {
    constexpr int kTileK = decltype(tile_k_tag)::value;
    constexpr int kStage = decltype(stage_tag)::value;
    assert((k / kTileK) % num_splits == 0 &&
           "group_gemm_fp8_route_splitk_async: split must divide K tiles");
    const int tiles_per_split = (k / kTileK) / num_splits;

    using GemmConfig = config::FP8GemmConfig<Tin, Tout, kTileM, kTileN, kTileK, kStage>;
    GemmConfig gemm_config;
    const int shm_size = gemm_config.kShmSize;
    auto kernel = kernels::group_gemm_fp8_route_kernel<GemmConfig, true, false>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

    const int max_grid = get_sm_count() * grid_multiplier(kTileM);
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(num_tasks < max_grid ? num_tasks : max_grid);
    cfg.blockDim = dim3(128);
    cfg.dynamicSmemBytes = shm_size;
    cfg.stream = stream;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, kernel, partial_ptr, x_ptr, w_ptr,
                       static_cast<const float *>(y_scale_ptr), nullptr, nullptr,
                       static_cast<const int *>(topk_ids_ptr), num_routes, num_topk, n, k,
                       num_expert_local, start_expert, num_splits, tiles_per_split,
                       0, 0, flat_divider, split_divider, topk_divider);
  };

  if (k % 128 == 0 && (k / 128) % num_splits == 0) {
    launch(Int<128>{}, Int<2>{});
  } else {
    launch(Int<64>{}, Int<3>{});
  }
}

}  // namespace group_gemm_cp_async
}  // namespace hpc
