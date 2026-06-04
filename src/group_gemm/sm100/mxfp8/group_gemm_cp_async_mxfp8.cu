// Copyright 2026 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include <cub/cub.cuh>

#include "cute/tensor.hpp"
#include "src/group_gemm/sm100/gemm_config.h"
#include "src/group_gemm/sm100/group_gemm.h"
#include "src/group_gemm/sm100/mxfp8/group_gemm_cp_async_mxfp8.cuh"
#include "src/group_gemm/sm100/mxfp8/group_gemm_mxfp8.cuh"  // for update_grouped_tma_mxfp8

namespace hpc {
namespace group_gemm {

template <typename GemmConfig, int kCtaPerSm = 1, bool kUsePDL = false>
void launch_group_gemm_1sm_cp_async_mxfp8(void *y_ptr, const void *x_ptr, const void *w_ptr,
                                          const void *sfx_packed_ptr, const void *sfw_packed_ptr,
                                          const void *seqlens_ptr, const void *cu_seqlens_ptr,
                                          void *tmas_ptr, void *tiles_ptr, void *cu_tiles_ptr,
                                          int num_group, int m, int n, int k, bool update_tma,
                                          cudaStream_t stream, const void *x_row_map_ptr,
                                          int x_num_rows) {
  using namespace cute;  // NOLINT

  using Tin = typename GemmConfig::Tin;
  using Tout = typename GemmConfig::Tout;
  using Tsf = typename GemmConfig::Tsf;
  constexpr int kTileM = GemmConfig::kTileM;
  constexpr int kTileN = GemmConfig::kTileN;
  constexpr int kSfxRows = GemmConfig::kSfxRows;
  constexpr int kSfVec = GemmConfig::kSfVec;

  int n_padded = ((n + 127) / 128) * 128;
  int k_sf = k / kSfVec;
  int k_sf_tiles = (k_sf + 3) / 4;

  // A is *not* a TMA tensor here; we only need a CPU-side dummy A tensor for
  // get_tma() (its tma_a is unused but the API still expects an A handle).
  // We pass the gathered logical shape (m, k) to satisfy any size queries.
  auto A = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(x_ptr)), make_shape(m, k),
                       make_stride(k, Int<1>{}));
  auto B = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(w_ptr)),
                       make_shape(n, k, num_group), make_stride(k, Int<1>{}, n * k));
  auto Y = make_tensor(make_gmem_ptr(reinterpret_cast<Tout *>(y_ptr)), make_shape(n, m),
                       make_stride(Int<1>{}, n));

  int sfx_max_tiles = m / kTileM + num_group;
  auto SFA =
      make_tensor(make_gmem_ptr(reinterpret_cast<const Tsf *>(sfx_packed_ptr)),
                  make_shape(Int<kSfxRows>{}, Int<16>{}, sfx_max_tiles, k_sf_tiles),
                  make_stride(Int<16>{}, Int<1>{}, k_sf_tiles * kSfxRows * 16, kSfxRows * 16));
  auto SFB = make_tensor(make_gmem_ptr(reinterpret_cast<const Tsf *>(sfw_packed_ptr)),
                         make_shape(Int<32>{}, Int<16>{}, num_group * (n_padded / 128), k_sf_tiles),
                         make_stride(Int<16>{}, Int<1>{}, k_sf_tiles * 32 * 16, 32 * 16));

  GemmConfig config;
  auto [tma_a, tma_b, tma_y, tma_sfa, tma_sfb] = config.get_tma(A, B, Y, SFA, SFB);

  auto *tma_ay = static_cast<cute::TmaDescriptor *>(tmas_ptr);
  int num_sm = get_sm_count();

  // Per-group Y TMA descriptors. We reuse update_grouped_tma_mxfp8 which also
  // touches the per-group X descriptors; those are unused by the cp.async
  // kernel but updating them is cheap (~ns) and keeps a single update path.
  if (update_tma) {
    vec_t<cute::TmaDescriptor, 2> td_ay{
        *tma_a.get_tma_descriptor(),
        *tma_y.get_tma_descriptor(),
    };

    constexpr int kGroupPerThread = 8;
    constexpr int kThreadPerBlock = 32;
    auto kernel = kernels::update_grouped_tma_mxfp8<Tin, Tout, decltype(tma_a), decltype(tma_y),
                                                    kTileM, kGroupPerThread, kThreadPerBlock>;
    kernel<<<num_group + 1, kThreadPerBlock, 0, stream>>>(
        td_ay, tma_ay, (const Tin *)x_ptr, (const Tout *)y_ptr, (const int *)seqlens_ptr,
        (const int *)cu_seqlens_ptr, (int *)tiles_ptr, (int *)cu_tiles_ptr, num_group, m, n, k);
  }

  // Group GEMM with cp.async A.
  {
    int num_tile_n_per_group = (n + kTileN - 1) / kTileN;
    cutlass::FastDivmod flat_divider(num_tile_n_per_group);

    int shm_seq = 2 * sizeof(int) * (num_group + 1);
    int shm_size = config.get_shm_size() + shm_seq;

    dim3 block(384);
    dim3 grid(num_sm * kCtaPerSm);

    auto kernel = kernels::group_gemm_1sm_cp_async_mxfp8_kernel<decltype(config), decltype(tma_b),
                                                                decltype(tma_y), decltype(tma_sfa),
                                                                decltype(tma_sfb), kUsePDL>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

    cudaLaunchConfig_t launch_config;
    memset(&launch_config, 0, sizeof(launch_config));
    launch_config.gridDim = grid;
    launch_config.blockDim = block;
    launch_config.dynamicSmemBytes = shm_size;
    launch_config.stream = stream;
    cudaLaunchAttribute attr[2];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim = {1, 1, 1};
    int num_attrs = 1;
    if constexpr (kUsePDL) {
      attr[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attr[1].val.programmaticStreamSerializationAllowed = 1;
      num_attrs = 2;
    }
    launch_config.numAttrs = num_attrs;
    launch_config.attrs = attr;

    cudaLaunchKernelEx(&launch_config, kernel, tma_b, tma_sfb, tma_sfa, tma_ay,
                       reinterpret_cast<const Tin *>(x_ptr),
                       reinterpret_cast<const int *>(x_row_map_ptr), x_num_rows, (int *)seqlens_ptr,
                       (int *)cu_seqlens_ptr, (int *)tiles_ptr, (int *)cu_tiles_ptr, num_group, m,
                       n, k, flat_divider);
  }
}

namespace {
template <typename T>
struct type_id {
  using type = T;
};
}  // namespace

void group_gemm_cp_async_mxfp8_async(void *y_ptr, const void *x_ptr, const void *w_ptr,
                                     const void *sfx_packed_ptr, const void *sfw_packed_ptr,
                                     const void *seqlens_ptr, const void *cu_seqlens_ptr,
                                     void *tmas_ptr, void *tiles_ptr, void *cu_tiles_ptr,
                                     int num_group, int m, int n, int k, int num_seq_per_group_avg,
                                     bool update_tma, cudaStream_t stream,
                                     const void *x_row_map_ptr, int x_num_rows, bool use_pdl) {
  using Tin = cute::float_e4m3_t;
  using Tout = cute::bfloat16_t;
  using Tsf = cutlass::float_ue8m0_t;

  // 1SM only — cp.async path does not use 2SM.
  int kTileM_dispatch = mxfp8_dispatch_kTileM_cp_async(num_seq_per_group_avg);

  auto launch = [&](auto cfg_tag) {
    using Cfg = typename decltype(cfg_tag)::type;
    if (use_pdl) {
      launch_group_gemm_1sm_cp_async_mxfp8<Cfg, 1, true>(
          y_ptr, x_ptr, w_ptr, sfx_packed_ptr, sfw_packed_ptr, seqlens_ptr, cu_seqlens_ptr,
          tmas_ptr, tiles_ptr, cu_tiles_ptr, num_group, m, n, k, update_tma, stream, x_row_map_ptr,
          x_num_rows);
    } else {
      launch_group_gemm_1sm_cp_async_mxfp8<Cfg, 1, false>(
          y_ptr, x_ptr, w_ptr, sfx_packed_ptr, sfw_packed_ptr, seqlens_ptr, cu_seqlens_ptr,
          tmas_ptr, tiles_ptr, cu_tiles_ptr, num_group, m, n, k, update_tma, stream, x_row_map_ptr,
          x_num_rows);
    }
  };

  assert(k % 32 == 0 && "group_gemm_cp_async_mxfp8: k must be a multiple of 32 (SF_VEC)");

  //        <Tin, Tout, Tsf, KTM, kTileN, KTK, EPI, STAGE, STAGE_TMA, kMmaSM, STAGE_TILE>
  switch (kTileM_dispatch) {
    case 16:
      return launch(type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 16, 128, 128, 16, 8, 4, 1, 4>>{});
    case 32:
      return launch(type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 32, 128, 128, 32, 6, 4, 1, 4>>{});
    case 48:
      return launch(type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 48, 128, 128, 16, 6, 4, 1, 4>>{});
    case 64:
      return launch(type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 64, 128, 128, 64, 6, 4, 1, 4>>{});
    case 128:
      return launch(type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 128, 128, 128, 64, 5, 3, 1, 3>>{});
    default:
      return launch(type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 256, 128, 128, 64, 3, 2, 1, 1>>{});
  }
}

}  // namespace group_gemm
}  // namespace hpc
