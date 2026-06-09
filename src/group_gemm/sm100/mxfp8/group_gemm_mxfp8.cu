// Copyright 2026 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include <cub/cub.cuh>
#include <type_traits>

#include "cute/tensor.hpp"
#include "src/group_gemm/sm100/gemm_config.h"
#include "src/group_gemm/sm100/group_gemm.h"
#include "src/group_gemm/sm100/mxfp8/group_gemm_mxfp8.cuh"

namespace hpc {
namespace group_gemm {

// prepack_mxfp8_w_scale_async
//   Offline: rearrange weight-side scale (SFW) into UTCCP-friendly layout.
//   SFW is regular (every group has n rows, tile-aligned). Call once at model load.
void prepack_mxfp8_w_scale_async(void *sfw_packed_ptr, const void *sfw_ptr, int num_group, int n,
                                 int k, bool is_2sm, cudaStream_t stream) {
  constexpr int kSfVec = 32;
  int k_sf = k / kSfVec;

  if (is_2sm) {
    constexpr int kRowsPerTile = 256;
    int row_tiles = n / kRowsPerTile;
    dim3 grid_w(num_group, row_tiles);
    kernels::prepack_mxfp8_scale_kernel<256, 64>
        <<<grid_w, 64, 0, stream>>>(reinterpret_cast<const uint8_t *>(sfw_ptr),
                                    reinterpret_cast<uint8_t *>(sfw_packed_ptr), n, k_sf);
  } else {
    constexpr int kRowsPerTile = 128;
    int row_tiles = n / kRowsPerTile;
    dim3 grid_w(num_group, row_tiles);
    kernels::prepack_mxfp8_scale_kernel<128, 32>
        <<<grid_w, 32, 0, stream>>>(reinterpret_cast<const uint8_t *>(sfw_ptr),
                                    reinterpret_cast<uint8_t *>(sfw_packed_ptr), n, k_sf);
  }
}

// prepack_mxfp8_x_scale_async
//   Online: rearrange activation-side scale (SFX) into UTCCP-friendly layout.
//   SFX is variable-seqlen; each group's dst offset derived from cu_seqlens + kTileM.
//   Must run every forward pass.
void prepack_mxfp8_x_scale_async(void *sfx_packed_ptr, const void *sfx_ptr,
                                 const void *cu_seqlens_ptr, int num_group, int m_total, int k,
                                 int kTileM, bool is_smallm, cudaStream_t stream) {
  constexpr int kSfVec = 32;
  int k_sf = k / kSfVec;

  // max_row_tiles: upper bound across all groups; out-of-range CTAs early-exit inside kernel
  int max_row_tiles = (m_total + kTileM - 1) / kTileM;
  if (is_smallm) {
    dim3 grid_x(num_group, max_row_tiles);
    kernels::prepack_mxfp8_scale_kernel<128, 32><<<grid_x, 32, 0, stream>>>(
        reinterpret_cast<const uint8_t *>(sfx_ptr), reinterpret_cast<uint8_t *>(sfx_packed_ptr),
        reinterpret_cast<const int *>(cu_seqlens_ptr), k_sf,
        /*row_stride=*/kTileM, num_group);
  } else {
    dim3 grid_x(num_group, max_row_tiles);
    kernels::prepack_mxfp8_scale_kernel<256, 64><<<grid_x, 64, 0, stream>>>(
        reinterpret_cast<const uint8_t *>(sfx_ptr), reinterpret_cast<uint8_t *>(sfx_packed_ptr),
        reinterpret_cast<const int *>(cu_seqlens_ptr), k_sf,
        /*row_stride=*/kTileM, num_group);
  }
}

template <typename GemmConfig, int kCtaPerSm = 1, bool kUsePDL = false>
void launch_group_gemm_1sm_mxfp8(void *y_ptr, const void *x_ptr, const void *w_ptr,
                                 const void *sfx_packed_ptr, const void *sfw_packed_ptr,
                                 const void *seqlens_ptr, const void *cu_seqlens_ptr,
                                 void *tmas_ptr, void *tiles_ptr, void *cu_tiles_ptr, int num_group,
                                 int m, int n, int k, bool update_tma, cudaStream_t stream) {
  using namespace cute;  // NOLINT

  using Tin = typename GemmConfig::Tin;
  using TinB = typename GemmConfig::TinB;
  using Tout = typename GemmConfig::Tout;
  using Tsf = typename GemmConfig::Tsf;
  constexpr int kTileM = GemmConfig::kTileM;
  constexpr int kTileN = GemmConfig::kTileN;
  constexpr int kSfxRows = GemmConfig::kSfxRows;
  constexpr int kSfVec = GemmConfig::kSfVec;

  int n_padded = ((n + 127) / 128) * 128;
  int k_sf = k / kSfVec;
  int k_sf_tiles = (k_sf + 3) / 4;

  auto A = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(x_ptr)), make_shape(m, k),
                       make_stride(k, Int<1>{}));
  auto B = make_tensor(make_gmem_ptr(reinterpret_cast<const TinB *>(w_ptr)),
                       make_shape(n, k, num_group), make_stride(k, Int<1>{}, n * k));
  auto Y = make_tensor(make_gmem_ptr(reinterpret_cast<Tout *>(y_ptr)), make_shape(n, m),
                       make_stride(Int<1>{}, n));

  // shape:  (kSfxRows,   16,    sfx_max_tiles,         k_sf_tiles    )
  // stride: (16,         1,     k_sf_tiles*kSfxRows*16, kSfxRows*16  )
  int sfx_max_tiles = m / kTileM + num_group;
  auto SFA =
      make_tensor(make_gmem_ptr(reinterpret_cast<const Tsf *>(sfx_packed_ptr)),
                  make_shape(Int<kSfxRows>{}, Int<16>{}, sfx_max_tiles, k_sf_tiles),
                  make_stride(Int<16>{}, Int<1>{}, k_sf_tiles * kSfxRows * 16, kSfxRows * 16));

  // shape:  (32,     16,     num_group * n_padded/128,    k_sf_tiles   )
  // stride: (16,     1,      k_sf_tiles*32*16,            32*16        )
  auto SFB = make_tensor(make_gmem_ptr(reinterpret_cast<const Tsf *>(sfw_packed_ptr)),
                         make_shape(Int<32>{}, Int<16>{}, num_group * (n_padded / 128), k_sf_tiles),
                         make_stride(Int<16>{}, Int<1>{}, k_sf_tiles * 32 * 16, 32 * 16));

  GemmConfig config;
  auto [tma_a, tma_b, tma_y, tma_sfa, tma_sfb] = config.get_tma(A, B, Y, SFA, SFB);

  auto *tma_ay = static_cast<cute::TmaDescriptor *>(tmas_ptr);
  int num_sm = get_sm_count();

  // 0. update tma
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

  // 1. group gemm
  {
    int num_tile_n_per_group = (n + kTileN - 1) / kTileN;
    cutlass::FastDivmod flat_divider(num_tile_n_per_group);

    // Extra smem: shm_tiles[num_group+1] + shm_cu_tiles[num_group+1]
    int shm_seq = 2 * sizeof(int) * (num_group + 1);
    int shm_size = config.get_shm_size() + shm_seq;

    dim3 block(256);
    dim3 grid(num_sm * kCtaPerSm);

    auto kernel =
        kernels::group_gemm_1sm_mxfp8_kernel<decltype(config), decltype(tma_a), decltype(tma_b),
                                             decltype(tma_y), decltype(tma_sfa), decltype(tma_sfb),
                                             kUsePDL>;
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

    cudaLaunchKernelEx(&launch_config, kernel, tma_b, tma_sfb, tma_sfa, tma_ay, (int *)seqlens_ptr,
                       (int *)tiles_ptr, (int *)cu_tiles_ptr, num_group, m, n, k, flat_divider);
  }
}

template <typename GemmConfig, int kCtaPerSm = 1, bool kUsePDL = false>
void launch_group_gemm_2sm_mxfp8(void *y_ptr, const void *x_ptr, const void *w_ptr,
                                 const void *sfx_packed_ptr, const void *sfw_packed_ptr,
                                 const void *seqlens_ptr, const void *cu_seqlens_ptr,
                                 void *tmas_ptr, void *tiles_ptr, void *cu_tiles_ptr, int num_group,
                                 int m, int n, int k, bool update_tma, cudaStream_t stream) {
  using namespace cute;  // NOLINT

  using Tin = typename GemmConfig::Tin;
  using TinB = typename GemmConfig::TinB;
  using Tout = typename GemmConfig::Tout;
  using Tsf = typename GemmConfig::Tsf;
  constexpr int kTileM = GemmConfig::kTileM;
  constexpr int kTileN = GemmConfig::kTileN;  // 256
  constexpr int kSfxRows = GemmConfig::kSfxRows;
  constexpr int kSfVec = GemmConfig::kSfVec;
  constexpr int kMmaSM = GemmConfig::kMmaSM;  // 2

  int n_padded = ((n + kTileN - 1) / kTileN) * kTileN;
  int k_sf = k / kSfVec;
  int k_sf_tiles = (k_sf + 3) / 4;

  auto A = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(x_ptr)), make_shape(m, k),
                       make_stride(k, Int<1>{}));
  auto B = make_tensor(make_gmem_ptr(reinterpret_cast<const TinB *>(w_ptr)),
                       make_shape(n, k, num_group), make_stride(k, Int<1>{}, n * k));
  auto Y = make_tensor(make_gmem_ptr(reinterpret_cast<Tout *>(y_ptr)), make_shape(n, m),
                       make_stride(Int<1>{}, n));

  int sfx_max_tiles = m / kTileM + num_group;
  auto SFA =
      make_tensor(make_gmem_ptr(reinterpret_cast<const Tsf *>(sfx_packed_ptr)),
                  make_shape(Int<kSfxRows>{}, Int<16>{}, sfx_max_tiles, k_sf_tiles),
                  make_stride(Int<16>{}, Int<1>{}, k_sf_tiles * kSfxRows * 16, kSfxRows * 16));

  auto SFB =
      make_tensor(make_gmem_ptr(reinterpret_cast<const Tsf *>(sfw_packed_ptr)),
                  make_shape(Int<64>{}, Int<16>{}, num_group * (n_padded / kTileN), k_sf_tiles),
                  make_stride(Int<16>{}, Int<1>{}, k_sf_tiles * 64 * 16, 64 * 16));

  GemmConfig config;
  auto [tma_a, tma_b, tma_y, tma_sfa, tma_sfb] = config.get_tma(A, B, Y, SFA, SFB);

  auto *tma_ay = static_cast<cute::TmaDescriptor *>(tmas_ptr);
  int num_sm = get_sm_count();
  num_sm = (num_sm / kMmaSM) * kMmaSM;  // round to even (each cluster uses 2 SMs)

  // 0. update tma
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

  // 1. group gemm (2SM cluster)
  {
    int num_tile_n_per_group = (n + kTileN - 1) / kTileN;
    cutlass::FastDivmod flat_divider(num_tile_n_per_group);

    int shm_seq = 2 * sizeof(int) * (num_group + 1);
    int shm_size = config.get_shm_size() + shm_seq;

    dim3 block(256);
    dim3 grid(num_sm * kCtaPerSm);

    auto kernel =
        kernels::group_gemm_2sm_mxfp8_kernel<decltype(config), decltype(tma_a), decltype(tma_b),
                                             decltype(tma_y), decltype(tma_sfa), decltype(tma_sfb),
                                             kUsePDL>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

    cudaLaunchConfig_t launch_config;
    memset(&launch_config, 0, sizeof(launch_config));
    launch_config.gridDim = grid;
    launch_config.blockDim = block;
    launch_config.dynamicSmemBytes = shm_size;
    launch_config.stream = stream;
    cudaLaunchAttribute attr[2];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim = {static_cast<unsigned>(kMmaSM), 1, 1};
    int num_attrs = 1;
    if constexpr (kUsePDL) {
      attr[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attr[1].val.programmaticStreamSerializationAllowed = 1;
      num_attrs = 2;
    }
    launch_config.numAttrs = num_attrs;
    launch_config.attrs = attr;

    cudaLaunchKernelEx(&launch_config, kernel, tma_b, tma_sfb, tma_sfa, tma_ay, (int *)seqlens_ptr,
                       (int *)tiles_ptr, (int *)cu_tiles_ptr, num_group, m, n, k, flat_divider);
  }
}

namespace {
template <typename T>
struct type_id {
  using type = T;
};
}  // namespace

void group_gemm_mxfp8_async(void *y_ptr, const void *x_ptr, const void *w_ptr,
                            const void *sfx_packed_ptr, const void *sfw_packed_ptr,
                            const void *seqlens_ptr, const void *cu_seqlens_ptr, void *tmas_ptr,
                            void *tiles_ptr, void *cu_tiles_ptr, int num_group, int m, int n, int k,
                            int num_seq_per_group_avg, bool update_tma, cudaStream_t stream,
                            bool use_pdl, bool is_fp4) {
  int kTileM_dispatch = mxfp8_dispatch_kTileM(num_seq_per_group_avg, n);
  bool use_2sm = (n % 256 == 0) && (num_seq_per_group_avg > 32);

  using Tin = cute::float_e4m3_t;
  using TinB_fp4 = cutlass::detail::float_e2m1_unpacksmem_t;
  using Tout = cute::bfloat16_t;
  using Tsf = cutlass::float_ue8m0_t;

  auto launch_1sm = [&](auto cfg_tag, auto cta_tag) {
    using Cfg = typename decltype(cfg_tag)::type;
    constexpr int kCtaPerSm = decltype(cta_tag)::value;
    if (use_pdl) {
      launch_group_gemm_1sm_mxfp8<Cfg, kCtaPerSm, true>(
          y_ptr, x_ptr, w_ptr, sfx_packed_ptr, sfw_packed_ptr, seqlens_ptr, cu_seqlens_ptr,
          tmas_ptr, tiles_ptr, cu_tiles_ptr, num_group, m, n, k, update_tma, stream);
    } else {
      launch_group_gemm_1sm_mxfp8<Cfg, kCtaPerSm, false>(
          y_ptr, x_ptr, w_ptr, sfx_packed_ptr, sfw_packed_ptr, seqlens_ptr, cu_seqlens_ptr,
          tmas_ptr, tiles_ptr, cu_tiles_ptr, num_group, m, n, k, update_tma, stream);
    }
  };

  auto launch_2sm = [&](auto cfg_tag) {
    using Cfg = typename decltype(cfg_tag)::type;
    if (use_pdl) {
      launch_group_gemm_2sm_mxfp8<Cfg, 1, true>(
          y_ptr, x_ptr, w_ptr, sfx_packed_ptr, sfw_packed_ptr, seqlens_ptr, cu_seqlens_ptr,
          tmas_ptr, tiles_ptr, cu_tiles_ptr, num_group, m, n, k, update_tma, stream);
    } else {
      launch_group_gemm_2sm_mxfp8<Cfg, 1, false>(
          y_ptr, x_ptr, w_ptr, sfx_packed_ptr, sfw_packed_ptr, seqlens_ptr, cu_seqlens_ptr,
          tmas_ptr, tiles_ptr, cu_tiles_ptr, num_group, m, n, k, update_tma, stream);
    }
  };

  using IC1 = std::integral_constant<int, 1>;
  using IC3 = std::integral_constant<int, 3>;

  assert(k % 32 == 0 && "group_gemm_mxfp8: k must be a multiple of 32 (SF_VEC)");

  constexpr int kTileK = 128;

  // 1SM dispatch table, parameterized by TinB (TinB=Tin for fp8, TinB=float_e2m1_unpacksmem_t
  //   for fp4). The Config's 12th template parameter = TinB.
  auto dispatch_1sm = [&](auto tinb_tag) {
    using TinB = typename decltype(tinb_tag)::type;
    constexpr int kMmaSM = 1;
    constexpr int kTileN = 128;
    switch (kTileM_dispatch) {
      // Tin, Tout, Tsf, KTM, kTileN, kTileK, EPI, STAGE, STAGE_TMA, kMmaSM, STAGE_TILE, TinB
      case 16:
        return launch_1sm(type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 16, kTileN, kTileK, 16, 2, 4,
                                                       kMmaSM, 4, TinB>>{},
                          IC3{});
      case 32:
        return launch_1sm(type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 32, kTileN, kTileK, 32, 2, 4,
                                                       kMmaSM, 4, TinB>>{},
                          IC3{});
      case 48:
        return launch_1sm(type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 48, kTileN, kTileK, 16, 6, 4,
                                                       kMmaSM, 4, TinB>>{},
                          IC1{});
      case 64:
        return launch_1sm(type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 64, kTileN, kTileK, 64, 6, 4,
                                                       kMmaSM, 4, TinB>>{},
                          IC1{});
      case 128:
        return launch_1sm(type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 128, kTileN, kTileK, 64, 5,
                                                       3, kMmaSM, 3, TinB>>{},
                          IC1{});
      default:
        return launch_1sm(type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 256, kTileN, kTileK, 64, 3,
                                                       2, kMmaSM, 1, TinB>>{},
                          IC1{});
    }
  };

  if (use_2sm) {
    // 2SM, kTileK=128 (TinB=Tin for fp8, TinB=float_e2m1_unpacksmem_t for fp4).
    //   The Config's 12th template parameter = TinB.
    auto dispatch_2sm = [&](auto tinb_tag) {
      using TinB = typename decltype(tinb_tag)::type;
      constexpr int kMmaSM = 2;
      constexpr int kTileN = 256;
      switch (kTileM_dispatch) {
        // Tin, Tout, Tsf, KTM, kTileN, kTileK, EPI, STAGE, STAGE_TMA, kMmaSM, STAGE_TILE, TinB
        case 32:
          return launch_2sm(type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 32, kTileN, kTileK, 32, 6,
                                                         4, kMmaSM, 4, TinB>>{});
        case 64:
          return launch_2sm(type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 64, kTileN, kTileK, 32, 6,
                                                         4, kMmaSM, 4, TinB>>{});
        case 96:
          return launch_2sm(type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 96, kTileN, kTileK, 32, 6,
                                                         4, kMmaSM, 4, TinB>>{});
        case 128:
          return launch_2sm(type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 128, kTileN, kTileK, 32, 6,
                                                         4, kMmaSM, 3, TinB>>{});
        case 160:
          return launch_2sm(type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 160, kTileN, kTileK, 32, 6,
                                                         4, kMmaSM, 2, TinB>>{});
        case 192:
          return launch_2sm(type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 192, kTileN, kTileK, 32, 6,
                                                         2, kMmaSM, 2, TinB>>{});
        default:
          return launch_2sm(type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 256, kTileN, kTileK, 64, 5,
                                                         2, kMmaSM, 1, TinB>>{});
      }
    };

    if (is_fp4) {
      return dispatch_2sm(type_id<TinB_fp4>{});
    } else {
      return dispatch_2sm(type_id<Tin>{});
    }
  } else {
    if (is_fp4) {
      return dispatch_1sm(type_id<TinB_fp4>{});
    } else {
      return dispatch_1sm(type_id<Tin>{});
    }
  }
}

}  // namespace group_gemm
}  // namespace hpc
