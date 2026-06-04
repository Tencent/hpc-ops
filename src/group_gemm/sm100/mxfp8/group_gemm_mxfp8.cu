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
  auto B = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(w_ptr)),
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
  auto B = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(w_ptr)),
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
                            bool use_pdl) {
  int kTileM_dispatch = mxfp8_dispatch_kTileM(num_seq_per_group_avg, n);
  bool use_2sm = (n % 256 == 0) && (num_seq_per_group_avg > 32);

  using Tin = cute::float_e4m3_t;
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

  if (use_2sm) {
    if (k % 128 == 0) {
      // 2SM, kTileK=128
      switch (kTileM_dispatch) {
        //        <Tin, Tout, Tsf, KTM, kTileN, KTK, EPI, STAGE, STAGE_TMA, kMmaSM, STAGE_TILE>
        case 32:
          return launch_2sm(
              type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 32, 256, 128, 32, 6, 4, 2, 4>>{});
        case 64:
          return launch_2sm(
              type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 64, 256, 128, 32, 6, 4, 2, 4>>{});
        case 96:
          return launch_2sm(
              type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 96, 256, 128, 32, 6, 4, 2, 4>>{});
        case 128:
          return launch_2sm(
              type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 128, 256, 128, 32, 6, 4, 2, 3>>{});
        case 160:
          return launch_2sm(
              type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 160, 256, 128, 32, 6, 4, 2, 2>>{});
        case 192:
          return launch_2sm(
              type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 192, 256, 128, 32, 6, 2, 2, 2>>{});
        default:
          return launch_2sm(
              type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 256, 256, 128, 64, 5, 2, 2, 1>>{});
      }
    } else {
      // 2SM, kTileK=64
      switch (kTileM_dispatch) {
        case 32:
          return launch_2sm(
              type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 32, 256, 64, 32, 10, 4, 2, 4>>{});
        case 64:
          return launch_2sm(
              type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 64, 256, 64, 32, 10, 4, 2, 4>>{});
        case 96:
          return launch_2sm(
              type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 96, 256, 64, 32, 10, 4, 2, 4>>{});
        case 128:
          return launch_2sm(
              type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 128, 256, 64, 32, 8, 4, 2, 3>>{});
        case 160:
          return launch_2sm(
              type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 160, 256, 64, 32, 8, 4, 2, 2>>{});
        case 192:
          return launch_2sm(
              type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 192, 256, 64, 32, 8, 2, 2, 2>>{});
        default:
          return launch_2sm(
              type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 256, 256, 64, 64, 8, 2, 2, 1>>{});
      }
    }
  } else {
    if (k % 128 == 0) {
      // 1SM, kTileK=128
      switch (kTileM_dispatch) {
        case 16:
          return launch_1sm(
              type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 16, 128, 128, 16, 2, 4, 1, 4>>{}, IC3{});
        case 32:
          return launch_1sm(
              type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 32, 128, 128, 32, 2, 4, 1, 4>>{}, IC3{});
        case 48:
          return launch_1sm(
              type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 48, 128, 128, 16, 6, 4, 1, 4>>{}, IC1{});
        case 64:
          return launch_1sm(
              type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 64, 128, 128, 64, 6, 4, 1, 4>>{}, IC1{});
        case 128:
          return launch_1sm(
              type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 128, 128, 128, 64, 5, 3, 1, 3>>{},
              IC1{});
        default:
          return launch_1sm(
              type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 256, 128, 128, 64, 3, 2, 1, 1>>{},
              IC1{});
      }
    } else {
      // 1SM, kTileK=64
      switch (kTileM_dispatch) {
        case 16:
          return launch_1sm(
              type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 16, 128, 64, 16, 4, 4, 1, 4>>{}, IC3{});
        case 32:
          return launch_1sm(
              type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 32, 128, 64, 32, 4, 4, 1, 4>>{}, IC3{});
        case 48:
          return launch_1sm(
              type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 48, 128, 64, 16, 10, 4, 1, 4>>{}, IC1{});
        case 64:
          return launch_1sm(
              type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 64, 128, 64, 64, 10, 4, 1, 4>>{}, IC1{});
        case 128:
          return launch_1sm(
              type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 128, 128, 64, 64, 8, 3, 1, 3>>{}, IC1{});
        default:
          return launch_1sm(
              type_id<GroupGEMMMxFp8Config<Tin, Tout, Tsf, 256, 128, 64, 64, 6, 2, 1, 1>>{}, IC1{});
      }
    }
  }
}

}  // namespace group_gemm
}  // namespace hpc
