// Copyright 2026 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include <cub/cub.cuh>
#include <type_traits>

#include "cute/tensor.hpp"
#include "src/group_gemm/sm100/gemm_config.h"
#include "src/group_gemm/sm100/group_gemm.h"
#include "src/group_gemm/sm100/mxfp8/dispatch.cuh"
#include "src/group_gemm/sm100/mxfp8/group_gemm_mxfp8_with_reduce.cuh"

namespace hpc {
namespace group_gemm {

template <typename GemmConfig, int kCtaPerSm = 1>
void launch_group_gemm_1sm_mxfp8_with_reduce(void *y_ptr, const void *x_ptr, const void *w_ptr,
                                             const void *sfx_packed_ptr, const void *sfw_packed_ptr,
                                             const void *seqlens_ptr, const void *cu_seqlens_ptr,
                                             void *tmas_ptr, void *tiles_ptr, void *cu_tiles_ptr,
                                             void *x_row_map_ptr, void *topk_scale_row_map_ptr,
                                             int num_group, int m, int n, int k, bool update_tma,
                                             cudaStream_t stream) {
  using namespace cute;  // NOLINT

  using Tin = typename GemmConfig::Tin;
  using TinB = typename GemmConfig::TinB;
  using Tout = typename GemmConfig::Tout;
  using Tsf = typename GemmConfig::Tsf;
  constexpr int kTileM = GemmConfig::kTileM;  // 1
  constexpr int kTileN = GemmConfig::kTileN;  // 128
  constexpr int kSfVec = GemmConfig::kSfVec;  // 32

  int n_padded = ((n + 127) / 128) * 128;
  int k_sf = k / kSfVec;
  int k_sf_tiles = (k_sf + 3) / 4;

  auto A = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(x_ptr)), make_shape(m, k),
                       make_stride(k, Int<1>{}));
  auto B = make_tensor(make_gmem_ptr(reinterpret_cast<const TinB *>(w_ptr)),
                       make_shape(n, k, num_group), make_stride(k, Int<1>{}, n * k));
  auto Y = make_tensor(make_gmem_ptr(reinterpret_cast<Tout *>(y_ptr)), make_shape(n, m),
                       make_stride(Int<1>{}, n));

  auto SFB = make_tensor(make_gmem_ptr(reinterpret_cast<const Tsf *>(sfw_packed_ptr)),
                         make_shape(Int<32>{}, Int<16>{}, num_group * (n_padded / 128), k_sf_tiles),
                         make_stride(Int<16>{}, Int<1>{}, k_sf_tiles * 32 * 16, 32 * 16));

  GemmConfig config;
  auto [tma_a, tma_b, tma_y, tma_sfb] = config.get_tma_without_sfa(A, B, Y, SFB);

  auto *tma_ay = static_cast<cute::TmaDescriptor *>(tmas_ptr);
  int num_sm = get_sm_count();

  // 0. update tma (only A descriptor needed for reduce path, but we reuse the full update kernel)
  if (update_tma) {
    vec_t<cute::TmaDescriptor, 2> td_ay{
        *tma_a.get_tma_descriptor(),
        *tma_y.get_tma_descriptor(),  // Y TMA desc unused by reduce kernel but update is cheap
    };

    constexpr int kGroupPerThread = 8;
    constexpr int kThreadPerBlock = 32;
    auto kernel = kernels::update_grouped_tma_mxfp8<Tin, Tout, decltype(tma_a), decltype(tma_y),
                                                    kTileM, kGroupPerThread, kThreadPerBlock>;
    kernel<<<num_group + 1, kThreadPerBlock, 0, stream>>>(
        td_ay, tma_ay, (const Tin *)x_ptr, (const Tout *)y_ptr, (const int *)seqlens_ptr,
        (const int *)cu_seqlens_ptr, (int *)tiles_ptr, (int *)cu_tiles_ptr, num_group, m, n, k);
  }

  // 1. group gemm with reduce
  {
    int num_tile_n_per_group = (n + kTileN - 1) / kTileN;
    cutlass::FastDivmod flat_divider(num_tile_n_per_group);

    // Extra smem: shm_tiles[num_group+1] + shm_cu_tiles[num_group+1]
    int shm_seq = 2 * sizeof(int) * (num_group + 1);
    int shm_size = config.get_shm_size() + shm_seq;

    dim3 block(256);
    dim3 grid(num_sm * kCtaPerSm);

    auto kernel =
        kernels::group_gemm_1sm_mxfp8_with_reduce_kernel<decltype(config), decltype(tma_a),
                                                         decltype(tma_b), decltype(tma_sfb), true>;
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
    attr[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[1].val.programmaticStreamSerializationAllowed = 1;
    launch_config.numAttrs = 2;
    launch_config.attrs = attr;

    cudaLaunchKernelEx(&launch_config, kernel, tma_b, tma_sfb, tma_ay, (Tout *)y_ptr,
                       (const uint8_t *)sfx_packed_ptr, (int *)seqlens_ptr, (int *)cu_seqlens_ptr,
                       (int *)tiles_ptr, (int *)cu_tiles_ptr, (int *)x_row_map_ptr,
                       (float *)topk_scale_row_map_ptr, num_group, m, n, k, flat_divider);
  }
}

template <typename GemmConfig, int kCtaPerSm = 1>
void launch_group_gemm_2sm_mxfp8_with_reduce(void *y_ptr, const void *x_ptr, const void *w_ptr,
                                             const void *sfx_packed_ptr, const void *sfw_packed_ptr,
                                             const void *seqlens_ptr, const void *cu_seqlens_ptr,
                                             void *tmas_ptr, void *tiles_ptr, void *cu_tiles_ptr,
                                             void *x_row_map_ptr, void *topk_scale_row_map_ptr,
                                             int num_group, int m, int n, int k, bool update_tma,
                                             cudaStream_t stream) {
  using namespace cute;  // NOLINT

  using Tin = typename GemmConfig::Tin;
  using TinB = typename GemmConfig::TinB;
  using Tout = typename GemmConfig::Tout;
  using Tsf = typename GemmConfig::Tsf;
  constexpr int kTileM = GemmConfig::kTileM;
  constexpr int kTileN = GemmConfig::kTileN;  // 256
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

  auto SFB =
      make_tensor(make_gmem_ptr(reinterpret_cast<const Tsf *>(sfw_packed_ptr)),
                  make_shape(Int<64>{}, Int<16>{}, num_group * (n_padded / kTileN), k_sf_tiles),
                  make_stride(Int<16>{}, Int<1>{}, k_sf_tiles * 64 * 16, 64 * 16));

  GemmConfig config;
  auto [tma_a, tma_b, tma_y, tma_sfb] = config.get_tma_without_sfa(A, B, Y, SFB);

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

  // 1. group gemm with reduce (2SM cluster)
  {
    int num_tile_n_per_group = (n + kTileN - 1) / kTileN;
    cutlass::FastDivmod flat_divider(num_tile_n_per_group);

    int shm_seq = 2 * sizeof(int) * (num_group + 1);
    int shm_size = config.get_shm_size() + shm_seq;

    dim3 block(256);
    dim3 grid(num_sm * kCtaPerSm);

    auto kernel =
        kernels::group_gemm_2sm_mxfp8_with_reduce_kernel<decltype(config), decltype(tma_a),
                                                         decltype(tma_b), decltype(tma_sfb), true>;
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
    attr[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[1].val.programmaticStreamSerializationAllowed = 1;
    launch_config.numAttrs = 2;
    launch_config.attrs = attr;

    cudaLaunchKernelEx(&launch_config, kernel, tma_b, tma_sfb, tma_ay, (Tout *)y_ptr,
                       (const uint8_t *)sfx_packed_ptr, (int *)seqlens_ptr, (int *)cu_seqlens_ptr,
                       (int *)tiles_ptr, (int *)cu_tiles_ptr, (int *)x_row_map_ptr,
                       (float *)topk_scale_row_map_ptr, num_group, m, n, k, flat_divider);
  }
}

void group_gemm_mxfp8_with_reduce_async(void *y_ptr, const void *x_ptr, const void *w_ptr,
                                        const void *sfx_packed_ptr, const void *sfw_packed_ptr,
                                        const void *seqlens_ptr, const void *cu_seqlens_ptr,
                                        void *tmas_ptr, void *tiles_ptr, void *cu_tiles_ptr,
                                        void *x_row_map_ptr, void *topk_scale_row_map_ptr,
                                        int num_group, int m, int n, int k,
                                        int num_seq_per_group_avg, bool update_tma,
                                        cudaStream_t stream, bool is_fp4) {
  int kTileM_dispatch = mxfp8_dispatch_kTileM(num_seq_per_group_avg);

  using Tin = cute::float_e4m3_t;
  using TinB_fp4 = cutlass::detail::float_e2m1_unpacksmem_t;
  using Tout = cute::bfloat16_t;
  using Tsf = cutlass::float_ue8m0_t;

  assert(k % 32 == 0 && "group_gemm_mxfp8_with_reduce: k must be a multiple of 32 (SF_VEC)");

  auto launch_1sm = [&](auto cfg_tag) {
    using Cfg = typename decltype(cfg_tag)::type;
    launch_group_gemm_1sm_mxfp8_with_reduce<Cfg, 1>(
        y_ptr, x_ptr, w_ptr, sfx_packed_ptr, sfw_packed_ptr, seqlens_ptr, cu_seqlens_ptr, tmas_ptr,
        tiles_ptr, cu_tiles_ptr, x_row_map_ptr, topk_scale_row_map_ptr, num_group, m, n, k,
        update_tma, stream);
  };

  // 1SM dispatch via the shared config selector (kTileN=128, kMmaSM=1). The down
  // GEMM always runs 1SM: it avoids the 2-CTA SFA cp.async prepack path entirely
  // (single CTA, the cp.async noinc arrive is consumed by the same CTA's MMA
  // warp). The 2SM launcher/kernel remain available
  // (launch_group_gemm_2sm_mxfp8_with_reduce) for perf experiments but are not on
  // the default path. SFW is prepacked once (is_2sm layout) and is read correctly
  // by both 1SM and 2SM SFB TMA views. The config table lives in dispatch.cuh so
  // fuse_moe stage-1 builds the down-X descriptor with the matching 1SM box.
  if (is_fp4) {
    return group_gemm_1sm_mxfp8_dispatch_selector<Tin, Tout, Tsf, TinB_fp4>(kTileM_dispatch,
                                                                            launch_1sm);
  } else {
    return group_gemm_1sm_mxfp8_dispatch_selector<Tin, Tout, Tsf, Tin>(kTileM_dispatch, launch_1sm);
  }
}

}  // namespace group_gemm
}  // namespace hpc
