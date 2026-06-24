// Copyright 2026 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include "cute/tensor.hpp"
#include "src/group_gemm/sm100/group_gemm.h"
#include "src/group_gemm/sm100/mxfp8/config.h"
#include "src/group_gemm/sm100/mxfp8/dispatch.cuh"
#include "src/group_gemm/sm100/mxfp8/group_gemm_cp_async_mxfp8_act_mul.cuh"
#include "src/group_gemm/sm100/mxfp8/group_gemm_mxfp8.cuh"

namespace hpc {
namespace group_gemm {

template <typename GemmConfig>
void launch_group_gemm_1sm_cp_async_mxfp8_act_mul(
    void *y_ptr, void *y_fp8_ptr, const void *x_ptr, const void *w_ptr, const void *sfx_ptr,
    const void *sfw_packed_ptr, const void *seqlens_ptr, const void *cu_seqlens_ptr, void *tmas_ptr,
    void *tiles_ptr, void *cu_tiles_ptr, void *out_scale_ptr, int num_group, int m, int n, int k,
    bool update_tma, cudaStream_t stream, const void *x_row_map_ptr, int x_num_rows) {
  using namespace cute;  // NOLINT

  using Tin = typename GemmConfig::Tin;
  using TinB = typename GemmConfig::TinB;
  using Tout = typename GemmConfig::Tout;
  using Tsf = typename GemmConfig::Tsf;
  constexpr int kTileM = GemmConfig::kTileM;
  constexpr int kTileN = GemmConfig::kTileN;  // 128
  constexpr int kSfVec = GemmConfig::kSfVec;
  constexpr int kMmaSM = GemmConfig::kMmaSM;

  static_assert(kMmaSM == 1, "act_mul mxfp8 is 1SM only");

  int n_half = n / 2;
  int n_padded = ((n + 127) / 128) * 128;
  int k_sf = k / kSfVec;
  int k_sf_tiles = (k_sf + 3) / 4;

  // Weight B: (N=interleaved, K, num_group)
  auto B = make_tensor(make_gmem_ptr(reinterpret_cast<const TinB *>(w_ptr)),
                       make_shape(n, k, num_group), make_stride(k, Int<1>{}, n * k));
  // SFB
  auto SFB = make_tensor(make_gmem_ptr(reinterpret_cast<const Tsf *>(sfw_packed_ptr)),
                         make_shape(Int<32>{}, Int<16>{}, num_group * (n_padded / 128), k_sf_tiles),
                         make_stride(Int<16>{}, Int<1>{}, k_sf_tiles * 32 * 16, 32 * 16));

  GemmConfig config;
  // only tma_b and tma_sfb are used
  auto [tma_a_unused, tma_b, tma_y_unused, tma_sfb] = config.get_tma_without_sfa(
      make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(x_ptr)), make_shape(m, k),
                  make_stride(k, Int<1>{})),
      B,
      make_tensor(make_gmem_ptr(reinterpret_cast<cute::bfloat16_t *>(y_ptr)), make_shape(n_half, m),
                  make_stride(Int<1>{}, n_half)),
      SFB);

  auto *tma_ay_unused = static_cast<cute::TmaDescriptor *>(tmas_ptr);
  int num_sm = get_sm_count();

  if (update_tma) {
    vec_t<cute::TmaDescriptor, 2> td_ay_unused{
        *tma_a_unused.get_tma_descriptor(),
        *tma_y_unused.get_tma_descriptor(),
    };

    constexpr int kGroupPerThread = 8;
    constexpr int kThreadPerBlock = 32;
    auto kernel =
        kernels::update_grouped_tma_mxfp8<Tin, Tout, decltype(tma_a_unused), decltype(tma_y_unused),
                                          kTileM, kGroupPerThread, kThreadPerBlock>;
    kernel<<<num_group + 1, kThreadPerBlock, 0, stream>>>(
        td_ay_unused, tma_ay_unused, (const Tin *)x_ptr, (const Tout *)y_ptr,
        (const int *)seqlens_ptr, (const int *)cu_seqlens_ptr, (int *)tiles_ptr,
        (int *)cu_tiles_ptr, num_group, m, n, k);
  }

  // Launch fused kernel
  {
    int num_tile_n_per_group = (n + kTileN - 1) / kTileN;
    cutlass::FastDivmod flat_divider(num_tile_n_per_group);

    int shm_seq = 2 * sizeof(int) * (num_group + 1);
    int shm_size = config.get_shm_size() + shm_seq;

    dim3 block(384);
    dim3 grid(num_sm);

    auto kernel =
        kernels::group_gemm_1sm_cp_async_mxfp8_act_mul_kernel<decltype(config), decltype(tma_b),
                                                              decltype(tma_sfb), true>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

    cudaLaunchConfig_t launch_config;
    memset(&launch_config, 0, sizeof(launch_config));
    launch_config.gridDim = grid;
    launch_config.blockDim = block;
    launch_config.dynamicSmemBytes = shm_size;
    launch_config.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    launch_config.numAttrs = 1;
    launch_config.attrs = attr;

    cudaLaunchKernelEx(
        &launch_config, kernel, tma_b, tma_sfb, reinterpret_cast<const Tin *>(x_ptr),
        reinterpret_cast<const uint8_t *>(sfx_ptr), reinterpret_cast<const int *>(x_row_map_ptr),
        x_num_rows, reinterpret_cast<Tout *>(y_ptr), reinterpret_cast<__nv_fp8_e4m3 *>(y_fp8_ptr),
        reinterpret_cast<uint8_t *>(out_scale_ptr), (int *)seqlens_ptr, (int *)cu_seqlens_ptr,
        (int *)tiles_ptr, (int *)cu_tiles_ptr, num_group, m, n, k, flat_divider);
  }
}

namespace {
template <typename T>
struct type_id_am {
  using type = T;
};
}  // namespace

void group_gemm_cp_async_mxfp8_act_mul_async(
    void *y_ptr, void *y_fp8_ptr, const void *x_ptr, const void *w_ptr, const void *sfx_ptr,
    const void *sfw_packed_ptr, const void *seqlens_ptr, const void *cu_seqlens_ptr, void *tmas_ptr,
    void *tiles_ptr, void *cu_tiles_ptr, void *out_scale_ptr, int num_group, int m, int n, int k,
    int num_seq_per_group_avg, bool update_tma, cudaStream_t stream, const void *x_row_map_ptr,
    int x_num_rows, bool is_fp4) {
  using Tin = cute::float_e4m3_t;
  using TinB_fp4 = cutlass::detail::float_e2m1_unpacksmem_t;
  using Tout = cute::bfloat16_t;  // keep bf16 for config (SLayoutYT not used by fused kernel)
  using Tsf = cutlass::float_ue8m0_t;

  int kTileM_dispatch = mxfp8_dispatch_kTileM(num_seq_per_group_avg);

  auto dispatch = [&](auto tinb_tag) {
    using TinB = typename decltype(tinb_tag)::type;
    constexpr int kTileN = 128;
    constexpr int kTileK = 128;
    constexpr int kMmaSM = 1;
    switch (kTileM_dispatch) {
      case 16:
        return launch_group_gemm_1sm_cp_async_mxfp8_act_mul<
            GroupGEMMMxFp8Config<Tin, Tout, Tsf, 16, kTileN, kTileK, 16, 8, 4, kMmaSM, 4, TinB>>(
            y_ptr, y_fp8_ptr, x_ptr, w_ptr, sfx_ptr, sfw_packed_ptr, seqlens_ptr, cu_seqlens_ptr,
            tmas_ptr, tiles_ptr, cu_tiles_ptr, out_scale_ptr, num_group, m, n, k, update_tma,
            stream, x_row_map_ptr, x_num_rows);
      case 64:
        return launch_group_gemm_1sm_cp_async_mxfp8_act_mul<
            GroupGEMMMxFp8Config<Tin, Tout, Tsf, 64, kTileN, kTileK, 64, 6, 4, kMmaSM, 4, TinB>>(
            y_ptr, y_fp8_ptr, x_ptr, w_ptr, sfx_ptr, sfw_packed_ptr, seqlens_ptr, cu_seqlens_ptr,
            tmas_ptr, tiles_ptr, cu_tiles_ptr, out_scale_ptr, num_group, m, n, k, update_tma,
            stream, x_row_map_ptr, x_num_rows);
      case 96:
        return launch_group_gemm_1sm_cp_async_mxfp8_act_mul<
            GroupGEMMMxFp8Config<Tin, Tout, Tsf, 96, kTileN, kTileK, 32, 4, 4, kMmaSM, 4, TinB>>(
            y_ptr, y_fp8_ptr, x_ptr, w_ptr, sfx_ptr, sfw_packed_ptr, seqlens_ptr, cu_seqlens_ptr,
            tmas_ptr, tiles_ptr, cu_tiles_ptr, out_scale_ptr, num_group, m, n, k, update_tma,
            stream, x_row_map_ptr, x_num_rows);
      case 128:
        return launch_group_gemm_1sm_cp_async_mxfp8_act_mul<
            GroupGEMMMxFp8Config<Tin, Tout, Tsf, 128, kTileN, kTileK, 64, 3, 3, kMmaSM, 3, TinB>>(
            y_ptr, y_fp8_ptr, x_ptr, w_ptr, sfx_ptr, sfw_packed_ptr, seqlens_ptr, cu_seqlens_ptr,
            tmas_ptr, tiles_ptr, cu_tiles_ptr, out_scale_ptr, num_group, m, n, k, update_tma,
            stream, x_row_map_ptr, x_num_rows);
      case 160:
        return launch_group_gemm_1sm_cp_async_mxfp8_act_mul<
            GroupGEMMMxFp8Config<Tin, Tout, Tsf, 160, kTileN, kTileK, 32, 3, 2, kMmaSM, 2, TinB>>(
            y_ptr, y_fp8_ptr, x_ptr, w_ptr, sfx_ptr, sfw_packed_ptr, seqlens_ptr, cu_seqlens_ptr,
            tmas_ptr, tiles_ptr, cu_tiles_ptr, out_scale_ptr, num_group, m, n, k, update_tma,
            stream, x_row_map_ptr, x_num_rows);
      case 192:
        return launch_group_gemm_1sm_cp_async_mxfp8_act_mul<
            GroupGEMMMxFp8Config<Tin, Tout, Tsf, 192, kTileN, kTileK, 32, 3, 2, kMmaSM, 2, TinB>>(
            y_ptr, y_fp8_ptr, x_ptr, w_ptr, sfx_ptr, sfw_packed_ptr, seqlens_ptr, cu_seqlens_ptr,
            tmas_ptr, tiles_ptr, cu_tiles_ptr, out_scale_ptr, num_group, m, n, k, update_tma,
            stream, x_row_map_ptr, x_num_rows);
      default:
        return launch_group_gemm_1sm_cp_async_mxfp8_act_mul<
            GroupGEMMMxFp8Config<Tin, Tout, Tsf, 256, kTileN, kTileK, 64, 3, 2, kMmaSM, 1, TinB>>(
            y_ptr, y_fp8_ptr, x_ptr, w_ptr, sfx_ptr, sfw_packed_ptr, seqlens_ptr, cu_seqlens_ptr,
            tmas_ptr, tiles_ptr, cu_tiles_ptr, out_scale_ptr, num_group, m, n, k, update_tma,
            stream, x_row_map_ptr, x_num_rows);
    }
  };

  if (is_fp4) {
    dispatch(type_id_am<TinB_fp4>{});
  } else {
    dispatch(type_id_am<Tin>{});
  }
}

}  // namespace group_gemm
}  // namespace hpc
