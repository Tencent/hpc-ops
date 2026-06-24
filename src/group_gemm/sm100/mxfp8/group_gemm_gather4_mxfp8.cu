// Copyright 2026 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include <cub/cub.cuh>

#include "cute/tensor.hpp"
#include "src/group_gemm/sm100/gemm_config.h"
#include "src/group_gemm/sm100/group_gemm.h"
#include "src/group_gemm/sm100/mxfp8/dispatch.cuh"
#include "src/group_gemm/sm100/mxfp8/group_gemm_gather4_mxfp8.cuh"
#include "src/group_gemm/sm100/mxfp8/group_gemm_mxfp8.cuh"

namespace hpc {
namespace group_gemm {

// ============================================================================
// 2SM gather4 launch wrapper (PDL always enabled)
// ============================================================================
template <typename GemmConfig>
void launch_group_gemm_2sm_gather4_mxfp8(void *y_ptr, const void *x_ptr, const void *w_ptr,
                                         const void *sfx_ptr, const void *sfw_packed_ptr,
                                         const void *seqlens_ptr, const void *cu_seqlens_ptr,
                                         void *tmas_ptr, void *tiles_ptr, void *cu_tiles_ptr,
                                         int num_group, int m, int n, int k, bool update_tma,
                                         cudaStream_t stream, const void *x_row_map_ptr,
                                         int x_num_rows) {
  using namespace cute;  // NOLINT

  using Tin = typename GemmConfig::Tin;
  using TinB = typename GemmConfig::TinB;
  using Tout = typename GemmConfig::Tout;
  using Tsf = typename GemmConfig::Tsf;
  constexpr int kTileM = GemmConfig::kTileM;
  constexpr int kTileN = GemmConfig::kTileN;
  constexpr int kTileK = GemmConfig::kTileK;
  constexpr int kSfVec = GemmConfig::kSfVec;
  constexpr int kMmaSM = GemmConfig::kMmaSM;

  static_assert(kMmaSM == 2, "gather4 async dispatch is 2SM only");

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
  num_sm = (num_sm / kMmaSM) * kMmaSM;

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

  // Create gather4 TensorMap for A
  int a_num_rows = (x_row_map_ptr != nullptr) ? x_num_rows : m;
  CUtensorMap tma_a_g4{};
  {
    constexpr uint32_t rank = 2;
    uint64_t sizes[rank] = {(uint64_t)k, (uint64_t)a_num_rows};
    uint64_t strides[rank - 1] = {(uint64_t)(k * sizeof(Tin))};
    uint32_t box_sizes[rank] = {(uint32_t)kTileK, 1};
    uint32_t elem_strides[rank] = {1, 1};
    auto swizzle = (kTileK == 128) ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_64B;
    cuTensorMapEncodeTiled(&tma_a_g4, CU_TENSOR_MAP_DATA_TYPE_UINT8, rank,
                           const_cast<void *>(x_ptr), sizes, strides, box_sizes, elem_strides,
                           CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                           CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  }

  // Launch 2SM gather4 kernel (PDL always on)
  {
    int num_tile_n_per_group = (n + kTileN - 1) / kTileN;
    cutlass::FastDivmod flat_divider(num_tile_n_per_group);

    int shm_seq = 4 * sizeof(int) * (num_group + 1);
    int shm_size = config.get_shm_size() + shm_seq;

    dim3 block(256);
    dim3 grid(num_sm);

    auto kernel = kernels::group_gemm_2sm_gather4_mxfp8_kernel<decltype(config), decltype(tma_b),
                                                               decltype(tma_y), decltype(tma_sfb)>;
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

    auto launch_err = cudaLaunchKernelEx(
        &launch_config, kernel, tma_a_g4, tma_b, tma_sfb, tma_ay,
        reinterpret_cast<const uint8_t *>(sfx_ptr), reinterpret_cast<const int *>(x_row_map_ptr),
        x_num_rows, (int *)seqlens_ptr, (int *)cu_seqlens_ptr, (int *)tiles_ptr,
        (int *)cu_tiles_ptr, num_group, m, n, k, flat_divider);
    if (launch_err != cudaSuccess) {
      printf("2SM gather4 launch FAILED: %s (shm=%d)\n", cudaGetErrorString(launch_err), shm_size);
    }
  }
}

// ============================================================================
// Dispatch entry point — 2SM only
// ============================================================================

namespace {
template <typename T>
struct type_id_g4 {
  using type = T;
};
}  // namespace

void group_gemm_gather4_mxfp8_async(void *y_ptr, const void *x_ptr, const void *w_ptr,
                                    const void *sfx_ptr, const void *sfw_packed_ptr,
                                    const void *seqlens_ptr, const void *cu_seqlens_ptr,
                                    void *tmas_ptr, void *tiles_ptr, void *cu_tiles_ptr,
                                    int num_group, int m, int n, int k, int num_seq_per_group_avg,
                                    bool update_tma, cudaStream_t stream, const void *x_row_map_ptr,
                                    int x_num_rows, bool is_fp4) {
  using Tin = cute::float_e4m3_t;
  using TinB_fp4 = cutlass::detail::float_e2m1_unpacksmem_t;
  using Tout = cute::bfloat16_t;
  using Tsf = cutlass::float_ue8m0_t;

  assert(k % 32 == 0 && "group_gemm_gather4_mxfp8: k must be a multiple of 32 (SF_VEC)");
  assert(n % 256 == 0 && "group_gemm_gather4_mxfp8: n must be a multiple of 256 for 2SM");

  int kTileM_dispatch = mxfp8_dispatch_kTileM(num_seq_per_group_avg);

  // 2SM dispatch: kTileN=256, kMmaSM=2, kTileK=128
  auto dispatch = [&](auto tinb_tag) {
    using TinB = typename decltype(tinb_tag)::type;
    constexpr int kTileN = 256;
    constexpr int kTileK = 128;
    constexpr int kMmaSM = 2;
    switch (kTileM_dispatch) {
      case 16:
        return launch_group_gemm_2sm_gather4_mxfp8<
            GroupGEMMMxFp8Config<Tin, Tout, Tsf, 16, kTileN, kTileK, 16, 8, 4, kMmaSM, 4, TinB>>(
            y_ptr, x_ptr, w_ptr, sfx_ptr, sfw_packed_ptr, seqlens_ptr, cu_seqlens_ptr, tmas_ptr,
            tiles_ptr, cu_tiles_ptr, num_group, m, n, k, update_tma, stream, x_row_map_ptr,
            x_num_rows);
      case 32:
        return launch_group_gemm_2sm_gather4_mxfp8<
            GroupGEMMMxFp8Config<Tin, Tout, Tsf, 32, kTileN, kTileK, 32, 6, 4, kMmaSM, 4, TinB>>(
            y_ptr, x_ptr, w_ptr, sfx_ptr, sfw_packed_ptr, seqlens_ptr, cu_seqlens_ptr, tmas_ptr,
            tiles_ptr, cu_tiles_ptr, num_group, m, n, k, update_tma, stream, x_row_map_ptr,
            x_num_rows);
      case 64:
        return launch_group_gemm_2sm_gather4_mxfp8<
            GroupGEMMMxFp8Config<Tin, Tout, Tsf, 64, kTileN, kTileK, 32, 6, 4, kMmaSM, 4, TinB>>(
            y_ptr, x_ptr, w_ptr, sfx_ptr, sfw_packed_ptr, seqlens_ptr, cu_seqlens_ptr, tmas_ptr,
            tiles_ptr, cu_tiles_ptr, num_group, m, n, k, update_tma, stream, x_row_map_ptr,
            x_num_rows);
      case 96:
        return launch_group_gemm_2sm_gather4_mxfp8<
            GroupGEMMMxFp8Config<Tin, Tout, Tsf, 96, kTileN, kTileK, 32, 6, 4, kMmaSM, 4, TinB>>(
            y_ptr, x_ptr, w_ptr, sfx_ptr, sfw_packed_ptr, seqlens_ptr, cu_seqlens_ptr, tmas_ptr,
            tiles_ptr, cu_tiles_ptr, num_group, m, n, k, update_tma, stream, x_row_map_ptr,
            x_num_rows);
      case 128:
        return launch_group_gemm_2sm_gather4_mxfp8<
            GroupGEMMMxFp8Config<Tin, Tout, Tsf, 128, kTileN, kTileK, 32, 6, 4, kMmaSM, 3, TinB>>(
            y_ptr, x_ptr, w_ptr, sfx_ptr, sfw_packed_ptr, seqlens_ptr, cu_seqlens_ptr, tmas_ptr,
            tiles_ptr, cu_tiles_ptr, num_group, m, n, k, update_tma, stream, x_row_map_ptr,
            x_num_rows);
      case 160:
        return launch_group_gemm_2sm_gather4_mxfp8<
            GroupGEMMMxFp8Config<Tin, Tout, Tsf, 160, kTileN, kTileK, 32, 6, 4, kMmaSM, 2, TinB>>(
            y_ptr, x_ptr, w_ptr, sfx_ptr, sfw_packed_ptr, seqlens_ptr, cu_seqlens_ptr, tmas_ptr,
            tiles_ptr, cu_tiles_ptr, num_group, m, n, k, update_tma, stream, x_row_map_ptr,
            x_num_rows);
      case 192:
        return launch_group_gemm_2sm_gather4_mxfp8<
            GroupGEMMMxFp8Config<Tin, Tout, Tsf, 192, kTileN, kTileK, 32, 6, 2, kMmaSM, 2, TinB>>(
            y_ptr, x_ptr, w_ptr, sfx_ptr, sfw_packed_ptr, seqlens_ptr, cu_seqlens_ptr, tmas_ptr,
            tiles_ptr, cu_tiles_ptr, num_group, m, n, k, update_tma, stream, x_row_map_ptr,
            x_num_rows);
      default:
        return launch_group_gemm_2sm_gather4_mxfp8<
            GroupGEMMMxFp8Config<Tin, Tout, Tsf, 256, kTileN, kTileK, 64, 5, 2, kMmaSM, 1, TinB>>(
            y_ptr, x_ptr, w_ptr, sfx_ptr, sfw_packed_ptr, seqlens_ptr, cu_seqlens_ptr, tmas_ptr,
            tiles_ptr, cu_tiles_ptr, num_group, m, n, k, update_tma, stream, x_row_map_ptr,
            x_num_rows);
    }
  };

  if (is_fp4) {
    dispatch(type_id_g4<TinB_fp4>{});
  } else {
    dispatch(type_id_g4<Tin>{});
  }
}

}  // namespace group_gemm
}  // namespace hpc
