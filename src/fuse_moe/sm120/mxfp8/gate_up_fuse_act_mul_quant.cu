// Copyright (C) 2026 Tencent.

#include <cuda_runtime.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "src/fuse_moe/fuse_moe_mxfp8.h"
#include "src/fuse_moe/sm120/mxfp8/config.h"
#include "src/fuse_moe/sm120/mxfp8/gate_up_fuse_act_mul_quant.cuh"
#include "src/utils/tma.cuh"
#include "src/utils/utils.h"

namespace hpc {
namespace fuse_moe {

using namespace cute;  // NOLINT

template <int kTileM_, int kNumTopk, typename TW>
static void launch_gate_up(void *y_ptr, void *y_scale_ptr, const void *x_ptr,
                           const void *x_scale_ptr, const void *w_ptr, const void *w_scale_ptr,
                           const void *seqlens_ptr, const void *cu_seqlens_ptr,
                           const void *tiles_ptr, const void *permuted_idx_to_raw_idx_ptr,
                           int num_group, int max_tiles, int n, int k, float limit,
                           cudaStream_t stream) {
  using Config = GateUpConfig<kTileM_, TW>;
  using TA = typename Config::TA;
  using TB = typename Config::TB;
  using TC = typename Config::TC;
  using TSF = typename Config::TSF;
  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;
  constexpr int kSFBlockM = Config::kSFBlockM;
  constexpr int kSFBlockBytes = Config::kSFBlockBytes;

  int m_pad = max_tiles * kTileM;
  int num_tile_k = k / kTileK;

  // W's gmem is read as TW: for fp4 make_tma_copy computes strides at 4 bits per element and
  // picks 16U4_ALIGN16B to expand it into shared memory.
  auto W =
      make_tensor(make_gmem_ptr(reinterpret_cast<const TW *>(w_ptr)), make_shape(n, k, num_group),
                  make_stride(static_cast<int64_t>(k), Int<1>{}, static_cast<int64_t>(n) * k));
  auto Y = make_tensor(make_gmem_ptr(reinterpret_cast<TC *>(y_ptr)), make_shape(m_pad, n / 2),
                       make_stride(static_cast<int64_t>(n / 2), Int<1>{}));
  auto SFW = make_tensor(
      make_gmem_ptr(reinterpret_cast<const TSF *>(w_scale_ptr)),
      make_shape(Int<4>{}, Int<128>{}, n / kSFBlockM, num_tile_k, num_group),
      make_stride(Int<128>{}, Int<1>{}, static_cast<int64_t>(kSFBlockBytes) * num_tile_k,
                  Int<kSFBlockBytes>{},
                  static_cast<int64_t>(kSFBlockBytes) * num_tile_k * (n / kSFBlockM)));

  auto tma_b = make_tma_copy(SM90_TMA_LOAD{}, W, typename Config::CopyBoxB{});
  auto tma_y = make_tma_copy(SM90_TMA_STORE{}, Y, typename Config::CopyBoxC{});
  auto tma_sfb = make_tma_copy(SM90_TMA_LOAD{}, SFW, typename Config::CopyBoxSF{});

  auto kernel =
      kernels::mxfp8::gate_up_fuse_act_mul_quant_kernel<Config, kNumTopk, decltype(tma_b),
                                                        decltype(tma_y), decltype(tma_sfb)>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Config::kShmSize);

  int num_tile_n = n / kTileN;
  cutlass::FastDivmod tile_n_divider(num_tile_n);

  dim3 block(Config::kNumThreads);
  int num_sm = get_sm_count();
  dim3 grid(std::min(num_sm, max_tiles * num_tile_n));

  cudaLaunchConfig_t config_launch{};
  config_launch.gridDim = grid;
  config_launch.blockDim = block;
  config_launch.dynamicSmemBytes = Config::kShmSize;
  config_launch.stream = stream;

  // PDL
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config_launch.attrs = attrs;
  config_launch.numAttrs = 1;

  cudaLaunchKernelEx(
      &config_launch, kernel, tma_b, tma_y, tma_sfb, reinterpret_cast<TSF *>(y_scale_ptr),
      reinterpret_cast<const TA *>(x_ptr), reinterpret_cast<const TSF *>(x_scale_ptr),
      reinterpret_cast<const int *>(seqlens_ptr), reinterpret_cast<const int *>(cu_seqlens_ptr),
      reinterpret_cast<const int *>(tiles_ptr),
      reinterpret_cast<const int *>(permuted_idx_to_raw_idx_ptr), num_group, m_pad, n, k, limit,
      tile_n_divider);
}

void gate_up_fuse_act_mul_quant_async(void *y_ptr, void *y_scale_ptr, const void *x_ptr,
                                      const void *x_scale_ptr, const void *w_ptr,
                                      const void *w_scale_ptr, const void *seqlens_ptr,
                                      const void *cu_seqlens_ptr, const void *tiles_ptr,
                                      const void *permuted_idx_to_raw_idx_ptr, int num_group,
                                      int max_tiles, int n, int k, int tile_m, int num_topk,
                                      float limit, bool is_fp4, cudaStream_t stream) {
  dispatch_tile_m(tile_m, [&](auto kTileM) {
    dispatch_topk(num_topk, [&](auto kNumTopk) {
      dispatch_w_type(is_fp4, [&](auto tw) {
        launch_gate_up<decltype(kTileM)::value, decltype(kNumTopk)::value, decltype(tw)>(
            y_ptr, y_scale_ptr, x_ptr, x_scale_ptr, w_ptr, w_scale_ptr, seqlens_ptr, cu_seqlens_ptr,
            tiles_ptr, permuted_idx_to_raw_idx_ptr, num_group, max_tiles, n, k, limit, stream);
      });
    });
  });
}

}  // namespace fuse_moe
}  // namespace hpc
