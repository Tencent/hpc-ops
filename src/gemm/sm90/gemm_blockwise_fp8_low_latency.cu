// Copyright 2025 hpc-ops authors

#include <cuda.h>

#include <algorithm>

#include "cute/arch/mma_sm89.hpp"
#include "cute/atom/mma_traits_sm89.hpp"
#include "cute/tensor.hpp"
#include "src/gemm/gemm.h"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace gemm {
namespace kernels {

using namespace cute;  // NOLINT

static constexpr int kTileM = 8;
static constexpr int kTileN = 16;
static constexpr int kTileK = 128;
static constexpr int kStage = 4;
static constexpr int kConsumerWarps = 4;
static constexpr int kNumConsumerThreads = kConsumerWarps * 32;
static constexpr int kNumThreads = kNumConsumerThreads + 32;

using Tin = cute::float_e4m3_t;
using Tout = cute::bfloat16_t;

using MmaAtom = SM89_16x8x32_F32E4M3E4M3F32_TN;
using TiledMmaT = decltype(make_tiled_mma(MMA_Atom<MmaAtom>{}, Layout<Shape<_1, _1, _4>>{}));

using SLayoutAtomT = decltype(GMMA::Layout_K_SW32_Atom<Tin>{});
using SLayoutA = decltype(tile_to_shape(SLayoutAtomT{},
                                        make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{})));
using SLayoutB = decltype(tile_to_shape(SLayoutAtomT{},
                                        make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{})));

template <typename TmaA, typename TmaB>
__global__ void __launch_bounds__(kNumThreads)
    gemm_blockwise_fp8_low_latency_kernel(const __grid_constant__ TmaA tma_a,
                                          const __grid_constant__ TmaB tma_b, Tout *y_ptr,
                                          const float *x_scale_ptr, const float *weight_scale_ptr,
                                          int m, int n, int k, int x_scale_stride,
                                          int w_scale_stride) {
  int idx = threadIdx.x;

  int itile_n = blockIdx.x;

  int num_tile_k = k / kTileK;

  int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
  int elected = cute::elect_one_sync();

  __shared__ uint64_t readable[kStage];
  __shared__ uint64_t writable[kStage];

  extern __shared__ uint8_t shm_data[] alignas(128);
  auto *shm_a = (Tin *)shm_data;
  auto *shm_b = (Tin *)(shm_a + cosize(SLayoutA{}));
  auto *shm_reduce = (float *)(shm_b + cosize(SLayoutB{}));

  auto sA = make_tensor(make_smem_ptr(shm_a), SLayoutA{});
  auto sB = make_tensor(make_smem_ptr(shm_b), SLayoutB{});

  auto gA = tma_a.get_tma_tensor(make_shape(m, k));
  auto gB = tma_b.get_tma_tensor(make_shape(n, k));

  auto btma_a = tma_a.get_slice(0);
  auto btma_b = tma_b.get_slice(0);

  auto tAg = btma_a.partition_S(gA);
  auto tAs = btma_a.partition_D(sA);
  auto tBg = btma_b.partition_S(gB);
  auto tBs = btma_b.partition_D(sB);

  bool is_leader_in_block = (iwarp == 0) && elected;

  if (is_leader_in_block) {
#pragma unroll
    for (int i = 0; i < kStage; ++i) {
      initialize_barrier(readable[i], 1);
      initialize_barrier(writable[i], kConsumerWarps);
    }
  }
  __syncthreads();

  constexpr int kTransactionBytes = sizeof(Tin) * cosize(SLayoutA{}(_, _, 0)) +  // NOLINT
                                    sizeof(Tin) * cosize(SLayoutB{}(_, _, 0));   // NOLINT

  if (idx >= kNumConsumerThreads) {
    int local_idx = idx - kNumConsumerThreads;
    int local_warp = __shfl_sync(0xFFFFFFFF, local_idx / 32, 0);
    int is_leader = (local_warp == 0) && elected;

    if (is_leader) {
      int phase = 1;
      int ismem_write = 0;

#pragma unroll 1
      for (int itile_k = 0; itile_k < num_tile_k; itile_k++) {
        wait_barrier(writable[ismem_write], phase);
        set_barrier_transaction_bytes(readable[ismem_write], kTransactionBytes);
        cute::copy(tma_a.with(readable[ismem_write]), tAg(_, 0, itile_k),
                   tAs(_, 0, 0, ismem_write));
        cute::copy(tma_b.with(readable[ismem_write]), tBg(_, itile_n, itile_k),
                   tBs(_, 0, 0, ismem_write));
        ++ismem_write;
        if (ismem_write == kStage) {
          ismem_write = 0;
          phase ^= 1;
        }
      }
    }
  } else {
    int consumer_warp_id = idx / 32;
    int lane_in_warp = idx % 32;
    TiledMmaT tiled_mma;
    auto thr_mma = tiled_mma.get_slice(idx);

    auto tAr = thr_mma.partition_fragment_A(sB(_, _, _0{}));
    auto tBr = thr_mma.partition_fragment_B(sA(_, _, _0{}));

    auto smem_copy_A = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, Tin>{}, tiled_mma);
    auto smem_copy_B = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, Tin>{}, tiled_mma);

    auto thr_copy_A = smem_copy_A.get_thread_slice(idx);
    auto thr_copy_B = smem_copy_B.get_thread_slice(idx);

    auto tCsB = thr_copy_A.partition_S(as_position_independent_swizzle_tensor(sB));
    auto tCsA = thr_copy_B.partition_S(as_position_independent_swizzle_tensor(sA));

    auto tArA_view = thr_copy_A.retile_D(tAr);
    auto tBrB_view = thr_copy_B.retile_D(tBr);

    auto gC =
        make_tensor(make_gmem_ptr((float *)(nullptr)), make_shape(Int<kTileN>{}, Int<kTileM>{}));
    auto tCr = thr_mma.partition_fragment_C(gC);
    auto tDr = make_tensor_like(tCr);
    clear(tDr);

    auto tCr_mn = retile_fragment(tCr);
    auto tDr_mn = retile_fragment(tDr);
    constexpr int kFragM = size<0>(tCr_mn);
    constexpr int kFragN = size<1>(tCr_mn);

    auto gI = make_identity_tensor(make_shape(Int<kTileN>{}, Int<kTileM>{}));
    auto tI = thr_mma.partition_C(gI);
    auto tI_mn = retile_fragment(tI);

    int w_scale_row = itile_n * kTileN / 128;

    int ismem_read = 0;
    int phase = 0;

#pragma unroll 1
    for (int itile_k = 0; itile_k < num_tile_k; itile_k++) {
      wait_barrier(readable[ismem_read], phase);

      cute::copy(smem_copy_A, tCsB(_, _, _0{}, ismem_read), tArA_view(_, _, _0{}));
      cute::copy(smem_copy_B, tCsA(_, _, _0{}, ismem_read), tBrB_view(_, _, _0{}));

      clear(tCr);
      cute::gemm(tiled_mma, tAr, tBr, tCr);

      float scale_w = __ldg(weight_scale_ptr + w_scale_row * w_scale_stride + itile_k);

      float tCS[kFragN];
#pragma unroll
      for (int in = 0; in < kFragN; in++) {
        int m_local = get<1>(tI_mn(0, in));
        float scale_x =
            (m_local < m) ? __ldg(x_scale_ptr + m_local * x_scale_stride + itile_k) : 0.f;
        tCS[in] = scale_x * scale_w;
      }

#pragma unroll
      for (int in = 0; in < kFragN; in++) {
        float yscale = tCS[in];
#pragma unroll
        for (int im = 0; im < kFragM; im++) {
          tDr_mn(im, in) += tCr_mn(im, in) * yscale;
        }
      }

      if (elected) {
        arrive_barrier(writable[ismem_read]);
      }

      ++ismem_read;
      if (ismem_read == kStage) {
        phase ^= 1;
        ismem_read = 0;
      }
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int n_local = get<0>(tI(i));
      int m_local = get<1>(tI(i));
      shm_reduce[consumer_warp_id * (kTileM * kTileN) + m_local * kTileN + n_local] = tDr(i);
    }

    hpc::bar_sync<kNumConsumerThreads>(1);

    if (consumer_warp_id == 0) {
      int m_row = lane_in_warp / (kTileN / 4);
      int n_group = lane_in_warp % (kTileN / 4);
      int n_start = n_group * 4;

      float acc[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        int smem_idx = m_row * kTileN + n_start + i;
        acc[i] = shm_reduce[smem_idx];
#pragma unroll
        for (int w = 1; w < kConsumerWarps; ++w) {
          acc[i] += shm_reduce[w * (kTileM * kTileN) + smem_idx];
        }
      }

      int n_base = itile_n * kTileN;
      if (m_row < m) {
        __nv_bfloat162 p0 = __float22bfloat162_rn(make_float2(acc[0], acc[1]));
        __nv_bfloat162 p1 = __float22bfloat162_rn(make_float2(acc[2], acc[3]));
        uint32_t bf16x2[2];
        bf16x2[0] = *reinterpret_cast<uint32_t *>(&p0);
        bf16x2[1] = *reinterpret_cast<uint32_t *>(&p1);
        *reinterpret_cast<uint64_t *>(&y_ptr[m_row * n + n_base + n_start]) =
            *reinterpret_cast<uint64_t *>(bf16x2);
      }
    }
  }
}

}  // namespace kernels

bool launch_gemm_blockwise_fp8_low_latency(void *y_ptr, const void *x_ptr, const void *w_ptr,
                                           const void *x_scale_ptr, const void *weight_scale_ptr,
                                           int m, int n, int k, int x_scale_stride,
                                           int w_scale_stride, cudaStream_t stream) {
  using namespace cute;     // NOLINT
  using namespace kernels;  // NOLINT

  if (k % 128 != 0 || n % 16 != 0) {
    return false;
  }
  if (m > 8) {
    return false;
  }

  auto X = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(x_ptr)), make_shape(m, k),
                       make_stride(k, Int<1>{}));
  auto W = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(w_ptr)), make_shape(n, k),
                       make_stride(k, Int<1>{}));

  auto tma_x = make_tma_copy(SM90_TMA_LOAD{}, X, SLayoutA{}(_, _, 0));  // NOLINT
  auto tma_w = make_tma_copy(SM90_TMA_LOAD{}, W, SLayoutB{}(_, _, 0));  // NOLINT

  static constexpr int shm_data_size = sizeof(Tin) * (cosize(SLayoutA{}) + cosize(SLayoutB{}));
  static constexpr int shm_reduce_size = kConsumerWarps * 128 * sizeof(float);
  static constexpr int shm_size = shm_data_size + shm_reduce_size;

  auto kernel = kernels::gemm_blockwise_fp8_low_latency_kernel<decltype(tma_x), decltype(tma_w)>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

  int num_tile_n = (n + kTileN - 1) / kTileN;
  dim3 block(kNumThreads);
  dim3 grid(num_tile_n);

  kernel<<<grid, block, shm_size, stream>>>(
      tma_x, tma_w, reinterpret_cast<Tout *>(y_ptr), reinterpret_cast<const float *>(x_scale_ptr),
      reinterpret_cast<const float *>(weight_scale_ptr), m, n, k, x_scale_stride, w_scale_stride);

  return true;
}

bool gemm_blockwise_fp8_low_latency_async(void *y_ptr, const void *x_ptr, const void *w_ptr,
                                          const void *x_scale_ptr, const void *weight_scale_ptr,
                                          int m, int n, int k, int x_scale_stride,
                                          int w_scale_stride, cudaStream_t stream) {
  return launch_gemm_blockwise_fp8_low_latency(y_ptr, x_ptr, w_ptr, x_scale_ptr, weight_scale_ptr,
                                               m, n, k, x_scale_stride, w_scale_stride, stream);
}

}  // namespace gemm
}  // namespace hpc
