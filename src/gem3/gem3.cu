// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "src/gem3/gem3.h"

namespace hpc {
namespace gem3 {

namespace kernels {

template <typename Tout, typename Tin, int kTileM, int kTileN, int kTileK, typename TiledMmaQK,
          typename TiledMmaAV, typename TmaQ, typename TmaK, typename TmaV, typename TmaY,
          typename SLayoutQ, typename SLayoutK, typename SLayoutV, typename SLayoutY>
__global__ void gem3_kernel(__grid_constant__ const TmaQ tma_q, __grid_constant__ const TmaK tma_k,
                            __grid_constant__ const TmaV tma_v, __grid_constant__ const TmaY tma_y,
                            int num_batch, int num_seq, int num_qk_dim, int num_v_dim,
                            void *y_ptr) {
  using namespace cute;  // NOLINT

  int idx = threadIdx.x;
  int itile_m = blockIdx.x;
  int ibatch = blockIdx.y;

  __shared__ uint64_t bar_q;
  __shared__ uint64_t bar_k;
  __shared__ uint64_t bar_v;
  extern __shared__ float4 shm_data[] alignas(128);

  auto *shm_q = (Tin *)shm_data;
  auto *shm_k = ((Tin *)shm_q) + cosize(SLayoutQ{});
  auto *shm_v = ((Tin *)shm_k) + cosize(SLayoutK{});
  auto *shm_y = ((Tout *)shm_v) + cosize(SLayoutV{});

  // Tensor Q/K/V/Y
  auto gQ = tma_q.get_tma_tensor(make_shape(num_seq, num_qk_dim, num_batch));
  auto gK = tma_k.get_tma_tensor(make_shape(num_seq, num_qk_dim, num_batch));
  auto gV = tma_v.get_tma_tensor(make_shape(num_v_dim, num_seq, num_batch));
  auto gY = tma_y.get_tma_tensor(make_shape(num_seq, num_v_dim, num_batch));

  constexpr int kTileV = 80;
  auto gAtt =
      make_tensor(make_gmem_ptr((float *)shm_data), make_shape(Int<kTileM>{}, Int<kTileN>{}),
                  make_stride(Int<kTileN>{}, Int<1>{}));
  auto gYY = make_tensor(make_gmem_ptr((Tout *)shm_data), make_shape(Int<kTileM>{}, Int<kTileV>{}),
                         make_stride(Int<kTileV>{}, Int<1>{}));

  // Tensor sQ/sK/sV
  auto sQ = make_tensor(make_smem_ptr(shm_q), SLayoutQ{});
  auto sK = make_tensor(make_smem_ptr(shm_k), SLayoutK{});
  auto sV = make_tensor(make_smem_ptr(shm_v), SLayoutV{});
  auto sY = make_tensor(make_smem_ptr(shm_y), SLayoutY{});

  // Block Level tma
  auto btma_q = tma_q.get_slice(0);
  auto btma_k = tma_k.get_slice(0);
  auto btma_v = tma_v.get_slice(0);
  auto btma_y = tma_y.get_slice(0);

  // Thread Level Tensor
  auto tQg = btma_q.partition_S(gQ);  // (TMA, TMA_M, TMA_K, batch)
  auto tKg = btma_k.partition_S(gK);  // (TMA, TMA_N, TMA_K, batch)
  auto tVg = btma_v.partition_S(gV);  // (TMA, TMA_V, TMA_N, batch)

  auto tQs = btma_q.partition_D(sQ);  // (TMA, _1, _1)
  auto tKs = btma_k.partition_D(sK);  // (TMA, _1, _1)
  auto tVs = btma_v.partition_D(sV);  // (TMA, _1, _1)

  TiledMmaQK tiled_mma_qk;
  TiledMmaAV tiled_mma_av;

  auto thr_mma_qk = tiled_mma_qk.get_slice(idx);
  auto thr_mma_av = tiled_mma_av.get_slice(idx);

  auto tQs4r = thr_mma_qk.partition_A(sQ);
  auto tKs4r = thr_mma_qk.partition_B(sK);
  auto tVs4r = thr_mma_av.partition_B(sV);

  auto tQr = thr_mma_qk.make_fragment_A(tQs4r);  // (MMA, MMA_M, MMA_K)
  auto tKr = thr_mma_qk.make_fragment_B(tKs4r);  // (MMA, MMA_N, MMA_K)
  auto tVr = thr_mma_av.make_fragment_B(tVs4r);  // (MMA, MMA_V, MMA_N)

  auto tAttr = thr_mma_qk.partition_fragment_C(gAtt);
  auto tYr = thr_mma_av.partition_fragment_C(gYY);

  // Load Q
  if (idx == 0) {
    initialize_barrier(bar_q, 1);
    set_barrier_transaction_bytes(bar_q, sizeof(Tin) * cosize(SLayoutQ{}));
    cute::copy(tma_q.with(bar_q), tQg(_, itile_m, _, ibatch), tQs(_, 0, _));
  }

  // init k/v barrier
  if (idx == 0) {
    initialize_barrier(bar_k, 1);
    initialize_barrier(bar_v, 1);
  }

  __syncthreads();
  wait_barrier(bar_q, 0);

  auto layout_asC = thr_mma_qk.partition_C(gAtt).layout();
  auto layout_asA = thr_mma_av.partition_A(gAtt).layout();
  auto tAttA = make_tensor(tAttr.data(), left_inverse(layout_asC).compose(layout_asA));

  clear(tYr);
  tiled_mma_av.accumulate_ = GMMA::ScaleOut::One;

#pragma unroll 1
  for (int itile_seq_kv = 0; itile_seq_kv < size<1>(tKg); ++itile_seq_kv) {
    __syncthreads();
    // load k/v
    if (idx == 0) {
      set_barrier_transaction_bytes(bar_k, sizeof(Tin) * cosize(SLayoutK{}));
      cute::copy(tma_k.with(bar_k), tKg(_, itile_seq_kv, _, ibatch), tKs(_, 0, _));

      set_barrier_transaction_bytes(bar_v, sizeof(Tin) * cosize(SLayoutV{}));
      cute::copy(tma_v.with(bar_v), tVg(_, _, itile_seq_kv, ibatch), tVs(_, _, 0));
    }
    __syncthreads();
    wait_barrier(bar_k, itile_seq_kv % 2);

    // P = QK
    tiled_mma_qk.accumulate_ = GMMA::ScaleOut::One;
    clear(tAttr);

    warpgroup_arrive();
    cute::gemm(tiled_mma_qk, tQr, tKr, tAttr);

    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(tAttr);

    // Y = PV
    auto tAttAbf16 = make_tensor_like<Tin>(tAttA);
#pragma unroll
    for (int i = 0; i < size(tAttA); ++i) {
      tAttAbf16(i) = (Tin)(tAttA(i));
    }

    __syncthreads();
    wait_barrier(bar_v, itile_seq_kv % 2);

    warpgroup_arrive();
    cute::gemm(tiled_mma_av, tAttAbf16, tVr, tYr);
    warpgroup_commit_batch();
    warpgroup_wait<0>();

    warpgroup_fence_operand(tYr);
  }

  warpgroup_fence_operand(tYr);

  // to bfloat16
  auto tYr_bf16 = make_tensor_like<Tout>(tYr);

#pragma unroll
  for (int i = 0; i < size(tYr); ++i) {
    Tout v{tYr(i)};
    tYr_bf16(i) = v;
  }

  warpgroup_fence_operand(tYr_bf16);

  // Epilogue: write register-C to global memory
  // simple mode
#if 0
  {
    auto Y = make_tensor(make_gmem_ptr((Tout *)y_ptr), make_shape(num_seq, num_v_dim, num_batch),
                         make_stride(num_v_dim, Int<1>{}, num_seq * num_v_dim));
    auto gY =
        local_tile(Y, make_tile(Int<kTileM>{}, Int<kTileV>{}), make_coord(itile_m, 0, ibatch));
    auto tYg = thr_mma_av.partition_C(gY);
    copy(tYr_bf16, tYg);
  }

#else

  using R2SCopyAtomC = Copy_Atom<cute::SM90_U32x4_STSM_N, Tout>;
  auto r2s_tiled_copy = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma_av);
  auto r2s_thr_copy = r2s_tiled_copy.get_slice(idx);

  auto tYr4s = r2s_thr_copy.retile_S(tYr_bf16);
  auto tYs4r = r2s_thr_copy.partition_D(sY);

  cute::copy(r2s_tiled_copy, tYr4s, tYs4r);
  __syncthreads();

  // using TMA to store
  if (idx == 0) {
    auto cY = tma_y.get_tma_tensor(make_shape(num_seq, num_v_dim, num_batch));
    auto btma_y = tma_y.get_slice(0);

    auto tYss = btma_y.partition_S(sY);  // (TMA, TMA_M, TMA_N)
    auto tYgg = btma_y.partition_D(cY);  // (TMA, TMA_M, TMA_N, b)

    cute::copy(tma_y, tYss(_, 0, 0), tYgg(_, itile_m, 0, ibatch));
  }
#endif
}

}  // namespace kernels

void gem3_async(void *y_ptr, const void *q_ptr, const void *k_ptr, const void *v_ptr, int num_batch,
                int num_seq, int num_qk_dim, int num_v_dim, cudaStream_t stream) {
  using namespace cute;  // NOLINT

  using Tin = cute::bfloat16_t;
  using Tout = cute::bfloat16_t;

  constexpr int kTileM = 128;
  constexpr int kTileN = 128;
  constexpr int kTileK = 128;
  constexpr int kTileV = 80;

  auto Q = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(q_ptr)),
                       make_shape(num_seq, num_qk_dim, num_batch),
                       make_stride(num_qk_dim, Int<1>{}, num_seq * num_qk_dim));
  auto K = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(k_ptr)),
                       make_shape(num_seq, num_qk_dim, num_batch),
                       make_stride(num_qk_dim, Int<1>{}, num_seq * num_qk_dim));
  auto V = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(v_ptr)),
                       make_shape(num_v_dim, num_seq, num_batch),
                       make_stride(Int<1>{}, num_v_dim, num_seq * num_v_dim));
  auto Y = make_tensor(make_gmem_ptr(reinterpret_cast<const Tout *>(y_ptr)),
                       make_shape(num_seq, num_v_dim, num_batch),
                       make_stride(num_v_dim, Int<1>{}, num_seq * num_v_dim));

  auto slayout_q =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{}, make_shape(Int<kTileM>{}, Int<kTileK>{}));
  auto slayout_k =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{}, make_shape(Int<kTileN>{}, Int<kTileK>{}));
  auto slayout_v =
      tile_to_shape(GMMA::Layout_MN_SW32_Atom<Tin>{}, make_shape(Int<kTileV>{}, Int<kTileN>{}));
  auto slayout_y =
      tile_to_shape(GMMA::Layout_K_SW32_Atom<Tout>{}, make_shape(Int<kTileM>{}, Int<kTileV>{}));

  auto tma_q = make_tma_copy(SM90_TMA_LOAD{}, Q, slayout_q);
  auto tma_k = make_tma_copy(SM90_TMA_LOAD{}, K, slayout_k);
  auto tma_v = make_tma_copy(SM90_TMA_LOAD{}, V, slayout_v);
  auto tma_y = make_tma_copy(SM90_TMA_STORE{}, Y, slayout_y);

  using TiledMmaQK =
      decltype(make_tiled_mma(SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{}));
  using TiledMmaAV =
      decltype(make_tiled_mma(SM90_64x80x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>{}));

  dim3 block(size(TiledMmaQK{}));
  dim3 grid((num_seq + kTileM - 1) / kTileM, num_batch);

  int shm_qkv = (cosize(slayout_q) + cosize(slayout_k) + cosize(slayout_v)) * sizeof(Tin);
  int shm_y = cosize(slayout_y) * sizeof(Tout);
  // int shm_size = std::max(shm_qkv, shm_y);
  int shm_size = shm_qkv + shm_y;

  printf("num_batch = %d, num_seq = %d, num_qk_dim = %d, num_v_dim = %d\n", num_batch, num_seq,
         num_qk_dim, num_v_dim);

  print("shm_size(byte) = %d\n", shm_size);

  printf("grid = (%d, %d, %d) block = (%d, %d, %d)\n", grid.x, grid.y, grid.z, block.x, block.y,
         block.z);

  auto kernel = kernels::gem3_kernel<Tout, Tin, kTileM, kTileN, kTileK, TiledMmaQK, TiledMmaAV,
                                     decltype(tma_q), decltype(tma_k), decltype(tma_v),
                                     decltype(tma_y), decltype(slayout_q), decltype(slayout_k),
                                     decltype(slayout_v), decltype(slayout_y)>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
  kernel<<<grid, block, shm_size, stream>>>(tma_q, tma_k, tma_v, tma_y, num_batch, num_seq,
                                            num_qk_dim, num_v_dim, y_ptr);
  /*
   */
}

}  // namespace gem3
}  // namespace hpc
