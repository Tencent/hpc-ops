// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "src/gem3/gem3.h"
#include "src/utils/tma.cuh"
#include "src/utils/utils.cuh"

namespace hpc {
namespace gem3 {

namespace kernels {

template <typename T, typename TmaQ, typename TmaK, typename TmaV, typename TmaQS, typename TmaKS,
          typename TmaY>
__global__ void update_batched_tma(const vec_t<cute::TmaDescriptor, 6> td_qkvqsksy0,
                                   cute::TmaDescriptor *tma_qkvqsksy, const T *q_ptr,
                                   const T *k_ptr, const T *v_ptr, const float *qs_ptr,
                                   const float *ks_ptr, const T *y_ptr, int num_batch, int num_seq,
                                   int num_qk_dim, int num_v_dim) {
  using namespace cute;  // NOLINT

  int idx = threadIdx.x;
  int ilane = idx % 32;
  int ibatch = blockIdx.x;

  __shared__ cute::TmaDescriptor smem_tma_desc[6];

  __syncwarp();
  if (ilane == 0) {
    smem_tma_desc[ibatch] = td_qkvqsksy0[ibatch];
    // Q
    if (ibatch == 0) {
      auto gQ = make_tensor(make_gmem_ptr(q_ptr), make_shape(num_seq, num_qk_dim, num_batch),
                            make_stride(num_qk_dim, Int<1>{}, num_qk_dim * num_seq));
      update_tma_gtensor<TmaQ>(smem_tma_desc[ibatch], gQ);
    }

    // K
    if (ibatch == 1) {
      auto gK = make_tensor(make_gmem_ptr(k_ptr), make_shape(num_seq, num_qk_dim, num_batch),
                            make_stride(num_qk_dim, Int<1>{}, num_qk_dim * num_seq));
      update_tma_gtensor<TmaK>(smem_tma_desc[ibatch], gK);
    }

    // V
    if (ibatch == 2) {
      auto gV = make_tensor(make_gmem_ptr(v_ptr), make_shape(num_v_dim, num_seq, num_batch),
                            make_stride(Int<1>{}, num_v_dim, num_v_dim * num_seq));
      update_tma_gtensor<TmaV>(smem_tma_desc[ibatch], gV);
    }

    // QS
    if (ibatch == 3) {
      auto gQS = make_tensor(make_gmem_ptr(qs_ptr), make_shape(num_seq, num_batch),
                             make_stride(Int<1>{}, num_seq));
      update_tma_gtensor<TmaQS>(smem_tma_desc[ibatch], gQS);
    }

    // KS
    if (ibatch == 4) {
      auto g = make_tensor(make_gmem_ptr(ks_ptr), make_shape(num_seq, num_batch),
                           make_stride(Int<1>{}, num_seq));
      update_tma_gtensor<TmaKS>(smem_tma_desc[ibatch], g);
    }

    // Y
    if (ibatch == 5) {
      auto g = make_tensor(make_gmem_ptr(y_ptr), make_shape(num_seq, num_v_dim, num_batch),
                           make_stride(num_v_dim, Int<1>{}, num_v_dim * num_seq));
      update_tma_gtensor<TmaY>(smem_tma_desc[ibatch], g);
    }
  }

  __syncwarp();

  tma_descriptor_cp_fence_release(tma_qkvqsksy + ibatch, smem_tma_desc[ibatch]);
}

template <typename Tout, typename Tin, int kTileM, int kTileN, int kTileK, int kTileV,
          typename TiledMmaQK, typename TiledMmaAV, typename TmaQ, typename TmaK, typename TmaV,
          typename TmaY, typename TmaQS, typename TmaKS, typename SLayoutQ, typename SLayoutK,
          typename SLayoutV, typename SLayoutY, typename SLayoutQS, typename SLayoutKS>
__global__ void gem3_kernel(cute::TmaDescriptor *td_qkvqsksy, int num_batch, int num_seq,
                            int num_qk_dim, int num_v_dim) {
  using namespace cute;  // NOLINT

  int idx = threadIdx.x;
  int itile_m = blockIdx.x;
  int ibatch = blockIdx.y;

  int elected = cute::elect_one_sync();
  int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);

  __shared__ uint64_t bar_q;
  __shared__ uint64_t bar_k;
  __shared__ uint64_t bar_v;
  __shared__ uint64_t bar_qs;
  __shared__ uint64_t bar_ks;
  extern __shared__ uint8_t shm_data[] alignas(128);

  auto *shm_q = (Tin *)shm_data;
  auto *shm_k = ((Tin *)shm_q) + cosize(SLayoutQ{});
  auto *shm_v = ((Tin *)shm_k) + cosize(SLayoutK{});
  auto *shm_qs = (float *)(((Tin *)shm_v) + cosize(SLayoutV{}));
  auto *shm_ks = (float *)(((float *)shm_qs) + cosize(SLayoutQS{}));
  auto *shm_y = ((Tout *)shm_data);  // Reuse All

  TmaQ tma_q;
  TmaK tma_k;
  TmaV tma_v;
  TmaQS tma_qs;
  TmaKS tma_ks;
  TmaY tma_y;

  auto *td_q = td_qkvqsksy + 0;
  auto *td_k = td_qkvqsksy + 1;
  auto *td_v = td_qkvqsksy + 2;
  auto *td_qs = td_qkvqsksy + 3;
  auto *td_ks = td_qkvqsksy + 4;
  auto *td_y = td_qkvqsksy + 5;

  // Tensor Q/K/V/Y
  auto gQ = tma_q.get_tma_tensor(make_shape(num_seq, num_qk_dim, num_batch));
  auto gK = tma_k.get_tma_tensor(make_shape(num_seq, num_qk_dim, num_batch));
  auto gV = tma_v.get_tma_tensor(make_shape(num_v_dim, num_seq, num_batch));
  auto gY = tma_y.get_tma_tensor(make_shape(num_seq, num_v_dim, num_batch));
  auto gQS = tma_qs.get_tma_tensor(make_shape(num_seq, num_batch));  // qscale
  auto gKS = tma_ks.get_tma_tensor(make_shape(num_seq, num_batch));  // kscale

  auto gAtt = make_tensor(make_gmem_ptr((float *)nullptr), make_shape(Int<kTileM>{}, Int<kTileN>{}),
                          make_stride(Int<kTileN>{}, Int<1>{}));
  auto gYY = make_tensor(make_gmem_ptr((Tout *)nullptr), make_shape(Int<kTileM>{}, Int<kTileV>{}),
                         make_stride(Int<kTileV>{}, Int<1>{}));

  // Tensor sQ/sK/sV
  auto sQ = make_tensor(make_smem_ptr(shm_q), SLayoutQ{});
  auto sK = make_tensor(make_smem_ptr(shm_k), SLayoutK{});
  auto sV = make_tensor(make_smem_ptr(shm_v), SLayoutV{});
  auto sY = make_tensor(make_smem_ptr(shm_y), SLayoutY{});
  auto sQS = make_tensor(make_smem_ptr(shm_qs), SLayoutQS{});
  auto sKS = make_tensor(make_smem_ptr(shm_ks), SLayoutKS{});

  // Block Level tma
  auto btma_q = tma_q.get_slice(0);
  auto btma_k = tma_k.get_slice(0);
  auto btma_v = tma_v.get_slice(0);
  auto btma_y = tma_y.get_slice(0);
  auto btma_qs = tma_qs.get_slice(0);
  auto btma_ks = tma_ks.get_slice(0);

  // Thread Level Tensor
  auto tQg = btma_q.partition_S(gQ);     // (TMA, TMA_M, TMA_K, batch)
  auto tKg = btma_k.partition_S(gK);     // (TMA, TMA_N, TMA_K, batch)
  auto tVg = btma_v.partition_S(gV);     // (TMA, TMA_V, TMA_N, batch)
  auto tQSg = btma_qs.partition_S(gQS);  // (TMA, TMA_M, batch)
  auto tKSg = btma_ks.partition_S(gKS);  // (TMA, TMA_N, batch)

  auto tQs = btma_q.partition_D(sQ);     // (TMA, _1, _1)
  auto tKs = btma_k.partition_D(sK);     // (TMA, _1, _1)
  auto tVs = btma_v.partition_D(sV);     // (TMA, _1, _1)
  auto tQSs = btma_qs.partition_D(sQS);  // (TMA, _1)
  auto tKSs = btma_ks.partition_D(sKS);  // (TMA, _1)

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

  auto gI = make_identity_tensor(gAtt.shape());
  auto tI = thr_mma_qk.partition_C(gI);

  // Load Q
  if ((iwarp == 0) && elected) {
    initialize_barrier(bar_q, 1);
    set_barrier_transaction_bytes(bar_q, sizeof(Tin) * cosize(SLayoutQ{}));
    cute::copy(tma_q.with(td_q, bar_q), tQg(_, itile_m, _, ibatch), tQs(_, 0, _));
  }

  // init k/v barrier
  if ((iwarp == 0) && elected) {
    initialize_barrier(bar_k, 1);
    initialize_barrier(bar_v, 1);
    initialize_barrier(bar_qs, 1);
    initialize_barrier(bar_ks, 1);
  }

  __syncthreads();
  wait_barrier(bar_q, 0);

  auto layout_asC = thr_mma_qk.partition_C(gAtt).layout();
  auto layout_asA = thr_mma_av.partition_A(gAtt).layout();
  auto tAttA = make_tensor(tAttr.data(), left_inverse(layout_asC).compose(layout_asA));

  __syncthreads();
  clear(tYr);

  tiled_mma_av.accumulate_ = GMMA::ScaleOut::One;

#pragma unroll 1
  for (int itile_seq_kv = 0; itile_seq_kv < size<1>(tKg); ++itile_seq_kv) {
    // casual mask, skip unused block
    int icol = itile_seq_kv * kTileN;
    int idiag = (itile_m + 1) * kTileM;
    if (icol >= idiag) {
      break;
    }

    // load k/scale/v
    if ((iwarp == 0) && elected) {
      // k
      set_barrier_transaction_bytes(bar_k, sizeof(Tin) * cosize(SLayoutK{}));
      cute::copy(tma_k.with(td_k, bar_k), tKg(_, itile_seq_kv, _, ibatch), tKs(_, 0, _));

      // qscale/kscale
      set_barrier_transaction_bytes(bar_qs, sizeof(float) * cosize(SLayoutQS{}));
      cute::copy(tma_qs.with(td_qs, bar_qs), tQSg(_, itile_m, ibatch), tQSs(_, 0));
      set_barrier_transaction_bytes(bar_ks, sizeof(float) * cosize(SLayoutKS{}));
      cute::copy(tma_ks.with(td_ks, bar_ks), tKSg(_, itile_seq_kv, ibatch), tKSs(_, 0));

      // v
      set_barrier_transaction_bytes(bar_v, sizeof(Tin) * cosize(SLayoutV{}));
      cute::copy(tma_v.with(td_v, bar_v), tVg(_, _, itile_seq_kv, ibatch), tVs(_, _, 0));
    }
    __syncthreads();
    wait_barrier(bar_k, itile_seq_kv % 2);

    // P = QK
    tiled_mma_qk.accumulate_ = GMMA::ScaleOut::One;
    clear(tAttr);

    warpgroup_fence_operand(tAttr);
    warpgroup_arrive();
    cute::gemm(tiled_mma_qk, tQr, tKr, tAttr);

    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(tAttr);

    // wait qscale/kscale loaded
    __syncthreads();
    wait_barrier(bar_qs, itile_seq_kv % 2);
    wait_barrier(bar_ks, itile_seq_kv % 2);

    // do causal mask
#pragma unroll
    for (int i = 0; i < size(tAttr); ++i) {
      int irow_local = get<0>(tI(i));
      int icol_local = get<1>(tI(i));

      int irow = irow_local + itile_m * kTileM;
      int icol = icol_local + itile_seq_kv * kTileN;

      if (icol > irow) {
        tAttr(i) = 0.f;
      } else {
        // tAttr(i) = tAttr(i) * expf_ftz(tQSs[irow_local] - tKSs[icol_local]);
        tAttr(i) = tAttr(i) * tQSs[irow_local] * tKSs[icol_local];
      }
    }

    // Y = PV
    auto tAttAbf16 = make_tensor_like<Tin>(tAttA);
#pragma unroll
    for (int i = 0; i < size(tAttA); ++i) {
      tAttAbf16(i) = (Tin)(tAttA(i));
    }

    __syncthreads();
    wait_barrier(bar_v, itile_seq_kv % 2);

    warpgroup_fence_operand(tYr);
    warpgroup_arrive();
    cute::gemm(tiled_mma_av, tAttAbf16, tVr, tYr);
    warpgroup_commit_batch();
    warpgroup_wait<0>();

    warpgroup_fence_operand(tYr);
    __syncthreads();
  }

  // to bfloat16
  auto tYr_bf16 = make_tensor_like<Tout>(tYr);

#pragma unroll
  for (int i = 0; i < size(tYr); ++i) {
    Tout v{tYr(i)};
    tYr_bf16(i) = v;
  }

  // Epilogue: write register-C to global memory
  using R2SCopyAtomC = Copy_Atom<cute::SM90_U32x4_STSM_N, Tout>;
  auto r2s_tiled_copy = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma_av);
  auto r2s_thr_copy = r2s_tiled_copy.get_slice(idx);

  auto tYr4s = r2s_thr_copy.retile_S(tYr_bf16);
  auto tYs4r = r2s_thr_copy.partition_D(sY);

  cute::copy(r2s_tiled_copy, tYr4s, tYs4r);
  __syncthreads();

  tma_store_fence();
  // using TMA to store
  if ((iwarp == 0) && elected) {
    auto cY = tma_y.get_tma_tensor(make_shape(num_seq, num_v_dim, num_batch));
    auto btma_y = tma_y.get_slice(0);

    auto tYss = btma_y.partition_S(sY);  // (TMA, TMA_M, TMA_N)
    auto tYgg = btma_y.partition_D(cY);  // (TMA, TMA_M, TMA_N, b)

    cute::copy(tma_y.with(td_y), tYss(_, 0, 0), tYgg(_, itile_m, 0, ibatch));
  }
}

}  // namespace kernels

void gem3_async(void *y_ptr, const void *q_ptr, const void *k_ptr, const void *v_ptr,
                const void *qs_ptr, const void *ks_ptr, void *tmas_ptr, int num_batch, int num_seq,
                int num_qk_dim, int num_v_dim, cudaStream_t stream) {
  using namespace cute;  // NOLINT

  using Tin = cute::bfloat16_t;
  using Tout = cute::bfloat16_t;

  constexpr int kTileM = 64;
  constexpr int kTileN = 64;
  constexpr int kTileK = 128;
  constexpr int kTileV = 80;

  assert(kTileK == num_qk_dim);

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
  auto QS = make_tensor(make_gmem_ptr(reinterpret_cast<const float *>(qs_ptr)),
                        make_shape(num_seq, num_batch), make_stride(Int<1>{}, num_seq));
  auto KS = make_tensor(make_gmem_ptr(reinterpret_cast<const float *>(ks_ptr)),
                        make_shape(num_seq, num_batch), make_stride(Int<1>{}, num_seq));

  // TODO(reed): optimize it
  auto slayout_q =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{}, make_shape(Int<kTileM>{}, Int<kTileK>{}));
  auto slayout_k =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<Tin>{}, make_shape(Int<kTileN>{}, Int<kTileK>{}));
  auto slayout_v =
      tile_to_shape(GMMA::Layout_MN_SW32_Atom<Tin>{}, make_shape(Int<kTileV>{}, Int<kTileN>{}));
  auto slayout_y =
      tile_to_shape(GMMA::Layout_K_SW32_Atom<Tout>{}, make_shape(Int<kTileM>{}, Int<kTileV>{}));

  auto slayout_qs = make_layout(make_shape(Int<kTileM>{}), make_stride(Int<1>{}));
  auto slayout_ks = make_layout(make_shape(Int<kTileN>{}), make_stride(Int<1>{}));

  auto tma_q = make_tma_copy(SM90_TMA_LOAD{}, Q, slayout_q);
  auto tma_k = make_tma_copy(SM90_TMA_LOAD{}, K, slayout_k);
  auto tma_v = make_tma_copy(SM90_TMA_LOAD{}, V, slayout_v);
  auto tma_y = make_tma_copy(SM90_TMA_STORE{}, Y, slayout_y);
  auto tma_qs = make_tma_copy(SM90_TMA_LOAD{}, QS, slayout_qs);
  auto tma_ks = make_tma_copy(SM90_TMA_LOAD{}, KS, slayout_ks);

  using TiledMmaQK =
      decltype(make_tiled_mma(SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{}));
  using TiledMmaAV =
      decltype(make_tiled_mma(SM90_64x80x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>{}));

  static_assert(kTileM >= get<0>(TiledMmaQK::Shape_MNK{}));
  static_assert(kTileM % get<0>(TiledMmaQK::Shape_MNK{}) == 0);
  static_assert(kTileM >= get<0>(TiledMmaAV::Shape_MNK{}));
  static_assert(kTileM % get<0>(TiledMmaAV::Shape_MNK{}) == 0);

  static_assert(kTileN >= get<1>(TiledMmaQK::Shape_MNK{}));
  static_assert(kTileN % get<1>(TiledMmaQK::Shape_MNK{}) == 0);
  static_assert(kTileN >= get<2>(TiledMmaAV::Shape_MNK{}));
  static_assert(kTileN % get<2>(TiledMmaAV::Shape_MNK{}) == 0);

  static_assert(kTileK >= get<2>(TiledMmaQK::Shape_MNK{}));
  static_assert(kTileK % get<2>(TiledMmaQK::Shape_MNK{}) == 0);

  static_assert(kTileV >= get<1>(TiledMmaAV::Shape_MNK{}));
  static_assert(kTileV % get<1>(TiledMmaAV::Shape_MNK{}) == 0);

  dim3 block(size(TiledMmaQK{}));
  dim3 grid((num_seq + kTileM - 1) / kTileM, num_batch);

  int shm_qkv = (cosize(slayout_q) + cosize(slayout_k) + cosize(slayout_v)) * sizeof(Tin);
  int shm_qs = cosize(slayout_qs) * sizeof(float);
  int shm_ks = cosize(slayout_ks) * sizeof(float);
  int shm_y = cosize(slayout_y) * sizeof(Tout);
  int shm_size = std::max(shm_qkv + shm_qs + shm_ks, shm_y);

  vec_t<cute::TmaDescriptor, 6> td_qkvqsksy0{
      *tma_q.get_tma_descriptor(),  *tma_k.get_tma_descriptor(),  *tma_v.get_tma_descriptor(),
      *tma_qs.get_tma_descriptor(), *tma_ks.get_tma_descriptor(), *tma_y.get_tma_descriptor(),
  };

  auto *tma_qkvqsksy = static_cast<cute::TmaDescriptor *>(tmas_ptr);
  kernels::update_batched_tma<Tin, decltype(tma_q), decltype(tma_k), decltype(tma_v),
                              decltype(tma_qs), decltype(tma_ks), decltype(tma_y)>
      <<<6, 32, 0, stream>>>(td_qkvqsksy0, tma_qkvqsksy, (Tin *)q_ptr, (Tin *)k_ptr, (Tin *)v_ptr,
                             (float *)qs_ptr, (float *)ks_ptr, (Tout *)y_ptr, num_batch, num_seq,
                             num_qk_dim, num_v_dim);

  auto kernel =
      kernels::gem3_kernel<Tout, Tin, kTileM, kTileN, kTileK, kTileV, TiledMmaQK, TiledMmaAV,
                           decltype(tma_q), decltype(tma_k), decltype(tma_v), decltype(tma_y),
                           decltype(tma_qs), decltype(tma_ks), decltype(slayout_q),
                           decltype(slayout_k), decltype(slayout_v), decltype(slayout_y),
                           decltype(slayout_qs), decltype(slayout_ks)>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
  kernel<<<grid, block, shm_size, stream>>>(tma_qkvqsksy, num_batch, num_seq, num_qk_dim,
                                            num_v_dim);
}

}  // namespace gem3
}  // namespace hpc
