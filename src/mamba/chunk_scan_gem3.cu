// Copyright 2025 hpc-ops authors

#include <cuda.h>
#include <stdio.h>

#include <algorithm>

#include "cute/tensor.hpp"
#include "src/mamba/selective_state_scan.h"
#include "src/utils/utils.cuh"

namespace hpc {
namespace mamba {
namespace kernels {

template <typename Tout, typename Tin, int kChunkSize, int kTmaDescCount, int kTileM, int kTileN,
          int kTileK, int kTileV, typename TiledMmaQK, typename TiledMmaPV, typename TmaQ,
          typename TmaK, typename TmaV, typename TmaY, typename TmaXS, typename TmaQYS,
          typename TmaKYS, typename TmaPreY, typename TmaZ, typename SLayoutQ, typename SLayoutK,
          typename SLayoutV, typename SLayoutY, typename SLayoutQS, typename SLayoutKS,
          typename SLayoutPreY, typename SLayoutZ>
__global__ void chunk_scan_gem3_kernel(__grid_constant__ const TmaPreY tma_prey, const float* D_ptr,
                                       const cute::TmaDescriptor* tensormaps,
                                       const int* split_metadata, int batch_size, int chunk_size,
                                       uint32_t nheads, uint32_t head_dim, uint32_t ngroups,
                                       uint32_t dstate, uint32_t heads_per_group,
                                       uint32_t total_chunks) {
  using namespace cute;  // NOLINT

  int idx = threadIdx.x;
  int itile_m = blockIdx.y;
  int ichunk = blockIdx.z;
  int ihead = blockIdx.x;
  int igroup = ihead / heads_per_group;
  const int* cu_chunks = split_metadata + 2 * (batch_size + 1);
  const int* seqlens = split_metadata + 3 * (batch_size + 1);

  int ibatch = 0;
  int ichunk_in_batch = 0;
  for (int i = 1; i < batch_size + 1; i++) {
    if (ichunk < cu_chunks[i]) {
      ibatch = i - 1;
      ichunk_in_batch = ichunk - cu_chunks[ibatch];
      break;
    }
  }

  int elected = cute::elect_one_sync();
  int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
  int seqlen = seqlens[ibatch];
  float D = D_ptr[ihead];

  const int im_chunk_load = ichunk_in_batch * chunk_size / kTileM + itile_m;

  __shared__ uint64_t bar_q;
  __shared__ uint64_t bar_k;
  __shared__ uint64_t bar_v;
  __shared__ uint64_t bar_xs;
  __shared__ uint64_t bar_qys;
  __shared__ uint64_t bar_kys;
  __shared__ uint64_t bar_prey;
  __shared__ uint64_t bar_z;
  extern __shared__ uint8_t shm_data[] alignas(128);

  auto* shm_q = (Tin*)shm_data;
  auto* shm_k = ((Tin*)shm_q) + cosize(SLayoutQ{});
  auto* shm_v = ((Tin*)shm_k) + cosize(SLayoutK{});
  auto* shm_xs = (float*)(shm_v + cosize(SLayoutV{}));
  auto* shm_qys = (float*)(shm_xs + cosize(SLayoutKS{}));
  auto* shm_kys = (float*)(shm_qys + cosize(SLayoutQS{}));

  auto* shm_prey = (float*)(shm_data);
  auto* shm_z = (Tin*)(shm_prey + cosize(SLayoutPreY{}));  // Reuse All
  auto* shm_y = (Tout*)(shm_z + cosize(SLayoutZ{}));       // Reuse All

  TmaQ tma_q;
  TmaK tma_k;
  TmaV tma_v;
  TmaY tma_y;
  TmaXS tma_xs;
  TmaQYS tma_qys;
  TmaKYS tma_kys;
  TmaZ tma_z;

  // Tensor Q/K/V/Y
  auto gQ = tma_q.get_tma_tensor(make_shape(seqlen, dstate, ngroups));
  auto gK = tma_k.get_tma_tensor(make_shape(seqlen, dstate, ngroups));
  auto gV = tma_v.get_tma_tensor(make_shape(head_dim, seqlen, nheads));
  auto gY = tma_y.get_tma_tensor(make_shape(seqlen, head_dim, nheads));
  auto gXS = tma_xs.get_tma_tensor(make_shape(seqlen, nheads));    // xscale
  auto gQYS = tma_qys.get_tma_tensor(make_shape(seqlen, nheads));  // yscale
  auto gKYS = tma_kys.get_tma_tensor(make_shape(seqlen, nheads));  // yscale
  auto gPreY =
      tma_prey.get_tma_tensor(make_shape(chunk_size, head_dim, total_chunks - batch_size, nheads));
  auto gZ = tma_z.get_tma_tensor(make_shape(seqlen, head_dim, nheads));
  auto gP = make_tensor(make_gmem_ptr((float*)nullptr), make_shape(Int<kTileM>{}, Int<kTileN>{}),
                        make_stride(Int<kTileN>{}, Int<1>{}));
  auto gYY = make_tensor(make_gmem_ptr((Tout*)nullptr), make_shape(Int<kTileM>{}, Int<kTileV>{}),
                         make_stride(Int<kTileV>{}, Int<1>{}));

  // Tensor sQ/sK/sV
  auto sQ = make_tensor(make_smem_ptr(shm_q), SLayoutQ{});
  auto sK = make_tensor(make_smem_ptr(shm_k), SLayoutK{});
  auto sV = make_tensor(make_smem_ptr(shm_v), SLayoutV{});
  auto sY = make_tensor(make_smem_ptr(shm_y), SLayoutY{});
  auto sXS = make_tensor(make_smem_ptr(shm_xs), SLayoutKS{});
  auto sQYS = make_tensor(make_smem_ptr(shm_qys), SLayoutQS{});
  auto sKYS = make_tensor(make_smem_ptr(shm_kys), SLayoutKS{});
  auto sPreY = make_tensor(make_smem_ptr(shm_prey), SLayoutPreY{});
  auto sZ = make_tensor(make_smem_ptr(shm_z), SLayoutZ{});

  // Block Level tma
  auto btma_q = tma_q.get_slice(0);
  auto btma_k = tma_k.get_slice(0);
  auto btma_v = tma_v.get_slice(0);
  auto btma_y = tma_y.get_slice(0);
  auto btma_xs = tma_xs.get_slice(0);
  auto btma_qys = tma_qys.get_slice(0);
  auto btma_kys = tma_kys.get_slice(0);
  auto btma_prey = tma_prey.get_slice(0);
  auto btma_z = tma_z.get_slice(0);

  // Thread Level Tensor
  auto tQg = btma_q.partition_S(gQ);        // (TMA, TMA_M, TMA_K, batch)
  auto tKg = btma_k.partition_S(gK);        // (TMA, TMA_N, TMA_K, batch)
  auto tVg = btma_v.partition_S(gV);        // (TMA, TMA_V, TMA_N, batch)
  auto tXSg = btma_xs.partition_S(gXS);     // (TMA, TMA_M, batch)
  auto tQYSg = btma_qys.partition_S(gQYS);  // (TMA, TMA_N, batch)
  auto tKYSg = btma_kys.partition_S(gKYS);  // (TMA, TMA_N, batch)
  auto tPreYg = btma_prey.partition_S(gPreY);
  auto tZg = btma_z.partition_S(gZ);

  auto tQs = btma_q.partition_D(sQ);        // (TMA, _1, _1)
  auto tKs = btma_k.partition_D(sK);        // (TMA, _1, _1)
  auto tVs = btma_v.partition_D(sV);        // (TMA, _1, _1)
  auto tXSs = btma_xs.partition_D(sXS);     // (TMA, _1)
  auto tQYSs = btma_qys.partition_D(sQYS);  // (TMA, _1)
  auto tKYSs = btma_kys.partition_D(sKYS);  // (TMA, _1)
  auto tPreYs = btma_prey.partition_D(sPreY);
  auto tZs = btma_z.partition_D(sZ);

  TiledMmaQK tiled_mma_qk;
  TiledMmaPV tiled_mma_pv;

  auto thr_mma_qk = tiled_mma_qk.get_slice(idx);
  auto thr_mma_pv = tiled_mma_pv.get_slice(idx);

  auto tQs4r = thr_mma_qk.partition_A(sQ);
  auto tKs4r = thr_mma_qk.partition_B(sK);
  auto tVs4r = thr_mma_pv.partition_B(sV);

  auto tQr = thr_mma_qk.make_fragment_A(tQs4r);  // (MMA, MMA_M, MMA_K)
  auto tKr = thr_mma_qk.make_fragment_B(tKs4r);  // (MMA, MMA_N, MMA_K)
  auto tVr = thr_mma_pv.make_fragment_B(tVs4r);  // (MMA, MMA_V, MMA_N)

  auto tP = thr_mma_qk.partition_fragment_C(gP);
  auto tYr = thr_mma_pv.partition_fragment_C(gYY);

  auto gI = make_identity_tensor(gP.shape());
  auto tI = thr_mma_qk.partition_C(gI);

  auto gYI = make_identity_tensor(gYY.shape());
  auto tYI = thr_mma_pv.partition_C(gYI);

  auto tPreY = thr_mma_pv.partition_C(sPreY);
  auto tZ = thr_mma_pv.partition_C(sZ);

  const auto* tdq = &tensormaps[kTmaDescCount * ibatch + 7];
  const auto* tdk = &tensormaps[kTmaDescCount * ibatch + 8];
  const auto* tdv = &tensormaps[kTmaDescCount * ibatch + 9];
  const auto* tdxs = &tensormaps[kTmaDescCount * ibatch + 4];
  const auto* tdqys = &tensormaps[kTmaDescCount * ibatch + 5];
  const auto* tdkys = &tensormaps[kTmaDescCount * ibatch + 6];
  const auto* tdy = &tensormaps[kTmaDescCount * ibatch + 10];
  const auto* tdz = &tensormaps[kTmaDescCount * ibatch + 0];

  // Load Q
  if ((iwarp == 0) && elected) {
    initialize_barrier(bar_q, 1);
    set_barrier_transaction_bytes(bar_q, sizeof(Tin) * cosize(SLayoutQ{}));
    cute::copy(tma_q.with(tdq, bar_q), tQg(_, im_chunk_load, _, igroup), tQs(_, 0, _));
  }

  // init k/v barrier
  if ((iwarp == 0) && elected) {
    initialize_barrier(bar_k, 1);
    initialize_barrier(bar_v, 1);
    initialize_barrier(bar_xs, 1);
    initialize_barrier(bar_qys, 1);
    initialize_barrier(bar_kys, 1);
    initialize_barrier(bar_prey, 1);
    initialize_barrier(bar_z, 1);
  }

  __syncthreads();
  wait_barrier(bar_q, 0);

  auto layout_asC = thr_mma_qk.partition_C(gP).layout();
  auto layout_asA = thr_mma_pv.partition_A(gP).layout();
  auto tPA = make_tensor(tP.data(), left_inverse(layout_asC).compose(layout_asA));

  __syncthreads();
  clear(tYr);

  tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;

  int iseq_kv_offset = ichunk_in_batch * chunk_size / kTileN;
  int ipre_chunk = ichunk - ibatch - 1;
  constexpr int kSeqKVIters = kChunkSize / kTileN;
  int idiag = (itile_m + 1) * kTileM;
#pragma unroll 1
  for (int itile_seq_kv = 0; itile_seq_kv < kSeqKVIters; ++itile_seq_kv) {
    // casual mask, skip unused block
    int icol = itile_seq_kv * kTileN;
    if (icol >= idiag) {
      break;
    }

    // load k/scale/v
    if ((iwarp == 0) && elected) {
      // k
      set_barrier_transaction_bytes(bar_k, sizeof(Tin) * cosize(SLayoutK{}));
      cute::copy(tma_k.with(tdk, bar_k), tKg(_, iseq_kv_offset + itile_seq_kv, _, igroup),
                 tKs(_, 0, _));
      // qyscale
      if (itile_seq_kv == 0) {
        set_barrier_transaction_bytes(bar_qys, sizeof(float) * cosize(SLayoutQS{}));
        cute::copy(tma_qys.with(tdqys, bar_qys), tQYSg(_, im_chunk_load, ihead), tQYSs(_, 0));
      }
      // xscale / ykscale
      set_barrier_transaction_bytes(bar_xs, sizeof(float) * cosize(SLayoutKS{}));
      cute::copy(tma_xs.with(tdxs, bar_xs), tXSg(_, iseq_kv_offset + itile_seq_kv, ihead),
                 tXSs(_, 0));
      set_barrier_transaction_bytes(bar_kys, sizeof(float) * cosize(SLayoutKS{}));
      cute::copy(tma_kys.with(tdkys, bar_kys), tKYSg(_, iseq_kv_offset + itile_seq_kv, ihead),
                 tKYSs(_, 0));
      // v
      set_barrier_transaction_bytes(bar_v, sizeof(Tin) * cosize(SLayoutV{}));
      cute::copy(tma_v.with(tdv, bar_v), tVg(_, _, iseq_kv_offset + itile_seq_kv, ihead),
                 tVs(_, _, 0));
    }
    __syncthreads();
    wait_barrier(bar_k, itile_seq_kv % 2);

    // P = QK
    tiled_mma_qk.accumulate_ = GMMA::ScaleOut::One;
    clear(tP);

    warpgroup_fence_operand(tP);
    warpgroup_arrive();
    cute::gemm(tiled_mma_qk, tQr, tKr, tP);

    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(tP);
    if ((iwarp == 0) && elected) {
      if (icol + kTileN >= idiag) {
        if (ichunk_in_batch > 0) {
          set_barrier_transaction_bytes(bar_prey, sizeof(float) * cosize(SLayoutPreY{}));
          cute::copy(tma_prey.with(bar_prey), tPreYg(_, itile_m, _, ipre_chunk, ihead),
                     tPreYs(_, 0, _));
        }
        set_barrier_transaction_bytes(bar_z, sizeof(Tin) * cosize(SLayoutZ{}));
        cute::copy(tma_z.with(tdz, bar_z), tZg(_, im_chunk_load, _, ihead), tZs(_, 0, _));
      }
    }

    __syncthreads();
    if (itile_seq_kv == 0) {
      wait_barrier(bar_qys, 0);
    }
    wait_barrier(bar_xs, itile_seq_kv % 2);
    wait_barrier(bar_kys, itile_seq_kv % 2);
    // do causal mask
#pragma unroll
    for (int i = 0; i < size(tP); ++i) {
      int irow_local = get<0>(tI(i));
      int icol_local = get<1>(tI(i));

      int irow = irow_local + itile_m * kTileM;
      int icol = icol_local + itile_seq_kv * kTileN;

      if (icol > irow) {
        tP(i) = 0.f;
      } else {
        tP(i) = tP(i) * expf_ftz(tQYSs[irow_local] - tKYSs[icol_local]);
        tP(i) = tP(i) * tXSs[icol_local];
      }

      if (irow == icol) {
        tP(i) = tP(i) + D;
      }
    }

    // Y = PV
    auto tPAbf16 = make_tensor_like<Tin>(tPA);
#pragma unroll
    for (int i = 0; i < size(tPA); ++i) {
      tPAbf16(i) = (Tin)(tPA(i));
    }

    __syncthreads();
    wait_barrier(bar_v, itile_seq_kv % 2);

    warpgroup_fence_operand(tYr);
    warpgroup_arrive();
    cute::gemm(tiled_mma_pv, tPAbf16, tVr, tYr);
    warpgroup_commit_batch();
    warpgroup_wait<0>();

    warpgroup_fence_operand(tYr);
    __syncthreads();
  }

  warpgroup_fence_operand(tYr);
  auto tYr_bf16 = make_tensor_like<Tout>(tYr);
  // Epilogue: write register-C to global memory

  if (ichunk_in_batch > 0) {
    __syncthreads();
    wait_barrier(bar_prey, 0);
#pragma unroll
    for (int i = 0; i < size(tYr); ++i) {
      int irow_local = get<0>(tYI(i));
      tYr(i) = tYr(i) + tPreY(i) * expf_ftz(tQYSs[irow_local]);
    }
  }

  __syncthreads();
  wait_barrier(bar_z, 0);
#pragma unroll
  for (int i = 0; i < size(tYr); ++i) {
    tYr(i) = tYr(i) * silu(tZ(i));
    Tout v{tYr(i)};
    tYr_bf16(i) = v;
  }

  using R2SCopyAtomC = Copy_Atom<cute::SM90_U32x4_STSM_N, Tout>;
  auto r2s_tiled_copy = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma_pv);
  auto r2s_thr_copy = r2s_tiled_copy.get_slice(idx);

  auto tYr4s = r2s_thr_copy.retile_S(tYr_bf16);
  auto tYs4r = r2s_thr_copy.partition_D(sY);

  cute::copy(r2s_tiled_copy, tYr4s, tYs4r);
  __syncthreads();

  tma_store_fence();
  // using TMA to store
  if ((iwarp == 0) && elected) {
    auto cY = tma_y.get_tma_tensor(make_shape(seqlen, head_dim, nheads));
    auto btma_y = tma_y.get_slice(0);

    auto tYss = btma_y.partition_S(sY);  // (TMA, TMA_M, TMA_N)
    auto tYgg = btma_y.partition_D(cY);  // (TMA, TMA_M, TMA_N, b)

    cute::copy(tma_y.with(tdy), tYss(_, 0, 0), tYgg(_, im_chunk_load, 0, ihead));
  }
}

}  // namespace kernels

bool chunk_scan_gem3_async(void* out_ptr, const void* zxbcdt_ptr, const void* pre_y,
                           const float* xscale, const float* yscale, const float* D_ptr,
                           const void* tensormaps, const int* split_metadata, int batch_size,
                           int total_chunks, int total_padded_seqlen, int nheads, int ngroups,
                           int head_dim, int dstate, int zxbcdt_row_stride, int chunk_size,
                           int tma_desc_count, cudaStream_t stream) {
  using namespace cute;  // NOLINT
  using Tin = cute::bfloat16_t;
  using Tout = cute::bfloat16_t;

  constexpr int kChunkSize = 256;
  constexpr int kTmaDescCount = 11;
  constexpr int kTileM = 64;
  constexpr int kTileN = 64;
  constexpr int kTileK = 128;
  constexpr int kTileV = 80;

  if (chunk_size != kChunkSize || tma_desc_count != kTmaDescCount) {
    printf("chunk_scan_gem3_async Only support chunk_size:%d, tma_desc_count:%d\n", kChunkSize,
           kTmaDescCount);
    return false;
  }

  const auto* z_ptr = reinterpret_cast<const Tin*>(zxbcdt_ptr);
  const auto* v_ptr = z_ptr + nheads * head_dim;
  const auto* k_ptr = v_ptr + nheads * head_dim;
  const auto* q_ptr = k_ptr + ngroups * dstate;

  auto Q = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin*>(q_ptr)),
                       make_shape(kChunkSize, dstate, ngroups),
                       make_stride(zxbcdt_row_stride, Int<1>{}, dstate));
  auto K = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin*>(k_ptr)),
                       make_shape(kChunkSize, dstate, ngroups),
                       make_stride(zxbcdt_row_stride, Int<1>{}, dstate));
  auto V = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin*>(v_ptr)),
                       make_shape(head_dim, kChunkSize, nheads),
                       make_stride(Int<1>{}, zxbcdt_row_stride, head_dim));
  auto Y = make_tensor(make_gmem_ptr(reinterpret_cast<const Tout*>(out_ptr)),
                       make_shape(kChunkSize, head_dim, nheads),
                       make_stride(nheads * head_dim, Int<1>{}, head_dim));
  auto XS = make_tensor(make_gmem_ptr(reinterpret_cast<const float*>(xscale)),
                        make_shape(kChunkSize, nheads), make_stride(Int<1>{}, total_padded_seqlen));
  auto YS = make_tensor(make_gmem_ptr(reinterpret_cast<const float*>(yscale)),
                        make_shape(kChunkSize, nheads), make_stride(Int<1>{}, total_padded_seqlen));
  auto PreY =
      make_tensor(make_gmem_ptr(reinterpret_cast<const float*>(pre_y)),
                  make_shape(kChunkSize, head_dim, umax(total_chunks - batch_size, 1), nheads),
                  make_stride(head_dim, Int<1>{}, kChunkSize * head_dim,
                              kChunkSize * head_dim * (total_chunks - batch_size)));
  auto Z = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin*>(v_ptr)),
                       make_shape(kChunkSize, head_dim, nheads),
                       make_stride(zxbcdt_row_stride, Int<1>{}, head_dim));

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

  auto slayout_prey =
      tile_to_shape(GMMA::Layout_K_SW32_Atom<float>{}, make_shape(Int<kTileM>{}, Int<kTileV>{}));
  auto slayout_z =
      tile_to_shape(GMMA::Layout_K_SW32_Atom<Tin>{}, make_shape(Int<kTileN>{}, Int<kTileV>{}));

  auto tma_q = make_tma_copy(SM90_TMA_LOAD{}, Q, slayout_q);
  auto tma_k = make_tma_copy(SM90_TMA_LOAD{}, K, slayout_k);
  auto tma_v = make_tma_copy(SM90_TMA_LOAD{}, V, slayout_v);
  auto tma_y = make_tma_copy(SM90_TMA_STORE{}, Y, slayout_y);
  auto tma_xs = make_tma_copy(SM90_TMA_LOAD{}, XS, slayout_ks);
  auto tma_qys = make_tma_copy(SM90_TMA_LOAD{}, YS, slayout_qs);
  auto tma_kys = make_tma_copy(SM90_TMA_LOAD{}, YS, slayout_ks);
  auto tma_prey = make_tma_copy(SM90_TMA_LOAD{}, PreY, slayout_prey);
  auto tma_z = make_tma_copy(SM90_TMA_LOAD{}, Z, slayout_z);

  using TiledMmaQK =
      decltype(make_tiled_mma(SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{}));
  using TiledMmaPV =
      decltype(make_tiled_mma(SM90_64x80x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>{}));

  static_assert(kTileM >= get<0>(TiledMmaQK::Shape_MNK{}));
  static_assert(kTileM % get<0>(TiledMmaQK::Shape_MNK{}) == 0);
  static_assert(kTileM >= get<0>(TiledMmaPV::Shape_MNK{}));
  static_assert(kTileM % get<0>(TiledMmaPV::Shape_MNK{}) == 0);

  static_assert(kTileN >= get<1>(TiledMmaQK::Shape_MNK{}));
  static_assert(kTileN % get<1>(TiledMmaQK::Shape_MNK{}) == 0);
  static_assert(kTileN >= get<2>(TiledMmaPV::Shape_MNK{}));
  static_assert(kTileN % get<2>(TiledMmaPV::Shape_MNK{}) == 0);

  static_assert(kTileK >= get<2>(TiledMmaQK::Shape_MNK{}));
  static_assert(kTileK % get<2>(TiledMmaQK::Shape_MNK{}) == 0);

  static_assert(kTileV >= get<1>(TiledMmaPV::Shape_MNK{}));
  static_assert(kTileV % get<1>(TiledMmaPV::Shape_MNK{}) == 0);

  dim3 block(size(TiledMmaQK{}));
  dim3 grid(nheads, (kChunkSize + kTileM - 1) / kTileM, total_chunks);

  int shm_qkv = (cosize(slayout_q) + cosize(slayout_k) + cosize(slayout_v)) * sizeof(Tin);
  int shm_scale = (cosize(slayout_qs) + cosize(slayout_ks) * 2) * sizeof(float);
  int shm_prey = cosize(slayout_prey) * sizeof(float);
  int shm_z = cosize(slayout_z) * sizeof(Tin);
  int shm_y = cosize(slayout_y) * sizeof(Tout);
  int shm_size = std::max(shm_qkv + shm_scale, shm_y + shm_z + shm_prey);

  auto kernel = kernels::chunk_scan_gem3_kernel<
      Tout, Tin, kChunkSize, kTmaDescCount, kTileM, kTileN, kTileK, kTileV, TiledMmaQK, TiledMmaPV,
      decltype(tma_q), decltype(tma_k), decltype(tma_v), decltype(tma_y), decltype(tma_xs),
      decltype(tma_qys), decltype(tma_kys), decltype(tma_prey), decltype(tma_z),
      decltype(slayout_q), decltype(slayout_k), decltype(slayout_v), decltype(slayout_y),
      decltype(slayout_qs), decltype(slayout_ks), decltype(slayout_prey), decltype(slayout_z)>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
  kernel<<<grid, block, shm_size, stream>>>(
      tma_prey, D_ptr, reinterpret_cast<const cute::TmaDescriptor*>(tensormaps), split_metadata,
      batch_size, kChunkSize, nheads, head_dim, ngroups, dstate, nheads / ngroups, total_chunks);
  return true;
}

}  // namespace mamba
}  // namespace hpc
