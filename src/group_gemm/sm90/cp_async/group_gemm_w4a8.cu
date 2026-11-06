// Copyright (C) 2026 Tencent.

#include <cuda.h>

#include <type_traits>

#include "cute/tensor.hpp"
#include "src/fuse_moe/sm90/w4a8_quant.cuh"
#include "src/group_gemm/group_gemm_w4a8.h"
#include "src/group_gemm/sm90/cp_async/common.cuh"
#include "src/group_gemm/sm90/cp_async/w4a8_operand.cuh"
#include "src/utils/utils.h"

namespace hpc {
namespace group_gemm_cp_async {
namespace kernels {

// This is the existing hpc scatter-copy geometry, with source rows cached once
// per output tile and one co-staged FP32 g128 scale for each activation row.
// Weight loading/decoding is the only new arithmetic operand policy.
template <class Config, class ActivationCopy, class SourceRows>
__device__ __forceinline__ void issue_w4a8_activation(ActivationCopy &destination,
                                                      float *shared_scale, const uint8_t *x,
                                                      const float *x_scale,
                                                      const SourceRows &sources, int scale_source,
                                                      int k_base, int k, int stage) {
  constexpr int kThreadsPerRow = Config::kTileK / 16;
  constexpr int kRowsPerIteration = 128 / kThreadsPerRow;
  constexpr int kIterations = (Config::kTileM + kRowsPerIteration - 1) / kRowsPerIteration;
  const int local_row = threadIdx.x / kThreadsPerRow;
  const int local_k = threadIdx.x % kThreadsPerRow * 16;
#pragma unroll
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    const int row = iteration * kRowsPerIteration + local_row;
    if (row < Config::kTileM) {
      const int source = sources(iteration);
      const int valid_bytes = source >= 0 ? 16 : 0;
      const int64_t offset = static_cast<int64_t>(source >= 0 ? source : 0) * k + k_base + local_k;
      void *target = &destination(cute::Int<0>{}, iteration, cute::Int<0>{}, stage);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                   :
                   : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(target))), "l"(x + offset),
                     "r"(valid_bytes)
                   : "memory");
    }
  }
  if (threadIdx.x < Config::kTileM) {
    const int row = threadIdx.x;
    const int source = scale_source;
    const int64_t offset =
        static_cast<int64_t>(source >= 0 ? source : 0) * (k / 128) + k_base / 128;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;"
                 :
                 : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(shared_scale + row))),
                   "l"(x_scale + offset), "r"(source >= 0 ? 4 : 0)
                 : "memory");
  }
}

template <class Config, bool kGather>
__global__ void __launch_bounds__(Config::kThreads,
                                  kGather &&Config::kTileM == 8 && Config::kTileN == 256 ? 3 : 2)
    group_gemm_w4a8_kernel(void *__restrict__ output, const void *__restrict__ input,
                           const float *__restrict__ input_scale,
                           const int32_t *__restrict__ weight, const uint8_t *__restrict__ relative,
                           const float *__restrict__ common_scale,
                           const int *__restrict__ row_indices, const int *__restrict__ seqlens,
                           const int *__restrict__ cu_seqlens, const int4 *__restrict__ task_map,
                           int task_map_len, int source_rows, float *__restrict__ output_scale) {
  using namespace cute;  // NOLINT
  using Tin = typename Config::Tin;
  using Tout = typename Config::Tout;
  using SmemA = typename Config::SmemLayoutA;
  using SmemC = typename Config::SmemLayoutCT;
  constexpr int kM = Config::kTileM;
  constexpr int kN = Config::kTileN;
  constexpr int kK = Config::kTileK;
  constexpr int kStages = Config::kStage;
  constexpr bool kPrefetchWeight = !kGather && kM == 8;
  constexpr bool kPairedGateUp = kGather && kN >= 256;
  static_assert(!kPairedGateUp || ((kN == 256 || kN == 512) && Config::kThreads == 256));
  constexpr int n = kGather ? 1024 : 2048;
  constexpr int k = kGather ? 2048 : 512;
  static_assert(kStages >= 2);

  extern __shared__ __align__(128) uint8_t memory[];
  auto *shared_a = reinterpret_cast<Tin *>(memory);
  auto *shared_w = memory + cosize(SmemA{});
  auto *shared_relative = shared_w + kStages * Config::kPackedStageBytes;
  auto *shared_scales =
      reinterpret_cast<float *>(shared_relative + kStages * Config::kRelativeStageBytes);
  auto *shared_sources = reinterpret_cast<int *>(memory + Config::kShmSize);
  auto a = make_tensor(make_smem_ptr(shared_a), SmemA{});
  auto c = make_tensor(make_smem_ptr(reinterpret_cast<Tout *>(memory)), SmemC{});
  typename Config::G2SCopyA copy_a;
  auto a_destination = copy_a.get_slice(threadIdx.x % 128).partition_D(a);

  int block = blockIdx.x;
  constexpr int num_k = k / kK;
  while (true) {
    if (block >= task_map_len) break;
    const int4 task = task_map[block];
    const int expert = task.x;
    if (expert < 0) break;
    const int tile_m = task.y;
    const int tile_n = task.z;
    block += gridDim.x;
    const int first = cu_seqlens[expert];
    const int valid_rows = seqlens[expert];
    const int row_base = first + tile_m * kM;
    const int n_base = tile_n * (kPairedGateUp ? kN / 2 : kN);
    for (int row = threadIdx.x; row < kM; row += Config::kThreads) {
      const bool valid = tile_m * kM + row < valid_rows;
      int source = valid ? row_base + row : -1;
      if constexpr (kGather) source = valid ? row_indices[source] : -1;
      shared_sources[row] = source >= 0 && source < source_rows ? source : -1;
    }
    __syncthreads();

    constexpr int kRowsPerCopy = 128 / (kK / 16);
    constexpr int kSourceIterations = (kM + kRowsPerCopy - 1) / kRowsPerCopy;
    auto activation_sources = make_tensor<int>(Int<kSourceIterations>{});
    if (threadIdx.x < 128) {
#pragma unroll
      for (int iteration = 0; iteration < kSourceIterations; ++iteration) {
        const int row = iteration * kRowsPerCopy + threadIdx.x / (kK / 16);
        activation_sources(iteration) = row < kM ? shared_sources[row] : -1;
      }
    }
    const int scale_source = threadIdx.x < kM ? shared_sources[threadIdx.x] : -1;

    auto run_math = [&](auto math_tag) {
      constexpr int kMathM = decltype(math_tag)::value;
      using Mma = decltype(make_tiled_mma(w4a8_detail::select_rs_mma<kMathM>(),
                                          typename Config::WarpGroupLayout{}));
      auto math_c =
          local_tile(c, make_tile(Int<kN>{}, Int<kMathM>{}), make_coord(Int<0>{}, Int<0>{}));
      Mma mma;
      auto thread_mma = mma.get_slice(threadIdx.x);
      auto coordinates =
          thread_mma.partition_C(make_identity_tensor(make_shape(Int<kN>{}, Int<kMathM>{})));
      // CUTE's partition_fragment_C allocates this same compact shape; no global
      // BF16 view is formed for the paired path's FP8 output buffer.
      auto total = make_tensor<float>(shape(coordinates));
      auto partial = make_tensor<float>(make_shape(shape<0>(total), Int<1>{}, Int<1>{}));
      auto decoded_weight = make_tensor<Tin>(make_shape(Int<16>{}, Int<1>{}, Int<4>{}));
      auto raw_weight = make_tensor<uint32_t>(Int<8>{});
      [[maybe_unused]] auto load_raw = [&](const uint8_t *packed, auto panel) {
#pragma unroll
        for (int plane = 0; plane < 2; ++plane) {
          const int tile = threadIdx.x / 128 + decltype(panel)::value * Config::kWarpGroups;
          const auto quad = reinterpret_cast<const uint4 *>(
              packed)[(plane * (kN / 64) + tile) * 128 + threadIdx.x % 128];
          const uint32_t words[4] = {quad.x, quad.y, quad.z, quad.w};
#pragma unroll
          for (int word = 0; word < 4; ++word) raw_weight(plane * 4 + word) = words[word];
        }
      };
      clear(total);

      auto prefetch = [&](int tile, int stage) {
        if (tile < num_k) {
          // The activation copy layout is the original 128-thread hpc layout.
          if (threadIdx.x < 128) {
            issue_w4a8_activation<Config>(a_destination, shared_scales + stage * kM,
                                          static_cast<const uint8_t *>(input), input_scale,
                                          activation_sources, scale_source, tile * kK, k, stage);
          }
          w4a8_detail::issue_weight_stage<Config, kPairedGateUp>(
              shared_w + stage * Config::kPackedStageBytes,
              shared_relative + stage * Config::kRelativeStageBytes, weight, relative, expert,
              n_base, tile * kK, n, k, threadIdx.x);
        }
        // Empty commits keep the same group count in the final pipeline drain.
        cp_async_fence();
      };
      constexpr int kPrefillStages = kStages - (kM == 40 ? 1 : 0);
#pragma unroll
      for (int stage = 0; stage < kPrefillStages; ++stage) prefetch(stage, stage);

      auto promote = [&](auto panel, int stage) {
        const float *scale = shared_scales + stage * kM;
#pragma unroll
        for (int value = 0; value < size<0>(total); ++value)
          total(value, panel, Int<0>{}) += partial(value, Int<0>{}, Int<0>{}) *
                                           scale[get<1>(coordinates(value, panel, Int<0>{}))];
      };

      if constexpr (kM != 40) {
        cp_async_wait<kStages - 1>();
        __syncthreads();
      }

      int stage = 0;
      constexpr int kUnroll = (!kGather && kM == 8) ? 4 : 1;
#pragma unroll kUnroll
      for (int tile = 0; tile < num_k; ++tile) {
        if constexpr (kM == 40) {
          cp_async_wait<kStages - 2>();
          __syncthreads();
        }
        const uint8_t *packed = shared_w + stage * Config::kPackedStageBytes;
        const uint8_t *scales = shared_relative + stage * Config::kRelativeStageBytes;
        auto a_stage = local_tile(a(_, _, stage), make_tile(Int<kMathM>{}, Int<kK>{}),
                                  make_coord(Int<0>{}, Int<0>{}));
        auto a_fragment = thread_mma.make_fragment_B(thread_mma.partition_B(a_stage));
        if constexpr (kPrefetchWeight) load_raw(packed, Int<0>{});
        cute::for_each(make_seq<Config::kPanelsPerWarpGroup>{}, [&](auto panel) {
          w4a8_detail::decode_weight_panel<Config, decltype(panel)::value, kPrefetchWeight>(
              packed, scales, decoded_weight, threadIdx.x, raw_weight.data());
          if constexpr (kPrefetchWeight &&
                        decltype(panel)::value + 1 < Config::kPanelsPerWarpGroup) {
            load_raw(packed, Int<decltype(panel)::value + 1>{});
            warpgroup_fence_operand(raw_weight);
          }
          warpgroup_fence_operand(decoded_weight);
          warpgroup_fence_operand(partial);
          mma.accumulate_ = GMMA::ScaleOut::Zero;
          warpgroup_arrive();
          gemm(mma, decoded_weight(_, _, Int<0>{}), a_fragment(_, _, Int<0>{}), partial);
          mma.accumulate_ = GMMA::ScaleOut::One;
          gemm(mma, decoded_weight(_, _, Int<1>{}), a_fragment(_, _, Int<1>{}), partial);
          gemm(mma, decoded_weight(_, _, Int<2>{}), a_fragment(_, _, Int<2>{}), partial);
          gemm(mma, decoded_weight(_, _, Int<3>{}), a_fragment(_, _, Int<3>{}), partial);
          warpgroup_commit_batch();
          if constexpr (decltype(panel)::value + 1 == Config::kPanelsPerWarpGroup) {
            if constexpr (kM == 40) {
              prefetch(tile + kStages - 1, stage == 0 ? kStages - 1 : stage - 1);
            } else if (tile + 1 < num_k) {
              cp_async_wait<kStages - 2>();
              __syncthreads();
            }
          }
          warpgroup_wait<0>();
          warpgroup_fence_operand(decoded_weight);
          warpgroup_fence_operand(partial);
          promote(panel, stage);
        });
        if constexpr (kM != 40) {
          if (tile + 1 < num_k) {
            __syncthreads();
            prefetch(tile + kStages, stage);
          }
        }
        stage = stage + 1 == kStages ? 0 : stage + 1;
      }
      cp_async_wait<0>();
      __syncthreads();

      // Preserve the BF16 gate/up boundary in the existing shared epilogue layout.
      auto narrowed = make_tensor_like<Tout>(total);
      const float common = common_scale[expert];
#pragma unroll
      for (int value = 0; value < size(total); ++value)
        narrowed(value) = Tout(total(value) * common);
      using MathR2SCopyAtomC = std::conditional_t<kMathM == 8, Copy_Atom<SM90_U16x4_STSM_T, Tout>,
                                                  typename Config::R2SCopyAtomC>;
      auto copy_c = make_tiled_copy_C(MathR2SCopyAtomC{}, mma);
      auto thread_copy_c = copy_c.get_slice(threadIdx.x);
      auto register_c = thread_copy_c.retile_S(narrowed);
      auto shared_c = thread_copy_c.partition_D(math_c);
      cute::copy(copy_c, register_c, shared_c);
      __syncthreads();
      if constexpr (kPairedGateUp) {
        fuse_moe::w4a8_kernels::quantize_paired_gateup<kM, kN>(
            math_c, static_cast<__nv_fp8_e4m3 *>(output), output_scale, row_base,
            valid_rows - tile_m * kM, tile_n);
      } else if (threadIdx.x < 128) {
        auto global_c = make_tensor(
            make_gmem_ptr(static_cast<Tout *>(output) + static_cast<int64_t>(first) * n),
            make_shape(n, valid_rows), make_stride(Int<1>{}, n));
        auto tile_c =
            local_tile(global_c, make_tile(Int<kN>{}, Int<kM>{}), make_coord(tile_n, tile_m));
        auto math_tile_c =
            local_tile(tile_c, make_tile(Int<kN>{}, Int<kMathM>{}), make_coord(Int<0>{}, Int<0>{}));
        typename Config::S2GCopyC store_c;
        auto thread_store_c = store_c.get_slice(threadIdx.x);
        auto source_c = thread_store_c.partition_S(math_c);
        auto destination_c = thread_store_c.partition_D(math_tile_c);
        auto identity_c =
            thread_store_c.partition_S(make_identity_tensor(make_shape(Int<kN>{}, Int<kMathM>{})));
        auto predicate_c = make_tensor<bool>(shape(identity_c));
#pragma unroll
        for (int value = 0; value < size(identity_c); ++value)
          predicate_c(value) = get<1>(identity_c(value)) < kMathM &&
                               tile_m * kM + get<1>(identity_c(value)) < valid_rows;
        cute::copy_if(store_c, predicate_c, source_c, destination_c);
      }
      __syncthreads();
    };
    if constexpr (kGather && kM == 40) {
      const int remaining_rows = valid_rows - tile_m * kM;
      const int tile_rows = remaining_rows < kM ? remaining_rows : kM;
      if (tile_rows <= 32)
        run_math(Int<32>{});
      else
        run_math(Int<40>{});
    } else if constexpr (kM == 16) {
      const int remaining_rows = valid_rows - tile_m * kM;
      const int tile_rows = remaining_rows < kM ? remaining_rows : kM;
      if (tile_rows <= 8)
        run_math(Int<8>{});
      else
        run_math(Int<16>{});
    } else {
      run_math(Int<kM>{});
    }
  }
}

}  // namespace kernels

W4A8GemmPlan group_gemm_w4a8_plan(int expanded_rows, int n, int k) {
  if (expanded_rows < 0 || !((n == 1024 && k == 2048) || (n == 2048 && k == 512))) return {0, 0, 0};
  // Share one M-tile prefix while specializing N/stages to each projection.
  const int tile_m = expanded_rows <= 512 ? 8 : expanded_rows <= 4096 ? 16 : 40;
  if (n == 1024) {
    if (tile_m == 40) return {40, 256, 5};
    if (tile_m == 16) return {16, 512, 3};
    if (expanded_rows <= 8) return {8, 64, 4};
    if (expanded_rows <= 88) return {8, 128, 4};
    return {8, 256, 4};
  }
  return {tile_m, tile_m == 40 ? 256 : 128, 4};
}

bool group_gemm_w4a8_async(void *output, const void *x, const float *x_scale, const void *w4,
                           const uint8_t *relative, const float *common_scale,
                           const int *row_indices, const int *seqlens, const int *cu_seqlens,
                           const void *task_map, int task_map_len, int source_rows, int output_rows,
                           int n, int k, cudaStream_t stream, float *output_scale) {
  using namespace cute;  // NOLINT
  if (output_rows == 0) return true;
  const W4A8GemmPlan plan = group_gemm_w4a8_plan(output_rows, n, k);
  if (plan.tile_m == 0 || source_rows <= 0 || output_rows < 0 || task_map == nullptr ||
      task_map_len <= 0 || ((row_indices != nullptr) != (n == 1024)) ||
      ((output_scale != nullptr) != (n == 1024 && plan.tile_n >= 256)))
    return false;
  bool launched = false;
  auto launch = [&](auto m_tag, auto n_tag, auto stages_tag, auto gather_tag) {
    constexpr int kM = decltype(m_tag)::value;
    constexpr int kN = decltype(n_tag)::value;
    constexpr int kStages = decltype(stages_tag)::value;
    constexpr bool kGather = decltype(gather_tag)::value;
    using Config = w4a8_detail::W4A8GemmConfig<bfloat16_t, kM, kN, 128, kStages>;
    const int shared = Config::kShmSize + kM * sizeof(int);
    auto kernel = kernels::group_gemm_w4a8_kernel<Config, kGather>;
    if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared) !=
        cudaSuccess)
      return;
    // Match resident-CTA waves for the two larger native configurations.
    const int multiplier = kM == 40 ? 6 : (kGather && kM == 16 ? 4 : grid_multiplier(kM));
    const int full_grid = get_sm_count() * multiplier;
    dim3 grid(task_map_len < full_grid ? task_map_len : full_grid);
    kernel<<<grid, Config::kThreads, shared, stream>>>(
        output, x, x_scale, static_cast<const int32_t *>(w4), relative, common_scale, row_indices,
        seqlens, cu_seqlens, static_cast<const int4 *>(task_map), task_map_len, source_rows,
        output_scale);
    launched = cudaGetLastError() == cudaSuccess;
  };
  if (row_indices) {
    if (plan.tile_m == 40)
      launch(Int<40>{}, Int<256>{}, Int<5>{}, std::true_type{});
    else if (plan.tile_m == 16)
      launch(Int<16>{}, Int<512>{}, Int<3>{}, std::true_type{});
    else if (plan.tile_n == 64)
      launch(Int<8>{}, Int<64>{}, Int<4>{}, std::true_type{});
    else if (plan.tile_n == 128)
      launch(Int<8>{}, Int<128>{}, Int<4>{}, std::true_type{});
    else
      launch(Int<8>{}, Int<256>{}, Int<4>{}, std::true_type{});
  } else {
    if (plan.tile_m == 40)
      launch(Int<40>{}, Int<256>{}, Int<4>{}, std::false_type{});
    else if (plan.tile_m == 16)
      launch(Int<16>{}, Int<128>{}, Int<4>{}, std::false_type{});
    else
      launch(Int<8>{}, Int<128>{}, Int<4>{}, std::false_type{});
  }
  return launched;
}

}  // namespace group_gemm_cp_async
}  // namespace hpc
