// Copyright 2025 hpc-ops authors

#ifndef SRC_GROUP_GEMM_SM100_GEMM_CONFIG_H_
#define SRC_GROUP_GEMM_SM100_GEMM_CONFIG_H_

#include "cute/tensor.hpp"
#include "src/group_gemm/sm100/config.h"

namespace hpc {
namespace group_gemm {

using namespace cute;  // NOLINT

enum GroupGemmFunc {
  GROUP_GEMM_CP_ASYNC_FP8,
  GROUP_GEMM_CP_ASYNC_FP8_ACT_MUL,
  GROUP_GEMM_1SM_FP8,
  GROUP_GEMM_2SM_FP8,
};

template <int kNumSeqPerGroupAvg>
static constexpr auto group_gemm_cp_async_fp8_dispatch_selector() {
  if constexpr (kNumSeqPerGroupAvg <= 16) {
    return GroupGEMMFp8Config<cute::float_e4m3_t, cute::bfloat16_t, float,
                              /*kTile=*/16, 128, 128, 16,
                              /*kCluster=*/1, 1, 1, 1,
                              /*kStage=*/8, 4, 4,
                              /*kSwizzle=*/128, 128, 128,
                              /*kCopyBox=*/16, 128, 128, 16, 128,
                              /*kSwapAB=*/true, /*kMmaTileN=*/1>{};
  } else if constexpr (kNumSeqPerGroupAvg <= 32) {
    return GroupGEMMFp8Config<cute::float_e4m3_t, cute::bfloat16_t, float,
                              /*kTile=*/32, 128, 128, 32,
                              /*kCluster=*/1, 1, 1, 1,
                              /*kStage=*/6, 4, 4,
                              /*kSwizzle=*/128, 128, 128,
                              /*kCopyBox=*/32, 128, 128, 32, 128,
                              /*kSwapAB=*/true, /*kMmaTileN=*/1>{};
  } else if constexpr (kNumSeqPerGroupAvg <= 48) {
    return GroupGEMMFp8Config<cute::float_e4m3_t, cute::bfloat16_t, float,
                              /*kTile=*/48, 128, 128, 48,
                              /*kCluster=*/1, 1, 1, 1,
                              /*kStage=*/6, 4, 4,
                              /*kSwizzle=*/128, 128, 128,
                              /*kCopyBox=*/48, 128, 128, 48, 128,
                              /*kSwapAB=*/true, /*kMmaTileN=*/1>{};
  } else if constexpr (kNumSeqPerGroupAvg <= 64) {
    return GroupGEMMFp8Config<cute::float_e4m3_t, cute::bfloat16_t, float,
                              /*kTile=*/64, 128, 128, 32,
                              /*kCluster=*/1, 1, 1, 1,
                              /*kStage=*/6, 4, 4,
                              /*kSwizzle=*/128, 128, 128,
                              /*kCopyBox=*/64, 128, 128, 32, 128,
                              /*kSwapAB=*/true, /*kMmaTileN=*/1>{};
  } else {
    return GroupGEMMFp8Config<cute::float_e4m3_t, cute::bfloat16_t, float,
                              /*kTile=*/128, 128, 128, 64,
                              /*kCluster=*/1, 1, 1, 1,
                              /*kStage=*/6, 4, 2,
                              /*kSwizzle=*/128, 128, 128,
                              /*kCopyBox=*/128, 128, 128, 64, 128,
                              /*kSwapAB=*/true, /*kMmaTileN=*/1>{};
  }
}

template <int kNumSeqPerGroupAvg>
static constexpr auto group_gemm_cp_async_fp8_act_mul_dispatch_selector() {
  if constexpr (kNumSeqPerGroupAvg <= 16) {
    return GroupGEMMFp8Config<cute::float_e4m3_t, cute::float_e4m3_t, float,
                              /*kTile=*/16, 64, 128, 16,
                              /*kCluster=*/1, 1, 1, 1,
                              /*kStage=*/8, 4, 4,
                              /*kSwizzle=*/128, 128, 64,
                              /*kCopyBox=*/16, 16, 128, 16, 64,
                              /*kSwapAB=*/true, /*kMmaTileN=*/2>{};
  } else if constexpr (kNumSeqPerGroupAvg <= 32) {
    return GroupGEMMFp8Config<cute::float_e4m3_t, cute::float_e4m3_t, float,
                              /*kTile=*/32, 64, 128, 32,
                              /*kCluster=*/1, 1, 1, 1,
                              /*kStage=*/6, 4, 4,
                              /*kSwizzle=*/128, 128, 64,
                              /*kCopyBox=*/32, 16, 128, 32, 64,
                              /*kSwapAB=*/true, /*kMmaTileN=*/2>{};
  } else if constexpr (kNumSeqPerGroupAvg <= 48) {
    return GroupGEMMFp8Config<cute::float_e4m3_t, cute::float_e4m3_t, float,
                              /*kTile=*/48, 64, 128, 48,
                              /*kCluster=*/1, 1, 1, 1,
                              /*kStage=*/6, 4, 4,
                              /*kSwizzle=*/128, 128, 64,
                              /*kCopyBox=*/48, 16, 128, 48, 64,
                              /*kSwapAB=*/true, /*kMmaTileN=*/2>{};
  } else if constexpr (kNumSeqPerGroupAvg <= 64) {
    return GroupGEMMFp8Config<cute::float_e4m3_t, cute::float_e4m3_t, float,
                              /*kTile=*/64, 64, 128, 32,
                              /*kCluster=*/1, 1, 1, 1,
                              /*kStage=*/6, 4, 4,
                              /*kSwizzle=*/128, 128, 64,
                              /*kCopyBox=*/64, 16, 128, 32, 64,
                              /*kSwapAB=*/true, /*kMmaTileN=*/2>{};
  } else {
    return GroupGEMMFp8Config<cute::float_e4m3_t, cute::float_e4m3_t, float,
                              /*kTile=*/128, 64, 128, 64,
                              /*kCluster=*/1, 1, 1, 1,
                              /*kStage=*/6, 2, 2,
                              /*kSwizzle=*/128, 128, 64,
                              /*kCopyBox=*/128, 16, 128, 64, 64,
                              /*kSwapAB=*/true, /*kMmaTileN=*/2>{};
  }
}

template <int kNumSeqPerGroupAvg>
static constexpr auto group_gemm_1sm_fp8_dispatch_selector() {
  if constexpr (kNumSeqPerGroupAvg <= 16) {
    return GroupGEMMFp8Config<cute::float_e4m3_t, cute::bfloat16_t, float,
                              /*kTile=*/16, 128, 128, 16,
                              /*kCluster=*/1, 1, 1, 1,
                              /*kStage=*/2, 4, 4,
                              /*kSwizzle=*/128, 128, 128,
                              /*kCopyBox=*/16, 128, 128, 16, 128,
                              /*kSwapAB=*/true, /*kMmaTileN=*/1>{};
  } else if constexpr (kNumSeqPerGroupAvg <= 32) {
    return GroupGEMMFp8Config<cute::float_e4m3_t, cute::bfloat16_t, float,
                              /*kTile=*/32, 128, 128, 16,
                              /*kCluster=*/1, 1, 1, 1,
                              /*kStage=*/2, 4, 4,
                              /*kSwizzle=*/128, 128, 128,
                              /*kCopyBox=*/32, 128, 128, 16, 128,
                              /*kSwapAB=*/true, /*kMmaTileN=*/1>{};
  } else if constexpr (kNumSeqPerGroupAvg <= 48) {
    return GroupGEMMFp8Config<cute::float_e4m3_t, cute::bfloat16_t, float,
                              /*kTile=*/48, 128, 128, 48,
                              /*kCluster=*/1, 1, 1, 1,
                              /*kStage=*/6, 4, 4,
                              /*kSwizzle=*/128, 128, 128,
                              /*kCopyBox=*/48, 128, 128, 48, 128,
                              /*kSwapAB=*/true, /*kMmaTileN=*/1>{};
  } else if constexpr (kNumSeqPerGroupAvg <= 64) {
    return GroupGEMMFp8Config<cute::float_e4m3_t, cute::bfloat16_t, float,
                              /*kTile=*/64, 128, 128, 32,
                              /*kCluster=*/1, 1, 1, 1,
                              /*kStage=*/6, 4, 4,
                              /*kSwizzle=*/128, 128, 128,
                              /*kCopyBox=*/64, 128, 128, 32, 128,
                              /*kSwapAB=*/true, /*kMmaTileN=*/1>{};
  } else {
    return GroupGEMMFp8Config<cute::float_e4m3_t, cute::bfloat16_t, float,
                              /*kTile=*/128, 128, 128, 64,
                              /*kCluster=*/1, 1, 1, 1,
                              /*kStage=*/5, 4, 4,
                              /*kSwizzle=*/128, 128, 128,
                              /*kCopyBox=*/128, 128, 128, 64, 128,
                              /*kSwapAB=*/true, /*kMmaTileN=*/1>{};
  }
}

template <int kNumSeqPerGroupAvg>
static constexpr auto group_gemm_2sm_fp8_dispatch_selector() {
  if constexpr (kNumSeqPerGroupAvg <= 32) {
    return GroupGEMMFp8Config<cute::float_e4m3_t, cute::bfloat16_t, float,
                              /*kTile=*/32, 256, 128, 32,
                              /*kCluster=*/1, 2, 1, 2,
                              /*kStage=*/6, 4, 4,
                              /*kSwizzle=*/128, 128, 128,
                              /*kCopyBox=*/32, 256, 128, 32, 128,
                              /*kSwapAB=*/true, /*kMmaTileN=*/1>{};
  } else if constexpr (kNumSeqPerGroupAvg <= 64) {
    return GroupGEMMFp8Config<cute::float_e4m3_t, cute::bfloat16_t, float,
                              /*kTile=*/64, 256, 128, 64,
                              /*kCluster=*/1, 2, 1, 2,
                              /*kStage=*/6, 4, 4,
                              /*kSwizzle=*/128, 128, 128,
                              /*kCopyBox=*/64, 256, 128, 64, 128,
                              /*kSwapAB=*/true, /*kMmaTileN=*/1>{};
  } else if constexpr (kNumSeqPerGroupAvg <= 96) {
    return GroupGEMMFp8Config<cute::float_e4m3_t, cute::bfloat16_t, float,
                              /*kTile=*/96, 256, 128, 32,
                              /*kCluster=*/1, 2, 1, 2,
                              /*kStage=*/6, 4, 4,
                              /*kSwizzle=*/128, 128, 128,
                              /*kCopyBox=*/96, 256, 128, 32, 128,
                              /*kSwapAB=*/true, /*kMmaTileN=*/1>{};
  } else if constexpr (kNumSeqPerGroupAvg <= 128) {
    return GroupGEMMFp8Config<cute::float_e4m3_t, cute::bfloat16_t, float,
                              /*kTile=*/128, 256, 128, 32,
                              /*kCluster=*/1, 2, 1, 2,
                              /*kStage=*/6, 4, 4,
                              /*kSwizzle=*/128, 128, 128,
                              /*kCopyBox=*/128, 256, 128, 32, 128,
                              /*kSwapAB=*/true, /*kMmaTileN=*/1>{};
  } else if constexpr (kNumSeqPerGroupAvg <= 160) {
    return GroupGEMMFp8Config<cute::float_e4m3_t, cute::bfloat16_t, float,
                              /*kTile=*/160, 256, 128, 32,
                              /*kCluster=*/1, 2, 1, 2,
                              /*kStage=*/6, 3, 4,
                              /*kSwizzle=*/128, 128, 128,
                              /*kCopyBox=*/160, 256, 128, 32, 128,
                              /*kSwapAB=*/true, /*kMmaTileN=*/1>{};
  } else if constexpr (kNumSeqPerGroupAvg <= 192) {
    return GroupGEMMFp8Config<cute::float_e4m3_t, cute::bfloat16_t, float,
                              /*kTile=*/192, 256, 128, 32,
                              /*kCluster=*/1, 2, 1, 2,
                              /*kStage=*/6, 2, 4,
                              /*kSwizzle=*/128, 128, 128,
                              /*kCopyBox=*/192, 256, 128, 32, 128,
                              /*kSwapAB=*/true, /*kMmaTileN=*/1>{};
  } else {
    return GroupGEMMFp8Config<cute::float_e4m3_t, cute::bfloat16_t, float,
                              /*kTile=*/256, 256, 128, 64,
                              /*kCluster=*/1, 2, 1, 2,
                              /*kStage=*/6, 2, 2,
                              /*kSwizzle=*/128, 128, 128,
                              /*kCopyBox=*/256, 256, 128, 64, 128,
                              /*kSwapAB=*/true, /*kMmaTileN=*/1>{};
  }
}

template <GroupGemmFunc kGemmFunc, int kNumSeqPerGroupAvg>
static constexpr auto get_group_gemm_config() {
  if constexpr (kGemmFunc == GroupGemmFunc::GROUP_GEMM_CP_ASYNC_FP8) {
    return group_gemm_cp_async_fp8_dispatch_selector<kNumSeqPerGroupAvg>();
  } else if constexpr (kGemmFunc == GroupGemmFunc::GROUP_GEMM_CP_ASYNC_FP8_ACT_MUL) {
    return group_gemm_cp_async_fp8_act_mul_dispatch_selector<kNumSeqPerGroupAvg>();
  } else if constexpr (kGemmFunc == GroupGemmFunc::GROUP_GEMM_1SM_FP8) {
    return group_gemm_1sm_fp8_dispatch_selector<kNumSeqPerGroupAvg>();
  } else if constexpr (kGemmFunc == GroupGemmFunc::GROUP_GEMM_2SM_FP8) {
    return group_gemm_2sm_fp8_dispatch_selector<kNumSeqPerGroupAvg>();
  }
}

}  // namespace group_gemm
}  // namespace hpc

#endif  // SRC_GROUP_GEMM_SM100_GEMM_CONFIG_H_
