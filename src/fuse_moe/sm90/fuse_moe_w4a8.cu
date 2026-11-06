// Copyright (C) 2026 Tencent.

#include "src/fuse_moe/fuse_moe.h"
#include "src/fuse_moe/fuse_moe_cp_async.h"
#include "src/fuse_moe/fuse_moe_w4a8.h"
#include "src/fuse_moe/sm90/w4a8_quant.cuh"
#include "src/group_gemm/group_gemm_w4a8.h"

namespace hpc {
namespace fuse_moe {
namespace {

constexpr int kExperts = 256, kHidden = 2048, kIntermediate = 512;

struct Workspace {
  size_t bytes = 0;
  unsigned char *base;
  int rows, tasks13, tasks2;
  bool paired_gateup;
  group_gemm_cp_async::W4A8GemmPlan plan13, plan2;
  void *a13, *a2, *y13, *y2;
  float *as13, *as2;
  int *row_indices, *topk_pos, *seqlens, *cu_seqlens, *tiles, *cu_tiles;
  int4 *task_map13, *task_map2;

  void *take(size_t size) {
    size_t offset = bytes;
    bytes += (size + 255) / 256 * 256;
    return base ? base + offset : nullptr;
  }

  Workspace(void *memory, int m, int topk) : base(static_cast<unsigned char *>(memory)) {
    rows = m * topk;
    plan13 = group_gemm_cp_async::group_gemm_w4a8_plan(rows, 2 * kIntermediate, kHidden);
    plan2 = group_gemm_cp_async::group_gemm_w4a8_plan(rows, kHidden, kIntermediate);
    paired_gateup = plan13.tile_n >= 256;
    // The existing count builder produces one shared M-tile prefix for both GEMMs.
    int active = rows < kExperts ? rows : kExperts;
    int max_tiles = (rows + active * (plan13.tile_m - 1)) / plan13.tile_m;
    tasks13 = max_tiles * (2 * kIntermediate / plan13.tile_n);
    tasks2 = max_tiles * (kHidden / plan2.tile_n);
    a13 = take(size_t(m) * kHidden);
    as13 = static_cast<float *>(take(size_t(m) * kHidden / 128 * sizeof(float)));
    a2 = take(size_t(rows) * kIntermediate);
    as2 = static_cast<float *>(take(size_t(rows) * kIntermediate / 128 * sizeof(float)));
    y13 = paired_gateup ? nullptr : take(size_t(rows) * 2 * kIntermediate * 2);
    y2 = take(size_t(rows) * kHidden * 2);
    row_indices = static_cast<int *>(take(size_t(rows) * sizeof(int)));
    topk_pos = static_cast<int *>(take(size_t(rows) * sizeof(int)));
    seqlens = static_cast<int *>(take(kExperts * sizeof(int)));
    cu_seqlens = static_cast<int *>(take((kExperts + 1) * sizeof(int)));
    tiles = static_cast<int *>(take(kExperts * sizeof(int)));
    cu_tiles = static_cast<int *>(take((kExperts + 1) * sizeof(int)));
    task_map13 = static_cast<int4 *>(take(size_t(tasks13) * sizeof(int4)));
    task_map2 = static_cast<int4 *>(take(size_t(tasks2) * sizeof(int4)));
  }
};

}  // namespace

size_t fuse_moe_w4a8_workspace_size(int num_seq, int top_k) {
  return Workspace(nullptr, num_seq, top_k).bytes;
}

bool fuse_moe_w4a8_async(void *output, const void *x, const void *w13, const uint8_t *s13,
                         const float *ss13, const void *w2, const uint8_t *s2, const float *ss2,
                         const int *ids, const float *weights, void *workspace, int num_seq,
                         int top_k, cudaStream_t stream) {
  if (num_seq == 0) return true;
  Workspace w(workspace, num_seq, top_k);
  w4a8_quantize_async<kHidden, false>(x, w.a13, w.as13, num_seq, stream,
                                      num_seq > 128 ? w.seqlens : nullptr);
  fuse_moe_cp_async::count_and_build_indices_async(
      ids, w.row_indices, w.topk_pos, w.seqlens, w.cu_seqlens, w.tiles, w.cu_tiles, w.task_map13,
      w.task_map2, 2 * kIntermediate / w.plan13.tile_n, kHidden / w.plan2.tile_n, w.tasks13,
      w.tasks2, num_seq, top_k, kExperts, 0, w.plan13.tile_m, stream);
  bool ok = group_gemm_cp_async::group_gemm_w4a8_async(
      w.paired_gateup ? w.a2 : w.y13, w.a13, w.as13, w13, s13, ss13, w.row_indices, w.seqlens,
      w.cu_seqlens, w.task_map13, w.tasks13, num_seq, w.rows, 2 * kIntermediate, kHidden, stream,
      w.paired_gateup ? w.as2 : nullptr);
  if (!ok) return false;
  if (!w.paired_gateup)
    w4a8_quantize_async<kIntermediate, true>(w.y13, w.a2, w.as2, w.rows, stream);
  ok = group_gemm_cp_async::group_gemm_w4a8_async(w.y2, w.a2, w.as2, w2, s2, ss2, nullptr,
                                                  w.seqlens, w.cu_seqlens, w.task_map2, w.tasks2,
                                                  w.rows, w.rows, kHidden, kIntermediate, stream);
  if (!ok) return false;
  reduce_async(output, w.y2, w.topk_pos, weights, nullptr, w.rows, num_seq, kHidden, top_k, false,
               stream);
  return cudaGetLastError() == cudaSuccess;
}

}  // namespace fuse_moe
}  // namespace hpc
