// Copyright 2026 hpc-ops authors

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include <tuple>

#include "src/group_gemm/sm90/cp_async/build_task_map.h"
#include "src/group_gemm/sm90/cp_async/group_gemm.h"
#include "src/utils/utils.h"

namespace hpc {
namespace group_gemm_cp_async {

// Must match the kTileM dispatch inside group_gemm_fp8_*_async.
static int pick_tile_m(int num_seq_per_group_avg) {
  if (num_seq_per_group_avg <= 8) {
    return 8;
  }
  if (num_seq_per_group_avg <= 16) {
    return 16;
  }
  if (num_seq_per_group_avg <= 32) {
    return 32;
  }
  if (num_seq_per_group_avg <= 48) {
    return 48;
  }
  return 64;
}

constexpr int kTileN = 64;

// Upper-bound the iblock space given the runtime-selected kTileM.
// Σ_i ceil(seq_i/kTileM) ≤ total_tokens/kTileM + num_group,
// so task_map_len = (total_tokens/kTileM + num_group) * num_tile_n.
static int compute_task_map_len(int total_tokens, int num_group, int n, int tile_m) {
  int num_tile_n = (n + kTileN - 1) / kTileN;
  int max_tile_m = total_tokens / tile_m + num_group;
  return max_tile_m * num_tile_n;
}

torch::Tensor group_gemm_fp8_entry(const torch::Tensor &x, const torch::Tensor &weight,
                                   const torch::Tensor &y_scale, const torch::Tensor &seqlens,
                                   const torch::Tensor &cu_seqlens, const torch::Tensor &tiles,
                                   const torch::Tensor &cu_tiles, bool use_task_map) {
  TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(x.is_contiguous() && weight.is_contiguous(), "inputs must be contiguous");
  TORCH_CHECK(x.scalar_type() == at::kFloat8_e4m3fn, "x must be float8_e4m3fn");
  TORCH_CHECK(weight.scalar_type() == at::kFloat8_e4m3fn, "weight must be float8_e4m3fn");
  TORCH_CHECK(y_scale.scalar_type() == at::kFloat, "y_scale must be float32");
  TORCH_CHECK(seqlens.size(0) <= 512, "num_group must be <= 512");
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  int total_tokens = x.size(0);
  int k = x.size(1);
  int n = weight.size(1);
  int num_group = seqlens.size(0);

  auto options = x.options();
  auto y = torch::empty({total_tokens, n}, options.dtype(torch::kBFloat16));

  auto *y_ptr = y.mutable_data_ptr();
  const auto *x_ptr = x.const_data_ptr();
  const auto *weight_ptr = weight.const_data_ptr();
  const auto *y_scale_ptr = y_scale.const_data_ptr();
  const auto *seqlens_ptr = seqlens.const_data_ptr();
  const auto *cu_seqlens_ptr = cu_seqlens.const_data_ptr();
  auto *tiles_ptr = tiles.mutable_data_ptr();
  auto *cu_tiles_ptr = cu_tiles.mutable_data_ptr();

  torch::Tensor task_map;
  void *task_map_ptr = nullptr;
  int task_map_len = 0;
  int num_seq_per_group_avg = (num_group > 0) ? (total_tokens / num_group) : 0;
  int tile_m = pick_tile_m(num_seq_per_group_avg);
  if (use_task_map) {
    task_map_len = compute_task_map_len(total_tokens, num_group, n, tile_m);
    task_map = torch::empty({task_map_len, 4}, options.dtype(torch::kInt32));
    task_map_ptr = task_map.mutable_data_ptr();
    cudaMemsetAsync(task_map_ptr, 0xFF, task_map_len * 4 * sizeof(int), stream);
    int num_tile_n = (n + kTileN - 1) / kTileN;
    launch_build_task_map(task_map_ptr, cu_tiles_ptr, tiles_ptr, num_group, num_tile_n,
                          /*use_pdl=*/false, stream);
  }

  group_gemm_fp8_multistage_async(
      y_ptr, x_ptr, weight_ptr, y_scale_ptr, seqlens_ptr, cu_seqlens_ptr, tiles_ptr, cu_tiles_ptr,
      task_map_ptr, task_map_len, total_tokens, n, k, num_group, num_seq_per_group_avg,
      /*use_pdl=*/false, stream);

  return y;
}

torch::Tensor group_gemm_fp8_scatter_entry(
    const torch::Tensor &x, const torch::Tensor &weight, const torch::Tensor &y_scale,
    const torch::Tensor &row_indices, const torch::Tensor &seqlens, const torch::Tensor &cu_seqlens,
    const torch::Tensor &tiles, const torch::Tensor &cu_tiles, bool use_task_map) {
  TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(x.is_contiguous() && weight.is_contiguous(), "inputs must be contiguous");
  TORCH_CHECK(x.scalar_type() == at::kFloat8_e4m3fn, "x must be float8_e4m3fn");
  TORCH_CHECK(weight.scalar_type() == at::kFloat8_e4m3fn, "weight must be float8_e4m3fn");
  TORCH_CHECK(y_scale.scalar_type() == at::kFloat, "y_scale must be float32");
  TORCH_CHECK(row_indices.scalar_type() == at::kInt, "row_indices must be int32");
  TORCH_CHECK(seqlens.size(0) <= 512, "group_gemm_fp8_scatter_cp_async: num_group must be <= 512");
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  int total_tokens = x.size(0);
  int k = x.size(1);
  int n = weight.size(1);
  int num_group = seqlens.size(0);

  auto options = x.options();
  auto y = torch::empty({total_tokens, n}, options.dtype(torch::kBFloat16));

  auto *y_ptr = y.mutable_data_ptr();
  const auto *x_ptr = x.const_data_ptr();
  const auto *weight_ptr = weight.const_data_ptr();
  const auto *y_scale_ptr = y_scale.const_data_ptr();
  const auto *row_indices_ptr = row_indices.const_data_ptr();
  const auto *seqlens_ptr = seqlens.const_data_ptr();
  const auto *cu_seqlens_ptr = cu_seqlens.const_data_ptr();
  auto *tiles_ptr = tiles.mutable_data_ptr();
  auto *cu_tiles_ptr = cu_tiles.mutable_data_ptr();

  torch::Tensor task_map;
  void *task_map_ptr = nullptr;
  int task_map_len = 0;
  int num_seq_per_group_avg = (num_group > 0) ? (total_tokens / num_group) : 0;
  int tile_m = pick_tile_m(num_seq_per_group_avg);
  if (use_task_map) {
    task_map_len = compute_task_map_len(total_tokens, num_group, n, tile_m);
    task_map = torch::empty({task_map_len, 4}, options.dtype(torch::kInt32));
    task_map_ptr = task_map.mutable_data_ptr();
    cudaMemsetAsync(task_map_ptr, 0xFF, task_map_len * 4 * sizeof(int), stream);
    int num_tile_n = (n + kTileN - 1) / kTileN;
    launch_build_task_map(task_map_ptr, cu_tiles_ptr, tiles_ptr, num_group, num_tile_n,
                          /*use_pdl=*/false, stream);
  }

  group_gemm_fp8_scatter_async(y_ptr, x_ptr, weight_ptr, y_scale_ptr, row_indices_ptr, seqlens_ptr,
                               cu_seqlens_ptr, tiles_ptr, cu_tiles_ptr, task_map_ptr, task_map_len,
                               total_tokens, n, k, num_group, num_seq_per_group_avg,
                               /*use_pdl=*/false, stream);

  return y;
}

}  // namespace group_gemm_cp_async
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "group_gemm_fp8_cp_async(Tensor x, Tensor weight, Tensor y_scale, Tensor seqlens, Tensor "
      "cu_seqlens, Tensor tiles, "
      "Tensor cu_tiles, bool use_task_map=False) -> "
      "(Tensor)");
  m.impl("group_gemm_fp8_cp_async", torch::kCUDA, &hpc::group_gemm_cp_async::group_gemm_fp8_entry);

  m.def(
      "group_gemm_fp8_scatter_cp_async(Tensor x, Tensor weight, Tensor y_scale, Tensor "
      "row_indices, Tensor seqlens, Tensor cu_seqlens, Tensor tiles, "
      "Tensor cu_tiles, bool use_task_map=False) -> "
      "(Tensor)");
  m.impl("group_gemm_fp8_scatter_cp_async", torch::kCUDA,
         &hpc::group_gemm_cp_async::group_gemm_fp8_scatter_entry);
}
