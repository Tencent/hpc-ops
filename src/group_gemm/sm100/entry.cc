// Copyright 2025 hpc-ops authors

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <torch/library.h>

#include <tuple>

#include "src/group_gemm/sm100/group_gemm.h"
#include "src/utils/utils.h"

namespace hpc {
namespace group_gemm {

torch::Tensor group_gemm_fp8_entry(const torch::Tensor &x, const torch::Tensor &weight,
                                   const torch::Tensor &seqlens, const torch::Tensor &cu_seqlens,
                                   const torch::Tensor &y_scale,
                                   const int64_t num_seq_per_group_avg,
                                   std::optional<torch::Tensor> output,
                                   std::optional<torch::Tensor> tma_desc,
                                   std::optional<torch::Tensor> task_map_workspace) {
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  TORCH_CHECK(x.device().is_cuda(), "x tensor must be cuda");
  TORCH_CHECK(weight.device().is_cuda(), "weight tensor must be cuda");
  TORCH_CHECK(seqlens.device().is_cuda(), "seqlens tensor must be cuda");
  TORCH_CHECK(cu_seqlens.device().is_cuda(), "cu_seqlens tensor must be cuda");
  TORCH_CHECK(x.dtype() == torch::kFloat8_e4m3fn && weight.dtype() == torch::kFloat8_e4m3fn,
              "x and weight dtype must be fp8_e4m3");
  TORCH_CHECK(seqlens.dtype() == torch::kInt32 && cu_seqlens.dtype() == torch::kInt32,
              "seqlens and cu_seqlens dtype must be int32");
  TORCH_CHECK(y_scale.dtype() == torch::kFloat32, "y_scale dtype must be float32");
  TORCH_CHECK(x.is_contiguous(), "x tensor a must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight tensor a must be contiguous");
  TORCH_CHECK(seqlens.size(0) == weight.size(0),
              "seqlens and weight must share the same num_group");
  TORCH_CHECK(x.size(1) == weight.size(2), "x and weight must share the same k");

  int m = x.size(0);
  int k = x.size(1);
  int n = weight.size(1);
  int num_group = seqlens.size(0);

  auto options = x.options();
  torch::Tensor y;
  if (output.has_value()) {
    y = output.value();
  } else {
    y = torch::empty({m, n}, options.dtype(torch::kBFloat16));
  }

  torch::Tensor tmas;
  bool update_tma = true;
  if (tma_desc.has_value()) {
    tmas = tma_desc.value();
    update_tma = false;
  } else {
    tmas = torch::empty({num_group * 2, 128}, options);
  }

  int num_waves = 0;
  torch::Tensor task_map;
  void *task_map_ptr = nullptr;

  if (num_seq_per_group_avg <= 8 && update_tma && task_map_workspace.has_value()) {
    num_waves = task_map_workspace.value().size(0);
    task_map_ptr = task_map_workspace.value().mutable_data_ptr();
  }

  torch::Tensor tiles = torch::empty({num_group}, options.dtype(torch::kInt32));
  torch::Tensor cu_tiles = torch::empty({num_group + 1}, options.dtype(torch::kInt32));

  const auto *x_ptr = x.const_data_ptr();
  const auto *weight_ptr = weight.const_data_ptr();
  const auto *seqlens_ptr = seqlens.const_data_ptr();
  const auto *cu_seqlens_ptr = cu_seqlens.const_data_ptr();
  const auto *yscale_ptr = y_scale.const_data_ptr();
  auto *tmas_ptr = tmas.mutable_data_ptr();
  auto *y_ptr = y.mutable_data_ptr();

  auto *tiles_ptr = tiles.mutable_data_ptr();
  auto *cu_tiles_ptr = cu_tiles.mutable_data_ptr();

  group_gemm_fp8_async(y_ptr, x_ptr, weight_ptr, seqlens_ptr, cu_seqlens_ptr, yscale_ptr, tmas_ptr,
                       tiles_ptr, cu_tiles_ptr, task_map_ptr, num_waves, num_group, m, n, k,
                       num_seq_per_group_avg, update_tma, false, stream);

  return y;
}

torch::Tensor group_gemm_blockwise_fp8_entry(
    const torch::Tensor &x, const torch::Tensor &weight, const torch::Tensor &seqlens,
    const torch::Tensor &cu_seqlens, const torch::Tensor &x_scale, const torch::Tensor &w_scale,
    const int64_t num_seq_per_group_avg, std::optional<torch::Tensor> output,
    std::optional<torch::Tensor> tma_desc, std::optional<torch::Tensor> task_map_workspace) {
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  TORCH_CHECK(x.device().is_cuda(), "x tensor must be cuda");
  TORCH_CHECK(weight.device().is_cuda(), "weight tensor must be cuda");
  TORCH_CHECK(seqlens.device().is_cuda(), "seqlens tensor must be cuda");
  TORCH_CHECK(cu_seqlens.device().is_cuda(), "cu_seqlens tensor must be cuda");
  TORCH_CHECK(x.is_contiguous(), "x tensor a must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight tensor a must be contiguous");
  TORCH_CHECK(x.dtype() == torch::kFloat8_e4m3fn && weight.dtype() == torch::kFloat8_e4m3fn,
              "x and weight dtype must be fp8_e4m3");
  TORCH_CHECK(seqlens.dtype() == torch::kInt32 && cu_seqlens.dtype() == torch::kInt32,
              "seqlens and cu_seqlens dtype must be int32");
  TORCH_CHECK(x_scale.dtype() == torch::kFloat32 && w_scale.dtype() == torch::kFloat32,
              "x_scale and w_scale dtype must be float32");
  TORCH_CHECK(seqlens.size(0) == weight.size(0),
              "seqlens and weight must share the same num_group");
  TORCH_CHECK(x.size(1) == weight.size(2), "x and weight must share the same k");
  TORCH_CHECK(w_scale.size(2) % 4 == 0, "w_scale must be multiple of 4");

  int m = x.size(0);
  int k = x.size(1);
  int n = weight.size(1);
  int m_pad = x_scale.size(1);
  int num_block_k_pad4 = w_scale.size(2);
  int num_group = seqlens.size(0);

  auto options = x.options();
  torch::Tensor y;
  if (output.has_value()) {
    y = output.value();
  } else {
    y = torch::empty({m, n}, options.dtype(torch::kBFloat16));
  }

  torch::Tensor tmas;
  bool update_tma = true;
  if (tma_desc.has_value()) {
    tmas = tma_desc.value();
    update_tma = false;
  } else {
    tmas = torch::empty({num_group * 2, 128}, options);
  }

  int num_waves = 0;
  torch::Tensor task_map;
  void *task_map_ptr = nullptr;

  if (num_seq_per_group_avg <= 8 && update_tma && task_map_workspace.has_value()) {
    num_waves = task_map_workspace.value().size(0);
    task_map_ptr = task_map_workspace.value().mutable_data_ptr();
  }

  torch::Tensor tiles = torch::empty({num_group}, options.dtype(torch::kInt32));
  torch::Tensor cu_tiles = torch::empty({num_group + 1}, options.dtype(torch::kInt32));

  const auto *x_ptr = x.const_data_ptr();
  const auto *weight_ptr = weight.const_data_ptr();
  const auto *seqlens_ptr = seqlens.const_data_ptr();
  const auto *cu_seqlens_ptr = cu_seqlens.const_data_ptr();
  const auto *xscale_ptr = x_scale.const_data_ptr();
  const auto *wscale_ptr = w_scale.const_data_ptr();
  auto *tmas_ptr = tmas.mutable_data_ptr();
  auto *y_ptr = y.mutable_data_ptr();

  auto *tiles_ptr = tiles.mutable_data_ptr();
  auto *cu_tiles_ptr = cu_tiles.mutable_data_ptr();

  group_gemm_blockwise_fp8_async(y_ptr, x_ptr, weight_ptr, seqlens_ptr, cu_seqlens_ptr, xscale_ptr,
                                 wscale_ptr, tmas_ptr, tiles_ptr, cu_tiles_ptr, task_map_ptr,
                                 num_waves, num_group, m, n, k, m_pad, num_block_k_pad4,
                                 num_seq_per_group_avg, update_tma, false, stream);

  return y;
}

torch::Tensor reformat_x_scale_entry(const torch::Tensor &x_scale, const torch::Tensor &seqlens,
                                     const torch::Tensor &cu_seqlens,
                                     std::optional<torch::Tensor> out_x_scale,
                                     const int64_t num_seq_per_group_avg) {
  auto stream = at::cuda::getCurrentCUDAStream(x_scale.get_device());
  TORCH_CHECK(x_scale.device().is_cuda(), "x_scale tensor must be cuda");
  TORCH_CHECK(seqlens.device().is_cuda(), "seqlens tensor must be cuda");
  TORCH_CHECK(cu_seqlens.device().is_cuda(), "cu_seqlens tensor must be cuda");
  TORCH_CHECK(x_scale.is_contiguous(), "x_scale tensor a must be contiguous");
  TORCH_CHECK(seqlens.is_contiguous(), "seqlens tensor a must be contiguous");
  TORCH_CHECK(cu_seqlens.is_contiguous(), "cu_seqlens tensor a must be contiguous");

  int m = x_scale.size(0);
  int n = x_scale.size(1);
  TORCH_CHECK(n == 16 || n == 32 || n == 56,
              "n must be 16, 32 or 56(for group gemm k=2048, k=4096 or k=7168)");

  int num_group = seqlens.size(0);
  int tilem = 0;
  // careful!!! here logit must be corresponds with group_gemm_blockwise_fp8_async
  if (num_seq_per_group_avg <= 8) {
    tilem = 8;
  } else if (num_seq_per_group_avg <= 16) {
    tilem = 16;
  } else if (num_seq_per_group_avg <= 32) {
    tilem = 32;
  } else if (num_seq_per_group_avg <= 48) {
    tilem = 48;
  } else {
    tilem = 64;
  }
  int num_seq_pad_per_group = m / num_group;
  TORCH_CHECK(num_seq_pad_per_group % tilem == 0,
              "The sparse pad length of x_scale for each group must be aligned to multiple of "
              "8/16/32/48/64 according to num_seq_per_group_avg");

  torch::Tensor output;
  if (out_x_scale.has_value()) {
    output = out_x_scale.value();
  } else {
    output = torch::empty({n, m}, x_scale.options());
  }

  const auto *xscale_ptr = x_scale.const_data_ptr();
  const auto *seqlens_ptr = seqlens.const_data_ptr();
  const auto *cu_seqlens_ptr = cu_seqlens.const_data_ptr();
  auto *output_ptr = output.mutable_data_ptr();

  reformat_x_scale_async(output_ptr, xscale_ptr, seqlens_ptr, cu_seqlens_ptr, num_group, m, n,
                         tilem, stream);

  return output;
}

torch::Tensor group_gemm_groupwise_w4a8_mma_entry(const torch::Tensor &x,
                                                  const torch::Tensor &weight,
                                                  const torch::Tensor &seqlens,
                                                  const torch::Tensor &cu_seqlens,
                                                  const torch::Tensor &y_scale, int64_t group_size,
                                                  std::optional<torch::Tensor> output) {
  return torch::empty({x.size(0), weight.size(1)}, x.options().dtype(torch::kBFloat16));
}

std::tuple<torch::Tensor, torch::Tensor> prepack_mxfp8_scale_entry(
    const std::optional<torch::Tensor> &sfx, const std::optional<torch::Tensor> &sfw,
    const std::optional<torch::Tensor> &cu_seqlens, const int64_t num_seq_per_group_avg) {
  TORCH_CHECK(sfx.has_value() || sfw.has_value(), "at least one of sfx/sfw must be provided");

  // Derive common parameters from whichever tensor is present.
  constexpr int kSfVec = 32;
  int k_sf = 0;
  int num_group = 0;
  int n = 0;
  int m_total = 0;
  torch::Device device = torch::kCUDA;

  if (sfw.has_value()) {
    TORCH_CHECK(sfw->device().is_cuda(), "sfw must be cuda");
    TORCH_CHECK(sfw->dtype() == torch::kUInt8, "sfw dtype must be uint8 (UE8M0 raw bits)");
    TORCH_CHECK(sfw->is_contiguous(), "sfw must be contiguous");
    TORCH_CHECK(sfw->dim() == 3, "sfw must be (num_group, n, k/kSfVec)");
    num_group = static_cast<int>(sfw->size(0));
    n = static_cast<int>(sfw->size(1));
    k_sf = static_cast<int>(sfw->size(2));
    TORCH_CHECK(n % 128 == 0, "n must be multiple of 128");
    device = sfw->device();
  }

  if (sfx.has_value()) {
    TORCH_CHECK(sfx->device().is_cuda(), "sfx must be cuda");
    TORCH_CHECK(sfx->dtype() == torch::kUInt8, "sfx dtype must be uint8 (UE8M0 raw bits)");
    TORCH_CHECK(sfx->is_contiguous(), "sfx must be contiguous");
    TORCH_CHECK(sfx->dim() == 2, "sfx must be (m_total, k/kSfVec)");
    TORCH_CHECK(cu_seqlens.has_value(), "cu_seqlens is required when sfx is provided");
    TORCH_CHECK(cu_seqlens->dtype() == torch::kInt32, "cu_seqlens dtype must be int32");
    m_total = static_cast<int>(sfx->size(0));
    if (k_sf == 0) {
      k_sf = static_cast<int>(sfx->size(1));
    } else {
      TORCH_CHECK(sfx->size(1) == k_sf, "sfx/sfw k_sf mismatch");
    }
    if (num_group == 0) {
      num_group = static_cast<int>(cu_seqlens->numel()) - 1;
    }
    TORCH_CHECK(cu_seqlens->numel() == num_group + 1, "cu_seqlens must have num_group+1 elements");
    device = sfx->device();
  }

  TORCH_CHECK(k_sf > 0, "k_sf must be > 0");
  int k = k_sf * kSfVec;
  TORCH_CHECK(k % 32 == 0, "k must be multiple of 32 (SF_VEC)");

  // For kTileM dispatch, we need n. When only sfx is provided, n is unknown;
  // use 128 (1SM path) as default since the caller knows is_2sm from context.
  int n_for_dispatch = (n > 0) ? n : 128;
  int kTileM = mxfp8_dispatch_kTileM(static_cast<int>(num_seq_per_group_avg), n_for_dispatch);
  bool is_smallm = (kTileM <= 128);
  int kSFAlignM = is_smallm ? 128 : 256;
  bool is_2sm = (n % 256 == 0) && (kTileM > 32) && (n > 0);

  auto stream = at::cuda::getCurrentCUDAStream(device.index());
  auto u8_opts = torch::dtype(torch::kUInt8).device(device);
  int64_t k_sf_padded = static_cast<int64_t>((k_sf + 3) / 4) * 4;

  // Allocate and prepack SFX (if provided)
  torch::Tensor sfx_packed;
  if (sfx.has_value()) {
    int sfx_max_tiles = m_total / kTileM + num_group;
    int64_t sfx_padded_max = static_cast<int64_t>(sfx_max_tiles) * kSFAlignM;
    int64_t sfx_buf_sz = sfx_padded_max * k_sf_padded;
    sfx_packed = torch::empty({sfx_buf_sz}, u8_opts);
    prepack_mxfp8_x_scale_async(sfx_packed.mutable_data_ptr(), sfx->const_data_ptr(),
                                cu_seqlens->const_data_ptr(), num_group, m_total, k, kTileM,
                                is_smallm, stream);
  } else {
    sfx_packed = torch::empty({0}, u8_opts);
  }

  // Allocate and prepack SFW (if provided)
  torch::Tensor sfw_packed;
  if (sfw.has_value()) {
    int64_t sfw_buf_sz = static_cast<int64_t>(num_group) * n * k_sf_padded;
    sfw_packed = torch::empty({sfw_buf_sz}, u8_opts);
    prepack_mxfp8_w_scale_async(sfw_packed.mutable_data_ptr(), sfw->const_data_ptr(), num_group, n,
                                k, is_2sm, stream);
  } else {
    sfw_packed = torch::empty({0}, u8_opts);
  }

  return std::make_tuple(sfx_packed, sfw_packed);
}

torch::Tensor group_gemm_mxfp8_entry(const torch::Tensor &x, const torch::Tensor &weight,
                                     const torch::Tensor &sfx_packed,
                                     const torch::Tensor &sfw_packed, const torch::Tensor &seqlens,
                                     const torch::Tensor &cu_seqlens,
                                     const int64_t num_seq_per_group_avg,
                                     std::optional<torch::Tensor> output,
                                     std::optional<torch::Tensor> tma_desc) {
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  TORCH_CHECK(x.device().is_cuda(), "x must be cuda");
  TORCH_CHECK(weight.device().is_cuda(), "weight must be cuda");
  TORCH_CHECK(seqlens.device().is_cuda(), "seqlens must be cuda");
  TORCH_CHECK(cu_seqlens.device().is_cuda(), "cu_seqlens must be cuda");
  TORCH_CHECK(sfx_packed.device().is_cuda() && sfw_packed.device().is_cuda(),
              "sfx_packed/sfw_packed must be cuda");
  TORCH_CHECK(x.dtype() == torch::kFloat8_e4m3fn, "x dtype must be fp8_e4m3");
  TORCH_CHECK(weight.dtype() == torch::kFloat8_e4m3fn || weight.dtype() == torch::kUInt8,
              "weight dtype must be fp8_e4m3 (mxfp8) or uint8 (mxfp4 packed)");
  TORCH_CHECK(sfx_packed.dtype() == torch::kUInt8 && sfw_packed.dtype() == torch::kUInt8,
              "prepacked sfx/sfw dtype must be uint8");
  TORCH_CHECK(seqlens.dtype() == torch::kInt32 && cu_seqlens.dtype() == torch::kInt32,
              "seqlens/cu_seqlens dtype must be int32");
  TORCH_CHECK(x.is_contiguous() && weight.is_contiguous(), "x/weight must be contiguous");
  TORCH_CHECK(seqlens.size(0) == weight.size(0),
              "seqlens and weight must share the same num_group");

  int k_x = static_cast<int>(x.size(1));
  int w_k = static_cast<int>(weight.size(2));
  bool is_fp4 = (w_k == k_x / 2);
  TORCH_CHECK(w_k == k_x || is_fp4, "weight.size(2) must equal k (mxfp8) or k/2 (mxfp4 packed)");
  TORCH_CHECK(k_x % 32 == 0, "k must be a multiple of 32 (SF_VEC)");
  // fp4(B) is sub-byte (e2m1); its K-dim TMA descriptor byte-alignment is stricter
  // than fp8, so k must be a multiple of 128 (otherwise cuTensorMapEncodeTiled
  // fails). fp8(B) has no such restriction (k%32 suffices).
  TORCH_CHECK(!is_fp4 || k_x % 128 == 0, "mxfp4 weight requires k to be a multiple of 128");

  int m = static_cast<int>(x.size(0));
  int k = static_cast<int>(x.size(1));
  int n = static_cast<int>(weight.size(1));
  int num_group = static_cast<int>(seqlens.size(0));

  auto options = x.options();
  torch::Tensor y;
  if (output.has_value()) {
    y = output.value();
  } else {
    y = torch::empty({m, n}, options.dtype(torch::kBFloat16));
  }

  torch::Tensor tmas;
  bool update_tma = true;
  if (tma_desc.has_value()) {
    tmas = tma_desc.value();
    update_tma = false;
  } else {
    // num_group * 2 TmaDescriptors (X / Y per group; SFX/SFW use static desc), 128B each
    tmas = torch::empty({num_group * 2, 128}, options);
  }

  torch::Tensor tiles = torch::empty({num_group}, options.dtype(torch::kInt32));
  torch::Tensor cu_tiles = torch::empty({num_group + 1}, options.dtype(torch::kInt32));

  group_gemm_mxfp8_async(y.mutable_data_ptr(), x.const_data_ptr(), weight.const_data_ptr(),
                         sfx_packed.const_data_ptr(), sfw_packed.const_data_ptr(),
                         seqlens.const_data_ptr(), cu_seqlens.const_data_ptr(),
                         tmas.mutable_data_ptr(), tiles.mutable_data_ptr(),
                         cu_tiles.mutable_data_ptr(), num_group, m, n, k,
                         static_cast<int>(num_seq_per_group_avg), update_tma, stream,
                         /*use_pdl=*/false, is_fp4);

  return y;
}

torch::Tensor group_gemm_cp_async_mxfp8_entry(
    const torch::Tensor &x, const torch::Tensor &weight, const torch::Tensor &sfx_packed,
    const torch::Tensor &sfw_packed, const torch::Tensor &seqlens, const torch::Tensor &cu_seqlens,
    const int64_t num_seq_per_group_avg, std::optional<torch::Tensor> x_row_map, int64_t x_num_rows,
    std::optional<torch::Tensor> output, std::optional<torch::Tensor> tma_desc) {
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  TORCH_CHECK(x.device().is_cuda() && weight.device().is_cuda(), "x/weight must be cuda");
  TORCH_CHECK(seqlens.device().is_cuda() && cu_seqlens.device().is_cuda(),
              "seqlens/cu_seqlens must be cuda");
  TORCH_CHECK(sfx_packed.device().is_cuda() && sfw_packed.device().is_cuda(),
              "sfx_packed/sfw_packed must be cuda");
  TORCH_CHECK(x.dtype() == torch::kFloat8_e4m3fn, "x dtype must be fp8_e4m3");
  TORCH_CHECK(weight.dtype() == torch::kFloat8_e4m3fn || weight.dtype() == torch::kUInt8,
              "weight dtype must be fp8_e4m3 (mxfp8) or uint8 (mxfp4 packed)");
  TORCH_CHECK(sfx_packed.dtype() == torch::kUInt8 && sfw_packed.dtype() == torch::kUInt8,
              "prepacked sfx/sfw dtype must be uint8");
  TORCH_CHECK(seqlens.dtype() == torch::kInt32 && cu_seqlens.dtype() == torch::kInt32,
              "seqlens/cu_seqlens dtype must be int32");
  TORCH_CHECK(x.is_contiguous() && weight.is_contiguous(), "x/weight must be contiguous");
  TORCH_CHECK(seqlens.size(0) == weight.size(0),
              "seqlens and weight must share the same num_group");
  // Determine fp8(== k) / fp4(== k/2) by the weight's third dimension.
  int k_x = static_cast<int>(x.size(1));
  int w_k = static_cast<int>(weight.size(2));
  bool is_fp4 = (w_k == k_x / 2);
  TORCH_CHECK(w_k == k_x || is_fp4, "weight.size(2) must equal k (mxfp8) or k/2 (mxfp4 packed)");
  TORCH_CHECK(k_x % 32 == 0, "k must be a multiple of 32 (SF_VEC)");
  // fp4(B) is sub-byte (e2m1); its K-dim TMA descriptor byte-alignment is stricter
  // than fp8, so k must be a multiple of 128 (otherwise cuTensorMapEncodeTiled
  // fails). fp8(B) has no such restriction (k%32 suffices).
  TORCH_CHECK(!is_fp4 || k_x % 128 == 0, "mxfp4 weight requires k to be a multiple of 128");

  int k = static_cast<int>(x.size(1));
  int n = static_cast<int>(weight.size(1));
  int num_group = static_cast<int>(seqlens.size(0));

  // m = total post-permutation rows (the shape the kernel pretends to see).
  // When x_row_map is provided, `x.size(0)` is the un-permuted row count
  // (== x_num_rows); the post-permutation count comes from cu_seqlens[-1].
  int m;
  const void *x_row_map_ptr = nullptr;
  int x_num_rows_int = static_cast<int>(x_num_rows);
  if (x_row_map.has_value()) {
    auto rm = x_row_map.value();
    TORCH_CHECK(rm.dtype() == torch::kInt32, "x_row_map dtype must be int32");
    TORCH_CHECK(rm.is_cuda() && rm.is_contiguous(), "x_row_map must be cuda + contiguous");
    x_row_map_ptr = rm.const_data_ptr();
    m = static_cast<int>(rm.size(0));  // post-permutation row count
    TORCH_CHECK(x_num_rows_int == static_cast<int>(x.size(0)),
                "x_num_rows must equal x.size(0) when row_map is provided");
  } else {
    m = static_cast<int>(x.size(0));
    x_num_rows_int = m;
  }

  auto options = x.options();
  torch::Tensor y;
  if (output.has_value()) {
    y = output.value();
  } else {
    y = torch::empty({m, n}, options.dtype(torch::kBFloat16));
  }

  torch::Tensor tmas;
  bool update_tma = true;
  if (tma_desc.has_value()) {
    tmas = tma_desc.value();
    update_tma = false;
  } else {
    tmas = torch::empty({num_group * 2, 128}, options);
  }

  torch::Tensor tiles = torch::empty({num_group}, options.dtype(torch::kInt32));
  torch::Tensor cu_tiles = torch::empty({num_group + 1}, options.dtype(torch::kInt32));

  group_gemm_cp_async_mxfp8_async(
      y.mutable_data_ptr(), x.const_data_ptr(), weight.const_data_ptr(),
      sfx_packed.const_data_ptr(), sfw_packed.const_data_ptr(), seqlens.const_data_ptr(),
      cu_seqlens.const_data_ptr(), tmas.mutable_data_ptr(), tiles.mutable_data_ptr(),
      cu_tiles.mutable_data_ptr(), num_group, m, n, k, static_cast<int>(num_seq_per_group_avg),
      update_tma, stream, x_row_map_ptr, x_num_rows_int, /*use_pdl=*/false, is_fp4);

  return y;
}

}  // namespace group_gemm
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "group_gemm_fp8(Tensor x, Tensor weight, Tensor seqlens, Tensor cu_seqlens, Tensor y_scale, "
      "int num_seq_per_group_avg, Tensor? output, Tensor? tma_desc, Tensor? task_map_workspace) -> "
      "(Tensor)");
  m.impl("group_gemm_fp8", torch::kCUDA, &hpc::group_gemm::group_gemm_fp8_entry);

  m.def(
      "group_gemm_blockwise_fp8(Tensor x, Tensor weight, Tensor seqlens, Tensor cu_seqlens, Tensor "
      "xscale, Tensor wscale,"
      "int num_seq_per_group_avg, Tensor? output, Tensor? tma_desc, Tensor? task_map_workspace) -> "
      "(Tensor)");
  m.impl("group_gemm_blockwise_fp8", torch::kCUDA,
         &hpc::group_gemm::group_gemm_blockwise_fp8_entry);

  m.def(
      "reformat_x_scale(Tensor x_scale, Tensor seqlens, Tensor cu_seqlens, "
      "Tensor? out_x_scale, int num_seq_per_group_avg) -> (Tensor)");
  m.impl("reformat_x_scale", torch::kCUDA, &hpc::group_gemm::reformat_x_scale_entry);

  m.def(
      "group_gemm_groupwise_w4a8_mma(Tensor x, Tensor weight, Tensor seqlens, Tensor cu_seqlens, "
      "Tensor y_scales, int group_size, Tensor? output) -> (Tensor)");
  m.impl("group_gemm_groupwise_w4a8_mma", torch::kCUDA,
         &hpc::group_gemm::group_gemm_groupwise_w4a8_mma_entry);

  m.def(
      "prepack_mxfp8_scale(Tensor? sfx, Tensor? sfw, Tensor? cu_seqlens, "
      "int num_seq_per_group_avg) "
      "-> (Tensor, Tensor)");
  m.impl("prepack_mxfp8_scale", torch::kCUDA, &hpc::group_gemm::prepack_mxfp8_scale_entry);

  m.def(
      "group_gemm_mxfp8(Tensor x, Tensor weight, Tensor sfx_packed, Tensor sfw_packed, "
      "Tensor seqlens, Tensor cu_seqlens, int num_seq_per_group_avg, "
      "Tensor? output, Tensor? tma_desc) -> (Tensor)");
  m.impl("group_gemm_mxfp8", torch::kCUDA, &hpc::group_gemm::group_gemm_mxfp8_entry);

  m.def(
      "group_gemm_cp_async_mxfp8(Tensor x, Tensor weight, Tensor sfx_packed, Tensor sfw_packed, "
      "Tensor seqlens, Tensor cu_seqlens, int num_seq_per_group_avg, "
      "Tensor? x_row_map, int x_num_rows, Tensor? output, Tensor? tma_desc) -> (Tensor)");
  m.impl("group_gemm_cp_async_mxfp8", torch::kCUDA,
         &hpc::group_gemm::group_gemm_cp_async_mxfp8_entry);
}
