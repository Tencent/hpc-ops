// Copyright (C) 2026 Tencent.

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <time.h>
#include <torch/all.h>
#include <torch/library.h>

#include "src/sampler/sampler.h"

namespace hpc {
namespace sampler {
// fused_sampler: 2-kernel fused sampling op. Kernel in fused_sampler.cu,
// wrapper in hpc/sampler.py.
torch::Tensor fused_sampler_entry(
    const torch::Tensor& logits, std::optional<torch::Tensor> penalty_mask,
    std::optional<torch::Tensor> slot_id, std::optional<torch::Tensor> repetition_penalty,
    double repetition_penalty_val, std::optional<torch::Tensor> temperature, double temperature_val,
    int64_t softmax_policy, std::optional<torch::Tensor> topk, int64_t topk_val,
    std::optional<torch::Tensor> topp, double topp_val, int64_t max_topk,
    std::optional<torch::Tensor> gumbel_noise, int64_t seed) {
  TORCH_CHECK(logits.dim() == 2, "logits tensor must be dim == 2");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat32 || logits.scalar_type() == torch::kBFloat16,
              "logits dtype must be float32 or bfloat16");
  const int batch_size = logits.size(0);
  const int vocab_size = logits.size(1);
  const int64_t logits_row_stride = logits.stride(0);
  TORCH_CHECK(logits.stride(1) == 1,
              "fused_sampler: logits must have contiguous inner dim (stride(1)=1), "
              "got stride(1)=",
              logits.stride(1));
  TORCH_CHECK(logits_row_stride >= vocab_size,
              "fused_sampler: logits stride(0)=", logits_row_stride,
              " must be >= vocab_size=", vocab_size);
  // Row may be strided; require only row-major + inner stride 1 (not is_contiguous).

  TORCH_CHECK(softmax_policy >= 0 && softmax_policy <= 2,
              "softmax_policy must be one of 0(NONE)/1(BEFORE_TOPK)/2(AFTER_TOPK)");

  const bool has_penalty_mask = penalty_mask.has_value();
  const bool has_slot_id = slot_id.has_value();
  TORCH_CHECK(has_penalty_mask == has_slot_id,
              "penalty_mask and slot_id must both be provided or both be omitted");

  uint8_t* penalty_mask_ptr = nullptr;
  const int32_t* slot_id_ptr = nullptr;
  if (has_penalty_mask) {
    TORCH_CHECK(penalty_mask->is_contiguous(), "penalty_mask tensor must be contiguous");
    TORCH_CHECK(penalty_mask->scalar_type() == torch::kUInt8, "penalty_mask dtype must be uint8");
    TORCH_CHECK(penalty_mask->dim() == 2, "penalty_mask must be 2D [MAX_BS, ceil(V/8)]");
    TORCH_CHECK(penalty_mask->size(1) >= (vocab_size + 7) / 8,
                "penalty_mask dim 1 must be >= ", (vocab_size + 7) / 8, ", got ",
                penalty_mask->size(1));

    TORCH_CHECK(slot_id->is_contiguous(), "slot_id tensor must be contiguous");
    TORCH_CHECK(slot_id->scalar_type() == torch::kInt32, "slot_id dtype must be int32");
    TORCH_CHECK(slot_id->dim() == 1, "slot_id must be 1D");
    TORCH_CHECK(slot_id->size(0) == batch_size,
                "slot_id.size(0) must equal batch_size=", batch_size, ", got ", slot_id->size(0));
    TORCH_CHECK(penalty_mask->size(0) >= batch_size,
                "penalty_mask.size(0)(MAX_BS) must be >= batch_size=", batch_size);

    penalty_mask_ptr = penalty_mask->data_ptr<uint8_t>();
    slot_id_ptr = slot_id->data_ptr<int32_t>();
  }

  auto check_1d_float = [&](const std::optional<torch::Tensor>& t, const char* name) {
    if (!t.has_value()) {
      return;
    }
    TORCH_CHECK(t->is_contiguous(), name, " tensor must be contiguous");
    TORCH_CHECK(t->scalar_type() == torch::kFloat32, name, " dtype must be float32");
    TORCH_CHECK(t->dim() == 1, name, " tensor must be 1D");
    TORCH_CHECK(t->size(0) == batch_size, name, " size must be [batch_size=", batch_size,
                "], got [", t->size(0), "]");
  };
  check_1d_float(repetition_penalty, "repetition_penalty");
  check_1d_float(temperature, "temperature");
  check_1d_float(topp, "topp");

  const float* rp_arr =
      repetition_penalty.has_value() ? repetition_penalty->data_ptr<float>() : nullptr;
  const float* temp_arr = temperature.has_value() ? temperature->data_ptr<float>() : nullptr;
  const float* topp_arr = topp.has_value() ? topp->data_ptr<float>() : nullptr;

  const void* topk_ptr = nullptr;
  int topk_int_bytes = 0;
  if (topk.has_value()) {
    TORCH_CHECK(topk->is_contiguous(), "topk tensor must be contiguous");
    TORCH_CHECK(topk->dim() == 1, "topk tensor must be 1D");
    TORCH_CHECK(topk->size(0) == batch_size, "topk size must be [batch_size=", batch_size,
                "], got [", topk->size(0), "]");
    if (topk->scalar_type() == torch::kInt32) {
      topk_int_bytes = 4;
    } else if (topk->scalar_type() == torch::kInt64) {
      topk_int_bytes = 8;
    } else {
      TORCH_CHECK(false, "topk dtype must be int32 or int64");
    }
    topk_ptr = topk->data_ptr();
  }

  // Enable flags (for constraint checks).
  const bool has_rp = (rp_arr != nullptr) || (repetition_penalty_val > 0.f);
  const bool has_topk = (topk_ptr != nullptr) || (topk_val > 0);
  const bool has_topp = (topp_arr != nullptr) || (topp_val > 0.f);

  // Repetition penalty gating.
  TORCH_CHECK(!has_rp || has_penalty_mask,
              "repetition_penalty is enabled but penalty_mask/slot_id are missing");

  // topp requires topk + softmax.
  TORCH_CHECK(!has_topp || has_topk,
              "topp requires topk to be enabled (kernel does not support bare topp)");
  TORCH_CHECK(!has_topp || softmax_policy != 0,
              "topp requires softmax_policy != NONE (BEFORE_TOPK or AFTER_TOPK)");
  // softmax requires topp: without topp, softmax is order-preserving and would
  // be silent overhead. Reject the conflict.
  TORCH_CHECK(softmax_policy == 0 || has_topp,
              "softmax_policy != NONE requires topp to be enabled "
              "(softmax has no effect on sampling without topp)");

  TORCH_CHECK(max_topk == 32 || max_topk == 64, "max_topk must be 32 or 64, got ", max_topk);

  const float* gumbel_noise_ptr = nullptr;
  if (gumbel_noise.has_value()) {
    TORCH_CHECK(gumbel_noise->is_contiguous(), "gumbel_noise tensor must be contiguous");
    TORCH_CHECK(gumbel_noise->scalar_type() == torch::kFloat32,
                "gumbel_noise dtype must be float32");
    TORCH_CHECK(gumbel_noise->dim() == 2, "gumbel_noise must be 2D");
    TORCH_CHECK(gumbel_noise->size(0) == batch_size && gumbel_noise->size(1) == vocab_size,
                "gumbel_noise shape must be [", batch_size, ", ", vocab_size, "]");
    gumbel_noise_ptr = gumbel_noise->data_ptr<float>();
  }

  torch::Tensor token_ids =
      torch::empty({batch_size, 1}, torch::dtype(torch::kInt32).device(logits.device()));

  int logits_dtype = (logits.scalar_type() == torch::kFloat32) ? 0 : 1;
  auto stream = at::cuda::getCurrentCUDAStream(logits.get_device());

  // RNG seed required (> 0) only when no external gumbel_noise is supplied.
  uint64_t rng_seed = 0;
  if (gumbel_noise_ptr == nullptr) {
    TORCH_CHECK(
        seed > 0,
        "fused_sampler: seed must be > 0 when gumbel_noise is not provided, got seed=", seed);
    rng_seed = static_cast<uint64_t>(seed);
  }

  fused_sampler_async(
      token_ids.mutable_data_ptr<int32_t>(), logits.data_ptr(), logits_dtype, penalty_mask_ptr,
      slot_id_ptr, rp_arr, static_cast<float>(repetition_penalty_val), temp_arr,
      static_cast<float>(temperature_val), static_cast<int>(softmax_policy), topk_ptr,
      topk_int_bytes, static_cast<int>(topk_val), topp_arr, static_cast<float>(topp_val),
      gumbel_noise_ptr, batch_size, vocab_size, static_cast<int>(logits_row_stride),
      static_cast<int>(max_topk), rng_seed, stream);

  return token_ids;
}
// fused_sampler_temperature_sample: temperature-only fast-path. Kernel in
// src/sampler/fused_sampler_temperature.cu.
torch::Tensor fused_sampler_temperature_sample_entry(const torch::Tensor& logits,
                                                     std::optional<torch::Tensor> temperature,
                                                     double temperature_val,
                                                     std::optional<torch::Tensor> gumbel_noise,
                                                     std::optional<torch::Tensor> draft_token_ids,
                                                     int64_t seed) {
  TORCH_CHECK(logits.dim() == 2, "logits tensor must be dim == 2");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat32 || logits.scalar_type() == torch::kBFloat16,
              "logits dtype must be float32 or bfloat16");
  const int batch_size = logits.size(0);
  const int vocab_size = logits.size(1);
  const int64_t logits_row_stride = logits.stride(0);
  TORCH_CHECK(logits.stride(1) == 1,
              "fused_sampler_temperature_sample: logits must have contiguous inner dim "
              "(stride(1)=1), got stride(1)=",
              logits.stride(1));
  TORCH_CHECK(logits_row_stride >= vocab_size,
              "fused_sampler_temperature_sample: logits stride(0)=", logits_row_stride,
              " must be >= vocab_size=", vocab_size);

  const float* temperature_ptr = nullptr;
  if (temperature.has_value()) {
    TORCH_CHECK(temperature->is_contiguous(), "temperature tensor must be contiguous");
    TORCH_CHECK(temperature->scalar_type() == torch::kFloat32, "temperature dtype must be float32");
    TORCH_CHECK(temperature->dim() == 1, "temperature tensor must be 1D");
    TORCH_CHECK(temperature->size(0) == batch_size,
                "temperature size must be [batch_size=", batch_size, "], got [",
                temperature->size(0), "]");
    // The temperature kernel divides logits by each per-batch temperature with
    // no in-kernel t<=0 guard (unlike the main fused_sampler kernel). A zero or
    // negative entry would yield +-inf / NaN scores, so require every element
    // strictly > 0 here.
    TORCH_CHECK(temperature->numel() == 0 || temperature->min().item<float>() > 0.f,
                "fused_sampler_temperature_sample: every temperature tensor element must be > 0, "
                "got min=",
                (temperature->numel() == 0 ? 0.f : temperature->min().item<float>()));
    temperature_ptr = temperature->data_ptr<float>();
  } else {
    TORCH_CHECK(temperature_val > 0.f,
                "fused_sampler_temperature_sample: scalar temperature must be > 0, got ",
                temperature_val);
  }

  const float* gumbel_noise_ptr = nullptr;
  if (gumbel_noise.has_value()) {
    TORCH_CHECK(gumbel_noise->is_contiguous(), "gumbel_noise tensor must be contiguous");
    TORCH_CHECK(gumbel_noise->scalar_type() == torch::kFloat32,
                "gumbel_noise dtype must be float32");
    TORCH_CHECK(gumbel_noise->dim() == 2, "gumbel_noise must be 2D");
    TORCH_CHECK(gumbel_noise->size(0) == batch_size && gumbel_noise->size(1) == vocab_size,
                "gumbel_noise shape must be [", batch_size, ", ", vocab_size, "]");
    gumbel_noise_ptr = gumbel_noise->data_ptr<float>();
  }

  // ---- Optional draft mask [batch_size] int64: -1 = unmasked, else that
  //   token's logit is treated as -inf. Out-of-range values ignored by kernel.
  const int64_t* draft_token_ids_ptr = nullptr;
  if (draft_token_ids.has_value()) {
    TORCH_CHECK(draft_token_ids->is_contiguous(), "draft_token_ids tensor must be contiguous");
    TORCH_CHECK(draft_token_ids->scalar_type() == torch::kInt64,
                "draft_token_ids dtype must be int64");
    TORCH_CHECK(draft_token_ids->dim() == 1, "draft_token_ids must be 1D");
    TORCH_CHECK(draft_token_ids->size(0) == batch_size,
                "draft_token_ids size must be [batch_size=", batch_size, "], got [",
                draft_token_ids->size(0), "]");
    draft_token_ids_ptr = draft_token_ids->data_ptr<int64_t>();
  }

  torch::Tensor token_ids =
      torch::empty({batch_size, 1}, torch::dtype(torch::kInt32).device(logits.device()));

  uint64_t rng_seed = 0;
  if (gumbel_noise_ptr == nullptr) {
    TORCH_CHECK(seed > 0,
                "fused_sampler_temperature_sample: seed must be > 0 when gumbel_noise "
                "is not provided, got seed=",
                seed);
    rng_seed = static_cast<uint64_t>(seed);
  }

  const int logits_dtype = (logits.scalar_type() == torch::kFloat32) ? 0 : 1;
  auto stream = at::cuda::getCurrentCUDAStream(logits.get_device());

  fused_sampler_temperature_async(
      token_ids.mutable_data_ptr<int32_t>(), logits.data_ptr(), logits_dtype,
      static_cast<int>(logits_row_stride), temperature_ptr, static_cast<float>(temperature_val),
      gumbel_noise_ptr, draft_token_ids_ptr, batch_size, vocab_size, rng_seed, stream);
  return token_ids;
}

}  // namespace sampler
}  // namespace hpc

TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "fused_sampler(Tensor logits, Tensor? penalty_mask, Tensor? slot_id, "
      "Tensor? repetition_penalty, float repetition_penalty_val, "
      "Tensor? temperature, float temperature_val, "
      "int softmax_policy, "
      "Tensor? topk, int topk_val, "
      "Tensor? topp, float topp_val, "
      "int max_topk, "
      "Tensor? gumbel_noise=None, int seed=0) -> Tensor");
  m.impl("fused_sampler", torch::kCUDA, &hpc::sampler::fused_sampler_entry);
  m.def(
      "fused_sampler_temperature_sample(Tensor logits, Tensor? temperature, "
      "float temperature_val, Tensor? gumbel_noise=None, "
      "Tensor? draft_token_ids=None, "
      "int seed=0) -> Tensor");
  m.impl("fused_sampler_temperature_sample", torch::kCUDA,
         &hpc::sampler::fused_sampler_temperature_sample_entry);
}
