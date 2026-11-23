# Copyright (C) 2026 Tencent.

import os
import threading

import torch
import triton
from triton.experimental.tle import language as tle
from triton.experimental.tle import require_tle

from hpc._amd.utils.validation import check_tensor, require

from .sampler import (
    _penalty_edges,
    _prepare_kernel,
    _reserve_rng,
    _sample_kernel,
    _temperature_finish,
    _temperature_kernel,
)


USE_TLE = os.getenv("HPC_TRITON_USE_TLE", "0") == "1"
_rng_states = {}
_rng_lock = threading.Lock()


SCHEMAS = (
    "fused_sampler(Tensor logits, Tensor? penalty_mask, Tensor? slot_id, Tensor? repetition_penalty, float repetition_penalty_val, Tensor? temperature, float temperature_val, int softmax_policy, Tensor? topk, int topk_val, Tensor? topp, float topp_val, int max_topk, Tensor? gumbel_noise=None, int seed=0) -> Tensor",
    "fused_sampler_temperature_sample(Tensor logits, Tensor? temperature, float temperature_val, Tensor? gumbel_noise=None, Tensor? draft_token_ids=None, int seed=0) -> Tensor",
)


def _rng_offset(reference, seed, external):
    if external:
        return None
    require(seed > 0, "seed must be > 0 when gumbel_noise is not provided")
    device = reference.device
    with _rng_lock:
        if device not in _rng_states:
            state = torch.zeros((), device=device, dtype=torch.int64)
            ready = torch.cuda.Event()
            ready.record()
            _rng_states[device] = state, ready
        state, ready = _rng_states[device]
    torch.cuda.current_stream(device).wait_event(ready)
    offset = torch.empty((), device=device, dtype=torch.int64)
    _reserve_rng[(1,)](state, offset, reference.numel(), num_warps=1)
    return offset


def _check_logits(logits):
    require(
        logits.dtype in (torch.float32, torch.bfloat16), "logits dtype must be float32 or bfloat16"
    )
    check_tensor(logits, logits, logits.dtype, "logits", contiguous=False)
    require(
        logits.ndim == 2 and logits.shape[1] == 120832, "unsupported vocab_size; expected 120832"
    )
    require(
        logits.stride(1) == 1 and logits.stride(0) >= logits.shape[1],
        "logits inner stride must be one and rows non-overlapping",
    )


def _vector(tensor, logits, dtype, name):
    if tensor is not None:
        check_tensor(tensor, logits, dtype, name)
        require(tensor.shape == (logits.shape[0],), f"{name} size must be [batch_size]")


def _check_noise(noise, logits):
    if noise is not None:
        check_tensor(noise, logits, torch.float32, "gumbel_noise")
        require(noise.shape == logits.shape, "gumbel_noise shape must match logits")


def temperature_sample(
    logits, temperature, temperature_val, gumbel_noise=None, draft_token_ids=None, seed=0
):
    _check_logits(logits)
    _vector(temperature, logits, torch.float32, "temperature")
    _check_noise(gumbel_noise, logits)
    if draft_token_ids is not None:
        require(draft_token_ids.dtype == torch.int64, "draft_token_ids dtype must be int64")
    _vector(draft_token_ids, logits, torch.int64, "draft_token_ids")
    require(temperature is not None or temperature_val > 0, "scalar temperature must be > 0")
    batch, width = logits.shape
    output = torch.empty((batch, 1), device=logits.device, dtype=torch.int32)
    if not batch:
        return output
    with torch.cuda.device(logits.device):
        if temperature is not None and not torch.cuda.is_current_stream_capturing():
            require(temperature.min().item() > 0, "every temperature tensor element must be > 0")
        offset = _rng_offset(logits, seed, gumbel_noise is not None)
        tiles = triton.cdiv(width, 4096)
        scores = torch.empty((batch, tiles), device=logits.device, dtype=torch.float32)
        tokens = torch.empty((batch, tiles), device=logits.device, dtype=torch.int32)
        _temperature_kernel[(tiles, batch)](
            logits,
            temperature,
            gumbel_noise,
            draft_token_ids,
            offset,
            scores,
            tokens,
            width,
            logits.stride(0),
            tiles,
            4096,
            temperature is not None,
            gumbel_noise is not None,
            draft_token_ids is not None,
            temperature_val,
            seed,
            num_warps=4,
            enable_fp_fusion=False,
        )
        _temperature_finish[(batch,)](
            scores, tokens, output, tiles, triton.next_power_of_2(tiles), num_warps=1
        )
    return output


def fused_sampler(
    logits,
    penalty_mask,
    slot_id,
    repetition_penalty,
    repetition_penalty_val,
    temperature,
    temperature_val,
    softmax_policy,
    topk,
    topk_val,
    topp,
    topp_val,
    max_topk,
    gumbel_noise=None,
    seed=0,
):
    _check_logits(logits)
    _check_noise(gumbel_noise, logits)
    require(softmax_policy in (0, 1, 2), "invalid softmax_policy")
    require(max_topk in (32, 64), "max_topk must be 32 or 64")
    require(
        (penalty_mask is None) == (slot_id is None),
        "penalty_mask and slot_id must both be provided or omitted",
    )
    for tensor, name in (
        (repetition_penalty, "repetition_penalty"),
        (temperature, "temperature"),
        (topp, "topp"),
    ):
        _vector(tensor, logits, torch.float32, name)
    if topk is not None:
        require(topk.dtype in (torch.int32, torch.int64), "topk dtype must be int32 or int64")
        _vector(topk, logits, topk.dtype, "topk")
    has_rp = repetition_penalty is not None or repetition_penalty_val > 0
    has_topk = topk is not None or topk_val > 0
    has_topp = topp is not None or topp_val > 0
    require(
        not has_rp or penalty_mask is not None, "repetition_penalty requires penalty_mask/slot_id"
    )
    require(not has_topp or has_topk, "topp requires topk")
    require(not has_topp or softmax_policy != 0, "topp requires softmax_policy != NONE")
    require(softmax_policy == 0 or has_topp, "softmax_policy requires topp")
    batch, width = logits.shape
    if penalty_mask is not None:
        check_tensor(penalty_mask, logits, torch.uint8, "penalty_mask")
        _vector(slot_id, logits, torch.int32, "slot_id")
        require(
            penalty_mask.ndim == 2
            and penalty_mask.shape[0] >= batch
            and penalty_mask.shape[1] >= triton.cdiv(width, 8),
            "invalid penalty_mask shape",
        )
    if USE_TLE:
        require_tle("amd", tle.cumsum)
    output = torch.empty((batch, 1), device=logits.device, dtype=torch.int32)
    if not batch:
        return output
    with torch.cuda.device(logits.device):
        offset = _rng_offset(logits, seed, gumbel_noise is not None)
        tiles = triton.cdiv(width, 4096)
        candidates = torch.empty((batch, tiles, max_topk), device=logits.device, dtype=torch.uint64)
        maxima = torch.empty((batch, tiles), device=logits.device, dtype=torch.float32)
        sums = torch.empty_like(maxima)
        pen_stride = penalty_mask.shape[1] if penalty_mask is not None else 0
        pen_bytes = penalty_mask.numel() if penalty_mask is not None else 0
        _prepare_kernel[(tiles, batch)](
            logits,
            candidates,
            penalty_mask,
            slot_id,
            repetition_penalty,
            temperature,
            maxima,
            sums,
            width,
            logits.stride(0),
            pen_stride,
            tiles,
            4096,
            max_topk,
            penalty_mask is not None,
            repetition_penalty is not None,
            temperature is not None,
            softmax_policy == 1,
            repetition_penalty_val,
            temperature_val,
            num_warps=4,
            enable_fp_fusion=False,
        )
        _sample_kernel[(batch,)](
            candidates,
            maxima,
            sums,
            topk,
            topp,
            gumbel_noise,
            offset,
            penalty_mask,
            slot_id,
            repetition_penalty,
            output,
            width,
            max_topk,
            tiles,
            triton.next_power_of_2(tiles),
            triton.next_power_of_2(tiles * max_topk),
            softmax_policy,
            topk is not None,
            topp is not None,
            gumbel_noise is not None,
            penalty_mask is not None,
            repetition_penalty is not None,
            pen_stride,
            pen_bytes,
            USE_TLE,
            topk_val,
            topp_val,
            repetition_penalty_val,
            seed,
            num_warps=4,
            enable_fp_fusion=False,
        )
        alignment = penalty_mask.data_ptr() % 4 if penalty_mask is not None else 0
        head_bytes = (4 - alignment) % 4
        tail_start = pen_bytes - (alignment + pen_bytes) % 4
        if penalty_mask is not None and (head_bytes or tail_start < pen_bytes):
            _penalty_edges[(1,)](
                penalty_mask,
                slot_id,
                output,
                repetition_penalty,
                batch,
                triton.next_power_of_2(batch),
                pen_stride,
                pen_bytes,
                head_bytes,
                tail_start,
                repetition_penalty is not None,
                repetition_penalty_val,
                num_warps=4,
            )
    return output


def register(library):
    implementations = {
        "fused_sampler": fused_sampler,
        "fused_sampler_temperature_sample": temperature_sample,
    }
    for schema in SCHEMAS:
        library.define(schema)
        name = schema.split("(", 1)[0]
        dispatch = "CompositeExplicitAutograd" if "workspace_size" in name else "CUDA"
        library.impl(name, implementations[name], dispatch)
