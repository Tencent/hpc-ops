# Copyright (C) 2026 Tencent.
"""Tests covering the fused sampler operators exposed via ``import hpc``.

1. ``hpc.fused_sampler`` — the 2-kernel fused sampler (top-``max_topk`` bounded)
   covering rep_penalty/temperature/softmax_policy/topk/topp/Gumbel-max/penalty
   writeback. With injected ``gumbel_noise`` it is bit-exact against the pure
   torch reference ``ref_fused_sampler``.
2. The temperature fast-path: ``hpc.fused_sampler(logits, temperature=...)`` with
   no other feature auto-dispatches to ``fused_sampler_temperature_sample``;
   it supports injected ``gumbel_noise`` (bit-exact) and a ``[B]`` int64
   ``draft_token_ids`` mask.
"""
import os
import sys
from pathlib import Path
from typing import Optional

import pytest

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import torch

import hpc
from hpc.sampler import SoftmaxPolicy


# Vocab sizes that the cluster (fused) path supports (see the .cu launcher).
CLUSTER_VOCAB_SIZES = [120832]


# ===========================================================================
# (1) fused_sampler: pure-torch reference + bit-exact tests.
# ===========================================================================
def _gumbel0_like(logits: torch.Tensor) -> torch.Tensor:
    """External Gumbel(0) noise matching the kernel convention:
    g = -log(-log(U)), U ~ Uniform(0, 1].

    Always float32 with the logits' [B, V] shape — the op requires the
    gumbel_noise buffer to be float32 regardless of the logits dtype.
    """
    u = torch.rand(logits.shape, dtype=torch.float32, device=logits.device)
    u = u.clamp_min_(1e-20)
    return -(-u.log()).log()


def ref_fused_sampler(
    logits: torch.Tensor,
    *,
    penalty_mask: torch.Tensor | None = None,
    slot_id: torch.Tensor | None = None,
    repetition_penalty: float | torch.Tensor = 0.0,
    temperature: float | torch.Tensor = 0.0,
    softmax_policy: SoftmaxPolicy = SoftmaxPolicy.NONE,
    topk: int | torch.Tensor = 0,
    topp: float | torch.Tensor = 0.0,
    max_topk: int = 32,
    gumbel_noise: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Python reference that mimics the kernel's numerical flow exactly.

    Returns (token_ids[B,1], expected penalty_mask after writeback-or-None).

    IMPORTANT: the kernel always collapses the vocab to the top-``max_topk``
    candidates (via cluster topk) before sampling, regardless of the user-
    supplied ``topk``. If the caller omits ``topk`` the kernel still sees
    ``effK = max_topk``, so the reference mirrors that.
    """
    device = logits.device
    B, V = logits.shape
    work = logits.float().clone()

    def _as_tensor(x, dtype):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype)
        return torch.full((B,), float(x), device=device, dtype=dtype)

    rp = _as_tensor(repetition_penalty, torch.float32)
    temp = _as_tensor(temperature, torch.float32)
    tp = _as_tensor(topp, torch.float32)
    if isinstance(topk, torch.Tensor):
        tk = topk.to(device=device, dtype=torch.int64)
    else:
        tk = torch.full((B,), int(topk), device=device, dtype=torch.int64)

    # 1. Repetition penalty: where mask bit set, logit *= (>0 ? 1/rp : rp).
    if penalty_mask is not None and slot_id is not None:
        for b in range(B):
            if rp[b].item() <= 0:
                continue
            row = penalty_mask[int(slot_id[b].item())]
            # Expand bits.
            bits = torch.zeros(row.numel() * 8, dtype=torch.bool, device=device)
            for bit in range(8):
                bits[bit::8] = ((row >> bit) & 1).bool()
            keep = bits[:V]
            r = rp[b].item()
            inv_r = 1.0 / r
            wb = work[b]
            pos = keep & (wb > 0)
            neg = keep & (wb <= 0)
            wb[pos] = wb[pos] * inv_r
            wb[neg] = wb[neg] * r

    # 2. Temperature.
    for b in range(B):
        t = temp[b].item()
        if t > 0:
            work[b] = work[b] / t

    # 3. Optional softmax1 over full vocab.
    if softmax_policy == SoftmaxPolicy.BEFORE_TOPK:
        work = torch.softmax(work, dim=-1)

    # 5. Topk (or full vocab).
    tokens_out = torch.empty((B, 1), dtype=torch.int32, device=device)
    for b in range(B):
        k_b = int(tk[b].item())
        # Kernel clamps: 0 or out-of-range -> max_topk.
        if k_b <= 0 or k_b > max_topk:
            k_b = max_topk
        vals, idx = torch.topk(work[b], k=k_b, largest=True, sorted=True)

        # softmax2 over top-K if AFTER_TOPK.
        if softmax_policy == SoftmaxPolicy.AFTER_TOPK:
            probs = torch.softmax(vals, dim=-1)
            val_for_gumbel = probs.log()
        elif softmax_policy == SoftmaxPolicy.BEFORE_TOPK:
            probs = vals  # already probs
            val_for_gumbel = torch.where(
                probs > 0, probs.log(), torch.full_like(probs, float("-inf"))
            )
        else:
            probs = None
            val_for_gumbel = vals  # use logits directly

        keep = torch.ones(k_b, dtype=torch.bool, device=device)
        tp_b = tp[b].item()
        if tp_b > 0:
            cumsum = torch.cumsum(probs, dim=-1)
            cumsum_excl = cumsum - probs
            keep = (torch.arange(k_b, device=device) == 0) | (cumsum_excl < tp_b)

        # Gumbel-max: key = val_for_gumbel + gumbel_noise[b, tok].
        noise = gumbel_noise[b, idx]
        key = val_for_gumbel + noise
        key = torch.where(keep, key, torch.full_like(key, float("-inf")))
        # Tie-break toward smaller token id.
        max_key = key.max()
        cand = (key == max_key).nonzero(as_tuple=True)[0]
        if cand.numel() == 0:
            best_pos = torch.tensor(0, device=device)
        else:
            cand_tokens = idx[cand]
            best_local = torch.argmin(cand_tokens)
            best_pos = cand[best_local]
        tok = int(idx[best_pos].item())
        tokens_out[b, 0] = tok

    # Penalty writeback.
    expected_mask = None
    if penalty_mask is not None and slot_id is not None:
        expected_mask = penalty_mask.clone()
        for b in range(B):
            tok = int(tokens_out[b, 0].item())
            row = int(slot_id[b].item())
            expected_mask[row, tok // 8] |= 1 << (tok % 8)
    return tokens_out, expected_mask


@pytest.mark.parametrize("vocab_size", [120832])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_only_logits(batch_size, vocab_size):
    torch.manual_seed(0)
    logits = torch.randn(batch_size, vocab_size, device="cuda")
    gumbel = _gumbel0_like(logits)

    tok = hpc.fused_sampler(logits, gumbel_noise=gumbel)
    assert tok.shape == (batch_size, 1)
    assert tok.dtype == torch.int32

    ref, _ = ref_fused_sampler(logits, gumbel_noise=gumbel, max_topk=32)
    assert torch.equal(tok, ref), f"mismatch my={tok.flatten()} ref={ref.flatten()}"


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("max_topk,topk_val", [(32, 20), (64, 50)])
@pytest.mark.parametrize(
    "softmax_policy,topp_val",
    [
        (SoftmaxPolicy.NONE, 0.0),
        (SoftmaxPolicy.BEFORE_TOPK, 0.9),
        (SoftmaxPolicy.AFTER_TOPK, 0.9),
    ],
)
def test_topk_topp_softmax_matrix(batch_size, max_topk, topk_val, softmax_policy, topp_val):
    vocab_size = 120832
    torch.manual_seed(1234 + max_topk + topk_val + int(topp_val * 10))
    logits = torch.randn(batch_size, vocab_size, device="cuda")
    gumbel = _gumbel0_like(logits)

    topk_t = torch.full((batch_size,), topk_val, dtype=torch.int32, device="cuda")
    if topp_val > 0:
        topp_t = torch.full((batch_size,), topp_val, dtype=torch.float32, device="cuda")
    else:
        topp_t = 0.0

    tok = hpc.fused_sampler(
        logits,
        softmax_policy=softmax_policy,
        topk=topk_t,
        topp=topp_t,
        max_topk=max_topk,
        gumbel_noise=gumbel,
    )
    ref, _ = ref_fused_sampler(
        logits,
        softmax_policy=softmax_policy,
        topk=topk_t,
        topp=topp_t,
        max_topk=max_topk,
        gumbel_noise=gumbel,
    )
    assert torch.equal(tok, ref), (
        f"mismatch @ max_topk={max_topk} policy={softmax_policy} topp={topp_val}:"
        f" my={tok.flatten()[:5]} ref={ref.flatten()[:5]}"
    )


@pytest.mark.parametrize("batch_size", [1, 4])
def test_repetition_penalty_and_writeback(batch_size):
    vocab_size = 120832
    torch.manual_seed(7)
    logits = torch.randn(batch_size, vocab_size, device="cuda")
    gumbel = _gumbel0_like(logits)

    # MAX_BS > batch_size so writeback must honor slot_id routing.
    max_bs = batch_size + 3
    row_bytes = (vocab_size + 7) // 8
    penalty = torch.zeros((max_bs, row_bytes), dtype=torch.uint8, device="cuda")
    # Pre-seed random bits (checks OR semantics).
    penalty.random_(0, 256)
    slot_id = torch.randperm(max_bs, device="cuda", dtype=torch.int32)[:batch_size]

    penalty_hpc = penalty.clone()
    ref_tok, ref_penalty = ref_fused_sampler(
        logits,
        penalty_mask=penalty.clone(),
        slot_id=slot_id,
        repetition_penalty=1.05,
        temperature=0.7,
        softmax_policy=SoftmaxPolicy.AFTER_TOPK,
        topk=torch.full((batch_size,), 20, dtype=torch.int32, device="cuda"),
        topp=torch.full((batch_size,), 0.9, dtype=torch.float32, device="cuda"),
        max_topk=32,
        gumbel_noise=gumbel,
    )

    tok = hpc.fused_sampler(
        logits,
        penalty_mask=penalty_hpc,
        slot_id=slot_id,
        repetition_penalty=1.05,
        temperature=0.7,
        softmax_policy=SoftmaxPolicy.AFTER_TOPK,
        topk=torch.full((batch_size,), 20, dtype=torch.int32, device="cuda"),
        topp=torch.full((batch_size,), 0.9, dtype=torch.float32, device="cuda"),
        max_topk=32,
        gumbel_noise=gumbel,
    )
    assert torch.equal(tok, ref_tok)
    assert torch.equal(penalty_hpc, ref_penalty), "penalty writeback mismatch"


def test_bf16_logits():
    torch.manual_seed(3)
    B, V = 2, 120832
    logits = torch.randn(B, V, device="cuda", dtype=torch.bfloat16)
    gumbel = _gumbel0_like(logits)
    tok = hpc.fused_sampler(
        logits,
        softmax_policy=SoftmaxPolicy.AFTER_TOPK,
        topk=20,
        topp=0.9,
        max_topk=32,
        gumbel_noise=gumbel,
    )
    # Reference uses fp32-cast logits (same as kernel).
    ref, _ = ref_fused_sampler(
        logits.float(),
        softmax_policy=SoftmaxPolicy.AFTER_TOPK,
        topk=20,
        topp=0.9,
        max_topk=32,
        gumbel_noise=gumbel,
    )
    assert torch.equal(tok, ref), f"my={tok.flatten()} ref={ref.flatten()}"


def test_scalar_vs_tensor_equivalence():
    torch.manual_seed(5)
    B, V = 4, 120832
    logits = torch.randn(B, V, device="cuda")
    gumbel = _gumbel0_like(logits)
    t1 = hpc.fused_sampler(
        logits,
        temperature=0.7,
        topk=20,
        max_topk=32,
        softmax_policy=SoftmaxPolicy.AFTER_TOPK,
        topp=0.9,
        gumbel_noise=gumbel,
    )
    t2 = hpc.fused_sampler(
        logits,
        temperature=torch.full((B,), 0.7, device="cuda"),
        topk=torch.full((B,), 20, dtype=torch.int32, device="cuda"),
        max_topk=32,
        softmax_policy=SoftmaxPolicy.AFTER_TOPK,
        topp=torch.full((B,), 0.9, device="cuda"),
        gumbel_noise=gumbel,
    )
    assert torch.equal(t1, t2)


def test_curand_smoke():
    """Curand fallback smoke test (no gumbel injection)."""
    torch.manual_seed(9)
    B, V = 2, 120832
    logits = torch.randn(B, V, device="cuda")
    tok = hpc.fused_sampler(logits, topk=20, max_topk=32, seed=42)
    assert tok.shape == (B, 1)
    assert tok.dtype == torch.int32
    # Sanity: sampled token is in top-20 per batch.
    top20 = torch.topk(logits, k=20, dim=-1).indices
    for b in range(B):
        assert tok[b, 0].item() in top20[b].tolist()


@pytest.mark.parametrize("pad", [4, 256])
def test_padded_logits_stride(pad):
    """Padded logits: stride(0) > vocab_size. The kernel steps between rows by
    the row stride (not vocab_size), so a padded buffer yields identical token
    ids as the compact slice."""
    vocab_size = 120832
    torch.manual_seed(17)
    B = 4
    V_pad = vocab_size + pad
    padded = torch.randn(B, V_pad, device="cuda")
    # Poison the padding so the kernel would fail matching if it read it.
    padded[:, vocab_size:] = float("inf")
    logits = padded[:, :vocab_size]
    assert logits.stride(0) == V_pad
    assert not logits.is_contiguous()

    gumbel = _gumbel0_like(logits)

    tok = hpc.fused_sampler(
        logits,
        softmax_policy=SoftmaxPolicy.AFTER_TOPK,
        topk=20,
        topp=0.9,
        max_topk=32,
        gumbel_noise=gumbel,
    )
    ref, _ = ref_fused_sampler(
        logits.contiguous(),
        softmax_policy=SoftmaxPolicy.AFTER_TOPK,
        topk=20,
        topp=0.9,
        max_topk=32,
        gumbel_noise=gumbel,
    )
    assert torch.equal(tok, ref), f"padded-stride mismatch my={tok.flatten()} ref={ref.flatten()}"


# ---------------------------------------------------------------------------
# fused_sampler error paths.
# ---------------------------------------------------------------------------
def test_error_topp_without_topk():
    logits = torch.randn(1, 120832, device="cuda")
    with pytest.raises((RuntimeError, ValueError)):
        hpc.fused_sampler(logits, softmax_policy=SoftmaxPolicy.AFTER_TOPK, topp=0.9)


def test_error_topp_without_softmax():
    logits = torch.randn(1, 120832, device="cuda")
    with pytest.raises((RuntimeError, ValueError)):
        hpc.fused_sampler(logits, topk=20, topp=0.9, max_topk=32)


def test_error_penalty_without_slot_id():
    logits = torch.randn(1, 120832, device="cuda")
    penalty = torch.zeros((4, (120832 + 7) // 8), dtype=torch.uint8, device="cuda")
    with pytest.raises((RuntimeError, ValueError)):
        hpc.fused_sampler(logits, penalty_mask=penalty)


def test_error_unsupported_vocab():
    logits = torch.randn(1, 12345, device="cuda")
    with pytest.raises((RuntimeError, ValueError)):
        hpc.fused_sampler(logits, seed=42)


def test_error_bad_max_topk():
    logits = torch.randn(1, 120832, device="cuda")
    with pytest.raises((RuntimeError, ValueError)):
        hpc.fused_sampler(logits, topk=20, max_topk=16)


def test_error_inner_stride_not_one():
    """stride(1) must be 1 — only row-major with inner-contiguous rows is
    supported (padding only allowed on the row stride)."""
    V = 120832
    base = torch.randn(2, V * 2, device="cuda")
    logits = base[:, ::2]  # strided inner dim
    assert logits.stride(1) != 1
    with pytest.raises((RuntimeError, ValueError)):
        hpc.fused_sampler(logits)


# ===========================================================================
# (2) Temperature-only fast-path tests.
#
# `fused_sampler(logits, temperature=t)` with every other feature disabled
# auto-dispatches to `fused_sampler_temperature_sample`. The reference mirrors
# the in-kernel scoring:
#   scores = logits.float() / temperature + gumbel
#   token  = argmax(scores, dim=-1)
# ===========================================================================
def ref_temperature_sample(
    logits: torch.Tensor, temperature: torch.Tensor, gumbel: torch.Tensor
) -> torch.Tensor:
    """PyTorch reference for temperature-only Gumbel-max sampling."""
    scores = logits.float() / temperature.view(-1, 1).float() + gumbel.float()
    return scores.argmax(dim=-1).to(torch.int32).view(-1, 1)


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("temp_mode", ["scalar", "tensor"])
@pytest.mark.parametrize("stride_mode", ["compact", "padded"])
def test_fused_sampler_temperature_only_golden(batch_size, dtype, temp_mode, stride_mode):
    """Bit-exact golden test using external Gumbel(0) noise."""
    vocab_size = 120832
    torch.manual_seed(0xC0FFEE + batch_size * 17 + vocab_size)

    # Build a logits tensor, optionally with stride(0) > vocab_size (padded).
    if stride_mode == "compact":
        logits = torch.randn(batch_size, vocab_size, dtype=dtype, device="cuda")
    else:
        pad = 128
        big = torch.randn(batch_size, vocab_size + pad, dtype=dtype, device="cuda")
        logits = big[:, :vocab_size]
        assert logits.stride(0) == vocab_size + pad
        assert logits.stride(1) == 1

    # Temperature: either scalar or per-batch tensor.
    if temp_mode == "scalar":
        temperature_arg = 0.7
        temperature_ref = torch.full((batch_size,), 0.7, dtype=torch.float32, device="cuda")
    else:
        temperature_ref = torch.rand(batch_size, dtype=torch.float32, device="cuda") * 1.5 + 0.3
        temperature_arg = temperature_ref

    # External Gumbel(0) noise: g = -log(-log(U)), U ~ Uniform(0, 1].
    u = torch.rand(batch_size, vocab_size, dtype=torch.float32, device="cuda")
    u = u.clamp_min_(1e-20)
    gumbel = -(-u.log()).log()

    ref_tokens = ref_temperature_sample(logits, temperature_ref, gumbel)

    out_tokens = hpc.fused_sampler(logits, temperature=temperature_arg, gumbel_noise=gumbel)

    assert out_tokens.shape == (batch_size, 1)
    assert out_tokens.dtype == torch.int32
    # bf16 quantization of logits is applied before division, so the reference
    # consumes the already-cast tensor; bit-exact match is expected.
    assert torch.equal(out_tokens, ref_tokens), (
        f"mismatch: out[:10]={out_tokens[:10].flatten().tolist()}, "
        f"ref[:10]={ref_tokens[:10].flatten().tolist()}"
    )


# ---------------------------------------------------------------------------
# Draft-mask coverage. A single [B] int64 tensor where
#   draft_token_ids[b] == -1  -> row b unmasked
#   draft_token_ids[b] >= 0   -> mask logits[b, draft_token_ids[b]] (-inf)
# Out-of-range non-negative values are silently ignored by the kernel.
# ---------------------------------------------------------------------------
def _ref_temperature_sample_with_mask(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    gumbel: torch.Tensor,
    draft_token_ids: Optional[torch.Tensor],
    vocab_size: int,
) -> torch.Tensor:
    """PyTorch reference: scatter -inf onto a copy of (logits/T), then argmax."""
    scaled = logits.float() / temperature.view(-1, 1).float()
    if draft_token_ids is not None:
        valid = (draft_token_ids >= 0) & (draft_token_ids < vocab_size)
        if valid.any():
            rows = torch.nonzero(valid, as_tuple=False).squeeze(1).to(torch.int64)
            cols = draft_token_ids[valid].to(torch.int64)
            scaled[rows, cols] = float("-inf")
    scores = scaled + gumbel.float()
    return scores.argmax(dim=-1).to(torch.int32).view(-1, 1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "mask_case",
    [
        "all_minus_one",  # all entries == -1: equivalent to no mask
        "single_per_row",  # one mask token per row, with some -1 sentinels
        "out_of_range_token",  # token ids >= vocab_size: silently ignored
    ],
)
def test_fused_sampler_temperature_only_with_draft_mask_golden(dtype, mask_case):
    """Bit-exact golden test for the [B]-shaped draft_token_ids arg."""
    torch.manual_seed(0xBEEF + hash(mask_case) % 1024)
    batch_size = 8
    vocab_size = 120832

    logits = torch.randn(batch_size, vocab_size, dtype=dtype, device="cuda")
    temperature = torch.rand(batch_size, dtype=torch.float32, device="cuda") * 1.5 + 0.3

    u = torch.rand(batch_size, vocab_size, dtype=torch.float32, device="cuda").clamp_min_(1e-20)
    gumbel = -(-u.log()).log()

    # Build a [B] mask tensor per case.
    if mask_case == "all_minus_one":
        draft_token_ids = torch.full((batch_size,), -1, dtype=torch.int64, device="cuda")
    elif mask_case == "single_per_row":
        draft_token_ids = torch.randint(
            0, vocab_size, (batch_size,), dtype=torch.int64, device="cuda"
        )
        # Sprinkle a -1 sentinel so that row stays unmasked.
        draft_token_ids[1] = -1
    elif mask_case == "out_of_range_token":
        # Mix of valid token ids, -1 sentinels, and out-of-range values
        # (>= vocab_size). The kernel must silently treat out-of-range as
        # no-mask (equivalent to -1).
        draft_token_ids = torch.tensor(
            [0, -1, vocab_size, 100, vocab_size + 17, -1, 200, vocab_size * 2],
            dtype=torch.int64,
            device="cuda",
        )
    else:
        raise AssertionError(mask_case)

    ref_tokens = _ref_temperature_sample_with_mask(
        logits, temperature, gumbel, draft_token_ids, vocab_size
    )

    out_tokens = hpc.fused_sampler(
        logits,
        temperature=temperature,
        gumbel_noise=gumbel,
        draft_token_ids=draft_token_ids,
    )

    assert out_tokens.shape == (batch_size, 1)
    assert out_tokens.dtype == torch.int32
    assert torch.equal(out_tokens, ref_tokens), (
        f"mismatch ({mask_case}): out[:10]={out_tokens[:10].flatten().tolist()}, "
        f"ref[:10]={ref_tokens[:10].flatten().tolist()}"
    )

    # Sanity: confirm the kernel did not modify the caller's logits in place.
    baseline_ref = ref_temperature_sample(logits, temperature, gumbel)
    baseline_out = hpc.fused_sampler(logits, temperature=temperature, gumbel_noise=gumbel)
    assert torch.equal(baseline_out, baseline_ref), (
        "logits appear to have been mutated by the masked call: "
        "baseline output diverged from baseline reference."
    )


def test_fused_sampler_temperature_draft_mask_shape_validation():
    """draft_token_ids must be [B] int64."""
    batch_size = 4
    logits = torch.randn(batch_size, 120832, dtype=torch.float32, device="cuda")

    # No mask: ok.
    _ = hpc.fused_sampler(logits, temperature=1.0, seed=42)
    # Correct shape: ok.
    draft = torch.full((batch_size,), -1, dtype=torch.int64, device="cuda")
    _ = hpc.fused_sampler(logits, temperature=1.0, draft_token_ids=draft, seed=42)
    # Wrong size: must raise.
    bad = torch.full((batch_size + 1,), -1, dtype=torch.int64, device="cuda")
    with pytest.raises(RuntimeError, match="draft_token_ids size must be"):
        hpc.fused_sampler(logits, temperature=1.0, draft_token_ids=bad, seed=42)
    # Wrong dtype: must raise.
    bad_dtype = torch.full((batch_size,), -1, dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="draft_token_ids dtype must be int64"):
        hpc.fused_sampler(logits, temperature=1.0, draft_token_ids=bad_dtype, seed=42)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_sampler_temperature_distribution(dtype):
    """Sanity test: empirical sampling frequencies track softmax(logits/T)."""
    vocab_size = 120832
    torch.manual_seed(1234)
    batch_size = 4
    # Concentrate probability on a small set of tokens so we can test with
    # modest iteration counts.
    K = 16
    top_tokens = torch.randint(0, vocab_size, (batch_size, K), device="cuda")
    logits = torch.full((batch_size, vocab_size), -20.0, dtype=torch.float32, device="cuda")
    for b in range(batch_size):
        logits[b, top_tokens[b]] = torch.randn(K, device="cuda") * 2.0 + 3.0
    logits_in = logits.to(dtype)
    temperature = 1.0

    N = 2000
    counts = torch.zeros(batch_size, vocab_size, dtype=torch.float32, device="cuda")
    for _ in range(N):
        tok = hpc.fused_sampler(logits_in, temperature=temperature, seed=42)  # [B, 1]
        counts.scatter_add_(1, tok.to(torch.int64), torch.ones_like(tok, dtype=torch.float32))
    freq = counts / N
    ref_prob = torch.softmax(logits / temperature, dim=-1)

    # TV distance between empirical freq and true prob should be small.
    tv = 0.5 * (freq - ref_prob).abs().sum(dim=-1)
    assert (tv < 0.1).all(), f"TV distance too large: {tv.tolist()}"
