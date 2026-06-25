"""Correctness test for the q-pertoken-perhead + KV-pertensor backend.

Tests cover:

  A. Math equivalence vs base op
     The output must be byte-identical to the base
     ``attention_with_kvcache_prefill_fp8_packed_cutedsl`` op when fed
     equivalent broadcast tensors (K-scale tail filled with the same
     scalar, vscale broadcast to per-head). The base op needs a tail-
     bearing KV cache; we construct one from the no-tail cache
     just for this comparison (see ``_run_base_op_with_constructed_tail``).

  C. From-scratch fp32 per-tensor dequant reference (load-bearing)
     Pure-Python attention reference dequantizing K/V using literal
     scalar multiplication (K_dq = k_const * K_fp8;
     V_dq = v_const * V_fp8), softmax in fp32, cast to bf16. If the
     output matches this reference (within atol), the math
     semantic ("K-scale = single scalar" matches "per-tensor K dequant")
     is end-to-end correct. 5 shapes (aligned + bad + GQA + ragged) x
     3 (k_scale, v_scale) combos x 3 schedulers = 45 cases.

  D. KV cache layout cross-check
     NHD / HND / NHD-cross-layer / HND-cross-layer all work —
     preserves the kernel's stride-agnostic property.

  E. Scalar / 1-element tensor equivalence
     ``k_scale=0.5`` (Python float) and ``k_scale=torch.tensor([0.5])``
     must produce byte-identical output.

  F. Validate sanity
     Multi-element k/v scale tensors must raise.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import List

import pytest
import torch

pytest.importorskip("cutlass.cute")
pytest.importorskip("cuda.bindings.driver")

# Reference helpers live next to this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _reference_helpers import (  # noqa: E402
    allclose,
    naive_attn_with_kvcache_func,
    quant_paged_cache_perhead,
    quant_paged_cache_pertoken,
)

import hpc  # noqa: E402


def _sm100_available() -> bool:
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    return cap[0] >= 10


requires_sm100 = pytest.mark.skipif(not _sm100_available(), reason="requires SM100 (Blackwell)")


# ============================================================================
# Builder — paged FP8 cache + ALL scales constant (the wrapper's target ABI)
# ============================================================================
def _build_inputs_per_tensor_kv(
    q_lens: List[int],
    kv_lens: List[int],
    num_head_q: int,
    num_head_kv: int,
    head_dim: int = 128,
    block_size: int = 64,
    *,
    k_scale_value: float = 1.0,
    v_scale_value: float = 1.0,
    seed: int = 10086,
):
    """Build inputs for the per-tensor-KV op (no K-scale tail).

    The KV cache is allocated as ``[num_pages, 2, block_size, Hkv, D]``
    (FP8 e4m3fn) — no extra ``scale_rows`` rows because the kernel does
    not load per-token K-scales any more. Storage is ~3% smaller per
    page than the base op's tail-bearing layout.

    The K cache is dequantized as ``q_fp8 * (1.0 * k_scale_value)`` per token,
    i.e. the underlying FP8 bytes carry the data scaled by ``1/k_scale_value``
    so that ``data_fp8 * k_scale_value`` recovers the bf16-equivalent value.
    Similarly for V.
    """
    assert len(q_lens) == len(kv_lens)
    num_batch = len(q_lens)
    dtype = torch.bfloat16
    fp8 = torch.float8_e4m3fn
    max_q = max(q_lens)
    max_seq_q_pad = ((max_q + 127) // 128) * 128

    torch.cuda.manual_seed(seed)

    q_per_seq, k_per_seq, v_per_seq = [], [], []
    for ql, kl in zip(q_lens, kv_lens):
        q_per_seq.append(
            (
                torch.randn((ql, num_head_q, head_dim), dtype=dtype, device="cuda")
                / math.sqrt(head_dim)
            ).to(fp8)
        )
        # Pre-divide K/V by their respective scales so that ``fp8_data *
        # scale`` recovers the original bf16-magnitude tensor.
        k_per_seq.append(
            torch.randn((kl, num_head_kv, head_dim), dtype=dtype, device="cuda")
            / math.sqrt(head_dim)
            / k_scale_value
        )
        v_per_seq.append(
            torch.randn((kl, num_head_kv, head_dim), dtype=dtype, device="cuda") / v_scale_value
        )
    q_packed = torch.cat(q_per_seq, dim=0)

    qscale_pth = (
        torch.abs(
            torch.randn((num_batch, num_head_q, max_seq_q_pad), dtype=torch.float32, device="cuda")
        )
        / 10
    )

    seqlens_q = torch.tensor(q_lens, dtype=torch.int32, device="cuda")
    seqlens_kvcache = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    cu_seqlens_q = torch.cumsum(
        torch.cat([torch.zeros(1, dtype=torch.int32, device="cuda"), seqlens_q]),
        dim=0,
    ).to(torch.int32)

    blocks_per_row = [(kl + block_size - 1) // block_size for kl in kv_lens]
    max_blocks = max(blocks_per_row)
    total_blocks = sum(blocks_per_row)
    pool_size = max(num_batch * max_blocks * 2, total_blocks + 16)

    # Stage 3: allocate just [P, 2, block_size, Hkv, D] — no scale_rows tail.
    kvcache_bf16 = torch.zeros(
        pool_size,
        2,
        block_size,
        num_head_kv,
        head_dim,
        dtype=dtype,
        device="cuda",
    )
    perm = torch.randperm(pool_size, device="cuda")[:total_blocks].to(torch.int32)
    block_ids = torch.zeros(num_batch, max_blocks, dtype=torch.int32, device="cuda")
    cursor = 0
    for b in range(num_batch):
        nb = blocks_per_row[b]
        block_ids[b, :nb] = perm[cursor : cursor + nb]
        cursor += nb
        for blk in range(nb):
            start = blk * block_size
            end = min(start + block_size, kv_lens[b])
            page = int(block_ids[b, blk].item())
            kvcache_bf16[page, 0, : end - start] = k_per_seq[b][start:end]
            kvcache_bf16[page, 1, : end - start] = v_per_seq[b][start:end]

    # Direct cast to fp8 (data already pre-divided so the magnitude is right).
    kvcache_fp8 = kvcache_bf16.to(fp8)

    return {
        "q_packed": q_packed,
        "kvcache_fp8": kvcache_fp8,
        "qscale_pth": qscale_pth,
        "k_scale_value": k_scale_value,
        "v_scale_value": v_scale_value,
        "cu_seqlens_q": cu_seqlens_q,
        "seqlens_kvcache": seqlens_kvcache,
        "block_ids": block_ids,
        "max_q": max_q,
        "max_seq_q_pad": max_seq_q_pad,
        "num_head_kv": num_head_kv,
    }


def _slice_kvc(kvcache_fp8, block_size):
    """Return (kcache, vcache) views — no K-scale tail."""
    return (
        kvcache_fp8[:, 0, :block_size, :, :],  # kcache
        kvcache_fp8[:, 1, :block_size, :, :],  # vcache
    )


def _to_hnd_view(kvcache_nhd_fp8, num_head_kv, head_dim, block_size):
    """HND view — no scale_rows tail."""
    fp8 = torch.float8_e4m3fn
    P = kvcache_nhd_fp8.shape[0]
    phys = torch.zeros(P, 2, num_head_kv, block_size, head_dim, dtype=fp8, device="cuda")
    hnd_view = phys.transpose(2, 3)
    hnd_view.copy_(kvcache_nhd_fp8)
    return hnd_view


def _to_nhd_cross_layer_view(
    kvcache_nhd_fp8, num_head_kv, head_dim, block_size, num_layers=32, layer_idx=0
):
    fp8 = torch.float8_e4m3fn
    P = kvcache_nhd_fp8.shape[0]
    phys = torch.zeros(
        P, num_layers, 2, block_size, num_head_kv, head_dim, dtype=fp8, device="cuda"
    )
    per_layer = phys[:, layer_idx]
    per_layer.copy_(kvcache_nhd_fp8)
    return per_layer


def _to_hnd_cross_layer_view(
    kvcache_nhd_fp8, num_head_kv, head_dim, block_size, num_layers=32, layer_idx=0
):
    fp8 = torch.float8_e4m3fn
    P = kvcache_nhd_fp8.shape[0]
    phys = torch.zeros(
        P, 2, num_head_kv, num_layers, block_size, head_dim, dtype=fp8, device="cuda"
    )
    per_layer = phys[:, :, :, layer_idx, :, :]
    nhd_logical = per_layer.transpose(2, 3)
    nhd_logical.copy_(kvcache_nhd_fp8)
    return nhd_logical


# ============================================================================
# True per-tensor reference — completely independent fp32 attention,
# does NOT call the base op or naive_attn_with_kvcache_func.
# This is the math-correctness load-bearing reference.
# ============================================================================
def _true_per_tensor_reference(bundle, *, num_head_q, head_dim, block_size):
    """Pure-Python from-scratch attention reference assuming **per-tensor**
    K and V dequantization.

    Explicitly dequantizes K/V using the scalar k_scale / v_scale and runs
    softmax in fp32 — no broadcast trickery and no dependency on the base
    op. If the wrapper's output matches this reference (within atol),
    the kernel produces output mathematically equivalent to a true
    per-tensor dequant + attention, end to end.

    Math (per query-token i, head head_q in batch b):
        h_kv     = head_q // (Hq // Hkv)
        q_dq[i]  = q_scale_pth[b, head_q, i] * Q_fp8[i, head_q, :]
        k_dq[j]  = k_scale_value * K_fp8[j, h_kv, :]               # per-tensor!
        v_dq[j]  = v_scale_value * V_fp8[j, h_kv, :]               # per-tensor!
        S[i, j]  = q_dq[i] @ k_dq[j].T  for j in [0, kv_len(b))
        S        := S - max_j(S, where causal-mask allows)
        P[i, j]  = softmax_along_j(S[i, :])
        O[i, :]  = sum_j P[i, j] * v_dq[j]
    Causal mask: j > i is masked to -inf.

    Output dtype is bf16 (cast at the end), matching what the kernel
    produces.
    """
    block_ids = bundle["block_ids"]
    seqlens_kv = bundle["seqlens_kvcache"]
    cu = bundle["cu_seqlens_q"].tolist()
    qscale_pth = bundle["qscale_pth"]
    k_s = float(bundle["k_scale_value"])
    v_s = float(bundle["v_scale_value"])
    q_packed = bundle["q_packed"]
    kvcache_fp8 = bundle["kvcache_fp8"]

    num_head_kv = bundle["num_head_kv"]
    h_r = num_head_q // num_head_kv

    # Slice once.
    kcache = kvcache_fp8[:, 0, :block_size, :, :]  # [P, blk, Hkv, D] fp8
    vcache = kvcache_fp8[:, 1, :block_size, :, :]  # [P, blk, Hkv, D] fp8
    P_pages, blk, Hkv, D = kcache.shape
    assert D == head_dim and blk == block_size and Hkv == num_head_kv

    out_per_b = []
    for b in range(len(cu) - 1):
        ql = cu[b + 1] - cu[b]
        kl = int(seqlens_kv[b].item())

        # Q for this batch row, fp32 dequantized: q_fp32[i, head_q, d] = q_scale[i, head_q] * q_fp8
        q_b_fp8 = q_packed[cu[b] : cu[b + 1]]  # [ql, Hq, D] fp8
        q_b_fp32 = q_b_fp8.float()
        q_scale_b = qscale_pth[b, :, :ql]  # [Hq, ql] fp32
        # broadcast multiply: [ql, Hq, D] * [Hq, ql, 1] -> [ql, Hq, D]
        q_b_dq = q_b_fp32 * q_scale_b.transpose(0, 1).unsqueeze(-1)

        # Gather per-batch K/V from page table: [kl, Hkv, D] fp32
        nb = (kl + block_size - 1) // block_size
        page_ids = block_ids[b, :nb].tolist()
        k_b_fp8 = torch.empty((kl, num_head_kv, head_dim), dtype=torch.float8_e4m3fn, device="cuda")
        v_b_fp8 = torch.empty_like(k_b_fp8)
        for blk_i, pg in enumerate(page_ids):
            start = blk_i * block_size
            end = min(start + block_size, kl)
            k_b_fp8[start:end] = kcache[pg, : end - start]
            v_b_fp8[start:end] = vcache[pg, : end - start]

        # Per-tensor dequant: just one multiply.
        k_b_dq = k_b_fp8.float() * k_s  # [kl, Hkv, D] fp32
        v_b_dq = v_b_fp8.float() * v_s  # [kl, Hkv, D] fp32

        # Attention per (Hq) — broadcast Hkv to Hq via h_r.
        # Use bf16 cast on output to match kernel.
        out_b = torch.empty((ql, num_head_q, head_dim), dtype=torch.float32, device="cuda")
        for hq in range(num_head_q):
            hkv = hq // h_r
            q_h = q_b_dq[:, hq, :]  # [ql, D]
            k_h = k_b_dq[:, hkv, :]  # [kl, D]
            v_h = v_b_dq[:, hkv, :]  # [kl, D]
            S = q_h @ k_h.T  # [ql, kl]
            # Causal mask (within batch): Q is aligned to the *end* of K,
            # i.e. Q starts at position kl - ql (the last ql tokens of KV
            # are aligned to the ql query positions). So query position i
            # sees keys [0, kl - ql + i].
            kv_offset = kl - ql
            i_idx = torch.arange(ql, device="cuda").unsqueeze(1)  # [ql, 1]
            j_idx = torch.arange(kl, device="cuda").unsqueeze(0)  # [1, kl]
            mask = j_idx <= (kv_offset + i_idx)  # [ql, kl] bool
            S = torch.where(mask, S, torch.full_like(S, float("-inf")))

            P = torch.softmax(S, dim=-1)  # [ql, kl] fp32
            out_b[:, hq, :] = P @ v_h  # [ql, D]

        out_per_b.append(out_b.to(torch.bfloat16))
    return torch.cat(out_per_b, dim=0)


# ============================================================================
# Helpers to drive the kernel
# ============================================================================
def _run_wrapper(
    bundle,
    num_head_q,
    scheduler,
    head_dim,
    block_size,
    k_scale_override=None,
    v_scale_override=None,
):
    """Stage 3 wrapper ABI: no kscale_tail, no prefill_kscale_tail."""
    kcache, vcache = _slice_kvc(bundle["kvcache_fp8"], block_size)
    total_q = bundle["q_packed"].shape[0]
    out = torch.empty((total_q, num_head_q, head_dim), dtype=torch.bfloat16, device="cuda")
    config = None
    if scheduler is not None:
        from dsl.attention import Fp8PagedPrefillKvPertensorConfig

        config = Fp8PagedPrefillKvPertensorConfig(is_persistent=(scheduler == "persistent"))

    k_s = bundle["k_scale_value"] if k_scale_override is None else k_scale_override
    v_s = bundle["v_scale_value"] if v_scale_override is None else v_scale_override

    return hpc.attention_with_kvcache_prefill_fp8_packed_cutedsl_q_pertoken_kv_pertensor(
        bundle["q_packed"],
        kcache,
        vcache,
        bundle["qscale_pth"],
        k_s,
        v_s,
        bundle["cu_seqlens_q"],
        bundle["block_ids"],
        bundle["seqlens_kvcache"],
        bundle["max_seq_q_pad"],
        output=out,
        config=config,
    )


def _run_base_op_with_constructed_tail(bundle, num_head_q, scheduler, head_dim, block_size):
    """Drive the base op with a constructed K-scale tail.

    Stage 3 KV cache is allocated WITHOUT a scale_rows tail. To compare
    against the base op (which still requires the tail) we re-allocate
    a tail-bearing copy here, fill the tail with k_scale_value, and
    invoke the base op on the new allocation. Used by bucket A (wrapper
    bit-equal check vs base op).
    """
    fp8 = torch.float8_e4m3fn
    kvc_no_tail = bundle["kvcache_fp8"]  # [P, 2, blk, Hkv, D]
    P, _, blk, num_head_kv, D = kvc_no_tail.shape
    assert blk == block_size
    scale_rows = block_size * 4 // head_dim  # = 2

    kvc_with_tail = torch.zeros(
        P,
        2,
        block_size + scale_rows,
        num_head_kv,
        D,
        dtype=fp8,
        device="cuda",
    )
    kvc_with_tail[:, :, :block_size, :, :] = kvc_no_tail
    # Fill K-scale tail with the broadcast k_scale_value (FP32 reinterpreted
    # as FP8 bytes — same trick the old wrapper used).
    kvc_with_tail[:, 0, block_size:, :, :].view(torch.float32).fill_(bundle["k_scale_value"])

    kcache = kvc_with_tail[:, 0, :block_size, :, :]
    vcache = kvc_with_tail[:, 1, :block_size, :, :]
    kscale_tail = kvc_with_tail[:, 0, block_size:, :, :]
    vscale_per_head = torch.full(
        (num_head_kv,),
        bundle["v_scale_value"],
        dtype=torch.float32,
        device="cuda",
    )

    total_q = bundle["q_packed"].shape[0]
    out = torch.empty((total_q, num_head_q, head_dim), dtype=torch.bfloat16, device="cuda")
    config = None
    if scheduler is not None:
        from dsl.attention import Fp8PagedPrefillConfig

        config = Fp8PagedPrefillConfig(is_persistent=(scheduler == "persistent"))

    return hpc.attention_with_kvcache_prefill_fp8_packed_cutedsl(
        bundle["q_packed"],
        kcache,
        vcache,
        bundle["qscale_pth"],
        kscale_tail,
        vscale_per_head,
        bundle["cu_seqlens_q"],
        bundle["block_ids"],
        bundle["seqlens_kvcache"],
        bundle["max_seq_q_pad"],
        output=out,
        config=config,
    )


# ============================================================================
# Shape mix
# ============================================================================
_SHAPES = [
    pytest.param([128], [128], 8, 1, id="aligned_B1_q128_Hkv1"),
    pytest.param([1024], [1024], 32, 4, id="aligned_B1_q1024_Hkv4_GQA"),
    pytest.param([192], [192], 8, 1, id="bad_B1_q192_Hkv1"),
    pytest.param([3938], [3938], 8, 1, id="bad_B1_q3938_Hkv1"),
    pytest.param([192], [192], 32, 4, id="bad_B1_q192_Hkv4_GQA"),
    pytest.param([3611, 3824, 3874, 3806], [3611, 3824, 3874, 3806], 8, 1, id="bad_B4_ragged_Hkv1"),
    pytest.param([192, 1024], [4096, 4096], 8, 1, id="bad_B2_q_lt_kv"),
]

_LAYOUTS = ["NHD", "HND", "NHD_cross_layer", "HND_cross_layer"]


# ============================================================================
# A. Math equivalence — wrapper output ≡ base op output (same scalar fed
#    both sides). Strongest correctness check — proves the wrapper is just
#    an ABI shim and does not perturb numerics.
# ============================================================================
@requires_sm100
@pytest.mark.parametrize("scheduler", [None, "persistent", "static"])
@pytest.mark.parametrize("q_lens,kv_lens,num_head_q,num_head_kv", _SHAPES)
def test_wrapper_matches_base_op_bit_for_bit(q_lens, kv_lens, num_head_q, num_head_kv, scheduler):
    head_dim, block_size = 128, 64
    bundle = _build_inputs_per_tensor_kv(
        q_lens,
        kv_lens,
        num_head_q,
        num_head_kv,
        head_dim,
        block_size,
        k_scale_value=0.5,
        v_scale_value=2.0,
    )

    # Re-build a clone of the bundle so each path gets a fresh KV cache
    # (the wrapper writes to kscale_tail).
    out_wrapper = _run_wrapper(bundle, num_head_q, scheduler, head_dim, block_size)
    bundle2 = _build_inputs_per_tensor_kv(
        q_lens,
        kv_lens,
        num_head_q,
        num_head_kv,
        head_dim,
        block_size,
        k_scale_value=0.5,
        v_scale_value=2.0,
    )
    out_base = _run_base_op_with_constructed_tail(
        bundle2,
        num_head_q,
        scheduler,
        head_dim,
        block_size,
    )

    assert torch.equal(out_wrapper, out_base), (
        f"wrapper diverges from base op for q_lens={q_lens}, "
        f"Hq={num_head_q}, Hkv={num_head_kv}, sched={scheduler}; "
        f"max abs diff = {(out_wrapper.float() - out_base.float()).abs().max().item():.6e}"
    )


# ============================================================================
# C. True per-tensor dequant reference (load-bearing math correctness).
#
#    Run a from-scratch fp32 attention reference that ASSUMES per-tensor
#    K/V dequant — i.e. K_dq[j] = k_scalar * K_fp8[j] (one scalar mul, no
#    broadcast trickery), softmax, P @ (v_scalar * V_fp8). If the wrapper
#    output matches this within atol, the wrapper's "fill K-scale tail
#    with k_scalar" strategy is mathematically equivalent to per-tensor
#    K dequant — end to end.
#
#    This complements test A (which only proves "wrapper ≡ base op fed
#    equivalent broadcast"): test A's bit-equality alone does not prove
#    that "K-scale tail = k_scalar everywhere" matches the per-tensor
#    semantic; this test does.
#
#    Tolerance: atol=0.1 — same FP8 noise floor as test B. The reference
#    runs in fp32 + cast to bf16 at the end; the kernel uses tcgen05 fp32
#    accumulators throughout. Differences come entirely from rounding /
#    summation order, not from the scale semantic.
# ============================================================================
@requires_sm100
@pytest.mark.parametrize("scheduler", [None, "persistent", "static"])
@pytest.mark.parametrize(
    "k_scale_value,v_scale_value",
    [
        pytest.param(1.0, 1.0, id="kvscale_unit"),
        pytest.param(0.5, 2.0, id="kvscale_half_double"),
        pytest.param(2.0, 0.25, id="kvscale_double_quarter"),
    ],
)
@pytest.mark.parametrize(
    "q_lens,kv_lens,num_head_q,num_head_kv",
    [
        pytest.param([128], [128], 8, 1, id="aligned_q128_Hkv1"),
        pytest.param([192], [192], 8, 1, id="bad_q192_Hkv1"),
        pytest.param([1024], [1024], 32, 4, id="aligned_q1024_Hkv4_GQA"),
        pytest.param([192], [192], 32, 4, id="bad_q192_Hkv4_GQA"),
        pytest.param(
            [3611, 3824, 3874, 3806], [3611, 3824, 3874, 3806], 8, 1, id="bad_B4_ragged_Hkv1"
        ),
    ],
)
def test_wrapper_matches_per_tensor_dequant_reference(
    q_lens,
    kv_lens,
    num_head_q,
    num_head_kv,
    k_scale_value,
    v_scale_value,
    scheduler,
):
    head_dim, block_size = 128, 64
    bundle = _build_inputs_per_tensor_kv(
        q_lens,
        kv_lens,
        num_head_q,
        num_head_kv,
        head_dim,
        block_size,
        k_scale_value=k_scale_value,
        v_scale_value=v_scale_value,
    )

    got = _run_wrapper(bundle, num_head_q, scheduler, head_dim, block_size)
    assert torch.isfinite(got).all(), "non-finite output"

    ref = _true_per_tensor_reference(
        bundle,
        num_head_q=num_head_q,
        head_dim=head_dim,
        block_size=block_size,
    )
    assert got.shape == ref.shape

    diff = (got.float() - ref.float()).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    assert max_abs <= 0.1, (
        f"wrapper diverges from from-scratch per-tensor dequant reference: "
        f"q_lens={q_lens}, Hq={num_head_q}, Hkv={num_head_kv}, "
        f"k_scale={k_scale_value}, v_scale={v_scale_value}, sched={scheduler}; "
        f"max_abs={max_abs:.4f} mean_abs={mean_abs:.4f}"
    )


# ============================================================================
# D. KV cache layout cross-check (NHD / HND / cross-layer).
# ============================================================================
@requires_sm100
@pytest.mark.parametrize("layout", _LAYOUTS)
@pytest.mark.parametrize("scheduler", [None, "persistent", "static"])
@pytest.mark.parametrize(
    "q_lens,kv_lens,num_head_q,num_head_kv",
    [
        pytest.param([128], [128], 8, 1, id="aligned_q128_Hkv1"),
        pytest.param([1024], [1024], 32, 4, id="aligned_q1024_Hkv4_GQA"),
        pytest.param([192], [192], 32, 4, id="bad_q192_Hkv4_GQA"),
        pytest.param(
            [3611, 3824, 3874, 3806], [3611, 3824, 3874, 3806], 8, 1, id="bad_B4_ragged_Hkv1"
        ),
    ],
)
def test_wrapper_layout_agnostic(layout, q_lens, kv_lens, num_head_q, num_head_kv, scheduler):
    """Each of the 4 supported KV layouts must give the same wrapper output."""
    head_dim, block_size = 128, 64
    bundle_nhd = _build_inputs_per_tensor_kv(
        q_lens,
        kv_lens,
        num_head_q,
        num_head_kv,
        head_dim,
        block_size,
        k_scale_value=0.5,
        v_scale_value=2.0,
    )
    out_nhd = _run_wrapper(bundle_nhd, num_head_q, scheduler, head_dim, block_size)

    if layout == "NHD":
        return  # already covered above

    bundle_xv = _build_inputs_per_tensor_kv(
        q_lens,
        kv_lens,
        num_head_q,
        num_head_kv,
        head_dim,
        block_size,
        k_scale_value=0.5,
        v_scale_value=2.0,
    )
    if layout == "HND":
        bundle_xv["kvcache_fp8"] = _to_hnd_view(
            bundle_xv["kvcache_fp8"],
            num_head_kv,
            head_dim,
            block_size,
        )
    elif layout == "NHD_cross_layer":
        bundle_xv["kvcache_fp8"] = _to_nhd_cross_layer_view(
            bundle_xv["kvcache_fp8"],
            num_head_kv,
            head_dim,
            block_size,
        )
    elif layout == "HND_cross_layer":
        bundle_xv["kvcache_fp8"] = _to_hnd_cross_layer_view(
            bundle_xv["kvcache_fp8"],
            num_head_kv,
            head_dim,
            block_size,
        )

    out_xv = _run_wrapper(bundle_xv, num_head_q, scheduler, head_dim, block_size)
    assert torch.equal(out_nhd, out_xv), (
        f"{layout} backend diverges from NHD for q_lens={q_lens}, "
        f"Hq={num_head_q}, Hkv={num_head_kv}, sched={scheduler}"
    )


# ============================================================================
# E. Scalar / 1-element tensor equivalence.
# ============================================================================
@requires_sm100
def test_scalar_or_tensor_scale_equivalent():
    head_dim, block_size = 128, 64
    bundle1 = _build_inputs_per_tensor_kv(
        [128],
        [128],
        8,
        1,
        head_dim,
        block_size,
        k_scale_value=0.7,
        v_scale_value=1.3,
    )
    bundle2 = _build_inputs_per_tensor_kv(
        [128],
        [128],
        8,
        1,
        head_dim,
        block_size,
        k_scale_value=0.7,
        v_scale_value=1.3,
    )

    out_python = _run_wrapper(
        bundle1,
        8,
        None,
        head_dim,
        block_size,
        k_scale_override=0.7,
        v_scale_override=1.3,
    )
    out_tensor = _run_wrapper(
        bundle2,
        8,
        None,
        head_dim,
        block_size,
        k_scale_override=torch.tensor([0.7], dtype=torch.float32, device="cuda"),
        v_scale_override=torch.tensor([1.3], dtype=torch.float32, device="cuda"),
    )
    assert torch.equal(
        out_python, out_tensor
    ), "Python float vs 1-elem tensor scale produced different output"


# ============================================================================
# F. Validate sanity — multi-element k/v scale must raise.
# ============================================================================
@requires_sm100
def test_validate_rejects_multielement_kv_scale():
    """k_scale / v_scale must be 1-element if passed as tensor."""
    head_dim, block_size = 128, 64
    bundle = _build_inputs_per_tensor_kv(
        [128],
        [128],
        8,
        1,
        head_dim,
        block_size,
        k_scale_value=1.0,
        v_scale_value=1.0,
    )
    with pytest.raises(ValueError, match="1-element"):
        _run_wrapper(
            bundle,
            8,
            None,
            head_dim,
            block_size,
            k_scale_override=torch.tensor([1.0, 2.0], dtype=torch.float32, device="cuda"),
        )
