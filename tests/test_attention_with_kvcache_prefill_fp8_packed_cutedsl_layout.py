"""KV-cache layout cross-check: NHD ≡ HND bit-equal, including bad cases.

The cutedsl FMHA backend should be **stride-agnostic** w.r.t. the K/V/K-scale
physical layout. Both NHD ([P, blk+sr, Hkv, hd], default contiguous) and HND
([P, Hkv, blk+sr, hd] physical with an NHD-logical transposed view) must
produce **bit-equal** output for the same logical inputs, and both must match
the naive bf16 reference.

This module verifies that property across:

  * aligned shapes (q_len % 128 == 0),
  * ragged shapes (q_len % 128 != 0 — the OOB-bug trigger),
  * GQA factors where NHD/HND stride asymmetry is real (Hkv >= 4).

The HND build path allocates a 5D physical buffer with Hkv before
block_size, then ``.transpose(2, 3)`` to expose an NHD-shape view whose
strides reflect HND physical order: the kernel sees an NHD-logical tensor
but reads bytes via HND strides.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import List

import pytest
import torch

# Skip early if CuTeDSL is missing.
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
# Builders: NHD baseline + HND via transpose
# ============================================================================
def _build_inputs_nhd(
    q_lens: List[int],
    kv_lens: List[int],
    num_head_q: int,
    num_head_kv: int,
    head_dim: int = 128,
    block_size: int = 64,
    *,
    seed: int = 10086,
):
    """NHD-physical varlen builder. Returns the same bundle shape as
    _build_varlen_inputs in the main OOB test module.

    The KV cache buffer here is contiguous in NHD order:
        physical: [num_pages, 2, block_size + scale_rows, num_head_kv, head_dim]
        stride[3] (Hkv) > stride[2] (scale_rows / block)  — NHD definition
    """
    assert len(q_lens) == len(kv_lens)
    num_batch = len(q_lens)
    dtype = torch.bfloat16
    fp8 = torch.float8_e4m3fn
    max_q = max(q_lens)
    max_seq_q_pad = ((max_q + 127) // 128) * 128
    scale_rows = block_size * 4 // head_dim

    torch.cuda.manual_seed(seed)

    q_per_seq, k_per_seq, v_per_seq = [], [], []
    for ql, kl in zip(q_lens, kv_lens):
        q_per_seq.append(
            (
                torch.randn((ql, num_head_q, head_dim), dtype=dtype, device="cuda")
                / math.sqrt(head_dim)
            ).to(fp8)
        )
        k_per_seq.append(
            torch.randn((kl, num_head_kv, head_dim), dtype=dtype, device="cuda")
            / math.sqrt(head_dim)
        )
        v_per_seq.append(torch.randn((kl, num_head_kv, head_dim), dtype=dtype, device="cuda"))
    q_packed = torch.cat(q_per_seq, dim=0)

    qscale = (
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

    # NHD physical: 5D contiguous with Hkv after scale_rows
    kvcache_bf16 = torch.zeros(
        pool_size,
        2,
        block_size + scale_rows,
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

    kvcache_fp8 = torch.empty_like(kvcache_bf16, dtype=fp8)
    kc, _ = quant_paged_cache_pertoken(kvcache_bf16[:, 0], block_size)
    vc, vscale = quant_paged_cache_perhead(kvcache_bf16[:, 1], block_size)
    kvcache_fp8[:, 0] = kc
    kvcache_fp8[:, 1] = vc

    return {
        "q_packed": q_packed,
        "q_per_seq_fp8": q_per_seq,
        "kvcache_fp8": kvcache_fp8,
        "qscale": qscale,
        "vscale": vscale,
        "cu_seqlens_q": cu_seqlens_q,
        "seqlens_kvcache": seqlens_kvcache,
        "block_ids": block_ids,
        "max_q": max_q,
        "max_seq_q_pad": max_seq_q_pad,
    }


def _to_hnd_view(
    kvcache_nhd_fp8: torch.Tensor, num_head_kv: int, head_dim: int, block_size: int
) -> torch.Tensor:
    """Allocate physically as HND, expose an NHD-logical view.

    Source kvcache_nhd_fp8 has shape:
        [P, 2, block_size+scale_rows, num_head_kv, head_dim]   (NHD physical)
        stride = (P_stride, mid_stride, Hkv*hd, hd, 1)

    We allocate physically as:
        [P, 2, num_head_kv, block_size+scale_rows, head_dim]   (HND physical)
        stride = (P_stride, mid_stride, (blk+sr)*hd, hd, 1)

    Then ``.transpose(2, 3)`` swaps the middle two dims to recover the
    NHD logical shape, but the underlying strides remain HND. Copying the
    NHD source into this view via ``.copy_()`` does an element-wise copy
    in logical order — values match NHD, but the bytes land in HND order.
    """
    fp8 = torch.float8_e4m3fn
    P = kvcache_nhd_fp8.shape[0]
    scale_rows = kvcache_nhd_fp8.shape[2] - block_size

    phys = torch.zeros(
        P,
        2,
        num_head_kv,
        block_size + scale_rows,
        head_dim,
        dtype=fp8,
        device="cuda",
    )
    hnd_view = phys.transpose(2, 3)  # logical: [P, 2, blk+sr, Hkv, hd]
    # Sanity: NHD-logical shape with HND-physical strides.
    assert hnd_view.shape == kvcache_nhd_fp8.shape, (
        f"HND view shape {hnd_view.shape} must equal NHD source shape " f"{kvcache_nhd_fp8.shape}"
    )
    hnd_view.copy_(kvcache_nhd_fp8)
    return hnd_view


def _to_nhd_cross_layer_view(
    kvcache_nhd_fp8: torch.Tensor,
    num_head_kv: int,
    head_dim: int,
    block_size: int,
    num_layers: int = 32,
    layer_idx: int = 0,
) -> torch.Tensor:
    """Cross-layer NHD: the ``include_num_layers_dimension=True`` mode.

    The NHD-with-cross-layer physical buffer is allocated as:

        physical: [num_blocks, num_layers, 2, block_size+scale_rows,
                   num_head_kv, head_dim]
        stride_order = (1, 0, 2, 3, 4, 5)

    All layers' KV slots for the same (block, kv_type, head, token) sit
    physically together so a disaggregated-transfer can move all layers in
    one contiguous chunk.

    Each attention call processes **one layer**: slicing
    ``[:, layer_idx, :, :, :, :]`` out of the 6D buffer gives the kernel a
    5D view with NHD-logical shape ``[P, 2, block+scale_rows, Hkv,
    head_dim]`` whose strides reflect the cross-layer layout (``P_stride``
    carries a ``num_layers`` factor).
    """
    fp8 = torch.float8_e4m3fn
    P = kvcache_nhd_fp8.shape[0]
    scale_rows = kvcache_nhd_fp8.shape[2] - block_size
    assert 0 <= layer_idx < num_layers

    phys = torch.zeros(
        P,
        num_layers,
        2,
        block_size + scale_rows,
        num_head_kv,
        head_dim,
        dtype=fp8,
        device="cuda",
    )
    # Pick one layer slice; the resulting view has NHD-logical shape but
    # with strides reflecting the cross-layer 6D allocation.
    per_layer = phys[:, layer_idx]  # [P, 2, blk+sr, Hkv, hd]
    assert per_layer.shape == kvcache_nhd_fp8.shape
    per_layer.copy_(kvcache_nhd_fp8)
    return per_layer


def _to_hnd_cross_layer_view(
    kvcache_nhd_fp8: torch.Tensor,
    num_head_kv: int,
    head_dim: int,
    block_size: int,
    num_layers: int = 32,
    layer_idx: int = 0,
) -> torch.Tensor:
    """Cross-layer HND: HND-with-cross-layer (stride_order
    ``(1, 2, 4, 0, 3, 5)``).

    The 6D physical buffer is::

        physical: [num_blocks, 2, num_head_kv, num_layers,
                   block_size+scale_rows, head_dim]

    i.e. HND with ``num_layers`` inserted between the head dim and the
    token dim. Per-layer slicing + transpose recovers the NHD-logical
    shape the kernel expects, with strides that carry both the HND
    ``head_kv``-stride permutation and the cross-layer ``num_layers``
    factor. Per-layer slicing leaves (H, BS, D) non-contiguous (the H
    stride contains a ``num_layers`` factor); an ``as_strided()`` wrap
    then gives the kernel a zero-copy view.
    """
    fp8 = torch.float8_e4m3fn
    P = kvcache_nhd_fp8.shape[0]
    scale_rows = kvcache_nhd_fp8.shape[2] - block_size
    assert 0 <= layer_idx < num_layers

    phys = torch.zeros(
        P,
        2,
        num_head_kv,
        num_layers,
        block_size + scale_rows,
        head_dim,
        dtype=fp8,
        device="cuda",
    )
    # Pick layer: [:, :, :, layer_idx, :, :] -> [P, 2, Hkv, blk+sr, hd]
    per_layer = phys[:, :, :, layer_idx, :, :]
    # transpose(2, 3) recovers NHD-logical shape [P, 2, blk+sr, Hkv, hd]
    nhd_logical_view = per_layer.transpose(2, 3)
    assert nhd_logical_view.shape == kvcache_nhd_fp8.shape
    nhd_logical_view.copy_(kvcache_nhd_fp8)
    return nhd_logical_view


def _slice_kvc(kvcache_fp8: torch.Tensor, block_size: int):
    return (
        kvcache_fp8[:, 0, :block_size, :, :],  # kcache
        kvcache_fp8[:, 1, :block_size, :, :],  # vcache
        kvcache_fp8[:, 0, block_size:, :, :],  # kscale tail
    )


def _naive_reference_varlen(bundle, *, num_head_q: int, head_dim: int, block_size: int):
    """Run the naive bf16 reference per-row, then concat."""
    kvcache_fp8 = bundle["kvcache_fp8"]
    kcache = kvcache_fp8[:, 0, :block_size, :, :]
    vcache = kvcache_fp8[:, 1, :block_size, :, :]
    kscale = kvcache_fp8[:, 0, block_size:, :, :]
    qscale = bundle["qscale"]
    vscale = bundle["vscale"]
    block_ids = bundle["block_ids"]
    seqlens_kv = bundle["seqlens_kvcache"]

    outs = []
    for b, q_b in enumerate(bundle["q_per_seq_fp8"]):
        q_ref = q_b.unsqueeze(0)
        qs_ref = qscale[b : b + 1, :, : q_b.shape[0]].contiguous()
        page_ref = block_ids[b : b + 1].contiguous()
        cs_ref = seqlens_kv[b : b + 1].contiguous()
        out = naive_attn_with_kvcache_func(
            q=q_ref,
            k_cache=kcache,
            v_cache=vcache,
            qscale=qs_ref,
            kscale=kscale,
            vscale=vscale,
            cache_seqlens=cs_ref,
            page_table=page_ref,
        )
        outs.append(out.reshape(-1, q_b.shape[1], head_dim))
    return torch.cat(outs, dim=0)


def _run_kernel(bundle, num_head_q, scheduler, block_size, head_dim):
    kcache, vcache, kscale = _slice_kvc(bundle["kvcache_fp8"], block_size)
    total_q = bundle["q_packed"].shape[0]
    out = torch.empty((total_q, num_head_q, head_dim), dtype=torch.bfloat16, device="cuda")

    config = None
    if scheduler is not None:
        from dsl.attention import Fp8PagedPrefillConfig

        config = Fp8PagedPrefillConfig(is_persistent=(scheduler == "persistent"))

    got = hpc.attention_with_kvcache_prefill_fp8_packed_cutedsl(
        bundle["q_packed"],
        kcache,
        vcache,
        bundle["qscale"],
        kscale,
        bundle["vscale"],
        bundle["cu_seqlens_q"],
        bundle["block_ids"],
        bundle["seqlens_kvcache"],
        bundle["max_seq_q_pad"],
        output=out,
        config=config,
    )
    torch.cuda.synchronize()
    return got


# ============================================================================
# Test buckets
# ============================================================================
# Parameter axis 1: shape tuples (q_lens, kv_lens, Hq, Hkv).
# Mixed aligned + bad-case + GQA, with Hkv >= 4 cases included so the
# stride asymmetry between NHD and HND is **real** (Hkv == 1 degenerates).
_SHAPES = [
    pytest.param([128], [128], 8, 1, id="aligned_B1_q128_Hkv1"),
    pytest.param([1024], [1024], 8, 1, id="aligned_B1_q1024_Hkv1"),
    pytest.param([1024], [1024], 32, 4, id="aligned_B1_q1024_Hkv4_GQA"),
    pytest.param([192], [192], 8, 1, id="bad_B1_q192_Hkv1"),
    pytest.param([3938], [3938], 8, 1, id="bad_B1_q3938_Hkv1"),
    pytest.param([192], [192], 32, 4, id="bad_B1_q192_Hkv4_GQA"),
    pytest.param([3611, 3824, 3874, 3806], [3611, 3824, 3874, 3806], 8, 1, id="bad_B4_ragged_Hkv1"),
    pytest.param(
        [3611, 3824, 3874, 3806], [3611, 3824, 3874, 3806], 32, 4, id="bad_B4_ragged_Hkv4_GQA"
    ),
    pytest.param([128, 256, 512], [128, 256, 512], 8, 1, id="aligned_B3_Hkv1"),
    pytest.param([192, 1024], [4096, 4096], 8, 1, id="bad_B2_q_lt_kv"),
]


@requires_sm100
@pytest.mark.parametrize("scheduler", [None, "persistent", "static"])
@pytest.mark.parametrize("q_lens,kv_lens,num_head_q,num_head_kv", _SHAPES)
def test_nhd_matches_hnd_bit_for_bit(q_lens, kv_lens, num_head_q, num_head_kv, scheduler):
    """The kernel must produce identical bytes for NHD and HND inputs.

    Builds the same logical inputs twice — once with NHD-physical KV cache,
    once with HND-physical (via transpose view). Asserts:

      1. The two underlying allocations have **different strides**
         (Hkv != 1 cases — sanity check that the test actually exercises
         the stride asymmetry).
      2. The two backend outputs are byte-identical (``torch.equal``).
    """
    head_dim, block_size = 128, 64
    nhd = _build_inputs_nhd(q_lens, kv_lens, num_head_q, num_head_kv, head_dim, block_size)
    hnd_view = _to_hnd_view(nhd["kvcache_fp8"], num_head_kv, head_dim, block_size)

    # Sanity 0: HND view is logical-shape-equal to NHD source but
    # stride-different when Hkv > 1.
    if num_head_kv > 1:
        assert hnd_view.stride() != nhd["kvcache_fp8"].stride(), (
            f"Hkv={num_head_kv} should give NHD/HND stride asymmetry but got "
            f"identical strides: {hnd_view.stride()}"
        )

    # Re-bundle HND with the same non-cache tensors.
    hnd_bundle = dict(nhd)
    hnd_bundle["kvcache_fp8"] = hnd_view

    got_nhd = _run_kernel(nhd, num_head_q, scheduler, block_size, head_dim)
    got_hnd = _run_kernel(hnd_bundle, num_head_q, scheduler, block_size, head_dim)

    assert torch.equal(got_nhd, got_hnd), (
        f"NHD vs HND backend outputs diverge for q_lens={q_lens}, "
        f"Hq={num_head_q}, Hkv={num_head_kv}, sched={scheduler}; "
        f"max abs diff = {(got_nhd.float() - got_hnd.float()).abs().max().item():.6e}"
    )


@requires_sm100
@pytest.mark.parametrize("layout", ["NHD", "HND", "NHD_cross_layer", "HND_cross_layer"])
@pytest.mark.parametrize("scheduler", [None, "persistent", "static"])
@pytest.mark.parametrize("q_lens,kv_lens,num_head_q,num_head_kv", _SHAPES)
def test_layout_matches_naive_reference(
    layout, q_lens, kv_lens, num_head_q, num_head_kv, scheduler
):
    """Each of the 4 supported physical layouts must match the bf16
    naive reference (atol=0.1).

    We split bit-equality (test above) and reference-equivalence (this one)
    so a failure clearly points at either the layout-handling code or the
    kernel's numerical algorithm — not both.
    """
    head_dim, block_size = 128, 64
    nhd = _build_inputs_nhd(q_lens, kv_lens, num_head_q, num_head_kv, head_dim, block_size)

    bundle = dict(nhd)
    if layout == "NHD":
        pass  # nhd as-is
    elif layout == "HND":
        bundle["kvcache_fp8"] = _to_hnd_view(
            nhd["kvcache_fp8"],
            num_head_kv,
            head_dim,
            block_size,
        )
    elif layout == "NHD_cross_layer":
        bundle["kvcache_fp8"] = _to_nhd_cross_layer_view(
            nhd["kvcache_fp8"],
            num_head_kv,
            head_dim,
            block_size,
        )
    elif layout == "HND_cross_layer":
        bundle["kvcache_fp8"] = _to_hnd_cross_layer_view(
            nhd["kvcache_fp8"],
            num_head_kv,
            head_dim,
            block_size,
        )
    else:
        raise AssertionError(f"unknown layout {layout}")

    got = _run_kernel(bundle, num_head_q, scheduler, block_size, head_dim)
    assert torch.isfinite(got).all(), f"{layout}: non-finite output"

    ref = _naive_reference_varlen(
        nhd,
        num_head_q=num_head_q,
        head_dim=head_dim,
        block_size=block_size,
    )  # reference always built from NHD (kernel-independent computation)

    assert got.shape == ref.shape
    assert allclose(got, ref, atol=0.1), (
        f"{layout} backend diverges from naive reference "
        f"(q_lens={q_lens}, Hq={num_head_q}, Hkv={num_head_kv}, sched={scheduler})"
    )


# ============================================================================
# Cross-layer bit-equality — cross-layer view must produce byte-identical
# output as the same logical inputs through a non-cross-layer view.
#
# Run on the GQA Hkv=4 cases primarily (stride asymmetry is real); Hkv=1
# included for coverage.
# ============================================================================
_CROSS_LAYER_SHAPES = [
    pytest.param([128], [128], 8, 1, id="aligned_B1_q128_Hkv1"),
    pytest.param([1024], [1024], 32, 4, id="aligned_B1_q1024_Hkv4_GQA"),
    pytest.param([3938], [3938], 8, 1, id="bad_B1_q3938_Hkv1"),
    pytest.param([192], [192], 32, 4, id="bad_B1_q192_Hkv4_GQA"),
    pytest.param([3611, 3824, 3874, 3806], [3611, 3824, 3874, 3806], 8, 1, id="bad_B4_ragged_Hkv1"),
    pytest.param(
        [3611, 3824, 3874, 3806], [3611, 3824, 3874, 3806], 32, 4, id="bad_B4_ragged_Hkv4_GQA"
    ),
]


@requires_sm100
@pytest.mark.parametrize("base_layout", ["NHD", "HND"])
@pytest.mark.parametrize("scheduler", [None, "persistent", "static"])
@pytest.mark.parametrize("q_lens,kv_lens,num_head_q,num_head_kv", _CROSS_LAYER_SHAPES)
def test_cross_layer_matches_non_cross_layer_bit_for_bit(
    base_layout,
    q_lens,
    kv_lens,
    num_head_q,
    num_head_kv,
    scheduler,
):
    """The cross-layer variant of each base layout (NHD/HND) must produce
    byte-identical output as the non-cross-layer variant.

    Cross-layer adds a ``num_layers`` dim into the physical 6D KV cache
    buffer; slicing one layer out hands the kernel a 5D view
    whose strides carry a ``num_layers`` factor. The kernel's
    stride-derived TMA descriptor + K-scale offset path should absorb
    that factor without any logic change. This test is the regression
    that proves it.

    This test will fire whenever something silently starts assuming
    contiguous (non-cross-layer) KV strides — e.g. a future kernel
    edit that hard-codes ``Hkv * head_dim`` somewhere instead of using
    the runtime stride.
    """
    head_dim, block_size = 128, 64
    nhd = _build_inputs_nhd(q_lens, kv_lens, num_head_q, num_head_kv, head_dim, block_size)

    if base_layout == "NHD":
        baseline_view = nhd["kvcache_fp8"]
        cross_view = _to_nhd_cross_layer_view(
            nhd["kvcache_fp8"],
            num_head_kv,
            head_dim,
            block_size,
        )
    else:  # HND
        baseline_view = _to_hnd_view(
            nhd["kvcache_fp8"],
            num_head_kv,
            head_dim,
            block_size,
        )
        cross_view = _to_hnd_cross_layer_view(
            nhd["kvcache_fp8"],
            num_head_kv,
            head_dim,
            block_size,
        )

    # Sanity: cross-layer stride differs from non-cross-layer (P-stride and
    # Hkv-stride both carry a num_layers factor in cross-layer).
    assert cross_view.stride() != baseline_view.stride(), (
        f"cross-layer view should have stride != non-cross-layer "
        f"({cross_view.stride()} vs {baseline_view.stride()})"
    )

    baseline_bundle = dict(nhd)
    baseline_bundle["kvcache_fp8"] = baseline_view
    cross_bundle = dict(nhd)
    cross_bundle["kvcache_fp8"] = cross_view

    got_baseline = _run_kernel(baseline_bundle, num_head_q, scheduler, block_size, head_dim)
    got_cross = _run_kernel(cross_bundle, num_head_q, scheduler, block_size, head_dim)

    assert torch.equal(got_baseline, got_cross), (
        f"{base_layout} cross-layer diverges from non-cross-layer "
        f"(q_lens={q_lens}, Hq={num_head_q}, Hkv={num_head_kv}, sched={scheduler}); "
        f"max abs diff = {(got_baseline.float() - got_cross.float()).abs().max().item():.6e}"
    )


# ============================================================================
# Validate sanity tests — all 4 layouts must pass _validate_inputs.
# ============================================================================
@requires_sm100
def test_validate_accepts_nhd():
    """The simplest possible NHD allocation must pass validation +
    actually execute the kernel."""
    nhd = _build_inputs_nhd([128], [128], 8, 1)
    # Triggering execution implicitly exercises _validate_inputs.
    out = _run_kernel(nhd, num_head_q=8, scheduler=None, block_size=64, head_dim=128)
    assert out.shape == (128, 8, 128)


@requires_sm100
def test_validate_accepts_hnd():
    """An HND-stride view must pass validation, even with GQA (Hkv=4)
    where the stride asymmetry is real."""
    nhd = _build_inputs_nhd([128], [128], 32, 4)
    hnd_view = _to_hnd_view(nhd["kvcache_fp8"], 4, 128, 64)
    bundle = dict(nhd)
    bundle["kvcache_fp8"] = hnd_view
    out = _run_kernel(bundle, num_head_q=32, scheduler=None, block_size=64, head_dim=128)
    assert out.shape == (128, 32, 128)


@requires_sm100
def test_validate_accepts_nhd_cross_layer():
    """NHD with cross-layer (P-stride carries num_layers factor) must pass
    validation. GQA Hkv=4 to make stride asymmetry real."""
    nhd = _build_inputs_nhd([128], [128], 32, 4)
    cross_view = _to_nhd_cross_layer_view(nhd["kvcache_fp8"], 4, 128, 64)
    bundle = dict(nhd)
    bundle["kvcache_fp8"] = cross_view
    out = _run_kernel(bundle, num_head_q=32, scheduler=None, block_size=64, head_dim=128)
    assert out.shape == (128, 32, 128)


@requires_sm100
def test_validate_accepts_hnd_cross_layer():
    """HND with cross-layer (per-layer slice + transpose) must pass
    validation. GQA Hkv=4 to make stride asymmetry real."""
    nhd = _build_inputs_nhd([128], [128], 32, 4)
    cross_view = _to_hnd_cross_layer_view(nhd["kvcache_fp8"], 4, 128, 64)
    bundle = dict(nhd)
    bundle["kvcache_fp8"] = cross_view
    out = _run_kernel(bundle, num_head_q=32, scheduler=None, block_size=64, head_dim=128)
    assert out.shape == (128, 32, 128)


@requires_sm100
def test_validate_rejects_mixed_nhd_hnd():
    """Mixing K=NHD with V=HND must fail the stride-consistency check
    in _validate_inputs (kcache.stride() == vcache.stride())."""
    nhd = _build_inputs_nhd([128], [128], 8, 4)
    hnd_view = _to_hnd_view(nhd["kvcache_fp8"], 4, 128, 64)
    block_size = 64
    kcache_nhd = nhd["kvcache_fp8"][:, 0, :block_size, :, :]  # NHD
    vcache_hnd = hnd_view[:, 1, :block_size, :, :]  # HND
    kscale_nhd = nhd["kvcache_fp8"][:, 0, block_size:, :, :]
    out = torch.empty((128, 8, 128), dtype=torch.bfloat16, device="cuda")
    # K and V now live in different physical allocations entirely — the
    # data_ptr equality check (validates K/V/Kscale view-same-allocation)
    # should fire first and raise before the stride mismatch is reached.
    with pytest.raises(ValueError, match="(stride|same allocation|share)"):
        hpc.attention_with_kvcache_prefill_fp8_packed_cutedsl(
            nhd["q_packed"],
            kcache_nhd,
            vcache_hnd,
            nhd["qscale"],
            kscale_nhd,
            nhd["vscale"],
            nhd["cu_seqlens_q"],
            nhd["block_ids"],
            nhd["seqlens_kvcache"],
            nhd["max_seq_q_pad"],
            output=out,
        )
