"""OOB-regression correctness test for attention_with_kvcache_prefill_fp8_packed_cutedsl.

Targets the bug where TMA Q-loads run past the real packed-Q allocation
when any batch's seqlen_q is not a multiple of tile_q (=128), which a
q_len = 128-only parametrization would not catch.

This file enumerates ragged-batch shapes that exercise:

  * single-batch q_len ∈ {q % 128 == 0, q % 128 != 0, q < 128, q == 1};
  * multi-batch B ∈ {2, 4, 8} with mixed q % 128 residues;
  * GQA factor h_r ∈ {1, 8} and num_head_kv ∈ {1, 4};
  * persistent vs static schedulers (where the bug fired differently).

For every shape we compare the cutedsl kernel output against the
naive bf16 reference, with atol=0.1 tolerance (FP8 + softmax noise
floor).
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import List, Optional

import pytest
import torch

# Skip early if the CuTeDSL Python toolchain is not installed.
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


# ------------------------------------------------------------------
# Variable-length input builder. Upstream `build_inputs` only handles
# uniform batches; we extend it to per-row q_len / kv_len so we can
# exercise q_len % 128 != 0 with B>1.
# ------------------------------------------------------------------
def _build_varlen_inputs(
    q_lens: List[int],
    kv_lens: List[int],
    num_head_q: int,
    num_head_kv: int,
    head_dim: int = 128,
    block_size: int = 64,
    *,
    seed: int = 10086,
):
    assert len(q_lens) == len(kv_lens), "q_lens and kv_lens must have equal length"
    num_batch = len(q_lens)
    dtype = torch.bfloat16
    fp8 = torch.float8_e4m3fn
    max_q = max(q_lens)
    # The qscale seq dim is rounded up to a multiple of 128.
    max_seq_q_pad = ((max_q + 127) // 128) * 128
    scale_rows = block_size * 4 // head_dim  # =2 for D=128, block=64

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
    q_packed = torch.cat(q_per_seq, dim=0)  # [sum(q_lens), Hq, D]

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


def _naive_reference_varlen(bundle, *, num_head_q: int, head_dim: int, block_size: int):
    """Run the naive bf16 reference per-row, then concat back to packed.

    Upstream `naive_attn_with_kvcache_func` walks the batch dim with a Python
    for-loop and slices each row's KV by `cache_seqlens[i]`. It expects a
    uniform-batch [B, S_q, Hq, D] q shape, so we call it once per row with
    batch=1 and concatenate.
    """
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
        q_ref = q_b.unsqueeze(0)  # [1, ql, Hq, D]
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
        )  # [1, ql, Hq, D] bf16
        outs.append(out.reshape(-1, q_b.shape[1], head_dim))
    return torch.cat(outs, dim=0)


def _run_one_case(q_lens, kv_lens, num_head_q, num_head_kv, scheduler):
    head_dim, block_size = 128, 64
    bundle = _build_varlen_inputs(
        q_lens,
        kv_lens,
        num_head_q,
        num_head_kv,
        head_dim,
        block_size,
    )
    kcache = bundle["kvcache_fp8"][:, 0, :block_size, :, :]
    vcache = bundle["kvcache_fp8"][:, 1, :block_size, :, :]
    kscale = bundle["kvcache_fp8"][:, 0, block_size:, :, :]
    total_q = bundle["q_packed"].shape[0]
    out = torch.empty((total_q, num_head_q, head_dim), dtype=torch.bfloat16, device="cuda")
    max_q_padded = bundle["max_seq_q_pad"]

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
        max_q_padded,
        output=out,
        config=config,
    )
    torch.cuda.synchronize()

    # Sanity 1: no NaN / Inf escaped from the kernel.
    assert torch.isfinite(got).all(), (
        f"non-finite output: NaN={torch.isnan(got).sum().item()}, "
        f"Inf={torch.isinf(got).sum().item()}"
    )

    # Sanity 2: the output= path was used in-place.
    assert got.data_ptr() == out.data_ptr(), "output= must be used in-place"

    # Sanity 3: numerics match the bf16 reference within FP8 noise floor.
    ref = _naive_reference_varlen(
        bundle,
        num_head_q=num_head_q,
        head_dim=head_dim,
        block_size=block_size,
    )
    assert got.shape == ref.shape, f"shape mismatch: {got.shape} vs {ref.shape}"
    assert allclose(got, ref, atol=0.1), (
        f"hpc cutedsl diverges from naive reference for "
        f"q_lens={q_lens} kv_lens={kv_lens} Hq={num_head_q} Hkv={num_head_kv} "
        f"sched={scheduler}"
    )


# ============================================================================
# A. Single-batch boundary cases — each q_len exercises one residue class.
# ============================================================================
@requires_sm100
@pytest.mark.parametrize("scheduler", [None, "persistent", "static"])
@pytest.mark.parametrize(
    "q_len",
    [
        # Aligned (existing test coverage; baseline must still pass).
        128,
        256,
        1024,
        4096,
        # Below tile_q — the whole work tile is partial.
        1,
        17,
        64,
        127,
        # Just over a tile boundary — residue 1.
        129,
        # Mid-tile residues spanning the full mod-128 range.
        137,  # residue 9
        192,  # residue 64 (half-tile)
        255,  # residue 127 (just-shy of tile boundary)
        # Large partial shapes with assorted residues.
        3711,  # residue 1 with q_tile=29
        3938,  # residue 98 with q_tile=31
        3973,  # residue 5 with q_tile=32
    ],
)
def test_single_batch_partial_qlen(q_len, scheduler):
    """Each call covers one (q_len mod 128, scheduler) combination.

    q_len % 128 != 0 must produce correct output: the descriptor M-bound
    is total_q, and partial last-tile rows past seqlen_q are masked by TMA
    hardware.
    """
    _run_one_case(
        q_lens=[q_len],
        kv_lens=[q_len],
        num_head_q=8,
        num_head_kv=1,
        scheduler=scheduler,
    )


# ============================================================================
# B. Multi-batch ragged — exercised with B>=2 where ragged batching is the
#    interesting case. B=1 can mask OOB (the over-read rows may land within
#    the same allocation) so multi-batch coverage is the load-bearing part.
# ============================================================================
@requires_sm100
@pytest.mark.parametrize("scheduler", [None, "persistent", "static"])
@pytest.mark.parametrize(
    "q_lens",
    [
        # Ragged batch (B=4, all non-128).
        [3611, 3824, 3874, 3806],
        # Other B=4 mixes — last batch aligned, others ragged.
        [3711, 3824, 3874, 3840],
        # Power-of-two batch sizes commonly seen in serving.
        [127, 129, 256, 1023],  # B=4, mostly residue != 0
        [192, 256, 64, 128, 1, 17, 192, 1024],  # B=8 max-stress: q=1, q<128, residue 64, ...
        # B=2 minimal — fastest of the multi-batch cases for CI
        [192, 1024],
        [3938, 3711],
        # Boundary case — first batch is q < tile_q.
        [1, 256, 192, 1024],
    ],
)
def test_multi_batch_ragged(q_lens, scheduler):
    """B>=2, mixed q_len % 128 residues."""
    _run_one_case(
        q_lens=q_lens,
        kv_lens=q_lens,
        num_head_q=8,
        num_head_kv=1,
        scheduler=scheduler,
    )


# ============================================================================
# C. KV != Q lengths — different prefill mode where KV cache extends
#    beyond Q (e.g. continued-decode prefill). All q_len % 128 != 0
#    to exercise the OOB path; KV is independent.
# ============================================================================
@requires_sm100
@pytest.mark.parametrize("scheduler", [None, "persistent", "static"])
@pytest.mark.parametrize(
    "q_lens,kv_lens",
    [
        # Single-batch, KV >> Q.
        ([192], [4096]),
        ([3938], [8192]),
        # Multi-batch ragged Q, varied KV.
        ([192, 256, 1024], [512, 1024, 2048]),
        ([3611, 3874], [4096, 4096]),
    ],
)
def test_multi_batch_q_lt_kv(q_lens, kv_lens, scheduler):
    _run_one_case(
        q_lens=q_lens,
        kv_lens=kv_lens,
        num_head_q=8,
        num_head_kv=1,
        scheduler=scheduler,
    )


# ============================================================================
# D. GQA / MHA head-shape variations — h_r in {1, 8} × Hkv in {1, 4}.
#    Each case uses a non-128-aligned q_len so the original bug would fire.
# ============================================================================
@requires_sm100
@pytest.mark.parametrize("scheduler", [None, "persistent", "static"])
@pytest.mark.parametrize(
    "num_head_q,num_head_kv",
    [
        (8, 1),  # h_r=8 (GQA)
        (8, 4),  # h_r=2 (less aggressive GQA)
        (8, 8),  # h_r=1 (MHA)
        (32, 4),  # h_r=8 with larger total head count
    ],
)
def test_head_shape_variants(num_head_q, num_head_kv, scheduler):
    """All non-128-aligned q_lens to exercise OOB on each head shape."""
    _run_one_case(
        q_lens=[192, 1024, 3938],
        kv_lens=[192, 1024, 3938],
        num_head_q=num_head_q,
        num_head_kv=num_head_kv,
        scheduler=scheduler,
    )


# ============================================================================
# E. Aligned-baseline regression — make sure the OOB fix didn't break
#    the 100%-tile-aligned path the existing in-tree test covered.
# ============================================================================
@requires_sm100
@pytest.mark.parametrize("scheduler", [None, "persistent", "static"])
@pytest.mark.parametrize(
    "q_lens",
    [
        [128],
        [256],
        [1024],
        [4096],
        [128, 256, 512],  # all aligned, multi-batch
        [128, 128, 128, 128],  # uniform aligned (the legacy test case)
    ],
)
def test_aligned_qlen_regression(q_lens, scheduler):
    """Aligned shapes — must continue to match the bf16 reference."""
    _run_one_case(
        q_lens=q_lens,
        kv_lens=q_lens,
        num_head_q=8,
        num_head_kv=1,
        scheduler=scheduler,
    )
