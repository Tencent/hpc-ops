"""Correctness test for attention_with_kvcache_prefill_fp8_packed_cutedsl.

Uses a ground-truth input builder and naive fp32 reference. When the
CuTeDSL toolchain is unavailable the whole module is skipped.
"""

import math
import os
import sys
from pathlib import Path

import pytest
import torch

# ------------------------------------------------------------------
# Skip early if the CuTeDSL Python toolchain is not installed. This must
# happen before importing `hpc`, because the call-site inside `hpc` only
# lazy-imports cute at first invocation.
# ------------------------------------------------------------------
pytest.importorskip("cutlass.cute")
pytest.importorskip("cuda.bindings.driver")

# Reference helpers live next to this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _reference_helpers import (  # noqa: E402
    allclose,
    build_inputs,
    naive_attn_with_kvcache_func,
)

import hpc  # noqa: E402


def _sm100_available() -> bool:
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    return cap[0] >= 10


requires_sm100 = pytest.mark.skipif(not _sm100_available(), reason="requires SM100 (Blackwell)")


@requires_sm100
def test_early_check_fails_on_non_sm100(monkeypatch):
    """Ensure the early arch check returns a readable error on non-SM100."""
    # Fake a non-SM100 capability to exercise the early guard.
    real_cap = torch.cuda.get_device_capability

    def fake_cap(*args, **kwargs):  # noqa: ANN001
        return (9, 0)

    monkeypatch.setattr(torch.cuda, "get_device_capability", fake_cap)
    try:
        q = torch.empty((1, 1, 128), dtype=torch.float8_e4m3fn, device="cuda")
        with pytest.raises(RuntimeError, match="requires SM100"):
            hpc.attention_with_kvcache_prefill_fp8_packed_cutedsl(
                q,
                q,
                q,
                q,
                q,
                q,
                q,
                q,
                q,
                1,
            )
    finally:
        monkeypatch.setattr(torch.cuda, "get_device_capability", real_cap)


@requires_sm100
@pytest.mark.parametrize("use_output", [False, True])
@pytest.mark.parametrize(
    "num_batch,num_seq_q,num_seq_kv,num_head_q,num_head_kv",
    [
        (1, 128, 128, 8, 1),
    ],
)
def test_cutedsl_matches_naive_reference(
    num_batch, num_seq_q, num_seq_kv, num_head_q, num_head_kv, use_output
):
    """End-to-end correctness vs. the naive bf16 fp32 reference."""
    head_dim = 128
    block_size = 64

    q, kvcache_fp8, qscale, vscale, cu_q, block_ids, seqlens_kvcache = build_inputs(
        num_batch, num_seq_q, num_seq_kv, num_head_q, num_head_kv, head_dim, block_size
    )
    kcache = kvcache_fp8[:, 0, :block_size, :, :]
    vcache = kvcache_fp8[:, 1, :block_size, :, :]
    kscale = kvcache_fp8[:, 0, block_size:, :, :]

    output = None
    if use_output:
        output = torch.empty(
            (num_batch * num_seq_q, num_head_q, head_dim),
            dtype=torch.bfloat16,
            device="cuda",
        )

    got = hpc.attention_with_kvcache_prefill_fp8_packed_cutedsl(
        q.reshape(-1, num_head_q, head_dim),
        kcache,
        vcache,
        qscale,
        kscale,
        vscale,
        cu_q,
        block_ids,
        seqlens_kvcache,
        num_seq_q,
        output=output,
    )
    if use_output:
        assert got.data_ptr() == output.data_ptr(), "output= must be used in-place"

    ref = naive_attn_with_kvcache_func(
        q=q,
        k_cache=kcache,
        v_cache=vcache,
        qscale=qscale,
        kscale=kscale,
        vscale=vscale,
        cache_seqlens=seqlens_kvcache,
        page_table=block_ids,
    ).reshape(-1, num_head_q, head_dim)

    assert got.shape == ref.shape
    assert got.dtype == torch.bfloat16
    assert allclose(got, ref, atol=0.1), (
        f"hpc cutedsl backend diverges from naive reference "
        f"(shape={got.shape}, dtype={got.dtype})"
    )
