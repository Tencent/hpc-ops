# Copyright (C) 2026 Tencent.
"""Tests and benchmarks for the Qwen3-TTS RoPE operator."""

import argparse
import os
import sys
from pathlib import Path

import pytest
import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

_build_libs = list(Path(__file__).parent.glob("../build/lib.*/"))
if not _build_libs:
    raise RuntimeError("build library not found; run setup.py build first")
sys.path.insert(0, os.path.realpath(_build_libs[0]))
import hpc  # noqa: E402

ROPE_THETA = 1_000_000.0
TTS_ROPE_CASES = {
    "qwen3_tts": (16, 16, 8),
    "qwen3_tts_full_groups": (17, 16, 8),
    "small_tts": (8, 8, 4),
    "gqa1": (17, 16, 1),
    "no_gqa": (16, 8, 8),
    "wide_heads": (32, 32, 8),
}
TTS_ROPE_SHAPES = (
    pytest.param("qwen3_tts", 16, 16, 8, id="qwen3_tts"),
    pytest.param("qwen3_tts_full_groups", 17, 16, 8, id="qwen3_tts_full_groups"),
    pytest.param("small_tts", 8, 8, 4, id="small_tts"),
    pytest.param("gqa1", 17, 16, 1, id="gqa1"),
    pytest.param("no_gqa", 16, 8, 8, id="no_gqa"),
    pytest.param("wide_heads", 32, 32, 8, id="wide_heads"),
)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def ref_tts_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def make_inputs(
    batch_size: int,
    seq_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(
        (batch_size, num_q_heads, seq_len, head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )
    k = torch.randn(
        (batch_size, num_kv_heads, seq_len, head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )
    inv_freq = 1.0 / (
        ROPE_THETA
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device="cuda") / head_dim)
    )
    position_ids = torch.arange(seq_len, device="cuda", dtype=torch.long).unsqueeze(0)
    position_ids = position_ids.expand(batch_size, -1).contiguous()
    inv_freq_expanded = inv_freq[None, :, None].float().expand(batch_size, -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    with torch.autocast(device_type="cuda", enabled=False):
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(
            1, 2
        )
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=torch.bfloat16)
        sin = emb.sin().to(dtype=torch.bfloat16)
    return q, k, cos, sin


if triton is not None:

    @triton.jit
    def _triton_tts_rope_one_tensor_kernel(
        x,
        cos,
        sin,
        y,
        seq_len: tl.constexpr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, block_size)
        half_dim = head_dim // 2
        pair_offs = tl.where(offs < half_dim, offs + half_dim, offs - half_dim)
        sign = tl.where(offs < half_dim, -1.0, 1.0)
        mask = offs < head_dim

        s = row % seq_len
        tmp = row // seq_len
        h = tmp % num_heads
        b = tmp // num_heads
        x_base = ((b * num_heads + h) * seq_len + s) * head_dim
        cos_base = (b * seq_len + s) * head_dim

        x_val = tl.load(x + x_base + offs, mask=mask)
        x_pair = tl.load(x + x_base + pair_offs, mask=mask)
        c = tl.load(cos + cos_base + offs, mask=mask)
        sn = tl.load(sin + cos_base + offs, mask=mask)
        y_val = x_val * c + sign * x_pair * sn
        tl.store(y + x_base + offs, y_val, mask=mask)

else:
    _triton_tts_rope_one_tensor_kernel = None


def triton_tts_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _triton_tts_rope_one_tensor_kernel is None:
        raise RuntimeError("triton is not available")
    batch_size, num_q_heads, seq_len, head_dim = q.shape
    num_kv_heads = k.shape[1]
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    _triton_tts_rope_one_tensor_kernel[(batch_size * num_q_heads * seq_len,)](
        q,
        cos,
        sin,
        q_out,
        seq_len,
        num_q_heads,
        head_dim,
        block_size=128,
    )
    _triton_tts_rope_one_tensor_kernel[(batch_size * num_kv_heads * seq_len,)](
        k,
        cos,
        sin,
        k_out,
        seq_len,
        num_kv_heads,
        head_dim,
        block_size=128,
    )
    return q_out, k_out


@pytest.mark.parametrize("batch_size", [1, 8, 32])
@pytest.mark.parametrize("shape_name,seq_len,num_q_heads,num_kv_heads", TTS_ROPE_SHAPES)
def test_qwen3_tts_rope_tts_shapes_golden(
    batch_size, shape_name, seq_len, num_q_heads, num_kv_heads
):
    del shape_name
    torch.manual_seed(0x20260709 + batch_size + seq_len + num_q_heads)
    q, k, cos, sin = make_inputs(batch_size, seq_len, num_q_heads, num_kv_heads)

    ref_q, ref_k = ref_tts_rope(q, k, cos, sin)
    out_q, out_k = hpc.qwen3_tts_rope(q, k, cos, sin)

    assert out_q.shape == q.shape
    assert out_k.shape == k.shape
    assert out_q.dtype == torch.bfloat16
    assert out_k.dtype == torch.bfloat16
    assert out_q.is_contiguous()
    assert out_k.is_contiguous()
    assert torch.allclose(out_q.float(), ref_q.float(), atol=1e-2, rtol=1e-2)
    assert torch.allclose(out_k.float(), ref_k.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(triton is None, reason="triton is not available")
def test_qwen3_tts_rope_triton_reference():
    torch.manual_seed(0x20260710)
    q, k, cos, sin = make_inputs(8, 16, 16, 8)
    ref_q, ref_k = ref_tts_rope(q, k, cos, sin)
    out_q, out_k = triton_tts_rope(q, k, cos, sin)
    assert torch.allclose(out_q.float(), ref_q.float(), atol=1e-2, rtol=1e-2)
    assert torch.allclose(out_k.float(), ref_k.float(), atol=1e-2, rtol=1e-2)


def test_qwen3_tts_rope_supports_strided_leading_dimensions():
    q = torch.randn((2, 16, 17, 130), dtype=torch.bfloat16, device="cuda")[..., :128]
    k = torch.randn((2, 8, 17, 130), dtype=torch.bfloat16, device="cuda")[..., :128]
    cos = torch.randn((2, 17, 130), dtype=torch.bfloat16, device="cuda")[..., :128]
    sin = torch.randn((2, 17, 130), dtype=torch.bfloat16, device="cuda")[..., :128]

    ref_q, ref_k = ref_tts_rope(q, k, cos, sin)
    out_q, out_k = hpc.qwen3_tts_rope(q, k, cos, sin)

    assert q.stride(3) == k.stride(3) == cos.stride(2) == sin.stride(2) == 1
    assert not q.is_contiguous()
    assert not k.is_contiguous()
    assert torch.allclose(out_q.float(), ref_q.float(), atol=1e-2, rtol=1e-2)
    assert torch.allclose(out_k.float(), ref_k.float(), atol=1e-2, rtol=1e-2)


def test_qwen3_tts_rope_rejects_bad_head_dim():
    q, k, cos, sin = make_inputs(1, 16, 16, 8, head_dim=64)
    with pytest.raises((RuntimeError, ValueError)):
        hpc.qwen3_tts_rope(q, k, cos, sin)


def test_qwen3_tts_rope_rejects_strided_last_dimension():
    q = torch.randn((1, 16, 16, 256), dtype=torch.bfloat16, device="cuda")[..., ::2]
    k = torch.randn((1, 8, 16, 256), dtype=torch.bfloat16, device="cuda")[..., ::2]
    cos = torch.randn((1, 16, 256), dtype=torch.bfloat16, device="cuda")[..., ::2]
    sin = torch.randn((1, 16, 256), dtype=torch.bfloat16, device="cuda")[..., ::2]

    with pytest.raises(
        (RuntimeError, ValueError), match="last dimension must be contiguous"
    ):
        hpc.qwen3_tts_rope(q, k, cos, sin)


def _time_event(fn, iters: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1e3


def _warmup(*fns, iters: int) -> None:
    for _ in range(iters):
        for fn in fns:
            fn()
    torch.cuda.synchronize()


def _reset_torch_compile_cache() -> None:
    try:
        import torch._dynamo as dynamo  # noqa: WPS433

        dynamo.reset()
    except Exception:
        pass


def _bench_shape(
    shape_name: str,
    seq_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    batch_size: int,
    iters: int,
    warmup: int,
) -> None:
    q, k, cos, sin = make_inputs(batch_size, seq_len, num_q_heads, num_kv_heads)
    _reset_torch_compile_cache()
    compiled_ref = torch.compile(
        ref_tts_rope, dynamic=False, options={"epilogue_fusion": False}
    )

    def eager_call():
        ref_tts_rope(q, k, cos, sin)

    def compile_call():
        compiled_ref(q, k, cos, sin)

    def hpc_call():
        hpc.qwen3_tts_rope(q, k, cos, sin)

    fns = [eager_call, compile_call, hpc_call]
    if _triton_tts_rope_one_tensor_kernel is not None:
        fns.append(lambda: triton_tts_rope(q, k, cos, sin))
    _warmup(*fns, iters=warmup)

    eager_us = _time_event(eager_call, iters)
    compile_us = _time_event(compile_call, iters)
    triton_us = (
        _time_event(lambda: triton_tts_rope(q, k, cos, sin), iters)
        if _triton_tts_rope_one_tensor_kernel is not None
        else float("nan")
    )
    hpc_us = _time_event(hpc_call, iters)
    speedup = (
        triton_us / hpc_us if triton_us == triton_us and hpc_us > 0 else float("nan")
    )
    rows = batch_size * seq_len * (num_q_heads + num_kv_heads)
    print(
        f"{shape_name:22s} bs={batch_size:3d} S={seq_len:2d} Hq={num_q_heads:2d} "
        f"Hkv={num_kv_heads:2d} rows={rows:5d} | eager={eager_us:8.2f}us "
        f"compile={compile_us:8.2f}us triton={triton_us:8.2f}us hpc={hpc_us:8.2f}us "
        f"triton/hpc={speedup:5.2f}x"
    )


def _select_shapes(names: str):
    if not names or names == "all":
        return [(name, *shape) for name, shape in TTS_ROPE_CASES.items()]
    want = {name.strip() for name in names.split(",") if name.strip()}
    missing = sorted(want - set(TTS_ROPE_CASES))
    if missing:
        raise ValueError(
            f"unknown shape names: {missing}; available={sorted(TTS_ROPE_CASES)}"
        )
    return [(name, *TTS_ROPE_CASES[name]) for name in want]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench", action="store_true", help="run timing table instead of pytest"
    )
    parser.add_argument(
        "--shapes", default="all", help="comma-separated shape names or 'all'"
    )
    parser.add_argument("--bs", default="1,8,32", help="comma-separated batch sizes")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    if not args.bench:
        parser.print_help()
        return 0

    batches = [int(x) for x in args.bs.split(",") if x]
    for shape in _select_shapes(args.shapes):
        for batch_size in batches:
            _bench_shape(*shape, batch_size, args.iters, args.warmup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
