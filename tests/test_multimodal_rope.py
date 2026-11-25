# Copyright (C) 2026 Tencent.
"""Multimodal RoPE tests and FlashInfer benchmark."""

import math
import os
import statistics
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest
import torch

_build_libs = list(Path(__file__).parent.glob("../build/lib.*/"))
if not _build_libs:
    raise RuntimeError("build library not found; run setup.py build first")
sys.path.insert(0, os.path.realpath(_build_libs[0]))

import hpc  # noqa: E402


@dataclass(frozen=True)
class RopeCase:
    profile: str
    stage: str
    batch_size: int
    seq_len: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    is_neox: bool

    @property
    def case_id(self) -> str:
        return f"{self.profile}-{self.stage}-b{self.batch_size}-s{self.seq_len}"


ROPE_PROFILES = (
    ("hq16-hkv8-d128-neox", 16, 8, 128, True),
    ("hq28-hkv4-d128-neox", 28, 4, 128, True),
    ("hq32-hkv8-d128-neox", 32, 8, 128, True),
    ("hq64-hkv8-d128-neox", 64, 8, 128, True),
    ("hq64-hkv4-d128-neox", 64, 4, 128, True),
    ("hq16-hkv8-d512-neox", 16, 8, 512, True),
    ("hq28-hkv4-d512-neox", 28, 4, 512, True),
    ("hq32-hkv8-d512-neox", 32, 8, 512, True),
    ("hq64-hkv8-d512-neox", 64, 8, 512, True),
    ("hq64-hkv4-d512-neox", 64, 4, 512, True),
    ("hq32-hkv1-d64-interleaved", 32, 1, 64, False),
    ("hq64-hkv1-d64-interleaved", 64, 1, 64, False),
)


def make_cases(batches: tuple[int, ...], seq_lens: tuple[int, ...]) -> list[RopeCase]:
    cases = []
    for profile, q_heads, kv_heads, head_dim, is_neox in ROPE_PROFILES:
        for batch_size in batches:
            for seq_len in seq_lens:
                cases.append(
                    RopeCase(
                        profile=profile,
                        stage="decode" if seq_len == 1 else "chunk-prefill",
                        batch_size=batch_size,
                        seq_len=seq_len,
                        num_q_heads=q_heads,
                        num_kv_heads=kv_heads,
                        head_dim=head_dim,
                        is_neox=is_neox,
                    )
                )
    return cases


TEST_CASES = make_cases((8, 32, 128), (1, 4, 16))


def make_cache_and_positions(
    batch_size: int,
    seq_len: int,
    head_dim: int,
    rope_theta: float = 1_000_000.0,
    position_base: int = 11,
) -> tuple[torch.Tensor, torch.Tensor]:
    frequency_ids = torch.arange(0, head_dim, 2, dtype=torch.float32, device="cuda")
    inv_freq = 1.0 / rope_theta ** (frequency_ids / head_dim)
    max_position = position_base + seq_len + 1
    positions = torch.arange(
        position_base,
        position_base + seq_len,
        dtype=torch.int64,
        device="cuda",
    ).unsqueeze(0)
    positions = positions.expand(batch_size, -1).contiguous()
    all_positions = torch.arange(max_position, dtype=torch.float32, device="cuda")
    frequencies = torch.outer(all_positions, inv_freq)
    cache = torch.cat((frequencies.cos(), frequencies.sin()), dim=-1)
    return cache, positions


def make_inputs(
    case: RopeCase,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_bshd = torch.randn(
        (
            case.batch_size,
            case.seq_len,
            case.num_q_heads,
            case.head_dim,
        ),
        dtype=torch.bfloat16,
        device="cuda",
    )
    k_bshd = torch.randn(
        (
            case.batch_size,
            case.seq_len,
            case.num_kv_heads,
            case.head_dim,
        ),
        dtype=torch.bfloat16,
        device="cuda",
    )
    cache, positions = make_cache_and_positions(case.batch_size, case.seq_len, case.head_dim)
    return q_bshd.transpose(1, 2), k_bshd.transpose(1, 2), cache, positions


def flashinfer_inputs(
    case: RopeCase,
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_flat = q.transpose(1, 2).reshape(-1, case.num_q_heads * case.head_dim)
    k_flat = k.transpose(1, 2).reshape(-1, case.num_kv_heads * case.head_dim)
    return positions.reshape(-1), q_flat.contiguous(), k_flat.contiguous()


def _rotate_half(x: torch.Tensor, is_neox: bool) -> torch.Tensor:
    if is_neox:
        first, second = x.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)
    even = x[..., ::2]
    odd = x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).reshape_as(x)


def reference_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cache: torch.Tensor,
    positions: torch.Tensor,
    is_neox: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    half_dim = cache.shape[-1] // 2
    gathered = cache[positions]
    cos_half = gathered[..., :half_dim]
    sin_half = gathered[..., half_dim:]
    if is_neox:
        cos = torch.cat((cos_half, cos_half), dim=-1)
        sin = torch.cat((sin_half, sin_half), dim=-1)
    else:
        cos = cos_half.repeat_interleave(2, dim=-1)
        sin = sin_half.repeat_interleave(2, dim=-1)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_float = q.float()
    k_float = k.float()
    q_out = q_float * cos + _rotate_half(q_float, is_neox) * sin
    k_out = k_float * cos + _rotate_half(k_float, is_neox) * sin
    return q_out, k_out


@pytest.mark.parametrize("case", TEST_CASES, ids=lambda case: case.case_id)
def test_multimodal_rope_matches_reference(case: RopeCase) -> None:
    torch.manual_seed(0x20260803 + case.batch_size + case.seq_len)
    q, k, cache, positions = make_inputs(case)
    expected_q, expected_k = reference_rope(q, k, cache, positions, case.is_neox)

    actual_q, actual_k = hpc.multimodal_rope(q, k, cache, positions, case.is_neox)

    assert actual_q.stride() == q.stride()
    assert actual_k.stride() == k.stride()
    torch.testing.assert_close(actual_q.float(), expected_q, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(actual_k.float(), expected_k, atol=1e-2, rtol=1e-2)


def _representative_case() -> RopeCase:
    return RopeCase("hq32-hkv8-d128-neox", "chunk-prefill", 8, 4, 32, 8, 128, True)


def test_multimodal_rope_preallocated_outputs() -> None:
    case = _representative_case()
    q, k, cache, positions = make_inputs(case)
    expected_q, expected_k = reference_rope(q, k, cache, positions, True)
    q_storage = torch.empty((*q.shape[:-1], q.shape[-1] + 2), dtype=q.dtype, device=q.device)
    k_storage = torch.empty((*k.shape[:-1], k.shape[-1] + 2), dtype=k.dtype, device=k.device)
    q_out = torch.as_strided(q_storage, q.shape, q_storage.stride())
    k_out = torch.as_strided(k_storage, k.shape, k_storage.stride())

    returned_q, returned_k = hpc.multimodal_rope(
        q,
        k,
        cache,
        positions,
        True,
        out_q=q_out,
        out_k=k_out,
    )

    assert returned_q.data_ptr() == q_out.data_ptr()
    assert returned_k.data_ptr() == k_out.data_ptr()
    torch.testing.assert_close(returned_q.float(), expected_q, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(returned_k.float(), expected_k, atol=1e-2, rtol=1e-2)


def test_multimodal_rope_requires_both_outputs() -> None:
    case = _representative_case()
    q, k, cache, positions = make_inputs(case)
    q_out = torch.empty_strided(q.shape, q.stride(), dtype=q.dtype, device=q.device)
    with pytest.raises(ValueError, match="provided together"):
        hpc.multimodal_rope(q, k, cache, positions, out_q=q_out)


def test_multimodal_rope_rejects_unsupported_profile() -> None:
    case = RopeCase("unsupported", "decode", 8, 1, 12, 4, 128, True)
    q, k, cache, positions = make_inputs(case)
    with pytest.raises(RuntimeError, match="supports NeoX"):
        hpc.multimodal_rope(q, k, cache, positions)


@pytest.mark.parametrize(
    ("transform", "message"),
    (
        (lambda q, k, cache, positions: (q.float(), k, cache, positions), "bfloat16"),
        (
            lambda q, k, cache, positions: (
                q,
                k,
                cache.to(torch.bfloat16),
                positions,
            ),
            "float32",
        ),
        (
            lambda q, k, cache, positions: (
                q,
                k,
                cache,
                positions.to(torch.int32),
            ),
            "int64",
        ),
    ),
)
def test_multimodal_rope_rejects_bad_dtypes(transform, message: str) -> None:
    q, k, cache, positions = make_inputs(_representative_case())
    q, k, cache, positions = transform(q, k, cache, positions)
    with pytest.raises(RuntimeError, match=message):
        hpc.multimodal_rope(q, k, cache, positions)


def test_multimodal_rope_rejects_strided_last_dimension() -> None:
    q, k, cache, positions = make_inputs(_representative_case())
    q = torch.empty((*q.shape, 2), dtype=q.dtype, device=q.device)[..., 0]
    with pytest.raises(RuntimeError, match="last dimension"):
        hpc.multimodal_rope(q, k, cache, positions)


def test_multimodal_rope_rejects_input_output_overlap() -> None:
    q, k, cache, positions = make_inputs(_representative_case())
    k_out = torch.empty_strided(k.shape, k.stride(), dtype=k.dtype, device=k.device)
    with pytest.raises(RuntimeError):
        hpc.multimodal_rope(q, k, cache, positions, out_q=q, out_k=k_out)


def _leading_strided(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    storage_shape = (*shape[:-1], shape[-1] + 8)
    storage = torch.randn(storage_shape, dtype=dtype, device="cuda")
    return torch.as_strided(storage, shape, storage.stride())


def test_multimodal_rope_large_strided_layout() -> None:
    q = _leading_strided((32, 16, 32, 128), torch.bfloat16)
    k = _leading_strided((32, 8, 32, 128), torch.bfloat16)
    cache, positions = make_cache_and_positions(32, 32, 128)
    expected_q, expected_k = reference_rope(q, k, cache, positions, True)

    actual_q, actual_k = hpc.multimodal_rope(q, k, cache, positions)

    assert actual_q.stride() == q.stride()
    assert actual_k.stride() == k.stride()
    torch.testing.assert_close(actual_q.float(), expected_q, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(actual_k.float(), expected_k, atol=1e-2, rtol=1e-2)


def test_multimodal_rope_supports_strided_leading_dimensions() -> None:
    q = _leading_strided((2, 16, 17, 128), torch.bfloat16)
    k = _leading_strided((2, 8, 17, 128), torch.bfloat16)
    cache = _leading_strided((20, 128), torch.float32)
    position_storage = torch.zeros((2, 34), dtype=torch.int64, device="cuda")
    positions = torch.as_strided(position_storage, (2, 17), (34, 2))
    positions.copy_(torch.arange(1, 18, dtype=torch.int64, device="cuda").expand(2, -1))
    expected_q, expected_k = reference_rope(q, k, cache, positions, True)

    actual_q, actual_k = hpc.multimodal_rope(q, k, cache, positions)

    assert not q.is_contiguous()
    assert not k.is_contiguous()
    assert not cache.is_contiguous()
    assert not positions.is_contiguous()
    torch.testing.assert_close(actual_q.float(), expected_q, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(actual_k.float(), expected_k, atol=1e-2, rtol=1e-2)


def test_multimodal_rope_rejects_wrong_cache_width() -> None:
    q, k, cache, positions = make_inputs(_representative_case())
    with pytest.raises(RuntimeError, match="last dimension"):
        hpc.multimodal_rope(q, k, cache[:, :64], positions)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
def test_multimodal_rope_uses_input_device() -> None:
    case = _representative_case()
    with torch.cuda.device(1):
        q, k, cache, positions = make_inputs(case)
        expected_q, expected_k = reference_rope(q, k, cache, positions, True)
    with torch.cuda.device(0):
        actual_q, actual_k = hpc.multimodal_rope(q, k, cache, positions)
    assert actual_q.device == q.device
    assert actual_k.device == k.device
    torch.testing.assert_close(actual_q.float(), expected_q, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(actual_k.float(), expected_k, atol=1e-2, rtol=1e-2)


def test_multimodal_rope_rejects_overlapping_input_layout() -> None:
    case = _representative_case()
    q, k, cache, positions = make_inputs(case)
    q_overlap = q[:, :1].expand(-1, case.num_q_heads, -1, -1)
    with pytest.raises(RuntimeError):
        hpc.multimodal_rope(q_overlap, k, cache, positions)


def test_multimodal_rope_rejects_output_output_overlap() -> None:
    q, k, cache, positions = make_inputs(_representative_case())
    storage = torch.empty(q.numel(), dtype=q.dtype, device=q.device)
    q_out = storage.view(q.shape)
    k_out = storage[: k.numel()].view(k.shape)
    with pytest.raises(RuntimeError):
        hpc.multimodal_rope(q, k, cache, positions, out_q=q_out, out_k=k_out)


BENCH_WARMUP = int(os.getenv("HPC_OPS_BENCH_WARMUP", "500"))
BENCH_ITERS = int(os.getenv("HPC_OPS_BENCH_ITERS", "500"))
BENCH_REPEATS = int(os.getenv("HPC_OPS_BENCH_REPEATS", "8"))
BENCH_GRAPH_CALLS = int(os.getenv("HPC_OPS_BENCH_GRAPH_CALLS", "16"))
BENCH_CASES = make_cases((8, 16, 32, 64, 128), (1, 2, 4, 8, 16))


def _package_version(*names: str) -> str:
    for name in names:
        try:
            return version(name)
        except PackageNotFoundError:
            pass
    return "unknown"


def _geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _capture_graph(fn) -> torch.cuda.CUDAGraph:
    for _ in range(BENCH_WARMUP):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(BENCH_GRAPH_CALLS):
            fn()
    torch.cuda.synchronize()
    return graph


def _time_graph(graph: torch.cuda.CUDAGraph) -> float:
    samples = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    graph.replay()
    torch.cuda.synchronize()
    for _ in range(BENCH_REPEATS):
        start.record()
        for _ in range(BENCH_ITERS):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / BENCH_ITERS / BENCH_GRAPH_CALLS)
    return statistics.median(samples)


def _capture_and_time(first_call, second_call) -> tuple[float, float]:
    first_graph = _capture_graph(first_call)
    second_graph = _capture_graph(second_call)
    return _time_graph(first_graph), _time_graph(second_graph)


def _verify_benchmark_outputs(
    case: RopeCase,
    q: torch.Tensor,
    k: torch.Tensor,
    cache: torch.Tensor,
    positions: torch.Tensor,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    q_flat_out: torch.Tensor,
    k_flat_out: torch.Tensor,
) -> None:
    expected_q, expected_k = reference_rope(q, k, cache, positions, case.is_neox)
    flashinfer_q = q_flat_out.view(
        case.batch_size,
        case.seq_len,
        case.num_q_heads,
        case.head_dim,
    ).transpose(1, 2)
    flashinfer_k = k_flat_out.view(
        case.batch_size,
        case.seq_len,
        case.num_kv_heads,
        case.head_dim,
    ).transpose(1, 2)
    for actual, expected in (
        (q_out, expected_q),
        (k_out, expected_k),
        (flashinfer_q, expected_q),
        (flashinfer_k, expected_k),
    ):
        torch.testing.assert_close(actual.float(), expected, atol=2e-2, rtol=1e-2)


def _benchmark_case(case: RopeCase, flashinfer_rope, reverse: bool) -> dict:
    torch.manual_seed(0xBADC0DE + case.batch_size * 17 + case.seq_len)
    q, k, cache, positions = make_inputs(case)
    positions_flat, q_flat, k_flat = flashinfer_inputs(case, q, k, positions)
    q_out = torch.empty_strided(q.shape, q.stride(), dtype=q.dtype, device=q.device)
    k_out = torch.empty_strided(k.shape, k.stride(), dtype=k.dtype, device=k.device)
    q_flat_out = torch.empty_like(q_flat)
    k_flat_out = torch.empty_like(k_flat)
    q_flat = q_flat.view(-1, case.num_q_heads, case.head_dim)
    k_flat = k_flat.view(-1, case.num_kv_heads, case.head_dim)
    q_flat_out_3d = q_flat_out.view(-1, case.num_q_heads, case.head_dim)
    k_flat_out_3d = k_flat_out.view(-1, case.num_kv_heads, case.head_dim)

    def hpc_call() -> None:
        hpc.multimodal_rope(
            q,
            k,
            cache,
            positions,
            case.is_neox,
            out_q=q_out,
            out_k=k_out,
        )

    def flashinfer_call() -> None:
        flashinfer_rope._apply_rope_pos_ids_cos_sin_cache(
            q_flat,
            k_flat,
            q_flat_out_3d,
            k_flat_out_3d,
            cache,
            positions_flat,
            interleave=not case.is_neox,
        )

    hpc_call()
    flashinfer_call()
    _verify_benchmark_outputs(
        case,
        q,
        k,
        cache,
        positions,
        q_out,
        k_out,
        q_flat_out,
        k_flat_out,
    )
    if reverse:
        flashinfer_us, hpc_us = _capture_and_time(flashinfer_call, hpc_call)
    else:
        hpc_us, flashinfer_us = _capture_and_time(hpc_call, flashinfer_call)
    return {
        "case": case,
        "hpc_us": hpc_us,
        "flashinfer_us": flashinfer_us,
        "speedup": flashinfer_us / hpc_us,
    }


def _print_benchmark_result(result: dict) -> None:
    case = result["case"]
    print(
        f"profile={case.profile:<31} stage={case.stage:<13} "
        f"B={case.batch_size:<3} S={case.seq_len:<2} "
        f"hpc_amortized={result['hpc_us']:7.3f}us "
        f"flashinfer_amortized={result['flashinfer_us']:7.3f}us "
        f"speedup={result['speedup']:5.2f}x"
    )


def _print_benchmark_summary(results: list[dict]) -> None:
    print("\nSummary")
    groups = sorted({(result["case"].profile, result["case"].stage) for result in results})
    for profile, stage in groups:
        speedups = [
            result["speedup"]
            for result in results
            if result["case"].profile == profile and result["case"].stage == stage
        ]
        print(
            f"profile={profile:<31} stage={stage:<13} "
            f"geomean={_geomean(speedups):.3f}x "
            f"min={min(speedups):.3f}x n={len(speedups)}"
        )
    speedups = [result["speedup"] for result in results]
    print(
        f"overall geomean={_geomean(speedups):.3f}x "
        f"min={min(speedups):.3f}x max={max(speedups):.3f}x "
        f"n={len(speedups)}"
    )


@pytest.mark.skipif(
    os.getenv("HPC_OPS_RUN_BENCHMARK") != "1",
    reason="set HPC_OPS_RUN_BENCHMARK=1 to run the performance benchmark",
)
def test_multimodal_rope_benchmark() -> None:
    flashinfer_rope = pytest.importorskip("flashinfer.rope")

    major, minor = torch.cuda.get_device_capability()
    print(
        f"gpu={torch.cuda.get_device_name()} sm={major}{minor} "
        f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"flashinfer={_package_version('flashinfer-python', 'flashinfer')} "
        f"warmup={BENCH_WARMUP} iters={BENCH_ITERS} "
        f"repeats={BENCH_REPEATS} graph_calls={BENCH_GRAPH_CALLS} "
        f"metric=amortized_us cases={len(BENCH_CASES)}"
    )
    results = []
    for index, case in enumerate(BENCH_CASES):
        result = _benchmark_case(case, flashinfer_rope, bool(index % 2))
        results.append(result)
        _print_benchmark_result(result)
    _print_benchmark_summary(results)
