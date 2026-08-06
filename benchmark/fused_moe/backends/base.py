# Copyright (C) 2026 Tencent.

"""Shared base classes and helpers for all FusedMoE backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Callable, Optional

import torch


# ---------------------------------------------------------------------------
# Per-cell shape spec
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BenchSpec:
    """Per-cell input shape that backends consume.

    Fields match the kernel-visible shape (post TP/EP simulation), not the
    full-model shape.  The driver derives these from the user's --tp/--ep
    args and stores them in the JSONL output for traceability.
    """
    num_seq: int                # batch_size
    hidden: int                 # K of Gate-Up; N of Down
    intermediate_per_rank: int  # N of one of {Gate, Up}; K of Down
    num_expert_local: int       # experts visible to this rank
    num_expert_total: int       # for sampling topk_ids; equals local under EP-rank-0 sim
    num_topk: int

    model: str = ""
    tp: int = 1
    ep: int = 1
    seed: int = 0
    dtype: str = "fp8"          # "fp8" (per-tensor) or "bf16"


# ---------------------------------------------------------------------------
# Backend ABC
# ---------------------------------------------------------------------------
class Backend(ABC):
    """Abstract backend.  Subclass for each registered benchmark backend."""

    name: str  # registry key, e.g. "hpcops"

    @abstractmethod
    def setup(self, spec: BenchSpec) -> Callable[[], None]:
        """Build tensors and return the timed call_fn."""
        raise NotImplementedError

    def cleanup(self) -> None:
        torch.cuda.empty_cache()

    def extra_metadata(self) -> dict:
        return {}


# ---------------------------------------------------------------------------
# Shared tensor builders
# ---------------------------------------------------------------------------
DTYPE_FP8 = torch.float8_e4m3fn
DTYPE_HALF = torch.half
DTYPE_BF16 = torch.bfloat16


def scaled_fp8_quant_local(x: torch.Tensor, scale: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    if scale is None:
        scale = x.float().abs().max().clamp_min(1e-6) / 448.0
    return (x.float() / scale).to(DTYPE_FP8), scale


try:
    from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant as _sglang_scaled_fp8_quant
    SGLANG_FP8_QUANT_AVAILABLE = True
except ImportError:
    _sglang_scaled_fp8_quant = None
    SGLANG_FP8_QUANT_AVAILABLE = False

try:
    from vllm import _custom_ops as ops
    VLLM_AVAILABLE = True
except ImportError:
    ops = None
    VLLM_AVAILABLE = False


def get_scaled_fp8_quant():
    if SGLANG_FP8_QUANT_AVAILABLE:
        return _sglang_scaled_fp8_quant
    if VLLM_AVAILABLE:
        return ops.scaled_fp8_quant
    return scaled_fp8_quant_local


def build_fp8_weights(
    num_expert_local: int,
    intermediate_per_rank: int,
    hidden: int,
    *,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build w1 (gate+up fused along N) and w2 (down) in fp8 + per-expert
    per-tensor scales.

    Layouts match Triton / CUTLASS / sglang convention:
        w1: [E, 2N, K]  (fp8)   w1_scale: [E, 1, 1] (fp32)
        w2: [E, K,  N]  (fp8)   w2_scale: [E, 1, 1] (fp32)

    Returns the same 4 tensors regardless of backend; HPC reshapes/views
    these to its own per-expert layout in its own backend module.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    E, N, K = num_expert_local, intermediate_per_rank, hidden
    scaled_fp8_quant = get_scaled_fp8_quant()

    w1_half = torch.randn(
        (E, 2 * N, K), dtype=torch.float, device="cuda", generator=g,
    ).to(DTYPE_HALF)
    w2_half = torch.randn(
        (E, K, N), dtype=torch.float, device="cuda", generator=g,
    ).to(DTYPE_HALF)

    w1_fp8 = torch.empty_like(w1_half, dtype=DTYPE_FP8)
    w2_fp8 = torch.empty_like(w2_half, dtype=DTYPE_FP8)
    w1_scale = torch.empty((E, 1, 1), device="cuda", dtype=torch.float32)
    w2_scale = torch.empty((E, 1, 1), device="cuda", dtype=torch.float32)
    for e in range(E):
        w1_fp8[e], s1 = scaled_fp8_quant(w1_half[e])
        w2_fp8[e], s2 = scaled_fp8_quant(w2_half[e])
        w1_scale[e, 0, 0] = s1
        w2_scale[e, 0, 0] = s2
    return w1_fp8, w2_fp8, w1_scale, w2_scale


def build_bf16_weights(
    num_expert_local: int,
    intermediate_per_rank: int,
    hidden: int,
    *,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build w1 (gate+up fused along N) and w2 (down) as unquantized bf16.

    Layout matches build_fp8_weights (and HPC-Ops / sglang convention):
        w1: [E, 2N, K]  (bf16)
        w2: [E, K,  N]  (bf16)
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    E, N, K = num_expert_local, intermediate_per_rank, hidden
    w1 = torch.randn(
        (E, 2 * N, K), dtype=torch.float, device="cuda", generator=g,
    ).to(DTYPE_BF16)
    w2 = torch.randn(
        (E, K, N), dtype=torch.float, device="cuda", generator=g,
    ).to(DTYPE_BF16)
    return w1, w2


def build_routing(
    num_seq: int,
    num_expert_total: int,
    num_topk: int,
    *,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample uniform `topk_ids` and a normalized `topk_weights`.

    Returns:
        topk_ids   : (num_seq, num_topk) int32, sorted along topk axis
        topk_w     : (num_seq, num_topk) float32, softmax-normalized
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    topk_ids = torch.stack([
        torch.sort(
            torch.randperm(
                num_expert_total, dtype=torch.int32, device="cuda",
                generator=g,
            )[:num_topk]
        ).values
        for _ in range(num_seq)
    ])
    topk_w = torch.softmax(
        torch.randn((num_seq, num_topk), dtype=torch.float32, device="cuda",
                    generator=g),
        dim=-1,
    )
    return topk_ids, topk_w


def build_activation(
    num_seq: int, hidden: int, *, seed: int = 0, dtype: torch.dtype = DTYPE_HALF,
) -> torch.Tensor:
    """Build an activation tensor (half by default, bf16 for the bf16 path)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(
        (num_seq, hidden), dtype=dtype, device="cuda", generator=g,
    ) / 10


A_SCALE_VALUE = 1e-2


def build_a_scale() -> torch.Tensor:
    return torch.full((), A_SCALE_VALUE, device="cuda", dtype=torch.float32)


# ---------------------------------------------------------------------------
# Method C timing harness
# ---------------------------------------------------------------------------
def run_method_c(call_fn: Callable[[], None], *, warmup: int = 3, n_timed: int = 52):
    """Run warmup, graph capture, replay warmup, and timed graph replays.

    Timing is read from Nsight Systems' NVTX GPU projected duration. Do not
    synchronize inside each timed range; synchronize once after all measured
    ranges have been submitted.
    """
    for _ in range(warmup):
        call_fn()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call_fn()

    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(n_timed):
        torch.cuda.nvtx.range_push("step")
        try:
            graph.replay()
        finally:
            torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


# ---------------------------------------------------------------------------
# Spec serialization (worker stdin <-> driver)
# ---------------------------------------------------------------------------
def spec_to_argv(spec: BenchSpec) -> list[str]:
    """Serialize a BenchSpec to argv for invoking worker.py."""
    return [
        "--num-seq", str(spec.num_seq),
        "--hidden", str(spec.hidden),
        "--intermediate-per-rank", str(spec.intermediate_per_rank),
        "--num-expert-local", str(spec.num_expert_local),
        "--num-expert-total", str(spec.num_expert_total),
        "--num-topk", str(spec.num_topk),
        "--model", spec.model,
        "--tp", str(spec.tp),
        "--ep", str(spec.ep),
        "--seed", str(spec.seed),
        "--dtype", spec.dtype,
    ]


def spec_from_args(args) -> BenchSpec:
    return BenchSpec(
        num_seq=args.num_seq,
        hidden=args.hidden,
        intermediate_per_rank=args.intermediate_per_rank,
        num_expert_local=args.num_expert_local,
        num_expert_total=args.num_expert_total,
        num_topk=args.num_topk,
        model=args.model,
        tp=args.tp,
        ep=args.ep,
        seed=args.seed,
        dtype=getattr(args, "dtype", "fp8"),
    )
