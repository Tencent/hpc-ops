"""Bench mxfp8_mxfp4 group GEMM: TMA vs cp.async paths.

Compares hpc.group_gemm_mxfp8 (TMA path) vs hpc.group_gemm_cp_async_mxfp8
(cp.async path) under mxfp8 activation x mxfp4 weight (auto-detected from
uint8 + K/2 weight dim).

Usage:
    python tests/bench_group_gemm_mxfp4_tma_vs_cpasync.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))


import torch
import torch.cuda.nvtx as nvtx

import hpc

# ---------------------------- config ----------------------------

NUM_EXPERT_TOTAL = 256
NUM_TOPK = 8

# Total token counts (m = num_seq) to sweep.
M_CASES = [256, 512, 1024, 2048, 4096, 8192]

# hy4.0  (label, n, k, num_group)
#   tp8/tp4: all 256 experts local (num_group=256)
#   ep8: 256/8=32 experts per card (num_group=32)
SHAPES = [
    ("gate_up_tp8", 512, 6144, 256),
    ("down_tp8", 6144, 256, 256),
    ("gate_up_tp4", 1024, 6144, 256),
    ("down_tp4", 6144, 512, 256),
    ("gate_up_ep8", 4096, 6144, 32),
    ("down_ep8", 6144, 2048, 32),
]

SF_VEC = 32
WARMUP = 5
ITER = 100


# ---------------------------- builders ----------------------------


def _build_inputs(num_group, seq, n, k, device):
    """Build shared inputs for both TMA and cp.async paths (mxfp4 weight)."""
    seqlens = torch.full((num_group,), seq, dtype=torch.int32, device=device)
    cu_seqlens = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.cumsum(seqlens, 0).to(torch.int32),
        ]
    )
    m_total = int(cu_seqlens[-1].item())

    x_fp8 = torch.randn((m_total, k), dtype=torch.float32, device=device).to(torch.float8_e4m3fn)

    # W: fp4 (e2m1) packed two-per-byte -> [num_group, n, k/2] uint8.
    lo = torch.randint(0, 16, (num_group, n, k // 2), dtype=torch.uint8, device=device)
    hi = torch.randint(0, 16, (num_group, n, k // 2), dtype=torch.uint8, device=device)
    w_fp4_packed = ((hi << 4) | lo).contiguous()

    sfx = torch.full((m_total, k // SF_VEC), 127, dtype=torch.uint8, device=device)
    sfw = torch.full((num_group, n, k // SF_VEC), 127, dtype=torch.uint8, device=device)
    sfx_packed, sfw_packed = hpc.prepack_mxfp8_scale(
        sfx, sfw, cu_seqlens, num_seq_per_group_avg=seq
    )

    return {
        "x_fp8": x_fp8,
        "w_fp4_packed": w_fp4_packed,
        "sfx_packed": sfx_packed,
        "sfw_packed": sfw_packed,
        "seqlens": seqlens,
        "cu_seqlens": cu_seqlens,
        "m_total": m_total,
    }


def _build_tma_runner(inputs, seq):
    """TMA path: hpc.group_gemm_mxfp8."""

    def run():
        return hpc.group_gemm_mxfp8(
            inputs["x_fp8"],
            inputs["w_fp4_packed"],
            inputs["sfx_packed"],
            inputs["sfw_packed"],
            inputs["seqlens"],
            inputs["cu_seqlens"],
            num_seq_per_group_avg=seq,
        )

    return run


def _build_cp_async_map_runner(inputs, seq, device):
    """cp.async path with random row_map (simulates fuse_moe row indirection)."""
    m_total = inputs["m_total"]
    perm = torch.randperm(m_total, device=device, dtype=torch.int64)
    x_permuted = inputs["x_fp8"][perm].contiguous()
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(m_total, device=device, dtype=torch.int64)
    row_map = inv_perm.to(torch.int32).contiguous()

    def run():
        return hpc.group_gemm_cp_async_mxfp8(
            x_permuted,
            inputs["w_fp4_packed"],
            inputs["sfx_packed"],
            inputs["sfw_packed"],
            inputs["seqlens"],
            inputs["cu_seqlens"],
            num_seq_per_group_avg=seq,
            x_row_map=row_map,
        )

    return run


# ---------------------------- timing ----------------------------


def _bench(label: str, fn) -> float:
    """Returns mean per-call latency in microseconds (CUDA graph when possible)."""
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()

    # Try CUDA graph capture; fall back to eager if capture fails.
    graph = None
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        torch.cuda.synchronize()
    except Exception:
        graph = None
        torch.cuda.synchronize()

    nvtx.range_push(label)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITER):
        if graph is not None:
            graph.replay()
        else:
            fn()
    end.record()
    torch.cuda.synchronize()
    nvtx.range_pop()
    elapsed_ms = start.elapsed_time(end)

    if graph is not None:
        del graph
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return elapsed_ms * 1e3 / ITER  # us


def _try_bench(label, fn):
    """Bench with error tolerance."""
    try:
        return _bench(label, fn)
    except Exception as e:
        return f"N/A ({type(e).__name__})"


# ---------------------------- main ----------------------------


def _fmt(us):
    if isinstance(us, str):
        return f"{us:>16}"
    return f"{us:>10.2f} us"


def _fmt_speedup(base, us):
    if isinstance(us, str) or isinstance(base, str):
        return "      ---"
    ratio = us / base
    return f"{ratio:6.2f}x"


def main():
    device = torch.device("cuda:0")
    torch.manual_seed(2026)

    print(f"WARMUP={WARMUP}  ITER={ITER}")
    print("Weight format: mxfp4 (e2m1 packed uint8)")
    print()

    for label, n, k, num_group in SHAPES:
        print(f"=== {label}  (n={n}, k={k}, num_group={num_group}) ===")
        header = (
            f"{'num_seq':>7} {'seq/grp':>7} {'TMA':>13} "
            f"{'cp.async-map':>13} {'cpa_map/TMA':>11}"
        )
        print(header)
        print("-" * len(header))

        for num_seq in M_CASES:
            # seq = tokens per group (expert) in the group_gemm
            seq = num_seq * NUM_TOPK // NUM_EXPERT_TOTAL

            inputs = _build_inputs(num_group, seq, n, k, device)

            # TMA
            t_tma = _try_bench(f"{label}_seq{seq}_TMA", _build_tma_runner(inputs, seq))

            # cp.async + row_map
            t_cpm = _try_bench(
                f"{label}_seq{seq}_cpa_map", _build_cp_async_map_runner(inputs, seq, device)
            )

            print(
                f"{num_seq:>7} {seq:>7} {_fmt(t_tma)} {_fmt(t_cpm)} "
                f"{_fmt_speedup(t_tma, t_cpm):>11}"
            )

        print()


if __name__ == "__main__":
    main()
