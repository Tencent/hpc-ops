# Copyright (C) 2025 Tencent.

"""Attention Decode FP8 benchmark for dynamic scheduling.


Default workloads:
    uniform_512, uniform_4096, skewed_mix, skewed_extreme,
    one_64k_7x4k, one_64k_15x4k, one_64k_31x4k, one_128k_31x4k,
    two_32k_30x4k

Recommended command:
    python3 benchmark/attention_decode/bench_attention_decode_fp8.py --csv attention_decode_fp8.csv

Quantization:
    - HPC static/dynamic: qpertoken_perhead + kvpertensor
    - FlashAttention-3 / FlashInfer: qkvpertensor (scalar per-tensor scales)

The benchmark compares HPC static split-k, HPC dynamic task map, FlashAttention-3,
and FlashInfer. Latency is reported in microseconds per operator call.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Callable, Iterable

import torch


sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../../build/lib.*/"))[0]))

import hpc  # noqa: E402


BLOCK_SIZE = 64
FA_BLOCK_SIZE = 256
DEFAULT_NUM_SEQ_Q = 1
DEFAULT_HEAD_DIM = 128
DEFAULT_KV_HEADS = 1
DEFAULT_Q_HEADS = 8

# HPC path: Q per-token-per-head, K/V per-tensor.
HPC_QUANT_TYPE = hpc.QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR

METHODS = ("static", "dynamic", "flashattn", "flashinfer")


CASES = {
    "uniform_512": [512] * 64,
    "uniform_4096": [4096] * 64,
    "skewed_mix": [128] * 32 + [4096] * 32,
    "skewed_extreme": [64] * 15 + [16 * 1024],
    "one_64k_7x4k": [64 * 1024] + [4096] * 7,
    "one_64k_15x4k": [64 * 1024] + [4096] * 15,
    "one_64k_31x4k": [64 * 1024] + [4096] * 31,
    "one_128k_31x4k": [128 * 1024] + [4096] * 31,
    "two_32k_30x4k": [32 * 1024] * 2 + [4096] * 30,
}


@dataclass
class Inputs:
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    block_ids: torch.Tensor
    kv_lens: torch.Tensor
    q_scale: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    output: torch.Tensor
    num_batch: int
    max_seq_kv: int


def _import_fa3_kvcache():
    """Import FA3 flash_attn_with_kvcache (supports FP8 + q/k/v_descale)."""
    try:
        from flash_attn_interface import flash_attn_with_kvcache

        return flash_attn_with_kvcache
    except Exception:
        pass
    try:
        from flash_attn_3.flash_attn_interface import flash_attn_with_kvcache

        return flash_attn_with_kvcache
    except Exception as e:
        raise ImportError(
            "FlashAttention-3 with FP8 is required for --methods flashattn. "
        ) from e


def build_block_ids(kv_lens: torch.Tensor, block_size: int, max_num_blocks: int) -> torch.Tensor:
    nblocks = (kv_lens + block_size - 1) // block_size
    packed_block_ids = torch.randperm(max_num_blocks, device="cuda")[: int(nblocks.sum())].to(
        torch.int32
    )
    block_ids = torch.empty((len(kv_lens), int(nblocks.max())), dtype=torch.int32, device="cuda")
    offset = 0
    for i, blocks in enumerate(nblocks.tolist()):
        block_ids[i, :blocks] = packed_block_ids[offset : offset + blocks]
        offset += blocks
    return block_ids


def make_task_map(
    kv_lens: torch.Tensor, num_head_kv: int, num_seq_q: int, min_process_len: int
) -> torch.Tensor:
    num_batch = len(kv_lens)
    max_seq_kv = int(kv_lens.max().item())
    task_map = hpc.get_attention_decode_task_workspace(
        num_batch, max_seq_kv, num_head_kv, min_process_len=min_process_len
    )
    hpc.assign_attention_decode_task(
        kv_lens,
        task_map,
        num_head_kv,
        num_seq_q,
        True,
        min_process_len=min_process_len,
    )
    return task_map


def make_inputs(
    kv_lengths: Iterable[int],
    num_seq_q: int,
    num_head_kv: int,
    num_head_q: int,
    head_dim: int,
) -> Inputs:
    """Build HPC inputs: Q per-token-per-head FP8, K/V per-tensor FP8."""
    torch.manual_seed(41)
    torch.cuda.manual_seed(41)

    kv_lens = torch.tensor(list(kv_lengths), dtype=torch.int32, device="cuda")
    num_batch = len(kv_lens)
    max_seq_kv = int(kv_lens.max().item())
    nblocks = (kv_lens + BLOCK_SIZE - 1) // BLOCK_SIZE
    max_num_blocks = int(nblocks.sum().item() * 1.2) + num_batch + 8

    q_bf16 = torch.randn(
        (num_batch * num_seq_q, num_head_q, head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    ) / math.sqrt(head_dim)
    q_scale = q_bf16.float().abs().max(-1)[0].clamp_min(1e-6)
    q = (q_bf16 / q_scale[:, :, None]).to(torch.float8_e4m3fn)

    k_cache = (
        torch.randn(
            max_num_blocks, BLOCK_SIZE, num_head_kv, head_dim, dtype=torch.bfloat16, device="cuda"
        )
        / math.sqrt(head_dim)
    ).to(torch.float8_e4m3fn)
    v_cache = torch.randn(
        max_num_blocks, BLOCK_SIZE, num_head_kv, head_dim, dtype=torch.bfloat16, device="cuda"
    ).to(torch.float8_e4m3fn)
    k_scale = torch.rand((1,), dtype=torch.float32, device="cuda").clamp_min(1e-6)
    v_scale = torch.rand((1,), dtype=torch.float32, device="cuda").clamp_min(1e-6)

    block_ids = build_block_ids(kv_lens, BLOCK_SIZE, max_num_blocks)
    output = torch.empty_like(q, dtype=torch.bfloat16)
    return Inputs(
        q,
        k_cache,
        v_cache,
        block_ids,
        kv_lens,
        q_scale,
        k_scale,
        v_scale,
        output,
        num_batch,
        max_seq_kv,
    )


def run_kernel(inputs: Inputs, task_map: torch.Tensor | None = None) -> torch.Tensor:
    return hpc.attention_decode_fp8(
        inputs.q,
        inputs.k_cache,
        inputs.v_cache,
        inputs.block_ids,
        inputs.kv_lens,
        inputs.q_scale,
        inputs.k_scale,
        inputs.v_scale,
        mtp=DEFAULT_NUM_SEQ_Q - 1,
        new_kv_included=True,
        quant_type=HPC_QUANT_TYPE,
        splitk=True,
        task_map=task_map,
        output=inputs.output,
    )


def make_flashattn_fn(inputs: Inputs, num_head_kv: int, head_dim: int) -> Callable[[], None]:
    """FA3 FP8 decode with qkvpertensor scales."""
    flash_attn_with_kvcache = _import_fa3_kvcache()

    nblocks_fa = (inputs.kv_lens + FA_BLOCK_SIZE - 1) // FA_BLOCK_SIZE
    max_num_blocks_fa = int(nblocks_fa.sum().item() * 1.2) + inputs.num_batch + 4
    max_pages = int(nblocks_fa.max().item())

    # qkvpertensor: one scale for Q / K / V each.
    q_scale = float(torch.rand((), device="cuda").clamp_min(1e-6).item())
    k_scale = float(torch.rand((), device="cuda").clamp_min(1e-6).item())
    v_scale = float(torch.rand((), device="cuda").clamp_min(1e-6).item())

    q_fa = (torch.randn_like(inputs.q, dtype=torch.bfloat16) / q_scale).to(torch.float8_e4m3fn)
    q_fa = q_fa.unsqueeze(1)  # (B*Sq, 1, Hq, D) for decode
    k_cache_fa = (
        torch.randn(
            max_num_blocks_fa,
            FA_BLOCK_SIZE,
            num_head_kv,
            head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        / k_scale
    ).to(torch.float8_e4m3fn)
    v_cache_fa = (
        torch.randn(
            max_num_blocks_fa,
            FA_BLOCK_SIZE,
            num_head_kv,
            head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        / v_scale
    ).to(torch.float8_e4m3fn)

    page_table = torch.zeros(inputs.num_batch, max_pages, dtype=torch.int32, device="cuda")
    offset = 0
    for i in range(inputs.num_batch):
        nb = int(nblocks_fa[i])
        page_table[i, :nb] = torch.arange(offset, offset + nb, dtype=torch.int32, device="cuda")
        offset += nb
    cache_seqlens = inputs.kv_lens.to(torch.int32)

    # FA3 descales are (batch, nheads_kv); broadcast tensor scales.
    q_descale = torch.full(
        (inputs.num_batch, num_head_kv), q_scale, dtype=torch.float32, device="cuda"
    )
    k_descale = torch.full(
        (inputs.num_batch, num_head_kv), k_scale, dtype=torch.float32, device="cuda"
    )
    v_descale = torch.full(
        (inputs.num_batch, num_head_kv), v_scale, dtype=torch.float32, device="cuda"
    )

    def call_fn():
        flash_attn_with_kvcache(
            q=q_fa,
            k_cache=k_cache_fa,
            v_cache=v_cache_fa,
            cache_seqlens=cache_seqlens,
            page_table=page_table,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            causal=True,
        )

    return call_fn


def make_flashinfer_fn(
    inputs: Inputs, num_head_kv: int, num_head_q: int, head_dim: int
) -> Callable[[], None]:
    """FlashInfer FP8 decode with qkvpertensor (scalar) scales."""
    import flashinfer

    nblocks_fi = (inputs.kv_lens + BLOCK_SIZE - 1) // BLOCK_SIZE
    kv_indptr = torch.zeros(inputs.num_batch + 1, dtype=torch.int32, device="cuda")
    torch.cumsum(nblocks_fi, dim=0, out=kv_indptr[1:])
    kv_indices = torch.zeros(int(nblocks_fi.sum()), dtype=torch.int32, device="cuda")
    offset = 0
    for i in range(inputs.num_batch):
        nb = int(nblocks_fi[i])
        kv_indices[offset : offset + nb] = torch.arange(
            offset, offset + nb, dtype=torch.int32, device="cuda"
        )
        offset += nb
    kv_last_page_len = ((inputs.kv_lens - 1) % BLOCK_SIZE + 1).to(torch.int32)

    q_scale = float(torch.rand((), device="cuda").clamp_min(1e-6).item())
    k_scale = float(torch.rand((), device="cuda").clamp_min(1e-6).item())
    v_scale = float(torch.rand((), device="cuda").clamp_min(1e-6).item())

    q_fi = (torch.randn_like(inputs.q, dtype=torch.bfloat16) / q_scale).to(torch.float8_e4m3fn)
    max_num_blocks_fi = int(nblocks_fi.sum().item()) + inputs.num_batch + 4
    k_cache_fi = (
        torch.randn(
            max_num_blocks_fi,
            BLOCK_SIZE,
            num_head_kv,
            head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        / k_scale
    ).to(torch.float8_e4m3fn)
    v_cache_fi = (
        torch.randn(
            max_num_blocks_fi,
            BLOCK_SIZE,
            num_head_kv,
            head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        / v_scale
    ).to(torch.float8_e4m3fn)

    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda")
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer,
        "NHD",
        use_cuda_graph=True,
        paged_kv_indptr_buffer=kv_indptr,
        paged_kv_indices_buffer=kv_indices,
        paged_kv_last_page_len_buffer=kv_last_page_len,
    )
    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_head_q,
        num_head_kv,
        head_dim,
        BLOCK_SIZE,
        q_data_type=torch.float8_e4m3fn,
        kv_data_type=torch.float8_e4m3fn,
    )

    def call_fn():
        wrapper.run(
            q_fi,
            (k_cache_fi, v_cache_fi),
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )

    return call_fn


def make_call_fn(
    method: str,
    inputs: Inputs,
    args: argparse.Namespace,
    task_map: torch.Tensor | None = None,
) -> Callable[[], None]:
    if method == "static":
        return lambda: run_kernel(inputs)
    if method == "dynamic":
        if task_map is None:
            task_map = make_task_map(
                inputs.kv_lens, args.num_head_kv, DEFAULT_NUM_SEQ_Q, args.min_process_len
            )

        def call_fn():
            if args.include_taskmap:
                hpc.assign_attention_decode_task(
                    inputs.kv_lens,
                    task_map,
                    args.num_head_kv,
                    DEFAULT_NUM_SEQ_Q,
                    True,
                    min_process_len=args.min_process_len,
                )
            run_kernel(inputs, task_map)

        return call_fn
    if method == "flashattn":
        return make_flashattn_fn(inputs, args.num_head_kv, args.head_dim)
    if method == "flashinfer":
        return make_flashinfer_fn(inputs, args.num_head_kv, args.num_head_q, args.head_dim)
    raise ValueError(f"Unknown method: {method}")


def bench_us(fn, warmup: int, iters: int, use_graph: bool) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    if use_graph:
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                fn()
        torch.cuda.current_stream().wait_stream(capture_stream)
        for _ in range(warmup):
            graph.replay()
        torch.cuda.synchronize()
        fn = graph.replay

    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(iters)
    ]
    for start, end in events:
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    times = sorted(start.elapsed_time(end) * 1000.0 for start, end in events)
    return times[len(times) // 2]


def bench_us_mean(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def run_method_c(call_fn, *, warmup: int, n_timed: int) -> None:
    """FusedMoE-style timing worker: warmup, graph capture, replay under NVTX."""
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
        graph.replay()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()


def extract_nvtx_us(report_prefix: Path) -> list[float]:
    cmd = [
        "nsys",
        "stats",
        "--report",
        "nvtx_gpu_proj_trace",
        "--force-export=true",
        "-q",
        "-f",
        "json",
        str(report_prefix) + ".nsys-rep",
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        raw = json.loads(out.decode())
        data = raw[0]["data"] if isinstance(raw, list) and raw and "data" in raw[0] else raw
        samples = []
        for entry in data:
            name = entry.get("Name", "").strip().strip('"')
            if name not in ("step", ":step"):
                continue
            samples.append(float(entry["Projected Duration (ns)"]) / 1000.0)
        return samples[2:]
    except Exception:
        return []


def run_nsys_worker(args: argparse.Namespace) -> None:
    case_name = args.cases[0]
    method = args.nsys_variant
    inputs = make_inputs(
        CASES[case_name],
        DEFAULT_NUM_SEQ_Q,
        args.num_head_kv,
        args.num_head_q,
        args.head_dim,
    )
    call_fn = make_call_fn(method, inputs, args)
    run_method_c(call_fn, warmup=args.warmup, n_timed=args.iters + 2)


def run_nsys_profile(
    args: argparse.Namespace, case_name: str, method: str, out_dir: Path
) -> tuple[float | None, int, str | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_prefix = out_dir / f"{case_name}_{method}"
    report_file = str(report_prefix) + ".nsys-rep"
    if os.path.exists(report_file):
        os.remove(report_file)

    cmd = [
        "nsys",
        "profile",
        "-f",
        "true",
        "-o",
        str(report_prefix),
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop-shutdown",
        "--cuda-graph-trace=node",
        "-t",
        "cuda,nvtx",
        sys.executable,
        str(Path(__file__).resolve()),
        "--nsys-worker",
        "--nsys-variant",
        method,
        "--cases",
        case_name,
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--min-process-len",
        str(args.min_process_len),
        "--num-head-kv",
        str(args.num_head_kv),
        "--num-head-q",
        str(args.num_head_q),
        "--head-dim",
        str(args.head_dim),
    ]
    if args.include_taskmap:
        cmd.append("--include-taskmap")

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=args.nsys_timeout,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired:
        return None, 0, "nsys profile timeout"

    if not os.path.exists(report_file) and proc.returncode != 0:
        stderr = proc.stderr.decode(errors="replace").strip().splitlines()
        return None, 0, stderr[-1] if stderr else "nsys profile failed"

    samples = extract_nvtx_us(report_prefix)
    if not samples:
        return None, 0, "no NVTX step samples"
    return float(median(samples)), len(samples), None


def speedup_vs(baseline_us, dynamic_us) -> float | None:
    """dynamic speedup over a baseline: baseline_us / dynamic_us (>1 means dynamic is faster)."""
    if baseline_us and dynamic_us:
        return baseline_us / dynamic_us
    return None


def build_result_row(
    case_name: str,
    batch: int,
    max_kv: int,
    results: dict,
    *,
    timing: str | None = None,
    samples: dict | None = None,
    error: str | None = None,
) -> dict:
    static_us = results.get("static")
    dynamic_us = results.get("dynamic")
    flashattn_us = results.get("flashattn")
    flashinfer_us = results.get("flashinfer")
    row = {
        "case": case_name,
        "quant_type": "qpertoken_perhead_kvpertensor",
        "batch": batch,
        "max_kv": max_kv,
        "static_us": static_us,
        "dynamic_us": dynamic_us,
        "flashattn_us": flashattn_us,
        "flashinfer_us": flashinfer_us,
        "speedup_vs_static": speedup_vs(static_us, dynamic_us),
        "speedup_vs_flashattn": speedup_vs(flashattn_us, dynamic_us),
        "speedup_vs_flashinfer": speedup_vs(flashinfer_us, dynamic_us),
    }
    if timing is not None:
        row["timing"] = timing
    if samples is not None:
        row.update(samples)
    row["error"] = error
    return row


def run_nsys_driver(args: argparse.Namespace) -> list[dict]:
    tag = args.tag or f"attention_decode_fp8_{int(time.time())}"
    out_dir = (
        Path(args.output_dir) / tag
        if args.output_dir
        else Path(__file__).resolve().parent / "log" / tag
    )
    rows = []
    for case_name in args.cases:
        results = {}
        errors = []
        for method in args.methods:
            us, n, err = run_nsys_profile(args, case_name, method, out_dir)
            results[method] = us
            results[f"{method}_samples"] = n
            if err:
                errors.append(f"{method}: {err}")
        row = build_result_row(
            case_name,
            len(CASES[case_name]),
            max(CASES[case_name]),
            results,
            timing="nsys_graph_nvtx_median",
            samples={
                "static_samples": results.get("static_samples", 0),
                "dynamic_samples": results.get("dynamic_samples", 0),
                "flashattn_samples": results.get("flashattn_samples", 0),
                "flashinfer_samples": results.get("flashinfer_samples", 0),
            },
            error="; ".join(errors) if errors else None,
        )
        rows.append(row)
        print_table(rows)
        if row["error"]:
            print(f"[warn] {case_name}: {row['error']}", file=sys.stderr)
    print(f"nsys reports: {out_dir}")
    return rows


def print_table(rows: list[dict]) -> None:
    def fmt_us(value) -> str:
        return f"{value:10.2f}" if isinstance(value, (int, float)) else f"{'ERR':>10}"

    def fmt_speedup(value) -> str:
        return f"{value:7.2f}x" if isinstance(value, (int, float)) else f"{'ERR':>8}"

    width = 125
    print("")
    print("=" * width)
    print(
        "Attention Decode FP8 | HPC=qpertoken+kvpertensor, FA3/FI=qkvpertensor | "
        "latency in us; x/* = baseline / dynamic"
    )
    print("-" * width)
    print(
        f"{'case':>18} | {'batch':>5} | {'max_kv':>7} | {'static':>10} | {'dynamic':>10} | "
        f"{'flashattn':>10} | {'flashinfer':>10} | {'x/sta':>8} | {'x/fa':>8} | {'x/fi':>8}"
    )
    print("-" * width)
    for row in rows:
        print(
            f"{row['case']:>18} | {row['batch']:5d} | {row['max_kv']:7d} | "
            f"{fmt_us(row.get('static_us'))} | {fmt_us(row.get('dynamic_us'))} | "
            f"{fmt_us(row.get('flashattn_us'))} | {fmt_us(row.get('flashinfer_us'))} | "
            f"{fmt_speedup(row.get('speedup_vs_static'))} | "
            f"{fmt_speedup(row.get('speedup_vs_flashattn'))} | "
            f"{fmt_speedup(row.get('speedup_vs_flashinfer'))}"
        )
    print("=" * width)


def write_csv(path: str, rows: list[dict]) -> None:
    if not path:
        return
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: str, rows: list[dict]) -> None:
    if not path:
        return
    with Path(path).open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Attention Decode FP8: HPC static/dynamic "
            "(qpertoken+kvpertensor) vs FA3/FlashInfer (qkvpertensor)."
        )
    )
    parser.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--min-process-len", type=int, default=64)
    parser.add_argument("--num-head-kv", type=int, default=DEFAULT_KV_HEADS)
    parser.add_argument("--num-head-q", type=int, default=DEFAULT_Q_HEADS)
    parser.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    parser.add_argument(
        "--timing",
        choices=["event", "nsys"],
        default="event",
        help="event: CUDA event around graph replay; nsys: FusedMoE-style nsys/NVTX graph replay median.",
    )
    parser.add_argument(
        "--no-graph",
        dest="graph",
        action="store_false",
        help="Use eager event timing instead of CUDA Graph replay.",
    )
    parser.add_argument(
        "--include-taskmap",
        action="store_true",
        help="Include assign_attention_decode_task in the dynamic timed region.",
    )
    parser.add_argument("--output-dir", default="", help="Output directory for nsys reports.")
    parser.add_argument("--tag", default="", help="Subdirectory name for nsys reports.")
    parser.add_argument("--nsys-timeout", type=int, default=300)
    parser.add_argument("--csv", default="", help="Optional CSV output path.")
    parser.add_argument("--jsonl", default="", help="Optional JSONL output path.")
    parser.add_argument(
        "--check",
        dest="check",
        action="store_true",
        help="Compare dynamic output with static output.",
    )
    parser.add_argument("--no-check", dest="check", action="store_false")
    parser.add_argument("--nsys-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--nsys-variant", choices=list(METHODS), default="static", help=argparse.SUPPRESS
    )
    parser.set_defaults(check=False)
    parser.set_defaults(graph=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.nsys_worker:
        run_nsys_worker(args)
        return
    if args.timing == "nsys":
        rows = run_nsys_driver(args)
        write_csv(args.csv, rows)
        write_jsonl(args.jsonl, rows)
        return

    rows = []
    for case_name in args.cases:
        inputs = make_inputs(
            CASES[case_name],
            DEFAULT_NUM_SEQ_Q,
            args.num_head_kv,
            args.num_head_q,
            args.head_dim,
        )
        task_map = None
        if "dynamic" in args.methods or args.check:
            task_map = make_task_map(
                inputs.kv_lens, args.num_head_kv, DEFAULT_NUM_SEQ_Q, args.min_process_len
            )

        results = {}
        for method in args.methods:
            call_fn = make_call_fn(method, inputs, args, task_map=task_map)
            try:
                if args.graph:
                    results[method] = bench_us(call_fn, args.warmup, args.iters, use_graph=True)
                else:
                    results[method] = bench_us_mean(call_fn, args.warmup, args.iters)
            except Exception as e:
                print(f"[warn] {case_name}/{method}: {e}", file=sys.stderr)
                results[method] = None

        if args.check:
            static_out = run_kernel(inputs).clone()
            dynamic_out = run_kernel(inputs, task_map)
            if not torch.allclose(static_out, dynamic_out, atol=0.2, rtol=0.2):
                raise AssertionError(f"{case_name}: dynamic output differs from static output")

        rows.append(
            build_result_row(
                case_name,
                inputs.num_batch,
                inputs.max_seq_kv,
                results,
            )
        )
    print_table(rows)
    write_csv(args.csv, rows)
    write_jsonl(args.jsonl, rows)


if __name__ == "__main__":
    main()
