# Copyright (C) 2025 Tencent.

"""Attention Decode FP8 benchmark for dynamic scheduling.


Default workloads:
    uniform_512, uniform_4096, skewed_mix, skewed_extreme,
    one_64k_7x4k, one_64k_15x4k, one_64k_31x4k, one_128k_31x4k,
    two_32k_30x4k

Recommended command:
    python3 benchmark/attention_decode/bench_attention_decode_fp8.py --csv attention_decode_fp8.csv

The benchmark compares static split-k with the dynamic task map used by the
SM90 FP8 decode kernels. Latency is reported in microseconds per operator call.
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
from statistics import mean, median
from typing import Iterable

import torch


sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../../build/lib.*/"))[0]))

import hpc  # noqa: E402


BLOCK_SIZE = 64
DEFAULT_NUM_SEQ_Q = 1
DEFAULT_HEAD_DIM = 128
DEFAULT_KV_HEADS = 1
DEFAULT_Q_HEADS = 8

QUANT_TYPES = {
    "qkpertoken_perhead_vperhead": hpc.QuantType.QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD,
    "qpertoken_perhead_kvpertensor": hpc.QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR,
}


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


def quant_paged_cache_pertoken(cache: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    num_blocks = cache.shape[0]
    head_dim = cache.shape[-1]
    num_head_kv = cache.shape[-2]
    scale = cache[:, :block_size, :, :].float().abs().max(-1)[0].clamp_min(1e-6) / 448
    cache_fp8 = torch.empty_like(cache, dtype=torch.float8_e4m3fn)
    cache_fp8[:, :block_size, :, :] = (cache[:, :block_size, :, :] / scale[:, :, :, None]).to(
        torch.float8_e4m3fn
    )
    scale = (
        scale.permute(0, 2, 1)
        .contiguous()
        .view(torch.float8_e4m3fn)
        .reshape(num_blocks, num_head_kv, -1, head_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    cache_fp8[:, block_size:, :, :] = scale
    return cache_fp8, cache_fp8[:, block_size:, :, :]


def quant_paged_cache_perhead(cache: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    num_head_kv = cache.shape[-2]
    scale = (
        cache[:, :block_size, :, :]
        .float()
        .abs()
        .permute(2, 0, 1, 3)
        .reshape(num_head_kv, -1)
        .max(-1)[0]
        .clamp_min(1e-6)
        / 448
    )
    cache_fp8 = (cache.float() / scale[None, None, :, None]).to(torch.float8_e4m3fn)
    return cache_fp8, scale


def build_block_ids(kv_lens: torch.Tensor, block_size: int, max_num_blocks: int) -> torch.Tensor:
    nblocks = (kv_lens + block_size - 1) // block_size
    packed_block_ids = torch.randperm(max_num_blocks, device="cuda")[: int(nblocks.sum())].to(torch.int32)
    block_ids = torch.empty((len(kv_lens), int(nblocks.max())), dtype=torch.int32, device="cuda")
    offset = 0
    for i, blocks in enumerate(nblocks.tolist()):
        block_ids[i, :blocks] = packed_block_ids[offset : offset + blocks]
        offset += blocks
    return block_ids


def make_task_map(kv_lens: torch.Tensor, num_head_kv: int, num_seq_q: int, min_process_len: int) -> torch.Tensor:
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
    quant_name: str,
    num_seq_q: int,
    num_head_kv: int,
    num_head_q: int,
    head_dim: int,
) -> Inputs:
    torch.manual_seed(41)
    torch.cuda.manual_seed(41)

    kv_lens = torch.tensor(list(kv_lengths), dtype=torch.int32, device="cuda")
    num_batch = len(kv_lens)
    max_seq_kv = int(kv_lens.max().item())
    nblocks = (kv_lens + BLOCK_SIZE - 1) // BLOCK_SIZE
    max_num_blocks = int(nblocks.sum().item() * 1.2) + num_batch + 8

    q = torch.randn(
        (num_batch * num_seq_q, num_head_q, head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    ) / math.sqrt(head_dim)
    q_scale = q.float().abs().max(-1)[0].clamp_min(1e-6)
    q = (q / q_scale[:, :, None]).to(torch.float8_e4m3fn)
    block_ids = build_block_ids(kv_lens, BLOCK_SIZE, max_num_blocks)

    if quant_name == "qpertoken_perhead_kvpertensor":
        k_cache = torch.randn(
            max_num_blocks, BLOCK_SIZE, num_head_kv, head_dim, dtype=torch.bfloat16, device="cuda"
        )
        v_cache = torch.randn_like(k_cache)
        k_cache = (k_cache / math.sqrt(head_dim)).to(torch.float8_e4m3fn)
        v_cache = v_cache.to(torch.float8_e4m3fn)
        k_scale = torch.rand((1,), dtype=torch.float32, device="cuda").clamp_min(1e-6)
        v_scale = torch.rand((1,), dtype=torch.float32, device="cuda").clamp_min(1e-6)
    else:
        scale_rows = BLOCK_SIZE * 4 // head_dim
        raw_cache = torch.randn(
            max_num_blocks,
            2,
            BLOCK_SIZE + scale_rows,
            num_head_kv,
            head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        k_cache, k_scale = quant_paged_cache_pertoken(raw_cache[:, 0], BLOCK_SIZE)
        v_cache, v_scale = quant_paged_cache_perhead(raw_cache[:, 1], BLOCK_SIZE)
        k_cache = k_cache[:, :BLOCK_SIZE]
        v_cache = v_cache[:, :BLOCK_SIZE]

    output = torch.empty_like(q, dtype=torch.bfloat16)
    return Inputs(q, k_cache, v_cache, block_ids, kv_lens, q_scale, k_scale, v_scale, output, num_batch, max_seq_kv)


def run_kernel(inputs: Inputs, quant_name: str, task_map: torch.Tensor | None = None) -> torch.Tensor:
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
        quant_type=QUANT_TYPES[quant_name],
        splitk=True,
        task_map=task_map,
        output=inputs.output,
    )


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
    quant_name = args.quant_types[0]
    inputs = make_inputs(
        CASES[case_name],
        quant_name,
        DEFAULT_NUM_SEQ_Q,
        args.num_head_kv,
        args.num_head_q,
        args.head_dim,
    )
    task_map = None
    if args.nsys_variant == "dynamic":
        task_map = make_task_map(inputs.kv_lens, args.num_head_kv, DEFAULT_NUM_SEQ_Q, args.min_process_len)

    def call_fn():
        if args.nsys_variant == "dynamic" and args.include_taskmap:
            hpc.assign_attention_decode_task(
                inputs.kv_lens,
                task_map,
                args.num_head_kv,
                DEFAULT_NUM_SEQ_Q,
                True,
                min_process_len=args.min_process_len,
            )
        run_kernel(inputs, quant_name, task_map)

    run_method_c(call_fn, warmup=args.warmup, n_timed=args.iters + 2)


def run_nsys_profile(args: argparse.Namespace, case_name: str, quant_name: str, variant: str, out_dir: Path) -> tuple[float | None, int, str | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_prefix = out_dir / f"{case_name}_{quant_name}_{variant}"
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
        variant,
        "--cases",
        case_name,
        "--quant-types",
        quant_name,
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


def run_nsys_driver(args: argparse.Namespace) -> list[dict]:
    tag = args.tag or f"attention_decode_{int(time.time())}"
    out_dir = Path(args.output_dir) / tag if args.output_dir else Path(__file__).resolve().parent / "log" / tag
    rows = []
    for case_name in args.cases:
        for quant_name in args.quant_types:
            static_us, static_n, static_err = run_nsys_profile(args, case_name, quant_name, "static", out_dir)
            dynamic_us, dynamic_n, dynamic_err = run_nsys_profile(args, case_name, quant_name, "dynamic", out_dir)
            row = {
                "case": case_name,
                "quant_type": quant_name,
                "batch": len(CASES[case_name]),
                "max_kv": max(CASES[case_name]),
                "static_us": static_us,
                "dynamic_us": dynamic_us,
                "speedup": static_us / dynamic_us if static_us and dynamic_us else None,
                "timing": "nsys_graph_nvtx_median",
                "static_samples": static_n,
                "dynamic_samples": dynamic_n,
                "error": static_err or dynamic_err,
            }
            rows.append(row)
            print_table(rows)
            if row["error"]:
                print(f"[warn] {case_name}/{quant_name}: {row['error']}", file=sys.stderr)
    print(f"nsys reports: {out_dir}")
    return rows


def print_table(rows: list[dict]) -> None:
    def fmt_us(value) -> str:
        return f"{value:10.2f}" if isinstance(value, (int, float)) else f"{'ERR':>10}"

    def fmt_speedup(value) -> str:
        return f"{value:7.2f}x" if isinstance(value, (int, float)) else f"{'ERR':>8}"

    print("")
    print("=" * 112)
    print("Attention Decode FP8 dynamic scheduling | latency in us (lower is better)")
    print("-" * 112)
    print(f"{'case':>18} | {'quant_type':>34} | {'batch':>5} | {'max_kv':>7} | {'static':>10} | {'dynamic':>10} | {'speedup':>8}")
    print("-" * 112)
    for row in rows:
        print(
            f"{row['case']:>18} | {row['quant_type']:>34} | {row['batch']:5d} | "
            f"{row['max_kv']:7d} | {fmt_us(row['static_us'])} | {fmt_us(row['dynamic_us'])} | "
            f"{fmt_speedup(row['speedup'])}"
        )
    print("=" * 112)


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
    parser = argparse.ArgumentParser(description="Benchmark Attention Decode FP8 static vs dynamic scheduling.")
    parser.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    parser.add_argument("--quant-types", nargs="+", default=list(QUANT_TYPES), choices=list(QUANT_TYPES))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--min-process-len", type=int, default=512)
    parser.add_argument("--num-head-kv", type=int, default=DEFAULT_KV_HEADS)
    parser.add_argument("--num-head-q", type=int, default=DEFAULT_Q_HEADS)
    parser.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    parser.add_argument("--timing", choices=["event", "nsys"], default="event", help="event: CUDA event around graph replay; nsys: FusedMoE-style nsys/NVTX graph replay median.")
    parser.add_argument("--no-graph", dest="graph", action="store_false", help="Use eager event timing instead of CUDA Graph replay.")
    parser.add_argument("--include-taskmap", action="store_true", help="Include assign_attention_decode_task in the dynamic timed region.")
    parser.add_argument("--output-dir", default="", help="Output directory for nsys reports.")
    parser.add_argument("--tag", default="", help="Subdirectory name for nsys reports.")
    parser.add_argument("--nsys-timeout", type=int, default=300)
    parser.add_argument("--csv", default="", help="Optional CSV output path.")
    parser.add_argument("--jsonl", default="", help="Optional JSONL output path.")
    parser.add_argument("--check", dest="check", action="store_true", help="Compare dynamic output with static output.")
    parser.add_argument("--no-check", dest="check", action="store_false")
    parser.add_argument("--nsys-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--nsys-variant", choices=["static", "dynamic"], default="static", help=argparse.SUPPRESS)
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
        for quant_name in args.quant_types:
            inputs = make_inputs(
                CASES[case_name],
                quant_name,
                DEFAULT_NUM_SEQ_Q,
                args.num_head_kv,
                args.num_head_q,
                args.head_dim,
            )
            task_map = make_task_map(inputs.kv_lens, args.num_head_kv, DEFAULT_NUM_SEQ_Q, args.min_process_len)
            bench_fn = bench_us if args.graph else lambda fn, warmup, iters, _use_graph: bench_us_mean(fn, warmup, iters)
            static_us = bench_fn(lambda: run_kernel(inputs, quant_name), args.warmup, args.iters, args.graph)
            dynamic_us = bench_fn(lambda: run_kernel(inputs, quant_name, task_map), args.warmup, args.iters, args.graph)
            if args.check:
                static_out = run_kernel(inputs, quant_name)
                dynamic_out = run_kernel(inputs, quant_name, task_map)
                if not torch.allclose(static_out, dynamic_out, atol=0.2, rtol=0.2):
                    raise AssertionError(f"{case_name}/{quant_name}: dynamic output differs from static output")
            rows.append(
                {
                    "case": case_name,
                    "quant_type": quant_name,
                    "batch": inputs.num_batch,
                    "max_kv": inputs.max_seq_kv,
                    "static_us": static_us,
                    "dynamic_us": dynamic_us,
                    "speedup": static_us / dynamic_us if dynamic_us else None,
                }
            )
    print_table(rows)
    write_csv(args.csv, rows)
    write_jsonl(args.jsonl, rows)


if __name__ == "__main__":
    main()
