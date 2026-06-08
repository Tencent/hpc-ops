#!/usr/bin/env python3
# Copyright (C) 2026 Tencent.

"""Sampler benchmark.

Scenarios:
    - temperature: temperature-only sampling fast path.
    - full: repetition penalty + temperature + top-k/top-p + sampling.

The nsys mode records eager sampler calls under NVTX ranges and reports median
latency from Nsight Systems' GPU projected duration. Timed loops do not
synchronize per iteration.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, median

import torch


sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../../build/lib.*/"))[0]))

from hpc.sampler import SoftmaxPolicy, fused_sampler  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent

VOCAB_SIZE = 120832
MAX_CONTEXT_LEN = 32768
REPETITION_PENALTY = 1.1
TEMPERATURE = 1.05
TOPK = 20
TOPP = 0.9
SAMPLER_SEED = 1

SCENES = ["temperature", "full"]
PROVIDERS = ["hpc", "torch", "flashinfer"]
DEFAULT_BATCHES = [1, 8, 16, 32, 64, 128, 256, 512]

SCENE_LABELS = {
    "temperature": "Temperature Sampling",
    "full": "Full Sampling",
}
DISPLAY = {
    "hpc": "HPC-Ops",
    "torch": "vLLM/PyTorch",
    "flashinfer": "FlashInfer",
}
PATH_BY_CELL = {
    ("temperature", "torch"): "stable-torch",
    ("temperature", "flashinfer"): "stable-fi",
    ("temperature", "hpc"): "stable-hpc",
    ("full", "torch"): "full-vllm",
    ("full", "flashinfer"): "full-fi",
    ("full", "hpc"): "full-hpc",
}


def build_inputs(batch: int, device: str = "cuda") -> dict:
    generator = torch.Generator(device="cpu").manual_seed(batch)
    logits = torch.randn(batch, VOCAB_SIZE, generator=generator).to(
        device=device, dtype=torch.bfloat16,
    ).contiguous()
    repetition = torch.full((batch,), REPETITION_PENALTY, dtype=torch.float32, device=device)

    prompt_cpu = torch.randint(
        0, VOCAB_SIZE, (batch, MAX_CONTEXT_LEN), dtype=torch.int64, generator=generator,
    )
    output_cpu = torch.randint(
        0, VOCAB_SIZE, (batch, MAX_CONTEXT_LEN), dtype=torch.int64, generator=generator,
    )
    try:
        prompt_cpu = prompt_cpu.pin_memory()
        output_cpu = output_cpu.pin_memory()
    except Exception:
        pass

    row_bytes = (VOCAB_SIZE + 7) // 8
    max_rows = max(batch, 64)
    bitmask = torch.zeros(max_rows, row_bytes, dtype=torch.uint8, device=device)
    token_ids = torch.cat([prompt_cpu.to(device), output_cpu.to(device)], dim=1).clamp(0, VOCAB_SIZE - 1)
    token_mask = torch.zeros(batch, VOCAB_SIZE, dtype=torch.bool, device=device)
    token_mask.scatter_(1, token_ids, torch.ones_like(token_ids, dtype=torch.bool))

    padded_vocab = row_bytes * 8
    if padded_vocab != VOCAB_SIZE:
        pad = torch.zeros(batch, padded_vocab - VOCAB_SIZE, dtype=torch.bool, device=device)
        token_mask = torch.cat([token_mask, pad], dim=1)
    bits = token_mask.view(batch, row_bytes, 8).to(torch.uint8)
    weights = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=device)
    bitmask[:batch] = (bits * weights).sum(dim=-1, dtype=torch.uint8)

    return {
        "logits": logits,
        "repetition": repetition,
        "prompt_cpu": prompt_cpu,
        "output_cpu": output_cpu,
        "bitmask": bitmask,
        "slot_id": torch.arange(batch, dtype=torch.int32, device=device),
    }


def setup_call(scene: str, provider: str, batch: int):
    inputs = build_inputs(batch)
    logits = inputs["logits"]

    if scene == "temperature" and provider == "torch":
        def run_temperature_torch():
            probs = logits.to(torch.float32).div_(TEMPERATURE).softmax(dim=-1)
            noise = torch.empty_like(probs).exponential_(1.0)
            return probs.div_(noise).argmax(dim=-1).view(-1, 1).to(torch.int32)
        return run_temperature_torch

    if scene == "temperature" and provider == "flashinfer":
        import flashinfer.sampling as sampling

        def run_temperature_flashinfer():
            probs = sampling.softmax(logits, temperature=TEMPERATURE)
            tokens = sampling.sampling_from_probs(probs, deterministic=True)
            return tokens.view(-1, 1).to(torch.int32)
        return run_temperature_flashinfer

    if scene == "temperature" and provider == "hpc":
        return lambda: fused_sampler(logits, temperature=TEMPERATURE, seed=SAMPLER_SEED)

    def penalty_temperature():
        from vllm._custom_ops import apply_repetition_penalties

        batch_size = logits.shape[0]
        work = logits.to(torch.float32)
        prompt_dev = inputs["prompt_cpu"].to("cuda", non_blocking=True)
        output_dev = inputs["output_cpu"].to("cuda", non_blocking=True)
        prompt_buf = torch.zeros(batch_size, VOCAB_SIZE + 1, dtype=torch.bool, device="cuda")
        output_buf = torch.zeros(batch_size, VOCAB_SIZE + 1, dtype=torch.bool, device="cuda")
        ones = torch.ones(1, dtype=torch.bool, device="cuda")
        prompt_buf.scatter_(1, prompt_dev.clamp(max=VOCAB_SIZE), ones.expand_as(prompt_dev))
        output_buf.scatter_(1, output_dev.clamp(max=VOCAB_SIZE), ones.expand_as(output_dev))
        apply_repetition_penalties(
            work,
            prompt_buf[:, :VOCAB_SIZE].contiguous(),
            output_buf[:, :VOCAB_SIZE].contiguous(),
            inputs["repetition"],
        )
        work.div_(TEMPERATURE)
        return work

    if scene == "full" and provider == "torch":
        def run_full_torch():
            work = penalty_temperature()
            sorted_logits, sorted_idx = work.sort(dim=-1, descending=False)
            cutoff = sorted_logits[..., VOCAB_SIZE - TOPK].unsqueeze(-1)
            sorted_logits = sorted_logits.masked_fill(sorted_logits < cutoff, float("-inf"))
            sorted_probs = sorted_logits.softmax(dim=-1)
            cumulative = sorted_probs.cumsum(dim=-1)
            drop = cumulative <= (1 - TOPP)
            drop[..., -1] = False
            sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
            filtered = torch.empty_like(work).scatter_(dim=-1, index=sorted_idx, src=sorted_logits)
            probs = filtered.softmax(dim=-1)
            noise = torch.empty_like(probs).exponential_(1.0)
            return probs.div_(noise).argmax(dim=-1).view(-1, 1).to(torch.int32)
        return run_full_torch

    if scene == "full" and provider == "flashinfer":
        import flashinfer.sampling as sampling

        def run_full_flashinfer():
            return sampling.top_k_top_p_sampling_from_logits(
                penalty_temperature(),
                TOPK,
                TOPP,
                filter_apply_order="top_k_first",
                deterministic=True,
            ).view(-1, 1).to(torch.int32)
        return run_full_flashinfer

    if scene == "full" and provider == "hpc":
        return lambda: fused_sampler(
            logits,
            penalty_mask=inputs["bitmask"],
            slot_id=inputs["slot_id"],
            repetition_penalty=inputs["repetition"],
            temperature=TEMPERATURE,
            softmax_policy=SoftmaxPolicy.AFTER_TOPK,
            topk=TOPK,
            topp=TOPP,
            max_topk=32,
            seed=SAMPLER_SEED,
        )

    raise ValueError(f"unsupported scene/provider: {scene}/{provider}")


def run_nsys_steps(call_fn, *, warmup: int, iters: int, range_name: str = "step") -> None:
    for _ in range(warmup):
        call_fn()
    torch.cuda.synchronize()

    for _ in range(iters):
        torch.cuda.nvtx.range_push(range_name)
        try:
            call_fn()
        finally:
            torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()


def bench_event(call_fn, *, warmup: int, iters: int) -> tuple[float, float, int]:
    for _ in range(warmup):
        call_fn()
    torch.cuda.synchronize()

    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for start, end in events:
        start.record()
        call_fn()
        end.record()
    torch.cuda.synchronize()
    values = [start.elapsed_time(end) * 1000.0 for start, end in events]
    return float(median(values)), float(mean(values)), len(values)


def run_worker(args) -> None:
    torch.cuda.set_device(0)
    cells = json.loads(args.worker_cells)
    torch.cuda.cudart().cudaProfilerStart()
    for cell in cells:
        scene = cell["scene"]
        provider = cell["provider"]
        batch = int(cell["batch"])
        range_name = PATH_BY_CELL[(scene, provider)] + f"|B{batch}"
        try:
            call_fn = setup_call(scene, provider, batch)
            run_nsys_steps(call_fn, warmup=args.warmup, iters=args.iters, range_name=range_name)
            print(f"[worker] done {scene} batch={batch} provider={provider}", file=sys.stderr, flush=True)
        except Exception as exc:
            print(
                f"[worker] failed {scene} batch={batch} provider={provider}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
        torch.cuda.empty_cache()
    torch.cuda.cudart().cudaProfilerStop()


def report_prefix(output_dir: Path, tag: str) -> Path:
    return output_dir / f"sampler_{tag}"


def run_nsys_profile(args, cells: list[dict], output_dir: Path, tag: str) -> tuple[Path, str | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = report_prefix(output_dir, tag)
    report_file = str(prefix) + ".nsys-rep"
    if os.path.exists(report_file):
        os.remove(report_file)

    cmd = [
        "nsys",
        "profile",
        "-f",
        "true",
        "-o",
        str(prefix),
        "--sample=none",
        "--cpuctxsw=none",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop-shutdown",
        "-t",
        "cuda,nvtx",
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--worker-cells",
        json.dumps(cells),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=args.nsys_timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return Path(report_file), "nsys profile timeout"

    if not os.path.exists(report_file):
        stderr = proc.stderr.decode(errors="replace").strip().splitlines()
        return Path(report_file), stderr[-1] if stderr else "nsys profile failed"
    return Path(report_file), None


def export_nvtx_csv(report_file: Path, output_dir: Path, tag: str) -> Path:
    output_prefix = output_dir / f"proj_{tag}"
    cmd = [
        "nsys",
        "stats",
        "-r",
        "nvtx_gpu_proj_trace",
        "--format",
        "csv",
        "--force-export=true",
        "-o",
        str(output_prefix),
        str(report_file),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return Path(str(output_prefix) + "_nvtx_gpu_proj_trace.csv")


def parse_nvtx_csv(path: Path) -> dict[str, list[float]]:
    samples: dict[str, list[float]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("Name", "").strip().strip('"').lstrip(":")
            if "|B" not in name:
                continue
            samples.setdefault(name, []).append(float(row["Projected Duration (ns)"]) / 1000.0)
    return samples


def build_cells(args) -> list[dict]:
    return [
        {"scene": scene, "batch": batch, "provider": provider}
        for scene in args.scenes
        for batch in args.batches
        for provider in args.providers
    ]


def make_row(scene: str, batch: int, provider: str, values: list[float], timing: str, error: str | None = None) -> dict:
    return {
        "scene": scene,
        "batch": batch,
        "provider": provider,
        "path": PATH_BY_CELL[(scene, provider)],
        "n": len(values),
        "median_us": float(median(values)) if values else None,
        "mean_us": float(mean(values)) if values else None,
        "timing": timing,
        "vocab_size": VOCAB_SIZE,
        "error": error,
    }


def rows_from_nsys(args, cells: list[dict], output_dir: Path, tag: str) -> tuple[list[dict], Path | None]:
    report_file, profile_error = run_nsys_profile(args, cells, output_dir, tag)
    if profile_error:
        return [
            make_row(c["scene"], c["batch"], c["provider"], [], "nsys_eager_nvtx_median", profile_error)
            for c in cells
        ], None

    nvtx_csv = export_nvtx_csv(report_file, output_dir, tag)
    samples = parse_nvtx_csv(nvtx_csv)
    rows = []
    for cell in cells:
        path = PATH_BY_CELL[(cell["scene"], cell["provider"])]
        key = f"{path}|B{cell['batch']}"
        values = samples.get(key, [])
        error = None if values else "no NVTX samples"
        rows.append(make_row(cell["scene"], cell["batch"], cell["provider"], values, "nsys_eager_nvtx_median", error))
    return rows, report_file


def rows_from_events(args, cells: list[dict]) -> list[dict]:
    rows = []
    for cell in cells:
        try:
            call_fn = setup_call(cell["scene"], cell["provider"], cell["batch"])
            median_us, mean_us, n = bench_event(call_fn, warmup=args.warmup, iters=args.iters)
            values = [median_us]
            row = make_row(cell["scene"], cell["batch"], cell["provider"], values, "event_eager_median")
            row["median_us"] = median_us
            row["mean_us"] = mean_us
            row["n"] = n
        except Exception as exc:
            row = make_row(
                cell["scene"], cell["batch"], cell["provider"], [],
                "event_eager_median", f"{type(exc).__name__}: {exc}",
            )
        rows.append(row)
    return rows


def print_table(rows: list[dict]) -> None:
    print("")
    print("=" * 118)
    print("Sampler latency | us per call (lower is better)")
    print("-" * 118)
    print(
        f"{'scene':>22} | {'batch':>5} | {'provider':>12} | "
        f"{'median_us':>10} | {'mean_us':>10} | {'samples':>7} | {'error':>20}"
    )
    print("-" * 118)
    for row in rows:
        med = f"{row['median_us']:.2f}" if isinstance(row.get("median_us"), (int, float)) else "ERR"
        avg = f"{row['mean_us']:.2f}" if isinstance(row.get("mean_us"), (int, float)) else "ERR"
        err = row.get("error") or ""
        print(
            f"{SCENE_LABELS[row['scene']]:>22} | {row['batch']:5d} | "
            f"{DISPLAY[row['provider']]:>12} | {med:>10} | {avg:>10} | "
            f"{row['n']:7d} | {err[:20]:>20}"
        )
    print("=" * 118)


def write_csv(path: str, rows: list[dict]) -> None:
    if not path or not rows:
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


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark HPC-Ops sampler kernels.")
    parser.add_argument("--scenes", nargs="+", default=SCENES, choices=SCENES)
    parser.add_argument("--batches", type=int, nargs="+", default=DEFAULT_BATCHES)
    parser.add_argument("--providers", nargs="+", default=PROVIDERS, choices=PROVIDERS)
    parser.add_argument("--timing", choices=["nsys", "event"], default="nsys")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--tag", default="")
    parser.add_argument("--nsys-timeout", type=int, default=600)
    parser.add_argument("--csv", default="")
    parser.add_argument("--jsonl", default="")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-cells", default="[]", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker:
        run_worker(args)
        return 0

    if args.iters < 1 or args.warmup < 0:
        raise SystemExit("--iters must be >= 1 and --warmup must be >= 0")

    cells = build_cells(args)
    tag = args.tag or f"sampler_{int(time.time())}"
    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "log"

    if args.timing == "nsys":
        rows, report_file = rows_from_nsys(args, cells, output_dir, tag)
    else:
        rows = rows_from_events(args, cells)
        report_file = None

    print_table(rows)
    write_csv(args.csv, rows)
    write_jsonl(args.jsonl, rows)
    if report_file:
        print(f"nsys report: {report_file}")
    if args.csv:
        print(f"CSV: {args.csv}")
    if args.jsonl:
        print(f"JSONL: {args.jsonl}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
