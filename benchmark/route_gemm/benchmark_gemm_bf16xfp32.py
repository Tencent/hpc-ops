# Copyright (C) 2025 Tencent.

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import median

import torch


sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../../build/lib.*/"))[0]))

import hpc  # noqa: E402


DEFAULT_M_LIST = [2, 4, 8, 16, 48, 96, 208, 512, 1024, 2048, 4096]
PROVIDERS = ["hpc-ops-bf16xfp32", "FP32(cuBLAS)", "TF32(cuBLAS)"]


def parse_int_list(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def percentile(values, pct):
    values = sorted(values)
    idx = int(round((len(values) - 1) * pct / 100.0))
    return values[idx]


def tflops(m, n, k, us):
    return (2.0 * m * n * k) * 1e-12 / (us * 1e-6)


def bench_cuda_events(fn, flush, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    out = None
    for _ in range(iters):
        flush.zero_()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        stop.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(stop) * 1000.0)
    return median(times), percentile(times, 90), out


def run_nsys_steps(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(iters):
        torch.cuda.nvtx.range_push("step")
        try:
            fn()
        finally:
            torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


def extract_nvtx_us(report_prefix: Path):
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
            if name in ("step", ":step"):
                samples.append(float(entry["Projected Duration (ns)"]) / 1000.0)
        return samples
    except Exception:
        return []


def error_metrics(out, ref):
    out = out.float()
    ref = ref.float()
    diff = (out - ref).abs()
    rel = diff / ref.abs().clamp_min(1e-6)
    cosine = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0)
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "max_rel": rel.max().item(),
        "mean_rel": rel.mean().item(),
        "cosine": cosine.item(),
    }


def make_inputs(m, n, k, scale, device):
    x = torch.randn((m, k), dtype=torch.float32, device=device).to(torch.bfloat16)
    w = torch.randn((n, k), dtype=torch.float32, device=device)
    w_high = w.to(torch.bfloat16)
    w_low = ((w - w_high.float()) / scale).to(torch.bfloat16)
    return x, w, w_high, w_low


def build_runner(provider, x, w, w_high, w_low, scale, split_flag):
    if provider == "hpc-ops-bf16xfp32":
        return lambda: hpc.gemm_bf16xfp32(x, w_high, w_low, scale, True, True, split_flag)
    if provider == "FP32(cuBLAS)":
        return lambda: torch.matmul(x.float(), w.t())
    if provider == "TF32(cuBLAS)":
        return lambda: torch.matmul(x.float(), w.t())
    raise ValueError(f"unknown provider: {provider}")


def run_nsys_worker(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    scale = 1.0 / 256.0
    x, w, w_high, w_low = make_inputs(args.worker_m, args.n, args.k, scale, "cuda")
    split_flag = hpc.get_gemm_bf16xfp32_workspace(args.n)
    torch.backends.cuda.matmul.allow_tf32 = args.worker_provider == "TF32(cuBLAS)"
    run = build_runner(args.worker_provider, x, w, w_high, w_low, scale, split_flag)
    run_nsys_steps(run, args.warmup, args.iters)


def bench_nsys_provider(provider, m, n, k, args, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    report_prefix = out_dir / f"m{m}_{provider_key(provider)}"
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
        "--sample=none",
        "--cpuctxsw=none",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop-shutdown",
        "-t",
        "cuda,nvtx",
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--worker-m",
        str(m),
        "--worker-provider",
        provider,
        "--n",
        str(n),
        "--k",
        str(k),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--seed",
        str(args.seed),
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=args.nsys_timeout,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired:
        return None, None, 0, "nsys profile timeout"
    if proc.returncode != 0 and not os.path.exists(report_file):
        stderr = proc.stderr.decode(errors="replace").strip().splitlines()
        return None, None, 0, stderr[-1] if stderr else "nsys profile failed"

    vals = extract_nvtx_us(report_prefix)
    if not vals:
        return None, None, 0, "no step samples in nsys output"
    return float(median(vals)), float(percentile(vals, 90)), len(vals), None


def benchmark_shape(m, n, k, providers, args, flush):
    scale = 1.0 / 256.0
    x, w, w_high, w_low = make_inputs(m, n, k, scale, "cuda")
    split_flag = hpc.get_gemm_bf16xfp32_workspace(n)

    torch.backends.cuda.matmul.allow_tf32 = False
    ref = torch.matmul(x.float(), w.t())
    torch.cuda.synchronize()

    results = {}
    outputs = {}
    for provider in providers:
        torch.backends.cuda.matmul.allow_tf32 = provider == "TF32(cuBLAS)"
        run = build_runner(provider, x, w, w_high, w_low, scale, split_flag)
        if args.timing == "event":
            us, p90_us, out = bench_cuda_events(run, flush, args.warmup, args.iters)
            samples = args.iters
            err = None
        else:
            us, p90_us, samples, err = bench_nsys_provider(provider, m, n, k, args, args._nsys_out_dir)
            out = run()
            torch.cuda.synchronize()
        results[provider] = {
            "us": us,
            "p90_us": p90_us,
            "tflops": tflops(m, n, k, us) if us else None,
            "samples": samples,
            "error": err,
        }
        outputs[provider] = out

    errors = {}
    for provider, out in outputs.items():
        errors[provider] = error_metrics(out, ref)

    if args.check:
        fp32_err = errors.get("FP32(cuBLAS)")
        if fp32_err is not None and fp32_err["max_abs"] != 0.0:
            raise AssertionError(f"FP32(cuBLAS) should match reference exactly, got {fp32_err['max_abs']}")
        hpc_err = errors.get("hpc-ops-bf16xfp32")
        if hpc_err is not None:
            if hpc_err["max_abs"] > args.max_abs_tol or hpc_err["mean_abs"] > args.mean_abs_tol:
                raise AssertionError(
                    "hpc-ops-bf16xfp32 accuracy check failed: "
                    f"max_abs={hpc_err['max_abs']:.6f}, mean_abs={hpc_err['mean_abs']:.6f}"
                )

    row = {"m": m, "n": n, "k": k}
    row["timing"] = "event_cuda_median" if args.timing == "event" else "nsys_eager_nvtx_median"
    for provider in PROVIDERS:
        if provider not in results:
            continue
        prefix = provider_key(provider)
        row[f"{prefix}_us"] = results[provider]["us"]
        row[f"{prefix}_p90_us"] = results[provider]["p90_us"]
        row[f"{prefix}_tflops"] = results[provider]["tflops"]
        row[f"{prefix}_samples"] = results[provider]["samples"]
        row[f"{prefix}_error"] = results[provider]["error"]
        row[f"{prefix}_max_abs"] = errors[provider]["max_abs"]
        row[f"{prefix}_mean_abs"] = errors[provider]["mean_abs"]
        row[f"{prefix}_cosine"] = errors[provider]["cosine"]

    if "hpc-ops-bf16xfp32" in results and "FP32(cuBLAS)" in results:
        if results["FP32(cuBLAS)"]["us"] and results["hpc-ops-bf16xfp32"]["us"]:
            row["hpc_vs_fp32_speedup"] = results["FP32(cuBLAS)"]["us"] / results["hpc-ops-bf16xfp32"]["us"]
    if "hpc-ops-bf16xfp32" in results and "TF32(cuBLAS)" in results:
        if results["TF32(cuBLAS)"]["us"] and results["hpc-ops-bf16xfp32"]["us"]:
            row["hpc_vs_tf32_speedup"] = results["TF32(cuBLAS)"]["us"] / results["hpc-ops-bf16xfp32"]["us"]
    return row


def provider_key(provider):
    return {
        "hpc-ops-bf16xfp32": "hpc",
        "FP32(cuBLAS)": "torch_fp32",
        "TF32(cuBLAS)": "torch_tf32",
    }[provider]


def print_tflops_table(rows, providers):
    headers = ["M"] + [f"{p} TFLOP/s" for p in providers]
    widths = [max(len(h), 8) for h in headers]
    print("\n" + "  ".join(h.rjust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        values = [str(row["m"])]
        for provider in providers:
            value = row.get(provider_key(provider) + "_tflops")
            values.append(f"{value:.2f}" if isinstance(value, (int, float)) else "ERR")
        print("  ".join(v.rjust(w) for v, w in zip(values, widths)))


def print_csv(rows):
    print(
        "\n"
        "m,n,k,hpc_us,hpc_p90_us,torch_fp32_us,torch_fp32_p90_us,"
        "torch_tf32_us,torch_tf32_p90_us,hpc_vs_fp32,hpc_vs_tf32,"
        "hpc_tflops,torch_fp32_tflops,torch_tf32_tflops,"
        "hpc_max_abs,hpc_mean_abs,tf32_max_abs,tf32_mean_abs"
    )
    for row in rows:
        print(
            f"{row['m']},{row['n']},{row['k']},"
            f"{row.get('hpc_us', float('nan')):.2f},{row.get('hpc_p90_us', float('nan')):.2f},"
            f"{row.get('torch_fp32_us', float('nan')):.2f},{row.get('torch_fp32_p90_us', float('nan')):.2f},"
            f"{row.get('torch_tf32_us', float('nan')):.2f},{row.get('torch_tf32_p90_us', float('nan')):.2f},"
            f"{row.get('hpc_vs_fp32_speedup', float('nan')):.2f},"
            f"{row.get('hpc_vs_tf32_speedup', float('nan')):.2f},"
            f"{row.get('hpc_tflops', float('nan')):.2f},"
            f"{row.get('torch_fp32_tflops', float('nan')):.2f},"
            f"{row.get('torch_tf32_tflops', float('nan')):.2f},"
            f"{row.get('hpc_max_abs', float('nan')):.6f},"
            f"{row.get('hpc_mean_abs', float('nan')):.6f},"
            f"{row.get('torch_tf32_max_abs', float('nan')):.6f},"
            f"{row.get('torch_tf32_mean_abs', float('nan')):.6f}"
        )


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = [
        "m",
        "n",
        "k",
        "hpc_us",
        "hpc_p90_us",
        "torch_fp32_us",
        "torch_fp32_p90_us",
        "torch_tf32_us",
        "torch_tf32_p90_us",
        "hpc_vs_fp32_speedup",
        "hpc_vs_tf32_speedup",
        "hpc_tflops",
        "torch_fp32_tflops",
        "torch_tf32_tflops",
        "hpc_max_abs",
        "hpc_mean_abs",
        "torch_tf32_max_abs",
        "torch_tf32_mean_abs",
        "hpc_cosine",
        "torch_fp32_cosine",
        "torch_tf32_cosine",
        "hpc_samples",
        "torch_fp32_samples",
        "torch_tf32_samples",
        "hpc_error",
        "torch_fp32_error",
        "torch_tf32_error",
        "timing",
    ]
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_jsonl(path, rows):
    if not path:
        return
    with Path(path).open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark hpc-ops BF16xFP32 GEMM vs cuBLAS.")
    parser.add_argument("--m-list", default=",".join(str(x) for x in DEFAULT_M_LIST))
    parser.add_argument("--n", type=int, default=192)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--timing", choices=["nsys", "event"], default="nsys")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=10086)
    parser.add_argument("--flush-mb", type=int, default=128)
    parser.add_argument("--providers", nargs="+", default=PROVIDERS, choices=PROVIDERS)
    parser.add_argument("--output-dir", default="", help="Output directory for nsys reports.")
    parser.add_argument("--tag", default="", help="Subdirectory name for nsys reports.")
    parser.add_argument("--nsys-timeout", type=int, default=300)
    parser.add_argument("--csv", type=str, default="", help="Optional output CSV path.")
    parser.add_argument("--jsonl", type=str, default="", help="Optional output JSONL path.")
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-abs-tol", type=float, default=0.01)
    parser.add_argument("--mean-abs-tol", type=float, default=0.001)
    parser.add_argument("--print-csv", action="store_true", help="Print machine-readable CSV rows.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-m", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--worker-provider", choices=PROVIDERS, default="hpc-ops-bf16xfp32", help=argparse.SUPPRESS)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.worker:
        run_nsys_worker(args)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("This benchmark is tuned for SM90 GPUs")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    m_values = parse_int_list(args.m_list)
    flush = torch.empty(args.flush_mb * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    tag = args.tag or f"route_gemm_{int(time.time())}"
    args._nsys_out_dir = (
        Path(args.output_dir) / tag
        if args.output_dir
        else Path(__file__).resolve().parent / "log" / tag
    )
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    rows = []
    try:
        print(f"Device: {torch.cuda.get_device_name()}  N={args.n} K={args.k}")
        print(f"Providers: {', '.join(args.providers)}")
        for m in m_values:
            rows.append(benchmark_shape(m, args.n, args.k, args.providers, args, flush))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32

    print_tflops_table(rows, args.providers)
    if args.print_csv:
        print_csv(rows)
    if args.csv:
        write_csv(args.csv, rows)
    if args.jsonl:
        write_jsonl(args.jsonl, rows)
    if args.timing == "nsys":
        print(f"\nnsys reports: {args._nsys_out_dir}")
    if args.csv:
        print(f"\nWrote CSV: {args.csv}")
    if args.jsonl:
        print(f"Wrote JSONL: {args.jsonl}")
    print("\nBenchmark finished!")


if __name__ == "__main__":
    main()
