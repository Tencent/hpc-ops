import argparse
import csv
import os
import sys
from pathlib import Path
from statistics import median

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_LIBS = list((REPO_ROOT / "build").glob("lib.*"))
if BUILD_LIBS:
    sys.path.insert(0, os.path.realpath(BUILD_LIBS[0]))
sys.path.insert(0, os.path.realpath(REPO_ROOT))

import hpc  # noqa: E402


DEFAULT_M_LIST = [1, 16, 48, 96, 208, 512, 1024, 2048, 4096]
PROVIDERS = ["hpc-ops-bf16xfp32", "FP32(cublas)", "TF32(cublas)"]


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
    if provider == "FP32(cublas)":
        return lambda: torch.matmul(x.float(), w.t())
    if provider == "TF32(cublas)":
        return lambda: torch.matmul(x.float(), w.t())
    raise ValueError(f"unknown provider: {provider}")


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
        torch.backends.cuda.matmul.allow_tf32 = provider == "TF32(cublas)"
        run = build_runner(provider, x, w, w_high, w_low, scale, split_flag)
        us, p90_us, out = bench_cuda_events(run, flush, args.warmup, args.iters)
        results[provider] = {
            "us": us,
            "p90_us": p90_us,
            "tflops": tflops(m, n, k, us),
        }
        outputs[provider] = out

    errors = {}
    for provider, out in outputs.items():
        errors[provider] = error_metrics(out, ref)

    if args.check:
        fp32_err = errors.get("FP32(cublas)")
        if fp32_err is not None and fp32_err["max_abs"] != 0.0:
            raise AssertionError(f"FP32(cublas) should match reference exactly, got {fp32_err['max_abs']}")
        hpc_err = errors.get("hpc-ops-bf16xfp32")
        if hpc_err is not None:
            if hpc_err["max_abs"] > args.max_abs_tol or hpc_err["mean_abs"] > args.mean_abs_tol:
                raise AssertionError(
                    "hpc-ops-bf16xfp32 accuracy check failed: "
                    f"max_abs={hpc_err['max_abs']:.6f}, mean_abs={hpc_err['mean_abs']:.6f}"
                )

    row = {"m": m, "n": n, "k": k}
    for provider in PROVIDERS:
        if provider not in results:
            continue
        prefix = provider_key(provider)
        row[f"{prefix}_us"] = results[provider]["us"]
        row[f"{prefix}_p90_us"] = results[provider]["p90_us"]
        row[f"{prefix}_tflops"] = results[provider]["tflops"]
        row[f"{prefix}_max_abs"] = errors[provider]["max_abs"]
        row[f"{prefix}_mean_abs"] = errors[provider]["mean_abs"]
        row[f"{prefix}_cosine"] = errors[provider]["cosine"]

    if "hpc-ops-bf16xfp32" in results and "FP32(cublas)" in results:
        row["hpc_vs_fp32_speedup"] = results["FP32(cublas)"]["us"] / results["hpc-ops-bf16xfp32"]["us"]
    if "hpc-ops-bf16xfp32" in results and "TF32(cublas)" in results:
        row["hpc_vs_tf32_speedup"] = results["TF32(cublas)"]["us"] / results["hpc-ops-bf16xfp32"]["us"]
    return row


def provider_key(provider):
    return {
        "hpc-ops-bf16xfp32": "hpc",
        "FP32(cublas)": "torch_fp32",
        "TF32(cublas)": "torch_tf32",
    }[provider]


def print_tflops_table(rows, providers):
    headers = ["M"] + [f"{p} TFLOP/s" for p in providers]
    widths = [max(len(h), 8) for h in headers]
    print("\n" + "  ".join(h.rjust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        values = [str(row["m"])]
        for provider in providers:
            values.append(f"{row[provider_key(provider) + '_tflops']:.2f}")
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
    ]
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark hpc-ops BF16xFP32 GEMM vs cuBLAS.")
    parser.add_argument("--m-list", default=",".join(str(x) for x in DEFAULT_M_LIST))
    parser.add_argument("--n", type=int, default=192)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=10086)
    parser.add_argument("--flush-mb", type=int, default=128)
    parser.add_argument("--providers", nargs="+", default=PROVIDERS, choices=PROVIDERS)
    parser.add_argument("--csv", type=str, default="", help="Optional output CSV path.")
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-abs-tol", type=float, default=0.01)
    parser.add_argument("--mean-abs-tol", type=float, default=0.001)
    parser.add_argument("--print-csv", action="store_true", help="Print machine-readable CSV rows.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("This benchmark is tuned for SM90 GPUs")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    m_values = parse_int_list(args.m_list)
    flush = torch.empty(args.flush_mb * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
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
        print(f"\nWrote CSV: {args.csv}")
    print("\nBenchmark finished!")


if __name__ == "__main__":
    main()
