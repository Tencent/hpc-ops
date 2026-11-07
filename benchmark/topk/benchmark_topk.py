# Copyright (C) 2026 Tencent.

"""Top-K benchmark.

Benchmarks the HPC-Ops exact FP32 Top-K operator (`hpc.topk`) against the
PyTorch `torch.topk` reference and, when its optional dependencies are
installed, the DeepSelect SM90 kernel over a grid of row counts, row widths,
top-k values and synthetic score distributions.

Distributions:
    - gauss3:  three-cluster Gaussian mixture (per-row weights).
    - normal:  standard Gaussian.
    - spike30: 30% of the entries pinned to one large value (heavy ties).
    - band8:   Gaussian quantized to eight discrete levels (heavy ties).
    - uniform: uniform [0, 1).
    - coarse:  fp16-rounded Gaussian (ties at the coarse-bin granularity).
    - conc64:  64 discrete levels (extreme tie groups).

Combinations whose element count exceeds ``--max-elements`` are skipped so a
single sweep never exhausts device memory. The `hpc-ops` provider leaves the
counters buffer self-cleaned between calls; `torch.topk` needs no scratch.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import zlib
from pathlib import Path
from statistics import median

import torch


sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../../build/lib.*/"))[0]))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backends import create, known  # noqa: E402


DISTRIBUTIONS = ["gauss3", "normal", "spike30", "band8", "uniform", "coarse", "conc64"]
DEFAULT_KS = [512, 2048]
DEFAULT_M_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
DEFAULT_N_LIST = [
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1048576,
    2097152,
    4194304,
]


def parse_int_list(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def generate_scores(dist, m, n, seed):
    """Deterministic per-shape score tensor for one distribution."""
    torch.manual_seed(zlib.crc32(f"{dist}:{m}:{n}:{seed}".encode()) % (2**31))
    if dist == "normal":
        return torch.randn(m, n, dtype=torch.float32, device="cuda")
    if dist == "uniform":
        return torch.rand(m, n, dtype=torch.float32, device="cuda")
    scores = torch.randn(m, n, dtype=torch.float32, device="cuda")
    if dist == "gauss3":
        pick = torch.rand(m, device="cuda")
        near = (pick < 0.5).nonzero().flatten()
        mid = ((pick >= 0.5) & (pick < 0.8)).nonzero().flatten()
        far = (pick >= 0.8).nonzero().flatten()
        scores[near] *= 0.25
        scores[mid] = scores[mid] * 0.5 + 4.0
        scores[far] = scores[far] * 0.25 - 4.0
        return scores
    if dist == "spike30":
        hits = torch.rand(m, n, device="cuda") < 0.30
        return scores.masked_fill(hits, 1000.0)
    if dist == "band8":
        return scores.mul_(1.5).add_(4.0).floor_().clamp_(0.0, 7.0)
    if dist == "coarse":
        return scores.half().float()
    if dist == "conc64":
        return scores.uniform_(0.0, 64.0).floor_()
    raise ValueError(f"unknown distribution {dist!r}")


def bench_median_us(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    return median(times)


def values_match(reference, got):
    """Sorted top-k values must match the torch reference exactly."""
    return torch.equal(torch.sort(reference).values, torch.sort(got).values)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ks", type=parse_int_list, default=DEFAULT_KS, help="top-k values")
    parser.add_argument("--m-list", type=parse_int_list, default=DEFAULT_M_LIST, help="row counts")
    parser.add_argument("--n-list", type=parse_int_list, default=DEFAULT_N_LIST, help="row widths")
    parser.add_argument("--dists", nargs="+", default=DISTRIBUTIONS, choices=DISTRIBUTIONS)
    parser.add_argument("--providers", nargs="+", default=known(), choices=known())
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0, help="extra seed mixed into every shape")
    parser.add_argument(
        "--max-elements",
        type=int,
        default=9_500_000_000,
        help="skip combinations above this element count",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="restrict the sweep to at most 256 rows, 3 warmup and 15 iters",
    )
    parser.add_argument(
        "--check", action="store_true", help="verify the top-k values of row 0 against torch.topk"
    )
    parser.add_argument("--csv", default="topk_latency.csv", help="output CSV path")
    args = parser.parse_args()

    if args.smoke:
        args.m_list = [m for m in args.m_list if m <= 256]
        args.warmup = min(args.warmup, 3)
        args.iters = min(args.iters, 15)

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    done = set()
    if os.path.exists(args.csv):
        with open(args.csv) as existing:
            for row in csv.reader(existing):
                if row and row[0] != "dist":
                    done.add(tuple(row[:5]))
        print(f"resume: {len(done)} measurements already present")

    fieldnames = ["dist", "k", "m", "n", "provider", "us", "check"]
    output = open(args.csv, "a", newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    if not done:
        writer.writeheader()

    backends = {name: create(name) for name in args.providers}
    device = torch.device("cuda")
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    for name, backend in backends.items():
        print(f"warming up {name} ...", flush=True)
        backend.warmup(device, args.ks)

    total = (
        len(args.dists) * len(args.ks) * len(args.m_list) * len(args.n_list) * len(args.providers)
    )
    finished = 0
    mismatches = []
    print(f"sm_count={sm_count}, {total} combinations, csv={args.csv}")

    for dist in args.dists:
        for m in args.m_list:
            for n in args.n_list:
                if m * n > args.max_elements:
                    continue
                pending = [
                    (k, p)
                    for k in args.ks
                    for p in args.providers
                    if (dist, str(k), str(m), str(n), p) not in done
                ]
                if not pending:
                    continue
                try:
                    logits = generate_scores(dist, m, n, args.seed)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"skip {dist} m={m} n={n}: score generation out of memory", flush=True)
                    continue
                ke = torch.full((m,), n, dtype=torch.int32, device="cuda")
                num_valid = torch.tensor([m], dtype=torch.int32, device="cuda")
                for top_k, provider in pending:
                    backend = backends[provider]
                    if not backend.supports(m, n):
                        writer.writerow(
                            {
                                "dist": dist,
                                "k": top_k,
                                "m": m,
                                "n": n,
                                "provider": provider,
                                "us": "",
                                "check": "SKIP_INT32",
                            }
                        )
                        output.flush()
                        print(
                            f"[..] {dist} k={top_k} m={m} n={n} {provider} "
                            f"SKIP_INT32 (rows*width >= 2^31)",
                            flush=True,
                        )
                        continue
                    try:
                        call, out = backend.prepare(logits, ke, num_valid, top_k, sm_count)
                        us = bench_median_us(call, args.warmup, args.iters)
                        note = ""
                        if args.check and (m == 1 or m * n >= 2**30):
                            call()
                            gathered = logits.gather(1, out.long())
                            rows = range(0, m, max(1, m // 4)) if m * n >= 2**30 else (0,)
                            bad = [
                                r
                                for r in rows
                                if not values_match(logits[r].topk(top_k).values, gathered[r])
                            ]
                            note = "OK" if not bad else f"MISMATCH{bad}"
                            if bad:
                                mismatches.append((dist, top_k, m, n, provider, bad))
                    except torch.cuda.OutOfMemoryError:
                        us, note = float("nan"), "OOM"
                        torch.cuda.empty_cache()
                    writer.writerow(
                        {
                            "dist": dist,
                            "k": top_k,
                            "m": m,
                            "n": n,
                            "provider": provider,
                            "us": f"{us:.2f}" if us == us else "OOM",
                            "check": note,
                        }
                    )
                    output.flush()
                    finished += 1
                    shown = f"{us:.1f}" if us == us else us
                    print(
                        f"[{finished}/{total}] {dist} k={top_k} m={m} n={n} "
                        f"{provider} {shown} us {note}",
                        flush=True,
                    )
                del logits, ke, num_valid
                torch.cuda.empty_cache()
    output.close()
    if mismatches:
        print("VALUE MISMATCHES:")
        for entry in mismatches:
            print(f"  {entry}")
    print(f"done, {finished}/{total} measured, csv={args.csv}")


if __name__ == "__main__":
    main()
