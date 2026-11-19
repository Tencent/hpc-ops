# Copyright (C) 2026 Tencent.

"""Benchmark the SM120 fused MoE against vLLM backends; see README.md.

Each (shape, backend) pair runs ``worker.py`` in its own process. All backends
of a model run one after another on the same GPU, because two GPUs of the same
model can differ by several percent at large batch sizes. With more GPUs than
models, each model runs on ``len(gpus) // len(models)`` GPUs and the tables
report the mean over them. Results are printed as markdown tables and written
as JSON and CSV.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from statistics import mean

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from worker import MODELS  # noqa: E402

# vLLM backends that run each format on SM120 with a stock install. CUTLASS and
# FlashInfer CUTLASS reject 128-block FP8 there, and the Triton MXFP4 paths reject
# MXFP4 x MXFP8. b12x also runs MXFP4 x MXFP8 but needs the optional b12x package.
DEFAULT_BACKENDS = {
    "fp8_block": ["hpc", "triton", "deep_gemm", "marlin"],
    "mxfp4": ["hpc", "marlin", "deep_gemm", "flashinfer_cutlass"],
}
DEFAULT_TOKENS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]


def parse_model_shape(value):
    parts = value.split(":")
    if len(parts) not in (6, 7) or parts[1] not in DEFAULT_BACKENDS:
        raise argparse.ArgumentTypeError(
            "expected name:{fp8_block,mxfp4}:experts:topk:hidden:intermediate[:swiglu_limit]"
        )
    name, quant, *dims = parts
    shape = dict(
        quant=quant,
        experts=int(dims[0]),
        topk=int(dims[1]),
        hidden=int(dims[2]),
        intermediate=int(dims[3]),
        swiglu_limit=float(dims[4]) if len(dims) == 5 else None,
    )
    return name, shape


def worker_cmd(shape, backend, args, out):
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "worker.py"),
        "--backend",
        backend,
        "--quant",
        shape["quant"],
        "--experts",
        str(shape["experts"]),
        "--topk",
        str(shape["topk"]),
        "--hidden",
        str(shape["hidden"]),
        "--intermediate",
        str(shape["intermediate"]),
        "--tokens",
        *map(str, args.tokens),
        "--rotate",
        str(args.rotate),
        "--iters",
        str(args.iters),
        "--reps",
        str(args.reps),
        "--out",
        str(out),
    ]
    if shape["swiglu_limit"] is not None:
        cmd += ["--swiglu-limit", str(shape["swiglu_limit"])]
    return cmd


def assign_gpus(models, gpus):
    """Map each GPU to the models it runs, in order."""
    assignment = {gpu: [] for gpu in gpus}
    if len(gpus) >= len(models):
        copies = len(gpus) // len(models)
        for i, model in enumerate(models):
            for gpu in gpus[i * copies : (i + 1) * copies]:
                assignment[gpu].append(model)
    else:
        for i, model in enumerate(models):
            assignment[gpus[i % len(gpus)]].append(model)
    return {gpu: ms for gpu, ms in assignment.items() if ms}


def run_jobs(assignment, plan, shapes, args, out_dir):
    results = {}
    lock = threading.Lock()

    def run(gpu, models):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        for model in models:
            for backend in plan[model]:
                stem = f"{model}__{backend}__gpu{gpu}"
                out = out_dir / f"{stem}.json"
                # A stale file from an earlier run must never stand in for this one.
                out.unlink(missing_ok=True)
                with open(out_dir / f"{stem}.log", "w") as log:
                    proc = subprocess.run(
                        worker_cmd(shapes[model], backend, args, out),
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                if proc.returncode != 0:
                    data = dict(
                        backend=backend,
                        rows=[],
                        error=f"worker exited with code {proc.returncode}; see {stem}.log",
                    )
                else:
                    try:
                        data = json.loads(out.read_text())
                    except (OSError, ValueError) as e:
                        data = dict(backend=backend, rows=[], error=f"no result: {e}")
                with lock:
                    results[(model, backend, gpu)] = data
                    print(
                        f"done {model} {backend} on GPU {gpu}: "
                        f"{data.get('error') or data.get('kernel')}",
                        flush=True,
                    )

    threads = [threading.Thread(target=run, args=item) for item in assignment.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def print_tables(model, backends, results, tokens, gpus):
    runs = {b: [results.get((model, b, g), {}) for g in gpus] for b in backends}
    us = {}
    for b in backends:
        samples = {}
        for run in runs[b]:
            for r in run.get("rows", []):
                if r.get("median_us"):
                    samples.setdefault(r["tokens"], []).append(r["median_us"])
        us[b] = {t: mean(v) for t, v in samples.items() if len(v) == len(gpus)}
    ref = next(
        (b for b in ("hpc", "hpc-bf16-direct", "hpc-direct") if b in backends and us[b]),
        None,
    )
    others = [b for b in backends if b != ref]

    gpu_list = ", ".join(map(str, gpus))
    print(f"\n## {model}: latency per MoE layer call (us, lower is better; GPU {gpu_list})\n")
    for b in backends:
        errors = [run["error"] for run in runs[b] if run.get("error")]
        if errors:
            print(f"- {b}: {errors[0]}")
        else:
            print(f"- {b}: {runs[b][0].get('kernel')}")
    header = ["tokens", *backends]
    if ref:
        header += [f"{b} / {ref}" for b in others]
    print("\n| " + " | ".join(header) + " |")
    print("|" + "|".join(["---:"] * len(header)) + "|")
    for t in tokens:
        cells = [str(t)] + [f"{us[b].get(t):.1f}" if us[b].get(t) else "-" for b in backends]
        if ref:
            for b in others:
                a, r = us[b].get(t), us[ref].get(t)
                cells.append(f"{a / r:.2f}x" if a and r else "-")
        print("| " + " | ".join(cells) + " |")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS),
        help=f"presets: {', '.join(MODELS)}; or names given with --model-shape",
    )
    p.add_argument(
        "--model-shape",
        type=parse_model_shape,
        action="append",
        default=[],
        metavar="NAME:QUANT:E:TOPK:H:I[:SWIGLU_LIMIT]",
        help="per-GPU shape; QUANT is fp8_block or mxfp4",
    )
    p.add_argument(
        "--backends",
        nargs="+",
        help="hpc-direct, hpc-bf16-direct, and/or vLLM --moe-backend names "
        "(default: hpc plus the backends that run the format on SM120)",
    )
    p.add_argument("--tokens", type=int, nargs="+", default=DEFAULT_TOKENS)
    p.add_argument("--gpus", type=int, nargs="+", default=[0])
    p.add_argument(
        "--rotate",
        type=int,
        default=8,
        help="routings captured per graph, so small batches are not L2-resident",
    )
    p.add_argument("--iters", type=int, default=20, help="graph replays per timing sample")
    p.add_argument("--reps", type=int, default=9, help="timing samples; the median is reported")
    p.add_argument("--out-dir", default="fused_moe_sm120_results")
    args = p.parse_args()

    shapes = dict(MODELS)
    shapes.update(dict(args.model_shape))
    unknown = [m for m in args.models if m not in shapes]
    if unknown:
        p.error(f"unknown model(s) {unknown}; known: {list(shapes)}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = {m: args.backends or DEFAULT_BACKENDS[shapes[m]["quant"]] for m in args.models}
    assignment = assign_gpus(args.models, args.gpus)
    results = run_jobs(assignment, plan, shapes, args, out_dir)

    with open(out_dir / "results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["model", "backend", "gpu", "kernel", "tokens", "median_us", "min_us", "error"]
        )
        for (model, backend, gpu), data in sorted(results.items()):
            # A worker can finish some token counts and then fail; keep its rows
            # and still record the error.
            if data.get("error"):
                writer.writerow([model, backend, gpu, data.get("kernel"), "", "", "", data["error"]])
            for r in data.get("rows", []):
                writer.writerow(
                    [
                        model,
                        backend,
                        gpu,
                        data.get("kernel"),
                        r["tokens"],
                        r.get("median_us"),
                        r.get("min_us"),
                        r.get("error"),
                    ]
                )
    with open(out_dir / "summary.json", "w") as f:
        json.dump({f"{m}__{b}__gpu{g}": d for (m, b, g), d in results.items()}, f, indent=1)

    for model, backends in plan.items():
        gpus = [gpu for gpu, models in assignment.items() if model in models]
        print_tables(model, backends, results, args.tokens, gpus)


if __name__ == "__main__":
    main()
