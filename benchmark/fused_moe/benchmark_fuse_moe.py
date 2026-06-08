#!/usr/bin/env python3
# Copyright (C) 2025 Tencent.

"""FusedMoE FP8 benchmark.

Usage:
    python3 benchmark/fused_moe/benchmark_fuse_moe.py --tp 8 --ep 1
    python3 benchmark/fused_moe/benchmark_fuse_moe.py --tp 1 --ep 8 --bs 4 16 64

Path discovery (CLI args override env vars):
    --vllm-root        / $VLLM_ROOT
    --sglang-root      / $SGLANG_ROOT
    --hpcops-root      / $HPCOPS_ROOT

Unavailable comparison backends are skipped with an explicit warning.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent

@dataclass(frozen=True)
class ModelSpec:
    experts: int
    topk: int
    hidden: int
    intermediate: int


# name -> (experts, topk, hidden, intermediate_full)
MODEL_PRESETS = {
    "qwen3-235b":  (128, 8, 4096, 1536),
    "hunyuan-v1":  (64,  8, 4096, 3072),
    "hunyuan-v2":  (128, 8, 4096, 4096),
    "hunyuan-v3":  (192, 8, 4096, 1536),
    "deepseek-v3": (256, 8, 7168, 2048),
}
MODELS = {name: ModelSpec(*shape) for name, shape in MODEL_PRESETS.items()}

ALL_BACKENDS = [
    "hpcops", "sglang", "vllm", "vllm_cutlass",
]

DISPLAY = {
    "hpcops":       "HPC-Ops",
    "sglang":       "SGLang",
    "vllm":         "vLLM Triton",
    "vllm_cutlass": "vLLM CUTLASS",
}

DEFAULT_TP_BATCHES = "4,16,32,64,128,256,512,1024,2048,4096,8192,16384"
DEFAULT_EP_BATCHES = "4,8,16,32,64,128,256,512,1024,2048"

DEFAULT_MODELS = ["qwen3-235b", "hunyuan-v3", "deepseek-v3"]


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
@dataclass
class Roots:
    vllm: Optional[str]
    sglang: Optional[str]
    hpcops: Optional[str]


def _candidate_roots() -> list[Path]:
    candidates: list[Path] = []
    for path in (SCRIPT_DIR, *SCRIPT_DIR.parents):
        candidates.append(path)

    deduped = []
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            deduped.append(resolved)
            seen.add(resolved)
    return deduped


def _discover_hpcops_root() -> Optional[str]:
    for root in _candidate_roots():
        if list((root / "hpc").glob("_C*.so")):
            return str(root)
        if list(root.glob("build/lib.*/hpc/_C*.so")) and (root / "hpc").is_dir():
            return str(root)
    return None


def resolve_roots(args) -> Roots:
    def _pick(cli_val, env_var, discover=None):
        if cli_val:
            return cli_val
        env_val = os.environ.get(env_var)
        if env_val:
            return env_val
        return discover() if discover is not None else None
    return Roots(
        vllm=_pick(args.vllm_root, "VLLM_ROOT"),
        sglang=_pick(args.sglang_root, "SGLANG_ROOT"),
        hpcops=_pick(args.hpcops_root, "HPCOPS_ROOT", _discover_hpcops_root),
    )


def required_roots(backend: str) -> list[str]:
    """Which source roots a backend needs."""
    if backend == "hpcops":
        return ["hpcops"]
    if backend in ("vllm", "vllm_cutlass"):
        return []
    if backend == "sglang":
        return []
    raise ValueError(backend)


def parse_int_list(value: str) -> list[int]:
    items = value.replace(",", " ").split()
    if not items:
        raise argparse.ArgumentTypeError("expected at least one integer")
    try:
        return [int(item) for item in items]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer list: {value}") from exc


def parse_model_shape(value: str) -> tuple[str, ModelSpec]:
    parts = value.split(":")
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            "model shape must be name:experts:topk:hidden:intermediate"
        )
    name = parts[0]
    try:
        spec = ModelSpec(*(int(item) for item in parts[1:]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid model shape: {value}") from exc
    return name, spec


# ---------------------------------------------------------------------------
# Subprocess env builder
# ---------------------------------------------------------------------------
def build_env(backend: str, roots: Roots, gpu_id: int) -> dict:
    """Construct the worker subprocess environment."""
    pp_parts = []
    ld_parts = []

    if backend == "hpcops":
        hpc_root = roots.hpcops
        build_libs = [str(p) for p in Path(hpc_root).glob("build/lib.*") if (p / "hpc").is_dir()]
        pp_parts.extend(build_libs)
        if list((Path(hpc_root) / "hpc").glob("_C*.so")):
            pp_parts.append(hpc_root)
        nvshmem = os.path.join(hpc_root, "3rd/ucl/nvshmem/lib")
        if os.path.isdir(nvshmem):
            ld_parts.append(nvshmem)

    if roots.vllm:
        pp_parts.append(roots.vllm)
    if backend == "sglang" and roots.sglang:
        pp_parts.append(os.path.join(roots.sglang, "python"))
    pp_parts.append(str(SCRIPT_DIR))

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = ":".join(p for p in pp_parts if p)
    if backend == "hpcops" and roots.hpcops:
        env["HPC_OPS_ROOT"] = roots.hpcops
    if ld_parts:
        env["LD_LIBRARY_PATH"] = ":".join(
            ld_parts + ([env["LD_LIBRARY_PATH"]] if "LD_LIBRARY_PATH" in env else [])
        )
    compat = os.environ.get("CUDA_COMPAT_LIBCUDA")
    if compat and os.path.exists(compat):
        env["LD_PRELOAD"] = compat

    if backend == "sglang" and roots.sglang:
        env["SGLANG_ROOT"] = roots.sglang
        env.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    return env


# ---------------------------------------------------------------------------
# Dry-run probe (fast import check)
# ---------------------------------------------------------------------------
def dry_run(
    backend: str, roots: Roots, gpu_id: int,
    *, num_seq: int, hidden: int, intermediate_per_rank: int,
    num_experts: int, topk: int, warmup: int, timeout: int,
) -> tuple[bool, str]:
    """Try a 1-step kernel call at minimal shape to confirm the backend
    actually works.  Returns (ok, message).
    """
    cmd = [
        sys.executable, str(SCRIPT_DIR / "worker.py"),
        "--backend", backend,
        "--num-seq", str(num_seq),
        "--hidden", str(hidden),
        "--intermediate-per-rank", str(intermediate_per_rank),
        "--num-expert-local", str(num_experts),
        "--num-expert-total", str(num_experts),
        "--num-topk", str(topk),
        "--warmup", str(warmup),
        "--n-timed", "1",
    ]
    env = build_env(backend, roots, gpu_id)
    try:
        out = subprocess.run(
            cmd, env=env, timeout=timeout,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if out.returncode == 0:
            return True, "ok"
        msg = (out.stderr.decode(errors="replace").strip().splitlines() or
               ["(no stderr)"])[-1]
        return False, msg
    except subprocess.TimeoutExpired:
        return False, "dry_run timeout"
    except Exception as e:
        return False, f"dry_run failed: {e}"


# ---------------------------------------------------------------------------
# nsys wrapper + nvtx extractor
# ---------------------------------------------------------------------------
def run_nsys_profile(
    backend: str, roots: Roots, gpu_id: int, worker_args: list[str],
    report_file: str, meta_out: str | None,
    *, attempts: int = 3, timeout: int = 600,
) -> int:
    """Run worker.py under nsys profile."""
    rep = report_file + ".nsys-rep"
    env = build_env(backend, roots, gpu_id)

    py_cmd = [sys.executable, str(SCRIPT_DIR / "worker.py"), *worker_args]
    if meta_out:
        py_cmd += ["--meta-out", meta_out]

    nsys_cmd = [
        "nsys", "profile", "-f", "true", "-o", report_file,
        "--sample=none",
        "--cpuctxsw=none",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop-shutdown",
        "--cuda-graph-trace=node",
        "-t", "cuda,nvtx",
        *py_cmd,
    ]
    for attempt in range(1, attempts + 1):
        if os.path.exists(rep):
            os.remove(rep)
        try:
            subprocess.run(
                nsys_cmd, env=env, timeout=timeout,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass
        if os.path.exists(rep):
            return attempt
    return attempts + 1  # signal failure


def extract_nvtx(report_file: str) -> list[int]:
    cmd = [
        "nsys", "stats", "--report", "nvtx_gpu_proj_trace",
        "--force-export=true", "-q", "-f", "json",
        report_file + ".nsys-rep",
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        raw = json.loads(out.decode())
        data = raw[0]["data"] if isinstance(raw, list) and "data" in raw[0] else raw
        stats = [int(e["Projected Duration (ns)"]) for e in data
                 if e.get("Name", "").strip().strip('"') in ("step", ":step")]
        return stats[2:]  # drop first 2 warmup-spill replays
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Per-cell driver
# ---------------------------------------------------------------------------
@dataclass
class CellResult:
    model: str
    bs: int
    backend: str
    tp: int
    ep: int
    n_expert_total: int
    n_expert_local: int
    topk: int
    hidden: int
    intermediate_full: int
    intermediate_per_rank: int
    avg_per_group: float
    median_us: Optional[float]
    mean_us: Optional[float]
    p10_us: Optional[float]
    n_samples: int
    sgl_cfg_source: Optional[str]
    nsys_attempts: int
    error: Optional[str]


def profile_one(
    backend: str, roots: Roots, gpu_id: int,
    model: str, bs: int, tp: int, ep: int,
    output_dir: Path, tag: str,
    models: dict[str, ModelSpec],
    *, warmup: int, iters: int, seed: int, nsys_attempts: int, nsys_timeout: int,
) -> CellResult:
    spec = models[model]
    E_total, topk, H, N_full = spec.experts, spec.topk, spec.hidden, spec.intermediate
    if E_total % ep != 0:
        return CellResult(
            model=model, bs=bs, backend=backend, tp=tp, ep=ep,
            n_expert_total=E_total, n_expert_local=0, topk=topk,
            hidden=H, intermediate_full=N_full, intermediate_per_rank=0,
            avg_per_group=0.0,
            median_us=None, mean_us=None, p10_us=None, n_samples=0,
            sgl_cfg_source=None, nsys_attempts=0,
            error=f"E={E_total} not divisible by ep={ep}",
        )

    E_local = E_total // ep
    N_per_rank = N_full // tp
    avg = bs * topk / E_local

    cell_dir = output_dir / tag / backend
    cell_dir.mkdir(parents=True, exist_ok=True)
    rpath = str(cell_dir / f"{model}_bs{bs}")
    meta_path = str(cell_dir / f"{model}_bs{bs}.meta.json")

    worker_args = [
        "--backend", backend,
        "--num-seq", str(bs), "--hidden", str(H),
        "--intermediate-per-rank", str(N_per_rank),
        "--num-expert-local", str(E_local),
        "--num-expert-total", str(E_local),  # EP-rank-0 simulation
        "--num-topk", str(topk),
        "--model", model, "--tp", str(tp), "--ep", str(ep),
        "--seed", str(seed),
        "--warmup", str(warmup),
        "--n-timed", str(iters),
    ]

    attempts = run_nsys_profile(
        backend, roots, gpu_id, worker_args,
        report_file=rpath, meta_out=meta_path,
        attempts=nsys_attempts, timeout=nsys_timeout,
    )
    if attempts > nsys_attempts:
        return CellResult(
            model=model, bs=bs, backend=backend, tp=tp, ep=ep,
            n_expert_total=E_total, n_expert_local=E_local, topk=topk,
            hidden=H, intermediate_full=N_full, intermediate_per_rank=N_per_rank,
            avg_per_group=avg,
            median_us=None, mean_us=None, p10_us=None, n_samples=0,
            sgl_cfg_source=None, nsys_attempts=attempts,
            error=f"nsys profile failed after {nsys_attempts} attempts",
        )

    ns = extract_nvtx(rpath)
    if not ns:
        return CellResult(
            model=model, bs=bs, backend=backend, tp=tp, ep=ep,
            n_expert_total=E_total, n_expert_local=E_local, topk=topk,
            hidden=H, intermediate_full=N_full, intermediate_per_rank=N_per_rank,
            avg_per_group=avg,
            median_us=None, mean_us=None, p10_us=None, n_samples=0,
            sgl_cfg_source=None, nsys_attempts=attempts,
            error="no :step samples in nsys output",
        )

    sgl_cfg = None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        sgl_cfg = meta.get("sgl_cfg_source")
    except Exception:
        pass

    arr = np.array(ns, dtype=np.float64) / 1000.0  # ns -> us
    return CellResult(
        model=model, bs=bs, backend=backend, tp=tp, ep=ep,
        n_expert_total=E_total, n_expert_local=E_local, topk=topk,
        hidden=H, intermediate_full=N_full, intermediate_per_rank=N_per_rank,
        avg_per_group=avg,
        median_us=float(np.median(arr)),
        mean_us=float(np.mean(arr)),
        p10_us=float(np.percentile(arr, 10)),
        n_samples=int(arr.size),
        sgl_cfg_source=sgl_cfg,
        nsys_attempts=attempts,
        error=None,
    )


def print_table(rows: list[dict]) -> None:
    print("")
    print("=" * 132)
    print("FusedMoE FP8 latency | us per call (lower is better)")
    print("-" * 132)
    print(
        f"{'model':>14} | {'mode':>9} | {'bs':>6} | {'backend':>12} | "
        f"{'median_us':>10} | {'mean_us':>10} | {'samples':>7} | {'avg/group':>9} | {'error':>18}"
    )
    print("-" * 132)
    for row in rows:
        med = f"{row['median_us']:.2f}" if isinstance(row.get("median_us"), (int, float)) else "ERR"
        avg = f"{row['mean_us']:.2f}" if isinstance(row.get("mean_us"), (int, float)) else "ERR"
        err = row.get("error") or ""
        mode = f"tp{row['tp']}_ep{row['ep']}"
        print(
            f"{row['model']:>14} | {mode:>9} | {row['bs']:6d} | {DISPLAY[row['backend']]:>12} | "
            f"{med:>10} | {avg:>10} | {row['n_samples']:7d} | {row['avg_per_group']:9.2f} | {err[:18]:>18}"
        )
    print("=" * 132)


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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(
        description="Benchmark FusedMoE FP8 backends.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--tp", type=int, default=8)
    p.add_argument("--ep", type=int, default=1)
    p.add_argument("--bs", type=int, nargs="+",
                   help="Batch sizes. Overrides --tp-batches/--ep-batches.")
    p.add_argument("--tp-batches", type=parse_int_list, default=parse_int_list(DEFAULT_TP_BATCHES),
                   help=f"Default batch list when ep=1. Default: {DEFAULT_TP_BATCHES}")
    p.add_argument("--ep-batches", type=parse_int_list, default=parse_int_list(DEFAULT_EP_BATCHES),
                   help=f"Default batch list when ep>1. Default: {DEFAULT_EP_BATCHES}")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                   help="Model names from presets or --model-shape.")
    p.add_argument(
        "--model-shape", action="append", type=parse_model_shape, default=[],
        metavar="NAME:E:TOPK:HIDDEN:INTERMEDIATE",
        help="Add or override a model shape, e.g. custom:128:8:4096:1536.",
    )
    p.add_argument("--providers", dest="backends", nargs="+",
                   default=ALL_BACKENDS, choices=ALL_BACKENDS)
    p.add_argument("--backends", dest="backends", nargs="+",
                   choices=ALL_BACKENDS, help=argparse.SUPPRESS)
    p.add_argument("--timing", choices=["nsys"], default="nsys",
                   help="nsys: CUDA Graph replay under NVTX step ranges, reported as median latency.")
    p.add_argument("--warmup", type=int, default=3,
                   help="Warmup calls before graph capture and replay warmup count.")
    p.add_argument("--iters", type=int, default=52,
                   help="NVTX-marked graph replays. The first two samples are dropped when reading nsys.")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed used by worker tensor builders.")
    p.add_argument("--nsys-attempts", type=int, default=3,
                   help="Retry count for each nsys profile cell.")
    p.add_argument("--nsys-timeout", type=int, default=600,
                   help="Timeout in seconds for each nsys profile attempt.")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("-o", "--output-dir",
                   default="",
                   help="Output directory for nsys reports; default: ./log next to benchmark_fuse_moe.py")
    p.add_argument("--tag", default="",
                   help="Subdir under output-dir; default: tp{tp}_ep{ep}_<ts>")
    p.add_argument("--csv", default="", help="Optional CSV output path.")
    p.add_argument("--jsonl", default="", help="Optional JSONL output path.")

    g = p.add_argument_group("backend roots (CLI overrides env)")
    g.add_argument("--vllm-root", help="default: $VLLM_ROOT")
    g.add_argument("--sglang-root", help="default: $SGLANG_ROOT")
    g.add_argument("--hpcops-root", help="default: $HPCOPS_ROOT")

    d = p.add_argument_group("dry-run probe")
    d.add_argument("--dry-run-num-seq", type=int, default=8)
    d.add_argument("--dry-run-hidden", type=int, default=128)
    d.add_argument("--dry-run-intermediate", type=int, default=128)
    d.add_argument("--dry-run-experts", type=int, default=2)
    d.add_argument("--dry-run-topk", type=int, default=1)
    d.add_argument("--dry-run-timeout", type=int, default=60)

    args = p.parse_args()
    if args.warmup < 0:
        p.error("--warmup must be >= 0")
    if args.iters <= 2:
        p.error("--iters must be > 2 because the first two nsys samples are dropped")
    if args.nsys_attempts < 1:
        p.error("--nsys-attempts must be >= 1")
    if args.nsys_timeout < 1 or args.dry_run_timeout < 1:
        p.error("timeouts must be >= 1 second")
    if args.dry_run_topk > args.dry_run_experts:
        p.error("--dry-run-topk must be <= --dry-run-experts")
    models = dict(MODELS)
    for name, spec in args.model_shape:
        models[name] = spec
    unknown_models = [name for name in args.models if name not in models]
    if unknown_models:
        p.error(
            f"unknown model(s): {', '.join(unknown_models)}. "
            "Use --model-shape NAME:E:TOPK:HIDDEN:INTERMEDIATE to add custom shapes."
        )
    roots = resolve_roots(args)
    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "log"

    # Default batch size by mode.
    bs_list = args.bs
    if bs_list is None:
        bs_list = args.ep_batches if args.ep > 1 else args.tp_batches

    tag = args.tag or f"tp{args.tp}_ep{args.ep}_{int(time.time())}"
    out_root = output_dir / tag
    out_root.mkdir(parents=True, exist_ok=True)

    # ------------- Backend availability gate -------------
    avail = []
    skipped = {}
    for b in args.backends:
        missing = [r for r in required_roots(b)
                   if getattr(roots, r) is None or
                   not os.path.isdir(getattr(roots, r))]
        if missing:
            skipped[b] = f"missing roots: {missing}"
            continue
        ok, msg = dry_run(
            b, roots, args.gpu,
            num_seq=args.dry_run_num_seq,
            hidden=args.dry_run_hidden,
            intermediate_per_rank=args.dry_run_intermediate,
            num_experts=args.dry_run_experts,
            topk=args.dry_run_topk,
            warmup=args.warmup,
            timeout=args.dry_run_timeout,
        )
        if not ok:
            skipped[b] = f"dry_run failed: {msg}"
            continue
        avail.append(b)

    if skipped:
        for b, why in skipped.items():
            print(f"[warn] skip {DISPLAY[b]}: {why}", file=sys.stderr)
    if not avail:
        print("No backends available; aborting.", file=sys.stderr)
        return 2

    rows = []
    log_path = out_root / "bench.log"
    with open(log_path, "w") as lf:
        for model in args.models:
            for bs in bs_list:
                for b in avail:
                    res = profile_one(
                        b, roots, args.gpu, model, bs, args.tp, args.ep,
                        output_dir, tag, models,
                        warmup=args.warmup, iters=args.iters, seed=args.seed,
                        nsys_attempts=args.nsys_attempts,
                        nsys_timeout=args.nsys_timeout,
                    )
                    row = asdict(res)
                    row["timing"] = "nsys_graph_nvtx_median"
                    rows.append(row)
                    med = f"{res.median_us:.2f}us" if res.median_us is not None else "ERR"
                    lf.write(f"{model} bs={bs} {DISPLAY[b]} {med}\n")
                    lf.flush()

    print_table(rows)
    write_csv(args.csv, rows)
    write_jsonl(args.jsonl, rows)
    print(f"nsys reports: {out_root}")
    if args.csv:
        print(f"CSV: {args.csv}")
    if args.jsonl:
        print(f"JSONL: {args.jsonl}")
    print(f"Log: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
