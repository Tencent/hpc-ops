#!/usr/bin/env python3
# Copyright (C) 2026 Tencent.

"""Unified FusedMoE bench driver.

Usage:
    python3 bench.py --tp 8 --ep 1                       # TP=8 sweep
    python3 bench.py --tp 1 --ep 8                       # EP=8 sweep
    python3 bench.py --tp 1 --ep 8 --bs 4 16 64 256      # custom batches
    python3 bench.py --tp 1 --ep 8 --models hunyuan-v3   # one model
    python3 bench.py --tp 1 --ep 8 --backends hpcops vllm vllm_cutlass sglang

Path discovery (CLI required, env vars as fallback):
    --vllm-root        / $VLLM_ROOT
    --sglang-root      / $SGLANG_ROOT
    --hpcops-root      / $HPCOPS_ROOT

If a backend is unavailable, the driver skips it with an explicit reason.
"""
from __future__ import annotations

import argparse
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

# (E, topk, hidden, intermediate_full)
MODELS = {
    "qwen3-235b":  (128, 8, 4096, 1536),
    "hunyuan-v1":  (64,  8, 4096, 3072),
    "hunyuan-v2":  (128, 8, 4096, 4096),
    "hunyuan-v3":  (192, 8, 4096, 1536),
    "deepseek-v3": (256, 8, 7168, 2048),
}

ALL_BACKENDS = [
    "hpcops", "sglang", "vllm", "vllm_cutlass",
]

DISPLAY = {
    "hpcops":       "hpc-ops",
    "sglang":       "SGLang",
    "vllm":         "vLLM-Triton",
    "vllm_cutlass": "vLLM-CUTLASS",
}

BS_TP = [4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
BS_EP = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

DEFAULT_MODELS = ["hunyuan-v3", "deepseek-v3", "qwen3-235b"]


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
@dataclass
class Roots:
    vllm: Optional[str]
    sglang: Optional[str]
    hpcops: Optional[str]


def resolve_roots(args) -> Roots:
    def _pick(cli_val, env_var):
        if cli_val:
            return cli_val
        return os.environ.get(env_var)
    return Roots(
        vllm=_pick(args.vllm_root, "VLLM_ROOT"),
        sglang=_pick(args.sglang_root, "SGLANG_ROOT"),
        hpcops=_pick(args.hpcops_root, "HPCOPS_ROOT"),
    )


def required_roots(backend: str) -> list[str]:
    """Which source roots a backend needs."""
    if backend == "hpcops":
        return ["hpcops", "vllm", "sglang"]
    if backend in ("vllm", "vllm_cutlass"):
        return ["vllm"]
    if backend == "sglang":
        return ["vllm", "sglang"]
    raise ValueError(backend)


# ---------------------------------------------------------------------------
# Subprocess env builder
# ---------------------------------------------------------------------------
def build_env(backend: str, roots: Roots, gpu_id: int) -> dict:
    """Construct the worker subprocess environment."""
    pp_parts = []
    ld_parts = []

    if backend == "hpcops":
        hpc_root = roots.hpcops
        pp_parts.append(hpc_root)
        nvshmem = os.path.join(hpc_root, "3rd/ucl/nvshmem/lib")
        if os.path.isdir(nvshmem):
            ld_parts.append(nvshmem)

    if roots.vllm:
        pp_parts.append(roots.vllm)
    if backend in ("sglang", "hpcops") and roots.sglang:
        pp_parts.append(os.path.join(roots.sglang, "python"))
    pp_parts.append(str(SCRIPT_DIR))

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = ":".join(p for p in pp_parts if p)
    if ld_parts:
        env["LD_LIBRARY_PATH"] = ":".join(
            ld_parts + ([env["LD_LIBRARY_PATH"]] if "LD_LIBRARY_PATH" in env else [])
        )
    compat = os.environ.get("CUDA_COMPAT_LIBCUDA")
    if compat and os.path.exists(compat):
        env["LD_PRELOAD"] = compat

    if backend in ("sglang", "hpcops") and roots.sglang:
        env["SGLANG_ROOT"] = roots.sglang
        env.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    return env


# ---------------------------------------------------------------------------
# Dry-run probe (fast import check)
# ---------------------------------------------------------------------------
def dry_run(backend: str, roots: Roots, gpu_id: int) -> tuple[bool, str]:
    """Try a 1-step kernel call at minimal shape to confirm the backend
    actually works.  Returns (ok, message).
    """
    cmd = [
        sys.executable, str(SCRIPT_DIR / "worker.py"),
        "--backend", backend,
        "--num-seq", "8", "--hidden", "128", "--intermediate-per-rank", "128",
        "--num-expert-local", "2", "--num-expert-total", "2", "--num-topk", "1",
        "--n-timed", "1",
    ]
    env = build_env(backend, roots, gpu_id)
    try:
        out = subprocess.run(
            cmd, env=env, timeout=60,
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
                 if e.get("Name", "").strip().strip('"') == ":step"]
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
) -> CellResult:
    E_total, topk, H, N_full = MODELS[model]
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
    ]

    attempts = run_nsys_profile(
        backend, roots, gpu_id, worker_args,
        report_file=rpath, meta_out=meta_path,
    )
    if attempts > 3:
        return CellResult(
            model=model, bs=bs, backend=backend, tp=tp, ep=ep,
            n_expert_total=E_total, n_expert_local=E_local, topk=topk,
            hidden=H, intermediate_full=N_full, intermediate_per_rank=N_per_rank,
            avg_per_group=avg,
            median_us=None, mean_us=None, p10_us=None, n_samples=0,
            sgl_cfg_source=None, nsys_attempts=attempts,
            error="nsys profile failed after 3 attempts",
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(
        description="Unified FusedMoE bench driver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--tp", type=int, default=8)
    p.add_argument("--ep", type=int, default=1)
    p.add_argument("--bs", type=int, nargs="+",
                   help="Batch sizes (default: TP=8 -> BS_TP, EP>1 -> BS_EP)")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                   choices=list(MODELS.keys()))
    p.add_argument("--backends", nargs="+", default=ALL_BACKENDS,
                   choices=ALL_BACKENDS)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("-o", "--output-dir",
                   default=None,
                   help="Output directory; default: ./log next to bench.py")
    p.add_argument("--tag", default=None,
                   help="Subdir under output-dir; default: tp{tp}_ep{ep}_<ts>")

    g = p.add_argument_group("backend roots (CLI overrides env)")
    g.add_argument("--vllm-root", help="default: $VLLM_ROOT")
    g.add_argument("--sglang-root", help="default: $SGLANG_ROOT")
    g.add_argument("--hpcops-root", help="default: $HPCOPS_ROOT")

    args = p.parse_args()
    roots = resolve_roots(args)
    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "log"

    # Default batch size by mode.
    bs_list = args.bs
    if bs_list is None:
        bs_list = BS_EP if args.ep > 1 else BS_TP

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
        ok, msg = dry_run(b, roots, args.gpu)
        if not ok:
            skipped[b] = f"dry_run failed: {msg}"
            continue
        avail.append(b)

    width = 22 + 14 * len(avail) + 14
    print("=" * width)
    print(f"Bench: TP={args.tp}  EP={args.ep}")
    print(f"Output: {out_root}")
    print(f"Backends ok: {[DISPLAY[b] for b in avail]}")
    if skipped:
        print(f"Backends skipped:")
        for b, why in skipped.items():
            print(f"  {DISPLAY[b]:>14}  {why}")
    print(f"Models: {args.models}")
    print(f"BS:     {bs_list}")
    print("=" * width)
    if not avail:
        print("No backends available; aborting.", file=sys.stderr)
        return 2

    hdr = f"{'Model':<14} {'BS':>6}"
    for b in avail:
        hdr += f"  {DISPLAY[b]:>12}"
    hdr += f"  {'avg/group':>10}"
    print(hdr)
    print("-" * width)

    results_path = out_root / "results.jsonl"
    log_path = out_root / "bench.log"
    with open(results_path, "w") as rf, open(log_path, "w") as lf:
        def _log(line: str):
            print(line, flush=True)
            lf.write(line + "\n"); lf.flush()

        for model in args.models:
            for bs in bs_list:
                row_us: dict[str, Optional[float]] = {}
                avg = 0.0
                for b in avail:
                    res = profile_one(
                        b, roots, args.gpu, model, bs, args.tp, args.ep,
                        output_dir, tag,
                    )
                    row_us[b] = res.median_us
                    avg = res.avg_per_group
                    rf.write(json.dumps(asdict(res)) + "\n"); rf.flush()
                row = f"{model:<14} {bs:>6}"
                for b in avail:
                    v = row_us[b]
                    row += f"  {v:>12.1f}" if v is not None else f"  {'ERR':>12}"
                row += f"  {avg:>10.2f}"
                _log(row)

    print("-" * width)
    print(f"All times in µs (median of 50 :step replays).")
    print(f"JSONL: {results_path}")
    print(f"Log  : {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
