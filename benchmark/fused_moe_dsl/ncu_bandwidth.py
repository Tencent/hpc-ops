# Copyright (C) 2026 Tencent.

"""DRAM traffic of the CuTe DSL fused MoE under Nsight Compute; see README.md.

For each (model, token count) the script profiles one warmed-up call of the hpc
entry point, wrapped in NVTX range ``moe``, with the cache flushed before each
kernel, and compares the DRAM bytes moved with the compulsory bytes: weights
and scales of the experts the routing touches, activations in, output out.
The MXFP4 entry point runs on BF16 activations (``x_scale=None``), the path
vLLM takes.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from worker import MODELS  # noqa: E402

DEFAULT_TOKENS = [1, 8, 64, 512, 2048, 8192]
METRICS = [
    "gpu__time_duration.sum",
    "dram__bytes_op_read.sum",
    "dram__bytes_op_write.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes.sum.peak_sustained",
    "dram__cycles_elapsed.avg.per_second",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
]
UNITS = {
    "": 1,
    "%": 1,
    "byte/cycle": 1,
    "byte": 1,
    "kbyte": 1e3,
    "mbyte": 1e6,
    "gbyte": 1e9,
    "ns": 1e-9,
    "us": 1e-6,
    "ms": 1e-3,
    "s": 1,
    "hz": 1,
    "khz": 1e3,
    "mhz": 1e6,
    "ghz": 1e9,
}


def run_op(shape, num_tokens, meta_path):
    """Profiled side: one call of the entry point inside NVTX range ``moe``."""
    import torch

    import hpc
    from hpc import pack_mxfp_weight_scale_sm120, prepare_fp8_mx_mma_weights

    E, H, inter, topk = shape["experts"], shape["hidden"], shape["intermediate"], shape["topk"]
    gen = torch.Generator(device="cuda").manual_seed(1234)

    def fp8(*size):
        return (torch.randn(size, generator=gen, device="cuda") * 0.5).to(torch.float8_e4m3fn)

    def ue8m0(*size):
        return torch.randint(120, 125, size, generator=gen, device="cuda", dtype=torch.uint8)

    if shape["quant"] == "fp8_block":
        w13, w2 = fp8(E, 2 * inter, H), fp8(E, H, inter)
        s13 = torch.rand(E, 2 * inter // 128, H // 128, generator=gen, device="cuda") * 0.01 + 0.005
        s2 = torch.rand(E, H // 128, inter // 128, generator=gen, device="cuda") * 0.01 + 0.005
        _, s13 = prepare_fp8_mx_mma_weights(w13, s13)
        _, s2 = prepare_fp8_mx_mma_weights(w2, s2)
        entry = hpc.fuse_moe_blockwise_fp8_dsl
        extra = {}
        x = fp8(num_tokens, H)
        x_scale = torch.exp2(
            torch.randint(-8, -4, (num_tokens, H // 128), generator=gen, device="cuda").float()
        )
        x_bytes = x.nbytes + x_scale.nbytes
    else:

        def fp4(*size):
            return torch.randint(0, 256, size, generator=gen, device="cuda", dtype=torch.uint8)

        w13, w2 = fp4(E, 2 * inter, H // 2), fp4(E, H, inter // 2)
        s13 = pack_mxfp_weight_scale_sm120(ue8m0(E, 2 * inter, H // 32), 2 * inter, H)
        s2 = pack_mxfp_weight_scale_sm120(ue8m0(E, H, inter // 32), H, inter)
        entry = hpc.fuse_moe_blockwise_w4a8_mxfp4_mxfp8_dsl
        extra = {"swiglu_limit": shape["swiglu_limit"]}
        x = (torch.randn(num_tokens, H, generator=gen, device="cuda") * 2).to(torch.bfloat16)
        x_scale = None
        x_bytes = x.nbytes

    rgen = torch.Generator(device="cuda").manual_seed(num_tokens)
    ids = torch.rand(num_tokens, E, generator=rgen, device="cuda").topk(topk, dim=-1).indices
    ids = ids.to(torch.int32)
    weights = torch.softmax(torch.randn(num_tokens, topk, generator=rgen, device="cuda"), -1)
    weights = weights.float()
    out = torch.empty(num_tokens, H, dtype=torch.bfloat16, device="cuda")

    touched = int(torch.unique(ids).numel())
    per_expert = (w13.nbytes + w2.nbytes + s13.nbytes + s2.nbytes) / E
    compulsory = touched * per_expert + x_bytes + out.nbytes + ids.nbytes + weights.nbytes
    Path(meta_path).write_text(
        json.dumps(dict(experts_touched=touched, compulsory_bytes=compulsory))
    )

    def call():
        entry(
            x,
            x_scale,
            w13,
            s13,
            w2,
            s2,
            ids,
            weights,
            rank_ep=0,
            num_expert_total=E,
            output=out,
            **extra,
        )

    for _ in range(3):
        call()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("moe")
    call()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()


def parse_ncu_csv(text):
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('"ID"'))
    head, units, *rows = csv.reader(lines[start:])
    col = {name: i for i, name in enumerate(head)}

    def get(row, metric):
        value = row[col[metric]].replace(",", "")
        unit = units[col[metric]].lower()
        if unit not in UNITS:
            raise ValueError(f"unknown ncu unit {unit!r} for {metric}")
        return float(value) * UNITS[unit] if value else 0.0

    kernels = []
    for row in rows:
        kernels.append(
            dict(
                name=row[col["Kernel Name"]],
                time=get(row, "gpu__time_duration.sum"),
                read=get(row, "dram__bytes_op_read.sum"),
                write=get(row, "dram__bytes_op_write.sum"),
                dram_pct=get(row, "dram__throughput.avg.pct_of_peak_sustained_elapsed"),
                peak=get(row, "dram__bytes.sum.peak_sustained")
                * get(row, "dram__cycles_elapsed.avg.per_second"),
                tensor_pct=get(
                    row, "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"
                ),
            )
        )
    return kernels


def profile(name, num_tokens, args):
    out_dir = Path(args.out_dir)
    tag = f"{name}_t{num_tokens}"
    meta_path = out_dir / f"{tag}.meta.json"
    cmd = [
        args.ncu,
        "--nvtx",
        "--nvtx-include",
        "moe/",
        "--cache-control",
        "all",
        "--clock-control",
        "none",
        "--metrics",
        ",".join(METRICS),
        "--csv",
        "--page",
        "raw",
        sys.executable,
        str(Path(__file__).resolve()),
        "--profiled",
        "--models",
        name,
        "--tokens",
        str(num_tokens),
        "--meta",
        str(meta_path),
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    (out_dir / f"{tag}.csv").write_text(proc.stdout)
    (out_dir / f"{tag}.log").write_text(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"ncu failed for {tag}, see {out_dir / (tag + '.log')}")
    kernels = parse_ncu_csv(proc.stdout)
    meta = json.loads(meta_path.read_text())
    gemms = [k for k in kernels if "gemm" in k["name"].lower()]
    time = sum(k["time"] for k in kernels)
    dram = sum(k["read"] + k["write"] for k in kernels)
    peak = max(k["peak"] for k in kernels)
    return dict(
        model=name,
        tokens=num_tokens,
        experts_touched=meta["experts_touched"],
        time_us=time * 1e6,
        dram_mb=dram / 1e6,
        compulsory_mb=meta["compulsory_bytes"] / 1e6,
        achieved_gbs=dram / time / 1e9,
        peak_gbs=peak / 1e9,
        gemm_dram_pct=[round(k["dram_pct"]) for k in gemms],
        gemm_tensor_pct=[round(k["tensor_pct"]) for k in gemms],
        kernels=[dict(k, name=k["name"][:80]) for k in kernels],
    )


def print_table(name, results):
    print(f"\n`{name}`:\n")
    print(
        "| tokens | experts touched | time (us) | DRAM (MB) | compulsory (MB) | DRAM / compulsory "
        "| achieved BW | GEMM1 / GEMM2 DRAM | GEMM1 / GEMM2 tensor active |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        pct = r["achieved_gbs"] / r["peak_gbs"] * 100
        print(
            f"| {r['tokens']} | {r['experts_touched']} | {r['time_us']:.1f} | {r['dram_mb']:.1f} "
            f"| {r['compulsory_mb']:.1f} | {r['dram_mb'] / r['compulsory_mb']:.2f}x "
            f"| {r['achieved_gbs']:.0f} GB/s ({pct:.0f}%) "
            f"| {' / '.join(f'{p}%' for p in r['gemm_dram_pct'])} "
            f"| {' / '.join(f'{p}%' for p in r['gemm_tensor_pct'])} |"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--tokens", nargs="+", type=int, default=DEFAULT_TOKENS)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ncu", default="ncu")
    parser.add_argument("--out-dir", default="fused_moe_dsl_ncu/")
    parser.add_argument("--profiled", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--meta", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.profiled:
        run_op(MODELS[args.models[0]], args.tokens[0], args.meta)
        return

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    summary = []
    for name in args.models:
        results = []
        for num_tokens in args.tokens:
            results.append(profile(name, num_tokens, args))
            print(f"{name} tokens={num_tokens} done", file=sys.stderr, flush=True)
        summary += results
        print_table(name, results)
    Path(args.out_dir, "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
