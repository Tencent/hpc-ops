import os
import sys
from pathlib import Path
from statistics import median

import pytest
import torch

repo_root = Path(__file__).resolve().parents[1]
build_libs = list((repo_root / "build").glob("lib.*"))
if build_libs:
    sys.path.insert(0, os.path.realpath(build_libs[0]))
sys.path.insert(0, os.path.realpath(repo_root))

import hpc  # noqa: E402


def _parse_int_list(name, default):
    text = os.getenv(name)
    if not text:
        return default
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _percentile(values, pct):
    values = sorted(values)
    idx = int(round((len(values) - 1) * pct / 100.0))
    return values[idx]


def _bench(fn, flush, warmup, iters):
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
    return median(times), _percentile(times, 90), out


def _error_metrics(out, ref):
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 9, reason="skip on non sm90")
def test_bench_gemm_bf16xfp32_n192_vs_torch():
    torch.manual_seed(10086)
    torch.cuda.manual_seed(10086)

    n = 192
    k = int(os.getenv("HPC_BENCH_GEMM_K", "4096"))
    m_values = _parse_int_list("HPC_BENCH_GEMM_M", [1, 16, 48, 96, 208, 512, 1024, 2048, 4096])
    warmup = int(os.getenv("HPC_BENCH_WARMUP", "8"))
    iters = int(os.getenv("HPC_BENCH_ITERS", "30"))
    scale = 1.0 / 256.0
    flush = torch.empty(128 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")

    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    rows = []
    try:
        for m in m_values:
            x = torch.randn((m, k), dtype=torch.float32, device="cuda").to(torch.bfloat16)
            w = torch.randn((n, k), dtype=torch.float32, device="cuda")
            w_high = w.to(torch.bfloat16)
            w_low = ((w - w_high.float()) / scale).to(torch.bfloat16)
            split_flag = hpc.get_gemm_bf16xfp32_workspace(n)

            torch.backends.cuda.matmul.allow_tf32 = False
            ref = torch.matmul(x.float(), w.t())
            torch.cuda.synchronize()

            hpc_us, hpc_p90, hpc_out = _bench(
                lambda: hpc.gemm_bf16xfp32(x, w_high, w_low, scale, True, True, split_flag),
                flush,
                warmup,
                iters,
            )

            torch.backends.cuda.matmul.allow_tf32 = False
            fp32_us, fp32_p90, fp32_out = _bench(lambda: torch.matmul(x.float(), w.t()), flush, warmup, iters)

            torch.backends.cuda.matmul.allow_tf32 = True
            tf32_us, tf32_p90, tf32_out = _bench(lambda: torch.matmul(x.float(), w.t()), flush, warmup, iters)

            hpc_err = _error_metrics(hpc_out, ref)
            tf32_err = _error_metrics(tf32_out, ref)
            fp32_err = _error_metrics(fp32_out, ref)
            rows.append((m, hpc_us, hpc_p90, fp32_us, fp32_p90, tf32_us, tf32_p90, hpc_err, tf32_err))

            assert fp32_err["max_abs"] == 0.0
            assert hpc_err["max_abs"] <= 0.01
            assert hpc_err["mean_abs"] <= 0.001
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32

    print(f"\nDevice: {torch.cuda.get_device_name()}  shape: n={n}, k={k}")
    print("m,hpc_us,hpc_p90_us,torch_fp32_us,torch_fp32_p90_us,torch_tf32_us,torch_tf32_p90_us,"
          "hpc_vs_fp32,hpc_vs_tf32,hpc_max_abs,hpc_mean_abs,tf32_max_abs,tf32_mean_abs")
    for m, hpc_us, hpc_p90, fp32_us, fp32_p90, tf32_us, tf32_p90, hpc_err, tf32_err in rows:
        print(
            f"{m},{hpc_us:.2f},{hpc_p90:.2f},{fp32_us:.2f},{fp32_p90:.2f},"
            f"{tf32_us:.2f},{tf32_p90:.2f},{fp32_us / hpc_us:.2f},"
            f"{tf32_us / hpc_us:.2f},{hpc_err['max_abs']:.6f},"
            f"{hpc_err['mean_abs']:.6f},{tf32_err['max_abs']:.6f},"
            f"{tf32_err['mean_abs']:.6f}"
        )
