import os
import sys
from pathlib import Path

sys.path.insert(
    0,
    os.path.realpath(
        sorted([p for p in Path(__file__).parent.glob("../build/lib.*/") if "linux" in str(p)])[0]
    ),
)

import torch
import torch.cuda.nvtx as nvtx

import hpc

# ---------------------------- config ----------------------------

NUM_GROUP = 192
# kTileM ladder: 16, 32, 48, 64, 128, 256 — sample one TM per bucket.
TM_CASES = [8, 16, 32, 48, 64, 128]

SHAPES = [
    ("gate_up_tp8", 384, 4096),
    ("down_tp8", 4096, 192),
]

SF_VEC = 32
WARMUP = 5
ITER = 200


def _mxfp8_kTileM(num_seq_per_group_avg: int, n: int) -> int:
    """Mirror C++ mxfp8_dispatch_kTileM (TMA path supports 1SM and 2SM)."""
    use_2sm = (n % 256 == 0) and (num_seq_per_group_avg > 32)
    if use_2sm:
        for ktm in (64, 96, 128, 160, 192):
            if num_seq_per_group_avg <= ktm:
                return ktm
        return 256
    for ktm in (16, 32, 48, 64, 128):
        if num_seq_per_group_avg <= ktm:
            return ktm
    return 256


# ---------------------------- builders ----------------------------


def _build_inputs(num_group, seq, n, k, device):
    seqlens = torch.full((num_group,), seq, dtype=torch.int32, device=device)
    cu_seqlens = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.cumsum(seqlens, 0).to(torch.int32),
        ]
    )
    m_total = int(cu_seqlens[-1].item())

    x_fp8 = torch.randn((m_total, k), dtype=torch.float32, device=device).to(torch.float8_e4m3fn)
    w_fp8 = torch.randn((num_group, n, k), dtype=torch.float32, device=device).to(
        torch.float8_e4m3fn
    )
    sfx = torch.full((m_total, k // SF_VEC), 127, dtype=torch.uint8, device=device)
    sfw = torch.full((num_group, n, k // SF_VEC), 127, dtype=torch.uint8, device=device)

    # SFA prepack uses TMA-path kTileM. cp_async ladder is identical to 1SM TMA
    # ladder for kTileM bucketing, so the same prepacked SFA works for both.
    kTileM = _mxfp8_kTileM(seq, n)
    sfx_packed, sfw_packed = hpc.prepack_mxfp8_scale(
        sfx, sfw, cu_seqlens, num_seq_per_group_avg=seq
    )

    return {
        "x_fp8": x_fp8,
        "w_fp8": w_fp8,
        "sfx_packed": sfx_packed,
        "sfw_packed": sfw_packed,
        "seqlens": seqlens,
        "cu_seqlens": cu_seqlens,
        "m_total": m_total,
        "kTileM": kTileM,
    }


def _build_tma_runner(inputs, seq):
    def run():
        return hpc.group_gemm_mxfp8(
            inputs["x_fp8"],
            inputs["w_fp8"],
            inputs["sfx_packed"],
            inputs["sfw_packed"],
            inputs["seqlens"],
            inputs["cu_seqlens"],
            num_seq_per_group_avg=seq,
        )

    return run


def _build_cp_async_seq_runner(inputs, seq):
    """cp.async with sequential read (x_row_map=None)."""

    def run():
        return hpc.group_gemm_cp_async_mxfp8(
            inputs["x_fp8"],
            inputs["w_fp8"],
            inputs["sfx_packed"],
            inputs["sfw_packed"],
            inputs["seqlens"],
            inputs["cu_seqlens"],
            num_seq_per_group_avg=seq,
            x_row_map=None,
        )

    return run


def _build_cp_async_map_runner(inputs, seq, device):
    """cp.async with random row_map (simulates fuse_moe row indirection)."""
    m_total = inputs["m_total"]
    perm = torch.randperm(m_total, device=device, dtype=torch.int64)
    x_permuted = inputs["x_fp8"][perm].contiguous()
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(m_total, device=device, dtype=torch.int64)
    row_map = inv_perm.to(torch.int32).contiguous()

    def run():
        return hpc.group_gemm_cp_async_mxfp8(
            x_permuted,
            inputs["w_fp8"],
            inputs["sfx_packed"],
            inputs["sfw_packed"],
            inputs["seqlens"],
            inputs["cu_seqlens"],
            num_seq_per_group_avg=seq,
            x_row_map=row_map,
        )

    return run


# ---------------------------- timing ----------------------------


def _time(fn, label):
    """Returns per-call latency in microseconds."""
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()

    nvtx.range_push(label)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITER):
        fn()
    end.record()
    torch.cuda.synchronize()
    nvtx.range_pop()
    return start.elapsed_time(end) * 1e3 / ITER  # us


def _try_time(fn, label):
    try:
        return _time(fn, label)
    except Exception as e:
        return f"N/A ({type(e).__name__})"


# ---------------------------- main ----------------------------


def _fmt(us):
    if isinstance(us, str):
        return f"{us:>16}"
    return f"{us:>10.2f} us"


def _fmt_speedup(base, us):
    if isinstance(us, str) or isinstance(base, str):
        return "      ---"
    delta = (base - us) / base * 100.0
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:6.1f}%"


def main():
    device = torch.device("cuda:0")
    torch.manual_seed(2026)

    print(f"NUM_GROUP={NUM_GROUP}  WARMUP={WARMUP}  ITER={ITER}")
    print()

    for label, n, k in SHAPES:
        print(f"=== {label}  (n={n}, k={k}) ===")
        header = (
            f"{'TM':>4} {'kTileM':>7} {'TMA':>13} {'cp.async-seq':>13} "
            f"{'cp.async-map':>13} {'seq vs TMA':>11} {'map vs TMA':>11}"
        )
        print(header)
        print("-" * len(header))

        for tm in TM_CASES:
            inputs = _build_inputs(NUM_GROUP, tm, n, k, device)
            kTileM = inputs["kTileM"]

            # TMA
            t_tma = _try_time(_build_tma_runner(inputs, tm), f"{label}_TM{tm}_TMA")

            # cp.async sequential
            t_cpa = _try_time(_build_cp_async_seq_runner(inputs, tm), f"{label}_TM{tm}_cpa_seq")

            # cp.async + row_map
            t_cpm = _try_time(
                _build_cp_async_map_runner(inputs, tm, device),
                f"{label}_TM{tm}_cpa_map",
            )

            print(
                f"{tm:>4} {kTileM:>7} {_fmt(t_tma)} {_fmt(t_cpa)} {_fmt(t_cpm)} "
                f"{_fmt_speedup(t_tma, t_cpa):>11} {_fmt_speedup(t_tma, t_cpm):>11}"
            )

        print()


if __name__ == "__main__":
    main()
