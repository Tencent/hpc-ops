"""Bench mxfp8_group_gemm: hpc.group_gemm_mxfp8 vs flashinfer.grouped_mm_mxfp8.

Mirrors the layout of `bench_mxfp8_vs_fp8.py` (TP8 only):

    NUM_GROUP = 192
    tp8 gate_up: (m = TM * 192,  n = 384,   k = 4096)
    tp8 down:    (m = TM * 192,  n = 4096,  k = 192)

For each (kernel, shape, TM), runs WARMUP + ITER calls inside an NVTX range.
Per-call latency = elapsed / ITER.

Caveats for the flashinfer side:

1. flashinfer's cuDNN backend requires the F8 scale to be 128x4-swizzled, which
   forces ``k // 32`` to be a multiple of 4. The real ``down_tp8`` shape uses
   k=192 (k/32=6) and therefore cannot be run on flashinfer; those cells are
   reported as N/A.

2. cuDNN's MOE block-scale grouped GEMM produces NaN/Inf for
   ``num_experts >= 64`` AND small avg tokens-per-expert (verified empirically
   on cuDNN 9.22 / sm_100). Latency is still reported, but the cell is tagged
   ``(NaN)``.

Run:

    # (optional) lock GPU clock for stable numbers
    nvidia-smi -i 0 -lgc 1965

    python3 tests/bench_group_gemm_mxfp8_vs_flashinfer.py

    # to also capture an nsys timeline:
    nsys profile --trace=cuda,nvtx --stats=true -f true -o /tmp/cmp \\
        python3 tests/bench_group_gemm_mxfp8_vs_flashinfer.py

    nvidia-smi -i 0 -rgc
"""

import math
import os
import sys
from pathlib import Path

# All three flashinfer packages are aligned at 0.6.12 in this env, but keep the
# bypass as a safety net so the bench runs even if a future version mismatch
# resurfaces. Also scrub a stale FLASHINFER_CUBINS_REPOSITORY some shells set.
os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
os.environ.pop("FLASHINFER_CUBINS_REPOSITORY", None)

sys.path.insert(
    0,
    os.path.realpath(
        sorted([p for p in Path(__file__).parent.glob("../build/lib.*/") if "linux" in str(p)])[0]
    ),
)

import torch
import torch.cuda.nvtx as nvtx

import hpc
from flashinfer.grouped_mm import grouped_mm_mxfp8

# ---------------------------- config ----------------------------

NUM_GROUP = 192
TM_CASES = [8, 16, 32, 64, 128, 256]

SHAPES = [
    ("gate_up_tp8", 384, 4096),
    ("down_tp8", 4096, 192),
]

SF_VEC = 32
WARMUP = 5
ITER = 100


# ---------------------------- builders ----------------------------


def _build_common(num_group, seq, n, k, device):
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
    # Uniform UE8M0 scale = 127 -> 2^(127-127) = 1.0 everywhere. Layout-agnostic
    # data, so the same buffer can be (a) prepacked for hpc and (b) reshaped /
    # reused as the swizzled view that flashinfer's cuDNN backend reads.
    sfx = torch.full((m_total, k // SF_VEC), 127, dtype=torch.uint8, device=device)
    sfw = torch.full((num_group, n, k // SF_VEC), 127, dtype=torch.uint8, device=device)
    return seqlens, cu_seqlens, m_total, x_fp8, w_fp8, sfx, sfw


def _build_hpc(num_group, seq, n, k, device):
    """Returns (run_fn, m_total) or (None, m_total) if hpc cannot run this shape."""
    if k % SF_VEC != 0:
        return None, None
    seqlens, cu_seqlens, m_total, x_fp8, w_fp8, sfx, sfw = _build_common(
        num_group, seq, n, k, device
    )

    sfx_packed, sfw_packed = hpc.prepack_mxfp8_scale(
        sfx, sfw, cu_seqlens, num_seq_per_group_avg=seq
    )

    def run():
        return hpc.group_gemm_mxfp8(
            x_fp8,
            w_fp8,
            sfx_packed,
            sfw_packed,
            seqlens,
            cu_seqlens,
            num_seq_per_group_avg=seq,
        )

    return run, m_total


def _build_fi(num_group, seq, n, k, device):
    """Returns (run_fn, m_total) or (None, m_total) if flashinfer cannot run this shape.

    flashinfer's cuDNN backend requires k/32 to be a multiple of 4 (128x4
    swizzle). All scale bytes are uniform 127 here, so reusing the plain
    (M, K/32) buffer in place of the swizzled view is layout-agnostic.
    """
    if (k // SF_VEC) % 4 != 0:
        return None, None
    _, cu_seqlens, m_total, x_fp8, w_fp8, sfx, sfw = _build_common(num_group, seq, n, k, device)

    def run():
        return grouped_mm_mxfp8(
            x_fp8,
            w_fp8,
            sfx,
            sfw,
            cu_seqlens,
            out_dtype=torch.bfloat16,
        )

    return run, m_total


# ---------------------------- timing ----------------------------


def _bench(label: str, fn) -> float:
    """Returns mean per-call latency in microseconds."""
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
    elapsed_ms = start.elapsed_time(end)
    return elapsed_ms * 1e3 / ITER  # us


def _is_finite(fn) -> bool:
    """One-shot finite check on the kernel output."""
    return fn().isfinite().all().item()


# ---------------------------- main ----------------------------


def main():
    device = torch.device("cuda:0")
    torch.manual_seed(2026)

    # results[shape_label][TM] = {hpc_us, fi_us, fi_finite}
    results = {label: {} for label, _, _ in SHAPES}

    for shape_label, n, k in SHAPES:
        print(f"\n=== shape={shape_label}  n={n}  k={k} ===", flush=True)
        for tm in TM_CASES:
            m = tm * NUM_GROUP

            run_hpc, _ = _build_hpc(NUM_GROUP, tm, n, k, device)
            t_hpc = _bench(f"hpc/{shape_label}/TM{tm}", run_hpc) if run_hpc else None

            run_fi, _ = _build_fi(NUM_GROUP, tm, n, k, device)
            if run_fi is None:
                t_fi = None
                fi_finite = None
            else:
                fi_finite = _is_finite(run_fi)
                t_fi = _bench(f"fi/{shape_label}/TM{tm}", run_fi)

            results[shape_label][tm] = {
                "hpc_us": t_hpc,
                "fi_us": t_fi,
                "fi_finite": fi_finite,
            }

            hpc_str = "    N/A" if t_hpc is None else f"{t_hpc:7.2f}"
            if t_fi is None:
                fi_str = "    N/A"
            else:
                fi_str = f"{t_fi:7.2f}{' (NaN)' if not fi_finite else ''}"
            print(
                f"  TM={tm:<4d}  m={m:<6d}  hpc={hpc_str} us   fi={fi_str} us",
                flush=True,
            )

    # ---- 2D summary (rows = TM, columns = shape x {hpc, fi, speedup}) ----
    print()
    print("### tp8")

    cell_w = 14
    header1 = ["       "]
    header2 = ["       "]
    sep = ["------"]
    for shape_label, _, _ in SHAPES:
        header1 += [shape_label, "", ""]
        header2 += ["hpc(us)", "fi(us)", "fi/hpc"]
        sep += ["------", "------", "------"]
    print("| " + " | ".join(c.ljust(cell_w) for c in header1) + " |")
    print("| " + " | ".join(c.ljust(cell_w) for c in header2) + " |")
    print("|-" + "-|-".join(c.ljust(cell_w, "-") for c in sep) + "-|")

    for tm in TM_CASES:
        cells = [f"TM={tm}"]
        for shape_label, _, _ in SHAPES:
            r = results[shape_label][tm]
            t_hpc = r["hpc_us"]
            t_fi = r["fi_us"]
            hpc_cell = "N/A" if t_hpc is None else f"{t_hpc:.2f}"
            if t_fi is None:
                cells += [hpc_cell, "N/A", "N/A"]
            else:
                fi_tag = " (NaN)" if not r["fi_finite"] else ""
                speedup = "N/A" if t_hpc is None else f"{t_fi / t_hpc:.2f}x"
                cells += [hpc_cell, f"{t_fi:.2f}{fi_tag}", speedup]
        print("| " + " | ".join(c.ljust(cell_w) for c in cells) + " |")

    # ---- per-shape geometric-mean speedup ----
    print()
    print("Geometric-mean speedup (fi / hpc) over TM cells where both kernels run:")
    for shape_label, _, _ in SHAPES:
        speeds = [
            r["fi_us"] / r["hpc_us"]
            for r in results[shape_label].values()
            if r["fi_us"] is not None and r["hpc_us"] is not None
        ]
        if not speeds:
            print(f"  {shape_label:14s}: N/A (flashinfer not supported on this shape)")
            continue
        gm = math.exp(sum(math.log(s) for s in speeds) / len(speeds))
        # Note: cells tagged (NaN) are still latency-only points.
        nan_count = sum(
            1
            for r in results[shape_label].values()
            if r["fi_finite"] is False and r["hpc_us"] is not None
        )
        nan_note = f"  ({nan_count} cells flashinfer=NaN)" if nan_count else ""
        print(f"  {shape_label:14s}: fi/hpc = {gm:.3f}x  over {len(speeds)} TM cells{nan_note}")


if __name__ == "__main__":
    main()
