"""Bench mxfp8_group_gemm vs fp8_group_gemm.
NUM_GROUP = 192
tp8 gate_up: (m = TM * 192 / 8,  n = 384,   k = 4096)
tp8 down:    (m = TM * 192 / 8,  n = 4096,  k = 192)
tp4 gate_up: (m = TM * 192 / 8,  n = 768,   k = 4096)
tp4 down:    (m = TM * 192 / 8,  n = 4096,  k = 384)

"""

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


NUM_GROUP = 192
TM_CASES = [8, 16, 32, 64, 128, 256]

SHAPES = [
    ("gate_up_tp8", 384, 4096),
    ("down_tp8", 4096, 192),
    ("gate_up_tp4", 768, 4096),
    ("down_tp4", 4096, 384),
]

SF_VEC = 32
WARMUP = 5
ITER = 100


# ---------------------------- builders ----------------------------


def _build_common(num_group: int, seq: int, n: int, k: int, device):
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
    return seqlens, cu_seqlens, x_fp8, w_fp8


def _build_fp8(num_group, seq, n, k, device):
    seqlens, cu_seqlens, x_fp8, w_fp8 = _build_common(num_group, seq, n, k, device)
    y_scale = torch.full((num_group,), 1.0, dtype=torch.float32, device=device)

    def run():
        return hpc.group_gemm_fp8(
            x_fp8,
            w_fp8,
            seqlens,
            cu_seqlens,
            y_scale,
            num_seq_per_group_avg=seq,
        )

    return run


def _build_mxfp8(num_group, seq, n, k, device):
    # kernel only requires k % SF_VEC (=32) == 0; trailing partial 128-K SF
    # tile is zero-padded by prepack and OOB-loaded by TMA, so k=192 etc. are
    # valid (test.md's older "k % 128 == 0" requirement is stale).
    if k % SF_VEC != 0:
        return None  # not supported

    seqlens, cu_seqlens, x_fp8, w_fp8 = _build_common(num_group, seq, n, k, device)

    sfx = torch.full(
        (int(cu_seqlens[-1].item()), k // SF_VEC), 127, dtype=torch.uint8, device=device
    )
    sfw = torch.full((num_group, n, k // SF_VEC), 127, dtype=torch.uint8, device=device)

    sfx_packed, sfw_packed = hpc.prepack_mxfp8_scale(
        sfx,
        sfw,
        cu_seqlens,
        num_seq_per_group_avg=seq,
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

    return run


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


def main():
    device = torch.device("cuda:0")
    torch.manual_seed(2026)

    # results[shape_label][TM] = (mxfp8_us, fp8_us)
    results = {label: {} for label, _, _ in SHAPES}

    for shape_label, n, k in SHAPES:
        print(f"\n=== shape={shape_label}  n={n}  k={k} ===", flush=True)
        for tm in TM_CASES:
            m = tm * NUM_GROUP // 8

            # mxfp8
            run_mx = _build_mxfp8(NUM_GROUP, tm, n, k, device)
            if run_mx is None:
                t_mx = None
            else:
                lab = f"mxfp8/{shape_label}/m{m}"
                t_mx = _bench(lab, run_mx)

            # fp8 per-tensor
            run_fp = _build_fp8(NUM_GROUP, tm, n, k, device)
            lab = f"fp8/{shape_label}/m{m}"
            t_fp = _bench(lab, run_fp)

            results[shape_label][tm] = (t_mx, t_fp)
            mx_str = f"{t_mx:7.2f}" if t_mx is not None else "    N/A"
            print(
                f"  m={m:>5d}  seq_per_group={tm:>3d}  mxfp8={mx_str} us   fp8={t_fp:7.2f} us",
                flush=True,
            )

    # Markdown summary tables (one per tp config to keep each narrow).
    tp_groups = {}
    for shape_label, _, _ in SHAPES:
        # split off trailing "_tpN" suffix to group; fall back to single group otherwise
        tp = shape_label.rsplit("_", 1)[-1] if "_tp" in shape_label else "all"
        tp_groups.setdefault(tp, []).append(shape_label)

    for tp, labels in tp_groups.items():
        print()
        print(f"### {tp}")
        header1 = ["m     ", "seq_per_grp"]
        header2 = ["      ", "           "]
        sep = ["------", "------"]
        for shape_label in labels:
            header1 += [shape_label, "", ""]
            header2 += ["mxfp8", "fp8_per_tensor", "speedup"]
            sep += ["------", "------", "------"]
        print("| " + " | ".join(c.ljust(14) for c in header1) + " |")
        print("| " + " | ".join(c.ljust(14) for c in header2) + " |")
        print("|-" + "-|-".join(c.ljust(14, "-") for c in sep) + "-|")
        for tm in TM_CASES:
            m = tm * NUM_GROUP // 8
            cells = [str(m), str(tm)]
            for shape_label in labels:
                t_mx, t_fp = results[shape_label][tm]
                if t_mx is None:
                    cells += ["N/A", f"{t_fp:.2f}", "N/A"]
                else:
                    speedup = t_fp / t_mx
                    cells += [f"{t_mx:.2f}", f"{t_fp:.2f}", f"{speedup:.2f}"]
            print("| " + " | ".join(c.ljust(14) for c in cells) + " |")


if __name__ == "__main__":
    main()
