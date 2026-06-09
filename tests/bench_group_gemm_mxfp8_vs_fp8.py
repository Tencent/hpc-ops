import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))


import torch
import torch.cuda.nvtx as nvtx

import hpc


NUM_EXPERT_TOTAL = 256
NUM_TOPK = 8

# Total token counts (m = num_seq) to sweep.
M_CASES = [256, 512, 1024, 2048, 4096, 8192]

# hy4.0
# (label, n, k, num_group)
SHAPES = [
    ("gate_up_tp8", 512, 6144, 256),
    ("down_tp8", 6144, 256, 256),
    ("gate_up_tp4", 1024, 6144, 256),
    ("down_tp4", 6144, 512, 256),
    ("gate_up_ep8", 4096, 6144, 32),
    ("down_ep8", 6144, 2048, 32),
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


def _build_mxfp4(num_group, seq, n, k, device):
    # mxfp8 activation x mxfp4 weight. Same kernel/binding as mxfp8: the
    # binding auto-detects fp4 from weight.size(2) == k/2. Requires k % 32 == 0.
    if k % SF_VEC != 0:
        return None  # not supported

    seqlens, cu_seqlens, x_fp8, _ = _build_common(num_group, seq, n, k, device)

    # W: fp4 (e2m1) packed two-per-byte → [num_group, n, k/2] uint8.
    lo = torch.randint(0, 16, (num_group, n, k // 2), dtype=torch.uint8, device=device)
    hi = torch.randint(0, 16, (num_group, n, k // 2), dtype=torch.uint8, device=device)
    w_fp4_packed = ((hi << 4) | lo).contiguous()

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
            w_fp4_packed,
            sfx_packed,
            sfw_packed,
            seqlens,
            cu_seqlens,
            num_seq_per_group_avg=seq,
        )

    return run


def _bench(label: str, fn) -> float:
    """Returns mean per-call latency in microseconds (CUDA graph when possible)."""
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()

    # Try CUDA graph capture; fall back to eager if capture fails.
    graph = None
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        torch.cuda.synchronize()
    except Exception:
        graph = None
        torch.cuda.synchronize()

    nvtx.range_push(label)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITER):
        if graph is not None:
            graph.replay()
        else:
            fn()
    end.record()
    torch.cuda.synchronize()
    nvtx.range_pop()
    elapsed_ms = start.elapsed_time(end)

    # Release graph to avoid interfering with subsequent captures from other libraries.
    del graph
    torch.cuda.synchronize()

    return elapsed_ms * 1e3 / ITER  # us


def main():
    device = torch.device("cuda:0")
    torch.manual_seed(2026)

    # results[shape_label][num_seq] = (mxfp8_us, mxfp4_us, fp8_us)
    results = {label: {} for label, _, _, _ in SHAPES}

    for shape_label, n, k, num_group in SHAPES:
        print(f"\n=== shape={shape_label}  n={n}  k={k}  num_group={num_group} ===", flush=True)
        for num_seq in M_CASES:
            # seq = tokens per group (expert) in the group_gemm
            seq = num_seq * NUM_TOPK // NUM_EXPERT_TOTAL

            # mxfp8
            run_mx = _build_mxfp8(num_group, seq, n, k, device)
            if run_mx is None:
                t_mx = None
            else:
                t_mx = _bench(f"mxfp8/{shape_label}/m{num_seq}", run_mx)

            # mxfp8 x mxfp4
            run_mx4 = _build_mxfp4(num_group, seq, n, k, device)
            if run_mx4 is None:
                t_mx4 = None
            else:
                t_mx4 = _bench(f"mxfp8_mxfp4/{shape_label}/m{num_seq}", run_mx4)

            # fp8 per-tensor
            run_fp = _build_fp8(num_group, seq, n, k, device)
            t_fp = _bench(f"fp8/{shape_label}/m{num_seq}", run_fp)

            results[shape_label][num_seq] = (t_mx, t_mx4, t_fp)
            mx_str = f"{t_mx:7.2f}" if t_mx is not None else "    N/A"
            mx4_str = f"{t_mx4:7.2f}" if t_mx4 is not None else "    N/A"
            print(
                f"  m={num_seq:>5d}  seq/grp={seq:>4d}  "
                f"mxfp8={mx_str} us   mxfp8_mxfp4={mx4_str} us   fp8={t_fp:7.2f} us",
                flush=True,
            )

    # Markdown summary tables (one table per shape to keep each narrow).
    for shape_label, _, _, num_group in SHAPES:
        print()
        print(f"### {shape_label}")
        header = ["m", "seq/grp", "mxfp8", "mxfp8_mxfp4", "fp8", "mxfp8_sp", "mxfp4_sp"]
        sep = ["------"] * len(header)
        print("| " + " | ".join(c.ljust(12) for c in header) + " |")
        print("|-" + "-|-".join(c.ljust(12, "-") for c in sep) + "-|")
        for num_seq in M_CASES:
            seq = num_seq * NUM_TOPK // NUM_EXPERT_TOTAL
            t_mx, t_mx4, t_fp = results[shape_label][num_seq]
            mx_cell = f"{t_mx:.2f}" if t_mx is not None else "N/A"
            mx4_cell = f"{t_mx4:.2f}" if t_mx4 is not None else "N/A"
            mx_sp = f"{t_fp / t_mx:.2f}" if t_mx is not None else "N/A"
            mx4_sp = f"{t_fp / t_mx4:.2f}" if t_mx4 is not None else "N/A"
            cells = [str(num_seq), str(seq), mx_cell, mx4_cell, f"{t_fp:.2f}", mx_sp, mx4_sp]
            print("| " + " | ".join(c.ljust(12) for c in cells) + " |")


if __name__ == "__main__":
    main()
