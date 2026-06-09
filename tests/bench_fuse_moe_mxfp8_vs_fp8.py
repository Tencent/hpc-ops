import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))


import torch
import torch.cuda.nvtx as nvtx

import hpc


# Routing config (kept constant across shapes).
NUM_EXPERT_TOTAL = 256
NUM_TOPK = 8

# Total token counts (m = num_seq) to sweep.
M_CASES = [256, 512, 1024, 2048, 4096, 8192]

# hy4.0
# (label, hidden, intermediate_size, num_expert_local, rank_ep)
SHAPES = [
    ("tp8", 6144, 256, 256, 0),
    ("tp4", 6144, 512, 256, 0),
    ("ep8", 6144, 2048, 32, 0),
]

SF_VEC = 32
FP8_MAX = 448.0
WARMUP = 5
ITER = 100


# ---------------------------- helpers ----------------------------


def _prepack_weight_scale(sfw, avg):
    """Offline weight-side SF prepack (kTileM-independent on the SFB side)."""
    _, sfw_packed = hpc.prepack_mxfp8_scale(None, sfw, None, num_seq_per_group_avg=avg)
    return sfw_packed


def _build_routing(num_seq, num_expert_local, rank_ep, device):
    """Build routing tensors (topk_ids uniformly over ALL experts).

    Real EP: tokens are uniformly routed across all NUM_EXPERT_TOTAL experts.
    Only ids in [rank_ep*E_local, (rank_ep+1)*E_local) are processed locally.
    """
    topk_ids = torch.randint(
        0,
        NUM_EXPERT_TOTAL,
        (num_seq, NUM_TOPK),
        device=device,
        dtype=torch.int32,
    )
    topk_scale = torch.rand((num_seq, NUM_TOPK), device=device, dtype=torch.float32) * 0.5 + 0.5
    return topk_ids, topk_scale


# ---------------------------- builders ----------------------------


def _build_fp8(hidden, inter, num_expert_local, rank_ep, num_seq, device):
    """fp8 per-tensor fuse_moe."""
    gate_up_n = inter * 2

    x = (torch.randn((num_seq, hidden), dtype=torch.float32, device=device) / 100).to(
        torch.float8_e4m3fn
    )
    gate_up_w = torch.randn(
        (num_expert_local, gate_up_n, hidden), dtype=torch.float32, device=device
    ).to(torch.float8_e4m3fn)
    down_w = torch.randn((num_expert_local, hidden, inter), dtype=torch.float32, device=device).to(
        torch.float8_e4m3fn
    )

    gate_up_scale = torch.full((num_expert_local,), 0.25, dtype=torch.float32, device=device)
    down_scale = torch.full((num_expert_local,), 0.25, dtype=torch.float32, device=device)
    act_and_mul_scale = torch.full((1,), 1.0, dtype=torch.float32, device=device)

    topk_ids, topk_scale = _build_routing(num_seq, num_expert_local, rank_ep, device)

    def run():
        return hpc.fuse_moe(
            x,
            gate_up_w,
            down_w,
            gate_up_scale,
            down_scale,
            act_and_mul_scale,
            topk_ids,
            topk_scale,
            rank_ep,
            NUM_EXPERT_TOTAL,
        )

    return run


def _build_mxfp8(hidden, inter, num_expert_local, rank_ep, num_seq, avg, device):
    """mxfp8 activation x mxfp8 weight fuse_moe."""
    if hidden % SF_VEC != 0 or inter % SF_VEC != 0:
        return None
    gate_up_n = inter * 2

    x = (torch.randn((num_seq, hidden), dtype=torch.float32, device=device) / 100).to(
        torch.float8_e4m3fn
    )
    x_scale = torch.full((num_seq, hidden // SF_VEC), 127, dtype=torch.uint8, device=device)

    gate_up_w = torch.randn(
        (num_expert_local, gate_up_n, hidden), dtype=torch.float32, device=device
    ).to(torch.float8_e4m3fn)
    down_w = torch.randn((num_expert_local, hidden, inter), dtype=torch.float32, device=device).to(
        torch.float8_e4m3fn
    )

    gate_up_w_scale = torch.full(
        (num_expert_local, gate_up_n, hidden // SF_VEC), 127, dtype=torch.uint8, device=device
    )
    down_w_scale = torch.full(
        (num_expert_local, hidden, inter // SF_VEC), 127, dtype=torch.uint8, device=device
    )
    gate_up_w_scale_packed = _prepack_weight_scale(gate_up_w_scale, avg)
    down_w_scale_packed = _prepack_weight_scale(down_w_scale, avg)

    topk_ids, topk_scale = _build_routing(num_seq, num_expert_local, rank_ep, device)

    def run():
        return hpc.fuse_moe_mxfp8(
            x,
            x_scale,
            gate_up_w,
            gate_up_w_scale_packed,
            down_w,
            down_w_scale_packed,
            topk_ids,
            topk_scale,
            rank_ep=rank_ep,
            num_expert_total=NUM_EXPERT_TOTAL,
        )

    return run


def _build_mxfp4(hidden, inter, num_expert_local, rank_ep, num_seq, avg, device):
    """mxfp8 activation x mxfp4 (e2m1) weight fuse_moe.

    Same binding as mxfp8: fp4 is auto-detected from the uint8 packed weights
    (last dim == K/2). Requires hidden % 128 == 0 and intermediate % 128 == 0
    (sub-byte e2m1 TMA byte alignment).
    """
    if hidden % 128 != 0 or inter % 128 != 0:
        return None
    gate_up_n = inter * 2

    x = (torch.randn((num_seq, hidden), dtype=torch.float32, device=device) / 100).to(
        torch.float8_e4m3fn
    )
    x_scale = torch.full((num_seq, hidden // SF_VEC), 127, dtype=torch.uint8, device=device)

    # fp4 weights: e2m1 packed two-per-byte.
    gate_up_w = torch.randint(
        0, 256, (num_expert_local, gate_up_n, hidden // 2), dtype=torch.uint8, device=device
    )
    down_w = torch.randint(
        0, 256, (num_expert_local, hidden, inter // 2), dtype=torch.uint8, device=device
    )

    # SF spans the full (unpacked) K, one byte per 32 elements.
    gate_up_w_scale = torch.full(
        (num_expert_local, gate_up_n, hidden // SF_VEC), 127, dtype=torch.uint8, device=device
    )
    down_w_scale = torch.full(
        (num_expert_local, hidden, inter // SF_VEC), 127, dtype=torch.uint8, device=device
    )
    gate_up_w_scale_packed = _prepack_weight_scale(gate_up_w_scale, avg)
    down_w_scale_packed = _prepack_weight_scale(down_w_scale, avg)

    topk_ids, topk_scale = _build_routing(num_seq, num_expert_local, rank_ep, device)

    def run():
        return hpc.fuse_moe_mxfp8(
            x,
            x_scale,
            gate_up_w,
            gate_up_w_scale_packed,
            down_w,
            down_w_scale_packed,
            topk_ids,
            topk_scale,
            rank_ep=rank_ep,
            num_expert_total=NUM_EXPERT_TOTAL,
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

    # results[shape_label][num_seq] = (mxfp8_us, mxfp8_mxfp4_us, fp8_us)
    results = {label: {} for label, _, _, _, _ in SHAPES}

    for shape_label, hidden, inter, num_expert_local, rank_ep in SHAPES:
        print(
            f"\n=== shape={shape_label}  hidden={hidden}  intermediate={inter}  "
            f"num_expert_local={num_expert_local}/{NUM_EXPERT_TOTAL}  "
            f"topk={NUM_TOPK}  rank_ep={rank_ep} ===",
            flush=True,
        )
        for num_seq in M_CASES:
            avg = num_seq * NUM_TOPK // NUM_EXPERT_TOTAL

            # mxfp8
            run_mx = _build_mxfp8(hidden, inter, num_expert_local, rank_ep, num_seq, avg, device)
            if run_mx is None:
                t_mx = None
            else:
                t_mx = _bench(f"mxfp8/{shape_label}/n{num_seq}", run_mx)

            # mxfp8 x mxfp4
            run_mx4 = _build_mxfp4(hidden, inter, num_expert_local, rank_ep, num_seq, avg, device)
            if run_mx4 is None:
                t_mx4 = None
            else:
                t_mx4 = _bench(f"mxfp8_mxfp4/{shape_label}/n{num_seq}", run_mx4)

            # fp8 per-tensor
            run_fp = _build_fp8(hidden, inter, num_expert_local, rank_ep, num_seq, device)
            t_fp = _bench(f"fp8/{shape_label}/n{num_seq}", run_fp)

            results[shape_label][num_seq] = (t_mx, t_mx4, t_fp)
            mx_str = f"{t_mx:7.2f}" if t_mx is not None else "    N/A"
            mx4_str = f"{t_mx4:7.2f}" if t_mx4 is not None else "    N/A"
            print(
                f"  m={num_seq:>6d}  seq/grp={avg:>4d}  "
                f"mxfp8={mx_str} us   mxfp8_mxfp4={mx4_str} us   fp8={t_fp:7.2f} us",
                flush=True,
            )

    # Markdown summary tables (one table per shape to keep each narrow).
    for shape_label, _, _, num_expert_local, _ in SHAPES:
        print()
        print(f"### {shape_label}")
        header = ["m", "seq/grp", "mxfp8", "mxfp8_mxfp4", "fp8", "mxfp8_sp", "mxfp4_sp"]
        sep = ["------"] * len(header)
        print("| " + " | ".join(c.ljust(12) for c in header) + " |")
        print("|-" + "-|-".join(c.ljust(12, "-") for c in sep) + "-|")
        for num_seq in M_CASES:
            avg = num_seq * NUM_TOPK // NUM_EXPERT_TOTAL
            t_mx, t_mx4, t_fp = results[shape_label][num_seq]
            mx_cell = f"{t_mx:.2f}" if t_mx is not None else "N/A"
            mx4_cell = f"{t_mx4:.2f}" if t_mx4 is not None else "N/A"
            mx_sp = f"{t_fp / t_mx:.2f}" if t_mx is not None else "N/A"
            mx4_sp = f"{t_fp / t_mx4:.2f}" if t_mx4 is not None else "N/A"
            cells = [str(num_seq), str(avg), mx_cell, mx4_cell, f"{t_fp:.2f}", mx_sp, mx4_sp]
            print("| " + " | ".join(c.ljust(12) for c in cells) + " |")


if __name__ == "__main__":
    main()
