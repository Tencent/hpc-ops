"""Benchmark: hpc.fuse_moe_mxfp8 (mxfp4 weight) vs flashinfer.cutlass_fused_moe (mxfp8 act + mxfp4 weight).

Both run the same fuse_moe workload: mxfp8 activation x mxfp4 weight, SwiGLU, topk reduce.
Routing is pre-computed and shared; only expert computation latency is measured.

Usage:
    python tests/bench_fuse_moe_mxfp4_hpc_vs_flashinfer.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import torch
import torch.cuda.nvtx as nvtx

import hpc
import flashinfer.fused_moe as fused_moe
from flashinfer import ActivationType, mxfp8_quantize, fp4_quantize
from flashinfer.fused_moe import (
    RoutingMethodType,
    trtllm_fp4_block_scale_routed_moe,
    trtllm_fp8_per_tensor_scale_moe,
)
from flashinfer.utils import device_support_pdl


# ---------------------------- config ----------------------------

NUM_EXPERT_TOTAL = 256
NUM_TOPK = 8

# Total token counts (m = num_seq) to sweep.
M_CASES = [256, 320, 384, 448, 512, 1024, 16384, 32768]
# M_CASES = [512]
# (label, hidden, intermediate_size, num_expert_local, rank_ep)
SHAPES = [
    # ("h6144_tp8", 6144, 256, 256, 0),
    # ("h6144_tp4", 6144, 512, 256, 0),
    ("h6144_ep8", 6144, 2048, 32, 0),
    # ("h4096_tp8", 4096, 192, 192, 0),
    # ("h4096_tp4", 4096, 384, 192, 0),
    # ("h4096_ep8", 4096, 1536, 24, 0),
]

SF_VEC = 32
WARMUP = 1
ITER = 10


# ---------------------------- helpers ----------------------------


def _prepack_weight_scale(sfw, avg):
    """Offline weight-side SF prepack for hpc."""
    _, sfw_packed = hpc.prepack_mxfp8_scale(None, sfw, None, num_seq_per_group_avg=avg)
    return sfw_packed


def _build_routing(num_seq, num_expert_local, rank_ep, device):
    """Build strictly uniform routing across ALL experts (simulates real EP).

    Returns topk_ids and topk_scale shared by both HPC and FlashInfer backends.
    Tokens are uniformly distributed over all NUM_EXPERT_TOTAL experts via modulo.
    Only ids in [rank_ep*num_expert_local, (rank_ep+1)*num_expert_local) are
    processed locally — matching the original EP semantics where each rank only
    handles its local shard (1/EP of total work).
    """
    total_slots = num_seq * NUM_TOPK
    flat = torch.arange(total_slots, device=device, dtype=torch.int32) % NUM_EXPERT_TOTAL
    topk_ids = flat.reshape(num_seq, NUM_TOPK).contiguous()

    topk_scale = torch.rand((num_seq, NUM_TOPK), device=device, dtype=torch.float32) * 0.5 + 0.5

    return topk_ids, topk_scale


# ---------------------------- hpc builder ----------------------------


def _build_hpc_mxfp8_mxfp4(hidden, inter, num_expert_local, rank_ep, num_seq, avg, device):
    """hpc.fuse_moe_mxfp8 with mxfp4 weights (auto-detected from uint8 + K/2 dim)."""
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

    shared_output = torch.randn((num_seq, hidden), dtype=torch.bfloat16, device=device)

    def run(topk_ids, topk_scale):
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
            shared_output=shared_output,
        )

    return run


def _build_hpc_fp8(hidden, inter, num_expert_local, rank_ep, num_seq, device):
    """hpc.fuse_moe with fp8 per-tensor quantization."""
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

    shared_output = torch.randn((num_seq, hidden), dtype=torch.bfloat16, device=device)

    def run(topk_ids, topk_scale):
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
            shared_output=shared_output,
        )

    return run


# ---------------------------- flashinfer builder ----------------------------


# ---------------------------- flashinfer trtllm builder ----------------------------


def _build_flashinfer_trtllm_mxfp4(
    hidden, inter, num_expert_local, rank_ep, num_seq, topk_ids, topk_scale, device
):
    """flashinfer.trtllm_fp4_block_scale_routed_moe with mxfp8 activation + mxfp4 weight.

    Uses the routed API that directly accepts topk_ids (no internal routing).
    """
    if hidden % 128 != 0 or inter % 128 != 0:
        return None
    gate_up_n = inter * 2
    local_expert_offset = rank_ep * num_expert_local
    enable_pdl = device_support_pdl(device)

    # Generate bf16 source tensors then quantize.
    x_bf16 = torch.randn((num_seq, hidden), dtype=torch.bfloat16, device=device) / 100
    w1_bf16 = (
        torch.randn((num_expert_local, gate_up_n, hidden), dtype=torch.bfloat16, device=device) / 10
    )
    w2_bf16 = (
        torch.randn((num_expert_local, hidden, inter), dtype=torch.bfloat16, device=device) / 10
    )

    # Quantize activation: mxfp8 (linear layout for trtllm)
    mxfp8_x, mxfp8_x_sf = mxfp8_quantize(x_bf16, False)
    mxfp8_x_sf = mxfp8_x_sf.view(torch.float8_e4m3fn).reshape(num_seq, -1)

    # Quantize weights: mxfp4 via fp4_quantize (sf_vec_size=32, sf_use_ue8m0=True)
    w1_fp4, w1_sf = fp4_quantize(
        w1_bf16, torch.tensor([1.0], device=device), sf_vec_size=32, sf_use_ue8m0=True
    )
    w1_sf = w1_sf.view(torch.float8_e4m3fn).reshape(num_expert_local, gate_up_n, -1)

    w2_fp4, w2_sf = fp4_quantize(
        w2_bf16, torch.tensor([1.0], device=device), sf_vec_size=32, sf_use_ue8m0=True
    )
    w2_sf = w2_sf.view(torch.float8_e4m3fn).reshape(num_expert_local, hidden, -1)

    # Output scale scalars (all 1.0 since global_scale=1.0)
    output_scale = torch.ones(num_expert_local, device=device, dtype=torch.float32)

    # topk_weights in bf16 for routed API
    topk_weights = topk_scale.to(torch.bfloat16)

    # Pre-allocate output buffer
    output = torch.empty(num_seq, hidden, device=device, dtype=torch.bfloat16)

    def run():
        return trtllm_fp4_block_scale_routed_moe(
            topk_ids=(topk_ids, topk_weights),
            routing_bias=None,
            hidden_states=mxfp8_x,
            hidden_states_scale=mxfp8_x_sf,
            gemm1_weights=w1_fp4,
            gemm1_weights_scale=w1_sf,
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=w2_fp4,
            gemm2_weights_scale=w2_sf,
            gemm2_bias=None,
            output1_scale_scalar=output_scale,
            output1_scale_gate_scalar=output_scale,
            output2_scale_scalar=output_scale,
            num_experts=NUM_EXPERT_TOTAL,
            top_k=NUM_TOPK,
            n_group=None,
            topk_group=None,
            intermediate_size=inter,
            local_expert_offset=local_expert_offset,
            local_num_experts=num_expert_local,
            routed_scaling_factor=None,
            routing_method_type=RoutingMethodType.Renormalize.value,
            do_finalize=True,
            enable_pdl=enable_pdl,
            activation_type=ActivationType.Swiglu.value,
            per_token_scale=None,
            output=output,
            tune_max_num_tokens=num_seq,
        )

    # Trigger JIT/cubin load eagerly so failures are caught here rather than in _bench.
    try:
        run()
        torch.cuda.synchronize()
    except Exception:
        return None

    return run


def _build_flashinfer_trtllm_fp8(
    hidden, inter, num_expert_local, rank_ep, num_seq, topk_ids, topk_scale, device
):
    """flashinfer.trtllm_fp8_per_tensor_scale_moe (fp8 per-tensor, includes routing).

    Since there is no routed variant for fp8 per-tensor, we construct routing_logits
    from topk_ids such that torch.topk reproduces the same routing.
    Note: this API includes routing overhead in the timing.
    """
    gate_up_n = inter * 2
    local_expert_offset = rank_ep * num_expert_local

    # FP8 per-tensor: cast hidden_states and weights to fp8
    x_bf16 = torch.randn((num_seq, hidden), dtype=torch.bfloat16, device=device) / 100
    x_fp8 = x_bf16.to(torch.float8_e4m3fn)

    w1_bf16 = (
        torch.randn((num_expert_local, gate_up_n, hidden), dtype=torch.bfloat16, device=device) / 10
    )
    w2_bf16 = (
        torch.randn((num_expert_local, hidden, inter), dtype=torch.bfloat16, device=device) / 10
    )
    w1_fp8 = w1_bf16.to(torch.float8_e4m3fn)
    w2_fp8 = w2_bf16.to(torch.float8_e4m3fn)

    # Per-expert output scales
    output1_scale = torch.full((num_expert_local,), 0.25, dtype=torch.float32, device=device)
    output1_gate_scale = torch.full((num_expert_local,), 0.25, dtype=torch.float32, device=device)
    output2_scale = torch.full((num_expert_local,), 0.25, dtype=torch.float32, device=device)

    # Construct routing_logits from topk_ids so that topk(logits) == topk_ids.
    # Use large spread: selected experts get high values, unselected get -1e4.
    routing_logits = torch.full(
        (num_seq, NUM_EXPERT_TOTAL), -1e4, device=device, dtype=torch.float32
    )
    # Assign decreasing values K, K-1, ..., 1 to ensure ordering matches topk_ids
    values = (
        torch.arange(NUM_TOPK, 0, -1, device=device, dtype=torch.float32)
        .unsqueeze(0)
        .expand(num_seq, -1)
    )
    routing_logits.scatter_(1, topk_ids.long(), values)

    def run():
        return trtllm_fp8_per_tensor_scale_moe(
            routing_logits,
            None,  # routing_bias
            x_fp8,
            w1_fp8,
            output1_scale,
            output1_gate_scale,
            w2_fp8,
            output2_scale,
            NUM_EXPERT_TOTAL,
            NUM_TOPK,
            None,  # n_group
            None,  # topk_group
            inter,
            local_expert_offset,  # local_expert_offset
            num_expert_local,
            None,  # routed_scaling_factor
            False,  # use_routing_scales_on_input
        )

    # Trigger JIT/cubin load eagerly
    try:
        run()
        torch.cuda.synchronize()
    except Exception:
        return None

    return run


# ---------------------------- timing ----------------------------


def _bench(label: str, fn) -> float:
    """Returns mean per-call latency in microseconds (CUDA graph when possible)."""
    # Warmup (also triggers flashinfer autotuning)
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

    # Timed loop
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


# ---------------------------- main ----------------------------


def main():
    device = torch.device("cuda:0")
    torch.manual_seed(2026)

    print(
        f"NUM_EXPERT_TOTAL={NUM_EXPERT_TOTAL}  NUM_TOPK={NUM_TOPK}  "
        f"WARMUP={WARMUP}  ITER={ITER}"
    )

    # results[shape_label][num_seq] = (hpc_mxfp8_mxfp4_us, hpc_fp8_us, trtllm_mxfp4_us, trtllm_fp8_us)
    results = {label: {} for label, _, _, _, _ in SHAPES}

    for shape_label, hidden, inter, num_expert_local, rank_ep in SHAPES:
        print(
            f"\n=== shape={shape_label}  hidden={hidden}  intermediate={inter}  "
            f"num_expert_local={num_expert_local}/{NUM_EXPERT_TOTAL}  "
            f"topk={NUM_TOPK}  rank_ep={rank_ep} ===",
            flush=True,
        )
        for num_seq in M_CASES:
            # num_seq_per_group = avg tokens per expert (for prepack kTileM).
            num_seq_per_group = num_seq * NUM_TOPK // NUM_EXPERT_TOTAL

            # Build shared routing (uniform distribution, same topk_ids for all backends)
            topk_ids, topk_scale = _build_routing(num_seq, num_expert_local, rank_ep, device)

            # hpc mxfp8_mxfp4
            run_hpc = _build_hpc_mxfp8_mxfp4(
                hidden, inter, num_expert_local, rank_ep, num_seq, num_seq_per_group, device
            )
            if run_hpc is None:
                t_hpc = None
            else:
                t_hpc = _bench(
                    f"hpc_mxfp8_mxfp4/{shape_label}/n{num_seq}",
                    lambda: run_hpc(topk_ids, topk_scale),
                )

            # hpc fp8 per-tensor
            run_fp8 = _build_hpc_fp8(hidden, inter, num_expert_local, rank_ep, num_seq, device)
            t_fp8 = _bench(
                f"hpc_fp8/{shape_label}/n{num_seq}",
                lambda: run_fp8(topk_ids, topk_scale),
            )

            # flashinfer trtllm mxfp4 (pre-routed, same topk_ids)
            run_trtllm_mxfp4 = _build_flashinfer_trtllm_mxfp4(
                hidden, inter, num_expert_local, rank_ep, num_seq, topk_ids, topk_scale, device
            )
            if run_trtllm_mxfp4 is None:
                t_trtllm_mxfp4 = None
            else:
                t_trtllm_mxfp4 = _bench(
                    f"fi_trtllm_mxfp4/{shape_label}/n{num_seq}", run_trtllm_mxfp4
                )

            # flashinfer trtllm fp8 block-scale (pre-routed, same topk_ids)
            run_trtllm_fp8 = _build_flashinfer_trtllm_fp8(
                hidden, inter, num_expert_local, rank_ep, num_seq, topk_ids, topk_scale, device
            )
            if run_trtllm_fp8 is None:
                t_trtllm_fp8 = None
            else:
                t_trtllm_fp8 = _bench(f"fi_trtllm_fp8/{shape_label}/n{num_seq}", run_trtllm_fp8)

            results[shape_label][num_seq] = (t_hpc, t_fp8, t_trtllm_mxfp4, t_trtllm_fp8)

            def _f(t):
                return f"{t:7.2f}" if t is not None else "    N/A"

            def _sp(base, t):
                if base is not None and t is not None:
                    return f"{t / base:.2f}x"
                return "N/A"

            print(
                f"  m={num_seq:>6d}  seq/grp={num_seq_per_group:>4d}  "
                f"hpc_mxfp8_mxfp4={_f(t_hpc)}  hpc_fp8={_f(t_fp8)}  "
                f"fi_mxfp8_mxfp4={_f(t_trtllm_mxfp4)}  fi_fp8={_f(t_trtllm_fp8)}",
                flush=True,
            )

    # Markdown summary tables
    for shape_label, hidden, inter, num_expert_local, rank_ep in SHAPES:
        print()
        print(f"### {shape_label} (hidden={hidden}, inter={inter}, E_local={num_expert_local})")
        print()
        print(
            f"| {'m':>5} | {'seq/grp':>7} | {'hpc_mxfp8_mxfp4':>15} | {'hpc_fp8':>9} | {'fi_mxfp8_mxfp4':>15} | {'fi_fp8':>9} | {'fi_mxfp8_mxfp4/hpc':>18} | {'fi_fp8/hpc':>11} |"
        )
        print(f"|{'-'*6}:|{'-'*8}:|{'-'*16}:|{'-'*10}:|{'-'*16}:|{'-'*10}:|{'-'*19}:|{'-'*12}:|")
        for num_seq in M_CASES:
            num_seq_per_group = num_seq * NUM_TOPK // NUM_EXPERT_TOTAL
            t_hpc, t_fp8, t_trtllm_mxfp4, t_trtllm_fp8 = results[shape_label][num_seq]

            def _c(t):
                return f"{t:.2f}" if t is not None else "N/A"

            def _s(base, t):
                if base is not None and t is not None:
                    return f"{t / base:.2f}x"
                return "N/A"

            print(
                f"| {num_seq:>5d} | {num_seq_per_group:>7d} | {_c(t_hpc):>15} | "
                f"{_c(t_fp8):>9} | {_c(t_trtllm_mxfp4):>15} | {_c(t_trtllm_fp8):>9} | "
                f"{_s(t_hpc, t_trtllm_mxfp4):>18} | {_s(t_fp8, t_trtllm_fp8):>11} |"
            )


if __name__ == "__main__":
    main()
