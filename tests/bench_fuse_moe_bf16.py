"""
Benchmark script for hpc.fuse_moe_bf16 vs sglang fused_moe (BF16).

Default model: Qwen3-235B-A22B, TP=8 (EP=8 simulation on a single GPU)
  hidden_size        = 4096
  moe_intermediate_size = 1536
  num_experts        = 128  ->  num_experts_local = 128 // 8 = 16 per GPU
  num_experts_per_tok = 8 (topk)

Usage:
    python tests/bench_fuse_moe_bf16.py
    python tests/bench_fuse_moe_bf16.py --hidden-size 4096 --intermediate-size 1536 \\
        --num-experts 128 --topk 8 --tp-size 8
    python tests/bench_fuse_moe_bf16.py --warmup 10 --iters 100
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import triton.language as tl
import torch
import hpc
from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_kernels import invoke_fused_moe_kernel
from sgl_kernel import moe_align_block_size as sgl_moe_align_block_size
from sgl_kernel import silu_and_mul

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden-size", type=int, default=4096)
    p.add_argument("--intermediate-size", type=int, default=1536,
                   help="moe_intermediate_size per expert (before gate/up split)")
    p.add_argument("--num-experts", type=int, default=128)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--tp-size", type=int, default=8,
                   help="Tensor/Expert parallelism size. num_experts_local = num_experts // tp_size")
    p.add_argument("--warmup", type=int, default=5,
                   help="Warmup iterations before graph capture")
    p.add_argument("--iters", type=int, default=100,
                   help="Graph replay iterations for timing")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Batch sizes to sweep
# ---------------------------------------------------------------------------

BATCH_SIZES = [i for i in range(1, 17)]


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------


def make_inputs(batch_size, hidden_size, intermediate_size, num_experts_local, topk,
                device="cuda"):
    """
    Build BF16 inputs for a single-GPU EP benchmark.
    topk_ids are sampled uniformly from local experts [0, num_experts_local).
    """
    dtype = torch.bfloat16

    x = torch.randn((batch_size, hidden_size), dtype=dtype, device=device)

    # gate+up fused: [E_local, inter*2, hidden]
    gate_up_weight = torch.randn(
        (num_experts_local, intermediate_size * 2, hidden_size), dtype=dtype, device=device
    )
    # down:          [E_local, hidden, inter]
    down_weight = torch.randn(
        (num_experts_local, hidden_size, intermediate_size), dtype=dtype, device=device
    )

    # topk_ids in [0, num_experts_local); topk_scale positive, sum-normalised
    topk_ids = torch.randint(
        0, num_experts_local, (batch_size, topk), dtype=torch.int32, device=device
    )
    raw_scale = torch.rand((batch_size, topk), dtype=torch.float32, device=device)
    topk_scale = raw_scale / raw_scale.sum(dim=1, keepdim=True)

    return x, gate_up_weight, down_weight, topk_ids, topk_scale


# ---------------------------------------------------------------------------
# FLOPs estimate (two GEMMs: gate_up and down, per token-expert assignment)
# ---------------------------------------------------------------------------


def tflops(batch_size, topk, hidden_size, intermediate_size, elapsed_ms):
    # gate_up:  batch*topk × (inter*2) × hidden  (factor 2 for gate+up)
    # down:     batch*topk × hidden × inter
    tokens = batch_size * topk
    flops = 2 * tokens * intermediate_size * 2 * hidden_size  # gate_up gemm
    flops += 2 * tokens * hidden_size * intermediate_size       # down gemm
    return flops / (elapsed_ms * 1e-3) / 1e12


# ---------------------------------------------------------------------------
# CUDA-graph benchmarking helper
# ---------------------------------------------------------------------------


def bench_cuda_graph(fn, warmup, iters):
    """Warm up, capture a CUDA graph, replay `iters` times, return avg ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    with torch.cuda.stream(capture_stream):
        fn()
        torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=capture_stream):
            fn()

    torch.cuda.synchronize()

    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        g.replay()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / iters


# ---------------------------------------------------------------------------
# sglang default BF16 config
# ---------------------------------------------------------------------------


def sglang_bf16_config(total_m, E):
    """
    Static BF16 config mirroring sglang's get_default_config logic:
      total_m <= E  ->  small-batch config (BLOCK_SIZE_M=16)
      total_m >  E  ->  regular config    (BLOCK_SIZE_M=64)
    """
    if total_m <= E:
        return {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 3,
        }
    else:
        return {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "num_warps": 4,
            "num_stages": 3,
        }


# ---------------------------------------------------------------------------
# Per-kernel bench helpers
# ---------------------------------------------------------------------------


def bench_hpc(x, gate_up_weight, down_weight, topk_ids, topk_scale,
              num_experts_local, warmup, iters):
    """Benchmark hpc.fuse_moe_bf16 (full pipeline)."""
    # rank_ep=0, num_expert_total=num_experts_local: all experts are local
    def fn():
        hpc.fuse_moe_bf16(
            x, gate_up_weight, down_weight,
            topk_ids, topk_scale,
            rank_ep=0,
            num_expert_total=num_experts_local,
        )

    return bench_cuda_graph(fn, warmup, iters)


def bench_sglang(x, gate_up_weight, down_weight, topk_ids, topk_scale,
                 num_experts_local, warmup, iters):
    """
    Benchmark sglang full fused_moe pipeline (BF16) via direct kernel calls.
    All steps are inside the CUDA graph for a fair comparison with hpc.fuse_moe_bf16:
      0. sgl_moe_align_block_size  token sorting  (= hpc count_and_gather)
      1. invoke_fused_moe_kernel   gate_up GEMM   (mul_routed_weight=False)
      2. silu_and_mul              SiLU activation
      3. invoke_fused_moe_kernel   down GEMM      (mul_routed_weight=True, top_k=1)
      4. torch.sum over topk dim   weighted reduce
    """
    batch_size = x.shape[0]
    hidden_size = x.shape[1]
    inter_x2 = gate_up_weight.shape[1]   # intermediate_size * 2
    inter = inter_x2 // 2
    topk = topk_ids.shape[1]
    total_tokens = batch_size * topk

    config = sglang_bf16_config(total_tokens, num_experts_local)
    block_size = config["BLOCK_SIZE_M"]

    # Pre-allocate routing buffers at max possible size (shapes are static for CUDA graph)
    if topk_ids.numel() < num_experts_local + 1:
        max_padded = topk_ids.numel() * block_size
    else:
        max_padded = topk_ids.numel() + (num_experts_local + 1) * (block_size - 1)
    max_m_blocks = (max_padded + block_size - 1) // block_size

    sorted_ids          = torch.empty((max_padded,),              dtype=torch.int32, device=x.device)
    expert_ids          = torch.empty((max_m_blocks,),            dtype=torch.int32, device=x.device)
    num_tokens_post_pad = torch.empty((1,),                       dtype=torch.int32, device=x.device)
    cumsum_buf          = torch.empty((num_experts_local + 2,),   dtype=torch.int32, device=x.device)

    # Intermediate compute buffers (max padded size)
    cache1 = torch.empty((max_padded, inter_x2),        dtype=torch.bfloat16, device=x.device)
    cache2 = torch.empty((max_padded, inter),            dtype=torch.bfloat16, device=x.device)
    cache3 = torch.empty((batch_size, topk, hidden_size), dtype=torch.bfloat16, device=x.device)
    out    = torch.empty((batch_size, hidden_size),       dtype=torch.bfloat16, device=x.device)

    def fn():
        # 0. token sorting
        sgl_moe_align_block_size(
            topk_ids, num_experts_local + 1, block_size,
            sorted_ids, expert_ids, num_tokens_post_pad, cumsum_buf, True,
        )
        # 1. gate_up GEMM
        invoke_fused_moe_kernel(
            x, gate_up_weight, None, cache1,
            None, None, None,
            topk_scale, topk_ids,
            sorted_ids, expert_ids, num_tokens_post_pad,
            False,  # mul_routed_weight
            topk,   # top_k
            config,
            compute_type=tl.bfloat16,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            block_shape=None,
        )
        # 2. SiLU activation: cache1 [max_padded, inter*2] -> cache2 [max_padded, inter]
        silu_and_mul(cache1, cache2)
        # 3. down GEMM – writes weighted results to cache3[batch, topk, hidden]
        invoke_fused_moe_kernel(
            cache2, down_weight, None, cache3,
            None, None, None,
            topk_scale, topk_ids,
            sorted_ids, expert_ids, num_tokens_post_pad,
            True,  # mul_routed_weight (applies routing weights during scatter)
            1,     # top_k=1: each sorted row maps to one (batch, topk_slot) in cache3
            config,
            compute_type=tl.bfloat16,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            block_shape=None,
        )
        # 4. Sum-reduce over topk slots -> [batch, hidden]
        torch.sum(cache3, dim=1, out=out)

    return bench_cuda_graph(fn, warmup, iters)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    hidden_size = args.hidden_size
    intermediate_size = args.intermediate_size
    num_experts = args.num_experts
    topk = args.topk
    tp_size = args.tp_size
    num_experts_local = num_experts // tp_size

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    prop = torch.cuda.get_device_properties(0)
    print(f"\nDevice              : {prop.name}")
    print(f"Model config        : hidden={hidden_size}, inter={intermediate_size}, "
          f"experts={num_experts}, topk={topk}, tp={tp_size}")
    print(f"Local experts/GPU   : {num_experts_local}")
    print(f"Weight shapes       : gate_up=[{num_experts_local}, {intermediate_size*2}, {hidden_size}], "
          f"down=[{num_experts_local}, {hidden_size}, {intermediate_size}]")
    print(f"Timing              : warmup={args.warmup}, iters={args.iters} (CUDA graph replay)\n")

    hdr = (
        f"{'batch':>7}  {'tokens':>8}  "
        f"{'hpc(ms)':>10}  {'hpc(TF)':>9}  "
        f"{'sgl(ms)':>10}  {'sgl(TF)':>9}  "
        f"{'speedup':>8}"
    )
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)

    for bs in BATCH_SIZES:
        x, gate_up_weight, down_weight, topk_ids, topk_scale = make_inputs(
            bs, hidden_size, intermediate_size, num_experts_local, topk
        )

        hpc_ms = bench_hpc(
            x, gate_up_weight, down_weight, topk_ids, topk_scale,
            num_experts_local, args.warmup, args.iters
        )
        sgl_ms = bench_sglang(
            x, gate_up_weight, down_weight, topk_ids, topk_scale,
            num_experts_local, args.warmup, args.iters
        )

        hpc_tf = tflops(bs, topk, hidden_size, intermediate_size, hpc_ms)
        sgl_tf = tflops(bs, topk, hidden_size, intermediate_size, sgl_ms)

        print(
            f"{bs:>7}  {bs*topk:>8}  "
            f"{hpc_ms:>10.4f}  {hpc_tf:>9.2f}  "
            f"{sgl_ms:>10.4f}  {sgl_tf:>9.2f}  "
            f"{sgl_ms/hpc_ms:>8.2f}x"
        )

    print(sep)
    print()


if __name__ == "__main__":
    main()
