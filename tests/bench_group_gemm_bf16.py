"""
Benchmark script for hpc.group_gemm_bf16 vs sglang fused_moe_kernel (BF16).

Usage:
    python tests/bench_group_gemm_bf16.py
    python tests/bench_group_gemm_bf16.py --E 128 --N 384 --K 4096
    python tests/bench_group_gemm_bf16.py --E 8 --N 4096 --K 7168 --warmup 10 --iters 100
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import torch
import triton.language as tl
import hpc
from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_kernels import invoke_fused_moe_kernel
from sglang.srt.layers.moe.fused_moe_triton.moe_align_block_size import moe_align_block_size

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--E", type=int, default=8, help="Number of groups (experts)")
    p.add_argument("--N", type=int, default=4096, help="Output dimension per group")
    p.add_argument("--K", type=int, default=7168, help="Input dimension (hidden size)")
    p.add_argument("--warmup", type=int, default=5, help="Warmup iterations before graph capture")
    p.add_argument("--iters", type=int, default=100, help="Graph replay iterations for timing")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

M_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def make_inputs(E, m_per_group, N, K, device="cuda"):
    dtype = torch.bfloat16
    total_m = E * m_per_group
    x = torch.randn((total_m, K), dtype=dtype, device=device)
    w = torch.randn((E, N, K), dtype=dtype, device=device)
    seqlens = torch.full((E,), m_per_group, dtype=torch.int32, device=device)
    cu_seqlens = torch.zeros(E + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.cumsum(seqlens, dim=0)
    output = torch.empty((total_m, N), dtype=dtype, device=device)
    return x, w, seqlens, cu_seqlens, output


def tflops(E, m_per_group, N, K, elapsed_ms):
    flops = 2 * E * m_per_group * N * K
    return flops / (elapsed_ms * 1e-3) / 1e12


# ---------------------------------------------------------------------------
# sglang default BF16 config
# (mirrors get_default_config logic for dtype=None, no server args needed)
# ---------------------------------------------------------------------------


def sglang_bf16_config(total_m, E):
    """
    Replicates sglang's get_default_config logic for plain BF16 (dtype=None).
      M <= E  ->  small-batch config (BLOCK_SIZE_M=16)
      M >  E  ->  regular config    (BLOCK_SIZE_M=64)
    num_warps / num_stages are typical Hopper-friendly defaults.
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
# Benchmark with CUDA graph
# ---------------------------------------------------------------------------


def bench_cuda_graph(fn, warmup, iters):
    """Warm up, capture a CUDA graph, replay `iters` times, return avg ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    with torch.cuda.stream(capture_stream):
        fn()  # dry run on capture stream
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
# Per-kernel bench helpers
# ---------------------------------------------------------------------------


def bench_hpc(x, w, seqlens, cu_seqlens, output, mean_seq, warmup, iters):
    def fn():
        hpc.group_gemm_bf16(
            x, w, seqlens, cu_seqlens, num_seq_per_group_avg=mean_seq, output=output
        )

    return bench_cuda_graph(fn, warmup, iters)


def bench_torch_ref(x, w, seqlens, cu_seqlens, output, warmup, iters):
    E = seqlens.shape[0]
    slices = [(int(cu_seqlens[i].item()), int(cu_seqlens[i + 1].item())) for i in range(E)]

    def fn():
        for i, (s, e) in enumerate(slices):
            output[s:e] = torch.matmul(x[s:e], w[i].t())

    return bench_cuda_graph(fn, warmup, iters)


def bench_sglang(x, w, E, m_per_group, N, warmup, iters):
    """
    Benchmark sglang's invoke_fused_moe_kernel for a single up/gate projection
    (top_k=1, each token routes to its group, no routed-weight multiply).

    moe_align_block_size is called once outside the graph; only the GEMM
    kernel itself is captured and replayed.
    """
    total_m = E * m_per_group
    config = sglang_bf16_config(total_m, E)

    # top_k=1: token i belongs to expert floor(i / m_per_group)
    topk_ids = (
        (torch.arange(total_m, device="cuda") // m_per_group).view(-1, 1).to(torch.int32)
    )  # [total_m, 1]
    topk_weights = torch.ones(total_m, 1, dtype=torch.bfloat16, device="cuda")

    # Routing tensors – computed once, reused by every graph replay
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], E
    )

    # Output shape follows sglang convention: [padded_tokens, N]
    sgl_out = torch.empty((sorted_token_ids.shape[0], N), dtype=torch.bfloat16, device="cuda")

    def fn():
        invoke_fused_moe_kernel(
            A=x,
            B=w,
            bias=None,
            C=sgl_out,
            A_scale=None,
            B_scale=None,
            B_zp=None,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            mul_routed_weight=False,
            top_k=1,
            config=config,
            compute_type=tl.bfloat16,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            block_shape=None,
        )

    return bench_cuda_graph(fn, warmup, iters)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    E, N, K = args.E, args.N, args.K

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    prop = torch.cuda.get_device_properties(0)
    print(f"\nDevice : {prop.name}")
    print(f"Config : E={E}, N={N}, K={K}")
    print(f"Timing : warmup={args.warmup}, iters={args.iters} (CUDA graph replay)\n")

    hdr = (
        f"{'M/group':>8}  {'total_M':>8}  "
        f"{'hpc(ms)':>10}  {'hpc(TF)':>9}  "
        f"{'sgl(ms)':>10}  {'sgl(TF)':>9}  "
        f"{'ref(ms)':>10}  {'ref(TF)':>9}  "
        f"{'hpc/sgl':>8}"
    )
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)

    for m in M_VALUES:
        x, w, seqlens, cu_seqlens, output = make_inputs(E, m, N, K)

        hpc_ms = bench_hpc(x, w, seqlens, cu_seqlens, output, m, args.warmup, args.iters)
        sgl_ms = bench_sglang(x, w, E, m, N, args.warmup, args.iters)
        ref_ms = bench_torch_ref(x, w, seqlens, cu_seqlens, output, args.warmup, args.iters)

        hpc_tf = tflops(E, m, N, K, hpc_ms)
        sgl_tf = tflops(E, m, N, K, sgl_ms)
        ref_tf = tflops(E, m, N, K, ref_ms)

        print(
            f"{m:>8}  {E*m:>8}  "
            f"{hpc_ms:>10.4f}  {hpc_tf:>9.2f}  "
            f"{sgl_ms:>10.4f}  {sgl_tf:>9.2f}  "
            f"{ref_ms:>10.4f}  {ref_tf:>9.2f}  "
            f"{sgl_ms/hpc_ms:>8.2f}x"
        )

    print(sep)
    print()


if __name__ == "__main__":
    main()
