# Copyright (C) 2026 Tencent.

"""One (shape, backend) run of the DSL fused MoE benchmark; see README.md.

``hpc-direct`` calls the hpc entry points on pre-quantized activations. Every
other backend name is a vLLM ``--moe-backend``: a single-rank vLLM ``FusedMoE``
layer is built with the checkpoint's quant config, its parameters are filled
with seeded random data, ``process_weights_after_loading`` converts them for
the backend, and ``quant_method.apply`` (activation quantization, both GEMMs,
top-k reduce; routing excluded) is timed.
"""

import argparse
import json
import socket
import traceback

import torch

# Per-GPU shapes. Top-k is sampled within the local experts, as in
# benchmark/fused_moe.
MODELS = {
    "qwen3.8-flash-next-ep4": dict(
        quant="fp8_block", experts=128, topk=10, hidden=2560, intermediate=640, swiglu_limit=None
    ),
    "deepseek-v4-flash-tp4": dict(
        quant="mxfp4", experts=256, topk=6, hidden=4096, intermediate=512, swiglu_limit=10.0
    ),
}


def make_routing(num_tokens, experts, topk, gen):
    ids = torch.rand(num_tokens, experts, generator=gen, device="cuda").topk(topk, dim=-1).indices
    weights = torch.softmax(torch.randn(num_tokens, topk, generator=gen, device="cuda"), -1)
    return weights.float(), ids.to(torch.int32)


def time_graph(fns, iters, reps):
    """Per-call latency of ``fns`` replayed from one CUDA graph.

    The graph holds one call per routing, so consecutive calls read different
    experts' weights and small batches are not served from L2 -- in a model
    every layer reads its own weights.
    """
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            for fn in fns:
                fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for fn in fns:
            fn()
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / iters / len(fns))
    samples.sort()
    return samples[len(samples) // 2], samples[0]


def _hpc_direct(shape, gen):
    """Build hpc inputs once; return ``make_fns(x, routings)``."""
    import hpc
    from hpc import pack_mxfp_weight_scale_sm120, prepare_fp8_mx_mma_weights

    E, H, inter = shape["experts"], shape["hidden"], shape["intermediate"]

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

        def quantized_x(num_tokens):
            scale = torch.exp2(
                torch.randint(-8, -4, (num_tokens, H // 128), generator=gen, device="cuda").float()
            )
            return fp8(num_tokens, H), scale

    else:

        def fp4(*size):
            return torch.randint(0, 256, size, generator=gen, device="cuda", dtype=torch.uint8)

        w13, w2 = fp4(E, 2 * inter, H // 2), fp4(E, H, inter // 2)
        s13 = pack_mxfp_weight_scale_sm120(ue8m0(E, 2 * inter, H // 32), 2 * inter, H)
        s2 = pack_mxfp_weight_scale_sm120(ue8m0(E, H, inter // 32), H, inter)
        entry = hpc.fuse_moe_blockwise_w4a8_mxfp4_mxfp8_dsl
        extra = {"swiglu_limit": shape["swiglu_limit"]}

        def quantized_x(num_tokens):
            return fp8(num_tokens, H), ue8m0(num_tokens, H // 32)

    def make_fns(num_tokens, routings):
        x, x_scale = quantized_x(num_tokens)
        out = torch.empty(num_tokens, H, dtype=torch.bfloat16, device="cuda")
        return [
            (
                lambda tw=tw, ti=ti: entry(
                    x,
                    x_scale,
                    w13,
                    s13,
                    w2,
                    s2,
                    ti,
                    tw,
                    rank_ep=0,
                    num_expert_total=E,
                    output=out,
                    **extra,
                )
            )
            for tw, ti in routings
        ]

    return make_fns, "hpc." + entry.__name__


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _fill(param, name, gen):
    if param.dtype == torch.float8_e4m3fn:
        param.data.copy_(
            (torch.randn(param.shape, generator=gen, device="cuda") * 0.5).to(param.dtype)
        )
    elif param.dtype == torch.uint8 and "scale" in name:
        param.data.copy_(
            torch.randint(120, 125, param.shape, generator=gen, device="cuda", dtype=torch.uint8)
        )
    elif param.dtype == torch.uint8:
        param.data.copy_(
            torch.randint(0, 256, param.shape, generator=gen, device="cuda", dtype=torch.uint8)
        )
    elif param.dtype == torch.float32 and "scale" in name:
        param.data.copy_(torch.rand(param.shape, generator=gen, device="cuda") * 0.01 + 0.005)
    elif param.is_floating_point():
        param.data.copy_(torch.randn(param.shape, generator=gen, device="cuda") * 0.01)


def _kernel_name(quant_method):
    kernel = getattr(quant_method, "moe_kernel", None)
    for attr in ("fused_experts", "experts", "impl"):
        inner = getattr(kernel, attr, None)
        if inner is not None:
            return type(getattr(inner, "fused_experts", None) or inner).__name__
    cls = getattr(quant_method, "experts_cls", None)
    return cls.__name__ if cls is not None else type(kernel).__name__


def _vllm_layer(shape, backend, max_tokens, gen, stack):
    """Build the vLLM layer for ``backend``; return ``make_fns(x, routings)``."""
    from vllm.config import (
        CompilationConfig,
        KernelConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.fused_moe import FusedMoEFactory
    from vllm.v1.worker.workspace import init_workspace_manager, is_workspace_manager_initialized

    config = VllmConfig(
        parallel_config=ParallelConfig(tensor_parallel_size=1),
        compilation_config=CompilationConfig(),
        scheduler_config=SchedulerConfig.default_factory(
            max_num_batched_tokens=max(max_tokens, 2048)
        ),
        kernel_config=KernelConfig(moe_backend=backend),
    )
    stack.enter_context(set_current_vllm_config(config))
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        backend="nccl",
        distributed_init_method=f"tcp://127.0.0.1:{_free_port()}",
    )
    initialize_model_parallel(1, 1)
    if not is_workspace_manager_initialized():
        init_workspace_manager(torch.device("cuda:0"))

    if shape["quant"] == "fp8_block":
        from vllm.model_executor.layers.quantization.fp8 import Fp8Config

        quant_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        )
    else:
        from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config

        quant_config = Mxfp4Config()

    torch.set_default_device("cuda")
    layer = FusedMoEFactory(
        num_experts=shape["experts"],
        top_k=shape["topk"],
        hidden_size=shape["hidden"],
        intermediate_size=shape["intermediate"],
        params_dtype=torch.bfloat16,
        renormalize=False,
        quant_config=quant_config,
        tp_size=1,
        dp_size=1,
        pcp_size=1,
        prefix="from_forward_context",
        activation="silu",
        swiglu_limit=shape["swiglu_limit"],
    )
    torch.set_default_device(None)
    experts = layer.routed_experts
    quant_method = layer._quant_method
    for name, param in sorted(experts.named_parameters()):
        _fill(param, name, gen)
    quant_method.process_weights_after_loading(experts)

    def make_fns(num_tokens, routings):
        x = torch.randn(
            num_tokens, shape["hidden"], generator=gen, device="cuda", dtype=torch.bfloat16
        )
        return [
            (lambda tw=tw, ti=ti: quant_method.apply(experts, x, tw, ti, None, None))
            for tw, ti in routings
        ]

    def context(num_tokens):
        return set_forward_context(None, config, num_tokens=num_tokens)

    return make_fns, _kernel_name(quant_method), context


def main():
    import contextlib

    p = argparse.ArgumentParser()
    p.add_argument("--backend", required=True)
    p.add_argument("--quant", choices=("fp8_block", "mxfp4"), required=True)
    p.add_argument("--experts", type=int, required=True)
    p.add_argument("--topk", type=int, required=True)
    p.add_argument("--hidden", type=int, required=True)
    p.add_argument("--intermediate", type=int, required=True)
    p.add_argument("--swiglu-limit", type=float, default=None)
    p.add_argument("--tokens", type=int, nargs="+", required=True)
    p.add_argument("--rotate", type=int, default=8)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--reps", type=int, default=9)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    shape = dict(
        quant=args.quant,
        experts=args.experts,
        topk=args.topk,
        hidden=args.hidden,
        intermediate=args.intermediate,
        swiglu_limit=args.swiglu_limit,
    )
    result = dict(backend=args.backend, shape=shape, rows=[])

    torch.cuda.set_device(0)
    gen = torch.Generator(device="cuda").manual_seed(1234)
    with contextlib.ExitStack() as stack:
        try:
            if args.backend == "hpc-direct":
                make_fns, result["kernel"] = _hpc_direct(shape, gen)
                context = lambda _: contextlib.nullcontext()  # noqa: E731
            else:
                make_fns, result["kernel"], context = _vllm_layer(
                    shape, args.backend, max(args.tokens), gen, stack
                )
            print(f"[{args.backend}] kernel={result['kernel']}", flush=True)
            for num_tokens in args.tokens:
                routing_gen = torch.Generator(device="cuda").manual_seed(num_tokens)
                routings = [
                    make_routing(num_tokens, shape["experts"], shape["topk"], routing_gen)
                    for _ in range(args.rotate)
                ]
                row = dict(tokens=num_tokens)
                try:
                    with context(num_tokens):
                        row["median_us"], row["min_us"] = time_graph(
                            make_fns(num_tokens, routings), args.iters, args.reps
                        )
                    print(
                        f"[{args.backend}] tokens={num_tokens:5d} {row['median_us']:9.1f} us",
                        flush=True,
                    )
                except Exception as e:
                    row["error"] = f"{type(e).__name__}: {e}"[:300]
                    print(f"[{args.backend}] tokens={num_tokens} ERROR {row['error']}", flush=True)
                    torch.cuda.synchronize()
                result["rows"].append(row)
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"[:500]
            result["traceback"] = traceback.format_exc()[-3000:]
            print(f"[{args.backend}] SETUP ERROR {result['error']}", flush=True)

    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
