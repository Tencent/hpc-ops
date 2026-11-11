#!/usr/bin/env python3
"""H20 baseline: actual SGLang indexed Humming W4A8, no model/server.

Run inside the tested SGLang/Humming venv and set PYTHONPATH to the exact
SGLang checkout's python/ directory. Init, MXFP4 quantization, weight repack,
JIT, reference calculation and route preparation are outside pure-GEMM timing.
The full indexed MoE timing includes route alignment, both activation
quantizations, both GEMMs, SiLU/multiply, workspace allocation and weighted sum.
It excludes router logits/TopK selection, shared expert, and communication.
"""

import argparse
import datetime
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

os.environ["SGLANG_HUMMING_USE_F16_ACCUM"] = "0"
os.environ["SGLANG_HUMMING_MOE_GEMM_TYPE"] = "indexed"

import torch
import torch.nn.functional as F

from humming.layer import HummingMethod
from humming.schema import HummingInputSchema, HummingWeightSchema
from humming.utils.weight import dequantize_weight
from sglang.srt.layers.moe.moe_runner import humming as hr
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def tensor_sha(tensor):
    data = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
    return hashlib.sha256(memoryview(data)).hexdigest()


def tensor_metadata(tensor):
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": tensor_sha(tensor),
    }


def gpu_snapshot():
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,"
        "utilization.gpu,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu",
        "--format=csv,noheader",
    ]
    try:
        return subprocess.check_output(command, text=True).strip()
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"


def errors(actual, reference):
    actual = actual.float()
    reference = reference.float()
    delta = actual - reference
    relative_l2 = (delta.norm() / reference.norm().clamp_min(1e-30)).item()
    result = {
        "finite": bool(torch.isfinite(actual).all()),
        "relative_l2": relative_l2,
        "rmse": delta.square().mean().sqrt().item(),
        "max_abs": delta.abs().max().item(),
        "cosine": F.cosine_similarity(actual.flatten(), reference.flatten(), dim=0).item(),
        "reference_rms": reference.square().mean().sqrt().item(),
    }
    result["passed"] = result["finite"] and relative_l2 < 0.035
    return result


def fp8_reference(values):
    groups = values.float().reshape(*values.shape[:-1], -1, 128)
    scale = (groups.abs().amax(-1, keepdim=True) / 448.0).clamp_min(1e-30)
    return ((groups / scale).to(torch.float8_e4m3fn).float() * scale).reshape_as(values)


def indexed_reference(inputs, weight, routes, first_gemm):
    """FP32 dot, narrowing to BF16 at the same boundary as Humming."""
    m, top_k = routes.shape
    result = torch.empty(m * top_k, weight.shape[-2], device="cuda", dtype=torch.bfloat16)
    flat = routes.flatten()
    for expert in range(weight.shape[0]):
        slots = torch.where(flat == expert)[0]
        if slots.numel() == 0:
            continue
        rows = slots // top_k if first_gemm else slots
        result[slots] = (inputs[rows].float() @ weight[expert].float().T).to(torch.bfloat16)
    return result


def weighted_sum(values, topk_weights):
    return (
        (values.reshape(*topk_weights.shape, -1).float() * topk_weights[..., None].float())
        .sum(dim=1)
        .to(torch.bfloat16)
    )


def time_callable(fn, *, repeats, iterations, warmup, graph_batch):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    direct = []
    for _ in range(repeats):
        begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        begin.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        direct.append(begin.elapsed_time(end) * 1000 / iterations)

    # Warm up on the capture stream after all lazy JIT has completed.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(graph_batch):
            captured = fn()
    graph.replay()
    torch.cuda.synchronize()
    graph_samples = []
    replays = max(1, math.ceil(iterations / graph_batch))
    for _ in range(repeats):
        begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        begin.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        graph_samples.append(begin.elapsed_time(end) * 1000 / (replays * graph_batch))
    captured_copy = captured.clone()
    torch.cuda.synchronize()
    result = {
        "unit": "us",
        "event_eager_median": statistics.median(direct),
        "event_eager_samples": direct,
        "event_graph_median": statistics.median(graph_samples),
        "event_graph_samples": graph_samples,
        "repeats": repeats,
        "eager_iterations_per_repeat": iterations,
        "graph_replays_per_repeat": replays,
        "graph_operations_per_replay": graph_batch,
    }
    del graph
    return result, captured_copy


def time_cold_callable(fn, *, repeats, samples, flush_mib):
    """One operation per graph replay; flush kernel is outside timed events.

    This is an explicit streaming cache-eviction approximation, not a guarantee
    that every cache line was evicted. It is reported separately from warm-L2.
    """
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = fn()
    flush = torch.zeros(flush_mib * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    flush.add_(1.0)
    torch.cuda.synchronize()
    begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    rounds = []
    for _ in range(repeats):
        values = []
        for _ in range(samples):
            flush.add_(1.0)
            begin.record()
            graph.replay()
            end.record()
            end.synchronize()
            values.append(begin.elapsed_time(end) * 1000)
        rounds.append(values)
    copy = captured.clone()
    torch.cuda.synchronize()
    del graph, flush
    return {
        "unit": "us",
        "graph_operations_per_replay": 1,
        "round_medians": [statistics.median(values) for values in rounds],
        "median": statistics.median([statistics.median(values) for values in rounds]),
        "samples": rounds,
        "flush_mib": flush_mib,
        "flush_scope": "FP32 streaming read-modify-write before each start event; excluded from timing",
        "cache_policy": "cold-L2 approximation: eviction buffer > H20 L2, one operation per replay",
    }, copy


def make_fixture(e, h, i, top_k, max_m, seed, save_path=None, load_path=None):
    if load_path is not None:
        data = torch.load(load_path, map_location="cpu", weights_only=True)
        expected = {"E": e, "H": h, "I": i, "top_k": top_k}
        if data["dimensions"] != expected or data["x"].shape[0] < max_m:
            raise ValueError(f"fixture does not cover {expected}, M={max_m}")
        return data

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    schema = HummingWeightSchema(
        b_dtype="float4e2m1", bs_dtype="float8e8m0", weight_scale_group_size=32
    )
    data = {"dimensions": {"E": e, "H": h, "I": i, "top_k": top_k}, "seed": seed, "weights": {}}
    for name, n, k in (("w13", 2 * i, h), ("w2", h, i)):
        original = torch.randn(e, n, k, device="cuda", dtype=torch.bfloat16) / math.sqrt(k)
        tensors = schema.quant_tensor(original, schema, torch.bfloat16)
        # Keep canonical pre-Humming-layout MXFP4 data, for both libraries.
        data["weights"][name] = {key: value.cpu() for key, value in tensors.items()}
        del original, tensors
    data["x"] = torch.randn(max_m, h, device="cuda", dtype=torch.bfloat16).cpu()
    # TopK selection is outside timing. Per-token experts are unique.
    data["topk_ids"] = (
        torch.rand(max_m, e, device="cuda").topk(top_k, dim=-1).indices.to(torch.int32).cpu()
    )
    data["topk_weights"] = torch.softmax(torch.randn(max_m, top_k, device="cuda"), dim=-1).cpu()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, save_path)
    return data


def build_runner(fixture):
    dims = fixture["dimensions"]
    e, h, i, top_k = (dims[x] for x in ("E", "H", "I", "top_k"))
    layer = torch.nn.Module()
    layer.hidden_size = h
    layer.intermediate_size_per_partition = i
    layer.params_dtype = torch.bfloat16
    layer.register_buffer("locks", torch.zeros(2048, dtype=torch.int32, device="cuda"))
    weight_schema = HummingWeightSchema(
        b_dtype="float4e2m1", bs_dtype="float8e8m0", weight_scale_group_size=32
    )
    input_schema = HummingInputSchema(
        a_dtype="float8e4m3", input_scale_group_size=128, input_scale_dtype="float32"
    )
    reference_weights = {}
    scale_metadata = {}
    for name, n, k in (("w13", 2 * i, h), ("w2", h, i)):
        tensors = {key: value.cuda() for key, value in fixture["weights"][name].items()}
        exponent = tensors["weight_scale"].view(torch.uint8).reshape(e, -1).to(torch.int16)
        lower = torch.maximum(exponent.amin(-1), exponent.amax(-1) - 11)
        changed = exponent < lower[:, None]
        scale_metadata[name] = {
            "per_expert_exponent_range_min": int((exponent.amax(-1) - exponent.amin(-1)).min()),
            "per_expert_exponent_range_max": int((exponent.amax(-1) - exponent.amin(-1)).max()),
            "fused_e8m0_adjusted_groups": int(changed.sum()),
            "total_groups": exponent.numel(),
            "canonical_parameters": {
                key: tensor_metadata(value) for key, value in fixture["weights"][name].items()
            },
        }
        reference_weights[name] = dequantize_weight(
            tensors["weight"],
            tensors.get("weight_scale"),
            tensors.get("zero_point"),
            tensors.get("weight_scale_2"),
            weight_schema.b_dtype,
            packed=True,
        )
        for key, value in tensors.items():
            layer.register_parameter(
                f"{name}_{key}", torch.nn.Parameter(value, requires_grad=False)
            )
        HummingMethod.prepare_layer_meta(
            layer,
            n,
            k,
            weight_schema=weight_schema,
            input_schema=input_schema,
            num_experts=e,
            torch_dtype=torch.bfloat16,
            sublayer_name=name,
            tensors=tensors,
        )
        HummingMethod.transform_humming_layer(layer, sublayer_name=name)
        del exponent, lower, changed, tensors

    runner = hr.HummingRunnerCore(
        MoeRunnerConfig(
            num_experts=e,
            num_local_experts=e,
            hidden_size=h,
            intermediate_size_per_partition=i,
            top_k=top_k,
            params_dtype=torch.bfloat16,
            activation="silu",
            gate_up_interleaved=False,
        )
    )
    runner.layer = layer
    return runner, reference_weights, scale_metadata


def run_shape(runner, reference_weights, x, ids, probabilities, args):
    m, h = x.shape
    top_k = ids.shape[1]
    i = runner.layer.intermediate_size_per_partition
    kwargs1, kwargs2 = runner._prepare_indexed_gemm_kwargs(ids)
    qx, sx, sx2 = runner._prepare_input(x, sublayer_name="w13")
    y1 = torch.empty(m * top_k, i * 2, device="cuda", dtype=torch.bfloat16)

    def gemm1():
        return HummingMethod.forward_layer(
            runner.layer,
            qx,
            outputs=y1,
            input_scale=sx,
            input_scale_2=sx2,
            sublayer_name="w13",
            **kwargs1,
        )

    # Explicit return tensor: forward_layer's return may be a view in some versions.
    def call1():
        gemm1()
        return y1

    call1()
    activated = torch.empty(m * top_k, i, device="cuda", dtype=torch.bfloat16)
    runner.apply_activation(y1, activated)
    qmiddle, smiddle, smiddle2 = runner._prepare_input(activated, sublayer_name="w2")
    y2 = torch.empty(m * top_k, h, device="cuda", dtype=torch.bfloat16)

    def call2():
        HummingMethod.forward_layer(
            runner.layer,
            qmiddle,
            outputs=y2,
            input_scale=smiddle,
            input_scale_2=smiddle2,
            sublayer_name="w2",
            **kwargs2,
        )
        return y2

    def full():
        return runner._run_indexed_gemm(x, probabilities, ids)

    print(f"benchmark E={runner.num_experts} M={m}: w13/w2/full", flush=True)
    timing_options = dict(
        repeats=args.repeats,
        iterations=args.iterations,
        warmup=args.warmup,
        graph_batch=args.graph_batch,
    )
    timings = {}
    graph_values = {}
    for name, fn in (
        ("w13_gemm_only", call1),
        ("w2_gemm_only", call2),
        ("moe_two_gemm_full", full),
    ):
        timings[name], graph_values[name] = time_callable(fn, **timing_options)
        timings[name][
            "cache_policy"
        ] = "warm repeated same inputs/routes/weights; active experts may reside in L2"
        if args.cold:
            timings[name]["cold_graph"], _ = time_cold_callable(
                fn,
                repeats=args.repeats,
                samples=args.cold_samples,
                flush_mib=args.flush_mib,
            )
    native1 = call1().clone()
    native2 = call2().clone()
    nativefull = full().clone()

    ref1 = indexed_reference(fp8_reference(x), reference_weights["w13"], ids, True)
    ref2_local = indexed_reference(fp8_reference(activated), reference_weights["w2"], ids, False)
    ref_gate, ref_up = ref1.chunk(2, dim=-1)
    ref_middle = (F.silu(ref_gate.float()) * ref_up.float()).to(torch.bfloat16)
    ref2_pipeline = indexed_reference(
        fp8_reference(ref_middle), reference_weights["w2"], ids, False
    )
    ref_full = weighted_sum(ref2_pipeline, probabilities)
    numerical = {
        "w13_vs_fp32_reference": errors(native1, ref1),
        "w2_vs_fp32_reference_actual_middle": errors(native2, ref2_local),
        "full_vs_fp32_reference": errors(nativefull, ref_full),
        "w13_graph_vs_eager": errors(graph_values["w13_gemm_only"], native1),
        "w2_graph_vs_eager": errors(graph_values["w2_gemm_only"], native2),
        "full_graph_vs_eager": errors(graph_values["moe_two_gemm_full"], nativefull),
    }
    for name, n, k in (("w13_gemm_only", i * 2, h), ("w2_gemm_only", h, i)):
        useful_flops = 2 * m * top_k * n * k
        timings[name]["useful_flops"] = useful_flops
        timings[name]["useful_tflops_graph"] = (
            useful_flops / timings[name]["event_graph_median"] / 1e6
        )
    return {
        "M": m,
        "top_k": top_k,
        "input": tensor_metadata(x),
        "topk_ids": tensor_metadata(ids),
        "topk_weights": tensor_metadata(probabilities),
        "timings": timings,
        "numerical": numerical,
        "passed": all(value["passed"] for value in numerical.values()),
        "gemm_tuning": runner.get_humming_gemm_configs(hr.HummingGemmType.INDEXED),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, help="Save/load canonical production fixture (.pt)")
    parser.add_argument("--tokens", default="1,4,8,16,64,256,1024")
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--graph-batch", type=int, default=4)
    parser.add_argument(
        "--cold", action="store_true", help="Also measure one-call graph replays after L2 eviction"
    )
    parser.add_argument("--cold-samples", type=int, default=20)
    parser.add_argument("--flush-mib", type=int, default=256)
    parser.add_argument("--skip-small-validation", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise SystemExit("This baseline targets an SM90/H20 CUDA device")
    if min(args.repeats, args.iterations, args.warmup, args.graph_batch) < 1:
        parser.error("timing counts must be positive")
    if args.cold_samples < 1 or args.flush_mib < 128:
        parser.error("cold-samples must be positive and eviction buffer >= 128 MiB")
    tokens = sorted({int(value) for value in args.tokens.split(",")})
    if not tokens or min(tokens) < 1 or args.hidden % 128 or args.intermediate % 128:
        parser.error("positive token counts and H/I multiples of 128 required")
    if not 1 <= args.top_k <= args.experts:
        parser.error("1 <= top-k <= experts required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_grad_enabled(False)
    props = torch.cuda.get_device_properties(0)
    result = {
        "status": "running",
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "argv": sys.argv,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {
            name: package_version(name) for name in ("torch", "humming-kernels", "sglang", "triton")
        },
        "torch_cuda": torch.version.cuda,
        "python": sys.version,
        "source": {
            "humming_runner": hr.__file__,
            "humming_method": sys.modules[HummingMethod.__module__].__file__,
        },
        "gpu": {
            "name": props.name,
            "capability": [props.major, props.minor],
            "sm_count": props.multi_processor_count,
            "total_memory": props.total_memory,
            "snapshot_before": gpu_snapshot(),
        },
        "protocol": {
            "seed": args.seed,
            "weight": "MXFP4 E2M1, E8M0 group32",
            "activation": "dynamic FP8 E4M3, FP32 scales, group128",
            "output": "BF16",
            "mma_accumulator": "FP32; SGLANG_HUMMING_USE_F16_ACCUM=0",
            "cache_policy": "warm repeated same inputs/routes/weights; active subset may fit in H20 60 MiB L2; no whole-model throughput inference",
            "cold_cache_policy": "optional --cold adds streaming L2 eviction before each separately timed one-call graph replay; flush excluded",
            "routing": "frozen unique-per-token synthetic TopK and FP32 softmax weights; no router computation in timing",
            "gemm_only_scope": "prequantized inputs + prealigned route metadata + preallocated outputs; excludes route/quant/init/JIT",
            "full_scope": "indexed route alignment + both activation quantizations + 2 GEMMs + BF16 SiLU/multiply + weighted sum + workspaces; excludes router/shared expert/communication/init/JIT",
            "reference": "original MXFP4 dequant, emulated dynamic FP8 g128, FP32 dots/silu/sum, BF16 stage boundaries; relative-L2 threshold .035",
            "background_service": "No processes are stopped by this script. Resident GPU users must be recorded by the orchestrator; event times may include interference.",
        },
        "cases": [],
    }
    atomic_json(args.output, result)
    configs = []
    if not args.skip_small_validation:
        configs.append((4, 512, 256, 2, [128], None))
    configs.append((args.experts, args.hidden, args.intermediate, args.top_k, tokens, args.fixture))
    start = time.time()
    try:
        for e, h, i, top_k, ms, fixture_path in configs:
            print(f"initializing E={e}, H={h}, I={i}, top_k={top_k}", flush=True)
            fixture = make_fixture(
                e,
                h,
                i,
                top_k,
                max(ms),
                args.seed,
                save_path=fixture_path,
                load_path=fixture_path if fixture_path and fixture_path.exists() else None,
            )
            runner, ref_weights, scale_metadata = build_runner(fixture)
            case = {
                "dimensions": fixture["dimensions"],
                "weight_metadata": scale_metadata,
                "layer_meta": {
                    name: meta.to_str() for name, meta in runner.layer.humming_metas.items()
                },
                "measurements": [],
            }
            result["cases"].append(case)
            x, ids, probs = (fixture[name].cuda() for name in ("x", "topk_ids", "topk_weights"))
            for m in ms:
                record = run_shape(
                    runner,
                    ref_weights,
                    x[:m].contiguous(),
                    ids[:m].contiguous(),
                    probs[:m].contiguous(),
                    args,
                )
                case["measurements"].append(record)
                atomic_json(args.output, result)
                print(
                    json.dumps(
                        {
                            "E": e,
                            "M": m,
                            "passed": record["passed"],
                            "graph_us": {
                                name: timing["event_graph_median"]
                                for name, timing in record["timings"].items()
                            },
                        }
                    ),
                    flush=True,
                )
                if not record["passed"]:
                    raise RuntimeError(f"numerical validation failed for E={e} M={m}")
            del runner, ref_weights, fixture, x, ids, probs
            gc.collect()
            torch.cuda.empty_cache()
        result["status"] = "complete"
    except BaseException as exc:
        result["status"] = "failed"
        result["exception"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["elapsed_seconds"] = time.time() - start
        result["gpu"]["snapshot_after"] = gpu_snapshot()
        result["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        atomic_json(args.output, result)


if __name__ == "__main__":
    main()
