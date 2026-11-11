#!/usr/bin/env python3
"""Paired H20 W4A8 full routed-MoE benchmark on the frozen canonical fixture.

Timed forward starts with BF16 x plus GPU Top-K IDs/probabilities and includes
routing, both FP8 quantizations/GEMMs, BF16 SiLU/multiply and weighted reduction.
Router/Top-K selection, shared expert, communication, weight preparation and JIT
are outside the scope, matching bench_humming_baseline.py's full-MoE lane.
"""

import argparse
import datetime
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bench_humming_baseline as baseline
from common import comparison, measure, raw_fp4_values
from bench_humming_baseline import (
    atomic_json,
    build_runner,
    gpu_snapshot,
    package_version,
    tensor_metadata,
)
import torch
import torch.nn.functional as F
import hpc
from hpc.w4a8 import prepare_w4a8_weights

FIXTURE_SHA256 = "50823fdf999ff9250137da24f028af68a7eac3c8e4d1c6289e5d79860182f859"
DIMENSIONS = {"E": 256, "H": 2048, "I": 512, "top_k": 8}
TOKENS = (1, 4, 8, 16, 64, 256, 1024)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CanonicalWeights:
    """Reference-only lazy decoder; avoids retaining a full FP32 weight copy.

    The frozen indexed_reference helper uses shape and expert indexing only.
    Decode original E2M1 nibbles/E8M0 bytes independently of either runtime's
    prepared weights. Its FP32 matmul/BF16 boundary is unchanged.
    """

    def __init__(self, tensors, device):
        self.packed = tensors["weight"].view(torch.uint8)
        self.exponents = tensors["weight_scale"].view(torch.uint8)
        e, n, half_k = self.packed.shape
        self.shape = (e, n, half_k * 2)
        self.device = device

    def __getitem__(self, expert):
        _, n, k = self.shape
        q4 = raw_fp4_values(self.packed[expert].to(self.device))
        scale = torch.exp2(self.exponents[expert].to(self.device, torch.float32) - 127)
        return (q4.reshape(n, k // 32, 32) * scale[..., None]).reshape(n, k)


def full_reference(fixture, x, ids, probabilities, *, keep_stages=False):
    """Same frozen full reference, with independent original MXFP4 decoding.

    keep_stages is an untimed debugging aid for unit tests, not an inference API.
    No actual Humming/hpc intermediate supplies the second reference GEMM input.
    """
    w13 = CanonicalWeights(fixture["weights"]["w13"], x.device)
    w2 = CanonicalWeights(fixture["weights"]["w2"], x.device)
    y1 = baseline.indexed_reference(baseline.fp8_reference(x), w13, ids, True)
    gate, up = y1.chunk(2, dim=-1)
    middle = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
    y2 = baseline.indexed_reference(baseline.fp8_reference(middle), w2, ids, False)
    output = baseline.weighted_sum(y2, probabilities)
    if keep_stages:
        return output, {"w13": y1, "activation": middle, "w2": y2}
    return output


def all_passed(value):
    """Include nested cold/eager checks that the original outer gate omitted."""
    if isinstance(value, dict):
        return all(value.get("passed", True) is not False and all_passed(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_passed(v) for v in value)
    return True


def run_shape(runner, prepared, fixture, m, args):
    x = fixture["x"][:m].cuda().contiguous()
    ids = fixture["topk_ids"][:m].cuda().contiguous()
    probabilities = fixture["topk_weights"][:m].cuda().contiguous()
    record = {
        "M": m,
        "input": tensor_metadata(x),
        "topk_ids": tensor_metadata(ids),
        "topk_weights": tensor_metadata(probabilities),
        "timings": {},
        "numerical": {},
    }

    def humming_call():
        return runner._run_indexed_gemm(x, probabilities, ids)

    def hpc_call():
        return hpc.fuse_moe_w4a8(x, *prepared["w13"], *prepared["w2"], ids, probabilities)

    print(f"M={m}: independent full-MoE FP32 reference", flush=True)
    reference = full_reference(fixture, x, ids, probabilities)
    # Correctness precedes performance; both paths own their full intermediate chain.
    initial = {
        name: call().clone() for name, call in (("humming", humming_call), ("hpc", hpc_call))
    }
    for name, output in initial.items():
        if output.shape != x.shape or output.dtype != torch.bfloat16:
            raise RuntimeError(f"{name}: expected BF16 output with shape {tuple(x.shape)}")
        record["numerical"][name + "_initial_vs_fp32"] = comparison(output, reference)
    if not all_passed(record["numerical"]):
        record["passed"] = False
        return record
    del initial

    actual = {}
    for name, call in (("humming", humming_call), ("hpc", hpc_call)):
        print(f"M={m}: timing full {name}", flush=True)
        timing, output, graph_check = measure(call, args)
        record["timings"][name] = timing
        record["numerical"][name + "_vs_fp32"] = comparison(output, reference)
        record["numerical"][name + "_graph_vs_eager"] = graph_check
        actual[name] = output
    record["numerical"]["hpc_vs_humming"] = comparison(actual["hpc"], actual["humming"])
    record["ratios"] = {
        "hpc_us_div_humming_us_warm_graph": (
            record["timings"]["hpc"]["event_graph_median"]
            / record["timings"]["humming"]["event_graph_median"]
        ),
        "meaning": "less than 1 means hpc is faster in this full routed-MoE scope",
    }
    if args.cold:
        record["ratios"]["hpc_us_div_humming_us_cold_graph"] = (
            record["timings"]["hpc"]["cold_graph"]["median"]
            / record["timings"]["humming"]["cold_graph"]["median"]
        )
    record["passed"] = all_passed(record["numerical"])
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", default=",".join(map(str, TOKENS)))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--graph-batch", type=int, default=4)
    parser.add_argument("--cold", action="store_true")
    parser.add_argument("--cold-samples", type=int, default=20)
    parser.add_argument("--flush-mib", type=int, default=256)
    args = parser.parse_args()
    tokens = sorted(set(int(v) for v in args.tokens.split(",")))
    if not tokens or min(tokens) < 1 or max(tokens) > max(TOKENS):
        parser.error("positive token counts up to 1024 required")
    if min(args.repeats, args.iterations, args.warmup, args.graph_batch, args.cold_samples) < 1:
        parser.error("timing counts must be positive")
    if args.flush_mib < 128:
        parser.error("flush-mib must be at least 128")
    if not args.fixture.is_file() or file_sha(args.fixture) != FIXTURE_SHA256:
        parser.error("fixture must be the existing frozen canonical-e256.pt (SHA mismatch/missing)")
    if torch.cuda.get_device_capability() != (9, 0) or "H20" not in torch.cuda.get_device_name():
        raise SystemExit("This comparison targets the NVIDIA H20 SM90 device")
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    fixture = torch.load(args.fixture, map_location="cpu", weights_only=True)
    if fixture["dimensions"] != DIMENSIONS or fixture["x"].shape[0] < max(tokens):
        parser.error("fixture dimensions do not match the frozen E256/H2048/I512/top8 setup")
    result = {
        "status": "running",
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "argv": sys.argv,
        "script_sha256": file_sha(__file__),
        "baseline_script_sha256": file_sha(baseline.__file__),
        "comparison_helper_sha256": file_sha(sys.modules[measure.__module__].__file__),
        "fixture": {"path": str(args.fixture), "sha256": FIXTURE_SHA256},
        "dimensions": DIMENSIONS,
        "versions": {
            n: package_version(n) for n in ("torch", "humming-kernels", "sglang", "triton")
        },
        "sources": {
            "hpc": hpc.__file__,
            "hpc_built_json": hpc.__built_json__,
            "humming_runner": baseline.hr.__file__,
        },
        "gpu_before": gpu_snapshot(),
        "protocol": {
            "scope": "BF16 input + GPU Top-K IDs/weights -> GPU routing, 2 quantizations, 2 GEMMs, BF16 SiLU/multiply, weighted Top-K reduction; native forward workspace allocation included",
            "excluded": "router/Top-K selection, shared expert, communication, fixture I/O, weight preparation, JIT and warmup",
            "independent_chains": "hpc computes its own first GEMM/activation/second FP8 input; no Humming middle is shared",
            "reference": "independent original E2M1/E8M0 decode; frozen emulated FP8 g128, FP32 dots/SiLU/sum, BF16 stage boundaries; finite and relative-L2 <0.035",
            "cold": "streaming eviction before one-call graph replay; eviction excluded, cold-cache approximation",
            "topk": "same frozen unique-per-token IDs and FP32 probabilities; router excluded on both sides",
            "frozen_counts": tokens == list(TOKENS)
            and args.repeats == 3
            and args.iterations == 50
            and args.warmup == 5
            and args.graph_batch == 4
            and args.cold
            and args.cold_samples == 20
            and args.flush_mib == 256,
            "background_service": "no other processes stopped; GPU snapshots do not prove exclusive use",
        },
        "measurements": [],
    }
    atomic_json(args.output, result)
    start = time.time()
    try:
        runner, reference_weights, weight_metadata = build_runner(fixture)
        # The full reference decodes original canonical weights itself.
        del reference_weights
        prepared = {
            name: prepare_w4a8_weights(
                fixture["weights"][name]["weight"].cuda(),
                fixture["weights"][name]["weight_scale"].cuda(),
            )
            for name in ("w13", "w2")
        }
        result["weight_metadata"] = weight_metadata
        result["hpc_prepared_weights"] = {
            name: [tensor_metadata(t) for t in values] for name, values in prepared.items()
        }
        for m in tokens:
            record = run_shape(runner, prepared, fixture, m, args)
            result["measurements"].append(record)
            atomic_json(args.output, result)
            print(
                json.dumps({"M": m, "passed": record["passed"], "ratios": record.get("ratios")}),
                flush=True,
            )
            if not record["passed"]:
                raise RuntimeError(f"numerical validation failed for M={m}")
            gc.collect()
            torch.cuda.empty_cache()
        result["status"] = "complete"
    except BaseException as exc:
        result["status"] = "failed"
        result["exception"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["elapsed_seconds"] = time.time() - start
        result["gpu_after"] = gpu_snapshot()
        result["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        atomic_json(args.output, result)


if __name__ == "__main__":
    main()
