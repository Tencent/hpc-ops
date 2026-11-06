# Copyright (C) 2026 Tencent.

"""Shared W4A8 benchmark checks; extracted without changing frozen helper bodies."""

import math

import torch

from bench_humming_baseline import errors, time_callable, time_cold_callable


def comparison(actual, reference):
    result = errors(actual, reference)
    # Preserve a checkpoint even when a numerical failure produces NaN metrics.
    for name, value in result.items():
        if isinstance(value, float) and not math.isfinite(value):
            result[name] = None
            result["passed"] = False
    return result


def measure(call, args):
    timing, graph_value = time_callable(
        call,
        repeats=args.repeats,
        iterations=args.iterations,
        warmup=args.warmup,
        graph_batch=args.graph_batch,
    )
    timing["cache_policy"] = (
        "warm repeated fixed inputs/routes/weights; active experts can fit in L2"
    )
    eager_value = call().clone()
    graph_check = comparison(graph_value, eager_value)
    graph_check["bit_equal"] = bool(torch.equal(graph_value, eager_value))
    del graph_value
    if args.cold:
        timing["cold_graph"], cold_value = time_cold_callable(
            call,
            repeats=args.repeats,
            samples=args.cold_samples,
            flush_mib=args.flush_mib,
        )
        graph_check["cold_vs_eager"] = comparison(cold_value, eager_value)
        graph_check["cold_bit_equal"] = bool(torch.equal(cold_value, eager_value))
        del cold_value
    return timing, eager_value, graph_check


def raw_fp4_values(packed):
    """Independent E2M1 decoder: low nibble precedes high nibble."""
    lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
        device=packed.device,
    )
    nibble = torch.stack((packed & 15, packed >> 4), dim=-1)
    return lut[nibble.long()].reshape(*packed.shape[:-1], packed.shape[-1] * 2)
