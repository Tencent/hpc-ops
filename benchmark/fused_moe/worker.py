# Copyright (C) 2025 Tencent.

"""Worker process used by benchmark_fuse_moe.py."""
from __future__ import annotations

import argparse
import json
import sys

import torch  # noqa: F401  (force CUDA init before any backend imports vllm)

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backends  # registers all backends as side effect
from backends.base import run_method_c, spec_from_args


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", required=True, choices=backends.known())
    p.add_argument("--num-seq", type=int, required=True)
    p.add_argument("--hidden", type=int, required=True)
    p.add_argument("--intermediate-per-rank", type=int, required=True)
    p.add_argument("--num-expert-local", type=int, required=True)
    p.add_argument("--num-expert-total", type=int, required=True)
    p.add_argument("--num-topk", type=int, required=True)
    p.add_argument("--model", default="")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--ep", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument(
        "--n-timed", type=int, default=52,
        help="Total NVTX-marked replays; first 2 are dropped post-extraction.",
    )
    p.add_argument(
        "--meta-out",
        help="If given, dump backend.extra_metadata() as JSON to this path. "
             "Used by the driver to recover sglang config source etc.",
    )
    args = p.parse_args()

    spec = spec_from_args(args)
    backend = backends.make(args.backend)
    call_fn = backend.setup(spec)
    if args.meta_out:
        with open(args.meta_out, "w") as f:
            json.dump(backend.extra_metadata(), f)
    run_method_c(call_fn, warmup=args.warmup, n_timed=args.n_timed)


if __name__ == "__main__":
    main()
