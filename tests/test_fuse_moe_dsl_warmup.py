# Copyright (C) 2026 Tencent.

"""The warmup has to reach every variant the pipeline can reach.

Its whole purpose is that nothing compiles once serving has started, and it
enumerates variants by rerunning the pipeline's host-side choices rather than by
sharing code with it. So the property worth testing is the one that would break
if the two ever drifted: warm a configuration, then run it at sizes across the
range and assert that not one kernel was compiled along the way.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import pytest
import torch

import hpc
import hpc.cute.fuse_moe
from hpc.cute.fuse_moe import (
    BlockwiseShape,
    blockwise_warmup_units,
    fp8_blockwise_warmup_units,
)
from test_fuse_moe_blockwise_fp8_dsl import _make_mx_case
from test_fuse_moe_w4a8_mxfp4_mxfp8_dsl import _is_sm120
from test_fuse_moe_w4a8_mxfp4_mxfp8_dsl import _dsl_available, _make_case, _run

pytestmark = pytest.mark.skipif(
    not _is_sm120() or not _dsl_available(),
    reason="the CuTe DSL fused MoE requires an SM120 GPU and nvidia-cutlass-dsl",
)

SHAPE = BlockwiseShape(
    num_topk=2,
    num_expert_local=8,
    num_expert_total=8,
    hidden_size=256,
    intermediate_size=128,
)

# Both sides of the routing kernels' token-count split, and enough sizes around
# it to cross the tile height and load box thresholds. 1024 tokens put 256 rows
# on each expert, past the FP8 prefill tile pair's cutoff.
TOKEN_COUNTS = (1, 2, 7, 32, 64, 128, 129, 200, 1024)


def _compiled_keys():
    """Every kernel the process has compiled, across the pipeline's modules."""
    dsl = hpc.cute.fuse_moe
    return {
        key
        for module in (dsl._routing, dsl._gather, dsl._grouped_gemm, dsl._reduce)
        for key in module._cache
    }


@pytest.fixture
def fuse_reduce(request, monkeypatch):
    """Pin how the down projection finishes the top-k sum.

    Shapes small enough to run in a test all land on the same side of the L2
    threshold that decides it, so the fused epilogue's variants would otherwise
    never be reached -- by the warmup or by the run.
    """
    monkeypatch.setattr(hpc.cute.fuse_moe._pipeline, "_fuse_reduce", lambda *_: request.param)
    return request.param


@pytest.mark.parametrize("bf16_input", [False, True])
@pytest.mark.parametrize("fuse_reduce", [False, True], indirect=True)
def test_warmup_leaves_nothing_to_compile(fuse_reduce, bf16_input):
    device = torch.device("cuda", 0)
    for unit in blockwise_warmup_units(
        SHAPE, range(1, max(TOKEN_COUNTS) + 1), device, bf16_input=bf16_input
    ):
        unit.compile()

    warmed = _compiled_keys()
    for num_tokens in TOKEN_COUNTS:
        case = _make_case(
            num_tokens,
            SHAPE.num_topk,
            SHAPE.num_expert_local,
            SHAPE.hidden_size,
            SHAPE.intermediate_size,
            seed=2031,
        )
        if bf16_input:
            case = dict(case, x=case["x_dequantized"].to(torch.bfloat16), x_scale=None)
        _run(case, hpc.fuse_moe_blockwise_w4a8_mxfp4_mxfp8_dsl)
        torch.cuda.synchronize()
        assert _compiled_keys() == warmed, (
            f"{num_tokens} tokens compiled a kernel the warmup missed: "
            f"{_compiled_keys() - warmed}"
        )


@pytest.mark.parametrize("fuse_reduce", [False, True], indirect=True)
def test_fp8_warmup_leaves_nothing_to_compile(fuse_reduce):
    """Same property for the FP8 path, which is the one SM120 actually serves."""
    device = torch.device("cuda", 0)
    for unit in fp8_blockwise_warmup_units(SHAPE, range(1, max(TOKEN_COUNTS) + 1), device):
        unit.compile()

    warmed = _compiled_keys()
    for num_tokens in TOKEN_COUNTS:
        case = _make_mx_case(
            num_tokens,
            SHAPE.num_topk,
            SHAPE.num_expert_local,
            SHAPE.hidden_size,
            SHAPE.intermediate_size,
            seed=2031,
        )
        hpc.fuse_moe_blockwise_fp8_dsl(
            case["x"],
            case["x_scale"],
            case["gate_up_weight"],
            case["gate_up_weight_scale"],
            case["down_weight"],
            case["down_weight_scale"],
            case["topk_ids"],
            case["topk_scale"],
            0,
            SHAPE.num_expert_local,
            mx_mma=True,
        )
        torch.cuda.synchronize()
        assert _compiled_keys() == warmed, (
            f"{num_tokens} tokens compiled a kernel the FP8 warmup missed: "
            f"{_compiled_keys() - warmed}"
        )


def test_warmup_collapses_to_a_few_variants():
    """A whole prefill budget's worth of sizes has to stay worth compiling.

    The units are compiled one at a time at startup, seconds each, so a sweep
    that stopped collapsing would quietly turn startup into minutes.
    """
    device = torch.device("cuda", 0)
    units = blockwise_warmup_units(SHAPE, range(1, 8449), device)
    assert len(units) == len({unit.key for unit in units})
    assert len(units) <= 24, [unit.key for unit in units]

    fp8_units = fp8_blockwise_warmup_units(SHAPE, range(1, 8449), device)
    assert len(fp8_units) == len({unit.key for unit in fp8_units})
    assert len(fp8_units) <= 24, [unit.key for unit in fp8_units]


def test_route_shape_agrees_with_the_buffers_it_sizes():
    """``RouteBuffers`` and the warmup have to read the same shape."""
    device = torch.device("cuda", 0)
    for num_tokens in TOKEN_COUNTS:
        buffers = hpc.cute.fuse_moe._routing.RouteBuffers(
            num_tokens, SHAPE.num_topk, SHAPE.num_expert_local, device
        )
        shape = hpc.cute.fuse_moe._routing.route_shape(
            num_tokens, SHAPE.num_topk, SHAPE.num_expert_local
        )
        assert (shape.tile_m, shape.m_tiles, shape.max_padded_rows) == (
            buffers.tile_m,
            buffers.expected_m_tiles,
            buffers.max_padded_rows,
        )
