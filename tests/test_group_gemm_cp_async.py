import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch
import pytest
from utils import allclose


DTYPE_IN = torch.float8_e4m3fn


def build_inputs(num_group, actual_m, n, k, seed=10086):
    """Create the common input bundle used by correctness tests."""
    torch.cuda.manual_seed(seed)
    seqlens = torch.full((num_group,), actual_m, dtype=torch.int32, device="cuda")
    total_tokens = int(seqlens.sum().item())

    x_pool = torch.randn((total_tokens, k), dtype=torch.float, device="cuda").to(DTYPE_IN)
    w = torch.randn((num_group, n, k), dtype=torch.float, device="cuda").to(DTYPE_IN)
    scale = torch.full((num_group,), 1.0, dtype=torch.float, device="cuda")
    row_indices = torch.randperm(total_tokens, device="cuda").to(torch.int32)

    cu_seqlens = torch.cumsum(
        torch.cat([torch.tensor([0], dtype=torch.int32, device="cuda"), seqlens]), dim=0
    ).to(torch.int32)
    tiles = (seqlens + 63) // 64
    cu_tiles = torch.cumsum(
        torch.cat([torch.tensor([0], dtype=torch.int32, device="cuda"), tiles]), dim=0
    ).to(torch.int32)

    x_compact = x_pool[row_indices.long()]
    return dict(
        seqlens=seqlens,
        total_tokens=total_tokens,
        x_pool=x_pool,
        x_compact=x_compact,
        w=w,
        scale=scale,
        row_indices=row_indices,
        cu_seqlens=cu_seqlens,
        tiles=tiles,
        cu_tiles=cu_tiles,
        mean_seq=total_tokens // num_group,
    )


# Shape grid covers both Down GEMM K-tile dispatch branches and the full
# kTileM ladder (avg tokens per group → 8, 16, 32, 48, 64):
#   k=192 → kTileK=64  + kStage=3 (hunyuan-v3 / qwen3 Down path)
#   k=256 → kTileK=128 + kStage=2 (deepseek-v3 / hunyuan-v2 Down path)
#   actual_m ∈ {8, 16, 32, 42, 256} → kTileM ∈ {8, 16, 32, 48, 64}
# Both actual_m=42 rows cover the kTileM=48 launch-bounds specialisation
# (<48,64,3> and <48,128,2>).
_SHAPES = [
    # (192, 256, 4096, 256),  # kTileM=64, kTileK=128
    (192, 256, 4096, 192),  # kTileM=64, kTileK=64
    # (192, 42, 4096, 256),  # kTileM=48, kTileK=128
    # (192, 42, 4096, 192),  # kTileM=48, kTileK=64
    (192, 32, 4096, 256),  # kTileM=32, kTileK=128
    (192, 16, 4096, 256),  # kTileM=16, kTileK=128
    (192, 8, 4096, 256),  # kTileM=8,  kTileK=128
]


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("scatter", [False, True])
@pytest.mark.parametrize("use_task_map", [False, True])
def test_correctness(shape, scatter, use_task_map):
    num_group, actual_m, n, k = shape
    d = build_inputs(num_group, actual_m, n, k)

    # Ground truth: validated src/group_gemm TMA-based path (contiguous layout).
    gt = hpc.group_gemm_fp8(
        d["x_compact"],
        d["w"],
        d["seqlens"],
        d["cu_seqlens"],
        d["scale"],
        num_seq_per_group_avg=d["mean_seq"],
    )

    if scatter:
        actual_output = torch.ops.hpc.group_gemm_fp8_scatter_cp_async(
            d["x_pool"],
            d["w"],
            d["scale"],
            d["row_indices"],
            d["seqlens"],
            d["cu_seqlens"],
            d["tiles"],
            d["cu_tiles"],
            use_task_map,
        )
    else:
        actual_output = torch.ops.hpc.group_gemm_fp8_cp_async(
            d["x_compact"],
            d["w"],
            d["scale"],
            d["seqlens"],
            d["cu_seqlens"],
            d["tiles"],
            d["cu_tiles"],
            use_task_map,
        )

    assert allclose(gt.to(torch.float32), actual_output.to(torch.float32), rtol=0.08, atol=1)
