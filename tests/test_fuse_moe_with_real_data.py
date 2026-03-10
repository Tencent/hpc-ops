import torch
import sys
import os
import pytest
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))
import hpc


@pytest.mark.skip()
def test_fuse_moe_with_real_data():
    dtype = torch.float8_e4m3fn

    num_group = 256
    m = 8
    n = 4096
    k = 256
    seqlens = torch.load("/cfs_cloud_code/scx/groupgemm_topk/xydebug_seqlens_10.pt")

    num_tiles = ((m + 7) // 8) * (n // 128) * num_group
    num_sm = 78
    num_waves = (num_tiles + num_sm - 1) // num_sm + 1
    task_map_workspace = torch.empty((num_waves, num_sm, 4), dtype=torch.int32, device="cuda")

    total_seq = torch.sum(seqlens)
    total_seq_pad = m * num_group

    x = (torch.randn((total_seq, k), dtype=torch.float, device="cuda") / 10).to(dtype)
    w = (torch.randn((num_group, n, k), dtype=torch.float, device="cuda") / 10).to(dtype)

    xscale = torch.randn((k // 128, total_seq_pad), dtype=torch.float, device="cuda")
    wscale = torch.randn(
        (num_group, n // 128, ((k // 128 + 3) // 4) * 4), dtype=torch.float, device="cuda"
    )

    cu_seqlens = torch.cumsum(
        torch.tensor(
            [0] + seqlens.cpu().tolist(),
            device="cuda",
        ),
        dim=0,
    ).to(torch.int32)

    mean_seq = int(total_seq / num_group)

    my = hpc.group_gemm_blockwise_fp8(
        x,
        w,
        seqlens,
        cu_seqlens,
        xscale,
        wscale,
        num_seq_per_group_avg=mean_seq,
        task_map_workspace=task_map_workspace,
    )

    num_tokens = 16
    num_topk = 6
    hidden_size = 4096
    intermediate_size = 2048 // 8
    size_ep = 1
    rank_ep = 0

    topk_ids = (
        torch.load("/cfs_cloud_code/scx/groupgemm_topk/xydebug_topk_ids_10.pt")
        .cuda()
        .to(torch.int32)
    )

    topk_scale = torch.randn((num_tokens, num_topk), dtype=torch.float, device="cuda") / num_topk

    x = (torch.randn((num_tokens, hidden_size), dtype=torch.float, device="cuda") / 100).to(dtype)
    x_scale = torch.randn((num_tokens, hidden_size // 128), dtype=torch.float, device="cuda")
    gate_up_weight = torch.randn(
        (num_group // size_ep, intermediate_size * 2, hidden_size),
        dtype=torch.float,
        device="cuda",
    ).to(dtype)
    gate_up_weight_scale = torch.randn(
        (num_group // size_ep, intermediate_size * 2 // 128, (hidden_size // 128 + 3) // 4 * 4),
        dtype=torch.float,
        device="cuda",
    )
    down_weight = torch.randn(
        (num_group // size_ep, hidden_size, intermediate_size),
        dtype=torch.float,
        device="cuda",
    ).to(dtype)
    down_weight_scale = torch.randn(
        (num_group // size_ep, hidden_size // 128, (intermediate_size // 128 + 3) // 4 * 4),
        dtype=torch.float,
        device="cuda",
    )
    shared_output = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda")

    my = hpc.fuse_moe_blockwise(
        x,
        x_scale,
        gate_up_weight,
        gate_up_weight_scale,
        down_weight,
        down_weight_scale,
        topk_ids,
        topk_scale,
        rank_ep,
        num_group,
        shared_output,
    )
