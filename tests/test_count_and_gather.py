import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch
import math
import pytest
from pathlib import Path

file_available = os.path.exists("/cfs_cloud_code/theocheng/fused_moe_topk")


@pytest.mark.skipif(not file_available, reason="fused_moe_topk files does not exists!!!")
@pytest.mark.parametrize("hidden_size", [4096])
@pytest.mark.parametrize("intermediate_size", [8192])
@pytest.mark.parametrize("num_expert", [128])
@pytest.mark.parametrize("eprank", [0])
def test_count_and_gather(hidden_size, intermediate_size, num_expert, eprank):
    torch.cuda.manual_seed(10086)
    dtype = torch.float8_e4m3fn

    path = Path("/cfs_cloud_code/theocheng/fused_moe_topk")

    for i, file_path in enumerate(path.rglob("*")):
        if i < 10:
            topk_ids = torch.load(file_path)
            print("i:", i, " file_path:", file_path)
            print("topk_ids:", topk_ids.dtype, topk_ids.shape)

            num_seq = topk_ids.size(0)
            num_topk = topk_ids.size(1)
            print(num_seq, num_topk)

            x = torch.randn((num_seq, hidden_size), dtype=torch.float, device="cuda").to(dtype)

            for _ in range(1):
                unique_values, counts = torch.unique(
                    topk_ids.flatten(), return_counts=True, sorted=True
                )
                (y, yg, topk_pos, seqlens, cu_seqlens, tiles, cu_tiles, tmas) = (
                    hpc.count_and_gather(x, topk_ids, num_expert, eprank, intermediate_size)
                )

                torch.cuda.synchronize()

            if counts.size(0) == num_expert:
                gt = counts
                my = seqlens
                print("gt")
                print(gt)

                print("my")
                print(my)

                abs_diff = torch.abs(gt - my)
                vals, idxs = torch.topk(abs_diff.view(-1), 10)
                idxs = [torch.unravel_index(idx, gt.shape) for idx in idxs]

                for i, idx in enumerate(idxs):
                    cpu_idx = tuple(tensor.cpu().item() for tensor in idx)
                    print(
                        "{:+.4f} vs {:+.4f} with diff = {:.4f}, @ {}".format(
                            gt[idx], my[idx], vals[i], cpu_idx
                        )
                    )

                assert gt.device == my.device
                assert gt.shape == my.shape
                assert torch.allclose(my.to(torch.float), gt.to(torch.float), rtol=0.08, atol=0.01)
