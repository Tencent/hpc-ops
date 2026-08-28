import os
import sys
from pathlib import Path

sys.path.insert(
    0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0])
)

import pytest

import hpc
import torch


pytestmark = pytest.mark.skipif(
    torch.cuda.get_device_capability() not in ((9, 0), (10, 3)),
    reason="Top-K is implemented for sm90 and sm103",
)


def test_topk_filtered_split_candidate_spill_sanitizer_smoke():
    # M=17 is the first non-cluster batch on SM90+: it selects split4 on H20 and
    # split8 on B300. Keep many distinct values in one FP16 coarse bucket so the
    # exact result set is stable while the split finisher must exercise gmem spill.
    m, n, top_k = 17, 131073, 2048
    row = torch.arange(n, dtype=torch.float32, device="cuda") * 2.5e-10
    assert row.unique().numel() == n
    logits = row.expand(m, -1).clone()
    ke = torch.full((m,), n, dtype=torch.int32, device="cuda")
    output = torch.full((m, top_k), -1, dtype=torch.int32, device="cuda")
    num_valid_rows = torch.tensor([m], dtype=torch.int32, device="cuda")

    # Leave scratch allocation to the public wrapper. This keeps the sanitizer
    # trace compact while still selecting the recommended KV-split workspace.
    hpc.topk_filtered(logits, ke, output, num_valid_rows, top_k)
    torch.cuda.synchronize()

    expected = torch.arange(n - top_k, n, dtype=torch.int32, device="cuda")
    assert torch.equal(output.sort(dim=1).values, expected.expand(m, -1))


def test_topk_filtered_cluster8_sanitizer_smoke():
    m, n, top_k = 1, 131072, 2048
    logits = torch.full((m, n), -10.0, dtype=torch.float32, device="cuda")
    selected = torch.arange(top_k, dtype=torch.int32, device="cuda") * (n // top_k)
    logits[0, selected.long()] = torch.arange(
        1, top_k + 1, dtype=torch.float32, device="cuda"
    )
    ke = torch.full((m,), n, dtype=torch.int32, device="cuda")
    output = torch.full((m, top_k), -1, dtype=torch.int32, device="cuda")
    num_valid_rows = torch.tensor([m], dtype=torch.int32, device="cuda")

    hpc.topk_filtered(logits, ke, output, num_valid_rows, top_k)
    torch.cuda.synchronize()

    assert torch.equal(output.sort(dim=1).values, selected.expand(m, -1))
