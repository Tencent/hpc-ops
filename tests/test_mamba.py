import sys
import os
import pytest
from pathlib import Path
from einops import rearrange, repeat
import torch.nn.functional as F
import torch
import math

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
from utils import allclose


def selective_state_update_ref(
    state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False
):
    """
    Argument:
        state: (batch, dstate, dim) or (batch, nheads, dstate, dim)
        x: (batch, dim) or (batch, nheads, dim)
        dt: (batch, dim) or (batch, nheads, dim)
        A: (dstate, dim) or (nheads, dstate, dim)
        B: (batch, dstate) or (batch, num_group, dstate)
        C: (batch, dstate) or (batch, num_group, dstate)
        D: (dim,) or (nheads, dim)
        z: (batch, dim) or (batch, nheads, dim)
        dt_bias: (dim,) or (nheads, dim)
    Return:
        out: (batch, dim) or (batch, nheads, dim)
    """
    has_heads = state.dim() > 3
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if dt.dim() == 2:
        dt = dt.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 2:
        B = B.unsqueeze(1)
    if C.dim() == 2:
        C = C.unsqueeze(1)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 2:
        z = z.unsqueeze(1)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    batch, nheads, dstate, dim = state.shape

    assert x.shape == (batch, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dstate, dim)
    num_group = B.shape[1]
    assert nheads % num_group == 0, "nheads must be divisible by num_group"
    assert B.shape == (batch, num_group, dstate)
    assert C.shape == B.shape

    if D is not None:
        assert D.shape == (nheads, dim)
    if z is not None:
        assert z.shape == x.shape
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, dim)
        dt = dt + dt_bias
    dt = F.softplus(dt) if dt_softplus else dt
    dA = torch.exp(rearrange(dt, "b h d -> b h 1 d") * A)  # (batch, nheads, dstate, dim)

    B = repeat(B, "b g n -> b (g h) n", h=nheads // num_group)  # (batch, nheads, dstate)
    C = repeat(C, "b g n -> b (g h) n", h=nheads // num_group)  # (batch, nheads, dstate)
    dB = rearrange(dt, "b h d -> b h 1 d") * rearrange(
        B.float(), "b h n -> b h n 1"
    )  # (batch, nheads, dstate, dim)
    state_new = state.float() * dA + dB * rearrange(
        x.float(), "b h d -> b h 1 d"
    )  # (batch, nheads, dstate, dim)
    state.copy_(state_new.to(state.dtype))
    out = torch.einsum("bhnd,bhn->bhd", state_new, C.float())
    if D is not None:
        out += x.float() * D
    out = (out if z is None else out * F.silu(z.float())).to(x.dtype)
    if not has_heads:
        out = out.squeeze(1)
    return out


def reference_torch_mamba_selective_scan_update(
    ssm_states, x, B, C, z, dt, A, D, dt_bias, indices, num_group
):
    copy_ssm_states = ssm_states.clone()
    local_ssm_states = copy_ssm_states[indices]
    # local_ssm_states = local_ssm_states.permute(0, 1, 3, 2).contiguous()
    headdim = local_ssm_states.shape[-1]
    dstate = local_ssm_states.shape[-2]

    A = repeat(A, "h -> h n p", p=headdim, n=dstate)
    dt = repeat(dt.squeeze(1), "b h -> b h p", p=headdim)
    dt_bias = repeat(dt_bias, "h -> h p", p=headdim)
    D = repeat(D, "h -> h p", p=headdim)
    B = rearrange(B.squeeze(1), "b (g n) -> b g n", g=num_group)
    C = rearrange(C.squeeze(1), "b (g n) -> b g n", g=num_group)
    x = rearrange(x.squeeze(1), "b (h p) -> b h p", p=headdim)
    z = rearrange(z.squeeze(1), "b (h p) -> b h p", p=headdim)
    out = selective_state_update_ref(
        local_ssm_states, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True
    )
    out = rearrange(out, "b h p -> b (h p)").unsqueeze(1)
    copy_ssm_states[indices] = local_ssm_states  # .permute(0, 1, 3, 2).contiguous()
    return copy_ssm_states, out


def generate_shuffled_batch_indices(num_batch, num_max_batch):
    n = min(num_batch, num_max_batch)
    indices = torch.cat(
        [
            torch.randperm(num_batch)[:n],
            torch.zeros(num_max_batch - n, dtype=torch.long),
        ]
    )
    return indices.to(torch.int32).cuda()


@pytest.mark.parametrize("batch_size", [64, 128, 129])
@pytest.mark.parametrize("nheads", [8])
@pytest.mark.parametrize("num_group", [2])
@pytest.mark.parametrize("dstate", [128])
@pytest.mark.parametrize("head_dim", [80])
def test_mamba_selective_scan_update(batch_size, nheads, num_group, dstate, head_dim):
    num_max_batch = 256
    num_batch = batch_size
    ssm_states = torch.rand((num_max_batch, nheads, head_dim, dstate), dtype=torch.float32).cuda()
    # indices = torch.randint(0, max_batch_size, (batch_size,), dtype=torch.int32).cuda()
    indices = torch.randperm(num_max_batch)[:num_batch].to(torch.int32).cuda()

    x = torch.rand((batch_size, nheads * head_dim), dtype=torch.bfloat16).cuda()
    b = torch.rand((batch_size, num_group * dstate), dtype=torch.bfloat16).cuda()
    c = torch.rand((batch_size, num_group * dstate), dtype=torch.bfloat16).cuda()
    z = torch.rand((batch_size, nheads * head_dim), dtype=torch.bfloat16).cuda()
    dt = torch.rand((batch_size, nheads), dtype=torch.bfloat16).cuda()
    """
  d_inner = nheads * head_dim
  zxbcdt = torch.rand(
      (batch_size, d_inner + d_inner + 2 * num_group * dstate + heads),
      dtype=torch.bfloat16).cuda()
  """

    zxbcdt = torch.cat((z, x, b, c, dt), dim=1).contiguous()

    A = torch.rand(nheads, dtype=torch.float32).cuda()
    D = torch.rand(nheads, dtype=torch.float32).cuda()

    AD = torch.stack((A, D), dim=1).flatten().contiguous()

    dt_bias = torch.rand(nheads, dtype=torch.float32).cuda()

    my_ssm_states = ssm_states.clone()
    org_ssm_states = ssm_states.clone().permute(0, 1, 3, 2).contiguous()
    gt_ssm_states, gt_out = reference_torch_mamba_selective_scan_update(
        org_ssm_states, x, b, c, z, dt, A, D, dt_bias, indices, num_group
    )
    gt_ssm_states = gt_ssm_states.permute(0, 1, 3, 2).contiguous()
    gt_out = gt_out.squeeze(1)

    my_out = hpc.selective_state_update(my_ssm_states, zxbcdt, AD, dt_bias, indices, num_group)

    # test ssm states
    gt = gt_ssm_states
    my = my_ssm_states

    assert allclose(gt.to(torch.float32), my.to(torch.float32), atol=1e-5)

    # test out
    gt = gt_out
    my = my_out

    print(gt)
    print(my)

    idx = torch.nonzero(torch.abs(gt - my) > 1e-4)
    print(idx)

    for i in idx:
        t = tuple(i.tolist())
        print(gt[t].item())
        print(my[t].item())

    assert allclose(gt.to(torch.float32), my.to(torch.float32), atol=0.5)
