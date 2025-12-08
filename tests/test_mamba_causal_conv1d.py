import sys
import os
import pytest
from pathlib import Path
from einops import rearrange, repeat
import torch.nn.functional as F
import torch

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
from utils import allclose


def causal_conv1d_update_ref(
    zxbcdt,
    conv_states,
    weight,
    bias,
    indices,
    num_batch,
    state_len,
    conv_dim,
    d_conv,
    d_inner,
    num_head,
):
    seq_len = 1
    z, xbc, dt = torch.split(
        zxbcdt,
        [d_inner, conv_dim, num_head],
        dim=-1,
    )
    xbc = xbc.reshape(num_batch, seq_len, conv_dim)  # （n, conv_dim) -> (batch, n, conv_dim)
    xbc = xbc.permute(0, 2, 1).contiguous()  # (batch, conv_dim. seq_len)

    conv_states = conv_states.permute(
        0, 2, 1
    ).contiguous()  # (max_batch, state_len, conv_dim) -> (max_batch, conv_dim, state_len)
    assert conv_states.size(1) == conv_dim
    assert conv_states.size(2) == state_len
    conv_states_select = conv_states[indices]
    assert conv_states_select.size(0) == num_batch

    weight = weight.permute(1, 0).contiguous()  # (d_conv, conv_dim) -> (conv_dim, d_conv)
    assert weight.size(0) == conv_dim
    assert weight.size(1) == d_conv

    padded_x = torch.cat([conv_states_select, xbc], dim=2).contiguous()
    # update states
    conv_states_select = padded_x[:, :, -state_len:]
    conv_states[indices] = conv_states_select

    y = F.conv1d(
        padded_x.to(torch.float32),
        weight.unsqueeze(1).to(torch.float32),
        bias=bias.to(torch.float32),
        groups=conv_dim,
    )
    y = F.silu(y).to(torch.bfloat16)
    # xbc = torch.ops.trtllm.causal_conv1d_update(xbc, conv_states, weight, bias,
    #                                             activation="silu", conv_state_indices=indices)

    xbc = y.permute(
        0, 2, 1
    ).contiguous()  # (batch, conv_dim, seq_len) -> (batch, seq_len, conv_dim)
    xbc = xbc.reshape(-1, xbc.shape[-1])  # (n, conv_dim)
    return xbc, conv_states.permute(0, 2, 1).contiguous()


def generate_shuffled_batch_indices(num_batch, num_max_batch):
    n = min(num_batch, num_max_batch)
    indices = torch.cat(
        [
            torch.randperm(num_batch)[:n],
            torch.zeros(num_max_batch - n, dtype=torch.long),
        ]
    )
    return indices.to(torch.int32).cuda()


@pytest.mark.parametrize("batch_size", [8, 16, 32])
@pytest.mark.parametrize("nheads", [8, 4])
@pytest.mark.parametrize("state_len", [3])
@pytest.mark.parametrize("d_conv", [4])
@pytest.mark.parametrize("d_inner", [640, 320])
@pytest.mark.parametrize("conv_dim", [1152, 576])
def test_mamba_causal_conv1d_update(batch_size, nheads, state_len, d_conv, d_inner, conv_dim):
    num_max_batch = 256

    conv_states = torch.rand((num_max_batch, state_len, conv_dim), dtype=torch.bfloat16).cuda()
    indices = torch.randperm(num_max_batch)[:batch_size].to(torch.int32).cuda()
    zxbcdt = torch.rand((batch_size, d_inner + conv_dim + nheads), dtype=torch.bfloat16).cuda()
    weight = torch.rand((d_conv, conv_dim), dtype=torch.bfloat16).cuda()
    bias = torch.rand((conv_dim), dtype=torch.bfloat16).cuda()

    conv_states_ref = conv_states.clone().cuda()
    zxbcdt_ref = zxbcdt.clone().cuda()
    weight_ref = weight.clone().cuda()

    hpc.causal_conv1d_update(zxbcdt, conv_states, weight, bias, indices, d_inner, nheads)
    _, xbc, _ = torch.split(zxbcdt, [d_inner, conv_dim, nheads], dim=-1)

    xbc_ref, conv_states_ref = causal_conv1d_update_ref(
        zxbcdt_ref,
        conv_states_ref,
        weight_ref,
        bias,
        indices,
        batch_size,
        state_len,
        conv_dim,
        d_conv,
        d_inner,
        nheads,
    )

    assert allclose(xbc, xbc_ref, rtol=1e-2)
    assert allclose(conv_states, conv_states_ref)


def causal_conv1d_update_with_spec_ref(
    zxbcdt,
    conv_states,
    weight,
    bias,
    indices,
    num_batch,
    state_len,
    conv_dim,
    d_conv,
    d_inner,
    num_head,
    spec_total_tokens,
    num_accept_tokens,
):
    assert state_len == d_conv
    seq_len = spec_total_tokens
    z, xbc, dt = torch.split(
        zxbcdt,
        [d_inner, conv_dim, num_head],
        dim=-1,
    )
    xbc = xbc.reshape(num_batch, seq_len, conv_dim)  # （n, conv_dim) -> (batch, n, conv_dim)
    xbc = xbc.permute(0, 2, 1).contiguous()  # (batch, conv_dim. seq_len)

    conv_states = conv_states.permute(
        0, 2, 1
    ).contiguous()  # (max_batch, state_len, conv_dim) -> (max_batch, conv_dim, state_len)

    weight = weight.permute(1, 0).contiguous()  # (d_conv, conv_dim) -> (conv_dim, d_conv)
    assert weight.size(0) == conv_dim
    assert weight.size(1) == d_conv

    # (batch, conv_dim, 3)
    output = []
    cur_conv_states = torch.zeros(
        (1, conv_dim, d_conv - 1), dtype=conv_states.dtype, device=conv_states.device
    )
    assert num_batch == indices.shape[0]
    for b in range(num_batch):
        if num_accept_tokens[b] == 1:
            cur_conv_states[0] = conv_states[indices[b], :, -4:-1].contiguous()
        elif num_accept_tokens[b] == 2:
            cur_conv_states[0] = conv_states[indices[b], :, -3:].contiguous()
        else:
            raise ValueError(f"wrong num_accept_tokens:{num_accept_tokens[i]} value")

        assert cur_conv_states.size(1) == conv_dim

        padded_x = torch.cat(
            [cur_conv_states, xbc[b : b + 1]], dim=2
        ).contiguous()  # (batch, conv_dim, seq_len)

        # updata states
        new_conv_states = padded_x[:, :, -state_len:]
        conv_states[indices[b]] = new_conv_states

        y = F.conv1d(
            padded_x.to(torch.float32),
            weight.unsqueeze(1).to(torch.float32),
            bias=bias.to(torch.float32),
            groups=conv_dim,
        )
        y = F.silu(y).to(torch.bfloat16)

        assert y.size(-1) == 2

        y = y.permute(0, 2, 1).contiguous()  # (1, conv_dim, seq_len) -> (1, seq_len, conv_dim)
        y = y.reshape(-1, y.shape[-1])  # (2, conv_dim)
        output.append(y)
    out = torch.cat(output, dim=0)
    return out, conv_states.permute(0, 2, 1).contiguous()


@pytest.mark.parametrize("batch_size", [8, 16, 32])
@pytest.mark.parametrize("nheads", [8, 4])
@pytest.mark.parametrize("state_len", [4])
@pytest.mark.parametrize("d_conv", [4])
@pytest.mark.parametrize("d_inner", [640, 320])
@pytest.mark.parametrize("conv_dim", [1152, 576])
def test_mamba_causal_conv1d_update_with_spec(
    batch_size, nheads, state_len, d_conv, d_inner, conv_dim
):
    spec_total_tokens = 2
    num_max_batch = 256

    conv_states = torch.rand((num_max_batch, state_len, conv_dim), dtype=torch.bfloat16).cuda()
    indices = torch.randperm(num_max_batch)[:batch_size].to(torch.int32).cuda()
    zxbcdt = torch.rand(
        (batch_size * spec_total_tokens, d_inner + conv_dim + nheads), dtype=torch.bfloat16
    ).cuda()
    weight = torch.rand((d_conv, conv_dim), dtype=torch.bfloat16).cuda()
    bias = torch.rand((conv_dim), dtype=torch.bfloat16).cuda()
    num_accept_tokens = torch.randint(
        low=1, high=3, size=(batch_size,), dtype=torch.int32
    ).cuda()  # int32

    conv_states_ref = conv_states.clone().cuda()
    zxbcdt_ref = zxbcdt.clone().cuda()
    weight_ref = weight.clone().cuda()

    hpc.causal_conv1d_update_with_spec(
        zxbcdt, conv_states, weight, bias, indices, d_inner, nheads, 2, num_accept_tokens
    )
    _, xbc, _ = torch.split(zxbcdt, [d_inner, conv_dim, nheads], dim=-1)

    xbc_ref, conv_states_ref = causal_conv1d_update_with_spec_ref(
        zxbcdt_ref,
        conv_states_ref,
        weight_ref,
        bias,
        indices,
        batch_size,
        state_len,
        conv_dim,
        d_conv,
        d_inner,
        nheads,
        2,
        num_accept_tokens,
    )
    assert allclose(xbc.to(torch.float32), xbc_ref.to(torch.float32), rtol=1e-2)
    assert allclose(conv_states.to(torch.float32), conv_states_ref.to(torch.float32))
