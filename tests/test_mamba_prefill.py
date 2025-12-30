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


def get_diff_item(input, other, rtol, atol, topk=1):
    input = input.reshape(-1)
    other = other.reshape(-1)
    a1 = abs(input - other)
    a2 = atol + rtol * abs(other)
    diff = a1 - a2
    diff_values, diff_indices = torch.topk(diff, topk)
    print("input:")
    print(input[diff_indices])
    print("other:")
    print(other[diff_indices])
    print("diff:")
    print(diff_values)


def chunk_state_ref(B, x, dt, dA_cumsum):
    """
    Argument:
        B: (batch, seqlen, ngroups, headdim)
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, nheads, nchunks, chunk_size)
        dA_cumsum: (batch, nheads, nchunks, chunk_size)
    Return:
        states: (batch, nchunks, nheads, headdim, dstate)
    """
    # Check constraints.
    batch, seqlen, nheads, headdim = x.shape
    dstate = B.shape[-1]
    _, _, nchunks, chunk_size = dt.shape
    assert seqlen <= nchunks * chunk_size
    assert x.shape == (batch, seqlen, nheads, headdim)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    ngroups = B.shape[2]
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    B = repeat(B, "b l g d -> b l (g h) d", h=nheads // ngroups)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    if seqlen < nchunks * chunk_size:
        x = F.pad(x, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
        B = F.pad(B, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
    x = rearrange(x, "b (c l) h p -> b c l h p", l=chunk_size)
    B = rearrange(B, "b (c l) ... -> b c l ...", l=chunk_size)
    decay_states = torch.exp((dA_cumsum[:, :, :, -1:] - dA_cumsum))
    res = torch.zeros([batch, nchunks, nheads, headdim, dstate], device=x.device, dtype=x.dtype)
    decay_states_dt = decay_states.to(x.dtype) * dt.to(x.dtype)
    # to save memory
    for i in range(chunk_size):
        tmp_res = torch.einsum(
            "bchn,bhc,bchp->bchpn",
            B[:, :, i, :, :].to(x.dtype),
            decay_states_dt[:, :, :, i],
            x[:, :, i, :, :],
        )
        res += tmp_res
    return res


def state_passing_ref(states, dA_chunk_cumsum, initial_states=None):
    """
    Argument:
        states: (batch, nchunks, nheads, dim)
        dA_chunk_cumsum: (batch, nheads, nchunks)
        initial_states: (batch, nheads, dim)
    Return:
        out: (batch, nchunks, nheads, dim)
        final_states: (batch, nheads, dim)
    """
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, 0])
    states = torch.cat([rearrange(initial_states, "b h d -> b 1 h d"), states], dim=1)
    dA_chunk_cumsum = F.pad(dA_chunk_cumsum, (1, 0))
    dA_chunk_cumsum = torch.cumsum(dA_chunk_cumsum, dim=-1)
    nchunks = dA_chunk_cumsum.shape[-1]
    # (batch, nheads, nchunks, nchunks)
    dt_chunk_segment_sum = dA_chunk_cumsum[:, :, :, None] - dA_chunk_cumsum[:, :, None, :]
    # (batch, nheads, nchunks, nchunks)
    decay_chunk = torch.exp(dt_chunk_segment_sum)
    causal_mask = torch.tril(
        torch.ones(nchunks, nchunks, device=states.device, dtype=bool), diagonal=0
    )
    decay_chunk = decay_chunk.masked_fill(~causal_mask, 0)
    out = torch.einsum("bhzc,bchd->bzhd", decay_chunk.to(dtype=states.dtype), states)
    return out[:, :-1], out[:, -1]


def chunk_scan_ref(B, C, x, dt, dA_cumsum, prev_states, D=None, z=None):
    """
    Argument:
        B: (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, ngroups, dstate)
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, nheads, nchunks, chunk_size)
        dA_cumsum: (batch, nheads, nchunks, chunk_size)
        prev_states: (batch, nchunks, nheads, headdim, dstate)
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
    Return:
        out: (batch, seqlen, nheads, headdim)
    """
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    assert B.shape == (batch, seqlen, ngroups, dstate)
    _, _, nchunks, chunk_size = dt.shape
    assert seqlen <= nchunks * chunk_size
    assert C.shape == B.shape
    B = repeat(B, "b l g d -> b l (g h) d", h=nheads // ngroups)
    C = repeat(C, "b l g d -> b l (g h) d", h=nheads // ngroups)
    if seqlen < nchunks * chunk_size:
        x = F.pad(x, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
        B = F.pad(B, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
        C = F.pad(C, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
        if z is not None:
            z = F.pad(z, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
    CB = torch.einsum(
        "bclhn,bcshn->bchls",
        rearrange(C, "b (c l) h n -> b c l h n", c=nchunks),
        rearrange(B, "b (c s) h n -> b c s h n", c=nchunks),
    )
    # (batch, nheads, nchunks, chunksize, chunksize)
    decay = torch.exp((dA_cumsum[:, :, :, :, None] - dA_cumsum[:, :, :, None, :]))
    scores_decay = CB * rearrange(decay, "b h c l s -> b c h l s")
    # to save memory
    del CB, decay
    causal_mask = torch.tril(
        torch.ones(chunk_size, chunk_size, device=x.device, dtype=bool), diagonal=0
    )
    scores_decay = scores_decay.masked_fill(~causal_mask, 0)
    out = torch.einsum(
        "bchls,bhcs,bcshp->bclhp",
        scores_decay.to(x.dtype),
        dt.to(x.dtype),
        rearrange(x, "b (c s) h p -> b c s h p", c=nchunks),
    )
    state_decay_out = torch.exp(rearrange(dA_cumsum, "b h c l -> b c l h 1"))
    out_prev = torch.einsum(
        "bclhn,bchpn->bclhp",
        rearrange(C, "b (c l) h n -> b c l h n", c=nchunks),
        prev_states.to(C.dtype),
    )
    out_prev = out_prev * state_decay_out
    out = out + out_prev
    out = rearrange(out, "b c l h p -> b (c l) h p")
    if D is not None:
        if D.dim() == 1:
            D = rearrange(D, "h -> h 1")
        out = out + x * D
    return (out if z is None else out * F.silu(z)).to(x.dtype)


def ssd_chunk_scan_combined_ref(
    x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None, dt_softplus=False
):
    """
    Argument:
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, seqlen, nheads)
        A: (nheads)
        B: (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, ngroups, dstate)
        chunk_size: int
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
        dt_bias: (nheads,)
    Return:
        out: (batch, seqlen, nheads, headdim)
        final_states: (batch, nheads, dstate, headdim)
    """
    batch, seqlen, nheads, headdim = x.shape
    dstate = B.shape[-1]
    if seqlen % chunk_size != 0:
        dt = F.pad(dt, (0, 0, 0, chunk_size - seqlen % chunk_size))
    mask = torch.zeros_like(dt)
    mask[:, 0:seqlen, :] = 1
    dt = rearrange(dt, "b (c l) h -> b h c l", l=chunk_size)
    mask = rearrange(mask, "b (c l) h -> b h c l", l=chunk_size)
    dt = dt.float()  # We want high precision for this before cumsum
    if dt_bias is not None:
        dt = dt + rearrange(dt_bias, "h -> h 1 1")
    if dt_softplus:
        dt = F.softplus(dt)
    dt = torch.clamp(dt, min=0)
    dt = dt * mask
    dA = dt * rearrange(A, "h -> h 1 1")
    dA_cumsum = torch.cumsum(dA, dim=-1)
    # 1. Compute the state for each chunk
    states = chunk_state_ref(B, x, dt, dA_cumsum)
    states_dtype = states.dtype
    if states.dtype not in [torch.float32, torch.float64]:
        states = states.to(torch.float32)
    # import pdb;pdb.set_trace()
    # 2. Pass the state to all the chunks by weighted cumsum.
    # state_passing_ref is much less numerically stable
    states, final_states = state_passing_ref(
        rearrange(states, "... p n -> ... (p n)"), dA_cumsum[:, :, :, -1]
    )
    states, final_states = [
        rearrange(t, "... (p n) -> ... p n", n=dstate) for t in [states, final_states]
    ]
    states = states.to(states_dtype)
    # final_states = final_states.to(states_dtype)
    final_states = final_states.permute(0, 1, 3, 2).contiguous()
    # 3. Compute the output for each chunk
    out = chunk_scan_ref(B, C, x, dt, dA_cumsum, states, D=D, z=z)
    if seqlen % chunk_size != 0:
        out = out[:, 0:seqlen, :, :]
    return out, final_states


def mamba_prefill_torch_ref(x, dt, A, B, C, D, z, dt_bias, cu_seqlens, seq_idx):
    batch_size = cu_seqlens.shape[0] - 1
    gt_ys = []
    gt_states = []
    for bi in range(batch_size):
        y_bi, final_states = ssd_chunk_scan_combined_ref(
            x[cu_seqlens[bi] : cu_seqlens[bi + 1]].unsqueeze(0),
            dt[cu_seqlens[bi] : cu_seqlens[bi + 1]].unsqueeze(0),
            A,
            B[cu_seqlens[bi] : cu_seqlens[bi + 1]].unsqueeze(0),
            C[cu_seqlens[bi] : cu_seqlens[bi + 1]].unsqueeze(0),
            256,
            D,
            z[cu_seqlens[bi] : cu_seqlens[bi + 1]].unsqueeze(0),
            dt_bias,
            dt_softplus=True,
        )
        gt_ys.append(y_bi.squeeze(0))
        gt_states.append(final_states)

    gt_ys = torch.cat(gt_ys, dim=0)
    gt_states = torch.cat(gt_states, dim=0)
    return gt_ys, gt_states


def mamba_prefill_trtllm_triton_ref(x, dt, A, B, C, D, z, dt_bias, cu_seqlens, seq_idx):
    from tensorrt_llm._torch.modules.mamba.ssd_combined import mamba_chunk_scan_combined

    y, current_ssm_states = mamba_chunk_scan_combined(
        x.unsqueeze(0),
        dt.unsqueeze(0),
        A,
        B.unsqueeze(0),
        C.unsqueeze(0),
        chunk_size=256,
        D=D,
        z=z.unsqueeze(0),
        dt_bias=dt_bias,
        initial_states=None,
        dt_softplus=True,
        cu_seqlens=cu_seqlens,
        seq_idx=seq_idx,
        return_varlen_states=True,
        return_final_states=False,
    )

    y = y.squeeze(0)
    current_ssm_states = current_ssm_states.transpose(-1, -2)
    return y, current_ssm_states


def mamba_prefill_hpc_torch_ref_perbatch_perhead(x, dt, A, B, C, D, z, dt_bias):
    """
    Args:
        x: [seqlen, head_dim]
        dt: [seqlen]
        A: [1]
        B: [seqlen, dstate]
        C: [seqlen, dstate]
        D: [1]
        z: [seqlen, head_dim]
        dt_bias: [1]
    Return:
        y: [seqlen, head_dim]
        states: [head_dim, dstate]
    """
    x = x.squeeze(0)
    dt = dt.squeeze(0)
    B = B.squeeze(0)
    C = C.squeeze(0)
    z = z.squeeze(0)
    seqlen = x.shape[0]
    head_dim = x.shape[-1]
    dstate = B.shape[-1]
    chunk_size = 256
    nchunks = (seqlen + chunk_size - 1) // chunk_size
    padded_seqlen = nchunks * chunk_size
    # get chunked tensor
    dt = dt.float() + dt_bias
    dt = F.softplus(dt)
    chunked_dt = torch.zeros((padded_seqlen), device=x.device, dtype=dt.dtype)
    chunked_x = torch.zeros((padded_seqlen, head_dim), device=x.device, dtype=x.dtype)
    chunked_B = torch.zeros((padded_seqlen, dstate), device=x.device, dtype=x.dtype)
    chunked_C = torch.zeros((padded_seqlen, dstate), device=x.device, dtype=x.dtype)
    chunked_dt[:seqlen] = dt
    chunked_x[:seqlen] = x
    chunked_B[:seqlen] = B
    chunked_C[:seqlen] = C

    chunked_dt = chunked_dt.reshape(nchunks, 1, chunk_size)
    chunked_x = chunked_x.reshape(nchunks, chunk_size, head_dim)
    chunked_B = chunked_B.reshape(nchunks, chunk_size, dstate)
    chunked_C = chunked_C.reshape(nchunks, chunk_size, dstate)

    # Step 1. Compute cumsum for dA
    dt_cumsum = torch.cumsum(chunked_dt, dim=-1)
    dA_cumsum = A * dt_cumsum
    chunked_decay_states = torch.exp(dA_cumsum[:, :, -1:] - dA_cumsum[:, :, :])

    # Step 2. Compute chunk state
    chunked_states = (
        chunked_x.transpose(-1, -2) * chunked_dt * chunked_decay_states
    ) @ chunked_B.float()  # [nchunks, head_dim, dstate]
    # Step 3. State passing: Compute cumsum for chunked_states
    chunked_states_cumsum = torch.empty_like(chunked_states)
    chunked_states_cumsum[0] = chunked_states[0]
    for ci in range(1, nchunks):
        chunked_states_cumsum[ci] = chunked_states[ci] + chunked_states_cumsum[ci - 1] * torch.exp(
            dA_cumsum[ci, 0, -1]
        )
    final_states = chunked_states_cumsum[-1]

    # Step 4. Compute pre_y
    pre_y = torch.zeros(
        (nchunks, chunk_size, head_dim), device=x.device, dtype=chunked_states_cumsum.dtype
    )
    if nchunks > 1:
        pre_y[1:] = (
            chunked_states_cumsum[:-1] @ (chunked_C[1:].transpose(-1, -2)).float()
        ).transpose(-1, -2)
        for ci in range(1, nchunks):
            for col in range(chunk_size):
                pre_y[ci, col] *= torch.exp(dA_cumsum[ci, 0, col])

    # Step 5. Chunk scan
    cur_states = chunked_C @ chunked_B.transpose(-1, -2)
    # apply mask
    for ci in range(nchunks):
        for row in range(chunk_size):
            for col in range(chunk_size):
                pre_len = ci * chunk_size
                if pre_len + row < seqlen and pre_len + col < seqlen:
                    cur_states[ci, row, col] *= (
                        torch.exp(dA_cumsum[ci, 0, row] - dA_cumsum[ci, 0, col]) * dt[pre_len + col]
                    )
                else:
                    cur_states[ci, row, col] = 0
    cur_states = torch.tril(cur_states)
    y = pre_y + cur_states @ chunked_x
    y = y.reshape(padded_seqlen, head_dim)[:seqlen]
    y = y + x * D
    y = y * F.silu(z)
    return y, final_states.transpose(-1, -2)


def mamba_prefill_hpc_torch_ref(x, dt, A, B, C, D, z, dt_bias, cu_seqlens, seq_idx):
    batch_size = cu_seqlens.shape[0] - 1
    nheads = x.shape[-2]
    head_dim = x.shape[-1]
    dstate = B.shape[-1]
    ngroups = B.shape[-2]
    nheads_per_group = nheads // ngroups
    gt_ys = torch.empty_like(x)
    gt_states = torch.empty(
        batch_size, nheads, dstate, head_dim, device=x.device, dtype=torch.float
    )

    for bi in range(batch_size):
        for ni in range(nheads):
            y, final_states = mamba_prefill_hpc_torch_ref_perbatch_perhead(
                x[cu_seqlens[bi] : cu_seqlens[bi + 1], ni].unsqueeze(0),
                dt[cu_seqlens[bi] : cu_seqlens[bi + 1], ni].unsqueeze(0),
                A[ni],
                B[cu_seqlens[bi] : cu_seqlens[bi + 1], ni // nheads_per_group].unsqueeze(0),
                C[cu_seqlens[bi] : cu_seqlens[bi + 1], ni // nheads_per_group].unsqueeze(0),
                D[ni],
                z[cu_seqlens[bi] : cu_seqlens[bi + 1], ni].unsqueeze(0),
                dt_bias[ni],
            )
            gt_ys[cu_seqlens[bi] : cu_seqlens[bi + 1], ni] = y
            gt_states[bi, ni] = final_states

    return gt_ys, gt_states


def exp_dA_chunked_cumsum_torch_ref(seqlens, dt, A, dt_bias, chunk_size):
    nheads = dt.shape[1]
    dt = dt.float() + dt_bias
    dt = F.softplus(dt)
    raw_dt = dt.clone()
    dt *= A
    cu_seqlens = (
        torch.cat(
            [torch.zeros(1).to(torch.device("cuda")), torch.cumsum(seqlens, dim=0)],
            dim=0,
        )
        .to(torch.int32)
        .unsqueeze(0)
    )
    padded_seqlens = ((seqlens + 4 - 1) // 4) * 4
    cu_padded_seqlens = (
        torch.cat(
            [torch.zeros(1).to(torch.device("cuda")), torch.cumsum(padded_seqlens, dim=0)],
            dim=0,
        )
        .to(torch.int32)
        .unsqueeze(0)
    )
    seqlens_for_cat = (
        torch.cat(
            [seqlens, torch.zeros(1).to(torch.device("cuda"))],
            dim=0,
        )
        .to(torch.int32)
        .unsqueeze(0)
    )
    nchunks = (seqlens + chunk_size - 1) // chunk_size
    cu_chunks = (
        torch.cat(
            [torch.zeros(1).to(torch.device("cuda")), torch.cumsum(nchunks, dim=0)],
            dim=0,
        )
        .to(torch.int32)
        .unsqueeze(0)
    )
    nchunks = (
        torch.cat(
            [nchunks, torch.zeros(1).to(torch.device("cuda"))],
            dim=0,
        )
        .to(torch.int32)
        .unsqueeze(0)
    )
    padded_seqlens = (
        torch.cat(
            [padded_seqlens, torch.zeros(1).to(torch.device("cuda"))],
            dim=0,
        )
        .to(torch.int32)
        .unsqueeze(0)
    )
    split_metadata = torch.cat(
        [cu_seqlens, cu_padded_seqlens, cu_chunks, seqlens_for_cat, padded_seqlens, nchunks], dim=0
    )

    exp_dA_cumsum = []
    dt_padded = torch.empty(cu_padded_seqlens[0, -1], nheads, dtype=torch.float32, device="cuda")
    for bi in range(seqlens.shape[0]):
        dt_batch = dt[cu_seqlens[0, bi] : cu_seqlens[0, bi] + seqlens[bi]]
        raw_dt_batch = raw_dt[cu_seqlens[0, bi] : cu_seqlens[0, bi] + seqlens[bi]]

        padded_count = nchunks[0, bi] * chunk_size - seqlens[bi]
        padded_dt_batch = F.pad(dt_batch, (0, 0, 0, padded_count), mode="constant", value=0)
        padded_dt_batch = padded_dt_batch.reshape(nchunks[0, bi], chunk_size, -1)
        dA_cumsum_batch = torch.cumsum(padded_dt_batch, dim=1)
        exp_dA_cumsum_batch = dA_cumsum_batch
        exp_dA_cumsum_batch = exp_dA_cumsum_batch.reshape(nchunks[0, bi] * chunk_size, -1)
        exp_dA_cumsum_batch = exp_dA_cumsum_batch[: padded_seqlens[0, bi], :]
        dt_padded[cu_padded_seqlens[0, bi] : cu_padded_seqlens[0, bi] + seqlens[bi]] = raw_dt_batch
        exp_dA_cumsum.append(exp_dA_cumsum_batch)

    return (
        torch.cat(exp_dA_cumsum, dim=0).transpose(1, 0),
        dt_padded.transpose(1, 0),
        split_metadata,
        cu_chunks[0, -1].item(),
    )


def causal_conv1d_prefill_with_scale_torch_ref(
    xbc,
    weight,
    bias,
    conv_states,
    indices,
    x_scale,
    y_scale,
    split_metadata,
    nheads,
    head_dim,
    chunk_size,
):
    local_conv_states = conv_states.clone()
    cu_seqlens = split_metadata[0]
    cu_padded_seqlens = split_metadata[1]
    cu_chunks = split_metadata[2]
    seqlens = split_metadata[3]
    padded_seqlens = split_metadata[4]
    nchunks = split_metadata[5]
    batch_size = nchunks.shape[0] - 1
    state_len = conv_states.shape[-2]
    conv_dim = conv_states.shape[-1]
    gt_xbc = []
    gt_scaled_x = []
    for bi in range(batch_size):
        xbc_batch = xbc[cu_seqlens[bi] : cu_seqlens[bi] + seqlens[bi], :]
        x_scale_batch = x_scale[:, cu_padded_seqlens[bi] : cu_padded_seqlens[bi] + seqlens[bi]]
        y_scale_batch = y_scale[:, cu_padded_seqlens[bi] : cu_padded_seqlens[bi] + seqlens[bi]]

        local_conv_states[indices[bi]] = xbc_batch[-state_len::]
        padded_x = torch.cat(
            [torch.zeros(state_len, conv_dim, device=xbc.device), xbc_batch], dim=0
        )
        y = F.conv1d(
            padded_x.permute(1, 0).to(torch.float32),
            weight.permute(1, 0).unsqueeze(1).to(torch.float32),
            bias=bias.to(torch.float32),
            groups=conv_dim,
        ).permute(1, 0)
        y = F.silu(y)
        gt_xbc.append(y.clone())
        x_scale_batch = (
            x_scale_batch.unsqueeze(-1)
            .repeat(1, 1, head_dim)
            .permute(1, 0, 2)
            .reshape(seqlens[bi], -1)
        )
        chunk_last_pos = torch.min(
            (torch.arange(seqlens[bi], device=y_scale_batch.device) // chunk_size * chunk_size)
            + chunk_size
            - 1,
            seqlens[bi] - 1,
        )
        y_scale_batch = torch.exp(y_scale_batch[:, chunk_last_pos] - y_scale_batch[:, :])
        y_scale_batch = (
            y_scale_batch.unsqueeze(-1)
            .repeat(1, 1, head_dim)
            .permute(1, 0, 2)
            .reshape(seqlens[bi], -1)
        )
        y[:, : nheads * head_dim] = y[:, : nheads * head_dim] * x_scale_batch * y_scale_batch
        gt_scaled_x.append(y[:, : nheads * head_dim].clone())
    gt_xbc = torch.cat(gt_xbc, dim=0).to(torch.bfloat16)
    gt_scaled_x = torch.cat(gt_scaled_x, dim=0).to(torch.bfloat16)
    return gt_xbc, local_conv_states, gt_scaled_x


def causal_conv1d_prefill_ref(
    xbc,
    weight,
    bias,
    conv_states,
    indices,
    cu_seqlens,
    seqlens,
):
    local_conv_states = conv_states.clone()
    batch_size = indices.shape[0]
    state_len = conv_states.shape[-2]
    conv_dim = conv_states.shape[-1]
    gt_xbc = []
    for bi in range(batch_size):
        xbc_batch = xbc[cu_seqlens[bi] : cu_seqlens[bi] + seqlens[bi], :]
        local_conv_states[indices[bi]] = xbc_batch[-state_len::]
        padded_x = torch.cat(
            [torch.zeros(state_len, conv_dim, device=xbc.device), xbc_batch], dim=0
        )
        y = F.conv1d(
            padded_x.permute(1, 0).to(torch.float32),
            weight.permute(1, 0).unsqueeze(1).to(torch.float32),
            bias=bias.to(torch.float32),
            groups=conv_dim,
        ).permute(1, 0)
        y = F.silu(y)
        gt_xbc.append(y.clone())

    gt_xbc = torch.cat(gt_xbc, dim=0).to(torch.bfloat16)
    return gt_xbc, local_conv_states


def chunked_states_torch_ref(B, x, split_metadata, chunk_size, nheads, head_dim):
    cu_seqlens = split_metadata[0]
    cu_padded_seqlens = split_metadata[1]
    cu_chunks = split_metadata[2]
    seqlens = split_metadata[3]
    padded_seqlens = split_metadata[4]
    nchunks = split_metadata[5]

    x = x.reshape(-1, nheads, head_dim)
    batch_size = nchunks.shape[0] - 1
    ngroups = B.shape[-2]
    dstate = B.shape[-1]
    chunk_states = torch.rand(
        nheads, nchunks.sum(), head_dim, dstate, dtype=torch.float, device="cuda"
    )
    for bi in range(batch_size):
        B_batch = B[cu_seqlens[bi] : cu_seqlens[bi] + seqlens[bi], :, :]
        padded_zeros = torch.zeros(
            nchunks[bi] * chunk_size - seqlens[bi],
            ngroups,
            dstate,
            device=B_batch.device,
            dtype=B_batch.dtype,
        )
        padded_B_batch = (
            torch.cat([B_batch, padded_zeros], dim=0)
            .repeat(1, 1, nheads // ngroups)
            .reshape(-1, nheads, dstate)
        )
        padded_B_batch = padded_B_batch.reshape(nchunks[bi], chunk_size, nheads, dstate).permute(
            0, 2, 3, 1
        )
        x_batch = x[cu_seqlens[bi] : cu_seqlens[bi] + seqlens[bi], :, :]
        padded_x_zeros = torch.zeros(
            nchunks[bi] * chunk_size - seqlens[bi],
            nheads,
            head_dim,
            device=x_batch.device,
            dtype=x_batch.dtype,
        )
        padded_x_batch = torch.cat([x_batch, padded_x_zeros], dim=0)
        padded_x_batch = padded_x_batch.reshape(nchunks[bi], chunk_size, nheads, head_dim).permute(
            0, 2, 1, 3
        )
        chunk_states_batch = (
            padded_B_batch.float() @ padded_x_batch.float()
        )  # [nchunks, nheads, dstate, head_dim]
        chunk_states_batch = chunk_states_batch.permute(1, 0, 3, 2)
        chunk_states[:, cu_chunks[bi] : cu_chunks[bi] + nchunks[bi], :, :] = chunk_states_batch

    return chunk_states


def state_passing_torch_ref(chunk_states, yscale, split_metadata, ssm_states, indices, chunk_size):
    cu_padded_seqlens = split_metadata[1]
    seqlens = split_metadata[3]
    nchunks = split_metadata[5]

    ssm_states_result = ssm_states.clone()
    pre_chunks = 0
    nheads = chunk_states.shape[0]
    head_dim = chunk_states.shape[-2]
    dstate = chunk_states.shape[-1]
    total_chunks = chunk_states.shape[1]
    batch_size = seqlens.shape[0] - 1

    chunk_states_cumsum = torch.empty(
        nheads, total_chunks - batch_size, head_dim, dstate, dtype=torch.bfloat16, device="cuda"
    )
    for bi in range(batch_size):
        nchunk = nchunks[bi]
        chunk_states_batch = chunk_states[:, pre_chunks : pre_chunks + nchunk, :, :].clone()
        yscale_batch = yscale[:, cu_padded_seqlens[bi] : cu_padded_seqlens[bi] + seqlens[bi]]
        for ci in range(1, nchunk):
            chunk_states_batch[:, ci, :, :] = chunk_states_batch[:, ci, :, :] + chunk_states_batch[
                :, ci - 1, :, :
            ] * torch.exp(
                yscale_batch[:, min(ci * chunk_size + chunk_size, seqlens[bi]) - 1]
            ).unsqueeze(
                -1
            ).unsqueeze(
                -1
            )

        chunk_states_cumsum[:, pre_chunks - bi : pre_chunks - bi + nchunk - 1, :, :] = (
            chunk_states_batch[:, :-1, :, :]
        )
        pre_chunks += nchunk
        ssm_states_result[indices[bi]] = chunk_states_batch[:, nchunk - 1, :, :]

    return chunk_states_cumsum, ssm_states_result


def pre_y_bmm_torch_ref(chunk_states_cumsum, C, split_metadata, chunk_size):
    cu_seqlens = split_metadata[0]
    cu_padded_seqlens = split_metadata[1]
    cu_chunks = split_metadata[2]
    seqlens = split_metadata[3]
    padded_seqlens = split_metadata[4]
    nchunks = split_metadata[5]

    batch_size = nchunks.shape[0] - 1
    total_chunks = chunk_states_cumsum.shape[1]
    nheads = chunk_states_cumsum.shape[0]
    head_dim = chunk_states_cumsum.shape[-2]
    dstate = chunk_states_cumsum.shape[-1]
    ngroups = C.shape[-2]

    pre_y = torch.empty(
        nheads, total_chunks, chunk_size, head_dim, dtype=torch.float, device="cuda"
    )
    for bi in range(batch_size):
        nchunk = nchunks[bi] - 1
        if nchunk == 0:
            continue
        C_batch = C[cu_seqlens[bi] + chunk_size : cu_seqlens[bi] + seqlens[bi], :, :]
        padded_zeros = torch.zeros(
            nchunks[bi] * chunk_size - seqlens[bi],
            ngroups,
            dstate,
            device=C_batch.device,
            dtype=C_batch.dtype,
        )
        padded_C_batch = (
            torch.cat([C_batch, padded_zeros], dim=0)
            .repeat(1, 1, nheads // ngroups)
            .reshape(-1, nheads, dstate)
        )
        padded_C_batch = padded_C_batch.reshape(
            nchunks[bi] - 1, chunk_size, nheads, dstate
        ).permute(0, 2, 1, 3)
        states_batch = chunk_states_cumsum[
            :, cu_chunks[bi] - bi : cu_chunks[bi] - bi + nchunks[bi] - 1, :, :
        ]
        states_batch = states_batch.reshape(nheads, nchunks[bi] - 1, head_dim, dstate).permute(
            1, 0, 3, 2
        )

        chunk_states_batch = (
            padded_C_batch.float() @ states_batch.float()
        )  # [nchunks, nheads, chunk_size, head_dim]
        pre_y[:, cu_chunks[bi] - bi : cu_chunks[bi] - bi + nchunks[bi] - 1, :, :] = (
            chunk_states_batch.permute(1, 0, 2, 3)
        )

    return pre_y


def chun_scan_torch_ref(B, C, x, pre_y, z, xs, ys, D, split_metadata, chunk_size):
    cu_seqlens = split_metadata[0]
    cu_padded_seqlens = split_metadata[1]
    cu_chunks = split_metadata[2]
    seqlens = split_metadata[3]
    padded_seqlens = split_metadata[4]
    nchunks = split_metadata[5]

    batch_size = nchunks.shape[0] - 1
    nheads = x.shape[-2]
    head_dim = x.shape[-1]
    dstate = B.shape[-1]
    ngroups = B.shape[-2]

    y_list = []
    for bi in range(batch_size):
        C_batch = C[cu_seqlens[bi] : cu_seqlens[bi] + seqlens[bi], :, :]
        padded_zeros = torch.zeros(
            nchunks[bi] * chunk_size - seqlens[bi],
            ngroups,
            dstate,
            device=C_batch.device,
            dtype=C_batch.dtype,
        )
        padded_C_batch = (
            torch.cat([C_batch, padded_zeros], dim=0)
            .repeat(1, 1, nheads // ngroups)
            .reshape(-1, nheads, dstate)
        )
        padded_C_batch = padded_C_batch.reshape(nchunks[bi], chunk_size, nheads, dstate).permute(
            0, 2, 1, 3
        )  # [nchunks, nheads, chunk_size, dstate]

        B_batch = B[cu_seqlens[bi] : cu_seqlens[bi] + seqlens[bi], :, :]
        padded_zeros = torch.zeros(
            nchunks[bi] * chunk_size - seqlens[bi],
            ngroups,
            dstate,
            device=B_batch.device,
            dtype=B_batch.dtype,
        )
        padded_B_batch = (
            torch.cat([B_batch, padded_zeros], dim=0)
            .repeat(1, 1, nheads // ngroups)
            .reshape(-1, nheads, dstate)
        )
        padded_B_batch = padded_B_batch.reshape(nchunks[bi], chunk_size, nheads, dstate).permute(
            0, 2, 3, 1
        )  # [nchunks, nheads, dstate, chunk_size]

        x_batch = x[cu_seqlens[bi] : cu_seqlens[bi] + seqlens[bi], :, :]
        padded_x_zeros = torch.zeros(
            nchunks[bi] * chunk_size - seqlens[bi],
            nheads,
            head_dim,
            device=x_batch.device,
            dtype=x_batch.dtype,
        )
        padded_x_batch = torch.cat([x_batch, padded_x_zeros], dim=0)
        padded_x_batch = padded_x_batch.reshape(nchunks[bi], chunk_size, nheads, head_dim).permute(
            0, 2, 1, 3
        )  # [nchunks, nheads, chunk_size, head_dim]

        z_batch = z[cu_seqlens[bi] : cu_seqlens[bi] + seqlens[bi], :, :]

        xs_batch = xs[:, cu_padded_seqlens[bi] : cu_padded_seqlens[bi] + seqlens[bi]]
        ys_batch = ys[:, cu_padded_seqlens[bi] : cu_padded_seqlens[bi] + seqlens[bi]]
        padded_scale_zeros = torch.zeros(
            nheads,
            nchunks[bi] * chunk_size - seqlens[bi],
            device=x_batch.device,
            dtype=x_batch.dtype,
        )
        # [nchunks, nheads, chunk_size]
        padded_xs_batch = (
            torch.cat([xs_batch, padded_scale_zeros], dim=1)
            .reshape(nheads, nchunks[bi], chunk_size)
            .permute(1, 0, 2)
        )
        padded_ys_batch = (
            torch.cat([ys_batch, padded_scale_zeros], dim=1)
            .reshape(nheads, nchunks[bi], chunk_size)
            .permute(1, 0, 2)
        )

        padded_P_batch = (
            padded_C_batch.float() @ padded_B_batch.float()
        )  # [nchunks, nheads, chunk_size, chunk_size]
        padded_P_batch = torch.tril(padded_P_batch)

        padded_P_batch *= padded_xs_batch[:, :, None, :]
        decay_padded_ys_batch = torch.exp(
            padded_ys_batch[:, :, :, None] - padded_ys_batch[:, :, None, :]
        )
        decay_padded_ys_batch = torch.tril(decay_padded_ys_batch)
        padded_P_batch *= decay_padded_ys_batch

        padded_P_batch = padded_P_batch.to(torch.bfloat16).to(torch.float32)

        padded_y_batch = (
            padded_P_batch @ padded_x_batch.float()
        )  # [nchunks, nheads, chunk_size, head_dim]
        if nchunks[bi] > 1:
            padded_y_batch[1:] += pre_y[
                :, cu_chunks[bi] - bi : cu_chunks[bi] - bi + nchunks[bi] - 1
            ].permute(1, 0, 2, 3) * torch.exp(padded_ys_batch[1:]).unsqueeze(-1)
        y_batch = (
            padded_y_batch.permute(0, 2, 1, 3)
            .reshape(-1, nheads, head_dim)[: seqlens[bi], :, :]
            .permute(1, 0, 2)
        )  # [nheads, seqlen, head_dim]
        y_batch = y_batch + x_batch.permute(1, 0, 2) * D.unsqueeze(-1).unsqueeze(-1)
        y_batch = y_batch * F.silu(z_batch.permute(1, 0, 2))
        y_list.append(y_batch.permute(1, 0, 2).reshape(-1, nheads * head_dim))

    result = torch.cat(y_list, dim=0).to(torch.bfloat16)
    return result


@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("max_seqlen", [32, 256, 512, 1024])
@pytest.mark.parametrize("nheads", [8])
@pytest.mark.parametrize("ngroups", [2])
@pytest.mark.parametrize("head_dim", [80])
@pytest.mark.parametrize("dstate", [128])
@pytest.mark.parametrize("d_conv", [4])
def test_mamba_prefill_by_step(batch_size, max_seqlen, nheads, ngroups, head_dim, dstate, d_conv):
    torch.manual_seed(0)
    num_max_batch = 256
    chunk_size = 256
    torch.random.manual_seed(0)
    seqlens = torch.randint(1, max_seqlen, (batch_size,), dtype=torch.int32, device="cuda")
    total_seqlen = seqlens.sum()
    x = torch.randn(total_seqlen, nheads, head_dim, dtype=torch.bfloat16, device="cuda")
    x.normal_(0, 0.1)
    dt = torch.randn(total_seqlen, nheads, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(total_seqlen, ngroups, dstate, dtype=torch.bfloat16, device="cuda")
    C = torch.randn(total_seqlen, ngroups, dstate, dtype=torch.bfloat16, device="cuda")
    z = torch.randn(total_seqlen, nheads, head_dim, dtype=torch.bfloat16, device="cuda")
    zxbcdt = torch.cat(
        (
            z.reshape(total_seqlen, -1),
            x.reshape(total_seqlen, -1),
            B.reshape(total_seqlen, -1),
            C.reshape(total_seqlen, -1),
            dt.reshape(total_seqlen, -1),
        ),
        dim=1,
    ).contiguous()
    conv_dim = nheads * head_dim + ngroups * dstate * 2
    A = -torch.rand(nheads, dtype=torch.float32).cuda() - 1.0
    D = torch.rand(nheads, dtype=torch.float32).cuda()
    dt_bias = torch.rand(nheads, dtype=torch.float32).cuda() - 4.0
    weight = torch.randn((d_conv, conv_dim), dtype=torch.bfloat16).cuda()
    weight.normal_(0, 0.1)
    bias = torch.randn((conv_dim), dtype=torch.bfloat16).cuda()

    conv_states = torch.rand((num_max_batch, d_conv - 1, conv_dim), dtype=torch.bfloat16).cuda()
    indices = torch.randperm(num_max_batch)[:batch_size].to(torch.int32).cuda()
    ssm_states = torch.rand((num_max_batch, nheads, head_dim, dstate), dtype=torch.float32).cuda()

    cu_seqlens = (
        torch.cat(
            [torch.zeros(1).to(torch.device("cuda")), torch.cumsum(seqlens, dim=0)],
            dim=0,
        )
        .to(torch.int32)
        .to(torch.device("cuda"))
    )

    seq_idx = torch.repeat_interleave(
        torch.arange(batch_size, dtype=torch.int32, device=cu_seqlens.device),
        cu_seqlens.diff(),
        output_size=cu_seqlens[-1],
    ).unsqueeze(0)

    trt_gt_xbc, trt_gt_conv_states = causal_conv1d_prefill_ref(
        torch.cat(
            (
                x.reshape(total_seqlen, -1),
                B.reshape(total_seqlen, -1),
                C.reshape(total_seqlen, -1),
            ),
            dim=1,
        ).contiguous(),
        weight,
        bias,
        conv_states,
        indices,
        cu_seqlens,
        seqlens,
    )
    # ground_truth
    trt_gt_y, trt_gt_ssm_states = mamba_prefill_torch_ref(
        trt_gt_xbc[:, : nheads * head_dim].clone().reshape(-1, nheads, head_dim).float(),
        dt.clone().float(),
        A.clone().float(),
        trt_gt_xbc[:, nheads * head_dim : nheads * head_dim + ngroups * dstate]
        .clone()
        .reshape(-1, ngroups, dstate)
        .float(),
        trt_gt_xbc[
            :, nheads * head_dim + ngroups * dstate : nheads * head_dim + 2 * ngroups * dstate
        ]
        .clone()
        .reshape(-1, ngroups, dstate)
        .float(),
        D.clone(),
        z.clone().float(),
        dt_bias.clone(),
        cu_seqlens,
        seq_idx,
    )
    trt_gt_y = trt_gt_y.reshape(-1, nheads * head_dim).to(torch.bfloat16)

    host_split_metadata = torch.empty(
        (6, batch_size + 1), dtype=torch.int32, device="cpu", pin_memory=True
    )
    # Step 1 exp dA cumsum
    # exp_dA_cumsum: [nheads, total_seqlen]
    yscale, xscale, split_metadata, total_chunks, tma_desc, y = hpc.mamba_exp_dA_chunked_cumsum(
        zxbcdt,
        A,
        dt_bias,
        seqlens.cpu(),
        host_split_metadata,
        chunk_size,
        head_dim,
        ngroups,
        dstate,
    )

    gt_yscale, gt_xscale, gt_split_metadata, gt_total_chunks = exp_dA_chunked_cumsum_torch_ref(
        seqlens, dt, A, dt_bias, chunk_size
    )

    assert allclose(gt_split_metadata, split_metadata)
    assert total_chunks == gt_total_chunks
    for bi in range(batch_size):
        assert allclose(
            gt_yscale[:, split_metadata[1][bi] : split_metadata[1][bi] + split_metadata[3][bi]],
            yscale[:, split_metadata[1][bi] : split_metadata[1][bi] + split_metadata[3][bi]],
            rtol=1e-3,
        )
        assert allclose(
            gt_xscale[:, split_metadata[1][bi] : split_metadata[1][bi] + split_metadata[3][bi]],
            xscale[:, split_metadata[1][bi] : split_metadata[1][bi] + split_metadata[3][bi]],
            rtol=5e-3,
        )

    # Step 2. conv1d prefill
    gt_xbc, gt_conv_states, gt_scaled_x = causal_conv1d_prefill_with_scale_torch_ref(
        zxbcdt[:, nheads * head_dim : -nheads],
        weight,
        bias,
        conv_states,
        indices,
        xscale,
        yscale,
        split_metadata,
        nheads,
        head_dim,
        chunk_size,
    )
    hpc.causal_conv1d_prefill(
        y,
        zxbcdt,
        conv_states,
        weight,
        bias,
        indices,
        split_metadata,
        xscale,
        yscale,
        chunk_size,
        total_chunks,
        nheads * head_dim,
        nheads,
    )
    assert allclose(gt_conv_states, conv_states, atol=1e-5, rtol=1e-3)
    assert allclose(gt_xbc, zxbcdt[:, nheads * head_dim : -nheads], atol=1e-5, rtol=1e-2)
    assert allclose(gt_scaled_x, y, atol=1e-5, rtol=1e-2)

    # Step 2. chunked state
    x = zxbcdt[:, nheads * head_dim : 2 * nheads * head_dim].clone().reshape(-1, nheads, head_dim)
    B = (
        zxbcdt[:, 2 * nheads * head_dim : 2 * nheads * head_dim + ngroups * dstate]
        .clone()
        .reshape(-1, ngroups, dstate)
    )
    gt_chunk_states = chunked_states_torch_ref(B, y, split_metadata, chunk_size, nheads, head_dim)
    chunk_states = hpc.mamba_chunk_states_bmm(
        zxbcdt,
        y,
        split_metadata,
        tma_desc,
        total_chunks,
        nheads,
        ngroups,
        head_dim,
        dstate,
    )
    assert allclose(gt_chunk_states, chunk_states, atol=2e-3, rtol=1e-2)

    # Step 3 state passing
    gt_chunk_states_cumsum, gt_ssm_states_result = state_passing_torch_ref(
        chunk_states, yscale, split_metadata, ssm_states, indices, chunk_size
    )
    chunk_states_cumsum = hpc.mamba_chunk_states_passing(
        chunk_states,
        yscale,
        split_metadata,
        ssm_states,
        indices,
        chunk_size,
    )
    assert allclose(gt_ssm_states_result, ssm_states)
    assert allclose(gt_chunk_states_cumsum, chunk_states_cumsum, rtol=1e-2)

    # Step 4 compute pre_y
    C = (
        zxbcdt[
            :,
            2 * nheads * head_dim + ngroups * dstate : 2 * nheads * head_dim + 2 * ngroups * dstate,
        ]
        .clone()
        .reshape(-1, ngroups, dstate)
    )
    gt_pre_y = pre_y_bmm_torch_ref(chunk_states_cumsum, C, split_metadata, chunk_size)
    pre_y = hpc.mamba_pre_y_bmm(
        chunk_states_cumsum, zxbcdt, split_metadata, tma_desc, ngroups, chunk_size
    )

    if pre_y.shape[1] > 0:
        for bi in range(batch_size):
            ichunk = split_metadata[2][bi + 1] - (bi + 1) - 1
            ilast = split_metadata[5][bi] * chunk_size - split_metadata[3][bi]
            gt_pre_y[:, ichunk, -ilast:, :] = pre_y[:, ichunk, -ilast:, :]
        assert allclose(gt_pre_y, pre_y, rtol=1e-3, atol=1e-5)

    # Step 5 chunk scan
    x = zxbcdt[:, nheads * head_dim : 2 * nheads * head_dim].clone().reshape(-1, nheads, head_dim)
    B = (
        zxbcdt[:, 2 * nheads * head_dim : 2 * nheads * head_dim + ngroups * dstate]
        .clone()
        .reshape(-1, ngroups, dstate)
    )
    C = (
        zxbcdt[
            :,
            2 * nheads * head_dim + ngroups * dstate : 2 * nheads * head_dim + 2 * ngroups * dstate,
        ]
        .clone()
        .reshape(-1, ngroups, dstate)
    )

    gt_y = chun_scan_torch_ref(B, C, x, pre_y, z, xscale, yscale, D, split_metadata, chunk_size)

    y = hpc.mamba_chunk_scan_gem3(
        y,
        zxbcdt,
        pre_y,
        xscale,
        yscale,
        D,
        split_metadata,
        tma_desc,
        ngroups,
        dstate,
        chunk_size,
    )
    norm_y = F.normalize(y, dim=-1)
    norm_gt_y = F.normalize(gt_y, dim=-1)
    # import pdb;pdb.set_trace
    assert allclose(norm_gt_y, norm_y, rtol=2e-2, atol=1e-1)
    assert allclose(trt_gt_y, y, rtol=2e-2, atol=1e-1)
    assert allclose(trt_gt_conv_states[indices], conv_states[indices], rtol=1e-3, atol=1e-5)
    assert allclose(trt_gt_ssm_states.transpose(-1, -2), ssm_states[indices], rtol=1e-2, atol=2e-3)


@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("max_seqlen", [32, 256, 512, 1024])
@pytest.mark.parametrize("nheads", [8])
@pytest.mark.parametrize("ngroups", [2])
@pytest.mark.parametrize("head_dim", [80])
@pytest.mark.parametrize("dstate", [128])
@pytest.mark.parametrize("d_conv", [4])
def test_mamba_prefill(batch_size, max_seqlen, nheads, ngroups, head_dim, dstate, d_conv):
    torch.manual_seed(0)
    num_max_batch = 256
    chunk_size = 256
    torch.random.manual_seed(0)
    seqlens = torch.randint(1, max_seqlen, (batch_size,), dtype=torch.int32, device="cuda")
    total_seqlen = seqlens.sum()
    x = torch.randn(total_seqlen, nheads, head_dim, dtype=torch.bfloat16, device="cuda")
    x.normal_(0, 0.1)
    dt = torch.randn(total_seqlen, nheads, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(total_seqlen, ngroups, dstate, dtype=torch.bfloat16, device="cuda")
    C = torch.randn(total_seqlen, ngroups, dstate, dtype=torch.bfloat16, device="cuda")
    z = torch.randn(total_seqlen, nheads, head_dim, dtype=torch.bfloat16, device="cuda")
    zxbcdt = torch.cat(
        (
            z.reshape(total_seqlen, -1),
            x.reshape(total_seqlen, -1),
            B.reshape(total_seqlen, -1),
            C.reshape(total_seqlen, -1),
            dt.reshape(total_seqlen, -1),
        ),
        dim=1,
    ).contiguous()
    conv_dim = nheads * head_dim + ngroups * dstate * 2
    A = -torch.rand(nheads, dtype=torch.float32).cuda() - 1.0
    D = torch.rand(nheads, dtype=torch.float32).cuda()
    dt_bias = torch.rand(nheads, dtype=torch.float32).cuda() - 4.0
    weight = torch.randn((d_conv, conv_dim), dtype=torch.bfloat16).cuda()
    weight.normal_(0, 0.1)
    bias = torch.randn((conv_dim), dtype=torch.bfloat16).cuda()

    conv_states = torch.rand((num_max_batch, d_conv - 1, conv_dim), dtype=torch.bfloat16).cuda()
    indices = torch.randperm(num_max_batch)[:batch_size].to(torch.int32).cuda()
    ssm_states = torch.rand((num_max_batch, nheads, head_dim, dstate), dtype=torch.float32).cuda()

    cu_seqlens = (
        torch.cat(
            [torch.zeros(1).to(torch.device("cuda")), torch.cumsum(seqlens, dim=0)],
            dim=0,
        )
        .to(torch.int32)
        .to(torch.device("cuda"))
    )

    seq_idx = torch.repeat_interleave(
        torch.arange(batch_size, dtype=torch.int32, device=cu_seqlens.device),
        cu_seqlens.diff(),
        output_size=cu_seqlens[-1],
    ).unsqueeze(0)

    trt_gt_xbc, trt_gt_conv_states = causal_conv1d_prefill_ref(
        torch.cat(
            (
                x.reshape(total_seqlen, -1),
                B.reshape(total_seqlen, -1),
                C.reshape(total_seqlen, -1),
            ),
            dim=1,
        ).contiguous(),
        weight,
        bias,
        conv_states,
        indices,
        cu_seqlens,
        seqlens,
    )
    # ground_truth
    trt_gt_y, trt_gt_ssm_states = mamba_prefill_torch_ref(
        trt_gt_xbc[:, : nheads * head_dim].clone().reshape(-1, nheads, head_dim).float(),
        dt.clone().float(),
        A.clone().float(),
        trt_gt_xbc[:, nheads * head_dim : nheads * head_dim + ngroups * dstate]
        .clone()
        .reshape(-1, ngroups, dstate)
        .float(),
        trt_gt_xbc[
            :, nheads * head_dim + ngroups * dstate : nheads * head_dim + 2 * ngroups * dstate
        ]
        .clone()
        .reshape(-1, ngroups, dstate)
        .float(),
        D.clone(),
        z.clone().float(),
        dt_bias.clone(),
        cu_seqlens,
        seq_idx,
    )
    trt_gt_y = trt_gt_y.reshape(-1, nheads * head_dim).to(torch.bfloat16)

    host_split_metadata = torch.empty(
        (6, batch_size + 1), dtype=torch.int32, device="cpu", pin_memory=True
    )

    y = hpc.mamba_prefill(
        zxbcdt,
        conv_states,
        ssm_states,
        indices,
        seqlens.cpu(),
        host_split_metadata,
        weight,
        bias,
        A,
        D,
        dt_bias,
        ngroups,
        chunk_size,
    )
    assert allclose(trt_gt_y, y, rtol=2e-2, atol=1e-1)
    assert allclose(trt_gt_conv_states[indices], conv_states[indices], rtol=1e-3, atol=1e-5)
    assert allclose(trt_gt_ssm_states.transpose(-1, -2), ssm_states[indices], rtol=1e-2, atol=2e-3)
