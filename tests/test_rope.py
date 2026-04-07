#!/usr/bin/env python3
"""
Comprehensive test suite for RoPE operation with blocked KV cache.
Tests both prefill and decoding modes, with and without QK normalization.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import hpc
from utils import allclose


def generate_cos_sin_cache(max_position, head_dim, base=10000.0):
    """Generate RoPE cos/sin cache."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_position).float()
    freqs = torch.outer(t, inv_freq)  # [max_position, head_dim/2]

    cos = freqs.cos()  # [max_position, head_dim/2]
    sin = freqs.sin()  # [max_position, head_dim/2]

    # Concatenate cos and sin: [max_position, head_dim]
    cos_sin = torch.cat([cos, sin], dim=-1)
    return cos_sin


def generate_kv_block_indices(kcache, req_length: list):

    num_req = len(req_length)
    total_blocks_in_pool = kcache.shape[0]
    kv_block_size = kcache.shape[1]
    num_blocks_per_req = [(length + kv_block_size - 1) // kv_block_size for length in req_length]
    total_blocks_used = sum(num_blocks_per_req)
    max_blocks_used = max(num_blocks_per_req)

    shuffled_blocks = torch.randperm(total_blocks_in_pool)

    # +4 for testing
    kv_indices = torch.ones(num_req, max_blocks_used + 4, dtype=torch.int32) * -1

    block_offset = 0
    for i in range(num_req):
        kv_indices[i, : num_blocks_per_req[i]] = shuffled_blocks[
            block_offset : block_offset + num_blocks_per_req[i]
        ]
        block_offset += num_blocks_per_req[i]

    assert block_offset == total_blocks_used

    return kv_indices


def apply_rms_norm_reference(x, weight, eps=1e-6):
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return x_normed * weight


def apply_rotary_pos_emb_neox_reference(x, cos_sin):
    num_tokens, num_heads, head_dim = x.shape
    half_dim = head_dim // 2

    # Split x into two halves
    x1 = x[..., :half_dim]  # [num_tokens, num_heads, half_dim]
    x2 = x[..., half_dim:]  # [num_tokens, num_heads, half_dim]

    # Extract cos and sin from cos_sin tensor
    cos_half = cos_sin[:, :half_dim].unsqueeze(1)  # [num_tokens, 1, half_dim]
    sin_half = cos_sin[:, half_dim:].unsqueeze(1)  # [num_tokens, 1, half_dim]

    # Apply rotation (neox version)
    o1 = x1 * cos_half - x2 * sin_half
    o2 = x2 * cos_half + x1 * sin_half

    # Concatenate
    output = torch.cat([o1, o2], dim=-1)
    return output


def prepare_prefill_input(
    num_req,
    req_length,
    num_q_heads,
    num_kv_heads,
    qk_head_dim,
    v_head_dim,
    kv_block_size,
    max_num_kv_blocks,
    max_rope_position,
    dtype=torch.bfloat16,
    device="cuda",
):

    if req_length is None:
        req_length = torch.randint(20, 200, (num_req,)).tolist()
    if isinstance(req_length, int):
        req_length = [req_length] * num_req
    total_rows = sum(req_length)
    qkv = torch.randn(
        total_rows,
        num_q_heads * qk_head_dim + num_kv_heads * qk_head_dim + num_kv_heads * v_head_dim,
        dtype=dtype,
        device=device,
    )
    cos_sin = generate_cos_sin_cache(max_rope_position, qk_head_dim).to(
        dtype=torch.float32, device=device
    )
    kcache = torch.randn(
        max_num_kv_blocks, kv_block_size, num_kv_heads, qk_head_dim, dtype=dtype, device=device
    )
    vcache = torch.randn(
        max_num_kv_blocks, kv_block_size, num_kv_heads, v_head_dim, dtype=dtype, device=device
    )

    kv_indices = generate_kv_block_indices(kcache, req_length).to(device)

    q_norm_weight = torch.randn(qk_head_dim, dtype=torch.float32, device=device)
    k_norm_weight = torch.randn(qk_head_dim, dtype=torch.float32, device=device)

    num_seqlen_per_req = torch.tensor(req_length, dtype=torch.int32, device=device)

    return (
        qkv,
        num_seqlen_per_req,
        cos_sin,
        kcache,
        vcache,
        kv_indices,
        q_norm_weight,
        k_norm_weight,
    )


def prepare_decode_input(
    num_req,
    req_length,
    num_q_heads,
    num_kv_heads,
    qk_head_dim,
    v_head_dim,
    kv_block_size,
    max_num_kv_blocks,
    max_rope_position,
    dtype=torch.bfloat16,
    device="cuda",
):
    if req_length is None:
        req_length = torch.randint(20, 200, (num_req,)).tolist()
    if isinstance(req_length, int):
        req_length = [req_length] * num_req
    # input req length is the existing length, not the new length, we add 1 here for kvcache update
    req_length = [x + 1 for x in req_length]
    total_rows = num_req
    qkv = torch.randn(
        total_rows,
        num_q_heads * qk_head_dim + num_kv_heads * qk_head_dim + num_kv_heads * v_head_dim,
        dtype=dtype,
        device=device,
    )
    cos_sin = generate_cos_sin_cache(max_rope_position, qk_head_dim).to(
        dtype=torch.float32, device=device
    )
    kcache = torch.randn(
        max_num_kv_blocks, kv_block_size, num_kv_heads, qk_head_dim, dtype=dtype, device=device
    )
    vcache = torch.randn(
        max_num_kv_blocks, kv_block_size, num_kv_heads, v_head_dim, dtype=dtype, device=device
    )

    kv_indices = generate_kv_block_indices(kcache, req_length).cuda()

    q_norm_weight = torch.randn(qk_head_dim, dtype=torch.float32, device=device)
    k_norm_weight = torch.randn(qk_head_dim, dtype=torch.float32, device=device)

    num_seqlen_per_req = torch.tensor(req_length, dtype=torch.int32, device=device)

    return (
        qkv,
        num_seqlen_per_req,
        cos_sin,
        kcache,
        vcache,
        kv_indices,
        q_norm_weight,
        k_norm_weight,
    )


def torch_rope_norm_blocked_prefill(
    kcache,
    vcache,
    qkv,
    cos_sin,
    num_seqlen_per_req,
    q_index,
    kv_indices,
    is_prefill=True,
    use_qknorm=False,
    q_norm_weight=None,
    k_norm_weight=None,
    qk_norm_policy=1,
    clear_kv_tail=False,
):
    """Test RoPE prefill mode with PyTorch reference implementation."""
    assert is_prefill
    assert (
        q_index.shape[0] == num_seqlen_per_req.shape[0] + 1
    )  # q_index is a prefix sum of each len
    dtype = qkv.dtype
    num_kv_heads = kcache.shape[2]
    v_head_dim = vcache.shape[3]
    qk_head_dim = kcache.shape[3]
    num_q_heads = (
        qkv.shape[1] - num_kv_heads * qk_head_dim - num_kv_heads * v_head_dim
    ) // qk_head_dim
    q_seq_lens = (q_index[1:] - q_index[:-1]).tolist()

    num_rows = q_index[-1].item()
    num_req = num_seqlen_per_req.shape[0]
    q_input = qkv[:, : num_q_heads * qk_head_dim].to(torch.float32)
    k_input = qkv[
        :, num_q_heads * qk_head_dim : num_q_heads * qk_head_dim + num_kv_heads * qk_head_dim
    ].to(torch.float32)
    v_input = qkv[:, num_q_heads * qk_head_dim + num_kv_heads * qk_head_dim :]

    # Build cos_sin for each token
    cos_sin_for_tokens = torch.zeros(num_rows, qk_head_dim, dtype=torch.float32, device="cuda")
    token_offset = 0
    for batch_idx in range(num_req):
        seq_len = num_seqlen_per_req[batch_idx].item()
        q_seq_len = q_seq_lens[batch_idx]
        cos_sin_for_tokens[token_offset : token_offset + q_seq_len] = cos_sin[
            seq_len - q_seq_len : seq_len
        ]
        token_offset += q_seq_len

    q_ref = q_input.view(num_rows, num_q_heads, qk_head_dim)
    k_ref = k_input.view(num_rows, num_kv_heads, qk_head_dim)
    v_ref = v_input.view(num_rows, num_kv_heads, v_head_dim)
    # Compute reference Q and K

    if use_qknorm and qk_norm_policy == 2:
        q_ref = apply_rms_norm_reference(q_ref, q_norm_weight)
        k_ref = apply_rms_norm_reference(k_ref, k_norm_weight)

    q_ref = apply_rotary_pos_emb_neox_reference(q_ref, cos_sin_for_tokens)
    k_ref = apply_rotary_pos_emb_neox_reference(k_ref, cos_sin_for_tokens)

    if use_qknorm and qk_norm_policy == 1:
        q_ref = apply_rms_norm_reference(q_ref, q_norm_weight)
        k_ref = apply_rms_norm_reference(k_ref, k_norm_weight)

    # update kvcache
    kv_block_size = kcache.shape[1]
    token_idx = 0
    # breakpoint()
    for req_idx in range(num_req):
        seq_len = num_seqlen_per_req[req_idx].item()
        q_seq_len = q_seq_lens[req_idx]
        for pos_in_seq in range(seq_len - q_seq_len, seq_len):
            block_idx_in_req = pos_in_seq // kv_block_size
            pos_in_block = pos_in_seq % kv_block_size
            cache_block_idx = kv_indices[req_idx, block_idx_in_req].item()
            assert cache_block_idx >= 0, f"Invalid cache block index: {cache_block_idx}"
            # Update K cache
            kcache[cache_block_idx, pos_in_block, :, :] = k_ref[token_idx, :, :].to(dtype)
            # Update V cache
            vcache[cache_block_idx, pos_in_block, :, :] = v_ref[token_idx, :, :].to(dtype)
            # Clear rows [pos_in_block+1, kv_block_size) for last token of each request
            if clear_kv_tail and pos_in_seq == seq_len - 1 and pos_in_block + 1 < kv_block_size:
                kcache[cache_block_idx, pos_in_block + 1 :, :, :] = 0
                vcache[cache_block_idx, pos_in_block + 1 :, :, :] = 0
            token_idx += 1

    out_q = q_ref.to(dtype)
    out_k = k_ref.to(dtype)
    return out_q, out_k, kcache, vcache


def torch_rope_norm_blocked_decode(
    kcache,
    vcache,
    qkv,
    cos_sin,
    num_seqlen_per_req,
    q_index,
    kv_indices,
    is_prefill=False,
    use_qknorm=False,
    q_norm_weight=None,
    k_norm_weight=None,
    qk_norm_policy=1,
    clear_kv_tail=False,
):
    """Test RoPE decode mode with PyTorch reference implementation."""
    assert not is_prefill
    dtype = qkv.dtype
    num_kv_heads = kcache.shape[2]
    v_head_dim = vcache.shape[3]
    qk_head_dim = kcache.shape[3]
    num_q_heads = (
        qkv.shape[1] - num_kv_heads * qk_head_dim - num_kv_heads * v_head_dim
    ) // qk_head_dim

    num_req = num_seqlen_per_req.shape[0]
    num_rows = num_req
    q_input = qkv[:, : num_q_heads * qk_head_dim].to(torch.float32)
    k_input = qkv[
        :, num_q_heads * qk_head_dim : num_q_heads * qk_head_dim + num_kv_heads * qk_head_dim
    ].to(torch.float32)
    v_input = qkv[:, num_q_heads * qk_head_dim + num_kv_heads * qk_head_dim :]

    # Build cos_sin for each token
    cos_sin_for_tokens = torch.zeros(num_rows, qk_head_dim, dtype=torch.float32, device="cuda")
    for batch_idx in range(num_req):
        seq_len = num_seqlen_per_req[batch_idx].item()
        position = seq_len - 1
        cos_sin_for_tokens[batch_idx] = cos_sin[position]

    q_ref = q_input.view(num_rows, num_q_heads, qk_head_dim)
    k_ref = k_input.view(num_rows, num_kv_heads, qk_head_dim)
    v_ref = v_input.view(num_rows, num_kv_heads, v_head_dim)

    if use_qknorm and qk_norm_policy == 2:
        q_ref = apply_rms_norm_reference(q_ref, q_norm_weight)
        k_ref = apply_rms_norm_reference(k_ref, k_norm_weight)

    # Compute reference Q and K
    q_ref = apply_rotary_pos_emb_neox_reference(q_ref, cos_sin_for_tokens)
    k_ref = apply_rotary_pos_emb_neox_reference(k_ref, cos_sin_for_tokens)

    if use_qknorm and qk_norm_policy == 1:
        q_ref = apply_rms_norm_reference(q_ref, q_norm_weight)
        k_ref = apply_rms_norm_reference(k_ref, k_norm_weight)

    # update kvcache
    kv_block_size = kcache.shape[1]
    token_idx = 0
    for req_idx in range(num_req):
        seq_len = num_seqlen_per_req[req_idx].item()
        pos_in_seq = seq_len - 1
        block_idx_in_req = pos_in_seq // kv_block_size
        pos_in_block = pos_in_seq % kv_block_size
        cache_block_idx = kv_indices[req_idx, block_idx_in_req].item()
        assert cache_block_idx >= 0, f"Invalid cache block index: {cache_block_idx}"
        # Update K cache
        kcache[cache_block_idx, pos_in_block, :, :] = k_ref[token_idx, :, :].to(dtype)
        # Update V cache
        vcache[cache_block_idx, pos_in_block, :, :] = v_ref[token_idx, :, :].to(dtype)

        # Clear KV cache tail rows
        if clear_kv_tail:
            # New unified clearing: always clear [pos_in_block+1, kv_block_size)
            if pos_in_block + 1 < kv_block_size:
                kcache[cache_block_idx, pos_in_block + 1 :, :, :] = 0
                vcache[cache_block_idx, pos_in_block + 1 :, :, :] = 0
        else:
            # Old behavior: clear only when pos_in_block == 0
            if pos_in_block == 0:
                kcache[cache_block_idx, 1:, :, :] = 0
                vcache[cache_block_idx, 1:, :, :] = 0

        token_idx += 1

    out_q = q_ref.to(dtype)
    out_k = k_ref.to(dtype)
    # out_qkv = torch.cat([out_q, out_k, out_v], dim=1)
    return out_q, out_k, kcache, vcache


def sample_and_extract_qkv(req_length, qkv):

    device = qkv.device
    req_length = torch.tensor(req_length).to(device)
    batch_size = req_length.size(0)

    # rand a ratio
    rand_factors = torch.rand(batch_size, device=device)
    q_length = (rand_factors * req_length).long() + 1
    q_length = torch.min(q_length, req_length)

    # ensure not larger
    req_cumsum = torch.cumsum(req_length, dim=0)

    slices = []

    for i in range(batch_size):
        curr_original_end = req_cumsum[i].item()
        curr_new_len = q_length[i].item()
        slice_start = curr_original_end - curr_new_len
        slice_end = curr_original_end
        slices.append(qkv[slice_start:slice_end])

    qkv_new = torch.cat(slices, dim=0)

    q_cumsum = torch.cumsum(q_length, dim=0)

    # add a zero
    zero_pad = torch.tensor([0], device=device, dtype=q_cumsum.dtype)
    q_index = torch.cat((zero_pad, q_cumsum), dim=0)

    return q_index.to(torch.int32), qkv_new


@pytest.mark.parametrize("num_req", [7])
@pytest.mark.parametrize("num_q_head_head_dim", [(4, 128), (8, 128), (8, 80)])
@pytest.mark.parametrize("num_kv_heads", [1])
@pytest.mark.parametrize("use_qknorm", [False, True])
@pytest.mark.parametrize("qk_norm_policy", [1, 2])
def test_rope_norm_blocked_prefill(
    num_req, num_q_head_head_dim, num_kv_heads, use_qknorm, qk_norm_policy
):
    req_length = torch.randint(20, 200, (num_req,)).tolist()
    num_q_heads = num_q_head_head_dim[0]
    qk_head_dim = num_q_head_head_dim[1]
    v_head_dim = num_q_head_head_dim[1]
    kv_block_size = 64
    max_num_kv_blocks = 1024
    max_rope_position = 2048
    dtype = torch.bfloat16
    (
        qkv,
        num_seqlen_per_req,
        cos_sin,
        kcache,
        vcache,
        kv_indices,
        q_norm_weight,
        k_norm_weight,
    ) = prepare_prefill_input(
        num_req,
        req_length,
        num_q_heads,
        num_kv_heads,
        qk_head_dim,
        v_head_dim,
        kv_block_size,
        max_num_kv_blocks,
        max_rope_position,
        dtype,
    )
    q_index, qkv_new = sample_and_extract_qkv(req_length, qkv)

    # clone input for torch ref, incase inplace update
    qkv_ref = qkv_new.clone()
    kcache_ref = kcache.clone()
    vcache_ref = vcache.clone()

    my_out_q, _ = hpc.rope_norm_blocked_kvcache(
        kcache,
        vcache,
        qkv_new,
        cos_sin,
        num_seqlen_per_req,
        q_index,
        kv_indices,
        True,  # is_refill
        use_qknorm,
        q_norm_weight,
        k_norm_weight,
        qk_norm_policy=qk_norm_policy,
    )

    torch_out_q, torch_out_k, torch_kcache, torch_vcache = torch_rope_norm_blocked_prefill(
        kcache_ref,
        vcache_ref,
        qkv_ref,
        cos_sin,
        num_seqlen_per_req,
        q_index,
        kv_indices,
        is_prefill=True,
        use_qknorm=use_qknorm,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        qk_norm_policy=qk_norm_policy,
    )

    assert allclose(torch_out_q, my_out_q, atol=5e-2)
    assert allclose(torch_kcache, kcache, atol=5e-2)
    assert allclose(torch_vcache, vcache, atol=5e-2)


@pytest.mark.parametrize("num_req", [7])
@pytest.mark.parametrize("num_q_head_head_dim", [(4, 128), (8, 128), (8, 80)])
@pytest.mark.parametrize("num_kv_heads", [1])
@pytest.mark.parametrize("use_qknorm", [False, True])
@pytest.mark.parametrize("qk_norm_policy", [1, 2])
def test_rope_norm_blocked_decode(
    num_req, num_q_head_head_dim, num_kv_heads, use_qknorm, qk_norm_policy
):
    req_length = torch.randint(20, 200, (num_req,)).tolist()
    num_q_heads = num_q_head_head_dim[0]
    qk_head_dim = num_q_head_head_dim[1]
    v_head_dim = num_q_head_head_dim[1]
    kv_block_size = 64
    max_num_kv_blocks = 1024
    max_rope_position = 2048
    dtype = torch.bfloat16
    (
        qkv,
        num_seqlen_per_req,
        cos_sin,
        kcache,
        vcache,
        kv_indices,
        q_norm_weight,
        k_norm_weight,
    ) = prepare_decode_input(
        num_req,
        req_length,
        num_q_heads,
        num_kv_heads,
        qk_head_dim,
        v_head_dim,
        kv_block_size,
        max_num_kv_blocks,
        max_rope_position,
        dtype,
    )

    # clone input for torch ref, incase inplace update
    qkv_ref = qkv.clone()
    kcache_ref = kcache.clone()
    vcache_ref = vcache.clone()

    my_out_q, _ = hpc.rope_norm_blocked_kvcache(
        kcache,
        vcache,
        qkv,
        cos_sin,
        num_seqlen_per_req,
        num_seqlen_per_req,
        kv_indices,
        False,  # is_refill
        use_qknorm,
        q_norm_weight,
        k_norm_weight,
        qk_norm_policy=qk_norm_policy,
    )

    torch_out_q, torch_out_k, torch_kcache, torch_vcache = torch_rope_norm_blocked_decode(
        kcache_ref,
        vcache_ref,
        qkv_ref,
        cos_sin,
        num_seqlen_per_req,
        num_seqlen_per_req,
        kv_indices,
        is_prefill=False,
        use_qknorm=use_qknorm,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        qk_norm_policy=qk_norm_policy,
    )

    assert allclose(torch_out_q, my_out_q, atol=5e-2)
    assert allclose(torch_kcache, kcache, atol=5e-2)
    assert allclose(torch_vcache, vcache, atol=5e-2)


@pytest.mark.skipif(bool(os.getenv("SANITIZER_CHECK")), reason="skip sanitizer")
@pytest.mark.parametrize("num_req", [7])
@pytest.mark.parametrize("num_q_head_head_dim", [(8, 128)])
@pytest.mark.parametrize("num_kv_heads", [1])
@pytest.mark.parametrize("use_qknorm", [True])
@pytest.mark.parametrize("qk_norm_policy", [1, 2])
def test_rope_norm_prefill_fp8(
    num_req, num_q_head_head_dim, num_kv_heads, use_qknorm, qk_norm_policy
):
    req_length = torch.randint(20, 200, (num_req,)).tolist()
    num_q_heads = num_q_head_head_dim[0]
    qk_head_dim = num_q_head_head_dim[1]
    v_head_dim = num_q_head_head_dim[1]
    kv_block_size = 64
    max_num_kv_blocks = 1024
    max_rope_position = 2048
    dtype = torch.bfloat16
    (
        qkv,
        num_seqlen_per_req,
        cos_sin,
        kcache,
        vcache,
        kv_indices,
        q_norm_weight,
        k_norm_weight,
    ) = prepare_prefill_input(
        num_req,
        req_length,
        num_q_heads,
        num_kv_heads,
        qk_head_dim,
        v_head_dim,
        kv_block_size,
        max_num_kv_blocks,
        max_rope_position,
        dtype,
    )
    q_index, qkv_new = sample_and_extract_qkv(req_length, qkv)

    # clone input for torch ref, incase inplace update
    qkv_ref = qkv_new.clone()
    kcache_ref = kcache.clone()
    vcache_ref = vcache.clone()

    k_scale = torch.tensor([0.1], dtype=torch.float32, device=qkv_new.device)
    v_scale = torch.tensor([0.1], dtype=torch.float32, device=qkv_new.device)
    kcache_fp8 = kcache.to(torch.float8_e4m3fn)
    vcache_fp8 = vcache.to(torch.float8_e4m3fn)

    seqlens = q_index[1:] - q_index[:-1]
    max_seqlens = seqlens.max().item()

    out_kv = torch.zeros(
        (qkv_new.shape[0], num_kv_heads * qk_head_dim + num_kv_heads * v_head_dim),
        dtype=torch.float8_e4m3fn,
        device=qkv_new.device,
    )

    out_q_fp8, out_k_fp8, out_v_fp8, qk_scale, split_k_flag, out_attention, tma_tensor = (
        hpc.rope_norm_w8c8(
            q=qkv_new[:, : num_q_heads * qk_head_dim].reshape(-1, num_q_heads, qk_head_dim),
            k=qkv_new[
                :,
                num_q_heads * qk_head_dim : num_q_heads * qk_head_dim + num_kv_heads * qk_head_dim,
            ].reshape(-1, num_kv_heads, qk_head_dim),
            v=qkv_new[:, num_q_heads * qk_head_dim + num_kv_heads * qk_head_dim :].reshape(
                -1, num_kv_heads, v_head_dim
            ),
            cos_sin=cos_sin,
            num_seqlen_per_req=num_seqlen_per_req,
            q_index=q_index,
            is_prefill=True,  # is_refill
            max_seqlens=max_seqlens,
            k_scale=k_scale,
            v_scale=v_scale,
            qk_norm_policy=qk_norm_policy,
            q_norm_weight=q_norm_weight,
            k_norm_weight=k_norm_weight,
            out_k=out_kv[:, : num_kv_heads * qk_head_dim].reshape(-1, num_kv_heads, qk_head_dim),
            out_v=out_kv[:, num_kv_heads * qk_head_dim :].reshape(-1, num_kv_heads, v_head_dim),
        )
    )

    # convert q_scale to original format
    num_seq = seqlens.shape[0]
    mask = torch.arange(qk_scale.shape[2]).expand(
        qk_scale.shape[0], qk_scale.shape[2]
    ).cuda() < seqlens.unsqueeze(1)
    qk_scale_normal = qk_scale.permute(0, 2, 1)[mask].cuda()

    q_bf16 = (out_q_fp8.to(torch.bfloat16) * qk_scale_normal[:, :, None]).to(torch.bfloat16) * (
        1 / k_scale.to(torch.bfloat16)
    )

    torch_out_q, torch_out_k, torch_kcache, torch_vcache = torch_rope_norm_blocked_prefill(
        kcache_ref,
        vcache_ref,
        qkv_ref,
        cos_sin,
        num_seqlen_per_req,
        q_index,
        kv_indices,
        is_prefill=True,
        use_qknorm=use_qknorm,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        qk_norm_policy=qk_norm_policy,
    )

    assert allclose(torch_out_q, q_bf16, atol=0.5)


@pytest.mark.skipif(bool(os.getenv("SANITIZER_CHECK")), reason="skip sanitizer")
@pytest.mark.parametrize("num_req", [7])
@pytest.mark.parametrize("num_q_head_head_dim", [(4, 128)])
@pytest.mark.parametrize("num_kv_heads", [1])
@pytest.mark.parametrize("use_qknorm", [True])
@pytest.mark.parametrize("qk_norm_policy", [1, 2])
def test_rope_norm_blocked_prefill_fp8(
    num_req, num_q_head_head_dim, num_kv_heads, use_qknorm, qk_norm_policy
):
    req_length = torch.randint(20, 200, (num_req,)).tolist()
    num_q_heads = num_q_head_head_dim[0]
    qk_head_dim = num_q_head_head_dim[1]
    v_head_dim = num_q_head_head_dim[1]
    kv_block_size = 64
    max_num_kv_blocks = 1024
    max_rope_position = 2048
    dtype = torch.bfloat16
    (
        qkv,
        num_seqlen_per_req,
        cos_sin,
        kcache,
        vcache,
        kv_indices,
        q_norm_weight,
        k_norm_weight,
    ) = prepare_prefill_input(
        num_req,
        req_length,
        num_q_heads,
        num_kv_heads,
        qk_head_dim,
        v_head_dim,
        kv_block_size,
        max_num_kv_blocks,
        max_rope_position,
        dtype,
    )
    q_index, qkv_new = sample_and_extract_qkv(req_length, qkv)

    # clone input for torch ref, incase inplace update
    qkv_ref = qkv_new.clone()
    kcache_ref = kcache.clone()
    vcache_ref = vcache.clone()

    k_scale = torch.tensor([0.1], dtype=torch.float32, device=qkv_new.device)
    v_scale = torch.tensor([0.1], dtype=torch.float32, device=qkv_new.device)
    kcache_fp8 = kcache.to(torch.float8_e4m3fn)
    vcache_fp8 = vcache.to(torch.float8_e4m3fn)

    seqlens = q_index[1:] - q_index[:-1]
    max_seqlens = seqlens.max().item()

    q_fp8, k_fp8, qk_scale, split_k_flag, out_attention, tma_tensor = (
        hpc.rope_norm_blocked_kvcache_w8c8_dqskv(
            key_cache=kcache_fp8,
            value_cache=vcache_fp8,
            qkv=qkv_new,
            cos_sin=cos_sin,
            num_seqlen_per_req=num_seqlen_per_req,
            q_index=q_index,
            kvcache_indices=kv_indices,
            is_prefill=True,  # is_refill
            use_qk_norm=use_qknorm,
            max_seqlens=max_seqlens,
            k_scale=k_scale,
            v_scale=v_scale,
            q_norm_weight=q_norm_weight,
            k_norm_weight=k_norm_weight,
            qk_norm_policy=qk_norm_policy,
        )
    )

    # convert q_scale to original format
    num_seq = seqlens.shape[0]
    mask = torch.arange(qk_scale.shape[2]).expand(
        qk_scale.shape[0], qk_scale.shape[2]
    ).cuda() < seqlens.unsqueeze(1)
    qk_scale_normal = qk_scale.permute(0, 2, 1)[mask].cuda()

    q_bf16 = (q_fp8.to(torch.bfloat16) * qk_scale_normal[:, :, None]).to(torch.bfloat16) * (
        1 / k_scale.to(torch.bfloat16)
    )

    torch_out_q, torch_out_k, torch_kcache, torch_vcache = torch_rope_norm_blocked_prefill(
        kcache_ref,
        vcache_ref,
        qkv_ref,
        cos_sin,
        num_seqlen_per_req,
        q_index,
        kv_indices,
        is_prefill=True,
        use_qknorm=use_qknorm,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        qk_norm_policy=qk_norm_policy,
    )

    my_kcache = kcache_fp8.float()
    my_vcache = vcache_fp8.float()
    torch_kcache = torch_kcache.float()
    torch_vcache = torch_vcache.float()

    assert allclose(torch_out_q, q_bf16, atol=0.5)


@pytest.mark.parametrize("num_req", [7])
@pytest.mark.parametrize("num_q_head_head_dim", [(4, 128)])
@pytest.mark.parametrize("num_kv_heads", [1])
@pytest.mark.parametrize("use_qknorm", [True])
@pytest.mark.parametrize("qk_norm_policy", [1, 2])
def test_rope_norm_blocked_decode_fp8(
    num_req, num_q_head_head_dim, num_kv_heads, use_qknorm, qk_norm_policy
):
    req_length = torch.randint(20, 200, (num_req,)).tolist()
    num_q_heads = num_q_head_head_dim[0]
    qk_head_dim = num_q_head_head_dim[1]
    v_head_dim = num_q_head_head_dim[1]
    kv_block_size = 64
    max_num_kv_blocks = 1024
    max_rope_position = 2048
    dtype = torch.bfloat16
    (
        qkv,
        num_seqlen_per_req,
        cos_sin,
        kcache,
        vcache,
        kv_indices,
        q_norm_weight,
        k_norm_weight,
    ) = prepare_decode_input(
        num_req,
        req_length,
        num_q_heads,
        num_kv_heads,
        qk_head_dim,
        v_head_dim,
        kv_block_size,
        max_num_kv_blocks,
        max_rope_position,
        dtype,
    )

    # clone input for torch ref, incase inplace update
    qkv_ref = qkv.clone()
    kcache_ref = kcache.clone()
    vcache_ref = vcache.clone()

    k_scale = torch.tensor([0.1], dtype=torch.float32, device=qkv.device)
    v_scale = torch.tensor([0.1], dtype=torch.float32, device=qkv.device)
    kcache_fp8 = kcache.to(torch.float8_e4m3fn)
    vcache_fp8 = vcache.to(torch.float8_e4m3fn)

    q_fp8, k_fp8, qk_scale, split_k_flag, out_attention, tma_tensor = (
        hpc.rope_norm_blocked_kvcache_w8c8_dqskv(
            key_cache=kcache_fp8,
            value_cache=vcache_fp8,
            qkv=qkv,
            cos_sin=cos_sin,
            num_seqlen_per_req=num_seqlen_per_req,
            q_index=num_seqlen_per_req,
            kvcache_indices=kv_indices,
            is_prefill=False,  # is_prefill
            use_qk_norm=use_qknorm,
            max_seqlens=1,
            k_scale=k_scale,
            v_scale=v_scale,
            q_norm_weight=q_norm_weight,
            k_norm_weight=k_norm_weight,
            qk_norm_policy=qk_norm_policy,
        )
    )

    q_bf16 = (q_fp8.to(torch.bfloat16) * qk_scale[:, :, None]).to(
        torch.bfloat16
    )  # no k_scale in decode

    torch_out_q, torch_out_k, torch_kcache, torch_vcache = torch_rope_norm_blocked_decode(
        kcache_ref,
        vcache_ref,
        qkv_ref,
        cos_sin,
        num_seqlen_per_req,
        num_seqlen_per_req,
        kv_indices,
        is_prefill=False,
        use_qknorm=use_qknorm,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        qk_norm_policy=qk_norm_policy,
    )

    my_kcache = kcache_fp8.float()
    my_vcache = vcache_fp8.float()
    torch_kcache = torch_kcache.float()
    torch_vcache = torch_vcache.float()

    assert allclose(torch_out_q, q_bf16, atol=0.5)


@pytest.mark.parametrize("num_req", [7])
@pytest.mark.parametrize(
    "num_q_heads,num_kv_heads,qk_head_dim",
    [(8, 1, 128), (64, 8, 128)],
)
@pytest.mark.parametrize("qk_norm_policy", [1])
def test_rope_norm_store_kv_prefill(
    num_req, num_q_heads, num_kv_heads, qk_head_dim, qk_norm_policy
):
    req_length = torch.randint(20, 200, (num_req,)).tolist()
    v_head_dim = qk_head_dim
    kv_block_size = 64
    max_num_kv_blocks = 1024
    max_rope_position = 2048
    dtype = torch.bfloat16
    (
        qkv,
        num_seqlen_per_req,
        cos_sin,
        kcache,
        vcache,
        kv_indices,
        q_norm_weight,
        k_norm_weight,
    ) = prepare_prefill_input(
        num_req,
        req_length,
        num_q_heads,
        num_kv_heads,
        qk_head_dim,
        v_head_dim,
        kv_block_size,
        max_num_kv_blocks,
        max_rope_position,
        dtype,
    )
    q_index, qkv_new = sample_and_extract_qkv(req_length, qkv)

    qkv_ref = qkv_new.clone()
    kcache_ref = kcache.clone()
    vcache_ref = vcache.clone()

    my_out_q = hpc.rope_norm_store_kv(
        kcache,
        vcache,
        qkv_new,
        cos_sin,
        num_seqlen_per_req,
        q_index,
        kv_indices,
        True,
        q_norm_weight=q_norm_weight if qk_norm_policy > 0 else None,
        k_norm_weight=k_norm_weight if qk_norm_policy > 0 else None,
        qk_norm_policy=qk_norm_policy,
    )

    torch_out_q, torch_out_k, torch_kcache, torch_vcache = torch_rope_norm_blocked_prefill(
        kcache_ref,
        vcache_ref,
        qkv_ref,
        cos_sin,
        num_seqlen_per_req,
        q_index,
        kv_indices,
        is_prefill=True,
        use_qknorm=(qk_norm_policy > 0),
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        qk_norm_policy=qk_norm_policy,
        clear_kv_tail=True,
    )

    assert allclose(torch_out_q, my_out_q, atol=8e-2)
    assert allclose(torch_kcache, kcache, atol=8e-2)
    assert allclose(torch_vcache, vcache, atol=8e-2)


def pad_decode_inputs_to_align8(qkv, num_seqlen_per_req, q_index, kv_indices):
    """Pad decode inputs so total rows and num_batch are aligned to 8.
    Simulates CUDA graph padding: extra batches have q_index[i+1]-q_index[i]=0
    and num_seqlen_per_req[i]=0.
    """
    num_rows = qkv.shape[0]
    num_batch = num_seqlen_per_req.shape[0]
    hidden = qkv.shape[1]

    padded_batch = (num_batch + 7) // 8 * 8
    pad_batch = padded_batch - num_batch
    padded_rows = (num_rows + 7) // 8 * 8
    pad_rows = padded_rows - num_rows

    if pad_rows > 0:
        qkv = torch.cat([qkv, torch.zeros(pad_rows, hidden, dtype=qkv.dtype, device=qkv.device)])

    if pad_batch > 0:
        num_seqlen_per_req = torch.cat(
            [
                num_seqlen_per_req,
                torch.zeros(
                    pad_batch, dtype=num_seqlen_per_req.dtype, device=num_seqlen_per_req.device
                ),
            ]
        )

    # q_index: original ends at num_rows, padding batches have 0 tokens each,
    # but we assign all pad_rows to the first padding batch so q_index covers padded_rows
    last_val = q_index[-1]  # == num_rows
    if pad_batch > 0:
        pad_q = torch.full((pad_batch,), padded_rows, dtype=q_index.dtype, device=q_index.device)
        q_index = torch.cat([q_index, pad_q])

    if pad_batch > 0:
        kv_indices = torch.cat(
            [
                kv_indices,
                torch.zeros(
                    pad_batch, kv_indices.shape[1], dtype=kv_indices.dtype, device=kv_indices.device
                ),
            ]
        )

    return qkv, num_seqlen_per_req, q_index, kv_indices, num_rows


@pytest.mark.parametrize("num_req", [8])
@pytest.mark.parametrize(
    "num_q_heads,num_kv_heads,qk_head_dim",
    [(8, 1, 128), (64, 8, 128)],
)
@pytest.mark.parametrize("qk_norm_policy", [1])
def test_rope_norm_store_kv_decode(num_req, num_q_heads, num_kv_heads, qk_head_dim, qk_norm_policy):
    req_length = torch.randint(20, 200, (num_req,)).tolist()
    v_head_dim = qk_head_dim
    kv_block_size = 64
    max_num_kv_blocks = 1024
    max_rope_position = 2048
    dtype = torch.bfloat16
    (
        qkv,
        num_seqlen_per_req,
        cos_sin,
        kcache,
        vcache,
        kv_indices,
        q_norm_weight,
        k_norm_weight,
    ) = prepare_decode_input(
        num_req,
        req_length,
        num_q_heads,
        num_kv_heads,
        qk_head_dim,
        v_head_dim,
        kv_block_size,
        max_num_kv_blocks,
        max_rope_position,
        dtype,
    )

    q_index_decode = torch.arange(num_req + 1, dtype=torch.int32, device=qkv.device)

    # Pad to align-8 (simulates CUDA graph padding)
    qkv, num_seqlen_per_req, q_index_decode, kv_indices, real_rows = pad_decode_inputs_to_align8(
        qkv, num_seqlen_per_req, q_index_decode, kv_indices
    )

    qkv_ref = qkv[:real_rows].clone()
    kcache_ref = kcache.clone()
    vcache_ref = vcache.clone()
    q_index_ref = torch.arange(num_req + 1, dtype=torch.int32, device=qkv.device)

    my_out_q = hpc.rope_norm_store_kv(
        kcache,
        vcache,
        qkv,
        cos_sin,
        num_seqlen_per_req,
        q_index_decode,
        kv_indices,
        False,  # is prefill
        q_norm_weight=q_norm_weight if qk_norm_policy > 0 else None,
        k_norm_weight=k_norm_weight if qk_norm_policy > 0 else None,
        qk_norm_policy=qk_norm_policy,
    )

    torch_out_q, torch_out_k, torch_kcache, torch_vcache = torch_rope_norm_blocked_decode(
        kcache_ref,
        vcache_ref,
        qkv_ref,
        cos_sin,
        num_seqlen_per_req[:num_req],
        q_index_ref,
        kv_indices[:num_req],
        is_prefill=False,
        use_qknorm=(qk_norm_policy > 0),
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        qk_norm_policy=qk_norm_policy,
        clear_kv_tail=True,
    )

    assert allclose(torch_out_q, my_out_q[:real_rows], atol=5e-2)
    assert allclose(torch_kcache, kcache, atol=5e-2)
    assert allclose(torch_vcache, vcache, atol=5e-2)


@pytest.mark.skipif(bool(os.getenv("SANITIZER_CHECK")), reason="skip sanitizer")
@pytest.mark.parametrize("num_req", [7])
@pytest.mark.parametrize(
    "num_q_heads,num_kv_heads,qk_head_dim",
    [(8, 1, 128), (64, 8, 128)],
)
@pytest.mark.parametrize("qk_norm_policy", [1, 2])
def test_rope_norm_store_kv_fp8_prefill_dqskv(
    num_req, num_q_heads, num_kv_heads, qk_head_dim, qk_norm_policy
):
    req_length = torch.randint(20, 200, (num_req,)).tolist()
    v_head_dim = qk_head_dim
    kv_block_size = 64
    max_num_kv_blocks = 1024
    max_rope_position = 2048
    dtype = torch.bfloat16
    (
        qkv,
        num_seqlen_per_req,
        cos_sin,
        kcache,
        vcache,
        kv_indices,
        q_norm_weight,
        k_norm_weight,
    ) = prepare_prefill_input(
        num_req,
        req_length,
        num_q_heads,
        num_kv_heads,
        qk_head_dim,
        v_head_dim,
        kv_block_size,
        max_num_kv_blocks,
        max_rope_position,
        dtype,
    )
    q_index, qkv_new = sample_and_extract_qkv(req_length, qkv)

    qkv_ref = qkv_new.clone()
    kcache_ref = kcache.clone()
    vcache_ref = vcache.clone()

    k_scale = torch.tensor([0.1], dtype=torch.float32, device=qkv_new.device)
    v_scale = torch.tensor([0.1], dtype=torch.float32, device=qkv_new.device)
    kcache_fp8 = kcache.to(torch.float8_e4m3fn)
    vcache_fp8 = vcache.to(torch.float8_e4m3fn)

    seqlens = q_index[1:] - q_index[:-1]
    max_seqlens = seqlens.max().item()

    q_fp8, q_scale, split_k_flag = hpc.rope_norm_store_kv_fp8(
        key_cache=kcache_fp8,
        value_cache=vcache_fp8,
        qkv=qkv_new,
        cos_sin=cos_sin,
        num_seqlen_per_req=num_seqlen_per_req,
        q_index=q_index,
        kvcache_indices=kv_indices,
        is_prefill=True,
        k_scale=k_scale,
        v_scale=v_scale,
        quant_policy=1,  # 1 for dqskv , 2 for sqskv
        max_seqlens=max_seqlens,
        q_norm_weight=q_norm_weight if qk_norm_policy > 0 else None,
        k_norm_weight=k_norm_weight if qk_norm_policy > 0 else None,
        qk_norm_policy=qk_norm_policy,  # 0 for no norm, 1 for rope first, 2 for norm first
    )

    mask = torch.arange(q_scale.shape[2]).expand(
        q_scale.shape[0], q_scale.shape[2]
    ).cuda() < seqlens.unsqueeze(1)
    qk_scale_normal = q_scale.permute(0, 2, 1)[mask].cuda()
    q_bf16 = (q_fp8.to(torch.bfloat16) * qk_scale_normal[:, :, None]).to(torch.bfloat16)

    torch_out_q, _, _, _ = torch_rope_norm_blocked_prefill(
        kcache_ref,
        vcache_ref,
        qkv_ref,
        cos_sin,
        num_seqlen_per_req,
        q_index,
        kv_indices,
        is_prefill=True,
        use_qknorm=(qk_norm_policy > 0),
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        qk_norm_policy=qk_norm_policy,
        clear_kv_tail=True,
    )

    assert allclose(torch_out_q, q_bf16, atol=0.5)


@pytest.mark.parametrize("num_req", [8])
@pytest.mark.parametrize(
    "num_q_heads,num_kv_heads,qk_head_dim",
    [(8, 1, 128), (64, 8, 128)],
)
@pytest.mark.parametrize("qk_norm_policy", [0, 1])
def test_rope_norm_store_kv_fp8_decode_dqskv(
    num_req, num_q_heads, num_kv_heads, qk_head_dim, qk_norm_policy
):
    req_length = torch.randint(20, 200, (num_req,)).tolist()
    v_head_dim = qk_head_dim
    kv_block_size = 64
    max_num_kv_blocks = 1024
    max_rope_position = 2048
    dtype = torch.bfloat16
    (
        qkv,
        num_seqlen_per_req,
        cos_sin,
        kcache,
        vcache,
        kv_indices,
        q_norm_weight,
        k_norm_weight,
    ) = prepare_decode_input(
        num_req,
        req_length,
        num_q_heads,
        num_kv_heads,
        qk_head_dim,
        v_head_dim,
        kv_block_size,
        max_num_kv_blocks,
        max_rope_position,
        dtype,
    )

    q_index_decode = torch.arange(num_req + 1, dtype=torch.int32, device=qkv.device)

    # Pad to align-8
    qkv, num_seqlen_per_req, q_index_decode, kv_indices, real_rows = pad_decode_inputs_to_align8(
        qkv, num_seqlen_per_req, q_index_decode, kv_indices
    )

    qkv_ref = qkv[:real_rows].clone()
    kcache_ref = kcache.clone()
    vcache_ref = vcache.clone()
    q_index_ref = torch.arange(num_req + 1, dtype=torch.int32, device=qkv.device)

    k_scale = torch.tensor([0.1], dtype=torch.float32, device=qkv.device)
    v_scale = torch.tensor([0.1], dtype=torch.float32, device=qkv.device)
    kcache_fp8 = kcache.to(torch.float8_e4m3fn)
    vcache_fp8 = vcache.to(torch.float8_e4m3fn)

    q_fp8, q_scale, split_k_flag = hpc.rope_norm_store_kv_fp8(
        key_cache=kcache_fp8,
        value_cache=vcache_fp8,
        qkv=qkv,
        cos_sin=cos_sin,
        num_seqlen_per_req=num_seqlen_per_req,
        q_index=q_index_decode,
        kvcache_indices=kv_indices,
        is_prefill=False,
        k_scale=k_scale,
        v_scale=v_scale,
        quant_policy=1,
        max_seqlens=1,
        q_norm_weight=q_norm_weight if qk_norm_policy > 0 else None,
        k_norm_weight=k_norm_weight if qk_norm_policy > 0 else None,
        qk_norm_policy=qk_norm_policy,
    )

    q_bf16 = (q_fp8[:real_rows].to(torch.bfloat16) * q_scale[:real_rows, :, None]).to(
        torch.bfloat16
    )

    torch_out_q, _, _, _ = torch_rope_norm_blocked_decode(
        kcache_ref,
        vcache_ref,
        qkv_ref,
        cos_sin,
        num_seqlen_per_req[:num_req],
        q_index_ref,
        kv_indices[:num_req],
        is_prefill=False,
        use_qknorm=(qk_norm_policy > 0),
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        qk_norm_policy=qk_norm_policy,
        clear_kv_tail=True,
    )

    assert allclose(torch_out_q, q_bf16, atol=0.5)


@pytest.mark.skipif(bool(os.getenv("SANITIZER_CHECK")), reason="skip sanitizer")
@pytest.mark.parametrize("num_req", [7])
@pytest.mark.parametrize(
    "num_q_heads,num_kv_heads,qk_head_dim",
    [(8, 1, 128), (64, 8, 128)],
)
@pytest.mark.parametrize("qk_norm_policy", [0, 2])
def test_rope_norm_store_kv_fp8_prefill_sqskv(
    num_req, num_q_heads, num_kv_heads, qk_head_dim, qk_norm_policy
):
    req_length = torch.randint(20, 200, (num_req,)).tolist()
    v_head_dim = qk_head_dim
    kv_block_size = 64
    max_num_kv_blocks = 1024
    max_rope_position = 2048
    dtype = torch.bfloat16
    (
        qkv,
        num_seqlen_per_req,
        cos_sin,
        kcache,
        vcache,
        kv_indices,
        q_norm_weight,
        k_norm_weight,
    ) = prepare_prefill_input(
        num_req,
        req_length,
        num_q_heads,
        num_kv_heads,
        qk_head_dim,
        v_head_dim,
        kv_block_size,
        max_num_kv_blocks,
        max_rope_position,
        dtype,
    )
    q_index, qkv_new = sample_and_extract_qkv(req_length, qkv)

    qkv_ref = qkv_new.clone()
    kcache_ref = kcache.clone()
    vcache_ref = vcache.clone()

    k_scale = torch.tensor([0.1], dtype=torch.float32, device=qkv_new.device)
    v_scale = torch.tensor([0.1], dtype=torch.float32, device=qkv_new.device)
    kcache_fp8 = kcache.to(torch.float8_e4m3fn)
    vcache_fp8 = vcache.to(torch.float8_e4m3fn)

    seqlens = q_index[1:] - q_index[:-1]
    max_seqlens = seqlens.max().item()

    q_scale_val = 2
    q_scale_inv = torch.tensor([1 / q_scale_val], dtype=torch.float32, device=qkv_new.device)

    q_fp8, q_scale, split_k_flag = hpc.rope_norm_store_kv_fp8(
        key_cache=kcache_fp8,
        value_cache=vcache_fp8,
        qkv=qkv_new,
        cos_sin=cos_sin,
        num_seqlen_per_req=num_seqlen_per_req,
        q_index=q_index,
        kvcache_indices=kv_indices,
        is_prefill=True,
        k_scale=k_scale,
        v_scale=v_scale,
        quant_policy=2,
        max_seqlens=max_seqlens,
        q_scale_inv=q_scale_inv,
        q_norm_weight=q_norm_weight if qk_norm_policy > 0 else None,
        k_norm_weight=k_norm_weight if qk_norm_policy > 0 else None,
        qk_norm_policy=qk_norm_policy,
    )

    assert q_scale is None
    q_bf16 = (q_fp8.to(torch.float32) * q_scale_val).to(torch.bfloat16)

    torch_out_q, _, _, _ = torch_rope_norm_blocked_prefill(
        kcache_ref,
        vcache_ref,
        qkv_ref,
        cos_sin,
        num_seqlen_per_req,
        q_index,
        kv_indices,
        is_prefill=True,
        use_qknorm=(qk_norm_policy > 0),
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        qk_norm_policy=qk_norm_policy,
        clear_kv_tail=True,
    )

    assert allclose(torch_out_q, q_bf16, atol=0.5)


@pytest.mark.parametrize("num_req", [8])
@pytest.mark.parametrize(
    "num_q_heads,num_kv_heads,qk_head_dim",
    [(8, 1, 128), (64, 8, 128)],
)
@pytest.mark.parametrize("qk_norm_policy", [0, 1, 2])
def test_rope_norm_store_kv_fp8_decode_sqskv(
    num_req, num_q_heads, num_kv_heads, qk_head_dim, qk_norm_policy
):
    req_length = torch.randint(20, 200, (num_req,)).tolist()
    v_head_dim = qk_head_dim
    kv_block_size = 64
    max_num_kv_blocks = 1024
    max_rope_position = 2048
    dtype = torch.bfloat16
    (
        qkv,
        num_seqlen_per_req,
        cos_sin,
        kcache,
        vcache,
        kv_indices,
        q_norm_weight,
        k_norm_weight,
    ) = prepare_decode_input(
        num_req,
        req_length,
        num_q_heads,
        num_kv_heads,
        qk_head_dim,
        v_head_dim,
        kv_block_size,
        max_num_kv_blocks,
        max_rope_position,
        dtype,
    )

    q_index_decode = torch.arange(num_req + 1, dtype=torch.int32, device=qkv.device)

    # Pad to align-8
    qkv, num_seqlen_per_req, q_index_decode, kv_indices, real_rows = pad_decode_inputs_to_align8(
        qkv, num_seqlen_per_req, q_index_decode, kv_indices
    )

    qkv_ref = qkv[:real_rows].clone()
    kcache_ref = kcache.clone()
    vcache_ref = vcache.clone()
    q_index_ref = torch.arange(num_req + 1, dtype=torch.int32, device=qkv.device)

    k_scale = torch.tensor([0.1], dtype=torch.float32, device=qkv.device)
    v_scale = torch.tensor([0.1], dtype=torch.float32, device=qkv.device)
    kcache_fp8 = kcache.to(torch.float8_e4m3fn)
    vcache_fp8 = vcache.to(torch.float8_e4m3fn)

    q_scale_val = 2
    q_scale_inv = torch.tensor([1 / q_scale_val], dtype=torch.float32, device=qkv.device)

    q_fp8, q_scale, split_k_flag = hpc.rope_norm_store_kv_fp8(
        key_cache=kcache_fp8,
        value_cache=vcache_fp8,
        qkv=qkv,
        cos_sin=cos_sin,
        num_seqlen_per_req=num_seqlen_per_req,
        q_index=q_index_decode,
        kvcache_indices=kv_indices,
        is_prefill=False,
        k_scale=k_scale,
        v_scale=v_scale,
        quant_policy=2,
        max_seqlens=1,
        q_scale_inv=q_scale_inv,
        q_norm_weight=q_norm_weight if qk_norm_policy > 0 else None,
        k_norm_weight=k_norm_weight if qk_norm_policy > 0 else None,
        qk_norm_policy=qk_norm_policy,
    )

    assert q_scale is None
    q_bf16 = (q_fp8[:real_rows].to(torch.float32) * q_scale_val).to(torch.bfloat16)

    torch_out_q, _, _, _ = torch_rope_norm_blocked_decode(
        kcache_ref,
        vcache_ref,
        qkv_ref,
        cos_sin,
        num_seqlen_per_req[:num_req],
        q_index_ref,
        kv_indices[:num_req],
        is_prefill=False,
        use_qknorm=(qk_norm_policy > 0),
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        qk_norm_policy=qk_norm_policy,
        clear_kv_tail=True,
    )

    assert allclose(torch_out_q, q_bf16, atol=0.5)


def prepare_mtp_decode_input(
    num_req,
    req_length,
    mtp_steps,
    num_q_heads,
    num_kv_heads,
    qk_head_dim,
    v_head_dim,
    kv_block_size,
    max_num_kv_blocks,
    max_rope_position,
    dtype=torch.bfloat16,
    device="cuda",
):
    """Prepare decode input with MTP (multi-token prediction).

    Each request contributes `mtp_steps` rows instead of 1.
    req_length[i] is the existing kv length (before this decode step).
    The new tokens occupy positions [req_length[i], req_length[i] + mtp_steps).
    """
    if req_length is None:
        req_length = torch.randint(20, 200, (num_req,)).tolist()
    if isinstance(req_length, int):
        req_length = [req_length] * num_req
    updated_req_length = [x + mtp_steps for x in req_length]
    total_rows = num_req * mtp_steps
    qkv = torch.randn(
        total_rows,
        num_q_heads * qk_head_dim + num_kv_heads * qk_head_dim + num_kv_heads * v_head_dim,
        dtype=dtype,
        device=device,
    )
    cos_sin = generate_cos_sin_cache(max_rope_position, qk_head_dim).to(
        dtype=torch.float32, device=device
    )
    kcache = torch.randn(
        max_num_kv_blocks, kv_block_size, num_kv_heads, qk_head_dim, dtype=dtype, device=device
    )
    vcache = torch.randn(
        max_num_kv_blocks, kv_block_size, num_kv_heads, v_head_dim, dtype=dtype, device=device
    )
    kv_indices = generate_kv_block_indices(kcache, updated_req_length).to(device)
    q_norm_weight = torch.randn(qk_head_dim, dtype=torch.float32, device=device)
    k_norm_weight = torch.randn(qk_head_dim, dtype=torch.float32, device=device)
    num_seqlen_per_req = torch.tensor(updated_req_length, dtype=torch.int32, device=device)
    q_lengths = [mtp_steps] * num_req
    q_cumsum = torch.cumsum(torch.tensor(q_lengths, device=device), dim=0)
    q_index = torch.cat((torch.tensor([0], device=device, dtype=q_cumsum.dtype), q_cumsum)).to(
        torch.int32
    )
    return (
        qkv,
        num_seqlen_per_req,
        q_index,
        cos_sin,
        kcache,
        vcache,
        kv_indices,
        q_norm_weight,
        k_norm_weight,
    )


@pytest.mark.parametrize("num_req", [8])
@pytest.mark.parametrize(
    "num_q_heads,num_kv_heads,qk_head_dim",
    [(8, 1, 128), (64, 8, 128)],
)
@pytest.mark.parametrize("qk_norm_policy", [0, 1, 2])
@pytest.mark.parametrize("mtp_steps", [1, 2])
def test_rope_norm_store_kv_mtp_decode(
    num_req, num_q_heads, num_kv_heads, qk_head_dim, qk_norm_policy, mtp_steps
):
    """MTP decode: each request has mtp_steps tokens (not just 1)."""
    req_length = torch.randint(20, 200, (num_req,)).tolist()
    v_head_dim = qk_head_dim
    kv_block_size = 64
    max_num_kv_blocks = 1024
    max_rope_position = 2048
    dtype = torch.bfloat16
    (
        qkv,
        num_seqlen_per_req,
        q_index,
        cos_sin,
        kcache,
        vcache,
        kv_indices,
        q_norm_weight,
        k_norm_weight,
    ) = prepare_mtp_decode_input(
        num_req,
        req_length,
        mtp_steps,
        num_q_heads,
        num_kv_heads,
        qk_head_dim,
        v_head_dim,
        kv_block_size,
        max_num_kv_blocks,
        max_rope_position,
        dtype,
    )
    # Pad to align-8
    qkv, num_seqlen_per_req, q_index, kv_indices, real_rows = pad_decode_inputs_to_align8(
        qkv, num_seqlen_per_req, q_index, kv_indices
    )

    qkv_ref = qkv[:real_rows].clone()
    kcache_ref = kcache.clone()
    vcache_ref = vcache.clone()
    q_index_ref = torch.cat(
        (
            torch.tensor([0], device=qkv.device, dtype=torch.int32),
            torch.cumsum(torch.tensor([mtp_steps] * num_req, device=qkv.device), dim=0).to(
                torch.int32
            ),
        )
    )

    my_out_q = hpc.rope_norm_store_kv(
        kcache,
        vcache,
        qkv,
        cos_sin,
        num_seqlen_per_req,
        q_index,
        kv_indices,
        False,
        q_norm_weight=q_norm_weight if qk_norm_policy > 0 else None,
        k_norm_weight=k_norm_weight if qk_norm_policy > 0 else None,
        qk_norm_policy=qk_norm_policy,
    )

    # Use prefill reference (handles multi-token per request correctly)
    torch_out_q, torch_out_k, torch_kcache, torch_vcache = torch_rope_norm_blocked_prefill(
        kcache_ref,
        vcache_ref,
        qkv_ref,
        cos_sin,
        num_seqlen_per_req[:num_req],
        q_index_ref,
        kv_indices[:num_req],
        is_prefill=True,
        use_qknorm=(qk_norm_policy > 0),
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        qk_norm_policy=qk_norm_policy,
        clear_kv_tail=True,
    )

    assert allclose(torch_out_q, my_out_q[:real_rows], atol=5e-2)
    assert allclose(torch_kcache, kcache, atol=5e-2)
    assert allclose(torch_vcache, vcache, atol=5e-2)


@pytest.mark.skipif(bool(os.getenv("SANITIZER_CHECK")), reason="skip sanitizer")
@pytest.mark.parametrize("num_req", [8])
@pytest.mark.parametrize(
    "num_q_heads,num_kv_heads,qk_head_dim",
    [(8, 1, 128), (64, 8, 128)],
)
@pytest.mark.parametrize("qk_norm_policy", [0, 1, 2])
@pytest.mark.parametrize("mtp_steps", [1, 2])
def test_rope_norm_store_kv_fp8_mtp_decode_dqskv(
    num_req, num_q_heads, num_kv_heads, qk_head_dim, qk_norm_policy, mtp_steps
):
    """MTP decode + FP8 dqskv."""
    req_length = torch.randint(20, 200, (num_req,)).tolist()
    v_head_dim = qk_head_dim
    kv_block_size = 64
    max_num_kv_blocks = 1024
    max_rope_position = 2048
    dtype = torch.bfloat16
    (
        qkv,
        num_seqlen_per_req,
        q_index,
        cos_sin,
        kcache,
        vcache,
        kv_indices,
        q_norm_weight,
        k_norm_weight,
    ) = prepare_mtp_decode_input(
        num_req,
        req_length,
        mtp_steps,
        num_q_heads,
        num_kv_heads,
        qk_head_dim,
        v_head_dim,
        kv_block_size,
        max_num_kv_blocks,
        max_rope_position,
        dtype,
    )
    # Pad to align-8
    qkv, num_seqlen_per_req, q_index, kv_indices, real_rows = pad_decode_inputs_to_align8(
        qkv, num_seqlen_per_req, q_index, kv_indices
    )

    qkv_ref = qkv[:real_rows].clone()
    kcache_ref = kcache.clone()
    vcache_ref = vcache.clone()
    q_index_ref = torch.cat(
        (
            torch.tensor([0], device=qkv.device, dtype=torch.int32),
            torch.cumsum(torch.tensor([mtp_steps] * num_req, device=qkv.device), dim=0).to(
                torch.int32
            ),
        )
    )

    k_scale = torch.tensor([0.1], dtype=torch.float32, device=qkv.device)
    v_scale = torch.tensor([0.1], dtype=torch.float32, device=qkv.device)
    kcache_fp8 = kcache.to(torch.float8_e4m3fn)
    vcache_fp8 = vcache.to(torch.float8_e4m3fn)

    q_fp8, q_scale, split_k_flag = hpc.rope_norm_store_kv_fp8(
        key_cache=kcache_fp8,
        value_cache=vcache_fp8,
        qkv=qkv,
        cos_sin=cos_sin,
        num_seqlen_per_req=num_seqlen_per_req,
        q_index=q_index,
        kvcache_indices=kv_indices,
        is_prefill=False,
        k_scale=k_scale,
        v_scale=v_scale,
        quant_policy=1,
        max_seqlens=mtp_steps,
        q_norm_weight=q_norm_weight if qk_norm_policy > 0 else None,
        k_norm_weight=k_norm_weight if qk_norm_policy > 0 else None,
        qk_norm_policy=qk_norm_policy,
    )

    # q_scale for decode is [num_rows, num_q_heads]
    q_bf16 = (q_fp8[:real_rows].to(torch.bfloat16) * q_scale[:real_rows, :, None]).to(
        torch.bfloat16
    )

    torch_out_q, _, _, _ = torch_rope_norm_blocked_prefill(
        kcache_ref,
        vcache_ref,
        qkv_ref,
        cos_sin,
        num_seqlen_per_req[:num_req],
        q_index_ref,
        kv_indices[:num_req],
        is_prefill=True,
        use_qknorm=(qk_norm_policy > 0),
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        qk_norm_policy=qk_norm_policy,
        clear_kv_tail=True,
    )

    assert allclose(torch_out_q, q_bf16, atol=0.5)
