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


def apply_rotary_pos_emb_neox_reference(x, cos_sin):
    """
    Reference implementation of RoPE (neox version) in Python.

    Args:
        x: [num_tokens, num_heads, head_dim]
        cos_sin: [num_tokens, head_dim] where first half is cos, second half is sin

    Returns:
        output: [num_tokens, num_heads, head_dim]
    """
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


def apply_rms_norm_reference(x, weight, eps=1e-6):
    """
    Reference implementation of RMS normalization.

    Args:
        x: [num_tokens, num_heads, head_dim]
        weight: [head_dim]
        eps: epsilon for numerical stability

    Returns:
        output: [num_tokens, num_heads, head_dim]
    """
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return x_normed * weight


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
    # seed = 42
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)

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


def torch_rope_norm_blocked_prefill(
    kcache,
    vcache,
    qkv,
    cos_sin,
    num_seqlen_per_req,
    kv_indices,
    is_prefill=True,
    use_qknorm=False,
    q_norm_weight=None,
    k_norm_weight=None,
):
    """Test RoPE prefill mode with PyTorch reference implementation."""
    assert is_prefill
    dtype = qkv.dtype
    num_kv_heads = kcache.shape[2]
    v_head_dim = vcache.shape[3]
    qk_head_dim = kcache.shape[3]
    num_q_heads = (
        qkv.shape[1] - num_kv_heads * qk_head_dim - num_kv_heads * v_head_dim
    ) // qk_head_dim

    num_rows = num_seqlen_per_req.sum().item()
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
        cos_sin_for_tokens[token_offset : token_offset + seq_len] = cos_sin[:seq_len]
        token_offset += seq_len

    q_input = q_input.view(num_rows, num_q_heads, qk_head_dim)
    k_input = k_input.view(num_rows, num_kv_heads, qk_head_dim)
    v_input = v_input.view(num_rows, num_kv_heads, v_head_dim)
    # Compute reference Q and K
    q_ref = apply_rotary_pos_emb_neox_reference(q_input, cos_sin_for_tokens)
    k_ref = apply_rotary_pos_emb_neox_reference(k_input, cos_sin_for_tokens)

    if use_qknorm:
        q_ref = apply_rms_norm_reference(q_ref, q_norm_weight)
        k_ref = apply_rms_norm_reference(k_ref, k_norm_weight)

    # update kvcache
    kv_block_size = kcache.shape[1]
    token_idx = 0
    for req_idx in range(num_req):
        seq_len = num_seqlen_per_req[req_idx].item()
        for pos_in_seq in range(seq_len):
            block_idx_in_req = pos_in_seq // kv_block_size
            pos_in_block = pos_in_seq % kv_block_size
            cache_block_idx = kv_indices[req_idx, block_idx_in_req].item()
            assert cache_block_idx >= 0, f"Invalid cache block index: {cache_block_idx}"
            # Update K cache
            kcache[cache_block_idx, pos_in_block, :, :] = k_ref[token_idx, :, :].to(dtype)
            # Update V cache
            vcache[cache_block_idx, pos_in_block, :, :] = v_input[token_idx, :, :].to(dtype)
            token_idx += 1

    out_q = q_ref.to(dtype)
    out_k = k_ref.to(dtype)
    # out_qkv = torch.cat([out_q, out_k, out_v], dim=1)
    return out_q, out_k, kcache, vcache


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
    # in decode, every req has length 1
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


def torch_rope_norm_blocked_decode(
    kcache,
    vcache,
    qkv,
    cos_sin,
    num_seqlen_per_req,
    kv_indices,
    is_prefill=False,
    use_qknorm=False,
    q_norm_weight=None,
    k_norm_weight=None,
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

    q_input = q_input.view(num_rows, num_q_heads, qk_head_dim)
    k_input = k_input.view(num_rows, num_kv_heads, qk_head_dim)
    v_input = v_input.view(num_rows, num_kv_heads, v_head_dim)
    # Compute reference Q and K
    q_ref = apply_rotary_pos_emb_neox_reference(q_input, cos_sin_for_tokens)
    k_ref = apply_rotary_pos_emb_neox_reference(k_input, cos_sin_for_tokens)

    if use_qknorm:
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
        vcache[cache_block_idx, pos_in_block, :, :] = v_input[token_idx, :, :].to(dtype)

        if pos_in_block == 0:
            kcache[cache_block_idx, 1:, :, :] = 0
            vcache[cache_block_idx, 1:, :, :] = 0

        token_idx += 1

    out_q = q_ref.to(dtype)
    out_k = k_ref.to(dtype)
    # out_qkv = torch.cat([out_q, out_k, out_v], dim=1)
    return out_q, out_k, kcache, vcache


@pytest.mark.parametrize("num_req", [7])
@pytest.mark.parametrize("num_q_head_head_dim", [(4, 128), (8, 128), (8, 80)])
@pytest.mark.parametrize("num_kv_heads", [1])
@pytest.mark.parametrize("use_qknorm", [False, True])
def test_rope_norm_blocked_prefill(num_req, num_q_head_head_dim, num_kv_heads, use_qknorm):
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

    # clone input for torch ref, incase inplace update
    qkv_ref = qkv.clone()
    kcache_ref = kcache.clone()
    vcache_ref = vcache.clone()

    my_out_q, my_out_k = hpc.rope_norm_blocked_kvcache(
        kcache,
        vcache,
        qkv,
        cos_sin,
        num_seqlen_per_req,
        kv_indices,
        True,  # is_refill
        use_qknorm,
        q_norm_weight,
        k_norm_weight,
    )

    torch_out_q, torch_out_k, torch_kcache, torch_vcache = torch_rope_norm_blocked_prefill(
        kcache_ref,
        vcache_ref,
        qkv_ref,
        cos_sin,
        num_seqlen_per_req,
        kv_indices,
        is_prefill=True,
        use_qknorm=use_qknorm,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
    )

    assert torch.allclose(my_out_q, torch_out_q, atol=5e-2)
    assert torch.allclose(my_out_k, torch_out_k, atol=5e-2)
    assert torch.allclose(kcache, torch_kcache, atol=5e-2)
    assert torch.allclose(vcache, torch_vcache, atol=5e-2)


@pytest.mark.parametrize("num_req", [7])
@pytest.mark.parametrize("num_q_head_head_dim", [(4, 128), (8, 128), (8, 80)])
@pytest.mark.parametrize("num_kv_heads", [1])
@pytest.mark.parametrize("use_qknorm", [False, True])
def test_rope_norm_blocked_decode(num_req, num_q_head_head_dim, num_kv_heads, use_qknorm):
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

    my_out_q, my_out_k = hpc.rope_norm_blocked_kvcache(
        kcache,
        vcache,
        qkv,
        cos_sin,
        num_seqlen_per_req,
        kv_indices,
        False,  # is_refill
        use_qknorm,
        q_norm_weight,
        k_norm_weight,
    )

    torch_out_q, torch_out_k, torch_kcache, torch_vcache = torch_rope_norm_blocked_decode(
        kcache_ref,
        vcache_ref,
        qkv_ref,
        cos_sin,
        num_seqlen_per_req,
        kv_indices,
        is_prefill=False,
        use_qknorm=use_qknorm,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
    )

    assert torch.allclose(my_out_q, torch_out_q, atol=5e-2)
    assert torch.allclose(kcache, torch_kcache, atol=5e-2)
    assert torch.allclose(vcache, torch_vcache, atol=5e-2)
