#!/usr/bin/env python3
"""
Comprehensive test suite for RoPE operation with blocked KV cache.
Tests both prefill and decoding modes, with and without QK normalization.
"""

import math
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
    q_index,
    kv_indices,
    is_prefill=True,
    use_qknorm=False,
    q_norm_weight=None,
    k_norm_weight=None,
    qk_norm_policy=1,
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
    q_index,
    kv_indices,
    is_prefill=False,
    use_qknorm=False,
    q_norm_weight=None,
    k_norm_weight=None,
    qk_norm_policy=1,
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


##############################################################################
#  New Rope Test
##############################################################################


def generate_cos_sin_cache(max_position, head_dim, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_position).float()
    freqs = torch.outer(t, inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)


def generate_kv_block_indices(kcache, req_length):
    num_req = len(req_length)
    kv_block_size = kcache.shape[1]
    num_blocks_per_req = [(l + kv_block_size - 1) // kv_block_size for l in req_length]
    shuffled = torch.randperm(kcache.shape[0])
    kv_idx = torch.ones(num_req, max(num_blocks_per_req) + 4, dtype=torch.int32) * -1
    offset = 0
    for i in range(num_req):
        n = num_blocks_per_req[i]
        kv_idx[i, :n] = shuffled[offset : offset + n]
        offset += n
    return kv_idx


def apply_rms_norm_reference(x, weight, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def apply_rotary_pos_emb_neox_reference(x, cos_sin):
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    c = cos_sin[:, :h].unsqueeze(1)
    s = cos_sin[:, h:].unsqueeze(1)
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


def hadamard_matrix(n, device, dtype=torch.float32):
    """Build an unscaled Hadamard matrix of size n (power of 2) by recursive doubling."""
    H = torch.tensor([[1.0]], device=device, dtype=dtype)
    size = 1
    while size < n:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
        size *= 2
    return H


def apply_hadamard_per_head(x, head_dim):
    """Apply normalized Hadamard transform along the last (head_dim) axis."""
    H = hadamard_matrix(head_dim, x.device, dtype=torch.float32)
    inv_sqrt = 1.0 / math.sqrt(head_dim)
    return torch.matmul(x.to(torch.float32), H.t()) * inv_sqrt


def rope_norm_ref(
    kcache,
    vcache,
    qkv,
    cos_sin,
    num_seqlen_per_req,
    q_index,
    kv_indices,
    q_norm_weight,
    k_norm_weight,
    qk_norm_policy,
    apply_hadamard=False,
):
    """Unified PyTorch reference: RoPE + optional RMSNorm + paged KV write.

    Handles prefill, decode (mtp=0), and MTP decode (mtp>=1) uniformly via q_index.
    """
    dtype = qkv.dtype
    num_kv = kcache.shape[2]
    v_dim = vcache.shape[3]
    qk_dim = kcache.shape[3]
    num_q = (qkv.shape[1] - num_kv * qk_dim - num_kv * v_dim) // qk_dim
    num_req = num_seqlen_per_req.shape[0]
    q_lens = (q_index[1:] - q_index[:-1]).tolist()
    num_rows = q_index[-1].item()
    blk = kcache.shape[1]

    q = qkv[:, : num_q * qk_dim].to(torch.float32).view(num_rows, num_q, qk_dim)
    k = (
        qkv[:, num_q * qk_dim : (num_q + num_kv) * qk_dim]
        .to(torch.float32)
        .view(num_rows, num_kv, qk_dim)
    )
    v = qkv[:, (num_q + num_kv) * qk_dim :].view(num_rows, num_kv, v_dim)

    # per-token cos/sin indexed by absolute position
    cs = torch.zeros(num_rows, qk_dim, dtype=torch.float32, device=qkv.device)
    off = 0
    for i in range(num_req):
        sl = num_seqlen_per_req[i].item()
        ql = q_lens[i]
        if ql > 0:
            cs[off : off + ql] = cos_sin[sl - ql : sl]
        off += ql

    if qk_norm_policy == 2:
        q = apply_rms_norm_reference(q, q_norm_weight)
        k = apply_rms_norm_reference(k, k_norm_weight)
    q = apply_rotary_pos_emb_neox_reference(q, cs)
    k = apply_rotary_pos_emb_neox_reference(k, cs)
    if qk_norm_policy == 1:
        q = apply_rms_norm_reference(q, q_norm_weight)
        k = apply_rms_norm_reference(k, k_norm_weight)

    if apply_hadamard:
        q = apply_hadamard_per_head(q, qk_dim)
        k = apply_hadamard_per_head(k, qk_dim)

    # write into paged KV cache; clear tail of last used slot per request
    tok = 0
    for ri in range(num_req):
        sl = num_seqlen_per_req[ri].item()
        ql = q_lens[ri]
        for pos in range(sl - ql, sl):
            bi, pb = pos // blk, pos % blk
            cb = kv_indices[ri, bi].item()
            kcache[cb, pb] = k[tok].to(dtype)
            vcache[cb, pb] = v[tok].to(dtype)
            if pos == sl - 1 and pb + 1 < blk:
                kcache[cb, pb + 1 :] = 0
                vcache[cb, pb + 1 :] = 0
            tok += 1

    return q.to(dtype)


def pad_decode_inputs_to_align8(qkv, num_seqlen, q_index, kv_indices):
    """Pad decode batch/rows to a multiple of 8 (simulates CUDA-graph padding)."""
    nr = qkv.shape[0]
    nb = num_seqlen.shape[0]
    pb = (nb + 7) // 8 * 8
    pr = (nr + 7) // 8 * 8

    if pr > nr:
        qkv = torch.cat(
            [qkv, torch.zeros(pr - nr, qkv.shape[1], dtype=qkv.dtype, device=qkv.device)]
        )
    if pb > nb:
        num_seqlen = torch.cat(
            [num_seqlen, torch.zeros(pb - nb, dtype=num_seqlen.dtype, device=num_seqlen.device)]
        )
        q_index = torch.cat(
            [q_index, torch.full((pb - nb,), pr, dtype=q_index.dtype, device=q_index.device)]
        )
        kv_indices = torch.cat(
            [
                kv_indices,
                torch.zeros(
                    pb - nb, kv_indices.shape[1], dtype=kv_indices.dtype, device=kv_indices.device
                ),
            ]
        )

    return qkv, num_seqlen, q_index, kv_indices, nr


def prepare_inputs(
    num_req,
    is_prefill,
    mtp,
    num_q_heads,
    num_kv_heads,
    qk_head_dim,
    v_head_dim=None,
    kv_block_size=64,
    max_num_kv_blocks=1024,
    max_rope_position=2048,
    dtype=torch.bfloat16,
    device="cuda",
):
    """Build all tensors required for rope_norm_store_kv[_fp8] tests.

    For prefill (is_prefill=True):  variable Q tokens per request (random suffix sampling).
    For decode  (is_prefill=False): tokens_per_req = mtp+1, batch padded to align-8.

    Returns:
        qkv, num_seqlen, q_index, kcache, vcache, kv_indices,
        q_norm_weight, k_norm_weight, cos_sin,
        real_rows   -- None for prefill; for decode = unpadded row count
    """
    if v_head_dim is None:
        v_head_dim = qk_head_dim
    hidden = num_q_heads * qk_head_dim + num_kv_heads * qk_head_dim + num_kv_heads * v_head_dim

    cos_sin = generate_cos_sin_cache(max_rope_position, qk_head_dim).to(
        dtype=torch.float32, device=device
    )
    kcache = torch.randn(
        max_num_kv_blocks, kv_block_size, num_kv_heads, qk_head_dim, dtype=dtype, device=device
    )
    vcache = torch.randn(
        max_num_kv_blocks, kv_block_size, num_kv_heads, v_head_dim, dtype=dtype, device=device
    )
    q_norm_w = torch.randn(qk_head_dim, dtype=torch.float32, device=device)
    k_norm_w = torch.randn(qk_head_dim, dtype=torch.float32, device=device)

    if is_prefill:
        req_len = torch.randint(20, 200, (num_req,)).tolist()
        qkv_full = torch.randn(sum(req_len), hidden, dtype=dtype, device=device)
        # sample a random-length suffix from each request's token sequence
        req_len_t = torch.tensor(req_len, device=device)
        q_len_t = torch.min((torch.rand(num_req, device=device) * req_len_t).long() + 1, req_len_t)
        cumsum = torch.cumsum(req_len_t, dim=0)
        qkv = torch.cat([qkv_full[cumsum[i] - q_len_t[i] : cumsum[i]] for i in range(num_req)])
        q_index = torch.cat(
            [torch.zeros(1, device=device, dtype=torch.int64), torch.cumsum(q_len_t, 0)]
        ).to(torch.int32)
        num_seqlen = torch.tensor(req_len, dtype=torch.int32, device=device)
        kv_indices = generate_kv_block_indices(kcache, req_len).to(device)
        real_rows = None
    else:
        tpr = mtp + 1  # tokens per request
        exist_len = torch.randint(20, 200, (num_req,)).tolist()
        upd_len = [x + tpr for x in exist_len]
        qkv_raw = torch.randn(num_req * tpr, hidden, dtype=dtype, device=device)
        q_idx_raw = torch.arange(0, (num_req + 1) * tpr, tpr, device=device, dtype=torch.int32)
        num_seqlen_raw = torch.tensor(upd_len, dtype=torch.int32, device=device)
        kv_idx_raw = generate_kv_block_indices(kcache, upd_len).to(device)
        qkv, num_seqlen, q_index, kv_indices, real_rows = pad_decode_inputs_to_align8(
            qkv_raw, num_seqlen_raw, q_idx_raw, kv_idx_raw
        )

    return (
        qkv,
        num_seqlen,
        q_index,
        kcache,
        vcache,
        kv_indices,
        q_norm_w,
        k_norm_w,
        cos_sin,
        real_rows,
    )


# skip sanitizer because the unused q_out will not be touched in test mode but will be cleared in sanitizer mode
@pytest.mark.skipif(bool(os.getenv("SANITIZER_CHECK")), reason="skip sanitizer")
@pytest.mark.parametrize("num_q_heads,num_kv_heads,qk_head_dim", [(8, 1, 128), (64, 8, 128)])
@pytest.mark.parametrize("qk_norm_policy", [0, 1, 2])
@pytest.mark.parametrize("num_req", [7, 16])
@pytest.mark.parametrize("is_prefill,mtp", [(True, None), (False, 0), (False, 1)])
def test_rope_norm_store_kv(
    num_q_heads, num_kv_heads, qk_head_dim, qk_norm_policy, num_req, is_prefill, mtp
):
    """Test rope_norm_store_kv: prefill / decode (mtp=0) / MTP decode (mtp=1)
    across all qk_norm_policy values and GQA/MQA head configs.
    num_req=7 exercises align-8 padding in decode.
    """
    qkv, num_seqlen, q_index, kcache, vcache, kv_indices, q_norm_w, k_norm_w, cos_sin, real_rows = (
        prepare_inputs(num_req, is_prefill, mtp, num_q_heads, num_kv_heads, qk_head_dim)
    )
    kcache_ref, vcache_ref = kcache.clone(), vcache.clone()

    out_q = hpc.rope_norm_store_kv(
        kcache,
        vcache,
        qkv,
        cos_sin,
        num_seqlen,
        q_index,
        kv_indices,
        is_prefill,
        q_norm_weight=q_norm_w if qk_norm_policy > 0 else None,
        k_norm_weight=k_norm_w if qk_norm_policy > 0 else None,
        qk_norm_policy=qk_norm_policy,
    )

    # Pass unpadded views to reference (padding entries have seqlen=0 and must be skipped)
    if real_rows is not None:
        qkv_r, ns_r = qkv[:real_rows], num_seqlen[:num_req]
        qi_r, ki_r = q_index[: num_req + 1], kv_indices[:num_req]
    else:
        qkv_r, ns_r, qi_r, ki_r = qkv, num_seqlen, q_index, kv_indices

    ref_q = rope_norm_ref(
        kcache_ref, vcache_ref, qkv_r, cos_sin, ns_r, qi_r, ki_r, q_norm_w, k_norm_w, qk_norm_policy
    )

    rows = real_rows if real_rows is not None else out_q.shape[0]
    assert allclose(ref_q, out_q[:rows], atol=8e-2)
    assert allclose(kcache_ref, kcache, atol=8e-2)
    assert allclose(vcache_ref, vcache, atol=8e-2)


# skip sanitizer because the unused q_out will not be touched in test mode but will be cleared in sanitizer mode
@pytest.mark.skipif(bool(os.getenv("SANITIZER_CHECK")), reason="skip sanitizer")
@pytest.mark.parametrize("num_q_heads,num_kv_heads,qk_head_dim", [(8, 1, 128), (64, 8, 128)])
@pytest.mark.parametrize("qk_norm_policy", [0, 1, 2])
# 0=dqksv 1=dqskv 2=sqskv 3=dqksv+hadamard 10=qbf16+dynKV 11=qbf16+staticKV
@pytest.mark.parametrize("quant_policy", [0, 1, 2, 3, 10, 11])
@pytest.mark.parametrize("num_req", [7, 16])
@pytest.mark.parametrize("is_prefill,mtp", [(True, None), (False, 0), (False, 1)])
def test_rope_norm_store_kv_fp8(
    num_q_heads,
    num_kv_heads,
    qk_head_dim,
    qk_norm_policy,
    quant_policy,
    num_req,
    is_prefill,
    mtp,
):
    """Test rope_norm_store_kv_fp8: all mode/quant/norm combinations.
    num_req=7 exercises align-8 padding in decode.
    """
    qkv, num_seqlen, q_index, kcache, vcache, kv_indices, q_norm_w, k_norm_w, cos_sin, real_rows = (
        prepare_inputs(num_req, is_prefill, mtp, num_q_heads, num_kv_heads, qk_head_dim)
    )
    kcache_ref, vcache_ref = kcache.clone(), vcache.clone()

    kv_block_size = kcache.shape[1]
    num_blocks = kcache.shape[0]
    device = qkv.device

    if quant_policy == 0 or quant_policy == 3 or quant_policy == 10:
        # Dynamic per-head per-token: k_scale is [num_blocks, R, num_kv_heads, L] (output)
        L = qk_head_dim * 1 // 4  # sizeof(fp8) / sizeof(float)
        R = kv_block_size // L
        k_scale = torch.zeros(num_blocks, R, num_kv_heads, L, dtype=torch.float32, device=device)
        # Per-head v_scale
        v_scale = torch.rand(num_kv_heads, dtype=torch.float32, device=device) * 0.2 + 0.05
    else:
        k_scale = torch.tensor([0.1], dtype=torch.float32, device=device)
        v_scale = torch.tensor([0.1], dtype=torch.float32, device=device)

    q_scale_val = 2.0
    q_scale_inv = torch.tensor([1.0 / q_scale_val], dtype=torch.float32, device=device)

    kcache_fp8 = kcache.to(torch.float8_e4m3fn)
    vcache_fp8 = vcache.to(torch.float8_e4m3fn)

    if is_prefill:
        max_seqlens = int((q_index[1:] - q_index[:-1]).max().item())
    else:
        max_seqlens = mtp + 1  # tokens per request in decode

    needs_q_scale_inv = quant_policy == 2
    apply_hadamard = quant_policy in (3, 10, 11)

    q_fp8, q_scale_out, split_k_flag = hpc.rope_norm_store_kv_fp8(
        key_cache=kcache_fp8,
        value_cache=vcache_fp8,
        qkv=qkv,
        cos_sin=cos_sin,
        num_seqlen_per_req=num_seqlen,
        q_index=q_index,
        kvcache_indices=kv_indices,
        is_prefill=is_prefill,
        k_scale=k_scale,
        v_scale=v_scale,
        quant_policy=hpc.QuantType(quant_policy),
        max_seqlens=max_seqlens,
        q_scale_inv=q_scale_inv if needs_q_scale_inv else None,
        q_norm_weight=q_norm_w if qk_norm_policy > 0 else None,
        k_norm_weight=k_norm_w if qk_norm_policy > 0 else None,
        qk_norm_policy=qk_norm_policy,
        apply_hadamard=apply_hadamard,
    )

    assert split_k_flag.shape == (num_seqlen.shape[0], num_kv_heads)
    assert split_k_flag.dtype == torch.int32

    if (
        quant_policy == 0 or quant_policy == 1 or quant_policy == 3
    ):  # dynamic Q: kernel computes per-token per-head scale
        if is_prefill:
            pad128 = ((max_seqlens + 127) // 128) * 128
            assert q_scale_out.shape == (num_seqlen.shape[0], num_q_heads, pad128)
            # dequant: select valid per-token scales via sequence-length mask
            seqlens = (q_index[1:] - q_index[:-1]).to(device)
            mask = torch.arange(pad128, device=device).expand(
                num_seqlen.shape[0], pad128
            ) < seqlens.unsqueeze(1)
            scale_flat = q_scale_out.permute(0, 2, 1)[mask]  # [total_real_rows, num_q_heads]
            rows = int(q_index[-1].item())
            q_bf16 = (q_fp8[:rows].to(torch.bfloat16) * scale_flat[:, :, None]).to(torch.bfloat16)
        else:
            assert q_scale_out.shape == (qkv.shape[0], num_q_heads)
            rows = real_rows  # num_req * tokens_per_req (before padding)
            q_bf16 = (q_fp8[:rows].to(torch.bfloat16) * q_scale_out[:rows, :, None]).to(
                torch.bfloat16
            )
    elif quant_policy == 10 or quant_policy == 11:  # Q kept in bf16, no quantization, no q_scale
        assert q_scale_out is None
        assert q_fp8.dtype == torch.bfloat16
        rows = real_rows if real_rows is not None else q_fp8.shape[0]
        q_bf16 = q_fp8[:rows]
    else:  # sqskv (with or without hadamard): static scale supplied by caller
        assert q_scale_out is None
        rows = real_rows if real_rows is not None else q_fp8.shape[0]
        q_bf16 = (q_fp8[:rows].to(torch.float32) * q_scale_val).to(torch.bfloat16)

    if real_rows is not None:
        qkv_r, ns_r = qkv[:real_rows], num_seqlen[:num_req]
        qi_r, ki_r = q_index[: num_req + 1], kv_indices[:num_req]
    else:
        qkv_r, ns_r, qi_r, ki_r = qkv, num_seqlen, q_index, kv_indices

    ref_q = rope_norm_ref(
        kcache_ref,
        vcache_ref,
        qkv_r,
        cos_sin,
        ns_r,
        qi_r,
        ki_r,
        q_norm_w,
        k_norm_w,
        qk_norm_policy,
        apply_hadamard=apply_hadamard,
    )
    assert allclose(ref_q, q_bf16, atol=0.8)

    # ========= Verify KV cache for all quant policies =========
    q_lens_r = (qi_r[1:] - qi_r[:-1]).tolist()
    if quant_policy == 0 or quant_policy == 3 or quant_policy == 10:
        L = qk_head_dim * 1 // 4
    tok = 0
    for ri in range(num_req):
        sl = int(ns_r[ri].item())
        ql = int(q_lens_r[ri])
        for pos in range(sl - ql, sl):
            bi, pb = pos // kv_block_size, pos % kv_block_size
            cb = int(ki_r[ri, bi].item())
            # V verification
            for h in range(num_kv_heads):
                v_fp8_vals = vcache_fp8[cb, pb, h, :].to(torch.float32)
                v_ref_vals = vcache_ref[cb, pb, h, :].to(torch.float32)
                if quant_policy == 0 or quant_policy == 3 or quant_policy == 10:
                    v_dequant = v_fp8_vals * v_scale[h]
                else:
                    v_dequant = v_fp8_vals * v_scale[0]
                assert allclose(
                    v_ref_vals, v_dequant, atol=0.8
                ), f"V mismatch at req={ri} pos={pos} head={h} policy={quant_policy}"
            # K verification
            for h in range(num_kv_heads):
                k_fp8_vals = kcache_fp8[cb, pb, h, :].to(torch.float32)
                k_ref_vals = kcache_ref[cb, pb, h, :].to(torch.float32)
                if quant_policy == 0 or quant_policy == 3 or quant_policy == 10:
                    r_idx = pb // L
                    l_idx = pb % L
                    k_s = k_scale[cb, r_idx, h, l_idx].item()
                else:
                    k_s = k_scale[0].item()
                k_dequant = k_fp8_vals * k_s
                assert allclose(
                    k_ref_vals, k_dequant, atol=0.8
                ), f"K mismatch at req={ri} pos={pos} head={h} policy={quant_policy}"
            tok += 1
