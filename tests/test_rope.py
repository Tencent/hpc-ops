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
    q_index,
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
    q_index, qkv_new = sample_and_extract_qkv(req_length, qkv)

    # clone input for torch ref, incase inplace update
    qkv_ref = qkv_new.clone()
    kcache_ref = kcache.clone()
    vcache_ref = vcache.clone()

    my_out_q, my_out_k = hpc.rope_norm_blocked_kvcache(
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
    )

    assert allclose(torch_out_q, my_out_q, atol=5e-2)
    assert allclose(torch_out_k, my_out_k, atol=5e-2)
    assert allclose(torch_kcache, kcache, atol=5e-2)
    assert allclose(torch_vcache, vcache, atol=5e-2)


@pytest.mark.skipif(bool(os.getenv("SANITIZER_CHECK")), reason="skip sanitizer")
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
        num_seqlen_per_req,
        kv_indices,
        is_prefill=False,
        use_qknorm=use_qknorm,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
    )

    assert allclose(torch_out_q, my_out_q, atol=5e-2)
    assert allclose(torch_kcache, kcache, atol=5e-2)
    assert allclose(torch_vcache, vcache, atol=5e-2)


@pytest.mark.skipif(bool(os.getenv("SANITIZER_CHECK")), reason="skip sanitizer")
@pytest.mark.parametrize("num_req", [7])
@pytest.mark.parametrize("num_q_head_head_dim", [(4, 128)])
@pytest.mark.parametrize("num_kv_heads", [1])
@pytest.mark.parametrize("use_qknorm", [True])
def test_rope_norm_blocked_prefill_fp8(num_req, num_q_head_head_dim, num_kv_heads, use_qknorm):
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
    k_bf16 = k_fp8.to(torch.bfloat16) * k_scale.to(torch.bfloat16)

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
    )

    my_kcache = kcache_fp8.float()
    my_vcache = vcache_fp8.float()
    torch_kcache = torch_kcache.float()
    torch_vcache = torch_vcache.float()

    assert allclose(torch_out_q, q_bf16, atol=0.5)
    assert allclose(torch_out_k, k_bf16, atol=0.5)


@pytest.mark.skipif(bool(os.getenv("SANITIZER_CHECK")), reason="skip sanitizer")
@pytest.mark.parametrize("num_req", [7])
@pytest.mark.parametrize("num_q_head_head_dim", [(4, 128)])
@pytest.mark.parametrize("num_kv_heads", [1])
@pytest.mark.parametrize("use_qknorm", [True])
def test_rope_norm_blocked_decode_fp8(num_req, num_q_head_head_dim, num_kv_heads, use_qknorm):
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
        )
    )

    q_bf16 = (q_fp8.to(torch.bfloat16) * qk_scale[:, :, None]).to(
        torch.bfloat16
    )  # no k_scale in decode
    k_bf16 = k_fp8.to(torch.bfloat16) * k_scale.to(torch.bfloat16)

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
    )

    my_kcache = kcache_fp8.float()
    my_vcache = vcache_fp8.float()
    torch_kcache = torch_kcache.float()
    torch_vcache = torch_vcache.float()

    assert allclose(torch_out_q, q_bf16, atol=0.5)
