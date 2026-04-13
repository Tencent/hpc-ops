import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import pytest
import torch

import hpc
from utils import allclose


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


@pytest.mark.skipif(bool(os.getenv("SANITIZER_CHECK")), reason="skip sanitizer")
@pytest.mark.parametrize("num_q_heads,num_kv_heads,qk_head_dim", [(8, 1, 128), (64, 8, 128)])
@pytest.mark.parametrize("qk_norm_policy", [0, 1, 2])
@pytest.mark.parametrize("quant_policy", [1, 2])  # 1=dqskv (dynamic), 2=sqskv (static)
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

    k_scale = torch.tensor([0.1], dtype=torch.float32, device=qkv.device)
    v_scale = torch.tensor([0.1], dtype=torch.float32, device=qkv.device)
    q_scale_val = 2.0
    q_scale_inv = torch.tensor([1.0 / q_scale_val], dtype=torch.float32, device=qkv.device)

    kcache_fp8 = kcache.to(torch.float8_e4m3fn)
    vcache_fp8 = vcache.to(torch.float8_e4m3fn)

    if is_prefill:
        max_seqlens = int((q_index[1:] - q_index[:-1]).max().item())
    else:
        max_seqlens = mtp + 1  # tokens per request in decode

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
        quant_policy=quant_policy,
        max_seqlens=max_seqlens,
        q_scale_inv=q_scale_inv if quant_policy == 2 else None,
        q_norm_weight=q_norm_w if qk_norm_policy > 0 else None,
        k_norm_weight=k_norm_w if qk_norm_policy > 0 else None,
        qk_norm_policy=qk_norm_policy,
    )

    assert split_k_flag.shape == (num_seqlen.shape[0], num_kv_heads)
    assert split_k_flag.dtype == torch.int32

    if quant_policy == 1:  # dqskv: kernel computes dynamic per-token per-head scale
        if is_prefill:
            pad128 = ((max_seqlens + 127) // 128) * 128
            assert q_scale_out.shape == (num_seqlen.shape[0], num_q_heads, pad128)
            # dequant: select valid per-token scales via sequence-length mask
            seqlens = (q_index[1:] - q_index[:-1]).to(qkv.device)
            mask = torch.arange(pad128, device=qkv.device).expand(
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
    else:  # sqskv: static scale supplied by caller; no dynamic scale tensor returned
        assert q_scale_out is None
        rows = real_rows if real_rows is not None else q_fp8.shape[0]
        q_bf16 = (q_fp8[:rows].to(torch.float32) * q_scale_val).to(torch.bfloat16)

    if real_rows is not None:
        qkv_r, ns_r = qkv[:real_rows], num_seqlen[:num_req]
        qi_r, ki_r = q_index[: num_req + 1], kv_indices[:num_req]
    else:
        qkv_r, ns_r, qi_r, ki_r = qkv, num_seqlen, q_index, kv_indices

    ref_q = rope_norm_ref(
        kcache_ref, vcache_ref, qkv_r, cos_sin, ns_r, qi_r, ki_r, q_norm_w, k_norm_w, qk_norm_policy
    )
    assert allclose(ref_q, q_bf16, atol=0.5)
