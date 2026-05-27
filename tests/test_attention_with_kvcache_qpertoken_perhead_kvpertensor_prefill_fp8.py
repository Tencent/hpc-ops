import sys
import os
import pytest
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch
import math
import torch.nn.functional as F
from utils import allclose


def naive_attn_with_kvcache_func(
    q,
    k_cache,
    v_cache,
    qscale,
    kscale,
    vscale,
    cache_seqlens,
    page_table,
    causal=True,
):

    num_batch, num_seq_q, num_head_q, num_dim_qk = q.shape
    num_blocks, block_size, num_head_kv, _ = k_cache.shape
    _, _, _, num_dim_v = v_cache.shape
    _, _, max_seq_q_pad = qscale.shape

    num_group = num_head_q // num_head_kv
    output = torch.empty_like(q).to(torch.bfloat16)

    kvcache_blocks = (cache_seqlens + block_size - 1) // block_size

    for i in range(num_batch):
        BQ = q[i].transpose(0, 1).float()  # [num_heads, sq, head_dim]
        blk_ids = page_table[i, : kvcache_blocks[i]]
        num_seq_kv = cache_seqlens[i]
        BK = (
            k_cache[blk_ids, :, :, :]
            .reshape(-1, num_head_kv, num_dim_qk)
            .transpose(0, 1)[:, :num_seq_kv, :]
            .repeat_interleave(num_group, dim=0)
        ).float()
        BV = (
            v_cache[blk_ids, :, :, :]
            .reshape(-1, num_head_kv, num_dim_v)
            .transpose(0, 1)[:, :num_seq_kv, :]
            .repeat_interleave(num_group, dim=0)
        ).float()

        scale = qscale[i, :, :].unsqueeze(-1)[:, :num_seq_q, :]

        scores = torch.matmul(BQ, BK.transpose(-2, -1)) * scale * kscale[0] / math.sqrt(num_dim_qk)
        if causal:
            causal_mask = (
                torch.tril(torch.ones(num_seq_kv, num_seq_kv, device=q.device, dtype=torch.bool))[
                    (num_seq_kv - num_seq_q) :, :
                ]
                .unsqueeze(0)
                .unsqueeze(0)
            )  # (1, 1, num_seq_q, num_seq_kv)
        else:
            causal_mask = torch.ones(
                (1, 1, num_seq_q, num_seq_kv), device=q.device, dtype=torch.bool
            )

        scores = scores.masked_fill(~causal_mask, float("-inf"))

        # Mirror the kernel's online-softmax + fixed-P-scale fp8 quant path.
        attn_weights = torch.exp(scores - scores.max(dim=-1, keepdim=True)[0])
        gSum = attn_weights.sum(dim=-1, keepdim=True)
        attn_weights = attn_weights * 256.0
        attn_weights = attn_weights.to(torch.float8_e4m3fn).float()

        out_head = torch.matmul(attn_weights, BV)
        out_head = out_head / gSum
        out_head = out_head * (vscale[0] / 256.0)

        output[i] = out_head.transpose(1, 2)

    return output


try:
    from flash_attn_interface import flash_attn_varlen_func, flash_attn_with_kvcache

    gt_attention_func = naive_attn_with_kvcache_func  # flash_attn_with_kvcache
except Exception as e:
    print(f"execute naive_attn_func: {e}")
    gt_attention_func = naive_attn_with_kvcache_func


@pytest.mark.parametrize("kv_layout", ["nhd", "hnd"])
@pytest.mark.parametrize("num_batch", [4])
@pytest.mark.parametrize("num_seq_q", [100, 500, 1000, 1500, 3904])
@pytest.mark.parametrize("num_seq_kv", [3904])
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("num_head_q", [4])
@pytest.mark.parametrize("num_head_kv", [1])
@pytest.mark.parametrize("num_dim_qk", [128])
@pytest.mark.parametrize("num_dim_v", [128])
@pytest.mark.parametrize("use_output", [False])
def test_attention_with_kvcache_prefill_fp8(
    kv_layout,
    num_batch,
    num_seq_q,
    num_seq_kv,
    block_size,
    num_head_q,
    num_head_kv,
    num_dim_qk,
    num_dim_v,
    use_output,
):

    torch.manual_seed(10086)
    torch.cuda.manual_seed(10086)
    T = torch.bfloat16
    T1 = torch.float8_e4m3fn

    num_seq_q_pad = (num_seq_q + 127) // 128 * 128

    Q = (
        torch.randn(
            (num_batch, num_seq_q, num_head_q, num_dim_qk),
            dtype=T,
            device="cuda",
        )
        / math.sqrt(num_dim_qk)
    ).to(T1)
    K = (
        torch.randn((num_batch * num_seq_q, num_head_kv, num_dim_qk), dtype=T, device="cuda")
        / math.sqrt(num_dim_qk)
    ).to(T1)
    V = torch.randn((num_batch * num_seq_q, num_head_kv, num_dim_v), dtype=T, device="cuda").to(T1)
    qscale = (
        torch.abs(
            torch.randn((num_batch, num_head_q, num_seq_q_pad), dtype=torch.float32, device="cuda")
        )
        / 10
    )

    kscale = torch.randn((1), dtype=torch.float32, device="cuda").abs() * 10
    print(f"kscale={kscale}")
    vscale = torch.randn((1), dtype=torch.float32, device="cuda")

    seqlens_q = torch.full((num_batch,), num_seq_q, dtype=torch.int32, device="cuda")
    seqlens_kvcache = torch.full((num_batch,), num_seq_kv, dtype=torch.int32, device="cuda")
    cu_seqlens_q = torch.cumsum(
        torch.cat([torch.tensor([0], dtype=torch.int32, device="cuda"), seqlens_q]), dim=0
    ).to(torch.int32)
    cu_seqlens_kvcache = torch.cumsum(
        torch.cat([torch.tensor([0], dtype=torch.int32, device="cuda"), seqlens_kvcache]), dim=0
    ).to(torch.int32)

    max_num_blocks = num_batch * (num_seq_kv + block_size - 1) // block_size * 2
    kvcache_blocks = (seqlens_kvcache + block_size - 1) // block_size
    total_kvcache_blocks = sum(kvcache_blocks)
    max_kvcache_blocks = max(kvcache_blocks)
    max_seqlens_q = max(seqlens_q)
    max_seqlens_kvcache = max(seqlens_kvcache)

    kvcache = torch.randn(
        max_num_blocks, 2, block_size, num_head_kv, num_dim_qk, dtype=T, device="cuda"
    ).to(T1)
    if kv_layout == "hnd":
        kcache = kvcache[:, 0].transpose(1, 2).contiguous().transpose(1, 2)
        vcache = kvcache[:, 1].transpose(1, 2).contiguous().transpose(1, 2)
    else:
        kcache = kvcache[:, 0]
        vcache = kvcache[:, 1]
    packed_block_ids = torch.randperm(max_num_blocks)[:total_kvcache_blocks].to(torch.int32).cuda()

    cu_blocks = 0
    block_ids = torch.empty(num_batch, max_kvcache_blocks, dtype=torch.int32, device="cuda")
    for i in range(num_batch):
        block_ids[i, : kvcache_blocks[i]] = packed_block_ids[
            cu_blocks : cu_blocks + kvcache_blocks[i]
        ]
        cu_blocks += kvcache_blocks[i]

    for i in range(10):
        gt = gt_attention_func(
            q=Q,
            k_cache=kcache,
            v_cache=vcache,
            qscale=qscale,
            kscale=kscale,
            vscale=vscale,
            cache_seqlens=seqlens_kvcache,
            page_table=block_ids,
            causal=True,
        )

        if use_output:
            my = torch.empty_like(Q.reshape(-1, num_head_q, num_dim_qk))
            hpc.attention_with_kvcache_prefill_fp8(
                Q.reshape(-1, num_head_q, num_dim_qk),
                kcache,
                vcache,
                qscale,
                kscale,
                vscale,
                cu_seqlens_q,
                block_ids,
                seqlens_kvcache,
                max_seqlens_q,
                quant_type=hpc.QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR,
                output=my,
            )
        else:
            my = hpc.attention_with_kvcache_prefill_fp8(
                Q.reshape(-1, num_head_q, num_dim_qk),
                kcache,
                vcache,
                qscale,
                kscale,
                vscale,
                cu_seqlens_q,
                block_ids,
                seqlens_kvcache,
                max_seqlens_q,
                quant_type=hpc.QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR,
            )
    gt = gt.reshape(-1, num_head_q, num_dim_v)

    print("\ngt\n")
    print(gt[:, 0, :])
    print("\nmy\n")
    print(my[:, 0, :])
    print("\n diff \n")
    print((gt - my)[:64, 0, :1])

    assert allclose(gt, my, atol=0.05)


# -----------------------------------------------------------------------------
# Fixed P-scale coverage
# -----------------------------------------------------------------------------


def naive_attn_with_kvcache_fixed_pscale_func(
    q,
    k_cache,
    v_cache,
    qscale,
    kscale,
    vscale,
    cache_seqlens,
    page_table,
    causal=True,
):
    """Reference for the kernel's fixed P-scale=256 FP8 path."""
    num_batch, num_seq_q, num_head_q, num_dim_qk = q.shape
    num_blocks, block_size, num_head_kv, _ = k_cache.shape
    _, _, _, num_dim_v = v_cache.shape

    num_group = num_head_q // num_head_kv
    output = torch.empty_like(q).to(torch.bfloat16)

    kvcache_blocks = (cache_seqlens + block_size - 1) // block_size

    for i in range(num_batch):
        BQ = q[i].transpose(0, 1).float()  # [num_heads, sq, head_dim]
        blk_ids = page_table[i, : kvcache_blocks[i]]
        num_seq_kv = cache_seqlens[i]
        BK = (
            k_cache[blk_ids, :, :, :]
            .reshape(-1, num_head_kv, num_dim_qk)
            .transpose(0, 1)[:, :num_seq_kv, :]
            .repeat_interleave(num_group, dim=0)
        ).float()
        BV = (
            v_cache[blk_ids, :, :, :]
            .reshape(-1, num_head_kv, num_dim_v)
            .transpose(0, 1)[:, :num_seq_kv, :]
            .repeat_interleave(num_group, dim=0)
        ).float()

        scale = qscale[i, :, :].unsqueeze(-1)[:, :num_seq_q, :]
        scores = torch.matmul(BQ, BK.transpose(-2, -1)) * scale * kscale[0] / math.sqrt(num_dim_qk)
        if causal:
            causal_mask = (
                torch.tril(torch.ones(num_seq_kv, num_seq_kv, device=q.device, dtype=torch.bool))[
                    (num_seq_kv - num_seq_q) :, :
                ]
                .unsqueeze(0)
                .unsqueeze(0)
            )
        scores = scores.masked_fill(~causal_mask, float("-inf"))

        # Mirror the kernel's online-softmax + un-normalised fp8 quant
        # path: quantise exp(scores - row_max) and divide by row_sum after
        # the PV gemm. Doing F.softmax-then-quant would collapse most
        # values into e4m3's low-resolution region and break atol=0.1.
        attn_weights = torch.exp(scores - scores.max(dim=-1, keepdim=True)[0])
        gSum = attn_weights.sum(dim=-1, keepdim=True)

        attn_weights = attn_weights * 256.0
        attn_weights = attn_weights.to(torch.float8_e4m3fn).float()

        out_head = torch.matmul(attn_weights, BV)
        out_head = out_head / gSum
        out_head = out_head * (vscale[0] / 256.0)

        output[i] = out_head.transpose(1, 2)

    return output


@pytest.mark.parametrize("kv_layout", ["nhd"])
@pytest.mark.parametrize("num_batch", [4])
@pytest.mark.parametrize("num_seq_q", [128, 256, 512])
@pytest.mark.parametrize("num_seq_kv", [1024])
@pytest.mark.parametrize("num_head_q", [4, 16])
@pytest.mark.parametrize("num_head_kv", [1, 4])
def test_attention_with_kvcache_prefill_fp8_fixed_pscale(
    kv_layout,
    num_batch,
    num_seq_q,
    num_seq_kv,
    num_head_q,
    num_head_kv,
):
    if num_head_q % num_head_kv != 0:
        pytest.skip("num_head_q must be a multiple of num_head_kv")

    torch.manual_seed(10086)
    torch.cuda.manual_seed(10086)

    block_size = 64
    num_dim_qk = 128
    num_dim_v = 128
    T = torch.bfloat16
    T1 = torch.float8_e4m3fn

    num_seq_q_pad = (num_seq_q + 127) // 128 * 128

    Q = (
        torch.randn((num_batch, num_seq_q, num_head_q, num_dim_qk), dtype=T, device="cuda")
        / math.sqrt(num_dim_qk)
    ).to(T1)
    qscale = (
        torch.abs(
            torch.randn((num_batch, num_head_q, num_seq_q_pad), dtype=torch.float32, device="cuda")
        )
        / 10
    )
    kscale = torch.randn((1,), dtype=torch.float32, device="cuda").abs() * 10
    vscale = torch.randn((1,), dtype=torch.float32, device="cuda")

    seqlens_q = torch.full((num_batch,), num_seq_q, dtype=torch.int32, device="cuda")
    seqlens_kvcache = torch.full((num_batch,), num_seq_kv, dtype=torch.int32, device="cuda")
    cu_seqlens_q = torch.cumsum(
        torch.cat([torch.tensor([0], dtype=torch.int32, device="cuda"), seqlens_q]), dim=0
    ).to(torch.int32)

    max_num_blocks = num_batch * (num_seq_kv + block_size - 1) // block_size * 2
    kvcache_blocks = (seqlens_kvcache + block_size - 1) // block_size
    total_kvcache_blocks = sum(kvcache_blocks)
    max_kvcache_blocks = max(kvcache_blocks)
    max_seqlens_q = max(seqlens_q)

    kvcache = torch.randn(
        max_num_blocks, 2, block_size, num_head_kv, num_dim_qk, dtype=T, device="cuda"
    ).to(T1)
    kcache = kvcache[:, 0]
    vcache = kvcache[:, 1]
    packed_block_ids = torch.randperm(max_num_blocks)[:total_kvcache_blocks].to(torch.int32).cuda()

    cu_blocks = 0
    block_ids = torch.empty(num_batch, max_kvcache_blocks, dtype=torch.int32, device="cuda")
    for i in range(num_batch):
        block_ids[i, : kvcache_blocks[i]] = packed_block_ids[
            cu_blocks : cu_blocks + kvcache_blocks[i]
        ]
        cu_blocks += kvcache_blocks[i]

    gt = naive_attn_with_kvcache_fixed_pscale_func(
        q=Q,
        k_cache=kcache,
        v_cache=vcache,
        qscale=qscale,
        kscale=kscale,
        vscale=vscale,
        cache_seqlens=seqlens_kvcache,
        page_table=block_ids,
        causal=True,
    )

    my = hpc.attention_with_kvcache_prefill_fp8(
        Q.reshape(-1, num_head_q, num_dim_qk),
        kcache,
        vcache,
        qscale,
        kscale,
        vscale,
        cu_seqlens_q,
        block_ids,
        seqlens_kvcache,
        max_seqlens_q,
        quant_type=hpc.QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR,
    )
    gt = gt.reshape(-1, num_head_q, num_dim_v)

    # atol = 0.05: golden mirrors the kernel's full round-trip. SQ<=512 in
    # the parametrized matrix; 30-seed sweep gives worst ~0.014. 0.05 leaves
    # ~3.5x headroom.
    assert allclose(
        gt, my, atol=0.05
    ), "prefill fp8 (qpertoken_kvpertensor) diverges from fixed-P-scale golden beyond atol=0.05"
