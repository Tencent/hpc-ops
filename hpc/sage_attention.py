from typing import Optional

import torch
from torch import Tensor


def sageattn_qk_int8_pv_fp8(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    tensor_layout: str = "HND",
    output: Optional[Tensor] = None,
) -> Tensor:
    """SageAttention v2 (QK INT8, PV FP8) for SM89 / SM90 / SM120, non-causal.

    Args:
        q: Query tensor  -  NHD: [B, S, Hq, D] or HND: [B, Hq, S, D].
        k: Key tensor    -  NHD: [B, S, Hkv, D] or HND: [B, Hkv, S, D].
        v: Value tensor  -  same layout as k.
        tensor_layout: "HND" or "NHD".
        output: Optional pre-allocated output tensor (bf16, same shape as q).

    Returns:
        Tensor: Attention output in bfloat16.
    """
    assert q.is_cuda, "Input tensors must be on cuda."
    assert q.dtype == torch.bfloat16, "q/k/v must be bfloat16"
    assert q.dtype == k.dtype == v.dtype
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
    assert q.size(-1) == 128, f"Only head_dim=128 is supported, got {q.size(-1)}"

    if tensor_layout not in ("HND", "NHD"):
        raise ValueError(f"Unknown tensor_layout: {tensor_layout}")

    is_nhd = tensor_layout == "NHD"
    layout_int = 0 if is_nhd else 1

    if torch.cuda.get_device_capability()[0] == 9:
        # SM90: the kernel allocates its own scratch buffers. is_causal=0.
        return torch.ops.hpc.sage_attn_fused(q, k, v, output, layout_int, 0)

    # SM89 / SM120: the caller pre-allocates all intermediate buffers.
    device = q.device
    batch_size = q.size(0)
    if is_nhd:
        qo_len, num_head_q = q.size(1), q.size(2)
        kv_len, num_head_kv = k.size(1), k.size(2)
    else:
        num_head_q, qo_len = q.size(1), q.size(2)
        num_head_kv, kv_len = k.size(1), k.size(2)
    head_dim = q.size(3)

    if is_nhd:
        km = torch.empty(batch_size, 1, num_head_kv, head_dim, dtype=torch.float32, device=device)
    else:
        km = torch.empty(batch_size, num_head_kv, 1, head_dim, dtype=torch.float32, device=device)
    v_scale = torch.empty(batch_size, num_head_kv, head_dim, dtype=torch.float32, device=device)

    q_int8 = torch.empty_like(q, dtype=torch.int8)
    k_int8 = torch.empty_like(k, dtype=torch.int8)
    q_scale = torch.empty(
        batch_size, num_head_q, (qo_len + 127) // 128 * 64, dtype=torch.float32, device=device
    )
    k_scale = torch.empty(
        batch_size, num_head_kv, (kv_len + 63) // 64 * 4, dtype=torch.float32, device=device
    )
    padded_len = (kv_len + 63) // 64 * 64
    if is_nhd:
        v_fp8 = torch.empty(
            batch_size, head_dim, num_head_kv, padded_len, dtype=torch.float8_e4m3fn, device=device
        )
    else:
        v_fp8 = torch.empty(
            batch_size, num_head_kv, head_dim, padded_len, dtype=torch.float8_e4m3fn, device=device
        )

    if output is None:
        output = torch.empty_like(q, dtype=torch.bfloat16)

    torch.ops.hpc.sage_attn_fused(
        q,
        k,
        v,
        km,
        v_scale,
        q_int8,
        q_scale,
        k_int8,
        k_scale,
        v_fp8,
        output,
        layout_int,
    )

    return output
