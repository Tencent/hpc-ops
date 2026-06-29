import pytest
import hpc
import torch


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.size(-1) // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_interleaved_rope(x: torch.Tensor, mrope_section: list) -> torch.Tensor:
    half_rd = x.shape[-1]
    idx = torch.arange(half_rd, device=x.device)
    h_mask = (idx % 3 == 1) & (idx < 3 * mrope_section[1])
    w_mask = (idx % 3 == 2) & (idx < 3 * mrope_section[2])
    out = torch.where(h_mask, x[1], x[0])
    out = torch.where(w_mask, x[2], out)
    return out


def rms_norm_hy(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return weight * hidden_states.to(input_dtype)


def do_rope_fp32(rotary_dim, cos, sin, query, key):
    cos = torch.cat((cos, cos), dim=-1).unsqueeze(1)
    sin = torch.cat((sin, sin), dim=-1).unsqueeze(1)
    num_tokens = query.size(0) if query.ndim == 3 else query.size(0) * query.size(1)
    head_size = query.size(-1)

    def rope(x):
        in_dtype = x.dtype
        in_shape = x.shape
        x = x.view(num_tokens, -1, head_size)
        rot_fp32 = x[..., :rotary_dim].float()
        x_pass = x[..., rotary_dim:]
        rot_out = (rot_fp32 * cos) + (_rotate_half(rot_fp32) * sin)
        rot_out = rot_out.to(in_dtype)
        if x_pass.numel() > 0:
            x = torch.cat((rot_out, x_pass), dim=-1)
        else:
            x = rot_out
        return x.reshape(in_shape)

    return rope(query), rope(key)


def rope_fp32(mrope_section, rotary_dim, cos_sin_cache, positions, query, key):
    assert positions.ndim in (1, 2)
    cos_sin = cos_sin_cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    if positions.ndim == 2:
        cos = apply_interleaved_rope(cos, mrope_section)
        sin = apply_interleaved_rope(sin, mrope_section)
    return do_rope_fp32(rotary_dim, cos, sin, query, key)


def ref_fuse_mm_qknorm_concat_rope_hy(
    und_qkv,
    gen_qkv,
    num_heads_q,
    num_heads_k,
    num_heads_v,
    head_dim,
    und_q_norm_weight,
    und_k_norm_weight,
    gen_q_norm_weight,
    gen_k_norm_weight,
    norm_eps,
    positions,
    mrope_section,
    cos_sin_cache,
    und_indices,
    gen_indices,
    cat_indices,
):
    und_q, und_k, und_v = und_qkv.split(
        [num_heads_q * head_dim, num_heads_k * head_dim, num_heads_v * head_dim], dim=-1
    )
    gen_q, gen_k, gen_v = gen_qkv.split(
        [num_heads_q * head_dim, num_heads_k * head_dim, num_heads_v * head_dim], dim=-1
    )
    und_q = rms_norm_hy(und_q.unflatten(-1, (-1, head_dim)), und_q_norm_weight, norm_eps)
    und_k = rms_norm_hy(und_k.unflatten(-1, (-1, head_dim)), und_k_norm_weight, norm_eps)
    gen_q = rms_norm_hy(gen_q.unflatten(-1, (-1, head_dim)), gen_q_norm_weight, norm_eps)
    gen_k = rms_norm_hy(gen_k.unflatten(-1, (-1, head_dim)), gen_k_norm_weight, norm_eps)

    combined_q = torch.cat((und_q, gen_q), dim=0)[cat_indices]
    combined_k = torch.cat((und_k, gen_k), dim=0)[cat_indices]
    combined_v = torch.cat((und_v, gen_v), dim=0)[cat_indices].unflatten(-1, (-1, head_dim))

    combined_q, combined_k = rope_fp32(
        mrope_section, head_dim, cos_sin_cache, positions, combined_q, combined_k
    )
    return combined_q.type_as(und_v), combined_k.type_as(und_v), combined_v


def ref_fuse_single_qknorm_rope_hy(
    qkv,
    num_heads_q,
    num_heads_k,
    num_heads_v,
    head_dim,
    q_norm_weight,
    k_norm_weight,
    norm_eps,
    positions,
    mrope_section,
    cos_sin_cache,
):
    q, k, v = qkv.split(
        [num_heads_q * head_dim, num_heads_k * head_dim, num_heads_v * head_dim], dim=-1
    )
    q = rms_norm_hy(q.unflatten(-1, (-1, head_dim)), q_norm_weight, norm_eps)
    k = rms_norm_hy(k.unflatten(-1, (-1, head_dim)), k_norm_weight, norm_eps)

    cos_sin = cos_sin_cache[positions[0]]
    cos, sin = cos_sin.chunk(2, dim=-1)

    q, k = do_rope_fp32(head_dim, cos, sin, q, k)

    return q.type_as(v), k.type_as(v), v.unflatten(-1, (-1, head_dim))


# ---------- Tests ----------


def generate_cos_sin_cache(max_position, head_dim, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_position).float()
    freqs = torch.outer(t, inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)


@pytest.mark.parametrize("num_tokens", [1, 17, 96])
@pytest.mark.parametrize("num_heads_q,num_heads_k,num_heads_v", [(8, 1, 1), (4, 2, 2), (8, 8, 8)])
@pytest.mark.parametrize("mrope_section", [[16, 16, 16], [20, 10, 6]])
def test_fused_qk_rmsnorm_rope_matches_single_reference(
    num_tokens, num_heads_q, num_heads_k, num_heads_v, mrope_section
):
    torch.manual_seed(42)

    head_dim = 128
    hidden = num_heads_q * head_dim + num_heads_k * head_dim + num_heads_v * head_dim
    max_pos = 2048

    qkv = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device="cuda")
    q_w = torch.randn(head_dim, dtype=torch.bfloat16, device="cuda")
    k_w = torch.randn(head_dim, dtype=torch.bfloat16, device="cuda")

    positions = torch.stack(
        [
            torch.randint(0, max_pos, (num_tokens,), device="cuda"),
            torch.randint(0, max_pos, (num_tokens,), device="cuda"),
            torch.randint(0, max_pos, (num_tokens,), device="cuda"),
        ]
    ).to(torch.int64)

    cos_sin_cache = generate_cos_sin_cache(max_pos, head_dim).to(dtype=torch.float32, device="cuda")
    norm_eps = 1e-6

    ref_q, ref_k, ref_v = ref_fuse_single_qknorm_rope_hy(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        q_w,
        k_w,
        norm_eps,
        positions,
        mrope_section,
        cos_sin_cache,
    )

    my_q, my_k, my_v = hpc.fused_qk_rmsnorm_rope(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        q_w,
        k_w,
        norm_eps,
        positions,
        cos_sin_cache,
    )
    my_q_1d, my_k_1d, my_v_1d = hpc.fused_qk_rmsnorm_rope(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        q_w,
        k_w,
        norm_eps,
        positions[0],
        cos_sin_cache,
    )

    assert torch.equal(ref_q, my_q), "Q mismatch"
    assert torch.equal(ref_k, my_k), "K mismatch"
    assert torch.equal(ref_v, my_v), "V mismatch"
    assert torch.equal(ref_q, my_q_1d), "Q mismatch for 1D positions"
    assert torch.equal(ref_k, my_k_1d), "K mismatch for 1D positions"
    assert torch.equal(ref_v, my_v_1d), "V mismatch for 1D positions"


@pytest.mark.parametrize("und_len,gen_len", [(32, 64), (17, 43), (1, 1)])
@pytest.mark.parametrize("num_heads_q,num_heads_k,num_heads_v", [(8, 1, 1), (4, 2, 2), (8, 8, 8)])
@pytest.mark.parametrize("mrope_section", [[16, 16, 16], [20, 10, 6]])
@pytest.mark.parametrize("shuffle_cat", [False, True])
def test_fused_qk_rmsnorm_mrope(
    und_len, gen_len, num_heads_q, num_heads_k, num_heads_v, mrope_section, shuffle_cat
):
    torch.manual_seed(42)

    head_dim = 128
    total_tokens = und_len + gen_len
    hidden = num_heads_q * head_dim + num_heads_k * head_dim + num_heads_v * head_dim
    max_pos = 2048

    und_qkv = torch.randn(und_len, hidden, dtype=torch.bfloat16, device="cuda")
    gen_qkv = torch.randn(gen_len, hidden, dtype=torch.bfloat16, device="cuda")

    und_q_w = torch.randn(head_dim, dtype=torch.bfloat16, device="cuda")
    und_k_w = torch.randn(head_dim, dtype=torch.bfloat16, device="cuda")
    gen_q_w = torch.randn(head_dim, dtype=torch.bfloat16, device="cuda")
    gen_k_w = torch.randn(head_dim, dtype=torch.bfloat16, device="cuda")

    positions = torch.stack(
        [
            torch.randint(0, max_pos, (total_tokens,), device="cuda"),
            torch.randint(0, max_pos, (total_tokens,), device="cuda"),
            torch.randint(0, max_pos, (total_tokens,), device="cuda"),
        ]
    ).to(torch.int64)

    cos_sin_cache = generate_cos_sin_cache(max_pos, head_dim).to(dtype=torch.float32, device="cuda")

    if shuffle_cat:
        cat_indices = torch.randperm(total_tokens, device="cuda", dtype=torch.int64)
    else:
        cat_indices = torch.arange(total_tokens, device="cuda", dtype=torch.int64)

    und_indices = torch.arange(und_len, device="cuda", dtype=torch.int64)
    gen_indices = torch.arange(gen_len, device="cuda", dtype=torch.int64)

    norm_eps = 1e-6

    ref_q, ref_k, ref_v = ref_fuse_mm_qknorm_concat_rope_hy(
        und_qkv,
        gen_qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        und_q_w,
        und_k_w,
        gen_q_w,
        gen_k_w,
        norm_eps,
        positions,
        mrope_section,
        cos_sin_cache,
        und_indices,
        gen_indices,
        cat_indices,
    )

    my_q, my_k, my_v = hpc.fused_qk_rmsnorm_mrope(
        und_qkv,
        gen_qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        und_q_w,
        und_k_w,
        gen_q_w,
        gen_k_w,
        norm_eps,
        positions,
        mrope_section,
        cos_sin_cache,
        und_indices,
        gen_indices,
        cat_indices,
    )

    assert torch.equal(ref_q, my_q), "Q mismatch"
    assert torch.equal(ref_k, my_k), "K mismatch"
    assert torch.equal(ref_v, my_v), "V mismatch"
