import argparse
import os
import sys
from dataclasses import dataclass
from statistics import median

import torch

try:
    import hpc
except ImportError:
    hpc = None

try:
    import flashinfer
except ImportError:
    flashinfer = None


@dataclass
class RopeInput:
    qkv: torch.Tensor
    num_seqlen: torch.Tensor
    q_index: torch.Tensor
    kcache: torch.Tensor
    vcache: torch.Tensor
    kv_indices: torch.Tensor
    q_norm_w: torch.Tensor
    k_norm_w: torch.Tensor
    cos_sin: torch.Tensor
    real_rows: int | None


def generate_cos_sin_cache(max_position, head_dim, base=10000.0):
    # Keep trig ops on CPU for wider compatibility across CUDA stacks.
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_position, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)


def generate_kv_block_indices(kcache, req_length, device):
    num_req = len(req_length)
    kv_block_size = kcache.shape[1]
    num_blocks_per_req = [(l + kv_block_size - 1) // kv_block_size for l in req_length]
    shuffled = torch.randperm(kcache.shape[0], device=device)
    kv_idx = torch.ones(num_req, max(num_blocks_per_req) + 4, dtype=torch.int32, device=device) * -1
    offset = 0
    for i, n in enumerate(num_blocks_per_req):
        kv_idx[i, :n] = shuffled[offset : offset + n]
        offset += n
    return kv_idx


def pad_decode_inputs_to_align8(qkv, num_seqlen, q_index, kv_indices):
    nb = qkv.shape[0]
    pb = ((nb + 7) // 8) * 8
    if pb == nb:
        return qkv, num_seqlen, q_index, kv_indices, nb

    pr = q_index[-1].item()
    qkv = torch.cat([qkv, torch.zeros(pb - nb, qkv.shape[1], dtype=qkv.dtype, device=qkv.device)])
    num_seqlen = torch.cat(
        [
            num_seqlen,
            torch.zeros(pb - nb, dtype=num_seqlen.dtype, device=num_seqlen.device),
        ]
    )
    q_index = torch.cat(
        [q_index, torch.full((pb - nb,), pr, dtype=q_index.dtype, device=q_index.device)]
    )
    kv_indices = torch.cat(
        [
            kv_indices,
            torch.zeros(pb - nb, kv_indices.shape[1], dtype=kv_indices.dtype, device=kv_indices.device),
        ]
    )
    return qkv, num_seqlen, q_index, kv_indices, nb


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
    if v_head_dim is None:
        v_head_dim = qk_head_dim

    hidden = num_q_heads * qk_head_dim + num_kv_heads * qk_head_dim + num_kv_heads * v_head_dim

    cos_sin = generate_cos_sin_cache(max_rope_position, qk_head_dim).to(torch.float32).to(device)
    kcache = torch.randn(
        max_num_kv_blocks,
        kv_block_size,
        num_kv_heads,
        qk_head_dim,
        dtype=dtype,
        device=device,
    )
    vcache = torch.randn(
        max_num_kv_blocks,
        kv_block_size,
        num_kv_heads,
        v_head_dim,
        dtype=dtype,
        device=device,
    )
    q_norm_w = torch.randn(qk_head_dim, dtype=torch.float32, device=device)
    k_norm_w = torch.randn(qk_head_dim, dtype=torch.float32, device=device)

    if is_prefill:
        req_len = torch.randint(20, 200, (num_req,), device=device).tolist()
        qkv_full = torch.randn(sum(req_len), hidden, dtype=dtype, device=device)
        req_len_t = torch.tensor(req_len, device=device)
        q_len_t = torch.min((torch.rand(num_req, device=device) * req_len_t).long() + 1, req_len_t)
        cumsum = torch.cumsum(req_len_t, dim=0)
        qkv = torch.cat([qkv_full[cumsum[i] - q_len_t[i] : cumsum[i]] for i in range(num_req)])
        q_index = torch.cat(
            [torch.zeros(1, device=device, dtype=torch.int64), torch.cumsum(q_len_t, 0)]
        ).to(torch.int32)
        num_seqlen = torch.tensor(req_len, dtype=torch.int32, device=device)
        kv_indices = generate_kv_block_indices(kcache, req_len, device)
        real_rows = None
    else:
        tpr = mtp + 1
        exist_len = torch.randint(20, 200, (num_req,), device=device).tolist()
        upd_len = [x + tpr for x in exist_len]
        qkv_raw = torch.randn(num_req * tpr, hidden, dtype=dtype, device=device)
        q_idx_raw = torch.arange(0, (num_req + 1) * tpr, tpr, device=device, dtype=torch.int32)
        num_seqlen_raw = torch.tensor(upd_len, dtype=torch.int32, device=device)
        kv_idx_raw = generate_kv_block_indices(kcache, upd_len, device)
        qkv, num_seqlen, q_index, kv_indices, real_rows = pad_decode_inputs_to_align8(
            qkv_raw,
            num_seqlen_raw,
            q_idx_raw,
            kv_idx_raw,
        )

    return RopeInput(
        qkv=qkv,
        num_seqlen=num_seqlen,
        q_index=q_index,
        kcache=kcache,
        vcache=vcache,
        kv_indices=kv_indices,
        q_norm_w=q_norm_w,
        k_norm_w=k_norm_w,
        cos_sin=cos_sin,
        real_rows=real_rows,
    )


def bench_us(step_fn, warmup, iters):
    for _ in range(warmup):
        step_fn()
    torch.cuda.synchronize()
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(iters)
    ]
    for s, e in events:
        s.record()
        step_fn()
        e.record()
    torch.cuda.synchronize()
    return median(s.elapsed_time(e) * 1000.0 for s, e in events)  # us


def rms_norm(x, weight, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def split_qkv(inp, num_q_heads, num_kv_heads, qk_head_dim):
    q_dim = num_q_heads * qk_head_dim
    k_dim = num_kv_heads * qk_head_dim
    q = inp.qkv[:, :q_dim].view(-1, num_q_heads, qk_head_dim)
    k = inp.qkv[:, q_dim : q_dim + k_dim].view(-1, num_kv_heads, qk_head_dim)
    v = inp.qkv[:, q_dim + k_dim : q_dim + 2 * k_dim].view(-1, num_kv_heads, qk_head_dim)
    return q, k, v


def build_flashinfer_meta(inp):
    num_req = inp.q_index.numel() - 1
    q_lens = (inp.q_index[1:] - inp.q_index[:-1]).to(torch.int32)
    kv_block_size = inp.kcache.shape[1]

    num_pages_per_req = []
    kv_last_page_len = []
    kv_indices_flat = []
    for i in range(num_req):
        seqlen = int(inp.num_seqlen[i].item())
        pages = (seqlen + kv_block_size - 1) // kv_block_size if seqlen > 0 else 0
        num_pages_per_req.append(pages)
        kv_last_page_len.append(((seqlen - 1) % kv_block_size) + 1 if seqlen > 0 else 0)
        if pages > 0:
            kv_indices_flat.append(inp.kv_indices[i, :pages])

    device = inp.qkv.device
    num_pages_per_req_t = torch.tensor(num_pages_per_req, dtype=torch.int32, device=device)
    kv_last_page_len_t = torch.tensor(kv_last_page_len, dtype=torch.int32, device=device)
    kv_indptr = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.cumsum(num_pages_per_req_t, dim=0),
        ]
    )
    kv_append_indptr = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), torch.cumsum(q_lens, dim=0)]
    )
    nnz = int(kv_append_indptr[-1].item())
    seq_lens = flashinfer.get_seq_lens(kv_indptr, kv_last_page_len_t, kv_block_size)
    batch_indices, positions = flashinfer.get_batch_indices_positions(kv_append_indptr, seq_lens, nnz)
    if kv_indices_flat:
        kv_indices = torch.cat(kv_indices_flat).to(torch.int32)
    else:
        kv_indices = torch.empty(0, dtype=torch.int32, device=device)

    return {
        "batch_indices": batch_indices,
        "positions": positions,
        "kv_indices": kv_indices,
        "kv_indptr": kv_indptr,
        "kv_last_page_len": kv_last_page_len_t,
        "page_size": kv_block_size,
        "nnz": nnz,
    }


def make_step_bf16(inp, is_prefill, qk_norm_policy, use_nvtx=False):
    qkv, num_seqlen, q_index = inp.qkv, inp.num_seqlen, inp.q_index
    kcache, vcache = inp.kcache, inp.vcache
    kv_indices, q_norm_w, k_norm_w, cos_sin = inp.kv_indices, inp.q_norm_w, inp.k_norm_w, inp.cos_sin

    def step():
        if use_nvtx:
            torch.cuda.nvtx.range_push("rope_norm_store_kv")
        hpc.rope_norm_store_kv(
            kcache, vcache, qkv, cos_sin, num_seqlen, q_index, kv_indices, is_prefill,
            q_norm_weight=q_norm_w if qk_norm_policy > 0 else None,
            k_norm_weight=k_norm_w if qk_norm_policy > 0 else None,
            qk_norm_policy=qk_norm_policy,
        )
        if use_nvtx:
            torch.cuda.nvtx.range_pop()

    return step


def make_step_fp8(inp, is_prefill, qk_norm_policy, quant_policy, mtp, use_nvtx=False):
    qkv, num_seqlen, q_index = inp.qkv, inp.num_seqlen, inp.q_index
    kcache, vcache = inp.kcache, inp.vcache
    kv_indices, q_norm_w, k_norm_w, cos_sin = inp.kv_indices, inp.q_norm_w, inp.k_norm_w, inp.cos_sin
    kcache_fp8 = kcache.to(torch.float8_e4m3fn)
    vcache_fp8 = vcache.to(torch.float8_e4m3fn)
    k_scale = torch.tensor([0.1], dtype=torch.float32, device=qkv.device)
    v_scale = torch.tensor([0.1], dtype=torch.float32, device=qkv.device)
    q_scale_inv = torch.tensor([0.5], dtype=torch.float32, device=qkv.device)
    max_seqlens = (
        int((q_index[1:] - q_index[:-1]).max().item()) if is_prefill else mtp + 1
    )

    def step():
        if use_nvtx:
            torch.cuda.nvtx.range_push("rope_norm_store_kv_fp8")
        hpc.rope_norm_store_kv_fp8(
            key_cache=kcache_fp8, value_cache=vcache_fp8, qkv=qkv, cos_sin=cos_sin,
            num_seqlen_per_req=num_seqlen, q_index=q_index, kvcache_indices=kv_indices,
            is_prefill=is_prefill, k_scale=k_scale, v_scale=v_scale,
            quant_policy=quant_policy, max_seqlens=max_seqlens,
            q_scale_inv=q_scale_inv if quant_policy == 2 else None,
            q_norm_weight=q_norm_w if qk_norm_policy > 0 else None,
            k_norm_weight=k_norm_w if qk_norm_policy > 0 else None,
            qk_norm_policy=qk_norm_policy,
        )
        if use_nvtx:
            torch.cuda.nvtx.range_pop()

    return step


def make_step_flashinfer_bf16(inp, num_q_heads, num_kv_heads, qk_head_dim, qk_norm_policy, use_nvtx=False):
    q, k, v = split_qkv(inp, num_q_heads, num_kv_heads, qk_head_dim)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    meta = build_flashinfer_meta(inp)
    q_norm_w = inp.q_norm_w
    k_norm_w = inp.k_norm_w
    q_flat = q.view(q.shape[0], -1)
    k_flat = k.view(k.shape[0], -1)

    def step():
        if use_nvtx:
            torch.cuda.nvtx.range_push("flashinfer_rope_append")

        if qk_norm_policy == 2:
            q.copy_(rms_norm(q, q_norm_w))
            k.copy_(rms_norm(k, k_norm_w))

        flashinfer.rope.apply_rope_with_cos_sin_cache_inplace(
            positions=meta["positions"],
            query=q_flat,
            key=k_flat,
            head_size=qk_head_dim,
            cos_sin_cache=inp.cos_sin,
            is_neox=True,
        )

        if qk_norm_policy == 1:
            q.copy_(rms_norm(q, q_norm_w))
            k.copy_(rms_norm(k, k_norm_w))

        flashinfer.append_paged_kv_cache(
            append_key=k,
            append_value=v,
            batch_indices=meta["batch_indices"],
            positions=meta["positions"],
            paged_kv_cache=(inp.kcache, inp.vcache),
            kv_indices=meta["kv_indices"],
            kv_indptr=meta["kv_indptr"],
            kv_last_page_len=meta["kv_last_page_len"],
            kv_layout="NHD",
        )

        if use_nvtx:
            torch.cuda.nvtx.range_pop()

    return step


def make_step_flashinfer_fp8(inp, num_q_heads, num_kv_heads, qk_head_dim, use_nvtx=False):
    q, k, v = split_qkv(inp, num_q_heads, num_kv_heads, qk_head_dim)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    meta = build_flashinfer_meta(inp)
    kcache_fp8 = inp.kcache.to(torch.float8_e4m3fn)
    vcache_fp8 = inp.vcache.to(torch.float8_e4m3fn)

    def step():
        if use_nvtx:
            torch.cuda.nvtx.range_push("flashinfer_rope_quantize_append")

        flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache(
            q_rope=q,
            k_rope=k,
            q_nope=None,
            k_nope=None,
            v=v,
            cos_sin_cache=inp.cos_sin,
            pos_ids=meta["positions"],
            paged_kv_cache=(kcache_fp8, vcache_fp8),
            kv_indices=meta["kv_indices"],
            kv_indptr=meta["kv_indptr"],
            batch_indices=meta["batch_indices"],
            positions=meta["positions"],
            is_neox=True,
            quantize_dtype=torch.float8_e4m3fn,
            quant_scale_q=1.0,
            quant_scale_kv=1.0,
            page_size=meta["page_size"],
            kv_layout="NHD",
            enable_pdl=False,
        )

        if use_nvtx:
            torch.cuda.nvtx.range_pop()

    return step


def resolve_backend(backend_arg):
    if backend_arg in ("hpc", "flashinfer"):
        return backend_arg

    major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
    if major == 9 and hpc is not None:
        return "hpc"
    return "flashinfer"


def run_torch_profile(step_fn, trace_path, active_iters=20):
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=2, warmup=2, active=active_iters, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_path),
        record_shapes=True,
        with_stack=False,
        profile_memory=False,
    ) as prof:
        total = 2 + 2 + active_iters
        for _ in range(total):
            step_fn()
            prof.step()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="auto", choices=["auto", "hpc", "flashinfer"])
    p.add_argument("--fp8", action="store_true", help="bench the fp8 (quant) variant")
    p.add_argument("--num-req", type=int, nargs="+", default=[16, 64, 256])
    p.add_argument("--heads", default="64,8,128", help="num_q,num_kv,qk_head_dim")
    p.add_argument("--modes", default="prefill,decode0,decode1", help="comma list: prefill,decode0,decode1")
    p.add_argument("--norm-policies", default="0,1,2", help="comma list of qk_norm_policy")
    p.add_argument("--quant-policies", default="1,2", help="comma list, used only when --fp8")
    p.add_argument("--device", default="cuda:0", help="cuda device")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--nvtx", action="store_true", help="wrap kernel calls with NVTX ranges")
    p.add_argument("--profile", action="store_true", help="run torch profiler for the first case")
    p.add_argument("--profile-trace-dir", default="./rope_profile_trace", help="torch profiler trace directory")
    p.add_argument("--profile-active-iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This benchmark requires a CUDA device.")

    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.device)
    backend = resolve_backend(args.backend)

    if backend == "hpc" and hpc is None:
        raise RuntimeError("backend=hpc requested but hpc package is not available")
    if backend == "flashinfer" and flashinfer is None:
        raise RuntimeError(
            "backend=flashinfer requested but flashinfer package is not available"
        )

    nq, nkv, hd = (int(x) for x in args.heads.split(","))

    mode_lut = {
        "prefill": (True, None),
        "decode0": (False, 0),
        "decode1": (False, 1),
    }
    modes = [mode_lut[m.strip()] for m in args.modes.split(",") if m.strip() in mode_lut]
    if not modes:
        raise ValueError("--modes must include at least one of: prefill,decode0,decode1")

    norm_policies = [int(x) for x in args.norm_policies.split(",") if x.strip()]
    quant_policies = [int(x) for x in args.quant_policies.split(",") if x.strip()] if args.fp8 else [None]

    print(f"[backend] {backend}")
    print(f"{'mode':>8} {'num_req':>8} {'norm':>5} {'quant':>6} {'us':>10}")
    profiled = False
    for is_prefill, mtp in modes:
        for num_req in args.num_req:
            inp = prepare_inputs(
                num_req=num_req,
                is_prefill=is_prefill,
                mtp=mtp,
                num_q_heads=nq,
                num_kv_heads=nkv,
                qk_head_dim=hd,
                device=args.device,
            )
            for qk_norm_policy in norm_policies:
                for quant_policy in quant_policies:
                    if args.fp8:
                        if backend == "hpc":
                            step = make_step_fp8(
                                inp,
                                is_prefill,
                                qk_norm_policy,
                                quant_policy,
                                mtp,
                                use_nvtx=args.nvtx,
                            )
                        else:
                            step = make_step_flashinfer_fp8(
                                inp,
                                nq,
                                nkv,
                                hd,
                                use_nvtx=args.nvtx,
                            )
                    else:
                        if backend == "hpc":
                            step = make_step_bf16(
                                inp,
                                is_prefill,
                                qk_norm_policy,
                                use_nvtx=args.nvtx,
                            )
                        else:
                            step = make_step_flashinfer_bf16(
                                inp,
                                nq,
                                nkv,
                                hd,
                                qk_norm_policy,
                                use_nvtx=args.nvtx,
                            )

                    if args.profile and not profiled:
                        os.makedirs(args.profile_trace_dir, exist_ok=True)
                        run_torch_profile(step, args.profile_trace_dir, args.profile_active_iters)
                        profiled = True

                    us = bench_us(step, args.warmup, args.iters)
                    tag = "prefill" if is_prefill else f"dec(mtp={mtp})"
                    print(f"{tag:>8} {num_req:>8} {qk_norm_policy:>5} "
                          f"{str(quant_policy):>6} {us:>10.2f}")

    if args.profile:
        print(f"[profile] torch profiler trace written to: {args.profile_trace_dir}")
        print("[profile] open with TensorBoard: tensorboard --logdir <trace_dir>")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        msg = str(e)
        if "no kernel image is available for execution on the device" in msg:
            print("[error] CUDA runtime reports no kernel image for this GPU architecture.")
            print("[hint] Try --backend flashinfer on non-SM90 GPUs, or rebuild for your target SM.")
            sys.exit(2)
        raise
