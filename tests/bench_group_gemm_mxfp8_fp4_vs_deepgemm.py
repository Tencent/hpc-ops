"""Bench fp8(act) x mxfp4(weight) group GEMM: hpc.group_gemm_mxfp8 vs DeepGEMM.

Both kernels run the SAME mixed precision: fp8_e4m3 activations x e2m1 (fp4)
weights with block scale factors (act per-128, weight per-32).

  hpc      : hpc.group_gemm_mxfp8(x_fp8, w_fp4_packed, sfx_packed, sfw_packed,
                                  seqlens, cu_seqlens, ...)
  deepgemm : deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(a=(x_fp8, sfx),
                                  b=(w_fp4, sfw), d, m_indices,
                                  recipe_a=(1,128), recipe_b=(1,32))
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))


import torch
import torch.cuda.nvtx as nvtx

import hpc
import deep_gemm
from deep_gemm.utils import (
    align,
    ceil_div,
    per_token_cast_to_fp8,
    per_token_cast_to_fp4,
)

# ---------------------------- config ----------------------------

# Total token counts (m = num_seq) to sweep.
M_CASES = [256, 512, 1024, 2048, 4096, 8192]
NUM_EXPERT_TOTAL = 256
NUM_TOPK = 8

# hy4.0  (label, n, k, num_group)
#   num_group = num_expert_local for that parallelism config.
#   tp8/tp4: all 256 experts local.
#   ep8: 256/8 = 32 experts per card.
SHAPES = [
    ("gate_up_tp8", 512, 6144, 256),
    ("down_tp8", 6144, 256, 256),
    ("gate_up_tp4", 1024, 6144, 256),
    ("down_tp4", 6144, 512, 256),
    ("gate_up_ep8", 4096, 6144, 32),
    ("down_ep8", 6144, 2048, 32),
]

SF_VEC = 32  # mxfp8 / mxfp4 block size
GRAN_K_A = 128  # DeepGEMM fp8 activation block (recipe_a = (1, 128))
GRAN_K_B = 32  # DeepGEMM fp4 weight block (recipe_b = (1, 32))
WARMUP = 5
ITER = 100


# ---------------------------- hpc builder ----------------------------


def _build_hpc(num_group, seq, n, k, device):
    """fp8 activation x fp4 weight via hpc.group_gemm_mxfp8. Requires k % 32 == 0."""
    if k % SF_VEC != 0:
        return None, None

    seqlens = torch.full((num_group,), seq, dtype=torch.int32, device=device)
    cu_seqlens = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.cumsum(seqlens, 0).to(torch.int32),
        ]
    )
    m_total = int(cu_seqlens[-1].item())

    x_fp8 = torch.randn((m_total, k), dtype=torch.float32, device=device).to(torch.float8_e4m3fn)

    # W: fp4 (e2m1) packed two-per-byte -> [num_group, n, k/2] uint8.
    lo = torch.randint(0, 16, (num_group, n, k // 2), dtype=torch.uint8, device=device)
    hi = torch.randint(0, 16, (num_group, n, k // 2), dtype=torch.uint8, device=device)
    w_fp4_packed = ((hi << 4) | lo).contiguous()

    sfx = torch.full((m_total, k // SF_VEC), 127, dtype=torch.uint8, device=device)
    sfw = torch.full((num_group, n, k // SF_VEC), 127, dtype=torch.uint8, device=device)
    sfx_packed, sfw_packed = hpc.prepack_mxfp8_scale(
        sfx, sfw, cu_seqlens, num_seq_per_group_avg=seq
    )

    def run():
        return hpc.group_gemm_mxfp8(
            x_fp8,
            w_fp4_packed,
            sfx_packed,
            sfw_packed,
            seqlens,
            cu_seqlens,
            num_seq_per_group_avg=seq,
        )

    return run, m_total


# ---------------------------- deepgemm builder ----------------------------


def _build_deepgemm(num_group, seq, n, k, device):
    """fp8 activation x fp4 weight via DeepGEMM contiguous grouped GEMM.

    Mirrors generators.generate_m_grouped_contiguous (fp8_fp4 path):
      a = per_token_cast_to_fp8(act, gran_k=128)          -> (fp8 [m,k], sf)
      b = per_token_cast_to_fp4(weight, gran_k=32)        -> (fp4 [.,.,k/2], sf)
    The contiguous layout pads each group's m to `mk_alignment`.

    Note: the kernel internally casts float32 scales to packed UE8M0 (cannot
    be pre-packed externally for this kernel variant).

    Returns (run_fn, m_padded) or (None, None) if unsupported.
    """
    # fp4 weight requires k even; both blocks need k divisible by their gran.
    if k % GRAN_K_A != 0 or k % GRAN_K_B != 0:
        return None, None

    # Pick the alignment DeepGEMM would use for this expected token count.
    alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(seq)
    deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)

    aligned_seq = align(seq, alignment)
    m = aligned_seq * num_group

    # m_indices: row -> group id (uniform aligned_seq rows per group).
    m_indices = torch.arange(num_group, device=device, dtype=torch.int32).repeat_interleave(
        aligned_seq
    )

    # Quantize activations to fp8 (per-token, gran_k=128). use_ue8m0 -> packed UE8M0.
    a_bf16 = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    a_fp8, a_sf = per_token_cast_to_fp8(a_bf16, use_ue8m0=True, gran_k=GRAN_K_A)
    a = (a_fp8, a_sf)

    # Quantize weights to fp4 (per-token, gran_k=32), per expert group.
    b_fp4 = torch.empty((num_group, n, k // 2), device=device, dtype=torch.int8)
    b_sf = torch.empty((num_group, n, ceil_div(k, GRAN_K_B)), device=device, dtype=torch.float)
    for i in range(num_group):
        w_i = torch.randn((n, k), device=device, dtype=torch.bfloat16)
        w_fp4_i, w_sf_i = per_token_cast_to_fp4(w_i, use_ue8m0=True, gran_k=GRAN_K_B)
        b_fp4[i], b_sf[i] = w_fp4_i, w_sf_i
    b = (b_fp4, b_sf)

    d = torch.empty((m, n), device=device, dtype=torch.bfloat16)

    recipe_a = (1, GRAN_K_A)
    recipe_b = (1, GRAN_K_B)

    def run():
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            a,
            b,
            d,
            m_indices,
            recipe_a=recipe_a,
            recipe_b=recipe_b,
        )
        return d

    return run, m


# ---------------------------- timing ----------------------------


def _bench(label: str, fn, use_cuda_graph: bool = True) -> float:
    """Returns mean per-call latency in microseconds (CUDA graph when possible)."""
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()

    # Try CUDA graph capture; fall back to eager if capture fails or disabled.
    graph = None
    if use_cuda_graph:
        try:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            torch.cuda.synchronize()
        except Exception:
            graph = None
            torch.cuda.synchronize()

    nvtx.range_push(label)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITER):
        if graph is not None:
            graph.replay()
        else:
            fn()
    end.record()
    torch.cuda.synchronize()
    nvtx.range_pop()
    elapsed_ms = start.elapsed_time(end)

    # Release graph to avoid interfering with subsequent captures from other libraries.
    if graph is not None:
        del graph
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return elapsed_ms * 1e3 / ITER  # us


def _try_bench(label, build_fn, *args, use_cuda_graph=True, retries=1):
    """Build + bench, tolerating per-shape failures (returns (us|None, m|None))."""
    for attempt in range(retries):
        try:
            run, m = build_fn(*args)
            if run is None:
                return None, None
            return _bench(label, run, use_cuda_graph=use_cuda_graph), m
        except Exception as e:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if attempt == retries - 1:
                print(f"  [skip {label}: {type(e).__name__}: {e}]", flush=True)
                return None, None


# ---------------------------- main ----------------------------


def main():
    device = torch.device("cuda:0")
    torch.manual_seed(2026)

    print(f"WARMUP={WARMUP}  ITER={ITER}", flush=True)

    # results[shape_label][num_seq] = (hpc_us, dg_us)
    results = {label: {} for label, _, _, _ in SHAPES}

    for shape_label, n, k, num_group in SHAPES:
        print(f"\n=== shape={shape_label}  n={n}  k={k}  num_group={num_group} ===", flush=True)
        for num_seq in M_CASES:
            # seq_per_group = tokens each expert (group) processes after routing
            seq = num_seq * NUM_TOPK // NUM_EXPERT_TOTAL

            t_hpc, _ = _try_bench(
                f"hpc/{shape_label}/seq{seq}", _build_hpc, num_group, seq, n, k, device
            )
            t_dg, _ = _try_bench(
                f"dg/{shape_label}/seq{seq}",
                _build_deepgemm,
                num_group,
                seq,
                n,
                k,
                device,
                use_cuda_graph=False,
                retries=3,
            )

            results[shape_label][num_seq] = (t_hpc, t_dg)
            hpc_str = f"{t_hpc:7.2f}" if t_hpc is not None else "    N/A"
            dg_str = f"{t_dg:7.2f}" if t_dg is not None else "    N/A"
            print(
                f"  m={num_seq:>5d}  seq/grp={seq:>4d}  hpc={hpc_str} us   deepgemm={dg_str} us",
                flush=True,
            )

    # Markdown summary tables (one table per shape).
    for shape_label, n, k, num_group in SHAPES:
        print()
        print(f"### {shape_label} (n={n}, k={k}, num_group={num_group})")
        header = ["m", "seq/grp", "hpc(us)", "deepgemm(us)", "dg/hpc"]
        sep = ["------"] * len(header)
        print("| " + " | ".join(c.ljust(12) for c in header) + " |")
        print("|-" + "-|-".join(c.ljust(12, "-") for c in sep) + "-|")
        for num_seq in M_CASES:
            seq = num_seq * NUM_TOPK // NUM_EXPERT_TOTAL
            t_hpc, t_dg = results[shape_label][num_seq]
            hpc_cell = f"{t_hpc:.2f}" if t_hpc is not None else "N/A"
            dg_cell = f"{t_dg:.2f}" if t_dg is not None else "N/A"
            if t_hpc is not None and t_dg is not None:
                ratio = f"{t_dg / t_hpc:.2f}x"
            else:
                ratio = "N/A"
            cells = [str(num_seq), str(seq), hpc_cell, dg_cell, ratio]
            print("| " + " | ".join(c.ljust(12) for c in cells) + " |")


if __name__ == "__main__":
    main()
