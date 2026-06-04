import os
import sys
from pathlib import Path

sys.path.insert(
    0,
    os.path.realpath(
        sorted([p for p in Path(__file__).parent.glob("../build/lib.*/") if "linux" in str(p)])[0]
    ),
)

import torch
import torch.cuda.nvtx as nvtx

import hpc

try:
    from flashinfer import mxfp8_quantize, shuffle_matrix_a, shuffle_matrix_sf_a, ActivationType
    from flashinfer.fused_moe import trtllm_fp8_block_scale_moe, WeightLayout
    from flashinfer.fused_moe.core import Fp8QuantizationType
    from flashinfer.utils import device_support_pdl

    HAS_FLASHINFER = True
except ImportError:
    HAS_FLASHINFER = False

SF_VEC = 32
NUM_GROUP = 192  # num_expert_total


# (label, num_expert_local, intermediate_half)
CONFIGS = [
    ("tp8", 192, 1536 // 8, 192),  # 192 local experts, inter_half=192, 192 total
    ("tp4", 192, 1536 // 4, 192),  # 192 local experts, inter_half=384, 192 total
    ("ep8", 192 // 8, 1536, 192),  # 24 local experts, inter_half=1536, 192 total
]
NUM_SEQ_CASES = [128, 256, 384, 512, 1024, 2048, 4096]
HIDDEN = 4096
NUM_TOPK = 8

WARMUP = 5
ITER = 100


def _build_mxfp8(num_seq, num_expert_local, inter_half, num_expert_total, device):
    hidden = HIDDEN
    num_topk = NUM_TOPK

    x_fp8 = torch.zeros((num_seq, hidden), dtype=torch.float8_e4m3fn, device=device)
    x_scale = torch.full((num_seq, hidden // SF_VEC), 127, dtype=torch.uint8, device=device)
    gate_up_w = torch.zeros(
        (num_expert_local, inter_half * 2, hidden), dtype=torch.float8_e4m3fn, device=device
    )
    gate_up_w_scale = torch.full(
        (num_expert_local, inter_half * 2, hidden // SF_VEC), 127, dtype=torch.uint8, device=device
    )
    down_w = torch.zeros(
        (num_expert_local, hidden, inter_half), dtype=torch.float8_e4m3fn, device=device
    )
    down_w_scale = torch.full(
        (num_expert_local, hidden, inter_half // SF_VEC), 127, dtype=torch.uint8, device=device
    )

    topk_ids = torch.randint(
        0, num_expert_total, (num_seq, num_topk), device=device, dtype=torch.int32
    )
    topk_scale = torch.ones((num_seq, num_topk), device=device, dtype=torch.float32)

    # Prepack SFW — use correct avg so layout (1SM vs 2SM) matches runtime dispatch.
    avg = (num_seq * num_topk) // num_expert_total

    def _prepack_sfw(sfw):
        _, sfw_packed = hpc.prepack_mxfp8_scale(None, sfw, None, num_seq_per_group_avg=avg)
        return sfw_packed

    gate_up_w_scale_packed = _prepack_sfw(gate_up_w_scale)
    down_w_scale_packed = _prepack_sfw(down_w_scale)

    def run():
        return hpc.fuse_moe_mxfp8(
            x_fp8,
            x_scale,
            gate_up_w,
            gate_up_w_scale_packed,
            down_w,
            down_w_scale_packed,
            topk_ids,
            topk_scale,
            rank_ep=0,
            num_expert_total=num_expert_total,
        )

    return run


def _build_fp8(num_seq, num_expert_local, inter_half, num_expert_total, device):
    hidden = HIDDEN
    num_topk = NUM_TOPK

    x_fp8 = torch.zeros((num_seq, hidden), dtype=torch.float8_e4m3fn, device=device)
    gate_up_w = torch.zeros(
        (num_expert_local, inter_half * 2, hidden), dtype=torch.float8_e4m3fn, device=device
    )
    down_w = torch.zeros(
        (num_expert_local, hidden, inter_half), dtype=torch.float8_e4m3fn, device=device
    )
    gate_up_scale = torch.ones(num_expert_local, dtype=torch.float32, device=device)
    down_scale = torch.ones(num_expert_local, dtype=torch.float32, device=device)
    act_and_mul_scale = torch.ones(num_expert_local, dtype=torch.float32, device=device)

    topk_ids = torch.randint(
        0, num_expert_total, (num_seq, num_topk), device=device, dtype=torch.int32
    )
    topk_scale = torch.ones((num_seq, num_topk), device=device, dtype=torch.float32)

    def run():
        return torch.ops.hpc.fuse_moe(
            x_fp8,
            gate_up_w,
            down_w,
            gate_up_scale,
            down_scale,
            act_and_mul_scale,
            topk_ids,
            topk_scale,
            None,  # shared_output
            0,  # rank_ep
            num_expert_total,
            False,  # use_bf16_mul
            None,  # output
        )

    return run


def _build_flashinfer_mxfp8(num_seq, num_expert_local, inter_half, num_expert_total, device):
    """Build flashinfer mxfp8 moe benchmark (includes routing from logits)."""
    if not HAS_FLASHINFER:
        return None
    hidden = HIDDEN
    num_topk = NUM_TOPK
    # flashinfer requires intermediate_size to be multiple of 128; pad up to 256 alignment
    intermediate_size = ((inter_half + 255) // 256) * 256
    epilogue_tile_m = 128

    # Routing logits (flashinfer does its own topk routing)
    routing_logits = torch.randn((num_seq, num_expert_total), device=device, dtype=torch.bfloat16)

    # Hidden states in mxfp8
    hidden_states_bf16 = torch.randn((num_seq, hidden), device=device, dtype=torch.bfloat16) * 0.1
    hidden_states, hidden_states_scale = mxfp8_quantize(hidden_states_bf16, False)
    hidden_states_scale = hidden_states_scale.reshape(num_seq, -1)

    # Weights in mxfp8 (use padded intermediate_size)
    w13_bf16 = (
        torch.randn(
            (num_expert_local, intermediate_size * 2, hidden), device=device, dtype=torch.bfloat16
        )
        * 0.1
    )
    w2_bf16 = (
        torch.randn(
            (num_expert_local, hidden, intermediate_size), device=device, dtype=torch.bfloat16
        )
        * 0.1
    )

    # Quantize weights
    w13_flat = w13_bf16.reshape(-1, hidden)
    w13_q, w13_s = mxfp8_quantize(w13_flat, False)
    w13_q = w13_q.reshape(num_expert_local, intermediate_size * 2, hidden)
    w13_s = w13_s.reshape(num_expert_local, intermediate_size * 2, -1)

    w2_flat = w2_bf16.reshape(-1, intermediate_size)
    w2_q, w2_s = mxfp8_quantize(w2_flat, False)
    w2_q = w2_q.reshape(num_expert_local, hidden, intermediate_size)
    w2_s = w2_s.reshape(num_expert_local, hidden, -1)

    # Shuffle weights and scales for flashinfer's layout
    w13_shuffled = []
    w13_scales_shuffled = []
    w2_shuffled = []
    w2_scales_shuffled = []
    for i in range(num_expert_local):
        w13_shuffled.append(
            shuffle_matrix_a(w13_q[i].view(torch.uint8), epilogue_tile_m)
            .contiguous()
            .view(w13_q.dtype)
        )
        w13_scales_shuffled.append(
            shuffle_matrix_sf_a(
                w13_s[i].reshape(intermediate_size * 2, -1), epilogue_tile_m
            ).contiguous()
        )
        w2_shuffled.append(
            shuffle_matrix_a(w2_q[i].view(torch.uint8), epilogue_tile_m)
            .contiguous()
            .view(w2_q.dtype)
        )
        w2_scales_shuffled.append(
            shuffle_matrix_sf_a(w2_s[i].reshape(hidden, -1), epilogue_tile_m).contiguous()
        )

    gemm1_weights = torch.stack(w13_shuffled)
    gemm1_scales = torch.stack(w13_scales_shuffled)
    gemm2_weights = torch.stack(w2_shuffled)
    gemm2_scales = torch.stack(w2_scales_shuffled)

    enable_pdl = device_support_pdl(device)

    def run():
        return trtllm_fp8_block_scale_moe(
            routing_logits=routing_logits,
            routing_bias=None,
            hidden_states=hidden_states,
            hidden_states_scale=hidden_states_scale,
            gemm1_weights=gemm1_weights,
            gemm1_weights_scale=gemm1_scales,
            gemm2_weights=gemm2_weights,
            gemm2_weights_scale=gemm2_scales,
            num_experts=num_expert_total,
            top_k=num_topk,
            n_group=None,
            topk_group=None,
            intermediate_size=intermediate_size,
            local_expert_offset=0,
            local_num_experts=num_expert_local,
            routed_scaling_factor=None,
            use_shuffled_weight=True,
            weight_layout=WeightLayout.MajorK.value,
            enable_pdl=enable_pdl,
            fp8_quantization_type=Fp8QuantizationType.MxFp8,
            activation_type=ActivationType.Swiglu.value,
        )

    return run


def _bench(label, fn, use_cuda_graph=False):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()

    if use_cuda_graph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        # warmup graph
        for _ in range(WARMUP):
            g.replay()
        torch.cuda.synchronize()

        nvtx.range_push(label)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(ITER):
            g.replay()
        end.record()
        torch.cuda.synchronize()
        nvtx.range_pop()
    else:
        nvtx.range_push(label)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(ITER):
            fn()
        end.record()
        torch.cuda.synchronize()
        nvtx.range_pop()
    return start.elapsed_time(end) * 1e3 / ITER


def main():
    device = torch.device("cuda:0")
    torch.manual_seed(2026)

    for cfg_label, num_expert_local, inter_half, num_expert_total in CONFIGS:
        print(
            f"\n=== {cfg_label}: experts={num_expert_local}/{num_expert_total}  inter_half={inter_half}"
            f"  hidden={HIDDEN}  topk={NUM_TOPK} ==="
        )
        print(
            f"{'num_seq':<10} {'avg':<5} {'mxfp8(us)':<11} {'fp8(us)':<11} {'flashinfer(us)':<15} {'vs_fi':<8}"
        )
        print("-" * 65)
        for num_seq in NUM_SEQ_CASES:
            avg = (num_seq * NUM_TOPK) // num_expert_total

            run_mx = _build_mxfp8(num_seq, num_expert_local, inter_half, num_expert_total, device)
            t_mx = _bench(f"mxfp8/{cfg_label}/seq{num_seq}", run_mx, use_cuda_graph=True)

            # fp8 per-tensor
            t_fp_str = "N/A"
            try:
                run_fp = _build_fp8(num_seq, num_expert_local, inter_half, num_expert_total, device)
                t_fp = _bench(f"fp8/{cfg_label}/seq{num_seq}", run_fp, use_cuda_graph=True)
                t_fp_str = f"{t_fp:.2f}"
            except Exception:
                pass

            # flashinfer mxfp8
            t_fi_str = "N/A"
            vs_fi_str = "N/A"
            try:
                run_fi = _build_flashinfer_mxfp8(
                    num_seq, num_expert_local, inter_half, num_expert_total, device
                )
                if run_fi is not None:
                    t_fi = _bench(
                        f"flashinfer/{cfg_label}/seq{num_seq}", run_fi, use_cuda_graph=True
                    )
                    t_fi_str = f"{t_fi:.2f}"
                    vs_fi_str = f"{t_fi / t_mx:.2f}"
            except Exception as e:
                t_fi_str = f"ERR"
                vs_fi_str = str(e)[:20]

            print(
                f"{num_seq:<10} {avg:<5} {t_mx:<11.2f} {t_fp_str:<11} {t_fi_str:<15} {vs_fi_str:<8}"
            )


if __name__ == "__main__":
    main()
