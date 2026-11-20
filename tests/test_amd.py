"""Correctness and integration coverage for the first portable HIP kernels."""

import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest
import torch

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))
import hpc

pytestmark = pytest.mark.skipif(not torch.version.hip, reason="AMD backend tests")


def _pre_reference(x, w, scale, base, weight, cast, head=False):
    flat = x.flatten(1).float()
    projection = torch.nn.functional.linear(flat, w)
    projection *= torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-5)
    pre = torch.sigmoid(projection[:, :4] * scale[0] + base[:4]) + 1e-6
    y = (pre.unsqueeze(-1) * x.float()).sum(1)
    if weight is not None:
        if cast:
            y = y.bfloat16().float()
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-5) * weight.float()
    post = None if head else 2 * torch.sigmoid(projection[:, 4:] * scale[1] + base[4:]) + 1e-6
    return y.bfloat16(), post


@pytest.mark.parametrize("hidden", [4096, 6144])
@pytest.mark.parametrize("head", [False, True])
@pytest.mark.parametrize("norm,cast", [(False, False), (True, False), (True, True)])
def test_ihc_signed_inputs_and_fused_norm(hidden, head, norm, cast):
    torch.manual_seed(84)
    x = torch.randn((7, 4, hidden), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((4 if head else 8, 4 * hidden), device="cuda") * 0.02
    scale = torch.tensor([0.7] if head else [0.7, 1.1], device="cuda")
    base = torch.randn((w.shape[0],), device="cuda")
    weight = torch.randn((hidden,), device="cuda").bfloat16() if norm else None
    ref_y, ref_post = _pre_reference(x, w, scale, base, weight, cast, head)
    if head:
        y = torch.ops.hpc.fuse_ihc_head(x, w, scale, base, 1e-5, 1e-6, weight, 1e-5, cast)
    else:
        y, post = hpc.fuse_ihc_pre(
            x, w, scale, base, rms_weight=weight, rms_eps=1e-5, cast_bfloat_for_norm=cast
        )
        torch.testing.assert_close(post, ref_post, atol=2e-5, rtol=1e-4)
    torch.testing.assert_close(y.float(), ref_y.float(), atol=3e-2, rtol=1e-2)


@pytest.mark.parametrize("hidden", [7, 256, 513, 4096])
@pytest.mark.parametrize("topk", [1, 4, 128])
@pytest.mark.parametrize("shared", [False, True])
def test_moe_reduce(hidden, topk, shared):
    torch.manual_seed(19)
    x = torch.randn((23, hidden), device="cuda").bfloat16()
    pos = torch.randint(0, 23, (7, topk), device="cuda", dtype=torch.int32)
    pos[:, ::3] = -1
    scale = torch.rand((7, topk), device="cuda") / topk
    extra = torch.randn((7, hidden), device="cuda").bfloat16() if shared else None
    gathered = x[pos.clamp_min(0).long()].float()
    gathered = torch.where((pos >= 0).unsqueeze(-1), gathered, 0.0)
    ref = (gathered * scale.unsqueeze(-1)).sum(1)
    if shared:
        ref += extra.float()
    actual = hpc.reduce(x, pos, scale, extra)
    torch.testing.assert_close(actual.float(), ref.bfloat16().float(), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("policy", [0, 1, 2])
@pytest.mark.parametrize("separate_kv", [False, True])
def test_rope_output_buffers_and_padding(policy, separate_kv):
    torch.manual_seed(10)
    device = "cuda"
    qkv = torch.randn((8, 10 * 128), device=device).bfloat16()
    lengths = torch.tensor([6, 0, 9, 0], dtype=torch.int32, device=device)
    q_index = torch.tensor([0, 2, 2, 5, 8], dtype=torch.int32, device=device)
    blocks = torch.tensor(
        [[0, 1, 2], [0, 0, 0], [3, 4, 5], [0, 0, 0]], dtype=torch.int32, device=device
    )
    k = torch.randn((8, 4, 1, 128), device=device).bfloat16()
    v = torch.randn_like(k)
    ref_k, ref_v = k.clone(), v.clone()
    cs = torch.randn((16, 128), device=device)
    qw, kw = torch.randn((2, 128), device=device)
    q_out = torch.full((8, 8, 128), 17.0, device=device).bfloat16()
    k_out = torch.full((8, 1, 128), 17.0, device=device).bfloat16() if separate_kv else None
    v_out = torch.full((8, 1, 128), 17.0, device=device).bfloat16() if separate_kv else None
    q_ref = qkv[:5, : 8 * 128].reshape(5, 8, 128).float()
    k_ref = qkv[:5, 8 * 128 : 9 * 128].reshape(5, 1, 128).float()
    v_ref = qkv[:5, 9 * 128 :].reshape(5, 1, 128)
    positions = torch.tensor([4, 5, 6, 7, 8], device=device)
    for tensor, weight in [(q_ref, qw), (k_ref, kw)]:
        if policy == 2:
            tensor *= torch.rsqrt(tensor.square().mean(-1, keepdim=True) + 1e-6) * weight
        first, second = tensor[..., :64].clone(), tensor[..., 64:].clone()
        cosine, sine = cs[positions, :64].unsqueeze(1), cs[positions, 64:].unsqueeze(1)
        tensor[..., :64] = first * cosine - second * sine
        tensor[..., 64:] = second * cosine + first * sine
        if policy == 1:
            tensor *= torch.rsqrt(tensor.square().mean(-1, keepdim=True) + 1e-6) * weight
    actual = hpc.rope_norm_store_kv(
        k,
        v,
        qkv,
        cs,
        lengths,
        q_index,
        blocks,
        False,
        qw if policy else None,
        kw if policy else None,
        q_out,
        k_out,
        v_out,
        policy,
    )
    assert actual.data_ptr() == q_out.data_ptr()
    torch.testing.assert_close(q_out[:5].float(), q_ref.bfloat16().float(), atol=3e-2, rtol=1e-2)
    assert torch.all(q_out[5:] == 17)
    for row, (physical, offset) in enumerate([(1, 0), (1, 1), (4, 2), (4, 3), (5, 0)]):
        if not separate_kv:
            ref_k[physical, offset] = k_ref[row].bfloat16()
            ref_v[physical, offset] = v_ref[row]
    ref_k[1, 2:] = ref_v[1, 2:] = 0
    ref_k[5, 1:] = ref_v[5, 1:] = 0
    torch.testing.assert_close(k.float(), ref_k.float(), atol=3e-2, rtol=1e-2)
    torch.testing.assert_close(v, ref_v, atol=0, rtol=0)
    if separate_kv:
        torch.testing.assert_close(
            k_out[:5].float(), k_ref.bfloat16().float(), atol=3e-2, rtol=1e-2
        )
        torch.testing.assert_close(v_out[:5], v_ref, atol=0, rtol=0)
        assert torch.all(k_out[5:] == 17) and torch.all(v_out[5:] == 17)


@pytest.mark.parametrize("policy", [0, 1, 2])
def test_rope_graph_replay_updates_metadata(policy):
    torch.manual_seed(30)
    qkv = torch.randn((1, 1280), device="cuda").bfloat16()
    k = torch.zeros((4, 4, 1, 128), dtype=torch.bfloat16, device="cuda")
    v = torch.zeros_like(k)
    cs = torch.randn((16, 128), device="cuda")
    lengths = torch.tensor([5], dtype=torch.int32, device="cuda")
    q_index = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    blocks = torch.arange(4, dtype=torch.int32, device="cuda").reshape(1, 4)
    qw, kw = torch.randn((2, 128), device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            hpc.rope_norm_store_kv(
                k, v, qkv, cs, lengths, q_index, blocks, False, qw, kw, qk_norm_policy=policy
            )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            q = hpc.rope_norm_store_kv(
                k, v, qkv, cs, lengths, q_index, blocks, False, qw, kw, qk_norm_policy=policy
            )
        for length in [6, 9]:
            lengths.fill_(length)
            graph.replay()
            for tensor, weight, actual in [
                (qkv[:, :1024].reshape(1, 8, 128).float(), qw, q),
                (
                    qkv[:, 1024:1152].reshape(1, 1, 128).float(),
                    kw,
                    k[(length - 1) // 4, (length - 1) % 4].unsqueeze(0),
                ),
            ]:
                if policy == 2:
                    tensor = (
                        tensor * torch.rsqrt(tensor.square().mean(-1, keepdim=True) + 1e-6) * weight
                    )
                first, second = tensor[..., :64], tensor[..., 64:]
                cosine, sine = cs[length - 1, :64], cs[length - 1, 64:]
                ref = torch.cat(
                    [first * cosine - second * sine, second * cosine + first * sine], dim=-1
                )
                if policy == 1:
                    ref = ref * torch.rsqrt(ref.square().mean(-1, keepdim=True) + 1e-6) * weight
                torch.testing.assert_close(
                    actual.float(), ref.bfloat16().float(), atol=3e-2, rtol=1e-2
                )
            torch.testing.assert_close(
                v[(length - 1) // 4, (length - 1) % 4].reshape(-1), qkv[0, 1152:], atol=0, rtol=0
            )


def test_stream_graph_replay_and_device_guard():
    device = min(1, torch.cuda.device_count() - 1)
    torch.cuda.set_device(0)
    with torch.cuda.device(device):
        stream = torch.cuda.Stream()
        x = torch.randn((7, 4096), device="cuda").bfloat16()
        residual = torch.randn((7, 4, 4096), device="cuda").bfloat16()
        gates = torch.rand((7, 4), device="cuda")
        pos = torch.arange(7, device="cuda", dtype=torch.int32).view(7, 1)
        scales = torch.ones((7, 1), device="cuda")
        w = torch.randn((8, 4 * 4096), device="cuda") * 0.01
        hc_scale = torch.ones(2, device="cuda")
        hc_base = torch.zeros(8, device="cuda")
        stream.wait_stream(torch.cuda.current_stream())
    # A call with GPU 0 current must select the input's GPU and restore GPU 0.
    out = hpc.fuse_ihc_post(x, residual, gates)
    assert out.device.index == device and torch.cuda.current_device() == 0
    # A non-default stream and graph must use that device's current stream.
    with torch.cuda.stream(stream):
        for _ in range(3):
            post = hpc.fuse_ihc_post(x, residual, gates)
            y, hp = hpc.fuse_ihc_pre(post, w, hc_scale, hc_base)
            reduced = hpc.reduce(y, pos, scales)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            post = hpc.fuse_ihc_post(x, residual, gates)
            y, hp = hpc.fuse_ihc_pre(post, w, hc_scale, hc_base)
            reduced = hpc.reduce(y, pos, scales)
        for value in (0.25, -0.75):
            x.fill_(value)
            graph.replay()
            ref_post = (gates[:, :, None] * x.float()[:, None] + residual.float()).bfloat16()
            ref_y, ref_hp = _pre_reference(ref_post, w, hc_scale, hc_base, None, False)
            torch.testing.assert_close(reduced.float(), ref_y.float(), atol=3e-2, rtol=1e-2)
            torch.testing.assert_close(hp, ref_hp, atol=2e-5, rtol=1e-4)
    assert torch.cuda.current_device() == 0


def test_empty_and_invalid_inputs():
    x = torch.empty((0, 4, 4096), device="cuda", dtype=torch.bfloat16)
    w = torch.zeros((8, 4 * 4096), device="cuda")
    scale, base = torch.ones(2, device="cuda"), torch.zeros(8, device="cuda")
    y, post = hpc.fuse_ihc_pre(x, w, scale, base)
    assert y.shape == (0, 4096) and post.shape == (0, 4)
    y = hpc.fuse_ihc_post(y, x, post)
    assert y.shape == x.shape
    reduced = hpc.reduce(
        torch.empty((0, 17), device="cuda", dtype=torch.bfloat16),
        torch.empty((0, 2), device="cuda", dtype=torch.int32),
        torch.empty((0, 2), device="cuda"),
    )
    assert reduced.shape == (0, 17)
    with pytest.raises(RuntimeError, match="dtype"):
        hpc.fuse_ihc_pre(x.float(), w, scale, base)
    with pytest.raises(RuntimeError, match="hc_scale"):
        hpc.fuse_ihc_pre(x, w, scale.cpu(), base)


def test_fake_tensors():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        x = torch.empty((7, 4, 4096), device="cuda", dtype=torch.bfloat16)
        w = torch.empty((8, 4 * 4096), device="cuda")
        scale, base = torch.empty(2, device="cuda"), torch.empty(8, device="cuda")
        y, hp = hpc.fuse_ihc_pre(x, w, scale, base)
        assert y.shape == (7, 4096) and hp.dtype == torch.float32
        assert hpc.fuse_ihc_post(y, x, hp).shape == x.shape
        head = hpc.fuse_ihc_head(x, w[:4], scale[:1], base[:4])
        assert head.shape == (7, 4096)


def test_todos_and_schema_parity():
    root = Path(__file__).resolve().parents[1]
    # Compare parsed operator schemas, including mutations and argument defaults.
    for path in root.glob("src/*/entry.cc"):
        for match in re.finditer(r'm\.def\(\s*((?:"(?:\\.|[^"\\])*"\s*)+)', path.read_text()):
            schema = "".join(
                ast.literal_eval(s) for s in re.findall(r'"(?:\\.|[^"\\])*"', match[1])
            )
            if "(" in schema:
                expected = torch._C.parse_schema("hpc::" + schema)
                actual = torch._C._dispatch_find_schema_or_throw(expected.name, "").schema()
                assert actual == expected, (path, actual, expected)
    x = torch.ones((1, 4096), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match="TODO.*FP8"):
        hpc.scaled_fp8_quant(x, torch.ones(1, device="cuda"))
    with pytest.raises(NotImplementedError, match="TODO.*matrix"):
        torch.ops.hpc.gated_mla_gemm(x, x, x)
    with pytest.raises(NotImplementedError, match="TODO"):
        torch.ops.hpc.fuse_ihc_post_pre(x, x, x, x, x, x, 1e-5, 1e-6, 2.0)
    assert json.loads(hpc.__built_json__)["backend"] == "hip"


def test_import_does_not_initialize_gpu():
    code = "import torch, hpc; assert not torch.cuda.is_initialized(); print(hpc.__backend__)"
    env = dict(os.environ, PYTHONPATH=str(Path(hpc.__file__).parent.parent))
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd="/tmp",
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "hip"
