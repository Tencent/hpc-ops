"""AMD fusion, selection, sampling, and layout contracts."""

import math

import pytest
import torch

import hpc

pytestmark = pytest.mark.skipif(not torch.version.hip, reason="AMD backend tests")


@pytest.mark.parametrize("hidden", [4096, 6144])
@pytest.mark.parametrize("cast", [False, True])
@pytest.mark.parametrize("norm", [False, True])
@pytest.mark.parametrize("use_tle", [False, True])
def test_post_pre_signed_rounding(hidden, cast, norm, use_tle, monkeypatch):
    from hpc._triton import ihc

    monkeypatch.setattr(ihc, "USE_TLE", use_tle)
    torch.manual_seed(38)
    xa = torch.randn((7, hidden), device="cuda").bfloat16()
    residual = torch.randn((7, 4, hidden), device="cuda").bfloat16()
    gates = torch.randn((7, 4), device="cuda")
    w = torch.randn((8, 4 * hidden), device="cuda") * 0.02
    scale, base = torch.randn(2, device="cuda"), torch.randn(8, device="cuda")
    weight = torch.randn(hidden, device="cuda").bfloat16() if norm else None
    middle = (xa.float()[:, None] * gates[:, :, None] + residual.float()).bfloat16()
    flat = middle.flatten(1).float()
    projection = torch.nn.functional.linear(flat, w) * torch.rsqrt(
        flat.square().mean(-1, keepdim=True) + 1e-5
    )
    pre = torch.sigmoid(projection[:, :4] * scale[0] + base[:4]) + 1e-6
    expected = (pre[:, :, None] * middle.float()).sum(1)
    if norm:
        if cast:
            expected = expected.bfloat16().float()
        expected = (
            expected * torch.rsqrt(expected.square().mean(-1, keepdim=True) + 1e-5) * weight.float()
        )
    expected_gates = 2 * torch.sigmoid(projection[:, 4:] * scale[1] + base[4:]) + 1e-6
    y, z, post = hpc.fuse_ihc_post_pre(
        xa,
        residual,
        gates,
        w,
        scale,
        base,
        rms_weight=weight,
        rms_eps=1e-5,
        cast_bfloat_for_norm=cast,
    )
    torch.testing.assert_close(y, middle, atol=0, rtol=0)
    torch.testing.assert_close(z.float(), expected.bfloat16().float(), atol=3e-2, rtol=1e-2)
    torch.testing.assert_close(post, expected_gates, atol=2e-5, rtol=1e-4)


@pytest.mark.parametrize("cols", [16, 32, 56])
@pytest.mark.parametrize("average,tile", [(1, 8), (12, 16), (24, 32), (40, 48), (64, 64)])
def test_scale_reformat_ragged_and_padding(cols, average, tile):
    lengths = [0, 3, tile + 1, tile - 1]
    capacity = tile * 3
    x = torch.randn((4 * capacity, cols), device="cuda")
    lens = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    cu = torch.arange(5, device="cuda", dtype=torch.int32) * capacity
    output = torch.full((cols, x.shape[0]), 17.0, device="cuda")
    expected = output.clone()
    offset = 0
    for group, length in enumerate(lengths):
        expected[:, offset : offset + length] = x[group * capacity : group * capacity + length].T
        offset += math.ceil(length / tile) * tile
    actual = hpc.reformat_x_scale(x, lens, cu, average, output)
    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def _stem_reference(scores, qlens, klens, prompts, block, alpha, initial, window, rates):
    values = scores.cpu()
    result = torch.zeros_like(values, dtype=torch.uint8)
    _, heads, rows, width = values.shape
    for batch, (qlen, klen, prompt) in enumerate(zip(qlens, klens, prompts)):
        qb, kb, pb = math.ceil(qlen / block), math.ceil(klen / block), math.ceil(prompt / block)
        offset = math.ceil((klen - qlen) / block)
        k = (
            pb
            if pb < 56
            else int(pb * rates[0]) + rates[1] if pb < 160 else int(pb * rates[2]) + rates[3]
        )
        for head in range(heads):
            for row in range(min(qb, rows)):
                if kb <= 0:
                    continue
                position = row + offset
                budget = k
                if position >= k and pb - k > 1:
                    budget = min(
                        k, max(1, math.floor(k + (position - k) / (pb - k - 1) * (k * alpha - k)))
                    )
                bits = values[batch, head, row].view(torch.int16).int() & 0xFFFF
                finite = (bits & 0x7F80) != 0x7F80
                valid = torch.arange(width) < kb
                ordered = torch.where(bits & 0x8000 != 0, ~bits & 0xFFFF, bits ^ 0x8000)
                finite &= valid
                count = int(finite.sum())
                threshold = (
                    0x80 if budget >= count else int(ordered[finite].topk(budget).values[-1])
                )
                selected = finite & (ordered >= threshold)
                diagonal = min(position, kb - 1)
                indices = torch.arange(width)
                selected |= (
                    (indices < initial)
                    | ((indices <= diagonal) & (indices > diagonal - window))
                    | (indices == diagonal)
                )
                result[batch, head, row] = (selected & valid).byte()
    return result.to(scores.device)


@pytest.mark.parametrize("width", [7, 57, 161, 1024, 32768])
@pytest.mark.parametrize("alpha", [0.2, 1.0])
def test_stem_threshold_ties_and_chunked_prefill(width, alpha):
    torch.manual_seed(53)
    scores = torch.randn((2, 2, 3, width), device="cuda").bfloat16()
    scores[..., ::5] = 1
    scores[..., 1::13] = float("-inf")
    scores[..., 2::17] = float("nan")
    block = 128
    qlens = [257, 0]
    klens = [width * block - 31, 0]
    prompts = [max(200 * block, klens[0] + block * 2), 0]
    metadata = [
        torch.tensor(value, device="cuda", dtype=torch.int32) for value in (qlens, klens, prompts)
    ]
    rates = (0.2, 30, 0.1, 30)
    actual = hpc.stem_tpd(
        scores, *metadata, block_size=block, alpha=alpha, initial_blocks=1, window_size=2
    )
    expected = _stem_reference(scores, qlens, klens, prompts, block, alpha, 1, 2, rates)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("seed", [0, 11])
@pytest.mark.parametrize("storage_offset", [0, 1, 2, 3])
def test_sampler_shared_penalty_slots_and_partial_word(seed, storage_offset):
    width = 120832
    preferred = torch.tensor([7, width - 1, width - 2, width - 9], device="cuda")
    logits = torch.zeros((4, width), device="cuda")
    logits[torch.arange(4, device="cuda"), preferred] = 50
    noise = torch.zeros_like(logits)
    size = 5 * (width // 8 + 1)
    backing = torch.full((size + storage_offset + 8,), 0xA5, device="cuda", dtype=torch.uint8)
    mask = backing[storage_offset : storage_offset + size].reshape(5, -1)
    mask.zero_()
    slots = torch.tensor([0, 4, 4, 4], device="cuda", dtype=torch.int32)
    expected = mask.clone()
    for slot, token in zip(slots.tolist(), preferred.tolist()):
        expected[slot, token // 8] |= 1 << (token % 8)
    actual = hpc.fused_sampler(
        logits,
        penalty_mask=mask,
        slot_id=slots,
        repetition_penalty=1.1,
        topk=1,
        gumbel_noise=noise if seed == 0 else None,
        seed=seed,
    )
    torch.testing.assert_close(actual[:, 0].long(), preferred)
    torch.testing.assert_close(mask, expected, atol=0, rtol=0)
    assert torch.all(backing[:storage_offset] == 0xA5)
    assert torch.all(backing[-8:] == 0xA5)


@pytest.mark.parametrize("use_tle", [False, True])
@pytest.mark.parametrize("policy", [1, 2])
def test_sampler_tle_top_p(use_tle, policy, monkeypatch):
    from hpc._triton import sampler
    from test_sampler import ref_fused_sampler

    monkeypatch.setattr(sampler, "USE_TLE", use_tle)
    torch.manual_seed(71)
    logits = torch.randn((4, 120832), device="cuda")
    noise = torch.randn_like(logits)
    p = torch.tensor([0.05, 0.5, 0.9, 1.0], device="cuda")
    k = torch.tensor([1, 7, 32, 64], device="cuda", dtype=torch.int32)
    expected, _ = ref_fused_sampler(
        logits,
        softmax_policy=hpc.SoftmaxPolicy(policy),
        topk=k,
        topp=p,
        max_topk=64,
        gumbel_noise=noise,
    )
    actual = hpc.fused_sampler(
        logits, softmax_policy=policy, topk=k, topp=p, max_topk=64, gumbel_noise=noise
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("temperature_only", [False, True])
@pytest.mark.parametrize("external", [False, True])
def test_sampler_graph_replay(temperature_only, external):
    logits = torch.zeros((8, 120832), device="cuda")
    noise = torch.randn_like(logits) if external else None
    kwargs = {"temperature": 0.7} if temperature_only else {"topk": 32}

    def forward():
        return hpc.fused_sampler(logits, gumbel_noise=noise, seed=42, **kwargs)

    for _ in range(3):
        forward()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = forward()
    samples = []
    for token in range(4):
        if external:
            noise.fill_(-100)
            noise[:, token] = 100
        graph.replay()
        samples.append(output.clone())
        if external:
            assert torch.all(output == token)
    if not external:
        assert any(not torch.equal(samples[0], value) for value in samples[1:])


def test_sampler_rng_uses_high_counter_bits():
    from hpc._triton import sampler

    logits = torch.zeros((8, 120832), device="cuda")
    hpc.fused_sampler(logits, temperature=1.0, seed=42)
    state, _ = sampler._rng_states[logits.device]
    state.fill_(0)
    first = hpc.fused_sampler(logits, temperature=1.0, seed=42)
    state.fill_(1 << 32)
    second = hpc.fused_sampler(logits, temperature=1.0, seed=42)
    assert not torch.equal(first, second)


def test_metadata_graph_on_input_gpu():
    device = min(1, torch.cuda.device_count() - 1)
    torch.cuda.set_device(0)
    with torch.cuda.device(device):
        xa = torch.randn((2, 4096), device="cuda").bfloat16()
        residual = torch.randn((2, 4, 4096), device="cuda").bfloat16()
        gates = torch.rand((2, 4), device="cuda")
        w = torch.randn((8, 16384), device="cuda") * 0.02
        scale, base = torch.ones(2, device="cuda"), torch.zeros(8, device="cuda")
        scores = torch.randn((2, 1, 3, 16), device="cuda").bfloat16()
        qlens = torch.tensor([129, 0], device="cuda", dtype=torch.int32)
        klens = torch.tensor([2048, 0], device="cuda", dtype=torch.int32)
        prompts = torch.full((2,), 2048, device="cuda", dtype=torch.int32)
        xscale = torch.randn((64, 16), device="cuda")
        lens = torch.tensor([3, 17], device="cuda", dtype=torch.int32)
        cu = torch.tensor([0, 32, 64], device="cuda", dtype=torch.int32)
        reformatted = torch.full((16, 64), 17.0, device="cuda")
        logits = torch.randn((2, 120832), device="cuda")
        noise = torch.randn_like(logits)
        temperature = torch.ones(2, device="cuda")
        stream = torch.cuda.Stream()

    def forward():
        fused = hpc.fuse_ihc_post_pre(xa, residual, gates, w, scale, base)
        mask = hpc.stem_tpd(scores, qlens, klens, prompts)
        hpc.reformat_x_scale(xscale, lens, cu, 16, reformatted)
        token = hpc.fused_sampler(logits, temperature=temperature, gumbel_noise=noise)
        full = hpc.fused_sampler(logits, topk=32, gumbel_noise=noise)
        return fused, mask, token, full

    initial = forward()
    assert initial[0][0].device.index == device and torch.cuda.current_device() == 0
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            forward()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = forward()
        for qvalues, kvalues, lengths in [
            ([129, 0], [2048, 0], [3, 17]),
            ([0, 257], [0, 1897], [21, 0]),
        ]:
            qlens.copy_(torch.tensor(qvalues, device=qlens.device, dtype=torch.int32))
            klens.copy_(torch.tensor(kvalues, device=klens.device, dtype=torch.int32))
            lens.copy_(torch.tensor(lengths, device=lens.device, dtype=torch.int32))
            xa.fill_(-0.75 if lengths[0] == 21 else 0.25)
            temperature.fill_(0.7 if lengths[0] == 21 else 1.0)
            reformatted.fill_(17)
            graph.replay()
            expected_mask = _stem_reference(
                scores, qvalues, kvalues, [2048, 2048], 128, 1.0, 4, 4, (0.2, 30, 0.1, 30)
            )
            torch.testing.assert_close(actual[1], expected_mask, atol=0, rtol=0)
            expected_scale = torch.full_like(reformatted, 17)
            offset = 0
            for group, length in enumerate(lengths):
                expected_scale[:, offset : offset + length] = xscale[
                    group * 32 : group * 32 + length
                ].T
                offset += math.ceil(length / 16) * 16
            torch.testing.assert_close(reformatted, expected_scale, atol=0, rtol=0)
            expected = forward()
            torch.testing.assert_close(actual, expected)
    assert torch.cuda.current_device() == 0


@pytest.mark.parametrize("width", [32768, 131072])
def test_topk_parallel_and_minimum_workspace(width):
    logits = torch.randn((3, width), device="cuda")
    logits[-1].zero_()
    logits[0, :7] = float("inf")
    lengths = torch.tensor([width, width - 317, width // 2], device="cuda", dtype=torch.int32)
    live = torch.tensor([3], device="cuda", dtype=torch.int32)
    counters_bytes, scratch_bytes = hpc.topk_workspace_size(3, width)
    counters = torch.zeros(counters_bytes, device="cuda", dtype=torch.uint8)
    scratch = torch.full((scratch_bytes,), 0xA5, device="cuda", dtype=torch.uint8)
    minimal = torch.empty(hpc.topk_min_workspace_size(width), device="cuda", dtype=torch.uint8)
    first = torch.empty((3, 512), device="cuda", dtype=torch.int32)
    second = torch.empty_like(first)
    hpc.topk(logits, lengths, first, live, 512, counters, scratch)
    hpc.topk(logits, lengths, second, live, 512, counters, minimal)
    for row, length in enumerate(lengths.tolist()):
        expected = logits[row, :length].topk(512).values.sort().values
        for output in (first, second):
            indices = output[row].long()
            assert indices.unique().numel() == 512
            assert torch.all((indices >= 0) & (indices < length))
            values = logits[row, indices].sort().values
            torch.testing.assert_close(values, expected, atol=0, rtol=0)
    assert not counters.any()


def test_compiled_fusion_selection_and_reformat():
    xa = torch.randn((2, 4096), device="cuda").bfloat16()
    residual = torch.randn((2, 4, 4096), device="cuda").bfloat16()
    gates = torch.rand((2, 4), device="cuda")
    w = torch.randn((8, 16384), device="cuda") * 0.02
    scale, base = torch.ones(2, device="cuda"), torch.zeros(8, device="cuda")
    scores = torch.randn((2, 1, 2, 8), device="cuda").bfloat16()
    qlens = torch.full((2,), 256, device="cuda", dtype=torch.int32)
    klens = torch.full((2,), 1024, device="cuda", dtype=torch.int32)
    xscale = torch.randn((64, 16), device="cuda")
    lens = torch.tensor([3, 17], device="cuda", dtype=torch.int32)
    cu = torch.tensor([0, 32, 64], device="cuda", dtype=torch.int32)
    logits = torch.randn((2, 120832), device="cuda")
    noise = torch.randn_like(logits)

    def forward(
        xa, residual, gates, w, scale, base, scores, qlens, klens, xscale, lens, cu, logits, noise
    ):
        fused = hpc.fuse_ihc_post_pre(xa, residual, gates, w, scale, base)
        mask = hpc.stem_tpd(scores, qlens, klens, klens)
        reformatted = hpc.reformat_x_scale(xscale, lens, cu, 16)
        token = hpc.fused_sampler(logits, temperature=0.7, gumbel_noise=noise)
        full_token = hpc.fused_sampler(logits, topk=32, gumbel_noise=noise)
        return fused, mask, reformatted, token, full_token

    args = (
        xa,
        residual,
        gates,
        w,
        scale,
        base,
        scores,
        qlens,
        klens,
        xscale,
        lens,
        cu,
        logits,
        noise,
    )
    expected = forward(*args)
    actual = torch.compile(forward, fullgraph=True)(*args)
    # Unwritten scale padding has no defined value.
    for index in range(2):
        start = 0 if index == 0 else 16
        length = (3, 17)[index]
        torch.testing.assert_close(
            actual[2][:, start : start + length], expected[2][:, start : start + length]
        )
    torch.testing.assert_close(actual[:2], expected[:2])
    torch.testing.assert_close(actual[3:], expected[3:])


def test_empty_fusion_sampler_and_validation():
    xa = torch.empty((0, 4096), device="cuda", dtype=torch.bfloat16)
    residual = torch.empty((0, 4, 4096), device="cuda", dtype=torch.bfloat16)
    gates = torch.empty((0, 4), device="cuda")
    w = torch.empty((8, 16384), device="cuda")
    y, z, post = hpc.fuse_ihc_post_pre(
        xa, residual, gates, w, torch.ones(2, device="cuda"), torch.zeros(8, device="cuda")
    )
    assert y.shape == residual.shape and z.shape == xa.shape and post.shape == gates.shape
    logits = torch.empty((0, 120832), device="cuda")
    assert hpc.fused_sampler(logits, temperature=1, seed=42).shape == (0, 1)
    assert hpc.fused_sampler(logits, topk=32, seed=42).shape == (0, 1)
    with pytest.raises(RuntimeError, match="same GPU"):
        hpc.reformat_x_scale(
            torch.ones((32, 16), device="cuda"),
            torch.ones(1, dtype=torch.int32),
            torch.tensor([0, 32], device="cuda", dtype=torch.int32),
            16,
        )
