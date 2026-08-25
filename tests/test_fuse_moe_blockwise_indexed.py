import pytest
import torch

import hpc


def _fp8(shape):
    return (torch.randn(shape, device="cuda") * 0.25).to(torch.float8_e4m3fn)


def _strided_rows(rows, columns, row_stride, dtype):
    storage = torch.empty(rows * row_stride, device="cuda", dtype=dtype)
    return torch.as_strided(storage, (rows, columns), (row_stride, 1))


def _case():
    torch.manual_seed(7)
    num_peers = 3
    rows_per_peer = 6
    logical_rows = 5
    hidden_size = 128
    intermediate_size = 128
    top_k = 2
    num_experts = 4

    inputs = []
    input_scales = []
    for _ in range(num_peers):
        shard = _strided_rows(rows_per_peer, hidden_size, 512, torch.float8_e4m3fn)
        shard.copy_(_fp8((rows_per_peer, hidden_size)))
        scale = _strided_rows(rows_per_peer, hidden_size // 128, 8, torch.float32)
        scale.copy_(torch.rand_like(scale))
        inputs.append(shard)
        input_scales.append(scale)

    # Rank-major encoding: owner_rank * rows_per_peer + local_row.
    source_rows = torch.tensor([1, 8, 15, 5, 12], device="cuda", dtype=torch.int32)
    topk_ids = torch.tensor(
        [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]], device="cuda", dtype=torch.int32
    )
    topk_weights = torch.rand((logical_rows, top_k), device="cuda", dtype=torch.float32)
    w13 = _fp8((num_experts, intermediate_size * 2, hidden_size))
    w2 = _fp8((num_experts, hidden_size, intermediate_size))
    w13_scale = torch.rand(
        (num_experts, intermediate_size * 2 // 128, 4),
        device="cuda",
        dtype=torch.float32,
    )
    w2_scale = torch.rand(
        (num_experts, hidden_size // 128, 4), device="cuda", dtype=torch.float32
    )
    return (
        inputs,
        input_scales,
        source_rows,
        w13,
        w13_scale,
        w2,
        w2_scale,
        topk_ids,
        topk_weights,
        0,
        num_experts,
        rows_per_peer,
    )


def _materialize_selected(inputs, source_rows, rows_per_peer):
    return torch.stack(
        [
            inputs[int(row) // rows_per_peer][int(row) % rows_per_peer]
            for row in source_rows.cpu()
        ]
    )


def _indexed_args(case):
    (
        inputs,
        input_scales,
        source_rows,
        w13,
        w13_scale,
        w2,
        w2_scale,
        topk_ids,
        topk_weights,
        rank_ep,
        num_experts,
        _,
    ) = case
    input_ptrs = torch.tensor(
        [tensor.data_ptr() for tensor in inputs], device="cuda", dtype=torch.int64
    )
    input_scale_ptrs = torch.tensor(
        [tensor.data_ptr() for tensor in input_scales], device="cuda", dtype=torch.int64
    )
    return (
        inputs[0],
        input_scales[0],
        input_ptrs,
        input_scale_ptrs,
        source_rows,
        w13,
        w13_scale,
        w2,
        w2_scale,
        topk_ids,
        topk_weights,
        rank_ep,
        num_experts,
    )


def test_indexed_blockwise_moe_reads_peer_rows():
    case = _case()
    (
        inputs,
        input_scales,
        source_rows,
        w13,
        w13_scale,
        w2,
        w2_scale,
        topk_ids,
        topk_weights,
        rank_ep,
        num_experts,
        rows_per_peer,
    ) = case

    expected = hpc.fuse_moe_blockwise(
        _materialize_selected(inputs, source_rows, rows_per_peer).contiguous(),
        _materialize_selected(input_scales, source_rows, rows_per_peer).contiguous(),
        w13,
        w13_scale,
        w2,
        w2_scale,
        topk_ids,
        topk_weights,
        rank_ep,
        num_experts,
    )
    result = hpc.fuse_moe_blockwise_indexed(*_indexed_args(case))

    assert result.shape == expected.shape
    torch.testing.assert_close(result, expected, rtol=2e-2, atol=2e-2)


def test_indexed_pull_blockwise_moe_matches_materialized_reference():
    case = _case()
    (
        inputs,
        input_scales,
        source_rows,
        w13,
        w13_scale,
        w2,
        w2_scale,
        topk_ids,
        topk_weights,
        rank_ep,
        num_experts,
        rows_per_peer,
    ) = case

    expected = hpc.fuse_moe_blockwise(
        _materialize_selected(inputs, source_rows, rows_per_peer).contiguous(),
        _materialize_selected(input_scales, source_rows, rows_per_peer).contiguous(),
        w13,
        w13_scale,
        w2,
        w2_scale,
        topk_ids,
        topk_weights,
        rank_ep,
        num_experts,
    )
    result = hpc.fuse_moe_blockwise_indexed_pull(*_indexed_args(case))

    assert result.shape == expected.shape
    torch.testing.assert_close(result, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
    "op", [hpc.fuse_moe_blockwise_indexed, hpc.fuse_moe_blockwise_indexed_pull]
)
def test_indexed_blockwise_moe_supports_cuda_graph_replay(op):
    # Keep every peer tensor alive while its data pointer is captured/replayed.
    case = _case()
    args = _indexed_args(case)

    op(*args)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = op(*args)
    graph.replay()
    torch.cuda.synchronize()

    assert torch.isfinite(result).all()


@pytest.mark.parametrize(
    "op", [hpc.fuse_moe_blockwise_indexed, hpc.fuse_moe_blockwise_indexed_pull]
)
def test_indexed_blockwise_moe_applies_bounded_swiglu(op):
    case = _case()
    (
        inputs,
        input_scales,
        source_rows,
        w13,
        w13_scale,
        w2,
        w2_scale,
        topk_ids,
        topk_weights,
        rank_ep,
        num_experts,
        rows_per_peer,
    ) = case
    swiglu_limit = 0.05

    expected = hpc.fuse_moe_blockwise(
        _materialize_selected(inputs, source_rows, rows_per_peer).contiguous(),
        _materialize_selected(input_scales, source_rows, rows_per_peer).contiguous(),
        w13,
        w13_scale,
        w2,
        w2_scale,
        topk_ids,
        topk_weights,
        rank_ep,
        num_experts,
        swiglu_limit=swiglu_limit,
    )
    result = op(*_indexed_args(case), swiglu_limit)

    torch.testing.assert_close(result, expected, rtol=2e-2, atol=2e-2)


def test_indexed_blockwise_moe_skips_activation_gather():
    # The pointer table does not own the peer tensors it references.
    case = _case()
    args = _indexed_args(case)

    hpc.fuse_moe_blockwise_indexed(*args)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        hpc.fuse_moe_blockwise_indexed(*args)
        torch.cuda.synchronize()

    kernel_names = {event.key for event in prof.key_averages()}
    assert not any("blockwise_gather_kernel" in name for name in kernel_names)
    assert any("group_gemm_fp8_scatter_kernel" in name for name in kernel_names)


def test_indexed_pull_uses_native_grouped_gemm_after_materialization():
    case = _case()
    args = _indexed_args(case)

    hpc.fuse_moe_blockwise_indexed_pull(*args)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        hpc.fuse_moe_blockwise_indexed_pull(*args)
        torch.cuda.synchronize()

    kernel_names = {event.key for event in prof.key_averages()}
    assert any("pull_indexed_rows_kernel" in name for name in kernel_names)
    assert any("group_gemm_blockwise_fp8_kernel" in name for name in kernel_names)
    assert not any("group_gemm_fp8_scatter_kernel" in name for name in kernel_names)


@pytest.mark.parametrize(
    "op",
    [hpc.fuse_moe_blockwise_indexed, hpc.fuse_moe_blockwise_indexed_pull],
)
def test_indexed_blockwise_moe_accepts_zero_logical_rows(op):
    args = list(_indexed_args(_case()))
    args[0] = args[0][:0]
    args[1] = args[1][:0]
    args[4] = args[4][:0]
    args[9] = args[9][:0]
    args[10] = args[10][:0]

    result = op(*args)
    torch.cuda.synchronize()

    assert result.shape == (0, 128)
    assert result.dtype == torch.bfloat16


def test_indexed_blockwise_moe_rejects_non_block_aligned_hidden_size():
    hidden_size = 64
    rows_per_peer = 1
    x = _fp8((rows_per_peer, hidden_size))
    x_scale = torch.empty((rows_per_peer, 0), device="cuda", dtype=torch.float32)
    input_ptrs = torch.tensor([x.data_ptr()], device="cuda", dtype=torch.int64)
    input_scale_ptrs = torch.tensor(
        [x_scale.data_ptr()], device="cuda", dtype=torch.int64
    )
    source_rows = torch.zeros(1, device="cuda", dtype=torch.int32)
    w13 = _fp8((1, 128, hidden_size))
    w13_scale = torch.empty((1, 1, 0), device="cuda", dtype=torch.float32)
    w2 = _fp8((1, hidden_size, hidden_size))
    w2_scale = torch.empty((1, 0, 0), device="cuda", dtype=torch.float32)
    topk_ids = torch.zeros((1, 1), device="cuda", dtype=torch.int32)
    topk_weights = torch.ones((1, 1), device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="positive multiple of 128"):
        hpc.fuse_moe_blockwise_indexed(
            x,
            x_scale,
            input_ptrs,
            input_scale_ptrs,
            source_rows,
            w13,
            w13_scale,
            w2,
            w2_scale,
            topk_ids,
            topk_weights,
            0,
            1,
        )


@pytest.mark.parametrize(
    "op", [hpc.fuse_moe_blockwise_indexed, hpc.fuse_moe_blockwise_indexed_pull]
)
@pytest.mark.parametrize(
    ("argument", "replacement", "message"),
    [
        (0, lambda args: args[0].flatten(), "x must be two-dimensional"),
        (
            1,
            lambda args: args[1].to(torch.float16),
            "x_scale dtype must be float32",
        ),
        (
            9,
            lambda args: args[9][:, :0],
            "num_topk must be in",
        ),
        (
            6,
            lambda args: args[6][:0],
            "weight scales must match their expert dimensions",
        ),
    ],
)
def test_indexed_blockwise_moe_rejects_invalid_public_inputs(
    op, argument, replacement, message
):
    args = list(_indexed_args(_case()))
    args[argument] = replacement(args)
    if argument == 9:
        args[10] = args[10][:, :0]

    with pytest.raises(RuntimeError, match=message):
        op(*args)


@pytest.mark.parametrize(
    "op", [hpc.fuse_moe_blockwise_indexed, hpc.fuse_moe_blockwise_indexed_pull]
)
def test_indexed_blockwise_moe_rejects_invalid_ep_domain(op):
    args = list(_indexed_args(_case()))
    args[-1] = 0

    with pytest.raises(RuntimeError, match="num_expert_total must be positive"):
        op(*args)


@pytest.mark.parametrize(
    "op", [hpc.fuse_moe_blockwise_indexed, hpc.fuse_moe_blockwise_indexed_pull]
)
def test_indexed_blockwise_moe_rejects_more_than_512_local_experts(op):
    args = list(_indexed_args(_case()))
    local_experts = 513
    for argument in (5, 6, 7, 8):
        repeats = (local_experts,) + (1,) * (args[argument].dim() - 1)
        args[argument] = args[argument][:1].repeat(repeats)
    args[-1] = local_experts

    with pytest.raises(RuntimeError, match="num_expert_local must be <= 512"):
        op(*args)


@pytest.mark.parametrize(
    "op", [hpc.fuse_moe_blockwise_indexed, hpc.fuse_moe_blockwise_indexed_pull]
)
def test_indexed_blockwise_moe_rejects_overflowing_padded_routes(op):
    args = list(_indexed_args(_case()))
    for argument in (5, 6, 7, 8):
        args[argument] = args[argument][:1].contiguous()
    args[-1] = torch.iinfo(torch.int32).max

    with pytest.raises(RuntimeError, match="padded route count exceeds int32 capacity"):
        op(*args)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
@pytest.mark.parametrize(
    "op", [hpc.fuse_moe_blockwise_indexed, hpc.fuse_moe_blockwise_indexed_pull]
)
def test_indexed_blockwise_moe_rejects_cross_device_metadata(op):
    args = list(_indexed_args(_case()))
    args[4] = args[4].to("cuda:1")

    with pytest.raises(RuntimeError, match="same CUDA device"):
        op(*args)


@pytest.mark.parametrize(
    "op", [hpc.fuse_moe_blockwise_indexed, hpc.fuse_moe_blockwise_indexed_pull]
)
def test_indexed_blockwise_moe_zeroes_out_of_range_peer_rows(op):
    case = _case()
    args = list(_indexed_args(case))
    rows_per_peer = case[-1]
    num_peers = len(case[0])
    args[4] = args[4].clone()
    args[4][0] = num_peers * rows_per_peer

    result = op(*args)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        result[0].float(),
        torch.zeros_like(result[0], dtype=torch.float32),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    ("op_name", "mutated_arguments"),
    [
        (
            "prepare_indexed_input",
            {"activation", "activation_scale", "output_ids", "output_weights"},
        ),
        ("scatter_indexed_output", {"output_ptrs"}),
        ("fuse_moe_ep_publish", {"signal_ptrs"}),
        ("fuse_moe_ep_wait", {"local_signal"}),
    ],
)
def test_side_effect_only_ops_declare_mutation(op_name, mutated_arguments):
    schema = torch._C._dispatch_find_schema_or_throw(f"hpc::{op_name}", "").schema()
    actual = {
        argument.name
        for argument in schema.arguments
        if argument.alias_info is not None and argument.alias_info.is_write
    }
    assert actual == mutated_arguments


def test_indexed_fake_result_preserves_input_device():
    from hpc.fuse_moe import fuse_moe_blockwise_indexed_fake

    x = torch.empty((0, 128), device="meta", dtype=torch.float8_e4m3fn)
    topk_ids = torch.empty((0, 2), device="meta", dtype=torch.int32)
    result = fuse_moe_blockwise_indexed_fake(
        x,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        topk_ids,
        None,
        0,
        4,
    )

    assert result.device == x.device
    assert result.dtype == torch.bfloat16
