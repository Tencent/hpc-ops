import sys
import os
import pytest
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import torch

# The kernel collects top-k indices via atomicAdd, so the output order differs
# between runs while the index set stays identical. conftest's sanitizer hook
# re-runs every call, which would make this exhaustive module prohibitively
# expensive. A focused split-and-spill sanitizer case lives in the companion
# test_topk_filtered_sanitizer.py module.
pytestmark = [
    pytest.mark.skipif(bool(os.getenv("SANITIZER_CHECK")), reason="skip sanitizer"),
    pytest.mark.skipif(
        torch.cuda.get_device_capability() not in ((9, 0), (10, 3)),
        reason="Top-K is implemented for sm90 and sm103",
    ),
]

TOP_K = 2048
SPLIT_TEST_N = 270336


def test_fake_tensor():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        logits = torch.empty((4, 8192), dtype=torch.float32, device="cuda")
        ke = torch.full((4,), 8192, dtype=torch.int32, device="cuda")
        output = torch.empty((4, TOP_K), dtype=torch.int32, device="cuda")
        num_valid_rows = torch.full((1,), 4, dtype=torch.int32, device="cuda")
        result = hpc.topk_filtered(logits, ke, output, num_valid_rows, TOP_K)

    assert result is output


def _sm_count():
    return torch.cuda.get_device_properties(0).multi_processor_count


def _sm_major():
    return torch.cuda.get_device_properties(0).major


def _persistent_grid():
    return 3 * _sm_count()


def _kv_split_schedule_boundaries():
    """Return both ends of every split-count band active on this GPU."""
    grid = _persistent_grid()
    first_split_row = 17 if _sm_major() >= 9 else 1
    eight_hi = min(192, max(16, _sm_count() // 8))
    four_hi = min(192, grid // 4)
    bands = (
        (8, first_split_row, eight_hi),
        (4, max(first_split_row, eight_hi + 1), four_hi),
        (2, max(first_split_row, four_hi + 1), 192),
    )
    cases = []
    for splits, lo, hi in bands:
        if lo > hi:
            continue
        cases.append((lo, splits))
        if hi != lo:
            cases.append((hi, splits))
    return cases


def _expected_split_workspace_bytes(m, n, splits):
    spill = max(_persistent_grid(), m) * 2 * n
    split_state = m * splits + m * splits * 2048 + m * n
    return (spill + split_state) * 4


def _assert_known_upper_tail(out, n, num_monotonic_rows):
    expected = torch.arange(TOP_K, dtype=torch.int32, device="cuda")
    if num_monotonic_rows:
        actual = out[:num_monotonic_rows].sort(dim=1).values
        assert torch.equal(actual, expected.expand(num_monotonic_rows, -1))
    for row in range(num_monotonic_rows, out.shape[0]):
        assert (out[row] >= 0).all() and (out[row] < n).all()
        assert out[row].unique().numel() == TOP_K


def _validate(logits, indices, ke, top_k=TOP_K):
    for r in range(ke.numel()):
        length = int(ke[r])
        k_i = min(top_k, length)
        if k_i <= 0:
            continue
        # Reference top-k runs on CPU: torch's GPU mbtopk reads uninitialized
        # slots of its own scratch buffer, which trips compute-sanitizer
        # initcheck (a torch internal, unrelated to the kernel under test).
        valid = logits[r, :length].cpu()
        ref = valid.topk(k_i, dim=-1)[1]
        cu = indices[r, :k_i].cpu()
        cu = cu[cu >= 0]
        assert cu.numel() == k_i, f"row {r}: expected {k_i} valid indices"
        assert (cu < length).all(), f"row {r}: OOB index (len {length})"
        assert cu.unique().numel() == k_i, f"row {r}: duplicate indices"
        cset, rset = set(cu.tolist()), set(ref.tolist())
        if cset == rset:
            continue
        cv = valid[cu].sort(descending=True)[0]
        rv = valid[ref].sort(descending=True)[0]
        assert torch.equal(cv, rv), f"row {r}: top-k values mismatch len={length} k={k_i}"


def _alloc_scratch(logits, ke):
    # counters must be zeroed; workspace needs no initialization, so it is filled
    # with garbage here to keep proving that.
    cnt_bytes, ws_bytes = hpc.topk_filtered_workspace_size(ke.numel(), logits.shape[1])
    cnt = torch.zeros(cnt_bytes, dtype=torch.uint8, device="cuda")
    ws = torch.full((ws_bytes,), 0xA5, dtype=torch.uint8, device="cuda")
    return cnt, ws


def _run(seq_lens, n=None, dirty_pad=False, extra_rows=0, top_k=TOP_K):
    torch.manual_seed(0)
    m_valid = len(seq_lens)
    n = n or max(seq_lens)
    m = m_valid + extra_rows
    logits = torch.randn(m, n, dtype=torch.float32, device="cuda")
    pad = 1e9 if dirty_pad else float("-inf")
    for i, sl in enumerate(seq_lens):
        if sl < n:
            logits[i, sl:] = pad
    ke = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    num_valid = torch.tensor([m_valid], dtype=torch.int32, device="cuda")
    # Kernel does not initialize the output buffer by design; pre-fill it so
    # compute-sanitizer initcheck does not flag reads of untouched slots.
    out = torch.full((m_valid, top_k), -1, dtype=torch.int32, device="cuda")
    hpc.topk_filtered(logits, ke, out, num_valid, top_k)
    torch.cuda.synchronize()
    _validate(logits, out, ke.cpu(), top_k=top_k)


@pytest.mark.parametrize(
    "seq_lens",
    [
        [1, 100, 2048],  # trivial (<= TopK)
        [2049, 2100, 3000],  # just above k
        [4096, 8192],  # medium
        [5000, 10000, 16384],
        [40000, 65536],  # candidate gmem spill regime
        [2000, 6000, 30000, 60000],  # mixed
    ],
)
def test_paths(seq_lens):
    _run(seq_lens)


def test_full_64k():
    _run([65536, 65536, 65536])


@pytest.mark.parametrize(
    "seq_lens,n",
    [
        ([200, 800, 3000], None),  # trivial (<= top_k=512) + non-trivial
        ([2048, 4096, 8192, 16384], None),  # medium
        ([40000, 65536], None),  # smem-spill regime
        ([80000, 131072], 131072),  # wide rows
    ],
)
def test_top_k_512(seq_lens, n):
    _run(seq_lens, n=n, top_k=512)


@pytest.mark.parametrize(
    "seq_lens,n",
    [
        ([65537, 70000], 70000),  # just past 64k
        ([80000, 100000, 131072], 131072),  # wide-row regime
        ([3000, 50000, 120000], 131072),  # mixed lengths, N > 64k
    ],
)
def test_int32_index_n_over_64k(seq_lens, n):
    _run(seq_lens, n=n)


def test_int32_index_padded_columns():
    # N > 64k with rows shorter than N exercises wide rows and padding.
    _run([5000, 40000, 90000], n=100000)


def test_int32_index_dirty_padding_excluded():
    _run([9000, 60000, 90000], n=100000, dirty_pad=True)


def test_spec_decode_lengths():
    torch.manual_seed(1)
    batch, next_n = 4, 4
    seq = torch.randint(4000, 60000, (batch,), dtype=torch.int32)
    offsets = torch.arange(next_n, dtype=torch.int32)
    ke = (seq.unsqueeze(1) - next_n + 1 + offsets).flatten().tolist()
    _run(ke, n=60000)


def test_padded_columns():
    _run([3000, 5000, 8000, 12000], n=65536)


def test_dirty_padding_excluded():
    _run([3000, 9000, 50000], n=65536, dirty_pad=True)


def test_extra_physical_rows():
    _run([2049, 5000, 40000], extra_rows=5)


def test_num_valid_rows_partial():
    # num_valid_rows < ke.numel(): the persistent grid processes only the first
    # nvr rows; rows [nvr, M) are left untouched (stay at the pre-filled -1).
    torch.manual_seed(3)
    seq_lens = [3000, 40000, 60000, 8000, 20000]
    m_valid, n = len(seq_lens), max(seq_lens)
    logits = torch.randn(m_valid, n, dtype=torch.float32, device="cuda")
    for i, sl in enumerate(seq_lens):
        if sl < n:
            logits[i, sl:] = float("-inf")
    ke = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    out = torch.full((m_valid, TOP_K), -1, dtype=torch.int32, device="cuda")
    nvr = 3
    num_valid = torch.tensor([nvr], dtype=torch.int32, device="cuda")
    hpc.topk_filtered(logits, ke, out, num_valid, TOP_K)
    torch.cuda.synchronize()
    _validate(logits[:nvr], out[:nvr], ke.cpu()[:nvr])
    assert (out[nvr:] == -1).all(), "rows beyond num_valid_rows must be untouched"


def test_workspace_size_query():
    # Python size helper matches externally allocated buffers.
    n = 65536
    seq_lens = [40000, 65536]
    logits = torch.randn(len(seq_lens), n, dtype=torch.float32, device="cuda")
    ke = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    cnt_bytes, ws_bytes = hpc.topk_filtered_workspace_size(ke.numel(), logits.shape[1])
    assert cnt_bytes > 0 and cnt_bytes % 4 == 0
    assert ws_bytes > 0 and ws_bytes % 4 == 0
    assert ws_bytes >= hpc.topk_filtered_min_workspace_size(logits.shape[1])
    cnt = torch.zeros(cnt_bytes, dtype=torch.uint8, device="cuda")
    ws = torch.empty(ws_bytes, dtype=torch.uint8, device="cuda")
    out = torch.full((len(seq_lens), TOP_K), -1, dtype=torch.int32, device="cuda")
    num_valid = torch.tensor([len(seq_lens)], dtype=torch.int32, device="cuda")
    hpc.topk_filtered(logits, ke, out, num_valid, TOP_K, counters=cnt, workspace=ws)
    torch.cuda.synchronize()
    _validate(logits, out, ke.cpu())


def test_peak_workspace_size_bounds_every_row_count():
    # The peak query exists for callers that reserve one buffer up front, before
    # the row counts they will see are known, so it must bound every row count --
    # including the interior maximum the KV-split path produces, which is why a
    # row-count ceiling is not a safe substitute.
    for n in (4096, 65536, 131072, 262144, 393216):
        peak = hpc.topk_filtered_peak_workspace_size(n)
        assert peak >= hpc.topk_filtered_min_workspace_size(n)
        worst_rows, worst = 0, 0
        for num_rows in range(1, 4096):
            ws_bytes = hpc.topk_filtered_workspace_size(num_rows, n)[1]
            assert ws_bytes <= peak, f"n={n} num_rows={num_rows} exceeds the peak"
            if ws_bytes > worst:
                worst_rows, worst = num_rows, ws_bytes
        assert worst == peak, f"n={n}: peak is not tight (worst at num_rows={worst_rows})"


def test_peak_workspace_size_serves_every_split_schedule():
    # Reuse one peak-sized allocation across both ends of every active split
    # band, then across the adjacent persistent route. The final row forces a
    # candidate spill, so this exercises the physical layout rather than only
    # comparing the size arithmetic.
    n = SPLIT_TEST_N
    split_cases = _kv_split_schedule_boundaries()
    assert split_cases, "this GPU exposes no KV-split schedule"
    cases = [(m, splits) for m, splits in split_cases]
    cases.append((193, None))

    peak = hpc.topk_filtered_peak_workspace_size(n)
    ws = torch.full((peak,), 0xA5, dtype=torch.uint8, device="cuda")
    max_m = max(m for m, _ in cases)
    cnt = torch.zeros(
        hpc.topk_filtered_workspace_size(max_m, n)[0], dtype=torch.uint8, device="cuda"
    )
    base = -torch.arange(n, dtype=torch.float32, device="cuda") / n

    for m, splits in cases:
        required = hpc.topk_filtered_workspace_size(m, n)[1]
        assert required <= peak
        if splits is None:
            assert required == hpc.topk_filtered_min_workspace_size(n)
        else:
            assert required == _expected_split_workspace_bytes(m, n, splits)

        logits = base.expand(m, -1).clone()
        logits[-1].zero_()
        ke = torch.full((m,), n, dtype=torch.int32, device="cuda")
        out = torch.full((m, TOP_K), -1, dtype=torch.int32, device="cuda")
        num_valid = torch.tensor([m], dtype=torch.int32, device="cuda")
        hpc.topk_filtered(logits, ke, out, num_valid, TOP_K, cnt, ws)
        torch.cuda.synchronize()

        _assert_known_upper_tail(out, n, m - 1)
        assert not cnt.any(), f"m={m} left the counters dirty"


def test_bounded_row_local_workspace_does_not_grow_with_rows():
    n = 65536
    grid = _persistent_grid()
    reference = hpc.topk_filtered_workspace_size(grid, n)[1]
    for m in (grid + 1, max(grid + 2, 512), 1024, 4096):
        assert hpc.topk_filtered_workspace_size(m, n)[1] == reference


def test_zero_valid_rows():
    # num_valid_rows == 0: the kernel must return without touching output, and
    # must leave the counters zeroed so the next launch can reuse them as-is.
    m, n = 8, 65536
    logits = torch.randn((m, n), dtype=torch.float32, device="cuda")
    ke = torch.full((m,), n, dtype=torch.int32, device="cuda")
    out = torch.full((m, TOP_K), -1, dtype=torch.int32, device="cuda")
    cnt, ws = _alloc_scratch(logits, ke)

    hpc.topk_filtered(
        logits, ke, out, torch.zeros(1, dtype=torch.int32, device="cuda"), TOP_K, cnt, ws
    )
    torch.cuda.synchronize()
    assert (out == -1).all(), "no row is valid, so output must be untouched"

    # The same buffers, with no memset in between, must still give exact results.
    num_valid = torch.tensor([m], dtype=torch.int32, device="cuda")
    hpc.topk_filtered(logits, ke, out, num_valid, TOP_K, cnt, ws)
    torch.cuda.synchronize()
    _validate(logits, out, ke.cpu())


@pytest.mark.parametrize(
    "m,n",
    [
        (31, 196608),  # KV-split path: also recycles the per-row arrive counters
        (2048, 65536),  # one CTA per row
    ],
)
def test_counters_are_self_cleaning(m, n):
    # The kernel hands the counters back zeroed, so one buffer serves repeated
    # launches with no memset in between. A stale work cursor or a stale per-row
    # arrive counter would corrupt the next launch, so every repetition has to stay
    # exact.
    torch.manual_seed(17)
    logits = torch.randn((m, n), dtype=torch.float32, device="cuda")
    ke = torch.full((m,), n, dtype=torch.int32, device="cuda")
    num_valid = torch.tensor([m], dtype=torch.int32, device="cuda")
    cnt, ws = _alloc_scratch(logits, ke)

    for _ in range(3):
        out = torch.full((m, TOP_K), -1, dtype=torch.int32, device="cuda")
        hpc.topk_filtered(logits, ke, out, num_valid, TOP_K, cnt, ws)
        torch.cuda.synchronize()
        # Every counter, including the per-row arrive counters, comes back zeroed.
        assert not cnt.any(), "counters must come back zeroed"
        _validate(logits, out, ke.cpu())


@pytest.mark.parametrize(
    "m,n,nvr",
    [
        (_sm_count(), 16384, _sm_count()),  # one static wave; no queue is entered
        (_persistent_grid() + 1, 65536, 32),  # captured capacity has a queue tail
    ],
)
def test_row_local_static_first_wave_repeated(m, n, nvr):
    torch.manual_seed(31 + m)
    logits = torch.randn((m, n), dtype=torch.float32, device="cuda")
    ke = torch.full((m,), n, dtype=torch.int32, device="cuda")
    out = torch.full((m, TOP_K), -1, dtype=torch.int32, device="cuda")
    num_valid = torch.tensor([nvr], dtype=torch.int32, device="cuda")
    cnt, ws = _alloc_scratch(logits, ke)

    for _ in range(3):
        hpc.topk_filtered(logits, ke, out, num_valid, TOP_K, cnt, ws)
        torch.cuda.synchronize()
        assert not cnt.any(), "static-first-wave launch left queue state dirty"
        _validate(logits[:nvr], out[:nvr], ke.cpu()[:nvr])
        assert (out[nvr:] == -1).all(), "rows beyond num_valid_rows must be untouched"


def test_persistent_queue_hardware_boundary_cuda_graph():
    # N is one element beyond the bounded row-local envelope, so a captured
    # grid+1 batch enters the persistent sampled queue. Exercise both sides of
    # its hardware-dependent grid boundary and force spill on the last two rows.
    grid = _persistent_grid()
    m, n = grid + 1, 65537
    base = -torch.arange(n, dtype=torch.float32, device="cuda") / n
    logits = base.expand(m, -1).clone()
    logits[grid - 1 :].zero_()
    ke = torch.full((m,), n, dtype=torch.int32, device="cuda")
    out = torch.full((m, TOP_K), -1, dtype=torch.int32, device="cuda")
    num_valid = torch.tensor([m], dtype=torch.int32, device="cuda")
    cnt, ws = _alloc_scratch(logits, ke)

    hpc.topk_filtered(logits, ke, out, num_valid, TOP_K, cnt, ws)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        hpc.topk_filtered(logits, ke, out, num_valid, TOP_K, cnt, ws)

    for nvr in (0, 1, grid - 1, grid, grid + 1):
        out.fill_(-1)
        num_valid.fill_(nvr)
        graph.replay()
        torch.cuda.synchronize()

        _assert_known_upper_tail(out[:nvr], n, min(nvr, grid - 1))
        assert (out[nvr:] == -1).all()
        assert not cnt.any(), f"num_valid_rows={nvr} left queue state dirty"


def test_padded_output_stride():
    torch.manual_seed(2)
    seq_lens = [2049, 5000, 40000]
    m_valid, n = len(seq_lens), max(seq_lens)
    logits = torch.randn(m_valid, n, dtype=torch.float32, device="cuda")
    for i, sl in enumerate(seq_lens):
        if sl < n:
            logits[i, sl:] = float("-inf")
    ke = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    num_valid = torch.tensor([m_valid], dtype=torch.int32, device="cuda")
    # Same as _run: initialize the padded output buffer to avoid initcheck reports.
    out = torch.full((m_valid, TOP_K + 64), -1, dtype=torch.int32, device="cuda")
    hpc.topk_filtered(logits, ke, out, num_valid, TOP_K)
    torch.cuda.synchronize()
    _validate(logits, out[:, :TOP_K], ke.cpu())


@pytest.mark.parametrize(
    "seq_lens,n,dirty_pad",
    [
        ([2049, 3000, 5000], 8192, False),  # short-row/fallback paths
        ([65536, 131072, 244650], 244652, False),  # sampled fast path + wide index
        ([9000, 60000, 90000], 100000, True),  # padded columns are excluded
    ],
)
def test_sampled_exact_paths(seq_lens, n, dirty_pad):
    _run(seq_lens, n=n, dirty_pad=dirty_pad)


def test_sampled_exact_adversarial_fallback():
    # Every sampled position is deliberately much larger than every unsampled
    # position. The sampled threshold therefore admits only N/64=1024 values,
    # forcing the <=K fallback to the full coarse histogram.
    n = 65536
    logits = torch.zeros((1, n), dtype=torch.float32, device="cuda")
    offset = 13  # row=0: (row * 17 + 13) & 63
    logits[0, offset::64] = 100.0
    ke = torch.tensor([n], dtype=torch.int32, device="cuda")
    out = torch.full((1, TOP_K), -1, dtype=torch.int32, device="cuda")
    num_valid = torch.tensor([1], dtype=torch.int32, device="cuda")
    hpc.topk_filtered(logits, ke, out, num_valid, TOP_K)
    torch.cuda.synchronize()
    _validate(logits, out, ke.cpu())


@pytest.mark.parametrize("m", [16, 2048])
def test_exact_auto_dispatch(m):
    n = 65536
    logits = torch.randn((m, n), dtype=torch.float32, device="cuda")
    ke = torch.full((m,), n, dtype=torch.int32, device="cuda")
    out = torch.full((m, TOP_K), -1, dtype=torch.int32, device="cuda")
    num_valid = torch.tensor([m], dtype=torch.int32, device="cuda")
    hpc.topk_filtered(logits, ke, out, num_valid, TOP_K)
    torch.cuda.synchronize()
    _validate(logits, out, ke.cpu())


def test_exact_auto_cuda_graph_replay():
    m, n = 2048, 65536
    logits = torch.randn((m, n), dtype=torch.float32, device="cuda")
    ke = torch.full((m,), n, dtype=torch.int32, device="cuda")
    out = torch.full((m, TOP_K), -1, dtype=torch.int32, device="cuda")
    num_valid = torch.tensor([m], dtype=torch.int32, device="cuda")
    cnt, workspace = _alloc_scratch(logits, ke)

    hpc.topk_filtered(logits, ke, out, num_valid, TOP_K, cnt, workspace)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        hpc.topk_filtered(logits, ke, out, num_valid, TOP_K, cnt, workspace)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    assert int(num_valid.item()) == m
    _validate(logits, out, ke.cpu())


# ---------------------------------------------------------------------------
# Row-local exact family: register-resident rows through 16K and 1024-thread
# streaming over the calibrated resident-wave M/N envelope.
# ---------------------------------------------------------------------------


def _row_local_dispatch_boundary_shapes():
    sm = _sm_count()
    grid = _persistent_grid()
    return [
        (1, 4096),
        (sm, 16384),
        (sm + 1, 16384),
        (2 * sm, 16384),
        (2 * sm + 1, 16384),
        (sm, 65536),
        (sm + 1, 65536),
        (2 * sm, 131072),
        (2 * sm + 1, 131072),
        (sm, 262144),
        (sm + 1, 262144),
        (31, 131072),
        (32, 131072),
        (63, 196608),
        (64, 196608),
        (grid, 49152),
        (grid + 1, 49152),
        (max(grid + 1, 512), 32768),
    ]


@pytest.mark.parametrize(
    "m,n",
    _row_local_dispatch_boundary_shapes(),
)
def test_short_exact_dispatch_boundaries(m, n):
    torch.manual_seed(30 + m)
    logits = torch.randn((m, n), dtype=torch.float32, device="cuda")
    ke = torch.full((m,), n, dtype=torch.int32, device="cuda")
    out = torch.full((m, TOP_K), -1, dtype=torch.int32, device="cuda")
    num_valid = torch.tensor([m], dtype=torch.int32, device="cuda")
    hpc.topk_filtered(logits, ke, out, num_valid, TOP_K)
    torch.cuda.synchronize()
    _validate(logits, out, ke.cpu())


def test_short_exact_special_values_and_ties():
    m, n = 512, 8192
    logits = torch.zeros((m, n), dtype=torch.float32, device="cuda")
    logits[:, 0::17] = 1.0
    logits[:, 1::17] = -0.0
    logits[:, 2::17] = float("inf")
    logits[:, 3::17] = float("-inf")
    ke = torch.tensor([n - i * 137 for i in range(m)], dtype=torch.int32, device="cuda")
    out = torch.full((m, TOP_K), -1, dtype=torch.int32, device="cuda")
    num_valid = torch.tensor([m], dtype=torch.int32, device="cuda")
    hpc.topk_filtered(logits, ke, out, num_valid, TOP_K)
    torch.cuda.synchronize()
    _validate(logits, out, ke.cpu())


@pytest.mark.parametrize("candidate_count", [31, 32, 33, 63, 64, 65, 127, 128, 129, 1024, 1025])
def test_row_local_candidate_resolver_boundaries(candidate_count):
    m, n = 1, 8192
    remaining = max(1, candidate_count // 2)
    above = TOP_K - remaining
    logits = torch.full((m, n), -100.0, dtype=torch.float32, device="cuda")
    logits[0, :above] = 10.0
    logits[0, above : above + candidate_count] = torch.linspace(
        1.0 - 2e-4,
        1.0 + 2e-4,
        candidate_count,
        dtype=torch.float32,
        device="cuda",
    )
    ke = torch.full((m,), n, dtype=torch.int32, device="cuda")
    out = torch.full((m, TOP_K), -1, dtype=torch.int32, device="cuda")
    num_valid = torch.tensor([m], dtype=torch.int32, device="cuda")

    hpc.topk_filtered(logits, ke, out, num_valid, TOP_K)
    torch.cuda.synchronize()
    _validate(logits, out, ke.cpu())


@pytest.mark.parametrize("n", [8190, 8191, 8192])
def test_bounded_row_local_vector_widths(n):
    _run([n] * (_persistent_grid() + 1), n=n)


@pytest.mark.parametrize(
    "m,n",
    [
        (32, 16384),
        (32, 131072),
        (64, 196608),
        (_sm_count() + 1, 32768),
        (2 * _sm_count(), 131072),
    ],
)
def test_short_exact_cuda_graph_replay(m, n):
    torch.manual_seed(40 + m)
    logits = torch.randn((m, n), dtype=torch.float32, device="cuda")
    ke = torch.full((m,), n, dtype=torch.int32, device="cuda")
    out = torch.full((m, TOP_K), -1, dtype=torch.int32, device="cuda")
    num_valid = torch.tensor([m], dtype=torch.int32, device="cuda")
    cnt, workspace = _alloc_scratch(logits, ke)

    hpc.topk_filtered(logits, ke, out, num_valid, TOP_K, cnt, workspace)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        hpc.topk_filtered(logits, ke, out, num_valid, TOP_K, cnt, workspace)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    _validate(logits, out, ke.cpu())


@pytest.mark.parametrize("n", [16384, 32768])
def test_short_exact_cuda_graph_mutable_valid_rows(n):
    grid = _persistent_grid()
    m = grid + 1
    torch.manual_seed(337)
    logits = torch.randn((m, n), dtype=torch.float32, device="cuda")
    ke = torch.full((m,), n, dtype=torch.int32, device="cuda")
    out = torch.full((m, TOP_K), -1, dtype=torch.int32, device="cuda")
    num_valid = torch.tensor([m], dtype=torch.int32, device="cuda")
    cnt, workspace = _alloc_scratch(logits, ke)

    hpc.topk_filtered(logits, ke, out, num_valid, TOP_K, cnt, workspace)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        hpc.topk_filtered(logits, ke, out, num_valid, TOP_K, cnt, workspace)

    boundaries = (1, 2 * _sm_count(), 2 * _sm_count() + 1, grid - 1, grid, grid + 1, 0)
    for nvr in dict.fromkeys(v for v in boundaries if 0 <= v <= m):
        out.fill_(-1)
        num_valid.fill_(nvr)
        graph.replay()
        torch.cuda.synchronize()
        assert not cnt.any(), f"num_valid_rows={nvr} left queue state dirty"
        if nvr:
            _validate(logits[:nvr], out[:nvr], ke[:nvr].cpu())
        assert (out[nvr:] == -1).all()


def test_row_local_cuda_graph_mutable_lengths():
    grid = _persistent_grid()
    m, n = grid + 1, 32768
    base = -torch.arange(n, dtype=torch.float32, device="cuda") / n
    logits = base.expand(m, -1).clone()
    ke = torch.full((m,), n, dtype=torch.int32, device="cuda")
    out = torch.full((m, TOP_K), -1, dtype=torch.int32, device="cuda")
    num_valid = torch.tensor([m], dtype=torch.int32, device="cuda")
    cnt, workspace = _alloc_scratch(logits, ke)

    hpc.topk_filtered(logits, ke, out, num_valid, TOP_K, cnt, workspace)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        hpc.topk_filtered(logits, ke, out, num_valid, TOP_K, cnt, workspace)

    patterns = (
        (TOP_K, 8192, 16384, 16385, n),
        (n, 16385, 16384, 8192, TOP_K),
    )
    expected = torch.arange(TOP_K, dtype=torch.int32, device="cuda").expand(m, -1)
    rows = torch.arange(m, device="cuda")
    for pattern in patterns:
        values = torch.tensor(pattern, dtype=torch.int32, device="cuda")
        ke.copy_(values[rows % len(pattern)])
        out.fill_(-1)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out.sort(dim=1).values, expected)
        assert not cnt.any(), "mixed row-local replay left queue state dirty"


# ---------------------------------------------------------------------------
# KV-split path: small batch + long rows makes choose_splits() return > 1, so a
# row is processed by several cooperating CTAs. These shapes stay outside the
# row-local envelope on every supported SM count.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "m,n",
    [
        (17, SPLIT_TEST_N),
        (31, 196608),
        (63, 262145),
        (148, 300000),
        (192, SPLIT_TEST_N),
    ],
)
def test_kv_split_uniform_rows(m, n):
    torch.manual_seed(11)
    logits = torch.randn((m, n), dtype=torch.float32, device="cuda")
    ke = torch.full((m,), n, dtype=torch.int32, device="cuda")
    out = torch.full((m, TOP_K), -1, dtype=torch.int32, device="cuda")
    num_valid = torch.tensor([m], dtype=torch.int32, device="cuda")
    cnt, ws = _alloc_scratch(logits, ke)
    hpc.topk_filtered(logits, ke, out, num_valid, TOP_K, cnt, ws)
    torch.cuda.synchronize()
    _validate(logits, out, ke.cpu())


def test_kv_split_row_owned_spill_across_grid_boundary():
    # Use the first two-way split whose direct task grid is larger than the
    # persistent CTA pool. H20 resolves this to 118 rows and 236 tasks over 234
    # pool slots. GPUs with at least 128 SMs cannot reach this condition before
    # the production split-row ceiling and therefore skip it explicitly.
    grid = _persistent_grid()
    first_split_row = 17 if _sm_major() >= 9 else 1
    eight_hi = max(16, _sm_count() // 8)
    m = max(first_split_row, eight_hi + 1, grid // 4 + 1, grid // 2 + 1)
    if m > 192:
        pytest.skip("KV-split task grid cannot exceed the persistent pool on this GPU")
    n = SPLIT_TEST_N
    _, recommended = hpc.topk_filtered_workspace_size(m, n)
    assert recommended == _expected_split_workspace_bytes(m, n, 2)
    assert 2 * m > grid

    base = -torch.arange(n, dtype=torch.float32, device="cuda") / n
    logits = base.expand(m, -1).clone()
    logits[-1].zero_()
    ke = torch.full((m,), n, dtype=torch.int32, device="cuda")
    out = torch.full((m, TOP_K), -1, dtype=torch.int32, device="cuda")
    num_valid = torch.tensor([m], dtype=torch.int32, device="cuda")
    cnt, ws = _alloc_scratch(logits, ke)

    for _ in range(5):
        hpc.topk_filtered(logits, ke, out, num_valid, TOP_K, cnt, ws)
    torch.cuda.synchronize()

    expected = torch.arange(TOP_K, dtype=torch.int32, device="cuda").expand(m - 1, -1)
    assert torch.equal(out[:-1].sort(dim=1).values, expected)
    assert (out[-1] >= 0).all() and (out[-1] < n).all()
    assert out[-1].unique().numel() == TOP_K
    assert not cnt.any(), "KV-split must return all arrival counters to zero"


def test_kv_split_ragged_rows():
    # Ragged lengths inside the split region: segment bounds are derived from
    # each row's own valid length, and short rows must still be exact.
    torch.manual_seed(12)
    n = 196608
    seq_lens = [100, 2048, 2049, 9000, 65536, 100000, n, 70000] * 2
    seq_lens.append(110000)
    m = len(seq_lens)
    logits = torch.randn((m, n), dtype=torch.float32, device="cuda")
    for i, sl in enumerate(seq_lens):
        if sl < n:
            logits[i, sl:] = float("-inf")
    ke = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    out = torch.full((m, TOP_K), -1, dtype=torch.int32, device="cuda")
    num_valid = torch.tensor([m], dtype=torch.int32, device="cuda")
    cnt, ws = _alloc_scratch(logits, ke)
    hpc.topk_filtered(logits, ke, out, num_valid, TOP_K, cnt, ws)
    torch.cuda.synchronize()
    _validate(logits, out, ke.cpu())


def test_kv_split_dirty_padding_excluded():
    torch.manual_seed(13)
    n = 196608
    seq_lens = [70000, 90000, 131072, 80000] * 4 + [150000]
    m = len(seq_lens)
    logits = torch.randn((m, n), dtype=torch.float32, device="cuda")
    for i, sl in enumerate(seq_lens):
        if sl < n:
            logits[i, sl:] = 1e9  # padding must be excluded despite being large
    ke = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    out = torch.full((m, TOP_K), -1, dtype=torch.int32, device="cuda")
    num_valid = torch.tensor([m], dtype=torch.int32, device="cuda")
    cnt, ws = _alloc_scratch(logits, ke)
    hpc.topk_filtered(logits, ke, out, num_valid, TOP_K, cnt, ws)
    torch.cuda.synchronize()
    _validate(logits, out, ke.cpu())


def test_kv_split_forces_fallback():
    # Every sampled position is far larger than every unsampled one, so the
    # sampled phases contain only TOP_K or TOP_K + 1 high values. All but the
    # phase starting at zero therefore force the single-CTA full-row fallback.
    n = 131073
    m = 17
    logits = torch.zeros((m, n), dtype=torch.float32, device="cuda")
    for r in range(m):
        offset = (r * 17 + 13) % 64
        logits[r, offset::64] = 100.0
    ke = torch.full((m,), n, dtype=torch.int32, device="cuda")
    out = torch.full((m, TOP_K), -1, dtype=torch.int32, device="cuda")
    num_valid = torch.tensor([m], dtype=torch.int32, device="cuda")
    cnt, ws = _alloc_scratch(logits, ke)
    hpc.topk_filtered(logits, ke, out, num_valid, TOP_K, cnt, ws)
    torch.cuda.synchronize()
    _validate(logits, out, ke.cpu())


def test_kv_split_matches_single_cta():
    # The split result must be the same index set the single-CTA path produces.
    # The minimum workspace is too small for the split path, so it gives a
    # reference from the same build on the same data.
    torch.manual_seed(14)
    m, n = 31, 196608
    logits = torch.randn((m, n), dtype=torch.float32, device="cuda")
    ke = torch.full((m,), n, dtype=torch.int32, device="cuda")
    num_valid = torch.tensor([m], dtype=torch.int32, device="cuda")

    out_split = torch.full((m, TOP_K), -1, dtype=torch.int32, device="cuda")
    cnt, ws_split = _alloc_scratch(logits, ke)
    hpc.topk_filtered(logits, ke, out_split, num_valid, TOP_K, cnt, ws_split)

    out_single = torch.full((m, TOP_K), -1, dtype=torch.int32, device="cuda")
    ws_single = torch.empty(
        hpc.topk_filtered_min_workspace_size(logits.shape[1]), dtype=torch.uint8, device="cuda"
    )
    assert ws_single.numel() < ws_split.numel(), "split workspace should be the larger one"
    hpc.topk_filtered(logits, ke, out_single, num_valid, TOP_K, cnt, ws_single)
    torch.cuda.synchronize()

    for r in range(m):
        assert set(out_split[r].tolist()) == set(out_single[r].tolist()), f"row {r} differs"


def test_kv_split_num_valid_rows_partial():
    torch.manual_seed(15)
    m, n, nvr = 17, 196608, 5
    logits = torch.randn((m, n), dtype=torch.float32, device="cuda")
    ke = torch.full((m,), n, dtype=torch.int32, device="cuda")
    out = torch.full((m, TOP_K), -1, dtype=torch.int32, device="cuda")
    num_valid = torch.tensor([nvr], dtype=torch.int32, device="cuda")
    cnt, ws = _alloc_scratch(logits, ke)
    hpc.topk_filtered(logits, ke, out, num_valid, TOP_K, cnt, ws)
    torch.cuda.synchronize()
    _validate(logits[:nvr], out[:nvr], ke.cpu()[:nvr])
    assert (out[nvr:] == -1).all(), "rows beyond num_valid_rows must be untouched"


def test_kv_split_cuda_graph_replay():
    # The split slot state must be reset for every runtime row count, including
    # zero, so one captured graph can replay the full dynamic range of its batch.
    m, n = 31, 196608
    torch.manual_seed(16)
    logits = torch.randn((m, n), dtype=torch.float32, device="cuda")
    ke = torch.full((m,), n, dtype=torch.int32, device="cuda")
    out = torch.full((m, TOP_K), -1, dtype=torch.int32, device="cuda")
    num_valid = torch.tensor([m], dtype=torch.int32, device="cuda")
    cnt, ws = _alloc_scratch(logits, ke)

    hpc.topk_filtered(logits, ke, out, num_valid, TOP_K, cnt, ws)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        hpc.topk_filtered(logits, ke, out, num_valid, TOP_K, cnt, ws)
    for nvr in (0, 1, m - 1, m):
        out.fill_(-1)
        num_valid.fill_(nvr)
        graph.replay()
        torch.cuda.synchronize()
        if nvr:
            _validate(logits[:nvr], out[:nvr], ke[:nvr].cpu())
        assert (out[nvr:] == -1).all()
        assert not cnt.any(), f"num_valid_rows={nvr} left split counters dirty"


@pytest.mark.parametrize(
    "m,n,uses_split",
    [
        (31, 196608, True),
        (32, 196608, False),
        (192, 393216, True),
        (193, 393216, False),
    ],
)
def test_kv_split_dispatch_workspace_boundaries(m, n, uses_split):
    _, recommended = hpc.topk_filtered_workspace_size(m, n)
    minimum = hpc.topk_filtered_min_workspace_size(n)
    assert (recommended > minimum) == uses_split


# ---------------------------------------------------------------------------
# Hardware-cluster path: at most 16 long rows are mapped to one eight-CTA
# cluster per row. These cases cover the DSM gather, its exact overflow escape,
# partial batches, and graph replay independently of the global KV-split path.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "m,n",
    [
        (1, 32768),
        (8, 32768),
        (16, 65536),
        (1, 73728),
        (15, 73728),
        (1, 98304),
        (15, 98304),
        (1, 196608),
        (15, 196608),
        (8, 244650),
    ],
)
def test_cluster_uniform_rows(m, n):
    torch.manual_seed(20 + m)
    logits = torch.randn((m, n), dtype=torch.float32, device="cuda")
    ke = torch.full((m,), n, dtype=torch.int32, device="cuda")
    out = torch.full((m, TOP_K), -1, dtype=torch.int32, device="cuda")
    num_valid = torch.tensor([m], dtype=torch.int32, device="cuda")
    cnt, ws = _alloc_scratch(logits, ke)
    hpc.topk_filtered(logits, ke, out, num_valid, TOP_K, cnt, ws)
    torch.cuda.synchronize()
    _validate(logits, out, ke.cpu())


def test_cluster_local_candidate_overflow_fallback():
    # One segment contributes far more exact coarse-boundary values than a
    # CTA's shared candidate capacity. The rescue must preserve the full set
    # instead of truncating that segment.
    m, n = 1, 131072
    logits = torch.full((m, n), -10.0, dtype=torch.float32, device="cuda")
    seg_len = n // 8
    logits[0, :seg_len] = 10.0
    logits[0, 13:seg_len:64] = 9.0
    ke = torch.full((m,), n, dtype=torch.int32, device="cuda")
    out = torch.full((m, TOP_K), -1, dtype=torch.int32, device="cuda")
    num_valid = torch.tensor([m], dtype=torch.int32, device="cuda")
    cnt, ws = _alloc_scratch(logits, ke)

    hpc.topk_filtered(logits, ke, out, num_valid, TOP_K, cnt, ws)
    torch.cuda.synchronize()
    _validate(logits, out, ke.cpu())


def test_cluster8_exact_candidate_overflow():
    # More than one segment-local shared candidate buffer maps to the exact
    # coarse boundary bin. The cluster route must preserve the full logical
    # candidate set through its disjoint overflow slices.
    m, n = 2, 196608
    logits = torch.full((m, n), -10.0, dtype=torch.float32, device="cuda")
    logits[:, ::17] = 10.0
    logits[:, 1::64] = 9.0
    ke = torch.tensor([n, n - 777], dtype=torch.int32, device="cuda")
    out = torch.full((m, TOP_K), -1, dtype=torch.int32, device="cuda")
    num_valid = torch.tensor([m], dtype=torch.int32, device="cuda")
    cnt, ws = _alloc_scratch(logits, ke)

    hpc.topk_filtered(logits, ke, out, num_valid, TOP_K, cnt, ws)
    torch.cuda.synchronize()
    _validate(logits, out, ke.cpu())


@pytest.mark.parametrize("padding", [1, 2])
def test_cluster8_unaligned_candidate_overflow(padding):
    # Exercise the scalar and two-wide continuations with a non-contiguous row
    # stride while forcing more candidates than one CTA can retain locally.
    m, n = 2, 131072
    backing = torch.full((m, n + padding), -10.0, dtype=torch.float32, device="cuda")
    logits = backing[:, :n]
    logits[:, ::17] = 10.0
    logits[:, 1::64] = 9.0
    ke = torch.tensor([n, n - 777], dtype=torch.int32, device="cuda")
    out = torch.full((m, TOP_K), -1, dtype=torch.int32, device="cuda")
    num_valid = torch.tensor([m], dtype=torch.int32, device="cuda")
    cnt, ws = _alloc_scratch(logits, ke)

    hpc.topk_filtered(logits, ke, out, num_valid, TOP_K, cnt, ws)
    torch.cuda.synchronize()
    _validate(logits, out, ke.cpu())


def test_cluster_num_valid_rows_partial():
    torch.manual_seed(18)
    m, n, nvr = 16, 131072, 5
    logits = torch.randn((m, n), dtype=torch.float32, device="cuda")
    ke = torch.full((m,), n, dtype=torch.int32, device="cuda")
    out = torch.full((m, TOP_K), -1, dtype=torch.int32, device="cuda")
    num_valid = torch.tensor([nvr], dtype=torch.int32, device="cuda")
    cnt, ws = _alloc_scratch(logits, ke)

    hpc.topk_filtered(logits, ke, out, num_valid, TOP_K, cnt, ws)
    torch.cuda.synchronize()
    _validate(logits[:nvr], out[:nvr], ke.cpu()[:nvr])
    assert (out[nvr:] == -1).all(), "rows beyond num_valid_rows must be untouched"


@pytest.mark.parametrize("m", [16, 17])
def test_cluster_dispatch_boundary_repeated_launches(m):
    # Sixteen rows are the last hardware-cluster shape; seventeen exercise the
    # immediately adjacent global KV-split route.
    torch.manual_seed(21)
    n = SPLIT_TEST_N
    logits = torch.randn((m, n), dtype=torch.float32, device="cuda")
    ke = torch.full((m,), n, dtype=torch.int32, device="cuda")
    out = torch.full((m, TOP_K), -1, dtype=torch.int32, device="cuda")
    num_valid = torch.tensor([m], dtype=torch.int32, device="cuda")
    cnt, ws = _alloc_scratch(logits, ke)

    for _ in range(3):
        hpc.topk_filtered(logits, ke, out, num_valid, TOP_K, cnt, ws)
    torch.cuda.synchronize()
    _validate(logits, out, ke.cpu())


@pytest.mark.parametrize("m,n", [(8, 131072), (8, 244650), (16, 131072)])
def test_cluster_cuda_graph_replay(m, n):
    torch.manual_seed(19)
    logits = torch.randn((m, n), dtype=torch.float32, device="cuda")
    ke = torch.full((m,), n, dtype=torch.int32, device="cuda")
    out = torch.full((m, TOP_K), -1, dtype=torch.int32, device="cuda")
    num_valid = torch.tensor([m], dtype=torch.int32, device="cuda")
    cnt, ws = _alloc_scratch(logits, ke)

    hpc.topk_filtered(logits, ke, out, num_valid, TOP_K, cnt, ws)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        hpc.topk_filtered(logits, ke, out, num_valid, TOP_K, cnt, ws)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    _validate(logits, out, ke.cpu())
