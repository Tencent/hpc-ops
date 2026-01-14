import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import pytest
import torch

import hpc
from utils import allclose


def compressor_torch(
    kv,
    score,
    cu_seqlens,
    cu_compressed_seqlens,
    kv_state,
    score_state,
    state_index,
    start_pos,
    ape,
    ratio,
    overlap,
    head_dim,
    is_prefill,
):
    """
    compressor_torch

    kv: multiple batches of kv, shape of [total_seqlen,head_dim] or [total_seqlen,head_dim*2]
    score: multiple batches of score, shape of [total_seqlen,head_dim]
    cu_seqlens: cumsum of seqlens of all batches, shape of [num_batch+1], first elem is 0, the last element is total_seqlen
    cu_compressed_seqlens: cumsum of compressed seqlens of all batches, shape of [num_batch+1], first elem is 0, the last element is total_compressed_seqlen
    kv_state: the whole kv_state of all batches, shape of [max_batch,ratio,head_dim], or if overlaped, shape of [max_batch,ratio*2,head_dim*2]
    score_state: the whole score_state of all batches, shape of [max_batch,ratio,head_dim], or if overlaped, shape of [max_batch,ratio*2,head_dim*2]
    state_index: index of used kv_state or score_state for each batch, shape of [num_batch]
    start_pos: start position of each batch, also the full context kv length of each batch, shape of [num_batch]
    ape: bias, shape of [ratio,head_dim] or [ratio,head_dim*2]
    ratio: compression ratio, 4 or 128
    overlap: whether to use overlapped kv_state
    head_dim: head dimension
    """

    # general assert
    assert ratio == 4 or ratio == 128
    assert kv.shape == score.shape
    assert kv_state.shape == score_state.shape
    assert kv.shape[-1] == kv_state.shape[-1]
    assert cu_seqlens.shape[0] == start_pos.shape[0] + 1
    assert start_pos.shape[0] == state_index.shape[0]
    assert kv.shape[0] == cu_seqlens[-1].item()
    assert (is_prefill and torch.all(start_pos == 0).item()) or (
        not is_prefill
    )  # this will decide how many output tokens

    if overlap:
        assert kv.shape[-1] == head_dim * 2
        assert kv_state.shape[1] == ratio * 2
        assert ratio == 4
    else:
        assert kv.shape[-1] == head_dim
        assert kv_state.shape[1] == ratio

    def compress_one_token(one_kv, one_score):
        assert one_kv.shape[1] == head_dim
        assert one_score.shape[1] == head_dim
        assert one_kv.shape[0] == one_score.shape[0]
        assert one_kv.shape[0] == 4 or one_kv.shape[0] == 8 or one_kv.shape[0] == 128

        result = (one_kv * (one_score.softmax(dim=0))).sum(dim=0, keepdim=True)

        assert result.shape[0] == 1 and result.shape[1] == head_dim

        return result

    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    num_batch = seqlens.shape[0]

    if is_prefill:
        # prepare output tensor
        compressed_seqlens = seqlens // ratio
        cu_compressed_seqlens_prefill = torch.cumsum(compressed_seqlens, dim=0)
        cu_compressed_seqlens_prefill = torch.cat(
            [
                torch.zeros(
                    1,
                    dtype=cu_compressed_seqlens_prefill.dtype,
                    device=cu_compressed_seqlens_prefill.device,
                ),
                cu_compressed_seqlens_prefill,
            ],
            dim=0,
        )
        compressed_kv = torch.zeros(
            cu_compressed_seqlens_prefill[-1].item(), head_dim, dtype=kv.dtype, device=kv.device
        )
    else:
        compressed_kv = torch.zeros(num_batch, head_dim, dtype=kv.dtype, device=kv.device)

    if overlap:
        double_ape = torch.cat([ape, ape], dim=0)

    for ibatch in range(num_batch):
        cur_state_index = state_index[ibatch]

        # slice current batch
        cur_kv = kv[cu_seqlens[ibatch] : cu_seqlens[ibatch + 1], :]  # slice kv of this batch
        cur_score = score[
            cu_seqlens[ibatch] : cu_seqlens[ibatch + 1], :
        ]  # slice score of this batch
        cur_seqlen = seqlens[ibatch]
        start_pos_this_batch = start_pos[ibatch]

        if start_pos_this_batch == 0 and is_prefill:  # prefill
            remainder = cur_seqlen % ratio
            cutoff = cur_seqlen - remainder
            if overlap:
                if cutoff >= ratio:  # if long enough, update i-1 kv_state
                    kv_state[cur_state_index, :ratio, :] = cur_kv[cutoff - ratio : cutoff, :]
                    score_state[cur_state_index, :ratio, :] = (
                        cur_score[cutoff - ratio : cutoff, :] + ape[:, :]
                    )
                if remainder > 0:  # if there are some token remain, update the rest of i kv_state
                    kv_state[cur_state_index, ratio : ratio + remainder, :] = cur_kv[
                        cutoff : cutoff + remainder, :
                    ]
                    score_state[cur_state_index, ratio : ratio + remainder, :] = (
                        cur_score[cutoff : cutoff + remainder, :] + ape[:remainder, :]
                    )
            else:  # no overlap
                if remainder > 0:  # if there are some token remain, update the rest of i kv_state
                    kv_state[cur_state_index, :remainder, :] = cur_kv[
                        cutoff : cutoff + remainder, :
                    ]
                    score_state[cur_state_index, :remainder, :] = (
                        cur_score[cutoff : cutoff + remainder, :] + ape[:remainder, :]
                    )

            if overlap:
                kv_for_compress = cur_kv[:cutoff, :].reshape(
                    -1, ratio, head_dim * 2
                )  # [cur_compress_len,ratio,head_dim*2]
                score_for_compress = (
                    cur_score[:cutoff, :].reshape(-1, ratio, head_dim * 2) + ape[:, :]
                )  # broadcast add ape
            else:
                kv_for_compress = cur_kv[:cutoff, :].reshape(
                    -1, ratio, head_dim
                )  # [cur_compress_len,ratio,head_dim]
                score_for_compress = (
                    cur_score[:cutoff, :].reshape(-1, ratio, head_dim) + ape[:, :]
                )  # broadcast add ape

            cur_compress_len = cutoff // ratio
            cur_compress_start = cu_compressed_seqlens_prefill[ibatch]
            for i in range(cur_compress_len):
                if overlap:
                    if i == 0:
                        upper_partial_kv = torch.zeros(
                            ratio, head_dim, dtype=kv.dtype, device=kv.device
                        )
                        upper_partial_score = torch.full(
                            (ratio, head_dim), float("-inf"), dtype=score.dtype, device=score.device
                        )
                    else:
                        upper_partial_kv = kv_for_compress[i - 1, :, :head_dim]  # [ratio,head_dim]
                        upper_partial_score = score_for_compress[
                            i - 1, :, :head_dim
                        ]  # [ratio,head_dim]
                    tmp_one_token_kv_for_compress = torch.cat(
                        [upper_partial_kv, kv_for_compress[i, :, head_dim:]], dim=0
                    )  # [2*ratio,head_dim]
                    tmp_one_token_score_for_compress = torch.cat(
                        [upper_partial_score, score_for_compress[i, :, head_dim:]], dim=0
                    )  # [2*ratio,head_dim]
                    tmp_one_token_compress_result = compress_one_token(
                        tmp_one_token_kv_for_compress, tmp_one_token_score_for_compress
                    )  # [1,head_dim]
                else:
                    tmp_one_token_compress_result = compress_one_token(
                        kv_for_compress[i, :, :], score_for_compress[i, :, :]
                    )  # [1,head_dim]

                compressed_kv[cur_compress_start + i, :] = tmp_one_token_compress_result[0, :]

        else:  # decode
            assert (
                cur_seqlen < ratio
            ), "cur_seqlen must be less than ratio, other wise here should compress more than once"
            for itoken in range(cur_seqlen):

                start_pos_this_batch = start_pos[ibatch] + itoken

                should_compress = (
                    start_pos_this_batch + 1
                ) % ratio == 0  # +1 because pos is index, we need length
                cur_write_pos = start_pos_this_batch % ratio

                if overlap:
                    # update i kv_state
                    kv_state[cur_state_index, ratio + cur_write_pos, :] = cur_kv[itoken, :]
                    score_state[cur_state_index, ratio + cur_write_pos, :] = (
                        cur_score[itoken, :] + ape[cur_write_pos, :]
                    )

                    if should_compress:
                        upper_partial_kv = kv_state[
                            cur_state_index, :ratio, :head_dim
                        ]  # [ratio,head_dim]
                        upper_partial_score = score_state[
                            cur_state_index, :ratio, :head_dim
                        ]  # [ratio,head_dim]
                        tmp_one_token_kv_for_compress = torch.cat(
                            [upper_partial_kv, kv_state[cur_state_index, ratio:, head_dim:]], dim=0
                        )
                        tmp_one_token_score_for_compress = torch.cat(
                            [upper_partial_score, score_state[cur_state_index, ratio:, head_dim:]],
                            dim=0,
                        )
                        tmp_one_token_compress_result = compress_one_token(
                            tmp_one_token_kv_for_compress, tmp_one_token_score_for_compress
                        )  # [1,head_dim]
                        compressed_kv[cu_compressed_seqlens[ibatch], :] = (
                            tmp_one_token_compress_result[0, :]
                        )

                        # move up kv states and score states
                        kv_state[cur_state_index, :ratio, :head_dim] = kv_state[
                            cur_state_index, ratio:, :head_dim
                        ]
                        score_state[cur_state_index, :ratio, :head_dim] = score_state[
                            cur_state_index, ratio:, :head_dim
                        ]
                else:
                    kv_state[cur_state_index, cur_write_pos, :] = cur_kv[itoken, :]
                    score_state[cur_state_index, cur_write_pos, :] = (
                        cur_score[itoken, :] + ape[cur_write_pos, :]
                    )

                    if should_compress:
                        tmp_one_token_compress_result = compress_one_token(
                            kv_state[cur_state_index, :, :], score_state[cur_state_index, :, :]
                        )  # [1,head_dim]
                        compressed_kv[cu_compressed_seqlens[ibatch], :] = (
                            tmp_one_token_compress_result[0, :]
                        )

    return compressed_kv


@pytest.mark.parametrize("batch", [128])
@pytest.mark.parametrize("dim", [512])
@pytest.mark.parametrize("ratio", [128])
def test_c128_kv_compressor_decode(batch, dim, ratio):
    kv_state = torch.randn((batch, ratio, dim), dtype=torch.float, device="cuda")
    score_state = torch.randn((batch, ratio, dim), dtype=torch.float, device="cuda")
    state_index = torch.arange(batch, dtype=torch.int, device="cuda")
    ape = torch.randn((ratio, dim), dtype=torch.float, device="cuda")

    seqlens = torch.ones((batch,), dtype=torch.int, device="cuda")
    cu_seqlens = torch.cumsum(seqlens, dim=0)
    cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.int, device="cuda"), cu_seqlens], dim=0).to(
        torch.int32
    )

    start_pos = torch.randint(low=0, high=127, size=(batch,), dtype=torch.int, device="cuda")
    kv_state_for_torch = kv_state.clone()
    score_state_for_torch = score_state.clone()

    kv = torch.randn((batch, dim), dtype=torch.float, device="cuda")
    score = torch.randn((batch, dim), dtype=torch.float, device="cuda")

    compressed_seqlens = ((start_pos + 1) % ratio == 0).to(torch.int32)
    cu_compressed_seqlens = torch.cumsum(compressed_seqlens, dim=0)
    cu_compressed_seqlens = torch.cat(
        [torch.zeros(1, dtype=torch.int, device="cuda"), cu_compressed_seqlens], dim=0
    ).to(torch.int32)

    compressed_kv = hpc.kv_compressor_decode(
        kv,
        score,
        ape,
        kv_state,
        score_state,
        state_index,
        start_pos,
        cu_compressed_seqlens,
        dim,
        ratio,
        False,
    )

    compressed_torch = compressor_torch(
        kv,
        score,
        cu_seqlens,
        cu_compressed_seqlens,
        kv_state_for_torch,
        score_state_for_torch,
        state_index,
        start_pos,
        ape,
        ratio,
        overlap=False,
        head_dim=dim,
        is_prefill=False,
    )
    assert allclose(kv_state, kv_state_for_torch)
    assert allclose(score_state, score_state_for_torch)
    valid_len = cu_compressed_seqlens[-1]
    if valid_len:
        assert allclose(compressed_kv, compressed_torch, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("batch", [128])
@pytest.mark.parametrize("dim", [128])
@pytest.mark.parametrize("ratio", [4])
def test_c4_kv_compressor_decode(batch, dim, ratio):
    kv_state = torch.randn((batch, 2 * ratio, 2 * dim), dtype=torch.float, device="cuda")
    score_state = torch.randn((batch, 2 * ratio, 2 * dim), dtype=torch.float, device="cuda")
    state_index = torch.arange(batch, dtype=torch.int, device="cuda")
    ape = torch.randn((ratio, 2 * dim), dtype=torch.float, device="cuda")

    seqlens = torch.ones((batch,), dtype=torch.int, device="cuda")
    cu_seqlens = torch.cumsum(seqlens, dim=0)
    cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.int, device="cuda"), cu_seqlens], dim=0).to(
        torch.int32
    )

    start_pos = torch.randint(low=0, high=4, size=(batch,), dtype=torch.int, device="cuda")
    kv_state_for_torch = kv_state.clone()
    score_state_for_torch = score_state.clone()

    kv = torch.randn((batch, 2 * dim), dtype=torch.float, device="cuda")
    score = torch.randn((batch, 2 * dim), dtype=torch.float, device="cuda")

    compressed_seqlens = ((start_pos + 1) % ratio == 0).to(torch.int32)
    cu_compressed_seqlens = torch.cumsum(compressed_seqlens, dim=0)
    cu_compressed_seqlens = torch.cat(
        [torch.zeros(1, dtype=torch.int, device="cuda"), cu_compressed_seqlens], dim=0
    ).to(torch.int32)
    compressed_kv = hpc.kv_compressor_decode(
        kv,
        score,
        ape,
        kv_state,
        score_state,
        state_index,
        start_pos,
        cu_compressed_seqlens,
        dim,
        ratio,
        True,
    )

    compressed_torch = compressor_torch(
        kv,
        score,
        cu_seqlens,
        cu_compressed_seqlens,
        kv_state_for_torch,
        score_state_for_torch,
        state_index,
        start_pos,
        ape,
        ratio,
        overlap=True,
        head_dim=dim,
        is_prefill=False,
    )
    valid_len = cu_compressed_seqlens[-1]
    if valid_len:
        assert allclose(compressed_kv, compressed_torch, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("batch", [128])
@pytest.mark.parametrize("dim", [512])
@pytest.mark.parametrize("ratio", [128])
@pytest.mark.parametrize("mtp", [1])
def test_c128_kv_compressor_decode_mtp(batch, dim, ratio, mtp):
    assert mtp == 1

    kv_state = torch.randn((batch, ratio, dim), dtype=torch.float, device="cuda")
    score_state = torch.randn((batch, ratio, dim), dtype=torch.float, device="cuda")
    state_index = torch.arange(batch, dtype=torch.int, device="cuda")
    ape = torch.randn((ratio, dim), dtype=torch.float, device="cuda")

    seqlens = torch.ones((batch,), dtype=torch.int, device="cuda") + mtp
    cu_seqlens = torch.cumsum(seqlens, dim=0)
    cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.int, device="cuda"), cu_seqlens], dim=0).to(
        torch.int32
    )

    start_pos = torch.randint(low=0, high=127, size=(batch,), dtype=torch.int, device="cuda")
    kv_state_for_torch = kv_state.clone()
    score_state_for_torch = score_state.clone()

    kv = torch.randn((batch * 2, dim), dtype=torch.float, device="cuda")
    score = torch.randn((batch * 2, dim), dtype=torch.float, device="cuda")

    cond1 = (start_pos + 1) % ratio == 0
    cond2 = (start_pos + 2) % ratio == 0
    compressed_seqlens = (cond1 | cond2).to(torch.int32)
    cu_compressed_seqlens = torch.cumsum(compressed_seqlens, dim=0)
    cu_compressed_seqlens = torch.cat(
        [torch.zeros(1, dtype=torch.int, device="cuda"), cu_compressed_seqlens], dim=0
    ).to(torch.int32)
    compressed_kv = hpc.kv_compressor_decode(
        kv,
        score,
        ape,
        kv_state,
        score_state,
        state_index,
        start_pos,
        cu_compressed_seqlens,
        dim,
        ratio,
        False,
    )

    compressed_torch = compressor_torch(
        kv,
        score,
        cu_seqlens,
        cu_compressed_seqlens,
        kv_state_for_torch,
        score_state_for_torch,
        state_index,
        start_pos,
        ape,
        ratio,
        overlap=False,
        head_dim=dim,
        is_prefill=False,
    )
    assert allclose(kv_state, kv_state_for_torch)
    assert allclose(score_state, score_state_for_torch)
    valid_len = cu_compressed_seqlens[-1]
    if valid_len:
        assert allclose(compressed_kv, compressed_torch, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("batch", [4])
@pytest.mark.parametrize("dim", [128])
@pytest.mark.parametrize("ratio", [4])
@pytest.mark.parametrize("mtp", [1])
def test_c4_kv_compressor_decode_mtp(batch, dim, ratio, mtp):
    assert mtp == 1

    kv_state = torch.randn((batch, 2 * ratio, 2 * dim), dtype=torch.float, device="cuda")
    score_state = torch.randn((batch, 2 * ratio, 2 * dim), dtype=torch.float, device="cuda")
    state_index = torch.arange(batch, dtype=torch.int, device="cuda")
    ape = torch.randn((ratio, 2 * dim), dtype=torch.float, device="cuda")

    seqlens = torch.ones((batch,), dtype=torch.int, device="cuda") + mtp
    cu_seqlens = torch.cumsum(seqlens, dim=0)
    cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.int, device="cuda"), cu_seqlens], dim=0).to(
        torch.int32
    )

    start_pos = torch.randint(low=0, high=4, size=(batch,), dtype=torch.int, device="cuda")
    start_pos.fill_(0)
    kv_state_for_torch = kv_state.clone()
    score_state_for_torch = score_state.clone()

    kv = torch.randn((batch * 2, 2 * dim), dtype=torch.float, device="cuda")
    score = torch.randn((batch * 2, 2 * dim), dtype=torch.float, device="cuda")

    cond1 = (start_pos + 1) % ratio == 0
    cond2 = (start_pos + 2) % ratio == 0
    compressed_seqlens = (cond1 | cond2).to(torch.int32)
    cu_compressed_seqlens = torch.cumsum(compressed_seqlens, dim=0)
    cu_compressed_seqlens = torch.cat(
        [torch.zeros(1, dtype=torch.int, device="cuda"), cu_compressed_seqlens], dim=0
    ).to(torch.int32)
    compressed_kv = hpc.kv_compressor_decode(
        kv,
        score,
        ape,
        kv_state,
        score_state,
        state_index,
        start_pos,
        cu_compressed_seqlens,
        dim,
        ratio,
        True,
    )

    compressed_torch = compressor_torch(
        kv,
        score,
        cu_seqlens,
        cu_compressed_seqlens,
        kv_state_for_torch,
        score_state_for_torch,
        state_index,
        start_pos,
        ape,
        ratio,
        overlap=True,
        head_dim=dim,
        is_prefill=False,
    )
    valid_len = cu_compressed_seqlens[-1]
    if valid_len:
        assert allclose(compressed_kv, compressed_torch, atol=1e-5, rtol=1e-5)
