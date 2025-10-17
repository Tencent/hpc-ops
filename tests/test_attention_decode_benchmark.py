import math
import os
import sys
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import pytest
import torch
import time
import flashinfer

import tensorrt_llm
from tensorrt_llm._torch.attention_backend import (
    AttentionInputType,
    AttentionBackend,
    TrtllmAttention,
)
from tensorrt_llm._torch.attention_backend.interface import PredefinedAttentionMask, MLAParams
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))
import hpc

atol = 1e-2
rtol = 5e-3

from enum import Enum


# class syntax
class AttnStage(Enum):
    PREFILL = 1
    DECODE = 2


@dataclass(kw_only=True)
class Scenario:
    dtype: torch.dtype = torch.bfloat16
    kvcache_dtype: torch.dtype = torch.bfloat16
    num_layers: int
    num_heads: int = 64
    num_kv_heads: int = 8
    head_dim: int = 128
    causal: bool = True

    batch_size: int = 1
    qo_len: int = 16 * 1024
    kv_cache_len: int = 0
    stage: AttnStage = AttnStage.PREFILL

    """paged params"""
    """flash-mla requires `page_size` to be a multiple of 64"""
    page_size: int = 64
    max_num_pages_factor = 2

    @property
    def kv_len_resolved(self) -> int:
        return self.kv_len or self.qo_len

    @property
    def num_kv_groups(self) -> int:
        return self.num_heads // self.num_kv_heads

    @property
    def kv_len(self) -> int:
        return self.kv_cache_len + self.qo_len

    @property
    def num_pages(self) -> int:
        return (self.kv_len + self.page_size - 1) // self.page_size

    @property
    def is_prefill(self) -> bool:
        return self.stage == AttnStage.PREFILL

    @property
    def max_num_pages(self) -> int:
        return self.batch_size * self.kv_cache_len // self.page_size * self.max_num_pages_factor

    @property
    def nnz_qo(self):
        return self.batch_size * self.qo_len

    @property
    def nnz_kv(self):
        return self.batch_size * self.qo_len


def creat_kvcache_manager(s: Scenario) -> KVCacheManager:

    num_blocks = s.max_num_pages
    tokens_per_block = s.page_size
    num_layers = s.num_layers
    num_kv_heads = s.num_kv_heads
    head_dim = s.head_dim
    max_seq_len = s.kv_len
    batch_size = s.batch_size

    if s.kvcache_dtype == torch.float16:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.HALF
    elif s.kvcache_dtype == torch.bfloat16:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.BF16
    elif s.kvcache_dtype == torch.float8_e4m3fn:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.FP8
    else:
        raise ValueError("Invalid dtype for unit test")

    kv_cache_config = KvCacheConfig(
        max_tokens=num_blocks * tokens_per_block, enable_block_reuse=False
    )

    mapping = Mapping(world_size=1, tp_size=1, rank=0)

    cache_type = tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF

    result = KVCacheManager(
        kv_cache_config,
        cache_type,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len * 2,
        max_batch_size=batch_size,
        mapping=mapping,
        dtype=kv_cache_dtype,
    )
    return result


def construct_test_datas(s: Scenario):
    dtype = s.dtype
    num_layers = s.num_layers
    num_heads = s.num_heads
    num_kv_heads = s.num_kv_heads
    num_kv_groups = s.num_kv_groups
    head_dim = s.head_dim
    page_size = s.page_size
    kv_cache_len = s.kv_cache_len
    qo_len = s.qo_len
    kv_len = s.kv_len_resolved
    batch_size = s.batch_size
    nnz_qo = s.nnz_qo
    nnz_kv = s.nnz_kv
    causal = s.causal

    q_at_layer = torch.randn(num_layers, nnz_qo, num_heads * head_dim, device="cuda").to(dtype)

    kv_at_layer = torch.randn(num_layers, nnz_kv, 2 * num_kv_heads * head_dim, device="cuda").to(
        dtype
    )

    kvcache_manager = creat_kvcache_manager(s)

    request_ids = list(range(s.batch_size))
    token_nums = [s.kv_len] * s.batch_size
    kvcache_manager.add_dummy_requests(request_ids, token_nums)

    return q_at_layer, kv_at_layer, kvcache_manager


def trtllm_attn_func(
    q_at_layer: torch.Tensor,
    kv_at_layer: torch.Tensor,
    kv_cache_manager: KVCacheManager,
    s: Scenario,
) -> list[torch.Tensor]:

    num_cached_tokens_per_seq = torch.full((s.batch_size,), s.kv_cache_len).int()
    token_nums = torch.full((s.batch_size,), s.kv_len).int()
    seqlensq = torch.full((s.batch_size,), s.qo_len).int()

    request_ids = list(range(s.batch_size))
    kv_cache_params = KVCacheParams(
        use_cache=True, num_cached_tokens_per_seq=num_cached_tokens_per_seq
    )

    metadata = TrtllmAttention.Metadata(
        num_contexts=s.batch_size if s.is_prefill else 0,
        kv_cache_params=kv_cache_params,
        seq_lens=seqlensq,
        # seq_lens_kv=seqlensq,
        max_num_requests=s.batch_size,
        max_num_tokens=s.kv_len,
        kv_cache_manager=kv_cache_manager,
        request_ids=request_ids,
        prompt_lens=num_cached_tokens_per_seq,
    )
    metadata.prepare()
    mask = PredefinedAttentionMask.CAUSAL if s.causal else PredefinedAttentionMask.FULL

    outputs = []
    qkvs = []

    for i in range(s.num_layers):
        attention = TrtllmAttention(
            layer_idx=i,
            num_heads=s.num_heads,
            num_kv_heads=s.num_kv_heads,
            head_dim=s.head_dim,
        )

        qkv = torch.concat([q_at_layer[i], kv_at_layer[i]], dim=-1).contiguous()

        # print(f"prefill:{s.is_prefill}")
        # import pdb;pdb.set_trace()

        for _ in range(20):
            o = attention.forward(
                qkv,
                None,
                None,
                metadata,
                attention_mask=mask,
                attention_input_type=(
                    AttentionInputType.context_only
                    if s.is_prefill
                    else AttentionInputType.generation_only
                ),
            )

        torch.cuda.synchronize()

        # kv_buffer = kv_cache_manager.get_buffers(i)
        # kv_buffer = kv_buffer.clone()

        assert list(o.shape) == [s.nnz_qo, s.num_heads * s.head_dim]
        outputs.append(o)
    # kv_cache_manager.shutdown()
    return outputs


def flash_attn_func(
    q_at_layer: torch.Tensor,
    kv_at_layer: torch.Tensor,
    kv_cache_manager: KVCacheManager,
    s: Scenario,
):
    outputs = []
    if s.is_prefill:
        from flash_attn_interface import flash_attn_varlen_func

        for i in range(s.num_layers):
            q = q_at_layer[i].view(s.batch_size * s.qo_len, s.num_heads, s.head_dim).contiguous()
            k, v = (
                kv_at_layer[i]
                .view(s.batch_size * s.qo_len, 2 * s.num_kv_heads * s.head_dim)
                .split([s.num_kv_heads * s.head_dim, s.num_kv_heads * s.head_dim], -1)
            )
            k = k.reshape(-1, s.num_kv_heads, s.head_dim).contiguous()
            v = v.reshape(-1, s.num_kv_heads, s.head_dim).contiguous()

            for _ in range(20):
                o = flash_attn_varlen_func(
                    q,
                    k,
                    v,
                    cu_seqlens_q=torch.tensor(
                        [i * s.qo_len for i in range(s.batch_size + 1)],
                        dtype=torch.int32,
                        device="cuda",
                    ),
                    cu_seqlens_k=torch.tensor(
                        [i * s.qo_len for i in range(s.batch_size + 1)],
                        dtype=torch.int32,
                        device="cuda",
                    ),
                    max_seqlen_q=s.qo_len,
                    max_seqlen_k=s.qo_len,
                    softmax_scale=1.0 / (s.head_dim**0.5),
                    causal=True,
                )
            o = o.reshape(s.batch_size * s.qo_len, s.num_heads * s.head_dim)
            outputs.append(o)
    else:
        for i in range(s.num_layers):
            cache_lens = torch.tensor(
                [s.kv_cache_len] * s.batch_size, dtype=torch.int32, device="cuda"
            )
            kvcache = kv_cache_manager.get_buffer(i)
            request_ids = list(range(s.batch_size))
            block_ids = kv_cache_manager.get_block_ids_per_seq(request_ids).cuda()
            # run flashAttention
            from flash_attn.flash_attn_interface import flash_attn_with_kvcache

            for _ in range(20):
                o = flash_attn_with_kvcache(
                    q=q_at_layer[i],
                    k_cache=kvcache[:, 0, :, :].contiguous(),
                    v_cache=kvcache[:, 1, :, :].contiguous(),
                    cache_seqlens=cache_lens,
                    block_table=block_ids,
                    causal=True,
                )
            o = o.reshape(s.batch_size * s.qo_len, s.num_heads * s.head_dim)
            outputs.append(o)

    return outputs


def flashinfer_attn_func(
    q_at_layer: torch.Tensor,
    kv_at_layer: torch.Tensor,
    kv_cache_manager: KVCacheManager,
    s: Scenario,
):
    outputs = []
    kv_layout = "NHD"
    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda:0")

    if s.is_prefill:
        wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
            workspace_buffer, kv_layout
        )
        q_indptr = torch.arange(0, s.batch_size + 1, device="cuda:0", dtype=torch.int32) * s.qo_len
        kv_indptr = torch.arange(0, s.batch_size + 1, device="cuda:0", dtype=torch.int32) * s.qo_len
        wrapper.plan(
            q_indptr,
            kv_indptr,
            s.num_heads,
            s.num_kv_heads,
            s.head_dim,
            causal=True,
            q_data_type=s.dtype,
        )
        for i in range(s.num_layers):
            q = q_at_layer[i].view(s.batch_size * s.qo_len, s.num_heads, s.head_dim).contiguous()
            k, v = (
                kv_at_layer[i]
                .view(s.batch_size * s.qo_len, 2 * s.num_kv_heads * s.head_dim)
                .split([s.num_kv_heads * s.head_dim, s.num_kv_heads * s.head_dim], -1)
            )
            k = k.reshape(-1, s.num_kv_heads, s.head_dim).contiguous()
            v = v.reshape(-1, s.num_kv_heads, s.head_dim).contiguous()
            for _ in range(20):
                o = wrapper.run(q, k, v)
            o = o.reshape(s.batch_size * s.qo_len, s.num_heads * s.head_dim)
            outputs.append(o)
    else:
        wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, kv_layout)
        kv_indptr = (
            torch.arange(0, s.batch_size + 1, device="cuda:0", dtype=torch.int32) * s.num_pages
        )
        request_ids = list(range(s.batch_size))
        kv_indices = kv_cache_manager.get_block_ids_per_seq(request_ids).cuda()
        kv_last_page_len = torch.full(
            (s.batch_size,), s.kv_cache_len % s.page_size + 1, dtype=torch.int32, device="cuda:0"
        )

        wrapper.plan(
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            s.num_heads,
            s.num_kv_heads,
            s.head_dim,
            s.page_size,
            data_type=s.kvcache_dtype,
            q_data_type=s.dtype,
        )

        for i in range(s.num_layers):
            q = q_at_layer[i].view(s.batch_size * s.qo_len, s.num_heads, s.head_dim).contiguous()
            kvcache = kv_cache_manager.get_buffer(i)

            for _ in range(20):
                o = wrapper.run(q, kvcache)
            o = o.reshape(s.batch_size * s.qo_len, s.num_heads * s.head_dim)
            outputs.append(o)

    torch.cuda.synchronize()
    return outputs


def hpc_attn_func(
    q_at_layer: torch.Tensor,
    kv_at_layer: torch.Tensor,
    kv_cache_manager: KVCacheManager,
    s: Scenario,
):
    outputs = []
    if s.is_prefill:
        for i in range(s.num_layers):
            q = q_at_layer[i].view(s.batch_size * s.qo_len, s.num_heads, s.head_dim).contiguous()
            k, v = (
                kv_at_layer[i]
                .view(s.batch_size * s.qo_len, 2 * s.num_kv_heads * s.head_dim)
                .split([s.num_kv_heads * s.head_dim, s.num_kv_heads * s.head_dim], -1)
            )
            k = k.reshape(-1, s.num_kv_heads, s.head_dim).contiguous()
            v = v.reshape(-1, s.num_kv_heads, s.head_dim).contiguous()
            for _ in range(20):
                my = hpc.attention_prefill_bf16(
                    q.view(s.batch_size, s.qo_len, s.num_heads, s.head_dim),
                    k.view(s.batch_size, s.qo_len, s.num_kv_heads, s.head_dim),
                    v.view(s.batch_size, s.qo_len, s.num_kv_heads, s.head_dim),
                )
        my = my.reshape(s.batch_size * s.qo_len, s.num_heads * s.head_dim)
        outputs.append(my)
    else:
        request_ids = list(range(s.batch_size))
        block_ids = kv_cache_manager.get_block_ids_per_seq(request_ids).cuda()
        for i in range(s.num_layers):
            q = q_at_layer[i].view(s.batch_size * s.qo_len, s.num_heads, s.head_dim).contiguous()
            kvcache = kv_cache_manager.get_buffers(i)
            cache_lens = torch.tensor(
                [s.kv_cache_len] * s.batch_size, dtype=torch.int32, device="cuda"
            )
            for _ in range(20):
                my = hpc.attention_decode_bf16(q, kvcache, block_ids, cache_lens)
        my = my.reshape(s.batch_size * s.qo_len, s.num_heads * s.head_dim)
        outputs.append(my)
    torch.cuda.synchronize()
    return outputs


def allclose(
    ref: Sequence[torch.Tensor],
    impls: dict[str, Sequence[torch.Tensor]],
    *,
    layer=0,
    atol=atol,
    rtol=rtol,
):
    for name, outputs in impls.items():
        print(f"{name} output: ", float(outputs[layer].abs().mean()))
    print("ref outputs: ", float(ref[layer].abs().mean()))
    for name, outputs in impls.items():
        ref_diff = (ref[layer] - outputs[layer]).abs() / (ref[layer].abs() + 1e-5)
        print(f"{name} & ref , mean ref diff is ", ref_diff.mean())
        print(f"{name} & ref , max ref diff is ", ref_diff.max())
        print(f"{name} & ref,  number of ref diff exceeding the limit is ", (ref_diff > rtol).sum())
        max_index = torch.argmax(ref_diff)
        print(
            f"{name} & ref , the max ref diff index is {max_index}, \
              the value of {name} is {outputs[layer].reshape(-1)[max_index]}, \
              the value of ref is {ref[layer].reshape(-1)[max_index]}"
        )

        abs_diff = (ref[layer] - outputs[layer]).abs()
        print(f"{name} & ref , mean abs diff is ", abs_diff.mean())
        print(f"{name} & ref , max abs diff is ", abs_diff.max())
        print(f"{name} & ref,  number of abs diff exceeding the limit is ", (abs_diff > atol).sum())
        max_index = torch.argmax(abs_diff)
        print(
            f"{name} & ref , the max abs diff index is {max_index}, \
              the value of {name} is {outputs[layer].reshape(-1)[max_index]}, \
              the value of ref is {ref[layer].reshape(-1)[max_index]}"
        )

    for name, outputs in impls.items():
        torch.testing.assert_close(
            outputs[layer], ref[layer], atol=atol, rtol=rtol, msg=f"Allclose failed: ref<->{name}"
        ),


@pytest.mark.skip()
def test_attention_backend(s: Scenario):
    q_at_layer, kv_at_layer, kvcache_manager = construct_test_datas(s)

    # run trtllmAttention
    trtllm_outputs = trtllm_attn_func(q_at_layer, kv_at_layer, kvcache_manager, s)

    # run flashattn
    # fa_outputs = flash_attn_func(q_at_layer, kv_at_layer, kvcache_manager, s)

    # run flash infer
    # fi_outputs = flashinfer_attn_func(q_at_layer, kv_at_layer, kvcache_manager, s)

    # run hpc
    # hpc_outputs = hpc_attn_func(q_at_layer, kv_at_layer, kvcache_manager, s)

    print("-------------Result Precision-------------")
    allclose(
        # fa_outputs,
        trtllm_outputs,
        {
            # "trtllm": trtllm_outputs,
            # "flashinfer": fi_outputs,
            "hpc": hpc_outputs,
        },
    )


if __name__ == "__main__":
    # hunyuanv1 decode
    test_attention_backend(
        Scenario(
            num_layers=1,
            num_heads=4,
            num_kv_heads=1,
            head_dim=128,
            batch_size=200,
            qo_len=1,
            kv_cache_len=1024,
            page_size=32,
            stage=AttnStage.DECODE,
        )
    )
