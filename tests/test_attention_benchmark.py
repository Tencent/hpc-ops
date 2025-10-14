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


@dataclass(kw_only=True)
class Scenario:
    dtype: torch.dtype = torch.bfloat16
    kvcache_dtype: torch.dtype = torch.bfloat16
    num_layers: int
    num_heads: int = 64
    num_kv_heads: int = 8
    page_size: int = 64
    """flash-mla requires `page_size` to be a multiple of 64"""
    qo_len: int = 16 * 1024
    num_pages: int = 1024
    """setting kv_len to non-zero to test cross attention"""
    kv_len: int = 0
    causal: bool = True
    batch_size: int = 1

    # mla
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 128
    v_head_dim: int = 128

    head_dim: int = qk_rope_head_dim + qk_nope_head_dim

    @property
    def cross(self) -> bool:
        return self.kv_len != 0

    @property
    def kv_len_resolved(self) -> int:
        return self.kv_len or self.qo_len

    @property
    def num_kv_groups(self) -> int:
        return self.num_heads // self.num_kv_heads

    @property
    def kv_cache_len(self) -> int:
        return self.qo_len

    @property
    def past_kv_len(self) -> int:
        return self.kv_cache_len - self.kv_len_resolved

    @property
    def max_num_pages(self) -> int:
        return self.batch_size * self.num_pages

    @property
    def nnz_qo(self):
        return self.batch_size * self.qo_len

    @property
    def nnz_kv(self):
        return self.batch_size * self.kv_len_resolved

    def __post_init__(self) -> None:
        self.num_pages = self.qo_len // self.page_size + 1
        assert (
            self.kv_len <= self.kv_cache_len
        ), f"KV len {self.kv_len} larger than cache len {self.kv_cache_len}"
        assert self.kv_len != 0 or self.qo_len <= self.kv_cache_len, "Seq len larger than cache len"
        assert not (self.cross and self.causal), "Cross attention cannot be causal"


@dataclass(kw_only=True)
class PagedScenario(Scenario):
    num_generations: int

    @property
    def num_contexts(self) -> int:
        return self.batch_size - self.num_generations

    @property
    def num_ctx_q_tokens(self) -> int:
        return self.num_contexts * self.qo_len

    @property
    def num_ctx_kv_tokens(self) -> int:
        return self.num_contexts * self.kv_len_resolved

    @property
    def nnz_qo(self) -> int:
        return self.num_ctx_q_tokens + self.num_generations

    @property
    def nnz_kv(self) -> int:
        n = self.num_ctx_kv_tokens
        if not self.cross:
            n += self.num_generations
        return n


paged_backends = {
    TrtllmAttention: True,
}


def kv_cache_manager_from(
    Attention: type[AttentionBackend], s: Scenario, kv_cache: torch.Tensor
) -> KVCacheManager:
    paged = paged_backends[Attention]

    num_blocks = s.max_num_pages if paged else s.batch_size
    tokens_per_block = s.page_size if paged else s.kv_cache_len
    num_layers = s.num_layers
    num_kv_heads = s.num_kv_heads
    head_dim = s.head_dim
    num_heads = s.num_kv_heads
    max_seq_len = num_blocks * tokens_per_block
    batch_size = s.batch_size

    if s.kvcache_dtype == torch.float16:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.HALF
    elif s.kvcache_dtype == torch.bfloat16:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.BF16
    elif s.kvcache_dtype == torch.float8_e4m3fn:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.FP8
    else:
        raise ValueError("Invalid dtype for unit test")

    kv_cache_config = KvCacheConfig(max_tokens=num_blocks * tokens_per_block)
    kv_cache_config.enable_block_reuse = False

    mapping = Mapping(world_size=1, tp_size=1, rank=0)

    cache_type = tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY

    result = KVCacheManager(
        kv_cache_config,
        cache_type,
        num_layers=num_layers,
        num_kv_heads=1,
        head_dim=s.kv_lora_rank + s.qk_rope_head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
        mapping=mapping,
        dtype=kv_cache_dtype,
        # num_extra_kv_tokens=0
    )
    return result


def produce_outputs(
    Attention: type[AttentionBackend],
    q_at_layer: torch.Tensor,
    kv_latent: torch.Tensor,
    kv_b_proj: torch.Tensor,
    s: Scenario,
    *,
    kv_cache: torch.Tensor,
    num_cached_tokens: Callable[[int], int] | int,
    num_contexts: int | None = None,
    seq_lens: torch.Tensor,
    seq_lens_kv: Optional[torch.Tensor] = None,
    quant_config: Optional[QuantConfig] = None,
) -> list[torch.Tensor]:
    num_cached_tokens_per_seq = [
        num_cached_tokens if isinstance(num_cached_tokens, int) else num_cached_tokens(i)
        for i in range(s.batch_size)
    ]

    kv_cache_params = KVCacheParams(
        use_cache=True, num_cached_tokens_per_seq=num_cached_tokens_per_seq
    )
    kv_cache_manager = kv_cache_manager_from(Attention, s, kv_cache)
    request_ids = list(range(s.batch_size))
    seq_lens_append = seq_lens_kv if seq_lens_kv is not None else seq_lens
    token_nums = (torch.tensor(num_cached_tokens_per_seq) + seq_lens_append).tolist()
    kv_cache_manager.add_dummy_requests(request_ids, token_nums)

    metadata = Attention.Metadata(
        num_contexts=num_contexts if num_contexts is not None else s.batch_size,
        kv_cache_params=kv_cache_params,
        seq_lens=seq_lens,
        seq_lens_kv=seq_lens_kv,
        max_num_requests=s.batch_size,
        max_num_tokens=128 * 1024,
        kv_cache_manager=kv_cache_manager,
        request_ids=request_ids,
        prompt_lens=token_nums,
    )
    metadata.prepare()
    mask = PredefinedAttentionMask.CAUSAL if s.causal else PredefinedAttentionMask.FULL
    outputs = []
    qkvs = []

    mla_params = None

    for i in range(s.num_layers):
        q = q_at_layer[i]

        if kv_b_proj is not None:
            kv = kv_b_proj(kv_latent[i, ..., : s.kv_lora_rank])
            k_nope, v = kv.split(
                [s.num_heads * s.qk_nope_head_dim, s.num_heads * s.v_head_dim],
                -1,
            )
            k = torch.empty_like(q).view(-1, s.num_heads, (s.qk_nope_head_dim + s.qk_rope_head_dim))
            k[..., : s.qk_nope_head_dim] = k_nope.view(-1, s.num_heads, s.qk_nope_head_dim)
            k = k.view(-1, s.num_heads * (s.qk_nope_head_dim + s.qk_rope_head_dim))
        else:
            kv_at_layer = kv_latent
            k, v = kv_at_layer[i].split(
                [s.num_kv_heads * s.head_dim, s.num_kv_heads * s.head_dim], -1
            )

        if Attention is TrtllmAttention:
            attention = Attention(
                layer_idx=i,
                num_heads=s.num_heads,
                num_kv_heads=s.num_kv_heads,
                head_dim=s.head_dim,
                quant_config=quant_config,
                mla_params=mla_params,
            )

        qkv = torch.concat([q.contiguous(), k.contiguous(), v.contiguous()], dim=-1)

        def run_attn():
            o = attention.forward(
                qkv,
                None,
                None,
                metadata,
                attention_mask=mask,
                attention_input_type=AttentionInputType.context_only,
                latent_cache=kv_latent[i] if kv_b_proj else None,
            )

            return o, qkv

        for _ in range(20):
            o, qkv = run_attn()

        torch.cuda.synchronize()
        qkvs.append(qkv)

        kv_buffer = kv_cache_manager.get_buffers(i)
        kv_buffer = kv_buffer.clone()

        assert list(o.shape) == [s.nnz_qo, s.num_heads * s.v_head_dim]
        outputs.append(o)
    kv_cache_manager.shutdown()
    return outputs, qkvs, kv_buffer


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
def test_qwen_prefill_attention_backend(s: Scenario):
    dtype = s.dtype
    num_layers = s.num_layers
    num_heads = s.num_heads
    num_kv_heads = s.num_kv_heads
    num_kv_groups = s.num_kv_groups
    head_dim = s.head_dim
    v_head_dim = s.v_head_dim
    page_size = s.page_size
    kv_cache_len = s.kv_cache_len
    qo_len = s.qo_len
    kv_len = s.kv_len_resolved
    past_kv_len = s.past_kv_len
    batch_size = s.batch_size
    nnz_qo = s.nnz_qo
    nnz_kv = s.nnz_kv
    causal = s.causal

    q_at_layer = torch.randn(num_layers, nnz_qo, num_heads * head_dim, device="cuda").to(dtype)

    kv_b_proj = None

    kv_at_layer = torch.randn(num_layers, nnz_kv, 2 * num_kv_heads * head_dim, device="cuda").to(
        dtype
    )

    def produce(Attention: type[AttentionBackend], kv_cache: torch.Tensor):
        return produce_outputs(
            Attention,
            q_at_layer,
            kv_at_layer,
            kv_b_proj,
            s,
            kv_cache=kv_cache,
            num_cached_tokens=past_kv_len,
            seq_lens=torch.full((batch_size,), qo_len).int(),
            seq_lens_kv=torch.full((batch_size,), kv_len).int() if s.cross else None,
        )

    # run trtllmAttention
    trtllm_outputs, trtllm_qkvs, metadata = produce(TrtllmAttention, None)

    # Test reference attention
    if causal:
        causal_mask = torch.full(
            (qo_len, kv_cache_len), fill_value=torch.finfo(dtype).min, dtype=dtype, device="cuda"
        )
        cache_position = torch.arange(past_kv_len, kv_cache_len).cuda()
        bool_causal_mask = torch.arange(kv_cache_len).cuda() <= cache_position.reshape(-1, 1)
        causal_mask *= ~bool_causal_mask
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
    else:
        causal_mask = 0

    ref_outputs = []
    hpc_outputs = []
    flashinfer_outputs = []
    qkvs = trtllm_qkvs
    for i in range(num_layers):
        q, k, v = qkvs[i].split(
            [num_heads * head_dim, num_kv_heads * head_dim, num_kv_heads * v_head_dim], -1
        )

        q = q.view(batch_size * qo_len, num_heads, head_dim)
        k = k.view(batch_size * kv_cache_len, num_kv_heads, head_dim)
        v = v.view(batch_size * kv_cache_len, num_kv_heads, v_head_dim)

        # run flashAttention
        from flash_attn_interface import flash_attn_varlen_func

        for _ in range(20):
            o = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=torch.tensor(
                    [i * qo_len for i in range(batch_size + 1)], dtype=torch.int32, device="cuda"
                ),
                cu_seqlens_k=torch.tensor(
                    [i * qo_len for i in range(batch_size + 1)], dtype=torch.int32, device="cuda"
                ),
                max_seqlen_q=qo_len,
                max_seqlen_k=qo_len,
                softmax_scale=1.0 / (head_dim**0.5),
                causal=True,
                window_size=(-1, -1),
            )
        o = o.reshape(batch_size * qo_len, num_heads * v_head_dim)
        ref_outputs.append(o)

        # run flashInfer
        kv_layout = "NHD"
        workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
        wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
            workspace_buffer, kv_layout
        )
        q_indptr = torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32) * qo_len
        kv_indptr = (
            torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32) * kv_cache_len
        )
        wrapper.plan(
            q_indptr,
            kv_indptr,
            num_heads,
            num_kv_heads,
            head_dim,
            causal=True,
            pos_encoding_mode="NONE",
            logits_soft_cap=0.0,
            q_data_type=s.dtype,
        )
        for _ in range(20):
            o = wrapper.run(q, k, v)
        o = o.reshape(batch_size * qo_len, num_heads * v_head_dim)
        flashinfer_outputs.append(o)

        # run hpcAttention
        for _ in range(20):
            my = hpc.attention_prefill_bf16(
                q.contiguous().view(batch_size, qo_len, num_heads, head_dim),
                k.contiguous().view(batch_size, kv_cache_len, num_kv_heads, head_dim),
                v.contiguous().view(batch_size, kv_cache_len, num_kv_heads, v_head_dim),
            )
        torch.cuda.synchronize()
        my = my.reshape(batch_size * qo_len, num_heads * v_head_dim)
        hpc_outputs.append(my)

    print("-------------Result Precision-------------")
    allclose(
        ref_outputs,
        {
            "trtllm": trtllm_outputs,
            "flashinfer": flashinfer_outputs,
            "hpc": hpc_outputs,
        },
    )


if __name__ == "__main__":
    # hunyuanv2
    test_qwen_prefill_attention_backend(
        Scenario(
            num_layers=1,
            num_heads=4,
            num_kv_heads=1,
            qo_len=3456,
            batch_size=1,
            head_dim=128,
            v_head_dim=128,
        )
    )
