from typing import Optional, Tuple

import torch
from torch import Tensor


def deepseekv4_routing_method(
    score: Tensor,
    bias: Optional[Tensor],
    input_ids: Optional[Tensor],
    tid2eid: Optional[Tensor],
    topk: int,
    route_scale: float,
    is_hash: bool,
    out_weights: Optional[Tensor] = None,
    out_indices: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Applies deepseekv4 routing method

    Executes via a custom high-performance GPU kernel.

    Args:
      score: expert score.
          Shape: [N, kNExpert]
          Dtype: bfloat16
      bias: expert score bias, None if hash mode
          Shape: [kNExpert, ]
          Dtype: float32
      input_ids: input token indices, None if non-hash mode
          Shape: [N, ]
          Dtype: int32
      tid2eid: hash moe weight, None if non-hash mode
          Shape: [vocab_size, kTopK]
          Dtype: int32
      topk: KN_Expert, int
      route_scale: route scale, float
      is_hash: whether use hash moe, bool

    Returns:
      out_weights: topk weights
          Shape: [N, kTopK]
          Dtype: float32
      out_indices: topk indices
          Shape: [N, kTopK]
          Dtype: int32
    """
    return torch.ops.hpc.deepseekv4_routing_method(
        score, bias, input_ids, tid2eid, topk, route_scale, is_hash, out_weights, out_indices
    )
