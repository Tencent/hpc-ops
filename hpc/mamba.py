import torch
from torch import Tensor


def selective_state_update(ssm_state: Tensor, zxbcdt: Tensor, AD: Tensor,
                           dt_bias: Tensor, indices: Tensor,
                           num_group: int) -> Tensor:
  """Selectively updates the state of a structured state-space model (SSM).

  This operator performs an in-place update of the SSM state for chosen groups /
  channels.  It is typically used inside the recurrence loop of a selective SSM
  layer (e.g. Mamba, S4D) to incorporate the latest input while respecting the
  time-step dependent gating mechanism.

  Args:
    ssm_state: containing the current hidden state of the SSM. **Modified in place**.
      Shape: [num_max_batch, num_head, head_dim, state_dim]
      Dtype: float32
    zxbcdt: packed z, x, B, C and dt
      Shape: [num_batch, num_head * head_dim * 2 + num_group * state_dim + num_group * state_dim * 2i + num_heads]
      Dtype: bfloat16
    AD: coefficient of A and D, with [a0, d0, a1, d1, a2, d2, ..., an-1, dn-1]
      Shape: [num_heads]
      Dtype: float32
    dt_bias: learned bias
      Shape: [num_heads]
      Dtype: float32
    indices: An int tensor indicate where the ssm stored the data, value must in [0, num_max_batch)
      Shape: [num_batch]
      Dtype: int32
    num_group: An int specifying the total number of groups

  Returns:
    output:
      Shape: [batch, num_heads * head_dim]
      Dtype: bfloat16

      Caution: ssm_state is **Modified in place**

  Raises:
    RuntimeError: If the shapes or dtypes do not satisfy the constraints above.
  """

  return torch.ops.hpc.selective_state_update(ssm_state, zxbcdt, AD, dt_bias,
                                              indices, num_group)


def causal_conv1d_update(zxbcdt: torch.Tensor, conv_state: torch.Tensor,
                         weight: torch.Tensor, bias: torch.Tensor,
                         indices: torch.Tensor, d_inner: int, num_head: int):
  """
    Args:
      zxbcdt: Input tensor.
        shape: [num_tokens, d_inner + conv_dim + n_heads]
        Dtype: bfloat16
      conv_state: Current state tensor of the state-space model (modified in-place).
        shape: [max_batch, state_len, conv_dim],  where state_len = d_conv - 1
        Dtype: bfloat16
      weight: convolution weight.
        shape: [d_conv, conv_dim], where d_conv = 4
        Dtype: bfloat16
      bias: convolution bias.
        shape: [conv_dim]
        Dtype: bfloat16
      indices: Specifies which state elements to update.
        Shape: [K] (K = number of elements to update, K <= max_batch)
        Dtype: int32 (must be contiguous)
    """
  return torch.ops.hpc.causal_conv1d_update(zxbcdt, conv_state, weight, bias,
                                            indices, d_inner, num_head)
