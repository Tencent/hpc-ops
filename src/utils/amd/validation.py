# Copyright (C) 2026 Tencent.
# Copyright 2026, The FlagOS Contributors.

import torch


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check_tensor(tensor, reference, dtype, name, contiguous=True):
    require(
        tensor.is_cuda and tensor.device == reference.device,
        f"{name} must be on the same GPU as the input",
    )
    require(tensor.dtype == dtype, f"{name} has an unsupported dtype")
    if contiguous:
        require(tensor.is_contiguous(), f"{name} must be contiguous")


def check_bf16(tensor, reference, name):
    check_tensor(tensor, reference, torch.bfloat16, name)
