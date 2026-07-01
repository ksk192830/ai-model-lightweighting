"""Global magnitude unstructured pruning."""

from __future__ import annotations

from typing import Any

import torch

from ..static_analysis import is_prunable


def prune_global_magnitude(state: dict[str, Any], ratio: float) -> int:
    if not 0 < ratio < 1:
        raise ValueError("Pruning ratio must be greater than 0 and less than 1.")
    tensors = [
        tensor for name, tensor in state.items() if is_prunable(name, tensor)
    ]
    total = sum(tensor.numel() for tensor in tensors)
    prune_count = round(total * ratio)
    magnitudes = torch.cat(
        [tensor.detach().abs().reshape(-1).cpu() for tensor in tensors]
    )
    selected = torch.topk(
        magnitudes,
        k=prune_count,
        largest=False,
        sorted=False,
    ).indices
    offset = 0
    for tensor in tensors:
        count = tensor.numel()
        local = selected[(selected >= offset) & (selected < offset + count)] - offset
        if local.numel():
            tensor.reshape(-1)[local.to(tensor.device)] = 0
        offset += count
    return prune_count
