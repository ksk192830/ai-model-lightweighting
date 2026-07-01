"""Structured pruning helpers for RF-DETR architectures."""

from __future__ import annotations

import re
from typing import Any

import torch

from ..static_analysis import checkpoint_state


_DECODER_LAYER = re.compile(r"^transformer\.decoder\.layers\.(\d+)\.")
_SEGMENTATION_BLOCK = re.compile(r"^segmentation_head\.blocks\.(\d+)\.")
_FFN_LINEAR1 = re.compile(
    r"^transformer\.decoder\.layers\.(\d+)\.linear1\.weight$"
)


def prune_decoder_layers(
    checkpoint: dict[str, Any],
    remove_layers: int,
    model_config: dict[str, Any],
) -> dict[str, Any]:
    """Remove trailing decoder layers and their segmentation blocks.

    RF-DETR segmentation applies one depthwise-convolution block per decoder
    output. Removing both trailing structures keeps their lengths aligned and
    reduces the exported dense graph.
    """
    if remove_layers < 1:
        raise ValueError("remove_layers must be at least 1")
    state = checkpoint_state(checkpoint)
    decoder_indices = {
        int(match.group(1))
        for name in state
        if (match := _DECODER_LAYER.match(name))
    }
    if not decoder_indices:
        raise ValueError("Checkpoint does not contain decoder layers.")
    original_layers = max(decoder_indices) + 1
    if decoder_indices != set(range(original_layers)):
        raise ValueError(
            f"Decoder layer indexes are not contiguous: {sorted(decoder_indices)}"
        )
    remaining_layers = original_layers - remove_layers
    if remaining_layers < 1:
        raise ValueError("Structured pruning must retain at least one decoder layer.")

    remove_indices = set(range(remaining_layers, original_layers))

    def should_remove(name: str) -> bool:
        decoder_match = _DECODER_LAYER.match(name)
        if decoder_match and int(decoder_match.group(1)) in remove_indices:
            return True
        segmentation_match = _SEGMENTATION_BLOCK.match(name)
        return bool(
            segmentation_match
            and int(segmentation_match.group(1)) in remove_indices
        )

    before_parameters = sum(
        value.numel() for value in state.values() if isinstance(value, torch.Tensor)
    )
    removed = {
        name: value
        for name, value in state.items()
        if should_remove(name)
    }
    if not removed:
        raise ValueError("No decoder parameters matched the removal request.")
    for name in removed:
        del state[name]

    lightning_state = checkpoint.get("state_dict")
    removed_lightning = []
    if isinstance(lightning_state, dict):
        for name in list(lightning_state):
            unprefixed = name.removeprefix("model.")
            if name.startswith("model.") and should_remove(unprefixed):
                removed_lightning.append(name)
                del lightning_state[name]

    updated_model_config = dict(model_config)
    updated_model_config["dec_layers"] = remaining_layers
    checkpoint["model_config"] = updated_model_config
    checkpoint_args = checkpoint.get("args")
    if isinstance(checkpoint_args, dict):
        checkpoint_args["dec_layers"] = remaining_layers

    after_parameters = sum(
        value.numel() for value in state.values() if isinstance(value, torch.Tensor)
    )
    removed_parameters = before_parameters - after_parameters
    decoder_removed_parameters = sum(
        value.numel()
        for name, value in removed.items()
        if _DECODER_LAYER.match(name) and isinstance(value, torch.Tensor)
    )
    segmentation_removed_parameters = sum(
        value.numel()
        for name, value in removed.items()
        if _SEGMENTATION_BLOCK.match(name)
        and isinstance(value, torch.Tensor)
    )
    return {
        "method": "decoder-layer",
        "original_decoder_layers": original_layers,
        "remaining_decoder_layers": remaining_layers,
        "removed_decoder_layers": sorted(remove_indices),
        "removed_state_entries": len(removed),
        "removed_lightning_state_entries": len(removed_lightning),
        "parameters_before": before_parameters,
        "parameters_after": after_parameters,
        "removed_parameters": removed_parameters,
        "parameter_reduction": removed_parameters / before_parameters,
        "decoder_removed_parameters": decoder_removed_parameters,
        "segmentation_removed_parameters": segmentation_removed_parameters,
    }


def prune_ffn_dimensions(
    checkpoint: dict[str, Any],
    reduction: float,
    alignment: int = 32,
) -> dict[str, Any]:
    """Reduce each decoder FFN using magnitude-ranked hidden units.

    A hidden unit is kept or removed as one structural group across the
    corresponding linear1 output/bias and linear2 input.  The resulting width
    is rounded to the nearest alignment multiple for deployment-friendly
    matrix shapes.
    """
    if not 0.0 < reduction < 1.0:
        raise ValueError("reduction must be between zero and one")
    if alignment < 1:
        raise ValueError("alignment must be positive")
    state = checkpoint_state(checkpoint)
    layer_indices = sorted(
        int(match.group(1))
        for name in state
        if (match := _FFN_LINEAR1.match(name))
    )
    if not layer_indices:
        raise ValueError("Checkpoint does not contain decoder FFN weights.")

    original_widths: set[int] = set()
    kept_indices: dict[str, list[int]] = {}
    parameters_before = sum(
        value.numel() for value in state.values() if isinstance(value, torch.Tensor)
    )
    removed_parameters = 0

    for index in layer_indices:
        prefix = f"transformer.decoder.layers.{index}"
        linear1_weight = state[f"{prefix}.linear1.weight"]
        linear1_bias = state[f"{prefix}.linear1.bias"]
        linear2_weight = state[f"{prefix}.linear2.weight"]
        original_width = linear1_weight.shape[0]
        if linear1_bias.shape[0] != original_width:
            raise ValueError(f"linear1 bias width mismatch in decoder layer {index}")
        if linear2_weight.shape[1] != original_width:
            raise ValueError(f"linear2 input width mismatch in decoder layer {index}")
        original_widths.add(original_width)
        target = round((original_width * (1.0 - reduction)) / alignment) * alignment
        target = max(alignment, min(original_width - alignment, target))

        scores = (
            linear1_weight.float().pow(2).sum(dim=1)
            + linear1_bias.float().pow(2)
            + linear2_weight.float().pow(2).sum(dim=0)
        )
        keep = torch.topk(scores, target, largest=True, sorted=False).indices
        keep = torch.sort(keep).values
        kept_indices[str(index)] = keep.tolist()

        old_count = (
            linear1_weight.numel()
            + linear1_bias.numel()
            + linear2_weight.numel()
        )
        state[f"{prefix}.linear1.weight"] = linear1_weight.index_select(0, keep)
        state[f"{prefix}.linear1.bias"] = linear1_bias.index_select(0, keep)
        state[f"{prefix}.linear2.weight"] = linear2_weight.index_select(1, keep)
        new_count = (
            state[f"{prefix}.linear1.weight"].numel()
            + state[f"{prefix}.linear1.bias"].numel()
            + state[f"{prefix}.linear2.weight"].numel()
        )
        removed_parameters += old_count - new_count

    if len(original_widths) != 1:
        raise ValueError(f"Decoder FFN widths differ: {sorted(original_widths)}")
    original_width = original_widths.pop()
    remaining_width = len(next(iter(kept_indices.values())))
    checkpoint["structured_architecture"] = {
        "method": "ffn-dimension",
        "dim_feedforward": remaining_width,
        "original_dim_feedforward": original_width,
        "alignment": alignment,
        "kept_indices": kept_indices,
    }
    parameters_after = parameters_before - removed_parameters
    return {
        "method": "ffn-dimension",
        "requested_reduction": reduction,
        "actual_ffn_reduction": 1.0 - remaining_width / original_width,
        "original_dim_feedforward": original_width,
        "remaining_dim_feedforward": remaining_width,
        "alignment": alignment,
        "decoder_layers": layer_indices,
        "parameters_before": parameters_before,
        "parameters_after": parameters_after,
        "removed_parameters": removed_parameters,
        "parameter_reduction": removed_parameters / parameters_before,
    }
