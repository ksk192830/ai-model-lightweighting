"""Compatibility helpers for RF-DETR structured architectures."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re
from typing import Any

import torch

from .static_analysis import checkpoint_state


_DECODER_LAYER = re.compile(r"^transformer\.decoder\.layers\.(\d+)\.")
_SEGMENTATION_BLOCK = re.compile(r"^segmentation_head\.blocks\.(\d+)\.")


def _contiguous_count(state: dict[str, Any], pattern: re.Pattern[str]) -> int | None:
    indices = {
        int(match.group(1))
        for name in state
        if (match := pattern.match(name))
    }
    if not indices:
        return None
    count = max(indices) + 1
    if indices != set(range(count)):
        raise ValueError(f"Structured checkpoint indexes are not contiguous: {indices}")
    return count


def load_rfdetr_checkpoint(
    checkpoint_path: Path,
    *,
    device: str = "cpu",
    num_classes: int,
    resolution: int | None = None,
) -> Any:
    """Load standard or project-defined structured RF-DETR checkpoints."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state = checkpoint_state(checkpoint)
    architecture = checkpoint.get("structured_architecture")
    ffn_architecture = (
        architecture
        if isinstance(architecture, dict)
        and architecture.get("method") == "ffn-dimension"
        else None
    )
    decoder_layers = _contiguous_count(state, _DECODER_LAYER)
    segmentation_blocks = _contiguous_count(state, _SEGMENTATION_BLOCK)
    if (
        decoder_layers is not None
        and segmentation_blocks is not None
        and decoder_layers != segmentation_blocks
    ):
        raise ValueError(
            "Decoder/segmentation structure mismatch: "
            f"{decoder_layers} != {segmentation_blocks}"
        )
    is_decoder_pruned = decoder_layers is not None and decoder_layers != 5
    if ffn_architecture is None and not is_decoder_pruned:
        from rfdetr import RFDETR

        return RFDETR.from_checkpoint(
            checkpoint_path,
            device=device,
            num_classes=num_classes,
            **({"resolution": resolution} if resolution is not None else {}),
        )

    from rfdetr.models import build_model_from_config
    from rfdetr.models._defaults import MODEL_DEFAULTS
    from rfdetr.variants import RFDETRSegLarge

    saved_config = checkpoint.get("model_config", {})
    model_fields = RFDETRSegLarge._model_config_class.model_fields
    constructor = {
        key: value
        for key, value in saved_config.items()
        if key in model_fields and key != "pretrain_weights"
    }
    constructor.update(
        {
            "pretrain_weights": None,
            "device": device,
            "num_classes": num_classes,
        }
    )
    if resolution is not None:
        constructor["resolution"] = resolution
    if is_decoder_pruned:
        constructor["dec_layers"] = decoder_layers
    wrapper = RFDETRSegLarge(**constructor)
    defaults = MODEL_DEFAULTS
    if ffn_architecture is not None:
        defaults = replace(
            defaults,
            dim_feedforward=int(ffn_architecture["dim_feedforward"]),
        )
    structured_model = build_model_from_config(
        wrapper.model_config,
        defaults=defaults,
    )
    incompatible = structured_model.load_state_dict(
        state,
        strict=False,
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Structured RF-DETR checkpoint state mismatch: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    wrapper.model.model = structured_model
    return wrapper
