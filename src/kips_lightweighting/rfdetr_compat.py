"""Compatibility helpers for RF-DETR structured architectures."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from .static_analysis import checkpoint_state


def load_rfdetr_checkpoint(
    checkpoint_path: Path,
    *,
    device: str = "cpu",
    num_classes: int,
) -> Any:
    """Load standard or project-defined structured RF-DETR checkpoints."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    architecture = checkpoint.get("structured_architecture")
    if not isinstance(architecture, dict) or architecture.get("method") != "ffn-dimension":
        from rfdetr import RFDETR

        return RFDETR.from_checkpoint(
            checkpoint_path,
            device=device,
            num_classes=num_classes,
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
    wrapper = RFDETRSegLarge(**constructor)
    defaults = replace(
        MODEL_DEFAULTS,
        dim_feedforward=int(architecture["dim_feedforward"]),
    )
    structured_model = build_model_from_config(
        wrapper.model_config,
        defaults=defaults,
    )
    incompatible = structured_model.load_state_dict(
        checkpoint_state(checkpoint),
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
