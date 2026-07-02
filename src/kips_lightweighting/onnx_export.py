"""Command construction for RF-DETR ONNX export."""

from __future__ import annotations

import sys
from pathlib import Path

from .registry import REPOSITORY_ROOT


def export_command(
    camera: str,
    checkpoint: Path,
    output_dir: Path,
    output_name: str,
    opset: int,
    input_shape: list[int] | tuple[int, ...] | None = None,
    force: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts" / "lightweighting" / "export_onnx.py"),
        "--camera",
        camera,
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
        "--output-name",
        output_name,
        "--opset",
        str(opset),
    ]
    if input_shape is not None:
        if len(input_shape) != 4 or input_shape[2] != input_shape[3]:
            raise ValueError(f"Expected square NCHW input shape: {input_shape}")
        command.extend(["--resolution", str(input_shape[2])])
    if force:
        command.append("--force")
    return command
