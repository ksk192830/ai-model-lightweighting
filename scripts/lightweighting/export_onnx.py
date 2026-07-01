#!/usr/bin/env python3
"""Export an RF-DETR segmentation checkpoint to ONNX for TensorRT."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(REPOSITORY_ROOT / "results" / ".cache" / "matplotlib"),
)

import onnx  # noqa: E402
import yaml  # noqa: E402
from rfdetr import RFDETR  # noqa: E402


DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "baseline.yaml"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "artifacts" / "onnx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export RF-DETR to ONNX.")
    parser.add_argument("--camera", choices=("front", "rear"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Override the checkpoint configured for the selected camera.",
    )
    parser.add_argument(
        "--output-name",
        help="Output filename without extension (defaults to parking_<camera>).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def load_model_config(config_path: Path, camera: str) -> dict:
    with resolve_path(config_path).open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    try:
        return config["models"][camera]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Missing model configuration for '{camera}'.") from error


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least one.")

    model_config = load_model_config(args.config, args.camera)
    checkpoint = resolve_path(
        args.checkpoint
        if args.checkpoint is not None
        else Path(model_config["checkpoint"])
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    class_names = list(model_config["classes"])
    output_dir = resolve_path(args.output_dir) / args.camera
    output_name = args.output_name or f"parking_{args.camera}"
    if Path(output_name).name != output_name:
        raise ValueError("--output-name must be a filename, not a path.")
    output_path = output_dir / f"{output_name.removesuffix('.onnx')}.onnx"
    metadata_path = output_path.with_suffix(".json")

    if output_path.exists() and not args.force:
        print(f"ONNX already exists: {output_path}")
        print("Use --force to export it again.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    model = RFDETR.from_checkpoint(
        checkpoint,
        device="cpu",
        num_classes=len(class_names),
    )
    exported_path = Path(
        model.export(
            output_dir=str(output_dir),
            format="onnx",
            opset_version=args.opset,
            batch_size=args.batch_size,
            dynamic_batch=False,
            verbose=False,
            notes={
                "camera": args.camera,
                "source_checkpoint": str(checkpoint),
                "purpose": "ONNX/TensorRT conversion",
            },
        )
    )

    if exported_path.resolve() != output_path.resolve():
        shutil.move(str(exported_path), output_path)

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    input_shapes = {
        tensor.name: [
            dimension.dim_value if dimension.dim_value else dimension.dim_param
            for dimension in tensor.type.tensor_type.shape.dim
        ]
        for tensor in onnx_model.graph.input
    }
    output_names = [tensor.name for tensor in onnx_model.graph.output]
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "camera": args.camera,
        "source_checkpoint": str(checkpoint),
        "onnx_path": str(output_path.relative_to(REPOSITORY_ROOT)),
        "onnx_size_bytes": output_path.stat().st_size,
        "opset": args.opset,
        "batch_size": args.batch_size,
        "inputs": input_shapes,
        "outputs": output_names,
        "classes": class_names,
        "validated_by_onnx_checker": True,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"ONNX: {output_path}")
    print(f"size: {output_path.stat().st_size / 1024 / 1024:.2f} MiB")
    print(f"metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
