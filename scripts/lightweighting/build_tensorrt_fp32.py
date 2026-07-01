#!/usr/bin/env python3
"""Build a TensorRT FP32 engine from an exported ONNX model."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from build_tensorrt_fp16 import (
    REPOSITORY_ROOT,
    build_with_python,
    resolve_path,
)


DEFAULT_ONNX_DIR = REPOSITORY_ROOT / "artifacts" / "onnx"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "artifacts" / "tensorrt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a TensorRT FP32 engine.")
    parser.add_argument("--camera", choices=("front", "rear"), required=True)
    parser.add_argument("--onnx-dir", type=Path, default=DEFAULT_ONNX_DIR)
    parser.add_argument("--onnx", type=Path, help="Override the input ONNX path.")
    parser.add_argument(
        "--output-name",
        help="Engine filename without extension (defaults to parking_<camera>_fp32).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workspace-mib", type=int, default=4096)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workspace_mib < 1:
        raise ValueError("--workspace-mib must be at least one.")

    onnx_path = (
        resolve_path(args.onnx)
        if args.onnx
        else resolve_path(args.onnx_dir)
        / args.camera
        / f"parking_{args.camera}.onnx"
    )
    output_dir = resolve_path(args.output_dir) / args.camera
    output_name = args.output_name or f"parking_{args.camera}_fp32"
    if Path(output_name).name != output_name:
        raise ValueError("--output-name must be a filename, not a path.")
    engine_path = output_dir / f"{output_name.removesuffix('.engine')}.engine"
    metadata_path = engine_path.with_suffix(".json")

    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    if engine_path.exists() and not args.force:
        print(f"TensorRT engine already exists: {engine_path}")
        print("Use --force to build it again.")
        return 0

    print(f"TensorRT Python FP32 build: {onnx_path} -> {engine_path}")
    if args.dry_run:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    tensorrt_version = build_with_python(
        onnx_path,
        engine_path,
        args.workspace_mib,
        enable_fp16=False,
    )
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "camera": args.camera,
        "precision": "fp32",
        "source_onnx": str(onnx_path.relative_to(REPOSITORY_ROOT)),
        "engine_path": str(engine_path.relative_to(REPOSITORY_ROOT)),
        "engine_size_bytes": engine_path.stat().st_size,
        "platform": platform.platform(),
        "build_backend": "tensorrt-python",
        "tensorrt_version": tensorrt_version,
        "workspace_mib": args.workspace_mib,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"TensorRT FP32 engine: {engine_path}")
    print(f"metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
