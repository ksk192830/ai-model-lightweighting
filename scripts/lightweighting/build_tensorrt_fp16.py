#!/usr/bin/env python3
"""Build a TensorRT FP16 engine from an exported ONNX model."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ONNX_DIR = REPOSITORY_ROOT / "artifacts" / "onnx"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "artifacts" / "tensorrt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a TensorRT FP16 engine.")
    parser.add_argument("--camera", choices=("front", "rear"), required=True)
    parser.add_argument("--onnx-dir", type=Path, default=DEFAULT_ONNX_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workspace-mib", type=int, default=4096)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def command_version(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return (result.stdout or result.stderr).strip()


def build_with_python(
    onnx_path: Path,
    engine_path: Path,
    workspace_mib: int,
    enable_fp16: bool = True,
) -> str:
    try:
        import tensorrt as trt
    except ImportError as error:
        raise RuntimeError(
            "Neither trtexec nor the TensorRT Python package is available."
        ) from error

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(
            str(parser.get_error(index))
            for index in range(parser.num_errors)
        )
        raise RuntimeError(f"TensorRT could not parse {onnx_path}:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        workspace_mib * 1024 * 1024,
    )
    if enable_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT FP16 engine build failed.")
    engine_path.write_bytes(serialized)
    return trt.__version__


def main() -> int:
    args = parse_args()
    if args.workspace_mib < 1:
        raise ValueError("--workspace-mib must be at least one.")

    onnx_path = resolve_path(args.onnx_dir) / args.camera / f"parking_{args.camera}.onnx"
    output_dir = resolve_path(args.output_dir) / args.camera
    engine_path = output_dir / f"parking_{args.camera}_fp16.engine"
    metadata_path = engine_path.with_suffix(".json")

    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    if engine_path.exists() and not args.force:
        print(f"TensorRT engine already exists: {engine_path}")
        print("Use --force to build it again.")
        return 0

    trtexec = shutil.which("trtexec")

    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        trtexec or "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16",
        f"--memPoolSize=workspace:{args.workspace_mib}",
        "--useCudaGraph",
        "--useSpinWait",
    ]
    build_backend = "trtexec" if trtexec else "tensorrt-python"
    if trtexec:
        print(" ".join(command))
    else:
        print(
            f"TensorRT Python FP16 build: {onnx_path} -> {engine_path}"
        )
    if args.dry_run:
        return 0

    if trtexec:
        subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
        tensorrt_version = ""
        trtexec_version = command_version([trtexec, "--version"])
    else:
        tensorrt_version = build_with_python(
            onnx_path,
            engine_path,
            args.workspace_mib,
        )
        trtexec_version = ""
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "camera": args.camera,
        "precision": "fp16",
        "source_onnx": str(onnx_path.relative_to(REPOSITORY_ROOT)),
        "engine_path": str(engine_path.relative_to(REPOSITORY_ROOT)),
        "engine_size_bytes": engine_path.stat().st_size,
        "platform": platform.platform(),
        "build_backend": build_backend,
        "tensorrt_version": tensorrt_version,
        "trtexec_version": trtexec_version,
        "workspace_mib": args.workspace_mib,
        "build_command": command,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"TensorRT FP16 engine: {engine_path}")
    print(f"metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
