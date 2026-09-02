#!/usr/bin/env python3
"""Build a TensorRT FP16 engine from an exported ONNX model."""

from __future__ import annotations

import argparse
import hashlib
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
    parser.add_argument(
        "--camera",
        required=True,
        help="Model key (e.g. front, rear, general_mission_with_crosswalk); "
        "selects the artifact subdirectory.",
    )
    parser.add_argument("--onnx-dir", type=Path, default=DEFAULT_ONNX_DIR)
    parser.add_argument("--onnx", type=Path, help="Override the input ONNX path.")
    parser.add_argument(
        "--output-name",
        help="Engine filename without extension (defaults to parking_<camera>_fp16).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workspace-mib", type=int, default=4096)
    parser.add_argument(
        "--sparse-weights",
        action="store_true",
        help="Allow TensorRT tactics for NVIDIA 2:4 structured sparsity.",
    )
    parser.add_argument(
        "--int8-qdq",
        action="store_true",
        help="Enable INT8 for ONNX models that already carry Q/DQ nodes "
        "(modelopt PTQ/QAT); FP16 remains enabled for unquantized layers.",
    )
    parser.add_argument(
        "--strongly-typed",
        action="store_true",
        help="Build a strongly typed network (required for FP8 Q/DQ ONNX); "
        "precision flags are ignored and follow the ONNX types.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose TensorRT logging, including sparse tactic selection.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def command_version(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return (result.stdout or result.stderr).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_with_python(
    onnx_path: Path,
    engine_path: Path,
    workspace_mib: int,
    enable_fp16: bool = True,
    enable_sparse: bool = False,
    enable_int8: bool = False,
    strongly_typed: bool = False,
    verbose: bool = False,
) -> str:
    try:
        import tensorrt as trt
    except ImportError as error:
        raise RuntimeError(
            "Neither trtexec nor the TensorRT Python package is available."
        ) from error

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if strongly_typed:
        network_flags |= 1 << int(
            trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED
        )
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(
            str(parser.get_error(index))
            for index in range(parser.num_errors)
        )
        raise RuntimeError(f"TensorRT could not parse {onnx_path}:\n{errors}")

    config = builder.create_builder_config()
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        workspace_mib * 1024 * 1024,
    )
    if enable_fp16 and not strongly_typed:
        config.set_flag(trt.BuilderFlag.FP16)
    if enable_int8 and not strongly_typed:
        config.set_flag(trt.BuilderFlag.INT8)
    if enable_sparse:
        config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT FP16 engine build failed.")
    engine_path.write_bytes(serialized)
    return trt.__version__


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
    output_name = args.output_name or f"parking_{args.camera}_fp16"
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
    if args.sparse_weights:
        command.append("--sparsity=enable")
    if args.int8_qdq:
        command.append("--int8")
    if args.strongly_typed:
        command.remove("--fp16")
        command.append("--stronglyTyped")
    if args.verbose:
        command.append("--verbose")
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
            enable_sparse=args.sparse_weights,
            enable_int8=args.int8_qdq,
            strongly_typed=args.strongly_typed,
            verbose=args.verbose,
        )
        trtexec_version = ""
    import torch

    def repo_relative(path: Path) -> str:
        return (
            str(path.relative_to(REPOSITORY_ROOT))
            if path.is_relative_to(REPOSITORY_ROOT)
            else str(path)
        )

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "camera": args.camera,
        "precision": (
            "fp8-qdq" if args.strongly_typed
            else "int8-qdq" if args.int8_qdq
            else "fp16"
        ),
        "source_onnx": repo_relative(onnx_path),
        "source_onnx_sha256": sha256(onnx_path),
        "engine_path": repo_relative(engine_path),
        "engine_size_bytes": engine_path.stat().st_size,
        "engine_sha256": sha256(engine_path),
        "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "compute_capability": (
            list(torch.cuda.get_device_capability(0))
            if torch.cuda.is_available()
            else None
        ),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "build_backend": build_backend,
        "tensorrt_version": tensorrt_version,
        "trtexec_version": trtexec_version,
        "workspace_mib": args.workspace_mib,
        "sparse_weights_enabled": args.sparse_weights,
        "verbose_logging": args.verbose,
        "profiling_verbosity": "DETAILED",
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
