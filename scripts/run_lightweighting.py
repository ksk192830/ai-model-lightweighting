#!/usr/bin/env python3
"""Run one or more model-lightweighting pipelines from a single command."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPOSITORY_ROOT / "scripts"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "results" / "lightweighting"
DEFAULT_CALIBRATION_DIR = REPOSITORY_ROOT / "data" / "calibration"
SUPPORTED_METHODS = ("onnx", "tensorrt-fp16", "tensorrt-int8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run lightweighting pipelines for front/rear RF-DETR checkpoints."
        )
    )
    parser.add_argument(
        "--camera",
        choices=("all", "front", "rear"),
        default="all",
        help="Camera models to process (default: all).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("all", *SUPPORTED_METHODS),
        default=["all"],
        help="Lightweighting methods to run (default: all).",
    )
    parser.add_argument(
        "--workspace-mib",
        type=int,
        default=4096,
        help="TensorRT builder workspace in MiB.",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=DEFAULT_CALIBRATION_DIR,
        help="Directory containing front/rear INT8 calibration images.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild outputs that already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for pipeline execution reports.",
    )
    return parser.parse_args()


def run_command(command: list[str], dry_run: bool) -> dict:
    printable = " ".join(command)
    print(f"\n$ {printable}")
    started = time.perf_counter()
    if dry_run:
        return {
            "command": command,
            "status": "dry-run",
            "elapsed_seconds": 0.0,
        }

    result = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False)
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {printable}"
        )
    return {
        "command": command,
        "status": "completed",
        "elapsed_seconds": elapsed,
    }


def command_output(command: list[str]) -> str:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    return (result.stdout or result.stderr).strip()


def selected_methods(values: list[str]) -> list[str]:
    if "all" in values:
        return list(SUPPORTED_METHODS)
    return list(dict.fromkeys(values))


def preflight(methods: list[str], dry_run: bool) -> dict:
    tools = {
        "python": sys.executable,
        "nvidia_smi": shutil.which("nvidia-smi"),
        "trtexec": shutil.which("trtexec"),
    }
    if not dry_run:
        required_tools = set()
        if {"tensorrt-fp16", "tensorrt-int8"}.intersection(methods):
            required_tools.add("nvidia_smi")
        if "tensorrt-fp16" in methods:
            required_tools.add("trtexec")
        missing = [name for name in sorted(required_tools) if tools[name] is None]
        if missing:
            raise RuntimeError(
                "Missing NVIDIA/TensorRT tools: "
                + ", ".join(missing)
                + ". Run this pipeline on the target NVIDIA machine."
            )

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "tools": tools,
        "nvidia_smi": (
            command_output([tools["nvidia_smi"], "-L"])
            if tools["nvidia_smi"]
            else ""
        ),
        "trtexec_version": (
            command_output([tools["trtexec"], "--version"])
            if tools["trtexec"]
            else ""
        ),
    }


def onnx_command(camera: str, force: bool) -> list[str]:
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "export_onnx.py"),
        "--camera",
        camera,
    ]
    if force:
        command.append("--force")
    return command


def tensorrt_fp16_command(
    camera: str,
    workspace_mib: int,
    force: bool,
    dry_run: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "build_tensorrt_fp16.py"),
        "--camera",
        camera,
        "--workspace-mib",
        str(workspace_mib),
    ]
    if force:
        command.append("--force")
    if dry_run:
        command.append("--dry-run")
    return command


def tensorrt_int8_command(
    camera: str,
    calibration_dir: Path,
    workspace_mib: int,
    force: bool,
    dry_run: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "build_tensorrt_int8.py"),
        "--camera",
        camera,
        "--calibration-dir",
        str(calibration_dir),
        "--workspace-mib",
        str(workspace_mib),
    ]
    if force:
        command.append("--force")
    if dry_run:
        command.append("--dry-run")
    return command


def main() -> int:
    args = parse_args()
    if args.workspace_mib < 1:
        raise ValueError("--workspace-mib must be at least one.")

    methods = selected_methods(args.methods)
    cameras = ("front", "rear") if args.camera == "all" else (args.camera,)
    calibration_dir = (
        args.calibration_dir
        if args.calibration_dir.is_absolute()
        else REPOSITORY_ROOT / args.calibration_dir
    )
    environment = preflight(methods, args.dry_run)
    executions: list[dict] = []

    print(f"cameras: {', '.join(cameras)}")
    print(f"methods: {', '.join(methods)}")

    for camera in cameras:
        print(f"\n=== {camera} ===")

        # TensorRT depends on ONNX, so export it automatically even if the user
        # selected only the TensorRT method.
        if (
            "onnx" in methods
            or "tensorrt-fp16" in methods
            or "tensorrt-int8" in methods
        ):
            execution = run_command(
                onnx_command(camera, args.force),
                args.dry_run,
            )
            executions.append(
                {"camera": camera, "method": "onnx", **execution}
            )

        if "tensorrt-fp16" in methods:
            execution = run_command(
                tensorrt_fp16_command(
                    camera,
                    args.workspace_mib,
                    args.force,
                    args.dry_run,
                ),
                args.dry_run,
            )
            executions.append(
                {"camera": camera, "method": "tensorrt-fp16", **execution}
            )

        if "tensorrt-int8" in methods:
            execution = run_command(
                tensorrt_int8_command(
                    camera,
                    calibration_dir,
                    args.workspace_mib,
                    args.force,
                    args.dry_run,
                ),
                args.dry_run,
            )
            executions.append(
                {"camera": camera, "method": "tensorrt-int8", **execution}
            )

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cameras": list(cameras),
        "methods": methods,
        "force": args.force,
        "dry_run": args.dry_run,
        "workspace_mib": args.workspace_mib,
        "calibration_dir": str(calibration_dir),
        "environment": environment,
        "executions": executions,
    }

    if not args.dry_run:
        output_dir = (
            args.output_dir
            if args.output_dir.is_absolute()
            else REPOSITORY_ROOT / args.output_dir
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / (
            f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nreport: {report_path}")

    print("\nLightweighting pipeline completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
