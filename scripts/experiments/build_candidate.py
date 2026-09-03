#!/usr/bin/env python3
"""Export ONNX or build TensorRT for a registered experiment."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from kips_lightweighting.artifacts import artifact_paths  # noqa: E402
from kips_lightweighting.metadata import (  # noqa: E402
    read_json,
    refresh_experiment_documents,
    write_json,
)
from kips_lightweighting.onnx_export import export_command  # noqa: E402
from kips_lightweighting.registry import ExperimentRegistry  # noqa: E402
from kips_lightweighting.tensorrt_build import (  # noqa: E402
    build_command,
    sparse_tactic_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_id")
    parser.add_argument("--camera", choices=("front", "rear"), required=True)
    parser.add_argument("--target", choices=("onnx", "engine"), required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def portable_command(command: list[str]) -> list[str]:
    """Make repository-local command paths runnable from the repository root."""
    recorded: list[str] = []
    for argument in command:
        path = Path(argument)
        if path.is_absolute():
            try:
                argument = path.relative_to(REPOSITORY_ROOT).as_posix()
            except ValueError:
                pass
        recorded.append(argument)
    return recorded


def main() -> int:
    args = parse_args()
    registry = ExperimentRegistry.load()
    experiment = registry.get(args.experiment_id)
    if args.camera not in experiment["cameras"]:
        raise ValueError("Camera is not registered for this experiment.")
    if experiment["status"] == "blocked":
        raise RuntimeError(f"{args.experiment_id} is blocked by its dependencies.")
    paths = artifact_paths(args.experiment_id, args.camera)
    defaults = registry.defaults

    if args.target == "onnx":
        checkpoint = paths.checkpoint
        source_experiment_id = experiment.get("artifact_source")
        if not checkpoint.is_file() and source_experiment_id:
            checkpoint = artifact_paths(source_experiment_id, args.camera).checkpoint
        if not checkpoint.is_file() and experiment["family"] in {
            "baseline",
            "input",
        }:
            with (REPOSITORY_ROOT / "configs" / "baseline.yaml").open(
                encoding="utf-8"
            ) as stream:
                checkpoint = REPOSITORY_ROOT / yaml.safe_load(stream)["models"][
                    args.camera
                ]["checkpoint"]
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        command = export_command(
            args.camera,
            checkpoint,
            paths.directory.parent,
            "model",
            defaults["export"]["onnx_opset"],
            experiment.get("input_shape", defaults["export"]["input_shape"]),
            args.force,
        )
    else:
        source_experiment_id = experiment.get(
            "artifact_source", args.experiment_id
        )
        # Quantized and resolution-specific candidates carry their own graph
        # even when their weights originate from another experiment.  Always
        # prefer that graph; using artifact_source here silently rebuilt B01 or
        # S01 and erased the candidate transformation.
        source_onnx = paths.onnx
        requires_candidate_onnx = experiment["stage"] in {
            "quantization",
            "quantization-study",
        } or (
            experiment["stage"] == "onnx"
            and source_experiment_id != args.experiment_id
        )
        if not source_onnx.is_file() and requires_candidate_onnx:
            raise FileNotFoundError(
                f"Candidate-specific ONNX not found: {source_onnx}"
            )
        if not source_onnx.is_file():
            source_onnx = artifact_paths(source_experiment_id, args.camera).onnx
        if not source_onnx.is_file() and experiment["family"] == "precision":
            source_onnx = artifact_paths("B01", args.camera).onnx
        if not source_onnx.is_file():
            raise FileNotFoundError(f"ONNX not found: {source_onnx}")
        if experiment["stage"] in {"quantization", "quantization-study"}:
            command = [
                sys.executable,
                str(
                    REPOSITORY_ROOT
                    / "scripts/lightweighting/build_tensorrt_fp16.py"
                ),
                "--camera",
                args.camera,
                "--onnx",
                str(source_onnx),
                "--output-dir",
                str(paths.directory.parent),
                "--output-name",
                "model",
                "--workspace-mib",
                str(defaults["tensorrt"]["workspace_mib"]),
            ]
            method = str(experiment["method"])
            if "fp8" in method or experiment["precision"] == "int4":
                command.append("--strongly-typed")
            else:
                command.append("--int8-qdq")
            if args.force:
                command.append("--force")
            if args.dry_run:
                command.append("--dry-run")
        else:
            command = build_command(
                args.camera,
                source_onnx,
                paths.directory.parent,
                "model",
                experiment["precision"],
                defaults["tensorrt"]["workspace_mib"],
                args.force,
                args.dry_run,
                enable_sparse=bool(experiment.get("sparse_tactic", False)),
                verbose=bool(experiment.get("sparse_tactic", False)),
                calibration_dir=(
                    REPOSITORY_ROOT
                    / defaults["reproducibility"][f"{args.camera}_calibration_dir"]
                    if experiment["precision"] == "int8"
                    else None
                ),
                calibration_count=(
                    defaults["reproducibility"]["calibration_image_count"]
                    if experiment["precision"] == "int8"
                    else None
                ),
                calibration_seed=(
                    defaults["reproducibility"]["calibration_seed"]
                    if experiment["precision"] == "int8"
                    else None
                ),
            )

    paths.directory.mkdir(parents=True, exist_ok=True)
    recorded_command = portable_command(command)
    paths.build_command.write_text(
        " ".join(recorded_command) + "\n", encoding="utf-8"
    )
    print("$ " + " ".join(recorded_command))
    if args.dry_run:
        print(f"command: {paths.build_command}")
        refresh_experiment_documents(args.experiment_id)
        return 0
    with paths.build_log.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"Build failed; see {paths.build_log}")
    generated_metadata = paths.directory / "model.json"
    if generated_metadata.is_file():
        generated_metadata.replace(
            paths.directory
            / ("onnx-build.json" if args.target == "onnx" else "engine-build.json")
        )
    generated_cache = paths.directory / "model.cache"
    if generated_cache.is_file():
        generated_cache.replace(paths.directory / "calibration.cache")
    if args.target == "engine" and experiment.get("sparse_tactic", False):
        write_json(
            paths.directory / "sparse-tactics.json",
            sparse_tactic_summary(paths.build_log),
        )
    if paths.metadata.is_file():
        metadata = read_json(paths.metadata)
        metadata.setdefault("artifacts", {})[args.target] = str(
            paths.onnx if args.target == "onnx" else paths.engine
        )
        metadata["status"] = (
            "onnx-exported" if args.target == "onnx" else "engine-built"
        )
        write_json(paths.metadata, metadata)
    print(f"log: {paths.build_log}")
    refresh_experiment_documents(args.experiment_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
