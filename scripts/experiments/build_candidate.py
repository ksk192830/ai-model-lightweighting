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
from kips_lightweighting.metadata import read_json, write_json  # noqa: E402
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
        if not checkpoint.is_file() and experiment["family"] == "baseline":
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
            args.force,
        )
    else:
        source_experiment_id = experiment.get(
            "artifact_source", args.experiment_id
        )
        source_onnx = artifact_paths(source_experiment_id, args.camera).onnx
        if not source_onnx.is_file() and experiment["family"] == "precision":
            source_onnx = artifact_paths("B01", args.camera).onnx
        if not source_onnx.is_file():
            raise FileNotFoundError(f"ONNX not found: {source_onnx}")
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
        )

    paths.directory.mkdir(parents=True, exist_ok=True)
    paths.build_command.write_text(" ".join(command) + "\n", encoding="utf-8")
    print("$ " + " ".join(command))
    if args.dry_run:
        print(f"command: {paths.build_command}")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
