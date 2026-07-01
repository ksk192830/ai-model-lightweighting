#!/usr/bin/env python3
"""Create or refresh canonical metadata for registered artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from kips_lightweighting.artifacts import (  # noqa: E402
    artifact_paths,
    artifact_status,
)
from kips_lightweighting.metadata import (  # noqa: E402
    read_json,
    runtime_metadata,
    sha256,
    write_json,
)
from kips_lightweighting.registry import ExperimentRegistry  # noqa: E402
from kips_lightweighting.tensorrt_build import sparse_tactic_summary  # noqa: E402


def relative(path: Path) -> str:
    return str(path.relative_to(REPOSITORY_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_ids", nargs="*")
    args = parser.parse_args()

    registry = ExperimentRegistry.load()
    selected = args.experiment_ids or list(registry.experiments)
    with (REPOSITORY_ROOT / "configs" / "baseline.yaml").open(
        encoding="utf-8"
    ) as stream:
        models = yaml.safe_load(stream)["models"]

    for experiment_id in selected:
        experiment = registry.get(experiment_id)
        for camera in experiment["cameras"]:
            paths = artifact_paths(experiment_id, camera)
            existing = paths.existing()
            comparisons = sorted(paths.directory.glob("comparison-*.json"))
            eligibility = paths.directory / "2to4-eligibility.json"
            onnx_2to4 = paths.directory / "onnx-2to4.json"
            sparse_tactics = paths.directory / "sparse-tactics.json"
            fine_tuning_preflight = (
                paths.directory / "fine-tuning-preflight.json"
            )
            structured_pruning = (
                paths.directory / "structured-pruning.json"
            )
            prototype_validation = (
                paths.directory / "prototype-validation.json"
            )
            if (
                experiment.get("sparse_tactic", False)
                and paths.build_log.is_file()
                and not sparse_tactics.is_file()
            ):
                write_json(
                    sparse_tactics,
                    sparse_tactic_summary(paths.build_log),
                )
            if (
                not existing
                and not comparisons
                and not eligibility.is_file()
                and not onnx_2to4.is_file()
                and not sparse_tactics.is_file()
                and not fine_tuning_preflight.is_file()
                and not structured_pruning.is_file()
                and not prototype_validation.is_file()
            ):
                continue
            source = REPOSITORY_ROOT / models[camera]["checkpoint"]
            artifact_source_id = experiment.get(
                "artifact_source", experiment_id
            )
            artifact_source_paths = artifact_paths(artifact_source_id, camera)
            recovery_training = (
                artifact_source_paths.directory
                / "recovery"
                / "recovery-training.json"
            )
            prototype_snapshot = (
                paths.directory / "prototype-before-recovery"
            )
            artifacts = {
                name: relative(path)
                for name, path in existing.items()
                if name != "metadata"
            }
            for name in ("calibration.cache", "onnx-build.json", "engine-build.json"):
                path = paths.directory / name
                if path.is_file():
                    artifacts[name.replace(".", "_")] = relative(path)
            for path in comparisons:
                artifacts[path.stem.replace("-", "_")] = relative(path)
            if eligibility.is_file():
                artifacts["2to4_eligibility"] = relative(eligibility)
            if onnx_2to4.is_file():
                artifacts["onnx_2to4"] = relative(onnx_2to4)
            if sparse_tactics.is_file():
                artifacts["sparse_tactics"] = relative(sparse_tactics)
            if fine_tuning_preflight.is_file():
                artifacts["fine_tuning_preflight"] = relative(
                    fine_tuning_preflight
                )
            if structured_pruning.is_file():
                artifacts["structured_pruning"] = relative(
                    structured_pruning
                )
            if prototype_validation.is_file():
                artifacts["prototype_validation"] = relative(
                    prototype_validation
                )
            if recovery_training.is_file():
                artifacts["recovery_training"] = relative(recovery_training)
            if prototype_snapshot.is_dir():
                artifacts["prototype_before_recovery"] = relative(
                    prototype_snapshot
                )
            build_details = {}
            for name in ("onnx-build.json", "engine-build.json"):
                path = paths.directory / name
                if path.is_file():
                    build_details[name.removesuffix(".json")] = read_json(path)
            metadata = {
                **runtime_metadata(),
                "experiment_id": experiment_id,
                "name": experiment["name"],
                "camera": camera,
                "source_checkpoint": relative(source),
                "source_checkpoint_sha256": sha256(source),
                "artifact_checkpoint_sha256": (
                    sha256(artifact_source_paths.checkpoint)
                    if artifact_source_paths.checkpoint.is_file()
                    else None
                ),
                "method": experiment["method"],
                "family": experiment["family"],
                "pruning": experiment.get("pruning"),
                "precision": experiment["precision"],
                "fine_tuning": experiment.get(
                    "fine_tuning", {"required": False}
                ),
                "input_shape": experiment.get(
                    "input_shape",
                    registry.defaults["export"]["input_shape"],
                ),
                "batch_size": registry.defaults["export"]["batch_size"],
                "classes": models[camera]["classes"],
                "onnx_opset": registry.defaults["export"]["onnx_opset"],
                "artifacts": artifacts,
                "build_details": build_details,
                "status": artifact_status(paths, experiment["status"]),
            }
            write_json(paths.metadata, metadata)
            print(f"metadata: {paths.metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
