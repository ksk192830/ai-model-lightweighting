#!/usr/bin/env python3
"""Package portable ONNX inputs needed to rebuild TensorRT engines."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from kips_lightweighting.metadata import sha256  # noqa: E402
from kips_lightweighting.registry import ExperimentRegistry  # noqa: E402
from kips_lightweighting.artifacts import artifact_paths  # noqa: E402

from build_engine_suite import (  # noqa: E402
    SUITES,
    artifact_source_id,
    suite_blockers,
    suite_experiments,
)


def shared_onnx_sources() -> dict[str, Path]:
    """Return only checksum-validated portable ONNX files from the manifest."""
    manifest_path = REPOSITORY_ROOT / "shared-models" / "manifest.yaml"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    import yaml

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    sources = {}
    for model in manifest.get("models", []):
        path = manifest_path.parent / model["file"]
        if not path.is_file():
            raise FileNotFoundError(f"Shared ONNX is missing: {path}")
        if path.stat().st_size != int(model["size_bytes"]):
            raise ValueError(f"Shared ONNX size mismatch: {path}")
        if sha256(path) != model["sha256"]:
            raise ValueError(f"Shared ONNX checksum mismatch: {path}")
        for experiment_id in model.get("experiment_ids", []):
            sources[str(experiment_id)] = path
    return sources


def link_or_copy(source: Path, target: Path) -> None:
    """Avoid duplicating multi-gigabyte ONNX payloads on the same filesystem."""
    target.unlink(missing_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=tuple(SUITES), default="final8")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "delivery" / "notebook-front",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate recovery and shared-model readiness without copying files.",
    )
    args = parser.parse_args()

    registry = ExperimentRegistry.load()
    experiments = suite_experiments(registry, args.suite)
    blockers = suite_blockers(
        registry,
        experiments,
        require_artifact_onnx=False,
        require_completed_recovery=args.suite != "stage1",
    )
    source_ids = {
        artifact_source_id(experiment_id, registry.get(experiment_id))
        for experiment_id in experiments
    }
    if args.suite == "stage1":
        sources = {
            source_id: artifact_paths(source_id, "front").onnx
            for source_id in source_ids
            if artifact_paths(source_id, "front").onnx.is_file()
        }
    else:
        sources = shared_onnx_sources()
    for source_id in sorted(source_ids):
        if source_id not in sources:
            blockers.append(
                f"{source_id}: validated ONNX is not registered in shared-models/manifest.yaml"
            )
    if blockers:
        print("bundle is not ready:")
        for blocker in blockers:
            print(f"- {blocker}")
        return 2

    if args.dry_run:
        for source_id in sorted(source_ids):
            print(f"{source_id}: {sources[source_id]}")
        return 0

    output = (
        args.output_dir
        if args.output_dir.is_absolute()
        else REPOSITORY_ROOT / args.output_dir
    )
    if output.exists() and not args.force:
        raise FileExistsError(f"Bundle already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    if args.suite == "stage1":
        for directory_name in ("configs", "scripts", "src"):
            shutil.copytree(
                REPOSITORY_ROOT / directory_name,
                output / directory_name,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        shutil.copy2(REPOSITORY_ROOT / "requirements.txt", output / "requirements.txt")
        stage1_report = REPOSITORY_ROOT / "results/stage1-static-evaluation.json"
        report_target = output / "results/stage1-static-evaluation.json"
        report_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stage1_report, report_target)

    files = []
    for source_id in sorted(source_ids):
        source = sources[source_id]
        target = (
            output
            / "artifacts"
            / "experiments"
            / source_id
            / "front"
            / "model.onnx"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        link_or_copy(source, target)
        files.append(
            {
                "experiment_id": source_id,
                "path": str(target.relative_to(output)),
                "size_bytes": target.stat().st_size,
                "sha256": sha256(target),
            }
        )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite": args.suite,
        "camera": "front",
        "experiments": list(experiments),
        "onnx_sources": files,
        "engine_plan": [
            {
                "order": order,
                "experiment_id": experiment_id,
                "source_experiment_id": artifact_source_id(
                    experiment_id, registry.get(experiment_id)
                ),
                "precision": registry.get(experiment_id)["precision"],
                "sparse_tactic": bool(
                    registry.get(experiment_id).get("sparse_tactic", False)
                ),
                "input_shape": registry.get(experiment_id).get(
                    "input_shape", registry.defaults["export"]["input_shape"]
                ),
            }
            for order, experiment_id in enumerate(experiments, start=1)
        ],
        "requires_calibration_data": any(
            registry.get(experiment_id)["precision"] == "int8"
            and registry.get(experiment_id)["stage"] == "tensorrt"
            for experiment_id in experiments
        ),
        "calibration_dir": registry.defaults["reproducibility"].get(
            "front_calibration_dir"
        ),
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"bundle: {output}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
