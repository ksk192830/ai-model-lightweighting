#!/usr/bin/env python3
"""Audit registered artifacts, metadata references, and recorded hashes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from kips_lightweighting.artifacts import artifact_paths, artifact_status  # noqa: E402
from kips_lightweighting.metadata import (  # noqa: E402
    read_json,
    runtime_metadata,
    sha256,
    write_json,
)
from kips_lightweighting.registry import ExperimentRegistry  # noqa: E402


DEPLOYED_STATUSES = {
    "engine-built",
    "ready-for-evaluation",
    "selected-for-rear",
    "delivered",
}
ONNX_STATUSES = DEPLOYED_STATUSES | {"onnx-exported"}
DECISION_STATUSES = {"static-analysis-rejected"}


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def add_issue(
    issues: list[dict[str, str]],
    severity: str,
    code: str,
    message: str,
) -> None:
    issues.append({"severity": severity, "code": code, "message": message})


def resolved_artifact_status(paths: Any, registry_status: str) -> str:
    """Resolve artifact maturity without overriding terminal decisions."""
    if registry_status in DECISION_STATUSES:
        return registry_status
    return artifact_status(paths, registry_status)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_ids", nargs="*")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts/experiments/audit-report.json",
    )
    args = parser.parse_args()

    registry = ExperimentRegistry.load()
    selected = args.experiment_ids or list(registry.experiments)
    required_metadata = set(registry.schema["metadata_required_fields"])
    results: list[dict[str, Any]] = []

    for experiment_id in selected:
        experiment = registry.get(experiment_id)
        for camera in experiment["cameras"]:
            paths = artifact_paths(experiment_id, camera)
            issues: list[dict[str, str]] = []
            inventory = {}
            for label, path in paths.existing().items():
                inventory[label] = {
                    "path": relative(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }

            status = experiment["status"]
            resolved_status = resolved_artifact_status(paths, status)
            source_id = experiment.get("artifact_source")
            if source_id is None and experiment["family"] == "precision":
                source_id = "B01"
            source_paths = artifact_paths(source_id or experiment_id, camera)

            if status in DEPLOYED_STATUSES and not paths.engine.is_file():
                add_issue(issues, "error", "missing-engine", str(paths.engine))
            if status in ONNX_STATUSES and not (
                paths.onnx.is_file() or source_paths.onnx.is_file()
            ):
                add_issue(issues, "error", "missing-onnx", str(source_paths.onnx))
            if experiment["family"] in {
                "unstructured",
                "semi-structured",
                "structured",
            } and not source_paths.checkpoint.is_file():
                add_issue(
                    issues,
                    "error",
                    "missing-checkpoint",
                    str(source_paths.checkpoint),
                )
            if status not in {"planned", "blocked"} and not paths.metadata.is_file():
                add_issue(issues, "error", "missing-metadata", str(paths.metadata))

            if status == "static-analysis-rejected":
                result = experiment.get("result")
                if not isinstance(result, dict) or result.get("decision") != "rejected":
                    add_issue(
                        issues,
                        "error",
                        "rejection-decision",
                        "result.decision must be 'rejected'",
                    )
                reason = result.get("reason") if isinstance(result, dict) else None
                if not isinstance(reason, str) or not reason.strip():
                    add_issue(
                        issues,
                        "error",
                        "rejection-reason",
                        "result.reason must be a non-empty string",
                    )

            if paths.metadata.is_file():
                metadata = read_json(paths.metadata)
                missing_fields = sorted(required_metadata.difference(metadata))
                if missing_fields:
                    add_issue(
                        issues,
                        "error",
                        "metadata-fields",
                        ", ".join(missing_fields),
                    )
                if metadata.get("experiment_id") != experiment_id:
                    add_issue(
                        issues,
                        "error",
                        "metadata-experiment",
                        str(metadata.get("experiment_id")),
                    )
                if metadata.get("camera") != camera:
                    add_issue(
                        issues,
                        "error",
                        "metadata-camera",
                        str(metadata.get("camera")),
                    )
                expected_status = resolved_status
                if metadata.get("status") != expected_status:
                    add_issue(
                        issues,
                        "error",
                        "metadata-status",
                        f"{metadata.get('status')} != {expected_status}",
                    )
                for label, value in metadata.get("artifacts", {}).items():
                    if not isinstance(value, str):
                        continue
                    referenced = Path(value)
                    if not referenced.is_absolute():
                        referenced = REPOSITORY_ROOT / referenced
                    if not referenced.exists():
                        add_issue(
                            issues,
                            "error",
                            "broken-artifact-reference",
                            f"{label}: {value}",
                        )
                source_value = metadata.get("source_checkpoint")
                if isinstance(source_value, str):
                    source = Path(source_value)
                    if not source.is_absolute():
                        source = REPOSITORY_ROOT / source
                    if not source.is_file():
                        add_issue(
                            issues,
                            "error",
                            "missing-source-checkpoint",
                            str(source),
                        )
                    elif metadata.get("source_checkpoint_sha256") != sha256(source):
                        add_issue(
                            issues,
                            "error",
                            "source-checkpoint-hash",
                            str(source),
                        )
                recorded_artifact_hash = metadata.get(
                    "artifact_checkpoint_sha256"
                )
                if source_paths.checkpoint.is_file() and (
                    recorded_artifact_hash != sha256(source_paths.checkpoint)
                ):
                    add_issue(
                        issues,
                        "error",
                        "artifact-checkpoint-hash",
                        str(source_paths.checkpoint),
                    )

            results.append(
                {
                    "experiment_id": experiment_id,
                    "camera": camera,
                    "registry_status": status,
                    "resolved_artifact_status": resolved_status,
                    "inventory": inventory,
                    "issues": issues,
                    "valid": not any(
                        issue["severity"] == "error" for issue in issues
                    ),
                }
            )

    errors = sum(
        issue["severity"] == "error"
        for result in results
        for issue in result["issues"]
    )
    report = {
        **runtime_metadata(),
        "experiments_checked": len(results),
        "errors": errors,
        "valid": errors == 0,
        "results": results,
    }
    output = args.output if args.output.is_absolute() else REPOSITORY_ROOT / args.output
    write_json(output, report)
    print(f"audit report: {output}")
    print(f"experiments checked: {len(results)}, errors: {errors}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
