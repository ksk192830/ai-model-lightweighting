#!/usr/bin/env python3
"""Generate the human-readable model artifact index from the registry."""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from kips_lightweighting.artifacts import (  # noqa: E402
    artifact_paths,
    artifact_status,
)
from kips_lightweighting.registry import ExperimentRegistry  # noqa: E402


def relative_link(path: Path, document: Path) -> str:
    return os.path.relpath(path, start=document.parent).replace(" ", "%20")


def load_shared_artifacts(document: Path) -> dict[tuple[str, str], list[str]]:
    """Index portable artifacts recorded in shared-models/manifest.yaml."""
    manifest_path = REPOSITORY_ROOT / "shared-models" / "manifest.yaml"
    indexed: dict[tuple[str, str], list[str]] = defaultdict(list)
    if not manifest_path.is_file():
        return indexed

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    for model in manifest.get("models", []):
        path = manifest_path.parent / model["file"]
        if not path.is_file():
            continue
        label = f"shared {path.suffix.lstrip('.').upper()}"
        link = f"[{label}]({relative_link(path, document)})"
        camera = model["camera"]
        for experiment_id in model.get("experiment_ids", []):
            indexed[(experiment_id, camera)].append(link)
    return indexed


def evaluation_path(
    experiment_id: str,
    camera: str,
    experiment: dict,
) -> Path:
    result = experiment.get("result", {})
    configured = result.get("evaluation") if isinstance(result, dict) else None
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else REPOSITORY_ROOT / path
    return (
        REPOSITORY_ROOT
        / "results"
        / "coco-evaluation"
        / f"{experiment_id}-{camera}-pth.json"
    )


def main() -> int:
    registry = ExperimentRegistry.load()
    document = REPOSITORY_ROOT / "docs" / "handoffs" / "model-artifact-index.md"
    shared_artifacts = load_shared_artifacts(document)
    lines = [
        "# 모델 Artifact 인덱스",
        "",
        "> 이 문서는 `scripts/experiments/update_index.py`가 생성한다.",
        "> 직접 편집하지 말고 `configs/experiments/registry.yaml`을 수정한다.",
        "",
        "| ID | 이름 | Camera | 방식 | Precision | 상태 | 파일 |",
        "|---|---|---|---|---|---|---|",
    ]
    for experiment_id, experiment in registry.experiments.items():
        for camera in experiment["cameras"]:
            paths = artifact_paths(experiment_id, camera)
            links = list(shared_artifacts.get((experiment_id, camera), []))
            for label, path in (
                ("PTH", paths.checkpoint),
                ("ONNX", paths.onnx),
                ("engine", paths.engine),
                ("metadata", paths.metadata),
                ("analysis", paths.static_analysis),
            ):
                if path.is_file():
                    links.append(
                        f"[{label}]({relative_link(path, document)})"
                    )
            for path in sorted(paths.directory.glob("comparison-*.json")):
                links.append(
                    f"[comparison]({relative_link(path, document)})"
                )
            sparsity = paths.directory / "sparsity.json"
            if sparsity.is_file():
                links.append(
                    f"[sparsity]({relative_link(sparsity, document)})"
                )
            eligibility = paths.directory / "2to4-eligibility.json"
            if eligibility.is_file():
                links.append(
                    f"[2:4 survey]({relative_link(eligibility, document)})"
                )
            onnx_2to4 = paths.directory / "onnx-2to4.json"
            if onnx_2to4.is_file():
                links.append(
                    f"[ONNX 2:4]({relative_link(onnx_2to4, document)})"
                )
            sparse_tactics = paths.directory / "sparse-tactics.json"
            if sparse_tactics.is_file():
                links.append(
                    f"[sparse tactics]({relative_link(sparse_tactics, document)})"
                )
            fine_tuning_preflight = (
                paths.directory / "fine-tuning-preflight.json"
            )
            if fine_tuning_preflight.is_file():
                links.append(
                    "[fine-tuning preflight]"
                    f"({relative_link(fine_tuning_preflight, document)})"
                )
            structured_pruning = (
                paths.directory / "structured-pruning.json"
            )
            if structured_pruning.is_file():
                links.append(
                    "[structured pruning]"
                    f"({relative_link(structured_pruning, document)})"
                )
            prototype_validation = (
                paths.directory / "prototype-validation.json"
            )
            if prototype_validation.is_file():
                links.append(
                    "[prototype validation]"
                    f"({relative_link(prototype_validation, document)})"
                )
            onnx_equivalence = paths.directory / "onnx-equivalence.json"
            if onnx_equivalence.is_file():
                links.append(
                    "[ONNX equivalence]"
                    f"({relative_link(onnx_equivalence, document)})"
                )
            coco_evaluation = evaluation_path(
                experiment_id,
                camera,
                experiment,
            )
            if coco_evaluation.is_file():
                links.append(
                    "[COCO evaluation]"
                    f"({relative_link(coco_evaluation, document)})"
                )
            source_experiment_id = experiment.get(
                "artifact_source", experiment_id
            )
            source_paths = artifact_paths(source_experiment_id, camera)
            recovery_training = (
                source_paths.directory / "recovery" / "recovery-training.json"
            )
            if recovery_training.is_file():
                links.append(
                    "[recovery training]"
                    f"({relative_link(recovery_training, document)})"
                )
            prototype_snapshot = (
                paths.directory / "prototype-before-recovery"
            )
            if prototype_snapshot.is_dir():
                links.append(
                    "[prototype snapshot]"
                    f"({relative_link(prototype_snapshot, document)})"
                )
            lines.append(
                "| "
                + " | ".join(
                    [
                        experiment_id,
                        experiment["name"],
                        camera,
                        experiment["method"],
                        experiment["precision"],
                        (
                            experiment["status"]
                            if experiment["status"]
                            == "static-analysis-rejected"
                            else artifact_status(paths)
                        ),
                        ", ".join(links) or "-",
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## 저장 규칙",
            "",
            "```text",
            "artifacts/experiments/<experiment-id>/<camera>/",
            "```",
            "",
            "평가 대상과 진행 순서는 "
            "[신규 Front 경량화 2차 계획]"
            "(../guides/front-lightweighting-round-2.md)을 따른다.",
        ]
    )
    document.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"index: {document}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
