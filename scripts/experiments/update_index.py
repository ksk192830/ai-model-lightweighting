#!/usr/bin/env python3
"""Generate the human-readable model artifact index from the registry."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from kips_lightweighting.artifacts import (  # noqa: E402
    artifact_paths,
    artifact_status,
)
from kips_lightweighting.registry import ExperimentRegistry  # noqa: E402


def relative_link(path: Path, document: Path) -> str:
    return str(path.relative_to(REPOSITORY_ROOT)).replace(" ", "%20")


def main() -> int:
    registry = ExperimentRegistry.load()
    document = REPOSITORY_ROOT / "docs" / "model-artifact-index.md"
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
            links = []
            for label, path in (
                ("PTH", paths.checkpoint),
                ("ONNX", paths.onnx),
                ("engine", paths.engine),
                ("metadata", paths.metadata),
                ("analysis", paths.static_analysis),
            ):
                if path.is_file():
                    links.append(
                        f"[{label}](../{relative_link(path, document)})"
                    )
            for path in sorted(paths.directory.glob("comparison-*.json")):
                links.append(
                    f"[comparison](../{relative_link(path, document)})"
                )
            sparsity = paths.directory / "sparsity.json"
            if sparsity.is_file():
                links.append(
                    f"[sparsity](../{relative_link(sparsity, document)})"
                )
            eligibility = paths.directory / "2to4-eligibility.json"
            if eligibility.is_file():
                links.append(
                    f"[2:4 survey](../{relative_link(eligibility, document)})"
                )
            onnx_2to4 = paths.directory / "onnx-2to4.json"
            if onnx_2to4.is_file():
                links.append(
                    f"[ONNX 2:4](../{relative_link(onnx_2to4, document)})"
                )
            sparse_tactics = paths.directory / "sparse-tactics.json"
            if sparse_tactics.is_file():
                links.append(
                    f"[sparse tactics](../{relative_link(sparse_tactics, document)})"
                )
            fine_tuning_preflight = (
                paths.directory / "fine-tuning-preflight.json"
            )
            if fine_tuning_preflight.is_file():
                links.append(
                    "[fine-tuning preflight]"
                    f"(../{relative_link(fine_tuning_preflight, document)})"
                )
            structured_pruning = (
                paths.directory / "structured-pruning.json"
            )
            if structured_pruning.is_file():
                links.append(
                    "[structured pruning]"
                    f"(../{relative_link(structured_pruning, document)})"
                )
            prototype_validation = (
                paths.directory / "prototype-validation.json"
            )
            if prototype_validation.is_file():
                links.append(
                    "[prototype validation]"
                    f"(../{relative_link(prototype_validation, document)})"
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
                    f"(../{relative_link(recovery_training, document)})"
                )
            prototype_snapshot = (
                paths.directory / "prototype-before-recovery"
            )
            if prototype_snapshot.is_dir():
                links.append(
                    "[prototype snapshot]"
                    f"(../{relative_link(prototype_snapshot, document)})"
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
                        artifact_status(paths, experiment["status"]),
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
            "[경량화 모델 실험 진행 계획](experiment-workflow.md)을 따른다.",
        ]
    )
    document.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"index: {document}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
