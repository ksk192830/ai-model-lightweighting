"""Canonical experiment-centric artifact paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .registry import REPOSITORY_ROOT


DEFAULT_ARTIFACT_ROOT = REPOSITORY_ROOT / "artifacts" / "experiments"


@dataclass(frozen=True)
class ArtifactPaths:
    directory: Path
    checkpoint: Path
    onnx: Path
    engine: Path
    metadata: Path
    sparsity: Path
    static_analysis: Path
    build_command: Path
    build_log: Path

    def existing(self) -> dict[str, Path]:
        return {
            name: path
            for name, path in self.__dict__.items()
            if name != "directory" and path.is_file()
        }


def artifact_paths(
    experiment_id: str,
    camera: str,
    root: Path = DEFAULT_ARTIFACT_ROOT,
) -> ArtifactPaths:
    directory = root / experiment_id / camera
    return ArtifactPaths(
        directory=directory,
        checkpoint=directory / "model.pth",
        onnx=directory / "model.onnx",
        engine=directory / "model.engine",
        metadata=directory / "metadata.json",
        sparsity=directory / "sparsity.json",
        static_analysis=directory / "static-analysis.json",
        build_command=directory / "build-command.txt",
        build_log=directory / "build.log",
    )


def artifact_status(paths: ArtifactPaths, fallback: str = "planned") -> str:
    if paths.engine.is_file():
        return "engine-built"
    if paths.onnx.is_file():
        return "onnx-exported"
    if paths.static_analysis.is_file():
        return "static-analysis-passed"
    if paths.checkpoint.is_file():
        return "checkpoint-created"
    return fallback
