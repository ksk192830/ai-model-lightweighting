#!/usr/bin/env python3
"""Promote recovery checkpoints and rebuild deployment artifacts."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from kips_lightweighting.artifacts import artifact_paths  # noqa: E402
from kips_lightweighting.metadata import refresh_experiment_documents  # noqa: E402
from kips_lightweighting.registry import ExperimentRegistry  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_ids", nargs="+")
    parser.add_argument("--camera", choices=("front", "rear"), required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Explicit recovery PTH; valid only with one experiment ID.",
    )
    parser.add_argument(
        "--compare-to",
        default="B01",
        help="Baseline experiment for the regenerated static comparison.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Required to replace prototype PTH/ONNX/engine artifacts.",
    )
    parser.add_argument(
        "--skip-engine",
        action="store_true",
        help=(
            "Promote the recovery PTH, rebuild ONNX, and rerun analysis "
            "without building TensorRT. Any stale prototype engine is moved "
            "to the prototype snapshot and removed from the active artifact set."
        ),
    )
    return parser.parse_args()


def repository_path(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def default_recovery_checkpoint(directory: Path) -> Path:
    candidates = [
        directory / "recovery" / "checkpoint_best_total.pth",
        directory / "recovery-portable" / "checkpoint_best_total.pth",
    ]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(
            "Expected exactly one default recovery checkpoint; checked: "
            + ", ".join(str(path) for path in candidates)
        )
    return existing[0]


def run(command: list[str], dry_run: bool) -> None:
    print("$ " + " ".join(command))
    if not dry_run:
        subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


def main() -> int:
    args = parse_args()
    if args.checkpoint and len(args.experiment_ids) != 1:
        raise ValueError("--checkpoint can only be used with one experiment ID.")
    if not args.force and not args.dry_run:
        raise ValueError("Pass --force to replace prototype deployment artifacts.")

    registry = ExperimentRegistry.load()
    jobs = []
    for experiment_id in args.experiment_ids:
        experiment = registry.get(experiment_id)
        if args.camera not in experiment["cameras"]:
            raise ValueError(f"{experiment_id} is not registered for {args.camera}.")
        if not experiment.get("fine_tuning", {}).get("required", False):
            raise ValueError(f"{experiment_id} is not a recovery-trained experiment.")

        paths = artifact_paths(experiment_id, args.camera)
        recovered = (
            repository_path(args.checkpoint)
            if args.checkpoint
            else default_recovery_checkpoint(paths.directory)
        )
        if not recovered.is_file():
            raise FileNotFoundError(f"Recovery checkpoint not found: {recovered}")
        jobs.append((experiment_id, paths, recovered))

    for experiment_id, paths, recovered in jobs:
        snapshot = paths.directory / "prototype-before-recovery"
        if not snapshot.exists() and not args.dry_run:
            snapshot.mkdir(parents=True)
            for path in (
                paths.checkpoint,
                paths.onnx,
                paths.engine,
                paths.metadata,
                paths.static_analysis,
                paths.directory / f"comparison-{args.compare_to}.json",
                paths.directory / "onnx-build.json",
                paths.directory / "engine-build.json",
            ):
                if path.is_file():
                    shutil.copy2(path, snapshot / path.name)
        if not args.dry_run:
            shutil.copy2(recovered, paths.checkpoint)
            if args.skip_engine:
                for path in (
                    paths.engine,
                    paths.directory / "engine-build.json",
                ):
                    path.unlink(missing_ok=True)

        run(
            [
                sys.executable,
                str(REPOSITORY_ROOT / "scripts/experiments/build_candidate.py"),
                experiment_id,
                "--camera",
                args.camera,
                "--target",
                "onnx",
                "--force",
            ],
            args.dry_run,
        )
        if not args.skip_engine:
            run(
                [
                    sys.executable,
                    str(REPOSITORY_ROOT / "scripts/experiments/build_candidate.py"),
                    experiment_id,
                    "--camera",
                    args.camera,
                    "--target",
                    "engine",
                    "--force",
                ],
                args.dry_run,
            )
        run(
            [
                sys.executable,
                str(REPOSITORY_ROOT / "scripts/experiments/analyze_candidate.py"),
                experiment_id,
                "--camera",
                args.camera,
                "--compare-to",
                args.compare_to,
            ],
            args.dry_run,
        )
        if not args.dry_run:
            refresh_experiment_documents(experiment_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
