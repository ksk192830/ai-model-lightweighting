#!/usr/bin/env python3
"""Promote recovery checkpoints and rebuild deployment artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from kips_lightweighting.artifacts import artifact_paths  # noqa: E402
from kips_lightweighting.metadata import (  # noqa: E402
    read_json,
    refresh_experiment_documents,
    runtime_metadata,
    sha256,
    write_json,
)
from kips_lightweighting.registry import ExperimentRegistry  # noqa: E402


SNAPSHOT_DIRECTORY_NAME = "prototype-before-recovery"
SNAPSHOT_MANIFEST_NAME = "prototype-snapshot.json"
SNAPSHOT_MANIFEST_VERSION = 1


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


def portable_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}-",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        write_json(temporary, data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _snapshot_reference(
    value: Any,
    active_directory: Path,
    stored_files: Path,
    reference_directory: Path,
) -> Any:
    """Rebase JSON references to root prototype files into the snapshot."""
    if isinstance(value, dict):
        return {
            key: _snapshot_reference(
                item,
                active_directory,
                stored_files,
                reference_directory,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _snapshot_reference(
                item,
                active_directory,
                stored_files,
                reference_directory,
            )
            for item in value
        ]
    if not isinstance(value, str):
        return value
    referenced = Path(value)
    resolved = referenced if referenced.is_absolute() else REPOSITORY_ROOT / referenced
    try:
        relative = resolved.relative_to(active_directory)
    except ValueError:
        return value
    if len(relative.parts) != 1 or not (stored_files / relative).is_file():
        return value
    return portable_path(reference_directory / relative)


def normalize_snapshot_json(
    snapshot: Path,
    active_directory: Path,
    reference_directory: Path | None = None,
) -> None:
    """Make copied JSON self-contained without changing recorded model hashes."""
    reference_directory = reference_directory or snapshot
    for path in sorted(snapshot.glob("*.json")):
        if path.name == SNAPSHOT_MANIFEST_NAME:
            continue
        try:
            data = read_json(path)
        except (json.JSONDecodeError, ValueError):
            continue
        data = _snapshot_reference(
            data,
            active_directory,
            snapshot,
            reference_directory,
        )
        if path.name == "metadata.json" and isinstance(data.get("artifacts"), dict):
            # A legacy snapshot can lack files that old finalization code did
            # not copy. Do not let its metadata silently resolve those names
            # to newer, recovered artifacts in the active directory.
            artifacts = {}
            for name, value in data["artifacts"].items():
                if not isinstance(value, str):
                    artifacts[name] = value
                    continue
                referenced = Path(value)
                resolved = (
                    referenced
                    if referenced.is_absolute()
                    else REPOSITORY_ROOT / referenced
                )
                try:
                    resolved.relative_to(reference_directory)
                    in_snapshot = True
                except ValueError:
                    in_snapshot = False
                try:
                    resolved.relative_to(active_directory)
                    in_active_directory = True
                except ValueError:
                    in_active_directory = False
                if in_snapshot or not in_active_directory:
                    artifacts[name] = value
            data["artifacts"] = artifacts
        write_json_atomic(path, data)


def snapshot_manifest(
    snapshot: Path,
    active_directory: Path,
    reference_directory: Path | None = None,
) -> dict[str, Any]:
    reference_directory = reference_directory or snapshot
    files = {
        path.name: {
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(snapshot.iterdir())
        if path.is_file() and path.name != SNAPSHOT_MANIFEST_NAME
    }
    return {
        **runtime_metadata(),
        "format_version": SNAPSHOT_MANIFEST_VERSION,
        "source_directory": portable_path(active_directory),
        "snapshot_directory": portable_path(reference_directory),
        "files": files,
    }


def validate_snapshot(snapshot: Path) -> None:
    manifest_path = snapshot / SNAPSHOT_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Snapshot manifest not found: {manifest_path}")
    manifest = read_json(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, dict) or "model.pth" not in files:
        raise ValueError(f"Invalid prototype snapshot manifest: {manifest_path}")
    for name, recorded in files.items():
        path = snapshot / name
        if not path.is_file():
            raise FileNotFoundError(f"Snapshot file not found: {path}")
        if not isinstance(recorded, dict):
            raise ValueError(f"Invalid snapshot entry for {name}: {manifest_path}")
        if recorded.get("size_bytes") != path.stat().st_size:
            raise ValueError(f"Snapshot size mismatch: {path}")
        if recorded.get("sha256") != sha256(path):
            raise ValueError(f"Snapshot hash mismatch: {path}")


def ensure_prototype_snapshot(active_directory: Path) -> Path:
    """Create an immutable, self-contained snapshot of root prototype files."""
    snapshot = active_directory / SNAPSHOT_DIRECTORY_NAME
    if snapshot.exists():
        if not snapshot.is_dir():
            raise NotADirectoryError(
                f"Prototype snapshot is not a directory: {snapshot}"
            )
        manifest_path = snapshot / SNAPSHOT_MANIFEST_NAME
        if manifest_path.is_file():
            validate_snapshot(snapshot)
            manifest = read_json(manifest_path)
            if manifest.get("format_version") == SNAPSHOT_MANIFEST_VERSION:
                return snapshot
            normalize_snapshot_json(snapshot, active_directory)
            write_json_atomic(
                manifest_path,
                snapshot_manifest(snapshot, active_directory),
            )
        else:
            # Upgrade a legacy snapshot in place. Never backfill it from the
            # active directory: after promotion those files are the recovered
            # model and would silently corrupt the preserved prototype.
            if not (snapshot / "model.pth").is_file():
                raise FileNotFoundError(
                    f"Legacy prototype snapshot has no model.pth: {snapshot}"
                )
            normalize_snapshot_json(snapshot, active_directory)
            write_json_atomic(
                manifest_path,
                snapshot_manifest(snapshot, active_directory),
            )
        validate_snapshot(snapshot)
        return snapshot

    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{SNAPSHOT_DIRECTORY_NAME}-",
            dir=active_directory,
        )
    )
    try:
        sources = sorted(path for path in active_directory.iterdir() if path.is_file())
        if not any(path.name == "model.pth" for path in sources):
            raise FileNotFoundError(
                f"Prototype checkpoint not found: {active_directory / 'model.pth'}"
            )
        for source in sources:
            shutil.copy2(source, temporary / source.name)
        normalize_snapshot_json(
            temporary,
            active_directory,
            reference_directory=snapshot,
        )
        write_json_atomic(
            temporary / SNAPSHOT_MANIFEST_NAME,
            snapshot_manifest(
                temporary,
                active_directory,
                reference_directory=snapshot,
            ),
        )
        temporary.rename(snapshot)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    validate_snapshot(snapshot)
    return snapshot


def relocate_recovery_provenance(
    active_directory: Path,
    snapshot: Path,
    recovered: Path,
) -> None:
    """Point training reports at the immutable checkpoint they actually used."""
    prototype = snapshot / "model.pth"
    prototype_hash = sha256(prototype)
    active_checkpoint = active_directory / "model.pth"
    recovered_hash = sha256(recovered)
    for report_path in sorted(active_directory.glob("**/recovery-training.json")):
        if snapshot in report_path.parents:
            continue
        report = read_json(report_path)
        original_report = dict(report)
        source_value = report.get("source_checkpoint")
        source_hash = report.get("source_checkpoint_sha256")
        if not isinstance(source_value, str):
            continue
        source = repository_path(Path(source_value))
        if (
            source not in {active_checkpoint, prototype}
            or source_hash != prototype_hash
        ):
            continue
        report["source_checkpoint"] = portable_path(prototype)
        report["source_checkpoint_sha256"] = prototype_hash
        if report_path == recovered.parent / "recovery-training.json":
            report["promoted_checkpoint"] = portable_path(active_checkpoint)
            report["promoted_checkpoint_sha256"] = recovered_hash
        if report != original_report:
            write_json_atomic(report_path, report)


def promote_checkpoint(recovered: Path, active_checkpoint: Path) -> None:
    """Atomically replace the active checkpoint with a verified recovery file."""
    if recovered.resolve() == active_checkpoint.resolve():
        return
    descriptor, name = tempfile.mkstemp(
        prefix=f".{active_checkpoint.name}-",
        dir=active_checkpoint.parent,
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        shutil.copy2(recovered, temporary)
        if sha256(temporary) != sha256(recovered):
            raise OSError(f"Promoted checkpoint hash mismatch: {recovered}")
        temporary.replace(active_checkpoint)
    finally:
        temporary.unlink(missing_ok=True)


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
        snapshot = paths.directory / SNAPSHOT_DIRECTORY_NAME
        if not args.dry_run:
            snapshot = ensure_prototype_snapshot(paths.directory)
            relocate_recovery_provenance(paths.directory, snapshot, recovered)
        if not args.dry_run:
            promote_checkpoint(recovered, paths.checkpoint)
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
