#!/usr/bin/env python3
"""Download the curated lightweighting datasets from Google Drive."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "dataset.yaml"
COMPLETION_MARKER = ".download_complete"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download curated calibration and labeled test datasets "
            "from Google Drive."
        )
    )
    parser.add_argument(
        "--subset",
        choices=("all", "calibration", "labeled-test"),
        default="all",
        help="Data subset to download (default: all).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Dataset configuration file.",
    )
    parser.add_argument(
        "--remote",
        help="Override the rclone remote name from the configuration file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned downloads without running rclone.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Sync again even when the local subset is marked complete.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    resolved_path = config_path.expanduser().resolve()
    with resolved_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid configuration in {resolved_path}")
    if not config.get("drive_root_folder_id"):
        raise ValueError(
            f"'drive_root_folder_id' is missing from {resolved_path}"
        )
    if not isinstance(config.get("datasets"), dict):
        raise ValueError(f"'datasets' mapping is missing from {resolved_path}")
    return config


def missing_required_paths(output_dir: Path, dataset: dict) -> list[str]:
    return [
        value
        for value in dataset.get("required_paths", [])
        if not (output_dir / value).is_file()
    ]


def download_dataset(
    name: str,
    dataset: dict,
    remote: str,
    root_folder_id: str,
    dry_run: bool,
    force: bool,
) -> None:
    source_dir = dataset.get("source_dir")
    output_value = dataset.get("output_dir")
    if not source_dir or not output_value:
        raise ValueError(
            f"Dataset '{name}' requires source_dir and output_dir."
        )

    output_dir = REPOSITORY_ROOT / output_value
    completion_marker = output_dir / COMPLETION_MARKER
    missing = missing_required_paths(output_dir, dataset)
    if completion_marker.is_file() and not missing and not force:
        print(f"[{name}] skipped: already complete in {output_dir}")
        return

    rclone = shutil.which("rclone")
    command = [
        rclone or "rclone",
        "copy",
        f"{remote}:{source_dir}",
        str(output_dir),
        "--drive-root-folder-id",
        root_folder_id,
        "--progress",
    ]
    print(f"[{name}] {remote}:{source_dir}")
    print(f"       Drive root: {root_folder_id}")
    print(f"       -> {output_dir}")
    if dry_run:
        print("       command:", " ".join(command))
        return
    if rclone is None:
        raise RuntimeError(
            "rclone is not installed. Install it and run 'rclone config' first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
    missing = missing_required_paths(output_dir, dataset)
    if missing:
        raise RuntimeError(
            f"Dataset '{name}' is incomplete; missing: {', '.join(missing)}"
        )
    completion_marker.touch()
    print(f"[{name}] download complete")


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        datasets = config["datasets"]
        remote = args.remote or config.get("rclone_remote")
        if not remote:
            raise ValueError("'rclone_remote' is missing from the configuration.")

        if args.subset == "all":
            selected = datasets
        else:
            selected = {args.subset: datasets[args.subset]}
        for name, dataset in selected.items():
            download_dataset(
                name=name,
                dataset=dataset,
                remote=remote,
                root_folder_id=config["drive_root_folder_id"],
                dry_run=args.dry_run,
                force=args.force,
            )
    except (
        KeyError,
        OSError,
        RuntimeError,
        ValueError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
