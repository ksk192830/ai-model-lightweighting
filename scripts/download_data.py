#!/usr/bin/env python3
"""Download the front and rear camera datasets from Google Drive."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "dataset.yaml"
COMPLETION_MARKER = ".download_complete"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download configured camera datasets from Google Drive."
    )
    parser.add_argument(
        "--camera",
        choices=("all", "front", "rear"),
        default="all",
        help="Dataset to download (default: all).",
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
        help="Download even when images already exist locally.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    resolved_path = config_path.expanduser().resolve()
    with resolved_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid configuration in {resolved_path}")

    datasets = config.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError(f"'datasets' mapping is missing from {resolved_path}")

    return config


def download_dataset(
    name: str,
    dataset: dict[str, str],
    remote: str,
    dry_run: bool,
    force: bool,
) -> None:
    folder_id = dataset.get("drive_folder_id")
    output_value = dataset.get("output_dir")
    if not folder_id or not output_value:
        raise ValueError(
            f"Dataset '{name}' requires drive_folder_id and output_dir."
        )

    output_dir = REPOSITORY_ROOT / output_value
    completion_marker = output_dir / COMPLETION_MARKER

    if completion_marker.is_file() and not force:
        print(f"[{name}] skipped: download already completed in {output_dir}")
        return

    rclone = shutil.which("rclone")

    command = [
        rclone or "rclone",
        "copy",
        f"{remote}:",
        str(output_dir),
        "--drive-root-folder-id",
        folder_id,
        "--progress",
    ]

    print(f"[{name}] Google Drive folder {folder_id}")
    print(f"       -> {output_dir}")
    if dry_run:
        return
    if rclone is None:
        raise RuntimeError(
            "rclone is not installed. Install it and run 'rclone config' first."
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
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

        selected = datasets if args.camera == "all" else {args.camera: datasets[args.camera]}
        for name, dataset in selected.items():
            download_dataset(name, dataset, remote, args.dry_run, args.force)
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
