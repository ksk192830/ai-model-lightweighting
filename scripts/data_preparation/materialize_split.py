#!/usr/bin/env python3
"""Copy files from a split list into a portable data directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy split-list files into one portable directory."
    )
    parser.add_argument("--split-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    split_list = resolve_path(args.split_list)
    output_dir = resolve_path(args.output)
    entries = [
        line.strip()
        for line in split_list.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for value in entries:
        source = resolve_path(Path(value))
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = output_dir / source.name
        if destination.exists():
            raise FileExistsError(destination)
        shutil.copy2(source, destination)
        records.append(
            {
                "file_name": destination.name,
                "size_bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )

    manifest = {
        "split_list": str(split_list.relative_to(REPOSITORY_ROOT)),
        "image_count": len(records),
        "files": records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"copied: {len(records)} images -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
