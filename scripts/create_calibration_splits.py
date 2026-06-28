#!/usr/bin/env python3
"""Create deterministic front/rear calibration and test image lists."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPOSITORY_ROOT / "data"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "splits"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
TRAILING_NUMBER = re.compile(r"(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create disjoint calibration and test image lists."
    )
    parser.add_argument(
        "--camera",
        choices=("all", "front", "rear"),
        default="all",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Modulo interval used to create each split.",
    )
    parser.add_argument(
        "--calibration-remainder",
        type=int,
        default=0,
        help="Numeric ID remainder for calibration images.",
    )
    parser.add_argument(
        "--test-remainder",
        type=int,
        default=2,
        help="Numeric ID remainder for test images.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def image_number(path: Path) -> int:
    match = TRAILING_NUMBER.search(path.stem)
    if match is None:
        raise ValueError(f"Image filename has no trailing numeric ID: {path.name}")
    return int(match.group(1))


def list_images(camera_dir: Path) -> list[Path]:
    images = [
        path
        for path in camera_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images, key=lambda path: (image_number(path), path.name))


def relative_lines(paths: list[Path]) -> list[str]:
    return [path.relative_to(REPOSITORY_ROOT).as_posix() for path in paths]


def write_split_files(
    camera: str,
    split_name: str,
    selected: list[Path],
    images: list[Path],
    output_dir: Path,
    interval: int,
    remainder: int,
    camera_dir: Path,
    missing_numbers: list[int],
) -> None:
    if not selected:
        raise ValueError(
            f"No {split_name} images matched remainder {remainder} "
            f"with interval {interval} in {camera_dir}"
        )

    list_path = output_dir / f"{camera}_{split_name}.txt"
    metadata_path = output_dir / f"{camera}_{split_name}.json"
    lines = relative_lines(selected)
    list_content = "\n".join(lines) + "\n"
    list_path.write_text(list_content, encoding="utf-8")

    all_numbers = [image_number(path) for path in images]
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "camera": camera,
        "split": split_name,
        "selection_method": "numeric_id_modulo",
        "interval": interval,
        "remainder": remainder,
        "selection_rule": f"numeric_id % {interval} == {remainder}",
        "source_dir": camera_dir.relative_to(REPOSITORY_ROOT).as_posix(),
        "source_image_count": len(images),
        "source_numeric_id_min": min(all_numbers),
        "source_numeric_id_max": max(all_numbers),
        "missing_numeric_ids": missing_numbers,
        f"{split_name}_image_count": len(selected),
        f"{split_name}_fraction": len(selected) / len(images),
        "list_path": list_path.relative_to(REPOSITORY_ROOT).as_posix(),
        "list_sha256": hashlib.sha256(list_content.encode()).hexdigest(),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"  {split_name}: {len(selected)}/{len(images)} images "
        f"({len(selected) / len(images):.2%})"
    )
    print(f"    list: {list_path}")
    print(f"    metadata: {metadata_path}")


def write_camera_splits(
    camera: str,
    data_dir: Path,
    output_dir: Path,
    interval: int,
    calibration_remainder: int,
    test_remainder: int,
) -> None:
    camera_dir = data_dir / camera
    if not camera_dir.is_dir():
        raise FileNotFoundError(f"Camera data directory not found: {camera_dir}")

    images = list_images(camera_dir)
    if not images:
        raise ValueError(f"No images found in {camera_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_numbers = [image_number(path) for path in images]
    expected_numbers = set(range(min(all_numbers), max(all_numbers) + 1))
    missing_numbers = sorted(expected_numbers - set(all_numbers))
    calibration = [
        path
        for path in images
        if image_number(path) % interval == calibration_remainder
    ]
    test = [
        path
        for path in images
        if image_number(path) % interval == test_remainder
    ]
    overlap = set(calibration) & set(test)
    if overlap:
        raise RuntimeError(
            f"{camera} calibration/test overlap: {len(overlap)} images"
        )

    print(f"{camera}:")
    write_split_files(
        camera=camera,
        split_name="calibration",
        selected=calibration,
        images=images,
        output_dir=output_dir,
        interval=interval,
        remainder=calibration_remainder,
        camera_dir=camera_dir,
        missing_numbers=missing_numbers,
    )
    write_split_files(
        camera=camera,
        split_name="test",
        selected=test,
        images=images,
        output_dir=output_dir,
        interval=interval,
        remainder=test_remainder,
        camera_dir=camera_dir,
        missing_numbers=missing_numbers,
    )

    print("  overlap: 0")
    if missing_numbers:
        print(f"  warning: missing numeric IDs: {missing_numbers}")


def main() -> int:
    args = parse_args()
    if args.interval < 1:
        raise ValueError("--interval must be at least one.")
    remainders = (args.calibration_remainder, args.test_remainder)
    if any(remainder < 0 or remainder >= args.interval for remainder in remainders):
        raise ValueError("Split remainders must be between 0 and interval - 1.")
    if args.calibration_remainder == args.test_remainder:
        raise ValueError("Calibration and test remainders must be different.")

    data_dir = resolve_path(args.data_dir)
    output_dir = resolve_path(args.output_dir)
    cameras = ("front", "rear") if args.camera == "all" else (args.camera,)
    for camera in cameras:
        write_camera_splits(
            camera=camera,
            data_dir=data_dir,
            output_dir=output_dir,
            interval=args.interval,
            calibration_remainder=args.calibration_remainder,
            test_remainder=args.test_remainder,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
