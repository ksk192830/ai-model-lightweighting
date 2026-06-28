#!/usr/bin/env python3
"""Create labeled test lists by replacing unavailable requested images."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ROBOFLOW_SUFFIX = re.compile(r"(.+)_(png|jpe?g)$", re.IGNORECASE)
IMAGE_NUMBER = re.compile(r"(?:image|frame)[_-]?0*(\d+)", re.IGNORECASE)
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a test list containing only labeled images."
    )
    parser.add_argument("--camera", choices=("front", "rear"), required=True)
    return parser.parse_args()


def original_names_from_coco(source: Path) -> set[str]:
    names: set[str] = set()
    for annotation_path in source.rglob("_annotations.coco.json"):
        coco = json.loads(annotation_path.read_text(encoding="utf-8"))
        for image in coco.get("images", []):
            stem = Path(image["file_name"]).name.split(".rf.", 1)[0]
            match = ROBOFLOW_SUFFIX.fullmatch(stem)
            if match:
                names.add(f"{match.group(1)}.{match.group(2).lower()}")
    return names


def image_number(name: str) -> int:
    match = IMAGE_NUMBER.search(name)
    if match is None:
        raise ValueError(f"Could not parse an image number from: {name}")
    return int(match.group(1))


def read_list(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def main() -> int:
    args = parse_args()
    camera = args.camera
    split_dir = REPOSITORY_ROOT / "splits"

    requested = read_list(split_dir / f"{camera}_test.txt")
    calibration = {
        Path(value).name
        for value in read_list(split_dir / f"{camera}_calibration.txt")
    }
    raw_names = {
        path.name
        for path in (REPOSITORY_ROOT / "data" / camera).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    labeled_names = original_names_from_coco(
        REPOSITORY_ROOT / "data" / "roboflow" / camera
    )
    eligible_names = (raw_names & labeled_names) - calibration

    selected = [
        value for value in requested if Path(value).name in eligible_names
    ]
    selected_names = {Path(value).name for value in selected}
    replacement_pool = eligible_names - selected_names
    missing = [
        value for value in requested if Path(value).name not in eligible_names
    ]
    replacements: list[tuple[str, str]] = []

    for missing_value in missing:
        missing_name = Path(missing_value).name
        target_number = image_number(missing_name)
        replacement = min(
            replacement_pool,
            key=lambda name: (
                abs(image_number(name) - target_number),
                image_number(name),
                name,
            ),
        )
        replacement_pool.remove(replacement)
        replacements.append((missing_name, replacement))
        selected.append(f"data/{camera}/{replacement}")

    selected.sort(key=lambda value: image_number(Path(value).name))
    overlap = sum(Path(value).name in calibration for value in selected)
    if overlap:
        raise RuntimeError(
            f"{camera} labeled test/calibration overlap: {overlap}"
        )

    output_path = split_dir / f"{camera}_test_labeled.txt"
    output_path.write_text(
        "\n".join(selected) + "\n",
        encoding="utf-8",
    )
    report = {
        "camera": camera,
        "target_count": len(requested),
        "retained_count": len(requested) - len(missing),
        "replacement_count": len(replacements),
        "calibration_overlap": overlap,
        "replacements": [
            {"missing": missing_name, "replacement": replacement}
            for missing_name, replacement in replacements
        ],
        "output": str(output_path.relative_to(REPOSITORY_ROOT)),
    }
    report_path = split_dir / f"{camera}_test_labeled.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"{camera}: {len(selected)} images "
        f"({len(replacements)} replacements), overlap={overlap}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
