#!/usr/bin/env python3
"""Package leakage-safe INT8 calibration and final-test data for a notebook."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    REPOSITORY_ROOT / "data" / "training" / "front_session_split_v1"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / "delivery" / "notebook-front-ready4"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--calibration-count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_coco(directory: Path) -> dict:
    path = directory / "_annotations.coco.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def verify_coco(directory: Path, coco: dict) -> list[Path]:
    paths = image_files(directory)
    actual = {path.name for path in paths}
    referenced = {str(image["file_name"]) for image in coco.get("images", [])}
    if actual != referenced:
        missing = sorted(referenced - actual)
        extra = sorted(actual - referenced)
        raise ValueError(
            f"COCO/image mismatch in {directory}: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    image_ids = {int(image["id"]) for image in coco.get("images", [])}
    bad_references = [
        annotation["id"]
        for annotation in coco.get("annotations", [])
        if int(annotation["image_id"]) not in image_ids
    ]
    if bad_references:
        raise ValueError(
            f"COCO annotations reference missing images in {directory}: "
            f"{bad_references[:3]}"
        )
    return paths


def filtered_coco(coco: dict, selected_names: set[str]) -> dict:
    selected_images = [
        image for image in coco.get("images", [])
        if str(image["file_name"]) in selected_names
    ]
    selected_ids = {int(image["id"]) for image in selected_images}
    output = dict(coco)
    output["info"] = dict(coco.get("info", {}))
    output["info"]["description"] = (
        "Deterministic train-only INT8 calibration subset; not an evaluation split"
    )
    output["images"] = selected_images
    output["annotations"] = [
        annotation for annotation in coco.get("annotations", [])
        if int(annotation["image_id"]) in selected_ids
    ]
    return output


def class_summary(coco: dict) -> dict:
    by_image: dict[int, set[int]] = defaultdict(set)
    annotation_counts: Counter[int] = Counter()
    for annotation in coco.get("annotations", []):
        category_id = int(annotation["category_id"])
        annotation_counts[category_id] += 1
        by_image[int(annotation["image_id"])].add(category_id)
    presence: Counter[int] = Counter()
    for categories in by_image.values():
        presence.update(categories)
    image_count = len(coco.get("images", []))
    negatives = sum(
        not by_image[int(image["id"])] for image in coco.get("images", [])
    )
    return {
        "images": image_count,
        "annotations": len(coco.get("annotations", [])),
        "negative_images": negatives,
        "class_annotations": {
            str(key): annotation_counts[key] for key in sorted(annotation_counts)
        },
        "class_image_presence": {
            str(key): presence[key] for key in sorted(presence)
        },
    }


def link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    os.link(source, target)


def main() -> int:
    args = parse_args()
    source = resolve(args.source)
    output = resolve(args.output_dir)
    if args.calibration_count < 1:
        raise ValueError("--calibration-count must be positive")

    train_dir = source / "train"
    test_dir = source / "test"
    train_coco = load_coco(train_dir)
    test_coco = load_coco(test_dir)
    train_images = verify_coco(train_dir, train_coco)
    test_images = verify_coco(test_dir, test_coco)
    if args.calibration_count > len(train_images):
        raise ValueError(
            f"Requested {args.calibration_count} calibration images from "
            f"only {len(train_images)} train images"
        )

    # Match quantize_modelopt.calibration_images: lexical order, seeded shuffle,
    # then the first N images. This samples natural negatives and classes without
    # looking at valid or test labels.
    candidates = list(train_images)
    random.Random(args.seed).shuffle(candidates)
    selected = candidates[: args.calibration_count]
    selected_names = {path.name for path in selected}
    calibration_coco = filtered_coco(train_coco, selected_names)

    train_hashes = {sha256(path) for path in train_images}
    test_hashes = {sha256(path) for path in test_images}
    filename_overlap = {path.name for path in train_images} & {
        path.name for path in test_images
    }
    content_overlap = train_hashes & test_hashes
    if filename_overlap or content_overlap:
        raise ValueError(
            "Train/test leakage detected: "
            f"filename_overlap={len(filename_overlap)}, "
            f"content_overlap={len(content_overlap)}"
        )

    data_root = output / "data" / "training" / "front_session_split_v1"
    calibration_out = data_root / "train"
    test_out = data_root / "test"
    metadata_paths = (
        output / "data-manifest.json",
        output / "data-checksums.sha256",
        output / "calibration-selection.txt",
    )
    occupied = [path for path in (data_root, *metadata_paths) if path.exists()]
    if occupied:
        raise FileExistsError(
            "Notebook data bundle already exists; refusing to overwrite: "
            + ", ".join(str(path) for path in occupied)
        )

    for path in selected:
        link(path, calibration_out / path.name)
    calibration_annotation = calibration_out / "_annotations.coco.json"
    calibration_annotation.write_text(
        json.dumps(calibration_coco, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for path in test_images:
        link(path, test_out / path.name)
    link(test_dir / "_annotations.coco.json", test_out / "_annotations.coco.json")

    selection_path = output / "calibration-selection.txt"
    selection_path.write_text(
        "".join(f"{path.name}\n" for path in selected), encoding="utf-8"
    )

    payload = sorted(
        [*image_files(calibration_out), calibration_annotation]
        + [*image_files(test_out), test_out / "_annotations.coco.json"]
        + [selection_path]
    )
    checksum_path = output / "data-checksums.sha256"
    checksum_rows = []
    payload_size = 0
    for path in payload:
        relative = path.relative_to(output).as_posix()
        checksum_rows.append(f"{sha256(path)}  {relative}\n")
        payload_size += path.stat().st_size
    checksum_path.write_text("".join(checksum_rows), encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "B03 train-only INT8 calibration and final test evaluation",
        "source_split_root": str(source.relative_to(REPOSITORY_ROOT)),
        "materialization": {
            "method": "hardlink",
            "source_files_modified": False,
            "portable_note": (
                "Archive or copy the delivery directory to materialize independent "
                "files on the notebook."
            ),
        },
        "calibration": {
            "path": str(calibration_out.relative_to(output)),
            "source_split": "train",
            "source_train_images": len(train_images),
            "selected_images": len(selected),
            "seed": args.seed,
            "selection": (
                "Sort train image paths lexicographically, shuffle with Python "
                "random.Random(seed), select first N; valid/test are never candidates."
            ),
            "selection_list": selection_path.relative_to(output).as_posix(),
            "selection_list_sha256": sha256(selection_path),
            "summary": class_summary(calibration_coco),
        },
        "test": {
            "path": str(test_out.relative_to(output)),
            "role": "final accuracy evaluation only; never calibration",
            "summary": class_summary(test_coco),
            "annotation_file": str(
                (test_out / "_annotations.coco.json").relative_to(output)
            ),
            "annotation_sha256": sha256(test_out / "_annotations.coco.json"),
        },
        "leakage_verification": {
            "scope": "all 3,625 source-train images versus all 437 test images",
            "train_test_filename_overlap": len(filename_overlap),
            "train_test_sha256_overlap": len(content_overlap),
            "coco_reference_errors": 0,
            "passed": True,
        },
        "integrity": {
            "payload_files": len(payload),
            "payload_size_bytes": payload_size,
            "checksums_file": checksum_path.relative_to(output).as_posix(),
            "checksums_file_sha256": sha256(checksum_path),
            "verification_command": "sha256sum -c data-checksums.sha256",
        },
    }
    manifest_path = output / "data-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"data bundle: {data_root}")
    print(f"calibration: {len(selected)} train images (seed={args.seed})")
    print(
        f"test: {len(test_images)} images, "
        f"{len(test_coco.get('annotations', []))} annotations"
    )
    print("train/test filename overlap: 0")
    print("train/test SHA-256 overlap: 0")
    print(f"payload: {len(payload)} files, {payload_size} bytes")
    print(f"manifest: {manifest_path}")
    print(f"checksums: {checksum_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
