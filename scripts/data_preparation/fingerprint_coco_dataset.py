#!/usr/bin/env python3
"""Create or compare privacy-preserving fingerprints for a COCO dataset.

The report contains hashes and aggregate counts only. Image bytes and annotation
geometry are read locally but are never copied into the output report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
SCHEMA_VERSION = 1
COMPARISON_KEYS = (
    "counts",
    "category_annotation_counts",
    "coco_evaluation_semantics_sha256",
    "image_collection_sha256",
    "dataset_fingerprint_sha256",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def normalize_number(value: int | float) -> int | str:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("COCO numeric values must be finite")
    decimal = Decimal(str(value))
    if decimal == 0:
        return 0
    if decimal == decimal.to_integral_value():
        return int(decimal)
    return format(decimal.normalize(), "f")


def normalize_value(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return normalize_number(value)
    if isinstance(value, list):
        return [normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): normalize_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    raise TypeError(f"Unsupported COCO value type: {type(value).__name__}")


def normalized_file_name(value: Any) -> str:
    name = str(value).replace("\\", "/")
    while name.startswith("./"):
        name = name[2:]
    return name


def image_path(dataset: Path, file_name: str) -> Path:
    direct = dataset / file_name
    if direct.is_file():
        return direct
    matches = [
        path
        for path in dataset.rglob(Path(file_name).name)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"COCO image is missing: {file_name}")
    raise ValueError(f"COCO image name is ambiguous: {file_name}")


def build_fingerprint(dataset: Path, label: str) -> dict[str, Any]:
    dataset = dataset.resolve()
    annotation_path = dataset / "_annotations.coco.json"
    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])
    image_by_id: dict[Any, dict[str, Any]] = {}
    for image in images:
        image_id = image.get("id")
        if image_id in image_by_id:
            raise ValueError(f"Duplicate COCO image id: {image_id}")
        image_by_id[image_id] = image
    category_by_id: dict[Any, dict[str, Any]] = {}
    for category in categories:
        category_id = category.get("id")
        if category_id in category_by_id:
            raise ValueError(f"Duplicate COCO category id: {category_id}")
        category_by_id[category_id] = category

    semantic_images = []
    image_hash_rows = []
    file_names: set[str] = set()
    for image in images:
        file_name = normalized_file_name(image.get("file_name"))
        if file_name in file_names:
            raise ValueError(f"Duplicate COCO image file_name: {file_name}")
        file_names.add(file_name)
        path = image_path(dataset, file_name)
        semantic_images.append(
            {
                "file_name": file_name,
                "width": normalize_value(image.get("width")),
                "height": normalize_value(image.get("height")),
            }
        )
        image_hash_rows.append(
            {
                "file_name": file_name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    semantic_categories = []
    for category in categories:
        semantic_categories.append(
            {
                "id": normalize_value(category.get("id")),
                "name": category.get("name"),
                "supercategory": category.get("supercategory"),
                "keypoints": normalize_value(category.get("keypoints")),
                "skeleton": normalize_value(category.get("skeleton")),
            }
        )

    semantic_annotations = []
    class_counts: Counter[str] = Counter()
    for annotation in annotations:
        image = image_by_id.get(annotation.get("image_id"))
        if image is None:
            raise ValueError(
                "Annotation references missing image id: "
                f"{annotation.get('image_id')}"
            )
        category = category_by_id.get(annotation.get("category_id"))
        if category is None:
            raise ValueError(
                "Annotation references missing category id: "
                f"{annotation.get('category_id')}"
            )
        category_name = str(category.get("name"))
        class_counts[category_name] += 1
        segmentation = normalize_value(annotation.get("segmentation"))
        if isinstance(segmentation, list):
            segmentation = sorted(segmentation, key=canonical_json)
        semantic_annotations.append(
            {
                "image_file_name": normalized_file_name(image.get("file_name")),
                "category_id": normalize_value(annotation.get("category_id")),
                "category_name": category_name,
                "bbox": normalize_value(annotation.get("bbox")),
                "segmentation": segmentation,
                "area": normalize_value(annotation.get("area")),
                "iscrowd": normalize_value(annotation.get("iscrowd", 0)),
            }
        )

    semantic_payload = {
        "images": sorted(semantic_images, key=canonical_json),
        "categories": sorted(semantic_categories, key=canonical_json),
        "annotations": sorted(semantic_annotations, key=canonical_json),
    }
    image_hash_rows.sort(key=canonical_json)
    semantic_sha256 = digest_json(semantic_payload)
    image_content_sha256 = digest_json(image_hash_rows)
    combined_sha256 = digest_json(
        {
            "schema_version": SCHEMA_VERSION,
            "coco_evaluation_semantics_sha256": semantic_sha256,
            "image_collection_sha256": image_content_sha256,
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "privacy": (
            "hashes and aggregate counts only; image bytes and annotation geometry "
            "are not included"
        ),
        "counts": {
            "images": len(images),
            "annotations": len(annotations),
            "categories": len(categories),
        },
        "category_annotation_counts": dict(sorted(class_counts.items())),
        "raw_annotation_sha256": sha256_file(annotation_path),
        "coco_evaluation_semantics_sha256": semantic_sha256,
        "image_collection_sha256": image_content_sha256,
        "dataset_fingerprint_sha256": combined_sha256,
    }


def validate_report(report: dict[str, Any], side: str) -> None:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{side} report schema_version must be {SCHEMA_VERSION}"
        )
    required = ("label", "raw_annotation_sha256", *COMPARISON_KEYS)
    missing = [key for key in required if key not in report]
    if missing:
        raise ValueError(f"{side} report lacks required fields: {missing}")


def compare_reports(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    validate_report(left, "left")
    validate_report(right, "right")
    comparisons = {
        key: left.get(key) == right.get(key) for key in COMPARISON_KEYS
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "left_label": left.get("label"),
        "right_label": right.get("label"),
        "equivalent": all(comparisons.values()),
        "raw_annotation_bytes_equal": (
            left.get("raw_annotation_sha256")
            == right.get("raw_annotation_sha256")
        ),
        "comparisons": comparisons,
        "interpretation": (
            "The evaluated COCO semantics and image bytes are identical. Raw JSON "
            "hashes may differ because ordering or serialization differs."
            if all(comparisons.values())
            else "The datasets are not proven equivalent; do not reuse accuracy results."
        ),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--dataset", type=Path, required=True)
    generate.add_argument("--label", required=True)
    generate.add_argument("--output", type=Path, required=True)
    compare = subparsers.add_parser("compare")
    compare.add_argument("left", type=Path)
    compare.add_argument("right", type=Path)
    compare.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "generate":
        payload = build_fingerprint(args.dataset, args.label)
        write_json(args.output, payload)
        print(f"dataset fingerprint: {payload['dataset_fingerprint_sha256']}")
        print(f"report: {args.output}")
        return 0
    left = json.loads(args.left.read_text(encoding="utf-8"))
    right = json.loads(args.right.read_text(encoding="utf-8"))
    payload = compare_reports(left, right)
    write_json(args.output, payload)
    print("equivalent" if payload["equivalent"] else "different")
    print(f"report: {args.output}")
    return 0 if payload["equivalent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
