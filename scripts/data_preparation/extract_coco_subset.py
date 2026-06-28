#!/usr/bin/env python3
"""Extract a labeled COCO subset using a repository split list."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageChops, ImageStat


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "data" / "labeled_test"
CAMERA_NUMBER_PATTERN = re.compile(r"(?:image|frame)[_-]?0*(\d+)", re.IGNORECASE)
ANY_NUMBER_PATTERN = re.compile(r"(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter Roboflow COCO Segmentation data using a split list."
    )
    parser.add_argument("--camera", choices=("front", "rear"), required=True)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Extracted Roboflow export containing COCO annotation JSON files.",
    )
    parser.add_argument(
        "--split-list",
        type=Path,
        help="Defaults to splits/<camera>_test.txt.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Defaults to data/labeled_test/<camera>.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Create a partial subset instead of failing on unmatched images.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def image_key(path_or_name: str | Path) -> int:
    name = Path(path_or_name).name
    preferred = CAMERA_NUMBER_PATTERN.search(name)
    if preferred is not None:
        return int(preferred.group(1))

    numbers = ANY_NUMBER_PATTERN.findall(Path(name).stem)
    if len(numbers) == 1:
        return int(numbers[0])
    raise ValueError(f"Could not determine a unique numeric image ID: {name}")


def load_requested(split_list: Path) -> dict[int, str]:
    requested: dict[int, str] = {}
    for line in split_list.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        key = image_key(value)
        if key in requested:
            raise ValueError(
                f"Duplicate numeric ID {key} in split list: "
                f"{requested[key]} and {value}"
            )
        requested[key] = value
    if not requested:
        raise ValueError(f"Split list is empty: {split_list}")
    return requested


def find_annotation_files(source: Path) -> list[Path]:
    preferred = sorted(source.rglob("_annotations.coco.json"))
    if preferred:
        return preferred

    candidates = []
    for path in sorted(source.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        required = {"images", "annotations", "categories"}
        if isinstance(payload, dict) and required <= payload.keys():
            candidates.append(path)
    return candidates


def resolve_image_file(annotation_path: Path, file_name: str) -> Path:
    candidates = (
        annotation_path.parent / file_name,
        annotation_path.parent / Path(file_name).name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Image referenced by {annotation_path} was not found: {file_name}"
    )


def pixel_fingerprint(path: Path) -> tuple[tuple[int, int], str]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return rgb.size, hashlib.sha256(rgb.tobytes()).hexdigest()


def original_filename(path_or_name: str | Path) -> str:
    """Restore the original filename encoded in a Roboflow export name."""
    name = Path(path_or_name).name
    stem = name.split(".rf.", 1)[0]
    match = re.fullmatch(r"(.+)_(png|jpe?g)", stem, re.IGNORECASE)
    if match:
        return f"{match.group(1)}.{match.group(2).lower()}"
    return name


def pixel_mae(reference: Path, candidate: Path) -> float:
    """Return RGB mean absolute error; lower means visually closer."""
    with Image.open(reference) as ref_image, Image.open(candidate) as candidate_image:
        ref_rgb = ref_image.convert("RGB")
        candidate_rgb = candidate_image.convert("RGB")
        if candidate_rgb.size != ref_rgb.size:
            candidate_rgb = candidate_rgb.resize(ref_rgb.size, Image.Resampling.BILINEAR)
        return sum(ImageStat.Stat(ImageChops.difference(ref_rgb, candidate_rgb)).mean) / 3


def comparable_annotations(annotations: list[dict]) -> str:
    normalized = [
        {key: value for key, value in item.items() if key not in {"id", "image_id"}}
        for item in annotations
    ]
    normalized.sort(key=lambda value: json.dumps(value, sort_keys=True))
    return json.dumps(normalized, sort_keys=True)


def load_matching_records(annotation_files: list[Path], requested: dict[int, str]):
    candidates: dict[int, list[dict]] = defaultdict(list)
    categories = None
    template = {}
    for annotation_path in annotation_files:
        coco = json.loads(annotation_path.read_text(encoding="utf-8"))
        current_categories = coco.get("categories", [])
        if categories is None:
            categories = current_categories
            template = {key: coco[key] for key in ("info", "licenses") if key in coco}
        elif current_categories != categories:
            raise ValueError(f"COCO categories differ between exports: {annotation_path}")
        grouped = defaultdict(list)
        for annotation in coco.get("annotations", []):
            grouped[annotation["image_id"]].append(annotation)
        for image in coco.get("images", []):
            try:
                key = image_key(image["file_name"])
            except ValueError:
                continue
            if key in requested:
                candidates[key].append({
                    "record": image,
                    "source_path": resolve_image_file(annotation_path, image["file_name"]),
                    "annotations": grouped[image["id"]],
                })
    matched_images = {}
    matched_annotations = defaultdict(list)
    candidate_counts = {}
    for key, requested_value in requested.items():
        original_path = resolve_path(Path(requested_value))
        if not original_path.is_file():
            raise FileNotFoundError(f"Original split image not found: {original_path}")
        key_candidates = candidates.get(key, [])
        requested_name = original_path.name.casefold()
        filename_matches = [
            item for item in key_candidates
            if original_filename(item["record"]["file_name"]).casefold() == requested_name
        ]
        candidate_counts[key] = len(key_candidates)
        if not filename_matches:
            continue
        ranked = [
            (pixel_mae(original_path, item["source_path"]), str(item["source_path"]), item)
            for item in filename_matches
        ]
        score, _, selected = min(ranked, key=lambda value: (value[0], value[1]))
        matched_images[key] = {
            "record": selected["record"],
            "source_path": selected["source_path"],
            "output_name": original_path.name,
            "pixel_mae": score,
        }
        matched_annotations[key] = selected["annotations"]
    return matched_images, matched_annotations, categories or [], template, candidate_counts


def build_subset(
    matched_images: dict[int, dict],
    matched_annotations: dict[int, list[dict]],
    categories: list[dict],
    template: dict,
    output_dir: Path,
) -> tuple[dict, int]:
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    output_images = []
    output_annotations = []
    negative_image_count = 0
    annotation_id = 1

    for new_image_id, key in enumerate(sorted(matched_images), start=1):
        match = matched_images[key]
        source_path: Path = match["source_path"]
        destination_path = images_dir / match["output_name"]
        if destination_path.exists():
            raise FileExistsError(
                f"Duplicate destination image filename: {destination_path.name}"
            )
        shutil.copy2(source_path, destination_path)

        image_record = dict(match["record"])
        image_record["id"] = new_image_id
        image_record["file_name"] = f"images/{destination_path.name}"
        output_images.append(image_record)

        source_annotations = matched_annotations.get(key, [])
        if not source_annotations:
            negative_image_count += 1
        for source_annotation in source_annotations:
            annotation = dict(source_annotation)
            annotation["id"] = annotation_id
            annotation["image_id"] = new_image_id
            output_annotations.append(annotation)
            annotation_id += 1

    subset = {
        **template,
        "images": output_images,
        "annotations": output_annotations,
        "categories": categories,
    }
    return subset, negative_image_count


def main() -> int:
    args = parse_args()
    source = resolve_path(args.source)
    split_list = resolve_path(
        args.split_list or Path("splits") / f"{args.camera}_test.txt"
    )
    output_dir = resolve_path(
        args.output or DEFAULT_OUTPUT_ROOT / args.camera
    )

    if not source.is_dir():
        raise FileNotFoundError(f"Roboflow export directory not found: {source}")
    if not split_list.is_file():
        raise FileNotFoundError(f"Split list not found: {split_list}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Move or clear the "
            "previous generated subset before running again."
        )

    requested = load_requested(split_list)
    annotation_files = find_annotation_files(source)
    if not annotation_files:
        raise FileNotFoundError(
            f"No COCO annotation JSON files found under {source}"
        )

    (
        matched_images,
        matched_annotations,
        categories,
        template,
        candidate_counts,
    ) = load_matching_records(annotation_files, requested)
    missing_ids = sorted(set(requested) - set(matched_images))
    if missing_ids and not args.allow_missing:
        raise ValueError(
            f"{len(missing_ids)} requested images were not found. "
            f"First missing IDs: {missing_ids[:20]}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    subset, negative_count = build_subset(
        matched_images=matched_images,
        matched_annotations=matched_annotations,
        categories=categories,
        template=template,
        output_dir=output_dir,
    )
    annotation_output = output_dir / "_annotations.coco.json"
    annotation_output.write_text(
        json.dumps(subset, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "camera": args.camera,
        "requested_image_count": len(requested),
        "matched_image_count": len(matched_images),
        "missing_numeric_ids": missing_ids,
        "candidate_count_min": min(candidate_counts.values()),
        "candidate_count_max": max(candidate_counts.values()),
        "ids_with_multiple_candidates": sum(count > 1 for count in candidate_counts.values()),
        "match_method": "normalized_original_filename_then_lowest_rgb_pixel_mae",
        "pixel_mae_mean": (
            sum(item["pixel_mae"] for item in matched_images.values()) / len(matched_images)
            if matched_images else None
        ),
        "annotation_count": len(subset["annotations"]),
        "negative_image_count": negative_count,
        "category_count": len(categories),
        "split_list": split_list.relative_to(REPOSITORY_ROOT).as_posix(),
        "annotation_files": [
            str(path.relative_to(source))
            for path in annotation_files
        ],
        "output_annotation_sha256": hashlib.sha256(
            annotation_output.read_bytes()
        ).hexdigest(),
    }
    report_path = output_dir / "extraction_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"camera: {args.camera}")
    print(f"requested images: {len(requested)}")
    print(f"matched images: {len(matched_images)}")
    print(f"annotations: {len(subset['annotations'])}")
    print(f"negative images: {negative_count}")
    print(f"COCO subset: {annotation_output}")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
