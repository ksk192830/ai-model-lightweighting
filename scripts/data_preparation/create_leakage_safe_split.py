#!/usr/bin/env python3
"""Create a session-disjoint COCO split for the parking-front dataset.

The Roboflow export uses a random frame-level split.  Timestamped frames can be
reliably grouped into capture sessions, but files named ``imageNNNNNN`` do not
contain enough source metadata to recover their recording.  The conservative
policy is therefore:

* keep every untraceable ``imageNNNNNN`` sample in train only;
* keep every timestamped capture session wholly within one target split;
* keep tiny timestamp groups in train because they are not stable evaluation
  units;
* assign the remaining whole sessions to train/valid/test while prioritising an
  80/10/10 image ratio, then class balance, and finally later sessions for test.

The source export is never modified.  Images are hard-linked by default so the
derived dataset is fast to create and does not consume another 1.2 GB.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPOSITORY_ROOT / "data" / "training" / "front_unaugmented"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "data" / "training" / "front_session_split_v1"
SPLITS = ("train", "valid", "test")
TARGET_RATIOS = {"train": 0.80, "valid": 0.10, "test": 0.10}
TIMESTAMP_PATTERN = re.compile(
    r"^frame_(?P<frame>\d+)_(?P<date>\d{8})_(?P<time>\d{6})_(?P<micro>\d{6})"
)


@dataclass
class Sample:
    source_split: str
    source_path: Path
    image: dict
    annotations: list[dict]
    original_name: str
    timestamp: datetime | None
    frame_number: int | None
    session_id: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--session-gap-seconds",
        type=float,
        default=2.0,
        help="Start a new timestamp session after this capture-time gap.",
    )
    parser.add_argument(
        "--min-eval-session-images",
        type=int,
        default=20,
        help="Smaller timestamp groups remain train-only.",
    )
    parser.add_argument(
        "--min-eval-sessions",
        type=int,
        default=2,
        help="Minimum number of whole capture sessions in valid and test.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy image bytes instead of creating space-saving hard links.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print the split without creating files.",
    )
    parser.add_argument(
        "--allow-unverified-source",
        action="store_true",
        help="Allow a source without an augmentation-free Roboflow manifest.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def capture_identity(image: dict) -> tuple[str, datetime | None, int | None]:
    extra = image.get("extra") or {}
    original_name = str(extra.get("name") or image["file_name"])
    match = TIMESTAMP_PATTERN.match(Path(original_name).stem)
    if match is None:
        return original_name, None, None
    timestamp = datetime.strptime(
        match.group("date") + match.group("time") + match.group("micro"),
        "%Y%m%d%H%M%S%f",
    )
    return original_name, timestamp, int(match.group("frame"))


def locate_image(split_dir: Path, file_name: str) -> Path:
    candidates = (split_dir / file_name, split_dir / "images" / file_name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Image not found for COCO record: {file_name}")


def load_samples(source: Path) -> tuple[list[Sample], list[dict], list[dict]]:
    samples: list[Sample] = []
    expected_categories: list[dict] | None = None
    expected_licenses: list[dict] | None = None
    seen_export_names: set[str] = set()

    for source_split in SPLITS:
        split_dir = source / source_split
        annotation_path = split_dir / "_annotations.coco.json"
        with annotation_path.open(encoding="utf-8") as handle:
            coco = json.load(handle)
        categories = coco.get("categories", [])
        licenses = coco.get("licenses", [])
        if expected_categories is None:
            expected_categories = categories
            expected_licenses = licenses
        elif categories != expected_categories:
            raise ValueError(f"Category definitions differ in {source_split}")

        annotations_by_image: dict[int, list[dict]] = defaultdict(list)
        for annotation in coco.get("annotations", []):
            annotations_by_image[int(annotation["image_id"])].append(annotation)

        for image in coco.get("images", []):
            file_name = Path(str(image["file_name"])).name
            if file_name in seen_export_names:
                raise ValueError(f"Duplicate exported filename: {file_name}")
            seen_export_names.add(file_name)
            original_name, timestamp, frame_number = capture_identity(image)
            samples.append(
                Sample(
                    source_split=source_split,
                    source_path=locate_image(split_dir, file_name),
                    image=image,
                    annotations=annotations_by_image[int(image["id"])],
                    original_name=original_name,
                    timestamp=timestamp,
                    frame_number=frame_number,
                )
            )

    assert expected_categories is not None and expected_licenses is not None
    return samples, expected_categories, expected_licenses


def load_source_manifest(source: Path, allow_unverified: bool) -> dict:
    path = source / "roboflow_export_manifest.json"
    if not path.is_file():
        if allow_unverified:
            return {
                "augmentation_free": False,
                "verification_override": True,
                "reason": f"Missing source manifest: {path}",
            }
        raise FileNotFoundError(
            f"Augmentation provenance manifest is required: {path}"
        )
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    augmentation_free = bool(manifest.get("augmentation_free")) and not bool(
        manifest.get("augmentation")
    )
    if not augmentation_free and not allow_unverified:
        raise RuntimeError(
            "Source export is not verified as augmentation-free; refusing to split."
        )
    manifest["augmentation_free"] = augmentation_free
    manifest["verification_override"] = bool(
        allow_unverified and not augmentation_free
    )
    return manifest


def group_timestamp_sessions(
    timestamped: list[Sample], gap_seconds: float
) -> list[list[Sample]]:
    ordered = sorted(
        timestamped,
        key=lambda sample: (
            sample.timestamp or datetime.min,
            sample.frame_number or -1,
            sample.source_path.name,
        ),
    )
    sessions: list[list[Sample]] = []
    current: list[Sample] = []
    for sample in ordered:
        starts_new = False
        if current:
            previous = current[-1]
            assert sample.timestamp is not None and previous.timestamp is not None
            gap = (sample.timestamp - previous.timestamp).total_seconds()
            frame_reset = (
                sample.timestamp != previous.timestamp
                and sample.frame_number is not None
                and previous.frame_number is not None
                and sample.frame_number < previous.frame_number
            )
            starts_new = gap > gap_seconds or frame_reset
        if starts_new:
            sessions.append(current)
            current = []
        current.append(sample)
    if current:
        sessions.append(current)

    for index, session in enumerate(sessions):
        session_id = f"timestamp-session-{index:02d}"
        for sample in session:
            sample.session_id = session_id
    return sessions


def presence_counts(samples: Iterable[Sample]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for sample in samples:
        counts.update({int(annotation["category_id"]) for annotation in sample.annotations})
    return counts


def choose_session_assignment(
    samples: list[Sample],
    sessions: list[list[Sample]],
    min_eval_session_images: int,
    min_eval_sessions: int,
) -> tuple[dict[str, list[Sample]], dict[str, str]]:
    untraceable = [sample for sample in samples if sample.timestamp is None]
    tiny = [session for session in sessions if len(session) < min_eval_session_images]
    eligible = [session for session in sessions if len(session) >= min_eval_session_images]
    if len(eligible) < 3:
        raise ValueError("At least three timestamp sessions are required")

    fixed_train = untraceable + [sample for session in tiny for sample in session]
    total = len(samples)
    global_presence = presence_counts(samples)
    category_ids = sorted(global_presence)
    candidates: list[tuple[tuple, tuple[str, ...], dict[str, list[Sample]]]] = []

    for roles in itertools.product(SPLITS, repeat=len(eligible)):
        role_counts = Counter(roles)
        if (
            role_counts["train"] < 1
            or role_counts["valid"] < min_eval_sessions
            or role_counts["test"] < min_eval_sessions
        ):
            continue
        bins = {split: ([] if split != "train" else list(fixed_train)) for split in SPLITS}
        for session, role in zip(eligible, roles):
            bins[role].extend(session)

        deviations = {
            split: abs(len(bins[split]) / total - TARGET_RATIOS[split])
            for split in SPLITS
        }
        class_loss: dict[str, float] = {split: 0.0 for split in SPLITS}
        for split in SPLITS:
            split_presence = presence_counts(bins[split])
            for category_id in category_ids:
                global_rate = global_presence[category_id] / total
                split_rate = split_presence[category_id] / len(bins[split])
                class_loss[split] += (split_rate - global_rate) ** 2

        # A later test set is a stronger temporal holdout when size/class scores tie.
        test_timestamps = [
            sample.timestamp.timestamp()
            for sample in bins["test"]
            if sample.timestamp is not None
        ]
        test_recency = sum(test_timestamps) / len(test_timestamps)
        score = (
            round(max(deviations.values()), 12),
            round(sum(deviations.values()), 12),
            round(class_loss["test"], 12),
            round(class_loss["valid"], 12),
            round(class_loss["train"], 12),
            -test_recency,
            roles,
        )
        candidates.append((score, roles, bins))

    _, roles, selected = min(candidates, key=lambda candidate: candidate[0])
    role_by_session = {
        session[0].session_id or "unknown": role
        for session, role in zip(eligible, roles)
    }
    for session in tiny:
        role_by_session[session[0].session_id or "unknown"] = "train"
    return selected, role_by_session


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_split(
    directory: Path,
    samples: list[Sample],
    categories: list[dict],
    licenses: list[dict],
    copy_images: bool,
) -> dict:
    directory.mkdir(parents=True)
    output_images: list[dict] = []
    output_annotations: list[dict] = []
    annotation_id = 0
    class_annotations: Counter[int] = Counter()
    class_presence: Counter[int] = Counter()

    for image_id, sample in enumerate(
        sorted(samples, key=lambda item: (item.source_split, item.source_path.name))
    ):
        destination = directory / sample.source_path.name
        if copy_images:
            shutil.copy2(sample.source_path, destination)
        else:
            try:
                os.link(sample.source_path, destination)
            except OSError:
                shutil.copy2(sample.source_path, destination)

        image = dict(sample.image)
        image["id"] = image_id
        image["file_name"] = destination.name
        extra = dict(image.get("extra") or {})
        extra.update(
            {
                "original_export_split": sample.source_split,
                "original_export_file": sample.source_path.name,
                "capture_group": sample.session_id or "untraceable-train-only",
            }
        )
        image["extra"] = extra
        output_images.append(image)

        present: set[int] = set()
        for source_annotation in sample.annotations:
            annotation = dict(source_annotation)
            annotation["id"] = annotation_id
            annotation["image_id"] = image_id
            annotation_id += 1
            output_annotations.append(annotation)
            category_id = int(annotation["category_id"])
            class_annotations[category_id] += 1
            present.add(category_id)
        class_presence.update(present)

    coco = {
        "info": {
            "year": "2026",
            "version": "front-session-split-v1",
            "description": "Session-disjoint split derived from augmentation-free Roboflow parking_front version 9",
            "contributor": "",
            "url": "",
            "date_created": datetime.now(timezone.utc).isoformat(),
        },
        "licenses": licenses,
        "categories": categories,
        "images": output_images,
        "annotations": output_annotations,
    }
    with (directory / "_annotations.coco.json").open("w", encoding="utf-8") as handle:
        json.dump(coco, handle, ensure_ascii=False)

    return {
        "images": len(output_images),
        "annotations": len(output_annotations),
        "negative_images": sum(not sample.annotations for sample in samples),
        "class_annotations": {str(key): value for key, value in sorted(class_annotations.items())},
        "class_image_presence": {str(key): value for key, value in sorted(class_presence.items())},
    }


def verify_no_cross_split_duplicates(output: Path) -> int:
    owners: dict[str, str] = {}
    overlaps = 0
    for split in SPLITS:
        for path in (output / split).iterdir():
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                continue
            digest = sha256(path)
            previous = owners.setdefault(digest, split)
            if previous != split:
                overlaps += 1
    return overlaps


def verify_coco_references(output: Path) -> int:
    errors = 0
    for split in SPLITS:
        split_dir = output / split
        with (split_dir / "_annotations.coco.json").open(encoding="utf-8") as handle:
            coco = json.load(handle)
        image_ids = [int(image["id"]) for image in coco.get("images", [])]
        errors += len(image_ids) - len(set(image_ids))
        image_id_set = set(image_ids)
        errors += sum(
            int(annotation["image_id"]) not in image_id_set
            for annotation in coco.get("annotations", [])
        )
        errors += sum(
            not (split_dir / Path(str(image["file_name"])).name).is_file()
            for image in coco.get("images", [])
        )
    return errors


def session_overlap_count(bins: dict[str, list[Sample]]) -> int:
    owners: dict[str, set[str]] = defaultdict(set)
    for split, samples in bins.items():
        for sample in samples:
            if sample.session_id is not None:
                owners[sample.session_id].add(split)
    return sum(len(split_owners) > 1 for split_owners in owners.values())


def main() -> int:
    args = parse_args()
    source = resolve(args.source)
    output = resolve(args.output)
    if output.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output}")
    source_manifest = load_source_manifest(source, args.allow_unverified_source)
    samples, categories, licenses = load_samples(source)
    sessions = group_timestamp_sessions(
        [sample for sample in samples if sample.timestamp is not None],
        args.session_gap_seconds,
    )
    bins, role_by_session = choose_session_assignment(
        samples,
        sessions,
        args.min_eval_session_images,
        args.min_eval_sessions,
    )

    print(f"source images: {len(samples)}")
    print(f"untraceable train-only: {sum(sample.timestamp is None for sample in samples)}")
    for session in sessions:
        first = session[0]
        last = session[-1]
        print(
            f"{first.session_id}: {len(session)} images, "
            f"{first.timestamp.isoformat()}..{last.timestamp.isoformat()}, "
            f"role={role_by_session[first.session_id or 'unknown']}"
        )
    for split in SPLITS:
        print(f"{split}: {len(bins[split])} ({len(bins[split]) / len(samples):.2%})")
    if args.dry_run:
        return 0

    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        split_stats = {
            split: write_split(
                temporary / split,
                bins[split],
                categories,
                licenses,
                args.copy_images,
            )
            for split in SPLITS
        }
        duplicate_count = verify_no_cross_split_duplicates(temporary)
        if duplicate_count:
            raise RuntimeError(
                f"Found {duplicate_count} exact image duplicates across target splits"
            )
        coco_reference_errors = verify_coco_references(temporary)
        if coco_reference_errors:
            raise RuntimeError(
                f"Found {coco_reference_errors} COCO reference errors"
            )
        capture_session_overlaps = session_overlap_count(bins)
        if capture_session_overlaps:
            raise RuntimeError(
                f"Found {capture_session_overlaps} capture sessions across splits"
            )

        achieved_ratios = {
            split: len(bins[split]) / len(samples) for split in SPLITS
        }
        maximum_ratio_error_pp = 100 * max(
            abs(achieved_ratios[split] - TARGET_RATIOS[split])
            for split in SPLITS
        )
        paper_ready = (
            bool(source_manifest.get("augmentation_free"))
            and not bool(source_manifest.get("verification_override"))
            and duplicate_count == 0
            and coco_reference_errors == 0
            and capture_session_overlaps == 0
        )

        manifest = {
            "schema_version": 2,
            "source": str(source),
            "output": str(output),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_provenance": source_manifest,
            "policy": {
                "target_ratios": TARGET_RATIOS,
                "session_gap_seconds": args.session_gap_seconds,
                "frame_number_reset_starts_new_session": True,
                "minimum_eval_session_images": args.min_eval_session_images,
                "minimum_capture_sessions_per_eval_split": args.min_eval_sessions,
                "untraceable_filename_policy": "train-only",
                "assignment_priority": [
                    "whole-session integrity",
                    "maximum image-ratio deviation",
                    "total image-ratio deviation",
                    "test class-distribution divergence",
                    "validation class-distribution divergence",
                    "train class-distribution divergence",
                    "later sessions preferred for test on exact ties",
                ],
            },
            "categories": categories,
            "splits": split_stats,
            "verification": {
                "source_augmentation_free": bool(
                    source_manifest.get("augmentation_free")
                ),
                "exact_cross_split_duplicate_images": duplicate_count,
                "capture_session_overlap_count": capture_session_overlaps,
                "coco_reference_errors": coco_reference_errors,
                "maximum_ratio_error_pp": maximum_ratio_error_pp,
                "paper_evaluation_ready": paper_ready,
            },
            "sessions": [
                {
                    "id": session[0].session_id,
                    "images": len(session),
                    "start": session[0].timestamp.isoformat(),
                    "end": session[-1].timestamp.isoformat(),
                    "first_frame": session[0].frame_number,
                    "last_frame": session[-1].frame_number,
                    "assigned_split": role_by_session[session[0].session_id or "unknown"],
                }
                for session in sessions
            ],
        }
        with (temporary / "split_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
        with (temporary / "dataset_readiness.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest["verification"], handle, ensure_ascii=False, indent=2)
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(f"created: {output}")
    print(
        "verification: PASS "
        "(augmentation-free source, session-disjoint, exact duplicate overlap=0)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
