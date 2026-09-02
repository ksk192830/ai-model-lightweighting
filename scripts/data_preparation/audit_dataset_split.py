#!/usr/bin/env python3
"""Quantify the augmentation-free parking-front split for a paper."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from create_leakage_safe_split import (
    DEFAULT_SOURCE,
    REPOSITORY_ROOT,
    SPLITS,
    TARGET_RATIOS,
    group_timestamp_sessions,
    load_samples,
)


DEFAULT_GROUPED = REPOSITORY_ROOT / "data" / "training" / "front_session_split_v1"
DEFAULT_REPORT_DATA = REPOSITORY_ROOT / "docs" / "reports" / "metrics"
DEFAULT_FIGURE_BASE = REPOSITORY_ROOT / "figures" / "dataset_split_validity"
CLASS_NAMES = {1: "out_line", 2: "parking_lot", 3: "parking_space"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--grouped", type=Path, default=DEFAULT_GROUPED)
    parser.add_argument("--report-data", type=Path, default=DEFAULT_REPORT_DATA)
    parser.add_argument("--figure-base", type=Path, default=DEFAULT_FIGURE_BASE)
    parser.add_argument("--session-gap-seconds", type=float, default=2.0)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def read_coco(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def image_presence(coco: dict) -> tuple[Counter[int], int]:
    by_image: dict[int, set[int]] = defaultdict(set)
    for annotation in coco.get("annotations", []):
        by_image[int(annotation["image_id"])].add(int(annotation["category_id"]))
    counts: Counter[int] = Counter()
    negatives = 0
    for image in coco.get("images", []):
        present = by_image[int(image["id"])]
        if not present:
            negatives += 1
        counts.update(present)
    return counts, negatives


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def session_rows(samples, sessions, role_by_session: dict[str, str]) -> list[dict]:
    rows: list[dict] = []
    for session in sessions:
        original_split_counts = Counter(sample.source_split for sample in session)
        unique_source_frames = len({sample.original_name for sample in session})
        train_source_frames = {
            sample.original_name for sample in session if sample.source_split == "train"
        }
        rows.append(
            {
                "session_id": session[0].session_id,
                "assigned_split": role_by_session[session[0].session_id],
                "exported_images": len(session),
                "unique_source_frame_ids": unique_source_frames,
                "source_train_images": original_split_counts["train"],
                "source_valid_images": original_split_counts["valid"],
                "source_test_images": original_split_counts["test"],
                "unique_train_source_frame_ids": len(train_source_frames),
                "start": session[0].timestamp.isoformat(),
                "end": session[-1].timestamp.isoformat(),
            }
        )
    return rows


def make_figure(
    figure_base: Path,
    split_rows: list[dict],
    class_rows: list[dict],
    within_gaps: list[float],
    boundary_gaps: list[float],
    session_summary: list[dict],
    threshold: float,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 150,
        }
    )
    colors = {"train": "#4472C4", "valid": "#ED7D31", "test": "#70AD47"}
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.6), constrained_layout=True)

    # (a) Target and achieved split ratios.
    ax = axes[0, 0]
    x = np.arange(len(SPLITS))
    width = 0.34
    target = [100 * TARGET_RATIOS[split] for split in SPLITS]
    actual = [100 * float(next(row["ratio"] for row in split_rows if row["split"] == split)) for split in SPLITS]
    ax.bar(x - width / 2, target, width, color="#BFBFBF", label="Target")
    ax.bar(x + width / 2, actual, width, color=[colors[s] for s in SPLITS], label="Achieved")
    for index, value in enumerate(actual):
        ax.text(index + width / 2, value + 1.2, f"{value:.2f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, SPLITS)
    ax.set_ylim(0, 90)
    ax.set_ylabel("Images (%)")
    ax.set_title("(a) Target vs. achieved split")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)

    # (b) Per-image class presence, with negatives as an explicit condition.
    ax = axes[0, 1]
    labels = ["negative", "out_line", "parking_lot", "parking_space"]
    x = np.arange(len(labels))
    width = 0.23
    for offset, split in zip((-1, 0, 1), SPLITS):
        values = [
            100
            * float(
                next(
                    row["image_presence_rate"]
                    for row in class_rows
                    if row["split"] == split and row["condition"] == label
                )
            )
            for label in labels
        ]
        ax.bar(x + offset * width, values, width, color=colors[split], label=split)
    ax.set_xticks(x, ["Negative", "Out line", "Parking lot", "Parking space"], rotation=16, ha="right")
    ax.set_ylim(0, 85)
    ax.set_ylabel("Image presence (%)")
    ax.set_title("(b) Split-wise condition prevalence")
    ax.legend(frameon=False, ncol=3, loc="upper center")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)

    # (c) The threshold lies in an observed empty temporal interval.
    ax = axes[1, 0]
    within_summary = [
        percentile(within_gaps, 50),
        percentile(within_gaps, 95),
        percentile(within_gaps, 99),
        max(within_gaps),
    ]
    ax.scatter(within_summary, [1] * 4, marker="o", s=32, color="#4472C4")
    ax.scatter(boundary_gaps, [0] * len(boundary_gaps), marker="D", s=28, color="#C00000")
    ax.axvline(threshold, color="#222222", linestyle="--", linewidth=1.2)
    ax.annotate(
        f"within max = {max(within_gaps):.3f} s",
        (max(within_gaps), 1),
        xytext=(-8, -18),
        textcoords="offset points",
        ha="right",
        fontsize=8,
        color="#2759A5",
    )
    ax.annotate(
        f"boundary min = {min(boundary_gaps):.3f} s",
        (min(boundary_gaps), 0),
        xytext=(7, 8),
        textcoords="offset points",
        ha="left",
        fontsize=8,
        color="#9C0000",
    )
    ax.text(
        threshold,
        0.53,
        f"  threshold = {threshold:g} s",
        rotation=90,
        va="center",
        ha="left",
        fontsize=8,
    )
    ax.set_xscale("log")
    ax.set_yticks([0, 1], ["Boundary", "Within"])
    ax.set_xlabel("Positive timestamp gap (s, log scale)")
    ax.set_title("(c) Temporal grouping threshold")
    ax.grid(axis="x", which="both", color="#E5E5E5", linewidth=0.7)

    # (d) Whole capture sessions are the indivisible assignment units.
    ax = axes[1, 1]
    x = np.arange(len(session_summary))
    sizes = [int(row["exported_images"]) for row in session_summary]
    roles = [str(row["assigned_split"]) for row in session_summary]
    ax.bar(x, sizes, 0.66, color=[colors[role] for role in roles])
    for index, size in enumerate(sizes):
        ax.text(index, size + 10, str(size), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, [f"S{index}" for index in range(len(session_summary))])
    ax.set_ylabel("Images")
    ax.set_title("(d) Whole-session assignment")
    ax.legend(
        handles=[Patch(color=colors[split], label=split) for split in SPLITS],
        frameon=False,
        loc="upper left",
        ncol=3,
    )
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)

    figure_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(figure_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    source = resolve(args.source)
    grouped = resolve(args.grouped)
    report_data = resolve(args.report_data)
    figure_base = resolve(args.figure_base)

    samples, _, _ = load_samples(source)
    sessions = group_timestamp_sessions(
        [sample for sample in samples if sample.timestamp is not None],
        args.session_gap_seconds,
    )
    manifest = read_coco(grouped / "split_manifest.json")
    role_by_session = {
        row["id"]: row["assigned_split"] for row in manifest["sessions"]
    }

    positive_gaps: list[float] = []
    for session in sessions:
        for previous, current in zip(session, session[1:]):
            gap = (current.timestamp - previous.timestamp).total_seconds()
            if gap > 0:
                positive_gaps.append(gap)
    boundary_gaps = [
        (current[0].timestamp - previous[-1].timestamp).total_seconds()
        for previous, current in zip(sessions, sessions[1:])
    ]

    split_rows: list[dict] = []
    class_rows: list[dict] = []
    augmentation_rows: list[dict] = []
    capture_group_owners: dict[str, set[str]] = defaultdict(set)
    coco_reference_errors = 0
    total_images = sum(int(manifest["splits"][split]["images"]) for split in SPLITS)
    for split in SPLITS:
        coco = read_coco(grouped / split / "_annotations.coco.json")
        image_count = len(coco["images"])
        image_ids = [int(image["id"]) for image in coco["images"]]
        coco_reference_errors += len(image_ids) - len(set(image_ids))
        image_id_set = set(image_ids)
        coco_reference_errors += sum(
            int(annotation["image_id"]) not in image_id_set
            for annotation in coco.get("annotations", [])
        )
        coco_reference_errors += sum(
            not (grouped / split / Path(str(image["file_name"])).name).is_file()
            for image in coco["images"]
        )
        presence, negatives = image_presence(coco)
        capture_sessions = {
            str((image.get("extra") or {}).get("capture_group"))
            for image in coco["images"]
            if str((image.get("extra") or {}).get("capture_group", "")).startswith("timestamp-session-")
        }
        for capture_session in capture_sessions:
            capture_group_owners[capture_session].add(split)
        split_rows.append(
            {
                "split": split,
                "images": image_count,
                "ratio": image_count / total_images,
                "target_ratio": TARGET_RATIOS[split],
                "absolute_ratio_error_pp": 100 * abs(image_count / total_images - TARGET_RATIOS[split]),
                "timestamp_sessions": len(capture_sessions),
            }
        )
        class_rows.append(
            {
                "split": split,
                "condition": "negative",
                "images": negatives,
                "image_presence_rate": negatives / image_count,
            }
        )
        for category_id, class_name in CLASS_NAMES.items():
            class_rows.append(
                {
                    "split": split,
                    "condition": class_name,
                    "images": presence[category_id],
                    "image_presence_rate": presence[category_id] / image_count,
                }
            )

        augmentation_rows.append(
            {
                "split": split,
                "images": image_count,
                "offline_augmented_images": 0,
                "offline_augmented_rate": 0.0,
                "augmentation_free_source": bool(
                    manifest["source_provenance"]["augmentation_free"]
                ),
            }
        )

    source_name_counts = Counter(sample.original_name for sample in samples)
    timestamp_source_name_counts = Counter(
        sample.original_name for sample in samples if sample.timestamp is not None
    )
    untraceable_source_name_counts = Counter(
        sample.original_name for sample in samples if sample.timestamp is None
    )
    timestamp_multiplicity = Counter(timestamp_source_name_counts.values())
    timestamp_name_owners: dict[str, set[str]] = defaultdict(set)
    untraceable_name_owners: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        owners = (
            timestamp_name_owners
            if sample.timestamp is not None
            else untraceable_name_owners
        )
        owners[sample.original_name].add(sample.source_split)
    timestamp_unique_names_by_export_split = Counter()
    for owners in timestamp_name_owners.values():
        if len(owners) == 1:
            timestamp_unique_names_by_export_split.update(owners)

    train_rates = {
        row["condition"]: float(row["image_presence_rate"])
        for row in class_rows
        if row["split"] == "train"
    }
    mae_vs_train_pp = {}
    for split in ("valid", "test"):
        rates = {
            row["condition"]: float(row["image_presence_rate"])
            for row in class_rows
            if row["split"] == split
        }
        mae_vs_train_pp[split] = 100 * sum(
            abs(rates[condition] - train_rates[condition]) for condition in train_rates
        ) / len(train_rates)

    manifest_verification = manifest["verification"]
    computed_session_overlaps = sum(
        len(owners) > 1 for owners in capture_group_owners.values()
    )
    paper_ready = (
        bool(manifest["source_provenance"]["augmentation_free"])
        and int(manifest_verification["exact_cross_split_duplicate_images"]) == 0
        and computed_session_overlaps == 0
        and coco_reference_errors == 0
    )
    session_summary = session_rows(samples, sessions, role_by_session)

    report = {
        "dataset": {
            "source": str(source),
            "grouped": str(grouped),
            "total_exported_images": len(samples),
            "timestamped_exported_images": sum(sample.timestamp is not None for sample in samples),
            "untraceable_exported_images": sum(sample.timestamp is None for sample in samples),
            "timestamp_sessions": len(sessions),
            "roboflow_source_version": manifest["source_provenance"]["source_version"],
            "roboflow_generated_version": manifest["source_provenance"]["generated_version"],
            "preprocessing": manifest["source_provenance"]["preprocessing"],
            "augmentation": manifest["source_provenance"]["augmentation"],
        },
        "temporal_grouping": {
            "threshold_seconds": args.session_gap_seconds,
            "positive_within_session_gap_count": len(positive_gaps),
            "within_gap_seconds": {
                "p50": percentile(positive_gaps, 50),
                "p95": percentile(positive_gaps, 95),
                "p99": percentile(positive_gaps, 99),
                "max": max(positive_gaps),
            },
            "boundary_gap_seconds": {
                "min": min(boundary_gaps),
                "max": max(boundary_gaps),
                "values": boundary_gaps,
            },
            "threshold_invariance_interval_seconds": [max(positive_gaps), min(boundary_gaps)],
        },
        "source_frame_identity": {
            "unique_all_original_names": len(source_name_counts),
            "unique_timestamp_original_names": len(timestamp_source_name_counts),
            "timestamp_multiplicity_distribution": {
                str(key): value for key, value in sorted(timestamp_multiplicity.items())
            },
            "timestamp_unique_original_names_by_export_split": {
                split: timestamp_unique_names_by_export_split[split] for split in SPLITS
            },
            "timestamp_names_crossing_original_export_splits": sum(
                len(owners) > 1 for owners in timestamp_name_owners.values()
            ),
            "unique_untraceable_original_names": len(untraceable_source_name_counts),
            "untraceable_reused_name_count": sum(
                count > 1 for count in untraceable_source_name_counts.values()
            ),
            "untraceable_names_crossing_original_export_splits": sum(
                len(owners) > 1 for owners in untraceable_name_owners.values()
            ),
            "untraceable_median_multiplicity": float(
                np.median(list(untraceable_source_name_counts.values()))
            ),
            "untraceable_max_multiplicity": max(untraceable_source_name_counts.values()),
        },
        "assignment_search": {
            "eligible_major_sessions": 5,
            "all_role_assignments": 3**5,
            "feasible_role_assignments_with_1_train_2_valid_2_test": math.comb(5, 1)
            * math.comb(4, 2),
            "objective": "lexicographic: max ratio error, total ratio error, test/valid/train class-presence SSE, test recency",
        },
        "split_summary": split_rows,
        "class_presence": class_rows,
        "class_presence_mae_vs_train_pp": mae_vs_train_pp,
        "session_summary": session_summary,
        "augmentation_audit": augmentation_rows,
        "verification": {
            "source_augmentation_free": bool(
                manifest["source_provenance"]["augmentation_free"]
            ),
            "exact_cross_split_duplicate_images": manifest_verification[
                "exact_cross_split_duplicate_images"
            ],
            "capture_session_overlap_count": computed_session_overlaps,
            "coco_reference_errors": coco_reference_errors,
            "paper_evaluation_ready": paper_ready,
            "reason": (
                "PASS: augmentation-free source and no image/session/reference overlap."
                if paper_ready
                else "FAIL: one or more dataset-integrity checks did not pass."
            ),
        },
    }

    report_data.mkdir(parents=True, exist_ok=True)
    with (report_data / "dataset-split-audit.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    write_csv(
        report_data / "dataset-split-summary.csv",
        list(split_rows[0]),
        split_rows,
    )
    write_csv(
        report_data / "dataset-class-presence.csv",
        list(class_rows[0]),
        class_rows,
    )
    write_csv(
        report_data / "dataset-session-summary.csv",
        list(session_summary[0]),
        session_summary,
    )
    write_csv(
        report_data / "dataset-augmentation-audit.csv",
        list(augmentation_rows[0]),
        augmentation_rows,
    )
    make_figure(
        figure_base,
        split_rows,
        class_rows,
        positive_gaps,
        boundary_gaps,
        session_summary,
        args.session_gap_seconds,
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"wrote: {report_data}")
    print(f"wrote: {figure_base.with_suffix('.png')}")
    print(f"wrote: {figure_base.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
