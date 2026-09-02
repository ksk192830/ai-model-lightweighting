#!/usr/bin/env python3
"""Generate paper tables and figures from the current desktop engine summary.

This consumes only results produced by ``run_desktop_tensorrt_queue.py``.
Legacy ``paper_metrics.csv`` rows are never mixed with the current benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path("results/desktop-engine-summary.json")
FINAL = Path("results/results_final.csv")
FIGURE_DIR = Path("figures")
BASELINE_ID = "B01"


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def load_rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("rows") if isinstance(value, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No completed engine rows in {path}")
    if not any(row.get("experiment_id") == BASELINE_ID for row in rows):
        raise ValueError("Desktop engine summary has no B01 baseline row")
    return rows


def add_deltas(rows: list[dict[str, Any]]) -> None:
    baseline = next(row for row in rows if row["experiment_id"] == BASELINE_ID)
    for row in rows:
        row["engine_size_reduction_vs_B01_pct"] = 100.0 * (
            baseline["engine_size_bytes"] - row["engine_size_bytes"]
        ) / baseline["engine_size_bytes"]
        row["median_latency_reduction_vs_B01_pct"] = 100.0 * (
            baseline["median_ms"] - row["median_ms"]
        ) / baseline["median_ms"]
        row["fps_increase_vs_B01_pct"] = 100.0 * (
            row["fps"] - baseline["fps"]
        ) / baseline["fps"]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "experiment_id",
        "precision",
        "selected_structured_source",
        "engine",
        "engine_sha256",
        "engine_size_bytes",
        "engine_size_reduction_vs_B01_pct",
        "benchmark_image_count",
        "benchmark_scope",
        "mean_ms",
        "median_ms",
        "standard_deviation_ms",
        "iqr_ms",
        "p95_ms",
        "p99_ms",
        "coefficient_of_variation",
        "fps",
        "gpu_peak_allocated_bytes",
        "gpu_peak_reserved_bytes",
        "bbox_ap",
        "bbox_ap_delta_vs_B01",
        "mask_ap",
        "mask_ap_delta_vs_B01",
        "semantic_miou",
        "semantic_miou_delta_vs_B01",
        "median_latency_reduction_vs_B01_pct",
        "fps_increase_vs_B01_pct",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def style(ax) -> None:
    for name in ("top", "right"):
        ax.spines[name].set_visible(False)
    ax.grid(axis="y", alpha=0.22, linewidth=0.7)
    ax.set_axisbelow(True)


def bar_figure(
    rows: list[dict[str, Any]],
    field: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    ids = [row["experiment_id"] for row in rows]
    values = [float(row[field]) for row in rows]
    colors = ["#7b7b7b" if item == BASELINE_ID else "#2a78d6" for item in ids]
    fig, ax = plt.subplots(figsize=(8.0, 4.4), dpi=200)
    bars = ax.bar(ids, values, color=colors)
    style(ax)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def accuracy_figure(rows: list[dict[str, Any]], path: Path) -> None:
    ids = [row["experiment_id"] for row in rows]
    x = list(range(len(ids)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=200)
    ax.bar(
        [value - width / 2 for value in x],
        [row["bbox_ap"] for row in rows],
        width,
        label="BBox AP@[.50:.95]",
        color="#2a78d6",
    )
    ax.bar(
        [value + width / 2 for value in x],
        [row["mask_ap"] for row in rows],
        width,
        label="Mask AP@[.50:.95]",
        color="#1baf7a",
    )
    style(ax)
    ax.set_xticks(x, ids)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("COCO AP")
    ax.set_title("Accuracy on the fixed 437-image benchmark", loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def pareto_figure(rows: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=200)
    for row in rows:
        ax.scatter(row["median_ms"], row["mask_ap"], s=48, color="#2a78d6")
        ax.annotate(
            row["experiment_id"],
            (row["median_ms"], row["mask_ap"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    style(ax)
    ax.set_xlabel("End-to-end median latency (ms, lower is better)")
    ax.set_ylabel("Mask AP@[.50:.95] (higher is better)")
    ax.set_title("Accuracy-latency trade-off", loc="left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=FINAL)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    args = parser.parse_args()
    root = args.root.resolve()
    source = resolve(root, args.source)
    rows = load_rows(source)
    add_deltas(rows)
    output = resolve(root, args.output)
    figure_dir = resolve(root, args.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output, rows)
    bar_figure(
        rows,
        "median_ms",
        "Median latency (ms)",
        "End-to-end TensorRT latency (32 images, 200 runs)",
        figure_dir / "latency_comparison.png",
    )
    bar_figure(
        rows,
        "engine_size_bytes",
        "Engine size (bytes)",
        "TensorRT engine size",
        figure_dir / "size_comparison.png",
    )
    accuracy_figure(rows, figure_dir / "accuracy_comparison.png")
    pareto_figure(rows, figure_dir / "pareto_mask_latency.png")
    print(output)
    print(figure_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
