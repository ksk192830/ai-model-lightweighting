"""Generate paper-ready outputs from results/paper_metrics.csv.

Outputs:
  results/results_final.csv  cleaned per-model metrics + deltas vs B01
  figures/*.png              bar charts (FPS, latency, size, mAP) and
                             Pareto scatter plots (mAP vs FPS/size/latency)

Usage:
  .venv/bin/python scripts/reporting/paper_results.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results" / "paper_metrics.csv"
FINAL = ROOT / "results" / "results_final.csv"
FIGDIR = ROOT / "figures"

BASELINE_ID = "B01"

# Experiment ID -> paper naming. Input size is repeated in the label because
# R01 (432) and the Ultralytics model (512) are not directly comparable to the
# 504x504 TensorRT candidates.
MODELS = {
    "B01": ("Baseline FP32", "TensorRT FP32 baseline"),
    "B02": ("FP16", "TensorRT FP16 quantization"),
    "B03": ("INT8 PTQ", "TensorRT INT8 post-training quantization"),
    "C01": ("Structured+FP16", "Structured decoder pruning + FP16"),
    "M01": ("2:4 Control", "2:4 sparsity dense control"),
    "M02": ("2:4 Sparse", "NVIDIA 2:4 structured sparsity"),
    "S01": ("Decoder-Pruned", "Structured decoder layer pruning (FP32)"),
    "R01": ("Low-Res FP16", "Reduced input resolution + FP16"),
    "parking_front": ("Ultralytics PT", "Legacy Ultralytics PyTorch model"),
}

# The paper's main comparison is restricted to RF-DETR TensorRT candidates.
# The legacy Ultralytics checkpoint remains in the raw CSV as a reference.
REFERENCE_MODELS = {"parking_front"}

# Validated light-mode palette (dataviz reference instance).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
AQUA = "#1baf7a"
ORANGE = "#d97706"


def experiment_id(model_path: str) -> str:
    parts = Path(model_path).parts
    if "experiments" in parts:
        return parts[parts.index("experiments") + 1]
    return Path(model_path).stem


def load_rows() -> list[dict]:
    rows = []
    with SOURCE.open() as fh:
        for raw in csv.DictReader(fh):
            if raw["Status"] != "ok":
                continue
            exp = experiment_id(raw["Model Path"])
            name, method = MODELS[exp]
            rows.append(
                {
                    "id": exp,
                    "name": name,
                    "method": method,
                    "imgsz": int(raw["Image Size"]),
                    "backend": raw["Backend"],
                    "precision": float(raw["Precision"]),
                    "recall": float(raw["Recall"]),
                    "map50": float(raw["mAP50"]),
                    "map5095": float(raw["mAP50-95"]),
                    "mask_ap": float(raw["Mask AP"]),
                    "mask_ap50": float(raw["Mask AP50"]),
                    "mask_ap75": float(raw["Mask AP75"]),
                    "mask_miou": float(raw["Mask mIoU"]),
                    "fps": float(raw["FPS"]),
                    "latency": float(raw["Latency(ms)"]),
                    "latency_p95": float(raw["Latency P95(ms)"]),
                    "size": float(raw["Size(MB)"]),
                    "gpu_peak": float(raw["GPU Process Peak(MB)"]),
                }
            )
    return rows


def add_deltas(rows: list[dict]) -> None:
    base = next(r for r in rows if r["id"] == BASELINE_ID)
    for r in rows:
        r["size_reduction"] = 100.0 * (base["size"] - r["size"]) / base["size"]
        r["fps_increase"] = 100.0 * (r["fps"] - base["fps"]) / base["fps"]
        r["latency_reduction"] = (
            100.0 * (base["latency"] - r["latency"]) / base["latency"]
        )
        r["map_delta"] = r["map5095"] - base["map5095"]


def write_final_csv(rows: list[dict]) -> None:
    header = [
        "ID",
        "Model",
        "Method",
        "Input Size",
        "Backend",
        "Precision",
        "Recall",
        "mAP50",
        "mAP50-95",
        "Mask AP",
        "Mask AP50",
        "Mask AP75",
        "Mask mIoU",
        "FPS",
        "Latency(ms)",
        "Latency P95(ms)",
        "Size(MB)",
        "GPU Peak(MB)",
        "Size Reduction vs B01(%)",
        "FPS Increase vs B01(%)",
        "Latency Reduction vs B01(%)",
        "mAP50-95 Delta vs B01",
        "Note",
    ]
    with FINAL.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for r in rows:
            note = ""
            if r["imgsz"] != 504:
                note = f"input size {r['imgsz']} (not directly comparable to 504)"
            writer.writerow(
                [
                    r["id"],
                    r["name"],
                    r["method"],
                    r["imgsz"],
                    r["backend"],
                    f"{r['precision']:.4f}",
                    f"{r['recall']:.4f}",
                    f"{r['map50']:.4f}",
                    f"{r['map5095']:.4f}",
                    f"{r['mask_ap']:.4f}",
                    f"{r['mask_ap50']:.4f}",
                    f"{r['mask_ap75']:.4f}",
                    f"{r['mask_miou']:.4f}",
                    f"{r['fps']:.2f}",
                    f"{r['latency']:.2f}",
                    f"{r['latency_p95']:.2f}",
                    f"{r['size']:.2f}",
                    f"{r['gpu_peak']:.0f}",
                    f"{r['size_reduction']:.2f}",
                    f"{r['fps_increase']:.2f}",
                    f"{r['latency_reduction']:.2f}",
                    f"{r['map_delta']:+.4f}",
                    note,
                ]
            )


def label(r: dict) -> str:
    tag = f"{r['name']} ({r['imgsz']})"
    return tag + " *" if r["id"] == BASELINE_ID else tag


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(colors=SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(SECONDARY)
    ax.yaxis.label.set_color(SECONDARY)
    ax.title.set_color(INK)


def bar_chart(
    rows, key, title, xlabel, filename, ascending, fmt, reference_ids=None
):
    reference_ids = set(reference_ids or ())
    data = sorted(rows, key=lambda r: r[key], reverse=not ascending)
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    names = [label(r) for r in data]
    values = [r[key] for r in data]
    colors = [
        ORANGE
        if r["id"] in reference_ids
        else MUTED
        if r["id"] == BASELINE_ID
        else BLUE
        for r in data
    ]
    bars = ax.barh(names, values, color=colors, height=0.62, zorder=3)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=12, loc="left", pad=12)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    span = max(values) - min(0, min(values))
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + span * 0.012,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(value),
            va="center",
            fontsize=8.5,
            color=SECONDARY,
        )
    ax.set_xlim(0, max(values) * 1.12)
    fig.text(0.01, 0.01, "* baseline (B01)", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIGDIR / filename, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def map_chart(rows, filename):
    data = sorted(rows, key=lambda r: r["map5095"], reverse=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    names = [label(r) for r in data]
    y = range(len(data))
    h = 0.36
    map50_bars = ax.barh(
        [i - h / 2 for i in y],
        [r["map50"] for r in data],
        height=h - 0.04,
        color=AQUA,
        label="mAP@50",
        zorder=3,
    )
    map5095_bars = ax.barh(
        [i + h / 2 for i in y],
        [r["map5095"] for r in data],
        height=h - 0.04,
        color=BLUE,
        label="mAP@50-95",
        zorder=3,
    )
    for bars, metric in ((map50_bars, "map50"), (map5095_bars, "map5095")):
        for bar, r in zip(bars, data):
            if r["id"] in REFERENCE_MODELS:
                bar.set_edgecolor(ORANGE)
                bar.set_linewidth(2.0)
                bar.set_hatch("///")
    ax.set_yticks(list(y), names)
    ax.invert_yaxis()
    ax.set_xlabel("COCO mAP", fontsize=10)
    ax.set_xlim(0, 1.12)
    ax.set_title(
        "Detection accuracy by model (test set, 296 images)",
        fontsize=12,
        loc="left",
        pad=12,
    )
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for i, r in enumerate(data):
        ax.text(
            r["map50"] + 0.012,
            i - h / 2,
            f"{r['map50']:.3f}",
            va="center",
            fontsize=8,
            color=SECONDARY,
        )
        ax.text(
            r["map5095"] + 0.012,
            i + h / 2,
            f"{r['map5095']:.3f}",
            va="center",
            fontsize=8,
            color=SECONDARY,
        )
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="white", edgecolor=ORANGE, hatch="///", linewidth=2))
    labels.append("Ultralytics reference")
    legend = ax.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=9,
        frameon=False,
        labelcolor=SECONDARY,
    )
    for text in legend.get_texts():
        text.set_color(SECONDARY)
    fig.text(0.01, 0.01, "* baseline (B01)", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 0.82, 1))
    fig.savefig(FIGDIR / filename, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def mask_chart(rows, filename):
    data = sorted(rows, key=lambda r: r["mask_ap"], reverse=True)
    fig, ax = plt.subplots(figsize=(8.2, 5.1), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    names = [label(r) for r in data]
    y = np.arange(len(data))
    width = 0.24
    metrics = (
        ("mask_ap50", "Mask AP50", AQUA),
        ("mask_ap", "Mask AP", BLUE),
        ("mask_ap75", "Mask AP75", MUTED),
    )
    for offset, (key, legend, color) in zip((-width, 0, width), metrics):
        bars = ax.barh(
            y + offset,
            [r[key] for r in data],
            height=width - 0.03,
            color=color,
            label=legend,
            zorder=3,
        )
        for bar, r in zip(bars, data):
            if r["id"] in REFERENCE_MODELS:
                bar.set_edgecolor(ORANGE)
                bar.set_linewidth(2.0)
                bar.set_hatch("///")
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel("COCO mask AP", fontsize=10)
    ax.set_xlim(0, 0.92)
    ax.set_title(
        "Segmentation accuracy by model (test set, 296 images)",
        fontsize=12,
        loc="left",
        pad=12,
    )
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="white", edgecolor=ORANGE, hatch="///", linewidth=2))
    labels.append("Ultralytics reference")
    ax.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=8.5,
        frameon=False,
    )
    fig.text(
        0.01,
        0.01,
        "Orange hatch = Ultralytics reference. * baseline (B01)",
        fontsize=8,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.03, 0.82, 1))
    fig.savefig(FIGDIR / filename, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def pareto_front(points, maximize_x):
    """points: (x, y, row); y (accuracy) is always maximized."""
    front = []
    for x, y, r in points:
        dominated = any(
            (ox >= x if maximize_x else ox <= x)
            and oy >= y
            and (ox, oy) != (x, y)
            for ox, oy, _ in points
        )
        if not dominated:
            front.append((x, y, r))
    front.sort(key=lambda p: p[0])
    return front


def pareto_chart(
    rows,
    xkey,
    xlabel,
    maximize_x,
    title,
    filename,
    offsets=None,
    reference_ids=None,
    ykey="map5095",
    ylabel="mAP@50-95",
):
    data = rows
    reference_ids = set(reference_ids or ())
    points = [(r[xkey], r[ykey], r) for r in data]
    front = pareto_front(points, maximize_x)
    front_ids = {r["id"] for _, _, r in front}

    fig, ax = plt.subplots(figsize=(6.8, 4.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    fx = [p[0] for p in front]
    fy = [p[1] for p in front]
    step_where = "post" if maximize_x else "pre"
    ax.step(
        fx, fy, where=step_where, color=BLUE, linewidth=1.4,
        linestyle=(0, (4, 3)), alpha=0.6, zorder=2,
    )
    offsets = offsets or {}
    for x, y, r in points:
        on_front = r["id"] in front_ids
        is_reference = r["id"] in reference_ids
        color = ORANGE if is_reference else BLUE if on_front else MUTED
        ax.scatter(
            x, y, s=76 if is_reference else 64, color=color, zorder=3,
            marker="D" if is_reference else "o",
            edgecolors=SURFACE, linewidths=1.5,
        )
        dx, dy = offsets.get(r["id"], (0, 10))
        ax.annotate(
            label(r),
            (x, y),
            textcoords="offset points",
            xytext=(dx, dy),
            ha="center",
            fontsize=8,
            color=ORANGE if is_reference else INK if on_front else MUTED,
        )
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, loc="left", pad=12)
    ymin = min(p[1] for p in points)
    ymax = max(p[1] for p in points)
    pad = (ymax - ymin) * 0.18
    ax.set_ylim(ymin - pad, ymax + pad * 1.6)
    xmin = min(p[0] for p in points)
    xmax = max(p[0] for p in points)
    xpad = (xmax - xmin) * 0.12
    ax.set_xlim(xmin - xpad, xmax + xpad)
    fig.text(
        0.01,
        0.01,
        "Blue = Pareto-optimal RF-DETR. Orange diamond = Ultralytics reference. "
        "* baseline (B01)",
        fontsize=7.5,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGDIR / filename, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return front


def main() -> None:
    FIGDIR.mkdir(exist_ok=True)
    all_rows = load_rows()
    rows = [row for row in all_rows if row["id"] not in REFERENCE_MODELS]
    add_deltas(rows)
    write_final_csv(rows)
    print(f"wrote {FINAL.relative_to(ROOT)} ({len(rows)} models)")

    bar_chart(
        all_rows, "fps",
        "Inference throughput by model (RTX 4050 Laptop, batch 1)",
        "FPS (higher is better)", "fps_comparison.png",
        ascending=False, fmt="{:.1f}", reference_ids=REFERENCE_MODELS,
    )
    bar_chart(
        all_rows, "latency",
        "Mean inference latency by model (RTX 4050 Laptop, batch 1)",
        "Latency (ms, lower is better)", "latency_comparison.png",
        ascending=True, fmt="{:.1f}", reference_ids=REFERENCE_MODELS,
    )
    bar_chart(
        all_rows, "size",
        "Model artifact size",
        "Size (MB, lower is better)", "size_comparison.png",
        ascending=True, fmt="{:.1f}", reference_ids=REFERENCE_MODELS,
    )
    map_chart(all_rows, "map_comparison.png")
    mask_chart(all_rows, "mask_comparison.png")
    bar_chart(
        all_rows,
        "mask_miou",
        "Semantic mask overlap by model (test set, 296 images)",
        "Mask mIoU (higher is better)",
        "mask_miou_comparison.png",
        ascending=False,
        fmt="{:.3f}",
        reference_ids=REFERENCE_MODELS,
    )

    fronts = {}
    fronts["fps"] = pareto_chart(
        all_rows, "fps", "FPS (higher is better)", True,
        "Accuracy vs. throughput trade-off", "pareto_map_fps.png",
        offsets={
            "C01": (0, 12),
            "B03": (-58, -3),
            "B02": (42, -8),
            "M01": (0, -16),
            "R01": (-6, 10),
            "parking_front": (0, -17),
        },
        reference_ids=REFERENCE_MODELS,
    )
    fronts["size"] = pareto_chart(
        all_rows, "size", "Model size (MB, lower is better)", False,
        "Accuracy vs. model size trade-off", "pareto_map_size.png",
        offsets={
            "C01": (-20, 12),
            "B03": (52, -3),
            "B02": (44, -16),
            "M01": (0, -16),
            "R01": (-14, -18),
            "parking_front": (30, -4),
        },
        reference_ids=REFERENCE_MODELS,
    )
    fronts["latency"] = pareto_chart(
        all_rows, "latency", "Mean latency (ms, lower is better)", False,
        "Accuracy vs. latency trade-off", "pareto_map_latency.png",
        offsets={
            "C01": (0, 12),
            "B03": (52, -3),
            "B02": (-44, -8),
            "M01": (0, -16),
            "parking_front": (0, -17),
        },
        reference_ids=REFERENCE_MODELS,
    )
    mask_fronts = {}
    mask_fronts["fps"] = pareto_chart(
        all_rows, "fps", "FPS (higher is better)", True,
        "Segmentation accuracy vs. throughput trade-off",
        "pareto_mask_fps.png",
        offsets={
            "C01": (-10, 12),
            "S01": (0, 12),
            "B01": (30, -16),
            "B03": (-56, -4),
            "B02": (34, 10),
            "M01": (-46, -10),
            "M02": (10, -18),
            "R01": (-14, -16),
            "parking_front": (0, -17),
        },
        reference_ids=REFERENCE_MODELS,
        ykey="mask_ap",
        ylabel="Mask AP",
    )
    mask_fronts["size"] = pareto_chart(
        all_rows, "size", "Model size (MB, lower is better)", False,
        "Segmentation accuracy vs. model size trade-off",
        "pareto_mask_size.png",
        offsets={
            "C01": (-34, 12),
            "S01": (0, 12),
            "B01": (0, -17),
            "B03": (58, -12),
            "B02": (50, 2),
            "M01": (-56, -14),
            "M02": (-56, 8),
            "R01": (58, -26),
            "parking_front": (52, -4),
        },
        reference_ids=REFERENCE_MODELS,
        ykey="mask_ap",
        ylabel="Mask AP",
    )

    for key, front in fronts.items():
        ids = ", ".join(r["id"] for _, _, r in front)
        print(f"pareto front (mAP50-95 vs {key}): {ids}")
    for key, front in mask_fronts.items():
        ids = ", ".join(r["id"] for _, _, r in front)
        print(f"pareto front (mask AP vs {key}): {ids}")
    print(f"wrote figures to {FIGDIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
