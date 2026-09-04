#!/usr/bin/env python3
"""Generate paper-ready RF-DETR figures from frozen repository evidence."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402
from pycocotools import mask as mask_utils  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
MAIN_DIR = Path("figures/paper")
APPENDIX_DIR = Path("figures/appendix")
STAGE1 = Path("results/stage1-static-evaluation.json")
STAGE2 = Path("results/stage2-notebook-summary.json")
STAGE3 = Path("results/stage3-pareto.json")
TRAINING_METRICS = Path("artifacts/training/front-rfdetr-seg-large-v1/metrics.csv")
DATASET = Path("data/training/front_session_split_v1/test")
MANIFEST = Path("results/paper-figure-manifest.json")

BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
GRAY = "#A7B0BA"
DARK = "#263238"
LIGHT = "#EEF2F6"
GRID = "#D9E0E7"
CLASS_COLORS = {1: "#E69F00", 2: "#0072B2", 3: "#009E73"}
CLASS_NAMES = {1: "out_line", 2: "parking_lot", 3: "parking_space"}
REALTIME_BUDGET_MS = 1000.0 / 30.0


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return value


def configure_style() -> None:
    preferred = ("Noto Sans CJK KR", "Noto Sans CJK JP", "DejaVu Sans")
    available = {font.name for font in font_manager.fontManager.ttflist}
    family = next((name for name in preferred if name in available), "DejaVu Sans")
    plt.rcParams.update(
        {
            "font.family": family,
            "axes.unicode_minus": False,
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def style_axis(ax: Any, *, grid_axis: str = "both") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.65, alpha=0.8)
    ax.set_axisbelow(True)


def save_pair(fig: Any, base: Path) -> list[str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return [str(png.relative_to(ROOT)), str(pdf.relative_to(ROOT))]


def add_box(
    ax: Any,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    detail: str,
    *,
    facecolor: str,
    edgecolor: str = "white",
    title_color: str = "white",
    detail_color: str = "white",
    linewidth: float = 1.1,
) -> None:
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        transform=ax.transAxes,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height * 0.63,
        title,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color=title_color,
    )
    ax.text(
        x + width / 2,
        y + height * 0.29,
        detail,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.2,
        color=detail_color,
        linespacing=1.18,
    )


def add_arrow(
    ax: Any,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = DARK,
    linestyle: str = "-",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            linestyle=linestyle,
            color=color,
        )
    )


def plot_pipeline(base: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(10.8, 4.2))
    ax.set_axis_off()
    stages = [
        (0.02, 0.60, "데이터 구축", "4,466장\n세션 기반 분할", BLUE),
        (0.22, 0.60, "Baseline 학습", "RF-DETR Seg. Large\n18 epoch", BLUE),
        (0.42, 0.60, "후보 생성", "precision·pruning\nresolution·결합", PURPLE),
        (0.62, 0.60, "Stage 1", "정적 artifact·recipe\n정확도 미사용", ORANGE),
        (0.82, 0.60, "Stage 2", "TensorRT build\n정확도·성능 실측", ORANGE),
    ]
    width, height = 0.16, 0.23
    for index, (x, y, title, detail, color) in enumerate(stages):
        add_box(ax, x, y, width, height, title, detail, facecolor=color)
        if index:
            add_arrow(ax, (x - 0.025, y + height / 2), (x - 0.004, y + height / 2))

    add_box(
        ax,
        0.52,
        0.15,
        0.19,
        0.22,
        "정확도 보존 gate",
        "ΔBBox ≥ −.010\nΔMask ≥ −.010\nΔmIoU ≥ −.020",
        facecolor=VERMILLION,
    )
    add_box(
        ax,
        0.77,
        0.15,
        0.19,
        0.22,
        "Stage 3 Pareto",
        "Mask AP ↑\nMedian latency ↓\nEngine size ↓",
        facecolor=GREEN,
    )
    add_arrow(ax, (0.90, 0.59), (0.69, 0.38))
    add_arrow(ax, (0.71, 0.26), (0.76, 0.26))
    ax.text(
        0.02,
        0.93,
        "정적 경량화와 장비 종속 성능을 분리한 평가 파이프라인",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color=DARK,
    )
    ax.text(
        0.02,
        0.04,
        "모든 단계는 registry, artifact hash, 환경 기록과 terminal failure 로그로 추적",
        transform=ax.transAxes,
        fontsize=8,
        color="#546E7A",
    )
    fig.tight_layout()
    return save_pair(fig, base)


def plot_candidate_taxonomy(base: Path) -> list[str]:
    groups = [
        ("Baseline", "B01", "FP32 기준", BLUE),
        ("Precision", "B02 · B03", "FP16 · INT8 PTQ", SKY),
        ("Unstructured", "U01 · U02 · U03", "Magnitude 10/30/50%", ORANGE),
        ("2:4 sparsity", "M01 · M02", "Dense control · Sparse tactic", ORANGE),
        ("Structured", "S01 · S02 · S03 · S04", "Decoder · FFN 축소", PURPLE),
        ("Combined", "C01 · C02 · C03 · C04", "구조 + precision/resolution", GREEN),
        ("Resolution", "R01 · R02 · R03", "432 · 480 · 384", SKY),
        ("ModelOpt Q/DQ", "Q01–Q07", "INT8 · INT4 · FP8 · mixed", VERMILLION),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 4.9))
    ax.set_axis_off()
    positions = [(0.02 + 0.25 * col, 0.57 - 0.37 * row) for row in range(2) for col in range(4)]
    for (title, ids, detail, color), (x, y) in zip(groups, positions, strict=True):
        add_box(
            ax,
            x,
            y,
            0.22,
            0.27,
            title,
            f"{ids}\n{detail}",
            facecolor="white",
            edgecolor=color,
            title_color=color,
            detail_color=DARK,
            linewidth=2.0,
        )
    ax.text(
        0.02,
        0.94,
        "동일 RF-DETR baseline에서 파생한 26개 경량화 후보",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color=DARK,
    )
    ax.text(
        0.02,
        0.04,
        "W 계열은 최종 연구 범위에서 제거되었으며 위 8개 후보군만 공식 registry에 포함",
        transform=ax.transAxes,
        fontsize=8,
        color="#546E7A",
    )
    fig.tight_layout()
    return save_pair(fig, base)


def annotation_mask(annotation: dict[str, Any], height: int, width: int) -> np.ndarray:
    segmentation = annotation.get("segmentation")
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        decoded = mask_utils.decode(mask_utils.merge(rles))
    elif isinstance(segmentation, dict):
        rle = dict(segmentation)
        if isinstance(rle.get("counts"), str):
            rle["counts"] = rle["counts"].encode("ascii")
        decoded = mask_utils.decode(rle)
    else:
        decoded = np.zeros((height, width), dtype=np.uint8)
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    return decoded.astype(bool)


def locate_image(dataset: Path, file_name: str) -> Path:
    candidates = [dataset / Path(file_name).name, dataset / "images" / Path(file_name).name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(file_name)


def select_examples(coco: dict[str, Any]) -> dict[str, dict[str, Any]]:
    annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco.get("annotations", []):
        annotations[int(annotation["image_id"])].append(annotation)
    images = {int(item["id"]): item for item in coco.get("images", [])}

    def features(image_id: int) -> tuple[float, int]:
        info = images[image_id]
        image_area = max(1.0, float(info["width"]) * float(info["height"]))
        total_area = sum(float(item.get("area") or item["bbox"][2] * item["bbox"][3]) for item in annotations[image_id])
        vertices = sum(
            len(poly) // 2
            for item in annotations[image_id]
            for poly in item.get("segmentation", [])
            if isinstance(poly, list)
        )
        return total_area / image_area, vertices

    all_three = [
        image_id
        for image_id in images
        if {int(item["category_id"]) for item in annotations[image_id]} >= {1, 2, 3}
    ]
    if len(all_three) < 3:
        raise ValueError("At least three images containing all evaluated classes are required")
    small = min(all_three, key=lambda image_id: (features(image_id)[0], image_id))
    large = max(all_three, key=lambda image_id: (features(image_id)[0], -image_id))
    remaining = [image_id for image_id in all_three if image_id not in {small, large}]
    complex_id = max(remaining, key=lambda image_id: (features(image_id)[1], -image_id))
    negatives = sorted(image_id for image_id in images if not annotations[image_id])
    if not negatives:
        raise ValueError("A negative benchmark example is required")
    return {
        "small_objects": images[small],
        "complex_boundary": images[complex_id],
        "large_regions": images[large],
        "negative": images[negatives[0]],
    }


def overlay_ground_truth(
    image: Image.Image,
    annotations: Iterable[dict[str, Any]],
) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    output = rgb.astype(float)
    height, width = rgb.shape[:2]
    for annotation in annotations:
        category_id = int(annotation["category_id"])
        color = np.array(matplotlib.colors.to_rgb(CLASS_COLORS[category_id])) * 255
        mask = annotation_mask(annotation, height, width)
        output[mask] = 0.55 * output[mask] + 0.45 * color
    return np.clip(output, 0, 255).astype(np.uint8)


def draw_gt(ax: Any, image: Image.Image, annotations: list[dict[str, Any]]) -> None:
    ax.imshow(overlay_ground_truth(image, annotations))
    for annotation in annotations:
        category_id = int(annotation["category_id"])
        x, y, width, height = (float(value) for value in annotation["bbox"])
        ax.add_patch(
            Rectangle(
                (x, y),
                width,
                height,
                fill=False,
                edgecolor=CLASS_COLORS[category_id],
                linewidth=1.2,
            )
        )
    ax.set_axis_off()


def load_dataset(root: Path) -> tuple[Path, dict[str, Any], dict[int, list[dict[str, Any]]]]:
    dataset = resolve(root, DATASET)
    coco = load_json(dataset / "_annotations.coco.json")
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco.get("annotations", []):
        by_image[int(annotation["image_id"])].append(annotation)
    return dataset, coco, by_image


def plot_dataset_examples(
    root: Path,
    base: Path,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    dataset, coco, by_image = load_dataset(root)
    selected = select_examples(coco)
    titles = {
        "small_objects": "작은 영역 비중이 큰 장면",
        "complex_boundary": "복잡한 polygon 경계",
        "large_regions": "큰 주차 영역",
        "negative": "평가 객체가 없는 장면",
    }
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.2), constrained_layout=True)
    for ax, (key, info) in zip(axes.flat, selected.items(), strict=True):
        path = locate_image(dataset, str(info["file_name"]))
        with Image.open(path) as source:
            draw_gt(ax, source.convert("RGB"), by_image[int(info["id"])])
        ax.set_title(f"{titles[key]}  (image_id={info['id']})", loc="left", fontsize=9)
    fig.legend(
        handles=[Patch(color=CLASS_COLORS[index], label=CLASS_NAMES[index]) for index in (1, 2, 3)],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.035),
        ncol=3,
        frameon=False,
    )
    return save_pair(fig, base), selected


def plot_training_curves(root: Path, base: Path) -> list[str]:
    with resolve(root, TRAINING_METRICS).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    epochs: list[int] = []
    train_loss: list[float] = []
    val_loss: list[float] = []
    bbox_map: list[float] = []
    mask_map: list[float] = []
    ema_mask_map: list[float] = []
    for epoch in sorted({int(row["epoch"]) for row in rows}):
        train = next((row for row in rows if int(row["epoch"]) == epoch and row["train/loss"]), None)
        valid = next((row for row in rows if int(row["epoch"]) == epoch and row["val/loss"]), None)
        if not train or not valid:
            continue
        epochs.append(epoch + 1)
        train_loss.append(float(train["train/loss"]))
        val_loss.append(float(valid["val/loss"]))
        bbox_map.append(float(valid["val/mAP_50_95"]))
        mask_map.append(float(valid["val/segm_mAP_50_95"]))
        ema_mask_map.append(float(valid["val/ema_segm_mAP_50_95"]))
    best_epoch = epochs[int(np.argmax(ema_mask_map))]
    stop_epoch = max(epochs)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), constrained_layout=True)
    ax = axes[0]
    ax.plot(epochs, train_loss, color=BLUE, marker="o", markersize=3, label="Train loss")
    ax.plot(epochs, val_loss, color=ORANGE, marker="s", markersize=3, label="Validation loss")
    ax.axvline(best_epoch, color=GREEN, linestyle="--", linewidth=1.1)
    ax.axvline(stop_epoch, color=VERMILLION, linestyle=":", linewidth=1.3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("(a) 학습 및 검증 loss", loc="left")
    ax.legend(frameon=False)
    style_axis(ax, grid_axis="y")

    ax = axes[1]
    ax.plot(epochs, bbox_map, color=BLUE, marker="o", markersize=3, label="BBox mAP")
    ax.plot(epochs, mask_map, color=ORANGE, marker="s", markersize=3, label="Mask mAP")
    ax.plot(epochs, ema_mask_map, color=GREEN, marker="^", markersize=3, label="EMA Mask mAP")
    ax.axvline(best_epoch, color=GREEN, linestyle="--", linewidth=1.1, label=f"Best EMA (epoch {best_epoch})")
    ax.axvline(stop_epoch, color=VERMILLION, linestyle=":", linewidth=1.3, label=f"Stop (epoch {stop_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation AP@[.50:.95]")
    ax.set_ylim(0.4, 0.72)
    ax.set_title("(b) 검증 정확도와 조기 종료", loc="left")
    ax.legend(frameon=False, ncol=2)
    style_axis(ax, grid_axis="y")
    return save_pair(fig, base)


def plot_candidate_flow(base: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(10.7, 5.3))
    ax.set_axis_off()
    main = [
        (0.05, 0.74, "최초 후보", "26", BLUE),
        (0.30, 0.74, "Stage 1 통과", "22", SKY),
        (0.55, 0.74, "Stage 2 완료", "21", SKY),
        (0.30, 0.24, "정확도 gate 통과", "11", ORANGE),
        (0.70, 0.24, "Pareto 비지배해", "2  ·  C01, R01", GREEN),
    ]
    for x, y, title, detail, color in main:
        add_box(ax, x, y, 0.18 if x != 0.30 or y != 0.24 else 0.22, 0.18, title, detail, facecolor=color)
    add_arrow(ax, (0.23, 0.83), (0.29, 0.83))
    add_arrow(ax, (0.48, 0.83), (0.54, 0.83))
    add_arrow(ax, (0.64, 0.73), (0.43, 0.43))
    add_arrow(ax, (0.53, 0.33), (0.69, 0.33))

    rejected = [
        (0.29, 0.51, "Stage 1 불통 4", "U01 · U02 · U03 · S03"),
        (0.76, 0.65, "Build 실패 1", "Q03 · INT4 parser"),
        (0.04, 0.24, "정확도 불통 10", "M01·M02·S02·S04·R03\nQ01·Q02·Q04·Q06·Q07"),
        (0.69, 0.02, "Pareto 지배 9", "B01·B02·B03·S01·C02\nC03·C04·R02·Q05"),
    ]
    for x, y, title, detail in rejected:
        add_box(
            ax,
            x,
            y,
            0.23,
            0.14,
            title,
            detail,
            facecolor="#F7F8FA",
            edgecolor=GRAY,
            title_color=VERMILLION,
            detail_color=DARK,
        )
    add_arrow(ax, (0.39, 0.73), (0.40, 0.66), color=GRAY, linestyle="--")
    add_arrow(ax, (0.73, 0.80), (0.77, 0.73), color=GRAY, linestyle="--")
    add_arrow(ax, (0.29, 0.33), (0.27, 0.31), color=GRAY, linestyle="--")
    add_arrow(ax, (0.80, 0.23), (0.80, 0.16), color=GRAY, linestyle="--")
    ax.text(
        0.05,
        0.95,
        "후보 선별 결과와 단계별 제외 사유",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color=DARK,
    )
    fig.tight_layout()
    return save_pair(fig, base)


def stage3_rows(root: Path) -> list[dict[str, Any]]:
    value = load_json(resolve(root, STAGE3)).get("rows", [])
    if len(value) != 21:
        raise ValueError(f"Expected 21 measured candidates, got {len(value)}")
    return value


def plot_accuracy_heatmap(root: Path, base: Path) -> list[str]:
    rows = stage3_rows(root)
    family_order = {name: index for index, name in enumerate(("baseline", "precision", "sparsity", "structured", "combined", "input"))}
    rows = sorted(rows, key=lambda row: (family_order.get(str(row["family"]), 99), str(row["experiment_id"])))
    fields = ["bbox_ap_delta_vs_B01", "mask_ap_delta_vs_B01", "semantic_miou_delta_vs_B01"]
    limits = np.array([0.01, 0.01, 0.02], dtype=float)
    deltas = np.asarray([[float(row[field]) for field in fields] for row in rows])
    normalized = np.clip(deltas / limits, -5.0, 2.0)
    cmap = LinearSegmentedColormap.from_list("gate", [VERMILLION, "#F7F7F7", GREEN])
    norm = TwoSlopeNorm(vmin=-5.0, vcenter=0.0, vmax=2.0)
    fig, ax = plt.subplots(figsize=(7.7, 8.0))
    image = ax.imshow(normalized, aspect="auto", cmap=cmap, norm=norm)
    labels = ["ΔBBox AP\n한계 −0.010", "ΔMask AP\n한계 −0.010", "ΔmIoU\n한계 −0.020"]
    ax.set_xticks(range(3), labels)
    ax.set_yticks(range(len(rows)), [str(row["experiment_id"]) for row in rows])
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0)
    for y, row in enumerate(rows):
        for x, value in enumerate(deltas[y]):
            ax.text(x, y, f"{value:+.3f}", ha="center", va="center", fontsize=7, color=DARK)
            if value < -limits[x]:
                ax.add_patch(Rectangle((x - 0.49, y - 0.49), 0.98, 0.98, fill=False, edgecolor=VERMILLION, linewidth=1.4))
        passed = bool(row["accuracy_gate_pass"])
        ax.text(
            3.05,
            y,
            "PASS" if passed else "FAIL",
            va="center",
            ha="left",
            fontsize=7.3,
            fontweight="bold",
            color=GREEN if passed else VERMILLION,
        )
    for y in range(len(rows) - 1):
        if rows[y]["family"] != rows[y + 1]["family"]:
            ax.axhline(y + 0.5, color="white", linewidth=2.2)
    ax.set_xlim(-0.5, 3.65)
    ax.set_title("후보별 B01 대비 정확도 변화와 보존 gate", loc="left", pad=14, fontweight="bold")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.09)
    colorbar.set_label("허용 하락폭 대비 변화 배수 (−1 = gate 경계)")
    fig.tight_layout()
    return save_pair(fig, base)


def plot_pareto(root: Path, base: Path) -> list[str]:
    rows = stage3_rows(root)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), constrained_layout=True)
    labels = {"B01", "B03", "C01", "C02", "R01"}
    offsets = {
        "B01": (5, 8),
        "B03": (5, 8),
        "C01": (-27, 10),
        "C02": (-28, -14),
        "R01": (5, -12),
    }
    for panel, (field, xlabel) in enumerate(
        (("median_ms", "Median latency (ms)"), ("engine_size_bytes", "TensorRT engine size (MiB)"))
    ):
        ax = axes[panel]
        for row in rows:
            candidate = str(row["experiment_id"])
            x = float(row[field])
            if field == "engine_size_bytes":
                x /= 1048576.0
            y = float(row["mask_ap"])
            passed = bool(row["accuracy_gate_pass"])
            pareto = bool(row["pareto_optimal"])
            color = GREEN if pareto else (BLUE if passed else GRAY)
            size = 86 if pareto else (42 if passed else 30)
            marker = "D" if candidate == "B01" else "o"
            if field == "median_ms":
                low = float(row["replicate_median_ci95_low_ms"])
                high = float(row["replicate_median_ci95_high_ms"])
                ax.errorbar(
                    x,
                    y,
                    xerr=np.array([[max(0.0, x - low)], [max(0.0, high - x)]]),
                    fmt="none",
                    ecolor="#C9D1D9",
                    elinewidth=0.8,
                    capsize=1.5,
                    zorder=1,
                )
            ax.scatter(x, y, s=size, color=color, edgecolor=DARK if pareto else "white", linewidth=0.8, marker=marker, zorder=3)
            if candidate in labels:
                ax.annotate(
                    candidate,
                    (x, y),
                    xytext=offsets[candidate],
                    textcoords="offset points",
                    fontsize=7.5,
                    fontweight="bold" if pareto else "normal",
                )
        if field == "median_ms":
            ax.axvline(REALTIME_BUDGET_MS, color=VERMILLION, linestyle="--", linewidth=1.1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mask AP@[.50:.95]" if panel == 0 else "")
        ax.set_title("(a) 정확도–지연시간" if panel == 0 else "(b) 정확도–engine 크기", loc="left")
        style_axis(ax, grid_axis="y")
    axes[0].legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor="white", label="정확도 gate 통과"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=GRAY, markeredgecolor="white", label="정확도 gate 불통"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=GREEN, markeredgecolor=DARK, label="Pareto"),
            Line2D([0], [0], color=VERMILLION, linestyle="--", label="33.33 ms"),
        ],
        loc="lower left",
        frameon=False,
        ncol=2,
    )
    return save_pair(fig, base)


def plot_static_reductions(root: Path, base: Path) -> list[str]:
    stage1 = load_json(resolve(root, STAGE1))
    rows = stage1.get("rows", [])
    ids = ["B01", "U01", "U02", "U03", "S01", "S02", "S03", "S04", "C03", "C04", "R01", "R02", "R03"]
    by_id = {str(row["experiment_id"]): row for row in rows}
    baseline = by_id["B01"]["static_metrics"]
    matrix = []
    for candidate in ids:
        metrics = by_id[candidate]["static_metrics"]
        matrix.append(
            [
                100 * (1 - float(metrics["onnx_size_bytes"]) / float(baseline["onnx_size_bytes"])),
                100 * (1 - float(metrics["onnx_nodes"]) / float(baseline["onnx_nodes"])),
                100 * (1 - float(metrics["estimated_macs"]) / float(baseline["estimated_macs"])),
            ]
        )
    values = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    image = ax.imshow(values, aspect="auto", cmap="Blues", vmin=0, vmax=max(46.5, float(values.max())))
    ax.set_xticks(range(3), ["ONNX 크기 감소", "Graph node 감소", "Dense MAC 감소"])
    ax.set_yticks(range(len(ids)), ids)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0)
    for y, candidate in enumerate(ids):
        for x, value in enumerate(values[y]):
            ax.text(x, y, f"{value:.1f}%", ha="center", va="center", fontsize=7, color=DARK)
        passed = bool(by_id[candidate]["stage1_passed"])
        ax.text(3.0, y, "PASS" if passed else "FAIL", va="center", color=GREEN if passed else VERMILLION, fontsize=7.2, fontweight="bold")
    ax.set_xlim(-0.5, 3.55)
    ax.set_title("구조·해상도 후보의 Stage 1 정적 감소율", loc="left", pad=14, fontweight="bold")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.08)
    colorbar.set_label("B01 대비 감소율 (%)")
    fig.tight_layout()
    return save_pair(fig, base)


def plot_latency_stability(root: Path, base: Path) -> list[str]:
    rows = sorted(stage3_rows(root), key=lambda row: float(row["replicate_median_cv"]), reverse=True)
    ids = [str(row["experiment_id"]) for row in rows]
    values = [100 * float(row["replicate_median_cv"]) for row in rows]
    colors = [GREEN if row["pareto_optimal"] else BLUE for row in rows]
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    bars = ax.bar(ids, values, color=colors)
    ax.axhline(5.0, color=VERMILLION, linestyle="--", linewidth=1.2, label="재측정 기준 5%")
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.08, f"{value:.2f}", ha="center", va="bottom", fontsize=6.2, rotation=90)
    ax.set_ylabel("반복 median CV (%)")
    ax.set_ylim(0, 5.55)
    ax.tick_params(axis="x", rotation=45)
    ax.set_title("후보별 반복 지연시간 안정성", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    style_axis(ax, grid_axis="y")
    fig.tight_layout()
    return save_pair(fig, base)


def plot_classwise_accuracy(root: Path, base: Path) -> list[str]:
    candidates = ["B01", "C01", "R01"]
    reports = {candidate: load_json(root / f"results/coco-evaluation/{candidate}-front.json") for candidate in candidates}
    metrics = [
        ("bbox_by_category", "BBox AP"),
        ("segm_by_category", "Mask AP"),
        ("semantic_iou_by_category", "Semantic IoU"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.7), constrained_layout=True)
    fig.suptitle("B01·C01·R01의 클래스별 정확도", x=0.025, ha="left", fontsize=11, fontweight="bold")
    x = np.arange(3)
    width = 0.24
    colors = {"B01": GRAY, "C01": GREEN, "R01": BLUE}
    for ax, (field, title) in zip(axes, metrics, strict=True):
        for index, candidate in enumerate(candidates):
            source = reports[candidate]["metrics"][field]
            values = []
            for category_id in (1, 2, 3):
                item = source[str(category_id)]
                values.append(float(item["ap"] if isinstance(item, dict) else item))
            ax.bar(x + (index - 1) * width, values, width, color=colors[candidate], label=candidate)
        ax.set_xticks(x, ["out_line", "parking_lot", "parking_space"], rotation=18, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel(title if ax is axes[0] else "")
        ax.set_title(title, loc="left")
        style_axis(ax, grid_axis="y")
    axes[0].legend(frameon=False, ncol=3, loc="upper left")
    return save_pair(fig, base)


def detections_overlay(image: Image.Image, detections: Any) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    output = rgb.astype(float)
    masks = detections.mask
    if masks is not None:
        for index in range(len(detections)):
            category_id = int(detections.class_id[index])
            if category_id not in CLASS_COLORS:
                continue
            color = np.array(matplotlib.colors.to_rgb(CLASS_COLORS[category_id])) * 255
            mask = masks[index].astype(bool)
            output[mask] = 0.55 * output[mask] + 0.45 * color
    return np.clip(output, 0, 255).astype(np.uint8)


def draw_prediction(ax: Any, image: Image.Image, detections: Any) -> None:
    ax.imshow(detections_overlay(image, detections))
    for index in range(len(detections)):
        category_id = int(detections.class_id[index])
        if category_id not in CLASS_COLORS:
            continue
        x1, y1, x2, y2 = (float(value) for value in detections.xyxy[index])
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=CLASS_COLORS[category_id], linewidth=1.0))
    ax.set_axis_off()


def plot_qualitative_comparison(
    root: Path,
    base: Path,
    selected: dict[str, dict[str, Any]],
) -> tuple[list[str], dict[str, Any]]:
    sys.path.insert(0, str(root / "scripts/evaluation"))
    from evaluate_coco_rfdetr_onnx import ONNXRunner  # noqa: PLC0415

    dataset, _, by_image = load_dataset(root)
    keys = ["small_objects", "complex_boundary", "large_regions"]
    inputs: dict[str, Image.Image] = {}
    for key in keys:
        info = selected[key]
        with Image.open(locate_image(dataset, str(info["file_name"]))) as source:
            inputs[key] = source.convert("RGB")
    models = {
        "B01": root / "artifacts/experiments/B01/front/model.onnx",
        "C01": root / "artifacts/experiments/S01/front/model.onnx",
        "R01": root / "artifacts/experiments/R01/front/model.onnx",
    }
    predictions: dict[str, dict[str, Any]] = {candidate: {} for candidate in models}
    for candidate, path in models.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        runner = ONNXRunner(path, intra_op_threads=4, inter_op_threads=1)
        for key in keys:
            predictions[candidate][key] = runner.predict(inputs[key], threshold=0.25)
        del runner

    fig, axes = plt.subplots(
        len(keys),
        5,
        figsize=(13.3, 5.25),
        gridspec_kw={"wspace": 0.025, "hspace": 0.07},
    )
    column_titles = ["Input", "Ground truth", "B01 source ONNX", "C01 source ONNX", "R01 source ONNX"]
    row_titles = ["작은 영역", "복잡한 경계", "큰 영역"]
    for row_index, key in enumerate(keys):
        image = inputs[key]
        info = selected[key]
        axes[row_index, 0].imshow(image)
        axes[row_index, 0].set_axis_off()
        draw_gt(axes[row_index, 1], image, by_image[int(info["id"])])
        for column_index, candidate in enumerate(("B01", "C01", "R01"), start=2):
            draw_prediction(axes[row_index, column_index], image, predictions[candidate][key])
        axes[row_index, 0].text(
            -0.055,
            0.5,
            f"{row_titles[row_index]}\nID {int(info['id'])}",
            transform=axes[row_index, 0].transAxes,
            ha="center",
            va="center",
            rotation=90,
            fontsize=8,
            fontweight="bold",
            color=DARK,
        )
        for column_index, title in enumerate(column_titles):
            if row_index == 0:
                axes[row_index, column_index].set_title(title, fontsize=8.5)
    fig.legend(
        handles=[Patch(color=CLASS_COLORS[index], label=CLASS_NAMES[index]) for index in (1, 2, 3)],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.025),
        ncol=3,
        frameon=False,
    )
    fig.subplots_adjust(left=0.035, right=0.998, top=0.95, bottom=0.085)
    source_info = {
        "threshold": 0.25,
        "warning": "Qualitative panels use source ONNX graphs, not the notebook TensorRT engines.",
        "models": {candidate: str(path.relative_to(root)) for candidate, path in models.items()},
        "images": {key: {"id": int(selected[key]["id"]), "file_name": str(selected[key]["file_name"])} for key in keys},
    }
    return save_pair(fig, base), source_info


def write_placement_plan(path: Path) -> None:
    text = """# 논문 시각자료 배치안

| 번호 | 파일 | 삽입 위치 | 핵심 주장 |
|---|---|---|---|
| 그림 1 | `paper/01_overall_pipeline.*` | 1장 말미 또는 4장 시작 | 정적 선별과 목표 장비 평가를 분리한 연구 절차 |
| 그림 2 | `paper/02_dataset_examples.*` | 3.1 데이터 구성 | 실제 입력과 세 클래스의 GT annotation 형태 |
| 그림 3 | `dataset_split_validity.*` | 3.3 데이터 분할 | 세션 기반 분할 비율·분포·임계값·배정 |
| 그림 4 | `paper/04_training_curves.*` | 4.1 Baseline 학습 | 18 epoch 종료 및 best EMA checkpoint 근거 |
| 그림 5 | `paper/05_candidate_taxonomy.*` | 4.2 후보 구성 | 26개 후보의 8개 방법군 |
| 그림 6 | `paper/06_candidate_flow.*` | 4.3 말미 또는 6장 시작 | 26→22→21→11→2 및 단계별 제외 사유 |
| 그림 7 | `paper/07_accuracy_gate_heatmap.*` | 6.2 Stage 2 결과 | B01 대비 세 정확도 변화와 gate 판정 |
| 그림 8 | `paper/08_pareto_tradeoff.*` | 6.3 Pareto 분석 | C01·R01의 정확도–속도–크기 비지배 관계 |
| 그림 9 | `paper/09_qualitative_source_onnx.*` | 6.4 정성 분석 | 동일 benchmark 이미지의 GT와 source ONNX 출력 비교 |
| 그림 A1 | `appendix/A1_stage1_static_reductions.*` | 부록 | 구조·해상도 후보의 정적 감소율 |
| 그림 A2 | `appendix/A2_latency_stability.*` | 부록 | 63개 반복의 CV 5% 이하 확인 |
| 그림 A3 | `appendix/A3_classwise_accuracy.*` | 부록 | B01·C01·R01 클래스별 정확도 |

## 사용 주의

- 그림 9는 로컬에 최종 TensorRT engine이 없으므로 대응 source ONNX graph의 정성 출력이다. TensorRT engine 출력으로 표현하면 안 된다.
- 30 FPS 선은 median 33.33 ms 기준이며 모든 프레임 또는 지속 처리량의 30 FPS 보장을 뜻하지 않는다.
- `test`는 논문에서 `fixed comparative benchmark` 또는 `benchmark`로 표기한다.
- 본문 삽입에는 PDF를 우선 사용하고 PNG는 Notion·README 미리보기용으로 사용한다.
- GPU memory는 후보 간 차이가 작아 별도 그래프로 만들지 않는다.

## 논문용 캡션 초안

- **그림 1. 정적 경량화와 장비 종속 성능을 분리한 평가 파이프라인.** 무증강 데이터 구성과 RF-DETR baseline 학습 후 26개 경량화 후보를 생성하였다. Stage 1은 정확도를 사용하지 않는 artifact·recipe 정적 검증이며, Stage 2는 동일 benchmark에서 TensorRT build, 정확도 및 지연시간을 측정한다. 정확도 보존 gate를 통과한 후보만 정확도 최대화, median latency 최소화, engine 크기 최소화의 3목적 Pareto 분석에 포함하였다.
- **그림 2. 고정 benchmark의 대표 입력 및 ground-truth annotation.** 작은 주석 영역 비율, polygon 꼭짓점 수, 큰 주석 영역 비율을 기준으로 각 극단 사례를 결정론적으로 선택하고, 평가 객체가 없는 negative 장면을 함께 제시하였다. 색상은 `out_line`, `parking_lot`, `parking_space`를 나타낸다.
- **그림 3. 무증강 데이터의 세션 단위 분할 타당성.** 전체 4,466장을 train 3,625장(81.17%), validation 404장(9.05%), benchmark 437장(9.79%)으로 분할하였다. 2초 임계값은 관측된 세션 내부 최대 간격 1.411초와 세션 경계 최소 간격 5.010초 사이에 위치하며, 하나의 촬영 세션은 둘 이상의 분할에 포함되지 않는다.
- **그림 4. RF-DETR baseline의 18 epoch 학습 과정.** 좌측은 train·validation loss, 우측은 validation BBox AP, Mask AP 및 EMA Mask AP를 나타낸다. EMA Mask AP의 최댓값은 epoch 12에서 기록되었고 이후 6 epoch 동안 0.001을 초과하는 개선이 없어 epoch 18에서 학습을 종료하였다.
- **그림 5. 동일 RF-DETR baseline에서 파생한 26개 경량화 후보의 방법 분류.** 후보는 baseline, 정밀도 변환, 비정형 pruning, 2:4 sparsity, 구조 축소, 결합형, 입력 해상도, ModelOpt quantization/dequantization의 8개 방법군으로 구성된다. 연구 범위에서 제외된 W 계열은 포함하지 않았다.
- **그림 6. 후보 선별 결과와 단계별 제외 사유.** 최초 26개 중 Stage 1에서 22개가 통과하였고, Q03은 TensorRT parser 문제로 build에 실패하여 21개가 Stage 2 측정을 완료하였다. 이 중 11개가 정확도 보존 gate를 통과했으며, 3목적 Pareto 분석 결과 C01과 R01이 비지배 후보로 남았다.
- **그림 7. B01 대비 후보별 정확도 변화와 보존 gate 판정.** 각 셀은 B01 대비 BBox AP, Mask AP, semantic mIoU 변화량이며, 허용 하락 한계는 각각 −0.010, −0.010, −0.020이다. 하나 이상의 하한을 위반한 셀은 테두리로 표시하고 후보별 최종 PASS/FAIL을 함께 제시하였다.
- **그림 8. 정확도 보존 후보의 성능–자원 trade-off와 3목적 Pareto 해.** 오차막대는 후보별 3회 반복 median의 평균에 대한 95% t 신뢰구간이며, 점의 x 좌표는 총 600회 측정의 pooled median이다. 수직선은 30 FPS에 대응하는 33.33 ms 기준이다. 녹색으로 강조한 C01과 R01은 Mask AP 최대화, median latency 최소화, TensorRT engine 크기 최소화에서 서로 지배되지 않는다.
- **그림 9. 동일 benchmark 사례에서의 ground truth와 source ONNX 정성 비교.** B01, C01, R01에 대응하는 source ONNX graph를 confidence threshold 0.25로 추론하였다. 이 그림은 최종 TensorRT engine의 정성 결과가 아니므로 장비 배포 결과와 구분해 해석한다.
- **그림 A1. 구조·해상도 후보의 Stage 1 정적 감소율.** B01 대비 ONNX 파일 크기, graph node 및 dense MAC 감소율과 정적 검증 통과 여부를 나타낸다.
- **그림 A2. 공식 지연시간 반복 측정의 안정성.** 21개 후보에서 얻은 총 63회 측정의 후보별 반복 median 변동계수(CV)를 제시하였다. 모든 후보가 사전 정의한 재측정 기준 5%보다 낮았다.
- **그림 A3. B01, C01, R01의 클래스별 정확도.** 세 후보의 `out_line`, `parking_lot`, `parking_space`에 대한 BBox AP, Mask AP 및 semantic IoU를 동일 축에서 비교하였다.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--skip-qualitative", action="store_true", help="Skip CPU ONNX inference panels")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    configure_style()
    main_dir = resolve(root, MAIN_DIR)
    appendix_dir = resolve(root, APPENDIX_DIR)
    outputs: dict[str, Any] = {}
    outputs["figure_1_pipeline"] = plot_pipeline(main_dir / "01_overall_pipeline")
    outputs["figure_2_dataset_examples"], selected = plot_dataset_examples(root, main_dir / "02_dataset_examples")
    outputs["figure_4_training_curves"] = plot_training_curves(root, main_dir / "04_training_curves")
    outputs["figure_5_candidate_taxonomy"] = plot_candidate_taxonomy(main_dir / "05_candidate_taxonomy")
    outputs["figure_6_candidate_flow"] = plot_candidate_flow(main_dir / "06_candidate_flow")
    outputs["figure_7_accuracy_gate"] = plot_accuracy_heatmap(root, main_dir / "07_accuracy_gate_heatmap")
    outputs["figure_8_pareto"] = plot_pareto(root, main_dir / "08_pareto_tradeoff")
    outputs["appendix_a1_static"] = plot_static_reductions(root, appendix_dir / "A1_stage1_static_reductions")
    outputs["appendix_a2_stability"] = plot_latency_stability(root, appendix_dir / "A2_latency_stability")
    outputs["appendix_a3_classwise"] = plot_classwise_accuracy(root, appendix_dir / "A3_classwise_accuracy")
    qualitative_info = None
    if not args.skip_qualitative:
        outputs["figure_9_qualitative"], qualitative_info = plot_qualitative_comparison(
            root,
            main_dir / "09_qualitative_source_onnx",
            selected,
        )
    payload = {
        "official_measurement_run_id": "20260904_104755",
        "generator": "scripts/reporting/generate_paper_visuals.py",
        "outputs": outputs,
        "dataset_examples": {
            key: {"id": int(value["id"]), "file_name": str(value["file_name"])}
            for key, value in selected.items()
        },
        "qualitative": qualitative_info,
    }
    manifest = resolve(root, MANIFEST)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_placement_plan(root / "docs/paper/figure-placement-plan.md")
    print(main_dir)
    print(appendix_dir)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
