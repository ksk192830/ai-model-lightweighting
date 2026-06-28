#!/usr/bin/env python3
"""Run reproducible baseline inference with an RF-DETR segmentation checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(REPOSITORY_ROOT / "results" / ".cache" / "matplotlib"),
)

import numpy as np  # noqa: E402
import supervision as sv  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402
from rfdetr import RFDETR  # noqa: E402


DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "baseline.yaml"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "results" / "baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline RF-DETR inference.")
    parser.add_argument("--camera", choices=("front", "rear"), required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def select_device(requested: str) -> str:
    if requested == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    return requested


def load_model_config(config_path: Path, camera: str) -> dict:
    with resolve_path(config_path).open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    try:
        return config["models"][camera]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Missing model configuration for '{camera}'.") from error


def detection_records(detections: sv.Detections, class_names: list[str]) -> list[dict]:
    records = []
    masks = detections.mask

    for index, box in enumerate(detections.xyxy):
        class_id = int(detections.class_id[index])
        confidence = float(detections.confidence[index])
        mask_area = int(np.count_nonzero(masks[index])) if masks is not None else None
        records.append(
            {
                "class_id": class_id,
                "class_name": class_names[class_id],
                "confidence": confidence,
                "box_xyxy": [float(value) for value in box],
                "mask_area_pixels": mask_area,
            }
        )

    return records


def annotate(
    image: Image.Image,
    detections: sv.Detections,
    class_names: list[str],
) -> np.ndarray:
    scene = np.asarray(image).copy()
    labels = [
        f"{class_names[int(class_id)]} {float(confidence):.2f}"
        for class_id, confidence in zip(
            detections.class_id,
            detections.confidence,
            strict=True,
        )
    ]

    if detections.mask is not None:
        scene = sv.MaskAnnotator().annotate(scene, detections)
    scene = sv.BoxAnnotator().annotate(scene, detections)
    return sv.LabelAnnotator().annotate(scene, detections, labels)


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")

    image_path = resolve_path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model_config = load_model_config(args.config, args.camera)
    checkpoint = resolve_path(Path(model_config["checkpoint"]))
    class_names = list(model_config["classes"])
    device = select_device(args.device)

    load_started = time.perf_counter()
    model = RFDETR.from_checkpoint(
        checkpoint,
        device=device,
        num_classes=len(class_names),
    )
    load_seconds = time.perf_counter() - load_started

    image = Image.open(image_path).convert("RGB")
    inference_started = time.perf_counter()
    detections = model.predict(image, threshold=args.threshold)
    inference_seconds = time.perf_counter() - inference_started

    output_dir = resolve_path(args.output_dir) / args.camera
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = output_dir / image_path.stem
    visualization_path = output_stem.with_suffix(".png")
    predictions_path = output_stem.with_suffix(".json")

    Image.fromarray(annotate(image, detections, class_names)).save(visualization_path)
    predictions = {
        "camera": args.camera,
        "checkpoint": str(checkpoint.relative_to(REPOSITORY_ROOT)),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "image": str(image_path.relative_to(REPOSITORY_ROOT)),
        "image_size": {"width": image.width, "height": image.height},
        "device": device,
        "precision": "fp32",
        "threshold": args.threshold,
        "model_load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "detection_count": len(detections),
        "detections": detection_records(detections, class_names),
    }
    predictions_path.write_text(
        json.dumps(predictions, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"camera: {args.camera}")
    print(f"device: {device}")
    print(f"detections: {len(detections)}")
    print(f"inference: {inference_seconds:.3f} s")
    print(f"predictions: {predictions_path}")
    print(f"visualization: {visualization_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
