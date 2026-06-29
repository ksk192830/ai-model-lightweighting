#!/usr/bin/env python3
"""Run and visualize PyTorch or TensorRT RF-DETR inference."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(REPOSITORY_ROOT / "results" / ".cache" / "matplotlib"),
)

import numpy as np  # noqa: E402
import supervision as sv  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from rfdetr import RFDETR  # noqa: E402

from benchmark_baseline import TensorRTRunner  # noqa: E402


DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "baseline.yaml"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "results" / "inference"
DEFAULT_ENGINE_DIR = REPOSITORY_ROOT / "artifacts" / "tensorrt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and visualize PyTorch or TensorRT RF-DETR inference."
    )
    parser.add_argument("--camera", choices=("front", "rear"), required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--backend",
        choices=("pytorch", "fp32", "fp16", "int8", "all", "tensorrt", "both"),
        default="pytorch",
        help=(
            "Model variant. 'all' compares TensorRT FP32, FP16, and INT8. "
            "'tensorrt' and 'both' are legacy aliases for INT8 and "
            "FP32+INT8."
        ),
    )
    parser.add_argument(
        "--engine",
        type=Path,
        help="Legacy alias for --int8-engine.",
    )
    parser.add_argument("--fp32-engine", type=Path)
    parser.add_argument("--fp16-engine", type=Path)
    parser.add_argument("--int8-engine", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def select_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
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
        class_name = (
            class_names[class_id]
            if 0 <= class_id < len(class_names)
            else f"class_{class_id}"
        )
        records.append(
            {
                "class_id": class_id,
                "class_name": class_name,
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
        (
            f"{class_names[int(class_id)]} {float(confidence):.2f}"
            if 0 <= int(class_id) < len(class_names)
            else f"class_{int(class_id)} {float(confidence):.2f}"
        )
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


def comparison_image(
    visualizations: dict[str, np.ndarray],
) -> Image.Image:
    titles = {
        "pytorch": "PyTorch FP32",
        "fp32": "TensorRT FP32",
        "fp16": "TensorRT FP16",
        "int8": "TensorRT INT8",
    }
    panels = []
    for backend, visualization in visualizations.items():
        panel = Image.fromarray(visualization)
        canvas = Image.new("RGB", (panel.width, panel.height + 40), "white")
        canvas.paste(panel, (0, 40))
        ImageDraw.Draw(canvas).text(
            (12, 12),
            titles[backend],
            fill="black",
        )
        panels.append(canvas)
    output = Image.new(
        "RGB",
        (sum(panel.width for panel in panels), max(panel.height for panel in panels)),
        "white",
    )
    x_offset = 0
    for panel in panels:
        output.paste(panel, (x_offset, 0))
        x_offset += panel.width
    return output


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

    image = Image.open(image_path).convert("RGB")
    backend_groups = {
        "all": ("fp32", "fp16", "int8"),
        "both": ("pytorch", "int8"),
        "tensorrt": ("int8",),
    }
    backends = backend_groups.get(args.backend, (args.backend,))
    output_root = resolve_path(args.output_dir)
    visualizations = {}

    for backend in backends:
        load_started = time.perf_counter()
        if backend == "pytorch":
            device = select_device(args.device)
            model_path = checkpoint
            model = RFDETR.from_checkpoint(
                checkpoint,
                device=device,
                num_classes=len(class_names),
            )
            predict = lambda: model.predict(
                image,
                threshold=args.threshold,
            )
            precision = "fp32"
        else:
            if args.device not in ("auto", "cuda"):
                raise ValueError("TensorRT only supports --device auto or cuda.")
            device = "cuda"
            engine_override = (
                args.fp32_engine
                if backend == "fp32"
                else (
                    args.fp16_engine
                    if backend == "fp16"
                    else args.int8_engine or args.engine
                )
            )
            model_path = resolve_path(
                engine_override
                or (
                    DEFAULT_ENGINE_DIR
                    / args.camera
                    / f"parking_{args.camera}_{backend}.engine"
                )
            )
            if not model_path.is_file():
                raise FileNotFoundError(
                    f"TensorRT engine not found: {model_path}"
                )
            runner = TensorRTRunner(model_path)
            predict = lambda: runner.predict(
                image,
                threshold=args.threshold,
            )
            precision = (
                "fp32"
                if backend == "fp32"
                else (
                    "fp16"
                    if backend == "fp16"
                    else "int8-fp16-fallback"
                )
            )
        load_seconds = time.perf_counter() - load_started

        inference_started = time.perf_counter()
        detections = predict()
        if device == "cuda":
            torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_started

        output_dir = output_root / backend / args.camera
        output_dir.mkdir(parents=True, exist_ok=True)
        output_stem = output_dir / image_path.stem
        visualization_path = output_stem.with_suffix(".png")
        predictions_path = output_stem.with_suffix(".json")
        visualization = annotate(image, detections, class_names)
        visualizations[backend] = visualization
        Image.fromarray(visualization).save(visualization_path)

        predictions = {
            "camera": args.camera,
            "backend": backend,
            "model": str(model_path.relative_to(REPOSITORY_ROOT)),
            "model_size_bytes": model_path.stat().st_size,
            "image": str(image_path.relative_to(REPOSITORY_ROOT)),
            "image_size": {"width": image.width, "height": image.height},
            "device": device,
            "precision": precision,
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

        print(f"backend: {backend}")
        print(f"device: {device}")
        print(f"detections: {len(detections)}")
        print(f"inference: {inference_seconds:.3f} s")
        print(f"predictions: {predictions_path}")
        print(f"visualization: {visualization_path}")

    if len(visualizations) > 1:
        comparison_dir = output_root / "comparison" / args.camera
        comparison_dir.mkdir(parents=True, exist_ok=True)
        comparison_path = comparison_dir / f"{image_path.stem}.png"
        comparison_image(visualizations).save(comparison_path)
        print(f"comparison: {comparison_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
