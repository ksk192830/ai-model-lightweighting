#!/usr/bin/env python3
"""Visualize sequential RF-DETR inference at a configurable interval."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from rfdetr import RFDETR

from benchmark_baseline import TensorRTRunner
from infer_baseline import annotate


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "baseline.yaml"
DEFAULT_ENGINE_DIR = REPOSITORY_ROOT / "artifacts" / "tensorrt"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "results" / "inference_stream"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sequential PyTorch/TensorRT inference visualization."
    )
    parser.add_argument("--camera", choices=("front", "rear"), required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument(
        "--backend",
        choices=("pytorch", "fp32", "fp16", "int8", "all", "tensorrt", "both"),
        default="all",
        help=(
            "Model variant. 'all' compares TensorRT FP32, FP16, and INT8. "
            "'tensorrt' and 'both' are legacy aliases."
        ),
    )
    parser.add_argument("--engine", type=Path, help="Legacy alias for --int8-engine.")
    parser.add_argument("--fp32-engine", type=Path)
    parser.add_argument("--fp16-engine", type=Path)
    parser.add_argument("--int8-engine", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between displayed frames (default: 1.0).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        help="Stop after this many images.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Show a live OpenCV window; press q or Esc to stop.",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        help=(
            "Output MP4 path. Defaults to "
            "results/inference_stream/<camera>_<backend>.mp4."
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def image_paths(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_model_config(path: Path, camera: str) -> dict:
    with resolve_path(path).open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    return config["models"][camera]


def labeled_panel(
    image: Image.Image,
    detections,
    class_names: list[str],
    title: str,
) -> np.ndarray:
    visualization = Image.fromarray(annotate(image, detections, class_names))
    panel = Image.new(
        "RGB",
        (visualization.width, visualization.height + 40),
        "white",
    )
    panel.paste(visualization, (0, 40))
    ImageDraw.Draw(panel).text((12, 12), title, fill="black")
    return np.asarray(panel)


def combine_panels(panels: list[np.ndarray]) -> np.ndarray:
    height = max(panel.shape[0] for panel in panels)
    padded = []
    for panel in panels:
        if panel.shape[0] < height:
            panel = cv2.copyMakeBorder(
                panel,
                0,
                height - panel.shape[0],
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(255, 255, 255),
            )
        padded.append(panel)
    return np.concatenate(padded, axis=1)


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")
    if args.interval <= 0:
        raise ValueError("--interval must be greater than zero.")
    if args.max_images is not None and args.max_images < 1:
        raise ValueError("--max-images must be at least one.")
    if not args.live and args.output_video is None:
        args.output_video = (
            DEFAULT_OUTPUT_DIR / f"{args.camera}_{args.backend}.mp4"
        )

    directory = resolve_path(args.image_dir)
    paths = image_paths(directory)
    if args.max_images is not None:
        paths = paths[: args.max_images]
    if not paths:
        raise FileNotFoundError(f"No images found in: {directory}")

    model_config = load_model_config(args.config, args.camera)
    class_names = list(model_config["classes"])
    backend_groups = {
        "all": ("fp32", "fp16", "int8"),
        "both": ("pytorch", "int8"),
        "tensorrt": ("int8",),
    }
    backends = backend_groups.get(args.backend, (args.backend,))
    predictors = {}
    if "pytorch" in backends:
        checkpoint = resolve_path(Path(model_config["checkpoint"]))
        model = RFDETR.from_checkpoint(
            checkpoint,
            device="cuda",
            num_classes=len(class_names),
        )
        predictors["pytorch"] = lambda image: model.predict(
            image,
            threshold=args.threshold,
        )
    for backend in ("fp32", "fp16", "int8"):
        if backend not in backends:
            continue
        engine_override = (
            args.fp32_engine
            if backend == "fp32"
            else (
                args.fp16_engine
                if backend == "fp16"
                else args.int8_engine or args.engine
            )
        )
        engine_path = resolve_path(
            engine_override
            or (
                DEFAULT_ENGINE_DIR
                / args.camera
                / f"parking_{args.camera}_{backend}.engine"
            )
        )
        if not engine_path.is_file():
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
        runner = TensorRTRunner(engine_path)
        predictors[backend] = lambda image, runner=runner: runner.predict(
            image,
            threshold=args.threshold,
        )

    output_path = (
        resolve_path(args.output_video)
        if args.output_video is not None
        else None
    )
    writer = None
    records = []
    try:
        for index, path in enumerate(paths, start=1):
            frame_started = time.perf_counter()
            image = Image.open(path).convert("RGB")
            panels = []
            counts = {}
            timings_ms = {}
            for backend in backends:
                started = time.perf_counter()
                detections = predictors[backend](image)
                torch.cuda.synchronize()
                timings_ms[backend] = (time.perf_counter() - started) * 1000
                counts[backend] = len(detections)
                backend_title = {
                    "pytorch": "PyTorch FP32",
                    "fp32": "TensorRT FP32",
                    "fp16": "TensorRT FP16",
                    "int8": "TensorRT INT8",
                }[backend]
                title = (
                    f"{backend_title}"
                    f" | {len(detections)} detections | "
                    f"{timings_ms[backend]:.1f} ms"
                )
                panels.append(
                    labeled_panel(image, detections, class_names, title)
                )

            rgb_frame = combine_panels(panels)
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            if output_path is not None:
                if writer is None:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        1.0 / args.interval,
                        (bgr_frame.shape[1], bgr_frame.shape[0]),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(
                            f"Could not open video writer: {output_path}"
                        )
                writer.write(bgr_frame)

            records.append(
                {
                    "image": str(path.relative_to(REPOSITORY_ROOT)),
                    "detection_counts": counts,
                    "inference_ms": timings_ms,
                }
            )
            print(
                f"\rframe: {index}/{len(paths)} {path.name} "
                f"counts={counts}",
                end="",
                flush=True,
            )

            elapsed = time.perf_counter() - frame_started
            wait_seconds = max(0.0, args.interval - elapsed)
            if args.live:
                cv2.imshow("RF-DETR inference comparison", bgr_frame)
                key = cv2.waitKey(max(1, int(wait_seconds * 1000)))
                if key in (27, ord("q")):
                    break
            elif wait_seconds:
                time.sleep(wait_seconds)
        print()
    finally:
        if writer is not None:
            writer.release()
        if args.live:
            cv2.destroyAllWindows()

    if output_path is not None:
        metadata_path = output_path.with_suffix(".json")
        metadata_path.write_text(
            json.dumps(
                {
                    "camera": args.camera,
                    "backend": args.backend,
                    "interval_seconds": args.interval,
                    "frames": records,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"video: {output_path}")
        print(f"metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
