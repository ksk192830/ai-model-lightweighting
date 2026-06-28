#!/usr/bin/env python3
"""Benchmark RF-DETR baseline inference with warmup and repeated runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(REPOSITORY_ROOT / "results" / ".cache" / "matplotlib"),
)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402
from rfdetr import RFDETR  # noqa: E402


DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "baseline.yaml"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "results" / "benchmarks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark baseline inference.")
    parser.add_argument("--camera", choices=("front", "rear"), required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Call RF-DETR optimize_for_inference before benchmarking.",
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
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    return requested


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def load_model_config(config_path: Path, camera: str) -> dict:
    with resolve_path(config_path).open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    try:
        return config["models"][camera]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Missing model configuration for '{camera}'.") from error


def percentile(values: list[float], percentile_value: float) -> float:
    return float(np.percentile(np.asarray(values), percentile_value))


def hardware_name(device: str) -> str:
    if device == "cuda":
        return torch.cuda.get_device_name(torch.cuda.current_device())
    if device == "mps":
        return f"Apple {platform.machine()}"
    return platform.processor() or platform.machine()


def append_csv(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(record))
        if not exists:
            writer.writeheader()
        writer.writerow(record)


def main() -> int:
    args = parse_args()
    if args.warmup < 0:
        raise ValueError("--warmup must be zero or greater.")
    if args.runs < 1:
        raise ValueError("--runs must be at least one.")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")

    image_path = resolve_path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model_config = load_model_config(args.config, args.camera)
    checkpoint = resolve_path(Path(model_config["checkpoint"]))
    class_names = list(model_config["classes"])
    device = select_device(args.device)
    image = Image.open(image_path).convert("RGB")

    model = RFDETR.from_checkpoint(
        checkpoint,
        device=device,
        num_classes=len(class_names),
    )
    if args.optimize:
        model.optimize_for_inference()

    print(f"device: {device} ({hardware_name(device)})")
    print(f"warmup: {args.warmup}")
    for index in range(args.warmup):
        model.predict(image, threshold=args.threshold)
        synchronize(device)
        print(f"\rwarmup: {index + 1}/{args.warmup}", end="", flush=True)
    if args.warmup:
        print()

    timings_ms: list[float] = []
    detection_counts: list[int] = []
    print(f"runs: {args.runs}")
    for index in range(args.runs):
        synchronize(device)
        started = time.perf_counter()
        detections = model.predict(image, threshold=args.threshold)
        synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        timings_ms.append(elapsed_ms)
        detection_counts.append(len(detections))
        print(f"\rrun: {index + 1}/{args.runs}", end="", flush=True)
    print()

    mean_ms = statistics.fmean(timings_ms)
    timestamp = datetime.now(timezone.utc).isoformat()
    result = {
        "timestamp_utc": timestamp,
        "camera": args.camera,
        "model": checkpoint.name,
        "precision": "fp32",
        "framework": "pytorch",
        "device": device,
        "hardware": hardware_name(device),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "rfdetr_version": version("rfdetr"),
        "cuda_version": torch.version.cuda or "",
        "cudnn_version": (
            torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else ""
        ),
        "image": str(image_path.relative_to(REPOSITORY_ROOT)),
        "image_width": image.width,
        "image_height": image.height,
        "threshold": args.threshold,
        "warmup_runs": args.warmup,
        "measured_runs": args.runs,
        "optimized": args.optimize,
        "model_size_bytes": checkpoint.stat().st_size,
        "mean_ms": mean_ms,
        "median_ms": statistics.median(timings_ms),
        "min_ms": min(timings_ms),
        "max_ms": max(timings_ms),
        "p95_ms": percentile(timings_ms, 95),
        "fps": 1000.0 / mean_ms,
        "detection_count_min": min(detection_counts),
        "detection_count_max": max(detection_counts),
    }

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    details_path = output_dir / f"{args.camera}_fp32_{run_id}.json"
    summary_path = output_dir / "summary.csv"
    details = {**result, "timings_ms": timings_ms}
    details_path.write_text(
        json.dumps(details, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    append_csv(summary_path, result)

    print(f"mean: {result['mean_ms']:.3f} ms")
    print(f"median: {result['median_ms']:.3f} ms")
    print(f"p95: {result['p95_ms']:.3f} ms")
    print(f"fps: {result['fps']:.2f}")
    print(f"details: {details_path}")
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
