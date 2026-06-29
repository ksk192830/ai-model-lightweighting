#!/usr/bin/env python3
"""Benchmark PyTorch checkpoints or TensorRT engines."""

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


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(REPOSITORY_ROOT / "results" / ".cache" / "matplotlib"),
)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchvision.transforms.functional as functional  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402
from rfdetr import RFDETR  # noqa: E402
from rfdetr.models.postprocess import PostProcess  # noqa: E402
from supervision import Detections  # noqa: E402


DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "baseline.yaml"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "results" / "benchmarks"
DEFAULT_ENGINE_DIR = REPOSITORY_ROOT / "artifacts" / "tensorrt"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark PyTorch or TensorRT inference."
    )
    parser.add_argument("--camera", choices=("front", "rear"), required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--backend",
        choices=("pytorch", "fp32", "fp16", "int8", "tensorrt"),
        default="pytorch",
        help=(
            "Inference backend/precision. 'tensorrt' is a legacy alias "
            "for INT8 (default: pytorch)."
        ),
    )
    parser.add_argument(
        "--engine",
        type=Path,
        help="Override the default TensorRT engine path.",
    )
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


class TensorRTRunner:
    """Execute an RF-DETR TensorRT engine with RF-DETR postprocessing."""

    def __init__(self, engine_path: Path) -> None:
        try:
            import tensorrt as trt
        except ImportError as error:
            raise RuntimeError(
                "TensorRT is not installed in the active Python environment."
            ) from error

        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT benchmarking requires CUDA.")

        self.trt = trt
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(
            engine_path.read_bytes()
        )
        if self.engine is None:
            raise RuntimeError(f"Could not deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Could not create TensorRT execution context.")

        self.input_name = ""
        self.outputs: dict[str, torch.Tensor] = {}
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            shape = tuple(self.engine.get_tensor_shape(name))
            if any(dimension < 0 for dimension in shape):
                raise RuntimeError(
                    f"Dynamic TensorRT tensor shapes are not supported: "
                    f"{name}={shape}"
                )
            numpy_dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            torch_dtype = torch.from_numpy(
                np.empty((), dtype=numpy_dtype)
            ).dtype
            tensor = torch.empty(shape, dtype=torch_dtype, device="cuda")
            self.context.set_tensor_address(name, tensor.data_ptr())
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                if self.input_name:
                    raise RuntimeError("Expected exactly one TensorRT input.")
                self.input_name = name
                self.input = tensor
            else:
                self.outputs[name] = tensor

        required_outputs = {"dets", "labels", "masks"}
        if not self.input_name or not required_outputs.issubset(self.outputs):
            raise RuntimeError(
                "Expected one input and dets/labels/masks TensorRT outputs."
            )
        if self.input.ndim != 4 or self.input.shape[0] != 1:
            raise RuntimeError(
                f"Expected a static NCHW batch-1 input, got {tuple(self.input.shape)}."
            )
        self.postprocess = PostProcess(num_select=300)

    def predict(self, image: Image.Image, threshold: float) -> Detections:
        source_image = np.array(image)
        original_width, original_height = image.size
        _, _, input_height, input_width = self.input.shape

        tensor = functional.to_tensor(image)
        tensor = tensor.to(device="cuda", dtype=self.input.dtype)
        tensor = functional.resize(tensor, [input_height, input_width])
        tensor = functional.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
        self.input.copy_(tensor.unsqueeze(0))

        stream = torch.cuda.current_stream()
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 failed.")

        results = self.postprocess(
            {
                "pred_boxes": self.outputs["dets"],
                "pred_logits": self.outputs["labels"],
                "pred_masks": self.outputs["masks"],
            },
            target_sizes=torch.tensor(
                [[original_height, original_width]],
                device="cuda",
            ),
        )
        result = results[0]
        keep = result["scores"] > threshold
        detections = Detections(
            xyxy=result["boxes"][keep].float().cpu().numpy(),
            confidence=result["scores"][keep].float().cpu().numpy(),
            class_id=result["labels"][keep].cpu().numpy(),
            mask=result["masks"][keep].squeeze(1).cpu().numpy(),
        )
        detections.metadata["source_image"] = source_image
        detections.data["source_shape"] = np.tile(
            np.array([original_height, original_width], dtype=np.int64),
            (len(detections), 1),
        )
        return detections


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
    image = Image.open(image_path).convert("RGB")

    if args.backend == "pytorch":
        device = select_device(args.device)
        model_path = checkpoint
        model = RFDETR.from_checkpoint(
            checkpoint,
            device=device,
            num_classes=len(class_names),
        )
        if args.optimize:
            model.optimize_for_inference()
        predict = lambda: model.predict(image, threshold=args.threshold)
        precision = "fp32"
        framework = "pytorch"
        framework_version = torch.__version__
    else:
        if args.device not in ("auto", "cuda"):
            raise ValueError("TensorRT only supports --device auto or cuda.")
        if args.optimize:
            raise ValueError("--optimize is only supported by the PyTorch backend.")
        device = "cuda"
        precision_name = "int8" if args.backend == "tensorrt" else args.backend
        model_path = resolve_path(
            args.engine
            or (
                DEFAULT_ENGINE_DIR
                / args.camera
                / f"parking_{args.camera}_{precision_name}.engine"
            )
        )
        if not model_path.is_file():
            raise FileNotFoundError(f"TensorRT engine not found: {model_path}")
        runner = TensorRTRunner(model_path)
        predict = lambda: runner.predict(image, threshold=args.threshold)
        precision = (
            "int8-fp16-fallback"
            if precision_name == "int8"
            else precision_name
        )
        framework = "tensorrt"
        framework_version = runner.trt.__version__

    print(f"device: {device} ({hardware_name(device)})")
    print(f"backend: {args.backend}")
    print(f"model: {model_path}")
    print(f"warmup: {args.warmup}")
    for index in range(args.warmup):
        predict()
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
        detections = predict()
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
        "model": model_path.name,
        "precision": precision,
        "framework": framework,
        "framework_version": framework_version,
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
        "model_size_bytes": model_path.stat().st_size,
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
    details_path = (
        output_dir
        / f"{args.camera}_{args.backend}_{precision}_{run_id}.json"
    )
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
