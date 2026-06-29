#!/usr/bin/env python3
"""Build a calibrated TensorRT INT8 engine from an RF-DETR ONNX model."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ONNX_DIR = REPOSITORY_ROOT / "artifacts" / "onnx"
DEFAULT_CALIBRATION_DIR = REPOSITORY_ROOT / "data" / "calibration"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "artifacts" / "tensorrt"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a calibrated TensorRT INT8 engine."
    )
    parser.add_argument("--camera", choices=("front", "rear"), required=True)
    parser.add_argument("--onnx-dir", type=Path, default=DEFAULT_ONNX_DIR)
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=DEFAULT_CALIBRATION_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workspace-mib", type=int, default=4096)
    parser.add_argument(
        "--no-fp16-fallback",
        action="store_true",
        help="Disable FP16 for layers that TensorRT cannot execute as INT8.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def list_calibration_images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def make_calibrator_class(trt):
    class ImageCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(
            self,
            images: list[Path],
            input_shape: tuple[int, int, int, int],
            cache_path: Path,
        ) -> None:
            super().__init__()
            import torch

            self.images = images
            self.input_shape = input_shape
            self.cache_path = cache_path
            self.index = 0
            self.device_input = torch.empty(
                input_shape,
                dtype=torch.float32,
                device="cuda",
            )

        def get_batch_size(self) -> int:
            return self.input_shape[0]

        def get_batch(self, names):
            del names
            if self.index >= len(self.images):
                return None

            import torch
            import torchvision.transforms.functional as functional
            from PIL import Image

            batch_size, channels, height, width = self.input_shape
            batch_paths = self.images[self.index : self.index + batch_size]
            if len(batch_paths) < batch_size:
                return None

            tensors = []
            for path in batch_paths:
                with Image.open(path) as image:
                    tensor = functional.to_tensor(image.convert("RGB"))
                tensor = functional.resize(tensor, [height, width])
                tensor = functional.normalize(
                    tensor,
                    IMAGENET_MEAN,
                    IMAGENET_STD,
                )
                tensors.append(tensor)

            self.device_input.copy_(torch.stack(tensors))
            self.index += batch_size
            print(
                f"\rcalibration: {self.index}/{len(self.images)}",
                end="",
                flush=True,
            )
            return [int(self.device_input.data_ptr())]

        def read_calibration_cache(self):
            if self.cache_path.is_file():
                print(f"Using calibration cache: {self.cache_path}")
                return self.cache_path.read_bytes()
            return None

        def write_calibration_cache(self, cache) -> None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_bytes(cache)
            print(f"\nCalibration cache: {self.cache_path}")

    return ImageCalibrator


def main() -> int:
    args = parse_args()
    if args.workspace_mib < 1:
        raise ValueError("--workspace-mib must be at least one.")

    onnx_path = (
        resolve_path(args.onnx_dir)
        / args.camera
        / f"parking_{args.camera}.onnx"
    )
    calibration_dir = resolve_path(args.calibration_dir) / args.camera
    output_dir = resolve_path(args.output_dir) / args.camera
    engine_path = output_dir / f"parking_{args.camera}_int8.engine"
    cache_path = output_dir / f"parking_{args.camera}_int8.cache"
    metadata_path = engine_path.with_suffix(".json")

    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    if not calibration_dir.is_dir():
        raise FileNotFoundError(
            f"Calibration directory not found: {calibration_dir}"
        )
    images = list_calibration_images(calibration_dir)
    if not images:
        raise ValueError(f"No calibration images found: {calibration_dir}")
    if engine_path.exists() and not args.force:
        print(f"TensorRT engine already exists: {engine_path}")
        print("Use --force to build it again.")
        return 0

    print(f"ONNX: {onnx_path}")
    print(f"Calibration images: {len(images)} ({calibration_dir})")
    print(f"INT8 engine: {engine_path}")
    print(f"Workspace: {args.workspace_mib} MiB")
    print(f"FP16 fallback: {not args.no_fp16_fallback}")
    if args.dry_run:
        return 0

    try:
        import tensorrt as trt
        import torch
    except ImportError as error:
        raise RuntimeError(
            "TensorRT Python bindings and CUDA-enabled PyTorch are required. "
            "Run this script on the target NVIDIA machine."
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to the PyTorch environment.")

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.force:
        engine_path.unlink(missing_ok=True)
        cache_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(
            str(parser.get_error(index))
            for index in range(parser.num_errors)
        )
        raise RuntimeError(f"TensorRT could not parse ONNX:\n{errors}")
    if network.num_inputs != 1:
        raise ValueError(
            f"Expected one ONNX input, found {network.num_inputs}."
        )

    input_tensor = network.get_input(0)
    input_shape = tuple(int(value) for value in input_tensor.shape)
    if len(input_shape) != 4 or any(value < 1 for value in input_shape):
        raise ValueError(f"Static NCHW input required, found {input_shape}.")
    if input_shape[1] != 3:
        raise ValueError(f"RGB input required, found {input_shape}.")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        args.workspace_mib * 1024 * 1024,
    )
    config.set_flag(trt.BuilderFlag.INT8)
    if not args.no_fp16_fallback:
        config.set_flag(trt.BuilderFlag.FP16)

    ImageCalibrator = make_calibrator_class(trt)
    calibrator = ImageCalibrator(images, input_shape, cache_path)
    config.int8_calibrator = calibrator

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT INT8 engine build failed.")
    engine_path.write_bytes(serialized_engine)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "camera": args.camera,
        "precision": "int8_ptq",
        "fp16_fallback": not args.no_fp16_fallback,
        "source_onnx": str(onnx_path.relative_to(REPOSITORY_ROOT)),
        "source_onnx_sha256": sha256(onnx_path),
        "calibration_dir": str(
            calibration_dir.relative_to(REPOSITORY_ROOT)
        ),
        "calibration_image_count": len(images),
        "calibration_images": [
            path.relative_to(REPOSITORY_ROOT).as_posix()
            for path in images
        ],
        "calibration_cache": str(cache_path.relative_to(REPOSITORY_ROOT)),
        "calibration_cache_sha256": sha256(cache_path),
        "preprocessing": {
            "color": "RGB",
            "resize": [input_shape[2], input_shape[3]],
            "scale": "uint8 / 255.0",
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
        },
        "input_shape": list(input_shape),
        "engine_path": str(engine_path.relative_to(REPOSITORY_ROOT)),
        "engine_size_bytes": engine_path.stat().st_size,
        "engine_sha256": sha256(engine_path),
        "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "tensorrt_version": trt.__version__,
        "workspace_mib": args.workspace_mib,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"TensorRT INT8 engine: {engine_path}")
    print(f"metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
