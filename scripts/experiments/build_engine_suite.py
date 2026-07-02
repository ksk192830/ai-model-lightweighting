#!/usr/bin/env python3
"""Build and smoke-test a registered front TensorRT engine suite."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts" / "evaluation"))

from benchmark_baseline import TensorRTRunner  # noqa: E402
from kips_lightweighting.artifacts import artifact_paths  # noqa: E402
from kips_lightweighting.metadata import sha256  # noqa: E402


SUITES = {
    "final8": ("B01", "B02", "B03", "M01", "M02", "S01", "C01", "R01"),
    "all13": (
        "B01",
        "B02",
        "B03",
        "U02",
        "M01",
        "M02",
        "S01",
        "S02",
        "S03",
        "S04",
        "C01",
        "C02",
        "R01",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=tuple(SUITES), default="final8")
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("data/labeled_test/front/images/image000002.png"),
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/notebook-engine-smoke.json"),
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def main() -> int:
    args = parse_args()
    experiments = SUITES[args.suite]
    if args.dry_run:
        for experiment_id in experiments:
            print(
                f"{sys.executable} scripts/experiments/build_candidate.py "
                f"{experiment_id} --camera front --target engine"
                + (" --force" if args.force else "")
            )
        return 0

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Run this command on an NVIDIA host and "
            "verify nvidia-smi before building TensorRT engines."
        )
    try:
        import tensorrt as trt
    except ImportError as error:
        raise RuntimeError(
            "TensorRT Python bindings are missing. "
            "Install requirements-tensorrt.txt."
        ) from error

    if not args.skip_build:
        for experiment_id in experiments:
            command = [
                sys.executable,
                "scripts/experiments/build_candidate.py",
                experiment_id,
                "--camera",
                "front",
                "--target",
                "engine",
            ]
            if args.force:
                command.append("--force")
            print("$ " + " ".join(command), flush=True)
            subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)

    image_path = resolve(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    image = Image.open(image_path).convert("RGB")
    records = []
    for experiment_id in experiments:
        engine = artifact_paths(experiment_id, "front").engine
        if not engine.is_file():
            raise FileNotFoundError(engine)
        runner = TensorRTRunner(engine)
        runner.predict(image, threshold=args.threshold)
        torch.cuda.synchronize()
        started = time.perf_counter()
        detections = runner.predict(image, threshold=args.threshold)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        record = {
            "experiment_id": experiment_id,
            "engine": str(engine.relative_to(REPOSITORY_ROOT)),
            "engine_size_bytes": engine.stat().st_size,
            "engine_sha256": sha256(engine),
            "input_shape": list(runner.input.shape),
            "output_shapes": {
                name: list(tensor.shape)
                for name, tensor in runner.outputs.items()
            },
            "detection_count": len(detections),
            "latency_ms": elapsed_ms,
            "fps": 1000.0 / elapsed_ms,
        }
        records.append(record)
        print(
            f"{experiment_id}: {elapsed_ms:.1f} ms, "
            f"{len(detections)} detections"
        )

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite": args.suite,
        "camera": "front",
        "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
        "tensorrt": trt.__version__,
        "image": str(image_path.relative_to(REPOSITORY_ROOT)),
        "experiments": records,
    }
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
