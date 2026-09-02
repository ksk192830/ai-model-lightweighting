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
from kips_lightweighting.registry import ExperimentRegistry  # noqa: E402


SUITES = {
    "ready4": ("B01", "B02", "B03", "R01"),
    "final8": ("B01", "B02", "B03", "R01", "S01", "C01", "M01", "M02"),
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


def artifact_source_id(experiment_id: str, experiment: dict) -> str:
    return str(experiment.get("artifact_source", experiment_id))


def suite_experiments(
    registry: ExperimentRegistry,
    suite: str,
) -> tuple[str, ...]:
    """Resolve the structured control from C01's registered selection."""
    experiments = SUITES[suite]
    if suite != "final8":
        return experiments
    selected = str(
        registry.get("C01").get(
            "selected_experiment",
            registry.get("C01").get("artifact_source", "S01"),
        )
    )
    return tuple(selected if item == "S01" else item for item in experiments)


def suite_blockers(
    registry: ExperimentRegistry,
    experiments: tuple[str, ...],
    *,
    require_artifact_onnx: bool = True,
) -> list[str]:
    """Reject missing ONNX and recovery prototypes before engine creation."""
    blockers = []
    checked_sources = set()
    for experiment_id in experiments:
        experiment = registry.get(experiment_id)
        source_id = artifact_source_id(experiment_id, experiment)
        selected_id = experiment.get("selected_experiment")
        if selected_id is not None and str(selected_id) != source_id:
            blockers.append(
                f"{experiment_id}: selected_experiment={selected_id} does not "
                f"match artifact_source={source_id}"
            )
        if source_id in checked_sources:
            continue
        checked_sources.add(source_id)
        source = registry.get(source_id)
        fine_tuning = source.get("fine_tuning", {})
        if fine_tuning.get("required") and not fine_tuning.get("completed"):
            blockers.append(
                f"{source_id}: recovery fine-tuning is not completed"
            )
        if (
            require_artifact_onnx
            and not artifact_paths(source_id, "front").onnx.is_file()
        ):
            blockers.append(f"{source_id}: source ONNX is missing")

    if any(registry.get(item)["precision"] == "int8" for item in experiments):
        calibration = REPOSITORY_ROOT / registry.defaults["reproducibility"][
            "front_calibration_dir"
        ]
        if not calibration.is_dir():
            blockers.append(f"B03: calibration directory is missing: {calibration}")
    return blockers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=tuple(SUITES), default="final8")
    parser.add_argument(
        "--image",
        type=Path,
        default=Path(
            "data/training/front_session_split_v1/test/"
            "frame_000005_20251222_185050_737914_png."
            "rf.fe04634c99d43c9e986eaf41db38fd7a.jpg"
        ),
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
    registry = ExperimentRegistry.load()
    experiments = suite_experiments(registry, args.suite)
    blockers = suite_blockers(registry, experiments)
    if blockers:
        print("suite is not ready:")
        for blocker in blockers:
            print(f"- {blocker}")
        return 2
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
            "Install requirements.txt on the NVIDIA notebook."
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
