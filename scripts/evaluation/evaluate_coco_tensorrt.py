#!/usr/bin/env python3
"""Evaluate a TensorRT RF-DETR engine on a labeled COCO split."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts" / "evaluation"))

from benchmark_baseline import TensorRTRunner  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--camera", choices=("front", "rear"), required=True)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "coco-evaluation",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def summarize_coco(evaluation: COCOeval) -> dict[str, float]:
    names = (
        "ap",
        "ap50",
        "ap75",
        "ap_small",
        "ap_medium",
        "ap_large",
        "ar1",
        "ar10",
        "ar100",
        "ar_small",
        "ar_medium",
        "ar_large",
    )
    return {
        name: float(value)
        for name, value in zip(names, evaluation.stats, strict=True)
    }


def image_path(dataset: Path, image_dir: Path, file_name: str) -> Path:
    direct = dataset / file_name
    return direct if direct.is_file() else image_dir / file_name


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between zero and one.")
    engine = resolve(
        args.engine
        or Path(f"artifacts/experiments/{args.experiment}/{args.camera}/model.engine")
    )
    dataset = resolve(
        args.dataset_dir or Path(f"data/labeled_test/{args.camera}")
    )
    annotation_path = dataset / "_annotations.coco.json"
    image_dir = dataset / "images"
    if not engine.is_file():
        raise FileNotFoundError(engine)
    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)

    coco = COCO(str(annotation_path))
    image_ids = sorted(coco.getImgIds())
    category_ids = sorted(
        category_id
        for category_id in coco.getCatIds()
        if coco.getAnnIds(catIds=[category_id])
    )
    runner = TensorRTRunner(engine)
    first = Image.open(
        image_path(
            dataset,
            image_dir,
            coco.loadImgs([image_ids[0]])[0]["file_name"],
        )
    ).convert("RGB")
    for _ in range(args.warmup):
        runner.predict(first, threshold=args.threshold)
    torch.cuda.synchronize()

    bbox_predictions: list[dict] = []
    segmentation_predictions: list[dict] = []
    timings_ms: list[float] = []
    for index, image_id in enumerate(image_ids, start=1):
        info = coco.loadImgs([image_id])[0]
        image = Image.open(
            image_path(dataset, image_dir, info["file_name"])
        ).convert("RGB")
        torch.cuda.synchronize()
        started = time.perf_counter()
        detections = runner.predict(image, threshold=args.threshold)
        torch.cuda.synchronize()
        timings_ms.append((time.perf_counter() - started) * 1000.0)

        masks = detections.mask
        for detection_index, xyxy in enumerate(detections.xyxy):
            category_id = int(detections.class_id[detection_index])
            if category_id not in category_ids:
                continue
            x1, y1, x2, y2 = (float(value) for value in xyxy)
            x1 = min(max(x1, 0.0), float(info["width"]))
            y1 = min(max(y1, 0.0), float(info["height"]))
            x2 = min(max(x2, x1), float(info["width"]))
            y2 = min(max(y2, y1), float(info["height"]))
            common = {
                "image_id": image_id,
                "category_id": category_id,
                "score": float(detections.confidence[detection_index]),
            }
            bbox_predictions.append(
                {
                    **common,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                }
            )
            if masks is not None:
                encoded = mask_utils.encode(
                    np.asfortranarray(masks[detection_index].astype(np.uint8))
                )
                encoded["counts"] = encoded["counts"].decode("ascii")
                segmentation_predictions.append(
                    {**common, "segmentation": encoded}
                )
        print(f"\rimages: {index}/{len(image_ids)}", end="", flush=True)
    print()

    evaluations = {}
    for evaluation_type, predictions in (
        ("bbox", bbox_predictions),
        ("segm", segmentation_predictions),
    ):
        detected = coco.loadRes(predictions)
        evaluation = COCOeval(coco, detected, evaluation_type)
        evaluation.params.imgIds = image_ids
        evaluation.params.catIds = category_ids
        evaluation.evaluate()
        evaluation.accumulate()
        evaluation.summarize()
        evaluations[evaluation_type] = summarize_coco(evaluation)

    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": args.experiment,
        "camera": args.camera,
        "engine": str(engine.relative_to(REPOSITORY_ROOT)),
        "engine_size_bytes": engine.stat().st_size,
        "dataset": str(dataset.relative_to(REPOSITORY_ROOT)),
        "image_count": len(image_ids),
        "category_ids": category_ids,
        "threshold": args.threshold,
        "warmup": args.warmup,
        "hardware": torch.cuda.get_device_name(0),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "tensorrt": runner.trt.__version__,
        "latency_ms": {
            "mean": statistics.fmean(timings_ms),
            "median": statistics.median(timings_ms),
            "p95": float(np.percentile(timings_ms, 95)),
            "min": min(timings_ms),
            "max": max(timings_ms),
        },
        "fps": 1000.0 / statistics.fmean(timings_ms),
        "prediction_count": len(bbox_predictions),
        "metrics": evaluations,
    }
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{args.experiment}-{args.camera}.json"
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"result: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
