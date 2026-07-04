#!/usr/bin/env python3
"""Evaluate a TensorRT RF-DETR engine on a labeled COCO split."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--camera", choices=("front", "rear"), required=True)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument(
        "--miou-threshold",
        type=float,
        default=0.25,
        help="Confidence threshold used for semantic mask mIoU (default: 0.25).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Optional warmup runs; unnecessary for accuracy-only evaluation.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "paper_metrics.csv",
        help="Existing benchmark CSV whose matching row will receive mask metrics.",
    )
    parser.add_argument(
        "--no-update-csv",
        action="store_true",
        help="Write only the detailed JSON result.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "coco-evaluation",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def summarize_coco(evaluation: object) -> dict[str, float]:
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


MASK_CSV_FIELDS = ("Mask AP", "Mask AP50", "Mask AP75", "Mask mIoU")


def annotation_mask(annotation: dict, height: int, width: int) -> np.ndarray:
    """Decode a COCO polygon/RLE annotation into a boolean image mask."""
    from pycocotools import mask as mask_utils

    segmentation = annotation.get("segmentation")
    if not segmentation:
        return np.zeros((height, width), dtype=bool)
    if isinstance(segmentation, dict):
        rle = dict(segmentation)
        if isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects(rle, height, width)
        elif isinstance(rle.get("counts"), str):
            rle["counts"] = rle["counts"].encode("ascii")
    else:
        rle = mask_utils.frPyObjects(segmentation, height, width)
        if isinstance(rle, list):
            rle = mask_utils.merge(rle)
    return mask_utils.decode(rle).astype(bool)


def semantic_iou(
    intersections: dict[int, int],
    unions: dict[int, int],
) -> tuple[float, dict[int, float]]:
    """Return class-mean dataset IoU and per-category IoU."""
    valid = {
        key: intersections.get(key, 0) / value
        for key, value in unions.items()
        if value
    }
    mean_iou = float(np.mean(list(valid.values()))) if valid else float("nan")
    return mean_iou, valid


def update_metrics_csv(
    path: Path,
    engine: Path,
    experiment: str,
    metrics: dict[str, float],
) -> None:
    """Add mask columns and update exactly one existing benchmark row atomically."""
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    for field in MASK_CSV_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)

    relative_engine = str(engine.relative_to(REPOSITORY_ROOT))
    matches = [
        row
        for row in rows
        if row.get("Model Path") == relative_engine
        or (
            f"/{experiment}/" in f"/{row.get('Model Path', '')}"
            and Path(row.get("Model Path", "")).name == engine.name
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one CSV row for {relative_engine}, found {len(matches)}."
        )
    matches[0].update({key: f"{value:.6f}" for key, value in metrics.items()})

    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary:
        writer = csv.DictWriter(temporary, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def main() -> int:
    args = parse_args()
    import torch
    from PIL import Image
    from pycocotools import mask as mask_utils
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    sys.path.insert(0, str(REPOSITORY_ROOT / "scripts" / "evaluation"))
    from benchmark_baseline import TensorRTRunner

    if not 0.0 <= args.threshold <= 1.0 or not 0.0 <= args.miou_threshold <= 1.0:
        raise ValueError("--threshold and --miou-threshold must be between zero and one.")
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

    segmentation_predictions: list[dict] = []
    intersections: dict[int, int] = defaultdict(int)
    unions: dict[int, int] = defaultdict(int)
    for index, image_id in enumerate(image_ids, start=1):
        info = coco.loadImgs([image_id])[0]
        image = Image.open(
            image_path(dataset, image_dir, info["file_name"])
        ).convert("RGB")
        detections = runner.predict(image, threshold=args.threshold)

        masks = detections.mask
        predicted_semantic = {
            category_id: np.zeros((info["height"], info["width"]), dtype=bool)
            for category_id in category_ids
        }
        for detection_index in range(len(detections)):
            category_id = int(detections.class_id[detection_index])
            if category_id not in category_ids:
                continue
            common = {
                "image_id": image_id,
                "category_id": category_id,
                "score": float(detections.confidence[detection_index]),
            }
            if masks is not None:
                mask = masks[detection_index].astype(bool)
                encoded = mask_utils.encode(
                    np.asfortranarray(mask.astype(np.uint8))
                )
                encoded["counts"] = encoded["counts"].decode("ascii")
                segmentation_predictions.append(
                    {**common, "segmentation": encoded}
                )
                if common["score"] >= args.miou_threshold:
                    predicted_semantic[category_id] |= mask

        ground_truth_semantic = {
            category_id: np.zeros((info["height"], info["width"]), dtype=bool)
            for category_id in category_ids
        }
        for annotation in coco.loadAnns(coco.getAnnIds(imgIds=[image_id])):
            category_id = int(annotation["category_id"])
            if category_id in ground_truth_semantic:
                ground_truth_semantic[category_id] |= annotation_mask(
                    annotation, info["height"], info["width"]
                )
        for category_id in category_ids:
            prediction = predicted_semantic[category_id]
            target = ground_truth_semantic[category_id]
            intersections[category_id] += int(np.count_nonzero(prediction & target))
            unions[category_id] += int(np.count_nonzero(prediction | target))
        print(f"\rimages: {index}/{len(image_ids)}", end="", flush=True)
    print()

    if not segmentation_predictions:
        raise RuntimeError("The engine produced no segmentation masks.")
    detected = coco.loadRes(segmentation_predictions)
    evaluation = COCOeval(coco, detected, "segm")
    evaluation.params.imgIds = image_ids
    evaluation.params.catIds = category_ids
    evaluation.evaluate()
    evaluation.accumulate()
    evaluation.summarize()
    segmentation_metrics = summarize_coco(evaluation)
    miou, category_ious = semantic_iou(intersections, unions)

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
        "miou_threshold": args.miou_threshold,
        "warmup": args.warmup,
        "hardware": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "tensorrt": runner.trt.__version__,
        "prediction_count": len(segmentation_predictions),
        "metrics": {
            "segm": segmentation_metrics,
            "semantic_miou": miou,
            "semantic_iou_by_category": {
                str(key): value for key, value in category_ious.items()
            },
        },
    }
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{args.experiment}-{args.camera}.json"
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if not args.no_update_csv:
        csv_path = resolve(args.csv)
        update_metrics_csv(
            csv_path,
            engine,
            args.experiment,
            {
                "Mask AP": segmentation_metrics["ap"],
                "Mask AP50": segmentation_metrics["ap50"],
                "Mask AP75": segmentation_metrics["ap75"],
                "Mask mIoU": miou,
            },
        )
        print(f"updated CSV: {csv_path}")
    print(f"result: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
