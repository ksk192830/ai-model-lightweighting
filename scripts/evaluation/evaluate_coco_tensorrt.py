#!/usr/bin/env python3
"""Evaluate a TensorRT RF-DETR engine on a labeled COCO split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_PROTOCOL = yaml.safe_load(
    (REPOSITORY_ROOT / "configs/experiments/defaults.yaml").read_text(
        encoding="utf-8"
    )
)["evaluation_protocol"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--camera", choices=("front", "rear"), required=True)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument(
        "--threshold",
        type=float,
        default=float(EVALUATION_PROTOCOL["coco_ap_confidence_threshold"]),
    )
    parser.add_argument(
        "--miou-threshold",
        type=float,
        default=float(EVALUATION_PROTOCOL["semantic_miou_confidence_threshold"]),
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


def repository_relative(path: Path) -> str:
    """Repo-relative when possible; datasets may live outside the repo."""
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def summarize_coco_by_category(
    evaluation: object,
    category_ids: list[int],
) -> dict[str, dict[str, float]]:
    """Extract auditable per-category AP/AR from one COCOeval accumulation.

    COCOeval's headline AP is a macro average across categories.  Publishing the
    corresponding per-category values makes class imbalance and regressions
    visible without changing the canonical overall score.
    """
    precision = evaluation.eval["precision"]
    recall = evaluation.eval["recall"]
    iou_thresholds = np.asarray(evaluation.params.iouThrs)
    max_detection_index = len(evaluation.params.maxDets) - 1

    def valid_mean(values: np.ndarray) -> float:
        valid = values[values > -1]
        return float(np.mean(valid)) if valid.size else float("nan")

    threshold_indices = {
        "ap50": int(np.argmin(np.abs(iou_thresholds - 0.50))),
        "ap75": int(np.argmin(np.abs(iou_thresholds - 0.75))),
    }
    rows: dict[str, dict[str, float]] = {}
    for category_index, category_id in enumerate(category_ids):
        rows[str(category_id)] = {
            "ap": valid_mean(
                precision[:, :, category_index, 0, max_detection_index]
            ),
            "ap50": valid_mean(
                precision[
                    threshold_indices["ap50"],
                    :,
                    category_index,
                    0,
                    max_detection_index,
                ]
            ),
            "ap75": valid_mean(
                precision[
                    threshold_indices["ap75"],
                    :,
                    category_index,
                    0,
                    max_detection_index,
                ]
            ),
            "ar100": valid_mean(
                recall[:, category_index, 0, max_detection_index]
            ),
        }
    return rows


def image_path(dataset: Path, image_dir: Path, file_name: str) -> Path:
    direct = dataset / file_name
    return direct if direct.is_file() else image_dir / file_name


COCO_CSV_FIELDS = (
    "BBox AP",
    "BBox AP50",
    "BBox AP75",
    "Mask AP",
    "Mask AP50",
    "Mask AP75",
    "Mask mIoU",
)


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
    for field in COCO_CSV_FIELDS:
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
        args.dataset_dir
        or Path("data/training/front_session_split_v1/test")
    )
    annotation_path = dataset / "_annotations.coco.json"
    image_dir = dataset / "images"
    if not engine.is_file():
        raise FileNotFoundError(engine)
    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)

    coco = COCO(str(annotation_path))
    image_ids = sorted(coco.getImgIds())
    expected_images = int(EVALUATION_PROTOCOL["expected_images"])
    if len(image_ids) != expected_images:
        raise ValueError(
            f"Expected {expected_images} final-test images, found {len(image_ids)}."
        )
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
            x1, y1, x2, y2 = (
                float(value) for value in detections.xyxy[detection_index]
            )
            bbox_predictions.append(
                {**common, "bbox": [x1, y1, x2 - x1, y2 - y1]}
            )
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

    if not bbox_predictions:
        raise RuntimeError("The engine produced no detections.")
    if not segmentation_predictions:
        raise RuntimeError("The engine produced no segmentation masks.")

    def evaluate(
        predictions: list[dict],
        iou_type: str,
    ) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
        detected = coco.loadRes(predictions)
        evaluation = COCOeval(coco, detected, iou_type)
        evaluation.params.imgIds = image_ids
        evaluation.params.catIds = category_ids
        evaluation.evaluate()
        evaluation.accumulate()
        evaluation.summarize()
        return (
            summarize_coco(evaluation),
            summarize_coco_by_category(evaluation, category_ids),
        )

    bbox_metrics, bbox_by_category = evaluate(bbox_predictions, "bbox")
    segmentation_metrics, segmentation_by_category = evaluate(
        segmentation_predictions, "segm"
    )
    miou, category_ious = semantic_iou(intersections, unions)

    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": args.experiment,
        "camera": args.camera,
        "engine": repository_relative(engine),
        "engine_size_bytes": engine.stat().st_size,
        "engine_sha256": sha256(engine),
        "dataset": repository_relative(dataset),
        "annotation_sha256": sha256(annotation_path),
        "image_count": len(image_ids),
        "category_ids": category_ids,
        "categories": {
            str(category["id"]): category["name"]
            for category in coco.loadCats(coco.getCatIds())
        },
        "threshold": args.threshold,
        "miou_threshold": args.miou_threshold,
        "postprocess_num_select": runner.num_select,
        "evaluation_protocol": {
            "source": "configs/experiments/defaults.yaml",
            "bbox": "COCOeval bbox AP@[IoU=0.50:0.05:0.95], maxDets=100",
            "segmentation": "COCOeval segm AP@[IoU=0.50:0.05:0.95], maxDets=100",
            "semantic_miou": (
                "dataset-level per-class intersection/union after unioning "
                "instance masks with confidence >= miou_threshold"
            ),
            "nms": "not applied; DETR query predictions are ranked directly",
        },
        "warmup": args.warmup,
        "hardware": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "tensorrt": runner.trt.__version__,
        "prediction_count": len(bbox_predictions),
        "prediction_count_at_operating_threshold": sum(
            prediction["score"] >= args.miou_threshold
            for prediction in bbox_predictions
        ),
        "metrics": {
            "bbox": bbox_metrics,
            "bbox_by_category": bbox_by_category,
            "segm": segmentation_metrics,
            "segm_by_category": segmentation_by_category,
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
                "BBox AP": bbox_metrics["ap"],
                "BBox AP50": bbox_metrics["ap50"],
                "BBox AP75": bbox_metrics["ap75"],
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
