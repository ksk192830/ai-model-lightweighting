#!/usr/bin/env python3
"""Evaluate an RF-DETR PyTorch checkpoint on a labeled COCO split.

Unlike evaluate_coco_tensorrt.py this runs the .pth checkpoint directly, so
baseline accuracy can be measured on a training machine before any ONNX or
TensorRT conversion exists. Reports bbox AP, mask AP, and semantic mask mIoU.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="COCO split directory containing _annotations.coco.json.",
    )
    parser.add_argument(
        "--name",
        help="Result file stem; defaults to the checkpoint stem.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument(
        "--miou-threshold",
        type=float,
        default=0.25,
        help="Confidence threshold used for semantic mask mIoU (default: 0.25).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "coco-evaluation",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def main() -> int:
    args = parse_args()
    import torch
    from PIL import Image
    from pycocotools import mask as mask_utils
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
    sys.path.insert(0, str(REPOSITORY_ROOT / "scripts" / "evaluation"))
    from evaluate_coco_tensorrt import (
        annotation_mask,
        image_path,
        semantic_iou,
        summarize_coco,
    )
    from kips_lightweighting.rfdetr_compat import load_rfdetr_checkpoint

    if not 0.0 <= args.threshold <= 1.0 or not 0.0 <= args.miou_threshold <= 1.0:
        raise ValueError(
            "--threshold and --miou-threshold must be between zero and one."
        )
    checkpoint = resolve(args.checkpoint)
    dataset = resolve(args.dataset_dir)
    annotation_path = dataset / "_annotations.coco.json"
    image_dir = dataset / "images"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)

    coco = COCO(str(annotation_path))
    image_ids = sorted(coco.getImgIds())
    category_ids = sorted(
        category_id
        for category_id in coco.getCatIds()
        if coco.getAnnIds(catIds=[category_id])
    )
    num_classes = len(coco.getCatIds())
    model = load_rfdetr_checkpoint(
        checkpoint,
        device=args.device,
        num_classes=num_classes,
    )

    bbox_predictions: list[dict] = []
    segmentation_predictions: list[dict] = []
    intersections: dict[int, int] = defaultdict(int)
    unions: dict[int, int] = defaultdict(int)
    for index, image_id in enumerate(image_ids, start=1):
        info = coco.loadImgs([image_id])[0]
        image = Image.open(
            image_path(dataset, image_dir, info["file_name"])
        ).convert("RGB")
        detections = model.predict(image, threshold=args.threshold)

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
        raise RuntimeError("The checkpoint produced no detections.")

    def evaluate(predictions: list[dict], iou_type: str) -> dict[str, float]:
        detected = coco.loadRes(predictions)
        evaluation = COCOeval(coco, detected, iou_type)
        evaluation.params.imgIds = image_ids
        evaluation.params.catIds = category_ids
        evaluation.evaluate()
        evaluation.accumulate()
        evaluation.summarize()
        return summarize_coco(evaluation)

    bbox_metrics = evaluate(bbox_predictions, "bbox")
    metrics: dict[str, object] = {"bbox": bbox_metrics}
    if segmentation_predictions:
        metrics["segm"] = evaluate(segmentation_predictions, "segm")
        miou, category_ious = semantic_iou(intersections, unions)
        metrics["semantic_miou"] = miou
        metrics["semantic_iou_by_category"] = {
            str(key): value for key, value in category_ious.items()
        }

    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "dataset": str(dataset),
        "image_count": len(image_ids),
        "category_ids": category_ids,
        "categories": {
            str(category["id"]): category["name"]
            for category in coco.loadCats(coco.getCatIds())
        },
        "threshold": args.threshold,
        "miou_threshold": args.miou_threshold,
        "hardware": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "cpu"
        ),
        "torch": torch.__version__,
        "prediction_count": len(bbox_predictions),
        "metrics": metrics,
    }
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or checkpoint.stem
    output = output_dir / f"{name}.json"
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"result: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
