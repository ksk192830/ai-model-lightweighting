#!/usr/bin/env python3
"""Evaluate Ultralytics segmentation masks on a labeled COCO split."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_coco_tensorrt import (  # noqa: E402
    annotation_mask,
    image_path,
    semantic_iou,
    summarize_coco,
    update_metrics_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument("--miou-threshold", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--csv",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "paper_metrics.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT
        / "results"
        / "coco-evaluation"
        / "parking_front-front.json",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def main() -> int:
    args = parse_args()
    import torch
    import torch.nn.functional as functional
    from pycocotools import mask as mask_utils
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from ultralytics import YOLO

    model_path = resolve(args.model)
    dataset = resolve(args.dataset_dir)
    annotation_path = dataset / "_annotations.coco.json"
    image_dir = dataset / "images"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)

    coco = COCO(str(annotation_path))
    image_ids = sorted(coco.getImgIds())
    category_ids = sorted(
        category_id
        for category_id in coco.getCatIds()
        if coco.getAnnIds(catIds=[category_id])
    )
    category_names = {
        int(category["id"]): str(category["name"])
        for category in coco.loadCats(category_ids)
    }

    model = YOLO(str(model_path))
    model_names = (
        {int(key): str(value) for key, value in model.names.items()}
        if isinstance(model.names, dict)
        else {index: str(value) for index, value in enumerate(model.names)}
    )
    dataset_ids_by_name = {name: category_id for category_id, name in category_names.items()}
    class_id_map = {
        model_id: dataset_ids_by_name[name]
        for model_id, name in model_names.items()
        if name in dataset_ids_by_name
    }
    if set(class_id_map) != set(model_names):
        missing = sorted(set(model_names.values()) - set(dataset_ids_by_name))
        raise RuntimeError(f"Model classes absent from COCO categories: {missing}")
    print(f"class ID mapping: {class_id_map}", flush=True)

    segmentation_predictions: list[dict] = []
    intersections: dict[int, int] = defaultdict(int)
    unions: dict[int, int] = defaultdict(int)
    for index, image_id in enumerate(image_ids, start=1):
        info = coco.loadImgs([image_id])[0]
        path = image_path(dataset, image_dir, info["file_name"])
        result = model.predict(
            source=str(path),
            imgsz=args.imgsz,
            conf=args.threshold,
            iou=args.iou,
            device=args.device.replace("cuda:", ""),
            verbose=False,
        )[0]
        predicted_semantic = {
            category_id: np.zeros((info["height"], info["width"]), dtype=bool)
            for category_id in category_ids
        }
        boxes = result.boxes
        masks = result.masks
        if boxes is not None and masks is not None:
            scores = boxes.conf.detach().cpu().numpy()
            classes = boxes.cls.detach().cpu().numpy().astype(int)
            mask_tensors = masks.data
            if tuple(mask_tensors.shape[-2:]) != (info["height"], info["width"]):
                mask_tensors = functional.interpolate(
                    mask_tensors[:, None].float(),
                    size=(info["height"], info["width"]),
                    mode="nearest",
                )[:, 0]
            mask_arrays = mask_tensors.detach().cpu().numpy() > 0.5
            for score, model_class_id, mask in zip(scores, classes, mask_arrays):
                category_id = class_id_map[int(model_class_id)]
                encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
                encoded["counts"] = encoded["counts"].decode("ascii")
                segmentation_predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": category_id,
                        "score": float(score),
                        "segmentation": encoded,
                    }
                )
                if float(score) >= args.miou_threshold:
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
        raise RuntimeError("The model produced no segmentation masks.")
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
        "model": str(model_path.relative_to(REPOSITORY_ROOT)),
        "dataset": str(dataset.relative_to(REPOSITORY_ROOT)),
        "image_count": len(image_ids),
        "category_ids": category_ids,
        "class_id_mapping": class_id_map,
        "threshold": args.threshold,
        "miou_threshold": args.miou_threshold,
        "hardware": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "prediction_count": len(segmentation_predictions),
        "metrics": {
            "segm": segmentation_metrics,
            "semantic_miou": miou,
            "semantic_iou_by_category": {
                str(key): value for key, value in category_ious.items()
            },
        },
    }
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    update_metrics_csv(
        resolve(args.csv),
        model_path,
        "parking_front",
        {
            "Mask AP": segmentation_metrics["ap"],
            "Mask AP50": segmentation_metrics["ap50"],
            "Mask AP75": segmentation_metrics["ap75"],
            "Mask mIoU": miou,
        },
    )
    print(f"updated CSV: {resolve(args.csv)}")
    print(f"result: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
