#!/usr/bin/env python3
"""Evaluate an RF-DETR segmentation ONNX model on a labeled COCO split.

The preprocessing, RF-DETR ``PostProcess``, confidence filtering, COCO metrics,
and semantic mIoU calculation intentionally match the PyTorch and TensorRT
evaluators in this directory. ONNX Runtime is restricted to its CPU provider so
this evaluator can run alongside GPU recovery training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
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
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
METRIC_PATHS = {
    "bbox_ap": ("bbox", "ap"),
    "mask_ap": ("segm", "ap"),
    "semantic_miou": ("semantic_miou",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment")
    parser.add_argument("--camera", choices=("front", "rear"))
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--name", help="Result file stem; defaults to EXPERIMENT-CAMERA-onnx.")
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
        "--reference-result",
        type=Path,
        help="Optional PyTorch result JSON used to calculate accuracy deltas.",
    )
    parser.add_argument(
        "--recheck-result",
        type=Path,
        help=(
            "Recalculate only an existing result's PyTorch parity metadata; "
            "does not run inference or change measured metrics."
        ),
    )
    parser.add_argument(
        "--max-absolute-ap-delta",
        type=float,
        default=float(EVALUATION_PROTOCOL["onnx_dataset_parity_max_absolute_delta"]),
        help="Maximum absolute bbox/mask AP delta for PyTorch parity (default: 0.005).",
    )
    parser.add_argument(
        "--max-absolute-miou-delta",
        type=float,
        default=float(EVALUATION_PROTOCOL["onnx_dataset_parity_max_absolute_delta"]),
        help="Maximum absolute semantic mIoU delta for PyTorch parity (default: 0.005).",
    )
    parser.add_argument("--intra-op-threads", type=int, default=3)
    parser.add_argument("--inter-op-threads", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "coco-evaluation",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def repository_relative(path: Path) -> str:
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


def nested_metric(metrics: dict, path: tuple[str, ...]) -> float:
    value: object = metrics
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"Reference result is missing metrics.{'.'.join(path)}")
        value = value[key]
    return float(value)


def accuracy_parity(
    candidate: dict,
    reference_result: Path,
    *,
    candidate_prediction_count: int,
    image_count: int,
    threshold: float,
    miou_threshold: float,
    max_absolute_ap_delta: float,
    max_absolute_miou_delta: float,
) -> dict[str, object]:
    """Compare ONNX metrics with a result produced by the PyTorch evaluator."""
    reference = json.loads(reference_result.read_text(encoding="utf-8"))
    reference_metrics = reference.get("metrics")
    if not isinstance(reference_metrics, dict):
        raise ValueError("Reference result has no metrics object.")

    protocol_checks = {
        "image_count": reference.get("image_count") == image_count,
        "threshold": reference.get("threshold") == threshold,
        "miou_threshold": reference.get("miou_threshold") == miou_threshold,
    }
    reference_prediction_count = int(reference.get("prediction_count", 0))
    prediction_count_delta = candidate_prediction_count - reference_prediction_count
    delta_absolute: dict[str, float] = {}
    reference_values: dict[str, float] = {}
    candidate_values: dict[str, float] = {}
    within_tolerance: dict[str, bool] = {}
    for name, path in METRIC_PATHS.items():
        reference_value = nested_metric(reference_metrics, path)
        candidate_value = nested_metric(candidate, path)
        delta = candidate_value - reference_value
        tolerance = (
            max_absolute_miou_delta
            if name == "semantic_miou"
            else max_absolute_ap_delta
        )
        reference_values[name] = reference_value
        candidate_values[name] = candidate_value
        delta_absolute[name] = delta
        within_tolerance[name] = abs(delta) <= tolerance

    return {
        "comparison_scope": (
            "dataset-level accuracy parity; raw tensor equivalence is validated separately"
        ),
        "reference_result": repository_relative(reference_result),
        "reference_result_sha256": sha256(reference_result),
        "protocol_checks": protocol_checks,
        "prediction_count": {
            "reference": reference_prediction_count,
            "candidate": candidate_prediction_count,
            "delta": prediction_count_delta,
            "absolute_delta": abs(prediction_count_delta),
            "absolute_delta_ratio": (
                abs(prediction_count_delta) / reference_prediction_count
                if reference_prediction_count
                else None
            ),
            "note": (
                "Net count difference after identical score > threshold filtering; "
                "a proxy for threshold-boundary crossings, not a paired crossing count."
            ),
        },
        "reference": reference_values,
        "candidate": candidate_values,
        "delta_absolute": delta_absolute,
        "tolerances": {
            "bbox_ap": max_absolute_ap_delta,
            "mask_ap": max_absolute_ap_delta,
            "semantic_miou": max_absolute_miou_delta,
        },
        "within_tolerance": within_tolerance,
        "passed": all(protocol_checks.values()) and all(within_tolerance.values()),
    }


def recheck_accuracy_parity(
    result_path: Path,
    reference_result: Path,
    *,
    max_absolute_ap_delta: float,
    max_absolute_miou_delta: float,
) -> dict[str, object]:
    """Atomically refresh parity policy metadata without rerunning inference."""
    result = json.loads(result_path.read_text(encoding="utf-8"))
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Candidate result has no metrics object.")
    result["pytorch_parity"] = accuracy_parity(
        metrics,
        reference_result,
        candidate_prediction_count=int(result["prediction_count"]),
        image_count=int(result["image_count"]),
        threshold=float(result["threshold"]),
        miou_threshold=float(result["miou_threshold"]),
        max_absolute_ap_delta=max_absolute_ap_delta,
        max_absolute_miou_delta=max_absolute_miou_delta,
    )
    result["parity_rechecked_at_utc"] = datetime.now(timezone.utc).isoformat()
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=result_path.parent,
        delete=False,
    ) as temporary:
        json.dump(result, temporary, indent=2, ensure_ascii=False)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(result_path)
    return result


class ONNXRunner:
    """Run one static batch-1 RF-DETR ONNX model on CPU."""

    def __init__(
        self,
        model_path: Path,
        *,
        intra_op_threads: int,
        inter_op_threads: int,
    ) -> None:
        import onnxruntime as ort
        from rfdetr.models.postprocess import PostProcess

        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_op_threads
        options.inter_op_num_threads = inter_op_threads
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.providers = self.session.get_providers()
        if self.providers != ["CPUExecutionProvider"]:
            raise RuntimeError(
                f"Expected only CPUExecutionProvider, got {self.providers}."
            )

        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise RuntimeError(f"Expected exactly one ONNX input, got {len(inputs)}.")
        self.input_name = inputs[0].name
        shape = inputs[0].shape
        if (
            len(shape) != 4
            or shape[0] != 1
            or shape[1] != 3
            or not isinstance(shape[2], int)
            or not isinstance(shape[3], int)
        ):
            raise RuntimeError(f"Expected a static NCHW batch-1 input, got {shape}.")
        if shape[2] != shape[3]:
            raise RuntimeError(f"Expected square RF-DETR input, got {shape[2:]}.")
        self.input_height = shape[2]
        self.input_width = shape[3]

        outputs = {output.name: output for output in self.session.get_outputs()}
        output_names = set(outputs)
        self.output_names = ["dets", "labels", "masks"]
        if not set(self.output_names).issubset(output_names):
            raise RuntimeError(
                "Expected dets/labels/masks ONNX outputs, got "
                f"{sorted(output_names)}."
            )
        det_shape = outputs["dets"].shape
        label_shape = outputs["labels"].shape
        mask_shape = outputs["masks"].shape
        if (
            len(det_shape) != 3
            or det_shape[0] != 1
            or not isinstance(det_shape[1], int)
            or det_shape[1] < 1
            or det_shape[2] != 4
        ):
            raise RuntimeError(f"Expected static dets output, got {det_shape}.")
        if (
            len(label_shape) != 3
            or label_shape[0] != det_shape[0]
            or label_shape[1] != det_shape[1]
            or not isinstance(label_shape[2], int)
        ):
            raise RuntimeError(f"labels output does not align with dets: {label_shape}.")
        if (
            len(mask_shape) != 4
            or mask_shape[0] != det_shape[0]
            or mask_shape[1] != det_shape[1]
            or not all(isinstance(value, int) for value in mask_shape[2:])
        ):
            raise RuntimeError(f"masks output does not align with dets: {mask_shape}.")
        # RF-DETR fine-tuned checkpoints use num_select == num_queries. Using
        # 300 here (the generic TensorRT runner default) would introduce extra
        # low-confidence query/class pairs at the AP threshold of 0.001 and
        # would therefore not reproduce model.predict().
        self.num_select = det_shape[1]
        self.postprocess = PostProcess(num_select=self.num_select)
        self.inference_seconds: list[float] = []

    def predict(self, image: object, threshold: float) -> object:
        import torch
        import torchvision.transforms.functional as functional
        from supervision import Detections

        original_width, original_height = image.size
        tensor = functional.to_tensor(image)
        tensor = functional.resize(tensor, [self.input_height, self.input_width])
        tensor = functional.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
        input_array = np.ascontiguousarray(tensor.unsqueeze(0).numpy())

        started = time.perf_counter()
        raw_outputs = self.session.run(
            self.output_names,
            {self.input_name: input_array},
        )
        self.inference_seconds.append(time.perf_counter() - started)

        outputs = {
            "pred_boxes": torch.from_numpy(raw_outputs[0]),
            "pred_logits": torch.from_numpy(raw_outputs[1]),
            "pred_masks": torch.from_numpy(raw_outputs[2]),
        }
        result = self.postprocess(
            outputs,
            target_sizes=torch.tensor([[original_height, original_width]]),
        )[0]
        keep = result["scores"] > threshold
        detections = Detections(
            xyxy=result["boxes"][keep].float().numpy(),
            confidence=result["scores"][keep].float().numpy(),
            class_id=result["labels"][keep].numpy(),
            mask=result["masks"][keep].squeeze(1).numpy(),
        )
        detections.data["source_shape"] = np.tile(
            np.array([original_height, original_width], dtype=np.int64),
            (len(detections), 1),
        )
        return detections


def main() -> int:
    args = parse_args()
    if args.max_absolute_ap_delta < 0 or args.max_absolute_miou_delta < 0:
        raise ValueError("Parity tolerances must be zero or greater.")
    if args.recheck_result is not None:
        if args.reference_result is None:
            raise ValueError("--recheck-result requires --reference-result.")
        result_path = resolve(args.recheck_result)
        reference_result = resolve(args.reference_result)
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
        if not reference_result.is_file():
            raise FileNotFoundError(reference_result)
        refreshed = recheck_accuracy_parity(
            result_path,
            reference_result,
            max_absolute_ap_delta=args.max_absolute_ap_delta,
            max_absolute_miou_delta=args.max_absolute_miou_delta,
        )
        parity = refreshed["pytorch_parity"]
        print(f"rechecked result: {result_path}")
        print(f"PyTorch parity: {'PASS' if parity['passed'] else 'FAIL'}")
        return 0 if parity["passed"] else 2

    if args.experiment is None or args.camera is None:
        raise ValueError("Inference requires --experiment and --camera.")
    if not 0.0 <= args.threshold <= 1.0 or not 0.0 <= args.miou_threshold <= 1.0:
        raise ValueError("--threshold and --miou-threshold must be between zero and one.")
    if args.intra_op_threads < 1 or args.inter_op_threads < 1:
        raise ValueError("ONNX Runtime thread counts must be at least one.")

    import onnxruntime as ort
    import torch
    from PIL import Image
    from pycocotools import mask as mask_utils
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    sys.path.insert(0, str(REPOSITORY_ROOT / "scripts" / "evaluation"))
    from evaluate_coco_tensorrt import (
        annotation_mask,
        image_path,
        semantic_iou,
        summarize_coco,
    )

    model_path = resolve(
        args.onnx
        or Path(f"artifacts/experiments/{args.experiment}/{args.camera}/model.onnx")
    )
    dataset = resolve(
        args.dataset_dir or Path("data/training/front_session_split_v1/test")
    )
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
    runner = ONNXRunner(
        model_path,
        intra_op_threads=args.intra_op_threads,
        inter_op_threads=args.inter_op_threads,
    )

    started_at = datetime.now(timezone.utc)
    wall_started = time.perf_counter()
    per_image_seconds: list[float] = []
    bbox_predictions: list[dict] = []
    segmentation_predictions: list[dict] = []
    intersections: dict[int, int] = defaultdict(int)
    unions: dict[int, int] = defaultdict(int)
    for index, image_id in enumerate(image_ids, start=1):
        item_started = time.perf_counter()
        info = coco.loadImgs([image_id])[0]
        with Image.open(image_path(dataset, image_dir, info["file_name"])) as source:
            detections = runner.predict(source.convert("RGB"), threshold=args.threshold)

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
                encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
                encoded["counts"] = encoded["counts"].decode("ascii")
                segmentation_predictions.append({**common, "segmentation": encoded})
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
        per_image_seconds.append(time.perf_counter() - item_started)
        mean_seconds = statistics.fmean(per_image_seconds)
        eta_seconds = mean_seconds * (len(image_ids) - index)
        print(
            f"\rimages: {index}/{len(image_ids)} | "
            f"mean {mean_seconds:.2f}s | ETA {eta_seconds / 60:.1f}m",
            end="",
            flush=True,
        )
    print()

    if not bbox_predictions:
        raise RuntimeError("The ONNX model produced no detections.")
    if not segmentation_predictions:
        raise RuntimeError("The ONNX model produced no segmentation masks.")

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
    segmentation_metrics = evaluate(segmentation_predictions, "segm")
    miou, category_ious = semantic_iou(intersections, unions)
    wall_seconds = time.perf_counter() - wall_started
    inference_seconds = sum(runner.inference_seconds)
    metrics = {
        "bbox": bbox_metrics,
        "segm": segmentation_metrics,
        "semantic_miou": miou,
        "semantic_iou_by_category": {
            str(key): value for key, value in category_ious.items()
        },
    }
    result: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "started_at_utc": started_at.isoformat(),
        "experiment_id": args.experiment,
        "camera": args.camera,
        "onnx": repository_relative(model_path),
        "onnx_size_bytes": model_path.stat().st_size,
        "onnx_sha256": sha256(model_path),
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
        "resolution": runner.input_height,
        "postprocess_num_select": runner.num_select,
        "provider": "CPUExecutionProvider",
        "providers_active": runner.providers,
        "onnxruntime": ort.__version__,
        "torch": torch.__version__,
        "rfdetr": version("rfdetr"),
        "hardware": platform.processor() or platform.machine(),
        "thread_limits": {
            "intra_op": args.intra_op_threads,
            "inter_op": args.inter_op_threads,
        },
        "prediction_count": len(bbox_predictions),
        "timing": {
            "wall_seconds": wall_seconds,
            "inference_seconds": inference_seconds,
            "inference_mean_ms": 1000.0 * statistics.fmean(runner.inference_seconds),
            "end_to_end_mean_ms": 1000.0 * statistics.fmean(per_image_seconds),
            "end_to_end_images_per_second": len(image_ids) / wall_seconds,
        },
        "metrics": metrics,
    }
    if args.reference_result is not None:
        reference_result = resolve(args.reference_result)
        if not reference_result.is_file():
            raise FileNotFoundError(reference_result)
        result["pytorch_parity"] = accuracy_parity(
            metrics,
            reference_result,
            candidate_prediction_count=len(bbox_predictions),
            image_count=len(image_ids),
            threshold=args.threshold,
            miou_threshold=args.miou_threshold,
            max_absolute_ap_delta=args.max_absolute_ap_delta,
            max_absolute_miou_delta=args.max_absolute_miou_delta,
        )

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{args.name or f'{args.experiment}-{args.camera}-onnx'}.json"
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"result: {output}")
    parity = result.get("pytorch_parity")
    if isinstance(parity, dict):
        print(f"PyTorch parity: {'PASS' if parity['passed'] else 'FAIL'}")
        return 0 if parity["passed"] else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
