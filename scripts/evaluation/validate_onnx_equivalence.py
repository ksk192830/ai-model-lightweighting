#!/usr/bin/env python3
"""Compare RF-DETR PyTorch and ONNX outputs on deterministic inputs.

Real preprocessed images are preferred. Random tensors remain available for a
quick graph smoke test, but they can amplify insignificant DETR proposal ties.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
import torchvision.transforms.functional as functional
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from kips_lightweighting.rfdetr_compat import load_rfdetr_checkpoint  # noqa: E402


OUTPUT_MAP = {
    "dets": "pred_boxes",
    "labels": "pred_logits",
    "masks": "pred_masks",
}
_GRAPH_PROTOCOL = yaml.safe_load(
    (REPOSITORY_ROOT / "configs/experiments/defaults.yaml").read_text(
        encoding="utf-8"
    )
)["evaluation_protocol"]["graph_equivalence"]
DEFAULT_LIMITS = {
    "dets_max": float(_GRAPH_PROTOCOL["boxes_max_absolute_error"]),
    "labels_max": float(_GRAPH_PROTOCOL["scores_max_absolute_error"]),
    "active_scores_max": float(_GRAPH_PROTOCOL["scores_max_absolute_error"]),
    "masks_max": float(_GRAPH_PROTOCOL["masks_max_absolute_error"]),
    "masks_mean": float(_GRAPH_PROTOCOL["masks_mean_absolute_error"]),
    "active_mask_binary_agreement": float(
        _GRAPH_PROTOCOL["active_mask_binary_agreement"]
    ),
    "active_membership_agreement": float(
        _GRAPH_PROTOCOL["active_membership_agreement"]
    ),
    "active_class_agreement": float(_GRAPH_PROTOCOL["active_class_agreement"]),
}


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=504)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--samples", type=int, default=int(_GRAPH_PROTOCOL["samples"]))
    parser.add_argument("--seed", type=int, default=int(_GRAPH_PROTOCOL["sample_seed"]))
    parser.add_argument(
        "--image-dir",
        type=Path,
        help="Use a reproducible sample of production images from this directory.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=float(_GRAPH_PROTOCOL["confidence_threshold"]),
    )
    return parser.parse_args()


def image_files(directory: Path, samples: int, seed: int) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    candidates = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )
    if len(candidates) < samples:
        raise ValueError(
            f"Requested {samples} image samples, but found {len(candidates)} in {directory}."
        )
    return random.Random(seed).sample(candidates, samples)


def preprocess_image(path: Path, resolution: int) -> torch.Tensor:
    with Image.open(path) as image:
        tensor = functional.to_tensor(image.convert("RGB"))
    tensor = functional.resize(tensor, [resolution, resolution])
    tensor = functional.normalize(
        tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    return tensor.unsqueeze(0)


def main() -> int:
    args = parse_args()
    checkpoint = resolve(args.checkpoint)
    onnx_path = resolve(args.onnx)
    output_path = resolve(args.output)
    if args.samples < 1:
        raise ValueError("--samples must be at least one")
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise ValueError("--confidence-threshold must be between zero and one")

    wrapper = load_rfdetr_checkpoint(
        checkpoint,
        device="cpu",
        num_classes=args.num_classes,
    )
    model = wrapper.model.model.eval()
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    maxima = {name: 0.0 for name in OUTPUT_MAP}
    means = {name: [] for name in OUTPUT_MAP}
    shapes: dict[str, list[int]] = {}
    active_maxima = {name: 0.0 for name in OUTPUT_MAP}
    active_score_maximum = 0.0
    active_query_count = 0
    active_membership_equal = 0
    active_membership_total = 0
    active_class_equal = 0
    active_class_total = 0
    mask_binary_equal = 0
    mask_binary_total = 0
    selected_images: list[str] = []
    if args.image_dir is not None:
        directory = resolve(args.image_dir)
        inputs = [
            (preprocess_image(path, args.resolution), path)
            for path in image_files(directory, args.samples, args.seed)
        ]
        input_mode = "real-images"
    else:
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        inputs = [
            (
                torch.rand(
                    1,
                    3,
                    args.resolution,
                    args.resolution,
                    generator=generator,
                ),
                None,
            )
            for _ in range(args.samples)
        ]
        input_mode = "random-tensors"

    for tensor, image_path in inputs:
        if image_path is not None:
            selected_images.append(str(image_path.relative_to(REPOSITORY_ROOT)))
        with torch.no_grad():
            pytorch_outputs = model(tensor)
        onnx_outputs = session.run(
            list(OUTPUT_MAP),
            {"input": tensor.numpy()},
        )
        for name, onnx_output in zip(OUTPUT_MAP, onnx_outputs):
            pytorch_output = pytorch_outputs[OUTPUT_MAP[name]].cpu().numpy()
            difference = np.abs(pytorch_output - onnx_output)
            maxima[name] = max(maxima[name], float(difference.max()))
            means[name].append(float(difference.mean()))
            shapes[name] = list(pytorch_output.shape)

        pytorch_scores = torch.sigmoid(pytorch_outputs["pred_logits"]).numpy()
        onnx_scores = 1.0 / (1.0 + np.exp(-onnx_outputs[1]))
        pytorch_confidence = pytorch_scores.max(axis=2)[0]
        onnx_confidence = onnx_scores.max(axis=2)[0]
        pytorch_active = pytorch_confidence >= args.confidence_threshold
        onnx_active = onnx_confidence >= args.confidence_threshold
        active_membership_equal += int(
            np.count_nonzero(pytorch_active == onnx_active)
        )
        active_membership_total += int(pytorch_active.size)
        active = np.maximum(pytorch_active, onnx_active)
        active_query_count += int(active.sum())
        if active.any():
            active_score_maximum = max(
                active_score_maximum,
                float(
                    np.abs(
                        pytorch_scores[0, active]
                        - onnx_scores[0, active]
                    ).max()
                ),
            )
            active_class_equal += int(
                np.count_nonzero(
                    pytorch_scores[0, active].argmax(axis=1)
                    == onnx_scores[0, active].argmax(axis=1)
                )
            )
            active_class_total += int(active.sum())
            for name, onnx_output in zip(OUTPUT_MAP, onnx_outputs):
                pytorch_output = pytorch_outputs[OUTPUT_MAP[name]].cpu().numpy()
                difference = np.abs(
                    pytorch_output[0, active] - onnx_output[0, active]
                )
                active_maxima[name] = max(
                    active_maxima[name], float(difference.max())
                )
            pytorch_masks = pytorch_outputs["pred_masks"].cpu().numpy()[0, active]
            onnx_masks = onnx_outputs[2][0, active]
            mask_binary_equal += int(
                np.count_nonzero((pytorch_masks > 0.0) == (onnx_masks > 0.0))
            )
            mask_binary_total += int(pytorch_masks.size)

    comparisons = {
        name: {
            "shape": shapes[name],
            "max_abs_error": maxima[name],
            "mean_abs_error": float(np.mean(means[name])),
            "max_limit": DEFAULT_LIMITS[f"{name}_max"],
            **(
                {"mean_limit": DEFAULT_LIMITS["masks_mean"]}
                if name == "masks"
                else {}
            ),
            "passed": (
                maxima[name] <= DEFAULT_LIMITS[f"{name}_max"]
                and (
                    name != "masks"
                    or float(np.mean(means[name])) <= DEFAULT_LIMITS["masks_mean"]
                )
            ),
        }
        for name in OUTPUT_MAP
    }
    active_mask_binary_agreement = (
        mask_binary_equal / mask_binary_total if mask_binary_total else None
    )
    active_membership_agreement = (
        active_membership_equal / active_membership_total
        if active_membership_total
        else None
    )
    active_class_agreement = (
        active_class_equal / active_class_total if active_class_total else None
    )
    production_comparison = {
        "confidence_threshold": args.confidence_threshold,
        "active_query_count": active_query_count,
        "max_abs_error": active_maxima,
        "active_score_max_abs_error": active_score_maximum,
        "active_score_max_limit": DEFAULT_LIMITS["active_scores_max"],
        "active_membership_agreement": active_membership_agreement,
        "active_membership_agreement_limit": DEFAULT_LIMITS[
            "active_membership_agreement"
        ],
        "active_class_agreement": active_class_agreement,
        "active_class_agreement_limit": DEFAULT_LIMITS[
            "active_class_agreement"
        ],
        "mask_binary_agreement": active_mask_binary_agreement,
        "mask_binary_agreement_limit": DEFAULT_LIMITS[
            "active_mask_binary_agreement"
        ],
        "passed": (
            active_query_count > 0
            and active_maxima["dets"] <= DEFAULT_LIMITS["dets_max"]
            and active_score_maximum <= DEFAULT_LIMITS["active_scores_max"]
            and active_maxima["masks"] <= DEFAULT_LIMITS["masks_max"]
            and active_membership_agreement is not None
            and active_membership_agreement
            >= DEFAULT_LIMITS["active_membership_agreement"]
            and active_class_agreement is not None
            and active_class_agreement
            >= DEFAULT_LIMITS["active_class_agreement"]
            and active_mask_binary_agreement is not None
            and active_mask_binary_agreement
            >= DEFAULT_LIMITS["active_mask_binary_agreement"]
        ),
    }
    raw_comparison_passed = all(
        item["passed"] for item in comparisons.values()
    )
    passed = (
        production_comparison["passed"]
        if input_mode == "real-images"
        else raw_comparison_passed
    )
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint.relative_to(REPOSITORY_ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
        "onnx": str(onnx_path.relative_to(REPOSITORY_ROOT)),
        "onnx_sha256": sha256(onnx_path),
        "input_shape": [1, 3, args.resolution, args.resolution],
        "samples": args.samples,
        "seed": args.seed,
        "input_mode": input_mode,
        "selected_images": selected_images,
        "provider": "CPUExecutionProvider",
        "protocol_source": "configs/experiments/defaults.yaml",
        "validation_policy": (
            "Production-image mode requires identical confidence-threshold "
            "membership and active-query classes, bounded confidence-score, "
            "box, and mask errors, plus binary-mask agreement. Raw logits and "
            "all-query errors remain diagnostic because low-confidence DETR "
            "proposal ties are not predictions."
            if input_mode == "real-images"
            else "Random-tensor mode requires all-query raw-error limits."
        ),
        "comparisons": comparisons,
        "raw_comparison_passed": raw_comparison_passed,
        "production_comparison": production_comparison,
        "passed": passed,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
