#!/usr/bin/env python3
"""Measure per-block quantization sensitivity of an RF-DETR checkpoint.

Leave-one-in sweep: calibrate the whole model once, disable every quantizer,
then re-enable one block at a time and measure the bbox/segm mAP drop on a
validation subset. The resulting map drives the measured mixed-precision
experiment (Q05): blocks whose drop exceeds the threshold stay FP16, the
rest are quantized (FP8 by default, per the registry).
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts" / "lightweighting"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

from quantize_modelopt import calibration_images, preprocess  # noqa: E402
from kips_lightweighting.rfdetr_compat import load_rfdetr_checkpoint  # noqa: E402


BACKBONE_LAYER = re.compile(r"backbone\.0\.encoder.*\.layer\.(\d+)\.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default="front")
    parser.add_argument(
        "--quant-config",
        default="FP8_DEFAULT_CFG",
        help="modelopt config name (default: FP8_DEFAULT_CFG).",
    )
    parser.add_argument("--calib-count", type=int, default=64)
    parser.add_argument("--eval-count", type=int, default=150)
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=Path("data/training/front/valid"),
    )
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone-groups", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/experiments/sensitivity/front-fp8-sensitivity.json"),
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def block_patterns(core, groups: int) -> dict[str, list[str]]:
    """Partition quantizable modules into named blocks of wildcard patterns."""
    layer_indices = sorted(
        {
            int(match.group(1))
            for name, _ in core.named_modules()
            if (match := BACKBONE_LAYER.search(name))
        }
    )
    blocks: dict[str, list[str]] = {}
    if layer_indices:
        chunk = max(1, (len(layer_indices) + groups - 1) // groups)
        for group_index in range(0, len(layer_indices), chunk):
            members = layer_indices[group_index : group_index + chunk]
            key = f"backbone_layers_{members[0]:02d}-{members[-1]:02d}"
            blocks[key] = [f"*backbone.0.encoder*.layer.{i}.*" for i in members]
    blocks["backbone_projector"] = ["*backbone.0.projector*"]
    decoder_layers = sorted(
        {
            int(match.group(1))
            for name, _ in core.named_modules()
            if (match := re.search(r"transformer\.decoder\.layers\.(\d+)\.", name))
        }
    )
    for index in decoder_layers:
        blocks[f"decoder_layer_{index}"] = [f"*transformer.decoder.layers.{index}.*"]
    blocks["enc_output_and_heads"] = [
        "*transformer.enc_output*",
        "*transformer.enc_out_class_embed*",
        "*transformer.enc_out_bbox_embed*",
    ]
    blocks["final_heads"] = [
        "*class_embed*",
        "*bbox_embed*",
        "*ref_point_head*",
    ]
    blocks["segmentation_head"] = ["*segmentation_head*"]
    return blocks


def set_block(core, patterns: list[str], enabled: bool) -> int:
    from modelopt.torch.quantization.nn import TensorQuantizer

    touched = 0
    for name, module in core.named_modules():
        if not isinstance(module, TensorQuantizer):
            continue
        if any(fnmatch.fnmatch(name, pattern) for pattern in patterns):
            module.enable() if enabled else module.disable()
            touched += 1
    return touched


def evaluate_subset(wrapper, coco, image_ids, dataset_dir: Path) -> dict:
    """bbox + segm mAP50-95 on a fixed validation subset."""
    from pycocotools import mask as mask_utils
    from pycocotools.cocoeval import COCOeval

    bbox_predictions, segm_predictions = [], []
    category_ids = sorted(
        cid for cid in coco.getCatIds() if coco.getAnnIds(catIds=[cid])
    )
    for image_id in image_ids:
        info = coco.loadImgs([image_id])[0]
        image = Image.open(dataset_dir / info["file_name"]).convert("RGB")
        detections = wrapper.predict(image, threshold=0.001)
        masks = detections.mask
        for index in range(len(detections)):
            category_id = int(detections.class_id[index])
            if category_id not in category_ids:
                continue
            common = {
                "image_id": image_id,
                "category_id": category_id,
                "score": float(detections.confidence[index]),
            }
            x1, y1, x2, y2 = (float(v) for v in detections.xyxy[index])
            bbox_predictions.append({**common, "bbox": [x1, y1, x2 - x1, y2 - y1]})
            if masks is not None:
                encoded = mask_utils.encode(
                    np.asfortranarray(masks[index].astype(np.uint8))
                )
                encoded["counts"] = encoded["counts"].decode("ascii")
                segm_predictions.append({**common, "segmentation": encoded})

    def run(predictions, iou_type):
        if not predictions:
            return 0.0
        detected = coco.loadRes(predictions)
        evaluation = COCOeval(coco, detected, iou_type)
        evaluation.params.imgIds = list(image_ids)
        evaluation.params.catIds = category_ids
        evaluation.evaluate()
        evaluation.accumulate()
        evaluation.summarize()
        return float(evaluation.stats[0])

    import contextlib, io

    with contextlib.redirect_stdout(io.StringIO()):
        return {"bbox_ap": run(bbox_predictions, "bbox"), "segm_ap": run(segm_predictions, "segm")}


def main() -> int:
    args = parse_args()
    warnings.filterwarnings("ignore")
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.nn import TensorQuantizer
    from pycocotools.coco import COCO

    with (REPOSITORY_ROOT / "configs" / "baseline.yaml").open(encoding="utf-8") as fh:
        model_config = yaml.safe_load(fh)["models"][args.camera]

    device = args.device if torch.cuda.is_available() else "cpu"
    wrapper = load_rfdetr_checkpoint(
        resolve(Path(model_config["checkpoint"])),
        device=device,
        num_classes=len(model_config["classes"]),
    )
    core = wrapper.model.model.eval().to(device)
    resolution = int(wrapper.model.resolution)

    eval_dir = resolve(args.eval_dir)
    calib = calibration_images(eval_dir, args.calib_count, seed=42)

    def forward_loop(model):
        with torch.no_grad():
            for index, path in enumerate(calib, start=1):
                model(preprocess(path, resolution, device))
                print(f"\rcalibration: {index}/{len(calib)}", end="", flush=True)
        print()

    config = getattr(mtq, args.quant_config)
    mtq.quantize(core, config, forward_loop)
    for _, module in core.named_modules():
        if isinstance(module, TensorQuantizer):
            module.disable()

    coco = COCO(str(eval_dir / "_annotations.coco.json"))
    image_ids = sorted(coco.getImgIds())[: args.eval_count]

    print(f"baseline (all quantizers disabled), {len(image_ids)} images")
    baseline = evaluate_subset(wrapper, coco, image_ids, eval_dir)
    print(f"  bbox {baseline['bbox_ap']:.4f}  segm {baseline['segm_ap']:.4f}")

    blocks = block_patterns(core, args.backbone_groups)
    results = {}
    for block_name, patterns in blocks.items():
        touched = set_block(core, patterns, enabled=True)
        metrics = evaluate_subset(wrapper, coco, image_ids, eval_dir)
        set_block(core, patterns, enabled=False)
        drop_bbox = baseline["bbox_ap"] - metrics["bbox_ap"]
        drop_segm = baseline["segm_ap"] - metrics["segm_ap"]
        results[block_name] = {
            "quantizers": touched,
            "bbox_ap": metrics["bbox_ap"],
            "segm_ap": metrics["segm_ap"],
            "drop_bbox": drop_bbox,
            "drop_segm": drop_segm,
            "drop_max": max(drop_bbox, drop_segm),
        }
        print(
            f"{block_name:<26} q={touched:<4} bbox {metrics['bbox_ap']:.4f} "
            f"({drop_bbox:+.4f})  segm {metrics['segm_ap']:.4f} ({drop_segm:+.4f})"
        )

    sensitive = sorted(
        (name for name, r in results.items() if r["drop_max"] > args.threshold),
        key=lambda name: -results[name]["drop_max"],
    )
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "camera": args.camera,
        "quant_config": args.quant_config,
        "checkpoint": model_config["checkpoint"],
        "calibration_images": len(calib),
        "eval_images": len(image_ids),
        "eval_dir": str(eval_dir),
        "threshold": args.threshold,
        "baseline": baseline,
        "blocks": results,
        "block_patterns": blocks,
        "sensitive_blocks": sensitive,
        "hardware": torch.cuda.get_device_name(0) if device != "cpu" else "cpu",
    }
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"\nsensitive blocks (drop > {args.threshold}): {sensitive or 'none'}")
    print(f"report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
