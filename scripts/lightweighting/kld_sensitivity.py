#!/usr/bin/env python3
"""KL-divergence quantization sensitivity map for RF-DETR (Unsloth-style).

Instead of per-block mAP drops (noisy at small sample sizes), this measures
the KL divergence between the FP32 model's output distributions and the
per-block-quantized model's, following the idea behind Unsloth Dynamic 2.0
GGUFs: distributional damage is a lower-variance, more faithful sensitivity
signal than task accuracy.

Per block (leave-one-in): Bernoulli KLD over sigmoid class logits, Bernoulli
KLD over subsampled sigmoid mask logits, and L1 drift on boxes. Blocks are
ranked by kld_total = kld_cls + kld_mask; the top fraction is marked
sensitive (kept FP16 by quantize_modelopt.py --mode fp8-mixed).

The output JSON is schema-compatible with sensitivity_sweep.py, so it plugs
straight into `quantize_modelopt.py --sensitivity-report`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts" / "lightweighting"))

import torch  # noqa: E402
import yaml  # noqa: E402

from quantize_modelopt import calibration_images, preprocess  # noqa: E402
from sensitivity_sweep import block_patterns, set_block  # noqa: E402
from kips_lightweighting.rfdetr_compat import load_rfdetr_checkpoint  # noqa: E402


MASK_STRIDE = 4
EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default="front")
    parser.add_argument("--quant-config", default="FP8_DEFAULT_CFG")
    parser.add_argument("--calib-count", type=int, default=64)
    parser.add_argument("--kld-images", type=int, default=48)
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=Path("data/training/front/valid"),
    )
    parser.add_argument(
        "--backbone-groups",
        type=int,
        default=12,
        help="12 = per-layer backbone granularity (KLD is cheap enough).",
    )
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.25,
        help="Fraction of blocks (by kld_total) marked sensitive/kept FP16.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/experiments/sensitivity/front-fp8-kld.json"),
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def bernoulli_kld(p_ref: torch.Tensor, p_test: torch.Tensor) -> float:
    p = p_ref.clamp(EPS, 1 - EPS)
    q = p_test.clamp(EPS, 1 - EPS)
    kld = p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()
    return float(kld.mean())


TOP_QUERIES = 50


@torch.no_grad()
def collect_outputs(core, tensors) -> list[dict[str, torch.Tensor]]:
    outputs = []
    for tensor in tensors:
        raw = core(tensor)
        outputs.append(
            {
                "cls": raw["pred_logits"][0].sigmoid().float().cpu(),
                "boxes": raw["pred_boxes"][0].float().cpu(),
                "masks": raw["pred_masks"][0][
                    ..., ::MASK_STRIDE, ::MASK_STRIDE
                ].sigmoid().float().cpu(),
            }
        )
    return outputs


def divergence(reference, candidate) -> dict[str, float]:
    """Permutation-invariant divergence.

    DETR queries are exchangeable: early-layer perturbations permute which
    query detects which object without harming the detection set, so a
    per-index comparison wildly overstates damage. We therefore take the
    reference's TOP_QUERIES most confident queries and Hungarian-match them
    to candidate queries (box L1 + class L1 cost) before measuring KLD.
    """
    from scipy.optimize import linear_sum_assignment

    kld_cls = kld_mask = box_l1 = 0.0
    for ref, cand in zip(reference, candidate):
        top = ref["cls"].max(dim=-1).values.topk(TOP_QUERIES).indices
        ref_cls, ref_boxes = ref["cls"][top], ref["boxes"][top]
        ref_masks = ref["masks"][top]
        cost = torch.cdist(ref_boxes, cand["boxes"], p=1) + torch.cdist(
            ref_cls, cand["cls"], p=1
        )
        row, col = linear_sum_assignment(cost.numpy())
        col = torch.as_tensor(col)
        kld_cls += bernoulli_kld(ref_cls, cand["cls"][col])
        kld_mask += bernoulli_kld(ref_masks, cand["masks"][col])
        box_l1 += float((ref_boxes - cand["boxes"][col]).abs().mean())
    count = len(reference)
    return {
        "kld_cls": kld_cls / count,
        "kld_mask": kld_mask / count,
        "kld_total": (kld_cls + kld_mask) / count,
        "box_l1": box_l1 / count,
    }


def main() -> int:
    args = parse_args()
    warnings.filterwarnings("ignore")
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.nn import TensorQuantizer

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
    kld_paths = calibration_images(eval_dir, args.kld_images, seed=7)
    tensors = [preprocess(path, resolution, device) for path in kld_paths]

    print(f"caching FP32 reference outputs on {len(tensors)} images")
    reference = collect_outputs(core, tensors)

    def forward_loop(model):
        with torch.no_grad():
            for index, path in enumerate(calib, start=1):
                model(preprocess(path, resolution, device))
                print(f"\rcalibration: {index}/{len(calib)}", end="", flush=True)
        print()

    mtq.quantize(core, getattr(mtq, args.quant_config), forward_loop)
    for _, module in core.named_modules():
        if isinstance(module, TensorQuantizer):
            module.disable()

    sanity = divergence(reference, collect_outputs(core, tensors))
    print(f"sanity (all disabled) kld_total={sanity['kld_total']:.2e}")

    blocks = block_patterns(core, args.backbone_groups)
    results = {}
    for block_name, patterns in blocks.items():
        touched = set_block(core, patterns, enabled=True)
        metrics = divergence(reference, collect_outputs(core, tensors))
        set_block(core, patterns, enabled=False)
        results[block_name] = {"quantizers": touched, **metrics}
        print(
            f"{block_name:<26} q={touched:<4} kld_cls {metrics['kld_cls']:.3e} "
            f"kld_mask {metrics['kld_mask']:.3e} total {metrics['kld_total']:.3e}"
        )

    ranked = sorted(results, key=lambda name: -results[name]["kld_total"])
    keep = max(1, math.ceil(args.top_fraction * len(ranked)))
    sensitive = ranked[:keep]

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": (
            "hungarian-matched kl-divergence leave-one-in "
            "(unsloth dynamic 2.0 style, permutation-invariant for DETR)"
        ),
        "top_queries": TOP_QUERIES,
        "camera": args.camera,
        "quant_config": args.quant_config,
        "checkpoint": model_config["checkpoint"],
        "calibration_images": len(calib),
        "kld_images": len(kld_paths),
        "eval_dir": str(eval_dir),
        "mask_stride": MASK_STRIDE,
        "top_fraction": args.top_fraction,
        "sanity_kld_total": sanity["kld_total"],
        "blocks": results,
        "block_patterns": blocks,
        "ranked_blocks": ranked,
        "sensitive_blocks": sensitive,
        "hardware": torch.cuda.get_device_name(0) if device != "cpu" else "cpu",
    }
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"\nsensitive (top {keep}/{len(ranked)} by kld_total): {sensitive}")
    print(f"report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
