#!/usr/bin/env python3
"""Extreme weight-only quantization study for RF-DETR (GGUF-style W8..W2).

Measures the bit-width vs accuracy curve using fake-quantized weights with
FP16 activations (the LLM weight-only recipe: RTN group-wise, plus AWQ for
4-bit). Mixed variants protect the blocks that the task-metric sensitivity
sweep flagged (final_heads, decoder_layer_4) — the signal we validated in
0705.md section 18 — mirroring llama.cpp K-quant mixes.

This is an accuracy study: TensorRT has no W2/W3 kernel path for this graph
(see Q03), so results quantify feasibility for a future custom-kernel or
QAT deployment, not a buildable engine.
"""

from __future__ import annotations

import argparse
import copy
import fnmatch
import json
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
from sensitivity_sweep import evaluate_subset  # noqa: E402
from kips_lightweighting.rfdetr_compat import load_rfdetr_checkpoint  # noqa: E402


# Task-metric-sensitive blocks (0705.md section 17/18.3): kept high precision
# in the mixed variants.
SENSITIVE_PATTERNS = (
    "*class_embed*",
    "*bbox_embed*",
    "*ref_point_head*",
    "*transformer.decoder.layers.4.*",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default="front")
    parser.add_argument("--calib-count", type=int, default=128)
    parser.add_argument("--screen-count", type=int, default=150)
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=Path("data/training/front/valid"),
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        help="Optional second dataset for finalist configs (e.g. datum test).",
    )
    parser.add_argument(
        "--test-configs",
        nargs="*",
        default=["w4_awq", "w4_mixed", "w2_mixed"],
        help="Configs re-evaluated on --test-dir.",
    )
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/experiments/sensitivity/front-weight-bitwidth.json"
        ),
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def weight_only_cfg(mtq, num_bits: int, group: int) -> dict:
    """Clone the blockwise weight-only config at an arbitrary bit width."""
    config = copy.deepcopy(mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG)
    rules = config["quant_cfg"]
    entries = rules.items() if isinstance(rules, dict) else (
        (rule.get("quantizer_name", ""), rule.get("cfg")) for rule in rules
    )
    for pattern, entry in entries:
        if "weight_quantizer" in pattern and isinstance(entry, dict):
            entry["num_bits"] = num_bits
            if "block_sizes" in entry:
                entry["block_sizes"] = {-1: group}
    return config


def study_configs(mtq, group: int) -> dict[str, dict]:
    return {
        "w8_rtn": {"config": weight_only_cfg(mtq, 8, group)},
        "w4_rtn": {"config": weight_only_cfg(mtq, 4, group)},
        "w4_awq": {"config": copy.deepcopy(mtq.INT4_AWQ_CFG)},
        "w3_rtn": {"config": weight_only_cfg(mtq, 3, group)},
        "w2_rtn": {"config": weight_only_cfg(mtq, 2, group)},
        "w4_mixed": {
            "config": copy.deepcopy(mtq.INT4_AWQ_CFG),
            "protect": SENSITIVE_PATTERNS,
        },
        "w2_mixed": {
            "config": weight_only_cfg(mtq, 2, group),
            "protect": SENSITIVE_PATTERNS,
        },
    }


def protect_blocks(core, patterns) -> int:
    from modelopt.torch.quantization.nn import TensorQuantizer

    disabled = 0
    for name, module in core.named_modules():
        if isinstance(module, TensorQuantizer) and any(
            fnmatch.fnmatch(name, pattern) for pattern in patterns
        ):
            module.disable()
            disabled += 1
    return disabled


def quantized_weight_stats(core, protected_patterns=()) -> dict:
    """Theoretical storage if quantized weights were packed."""
    from modelopt.torch.quantization.nn import TensorQuantizer

    quantized = 0
    for name, module in core.named_modules():
        if isinstance(module, TensorQuantizer) and name.endswith(
            "weight_quantizer"
        ):
            if module.is_enabled:
                quantized += 1
    total_params = sum(p.numel() for p in core.parameters())
    return {"enabled_weight_quantizers": quantized, "total_params": total_params}


def theoretical_size_mb(core, bits: int, protected_patterns=()) -> float:
    from modelopt.torch.quantization.nn import TensorQuantizer

    quantized_params = 0
    parents = {}
    for name, module in core.named_modules():
        parents[name] = module
    total = 0
    for name, module in core.named_modules():
        if not hasattr(module, "weight") or module.weight is None:
            continue
        if not isinstance(module.weight, torch.Tensor):
            continue
        total += module.weight.numel()
        quantizer_name = f"{name}.weight_quantizer"
        quantizer = parents.get(quantizer_name)
        if (
            quantizer is not None
            and isinstance(quantizer, TensorQuantizer)
            and quantizer.is_enabled
        ):
            quantized_params += module.weight.numel()
    other = sum(p.numel() for p in core.parameters()) - total
    size_bits = quantized_params * bits + (total - quantized_params + other) * 16
    return size_bits / 8 / 1e6


def main() -> int:
    args = parse_args()
    warnings.filterwarnings("ignore")
    import modelopt.torch.quantization as mtq
    from pycocotools.coco import COCO

    with (REPOSITORY_ROOT / "configs" / "baseline.yaml").open(encoding="utf-8") as fh:
        model_config = yaml.safe_load(fh)["models"][args.camera]
    device = args.device if torch.cuda.is_available() else "cpu"
    checkpoint = resolve(Path(model_config["checkpoint"]))
    num_classes = len(model_config["classes"])

    eval_dir = resolve(args.eval_dir)
    coco = COCO(str(eval_dir / "_annotations.coco.json"))
    screen_ids = sorted(coco.getImgIds())[: args.screen_count]
    calib = calibration_images(eval_dir, args.calib_count, seed=42)

    test_dir = resolve(args.test_dir) if args.test_dir else None
    test_coco = COCO(str(test_dir / "_annotations.coco.json")) if test_dir else None

    def fresh_model():
        wrapper = load_rfdetr_checkpoint(
            checkpoint, device=device, num_classes=num_classes
        )
        wrapper.model.model.eval().to(device)
        return wrapper

    print("FP16 reference on screen subset")
    reference = fresh_model()
    baseline = evaluate_subset(reference, coco, screen_ids, eval_dir)
    print(f"  bbox {baseline['bbox_ap']:.4f}  segm {baseline['segm_ap']:.4f}")
    del reference
    torch.cuda.empty_cache()

    results = {}
    bits_of = {"w8_rtn": 8, "w4_rtn": 4, "w4_awq": 4, "w3_rtn": 3,
               "w2_rtn": 2, "w4_mixed": 4, "w2_mixed": 2}
    for name, spec in study_configs(mtq, args.group_size).items():
        wrapper = fresh_model()
        core = wrapper.model.model

        def forward_loop(model):
            with torch.no_grad():
                for index, path in enumerate(calib, start=1):
                    model(preprocess(path, int(wrapper.model.resolution), device))
                    print(f"\r{name} calibration: {index}/{len(calib)}",
                          end="", flush=True)
            print()

        try:
            mtq.quantize(core, spec["config"], forward_loop)
        except Exception as error:
            print(f"{name}: quantize FAILED ({type(error).__name__}: {error})")
            results[name] = {"status": "failed", "error": str(error)}
            del wrapper
            torch.cuda.empty_cache()
            continue
        protected = 0
        if spec.get("protect"):
            protected = protect_blocks(core, spec["protect"])
        metrics = evaluate_subset(wrapper, coco, screen_ids, eval_dir)
        entry = {
            "status": "ok",
            "bits": bits_of[name],
            "protected_quantizers": protected,
            "screen_bbox_ap": metrics["bbox_ap"],
            "screen_segm_ap": metrics["segm_ap"],
            "screen_drop_bbox": baseline["bbox_ap"] - metrics["bbox_ap"],
            "screen_drop_segm": baseline["segm_ap"] - metrics["segm_ap"],
            "theoretical_weights_mb": theoretical_size_mb(core, bits_of[name]),
        }
        print(
            f"{name:<10} bbox {metrics['bbox_ap']:.4f} "
            f"({entry['screen_drop_bbox']:+.4f})  segm {metrics['segm_ap']:.4f} "
            f"({entry['screen_drop_segm']:+.4f})  ~{entry['theoretical_weights_mb']:.0f}MB"
        )
        if test_coco is not None and name in args.test_configs:
            test_ids = sorted(test_coco.getImgIds())
            test_metrics = evaluate_subset(wrapper, test_coco, test_ids, test_dir)
            entry["test_bbox_ap"] = test_metrics["bbox_ap"]
            entry["test_segm_ap"] = test_metrics["segm_ap"]
            print(
                f"{name:<10} [test {len(test_ids)}] bbox "
                f"{test_metrics['bbox_ap']:.4f}  segm {test_metrics['segm_ap']:.4f}"
            )
        results[name] = entry
        del wrapper
        torch.cuda.empty_cache()

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "camera": args.camera,
        "checkpoint": model_config["checkpoint"],
        "group_size": args.group_size,
        "calibration_images": len(calib),
        "screen_images": len(screen_ids),
        "screen_baseline": baseline,
        "sensitive_patterns": list(SENSITIVE_PATTERNS),
        "configs": results,
        "note": (
            "fake-quant weight-only accuracy study; no TensorRT kernel path "
            "for W2/W3 (see Q03), deployment would need QAT or custom kernels"
        ),
        "hardware": torch.cuda.get_device_name(0) if device != "cpu" else "cpu",
    }
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
