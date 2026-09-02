#!/usr/bin/env python3
"""Quantize an RF-DETR checkpoint with nvidia-modelopt and export Q/DQ ONNX.

Implements the registered Q-series experiments:

  Q01  --mode smoothquant  activation-aware uniform INT8 (LLM imatrix analogue)
  Q02  --mode mixed        INT8 everywhere except accuracy-sensitive layers,
                           which keep FP16 (quantizers disabled by pattern)
  Q03  --mode int4-ffn     blockwise weight-only INT4 on FFN linears only

The exported ONNX carries explicit Q/DQ (fake-quant) nodes; build the engine
with `build_tensorrt_fp16.py --int8-qdq` so TensorRT honours them and runs
the remaining layers in FP16.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import random
import shutil
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

from kips_lightweighting.rfdetr_compat import load_rfdetr_checkpoint  # noqa: E402


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MODE_CONFIGS = {
    "smoothquant": "INT8_SMOOTHQUANT_CFG",
    "mixed": "INT8_DEFAULT_CFG",
    "int4-ffn": "INT4_BLOCKWISE_WEIGHT_ONLY_CFG",
    "fp8": "FP8_DEFAULT_CFG",
    "fp8-mixed": "FP8_DEFAULT_CFG",
}

# FP8 QuantizeLinear/DequantizeLinear needs the float8 ONNX types (opset 19+).
FP8_MIN_OPSET = 19

# Q02: layers that stay FP16 (quantizers disabled). Heads, deformable
# sampling offsets, and encoder output projections are the usual
# accuracy-sensitive suspects in DETR-family models.
DEFAULT_KEEP_FP16 = (
    "*class_embed*",
    "*bbox_embed*",
    "*segmentation_head*",
    "*sampling_offsets*",
    "*enc_out*",
    "*ref_point*",
)

# Q03: only these get INT4 weight-only quantization.
DEFAULT_INT4_TARGETS = (
    "*transformer.decoder.layers.*.linear1*",
    "*transformer.decoder.layers.*.linear2*",
    "*mlp.fc1*",
    "*mlp.fc2*",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, help="e.g. Q01")
    parser.add_argument(
        "--camera",
        default="front",
        help="Model key under 'models' in configs/baseline.yaml.",
    )
    parser.add_argument(
        "--mode",
        choices=tuple(MODE_CONFIGS),
        required=True,
    )
    parser.add_argument(
        "--calib-dir",
        type=Path,
        default=Path("data/training/front_session_split_v1/train"),
        help="Directory with calibration images.",
    )
    parser.add_argument("--calib-count", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--keep-fp16-pattern",
        action="append",
        default=None,
        help="Override Q02 sensitive-layer patterns (repeatable).",
    )
    parser.add_argument(
        "--quantize-pattern",
        action="append",
        default=None,
        help="Override Q03 target patterns (repeatable).",
    )
    parser.add_argument(
        "--sensitivity-report",
        type=Path,
        help="fp8-mixed: sensitivity_sweep.py JSON whose sensitive blocks "
        "stay FP16 (their quantizers are disabled).",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def calibration_images(directory: Path, count: int, seed: int) -> list[Path]:
    files = sorted(
        path
        for path in directory.iterdir()
        if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
    )
    if not files:
        raise FileNotFoundError(f"No calibration images in {directory}")
    random.Random(seed).shuffle(files)
    return files[:count]


def preprocess(path: Path, resolution: int, device: str) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize(
        (resolution, resolution), Image.BILINEAR
    )
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor.to(device)


def set_quantizers(model, patterns, enabled: bool, invert: bool = False) -> int:
    """Enable/disable TensorQuantizer modules whose names match patterns."""
    from modelopt.torch.quantization.nn import TensorQuantizer

    touched = 0
    for name, module in model.named_modules():
        if not isinstance(module, TensorQuantizer):
            continue
        matches = any(fnmatch.fnmatch(name, pattern) for pattern in patterns)
        if matches != invert:
            if enabled:
                module.enable()
            else:
                module.disable()
            touched += 1
    return touched


def main() -> int:
    args = parse_args()
    warnings.filterwarnings("ignore")
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.nn import TensorQuantizer

    with (REPOSITORY_ROOT / "configs" / "baseline.yaml").open(
        encoding="utf-8"
    ) as stream:
        model_config = yaml.safe_load(stream)["models"][args.camera]

    output_dir = resolve(
        args.output_dir
        or Path(f"artifacts/experiments/{args.experiment}/{args.camera}")
    )
    onnx_path = output_dir / "model.onnx"
    report_path = output_dir / "quantization.json"
    if onnx_path.exists() and not args.force:
        print(f"exists: {onnx_path} (use --force to redo)")
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    checkpoint = resolve(Path(model_config["checkpoint"]))
    wrapper = load_rfdetr_checkpoint(
        checkpoint,
        device=device,
        num_classes=len(model_config["classes"]),
    )
    core = wrapper.model.model.eval().to(device)
    resolution = int(wrapper.model.resolution)

    calib_dir = resolve(args.calib_dir)
    images = calibration_images(calib_dir, args.calib_count, args.seed)
    print(f"calibrating on {len(images)} images from {calib_dir}")

    def forward_loop(model) -> None:
        with torch.no_grad():
            for index, path in enumerate(images, start=1):
                model(preprocess(path, resolution, device))
                print(f"\rcalibration: {index}/{len(images)}", end="", flush=True)
        print()

    config = getattr(mtq, MODE_CONFIGS[args.mode])
    mtq.quantize(core, config, forward_loop)

    keep_fp16 = tuple(args.keep_fp16_pattern or DEFAULT_KEEP_FP16)
    int4_targets = tuple(args.quantize_pattern or DEFAULT_INT4_TARGETS)
    sensitive_blocks: list[str] = []
    disabled = 0
    if args.mode == "mixed":
        disabled = set_quantizers(core, keep_fp16, enabled=False)
        print(f"disabled {disabled} quantizers on sensitive layers")
    elif args.mode == "int4-ffn":
        disabled = set_quantizers(core, int4_targets, enabled=False, invert=True)
        print(f"disabled {disabled} quantizers outside FFN targets")
    if args.mode.startswith("fp8"):
        # FP8 Q/DQ on convolutions is not exportable through torch.onnx
        # (unknown-shape kernel after TRT_FP8DequantizeLinear) and TensorRT
        # FP8 acceleration targets GEMMs anyway: keep convs in FP16.
        conv_patterns = tuple(
            f"{name}.*"
            for name, module in core.named_modules()
            if isinstance(module, torch.nn.Conv2d)
        )
        disabled_conv = set_quantizers(core, conv_patterns, enabled=False)
        print(f"disabled {disabled_conv} conv quantizers (FP8 keeps convs FP16)")

    if args.mode == "fp8-mixed":
        if not args.sensitivity_report:
            raise ValueError("fp8-mixed requires --sensitivity-report")
        sensitivity = json.loads(
            resolve(args.sensitivity_report).read_text(encoding="utf-8")
        )
        sensitive_blocks = list(sensitivity["sensitive_blocks"])
        keep_fp16 = tuple(
            pattern
            for block in sensitive_blocks
            for pattern in sensitivity["block_patterns"][block]
        )
        disabled = set_quantizers(core, keep_fp16, enabled=False)
        print(
            f"disabled {disabled} quantizers on measured-sensitive blocks: "
            f"{sensitive_blocks or 'none'}"
        )

    active = sum(
        1
        for _, module in core.named_modules()
        if isinstance(module, TensorQuantizer) and module.is_enabled
    )
    total = sum(
        1
        for _, module in core.named_modules()
        if isinstance(module, TensorQuantizer)
    )
    print(f"active quantizers: {active}/{total}")

    opset = args.opset
    if args.mode.startswith("fp8") and opset < FP8_MIN_OPSET:
        opset = FP8_MIN_OPSET
        print(f"fp8 mode: raising ONNX opset to {opset}")

    exported = Path(
        wrapper.export(
            output_dir=str(output_dir),
            format="onnx",
            opset_version=opset,
            batch_size=1,
            dynamic_batch=False,
            verbose=False,
            notes={
                "experiment": args.experiment,
                "mode": args.mode,
                "purpose": "modelopt Q/DQ ONNX for TensorRT",
            },
        )
    )
    if exported.resolve() != onnx_path.resolve():
        shutil.move(str(exported), onnx_path)

    import onnx

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    qdq_nodes = sum(
        1
        for node in onnx_model.graph.node
        if node.op_type
        in (
            "QuantizeLinear",
            "DequantizeLinear",
            "TRT_FP8QuantizeLinear",
            "TRT_FP8DequantizeLinear",
        )
    )
    print(f"ONNX Q/DQ nodes: {qdq_nodes}")

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": args.experiment,
        "camera": args.camera,
        "mode": args.mode,
        "modelopt_config": MODE_CONFIGS[args.mode],
        "checkpoint": str(checkpoint),
        "calibration_dir": str(calib_dir),
        "calibration_images": len(images),
        "seed": args.seed,
        "resolution": resolution,
        "quantizers_total": total,
        "quantizers_active": active,
        "quantizers_disabled": disabled,
        "keep_fp16_patterns": list(keep_fp16) if args.mode == "mixed" else [],
        "quantize_patterns": list(int4_targets) if args.mode == "int4-ffn" else [],
        "onnx_path": (
            str(onnx_path.relative_to(REPOSITORY_ROOT))
            if onnx_path.is_relative_to(REPOSITORY_ROOT)
            else str(onnx_path)
        ),
        "onnx_size_bytes": onnx_path.stat().st_size,
        "onnx_qdq_nodes": qdq_nodes,
        "opset": args.opset,
        "device": device,
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"ONNX: {onnx_path}")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
