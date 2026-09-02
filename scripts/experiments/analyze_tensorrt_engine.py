#!/usr/bin/env python3
"""Record static TensorRT engine structure and precision evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def portable(path: Path) -> str:
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


def precision_evidence(layer: Any) -> set[str]:
    """Return non-exclusive precision tokens exposed by an inspector layer."""
    text = json.dumps(layer, ensure_ascii=False).lower()
    evidence = set()
    patterns = {
        "int8": r"(?<![a-z0-9])(?:int8|kint8)(?![a-z0-9])",
        "fp16": r"(?<![a-z0-9])(?:fp16|half|khalf)(?![a-z0-9])",
        "fp32": r"(?<![a-z0-9])(?:fp32|float|kfloat)(?![a-z0-9])",
        "fp8": r"(?<![a-z0-9])(?:fp8|kfp8)(?![a-z0-9])",
    }
    for name, pattern in patterns.items():
        if re.search(pattern, text):
            evidence.add(name)
    return evidence


def parse_inspector_json(raw: str) -> tuple[list[Any], str | None]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        return [], f"{type(error).__name__}: {error}"
    if isinstance(parsed, list):
        return parsed, None
    if isinstance(parsed, dict):
        for key in ("Layers", "layers"):
            if isinstance(parsed.get(key), list):
                return parsed[key], None
        return [parsed], None
    return [], f"Unexpected inspector JSON root: {type(parsed).__name__}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--camera", choices=("front", "rear"), required=True)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import tensorrt as trt
    import torch

    engine_path = resolve(
        args.engine
        or Path(f"artifacts/experiments/{args.experiment}/{args.camera}/model.engine")
    )
    output = resolve(
        args.output
        or Path(
            f"artifacts/experiments/{args.experiment}/{args.camera}/"
            "engine-static-analysis.json"
        )
    )
    if not engine_path.is_file():
        raise FileNotFoundError(engine_path)

    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"Could not deserialize TensorRT engine: {engine_path}")
    inspector = engine.create_engine_inspector()
    raw = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    layers, parse_error = parse_inspector_json(raw)

    layer_types: Counter[str] = Counter()
    precision_counts: Counter[str] = Counter()
    layer_records = []
    for index, layer in enumerate(layers):
        if isinstance(layer, dict):
            layer_type = str(
                layer.get("LayerType", layer.get("type", layer.get("Type", "unknown")))
            )
            name = str(layer.get("Name", layer.get("name", f"layer-{index}")))
        else:
            layer_type = "unknown"
            name = f"layer-{index}"
        evidence = sorted(precision_evidence(layer))
        layer_types[layer_type] += 1
        precision_counts.update(evidence or ["unreported"])
        layer_records.append(
            {
                "index": index,
                "name": name,
                "type": layer_type,
                "precision_evidence": evidence,
                "inspector": layer,
            }
        )

    io_tensors = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        io_tensors.append(
            {
                "name": name,
                "mode": str(engine.get_tensor_mode(name)),
                "dtype": str(engine.get_tensor_dtype(name)),
                "shape": list(engine.get_tensor_shape(name)),
                "format": str(engine.get_tensor_format_desc(name)),
            }
        )

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": args.experiment,
        "camera": args.camera,
        "engine": portable(engine_path),
        "engine_size_bytes": engine_path.stat().st_size,
        "engine_sha256": sha256(engine_path),
        "tensorrt_version": trt.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "profiling_verbosity": str(engine.profiling_verbosity),
        "num_layers": engine.num_layers,
        "inspector_layer_records": len(layer_records),
        "inspector_parse_error": parse_error,
        "precision_evidence_scope": (
            "Non-exclusive token evidence from TensorRT detailed layer inspector; "
            "build logs remain authoritative for sparse tactic selection."
        ),
        "layers_with_precision_evidence": dict(sorted(precision_counts.items())),
        "layer_types": dict(sorted(layer_types.items())),
        "io_tensors": io_tensors,
        "layers": layer_records,
        "raw_inspector_json": raw if parse_error else None,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"engine static analysis: {output}")
    print(f"layers: {engine.num_layers}, inspector records: {len(layer_records)}")
    print(f"precision evidence: {dict(sorted(precision_counts.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
