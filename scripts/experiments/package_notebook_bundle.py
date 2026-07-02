#!/usr/bin/env python3
"""Package portable ONNX inputs needed to rebuild TensorRT engines."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from kips_lightweighting.artifacts import artifact_paths  # noqa: E402
from kips_lightweighting.metadata import sha256  # noqa: E402
from kips_lightweighting.registry import ExperimentRegistry  # noqa: E402

from build_engine_suite import SUITES  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=tuple(SUITES), default="final8")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "delivery" / "notebook-front",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    registry = ExperimentRegistry.load()
    output = (
        args.output_dir
        if args.output_dir.is_absolute()
        else REPOSITORY_ROOT / args.output_dir
    )
    if output.exists() and not args.force:
        raise FileExistsError(f"Bundle already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    source_ids = set()
    for experiment_id in SUITES[args.suite]:
        experiment = registry.get(experiment_id)
        source_id = experiment.get("artifact_source", experiment_id)
        if experiment["family"] == "precision":
            source_id = "B01"
        source_ids.add(source_id)

    files = []
    for source_id in sorted(source_ids):
        source = artifact_paths(source_id, "front").onnx
        if not source.is_file():
            raise FileNotFoundError(source)
        target = (
            output
            / "artifacts"
            / "experiments"
            / source_id
            / "front"
            / "model.onnx"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        files.append(
            {
                "experiment_id": source_id,
                "path": str(target.relative_to(output)),
                "size_bytes": target.stat().st_size,
                "sha256": sha256(target),
            }
        )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite": args.suite,
        "camera": "front",
        "experiments": list(SUITES[args.suite]),
        "onnx_sources": files,
        "requires_calibration_data": any(
            registry.get(experiment_id)["precision"] == "int8"
            for experiment_id in SUITES[args.suite]
        ),
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"bundle: {output}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
