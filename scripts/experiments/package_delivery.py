#!/usr/bin/env python3
"""Package selected experiment artifacts for evaluator handoff."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="+", required=True)
    parser.add_argument("--camera", choices=("front", "rear"), required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "delivery",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    registry = ExperimentRegistry.load()
    output_root = args.output_dir / args.camera
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "camera": args.camera,
        "experiments": [],
    }
    for experiment_id in args.experiments:
        experiment = registry.get(experiment_id)
        if args.camera not in experiment["cameras"]:
            raise ValueError(f"{experiment_id} is not registered for {args.camera}.")
        source = artifact_paths(experiment_id, args.camera)
        existing = source.existing()
        if "engine" not in existing:
            raise FileNotFoundError(f"No engine to deliver for {experiment_id}.")
        destination = output_root / experiment_id
        if destination.exists() and not args.force:
            raise FileExistsError(f"Delivery already exists: {destination}")
        destination.mkdir(parents=True, exist_ok=True)
        files = []
        for label, path in existing.items():
            target = destination / path.name
            shutil.copy2(path, target)
            files.append(
                {
                    "type": label,
                    "path": str(target.relative_to(args.output_dir)),
                    "sha256": sha256(target),
                }
            )
        manifest["experiments"].append(
            {
                "experiment_id": experiment_id,
                "name": experiment["name"],
                "files": files,
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / f"manifest-{args.camera}.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
