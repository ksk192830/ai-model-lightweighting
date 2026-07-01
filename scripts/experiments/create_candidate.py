#!/usr/bin/env python3
"""Create a registered lightweighting candidate checkpoint."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from kips_lightweighting.artifacts import artifact_paths  # noqa: E402
from kips_lightweighting.metadata import (  # noqa: E402
    runtime_metadata,
    sha256,
    write_json,
)
from kips_lightweighting.pruning import (  # noqa: E402
    prune_global_magnitude,
    prune_two_of_four,
)
from kips_lightweighting.registry import ExperimentRegistry  # noqa: E402
from kips_lightweighting.static_analysis import checkpoint_state  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_id")
    parser.add_argument("--camera", choices=("front", "rear"), required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def model_config(camera: str) -> dict:
    with (REPOSITORY_ROOT / "configs" / "baseline.yaml").open(
        encoding="utf-8"
    ) as stream:
        return yaml.safe_load(stream)["models"][camera]


def sync_lightning_state(checkpoint: dict) -> None:
    state = checkpoint_state(checkpoint)
    lightning_state = checkpoint.get("state_dict")
    if not isinstance(lightning_state, dict):
        return
    for name, value in state.items():
        prefixed = f"model.{name}"
        if prefixed in lightning_state:
            lightning_state[prefixed] = value


def main() -> int:
    args = parse_args()
    registry = ExperimentRegistry.load()
    experiment = registry.get(args.experiment_id)
    if args.camera not in experiment["cameras"]:
        raise ValueError(
            f"{args.experiment_id} is not registered for camera {args.camera}."
        )
    if experiment["family"] not in {"unstructured", "semi-structured"}:
        raise NotImplementedError(
            "Candidate creation currently supports unstructured and "
            "semi-structured experiments."
        )
    if (
        experiment["family"] == "semi-structured"
        and experiment.get("artifact_source")
    ):
        raise ValueError(
            f"{args.experiment_id} shares artifacts from "
            f"{experiment['artifact_source']} and does not create a checkpoint."
        )

    config = model_config(args.camera)
    source = REPOSITORY_ROOT / config["checkpoint"]
    paths = artifact_paths(args.experiment_id, args.camera)
    if paths.checkpoint.exists() and not args.force:
        print(f"exists: {paths.checkpoint}")
        return 0

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if experiment["family"] == "unstructured":
        ratio = float(experiment["pruning"]["sparsity"])
        pruned = prune_global_magnitude(checkpoint_state(checkpoint), ratio)
        pruning_result = {
            "method": experiment["method"],
            "ratio": ratio,
            "pruned_parameters": pruned,
        }
    else:
        eligibility_path = paths.directory / "2to4-eligibility.json"
        if not eligibility_path.is_file():
            raise FileNotFoundError(
                "Run analyze_candidate.py with --inspect-2to4 first: "
                f"{eligibility_path}"
            )
        import json

        eligibility = json.loads(eligibility_path.read_text(encoding="utf-8"))
        pruning_result = prune_two_of_four(
            checkpoint_state(checkpoint),
            eligibility["layers"],
        )
        if not pruning_result["valid"]:
            raise RuntimeError("Generated checkpoint does not satisfy 2:4.")
        write_json(paths.sparsity, pruning_result)
    sync_lightning_state(checkpoint)
    checkpoint["optimizer_states"] = []
    checkpoint["lr_schedulers"] = []
    checkpoint["pruning"] = {
        "experiment_id": args.experiment_id,
        **pruning_result,
        "source_checkpoint": str(source.relative_to(REPOSITORY_ROOT)),
    }
    paths.directory.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, paths.checkpoint)

    metadata = {
        **runtime_metadata(),
        "experiment_id": args.experiment_id,
        "camera": args.camera,
        "source_checkpoint": str(source.relative_to(REPOSITORY_ROOT)),
        "source_checkpoint_sha256": sha256(source),
        "method": experiment["method"],
        "pruning": experiment["pruning"],
        "precision": experiment["precision"],
        "fine_tuning": experiment.get("fine_tuning", {"required": False}),
        "input_shape": registry.defaults["export"]["input_shape"],
        "batch_size": registry.defaults["export"]["batch_size"],
        "classes": config["classes"],
        "onnx_opset": registry.defaults["export"]["onnx_opset"],
        "artifacts": {
            "checkpoint": str(paths.checkpoint),
            **(
                {"sparsity": str(paths.sparsity)}
                if paths.sparsity.is_file()
                else {}
            ),
        },
        "status": "checkpoint-created",
    }
    write_json(paths.metadata, metadata)
    print(f"checkpoint: {paths.checkpoint}")
    print(f"metadata: {paths.metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
