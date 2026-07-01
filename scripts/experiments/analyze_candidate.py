#!/usr/bin/env python3
"""Run static analysis for a registered experiment artifact."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from kips_lightweighting.artifacts import artifact_paths  # noqa: E402
from kips_lightweighting.metadata import (  # noqa: E402
    read_json,
    refresh_experiment_documents,
    runtime_metadata,
    write_json,
)
from kips_lightweighting.registry import ExperimentRegistry  # noqa: E402
from kips_lightweighting.static_analysis import (  # noqa: E402
    checkpoint_analysis,
    onnx_analysis,
    onnx_two_to_four_analysis,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_id")
    parser.add_argument("--camera", choices=("front", "rear"), required=True)
    parser.add_argument(
        "--compare-to",
        help="Baseline experiment ID to compare against after analysis.",
    )
    parser.add_argument(
        "--inspect-2to4",
        action="store_true",
        help="Inventory theoretical TensorRT 2:4 Conv/Linear candidates.",
    )
    parser.add_argument(
        "--fine-tuning-preflight",
        action="store_true",
        help="Validate RF-DETR recovery fine-tuning prerequisites without training.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help="Override the configured recovery dataset path during preflight.",
    )
    args = parser.parse_args()

    registry = ExperimentRegistry.load()
    experiment = registry.get(args.experiment_id)
    if args.camera not in experiment["cameras"]:
        raise ValueError("Camera is not registered for this experiment.")
    paths = artifact_paths(args.experiment_id, args.camera)
    if args.fine_tuning_preflight:
        if experiment["family"] != "semi-structured":
            raise ValueError("Fine-tuning preflight currently targets 2:4 experiments.")
        source_experiment_id = experiment.get(
            "artifact_source", args.experiment_id
        )
        source_paths = artifact_paths(source_experiment_id, args.camera)
        checkpoint_path = source_paths.checkpoint
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        with (REPOSITORY_ROOT / "configs" / "baseline.yaml").open(
            encoding="utf-8"
        ) as stream:
            model_config = yaml.safe_load(stream)["models"][args.camera]
        train_defaults = registry.defaults["fine_tuning"]["recovery_2to4"]
        checkpoint_data = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        checkpoint_args = checkpoint_data.get("args", {})
        original_dataset = Path(str(checkpoint_args.get("dataset_dir", "")))
        configured_dataset_value = (
            args.dataset_dir
            if args.dataset_dir is not None
            else train_defaults.get("dataset_dir")
        )
        configured_dataset = (
            (
                configured_dataset_value
                if Path(configured_dataset_value).is_absolute()
                else REPOSITORY_ROOT / configured_dataset_value
            )
            if configured_dataset_value
            else None
        )
        required_splits = list(train_defaults["required_splits"])
        required_annotation = train_defaults["required_annotation_file"]
        split_checks = {
            split: bool(
                configured_dataset
                and (configured_dataset / split / required_annotation).is_file()
            )
            for split in required_splits
        }
        configured_categories = {}
        configured_format_checks = {}
        for split in required_splits:
            annotation_path = (
                configured_dataset / split / required_annotation
                if configured_dataset
                else None
            )
            if not annotation_path or not annotation_path.is_file():
                configured_categories[split] = []
                configured_format_checks[split] = False
                continue
            split_data = json.loads(annotation_path.read_text(encoding="utf-8"))
            configured_format_checks[split] = all(
                key in split_data
                for key in ("images", "annotations", "categories")
            )
            configured_categories[split] = [
                category["name"]
                for category in sorted(
                    split_data.get("categories", []),
                    key=lambda category: category["id"],
                )
            ]
        dependency_checks = {
            package: importlib.util.find_spec(package) is not None
            for package in (
                "pytorch_lightning",
                "albumentations",
                "pycocotools",
            )
        }
        labeled_test_path = (
            REPOSITORY_ROOT
            / "data"
            / "labeled_test"
            / args.camera
            / "_annotations.coco.json"
        )
        labeled_categories = []
        labeled_test_format_valid = False
        if labeled_test_path.is_file():
            labeled_data = json.loads(labeled_test_path.read_text(encoding="utf-8"))
            labeled_test_format_valid = all(
                key in labeled_data
                for key in ("images", "annotations", "categories")
            )
            labeled_categories = [
                category["name"] for category in labeled_data["categories"]
            ]
        checkpoint_classes = list(checkpoint_args.get("class_names", []))
        expected_classes = list(model_config["classes"])
        checkpoint_load_success = False
        checkpoint_load_error = ""
        try:
            from rfdetr import RFDETR

            RFDETR.from_checkpoint(
                checkpoint_path,
                device="cpu",
                num_classes=len(expected_classes),
            )
            checkpoint_load_success = True
        except Exception as error:  # pragma: no cover - diagnostic path
            checkpoint_load_error = f"{type(error).__name__}: {error}"

        blockers = []
        if configured_dataset is None:
            blockers.append("fine_tuning.recovery_2to4.dataset_dir is not configured")
        missing_splits = [
            split for split, present in split_checks.items() if not present
        ]
        if missing_splits:
            blockers.append(
                "missing Roboflow training splits: " + ", ".join(missing_splits)
            )
        missing_dependencies = [
            package
            for package, present in dependency_checks.items()
            if not present
        ]
        if missing_dependencies:
            blockers.append(
                "missing RF-DETR training dependencies: "
                + ", ".join(missing_dependencies)
            )
        unset_hyperparameters = [
            name
            for name in (
                "epochs",
                "lr",
                "lr_encoder",
                "batch_size",
                "grad_accum_steps",
            )
            if train_defaults.get(name) is None
        ]
        if unset_hyperparameters:
            blockers.append(
                "recovery hyperparameters are not approved: "
                + ", ".join(unset_hyperparameters)
            )
        if not checkpoint_load_success:
            blockers.append("M01 checkpoint failed RF-DETR load")
        if not (
            expected_classes == checkpoint_classes == labeled_categories
        ):
            blockers.append("class order mismatch")
        invalid_configured_splits = [
            split
            for split in required_splits
            if not configured_format_checks.get(split)
        ]
        if invalid_configured_splits:
            blockers.append(
                "invalid configured COCO splits: "
                + ", ".join(invalid_configured_splits)
            )
        configured_class_mismatches = [
            split
            for split in required_splits
            if configured_categories.get(split) != expected_classes
        ]
        if configured_class_mismatches:
            blockers.append(
                "configured dataset class order mismatch: "
                + ", ".join(configured_class_mismatches)
            )

        report = {
            **runtime_metadata(),
            "experiment_id": args.experiment_id,
            "camera": args.camera,
            "training_api": {
                "entry_point": "RFDETR.train(**kwargs)",
                "backend": "PyTorch Lightning",
                "custom_callback_note": (
                    "RFDETR.train discards legacy callbacks dictionaries; use "
                    "RFDETRModelModule/RFDETRDataModule/build_trainer and append "
                    "the 2:4 Lightning callback before trainer.fit."
                ),
            },
            "checkpoint": {
                "path": str(checkpoint_path.relative_to(REPOSITORY_ROOT)),
                "load_success": checkpoint_load_success,
                "load_error": checkpoint_load_error,
                "original_dataset_dir": str(original_dataset),
                "original_dataset_available": original_dataset.is_dir(),
                "original_hyperparameters": {
                    name: checkpoint_args.get(name)
                    for name in (
                        "epochs",
                        "lr",
                        "lr_encoder",
                        "batch_size",
                        "grad_accum_steps",
                        "dataset_file",
                    )
                },
            },
            "configured_recovery": train_defaults,
            "dataset": {
                "configured_path": (
                    str(configured_dataset) if configured_dataset else None
                ),
                "required_split_annotations": split_checks,
                "split_is_coco": configured_format_checks,
                "split_categories": configured_categories,
                "labeled_test_path": str(
                    labeled_test_path.relative_to(REPOSITORY_ROOT)
                ),
                "labeled_test_is_coco": labeled_test_format_valid,
                "labeled_test_is_training_dataset": False,
                "reason": (
                    "labeled_test is an evaluation-only split and has no "
                    "train/valid directory pair"
                ),
            },
            "classes": {
                "expected": expected_classes,
                "checkpoint": checkpoint_classes,
                "labeled_test": labeled_categories,
                "order_matches": (
                    expected_classes == checkpoint_classes == labeled_categories
                ),
            },
            "dependencies": dependency_checks,
            "blockers": blockers,
            "training_allowed": not blockers,
        }
        report_path = paths.directory / "fine-tuning-preflight.json"
        write_json(report_path, report)
        print(f"fine-tuning preflight: {report_path}")
        print(f"training allowed: {report['training_allowed']}")
        refresh_experiment_documents(args.experiment_id)
        return 0
    if args.inspect_2to4:
        from rfdetr import RFDETR

        from kips_lightweighting.static_analysis import two_to_four_eligibility

        with (REPOSITORY_ROOT / "configs" / "baseline.yaml").open(
            encoding="utf-8"
        ) as stream:
            model_config = yaml.safe_load(stream)["models"][args.camera]
        checkpoint = REPOSITORY_ROOT / model_config["checkpoint"]
        wrapper = RFDETR.from_checkpoint(
            checkpoint,
            device="cpu",
            num_classes=len(model_config["classes"]),
        )
        report = {
            **runtime_metadata(),
            "experiment_id": args.experiment_id,
            "camera": args.camera,
            "source_checkpoint": str(checkpoint.relative_to(REPOSITORY_ROOT)),
            **two_to_four_eligibility(wrapper.model.model),
        }
        report_path = paths.directory / "2to4-eligibility.json"
        write_json(report_path, report)
        print(f"2:4 eligibility: {report_path}")
        refresh_experiment_documents(args.experiment_id)
        return 0

    analysis = {**runtime_metadata(), "experiment_id": args.experiment_id}
    source_experiment_id = experiment.get(
        "artifact_source", args.experiment_id
    )
    source_paths = artifact_paths(source_experiment_id, args.camera)
    checkpoint = source_paths.checkpoint
    if not checkpoint.is_file() and experiment["family"] == "baseline":
        with (REPOSITORY_ROOT / "configs" / "baseline.yaml").open(
            encoding="utf-8"
        ) as stream:
            checkpoint = REPOSITORY_ROOT / yaml.safe_load(stream)["models"][
                args.camera
            ]["checkpoint"]
    if checkpoint.is_file():
        analysis["checkpoint"] = checkpoint_analysis(checkpoint)
    source_onnx = source_paths.onnx
    if source_onnx.is_file():
        analysis["onnx"] = onnx_analysis(source_onnx)
        if source_experiment_id != args.experiment_id:
            analysis["onnx"]["source_experiment_id"] = source_experiment_id
        if experiment["family"] == "semi-structured":
            onnx_2to4 = onnx_two_to_four_analysis(source_onnx)
            onnx_2to4_path = source_paths.directory / "onnx-2to4.json"
            write_json(onnx_2to4_path, onnx_2to4)
            analysis["onnx_2to4"] = {
                key: value
                for key, value in onnx_2to4.items()
                if key != "layers"
            }
    if paths.engine.is_file():
        analysis["engine"] = {"engine_size_bytes": paths.engine.stat().st_size}
    if not {"checkpoint", "onnx", "engine"}.intersection(analysis):
        raise FileNotFoundError(f"No checkpoint or ONNX artifact in {paths.directory}")
    write_json(paths.static_analysis, analysis)

    if paths.metadata.is_file():
        metadata = read_json(paths.metadata)
        metadata.setdefault("artifacts", {})["static_analysis"] = str(
            paths.static_analysis
        )
        onnx_2to4_path = source_paths.directory / "onnx-2to4.json"
        if onnx_2to4_path.is_file():
            metadata["artifacts"]["onnx_2to4"] = str(onnx_2to4_path)
        write_json(paths.metadata, metadata)
    print(f"analysis: {paths.static_analysis}")

    if args.compare_to:
        baseline_paths = artifact_paths(args.compare_to, args.camera)
        if not baseline_paths.static_analysis.is_file():
            raise FileNotFoundError(
                f"Baseline analysis not found: {baseline_paths.static_analysis}"
            )
        baseline = read_json(baseline_paths.static_analysis)
        current = read_json(paths.static_analysis)

        def size(path: Path) -> int | None:
            return path.stat().st_size if path.is_file() else None

        def difference(
            candidate: int | float | None,
            reference: int | float | None,
        ) -> dict:
            if candidate is None or reference is None:
                return {
                    "baseline": reference,
                    "candidate": candidate,
                    "delta": None,
                    "delta_ratio": None,
                }
            delta = candidate - reference
            return {
                "baseline": reference,
                "candidate": candidate,
                "delta": delta,
                "delta_ratio": delta / reference if reference else None,
            }

        baseline_onnx = baseline["onnx"]
        current_onnx = current["onnx"]
        baseline_checkpoint = baseline.get("checkpoint", {})
        current_checkpoint = current.get("checkpoint", {})
        comparison = {
            **runtime_metadata(),
            "baseline_experiment_id": args.compare_to,
            "candidate_experiment_id": args.experiment_id,
            "camera": args.camera,
            "engine_size_bytes": difference(
                size(paths.engine), size(baseline_paths.engine)
            ),
            "onnx_size_bytes": difference(
                current_onnx["onnx_size_bytes"],
                baseline_onnx["onnx_size_bytes"],
            ),
            "onnx_initializer_parameters": difference(
                current_onnx["onnx_initializer_parameters"],
                baseline_onnx["onnx_initializer_parameters"],
            ),
            "onnx_nodes": difference(
                current_onnx["onnx_nodes"],
                baseline_onnx["onnx_nodes"],
            ),
            "prunable_parameters": difference(
                current_checkpoint.get("prunable_parameters", 0),
                baseline_checkpoint.get("prunable_parameters", 0),
            ),
            "sparsity": {
                "baseline": baseline_checkpoint.get("sparsity"),
                "candidate": current_checkpoint.get("sparsity"),
            },
            "inputs": {
                "baseline": baseline_onnx["inputs"],
                "candidate": current_onnx["inputs"],
                "equal": baseline_onnx["inputs"] == current_onnx["inputs"],
            },
            "outputs": {
                "baseline": baseline_onnx["outputs"],
                "candidate": current_onnx["outputs"],
                "equal": baseline_onnx["outputs"] == current_onnx["outputs"],
            },
        }
        comparison["conclusion"] = {
            "dense_graph_shape_reduced": (
                comparison["onnx_nodes"]["candidate"]
                < comparison["onnx_nodes"]["baseline"]
                or comparison["onnx_initializer_parameters"]["candidate"]
                < comparison["onnx_initializer_parameters"]["baseline"]
            ),
            "engine_size_materially_reduced": (
                comparison["engine_size_bytes"]["delta_ratio"] is not None
                and comparison["engine_size_bytes"]["delta_ratio"] < -0.01
            ),
        }
        comparison_path = (
            paths.directory / f"comparison-{args.compare_to}.json"
        )
        write_json(comparison_path, comparison)
        print(f"comparison: {comparison_path}")
    refresh_experiment_documents(args.experiment_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
