#!/usr/bin/env python3
"""Preflight and train the paper baseline RF-DETR Segmentation Large model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPOSITORY_ROOT / "configs" / "training" / "front_rfdetr_seg_large.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or cuda:0 (default: auto)",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate the environment, data, and configuration without training.",
    )
    parser.add_argument(
        "--fast-dev-run",
        type=int,
        default=0,
        help="Run this many train/validation batches as a smoke test.",
    )
    parser.add_argument(
        "--allow-provisional-split",
        action="store_true",
        help="Permit the known provisional split for preflight/smoke only.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def file_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or config.get("version") != 1:
        raise ValueError(f"Unsupported training config: {path}")
    if config["model"]["architecture"] != "RFDETRSegLarge":
        raise ValueError("Only RFDETRSegLarge is supported by this training entry point.")
    training = config["training"]
    if (
        int(training["batch_size"]) * int(training["grad_accum_steps"])
        != int(training["effective_batch_size"])
    ):
        raise ValueError(
            "batch_size * grad_accum_steps must equal effective_batch_size"
        )
    return config


def resolve_device(requested: str) -> dict[str, Any]:
    normalized = requested.lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "cpu":
        return {
            "device": "cpu",
            "accelerator": "cpu",
            "devices": 1,
            "gpu_name": None,
            "gpu_memory_bytes": None,
        }
    if normalized not in {"cuda", "cuda:0"}:
        raise ValueError("--device must be auto, cpu, cuda, or cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    properties = torch.cuda.get_device_properties(0)
    return {
        "device": "cuda",
        "accelerator": "gpu",
        "devices": 1,
        "gpu_name": properties.name,
        "gpu_memory_bytes": properties.total_memory,
    }


def find_image(split_dir: Path, file_name: str) -> Path | None:
    name = Path(file_name).name
    for candidate in (split_dir / name, split_dir / "images" / name):
        if candidate.is_file():
            return candidate
    return None


def validate_split(
    split_dir: Path,
    annotation_file: str,
    expected_classes: list[str],
) -> dict[str, Any]:
    annotation_path = split_dir / annotation_file
    if not annotation_path.is_file():
        raise FileNotFoundError(f"COCO annotation is missing: {annotation_path}")
    with annotation_path.open(encoding="utf-8") as handle:
        coco = json.load(handle)
    categories = sorted(coco.get("categories", []), key=lambda row: int(row["id"]))
    class_names = [str(category["name"]) for category in categories]
    if class_names != expected_classes:
        raise ValueError(
            f"Class order mismatch in {split_dir.name}: "
            f"{class_names} != {expected_classes}"
        )
    image_ids = [int(image["id"]) for image in coco.get("images", [])]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError(f"Duplicate COCO image IDs in {annotation_path}")
    image_id_set = set(image_ids)
    orphan_annotations = [
        annotation
        for annotation in coco.get("annotations", [])
        if int(annotation["image_id"]) not in image_id_set
    ]
    if orphan_annotations:
        raise ValueError(
            f"{len(orphan_annotations)} orphan annotations in {annotation_path}"
        )
    missing_images = [
        image["file_name"]
        for image in coco.get("images", [])
        if find_image(split_dir, str(image["file_name"])) is None
    ]
    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} COCO images are missing in {split_dir}; "
            f"first={missing_images[0]}"
        )
    annotations_by_image: dict[int, int] = defaultdict(int)
    category_annotations: Counter[int] = Counter()
    for annotation in coco.get("annotations", []):
        annotations_by_image[int(annotation["image_id"])] += 1
        category_annotations[int(annotation["category_id"])] += 1
    return {
        "images": len(image_ids),
        "annotations": len(coco.get("annotations", [])),
        "negative_images": sum(annotations_by_image[image_id] == 0 for image_id in image_ids),
        "class_annotations": {
            str(category["name"]): category_annotations[int(category["id"])]
            for category in categories
        },
    }


def read_paper_readiness(
    dataset_dir: Path,
    readiness_report: Path,
) -> dict[str, Any]:
    if not readiness_report.is_file():
        return {
            "ready": False,
            "reason": f"Readiness report is missing: {readiness_report}",
            "report": str(readiness_report),
        }
    report = json.loads(readiness_report.read_text(encoding="utf-8"))
    verification = report.get("verification", {})
    audited_dataset = report.get("dataset", {}).get("grouped")
    same_dataset = False
    if audited_dataset:
        same_dataset = Path(audited_dataset).resolve() == dataset_dir.resolve()
    ready = bool(verification.get("paper_evaluation_ready")) and same_dataset
    reason = verification.get("reason")
    if not same_dataset:
        reason = "Readiness report does not refer to the selected dataset directory."
    return {
        "ready": ready,
        "reason": reason,
        "report": str(readiness_report),
        "audited_dataset": audited_dataset,
    }


def environment_report(hardware: dict[str, Any]) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": package_version("torchvision"),
        "rfdetr": package_version("rfdetr"),
        "pytorch_lightning": package_version("pytorch-lightning"),
        "albumentations": package_version("albumentations"),
        "pycocotools": package_version("pycocotools"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "hardware": hardware,
    }


def preflight(
    config: dict[str, Any],
    dataset_dir: Path,
    readiness_report: Path,
    hardware: dict[str, Any],
    allow_provisional: bool,
) -> dict[str, Any]:
    required_packages = {
        "rfdetr": package_version("rfdetr"),
        "pytorch-lightning": package_version("pytorch-lightning"),
        "albumentations": package_version("albumentations"),
        "pycocotools": package_version("pycocotools"),
    }
    missing_packages = [name for name, value in required_packages.items() if value is None]
    if missing_packages:
        raise RuntimeError("Missing training packages: " + ", ".join(missing_packages))
    if required_packages["rfdetr"] != "1.8.1":
        raise RuntimeError(
            f"RF-DETR 1.8.1 is required; found {required_packages['rfdetr']}"
        )
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory is missing: {dataset_dir}")
    dataset_config = config["dataset"]
    split_stats = {
        split: validate_split(
            dataset_dir / split,
            str(dataset_config["annotation_file"]),
            list(dataset_config["classes"]),
        )
        for split in dataset_config["required_splits"]
    }
    readiness = read_paper_readiness(dataset_dir, readiness_report)
    if dataset_config.get("require_paper_ready", True) and not readiness["ready"]:
        if not allow_provisional:
            raise RuntimeError(
                "Dataset is not approved for paper training/evaluation: "
                + str(readiness["reason"])
            )
    return {
        "checked_at_utc": now_utc(),
        "environment": environment_report(hardware),
        "dataset_dir": str(dataset_dir),
        "splits": split_stats,
        "paper_readiness": readiness,
        "provisional_override": allow_provisional,
        "status": "pass" if readiness["ready"] else "provisional-pass",
    }


def main() -> int:
    args = parse_args()
    if args.allow_provisional_split and not (args.preflight_only or args.fast_dev_run):
        raise ValueError(
            "--allow-provisional-split is restricted to preflight or smoke tests; "
            "it cannot authorize a full training run."
        )
    config_path = resolve(args.config)
    config = load_config(config_path)
    dataset_dir = resolve(args.dataset_dir or Path(config["dataset"]["directory"]))
    output_dir = resolve(args.output_dir or Path(config["run"]["output_dir"]))
    if args.fast_dev_run:
        output_dir = output_dir / "smoke"
    readiness_report = resolve(Path(config["dataset"]["readiness_report"]))
    hardware = resolve_device(args.device)
    preflight_path = output_dir / "preflight.json"
    try:
        preflight_report = preflight(
            config,
            dataset_dir,
            readiness_report,
            hardware,
            args.allow_provisional_split,
        )
    except Exception as error:
        write_json(
            preflight_path,
            {
                "checked_at_utc": now_utc(),
                "status": "blocked",
                "dataset_dir": str(dataset_dir),
                "environment": environment_report(hardware),
                "error": str(error),
            },
        )
        raise
    write_json(preflight_path, preflight_report)
    print(json.dumps(preflight_report, indent=2, ensure_ascii=False))
    print(f"preflight report: {preflight_path}")
    if args.preflight_only:
        return 0

    existing = [path for path in output_dir.iterdir() if path.name != "preflight.json"]
    resume_path = resolve(args.resume) if args.resume else None
    if existing and resume_path is None:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Use a new directory or pass --resume checkpoint_<epoch>.ckpt."
        )
    if resume_path is not None and not resume_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint is missing: {resume_path}")

    model_config = config["model"]
    pretrained = resolve(Path(model_config["pretrained_weights"]))
    pretrained.parent.mkdir(parents=True, exist_ok=True)
    expected_md5 = str(model_config["pretrained_md5"])
    if pretrained.is_file() and file_md5(pretrained) != expected_md5:
        raise RuntimeError(f"Pretrained-weight MD5 mismatch: {pretrained}")

    from pytorch_lightning import seed_everything
    from rfdetr import RFDETRSegLarge
    from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer

    seed_everything(int(config["training"]["random_seed"]), workers=True)
    model = RFDETRSegLarge(
        pretrain_weights=str(pretrained),
        device=hardware["device"],
        resolution=int(model_config["resolution"]),
    )
    if not pretrained.is_file() or file_md5(pretrained) != expected_md5:
        raise RuntimeError(f"Pretrained weights were not downloaded correctly: {pretrained}")

    training = config["training"]
    train_kwargs = {
        "dataset_dir": str(dataset_dir),
        "dataset_file": config["dataset"]["format"],
        "output_dir": str(output_dir),
        "epochs": int(training["epochs"]),
        "batch_size": int(training["batch_size"]),
        "grad_accum_steps": int(training["grad_accum_steps"]),
        "lr": float(training["lr"]),
        "lr_encoder": float(training["lr_encoder"]),
        "weight_decay": float(training["weight_decay"]),
        "lr_scheduler": str(training["lr_scheduler"]),
        "lr_drop": int(training["lr_drop"]),
        "warmup_epochs": float(training["warmup_epochs"]),
        "multi_scale": bool(training["multi_scale"]),
        "expanded_scales": bool(training["expanded_scales"]),
        "use_ema": bool(training["use_ema"]),
        "ema_decay": float(training["ema_decay"]),
        "early_stopping": bool(training["early_stopping"]),
        "early_stopping_patience": int(training["early_stopping_patience"]),
        "early_stopping_min_delta": float(training["early_stopping_min_delta"]),
        "checkpoint_interval": int(training["checkpoint_interval"]),
        "eval_interval": int(training["eval_interval"]),
        "eval_max_dets": int(training["eval_max_dets"]),
        "cls_loss_coef": float(training["cls_loss_coef"]),
        "mask_point_sample_ratio": int(training["mask_point_sample_ratio"]),
        "mask_ce_loss_coef": float(training["mask_ce_loss_coef"]),
        "mask_dice_loss_coef": float(training["mask_dice_loss_coef"]),
        "num_workers": int(training["num_workers"]),
        "seed": int(training["random_seed"]),
        "tensorboard": bool(training["tensorboard"]),
        "run_test": bool(training["run_test"]),
        "resume": str(resume_path) if resume_path else None,
        "accelerator": hardware["accelerator"],
        "devices": hardware["devices"],
        "strategy": "auto",
        "progress_bar": "tqdm",
        "notes": {
            "run_name": config["run"]["name"],
            "config": str(config_path.relative_to(REPOSITORY_ROOT)),
            "dataset_readiness": preflight_report["paper_readiness"],
            "effective_batch_size": int(training["effective_batch_size"]),
            "original_checkpoint_training_match": {
                "architecture": True,
                "resolution": True,
                "historical_epoch_cap": 40,
                "historical_epochs_completed": 19,
                "new_resumable_epoch_cap": int(training["epochs"]),
                "optimizer_hyperparameters": True,
                "effective_batch_size": True,
                "seed_fixed_for_reproducibility": 42,
                "automatic_test_disabled_for_blind_final_evaluation": True,
            },
        },
    }
    train_config = model.get_train_config(**train_kwargs)
    model._align_num_classes_from_dataset(str(dataset_dir))
    model.model_config.model_name = type(model).__name__
    module = RFDETRModelModule(model.model_config, train_config)
    datamodule = RFDETRDataModule(model.model_config, train_config)
    trainer_kwargs: dict[str, Any] = {
        "accelerator": hardware["accelerator"],
        "devices": hardware["devices"],
    }
    if args.fast_dev_run:
        trainer_kwargs["fast_dev_run"] = args.fast_dev_run
    trainer = build_trainer(train_config, model.model_config, **trainer_kwargs)

    run_report = {
        "created_at_utc": now_utc(),
        "status": "running",
        "config": str(config_path),
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "pretrained_weights": str(pretrained),
        "pretrained_md5": file_md5(pretrained),
        "resume": str(resume_path) if resume_path else None,
        "fast_dev_run": args.fast_dev_run,
        "environment": environment_report(hardware),
        "resolved_training": train_config.model_dump(),
    }
    run_report_path = output_dir / "training-run.json"
    write_json(run_report_path, run_report)
    try:
        trainer.fit(
            module,
            datamodule,
            ckpt_path=str(resume_path) if resume_path else None,
        )
        run_report["status"] = "smoke-pass" if args.fast_dev_run else "complete"
        run_report["finished_at_utc"] = now_utc()
        run_report["current_epoch"] = trainer.current_epoch
        run_report["global_step"] = trainer.global_step
        write_json(run_report_path, run_report)
    except Exception as error:
        run_report["status"] = "failed"
        run_report["finished_at_utc"] = now_utc()
        run_report["error"] = repr(error)
        run_report["traceback"] = traceback.format_exc()
        write_json(run_report_path, run_report)
        raise

    print(f"training report: {run_report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
