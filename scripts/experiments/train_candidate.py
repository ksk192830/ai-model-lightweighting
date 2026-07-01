#!/usr/bin/env python3
"""Recovery fine-tune a registered pruning candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from kips_lightweighting.artifacts import artifact_paths  # noqa: E402
from kips_lightweighting.metadata import (  # noqa: E402
    refresh_experiment_documents,
    runtime_metadata,
    sha256,
    write_json,
)
from kips_lightweighting.pruning.sparse_2to4 import (  # noqa: E402
    TwoOfFourMaskController,
    make_lightning_mask_callback,
)
from kips_lightweighting.registry import ExperimentRegistry  # noqa: E402
from kips_lightweighting.rfdetr_compat import load_rfdetr_checkpoint  # noqa: E402
from kips_lightweighting.static_analysis import checkpoint_state  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_id")
    parser.add_argument("--camera", choices=("front", "rear"), required=True)
    parser.add_argument(
        "--device",
        default="auto",
        help="Training device: auto, cuda, cuda:N, or cpu (default: auto).",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help="Override the registry dataset path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the registry recovery output path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override the automatically selected per-device batch size.",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        help="Override accumulation; defaults to effective batch size 16.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume from an RF-DETR/Lightning .ckpt file.",
    )
    parser.add_argument(
        "--fast-dev-run",
        type=int,
        default=0,
        help="Run a limited number of train/validation batches for integration testing.",
    )
    return parser.parse_args()


def repository_path(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def resolve_hardware(device: str) -> dict[str, Any]:
    normalized = device.lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "cpu":
        return {
            "device": "cpu",
            "accelerator": "cpu",
            "config_devices": 1,
            "trainer_devices": 1,
            "gpu_index": None,
            "gpu_name": None,
            "gpu_memory_bytes": None,
        }
    if normalized == "cuda":
        gpu_index = 0
    elif normalized.startswith("cuda:") and normalized[5:].isdigit():
        gpu_index = int(normalized[5:])
    else:
        raise ValueError("--device must be auto, cpu, cuda, or cuda:N")
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA was requested ({device}) but torch.cuda.is_available() is false."
        )
    if gpu_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device {gpu_index} does not exist; "
            f"available device count is {torch.cuda.device_count()}."
        )
    if gpu_index != 0:
        raise ValueError(
            "Select a non-default GPU with CUDA_VISIBLE_DEVICES before running "
            "the script, then use --device cuda."
        )
    properties = torch.cuda.get_device_properties(gpu_index)
    return {
        "device": f"cuda:{gpu_index}",
        "accelerator": "gpu",
        "config_devices": 1,
        "trainer_devices": 1,
        "gpu_index": gpu_index,
        "gpu_name": properties.name,
        "gpu_memory_bytes": properties.total_memory,
    }


def automatic_batch_config(
    hardware: dict[str, Any],
    target_effective_batch: int,
) -> tuple[int, int]:
    memory = hardware["gpu_memory_bytes"]
    if memory is None:
        batch_size = 1
    else:
        gib = memory / 1024**3
        if gib >= 40:
            batch_size = 8
        elif gib >= 16:
            batch_size = 4
        elif gib >= 8:
            batch_size = 2
        else:
            batch_size = 1
    accumulation = max(1, target_effective_batch // batch_size)
    return batch_size, accumulation


def validate_dataset(dataset_dir: Path, expected_classes: list[str]) -> None:
    for split in ("train", "valid"):
        annotation_path = dataset_dir / split / "_annotations.coco.json"
        if not annotation_path.is_file():
            raise FileNotFoundError(
                f"Required COCO annotation not found: {annotation_path}"
            )
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        categories = [
            category["name"]
            for category in sorted(
                data.get("categories", []),
                key=lambda category: category["id"],
            )
        ]
        if categories != expected_classes:
            raise ValueError(
                f"{split} class order mismatch: "
                f"{categories} != {expected_classes}"
            )


def load_model_config(camera: str) -> dict:
    with (REPOSITORY_ROOT / "configs" / "baseline.yaml").open(
        encoding="utf-8"
    ) as stream:
        return yaml.safe_load(stream)["models"][camera]


def main() -> int:
    args = parse_args()
    registry = ExperimentRegistry.load()
    experiment = registry.get(args.experiment_id)
    supported_methods = {"nvidia-2to4", "decoder-layer", "ffn-dimension"}
    if experiment["method"] not in supported_methods:
        raise ValueError(
            "Unsupported recovery-training method."
        )
    if args.camera not in experiment["cameras"]:
        raise ValueError(
            f"{args.experiment_id} is not registered for camera {args.camera}."
        )

    paths = artifact_paths(args.experiment_id, args.camera)
    uses_fixed_mask = experiment["method"] == "nvidia-2to4"
    if uses_fixed_mask:
        preflight_path = paths.directory / "fine-tuning-preflight.json"
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        if not preflight.get("training_allowed"):
            raise RuntimeError(
                "Fine-tuning preflight is blocked: "
                + "; ".join(preflight.get("blockers", []))
            )

    profile_name = experiment["fine_tuning"]["profile"]
    profile = registry.defaults["fine_tuning"][profile_name]
    hardware = resolve_hardware(args.device)
    configured_output = str(profile["output_dir"]).format(
        experiment_id=args.experiment_id,
        camera=args.camera,
    )
    output_dir = repository_path(
        args.output_dir or Path(configured_output)
    )
    if args.fast_dev_run:
        output_dir = output_dir / "smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_outputs = [
        item for item in output_dir.iterdir() if item.name != "smoke"
    ]
    resume_path = repository_path(args.resume) if args.resume else None
    if resume_path and not resume_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
    if existing_outputs and not resume_path:
        raise FileExistsError(
            f"Recovery output directory is not empty: {output_dir}. "
            "Pass --resume <checkpoint_*.ckpt> to continue it."
        )

    source_checkpoint = paths.checkpoint
    checkpoint = torch.load(
        source_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    controller = None
    if uses_fixed_mask:
        eligibility_path = paths.directory / "2to4-eligibility.json"
        eligibility = json.loads(eligibility_path.read_text(encoding="utf-8"))
        controller = TwoOfFourMaskController.from_state_dict(
            checkpoint_state(checkpoint),
            eligibility["layers"],
        )

    from rfdetr.training import (
        RFDETRDataModule,
        RFDETRModelModule,
        build_trainer,
    )

    baseline = load_model_config(args.camera)
    model = load_rfdetr_checkpoint(
        source_checkpoint,
        device="cpu",
        num_classes=len(baseline["classes"]),
    )
    dataset_dir = repository_path(
        args.dataset_dir or Path(profile["dataset_dir"])
    )
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    validate_dataset(dataset_dir, list(baseline["classes"]))
    auto_batch_size, auto_accumulation = automatic_batch_config(
        hardware,
        int(profile["effective_batch_size"]),
    )
    batch_size = args.batch_size or auto_batch_size
    grad_accum_steps = (
        args.grad_accum_steps
        or (
            max(1, int(profile["effective_batch_size"]) // batch_size)
            if args.batch_size
            else auto_accumulation
        )
    )
    train_kwargs = {
        key: profile[key]
        for key in (
            "epochs",
            "lr",
            "lr_encoder",
            "weight_decay",
            "lr_scheduler",
            "warmup_epochs",
            "use_ema",
            "ema_decay",
            "early_stopping",
            "early_stopping_patience",
            "checkpoint_interval",
        )
    }
    train_kwargs.update(
        {
            "dataset_dir": str(dataset_dir),
            "dataset_file": profile["dataset_file"],
            "output_dir": str(output_dir),
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps,
            "resume": str(resume_path) if resume_path else None,
            "seed": profile["random_seed"],
            "accelerator": hardware["accelerator"],
            "devices": hardware["config_devices"],
            "strategy": "auto",
            "progress_bar": "tqdm",
            "notes": {
                "experiment_id": args.experiment_id,
                "camera": args.camera,
                "method": experiment["method"],
                "mask_policy": (
                    "gradient masking plus post-optimizer mask reapplication"
                    if uses_fixed_mask
                    else "not applicable to structured layer removal"
                ),
                "source_checkpoint_sha256": sha256(source_checkpoint),
            },
        }
    )
    config = model.get_train_config(**train_kwargs)
    model._align_num_classes_from_dataset(str(dataset_dir))
    model.model_config.model_name = type(model).__name__
    module = RFDETRModelModule(model.model_config, config)
    if experiment["method"] == "ffn-dimension":
        # RF-DETR 1.8.1 does not expose dim_feedforward in ModelConfig.
        # Replace the default-width training model with the already loaded
        # project-defined structured model before optimizer construction.
        module.model = model.model.model
    datamodule = RFDETRDataModule(model.model_config, config)
    trainer_kwargs = {
        "accelerator": hardware["accelerator"],
        "devices": hardware["trainer_devices"],
    }
    if args.fast_dev_run:
        trainer_kwargs["fast_dev_run"] = args.fast_dev_run
    trainer = build_trainer(config, model.model_config, **trainer_kwargs)
    if controller is not None:
        trainer.callbacks.append(make_lightning_mask_callback(controller))

    run_report = {
        **runtime_metadata(),
        "experiment_id": args.experiment_id,
        "camera": args.camera,
        "status": "running",
        "source_checkpoint": str(
            source_checkpoint.relative_to(REPOSITORY_ROOT)
        ),
        "source_checkpoint_sha256": sha256(source_checkpoint),
        "output_dir": str(output_dir.relative_to(REPOSITORY_ROOT)),
        "profile": profile,
        "resolved_hardware": hardware,
        "resolved_training": {
            "dataset_dir": str(dataset_dir),
            "output_dir": str(output_dir),
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps,
            "effective_batch_size": batch_size * grad_accum_steps,
            "resume": str(resume_path) if resume_path else None,
        },
        "fast_dev_run": args.fast_dev_run,
        "mask_layers": len(controller.masks) if controller else 0,
    }
    report_path = output_dir / "recovery-training.json"
    write_json(report_path, run_report)

    try:
        trainer.fit(
            module,
            datamodule,
            ckpt_path=str(resume_path) if resume_path else None,
        )
        if controller is not None:
            verification = controller.verify(module.model)
            if not verification["valid"]:
                raise RuntimeError(
                    "2:4 mask verification failed after recovery training."
                )
        elif experiment["method"] == "decoder-layer":
            expected_layers = int(model.model_config.dec_layers)
            loaded_layers = len(module.model.transformer.decoder.layers)
            loaded_blocks = len(module.model.segmentation_head.blocks)
            verification = {
                "expected_decoder_layers": expected_layers,
                "decoder_layers": loaded_layers,
                "segmentation_blocks": loaded_blocks,
                "valid": (
                    expected_layers == loaded_layers == loaded_blocks
                ),
            }
            if not verification["valid"]:
                raise RuntimeError(
                    "Structured decoder/segmentation layer verification failed."
                )
        else:
            architecture = checkpoint["structured_architecture"]
            expected_width = int(architecture["dim_feedforward"])
            widths = [
                (layer.linear1.out_features, layer.linear2.in_features)
                for layer in module.model.transformer.decoder.layers
            ]
            verification = {
                "expected_dim_feedforward": expected_width,
                "decoder_ffn_widths": widths,
                "valid": all(
                    incoming == outgoing == expected_width
                    for incoming, outgoing in widths
                ),
            }
            if not verification["valid"]:
                raise RuntimeError("Structured FFN width verification failed.")
        best_checkpoint = output_dir / "checkpoint_best_total.pth"
        if (
            experiment["method"] == "ffn-dimension"
            and best_checkpoint.is_file()
        ):
            trained_checkpoint = torch.load(
                best_checkpoint,
                map_location="cpu",
                weights_only=False,
            )
            trained_checkpoint["structured_architecture"] = checkpoint[
                "structured_architecture"
            ]
            trained_checkpoint["pruning"] = checkpoint["pruning"]
            torch.save(trained_checkpoint, best_checkpoint)
        run_report.update(
            {
                "status": "completed",
                "epochs_completed": trainer.current_epoch,
                "global_step": trainer.global_step,
                "stopped_early": trainer.should_stop,
                "structure_verification": verification,
                "best_checkpoint": (
                    str(best_checkpoint.relative_to(REPOSITORY_ROOT))
                    if best_checkpoint.is_file()
                    else None
                ),
                "best_checkpoint_sha256": (
                    sha256(best_checkpoint)
                    if best_checkpoint.is_file()
                    else None
                ),
            }
        )
    except BaseException as error:
        run_report.update(
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "epochs_completed": trainer.current_epoch,
                "global_step": trainer.global_step,
            }
        )
        raise
    finally:
        write_json(report_path, run_report)
        if controller is not None:
            controller.close()

    print(f"recovery checkpoint: {run_report['best_checkpoint']}")
    print(f"training report: {report_path}")
    refresh_experiment_documents(args.experiment_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
