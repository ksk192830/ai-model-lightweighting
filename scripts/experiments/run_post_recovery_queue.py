#!/usr/bin/env python3
"""Wait for recovery training, then run CPU-only finalization and evaluation.

The queue is deliberately conservative: it only promotes a checkpoint after a
completed training report and a matching checkpoint hash. A file lock and a
completed-state guard prevent duplicate runs. Registry, generated docs, and
shared model publication are left for a later accuracy-review step.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEST_DIR = Path("data/training/front_session_split_v1/test")
DEFAULTS_PATH = REPOSITORY_ROOT / "configs/experiments/defaults.yaml"
TERMINAL_FAILURES = {"failed", "cancelled", "canceled", "aborted", "blocked"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def portable(path: Path) -> str:
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}-", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def active_training_processes(experiment_id: str, camera: str) -> list[int]:
    """Return live train_candidate PIDs for the exact experiment/camera."""
    matches: list[int] = []
    for cmdline_path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            arguments = [
                item.decode(errors="replace")
                for item in cmdline_path.read_bytes().split(b"\0")
                if item
            ]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if not any(item.endswith("scripts/experiments/train_candidate.py") for item in arguments):
            continue
        if experiment_id not in arguments:
            continue
        try:
            camera_index = arguments.index("--camera")
        except ValueError:
            continue
        if camera_index + 1 >= len(arguments) or arguments[camera_index + 1] != camera:
            continue
        matches.append(int(cmdline_path.parent.name))
    return sorted(matches)


def validate_completed_training(
    report_path: Path,
    experiment_id: str,
    camera: str,
) -> tuple[dict[str, Any], Path]:
    report = read_json(report_path)
    if report.get("status") != "completed":
        raise ValueError(
            f"Training is not completed: status={report.get('status')!r}"
        )
    if report.get("experiment_id") != experiment_id:
        raise ValueError("Training report experiment_id does not match the queue")
    if report.get("camera") != camera:
        raise ValueError("Training report camera does not match the queue")
    verification = report.get("structure_verification")
    if not isinstance(verification, dict) or verification.get("valid") is not True:
        raise ValueError("Training report has no valid structure verification")
    checkpoint_value = report.get("best_checkpoint")
    expected_hash = report.get("best_checkpoint_sha256")
    if not isinstance(checkpoint_value, str) or not checkpoint_value:
        raise ValueError("Training report has no best checkpoint")
    if not isinstance(expected_hash, str) or not expected_hash:
        raise ValueError("Training report has no best checkpoint hash")
    checkpoint = resolve(Path(checkpoint_value))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint}")
    actual_hash = sha256(checkpoint)
    if actual_hash != expected_hash:
        raise ValueError(
            "Best checkpoint hash mismatch: "
            f"expected={expected_hash}, actual={actual_hash}"
        )
    return report, checkpoint


def save_state(path: Path, state: dict[str, Any], **updates: Any) -> None:
    state.update(updates)
    state["updated_at_utc"] = utc_now()
    write_json_atomic(path, state)


def run_stage(
    stage: dict[str, Any],
    state: dict[str, Any],
    state_path: Path,
    command: list[str],
    environment: dict[str, str],
) -> None:
    stage.update(
        {
            "status": "running",
            "started_at_utc": utc_now(),
            "command": command,
        }
    )
    save_state(state_path, state, status="running", current_stage=stage["name"])
    print("$ " + " ".join(command), flush=True)
    result = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
    )
    stage["returncode"] = result.returncode
    stage["finished_at_utc"] = utc_now()
    if result.returncode != 0:
        stage["status"] = "failed"
        save_state(
            state_path,
            state,
            status="failed",
            current_stage=stage["name"],
            error=f"{stage['name']} exited with code {result.returncode}",
        )
        raise RuntimeError(state["error"])
    stage["status"] = "completed"
    save_state(state_path, state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_id")
    parser.add_argument("--camera", choices=("front", "rear"), required=True)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--resolution", type=int, default=504)
    parser.add_argument("--equivalence-samples", type=int, default=10)
    parser.add_argument("--evaluation-name", required=True)
    parser.add_argument("--poll-interval", type=float, default=15.0)
    parser.add_argument(
        "--retry",
        action="store_true",
        help="Allow a deliberate rerun after a previous failed queue.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.poll_interval <= 0:
        raise ValueError("--poll-interval must be positive")
    if args.equivalence_samples < 1:
        raise ValueError("--equivalence-samples must be positive")
    defaults = yaml.safe_load(DEFAULTS_PATH.read_text(encoding="utf-8"))
    evaluation_protocol = defaults["evaluation_protocol"]
    equivalence_protocol = evaluation_protocol["graph_equivalence"]

    artifact_dir = (
        REPOSITORY_ROOT
        / "artifacts"
        / "experiments"
        / args.experiment_id
        / args.camera
    )
    recovery_dir = artifact_dir / "recovery"
    report_path = recovery_dir / "recovery-training.json"
    state_path = artifact_dir / "post-recovery-queue.json"
    lock_path = artifact_dir / "post-recovery-queue.lock"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"queue already active: {lock_path}", flush=True)
            return 0

        existing = read_json(state_path) if state_path.is_file() else {}
        if existing.get("status") == "completed":
            print(f"queue already completed: {state_path}", flush=True)
            return 0
        if existing.get("status") == "failed" and not args.retry:
            print(
                f"queue previously failed; inspect {state_path} or pass --retry",
                flush=True,
            )
            return 2

        state: dict[str, Any] = {
            "created_at_utc": existing.get("created_at_utc", utc_now()),
            "updated_at_utc": utc_now(),
            "experiment_id": args.experiment_id,
            "camera": args.camera,
            "pid": os.getpid(),
            "status": "waiting-for-training",
            "current_stage": None,
            "training_report": portable(report_path),
            "cpu_only": True,
            "cpu_threads": 3,
            "protocol_source": portable(DEFAULTS_PATH),
            "evaluation_protocol": evaluation_protocol,
            "publication_deferred_until_accuracy_review": True,
            "stages": [
                {"name": "finalize-recovery", "status": "pending"},
                {"name": "onnx-equivalence", "status": "pending"},
                {"name": "pth-test-evaluation", "status": "pending"},
            ],
        }
        save_state(state_path, state)

        missing_process_polls = 0
        while True:
            processes = active_training_processes(args.experiment_id, args.camera)
            if not report_path.is_file():
                status = "missing"
                report = {}
            else:
                try:
                    report = read_json(report_path)
                    status = str(report.get("status", "unknown")).lower()
                except (json.JSONDecodeError, ValueError):
                    report = {}
                    status = "being-written"
            save_state(
                state_path,
                state,
                training_status=status,
                training_pids=processes,
            )
            if status in TERMINAL_FAILURES:
                save_state(
                    state_path,
                    state,
                    status="blocked-training-failed",
                    error=f"Training ended with status={status}; nothing promoted.",
                )
                print(state["error"], flush=True)
                return 3
            if status == "completed" and not processes:
                break
            if processes:
                missing_process_polls = 0
            else:
                missing_process_polls += 1
            if status == "running" and missing_process_polls >= 2:
                save_state(
                    state_path,
                    state,
                    status="blocked-training-disappeared",
                    error=(
                        "Training processes disappeared without a completed report; "
                        "nothing promoted."
                    ),
                )
                print(state["error"], flush=True)
                return 4
            print(
                f"waiting: training_status={status}, pids={processes}",
                flush=True,
            )
            time.sleep(args.poll_interval)

        try:
            report, checkpoint = validate_completed_training(
                report_path, args.experiment_id, args.camera
            )
        except BaseException as error:
            save_state(
                state_path,
                state,
                status="blocked-invalid-training-output",
                error=f"{type(error).__name__}: {error}",
            )
            print(state["error"], flush=True)
            return 5

        state["validated_training"] = {
            "epochs_completed": report.get("epochs_completed"),
            "stopped_early": report.get("stopped_early"),
            "checkpoint": portable(checkpoint),
            "checkpoint_sha256": report["best_checkpoint_sha256"],
        }
        save_state(state_path, state, status="ready")

        python = str(REPOSITORY_ROOT / ".venv" / "bin" / "python")
        dataset_dir = resolve(args.dataset_dir)
        checkpoint_path = artifact_dir / "model.pth"
        onnx_path = artifact_dir / "model.onnx"
        equivalence_path = artifact_dir / "onnx-equivalence.json"
        evaluation_path = (
            REPOSITORY_ROOT
            / "results"
            / "coco-evaluation"
            / f"{args.evaluation_name}.json"
        )
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": "",
                "OMP_NUM_THREADS": "3",
                "MKL_NUM_THREADS": "3",
                "OPENBLAS_NUM_THREADS": "3",
                "KIPS_SKIP_DOCUMENT_REFRESH": "1",
            }
        )
        commands = [
            [
                python,
                str(REPOSITORY_ROOT / "scripts/experiments/finalize_recovery.py"),
                args.experiment_id,
                "--camera",
                args.camera,
                "--force",
                "--skip-engine",
            ],
            [
                python,
                str(REPOSITORY_ROOT / "scripts/evaluation/validate_onnx_equivalence.py"),
                "--checkpoint",
                str(checkpoint_path),
                "--onnx",
                str(onnx_path),
                "--output",
                str(equivalence_path),
                "--image-dir",
                str(dataset_dir),
                "--samples",
                str(args.equivalence_samples),
                "--seed",
                str(equivalence_protocol["sample_seed"]),
                "--confidence-threshold",
                str(equivalence_protocol["confidence_threshold"]),
                "--num-classes",
                str(args.num_classes),
                "--resolution",
                str(args.resolution),
            ],
            [
                python,
                str(REPOSITORY_ROOT / "scripts/evaluation/evaluate_coco_rfdetr_pth.py"),
                "--checkpoint",
                str(checkpoint_path),
                "--dataset-dir",
                str(dataset_dir),
                "--name",
                args.evaluation_name,
                "--device",
                "cpu",
                "--threshold",
                str(evaluation_protocol["coco_ap_confidence_threshold"]),
                "--miou-threshold",
                str(evaluation_protocol["semantic_miou_confidence_threshold"]),
            ],
        ]

        try:
            for stage, command in zip(state["stages"], commands):
                run_stage(stage, state, state_path, command, environment)
        except BaseException as error:
            print(f"post-recovery queue failed: {error}", flush=True)
            return 10

        outputs = {
            "checkpoint": portable(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "onnx": portable(onnx_path),
            "onnx_sha256": sha256(onnx_path),
            "onnx_equivalence": portable(equivalence_path),
            "onnx_equivalence_sha256": sha256(equivalence_path),
            "evaluation": portable(evaluation_path),
            "evaluation_sha256": sha256(evaluation_path),
        }
        save_state(
            state_path,
            state,
            status="completed",
            current_stage=None,
            outputs=outputs,
            completed_at_utc=utc_now(),
        )
        print(f"post-recovery queue completed: {state_path}", flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
