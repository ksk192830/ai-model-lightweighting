#!/usr/bin/env python3
"""Show a compact, continuously refreshed RF-DETR training dashboard."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

import psutil
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = (
    REPOSITORY_ROOT / "artifacts" / "training" / "front-rfdetr-seg-large-v1"
)
DEFAULT_CONFIG = (
    REPOSITORY_ROOT / "configs" / "training" / "front_rfdetr_seg_large.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--interval", type=float, default=3.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def read_config(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except (FileNotFoundError, yaml.YAMLError):
        return {}


def read_metrics(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except (FileNotFoundError, OSError):
        return []


def populated(row: dict[str, str], key: str) -> bool:
    return bool(row.get(key, "").strip())


def value(row: dict[str, str], key: str) -> str:
    raw = row.get(key, "").strip()
    if not raw:
        return "-"
    try:
        return f"{float(raw):.4f}"
    except ValueError:
        return raw


def training_process(run_dir: Path) -> tuple[str, str]:
    matches: list[psutil.Process] = []
    experiment_id = (
        run_dir.parts[run_dir.parts.index("experiments") + 1]
        if "experiments" in run_dir.parts
        else None
    )
    for process in psutil.process_iter(["pid", "cmdline", "create_time"]):
        try:
            command = " ".join(process.info["cmdline"] or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        is_baseline = "scripts/training/train_rfdetr_front.py" in command
        is_candidate = (
            "scripts/experiments/train_candidate.py" in command
            and experiment_id is not None
            and f"train_candidate.py {experiment_id} " in command
        )
        if (is_baseline or is_candidate) and "fast-dev-run" not in command:
            matches.append(process)
    if not matches:
        return "stopped", "-"
    process = max(matches, key=lambda item: item.info.get("create_time") or 0)
    elapsed = max(0, int(time.time() - float(process.info["create_time"])))
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"running (PID {process.pid})", f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def gpu_status() -> str:
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(
            command,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return "unavailable"
    fields = [part.strip() for part in output.split(",")]
    if len(fields) < 5:
        return output
    return (
        f"util {fields[0]}% | memory {fields[1]}/{fields[2]} MiB | "
        f"temp {fields[3]}°C | power {fields[4]} W"
    )


def latest_checkpoint(run_dir: Path) -> str:
    checkpoints = [
        path
        for path in run_dir.glob("*.ckpt")
        if path.is_file()
    ] + [
        path
        for path in run_dir.glob("checkpoint_best*.pth")
        if path.is_file()
    ]
    if not checkpoints:
        return "none yet"
    path = max(checkpoints, key=lambda item: item.stat().st_mtime)
    modified = datetime.fromtimestamp(path.stat().st_mtime).strftime("%H:%M:%S")
    return f"{path.name} ({modified})"


def timing_estimate(
    run_dir: Path,
    run: dict,
    completed_epochs: int,
    epoch_cap: int,
) -> tuple[str, str]:
    if run.get("status") in {"complete", "completed", "failed"}:
        return "finished", "-"
    if completed_epochs < 1:
        return "waiting for first epoch", "-"
    try:
        started = datetime.fromisoformat(str(run["created_at_utc"])).timestamp()
    except (KeyError, TypeError, ValueError):
        return "unavailable", "-"
    checkpoints = sorted(
        run_dir.glob("checkpoint_*.ckpt"),
        key=lambda path: path.stat().st_mtime,
    )
    if not checkpoints:
        return "unavailable", "-"
    last = checkpoints[-1]
    last_completed = last.stat().st_mtime
    seconds_per_epoch = (last_completed - started) / completed_epochs
    if seconds_per_epoch <= 0:
        return "unavailable", "-"
    spent_in_current = max(0.0, time.time() - last_completed)
    remaining = max(
        0.0,
        seconds_per_epoch * (epoch_cap - completed_epochs) - spent_in_current,
    )
    finish = datetime.now() + timedelta(seconds=remaining)
    average = str(timedelta(seconds=int(seconds_per_epoch)))
    remaining_text = str(timedelta(seconds=int(remaining)))
    return f"{remaining_text} (about {finish:%H:%M})", average


def render(run_dir: Path, config_path: Path) -> str:
    run = read_json(run_dir / "training-run.json")
    if not run:
        run = read_json(run_dir / "recovery-training.json")
    config = read_config(config_path)
    metrics = read_metrics(run_dir / "metrics.csv")
    process_state, elapsed = training_process(run_dir)
    training = config.get("training", {})
    epoch_cap = int(
        run.get("profile", {}).get("epochs", training.get("epochs", 0))
    )

    validation_rows = [
        row
        for row in metrics
        if populated(row, "val/segm_mAP_50_95")
    ]
    completed_epochs = 0
    if validation_rows:
        completed_epochs = (
            int(max(float(row["epoch"]) for row in validation_rows)) + 1
        )
    latest_validation = validation_rows[-1] if validation_rows else {}
    best_validation = (
        max(validation_rows, key=lambda row: float(row["val/segm_mAP_50_95"]))
        if validation_rows
        else {}
    )
    ema_rows = [
        row
        for row in validation_rows
        if populated(row, "val/ema_segm_mAP_50_95")
    ]
    best_ema = (
        max(ema_rows, key=lambda row: float(row["val/ema_segm_mAP_50_95"]))
        if ema_rows
        else {}
    )
    train_rows = [row for row in metrics if populated(row, "train/loss")]
    latest_train = train_rows[-1] if train_rows else {}
    recorded_steps = [
        int(float(row["step"]))
        for row in metrics
        if populated(row, "step")
    ]
    steps_per_epoch = (
        int(float(validation_rows[0]["step"])) + 1
        if validation_rows and populated(validation_rows[0], "step")
        else None
    )
    current_progress = "waiting for first epoch"
    if recorded_steps and steps_per_epoch:
        latest_step = max(recorded_steps)
        current_epoch = min(epoch_cap, latest_step // steps_per_epoch + 1)
        percent = min(
            100,
            round(((latest_step % steps_per_epoch) + 1) / steps_per_epoch * 100),
        )
        current_progress = f"{current_epoch}/{epoch_cap} (~{percent}%, sampled)"
    full_cap_eta, average_epoch = timing_estimate(
        run_dir,
        run,
        completed_epochs,
        epoch_cap,
    )

    status = run.get("status", "waiting for training-run.json")
    if process_state == "stopped" and status == "running":
        status = "process stopped; report has not been finalized"
    elif status in {"complete", "completed"}:
        process_state = "finished"

    lines = [
        "RF-DETR TRAINING MONITOR",
        "=" * 72,
        f"updated       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"process       : {process_state}",
        f"elapsed       : {elapsed}",
        f"run status    : {status}",
        f"epoch         : {completed_epochs}/{epoch_cap} completed",
        f"current epoch : {current_progress}",
        f"avg/epoch     : {average_epoch}",
        f"max-epoch ETA : {full_cap_eta}",
        f"GPU           : {gpu_status()}",
        "",
        "LATEST METRICS (updated after each epoch)",
        "-" * 72,
        f"train loss    : {value(latest_train, 'train/loss')}",
        f"bbox mAP      : {value(latest_validation, 'val/mAP_50_95')}",
        f"mask mAP      : {value(latest_validation, 'val/segm_mAP_50_95')}",
        f"EMA mask mAP  : {value(latest_validation, 'val/ema_segm_mAP_50_95')}",
        f"best mask mAP : {value(best_validation, 'val/segm_mAP_50_95')}",
        f"best EMA mask : {value(best_ema, 'val/ema_segm_mAP_50_95')}",
        f"latest ckpt   : {latest_checkpoint(run_dir)}",
        "",
        f"metrics file  : {run_dir / 'metrics.csv'}",
        "Batch-level progress is visible in the original training terminal.",
        (
            "Training is complete; Ctrl+C closes this monitor."
            if status == "complete"
            else "Press Ctrl+C to close this monitor only; training continues."
        ),
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    run_dir = resolve(args.run_dir)
    config_path = resolve(args.config)
    if args.interval <= 0:
        raise ValueError("--interval must be greater than zero")
    try:
        while True:
            print("\033[2J\033[H" + render(run_dir, config_path), flush=True)
            if args.once:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitor closed. Training was not interrupted.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
