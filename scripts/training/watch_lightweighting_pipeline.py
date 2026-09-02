#!/usr/bin/env python3
"""Show the lightweighting pipeline in one low-overhead terminal dashboard.

This monitor is read-only.  It summarizes the S02 -> M01 GPU lane, the S02
post-recovery CPU queue, independent CPU evaluation jobs, delivery artifacts,
and current GPU utilization without starting or stopping any work.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import psutil


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTERVAL = 3.0
EXPERIMENTS = ("S02", "M01")
CPU_JOB_SCRIPTS = {
    "evaluate_coco_rfdetr_onnx.py": "ONNX full-test",
    "validate_onnx_equivalence.py": "ONNX equivalence",
    "evaluate_coco_rfdetr_pth.py": "PTH full-test",
    "finalize_recovery.py": "recovery finalization",
}


def _load_training_monitor():
    """Load the sibling monitor both as a script and as an imported test module."""
    try:
        from scripts.training import watch_rfdetr_training

        return watch_rfdetr_training
    except ModuleNotFoundError:
        path = Path(__file__).with_name("watch_rfdetr_training.py")
        spec = importlib.util.spec_from_file_location(
            "watch_rfdetr_training", path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load training monitor: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


training_monitor = _load_training_monitor()


@dataclass(frozen=True)
class ProcessRecord:
    pid: int
    command: tuple[str, ...]
    created_at: float

    @property
    def elapsed(self) -> str:
        seconds = max(0, int(time.time() - self.created_at))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def collect_processes() -> list[ProcessRecord]:
    """Take one process snapshot per refresh to keep monitoring inexpensive."""
    records: list[ProcessRecord] = []
    for process in psutil.process_iter(["pid", "cmdline", "create_time"]):
        try:
            command = tuple(process.info.get("cmdline") or ())
            created_at = float(process.info.get("create_time") or time.time())
        except (psutil.AccessDenied, psutil.NoSuchProcess, TypeError, ValueError):
            continue
        if command:
            records.append(ProcessRecord(process.pid, command, created_at))
    return records


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def script_index(command: tuple[str, ...], script_name: str) -> int | None:
    for index, argument in enumerate(command):
        if Path(argument).name == script_name:
            return index
    return None


def option_value(command: tuple[str, ...], option: str) -> str | None:
    try:
        index = command.index(option)
    except ValueError:
        return None
    return command[index + 1] if index + 1 < len(command) else None


def training_processes(
    processes: Iterable[ProcessRecord], experiment_id: str
) -> list[ProcessRecord]:
    matches: list[ProcessRecord] = []
    for process in processes:
        index = script_index(process.command, "train_candidate.py")
        if index is None or index + 1 >= len(process.command):
            continue
        if process.command[index + 1] == experiment_id:
            matches.append(process)
    return matches


def m01_queue_processes(
    processes: Iterable[ProcessRecord], m01_training: Iterable[ProcessRecord]
) -> list[ProcessRecord]:
    training_pids = {item.pid for item in m01_training}
    matches: list[ProcessRecord] = []
    for process in processes:
        if process.pid in training_pids:
            continue
        command = " ".join(process.command)
        if (
            "train_candidate.py M01" in command
            and "S02" in command
            and ("while kill -0" in command or "queue" in command.lower())
        ):
            matches.append(process)
    return matches


def process_status(records: list[ProcessRecord]) -> tuple[str, str]:
    if not records:
        return "stopped", "-"
    leader = min(records, key=lambda item: item.created_at)
    return f"running (PID {leader.pid})", leader.elapsed


def training_summary(
    root: Path,
    experiment_id: str,
    processes: list[ProcessRecord],
) -> dict[str, str]:
    run_dir = root / "artifacts/experiments" / experiment_id / "front/recovery"
    report = read_json(run_dir / "recovery-training.json")
    candidates = training_processes(processes, experiment_id)
    process_state, elapsed = process_status(candidates)
    if not report:
        return {
            "status": "pending (not started)",
            "process": process_state,
            "elapsed": elapsed,
            "epoch": "0/? completed",
            "current": "waiting to start",
            "bbox": "-",
            "mask": "-",
            "best_mask": "-",
            "eta": "-",
            "checkpoint": "none yet",
        }

    metrics = training_monitor.read_metrics(run_dir / "metrics.csv")
    epoch_cap = int(report.get("profile", {}).get("epochs", 0) or 0)
    validation_rows = [
        row
        for row in metrics
        if training_monitor.populated(row, "val/segm_mAP_50_95")
    ]
    completed_epochs = (
        int(max(float(row["epoch"]) for row in validation_rows)) + 1
        if validation_rows
        else 0
    )
    latest_validation = validation_rows[-1] if validation_rows else {}
    best_validation = (
        max(validation_rows, key=lambda row: float(row["val/segm_mAP_50_95"]))
        if validation_rows
        else {}
    )

    recorded_steps = [
        int(float(row["step"]))
        for row in metrics
        if training_monitor.populated(row, "step")
    ]
    steps_per_epoch = (
        int(float(validation_rows[0]["step"])) + 1
        if validation_rows
        and training_monitor.populated(validation_rows[0], "step")
        else None
    )
    current = "waiting for first epoch"
    if recorded_steps and steps_per_epoch and epoch_cap:
        latest_step = max(recorded_steps)
        current_epoch = min(epoch_cap, latest_step // steps_per_epoch + 1)
        percent = min(
            100,
            round(((latest_step % steps_per_epoch) + 1) / steps_per_epoch * 100),
        )
        current = f"{current_epoch}/{epoch_cap} (~{percent}%, sampled)"

    eta, _ = training_monitor.timing_estimate(
        run_dir, report, completed_epochs, epoch_cap
    )
    status = str(report.get("status", "unknown"))
    if status in {"completed", "complete"}:
        process_state = "finished"
        current = "finished"
    elif status == "running" and not candidates:
        status = "process stopped; report not finalized"

    return {
        "status": status,
        "process": process_state,
        "elapsed": elapsed,
        "epoch": f"{completed_epochs}/{epoch_cap} completed",
        "current": current,
        "bbox": training_monitor.value(latest_validation, "val/mAP_50_95"),
        "mask": training_monitor.value(
            latest_validation, "val/segm_mAP_50_95"
        ),
        "best_mask": training_monitor.value(
            best_validation, "val/segm_mAP_50_95"
        ),
        "eta": eta,
        "checkpoint": training_monitor.latest_checkpoint(run_dir),
    }


def queue_summary(
    root: Path,
    processes: list[ProcessRecord],
    experiment_id: str,
) -> list[str]:
    state_path = (
        root
        / "artifacts/experiments"
        / experiment_id
        / "front/post-recovery-queue.json"
    )
    state = read_json(state_path)
    if not state:
        return [f"{experiment_id} CPU queue : pending (state file not created)"]

    pid = state.get("pid")
    live_pids = {process.pid for process in processes}
    status = str(state.get("status", "unknown"))
    activity = "active" if isinstance(pid, int) and pid in live_pids else "not active"
    current = state.get("current_stage") or "waiting for training"
    stages = state.get("stages") if isinstance(state.get("stages"), list) else []
    stage_text = ", ".join(
        f"{stage.get('name', '?')}={stage.get('status', '?')}"
        for stage in stages
        if isinstance(stage, dict)
    ) or "none"
    return [
        f"{experiment_id} CPU queue : {status} | {activity} | PID {pid or '-'}",
        f"current stage : {current}",
        f"stages        : {stage_text}",
    ]


def cpu_jobs(processes: list[ProcessRecord]) -> list[str]:
    jobs: list[str] = []
    for process in processes:
        for script, label in CPU_JOB_SCRIPTS.items():
            index = script_index(process.command, script)
            if index is None:
                continue
            name = option_value(process.command, "--name")
            experiment = option_value(process.command, "--experiment")
            target = name or experiment
            suffix = f" | {target}" if target else ""
            jobs.append(
                f"{label:<21}: running (PID {process.pid}, {process.elapsed}){suffix}"
            )
            break
    return jobs or ["CPU evaluation jobs  : none running"]


def format_size(path: Path) -> str:
    try:
        size = path.stat().st_size
    except OSError:
        return "-"
    units = ("B", "KiB", "MiB", "GiB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def delivery_summary(root: Path) -> list[str]:
    bundle = root / "delivery/notebook-front-ready4"
    archive = root / "delivery/notebook-front-ready4.tar.gz"
    manifest = bundle / "manifest.json"
    data_manifest = bundle / "data-manifest.json"
    bundle_status = (
        "complete (manifests present)"
        if bundle.is_dir() and manifest.is_file() and data_manifest.is_file()
        else "pending/incomplete"
    )
    archive_status = (
        f"complete ({format_size(archive)})" if archive.is_file() else "pending"
    )
    return [
        f"notebook bundle : {bundle_status}",
        f"bundle archive  : {archive_status}",
    ]


def render(
    root: Path = REPOSITORY_ROOT,
    *,
    processes: list[ProcessRecord] | None = None,
    gpu: str | None = None,
) -> str:
    processes = collect_processes() if processes is None else processes
    gpu = training_monitor.gpu_status() if gpu is None else gpu
    summaries = {
        experiment: training_summary(root, experiment, processes)
        for experiment in EXPERIMENTS
    }
    m01_training = training_processes(processes, "M01")
    waiting = m01_queue_processes(processes, m01_training)
    if m01_training:
        m01_queue = "released; M01 training is active"
    elif summaries["M01"]["status"] in {"completed", "complete"}:
        m01_queue = "complete"
    elif waiting:
        leader = min(waiting, key=lambda item: item.created_at)
        m01_queue = f"waiting for S02 (PID {leader.pid}, {leader.elapsed})"
    else:
        m01_queue = "pending (no active queue process)"

    lines = [
        "RF-DETR LIGHTWEIGHTING PIPELINE",
        "=" * 88,
        f"updated       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"GPU           : {gpu}",
        "",
        "GPU LANE (one training job at a time)",
        "-" * 88,
    ]
    for experiment in EXPERIMENTS:
        item = summaries[experiment]
        lines.extend(
            [
                f"{experiment} training  : {item['status']} | {item['process']} | elapsed {item['elapsed']}",
                f"epoch/current : {item['epoch']} | {item['current']}",
                f"latest metrics: bbox {item['bbox']} | mask {item['mask']} | best mask {item['best_mask']}",
                f"max-epoch ETA : {item['eta']} | ckpt {item['checkpoint']}",
            ]
        )
        if experiment == "S02":
            lines.append(f"M01 GPU queue : {m01_queue}")
        lines.append("")

    lines.extend(["CPU LANE", "-" * 88])
    for index, experiment in enumerate(EXPERIMENTS):
        if index:
            lines.append("")
        lines.extend(queue_summary(root, processes, experiment))
    lines.append("")
    lines.extend(cpu_jobs(processes))
    lines.extend(["", "NOTEBOOK DELIVERY", "-" * 88])
    lines.extend(delivery_summary(root))
    lines.extend(
        [
            "",
            "Press Ctrl+C to close this monitor only; pipeline jobs continue.",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.interval <= 0:
        raise ValueError("--interval must be greater than zero")
    try:
        while True:
            print("\033[2J\033[H" + render(), flush=True)
            if args.once:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitor closed. Pipeline jobs were not interrupted.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
