#!/usr/bin/env python3
"""Build and evaluate every Stage-1-passing candidate on the notebook GPU.

Every candidate reaches a terminal Stage-2 state (completed or failed with a
captured log).  Pareto analysis is emitted only after no eligible candidate is
left pending.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
STAGE1 = Path("results/stage1-static-evaluation.json")
STATE = Path("results/stage2-notebook-state.json")
SUMMARY_JSON = Path("results/stage2-notebook-summary.json")
SUMMARY_CSV = Path("results/stage2-notebook-summary.csv")
PARETO_JSON = Path("results/stage3-pareto.json")
PARETO_CSV = Path("results/stage3-pareto.csv")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(command: list[str], log: Path) -> tuple[bool, str]:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("$ " + " ".join(command), flush=True)
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return result.returncode == 0, f"returncode={result.returncode}; log={log.relative_to(ROOT)}"


def metric(payload: dict[str, Any], name: str) -> float:
    metrics = payload["metrics"]
    if name == "bbox_ap":
        return float(metrics["bbox"]["ap"])
    if name == "mask_ap":
        return float(metrics["segm"]["ap"])
    return float(metrics["semantic_miou"])


def newest_json(directory: Path) -> Path:
    paths = sorted(directory.glob("*.json"), key=lambda path: path.stat().st_mtime_ns)
    if not paths:
        raise FileNotFoundError(f"No JSON result in {directory}")
    return paths[-1]


def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    # Lower is better after negating accuracy metrics.
    left_values = (
        -left["bbox_ap"],
        -left["mask_ap"],
        -left["semantic_miou"],
        left["median_ms"],
        left["gpu_peak_allocated_bytes"],
        left["engine_size_bytes"],
    )
    right_values = (
        -right["bbox_ap"],
        -right["mask_ap"],
        -right["semantic_miou"],
        right["median_ms"],
        right["gpu_peak_allocated_bytes"],
        right["engine_size_bytes"],
    )
    return all(a <= b for a, b in zip(left_values, right_values)) and any(
        a < b for a, b in zip(left_values, right_values)
    )


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", help="Optional subset for debugging")
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stage1 = load_json(ROOT / STAGE1)
    if stage1.get("unperformed_count") != 0:
        raise RuntimeError("Stage 1 is incomplete; refusing Stage 2")
    full_eligible = [
        str(row["experiment_id"])
        for row in stage1.get("rows", [])
        if row.get("stage2_notebook_eligible") is True
    ]
    eligible = list(full_eligible)
    if args.only:
        unknown = sorted(set(args.only) - set(eligible))
        if unknown:
            raise ValueError(f"Not Stage-1 eligible: {unknown}")
        eligible = [item for item in eligible if item in set(args.only)]

    if args.dry_run:
        print(f"Stage-1 eligible candidates: {len(eligible)}")
        print(" ".join(eligible))
        print("Stage 2 will build, inspect, benchmark, and evaluate every candidate.")
        print("Stage 3 Pareto remains gated on all candidates reaching a terminal state.")
        return 0

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Stage 2 requires a CUDA notebook GPU")
    import tensorrt as trt

    defaults = yaml.safe_load((ROOT / "configs/experiments/defaults.yaml").read_text(encoding="utf-8"))
    registry = yaml.safe_load((ROOT / "configs/experiments/registry.yaml").read_text(encoding="utf-8"))[
        "experiments"
    ]
    protocol = defaults["evaluation_protocol"]
    latency = protocol["latency"]
    state: dict[str, Any] = {
        "created_at_utc": now(),
        "status": "running",
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "cuda": torch.version.cuda,
        "tensorrt": trt.__version__,
        "eligible_candidates": eligible,
        "candidates": {},
    }
    write_json(ROOT / STATE, state)
    logs = ROOT / "results/stage2-notebook-logs"
    completed_rows: list[dict[str, Any]] = []

    for experiment_id in eligible:
        experiment = registry[experiment_id]
        record: dict[str, Any] = {"status": "running", "started_at_utc": now()}
        state["candidates"][experiment_id] = record
        write_json(ROOT / STATE, state)
        engine = ROOT / f"artifacts/experiments/{experiment_id}/front/model.engine"
        if not args.skip_build:
            command = [
                sys.executable,
                "scripts/experiments/build_candidate.py",
                experiment_id,
                "--camera",
                "front",
                "--target",
                "engine",
            ]
            if args.force_build:
                command.append("--force")
            ok, evidence = run(command, logs / f"{experiment_id}-build.log")
            if not ok:
                record.update(status="build-failed", reason=evidence, finished_at_utc=now())
                write_json(ROOT / STATE, state)
                continue
        if not engine.is_file():
            record.update(status="build-failed", reason="engine file missing", finished_at_utc=now())
            write_json(ROOT / STATE, state)
            continue

        ok, evidence = run(
            [
                sys.executable,
                "scripts/experiments/analyze_tensorrt_engine.py",
                "--experiment",
                experiment_id,
                "--camera",
                "front",
                "--engine",
                str(engine),
                "--precision-label",
                str(experiment["precision"]),
            ],
            logs / f"{experiment_id}-engine-static.log",
        )
        if not ok:
            record.update(status="engine-analysis-failed", reason=evidence, finished_at_utc=now())
            write_json(ROOT / STATE, state)
            continue

        benchmark_dir = ROOT / f"results/benchmarks/notebook/{experiment_id}"
        ok, evidence = run(
            [
                sys.executable,
                "scripts/evaluation/benchmark_baseline.py",
                "--camera",
                "front",
                "--backend",
                "tensorrt",
                "--engine",
                str(engine),
                "--image-dir",
                str(ROOT / protocol["dataset_dir"]),
                "--sample-count",
                str(latency["sampled_images"]),
                "--seed",
                str(latency["sample_seed"]),
                "--threshold",
                str(latency["postprocess_confidence_threshold"]),
                "--warmup",
                str(latency["warmup_runs"]),
                "--runs",
                str(latency["measured_runs"]),
                "--output-dir",
                str(benchmark_dir),
            ],
            logs / f"{experiment_id}-benchmark.log",
        )
        if not ok:
            record.update(status="benchmark-failed", reason=evidence, finished_at_utc=now())
            write_json(ROOT / STATE, state)
            continue
        benchmark_path = newest_json(benchmark_dir)
        benchmark = load_json(benchmark_path)

        ok, evidence = run(
            [
                sys.executable,
                "scripts/evaluation/evaluate_coco_tensorrt.py",
                "--experiment",
                experiment_id,
                "--camera",
                "front",
                "--engine",
                str(engine),
                "--dataset-dir",
                str(ROOT / protocol["dataset_dir"]),
                "--threshold",
                str(protocol["coco_ap_confidence_threshold"]),
                "--miou-threshold",
                str(protocol["semantic_miou_confidence_threshold"]),
                "--no-update-csv",
            ],
            logs / f"{experiment_id}-accuracy.log",
        )
        if not ok:
            record.update(status="accuracy-failed", reason=evidence, finished_at_utc=now())
            write_json(ROOT / STATE, state)
            continue
        evaluation_path = ROOT / f"results/coco-evaluation/{experiment_id}-front.json"
        evaluation = load_json(evaluation_path)
        row = {
            "experiment_id": experiment_id,
            "family": str(experiment["family"]),
            "method": str(experiment["method"]),
            "precision": str(experiment["precision"]),
            "bbox_ap": metric(evaluation, "bbox_ap"),
            "mask_ap": metric(evaluation, "mask_ap"),
            "semantic_miou": metric(evaluation, "semantic_miou"),
            "median_ms": float(benchmark["median_ms"]),
            "p95_ms": float(benchmark["p95_ms"]),
            "fps": float(benchmark["fps"]),
            "gpu_peak_allocated_bytes": int(benchmark["gpu_peak_allocated_bytes"] or 0),
            "gpu_peak_reserved_bytes": int(benchmark["gpu_peak_reserved_bytes"] or 0),
            "engine_size_bytes": engine.stat().st_size,
            "engine": str(engine.relative_to(ROOT)),
            "benchmark": str(benchmark_path.relative_to(ROOT)),
            "evaluation": str(evaluation_path.relative_to(ROOT)),
        }
        completed_rows.append(row)
        record.update(status="completed", result=row, finished_at_utc=now())
        write_json(ROOT / STATE, state)
        print(f"{experiment_id}: Stage 2 completed", flush=True)

    terminal = {
        "completed",
        "build-failed",
        "engine-analysis-failed",
        "benchmark-failed",
        "accuracy-failed",
    }
    pending = [
        experiment_id
        for experiment_id in eligible
        if state["candidates"].get(experiment_id, {}).get("status") not in terminal
    ]
    state["status"] = "completed" if not pending else "incomplete"
    state["finished_at_utc"] = now()
    state["pending_candidates"] = pending
    write_json(ROOT / STATE, state)
    summary = {
        "created_at_utc": now(),
        "eligible_count": len(eligible),
        "completed_count": len(completed_rows),
        "terminal_failure_count": len(eligible) - len(completed_rows) - len(pending),
        "pending_count": len(pending),
        "hardware": {key: state[key] for key in ("gpu", "compute_capability", "cuda", "tensorrt")},
        "protocol": protocol,
        "rows": completed_rows,
    }
    write_json(ROOT / SUMMARY_JSON, summary)
    write_table(ROOT / SUMMARY_CSV, completed_rows)

    # A debug subset must never overwrite the canonical all-candidate Pareto result.
    if not pending and not args.only and eligible == full_eligible:
        pareto_ids = [
            row["experiment_id"]
            for row in completed_rows
            if not any(dominates(other, row) for other in completed_rows if other is not row)
        ]
        pareto_rows = [
            {**row, "pareto_optimal": row["experiment_id"] in pareto_ids}
            for row in completed_rows
        ]
        pareto = {
            "created_at_utc": now(),
            "stage2_all_candidates_terminal": True,
            "objectives": {
                "maximize": ["bbox_ap", "mask_ap", "semantic_miou"],
                "minimize": ["median_ms", "gpu_peak_allocated_bytes", "engine_size_bytes"],
            },
            "pareto_candidate_ids": pareto_ids,
            "rows": pareto_rows,
        }
        write_json(ROOT / PARETO_JSON, pareto)
        write_table(ROOT / PARETO_CSV, pareto_rows)
        print(f"Pareto: {pareto_ids}")
    elif args.only:
        print("Pareto skipped: --only is a debug subset, not the complete Stage-2 cohort.")
    print(f"Stage 2 terminal: {len(eligible) - len(pending)}/{len(eligible)}; pending={pending}")
    return 0 if not pending else 1


if __name__ == "__main__":
    raise SystemExit(main())
