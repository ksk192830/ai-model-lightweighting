#!/usr/bin/env python3
"""Finish the local RF-DETR TensorRT build, benchmark, and evaluation queue.

The queue waits until S02 and M01 recovery/post-processing are complete so it
never contends with training for the desktop GPU or evaluates prototype ONNX
files. Independent engine failures are recorded and do not prevent the other
candidates from being attempted.
"""

from __future__ import annotations

import argparse
import csv
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
PYTHON = REPOSITORY_ROOT / ".venv/bin/python"
DEFAULTS_PATH = REPOSITORY_ROOT / "configs/experiments/defaults.yaml"
RECOVERY_IDS = ("S02", "M01")
PRE_SELECTION_ENGINES = (
    "B01",
    "B02",
    "B03",
    "R01",
    "S01",
    "S02",
    "M01",
    "M02",
)
BACKENDS = {
    "B01": "fp32",
    "B02": "fp16",
    "B03": "int8",
    "R01": "fp16",
    "S01": "fp32",
    "S02": "fp32",
    "C01": "fp16",
    "M01": "fp16",
    "M02": "fp16",
}
PTH_RESULTS = {
    "S01": Path("results/coco-evaluation/S01-front-after-recovery.json"),
    "S02": Path("results/coco-evaluation/S02-front-after-recovery.json"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def portable(path: Path) -> str:
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def read_defaults() -> dict[str, Any]:
    value = yaml.safe_load(DEFAULTS_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {DEFAULTS_PATH}")
    return value


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def recovery_state(experiment_id: str) -> tuple[str, str]:
    queue = (
        REPOSITORY_ROOT
        / "artifacts/experiments"
        / experiment_id
        / "front/post-recovery-queue.json"
    )
    if not queue.is_file():
        return "missing", portable(queue)
    data = read_json(queue)
    return str(data.get("status", "unknown")), portable(queue)


def wait_for_recovery(state: dict[str, Any], state_path: Path, poll: float) -> None:
    failures = (
        "failed",
        "blocked",
        "cancelled",
        "canceled",
        "aborted",
        "disappeared",
        "invalid",
    )
    while True:
        current = {item: recovery_state(item)[0] for item in RECOVERY_IDS}
        state["recovery"] = current
        state["status"] = "waiting-for-recovery"
        state["updated_at_utc"] = utc_now()
        write_json_atomic(state_path, state)
        print("recovery: " + ", ".join(f"{key}={value}" for key, value in current.items()), flush=True)
        if all(value == "completed" for value in current.values()):
            return
        terminal = {
            key: value
            for key, value in current.items()
            if any(token in value.lower() for token in failures)
        }
        if terminal:
            raise RuntimeError(f"Recovery queue failed: {terminal}")
        time.sleep(poll)


def run_stage(
    state: dict[str, Any],
    state_path: Path,
    name: str,
    command: list[str],
    log_dir: Path,
) -> bool:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    stage = {
        "name": name,
        "status": "running",
        "started_at_utc": utc_now(),
        "command": command,
        "log": portable(log_path),
    }
    state.setdefault("stages", {})[name] = stage
    state["status"] = "running"
    state["current_stage"] = name
    state["updated_at_utc"] = utc_now()
    write_json_atomic(state_path, state)
    print("$ " + " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    stage.update(
        {
            "status": "completed" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "finished_at_utc": utc_now(),
        }
    )
    state["updated_at_utc"] = utc_now()
    write_json_atomic(state_path, state)
    if result.returncode:
        state.setdefault("errors", []).append(
            {"stage": name, "returncode": result.returncode, "log": portable(log_path)}
        )
        write_json_atomic(state_path, state)
        print(f"failed: {name}; see {log_path}", flush=True)
        return False
    print(f"completed: {name}", flush=True)
    return True


def engine_path(experiment_id: str) -> Path:
    return REPOSITORY_ROOT / f"artifacts/experiments/{experiment_id}/front/model.engine"


def analyze_engine(
    experiment_id: str,
    state: dict[str, Any],
    state_path: Path,
    log_dir: Path,
) -> bool:
    engine = engine_path(experiment_id)
    if not engine.is_file():
        return False
    return run_stage(
        state,
        state_path,
        f"static-engine-{experiment_id}",
        [
            str(PYTHON),
            "scripts/experiments/analyze_tensorrt_engine.py",
            "--experiment",
            experiment_id,
            "--camera",
            "front",
            "--engine",
            str(engine),
        ],
        log_dir,
    )


def build_registered(
    experiment_id: str,
    state: dict[str, Any],
    state_path: Path,
    log_dir: Path,
    force: bool,
) -> bool:
    command = [
        str(PYTHON),
        "scripts/experiments/build_candidate.py",
        experiment_id,
        "--camera",
        "front",
        "--target",
        "engine",
    ]
    if force:
        command.append("--force")
    return run_stage(
        state, state_path, f"build-{experiment_id}", command, log_dir
    ) and engine_path(experiment_id).is_file()


def build_selected_c01(
    selected: str,
    state: dict[str, Any],
    state_path: Path,
    log_dir: Path,
    force: bool,
) -> bool:
    onnx = REPOSITORY_ROOT / f"artifacts/experiments/{selected}/front/model.onnx"
    command = [
        str(PYTHON),
        "scripts/lightweighting/build_tensorrt_fp16.py",
        "--camera",
        "front",
        "--onnx",
        str(onnx),
        "--output-dir",
        str(REPOSITORY_ROOT / "artifacts/experiments/C01"),
        "--output-name",
        "model",
        "--workspace-mib",
        "4096",
    ]
    if force:
        command.append("--force")
    passed = run_stage(state, state_path, "build-C01", command, log_dir)
    directory = REPOSITORY_ROOT / "artifacts/experiments/C01/front"
    generated = directory / "model.json"
    if passed and generated.is_file():
        generated.replace(directory / "engine-build.json")
    if passed:
        (directory / "selected-source.json").write_text(
            json.dumps(
                {
                    "created_at_utc": utc_now(),
                    "selected_experiment": selected,
                    "source_onnx": portable(onnx),
                    "source_onnx_sha256": sha256(onnx),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    return passed and engine_path("C01").is_file()


def newest_benchmark(experiment_id: str) -> dict[str, Any]:
    directory = REPOSITORY_ROOT / "results/benchmarks/desktop" / experiment_id
    files = sorted(directory.glob("*.json"), key=lambda path: path.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No benchmark JSON in {directory}")
    return read_json(files[-1])


def benchmark(
    experiment_id: str,
    image_dir: Path,
    sample_count: int,
    seed: int,
    threshold: float,
    warmup: int,
    runs: int,
    state: dict[str, Any],
    state_path: Path,
    log_dir: Path,
) -> dict[str, Any] | None:
    engine = engine_path(experiment_id)
    if not engine.is_file():
        return None
    command = [
        str(PYTHON),
        "scripts/evaluation/benchmark_baseline.py",
        "--camera",
        "front",
        "--backend",
        BACKENDS[experiment_id],
        "--engine",
        str(engine),
        "--image-dir",
        str(image_dir),
        "--sample-count",
        str(sample_count),
        "--seed",
        str(seed),
        "--threshold",
        str(threshold),
        "--warmup",
        str(warmup),
        "--runs",
        str(runs),
        "--output-dir",
        str(REPOSITORY_ROOT / "results/benchmarks/desktop" / experiment_id),
    ]
    if not run_stage(
        state, state_path, f"benchmark-{experiment_id}", command, log_dir
    ):
        return None
    return newest_benchmark(experiment_id)


def metric(result: dict[str, Any], name: str) -> float:
    metrics = result["metrics"]
    if name == "bbox_ap":
        return float(metrics["bbox"]["ap"])
    if name == "mask_ap":
        return float(metrics["segm"]["ap"])
    if name == "semantic_miou":
        return float(metrics["semantic_miou"])
    raise KeyError(name)


def choose_structured(
    s01_accuracy: dict[str, Any],
    s02_accuracy: dict[str, Any],
    s01_benchmark: dict[str, Any],
    s02_benchmark: dict[str, Any],
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy = policy or {
        "bbox_ap_max_absolute_drop": 0.005,
        "mask_ap_max_absolute_drop": 0.005,
        "semantic_miou_max_absolute_drop": 0.01,
        "median_latency_min_relative_reduction": 0.05,
    }
    tolerances = {
        "bbox_ap": float(policy["bbox_ap_max_absolute_drop"]),
        "mask_ap": float(policy["mask_ap_max_absolute_drop"]),
        "semantic_miou": float(policy["semantic_miou_max_absolute_drop"]),
    }
    accuracy_gates = {
        name: metric(s02_accuracy, name) >= metric(s01_accuracy, name) - tolerance
        for name, tolerance in tolerances.items()
    }
    s01_median = float(s01_benchmark["median_ms"])
    s02_median = float(s02_benchmark["median_ms"])
    latency_reduction = (s01_median - s02_median) / s01_median
    minimum_latency_reduction = float(
        policy["median_latency_min_relative_reduction"]
    )
    latency_gate = latency_reduction >= minimum_latency_reduction
    selected = "S02" if all(accuracy_gates.values()) and latency_gate else "S01"
    return {
        "created_at_utc": utc_now(),
        "selected_experiment": selected,
        "policy_source": portable(DEFAULTS_PATH),
        "policy": policy,
        "metric_deltas_candidate_minus_reference": {
            name: metric(s02_accuracy, name) - metric(s01_accuracy, name)
            for name in tolerances
        },
        "accuracy_gates": accuracy_gates,
        "latency_gate": latency_gate,
        "s01_median_ms": s01_median,
        "s02_median_ms": s02_median,
        "s02_latency_reduction_fraction": latency_reduction,
    }


def evaluate(
    experiment_id: str,
    dataset: Path,
    state: dict[str, Any],
    state_path: Path,
    log_dir: Path,
    ap_threshold: float,
    miou_threshold: float,
) -> dict[str, Any] | None:
    engine = engine_path(experiment_id)
    if not engine.is_file():
        return None
    command = [
        str(PYTHON),
        "scripts/evaluation/evaluate_coco_tensorrt.py",
        "--experiment",
        experiment_id,
        "--camera",
        "front",
        "--engine",
        str(engine),
        "--dataset-dir",
        str(dataset),
        "--threshold",
        str(ap_threshold),
        "--miou-threshold",
        str(miou_threshold),
        "--no-update-csv",
    ]
    if not run_stage(
        state, state_path, f"evaluate-{experiment_id}", command, log_dir
    ):
        return None
    output = REPOSITORY_ROOT / f"results/coco-evaluation/{experiment_id}-front.json"
    return read_json(output)


def write_summary(
    selection: dict[str, Any],
    benchmarks: dict[str, dict[str, Any]],
    evaluations: dict[str, dict[str, Any]],
) -> None:
    baseline = evaluations.get("B01")
    rows = []
    for experiment_id in BACKENDS:
        engine = engine_path(experiment_id)
        benchmark_result = benchmarks.get(experiment_id)
        evaluation = evaluations.get(experiment_id)
        if not engine.is_file() or benchmark_result is None or evaluation is None:
            continue
        row: dict[str, Any] = {
            "experiment_id": experiment_id,
            "precision": BACKENDS[experiment_id],
            "selected_structured_source": (
                selection["selected_experiment"] if experiment_id == "C01" else ""
            ),
            "engine": portable(engine),
            "engine_sha256": sha256(engine),
            "engine_size_bytes": engine.stat().st_size,
            "mean_ms": benchmark_result["mean_ms"],
            "median_ms": benchmark_result["median_ms"],
            "standard_deviation_ms": benchmark_result.get("standard_deviation_ms"),
            "iqr_ms": benchmark_result.get("iqr_ms"),
            "p95_ms": benchmark_result["p95_ms"],
            "p99_ms": benchmark_result.get("p99_ms"),
            "coefficient_of_variation": benchmark_result.get(
                "coefficient_of_variation"
            ),
            "fps": benchmark_result["fps"],
            "benchmark_image_count": benchmark_result.get("image_count"),
            "benchmark_scope": benchmark_result.get("benchmark_scope"),
            "gpu_peak_allocated_bytes": benchmark_result.get("gpu_peak_allocated_bytes"),
            "gpu_peak_reserved_bytes": benchmark_result.get("gpu_peak_reserved_bytes"),
            "bbox_ap": metric(evaluation, "bbox_ap"),
            "mask_ap": metric(evaluation, "mask_ap"),
            "semantic_miou": metric(evaluation, "semantic_miou"),
        }
        if baseline is not None:
            row.update(
                {
                    "bbox_ap_delta_vs_B01": row["bbox_ap"] - metric(baseline, "bbox_ap"),
                    "mask_ap_delta_vs_B01": row["mask_ap"] - metric(baseline, "mask_ap"),
                    "semantic_miou_delta_vs_B01": row["semantic_miou"]
                    - metric(baseline, "semantic_miou"),
                }
            )
        rows.append(row)

    output_json = REPOSITORY_ROOT / "results/desktop-engine-summary.json"
    write_json_atomic(
        output_json,
        {
            "created_at_utc": utc_now(),
            "structured_selection": selection,
            "rows": rows,
        },
    )
    output_csv = REPOSITORY_ROOT / "results/desktop-engine-summary.csv"
    if rows:
        with output_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-interval", type=float, default=30.0)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--runs", type=int)
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/training/front_session_split_v1/test"),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    defaults = read_defaults()
    evaluation_protocol = defaults["evaluation_protocol"]
    latency_protocol = evaluation_protocol["latency"]
    structured_policy = defaults["selection"]["structured_comparison"]
    warmup = args.warmup if args.warmup is not None else int(latency_protocol["warmup_runs"])
    runs = args.runs if args.runs is not None else int(latency_protocol["measured_runs"])
    sample_count = (
        args.sample_count
        if args.sample_count is not None
        else int(latency_protocol["sampled_images"])
    )
    seed = args.seed if args.seed is not None else int(latency_protocol["sample_seed"])
    benchmark_threshold = float(latency_protocol["postprocess_confidence_threshold"])
    if args.poll_interval <= 0 or warmup < 0 or runs < 1 or sample_count < 1:
        raise ValueError("Invalid polling or benchmark count")
    dataset = resolve(args.dataset_dir)
    state_path = REPOSITORY_ROOT / "results/desktop-tensorrt-pipeline.json"
    lock_path = REPOSITORY_ROOT / "results/desktop-tensorrt-pipeline.lock"
    log_dir = REPOSITORY_ROOT / "results/desktop-pipeline-logs"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"desktop queue already active: {lock_path}")
            return 0

        state: dict[str, Any] = {
            "created_at_utc": utc_now(),
            "updated_at_utc": utc_now(),
            "pid": os.getpid(),
            "status": "starting",
            "current_stage": None,
            "stages": {},
            "errors": [],
            "protocol_source": portable(DEFAULTS_PATH),
            "evaluation_protocol": evaluation_protocol,
        }
        write_json_atomic(state_path, state)

        try:
            wait_for_recovery(state, state_path, args.poll_interval)
            preflight = [
                str(PYTHON),
                "-c",
                (
                    "import tensorrt as trt, torch; "
                    "assert torch.cuda.is_available(); "
                    "print(trt.__version__, torch.cuda.get_device_name(0))"
                ),
            ]
            if not run_stage(state, state_path, "preflight", preflight, log_dir):
                raise RuntimeError("TensorRT/CUDA preflight failed")

            built = {
                experiment_id: build_registered(
                    experiment_id, state, state_path, log_dir, args.force
                )
                for experiment_id in PRE_SELECTION_ENGINES
            }
            run_stage(
                state,
                state_path,
                "static-onnx-M02",
                [
                    str(PYTHON),
                    "scripts/experiments/analyze_candidate.py",
                    "M02",
                    "--camera",
                    "front",
                    "--compare-to",
                    "B01",
                ],
                log_dir,
            )

            benchmarks: dict[str, dict[str, Any]] = {}
            for experiment_id in ("S01", "S02"):
                result = benchmark(
                    experiment_id,
                    dataset,
                    sample_count,
                    seed,
                    benchmark_threshold,
                    warmup,
                    runs,
                    state,
                    state_path,
                    log_dir,
                )
                if result is not None:
                    benchmarks[experiment_id] = result

            selection_path = REPOSITORY_ROOT / "results/desktop-structured-selection.json"
            if all(item in benchmarks for item in ("S01", "S02")):
                selection = choose_structured(
                    read_json(resolve(PTH_RESULTS["S01"])),
                    read_json(resolve(PTH_RESULTS["S02"])),
                    benchmarks["S01"],
                    benchmarks["S02"],
                    structured_policy,
                )
            else:
                selection = {
                    "created_at_utc": utc_now(),
                    "selected_experiment": "S01",
                    "policy": "S01 fallback because S01/S02 benchmark pair was incomplete.",
                }
            write_json_atomic(selection_path, selection)
            state["structured_selection"] = selection
            write_json_atomic(state_path, state)
            built["C01"] = build_selected_c01(
                selection["selected_experiment"],
                state,
                state_path,
                log_dir,
                args.force,
            )

            engine_static_analysis = {
                experiment_id: analyze_engine(
                    experiment_id, state, state_path, log_dir
                )
                for experiment_id in BACKENDS
                if built.get(experiment_id, False)
            }

            for experiment_id in BACKENDS:
                if experiment_id in benchmarks or not built.get(experiment_id, False):
                    continue
                result = benchmark(
                    experiment_id,
                    dataset,
                    sample_count,
                    seed,
                    benchmark_threshold,
                    warmup,
                    runs,
                    state,
                    state_path,
                    log_dir,
                )
                if result is not None:
                    benchmarks[experiment_id] = result

            evaluations: dict[str, dict[str, Any]] = {}
            for experiment_id in BACKENDS:
                if not built.get(experiment_id, False):
                    continue
                result = evaluate(
                    experiment_id,
                    dataset,
                    state,
                    state_path,
                    log_dir,
                    float(evaluation_protocol["coco_ap_confidence_threshold"]),
                    float(evaluation_protocol["semantic_miou_confidence_threshold"]),
                )
                if result is not None:
                    evaluations[experiment_id] = result

            write_summary(selection, benchmarks, evaluations)
            run_stage(
                state,
                state_path,
                "paper-results",
                [str(PYTHON), "scripts/reporting/paper_results.py"],
                log_dir,
            )
            run_stage(
                state,
                state_path,
                "paper-static-report",
                [
                    str(PYTHON),
                    "scripts/reporting/generate_paper_static_analysis.py",
                ],
                log_dir,
            )
            run_stage(
                state,
                state_path,
                "evaluation-automation-audit",
                [
                    str(PYTHON),
                    "scripts/experiments/audit_evaluation_protocol.py",
                ],
                log_dir,
            )
            state["status"] = (
                "completed-with-errors" if state.get("errors") else "completed"
            )
            state["current_stage"] = None
            state["completed_at_utc"] = utc_now()
            state["updated_at_utc"] = utc_now()
            state["built"] = built
            state["benchmark_count"] = len(benchmarks)
            state["evaluation_count"] = len(evaluations)
            state["engine_static_analysis"] = engine_static_analysis
            write_json_atomic(state_path, state)
            return 0 if not state.get("errors") else 1
        except BaseException as error:
            state["status"] = "failed"
            state["current_stage"] = None
            state["error"] = f"{type(error).__name__}: {error}"
            state["updated_at_utc"] = utc_now()
            write_json_atomic(state_path, state)
            raise


if __name__ == "__main__":
    raise SystemExit(main())
