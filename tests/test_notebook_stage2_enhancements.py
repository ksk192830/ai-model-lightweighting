"""Unit tests for repeated notebook metrics and Pareto pre-gating."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module(relative: str, name: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage2 = load_module(
    "scripts/experiments/run_notebook_stage2.py", "run_notebook_stage2"
)
evaluator = load_module(
    "scripts/evaluation/evaluate_coco_tensorrt.py", "evaluate_coco_tensorrt"
)


def benchmark(timings: list[float]) -> dict:
    ordered = sorted(timings)
    return {
        "timings_ms": timings,
        "median_ms": sum(ordered[1:3]) / 2,
        "mean_ms": sum(timings) / len(timings),
        "measured_runs": len(timings),
        "gpu_peak_allocated_bytes": 100,
        "gpu_peak_reserved_bytes": 120,
    }


def test_three_repetition_aggregate_has_tail_and_uncertainty_metrics() -> None:
    result = stage2.aggregate_benchmarks(
        [
            benchmark([10.0, 11.0, 12.0, 13.0]),
            benchmark([20.0, 21.0, 22.0, 23.0]),
            benchmark([30.0, 31.0, 32.0, 33.0]),
        ]
    )
    assert result["benchmark_repetitions"] == 3
    assert result["total_measured_runs"] == 12
    assert result["median_ms"] == 21.5
    assert result["p99_ms"] > result["p95_ms"] > result["median_ms"]
    assert result["replicate_median_ci95_low_ms"] < 21.5
    assert result["replicate_median_ci95_high_ms"] > 21.5


def candidate(experiment_id: str, *, bbox: float, mask: float, miou: float, ms: float) -> dict:
    return {
        "experiment_id": experiment_id,
        "bbox_ap": bbox,
        "mask_ap": mask,
        "semantic_miou": miou,
        "median_ms": ms,
        "p95_ms": ms * 1.1,
        "replicate_median_cv": 0.01,
        "total_measured_runs": 600,
        "engine_size_bytes": 100 if experiment_id == "B01" else 60,
        "gpu_peak_allocated_bytes": 100 if experiment_id == "B01" else 80,
    }


def test_accuracy_gate_precedes_pareto() -> None:
    defaults = {
        "evaluation_protocol": {
            "latency": {"replicate_median_cv_warning_threshold": 0.05}
        },
        "selection": {
            "accuracy_vs_baseline": {
                "bbox_ap_max_absolute_drop": 0.01,
                "mask_ap_max_absolute_drop": 0.01,
                "semantic_miou_max_absolute_drop": 0.02,
            }
        },
    }
    rows = stage2.add_baseline_comparisons(
        [
            candidate("B01", bbox=0.70, mask=0.60, miou=0.72, ms=20.0),
            candidate("GOOD", bbox=0.695, mask=0.595, miou=0.71, ms=10.0),
            candidate("FAST_BAD", bbox=0.60, mask=0.50, miou=0.60, ms=5.0),
        ],
        defaults,
    )
    by_id = {row["experiment_id"]: row for row in rows}
    assert by_id["GOOD"]["accuracy_gate_pass"] is True
    assert by_id["GOOD"]["stage3_pareto_eligible"] is True
    assert by_id["FAST_BAD"]["accuracy_gate_pass"] is False
    assert by_id["FAST_BAD"]["stage3_pareto_eligible"] is False
    assert by_id["GOOD"]["median_latency_reduction_vs_B01_pct"] == 50.0


def test_per_category_coco_summary_uses_valid_precision_entries() -> None:
    precision = np.full((2, 3, 2, 1, 1), -1.0)
    recall = np.full((2, 2, 1, 1), -1.0)
    precision[:, :, 0, 0, 0] = 0.8
    precision[:, :, 1, 0, 0] = 0.4
    recall[:, 0, 0, 0] = 0.75
    recall[:, 1, 0, 0] = 0.35
    evaluation = SimpleNamespace(
        eval={"precision": precision, "recall": recall},
        params=SimpleNamespace(iouThrs=[0.50, 0.75], maxDets=[100]),
    )
    result = evaluator.summarize_coco_by_category(evaluation, [1, 3])
    assert result["1"] == pytest.approx(
        {"ap": 0.8, "ap50": 0.8, "ap75": 0.8, "ar100": 0.75}
    )
    assert result["3"] == pytest.approx(
        {"ap": 0.4, "ap50": 0.4, "ap75": 0.4, "ar100": 0.35}
    )
