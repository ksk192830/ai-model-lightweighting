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


def load_config() -> dict:
    return {
        "enabled": True,
        "required_platform_profile": "performance",
        "required_cpu_energy_performance_preference": "performance",
        "sample_interval_seconds": 0,
        "required_consecutive_samples": 3,
        "timeout_seconds": 20,
        "max_cpu_utilization_percent": 20.0,
        "max_gpu_utilization_percent": 40.0,
        "max_gpu_memory_utilization_percent": 25.0,
        "max_gpu_temperature_c": 65.0,
        "max_cpu_utilization_span_percent": 10.0,
        "max_gpu_utilization_span_percent": 8.0,
        "max_gpu_memory_utilization_span_percent": 5.0,
        "max_gpu_temperature_span_c": 3.0,
        "max_cpu_utilization_delta_from_reference_percent": 10.0,
        "max_gpu_utilization_delta_from_reference_percent": 8.0,
        "max_gpu_memory_utilization_delta_from_reference_percent": 5.0,
        "max_gpu_temperature_delta_from_reference_c": 5.0,
        "require_ac_power": True,
        "recorded_sample_limit": 10,
    }


def load_sample(*, cpu: float = 2.0, gpu: float = 0.0, temperature: float = 45.0) -> dict:
    return {
        "timestamp_utc": "2026-09-04T00:00:00+00:00",
        "cpu_utilization_percent": cpu,
        "system_memory_utilization_percent": 20.0,
        "load_average_1m": 0.1,
        "ac_power_connected": True,
        "gpu_utilization_percent": gpu,
        "gpu_memory_utilization_percent": 0.0,
        "gpu_temperature_c": temperature,
    }


def test_load_gate_resets_after_busy_sample_then_accepts_stable_window() -> None:
    samples = iter(
        [
            load_sample(),
            load_sample(cpu=35.0),
            load_sample(cpu=2.0, temperature=45.0),
            load_sample(cpu=3.0, temperature=46.0),
            load_sample(cpu=2.5, temperature=45.0),
        ]
    )
    clock = iter(float(value) for value in range(20))
    result = stage2.wait_for_stable_load(
        load_config(),
        sample_fn=lambda _interval: next(samples),
        monotonic_fn=lambda: next(clock),
    )
    assert result["status"] == "stable"
    assert result["total_samples"] == 5
    assert result["accepted_window_mean"]["cpu_utilization_percent"] == pytest.approx(2.5)


def test_load_gate_rejects_temperature_far_from_session_reference() -> None:
    reasons = stage2.assess_load_window(
        [load_sample(temperature=55.0) for _ in range(3)],
        load_config(),
        {"gpu_temperature_c": 45.0},
    )
    assert any("session reference" in reason for reason in reasons)


def test_terminal_progress_line_accepts_current_repetition(capsys: pytest.CaptureFixture) -> None:
    state = {
        "progress": {
            "completed_units": 1,
            "total_units": 63,
            "percent": 100 / 63,
            "current_candidate": "B01",
            "current_stage": "latency benchmark",
            "current_repetition": 2,
            "elapsed_seconds": 12.0,
            "estimated_finish_at_utc": None,
        }
    }
    stage2.print_progress(state)
    output = capsys.readouterr().out
    assert "1/63" in output
    assert "B01 latency benchmark repeat 2" in output


def test_measurement_environment_requires_registered_performance_policy() -> None:
    environment = {
        "platform_profile": "performance",
        "cpu_energy_performance_preference": "performance",
        "ac_power_connected": True,
    }
    stage2.validate_measurement_environment(environment, load_config())

    environment["platform_profile"] = "quiet"
    with pytest.raises(RuntimeError, match="platform_profile"):
        stage2.validate_measurement_environment(environment, load_config())


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
