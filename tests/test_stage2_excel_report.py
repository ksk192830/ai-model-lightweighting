"""Workbook generation contract for the notebook's single-file handoff."""

from __future__ import annotations

import importlib.util
import json
import shutil
import zipfile
from pathlib import Path

import pytest


pytest.importorskip("xlsxwriter")
ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts/reporting/generate_stage2_excel.py"
    spec = importlib.util.spec_from_file_location("generate_stage2_excel", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def result_row(experiment_id: str, value: float, benchmark: str, evaluation: str) -> dict:
    return {
        "experiment_id": experiment_id,
        "family": "baseline" if experiment_id == "B01" else "precision",
        "method": "none" if experiment_id == "B01" else "tensorrt-fp16",
        "precision": "fp32" if experiment_id == "B01" else "fp16",
        "input_shape": "1x3x504x504",
        "bbox_ap": value,
        "bbox_ap50": value,
        "bbox_ap75": value,
        "bbox_ap_small": value,
        "bbox_ap_medium": value,
        "bbox_ap_large": value,
        "bbox_ar100": value,
        "mask_ap": value,
        "mask_ap50": value,
        "mask_ap75": value,
        "mask_ap_small": value,
        "mask_ap_medium": value,
        "mask_ap_large": value,
        "mask_ar100": value,
        "semantic_miou": value,
        "mean_ms": 10.0,
        "median_ms": 9.5,
        "p95_ms": 11.0,
        "p99_ms": 12.0,
        "iqr_ms": 1.0,
        "coefficient_of_variation": 0.03,
        "replicate_median_cv": 0.01,
        "replicate_median_ci95_low_ms": 9.0,
        "replicate_median_ci95_high_ms": 10.0,
        "fps": 100.0,
        "gpu_peak_allocated_bytes": 104857600,
        "gpu_peak_reserved_bytes": 125829120,
        "engine_size_bytes": 52428800,
        "benchmark_repetitions": 1,
        "total_measured_runs": 200,
        "benchmarks": [benchmark],
        "evaluation": evaluation,
        "measurement_valid": True,
        "accuracy_gate_pass": True,
        "stage3_pareto_eligible": True,
        "realtime_30fps_pass": True,
        "p95_latency_warning": False,
        "latency_stability_warning": False,
        "bbox_ap_delta_vs_B01": 0.0,
        "mask_ap_delta_vs_B01": 0.0,
        "semantic_miou_delta_vs_B01": 0.0,
        "median_latency_reduction_vs_B01_pct": 0.0,
        "speedup_vs_B01": 1.0,
        "gpu_memory_reduction_vs_B01_pct": 0.0,
        "engine_size_reduction_vs_B01_pct": 0.0,
        "exclusion_reason": "",
        "engine": f"artifacts/{experiment_id}.engine",
        "engine_sha256": "a" * 64,
        "annotation_sha256": "b" * 64,
        "test_image_count": 437,
        "prediction_count": 1000,
    }


def test_excel_contains_results_raw_evidence_protocol_and_formulas(tmp_path: Path) -> None:
    module = load_module()
    (tmp_path / "configs/experiments").mkdir(parents=True)
    shutil.copy2(
        ROOT / "configs/experiments/defaults.yaml",
        tmp_path / "configs/experiments/defaults.yaml",
    )
    shutil.copy2(
        ROOT / "configs/experiments/registry.yaml",
        tmp_path / "configs/experiments/registry.yaml",
    )
    stage1_rows = [
        {"experiment_id": "B01", "stage2_notebook_eligible": True, "failure_reason": ""},
        {"experiment_id": "B02", "stage2_notebook_eligible": True, "failure_reason": ""},
        {"experiment_id": "U01", "stage2_notebook_eligible": False, "failure_reason": "static gate"},
    ]
    write_json(
        tmp_path / "results/stage1-static-evaluation.json",
        {"rows": stage1_rows},
    )
    rows = []
    for experiment_id, value in (("B01", 0.70), ("B02", 0.695)):
        benchmark = f"results/benchmarks/{experiment_id}.json"
        evaluation = f"results/evaluations/{experiment_id}.json"
        rows.append(result_row(experiment_id, value, benchmark, evaluation))
        write_json(
            tmp_path / benchmark,
            {
                "image_count": 32,
                "warmup_runs": 20,
                "measured_runs": 200,
                "mean_ms": 10.0,
                "median_ms": 9.5,
                "p95_ms": 11.0,
                "p99_ms": 12.0,
                "iqr_ms": 1.0,
                "coefficient_of_variation": 0.03,
                "fps": 100.0,
                "gpu_peak_allocated_bytes": 104857600,
                "gpu_peak_reserved_bytes": 125829120,
                "model_sha256": "a" * 64,
                "images": ["fixed.jpg"],
            },
        )
        per_class = {"1": {"ap": value, "ap50": value, "ap75": value, "ar100": value}}
        write_json(
            tmp_path / evaluation,
            {
                "categories": {"1": "parking_space"},
                "metrics": {
                    "bbox_by_category": per_class,
                    "segm_by_category": per_class,
                    "semantic_iou_by_category": {"1": value},
                },
            },
        )
    write_json(
        tmp_path / "results/stage2-notebook-summary.json",
        {"rows": rows, "failures": [], "preflight": {"test_images": 437}},
    )
    write_json(
        tmp_path / "results/stage2-notebook-state.json",
        {"candidates": {item: {"status": "completed"} for item in ("B01", "B02")}},
    )
    write_json(
        tmp_path / "results/stage3-pareto.json",
        {
            "objectives": {
                "maximize": ["mask_ap"],
                "minimize": ["median_ms", "engine_size_bytes"],
            },
            "pareto_candidate_ids": ["B02"],
            "deployment_candidate_ids": ["B02"],
        },
    )
    output = tmp_path / "results/stage2-evaluation-report.xlsx"
    module.build_report(tmp_path, output)
    assert output.is_file() and output.stat().st_size > 10000
    with zipfile.ZipFile(output) as archive:
        text = "\n".join(
            archive.read(name).decode("utf-8", errors="ignore")
            for name in archive.namelist()
            if name.endswith(".xml")
        )
    for label in (
        "요약",
        "분석결과",
        "원시결과",
        "클래스별 정확도",
        "반복측정",
        "실패·제외",
        "평가기준",
        "실험환경",
        "30 FPS 적합",
        "최종 배포 후보",
        "P95 예산 초과 경고",
    ):
        assert label in text
    assert "MATCH" in text
    assert "parking_space" in text
