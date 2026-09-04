"""Stage-3 paper evidence generation contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts/reporting/generate_stage3_paper_assets.py"
    spec = importlib.util.spec_from_file_location("generate_stage3_paper_assets", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def result_row(
    experiment_id: str,
    *,
    mask_ap: float,
    median_ms: float,
    engine_size_bytes: int,
    gate: bool = True,
) -> dict:
    return {
        "experiment_id": experiment_id,
        "bbox_ap": 0.70 if gate else 0.50,
        "mask_ap": mask_ap,
        "semantic_miou": 0.70 if gate else 0.50,
        "bbox_ap_delta_vs_B01": 0.0 if gate else -0.20,
        "mask_ap_delta_vs_B01": mask_ap - 0.60,
        "semantic_miou_delta_vs_B01": 0.0 if gate else -0.20,
        "median_ms": median_ms,
        "p95_ms": median_ms + 2.0,
        "replicate_median_cv": 0.01,
        "engine_size_bytes": engine_size_bytes,
        "accuracy_gate_pass": gate,
        "realtime_30fps_pass": median_ms <= 1000 / 30,
        "p95_latency_warning": median_ms + 2.0 > 1000 / 30,
        "pareto_optimal": experiment_id in {"C01", "R01"},
    }


def test_build_decisions_distinguishes_not_applicable_failed_and_dominated() -> None:
    module = load_module()
    stage1 = {
        "rows": [
            {"experiment_id": "U01", "method_summary": "pruning", "stage2_notebook_eligible": False, "failure_reason": "static gate"},
            {"experiment_id": "Q03", "method_summary": "int4", "stage2_notebook_eligible": True, "failure_reason": ""},
            {"experiment_id": "B01", "method_summary": "none", "stage2_notebook_eligible": True, "failure_reason": ""},
            {"experiment_id": "C01", "method_summary": "combined", "stage2_notebook_eligible": True, "failure_reason": ""},
            {"experiment_id": "R01", "method_summary": "resolution", "stage2_notebook_eligible": True, "failure_reason": ""},
        ]
    }
    state = {
        "candidates": {
            "Q03": {"status": "build-failed", "reason": "parser error"},
            "B01": {"status": "completed"},
            "C01": {"status": "completed"},
            "R01": {"status": "completed"},
        }
    }
    stage3 = {
        "pareto_candidate_ids": ["C01", "R01"],
        "deployment_candidate_ids": ["C01", "R01"],
        "rows": [
            result_row("B01", mask_ap=0.60, median_ms=45.0, engine_size_bytes=130),
            result_row("C01", mask_ap=0.601, median_ms=25.0, engine_size_bytes=65),
            result_row("R01", mask_ap=0.594, median_ms=24.0, engine_size_bytes=68),
        ],
    }
    rows = {row["experiment_id"]: row for row in module.build_decisions(stage1, state, stage3)}
    assert rows["U01"]["stage2_status"] == "대상 외"
    assert rows["Q03"]["stage2_status"] == "엔진 생성 실패"
    assert rows["B01"]["dominated_by"] == "C01"
    assert rows["C01"]["deployment_candidate"] is True
    assert rows["R01"]["pareto_optimal"] is True
