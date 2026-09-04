"""Tests for the three-stage progress summary."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/reporting/show_project_status.py"
SPEC = importlib.util.spec_from_file_location("show_project_status", MODULE_PATH)
assert SPEC and SPEC.loader
status_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = status_module
SPEC.loader.exec_module(status_module)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_reports_unstarted_not_unperformed_after_stage1(tmp_path: Path) -> None:
    write_json(
        tmp_path / "results/stage1-static-evaluation.json",
        {
            "candidate_count": 3,
            "evaluated_count": 3,
            "passed_count": 2,
            "unperformed_count": 0,
            "rows": [
                {"experiment_id": "B01", "stage2_notebook_eligible": True},
                {"experiment_id": "Q01", "stage2_notebook_eligible": True},
                {"experiment_id": "U01", "stage2_notebook_eligible": False},
            ],
        },
    )

    status = status_module.collect_status(tmp_path)

    assert status["stage1"]["status"] == "COMPLETE"
    assert status["stage1"]["unperformed"] == 0
    assert status["stage2"]["status"] == "NOT_STARTED"
    assert status["stage2"]["pending"] == ["B01", "Q01"]


def test_stage2_failures_are_terminal_and_allow_pareto_status(tmp_path: Path) -> None:
    write_json(
        tmp_path / "results/stage1-static-evaluation.json",
        {
            "candidate_count": 2,
            "evaluated_count": 2,
            "passed_count": 2,
            "unperformed_count": 0,
            "rows": [
                {"experiment_id": "B01", "stage2_notebook_eligible": True},
                {"experiment_id": "Q03", "stage2_notebook_eligible": True},
            ],
        },
    )
    write_json(
        tmp_path / "results/stage2-notebook-state.json",
        {
            "candidates": {
                "B01": {"status": "completed"},
                "Q03": {"status": "build-failed"},
            }
        },
    )
    write_json(
        tmp_path / "results/stage3-pareto.json",
        {
            "objectives": {
                "maximize": ["mask_ap"],
                "minimize": ["median_ms", "engine_size_bytes"],
            },
            "pareto_candidate_ids": ["B01"],
            "deployment_candidate_ids": ["B01"],
        },
    )

    status = status_module.collect_status(tmp_path)

    assert status["stage2"]["status"] == "COMPLETE"
    assert status["stage2"]["terminal"] == 2
    assert status["stage2"]["failed"] == 1
    assert status["stage3"]["status"] == "COMPLETE"
    assert status["stage3"]["deployment_candidate_ids"] == ["B01"]


def test_old_pareto_is_marked_stale_while_stage2_is_incomplete(tmp_path: Path) -> None:
    write_json(
        tmp_path / "results/stage1-static-evaluation.json",
        {
            "candidate_count": 1,
            "evaluated_count": 1,
            "passed_count": 1,
            "unperformed_count": 0,
            "rows": [
                {"experiment_id": "B01", "stage2_notebook_eligible": True},
            ],
        },
    )
    write_json(
        tmp_path / "results/stage3-pareto.json",
        {"pareto_candidate_ids": ["old-result"]},
    )

    status = status_module.collect_status(tmp_path)

    assert status["stage2"]["status"] == "NOT_STARTED"
    assert status["stage3"]["status"] == "STALE"


def test_superseded_pareto_objectives_are_marked_stale(tmp_path: Path) -> None:
    write_json(
        tmp_path / "results/stage1-static-evaluation.json",
        {
            "candidate_count": 1,
            "evaluated_count": 1,
            "passed_count": 1,
            "unperformed_count": 0,
            "rows": [
                {"experiment_id": "B01", "stage2_notebook_eligible": True},
            ],
        },
    )
    write_json(
        tmp_path / "results/stage2-notebook-state.json",
        {"candidates": {"B01": {"status": "completed"}}},
    )
    write_json(
        tmp_path / "results/stage3-pareto.json",
        {
            "objectives": {
                "maximize": ["bbox_ap", "mask_ap", "semantic_miou"],
                "minimize": ["median_ms", "p95_ms"],
            },
            "pareto_candidate_ids": ["old-result"],
        },
    )

    status = status_module.collect_status(tmp_path)

    assert status["stage2"]["status"] == "COMPLETE"
    assert status["stage3"]["status"] == "STALE"
    assert status["stage3"]["objectives_current"] is False


def test_controlled_rerun_shows_repeat_progress_and_eta(tmp_path: Path) -> None:
    write_json(
        tmp_path / "results/stage1-static-evaluation.json",
        {
            "candidate_count": 1,
            "evaluated_count": 1,
            "passed_count": 1,
            "unperformed_count": 0,
            "rows": [
                {"experiment_id": "B01", "stage2_notebook_eligible": True},
            ],
        },
    )
    write_json(
        tmp_path / "results/stage2-notebook-state.json",
        {
            "status": "running",
            "measurement_mode": "controlled-latency-remeasurement",
            "candidates": {"B01": {"status": "running"}},
            "progress": {
                "completed_units": 1,
                "total_units": 3,
                "percent": 33.333,
                "current_candidate": "B01",
                "current_stage": "latency benchmark",
                "current_repetition": 2,
                "estimated_finish_at_utc": "2026-09-04T02:00:00+00:00",
            },
        },
    )

    rendered = status_module.render(status_module.collect_status(tmp_path))

    assert "1/3 (33.3%)" in rendered
    assert "B01 / latency benchmark / 반복 2" in rendered
    assert "예상 종료" in rendered
