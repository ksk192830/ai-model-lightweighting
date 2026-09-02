"""Regression tests for the all-candidate Stage-1 static gate."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import yaml


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/reporting/generate_stage1_static_evaluation.py"
SPEC = importlib.util.spec_from_file_location("stage1_static", MODULE_PATH)
assert SPEC and SPEC.loader
stage1 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = stage1
SPEC.loader.exec_module(stage1)


POLICY = {
    "onnx_size_min_relative_reduction": 0.05,
    "onnx_nodes_min_relative_reduction": 0.05,
    "dense_macs_min_relative_reduction": 0.05,
}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_qdq_graph_passes_without_accuracy_result(tmp_path: Path) -> None:
    directory = tmp_path / "artifacts/experiments/Q01/front"
    directory.mkdir(parents=True)
    (directory / "model.onnx").write_bytes(b"portable-qdq-onnx")
    write_json(
        directory / "static-analysis.json",
        {
            "onnx": {
                "onnx_checker_valid": True,
                "onnx_size_bytes": 100,
                "onnx_nodes": 20,
                "estimated_macs": 10,
                "estimated_flops": 20,
                "qdq_nodes": 12,
            }
        },
    )
    write_json(
        directory / "quantization.json",
        {"onnx_qdq_nodes": 12, "quantizers_total": 20, "quantizers_active": 10},
    )
    experiment = {
        "method": "modelopt-int8-smoothquant",
        "stage": "quantization",
        "artifact_source": "B01",
        "quantization": {"config": "INT8_SMOOTHQUANT_CFG"},
    }

    result = stage1.audit_candidate(tmp_path, "Q01", experiment, POLICY)

    assert result["stage1_evaluated"] is True
    assert result["stage1_passed"] is True
    assert "accuracy" in result["excluded_from_stage1_gate"]


def test_dense_graph_without_five_percent_effect_is_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "artifacts/experiments/U01/front"
    directory.mkdir(parents=True)
    (directory / "model.onnx").write_bytes(b"dense-onnx")
    write_json(
        directory / "static-analysis.json",
        {
            "onnx": {
                "onnx_checker_valid": True,
                "onnx_size_bytes": 100,
                "onnx_nodes": 20,
                "estimated_macs": 10,
                "estimated_flops": 20,
            }
        },
    )
    write_json(
        directory / "comparison-B01.json",
        {
            name: {"delta_ratio": 0.0}
            for name in ("onnx_size_bytes", "onnx_nodes", "estimated_macs")
        },
    )
    experiment = {
        "method": "global-magnitude",
        "stage": "static-analysis",
        "pruning": {"sparsity": 0.1},
    }

    result = stage1.audit_candidate(tmp_path, "U01", experiment, POLICY)

    assert result["stage1_evaluated"] is True
    assert result["stage1_passed"] is False
    assert "5%" in result["failure_reason"]


def test_markdown_does_not_hide_unperformed_candidates() -> None:
    markdown = stage1.render_markdown(
        [
            {
                "experiment_id": "A01",
                "method_summary": "control",
                "stage1_evaluated": True,
                "stage1_passed": True,
                "failure_reason": "",
            },
            {
                "experiment_id": "A02",
                "method_summary": "pending",
                "stage1_evaluated": False,
                "stage1_passed": False,
                "failure_reason": "not run",
            },
        ]
    )

    assert "미수행 1개" in markdown


def test_repository_registry_and_stage1_report_exclude_w_family() -> None:
    root = MODULE_PATH.parents[2]
    registry = yaml.safe_load(
        (root / "configs/experiments/registry.yaml").read_text(encoding="utf-8")
    )["experiments"]
    report = json.loads(
        (root / "results/stage1-static-evaluation.json").read_text(encoding="utf-8")
    )

    assert len(registry) == 26
    assert not any(experiment_id.startswith("W") for experiment_id in registry)
    assert report["candidate_count"] == 26
    assert not any(
        row["experiment_id"].startswith("W") for row in report["rows"]
    )
