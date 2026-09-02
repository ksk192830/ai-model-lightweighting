"""Tests for the round-2 paper table generator."""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import sys
from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPOSITORY_ROOT / "scripts/reporting/generate_round2_candidate_summary.py"
SPEC = importlib.util.spec_from_file_location("round2_summary", MODULE_PATH)
assert SPEC and SPEC.loader
summary = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = summary
SPEC.loader.exec_module(summary)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def evaluation(*, value: float, source: str = "PTH", resolution: int = 504) -> dict:
    payload = {
        "dataset": "data/test",
        "image_count": 10,
        "threshold": 0.001,
        "miou_threshold": 0.25,
        "resolution": resolution,
        "metrics": {
            "bbox": {"ap": value},
            "segm": {"ap": value - 0.1},
            "semantic_miou": value + 0.05,
        },
    }
    if source == "ONNX":
        payload.update(
            {
                "onnx": "model.onnx",
                "provider": "CPUExecutionProvider",
                "postprocess_num_select": 200,
                "pytorch_parity": {"passed": True},
            }
        )
    else:
        payload["checkpoint"] = "model.pth"
    return payload


def build_fixture(root: Path) -> None:
    experiments = {}
    for experiment_id in summary.EXPERIMENT_IDS:
        experiment = {
            "method": "none" if experiment_id == "B01" else "test-method",
            "status": "onnx-exported",
            "input_shape": [1, 3, 504, 504],
            "fine_tuning": {"required": False},
        }
        if experiment_id in {"S02", "M01"}:
            experiment["fine_tuning"] = {
                "required": True,
                "completed": False,
                "status": "ready",
            }
        if experiment_id == "S03":
            experiment["status"] = "static-analysis-rejected"
            experiment["fine_tuning"] = {
                "required": True,
                "completed": False,
                "status": "planned",
            }
        experiments[experiment_id] = experiment
    registry = {"experiments": experiments}
    registry_path = root / "configs/experiments/registry.yaml"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(yaml.safe_dump(registry), encoding="utf-8")

    baseline_evaluation = evaluation(value=0.7)
    write_json(root / summary.DEFAULT_BASELINE_EVALUATION, baseline_evaluation)

    baseline_values = {
        "onnx_size_bytes": 1000,
        "onnx_nodes": 100,
        "onnx_initializer_parameters": 10000,
        "estimated_macs": 20000,
        "estimated_flops": 40000,
    }
    for index, experiment_id in enumerate(summary.EXPERIMENT_IDS):
        onnx_values = {
            key: value if experiment_id == "B01" else value - index
            for key, value in baseline_values.items()
        }
        onnx_values.update(
            {
                "onnx_checker_valid": True,
                "inputs": {"input": [1, 3, 504, 504]},
                "outputs": {"dets": [1, 200, 4]},
            }
        )
        directory = root / f"artifacts/experiments/{experiment_id}/front"
        write_json(
            directory / "static-analysis.json",
            {"experiment_id": experiment_id, "onnx": onnx_values},
        )
        if experiment_id != "B01":
            comparison = {
                "baseline_experiment_id": "B01",
                "candidate_experiment_id": experiment_id,
            }
            for _, source_name in summary.STRUCTURAL_FIELDS:
                candidate = onnx_values[source_name]
                baseline = baseline_values[source_name]
                comparison[source_name] = {
                    "baseline": baseline,
                    "candidate": candidate,
                    "delta": candidate - baseline,
                    "delta_ratio": (candidate - baseline) / baseline,
                }
            write_json(directory / "comparison-B01.json", comparison)

    for experiment_id in ("U02", "R01", "S01"):
        path = root / f"results/coco-evaluation/{experiment_id}-front-pth.json"
        write_json(path, evaluation(value=0.69))
        experiments[experiment_id]["result"] = {
            "evaluation": path.relative_to(root).as_posix(),
            "evaluation_sha256": summary.sha256(path),
        }

    # These diagnostics must be audited but never become final accuracy while
    # recovery is unfinished.
    for experiment_id in ("S02", "M01"):
        write_json(
            root / f"results/coco-evaluation/{experiment_id}-front-before-recovery.json",
            evaluation(value=0.2),
        )

    # A valid ONNX alternative and the historic bad num_select=300 result.
    write_json(
        root / "results/coco-evaluation/S01-front-after-recovery-onnx.json",
        evaluation(value=0.691, source="ONNX"),
    )
    invalid = evaluation(value=0.99, source="ONNX")
    invalid["postprocess_num_select"] = 300
    invalid["pytorch_parity"] = {"passed": False}
    write_json(
        root / "results/coco-evaluation/S01-front-after-recovery-onnx-invalid-numselect300.json",
        invalid,
    )

    registry_path.write_text(yaml.safe_dump(registry), encoding="utf-8")


def test_pending_recovery_and_invalid_onnx_are_never_selected(tmp_path: Path) -> None:
    build_fixture(tmp_path)

    rows, audits = summary.generate(tmp_path)
    by_id = {row["experiment_id"]: row for row in rows}

    assert by_id["S01"]["accuracy_source"] == "PTH"
    assert by_id["S01"]["bbox_ap"] == 0.69
    assert by_id["S02"]["accuracy_status"] == "pending"
    assert by_id["S02"]["bbox_ap"] is None
    assert by_id["M01"]["accuracy_status"] == "pending"
    invalid = next(audit for audit in audits if "invalid-numselect300" in audit.path.name)
    assert invalid.disposition == "rejected"
    assert "postprocess_num_select=300" in invalid.reason
    assert "PyTorch parity failed" in invalid.reason


def test_csv_contains_exact_baseline_deltas_and_na(tmp_path: Path) -> None:
    build_fixture(tmp_path)

    rows, _ = summary.generate(tmp_path)
    parsed = list(csv.DictReader(io.StringIO(summary.render_csv(rows))))
    by_id = {row["experiment_id"]: row for row in parsed}

    assert by_id["B01"]["onnx_params_delta_abs"] == "0"
    assert by_id["B01"]["bbox_ap_delta_rel_pct"] == "0"
    assert by_id["U02"]["onnx_params_delta_abs"] == str(
        -summary.EXPERIMENT_IDS.index("U02")
    )
    assert by_id["B01"]["candidate_decision"] == "control"
    assert by_id["S02"]["bbox_ap"] == "N/A"
    assert by_id["S02"]["accuracy_source"] == "N/A"


def test_markdown_records_selected_superseded_and_rejected(tmp_path: Path) -> None:
    build_fixture(tmp_path)
    before = tmp_path / "results/coco-evaluation/S01-front-before-recovery.json"
    write_json(before, evaluation(value=0.1))
    registry_path = tmp_path / "configs/experiments/registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    registry["experiments"]["S01"]["fine_tuning"] = {
        "required": True,
        "completed": True,
    }
    registry_path.write_text(yaml.safe_dump(registry), encoding="utf-8")

    rows, audits = summary.generate(tmp_path)
    markdown = summary.render_markdown(rows, audits, tmp_path)

    assert "S01-front-pth.json | PTH | selected" in markdown
    assert "S01-front-before-recovery.json | PTH | superseded" in markdown
    assert "invalid-numselect300.json | ONNX | rejected" in markdown
