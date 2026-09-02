#!/usr/bin/env python3
"""Audit every registered candidate before notebook performance evaluation.

Stage 1 is deliberately static.  It verifies that each candidate has a valid,
reproducible deployment artifact or recipe and that the requested
transformation is visible in that artifact.  Accuracy, latency, FPS, and GPU
memory are not Stage-1 gates; those belong to the notebook Stage-2 protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = Path("configs/experiments/registry.yaml")
DEFAULTS = Path("configs/experiments/defaults.yaml")
ARTIFACT_ROOT = Path("artifacts/experiments")
OUTPUT_JSON = Path("results/stage1-static-evaluation.json")
OUTPUT_CSV = Path("results/stage1-static-evaluation.csv")
OUTPUT_MD = Path("results/stage1-static-evaluation.md")
QDQ_TYPES = (
    "QuantizeLinear",
    "DequantizeLinear",
    "TRT_FP8QuantizeLinear",
    "TRT_FP8DequantizeLinear",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def portable(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reduction(comparison: dict[str, Any], name: str) -> float | None:
    block = comparison.get(name)
    if not isinstance(block, dict):
        return None
    ratio = block.get("delta_ratio")
    if not isinstance(ratio, (int, float)):
        return None
    return -float(ratio)


def method_summary(experiment: dict[str, Any]) -> str:
    method = str(experiment["method"])
    details: list[str] = []
    if isinstance(experiment.get("pruning"), dict):
        details.extend(
            f"{key}={value}" for key, value in experiment["pruning"].items()
        )
    shape = experiment.get("input_shape")
    if isinstance(shape, list) and len(shape) == 4:
        details.append(f"input={shape[2]}x{shape[3]}")
    if isinstance(experiment.get("quantization"), dict):
        config = experiment["quantization"].get("config")
        configs = experiment["quantization"].get("configs")
        if config:
            details.append(str(config))
        if configs:
            details.append(f"{len(configs)} bit-width configs")
    return method + ("; " + ", ".join(details) if details else "")


def check(name: str, passed: bool, evidence: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def audit_candidate(
    root: Path,
    experiment_id: str,
    experiment: dict[str, Any],
    efficiency_policy: dict[str, Any],
    *,
    check_mode: bool = False,
) -> dict[str, Any]:
    directory = root / ARTIFACT_ROOT / experiment_id / "front"
    static_path = directory / "static-analysis.json"
    comparison_path = directory / "comparison-B01.json"
    own_onnx = directory / "model.onnx"
    source_id = str(experiment.get("artifact_source", experiment_id))
    source_onnx = root / ARTIFACT_ROOT / source_id / "front/model.onnx"
    build_command = directory / "build-command.txt"
    stage = str(experiment["stage"])
    checks: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    deployment_artifact = own_onnx if own_onnx.is_file() else source_onnx
    artifact_kind = "onnx"

    static_present = static_path.is_file()
    checks.append(
        check("static-report-present", static_present, portable(root, static_path))
    )
    static = load_json(static_path) if static_present else {}
    onnx = static.get("onnx", {}) if isinstance(static.get("onnx"), dict) else {}
    onnx_valid = onnx.get("onnx_checker_valid") is True
    checks.append(check("onnx-checker", onnx_valid, "onnx.checker.check_model"))
    metrics.update(
        {
            "onnx_size_bytes": onnx.get("onnx_size_bytes"),
            "onnx_nodes": onnx.get("onnx_nodes"),
            "estimated_macs": onnx.get("estimated_macs"),
            "estimated_flops": onnx.get("estimated_flops"),
            "qdq_nodes": onnx.get("qdq_nodes", 0),
        }
    )

    if experiment_id.startswith("Q"):
        artifact_kind = "qdq-onnx"
        quant_path = directory / "quantization.json"
        quant_present = quant_path.is_file()
        checks.append(
            check(
                "quantization-report-present",
                quant_present,
                portable(root, quant_path),
            )
        )
        qdq_nodes = int(onnx.get("qdq_nodes", 0) or 0)
        if quant_present:
            quant = load_json(quant_path)
            qdq_nodes = int(quant.get("onnx_qdq_nodes", qdq_nodes) or 0)
            metrics.update(
                {
                    "quantizers_total": quant.get("quantizers_total"),
                    "quantizers_active": quant.get("quantizers_active"),
                    "qdq_nodes": qdq_nodes,
                }
            )
        checks.append(
            check(
                "quantization-visible-in-graph",
                qdq_nodes > 0,
                f"Q/DQ nodes={qdq_nodes}",
            )
        )
        passed = all(item["passed"] for item in checks)
        evaluated = True
        reason = "" if passed else "유효한 Q/DQ ONNX 또는 양자화 적용 근거가 없음"
    elif stage == "tensorrt":
        artifact_kind = "tensorrt-build-recipe"
        source_present = source_onnx.is_file()
        recipe_present = build_command.is_file() and bool(
            build_command.read_text(encoding="utf-8").strip()
        )
        checks.append(
            check("source-onnx-present", source_present, portable(root, source_onnx))
        )
        checks.append(
            check(
                "deterministic-build-recipe",
                recipe_present,
                portable(root, build_command),
            )
        )
        passed = all(item["passed"] for item in checks)
        evaluated = True
        reason = "" if passed else "원본 ONNX 또는 재현 가능한 TensorRT build recipe가 없음"
    else:
        artifact_kind = "onnx-graph"
        own_present = own_onnx.is_file()
        checks.append(
            check("candidate-onnx-present", own_present, portable(root, own_onnx))
        )
        if experiment_id == "B01":
            transformation_visible = True
            transform_evidence = "baseline control"
        elif experiment_id == "M01":
            sparse_path = directory / "onnx-2to4.json"
            sparse = load_json(sparse_path) if sparse_path.is_file() else {}
            transformation_visible = sparse.get("valid") is True
            transform_evidence = (
                f"2:4 compliant ops={sparse.get('compliant_ops', 0)}/"
                f"{sparse.get('eligible_ops_by_shape', 0)}"
            )
        else:
            comparison = load_json(comparison_path) if comparison_path.is_file() else {}
            reductions = {
                "onnx_size": reduction(comparison, "onnx_size_bytes"),
                "onnx_nodes": reduction(comparison, "onnx_nodes"),
                "dense_macs": reduction(comparison, "estimated_macs"),
            }
            thresholds = {
                "onnx_size": float(
                    efficiency_policy["onnx_size_min_relative_reduction"]
                ),
                "onnx_nodes": float(
                    efficiency_policy["onnx_nodes_min_relative_reduction"]
                ),
                "dense_macs": float(
                    efficiency_policy["dense_macs_min_relative_reduction"]
                ),
            }
            transformation_visible = any(
                reductions[name] is not None and reductions[name] >= thresholds[name]
                for name in thresholds
            )
            metrics["relative_reductions"] = reductions
            transform_evidence = ", ".join(
                f"{name}={100 * value:.2f}%" if value is not None else f"{name}=N/A"
                for name, value in reductions.items()
            )
        checks.append(
            check("static-effect-visible", transformation_visible, transform_evidence)
        )
        passed = all(item["passed"] for item in checks)
        evaluated = True
        if passed:
            reason = ""
        elif not static_present or not onnx_valid or not own_present:
            reason = "후보 자체의 유효한 ONNX 산출물이 없음"
        else:
            reason = (
                "ONNX 크기·node·dense MACs 중 어느 항목도 사전 고정 "
                "5% 감소 기준을 충족하지 못함"
            )

    candidate_report = directory / "stage1-static-evaluation.json"
    existing_created_at = None
    if check_mode and candidate_report.is_file():
        existing_created_at = load_json(candidate_report).get("created_at_utc")
    report = {
        "created_at_utc": existing_created_at or datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "method_summary": method_summary(experiment),
        "artifact_kind": artifact_kind,
        "stage1_evaluated": evaluated,
        "stage1_passed": passed,
        "stage2_notebook_eligible": passed,
        "failure_reason": reason,
        "checks": checks,
        "static_metrics": metrics,
        "deployment_onnx": (
            portable(root, deployment_artifact)
            if deployment_artifact.is_file()
            else None
        ),
        "deployment_onnx_sha256": (
            sha256(deployment_artifact)
            if deployment_artifact.is_file()
            else None
        ),
        "excluded_from_stage1_gate": ["accuracy", "latency", "fps", "gpu_memory"],
    }
    candidate_text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if check_mode:
        if not candidate_report.is_file() or candidate_report.read_text(encoding="utf-8") != candidate_text:
            raise SystemExit(f"out of date: {candidate_report}")
    else:
        directory.mkdir(parents=True, exist_ok=True)
        candidate_report.write_text(candidate_text, encoding="utf-8")
    report["report"] = portable(root, candidate_report)
    report["report_sha256"] = sha256(candidate_report)
    return report


def render_markdown(rows: list[dict[str, Any]]) -> str:
    passed = sum(bool(row["stage1_passed"]) for row in rows)
    evaluated = sum(bool(row["stage1_evaluated"]) for row in rows)
    unperformed = len(rows) - evaluated
    lines = [
        "# RF-DETR 1차 정적평가 전체 후보표",
        "",
        (
            f"등록된 {len(rows)}개 후보 중 {evaluated}개의 1차 평가를 완료했다. "
            f"통과 {passed}개, 불통 {evaluated - passed}개, 미수행 {unperformed}개다."
        ),
        "",
        "1차는 산출물 유효성, 그래프 구조, 정적 효율, 양자화/희소성 적용 여부만 판정한다. "
        "정확도, latency, FPS, GPU memory는 노트북 2차 평가에서만 판정한다.",
        "",
        "| ID | 경량화 방식 요약 | 1차 평가 통과 여부 | 불통 사유(통과하지 못한 후보만) |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {experiment_id} | {method_summary} | {status} | {reason} |".format(
                experiment_id=row["experiment_id"],
                method_summary=str(row["method_summary"]).replace("|", "\\|"),
                status="통과" if row["stage1_passed"] else "불통",
                reason=(row["failure_reason"] or "-").replace("|", "\\|"),
            )
        )
    lines.extend(
        [
            "",
            "## 후속 절차",
            "",
            "1. `stage2_notebook_eligible=true`인 후보를 노트북에서 TensorRT engine으로 빌드한다.",
            "2. 고정 test 437장으로 bbox AP, mask AP, semantic mIoU를 측정한다.",
            "3. warm-up 20회 뒤 200회 반복으로 median/p95 latency, FPS, peak GPU memory를 측정한다.",
            "4. 모든 2차 결과가 모인 뒤 정확도-지연시간-메모리-크기의 비지배해를 Pareto 분석한다.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    registry = load_yaml(root / REGISTRY)["experiments"]
    defaults = load_yaml(root / DEFAULTS)
    policy = defaults["selection"]["first_stage_efficiency"]
    rows = [
        audit_candidate(
            root,
            experiment_id,
            experiment,
            policy,
            check_mode=args.check,
        )
        for experiment_id, experiment in registry.items()
    ]
    existing_created_at = None
    if args.check and (root / OUTPUT_JSON).is_file():
        existing_created_at = load_json(root / OUTPUT_JSON).get("created_at_utc")
    payload = {
        "created_at_utc": existing_created_at or datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "stage1": "static artifact/recipe validation only",
            "stage2": "notebook accuracy and runtime performance",
            "stage3": "Pareto analysis after complete Stage-2 measurements",
            "accuracy_used_as_stage1_gate": False,
        },
        "candidate_count": len(rows),
        "evaluated_count": sum(bool(row["stage1_evaluated"]) for row in rows),
        "passed_count": sum(bool(row["stage1_passed"]) for row in rows),
        "unperformed_count": sum(not bool(row["stage1_evaluated"]) for row in rows),
        "rows": rows,
    }
    json_text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    markdown_text = render_markdown(rows)
    csv_fields = (
        "experiment_id",
        "method_summary",
        "artifact_kind",
        "stage1_evaluated",
        "stage1_passed",
        "stage2_notebook_eligible",
        "failure_reason",
        "report",
        "report_sha256",
    )

    destinations = {
        root / OUTPUT_JSON: json_text,
        root / OUTPUT_MD: markdown_text,
    }
    for path, content in destinations.items():
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                raise SystemExit(f"out of date: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=csv_fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows({field: row.get(field, "") for field in csv_fields} for row in rows)
    csv_text = csv_buffer.getvalue()
    csv_path = root / OUTPUT_CSV
    if args.check:
        if not csv_path.is_file() or csv_path.read_text(encoding="utf-8") != csv_text:
            raise SystemExit(f"out of date: {csv_path}")
    else:
        csv_path.write_text(csv_text, encoding="utf-8")
    print(
        f"stage1: evaluated={payload['evaluated_count']}/{payload['candidate_count']} "
        f"passed={payload['passed_count']} unperformed={payload['unperformed_count']}"
    )
    print(root / OUTPUT_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
