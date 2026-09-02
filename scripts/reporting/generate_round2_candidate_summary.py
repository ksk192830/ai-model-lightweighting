#!/usr/bin/env python3
"""Generate the reproducible RF-DETR round-2 candidate summary.

The report is intentionally built from machine-readable experiment artifacts.  It
does not import or execute a model and is therefore safe to run while GPU work is
in progress.

Outputs:
  results/round2-candidate-summary.csv
  results/round2-candidate-summary.md

Use ``--check`` in CI to verify that committed reports match their sources.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_IDS = ("B01", "U02", "R01", "S01", "S02", "S03", "M01")
BASELINE_ID = "B01"
DEFAULT_BASELINE_EVALUATION = Path(
    "results/coco-evaluation/front-rfdetr-seg-large-v1-test.json"
)
INVALID_FILENAME_MARKERS = ("invalid", "superseded")

STRUCTURAL_FIELDS = (
    ("onnx_params", "onnx_initializer_parameters"),
    ("onnx_nodes", "onnx_nodes"),
    ("estimated_macs", "estimated_macs"),
    ("estimated_flops", "estimated_flops"),
    ("onnx_size_bytes", "onnx_size_bytes"),
)
ACCURACY_FIELDS = ("bbox_ap", "mask_ap", "semantic_miou")

CSV_COLUMNS = [
    "experiment_id",
    "method",
    "method_detail",
    "status",
    "resolution_px",
]
for output_name, _ in STRUCTURAL_FIELDS:
    CSV_COLUMNS.extend(
        [output_name, f"{output_name}_delta_abs", f"{output_name}_delta_rel_pct"]
    )
CSV_COLUMNS.extend(["bbox_ap", "bbox_ap_delta_abs", "bbox_ap_delta_rel_pct"])
CSV_COLUMNS.extend(["mask_ap", "mask_ap_delta_abs", "mask_ap_delta_rel_pct"])
CSV_COLUMNS.extend(
    [
        "semantic_miou",
        "semantic_miou_delta_abs",
        "semantic_miou_delta_rel_pct",
        "accuracy_source",
        "accuracy_status",
        "accuracy_result",
        "static_analysis",
        "comparison",
    ]
)


@dataclass(frozen=True)
class EvaluationAudit:
    path: Path
    experiment_id: str
    source: str
    disposition: str
    reason: str
    payload: dict[str, Any] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("results/round2-candidate-summary.csv"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("results/round2-candidate-summary.md"),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail instead of writing when generated files are out of date.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def load_registry(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    experiments = data.get("experiments") if isinstance(data, dict) else None
    if not isinstance(experiments, dict):
        raise ValueError(f"Missing experiments mapping: {path}")
    missing = [experiment_id for experiment_id in EXPERIMENT_IDS if experiment_id not in experiments]
    if missing:
        raise ValueError(f"Registry is missing round-2 candidates: {', '.join(missing)}")
    return experiments


def resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_number(value: Any, context: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Expected numeric {context}, got {value!r}")
    if not math.isfinite(float(value)):
        raise ValueError(f"Expected finite {context}, got {value!r}")
    return value


def nested_metric(payload: dict[str, Any], name: str) -> int | float:
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("missing metrics object")
    if name == "bbox_ap":
        value = metrics.get("bbox", {}).get("ap")
    elif name == "mask_ap":
        value = metrics.get("segm", {}).get("ap")
    else:
        value = metrics.get("semantic_miou")
    return require_number(value, name)


def infer_accuracy_source(payload: dict[str, Any]) -> str:
    if "onnx" in payload or "provider" in payload or "providers_active" in payload:
        return "ONNX"
    if "checkpoint" in payload:
        return "PTH"
    return "UNKNOWN"


def input_resolution(experiment: dict[str, Any], static: dict[str, Any]) -> int:
    shape = experiment.get("input_shape")
    if not isinstance(shape, list):
        shape = static.get("onnx", {}).get("inputs", {}).get("input")
    if not isinstance(shape, list) or len(shape) != 4:
        raise ValueError("Cannot resolve input shape")
    height = require_number(shape[2], "input height")
    width = require_number(shape[3], "input width")
    if height != width:
        raise ValueError(f"Expected square input, got {height}x{width}")
    return int(height)


def expected_query_count(static: dict[str, Any]) -> int | None:
    shape = static.get("onnx", {}).get("outputs", {}).get("dets")
    if isinstance(shape, list) and len(shape) >= 2 and isinstance(shape[1], int):
        return shape[1]
    return None


def method_detail(experiment: dict[str, Any]) -> str:
    details: list[str] = []
    pruning = experiment.get("pruning")
    if isinstance(pruning, dict):
        details.extend(f"{key}={value}" for key, value in sorted(pruning.items()))
    shape = experiment.get("input_shape")
    if experiment.get("method") == "input-resolution" and isinstance(shape, list):
        details.append(f"input={shape[-2]}x{shape[-1]}")
    return "; ".join(details) if details else "N/A"


def recovery_report(root: Path, experiment_id: str) -> dict[str, Any] | None:
    path = root / "artifacts" / "experiments" / experiment_id / "front" / "recovery" / "recovery-training.json"
    return load_json(path) if path.is_file() else None


def recovery_completed(experiment: dict[str, Any], report: dict[str, Any] | None) -> bool:
    tuning = experiment.get("fine_tuning")
    if not isinstance(tuning, dict) or not tuning.get("required"):
        return True
    if tuning.get("completed") is True:
        return True
    return bool(report and report.get("status") in {"completed", "complete"})


def effective_status(experiment: dict[str, Any], report: dict[str, Any] | None) -> str:
    registered = str(experiment.get("status", "unknown"))
    if "rejected" in registered:
        return registered
    tuning = experiment.get("fine_tuning")
    if not isinstance(tuning, dict) or not tuning.get("required") or tuning.get("completed"):
        return registered
    runtime_status = str(report.get("status", "")) if report else ""
    if runtime_status == "running":
        return "recovery-running"
    if runtime_status in {"completed", "complete"}:
        return "recovery-completed-awaiting-finalization"
    if runtime_status in {"failed", "error"}:
        return "recovery-failed"
    return "recovery-pending"


def protocol_matches_baseline(
    root: Path, payload: dict[str, Any], baseline: dict[str, Any]
) -> tuple[bool, str]:
    for key in ("image_count", "threshold", "miou_threshold"):
        if payload.get(key) != baseline.get(key):
            return False, f"protocol mismatch: {key}"
    candidate_dataset = payload.get("dataset")
    baseline_dataset = baseline.get("dataset")
    if candidate_dataset and baseline_dataset:
        if resolve(root, candidate_dataset).resolve() != resolve(root, baseline_dataset).resolve():
            return False, "protocol mismatch: dataset"
    return True, "protocol matches baseline"


def audit_evaluation(
    *,
    root: Path,
    path: Path,
    experiment_id: str,
    experiment: dict[str, Any],
    static: dict[str, Any],
    baseline: dict[str, Any],
    registered_path: Path | None,
    registered_sha256: str | None,
    completed: bool,
) -> EvaluationAudit:
    reasons: list[str] = []
    disposition = "valid-not-selected"
    source = "UNKNOWN"
    payload: dict[str, Any] | None = None

    try:
        payload = load_json(path)
        source = infer_accuracy_source(payload)
        for metric in ACCURACY_FIELDS:
            nested_metric(payload, metric)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        return EvaluationAudit(path, experiment_id, source, "rejected", str(error), payload)

    lower_name = path.name.lower()
    filename_markers = [marker for marker in INVALID_FILENAME_MARKERS if marker in lower_name]
    if filename_markers:
        reasons.append(f"filename marked {','.join(filename_markers)}")

    payload_status = str(payload.get("status", "")).lower()
    if payload.get("valid") is False or payload_status in {"invalid", "superseded"}:
        reasons.append(f"payload status is {payload_status or 'invalid'}")
    if payload.get("superseded_by"):
        reasons.append("payload declares superseded_by")

    protocol_ok, protocol_reason = protocol_matches_baseline(root, payload, baseline)
    if not protocol_ok:
        reasons.append(protocol_reason)

    if source == "ONNX":
        query_count = expected_query_count(static)
        num_select = payload.get("postprocess_num_select")
        if not isinstance(num_select, int):
            reasons.append("ONNX result lacks explicit postprocess_num_select")
        elif query_count is not None and num_select != query_count:
            reasons.append(
                f"postprocess_num_select={num_select} does not match query_count={query_count}"
            )
        parity = payload.get("pytorch_parity")
        if isinstance(parity, dict) and parity.get("passed") is False:
            reasons.append("PyTorch parity failed")

    is_registered = registered_path is not None and path.resolve() == registered_path.resolve()
    if is_registered and registered_sha256 and sha256(path) != registered_sha256:
        reasons.append("registry SHA-256 mismatch")

    if reasons:
        return EvaluationAudit(
            path,
            experiment_id,
            source,
            "rejected",
            "; ".join(reasons),
            payload,
        )

    if "before-recovery" in lower_name and completed:
        disposition = "superseded"
        reason = "pre-recovery result superseded by completed recovery"
    elif "before-recovery" in lower_name and not completed:
        disposition = "pending-input"
        reason = "pre-recovery diagnostic; recovery is unfinished"
    elif is_registered:
        disposition = "selected"
        reason = f"registry-selected; {protocol_reason}"
    else:
        reason = protocol_reason
    return EvaluationAudit(path, experiment_id, source, disposition, reason, payload)


def evaluation_paths(
    root: Path, experiment_id: str, registered_path: Path | None
) -> list[Path]:
    paths: set[Path] = set()
    if registered_path and registered_path.is_file():
        paths.add(registered_path)
    evaluation_dir = root / "results" / "coco-evaluation"
    if evaluation_dir.is_dir():
        paths.update(evaluation_dir.glob(f"{experiment_id}-front*.json"))
    if experiment_id == BASELINE_ID:
        baseline_path = root / DEFAULT_BASELINE_EVALUATION
        if baseline_path.is_file():
            paths.add(baseline_path)
    return sorted(paths, key=lambda item: item.as_posix())


def choose_evaluation(
    audits: Iterable[EvaluationAudit], registered_path: Path | None
) -> EvaluationAudit | None:
    eligible = [audit for audit in audits if audit.disposition in {"selected", "valid-not-selected"}]
    if not eligible:
        return None
    if registered_path:
        for audit in eligible:
            if audit.path.resolve() == registered_path.resolve():
                return audit
    # Prefer final PTH accuracy.  ONNX remains an independently audited parity
    # result unless the registry explicitly designates it as the paper result.
    return sorted(
        eligible,
        key=lambda audit: (
            "after-recovery" in audit.path.name,
            audit.source == "PTH",
            audit.path.stat().st_mtime_ns,
        ),
        reverse=True,
    )[0]


def delta(value: int | float, baseline: int | float) -> tuple[int | float, float]:
    absolute = value - baseline
    relative_percent = 100.0 * absolute / baseline
    return absolute, relative_percent


def validate_comparison(
    experiment_id: str,
    comparison: dict[str, Any],
    candidate_values: dict[str, int | float],
    baseline_values: dict[str, int | float],
) -> None:
    if comparison.get("baseline_experiment_id") != BASELINE_ID:
        raise ValueError(f"{experiment_id}: comparison baseline is not {BASELINE_ID}")
    if comparison.get("candidate_experiment_id") != experiment_id:
        raise ValueError(f"{experiment_id}: comparison candidate ID mismatch")
    for output_name, source_name in STRUCTURAL_FIELDS:
        block = comparison.get(source_name)
        if not isinstance(block, dict):
            raise ValueError(f"{experiment_id}: comparison missing {source_name}")
        actual = candidate_values[output_name]
        baseline = baseline_values[output_name]
        expected_abs, expected_rel_pct = delta(actual, baseline)
        if block.get("candidate") != actual or block.get("baseline") != baseline:
            raise ValueError(f"{experiment_id}: stale {source_name} comparison values")
        if block.get("delta") != expected_abs:
            raise ValueError(f"{experiment_id}: stale {source_name} absolute delta")
        ratio = block.get("delta_ratio")
        if not isinstance(ratio, (int, float)) or not math.isclose(
            100.0 * ratio, expected_rel_pct, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(f"{experiment_id}: stale {source_name} relative delta")


def structural_values(static: dict[str, Any], experiment_id: str) -> dict[str, int | float]:
    if static.get("experiment_id") != experiment_id:
        raise ValueError(f"Static-analysis experiment ID mismatch for {experiment_id}")
    onnx = static.get("onnx")
    if not isinstance(onnx, dict) or onnx.get("onnx_checker_valid") is not True:
        raise ValueError(f"Static-analysis ONNX is not valid for {experiment_id}")
    return {
        output_name: require_number(onnx.get(source_name), f"{experiment_id}.{source_name}")
        for output_name, source_name in STRUCTURAL_FIELDS
    }


def add_value_and_delta(
    row: dict[str, Any], name: str, value: int | float, baseline: int | float
) -> None:
    absolute, relative_percent = delta(value, baseline)
    row[name] = value
    row[f"{name}_delta_abs"] = absolute
    row[f"{name}_delta_rel_pct"] = relative_percent


def generate(root: Path) -> tuple[list[dict[str, Any]], list[EvaluationAudit]]:
    registry_path = root / "configs" / "experiments" / "registry.yaml"
    experiments = load_registry(registry_path)
    baseline_path = root / DEFAULT_BASELINE_EVALUATION
    baseline_evaluation = load_json(baseline_path)
    baseline_static_path = root / "artifacts" / "experiments" / BASELINE_ID / "front" / "static-analysis.json"
    baseline_static = load_json(baseline_static_path)
    baseline_values = structural_values(baseline_static, BASELINE_ID)
    baseline_accuracy = {
        name: nested_metric(baseline_evaluation, name) for name in ACCURACY_FIELDS
    }

    rows: list[dict[str, Any]] = []
    all_audits: list[EvaluationAudit] = []
    for experiment_id in EXPERIMENT_IDS:
        experiment = experiments[experiment_id]
        directory = root / "artifacts" / "experiments" / experiment_id / "front"
        static_path = directory / "static-analysis.json"
        static = load_json(static_path)
        values = structural_values(static, experiment_id)
        comparison_path = directory / "comparison-B01.json"
        if experiment_id != BASELINE_ID:
            comparison = load_json(comparison_path)
            validate_comparison(experiment_id, comparison, values, baseline_values)

        report = recovery_report(root, experiment_id)
        completed = recovery_completed(experiment, report)
        result = experiment.get("result") if isinstance(experiment.get("result"), dict) else {}
        registered_value = result.get("evaluation")
        registered_path = resolve(root, registered_value) if registered_value else None
        registered_hash = result.get("evaluation_sha256")
        audits = [
            audit_evaluation(
                root=root,
                path=path,
                experiment_id=experiment_id,
                experiment=experiment,
                static=static,
                baseline=baseline_evaluation,
                registered_path=registered_path,
                registered_sha256=registered_hash,
                completed=completed,
            )
            for path in evaluation_paths(root, experiment_id, registered_path)
        ]

        tuning = experiment.get("fine_tuning")
        unfinished_recovery = (
            isinstance(tuning, dict)
            and tuning.get("required") is True
            and not completed
            and "rejected" not in str(experiment.get("status", ""))
        )
        selected = None if unfinished_recovery else choose_evaluation(audits, registered_path)
        if selected:
            audits = [
                EvaluationAudit(
                    audit.path,
                    audit.experiment_id,
                    audit.source,
                    "selected" if audit.path == selected.path else audit.disposition,
                    audit.reason,
                    audit.payload,
                )
                for audit in audits
            ]
        all_audits.extend(audits)

        row: dict[str, Any] = {
            "experiment_id": experiment_id,
            "method": experiment.get("method", "N/A"),
            "method_detail": method_detail(experiment),
            "status": effective_status(experiment, report),
            "resolution_px": input_resolution(experiment, static),
            "static_analysis": relative(root, static_path),
            "comparison": (
                relative(root, comparison_path) if experiment_id != BASELINE_ID else "N/A"
            ),
        }
        for output_name, _ in STRUCTURAL_FIELDS:
            add_value_and_delta(row, output_name, values[output_name], baseline_values[output_name])

        if selected and selected.payload:
            for name in ACCURACY_FIELDS:
                add_value_and_delta(
                    row,
                    name,
                    nested_metric(selected.payload, name),
                    baseline_accuracy[name],
                )
            row["accuracy_source"] = selected.source
            row["accuracy_status"] = "valid"
            row["accuracy_result"] = relative(root, selected.path)
        else:
            for name in ACCURACY_FIELDS:
                row[name] = None
                row[f"{name}_delta_abs"] = None
                row[f"{name}_delta_rel_pct"] = None
            row["accuracy_source"] = "N/A"
            if unfinished_recovery:
                row["accuracy_status"] = "pending"
            elif "rejected" in str(experiment.get("status", "")):
                row["accuracy_status"] = "rejected"
            else:
                row["accuracy_status"] = "missing"
            row["accuracy_result"] = "N/A"
        rows.append(row)
    return rows, all_audits


def csv_value(value: Any) -> str | int:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return format(value, ".12g")
    return str(value)


def render_csv(rows: list[dict[str, Any]]) -> str:
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: csv_value(row.get(column)) for column in CSV_COLUMNS})
    return stream.getvalue()


def md_value(value: Any, *, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value).replace("|", "\\|")


def md_delta(value: Any, *, percent: bool = False) -> str:
    if value is None:
        return "N/A"
    suffix = "%" if percent else ""
    if isinstance(value, int):
        return f"{value:+,}{suffix}"
    return f"{float(value):+.6f}{suffix}"


def render_markdown(
    rows: list[dict[str, Any]], audits: list[EvaluationAudit], root: Path
) -> str:
    lines = [
        "# RF-DETR Round-2 Candidate Summary",
        "",
        "이 문서는 `scripts/reporting/generate_round2_candidate_summary.py`가 실제 registry, "
        "static-analysis/comparison JSON 및 평가 JSON에서 생성한다. 수치는 추정하지 않는다.",
        "",
        "- ONNX initializer elements는 상수 tensor의 element 수이며 학습 가능 "
        "parameter 수와 동의어가 아니다.",
        "- MACs/FLOPs는 Conv/MatMul/Gemm dense 산술만 포함한 부분 하한 "
        "추정치이며 실측 latency를 대체하지 않는다.",
        "- 모든 `Δ`는 B01 대비 `candidate - B01`, 상대 변화율은 `100 × Δ / B01`이다.",
        "- 복구 학습이 끝나지 않은 후보의 정확도는 복구 전 진단값 대신 `N/A (pending)`으로 표시한다.",
        "",
        "## 구조 및 연산량",
        "",
        "| ID | Method | Status | Res. | Initializer elements | Δ elements | Δ elements (%) | Nodes | Δ nodes | Δ nodes (%) | MACs* | Δ MACs | Δ MACs (%) | FLOPs* | Δ FLOPs | Δ FLOPs (%) | ONNX bytes | Δ bytes | Δ size (%) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    md_value(row["experiment_id"]),
                    md_value(row["method"]),
                    md_value(row["status"]),
                    md_value(row["resolution_px"]),
                    md_value(row["onnx_params"]),
                    md_delta(row["onnx_params_delta_abs"]),
                    md_delta(row["onnx_params_delta_rel_pct"], percent=True),
                    md_value(row["onnx_nodes"]),
                    md_delta(row["onnx_nodes_delta_abs"]),
                    md_delta(row["onnx_nodes_delta_rel_pct"], percent=True),
                    md_value(row["estimated_macs"]),
                    md_delta(row["estimated_macs_delta_abs"]),
                    md_delta(row["estimated_macs_delta_rel_pct"], percent=True),
                    md_value(row["estimated_flops"]),
                    md_delta(row["estimated_flops_delta_abs"]),
                    md_delta(row["estimated_flops_delta_rel_pct"], percent=True),
                    md_value(row["onnx_size_bytes"]),
                    md_delta(row["onnx_size_bytes_delta_abs"]),
                    md_delta(row["onnx_size_bytes_delta_rel_pct"], percent=True),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 정확도",
            "",
            "| ID | Accuracy status | Source | BBox AP | Δ abs. | Δ rel. (%) | Mask AP | Δ abs. | Δ rel. (%) | mIoU | Δ abs. | Δ rel. (%) | Result JSON |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    md_value(row["experiment_id"]),
                    md_value(row["accuracy_status"]),
                    md_value(row["accuracy_source"]),
                    md_value(row["bbox_ap"]),
                    md_delta(row["bbox_ap_delta_abs"]),
                    md_delta(row["bbox_ap_delta_rel_pct"], percent=True),
                    md_value(row["mask_ap"]),
                    md_delta(row["mask_ap_delta_abs"]),
                    md_delta(row["mask_ap_delta_rel_pct"], percent=True),
                    md_value(row["semantic_miou"]),
                    md_delta(row["semantic_miou_delta_abs"]),
                    md_delta(row["semantic_miou_delta_rel_pct"], percent=True),
                    md_value(row["accuracy_result"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 평가 산출물 판정",
            "",
            "| Experiment | Evaluation JSON | Backend | Disposition | Reason |",
            "|---|---|---|---|---|",
        ]
    )
    for audit in sorted(audits, key=lambda item: (item.experiment_id, item.path.name)):
        lines.append(
            "| "
            + " | ".join(
                [
                    audit.experiment_id,
                    relative(root, audit.path),
                    audit.source,
                    audit.disposition,
                    audit.reason.replace("|", "\\|"),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def output_path(root: Path, configured: Path) -> Path:
    return configured if configured.is_absolute() else root / configured


def update_or_check(path: Path, content: str, check: bool) -> bool:
    current = path.read_text(encoding="utf-8") if path.is_file() else None
    if current == content:
        return True
    if check:
        print(f"out of date: {path}", file=sys.stderr)
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    rows, audits = generate(root)
    csv_path = output_path(root, args.csv_output)
    markdown_path = output_path(root, args.markdown_output)
    valid = update_or_check(csv_path, render_csv(rows), args.check)
    valid &= update_or_check(markdown_path, render_markdown(rows, audits, root), args.check)
    if valid and not args.check:
        print(relative(root, csv_path))
        print(relative(root, markdown_path))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
