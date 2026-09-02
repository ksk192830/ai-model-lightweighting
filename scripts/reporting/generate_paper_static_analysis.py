#!/usr/bin/env python3
"""Generate a paper-ready static-analysis table from recorded artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_IDS = ("B01", "U02", "R01", "S01", "S02", "S03", "M01")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def portable(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def percent_change(candidate: float, baseline: float) -> float:
    return 100.0 * (candidate - baseline) / baseline


def generate(root: Path) -> dict[str, Any]:
    registry = yaml.safe_load(
        (root / "configs/experiments/registry.yaml").read_text(encoding="utf-8")
    )["experiments"]
    rows = []
    for experiment_id in EXPERIMENT_IDS:
        path = root / f"artifacts/experiments/{experiment_id}/front/static-analysis.json"
        if not path.is_file():
            continue
        report = read_json(path)
        onnx = report.get("onnx", {})
        checkpoint = report.get("checkpoint", {})
        experiment = registry[experiment_id]
        total_nodes = int(onnx.get("total_graph_nodes", onnx.get("onnx_nodes", 0)))
        supported = int(onnx.get("supported_compute_nodes", 0))
        unresolved = int(onnx.get("unresolved_compute_nodes", 0))
        estimated_fraction = onnx.get("estimated_node_fraction_of_graph")
        if estimated_fraction is None:
            estimated_fraction = supported / total_nodes if total_nodes else 0.0
        by_operator = onnx.get("by_operator", {})
        source_id = experiment.get("artifact_source", experiment_id)
        source_onnx = root / f"artifacts/experiments/{source_id}/front/model.onnx"
        status = str(experiment.get("status", "unknown"))
        recovery = experiment.get("fine_tuning", {})
        if (
            recovery.get("required")
            and not recovery.get("completed")
            and "rejected" not in status
        ):
            status = "provisional-recovery-running"
        rows.append(
            {
                "experiment_id": experiment_id,
                "method": experiment.get("method", "unknown"),
                "status": status,
                "input_resolution": experiment.get("input_shape", [1, 3, 504, 504])[-1],
                "checkpoint_model_state_elements": checkpoint.get("model_state_elements"),
                "checkpoint_prunable_elements": checkpoint.get("prunable_parameters"),
                "checkpoint_prunable_sparsity": checkpoint.get("sparsity"),
                "onnx_initializer_elements": onnx.get(
                    "onnx_initializer_elements",
                    onnx.get("onnx_initializer_parameters"),
                ),
                "onnx_initializer_tensor_bytes": onnx.get("onnx_initializer_tensor_bytes"),
                "onnx_file_bytes": onnx.get("onnx_size_bytes"),
                "onnx_nodes": total_nodes,
                "estimated_macs": onnx.get("estimated_macs"),
                "estimated_flops": onnx.get("estimated_flops"),
                "supported_compute_nodes": supported,
                "unresolved_compute_nodes": unresolved,
                "excluded_operator_nodes": onnx.get(
                    "excluded_operator_nodes",
                    sum(onnx.get("excluded_operator_types", {}).values()),
                ),
                "conv_macs": by_operator.get("Conv", {}).get("macs", 0),
                "matmul_macs": by_operator.get("MatMul", {}).get("macs", 0),
                "gemm_macs": by_operator.get("Gemm", {}).get("macs", 0),
                "resolved_supported_compute_coverage": onnx.get("compute_node_coverage"),
                "estimated_node_fraction_of_graph": estimated_fraction,
                "flops_status": onnx.get("flops_status", "legacy-partial"),
                "static_analysis": portable(root, path),
                "source_onnx": portable(root, source_onnx),
                "static_current_for_onnx": (
                    source_onnx.is_file()
                    and path.stat().st_mtime_ns >= source_onnx.stat().st_mtime_ns
                    and not (
                        recovery.get("required")
                        and not recovery.get("completed")
                        and "rejected" not in status
                    )
                ),
            }
        )
    baseline = next(row for row in rows if row["experiment_id"] == "B01")
    for row in rows:
        for field in (
            "onnx_initializer_elements",
            "onnx_file_bytes",
            "onnx_nodes",
            "estimated_macs",
            "estimated_flops",
        ):
            value = row[field]
            base = baseline[field]
            row[f"{field}_delta_vs_B01_pct"] = (
                percent_change(float(value), float(base))
                if value is not None and base not in {None, 0}
                else None
            )

    two_to_four: dict[str, Any] | None = None
    path_2to4 = root / "artifacts/experiments/M01/front/onnx-2to4.json"
    if path_2to4.is_file():
        raw = read_json(path_2to4)
        two_to_four = {
            key: raw.get(key)
            for key in (
                "constant_weight_ops",
                "eligible_ops_by_shape",
                "compliant_ops",
                "compliance",
                "valid",
            )
        }

    engine_rows = []
    for path in sorted(
        (root / "artifacts/experiments").glob("*/front/engine-static-analysis.json")
    ):
        report = read_json(path)
        engine_rows.append(
            {
                "experiment_id": report.get("experiment_id", path.parents[1].name),
                "engine_size_bytes": report.get("engine_size_bytes"),
                "engine_sha256": report.get("engine_sha256"),
                "num_layers": report.get("num_layers"),
                "profiling_verbosity": report.get("profiling_verbosity"),
                "precision_evidence": report.get("layers_with_precision_evidence"),
                "gpu": report.get("gpu"),
                "tensorrt_version": report.get("tensorrt_version"),
                "report": portable(root, path),
            }
        )
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": {
            "checkpoint_inventory": (
                "Every tensor in checkpoint['model'] is counted separately from "
                "the prunable subset (floating ndim>=2 keys ending in weight)."
            ),
            "onnx_inventory": (
                "Initializer elements and their dtype-implied tensor storage are "
                "counted. Initializers are constants, not necessarily trainable parameters."
            ),
            "mac_flop_formula": "Conv/MatMul/Gemm: one MAC = one multiply plus accumulate = two FLOPs.",
            "scope": (
                "Partial lower-bound dense arithmetic. Unsupported operators, data "
                "movement, kernel fusion, and backend scheduling are excluded."
            ),
        },
        "rows": rows,
        "two_to_four": two_to_four,
        "engine_rows": engine_rows,
    }


def render_csv(report: dict[str, Any]) -> str:
    from io import StringIO

    rows = report["rows"]
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "PASS" if value else "PENDING/STALE"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{float(value):.{digits}f}"


def delta(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):+.3f}%"


def render_markdown(report: dict[str, Any]) -> str:
    rows = report["rows"]
    lines = [
        "# RF-DETR 경량화 정적 분석",
        "",
        "## 논문 기재용 분석 방법",
        "",
        "체크포인트는 `checkpoint['model']`의 전체 tensor element와 pruning 대상 "
        "weight element를 분리해 집계했다. ONNX는 initializer element, dtype 기준 "
        "tensor 저장 바이트, graph node 수를 집계하고 ONNX checker를 통과한 graph만 "
        "비교에 포함했다.",
        "",
        "연산량은 shape inference 이후 Conv, MatMul, Gemm에 대해 산출했다. 곱셈-누산 "
        "1회를 1 MAC 또는 2 FLOPs로 정의했다. 동적 차원은 1로 치환했으며 elementwise, "
        "normalization, activation, resize, control-flow 및 data movement는 제외했다. "
        "따라서 아래 MAC/FLOP는 전체 실행 비용이 아니라 동일 분석기로 얻은 dense "
        "산술의 **부분 하한 추정치**다.",
        "",
        "## 구조 및 연산량 결과",
        "",
        "| ID | 상태 | 입력 | ONNX initializer elements | Δ vs B01 | Nodes | Δ vs B01 | MACs (lower bound) | Δ vs B01 | FLOPs (lower bound) | Estimated/all nodes | Current |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["experiment_id"],
                    row["status"],
                    number(row["input_resolution"]),
                    number(row["onnx_initializer_elements"]),
                    delta(row["onnx_initializer_elements_delta_vs_B01_pct"]),
                    number(row["onnx_nodes"]),
                    delta(row["onnx_nodes_delta_vs_B01_pct"]),
                    number(row["estimated_macs"]),
                    delta(row["estimated_macs_delta_vs_B01_pct"]),
                    number(row["estimated_flops"]),
                    number(100 * row["estimated_node_fraction_of_graph"], 2) + "%",
                    number(row["static_current_for_onnx"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "`ONNX initializer elements`는 ONNX 상수 tensor element 수이며 학습 가능한 "
            "parameter 수와 동일하다고 단정하지 않는다. 실제 지연시간 감소는 이 표가 "
            "아니라 동일 GPU에서 측정한 TensorRT median/IQR/P95로 검증해야 한다.",
            "",
            "## 연산량 추정 커버리지",
            "",
            "| ID | Conv MACs | MatMul MACs | Gemm MACs | Estimated nodes | Unresolved compute nodes | Excluded nodes | Resolved compute coverage |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['experiment_id']} | {number(row['conv_macs'])} | "
            f"{number(row['matmul_macs'])} | {number(row['gemm_macs'])} | "
            f"{number(row['supported_compute_nodes'])} | "
            f"{number(row['unresolved_compute_nodes'])} | "
            f"{number(row['excluded_operator_nodes'])} | "
            f"{number(100 * row['resolved_supported_compute_coverage'], 3)}% |"
        )
    lines.extend(
        [
            "",
            "`Resolved compute coverage`는 Conv/MatMul/Gemm 후보 중 shape을 해석한 "
            "비율이며 전체 graph 연산 커버리지가 아니다. `Excluded nodes`는 "
            "자료 이동·activation 등을 포함하므로 node 개수만으로 비용 비율을 "
            "추론하지 않는다.",
            "",
            "## 체크포인트 희소도",
            "",
            "| ID | Model-state elements | Prunable weight elements | Zero sparsity |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['experiment_id']} | {number(row['checkpoint_model_state_elements'])} | "
            f"{number(row['checkpoint_prunable_elements'])} | "
            f"{number(100 * row['checkpoint_prunable_sparsity'], 3) + '%' if row['checkpoint_prunable_sparsity'] is not None else 'N/A'} |"
        )
    two_to_four = report.get("two_to_four")
    lines.extend(["", "## 2:4 구조 검증", ""])
    if two_to_four:
        lines.append(
            f"ONNX constant-weight 연산 {two_to_four['constant_weight_ops']}개 중 shape 적격 "
            f"{two_to_four['eligible_ops_by_shape']}개, 2:4 준수 {two_to_four['compliant_ops']}개로 "
            f"graph-level compliance는 {number(100 * two_to_four['compliance'], 3)}%다. "
            "다만 이 결과는 pattern 존재만 증명하며 실제 sparse tactic 선택은 TensorRT "
            "build log와 M01 dense-control 대비 M02 latency로 별도 입증한다."
        )
    else:
        lines.append("M01 복구 및 최종 ONNX 검증 완료 후 자동 생성된다.")
    lines.extend(["", "## TensorRT engine 정적 분석", ""])
    if report["engine_rows"]:
        lines.extend(
            [
                "| ID | Engine bytes | Layers | Precision evidence | GPU | TensorRT |",
                "|---|---:|---:|---|---|---|",
            ]
        )
        for row in report["engine_rows"]:
            precision_evidence = str(row["precision_evidence"]).replace(
                "|", "\\|"
            )
            lines.append(
                f"| {row['experiment_id']} | {number(row['engine_size_bytes'])} | "
                f"{number(row['num_layers'])} | {precision_evidence} | "
                f"{row['gpu']} | {row['tensorrt_version']} |"
            )
    else:
        lines.append("현재 M01 복구 완료를 대기 중이며 engine build 후 자동 채워진다.")
    lines.extend(
        [
            "",
            "## 해석상 제한",
            "",
            "- 후보 간 MAC/FLOP 비교는 동일 분석기와 동일 연산 범위에서만 해석한다.",
            "- S01/S02 decoder 제거처럼 node와 initializer가 감소해도 attention 계열의 "
            "산술 하한 감소폭은 작을 수 있다.",
            "- 비정형 또는 2:4 pruning은 dense ONNX node/MAC를 줄이지 않는다. 지원 "
            "kernel 선택과 실측 latency가 가속 증거다.",
            "- 복구 학습 중인 M01 행은 최종 ONNX가 생성될 때까지 잠정치다.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("results/paper-static-analysis.csv"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("docs/reports/paper-static-analysis.md"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    report = generate(root)
    csv_output = resolve(root, args.csv_output)
    markdown_output = resolve(root, args.markdown_output)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    csv_output.write_text(render_csv(report), encoding="utf-8")
    markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(portable(root, csv_output))
    print(portable(root, markdown_output))
    return 0


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


if __name__ == "__main__":
    raise SystemExit(main())
