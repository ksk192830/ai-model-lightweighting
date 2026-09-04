#!/usr/bin/env python3
"""Generate the frozen Stage-3 paper evidence table, report, and figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
STAGE1 = Path("results/stage1-static-evaluation.json")
STAGE2_STATE = Path("results/stage2-notebook-state.json")
STAGE3 = Path("results/stage3-pareto.json")
DECISIONS = Path("results/stage3-paper-candidate-decisions.csv")
RECOMMENDATIONS = Path("results/stage3-final-recommendations.json")
REPORT = Path("docs/reports/stage3-final-analysis.md")
EVIDENCE_MAP = Path("docs/paper/stage3-evidence-map.md")
FIGURE_DIR = Path("figures")

EXPECTED_OBJECTIVES = {
    "maximize": ["mask_ap"],
    "minimize": ["median_ms", "engine_size_bytes"],
}
REALTIME_BUDGET_MS = 1000.0 / 30.0


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def configure_font() -> None:
    preferred = (
        "Noto Sans CJK KR",
        "Noto Sans CJK JP",
        "DejaVu Sans",
    )
    available = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((name for name in preferred if name in available), "DejaVu Sans")
    plt.rcParams.update(
        {
            "font.family": selected,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def stage2_label(status: str) -> str:
    return {
        "completed": "완료",
        "build-failed": "엔진 생성 실패",
        "engine-analysis-failed": "엔진 검사 실패",
        "benchmark-failed": "속도 평가 실패",
        "accuracy-failed": "정확도 평가 실패",
        "running": "진행 중",
        "pending": "대기",
        "not-applicable": "대상 외",
    }.get(status, status)


def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    no_worse = (
        float(left["mask_ap"]) >= float(right["mask_ap"])
        and float(left["median_ms"]) <= float(right["median_ms"])
        and int(left["engine_size_bytes"]) <= int(right["engine_size_bytes"])
    )
    strictly_better = (
        float(left["mask_ap"]) > float(right["mask_ap"])
        or float(left["median_ms"]) < float(right["median_ms"])
        or int(left["engine_size_bytes"]) < int(right["engine_size_bytes"])
    )
    return no_worse and strictly_better


def gate_failure_reason(row: dict[str, Any]) -> str:
    failures: list[str] = []
    if float(row["bbox_ap_delta_vs_B01"]) < -0.01:
        failures.append(f"BBox AP Δ {float(row['bbox_ap_delta_vs_B01']):+.4f} < -0.0100")
    if float(row["mask_ap_delta_vs_B01"]) < -0.01:
        failures.append(f"Mask AP Δ {float(row['mask_ap_delta_vs_B01']):+.4f} < -0.0100")
    if float(row["semantic_miou_delta_vs_B01"]) < -0.02:
        failures.append(
            f"mIoU Δ {float(row['semantic_miou_delta_vs_B01']):+.4f} < -0.0200"
        )
    return "; ".join(failures) or str(row.get("exclusion_reason") or "정확도 gate 불통")


def build_decisions(
    stage1: dict[str, Any],
    state: dict[str, Any],
    stage3: dict[str, Any],
) -> list[dict[str, Any]]:
    measured = {str(row["experiment_id"]): row for row in stage3.get("rows", [])}
    states = state.get("candidates", {})
    if not isinstance(states, dict):
        raise ValueError("Stage-2 state has no candidate mapping")
    pareto_ids = set(stage3.get("pareto_candidate_ids", []))
    deployment_ids = set(stage3.get("deployment_candidate_ids", []))
    pareto_rows = [measured[item] for item in pareto_ids if item in measured]
    decisions: list[dict[str, Any]] = []

    for item in stage1.get("rows", []):
        experiment_id = str(item["experiment_id"])
        stage1_pass = bool(item.get("stage2_notebook_eligible"))
        result = measured.get(experiment_id)
        state_item = states.get(experiment_id, {})
        if not stage1_pass:
            status = "not-applicable"
            reason = str(item.get("failure_reason") or "1차 정적평가 불통")
        else:
            status = str(state_item.get("status") or ("completed" if result else "pending"))
            reason = str(state_item.get("reason") or "")
            if experiment_id == "Q03" and status == "build-failed":
                reason = (
                    "TensorRT 10.16.1.11 INT4 DequantizeLinear parser 오류 "
                    "(상세 로그: results/stage2-notebook-logs/Q03-build.log)"
                )

        gate_pass = bool(result and result.get("accuracy_gate_pass"))
        realtime_pass = bool(result and result.get("realtime_30fps_pass"))
        p95_warning = bool(result and result.get("p95_latency_warning"))
        pareto = experiment_id in pareto_ids
        deployment = experiment_id in deployment_ids
        dominated_by = ""

        if not stage1_pass:
            final_reason = reason
        elif status != "completed":
            final_reason = reason or f"2차 {stage2_label(status)}"
        elif not gate_pass and result is not None:
            final_reason = gate_failure_reason(result)
        elif pareto:
            final_reason = "최종 Pareto 비지배해"
        elif result is not None:
            dominators = [
                str(row["experiment_id"])
                for row in pareto_rows
                if dominates(row, result)
            ]
            dominated_by = ", ".join(sorted(dominators))
            final_reason = (
                f"정확도 gate 통과 후 {dominated_by}에 지배됨"
                if dominated_by
                else "정확도 gate 통과 후 Pareto 비지배해 아님"
            )
        else:
            final_reason = "측정 결과 없음"

        decisions.append(
            {
                "experiment_id": experiment_id,
                "method_summary": item.get("method_summary", ""),
                "stage1_result": "통과" if stage1_pass else "불통",
                "stage2_status": stage2_label(status),
                "bbox_ap": result.get("bbox_ap") if result else None,
                "mask_ap": result.get("mask_ap") if result else None,
                "semantic_miou": result.get("semantic_miou") if result else None,
                "median_ms": result.get("median_ms") if result else None,
                "p95_ms": result.get("p95_ms") if result else None,
                "replicate_median_cv": result.get("replicate_median_cv") if result else None,
                "engine_size_bytes": result.get("engine_size_bytes") if result else None,
                "accuracy_gate_pass": gate_pass if result else None,
                "realtime_30fps_pass": realtime_pass if result else None,
                "p95_latency_warning": p95_warning if result else None,
                "pareto_optimal": pareto,
                "deployment_candidate": deployment,
                "dominated_by": dominated_by,
                "final_disposition_reason": final_reason,
            }
        )
    return decisions


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def yes_no(value: Any) -> str:
    if value is None:
        return "—"
    return "통과" if bool(value) else "불통"


def write_report(
    path: Path,
    stage1: dict[str, Any],
    state: dict[str, Any],
    stage3: dict[str, Any],
    decisions: list[dict[str, Any]],
) -> None:
    measured = {row["experiment_id"]: row for row in stage3["rows"]}
    baseline = measured["B01"]
    c01 = measured["C01"]
    r01 = measured["R01"]
    gate_ids = [row["experiment_id"] for row in stage3["rows"] if row["accuracy_gate_pass"]]
    p95_ids = [row["experiment_id"] for row in stage3["rows"] if row["p95_latency_warning"]]
    max_cv = max(stage3["rows"], key=lambda row: float(row["replicate_median_cv"]))
    run_id = state.get("measurement_run_id", "20260904_104755")

    lines = [
        "# Stage 3 최종 분석 및 논문 작성 입력",
        "",
        "> 이 문서는 논문 결과 장을 작성하기 전에 수치, 판정, 추천 원칙과 근거를",
        "> 하나의 공식 실행으로 고정한다. 모든 표와 그림은 machine-readable 결과에서 자동 생성한다.",
        "",
        "## 공식 결과 동결",
        "",
        f"- 공식 성능 실행 ID: `{run_id}`",
        "- 장비: NVIDIA GeForce RTX 4050 Laptop GPU, CUDA 12.8, TensorRT 10.16.1.11",
        "- 조건: AC 전원, platform profile 및 CPU EPP `performance`",
        "- 후보 흐름: 1차 26개 → 2차 대상 22개 → 완료 21개 → 정확도 gate 통과 11개 → Pareto 2개",
        "- Q03: INT4 block quantization parser 오류로 engine build 실패",
        "- 지연시간: 고정 32장, warm-up 20회, 200회 × 3반복; 총 63개 반복 모두 부하 gate 통과",
        "- CV 5% 초과 후보와 재측정 후보: 0개",
        f"- 최대 반복 median CV: {max_cv['experiment_id']} {100*float(max_cv['replicate_median_cv']):.2f}%",
        "- 데이터셋 의미적 fingerprint와 이미지 collection fingerprint: 노트북·데스크톱 일치",
        "- 최종 Pareto engine C01·R01: 노트북 보존 파일의 SHA-256 일치",
        "",
        "## 평가 흐름",
        "",
        "| 단계 | 후보 수 | 판정 |",
        "|---|---:|---|",
        f"| 최초 후보 | {int(stage1['candidate_count'])} | registry 전체 |",
        f"| 1차 정적 통과 | {int(stage1['passed_count'])} | 2차 대상 |",
        "| 2차 완료 | 21 | 정확도·성능 실측 확보 |",
        "| 2차 실패 | 1 | Q03 engine build 실패 |",
        f"| 정확도 gate 통과 | {len(gate_ids)} | {', '.join(gate_ids)} |",
        "| Pareto 비지배해 | 2 | C01, R01 |",
        "| 최종 30 FPS 배포 후보 | 2 | C01, R01 |",
        "",
        "![후보 평가 흐름](../../figures/stage3_candidate_funnel.png)",
        "",
        "## 최종 판정표",
        "",
        "| ID | 경량화 방식 | 1차 | 2차 | 정확도 gate | 30 FPS | P95 경고 | Pareto | 최종 사유 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in decisions:
        lines.append(
            "| {experiment_id} | {method_summary} | {stage1_result} | {stage2_status} | "
            "{accuracy} | {realtime} | {p95} | {pareto} | {reason} |".format(
                **row,
                accuracy=yes_no(row["accuracy_gate_pass"]),
                realtime=yes_no(row["realtime_30fps_pass"]),
                p95="경고" if row["p95_latency_warning"] else "—",
                pareto="예" if row["pareto_optimal"] else "아니오",
                reason=str(row["final_disposition_reason"]).replace("|", "/"),
            )
        )

    lines.extend(
        [
            "",
            "## Pareto 결과",
            "",
            "주 목적은 Mask AP 최대화, median latency 최소화, engine size 최소화다.",
            "BBox AP, semantic mIoU, P95 latency와 GPU memory는 해석용 보조지표다.",
            "정확도 gate를 통과한 후보 중 C01과 R01만 비지배해이며, 나머지는 두 후보 중",
            "하나에 세 주 목적 모두에서 지배된다.",
            "",
            "![Mask AP와 지연시간](../../figures/stage3_mask_latency.png)",
            "",
            "![Mask AP와 엔진 크기](../../figures/stage3_mask_engine_size.png)",
            "",
            "| 후보 | Mask AP | Median(ms) | P95(ms) | Engine(MiB) | B01 대비 latency | B01 대비 engine |",
            "|---|---:|---:|---:|---:|---:|---:|",
            f"| B01 | {fmt(baseline['mask_ap'])} | {fmt(baseline['median_ms'], 3)} | {fmt(baseline['p95_ms'], 3)} | {float(baseline['engine_size_bytes'])/1048576:.1f} | 기준 | 기준 |",
            f"| C01 | {fmt(c01['mask_ap'])} | {fmt(c01['median_ms'], 3)} | {fmt(c01['p95_ms'], 3)} | {float(c01['engine_size_bytes'])/1048576:.1f} | {float(c01['median_latency_reduction_vs_B01_pct']):.2f}% 감소 | {float(c01['engine_size_reduction_vs_B01_pct']):.2f}% 감소 |",
            f"| R01 | {fmt(r01['mask_ap'])} | {fmt(r01['median_ms'], 3)} | {fmt(r01['p95_ms'], 3)} | {float(r01['engine_size_bytes'])/1048576:.1f} | {float(r01['median_latency_reduction_vs_B01_pct']):.2f}% 감소 | {float(r01['engine_size_reduction_vs_B01_pct']):.2f}% 감소 |",
            "",
            f"C01은 R01보다 median latency가 {float(c01['median_ms'])-float(r01['median_ms']):.3f} ms 느리지만 "
            f"Mask AP가 {float(c01['mask_ap'])-float(r01['mask_ap']):.5f} 높고 engine은 "
            f"{(float(r01['engine_size_bytes'])-float(c01['engine_size_bytes']))/1048576:.1f} MiB 작다.",
            "",
            "## 최종 추천 원칙",
            "",
            "단일 가중합 점수로 임의의 절대 우승자를 만들지 않고 배포 목적별로 추천한다.",
            "",
            "| 시나리오 | 추천 | 근거 |",
            "|---|---|---|",
            "| 균형형 | C01 | B01 수준 Mask AP, 더 높은 mIoU, 44.79% latency 감소, 49.95% engine 감소 |",
            "| 정확도 우선 | C01 | 정확도 gate 통과 후보 중 최고 Mask AP |",
            "| 크기 우선 | C01 | Pareto 후보 중 더 작은 64.7 MiB engine |",
            "| 속도 우선 | R01 | 정확도 gate 통과 후보 중 최저 median 24.010 ms |",
            "",
            "B03은 정확도 gate 통과 후보 중 최고 BBox AP, C02는 같은 집합에서 최고 semantic mIoU를 기록한 보조 비교 후보로 보고하되,",
            "세 주 목적에서는 C01에 지배되므로 최종 Pareto 추천에는 포함하지 않는다.",
            "",
            "## 측정 안정성과 실시간 해석",
            "",
            f"- P95 33.33 ms 초과 경고 후보: {', '.join(p95_ids)}",
            "- P95 경고는 진단 지표이며 정확도 gate, Pareto 또는 배포 후보의 hard gate가 아니다.",
            "- 30 FPS 판정은 median latency ≤ 33.33 ms이며 지속 처리량 또는 모든 프레임의 30 FPS를 보장하지 않는다.",
            "- 반복 median CV는 모두 5% 이하여서 공식 재측정 라운드는 발생하지 않았다.",
            "",
            "![반복 median CV](../../figures/stage3_latency_cv.png)",
            "",
            "## 논문에서 허용되는 주장과 제한",
            "",
            "- 허용: 기록된 RTX 4050 Laptop GPU 성능 우선 조건에서 C01·R01이 3축 Pareto 비지배해였다.",
            "- 허용: C01과 R01 모두 median 기준 30 FPS 예산을 충족했다.",
            "- 제한: 437장은 반복 비교용 benchmark이므로 완전히 손대지 않은 confirmatory test로 표현하지 않는다.",
            "- 제한: 외부 장소·카메라·날씨에 대한 일반화, 실제 ROS2 end-to-end latency와 주차 성공률은 검증하지 않았다.",
            "- 제한: Q03의 실패는 현재 TensorRT parser와 export 형식의 호환성 실패이며 INT4 전체의 일반적 실패로 확대하지 않는다.",
            "- 제한: 단일 학습 seed와 단일 GPU 결과이므로 학습·장비 간 분산을 추정하지 않는다.",
            "",
            "## 논문 작성 전에 남은 비실험 작업",
            "",
            "1. 본 문서의 수치와 근거 연결표를 이용해 개조식 구성안과 줄글 초안을 Stage 3로 갱신한다.",
            "2. 참고문헌의 저자·학회·페이지·DOI를 제출 양식에 맞게 검증한다.",
            "3. 저장소에 근거가 없는 데이터 수집 조건, 라벨링 기준, 기존 ASK 논문 정보와 사사 문구는 사용자 자료를 받아 채운다.",
            "4. 부하 수준별 스트레스 실험, 외부 holdout과 Q03 재export는 추가 주장을 원할 때만 수행한다.",
            "",
            "## 자동 생성 근거",
            "",
            "- [Stage 1 전체 결과](../../results/stage1-static-evaluation.json)",
            "- [Stage 2 상태](../../results/stage2-notebook-state.json)",
            "- [Stage 2 통합 결과](../../results/stage2-notebook-summary.json)",
            "- [Stage 3 Pareto](../../results/stage3-pareto.json)",
            "- [전체 후보 판정 CSV](../../results/stage3-paper-candidate-decisions.csv)",
            "- [최종 추천 JSON](../../results/stage3-final-recommendations.json)",
            "- [통합 Excel](../../results/stage2-evaluation-report.xlsx)",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_recommendations(path: Path, state: dict[str, Any], stage3: dict[str, Any]) -> None:
    measured = {row["experiment_id"]: row for row in stage3["rows"]}
    gate_rows = [row for row in stage3["rows"] if row["accuracy_gate_pass"]]
    payload = {
        "official_measurement_run_id": state.get("measurement_run_id", "20260904_104755"),
        "selection_policy": "scenario-specific Pareto recommendation; no weighted single winner",
        "pareto_objectives": EXPECTED_OBJECTIVES,
        "pareto_candidate_ids": stage3.get("pareto_candidate_ids", []),
        "deployment_candidate_ids": stage3.get("deployment_candidate_ids", []),
        "recommendations": {
            "balanced": "C01",
            "accuracy_priority_mask_ap": "C01",
            "engine_size_priority_within_pareto": "C01",
            "latency_priority": "R01",
        },
        "secondary_metric_leaders_among_accuracy_gate_pass": {
            "bbox_ap": max(gate_rows, key=lambda row: row["bbox_ap"])["experiment_id"],
            "semantic_miou": max(gate_rows, key=lambda row: row["semantic_miou"])["experiment_id"],
        },
        "c01": {key: measured["C01"][key] for key in ("mask_ap", "bbox_ap", "semantic_miou", "median_ms", "p95_ms", "engine_size_bytes")},
        "r01": {key: measured["R01"][key] for key in ("mask_ap", "bbox_ap", "semantic_miou", "median_ms", "p95_ms", "engine_size_bytes")},
        "limitations": [
            "fixed 437-image repeated comparison benchmark",
            "single training seed",
            "single RTX 4050 Laptop GPU environment",
            "30 FPS classification is based on median latency",
            "Q03 excluded after TensorRT INT4 parser build failure",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_evidence_map(path: Path) -> None:
    text = """# Stage 3 논문 근거 연결표

> 논문 작성자는 아래 근거가 있는 범위에서만 수치와 결론을 사용한다. 저장소에
> 근거가 없는 항목은 추정하지 않고 자료 필요 상태로 남긴다.

| 논문 내용 | 공식 근거 | 사용 가능한 주장 | 주의사항 |
|---|---|---|---|
| 데이터 분할 | `docs/reports/dataset-split-validity.md`, `docs/reports/metrics/dataset-split-audit.json` | train 3,625 / valid 404 / benchmark 437, 이미지·세션 중복 0 | benchmark는 반복 비교용 |
| 노트북 데이터 동일성 | `docs/reports/metrics/dataset-fingerprint-comparison.json` | 의미적·이미지·최종 fingerprint 일치 | raw JSON byte hash 차이는 직렬화 순서 차이 |
| 기준 모델 학습 | `configs/training/front_rfdetr_seg_large.yaml`, `docs/guides/front-baseline-training.md` | 설정과 18 epoch 조기 종료 | 단일 seed |
| 후보 구성 | `configs/experiments/registry.yaml`, `results/round2-candidate-summary.md` | 전체 26개 후보와 경량화 방식 | W 계열은 연구 범위에서 제거됨 |
| 1차 정적평가 | `results/stage1-static-evaluation.json`, `results/stage1-static-evaluation.md` | 26개 중 22개 통과, 4개 불통 | MAC/FLOP는 해석 가능한 연산의 dense 하한 |
| 평가 기준 | `configs/experiments/defaults.yaml`, `docs/guides/three-stage-candidate-evaluation.md` | 정확도 gate, 3회 지연시간, CV, 30 FPS, P95 정책 | engineering gate이며 통계적 유의성 기준 아님 |
| 측정 부하 통제 | `docs/reports/measurement-load-control.md`, `results/measurement-environment/20260904_104755/` | AC·performance 조건, 63개 부하 gate 통과 | 부하 스트레스 실험과 구분 |
| 2차 후보 결과 | `results/stage2-notebook-summary.json`, `docs/reports/notebook-stage2-results.md` | 21개 완료, Q03 실패, 모든 정확도·속도·크기 수치 | 공식 실행 외 기존 latency와 혼합 금지 |
| Pareto 분석 | `results/stage3-pareto.json`, `docs/reports/stage3-final-analysis.md` | C01·R01 비지배해와 목적별 추천 | 임의 가중합 단일 우승자 없음 |
| 최종 엔진 보존 | `docs/reports/metrics/notebook-final-engine-verification.json` | C01·R01 engine hash 일치 | engine은 노트북에만 보존되는 장비 종속 파일 |
| 종합 표 | `results/stage3-paper-candidate-decisions.csv`, `results/stage2-evaluation-report.xlsx` | 26개 전체 단계별 판정 추적 | Q03에는 성능 수치 없음 |

## 저장소에 아직 없는 자료

- 원본 카메라 모델, 영상 해상도, 프레임 추출 간격, 장소·조명·날씨와 데이터 사용 권한
- 클래스 정의, polygon 경계·가림·잘림 처리 규칙, 라벨링 도구와 검수 이력
- 기존 ASK 2026 논문의 정확한 서지정보와 본 연구와의 연결
- 지원기관, 과제번호와 공식 사사 문구

이 자료가 제공되기 전에는 논문에서 사실처럼 서술하지 않는다.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def style_axis(ax: Any) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#D8DEE8", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)


def save_figure(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_funnel(path: Path) -> None:
    labels = ["최초 후보", "1차 통과", "2차 완료", "정확도 gate", "Pareto", "30 FPS 배포"]
    values = [26, 22, 21, 11, 2, 2]
    colors = ["#93A4B8", "#6E8EAD", "#497FB5", "#2F75B5", "#00A6A6", "#157A6E"]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], height=0.62)
    style_axis(ax)
    ax.grid(axis="x", color="#D8DEE8", linewidth=0.7, alpha=0.7)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("후보 수")
    ax.set_title("3단계 후보 평가 흐름", loc="left", fontweight="bold")
    ax.set_xlim(0, 28)
    for bar, value in zip(bars, values[::-1]):
        ax.text(value + 0.35, bar.get_y() + bar.get_height() / 2, str(value), va="center")
    save_figure(fig, path)


def plot_scatter(rows: list[dict[str, Any]], field: str, xlabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    offsets = {
        "median_ms": {
            "C01": (-24, 12),
            "B03": (5, 12),
            "B02": (5, -13),
            "C02": (5, 7),
            "R01": (5, 5),
            "M01": (5, 8),
            "M02": (5, -12),
        },
        "engine_mib": {
            "C01": (-24, 12),
            "C02": (-25, -14),
            "B03": (6, 12),
            "B02": (6, -14),
            "R01": (6, -1),
            "B01": (6, 10),
            "R02": (6, -13),
            "S01": (6, 10),
            "C04": (6, -12),
        },
    }
    for row in rows:
        gate = bool(row["accuracy_gate_pass"])
        pareto = bool(row["pareto_optimal"])
        color = "#157A6E" if pareto else ("#2F75B5" if gate else "#AEB7C2")
        size = 105 if pareto else 48
        edge = "#0B4F46" if pareto else "white"
        ax.scatter(float(row[field]), float(row["mask_ap"]), s=size, color=color, edgecolor=edge, linewidth=0.9, zorder=3)
        ax.annotate(
            str(row["experiment_id"]),
            (float(row[field]), float(row["mask_ap"])),
            xytext=offsets.get(field, {}).get(str(row["experiment_id"]), (4, 4)),
            textcoords="offset points",
            fontsize=7.5,
            fontweight="bold" if pareto else "normal",
        )
    legend_items = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#2F75B5", markeredgecolor="white", markersize=7, label="정확도 gate 통과"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#AEB7C2", markeredgecolor="white", markersize=7, label="정확도 gate 불통"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#157A6E", markeredgecolor="#0B4F46", markersize=9, label="Pareto"),
    ]
    if field == "median_ms":
        ax.axvline(REALTIME_BUDGET_MS, color="#C45A3C", linestyle="--", linewidth=1.2, label="30 FPS median 예산")
        legend_items.append(Line2D([0], [0], color="#C45A3C", linestyle="--", linewidth=1.2, label="30 FPS median 예산"))
    ax.legend(handles=legend_items, frameon=False, loc="lower left", ncol=2)
    style_axis(ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mask AP@[.50:.95]")
    ax.set_title("정확도 gate와 Pareto 후보", loc="left", fontweight="bold")
    save_figure(fig, path)


def plot_cv(rows: list[dict[str, Any]], path: Path) -> None:
    ordered = sorted(rows, key=lambda row: float(row["replicate_median_cv"]), reverse=True)
    ids = [str(row["experiment_id"]) for row in ordered]
    values = [100 * float(row["replicate_median_cv"]) for row in ordered]
    colors = ["#157A6E" if row["pareto_optimal"] else "#6E8EAD" for row in ordered]
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    ax.bar(ids, values, color=colors)
    ax.axhline(5.0, color="#C45A3C", linestyle="--", linewidth=1.2, label="재측정 기준 5%")
    style_axis(ax)
    ax.set_ylabel("반복 median CV (%)")
    ax.set_title("공식 지연시간 측정 안정성", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    ax.tick_params(axis="x", rotation=45)
    save_figure(fig, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()

    stage1 = load_json(resolve(root, STAGE1))
    state = load_json(resolve(root, STAGE2_STATE))
    stage3 = load_json(resolve(root, STAGE3))
    if stage3.get("objectives") != EXPECTED_OBJECTIVES:
        raise ValueError("Stage-3 objectives do not match the frozen paper protocol")
    if not stage3.get("stage2_all_candidates_terminal"):
        raise ValueError("Stage-2 is not terminal for every eligible candidate")
    if set(stage3.get("pareto_candidate_ids", [])) != {"C01", "R01"}:
        raise ValueError("Unexpected final Pareto set")

    decisions = build_decisions(stage1, state, stage3)
    if len(decisions) != 26:
        raise ValueError(f"Expected 26 candidate decisions, got {len(decisions)}")
    measured_rows = stage3["rows"]
    if len(measured_rows) != 21:
        raise ValueError(f"Expected 21 measured candidates, got {len(measured_rows)}")

    write_csv(resolve(root, DECISIONS), decisions)
    write_recommendations(resolve(root, RECOMMENDATIONS), state, stage3)
    write_report(resolve(root, REPORT), stage1, state, stage3, decisions)
    write_evidence_map(resolve(root, EVIDENCE_MAP))

    configure_font()
    figure_dir = resolve(root, FIGURE_DIR)
    plot_funnel(figure_dir / "stage3_candidate_funnel.png")
    plot_scatter(measured_rows, "median_ms", "Median latency (ms)", figure_dir / "stage3_mask_latency.png")
    engine_rows = [dict(row, engine_mib=float(row["engine_size_bytes"]) / 1048576) for row in measured_rows]
    plot_scatter(engine_rows, "engine_mib", "TensorRT engine size (MiB)", figure_dir / "stage3_mask_engine_size.png")
    plot_cv(measured_rows, figure_dir / "stage3_latency_cv.png")

    print(resolve(root, REPORT))
    print(resolve(root, DECISIONS))
    print(figure_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
