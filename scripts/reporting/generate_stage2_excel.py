#!/usr/bin/env python3
"""Create one auditable Excel workbook from Stage 1/2/3 result files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path("results/stage2-evaluation-report.xlsx")


def load_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def formula_cache(value: Any) -> Any:
    """XlsxWriter requires an empty string, not None, for a blank cached result."""
    value = clean(value)
    return "" if value is None else value


def flatten(prefix: str, value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from flatten(path, child)
    elif isinstance(value, list):
        yield prefix, json.dumps(value, ensure_ascii=False)
    else:
        yield prefix, value


def status_label(status: str) -> str:
    labels = {
        "completed": "완료",
        "build-failed": "엔진 생성 실패",
        "engine-analysis-failed": "엔진 검사 실패",
        "benchmark-failed": "속도 평가 실패",
        "accuracy-failed": "정확도 평가 실패",
        "running": "진행 중",
        "pending": "대기",
    }
    return labels.get(status, status)


def add_table(
    sheet: Any,
    headers: list[str],
    rows: list[list[Any]],
    name: str,
    *,
    start_row: int = 0,
    start_col: int = 0,
    style: str = "Table Style Medium 2",
) -> None:
    sheet.write_row(start_row, start_col, headers)
    for offset, row in enumerate(rows, start=1):
        sheet.write_row(start_row + offset, start_col, [clean(value) for value in row])
    if rows:
        sheet.add_table(
            start_row,
            start_col,
            start_row + len(rows),
            start_col + len(headers) - 1,
            {
                "name": name,
                "style": style,
                "columns": [{"header": header} for header in headers],
            },
        )


def build_report(root: Path, output: Path) -> None:
    try:
        import xlsxwriter
        from xlsxwriter.utility import xl_col_to_name
    except ImportError as error:
        raise RuntimeError(
            "XlsxWriter is required. Install the bundled requirements.txt first."
        ) from error

    stage1 = load_json(root / "results/stage1-static-evaluation.json")
    if not isinstance(stage1, dict):
        raise FileNotFoundError(root / "results/stage1-static-evaluation.json")
    summary = load_json(root / "results/stage2-notebook-summary.json", {}) or {}
    state = load_json(root / "results/stage2-notebook-state.json", {}) or {}
    pareto = load_json(root / "results/stage3-pareto.json", {}) or {}
    defaults = load_json(root / "configs/experiments/defaults.json", None)
    if defaults is None:
        import yaml

        defaults = yaml.safe_load(
            (root / "configs/experiments/defaults.yaml").read_text(encoding="utf-8")
        )
    import yaml

    registry = yaml.safe_load(
        (root / "configs/experiments/registry.yaml").read_text(encoding="utf-8")
    )["experiments"]

    stage1_rows = stage1.get("rows", [])
    result_map = {
        str(row["experiment_id"]): row for row in summary.get("rows", [])
    }
    state_map = state.get("candidates", {}) if isinstance(state, dict) else {}
    pareto_ids = set(pareto.get("pareto_candidate_ids", []))
    eligible_ids = [
        str(row["experiment_id"])
        for row in stage1_rows
        if row.get("stage2_notebook_eligible") is True
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.xlsx")
    temporary.unlink(missing_ok=True)
    workbook = xlsxwriter.Workbook(temporary)
    workbook.set_properties(
        {
            "title": "RF-DETR 경량화 2차 평가 결과",
            "subject": "TensorRT 정확도·지연시간·자원·Pareto 통합 결과",
            "author": "RF-DETR lightweighting evaluation pipeline",
            "comments": "Generated from immutable JSON/CSV evaluation evidence.",
        }
    )

    colors = {
        "navy": "#17365D",
        "blue": "#2F75B5",
        "teal": "#00A6A6",
        "green": "#70AD47",
        "red": "#C00000",
        "amber": "#FFC000",
        "light_blue": "#D9EAF7",
        "light_green": "#E2F0D9",
        "light_red": "#FCE4D6",
        "light_gray": "#F2F2F2",
        "white": "#FFFFFF",
        "text": "#1F2937",
    }
    title = workbook.add_format(
        {
            "bold": True,
            "font_size": 20,
            "font_color": colors["white"],
            "bg_color": colors["navy"],
            "align": "left",
            "valign": "vcenter",
        }
    )
    subtitle = workbook.add_format(
        {"font_size": 10, "font_color": "#5B6573", "text_wrap": True}
    )
    section = workbook.add_format(
        {
            "bold": True,
            "font_size": 12,
            "font_color": colors["white"],
            "bg_color": colors["blue"],
        }
    )
    card_label = workbook.add_format(
        {
            "bold": True,
            "font_color": "#44546A",
            "bg_color": colors["light_blue"],
            "align": "center",
            "border": 1,
            "border_color": "#B4C6E7",
        }
    )
    card_value = workbook.add_format(
        {
            "bold": True,
            "font_size": 16,
            "font_color": colors["navy"],
            "align": "center",
            "border": 1,
            "border_color": "#B4C6E7",
        }
    )
    header = workbook.add_format(
        {
            "bold": True,
            "font_color": colors["white"],
            "bg_color": colors["navy"],
            "align": "center",
            "valign": "vcenter",
            "text_wrap": True,
            "border": 1,
            "border_color": "#D9E2F3",
        }
    )
    percent = workbook.add_format({"num_format": "0.0%"})
    delta = workbook.add_format({"num_format": "+0.000;-0.000;0.000"})
    decimal3 = workbook.add_format({"num_format": "0.000"})
    decimal4 = workbook.add_format({"num_format": "0.0000"})
    one_decimal = workbook.add_format({"num_format": "0.0"})
    wrapped = workbook.add_format({"text_wrap": True, "valign": "top"})
    note = workbook.add_format(
        {"font_color": "#5B6573", "italic": True, "text_wrap": True}
    )

    # Evaluation criteria is created first because formula sheets reference it.
    criteria_sheet = workbook.add_worksheet("평가기준")
    criteria_sheet.hide_gridlines(2)
    criteria_sheet.freeze_panes(1, 0)
    criteria_sheet.set_column("A:A", 46)
    criteria_sheet.set_column("B:B", 24)
    criteria_sheet.set_column("C:C", 72)
    criteria_rows = [
        (
            "평가 데이터",
            defaults["evaluation_protocol"]["dataset_dir"],
            "누수 방지 후 고정한 test split만 정확도 평가에 사용",
        ),
        (
            "평가 이미지 수",
            defaults["evaluation_protocol"]["expected_images"],
            "모든 성공 후보에서 반드시 동일",
        ),
        (
            "COCO AP confidence 최소값",
            defaults["evaluation_protocol"]["coco_ap_confidence_threshold"],
            "점수 곡선을 보존하기 위한 낮은 수집 threshold",
        ),
        (
            "Semantic mIoU threshold",
            defaults["evaluation_protocol"]["semantic_miou_confidence_threshold"],
            "배포 운용점에서 클래스별 instance mask를 합집합",
        ),
        (
            "지연시간 표본 이미지",
            defaults["evaluation_protocol"]["latency"]["sampled_images"],
            "seed 42로 선택된 동일 이미지 집합",
        ),
        (
            "Warm-up / 반복당 측정",
            f"{defaults['evaluation_protocol']['latency']['warmup_runs']} / "
            f"{defaults['evaluation_protocol']['latency']['measured_runs']}",
            "GPU 동기화 전후를 포함한 end-to-end in-memory 측정",
        ),
        (
            "Benchmark 반복 횟수",
            defaults["evaluation_protocol"]["latency"]["benchmark_repetitions"],
            "반복별 warm-up 후 통계를 재측정",
        ),
        (
            "반복 median CV 경고선",
            defaults["evaluation_protocol"]["latency"][
                "replicate_median_cv_warning_threshold"
            ],
            "초과 시 제외하지 않고 재측정 필요 경고",
        ),
        (
            "BBox AP 허용 하락",
            defaults["selection"]["accuracy_vs_baseline"][
                "bbox_ap_max_absolute_drop"
            ],
            "B01 TensorRT 재측정값 대비 절대 하락 한계",
        ),
        (
            "Mask AP 허용 하락",
            defaults["selection"]["accuracy_vs_baseline"][
                "mask_ap_max_absolute_drop"
            ],
            "B01 TensorRT 재측정값 대비 절대 하락 한계",
        ),
        (
            "Semantic mIoU 허용 하락",
            defaults["selection"]["accuracy_vs_baseline"][
                "semantic_miou_max_absolute_drop"
            ],
            "B01 TensorRT 재측정값 대비 절대 하락 한계",
        ),
        (
            "Pareto 최대화",
            "BBox AP, Mask AP, mIoU",
            "정확도 보존 gate와 측정 유효성 통과 후 적용",
        ),
        (
            "Pareto 최소화",
            "Median, P95, GPU memory, engine size",
            "단일 가중합 점수는 사용하지 않음",
        ),
    ]
    add_table(
        criteria_sheet,
        ["항목", "고정값", "해석·재현 규칙"],
        [list(row) for row in criteria_rows],
        "EvaluationProtocol",
        style="Table Style Medium 2",
    )
    criteria_sheet.set_row(0, 34, header)
    criteria_cells = {
        "latency_cv": "$B$9",
        "bbox_drop": "$B$10",
        "mask_drop": "$B$11",
        "miou_drop": "$B$12",
    }
    criteria_sheet.write(15, 0, "보고 지표 정의", section)
    definitions = [
        ("BBox/Mask AP", "COCO AP@[IoU=.50:.05:.95], maxDets=100"),
        ("AP50 / AP75", "완화/엄격 IoU 조건의 위치·경계 품질"),
        ("AP small/medium/large", "객체 면적별 성능 편향 점검"),
        ("AR100", "이미지당 최대 100개 예측에서 평균 재현율"),
        ("Semantic mIoU", "클래스별 dataset intersection/union의 macro mean"),
        ("Median/P95/P99", "일반·tail latency를 함께 보고"),
        ("반복 median 95% CI", "3회 반복 median 평균의 t 구간"),
        ("GPU memory", "3회 측정 중 max allocated/reserved"),
    ]
    criteria_sheet.write_row(16, 0, ["지표", "정의"], header)
    for index, row in enumerate(definitions, start=17):
        criteria_sheet.write_row(index, 0, row)

    raw_sheet = workbook.add_worksheet("원시결과")
    raw_sheet.hide_gridlines(2)
    raw_sheet.freeze_panes(1, 5)
    raw_headers = [
        "ID",
        "경량화 계열",
        "방식",
        "정밀도",
        "입력크기",
        "1차 판정",
        "1차 불통 사유",
        "2차 상태",
        "2차 실패 사유",
        "BBox AP",
        "BBox AP50",
        "BBox AP75",
        "BBox AP-small",
        "BBox AP-medium",
        "BBox AP-large",
        "BBox AR100",
        "Mask AP",
        "Mask AP50",
        "Mask AP75",
        "Mask AP-small",
        "Mask AP-medium",
        "Mask AP-large",
        "Mask AR100",
        "Semantic mIoU",
        "Mean ms",
        "Median ms",
        "P95 ms",
        "P99 ms",
        "IQR ms",
        "전체 CV",
        "반복 Median CV",
        "반복 Median CI95 하한",
        "반복 Median CI95 상한",
        "FPS",
        "GPU allocated bytes",
        "GPU reserved bytes",
        "Engine bytes",
        "측정 반복 수",
        "총 latency 횟수",
        "Engine 경로",
        "평가 JSON",
        "Benchmark JSON 목록",
        "Test 이미지 수",
        "예측 수",
        "Engine SHA256",
        "Annotation SHA256",
    ]
    raw_rows: list[list[Any]] = []
    raw_index: dict[str, int] = {}
    for stage1_row in stage1_rows:
        experiment_id = str(stage1_row["experiment_id"])
        result = result_map.get(experiment_id, {})
        record = state_map.get(experiment_id, {})
        experiment = registry.get(experiment_id, {})
        stage2_status = record.get("status", "completed" if result else "pending")
        raw_index[experiment_id] = len(raw_rows) + 2
        raw_rows.append(
            [
                experiment_id,
                result.get("family") or experiment.get("family"),
                result.get("method") or experiment.get("method") or stage1_row.get("method_summary"),
                result.get("precision") or experiment.get("precision"),
                result.get("input_shape")
                or "x".join(
                    str(item)
                    for item in experiment.get(
                        "input_shape", defaults["export"]["input_shape"]
                    )
                ),
                "통과" if stage1_row.get("stage2_notebook_eligible") else "불통",
                stage1_row.get("failure_reason") or stage1_row.get("reason") or "",
                status_label(stage2_status),
                record.get("reason", ""),
                result.get("bbox_ap"),
                result.get("bbox_ap50"),
                result.get("bbox_ap75"),
                result.get("bbox_ap_small"),
                result.get("bbox_ap_medium"),
                result.get("bbox_ap_large"),
                result.get("bbox_ar100"),
                result.get("mask_ap"),
                result.get("mask_ap50"),
                result.get("mask_ap75"),
                result.get("mask_ap_small"),
                result.get("mask_ap_medium"),
                result.get("mask_ap_large"),
                result.get("mask_ar100"),
                result.get("semantic_miou"),
                result.get("mean_ms"),
                result.get("median_ms"),
                result.get("p95_ms"),
                result.get("p99_ms"),
                result.get("iqr_ms"),
                result.get("coefficient_of_variation"),
                result.get("replicate_median_cv"),
                result.get("replicate_median_ci95_low_ms"),
                result.get("replicate_median_ci95_high_ms"),
                result.get("fps"),
                result.get("gpu_peak_allocated_bytes"),
                result.get("gpu_peak_reserved_bytes"),
                result.get("engine_size_bytes"),
                result.get("benchmark_repetitions"),
                result.get("total_measured_runs"),
                result.get("engine"),
                result.get("evaluation"),
                result.get("benchmarks"),
                result.get("test_image_count"),
                result.get("prediction_count"),
                result.get("engine_sha256"),
                result.get("annotation_sha256"),
            ]
        )
    add_table(raw_sheet, raw_headers, raw_rows, "RawCandidateResults")
    raw_sheet.set_row(0, 42, header)
    raw_sheet.set_column("A:A", 9)
    raw_sheet.set_column("B:B", 16)
    raw_sheet.set_column("C:C", 38)
    raw_sheet.set_column("D:H", 14)
    raw_sheet.set_column("I:I", 42)
    raw_sheet.set_column("J:X", 13, decimal4)
    raw_sheet.set_column("Y:AH", 14, decimal3)
    raw_sheet.set_column("AI:AK", 18)
    raw_sheet.set_column("AL:AM", 14)
    raw_sheet.set_column("AN:AP", 46)
    raw_sheet.set_column("AQ:AR", 14)
    raw_sheet.set_column("AS:AT", 66)

    analysis_sheet = workbook.add_worksheet("분석결과")
    analysis_sheet.hide_gridlines(2)
    analysis_sheet.freeze_panes(1, 5)
    analysis_headers = [
        "ID",
        "계열",
        "경량화 방식",
        "정밀도",
        "2차 상태",
        "BBox AP",
        "Mask AP",
        "Semantic mIoU",
        "BBox Δ vs B01",
        "Mask Δ vs B01",
        "mIoU Δ vs B01",
        "Median ms",
        "P95 ms",
        "P99 ms",
        "FPS",
        "반복 Median CV",
        "CI95 하한 ms",
        "CI95 상한 ms",
        "GPU allocated MiB",
        "Engine MiB",
        "Latency 감소율",
        "Speedup",
        "Memory 감소율",
        "Engine 감소율",
        "측정 유효",
        "정확도 Gate",
        "Pareto 대상",
        "Pareto 최적",
        "재측정 경고",
        "제외 사유",
    ]
    analysis_sheet.write_row(0, 0, analysis_headers, header)
    for index, experiment_id in enumerate(eligible_ids, start=1):
        excel_row = index + 1
        raw_row = raw_index[experiment_id]
        result = result_map.get(experiment_id, {})
        record = state_map.get(experiment_id, {})
        completed = record.get("status", "completed" if result else "pending") == "completed" and bool(result)
        raw_columns = [0, 1, 2, 3, 7, 9, 16, 23]
        cached = [
            experiment_id,
            registry.get(experiment_id, {}).get("family", ""),
            registry.get(experiment_id, {}).get("method", ""),
            registry.get(experiment_id, {}).get("precision", ""),
            status_label(record.get("status", "completed" if result else "pending")),
            result.get("bbox_ap"),
            result.get("mask_ap"),
            result.get("semantic_miou"),
        ]
        for column, (raw_column, cached_value) in enumerate(zip(raw_columns, cached)):
            source = f"'원시결과'!{xl_col_to_name(raw_column)}{raw_row}"
            formula = f"={source}" if column < 5 else f'=IF({source}="","",{source})'
            analysis_sheet.write_formula(
                index, column, formula, None, formula_cache(cached_value)
            )

        delta_values = [
            result.get("bbox_ap_delta_vs_B01"),
            result.get("mask_ap_delta_vs_B01"),
            result.get("semantic_miou_delta_vs_B01"),
        ]
        for offset, (source_col, cached_value) in enumerate(zip((5, 6, 7), delta_values), start=8):
            letter = xl_col_to_name(source_col)
            formula = (
                f'=IF({letter}{excel_row}="","",{letter}{excel_row}-'
                f'INDEX({letter}$2:{letter}${len(eligible_ids)+1},'
                f'MATCH("B01",$A$2:$A${len(eligible_ids)+1},0)))'
            )
            analysis_sheet.write_formula(
                index, offset, formula, delta, formula_cache(cached_value)
            )

        linked_raw = {
            11: 25,
            12: 26,
            13: 27,
            14: 33,
            15: 30,
            16: 31,
            17: 32,
        }
        cached_fields = {
            11: "median_ms",
            12: "p95_ms",
            13: "p99_ms",
            14: "fps",
            15: "replicate_median_cv",
            16: "replicate_median_ci95_low_ms",
            17: "replicate_median_ci95_high_ms",
        }
        for target_col, raw_col in linked_raw.items():
            source = f"'원시결과'!{xl_col_to_name(raw_col)}{raw_row}"
            formula = f'=IF({source}="","",{source})'
            analysis_sheet.write_formula(
                index,
                target_col,
                formula,
                None,
                formula_cache(result.get(cached_fields[target_col])),
            )
        analysis_sheet.write_formula(
            index,
            18,
            f'=IF(\'원시결과\'!AI{raw_row}="","",\'원시결과\'!AI{raw_row}/1048576)',
            one_decimal,
            formula_cache(
                result.get("gpu_peak_allocated_bytes", 0) / 1048576
                if result.get("gpu_peak_allocated_bytes") is not None
                else None
            ),
        )
        analysis_sheet.write_formula(
            index,
            19,
            f'=IF(\'원시결과\'!AK{raw_row}="","",\'원시결과\'!AK{raw_row}/1048576)',
            one_decimal,
            formula_cache(
                result.get("engine_size_bytes", 0) / 1048576
                if result.get("engine_size_bytes") is not None
                else None
            ),
        )
        for target_col, field in (
            (20, "median_latency_reduction_vs_B01_pct"),
            (22, "gpu_memory_reduction_vs_B01_pct"),
            (23, "engine_size_reduction_vs_B01_pct"),
        ):
            cached_value = result.get(field)
            metric_col = {20: 11, 22: 18, 23: 19}[target_col]
            letter = xl_col_to_name(metric_col)
            formula = (
                f'=IF({letter}{excel_row}="","",('
                f'INDEX({letter}$2:{letter}${len(eligible_ids)+1},'
                f'MATCH("B01",$A$2:$A${len(eligible_ids)+1},0))-{letter}{excel_row})/'
                f'INDEX({letter}$2:{letter}${len(eligible_ids)+1},'
                f'MATCH("B01",$A$2:$A${len(eligible_ids)+1},0)))'
            )
            analysis_sheet.write_formula(
                index,
                target_col,
                formula,
                percent,
                formula_cache(cached_value / 100 if cached_value is not None else None),
            )
        speedup = result.get("speedup_vs_B01")
        analysis_sheet.write_formula(
            index,
            21,
            (
                f'=IF(L{excel_row}="","",INDEX(L$2:L${len(eligible_ids)+1},'
                f'MATCH("B01",$A$2:$A${len(eligible_ids)+1},0))/L{excel_row})'
            ),
            decimal3,
            formula_cache(speedup),
        )
        measurement_valid = bool(result.get("measurement_valid")) if completed else False
        accuracy_gate = bool(result.get("accuracy_gate_pass")) if completed else False
        pareto_eligible = bool(result.get("stage3_pareto_eligible")) if completed else False
        latency_warning = bool(result.get("latency_stability_warning")) if completed else False
        analysis_sheet.write(index, 24, measurement_valid)
        gate_formula = (
            f'=AND(I{excel_row}>=-\'평가기준\'!{criteria_cells["bbox_drop"]},'
            f'J{excel_row}>=-\'평가기준\'!{criteria_cells["mask_drop"]},'
            f'K{excel_row}>=-\'평가기준\'!{criteria_cells["miou_drop"]})'
        )
        analysis_sheet.write_formula(index, 25, gate_formula, None, accuracy_gate)
        analysis_sheet.write_formula(
            index,
            26,
            f"=AND(Y{excel_row},Z{excel_row})",
            None,
            pareto_eligible,
        )
        analysis_sheet.write(index, 27, experiment_id in pareto_ids)
        analysis_sheet.write_formula(
            index,
            28,
            f'=P{excel_row}>\'평가기준\'!{criteria_cells["latency_cv"]}',
            None,
            latency_warning,
        )
        analysis_sheet.write(index, 29, result.get("exclusion_reason", ""), wrapped)

    if eligible_ids:
        analysis_sheet.add_table(
            0,
            0,
            len(eligible_ids),
            len(analysis_headers) - 1,
            {
                "name": "AnalyzedCandidateResults",
                "style": "Table Style Medium 2",
                "columns": [{"header": item} for item in analysis_headers],
            },
        )
    analysis_sheet.set_row(0, 42, header)
    analysis_sheet.set_column("A:B", 11)
    analysis_sheet.set_column("C:C", 40)
    analysis_sheet.set_column("D:E", 15)
    analysis_sheet.set_column("F:H", 14, decimal4)
    analysis_sheet.set_column("I:K", 14, delta)
    analysis_sheet.set_column("L:O", 14, decimal3)
    analysis_sheet.set_column("P:P", 14, percent)
    analysis_sheet.set_column("Q:T", 16, decimal3)
    analysis_sheet.set_column("U:U", 15, percent)
    analysis_sheet.set_column("V:V", 12, decimal3)
    analysis_sheet.set_column("W:X", 15, percent)
    analysis_sheet.set_column("Y:AC", 13)
    analysis_sheet.set_column("AD:AD", 42, wrapped)
    analysis_sheet.conditional_format(
        1, 4, len(eligible_ids), 4, {"type": "text", "criteria": "containing", "value": "완료", "format": workbook.add_format({"bg_color": colors["light_green"]})}
    )
    analysis_sheet.conditional_format(
        1, 4, len(eligible_ids), 4, {"type": "text", "criteria": "containing", "value": "실패", "format": workbook.add_format({"bg_color": colors["light_red"], "font_color": colors["red"]})}
    )
    for column in (24, 25, 26, 27):
        analysis_sheet.conditional_format(
            1, column, len(eligible_ids), column, {"type": "cell", "criteria": "==", "value": True, "format": workbook.add_format({"bg_color": colors["light_green"], "font_color": "#375623"})}
        )
    analysis_sheet.conditional_format(
        1, 28, len(eligible_ids), 28, {"type": "cell", "criteria": "==", "value": True, "format": workbook.add_format({"bg_color": "#FFF2CC", "font_color": "#7F6000"})}
    )

    failure_sheet = workbook.add_worksheet("실패·제외")
    failure_sheet.hide_gridlines(2)
    failure_sheet.freeze_panes(1, 0)
    failure_rows: list[list[Any]] = []
    for row in stage1_rows:
        if row.get("stage2_notebook_eligible") is not True:
            failure_rows.append(
                [row["experiment_id"], "1차 정적평가", "불통", row.get("failure_reason") or row.get("reason") or "", "후보 산출물/근거 수정 후 1차 재평가"]
            )
    for item in summary.get("failures", []):
        failure_rows.append(
            [item["experiment_id"], "2차 노트북평가", status_label(item["status"]), item.get("reason", ""), "후보 로그 확인 후 동일 환경에서 재실행"]
        )
    for row in summary.get("rows", []):
        if row.get("measurement_valid") and not row.get("accuracy_gate_pass"):
            failure_rows.append(
                [row["experiment_id"], "Pareto 사전 gate", "제외", row.get("exclusion_reason", ""), "실측값은 보존하되 최종 Pareto 대상에서 제외"]
            )
    add_table(
        failure_sheet,
        ["ID", "단계", "상태", "불통·제외 사유", "후속 조치"],
        failure_rows,
        "FailuresAndExclusions",
        style="Table Style Medium 3",
    )
    failure_sheet.set_row(0, 36, header)
    failure_sheet.set_column("A:C", 18)
    failure_sheet.set_column("D:E", 64, wrapped)

    class_sheet = workbook.add_worksheet("클래스별 정확도")
    class_sheet.hide_gridlines(2)
    class_sheet.freeze_panes(1, 0)
    class_rows: list[list[Any]] = []
    for experiment_id, row in result_map.items():
        evaluation = load_json(root / row.get("evaluation", ""), {}) or {}
        metrics = evaluation.get("metrics", {})
        categories = evaluation.get("categories", {})
        bbox = metrics.get("bbox_by_category", {})
        mask = metrics.get("segm_by_category", {})
        iou = metrics.get("semantic_iou_by_category", {})
        for category_id in sorted(categories, key=lambda value: int(value)):
            class_rows.append(
                [
                    experiment_id,
                    int(category_id),
                    categories[category_id],
                    bbox.get(category_id, {}).get("ap"),
                    bbox.get(category_id, {}).get("ap50"),
                    bbox.get(category_id, {}).get("ap75"),
                    bbox.get(category_id, {}).get("ar100"),
                    mask.get(category_id, {}).get("ap"),
                    mask.get(category_id, {}).get("ap50"),
                    mask.get(category_id, {}).get("ap75"),
                    mask.get(category_id, {}).get("ar100"),
                    iou.get(category_id),
                ]
            )
    add_table(
        class_sheet,
        ["ID", "Class ID", "Class", "BBox AP", "BBox AP50", "BBox AP75", "BBox AR100", "Mask AP", "Mask AP50", "Mask AP75", "Mask AR100", "Semantic IoU"],
        class_rows,
        "PerClassAccuracy",
    )
    class_sheet.set_row(0, 36, header)
    class_sheet.set_column("A:C", 16)
    class_sheet.set_column("D:L", 14, decimal4)

    repeat_sheet = workbook.add_worksheet("반복측정")
    repeat_sheet.hide_gridlines(2)
    repeat_sheet.freeze_panes(1, 0)
    repeat_rows: list[list[Any]] = []
    for experiment_id, row in result_map.items():
        for repetition, relative in enumerate(row.get("benchmarks", []), start=1):
            payload = load_json(root / relative, {}) or {}
            repeat_rows.append(
                [
                    experiment_id,
                    repetition,
                    payload.get("image_count"),
                    payload.get("warmup_runs"),
                    payload.get("measured_runs"),
                    payload.get("mean_ms"),
                    payload.get("median_ms"),
                    payload.get("p95_ms"),
                    payload.get("p99_ms"),
                    payload.get("iqr_ms"),
                    payload.get("coefficient_of_variation"),
                    payload.get("fps"),
                    payload.get("gpu_peak_allocated_bytes"),
                    payload.get("gpu_peak_reserved_bytes"),
                    payload.get("model_sha256"),
                    payload.get("images"),
                ]
            )
    add_table(
        repeat_sheet,
        ["ID", "반복", "표본 이미지", "Warm-up", "측정 횟수", "Mean ms", "Median ms", "P95 ms", "P99 ms", "IQR ms", "CV", "FPS", "GPU allocated bytes", "GPU reserved bytes", "Engine SHA256", "고정 이미지 목록"],
        repeat_rows,
        "LatencyRepetitions",
    )
    repeat_sheet.set_row(0, 36, header)
    repeat_sheet.set_column("A:E", 13)
    repeat_sheet.set_column("F:J", 14, decimal3)
    repeat_sheet.set_column("K:K", 12, percent)
    repeat_sheet.set_column("L:N", 18)
    repeat_sheet.set_column("O:O", 66)
    repeat_sheet.set_column("P:P", 80, wrapped)

    environment_sheet = workbook.add_worksheet("실험환경")
    environment_sheet.hide_gridlines(2)
    environment_sheet.freeze_panes(1, 0)
    environment_source = {
        "state": {
            key: state.get(key)
            for key in (
                "created_at_utc",
                "finished_at_utc",
                "status",
                "gpu",
                "compute_capability",
                "cuda",
                "tensorrt",
                "nvidia_driver_and_bus",
                "benchmark_repetitions",
            )
        },
        "preflight": summary.get("preflight") or state.get("preflight", {}),
        "environment": summary.get("environment") or state.get("environment", {}),
        "load_stabilization_protocol": (
            summary.get("load_stabilization_protocol")
            or state.get("load_stabilization_protocol", {})
        ),
        "load_stabilization_reference": (
            summary.get("load_stabilization_reference")
            or state.get("load_stabilization_reference")
        ),
    }
    environment_rows = [list(item) for item in flatten("", environment_source)]
    add_table(
        environment_sheet,
        ["환경 항목", "기록값"],
        environment_rows,
        "ExperimentEnvironment",
    )
    environment_sheet.set_row(0, 34, header)
    environment_sheet.set_column("A:A", 48)
    environment_sheet.set_column("B:B", 88, wrapped)

    chart_data = workbook.add_worksheet("_차트데이터")
    chart_data.hide()
    chart_data.write_row(0, 0, ["ID", "Median ms", "Mask AP", "Engine MiB"])
    completed_for_chart = [
        experiment_id for experiment_id in eligible_ids if experiment_id in result_map
    ]
    for chart_row, experiment_id in enumerate(completed_for_chart, start=1):
        analysis_excel_row = eligible_ids.index(experiment_id) + 2
        result = result_map[experiment_id]
        chart_data.write_formula(
            chart_row, 0, f"='분석결과'!A{analysis_excel_row}", None, experiment_id
        )
        chart_data.write_formula(
            chart_row,
            1,
            f"='분석결과'!L{analysis_excel_row}",
            None,
            formula_cache(result.get("median_ms")),
        )
        chart_data.write_formula(
            chart_row,
            2,
            f"='분석결과'!G{analysis_excel_row}",
            None,
            formula_cache(result.get("mask_ap")),
        )
        chart_data.write_formula(
            chart_row,
            3,
            f"='분석결과'!T{analysis_excel_row}",
            None,
            formula_cache(
                result.get("engine_size_bytes", 0) / 1048576
                if result.get("engine_size_bytes") is not None
                else None
            ),
        )

    dashboard = workbook.add_worksheet("요약")
    dashboard.hide_gridlines(2)
    dashboard.set_tab_color(colors["teal"])
    dashboard.set_column("A:A", 3)
    dashboard.set_column("B:I", 16)
    dashboard.set_column("J:J", 3)
    dashboard.set_column("K:R", 14)
    dashboard.set_row(0, 34)
    dashboard.merge_range("B2:R3", "RF-DETR 경량화 2차 평가 통합 보고서", title)
    dashboard.merge_range(
        "B4:R5",
        "엔진 생성부터 고정 test 정확도, 3회 반복 latency, 자원 사용량, "
        "정확도 보존 gate와 Pareto 결과까지 한 파일에서 추적합니다.",
        subtitle,
    )
    completed_count = len(result_map)
    failure_count = len(summary.get("failures", []))
    pending_count = max(len(eligible_ids) - completed_count - failure_count, 0)
    gate_count = sum(bool(row.get("accuracy_gate_pass")) for row in result_map.values())
    cards = [
        ("1차 전체 후보", len(stage1_rows)),
        ("2차 평가 대상", len(eligible_ids)),
        ("2차 완료", completed_count),
        ("2차 실패", failure_count),
        ("대기", pending_count),
        ("정확도 Gate 통과", gate_count),
        ("Pareto 최적", len(pareto_ids)),
    ]
    for index, (label, value) in enumerate(cards):
        column = 1 + index * 2
        dashboard.merge_range(6, column, 6, column + 1, label, card_label)
        dashboard.merge_range(7, column, 8, column + 1, value, card_value)
    dashboard.merge_range("B11:I11", "판정 원칙", section)
    dashboard.merge_range(
        "B12:I16",
        "1) 1차 정적평가 통과 후보 22개를 모두 실측합니다.\n"
        "2) 엔진 생성 실패도 terminal 결과로 보존합니다.\n"
        "3) 정확도는 고정 test 437장, 속도는 동일 32장 × 200회 × 3반복입니다.\n"
        "4) B01 대비 정확도 보존 gate를 통과한 유효 측정만 Pareto에 투입합니다.\n"
        "5) Pareto는 정확도 3종을 최대화하고 median/P95·메모리·크기를 최소화합니다.",
        wrapped,
    )
    dashboard.merge_range("K11:R11", "파일 내 시트", section)
    sheet_notes = [
        ("분석결과", "논문용 비교와 B01 대비 변화·gate·Pareto"),
        ("원시결과", "1차/2차 후보 원시 집계값"),
        ("클래스별 정확도", "클래스별 bbox/mask AP·AR와 semantic IoU"),
        ("반복측정", "반복별 latency와 환경 추적값"),
        ("실패·제외", "정적 불통·엔진 실패·gate 제외 사유"),
        ("평가기준", "고정 지표·threshold·판정 규칙"),
        ("실험환경", "GPU/CUDA/TensorRT/Python/checksum 기록"),
    ]
    for row_index, (sheet_name, description) in enumerate(sheet_notes, start=11):
        dashboard.write_url(row_index, 10, f"internal:'{sheet_name}'!A1", string=sheet_name)
        dashboard.merge_range(row_index, 11, row_index, 17, description)

    if completed_count:
        first = 2
        last = completed_count + 1
        scatter = workbook.add_chart({"type": "scatter", "subtype": "straight_with_markers"})
        scatter.add_series(
            {
                "name": "Mask AP vs Median latency",
                "categories": f"='_차트데이터'!$B${first}:$B${last}",
                "values": f"='_차트데이터'!$C${first}:$C${last}",
                "marker": {"type": "circle", "size": 6, "border": {"color": colors["blue"]}, "fill": {"color": colors["blue"]}},
                "line": {"none": True},
            }
        )
        scatter.set_title({"name": "정확도–지연시간 절충"})
        scatter.set_x_axis({"name": "Median latency (ms)", "major_gridlines": {"visible": False}})
        scatter.set_y_axis({"name": "Mask AP", "num_format": "0.000"})
        scatter.set_legend({"none": True})
        scatter.set_style(10)
        dashboard.insert_chart("B19", scatter, {"x_scale": 1.15, "y_scale": 1.1})

        size_chart = workbook.add_chart({"type": "column"})
        size_chart.add_series(
            {
                "name": "Engine MiB",
                "categories": f"='_차트데이터'!$A${first}:$A${last}",
                "values": f"='_차트데이터'!$D${first}:$D${last}",
                "fill": {"color": colors["teal"]},
                "border": {"none": True},
            }
        )
        size_chart.set_title({"name": "TensorRT 엔진 크기"})
        size_chart.set_y_axis({"name": "MiB", "major_gridlines": {"visible": True, "line": {"color": "#E7E6E6"}}})
        size_chart.set_legend({"none": True})
        size_chart.set_style(10)
        dashboard.insert_chart("K19", size_chart, {"x_scale": 1.15, "y_scale": 1.1})

    dashboard.activate()
    workbook.close()
    temporary.replace(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    build_report(root, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
