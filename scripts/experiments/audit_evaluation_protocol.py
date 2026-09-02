#!/usr/bin/env python3
"""Audit dataset, evaluation, selection, and automation evidence.

The report separates implementation integrity from candidate gate outcomes:
an intentionally rejected candidate is not an audit failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = Path("configs/experiments/defaults.yaml")
REGISTRY = Path("configs/experiments/registry.yaml")
SPLIT_AUDIT = Path("docs/reports/metrics/dataset-split-audit.json")
BASELINE_RESULT = Path(
    "results/coco-evaluation/front-rfdetr-seg-large-v1-test.json"
)


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def portable(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def metric(payload: dict[str, Any], name: str) -> float:
    metrics = payload.get("metrics", {})
    if name == "bbox_ap":
        value = metrics.get("bbox", {}).get("ap")
    elif name == "mask_ap":
        value = metrics.get("segm", {}).get("ap")
    elif name == "semantic_miou":
        value = metrics.get("semantic_miou")
    else:
        raise KeyError(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Missing numeric metric {name}")
    if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"Metric outside [0,1]: {name}={value}")
    return float(value)


def candidate_rejected(experiment: dict[str, Any] | None) -> bool:
    """Return whether a measured failure is an expected candidate outcome."""
    if not isinstance(experiment, dict):
        return False
    result = experiment.get("result")
    return "rejected" in str(experiment.get("status", "")) or (
        isinstance(result, dict) and result.get("decision") == "rejected"
    )


def audit(root: Path) -> dict[str, Any]:
    defaults = load_yaml(resolve(root, DEFAULTS))
    protocol = defaults["evaluation_protocol"]
    selection = defaults["selection"]
    registry = load_yaml(resolve(root, REGISTRY))["experiments"]
    checks: list[dict[str, Any]] = []
    gate_outcomes: list[dict[str, Any]] = []

    def check(
        identifier: str,
        status: str,
        evidence: str,
        *,
        category: str,
        severity: str = "required",
    ) -> None:
        checks.append(
            {
                "id": identifier,
                "category": category,
                "status": status,
                "severity": severity,
                "evidence": evidence,
            }
        )

    split_path = resolve(root, SPLIT_AUDIT)
    split = load_json(split_path)
    verification = split["verification"]
    split_pass = (
        verification.get("source_augmentation_free") is True
        and verification.get("exact_cross_split_duplicate_images") == 0
        and verification.get("capture_session_overlap_count") == 0
        and verification.get("coco_reference_errors") == 0
        and verification.get("paper_evaluation_ready") is True
    )
    check(
        "dataset-leakage-integrity",
        "PASS" if split_pass else "FAIL",
        (
            f"augmentation-free={verification.get('source_augmentation_free')}; "
            f"cross-split duplicates={verification.get('exact_cross_split_duplicate_images')}; "
            f"session overlap={verification.get('capture_session_overlap_count')}; "
            f"COCO reference errors={verification.get('coco_reference_errors')}"
        ),
        category="dataset",
    )

    dataset = resolve(root, protocol["dataset_dir"])
    annotation = dataset / "_annotations.coco.json"
    annotation_data = load_json(annotation)
    actual_images = len(annotation_data.get("images", []))
    check(
        "evaluation-image-count",
        "PASS" if actual_images == protocol["expected_images"] else "FAIL",
        f"configured={protocol['expected_images']}; actual COCO images={actual_images}",
        category="dataset",
    )
    split_counts = {
        item["split"]: item["images"] for item in split.get("split_summary", [])
    }
    total_exported = int(split.get("dataset", {}).get("total_exported_images", 0))
    check(
        "split-ratio-realization",
        "PASS",
        (
            f"train={split_counts.get('train')} ({split_counts.get('train', 0) / total_exported:.2%}), "
            f"valid={split_counts.get('valid')} ({split_counts.get('valid', 0) / total_exported:.2%}), "
            f"test={split_counts.get('test')} ({split_counts.get('test', 0) / total_exported:.2%})"
        ),
        category="dataset",
    )

    baseline_path = resolve(root, BASELINE_RESULT)
    baseline = load_json(baseline_path)
    evaluation_paths: dict[str, Path] = {"B01": baseline_path}
    # Every registered deployable candidate is required to reach Stage 1.
    # W01 is a multi-configuration research study and is audited through the
    # consolidated Stage-1 report below rather than an ONNX graph report.
    static_candidate_ids = set(registry) - {"W01"}
    for experiment_id, experiment in registry.items():
        if experiment_id not in static_candidate_ids:
            continue
        result = experiment.get("result")
        if isinstance(result, dict) and result.get("evaluation"):
            evaluation_paths[experiment_id] = resolve(root, result["evaluation"])

    evaluation_records: list[dict[str, Any]] = []
    expected_annotation_hash = sha256(annotation)
    for experiment_id, path in sorted(evaluation_paths.items()):
        if not path.is_file():
            check(
                f"evaluation-{experiment_id}",
                "FAIL",
                f"missing {portable(root, path)}",
                category="accuracy",
            )
            continue
        result = load_json(path)
        failures = []
        if result.get("image_count") != protocol["expected_images"]:
            failures.append("image_count")
        if result.get("threshold") != protocol["coco_ap_confidence_threshold"]:
            failures.append("AP threshold")
        if result.get("miou_threshold") != protocol["semantic_miou_confidence_threshold"]:
            failures.append("mIoU threshold")
        result_dataset = result.get("dataset")
        if not result_dataset or resolve(root, result_dataset).resolve() != dataset.resolve():
            failures.append("dataset path")
        try:
            metrics = {
                name: metric(result, name)
                for name in ("bbox_ap", "mask_ap", "semantic_miou")
            }
        except ValueError as error:
            failures.append(str(error))
            metrics = {}
        embedded_annotation_hash = result.get("annotation_sha256")
        if embedded_annotation_hash and embedded_annotation_hash != expected_annotation_hash:
            failures.append("annotation SHA-256")
        check(
            f"evaluation-{experiment_id}",
            "FAIL" if failures else "PASS",
            (
                f"{portable(root, path)}; "
                + ("mismatch: " + ", ".join(failures) if failures else "protocol and metrics valid")
            ),
            category="accuracy",
        )
        if not embedded_annotation_hash:
            check(
                f"evaluation-provenance-{experiment_id}",
                "WARN",
                (
                    "legacy result lacks embedded annotation SHA-256; current "
                    f"annotation independently hashes to {expected_annotation_hash}"
                ),
                category="reproducibility",
                severity="advisory",
            )
        evaluation_records.append(
            {
                "experiment_id": experiment_id,
                "path": portable(root, path),
                "sha256": sha256(path),
                "metrics": metrics,
            }
        )

    for path in sorted((root / "results/coco-evaluation").glob("*-onnx.json")):
        if "invalid" in path.name or "superseded" in path.name:
            continue
        result = load_json(path)
        experiment_id = str(result.get("experiment_id") or path.name.split("-")[0])
        rejected_candidate = candidate_rejected(registry.get(experiment_id))
        failures = []
        if result.get("postprocess_num_select") != 200:
            failures.append("postprocess_num_select != 200")
        parity = result.get("pytorch_parity")
        parity_failed = isinstance(parity, dict) and parity.get("passed") is not True
        if parity_failed and not rejected_candidate:
            failures.append("PyTorch dataset parity failed")
        if path.name.endswith("after-recovery-onnx.json") and not isinstance(parity, dict):
            failures.append("missing PyTorch dataset parity")
        check(
            f"onnx-evaluation-{experiment_id}-{path.stem}",
            "FAIL" if failures else "PASS",
            f"{portable(root, path)}; "
            + (
                ", ".join(failures)
                if failures
                else (
                    "candidate decision=rejected; dataset parity failure recorded"
                    if parity_failed and rejected_candidate
                    else "query count/parity valid"
                )
            ),
            category="conversion",
        )

    equivalence_protocol = protocol["graph_equivalence"]
    equivalence_ids = {
        "B01",
        "R01",
        "S01",
        "S02",
        "U02",
        "S03",
        "M01",
    }
    equivalence_ids.update(
        path.parents[1].name
        for path in (root / "artifacts/experiments").glob(
            "*/front/onnx-equivalence.json"
        )
    )
    for experiment_id in sorted(equivalence_ids):
        path = root / f"artifacts/experiments/{experiment_id}/front/onnx-equivalence.json"
        experiment = registry[experiment_id]
        rejected_candidate = candidate_rejected(experiment)
        active_recovery = (
            experiment_id == "M01"
            and not experiment.get("fine_tuning", {}).get("completed")
        )
        if not path.is_file():
            check(
                f"graph-equivalence-{experiment_id}",
                "PENDING" if active_recovery else "FAIL",
                f"missing {portable(root, path)}",
                category="conversion",
            )
            continue
        result = load_json(path)
        protocol_match = (
            result.get("samples") == equivalence_protocol["samples"]
            and result.get("seed") == equivalence_protocol["sample_seed"]
            and result.get("input_mode") == "real-images"
            and result.get("production_comparison", {}).get("confidence_threshold")
            == equivalence_protocol["confidence_threshold"]
        )
        if active_recovery:
            status = "PENDING"
            evidence = "prototype report is superseded when recovery completes"
        elif rejected_candidate:
            status = "PASS" if protocol_match else "FAIL"
            evidence = (
                f"candidate decision=rejected; equivalence passed={result.get('passed')}; "
                f"protocol match={protocol_match}"
            )
        else:
            status = "PASS" if protocol_match and result.get("passed") is True else "FAIL"
            evidence = (
                f"passed={result.get('passed')}; samples={result.get('samples')}; "
                f"seed={result.get('seed')}; mode={result.get('input_mode')}; "
                f"protocol match={protocol_match}"
            )
        check(
            f"graph-equivalence-{experiment_id}",
            status,
            f"{portable(root, path)}; {evidence}",
            category="conversion",
        )
        production = result.get("production_comparison", {})
        strict_fields = (
            "active_membership_agreement",
            "active_class_agreement",
            "active_score_max_abs_error",
        )
        if (
            status == "PASS"
            and not rejected_candidate
            and not all(field in production for field in strict_fields)
        ):
            check(
                f"graph-equivalence-schema-{experiment_id}",
                "WARN",
                (
                    "legacy graph report predates strict active-membership/class/score "
                    "fields; full-dataset ONNX accuracy parity is available"
                ),
                category="conversion",
                severity="advisory",
            )

    for experiment_id, experiment in registry.items():
        if experiment_id not in static_candidate_ids:
            continue
        if "front" not in experiment.get("cameras", []):
            continue
        own_onnx_path = root / f"artifacts/experiments/{experiment_id}/front/model.onnx"
        source_onnx_path = root / f"artifacts/experiments/{experiment.get('artifact_source', experiment_id)}/front/model.onnx"
        onnx_path = own_onnx_path if own_onnx_path.is_file() else source_onnx_path
        static_path = root / f"artifacts/experiments/{experiment_id}/front/static-analysis.json"
        if not onnx_path.is_file():
            continue
        if not static_path.is_file():
            blocked = experiment.get("status") == "blocked"
            status = (
                "PASS"
                if blocked
                else "PENDING" if experiment_id in {"M01", "M02"} else "FAIL"
            )
            check(
                f"static-analysis-{experiment_id}",
                status,
                (
                    f"not required because upstream candidate is blocked; "
                    f"missing {portable(root, static_path)}"
                    if blocked
                    else f"missing {portable(root, static_path)}"
                ),
                category="static-analysis",
            )
            continue
        static = load_json(static_path)
        onnx_report = static.get("onnx", {})
        failures = []
        if onnx_report.get("onnx_checker_valid") is not True:
            failures.append("ONNX checker")
        if onnx_report.get("flops_status") not in {"partial", "partial-lower-bound"}:
            failures.append("FLOP scope label")
        stale = static_path.stat().st_mtime_ns < onnx_path.stat().st_mtime_ns
        if stale:
            failures.append("report older than ONNX")
        active_recovery = experiment_id == "M01" and not experiment.get("fine_tuning", {}).get("completed")
        status = "PENDING" if active_recovery else "FAIL" if failures else "PASS"
        check(
            f"static-analysis-{experiment_id}",
            status,
            (
                f"{portable(root, static_path)}; status={onnx_report.get('flops_status')}; "
                f"resolved Conv/MatMul/Gemm coverage={onnx_report.get('compute_node_coverage')}; "
                f"estimated/all-node fraction={onnx_report.get('estimated_node_fraction_of_graph', 'awaiting refresh')}"
                + (f"; issues={','.join(failures)}" if failures else "")
            ),
            category="static-analysis",
        )

    stage1_path = root / "results/stage1-static-evaluation.json"
    if not stage1_path.is_file():
        check(
            "stage1-all-candidate-coverage",
            "FAIL",
            f"missing {portable(root, stage1_path)}",
            category="static-analysis",
        )
    else:
        stage1 = load_json(stage1_path)
        rows = stage1.get("rows", [])
        row_ids = {
            row.get("experiment_id")
            for row in rows
            if isinstance(row, dict) and row.get("experiment_id")
        }
        missing_ids = sorted(set(registry) - row_ids)
        extra_ids = sorted(row_ids - set(registry))
        unperformed = int(stage1.get("unperformed_count", -1))
        protocol_static_only = (
            stage1.get("protocol", {}).get("accuracy_used_as_stage1_gate") is False
        )
        valid = not missing_ids and not extra_ids and unperformed == 0 and protocol_static_only
        check(
            "stage1-all-candidate-coverage",
            "PASS" if valid else "FAIL",
            (
                f"registered={len(registry)}; rows={len(row_ids)}; "
                f"unperformed={unperformed}; missing={missing_ids or 'none'}; "
                f"extra={extra_ids or 'none'}; static-only={protocol_static_only}"
            ),
            category="static-analysis",
        )

    efficiency_policy = selection.get("first_stage_efficiency", {})
    efficiency_metrics = {
        "onnx_size": (
            "onnx_size_bytes",
            "onnx_size_min_relative_reduction",
        ),
        "onnx_nodes": (
            "onnx_nodes",
            "onnx_nodes_min_relative_reduction",
        ),
        "dense_macs": (
            "estimated_macs",
            "dense_macs_min_relative_reduction",
        ),
    }
    if efficiency_policy:
        graph_efficiency_ids = {
            experiment_id
            for experiment_id in static_candidate_ids
            if registry[experiment_id].get("method")
            in {
                "global-magnitude",
                "decoder-layer",
                "ffn-dimension",
                "input-resolution",
                "structured-resolution",
            }
        }
        for experiment_id in sorted(graph_efficiency_ids):
            comparison_path = (
                root
                / f"artifacts/experiments/{experiment_id}/front/comparison-B01.json"
            )
            if not comparison_path.is_file():
                gate_outcomes.append(
                    {
                        "experiment_id": experiment_id,
                        "comparison": "efficiency_vs_B01",
                        "outcome": "PENDING",
                        "reason": "static comparison is missing",
                    }
                )
                continue
            comparison = load_json(comparison_path)
            gates: dict[str, bool] = {}
            observed_reduction: dict[str, float | None] = {}
            for gate_name, (metric_name, policy_name) in efficiency_metrics.items():
                ratio = comparison.get(metric_name, {}).get("delta_ratio")
                valid_ratio = (
                    float(ratio)
                    if isinstance(ratio, (int, float)) and not isinstance(ratio, bool)
                    else None
                )
                observed_reduction[gate_name] = (
                    -valid_ratio if valid_ratio is not None else None
                )
                gates[gate_name] = bool(
                    valid_ratio is not None
                    and valid_ratio <= -float(efficiency_policy[policy_name])
                )
            rule = efficiency_policy.get("rule", "any")
            passed = all(gates.values()) if rule == "all" else any(gates.values())
            gate_outcomes.append(
                {
                    "experiment_id": experiment_id,
                    "comparison": "efficiency_vs_B01",
                    "outcome": "PASS" if passed else "REJECT",
                    "rule": rule,
                    "metric_gates": gates,
                    "observed_relative_reduction": observed_reduction,
                }
            )

    baseline_metrics = {
        name: metric(baseline, name)
        for name in ("bbox_ap", "mask_ap", "semantic_miou")
    }
    base_policy = selection["accuracy_vs_baseline"]
    for record in evaluation_records:
        if record["experiment_id"] == "B01" or not record["metrics"]:
            continue
        gates = {
            "bbox_ap": record["metrics"]["bbox_ap"]
            >= baseline_metrics["bbox_ap"] - base_policy["bbox_ap_max_absolute_drop"],
            "mask_ap": record["metrics"]["mask_ap"]
            >= baseline_metrics["mask_ap"] - base_policy["mask_ap_max_absolute_drop"],
            "semantic_miou": record["metrics"]["semantic_miou"]
            >= baseline_metrics["semantic_miou"] - base_policy["semantic_miou_max_absolute_drop"],
        }
        gate_outcomes.append(
            {
                "experiment_id": record["experiment_id"],
                "comparison": "desktop_preliminary_accuracy_vs_B01",
                "outcome": "PASS" if all(gates.values()) else "REJECT",
                "metric_gates": gates,
                "metric_delta_candidate_minus_reference": {
                    name: record["metrics"][name] - baseline_metrics[name]
                    for name in gates
                },
            }
        )

    queue_path = root / "results/desktop-tensorrt-pipeline.json"
    if queue_path.is_file():
        queue = load_json(queue_path)
        queue_protocol = queue.get("evaluation_protocol")
        check(
            "desktop-pipeline-protocol-binding",
            "PASS" if queue_protocol == protocol else "PENDING",
            (
                f"pipeline status={queue.get('status')}; "
                + (
                    "running queue records the canonical protocol"
                    if queue_protocol == protocol
                    else "running queue predates canonical protocol and must be restarted"
                )
            ),
            category="automation",
        )
    else:
        check(
            "desktop-pipeline-protocol-binding",
            "PENDING",
            "pipeline state has not been created",
            category="automation",
        )
    post_queue_path = root / "artifacts/experiments/M01/front/post-recovery-queue.json"
    if post_queue_path.is_file():
        post_queue = load_json(post_queue_path)
        post_protocol_match = post_queue.get("evaluation_protocol") == protocol
        check(
            "post-recovery-protocol-binding",
            "PASS" if post_protocol_match else "FAIL",
            (
                f"M01 post-recovery status={post_queue.get('status')}; "
                f"canonical protocol match={post_protocol_match}"
            ),
            category="automation",
        )

    benchmark_files = sorted((root / "results/benchmarks/desktop").glob("*/*.json"))
    if benchmark_files:
        for path in benchmark_files:
            result = load_json(path)
            valid = (
                result.get("image_count") == protocol["latency"]["sampled_images"]
                and result.get("warmup_runs") == protocol["latency"]["warmup_runs"]
                and result.get("measured_runs") == protocol["latency"]["measured_runs"]
                and result.get("benchmark_scope")
            )
            check(
                f"latency-{path.parent.name}-{path.stem}",
                "PASS" if valid else "WARN",
                f"{portable(root, path)}; multi-image/statistical protocol={'valid' if valid else 'legacy or incomplete'}",
                category="latency",
                severity="advisory" if not valid else "required",
            )
    else:
        check(
            "latency-results",
            "PENDING",
            "retained candidates await separate TensorRT engine benchmarking",
            category="latency",
        )

    engine_summary_path = root / "results/desktop-engine-summary.json"
    if engine_summary_path.is_file():
        summary = load_json(engine_summary_path)
        runtime_rows = {
            row["experiment_id"]: row
            for row in summary.get("rows", [])
            if isinstance(row, dict) and row.get("experiment_id")
        }
        for experiment_id, row in runtime_rows.items():
            engine = resolve(root, row["engine"])
            engine_static = root / f"artifacts/experiments/{experiment_id}/front/engine-static-analysis.json"
            failures = []
            if not engine.is_file() or sha256(engine) != row.get("engine_sha256"):
                failures.append("engine hash")
            if not engine_static.is_file():
                failures.append("engine static analysis")
            elif load_json(engine_static).get("engine_sha256") != row.get("engine_sha256"):
                failures.append("engine static-analysis hash")
            check(
                f"engine-provenance-{experiment_id}",
                "FAIL" if failures else "PASS",
                f"{portable(root, engine)}; " + (", ".join(failures) if failures else "hash/static evidence valid"),
                category="engine",
            )

        selection_path = root / "results/desktop-structured-selection.json"
        if selection_path.is_file():
            selected = load_json(selection_path)
            policy_match = selected.get("policy") == selection["structured_comparison"]
            check(
                "structured-selection-policy",
                "PASS" if policy_match else "FAIL",
                f"selected={selected.get('selected_experiment')}; canonical policy match={policy_match}",
                category="selection",
            )

        def runtime_gate(name: str, gate: dict[str, Any]) -> None:
            candidate_id = gate["candidate"]
            accuracy_reference_id = gate["accuracy_reference"]
            latency_reference_id = gate["latency_reference"]
            required = {candidate_id, accuracy_reference_id, latency_reference_id}
            if not required.issubset(runtime_rows):
                gate_outcomes.append(
                    {
                        "experiment_id": candidate_id,
                        "comparison": name,
                        "outcome": "PENDING",
                        "reason": "required engine result is missing",
                    }
                )
                return
            candidate = runtime_rows[candidate_id]
            accuracy_reference = runtime_rows[accuracy_reference_id]
            latency_reference = runtime_rows[latency_reference_id]
            gates = {
                "bbox_ap": candidate["bbox_ap"]
                >= accuracy_reference["bbox_ap"] - gate["bbox_ap_max_absolute_drop"],
                "mask_ap": candidate["mask_ap"]
                >= accuracy_reference["mask_ap"] - gate["mask_ap_max_absolute_drop"],
                "latency": (
                    latency_reference["median_ms"] - candidate["median_ms"]
                )
                / latency_reference["median_ms"]
                >= gate["median_latency_min_relative_reduction"],
            }
            if gate.get("require_sparse_tactic_evidence"):
                tactic_path = root / f"artifacts/experiments/{candidate_id}/front/sparse-tactics.json"
                gates["sparse_tactic"] = bool(
                    tactic_path.is_file()
                    and load_json(tactic_path).get("sparse_tactic_selected") is True
                )
            gate_outcomes.append(
                {
                    "experiment_id": candidate_id,
                    "comparison": name,
                    "outcome": "PASS" if all(gates.values()) else "REJECT",
                    "metric_gates": gates,
                    "accuracy_reference": accuracy_reference_id,
                    "latency_reference": latency_reference_id,
                    "latency_reduction_fraction": (
                        latency_reference["median_ms"] - candidate["median_ms"]
                    )
                    / latency_reference["median_ms"],
                }
            )

        runtime_gate("int8_vs_fp16", selection["int8_vs_fp16"])
        runtime_gate(
            "sparse_vs_dense_control",
            selection["sparse_vs_dense_control"],
        )
    else:
        check(
            "engine-summary",
            "PENDING",
            "desktop engine summary will be generated during the second-stage engine evaluation",
            category="automation",
        )

    queue_source = (root / "scripts/experiments/run_desktop_tensorrt_queue.py").read_text(
        encoding="utf-8"
    )
    required_queue_stages = (
        "analyze_tensorrt_engine.py",
        "paper_results.py",
        "generate_paper_static_analysis.py",
        "audit_evaluation_protocol.py",
    )
    missing_queue_stages = [
        name for name in required_queue_stages if name not in queue_source
    ]
    check(
        "pipeline-stage-wiring",
        "FAIL" if missing_queue_stages else "PASS",
        (
            "missing: " + ", ".join(missing_queue_stages)
            if missing_queue_stages
            else "engine inspection, paper table/figure, static report, and final audit are wired"
        ),
        category="automation",
    )
    notebook_source = (
        root / "scripts/experiments/run_notebook_stage2.py"
    ).read_text(encoding="utf-8")
    notebook_required_tokens = (
        "stage1-static-evaluation.json",
        "stage2_notebook_eligible",
        "build-failed",
        "benchmark-failed",
        "accuracy-failed",
        "pending_candidates",
        "stage3-pareto.json",
        "dominates",
    )
    missing_notebook_tokens = [
        token for token in notebook_required_tokens if token not in notebook_source
    ]
    package_source = (
        root / "scripts/experiments/package_notebook_bundle.py"
    ).read_text(encoding="utf-8")
    suite_source = (
        root / "scripts/experiments/build_engine_suite.py"
    ).read_text(encoding="utf-8")
    all_candidate_wiring = (
        not missing_notebook_tokens
        and '"stage1"' in package_source
        and '"stage1"' in suite_source
    )
    check(
        "all-candidate-notebook-pareto-wiring",
        "PASS" if all_candidate_wiring else "FAIL",
        (
            "Stage-1 eligible set -> per-candidate terminal Stage 2 -> Pareto gate is wired"
            if all_candidate_wiring
            else "missing automation tokens: " + ", ".join(missing_notebook_tokens)
        ),
        category="automation",
    )
    reporting_source = (root / "scripts/reporting/paper_results.py").read_text(
        encoding="utf-8"
    )
    report_source_valid = (
        'SOURCE = Path("results/desktop-engine-summary.json")' in reporting_source
    )
    check(
        "paper-report-source",
        "PASS" if report_source_valid else "FAIL",
        "paper outputs consume only the current desktop engine summary",
        category="automation",
    )
    calibration_valid = (
        defaults["reproducibility"].get("calibration_image_count") == 128
        and defaults["reproducibility"].get("calibration_seed") == 42
    )
    check(
        "int8-calibration-binding",
        "PASS" if calibration_valid else "FAIL",
        (
            f"train-only deterministic calibration: count="
            f"{defaults['reproducibility'].get('calibration_image_count')}, "
            f"seed={defaults['reproducibility'].get('calibration_seed')}"
        ),
        category="automation",
    )
    check(
        "fixed-benchmark-reuse",
        "WARN",
        (
            "437 images are a repeated comparative benchmark, not an untouched "
            "confirmatory test; collect a new session for external-generalization claims"
        ),
        category="study-design",
        severity="advisory",
    )
    check(
        "training-seed-replication",
        "WARN",
        (
            "current recovery results use seed 42 only; report this as a deterministic "
            "engineering comparison or add multiple training seeds for variance estimates"
        ),
        category="study-design",
        severity="advisory",
    )

    required_failures = [
        item for item in checks
        if item["status"] == "FAIL" and item["severity"] == "required"
    ]
    pending = [item for item in checks if item["status"] == "PENDING"]
    warnings = [item for item in checks if item["status"] == "WARN"]
    if required_failures:
        overall = "FAIL"
    elif pending:
        overall = "IN_PROGRESS"
    elif warnings:
        overall = "PASS_WITH_ADVISORIES"
    else:
        overall = "PASS"
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall,
        "protocol_source": DEFAULTS.as_posix(),
        "protocol": protocol,
        "selection_gates": selection,
        "summary": {
            "checks": len(checks),
            "required_failures": len(required_failures),
            "pending": len(pending),
            "advisories": len(warnings),
        },
        "checks": checks,
        "candidate_gate_outcomes": gate_outcomes,
        "validity_note": (
            "The 437-image split has been used repeatedly during candidate "
            "development. It is valid as a fixed comparative benchmark but is "
            "not an untouched confirmatory test set. Claims of final external "
            "generalization require a new held-out capture session."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 평가 기준 및 자동화 감사 보고서",
        "",
        f"- 감사 상태: **{report['overall_status']}**",
        f"- 기준 원본: `{report['protocol_source']}`",
        f"- 검사 수: {report['summary']['checks']}",
        f"- 필수 실패: {report['summary']['required_failures']}",
        f"- 실행 대기: {report['summary']['pending']}",
        f"- 권고 사항: {report['summary']['advisories']}",
        "",
        "## 감사 판정",
        "",
        "| 영역 | 검사 | 상태 | 근거 |",
        "|---|---|---|---|",
    ]
    for item in report["checks"]:
        evidence = str(item["evidence"]).replace("|", "\\|")
        lines.append(
            f"| {item['category']} | {item['id']} | {item['status']} | {evidence} |"
        )
    lines.extend(
        [
            "",
            "## 사전 정의된 후보 수용 기준의 적용 결과",
            "",
            "`REJECT`는 자동화 결함이 아니라 해당 후보가 정확도 보존 기준을 통과하지 "
            "못했다는 뜻이다.",
            "",
            "| 후보 | 비교 | 판정 | Gate details |",
            "|---|---|---|---|",
        ]
    )
    for item in report["candidate_gate_outcomes"]:
        gates = json.dumps(
            item.get("metric_gates", {"reason": item.get("reason")}),
            ensure_ascii=False,
            sort_keys=True,
        ).replace("|", "\\|")
        lines.append(
            f"| {item['experiment_id']} | {item['comparison']} | {item['outcome']} | "
            f"{gates} |"
        )
    lines.extend(
        [
            "",
            "## 타당성 한계",
            "",
            report["validity_note"],
            "",
            "정적 MAC/FLOP는 Conv·MatMul·Gemm의 dense 산술만 포함한 하한 추정치다. "
            "TensorRT layer fusion, 메모리 이동, resize·normalization·activation 비용은 "
            "포함하지 않으므로 실제 속도는 동일 장비의 측정 latency로 판단한다.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("results/evaluation-automation-audit.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("docs/reports/evaluation-automation-audit.md"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    report = audit(root)
    json_output = resolve(root, args.json_output)
    markdown_output = resolve(root, args.markdown_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(f"status: {report['overall_status']}")
    print(f"json: {portable(root, json_output)}")
    print(f"markdown: {portable(root, markdown_output)}")
    return 1 if report["overall_status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
