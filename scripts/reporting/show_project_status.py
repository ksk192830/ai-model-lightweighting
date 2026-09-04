#!/usr/bin/env python3
"""Show the authoritative three-stage experiment progress."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TERMINAL_STAGE2_STATUSES = {
    "completed",
    "build-failed",
    "engine-analysis-failed",
    "benchmark-failed",
    "accuracy-failed",
}


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def collect_status(root: Path) -> dict[str, Any]:
    stage1 = read_json(root / "results/stage1-static-evaluation.json")
    if stage1 is None:
        return {
            "stage1": {"status": "NOT_STARTED"},
            "stage2": {"status": "BLOCKED", "reason": "Stage 1 report missing"},
            "stage3": {"status": "BLOCKED", "reason": "Stage 1 report missing"},
        }

    eligible = [
        str(row["experiment_id"])
        for row in stage1.get("rows", [])
        if row.get("stage2_notebook_eligible") is True
    ]
    stage2_state = read_json(root / "results/stage2-notebook-state.json")
    candidate_states = (
        stage2_state.get("candidates", {})
        if isinstance(stage2_state, dict)
        else {}
    )
    if not isinstance(candidate_states, dict):
        candidate_states = {}

    completed = sorted(
        experiment_id
        for experiment_id in eligible
        if candidate_states.get(experiment_id, {}).get("status") == "completed"
    )
    failed = sorted(
        experiment_id
        for experiment_id in eligible
        if candidate_states.get(experiment_id, {}).get("status")
        in TERMINAL_STAGE2_STATUSES - {"completed"}
    )
    running = sorted(
        experiment_id
        for experiment_id in eligible
        if candidate_states.get(experiment_id, {}).get("status") == "running"
    )
    terminal = set(completed) | set(failed)
    pending = sorted(set(eligible) - terminal - set(running))

    if len(terminal) == len(eligible):
        stage2_status = "COMPLETE"
    elif candidate_states or (
        stage2_state
        and stage2_state.get("status") in {"running", "paused-load-not-stable"}
    ):
        stage2_status = "RUNNING"
    else:
        stage2_status = "NOT_STARTED"

    pareto = read_json(root / "results/stage3-pareto.json")
    if pareto is not None and stage2_status == "COMPLETE":
        stage3_status = "COMPLETE"
    elif pareto is not None:
        stage3_status = "STALE"
    else:
        stage3_status = "WAITING"
    pareto_ids = pareto.get("pareto_candidate_ids", []) if pareto else []

    return {
        "stage1": {
            "status": (
                "COMPLETE" if int(stage1.get("unperformed_count", -1)) == 0 else "INCOMPLETE"
            ),
            "registered": int(stage1.get("candidate_count", 0)),
            "evaluated": int(stage1.get("evaluated_count", 0)),
            "passed": int(stage1.get("passed_count", 0)),
            "failed": int(stage1.get("evaluated_count", 0))
            - int(stage1.get("passed_count", 0)),
            "unperformed": int(stage1.get("unperformed_count", 0)),
        },
        "stage2": {
            "status": stage2_status,
            "eligible": len(eligible),
            "terminal": len(terminal),
            "completed": len(completed),
            "failed": len(failed),
            "running": running,
            "pending": pending,
            "measurement_mode": (
                stage2_state.get("measurement_mode") if stage2_state else None
            ),
            "progress": stage2_state.get("progress", {}) if stage2_state else {},
            "state_file": (
                "results/stage2-notebook-state.json" if stage2_state is not None else None
            ),
        },
        "stage3": {
            "status": stage3_status,
            "pareto_candidate_ids": pareto_ids,
            "result_file": "results/stage3-pareto.json" if pareto is not None else None,
        },
    }


def render(status: dict[str, Any]) -> str:
    stage1 = status["stage1"]
    stage2 = status["stage2"]
    stage3 = status["stage3"]
    lines = ["RF-DETR lightweighting project status"]
    if stage1["status"] == "NOT_STARTED":
        lines.append("Stage 1 static: NOT_STARTED")
    else:
        lines.append(
            "Stage 1 static: {status} | evaluated {evaluated}/{registered} | "
            "pass {passed} | fail {failed} | unperformed {unperformed}".format(**stage1)
        )
    if stage2["status"] == "BLOCKED":
        lines.append(f"Stage 2 notebook: BLOCKED | {stage2['reason']}")
    else:
        lines.append(
            "Stage 2 notebook: {status} | terminal {terminal}/{eligible} | "
            "completed {completed} | failed {failed}".format(**stage2)
        )
        if stage2["running"]:
            lines.append("  running: " + " ".join(stage2["running"]))
        if stage2["pending"]:
            lines.append("  pending: " + " ".join(stage2["pending"]))
        progress = stage2.get("progress", {})
        if progress and progress.get("total_units"):
            eta = progress.get("estimated_finish_at_utc")
            eta_local = (
                datetime.fromisoformat(eta).astimezone().strftime(
                    "%Y-%m-%d %H:%M:%S %Z"
                )
                if eta
                else "첫 측정 완료 후 계산"
            )
            current = progress.get("current_candidate") or "-"
            repetition = progress.get("current_repetition")
            repeat_text = f" / 반복 {repetition}" if repetition else ""
            lines.append(
                "  성능 재측정: {completed_units}/{total_units} ({percent:.1f}%) | "
                "현재 {current} / {stage}{repeat} | 예상 종료 {eta}".format(
                    **progress,
                    current=current,
                    stage=progress.get("current_stage", "-"),
                    repeat=repeat_text,
                    eta=eta_local,
                )
            )
    if stage3["status"] == "COMPLETE":
        lines.append(
            "Stage 3 Pareto: COMPLETE | "
            + " ".join(stage3["pareto_candidate_ids"])
        )
    else:
        lines.append("Stage 3 Pareto: " + stage3["status"])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    status = collect_status(args.root.resolve())
    print(json.dumps(status, indent=2, ensure_ascii=False) if args.json else render(status))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
