"""Command construction for TensorRT engine builds."""

from __future__ import annotations

import sys
import re
from pathlib import Path

from .registry import REPOSITORY_ROOT


def sparse_tactic_summary(log_path: Path) -> dict:
    pattern = re.compile(
        r"\(Sparsity\) (Found|Chose) (\d+) layer\(s\) "
        r"(?:eligible to use|using) sparse tactics:\s*(.*)"
    )
    events = []
    unique = {"Found": set(), "Chose": set()}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        kind, count_text, names_text = match.groups()
        names = [name.strip() for name in names_text.split(",") if name.strip()]
        unique[kind].update(names)
        events.append(
            {
                "kind": kind.lower(),
                "reported_count": int(count_text),
                "layers": names,
            }
        )
    chosen_events = [event for event in events if event["kind"] == "chose"]
    return {
        "events": events,
        "max_eligible_in_event": max(
            (
                event["reported_count"]
                for event in events
                if event["kind"] == "found"
            ),
            default=0,
        ),
        "max_chosen_in_event": max(
            (event["reported_count"] for event in chosen_events),
            default=0,
        ),
        "unique_eligible_layer_names": sorted(unique["Found"]),
        "unique_chosen_layer_names": sorted(unique["Chose"]),
        "sparse_tactic_selected": any(
            event["reported_count"] > 0 for event in chosen_events
        ),
    }


def build_command(
    camera: str,
    onnx_path: Path,
    output_dir: Path,
    output_name: str,
    precision: str,
    workspace_mib: int,
    force: bool = False,
    dry_run: bool = False,
    enable_sparse: bool = False,
    verbose: bool = False,
    calibration_dir: Path | None = None,
    calibration_count: int | None = None,
    calibration_seed: int | None = None,
) -> list[str]:
    if precision not in {"fp32", "fp16", "int8"}:
        raise ValueError(f"Unsupported TensorRT precision: {precision}")
    command = [
        sys.executable,
        str(
            REPOSITORY_ROOT
            / "scripts"
            / "lightweighting"
            / f"build_tensorrt_{precision}.py"
        ),
        "--camera",
        camera,
        "--onnx",
        str(onnx_path),
        "--output-dir",
        str(output_dir),
        "--output-name",
        output_name,
        "--workspace-mib",
        str(workspace_mib),
    ]
    if force:
        command.append("--force")
    if dry_run:
        command.append("--dry-run")
    if enable_sparse:
        if precision not in {"fp16", "int8"}:
            raise ValueError("Sparse weights require FP16 or INT8.")
        command.append("--sparse-weights")
    if verbose:
        command.append("--verbose")
    if precision == "int8" and calibration_dir is not None:
        command.extend(["--calibration-dir", str(calibration_dir)])
    if precision == "int8" and calibration_count is not None:
        command.extend(["--calibration-count", str(calibration_count)])
    if precision == "int8" and calibration_seed is not None:
        command.extend(["--calibration-seed", str(calibration_seed)])
    return command
