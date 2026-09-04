#!/usr/bin/env python3
"""Run the complete notebook deployment evaluation from ONNX to Excel.

For every Stage-1-passing candidate this command builds and inspects a
TensorRT engine, repeats the latency benchmark, evaluates all 437 held-out
test images, records terminal failures, computes the gated Pareto set, and
creates one Excel workbook. Completed candidates are reused on restart unless
``--restart`` or ``--force-build`` is requested.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml
import psutil


ROOT = Path(__file__).resolve().parents[2]
STAGE1 = Path("results/stage1-static-evaluation.json")
STATE = Path("results/stage2-notebook-state.json")
SUMMARY_JSON = Path("results/stage2-notebook-summary.json")
SUMMARY_CSV = Path("results/stage2-notebook-summary.csv")
PARETO_JSON = Path("results/stage3-pareto.json")
PARETO_CSV = Path("results/stage3-pareto.csv")
REPORT_XLSX = Path("results/stage2-evaluation-report.xlsx")
PRE_CONTROL_STATE = Path("results/stage2-notebook-state-before-controlled-rerun.json")
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
TERMINAL_STATUSES = {
    "completed",
    "build-failed",
    "engine-analysis-failed",
    "benchmark-failed",
    "accuracy-failed",
}
T_CRITICAL_95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
PARETO_MAXIMIZE = ("mask_ap",)
PARETO_MINIMIZE = ("median_ms", "engine_size_bytes")
PARETO_SECONDARY = (
    "bbox_ap",
    "semantic_miou",
    "p95_ms",
    "gpu_peak_allocated_bytes",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run(command: list[str], log: Path) -> tuple[bool, str]:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("$ " + " ".join(command), flush=True)
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return result.returncode == 0, (
        f"returncode={result.returncode}; log={log.relative_to(ROOT)}"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], quantile: float) -> float:
    """Linear percentile matching NumPy's default for a portable runner."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Cannot calculate a percentile of an empty sequence")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def newest_json(directory: Path) -> Path:
    paths = sorted(directory.glob("*.json"), key=lambda path: path.stat().st_mtime_ns)
    if not paths:
        raise FileNotFoundError(f"No JSON result in {directory}")
    return paths[-1]


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def optional_float(value: str) -> float | None:
    """Parse an nvidia-smi numeric field while preserving unsupported values."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def ac_power_connected() -> bool | None:
    """Return the external-power state without assuming a battery is present."""
    online_files = sorted(Path("/sys/class/power_supply").glob("*/online"))
    values: list[bool] = []
    for path in online_files:
        try:
            values.append(path.read_text(encoding="utf-8").strip() == "1")
        except OSError:
            continue
    if values:
        return any(values)
    battery = psutil.sensors_battery()
    return None if battery is None else bool(battery.power_plugged)


def read_first_line(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").splitlines()[0].strip()
    except (OSError, IndexError):
        return None


def cpu_model_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except (OSError, IndexError):
        pass
    return platform.processor() or platform.machine()


def query_nvidia(fields: tuple[str, ...]) -> dict[str, str] | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=" + ",".join(fields),
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        values = [item.strip() for item in completed.stdout.splitlines()[0].split(",")]
        return dict(zip(fields, values, strict=True))
    except (FileNotFoundError, IndexError, subprocess.CalledProcessError, ValueError):
        return None


def capture_environment() -> dict[str, Any]:
    """Capture static host facts that can explain runtime-measurement changes."""
    gpu = query_nvidia(
        (
            "name",
            "memory.total",
            "driver_version",
            "power.limit",
            "temperature.gpu",
            "clocks.sm",
            "clocks.mem",
            "pstate",
        )
    )
    return {
        "recorded_at_utc": now(),
        "cpu": cpu_model_name(),
        "logical_cpus": psutil.cpu_count(logical=True),
        "physical_cpus": psutil.cpu_count(logical=False),
        "ram_total_bytes": psutil.virtual_memory().total,
        "platform": platform.platform(),
        "kernel": platform.release(),
        "cpu_scaling_governor": read_first_line(
            Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
        ),
        "cpu_energy_performance_preference": read_first_line(
            Path(
                "/sys/devices/system/cpu/cpu0/cpufreq/"
                "energy_performance_preference"
            )
        ),
        "platform_profile": read_first_line(Path("/sys/firmware/acpi/platform_profile")),
        "ac_power_connected": ac_power_connected(),
        "gpu": gpu,
    }


def validate_measurement_environment(
    environment: dict[str, Any], config: dict[str, Any]
) -> None:
    """Reject a latency run whose selected power policy differs from protocol."""
    requirements = {
        "platform_profile": config.get("required_platform_profile"),
        "cpu_energy_performance_preference": config.get(
            "required_cpu_energy_performance_preference"
        ),
    }
    mismatches = [
        f"{field}={environment.get(field)!r}, required={required!r}"
        for field, required in requirements.items()
        if required is not None and environment.get(field) != required
    ]
    if mismatches:
        raise RuntimeError(
            "Latency measurement power policy mismatch: " + "; ".join(mismatches)
        )
    if config.get("require_ac_power") and environment.get("ac_power_connected") is not True:
        raise RuntimeError("Latency measurement requires confirmed AC power")


def capture_load_sample(interval_seconds: float) -> dict[str, Any]:
    """Measure host and GPU load across a short interval before a benchmark."""
    cpu_percent = psutil.cpu_percent(interval=interval_seconds)
    gpu = query_nvidia(
        (
            "utilization.gpu",
            "utilization.memory",
            "memory.used",
            "memory.total",
            "temperature.gpu",
            "power.draw",
            "clocks.sm",
            "clocks.mem",
            "pstate",
        )
    )
    return {
        "timestamp_utc": now(),
        "cpu_utilization_percent": float(cpu_percent),
        "system_memory_utilization_percent": float(psutil.virtual_memory().percent),
        "load_average_1m": float(psutil.getloadavg()[0]),
        "ac_power_connected": ac_power_connected(),
        "gpu_utilization_percent": optional_float(
            (gpu or {}).get("utilization.gpu", "")
        ),
        "gpu_memory_utilization_percent": optional_float(
            (gpu or {}).get("utilization.memory", "")
        ),
        "gpu_memory_used_mib": optional_float((gpu or {}).get("memory.used", "")),
        "gpu_memory_total_mib": optional_float((gpu or {}).get("memory.total", "")),
        "gpu_temperature_c": optional_float((gpu or {}).get("temperature.gpu", "")),
        "gpu_power_draw_w": optional_float((gpu or {}).get("power.draw", "")),
        "gpu_sm_clock_mhz": optional_float((gpu or {}).get("clocks.sm", "")),
        "gpu_memory_clock_mhz": optional_float((gpu or {}).get("clocks.mem", "")),
        "gpu_performance_state": (gpu or {}).get("pstate"),
    }


def mean_available(samples: list[dict[str, Any]], key: str) -> float | None:
    values = [float(sample[key]) for sample in samples if sample.get(key) is not None]
    return statistics.fmean(values) if values else None


def load_window_summary(samples: list[dict[str, Any]]) -> dict[str, float | None]:
    keys = (
        "cpu_utilization_percent",
        "gpu_utilization_percent",
        "gpu_memory_utilization_percent",
        "gpu_temperature_c",
        "gpu_power_draw_w",
        "gpu_sm_clock_mhz",
        "gpu_memory_clock_mhz",
    )
    return {key: mean_available(samples, key) for key in keys}


def assess_load_window(
    samples: list[dict[str, Any]],
    config: dict[str, Any],
    reference: dict[str, Any] | None = None,
) -> list[str]:
    """Return reasons why a full pre-measurement sample window is unsuitable."""
    required = int(config["required_consecutive_samples"])
    if len(samples) < required:
        return [f"need {required - len(samples)} more consecutive sample(s)"]
    window = samples[-required:]
    reasons: list[str] = []
    required_metrics = (
        "gpu_utilization_percent",
        "gpu_memory_utilization_percent",
        "gpu_temperature_c",
    )
    for metric in required_metrics:
        if any(sample.get(metric) is None for sample in window):
            reasons.append(f"{metric} unavailable")

    threshold_map = {
        "cpu_utilization_percent": "max_cpu_utilization_percent",
        "gpu_utilization_percent": "max_gpu_utilization_percent",
        "gpu_memory_utilization_percent": "max_gpu_memory_utilization_percent",
        "gpu_temperature_c": "max_gpu_temperature_c",
    }
    for metric, threshold_name in threshold_map.items():
        values = [float(sample[metric]) for sample in window if sample.get(metric) is not None]
        threshold = float(config[threshold_name])
        if values and max(values) > threshold:
            reasons.append(f"{metric} max {max(values):.1f} > {threshold:.1f}")

    span_map = {
        "cpu_utilization_percent": "max_cpu_utilization_span_percent",
        "gpu_utilization_percent": "max_gpu_utilization_span_percent",
        "gpu_memory_utilization_percent": (
            "max_gpu_memory_utilization_span_percent"
        ),
        "gpu_temperature_c": "max_gpu_temperature_span_c",
    }
    for metric, threshold_name in span_map.items():
        values = [float(sample[metric]) for sample in window if sample.get(metric) is not None]
        threshold = float(config[threshold_name])
        if len(values) == required and max(values) - min(values) > threshold:
            reasons.append(
                f"{metric} span {max(values) - min(values):.1f} > {threshold:.1f}"
            )

    if config.get("require_ac_power") and any(
        sample.get("ac_power_connected") is not True for sample in window
    ):
        reasons.append("AC power is not confirmed")

    reference_map = {
        "cpu_utilization_percent": (
            "max_cpu_utilization_delta_from_reference_percent"
        ),
        "gpu_utilization_percent": (
            "max_gpu_utilization_delta_from_reference_percent"
        ),
        "gpu_memory_utilization_percent": (
            "max_gpu_memory_utilization_delta_from_reference_percent"
        ),
        "gpu_temperature_c": "max_gpu_temperature_delta_from_reference_c",
    }
    for metric, threshold_name in reference_map.items():
        if not reference or reference.get(metric) is None:
            continue
        current = mean_available(window, metric)
        allowed_delta = float(config[threshold_name])
        if current is not None and abs(current - float(reference[metric])) > allowed_delta:
            reasons.append(
                f"{metric} differs from session reference by "
                f"{abs(current - float(reference[metric])):.1f} > {allowed_delta:.1f}"
            )
    return reasons


def wait_for_stable_load(
    config: dict[str, Any],
    reference: dict[str, Any] | None = None,
    *,
    sample_fn: Any = capture_load_sample,
    monotonic_fn: Any = time.monotonic,
) -> dict[str, Any]:
    """Wait until a complete comparable idle window is observed or time out."""
    started_at = now()
    started = monotonic_fn()
    if not config.get("enabled", True):
        return {"status": "disabled", "started_at_utc": started_at}
    timeout = float(config["timeout_seconds"])
    interval = float(config["sample_interval_seconds"])
    required = int(config["required_consecutive_samples"])
    sample_limit = int(config.get("recorded_sample_limit", 30))
    consecutive: list[dict[str, Any]] = []
    recorded: list[dict[str, Any]] = []
    total_samples = 0
    last_reasons: list[str] = []

    while monotonic_fn() - started < timeout:
        sample = sample_fn(interval)
        total_samples += 1
        recorded.append(sample)
        recorded = recorded[-sample_limit:]
        single_reasons = assess_load_window([sample], {**config, "required_consecutive_samples": 1})
        if single_reasons:
            consecutive.clear()
            last_reasons = single_reasons
            continue
        consecutive.append(sample)
        consecutive = consecutive[-required:]
        if len(consecutive) < required:
            continue
        last_reasons = assess_load_window(consecutive, config, reference)
        if not last_reasons:
            summary = load_window_summary(consecutive)
            return {
                "status": "stable",
                "started_at_utc": started_at,
                "finished_at_utc": now(),
                "wait_seconds": monotonic_fn() - started,
                "total_samples": total_samples,
                "accepted_window": consecutive,
                "accepted_window_mean": summary,
                "session_reference": reference,
                "thresholds": dict(config),
                "recent_samples": recorded,
            }
        consecutive.pop(0)

    return {
        "status": "timeout",
        "started_at_utc": started_at,
        "finished_at_utc": now(),
        "wait_seconds": monotonic_fn() - started,
        "total_samples": total_samples,
        "last_rejection_reasons": last_reasons,
        "session_reference": reference,
        "thresholds": dict(config),
        "recent_samples": recorded,
    }


def update_progress(
    state: dict[str, Any],
    *,
    completed_units: int,
    total_units: int,
    current_candidate: str | None,
    current_stage: str,
    current_repetition: int | None = None,
) -> None:
    """Persist enough timing information for a second terminal to show an ETA."""
    started = datetime.fromisoformat(state["created_at_utc"])
    elapsed = max(0.0, (datetime.now(timezone.utc) - started).total_seconds())
    rate = completed_units / elapsed if completed_units and elapsed else None
    remaining_seconds = (
        (total_units - completed_units) / rate if rate and total_units >= completed_units else None
    )
    eta = (
        datetime.fromtimestamp(time.time() + remaining_seconds, timezone.utc).isoformat()
        if remaining_seconds is not None
        else None
    )
    state["progress"] = {
        "updated_at_utc": now(),
        "completed_units": completed_units,
        "total_units": total_units,
        "percent": (100.0 * completed_units / total_units if total_units else 100.0),
        "current_candidate": current_candidate,
        "current_stage": current_stage,
        "current_repetition": current_repetition,
        "elapsed_seconds": elapsed,
        "estimated_remaining_seconds": remaining_seconds,
        "estimated_finish_at_utc": eta,
    }


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "계산 중"
    rounded = max(0, int(round(seconds)))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def print_progress(state: dict[str, Any]) -> None:
    progress = state["progress"]
    eta = progress["estimated_finish_at_utc"]
    eta_local = (
        datetime.fromisoformat(eta).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        if eta
        else "계산 중"
    )
    repetition = (
        f" repeat {progress['current_repetition']}"
        if progress.get("current_repetition") is not None
        else ""
    )
    print(
        "[진행 {completed_units}/{total_units} | {percent:.1f}%] "
        "{current_candidate} {current_stage}{repetition} | 경과 {elapsed} | "
        "예상 종료 {eta}".format(
            **progress,
            elapsed=format_duration(progress["elapsed_seconds"]),
            eta=eta_local,
            repetition=repetition,
        ),
        flush=True,
    )


def preflight(
    protocol: dict[str, Any],
    eligible: list[str],
    stage1_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fail before a multi-hour run if the portable payload is incomplete."""
    dataset = ROOT / protocol["dataset_dir"]
    annotation = dataset / "_annotations.coco.json"
    if not annotation.is_file():
        raise FileNotFoundError(annotation)
    coco = load_json(annotation)
    images = coco.get("images", [])
    expected_images = int(protocol["expected_images"])
    if len(images) != expected_images:
        raise ValueError(
            f"Expected {expected_images} final-test images, found {len(images)}"
        )
    referenced = {str(item["file_name"]) for item in images}
    actual = {
        path.name
        for path in dataset.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    if actual != referenced:
        raise ValueError(
            "Final-test COCO/image mismatch: "
            f"missing={sorted(referenced - actual)[:3]}, "
            f"extra={sorted(actual - referenced)[:3]}"
        )

    verified_data_files = 0
    checksums = ROOT / "data-checksums.sha256"
    if checksums.is_file():
        for line in checksums.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            expected, relative = line.split(maxsplit=1)
            target = ROOT / relative.strip()
            if not target.is_file() or sha256(target) != expected:
                raise ValueError(f"Data checksum mismatch: {relative.strip()}")
            verified_data_files += 1

    verified_onnx_files = 0
    stage1_rows = stage1_rows or load_json(ROOT / STAGE1).get("rows", [])
    eligible_set = set(eligible)
    deployment_sources: dict[str, dict[str, Any]] = {}
    consumers: dict[str, list[str]] = {}
    for row in stage1_rows:
        experiment_id = str(row.get("experiment_id", ""))
        if experiment_id not in eligible_set:
            continue
        relative = str(row.get("deployment_onnx", ""))
        expected_sha256 = str(row.get("deployment_onnx_sha256", ""))
        expected_size = row.get("static_metrics", {}).get("onnx_size_bytes")
        if not relative or not expected_sha256 or expected_size is None:
            raise ValueError(
                f"Stage 1 lacks deployment identity for {experiment_id}"
            )
        identity = {
            "path": relative,
            "sha256": expected_sha256,
            "size_bytes": int(expected_size),
        }
        previous_identity = deployment_sources.setdefault(relative, identity)
        if previous_identity != identity:
            raise ValueError(
                f"Stage 1 has conflicting deployment identities for {relative}"
            )
        consumers.setdefault(relative, []).append(experiment_id)

    missing_or_mismatched = []
    digest_cache: dict[tuple[int, int, int, int], str] = {}
    for relative, identity in sorted(deployment_sources.items()):
        target = ROOT / relative
        if not target.is_file():
            missing_or_mismatched.append(
                f"{relative}: missing (used by {', '.join(consumers[relative])})"
            )
            continue
        actual_size = target.stat().st_size
        if actual_size != identity["size_bytes"]:
            missing_or_mismatched.append(
                f"{relative}: size {actual_size}, expected {identity['size_bytes']}"
            )
            continue
        stat = target.stat()
        key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        digest_cache.setdefault(key, sha256(target))
        if digest_cache[key] != identity["sha256"]:
            missing_or_mismatched.append(
                f"{relative}: SHA-256 {digest_cache[key]}, "
                f"expected {identity['sha256']}"
            )
            continue
        verified_onnx_files += 1
    if missing_or_mismatched:
        details = "\n- ".join(missing_or_mismatched)
        raise ValueError(
            "Stage 2 requires the exact ONNX files recorded by Stage 1; "
            "older or unidentified artifacts are rejected:\n- " + details
        )

    verified_automation_files = 0
    manifest_path = ROOT / "manifest.json"
    if manifest_path.is_file():
        manifest = load_json(manifest_path)
        if set(manifest.get("experiments", [])) != set(eligible):
            raise ValueError("manifest.json candidate set differs from Stage 1")
        for item in manifest.get("onnx_sources", []):
            target = ROOT / item["path"]
            if not target.is_file() or target.stat().st_size != int(item["size_bytes"]):
                raise ValueError(f"ONNX size/missing error: {item['path']}")
            stat = target.stat()
            key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
            digest_cache.setdefault(key, sha256(target))
            if digest_cache[key] != item["sha256"]:
                raise ValueError(f"ONNX checksum mismatch: {item['path']}")
            # Stage-1 identity verification above is authoritative. The bundle
            # manifest is an additional transport-integrity check.
        for item in (manifest.get("automation") or {}).get("files", []):
            target = ROOT / item["path"]
            if not target.is_file() or target.stat().st_size != int(item["size_bytes"]):
                raise ValueError(f"Automation file size/missing error: {item['path']}")
            if sha256(target) != item["sha256"]:
                raise ValueError(f"Automation checksum mismatch: {item['path']}")
            verified_automation_files += 1

    data_manifest_path = ROOT / "data-manifest.json"
    if data_manifest_path.is_file():
        data_manifest = load_json(data_manifest_path)
        if data_manifest.get("test", {}).get("summary", {}).get("images") != expected_images:
            raise ValueError("data-manifest.json final-test image count mismatch")
        if data_manifest.get("leakage_verification", {}).get("passed") is not True:
            raise ValueError("data-manifest.json does not confirm leakage verification")
        registered_defaults = yaml.safe_load(
            (ROOT / "configs/experiments/defaults.yaml").read_text(encoding="utf-8")
        )
        expected_calibration = int(
            registered_defaults["reproducibility"]["calibration_image_count"]
        )
        calibration = data_manifest.get("calibration", {})
        calibration_dir = ROOT / calibration.get("path", "")
        calibration_images = {
            path.name
            for path in calibration_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
        if (
            calibration.get("source_split") != "train"
            or calibration.get("selected_images") != expected_calibration
            or len(calibration_images) != expected_calibration
        ):
            raise ValueError(
                "INT8 calibration must contain the registered train-only image set: "
                f"expected={expected_calibration}, actual={len(calibration_images)}"
            )
    registered_defaults = yaml.safe_load(
        (ROOT / "configs/experiments/defaults.yaml").read_text(encoding="utf-8")
    )
    free_disk_bytes = shutil.disk_usage(ROOT).free
    minimum_free_disk_gib = float(
        registered_defaults["reproducibility"].get("notebook_min_free_disk_gib", 10)
    )
    if free_disk_bytes < minimum_free_disk_gib * 1024**3:
        raise RuntimeError(
            f"At least {minimum_free_disk_gib:g} GiB free disk is required; "
            f"found {free_disk_bytes / 1024**3:.1f} GiB"
        )

    driver = None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,pci.bus_id",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        driver = result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    return {
        "checked_at_utc": now(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "nvidia_driver_and_bus": driver,
        "dataset": str(dataset.relative_to(ROOT)),
        "annotation_sha256": sha256(annotation),
        "test_images": len(images),
        "annotations": len(coco.get("annotations", [])),
        "free_disk_bytes": free_disk_bytes,
        "minimum_free_disk_gib": minimum_free_disk_gib,
        "verified_data_files": verified_data_files,
        "verified_onnx_manifest_entries": verified_onnx_files,
        "verified_automation_manifest_entries": verified_automation_files,
        "package_versions": {
            name: package_version(name)
            for name in (
                "rfdetr",
                "torch",
                "torchvision",
                "tensorrt-cu13",
                "pycocotools",
                "numpy",
                "XlsxWriter",
            )
        },
    }


def aggregate_benchmarks(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    if not payloads:
        raise ValueError("At least one benchmark repetition is required")
    timings = [
        float(value)
        for payload in payloads
        for value in payload.get("timings_ms", [])
    ]
    if not timings:
        raise ValueError("Benchmark details contain no timings_ms")
    replicate_medians = [float(payload["median_ms"]) for payload in payloads]
    replicate_means = [float(payload["mean_ms"]) for payload in payloads]
    mean_ms = statistics.fmean(timings)
    standard_deviation_ms = statistics.stdev(timings) if len(timings) > 1 else 0.0
    replicate_median_mean = statistics.fmean(replicate_medians)
    replicate_median_std = (
        statistics.stdev(replicate_medians) if len(replicate_medians) > 1 else 0.0
    )
    critical = T_CRITICAL_95.get(len(replicate_medians), 1.96)
    margin = critical * replicate_median_std / math.sqrt(len(replicate_medians))
    return {
        "benchmark_repetitions": len(payloads),
        "measured_runs_per_repetition": int(payloads[0]["measured_runs"]),
        "total_measured_runs": len(timings),
        "replicate_median_ms": replicate_medians,
        "replicate_mean_ms": replicate_means,
        "replicate_median_mean_ms": replicate_median_mean,
        "replicate_median_std_ms": replicate_median_std,
        "replicate_median_cv": (
            replicate_median_std / replicate_median_mean
            if replicate_median_mean
            else float("nan")
        ),
        "replicate_median_ci95_low_ms": replicate_median_mean - margin,
        "replicate_median_ci95_high_ms": replicate_median_mean + margin,
        "mean_ms": mean_ms,
        "median_ms": statistics.median(timings),
        "standard_deviation_ms": standard_deviation_ms,
        "coefficient_of_variation": standard_deviation_ms / mean_ms,
        "min_ms": min(timings),
        "max_ms": max(timings),
        "p05_ms": percentile(timings, 0.05),
        "p25_ms": percentile(timings, 0.25),
        "p75_ms": percentile(timings, 0.75),
        "p95_ms": percentile(timings, 0.95),
        "p99_ms": percentile(timings, 0.99),
        "iqr_ms": percentile(timings, 0.75) - percentile(timings, 0.25),
        "fps": 1000.0 / mean_ms,
        "gpu_peak_allocated_bytes": max(
            int(payload.get("gpu_peak_allocated_bytes") or 0) for payload in payloads
        ),
        "gpu_peak_reserved_bytes": max(
            int(payload.get("gpu_peak_reserved_bytes") or 0) for payload in payloads
        ),
    }


def accuracy_columns(payload: dict[str, Any]) -> dict[str, float]:
    metrics = payload["metrics"]
    bbox = metrics["bbox"]
    mask = metrics["segm"]
    return {
        "bbox_ap": float(bbox["ap"]),
        "bbox_ap50": float(bbox["ap50"]),
        "bbox_ap75": float(bbox["ap75"]),
        "bbox_ap_small": float(bbox["ap_small"]),
        "bbox_ap_medium": float(bbox["ap_medium"]),
        "bbox_ap_large": float(bbox["ap_large"]),
        "bbox_ar100": float(bbox["ar100"]),
        "mask_ap": float(mask["ap"]),
        "mask_ap50": float(mask["ap50"]),
        "mask_ap75": float(mask["ap75"]),
        "mask_ap_small": float(mask["ap_small"]),
        "mask_ap_medium": float(mask["ap_medium"]),
        "mask_ap_large": float(mask["ap_large"]),
        "mask_ar100": float(mask["ar100"]),
        "semantic_miou": float(metrics["semantic_miou"]),
    }


def add_baseline_comparisons(
    rows: list[dict[str, Any]], defaults: dict[str, Any]
) -> list[dict[str, Any]]:
    output = [dict(row) for row in rows]
    baseline = next(
        (row for row in output if row.get("experiment_id") == "B01"), None
    )
    if baseline is None:
        for row in output:
            row.update(
                measurement_valid=False,
                accuracy_gate_pass=False,
                realtime_30fps_pass=False,
                stage3_pareto_eligible=False,
                exclusion_reason="B01 baseline measurement is unavailable",
            )
        return output

    limits = defaults["selection"]["accuracy_vs_baseline"]
    realtime = defaults["selection"]["deployment_realtime"]
    latency_budget_ms = float(realtime["median_latency_max_ms"])
    target_fps = float(realtime["target_fps"])
    stability_warning = float(
        defaults["evaluation_protocol"]["latency"].get(
            "replicate_median_cv_warning_threshold", 0.05
        )
    )
    for row in output:
        row["bbox_ap_delta_vs_B01"] = row["bbox_ap"] - baseline["bbox_ap"]
        row["mask_ap_delta_vs_B01"] = row["mask_ap"] - baseline["mask_ap"]
        row["semantic_miou_delta_vs_B01"] = (
            row["semantic_miou"] - baseline["semantic_miou"]
        )
        row["median_latency_reduction_vs_B01_pct"] = 100.0 * (
            baseline["median_ms"] - row["median_ms"]
        ) / baseline["median_ms"]
        row["speedup_vs_B01"] = baseline["median_ms"] / row["median_ms"]
        row["engine_size_reduction_vs_B01_pct"] = 100.0 * (
            baseline["engine_size_bytes"] - row["engine_size_bytes"]
        ) / baseline["engine_size_bytes"]
        baseline_memory = int(baseline["gpu_peak_allocated_bytes"])
        row["gpu_memory_reduction_vs_B01_pct"] = (
            100.0
            * (baseline_memory - int(row["gpu_peak_allocated_bytes"]))
            / baseline_memory
            if baseline_memory
            else None
        )
        row["measurement_valid"] = all(
            math.isfinite(float(row[name]))
            for name in (
                "bbox_ap",
                "mask_ap",
                "semantic_miou",
                "median_ms",
                "p95_ms",
            )
        ) and int(row["total_measured_runs"]) > 0
        row["latency_stability_warning"] = (
            float(row["replicate_median_cv"]) > stability_warning
        )
        row["accuracy_gate_pass"] = (
            row["bbox_ap_delta_vs_B01"] >= -float(limits["bbox_ap_max_absolute_drop"])
            and row["mask_ap_delta_vs_B01"]
            >= -float(limits["mask_ap_max_absolute_drop"])
            and row["semantic_miou_delta_vs_B01"]
            >= -float(limits["semantic_miou_max_absolute_drop"])
        )
        row["realtime_target_fps"] = target_fps
        row["median_latency_budget_ms"] = latency_budget_ms
        row["realtime_30fps_pass"] = (
            row["measurement_valid"]
            and float(row["median_ms"]) <= latency_budget_ms
        )
        row["stage3_pareto_eligible"] = (
            row["measurement_valid"] and row["accuracy_gate_pass"]
        )
        reasons = []
        if not row["measurement_valid"]:
            reasons.append("invalid or incomplete measurement")
        if not row["accuracy_gate_pass"]:
            reasons.append("accuracy retention gate failed")
        row["exclusion_reason"] = "; ".join(reasons)
    return output


def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    # Primary three-objective Pareto: segmentation quality, latency, artifact size.
    # The remaining measured metrics are reported as secondary evidence and have
    # already contributed to the accuracy-retention and validity gates.
    left_values = tuple(-left[name] for name in PARETO_MAXIMIZE) + tuple(
        left[name] for name in PARETO_MINIMIZE
    )
    right_values = tuple(-right[name] for name in PARETO_MAXIMIZE) + tuple(
        right[name] for name in PARETO_MINIMIZE
    )
    return all(a <= b for a, b in zip(left_values, right_values)) and any(
        a < b for a, b in zip(left_values, right_values)
    )


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary_payload(
    state: dict[str, Any],
    eligible: list[str],
    rows: list[dict[str, Any]],
    defaults: dict[str, Any],
) -> dict[str, Any]:
    candidates = state.get("candidates", {})
    pending = [
        item
        for item in eligible
        if candidates.get(item, {}).get("status") not in TERMINAL_STATUSES
    ]
    failures = [
        {
            "experiment_id": item,
            "status": candidates.get(item, {}).get("status", "pending"),
            "reason": candidates.get(item, {}).get("reason", ""),
        }
        for item in eligible
        if candidates.get(item, {}).get("status") in TERMINAL_STATUSES
        and candidates.get(item, {}).get("status") != "completed"
    ]
    compared = add_baseline_comparisons(rows, defaults)
    return {
        "created_at_utc": now(),
        "eligible_count": len(eligible),
        "completed_count": len(compared),
        "terminal_failure_count": len(failures),
        "pending_count": len(pending),
        "hardware": {
            key: state.get(key)
            for key in (
                "gpu",
                "compute_capability",
                "cuda",
                "tensorrt",
                "nvidia_driver_and_bus",
            )
        },
        "preflight": state.get("preflight", {}),
        "environment": state.get("environment", {}),
        "load_stabilization_protocol": state.get(
            "load_stabilization_protocol", {}
        ),
        "load_stabilization_reference": state.get(
            "load_stabilization_reference"
        ),
        "progress": state.get("progress", {}),
        "protocol": defaults["evaluation_protocol"],
        "selection": defaults["selection"],
        "rows": compared,
        "failures": failures,
        "pending_candidates": pending,
    }


def persist_summary(
    state: dict[str, Any],
    eligible: list[str],
    rows: list[dict[str, Any]],
    defaults: dict[str, Any],
) -> dict[str, Any]:
    summary = summary_payload(state, eligible, rows, defaults)
    write_json(ROOT / SUMMARY_JSON, summary)
    write_table(ROOT / SUMMARY_CSV, summary["rows"])
    return summary


def create_excel_report(log: Path) -> tuple[bool, str]:
    return run(
        [
            sys.executable,
            "scripts/reporting/generate_stage2_excel.py",
            "--output",
            str(REPORT_XLSX),
        ],
        log,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", help="Optional subset for debugging")
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument(
        "--remeasure-latency",
        action="store_true",
        help=(
            "Reuse verified engines and accuracy results, but replace every "
            "successful candidate's latency measurements under the registered "
            "load-stabilization gate."
        ),
    )
    parser.add_argument(
        "--benchmark-repetitions",
        type=int,
        help="Override the registered latency repetition count.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.remeasure_latency and (args.force_build or args.skip_build or args.restart):
        raise ValueError(
            "--remeasure-latency cannot be combined with --force-build, "
            "--skip-build, or --restart"
        )
    if args.remeasure_latency and args.only:
        raise ValueError(
            "--remeasure-latency replaces the canonical comparison and must run "
            "the complete eligible cohort; --only is not supported"
        )
    if args.report_only:
        ok, evidence = create_excel_report(ROOT / "results/stage2-notebook-logs/report.log")
        if not ok:
            raise RuntimeError(f"Excel report generation failed: {evidence}")
        print(f"Excel report: {REPORT_XLSX}")
        return 0

    stage1 = load_json(ROOT / STAGE1)
    if stage1.get("unperformed_count") != 0:
        raise RuntimeError("Stage 1 is incomplete; refusing Stage 2")
    full_eligible = [
        str(row["experiment_id"])
        for row in stage1.get("rows", [])
        if row.get("stage2_notebook_eligible") is True
    ]
    eligible = list(full_eligible)
    if args.only:
        unknown = sorted(set(args.only) - set(eligible))
        if unknown:
            raise ValueError(f"Not Stage-1 eligible: {unknown}")
        eligible = [item for item in eligible if item in set(args.only)]

    defaults = yaml.safe_load(
        (ROOT / "configs/experiments/defaults.yaml").read_text(encoding="utf-8")
    )
    protocol = defaults["evaluation_protocol"]
    preflight_result = preflight(protocol, full_eligible, stage1.get("rows", []))
    repetitions = int(
        args.benchmark_repetitions
        or protocol["latency"].get("benchmark_repetitions", 3)
    )
    if repetitions < 2:
        raise ValueError("At least two benchmark repetitions are required")

    if args.dry_run or args.preflight_only:
        print(f"Preflight PASS: {preflight_result}")
        print(f"Stage-1 eligible candidates: {len(eligible)}")
        print(" ".join(eligible))
        print(f"Latency: {repetitions} repetitions per candidate")
        print(
            "Load stabilization: "
            + json.dumps(
                protocol["latency"].get("load_stabilization", {}),
                ensure_ascii=False,
            )
        )
        print("Stage 2: build -> inspect -> repeated benchmark -> 437-image accuracy")
        print("Stage 3: accuracy-gated Pareto -> one Excel workbook")
        return 0

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Stage 2 requires a CUDA notebook GPU")
    import tensorrt as trt

    registry = yaml.safe_load(
        (ROOT / "configs/experiments/registry.yaml").read_text(encoding="utf-8")
    )["experiments"]
    latency = protocol["latency"]
    load_config = latency.get("load_stabilization", {})
    current_state = load_json(ROOT / STATE) if (ROOT / STATE).is_file() else {}
    previous: dict[str, Any] = {}
    source_candidates: dict[str, Any] = {}
    measurement_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    initial_load_reference: dict[str, Any] | None = None
    if args.remeasure_latency:
        if not current_state:
            raise RuntimeError(
                "--remeasure-latency requires an existing completed Stage 2 state"
            )
        if current_state.get("measurement_mode") == "controlled-latency-remeasurement":
            if not (ROOT / PRE_CONTROL_STATE).is_file():
                raise FileNotFoundError(
                    "Controlled remeasurement backup is missing: " + str(PRE_CONTROL_STATE)
                )
            source_state = load_json(ROOT / PRE_CONTROL_STATE)
            source_candidates = dict(source_state.get("candidates", {}))
            previous = {
                **source_candidates,
                **current_state.get("candidates", {}),
            }
            measurement_run_id = str(current_state["measurement_run_id"])
            initial_load_reference = current_state.get("load_stabilization_reference")
        else:
            source_state = current_state
            write_json(ROOT / PRE_CONTROL_STATE, source_state)
            source_candidates = dict(source_state.get("candidates", {}))
            previous = dict(source_candidates)
        missing_sources = [
            experiment_id
            for experiment_id in eligible
            if experiment_id not in previous
        ]
        if missing_sources:
            raise RuntimeError(
                "Existing Stage 2 state is incomplete; missing candidates: "
                + " ".join(missing_sources)
            )
    elif not args.restart and not args.force_build:
        previous = dict(current_state.get("candidates", {}))
    if not args.only:
        (ROOT / PARETO_JSON).unlink(missing_ok=True)
        (ROOT / PARETO_CSV).unlink(missing_ok=True)
    measurable_ids = (
        [
            experiment_id
            for experiment_id in eligible
            if source_candidates.get(experiment_id, {}).get("status") == "completed"
        ]
        if args.remeasure_latency
        else []
    )
    total_progress_units = len(measurable_ids) * repetitions
    completed_progress_units = 0
    measurement_environment = capture_environment()
    validate_measurement_environment(measurement_environment, load_config)
    state: dict[str, Any] = {
        "created_at_utc": now(),
        "status": "running",
        "measurement_mode": (
            "controlled-latency-remeasurement"
            if args.remeasure_latency
            else "full-stage2"
        ),
        "measurement_run_id": measurement_run_id,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "cuda": torch.version.cuda,
        "tensorrt": trt.__version__,
        "nvidia_driver_and_bus": preflight_result["nvidia_driver_and_bus"],
        "preflight": preflight_result,
        "environment": measurement_environment,
        "load_stabilization_protocol": load_config,
        "load_stabilization_reference": initial_load_reference,
        "eligible_candidates": eligible,
        "benchmark_repetitions": repetitions,
        "candidates": {},
    }
    if args.remeasure_latency:
        state["source_state_backup"] = str(PRE_CONTROL_STATE)
        state["reused_artifacts"] = ["TensorRT engines", "437-image accuracy results"]
    update_progress(
        state,
        completed_units=0,
        total_units=total_progress_units,
        current_candidate=None,
        current_stage="starting",
    )
    write_json(ROOT / STATE, state)
    logs = ROOT / "results/stage2-notebook-logs"
    completed_rows: list[dict[str, Any]] = []
    for experiment_id in eligible:
        old = previous.get(experiment_id, {})
        source_record = source_candidates.get(experiment_id, {})
        source_result: dict[str, Any] | None = None
        if args.remeasure_latency and source_record.get("status") != "completed":
            record = dict(source_record)
            record["reused_at_utc"] = now()
            record["reused_reason"] = "prior terminal non-latency result"
            state["candidates"][experiment_id] = record
            write_json(ROOT / STATE, state)
            print(f"{experiment_id}: prior terminal failure retained", flush=True)
            continue
        if (
            args.remeasure_latency
            and old.get("status") == "completed"
            and old.get("measurement_mode") == "controlled-latency-remeasurement"
            and isinstance(old.get("result"), dict)
        ):
            result = dict(old["result"])
            required_paths = [ROOT / result[name] for name in ("engine", "evaluation")]
            required_paths.extend(ROOT / path for path in result.get("benchmarks", []))
            if all(path.is_file() for path in required_paths):
                record = dict(old)
                record["reused_at_utc"] = now()
                state["candidates"][experiment_id] = record
                completed_rows.append(result)
                completed_progress_units += len(result.get("benchmarks", []))
                update_progress(
                    state,
                    completed_units=completed_progress_units,
                    total_units=total_progress_units,
                    current_candidate=experiment_id,
                    current_stage="controlled result reused",
                )
                write_json(ROOT / STATE, state)
                print_progress(state)
                continue
        if (
            not args.remeasure_latency
            and old.get("status") == "completed"
            and isinstance(old.get("result"), dict)
        ):
            result = old["result"]
            required_paths = [ROOT / result[name] for name in ("engine", "evaluation")]
            required_paths.extend(ROOT / path for path in result.get("benchmarks", []))
            if all(path.is_file() for path in required_paths):
                record = dict(old)
                record["reused_at_utc"] = now()
                state["candidates"][experiment_id] = record
                completed_rows.append(dict(result))
                write_json(ROOT / STATE, state)
                print(f"{experiment_id}: reusing completed Stage 2 result", flush=True)
                continue

        experiment = registry[experiment_id]
        record: dict[str, Any] = {
            "status": "running",
            "started_at_utc": now(),
            "measurement_mode": state["measurement_mode"],
            "load_stabilization_records": [],
        }
        state["candidates"][experiment_id] = record
        write_json(ROOT / STATE, state)
        if args.remeasure_latency:
            if not isinstance(source_record.get("result"), dict):
                raise RuntimeError(f"Missing reusable Stage 2 result for {experiment_id}")
            source_result = source_record["result"]
            engine = ROOT / source_result["engine"]
        else:
            engine = ROOT / f"artifacts/experiments/{experiment_id}/front/model.engine"
        if not args.skip_build and not args.remeasure_latency:
            command = [
                sys.executable,
                "scripts/experiments/build_candidate.py",
                experiment_id,
                "--camera",
                "front",
                "--target",
                "engine",
            ]
            if args.force_build:
                command.append("--force")
            ok, evidence = run(command, logs / f"{experiment_id}-build.log")
            if not ok:
                record.update(status="build-failed", reason=evidence, finished_at_utc=now())
                write_json(ROOT / STATE, state)
                persist_summary(state, eligible, completed_rows, defaults)
                continue
        if not engine.is_file():
            record.update(
                status="build-failed",
                reason="engine file missing",
                finished_at_utc=now(),
            )
            write_json(ROOT / STATE, state)
            persist_summary(state, eligible, completed_rows, defaults)
            continue

        if not args.remeasure_latency:
            ok, evidence = run(
                [
                    sys.executable,
                    "scripts/experiments/analyze_tensorrt_engine.py",
                    "--experiment",
                    experiment_id,
                    "--camera",
                    "front",
                    "--engine",
                    str(engine),
                ],
                logs / f"{experiment_id}-engine-static.log",
            )
            if not ok:
                record.update(
                    status="engine-analysis-failed",
                    reason=evidence,
                    finished_at_utc=now(),
                )
                write_json(ROOT / STATE, state)
                persist_summary(state, eligible, completed_rows, defaults)
                continue

        benchmark_payloads: list[dict[str, Any]] = []
        benchmark_paths: list[Path] = []
        benchmark_failed = False
        for repetition in range(1, repetitions + 1):
            if args.remeasure_latency:
                benchmark_dir = (
                    ROOT
                    / "results/benchmarks/notebook-controlled"
                    / measurement_run_id
                    / experiment_id
                    / f"repeat-{repetition:02d}"
                )
            else:
                benchmark_dir = (
                    ROOT
                    / f"results/benchmarks/notebook/{experiment_id}/repeat-{repetition:02d}"
                )
            update_progress(
                state,
                completed_units=completed_progress_units,
                total_units=total_progress_units,
                current_candidate=experiment_id,
                current_stage="waiting for stable load",
                current_repetition=repetition,
            )
            write_json(ROOT / STATE, state)
            print_progress(state)
            gate = wait_for_stable_load(
                load_config,
                state.get("load_stabilization_reference"),
            )
            gate_path = (
                ROOT
                / "results/measurement-environment"
                / measurement_run_id
                / experiment_id
                / f"repeat-{repetition:02d}.json"
            )
            write_json(gate_path, gate)
            record["load_stabilization_records"].append(
                str(gate_path.relative_to(ROOT))
            )
            if gate["status"] == "stable" and state["load_stabilization_reference"] is None:
                state["load_stabilization_reference"] = gate["accepted_window_mean"]
                gate["established_session_reference"] = True
                write_json(gate_path, gate)
            if gate["status"] not in {"stable", "disabled"}:
                record.update(
                    status="running",
                    reason=(
                        "load stabilization timed out; benchmark was not started: "
                        + "; ".join(gate.get("last_rejection_reasons", []))
                    ),
                    waiting_repetition=repetition,
                )
                state["status"] = "paused-load-not-stable"
                update_progress(
                    state,
                    completed_units=completed_progress_units,
                    total_units=total_progress_units,
                    current_candidate=experiment_id,
                    current_stage="paused: load did not stabilize",
                    current_repetition=repetition,
                )
                write_json(ROOT / STATE, state)
                persist_summary(state, eligible, completed_rows, defaults)
                print(
                    f"{experiment_id}: load did not stabilize within "
                    f"{load_config['timeout_seconds']} seconds; no benchmark was run",
                    flush=True,
                )
                return 2
            accepted = gate.get("accepted_window_mean", {})
            print(
                f"{experiment_id} repeat {repetition}: stable load accepted "
                f"(CPU {accepted.get('cpu_utilization_percent')}%, "
                f"GPU {accepted.get('gpu_utilization_percent')}%, "
                f"GPU {accepted.get('gpu_temperature_c')} C)",
                flush=True,
            )
            update_progress(
                state,
                completed_units=completed_progress_units,
                total_units=total_progress_units,
                current_candidate=experiment_id,
                current_stage="latency benchmark",
                current_repetition=repetition,
            )
            write_json(ROOT / STATE, state)
            ok, evidence = run(
                [
                    sys.executable,
                    "scripts/evaluation/benchmark_baseline.py",
                    "--camera",
                    "front",
                    "--backend",
                    "tensorrt",
                    "--precision-label",
                    str(experiment["precision"]),
                    "--engine",
                    str(engine),
                    "--image-dir",
                    str(ROOT / protocol["dataset_dir"]),
                    "--sample-count",
                    str(latency["sampled_images"]),
                    "--seed",
                    str(latency["sample_seed"]),
                    "--threshold",
                    str(latency["postprocess_confidence_threshold"]),
                    "--warmup",
                    str(latency["warmup_runs"]),
                    "--runs",
                    str(latency["measured_runs"]),
                    "--output-dir",
                    str(benchmark_dir),
                ],
                logs / f"{experiment_id}-benchmark-{repetition:02d}.log",
            )
            if not ok:
                record.update(
                    status="benchmark-failed",
                    reason=evidence,
                    failed_repetition=repetition,
                    finished_at_utc=now(),
                )
                write_json(ROOT / STATE, state)
                persist_summary(state, eligible, completed_rows, defaults)
                benchmark_failed = True
                break
            benchmark_path = newest_json(benchmark_dir)
            benchmark_paths.append(benchmark_path)
            benchmark_payload = load_json(benchmark_path)
            benchmark_payload["pre_measurement_load_stabilization"] = gate
            benchmark_payload["measurement_environment_record"] = str(
                gate_path.relative_to(ROOT)
            )
            write_json(benchmark_path, benchmark_payload)
            benchmark_payloads.append(benchmark_payload)
            completed_progress_units += 1
            record["completed_benchmarks"] = [
                str(path.relative_to(ROOT)) for path in benchmark_paths
            ]
            update_progress(
                state,
                completed_units=completed_progress_units,
                total_units=total_progress_units,
                current_candidate=experiment_id,
                current_stage="latency repetition completed",
                current_repetition=repetition,
            )
            write_json(ROOT / STATE, state)
            print_progress(state)
        if benchmark_failed:
            continue
        aggregate = aggregate_benchmarks(benchmark_payloads)

        if args.remeasure_latency:
            assert source_result is not None
            evaluation_path = ROOT / source_result["evaluation"]
            if not evaluation_path.is_file():
                record.update(
                    status="accuracy-failed",
                    reason=f"reusable accuracy result missing: {evaluation_path}",
                    finished_at_utc=now(),
                )
                write_json(ROOT / STATE, state)
                persist_summary(state, eligible, completed_rows, defaults)
                continue
        else:
            ok, evidence = run(
                [
                    sys.executable,
                    "scripts/evaluation/evaluate_coco_tensorrt.py",
                    "--experiment",
                    experiment_id,
                    "--camera",
                    "front",
                    "--engine",
                    str(engine),
                    "--dataset-dir",
                    str(ROOT / protocol["dataset_dir"]),
                    "--threshold",
                    str(protocol["coco_ap_confidence_threshold"]),
                    "--miou-threshold",
                    str(protocol["semantic_miou_confidence_threshold"]),
                    "--no-update-csv",
                ],
                logs / f"{experiment_id}-accuracy.log",
            )
            if not ok:
                record.update(
                    status="accuracy-failed", reason=evidence, finished_at_utc=now()
                )
                write_json(ROOT / STATE, state)
                persist_summary(state, eligible, completed_rows, defaults)
                continue
            evaluation_path = ROOT / f"results/coco-evaluation/{experiment_id}-front.json"
        evaluation = load_json(evaluation_path)
        if evaluation.get("engine_sha256") != sha256(engine):
            raise ValueError(
                f"Accuracy result engine identity mismatch for {experiment_id}"
            )
        row = {
            "experiment_id": experiment_id,
            "family": str(experiment["family"]),
            "method": str(experiment["method"]),
            "precision": str(experiment["precision"]),
            "input_shape": "x".join(
                str(item)
                for item in experiment.get(
                    "input_shape", defaults["export"]["input_shape"]
                )
            ),
            **accuracy_columns(evaluation),
            **aggregate,
            "engine_size_bytes": engine.stat().st_size,
            "engine_sha256": evaluation["engine_sha256"],
            "annotation_sha256": evaluation["annotation_sha256"],
            "test_image_count": int(evaluation["image_count"]),
            "prediction_count": int(evaluation["prediction_count"]),
            "engine": str(engine.relative_to(ROOT)),
            "benchmarks": [str(path.relative_to(ROOT)) for path in benchmark_paths],
            "load_stabilization_records": list(
                record["load_stabilization_records"]
            ),
            "evaluation": str(evaluation_path.relative_to(ROOT)),
        }
        completed_rows.append(row)
        record.update(
            status="completed",
            result=row,
            finished_at_utc=now(),
            accuracy_reused=args.remeasure_latency,
            engine_reused=args.remeasure_latency,
        )
        write_json(ROOT / STATE, state)
        persist_summary(state, eligible, completed_rows, defaults)
        print(f"{experiment_id}: Stage 2 completed", flush=True)

    pending = [
        experiment_id
        for experiment_id in eligible
        if state["candidates"].get(experiment_id, {}).get("status")
        not in TERMINAL_STATUSES
    ]
    state["status"] = "completed" if not pending else "incomplete"
    state["finished_at_utc"] = now()
    state["pending_candidates"] = pending
    update_progress(
        state,
        completed_units=completed_progress_units,
        total_units=total_progress_units,
        current_candidate=None,
        current_stage="creating final report",
    )
    write_json(ROOT / STATE, state)
    summary = persist_summary(state, eligible, completed_rows, defaults)

    # A debug subset must never overwrite the canonical all-candidate Pareto result.
    if not pending and not args.only and eligible == full_eligible:
        eligible_rows = [
            row for row in summary["rows"] if row["stage3_pareto_eligible"]
        ]
        pareto_ids = [
            row["experiment_id"]
            for row in eligible_rows
            if not any(
                dominates(other, row)
                for other in eligible_rows
                if other is not row
            )
        ]
        pareto_rows = [
            {
                **row,
                "pareto_optimal": row["experiment_id"] in pareto_ids,
                "deployment_candidate": (
                    row["experiment_id"] in pareto_ids
                    and bool(row.get("realtime_30fps_pass"))
                ),
            }
            for row in summary["rows"]
        ]
        deployment_candidate_ids = [
            row["experiment_id"]
            for row in pareto_rows
            if row["deployment_candidate"]
        ]
        pareto = {
            "created_at_utc": now(),
            "stage2_all_candidates_terminal": True,
            "pre_pareto_gate": (
                "valid measurement and B01-relative accuracy-retention gate"
            ),
            "objectives": {
                "maximize": list(PARETO_MAXIMIZE),
                "minimize": list(PARETO_MINIMIZE),
            },
            "secondary_reported_metrics": list(PARETO_SECONDARY),
            "deployment_constraint": defaults["selection"][
                "deployment_realtime"
            ],
            "pareto_candidate_ids": pareto_ids,
            "deployment_candidate_ids": deployment_candidate_ids,
            "rows": pareto_rows,
        }
        write_json(ROOT / PARETO_JSON, pareto)
        write_table(ROOT / PARETO_CSV, pareto_rows)
        print(f"Pareto: {pareto_ids}")
    elif args.only:
        print("Pareto skipped: --only is a debug subset, not the complete cohort.")

    report_ok, report_evidence = create_excel_report(logs / "excel-report.log")
    state["excel_report"] = {
        "status": "completed" if report_ok else "failed",
        "path": str(REPORT_XLSX),
        "evidence": report_evidence,
    }
    update_progress(
        state,
        completed_units=completed_progress_units,
        total_units=total_progress_units,
        current_candidate=None,
        current_stage="completed" if report_ok and not pending else "incomplete",
    )
    write_json(ROOT / STATE, state)
    print_progress(state)
    print(f"Stage 2 terminal: {len(eligible) - len(pending)}/{len(eligible)}; pending={pending}")
    if report_ok:
        print(f"Excel report: {REPORT_XLSX}")
    else:
        print(f"Excel report FAILED: {report_evidence}")
    return 0 if not pending and report_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
