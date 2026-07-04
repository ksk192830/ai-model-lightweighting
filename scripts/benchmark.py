#!/usr/bin/env python3
"""Benchmark model speed, latency, size, memory, and validation accuracy.

Example:
    python3 benchmark.py --model models/fp16.pt --data data.yaml --imgsz 640

Results are appended to results/results_raw.csv by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import subprocess
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import psutil
import torch

from evaluate import (
    BasePredictor,
    DatasetSample,
    EvaluationError,
    batched,
    create_predictor,
    evaluate_predictions,
    load_dataset_samples,
    synchronize_if_cuda,
)


CSV_FIELDS = [
    "Run Timestamp",
    "Model",
    "Model Path",
    "Backend",
    "Dataset",
    "Split",
    "Image Size",
    "Batch Size",
    "Confidence Threshold",
    "IoU Threshold",
    "Device",
    "Warmup Runs",
    "Timed Runs",
    "Repetitions",
    "Warmup Excluded",
    "Speed Samples",
    "FPS",
    "FPS Min",
    "FPS Max",
    "FPS Std",
    "Latency(ms)",
    "Latency Min(ms)",
    "Latency Max(ms)",
    "Latency Std(ms)",
    "Latency P50(ms)",
    "Latency P95(ms)",
    "Latency P99(ms)",
    "Precision",
    "Recall",
    "mAP50",
    "mAP50-95",
    "Metric Backend",
    "Evaluation Confidence Threshold",
    "Size(MB)",
    "Baseline Size(MB)",
    "Size Reduction(%)",
    "Memory(MB)",
    "GPU Memory(MB)",
    "GPU Reserved(MB)",
    "GPU Process Peak(MB)",
    "GPU Process Delta(MB)",
    "GPU Memory Source",
    "RAM Usage(MB)",
    "CPU RSS(MB)",
    "CPU Memory Delta(MB)",
    "Images",
    "Labels",
    "Predictions",
    "Python",
    "Torch",
    "CUDA Available",
    "CUDA Device",
    "OS",
    "Status",
    "Error",
]


@dataclass(frozen=True)
class BenchmarkConfig:
    data: Path
    imgsz: int
    batch_size: int
    conf: float
    iou: float
    device: str
    split: str
    backend: str
    warmup_runs: int
    timed_runs: int
    repetitions: int
    speed_samples: int
    max_images: int | None
    metric_backend: str
    eval_conf: float


class MemoryMonitor:
    """Sample process and system memory while inference is running."""

    def __init__(self, interval: float = 0.25) -> None:
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.start_rss = 0
        self.peak_rss = 0
        self.peak_system_used = 0
        self.start_gpu_process_mb = math.nan
        self.peak_gpu_process_mb = math.nan
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "MemoryMonitor":
        self.start_rss = self.process.memory_info().rss
        self.peak_rss = self.start_rss
        self.peak_system_used = psutil.virtual_memory().used
        self.start_gpu_process_mb = query_gpu_process_memory_mb(os.getpid())
        self.peak_gpu_process_mb = self.start_gpu_process_mb
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._record_sample()

    def _sample(self) -> None:
        while not self._stop_event.is_set():
            self._record_sample()
            self._stop_event.wait(self.interval)

    def _record_sample(self) -> None:
        try:
            self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
            self.peak_system_used = max(self.peak_system_used, psutil.virtual_memory().used)
        except psutil.Error:
            pass
        gpu_process_mb = query_gpu_process_memory_mb(os.getpid())
        if not math.isnan(gpu_process_mb):
            if math.isnan(self.peak_gpu_process_mb):
                self.peak_gpu_process_mb = gpu_process_mb
            else:
                self.peak_gpu_process_mb = max(self.peak_gpu_process_mb, gpu_process_mb)

    @property
    def peak_rss_mb(self) -> float:
        return bytes_to_mb(self.peak_rss)

    @property
    def rss_delta_mb(self) -> float:
        return bytes_to_mb(max(0, self.peak_rss - self.start_rss))

    @property
    def system_used_mb(self) -> float:
        return bytes_to_mb(self.peak_system_used)

    @property
    def gpu_process_delta_mb(self) -> float:
        if math.isnan(self.peak_gpu_process_mb) or math.isnan(self.start_gpu_process_mb):
            return math.nan
        return max(0.0, self.peak_gpu_process_mb - self.start_gpu_process_mb)


def benchmark_model(
    model_path: Path,
    samples: Sequence[DatasetSample],
    baseline_size_mb: float,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    model_size_mb = file_size_mb(model_path)
    row = base_row(model_path, model_size_mb, baseline_size_mb, config)
    predictor: BasePredictor | None = None

    try:
        reset_cuda_peak_memory(config.device)
        predictor = create_predictor(
            model_path=model_path,
            backend=config.backend,
            imgsz=config.imgsz,
            conf=config.conf,
            iou=config.iou,
            device=config.device,
            batch_size=config.batch_size,
        )
        row["Backend"] = predictor.backend_name
        class_mapping = predictor.align_classes(samples)
        if class_mapping:
            print(f"class mapping ({model_path.name}): {class_mapping}")

        with MemoryMonitor() as memory:
            speed_metrics = measure_fps(predictor, samples, config)
            latency_metrics = measure_latency(predictor, samples, config)
            accuracy_metrics = measure_accuracy(predictor, samples, config)

        gpu_memory_mb, gpu_reserved_mb = cuda_peak_memory(config.device)
        row.update(speed_metrics)
        row.update(latency_metrics)
        row.update(
            {
                "Precision": accuracy_metrics["precision"],
                "Recall": accuracy_metrics["recall"],
                "mAP50": accuracy_metrics["map50"],
                "mAP50-95": accuracy_metrics["map50_95"],
                "Images": int(accuracy_metrics["num_images"]),
                "Labels": int(accuracy_metrics["num_labels"]),
                "Predictions": int(accuracy_metrics["num_predictions"]),
                "Memory(MB)": (
                    memory.peak_gpu_process_mb
                    if not math.isnan(memory.peak_gpu_process_mb)
                    else max_not_nan(gpu_memory_mb, memory.peak_rss_mb)
                ),
                "GPU Memory(MB)": gpu_memory_mb,
                "GPU Reserved(MB)": gpu_reserved_mb,
                "GPU Process Peak(MB)": memory.peak_gpu_process_mb,
                "GPU Process Delta(MB)": memory.gpu_process_delta_mb,
                "GPU Memory Source": "nvidia-smi process" if not math.isnan(memory.peak_gpu_process_mb) else "torch-only",
                "RAM Usage(MB)": memory.system_used_mb,
                "CPU RSS(MB)": memory.peak_rss_mb,
                "CPU Memory Delta(MB)": memory.rss_delta_mb,
                "Status": "ok",
                "Error": "",
            }
        )
    except Exception as exc:
        row["Status"] = "failed"
        row["Error"] = f"{type(exc).__name__}: {exc}"
        row["Debug Traceback"] = traceback.format_exc()
    finally:
        if predictor is not None:
            predictor.close()
    return row


def measure_fps(
    predictor: BasePredictor,
    samples: Sequence[DatasetSample],
    config: BenchmarkConfig,
) -> dict[str, float | int | str]:
    speed_samples = select_speed_samples(samples, config.speed_samples)
    if not speed_samples:
        raise EvaluationError("No speed samples available.")

    for _ in range(config.warmup_runs):
        for batch in batched(speed_samples, config.batch_size):
            predictor.predict_batch([sample.image_path for sample in batch])
            synchronize_if_cuda()

    batch_fps_values: list[float] = []
    repetition_fps: list[float] = []
    for _ in range(config.repetitions):
        total_images = 0
        total_time = 0.0
        for _ in range(config.timed_runs):
            for batch in batched(speed_samples, config.batch_size):
                start = time.perf_counter()
                predictor.predict_batch([sample.image_path for sample in batch])
                synchronize_if_cuda()
                elapsed = time.perf_counter() - start
                if elapsed <= 0:
                    continue
                batch_size = len(batch)
                total_images += batch_size
                total_time += elapsed
                batch_fps_values.append(batch_size / elapsed)
        if total_time > 0:
            repetition_fps.append(total_images / total_time)

    fps = statistics.mean(repetition_fps) if repetition_fps else math.nan
    return {
        "Speed Samples": len(speed_samples),
        "FPS": fps,
        "FPS Min": min(batch_fps_values) if batch_fps_values else math.nan,
        "FPS Max": max(batch_fps_values) if batch_fps_values else math.nan,
        "FPS Std": statistics.pstdev(repetition_fps) if len(repetition_fps) > 1 else 0.0,
        "Warmup Excluded": "yes",
    }


def measure_latency(
    predictor: BasePredictor,
    samples: Sequence[DatasetSample],
    config: BenchmarkConfig,
) -> dict[str, float]:
    sample = samples[0]
    image_path = sample.image_path
    for _ in range(config.warmup_runs):
        predictor.predict_batch([image_path])
        synchronize_if_cuda()

    latencies_ms: list[float] = []
    for _ in range(config.repetitions):
        for _ in range(config.timed_runs):
            start = time.perf_counter()
            predictor.predict_batch([image_path])
            synchronize_if_cuda()
            latencies_ms.append((time.perf_counter() - start) * 1000.0)

    return {
        "Latency(ms)": statistics.mean(latencies_ms) if latencies_ms else math.nan,
        "Latency Min(ms)": min(latencies_ms) if latencies_ms else math.nan,
        "Latency Max(ms)": max(latencies_ms) if latencies_ms else math.nan,
        "Latency Std(ms)": statistics.pstdev(latencies_ms) if len(latencies_ms) > 1 else 0.0,
        "Latency P50(ms)": percentile(latencies_ms, 50),
        "Latency P95(ms)": percentile(latencies_ms, 95),
        "Latency P99(ms)": percentile(latencies_ms, 99),
    }


def measure_accuracy(
    predictor: BasePredictor,
    samples: Sequence[DatasetSample],
    config: BenchmarkConfig,
) -> dict[str, float]:
    original_conf = getattr(predictor, "conf", None)
    if original_conf is not None:
        predictor.conf = config.eval_conf
    try:
        predictions = []
        for batch in batched(samples, config.batch_size):
            image_paths = [sample.image_path for sample in batch]
            for per_image in predictor.predict_batch(image_paths):
                predictions.extend(per_image)
            synchronize_if_cuda()
        return evaluate_predictions(
            samples,
            predictions,
            metric_backend=config.metric_backend,
            operating_conf=config.conf,
        )
    finally:
        if original_conf is not None:
            predictor.conf = original_conf


def select_speed_samples(samples: Sequence[DatasetSample], requested: int) -> Sequence[DatasetSample]:
    if requested <= 0 or requested >= len(samples):
        return samples
    return samples[:requested]


def base_row(model_path: Path, model_size_mb: float, baseline_size_mb: float, config: BenchmarkConfig) -> dict[str, Any]:
    row = {field: "" for field in CSV_FIELDS}
    row.update(
        {
            "Run Timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "Model": model_path.stem,
            "Model Path": str(model_path),
            "Backend": config.backend,
            "Dataset": str(config.data),
            "Split": config.split,
            "Image Size": config.imgsz,
            "Batch Size": config.batch_size,
            "Confidence Threshold": config.conf,
            "IoU Threshold": config.iou,
            "Device": resolved_device_label(config.device),
            "Warmup Runs": config.warmup_runs,
            "Timed Runs": config.timed_runs,
            "Repetitions": config.repetitions,
            "Warmup Excluded": "yes",
            "Size(MB)": model_size_mb,
            "Baseline Size(MB)": baseline_size_mb,
            "Size Reduction(%)": size_reduction_percent(model_size_mb, baseline_size_mb),
            "Metric Backend": config.metric_backend,
            "Evaluation Confidence Threshold": config.eval_conf,
            "Python": platform.python_version(),
            "Torch": torch.__version__,
            "CUDA Available": torch.cuda.is_available(),
            "CUDA Device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
            "OS": platform.platform(),
            "Status": "not_run",
            "Error": "",
        }
    )
    return row


def write_results_csv(output_path: Path, rows: Sequence[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    if not write_header:
        with output_path.open("r", newline="", encoding="utf-8") as handle:
            existing_header = next(csv.reader(handle), [])
        if existing_header != CSV_FIELDS:
            raise EvaluationError(
                f"CSV schema differs from the current benchmark fields: {output_path}. "
                "Choose a new --output path or archive/remove the old CSV."
            )
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            cleaned = {field: format_csv_value(row.get(field, "")) for field in CSV_FIELDS}
            writer.writerow(cleaned)


def format_csv_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6f}"
    return value


def file_size_mb(path: Path) -> float:
    return bytes_to_mb(path.stat().st_size)


def bytes_to_mb(value: int | float) -> float:
    return float(value) / (1024.0 * 1024.0)


def size_reduction_percent(model_size_mb: float, baseline_size_mb: float) -> float:
    if baseline_size_mb <= 0:
        return math.nan
    return (1.0 - (model_size_mb / baseline_size_mb)) * 100.0


def max_not_nan(*values: float) -> float:
    valid = [value for value in values if not math.isnan(value)]
    return max(valid) if valid else math.nan


def percentile(values: Sequence[float], percentile_value: float) -> float:
    if not values:
        return math.nan
    return float(torch.tensor(values, dtype=torch.float64).quantile(percentile_value / 100.0).item())


def query_gpu_process_memory_mb(pid: int) -> float:
    """Return process GPU memory from NVIDIA's driver, including TensorRT allocations."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return math.nan
    total = 0.0
    found = False
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            process_pid = int(parts[0])
            memory_mb = float(parts[1])
        except ValueError:
            continue
        if process_pid == pid:
            total += memory_mb
            found = True
    return total if found else math.nan


def reset_cuda_peak_memory(device: str) -> None:
    if not torch.cuda.is_available() or resolved_device_label(device) == "cpu":
        return
    cuda_device = torch.device(resolved_device_label(device))
    torch.cuda.reset_peak_memory_stats(cuda_device)


def cuda_peak_memory(device: str) -> tuple[float, float]:
    if not torch.cuda.is_available() or resolved_device_label(device) == "cpu":
        return math.nan, math.nan
    cuda_device = torch.device(resolved_device_label(device))
    return (
        bytes_to_mb(torch.cuda.max_memory_allocated(cuda_device)),
        bytes_to_mb(torch.cuda.max_memory_reserved(cuda_device)),
    )


def resolved_device_label(device: str) -> str:
    if device in {"", "auto"}:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        return "cuda:0"
    if device.isdigit():
        return f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    return device


def collect_models(args: argparse.Namespace) -> list[Path]:
    model_values: list[str] = []
    if args.model:
        model_values.extend(args.model)
    if args.models:
        model_values.extend(args.models)
    models = [Path(value) for value in model_values]
    if not models:
        raise SystemExit("At least one --model path is required.")
    missing = [str(path) for path in models if not path.exists()]
    if missing:
        raise SystemExit(f"Model file(s) not found: {', '.join(missing)}")
    return models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark detection models and append raw metrics to CSV.")
    parser.add_argument("--model", action="append", help="Model path. Repeat to benchmark multiple models.")
    parser.add_argument("--models", nargs="+", help="Additional model paths.")
    parser.add_argument("--baseline-model", default="", help="Baseline model path used for size reduction. Defaults to first model.")
    parser.add_argument("--data", required=True, help="Dataset YAML, dataset directory, COCO JSON, or image list.")
    parser.add_argument("--imgsz", type=int, default=640, help="Fixed input image size.")
    parser.add_argument("--batch-size", type=int, default=1, help="Fixed batch size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Fixed confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="Fixed IoU/NMS threshold.")
    parser.add_argument("--device", default="auto", help="Fixed device: auto, cpu, cuda, cuda:0, or Ultralytics id.")
    parser.add_argument("--split", default="val", help="Dataset split: val, valid, or test.")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "ultralytics", "rfdetr", "rfdetr_engine"],
        help="Model backend.",
    )
    parser.add_argument("--warmup-runs", type=int, default=3, help="Warm-up runs excluded from speed/latency stats.")
    parser.add_argument("--timed-runs", type=int, default=30, help="Timed repetitions for FPS and latency stats.")
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Repeated timing blocks used for mean/std and latency percentiles.",
    )
    parser.add_argument("--speed-samples", type=int, default=32, help="Images used for FPS timing. 0 means all loaded images.")
    parser.add_argument("--max-images", type=int, default=0, help="Optional cap for validation accuracy. 0 means all images.")
    parser.add_argument(
        "--metric-backend",
        choices=["coco", "local"],
        default="coco",
        help="AP implementation. 'coco' uses official pycocotools COCOeval.",
    )
    parser.add_argument(
        "--eval-conf",
        type=float,
        default=0.001,
        help="Low confidence floor used only for AP evaluation; timing still uses --conf.",
    )
    parser.add_argument("--output", default="results/results_raw.csv", help="CSV output path.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first failed model instead of recording failure rows.")
    parser.add_argument("--print-json", action="store_true", help="Print result rows as JSON after writing CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = collect_models(args)
    baseline_path = Path(args.baseline_model) if args.baseline_model else models[0]
    if not baseline_path.exists():
        raise SystemExit(f"Baseline model file not found: {baseline_path}")

    config = BenchmarkConfig(
        data=Path(args.data),
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        split=args.split,
        backend=args.backend,
        warmup_runs=max(0, args.warmup_runs),
        timed_runs=max(1, args.timed_runs),
        repetitions=max(1, args.repetitions),
        speed_samples=max(0, args.speed_samples),
        max_images=args.max_images or None,
        metric_backend=args.metric_backend,
        eval_conf=max(0.0, min(1.0, args.eval_conf)),
    )

    samples = load_dataset_samples(config.data, split=config.split, max_images=config.max_images)
    baseline_size_mb = file_size_mb(baseline_path)

    rows = []
    for model_path in models:
        row = benchmark_model(model_path, samples, baseline_size_mb, config)
        rows.append(row)
        if args.fail_fast and row["Status"] != "ok":
            break

    output_path = Path(args.output)
    write_results_csv(output_path, rows)
    print(f"Wrote {len(rows)} row(s) to {output_path}")
    if args.print_json:
        printable = [{field: row.get(field, "") for field in CSV_FIELDS} for row in rows]
        print(json.dumps(printable, indent=2, ensure_ascii=False, default=str))

    if args.fail_fast and rows and rows[-1]["Status"] != "ok":
        raise SystemExit(str(rows[-1]["Error"]))


if __name__ == "__main__":
    main()
