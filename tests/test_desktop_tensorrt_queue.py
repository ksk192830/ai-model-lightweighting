from __future__ import annotations

import importlib.util
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = REPOSITORY_ROOT / "scripts/experiments/run_desktop_tensorrt_queue.py"
    spec = importlib.util.spec_from_file_location("run_desktop_tensorrt_queue", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def result(bbox: float, mask: float, miou: float) -> dict:
    return {
        "metrics": {
            "bbox": {"ap": bbox},
            "segm": {"ap": mask},
            "semantic_miou": miou,
        }
    }


def benchmark(median: float) -> dict:
    return {"median_ms": median}


def test_s02_requires_accuracy_and_latency_gates() -> None:
    module = load_module()
    selected = module.choose_structured(
        result(0.73, 0.60, 0.71),
        result(0.728, 0.598, 0.705),
        benchmark(20.0),
        benchmark(18.0),
    )
    assert selected["selected_experiment"] == "S02"


def test_s01_remains_when_s02_speed_gain_is_too_small() -> None:
    module = load_module()
    selected = module.choose_structured(
        result(0.73, 0.60, 0.71),
        result(0.73, 0.60, 0.71),
        benchmark(20.0),
        benchmark(19.5),
    )
    assert selected["selected_experiment"] == "S01"


def test_s01_remains_when_s02_accuracy_drop_is_too_large() -> None:
    module = load_module()
    selected = module.choose_structured(
        result(0.73, 0.60, 0.71),
        result(0.72, 0.59, 0.69),
        benchmark(20.0),
        benchmark(15.0),
    )
    assert selected["selected_experiment"] == "S01"
